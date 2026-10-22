# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Synapse models: kinetics between edge delivery and the port deposit (K0 spec §4.2, T6).

A :class:`SynapseModel` is attached to a projection with
``net.connect(..., synapse=Expon(tau=5 * u.ms))``. Each step the projection's
delivered event amplitudes (``sum_e weight_e * count_e``, in the port unit) enter the
synapse state, and the synapse output is deposited into an **unfiltered**
``Current`` / ``Conductance`` input port of the target. The neuron then treats it
like any value input (a latched ``Current`` port -- ``latch=True``, NEST models --
stores it for the next step, an unlatched one uses it in the same step, a
``Conductance`` port sees it inside the ODE at every substep).

Single owner of kinetics
------------------------
A port declared ``filtered=True`` already filters its synaptic kinetics inside the
neuron (NEST style, e.g. ``iaf_cond_exp.ex``); attaching a filtering synapse to it is
a :class:`~brainpy_state._kernel.ports.PortError` (the kinetics would be applied
twice).

Merge pass
----------
K0 synapses are **post-aligned** (state per target neuron). The compiler groups the
synapse states per ``(target population, port, merge_key)``
(:func:`plan_synapse_groups`): linear synapses with equal parameters onto the same port
share one state fed by the sum of their deliveries (the AlignPost ``O(N_post)``
result). ``merge_key() is None`` (non-linear kinetics, or ``merge=False``) keeps a
state per projection. The plan reports the merged groups (``Plan.describe()``).
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.ports import (
    Conductance,
    Current,
    Event,
    PortError,
    PortKind,
    check_kind_compatible,
    check_single_owner,
    check_weight_sign,
    check_weight_unit,
)
from brainpy_state._kernel.units import check_unit, to_mantissa

__all__ = [
    'SynapseModel',
    'Expon',
    'SynapseGroup',
    'plan_synapse_groups',
    'check_synapse_connection',
]


class SynapseModel:
    """Contract of a synapse model attached to a projection.

    Subclasses declare the class attributes and implement :meth:`init_state`,
    :meth:`step` and :meth:`merge_key`. All methods are pure (state in, state out)
    and work on unit-free mantissas in the target port's unit.

    Attributes
    ----------
    alignment : {'post'}
        Where the state lives. K0 implements post-aligned synapses only (one value
        per target neuron); ``'pre'`` and ``'edge'`` are K1.
    linear : bool
        Whether the kinetics are linear in the input (only linear synapses with
        equal parameters may be merged).
    filters : bool
        Whether the synapse has its own kinetics (single-owner check against
        :attr:`PortKind.filtered <brainpy_state._kernel.ports.PortKind>`).
    targets : tuple of type
        Port kinds the synapse output may be deposited into.
    """

    alignment: ClassVar[str] = 'post'
    linear: ClassVar[bool] = True
    filters: ClassVar[bool] = True
    targets: ClassVar[tuple] = (Current, Conductance)

    def output_kind(self, dst: PortKind) -> PortKind:
        """Kind of the value the synapse deposits into ``dst`` (same kind and unit)."""
        if isinstance(dst, Conductance):
            return Conductance(dst.unit)
        if isinstance(dst, Current):
            return Current(dst.unit)
        raise PortError(f'{type(self).__name__} deposits a current or a conductance; it cannot '
                        f'feed a {dst.kind!r} port')

    def init_state(self, n: int, dtype: Any) -> dict:  # pragma: no cover - abstract
        """Initial state ``{name: (n,) mantissa}``."""
        raise NotImplementedError

    def step(self, state: dict, x: jax.Array, ctx: Any) -> tuple[dict, jax.Array]:  # pragma: no cover
        """Advance one step.

        Parameters
        ----------
        state : dict
            Current state (mantissas, ``(n,)``).
        x : jax.Array
            ``(n,)`` summed edge input of this step (port unit mantissa).
        ctx : StepContext
            Step context (``ctx.dt``).

        Returns
        -------
        state : dict
            New state.
        out : jax.Array
            ``(n,)`` value deposited into the port (port unit mantissa).
        """
        raise NotImplementedError

    def merge_key(self) -> Hashable | None:  # pragma: no cover - abstract
        """Hashable identity of the kinetics, or ``None`` if never mergeable."""
        raise NotImplementedError

    def describe(self) -> str:
        """Short label for ``Plan.describe()``."""
        return type(self).__name__


def _positive_ms(value: Any, name: str) -> float:
    check_unit(value, u.ms, name)
    m = np.asarray(u.Quantity(value).to_decimal(u.ms), dtype=np.float64)
    if m.ndim != 0:
        raise ValueError(f'{name} must be a scalar (per-neuron synapse parameters are K1), '
                         f'got shape {m.shape}')
    if not np.isfinite(m) or m <= 0:
        raise ValueError(f'{name} must be positive and finite, got {value}')
    return float(m)


class Expon(SynapseModel):
    r"""Exponentially decaying post-aligned synapse (BrainPy ``Expon``).

    .. math::

       g_n = g_{n-1}\, e^{-\Delta t/\tau} + x_n, \qquad \text{output } g_n,

    where :math:`x_n` is the summed event amplitude delivered in step :math:`n`
    (weights in the port unit). This is the order of the old
    ``brainpy_state/_brainpy/exponential.py`` ``Expon.update`` (exponential-Euler
    decay, exact for this linear ODE, then the delta input).

    Parameters
    ----------
    tau : brainunit.Quantity
        Decay time constant (scalar, > 0).
    merge : bool, default True
        Allow the compiler to merge this synapse with equal ones onto the same port.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from brainpy_state._kernel import StepContext
        >>> from brainpy_state._kernel.synapse import Expon
        >>> syn = Expon(tau=10.0 * u.ms)
        >>> ctx = StepContext.create(0.0 * u.ms, 0.1 * u.ms)
        >>> s = syn.init_state(2, jnp.float32)
        >>> s, g = syn.step(s, jnp.asarray([1.0, 0.0]), ctx)
        >>> g.tolist()
        [1.0, 0.0]
        >>> s, g = syn.step(s, jnp.zeros(2), ctx)
        >>> round(float(g[0]), 6)
        0.99005
        >>> Expon(tau=10.0 * u.ms).merge_key() == syn.merge_key()
        True
    """

    linear: ClassVar[bool] = True

    def __init__(self, tau: Any = 8.0 * u.ms, merge: bool = True) -> None:
        self.tau_ms = _positive_ms(tau, 'tau')
        self.tau = self.tau_ms * u.ms
        if not isinstance(merge, (bool, np.bool_)):
            raise TypeError(f'merge must be a bool, got {merge!r}')
        self.merge = bool(merge)

    def __repr__(self) -> str:
        return f'Expon(tau={self.tau_ms:g} ms)'

    def describe(self) -> str:
        return repr(self)

    def init_state(self, n: int, dtype: Any) -> dict:
        return {'g': jnp.zeros((int(n),), dtype)}

    def step(self, state: dict, x: jax.Array, ctx: Any) -> tuple[dict, jax.Array]:
        g = state['g']
        dt = to_mantissa(ctx.dt, u.ms)
        decay = jnp.exp(-jnp.asarray(dt, g.dtype) / jnp.asarray(self.tau_ms, g.dtype))
        g = g * decay + jnp.asarray(x, g.dtype)
        return {'g': g}, g

    def merge_key(self) -> Hashable | None:
        if not self.merge:
            return None
        return ('Expon', self.tau_ms)


def check_synapse_connection(synapse: Any, src: PortKind, dst: PortKind, weight: Any,
                             where: str = '') -> None:
    """Connect-time checks of a projection with a synapse model.

    The synapse must receive events (K0) and deposit into an unfiltered port of a kind
    it supports; the weight carries the port unit and respects its sign rule.

    Parameters
    ----------
    synapse : SynapseModel
        The synapse model.
    src : PortKind
        Source output kind.
    dst : PortKind
        Target input kind.
    weight : optional
        Weight specification (checked for unit and sign).
    where : str
        Context for error messages.

    Raises
    ------
    TypeError
        If ``synapse`` is not a :class:`SynapseModel`.
    PortError
        On a non-event source, an unsupported or filtered target port.
    """
    pre = f'{where}: ' if where else ''
    if not isinstance(synapse, SynapseModel):
        raise TypeError(f'synapse must be a SynapseModel, got {type(synapse).__name__}')
    if synapse.alignment != 'post':  # pragma: no cover - only post-aligned exist in K0
        raise NotImplementedError(f'{synapse.alignment}-aligned synapses are K1')
    if not isinstance(src, Event):
        raise PortError(f'{pre}{type(synapse).__name__} receives spike events; the source port is '
                        f'a {src.kind!r} port')
    check_single_owner(dst, synapse.filters, where)
    if not isinstance(dst, synapse.targets):
        raise PortError(f'{pre}{type(synapse).__name__} cannot feed a {dst.kind!r} port '
                        f'(allowed: {[k.kind for k in synapse.targets]})')
    check_kind_compatible(synapse.output_kind(dst), dst, where)
    if weight is not None:
        # events carry the weight in the port unit (amplitude of the synaptic jump)
        check_weight_unit(weight, src, dst, where)
        check_weight_sign(weight, dst, where)


@dataclass
class SynapseGroup:
    """One synapse state of a compiled plan (possibly shared by merged projections).

    Attributes
    ----------
    key : str
        Name of the state in the runner's state pytree.
    pop : int
        Target population index.
    port : str
        Target input port.
    synapse : SynapseModel
        Kinetics (the first member's model; members are equal by ``merge_key``).
    members : list of int
        Projection indices feeding this state.
    """

    key: str
    pop: int
    port: str
    synapse: SynapseModel
    members: list = field(default_factory=list)

    @property
    def merged(self) -> bool:
        """Whether several projections share this state."""
        return len(self.members) > 1


def plan_synapse_groups(targets: Sequence[tuple], merge: bool = True) -> tuple[list, dict]:
    """The merge pass: group synapse states per ``(population, port, merge_key)``.

    Parameters
    ----------
    targets : sequence of (int, int, str, SynapseModel, str)
        ``(projection index, target population index, port, synapse, label)`` for
        every (projection, target population) pair with a synapse, in projection
        order. ``label`` names the state when it is not merged.
    merge : bool, default True
        Enable merging (``False``: one state per pair).

    Returns
    -------
    groups : list of SynapseGroup
        In first-appearance order.
    route : dict
        ``(projection index, population index) -> group index``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.synapse import Expon, plan_synapse_groups
        >>> t = [(0, 1, 'I', Expon(5 * u.ms), 'p0'), (1, 1, 'I', Expon(5 * u.ms), 'p1'),
        ...      (2, 1, 'I', Expon(2 * u.ms), 'p2')]
        >>> groups, route = plan_synapse_groups(t)
        >>> [g.members for g in groups], route[(1, 1)]
        ([[0, 1], [2]], 0)
        >>> [g.members for g in plan_synapse_groups(t, merge=False)[0]]
        [[0], [1], [2]]
    """
    groups: list[SynapseGroup] = []
    route: dict = {}
    index: dict = {}
    for j, pop, port, syn, label in targets:
        mk = syn.merge_key() if merge else None
        k = None if mk is None else (pop, port, mk)
        if k is not None and k in index:
            gi = index[k]
            groups[gi].members.append(j)
        else:
            gi = len(groups)
            groups.append(SynapseGroup(key=label, pop=pop, port=port, synapse=syn, members=[j]))
            if k is not None:
                index[k] = gi
        route[(j, pop)] = gi
    for g in groups:
        if g.merged:
            g.key = f'merged[{g.port}@pop{g.pop}:' + ','.join(str(m) for m in g.members) + ']'
    return groups, route
