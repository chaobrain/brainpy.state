# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``volume_transmitter`` — JAX-native dopamine broadcast node.

Rebuilt as a :class:`brainstate.nn.Module` device that maintains the dopamine
concentration ``n(t)`` as a broadcast :class:`brainstate.HiddenState`, computed
once per step and read identically by every edge of a dopamine-modulated
projection (the cluster-08 ``signal_reads`` seam of
:class:`~brainpy_state._network._event_plastic.VoltageCoupledPlasticProj`).

The previous imperative ring-buffer port (NEST delivery stamps, spike-history
lists, host scalars) is retired: with the online per-step weight integral, the
synapse never needs the buffered spike history — only the running ``n``.

**NEST divergence (documented).** In NEST the ``volume_transmitter`` only *buffers
and relays* dopaminergic spikes; each ``stdp_dopamine_synapse`` integrates its own
``n`` from the relayed train (``stdp_dopamine_synapse.h:419-425``). Here ``n`` is
moved onto the transmitter and broadcast — valid because ``n``'s dynamics depend
only on ``tau_n`` and the dopa train (a common property shared by every synapse on
the transmitter), not on any per-synapse state. The transmitter's ``tau_n`` must
match the dopamine synapse spec's ``tau_n`` (the parity drive sets both to the one
NEST value).
"""
from __future__ import annotations

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u
from brainstate.typing import ArrayLike, Size

from ._base import NESTDevice
from ._plastic_base import to_ms, to_scalar_int

__all__ = [
    'volume_transmitter',
]


class volume_transmitter(NESTDevice):
    r"""Dopamine broadcast node (NEST ``volume_transmitter``), JAX-native rebuild.

    Collects dopaminergic spikes from one or more bound sources and maintains the
    neuromodulator concentration ``n(t)`` exposed to every dopamine-modulated
    synapse as a single broadcast scalar. Each step advances ``n`` by NEST's
    ``update_dopamine_`` recursion (``stdp_dopamine_synapse.h:419-425``):

    .. math::

       n \leftarrow n \, e^{-\Delta t / \tau_n} \;+\; \frac{c}{\tau_n},

    where :math:`c` is the number of dopaminergic spikes delivered this step
    (summed over all bound sources) and :math:`\tau_n` the concentration time
    constant. The increment carries **no** :math:`\Delta t` factor (it is
    ``multiplicity / tau_n`` per spike, exactly as in NEST); the decay alone
    carries the step.

    Parameters
    ----------
    in_size : Size, optional
        Number of independent broadcast channels; ``n`` has shape ``(in_size,)``.
        Independent transmitters are normally separate nodes, so the default
        ``1`` is the common case. Default ``1``.
    tau_n : Quantity or float, optional
        Dopamine concentration time constant (> 0). Bare numbers are interpreted
        as milliseconds. **Must equal the dopamine synapse spec's** ``tau_n``
        (see module note). Default ``200.0 ms``.
    deliver_interval : int, optional
        NEST trigger period (in units of ``min_delay``). Accepted for NEST-API
        parity and validated ``>= 1``, but a **no-op** in the online integration
        scheme (the weight integral runs every step, so there is no batched
        delivery). Default ``1``.
    name : str, optional
        Optional node name. Default ``None``.

    Attributes
    ----------
    n : brainstate.HiddenState
        The broadcast dopamine concentration, shape ``(in_size,)``, init ``0.0``.
    tau_n : saiunit.Quantity
        The concentration time constant in ms.
    deliver_interval : int
        The accepted (no-op) NEST delivery period.

    See Also
    --------
    brainpy_state._nest.stdp_dopamine_synapse.stdp_dopamine_synapse : reads ``n``.

    Notes
    -----
    A dopa source is bound with :meth:`bind_dopa` (the :class:`Simulator` wires
    this from ``connect(dopa_pool, vt)``): each binding is a ``(reader, local_idx)``
    pair where ``reader()`` returns the source population's per-step spike vector
    and ``local_idx`` selects the dopaminergic entries. The per-step count is
    ``sum_sources sum(reader()[local_idx])``. The bound reader carries the
    substrate's intrinsic one-step lag (a dopa neuron firing at step ``j`` is read
    at step ``j+1``), matching NEST's ``+1`` delivery stamp
    (``volume_transmitter.cpp:113``).

    References
    ----------
    .. [1] NEST ``models/volume_transmitter.{h,cpp}`` and
       ``models/stdp_dopamine_synapse.h`` (``update_dopamine_``).

    Examples
    --------
    Advance ``n`` one step after a single dopaminergic spike
    (:math:`n = 1/\tau_n = 0.005` for ``tau_n = 200 ms``):

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> import saiunit as u
        >>> from brainpy.state import volume_transmitter
        >>> with brainstate.environ.context(dt=1.0 * u.ms):
        ...     vt = volume_transmitter(1, tau_n=200.0 * u.ms)
        ...     holder = brainstate.ShortTermState(jnp.zeros(1))
        ...     vt.bind_dopa(lambda: holder.value, jnp.array([0]))
        ...     _ = brainstate.nn.init_all_states(vt)
        ...     holder.value = jnp.asarray([1.0])
        ...     with brainstate.environ.context(t=0.0 * u.ms, i=0):
        ...         _ = vt.update()
        ...     round(float(vt.n.value[0]), 4)
        0.005
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        tau_n: ArrayLike = 200.0 * u.ms,
        deliver_interval: int = 1,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self._tau_n_ms = to_ms(tau_n, name='tau_n')
        if self._tau_n_ms <= 0.0:
            raise ValueError("'tau_n' must be > 0.")
        self.tau_n = self._tau_n_ms * u.ms

        self.deliver_interval = to_scalar_int(deliver_interval, name='deliver_interval')
        if self.deliver_interval < 1:
            raise ValueError("'deliver_interval' must be >= 1.")

        # bound dopa sources: list of (reader: () -> (n,) spikes, local_idx)
        self._dopa_sources: list[tuple] = []

    # -- state -------------------------------------------------------------
    def init_state(self, batch_size: int = None, **kwargs):
        r"""Allocate the broadcast concentration ``n`` (shape ``(in_size,)``, init 0)."""
        self.n = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset the broadcast concentration ``n`` to ``0.0``."""
        self.n.value = braintools.init.param(
            braintools.init.Constant(0.0), self.varshape, batch_size
        )

    # -- dopa binding ------------------------------------------------------
    def bind_dopa(self, reader, local_idx: ArrayLike):
        r"""Register a dopaminergic spike source read each step.

        Parameters
        ----------
        reader : callable
            Zero-argument callable returning the source population's per-step
            spike vector (the Simulator passes a holder reader).
        local_idx : ArrayLike
            Indices into ``reader()`` selecting this transmitter's dopaminergic
            entries; their spikes are summed into the per-step count.
        """
        self._dopa_sources.append((reader, jnp.asarray(local_idx)))

    # -- recursion ---------------------------------------------------------
    @staticmethod
    def _advance(n, count, dt_ms, tau_n_ms):
        r"""NEST ``update_dopamine_`` step: ``n*exp(-dt/tau_n) + count/tau_n``."""
        return n * jnp.exp(-dt_ms / tau_n_ms) + count / tau_n_ms

    def update(self):
        r"""Advance ``n`` by one step from the bound dopa spike count.

        Returns
        -------
        jax.Array
            The updated broadcast concentration ``n`` (shape ``(in_size,)``).
        """
        dt_ms = u.Quantity(brainstate.environ.get_dt()).to_decimal(u.ms)
        count = 0.0
        for reader, idx in self._dopa_sources:
            count = count + jnp.sum(jnp.asarray(reader())[idx])
        self.n.value = self._advance(self.n.value, count, dt_ms, self._tau_n_ms)
        return self.n.value
