# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Port kinds, payload containers and connect-time validation.

Every connection goes from an output port to an input port (architecture doc §7,
K0 spec §4.2). A port *kind* fixes the payload and the point in the step where it
is applied (§7.1):

========================  =====================================================
kind                      payload delivered to an input port per step
========================  =====================================================
:class:`Event`            ``(n,)`` sum over edges of ``weight × count``; applied
                          as a jump after the step's propagation
:class:`TimedEvent`       :class:`TimedEventPayload` (fixed capacity per target,
                          overflow counter); integrated between events
:class:`Current`          ``(n,)`` current, summed; latched for the next step
:class:`Conductance`      ``(n,)`` conductance; seen inside the ODE at every
                          substep, with ``E_rev`` and an optional gate ``gate(V)``
:class:`Graded`           ``(n,)`` continuous value with a declared unit; latched
========================  =====================================================

``gap`` and ``modulatory`` kinds are K1 and not declared here.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.params import Distribution
from brainpy_state._kernel.units import (
    as_unit,
    check_unit,
    from_mantissa,
    to_mantissa,
    unit_of,
)

__all__ = [
    'PortError',
    'PortKind',
    'Event',
    'TimedEvent',
    'Current',
    'Conductance',
    'Graded',
    'Ports',
    'TimedEventPayload',
    'check_kind_compatible',
    'expected_weight_unit',
    'check_weight_unit',
    'check_weight_sign',
    'check_capacity',
    'check_single_owner',
    'check_connection',
]


class PortError(ValueError):
    """Invalid port declaration or connection (kind, existence, sign, capacity)."""


# ---------------------------------------------------------------------------
# Port kinds
# ---------------------------------------------------------------------------

class PortKind:
    """Base class of port kinds.

    Attributes
    ----------
    kind : str
        Short kind name (``'event'``, ``'timed_event'``, ...).
    unit : brainunit.Unit
        Unit of the payload deposited into the port (dimensionless ``UNITLESS``
        when not declared).
    filtered : bool
        The owning neuron filters the synaptic kinetics of this port itself
        (e.g. ``g_ex`` in ``iaf_cond_exp``); a filtering ``SynapseModel`` must then
        not be connected to it (single owner of kinetics).
    """

    kind: ClassVar[str] = ''
    unit: u.Unit
    filtered: bool

    #: Weights into this port must be non-negative.
    nonnegative_weights: ClassVar[bool] = False

    def zeros(self, n: int, dtype: Any = np.float32) -> Any:
        """Empty payload for ``n`` targets (what a port receives with no input).

        Parameters
        ----------
        n : int
            Number of targets.
        dtype : dtype, default float32
            Float dtype of the payload.
        """
        return from_mantissa(jnp.zeros((int(n),), dtype), self.unit)


def _check_filtered(filtered: Any) -> bool:
    if not isinstance(filtered, (bool, np.bool_)):
        raise TypeError(f'filtered must be a bool, got {filtered!r}')
    return bool(filtered)


@dataclass(frozen=True, init=False)
class Event(PortKind):
    """Spike events on the time grid.

    Parameters
    ----------
    max_multiplicity : int or None, default 1
        Largest number of spikes one source can emit per step; ``None`` means
        unbounded (generators, ``aeif`` with several in-step spikes). Sources with
        ``max_multiplicity == 1`` may use the event-driven binary payload path;
        all others use the exact float (count) path.
    unit : brainunit.Unit, optional
        On an input port, the unit of the synaptic weight (e.g. ``u.pA`` for a PSC
        jump); ``None`` means dimensionless. Ignored on output ports.
    filtered : bool, default False
        See :class:`PortKind`.
    """

    kind: ClassVar[str] = 'event'
    max_multiplicity: int | None
    unit: u.Unit
    filtered: bool

    def __init__(self, max_multiplicity: int | None = 1, unit: Any = None,
                 filtered: bool = False) -> None:
        if max_multiplicity is not None:
            if isinstance(max_multiplicity, bool) or int(max_multiplicity) != max_multiplicity \
                    or max_multiplicity < 1:
                raise PortError(f'max_multiplicity must be an int >= 1 or None, got {max_multiplicity!r}')
            max_multiplicity = int(max_multiplicity)
        object.__setattr__(self, 'max_multiplicity', max_multiplicity)
        object.__setattr__(self, 'unit', as_unit(unit))
        object.__setattr__(self, 'filtered', _check_filtered(filtered))

    @property
    def binary(self) -> bool:
        """Whether at most one spike per source and step is emitted."""
        return self.max_multiplicity == 1


@dataclass(frozen=True, init=False)
class TimedEvent(PortKind):
    """Events with precise sub-step offsets (``*_ps`` models).

    Parameters
    ----------
    capacity : int
        Maximum number of events per target and step (``K``). Excess events are
        counted in :attr:`TimedEventPayload.overflow`, never dropped silently.
    unit : brainunit.Unit, optional
        Weight unit on input ports; ``None`` means dimensionless.
    filtered : bool, default False
        See :class:`PortKind`.
    """

    kind: ClassVar[str] = 'timed_event'
    capacity: int
    unit: u.Unit
    filtered: bool

    def __init__(self, capacity: int, unit: Any = None, filtered: bool = False) -> None:
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise PortError(f'capacity must be an int >= 1, got {capacity!r}')
        object.__setattr__(self, 'capacity', int(capacity))
        object.__setattr__(self, 'unit', as_unit(unit))
        object.__setattr__(self, 'filtered', _check_filtered(filtered))

    def zeros(self, n: int, dtype: Any = np.float32) -> 'TimedEventPayload':
        """Empty :class:`TimedEventPayload` of shape ``(n, capacity)``."""
        return TimedEventPayload.empty(n, self.capacity, dtype=dtype, weight_unit=self.unit)


@dataclass(frozen=True, init=False)
class Current(PortKind):
    """Summed current input, latched and acting during the next step.

    Parameters
    ----------
    unit : brainunit.Unit, default ``u.pA``
        Payload unit; must be a current.
    filtered : bool, default False
        See :class:`PortKind`.
    """

    kind: ClassVar[str] = 'current'
    unit: u.Unit
    filtered: bool

    def __init__(self, unit: Any = u.pA, filtered: bool = False) -> None:
        unit = as_unit(unit)
        check_unit(unit, u.pA, 'Current unit')
        object.__setattr__(self, 'unit', unit)
        object.__setattr__(self, 'filtered', _check_filtered(filtered))


@dataclass(frozen=True, init=False)
class Conductance(PortKind):
    """Conductance input entering the ODE at every substep.

    Parameters
    ----------
    unit : brainunit.Unit, default ``u.nS``
        Payload unit; must be a conductance.
    E_rev : brainunit.Quantity or str, optional
        Reversal potential (a voltage) or the name of the model parameter holding
        it; ``None`` when the model computes the driving force itself.
    gate : callable, optional
        ``gate(V) -> dimensionless`` evaluated inside ``derivatives`` at every
        substep (e.g. the NMDA Mg block).
    filtered : bool, default False
        See :class:`PortKind`.

    Notes
    -----
    Conductance weights must be non-negative (inhibition comes from ``E_rev``).
    """

    kind: ClassVar[str] = 'conductance'
    nonnegative_weights: ClassVar[bool] = True
    unit: u.Unit
    E_rev: Any
    gate: Callable[[Any], Any] | None
    filtered: bool

    def __init__(self, unit: Any = u.nS, E_rev: Any = None,
                 gate: Callable[[Any], Any] | None = None, filtered: bool = False) -> None:
        unit = as_unit(unit)
        check_unit(unit, u.nS, 'Conductance unit')
        if E_rev is not None and not isinstance(E_rev, str):
            check_unit(E_rev, u.mV, 'Conductance E_rev')
        if gate is not None and not callable(gate):
            raise TypeError(f'gate must be callable or None, got {type(gate).__name__}')
        object.__setattr__(self, 'unit', unit)
        object.__setattr__(self, 'E_rev', E_rev)
        object.__setattr__(self, 'gate', gate)
        object.__setattr__(self, 'filtered', _check_filtered(filtered))


@dataclass(frozen=True, init=False)
class Graded(PortKind):
    """Continuous value with a declared unit, latched for the next step.

    Parameters
    ----------
    unit : brainunit.Unit or None
        Payload unit (``None`` = dimensionless).
    filtered : bool, default False
        See :class:`PortKind`.
    """

    kind: ClassVar[str] = 'graded'
    unit: u.Unit
    filtered: bool

    def __init__(self, unit: Any, filtered: bool = False) -> None:
        object.__setattr__(self, 'unit', as_unit(unit))
        object.__setattr__(self, 'filtered', _check_filtered(filtered))


_EVENT_KINDS = (Event, TimedEvent)


class Ports(Mapping):
    """Named collection of port kinds of one model (inputs or outputs).

    Parameters
    ----------
    **ports : PortKind
        Port name to kind.

    Raises
    ------
    TypeError
        If a value is not a :class:`PortKind`.
    """

    def __init__(self, **ports: PortKind) -> None:
        for name, kind in ports.items():
            if not isinstance(kind, PortKind):
                raise TypeError(f'port {name!r} must be a PortKind, got {type(kind).__name__}')
        self._ports = dict(ports)

    def __getitem__(self, name: str) -> PortKind:
        return self._ports[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ports)

    def __len__(self) -> int:
        return len(self._ports)

    def __repr__(self) -> str:
        inner = ', '.join(f'{k}={type(v).__name__}' for k, v in self._ports.items())
        return f'Ports({inner})'

    def require(self, name: str, where: str = '') -> PortKind:
        """Return port ``name`` or raise :class:`PortError` listing the available ports."""
        try:
            return self._ports[name]
        except KeyError:
            raise PortError(
                f'{_prefix(where)}port {name!r} does not exist; available: {list(self._ports)}'
            ) from None

    def zeros(self, n: int, dtype: Any = np.float32) -> dict[str, Any]:
        """Empty payload for every port (``{name: kind.zeros(n, dtype)}``)."""
        return {k: v.zeros(n, dtype) for k, v in self._ports.items()}


# ---------------------------------------------------------------------------
# TimedEvent payload
# ---------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TimedEventPayload:
    """Fixed-capacity per-target list of precisely timed events for one step.

    Parameters
    ----------
    offsets : brainunit.Quantity, shape (n, K)
        Event time measured from the **start** of the step, in ``[0, dt]``.
    weights : brainunit.Quantity or jax.Array, shape (n, K)
        Event weights (in the receiving port's unit).
    valid : jax.Array of bool, shape (n, K)
        Which slots hold an event.
    overflow : jax.Array of int32, shape (n,)
        Number of events that did not fit (cumulative over merges).

    Notes
    -----
    Payloads built by :meth:`empty`, :meth:`pack` and :meth:`merge` keep valid
    events first, sorted by increasing offset; invalid slots have offset ``+inf``.
    """

    offsets: Any
    weights: Any
    valid: Any
    overflow: Any

    @property
    def capacity(self) -> int:
        """Number of slots ``K`` per target."""
        return int(self.valid.shape[-1])

    @property
    def count(self) -> jax.Array:
        """Number of valid events per target, shape ``(n,)``."""
        return jnp.sum(self.valid, axis=-1, dtype=jnp.int32)

    @classmethod
    def empty(cls, n: int, capacity: int, dtype: Any = np.float32,
              weight_unit: Any = None) -> 'TimedEventPayload':
        """Payload with no events.

        Parameters
        ----------
        n : int
            Number of targets.
        capacity : int
            Slots per target.
        dtype : dtype, default float32
            Float dtype of offsets and weights.
        weight_unit : brainunit.Unit, optional
            Unit of the weights.
        """
        shape = (int(n), int(capacity))
        return cls(
            offsets=from_mantissa(jnp.full(shape, jnp.inf, dtype), u.ms),
            weights=from_mantissa(jnp.zeros(shape, dtype), as_unit(weight_unit)),
            valid=jnp.zeros(shape, bool),
            overflow=jnp.zeros((int(n),), jnp.int32),
        )

    @classmethod
    def pack(cls, offsets: Any, weights: Any, valid: Any, capacity: int,
             overflow: Any = None) -> 'TimedEventPayload':
        """Sort ``(n, M)`` candidate events and keep the earliest ``capacity``.

        Parameters
        ----------
        offsets : brainunit.Quantity, shape (n, M)
            Event offsets from the step start (time dimension).
        weights : brainunit.Quantity or jax.Array, shape (n, M)
            Event weights.
        valid : array_like of bool, shape (n, M)
            Which candidates are events.
        capacity : int
            Slots kept per target.
        overflow : array_like of int, shape (n,), optional
            Previous overflow count to add to.

        Returns
        -------
        TimedEventPayload
            Valid events first in increasing offset; events beyond ``capacity``
            (the latest ones) are counted in ``overflow``.
        """
        check_unit(offsets, u.ms, 'TimedEvent offsets')
        valid = jnp.asarray(valid, bool)
        unit_w = unit_of(weights)
        off_m = to_mantissa(offsets, u.ms)
        w_m = to_mantissa(weights, unit_w)
        n, m = valid.shape
        capacity = int(capacity)
        key = jnp.where(valid, off_m, jnp.inf)
        order = jnp.argsort(key, axis=-1, stable=True)
        take = lambda a: jnp.take_along_axis(a, order, axis=-1)
        key, w_m, valid = take(key), take(w_m), take(valid)
        if m < capacity:
            pad = ((0, 0), (0, capacity - m))
            key = jnp.pad(key, pad, constant_values=jnp.inf)
            w_m = jnp.pad(w_m, pad)
            valid = jnp.pad(valid, pad)
        dropped = jnp.sum(valid[:, capacity:], axis=-1, dtype=jnp.int32)
        prev = jnp.zeros((n,), jnp.int32) if overflow is None else jnp.asarray(overflow, jnp.int32)
        valid_k = valid[:, :capacity]
        return cls(
            offsets=from_mantissa(key[:, :capacity], u.ms),
            weights=from_mantissa(jnp.where(valid_k, w_m[:, :capacity], 0), unit_w),
            valid=valid_k,
            overflow=prev + dropped,
        )

    def merge(self, other: 'TimedEventPayload') -> 'TimedEventPayload':
        """Combine with ``other`` (same targets) keeping this payload's capacity.

        Overflow counters add, plus any events that no longer fit.

        Raises
        ------
        UnitMismatchError
            If the weight units have different dimensions.
        """
        unit_w = unit_of(self.weights)
        check_unit(other.weights, unit_w, 'merged TimedEvent weights')
        offsets = from_mantissa(jnp.concatenate(
            [to_mantissa(self.offsets, u.ms), to_mantissa(other.offsets, u.ms)], axis=-1), u.ms)
        weights = from_mantissa(jnp.concatenate(
            [to_mantissa(self.weights, unit_w), to_mantissa(other.weights, unit_w)], axis=-1), unit_w)
        valid = jnp.concatenate([self.valid, other.valid], axis=-1)
        return TimedEventPayload.pack(offsets, weights, valid, self.capacity,
                                      overflow=self.overflow + other.overflow)


# ---------------------------------------------------------------------------
# Connect-time validation
# ---------------------------------------------------------------------------

def _prefix(where: str) -> str:
    return f'{where}: ' if where else ''


#: source kind -> destination kinds it may connect to without a synapse model
_COMPATIBLE: dict[type, tuple[type, ...]] = {
    Event: (Event, TimedEvent),
    TimedEvent: (TimedEvent, Event),
    Current: (Current,),
    Conductance: (Conductance,),
    Graded: (Graded, Current, Conductance),
}


def check_kind_compatible(src: PortKind, dst: PortKind, where: str = '') -> None:
    """Raise :class:`PortError` unless ``src`` may feed ``dst``.

    Allowed: event -> event / timed_event; timed_event -> timed_event / event (the
    offset is dropped, as in NEST); current -> current; conductance -> conductance;
    graded -> graded / current / conductance. Event sources reach ``current`` or
    ``conductance`` ports through a ``SynapseModel``, whose *output* kind is then
    checked as ``src`` -- or directly when the port is ``filtered``: the neuron owns
    the kinetics and applies each event as a jump of the filtered state (NEST
    ``iaf_cond_exp``: ``g_ex += w``; added in K0 T5).

    Parameters
    ----------
    src, dst : PortKind
        Output and input port kinds.
    where : str
        Context for the error message (e.g. ``'E -> I.ex'``).
    """
    allowed = _COMPATIBLE.get(type(src))
    if isinstance(src, _EVENT_KINDS) and isinstance(dst, (Current, Conductance)) and dst.filtered:
        return
    if allowed is None or not isinstance(dst, allowed):
        ok = ', '.join(k.kind for k in (allowed or ()))
        raise PortError(
            f'{_prefix(where)}a {src.kind or type(src).__name__!r} output cannot feed a '
            f'{dst.kind or type(dst).__name__!r} input (allowed targets: {ok or "none"})'
        )


def expected_weight_unit(src: PortKind, dst: PortKind) -> u.Unit:
    """Unit a weight must have on a ``src -> dst`` connection.

    For event sources the weight carries the destination unit; for value sources
    (current, conductance, graded) it converts ``src.unit`` into ``dst.unit``.
    """
    if isinstance(src, _EVENT_KINDS):
        return dst.unit
    return dst.unit / src.unit


def _weight_unit(weight: Any) -> u.Unit:
    return weight.unit if isinstance(weight, Distribution) else unit_of(weight)


def check_weight_unit(weight: Any, src: PortKind, dst: PortKind, where: str = '') -> None:
    """Raise :class:`~brainpy_state._kernel.units.UnitMismatchError` on a weight-dimension mismatch.

    Parameters
    ----------
    weight : scalar, array, Quantity or Distribution
        Connection weight specification.
    src, dst : PortKind
        Output and input port kinds.
    where : str
        Context for the error message.
    """
    expected = expected_weight_unit(src, dst)
    check_unit(_weight_unit(weight), expected, f'{_prefix(where)}weight')


def check_weight_sign(weight: Any, dst: PortKind, where: str = '') -> None:
    """Raise :class:`PortError` if ``dst`` needs non-negative weights and ``weight`` can be negative.

    Distributions are checked through their (clipped) support, concrete values
    numerically. Traced values cannot be inspected and are not checked.

    Parameters
    ----------
    weight : scalar, array, Quantity or Distribution
        Connection weight specification.
    dst : PortKind
        Input port kind.
    where : str
        Context for the error message.
    """
    if not dst.nonnegative_weights:
        return
    if isinstance(weight, Distribution):
        lo, _ = weight.bounds()
        if lo < 0:
            raise PortError(
                f'{_prefix(where)}{dst.kind} weights must be non-negative but the weight '
                f'distribution can be negative (lower bound {lo}); clip it with low=0'
            )
        return
    try:
        m = np.asarray(u.get_mantissa(weight))
    except jax.errors.TracerArrayConversionError:
        return
    if m.size and np.nanmin(m) < 0:
        raise PortError(
            f'{_prefix(where)}{dst.kind} weights must be non-negative, got minimum {np.nanmin(m)}'
        )


def check_capacity(dst: PortKind, max_events: int, strict: bool = True, where: str = '') -> bool:
    """Check that a ``TimedEvent`` port can hold the worst-case events per step.

    Parameters
    ----------
    dst : PortKind
        Input port kind; non-``TimedEvent`` kinds always pass.
    max_events : int
        Worst-case events per target and step (e.g. in-degree × multiplicity).
    strict : bool, default True
        Raise when it may overflow; otherwise return False (runtime overflow is
        then counted, never dropped silently).
    where : str
        Context for the error message.

    Returns
    -------
    bool
        True when the capacity suffices.
    """
    if not isinstance(dst, TimedEvent) or max_events <= dst.capacity:
        return True
    if strict:
        raise PortError(
            f'{_prefix(where)}up to {max_events} events per target and step can arrive but '
            f'the timed_event port capacity is {dst.capacity}; raise capacity or accept '
            f'counted overflow (strict=False)'
        )
    return False


def check_single_owner(dst: PortKind, synapse_filters: bool, where: str = '') -> None:
    """Raise :class:`PortError` if a filtering synapse targets a neuron-filtered port.

    Parameters
    ----------
    dst : PortKind
        Input port kind.
    synapse_filters : bool
        Whether the projection's ``SynapseModel`` has its own kinetics.
    where : str
        Context for the error message.
    """
    if synapse_filters and dst.filtered:
        raise PortError(
            f'{_prefix(where)}the target port already filters its synaptic kinetics; '
            f'connecting a filtering synapse model would apply them twice'
        )


def check_connection(src: PortKind, dst: PortKind, weight: Any = None, *,
                     synapse_filters: bool = False, max_events: int | None = None,
                     strict_capacity: bool = True, where: str = '') -> bool:
    """Run all connect-time checks for one ``src -> dst`` projection.

    Parameters
    ----------
    src, dst : PortKind
        Output kind (or the synapse model's output kind) and input kind.
    weight : optional
        Weight specification; unit and sign are checked when given.
    synapse_filters : bool, default False
        See :func:`check_single_owner`.
    max_events : int, optional
        See :func:`check_capacity`.
    strict_capacity : bool, default True
        See :func:`check_capacity`.
    where : str
        Context for error messages.

    Returns
    -------
    bool
        False if a ``TimedEvent`` capacity may overflow (only with
        ``strict_capacity=False``), else True.
    """
    check_kind_compatible(src, dst, where)
    check_single_owner(dst, synapse_filters, where)
    if weight is not None:
        check_weight_unit(weight, src, dst, where)
        check_weight_sign(weight, dst, where)
    if max_events is not None:
        return check_capacity(dst, max_events, strict_capacity, where)
    return True

