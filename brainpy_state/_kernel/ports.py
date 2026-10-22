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
:class:`Current`          ``(n,)`` current, summed; ``latch=True`` (default):
                          acts during the next step (NEST ``CurrentEvent``),
                          ``latch=False``: acts in the step it arrives in
:class:`Conductance`      ``(n,)`` conductance; seen inside the ODE at every
                          substep, with ``E_rev`` and an optional gate ``gate(V)``
:class:`Graded`           ``(n,)`` continuous value with a declared unit;
                          ``latch`` as for :class:`Current`
========================  =====================================================

The latch flag (decision D-a) is a property of an **input** port: the receiving
port decides when a value acts. On output ports (devices, synapse outputs) it has
no effect. A zero delay into an unlatched port is same-step (topological order); a
latched port adds one step by its port semantics (architecture doc §7 L2, D4a).

Receptor ports (decision D-b)
-----------------------------
Synaptic kinetics have exactly one owner per port, with one encoding per owner:

* **neuron-owned**: an :class:`Event` (or :class:`TimedEvent`) input port that
  names the state it filters into, ``ex=Event(unit=u.nS, into='g_ex')``. The
  weight is measured in that state's unit; the neuron applies each delivered
  weight as the jump of its own kinetics after the propagation (the default
  ``NeuronModel.on_input`` adds it to the named state) and its ``derivatives``
  filter it. Projections into such a port carry no ``SynapseModel``.
* **projection-owned**: a ``SynapseModel`` on the projection (``Expon``, ...)
  deposits into a plain :class:`Current` or :class:`Conductance` port.

``into`` is a property of an input port; on output ports it has no effect (as
``latch``).

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
    'is_latched',
    'is_receptor',
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
    into : str or None
        On a receptor port (:class:`Event` / :class:`TimedEvent` with ``into=``):
        the state the neuron filters the input into; ``None`` for every other port.
    """

    kind: ClassVar[str] = ''
    unit: u.Unit
    into: str | None = None

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


def _check_flag(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be a bool, got {value!r}')
    return bool(value)


def _check_into(into: Any) -> str | None:
    if into is None:
        return None
    if not isinstance(into, str) or not into.isidentifier():
        raise PortError(f'into must be a state name (an identifier) or None, got {into!r}')
    return into


def _is_conductance_unit(unit: u.Unit) -> bool:
    return u.get_dim(unit) == u.get_dim(u.nS)


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
    into : str, optional
        On an input port: the name of the state the neuron filters this input
        into, making it a neuron-owned **receptor port** (decision D-b). The
        state must be declared by the model with the dimension of ``unit``
        (checked at class creation). Projections into a receptor port carry no
        ``SynapseModel``; a receptor port with a conductance unit takes
        non-negative weights only. Ignored on output ports.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.ports import Event, is_receptor
        >>> ex = Event(unit=u.nS, into='g_ex')
        >>> ex.into, is_receptor(ex), ex.nonnegative_weights
        ('g_ex', True, True)
        >>> is_receptor(Event(unit=u.mV))
        False
    """

    kind: ClassVar[str] = 'event'
    max_multiplicity: int | None
    unit: u.Unit
    into: str | None

    def __init__(self, max_multiplicity: int | None = 1, unit: Any = None, *,
                 into: str | None = None) -> None:
        if max_multiplicity is not None:
            if isinstance(max_multiplicity, bool) or int(max_multiplicity) != max_multiplicity \
                    or max_multiplicity < 1:
                raise PortError(f'max_multiplicity must be an int >= 1 or None, got {max_multiplicity!r}')
            max_multiplicity = int(max_multiplicity)
        object.__setattr__(self, 'max_multiplicity', max_multiplicity)
        object.__setattr__(self, 'unit', as_unit(unit))
        object.__setattr__(self, 'into', _check_into(into))

    @property
    def nonnegative_weights(self) -> bool:  # type: ignore[override]  # frozen: never assigned
        """Weights must be non-negative: a receptor port filtering into a conductance."""
        return self.into is not None and _is_conductance_unit(self.unit)

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
    into : str, optional
        As for :class:`Event`: on an input port, the state the neuron filters the
        timed events into (a neuron-owned receptor port, decision D-b).
    """

    kind: ClassVar[str] = 'timed_event'
    capacity: int
    unit: u.Unit
    into: str | None

    def __init__(self, capacity: int, unit: Any = None, *, into: str | None = None) -> None:
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise PortError(f'capacity must be an int >= 1, got {capacity!r}')
        object.__setattr__(self, 'capacity', int(capacity))
        object.__setattr__(self, 'unit', as_unit(unit))
        object.__setattr__(self, 'into', _check_into(into))

    @property
    def nonnegative_weights(self) -> bool:  # type: ignore[override]  # frozen: never assigned
        """Weights must be non-negative: a receptor port filtering into a conductance."""
        return self.into is not None and _is_conductance_unit(self.unit)

    def zeros(self, n: int, dtype: Any = np.float32) -> 'TimedEventPayload':
        """Empty :class:`TimedEventPayload` of shape ``(n, capacity)``."""
        return TimedEventPayload.empty(n, self.capacity, dtype=dtype, weight_unit=self.unit)


@dataclass(frozen=True, init=False)
class Current(PortKind):
    """Summed current input; latched (next step) or same-step (decision D-a).

    Parameters
    ----------
    unit : brainunit.Unit, default ``u.pA``
        Payload unit; must be a current.
    latch : bool, default True
        On an input port: ``True`` -- the value delivered in step ``n`` acts during
        step ``n + 1`` (NEST ``CurrentEvent`` / ring-buffer semantics; the model
        gets a ``latch_<port>`` state); ``False`` -- it acts during step ``n``
        itself (BrainPy-family models; a zero-delay acyclic path then propagates
        within one step). Ignored on output ports.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.ports import Current
        >>> Current(u.pA).latch
        True
        >>> Current(u.mA, latch=False).latch
        False
    """

    kind: ClassVar[str] = 'current'
    unit: u.Unit
    latch: bool

    def __init__(self, unit: Any = u.pA, *, latch: bool = True) -> None:
        unit = as_unit(unit)
        check_unit(unit, u.pA, 'Current unit')
        object.__setattr__(self, 'unit', unit)
        object.__setattr__(self, 'latch', _check_flag(latch, 'latch'))


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

    Notes
    -----
    Conductance weights must be non-negative (inhibition comes from ``E_rev``).
    The payload is the conductance itself: the kinetics belong to the source or
    to a projection's ``SynapseModel``. A neuron that filters its own
    conductance declares a receptor port instead,
    ``Event(unit=u.nS, into='g_ex')`` (decision D-b).
    """

    kind: ClassVar[str] = 'conductance'
    nonnegative_weights: ClassVar[bool] = True
    unit: u.Unit
    E_rev: Any
    gate: Callable[[Any], Any] | None

    def __init__(self, unit: Any = u.nS, E_rev: Any = None,
                 gate: Callable[[Any], Any] | None = None) -> None:
        unit = as_unit(unit)
        check_unit(unit, u.nS, 'Conductance unit')
        if E_rev is not None and not isinstance(E_rev, str):
            check_unit(E_rev, u.mV, 'Conductance E_rev')
        if gate is not None and not callable(gate):
            raise TypeError(f'gate must be callable or None, got {type(gate).__name__}')
        object.__setattr__(self, 'unit', unit)
        object.__setattr__(self, 'E_rev', E_rev)
        object.__setattr__(self, 'gate', gate)


@dataclass(frozen=True, init=False)
class Graded(PortKind):
    """Continuous value with a declared unit; latched (next step) or same-step.

    Parameters
    ----------
    unit : brainunit.Unit or None
        Payload unit (``None`` = dimensionless).
    latch : bool, default True
        As for :class:`Current`: on an input port, ``True`` acts during the next
        step, ``False`` in the step the value arrives in. Ignored on output ports.
    """

    kind: ClassVar[str] = 'graded'
    unit: u.Unit
    latch: bool

    def __init__(self, unit: Any, *, latch: bool = True) -> None:
        object.__setattr__(self, 'unit', as_unit(unit))
        object.__setattr__(self, 'latch', _check_flag(latch, 'latch'))


def is_latched(kind: PortKind) -> bool:
    """Whether input port ``kind`` acts one step after delivery (a latched current/graded port).

    Parameters
    ----------
    kind : PortKind
        Input port kind.

    Returns
    -------
    bool
        ``True`` for :class:`Current` / :class:`Graded` ports declared
        ``latch=True``; ``False`` for every other kind and for ``latch=False``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.ports import Current, Event, is_latched
        >>> is_latched(Current(u.pA)), is_latched(Current(u.pA, latch=False)), is_latched(Event())
        (True, False, False)
    """
    return isinstance(kind, (Current, Graded)) and kind.latch


def is_receptor(kind: PortKind) -> bool:
    """Whether input port ``kind`` is a neuron-owned receptor port (decision D-b).

    Parameters
    ----------
    kind : PortKind
        Input port kind.

    Returns
    -------
    bool
        ``True`` for :class:`Event` / :class:`TimedEvent` ports declared with
        ``into=<state>``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.ports import Conductance, Event, is_receptor
        >>> is_receptor(Event(unit=u.nS, into='g_ex')), is_receptor(Conductance(u.nS))
        (True, False)
    """
    return isinstance(kind, (Event, TimedEvent)) and kind.into is not None


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
        def tag(v: PortKind) -> str:
            if isinstance(v, (Current, Graded)):
                return f'(latch={v.latch})'
            if is_receptor(v):
                return f'(into={v.into!r})'
            return ''

        inner = ', '.join(f'{k}={type(v).__name__}{tag(v)}' for k, v in self._ports.items())
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
_COMPATIBLE: dict[type[PortKind], tuple[type[PortKind], ...]] = {
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
    ``conductance`` ports only through a ``SynapseModel``, whose *output* kind is
    then checked as ``src``; a neuron that owns the kinetics declares a receptor
    ``event`` port instead (``Event(unit=u.nS, into='g_ex')``, decision D-b).

    Parameters
    ----------
    src, dst : PortKind
        Output and input port kinds.
    where : str
        Context for the error message (e.g. ``'E -> I.ex'``).
    """
    allowed = _COMPATIBLE.get(type(src))
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
    """Raise :class:`PortError` if a synapse model targets a neuron-owned receptor port.

    Parameters
    ----------
    dst : PortKind
        Input port kind.
    synapse_filters : bool
        Whether the projection carries a ``SynapseModel`` (kinetics of its own).
    where : str
        Context for the error message (``'<source> -> <Model>.<port>'`` names the
        neuron model that owns the kinetics).
    """
    if synapse_filters and is_receptor(dst):
        raise PortError(
            f'{_prefix(where)}the target is a neuron-owned receptor port: the neuron model '
            f'owns its kinetics (it filters the input into state {dst.into!r}), so a '
            f'projection into it carries no SynapseModel (decision D-b); connect without '
            f'synapse=, or target a plain Current/Conductance port for projection-owned kinetics'
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

