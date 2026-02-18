# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-


import math
from collections import defaultdict

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike

from ._base import NESTSynapse

__all__ = [
    'static_synapse',
]

_UNSET = object()
_SPIKE_EVENT = 'spike'
_CURRENT_EVENT_TYPES = {'rate', 'current', 'conductance'}
_PASS_THROUGH_EVENT_TYPES = {'double_data', 'data_logging'}
_ALL_EVENT_TYPES = {_SPIKE_EVENT, *_CURRENT_EVENT_TYPES, *_PASS_THROUGH_EVENT_TYPES}


class static_synapse(NESTSynapse):
    r"""NEST-compatible static (non-plastic) synapse connection model.

    ``static_synapse`` implements a fixed-weight, fixed-delay synaptic connection
    that transmits events from presynaptic to postsynaptic neurons without any
    plasticity. This is the simplest and most commonly used synapse model in NEST,
    serving as the baseline for all connection types.

    The model maintains three immutable parameters (unless explicitly changed via
    :meth:`set`): synaptic weight, transmission delay, and receptor port. When a
    presynaptic event occurs, the payload is scaled by ``weight`` and scheduled
    for delivery to the postsynaptic neuron after ``delay`` milliseconds at the
    specified ``receptor_type`` port.

    **1. Mathematical Model**

    The static synapse performs no dynamics computation. Event transmission is
    purely a scheduling and scaling operation:

    .. math::

       \text{output}(t + d) = w \cdot \text{input}(t)

    where:

    - :math:`w` is the synaptic weight (dimensionless or with units depending on receiver)
    - :math:`d` is the transmission delay (ms)
    - :math:`\text{input}(t)` is the presynaptic event multiplicity at time :math:`t`
    - :math:`\text{output}(t + d)` is the postsynaptic input delivered at :math:`t + d`

    No state variables evolve over time. The weight and delay remain constant until
    explicitly modified.

    **2. Event Transmission Semantics**

    This implementation replicates the NEST ``static_synapse`` event processing
    pipeline from ``models/static_synapse.h``:

    1. **Weight scaling**: Multiply incoming multiplicity by ``weight``
    2. **Delay computation**: Convert delay from milliseconds to simulation steps
    3. **Receiver selection**: Identify the target postsynaptic neuron
    4. **Receptor port assignment**: Route to the specified ``receptor_type``
    5. **Scheduled delivery**: Enqueue the event for future delivery

    The event is stored in an internal queue and delivered when the simulation
    clock reaches the target time step.

    **3. Delay Discretization**

    NEST represents delays as integer multiples of the simulation time step :math:`dt`.
    The conversion from continuous time to discrete steps follows the NEST ``ld_round``
    convention (round-to-nearest, ties round up):

    .. math::

       d_{\text{steps}} = \left\lfloor \frac{d_{\text{ms}}}{dt_{\text{ms}}} + 0.5 \right\rfloor

    **Constraints:**

    - :math:`d_{\text{steps}} \geq 1` (minimum one-step delay)
    - :math:`d_{\text{ms}} > 0` (delay must be strictly positive)

    The effective delivered delay is quantized:

    .. math::

       d_{\text{effective}} = d_{\text{steps}} \cdot dt_{\text{ms}}

    **Example:** With :math:`dt = 0.1\,\text{ms}`:

    - Requested delay 1.44 ms → :math:`\lfloor 1.44/0.1 + 0.5 \rfloor = 14` steps → 1.4 ms effective
    - Requested delay 1.45 ms → :math:`\lfloor 1.45/0.1 + 0.5 \rfloor = 15` steps → 1.5 ms effective

    **4. Event Type Routing**

    The synapse supports multiple event types corresponding to NEST's event class
    hierarchy. Event delivery is routed to the appropriate receiver input method:

    **Event type → Receiver method mapping:**

    ==================  ==========================================  ============================
    Event Type          Receiver Method                             Typical Use Case
    ==================  ==========================================  ============================
    ``'spike'``         ``add_delta_input(key, value, label)``      Binary spike transmission
    ``'rate'``          ``add_current_input(key, value, label)``    Rate-coded signals
    ``'current'``       ``add_current_input(key, value, label)``    Direct current injection
    ``'conductance'``   ``add_current_input(key, value, label)``    Conductance-based input
    ``'double_data'``   ``add_current_input`` (fallback)            Arbitrary data transmission
    ``'data_logging'``  ``add_current_input`` (fallback)            Logging/monitoring signals
    ==================  ==========================================  ============================

    If the receiver implements ``handle_static_synapse_event(value, receptor_type, event_type)``,
    that callback takes precedence over the standard routing.

    **5. Receptor Port Mechanism**

    The ``receptor_type`` parameter allows a single postsynaptic neuron to distinguish
    between different input sources (e.g., excitatory vs. inhibitory, AMPA vs. NMDA).
    This is implemented through labeled input accumulation:

    - Receptor 0 → label ``"receptor_0"``
    - Receptor 1 → label ``"receptor_1"``
    - Receptor :math:`n` → label ``"receptor_n"``

    The postsynaptic neuron's ``current_inputs`` and ``delta_inputs`` dictionaries
    store accumulated input per receptor label.

    Parameters
    ----------
    weight : float, array-like, or Quantity, optional
        Fixed synaptic weight. Scalar value, dimensionless or with units.
        Units depend on receiver requirements (e.g., pA for current-based,
        nS for conductance-based, mV for voltage-based).
        Default: ``1.0`` (dimensionless).
    delay : float, array-like, or Quantity, optional
        Synaptic transmission delay. Must be a positive scalar with time units
        (recommended: ``brainunit.ms``). Will be discretized to integer time
        steps according to simulation resolution ``dt``.
        Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receptor port identifier on the postsynaptic neuron. Non-negative integer
        specifying which input channel receives the event. Different receptor types
        can implement different synaptic kinetics or reversal potentials.
        Default: ``0`` (primary receptor port).
    post : Dynamics, optional
        Default postsynaptic receiver object. If provided, :meth:`send` and
        :meth:`update` will target this receiver unless overridden. Must implement
        either ``add_delta_input`` or ``add_current_input`` methods.
        Default: ``None`` (must provide receiver explicitly in method calls).
    event_type : str, optional
        Type of event to transmit. Determines delivery method and receiver handling.
        Must be one of: ``'spike'``, ``'rate'``, ``'current'``, ``'conductance'``,
        ``'double_data'``, ``'data_logging'``.
        Default: ``'spike'`` (binary spike events).
    name : str, optional
        Unique identifier for this synapse instance.
        Default: auto-generated.

    Parameter Mapping

    NEST ``static_synapse`` parameters map to this implementation as follows:

    ==================  ====================  ========================================
    NEST Parameter      brainpy.state Param   Notes
    ==================  ====================  ========================================
    ``weight``          ``weight``            Scalar, units depend on receiver
    ``delay``           ``delay``             Converted to ms, discretized to steps
    ``receptor_type``   ``receptor_type``     Integer ≥ 0
    (connection target) ``post``              Explicit receiver object
    (event class)       ``event_type``        String identifier for event routing
    ==================  ====================  ========================================

    Attributes
    ----------
    weight : float or Quantity
        Current synaptic weight (read/write via :meth:`set`).
    delay : float
        Effective transmission delay in milliseconds (quantized to time steps).
    receptor_type : int
        Current receptor port identifier.
    post : Dynamics or None
        Default postsynaptic receiver.
    event_type : str
        Current event transmission type.

    Notes
    -----
    **Design differences from NEST:**

    1. **Single-connection scope**: This class represents one synaptic connection.
       NEST's ``static_synapse`` is a template applied to many connections. For
       large-scale networks, use vectorized projection classes.

    2. **Event queue**: The internal ``_queue`` is a simple defaultdict. For
       production use with many synapses, consider a more efficient global event
       delivery system.

    3. **Weight units**: NEST is unit-agnostic at the connection level. This
       implementation supports ``brainunit`` quantities, allowing type-safe
       dimensional analysis.

    4. **Delay caching**: The delay is recomputed whenever ``dt`` changes. NEST
       performs this conversion once during connection setup.

    **Typical usage patterns:**

    - **Static networks**: Define fixed connectivity with heterogeneous weights/delays
    - **Baseline benchmarks**: Compare against plastic synapse models
    - **Event generators**: Connect spike generators to network populations
    - **Hybrid architectures**: Interface between different neuron types

    **Performance considerations:**

    - Lightweight: No state updates, minimal computation per event
    - Memory overhead: Queue storage scales with number of in-flight events
    - Thread safety: Not thread-safe; use separate instances per thread

    See Also
    --------
    static_synapse_hom_w : Homogeneous weight variant (all connections share one weight)
    tsodyks_synapse : Short-term plasticity extension
    stdp_synapse : Spike-timing dependent plasticity extension

    References
    ----------
    .. [1] Diesmann, M., Gewaltig, M.-O., & Aertsen, A. (1999). Stable propagation
           of synchronous spiking in cortical neural networks. *Nature*, 402(6761),
           529-533. https://doi.org/10.1038/990101
    .. [2] NEST Initiative (2025). NEST Synapse Specification.
           https://nest-simulator.readthedocs.io/en/stable/synapses/synapse_specification.html
    .. [3] NEST source code: ``models/static_synapse.h``, ``models/static_synapse.cpp``
           https://github.com/nest/nest-simulator

    Examples
    --------
    **Basic usage with explicit receiver:**

    .. code-block:: python

        >>> import brainpy.state as bs
        >>> import brainunit as u
        >>> import brainstate
        >>> with brainstate.environ.context(dt=0.1 * u.ms):
        ...     # Create postsynaptic neuron
        ...     post_neuron = bs.LIF(1, V_rest=-65*u.mV, V_th=-50*u.mV, tau=20*u.ms)
        ...
        ...     # Create static synapse
        ...     syn = bs.static_synapse(
        ...         weight=0.5*u.nS,
        ...         delay=1.5*u.ms,
        ...         receptor_type=0,
        ...         post=post_neuron,
        ...     )
        ...
        ...     # Send spike event
        ...     syn.send(multiplicity=1.0)
        ...
        ...     # Inspect parameters
        ...     params = syn.get()
        ...     print(f"Weight: {params['weight']}")
        ...     print(f"Effective delay: {params['delay']} ms")
        ...     print(f"Delay steps: {params['delay_steps']}")

    **Simulation loop with update:**

    .. code-block:: python

        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     post = bs.LIF(1, V_rest=-65*u.mV, V_th=-50*u.mV, tau=20*u.ms)
        ...     syn = bs.static_synapse(weight=1.0, delay=1.0*u.ms, post=post)
        ...
        ...     # Initialize states
        ...     post.init_all_states()
        ...     syn.init_all_states()
        ...
        ...     # Simulate 10 steps
        ...     for step in range(10):
        ...         # Deliver queued events and schedule new ones
        ...         delivered = syn.update(pre_spike=1.0 if step == 3 else 0.0)
        ...         # Update postsynaptic neuron
        ...         post.update()

    **Multi-receptor configuration:**

    .. code-block:: python

        >>> # Excitatory and inhibitory inputs to same neuron
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     target = bs.LIF(1, V_rest=-65*u.mV, V_th=-50*u.mV, tau=20*u.ms)
        ...
        ...     # Excitatory synapse on receptor 0
        ...     exc_syn = bs.static_synapse(
        ...         weight=0.8*u.nS,
        ...         delay=1.0*u.ms,
        ...         receptor_type=0,
        ...         post=target,
        ...     )
        ...
        ...     # Inhibitory synapse on receptor 1
        ...     inh_syn = bs.static_synapse(
        ...         weight=-0.4*u.nS,
        ...         delay=1.2*u.ms,
        ...         receptor_type=1,
        ...         post=target,
        ...     )
        ...
        ...     # Both synapses can deliver to same neuron concurrently
        ...     exc_syn.send(multiplicity=1.0)
        ...     inh_syn.send(multiplicity=1.0)

    **Dynamic parameter modification:**

    .. code-block:: python

        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     syn = bs.static_synapse(weight=1.0, delay=1.0*u.ms)
        ...
        ...     # Change weight during simulation
        ...     syn.set(weight=2.0)
        ...
        ...     # Change multiple parameters at once
        ...     syn.set(weight=0.5, delay=2.0*u.ms, receptor_type=1)
        ...
        ...     # Verify changes
        ...     params = syn.get()
        ...     assert params['weight'] == 0.5
        ...     assert params['receptor_type'] == 1

    **Non-spike event types:**

    .. code-block:: python

        >>> # Rate-coded input
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     rate_syn = bs.static_synapse(
        ...         weight=0.1,
        ...         delay=1.0*u.ms,
        ...         event_type='rate',
        ...         post=target,
        ...     )
        ...
        ...     # Send continuous rate value
        ...     rate_syn.send(multiplicity=42.5)  # 42.5 Hz
        ...
        ...     # Current injection
        ...     current_syn = bs.static_synapse(
        ...         weight=100.0*u.pA,
        ...         delay=0.5*u.ms,
        ...         event_type='current',
        ...         post=target,
        ...     )
        ...     current_syn.send(multiplicity=1.0)
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        post=None,
        event_type: str = _SPIKE_EVENT,
        name: str | None = None,
    ):
        super().__init__(in_size=1, name=name)

        self.weight = self._normalize_scalar_weight(weight)
        self._delay_requested_ms = self._to_scalar_time_ms(delay, name='delay')
        self.delay = float(self._delay_requested_ms)
        self.receptor_type = self._to_receptor_type(receptor_type)
        self.post = post
        self.event_type = self._normalize_event_type(event_type)

        self._validate_delay(self._delay_requested_ms)

        self._delay_steps = 1
        self._dt_cache_ms = np.nan
        self._queue = defaultdict(list)
        self._delivery_counter = 0

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_delay_cache(dt_ms)

    @staticmethod
    def _to_scalar_time_ms(value: ArrayLike, *, name: str) -> float:
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.ms), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_receptor_type(value: ArrayLike) -> int:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('receptor_type must be scalar.')
        receptor = float(arr.reshape(()))
        if not float(receptor).is_integer():
            raise ValueError('receptor_type must be an integer.')
        receptor_int = int(receptor)
        if receptor_int < 0:
            raise ValueError('receptor_type must be non-negative.')
        return receptor_int

    @staticmethod
    def _normalize_scalar_weight(weight: ArrayLike):
        if isinstance(weight, u.Quantity):
            unit = u.get_unit(weight)
            arr = np.asarray(weight.to_decimal(unit), dtype=np.float64)
            if arr.size != 1:
                raise ValueError('weight must be scalar.')
            return float(arr.reshape(())) * unit
        arr = np.asarray(u.math.asarray(weight, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('weight must be scalar.')
        scalar = float(arr.reshape(()))
        return scalar

    @staticmethod
    def _normalize_event_type(event_type: str) -> str:
        if not isinstance(event_type, str):
            raise ValueError('event_type must be a string.')
        ev = event_type.strip().lower()
        if ev not in _ALL_EVENT_TYPES:
            raise ValueError(
                f'Unsupported event_type "{event_type}". '
                f'Expected one of {sorted(_ALL_EVENT_TYPES)}.'
            )
        return ev

    @staticmethod
    def _validate_delay(delay_ms: float):
        if not np.isfinite(delay_ms):
            raise ValueError('delay must be finite.')
        if delay_ms <= 0.0:
            raise ValueError('delay must be strictly positive.')

    @staticmethod
    def _ld_round(x: float) -> int:
        # NEST ld_round: round-to-nearest, midpoint-up.
        return int(math.floor(float(x) + 0.5))

    @staticmethod
    def _delay_ms_to_steps(delay_ms: float, dt_ms: float) -> int:
        return static_synapse._ld_round(delay_ms / dt_ms)

    @staticmethod
    def _weight_to_float(weight) -> float:
        if isinstance(weight, u.Quantity):
            unit = u.get_unit(weight)
            return float(np.asarray(weight.to_decimal(unit), dtype=np.float64).reshape(()))
        return float(np.asarray(u.math.asarray(weight), dtype=np.float64).reshape(()))

    @staticmethod
    def _is_nonzero(value) -> bool:
        arr = np.asarray(u.math.asarray(value), dtype=np.float64)
        return bool(np.any(arr != 0.0))

    def _maybe_dt_ms(self) -> float | None:
        dt = brainstate.environ.get('dt', default=None)
        if dt is None:
            return None
        return self._to_scalar_time_ms(dt, name='dt')

    def _dt_ms(self) -> float:
        dt = brainstate.environ.get('dt', default=None)
        if dt is None:
            raise ValueError(
                'Simulation resolution `dt` must be defined in brainstate.environ '
                'before using static_synapse.update().'
            )
        return self._to_scalar_time_ms(dt, name='dt')

    def _current_time_ms(self) -> float:
        t = brainstate.environ.get('t', default=0.0 * u.ms)
        if t is None:
            return 0.0
        return self._to_scalar_time_ms(t, name='t')

    def _refresh_delay_cache(self, dt_ms: float):
        if dt_ms <= 0.0:
            raise ValueError('Simulation resolution must be strictly positive.')

        steps = self._delay_ms_to_steps(self._delay_requested_ms, dt_ms)
        if steps < 1:
            raise ValueError('Delay must be greater than or equal to resolution.')

        self._delay_steps = int(steps)
        self.delay = float(self._delay_steps * dt_ms)
        self._dt_cache_ms = float(dt_ms)

    def _refresh_delay_if_needed(self):
        dt_ms = self._dt_ms()
        if (
            (not np.isfinite(self._dt_cache_ms))
            or (not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15))
        ):
            self._refresh_delay_cache(dt_ms)
        return dt_ms

    def _curr_step(self, dt_ms: float) -> int:
        return self._ld_round(self._current_time_ms() / dt_ms)

    @staticmethod
    def _receiver_label(receptor_type: int) -> str:
        return f'receptor_{int(receptor_type)}'

    def _resolve_receiver(self, post):
        receiver = self.post if post is None else post
        if receiver is None:
            raise ValueError(
                'No receiver is configured. Provide `post` in the constructor or '
                'pass `post=...` when calling send()/update().'
            )
        return receiver

    def _deliver_event(self, receiver, value, receptor_type: int, event_type: str):
        if hasattr(receiver, 'handle_static_synapse_event'):
            receiver.handle_static_synapse_event(value, receptor_type, event_type)
            return

        key = f'{self.name}_event_{self._delivery_counter}'
        self._delivery_counter += 1
        label = self._receiver_label(receptor_type)

        if event_type == _SPIKE_EVENT:
            if not hasattr(receiver, 'add_delta_input'):
                raise TypeError(
                    'Receiver does not support spike delivery: '
                    'missing add_delta_input(...) method.'
                )
            receiver.add_delta_input(key, value, label=label)
            return

        if event_type in _CURRENT_EVENT_TYPES:
            if not hasattr(receiver, 'add_current_input'):
                raise TypeError(
                    f'Receiver does not support {event_type} delivery: '
                    'missing add_current_input(...) method.'
                )
            receiver.add_current_input(key, value, label=label)
            return

        # Best-effort fallback for data-like events.
        if event_type in _PASS_THROUGH_EVENT_TYPES:
            if hasattr(receiver, 'add_current_input'):
                receiver.add_current_input(key, value, label=label)
                return
            if hasattr(receiver, 'add_delta_input'):
                receiver.add_delta_input(key, value, label=label)
                return
            raise TypeError(
                f'Receiver does not support {event_type} delivery: '
                'missing add_current_input(...) and add_delta_input(...).'
            )

        raise ValueError(f'Unsupported event_type "{event_type}".')

    def _deliver_due_events(self, step: int) -> int:
        queued = self._queue.pop(int(step), None)
        if queued is None:
            return 0
        for receiver, value, receptor_type, event_type in queued:
            self._deliver_event(receiver, value, receptor_type, event_type)
        return len(queued)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize or reset synapse state (event queue).

        Clears the internal event queue and resets the delivery counter. Should be
        called before starting a new simulation or when resetting network state.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for state initialization (ignored; static synapses are scalar).
        **kwargs
            Additional initialization arguments (ignored).

        Notes
        -----
        - The event queue is a per-synapse structure mapping future time steps to
          lists of pending events: ``{step: [(receiver, value, receptor, type), ...]}``.
        - Delivery counter generates unique input keys for the receiver's input dicts.
        - Unlike stateful synapses (e.g., with facilitation/depression), static synapses
          have no evolving state variables—only the transient event queue.
        """
        del batch_size, kwargs
        self._queue = defaultdict(list)
        self._delivery_counter = 0

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        post: object = _UNSET,
        event_type: str | object = _UNSET,
    ):
        r"""Update synapse parameters dynamically (NEST ``SetStatus`` equivalent).

        Modifies one or more synapse parameters while preserving others. Delay changes
        trigger re-discretization based on the current simulation time step. Parameters
        can be changed at any point during simulation; in-flight events are not affected.

        Parameters
        ----------
        weight : float, array-like, or Quantity, optional
            New synaptic weight. Must be scalar. If omitted, weight is unchanged.
        delay : float, array-like, or Quantity, optional
            New transmission delay (ms). Must be positive. Will be discretized to
            integer time steps. If omitted, delay is unchanged.
        receptor_type : int, optional
            New receptor port identifier. Must be non-negative integer. If omitted,
            receptor type is unchanged.
        post : Dynamics, optional
            New default postsynaptic receiver. If omitted, receiver is unchanged.
        event_type : str, optional
            New event transmission type. Must be one of ``'spike'``, ``'rate'``,
            ``'current'``, ``'conductance'``, ``'double_data'``, ``'data_logging'``.
            If omitted, event type is unchanged.

        Raises
        ------
        ValueError
            If delay is non-positive, non-finite, or discretizes to less than one step.
        ValueError
            If weight or receptor_type is not scalar.
        ValueError
            If receptor_type is negative or non-integer.
        ValueError
            If event_type is not a recognized string.

        Notes
        -----
        **Delay re-discretization:**

        When ``delay`` is changed, the new value is immediately converted to time steps
        using the current ``dt``. If ``dt`` has not been set in the environment, the
        raw millisecond value is stored and discretization occurs on the next method
        call that requires ``dt``.

        **In-flight events:**

        Events already scheduled in the queue are NOT retroactively modified. Only
        future calls to :meth:`send` or :meth:`update` will use the new parameters.

        **Thread safety:**

        Not thread-safe. Concurrent calls to :meth:`set` from multiple threads will
        cause race conditions on parameter updates.

        Examples
        --------
        **Update single parameter:**

        .. code-block:: python

            >>> import brainpy.state as bs
            >>> import brainunit as u
            >>> syn = bs.static_synapse(weight=1.0, delay=1.0*u.ms)
            >>> syn.set(weight=2.5)  # Only weight changes
            >>> assert syn.weight == 2.5

        **Update multiple parameters atomically:**

        .. code-block:: python

            >>> syn.set(weight=0.8, delay=2.0*u.ms, receptor_type=1)
            >>> params = syn.get()
            >>> assert params['weight'] == 0.8
            >>> assert params['receptor_type'] == 1

        **Switching event type during simulation:**

        .. code-block:: python

            >>> syn = bs.static_synapse(event_type='spike')
            >>> syn.set(event_type='rate')  # Change to rate-coded transmission

        **Delay quantization example:**

        .. code-block:: python

            >>> import brainstate
            >>> with brainstate.environ.context(dt=0.1*u.ms):
            ...     syn = bs.static_synapse(delay=1.44*u.ms)
            ...     print(syn.get()['delay'])  # 1.4 ms (14 steps)
            ...     syn.set(delay=1.45*u.ms)
            ...     print(syn.get()['delay'])  # 1.5 ms (15 steps)
        """
        new_weight = self.weight if weight is _UNSET else self._normalize_scalar_weight(weight)
        new_delay_ms = (
            self._delay_requested_ms
            if delay is _UNSET
            else self._to_scalar_time_ms(delay, name='delay')
        )
        new_receptor = (
            self.receptor_type
            if receptor_type is _UNSET
            else self._to_receptor_type(receptor_type)
        )
        new_post = self.post if post is _UNSET else post
        new_event_type = (
            self.event_type
            if event_type is _UNSET
            else self._normalize_event_type(event_type)
        )

        self._validate_delay(new_delay_ms)
        self.weight = new_weight
        self._delay_requested_ms = float(new_delay_ms)
        self.receptor_type = int(new_receptor)
        self.post = new_post
        self.event_type = new_event_type

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_delay_cache(dt_ms)
        else:
            self.delay = float(self._delay_requested_ms)

    def get(self) -> dict:
        r"""Retrieve current synapse parameters (NEST ``GetStatus`` equivalent).

        Returns a dictionary of all public synapse parameters, including the
        discretized delay in both milliseconds and time steps.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``'weight'`` : float — Current synaptic weight (dimensionless if no units)
            - ``'delay'`` : float — Effective delay in milliseconds (quantized)
            - ``'delay_steps'`` : int — Delay in simulation time steps
            - ``'receptor_type'`` : int — Receptor port identifier
            - ``'event_type'`` : str — Event transmission type
            - ``'synapse_model'`` : str — Always ``'static_synapse'`` (NEST compatibility)

        Notes
        -----
        **Delay consistency:**

        The method ensures delay values reflect the current simulation resolution by
        calling :meth:`_refresh_delay_if_needed`. If ``dt`` changes between synapse
        creation and this call, the delay is automatically re-discretized.

        **Weight units:**

        If ``weight`` was initialized as a ``brainunit.Quantity``, the returned value
        is the dimensionless magnitude in the original units. To preserve units, access
        ``synapse.weight`` directly.

        **NEST compatibility:**

        The returned dictionary structure matches NEST's ``GetStatus`` output format,
        allowing easy translation between NEST scripts and brainpy.state models.

        Examples
        --------
        **Basic parameter retrieval:**

        .. code-block:: python

            >>> import brainpy.state as bs
            >>> import brainunit as u
            >>> import brainstate
            >>> with brainstate.environ.context(dt=0.1*u.ms):
            ...     syn = bs.static_synapse(weight=1.5, delay=2.0*u.ms, receptor_type=1)
            ...     params = syn.get()
            ...     print(params)
            {'weight': 1.5, 'delay': 2.0, 'delay_steps': 20,
             'receptor_type': 1, 'event_type': 'spike',
             'synapse_model': 'static_synapse'}

        **Verifying delay quantization:**

        .. code-block:: python

            >>> with brainstate.environ.context(dt=0.1*u.ms):
            ...     syn = bs.static_synapse(delay=1.47*u.ms)
            ...     params = syn.get()
            ...     print(f"Requested: 1.47 ms")
            ...     print(f"Actual: {params['delay']} ms ({params['delay_steps']} steps)")
            Requested: 1.47 ms
            Actual: 1.5 ms (15 steps)

        **Extracting all parameters for logging:**

        .. code-block:: python

            >>> synapses = [bs.static_synapse(weight=w) for w in [0.5, 1.0, 1.5]]
            >>> weights = [s.get()['weight'] for s in synapses]
            >>> print(f"Synapse weights: {weights}")
            Synapse weights: [0.5, 1.0, 1.5]
        """
        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_delay_if_needed()

        return {
            'weight': self._weight_to_float(self.weight),
            'delay': float(self.delay),
            'delay_steps': int(self._delay_steps),
            'receptor_type': int(self.receptor_type),
            'event_type': self.event_type,
            'synapse_model': 'static_synapse',
        }

    def set_weight(self, weight: ArrayLike):
        r"""Update synaptic weight (NEST connection API compatibility).

        Convenience method that modifies only the weight parameter, leaving delay,
        receptor type, and event type unchanged. Equivalent to ``set(weight=...)``.

        Parameters
        ----------
        weight : float, array-like, or Quantity
            New synaptic weight. Must be scalar.

        Raises
        ------
        ValueError
            If weight is not scalar.

        See Also
        --------
        set : General parameter update method

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bs
            >>> syn = bs.static_synapse(weight=1.0)
            >>> syn.set_weight(2.5)
            >>> assert syn.weight == 2.5
        """
        self.set(weight=weight)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
        event_type: str | None = None,
    ) -> bool:
        r"""Schedule an outgoing event for delayed delivery.

        Immediately scales the input ``multiplicity`` by the synaptic ``weight`` and
        enqueues the result for delivery after ``delay`` milliseconds. The event will
        be delivered to the specified (or default) postsynaptic receiver at the target
        time step.

        This method replicates the NEST ``send`` event processing pipeline:

        1. Check multiplicity is non-zero (zero events are ignored)
        2. Scale by weight: ``payload = multiplicity * weight``
        3. Compute delivery step: ``current_step + delay_steps``
        4. Resolve receiver and receptor type
        5. Enqueue for future delivery

        Parameters
        ----------
        multiplicity : float, array-like, or Quantity, optional
            Event magnitude to transmit. For spike events, typically ``1.0`` (single spike).
            For rate-coded or current events, can be any real value. If zero or
            all-zeros array, the event is ignored and nothing is scheduled.
            Default: ``1.0`` (single unit event).
        post : Dynamics, optional
            Postsynaptic receiver for this event. Overrides the default ``self.post``
            for this call only. Must implement ``add_delta_input`` or ``add_current_input``.
            Default: ``None`` (use ``self.post``).
        receptor_type : int, optional
            Receptor port for this event. Overrides the default ``self.receptor_type``
            for this call only. Must be non-negative integer.
            Default: ``None`` (use ``self.receptor_type``).
        event_type : str, optional
            Event transmission type for this event. Overrides ``self.event_type``
            for this call only. Must be valid event type string.
            Default: ``None`` (use ``self.event_type``).

        Returns
        -------
        bool
            ``True`` if the event was scheduled (non-zero multiplicity),
            ``False`` if the event was ignored (zero multiplicity).

        Raises
        ------
        ValueError
            If no receiver is available (``self.post`` is ``None`` and ``post`` not provided).
        ValueError
            If simulation resolution ``dt`` is not defined in ``brainstate.environ``.
        ValueError
            If delay is invalid or discretizes to less than one time step.
        TypeError
            If the receiver does not implement required input methods.

        Notes
        -----
        **Event scheduling semantics:**

        The event is delivered at simulation time :math:`t_{\text{delivery}} = t_{\text{current}} + d`,
        where :math:`t_{\text{current}}` is the current simulation time and :math:`d`
        is the effective quantized delay. The receiver's input accumulation occurs when
        :meth:`update` is called at that future time step.

        **Zero-weight handling:**

        If ``weight`` is zero, the payload is zero regardless of multiplicity. The event
        is still scheduled (returns ``True``), but the receiver will accumulate a zero
        input contribution. To avoid unnecessary queue overhead, check weights before
        calling :meth:`send`.

        **Concurrent receiver override:**

        Providing ``post`` does NOT change ``self.post``—it only applies to this single
        event. Subsequent :meth:`send` calls without ``post`` argument will revert to
        the default receiver.

        **Performance considerations:**

        - Each call allocates a queue entry: ``(receiver, payload, receptor, event_type)``
        - Large numbers of events can cause queue memory overhead
        - Consider using vectorized projection classes for high-throughput scenarios

        Examples
        --------
        **Basic spike transmission:**

        .. code-block:: python

            >>> import brainpy.state as bs
            >>> import brainunit as u
            >>> import brainstate
            >>> with brainstate.environ.context(dt=0.1*u.ms, t=0.0*u.ms):
            ...     post = bs.LIF(1, V_rest=-65*u.mV, V_th=-50*u.mV, tau=20*u.ms)
            ...     syn = bs.static_synapse(weight=1.0, delay=1.0*u.ms, post=post)
            ...
            ...     # Send single spike
            ...     scheduled = syn.send(multiplicity=1.0)
            ...     assert scheduled is True
            ...
            ...     # Sending zero has no effect
            ...     scheduled = syn.send(multiplicity=0.0)
            ...     assert scheduled is False

        **Rate-coded transmission:**

        .. code-block:: python

            >>> with brainstate.environ.context(dt=0.1*u.ms):
            ...     syn = bs.static_synapse(
            ...         weight=0.1,
            ...         delay=0.5*u.ms,
            ...         event_type='rate',
            ...         post=rate_neuron,
            ...     )
            ...
            ...     # Send rate value (Hz)
            ...     syn.send(multiplicity=42.5)

        **Override receiver and receptor for single event:**

        .. code-block:: python

            >>> # Default: send to neuron_A on receptor 0
            >>> syn = bs.static_synapse(weight=1.0, post=neuron_A, receptor_type=0)
            >>> syn.send(multiplicity=1.0)  # Goes to neuron_A, receptor 0
            >>>
            >>> # Override for this event only
            >>> syn.send(multiplicity=1.0, post=neuron_B, receptor_type=1)
            >>> # Next call reverts to default
            >>> syn.send(multiplicity=1.0)  # Back to neuron_A, receptor 0

        **Multi-spike burst:**

        .. code-block:: python

            >>> # Simulate burst of 5 spikes at 100 Hz
            >>> with brainstate.environ.context(dt=0.1*u.ms):
            ...     syn = bs.static_synapse(weight=0.5, delay=1.0*u.ms, post=target)
            ...     for spike_time in [0.0, 10.0, 20.0, 30.0, 40.0]:  # ms
            ...         brainstate.environ.set(t=spike_time*u.ms)
            ...         syn.send(multiplicity=1.0)

        **Weighted population spike count:**

        .. code-block:: python

            >>> # 100 presynaptic neurons, 23 spike this step
            >>> syn.send(multiplicity=23.0)  # Equivalent to 23 unit events
        """
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST ordering: weight -> delay steps -> receiver -> rport -> deliver.
        weighted_payload = multiplicity * self.weight
        delay_steps = int(self._delay_steps)
        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        ev_type = self.event_type if event_type is None else self._normalize_event_type(event_type)

        delivery_step = int(current_step + delay_steps)
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), ev_type))
        return True

    def update(
        self,
        pre_spike: ArrayLike = 0.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
        event_type: str | None = None,
    ) -> int:
        r"""Execute one simulation time step: deliver queued events and schedule new input.

        This method implements the standard NEST synapse update cycle:

        1. **Delivery phase**: Deliver all events whose scheduled time has arrived
           (current simulation step). Events are removed from the queue and injected
           into the postsynaptic receiver's input accumulation dictionaries.

        2. **Accumulation phase**: Sum the ``pre_spike`` argument with any inputs
           registered via ``add_current_input`` or ``add_delta_input`` (from other
           sources connected to this synapse object).

        3. **Scheduling phase**: If the total input is non-zero, call :meth:`send`
           to schedule a new delayed event with the aggregated multiplicity.

        This two-phase design ensures causal event delivery: events scheduled for
        time :math:`t` are delivered before new inputs at time :math:`t` are processed.

        Parameters
        ----------
        pre_spike : float, array-like, or Quantity, optional
            Presynaptic input for this time step. For spike events, typically ``1.0``
            (spike present) or ``0.0`` (no spike). For rate-coded signals, can be
            any real value representing instantaneous firing rate. This is added to
            any inputs already registered via ``add_current_input``/``add_delta_input``.
            Default: ``0.0`` (no external input).
        post : Dynamics, optional
            Override default postsynaptic receiver for scheduled event (not for delivery).
            Delivered events use the receiver stored when they were scheduled.
            Default: ``None`` (use ``self.post``).
        receptor_type : int, optional
            Override receptor port for newly scheduled event only.
            Default: ``None`` (use ``self.receptor_type``).
        event_type : str, optional
            Override event type for newly scheduled event only.
            Default: ``None`` (use ``self.event_type``).

        Returns
        -------
        int
            Number of events delivered to postsynaptic receiver(s) during the delivery
            phase. Zero if no events were due. Does NOT count the newly scheduled event.

        Raises
        ------
        ValueError
            If simulation resolution ``dt`` is not defined in ``brainstate.environ``.
        ValueError
            If receiver is not configured when attempting to schedule a new event.
        TypeError
            If receiver does not implement required input methods during delivery.

        Notes
        -----
        **Simulation loop integration:**

        Typical usage pattern in a network simulation:

        .. code-block:: python

            for step in range(n_steps):
                # 1. Update all synapses (deliver + schedule)
                for syn in synapses:
                    syn.update(pre_spike=presynaptic_activity[step])

                # 2. Update all neurons (accumulate inputs, integrate dynamics)
                for neuron in neurons:
                    neuron.update()

                # 3. Record outputs
                record_state()

        **Input accumulation semantics:**

        The method calls:

        .. code-block:: python

            total = self.sum_current_inputs(pre_spike)
            total = self.sum_delta_inputs(total)

        This accumulates:

        - ``pre_spike`` argument
        - All ``current_inputs`` registered via ``add_current_input(key, value, ...)``
        - All ``delta_inputs`` registered via ``add_delta_input(key, value, ...)``

        After accumulation, these input dictionaries are automatically cleared for
        the next time step (handled by ``Dynamics`` base class).

        **Event delivery timing:**

        Events scheduled at time :math:`t` with delay :math:`d` are delivered when
        ``update()`` is called at time :math:`t + d`. The delivery step is:

        .. math::

           s_{\text{delivery}} = s_{\text{schedule}} + d_{\text{steps}}

        where :math:`s` are discrete time step indices.

        **Zero-input optimization:**

        If the aggregated input is exactly zero, :meth:`send` is NOT called, avoiding
        unnecessary queue overhead. However, the delivery phase still executes even
        if no new events are scheduled.

        **Return value interpretation:**

        The return value indicates how many events were **delivered** (past events
        arriving now), not how many were **scheduled** (new events for the future).
        Use this for monitoring event throughput or debugging delivery issues.

        Examples
        --------
        **Basic simulation loop:**

        .. code-block:: python

            >>> import brainpy.state as bs
            >>> import brainunit as u
            >>> import brainstate
            >>> with brainstate.environ.context(dt=0.1*u.ms, t=0.0*u.ms):
            ...     post = bs.LIF(1, V_rest=-65*u.mV, V_th=-50*u.mV, tau=20*u.ms)
            ...     syn = bs.static_synapse(weight=1.0, delay=1.0*u.ms, post=post)
            ...
            ...     post.init_all_states()
            ...     syn.init_all_states()
            ...
            ...     # Simulate 20 steps (2 ms)
            ...     for step in range(20):
            ...         t = step * 0.1 * u.ms
            ...         brainstate.environ.set(t=t)
            ...
            ...         # Spike at step 5 (0.5 ms)
            ...         spike = 1.0 if step == 5 else 0.0
            ...
            ...         # Update synapse
            ...         delivered = syn.update(pre_spike=spike)
            ...         if delivered > 0:
            ...             print(f"Step {step}: delivered {delivered} event(s)")
            ...
            ...         # Update neuron
            ...         post.update()
            Step 15: delivered 1 event(s)

        **Monitoring event delivery:**

        .. code-block:: python

            >>> total_delivered = 0
            >>> for step in range(100):
            ...     brainstate.environ.set(t=step*0.1*u.ms)
            ...     n = syn.update(pre_spike=poisson_spike[step])
            ...     total_delivered += n
            >>> print(f"Total events delivered: {total_delivered}")

        **Multi-source input accumulation:**

        .. code-block:: python

            >>> # Synapse receives input from multiple sources
            >>> syn = bs.static_synapse(weight=1.0, post=target)
            >>>
            >>> # Source 1: direct spike input
            >>> syn.add_delta_input('source1', 1.0, label='receptor_0')
            >>>
            >>> # Source 2: continuous current
            >>> syn.add_current_input('source2', 0.5, label='receptor_0')
            >>>
            >>> # Update aggregates both sources
            >>> syn.update(pre_spike=1.0)  # Total multiplicity = 1.0 + 1.0 + 0.5 = 2.5

        **Conditional scheduling based on delivery count:**

        .. code-block:: python

            >>> # Homeostatic mechanism: reduce weight if too many events arrive
            >>> delivered = syn.update(pre_spike=spike)
            >>> if delivered > 5:
            ...     syn.set_weight(syn.weight * 0.95)  # 5% weight decrease

        **Override parameters for specific time step:**

        .. code-block:: python

            >>> # Normally send to receptor 0
            >>> for step in range(10):
            ...     brainstate.environ.set(t=step*0.1*u.ms)
            ...     if step == 5:
            ...         # Special case: route to receptor 1 this step only
            ...         syn.update(pre_spike=1.0, receptor_type=1)
            ...     else:
            ...         syn.update(pre_spike=1.0)

        See Also
        --------
        send : Direct event scheduling without delivery phase
        init_state : Reset event queue and delivery counter
        """
        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        total = self.sum_current_inputs(pre_spike)
        total = self.sum_delta_inputs(total)
        if self._is_nonzero(total):
            self.send(
                total,
                post=post,
                receptor_type=receptor_type,
                event_type=event_type,
            )
        return delivered
