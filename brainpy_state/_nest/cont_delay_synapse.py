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

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence

import brainunit as u
from brainstate.typing import ArrayLike

from .static_synapse import static_synapse

__all__ = [
    'cont_delay_synapse',
]


class cont_delay_synapse(static_synapse):
    r"""NEST-compatible static synapse with continuous (off-grid) delays.

    This synapse model extends :class:`static_synapse` to support precise
    spike timing by decomposing transmission delays into integer delay steps
    and fractional sub-timestep offsets. It mirrors the NEST
    ``cont_delay_synapse`` model and implements the exact subthreshold
    integration method described in Morrison et al. (2007).

    Parameters
    ----------
    weight : ArrayLike, optional
        Synaptic weight multiplier applied to incoming spike events.
        Dimensionless scalar or array. Default: ``1.0``.
    delay : ArrayLike, optional
        Total synaptic transmission delay. Must be ``>= dt`` (simulation
        timestep). Decomposed internally into integer steps and fractional
        offset. Unit: millisecond. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Target receptor port index on the postsynaptic neuron. Used to
        differentiate excitatory/inhibitory or multiple receptor types.
        Default: ``0``.
    post : brainstate.nn.Module, optional
        Default postsynaptic target object. Must implement either
        ``handle_cont_delay_synapse_event(value, receptor_type, event_type, offset)``
        or ``add_precise_spike_event(key, value, offset, label)`` for
        off-grid event delivery. On-grid events fall back to standard
        :class:`static_synapse` delivery via ``add_current_input`` or
        ``add_delta_input``. Default: ``None``.
    event_type : str, optional
        Event transmission mode. Supported values: ``'spike'`` (discrete
        delta events), ``'rate'`` (continuous rate signals), ``'current'``
        (arbitrary current injection). Default: ``'spike'``.
    name : str, optional
        Unique identifier for this synapse instance. Used for debugging and
        event tracking. Default: ``None`` (auto-generated).

    Mathematical Model
    ------------------

    **1. Delay Decomposition**

    Given a continuous delay :math:`d` and simulation timestep :math:`\Delta t`,
    the model decomposes :math:`d` into:

    - Integer delay steps: :math:`d_{\text{steps}} \in \mathbb{N}`
    - Fractional offset: :math:`d_{\text{offset}} \in [0, \Delta t)` (in ms)

    such that the effective delay satisfies:

    .. math::

        d_{\text{eff}} = d_{\text{steps}} \cdot \Delta t - d_{\text{offset}}

    The decomposition algorithm follows NEST conventions:

    - **Case 1: On-grid delay** (:math:`d / \Delta t \in \mathbb{N}`):

      .. math::

          d_{\text{steps}} &= \frac{d}{\Delta t} \\
          d_{\text{offset}} &= 0

    - **Case 2: Off-grid delay** (:math:`d / \Delta t \notin \mathbb{N}`):

      .. math::

          d_{\text{steps}} &= \lfloor d / \Delta t \rfloor + 1 \\
          d_{\text{offset}} &= \Delta t \cdot \left(1 - \text{frac}(d / \Delta t)\right)

      where :math:`\text{frac}(x) = x - \lfloor x \rfloor` is the fractional part.

    **Constraint:** :math:`d \geq \Delta t` must hold. Violations raise
    ``ValueError`` during initialization or timestep changes.

    **2. Event Scheduling with Source Offsets**

    When a presynaptic spike arrives with source offset :math:`o_{\text{src}}`
    (measured from the right edge of the current timestep), the effective
    event offset becomes:

    .. math::

        o_{\text{total}} = o_{\text{src}} + d_{\text{offset}}

    **Carry handling:** If :math:`o_{\text{total}} \geq \Delta t`, the event
    "carries over" to the next timestep:

    .. math::

        d_{\text{steps}}^{\text{adj}} &= d_{\text{steps}} - 1 \\
        o_{\text{event}} &= o_{\text{total}} - \Delta t

    Otherwise:

    .. math::

        d_{\text{steps}}^{\text{adj}} &= d_{\text{steps}} \\
        o_{\text{event}} &= o_{\text{total}}

    The adjusted delay steps determine when the event is delivered, and
    :math:`o_{\text{event}}` specifies the sub-timestep delivery time.

    **3. Event Delivery**

    Events are queued at timestep :math:`t_{\text{deliver}} = t_{\text{current}} + d_{\text{steps}}^{\text{adj}}`
    and delivered with offset :math:`o_{\text{event}}`. The delivery mechanism
    depends on the receiver's capabilities:

    - Off-grid delivery: Calls ``receiver.handle_cont_delay_synapse_event(value, receptor_type, event_type, offset)``
      if available. The receiver integrates the event at precise time
      :math:`t_{\text{step}} - o_{\text{event}}` (measured from right edge).
    - On-grid fallback: If :math:`o_{\text{event}} \approx 0`, uses
      standard :class:`static_synapse` delivery (``add_current_input`` or
      ``add_delta_input``).
    - Precise spike API: For spike events, optionally calls
      ``receiver.add_precise_spike_event(key, value, offset, label)`` if
      ``handle_cont_delay_synapse_event`` is unavailable.

    **4. Weight Multiplication**

    Spike multiplicity is scaled by the synaptic weight:

    .. math::

        \text{payload} = m \cdot w

    where :math:`m` is the incoming spike count and :math:`w` is the weight.

    Parameter Mapping
    -----------------

    ================= ================================ ========================
    NEST Parameter    brainpy.state Parameter          Notes
    ================= ================================ ========================
    ``weight``        ``weight``                       Dimensionless multiplier
    ``delay``         ``delay``                        Total delay (ms)
    ``delay_steps``   ``_delay_steps`` (internal)      Integer steps (auto)
    ``delay_offset``  ``_delay_offset_ms`` (internal)  Fractional offset (auto)
    ``receptor_type`` ``receptor_type``                Receptor port index
    ================= ================================ ========================

    Internal state ``_delay_steps`` and ``_delay_offset_ms`` are computed
    automatically during initialization and when the simulation timestep
    changes. They are exposed via ``get()`` for inspection.

    Computational Properties
    ------------------------

    - Time complexity: :math:`O(1)` per event (queue insertion/retrieval
      uses dict lookups).
    - Space complexity: :math:`O(E \cdot D)` where :math:`E` is the number
      of pending events and :math:`D` is the maximum delay in timesteps.
    - Numerical precision: Offset arithmetic uses double precision. Offsets
      within :math:`10^{-15}` ms of zero are treated as on-grid for
      numerical stability.
    - Exact integration: When combined with compatible neuron models
      (e.g., NEST's ``*_ps`` variants), achieves machine-precision spike
      timing independent of timestep choice (within numerical limits).

    Failure Modes
    -------------

    - **ValueError:** Raised if ``delay < dt`` or ``dt <= 0`` during
      initialization or ``_refresh_delay_cache``.
    - **TypeError:** Raised if receiver does not implement off-grid event
      delivery API when required (offset > 0).
    - **ValueError:** Raised if source offsets violate ``0 <= offset <= dt``.

    See Also
    --------
    static_synapse : Base class for fixed-weight synapses
    iaf_psc_exp_ps : NEST neuron model with precise spike timing support

    Notes
    -----
    **Event Format:** The ``update`` method accepts precise spike events via
    the ``spike_events`` parameter in two formats:

    - **Tuple format:** ``(offset, multiplicity)`` where ``offset`` is in
      milliseconds.
    - **Dict format:** ``{'offset': value, 'multiplicity': value}`` for
      explicit labeling.
    - **Sequence format:** List of tuples or dicts for multiple events in
      one timestep.

    **Offset Convention:** All offsets follow NEST's precise-time convention:
    measured **backward** from the right edge of the current timestep. An
    offset of ``0`` means delivery at the end of the step; ``dt`` means
    delivery at the start. This matches the continuous-time interpretation
    where time increases left-to-right within the discrete step.

    **Timestep Changes:** When the simulation timestep changes (e.g., via
    ``brainstate.environ.context(dt=...)``), the model automatically recomputes
    ``_delay_steps`` and ``_delay_offset_ms`` to maintain the requested delay
    value. This may alter the effective delay within machine precision bounds.

    **Warnings:** If ``delay`` is specified in NEST-style ``Connect`` calls
    (via ``check_synapse_params``), a warning is issued because connect-time
    delays are rounded to integer timesteps. For precise delays, define them
    in the synapse model itself (e.g., via NEST's ``CopyModel``).

    References
    ----------
    .. [1] Morrison A, Straube S, Plesser HE, Diesmann M (2007). Exact
           Subthreshold Integration with Continuous Spike Times in Discrete
           Time Neural Network Simulations. *Neural Computation* 19(1):47-79.
           DOI: https://doi.org/10.1162/neco.2007.19.1.47
    .. [2] NEST Simulator source code: ``models/cont_delay_synapse.h``,
           ``models/cont_delay_synapse_impl.h``, ``models/cont_delay_synapse.cpp``
           (version 3.0+). Available at https://github.com/nest/nest-simulator

    Examples
    --------
    **Basic usage with on-grid delay:**

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainstate as bst
        >>> import brainunit as u
        >>> with bst.environ.context(dt=0.1 * u.ms):
        ...     syn = bst.nest.cont_delay_synapse(
        ...         weight=2.5,
        ...         delay=1.0 * u.ms,  # 10 steps at dt=0.1 ms
        ...     )
        ...     print(syn.get())
        {'weight': 2.5, 'delay': 1.0, 'delay_offset': 0.0, 'receptor_type': 0,
         'synapse_model': 'cont_delay_synapse'}

    **Off-grid delay with fractional offset:**

    .. code-block:: python

        >>> with bst.environ.context(dt=0.1 * u.ms):
        ...     syn = bst.nest.cont_delay_synapse(
        ...         weight=1.0,
        ...         delay=1.23 * u.ms,  # Not a multiple of 0.1 ms
        ...     )
        ...     params = syn.get()
        ...     print(f"delay_steps: {syn._delay_steps}, "
        ...           f"delay_offset: {params['delay_offset']:.4f} ms")
        delay_steps: 13, delay_offset: 0.0700 ms
        # Effective delay: 13 * 0.1 - 0.07 = 1.23 ms

    **Sending events with source offsets:**

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> with bst.environ.context(dt=0.1 * u.ms):
        ...     neuron = bst.nn.LIF(1, V_rest=-70 * u.mV)
        ...     syn = bst.nest.cont_delay_synapse(
        ...         weight=5.0,
        ...         delay=0.5 * u.ms,
        ...         post=neuron,
        ...     )
        ...     syn.init_all_states()
        ...     neuron.init_all_states()
        ...     # Send spike with 0.03 ms offset from step edge
        ...     syn.send(multiplicity=1.0, source_offset=0.03 * u.ms)

    **Processing multiple precise events per timestep:**

    .. code-block:: python

        >>> with bst.environ.context(dt=0.1 * u.ms):
        ...     syn = bst.nest.cont_delay_synapse(weight=1.0, delay=0.5 * u.ms)
        ...     syn.init_all_states()
        ...     # Pass list of (offset, multiplicity) tuples
        ...     events = [
        ...         (0.02 * u.ms, 1.0),  # Early spike
        ...         (0.08 * u.ms, 2.0),  # Late spike (double)
        ...     ]
        ...     delivered = syn.update(spike_events=events)

    **Checking delay decomposition for validation:**

    .. code-block:: python

        >>> with bst.environ.context(dt=0.1 * u.ms):
        ...     syn = bst.nest.cont_delay_synapse(delay=0.37 * u.ms)
        ...     syn.init_all_states()
        ...     params = syn.get()
        ...     d_eff = syn._delay_steps * 0.1 - params['delay_offset']
        ...     print(f"Requested: 0.37 ms, Effective: {d_eff:.2f} ms")
        Requested: 0.37 ms, Effective: 0.37 ms
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        post=None,
        event_type: str = 'spike',
        name: str | None = None,
    ):
        self._delay_offset_ms = 0.0
        super().__init__(
            weight=weight,
            delay=delay,
            receptor_type=receptor_type,
            post=post,
            event_type=event_type,
            name=name,
        )

    def _refresh_delay_cache(self, dt_ms: float):
        if dt_ms <= 0.0:
            raise ValueError('Simulation resolution must be strictly positive.')

        delay_ms = float(self._delay_requested_ms)
        if delay_ms < dt_ms:
            raise ValueError('Continuous delay must be greater than or equal to resolution.')

        ratio = delay_ms / dt_ms
        frac_part, int_part = math.modf(ratio)
        int_part_i = int(int_part)

        if frac_part == 0.0:
            if int_part_i < 1:
                raise ValueError('Continuous delay must be greater than or equal to resolution.')
            self._delay_steps = int_part_i
            self._delay_offset_ms = 0.0
        else:
            lowerbound = int_part_i
            if lowerbound < 1:
                raise ValueError('Continuous delay must be greater than or equal to resolution.')
            self._delay_steps = lowerbound + 1
            self._delay_offset_ms = dt_ms * (1.0 - frac_part)

        self.delay = float(self._delay_steps * dt_ms - self._delay_offset_ms)
        self._dt_cache_ms = float(dt_ms)

    def get(self) -> dict:
        r"""Return current public parameters including delay decomposition.

        Returns
        -------
        dict
            Parameter dictionary with keys:

            - ``'weight'`` : float — Synaptic weight multiplier.
            - ``'delay'`` : float — Effective total delay in milliseconds
              (computed as :math:`d_{\text{steps}} \cdot \Delta t - d_{\text{offset}}`).
            - ``'delay_offset'`` : float — Fractional sub-timestep offset in
              milliseconds (range ``[0, dt)``). Zero for on-grid delays.
            - ``'receptor_type'`` : int — Target receptor port index.
            - ``'synapse_model'`` : str — Always ``'cont_delay_synapse'``.

        Notes
        -----
        The ``delay_offset`` field is specific to this model and not present
        in the base :class:`static_synapse`. Integer delay steps
        (``_delay_steps``) are not exposed but can be accessed via the
        internal attribute for debugging.
        """
        params = super().get()
        params['delay_offset'] = float(self._delay_offset_ms)
        params['synapse_model'] = 'cont_delay_synapse'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        r"""Validate and warn about connect-time synapse specifications.

        Issues a warning if ``delay`` is specified in connect-time synapse
        dictionaries (e.g., in NEST-style ``Connect`` calls), because such
        delays are rounded to integer multiples of the simulation timestep
        rather than using the continuous-delay decomposition.

        Parameters
        ----------
        syn_spec : dict or None
            Synapse specification dictionary, typically from NEST-style
            connection APIs. Expected keys: ``'delay'``, ``'weight'``, etc.

        Warns
        -----
        UserWarning
            If ``'delay'`` key is present in ``syn_spec``. The warning message
            advises defining delays in the synapse model definition (e.g.,
            via NEST's ``CopyModel``) to preserve precise sub-timestep offsets.

        Notes
        -----
        This method mirrors NEST's ``cont_delay_synapse`` behavior, which
        prints a similar warning when delays are supplied at connection time.
        To avoid rounding, instantiate the synapse with the desired ``delay``
        parameter directly rather than passing it via connection specs.
        """
        if syn_spec is None:
            return
        if 'delay' in syn_spec:
            warnings.warn(
                'The delay will be rounded to the next multiple of the time step. '
                'To use a more precise time delay it needs to be defined within '
                'the synapse, e.g. with CopyModel().',
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def _canonicalize_spike_events(spike_events):
        if spike_events is None:
            return []
        if isinstance(spike_events, dict):
            return [spike_events]
        if isinstance(spike_events, tuple) and len(spike_events) == 2:
            return [spike_events]
        if isinstance(spike_events, Sequence):
            return spike_events
        raise ValueError(f'Unsupported spike event format: {spike_events}.')

    def _parse_source_events(self, spike_events, dt_ms: float):
        parsed = []
        for ev in self._canonicalize_spike_events(spike_events):
            if isinstance(ev, Mapping):
                if 'offset' not in ev or 'multiplicity' not in ev:
                    raise ValueError(
                        'Each source event dict must contain "offset" and "multiplicity".'
                    )
                offset, multiplicity = ev['offset'], ev['multiplicity']
            else:
                offset, multiplicity = ev

            offset_ms = self._to_scalar_time_ms(offset, name='offset')
            if offset_ms < 0.0 or offset_ms > dt_ms:
                raise ValueError('All source event offsets must satisfy 0 <= offset <= dt.')
            if self._is_nonzero(multiplicity):
                parsed.append((offset_ms, multiplicity))

        return parsed

    def _deliver_event_with_offset(
        self,
        receiver,
        value,
        receptor_type: int,
        event_type: str,
        offset_ms: float,
    ):
        if hasattr(receiver, 'handle_cont_delay_synapse_event'):
            receiver.handle_cont_delay_synapse_event(
                value,
                int(receptor_type),
                event_type,
                float(offset_ms),
            )
            return

        # On-grid fallback: use static synapse delivery path.
        if math.isclose(float(offset_ms), 0.0, rel_tol=0.0, abs_tol=1e-15):
            super()._deliver_event(receiver, value, receptor_type, event_type)
            return

        # Optional precise spike API.
        if event_type == 'spike' and hasattr(receiver, 'add_precise_spike_event'):
            key = f'{self.name}_event_{self._delivery_counter}'
            self._delivery_counter += 1
            label = self._receiver_label(receptor_type)
            receiver.add_precise_spike_event(
                key,
                value,
                float(offset_ms),
                label=label,
            )
            return

        raise TypeError(
            'Receiver does not support off-grid event delivery. '
            'Provide handle_cont_delay_synapse_event(...) or add_precise_spike_event(...).'
        )

    def _deliver_due_events(self, step: int) -> int:
        queued = self._queue.pop(int(step), None)
        if queued is None:
            return 0
        for receiver, value, receptor_type, event_type, offset_ms in queued:
            self._deliver_event_with_offset(receiver, value, receptor_type, event_type, offset_ms)
        return len(queued)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        source_offset: ArrayLike = 0.0 * u.ms,
        post=None,
        receptor_type: ArrayLike | None = None,
        event_type: str | None = None,
    ) -> bool:
        r"""Schedule an outgoing synaptic event with continuous-delay offset.

        Implements the NEST ``cont_delay_synapse`` event scheduling algorithm:
        combines the source spike offset with the synapse's fractional delay
        offset, handles carry-over if the sum exceeds the timestep, and queues
        the event for delivery at the appropriate future timestep.

        Parameters
        ----------
        multiplicity : ArrayLike, optional
            Spike count or event magnitude. Multiplied by ``self.weight`` to
            compute the delivered payload. Must be non-negative scalar or
            array. Default: ``1.0`` (single spike).
        source_offset : ArrayLike, optional
            Sub-timestep offset of the source spike, measured backward from
            the right edge of the current timestep. Must satisfy
            ``0 <= source_offset <= dt``. Unit: millisecond. Default: ``0.0 * u.ms``
            (spike occurred at step boundary).
        post : brainstate.nn.Module, optional
            Override the default postsynaptic target for this event. If
            ``None``, uses ``self.post``. Default: ``None``.
        receptor_type : ArrayLike, optional
            Override the default receptor port index for this event. If
            ``None``, uses ``self.receptor_type``. Must be integer-valued.
            Default: ``None``.
        event_type : str, optional
            Override the default event transmission type for this event. If
            ``None``, uses ``self.event_type``. Supported values: ``'spike'``,
            ``'rate'``, ``'current'``. Default: ``None``.

        Returns
        -------
        bool
            ``True`` if the event was scheduled (or delivered immediately),
            ``False`` if the event was skipped due to zero multiplicity.

        Raises
        ------
        ValueError
            If ``source_offset`` violates the constraint ``0 <= source_offset <= dt``.
        TypeError
            If the receiver does not implement the required off-grid event
            delivery API when the effective offset is non-zero.

        Notes
        -----
        **Offset Arithmetic:**

        The effective event offset is computed as:

        .. math::

            o_{\text{total}} = o_{\text{src}} + d_{\text{offset}}

        If :math:`o_{\text{total}} \geq \Delta t`, a carry occurs:

        - Adjusted delay steps -- :math:`d_{\text{steps}} - 1`
        - Final event offset -- :math:`o_{\text{total}} - \Delta t`

        Otherwise, no carry:

        - Adjusted delay steps -- :math:`d_{\text{steps}}`
        - Final event offset -- :math:`o_{\text{total}}`

        **Immediate Delivery:** If the carry reduces delay steps to zero, the
        event is delivered immediately via ``_deliver_event_with_offset``
        rather than being queued.

        **Queue Structure:** Events are stored in ``self._queue`` (a
        defaultdict mapping delivery timestep to event list). Each queued
        entry is a tuple: ``(receiver, payload, receptor_type, event_type, offset)``.

        **Example:**

        .. code-block:: python

            >>> # Synapse with 1.23 ms delay (13 steps, 0.07 ms offset at dt=0.1)
            >>> # Source spike at 0.05 ms offset
            >>> # Total: 0.05 + 0.07 = 0.12 ms >= 0.1 ms → carry
            >>> # Adjusted: 12 steps, 0.02 ms offset
            >>> syn.send(multiplicity=1.0, source_offset=0.05 * u.ms)
            True
        """
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        weighted_payload = multiplicity * self.weight
        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        ev_type = self.event_type if event_type is None else self._normalize_event_type(event_type)

        source_offset_ms = self._to_scalar_time_ms(source_offset, name='source_offset')
        if source_offset_ms < 0.0 or source_offset_ms > dt_ms:
            raise ValueError('source_offset must satisfy 0 <= source_offset <= dt.')

        total_offset_ms = source_offset_ms + float(self._delay_offset_ms)
        if total_offset_ms < dt_ms:
            delay_steps = int(self._delay_steps)
            event_offset_ms = float(total_offset_ms)
        else:
            delay_steps = int(self._delay_steps - 1)
            event_offset_ms = float(total_offset_ms - dt_ms)

        # If carry reduces delay to zero steps, the event is due in this step.
        if delay_steps == 0:
            self._deliver_event_with_offset(
                receiver,
                weighted_payload,
                int(rport),
                ev_type,
                event_offset_ms,
            )
            return True

        delivery_step = int(current_step + delay_steps)
        self._queue[delivery_step].append(
            (
                receiver,
                weighted_payload,
                int(rport),
                ev_type,
                event_offset_ms,
            )
        )
        return True

    def update(
        self,
        pre_spike: ArrayLike = 0.0,
        *,
        spike_events=None,
        post=None,
        receptor_type: ArrayLike | None = None,
        event_type: str | None = None,
    ) -> int:
        r"""Process one simulation timestep: deliver queued events and schedule new ones.

        This method implements the standard synapse update cycle:

        1. **Deliver due events:** Retrieve and deliver all events scheduled for
           the current timestep from the internal queue.
        2. **Schedule on-grid events:** Sum presynaptic input from ``pre_spike``
           and registered current/delta inputs, then schedule with zero offset.
        3. **Schedule precise events:** Process each event in ``spike_events``
           with its specified sub-timestep offset.

        Parameters
        ----------
        pre_spike : ArrayLike, optional
            On-grid presynaptic spike count or rate. Treated as occurring at
            the right edge of the timestep (offset = 0). Scalar or array.
            Default: ``0.0`` (no on-grid input).
        spike_events : list of tuples/dicts, tuple, dict, or None, optional
            Precise spike events with sub-timestep timing. Supported formats:

            - Single tuple: ``(offset, multiplicity)``
            - Single dict: ``{'offset': value, 'multiplicity': value}``
            - List of tuples/dicts: Multiple events in one step

            Each ``offset`` must satisfy ``0 <= offset <= dt`` (in ms).
            Default: ``None`` (no precise events).
        post : brainstate.nn.Module, optional
            Override the default postsynaptic target. Default: ``None``
            (use ``self.post``).
        receptor_type : ArrayLike, optional
            Override the default receptor port index. Default: ``None``
            (use ``self.receptor_type``).
        event_type : str, optional
            Override the default event type. Supported: ``'spike'``, ``'rate'``,
            ``'current'``. Default: ``None`` (use ``self.event_type``).

        Returns
        -------
        int
            Number of events delivered during this timestep (from the queue).
            Does not include newly scheduled events.

        Raises
        ------
        ValueError
            If any event offset in ``spike_events`` violates ``0 <= offset <= dt``.
        ValueError
            If event dicts are missing required keys ``'offset'`` or
            ``'multiplicity'``.

        Notes
        -----
        **Processing Order:** The three-step sequence (deliver → on-grid → precise)
        ensures deterministic behavior when events from previous timesteps
        arrive simultaneously with new inputs. Queued events are always processed
        before new scheduling occurs.

        **Input Aggregation:** On-grid inputs are summed across all sources:

        - Explicit ``pre_spike`` argument
        - Inputs registered via ``add_current_input(label, value)``
        - Inputs registered via ``add_delta_input(label, value)``

        **Event Format Examples:**

        .. code-block:: python

            # Single tuple
            syn.update(spike_events=(0.05 * u.ms, 2.0))

            # Single dict
            syn.update(spike_events={'offset': 0.05 * u.ms, 'multiplicity': 2.0})

            # Multiple events
            syn.update(spike_events=[
                (0.02 * u.ms, 1.0),
                (0.08 * u.ms, 3.0),
            ])

        **Return Value:** Only counts delivered events. Newly scheduled events
        (from ``pre_spike`` or ``spike_events``) will be counted in future
        timesteps when they are delivered.

        See Also
        --------
        send : Schedule a single event with offset
        """
        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        total = self.sum_current_inputs(pre_spike)
        total = self.sum_delta_inputs(total)
        if self._is_nonzero(total):
            self.send(
                total,
                source_offset=0.0 * u.ms,
                post=post,
                receptor_type=receptor_type,
                event_type=event_type,
            )

        for offset_ms, multiplicity in self._parse_source_events(spike_events, dt_ms):
            self.send(
                multiplicity,
                source_offset=offset_ms * u.ms,
                post=post,
                receptor_type=receptor_type,
                event_type=event_type,
            )

        return delivered
