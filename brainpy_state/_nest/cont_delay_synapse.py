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
    r"""NEST-compatible ``cont_delay_synapse`` connection model.

    Short description
    -----------------

    Static synapse with continuous (off-grid) delays.

    Description
    -----------

    ``cont_delay_synapse`` mirrors NEST ``models/cont_delay_synapse``.
    It relaxes the integer-delay restriction by decomposing delay into:

    - integer delay steps ``delay_steps``,
    - fractional delay offset ``delay_offset`` in ms,

    so that effective delay is

    .. math::

       d_{\mathrm{eff}} = d_{\mathrm{steps}} \cdot dt - d_{\mathrm{offset}}.

    Delay decomposition follows NEST source logic:

    - if ``delay / dt`` is an integer, use on-grid delay with
      ``delay_offset = 0``;
    - otherwise set
      ``delay_steps = floor(delay / dt) + 1`` and
      ``delay_offset = dt * (1 - frac(delay / dt))``.

    Continuous delays must satisfy ``delay >= dt``.

    Event send ordering (NEST source equivalent)
    --------------------------------------------

    NEST ``models/cont_delay_synapse.h`` sends events in this order:

    1. set receiver
    2. set weight
    3. set receptor port
    4. combine source event offset with synaptic ``delay_offset``
    5. if carry occurs (sum >= ``dt``), reduce delay steps by one and subtract ``dt`` from offset
    6. deliver event and restore source offset

    This implementation preserves the same semantics through ``send(...)``
    and ``update(...)``.

    Parameters
    ----------
    weight : ArrayLike, optional
        Fixed synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Continuous synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    post : object, optional
        Default receiver object.
    event_type : str, optional
        Event transmission type. Same supported values as
        :class:`static_synapse`. Default: ``'spike'``.
    name : str, optional
        Object name.

    Notes
    -----
    - ``update(spike_events=...)`` accepts source events as
      ``(offset, multiplicity)`` tuples or dicts with keys
      ``offset`` and ``multiplicity``.
    - Offsets are measured from the right edge of the current step
      (NEST precise-time convention), constrained by ``0 <= offset <= dt``.
    - Off-grid delivery requires one of:
      ``receiver.handle_cont_delay_synapse_event(...)`` or
      ``receiver.add_precise_spike_event(...)``.
      On-grid events fall back to :class:`static_synapse` delivery.

    References
    ----------
    .. [1] NEST source: ``models/cont_delay_synapse.h``,
           ``models/cont_delay_synapse_impl.h`` and
           ``models/cont_delay_synapse.cpp``.
    .. [2] Morrison A, Straube S, Plesser HE, Diesmann M (2007).
           Exact Subthreshold Integration with Continuous Spike Times in
           Discrete Time Neural Network Simulations. Neural Computation.
           DOI: https://doi.org/10.1162/neco.2007.19.1.47
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
        """Return current public parameters."""
        params = super().get()
        params['delay_offset'] = float(self._delay_offset_ms)
        params['synapse_model'] = 'cont_delay_synapse'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        """Warn when ``delay`` is supplied in connect-time synapse specs.

        This mirrors the NEST warning that connect-time delay values are rounded.
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
        """Schedule one outgoing event with continuous-delay offset handling."""
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
        """Deliver due events, then schedule current-step source events.

        Processing order mirrors :class:`static_synapse`:

        1. Deliver events due at current step.
        2. Schedule on-grid source multiplicity from ``pre_spike`` plus registered
           current/delta inputs (offset ``0``).
        3. Schedule additional precise source events provided by ``spike_events``.
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
