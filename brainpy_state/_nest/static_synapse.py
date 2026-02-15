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
from collections import defaultdict

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike

from brainpy_state._base import Dynamics

__all__ = [
    'static_synapse',
]


_UNSET = object()
_SPIKE_EVENT = 'spike'
_CURRENT_EVENT_TYPES = {'rate', 'current', 'conductance'}
_PASS_THROUGH_EVENT_TYPES = {'double_data', 'data_logging'}
_ALL_EVENT_TYPES = {_SPIKE_EVENT, *_CURRENT_EVENT_TYPES, *_PASS_THROUGH_EVENT_TYPES}


class static_synapse(Dynamics):
    r"""NEST-compatible ``static_synapse`` connection model.

    Short description
    -----------------

    Synapse type for static (non-plastic) connections.

    Description
    -----------

    ``static_synapse`` mirrors the NEST connection model of the same name.
    The synapse stores fixed parameters and performs no plasticity:

    - synaptic weight ``weight``,
    - synaptic delay ``delay``,
    - receiver port ``receptor_type``.

    On every outgoing event, the payload is scaled by ``weight`` and scheduled
    for delivery after ``delay``. Parameters remain constant unless explicitly
    changed via :meth:`set`.

    Event send ordering (NEST source equivalent)
    --------------------------------------------

    NEST ``models/static_synapse.h`` applies event fields in this strict order:

    1. ``e.set_weight(weight_)``
    2. ``e.set_delay_steps(get_delay_steps())``
    3. ``e.set_receiver(*get_target(tid))``
    4. ``e.set_rport(get_rport())``
    5. ``e()`` (deliver event)

    This implementation preserves the same semantic ordering during scheduling:
    weight scaling is computed first, delay steps second, receiver selection
    third, receptor port fourth, and delivery is performed when the scheduled
    step is reached.

    Delay semantics
    ---------------

    NEST stores delays in integer simulation steps and converts
    milliseconds to steps using ``ld_round`` (round-to-nearest, midpoint-up).
    This implementation reproduces that mapping:

    .. math::

       d_{\mathrm{steps}} = \left\lfloor \frac{d_{\mathrm{ms}}}{dt_{\mathrm{ms}}} + 0.5 \right\rfloor

    with the validity constraint :math:`d_{\mathrm{steps}} \ge 1`.
    The effective public delay is then
    :math:`d_{\mathrm{eff}} = d_{\mathrm{steps}} \cdot dt`.

    Event types
    -----------

    The NEST model transmits several event classes (Spike/Rate/Current/
    Conductance/DoubleData/DataLoggingRequest). In this backend, event delivery
    is represented through receiver input APIs:

    - ``event_type='spike'`` -> ``receiver.add_delta_input(...)``
    - ``event_type in {'rate', 'current', 'conductance'}`` ->
      ``receiver.add_current_input(...)``
    - ``event_type in {'double_data', 'data_logging'}`` ->
      try ``add_current_input``, otherwise ``add_delta_input``.

    If the receiver exposes ``handle_static_synapse_event(value, receptor_type, event_type)``,
    that callback is used directly.

    Parameters
    ----------
    weight : ArrayLike, optional
        Fixed synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    post : object, optional
        Default receiver object. If omitted, a receiver must be passed to
        :meth:`send` / :meth:`update` when scheduling events.
    event_type : str, optional
        Event transmission type. One of ``'spike'``, ``'rate'``, ``'current'``,
        ``'conductance'``, ``'double_data'``, ``'data_logging'``.
        Default: ``'spike'``.
    name : str, optional
        Object name.

    Notes
    -----
    - This is a connection abstraction (not a plastic synapse dynamics model).
    - The class is intentionally lightweight and single-connection oriented.
    - Weight units are receiver dependent (as in NEST). For spike events this
      is typically pA, nS, or mV depending on the target model.

    References
    ----------
    .. [1] NEST source: ``models/static_synapse.h`` and
           ``models/static_synapse.cpp``.
    .. [2] NEST synapse specification docs:
           https://nest-simulator.readthedocs.io/en/stable/synapses/synapse_specification.html
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
        """Set NEST-style public parameters."""
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
        """Return current public parameters."""
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
        """Compatibility wrapper mirroring NEST ``set_weight``."""
        self.set(weight=weight)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
        event_type: str | None = None,
    ) -> bool:
        """Schedule one outgoing event according to static synapse parameters."""
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
        """Deliver due events, then schedule current-step presynaptic input.

        This mirrors one simulation-step connection processing:

        1. Deliver all events scheduled for the current step.
        2. Aggregate presynaptic multiplicity from argument plus registered
           current/delta inputs.
        3. If non-zero, schedule the weighted event with configured delay.

        Returns
        -------
        int
            Number of events delivered in this step.
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
