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

from typing import Callable

import numpy as np

import brainstate
import braintools
import brainunit as u
from brainstate.typing import Size

from brainpy_state._base import Dynamics

__all__ = [
    'rate_transformer_node',
]


class rate_transformer_node(Dynamics):
    r"""NEST-compatible ``rate_transformer_node`` template model.

    Short description
    -----------------

    Rate transformer node that sums incoming rates and applies an input
    nonlinearity.

    Description
    -----------

    ``rate_transformer_node`` reproduces NEST's template model
    ``rate_transformer_node<TNonlinearities>``:

    .. math::

       X_i(t) = \phi\!\left(\sum_j w_{ij}\,\psi\!\left(X_j(t-d_{ij})\right)\right)

    The model has no intrinsic rate dynamics and no noise term.
    It only transforms incoming rate events and forwards the transformed result.

    The boolean parameter ``linear_summation`` follows NEST semantics:

    - ``linear_summation=True`` (default): event handlers store weighted rates,
      and the nonlinearity is applied to the summed input during update
      (``input`` behaves as :math:`\phi`).
    - ``linear_summation=False``: event handlers apply the nonlinearity per event
      before summation (``input`` behaves as :math:`\psi`).

    This is the same split as implemented in NEST
    ``rate_transformer_node_impl.h`` event handlers.

    Update ordering (matching NEST ``rate_transformer_node_impl.h``)
    ................................................................

    For each simulation step:

    1. Store outgoing delayed value as previous ``rate``.
    2. Reset the internal accumulator to zero.
    3. Read delayed and instantaneous event buffers.
    4. Compute new rate:
       - ``linear_summation=True``:
         ``rate <- input(delayed + instant)``
       - ``linear_summation=False``:
         ``rate <- delayed + instant``
         (events were already transformed in handlers).
    5. Store outgoing instantaneous value as updated ``rate``.

    Parameters
    ----------
    in_size : Size
        Population shape.
    linear_summation : bool, optional
        Switch controlling where the nonlinearity is applied.
        Default ``True``.
    g : float, optional
        Gain used by the default linear nonlinearity
        ``input(h) = g * h``. Default ``1.0``.
    input_nonlinearity : Callable, optional
        Custom input nonlinearity replacing template ``input``.
        Callable signature can be ``f(h)`` or ``f(model, h)``.
    rate_initializer : Callable, optional
        Initializer for ``rate``. Default ``Constant(0.0)``.
    name : str, optional
        Module name.

    Notes
    -----
    Runtime events:

    - ``instant_rate_events`` are applied in the current step.
    - ``delayed_rate_events`` use integer ``delay_steps``.
    - Event format supports dict or tuple:
      ``(rate, weight)``, ``(rate, weight, delay_steps)``,
      ``(rate, weight, delay_steps, multiplicity)``.

    Connection delays are honored on both incoming and outgoing events.
    As in NEST, inserting a transformer between two neurons introduces an
    extra simulation-step latency compared with direct instantaneous coupling.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        linear_summation: bool = True,
        g: float = 1.0,
        input_nonlinearity: Callable | None = None,
        rate_initializer: Callable = braintools.init.Constant(0.0),
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.linear_summation = bool(linear_summation)
        self.g = braintools.init.param(g, self.varshape)
        self.input_nonlinearity = input_nonlinearity
        self.rate_initializer = rate_initializer

        self._delayed_queue = {}

    @property
    def recordables(self):
        return ['rate']

    @property
    def receptor_types(self):
        return {'RATE': 0}

    @staticmethod
    def _to_numpy(x):
        return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @staticmethod
    def _to_int_scalar(x, name: str):
        arr = np.asarray(u.math.asarray(x), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return int(arr[0])

    @staticmethod
    def _coerce_events(events):
        if events is None:
            return []
        if isinstance(events, dict):
            return [events]
        if isinstance(events, tuple):
            if len(events) == 0:
                return []
            if isinstance(events[0], (dict, tuple, list)):
                return list(events)
            if len(events) in (2, 3, 4):
                return [events]
        if isinstance(events, list):
            if len(events) == 0:
                return []
            if isinstance(events[0], (dict, tuple, list)):
                return events
            if len(events) in (2, 3, 4):
                return [tuple(events)]
        return [events]

    def _call_nl(self, fn: Callable, x: np.ndarray):
        try:
            return fn(self, x)
        except TypeError as first_error:
            try:
                return fn(x)
            except TypeError:
                raise first_error

    def _input_transform(self, h: np.ndarray, state_shape):
        h_np = self._broadcast_to_state(self._to_numpy(h), state_shape)
        if self.input_nonlinearity is None:
            g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
            return g * h_np
        y = self._call_nl(self.input_nonlinearity, h_np)
        return self._broadcast_to_state(self._to_numpy(y), state_shape)

    def _extract_event_fields(self, ev, default_delay_steps: int):
        if isinstance(ev, dict):
            rate = ev.get('rate', ev.get('coeff', ev.get('value', 0.0)))
            weight = ev.get('weight', 1.0)
            multiplicity = ev.get('multiplicity', 1.0)
            delay_steps = ev.get('delay_steps', ev.get('delay', default_delay_steps))
        elif isinstance(ev, (tuple, list)):
            if len(ev) == 2:
                rate, weight = ev
                delay_steps = default_delay_steps
                multiplicity = 1.0
            elif len(ev) == 3:
                rate, weight, delay_steps = ev
                multiplicity = 1.0
            elif len(ev) == 4:
                rate, weight, delay_steps, multiplicity = ev
            else:
                raise ValueError('Rate event tuples must have length 2, 3, or 4.')
        else:
            rate = ev
            weight = 1.0
            multiplicity = 1.0
            delay_steps = default_delay_steps

        delay_steps = self._to_int_scalar(delay_steps, name='delay_steps')
        return rate, weight, multiplicity, delay_steps

    def _event_to_weighted_value(self, ev, default_delay_steps: int, state_shape):
        rate, weight, multiplicity, delay_steps = self._extract_event_fields(ev, default_delay_steps)

        rate_np = self._broadcast_to_state(self._to_numpy(rate), state_shape)
        weight_np = self._broadcast_to_state(self._to_numpy(weight), state_shape)
        multiplicity_np = self._broadcast_to_state(self._to_numpy(multiplicity), state_shape)

        if self.linear_summation:
            weighted_value = rate_np * weight_np * multiplicity_np
        else:
            weighted_value = self._input_transform(rate_np, state_shape) * weight_np * multiplicity_np

        return weighted_value, delay_steps

    @staticmethod
    def _queue_add(queue: dict, step_idx: int, value: np.ndarray):
        if step_idx in queue:
            queue[step_idx] = queue[step_idx] + value
        else:
            queue[step_idx] = np.array(value, dtype=np.float64, copy=True)

    def _drain_delayed_queue(self, step_idx: int, state_shape):
        value = self._delayed_queue.pop(step_idx, None)
        if value is None:
            return np.zeros(state_shape, dtype=np.float64)
        return np.array(self._broadcast_to_state(np.asarray(value, dtype=np.float64), state_shape), copy=True)

    def _accumulate_instant_events(self, events, state_shape):
        total = np.zeros(state_shape, dtype=np.float64)
        for ev in self._coerce_events(events):
            value, delay_steps = self._event_to_weighted_value(
                ev,
                default_delay_steps=0,
                state_shape=state_shape,
            )
            if delay_steps != 0:
                raise ValueError('instant_rate_events must not specify non-zero delay_steps.')
            total += value
        return total

    def _schedule_delayed_events(self, events, step_idx: int, state_shape):
        total_now = np.zeros(state_shape, dtype=np.float64)
        for ev in self._coerce_events(events):
            value, delay_steps = self._event_to_weighted_value(
                ev,
                default_delay_steps=1,
                state_shape=state_shape,
            )
            if delay_steps < 0:
                raise ValueError('delay_steps for delayed_rate_events must be >= 0.')
            if delay_steps == 0:
                total_now += value
            else:
                self._queue_add(self._delayed_queue, step_idx + delay_steps, value)
        return total_now

    def init_state(self, batch_size: int = None, **kwargs):
        rate = braintools.init.param(self.rate_initializer, self.varshape, batch_size)
        rate_np = self._to_numpy(rate)

        self.rate = brainstate.ShortTermState(rate_np)
        self.instant_rate = brainstate.ShortTermState(np.array(rate_np, dtype=np.float64, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(rate_np, dtype=np.float64, copy=True))
        self._step_count = brainstate.ShortTermState(np.asarray(0, dtype=np.int64))

        self._delayed_queue = {}

    def update(self, x=0.0, instant_rate_events=None, delayed_rate_events=None):
        del x  # NEST rate transformer has no intrinsic current input.

        state_shape = self.rate.value.shape
        step_idx = int(np.asarray(self._step_count.value, dtype=np.int64).reshape(-1)[0])

        delayed_total = self._drain_delayed_queue(step_idx, state_shape)
        delayed_total += self._schedule_delayed_events(
            delayed_rate_events,
            step_idx=step_idx,
            state_shape=state_shape,
        )
        instant_total = self._accumulate_instant_events(
            instant_rate_events,
            state_shape=state_shape,
        )

        rate_prev = self._broadcast_to_state(self._to_numpy(self.rate.value), state_shape)
        if self.linear_summation:
            rate_new = self._input_transform(delayed_total + instant_total, state_shape)
        else:
            rate_new = delayed_total + instant_total

        self.rate.value = rate_new
        self.delayed_rate.value = rate_prev
        self.instant_rate.value = rate_new
        self._step_count.value = np.asarray(step_idx + 1, dtype=np.int64)
        return rate_new
