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
from dataclasses import dataclass
from typing import Mapping, Sequence

import brainstate
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'multimeter',
]


@dataclass
class _PendingSample:
    stamp_step: int
    senders: np.ndarray
    values: dict[str, np.ndarray]


@dataclass
class _StepCalibration:
    dt_ms: float
    interval_steps: int
    offset_steps: int
    t_min_steps: int
    t_max_steps: float


class multimeter(brainstate.nn.Dynamics):
    r"""NEST-compatible analog recorder for neuron/device state variables.

    Description
    -----------
    ``multimeter`` samples analog variables from connected targets, matching
    the NEST ``multimeter`` model semantics:

    * Sample timestamps are aligned by ``offset + k * interval``.
    * Recording window is ``(origin + start, origin + stop]``.
    * ``start`` is exclusive and ``stop`` is inclusive.
    * Samples are delivered with request/reply lag: values sampled at one step
      become visible on the next call to :meth:`update` (or :meth:`flush`).

    This implementation is intended for brainpy.state step loops where users
    call ``multimeter.update(...)`` once per simulation step after updating the
    recorded neuron state.

    Parameters
    ----------
    in_size : int, optional
        Device batch size. Defaults to 1.
    record_from : sequence of str, optional
        Names of variables to record. Defaults to ``()``.
    interval : Quantity[ms], optional
        Sampling interval. Must be a positive multiple of ``dt``.
        Defaults to ``1.0 * u.ms``.
    offset : Quantity[ms], optional
        Sampling offset relative to global time origin. Must be 0 or a
        positive multiple of ``dt``. Defaults to ``0.0 * u.ms``.
    start : Quantity[ms], optional
        Recording window start, relative to ``origin``.
        Defaults to ``0.0 * u.ms``.
    stop : Quantity[ms] or None, optional
        Recording window stop, relative to ``origin``.
        ``None`` means +infinity. Defaults to ``None``.
    origin : Quantity[ms], optional
        Recording window origin shift. Defaults to ``0.0 * u.ms``.
    time_in_steps : bool, optional
        If ``True``, reported ``events['times']`` are integer step indices
        (NEST ``time_in_steps`` style) and ``events['offsets']`` are included.
        Defaults to ``False``.
    frozen : bool, optional
        Kept for NEST API compatibility. ``True`` is rejected because NEST
        protects multimeters from freezing.
    name : str, optional
        Module name.

    Notes
    -----
    * In NEST, ``interval``, ``offset``, and ``record_from`` cannot be changed
      after connecting to targets. The same rule is enforced here once
      :meth:`connect` is called or after first data-carrying :meth:`update`.
    * Event buffers mirror NEST's in-memory ``events`` dictionary with keys
      ``times``, ``senders``, and one array per requested recordable.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>> import numpy as np
    >>>
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    ...     neuron = brainpy.state.iaf_psc_delta(1, I_e=500. * u.pA)
    ...     neuron.init_state()
    ...     mm = brainpy.state.multimeter(record_from=['V_m'], interval=0.1 * u.ms)
    ...     for k in range(100):
    ...         with brainstate.environ.context(t=k * 0.1 * u.ms):
    ...             neuron.update()
    ...             vm = float(neuron.V.value[0] / u.mV)
    ...             mm.update({'V_m': np.array([vm])}, senders=np.array([1]))
    ...     events = mm.flush()
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        record_from: Sequence[str] = (),
        interval: ArrayLike = 1.0 * u.ms,
        offset: ArrayLike = 0.0 * u.ms,
        start: ArrayLike = 0.0 * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0.0 * u.ms,
        time_in_steps: bool = False,
        frozen: bool = False,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        if frozen:
            raise ValueError('multimeter cannot be frozen.')

        self._has_targets = False
        self._interval = interval
        self._offset = offset
        self._record_from = ()

        self.start = start
        self.stop = stop
        self.origin = origin
        self.time_in_steps = bool(time_in_steps)

        self._pending: list[_PendingSample] = []
        self.record_from = tuple(record_from)
        self.clear_events()

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        if self._has_targets:
            raise ValueError(
                'The recording interval, the interval offset and the list of '
                'properties to record cannot be changed after the multimeter '
                'has been connected to nodes.'
            )
        self._interval = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        if self._has_targets:
            raise ValueError(
                'The recording interval, the interval offset and the list of '
                'properties to record cannot be changed after the multimeter '
                'has been connected to nodes.'
            )
        self._offset = value

    @property
    def record_from(self):
        return self._record_from

    @record_from.setter
    def record_from(self, value):
        if self._has_targets:
            raise ValueError(
                'The recording interval, the interval offset and the list of '
                'properties to record cannot be changed after the multimeter '
                'has been connected to nodes.'
            )
        self._record_from = tuple(str(v) for v in value)
        self._events_values = {name: [] for name in self._record_from}
        self._pending.clear()

    @property
    def n_events(self) -> int:
        return len(self._events_times)

    @property
    def events(self) -> dict[str, np.ndarray]:
        out = {
            'times': np.asarray(self._events_times, dtype=np.float64),
            'senders': np.asarray(self._events_senders, dtype=np.int64),
        }
        if self.time_in_steps:
            out['offsets'] = np.zeros(out['times'].shape, dtype=np.float64)
        for key in self._record_from:
            out[key] = np.asarray(self._events_values[key], dtype=np.float64)
        return out

    def get(self, key: str = 'events'):
        if key == 'events':
            return self.events
        if key == 'n_events':
            return self.n_events
        raise KeyError(f'Unsupported key "{key}" for multimeter.get().')

    def clear_events(self):
        self._events_times: list[float] = []
        self._events_senders: list[int] = []
        self._events_values = {name: [] for name in self._record_from}

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self.clear_events()
        self._pending.clear()

    def connect(self):
        self._has_targets = True

    def flush(self):
        dt = brainstate.environ.get_dt()
        calib = self._get_step_calibration(dt)
        self._emit_pending(calib)
        return self.events

    def update(
        self,
        data: Mapping[str, ArrayLike] = None,
        senders: ArrayLike = None,
    ):
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        calib = self._get_step_calibration(dt)

        self._emit_pending(calib)

        if data is None:
            return self.events

        self._has_targets = True

        if len(self._record_from) == 0:
            return self.events

        step_now = self._time_to_step(t, calib.dt_ms)
        stamp_step = step_now + 1
        if self._should_sample(stamp_step, calib.interval_steps, calib.offset_steps):
            self._pending.append(self._pack_sample(stamp_step, data, senders))

        return self.events

    @staticmethod
    def _to_ms_scalar(value, name: str, allow_inf: bool = False) -> float:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value / u.ms)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be a scalar time value.')
        val = float(arr[0])
        if (not allow_inf) and (not math.isfinite(val)):
            raise ValueError(f'{name} must be finite.')
        return val

    @classmethod
    def _to_step_count(
        cls,
        value,
        dt_ms: float,
        name: str,
        allow_inf: bool = False,
    ):
        if value is None:
            if allow_inf:
                return math.inf
            raise ValueError(f'{name} cannot be None.')
        ms = cls._to_ms_scalar(value, name=name, allow_inf=allow_inf)
        if math.isinf(ms):
            if allow_inf:
                return math.inf
            raise ValueError(f'{name} must be finite.')
        steps_f = ms / dt_ms
        steps_i = int(np.rint(steps_f))
        if not np.isclose(steps_f, steps_i, atol=1e-12, rtol=1e-12):
            raise ValueError(f'{name} must be a multiple of the simulation resolution.')
        return steps_i

    def _get_step_calibration(self, dt) -> _StepCalibration:
        dt_ms = self._to_ms_scalar(dt, name='dt')
        if dt_ms <= 0.0:
            raise ValueError('Simulation resolution dt must be positive.')

        interval_steps = self._to_step_count(self.interval, dt_ms, 'interval')
        if interval_steps < 1:
            raise ValueError('The sampling interval must be at least as long as the simulation resolution.')

        offset_steps = self._to_step_count(self.offset, dt_ms, 'offset')
        if offset_steps != 0 and offset_steps < 1:
            raise ValueError('The offset for the sampling interval must be at least as long as the simulation resolution.')

        start_steps = self._to_step_count(self.start, dt_ms, 'start')
        stop_value = math.inf if self.stop is None else self.stop
        stop_steps = self._to_step_count(stop_value, dt_ms, 'stop', allow_inf=True)
        if not math.isinf(stop_steps) and stop_steps < start_steps:
            raise ValueError('stop >= start required.')

        origin_steps = self._to_step_count(self.origin, dt_ms, 'origin')
        t_min_steps = origin_steps + start_steps
        t_max_steps = math.inf if math.isinf(stop_steps) else origin_steps + stop_steps

        return _StepCalibration(
            dt_ms=dt_ms,
            interval_steps=interval_steps,
            offset_steps=offset_steps,
            t_min_steps=t_min_steps,
            t_max_steps=t_max_steps,
        )

    def _time_to_step(self, t, dt_ms: float) -> int:
        t_ms = self._to_ms_scalar(t, name='t')
        steps_f = t_ms / dt_ms
        steps_i = int(np.rint(steps_f))
        if not np.isclose(steps_f, steps_i, atol=1e-12, rtol=1e-12):
            raise ValueError('Current simulation time t must be aligned to the simulation grid.')
        return steps_i

    @staticmethod
    def _should_sample(stamp_step: int, interval_steps: int, offset_steps: int) -> bool:
        if offset_steps == 0:
            return (stamp_step % interval_steps) == 0
        if stamp_step < offset_steps:
            return False
        return ((stamp_step - offset_steps) % interval_steps) == 0

    @staticmethod
    def _is_active(stamp_step: int, t_min_steps: int, t_max_steps: float) -> bool:
        if stamp_step <= t_min_steps:
            return False
        if math.isinf(t_max_steps):
            return True
        return stamp_step <= t_max_steps

    @staticmethod
    def _to_float_array(x, name: str) -> np.ndarray:
        if isinstance(x, u.Quantity):
            x = u.get_mantissa(x)
        arr = np.asarray(u.math.asarray(x), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f'Recordable "{name}" must contain at least one value.')
        return arr

    def _pack_sample(
        self,
        stamp_step: int,
        data: Mapping[str, ArrayLike],
        senders: ArrayLike = None,
    ) -> _PendingSample:
        if not isinstance(data, Mapping):
            raise ValueError('data must be a mapping from recordable names to values.')

        values: dict[str, np.ndarray] = {}
        n_items = None
        for key in self._record_from:
            if key not in data:
                raise ValueError(f'Missing recordable "{key}" in data.')
            arr = self._to_float_array(data[key], key)
            if n_items is None:
                n_items = arr.size
            elif arr.size == 1 and n_items > 1:
                arr = np.full((n_items,), arr[0], dtype=np.float64)
            elif arr.size != n_items:
                raise ValueError(f'All recordables must have the same size, got "{key}" with size {arr.size}.')
            values[key] = arr

        if n_items is None:
            n_items = 0

        if senders is None:
            sender_arr = np.ones((n_items,), dtype=np.int64)
        else:
            sender_arr = np.asarray(u.math.asarray(senders), dtype=np.int64).reshape(-1)
            if sender_arr.size == 1 and n_items > 1:
                sender_arr = np.full((n_items,), sender_arr[0], dtype=np.int64)
            elif sender_arr.size != n_items:
                raise ValueError(
                    f'senders size ({sender_arr.size}) does not match recordable size ({n_items}).'
                )

        return _PendingSample(
            stamp_step=stamp_step,
            senders=sender_arr,
            values=values,
        )

    def _emit_pending(self, calib: _StepCalibration):
        if len(self._pending) == 0:
            return

        for sample in self._pending:
            if not self._is_active(sample.stamp_step, calib.t_min_steps, calib.t_max_steps):
                continue

            if self.time_in_steps:
                timestamp = float(sample.stamp_step)
            else:
                timestamp = sample.stamp_step * calib.dt_ms

            n_items = sample.senders.size
            self._events_times.extend([timestamp] * n_items)
            self._events_senders.extend(sample.senders.tolist())
            for key in self._record_from:
                self._events_values[key].extend(sample.values[key].tolist())

        self._pending.clear()
