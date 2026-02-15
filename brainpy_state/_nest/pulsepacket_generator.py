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
from collections import deque
from typing import Sequence

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'pulsepacket_generator',
]


_UNSET = object()


class pulsepacket_generator(brainstate.nn.Dynamics):
    r"""Gaussian pulse-packet spike generator compatible with NEST.

    Description
    -----------

    ``pulsepacket_generator`` re-implements NEST's stimulation device with the
    same name. Each configured pulse center ``t_c`` produces exactly
    ``activity`` spike-time samples

    .. math::

       x \sim \mathcal{N}(t_c, \mathrm{sdev}^2),

    and each sampled time is discretized to the simulation grid and emitted as
    spike multiplicity in the corresponding step.

    NEST update ordering (source-equivalent)
    ----------------------------------------

    This implementation mirrors ``models/pulsepacket_generator.cpp``:

    1. Keep indices ``start_center_idx``/``stop_center_idx`` into sorted
       ``pulse_times`` for a moving window of centers around current time.
    2. At each update step, extend the right edge of that center window while
       ``center_time - t <= tolerance``.
    3. For each newly entered center, sample ``activity`` Gaussian times,
       keep only samples with ``sample_time >= t``, convert them to integer
       steps, and append to a per-generator queue.
    4. Sort each queue.
    5. Emit (pop) all queued spikes whose integer step is in the current
       delivery interval and return per-step multiplicity.

    As in NEST, ``tolerance = sdev * 10`` for ``sdev > 0`` and
    ``tolerance = 1.0 ms`` otherwise.

    Timing semantics
    ----------------

    NEST classifies this model as ``CURRENT_GENERATOR`` in
    ``get_type()``. Therefore activity is evaluated with the
    ``StimulationDevice`` current-generator shift:

    .. math::

       t_{\min} < (n + 2) \le t_{\max},

    where ``n`` is the current simulation step and
    ``t_min = origin + start``, ``t_max = origin + stop`` (in steps).

    This differs from regular spike generators and is intentionally preserved
    for behavioral parity.

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of generator instances. Default: ``1``.
    pulse_times : sequence, optional
        Pulse center times in ms. Values are sorted internally.
        Default: ``None`` (empty).
    activity : ArrayLike, optional
        Number of spikes generated per pulse center (integer, ``>= 0``).
        Default: ``0``.
    sdev : ArrayLike, optional
        Standard deviation of Gaussian pulse jitter in ms (``>= 0``).
        Default: ``0.0 * u.ms``.
    start : ArrayLike, optional
        Activation start time relative to ``origin`` in ms.
        Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation stop time relative to ``origin`` in ms.
        ``None`` means infinity. Default: ``None``.
    origin : ArrayLike, optional
        Time origin for ``start`` and ``stop`` in ms.
        Default: ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed for Gaussian sampling. Default: ``0``.
    sdev_tolerance : float, optional
        Multiplicative factor used in tolerance window
        (NEST default ``10.0``).
    name : str, optional
        Object name.

    Notes
    -----
    - ``set(activity=...)`` and ``set(sdev=...)`` trigger pulse
      re-generation behavior by clearing queued spikes, matching NEST.
    - Stimulation-backend update order in NEST is
      ``[activity, sdev, pulse_times...]`` and is exposed via
      :meth:`set_data_from_stimulation_backend`.

    References
    ----------
    .. [1] NEST source: ``models/pulsepacket_generator.cpp`` and
           ``models/pulsepacket_generator.h``.
    .. [2] NEST source: ``nestkernel/stimulation_device.cpp``.
    .. [3] NEST model docs:
           https://nest-simulator.readthedocs.io/en/stable/models/pulsepacket_generator.html
    """
    __module__ = 'brainpy.state'

    _TICS_PER_MS = 1000.0

    def __init__(
        self,
        in_size: Size = 1,
        pulse_times: Sequence[ArrayLike] | ArrayLike | None = None,
        activity: ArrayLike = 0,
        sdev: ArrayLike = 0. * u.ms,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        sdev_tolerance: float = 10.0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.activity = self._to_scalar_int(activity, name='activity')
        self.sdev = self._to_scalar_time_ms(sdev)
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self.sdev_tolerance = float(sdev_tolerance)
        if self.sdev_tolerance <= 0.0:
            raise ValueError('sdev_tolerance must be positive.')

        self._pulse_times_ms = np.asarray([], dtype=np.float64)
        if pulse_times is not None:
            self._pulse_times_ms = np.sort(self._to_time_array_ms(pulse_times))

        self._validate_parameters(
            activity=self.activity,
            sdev=self.sdev,
            start=self.start,
            stop=self.stop,
        )

        self._num_generators = int(np.prod(self.varshape))
        self._dt_cache_ms = np.nan
        self._dt_tics = 0
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max
        self._tolerance_ms = 1.0

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

    @staticmethod
    def _to_scalar_time_ms(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.ms), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('Time parameters must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_time_array_ms(values: Sequence[ArrayLike] | ArrayLike) -> np.ndarray:
        if not isinstance(values, u.Quantity):
            arr0 = np.asarray(values)
            if arr0.size == 0:
                return np.asarray([], dtype=np.float64)
        if isinstance(values, u.Quantity):
            arr = values.to_decimal(u.ms)
        else:
            arr = u.math.asarray(values, dtype=jnp.float64)
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    @staticmethod
    def _to_scalar_int(value: ArrayLike, *, name: str) -> int:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        scalar = float(arr.reshape(()))
        nearest = np.rint(scalar)
        if not math.isclose(scalar, nearest, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be an integer.')
        return int(nearest)

    @staticmethod
    def _validate_parameters(
        *,
        activity: int,
        sdev: float,
        start: float,
        stop: float,
    ):
        if activity < 0:
            raise ValueError('The activity cannot be negative.')
        if sdev < 0.0:
            raise ValueError('The standard deviation cannot be negative.')
        if stop < start:
            raise ValueError('stop >= start required.')

    @classmethod
    def _ms_to_tics(cls, time_ms: float) -> int:
        # Match NEST Time(ms): static_cast<long>(ms * TICS_PER_MS + 0.5).
        return int(time_ms * cls._TICS_PER_MS + 0.5)

    @staticmethod
    def _assert_grid_time(name: str, time_ms: float, dt_ms: float):
        if not np.isfinite(time_ms):
            return
        ratio = time_ms / dt_ms
        nearest = np.rint(ratio)
        if not math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be a multiple of the simulation resolution.')

    def _dt_ms(self) -> float:
        dt = brainstate.environ.get_dt()
        return self._to_scalar_time_ms(dt)

    def _maybe_dt_ms(self) -> float | None:
        dt = brainstate.environ.get('dt', default=None)
        if dt is None:
            return None
        return self._to_scalar_time_ms(dt)

    def _current_time_ms(self) -> float:
        t = brainstate.environ.get('t', default=0. * u.ms)
        if t is None:
            return 0.0
        return self._to_scalar_time_ms(t)

    def _time_to_step(self, time_ms: float, dt_ms: float) -> int:
        return int(np.rint(time_ms / dt_ms))

    def _time_to_delivery_step(self, time_ms: float) -> int:
        tic = self._ms_to_tics(time_ms)
        if self._dt_tics <= 0:
            return 0
        return int(math.ceil(float(tic) / float(self._dt_tics)))

    def _refresh_runtime_cache(self, dt_ms: float):
        self._assert_grid_time('origin', self.origin, dt_ms)
        self._assert_grid_time('start', self.start, dt_ms)
        self._assert_grid_time('stop', self.stop, dt_ms)

        self._dt_tics = int(np.rint(dt_ms * self._TICS_PER_MS))
        if self._dt_tics <= 0:
            raise ValueError('Simulation resolution must be positive.')

        self._t_min_step = self._time_to_step(self.origin + self.start, dt_ms)
        if np.isfinite(self.stop):
            self._t_max_step = self._time_to_step(self.origin + self.stop, dt_ms)
        else:
            self._t_max_step = np.iinfo(np.int64).max

        if self.sdev > 0.0:
            self._tolerance_ms = self.sdev * self.sdev_tolerance
        else:
            self._tolerance_ms = 1.0

        self._dt_cache_ms = float(dt_ms)

    def _is_active(self, curr_step: int) -> bool:
        shifted_step = curr_step + 2
        return (self._t_min_step < shifted_step) and (shifted_step <= self._t_max_step)

    def _clear_spike_queues(self):
        if hasattr(self, '_spike_queues'):
            for i in range(len(self._spike_queues)):
                self._spike_queues[i].clear()

    def _all_queues_empty(self) -> bool:
        return all(len(q) == 0 for q in self._spike_queues)

    def _pre_run_hook(self, now_ms: float):
        assert self._start_center_idx <= self._stop_center_idx

        self._start_center_idx = 0
        self._stop_center_idx = 0

        now_tic = self._ms_to_tics(now_ms)

        n_centers = self._pulse_times_ms.size
        while self._stop_center_idx < n_centers:
            center_tic = self._ms_to_tics(float(self._pulse_times_ms[self._stop_center_idx]))
            if ((center_tic - now_tic) / self._TICS_PER_MS) > self._tolerance_ms:
                break
            if (abs(center_tic - now_tic) / self._TICS_PER_MS) > self._tolerance_ms:
                self._start_center_idx += 1
            self._stop_center_idx += 1

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

        self._rng = np.random.default_rng(self.rng_seed)
        self._spike_queues = [deque() for _ in range(self._num_generators)]
        self._start_center_idx = 0
        self._stop_center_idx = 0

        self._pre_run_hook(self._current_time_ms())

    def set(
        self,
        *,
        pulse_times: Sequence[ArrayLike] | ArrayLike | object = _UNSET,
        activity: ArrayLike | object = _UNSET,
        sdev: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set NEST-style public parameters."""
        new_activity = (
            self.activity
            if activity is _UNSET
            else self._to_scalar_int(activity, name='activity')
        )
        new_sdev = self.sdev if sdev is _UNSET else self._to_scalar_time_ms(sdev)

        new_start = self.start if start is _UNSET else self._to_scalar_time_ms(start)
        if stop is _UNSET:
            new_stop = self.stop
        elif stop is None:
            new_stop = np.inf
        else:
            new_stop = self._to_scalar_time_ms(stop)
        new_origin = self.origin if origin is _UNSET else self._to_scalar_time_ms(origin)

        self._validate_parameters(
            activity=new_activity,
            sdev=new_sdev,
            start=new_start,
            stop=new_stop,
        )

        need_new_pulse = (new_activity != self.activity) or (not math.isclose(new_sdev, self.sdev, rel_tol=0.0, abs_tol=0.0))

        if pulse_times is _UNSET:
            new_pulse_times = self._pulse_times_ms.copy()
        else:
            new_pulse_times = self._to_time_array_ms(pulse_times)

        if pulse_times is not _UNSET or need_new_pulse:
            new_pulse_times = np.sort(np.asarray(new_pulse_times, dtype=np.float64).reshape(-1))
            self._pulse_times_ms = new_pulse_times
            self._clear_spike_queues()

        self.activity = new_activity
        self.sdev = new_sdev
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

    def get(self) -> dict:
        """Return current public parameters."""
        return {
            'pulse_times': self._pulse_times_ms.tolist(),
            'activity': int(self.activity),
            'sdev': float(self.sdev),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def set_data_from_stimulation_backend(self, input_param: Sequence[float] | np.ndarray):
        """Update from NEST backend order: [activity, sdev, pulse_times...]."""
        data = np.asarray(input_param, dtype=np.float64).reshape(-1)
        if data.size == 0:
            return
        if data.size < 3:
            raise ValueError(
                'The size of the data for pulsepacket_generator must be at least 3 '
                '[activity, sdev, pulse_times...].'
            )
        self.set(
            activity=data[0],
            sdev=data[1] * u.ms,
            pulse_times=data[2:] * u.ms,
        )

    def _generate_new_pulses(self, curr_tic: int):
        if self._start_center_idx >= self._stop_center_idx or self.activity <= 0:
            return

        need_sort = False

        while self._start_center_idx < self._stop_center_idx:
            center = float(self._pulse_times_ms[self._start_center_idx])

            if self.sdev > 0.0:
                sampled = self._rng.normal(
                    loc=center,
                    scale=self.sdev,
                    size=(self._num_generators, self.activity),
                )
            else:
                sampled = np.full(
                    (self._num_generators, self.activity),
                    center,
                    dtype=np.float64,
                )

            for i in range(self._num_generators):
                queue_i = self._spike_queues[i]
                for x in sampled[i]:
                    x_tic = self._ms_to_tics(float(x))
                    if x_tic >= curr_tic:
                        queue_i.append(self._time_to_delivery_step(float(x)))

            need_sort = True
            self._start_center_idx += 1

        if need_sort:
            for i in range(self._num_generators):
                q = self._spike_queues[i]
                if len(q) > 1:
                    self._spike_queues[i] = deque(sorted(q))

    def update(self):
        if not hasattr(self, '_rng'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_runtime_cache(dt_ms)

        curr_t_ms = self._current_time_ms()
        curr_step = self._time_to_step(curr_t_ms, dt_ms)

        if (
            (self._start_center_idx == self._pulse_times_ms.size and self._all_queues_empty())
            or (not self._is_active(curr_step))
        ):
            return jnp.zeros(self.varshape, dtype=jnp.int64)

        curr_tic = self._ms_to_tics(curr_t_ms)

        n_centers = self._pulse_times_ms.size
        while self._stop_center_idx < n_centers:
            center_tic = self._ms_to_tics(float(self._pulse_times_ms[self._stop_center_idx]))
            if ((center_tic - curr_tic) / self._TICS_PER_MS) > self._tolerance_ms:
                break
            self._stop_center_idx += 1

        self._generate_new_pulses(curr_tic)

        step_limit = curr_step + 1
        counts = np.zeros(self._num_generators, dtype=np.int64)
        for i in range(self._num_generators):
            q = self._spike_queues[i]
            n = 0
            while len(q) > 0 and q[0] < step_limit:
                q.popleft()
                n += 1
            counts[i] = n

        return jnp.asarray(counts.reshape(self.varshape), dtype=jnp.int64)
