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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'sinusoidal_poisson_generator',
]


_UNSET = object()


class sinusoidal_poisson_generator(brainstate.nn.Dynamics):
    r"""Sinusoidally modulated Poisson spike generator compatible with NEST.

    Description
    -----------

    ``sinusoidal_poisson_generator`` re-implements NEST's stimulation device
    of the same name. The instantaneous rate is

    .. math::

       f(t) = \max\left(
         0,\ r + a \sin\left( 2\pi f_{\mathrm{mod}} t / 1000 + \phi \right)
       \right),

    where:

    - ``r`` is ``rate`` (spikes/s),
    - ``a`` is ``amplitude`` (spikes/s),
    - ``f_mod`` is ``frequency`` (Hz),
    - ``phi`` is ``phase`` (deg, internally converted to radians),
    - ``t`` is simulation time in ms.

    At each simulation step, output multiplicities are sampled from

    .. math::

       k_n \sim \mathrm{Poisson}(f_n \Delta t / 1000).

    Update Ordering (NEST source order)
    -----------------------------------

    The internal two-component oscillator state is updated exactly in the
    order used by NEST ``models/sinusoidal_poisson_generator.cpp``:

    1. Start from the DC component ``rate``.
    2. Rotate oscillator state ``(y_0, y_1)`` by one step.
    3. Add the rotated ``y_1`` to obtain instantaneous rate.
    4. Clamp rate at zero.
    5. Sample Poisson multiplicities if active.

    The per-step recorded ``rate`` value in NEST corresponds to this updated
    post-rotation rate. This implementation exposes it via
    :meth:`get_recorded_rate`.

    Timing Semantics
    ----------------

    NEST currently classifies this model as ``CURRENT_GENERATOR`` in
    ``get_type()``. Consequently, activity is evaluated with a two-step shift
    in ``StimulationDevice::is_active``:

    .. math::

       t_{\min} < (n + 2) \le t_{\max},

    where ``n`` is current simulation step index and
    ``t_min = origin + start``, ``t_max = origin + stop`` (in steps).

    This differs from regular spike generators and is intentionally replicated
    here to match NEST behavior.

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of output spike trains. Default: ``1``.
    rate : ArrayLike, optional
        Mean firing rate in spikes/s. Default: ``0.0 * u.Hz``.
    amplitude : ArrayLike, optional
        Sinusoidal modulation amplitude in spikes/s. Default: ``0.0 * u.Hz``.
    frequency : ArrayLike, optional
        Modulation frequency in Hz. Default: ``0.0 * u.Hz``.
    phase : ArrayLike, optional
        Modulation phase in degrees. Default: ``0.0``.
    individual_spike_trains : bool, optional
        If ``True`` (default), output trains are sampled independently.
        If ``False``, one sampled multiplicity is broadcast to all outputs.
    start : ArrayLike, optional
        Activation start time (ms), relative to ``origin``.
        Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation stop time (ms), relative to ``origin``.
        ``None`` means infinity. Default: ``None``.
    origin : ArrayLike, optional
        Time origin (ms) for start/stop. Default: ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed for Poisson sampling. Default: ``0``.
    name : str, optional
        Object name.

    Notes
    -----
    - Time parameters are validated on the simulation grid when ``dt`` is
      available, matching repository conventions used by other NEST-compatible
      generators.
    - The oscillator state is re-initialized from absolute simulation time
      whenever the simulation resolution changes, matching NEST pre-run
      calibration behavior.

    References
    ----------
    .. [1] NEST source:
           ``models/sinusoidal_poisson_generator.h`` and
           ``models/sinusoidal_poisson_generator.cpp``.
    .. [2] NEST source:
           ``nestkernel/stimulation_device.cpp``.
    .. [3] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/sinusoidal_poisson_generator.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        amplitude: ArrayLike = 0. * u.Hz,
        frequency: ArrayLike = 0. * u.Hz,
        phase: ArrayLike = 0.0,
        individual_spike_trains: bool = True,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.amplitude = self._to_scalar_rate_hz(amplitude)
        self.frequency = self._to_scalar_rate_hz(frequency)
        self.phase = self._to_scalar_float(phase, name='phase')
        self.individual_spike_trains = bool(individual_spike_trains)

        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        if self.stop < self.start:
            raise ValueError('stop >= start required.')

        self._rate_per_ms = self.rate / 1000.0
        self._amplitude_per_ms = self.amplitude / 1000.0
        self._om_rad_per_ms = self.frequency * (2.0 * math.pi / 1000.0)
        self._phi_rad = self.phase * (math.pi / 180.0)

        self._dt_cache_ms = np.nan
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max
        self._sin_step = 0.0
        self._cos_step = 1.0

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)
            self._refresh_step_rotation_cache(dt_ms)

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
    def _to_scalar_rate_hz(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.Hz), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('Rate parameters must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _time_to_step(time_ms: float, dt_ms: float) -> int:
        return int(np.rint(time_ms / dt_ms))

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

    def _refresh_timing_cache(self, dt_ms: float):
        self._assert_grid_time('origin', self.origin, dt_ms)
        self._assert_grid_time('start', self.start, dt_ms)
        self._assert_grid_time('stop', self.stop, dt_ms)

        self._t_min_step = self._time_to_step(self.origin + self.start, dt_ms)
        if np.isfinite(self.stop):
            self._t_max_step = self._time_to_step(self.origin + self.stop, dt_ms)
        else:
            self._t_max_step = np.iinfo(np.int64).max
        self._dt_cache_ms = float(dt_ms)

    def _refresh_step_rotation_cache(self, dt_ms: float):
        self._sin_step = math.sin(dt_ms * self._om_rad_per_ms)
        self._cos_step = math.cos(dt_ms * self._om_rad_per_ms)

    def _reset_oscillator_state(self, t_ms: float):
        y0 = self._amplitude_per_ms * math.cos(self._om_rad_per_ms * t_ms + self._phi_rad)
        y1 = self._amplitude_per_ms * math.sin(self._om_rad_per_ms * t_ms + self._phi_rad)
        self.y_0.value = jnp.asarray(y0, dtype=jnp.float64)
        self.y_1.value = jnp.asarray(y1, dtype=jnp.float64)
        self._recorded_rate_hz.value = jnp.asarray(0.0, dtype=jnp.float64)

    def _is_active(self, curr_step: int) -> bool:
        # Match NEST's current-generator activity handling for this model:
        # StimulationDevice::is_active uses step+2 for CURRENT_GENERATOR.
        shifted_step = curr_step + 2
        return (self._t_min_step < shifted_step) and (shifted_step <= self._t_max_step)

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))
        self.y_0 = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))
        self.y_1 = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))
        self._recorded_rate_hz = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)
            self._refresh_step_rotation_cache(dt_ms)
            self._reset_oscillator_state(self._current_time_ms())

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        amplitude: ArrayLike | object = _UNSET,
        frequency: ArrayLike | object = _UNSET,
        phase: ArrayLike | object = _UNSET,
        individual_spike_trains: bool | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set NEST-style public parameters."""
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_amplitude = (
            self.amplitude if amplitude is _UNSET else self._to_scalar_rate_hz(amplitude)
        )
        new_frequency = (
            self.frequency if frequency is _UNSET else self._to_scalar_rate_hz(frequency)
        )
        new_phase = self.phase if phase is _UNSET else self._to_scalar_float(phase, name='phase')
        new_individual = (
            self.individual_spike_trains
            if individual_spike_trains is _UNSET
            else bool(individual_spike_trains)
        )

        new_start = self.start if start is _UNSET else self._to_scalar_time_ms(start)
        if stop is _UNSET:
            new_stop = self.stop
        elif stop is None:
            new_stop = np.inf
        else:
            new_stop = self._to_scalar_time_ms(stop)
        new_origin = self.origin if origin is _UNSET else self._to_scalar_time_ms(origin)

        if new_stop < new_start:
            raise ValueError('stop >= start required.')

        self.rate = new_rate
        self.amplitude = new_amplitude
        self.frequency = new_frequency
        self.phase = new_phase
        self.individual_spike_trains = new_individual
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        self._rate_per_ms = self.rate / 1000.0
        self._amplitude_per_ms = self.amplitude / 1000.0
        self._om_rad_per_ms = self.frequency * (2.0 * math.pi / 1000.0)
        self._phi_rad = self.phase * (math.pi / 180.0)

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)
            self._refresh_step_rotation_cache(dt_ms)
            if hasattr(self, 'y_0'):
                self._reset_oscillator_state(self._current_time_ms())

    def get(self) -> dict:
        """Return current parameters and oscillator state in public units."""
        y0 = 0.0
        y1 = 0.0
        if hasattr(self, 'y_0'):
            y0 = float(np.asarray(self.y_0.value, dtype=np.float64).reshape(()))
            y1 = float(np.asarray(self.y_1.value, dtype=np.float64).reshape(()))

        return {
            'rate': float(self.rate),
            'frequency': float(self.frequency),
            'phase': float(self.phase),
            'amplitude': float(self.amplitude),
            'individual_spike_trains': bool(self.individual_spike_trains),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
            'y_0': y0,
            'y_1': y1,
        }

    def get_recorded_rate(self) -> float:
        """Return the latest post-update instantaneous rate in spikes/s."""
        if not hasattr(self, '_recorded_rate_hz'):
            return 0.0
        return float(np.asarray(self._recorded_rate_hz.value, dtype=np.float64).reshape(()))

    def _sample_poisson_individual(self, lam: float) -> jax.Array:
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.poisson(
            subkey,
            lam=jnp.asarray(lam, dtype=jnp.float64),
            shape=self.varshape,
        ).astype(jnp.int64)

    def _sample_poisson_shared(self, lam: float) -> int:
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        sample = jax.random.poisson(
            subkey,
            lam=jnp.asarray(lam, dtype=jnp.float64),
            shape=(),
        ).astype(jnp.int64)
        return int(np.asarray(sample, dtype=np.int64).reshape(()))

    def update(self):
        if not hasattr(self, 'rng_key'):
            self.init_state()

        dt_ms = self._dt_ms()
        curr_t_ms = self._current_time_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_timing_cache(dt_ms)
            self._refresh_step_rotation_cache(dt_ms)
            self._reset_oscillator_state(curr_t_ms)

        curr_step = self._time_to_step(curr_t_ms, dt_ms)

        # Update oscillator blocks in NEST ordering.
        y0 = float(np.asarray(self.y_0.value, dtype=np.float64).reshape(()))
        y1 = float(np.asarray(self.y_1.value, dtype=np.float64).reshape(()))

        rate_per_ms = self._rate_per_ms
        new_y0 = self._cos_step * y0 - self._sin_step * y1
        y1 = self._sin_step * y0 + self._cos_step * y1
        y0 = new_y0
        rate_per_ms += y1
        if rate_per_ms < 0.0:
            rate_per_ms = 0.0

        self.y_0.value = jnp.asarray(y0, dtype=jnp.float64)
        self.y_1.value = jnp.asarray(y1, dtype=jnp.float64)
        self._recorded_rate_hz.value = jnp.asarray(rate_per_ms * 1000.0, dtype=jnp.float64)

        if rate_per_ms > 0.0 and self._is_active(curr_step):
            lam = rate_per_ms * dt_ms
            if self.individual_spike_trains:
                return self._sample_poisson_individual(lam)

            n_spikes = self._sample_poisson_shared(lam)
            return jnp.full(self.varshape, n_spikes, dtype=jnp.int64)

        return jnp.zeros(self.varshape, dtype=jnp.int64)
