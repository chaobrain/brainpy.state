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
    'sinusoidal_gamma_generator',
]


_UNSET = object()


class sinusoidal_gamma_generator(brainstate.nn.Dynamics):
    r"""Sinusoidally modulated gamma spike generator compatible with NEST.

    Description
    -----------

    ``sinusoidal_gamma_generator`` re-implements NEST's stimulation device of
    the same name. The instantaneous rate (spikes/ms) is

    .. math::

       \lambda(t) = r + a \sin(\omega t + \phi),

    where:

    - :math:`r = \mathrm{rate}/1000`,
    - :math:`a = \mathrm{amplitude}/1000`,
    - :math:`\omega = 2\pi \cdot \mathrm{frequency}/1000`,
    - :math:`\phi = \mathrm{phase} \cdot \pi/180`.

    Spikes are generated from an inhomogeneous gamma renewal process with
    order ``order >= 1``. Let ``t0`` be the most recent spike time (or latest
    parameter-change integration boundary). Define

    .. math::

       \Lambda(t) = \mathrm{order} \int_{t_0}^{t} \lambda(s)\,ds.

    The per-step spike hazard (already multiplied by ``dt``) follows NEST:

    .. math::

       h(t) = dt \cdot \frac{
         \mathrm{order}\,\lambda(t)\,\Lambda(t)^{\mathrm{order}-1} e^{-\Lambda(t)}
       }{
         \Gamma(\mathrm{order}, \Lambda(t))
       }.

    Update Ordering (NEST source order)
    -----------------------------------

    The implementation mirrors ``models/sinusoidal_gamma_generator.cpp``:

    1. Compute the evaluation time at the right edge of the step ``t + dt``.
    2. Compute instantaneous sinusoidal rate at ``t + dt``.
    3. If active and rate is positive, sample spike hazard:
       - ``individual_spike_trains=True``: one hazard draw per output train.
       - ``individual_spike_trains=False``: one shared hazard draw broadcast to
         all outputs.
    4. For each emitted train, reset its renewal state ``t0`` and
       ``Lambda_t0``.
    5. Store recorded rate (spikes/s) corresponding to step-end rate.

    Timing Semantics
    ----------------

    NEST classifies this model as a spike generator
    (``StimulationDevice::Type::SPIKE_GENERATOR``). Activity is therefore
    checked on the current step index ``n`` with

    .. math::

       t_{\min} < n \le t_{\max},

    where ``t_min = origin + start`` and ``t_max = origin + stop`` in steps.

    Piecewise Integral Semantics on Parameter Changes
    -------------------------------------------------

    NEST keeps renewal history consistent when parameters are changed between
    simulation runs by accumulating the old-parameter contribution up to the
    change time. This implementation applies the same piecewise integral
    semantics when :meth:`set` is called after initialization:

    .. math::

       \Lambda(t) =
       \Lambda_{\mathrm{old}}(t_c) +
       \mathrm{order}_{new}\int_{t_c}^{t}\lambda_{new}(s)\,ds.

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of output spike trains. Default: ``1``.
    rate : ArrayLike, optional
        Mean firing rate in spikes/s. Default: ``0.0 * u.Hz``.
    amplitude : ArrayLike, optional
        Firing-rate modulation amplitude in spikes/s. Must satisfy
        ``0 <= amplitude <= rate``. Default: ``0.0 * u.Hz``.
    frequency : ArrayLike, optional
        Modulation frequency in Hz. Default: ``0.0 * u.Hz``.
    phase : ArrayLike, optional
        Modulation phase in degrees. Default: ``0.0``.
    order : ArrayLike, optional
        Gamma order. Must satisfy ``order >= 1``. Default: ``1.0``.
    individual_spike_trains : bool, optional
        If ``True`` (default), independent train per output.
        If ``False``, one shared train is broadcast to all outputs.
    start : ArrayLike, optional
        Activation start time (ms), relative to ``origin``.
        Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation stop time (ms), relative to ``origin``.
        ``None`` means infinity. Default: ``None``.
    origin : ArrayLike, optional
        Time origin (ms) for ``start``/``stop``. Default: ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed for hazard sampling. Default: ``0``.
    name : str, optional
        Object name.

    Notes
    -----
    - This model emits at most one spike per train per simulation step, in
      line with NEST's per-step hazard-threshold implementation.
    - The generator state is revalidated against the simulation grid whenever
      ``dt`` changes.
    - The latest instantaneous rate can be queried with
      :meth:`get_recorded_rate`, matching NEST's ``rate`` recordable.

    References
    ----------
    .. [1] NEST source:
           ``models/sinusoidal_gamma_generator.h`` and
           ``models/sinusoidal_gamma_generator.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/sinusoidal_gamma_generator.html
    .. [3] NEST source:
           ``nestkernel/stimulation_device.cpp``.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        amplitude: ArrayLike = 0. * u.Hz,
        frequency: ArrayLike = 0. * u.Hz,
        phase: ArrayLike = 0.0,
        order: ArrayLike = 1.0,
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
        self.order = self._to_scalar_float(order, name='order')
        self.individual_spike_trains = bool(individual_spike_trains)

        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate_hz=self.rate,
            amplitude_hz=self.amplitude,
            order=self.order,
            start_ms=self.start,
            stop_ms=self.stop,
        )

        self._num_targets = int(np.prod(self.varshape))
        self._num_trains = self._num_targets if self.individual_spike_trains else 1

        self._rate_per_ms = 0.0
        self._amplitude_per_ms = 0.0
        self._om_rad_per_ms = 0.0
        self._phi_rad = 0.0
        self._proc_params = (0.0, 0.0, 1.0, 0.0, 0.0)
        self._proc_params_prev = self._proc_params
        self._refresh_process_parameter_cache()

        self._dt_cache_ms = np.nan
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max
        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

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

    @staticmethod
    def _validate_parameters(
        *,
        rate_hz: float,
        amplitude_hz: float,
        order: float,
        start_ms: float,
        stop_ms: float,
    ):
        if order < 1.0:
            raise ValueError('The gamma order must be at least 1.')
        if not (0.0 <= amplitude_hz <= rate_hz):
            raise ValueError('Rate parameters must fulfill 0 <= amplitude <= rate.')
        if stop_ms < start_ms:
            raise ValueError('stop >= start required.')

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

    def _refresh_process_parameter_cache(self):
        self._rate_per_ms = self.rate / 1000.0
        self._amplitude_per_ms = self.amplitude / 1000.0
        self._om_rad_per_ms = self.frequency * (2.0 * math.pi / 1000.0)
        self._phi_rad = self.phase * (math.pi / 180.0)
        self._proc_params = (
            self._om_rad_per_ms,
            self._phi_rad,
            self.order,
            self._rate_per_ms,
            self._amplitude_per_ms,
        )

    def _is_active(self, curr_step: int) -> bool:
        return (self._t_min_step < curr_step) and (curr_step <= self._t_max_step)

    @staticmethod
    def _delta_lambda(params: tuple[float, float, float, float, float], t_a, t_b):
        om, phi, order, rate, amplitude = params
        t_a_arr = np.asarray(t_a, dtype=np.float64)
        if t_a_arr.ndim == 0:
            if float(t_a_arr) == float(t_b):
                return np.asarray(0.0, dtype=np.float64)
        elif np.all(t_a_arr == float(t_b)):
            return np.zeros_like(t_a_arr, dtype=np.float64)

        delta = order * rate * (t_b - t_a_arr)
        if abs(amplitude) > 0.0 and abs(om) > 0.0:
            delta += -order * amplitude / om * (
                np.cos(om * t_b + phi) - np.cos(om * t_a_arr + phi)
            )
        return delta

    def _accumulate_lambda_to_time(self, t_ms: float):
        if self._num_trains == 0:
            return
        t0 = np.asarray(self.t0_ms.value, dtype=np.float64).reshape(-1).copy()
        lam0 = np.asarray(self.Lambda_t0.value, dtype=np.float64).reshape(-1).copy()

        lam0 += np.asarray(self._delta_lambda(self._proc_params_prev, t0, t_ms), dtype=np.float64)
        t0.fill(t_ms)

        self.t0_ms.value = t0
        self.Lambda_t0.value = lam0

    def _resize_train_state(self, now_ms: float, new_num_trains: int):
        old_t0 = np.asarray(self.t0_ms.value, dtype=np.float64).reshape(-1)
        old_lam = np.asarray(self.Lambda_t0.value, dtype=np.float64).reshape(-1)
        old_n = old_t0.size

        if new_num_trains == old_n:
            return
        if new_num_trains < old_n:
            self.t0_ms.value = old_t0[:new_num_trains].copy()
            self.Lambda_t0.value = old_lam[:new_num_trains].copy()
            return

        add_n = new_num_trains - old_n
        self.t0_ms.value = np.concatenate(
            [old_t0, np.full(add_n, now_ms, dtype=np.float64)]
        )
        self.Lambda_t0.value = np.concatenate(
            [old_lam, np.zeros(add_n, dtype=np.float64)]
        )

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))

        curr_t_ms = self._current_time_ms()
        self.t0_ms = brainstate.ShortTermState(
            np.full(self._num_trains, curr_t_ms, dtype=np.float64)
        )
        self.Lambda_t0 = brainstate.ShortTermState(
            np.zeros(self._num_trains, dtype=np.float64)
        )
        self._recorded_rate_hz = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))
        self._proc_params_prev = self._proc_params

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        amplitude: ArrayLike | object = _UNSET,
        frequency: ArrayLike | object = _UNSET,
        phase: ArrayLike | object = _UNSET,
        order: ArrayLike | object = _UNSET,
        individual_spike_trains: bool | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set NEST-style public parameters."""
        now_ms = self._current_time_ms() if hasattr(self, 't0_ms') else 0.0
        if hasattr(self, 't0_ms'):
            self._accumulate_lambda_to_time(now_ms)

        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_amplitude = (
            self.amplitude if amplitude is _UNSET else self._to_scalar_rate_hz(amplitude)
        )
        new_frequency = (
            self.frequency if frequency is _UNSET else self._to_scalar_rate_hz(frequency)
        )
        new_phase = self.phase if phase is _UNSET else self._to_scalar_float(phase, name='phase')
        new_order = self.order if order is _UNSET else self._to_scalar_float(order, name='order')
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

        self._validate_parameters(
            rate_hz=new_rate,
            amplitude_hz=new_amplitude,
            order=new_order,
            start_ms=new_start,
            stop_ms=new_stop,
        )

        self.rate = new_rate
        self.amplitude = new_amplitude
        self.frequency = new_frequency
        self.phase = new_phase
        self.order = new_order
        self.individual_spike_trains = new_individual
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        self._num_trains = self._num_targets if self.individual_spike_trains else 1
        self._refresh_process_parameter_cache()

        if hasattr(self, 't0_ms'):
            self._resize_train_state(now_ms, self._num_trains)
            self._proc_params_prev = self._proc_params

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    def get(self) -> dict:
        """Return current public parameters."""
        return {
            'rate': float(self.rate),
            'frequency': float(self.frequency),
            'phase': float(self.phase),
            'amplitude': float(self.amplitude),
            'order': float(self.order),
            'individual_spike_trains': bool(self.individual_spike_trains),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def get_recorded_rate(self) -> float:
        """Return the latest step-end instantaneous rate in spikes/s."""
        if not hasattr(self, '_recorded_rate_hz'):
            return 0.0
        return float(np.asarray(self._recorded_rate_hz.value, dtype=np.float64).reshape(()))

    def _sample_uniform(self, shape=()):
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.uniform(subkey, shape=shape, dtype=jnp.float64)

    def _compute_hazard(self, lambda_val: np.ndarray, rate_per_ms: float, dt_ms: float) -> np.ndarray:
        hazard = np.zeros_like(lambda_val, dtype=np.float64)

        # Guard tiny negative values caused by floating-point roundoff only.
        tiny_neg = np.logical_and(lambda_val < 0.0, lambda_val > -1e-15)
        if np.any(tiny_neg):
            lambda_val = lambda_val.copy()
            lambda_val[tiny_neg] = 0.0

        valid = lambda_val >= 0.0
        if not np.any(valid):
            return hazard

        lam = lambda_val[valid]
        q = np.asarray(
            jax.lax.igammac(
                jnp.asarray(self.order, dtype=jnp.float64),
                jnp.asarray(lam, dtype=jnp.float64),
            ),
            dtype=np.float64,
        )
        denom = math.gamma(self.order) * q
        numer = (
            dt_ms
            * self.order
            * rate_per_ms
            * np.power(lam, self.order - 1.0)
            * np.exp(-lam)
        )
        hazard_valid = np.divide(
            numer,
            denom,
            out=np.zeros_like(numer, dtype=np.float64),
            where=denom > 0.0,
        )
        hazard[valid] = hazard_valid
        return hazard

    def update(self):
        if not hasattr(self, 'rng_key'):
            self.init_state()

        dt_ms = self._dt_ms()
        curr_t_ms = self._current_time_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_timing_cache(dt_ms)

        curr_step = self._time_to_step(curr_t_ms, dt_ms)
        t_eval_ms = (curr_step + 1) * dt_ms

        rate_per_ms = self._rate_per_ms + self._amplitude_per_ms * math.sin(
            self._om_rad_per_ms * t_eval_ms + self._phi_rad
        )
        self._recorded_rate_hz.value = jnp.asarray(
            rate_per_ms * 1000.0,
            dtype=jnp.float64,
        )

        if (
            self._num_trains == 0
            or rate_per_ms <= 0.0
            or (not self._is_active(curr_step))
        ):
            return jnp.zeros(self.varshape, dtype=jnp.int64)

        t0 = np.asarray(self.t0_ms.value, dtype=np.float64).reshape(-1).copy()
        lam0 = np.asarray(self.Lambda_t0.value, dtype=np.float64).reshape(-1).copy()
        lambda_eval = lam0 + np.asarray(
            self._delta_lambda(self._proc_params, t0, t_eval_ms),
            dtype=np.float64,
        )

        hazard = self._compute_hazard(lambda_eval, rate_per_ms, dt_ms)

        if self.individual_spike_trains:
            draws = np.asarray(
                self._sample_uniform(shape=(self._num_trains,)),
                dtype=np.float64,
            )
            spikes = draws < hazard
            if np.any(spikes):
                t0[spikes] = t_eval_ms
                lam0[spikes] = 0.0
            self.t0_ms.value = t0
            self.Lambda_t0.value = lam0
            return jnp.asarray(spikes.reshape(self.varshape), dtype=jnp.int64)

        draw = float(np.asarray(self._sample_uniform(shape=()), dtype=np.float64).reshape(()))
        spike = int(draw < float(hazard[0]))
        if spike:
            t0[0] = t_eval_ms
            lam0[0] = 0.0
            self.t0_ms.value = t0
            self.Lambda_t0.value = lam0
        return jnp.full(self.varshape, spike, dtype=jnp.int64)
