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
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'ppd_sup_generator',
]


_UNSET = object()


class ppd_sup_generator(brainstate.nn.Dynamics):
    r"""Superposition of Poisson processes with dead time (NEST-compatible).

    Description
    -----------

    ``ppd_sup_generator`` re-implements NEST's stimulation device with the
    same name. For each output train, it simulates the pooled spike train of
    ``n_proc`` independent Poisson processes with absolute dead time.

    Let ``rate`` be the mean rate (Hz) of each component process and
    ``dead_time`` be the absolute dead time (ms). NEST evolves an age
    distribution per output train:

    - ``occ_refractory[a]`` stores how many processes are in dead-time bin
      ``a``, for ``a = 0, ..., floor(dead_time / dt) - 1``.
    - ``occ_active`` stores how many processes are currently available to
      spike.

    At each step, available processes emit with hazard

    .. math::

       h_{\mathrm{step}} = \frac{dt}{1000 / \mathrm{rate} - \mathrm{dead\_time}}.

    If sinusoidal modulation is enabled, NEST scales this hazard as

    .. math::

       h_t = h_{\mathrm{step}}
         \left(1 + A \sin\left(2\pi f t / 1000\right)\right),

    where ``A`` is ``relative_amplitude`` in ``[0, 1]`` and ``f`` is
    ``frequency`` in Hz.

    NEST update ordering (source-equivalent)
    ----------------------------------------

    This implementation follows ``models/ppd_sup_generator.cpp``:

    1. Check activity at the current left step edge ``t``.
    2. Compute current hazard ``h_t`` (including optional sinusoidal factor).
    3. For each output train, draw emitted multiplicity from its current
       active pool:
       - Binomial branch: ``Binomial(occ_active, h_t)``.
       - Poisson approximation branch (NEST heuristic):
         - if ``occ_active >= 100`` and ``h_t <= 0.01``, or
         - if ``occ_active >= 500`` and ``h_t * occ_active <= 0.1``,
         sample ``Poisson(h_t * occ_active)`` and clip to ``occ_active``.
    4. Move emitted processes into dead-time bins via a rotating pointer and
       reactivate processes whose dead time has elapsed.
    5. Return per-step multiplicity per train (int64).

    Timing semantics
    ----------------

    As a spike generator, activity follows NEST ``StimulationDevice``:

    .. math::

       t_{\min} < t \le t_{\max},

    with ``t_min = origin + start`` and ``t_max = origin + stop``.
    Therefore ``start`` is exclusive and ``stop`` is inclusive.

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of independent output spike trains. Default: ``1``.
    rate : ArrayLike, optional
        Mean firing rate of each component process in spikes/s.
        Default: ``0.0 * u.Hz``.
    dead_time : ArrayLike, optional
        Absolute dead time in ms. Must be non-negative. Default: ``0.0 * u.ms``.
    n_proc : ArrayLike, optional
        Number of independent component processes (integer, ``>= 1``).
        Default: ``1``.
    frequency : ArrayLike, optional
        Sinusoidal modulation frequency in Hz. Default: ``0.0 * u.Hz``.
    relative_amplitude : ArrayLike, optional
        Relative modulation amplitude in ``[0, 1]``. Default: ``0.0``.
    start : ArrayLike, optional
        Activation start time relative to ``origin`` (ms).
        Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation stop time relative to ``origin`` (ms).
        ``None`` means infinity. Default: ``None``.
    origin : ArrayLike, optional
        Time origin for ``start`` and ``stop`` in ms.
        Default: ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed for internal random sampling. Default: ``0``.
    name : str, optional
        Object name.

    Notes
    -----
    - Initial occupation matches NEST ``pre_run_hook()``:
      ``floor(rate/1000 * n_proc * dt)`` in each dead-time bin and the
      remainder in ``occ_active``.
    - NEST does not initialize to sinusoidal equilibrium; modulation may show
      initial transients.
    - The stimulation-backend parameter order in NEST is
      ``[dead_time, rate, n_proc, frequency, relative_amplitude]``.

    References
    ----------
    .. [1] NEST source: ``models/ppd_sup_generator.h`` and
           ``models/ppd_sup_generator.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/ppd_sup_generator.html
    .. [3] Deger M, Helias M, Boucsein C, Rotter S (2011).
           Statistical properties of superimposed stationary spike trains.
           Journal of Computational Neuroscience.
           https://doi.org/10.1007/s10827-011-0362-8
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        dead_time: ArrayLike = 0. * u.ms,
        n_proc: ArrayLike = 1,
        frequency: ArrayLike = 0. * u.Hz,
        relative_amplitude: ArrayLike = 0.0,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.dead_time = self._to_scalar_time_ms(dead_time)
        self.n_proc = self._to_scalar_int(n_proc, name='n_proc')
        self.frequency = self._to_scalar_rate_hz(frequency)
        self.relative_amplitude = self._to_scalar_float(
            relative_amplitude,
            name='relative_amplitude',
        )
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate=self.rate,
            dead_time=self.dead_time,
            n_proc=self.n_proc,
            relative_amplitude=self.relative_amplitude,
            start=self.start,
            stop=self.stop,
        )

        self._num_targets = int(np.prod(self.varshape))
        self._hazard_step = 0.0
        self._omega_rad_per_ms = 0.0
        self._num_age_bins = 0
        self._dt_cache_ms = np.nan
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max

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
    def _to_scalar_rate_hz(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.Hz), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('rate must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

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
        rate: float,
        dead_time: float,
        n_proc: int,
        relative_amplitude: float,
        start: float,
        stop: float,
    ):
        if dead_time < 0.0:
            raise ValueError('The dead time cannot be negative.')

        inv_rate = np.inf if rate == 0.0 else (1000.0 / rate)
        if inv_rate <= dead_time:
            raise ValueError('The inverse rate has to be larger than the dead time.')

        if n_proc < 1:
            raise ValueError('The number of component processes cannot be smaller than one')

        if relative_amplitude < 0.0 or relative_amplitude > 1.0:
            raise ValueError('The relative amplitude of the rate modulation must be in [0,1].')

        if stop < start:
            raise ValueError('stop >= start required.')

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

    def _refresh_runtime_cache(self, dt_ms: float):
        self._assert_grid_time('origin', self.origin, dt_ms)
        self._assert_grid_time('start', self.start, dt_ms)
        self._assert_grid_time('stop', self.stop, dt_ms)

        self._t_min_step = self._time_to_step(self.origin + self.start, dt_ms)
        if np.isfinite(self.stop):
            self._t_max_step = self._time_to_step(self.origin + self.stop, dt_ms)
        else:
            self._t_max_step = np.iinfo(np.int64).max

        self._num_age_bins = int(self.dead_time / dt_ms)
        self._omega_rad_per_ms = 2.0 * math.pi * self.frequency / 1000.0
        if self.rate > 0.0:
            self._hazard_step = dt_ms / (1000.0 / self.rate - self.dead_time)
        else:
            self._hazard_step = 0.0
        self._dt_cache_ms = float(dt_ms)

    def _is_active(self, curr_step: int) -> bool:
        return (self._t_min_step < curr_step) and (curr_step <= self._t_max_step)

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)
        else:
            dt_ms = 0.0

        ini_occ_ref = int(self.rate / 1000.0 * self.n_proc * dt_ms)
        ini_occ_act = int(self.n_proc - ini_occ_ref * self._num_age_bins)

        self.occ_refractory = brainstate.ShortTermState(
            np.full(
                (self._num_targets, self._num_age_bins),
                ini_occ_ref,
                dtype=np.int64,
            )
        )
        self.occ_active = brainstate.ShortTermState(
            np.full(self._num_targets, ini_occ_act, dtype=np.int64)
        )
        self.activate = brainstate.ShortTermState(
            np.zeros(self._num_targets, dtype=np.int64)
        )
        self._rng = np.random.default_rng(self.rng_seed)

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        dead_time: ArrayLike | object = _UNSET,
        n_proc: ArrayLike | object = _UNSET,
        frequency: ArrayLike | object = _UNSET,
        relative_amplitude: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set NEST-style public parameters."""
        new_dead_time = (
            self.dead_time if dead_time is _UNSET else self._to_scalar_time_ms(dead_time)
        )
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_n_proc = (
            self.n_proc if n_proc is _UNSET else self._to_scalar_int(n_proc, name='n_proc')
        )
        new_frequency = (
            self.frequency if frequency is _UNSET else self._to_scalar_rate_hz(frequency)
        )
        new_relative_amplitude = (
            self.relative_amplitude
            if relative_amplitude is _UNSET
            else self._to_scalar_float(relative_amplitude, name='relative_amplitude')
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
            rate=new_rate,
            dead_time=new_dead_time,
            n_proc=new_n_proc,
            relative_amplitude=new_relative_amplitude,
            start=new_start,
            stop=new_stop,
        )

        self.dead_time = new_dead_time
        self.rate = new_rate
        self.n_proc = new_n_proc
        self.frequency = new_frequency
        self.relative_amplitude = new_relative_amplitude
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

    def get(self) -> dict:
        """Return current public parameters."""
        return {
            'rate': float(self.rate),
            'dead_time': float(self.dead_time),
            'n_proc': int(self.n_proc),
            'frequency': float(self.frequency),
            'relative_amplitude': float(self.relative_amplitude),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_poisson(self, lam: float) -> int:
        return int(self._rng.poisson(lam))

    def _sample_binomial(self, n: int, p: float) -> int:
        # Clamp only for numerical safety around invalid domain boundaries.
        if p <= 0.0:
            return 0
        if p >= 1.0:
            return int(n)
        return int(self._rng.binomial(n, p))

    def _update_age_distribution_single(
        self,
        occ_ref_row: np.ndarray,
        occ_active: int,
        activate_idx: int,
        hazard_step_t: float,
    ) -> tuple[int, int, int]:
        if occ_active > 0:
            use_poisson_approx = (
                (occ_active >= 100 and hazard_step_t <= 0.01)
                or (occ_active >= 500 and hazard_step_t * occ_active <= 0.1)
            )
            if use_poisson_approx:
                n_spikes = self._sample_poisson(hazard_step_t * occ_active)
                if n_spikes > occ_active:
                    n_spikes = occ_active
            else:
                n_spikes = self._sample_binomial(occ_active, hazard_step_t)
        else:
            n_spikes = 0

        if occ_ref_row.size > 0:
            occ_active = int(occ_active + occ_ref_row[activate_idx] - n_spikes)
            occ_ref_row[activate_idx] = n_spikes
            activate_idx = int((activate_idx + 1) % occ_ref_row.size)

        return int(n_spikes), int(occ_active), int(activate_idx)

    def update(self):
        if not hasattr(self, 'occ_refractory'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_runtime_cache(dt_ms)

        if self.rate <= 0.0 or self._num_targets == 0:
            return jnp.zeros(self.varshape, dtype=jnp.int64)

        curr_t_ms = self._current_time_ms()
        curr_step = self._time_to_step(curr_t_ms, dt_ms)
        if not self._is_active(curr_step):
            return jnp.zeros(self.varshape, dtype=jnp.int64)

        if self.relative_amplitude > 0.0 and self.frequency != 0.0:
            hazard_step_t = self._hazard_step * (
                1.0 + self.relative_amplitude * math.sin(self._omega_rad_per_ms * curr_t_ms)
            )
            if hazard_step_t < 0.0 and hazard_step_t > -1e-15:
                hazard_step_t = 0.0
        else:
            hazard_step_t = self._hazard_step

        occ_ref = np.asarray(self.occ_refractory.value, dtype=np.int64).copy()
        occ_active = np.asarray(self.occ_active.value, dtype=np.int64).copy()
        activate = np.asarray(self.activate.value, dtype=np.int64).copy()
        counts = np.zeros(self._num_targets, dtype=np.int64)

        for idx in range(self._num_targets):
            n_spikes, occ_act_i, activate_i = self._update_age_distribution_single(
                occ_ref[idx],
                int(occ_active[idx]),
                int(activate[idx]),
                hazard_step_t,
            )
            counts[idx] = n_spikes
            occ_active[idx] = occ_act_i
            activate[idx] = activate_i

        self.occ_refractory.value = occ_ref
        self.occ_active.value = occ_active
        self.activate.value = activate
        return jnp.asarray(counts.reshape(self.varshape), dtype=jnp.int64)
