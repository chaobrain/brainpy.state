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
from typing import Sequence

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'inhomogeneous_poisson_generator',
]


_UNSET = object()


class inhomogeneous_poisson_generator(brainstate.nn.Dynamics):
    r"""Inhomogeneous Poisson spike generator with piecewise-constant rate.

    Description
    -----------

    ``inhomogeneous_poisson_generator`` re-implements the NEST stimulation
    device of the same name. It produces Poisson-distributed spike counts with
    a piecewise-constant rate schedule specified by ``rate_times`` and
    ``rate_values``.

    A schedule entry ``(t_k, r_k)`` means that the instantaneous rate becomes
    ``r_k`` spikes/s at time ``t_k`` (relative to global time, in ms). As in
    NEST, the internal rate switch is applied one simulation step ahead so that
    causality and delivery ordering match stimulation-device semantics.

    Update ordering (NEST semantics)
    --------------------------------

    At each simulation step with index ``n``:

    1. Skip all schedule entries whose aligned step is in the past
       (``step <= n``).
    2. If the next schedule entry occurs at ``n + 1``, switch the internal
       rate immediately for this update.
    3. If the device is active and the rate is positive, sample multiplicity
       ``k ~ Poisson(rate * dt)`` and return ``k``.

    The activity window follows NEST spike-device convention:

    .. math::

       t_{\min} < t \le t_{\max},

    where :math:`t_{\min} = origin + start`, :math:`t_{\max} = origin + stop`.
    Thus ``start`` is exclusive and ``stop`` is inclusive in terms of spike
    timestamps.

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of independent output spike trains. Default: ``1``.
    rate_times : sequence, optional
        Times (ms) at which rates change. Must be strictly increasing after
        grid alignment. Default: ``None`` (no scheduled rate changes).
    rate_values : sequence, optional
        Rate values (spikes/s) paired with ``rate_times``. Must be same length
        as ``rate_times``. Default: ``None``.
    allow_offgrid_times : bool, optional
        If ``False`` (default), non-grid times raise an error. If ``True``,
        non-grid times are aligned upward to the end of the containing step
        (while near-grid times are rounded to that grid point), matching NEST's
        off-grid policy.
    start : ArrayLike, optional
        Activation start time relative to ``origin``. Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation time relative to ``origin``. ``None`` means infinity.
        Default: ``None``.
    origin : ArrayLike, optional
        Global time offset for ``start`` and ``stop``. Default: ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed used for Poisson sampling. Default: ``0``.
    name : str, optional
        Object name.

    Notes
    -----
    - Output entries are integer spike multiplicities per step, not binary
      spikes.
    - ``rate_times`` and ``rate_values`` follow NEST setter constraints:
      both must be set together, lengths must match, and aligned rate times
      must be strictly increasing.
    - Rate times must be strictly in the future relative to current simulation
      time when set.

    References
    ----------
    .. [1] NEST Simulator model: ``inhomogeneous_poisson_generator``.
           https://nest-simulator.readthedocs.io/en/stable/models/inhomogeneous_poisson_generator.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate_times: Sequence[ArrayLike] | ArrayLike | None = None,
        rate_values: Sequence[ArrayLike] | ArrayLike | None = None,
        allow_offgrid_times: bool = False,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.allow_offgrid_times = bool(allow_offgrid_times)
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        if self.stop < self.start:
            raise ValueError('stop must be greater than or equal to start.')

        self._rate_times_ms = np.asarray([], dtype=np.float64)
        self._rate_values_hz = np.asarray([], dtype=np.float64)
        self._rate_steps = np.asarray([], dtype=np.int64)

        if (rate_times is None) ^ (rate_values is None):
            raise ValueError('Rate times and values must be reset together.')
        if rate_times is not None:
            self.set(rate_times=rate_times, rate_values=rate_values)

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
    def _to_rate_array_hz(values: Sequence[ArrayLike] | ArrayLike) -> np.ndarray:
        if not isinstance(values, u.Quantity):
            arr0 = np.asarray(values)
            if arr0.size == 0:
                return np.asarray([], dtype=np.float64)
        if isinstance(values, u.Quantity):
            arr = values.to_decimal(u.Hz)
        else:
            arr = u.math.asarray(values, dtype=jnp.float64)
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    @staticmethod
    def _array_to_public(value: np.ndarray):
        if value.size == 1:
            return float(value[0])
        return value.tolist()

    @staticmethod
    def _time_to_step(time_ms: float, dt_ms: float) -> int:
        return int(np.rint(time_ms / dt_ms))

    def _dt_ms(self) -> float:
        dt = brainstate.environ.get_dt()
        return self._to_scalar_time_ms(dt)

    def _current_time_ms(self) -> float:
        t = brainstate.environ.get('t', default=0. * u.ms)
        if t is None:
            return 0.0
        return self._to_scalar_time_ms(t)

    def _align_rate_time_to_grid(self, time_ms: float, dt_ms: float) -> tuple[int, float]:
        ratio = time_ms / dt_ms
        nearest = np.rint(ratio)

        if math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-12):
            step = int(nearest)
        elif self.allow_offgrid_times:
            step = int(math.ceil(ratio - 1e-12))
        else:
            raise ValueError(
                f'inhomogeneous_poisson_generator: Time point {time_ms} '
                f'is not representable in current resolution.'
            )

        return step, float(step) * dt_ms

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self._rate_idx = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int64))
        self._rate_hz = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))

    def set(
        self,
        *,
        rate_times: Sequence[ArrayLike] | ArrayLike | object = _UNSET,
        rate_values: Sequence[ArrayLike] | ArrayLike | object = _UNSET,
        allow_offgrid_times: bool | object = _UNSET,
    ):
        """Set NEST-style model parameters.

        This mirrors NEST setter constraints for schedule and off-grid options.
        """
        times_given = rate_times is not _UNSET
        rates_given = rate_values is not _UNSET

        if allow_offgrid_times is not _UNSET:
            new_flag = bool(allow_offgrid_times)
            if (
                new_flag != self.allow_offgrid_times
                and not (times_given or self._rate_times_ms.size == 0)
            ):
                raise ValueError(
                    'Option can only be set together with rate times '
                    'or if no rate times have been set.'
                )
            self.allow_offgrid_times = new_flag

        if times_given ^ rates_given:
            raise ValueError('Rate times and values must be reset together.')

        if not (times_given or rates_given):
            return

        times_ms = self._to_time_array_ms(rate_times)
        values_hz = self._to_rate_array_hz(rate_values)

        if times_ms.size != values_hz.size:
            raise ValueError('Rate times and values have to be the same size.')

        if times_ms.size == 0:
            self._rate_times_ms = np.asarray([], dtype=np.float64)
            self._rate_values_hz = np.asarray([], dtype=np.float64)
            self._rate_steps = np.asarray([], dtype=np.int64)
            if hasattr(self, '_rate_idx'):
                self._rate_idx.value = jnp.asarray(0, dtype=jnp.int64)
            return

        dt_ms = self._dt_ms()
        now_ms = self._current_time_ms()

        aligned_times = np.empty_like(times_ms, dtype=np.float64)
        aligned_steps = np.empty_like(times_ms, dtype=np.int64)

        for i, t_ms in enumerate(times_ms):
            if t_ms <= now_ms:
                raise ValueError('Time points must lie strictly in the future.')

            step, aligned_ms = self._align_rate_time_to_grid(float(t_ms), dt_ms)
            aligned_steps[i] = step
            aligned_times[i] = aligned_ms

            if i > 0 and aligned_steps[i - 1] >= aligned_steps[i]:
                raise ValueError('Rate times must be strictly increasing.')

        self._rate_times_ms = aligned_times
        self._rate_values_hz = values_hz
        self._rate_steps = aligned_steps

        # Match NEST setter semantics: schedule index is reset on new data.
        if hasattr(self, '_rate_idx'):
            self._rate_idx.value = jnp.asarray(0, dtype=jnp.int64)

    def get(self) -> dict:
        """Return NEST-style public parameters for inspection/tests."""
        return {
            'rate_times': self._array_to_public(self._rate_times_ms),
            'rate_values': self._array_to_public(self._rate_values_hz),
            'allow_offgrid_times': bool(self.allow_offgrid_times),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _is_active(self, curr_step: int, dt_ms: float) -> bool:
        t_ms = curr_step * dt_ms
        t_min = self.origin + self.start
        t_max = self.origin + self.stop
        return (t_min < t_ms) and (t_ms <= t_max)

    def _sample_poisson(self, lam: float) -> jax.Array:
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.poisson(
            subkey,
            lam=jnp.asarray(lam, dtype=jnp.float64),
            shape=self.varshape,
        ).astype(jnp.int64)

    def update(self):
        if not hasattr(self, '_rate_idx'):
            self.init_state()

        dt_ms = self._dt_ms()
        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)

        idx = int(self._rate_idx.value)
        while idx < self._rate_steps.size and int(self._rate_steps[idx]) <= curr_step:
            idx += 1

        rate_hz = float(self._rate_hz.value)
        if idx < self._rate_steps.size and curr_step + 1 == int(self._rate_steps[idx]):
            rate_hz = float(self._rate_values_hz[idx])
            idx += 1

        self._rate_idx.value = jnp.asarray(idx, dtype=jnp.int64)
        self._rate_hz.value = jnp.asarray(rate_hz, dtype=jnp.float64)

        if rate_hz > 0.0 and self._is_active(curr_step, dt_ms):
            lam = rate_hz * dt_ms / 1000.0
            return self._sample_poisson(lam)

        return jnp.zeros(self.varshape, dtype=jnp.int64)
