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
    'poisson_generator',
]


_UNSET = object()


class poisson_generator(brainstate.nn.Dynamics):
    r"""Poisson spike generator compatible with NEST.

    Description
    -----------

    ``poisson_generator`` re-implements NEST's stimulation device of the same
    name. The device emits Poisson-distributed spike multiplicities per
    simulation step:

    .. math::

       k_n \sim \mathrm{Poisson}(\lambda), \quad
       \lambda = r \, \Delta t / 1000,

    where :math:`r` is the configured rate in spikes/s (Hz) and
    :math:`\Delta t` is the simulation step in ms.

    As in NEST, multiplicity values are integer spike counts per step
    (``0, 1, 2, ...``), not binary spike indicators.

    Timing semantics (NEST spike devices)
    -------------------------------------

    The active interval follows NEST ``StimulationDevice::is_active`` for spike
    generators:

    .. math::

       t_{\min} < t \le t_{\max},

    where :math:`t_{\min} = \mathrm{origin} + \mathrm{start}` and
    :math:`t_{\max} = \mathrm{origin} + \mathrm{stop}`.

    Therefore:

    - ``start`` is exclusive (no emission at ``t == origin + start``),
    - ``stop`` is inclusive (emission allowed at ``t == origin + stop``).

    This matches NEST user docs and source implementation in
    ``models/poisson_generator.cpp`` and ``nestkernel/stimulation_device.cpp``.

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of independent output spike trains. Default: ``1``.
    rate : ArrayLike, optional
        Mean firing rate in spikes/s. Must be non-negative. Default:
        ``0.0 * u.Hz``.
    start : ArrayLike, optional
        Activation time relative to ``origin`` in ms. Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation time relative to ``origin`` in ms. ``None`` means
        infinity. Default: ``None``.
    origin : ArrayLike, optional
        Global time offset for ``start`` and ``stop`` in ms. Default:
        ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed for internal Poisson sampling. Default: ``0``.
    name : str, optional
        Object name.

    Notes
    -----
    - Time parameters follow NEST constraints: finite ``origin``, ``start``,
      and ``stop`` must be representable on the simulation grid.
    - A single generator produces independent trains for each element in
      ``in_size``, analogous to NEST's unique train per target behavior.

    References
    ----------
    .. [1] NEST source: ``models/poisson_generator.cpp`` and
           ``models/poisson_generator.h``.
    .. [2] NEST source: ``nestkernel/stimulation_device.h`` and
           ``nestkernel/stimulation_device.cpp``.
    .. [3] NEST model docs:
           https://nest-simulator.readthedocs.io/en/stable/models/poisson_generator.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        if self.rate < 0.0:
            raise ValueError('The rate cannot be negative.')
        if self.stop < self.start:
            raise ValueError('stop >= start required.')

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
            raise ValueError('rate must be scalar.')
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

    def _is_active(self, curr_step: int) -> bool:
        return (self._t_min_step < curr_step) and (curr_step <= self._t_max_step)

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set NEST-style parameters for the generator."""
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_start = self.start if start is _UNSET else self._to_scalar_time_ms(start)
        if stop is _UNSET:
            new_stop = self.stop
        elif stop is None:
            new_stop = np.inf
        else:
            new_stop = self._to_scalar_time_ms(stop)
        new_origin = self.origin if origin is _UNSET else self._to_scalar_time_ms(origin)

        if new_rate < 0.0:
            raise ValueError('The rate cannot be negative.')
        if new_stop < new_start:
            raise ValueError('stop >= start required.')

        self.rate = new_rate
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    def get(self) -> dict:
        """Return current public parameters."""
        return {
            'rate': float(self.rate),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_poisson(self, lam: float) -> jax.Array:
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.poisson(
            subkey,
            lam=jnp.asarray(lam, dtype=jnp.float64),
            shape=self.varshape,
        ).astype(jnp.int64)

    def update(self):
        if not hasattr(self, 'rng_key'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)):
            self._refresh_timing_cache(dt_ms)

        if self.rate <= 0.0:
            return jnp.zeros(self.varshape, dtype=jnp.int64)

        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)
        if self._is_active(curr_step):
            lam = self.rate * dt_ms / 1000.0
            return self._sample_poisson(lam)
        return jnp.zeros(self.varshape, dtype=jnp.int64)
