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
    'mip_generator',
]


_UNSET = object()


class mip_generator(brainstate.nn.Dynamics):
    r"""Correlated spike trains from a Multiple Interaction Process (MIP).

    Description
    -----------

    ``mip_generator`` re-implements NEST's stimulation device of the same
    name. It generates child spike trains via a shared Poisson parent process:

    1. Draw the number of parent spikes in a step from
       :math:`N \sim \mathrm{Poisson}(\lambda)`, with
       :math:`\lambda = r \, \Delta t / 1000`.
    2. For each output train, copy each of the ``N`` parent spikes with
       probability ``p_copy``.

    Consequently, each child train has mean rate ``p_copy * rate`` and
    theoretical pairwise count correlation ``p_copy``.

    NEST update ordering (source-equivalent)
    ----------------------------------------

    This implementation mirrors ``models/mip_generator.cpp``:

    1. Evaluate activity for the current simulation step.
    2. Draw parent multiplicity from the parent Poisson process.
    3. For each target/output train, run the copy process by Bernoulli trials
       per parent spike and return resulting multiplicity.

    NEST uses an explicit Bernoulli loop (rather than direct binomial
    sampling) in ``event_hook()``; the same sampling order is preserved here.

    Timing semantics
    ----------------

    As a NEST spike stimulation device, activity follows

    .. math::

       t_{\min} < t \le t_{\max},

    where :math:`t_{\min} = \mathrm{origin} + \mathrm{start}` and
    :math:`t_{\max} = \mathrm{origin} + \mathrm{stop}`.
    Therefore ``start`` is exclusive and ``stop`` is inclusive.

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of output child spike trains. Default: ``1``.
    rate : ArrayLike, optional
        Parent process rate in spikes/s (Hz). Must be non-negative.
        Default: ``0.0 * u.Hz``.
    p_copy : ArrayLike, optional
        Per-spike copy probability into each child process. Must lie in
        ``[0, 1]``. Default: ``1.0``.
    start : ArrayLike, optional
        Activation time relative to ``origin`` in ms.
        Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation time relative to ``origin`` in ms. ``None`` means
        infinity. Default: ``None``.
    origin : ArrayLike, optional
        Time offset for ``start`` and ``stop`` in ms.
        Default: ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed for internal random streams. Default: ``0``.
    name : str, optional
        Object name.

    Notes
    -----
    - Output values are integer per-step spike multiplicities (``0, 1, ...``),
      matching NEST ``SpikeEvent`` multiplicity semantics.
    - One RNG stream is used for parent Poisson draws and one for child-copy
      Bernoulli draws, matching NEST's separate synced/specific RNG usage.

    References
    ----------
    .. [1] NEST source: ``models/mip_generator.h`` and
           ``models/mip_generator.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/mip_generator.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        p_copy: ArrayLike = 1.0,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.p_copy = self._to_scalar_float(p_copy, name='p_copy')
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate=self.rate,
            p_copy=self.p_copy,
            start=self.start,
            stop=self.stop,
        )

        self._num_targets = int(np.prod(self.varshape))
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
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _validate_parameters(
        *,
        rate: float,
        p_copy: float,
        start: float,
        stop: float,
    ):
        if rate < 0.0:
            raise ValueError('Rate must be non-negative.')
        if p_copy < 0.0 or p_copy > 1.0:
            raise ValueError('Copy probability must be in [0, 1].')
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
        seed_seq = np.random.SeedSequence(self.rng_seed)
        parent_seed, child_seed = seed_seq.spawn(2)
        self._rng_parent = np.random.default_rng(parent_seed)
        self._rng_child = np.random.default_rng(child_seed)

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        p_copy: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set NEST-style public parameters."""
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_p_copy = (
            self.p_copy
            if p_copy is _UNSET
            else self._to_scalar_float(p_copy, name='p_copy')
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
            p_copy=new_p_copy,
            start=new_start,
            stop=new_stop,
        )

        self.rate = new_rate
        self.p_copy = new_p_copy
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
            'p_copy': float(self.p_copy),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_parent_spikes(self, lam: float) -> int:
        return int(self._rng_parent.poisson(lam))

    def _sample_child_spikes(self, n_parent_spikes: int) -> np.ndarray:
        out = np.zeros(self._num_targets, dtype=np.int64)

        if n_parent_spikes <= 0 or self._num_targets == 0:
            return out
        if self.p_copy <= 0.0:
            return out
        if self.p_copy >= 1.0:
            out.fill(int(n_parent_spikes))
            return out

        for i in range(self._num_targets):
            copied = np.count_nonzero(self._rng_child.random(n_parent_spikes) < self.p_copy)
            out[i] = int(copied)
        return out

    def update(self):
        if not hasattr(self, '_rng_parent'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_timing_cache(dt_ms)

        if self.rate <= 0.0:
            return np.zeros(self.varshape, dtype=np.int64)

        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)
        if not self._is_active(curr_step):
            return np.zeros(self.varshape, dtype=np.int64)

        lam = self.rate * dt_ms / 1000.0
        n_parent_spikes = self._sample_parent_spikes(lam)
        if n_parent_spikes <= 0:
            return np.zeros(self.varshape, dtype=np.int64)

        child_counts = self._sample_child_spikes(n_parent_spikes)
        return child_counts.reshape(self.varshape)
