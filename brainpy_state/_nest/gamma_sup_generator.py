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
    'gamma_sup_generator',
]


_UNSET = object()


class gamma_sup_generator(brainstate.nn.Dynamics):
    r"""Superposition of independent gamma processes (NEST-compatible).

    Description
    -----------

    ``gamma_sup_generator`` re-implements NEST's stimulation device of the
    same name. It generates, per output train, the pooled spike train from
    ``n_proc`` independent component processes with gamma-interval statistics.

    NEST implementation model
    -------------------------

    NEST represents each target train by an occupation vector over
    ``gamma_shape`` internal states:

    .. math::

       \mathbf{occ} = (occ_0, \dots, occ_{k-1}), \quad k=\text{gamma_shape}.

    At each simulation step:

    1. For each state ``i``, draw transitions
       ``n_trans[i] ~ Binomial(occ[i], p)``, where
       ``p = rate * gamma_shape * dt / 1000``.
    2. Apply NEST's Poisson approximation branch for sparse/high-count cases:
       - if ``occ[i] >= 100 and p <= 0.01``, or
       - if ``occ[i] >= 500 and p * occ[i] <= 0.1``,
       sample ``Poisson(p * occ[i])`` and clip to ``occ[i]``.
    3. Move transitioning components to the next state cyclically.
       Transitions from the last state emit spikes and return to state 0.

    The returned output is spike multiplicity per train and step (int64), i.e.
    it can be larger than 1, matching NEST ``SpikeEvent`` multiplicity.

    Timing semantics (NEST spike generators)
    ----------------------------------------

    Activity follows NEST ``StimulationDevice::is_active`` for spike devices:

    .. math::

       t_{\min} < t \le t_{\max},

    with ``t_min = origin + start`` and ``t_max = origin + stop``.
    Therefore ``start`` is exclusive and ``stop`` is inclusive.

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of independent output spike trains. Default: ``1``.
    rate : ArrayLike, optional
        Rate of each component process in spikes/s. Must be non-negative.
        Default: ``0.0 * u.Hz``.
    gamma_shape : ArrayLike, optional
        Gamma shape parameter (integer, ``>= 1``). Default: ``1``.
    n_proc : ArrayLike, optional
        Number of superimposed component processes (integer, ``>= 1``).
        Default: ``1``.
    start : ArrayLike, optional
        Activation time relative to ``origin`` in ms. Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation time relative to ``origin`` in ms. ``None`` means
        infinity. Default: ``None``.
    origin : ArrayLike, optional
        Time origin for ``start`` and ``stop`` in ms. Default: ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed for transition sampling. Default: ``0``.
    name : str, optional
        Object name.

    Notes
    -----
    - Initial occupation is the NEST equilibrium approximation used in
      ``pre_run_hook()``:
      ``floor(n_proc / gamma_shape)`` in all bins, with the remainder added
      to the last bin.
    - As in NEST, each output train maintains independent internal occupation
      states.

    References
    ----------
    .. [1] NEST source: ``models/gamma_sup_generator.cpp`` and
           ``models/gamma_sup_generator.h``.
    .. [2] NEST model docs:
           https://nest-simulator.readthedocs.io/en/stable/models/gamma_sup_generator.html
    .. [3] Deger M, Helias M, Boucsein C, Rotter S (2012).
           Statistical properties of superimposed stationary spike trains.
           Journal of Computational Neuroscience.
           https://doi.org/10.1007/s10827-011-0362-8
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        gamma_shape: ArrayLike = 1,
        n_proc: ArrayLike = 1,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.gamma_shape = self._to_scalar_int(gamma_shape, name='gamma_shape')
        self.n_proc = self._to_scalar_int(n_proc, name='n_proc')
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate=self.rate,
            gamma_shape=self.gamma_shape,
            n_proc=self.n_proc,
            start=self.start,
            stop=self.stop,
        )

        self._num_targets = int(np.prod(self.varshape))
        self._transition_prob = 0.0
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
        gamma_shape: int,
        n_proc: int,
        start: float,
        stop: float,
    ):
        if gamma_shape < 1:
            raise ValueError('The shape must be larger or equal 1')
        if rate < 0.0:
            raise ValueError('The rate must be larger than 0.')
        if n_proc < 1:
            raise ValueError('The number of component processes cannot be smaller than one')
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

        self._transition_prob = self.rate * self.gamma_shape * dt_ms / 1000.0
        self._dt_cache_ms = float(dt_ms)

    def _is_active(self, curr_step: int) -> bool:
        return (self._t_min_step < curr_step) and (curr_step <= self._t_max_step)

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs

        ini_occ_ref = int(self.n_proc // self.gamma_shape)
        ini_occ_act = int(self.n_proc - ini_occ_ref * self.gamma_shape)

        occ = np.full(
            (self._num_targets, self.gamma_shape),
            ini_occ_ref,
            dtype=np.int64,
        )
        occ[:, -1] += ini_occ_act
        self.occ = brainstate.ShortTermState(occ)
        self._rng = np.random.default_rng(self.rng_seed)

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        gamma_shape: ArrayLike | object = _UNSET,
        n_proc: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set NEST-style public parameters."""
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_gamma_shape = (
            self.gamma_shape
            if gamma_shape is _UNSET
            else self._to_scalar_int(gamma_shape, name='gamma_shape')
        )
        new_n_proc = (
            self.n_proc
            if n_proc is _UNSET
            else self._to_scalar_int(n_proc, name='n_proc')
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
            gamma_shape=new_gamma_shape,
            n_proc=new_n_proc,
            start=new_start,
            stop=new_stop,
        )

        self.rate = new_rate
        self.gamma_shape = new_gamma_shape
        self.n_proc = new_n_proc
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
            'gamma_shape': int(self.gamma_shape),
            'n_proc': int(self.n_proc),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_poisson(self, lam: float) -> int:
        return int(self._rng.poisson(lam))

    def _sample_binomial(self, n: int, p: float) -> int:
        return int(self._rng.binomial(n, p))

    def _update_internal_states(self, occ_row: np.ndarray, transition_prob: float) -> int:
        n_bins = occ_row.size
        n_trans = np.zeros(n_bins, dtype=np.int64)

        for i in range(n_bins):
            occ_i = int(occ_row[i])
            if occ_i <= 0:
                continue

            use_poisson_approx = (
                (occ_i >= 100 and transition_prob <= 0.01)
                or (occ_i >= 500 and transition_prob * occ_i <= 0.1)
            )

            if use_poisson_approx:
                n_i = self._sample_poisson(transition_prob * occ_i)
                if n_i > occ_i:
                    n_i = occ_i
            else:
                # NEST uses std::binomial_distribution directly.
                # Clamp p numerically to avoid invalid values in Python RNG.
                if transition_prob <= 0.0:
                    n_i = 0
                elif transition_prob >= 1.0:
                    n_i = occ_i
                else:
                    n_i = self._sample_binomial(occ_i, transition_prob)
            n_trans[i] = int(n_i)

        for i in range(n_bins):
            n_i = int(n_trans[i])
            if n_i <= 0:
                continue
            occ_row[i] -= n_i
            if i == n_bins - 1:
                occ_row[0] += n_i
            else:
                occ_row[i + 1] += n_i

        return int(n_trans[-1])

    def update(self):
        if not hasattr(self, 'occ'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_runtime_cache(dt_ms)

        if self.rate <= 0.0 or self._num_targets == 0:
            return jnp.zeros(self.varshape, dtype=jnp.int64)

        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)
        if not self._is_active(curr_step):
            return jnp.zeros(self.varshape, dtype=jnp.int64)

        occ = np.asarray(self.occ.value, dtype=np.int64).copy()
        counts = np.zeros(self._num_targets, dtype=np.int64)

        for idx in range(self._num_targets):
            counts[idx] = self._update_internal_states(occ[idx], self._transition_prob)

        self.occ.value = occ
        return jnp.asarray(counts.reshape(self.varshape), dtype=jnp.int64)
