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

from brainpy_state._base import Dynamics

__all__ = [
    'spike_dilutor',
]


_UNSET = object()


class spike_dilutor(Dynamics):
    r"""Repeat incoming spikes with Bernoulli copy probability.

    Description
    -----------

    ``spike_dilutor`` re-implements NEST's deprecated ``spike_dilutor`` device.
    The model receives incoming spike multiplicities (mother process) and
    repeats each incoming spike independently to each target train with
    probability ``p_copy``.

    For each simulation step and each output target:

    .. math::

       N_\text{child} = \sum_{k=1}^{N_\text{mother}} \mathbf{1}[U_k < p_\text{copy}],
       \quad U_k \sim \mathrm{Uniform}(0, 1).

    Output values are integer multiplicities per step (``0, 1, 2, ...``),
    matching NEST ``SpikeEvent`` multiplicity semantics.

    NEST update ordering (source-equivalent)
    ----------------------------------------

    NEST ``models/spike_dilutor.cpp`` performs:

    1. Check stimulation-device activity at current step.
    2. Read mother multiplicity from input ring buffer for this step.
    3. Send one ``DSSpikeEvent`` carrying mother multiplicity.
    4. In ``event_hook()``, for each receiver independently:
       - run an explicit Bernoulli loop over mother spikes,
       - temporarily overwrite event multiplicity with copied count,
       - deliver to receiver if copied count is non-zero,
       - restore original mother multiplicity before next receiver.

    This implementation mirrors that semantics by generating one copied
    multiplicity per output element of ``in_size`` from the same mother
    multiplicity and by preserving explicit Bernoulli-trial behavior.

    Timing semantics
    ----------------

    As in NEST spike stimulation devices, activity follows

    .. math::

       t_{\min} < t \le t_{\max},

    where :math:`t_{\min}=\mathrm{origin}+\mathrm{start}` and
    :math:`t_{\max}=\mathrm{origin}+\mathrm{stop}`.

    Therefore:

    - ``start`` is exclusive,
    - ``stop`` is inclusive.

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of output child spike trains (targets). Default: ``1``.
    p_copy : ArrayLike, optional
        Copy probability for each incoming mother spike and each target.
        Must be in ``[0, 1]``. Default: ``1.0``.
    start : ArrayLike, optional
        Activation time relative to ``origin`` in ms. Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation time relative to ``origin`` in ms. ``None`` means
        infinity. Default: ``None``.
    origin : ArrayLike, optional
        Time offset for ``start``/``stop`` in ms. Default: ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed for internal copy-process random draws. Default: ``0``.
    name : str, optional
        Object name.

    Notes
    -----
    - Incoming mother spikes are provided through the ``mother_spikes``
      argument of :meth:`update`, and can also be accumulated via
      :meth:`add_delta_input` / :meth:`add_current_input`.
    - Like NEST, this model is deprecated in favor of probabilistic synapses
      (e.g., ``bernoulli_synapse`` in NEST).
    - NEST restricts ``spike_dilutor`` to single-threaded simulations. This
      backend does not expose NEST thread kernels, so that restriction is not
      modeled here.

    References
    ----------
    .. [1] NEST source: ``models/spike_dilutor.h`` and
           ``models/spike_dilutor.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/spike_dilutor.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        p_copy: ArrayLike = 1.0,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.p_copy = self._to_scalar_float(p_copy, name='p_copy')
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
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
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _validate_parameters(
        *,
        p_copy: float,
        start: float,
        stop: float,
    ):
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

    @staticmethod
    def _to_nonnegative_count(value: ArrayLike) -> int:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        total = float(arr.sum())
        if total < 0.0:
            raise ValueError('mother_spikes must be non-negative.')
        return int(np.trunc(total))

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
        self._rng = np.random.default_rng(self.rng_seed)

    def set(
        self,
        *,
        p_copy: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set NEST-style public parameters."""
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
            p_copy=new_p_copy,
            start=new_start,
            stop=new_stop,
        )

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
            'p_copy': float(self.p_copy),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_child_spikes(self, n_mother_spikes: int) -> np.ndarray:
        out = np.zeros(self._num_targets, dtype=np.int64)

        if n_mother_spikes <= 0 or self._num_targets == 0:
            return out
        if self.p_copy <= 0.0:
            return out
        if self.p_copy >= 1.0:
            out.fill(int(n_mother_spikes))
            return out

        # One Bernoulli trial per (target, mother spike), matching NEST's
        # explicit event_hook loop semantics.
        draws = self._rng.random((self._num_targets, int(n_mother_spikes)))
        out[:] = np.count_nonzero(draws < self.p_copy, axis=1).astype(np.int64)
        return out

    def update(self, mother_spikes: ArrayLike = 0.0):
        if not hasattr(self, '_rng'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_timing_cache(dt_ms)

        # Mother multiplicity for the current step: direct input argument plus
        # optional registered current/delta inputs.
        total_spikes = self.sum_current_inputs(mother_spikes)
        total_spikes = self.sum_delta_inputs(total_spikes)
        n_mother_spikes = self._to_nonnegative_count(total_spikes)

        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)
        if not self._is_active(curr_step) or n_mother_spikes <= 0:
            return np.zeros(self.varshape, dtype=np.int64)

        child_counts = self._sample_child_spikes(n_mother_spikes)
        return child_counts.reshape(self.varshape)
