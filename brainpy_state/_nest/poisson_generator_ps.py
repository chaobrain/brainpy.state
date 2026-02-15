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

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'poisson_generator_ps',
]


_UNSET = object()


class poisson_generator_ps(brainstate.nn.Dynamics):
    r"""Precise-time Poisson spike generator with dead time (NEST-compatible).

    Description
    -----------

    ``poisson_generator_ps`` re-implements NEST's precise-time stimulation
    device ``poisson_generator_ps``. For each output train, spikes are produced
    by a renewal process with absolute dead time and exponential tail:

    .. math::

       \Delta t = t_{\mathrm{dead}} + \xi \left(\frac{1000}{r} - t_{\mathrm{dead}}\right),
       \quad \xi \sim \mathrm{Exp}(1),

    where ``r`` is the configured mean rate in spikes/s (Hz) and
    ``t_dead`` is dead time in ms.

    The first spike in an active interval is initialized from the equilibrium
    backward-recurrence distribution used by NEST:

    - uniform branch on ``[0, dead_time)`` with probability
      ``dead_time * rate / 1000``,
    - exponential branch on ``[dead_time, +inf)`` otherwise.

    This preserves stationary-rate behavior at activation.

    Update Ordering and Activity Window
    -----------------------------------

    At each simulation step with left edge ``t`` and right edge ``t + dt``:

    1. Compute active slice limits

       .. math::
          t_{\min} = \max(t, origin + start), \qquad
          t_{\max} = \min(t + dt, origin + stop).

    2. If ``t_min < t_max`` and ``rate > 0``, emit all spikes with
       ``t_min < spike_time <= t_max`` for each output train.

    This mirrors NEST ``poisson_generator_ps`` update semantics
    (``models/poisson_generator_ps.cpp``).

    Parameters
    ----------
    in_size : Size, optional
        Number/shape of independent output precise spike trains.
        Default: ``1``.
    rate : ArrayLike, optional
        Mean firing rate in spikes/s. Must be non-negative.
        Default: ``0.0 * u.Hz``.
    dead_time : ArrayLike, optional
        Absolute dead time in ms. Must be non-negative and satisfy
        ``dead_time <= 1000 / rate`` when ``rate > 0``.
        Default: ``0.0 * u.ms``.
    start : ArrayLike, optional
        Activation time relative to ``origin`` (ms). Default: ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Deactivation time relative to ``origin`` (ms). ``None`` means infinity.
        Default: ``None``.
    origin : ArrayLike, optional
        Global time offset for ``start`` and ``stop`` (ms).
        Default: ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed used for internal random streams. Default: ``0``.
    name : str, optional
        Object name.

    Notes
    -----
    - Unlike grid-constrained ``poisson_generator``, this model tracks and
      emits precise (off-grid) spike times.
    - ``update()`` returns per-step spike multiplicities (int64) shaped by
      ``in_size``.
    - ``update(return_precise_times=True)`` additionally returns per-train
      precise spike times (ms) emitted in the current step.
    - The state variables ``last_spike_time`` and ``last_spike_offset`` store
      the most recently emitted spike time and offset from the right step edge
      (ms), following the convention used in this repository's precise models.

    References
    ----------
    .. [1] NEST source: ``models/poisson_generator_ps.h`` and
           ``models/poisson_generator_ps.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/poisson_generator_ps.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        dead_time: ArrayLike = 0. * u.ms,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.dead_time = self._to_scalar_time_ms(dead_time)
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate=self.rate,
            dead_time=self.dead_time,
            start=self.start,
            stop=self.stop,
            origin=self.origin,
        )

        self._num_targets = int(np.prod(self.varshape))
        self._last_step_spike_times_ms = tuple(
            np.asarray([], dtype=np.float64) for _ in range(self._num_targets)
        )

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
    def _validate_parameters(
        *,
        rate: float,
        dead_time: float,
        start: float,
        stop: float,
        origin: float,
    ):
        if rate < 0.0:
            raise ValueError('The rate cannot be negative.')
        if dead_time < 0.0:
            raise ValueError('The dead time cannot be negative.')
        if stop < start:
            raise ValueError('stop >= start required.')
        if not np.isfinite(start):
            raise ValueError('start must be finite.')
        if not np.isfinite(origin):
            raise ValueError('origin must be finite.')
        if (not np.isinf(stop)) and (not np.isfinite(stop)):
            raise ValueError('stop must be finite or infinity.')
        if rate > 0.0 and (1000.0 / rate < dead_time):
            raise ValueError('The inverse rate cannot be smaller than the dead time.')

    def _dt_ms(self) -> float:
        dt = brainstate.environ.get_dt()
        return self._to_scalar_time_ms(dt)

    def _current_time_ms(self) -> float:
        t = brainstate.environ.get('t', default=0. * u.ms)
        if t is None:
            return 0.0
        return self._to_scalar_time_ms(t)

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self.next_spike_time = brainstate.ShortTermState(
            np.full(self._num_targets, -np.inf, dtype=np.float64)
        )
        self.last_spike_time = brainstate.ShortTermState(
            np.full(self.varshape, -np.inf, dtype=np.float64)
        )
        self.last_spike_offset = brainstate.ShortTermState(
            np.zeros(self.varshape, dtype=np.float64)
        )

        # Independent random streams per target keep train generation stable
        # across different simulation resolutions.
        seed_seq = np.random.SeedSequence(self.rng_seed)
        self._rngs = tuple(
            np.random.default_rng(s) for s in seed_seq.spawn(self._num_targets)
        )
        self._last_step_spike_times_ms = tuple(
            np.asarray([], dtype=np.float64) for _ in range(self._num_targets)
        )

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        dead_time: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set NEST-style parameters for the precise Poisson generator."""
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_dead_time = (
            self.dead_time if dead_time is _UNSET else self._to_scalar_time_ms(dead_time)
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
            start=new_start,
            stop=new_stop,
            origin=new_origin,
        )

        self.rate = new_rate
        self.dead_time = new_dead_time
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        # NEST resets next spike states when "rate" is set.
        if (rate is not _UNSET) and hasattr(self, 'next_spike_time'):
            self.next_spike_time.value = np.full(
                self._num_targets, -np.inf, dtype=np.float64
            )

        # Match NEST pre-run behavior when start/origin are shifted forward:
        # if previous next-spike times lie before the new activation start,
        # reinitialize all target streams.
        if (
            (start is not _UNSET or origin is not _UNSET)
            and hasattr(self, 'next_spike_time')
        ):
            vals = np.asarray(self.next_spike_time.value, dtype=np.float64)
            finite = np.isfinite(vals)
            if finite.any() and float(np.min(vals[finite])) < (self.origin + self.start):
                self.next_spike_time.value = np.full(
                    self._num_targets, -np.inf, dtype=np.float64
                )

    def get(self) -> dict:
        """Return current public parameters."""
        return {
            'rate': float(self.rate),
            'dead_time': float(self.dead_time),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    @property
    def step_spike_times_ms(self):
        """Precise spike times (ms) emitted in the latest update, per target."""
        return self._last_step_spike_times_ms

    def _sample_initial_offset_ms(self, rng: np.random.Generator, inv_rate_ms: float) -> float:
        if self.dead_time > 0.0 and rng.random() < (self.dead_time * self.rate / 1000.0):
            # Uniform branch on [0, dead_time).
            return float(rng.random() * self.dead_time)
        # Exponential branch on [dead_time, +inf).
        return float(rng.exponential() * inv_rate_ms + self.dead_time)

    def _sample_isi_ms(self, rng: np.random.Generator, inv_rate_ms: float) -> float:
        return float(rng.exponential() * inv_rate_ms + self.dead_time)

    def update(self, return_precise_times: bool = False):
        if not hasattr(self, 'next_spike_time'):
            self.init_state()

        dt_ms = self._dt_ms()
        if dt_ms <= 0.0:
            raise ValueError('Simulation time step must be positive.')

        t_ms = self._current_time_ms()
        t_min_active = max(t_ms, self.origin + self.start)
        t_max_active = min(t_ms + dt_ms, self.origin + self.stop)

        counts = np.zeros(self._num_targets, dtype=np.int64)
        empty_events = tuple(np.asarray([], dtype=np.float64) for _ in range(self._num_targets))

        if self._num_targets == 0 or self.rate <= 0.0 or not (t_min_active < t_max_active):
            self._last_step_spike_times_ms = empty_events
            out = counts.reshape(self.varshape)
            if return_precise_times:
                return out, self._last_step_spike_times_ms
            return out

        inv_rate_ms = 1000.0 / self.rate - self.dead_time
        next_spike = np.asarray(self.next_spike_time.value, dtype=np.float64).copy()
        last_time = np.asarray(self.last_spike_time.value, dtype=np.float64).reshape(-1).copy()
        last_offset = np.asarray(self.last_spike_offset.value, dtype=np.float64).reshape(-1).copy()

        right_edge = t_ms + dt_ms
        events = []

        for i in range(self._num_targets):
            rng = self._rngs[i]
            next_t = float(next_spike[i])

            if np.isneginf(next_t):
                next_t = t_min_active + self._sample_initial_offset_ms(rng, inv_rate_ms)

            ev = []
            while next_t <= t_max_active:
                counts[i] += 1
                ev.append(next_t)
                last_time[i] = next_t
                off = right_edge - next_t
                if off < 0.0 and off > -1e-12:
                    off = 0.0
                last_offset[i] = off
                next_t += self._sample_isi_ms(rng, inv_rate_ms)

            next_spike[i] = next_t
            events.append(np.asarray(ev, dtype=np.float64))

        self.next_spike_time.value = next_spike
        self.last_spike_time.value = last_time.reshape(self.varshape)
        self.last_spike_offset.value = last_offset.reshape(self.varshape)
        self._last_step_spike_times_ms = tuple(events)

        out = counts.reshape(self.varshape)
        if return_precise_times:
            return out, self._last_step_spike_times_ms
        return out
