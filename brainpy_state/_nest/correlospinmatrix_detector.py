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

import math
from collections import deque
from dataclasses import dataclass

import brainstate
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'correlospinmatrix_detector',
]


@dataclass
class _BinaryPulse:
    t_on: int
    t_off: int
    receptor_channel: int


@dataclass
class _Calibration:
    dt_ms: float
    t_min_steps: int
    t_max_steps: float
    delta_tau_ms: float
    delta_tau_steps: int
    tau_max_ms: float
    tau_max_steps: int
    tau_edge_steps: int
    tstart_ms: float
    tstop_ms: float
    n_channels: int
    n_bins: int
    min_delay_steps: int
    signature: tuple


class correlospinmatrix_detector(brainstate.nn.Dynamics):
    r"""NEST-compatible ``correlospinmatrix_detector`` device.

    Short Description
    -----------------
    ``correlospinmatrix_detector`` receives binary-state spike streams from
    multiple receptor channels and accumulates raw auto/cross covariance
    histograms over positive and negative lags.

    Description
    -----------
    This class mirrors NEST ``correlospinmatrix_detector``
    (``models/correlospinmatrix_detector.{h,cpp}``) including event decoding,
    down-transition handling, queue pruning, and lag-bin updates.

    Each receptor channel is interpreted as a binary signal. Binary transitions
    are decoded using the same NEST rule:

    - one spike (multiplicity ``1``): tentative transition to state ``0``,
    - two spikes at one channel and one stamp (multiplicity ``2`` or two
      consecutive multiplicity-1 events with equal channel+stamp): transition
      to state ``1``.

    Covariance updates are performed only when a down transition is confirmed.
    The finished binary pulse is inserted into a history deque sorted by pulse
    off-time, then correlated against all pulses still in the queue.

    Output
    ------
    ``count_covariance`` is a rank-3 int64 tensor with shape
    ``(N_channels, N_channels, 2 * tau_max / delta_tau + 1)``.
    Bin index ``tau_max / delta_tau`` corresponds to zero lag.

    Bin-edge semantics follow NEST C++ loops:

    - lower-triangular entries use left-closed, right-open handling,
    - diagonal and upper-triangular entries use left-open, right-closed
      handling.

    Parameters
    ----------
    in_size : int, optional
        Device batch size. Defaults to ``1``.
    delta_tau : Quantity[ms] or float or None, optional
        Bin width. Must be a multiple of simulation ``dt``.
        If ``None``, defaults to simulation ``dt``.
    tau_max : Quantity[ms] or float or None, optional
        One-sided lag range. Must be a multiple of ``delta_tau``.
        If ``None``, defaults to ``10 * delta_tau``.
    Tstart : Quantity[ms] or float, optional
        Kept for NEST API compatibility. Defaults to ``0.0 * u.ms``.
        As in current NEST source, it triggers reset behavior when changed but
        does not gate accumulation in ``handle``.
    Tstop : Quantity[ms] or float or None, optional
        Kept for NEST API compatibility. Defaults to ``None`` (+infinity).
        As in current NEST source, it triggers reset behavior when changed but
        does not gate accumulation in ``handle``.
    N_channels : int, optional
        Number of receptor pools. Must be >= 1. Defaults to ``1``.
    start : Quantity[ms], optional
        Activity window start relative to ``origin`` (exclusive).
    stop : Quantity[ms] or None, optional
        Activity window stop relative to ``origin`` (inclusive).
        ``None`` means +infinity.
    origin : Quantity[ms], optional
        Activity window origin shift.
    name : str, optional
        Module name.

    Notes
    -----
    - Connection delays and weights are ignored, matching NEST.
    - Optional ``multiplicities`` emulate NEST ``SpikeEvent`` multiplicity.
    - History pruning uses the default NEST minimum delay semantics
      (`min_delay = 1` simulation step).

    References
    ----------
    .. [1] NEST Simulator, ``correlospinmatrix_detector`` model.
           https://nest-simulator.readthedocs.io/en/stable/models/correlospinmatrix_detector.html
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        delta_tau: ArrayLike = None,
        tau_max: ArrayLike = None,
        Tstart: ArrayLike = 0.0 * u.ms,
        Tstop: ArrayLike = None,
        N_channels: int = 1,
        start: ArrayLike = 0.0 * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0.0 * u.ms,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.delta_tau = delta_tau
        self.tau_max = tau_max
        self.Tstart = Tstart
        self.Tstop = Tstop
        self.N_channels = N_channels

        self.start = start
        self.stop = stop
        self.origin = origin

        self._calib: _Calibration | None = None
        self._incoming: deque[_BinaryPulse] = deque()

        self._last_i = 0
        self._t_last_in_spike = -2**62
        self._tentative_down = False
        self._curr_state = np.zeros((0,), dtype=np.bool_)
        self._last_change = np.zeros((0,), dtype=np.int64)
        self._count_covariance = np.zeros((0, 0, 0), dtype=np.int64)

        self._ensure_calibrated_from_env_if_available()

    @property
    def count_covariance(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        return np.asarray(self._count_covariance, dtype=np.int64)

    def get(self, key: str = 'count_covariance'):
        if key == 'count_covariance':
            return self.count_covariance
        if key == 'delta_tau':
            self._ensure_calibrated_from_env_if_available()
            return float(self._calib.delta_tau_ms) if self._calib is not None else None
        if key == 'tau_max':
            self._ensure_calibrated_from_env_if_available()
            return float(self._calib.tau_max_ms) if self._calib is not None else None
        if key == 'Tstart':
            return self._to_ms_scalar(self.Tstart, name='Tstart', allow_inf=True)
        if key == 'Tstop':
            stop_val = math.inf if self.Tstop is None else self.Tstop
            return self._to_ms_scalar(stop_val, name='Tstop', allow_inf=True)
        if key == 'N_channels':
            return int(self._to_int_scalar(self.N_channels, name='N_channels'))
        if key == 'start':
            return self._to_ms_scalar(self.start, name='start')
        if key == 'stop':
            stop_val = math.inf if self.stop is None else self.stop
            return self._to_ms_scalar(stop_val, name='stop', allow_inf=True)
        if key == 'origin':
            return self._to_ms_scalar(self.origin, name='origin')
        raise KeyError(f'Unsupported key "{key}" for correlospinmatrix_detector.get().')

    def connect(self):
        return None

    def flush(self):
        return {
            'count_covariance': self.count_covariance,
        }

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self._ensure_calibrated_from_env_if_available()
        self._reset_state()

    def update(
        self,
        spikes: ArrayLike = None,
        receptor_ports: ArrayLike = None,
        receptor_types: ArrayLike = None,
        multiplicities: ArrayLike = None,
        stamp_steps: ArrayLike = None,
    ):
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        calib = self._ensure_calibrated(dt)

        step_now = self._time_to_step(t, calib.dt_ms)

        if spikes is None:
            return self.flush()

        spike_arr = self._to_float_array(spikes, name='spikes')
        if spike_arr.size == 0:
            return self.flush()

        n_items = spike_arr.size

        if receptor_ports is None and receptor_types is not None:
            receptor_ports = receptor_types
        port_arr = self._to_int_array(receptor_ports, name='receptor_ports', default=0, size=n_items)

        if multiplicities is None:
            rounded = np.rint(spike_arr)
            is_integer_like = np.allclose(spike_arr, rounded, atol=1e-12, rtol=1e-12)
            if is_integer_like:
                counts = np.maximum(rounded.astype(np.int64), 0)
            else:
                counts = (spike_arr > 0.0).astype(np.int64)
        else:
            mult_arr = self._to_int_array(multiplicities, name='multiplicities', size=n_items)
            if np.any(mult_arr < 0):
                raise ValueError('multiplicities must be non-negative.')
            counts = np.where(spike_arr > 0.0, mult_arr, 0)

        if stamp_steps is None:
            stamp_arr = np.full((n_items,), step_now + 1, dtype=np.int64)
        else:
            stamp_arr = self._to_int_array(stamp_steps, name='stamp_steps', size=n_items)

        for i in range(n_items):
            multiplicity = int(counts[i])
            if multiplicity <= 0:
                continue

            curr_i = int(port_arr[i])
            if curr_i < 0 or curr_i >= calib.n_channels:
                raise ValueError(f'Unknown receptor_type {curr_i} for correlospinmatrix_detector.')

            stamp_step = int(stamp_arr[i])
            if not self._is_active(stamp_step, calib.t_min_steps, calib.t_max_steps):
                continue

            self._handle_event(curr_i=curr_i, stamp_step=stamp_step, multiplicity=multiplicity, calib=calib)

        return self.flush()

    def _handle_event(
        self,
        curr_i: int,
        stamp_step: int,
        multiplicity: int,
        calib: _Calibration,
    ):
        down_transition = False

        if multiplicity == 1:
            if curr_i == self._last_i and stamp_step == self._t_last_in_spike:
                self._curr_state[curr_i] = True
                self._last_change[curr_i] = int(stamp_step)
                self._tentative_down = False
            else:
                if self._tentative_down:
                    down_transition = True
                self._tentative_down = True
        elif multiplicity == 2:
            self._curr_state[curr_i] = True

            if self._tentative_down:
                down_transition = True

            self._curr_state[self._last_i] = False
            self._last_change[curr_i] = int(stamp_step)
            self._tentative_down = False

        if down_transition:
            self._process_down_transition(calib=calib)

        self._last_i = curr_i
        self._t_last_in_spike = int(stamp_step)

    def _process_down_transition(self, calib: _Calibration):
        i = int(self._last_i)
        t_i_on = int(self._last_change[i])
        t_i_off = int(self._t_last_in_spike)

        t_min_on = t_i_on
        for n in range(calib.n_channels):
            if bool(self._curr_state[n]) and int(self._last_change[n]) < t_min_on:
                t_min_on = int(self._last_change[n])

        while len(self._incoming) > 0:
            if (t_min_on - self._incoming[0].t_off) >= (calib.tau_edge_steps + calib.min_delay_steps):
                self._incoming.popleft()
            else:
                break

        pulse_i = _BinaryPulse(t_on=t_i_on, t_off=t_i_off, receptor_channel=i)

        insert_pos = len(self._incoming)
        for idx, pulse in enumerate(self._incoming):
            if pulse.t_off > pulse_i.t_off:
                insert_pos = idx
                break
        self._incoming.insert(insert_pos, pulse_i)

        t0 = calib.tau_max_steps // calib.delta_tau_steps
        dt = calib.delta_tau_steps

        for pulse_j in self._incoming:
            j = int(pulse_j.receptor_channel)
            t_j_on = int(pulse_j.t_on)
            t_j_off = int(pulse_j.t_off)

            delta_ij_min = max(t_j_on - t_i_off, -calib.tau_max_steps)
            delta_ij_max = min(t_j_off - t_i_on, calib.tau_max_steps)

            lag = min(t_i_off, t_j_off) - max(t_i_on, t_j_on)
            if lag > 0:
                self._count_covariance[i, j, t0] += int(lag)
                if i != j:
                    self._count_covariance[j, i, t0] += int(lag)

            delta_start = self._trunc_div(delta_ij_min, dt)
            for delta in range(delta_start, 0):
                lag = min(t_i_off, t_j_off - delta * dt) - max(t_i_on, t_j_on - delta * dt)
                if lag > 0:
                    self._count_covariance[i, j, t0 - delta] += int(lag)
                    self._count_covariance[j, i, t0 + delta] += int(lag)

            if i != j:
                delta_end = self._trunc_div(delta_ij_max, dt)
                for delta in range(1, delta_end + 1):
                    lag = min(t_i_off, t_j_off - delta * dt) - max(t_i_on, t_j_on - delta * dt)
                    if lag > 0:
                        self._count_covariance[i, j, t0 - delta] += int(lag)
                        self._count_covariance[j, i, t0 + delta] += int(lag)

        self._last_change[i] = int(t_i_off)

    @staticmethod
    def _trunc_div(a: int, b: int) -> int:
        return int(float(a) / float(b))

    def _ensure_calibrated_from_env_if_available(self):
        try:
            dt = brainstate.environ.get_dt()
        except KeyError:
            return
        self._ensure_calibrated(dt)

    def _ensure_calibrated(self, dt) -> _Calibration:
        new_calib = self._compute_calibration(dt)

        if self._calib is None or self._calib.signature != new_calib.signature:
            self._calib = new_calib
            self._reset_state()

        return self._calib

    def _reset_state(self):
        self._incoming = deque()
        self._last_i = 0
        self._t_last_in_spike = -2**62
        self._tentative_down = False

        if self._calib is None:
            self._curr_state = np.zeros((0,), dtype=np.bool_)
            self._last_change = np.zeros((0,), dtype=np.int64)
            self._count_covariance = np.zeros((0, 0, 0), dtype=np.int64)
            return

        n_channels = int(self._calib.n_channels)
        n_bins = int(self._calib.n_bins)

        self._curr_state = np.zeros((n_channels,), dtype=np.bool_)
        self._last_change = np.zeros((n_channels,), dtype=np.int64)
        self._count_covariance = np.zeros((n_channels, n_channels, n_bins), dtype=np.int64)

    def _compute_calibration(self, dt) -> _Calibration:
        dt_ms = self._to_ms_scalar(dt, name='dt')
        if dt_ms <= 0.0:
            raise ValueError('Simulation resolution dt must be positive.')

        start_steps = self._to_step_count(self.start, dt_ms, 'start')
        stop_value = math.inf if self.stop is None else self.stop
        stop_steps = self._to_step_count(stop_value, dt_ms, 'stop', allow_inf=True)
        if (not math.isinf(stop_steps)) and (stop_steps < start_steps):
            raise ValueError('stop >= start required.')

        origin_steps = self._to_step_count(self.origin, dt_ms, 'origin')
        t_min_steps = origin_steps + start_steps
        t_max_steps = math.inf if math.isinf(stop_steps) else origin_steps + stop_steps

        if self.delta_tau is None:
            delta_tau_ms = dt_ms
        else:
            delta_tau_ms = self._to_ms_scalar(self.delta_tau, name='delta_tau')

        if (not math.isfinite(delta_tau_ms)) or (delta_tau_ms <= 0.0):
            raise ValueError('/delta_tau must be positive and finite.')
        delta_tau_steps = self._to_step_count(delta_tau_ms, dt_ms, 'delta_tau')

        if self.tau_max is None:
            tau_max_ms = 10.0 * delta_tau_ms
        else:
            tau_max_ms = self._to_ms_scalar(self.tau_max, name='tau_max')

        if (not math.isfinite(tau_max_ms)) or (tau_max_ms < 0.0):
            raise ValueError('/tau_max must be finite and non-negative.')
        tau_max_steps = self._to_step_count(tau_max_ms, dt_ms, 'tau_max')
        if tau_max_steps % delta_tau_steps != 0:
            raise ValueError('tau_max must be a multiple of delta_tau.')

        tstart_ms = self._to_ms_scalar(self.Tstart, name='Tstart', allow_inf=True)
        tstop_value = math.inf if self.Tstop is None else self.Tstop
        tstop_ms = self._to_ms_scalar(tstop_value, name='Tstop', allow_inf=True)
        if tstart_ms < 0.0:
            raise ValueError('/Tstart must not be negative.')
        if tstop_ms < 0.0:
            raise ValueError('/Tstop must not be negative.')

        n_channels = self._to_int_scalar(self.N_channels, name='N_channels')
        if n_channels < 1:
            raise ValueError('/N_channels can only be larger than zero.')

        n_bins = int(1 + (2 * tau_max_steps) // delta_tau_steps)
        min_delay_steps = 1

        signature = (
            float(dt_ms),
            int(t_min_steps),
            float(t_max_steps),
            float(delta_tau_ms),
            int(delta_tau_steps),
            float(tau_max_ms),
            int(tau_max_steps),
            float(tstart_ms),
            float(tstop_ms),
            int(n_channels),
            int(n_bins),
            int(min_delay_steps),
        )

        return _Calibration(
            dt_ms=float(dt_ms),
            t_min_steps=int(t_min_steps),
            t_max_steps=float(t_max_steps),
            delta_tau_ms=float(delta_tau_ms),
            delta_tau_steps=int(delta_tau_steps),
            tau_max_ms=float(tau_max_ms),
            tau_max_steps=int(tau_max_steps),
            tau_edge_steps=int(tau_max_steps + delta_tau_steps),
            tstart_ms=float(tstart_ms),
            tstop_ms=float(tstop_ms),
            n_channels=int(n_channels),
            n_bins=int(n_bins),
            min_delay_steps=int(min_delay_steps),
            signature=signature,
        )

    @staticmethod
    def _to_ms_scalar(value, name: str, allow_inf: bool = False) -> float:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value / u.ms)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be a scalar time value.')
        val = float(arr[0])
        if (not allow_inf) and (not math.isfinite(val)):
            raise ValueError(f'{name} must be finite.')
        return val

    @staticmethod
    def _to_int_scalar(value, name: str) -> int:
        arr = np.asarray(u.math.asarray(value), dtype=np.int64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be a scalar integer value.')
        return int(arr[0])

    @classmethod
    def _to_step_count(
        cls,
        value,
        dt_ms: float,
        name: str,
        allow_inf: bool = False,
    ):
        if value is None:
            if allow_inf:
                return math.inf
            raise ValueError(f'{name} cannot be None.')

        ms = cls._to_ms_scalar(value, name=name, allow_inf=allow_inf)
        if math.isinf(ms):
            if allow_inf:
                return math.inf
            raise ValueError(f'{name} must be finite.')

        steps_f = ms / dt_ms
        steps_i = int(np.rint(steps_f))
        if not np.isclose(steps_f, steps_i, atol=1e-12, rtol=1e-12):
            raise ValueError(f'{name} must be a multiple of the simulation resolution.')
        return steps_i

    def _time_to_step(self, t, dt_ms: float) -> int:
        t_ms = self._to_ms_scalar(t, name='t')
        steps_f = t_ms / dt_ms
        steps_i = int(np.rint(steps_f))
        if not np.isclose(steps_f, steps_i, atol=1e-12, rtol=1e-12):
            raise ValueError('Current simulation time t must be aligned to the simulation grid.')
        return steps_i

    @staticmethod
    def _is_active(stamp_step: int, t_min_steps: int, t_max_steps: float) -> bool:
        if stamp_step <= t_min_steps:
            return False
        if math.isinf(t_max_steps):
            return True
        return stamp_step <= t_max_steps

    @staticmethod
    def _to_float_array(
        x,
        name: str,
        default: float = None,
        size: int = None,
    ) -> np.ndarray:
        if x is None:
            if default is None:
                raise ValueError(f'{name} cannot be None.')
            arr = np.asarray([default], dtype=np.float64)
        else:
            if isinstance(x, u.Quantity):
                x = u.get_mantissa(x)
            arr = np.asarray(u.math.asarray(x), dtype=np.float64).reshape(-1)

        if arr.size == 0 and size is not None:
            return np.zeros((0,), dtype=np.float64)

        if not np.all(np.isfinite(arr)):
            raise ValueError(f'{name} must contain finite values.')

        if size is None:
            return arr

        if arr.size == 1 and size > 1:
            return np.full((size,), arr[0], dtype=np.float64)
        if arr.size != size:
            raise ValueError(f'{name} size ({arr.size}) does not match spikes size ({size}).')
        return arr.astype(np.float64, copy=False)

    @staticmethod
    def _to_int_array(
        x,
        name: str,
        default: int = None,
        size: int = None,
    ) -> np.ndarray:
        if x is None:
            if default is None:
                raise ValueError(f'{name} cannot be None.')
            arr = np.asarray([default], dtype=np.int64)
        else:
            arr = np.asarray(u.math.asarray(x), dtype=np.int64).reshape(-1)

        if size is None:
            return arr.astype(np.int64, copy=False)

        if arr.size == 1 and size > 1:
            return np.full((size,), int(arr[0]), dtype=np.int64)
        if arr.size != size:
            raise ValueError(f'{name} size ({arr.size}) does not match spikes size ({size}).')
        return arr.astype(np.int64, copy=False)
