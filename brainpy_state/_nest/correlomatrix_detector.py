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
    'correlomatrix_detector',
]


@dataclass
class _Spike:
    timestep: int
    weight: float
    receptor_channel: int


@dataclass
class _Calibration:
    dt_ms: float
    start_step: int
    stop_step: float
    origin_step: int
    t_min_steps: int
    t_max_steps: float
    delta_tau_ms: float
    delta_tau_steps: int
    tau_max_ms: float
    tau_max_steps: int
    tau_edge_steps: float
    tstart_ms: float
    tstop_ms: float
    n_channels: int
    n_bins: int
    min_delay_steps: int
    signature: tuple


class correlomatrix_detector(brainstate.nn.Dynamics):
    r"""NEST-compatible ``correlomatrix_detector`` device.

    Short Description
    -----------------
    ``correlomatrix_detector`` receives spikes from ``N_channels`` receptor
    pools and accumulates binned auto/cross-covariance matrices for
    non-negative lags.

    Description
    -----------
    This class mirrors NEST ``correlomatrix_detector``
    (``models/correlomatrix_detector.{h,cpp}``) including update ordering and
    matrix bin semantics:

    - Inputs are grouped by receptor port ``0 .. N_channels-1``.
    - Connection delays are ignored for correlation content.
    - A single time-sorted queue stores accepted spikes as
      ``(timestep, multiplicity*weight, receptor_channel)``.
    - For each new accepted event, the event is inserted into the queue first,
      old spikes are pruned, and then covariance/count matrices are updated
      against all remaining queued spikes.

    Two outputs are maintained:

    - ``count_covariance``: unweighted integer covariance counts,
    - ``covariance``: weighted covariance (float64).

    Shapes are ``(N_channels, N_channels, N_bins)`` with
    ``N_bins = tau_max / delta_tau + 1`` (non-negative lags only).

    Bin Semantics
    -------------
    Bin edge handling matches NEST C++ implementation:

    - Lower-triangular entries (``i > j``): intervals are left-closed,
      right-open.
    - Diagonal and upper-triangular entries (``i <= j``): intervals are
      left-open, right-closed.

    This preserves symmetry reconstruction
    :math:`C(t) = C^T(-t)` when stacking lower/upper parts.

    Time Windows
    ------------
    As in NEST, two windows apply:

    - Device activity window
      :math:`(\mathrm{origin}+\mathrm{start},\mathrm{origin}+\mathrm{stop}]`:
      events outside are discarded completely.
    - Counting window ``[Tstart, Tstop]``:
      only accepted events within this interval increment
      ``n_events``, ``covariance``, and ``count_covariance``.

    Parameters
    ----------
    in_size : int, optional
        Device batch size. Defaults to ``1``.
    delta_tau : Quantity[ms] or float or None, optional
        Bin width. Must be an odd multiple of simulation ``dt``.
        If ``None``, defaults to ``5 * dt``.
    tau_max : Quantity[ms] or float or None, optional
        One-sided lag range. Must be a multiple of ``delta_tau``.
        If ``None``, defaults to ``10 * delta_tau``.
    Tstart : Quantity[ms] or float, optional
        Counting window start (inclusive). Defaults to ``0.0 * u.ms``.
    Tstop : Quantity[ms] or float or None, optional
        Counting window stop (inclusive). ``None`` means +infinity.
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

    State Access
    ------------
    - ``covariance``: weighted covariance matrix (float64)
    - ``count_covariance``: unweighted covariance matrix (int64)
    - ``n_events``: per-channel counted events (int64)

    Notes
    -----
    - Unlike some NEST recording devices, ``n_events`` is read-only here,
      matching ``correlomatrix_detector`` behavior.
    - This implementation uses default NEST kernel minimum delay semantics in
      pruning (`min_delay = 1` simulation step).
    - Optional ``multiplicities`` emulate NEST ``SpikeEvent`` multiplicity.

    References
    ----------
    .. [1] NEST Simulator, ``correlomatrix_detector`` model.
           https://nest-simulator.readthedocs.io/en/stable/models/correlomatrix_detector.html
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
        self._incoming: deque[_Spike] = deque()
        self._n_events = np.zeros((0,), dtype=np.int64)
        self._covariance = np.zeros((0, 0, 0), dtype=np.float64)
        self._count_covariance = np.zeros((0, 0, 0), dtype=np.int64)

        self._ensure_calibrated_from_env_if_available()

    @property
    def n_events(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        return np.asarray(self._n_events, dtype=np.int64)

    @property
    def covariance(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        return np.asarray(self._covariance, dtype=np.float64)

    @property
    def count_covariance(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        return np.asarray(self._count_covariance, dtype=np.int64)

    def get(self, key: str = 'covariance'):
        if key == 'covariance':
            return self.covariance
        if key == 'count_covariance':
            return self.count_covariance
        if key == 'n_events':
            return self.n_events
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
        raise KeyError(f'Unsupported key "{key}" for correlomatrix_detector.get().')

    def connect(self):
        return None

    def flush(self):
        return {
            'covariance': self.covariance,
            'count_covariance': self.count_covariance,
            'n_events': self.n_events,
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
        weights: ArrayLike = None,
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
        weight_arr = self._to_float_array(weights, name='weights', default=1.0, size=n_items)

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

            sender = int(port_arr[i])
            if sender < 0 or sender >= calib.n_channels:
                raise ValueError(f'Unknown receptor_type {sender} for correlomatrix_detector.')

            stamp_step = int(stamp_arr[i])
            if not self._is_active(stamp_step, calib.t_min_steps, calib.t_max_steps):
                continue

            self._handle_event(
                sender=sender,
                stamp_step=stamp_step,
                weight=float(weight_arr[i]),
                multiplicity=multiplicity,
                calib=calib,
            )

        return self.flush()

    def _handle_event(
        self,
        sender: int,
        stamp_step: int,
        weight: float,
        multiplicity: int,
        calib: _Calibration,
    ):
        own_weight = float(multiplicity) * float(weight)

        spike_i = _Spike(
            timestep=stamp_step,
            weight=own_weight,
            receptor_channel=sender,
        )

        insert_pos = len(self._incoming)
        for idx, old_spike in enumerate(self._incoming):
            if old_spike.timestep > stamp_step:
                insert_pos = idx
                break
        self._incoming.insert(insert_pos, spike_i)

        while len(self._incoming) > 0:
            dt_steps = stamp_step - self._incoming[0].timestep
            if dt_steps >= calib.tau_edge_steps + calib.min_delay_steps:
                self._incoming.popleft()
            else:
                break

        stamp_ms = float(stamp_step) * calib.dt_ms
        if not self._is_in_count_window(stamp_ms, calib.tstart_ms, calib.tstop_ms):
            return

        self._n_events[sender] += 1
        n_bins = self._covariance.shape[2]

        for spike_j in self._incoming:
            other = spike_j.receptor_channel
            diff_steps = stamp_step - spike_j.timestep
            abs_diff_steps = abs(diff_steps)

            if stamp_step < spike_j.timestep:
                sender_ind = other
                other_ind = sender
            else:
                sender_ind = sender
                other_ind = other

            if sender_ind <= other_ind:
                bin_index = int(
                    -math.floor(
                        (0.5 * calib.delta_tau_steps - abs_diff_steps)
                        / calib.delta_tau_steps
                    )
                )
            else:
                bin_index = int(
                    math.floor(
                        (0.5 * calib.delta_tau_steps + abs_diff_steps)
                        / calib.delta_tau_steps
                    )
                )

            if bin_index >= n_bins:
                continue

            contribution = own_weight * spike_j.weight
            self._covariance[sender_ind, other_ind, bin_index] += contribution
            self._count_covariance[sender_ind, other_ind, bin_index] += multiplicity

            if bin_index == 0 and (diff_steps != 0 or other != sender):
                self._covariance[other_ind, sender_ind, bin_index] += contribution
                self._count_covariance[other_ind, sender_ind, bin_index] += multiplicity

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

        if self._calib is None:
            self._n_events = np.zeros((0,), dtype=np.int64)
            self._covariance = np.zeros((0, 0, 0), dtype=np.float64)
            self._count_covariance = np.zeros((0, 0, 0), dtype=np.int64)
            return

        n_channels = int(self._calib.n_channels)
        n_bins = int(self._calib.n_bins)

        self._n_events = np.zeros((n_channels,), dtype=np.int64)
        self._covariance = np.zeros((n_channels, n_channels, n_bins), dtype=np.float64)
        self._count_covariance = np.zeros((n_channels, n_channels, n_bins), dtype=np.int64)

    def _compute_calibration(self, dt) -> _Calibration:
        dt_ms = self._to_ms_scalar(dt, name='dt')
        if dt_ms <= 0.0:
            raise ValueError('Simulation resolution dt must be positive.')

        start_steps = self._to_step_count(self.start, dt_ms, 'start')
        stop_value = math.inf if self.stop is None else self.stop
        stop_steps = self._to_step_count(stop_value, dt_ms, 'stop', allow_inf=True)
        if not math.isinf(stop_steps) and stop_steps < start_steps:
            raise ValueError('stop >= start required.')

        origin_steps = self._to_step_count(self.origin, dt_ms, 'origin')
        t_min_steps = origin_steps + start_steps
        t_max_steps = math.inf if math.isinf(stop_steps) else origin_steps + stop_steps

        if self.delta_tau is None:
            delta_tau_ms = 5.0 * dt_ms
        else:
            delta_tau_ms = self._to_ms_scalar(self.delta_tau, name='delta_tau')

        if not math.isfinite(delta_tau_ms) or delta_tau_ms <= 0.0:
            raise ValueError('delta_tau must be positive and finite.')

        delta_tau_steps = self._to_step_count(delta_tau_ms, dt_ms, 'delta_tau')
        if delta_tau_steps % 2 != 1:
            raise ValueError('/delta_tau must be odd multiple of resolution.')

        if self.tau_max is None:
            tau_max_ms = 10.0 * delta_tau_ms
        else:
            tau_max_ms = self._to_ms_scalar(self.tau_max, name='tau_max')

        if not math.isfinite(tau_max_ms) or tau_max_ms < 0.0:
            raise ValueError('tau_max must be finite and non-negative.')

        tau_max_steps = self._to_step_count(tau_max_ms, dt_ms, 'tau_max')
        if tau_max_steps % delta_tau_steps != 0:
            raise ValueError('tau_max must be a multiple of delta_tau.')

        tstart_ms = self._to_ms_scalar(self.Tstart, name='Tstart', allow_inf=True)
        tstop_value = math.inf if self.Tstop is None else self.Tstop
        tstop_ms = self._to_ms_scalar(tstop_value, name='Tstop', allow_inf=True)

        n_channels = self._to_int_scalar(self.N_channels, name='N_channels')
        if n_channels < 1:
            raise ValueError('/N_channels can only be larger than zero.')

        n_bins = int(1 + tau_max_steps // delta_tau_steps)
        min_delay_steps = 1

        signature = (
            float(dt_ms),
            int(start_steps),
            float(stop_steps),
            int(origin_steps),
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
            start_step=int(start_steps),
            stop_step=float(stop_steps),
            origin_step=int(origin_steps),
            t_min_steps=int(t_min_steps),
            t_max_steps=float(t_max_steps),
            delta_tau_ms=float(delta_tau_ms),
            delta_tau_steps=int(delta_tau_steps),
            tau_max_ms=float(tau_max_ms),
            tau_max_steps=int(tau_max_steps),
            tau_edge_steps=float(tau_max_steps) + 0.5 * float(delta_tau_steps),
            tstart_ms=float(tstart_ms),
            tstop_ms=float(tstop_ms),
            n_channels=int(n_channels),
            n_bins=int(n_bins),
            min_delay_steps=int(min_delay_steps),
            signature=signature,
        )

    @staticmethod
    def _is_in_count_window(stamp_ms: float, tstart_ms: float, tstop_ms: float) -> bool:
        return (stamp_ms >= tstart_ms - 1e-12) and (stamp_ms <= tstop_ms + 1e-12)

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
        unit=None,
    ) -> np.ndarray:
        if x is None:
            if default is None:
                raise ValueError(f'{name} cannot be None.')
            arr = np.asarray([default], dtype=np.float64)
        else:
            if unit is not None and isinstance(x, u.Quantity):
                x = x / unit
            elif isinstance(x, u.Quantity):
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
