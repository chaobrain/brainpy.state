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
from dataclasses import dataclass

import brainstate
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'volume_transmitter',
]


@dataclass(frozen=True)
class spikecounter:
    """Entry in neuromodulatory spike history.

    Attributes
    ----------
    spike_time : float
        Spike time in ms (grid stamp time).
    multiplicity : float
        Summed multiplicity of all spikes at ``spike_time``.
    """

    spike_time: float
    multiplicity: float


@dataclass(frozen=True)
class _StepCalibration:
    dt_ms: float
    min_delay_steps: int
    delivery_period_steps: int


class volume_transmitter(brainstate.nn.Dynamics):
    r"""NEST-compatible ``volume_transmitter`` support device.

    Short Description
    -----------------
    ``volume_transmitter`` collects neuromodulatory spikes and periodically
    exposes their history for dopamine-modulated synapses.

    Description
    -----------
    This class mirrors NEST ``models/volume_transmitter.{h,cpp}`` update
    ordering and spike-history semantics:

    1. A ring-buffer-like accumulator stores incoming spike multiplicities by
       delivery step.
    2. At each simulation step, multiplicity scheduled for the current stamp
       step is appended to the spike-history vector as
       ``spikecounter(t_spike, multiplicity)`` with
       :math:`t_{spike} = (step + 1)\,dt`.
    3. Periodically, at
       ``delivery_period = deliver_interval * d_min_steps``, the complete spike
       history is "delivered" to target synapses.
    4. After delivery, the history is cleared and replaced by one pseudo spike
       ``(t_trig, 0.0)``, matching NEST behavior used by
       ``stdp_dopamine_synapse``.

    As in NEST, an initial pseudo spike ``(0.0, 0.0)`` exists immediately after
    :meth:`init_state`.

    Delivery Period
    ...............
    In NEST, delivery period is ``deliver_interval * d_min`` where ``d_min`` is
    the global minimal synaptic delay. Since this standalone implementation has
    no kernel-level delay manager, ``d_min`` is provided explicitly by
    ``min_delay``.

    Parameters
    ----------
    in_size : int, optional
        Device size. Defaults to ``1``.
    deliver_interval : int, optional
        Number of ``d_min`` intervals between deliveries. Must be >= 1.
        Defaults to ``1``.
    min_delay : Quantity[ms], optional
        Effective global minimal synaptic delay used to compute delivery
        periodicity. Must be a positive multiple of ``dt``. Defaults to
        ``1.0 * u.ms``.
    name : str, optional
        Module name.

    State Access
    ------------
    - :meth:`deliver_spikes` returns current spike-history vector.
    - ``last_delivery_spikes`` stores spike-history content delivered at the
      most recent trigger.
    - ``last_delivery_time`` stores most recent trigger time in ms.

    Notes
    -----
    - ``update`` accepts optional event payloads and aggregates multiplicities
      exactly by delivery step, as NEST's internal ring buffer does.
    - ``handles_test_event`` accepts only receptor type ``0``, matching NEST.
    - ``set_local_device_id`` / ``get_local_device_id`` are provided for
      compatibility with NEST's device duplication logic.

    References
    ----------
    .. [1] NEST Simulator, ``volume_transmitter`` model.
           https://github.com/nest/nest-simulator/blob/master/models/volume_transmitter.cpp
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        deliver_interval: ArrayLike = 1,
        min_delay: ArrayLike = 1.0 * u.ms,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.deliver_interval = int(self._to_int_scalar(deliver_interval, name='deliver_interval'))
        if self.deliver_interval < 1:
            raise ValueError('deliver_interval must be >= 1.')

        self.min_delay = min_delay
        self._local_device_id = 0

        self._pending_multiplicities: dict[int, float] = {}
        self._spikecounter: list[spikecounter] = []
        self._last_delivery_spikes: tuple[spikecounter, ...] = ()
        self._last_delivery_time_ms: float = 0.0
        self._delivery_count = 0

        self.init_state()

    @property
    def local_device_id(self) -> int:
        return int(self._local_device_id)

    @property
    def last_delivery_time(self) -> float:
        return float(self._last_delivery_time_ms)

    @property
    def last_delivery_spikes(self) -> tuple[spikecounter, ...]:
        return tuple(self._last_delivery_spikes)

    @property
    def n_deliveries(self) -> int:
        return int(self._delivery_count)

    def set_local_device_id(self, ldid: ArrayLike):
        self._local_device_id = int(self._to_int_scalar(ldid, name='local_device_id'))

    def get_local_device_id(self) -> int:
        return int(self._local_device_id)

    def handles_test_event(self, receptor_type: ArrayLike) -> int:
        r = int(self._to_int_scalar(receptor_type, name='receptor_type'))
        if r != 0:
            raise ValueError(f'Unknown receptor_type {r} for volume_transmitter.')
        return 0

    def deliver_spikes(self) -> tuple[spikecounter, ...]:
        return tuple(self._spikecounter)

    def get(self, key: str = 'deliver_interval'):
        if key == 'deliver_interval':
            return int(self.deliver_interval)
        if key == 'min_delay':
            return self.min_delay
        if key == 'local_device_id':
            return int(self._local_device_id)
        if key == 'spike_history':
            return self.deliver_spikes()
        if key == 'last_delivery_spikes':
            return self.last_delivery_spikes
        if key == 'last_delivery_time':
            return self.last_delivery_time
        if key == 'n_deliveries':
            return self.n_deliveries
        raise KeyError(f'Unsupported key "{key}" for volume_transmitter.get().')

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self._pending_multiplicities.clear()
        self._spikecounter = [spikecounter(0.0, 0.0)]
        self._last_delivery_spikes = ()
        self._last_delivery_time_ms = 0.0
        self._delivery_count = 0

    def connect(self):
        return None

    def flush(self):
        return {
            'triggered': False,
            't_trig': None,
            'delivered_spikes': (),
            'spike_history': self.deliver_spikes(),
        }

    def update(
        self,
        spikes: ArrayLike = None,
        multiplicities: ArrayLike = None,
        stamp_steps: ArrayLike = None,
    ):
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        calib = self._get_step_calibration(dt)

        step_now = self._time_to_step(t, calib.dt_ms)
        stamp_now = step_now + 1

        self._schedule_incoming(
            spikes=spikes,
            multiplicities=multiplicities,
            stamp_steps=stamp_steps,
            stamp_now=stamp_now,
        )

        multiplicity = float(self._pending_multiplicities.pop(stamp_now, 0.0))
        if multiplicity > 0.0:
            t_spike = float(stamp_now) * calib.dt_ms
            self._spikecounter.append(spikecounter(t_spike, multiplicity))

        triggered = (stamp_now % calib.delivery_period_steps) == 0
        delivered_spikes: tuple[spikecounter, ...] = ()
        t_trig = None

        if triggered:
            t_trig = float(stamp_now) * calib.dt_ms
            if len(self._spikecounter) > 0:
                delivered_spikes = tuple(self._spikecounter)
                self._last_delivery_spikes = delivered_spikes
                self._last_delivery_time_ms = t_trig
                self._delivery_count += 1

            self._spikecounter.clear()
            self._spikecounter.append(spikecounter(t_trig, 0.0))

        return {
            'triggered': bool(triggered),
            't_trig': t_trig,
            'delivered_spikes': delivered_spikes,
            'spike_history': self.deliver_spikes(),
        }

    def _schedule_incoming(
        self,
        spikes: ArrayLike,
        multiplicities: ArrayLike,
        stamp_steps: ArrayLike,
        stamp_now: int,
    ):
        if spikes is None:
            return

        spike_arr = self._to_float_array(spikes, name='spikes')
        if spike_arr.size == 0:
            return

        n_items = spike_arr.size

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
            stamp_arr = np.full((n_items,), stamp_now, dtype=np.int64)
        else:
            stamp_arr = self._to_int_array(stamp_steps, name='stamp_steps', size=n_items)
            if np.any(stamp_arr < stamp_now):
                raise ValueError('stamp_steps must be >= current delivery step.')

        for i in range(n_items):
            c = int(counts[i])
            if c <= 0:
                continue
            s = int(stamp_arr[i])
            self._pending_multiplicities[s] = float(self._pending_multiplicities.get(s, 0.0) + c)

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
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be a scalar integer value.')
        val = float(arr[0])
        ival = int(np.rint(val))
        if not np.isclose(val, ival, atol=1e-12, rtol=1e-12):
            raise ValueError(f'{name} must be an integer value.')
        return ival

    @classmethod
    def _to_step_count(
        cls,
        value,
        dt_ms: float,
        name: str,
    ) -> int:
        ms = cls._to_ms_scalar(value, name=name)
        steps_f = ms / dt_ms
        steps_i = int(np.rint(steps_f))
        if not np.isclose(steps_f, steps_i, atol=1e-12, rtol=1e-12):
            raise ValueError(f'{name} must be a multiple of the simulation resolution.')
        return steps_i

    def _get_step_calibration(self, dt) -> _StepCalibration:
        dt_ms = self._to_ms_scalar(dt, name='dt')
        if dt_ms <= 0.0:
            raise ValueError('Simulation resolution dt must be positive.')

        min_delay_steps = self._to_step_count(self.min_delay, dt_ms=dt_ms, name='min_delay')
        if min_delay_steps < 1:
            raise ValueError('min_delay must be at least one simulation step.')

        period = int(self.deliver_interval) * int(min_delay_steps)
        if period < 1:
            raise ValueError('deliver_interval * min_delay_steps must be >= 1.')

        return _StepCalibration(
            dt_ms=dt_ms,
            min_delay_steps=min_delay_steps,
            delivery_period_steps=period,
        )

    def _time_to_step(self, t, dt_ms: float) -> int:
        t_ms = self._to_ms_scalar(t, name='t')
        steps_f = t_ms / dt_ms
        steps_i = int(np.rint(steps_f))
        if not np.isclose(steps_f, steps_i, atol=1e-12, rtol=1e-12):
            raise ValueError('Current simulation time t must be aligned to the simulation grid.')
        return steps_i

    @staticmethod
    def _to_float_array(x, name: str) -> np.ndarray:
        if isinstance(x, u.Quantity):
            x = u.get_mantissa(x)
        arr = np.asarray(u.math.asarray(x), dtype=np.float64).reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f'{name} must be a scalar or 1D array.')
        return arr

    @classmethod
    def _to_int_array(
        cls,
        x,
        name: str,
        size: int = None,
    ) -> np.ndarray:
        arr = cls._to_float_array(x, name=name)
        if size is not None and arr.size != size:
            raise ValueError(f'{name} must have size {size}, got {arr.size}.')
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded, atol=1e-12, rtol=1e-12):
            raise ValueError(f'{name} must contain integer values.')
        return rounded.astype(np.int64)
