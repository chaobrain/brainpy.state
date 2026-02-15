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
    'weight_recorder',
]


@dataclass
class _StepCalibration:
    dt_ms: float
    t_min_steps: int
    t_max_steps: float


class weight_recorder(brainstate.nn.Dynamics):
    r"""NEST-compatible recorder for synaptic weights.

    Short Description
    -----------------
    ``weight_recorder`` stores one event for each transmitted synaptic event
    observed by a plastic synapse model, including sender/target IDs and weight.

    Description
    -----------
    This class re-implements core behavior of NEST ``weight_recorder``
    (``models/weight_recorder.{h,cpp}`` and
    ``nestkernel/{recording_device.*,connector_base_impl.h}``):

    - Records one event per transmitted synaptic event.
    - Stores ``weights`` plus event metadata ``senders``, ``targets``,
      ``receptors`` (rport), and ``ports``.
    - Uses recorder activity window
      :math:`(\mathrm{origin}+\mathrm{start},\;\mathrm{origin}+\mathrm{stop}]`
      in simulation steps (start exclusive, stop inclusive).
    - Supports optional sender/target filters via ``senders`` and ``targets``.
    - ``n_events`` can only be set to ``0`` to clear memory.

    Update Semantics
    ----------------
    In NEST, weight recorder events are created when a synaptic event is
    transmitted, and the event stamp is copied from the original synaptic event.
    Here, :meth:`update` accepts batched event payloads directly.

    If ``stamp_steps`` is omitted, events are stamped at ``t + dt`` (the same
    per-step convention used by NEST spike events generated in ``(t, t+dt]``).

    If an event offset ``delta`` (ms) is provided, reported physical time is

    .. math::

       t_{event} = t_{stamp} - \delta,

    where :math:`t_{stamp}` is the event stamp in grid time.

    Parameters
    ----------
    in_size : int, optional
        Device batch size. Defaults to 1.
    senders : array-like of int, optional
        Allowed sender node IDs filter. Empty means no sender filtering.
    targets : array-like of int, optional
        Allowed target node IDs filter. Empty means no target filtering.
    start : Quantity[ms], optional
        Recording window start relative to ``origin``. Defaults to
        ``0.0 * u.ms``.
    stop : Quantity[ms] or None, optional
        Recording window stop relative to ``origin``. ``None`` means +infinity.
    origin : Quantity[ms], optional
        Recording window origin shift. Defaults to ``0.0 * u.ms``.
    time_in_steps : bool, optional
        If ``False`` (default), ``events['times']`` are float ms.
        If ``True``, ``events['times']`` are integer step stamps and
        ``events['offsets']`` are included.
    frozen : bool, optional
        Kept for NEST API compatibility. ``True`` is rejected.
    name : str, optional
        Module name.

    Notes
    -----
    - Filters are applied as in NEST ``weight_recorder::handle``:
      sender filter and target filter are checked before writing.
    - This class records events passed to :meth:`update`; it does not inspect
      synapse objects directly.

    References
    ----------
    .. [1] NEST Simulator, ``weight_recorder`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/weight_recorder.html
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        senders: ArrayLike = (),
        targets: ArrayLike = (),
        start: ArrayLike = 0.0 * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0.0 * u.ms,
        time_in_steps: bool = False,
        frozen: bool = False,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        if frozen:
            raise ValueError('weight_recorder cannot be frozen.')

        self.start = start
        self.stop = stop
        self.origin = origin

        self._time_in_steps = bool(time_in_steps)
        self._has_been_simulated = False

        self._senders_filter: tuple[int, ...] = ()
        self._targets_filter: tuple[int, ...] = ()
        self._senders_filter_set: set[int] | None = None
        self._targets_filter_set: set[int] | None = None

        self.senders = senders
        self.targets = targets

        self.clear_events()

    @property
    def senders(self) -> tuple[int, ...]:
        return self._senders_filter

    @senders.setter
    def senders(self, value):
        ids = self._normalize_filter(value, name='senders')
        self._senders_filter = ids
        self._senders_filter_set = set(ids) if len(ids) > 0 else None

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets_filter

    @targets.setter
    def targets(self, value):
        ids = self._normalize_filter(value, name='targets')
        self._targets_filter = ids
        self._targets_filter_set = set(ids) if len(ids) > 0 else None

    @property
    def time_in_steps(self) -> bool:
        return self._time_in_steps

    @time_in_steps.setter
    def time_in_steps(self, value: bool):
        if self._has_been_simulated:
            raise ValueError('Property time_in_steps cannot be set after Simulate has been called.')
        self._time_in_steps = bool(value)

    @property
    def n_events(self) -> int:
        return len(self._events_senders)

    @n_events.setter
    def n_events(self, value: int):
        value = int(value)
        if value != 0:
            raise ValueError('Property n_events can only be set to 0 (which clears all stored events).')
        self.clear_events()

    @property
    def events(self) -> dict[str, np.ndarray]:
        out = {
            'senders': np.asarray(self._events_senders, dtype=np.int64),
            'targets': np.asarray(self._events_targets, dtype=np.int64),
            'weights': np.asarray(self._events_weights, dtype=np.float64),
            'receptors': np.asarray(self._events_receptors, dtype=np.int64),
            'ports': np.asarray(self._events_ports, dtype=np.int64),
        }
        if self.time_in_steps:
            out['times'] = np.asarray(self._events_times_steps, dtype=np.int64)
            out['offsets'] = np.asarray(self._events_offsets, dtype=np.float64)
        else:
            out['times'] = np.asarray(self._events_times_ms, dtype=np.float64)
        return out

    def get(self, key: str = 'events'):
        if key == 'events':
            return self.events
        if key == 'n_events':
            return self.n_events
        if key == 'time_in_steps':
            return self.time_in_steps
        if key == 'senders':
            return np.asarray(self.senders, dtype=np.int64)
        if key == 'targets':
            return np.asarray(self.targets, dtype=np.int64)
        raise KeyError(f'Unsupported key "{key}" for weight_recorder.get().')

    def clear_events(self):
        self._events_senders: list[int] = []
        self._events_targets: list[int] = []
        self._events_weights: list[float] = []
        self._events_receptors: list[int] = []
        self._events_ports: list[int] = []
        self._events_times_ms: list[float] = []
        self._events_times_steps: list[int] = []
        self._events_offsets: list[float] = []

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self.clear_events()

    def connect(self):
        return None

    def flush(self):
        return self.events

    def update(
        self,
        weights: ArrayLike = None,
        senders: ArrayLike = None,
        targets: ArrayLike = None,
        receptors: ArrayLike = None,
        ports: ArrayLike = None,
        offsets: ArrayLike = None,
        stamp_steps: ArrayLike = None,
    ):
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        calib = self._get_step_calibration(dt)

        step_now = self._time_to_step(t, calib.dt_ms)

        self._has_been_simulated = True

        if weights is None:
            return self.events

        weight_arr = self._to_float_array(weights, name='weights')
        if weight_arr.size == 0:
            return self.events

        n_items = weight_arr.size

        sender_arr = self._to_int_array(senders, name='senders', default=1, size=n_items)
        target_arr = self._to_int_array(targets, name='targets', default=1, size=n_items)
        receptor_arr = self._to_int_array(receptors, name='receptors', default=0, size=n_items)
        port_arr = self._to_int_array(ports, name='ports', default=-1, size=n_items)
        offset_arr = self._to_float_array(offsets, name='offsets', default=0.0, size=n_items, unit=u.ms)

        if stamp_steps is None:
            stamp_arr = np.full((n_items,), step_now + 1, dtype=np.int64)
        else:
            stamp_arr = self._to_int_array(stamp_steps, name='stamp_steps', size=n_items)

        active = self._is_active_steps(stamp_arr, calib.t_min_steps, calib.t_max_steps)
        if self._senders_filter_set is not None:
            active &= np.asarray([sid in self._senders_filter_set for sid in sender_arr], dtype=bool)
        if self._targets_filter_set is not None:
            active &= np.asarray([tid in self._targets_filter_set for tid in target_arr], dtype=bool)

        if not np.any(active):
            return self.events

        w = weight_arr[active]
        s = sender_arr[active]
        tarr = target_arr[active]
        r = receptor_arr[active]
        p = port_arr[active]
        o = offset_arr[active]
        stamp = stamp_arr[active]

        self._events_weights.extend(w.tolist())
        self._events_senders.extend(s.tolist())
        self._events_targets.extend(tarr.tolist())
        self._events_receptors.extend(r.tolist())
        self._events_ports.extend(p.tolist())

        if self.time_in_steps:
            self._events_times_steps.extend(stamp.tolist())
            self._events_offsets.extend(o.tolist())
        else:
            time_ms = stamp.astype(np.float64) * calib.dt_ms - o
            self._events_times_ms.extend(time_ms.tolist())

        return self.events

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

    def _get_step_calibration(self, dt) -> _StepCalibration:
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

        return _StepCalibration(
            dt_ms=dt_ms,
            t_min_steps=t_min_steps,
            t_max_steps=t_max_steps,
        )

    def _time_to_step(self, t, dt_ms: float) -> int:
        t_ms = self._to_ms_scalar(t, name='t')
        steps_f = t_ms / dt_ms
        steps_i = int(np.rint(steps_f))
        if not np.isclose(steps_f, steps_i, atol=1e-12, rtol=1e-12):
            raise ValueError('Current simulation time t must be aligned to the simulation grid.')
        return steps_i

    @staticmethod
    def _is_active_steps(stamp_steps: np.ndarray, t_min_steps: int, t_max_steps: float) -> np.ndarray:
        active = stamp_steps > t_min_steps
        if not math.isinf(t_max_steps):
            active &= stamp_steps <= int(t_max_steps)
        return active

    @staticmethod
    def _normalize_filter(value, name: str) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, (tuple, list)) and len(value) == 0:
            return ()
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return ()
        if np.any(arr <= 0):
            raise ValueError(f'{name} must contain positive node IDs.')
        return tuple(int(v) for v in arr)

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
            raise ValueError(f'{name} size ({arr.size}) does not match weights size ({size}).')
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
            raise ValueError(f'{name} size ({arr.size}) does not match weights size ({size}).')
        return arr.astype(np.int64, copy=False)
