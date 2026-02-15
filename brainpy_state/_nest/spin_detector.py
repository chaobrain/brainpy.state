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
    'spin_detector',
]


@dataclass
class _StepCalibration:
    dt_ms: float
    t_min_steps: int
    t_max_steps: float


class spin_detector(brainstate.nn.Dynamics):
    r"""NEST-compatible detector for binary state decoding from spikes.

    Short Description
    -----------------
    ``spin_detector`` decodes binary states from incoming spikes and records
    ``senders``, ``times``, and ``state`` events.

    Description
    -----------
    This class re-implements core behavior of NEST ``spin_detector``
    (``models/spin_detector.{h,cpp}`` and ``nestkernel/recording_device.*``):

    - Decoding rule follows NEST exactly:

      - one spike (multiplicity ``1``) decodes state ``0``,
      - two spikes at the same sender and timestamp decode state ``1``.

    - Decoder order and write ordering match ``spin_detector::handle``:

      - potential 0->1 revision is applied before writing buffered state,
      - buffered state is written before processing the current event body,
      - multiplicity ``2`` writes state ``1`` immediately.

    - Recording window is
      :math:`(\mathrm{origin}+\mathrm{start},\;\mathrm{origin}+\mathrm{stop}]`
      in simulation steps (start exclusive, stop inclusive).
    - ``n_events`` can only be set to ``0`` to clear memory.

    Update Semantics
    ----------------
    In NEST, ``handle(SpikeEvent&)`` performs decoding and ``update(...)``
    finalizes any buffered event for that cycle. This implementation mirrors
    that sequence inside :meth:`update`: it decodes all provided events in
    order, then finalizes a remaining buffered event.

    The detector expects events at simulation time ``t`` to carry stamp
    ``t + dt`` by default. For precise times, provide ``offsets`` (ms) so that

    .. math::

       t_{event} = t_{stamp} - \delta.

    Parameters
    ----------
    in_size : int, optional
        Device batch size. Defaults to ``1``.
    start : Quantity[ms], optional
        Recording window start relative to ``origin``. Defaults to
        ``0.0 * u.ms``.
    stop : Quantity[ms] or None, optional
        Recording window stop relative to ``origin``. ``None`` means +infinity.
        Defaults to ``None``.
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
    - Input events are processed in the order supplied to :meth:`update`,
      matching the sequential decode behavior in NEST ``handle``.
    - Connection weight and delay are not part of decoding, consistent with
      NEST ``spin_detector``.

    References
    ----------
    .. [1] NEST Simulator, ``spin_detector`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/spin_detector.html
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        start: ArrayLike = 0.0 * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0.0 * u.ms,
        time_in_steps: bool = False,
        frozen: bool = False,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        if frozen:
            raise ValueError('spin_detector cannot be frozen.')

        self.start = start
        self.stop = stop
        self.origin = origin

        self._time_in_steps = bool(time_in_steps)
        self._has_been_simulated = False

        self._clear_last_event()
        self.clear_events()

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
            'state': np.asarray(self._events_state, dtype=np.int64),
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
        raise KeyError(f'Unsupported key "{key}" for spin_detector.get().')

    def clear_events(self):
        self._events_senders: list[int] = []
        self._events_state: list[int] = []
        self._events_times_ms: list[float] = []
        self._events_times_steps: list[int] = []
        self._events_offsets: list[float] = []

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self._clear_last_event()
        self.clear_events()

    def connect(self):
        return None

    def flush(self):
        return self.events

    def update(
        self,
        spikes: ArrayLike = None,
        senders: ArrayLike = None,
        offsets: ArrayLike = None,
        multiplicities: ArrayLike = None,
        stamp_steps: ArrayLike = None,
    ):
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        calib = self._get_step_calibration(dt)

        step_now = self._time_to_step(t, calib.dt_ms)
        self._has_been_simulated = True

        if spikes is not None:
            spike_arr = self._to_float_array(spikes, name='spikes')
            n_items = spike_arr.size

            if n_items > 0:
                sender_arr = self._to_int_array(senders, name='senders', default=1, size=n_items)
                offset_arr = self._to_float_array(offsets, name='offsets', default=0.0, size=n_items, unit=u.ms)

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

                    stamp_step = int(stamp_arr[i])
                    if not self._is_active(stamp_step, calib.t_min_steps, calib.t_max_steps):
                        continue

                    self._handle_event(
                        sender=int(sender_arr[i]),
                        stamp_step=stamp_step,
                        offset_ms=float(offset_arr[i]),
                        multiplicity=multiplicity,
                        dt_ms=calib.dt_ms,
                    )

        self._flush_last_event(dt_ms=calib.dt_ms)
        return self.events

    def _handle_event(
        self,
        sender: int,
        stamp_step: int,
        offset_ms: float,
        multiplicity: int,
        dt_ms: float,
    ):
        if multiplicity == 1 and sender == self._last_sender and stamp_step == self._last_stamp_step:
            self._last_state = 1

        if self._last_sender != 0:
            self._write_event(
                sender=self._last_sender,
                stamp_step=self._last_stamp_step,
                offset_ms=self._last_offset_ms,
                state=self._last_state,
                dt_ms=dt_ms,
            )

        if multiplicity == 2:
            self._write_event(
                sender=sender,
                stamp_step=stamp_step,
                offset_ms=offset_ms,
                state=1,
                dt_ms=dt_ms,
            )
            self._clear_last_event()
        else:
            if self._last_sender == 0:
                self._last_sender = sender
                self._last_stamp_step = stamp_step
                self._last_offset_ms = offset_ms
                self._last_state = 0
            else:
                self._clear_last_event()

    def _flush_last_event(self, dt_ms: float):
        if self._last_sender != 0:
            self._write_event(
                sender=self._last_sender,
                stamp_step=self._last_stamp_step,
                offset_ms=self._last_offset_ms,
                state=self._last_state,
                dt_ms=dt_ms,
            )
            self._clear_last_event()

    def _write_event(
        self,
        sender: int,
        stamp_step: int,
        offset_ms: float,
        state: int,
        dt_ms: float,
    ):
        self._events_senders.append(int(sender))
        self._events_state.append(int(state))
        if self.time_in_steps:
            self._events_times_steps.append(int(stamp_step))
            self._events_offsets.append(float(offset_ms))
        else:
            self._events_times_ms.append(float(stamp_step) * dt_ms - float(offset_ms))

    def _clear_last_event(self):
        self._last_sender = 0
        self._last_stamp_step = 0
        self._last_offset_ms = 0.0
        self._last_state = 0

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
