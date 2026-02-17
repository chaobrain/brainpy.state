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
from typing import Mapping, Sequence

import brainstate
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'multimeter',
]


@dataclass
class _PendingSample:
    stamp_step: int
    senders: np.ndarray
    values: dict[str, np.ndarray]


@dataclass
class _StepCalibration:
    dt_ms: float
    interval_steps: int
    offset_steps: int
    t_min_steps: int
    t_max_steps: float


class multimeter(brainstate.nn.Dynamics):
    r"""NEST-compatible analog recorder for neuron/device state variables.

    ``multimeter`` records analog state samples from connected targets into an
    in-memory ``events`` dictionary compatible with NEST ``multimeter``
    semantics.  The NEST device-level timing model is reproduced while
    exposing a Python update API:

    - Sampling times are constrained to a step-grid lattice defined by
      ``interval`` and ``offset``.
    - Recording is gated by a window
      :math:`(\mathrm{origin}+\mathrm{start},\;\mathrm{origin}+\mathrm{stop}]`
      (start exclusive, stop inclusive) evaluated in simulation steps.
    - Samples are enqueued at the current step and emitted on the next
      :meth:`update` call (or immediately by :meth:`flush`), reproducing the
      one-step request/reply lag used by NEST multimeters.

    **1. Step-Grid Sampling Model**

    Let :math:`dt` be simulation resolution in ms, and let step index
    :math:`n = \mathrm{round}(t/dt)`. During :meth:`update`, sampled values are
    stamped at

    .. math::

       s = n + 1.

    Define integer grid parameters

    .. math::

       m = \frac{\mathrm{interval}}{dt}, \qquad
       o = \frac{\mathrm{offset}}{dt}.

    A sample is enqueued iff :math:`s` lies on the lattice:

    .. math::

       s \equiv 0 \ (\mathrm{mod}\ m) \quad \text{if}\ o = 0, \qquad
       s \equiv o \ (\mathrm{mod}\ m),\ s \ge o \quad \text{if}\ o > 0.

    Both ``interval`` and ``offset`` must be exact integer multiples of
    ``dt`` (verified to within ``1e-12`` tolerance in floating conversion).

    **2. Active Window and Delivery Lag**

    With :math:`s_\min = (\mathrm{origin}+\mathrm{start})/dt` and
    :math:`s_\max = (\mathrm{origin}+\mathrm{stop})/dt` (or :math:`+\infty`
    when ``stop`` is ``None``), a pending sample is written to ``events``
    only when

    .. math::

       s > s_\min \quad \land \quad s \le s_\max.

    Because pending samples are emitted before new sampling in each
    :meth:`update`, values observed at step :math:`n` become visible in
    ``events`` at step :math:`n+1` unless :meth:`flush` is called.

    **3. Payload Normalization and Shape Constraints**

    For each requested recordable ``k`` in ``record_from``, ``data[k]`` is
    converted to ``np.float64`` and flattened to shape ``(N,)``. All
    recordables must share the same ``N``; scalar payloads (size 1) are
    broadcast to ``(N,)`` when another recordable defines ``N > 1``.
    ``senders`` is converted to ``np.int64`` with the same broadcast rule,
    defaulting to ones when omitted. Stored event arrays are one-dimensional
    with length equal to the total number of emitted samples across all steps.

    **4. Computational Implications**

    Per :meth:`update` call with payload size ``N`` and
    ``R = len(record_from)``, enqueue work is :math:`O(RN)`. Pending
    emission is linear in the number of buffered items and the appended event
    count. Memory usage grows linearly with total emitted events for
    ``times``, ``senders``, and each requested recordable channel.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape argument consumed by :class:`brainstate.nn.Dynamics`.
        This recorder is stateful and returns event dictionaries; ``in_size``
        is retained for API consistency only. Default is ``1``.
    record_from : Sequence[str], optional
        Ordered names of recordable state variables expected as keys in
        ``data`` during :meth:`update`. If empty, incoming payloads are
        silently ignored and no values are stored. Default is ``()``.
    interval : brainunit.Quantity or float, optional
        Scalar sampling interval in time units convertible to ms
        (typically ``u.ms``). Must satisfy ``interval >= dt`` and be an exact
        integer multiple of ``dt`` (checked to within ``1e-12`` tolerance).
        Default is ``1.0 * u.ms``.
    offset : brainunit.Quantity or float, optional
        Scalar phase offset of the sampling lattice relative to the simulation
        origin, convertible to ms.  Must be ``0.0`` or a positive integer
        multiple of ``dt``; non-zero offsets shift the first sample to step
        :math:`o` and every :math:`m`-th step thereafter.
        Default is ``0.0 * u.ms``.
    start : brainunit.Quantity or float, optional
        Scalar exclusive lower bound of the recording window relative to
        ``origin``, convertible to ms.  A pending sample at stamp step
        :math:`s` is discarded when :math:`s \le s_\min`.
        Default is ``0.0 * u.ms``.
    stop : brainunit.Quantity, float, or None, optional
        Scalar inclusive upper bound of the recording window relative to
        ``origin``, convertible to ms.  Must satisfy ``stop >= start`` when
        not ``None``.  ``None`` means no upper bound
        (:math:`s_\max = +\infty`). Default is ``None``.
    origin : brainunit.Quantity or float, optional
        Scalar global time-origin shift added to both ``start`` and ``stop``
        when constructing the active window, convertible to ms.  Shifting the
        origin displaces the entire recording window without changing its
        duration. Default is ``0.0 * u.ms``.
    time_in_steps : bool, optional
        Controls the unit of ``events['times']``. If ``False``, timestamps
        are stored as float milliseconds (``stamp_step * dt``). If ``True``,
        timestamps are stored as integer-valued step numbers cast to
        ``float64``, and an additional ``events['offsets']`` key is emitted
        as a zero-filled array of matching shape.  Default is ``False``.
    frozen : bool, optional
        NEST-compatibility flag.  ``True`` is unconditionally rejected because
        multimeters cannot be frozen. Default is ``False``.
    name : str or None, optional
        Optional node name forwarded to :class:`brainstate.nn.Dynamics`.
        Default is ``None``.

    Parameter Mapping
    -----------------
    .. list-table:: Mapping of constructor parameters to model symbols
       :header-rows: 1
       :widths: 22 18 22 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``interval``
         - ``1.0 * u.ms``
         - :math:`m \cdot dt`
         - Sampling period on the simulation step grid.
       * - ``offset``
         - ``0.0 * u.ms``
         - :math:`o \cdot dt`
         - Phase shift of the sampling lattice.
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower bound of the activity window.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound of the activity window.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global time-origin shift for the recording window.
       * - ``record_from``
         - ``()``
         - :math:`\{x_r\}_{r=1}^{R}`
         - Ordered recordable channels expected in each payload.

    Returns
    -------
    events : dict[str, np.ndarray]
        Returned by :meth:`update` and :meth:`flush`.  All arrays are
        one-dimensional with length ``E`` equal to the total number of
        emitted samples accumulated so far:

        - ``'times'`` — ``float64``, shape ``(E,)``: timestamp of each
          sample in ms (or integer step index when
          ``time_in_steps=True``).
        - ``'senders'`` — ``int64``, shape ``(E,)``: sender node ID for
          each sample, defaulting to ``1`` when not supplied.
        - ``'offsets'`` — ``float64``, shape ``(E,)`` (only present when
          ``time_in_steps=True``): zero-filled offset array matching NEST
          semantics.
        - ``<key>`` — ``float64``, shape ``(E,)`` for each name in
          ``record_from``: recorded values for that channel.

    Raises
    ------
    ValueError
        If ``frozen=True``; if any timing parameter (``interval``,
        ``offset``, ``start``, ``stop``, ``origin``, ``dt``) is not
        scalar-convertible, not finite when required, not aligned to
        ``dt``, or violates ordering constraints (e.g. ``interval < dt``
        or ``stop < start``); if ``data`` passed to :meth:`update` is not
        a mapping; if a required recordable key is absent from ``data``;
        if a recordable payload is empty after conversion; or if
        recordable/sender lengths are inconsistent after
        flattening/broadcasting.
    TypeError
        If unit conversion or array casting of any time parameter or
        payload value to a numeric type fails.
    KeyError
        If :meth:`get` is called with a key other than ``'events'`` or
        ``'n_events'``.

    Notes
    -----
    - After the first :meth:`connect` call or the first data-carrying
      :meth:`update`, properties ``interval``, ``offset``, and
      ``record_from`` become immutable and further assignments raise
      ``ValueError``.
    - This recorder does not read neuron states autonomously; the caller
      is responsible for extracting state values and passing them via
      ``data`` in each :meth:`update` call after state integration.
    - :meth:`init_state` clears all accumulated events and the pending
      buffer; it can be used to reset the recorder between simulation
      segments without reconstructing the object.

    References
    ----------
    .. [1] NEST Simulator, ``multimeter`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/multimeter.html

    Examples
    --------
    Record membrane potential from a single ``iaf_psc_delta`` neuron for
    50 steps at 0.1 ms resolution, with the recording window clipped to
    the first 5 ms:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neuron = brainpy.state.iaf_psc_delta(1, I_e=500.0 * u.pA)
       ...     neuron.init_state()
       ...     mm = brainpy.state.multimeter(
       ...         record_from=['V_m'],
       ...         interval=0.1 * u.ms,
       ...         start=0.0 * u.ms,
       ...         stop=5.0 * u.ms,
       ...     )
       ...     for k in range(50):
       ...         with brainstate.environ.context(t=k * 0.1 * u.ms):
       ...             neuron.update()
       ...             vm = float(neuron.V.value[0] / u.mV)
       ...             _ = mm.update(
       ...                 {'V_m': np.array([vm], dtype=np.float64)},
       ...                 senders=np.array([1]),
       ...             )
       ...     events = mm.flush()
       ...     _ = events['V_m'].shape
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        record_from: Sequence[str] = (),
        interval: ArrayLike = 1.0 * u.ms,
        offset: ArrayLike = 0.0 * u.ms,
        start: ArrayLike = 0.0 * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0.0 * u.ms,
        time_in_steps: bool = False,
        frozen: bool = False,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        if frozen:
            raise ValueError('multimeter cannot be frozen.')

        self._has_targets = False
        self._interval = interval
        self._offset = offset
        self._record_from = ()

        self.start = start
        self.stop = stop
        self.origin = origin
        self.time_in_steps = bool(time_in_steps)

        self._pending: list[_PendingSample] = []
        self.record_from = tuple(record_from)
        self.clear_events()

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        if self._has_targets:
            raise ValueError(
                'The recording interval, the interval offset and the list of '
                'properties to record cannot be changed after the multimeter '
                'has been connected to nodes.'
            )
        self._interval = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        if self._has_targets:
            raise ValueError(
                'The recording interval, the interval offset and the list of '
                'properties to record cannot be changed after the multimeter '
                'has been connected to nodes.'
            )
        self._offset = value

    @property
    def record_from(self):
        return self._record_from

    @record_from.setter
    def record_from(self, value):
        if self._has_targets:
            raise ValueError(
                'The recording interval, the interval offset and the list of '
                'properties to record cannot be changed after the multimeter '
                'has been connected to nodes.'
            )
        self._record_from = tuple(str(v) for v in value)
        self._events_values = {name: [] for name in self._record_from}
        self._pending.clear()

    @property
    def n_events(self) -> int:
        return len(self._events_times)

    @property
    def events(self) -> dict[str, np.ndarray]:
        out = {
            'times': np.asarray(self._events_times, dtype=np.float64),
            'senders': np.asarray(self._events_senders, dtype=np.int64),
        }
        if self.time_in_steps:
            out['offsets'] = np.zeros(out['times'].shape, dtype=np.float64)
        for key in self._record_from:
            out[key] = np.asarray(self._events_values[key], dtype=np.float64)
        return out

    def get(self, key: str = 'events'):
        if key == 'events':
            return self.events
        if key == 'n_events':
            return self.n_events
        raise KeyError(f'Unsupported key "{key}" for multimeter.get().')

    def clear_events(self):
        self._events_times: list[float] = []
        self._events_senders: list[int] = []
        self._events_values = {name: [] for name in self._record_from}

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self.clear_events()
        self._pending.clear()

    def connect(self):
        self._has_targets = True

    def flush(self):
        r"""Emit all buffered pending samples and return the current event store.

        Reads ``dt`` from :func:`brainstate.environ.get_dt`, validates timing
        calibration, converts pending step stamps to output times, and appends
        all active samples to the internal event arrays.  After the call the
        pending buffer is empty.

        Returns
        -------
        events : dict[str, np.ndarray]
            Event dictionary identical to :attr:`events`, reflecting all
            samples emitted up to and including this call.  See the class-level
            ``Returns`` section for the full description of keys and dtypes.

        Raises
        ------
        ValueError
            If ``dt`` obtained from the simulation environment is non-positive,
            not scalar-convertible, or incompatible with the configured timing
            parameters (``interval``, ``offset``, ``start``, ``stop``,
            ``origin``).
        TypeError
            If ``dt`` cannot be converted to a scalar ``float`` ms value.
        """
        dt = brainstate.environ.get_dt()
        calib = self._get_step_calibration(dt)
        self._emit_pending(calib)
        return self.events

    def update(
        self,
        data: Mapping[str, ArrayLike] = None,
        senders: ArrayLike = None,
    ):
        r"""Process one simulation step and optionally enqueue a new sample.

        Parameters
        ----------
        data : Mapping[str, ArrayLike] or None, optional
            Mapping from each name in ``record_from`` to its current analog
            value payload. Each payload is converted to ``np.float64`` and
            flattened to shape ``(N,)``. Scalars (size 1) are broadcast to
            ``(N,)`` when another recordable defines ``N > 1``. If ``None``,
            no new sample is enqueued and only pending samples are emitted.
            Default is ``None``.
        senders : ArrayLike or None, optional
            Sender IDs associated with the payload. Converted to ``np.int64``
            and flattened to shape ``(N,)`` using the same scalar-broadcast rule
            as recordables. If ``None``, all sender IDs default to ``1``.
            Default is ``None``.

        Returns
        -------
        events : dict[str, np.ndarray]
            Event dictionary identical to :attr:`events` after emitting all
            pending samples and optionally enqueuing the new payload.  See the
            class-level ``Returns`` section for the full description of keys
            and dtypes.

        Raises
        ------
        ValueError
            If current simulation time ``t`` is not aligned to the simulation
            grid; if timing parameters are incompatible with ``dt``; if
            ``data`` is not a ``Mapping``; if a required recordable key is
            absent from ``data``; if a recordable payload is empty after
            conversion; or if recordable/sender lengths are inconsistent after
            the scalar-broadcast rule.
        TypeError
            If conversion of ``t``, ``dt``, or any payload value to a numeric
            array fails.
        KeyError
            If ``brainstate.environ`` does not provide the ``'t'`` or ``dt``
            context keys required for step computation.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        calib = self._get_step_calibration(dt)

        self._emit_pending(calib)

        if data is None:
            return self.events

        self._has_targets = True

        if len(self._record_from) == 0:
            return self.events

        step_now = self._time_to_step(t, calib.dt_ms)
        stamp_step = step_now + 1
        if self._should_sample(stamp_step, calib.interval_steps, calib.offset_steps):
            self._pending.append(self._pack_sample(stamp_step, data, senders))

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

        interval_steps = self._to_step_count(self.interval, dt_ms, 'interval')
        if interval_steps < 1:
            raise ValueError('The sampling interval must be at least as long as the simulation resolution.')

        offset_steps = self._to_step_count(self.offset, dt_ms, 'offset')
        if offset_steps != 0 and offset_steps < 1:
            raise ValueError('The offset for the sampling interval must be at least as long as the simulation resolution.')

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
            interval_steps=interval_steps,
            offset_steps=offset_steps,
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
    def _should_sample(stamp_step: int, interval_steps: int, offset_steps: int) -> bool:
        if offset_steps == 0:
            return (stamp_step % interval_steps) == 0
        if stamp_step < offset_steps:
            return False
        return ((stamp_step - offset_steps) % interval_steps) == 0

    @staticmethod
    def _is_active(stamp_step: int, t_min_steps: int, t_max_steps: float) -> bool:
        if stamp_step <= t_min_steps:
            return False
        if math.isinf(t_max_steps):
            return True
        return stamp_step <= t_max_steps

    @staticmethod
    def _to_float_array(x, name: str) -> np.ndarray:
        if isinstance(x, u.Quantity):
            x = u.get_mantissa(x)
        arr = np.asarray(u.math.asarray(x), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f'Recordable "{name}" must contain at least one value.')
        return arr

    def _pack_sample(
        self,
        stamp_step: int,
        data: Mapping[str, ArrayLike],
        senders: ArrayLike = None,
    ) -> _PendingSample:
        if not isinstance(data, Mapping):
            raise ValueError('data must be a mapping from recordable names to values.')

        values: dict[str, np.ndarray] = {}
        n_items = None
        for key in self._record_from:
            if key not in data:
                raise ValueError(f'Missing recordable "{key}" in data.')
            arr = self._to_float_array(data[key], key)
            if n_items is None:
                n_items = arr.size
            elif arr.size == 1 and n_items > 1:
                arr = np.full((n_items,), arr[0], dtype=np.float64)
            elif arr.size != n_items:
                raise ValueError(f'All recordables must have the same size, got "{key}" with size {arr.size}.')
            values[key] = arr

        if n_items is None:
            n_items = 0

        if senders is None:
            sender_arr = np.ones((n_items,), dtype=np.int64)
        else:
            sender_arr = np.asarray(u.math.asarray(senders), dtype=np.int64).reshape(-1)
            if sender_arr.size == 1 and n_items > 1:
                sender_arr = np.full((n_items,), sender_arr[0], dtype=np.int64)
            elif sender_arr.size != n_items:
                raise ValueError(
                    f'senders size ({sender_arr.size}) does not match recordable size ({n_items}).'
                )

        return _PendingSample(
            stamp_step=stamp_step,
            senders=sender_arr,
            values=values,
        )

    def _emit_pending(self, calib: _StepCalibration):
        if len(self._pending) == 0:
            return

        for sample in self._pending:
            if not self._is_active(sample.stamp_step, calib.t_min_steps, calib.t_max_steps):
                continue

            if self.time_in_steps:
                timestamp = float(sample.stamp_step)
            else:
                timestamp = sample.stamp_step * calib.dt_ms

            n_items = sample.senders.size
            self._events_times.extend([timestamp] * n_items)
            self._events_senders.extend(sample.senders.tolist())
            for key in self._record_from:
                self._events_values[key].extend(sample.values[key].tolist())

        self._pending.clear()
