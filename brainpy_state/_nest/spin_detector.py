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
import saiunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTDevice

__all__ = [
    'spin_detector',
]


@dataclass
class _StepCalibration:
    dt_ms: float
    t_min_steps: int
    t_max_steps: float


class spin_detector(NESTDevice):
    r"""NEST-compatible detector for binary state decoding from spikes.

    ``spin_detector`` decodes binary activity (``state`` :math:`\in \{0, 1\}`)
    from spike-event multiplicities and stores a chronological event log
    containing ``senders``, ``times``, and decoded ``state`` for every
    emitted event.  The decode logic mirrors NEST
    ``models/spin_detector.{h,cpp}`` with explicit per-event buffering:
    a single provisional event is held in a one-slot buffer and revised
    from state ``0`` to ``1`` before being written whenever a same-sender,
    same-stamp event with multiplicity ``1`` arrives, while multiplicity
    ``2`` events bypass the buffer and are written immediately as state ``1``.

    **1. Event Decoding on a Sender-Time Lattice**

    Let incoming normalized events be
    :math:`e_j=(i_j, s_j, \delta_j, m_j)` with sender :math:`i_j \in \mathbb{N}`,
    step stamp :math:`s_j \in \mathbb{Z}`, offset :math:`\delta_j` (ms), and
    multiplicity :math:`m_j \ge 0`. The detector maintains one buffered tuple
    :math:`b=(i_b, s_b, \delta_b, x_b)` where :math:`x_b \in \{0,1\}` is the
    provisional decoded state.

    For each accepted event in order:

    - If :math:`m_j = 1` and :math:`(i_j, s_j) = (i_b, s_b)`, revise
      :math:`x_b \leftarrow 1` before writing.
    - If a buffer exists, write :math:`b` to output.
    - If :math:`m_j = 2`, write current event immediately with state ``1``
      and clear the buffer.
    - Otherwise, set buffer to current event with provisional state ``0``
      when the buffer is empty; if the buffer is not empty, clear it instead.

    This ordering ensures that a possible ``0 -> 1`` revision is applied
    before the buffered-write emission, exactly as in the NEST C++ reference.

    **2. Time Model and Activity Window**

    With simulation resolution :math:`dt > 0` (ms), current simulation time
    :math:`t`, and step index :math:`n = \mathrm{round}(t/dt)`, the default
    event stamp for events received at step :math:`n` is

    .. math::

       s = n + 1.

    The physical event time in milliseconds is reconstructed as

    .. math::

       t_{\mathrm{event}} = s \cdot dt - \delta.

    Recording is gated on stamps by the half-open interval
    :math:`(s_{\min},\, s_{\max}]` where

    .. math::

       s_{\min} = \frac{\mathrm{origin} + \mathrm{start}}{dt}, \qquad
       s_{\max} = \frac{\mathrm{origin} + \mathrm{stop}}{dt}
       \quad (\text{or } +\infty \text{ when stop is None}),

    so an event is accepted iff :math:`s > s_{\min} \land s \le s_{\max}`.
    The ``start`` bound is exclusive and ``stop`` is inclusive.

    **3. Input Normalization and Multiplicity Inference**

    Runtime ``update`` arrays are flattened to one-dimensional vectors of
    length :math:`N`. Scalars for ``senders``, ``offsets``, and
    ``stamp_steps`` are broadcast to :math:`(N,)`.

    Let :math:`a_j = \mathrm{spikes}[j]`. Per-item event multiplicity
    :math:`c_j` is determined as follows:

    - If ``multiplicities is None`` and all :math:`a_j` are integer-like
      (within ``1e-12`` tolerance):
      :math:`c_j = \max(\mathrm{round}(a_j),\, 0)`.
    - If ``multiplicities is None`` and any :math:`a_j` is non-integer:
      :math:`c_j = \mathbf{1}[a_j > 0]` (binary threshold).
    - If ``multiplicities`` is provided with non-negative integers
      :math:`m_j`: :math:`c_j = m_j \,\mathbf{1}[a_j > 0]`.

    Each event item contributes **at most one decode step** because
    :math:`c_j` is passed as the multiplicity to :meth:`_handle_event`
    rather than used for repeated writes.

    **4. Assumptions, Constraints, and Computational Implications**

    ``dt``, ``t``, ``start``, ``stop`` (when finite), and ``origin`` must be
    scalar-convertible and aligned to the simulation lattice. Alignment is
    verified by round-trip integer checks with ``1e-12`` tolerance.  Per
    :meth:`update` call, normalization is :math:`O(N)` and decoding is
    :math:`O(N)`, with persistent storage cost linear in the total number of
    emitted events :math:`E`.

    Parameters
    ----------
    in_size : Size, optional
        Shape/size metadata consumed by :class:`brainstate.nn.Dynamics`.
        The detector is event-driven and does not return dense tensors, so
        ``in_size`` is retained for API compatibility only. Default is ``1``.
    start : saiunit.Quantity or float, optional
        Scalar relative exclusive lower bound of the recording window,
        convertible to ms. Must be finite and an integer multiple of ``dt``.
        The effective gate is ``stamp_step > (origin + start) / dt``.
        Default is ``0.0 * u.ms``.
    stop : saiunit.Quantity, float, or None, optional
        Scalar relative inclusive upper bound of the recording window,
        convertible to ms. Must be ``None`` or finite and aligned to ``dt``.
        Must satisfy ``stop >= start`` when not ``None``. The effective gate
        is ``stamp_step <= (origin + stop) / dt``. ``None`` means no upper
        bound (:math:`s_{\max} = +\infty`). Default is ``None``.
    origin : saiunit.Quantity or float, optional
        Scalar global time-origin shift added to both ``start`` and ``stop``
        when constructing the active window, convertible to ms. Shifting the
        origin displaces the entire recording window without changing its
        duration. Must be finite and aligned to ``dt``. Default is
        ``0.0 * u.ms``.
    time_in_steps : bool, optional
        Controls the time representation in ``events``. If ``False``,
        ``events['times']`` stores ``float64`` milliseconds computed as
        :math:`s \cdot dt - \delta_j`. If ``True``, ``events['times']``
        stores integer step stamps (``int64``) and ``events['offsets']``
        stores the corresponding ``float64`` offsets in ms. Becomes immutable
        after the first :meth:`update` call. Default is ``False``.
    frozen : bool, optional
        NEST-compatibility flag. ``True`` is unconditionally rejected because
        this device cannot be frozen. Default is ``False``.
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
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower bound of the recording window.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound of the recording window.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global shift applied to both window boundaries.
       * - ``time_in_steps``
         - ``False``
         - :math:`\mathrm{repr}_t`
         - Output-time representation: ms float or integer ``(step, offset)`` pair.

    Raises
    ------
    ValueError
        If ``frozen=True``; if ``dt`` is non-positive; if time parameters are
        non-scalar, non-finite where finite values are required, misaligned to
        the simulation step, or violate ``stop >= start``; if ``t`` is not on
        the simulation grid; if ``time_in_steps`` is modified after simulation
        begins; if ``n_events`` is set to any value other than ``0``; if
        provided arrays have inconsistent sizes; if ``spikes``/``offsets``
        contain non-finite values; or if explicit ``multiplicities`` contain
        negative entries.
    TypeError
        If unit conversion or numeric coercion of scalar/array inputs fails.
    KeyError
        If :meth:`get` is called with an unsupported key, or if required
        simulation context values (``'t'`` or ``dt``) are unavailable via
        ``brainstate.environ``.

    Notes
    -----
    - Input events are processed strictly in the order supplied, and one
      buffered event is finalized at the end of every :meth:`update` call.
    - Connection weight and delay do not participate in decode logic.
    - ``time_in_steps`` becomes immutable after the first :meth:`update`
      call that accesses simulation context, matching NEST backend constraints.
    - NEST semantics are defined for multiplicities ``1`` and ``2``. This
      implementation also accepts other non-negative values, which follow the
      ``m != 2`` branch in :meth:`_handle_event`.
    - ``spikes=None`` is a no-op that flushes the buffer and returns the
      current ``events`` without writing any new events.
    - :meth:`init_state` clears all accumulated events and the one-slot
      buffer; it can be used to reset the detector between simulation
      segments without reconstructing the object.

    References
    ----------
    .. [1] NEST Simulator, ``spin_detector`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/spin_detector.html

    Examples
    --------
    Detect binary state for two same-sender, same-stamp events — the second
    event (multiplicity 1, matching sender and stamp) upgrades the state to
    ``1`` before the buffered event is written:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     det = brainpy.state.spin_detector(start=0.0 * u.ms, stop=1.0 * u.ms)
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       dftype = brainstate.environ.dftype()
       ditype = brainstate.environ.ditype()
       ...         _ = det.update(
       ...             spikes=np.array([1.0, 1.0], dtype=dftype),
       ...             senders=np.array([7, 7], dtype=ditype),
       ...             stamp_steps=np.array([1, 1], dtype=ditype),
       ...         )
       ...     ev = det.flush()
       ...     _ = (ev['senders'][0], ev['state'][0])

    Record a multiplicity-2 event with a sub-step offset using
    ``time_in_steps=True``, which splits the timestamp into an integer step
    index and a float offset — multiplicity ``2`` events are written
    immediately with state ``1``:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     det = brainpy.state.spin_detector(time_in_steps=True)
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = det.update(
       ...             spikes=np.array([2.0], dtype=dftype),
       ...             senders=np.array([3], dtype=ditype),
       ...             offsets=np.array([0.02], dtype=dftype) * u.ms,
       ...         )
       ...     ev = det.events
       ...     _ = (ev['times'][0], ev['offsets'][0], ev['state'][0])
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
        ditype = brainstate.environ.ditype()
        out = {
            'senders': np.asarray(self._events_senders, dtype=ditype),
            'state': np.asarray(self._events_state, dtype=ditype),
        }
        if self.time_in_steps:
            out['times'] = np.asarray(self._events_times_steps, dtype=ditype)
            dftype = brainstate.environ.dftype()
            out['offsets'] = np.asarray(self._events_offsets, dtype=dftype)
        else:
            out['times'] = np.asarray(self._events_times_ms, dtype=dftype)
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
        r"""Decode binary states from spike events for the current simulation step.

        Reads the current simulation time ``t`` and resolution ``dt`` from
        ``brainstate.environ``, derives the default stamp step
        :math:`s = \mathrm{round}(t/dt) + 1`, normalizes the input arrays,
        applies the activity-window gate, and passes each accepted event through
        the one-slot decode buffer via :meth:`_handle_event`. After all items
        are processed, :meth:`_flush_last_event` finalizes any remaining
        buffered event.

        Parameters
        ----------
        spikes : ArrayLike or None, optional
            Input spike payload, flattened to shape ``(N,)``. Accepted dtypes
            include boolean, integer, and floating-point values.

            - ``None``: no new events are processed; the buffer is flushed
              and the current ``events`` dict is returned immediately.
            - Integer-like values (all within ``1e-12`` of an integer) with
              ``multiplicities is None``: each element :math:`j` contributes
              multiplicity :math:`c_j = \max(\mathrm{round}(a_j),\, 0)`.
            - Non-integer floating values with ``multiplicities is None``:
              each element contributes :math:`c_j = \mathbf{1}[a_j > 0]`
              (binary threshold).
        senders : ArrayLike or None, optional
            Sender node IDs cast to ``int64``, shape ``(N,)`` or scalar
            broadcastable to ``(N,)``. Default sender ID is ``1`` for all
            entries when ``None``.
        offsets : ArrayLike or None, optional
            Per-event sub-step timing offsets :math:`\delta_j` in ms, shape
            ``(N,)`` or scalar broadcastable to ``(N,)``. Values may carry a
            ``saiunit`` time unit and are converted to ms. Must contain only
            finite values. Default is ``0.0 * u.ms`` for all entries.
        multiplicities : ArrayLike or None, optional
            Explicit non-negative integer event multiplicities cast to
            ``int64``, shape ``(N,)`` or scalar broadcastable to ``(N,)``.
            When provided, the integer-like inference path from ``spikes`` is
            disabled; the count rule becomes
            :math:`c_j = m_j \,\mathbf{1}[a_j > 0]`. Negative values raise
            ``ValueError``. Default is ``None``.
        stamp_steps : ArrayLike or None, optional
            Explicit integer step stamps :math:`s_j` for each event, cast to
            ``int64``, shape ``(N,)`` or scalar broadcastable to ``(N,)``.
            When ``None``, all events are stamped at :math:`s = n + 1` where
            :math:`n = \mathrm{round}(t/dt)`. Providing custom stamps allows
            events generated at different simulation steps to be injected in
            a single call. Default is ``None``.

        Returns
        -------
        events : dict[str, np.ndarray]
            Current accumulated events dictionary after processing this step.
            All arrays are one-dimensional with length :math:`E` equal to
            the total number of stored events:

            - ``'senders'`` — ``int64``, shape ``(E,)``.
            - ``'state'`` — ``int64``, shape ``(E,)``: decoded binary state
              (:math:`0` or :math:`1`).
            - ``'times'`` — ``float64`` ms when ``time_in_steps=False``;
              ``int64`` step stamps when ``time_in_steps=True``.
            - ``'offsets'`` — ``float64`` ms, shape ``(E,)`` (only present
              when ``time_in_steps=True``).

        Raises
        ------
        ValueError
            If ``t`` is not grid-aligned to ``dt``; if ``start``, ``stop``,
            or ``origin`` are invalid with respect to ``dt``; if ``dt <= 0``;
            if provided payload array sizes are incompatible with the ``N``
            inferred from ``spikes``; if ``offsets`` contain non-finite
            values; or if explicit ``multiplicities`` contain negative entries.
        TypeError
            If numeric or unit conversion of any payload or time parameter
            fails.
        KeyError
            If required simulation context entries (``'t'`` or ``dt``) are
            not available via ``brainstate.environ``.

        Notes
        -----
        Events are stamped at :math:`s = \mathrm{round}(t/dt) + 1` by
        default and then gated by the active window
        :math:`(s_{\min},\, s_{\max}]` in step space. Events outside the
        window are discarded before reaching :meth:`_handle_event`. The
        one-slot buffer is always flushed at the end of each call regardless
        of how many new events were processed.
        """
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
                    ditype = brainstate.environ.ditype()
                    stamp_arr = np.full((n_items,), step_now + 1, dtype=ditype)
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
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
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
            dftype = brainstate.environ.dftype()
            arr = np.asarray([default], dtype=dftype)
        else:
            if unit is not None and isinstance(x, u.Quantity):
                x = x / unit
            elif isinstance(x, u.Quantity):
                x = u.get_mantissa(x)
            arr = np.asarray(u.math.asarray(x), dtype=dftype).reshape(-1)

        if arr.size == 0 and size is not None:
            return np.zeros((0,), dtype=dftype)

        if not np.all(np.isfinite(arr)):
            raise ValueError(f'{name} must contain finite values.')

        if size is None:
            return arr

        if arr.size == 1 and size > 1:
            return np.full((size,), arr[0], dtype=dftype)
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
            ditype = brainstate.environ.ditype()
            arr = np.asarray([default], dtype=ditype)
        else:
            arr = np.asarray(u.math.asarray(x), dtype=ditype).reshape(-1)

        if size is None:
            return arr.astype(np.int64, copy=False)

        if arr.size == 1 and size > 1:
            return np.full((size,), int(arr[0]), dtype=ditype)
        if arr.size != size:
            raise ValueError(f'{name} size ({arr.size}) does not match spikes size ({size}).')
        return arr.astype(np.int64, copy=False)
