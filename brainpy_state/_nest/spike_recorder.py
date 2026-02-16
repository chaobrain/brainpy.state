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
    'spike_recorder',
]


@dataclass
class _StepCalibration:
    dt_ms: float
    t_min_steps: int
    t_max_steps: float


class spike_recorder(brainstate.nn.Dynamics):
    r"""NEST-compatible spike recording device.

    Short Description
    -----------------
    ``spike_recorder`` accumulates spike events into an in-memory ``events``
    dictionary, with timestamping and activity-window semantics matching NEST
    ``spike_recorder``.

    Description
    -----------
    This implementation follows the NEST recording-device timing rules while
    exposing a Python batch API.

    **1. Step-stamp and physical-time model**

    Let :math:`dt > 0` be simulation resolution (ms), and let
    :math:`n = \mathrm{round}(t/dt)` be the current step index when
    :meth:`update` is called at simulation time :math:`t`. Incoming events are
    stamped at

    .. math::

       s = n + 1,

    i.e., spikes are interpreted as generated during :math:`(t, t + dt]`. If
    per-event offsets :math:`\delta_j` (ms) are provided, the stored physical
    event time for item :math:`j` is

    .. math::

       t_j = s \cdot dt - \delta_j.

    With ``time_in_steps=True``, storage is split into integer stamps
    ``events['times']`` (step index :math:`s`) and continuous offsets
    ``events['offsets']`` (:math:`\delta_j`, ms), preserving sub-step timing.

    **2. Activity-window gate on the step lattice**

    Define step bounds

    .. math::

       s_{\min} = \frac{\mathrm{origin} + \mathrm{start}}{dt}, \qquad
       s_{\max} = \frac{\mathrm{origin} + \mathrm{stop}}{dt}
       \ \ (\text{or } +\infty \text{ if stop is None}).

    The recorder is active for stamp step :math:`s` iff

    .. math::

       s > s_{\min} \ \land\ s \le s_{\max}.

    Therefore, ``start`` is exclusive and ``stop`` is inclusive, exactly as in
    NEST recording devices.

    **3. Multiplicity inference and payload normalization**

    Incoming arrays are flattened to one-dimensional vectors of length
    :math:`N`. Scalars are broadcast to :math:`(N,)` for ``senders`` and
    ``offsets``. Let :math:`x_j` be ``spikes[j]``:

    - If ``multiplicities is None`` and all ``spikes`` are integer-like
      (within ``1e-12`` tolerance), event counts are
      :math:`c_j = \max(\mathrm{round}(x_j), 0)`.
    - If ``multiplicities is None`` and ``spikes`` contains non-integer values,
      :math:`c_j = \mathbf{1}[x_j > 0]`.
    - If ``multiplicities`` is provided, with non-negative integers
      :math:`m_j`, then :math:`c_j = m_j \mathbf{1}[x_j > 0]`.

    Each item contributes exactly :math:`c_j` stored events by repetition.

    **4. Constraints and computational implications**

    ``start``, ``stop`` (when not ``None``), ``origin``, current ``t``, and
    ``dt`` must be scalar-convertible and aligned to the simulation grid.
    Alignment is enforced by round-trip integer checks with ``1e-12``
    tolerance. Per :meth:`update` call, normalization is :math:`O(N)` and event
    expansion is :math:`O(E_{\mathrm{new}})` where
    :math:`E_{\mathrm{new}} = \sum_j c_j`. Persistent memory usage is linear in
    total stored events.

    Parameters
    ----------
    in_size : Size, optional
        Shape/size argument consumed by :class:`brainstate.nn.Dynamics`. The
        recorder returns dictionaries rather than dense tensors; ``in_size`` is
        retained for API compatibility. Default is ``1``.
    start : ArrayLike, optional
        Scalar relative lower bound (typically ms) of the recording window.
        Must be finite and an integer multiple of ``dt``. Effective gate is
        ``stamp_step > (origin + start) / dt``. Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative upper bound (typically ms) of the recording window.
        Must be ``None`` or finite and aligned to ``dt``. Effective gate is
        ``stamp_step <= (origin + stop) / dt``. ``None`` means no upper bound.
        Default is ``None``.
    origin : ArrayLike, optional
        Scalar origin shift (typically ms) added to ``start`` and ``stop``.
        Must be finite and aligned to ``dt``. Default is ``0.0 * u.ms``.
    time_in_steps : bool, optional
        Output time representation. If ``False``, ``events['times']`` stores
        ``float64`` milliseconds. If ``True``, ``events['times']`` stores
        integer step stamps (``int64``) and ``events['offsets']`` stores
        ``float64`` milliseconds. Default is ``False``.
    frozen : bool, optional
        NEST compatibility argument. ``True`` is rejected because this recorder
        is not freezable. Default is ``False``.
    name : str or None, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 22 18 22 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower bound of the active window.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound of the active window.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global origin shift applied before window gating.
       * - ``time_in_steps``
         - ``False``
         - :math:`\mathrm{repr}_t`
         - Time storage mode: physical ms or ``(step, offset)``.

    Returns
    -------
    out : Any
        Dynamics node. :meth:`update`, :meth:`flush`, and ``get('events')``
        return an ``events`` dictionary containing one-dimensional arrays:
        ``senders`` (``int64``, shape ``(E,)``), ``times`` (shape ``(E,)``,
        ``float64`` in ms or ``int64`` in steps), and optional ``offsets``
        (``float64``, shape ``(E,)``) when ``time_in_steps=True``.

    Raises
    ------
    ValueError
        If ``frozen=True``; if time parameters are non-scalar, non-finite, not
        aligned to ``dt``, or violate ``stop >= start``; if ``time_in_steps``
        is modified after simulation begins; if ``n_events`` is set to a value
        other than ``0``; if array sizes mismatch; or if explicit
        ``multiplicities`` contain negative entries.
    TypeError
        If unit conversion or numeric casting of inputs fails.
    KeyError
        If :meth:`get` is called with an unsupported key, or if simulation
        context keys required by :meth:`update` are unavailable.

    Notes
    -----
    - Event writes are immediate (no one-step delivery lag), unlike
      request/reply devices such as ``multimeter``.
    - ``time_in_steps`` becomes immutable after the first :meth:`update` call
      that accesses simulation context, matching NEST backend constraints.
    - ``spikes=None`` is treated as a no-op update that returns current
      ``events``.

    References
    ----------
    .. [1] NEST Simulator, ``spike_recorder`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/spike_recorder.html

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     sr = brainpy.state.spike_recorder(start=0.0 * u.ms, stop=1.0 * u.ms)
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = sr.update(
       ...             spikes=np.array([1.0, 0.0, 2.0], dtype=np.float64),
       ...             senders=np.array([3, 4, 5], dtype=np.int64),
       ...         )
       ...     ev = sr.flush()
       ...     _ = ev['times'].shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     sr = brainpy.state.spike_recorder(time_in_steps=True)
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = sr.update(
       ...             spikes=np.array([1.0], dtype=np.float64),
       ...             senders=np.array([9], dtype=np.int64),
       ...             offsets=np.array([0.03], dtype=np.float64) * u.ms,
       ...         )
       ...     ev = sr.events
       ...     _ = (ev['times'][0], ev['offsets'][0])
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
            raise ValueError('spike_recorder cannot be frozen.')

        self.start = start
        self.stop = stop
        self.origin = origin

        self._time_in_steps = bool(time_in_steps)
        self._has_been_simulated = False

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
        raise KeyError(f'Unsupported key "{key}" for spike_recorder.get().')

    def clear_events(self):
        self._events_senders: list[int] = []
        self._events_times_ms: list[float] = []
        self._events_times_steps: list[int] = []
        self._events_offsets: list[float] = []

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        self.clear_events()

    def connect(self):
        # Kept for API symmetry with multimeter.
        return None

    def flush(self):
        return self.events

    def update(
        self,
        spikes: ArrayLike = None,
        senders: ArrayLike = None,
        offsets: ArrayLike = None,
        multiplicities: ArrayLike = None,
    ):
        r"""Record spike events for the current simulation step.

        Parameters
        ----------
        spikes : ArrayLike or None, optional
            Input spike payload flattened to shape ``(N,)``. Accepted dtypes
            include boolean, integer, and floating values.

            - If ``None``, no new events are written and current ``events`` are
              returned.
            - If integer-like (all values within ``1e-12`` of integers) and
              ``multiplicities is None``, each element contributes
              ``max(round(value), 0)`` events.
            - Otherwise (floating path, ``multiplicities is None``), each
              element contributes one event iff ``value > 0``.
        senders : ArrayLike or None, optional
            Sender IDs as integers, shape ``(N,)`` or scalar broadcastable to
            ``(N,)``. Default is ``1`` for all entries.
        offsets : ArrayLike or None, optional
            Event offsets in milliseconds, shape ``(N,)`` or scalar
            broadcastable to ``(N,)``. Values may be unitful and convertible to
            ``u.ms``. Default is ``0.0 * u.ms``.
        multiplicities : ArrayLike or None, optional
            Explicit non-negative integer multiplicities, shape ``(N,)`` or
            scalar broadcastable to ``(N,)``. When provided, count rule is
            ``multiplicities * (spikes > 0)`` and integer-like inference from
            ``spikes`` is disabled. Default is ``None``.

        Returns
        -------
        out : Any
            Current ``events`` dictionary after processing, with arrays:
            ``senders`` (``int64``, shape ``(E,)``) and ``times`` (shape
            ``(E,)``; ``float64`` in ms when ``time_in_steps=False``, or
            ``int64`` step stamps when ``time_in_steps=True``). If
            ``time_in_steps=True``, ``offsets`` (``float64``, shape ``(E,)``)
            is also present.

        Raises
        ------
        ValueError
            If ``t`` is not grid-aligned; if ``start``/``stop``/``origin`` are
            invalid with respect to ``dt``; if ``dt <= 0``; if provided array
            sizes are incompatible with ``spikes`` length; if offsets contain
            non-finite values; or if explicit ``multiplicities`` contain
            negative values.
        TypeError
            If numeric or unit conversion of any payload fails.
        KeyError
            If required simulation context entries (``'t'`` or ``dt``) are not
            available via ``brainstate.environ``.

        Notes
        -----
        Events are written at stamp step ``round(t / dt) + 1`` and then gated
        by the active window ``(origin + start, origin + stop]`` in step space.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        calib = self._get_step_calibration(dt)

        step_now = self._time_to_step(t, calib.dt_ms)
        stamp_step = step_now + 1

        self._has_been_simulated = True

        if spikes is None:
            return self.events

        spike_arr = self._to_float_array(spikes, name='spikes')
        if spike_arr.size == 0:
            return self.events

        n_items = spike_arr.size

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

        if not self._is_active(stamp_step, calib.t_min_steps, calib.t_max_steps):
            return self.events

        active = counts > 0
        if not np.any(active):
            return self.events

        out_senders = np.repeat(sender_arr[active], counts[active])
        out_offsets = np.repeat(offset_arr[active], counts[active])

        self._events_senders.extend(out_senders.tolist())
        if self.time_in_steps:
            out_steps = np.full(out_senders.shape, stamp_step, dtype=np.int64)
            self._events_times_steps.extend(out_steps.tolist())
            self._events_offsets.extend(out_offsets.tolist())
        else:
            out_times_ms = stamp_step * calib.dt_ms - out_offsets
            self._events_times_ms.extend(out_times_ms.tolist())

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
