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
    ``weight_recorder`` accumulates transmitted synaptic events in memory,
    storing weight, sender/target IDs, receptor/port metadata, and event time
    with NEST-compatible activity-window and filtering semantics.

    Description
    -----------
    This implementation follows NEST ``weight_recorder`` behavior
    (``models/weight_recorder.{h,cpp}`` and
    ``nestkernel/{recording_device.*,connector_base_impl.h}``) while exposing a
    direct batch API.

    **1. Event payload model on the simulation grid**

    Let :math:`dt > 0` be simulation resolution in ms, and
    :math:`n = \mathrm{round}(t/dt)` the current step when :meth:`update` is
    called at simulation time :math:`t`. For each incoming item
    :math:`j \in \{1,\dots,N\}`, define the payload tuple
    :math:`(w_j, s_j, q_j, r_j, p_j, \delta_j)`:

    - :math:`w_j`: synaptic weight (unitless/implementation-specific value),
    - :math:`s_j`: sender node ID,
    - :math:`q_j`: target node ID,
    - :math:`r_j`: receptor port (rport),
    - :math:`p_j`: connection port metadata,
    - :math:`\delta_j`: sub-step offset in ms.

    If ``stamp_steps`` is omitted, :math:`s^{(\mathrm{step})}_j = n + 1` for all
    items, matching NEST's event stamp convention for events generated during
    :math:`(t, t+dt]`. Otherwise, user-provided integer stamp steps are used.

    With ``time_in_steps=False``, stored physical time for item :math:`j` is

    .. math::

       t_j = s^{(\mathrm{step})}_j \cdot dt - \delta_j.

    With ``time_in_steps=True``, time is represented as
    ``(events['times'], events['offsets']) = (s^{(\mathrm{step})}_j, \delta_j)``.

    **2. Activity-window gate and sender/target filtering**

    Define recorder bounds on the step lattice:

    .. math::

       s_{\min} = \frac{\mathrm{origin} + \mathrm{start}}{dt}, \qquad
       s_{\max} = \frac{\mathrm{origin} + \mathrm{stop}}{dt}
       \ \ (\text{or } +\infty \text{ if stop is None}).

    An event is recordable iff

    .. math::

       s^{(\mathrm{step})}_j > s_{\min} \ \land\
       s^{(\mathrm{step})}_j \le s_{\max}.

    Therefore ``start`` is exclusive and ``stop`` is inclusive. Optional
    filter sets :math:`\mathcal{S}` (``senders``) and :math:`\mathcal{T}`
    (``targets``) further constrain recording:

    .. math::

       s_j \in \mathcal{S} \text{ if } \mathcal{S}\neq\varnothing,\qquad
       q_j \in \mathcal{T} \text{ if } \mathcal{T}\neq\varnothing.

    Events failing any gate/filter condition are discarded.

    **3. Input normalization and broadcast rules**

    All update payload inputs are flattened to one-dimensional arrays.
    ``weights`` defines :math:`N`. Optional payload arrays are interpreted as:

    - ``None`` -> default scalar (broadcast to ``(N,)``),
    - scalar-like -> broadcast to ``(N,)``,
    - length-``N`` vector -> used directly.

    This matches NEST-like behavior where missing sender/target/receptor/port
    fields receive default metadata and each accepted item contributes exactly
    one stored event.

    **4. Assumptions, constraints, and computational implications**

    ``start``, ``stop`` (if finite), ``origin``, current ``t``, and ``dt`` must
    be scalar-convertible and aligned to the simulation grid (checked by
    round-trip integer conversion with ``1e-12`` tolerance). ``stop`` must
    satisfy ``stop >= start`` when finite. Sender/target filters must contain
    strictly positive integer node IDs.

    Per :meth:`update`, normalization and masking are :math:`O(N)`, and appends
    are :math:`O(E_{\mathrm{new}})` where :math:`E_{\mathrm{new}}` is the
    number of accepted events. Persistent storage cost is linear in total
    accumulated events across calls.

    Parameters
    ----------
    in_size : Size, optional
        Shape/size passed to :class:`brainstate.nn.Dynamics`. This recorder
        emits dictionary outputs instead of dense tensors; ``in_size`` is
        retained for API compatibility. Default is ``1``.
    senders : ArrayLike, optional
        Sender filter IDs. Interpreted as a 1-D integer array of shape
        ``(K_s,)`` with strictly positive entries. Empty input disables sender
        filtering. Default is ``()``.
    targets : ArrayLike, optional
        Target filter IDs. Interpreted as a 1-D integer array of shape
        ``(K_t,)`` with strictly positive entries. Empty input disables target
        filtering. Default is ``()``.
    start : ArrayLike, optional
        Scalar relative lower bound of recording window, typically in
        milliseconds (e.g., ``Quantity[ms]``). Effective gate is strict:
        ``stamp_step > (origin + start) / dt``. Must be finite and dt-aligned.
        Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative upper bound of recording window, typically in
        milliseconds. Gate is inclusive:
        ``stamp_step <= (origin + stop) / dt``. ``None`` means no upper bound.
        Finite values must be dt-aligned and satisfy ``stop >= start``.
        Default is ``None``.
    origin : ArrayLike, optional
        Scalar global origin shift (typically milliseconds) added to ``start``
        and ``stop`` before window evaluation. Must be finite and dt-aligned.
        Default is ``0.0 * u.ms``.
    time_in_steps : bool, optional
        Time output representation. If ``False``, ``events['times']`` is
        ``float64`` in ms. If ``True``, ``events['times']`` is ``int64`` step
        stamps and ``events['offsets']`` is ``float64`` in ms. Default is
        ``False``.
    frozen : bool, optional
        NEST-compatibility flag. ``True`` is rejected because this recorder is
        not freezable in this backend. Default is ``False``.
    name : str or None, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 20 16 24 40

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower bound of the active stamp-step interval.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound of the active stamp-step interval.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Origin shift added to both relative bounds before discretization.
       * - ``senders``
         - ``()``
         - :math:`\mathcal{S}`
         - Optional sender-ID whitelist set.
       * - ``targets``
         - ``()``
         - :math:`\mathcal{T}`
         - Optional target-ID whitelist set.
       * - ``time_in_steps``
         - ``False``
         - :math:`\mathrm{repr}_t`
         - Time representation mode (physical ms vs. step+offset).

    Returns
    -------
    out : Any
        Dynamics node. :meth:`update`, :meth:`flush`, and ``get('events')``
        return an ``events`` dictionary of 1-D arrays with common length
        ``(E,)``:

        - ``senders``: ``int64``,
        - ``targets``: ``int64``,
        - ``weights``: ``float64``,
        - ``receptors``: ``int64``,
        - ``ports``: ``int64``,
        - ``times``: ``float64`` ms when ``time_in_steps=False``; otherwise
          ``int64`` step stamps,
        - ``offsets``: ``float64`` ms, present only when
          ``time_in_steps=True``.

    Raises
    ------
    ValueError
        If ``frozen=True``; if filter IDs are non-positive; if timing
        parameters are non-scalar, non-finite (where required), off-grid with
        respect to ``dt``, or violate ``stop >= start``; if ``time_in_steps``
        is modified after simulation has started; if ``n_events`` is set to a
        value other than ``0``; if event payload sizes mismatch; if
        ``weights``/``offsets`` contain non-finite values; or if runtime
        simulation time ``t`` is not grid-aligned.
    TypeError
        If unit conversion or numeric casting of input arrays fails.
    KeyError
        If :meth:`get` is called with an unsupported key, or if simulation
        context values needed by :meth:`update` are unavailable.

    Notes
    -----
    - This recorder only stores events explicitly passed into :meth:`update`;
      it does not introspect synapse objects or connection containers.
    - Filter checks are evaluated before event insertion, following NEST's
      handler ordering for ``weight_recorder``.
    - ``n_events`` is write-restricted to ``0`` to support explicit buffer
      reset while preventing partial truncation.

    References
    ----------
    .. [1] NEST Simulator, ``weight_recorder`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/weight_recorder.html
    .. [2] NEST source files:
           ``models/weight_recorder.h``, ``models/weight_recorder.cpp``,
           ``nestkernel/recording_device.h``, ``nestkernel/connector_base_impl.h``.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     wr = brainpy.state.weight_recorder(
       ...         senders=np.array([10, 11], dtype=np.int64),
       ...         start=0.0 * u.ms,
       ...         stop=1.0 * u.ms,
       ...     )
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = wr.update(
       ...             weights=np.array([0.5, 0.7], dtype=np.float64),
       ...             senders=np.array([10, 12], dtype=np.int64),
       ...             targets=np.array([3, 4], dtype=np.int64),
       ...         )
       ...     ev = wr.flush()
       ...     _ = ev['weights'].shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     wr = brainpy.state.weight_recorder(time_in_steps=True)
       ...     with brainstate.environ.context(t=1.0 * u.ms):
       ...         _ = wr.update(
       ...             weights=np.array([1.2], dtype=np.float64),
       ...             senders=np.array([5], dtype=np.int64),
       ...             targets=np.array([6], dtype=np.int64),
       ...             offsets=np.array([0.03], dtype=np.float64) * u.ms,
       ...             stamp_steps=np.array([12], dtype=np.int64),
       ...         )
       ...     ev = wr.events
       ...     _ = (ev['times'][0], ev['offsets'][0])
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
        """Return a recorder property by key.

        Parameters
        ----------
        key : {'events', 'n_events', 'time_in_steps', 'senders', 'targets'}, optional
            Property selector. Default is ``'events'``.

        Returns
        -------
        out : Any
            Selected value:

            - ``'events'`` -> ``dict[str, np.ndarray]`` with event arrays,
            - ``'n_events'`` -> ``int``,
            - ``'time_in_steps'`` -> ``bool``,
            - ``'senders'`` or ``'targets'`` -> ``np.ndarray`` of ``int64``.

        Raises
        ------
        KeyError
            If ``key`` is unsupported.
        """
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
        """Clear all stored event buffers in-place.

        Returns
        -------
        out : Any
            ``None``. Internal Python lists for all event fields are reset to
            empty lists.
        """
        self._events_senders: list[int] = []
        self._events_targets: list[int] = []
        self._events_weights: list[float] = []
        self._events_receptors: list[int] = []
        self._events_ports: list[int] = []
        self._events_times_ms: list[float] = []
        self._events_times_steps: list[int] = []
        self._events_offsets: list[float] = []

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize dynamic state by clearing recorded events.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused compatibility argument accepted by
            :class:`brainstate.nn.Dynamics`.
        **kwargs
            Unused extra keyword arguments for framework compatibility.

        Returns
        -------
        out : Any
            ``None``.
        """
        del batch_size, kwargs
        self.clear_events()

    def connect(self):
        """Compatibility no-op for device connection phase.

        Returns
        -------
        out : Any
            ``None``.
        """
        return None

    def flush(self):
        """Return a snapshot of all recorded events.

        Returns
        -------
        out : Any
            Same object contract as :attr:`events`: a dictionary of NumPy
            arrays whose fields are synchronized to equal length ``(E,)``.
        """
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
        r"""Record a batch of transmitted synaptic events for the current step.

        Parameters
        ----------
        weights : ArrayLike or None, optional
            Event weights. Flattened to shape ``(N,)`` with dtype ``float64``.
            Unitless/implementation-specific weight values. If ``None``, this
            call is a no-op and current events are returned.
        senders : ArrayLike or None, optional
            Sender node IDs for each item, shape ``(N,)`` after broadcast,
            dtype ``int64``. ``None`` defaults to ``1`` for all items.
        targets : ArrayLike or None, optional
            Target node IDs for each item, shape ``(N,)`` after broadcast,
            dtype ``int64``. ``None`` defaults to ``1`` for all items.
        receptors : ArrayLike or None, optional
            Receptor IDs (rport) per item, shape ``(N,)`` after broadcast,
            dtype ``int64``. ``None`` defaults to ``0``.
        ports : ArrayLike or None, optional
            Port metadata per item, shape ``(N,)`` after broadcast, dtype
            ``int64``. ``None`` defaults to ``-1``.
        offsets : ArrayLike or None, optional
            Per-item timing offsets :math:`\delta_j` in milliseconds. Flattened
            to shape ``(N,)`` with dtype ``float64``; ``Quantity`` values are
            converted from ``ms``. ``None`` defaults to ``0.0`` ms.
        stamp_steps : ArrayLike or None, optional
            Integer event stamp steps, shape ``(N,)`` after broadcast, dtype
            ``int64``. If ``None``, all items use current step ``+1``.

        Returns
        -------
        out : Any
            Updated events dictionary ``dict[str, np.ndarray]``. If no events
            are accepted by window/filter gates, returns current unchanged
            buffers.

        Raises
        ------
        ValueError
            If simulation ``dt`` is non-positive; if ``t``, ``start``, ``stop``,
            or ``origin`` are not scalar/grid-aligned; if ``stop < start``;
            if payload arrays do not match ``weights`` length after broadcasting;
            or if ``weights``/``offsets`` contain non-finite values.
        TypeError
            If payload values cannot be cast to required numeric types.
        KeyError
            If simulation context values (for example ``t`` or ``dt``) are not
            available from ``brainstate.environ``.
        """
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
