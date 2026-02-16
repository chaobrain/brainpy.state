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
    """Discrete-time calibration used by :class:`volume_transmitter`.

    Attributes
    ----------
    dt_ms : float
        Simulation resolution in milliseconds.
    min_delay_steps : int
        ``min_delay`` converted to integer simulation steps.
    delivery_period_steps : int
        Trigger period in steps, equal to
        ``deliver_interval * min_delay_steps``.
    """

    dt_ms: float
    min_delay_steps: int
    delivery_period_steps: int


class volume_transmitter(brainstate.nn.Dynamics):
    r"""NEST-compatible ``volume_transmitter`` support device.

    Short description
    -----------------
    ``volume_transmitter`` collects neuromodulatory spikes and periodically
    exposes their cumulative spike history to dopamine-modulated synapses.

    Description
    -----------
    **1. Discrete-time model and state**

    Let simulation resolution be :math:`\Delta t` in ms, and define the
    on-grid delivery stamp for current step ``step`` as
    :math:`n = step + 1`. Internal mutable state is:

    - :math:`P[s]`: pending multiplicity scheduled for delivery stamp ``s``
      (implemented as ``dict[int, float]``).
    - :math:`H`: ordered spike-history vector of entries
      :math:`(t_i, m_i)` with time in ms and multiplicity.
    - delivery metadata: latest delivered history, trigger time, and count.

    Immediately after :meth:`init_state`, :math:`H=[(0, 0)]` (NEST pseudo
    spike).

    **2. Update equations and trigger rule**

    For each input item ``i`` in :meth:`update`, an effective count
    :math:`c_i \ge 0` is added to :math:`P[s_i]`, where ``s_i`` is either
    ``stamp_steps[i]`` or current stamp ``n``.

    At each call:

    .. math::

       m_n = P[n] \;\;(\text{or } 0 \text{ if absent}),
       \qquad
       t_n = n \Delta t.

    If :math:`m_n > 0`, append :math:`(t_n, m_n)` to :math:`H`.

    Delivery period is
    :math:`T_s = \texttt{deliver\_interval} \cdot d_{\min,s}`, where
    :math:`d_{\min,s} = \mathrm{round}(d_{\min}/\Delta t)` is ``min_delay``
    in steps. A trigger occurs when :math:`n \bmod T_s = 0`.

    On trigger:

    - return delivered history :math:`D = H`,
    - store delivery metadata,
    - reset history to one pseudo spike :math:`H=[(t_n, 0)]`.

    This preserves NEST ordering where spikes stamped at the trigger step are
    included in the delivered history before reset.

    **3. Assumptions and constraints**

    - ``deliver_interval`` must be scalar integer ``>= 1``.
    - ``min_delay`` must be scalar positive and an integer multiple of
      ``dt``.
    - simulation time ``t`` must be aligned to the simulation grid.
    - if ``stamp_steps`` is provided, every stamp must satisfy
      ``stamp_steps >= current_stamp``.

    **4. Computational implications**

    For ``N`` incoming entries in one call, scheduling is :math:`O(N)` with
    dictionary accumulation by target stamp. History memory grows with the
    number of unique stamped events between consecutive triggers.

    Parameters
    ----------
    in_size : Size, optional
        Dynamics size/shape specification consumed by
        :class:`brainstate.nn.Dynamics`. The value is stored for compatibility
        with other devices; it does not change transmitter state-update logic.
        Default is ``1``.
    deliver_interval : ArrayLike, optional
        Scalar integer-like value (unitless) defining trigger period in units
        of ``min_delay``. Converted with nearest-integer validation and must be
        ``>= 1``. Default is ``1``.
    min_delay : ArrayLike, optional
        Scalar delay representing effective global minimal synaptic delay.
        Unitful values are converted to ms; unitless values are interpreted as
        ms. Must be strictly positive and grid-aligned to current ``dt`` when
        :meth:`update` is executed. Default is ``1.0 * u.ms``.
    name : str or None, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 24 16 24 36

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``deliver_interval``
         - ``1``
         - :math:`k`
         - Number of minimal-delay intervals per trigger.
       * - ``min_delay``
         - ``1.0 * u.ms``
         - :math:`d_{\min}`
         - Effective global minimal synaptic delay used for trigger period.
       * - ``dt`` (environment)
         - runtime
         - :math:`\Delta t`
         - Simulation resolution used to compute stamp index and ms times.
       * - ``delivery_period_steps``
         - runtime
         - :math:`T_s`
         - :math:`k \cdot \mathrm{round}(d_{\min}/\Delta t)`.

    Returns
    -------
    out : Any
        Dynamics node instance. Calling :meth:`update` returns a dictionary
        with trigger flag, trigger time (ms or ``None``), delivered spike
        history, and current spike history.

    Raises
    ------
    ValueError
        If ``deliver_interval`` is non-scalar or not integer-like, or if
        ``deliver_interval < 1``.
    ValueError
        At update time, if ``dt <= 0``, if ``min_delay`` is not a positive
        integer multiple of ``dt``, if time ``t`` is not grid-aligned, or if
        invalid ``multiplicities``/``stamp_steps`` payloads are provided.
    KeyError
        At update time, if simulation context is missing required entries
        (typically ``'t'`` or ``dt``), depending on
        :mod:`brainstate.environ` behavior.
    TypeError
        If provided scalar/array inputs cannot be converted by
        ``brainunit``/NumPy conversion paths.

    Notes
    -----
    - :meth:`deliver_spikes` returns current history vector.
    - ``last_delivery_spikes`` stores history delivered at the most recent
      trigger.
    - ``last_delivery_time`` stores the most recent trigger time in ms.
    - :meth:`update` aggregates multiplicities exactly by delivery step, as
      NEST's internal ring-buffer logic does.
    - ``handles_test_event`` accepts only receptor type ``0``, matching NEST.
    - ``set_local_device_id`` / ``get_local_device_id`` are provided for
      compatibility with NEST's device duplication logic.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> import numpy as np
       >>> from brainpy.state import volume_transmitter
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     vt = volume_transmitter(deliver_interval=2, min_delay=0.3 * u.ms)
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         y0 = vt.update(
       ...             spikes=np.array([1.0, 1.0]),
       ...             multiplicities=np.array([1, 2]),
       ...         )
       ...     with brainstate.environ.context(t=0.5 * u.ms):
       ...         y1 = vt.update()
       ...     _ = (y0['triggered'], y1['triggered'])

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
        """Return local device id used for NEST-compatible duplication logic."""
        return int(self._local_device_id)

    @property
    def last_delivery_time(self) -> float:
        """Return most recent trigger time in milliseconds."""
        return float(self._last_delivery_time_ms)

    @property
    def last_delivery_spikes(self) -> tuple[spikecounter, ...]:
        """Return spike-history tuple delivered at the most recent trigger."""
        return tuple(self._last_delivery_spikes)

    @property
    def n_deliveries(self) -> int:
        """Return number of completed delivery triggers since initialization."""
        return int(self._delivery_count)

    def set_local_device_id(self, ldid: ArrayLike):
        """Set local device id from a scalar integer-like value."""
        self._local_device_id = int(self._to_int_scalar(ldid, name='local_device_id'))

    def get_local_device_id(self) -> int:
        """Get current local device id as Python ``int``."""
        return int(self._local_device_id)

    def handles_test_event(self, receptor_type: ArrayLike) -> int:
        """Validate receptor id and return accepted receptor index.

        Parameters
        ----------
        receptor_type : ArrayLike
            Scalar integer-like receptor identifier. Only ``0`` is accepted.

        Returns
        -------
        out : Any
            Integer receptor index ``0``.

        Raises
        ------
        ValueError
            If ``receptor_type`` is non-scalar, non-integer-like, or not ``0``.
        """
        r = int(self._to_int_scalar(receptor_type, name='receptor_type'))
        if r != 0:
            raise ValueError(f'Unknown receptor_type {r} for volume_transmitter.')
        return 0

    def deliver_spikes(self) -> tuple[spikecounter, ...]:
        """Return current spike-history vector.

        Returns
        -------
        out : Any
            Tuple of :class:`spikecounter` entries representing current
            undelivered history. Includes one pseudo spike immediately after
            :meth:`init_state` and after each trigger reset.
        """
        return tuple(self._spikecounter)

    def get(self, key: str = 'deliver_interval'):
        """Query transmitter parameters and mutable state by key.

        Parameters
        ----------
        key : str, optional
            Selector key. Supported values are ``'deliver_interval'``,
            ``'min_delay'``, ``'local_device_id'``, ``'spike_history'``,
            ``'last_delivery_spikes'``, ``'last_delivery_time'``, and
            ``'n_deliveries'``. Default is ``'deliver_interval'``.

        Returns
        -------
        out : Any
            Selected scalar, quantity, or tuple depending on ``key``.

        Raises
        ------
        KeyError
            If ``key`` is not one of the supported selectors.
        """
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
        """Reset queue/history state to NEST-compatible initial conditions.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused placeholder for :class:`brainstate.nn.Dynamics` API
            compatibility.
        **kwargs
            Unused keyword arguments for API compatibility.

        Returns
        -------
        out : Any
            ``None``. Internal state is reset in-place.
        """
        del batch_size, kwargs
        self._pending_multiplicities.clear()
        self._spikecounter = [spikecounter(0.0, 0.0)]
        self._last_delivery_spikes = ()
        self._last_delivery_time_ms = 0.0
        self._delivery_count = 0

    def connect(self):
        """No-op compatibility hook for NEST-style device interface."""
        return None

    def flush(self):
        """Return a non-triggering snapshot payload for integration code.

        Returns
        -------
        out : Any
            Dictionary with keys ``'triggered'``, ``'t_trig'``,
            ``'delivered_spikes'``, and ``'spike_history'`` representing the
            current history without consuming or resetting it.
        """
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
        r"""Advance transmitter state by one simulation step.

        Parameters
        ----------
        spikes : ArrayLike or None, optional
            Scalar or 1-D array of event indicators/count-like values for the
            current call, shape ``(N,)`` after flattening. Unitful inputs are
            accepted but only mantissas are used. Semantics:

            - if ``multiplicities is None`` and ``spikes`` is integer-like,
              each item contributes ``max(round(spikes[i]), 0)`` events;
            - otherwise each item contributes ``1`` when ``spikes[i] > 0`` and
              ``0`` when ``spikes[i] <= 0``.

            ``None`` means no incoming events.
        multiplicities : ArrayLike or None, optional
            Scalar or 1-D integer-like array with shape ``(N,)`` matching
            ``spikes``. Each value must be non-negative. Applied only where
            ``spikes[i] > 0``; non-positive ``spikes`` force zero contribution.
            ``None`` enables implicit multiplicity inference from ``spikes``.
        stamp_steps : ArrayLike or None, optional
            Scalar or 1-D integer-like array with shape ``(N,)`` matching
            ``spikes``. Values are absolute delivery-stamp indices (step-space)
            and must satisfy ``stamp_steps >= current_stamp``. ``None`` assigns
            all contributions to current stamp.

        Returns
        -------
        out : Any
            Dictionary with keys:

            - ``'triggered'``: ``bool``, whether this step triggers delivery;
            - ``'t_trig'``: ``float`` ms trigger time or ``None``;
            - ``'delivered_spikes'``: tuple of :class:`spikecounter` entries
              delivered at this trigger, else empty tuple;
            - ``'spike_history'``: current history tuple after step processing
              (post-reset history if triggered).

        Raises
        ------
        ValueError
            If ``dt`` is non-positive, ``min_delay`` is not a positive multiple
            of ``dt``, ``t`` is not grid-aligned, ``multiplicities`` contains
            negative values, ``stamp_steps`` contains past steps, or provided
            arrays are non-integer where integer values are required.
        ValueError
            If ``spikes``, ``multiplicities``, or ``stamp_steps`` are not
            scalar/1-D or have mismatched flattened sizes.
        TypeError
            If numeric/unit conversion fails for payload values or environment
            time values.
        KeyError
            If required environment values (``'t'`` or ``dt``) are unavailable,
            depending on :mod:`brainstate.environ` behavior.

        Notes
        -----
        Trigger evaluation uses current stamp ``n = step + 1`` and period
        ``deliver_interval * min_delay_steps``. Spikes stamped at a trigger
        stamp are included in ``delivered_spikes`` before history reset.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate
           >>> import brainunit as u
           >>> import numpy as np
           >>> from brainpy.state import volume_transmitter
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     vt = volume_transmitter(deliver_interval=1, min_delay=0.2 * u.ms)
           ...     with brainstate.environ.context(t=0.0 * u.ms):
           ...         out0 = vt.update(
           ...             spikes=np.array([1.0, 1.0, 0.0]),
           ...             multiplicities=np.array([2, 3, 7]),
           ...             stamp_steps=np.array([2, 2, 2]),
           ...         )
           ...     with brainstate.environ.context(t=0.1 * u.ms):
           ...         out1 = vt.update()
           ...     _ = (out0['triggered'], out1['delivered_spikes'])
        """
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
