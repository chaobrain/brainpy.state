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
    'volume_transmitter',
]


@dataclass(frozen=True)
class spikecounter:
    r"""Immutable entry in a neuromodulatory spike-history vector.

    Each instance records one on-grid delivery event produced by
    :class:`volume_transmitter`.  The pseudo-spike inserted by
    :meth:`volume_transmitter.init_state` and after each trigger reset
    carries ``multiplicity=0.0``.

    Attributes
    ----------
    spike_time : float
        On-grid spike time in milliseconds, computed as
        :math:`s \cdot \Delta t` where :math:`s` is the delivery stamp index
        and :math:`\Delta t` is the simulation resolution in ms.
    multiplicity : float
        Summed multiplicity of all spikes assigned to ``spike_time``.
        Always ``>= 0.0``; the pseudo-spike inserted at reset has value
        ``0.0``.
    """

    spike_time: float
    multiplicity: float


@dataclass(frozen=True)
class _StepCalibration:
    r"""Immutable discrete-time calibration record used by :class:`volume_transmitter`.

    Computed once per :meth:`volume_transmitter.update` call from the
    simulation environment's ``dt`` and the transmitter's ``min_delay`` and
    ``deliver_interval`` parameters.

    Attributes
    ----------
    dt_ms : float
        Simulation resolution :math:`\Delta t` in milliseconds, derived from
        ``brainstate.environ.get_dt()`` and converted to ms.  Strictly positive.
    min_delay_steps : int
        ``min_delay`` converted to an integer number of simulation steps via
        :math:`\mathrm{round}(d_{\min} / \Delta t)`.  Must be ``>= 1``.
    delivery_period_steps : int
        Trigger period in steps:
        :math:`T_s = \texttt{deliver\_interval} \times d_{\min,s}`.
        A delivery trigger fires when
        :math:`\mathrm{stamp} \bmod T_s = 0`.
    """

    dt_ms: float
    min_delay_steps: int
    delivery_period_steps: int


class volume_transmitter(NESTDevice):
    r"""NEST-compatible ``volume_transmitter`` support device.

    ``volume_transmitter`` collects neuromodulatory spikes and periodically
    exposes their cumulative spike history to dopamine-modulated synapses
    (e.g. ``stdp_dopamine_synapse``).  It reproduces the NEST ring-buffer
    scheduling, trigger logic, and pseudo-spike reset conventions while
    exposing a Python batch-update API.

    **1. Discrete-Time State**

    Let simulation resolution be :math:`\Delta t` (ms) and define the
    on-grid delivery stamp for the current step index
    :math:`n = \mathrm{round}(t / \Delta t)` as :math:`s = n + 1`.
    Internal mutable state consists of:

    - :math:`P[s]` — pending multiplicity map, ``dict[int, float]``,
      accumulating contributions scheduled for future delivery stamp ``s``.
    - :math:`H` — ordered spike-history list of :class:`spikecounter` entries
      :math:`(t_i, m_i)`, with time in ms and non-negative multiplicity.
    - Delivery metadata: ``last_delivery_spikes``, ``last_delivery_time_ms``,
      and ``delivery_count``.

    Immediately after :meth:`init_state`, :math:`H = [(0.0,\; 0.0)]`
    (a NEST-compatible pseudo-spike with zero multiplicity).

    **2. Update Equations and Trigger Rule**

    For each input item :math:`i` with spike indicator :math:`x_i > 0`,
    an effective count :math:`c_i \ge 0` is accumulated into
    :math:`P[s_i]`, where :math:`s_i` is either ``stamp_steps[i]`` or the
    current stamp :math:`s`.

    At each :meth:`update` call, the pending multiplicity for stamp :math:`s`
    is consumed:

    .. math::

       m_s = P[s] \quad (\text{or } 0 \text{ if absent}),
       \qquad
       t_s = s \cdot \Delta t.

    If :math:`m_s > 0`, append :math:`(t_s,\, m_s)` to :math:`H`.

    The delivery period in steps is

    .. math::

       T_s = k \cdot d_{\min,s},
       \qquad
       d_{\min,s} = \mathrm{round}\!\left(\frac{d_{\min}}{\Delta t}\right),

    where :math:`k = \texttt{deliver\_interval}`.  A delivery trigger fires
    when :math:`s \bmod T_s = 0`.

    On trigger:

    1. Capture :math:`D = H` as the delivered history (spikes at stamp
       :math:`s` are included before the reset).
    2. Store :math:`D` and trigger time :math:`t_s` in delivery metadata.
    3. Increment the delivery counter.
    4. Reset :math:`H = [(t_s,\; 0.0)]` (new pseudo-spike).

    **3. Multiplicity Inference**

    Incoming arrays are flattened to one-dimensional vectors of length
    :math:`N`.  Let :math:`x_j` denote ``spikes[j]``:

    - If ``multiplicities is None`` and all :math:`x_j` are integer-like
      (within ``1e-12`` tolerance):
      :math:`c_j = \max(\mathrm{round}(x_j),\; 0)`.
    - If ``multiplicities is None`` and :math:`x_j` contains non-integer
      values: :math:`c_j = \mathbf{1}[x_j > 0]`.
    - If ``multiplicities`` is provided with non-negative integers
      :math:`m_j`: :math:`c_j = m_j \,\mathbf{1}[x_j > 0]`.

    **4. Assumptions and Constraints**

    - ``deliver_interval`` must be a scalar integer ``>= 1``.
    - ``min_delay`` must be scalar, strictly positive, and an integer multiple
      of ``dt``.
    - Simulation time ``t`` must be aligned to the simulation grid (enforced
      at each :meth:`update` call).
    - If ``stamp_steps`` is provided, every entry must satisfy
      ``stamp_steps[i] >= current_stamp``.

    **5. Computational Implications**

    For :math:`N` incoming items per call, scheduling is :math:`O(N)` via
    dictionary accumulation by target stamp.  History memory grows linearly
    with the number of unique stamped events between consecutive triggers.
    Each trigger resets the history to a single pseudo-spike, bounding
    worst-case growth to one trigger period.

    Parameters
    ----------
    in_size : Size, optional
        Shape/size argument consumed by :class:`brainstate.nn.Dynamics`.
        Stored for API compatibility with other device models; it does not
        affect transmitter state-update logic. Default is ``1``.
    deliver_interval : ArrayLike, optional
        Scalar integer-like value (unitless) specifying the trigger period in
        units of ``min_delay``.  Converted via nearest-integer rounding and
        validated to be ``>= 1``.  Increasing this value reduces how often
        connected synapses receive delivered spike histories.
        Default is ``1``.
    min_delay : saiunit.Quantity or float, optional
        Scalar effective global minimal synaptic delay.  Unitful values are
        converted to ms; plain floats are interpreted as ms.  Must be strictly
        positive and an integer multiple of the simulation ``dt`` at the time
        :meth:`update` is called.  Default is ``1.0 * u.ms``.
    name : str or None, optional
        Optional node name forwarded to :class:`brainstate.nn.Dynamics`.
        Default is ``None``.

    Parameter Mapping
    -----------------
    .. list-table:: Mapping of constructor parameters to model symbols
       :header-rows: 1
       :widths: 24 16 22 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``deliver_interval``
         - ``1``
         - :math:`k`
         - Number of minimal-delay intervals per delivery trigger.
       * - ``min_delay``
         - ``1.0 * u.ms``
         - :math:`d_{\min}`
         - Effective global minimal synaptic delay for trigger period.
       * - ``dt`` (environment)
         - runtime
         - :math:`\Delta t`
         - Simulation resolution for stamp conversion and ms time computation.
       * - ``delivery_period_steps``
         - runtime
         - :math:`T_s`
         - :math:`k \cdot \mathrm{round}(d_{\min} / \Delta t)`.

    Raises
    ------
    ValueError
        If ``deliver_interval`` is non-scalar, not integer-like, or ``< 1``
        (raised during ``__init__``).
    ValueError
        At :meth:`update` time: if ``dt <= 0``; if ``min_delay`` is not a
        positive integer multiple of ``dt``; if time ``t`` is not
        grid-aligned; if ``multiplicities`` contains negative values; if
        ``stamp_steps`` contains past stamps; or if payload arrays are
        non-integer where integer values are required or have mismatched
        flattened sizes.
    TypeError
        If provided scalar/array inputs cannot be converted by
        ``saiunit`` or NumPy conversion paths.
    KeyError
        At :meth:`update` time, if the simulation context is missing the
        required ``'t'`` or ``dt`` entries (depends on
        :mod:`brainstate.environ` behaviour).

    Notes
    -----
    - :meth:`deliver_spikes` returns the current (undelivered) history vector.
    - :attr:`last_delivery_spikes` stores the history snapshot delivered at
      the most recent trigger.
    - :attr:`last_delivery_time` stores the most recent trigger time in ms.
    - :meth:`update` aggregates multiplicities exactly by delivery step,
      mirroring NEST's internal ring-buffer logic.
    - :meth:`handles_test_event` accepts only receptor type ``0``, matching
      the NEST ``volume_transmitter`` interface.
    - :meth:`set_local_device_id` / :meth:`get_local_device_id` are provided
      for compatibility with NEST's device duplication logic.

    References
    ----------
    .. [1] NEST Simulator, ``volume_transmitter`` model.
           https://github.com/nest/nest-simulator/blob/master/models/volume_transmitter.cpp

    Examples
    --------
    Instantiate a transmitter with a two-step delivery period, inject two
    simultaneous spikes, and advance one more step:

    .. code-block:: python

       >>> import brainstate
       >>> import saiunit as u
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

    Query transmitter state and delivery metadata via :meth:`get`:

    .. code-block:: python

       >>> import brainstate
       >>> import saiunit as u
       >>> from brainpy.state import volume_transmitter
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     vt = volume_transmitter(deliver_interval=1, min_delay=0.1 * u.ms)
       ...     _ = vt.get('deliver_interval')
       ...     _ = vt.get('spike_history')
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
        r"""Local device ID used for NEST-compatible device-duplication logic.

        Returns
        -------
        int
            Current local device ID as a Python ``int``.
        """
        return int(self._local_device_id)

    @property
    def last_delivery_time(self) -> float:
        r"""Most recent delivery trigger time in milliseconds.

        Returns ``0.0`` if no trigger has occurred since :meth:`init_state`.

        Returns
        -------
        float
            Trigger time :math:`t_s = s \cdot \Delta t` in ms at the most
            recent trigger step, or ``0.0`` before the first trigger.
        """
        return float(self._last_delivery_time_ms)

    @property
    def last_delivery_spikes(self) -> tuple[spikecounter, ...]:
        r"""Spike-history tuple delivered at the most recent trigger.

        Returns an empty tuple before the first trigger.

        Returns
        -------
        tuple[spikecounter, ...]
            Immutable copy of the history vector :math:`H` that was captured
            at the most recent delivery trigger, including the pseudo-spike at
            the trigger stamp.  Empty tuple if no trigger has fired yet.
        """
        return tuple(self._last_delivery_spikes)

    @property
    def n_deliveries(self) -> int:
        r"""Number of completed delivery triggers since initialization.

        Returns
        -------
        int
            Count of trigger events that have fired since the last
            :meth:`init_state` call.  Increments only when
            :math:`s \bmod T_s = 0` and the history vector is non-empty.
        """
        return int(self._delivery_count)

    def set_local_device_id(self, ldid: ArrayLike):
        r"""Set the local device ID from a scalar integer-like value.

        Parameters
        ----------
        ldid : ArrayLike
            Scalar integer-like value for the new local device ID.  Converted
            via nearest-integer rounding; non-integer values raise
            ``ValueError``.

        Raises
        ------
        ValueError
            If ``ldid`` is non-scalar or not integer-like.
        TypeError
            If ``ldid`` cannot be converted to a numeric array.
        """
        self._local_device_id = int(self._to_int_scalar(ldid, name='local_device_id'))

    def get_local_device_id(self) -> int:
        r"""Return the current local device ID as a Python ``int``.

        Returns
        -------
        int
            Current value of the local device ID.
        """
        return int(self._local_device_id)

    def handles_test_event(self, receptor_type: ArrayLike) -> int:
        r"""Validate a receptor type identifier and return the accepted index.

        Mirrors the NEST ``volume_transmitter::handles_test_event`` method.
        Only receptor type ``0`` is accepted; all other values raise
        ``ValueError``.

        Parameters
        ----------
        receptor_type : ArrayLike
            Scalar integer-like receptor identifier to validate.

        Returns
        -------
        int
            Always ``0`` when the receptor type is valid.

        Raises
        ------
        ValueError
            If ``receptor_type`` is non-scalar, not integer-like, or not
            equal to ``0``.
        TypeError
            If ``receptor_type`` cannot be converted to a numeric array.
        """
        r = int(self._to_int_scalar(receptor_type, name='receptor_type'))
        if r != 0:
            raise ValueError(f'Unknown receptor_type {r} for volume_transmitter.')
        return 0

    def deliver_spikes(self) -> tuple[spikecounter, ...]:
        r"""Return the current (undelivered) spike-history vector.

        The history always contains at least one entry: the pseudo-spike
        ``spikecounter(t, 0.0)`` inserted by :meth:`init_state` or the most
        recent trigger reset.

        Returns
        -------
        tuple[spikecounter, ...]
            Immutable copy of the current internal history list :math:`H`,
            ordered chronologically by delivery stamp.
        """
        return tuple(self._spikecounter)

    def get(self, key: str = 'deliver_interval'):
        r"""Query transmitter parameters and mutable state by string key.

        Parameters
        ----------
        key : str, optional
            Selector string.  Supported values:

            - ``'deliver_interval'`` — returns ``int``.
            - ``'min_delay'`` — returns the stored ``min_delay`` value as
              passed to the constructor (``saiunit.Quantity`` or ``float``).
            - ``'local_device_id'`` — returns ``int``.
            - ``'spike_history'`` — returns ``tuple[spikecounter, ...]``
              (same as :meth:`deliver_spikes`).
            - ``'last_delivery_spikes'`` — returns ``tuple[spikecounter, ...]``
              (same as :attr:`last_delivery_spikes`).
            - ``'last_delivery_time'`` — returns ``float`` ms.
            - ``'n_deliveries'`` — returns ``int``.

            Default is ``'deliver_interval'``.

        Returns
        -------
        int or float or saiunit.Quantity or tuple[spikecounter, ...]
            The selected value.  Type depends on ``key`` as described above.

        Raises
        ------
        KeyError
            If ``key`` is not one of the supported selector strings.
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
        r"""Reset all queue and history state to NEST-compatible initial conditions.

        Clears the pending-multiplicity map :math:`P`, resets the spike-history
        vector :math:`H` to the single pseudo-spike ``spikecounter(0.0, 0.0)``,
        and resets all delivery metadata (``last_delivery_spikes``,
        ``last_delivery_time_ms``, ``delivery_count``) to their initial values.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused placeholder required by the :class:`brainstate.nn.Dynamics`
            API.  Ignored. Default is ``None``.
        **kwargs
            Additional keyword arguments accepted for API compatibility and
            silently ignored.

        """
        del batch_size, kwargs
        self._pending_multiplicities.clear()
        self._spikecounter = [spikecounter(0.0, 0.0)]
        self._last_delivery_spikes = ()
        self._last_delivery_time_ms = 0.0
        self._delivery_count = 0

    def connect(self):
        r"""No-op compatibility hook matching the NEST device interface.

        Provided so that code that calls ``connect()`` on NEST devices works
        without modification when targeting :class:`volume_transmitter`.


        """
        return None

    def flush(self):
        r"""Return a non-triggering snapshot of the current state.

        Unlike :meth:`update`, this method does not advance the simulation
        step, consume pending multiplicities, or reset the history.  It is
        useful for inspecting state between :meth:`update` calls (e.g. at
        the end of a simulation run).

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``'triggered'`` — ``False`` (no trigger occurs on flush).
            - ``'t_trig'`` — ``None`` (no trigger time).
            - ``'delivered_spikes'`` — empty ``tuple`` (no delivery).
            - ``'spike_history'`` — ``tuple[spikecounter, ...]``, the current
              history returned by :meth:`deliver_spikes`.
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

        Reads the current simulation time ``t`` and resolution ``dt`` from
        :mod:`brainstate.environ`, schedules incoming spike contributions into
        the pending map :math:`P`, consumes the current-stamp entry, optionally
        appends a new history entry, and evaluates the delivery trigger.

        Parameters
        ----------
        spikes : ArrayLike or None, optional
            Scalar or 1-D array of spike event indicators/counts for the
            current call, shape ``(N,)`` after flattening.  Unitful inputs are
            accepted; only the mantissa is used.  Multiplicity inference rules:

            - ``multiplicities is None`` and all values are integer-like
              (within ``1e-12``): :math:`c_j = \max(\mathrm{round}(x_j),\; 0)`.
            - ``multiplicities is None`` and values contain non-integers:
              :math:`c_j = \mathbf{1}[x_j > 0]` (binary threshold).
            - ``multiplicities`` provided: :math:`c_j = m_j \,\mathbf{1}[x_j > 0]`.

            ``None`` means no incoming events for this step.
        multiplicities : ArrayLike or None, optional
            Scalar or 1-D integer-like array, shape ``(N,)`` matching the
            flattened size of ``spikes``.  Each value must be non-negative.
            Applied only where ``spikes[j] > 0``; non-positive spike indicators
            force zero contribution regardless of ``multiplicities[j]``.
            ``None`` enables implicit inference from ``spikes``.
        stamp_steps : ArrayLike or None, optional
            Scalar or 1-D integer-like array, shape ``(N,)`` matching the
            flattened size of ``spikes``.  Values are absolute delivery-stamp
            indices in step-space and must satisfy
            ``stamp_steps[j] >= current_stamp`` (past stamps raise
            ``ValueError``).  ``None`` assigns all contributions to the
            current stamp :math:`s`.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``'triggered'`` — ``bool``: whether the current stamp fires the
              delivery trigger (:math:`s \bmod T_s = 0`).
            - ``'t_trig'`` — ``float`` ms or ``None``: trigger time
              :math:`t_s = s \cdot \Delta t` if triggered, else ``None``.
            - ``'delivered_spikes'`` — ``tuple[spikecounter, ...]``: history
              :math:`H` captured before the trigger reset, or empty tuple if
              not triggered.
            - ``'spike_history'`` — ``tuple[spikecounter, ...]``: current
              history after all processing (post-reset pseudo-spike if
              triggered).

        Raises
        ------
        ValueError
            If ``dt <= 0``; if ``min_delay`` is not a positive integer multiple
            of ``dt``; if ``t`` is not grid-aligned to ``dt``; if
            ``multiplicities`` contains negative values; if ``stamp_steps``
            contains stamps earlier than the current stamp; or if any array
            payload is non-integer where integer values are required.
        ValueError
            If ``spikes``, ``multiplicities``, or ``stamp_steps`` are not
            scalar or 1-D, or have mismatched flattened sizes.
        TypeError
            If numeric or unit conversion fails for any payload or environment
            time value.
        KeyError
            If required environment values (``'t'`` or ``dt``) are unavailable
            from :mod:`brainstate.environ`.

        Notes
        -----
        Trigger evaluation uses stamp :math:`s = n + 1` (one ahead of the step
        index) and period :math:`T_s = k \cdot d_{\min,s}`.  Spikes stamped
        exactly at the trigger stamp are included in ``'delivered_spikes'``
        before the history reset, matching NEST ordering semantics.

        Examples
        --------
        Schedule spikes at a future stamp and confirm delivery:

        .. code-block:: python

           >>> import brainstate
           >>> import saiunit as u
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
            ditype = brainstate.environ.ditype()
            stamp_arr = np.full((n_items,), stamp_now, dtype=ditype)
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
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be a scalar time value.')
        val = float(arr[0])
        if (not allow_inf) and (not math.isfinite(val)):
            raise ValueError(f'{name} must be finite.')
        return val

    @staticmethod
    def _to_int_scalar(value, name: str) -> int:
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
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
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(x), dtype=dftype).reshape(-1)
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
