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

from brainpy_state._nest_base.base import NESTDevice

__all__ = [
    'correlospinmatrix_detector',
]


@dataclass
class _BinaryPulse:
    t_on: int
    t_off: int
    receptor_channel: int


@dataclass
class _Calibration:
    dt_ms: float
    t_min_steps: int
    t_max_steps: float
    delta_tau_ms: float
    delta_tau_steps: int
    tau_max_ms: float
    tau_max_steps: int
    tau_edge_steps: int
    tstart_ms: float
    tstop_ms: float
    n_channels: int
    n_bins: int
    min_delay_steps: int
    signature: tuple


class correlospinmatrix_detector(NESTDevice):
    r"""NEST-compatible ``correlospinmatrix_detector`` device.

    **1. Overview**

    ``correlospinmatrix_detector`` receives binary-state spike streams from
    multiple receptor channels and accumulates raw auto/cross covariance
    histograms over negative, zero, and positive lags. It mirrors NEST
    ``models/correlospinmatrix_detector.{h,cpp}`` for event decoding,
    pulse finalization, and lag-bin accumulation.

    **2. Binary-State Decoding and Pulse Construction**

    For receptor channel :math:`c`, define a binary state
    :math:`x_c(t)\in\{0,1\}` on integer simulation steps.
    Runtime events are triples :math:`e=(c, t, m)`, where :math:`m` is
    multiplicity.

    Decoding rule (NEST compatible):

    - :math:`m=1` — mark a tentative down-transition.
    - A second event with identical ``(channel, stamp_step)`` or
      :math:`m=2`: confirm up-transition (:math:`x_c\leftarrow 1`) and cancel
      tentative-down handling for that event pair.

    A covariance update is triggered only when a previous tentative
    down-transition becomes confirmed. The finalized pulse is
    :math:`p_i=(i, t_i^{\mathrm{on}}, t_i^{\mathrm{off}})`, where
    :math:`t_i^{\mathrm{on}}` is taken from ``_last_change[i]`` and
    :math:`t_i^{\mathrm{off}}` from the confirmed down-transition stamp.

    **3. Lag-Bin Accumulation Equations**

    Let :math:`\Delta=\Delta_{\tau,\mathrm{steps}}`,
    :math:`H=\tau_{\max,\mathrm{steps}}/\Delta`, and
    :math:`k_0=H` (zero-lag bin index).
    For each finalized pulse :math:`p_i`, iterate retained history pulses
    :math:`p_j=(j, t_j^{\mathrm{on}}, t_j^{\mathrm{off}})`.

    Integer lag offsets are constrained by

    .. math::

       \delta_{\min}=\max\!\bigl(t_j^{\mathrm{on}}-t_i^{\mathrm{off}},
       -\tau_{\max,\mathrm{steps}}\bigr),\qquad
       \delta_{\max}=\min\!\bigl(t_j^{\mathrm{off}}-t_i^{\mathrm{on}},
       \tau_{\max,\mathrm{steps}}\bigr).

    For an offset :math:`\delta`, overlap length in simulation steps is

    .. math::

       L_{ij}(\delta)=
       \min\!\bigl(t_i^{\mathrm{off}},\; t_j^{\mathrm{off}}-\delta\Delta\bigr)
       -\max\!\bigl(t_i^{\mathrm{on}},\; t_j^{\mathrm{on}}-\delta\Delta\bigr).

    If :math:`L_{ij}(\delta)>0`, add this integer duration to
    ``count_covariance`` bins:

    - **Zero lag** (:math:`\delta=0`): update ``(i, j, k_0)`` and, when
      :math:`i\neq j`, the mirrored entry ``(j, i, k_0)``.
    - **Negative offsets** (:math:`\delta<0`): update both mirrored bins
      ``(i, j, k_0-\delta)`` and ``(j, i, k_0+\delta)``.
    - **Positive offsets** (:math:`\delta>0`): update mirrored bins only
      for :math:`i\neq j`, matching NEST triangular-edge conventions.

    ``count_covariance`` has shape ``(N_channels, N_channels, N_bins)`` with

    .. math::

       N_{\mathrm{bins}} = 1 + 2\,\frac{\tau_{\max,\mathrm{steps}}}
                                        {\Delta_{\tau,\mathrm{steps}}},

    and stores overlap lengths in simulation-step units (not milliseconds).

    **4. Windowing, Assumptions, and Constraints**

    Activity filtering follows the half-open interval
    ``(origin + start, origin + stop]`` in step space. Events outside this
    interval are discarded before decoding.

    ``Tstart`` and ``Tstop`` are validated and included in calibration
    signatures for reset behavior, but they do not gate accumulation in
    :meth:`update`, matching current NEST source behavior.

    Calibration constraints:

    - ``dt > 0``.
    - ``start``, ``stop`` (if finite), ``origin``, ``delta_tau``, and
      ``tau_max`` must each align exactly to integer multiples of ``dt``.
    - ``delta_tau`` must be finite and strictly positive.
    - ``tau_max`` must be finite, non-negative, and divisible by
      ``delta_tau``.
    - ``N_channels >= 1``; runtime receptor IDs must be in
      ``[0, N_channels - 1]``.

    **5. Computational Implications**

    For each accepted event, down-transition handling is constant-time, while
    pulse insertion and pulse-to-history correlation are linear in the retained
    queue length :math:`Q`. Memory scales as
    :math:`O\!\left(Q + N_{\mathrm{channels}}^2 N_{\mathrm{bins}}\right)`.
    Queue pruning uses NEST minimum-delay semantics with
    ``min_delay_steps = 1``.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape metadata used by :class:`brainstate.nn.Dynamics`.
        The detector is event-driven and stores internal tensors, so
        ``in_size`` does not change ``count_covariance`` shape.
        Default is ``1``.
    delta_tau : quantity (ms) or float or None, optional
        Lag-bin width :math:`\Delta_\tau`. Unitful ``brainunit`` quantities are
        accepted and converted to ms; bare floats are interpreted as ms.
        Must be finite, strictly positive, and an integer multiple of
        simulation ``dt``. ``None`` auto-selects ``dt``.
        Default is ``None``.
    tau_max : quantity (ms) or float or None, optional
        One-sided lag horizon :math:`\tau_{\max}`. Unitful quantities accepted.
        Must be finite, non-negative, an integer multiple of ``dt``, and an
        exact integer multiple of ``delta_tau``. ``None`` auto-selects
        ``10 * delta_tau``. Default is ``None``.
    Tstart : quantity (ms) or float, optional
        Non-negative scalar lower time bound in ms retained for NEST API
        compatibility. Participates in calibration signature and triggers a
        state reset when changed. Default is ``0.0 * u.ms``.
    Tstop : quantity (ms) or float or None, optional
        Non-negative scalar upper time bound in ms retained for NEST API
        compatibility. ``None`` means :math:`+\infty`. Participates in
        calibration signature and triggers a state reset when changed.
        Default is ``None``.
    N_channels : int or ArrayLike, optional
        Number of receptor channels. Must resolve to a scalar integer ``>= 1``.
        Runtime channel IDs must be in ``[0, N_channels - 1]``.
        Default is ``1``.
    start : quantity (ms) or float, optional
        Exclusive lower bound of the activity window relative to ``origin``
        in ms. Must be scalar-convertible and aligned to simulation ``dt``.
        Default is ``0.0 * u.ms``.
    stop : quantity (ms) or float or None, optional
        Inclusive upper bound of the activity window relative to ``origin``
        in ms. Must be scalar-convertible and aligned to ``dt`` when finite.
        ``None`` means :math:`+\infty`. Default is ``None``.
    origin : quantity (ms) or float, optional
        Activity-window origin shift in ms. The effective activity window
        becomes ``(origin + start, origin + stop]``. Must be
        scalar-convertible and aligned to ``dt``.
        Default is ``0.0 * u.ms``.
    name : str or None, optional
        Optional node name forwarded to :class:`brainstate.nn.Dynamics`.
        If ``None``, a name is auto-generated. Default is ``None``.

    Parameter Mapping
    -----------------

    .. list-table::
       :header-rows: 1
       :widths: 18 16 24 42

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``delta_tau``
         - ``None`` → ``dt``
         - :math:`\Delta_\tau`
         - Lag-bin width; auto-resolved to simulation ``dt`` when omitted.
       * - ``tau_max``
         - ``None`` → ``10 * delta_tau``
         - :math:`\tau_{\max}`
         - One-sided lag horizon; auto-resolved when omitted.
       * - ``Tstart``
         - ``0.0 ms``
         - :math:`T_{\mathrm{start}}`
         - Calibration/reset compatibility parameter (not a runtime gate).
       * - ``Tstop``
         - ``None`` (:math:`+\infty`)
         - :math:`T_{\mathrm{stop}}`
         - Calibration/reset compatibility parameter (not a runtime gate).
       * - ``N_channels``
         - ``1``
         - :math:`N_{\mathrm{channels}}`
         - Number of receptor channels and covariance matrix axes.
       * - ``start``
         - ``0.0 ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower bound of the activity window.
       * - ``stop``
         - ``None`` (:math:`+\infty`)
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound of the activity window.
       * - ``origin``
         - ``0.0 ms``
         - :math:`t_0`
         - Global shift applied to ``start`` and ``stop`` boundaries.

    Raises
    ------
    ValueError
        If scalar parameters are non-scalar, non-finite where finite values
        are required, misaligned to simulation resolution, or violate
        constraints (e.g. ``tau_max % delta_tau != 0``, ``stop < start``,
        ``N_channels < 1``, invalid runtime receptor IDs, negative
        multiplicities, non-finite ``spikes``, or mismatched event-array
        sizes).
    KeyError
        If runtime environment keys such as simulation time ``'t'`` or ``dt``
        are unavailable during calibration or update.

    Notes
    -----
    - Connection delays and weights are ignored, matching NEST.
    - Runtime events are provided through :meth:`update` arrays
      (``spikes``, ``receptor_ports``/``receptor_types``, ``multiplicities``,
      ``stamp_steps``), each scalar-broadcastable to one event axis.
    - Optional ``multiplicities`` emulate NEST ``SpikeEvent`` multiplicity.
    - History pruning uses NEST minimum-delay semantics with
      ``min_delay_steps = 1``.
    - Calibration is cached and reused across steps; it is automatically
      invalidated if ``dt`` or any window parameter changes between calls.

    Examples
    --------
    Two-channel detector with explicit events:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     det = brainpy.state.correlospinmatrix_detector(
       ...         N_channels=2,
       ...         delta_tau=1.0 * u.ms,
       ...         tau_max=3.0 * u.ms,
       ...     )
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         det.update(
       ...             spikes=np.array([1.0, 1.0]),
       ...             receptor_ports=np.array([0, 0]),
       ...             multiplicities=np.array([1, 1]),
       ...             stamp_steps=np.array([1, 1]),
       ...         )
       ...     out = det.flush()
       >>> out['count_covariance'].shape
       (2, 2, 7)

    Default parameters with no input events and explicit state reset:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     det = brainpy.state.correlospinmatrix_detector()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = det.update()  # no events; returns current state
       ...     det.init_state()  # explicit reset

    References
    ----------
    .. [1] NEST Simulator, ``correlospinmatrix_detector`` model.
           https://nest-simulator.readthedocs.io/en/stable/models/correlospinmatrix_detector.html
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
        name: str | None = None,
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
        self._calib_param_ids: tuple | None = None
        self._incoming: deque[_BinaryPulse] = deque()

        self._last_i = 0
        self._t_last_in_spike = -2 ** 62
        self._tentative_down = False
        self._curr_state = np.zeros((0,), dtype=np.bool_)
        ditype = brainstate.environ.ditype()
        self._last_change = np.zeros((0,), dtype=ditype)
        self._count_covariance = np.zeros((0, 0, 0), dtype=ditype)

        self._ensure_calibrated_from_env_if_available()

    @property
    def count_covariance(self) -> np.ndarray:
        r"""Return accumulated raw covariance histogram.

        Returns
        -------
        np.ndarray
            ``int64`` tensor with shape ``(N_channels, N_channels, N_bins)``.
            Entries are accumulated overlap durations measured in simulation
            steps.
        """
        self._ensure_calibrated_from_env_if_available()
        ditype = brainstate.environ.ditype()
        return np.asarray(self._count_covariance, dtype=ditype)

    def get(self, key: str = 'count_covariance'):
        r"""Read a detector attribute using NEST-style string keys.

        Parameters
        ----------
        key : str, optional
            Attribute name. Supported values are ``'count_covariance'``,
            ``'delta_tau'``, ``'tau_max'``, ``'Tstart'``, ``'Tstop'``,
            ``'N_channels'``, ``'start'``, ``'stop'``, and ``'origin'``.
            Default is ``'count_covariance'``.

        Returns
        -------
        out : dict
            Requested value. Time-valued scalars are returned in milliseconds,
            ``count_covariance`` is returned as ``np.ndarray[int64]``.

        Raises
        ------
        KeyError
            If ``key`` is unsupported.
        ValueError
            If stored parameter values fail scalar/unit conversion checks.
        """
        if key == 'count_covariance':
            return self.count_covariance
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
        raise KeyError(f'Unsupported key "{key}" for correlospinmatrix_detector.get().')

    def connect(self):
        r"""Compatibility no-op for device connection API.

        """
        return None

    def flush(self):
        r"""Return current detector outputs without mutating state.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping with one key, ``'count_covariance'``, containing the
            current ``int64`` covariance tensor.
        """
        return {
            'count_covariance': self.count_covariance,
        }

    def init_state(self, batch_size: int | None = None, **kwargs):
        r"""Reset internal detector state.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused. Present for :class:`brainstate.nn.Dynamics` compatibility.
        **kwargs
            Unused keyword arguments.
        """
        del batch_size, kwargs
        self._ensure_calibrated_from_env_if_available()
        self._reset_state()

    def update(
        self,
        spikes: ArrayLike = None,
        receptor_ports: ArrayLike = None,
        receptor_types: ArrayLike = None,
        multiplicities: ArrayLike = None,
        stamp_steps: ArrayLike = None,
    ):
        r"""Process one simulation tick worth of incoming events.

        Parameters
        ----------
        spikes : ArrayLike or None, optional
            Event-presence values for this call. Scalar or 1-D array of shape
            ``(n_events,)``. Values ``<= 0`` are ignored. If
            ``multiplicities is None``, integer-like positive values are used
            as multiplicities; otherwise positive entries act as an event mask.
            ``None`` means no new events.
        receptor_ports : ArrayLike or None, optional
            Receptor channel IDs, scalar-broadcastable or shape
            ``(n_events,)``. Required when events target channels other than
            ``0``. Each value must be an integer in
            ``[0, N_channels - 1]``.
        receptor_types : ArrayLike or None, optional
            Alias of ``receptor_ports`` for NEST naming compatibility. Used
            only when ``receptor_ports is None``.
        multiplicities : ArrayLike or None, optional
            Non-negative integer multiplicities, scalar-broadcastable or shape
            ``(n_events,)``. If provided, effective multiplicity is
            ``multiplicities[i]`` for positive ``spikes[i]`` and ``0``
            otherwise. ``None`` triggers multiplicity inference from
            ``spikes``.
        stamp_steps : ArrayLike or None, optional
            Integer simulation stamps (1-based step count), scalar-broadcastable
            or shape ``(n_events,)``. ``None`` uses ``current_step + 1`` for
            all events.

        Returns
        -------
        out : jax.Array
            Mapping ``{'count_covariance': ndarray}``, where the ndarray is
            ``int64`` with shape ``(N_channels, N_channels, N_bins)``.

        Raises
        ------
        ValueError
            If runtime arrays are non-finite, negative where forbidden
            (for example ``multiplicities``), size-incompatible, or contain
            unknown receptor channels.
        KeyError
            If simulation time ``'t'`` or ``dt`` is unavailable in
            :mod:`brainstate.environ`.
        """
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

            curr_i = int(port_arr[i])
            if curr_i < 0 or curr_i >= calib.n_channels:
                raise ValueError(f'Unknown receptor_type {curr_i} for correlospinmatrix_detector.')

            stamp_step = int(stamp_arr[i])
            if not self._is_active(stamp_step, calib.t_min_steps, calib.t_max_steps):
                continue

            self._handle_event(curr_i=curr_i, stamp_step=stamp_step, multiplicity=multiplicity, calib=calib)

        return self.flush()

    def _handle_event(
        self,
        curr_i: int,
        stamp_step: int,
        multiplicity: int,
        calib: _Calibration,
    ):
        down_transition = False

        if multiplicity == 1:
            if curr_i == self._last_i and stamp_step == self._t_last_in_spike:
                self._curr_state[curr_i] = True
                self._last_change[curr_i] = int(stamp_step)
                self._tentative_down = False
            else:
                if self._tentative_down:
                    down_transition = True
                self._tentative_down = True
        elif multiplicity == 2:
            self._curr_state[curr_i] = True

            if self._tentative_down:
                down_transition = True

            self._curr_state[self._last_i] = False
            self._last_change[curr_i] = int(stamp_step)
            self._tentative_down = False

        if down_transition:
            self._process_down_transition(calib=calib)

        self._last_i = curr_i
        self._t_last_in_spike = int(stamp_step)

    def _process_down_transition(self, calib: _Calibration):
        i = int(self._last_i)
        t_i_on = int(self._last_change[i])
        t_i_off = int(self._t_last_in_spike)

        t_min_on = t_i_on
        for n in range(calib.n_channels):
            if bool(self._curr_state[n]) and int(self._last_change[n]) < t_min_on:
                t_min_on = int(self._last_change[n])

        while len(self._incoming) > 0:
            if (t_min_on - self._incoming[0].t_off) >= (calib.tau_edge_steps + calib.min_delay_steps):
                self._incoming.popleft()
            else:
                break

        pulse_i = _BinaryPulse(t_on=t_i_on, t_off=t_i_off, receptor_channel=i)

        insert_pos = len(self._incoming)
        for idx, pulse in enumerate(self._incoming):
            if pulse.t_off > pulse_i.t_off:
                insert_pos = idx
                break
        self._incoming.insert(insert_pos, pulse_i)

        t0 = calib.tau_max_steps // calib.delta_tau_steps
        dt = calib.delta_tau_steps

        for pulse_j in self._incoming:
            j = int(pulse_j.receptor_channel)
            t_j_on = int(pulse_j.t_on)
            t_j_off = int(pulse_j.t_off)

            delta_ij_min = max(t_j_on - t_i_off, -calib.tau_max_steps)
            delta_ij_max = min(t_j_off - t_i_on, calib.tau_max_steps)

            lag = min(t_i_off, t_j_off) - max(t_i_on, t_j_on)
            if lag > 0:
                self._count_covariance[i, j, t0] += int(lag)
                if i != j:
                    self._count_covariance[j, i, t0] += int(lag)

            delta_start = self._trunc_div(delta_ij_min, dt)
            for delta in range(delta_start, 0):
                lag = min(t_i_off, t_j_off - delta * dt) - max(t_i_on, t_j_on - delta * dt)
                if lag > 0:
                    self._count_covariance[i, j, t0 - delta] += int(lag)
                    self._count_covariance[j, i, t0 + delta] += int(lag)

            if i != j:
                delta_end = self._trunc_div(delta_ij_max, dt)
                for delta in range(1, delta_end + 1):
                    lag = min(t_i_off, t_j_off - delta * dt) - max(t_i_on, t_j_on - delta * dt)
                    if lag > 0:
                        self._count_covariance[i, j, t0 - delta] += int(lag)
                        self._count_covariance[j, i, t0 + delta] += int(lag)

        self._last_change[i] = int(t_i_off)

    @staticmethod
    def _trunc_div(a: int, b: int) -> int:
        return int(float(a) / float(b))

    def _ensure_calibrated_from_env_if_available(self):
        try:
            dt = brainstate.environ.get_dt()
        except KeyError:
            return
        self._ensure_calibrated(dt)

    def _ensure_calibrated(self, dt) -> _Calibration:
        # Fast path: skip full recomputation when dt and all params are unchanged.
        # _compute_calibration is expensive (many JAX ops); avoid calling it every step.
        if self._calib is not None and self._calib_param_ids is not None:
            dt_ms = self._fast_dt_ms(dt)
            if dt_ms == self._calib.dt_ms and self._param_ids() == self._calib_param_ids:
                return self._calib

        new_calib = self._compute_calibration(dt)

        if self._calib is None or self._calib.signature != new_calib.signature:
            self._calib = new_calib
            self._reset_state()

        self._calib_param_ids = self._param_ids()
        return self._calib

    def _param_ids(self) -> tuple:
        """Lightweight snapshot for detecting parameter changes via object identity."""
        return (
            id(self.delta_tau),
            id(self.tau_max),
            id(self.Tstart),
            id(self.Tstop),
            self.N_channels,
            id(self.start),
            id(self.stop),
            id(self.origin),
        )

    @staticmethod
    def _fast_dt_ms(dt) -> float:
        """Extract dt in ms cheaply, avoiding full JAX/numpy pipeline."""
        if isinstance(dt, u.Quantity):
            return float(u.get_mantissa(dt / u.ms))
        return float(dt)

    def _reset_state(self):
        self._incoming = deque()
        self._last_i = 0
        self._t_last_in_spike = -2 ** 62
        self._tentative_down = False

        ditype = brainstate.environ.ditype()

        if self._calib is None:
            self._curr_state = np.zeros((0,), dtype=np.bool_)
            self._last_change = np.zeros((0,), dtype=ditype)
            self._count_covariance = np.zeros((0, 0, 0), dtype=ditype)
            return

        n_channels = int(self._calib.n_channels)
        n_bins = int(self._calib.n_bins)

        self._curr_state = np.zeros((n_channels,), dtype=np.bool_)
        self._last_change = np.zeros((n_channels,), dtype=ditype)
        self._count_covariance = np.zeros((n_channels, n_channels, n_bins), dtype=ditype)

    def _compute_calibration(self, dt) -> _Calibration:
        dt_ms = self._to_ms_scalar(dt, name='dt')
        if dt_ms <= 0.0:
            raise ValueError('Simulation resolution dt must be positive.')

        start_steps = self._to_step_count(self.start, dt_ms, 'start')
        stop_value = math.inf if self.stop is None else self.stop
        stop_steps = self._to_step_count(stop_value, dt_ms, 'stop', allow_inf=True)
        if (not math.isinf(stop_steps)) and (stop_steps < start_steps):
            raise ValueError('stop >= start required.')

        origin_steps = self._to_step_count(self.origin, dt_ms, 'origin')
        t_min_steps = origin_steps + start_steps
        t_max_steps = math.inf if math.isinf(stop_steps) else origin_steps + stop_steps

        if self.delta_tau is None:
            delta_tau_ms = dt_ms
        else:
            delta_tau_ms = self._to_ms_scalar(self.delta_tau, name='delta_tau')

        if (not math.isfinite(delta_tau_ms)) or (delta_tau_ms <= 0.0):
            raise ValueError('/delta_tau must be positive and finite.')
        delta_tau_steps = self._to_step_count(delta_tau_ms, dt_ms, 'delta_tau')

        if self.tau_max is None:
            tau_max_ms = 10.0 * delta_tau_ms
        else:
            tau_max_ms = self._to_ms_scalar(self.tau_max, name='tau_max')

        if (not math.isfinite(tau_max_ms)) or (tau_max_ms < 0.0):
            raise ValueError('/tau_max must be finite and non-negative.')
        tau_max_steps = self._to_step_count(tau_max_ms, dt_ms, 'tau_max')
        if tau_max_steps % delta_tau_steps != 0:
            raise ValueError('tau_max must be a multiple of delta_tau.')

        tstart_ms = self._to_ms_scalar(self.Tstart, name='Tstart', allow_inf=True)
        tstop_value = math.inf if self.Tstop is None else self.Tstop
        tstop_ms = self._to_ms_scalar(tstop_value, name='Tstop', allow_inf=True)
        if tstart_ms < 0.0:
            raise ValueError('/Tstart must not be negative.')
        if tstop_ms < 0.0:
            raise ValueError('/Tstop must not be negative.')

        n_channels = self._to_int_scalar(self.N_channels, name='N_channels')
        if n_channels < 1:
            raise ValueError('/N_channels can only be larger than zero.')

        n_bins = int(1 + (2 * tau_max_steps) // delta_tau_steps)
        min_delay_steps = 1

        signature = (
            float(dt_ms),
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
            t_min_steps=int(t_min_steps),
            t_max_steps=float(t_max_steps),
            delta_tau_ms=float(delta_tau_ms),
            delta_tau_steps=int(delta_tau_steps),
            tau_max_ms=float(tau_max_ms),
            tau_max_steps=int(tau_max_steps),
            tau_edge_steps=int(tau_max_steps + delta_tau_steps),
            tstart_ms=float(tstart_ms),
            tstop_ms=float(tstop_ms),
            n_channels=int(n_channels),
            n_bins=int(n_bins),
            min_delay_steps=int(min_delay_steps),
            signature=signature,
        )

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
        ditype = brainstate.environ.ditype()
        arr = np.asarray(u.math.asarray(value), dtype=ditype).reshape(-1)
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
        default: float | None = None,
        size: int | None = None,
    ) -> np.ndarray:
        dftype = brainstate.environ.dftype()
        if x is None:
            if default is None:
                raise ValueError(f'{name} cannot be None.')
            arr = np.asarray([default], dtype=dftype)
        else:
            if isinstance(x, u.Quantity):
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
        default: int | None = None,
        size: int | None = None,
    ) -> np.ndarray:
        ditype = brainstate.environ.ditype()
        if x is None:
            if default is None:
                raise ValueError(f'{name} cannot be None.')
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
