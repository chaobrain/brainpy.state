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

from brainpy_state._nest._base import NESTDevice
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'correlation_detector',
]


@dataclass
class _Spike:
    timestep: int
    weight: float


@dataclass
class _Calibration:
    dt_ms: float
    start_step: int
    stop_step: float
    origin_step: int
    t_min_steps: int
    t_max_steps: float
    delta_tau_ms: float
    delta_tau_steps: int
    tau_max_ms: float
    tau_max_steps: int
    tau_edge_steps: float
    tstart_ms: float
    tstop_ms: float
    n_bins: int
    signature: tuple


class correlation_detector(NESTDevice):
    r"""NEST-compatible ``correlation_detector`` device.

    **1. Overview**

    ``correlation_detector`` receives spikes from two receptor ports
    (``0`` and ``1``) and accumulates lag histograms in both weighted
    (float64) and unweighted (int64) forms, following NEST event ordering.
    It mirrors the semantics of the NEST ``correlation_detector`` recording
    device, including dual-window filtering (activity window and counting
    window), Kahan-compensated weighted histogram accumulation, and
    NEST-compatible bin-edge conventions.

    **2. Event Model and Histogram Equations**

    Let an accepted event be represented as
    :math:`e=(s, t, m, w)` where :math:`s\in\{0,1\}` is receptor port,
    :math:`t` is integer simulation step, :math:`m` is multiplicity, and
    :math:`w` is scalar connection weight. Each stored queue entry keeps
    :math:`\hat{w}=m\cdot w`.

    For each new event, the detector correlates it against all queued events
    of the opposite port that survive lag-window pruning. The bin index is

    .. math::

       b = \left\lfloor
       \frac{\tau_{\mathrm{edge}} + \sigma_s (t - t_j)}
            {\Delta_\tau}
       \right\rfloor,
       \qquad
       \sigma_s = 2s - 1,
       \qquad
       \tau_{\mathrm{edge}} = \tau_{\max} + \frac{\Delta_\tau}{2},

    with all times represented in integer steps for the index computation.
    :math:`\sigma_s` encodes causality direction: ``+1`` for port-1 events
    (event is the "post" spike) and ``-1`` for port-0 events (event is the
    "pre" spike), so positive lags correspond to port-1 spikes occurring
    after port-0 spikes.

    For each matched opposite event :math:`j`, the histograms are updated as

    .. math::

       H_b \leftarrow H_b + (m w)\,\hat{w}_j,
       \qquad
       C_b \leftarrow C_b + m,

    where :math:`H_b` is ``histogram`` and :math:`C_b` is
    ``count_histogram``. ``histogram`` uses Kahan summation per bin to
    reduce floating-point accumulation error; the compensation terms are
    exposed as ``histogram_correction``.

    The number of bins is

    .. math::

       N_{\mathrm{bins}} = 1 + 2 \cdot
       \left(\frac{\tau_{\max,\mathrm{steps}}}{\Delta_{\tau,\mathrm{steps}}}\right).

    Bin intervals are left-closed/right-open in the internal index rule, which
    matches NEST edge handling in ``correlation_detector``. The centre bin
    (index :math:`N_{\mathrm{bins}}//2`) corresponds to zero lag.

    **3. Windowing, Assumptions, and Constraints**

    Two windows are applied exactly as in NEST:

    - **Activity window**:
      :math:`(\mathrm{origin}+\mathrm{start},\ \mathrm{origin}+\mathrm{stop}]`
      in simulation time. Events outside are discarded and never buffered.
    - **Counting window**:
      :math:`[\mathrm{Tstart},\ \mathrm{Tstop}]`. Only events in this window
      increment ``n_events`` and update histograms. Events outside this window
      can still be buffered and can affect later counted events via
      cross-correlation with subsequently counted events.

    Grid-alignment constraints are strict: ``start``, ``stop`` (if finite),
    ``origin``, ``delta_tau``, and ``tau_max`` must map to integer multiples
    of simulation ``dt``. Additionally, ``tau_max`` must be an exact multiple
    of ``delta_tau``. Violations raise ``ValueError`` at calibration time.

    **4. Computational Implications**

    Per accepted event, work is linear in queue lengths:

    - :math:`O(Q_{\mathrm{other}})` for pruning and correlation against the
      opposite-port queue,
    - :math:`O(Q_{\mathrm{self}})` for sorted insertion into the sender queue.

    Memory usage is :math:`O(Q_0 + Q_1 + N_{\mathrm{bins}})`, where queue
    length depends on event rate and ``tau_max``. Calibration is triggered
    lazily on first access; subsequent calls reuse cached state unless ``dt``
    or window parameters change.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape metadata consumed by :class:`brainstate.nn.Dynamics`.
        This detector is event-driven and stores scalar histograms; ``in_size``
        is retained for API consistency and does not affect histogram shape.
        Default is ``1``.
    delta_tau : quantity (ms) or float or None, optional
        Bin width :math:`\Delta_\tau`. Unitful ``brainunit`` quantities are
        accepted and converted to ms; bare floats are interpreted as ms.
        Must be finite, strictly positive, and an integer multiple of
        simulation ``dt``. ``None`` auto-selects ``5 * dt``.
        Default is ``None``.
    tau_max : quantity (ms) or float or None, optional
        One-sided lag limit :math:`\tau_{\max}`. Unitful quantities accepted.
        Must be finite, non-negative, an integer multiple of ``dt``, and an
        exact integer multiple of ``delta_tau``. ``None`` auto-selects
        ``10 * delta_tau``. Default is ``None``.
    Tstart : quantity (ms) or float, optional
        Inclusive lower bound of the counting window in ms. Scalar-convertible;
        unitful values are converted to ms. Default is ``0.0 * u.ms``.
    Tstop : quantity (ms) or float or None, optional
        Inclusive upper bound of the counting window in ms. ``None`` means
        :math:`+\infty` (no upper bound). Scalar-convertible when provided.
        Default is ``None``.
    start : quantity (ms) or float, optional
        Exclusive lower bound of the activity window relative to ``origin``
        in ms. Must be scalar-convertible and aligned to simulation ``dt``.
        Default is ``0.0 * u.ms``.
    stop : quantity (ms) or float or None, optional
        Inclusive upper bound of the activity window relative to ``origin``
        in ms. Must be scalar-convertible and aligned to ``dt`` when finite.
        ``None`` means :math:`+\infty`. Default is ``None``.
    origin : quantity (ms) or float, optional
        Global time origin shift in ms for activity-window evaluation.
        The effective activity window becomes
        ``(origin + start, origin + stop]``. Must be scalar-convertible
        and aligned to ``dt``. Default is ``0.0 * u.ms``.
    name : str or None, optional
        Optional node name forwarded to :class:`brainstate.nn.Dynamics`.
        If ``None``, a name is auto-generated. Default is ``None``.

    Parameter Mapping
    -----------------

    .. list-table::
       :header-rows: 1
       :widths: 18 17 24 41

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``delta_tau``
         - ``None`` → ``5 * dt``
         - :math:`\Delta_\tau`
         - Lag-histogram bin width; auto-resolved when omitted.
       * - ``tau_max``
         - ``None`` → ``10 * delta_tau``
         - :math:`\tau_{\max}`
         - One-sided correlation horizon; auto-resolved when omitted.
       * - ``Tstart``
         - ``0.0 ms``
         - :math:`T_{\mathrm{start}}`
         - Inclusive start of histogram and event-count update window.
       * - ``Tstop``
         - ``None`` (:math:`+\infty`)
         - :math:`T_{\mathrm{stop}}`
         - Inclusive end of histogram and event-count update window.
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
         - Global offset applied to ``start`` and ``stop`` boundaries.

    Raises
    ------
    ValueError
        If time parameters are non-scalar, non-finite where finite values are
        required, misaligned to simulation resolution, or violate consistency
        constraints (e.g. ``tau_max % delta_tau != 0`` or ``stop < start``).
        Also raised for invalid runtime event arguments (unknown receptor port,
        negative multiplicity, non-finite ``weights``, or size mismatches).
    KeyError
        If runtime environment keys such as ``'t'`` or simulation ``dt`` are
        unavailable when calibration or update is attempted.
    RuntimeError
        If an internal lag-bin index falls outside histogram range; this
        indicates inconsistency between calibration and event processing.

    Notes
    -----
    - ``n_events`` can only be assigned ``[0, 0]``, which resets all detector
      state and clears histograms, matching NEST's reset semantics.
    - Runtime input events are provided through :meth:`update`:
      ``spikes``, ``receptor_ports``, ``weights``, ``multiplicities``, and
      ``stamp_steps`` are each scalar-broadcastable to a common 1-D event axis.
    - Receptor ports are restricted to integer values ``0`` and ``1``.
    - Connection delays are ignored by design; only event time stamps are used
      for lag computation.
    - Calibration is cached and reused across steps; it is automatically
      invalidated if ``dt`` or any window parameter changes between calls.

    Examples
    --------
    Basic correlation of simultaneous spikes on opposite ports:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     det = brainpy.state.correlation_detector(
       ...         delta_tau=0.5 * u.ms,
       ...         tau_max=5.0 * u.ms,
       ...     )
       ...     with brainstate.environ.context(t=1.0 * u.ms):
       ...         out = det.update(
       ...             spikes=np.array([1.0, 1.0]),
       ...             receptor_ports=np.array([0, 1]),
       ...             weights=np.array([1.0, 2.0]),
       ...             multiplicities=np.array([1, 1]),
       ...             stamp_steps=np.array([11, 11]),
       ...         )
       ...     _ = out['histogram'].shape  # (21,) for tau_max=5ms, delta_tau=0.5ms

    Default parameters with no input events and explicit state reset:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     det = brainpy.state.correlation_detector()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = det.update()  # no input events; returns current state
       ...     det.n_events = [0, 0]  # explicit reset, NEST-compatible

    References
    ----------
    .. [1] NEST Simulator, ``correlation_detector`` model.
           https://nest-simulator.readthedocs.io/en/stable/models/correlation_detector.html
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        delta_tau: ArrayLike = None,
        tau_max: ArrayLike = None,
        Tstart: ArrayLike = 0.0 * u.ms,
        Tstop: ArrayLike = None,
        start: ArrayLike = 0.0 * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0.0 * u.ms,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.delta_tau = delta_tau
        self.tau_max = tau_max
        self.Tstart = Tstart
        self.Tstop = Tstop

        self.start = start
        self.stop = stop
        self.origin = origin

        self._calib: _Calibration | None = None
        self._incoming = [deque(), deque()]
        self._n_events = np.zeros((2,), dtype=np.int64)
        self._histogram = np.zeros((0,), dtype=np.float64)
        self._histogram_correction = np.zeros((0,), dtype=np.float64)
        self._count_histogram = np.zeros((0,), dtype=np.int64)

        self._ensure_calibrated_from_env_if_available()

    @property
    def n_events(self) -> np.ndarray:
        return np.asarray(self._n_events, dtype=np.int64)

    @n_events.setter
    def n_events(self, value):
        arr = np.asarray(u.math.asarray(value), dtype=np.int64).reshape(-1)
        if arr.size != 2 or arr[0] != 0 or arr[1] != 0:
            raise ValueError('/n_events can only be set to [0 0].')
        self._reset_state()

    @property
    def histogram(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        return np.asarray(self._histogram, dtype=np.float64)

    @property
    def histogram_correction(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        return np.asarray(self._histogram_correction, dtype=np.float64)

    @property
    def count_histogram(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        return np.asarray(self._count_histogram, dtype=np.int64)

    def get(self, key: str = 'histogram'):
        r"""Return one detector state variable or calibrated scalar parameter.

        Parameters
        ----------
        key : str, optional
            Query key. Supported values are ``'histogram'``,
            ``'histogram_correction'``, ``'count_histogram'``, ``'n_events'``,
            ``'delta_tau'``, ``'tau_max'``, ``'Tstart'``, ``'Tstop'``,
            ``'start'``, ``'stop'``, and ``'origin'``. Default is
            ``'histogram'``.

        Returns
        -------
        out : dict
            Requested value. Histogram outputs are NumPy arrays with shapes
            ``(N_bins,)`` (float64/int64). ``n_events`` has shape ``(2,)``.
            Time scalar outputs are returned in milliseconds as Python
            ``float``; infinite bounds are returned as ``math.inf``.

        Raises
        ------
        KeyError
            If ``key`` is unsupported.
        ValueError
            If scalar conversion of configured time parameters fails.
        """
        if key == 'histogram':
            return self.histogram
        if key == 'histogram_correction':
            return self.histogram_correction
        if key == 'count_histogram':
            return self.count_histogram
        if key == 'n_events':
            return self.n_events
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
        if key == 'start':
            return self._to_ms_scalar(self.start, name='start')
        if key == 'stop':
            stop_val = math.inf if self.stop is None else self.stop
            return self._to_ms_scalar(stop_val, name='stop', allow_inf=True)
        if key == 'origin':
            return self._to_ms_scalar(self.origin, name='origin')
        raise KeyError(f'Unsupported key "{key}" for correlation_detector.get().')

    def connect(self):
        r"""Compatibility no-op for NEST-like device interface.

        """
        return None

    def flush(self):
        r"""Return current detector outputs without consuming internal state.

        """
        return {
            'histogram': self.histogram,
            'histogram_correction': self.histogram_correction,
            'count_histogram': self.count_histogram,
            'n_events': self.n_events,
        }

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Reset detector buffers and histogram state for current calibration.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused placeholder for :class:`brainstate.nn.Dynamics`
            compatibility.
        **kwargs
            Unused compatibility arguments.

        """
        del batch_size, kwargs
        self._ensure_calibrated_from_env_if_available()
        self._reset_state()

    def update(
        self,
        spikes: ArrayLike = None,
        receptor_ports: ArrayLike = None,
        receptor_types: ArrayLike = None,
        weights: ArrayLike = None,
        multiplicities: ArrayLike = None,
        stamp_steps: ArrayLike = None,
    ):
        r"""Process one simulation step of incoming events and return outputs.

        Parameters
        ----------
        spikes : ArrayLike or None, optional
            Event-presence/multiplicity proxy with shape ``(N,)`` after
            flattening (scalars are broadcast). If ``None``, no events are
            processed and current state is returned. When ``multiplicities`` is
            ``None``, integer-like ``spikes`` values are rounded and clipped to
            non-negative multiplicities; otherwise non-integer values are
            interpreted as binary ``spike > 0`` flags.
        receptor_ports : ArrayLike or None, optional
            Receptor port indices with shape ``(N,)`` (or scalar broadcast).
            Valid values are ``0`` and ``1`` only. If ``None``, defaults to
            ``0`` for all events unless ``receptor_types`` is provided.
        receptor_types : ArrayLike or None, optional
            Alias for ``receptor_ports`` kept for NEST API compatibility. Used
            only when ``receptor_ports`` is ``None``.
        weights : ArrayLike or None, optional
            Per-event connection weights with shape ``(N,)`` (or scalar
            broadcast). Must be finite. Default is ``1.0`` when omitted.
        multiplicities : ArrayLike or None, optional
            Explicit non-negative integer multiplicities with shape ``(N,)``
            (or scalar broadcast). Effective multiplicity is forced to zero
            where corresponding ``spikes <= 0``.
        stamp_steps : ArrayLike or None, optional
            Integer event time stamps in simulation steps with shape ``(N,)``
            (or scalar broadcast). If ``None``, all events are stamped at
            ``current_step + 1``.

        Returns
        -------
        out : jax.Array
            Same dictionary as :meth:`flush`, containing current
            ``histogram``, ``histogram_correction``, ``count_histogram``, and
            ``n_events`` after processing this call.

        Raises
        ------
        KeyError
            If required environment values (``'t'`` or ``dt``) are missing.
        ValueError
            If argument sizes are inconsistent, receptor ports are outside
            ``{0, 1}``, multiplicities are negative, times are not grid-aligned,
            or calibration constraints are violated.
        RuntimeError
            If a computed bin index is outside histogram bounds.
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
        weight_arr = self._to_float_array(weights, name='weights', default=1.0, size=n_items)

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

            sender = int(port_arr[i])
            if sender < 0 or sender > 1:
                raise ValueError(f'Unknown receptor_type {sender} for correlation_detector.')

            stamp_step = int(stamp_arr[i])
            if not self._is_active(stamp_step, calib.t_min_steps, calib.t_max_steps):
                continue

            self._handle_event(
                sender=sender,
                stamp_step=stamp_step,
                weight=float(weight_arr[i]),
                multiplicity=multiplicity,
                calib=calib,
            )

        return self.flush()

    def _handle_event(
        self,
        sender: int,
        stamp_step: int,
        weight: float,
        multiplicity: int,
        calib: _Calibration,
    ):
        other = 1 - sender
        other_spikes = self._incoming[other]

        while len(other_spikes) > 0:
            dt_steps = stamp_step - other_spikes[0].timestep
            if dt_steps - 0.5 * other >= calib.tau_edge_steps:
                other_spikes.popleft()
            else:
                break

        stamp_ms = float(stamp_step) * calib.dt_ms
        if self._is_in_count_window(stamp_ms, calib.tstart_ms, calib.tstop_ms):
            self._n_events[sender] += 1

            sign = 2 * sender - 1
            own_weight = float(multiplicity) * float(weight)

            for spike_j in other_spikes:
                bin_index = int(
                    math.floor(
                        (calib.tau_edge_steps + sign * (stamp_step - spike_j.timestep))
                        / calib.delta_tau_steps
                    )
                )
                if bin_index < 0 or bin_index >= self._histogram.size:
                    raise RuntimeError('Internal bin index out of range in correlation_detector.')

                y = own_weight * spike_j.weight - self._histogram_correction[bin_index]
                t = self._histogram[bin_index] + y
                self._histogram_correction[bin_index] = (t - self._histogram[bin_index]) - y
                self._histogram[bin_index] = t
                self._count_histogram[bin_index] += multiplicity

        spike_entry = _Spike(timestep=stamp_step, weight=float(multiplicity) * float(weight))
        queue = self._incoming[sender]

        insert_pos = len(queue)
        for idx, old_spike in enumerate(queue):
            if old_spike.timestep > stamp_step:
                insert_pos = idx
                break
        queue.insert(insert_pos, spike_entry)

    def _ensure_calibrated_from_env_if_available(self):
        try:
            dt = brainstate.environ.get_dt()
        except KeyError:
            return
        self._ensure_calibrated(dt)

    def _ensure_calibrated(self, dt) -> _Calibration:
        new_calib = self._compute_calibration(dt)

        if self._calib is None or self._calib.signature != new_calib.signature:
            self._calib = new_calib
            self._reset_state()

        return self._calib

    def _reset_state(self):
        self._n_events = np.zeros((2,), dtype=np.int64)
        self._incoming = [deque(), deque()]

        if self._calib is None:
            self._histogram = np.zeros((0,), dtype=np.float64)
            self._histogram_correction = np.zeros((0,), dtype=np.float64)
            self._count_histogram = np.zeros((0,), dtype=np.int64)
            return

        n_bins = int(self._calib.n_bins)
        self._histogram = np.zeros((n_bins,), dtype=np.float64)
        self._histogram_correction = np.zeros((n_bins,), dtype=np.float64)
        self._count_histogram = np.zeros((n_bins,), dtype=np.int64)

    def _compute_calibration(self, dt) -> _Calibration:
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

        if self.delta_tau is None:
            delta_tau_ms = 5.0 * dt_ms
        else:
            delta_tau_ms = self._to_ms_scalar(self.delta_tau, name='delta_tau')
        if not math.isfinite(delta_tau_ms) or delta_tau_ms <= 0.0:
            raise ValueError('delta_tau must be positive and finite.')
        delta_tau_steps = self._to_step_count(delta_tau_ms, dt_ms, 'delta_tau')

        if self.tau_max is None:
            tau_max_ms = 10.0 * delta_tau_ms
        else:
            tau_max_ms = self._to_ms_scalar(self.tau_max, name='tau_max')
        if not math.isfinite(tau_max_ms) or tau_max_ms < 0.0:
            raise ValueError('tau_max must be finite and non-negative.')
        tau_max_steps = self._to_step_count(tau_max_ms, dt_ms, 'tau_max')

        if tau_max_steps % delta_tau_steps != 0:
            raise ValueError('tau_max must be a multiple of delta_tau.')

        tstart_ms = self._to_ms_scalar(self.Tstart, name='Tstart', allow_inf=True)
        tstop_value = math.inf if self.Tstop is None else self.Tstop
        tstop_ms = self._to_ms_scalar(tstop_value, name='Tstop', allow_inf=True)

        n_bins = int(1 + 2 * (tau_max_steps // delta_tau_steps))

        signature = (
            float(dt_ms),
            int(start_steps),
            float(stop_steps),
            int(origin_steps),
            int(t_min_steps),
            float(t_max_steps),
            float(delta_tau_ms),
            int(delta_tau_steps),
            float(tau_max_ms),
            int(tau_max_steps),
            float(tstart_ms),
            float(tstop_ms),
            int(n_bins),
        )

        return _Calibration(
            dt_ms=float(dt_ms),
            start_step=int(start_steps),
            stop_step=float(stop_steps),
            origin_step=int(origin_steps),
            t_min_steps=int(t_min_steps),
            t_max_steps=float(t_max_steps),
            delta_tau_ms=float(delta_tau_ms),
            delta_tau_steps=int(delta_tau_steps),
            tau_max_ms=float(tau_max_ms),
            tau_max_steps=int(tau_max_steps),
            tau_edge_steps=float(tau_max_steps) + 0.5 * float(delta_tau_steps),
            tstart_ms=float(tstart_ms),
            tstop_ms=float(tstop_ms),
            n_bins=int(n_bins),
            signature=signature,
        )

    @staticmethod
    def _is_in_count_window(stamp_ms: float, tstart_ms: float, tstop_ms: float) -> bool:
        return (stamp_ms >= tstart_ms - 1e-12) and (stamp_ms <= tstop_ms + 1e-12)

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
