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
import saiunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTDevice

__all__ = [
    'correlomatrix_detector',
]


@dataclass
class _Spike:
    timestep: int
    weight: float
    receptor_channel: int


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
    n_channels: int
    n_bins: int
    min_delay_steps: int
    signature: tuple


class correlomatrix_detector(NESTDevice):
    r"""NEST-compatible ``correlomatrix_detector`` device.

    **1. Overview**

    ``correlomatrix_detector`` receives spikes from ``N_channels`` receptor
    pools and accumulates binned auto/cross-covariance matrices for
    non-negative lags. It mirrors the semantics of the NEST
    ``correlomatrix_detector`` recording device, including dual-window
    filtering (activity window and counting window) and NEST-compatible
    bin-edge and matrix-ordering conventions.

    **2. Event Model and Covariance Tensor Equations**

    Let an accepted event be
    :math:`e=(c, t, m, w)` with receptor channel :math:`c`,
    integer simulation step :math:`t`, multiplicity :math:`m`, and
    connection weight :math:`w`. The queued event weight is
    :math:`\hat{w}=m\cdot w`.

    The detector stores all accepted events in one queue sorted by time.
    For each new accepted event :math:`i`:

    1. Insert :math:`i` into the queue (sorted by ``stamp_step``).
    2. Prune events older than the lag horizon
       :math:`\tau_{\mathrm{edge}}=\tau_{\max}+\Delta_\tau/2`,
       including minimum delay offset.
    3. If :math:`t_i \in [T_{\mathrm{start}}, T_{\mathrm{stop}}]`, update
       covariance bins against every remaining queued event :math:`j`.

    For pair :math:`(i,j)`, define :math:`d=|t_i-t_j|` (in steps).
    Channel ordering follows NEST matrix layout:

    - if :math:`t_i \ge t_j`, write into ``(c_i, c_j, b)``,
    - otherwise write into ``(c_j, c_i, b)``.

    The bin index :math:`b` (step domain) is computed as

    .. math::

       b =
       \begin{cases}
       -\left\lfloor \dfrac{\Delta_{\tau,\mathrm{steps}}/2 - d}
       {\Delta_{\tau,\mathrm{steps}}} \right\rfloor,
       & c_{\mathrm{row}} \le c_{\mathrm{col}} \\
       \left\lfloor \dfrac{\Delta_{\tau,\mathrm{steps}}/2 + d}
       {\Delta_{\tau,\mathrm{steps}}} \right\rfloor,
       & c_{\mathrm{row}} > c_{\mathrm{col}}
       \end{cases}

    and contributes

    .. math::

       \mathrm{cov}[c_{\mathrm{row}}, c_{\mathrm{col}}, b]
       \leftarrow
       \mathrm{cov}[c_{\mathrm{row}}, c_{\mathrm{col}}, b]
       + (m_i w_i)\hat{w}_j,

       \mathrm{count}[c_{\mathrm{row}}, c_{\mathrm{col}}, b]
       \leftarrow
       \mathrm{count}[c_{\mathrm{row}}, c_{\mathrm{col}}, b] + m_i.

    At zero lag, off-diagonal or non-identical-event pairs mirror-update the
    transposed entry, reproducing NEST's symmetric zero-lag handling.

    The number of bins is

    .. math::

       N_{\mathrm{bins}} = 1 + \frac{\tau_{\max,\mathrm{steps}}}
                                    {\Delta_{\tau,\mathrm{steps}}}.

    Output tensor shapes are
    ``(N_channels, N_channels, N_bins)`` where bin ``0`` corresponds to zero
    lag and bin ``k`` to lag :math:`k \cdot \Delta_\tau`.

    **3. Windowing, Assumptions, and Constraints**

    Two windows are applied:

    - *Activity window*:
      :math:`(\mathrm{origin}+\mathrm{start},\ \mathrm{origin}+\mathrm{stop}]`.
      Events outside this interval are discarded and never queued.
    - *Counting window*:
      :math:`[T_{\mathrm{start}},\ T_{\mathrm{stop}}]`. Only accepted events
      in this interval update ``n_events``, ``covariance``, and
      ``count_covariance``.

    Calibration constraints mirror NEST semantics in this implementation:

    - ``dt > 0`` and all finite time parameters are scalar-convertible.
    - ``start``, ``stop`` (when finite), ``origin``, ``delta_tau``, and
      ``tau_max`` must align to integer simulation steps.
    - ``delta_tau`` must be an odd multiple of ``dt``.
    - ``tau_max`` must be a non-negative multiple of ``delta_tau``.
    - ``N_channels >= 1``.

    **4. Computational Implications**

    Per accepted event, insertion is :math:`O(Q)` in queue length and
    correlation updates are :math:`O(Q)` over retained events, so total update
    work scales linearly with the active queue size. Memory usage is
    :math:`O(Q + N_{\mathrm{channels}}^2 \cdot N_{\mathrm{bins}})`.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape metadata consumed by :class:`brainstate.nn.Dynamics`.
        This detector stores internal tensors and does not emit batch-shaped
        arrays through ``update``. Default is ``1``.
    delta_tau : ArrayLike or None, optional
        Lag bin width :math:`\Delta_\tau` in milliseconds. Accepts a scalar
        float-like value or a ``saiunit`` quantity convertible to ms.
        Must be finite, strictly positive, aligned to ``dt``, and an odd
        multiple of ``dt``. ``None`` resolves to ``5 * dt``.
        Default is ``None``.
    tau_max : ArrayLike or None, optional
        One-sided lag horizon :math:`\tau_{\max}` in milliseconds. Accepts a
        scalar float-like value or a quantity convertible to ms. Must be
        finite, non-negative, aligned to ``dt``, and an exact multiple of
        ``delta_tau``. ``None`` resolves to ``10 * delta_tau``.
        Default is ``None``.
    Tstart : ArrayLike, optional
        Inclusive lower bound of the counting window in milliseconds.
        Must be scalar-convertible; ``saiunit`` quantities are converted
        to ms. Default is ``0.0 * u.ms``.
    Tstop : ArrayLike or None, optional
        Inclusive upper bound of the counting window in milliseconds.
        Must be scalar-convertible when provided. ``None`` means
        :math:`+\infty`. Default is ``None``.
    N_channels : int or ArrayLike, optional
        Number of receptor channels. Must resolve to a scalar integer
        ``>= 1``. Channel IDs accepted at runtime are
        ``0, 1, ..., N_channels - 1``. Default is ``1``.
    start : ArrayLike, optional
        Exclusive lower bound of the activity window relative to ``origin``
        in milliseconds. Must be scalar-convertible and aligned to ``dt``.
        Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Inclusive upper bound of the activity window relative to ``origin``
        in milliseconds. Must be scalar-convertible and aligned to ``dt``
        when finite. ``None`` means :math:`+\infty`. Default is ``None``.
    origin : ArrayLike, optional
        Activity-window origin shift in milliseconds. Must be
        scalar-convertible and aligned to ``dt``.
        Default is ``0.0 * u.ms``.
    name : str or None, optional
        Optional node name forwarded to :class:`brainstate.nn.Dynamics`.
        Default is ``None``.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 18 17 24 41

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``delta_tau``
         - ``None``
         - :math:`\Delta_\tau`
         - Lag-bin width; resolved as ``5 * dt`` when omitted.
       * - ``tau_max``
         - ``None``
         - :math:`\tau_{\max}`
         - One-sided lag horizon; resolved as ``10 * delta_tau`` when omitted.
       * - ``Tstart``
         - ``0.0 * u.ms``
         - :math:`T_{\mathrm{start}}`
         - Inclusive start of covariance/count update window.
       * - ``Tstop``
         - ``None``
         - :math:`T_{\mathrm{stop}}`
         - Inclusive end of covariance/count update window.
       * - ``N_channels``
         - ``1``
         - :math:`N_{\mathrm{channels}}`
         - Number of receptor channels and matrix axes.
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
         - Global shift added to ``start`` and ``stop`` boundaries.

    Raises
    ------
    ValueError
        If scalar parameters are invalid (non-scalar, non-finite where finite
        values are required, or misaligned to ``dt``), if consistency
        constraints are violated (e.g., ``delta_tau`` even in steps,
        ``tau_max`` not divisible by ``delta_tau``, ``stop < start``, or
        ``N_channels < 1``), or if runtime event arrays contain invalid
        values/sizes (negative multiplicities, non-finite ``weights``,
        unknown receptor channel, or mismatched vector lengths).
    KeyError
        If runtime environment keys such as simulation time ``'t'`` or
        resolution ``dt`` are unavailable when calibration or update is
        called.

    Notes
    -----
    - Unlike some NEST recording devices, ``n_events`` is read-only here,
      matching ``correlomatrix_detector`` semantics.
    - This implementation uses default NEST kernel minimum delay semantics in
      pruning (``min_delay = 1`` simulation step).
    - Optional ``multiplicities`` emulate NEST ``SpikeEvent`` multiplicity.
    - Runtime event arguments accepted by :meth:`update` are one-dimensional
      scalar-broadcastable arrays over the same event axis:
      ``spikes``, ``receptor_ports``/``receptor_types``, ``weights``,
      ``multiplicities``, and ``stamp_steps``.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> import numpy as np
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     det = brainpy.state.correlomatrix_detector(
       ...         N_channels=2,
       ...         delta_tau=0.5 * u.ms,
       ...         tau_max=2.0 * u.ms,
       ...     )
       ...     det.init_state()
       ...     _ = det.update(
       ...         spikes=np.array([1.0, 1.0]),
       ...         receptor_ports=np.array([0, 1]),
       ...         weights=np.array([1.0, 2.0]),
       ...         stamp_steps=np.array([11, 12]),
       ...     )
       ...     out = det.flush()
       >>> out["covariance"].shape
       (2, 2, 5)
       >>> out["count_covariance"].dtype
       dtype('int64')

    References
    ----------
    .. [1] NEST Simulator, ``correlomatrix_detector`` model.
           https://nest-simulator.readthedocs.io/en/stable/models/correlomatrix_detector.html
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
        name: str = None,
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
        self._incoming: deque[_Spike] = deque()
        ditype = brainstate.environ.ditype()
        self._n_events = np.zeros((0,), dtype=ditype)
        dftype = brainstate.environ.dftype()
        self._covariance = np.zeros((0, 0, 0), dtype=dftype)
        self._count_covariance = np.zeros((0, 0, 0), dtype=ditype)

        self._ensure_calibrated_from_env_if_available()

    @property
    def n_events(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        ditype = brainstate.environ.ditype()
        return np.asarray(self._n_events, dtype=ditype)

    @property
    def covariance(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        dftype = brainstate.environ.dftype()
        return np.asarray(self._covariance, dtype=dftype)

    @property
    def count_covariance(self) -> np.ndarray:
        self._ensure_calibrated_from_env_if_available()
        ditype = brainstate.environ.ditype()
        return np.asarray(self._count_covariance, dtype=ditype)

    def get(self, key: str = 'covariance'):
        r"""Retrieve a named scalar or array from the detector.

        Parameters
        ----------
        key : str, optional
            Name of the quantity to retrieve. Supported keys:

            - ``'covariance'`` — accumulated weighted covariance tensor,
              shape ``(N_channels, N_channels, N_bins)``, dtype float64.
            - ``'count_covariance'`` — unweighted spike-count covariance
              tensor, same shape, dtype int64.
            - ``'n_events'`` — per-channel accepted event counts,
              shape ``(N_channels,)``, dtype int64.
            - ``'delta_tau'`` — calibrated lag-bin width in ms, scalar
              float or ``None`` if not yet calibrated.
            - ``'tau_max'`` — calibrated one-sided lag horizon in ms,
              scalar float or ``None`` if not yet calibrated.
            - ``'Tstart'`` — counting-window lower bound in ms, scalar
              float (may be ``-inf``).
            - ``'Tstop'`` — counting-window upper bound in ms, scalar
              float (may be ``+inf``).
            - ``'N_channels'`` — number of receptor channels, scalar int.
            - ``'start'`` — activity-window lower bound (relative) in ms,
              scalar float.
            - ``'stop'`` — activity-window upper bound (relative) in ms,
              scalar float (may be ``+inf``).
            - ``'origin'`` — activity-window origin shift in ms, scalar
              float.

            Default is ``'covariance'``.

        Returns
        -------
        value : np.ndarray or float or int or None
            The requested quantity. Array types match the shapes and dtypes
            described above; scalar keys return Python numeric scalars.

        Raises
        ------
        KeyError
            If ``key`` is not one of the supported strings listed above.
        ValueError
            If the underlying parameter is non-scalar or non-convertible
            during retrieval.
        """
        if key == 'covariance':
            return self.covariance
        if key == 'count_covariance':
            return self.count_covariance
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
        if key == 'N_channels':
            return int(self._to_int_scalar(self.N_channels, name='N_channels'))
        if key == 'start':
            return self._to_ms_scalar(self.start, name='start')
        if key == 'stop':
            stop_val = math.inf if self.stop is None else self.stop
            return self._to_ms_scalar(stop_val, name='stop', allow_inf=True)
        if key == 'origin':
            return self._to_ms_scalar(self.origin, name='origin')
        raise KeyError(f'Unsupported key "{key}" for correlomatrix_detector.get().')

    def connect(self):
        return None

    def flush(self):
        r"""Return the current accumulated state as a dictionary.

        Snapshots all three accumulated arrays without modifying internal
        state. This is equivalent to calling ``get`` for each of the three
        primary output keys.

        Returns
        -------
        out : dict
            A dictionary with the following keys:

            - ``'covariance'`` : np.ndarray, shape
              ``(N_channels, N_channels, N_bins)``, dtype float64.
              Weighted auto/cross-covariance accumulated since the last
              ``init_state`` call.
            - ``'count_covariance'`` : np.ndarray, shape
              ``(N_channels, N_channels, N_bins)``, dtype int64.
              Unweighted spike-count covariance accumulated since the last
              ``init_state`` call.
            - ``'n_events'`` : np.ndarray, shape ``(N_channels,)``,
              dtype int64. Total number of accepted events per channel
              within the counting window since the last ``init_state``
              call.
        """
        return {
            'covariance': self.covariance,
            'count_covariance': self.count_covariance,
            'n_events': self.n_events,
        }

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Reset accumulated state and recalibrate from the environment.

        Clears the event queue, zeroes all accumulated arrays
        (``covariance``, ``count_covariance``, ``n_events``), and
        recomputes calibration from the current ``brainstate`` environment
        if ``dt`` is available. Must be called before the first
        :meth:`update` when running inside a ``brainstate.environ.context``.

        Parameters
        ----------
        batch_size : int or None, optional
            Ignored. Accepted for API compatibility with
            :class:`brainstate.nn.Dynamics`. Default is ``None``.
        **kwargs
            Ignored. Accepted for API compatibility.
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
        r"""Process one batch of incoming spike events and update accumulators.

        Reads the current simulation time ``'t'`` and resolution ``dt``
        from the ``brainstate`` environment, calibrates if necessary, then
        iterates over each event in the batch. Events outside the activity
        window are silently discarded. Events inside the counting window
        update ``covariance``, ``count_covariance``, and ``n_events``.

        Parameters
        ----------
        spikes : ArrayLike or None, optional
            1-D array of spike indicators over a batch of ``n_items``
            senders. A value ``> 0`` is treated as a spike. If the array
            contains integer-like floats, the rounded value is used as
            multiplicity when ``multiplicities`` is ``None``. ``None`` or
            empty array causes an immediate return of :meth:`flush` output.
        receptor_ports : ArrayLike or None, optional
            1-D integer array of receptor channel indices, shape
            ``(n_items,)`` or broadcastable scalar. Values must be in
            ``[0, N_channels - 1]``. Alias ``receptor_types`` is also
            accepted; if both are provided, ``receptor_ports`` takes
            precedence. Default (``None``) maps all events to channel ``0``.
        receptor_types : ArrayLike or None, optional
            Alias for ``receptor_ports``. Ignored when ``receptor_ports`` is
            also provided.
        weights : ArrayLike or None, optional
            1-D float array of connection weights, shape ``(n_items,)`` or
            broadcastable scalar. Must contain finite values. Default
            (``None``) uses weight ``1.0`` for all events.
        multiplicities : ArrayLike or None, optional
            1-D non-negative integer array of NEST ``SpikeEvent``
            multiplicities, shape ``(n_items,)`` or broadcastable scalar.
            When ``None``, multiplicities are inferred from ``spikes``:
            integer-like spike values are used directly; non-integer spike
            values are binarized to ``0`` or ``1``.
        stamp_steps : ArrayLike or None, optional
            1-D integer array of simulation step stamps for each event,
            shape ``(n_items,)`` or broadcastable scalar. When ``None``,
            all events are stamped at ``step_now + 1`` (next step), matching
            NEST's default delivery delay of one step.

        Returns
        -------
        out : dict
            Same mapping as :meth:`flush`:
            ``{'covariance': ..., 'count_covariance': ..., 'n_events': ...}``.

        Raises
        ------
        ValueError
            If any of the following occur:

            - ``multiplicities`` contains negative values.
            - ``weights`` contains non-finite values.
            - ``receptor_ports`` contains a channel index outside
              ``[0, N_channels - 1]``.
            - Any size-mismatched pair of ``(spikes, receptor_ports)``,
              ``(spikes, weights)``, ``(spikes, multiplicities)``, or
              ``(spikes, stamp_steps)`` where neither has size ``1``.
        KeyError
            If the ``brainstate`` environment does not expose ``'t'`` or
            ``dt`` at call time.
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
            ditype = brainstate.environ.ditype()
            stamp_arr = np.full((n_items,), step_now + 1, dtype=ditype)
        else:
            stamp_arr = self._to_int_array(stamp_steps, name='stamp_steps', size=n_items)

        for i in range(n_items):
            multiplicity = int(counts[i])
            if multiplicity <= 0:
                continue

            sender = int(port_arr[i])
            if sender < 0 or sender >= calib.n_channels:
                raise ValueError(f'Unknown receptor_type {sender} for correlomatrix_detector.')

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
        own_weight = float(multiplicity) * float(weight)

        spike_i = _Spike(
            timestep=stamp_step,
            weight=own_weight,
            receptor_channel=sender,
        )

        insert_pos = len(self._incoming)
        for idx, old_spike in enumerate(self._incoming):
            if old_spike.timestep > stamp_step:
                insert_pos = idx
                break
        self._incoming.insert(insert_pos, spike_i)

        while len(self._incoming) > 0:
            dt_steps = stamp_step - self._incoming[0].timestep
            if dt_steps >= calib.tau_edge_steps + calib.min_delay_steps:
                self._incoming.popleft()
            else:
                break

        stamp_ms = float(stamp_step) * calib.dt_ms
        if not self._is_in_count_window(stamp_ms, calib.tstart_ms, calib.tstop_ms):
            return

        self._n_events[sender] += 1
        n_bins = self._covariance.shape[2]

        for spike_j in self._incoming:
            other = spike_j.receptor_channel
            diff_steps = stamp_step - spike_j.timestep
            abs_diff_steps = abs(diff_steps)

            if stamp_step < spike_j.timestep:
                sender_ind = other
                other_ind = sender
            else:
                sender_ind = sender
                other_ind = other

            if sender_ind <= other_ind:
                bin_index = int(
                    -math.floor(
                        (0.5 * calib.delta_tau_steps - abs_diff_steps)
                        / calib.delta_tau_steps
                    )
                )
            else:
                bin_index = int(
                    math.floor(
                        (0.5 * calib.delta_tau_steps + abs_diff_steps)
                        / calib.delta_tau_steps
                    )
                )

            if bin_index >= n_bins:
                continue

            contribution = own_weight * spike_j.weight
            self._covariance[sender_ind, other_ind, bin_index] += contribution
            self._count_covariance[sender_ind, other_ind, bin_index] += multiplicity

            if bin_index == 0 and (diff_steps != 0 or other != sender):
                self._covariance[other_ind, sender_ind, bin_index] += contribution
                self._count_covariance[other_ind, sender_ind, bin_index] += multiplicity

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
        self._incoming = deque()

        if self._calib is None:
            ditype = brainstate.environ.ditype()
            self._n_events = np.zeros((0,), dtype=ditype)
            dftype = brainstate.environ.dftype()
            self._covariance = np.zeros((0, 0, 0), dtype=dftype)
            self._count_covariance = np.zeros((0, 0, 0), dtype=ditype)
            return

        n_channels = int(self._calib.n_channels)
        n_bins = int(self._calib.n_bins)

        self._n_events = np.zeros((n_channels,), dtype=ditype)
        self._covariance = np.zeros((n_channels, n_channels, n_bins), dtype=dftype)
        self._count_covariance = np.zeros((n_channels, n_channels, n_bins), dtype=ditype)

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
        if delta_tau_steps % 2 != 1:
            raise ValueError('/delta_tau must be odd multiple of resolution.')

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

        n_channels = self._to_int_scalar(self.N_channels, name='N_channels')
        if n_channels < 1:
            raise ValueError('/N_channels can only be larger than zero.')

        n_bins = int(1 + tau_max_steps // delta_tau_steps)
        min_delay_steps = 1

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
            int(n_channels),
            int(n_bins),
            int(min_delay_steps),
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
            n_channels=int(n_channels),
            n_bins=int(n_bins),
            min_delay_steps=int(min_delay_steps),
            signature=signature,
        )

    @staticmethod
    def _is_in_count_window(stamp_ms: float, tstart_ms: float, tstop_ms: float) -> bool:
        return (stamp_ms >= tstart_ms - 1e-12) and (stamp_ms <= tstop_ms + 1e-12)

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
