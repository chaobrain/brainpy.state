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


import brainstate
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTDevice

__all__ = [
    'poisson_generator_ps',
]

_UNSET = object()


class poisson_generator_ps(NESTDevice):
    r"""Precise-time Poisson spike generator with dead time (NEST-compatible).

    Description
    -----------
    ``poisson_generator_ps`` re-implements NEST's precise-time stimulation
    device ``poisson_generator_ps`` and emits off-grid spike times generated
    by an absolute-refractory renewal process.

    **1. Renewal process with dead time**

    Let ``r`` be the configured mean rate (spikes/s) and
    :math:`t_{\mathrm{dead}}` be dead time (ms). Inter-spike intervals (ISIs)
    are sampled as

    .. math::

       \Delta t = t_{\mathrm{dead}} + \xi \alpha, \qquad
       \xi \sim \mathrm{Exp}(1), \qquad
       \alpha = \frac{1000}{r} - t_{\mathrm{dead}}.

    The mean ISI is

    .. math::

       \mathbb{E}[\Delta t] = t_{\mathrm{dead}} + \alpha = \frac{1000}{r},

    so the stationary mean rate is preserved exactly when
    :math:`\alpha \ge 0`, which is equivalent to
    ``dead_time <= 1000 / rate`` for ``rate > 0``.

    When a stream is (re)initialized at an active step, the first offset from
    the local active left boundary is sampled from the stationary
    backward-recurrence distribution used by NEST:

    - uniform branch on ``[0, dead_time)`` with probability
      ``dead_time * rate / 1000``,
    - shifted exponential branch on ``[dead_time, +inf)`` otherwise.

    This avoids transient rate bias immediately after activation.

    **2. Activity window and update ordering**

    For one simulation step with left edge :math:`t` and width :math:`dt`
    (both in ms), define

    .. math::

       t_{\min} = \max(t,\, origin + start), \qquad
       t_{\max} = \min(t + dt,\, origin + stop).

    If ``t_min < t_max`` and ``rate > 0``, spikes are emitted using
    ``t_min < spike_time <= t_max`` (left-open, right-closed on the active
    slice). This matches NEST ``poisson_generator_ps.cpp`` ordering semantics.

    **3. Assumptions, constraints, and computational implications**

    - ``rate``, ``dead_time``, ``start``, ``stop``, and ``origin`` are scalar
      public parameters converted to ``float64`` (Hz/ms).
    - ``start`` and ``origin`` must be finite; ``stop`` must be finite or
      ``+inf``; ``stop >= start``.
    - Per-target streams are independent ``numpy.random.Generator`` instances
      spawned from ``rng_seed`` via ``numpy.random.SeedSequence``.
    - :meth:`update` cost is :math:`O(N + S)` per step where
      :math:`N=\prod \mathrm{varshape}` and :math:`S` is the number of spikes
      emitted in the step across all streams.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification for :class:`brainstate.nn.Dynamics`.
        The generated multiplicity tensor from :meth:`update` has shape
        ``self.varshape`` derived from ``in_size``. Each element corresponds
        to one independent output train. Default is ``1``.
    rate : ArrayLike, optional
        Scalar mean firing rate in spikes/s (Hz). Accepts any ``ArrayLike``
        with exactly one element, optionally a :class:`brainunit.Quantity`
        convertible to ``u.Hz``. Must satisfy ``rate >= 0``.
        Default is ``0.0 * u.Hz``.
    dead_time : ArrayLike, optional
        Scalar absolute dead time in ms. Accepts one-element ``ArrayLike`` or
        quantity convertible to ``u.ms``. Must satisfy ``dead_time >= 0`` and
        ``dead_time <= 1000 / rate`` when ``rate > 0``.
        Default is ``0.0 * u.ms``.
    start : ArrayLike, optional
        Scalar relative activation time in ms. Effective lower activity bound
        is ``origin + start``. Must be finite after conversion.
        Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative deactivation time in ms. Effective upper activity bound
        is ``origin + stop``. ``None`` maps to ``+inf``. Must satisfy
        ``stop >= start`` after conversion. Default is ``None``.
    origin : ArrayLike, optional
        Scalar global time offset in ms added to ``start`` and ``stop``.
        Must be finite after conversion. Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed passed to ``numpy.random.SeedSequence`` for per-target RNG stream
        construction in :meth:`init_state`. Default is ``0``.
    name : str or None, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 20 16 20 44

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``rate``
         - ``0.0 * u.Hz``
         - :math:`r`
         - Target stationary rate in spikes/s.
       * - ``dead_time``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{dead}}`
         - Absolute refractory interval added to every ISI.
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative activity-window lower bound (inclusive in slicing step).
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative activity-window upper bound; ``None`` maps to ``+\infty``.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global offset added to ``start`` and ``stop``.
       * - ``in_size``
         - ``1``
         - -
         - Defines ``self.varshape`` and number of independent trains.
       * - ``rng_seed``
         - ``0``
         - -
         - Root seed used to spawn independent per-target RNG streams.

    Raises
    ------
    ValueError
        If ``rate < 0``; ``dead_time < 0``; ``stop < start``; ``start`` or
        ``origin`` is non-finite; ``stop`` is neither finite nor ``+inf``; or
        ``dead_time > 1000 / rate`` when ``rate > 0``.
    TypeError
        If supplied ``ArrayLike`` values cannot be converted to scalar Hz/ms.
    KeyError
        At update time, if required simulation context entries such as ``dt``
        are unavailable through ``brainstate.environ``.

    Notes
    -----
    - Unlike the grid-constrained :class:`poisson_generator`, this model tracks
      and emits off-grid spike times with sub-step precision.
    - ``last_spike_time`` stores the latest emitted precise time per output.
    - ``last_spike_offset`` stores ``(t + dt) - last_spike_time`` at the step
      where that spike was emitted, using ms units.
    - Calling :meth:`set` with ``rate=...`` resets ``next_spike_time`` state,
      mirroring NEST behavior.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.poisson_generator_ps(
       ...         in_size=(2,),
       ...         rate=800.0 * u.Hz,
       ...         dead_time=0.5 * u.ms,
       ...         start=5.0 * u.ms,
       ...         stop=30.0 * u.ms,
       ...         rng_seed=7,
       ...     )
       ...     with brainstate.environ.context(t=10.0 * u.ms):
       ...         counts, times = gen.update(return_precise_times=True)
       ...     _ = counts.shape, len(times)

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> gen = brainpy.state.poisson_generator_ps(rate=500.0 * u.Hz)
       >>> gen.set(dead_time=0.8 * u.ms, stop=None)
       >>> params = gen.get()
       >>> _ = params["rate"], params["dead_time"], params["stop"]

    See Also
    --------
    poisson_generator : Grid-constrained homogeneous Poisson generator.
    inhomogeneous_poisson_generator : Time-varying Poisson generator.

    References
    ----------
    .. [1] NEST source: ``models/poisson_generator_ps.h`` and
           ``models/poisson_generator_ps.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/poisson_generator_ps.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        dead_time: ArrayLike = 0. * u.ms,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.dead_time = self._to_scalar_time_ms(dead_time)
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate=self.rate,
            dead_time=self.dead_time,
            start=self.start,
            stop=self.stop,
            origin=self.origin,
        )

        self._num_targets = int(np.prod(self.varshape))
        self._last_step_spike_times_ms = tuple(
            np.asarray([], dtype=np.float64) for _ in range(self._num_targets)
        )

    @staticmethod
    def _to_scalar_time_ms(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.ms), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=np.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('Time parameters must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_rate_hz(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.Hz), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=np.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('rate must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _validate_parameters(
        *,
        rate: float,
        dead_time: float,
        start: float,
        stop: float,
        origin: float,
    ):
        if rate < 0.0:
            raise ValueError('The rate cannot be negative.')
        if dead_time < 0.0:
            raise ValueError('The dead time cannot be negative.')
        if stop < start:
            raise ValueError('stop >= start required.')
        if not np.isfinite(start):
            raise ValueError('start must be finite.')
        if not np.isfinite(origin):
            raise ValueError('origin must be finite.')
        if (not np.isinf(stop)) and (not np.isfinite(stop)):
            raise ValueError('stop must be finite or infinity.')
        if rate > 0.0 and (1000.0 / rate < dead_time):
            raise ValueError('The inverse rate cannot be smaller than the dead time.')

    def _dt_ms(self) -> float:
        dt = brainstate.environ.get_dt()
        return self._to_scalar_time_ms(dt)

    def _current_time_ms(self) -> float:
        t = brainstate.environ.get('t', default=0. * u.ms)
        if t is None:
            return 0.0
        return self._to_scalar_time_ms(t)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize precise-spike state buffers and per-target RNG streams.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused. Included for framework API compatibility with
            :class:`brainstate.nn.Dynamics`. Default is ``None``.
        **kwargs : Any
            Unused keyword arguments accepted for API compatibility.


        Notes
        -----
        Calling :meth:`init_state` resets all state buffers and re-seeds all
        RNG streams. Repeated calls therefore restart the stochastic sequence
        from the beginning. :meth:`update` calls this method lazily on the
        first step if :meth:`init_state` has not been invoked explicitly.

        See Also
        --------
        poisson_generator_ps.update : Consumes state buffers populated here.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate
           >>> import brainunit as u
           >>> from brainpy.state import poisson_generator_ps
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     gen = poisson_generator_ps(in_size=4, rate=800.0 * u.Hz, rng_seed=7)
           ...     gen.init_state()
        """
        del batch_size, kwargs
        self.next_spike_time = brainstate.ShortTermState(
            np.full(self._num_targets, -np.inf, dtype=np.float64)
        )
        self.last_spike_time = brainstate.ShortTermState(
            np.full(self.varshape, -np.inf, dtype=np.float64)
        )
        self.last_spike_offset = brainstate.ShortTermState(
            np.zeros(self.varshape, dtype=np.float64)
        )

        # Independent random streams per target keep train generation stable
        # across different simulation resolutions.
        seed_seq = np.random.SeedSequence(self.rng_seed)
        self._rngs = tuple(
            np.random.default_rng(s) for s in seed_seq.spawn(self._num_targets)
        )
        self._last_step_spike_times_ms = tuple(
            np.asarray([], dtype=np.float64) for _ in range(self._num_targets)
        )

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        dead_time: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        r"""Update public parameters and refresh generator state when required.

        Only keyword arguments that are explicitly passed are modified; omitted
        arguments retain their current values.

        Parameters
        ----------
        rate : ArrayLike or object, optional
            New scalar mean firing rate in spikes/s (Hz). Accepts any
            ``ArrayLike`` with exactly one element, or a
            :class:`brainunit.Quantity` convertible to ``u.Hz``. Must satisfy
            ``rate >= 0`` after conversion. Setting this parameter resets
            ``next_spike_time`` state to ``-inf`` for all targets, matching
            NEST behavior. Omit to keep the current value.
        dead_time : ArrayLike or object, optional
            New scalar absolute dead time in ms. Accepts one-element
            ``ArrayLike`` or quantity convertible to ``u.ms``. Must satisfy
            ``dead_time >= 0`` and ``dead_time <= 1000 / rate`` when
            ``rate > 0`` after conversion. Omit to keep the current value.
        start : ArrayLike or object, optional
            New scalar relative activation time in ms. Effective lower activity
            bound is ``origin + start``. Must be finite after conversion.
            Omit to keep the current value.
        stop : ArrayLike or None or object, optional
            New scalar relative deactivation time in ms. ``None`` maps to
            ``+inf``. Must satisfy ``stop >= start`` after conversion. Omit to
            keep the current value.
        origin : ArrayLike or object, optional
            New scalar global time offset in ms added to ``start`` and
            ``stop``. Must be finite after conversion. Omit to keep the
            current value.


        Raises
        ------
        ValueError
            If updated parameters violate model constraints: negative
            ``rate``/``dead_time``, ``stop < start``, non-finite
            ``origin``/``start``, or ``dead_time > 1000 / rate`` for
            ``rate > 0``.
        TypeError
            If any supplied parameter cannot be converted to scalar Hz/ms.

        See Also
        --------
        poisson_generator_ps.get : Read-back current parameter values.

        Examples
        --------
        .. code-block:: python

           >>> import brainpy
           >>> import brainunit as u
           >>> gen = brainpy.state.poisson_generator_ps(rate=500.0 * u.Hz)
           >>> gen.set(rate=1000.0 * u.Hz, dead_time=0.5 * u.ms)
           >>> params = gen.get()
           >>> _ = params['rate'], params['dead_time']
        """
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_dead_time = (
            self.dead_time if dead_time is _UNSET else self._to_scalar_time_ms(dead_time)
        )
        new_start = self.start if start is _UNSET else self._to_scalar_time_ms(start)
        if stop is _UNSET:
            new_stop = self.stop
        elif stop is None:
            new_stop = np.inf
        else:
            new_stop = self._to_scalar_time_ms(stop)
        new_origin = self.origin if origin is _UNSET else self._to_scalar_time_ms(origin)

        self._validate_parameters(
            rate=new_rate,
            dead_time=new_dead_time,
            start=new_start,
            stop=new_stop,
            origin=new_origin,
        )

        self.rate = new_rate
        self.dead_time = new_dead_time
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        # NEST resets next spike states when "rate" is set.
        if (rate is not _UNSET) and hasattr(self, 'next_spike_time'):
            self.next_spike_time.value = np.full(
                self._num_targets, -np.inf, dtype=np.float64
            )

        # Match NEST pre-run behavior when start/origin are shifted forward:
        # if previous next-spike times lie before the new activation start,
        # reinitialize all target streams.
        if (
            (start is not _UNSET or origin is not _UNSET)
            and hasattr(self, 'next_spike_time')
        ):
            vals = np.asarray(self.next_spike_time.value, dtype=np.float64)
            finite = np.isfinite(vals)
            if finite.any() and float(np.min(vals[finite])) < (self.origin + self.start):
                self.next_spike_time.value = np.full(
                    self._num_targets, -np.inf, dtype=np.float64
                )

    def get(self) -> dict:
        r"""Return current public parameters as plain Python scalars.

        Returns
        -------
        params : dict
            Dictionary with five ``float`` entries:

            - ``'rate'`` -- mean firing rate in spikes/s (Hz).
            - ``'dead_time'`` -- absolute refractory dead time in ms.
            - ``'start'`` -- relative exclusive-lower activity bound in ms.
            - ``'stop'`` -- relative inclusive-upper activity bound in ms;
              ``inf`` when deactivation is disabled (``stop=None`` was passed).
            - ``'origin'`` -- global time offset in ms.

        Notes
        -----
        Returned values are plain Python ``float`` scalars (``float64``
        precision). They mirror the internal scalar attributes set in
        :meth:`__init__` or updated by :meth:`set` and are not bound to any
        ``brainunit`` quantities.

        See Also
        --------
        poisson_generator_ps.set : Update one or more parameters in place.

        Examples
        --------
        .. code-block:: python

           >>> import brainpy
           >>> import brainunit as u
           >>> gen = brainpy.state.poisson_generator_ps(
           ...     rate=800.0 * u.Hz,
           ...     dead_time=0.5 * u.ms,
           ...     start=5.0 * u.ms,
           ...     stop=100.0 * u.ms,
           ...     origin=2.0 * u.ms,
           ... )
           >>> params = gen.get()
           >>> params['rate']
           800.0
           >>> params['dead_time']
           0.5
           >>> params['stop']
           100.0
        """
        return {
            'rate': float(self.rate),
            'dead_time': float(self.dead_time),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    @property
    def step_spike_times_ms(self):
        r"""Precise spike times emitted in the most recent :meth:`update` call.

        Returns
        -------
        spike_times : tuple of numpy.ndarray
            Tuple of length ``np.prod(self.varshape)``. Each element is a
            one-dimensional ``numpy.ndarray`` with dtype ``float64`` containing
            the emitted precise spike times (ms) for that flattened output
            train in the latest simulation step. Arrays are empty when no
            spikes were emitted for the train in the most recent step.

        Notes
        -----
        The tuple is replaced atomically at each :meth:`update` call. Holding
        a reference to a previous value is safe because each call creates new
        arrays. Spike times are in the half-open interval
        ``(t_min_active, t_max_active]`` corresponding to the step window.

        See Also
        --------
        poisson_generator_ps.update : Produces the spike-time arrays stored
            here.
        """
        return self._last_step_spike_times_ms

    def _sample_initial_offset_ms(self, rng: np.random.Generator, inv_rate_ms: float) -> float:
        r"""Sample the first spike offset from the stationary backward-recurrence distribution.

        Uses the two-branch mixture that NEST employs to avoid transient rate
        bias when a stream starts mid-process:

        - **Dead-time branch** (probability ``dead_time * rate / 1000``):
          uniform draw on ``[0, dead_time)``.
        - **Exponential branch** (complementary probability):
          shifted exponential ``Exp(1/alpha) + dead_time``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Per-target RNG instance. Consumed in place.
        inv_rate_ms : float
            Scale parameter :math:`\alpha = 1000 / r - t_{\mathrm{dead}}` in
            ms. Must be non-negative (enforced by parameter validation).

        Returns
        -------
        offset_ms : float
            Initial spike offset in ms from ``t_min_active``. Always
            non-negative.
        """
        if self.dead_time > 0.0 and rng.random() < (self.dead_time * self.rate / 1000.0):
            # Uniform branch on [0, dead_time).
            return float(rng.random() * self.dead_time)
        # Exponential branch on [dead_time, +inf).
        return float(rng.exponential() * inv_rate_ms + self.dead_time)

    def _sample_isi_ms(self, rng: np.random.Generator, inv_rate_ms: float) -> float:
        r"""Sample one inter-spike interval (ISI) from the renewal distribution.

        Draws a single ISI as

        .. math::

           \Delta t = t_{\mathrm{dead}} + \xi \, \alpha, \qquad
           \xi \sim \mathrm{Exp}(1),

        where :math:`\alpha = 1000 / r - t_{\mathrm{dead}}` is the scale
        parameter in ms. When ``dead_time == 0``, this reduces to a pure
        exponential ISI, recovering the memoryless Poisson process.

        Parameters
        ----------
        rng : numpy.random.Generator
            Per-target RNG instance. Consumed in place; one exponential variate
            is drawn per call.
        inv_rate_ms : float
            Scale parameter :math:`\alpha = 1000 / r - t_{\mathrm{dead}}` in
            ms. Must be non-negative.

        Returns
        -------
        isi_ms : float
            Sampled inter-spike interval in ms. Always satisfies
            ``isi_ms >= dead_time``.
        """
        return float(rng.exponential() * inv_rate_ms + self.dead_time)

    def update(self, return_precise_times: bool = False):
        r"""Advance one simulation step and emit precise Poisson events.

        Parameters
        ----------
        return_precise_times : bool, optional
            If ``False`` (default), return only per-target spike
            multiplicities. If ``True``, also return per-target precise spike
            times emitted in the current step.

        Returns
        -------
        counts : numpy.ndarray
            Integer array with dtype ``int64`` and shape ``self.varshape``
            containing per-step spike multiplicities. Returned in both modes.

            - **Active and** ``rate > 0``: each element counts how many spikes
              fell in ``(t_min_active, t_max_active]`` for that output train.
            - **Inactive or** ``rate <= 0``: all entries are exactly ``0``.

        spike_times_tuple : tuple of numpy.ndarray, only when ``return_precise_times=True``
            Tuple of length ``np.prod(self.varshape)``. Each element is a
            one-dimensional ``float64`` array of emitted precise spike times
            (ms) for the corresponding flattened output train in this step.
            Arrays are empty when no spikes were emitted. Also accessible via
            :attr:`step_spike_times_ms` after the call.

        Raises
        ------
        ValueError
            If simulation step size ``dt`` is non-positive after conversion
            from the runtime environment.
        KeyError
            If required runtime entries (notably ``dt``) are unavailable in
            ``brainstate.environ``.
        TypeError
            If environment values cannot be converted to scalar milliseconds.

        Notes
        -----
        The update proceeds as follows each call:

        1. **Lazy init** -- If ``next_spike_time`` has not been created by
           :meth:`init_state`, it is initialized automatically with
           ``self.rng_seed``.
        2. **Window clipping** -- Computes ``t_min_active`` and
           ``t_max_active`` by intersecting the step interval
           ``[t, t + dt]`` with the configured activity window. Returns zeros
           immediately if the window is empty or ``rate <= 0``.
        3. **Stream initialization** -- For any target whose
           ``next_spike_time`` is ``-inf`` (first call or after a rate reset),
           the initial spike is placed at
           ``t_min_active + _sample_initial_offset_ms(...)`` using the
           stationary backward-recurrence distribution.
        4. **Spike emission loop** -- Advances each target stream forward,
           emitting spikes at ``next_t <= t_max_active`` and drawing new ISIs
           via :meth:`_sample_isi_ms` until the next spike lies outside the
           current step.
        5. **State update** -- Writes back ``next_spike_time``,
           ``last_spike_time``, and ``last_spike_offset`` states and caches
           per-target spike-time arrays in :attr:`step_spike_times_ms`.

        See Also
        --------
        poisson_generator_ps.init_state : State buffers consumed here.
        poisson_generator_ps.set : Update parameters between runs.
        poisson_generator_ps.step_spike_times_ms : Access precise times after
            update without the overhead of the return value.
        """
        if not hasattr(self, 'next_spike_time'):
            self.init_state()

        dt_ms = self._dt_ms()
        if dt_ms <= 0.0:
            raise ValueError('Simulation time step must be positive.')

        t_ms = self._current_time_ms()
        t_min_active = max(t_ms, self.origin + self.start)
        t_max_active = min(t_ms + dt_ms, self.origin + self.stop)

        counts = np.zeros(self._num_targets, dtype=np.int64)
        empty_events = tuple(np.asarray([], dtype=np.float64) for _ in range(self._num_targets))

        if self._num_targets == 0 or self.rate <= 0.0 or not (t_min_active < t_max_active):
            self._last_step_spike_times_ms = empty_events
            out = counts.reshape(self.varshape)
            if return_precise_times:
                return out, self._last_step_spike_times_ms
            return out

        inv_rate_ms = 1000.0 / self.rate - self.dead_time
        next_spike = np.asarray(self.next_spike_time.value, dtype=np.float64).copy()
        last_time = np.asarray(self.last_spike_time.value, dtype=np.float64).reshape(-1).copy()
        last_offset = np.asarray(self.last_spike_offset.value, dtype=np.float64).reshape(-1).copy()

        right_edge = t_ms + dt_ms
        events = []

        for i in range(self._num_targets):
            rng = self._rngs[i]
            next_t = float(next_spike[i])

            if np.isneginf(next_t):
                next_t = t_min_active + self._sample_initial_offset_ms(rng, inv_rate_ms)

            ev = []
            while next_t <= t_max_active:
                counts[i] += 1
                ev.append(next_t)
                last_time[i] = next_t
                off = right_edge - next_t
                if off < 0.0 and off > -1e-12:
                    off = 0.0
                last_offset[i] = off
                next_t += self._sample_isi_ms(rng, inv_rate_ms)

            next_spike[i] = next_t
            events.append(np.asarray(ev, dtype=np.float64))

        self.next_spike_time.value = next_spike
        self.last_spike_time.value = last_time.reshape(self.varshape)
        self.last_spike_offset.value = last_offset.reshape(self.varshape)
        self._last_step_spike_times_ms = tuple(events)

        out = counts.reshape(self.varshape)
        if return_precise_times:
            return out, self._last_step_spike_times_ms
        return out
