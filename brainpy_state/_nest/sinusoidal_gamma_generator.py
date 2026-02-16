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

from __future__ import annotations

import math

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'sinusoidal_gamma_generator',
]


_UNSET = object()


class sinusoidal_gamma_generator(brainstate.nn.Dynamics):
    r"""Sinusoidally modulated gamma spike generator compatible with NEST.

    Description
    -----------
    ``sinusoidal_gamma_generator`` re-implements NEST's stimulation device of
    the same name. It emits binary spikes from an inhomogeneous gamma renewal
    process whose instantaneous rate is sinusoidally modulated.

    **1. Instantaneous-rate model**

    The internal rate in spikes/ms is

    .. math::

       \lambda(t) = r + a \sin(\omega t + \phi),

    with parameter-to-symbol conversion:

    - :math:`r = \mathrm{rate}/1000`,
    - :math:`a = \mathrm{amplitude}/1000`,
    - :math:`\omega = 2\pi \cdot \mathrm{frequency}/1000` (rad/ms),
    - :math:`\phi = \mathrm{phase}\cdot\pi/180` (rad).

    The validated constraint ``0 <= amplitude <= rate`` guarantees
    :math:`\lambda(t) >= 0` for all :math:`t`.

    **2. Renewal integral, closed-form increment, and hazard**

    For gamma order :math:`k = \mathrm{order}` and train-specific renewal
    origin :math:`t_0`, define

    .. math::

       \Lambda(t) = k\int_{t_0}^{t}\lambda(s)\,ds.

    The implementation keeps ``t0_ms`` and ``Lambda_t0`` as state and updates
    :math:`\Lambda` by the closed-form increment used in
    :meth:`_delta_lambda`:

    .. math::

       \Delta\Lambda = k r (t_b - t_a)
       - \frac{k a}{\omega}\left[
         \cos(\omega t_b + \phi) - \cos(\omega t_a + \phi)
       \right].

    For ``amplitude == 0`` or ``frequency == 0`` (equivalently
    :math:`\omega = 0`), the sinusoidal term is skipped and
    :math:`\Delta\Lambda = k r (t_b - t_a)` to avoid division by zero and to
    match the homogeneous-rate limit.

    The per-step hazard already multiplied by ``dt`` is

    .. math::

       h(t) = dt \cdot
       \frac{k\,\lambda(t)\,\Lambda(t)^{k-1}e^{-\Lambda(t)}}
            {\Gamma(k,\Lambda(t))},

    where :math:`\Gamma(k,\Lambda)` is the upper incomplete gamma function.
    The denominator is evaluated through ``jax.lax.igammac`` and
    ``math.gamma``.

    **3. Update ordering and activity-window semantics**

    Update order mirrors NEST ``models/sinusoidal_gamma_generator.cpp``:

    1. Evaluate time at the right edge of the current step: ``t_eval = t + dt``.
    2. Evaluate :math:`\lambda(t_eval)` and store recorded rate in spikes/s.
    3. If active and :math:`\lambda(t_eval) > 0`, compute hazard and sample.
    4. Reset ``t0_ms`` and ``Lambda_t0`` for trains that emitted a spike.
    5. Return binary spike outputs as ``int64`` with shape ``self.varshape``.

    NEST spike-generator activity semantics are

    .. math::

       t_{\min} < n \le t_{\max},

    where :math:`n` is the current integer step index and
    ``t_min = origin + start``, ``t_max = origin + stop`` after projection to
    grid steps.

    **4. Piecewise-integral semantics on parameter changes**

    When :meth:`set` is called after initialization, the existing renewal
    state is advanced to the change time using previous process parameters,
    then future increments use new parameters:

    .. math::

       \Lambda(t) = \Lambda_{\mathrm{old}}(t_c)
       + k_{\mathrm{new}}\int_{t_c}^{t}\lambda_{\mathrm{new}}(s)\,ds.

    This matches NEST behavior for preserving renewal history across
    parameter updates.

    **5. Assumptions, constraints, and computational implications**

    - Public parameters are scalarized to ``float64`` (or ``int`` for
      ``rng_seed``); non-scalar inputs raise :class:`ValueError`.
    - Enforced constraints: ``order >= 1``, ``0 <= amplitude <= rate``,
      and ``stop >= start``.
    - When ``dt`` is available, finite ``origin``/``start``/``stop`` must be
      representable on the simulation grid (absolute tolerance ``1e-12`` in
      ``time / dt`` ratio).
    - ``individual_spike_trains=True`` keeps one renewal state per output;
      ``False`` keeps one shared renewal state and broadcasts spikes.
    - Per-step runtime is :math:`O(n_{\mathrm{trains}})` for hazard
      evaluation/sampling with memory :math:`O(n_{\mathrm{trains}})` for
      ``t0_ms`` and ``Lambda_t0``.
    - By design, at most one spike per train can be emitted per step because
      spike decisions are Bernoulli threshold comparisons against per-step
      hazard values.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification for :class:`brainstate.nn.Dynamics`.
        ``self.varshape`` derived from ``in_size`` is the exact output shape
        of :meth:`update`; each element corresponds to one emitted train.
        Default is ``1``.
    rate : ArrayLike, optional
        Scalar mean firing rate in spikes/s (Hz), shape ``()`` after
        conversion. Accepted as scalar ``ArrayLike`` or
        :class:`brainunit.Quantity` convertible to ``u.Hz``.
        Default is ``0.0 * u.Hz``.
    amplitude : ArrayLike, optional
        Scalar modulation amplitude in spikes/s (Hz), shape ``()`` after
        conversion. Must satisfy ``0 <= amplitude <= rate`` after conversion.
        Default is ``0.0 * u.Hz``.
    frequency : ArrayLike, optional
        Scalar modulation frequency in Hz, shape ``()`` after conversion.
        Internally mapped to angular frequency in rad/ms.
        Default is ``0.0 * u.Hz``.
    phase : ArrayLike, optional
        Scalar phase in degrees, shape ``()`` after conversion; internally
        converted to radians.
        Default is ``0.0``.
    order : ArrayLike, optional
        Scalar gamma order :math:`k`, shape ``()`` after conversion.
        Must satisfy ``order >= 1``.
        Default is ``1.0``.
    individual_spike_trains : bool, optional
        Spike-generation mode selector.
        If ``True``, each output index in ``self.varshape`` keeps independent
        renewal state and independent random draws.
        If ``False``, one shared renewal process is sampled and the same
        binary spike value is broadcast to all outputs.
        Default is ``True``.
    start : ArrayLike, optional
        Scalar relative activation start time in ms, shape ``()`` after
        conversion. Effective lower activity bound is ``origin + start`` and
        is exclusive in step space.
        Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative deactivation stop time in ms, shape ``()`` after
        conversion. ``None`` maps to ``+inf``. Effective upper activity bound
        is ``origin + stop`` and is inclusive in step space.
        Must satisfy ``stop >= start`` after conversion.
        Default is ``None``.
    origin : ArrayLike, optional
        Scalar origin offset in ms, shape ``()`` after conversion, added to
        ``start`` and ``stop`` to obtain absolute activity bounds.
        Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed used to initialize ``jax.random.PRNGKey`` during
        :meth:`init_state` and lazy initialization in :meth:`update`.
        Default is ``0``.
    name : str or None, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 22 18 20 40

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``rate``
         - ``0.0 * u.Hz``
         - :math:`r`
         - Baseline firing-rate term in spikes/ms after division by ``1000``.
       * - ``amplitude``
         - ``0.0 * u.Hz``
         - :math:`a`
         - Sinusoidal modulation amplitude in spikes/ms after division by ``1000``.
       * - ``frequency``
         - ``0.0 * u.Hz``
         - :math:`f`
         - Frequency in Hz mapped to :math:`\omega = 2\pi f/1000` (rad/ms).
       * - ``phase``
         - ``0.0``
         - :math:`\phi`
         - Phase in degrees mapped to radians.
       * - ``order``
         - ``1.0``
         - :math:`k`
         - Gamma renewal order used in :math:`\Lambda` and hazard.
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive activity lower bound.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive activity upper bound; ``None`` maps to ``+\infty``.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global time offset added to ``start`` and ``stop``.
       * - ``in_size``
         - ``1``
         - -
         - Defines ``self.varshape`` and output train count.
       * - ``individual_spike_trains``
         - ``True``
         - -
         - Independent per-output renewal states vs shared broadcast process.
       * - ``rng_seed``
         - ``0``
         - -
         - Seed for JAX random key initialization and splitting.

    Returns
    -------
    out : Any
        Dynamics node instance. Each :meth:`update` call returns an ``int64``
        JAX array with shape ``self.varshape`` containing binary spike events
        (``0`` or ``1``) for the current simulation step.

    Raises
    ------
    ValueError
        If scalar-conversion fails due to non-scalar inputs; if
        ``0 <= amplitude <= rate`` is violated; if ``order < 1``; if
        ``stop < start``; or if finite ``origin``/``start``/``stop`` are not
        multiples of simulation resolution when ``dt`` is available.
    TypeError
        If provided values cannot be converted to numeric values or required
        units (for example, non-convertible ``u.Hz``/``u.ms`` quantities).
    KeyError
        At runtime, if required simulation-context entries (for example
        ``dt`` in :meth:`update`) are unavailable from ``brainstate.environ``.

    Notes
    -----
    - Hazard values are computed in ``float64`` and tiny negative
      :math:`\Lambda` values from roundoff are clamped to zero before hazard
      evaluation.
    - Recorded rate from :meth:`get_recorded_rate` is the step-end
      instantaneous rate in spikes/s, matching NEST's ``rate`` recordable.
    - Renewal state is revalidated against timing-grid assumptions whenever
      ``dt`` changes.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.sinusoidal_gamma_generator(
       ...         in_size=(2, 3),
       ...         rate=50.0 * u.Hz,
       ...         amplitude=20.0 * u.Hz,
       ...         frequency=8.0 * u.Hz,
       ...         phase=30.0,
       ...         order=3.0,
       ...         start=5.0 * u.ms,
       ...         stop=80.0 * u.ms,
       ...         rng_seed=9,
       ...     )
       ...     with brainstate.environ.context(t=12.0 * u.ms):
       ...         spikes = gen.update()
       ...     _ = spikes.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.sinusoidal_gamma_generator(
       ...         individual_spike_trains=False
       ...     )
       ...     gen.set(rate=40.0 * u.Hz, amplitude=10.0 * u.Hz, order=2.0)
       ...     params = gen.get()
       ...     _ = params['rate'], params['order']

    See Also
    --------
    sinusoidal_poisson_generator : Sinusoidally modulated Poisson generator.
    gamma_sup_generator : Superposition of gamma-renewal processes.

    References
    ----------
    .. [1] NEST source:
           ``models/sinusoidal_gamma_generator.h`` and
           ``models/sinusoidal_gamma_generator.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/sinusoidal_gamma_generator.html
    .. [3] NEST source:
           ``nestkernel/stimulation_device.cpp``.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        amplitude: ArrayLike = 0. * u.Hz,
        frequency: ArrayLike = 0. * u.Hz,
        phase: ArrayLike = 0.0,
        order: ArrayLike = 1.0,
        individual_spike_trains: bool = True,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.amplitude = self._to_scalar_rate_hz(amplitude)
        self.frequency = self._to_scalar_rate_hz(frequency)
        self.phase = self._to_scalar_float(phase, name='phase')
        self.order = self._to_scalar_float(order, name='order')
        self.individual_spike_trains = bool(individual_spike_trains)

        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate_hz=self.rate,
            amplitude_hz=self.amplitude,
            order=self.order,
            start_ms=self.start,
            stop_ms=self.stop,
        )

        self._num_targets = int(np.prod(self.varshape))
        self._num_trains = self._num_targets if self.individual_spike_trains else 1

        self._rate_per_ms = 0.0
        self._amplitude_per_ms = 0.0
        self._om_rad_per_ms = 0.0
        self._phi_rad = 0.0
        self._proc_params = (0.0, 0.0, 1.0, 0.0, 0.0)
        self._proc_params_prev = self._proc_params
        self._refresh_process_parameter_cache()

        self._dt_cache_ms = np.nan
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max
        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    @staticmethod
    def _to_scalar_time_ms(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.ms), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('Time parameters must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_rate_hz(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.Hz), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('Rate parameters must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _time_to_step(time_ms: float, dt_ms: float) -> int:
        return int(np.rint(time_ms / dt_ms))

    @staticmethod
    def _assert_grid_time(name: str, time_ms: float, dt_ms: float):
        if not np.isfinite(time_ms):
            return
        ratio = time_ms / dt_ms
        nearest = np.rint(ratio)
        if not math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be a multiple of the simulation resolution.')

    @staticmethod
    def _validate_parameters(
        *,
        rate_hz: float,
        amplitude_hz: float,
        order: float,
        start_ms: float,
        stop_ms: float,
    ):
        if order < 1.0:
            raise ValueError('The gamma order must be at least 1.')
        if not (0.0 <= amplitude_hz <= rate_hz):
            raise ValueError('Rate parameters must fulfill 0 <= amplitude <= rate.')
        if stop_ms < start_ms:
            raise ValueError('stop >= start required.')

    def _dt_ms(self) -> float:
        dt = brainstate.environ.get_dt()
        return self._to_scalar_time_ms(dt)

    def _maybe_dt_ms(self) -> float | None:
        dt = brainstate.environ.get('dt', default=None)
        if dt is None:
            return None
        return self._to_scalar_time_ms(dt)

    def _current_time_ms(self) -> float:
        t = brainstate.environ.get('t', default=0. * u.ms)
        if t is None:
            return 0.0
        return self._to_scalar_time_ms(t)

    def _refresh_timing_cache(self, dt_ms: float):
        self._assert_grid_time('origin', self.origin, dt_ms)
        self._assert_grid_time('start', self.start, dt_ms)
        self._assert_grid_time('stop', self.stop, dt_ms)

        self._t_min_step = self._time_to_step(self.origin + self.start, dt_ms)
        if np.isfinite(self.stop):
            self._t_max_step = self._time_to_step(self.origin + self.stop, dt_ms)
        else:
            self._t_max_step = np.iinfo(np.int64).max
        self._dt_cache_ms = float(dt_ms)

    def _refresh_process_parameter_cache(self):
        self._rate_per_ms = self.rate / 1000.0
        self._amplitude_per_ms = self.amplitude / 1000.0
        self._om_rad_per_ms = self.frequency * (2.0 * math.pi / 1000.0)
        self._phi_rad = self.phase * (math.pi / 180.0)
        self._proc_params = (
            self._om_rad_per_ms,
            self._phi_rad,
            self.order,
            self._rate_per_ms,
            self._amplitude_per_ms,
        )

    def _is_active(self, curr_step: int) -> bool:
        return (self._t_min_step < curr_step) and (curr_step <= self._t_max_step)

    @staticmethod
    def _delta_lambda(params: tuple[float, float, float, float, float], t_a, t_b):
        om, phi, order, rate, amplitude = params
        t_a_arr = np.asarray(t_a, dtype=np.float64)
        if t_a_arr.ndim == 0:
            if float(t_a_arr) == float(t_b):
                return np.asarray(0.0, dtype=np.float64)
        elif np.all(t_a_arr == float(t_b)):
            return np.zeros_like(t_a_arr, dtype=np.float64)

        delta = order * rate * (t_b - t_a_arr)
        if abs(amplitude) > 0.0 and abs(om) > 0.0:
            delta += -order * amplitude / om * (
                np.cos(om * t_b + phi) - np.cos(om * t_a_arr + phi)
            )
        return delta

    def _accumulate_lambda_to_time(self, t_ms: float):
        if self._num_trains == 0:
            return
        t0 = np.asarray(self.t0_ms.value, dtype=np.float64).reshape(-1).copy()
        lam0 = np.asarray(self.Lambda_t0.value, dtype=np.float64).reshape(-1).copy()

        lam0 += np.asarray(self._delta_lambda(self._proc_params_prev, t0, t_ms), dtype=np.float64)
        t0.fill(t_ms)

        self.t0_ms.value = t0
        self.Lambda_t0.value = lam0

    def _resize_train_state(self, now_ms: float, new_num_trains: int):
        old_t0 = np.asarray(self.t0_ms.value, dtype=np.float64).reshape(-1)
        old_lam = np.asarray(self.Lambda_t0.value, dtype=np.float64).reshape(-1)
        old_n = old_t0.size

        if new_num_trains == old_n:
            return
        if new_num_trains < old_n:
            self.t0_ms.value = old_t0[:new_num_trains].copy()
            self.Lambda_t0.value = old_lam[:new_num_trains].copy()
            return

        add_n = new_num_trains - old_n
        self.t0_ms.value = np.concatenate(
            [old_t0, np.full(add_n, now_ms, dtype=np.float64)]
        )
        self.Lambda_t0.value = np.concatenate(
            [old_lam, np.zeros(add_n, dtype=np.float64)]
        )

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize random key and per-train renewal state.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused placeholder kept for :class:`brainstate.nn.Dynamics`
            signature compatibility.
        **kwargs
            Unused extra keyword arguments.

        Returns
        -------
        None
        """
        del batch_size, kwargs
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))

        curr_t_ms = self._current_time_ms()
        self.t0_ms = brainstate.ShortTermState(
            np.full(self._num_trains, curr_t_ms, dtype=np.float64)
        )
        self.Lambda_t0 = brainstate.ShortTermState(
            np.zeros(self._num_trains, dtype=np.float64)
        )
        self._recorded_rate_hz = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))
        self._proc_params_prev = self._proc_params

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        amplitude: ArrayLike | object = _UNSET,
        frequency: ArrayLike | object = _UNSET,
        phase: ArrayLike | object = _UNSET,
        order: ArrayLike | object = _UNSET,
        individual_spike_trains: bool | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Set public parameters and refresh cached process/timing state.

        Parameters
        ----------
        rate : ArrayLike or object, optional
            Scalar mean rate in spikes/s (Hz). ``_UNSET`` keeps current value.
        amplitude : ArrayLike or object, optional
            Scalar modulation amplitude in spikes/s (Hz). ``_UNSET`` keeps
            current value.
        frequency : ArrayLike or object, optional
            Scalar modulation frequency in Hz. ``_UNSET`` keeps current value.
        phase : ArrayLike or object, optional
            Scalar modulation phase in degrees. ``_UNSET`` keeps current value.
        order : ArrayLike or object, optional
            Scalar gamma order. ``_UNSET`` keeps current value.
        individual_spike_trains : bool or object, optional
            Sampling mode flag. ``_UNSET`` keeps current value.
        start : ArrayLike or object, optional
            Scalar relative start time in ms. ``_UNSET`` keeps current value.
        stop : ArrayLike, None, or object, optional
            Scalar relative stop time in ms, or ``None`` for ``+inf``.
            ``_UNSET`` keeps current value.
        origin : ArrayLike or object, optional
            Scalar origin time in ms. ``_UNSET`` keeps current value.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If scalar conversion fails; if validated constraints
            (``order >= 1``, ``0 <= amplitude <= rate``, ``stop >= start``)
            are violated; or if finite timing parameters do not lie on the
            simulation grid when ``dt`` is available.
        TypeError
            If numeric or unit conversion fails for supplied inputs.
        """
        now_ms = self._current_time_ms() if hasattr(self, 't0_ms') else 0.0
        if hasattr(self, 't0_ms'):
            self._accumulate_lambda_to_time(now_ms)

        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_amplitude = (
            self.amplitude if amplitude is _UNSET else self._to_scalar_rate_hz(amplitude)
        )
        new_frequency = (
            self.frequency if frequency is _UNSET else self._to_scalar_rate_hz(frequency)
        )
        new_phase = self.phase if phase is _UNSET else self._to_scalar_float(phase, name='phase')
        new_order = self.order if order is _UNSET else self._to_scalar_float(order, name='order')
        new_individual = (
            self.individual_spike_trains
            if individual_spike_trains is _UNSET
            else bool(individual_spike_trains)
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
            rate_hz=new_rate,
            amplitude_hz=new_amplitude,
            order=new_order,
            start_ms=new_start,
            stop_ms=new_stop,
        )

        self.rate = new_rate
        self.amplitude = new_amplitude
        self.frequency = new_frequency
        self.phase = new_phase
        self.order = new_order
        self.individual_spike_trains = new_individual
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        self._num_trains = self._num_targets if self.individual_spike_trains else 1
        self._refresh_process_parameter_cache()

        if hasattr(self, 't0_ms'):
            self._resize_train_state(now_ms, self._num_trains)
            self._proc_params_prev = self._proc_params

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    def get(self) -> dict:
        """Return current public parameters as plain Python scalars.

        Returns
        -------
        out : dict
            Dictionary with keys ``rate``, ``frequency``, ``phase``,
            ``amplitude``, ``order``, ``individual_spike_trains``, ``start``,
            ``stop``, and ``origin``. Numeric values are returned as
            ``float``; ``individual_spike_trains`` is returned as ``bool``.
        """
        return {
            'rate': float(self.rate),
            'frequency': float(self.frequency),
            'phase': float(self.phase),
            'amplitude': float(self.amplitude),
            'order': float(self.order),
            'individual_spike_trains': bool(self.individual_spike_trains),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def get_recorded_rate(self) -> float:
        """Return latest step-end instantaneous rate in spikes/s.

        Returns
        -------
        out : float
            Most recently cached instantaneous rate in spikes/s. Returns
            ``0.0`` if state has not been initialized yet.
        """
        if not hasattr(self, '_recorded_rate_hz'):
            return 0.0
        return float(np.asarray(self._recorded_rate_hz.value, dtype=np.float64).reshape(()))

    def _sample_uniform(self, shape=()):
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.uniform(subkey, shape=shape, dtype=jnp.float64)

    def _compute_hazard(self, lambda_val: np.ndarray, rate_per_ms: float, dt_ms: float) -> np.ndarray:
        hazard = np.zeros_like(lambda_val, dtype=np.float64)

        # Guard tiny negative values caused by floating-point roundoff only.
        tiny_neg = np.logical_and(lambda_val < 0.0, lambda_val > -1e-15)
        if np.any(tiny_neg):
            lambda_val = lambda_val.copy()
            lambda_val[tiny_neg] = 0.0

        valid = lambda_val >= 0.0
        if not np.any(valid):
            return hazard

        lam = lambda_val[valid]
        q = np.asarray(
            jax.lax.igammac(
                jnp.asarray(self.order, dtype=jnp.float64),
                jnp.asarray(lam, dtype=jnp.float64),
            ),
            dtype=np.float64,
        )
        denom = math.gamma(self.order) * q
        numer = (
            dt_ms
            * self.order
            * rate_per_ms
            * np.power(lam, self.order - 1.0)
            * np.exp(-lam)
        )
        hazard_valid = np.divide(
            numer,
            denom,
            out=np.zeros_like(numer, dtype=np.float64),
            where=denom > 0.0,
        )
        hazard[valid] = hazard_valid
        return hazard

    def update(self):
        """Advance one simulation step and emit binary spike events.

        Returns
        -------
        out : Any
            ``int64`` JAX array with shape ``self.varshape`` containing one
            binary spike decision (``0`` or ``1``) per emitted train for the
            current step.

        Raises
        ------
        KeyError
            If required simulation-context entries (notably ``dt``) are not
            available through ``brainstate.environ``.
        ValueError
            If timing-parameter grid validation fails after a simulation
            resolution change.
        """
        if not hasattr(self, 'rng_key'):
            self.init_state()

        dt_ms = self._dt_ms()
        curr_t_ms = self._current_time_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_timing_cache(dt_ms)

        curr_step = self._time_to_step(curr_t_ms, dt_ms)
        t_eval_ms = (curr_step + 1) * dt_ms

        rate_per_ms = self._rate_per_ms + self._amplitude_per_ms * math.sin(
            self._om_rad_per_ms * t_eval_ms + self._phi_rad
        )
        self._recorded_rate_hz.value = jnp.asarray(
            rate_per_ms * 1000.0,
            dtype=jnp.float64,
        )

        if (
            self._num_trains == 0
            or rate_per_ms <= 0.0
            or (not self._is_active(curr_step))
        ):
            return jnp.zeros(self.varshape, dtype=jnp.int64)

        t0 = np.asarray(self.t0_ms.value, dtype=np.float64).reshape(-1).copy()
        lam0 = np.asarray(self.Lambda_t0.value, dtype=np.float64).reshape(-1).copy()
        lambda_eval = lam0 + np.asarray(
            self._delta_lambda(self._proc_params, t0, t_eval_ms),
            dtype=np.float64,
        )

        hazard = self._compute_hazard(lambda_eval, rate_per_ms, dt_ms)

        if self.individual_spike_trains:
            draws = np.asarray(
                self._sample_uniform(shape=(self._num_trains,)),
                dtype=np.float64,
            )
            spikes = draws < hazard
            if np.any(spikes):
                t0[spikes] = t_eval_ms
                lam0[spikes] = 0.0
            self.t0_ms.value = t0
            self.Lambda_t0.value = lam0
            return jnp.asarray(spikes.reshape(self.varshape), dtype=jnp.int64)

        draw = float(np.asarray(self._sample_uniform(shape=()), dtype=np.float64).reshape(()))
        spike = int(draw < float(hazard[0]))
        if spike:
            t0[0] = t_eval_ms
            lam0[0] = 0.0
            self.t0_ms.value = t0
            self.Lambda_t0.value = lam0
        return jnp.full(self.varshape, spike, dtype=jnp.int64)
