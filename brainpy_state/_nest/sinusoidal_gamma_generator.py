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
    :math:`\lambda(t) \ge 0` for all :math:`t`.

    **2. Renewal integral, closed-form increment, and hazard**

    For gamma order :math:`k = \mathrm{order}` and train-specific renewal
    origin :math:`t_0`, define the scaled integrated hazard as

    .. math::

       \Lambda(t) = k \int_{t_0}^{t} \lambda(s)\,ds.

    The implementation keeps ``t0_ms`` and ``Lambda_t0`` as per-train state
    variables and advances :math:`\Lambda` each step via the closed-form
    increment computed in :meth:`_delta_lambda`:

    .. math::

       \Delta\Lambda = k r (t_b - t_a)
       - \frac{k a}{\omega}\Bigl[
         \cos(\omega t_b + \phi) - \cos(\omega t_a + \phi)
       \Bigr].

    When ``amplitude == 0`` or ``frequency == 0`` (i.e. :math:`\omega = 0`),
    the cosine term is omitted and :math:`\Delta\Lambda = k r (t_b - t_a)`,
    which avoids division by zero and recovers the homogeneous Poisson limit
    (:math:`k = 1`) or homogeneous gamma limit (:math:`k > 1`).

    The per-step hazard (already multiplied by ``dt``) evaluated at time
    :math:`t` is

    .. math::

       h(t) = \Delta t \cdot
       \frac{k\,\lambda(t)\,\Lambda(t)^{k-1}\,e^{-\Lambda(t)}}
            {\Gamma(k,\,\Lambda(t))},

    where :math:`\Gamma(k, \Lambda)` is the upper incomplete gamma function
    evaluated via ``jax.lax.igammac`` and ``math.gamma``.  The ratio
    :math:`h(t)` approximates :math:`\Pr(\text{spike in } [t, t+\Delta t))`
    under the gamma renewal model.

    **3. Update ordering and activity-window semantics**

    Each call to :meth:`update` mirrors the ordering in NEST
    ``models/sinusoidal_gamma_generator.cpp``:

    1. Evaluate time at the right edge of the current step:
       ``t_eval = (step + 1) * dt``.
    2. Compute :math:`\lambda(t_{\mathrm{eval}})` and cache the value as the
       step-end instantaneous rate in spikes/s (accessible via
       :meth:`get_recorded_rate`).
    3. If the generator is active and :math:`\lambda(t_{\mathrm{eval}}) > 0`,
       compute the per-train hazard and sample Bernoulli decisions.
    4. Reset ``t0_ms`` and ``Lambda_t0`` to ``t_eval`` / ``0`` for any train
       that fired.
    5. Return binary spike outputs as ``int64`` with shape ``self.varshape``.

    NEST spike-generator activity semantics use the half-open-right window

    .. math::

       t_{\min} < n \le t_{\max},

    where :math:`n` is the current integer step index,
    ``t_min = round((origin + start) / dt)``, and
    ``t_max = round((origin + stop) / dt)`` after projection to grid steps.

    **4. Piecewise-integral semantics on parameter changes**

    When :meth:`set` is called after initialization, the existing per-train
    renewal state is first advanced to the change time :math:`t_c` using the
    *previous* process parameters, then future increments use the *new*
    parameters:

    .. math::

       \Lambda(t) = \Lambda_{\mathrm{old}}(t_c)
       + k_{\mathrm{new}} \int_{t_c}^{t} \lambda_{\mathrm{new}}(s)\,ds.

    This preserves renewal history across parameter updates, matching NEST
    ``SetStatus`` behavior.

    **5. Assumptions, constraints, and computational implications**

    - Public parameters are scalarized to ``float64`` (or ``int`` for
      ``rng_seed``); non-scalar inputs raise :class:`ValueError`.
    - Enforced constraints: ``order >= 1``, ``0 <= amplitude <= rate``,
      and ``stop >= start``.
    - When ``dt`` is available at construction time, finite
      ``origin`` / ``start`` / ``stop`` must be representable on the
      simulation grid (absolute tolerance ``1e-12`` in ``time / dt`` ratio).
    - ``individual_spike_trains=True`` allocates one independent renewal
      state per element of ``self.varshape`` and draws independent Bernoulli
      samples; ``individual_spike_trains=False`` maintains one shared renewal
      state and broadcasts the single Bernoulli draw to all outputs.
    - Per-step runtime is :math:`O(n_{\mathrm{trains}})` for hazard
      evaluation and sampling, with memory
      :math:`O(n_{\mathrm{trains}})` for ``t0_ms`` and ``Lambda_t0``.
    - At most one spike per train can be emitted per step because spike
      decisions are independent Bernoulli trials against the per-step hazard.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification for :class:`brainstate.nn.Dynamics`.
        ``self.varshape`` derived from ``in_size`` is the exact output shape
        of :meth:`update`; each element corresponds to one emitted train.
        Default is ``1``.
    rate : ArrayLike, optional
        Scalar mean firing rate in spikes/s (Hz), shape ``()`` after
        conversion. Accepted as a scalar ``ArrayLike`` or a
        :class:`brainunit.Quantity` convertible to ``u.Hz``.
        Must satisfy ``0 <= amplitude <= rate``.
        Default is ``0.0 * u.Hz``.
    amplitude : ArrayLike, optional
        Scalar sinusoidal modulation amplitude in spikes/s (Hz), shape ``()``
        after conversion. Must satisfy ``0 <= amplitude <= rate`` after
        conversion; the constraint ensures :math:`\lambda(t) \ge 0`.
        Default is ``0.0 * u.Hz``.
    frequency : ArrayLike, optional
        Scalar modulation frequency in Hz, shape ``()`` after conversion.
        Internally mapped to angular frequency
        :math:`\omega = 2\pi f / 1000` (rad/ms).
        Default is ``0.0 * u.Hz``.
    phase : ArrayLike, optional
        Scalar initial phase in degrees, shape ``()`` after conversion;
        internally converted to radians as :math:`\phi = \phi_{\deg} \pi / 180`.
        Default is ``0.0``.
    order : ArrayLike, optional
        Scalar gamma renewal order :math:`k`, shape ``()`` after conversion.
        Must satisfy ``order >= 1``; order ``1`` recovers an inhomogeneous
        Poisson process.
        Default is ``1.0``.
    individual_spike_trains : bool, optional
        Spike-generation mode selector.
        If ``True``, each output index in ``self.varshape`` maintains
        independent renewal state ``(t0_ms, Lambda_t0)`` and receives
        independent Bernoulli draws.
        If ``False``, one shared renewal process is maintained and the same
        binary spike decision is broadcast to all outputs.
        Default is ``True``.
    start : ArrayLike, optional
        Scalar relative activation start time in ms, shape ``()`` after
        conversion. Effective lower activity bound is ``origin + start``;
        the bound is exclusive in step space (``t_min_step < curr_step``).
        Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative deactivation stop time in ms, shape ``()`` after
        conversion. ``None`` maps to ``+inf``. Effective upper activity bound
        is ``origin + stop``; the bound is inclusive in step space
        (``curr_step <= t_max_step``). Must satisfy ``stop >= start`` after
        conversion.
        Default is ``None``.
    origin : ArrayLike, optional
        Scalar origin offset in ms, shape ``()`` after conversion. Added to
        ``start`` and ``stop`` to obtain the absolute activity bounds
        ``t_min`` and ``t_max``.
        Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed used to initialize ``jax.random.PRNGKey`` during
        :meth:`init_state` and lazy initialization in :meth:`update`.
        Default is ``0``.
    name : str or None, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.
        Default is ``None``.

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
         - Phase in degrees mapped to radians as :math:`\phi_{\deg}\pi/180`.
       * - ``order``
         - ``1.0``
         - :math:`k`
         - Gamma renewal order; ``1`` = inhomogeneous Poisson.
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower bound of activity window.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound; ``None`` maps to :math:`+\infty`.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{origin}}`
         - Global time offset added to ``start`` and ``stop``.
       * - ``in_size``
         - ``1``
         - —
         - Defines ``self.varshape`` and the total output train count.
       * - ``individual_spike_trains``
         - ``True``
         - —
         - Independent per-output renewal states vs. shared broadcast process.
       * - ``rng_seed``
         - ``0``
         - —
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
        ``stop < start``; or if finite ``origin`` / ``start`` / ``stop`` are
        not multiples of the simulation resolution when ``dt`` is available.
    TypeError
        If provided values cannot be converted to numeric values or to the
        required units (e.g. a non-convertible ``u.Hz`` or ``u.ms`` quantity).
    KeyError
        At runtime in :meth:`update`, if required simulation-context entries
        (notably ``dt``) are unavailable from ``brainstate.environ``.

    Notes
    -----
    - Hazard values are computed in ``float64``; tiny negative
      :math:`\Lambda` values arising from floating-point roundoff are clamped
      to zero before hazard evaluation.
    - The value returned by :meth:`get_recorded_rate` is the step-end
      instantaneous rate in spikes/s, matching NEST's ``rate`` recordable.
    - Renewal state is revalidated against the timing grid whenever ``dt``
      changes between :meth:`update` calls.

    Examples
    --------
    Simulate a 2×3 array of independent sinusoidally modulated gamma trains:

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

    Use ``individual_spike_trains=False`` and update parameters at runtime:

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
    gamma_sup_generator : Superposition of independent gamma-renewal processes.

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
        r"""Initialize random key and per-train renewal state.

        Allocates three :class:`brainstate.ShortTermState` variables:

        - ``rng_key``: a JAX ``PRNGKey`` seeded from ``self.rng_seed``.
        - ``t0_ms``: per-train renewal origin, initialized to the current
          simulation time (``float64`` array of length ``self._num_trains``).
        - ``Lambda_t0``: per-train accumulated scaled hazard at ``t0_ms``,
          initialized to zero (``float64`` array of length ``self._num_trains``).
        - ``_recorded_rate_hz``: cached step-end instantaneous rate in
          spikes/s, initialized to ``0.0``.

        The timing cache (``_t_min_step``, ``_t_max_step``) is also refreshed
        if ``dt`` is available in the current ``brainstate.environ`` context.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused placeholder kept for :class:`brainstate.nn.Dynamics`
            signature compatibility. Default is ``None``.
        **kwargs
            Unused extra keyword arguments; silently ignored.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If finite ``origin`` / ``start`` / ``stop`` do not lie on the
            simulation grid when ``dt`` is available.
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
        r"""Update public parameters and refresh the process and timing caches.

        Any parameter left at its sentinel value ``_UNSET`` retains its
        current value. When called after :meth:`init_state`, the internal
        renewal state is first advanced to the current simulation time using
        the *previous* process parameters before switching to the new ones,
        preserving the piecewise-integral semantics described in the class
        docstring. If ``individual_spike_trains`` changes in a way that alters
        the required number of trains, ``t0_ms`` and ``Lambda_t0`` are
        grown (new trains start fresh) or truncated accordingly.

        Parameters
        ----------
        rate : ArrayLike or None, optional
            New scalar mean rate in spikes/s (Hz), shape ``()`` after
            conversion. ``_UNSET`` retains the current value.
        amplitude : ArrayLike or None, optional
            New scalar modulation amplitude in spikes/s (Hz), shape ``()``
            after conversion. ``_UNSET`` retains the current value.
        frequency : ArrayLike or None, optional
            New scalar modulation frequency in Hz, shape ``()`` after
            conversion. ``_UNSET`` retains the current value.
        phase : ArrayLike or None, optional
            New scalar modulation phase in degrees, shape ``()`` after
            conversion. ``_UNSET`` retains the current value.
        order : ArrayLike or None, optional
            New scalar gamma order :math:`k`, shape ``()`` after conversion.
            ``_UNSET`` retains the current value.
        individual_spike_trains : bool or None, optional
            New spike-generation mode. ``_UNSET`` retains the current value.
        start : ArrayLike or None, optional
            New scalar relative activation start time in ms, shape ``()``
            after conversion. ``_UNSET`` retains the current value.
        stop : ArrayLike, None, or sentinel, optional
            New scalar relative stop time in ms, shape ``()`` after
            conversion; explicit ``None`` maps to ``+inf``. ``_UNSET``
            retains the current value.
        origin : ArrayLike or None, optional
            New scalar origin offset in ms, shape ``()`` after conversion.
            ``_UNSET`` retains the current value.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If scalar conversion fails due to non-scalar inputs; if the
            constraints ``order >= 1``, ``0 <= amplitude <= rate``, or
            ``stop >= start`` are violated after resolving new values; or if
            finite timing parameters do not lie on the simulation grid when
            ``dt`` is available.
        TypeError
            If numeric or unit conversion fails for any supplied input.
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
        r"""Return current public parameters as plain Python scalars.

        Returns
        -------
        out : dict
            Mapping of parameter names to their current values. Keys and
            value types are:

            - ``'rate'`` : ``float`` — mean firing rate in spikes/s.
            - ``'amplitude'`` : ``float`` — sinusoidal modulation amplitude
              in spikes/s.
            - ``'frequency'`` : ``float`` — modulation frequency in Hz.
            - ``'phase'`` : ``float`` — modulation phase in degrees.
            - ``'order'`` : ``float`` — gamma renewal order :math:`k`.
            - ``'individual_spike_trains'`` : ``bool`` — spike-generation
              mode flag.
            - ``'start'`` : ``float`` — relative activation start in ms.
            - ``'stop'`` : ``float`` — relative deactivation stop in ms
              (``float('inf')`` when no stop was set).
            - ``'origin'`` : ``float`` — time-origin offset in ms.
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
        r"""Return the latest step-end instantaneous rate in spikes/s.

        The value is updated by :meth:`update` at each simulation step to
        :math:`\lambda(t_{\mathrm{eval}}) \times 1000` spikes/s, where
        :math:`t_{\mathrm{eval}} = (\mathrm{step} + 1) \times dt` is the
        right edge of the current step. This matches NEST's ``rate``
        recordable quantity.

        Returns
        -------
        out : float
            Most recently cached instantaneous rate in spikes/s (``float64``
            scalar). Returns ``0.0`` if :meth:`init_state` has not been
            called yet.
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
        r"""Advance one simulation step and emit binary spike events.

        Reads the current time ``t`` and resolution ``dt`` from
        ``brainstate.environ``, lazily calls :meth:`init_state` if state has
        not been allocated, and refreshes the timing cache if ``dt`` has
        changed since the last call. The step-end time
        ``t_eval = (step + 1) * dt`` is used for rate evaluation and
        :math:`\Lambda` accumulation. Trains outside the active window or
        with :math:`\lambda(t_{\mathrm{eval}}) \le 0` receive zero spikes
        without consuming random draws.

        Returns
        -------
        out : jax.Array
            ``int64`` JAX array with shape ``self.varshape``. Each element is
            ``0`` or ``1``, giving the binary spike decision for the
            corresponding output train in the current step. When
            ``individual_spike_trains=False``, all elements share the same
            value.

        Raises
        ------
        KeyError
            If ``dt`` is not available in the current ``brainstate.environ``
            context.
        ValueError
            If timing-parameter grid validation fails after the simulation
            resolution changes between calls.
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
