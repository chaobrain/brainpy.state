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

import brainstate

from ._base import NESTDevice
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'sinusoidal_poisson_generator',
]


_UNSET = object()


class sinusoidal_poisson_generator(NESTDevice):
    r"""Sinusoidally modulated Poisson spike generator compatible with NEST.

    Description
    -----------
    ``sinusoidal_poisson_generator`` re-implements NEST's stimulation device
    of the same name and emits per-step spike multiplicities.

    **1. Stochastic model and discretization**

    The instantaneous firing rate in spikes/s is

    .. math::

       f(t) = \max\left(
         0,\ r + a \sin\left( 2\pi f_{\mathrm{mod}} t / 1000 + \phi \right)
       \right),

    where:

    - :math:`r` is ``rate`` (spikes/s),
    - :math:`a` is ``amplitude`` (spikes/s),
    - :math:`f_{\mathrm{mod}}` is ``frequency`` (Hz),
    - :math:`\phi` is ``phase`` (deg, internally converted to radians),
    - :math:`t` is simulation time in ms.

    For simulation resolution :math:`\Delta t` in ms, each output train
    samples a Poisson multiplicity

    .. math::

       K_n \sim \mathrm{Poisson}(\lambda_n), \qquad
       \lambda_n = f_n \Delta t / 1000,

    where the ``1000`` factor converts Hz * ms to a dimensionless mean.
    ``K_n`` is an integer count ``0, 1, 2, ...`` and may exceed ``1``.

    **2. Oscillator-state recurrence and derivation**

    Following NEST, sinusoidal modulation is stored in a rotated two-component
    oscillator state:

    .. math::

       y_0(t) = a/1000 \cdot \cos(\omega t + \phi), \qquad
       y_1(t) = a/1000 \cdot \sin(\omega t + \phi),

    with :math:`\omega = 2\pi f_{\mathrm{mod}}/1000` (rad/ms). One-step
    propagation by :math:`\Delta t` uses a rotation matrix
    :math:`R(\omega\Delta t)`:

    .. math::

       \begin{bmatrix}
       y_0' \\
       y_1'
       \end{bmatrix}
       =
       \begin{bmatrix}
       \cos(\omega\Delta t) & -\sin(\omega\Delta t) \\
       \sin(\omega\Delta t) &  \cos(\omega\Delta t)
       \end{bmatrix}
       \begin{bmatrix}
       y_0 \\
       y_1
       \end{bmatrix}.

    The post-rotation ``y_1'`` is then added to ``rate/1000`` and clamped at
    zero before Poisson sampling. This avoids recomputing trigonometric
    functions each step and keeps per-step modulation update constant-time.

    **3. Update ordering (NEST source order)**

    The internal two-component oscillator state is updated exactly in the
    order used by NEST ``models/sinusoidal_poisson_generator.cpp``:

    1. Start from the DC component ``rate``.
    2. Rotate oscillator state ``(y_0, y_1)`` by one step.
    3. Add the rotated ``y_1`` to obtain instantaneous rate.
    4. Clamp rate at zero.
    5. Sample Poisson multiplicities if active.

    The per-step recorded ``rate`` value in NEST corresponds to this updated
    post-rotation rate. This implementation exposes it via
    :meth:`get_recorded_rate`.

    **4. Timing semantics**

    NEST currently classifies this model as ``CURRENT_GENERATOR`` in
    ``get_type()``. Consequently, activity is evaluated with a two-step shift
    in ``StimulationDevice::is_active``:

    .. math::

       t_{\min} < (n + 2) \le t_{\max},

    where ``n`` is current simulation step index and
    ``t_{\min} = \mathrm{origin} + \mathrm{start}``,
    ``t_{\max} = \mathrm{origin} + \mathrm{stop}`` (in steps).

    This differs from regular spike generators and is intentionally replicated
    here to match NEST behavior.

    **5. Assumptions, constraints, and computational implications**

    - Public parameters are scalar-only; non-scalar values raise
      :class:`ValueError`.
    - ``stop`` must satisfy ``stop >= start`` after unit conversion.
    - When ``dt`` is available, finite ``origin``, ``start``, and ``stop``
      must be representable on the simulation grid.
    - If ``dt`` changes, timing caches and oscillator state are recomputed from
      absolute simulation time to preserve NEST-compatible behavior.
    - Per-step complexity is :math:`O(\prod \mathrm{varshape})` for Poisson
      sampling and :math:`O(1)` for oscillator/timing updates.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification for :class:`brainstate.nn.Dynamics`.
        The derived ``self.varshape`` is the shape of values returned by
        :meth:`update`; each element corresponds to one emitted train.
        Default is ``1``.
    rate : ArrayLike, optional
        Scalar baseline firing rate in spikes/s (Hz), shape ``()`` after
        conversion. Accepted inputs include scalar ``ArrayLike`` and
        :class:`brainunit.Quantity` convertible to ``u.Hz``.
        Default is ``0.0 * u.Hz``.
    amplitude : ArrayLike, optional
        Scalar sinusoidal modulation amplitude in spikes/s (Hz), shape ``()``
        after conversion. Units and conversion rules match ``rate``.
        Default is ``0.0 * u.Hz``.
    frequency : ArrayLike, optional
        Scalar modulation frequency in Hz, shape ``()`` after conversion.
        Internally converted to angular frequency in rad/ms.
        Default is ``0.0 * u.Hz``.
    phase : ArrayLike, optional
        Scalar modulation phase in degrees, shape ``()`` after conversion.
        Internally converted to radians.
        Default is ``0.0``.
    individual_spike_trains : bool, optional
        Sampling mode selector.
        If ``True``, Poisson sampling is independent for each index of
        ``self.varshape``.
        If ``False``, one sampled multiplicity is broadcast to all outputs.
        Default is ``True``.
    start : ArrayLike, optional
        Scalar relative activation start time in ms, shape ``()`` after
        conversion. Activity uses NEST current-generator semantics with a
        two-step shifted check. Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative deactivation stop time in ms, shape ``()`` after
        conversion. ``None`` maps to ``+inf``.
        Must satisfy ``stop >= start`` after conversion.
        Default is ``None``.
    origin : ArrayLike, optional
        Scalar global time offset in ms, shape ``()`` after conversion.
        Added to ``start`` and ``stop`` for activity-window bounds.
        Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed used to initialize ``jax.random.PRNGKey`` in :meth:`init_state`
        and lazy initialization in :meth:`update`. Default is ``0``.
    name : str, optional
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 24 18 20 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``rate``
         - ``0.0 * u.Hz``
         - :math:`r`
         - Baseline firing rate in spikes/s.
       * - ``amplitude``
         - ``0.0 * u.Hz``
         - :math:`a`
         - Sinusoidal modulation amplitude in spikes/s.
       * - ``frequency``
         - ``0.0 * u.Hz``
         - :math:`f_{\mathrm{mod}}`
         - Modulation frequency in Hz.
       * - ``phase``
         - ``0.0``
         - :math:`\phi`
         - Modulation phase in degrees (internally radians).
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative lower activity bound (NEST shifted semantics).
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative upper activity bound; ``None`` maps to ``+\infty``.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global time offset applied to start/stop.
       * - ``in_size``
         - ``1``
         - -
         - Defines output train count/shape via ``self.varshape``.
       * - ``individual_spike_trains``
         - ``True``
         - -
         - Independent-per-output sampling vs shared broadcast sample.
       * - ``rng_seed``
         - ``0``
         - -
         - Seed for JAX random key evolution.

    Raises
    ------
    ValueError
        If any scalar-constrained parameter cannot be reduced to one value; if
        ``stop < start``; or if finite ``origin``/``start``/``stop`` are not
        representable on the simulation grid when ``dt`` is available.
    TypeError
        If numeric/unit conversion fails for provided rate/time inputs.
    KeyError
        At runtime, if required simulation context keys (for example ``dt`` in
        :meth:`update`) are unavailable through ``brainstate.environ``.

    Notes
    -----
    - Time parameters are validated on the simulation grid when ``dt`` is
      available, matching repository conventions used by other NEST-compatible
      generators.
    - The oscillator state is re-initialized from absolute simulation time
      whenever the simulation resolution changes, matching NEST pre-run
      calibration behavior.
    - Recorded rate from :meth:`get_recorded_rate` is the post-rotation,
      post-clamp value in spikes/s used for current-step sampling logic.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.sinusoidal_poisson_generator(
       ...         in_size=4,
       ...         rate=800.0 * u.Hz,
       ...         amplitude=200.0 * u.Hz,
       ...         frequency=10.0 * u.Hz,
       ...         phase=90.0,
       ...         start=5.0 * u.ms,
       ...         stop=50.0 * u.ms,
       ...         rng_seed=123,
       ...     )
       ...     with brainstate.environ.context(t=10.0 * u.ms):
       ...         counts = gen.update()
       ...     _ = counts.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.sinusoidal_poisson_generator(
       ...         individual_spike_trains=False
       ...     )
       ...     gen.set(rate=500.0 * u.Hz, amplitude=300.0 * u.Hz, phase=45.0)
       ...     params = gen.get()
       ...     _ = params['rate'], params['amplitude']

    See Also
    --------
    poisson_generator : Homogeneous Poisson generator.
    inhomogeneous_poisson_generator : Piecewise-constant inhomogeneous Poisson generator.
    sinusoidal_gamma_generator : Sinusoidally modulated gamma-renewal generator.

    References
    ----------
    .. [1] NEST source:
           ``models/sinusoidal_poisson_generator.h`` and
           ``models/sinusoidal_poisson_generator.cpp``.
    .. [2] NEST source:
           ``nestkernel/stimulation_device.cpp``.
    .. [3] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/sinusoidal_poisson_generator.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        amplitude: ArrayLike = 0. * u.Hz,
        frequency: ArrayLike = 0. * u.Hz,
        phase: ArrayLike = 0.0,
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
        self.individual_spike_trains = bool(individual_spike_trains)

        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        if self.stop < self.start:
            raise ValueError('stop >= start required.')

        self._rate_per_ms = self.rate / 1000.0
        self._amplitude_per_ms = self.amplitude / 1000.0
        self._om_rad_per_ms = self.frequency * (2.0 * math.pi / 1000.0)
        self._phi_rad = self.phase * (math.pi / 180.0)

        self._dt_cache_ms = np.nan
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max
        self._sin_step = 0.0
        self._cos_step = 1.0

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)
            self._refresh_step_rotation_cache(dt_ms)

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

    def _refresh_step_rotation_cache(self, dt_ms: float):
        self._sin_step = math.sin(dt_ms * self._om_rad_per_ms)
        self._cos_step = math.cos(dt_ms * self._om_rad_per_ms)

    def _reset_oscillator_state(self, t_ms: float):
        y0 = self._amplitude_per_ms * math.cos(self._om_rad_per_ms * t_ms + self._phi_rad)
        y1 = self._amplitude_per_ms * math.sin(self._om_rad_per_ms * t_ms + self._phi_rad)
        self.y_0.value = jnp.asarray(y0, dtype=jnp.float64)
        self.y_1.value = jnp.asarray(y1, dtype=jnp.float64)
        self._recorded_rate_hz.value = jnp.asarray(0.0, dtype=jnp.float64)

    def _is_active(self, curr_step: int) -> bool:
        # Match NEST's current-generator activity handling for this model:
        # StimulationDevice::is_active uses step+2 for CURRENT_GENERATOR.
        shifted_step = curr_step + 2
        return (self._t_min_step < shifted_step) and (shifted_step <= self._t_max_step)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize RNG, oscillator states, and cached recorded rate.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused placeholder for :class:`brainstate.nn.Dynamics`
            compatibility.
        **kwargs
            Unused extra keyword arguments.


        """
        del batch_size, kwargs
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))
        self.y_0 = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))
        self.y_1 = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))
        self._recorded_rate_hz = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)
            self._refresh_step_rotation_cache(dt_ms)
            self._reset_oscillator_state(self._current_time_ms())

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        amplitude: ArrayLike | object = _UNSET,
        frequency: ArrayLike | object = _UNSET,
        phase: ArrayLike | object = _UNSET,
        individual_spike_trains: bool | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        r"""Set public parameters and refresh dependent cached state.

        Parameters
        ----------
        rate : ArrayLike or object, optional
            Scalar rate in spikes/s (Hz). ``_UNSET`` keeps current value.
        amplitude : ArrayLike or object, optional
            Scalar sinusoidal amplitude in spikes/s (Hz). ``_UNSET`` keeps
            current value.
        frequency : ArrayLike or object, optional
            Scalar frequency in Hz. ``_UNSET`` keeps current value.
        phase : ArrayLike or object, optional
            Scalar phase in degrees. ``_UNSET`` keeps current value.
        individual_spike_trains : bool or object, optional
            Sampling mode flag. ``_UNSET`` keeps current value.
        start : ArrayLike or object, optional
            Scalar relative start time in ms. ``_UNSET`` keeps current value.
        stop : ArrayLike, None, or object, optional
            Scalar relative stop time in ms, or ``None`` for ``+inf``.
            ``_UNSET`` keeps current value.
        origin : ArrayLike or object, optional
            Scalar origin time in ms. ``_UNSET`` keeps current value.

        Raises
        ------
        ValueError
            If scalar conversion fails, ``stop < start``, or grid-time
            validation fails when ``dt`` is available.
        TypeError
            If unit or numeric conversion fails for supplied inputs.
        """
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_amplitude = (
            self.amplitude if amplitude is _UNSET else self._to_scalar_rate_hz(amplitude)
        )
        new_frequency = (
            self.frequency if frequency is _UNSET else self._to_scalar_rate_hz(frequency)
        )
        new_phase = self.phase if phase is _UNSET else self._to_scalar_float(phase, name='phase')
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

        if new_stop < new_start:
            raise ValueError('stop >= start required.')

        self.rate = new_rate
        self.amplitude = new_amplitude
        self.frequency = new_frequency
        self.phase = new_phase
        self.individual_spike_trains = new_individual
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        self._rate_per_ms = self.rate / 1000.0
        self._amplitude_per_ms = self.amplitude / 1000.0
        self._om_rad_per_ms = self.frequency * (2.0 * math.pi / 1000.0)
        self._phi_rad = self.phase * (math.pi / 180.0)

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)
            self._refresh_step_rotation_cache(dt_ms)
            if hasattr(self, 'y_0'):
                self._reset_oscillator_state(self._current_time_ms())

    def get(self) -> dict:
        r"""Return current public parameters and oscillator state snapshot.

        Returns
        -------
        dict
            Dictionary with keys ``rate``, ``frequency``, ``phase``,
            ``amplitude``, ``individual_spike_trains``, ``start``, ``stop``,
            ``origin``, ``y_0``, and ``y_1``. Rates are in spikes/s, times are
            in ms, and oscillator states are in spikes/ms.
        """
        y0 = 0.0
        y1 = 0.0
        if hasattr(self, 'y_0'):
            y0 = float(np.asarray(self.y_0.value, dtype=np.float64).reshape(()))
            y1 = float(np.asarray(self.y_1.value, dtype=np.float64).reshape(()))

        return {
            'rate': float(self.rate),
            'frequency': float(self.frequency),
            'phase': float(self.phase),
            'amplitude': float(self.amplitude),
            'individual_spike_trains': bool(self.individual_spike_trains),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
            'y_0': y0,
            'y_1': y1,
        }

    def get_recorded_rate(self) -> float:
        r"""Return latest post-update instantaneous rate in spikes/s.

        Returns
        -------
        float
            Most recent stored value of the post-rotation, post-clamp
            instantaneous rate. Returns ``0.0`` before state initialization.
        """
        if not hasattr(self, '_recorded_rate_hz'):
            return 0.0
        return float(np.asarray(self._recorded_rate_hz.value, dtype=np.float64).reshape(()))

    def _sample_poisson_individual(self, lam: float) -> jax.Array:
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.poisson(
            subkey,
            lam=jnp.asarray(lam, dtype=jnp.float64),
            shape=self.varshape,
        ).astype(jnp.int64)

    def _sample_poisson_shared(self, lam: float) -> int:
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        sample = jax.random.poisson(
            subkey,
            lam=jnp.asarray(lam, dtype=jnp.float64),
            shape=(),
        ).astype(jnp.int64)
        return int(np.asarray(sample, dtype=np.int64).reshape(()))

    def update(self):
        r"""Advance generator by one simulation step and emit spike counts.

        Returns
        -------
        jax.Array
            ``int64`` array with shape ``self.varshape``. Values are per-step
            spike multiplicities sampled from the configured sinusoidal Poisson
            process, or zeros when inactive/non-positive-rate.

        Raises
        ------
        KeyError
            If required environment entries (for example ``dt``) are not
            available through ``brainstate.environ`` at runtime.
        ValueError
            If cached timing constraints become invalid after environment
            changes (for example non-grid-aligned finite time bounds).
        """
        if not hasattr(self, 'rng_key'):
            self.init_state()

        dt_ms = self._dt_ms()
        curr_t_ms = self._current_time_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_timing_cache(dt_ms)
            self._refresh_step_rotation_cache(dt_ms)
            self._reset_oscillator_state(curr_t_ms)

        curr_step = self._time_to_step(curr_t_ms, dt_ms)

        # Update oscillator blocks in NEST ordering.
        y0 = float(np.asarray(self.y_0.value, dtype=np.float64).reshape(()))
        y1 = float(np.asarray(self.y_1.value, dtype=np.float64).reshape(()))

        rate_per_ms = self._rate_per_ms
        new_y0 = self._cos_step * y0 - self._sin_step * y1
        y1 = self._sin_step * y0 + self._cos_step * y1
        y0 = new_y0
        rate_per_ms += y1
        if rate_per_ms < 0.0:
            rate_per_ms = 0.0

        self.y_0.value = jnp.asarray(y0, dtype=jnp.float64)
        self.y_1.value = jnp.asarray(y1, dtype=jnp.float64)
        self._recorded_rate_hz.value = jnp.asarray(rate_per_ms * 1000.0, dtype=jnp.float64)

        if rate_per_ms > 0.0 and self._is_active(curr_step):
            lam = rate_per_ms * dt_ms
            if self.individual_spike_trains:
                return self._sample_poisson_individual(lam)

            n_spikes = self._sample_poisson_shared(lam)
            return jnp.full(self.varshape, n_spikes, dtype=jnp.int64)

        return jnp.zeros(self.varshape, dtype=jnp.int64)
