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
import brainunit as u
import jax
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base.base import NESTDevice

__all__ = [
    'poisson_generator',
]

_UNSET = object()


class poisson_generator(NESTDevice):
    r"""Poisson spike generator compatible with NEST.

    Description
    -----------
    ``poisson_generator`` re-implements NEST's stimulation device of the same
    name and emits per-step spike multiplicities.

    **1. Point-process model and discretization**

    Let ``r`` be the configured homogeneous rate in spikes/s and
    :math:`\Delta t` be the simulation step in ms. For one output train, the
    count in one discrete bin is sampled as

    .. math::

       K_n \sim \mathrm{Poisson}(\lambda_n), \qquad
       \lambda_n = r \, \Delta t / 1000.

    The factor ``1000`` converts milliseconds to seconds, so
    :math:`\lambda_n` is dimensionless. This is the standard bin-count
    reduction of a homogeneous Poisson process where
    :math:`\mathbb{P}(K_n=k)=e^{-\lambda_n}\lambda_n^k/k!`.

    Implementation detail: :meth:`update` draws one vectorized Poisson sample
    with ``shape=self.varshape`` via ``jax.random.poisson``. Each element is an
    independent train; values are integer multiplicities ``0, 1, 2, ...`` and
    are not clipped to binary spikes.

    **2. Activity window and NEST timing semantics**

    The active interval follows NEST ``StimulationDevice::is_active`` for spike
    generators:

    .. math::

       t_{\min} < t \le t_{\max}, \qquad
       t_{\min} = origin + start,\quad t_{\max} = origin + stop.

    Therefore ``start`` is exclusive and ``stop`` is inclusive.
    Internally, times are projected to integer steps with
    ``round(time_ms / dt_ms)`` and activity is evaluated as
    ``t_min_step < curr_step <= t_max_step``.

    **3. Assumptions, constraints, and computational implications**

    Scalar parameters are converted to ``float64`` in public units (Hz or ms).
    If ``dt`` is available, finite ``origin``, ``start``, and ``stop`` must lie
    on the simulation grid (absolute tolerance ``1e-12`` in ``time/dt`` ratio).
    Cache refresh is triggered when ``dt`` changes. Per-step runtime is
    :math:`O(\prod \text{varshape})` for sampling and memory is proportional to
    output size. When ``rate <= 0`` or inactive, the update path returns a
    zero ``int64`` array without Poisson sampling.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification for :class:`brainstate.nn.Dynamics`.
        The derived ``self.varshape`` is the exact shape of arrays returned by
        :meth:`update`. Each element corresponds to one independent output
        train. Default is ``1``.
    rate : ArrayLike, optional
        Scalar firing rate in spikes/s (Hz). Accepted forms are any
        ``ArrayLike`` with exactly one element, optionally a
        :class:`brainunit.Quantity` convertible to ``u.Hz``.
        Must satisfy ``rate >= 0``. Default is ``0.0 * u.Hz``.
    start : ArrayLike, optional
        Scalar relative start time in ms (exclusive lower bound after adding
        ``origin``). Must be scalar-convertible to ``float64`` and, when
        ``dt`` is available, grid representable. Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative stop time in ms (inclusive upper bound after adding
        ``origin``). ``None`` is mapped to ``+inf``. If finite, must be
        scalar-convertible and grid representable when ``dt`` is available.
        Must satisfy ``stop >= start`` after conversion. Default is ``None``.
    origin : ArrayLike, optional
        Scalar time origin offset in ms added to both ``start`` and ``stop``.
        Must be scalar-convertible and grid representable when ``dt`` is
        available. Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed used to initialize ``jax.random.PRNGKey`` inside
        :meth:`init_state`. Different seeds lead to different stochastic
        realizations for otherwise identical parameters. Default is ``0``.
    name : str or None, optional
        Optional dynamics node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 22 18 18 42

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``rate``
         - ``0.0 * u.Hz``
         - :math:`r`
         - Homogeneous firing rate in spikes/s.
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower bound of activity.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound; ``None`` maps to ``+\infty``.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global offset added to ``start`` and ``stop``.
       * - ``in_size``
         - ``1``
         - -
         - Defines ``self.varshape`` (number/shape of independent trains).
       * - ``rng_seed``
         - ``0``
         - -
         - Seed for JAX key state used by Poisson sampling.

    Raises
    ------
    ValueError
        If ``rate < 0``; if ``stop < start``; if time/rate inputs are not
        scalar-convertible; or if finite ``origin``/``start``/``stop`` are not
        multiples of simulation resolution when ``dt`` is available.
    TypeError
        If unit conversion to ``u.Hz`` or ``u.ms`` fails for supplied inputs.
    KeyError
        At runtime, if required simulation context entries (for example ``dt``
        via ``brainstate.environ.get_dt()``) are missing.

    Notes
    -----
    - ``update`` lazily initializes RNG state if :meth:`init_state` has not
      been called explicitly.
    - Parameter updates through :meth:`set` recompute cached step bounds when
      ``dt`` is present in the environment.
    - As in NEST, one generator can fan out to many targets while maintaining
      independent trains per output element.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.poisson_generator(
       ...         in_size=(2, 3),
       ...         rate=1200.0 * u.Hz,
       ...         start=5.0 * u.ms,
       ...         stop=20.0 * u.ms,
       ...         rng_seed=11,
       ...     )
       ...     with brainstate.environ.context(t=10.0 * u.ms):
       ...         counts = gen.update()
       ...     _ = counts.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> gen = brainpy.state.poisson_generator(rate=500.0 * u.Hz)
       >>> gen.set(start=2.0 * u.ms, stop=None, origin=1.0 * u.ms)
       >>> params = gen.get()
       >>> _ = params['rate'], params['stop']

    See Also
    --------
    poisson_generator_ps : Precise-time Poisson generator with dead time.
    inhomogeneous_poisson_generator : Piecewise-constant time-varying Poisson rate.
    sinusoidal_poisson_generator : Sinusoidally modulated Poisson rate.

    References
    ----------
    .. [1] NEST source: ``models/poisson_generator.cpp`` and
           ``models/poisson_generator.h``.
    .. [2] NEST source: ``nestkernel/stimulation_device.h`` and
           ``nestkernel/stimulation_device.cpp``.
    .. [3] NEST model docs:
           https://nest-simulator.readthedocs.io/en/stable/models/poisson_generator.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        if self.rate < 0.0:
            raise ValueError('The rate cannot be negative.')
        if self.stop < self.start:
            raise ValueError('stop >= start required.')

        self._dt_cache_ms = np.nan
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max
        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    @staticmethod
    def _to_scalar_time_ms(value: ArrayLike) -> float:
        dftype = brainstate.environ.dftype()
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.ms), dtype=dftype)
        else:
            arr = np.asarray(u.math.asarray(value), dtype=dftype)
        if arr.size != 1:
            raise ValueError('Time parameters must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_rate_hz(value: ArrayLike) -> float:
        dftype = brainstate.environ.dftype()
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.Hz), dtype=dftype)
        else:
            arr = np.asarray(u.math.asarray(value), dtype=dftype)
        if arr.size != 1:
            raise ValueError('rate must be scalar.')
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

    def _is_active(self, curr_step: int) -> bool:
        return (self._t_min_step < curr_step) and (curr_step <= self._t_max_step)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize the RNG state used by Poisson sampling.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused. Present for framework API compatibility with
            :class:`brainstate.nn.Dynamics`. Default is ``None``.
        **kwargs : Any
            Unused keyword arguments accepted for API compatibility.


        Notes
        -----
        :meth:`update` lazily calls this method on the first step if
        ``init_state`` has not been invoked explicitly. Calling ``init_state``
        resets the RNG to the original seed, so repeated calls restart the
        stochastic sequence from the beginning.

        See Also
        --------
        poisson_generator.update : Consumes ``rng_key`` populated here.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate
           >>> import brainunit as u
           >>> from brainpy.state import poisson_generator
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     gen = poisson_generator(in_size=4, rate=800.0 * u.Hz, rng_seed=7)
           ...     gen.init_state()
        """
        del batch_size, kwargs
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        r"""Update public parameters and refresh the timing cache when needed.

        Only keyword arguments that are explicitly passed are modified; omitted
        arguments retain their current values.

        Parameters
        ----------
        rate : ArrayLike or object, optional
            New scalar firing rate in spikes/s (Hz). Accepts any
            ``ArrayLike`` with exactly one element, or a
            :class:`brainunit.Quantity` convertible to ``u.Hz``.
            Must satisfy ``rate >= 0`` after conversion. Omit to keep the
            current value.
        start : ArrayLike or object, optional
            New scalar relative start time in ms (exclusive lower bound after
            adding ``origin``). Must be scalar-convertible and, when ``dt`` is
            in the environment, grid-representable. Omit to keep the current
            value.
        stop : ArrayLike or None or object, optional
            New scalar relative stop time in ms (inclusive upper bound after
            adding ``origin``). ``None`` maps to ``+inf``. Must satisfy
            ``stop >= start`` after conversion. Omit to keep the current value.
        origin : ArrayLike or object, optional
            New scalar time origin offset in ms added to both ``start`` and
            ``stop``. Must be scalar-convertible and grid-representable when
            ``dt`` is available. Omit to keep the current value.


        Raises
        ------
        ValueError
            If ``rate < 0`` after conversion; if ``stop < start`` after
            conversion; or if any finite timing parameter is not representable
            on the current simulation grid (checked via
            :meth:`_assert_grid_time`).
        TypeError
            If unit conversion to ``u.Hz`` or ``u.ms`` fails for any supplied
            value.

        See Also
        --------
        poisson_generator.get : Read-back current parameter values.

        Examples
        --------
        .. code-block:: python

           >>> import brainpy
           >>> import brainunit as u
           >>> gen = brainpy.state.poisson_generator(rate=500.0 * u.Hz)
           >>> gen.set(rate=1000.0 * u.Hz, stop=50.0 * u.ms)
           >>> params = gen.get()
           >>> _ = params['rate'], params['stop']
        """
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_start = self.start if start is _UNSET else self._to_scalar_time_ms(start)
        if stop is _UNSET:
            new_stop = self.stop
        elif stop is None:
            new_stop = np.inf
        else:
            new_stop = self._to_scalar_time_ms(stop)
        new_origin = self.origin if origin is _UNSET else self._to_scalar_time_ms(origin)

        if new_rate < 0.0:
            raise ValueError('The rate cannot be negative.')
        if new_stop < new_start:
            raise ValueError('stop >= start required.')

        self.rate = new_rate
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    def get(self) -> dict:
        r"""Return current public parameters as scalar SI-compatible values.

        Returns
        -------
        params : dict
            Dictionary with four ``float`` entries:

            - ``'rate'`` -- firing rate in spikes/s (Hz).
            - ``'start'`` -- relative exclusive lower bound in ms.
            - ``'stop'`` -- relative inclusive upper bound in ms; ``inf``
              when no deactivation time has been set.
            - ``'origin'`` -- time origin offset in ms.

        Notes
        -----
        Returned values are plain Python ``float`` scalars (``float64``
        precision). They mirror the internal scalar attributes set in
        :meth:`__init__` or updated by :meth:`set` and are not bound to any
        ``brainunit`` quantities.

        See Also
        --------
        poisson_generator.set : Update one or more parameters in place.

        Examples
        --------
        .. code-block:: python

           >>> import brainpy
           >>> import brainunit as u
           >>> gen = brainpy.state.poisson_generator(
           ...     rate=800.0 * u.Hz,
           ...     start=5.0 * u.ms,
           ...     stop=100.0 * u.ms,
           ...     origin=2.0 * u.ms,
           ... )
           >>> params = gen.get()
           >>> params['rate']
           800.0
           >>> params['stop']
           100.0
        """
        return {
            'rate': float(self.rate),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_poisson(self, lam: float) -> jax.Array:
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.poisson(
            subkey,
            lam=lam,
            shape=self.varshape,
        ).astype(np.int64)

    def update(self):
        r"""Advance one simulation step and return per-step spike multiplicities.

        Returns
        -------
        spikes : jax.Array
            Integer array with dtype ``int64`` and shape ``self.varshape``.
            Each element is the number of spikes emitted by the corresponding
            independent output train in the current time step.

            - **Active and** ``rate > 0``: entries are i.i.d.
              Poisson(:math:`\lambda`) samples with
              :math:`\lambda = r \cdot \Delta t / 1000`.
            - **Inactive or** ``rate <= 0``: all entries are exactly ``0``.

        Raises
        ------
        ValueError
            If the timing cache is stale and a finite ``origin``, ``start``,
            or ``stop`` is not representable on the current simulation grid
            (checked by :meth:`_assert_grid_time`).
        KeyError
            If ``dt`` is unavailable from ``brainstate.environ.get_dt()`` or
            ``t`` is expected but cannot be resolved.

        Notes
        -----
        The update proceeds as follows each call:

        1. **Lazy init** -- If ``rng_key`` has not been created by
           :meth:`init_state`, it is initialized automatically with
           ``self.rng_seed``.
        2. **Cache refresh** -- When ``dt`` changes from the previously cached
           value, :meth:`_refresh_timing_cache` recomputes the integer step
           bounds :math:`t_{\min}` and :math:`t_{\max}`.
        3. **Rate guard** -- If ``rate <= 0``, an all-zero array is returned
           without touching the PRNG state.
        4. **Activity check** -- The current step index is compared against
           the cached step bounds: active iff
           :math:`t_{\min,\mathrm{step}} < \mathrm{curr\_step} \le
           t_{\max,\mathrm{step}}`. Inactive steps return zeros.
        5. **Poisson draw** -- If active, one vectorized sample
           ``jax.random.poisson(lam, shape=self.varshape)`` is drawn via
           :meth:`_sample_poisson`, consuming one PRNG split.

        See Also
        --------
        poisson_generator.init_state : RNG initialization called lazily here.
        poisson_generator.set : Update parameters between runs.
        """
        if not hasattr(self, 'rng_key'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
        not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)):
            self._refresh_timing_cache(dt_ms)

        ditype = brainstate.environ.ditype()

        if self.rate <= 0.0:
            return jax.numpy.zeros(self.varshape, dtype=ditype)

        # JAX-compatible activity check (works under jit / for_loop tracing).
        # t may be a traced abstract value inside for_loop, so we avoid float().
        t = brainstate.environ.get('t', default=0. * u.ms)
        if isinstance(t, u.Quantity):
            t_ms_num = t.to_decimal(u.ms)
        else:
            t_ms_num = jax.numpy.asarray(t)
        curr_step = jax.numpy.rint(t_ms_num / dt_ms).astype(jax.numpy.int64)
        is_active = (self._t_min_step < curr_step) & (curr_step <= self._t_max_step)

        lam = self.rate * dt_ms / 1000.0
        spikes = self._sample_poisson(lam)
        return jax.numpy.where(is_active, spikes, jax.numpy.zeros(self.varshape, dtype=ditype))
