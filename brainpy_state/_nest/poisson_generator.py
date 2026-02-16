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
    'poisson_generator',
]


_UNSET = object()


class poisson_generator(brainstate.nn.Dynamics):
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

    Returns
    -------
    out : Any
        Dynamics node instance. Each :meth:`update` call returns a JAX array
        of dtype ``int64`` and shape ``self.varshape`` containing per-step
        spike multiplicities.

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
        """Initialize RNG state used by Poisson sampling.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused. Present for framework API compatibility.
        **kwargs
            Unused keyword arguments for API compatibility.

        Returns
        -------
        out : Any
            ``None``. Side effect: creates ``rng_key`` as a
            :class:`brainstate.ShortTermState` wrapping ``jax.random.PRNGKey``.
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
        """Update public parameters and refresh timing cache when needed.

        Parameters
        ----------
        rate : ArrayLike or object, optional
            New scalar rate in spikes/s (Hz). Use ``_UNSET`` to keep current
            value. Must be non-negative after conversion.
        start : ArrayLike or object, optional
            New scalar relative start in ms. Use ``_UNSET`` to keep current
            value.
        stop : ArrayLike or None or object, optional
            New scalar relative stop in ms. ``None`` maps to ``+inf``.
            Use ``_UNSET`` to keep current value.
        origin : ArrayLike or object, optional
            New scalar origin offset in ms. Use ``_UNSET`` to keep current
            value.

        Returns
        -------
        out : Any
            ``None``. Side effect: updates ``rate``, ``start``, ``stop``,
            ``origin`` and recalculates cached step bounds when ``dt`` is
            available in ``brainstate.environ``.

        Raises
        ------
        ValueError
            If ``rate < 0``; if ``stop < start`` after conversion; or if
            finite timing parameters are off-grid for the current ``dt``.
        TypeError
            If unit conversion to Hz/ms fails for supplied values.
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
        """Return current public parameters in scalar SI-compatible values.

        Parameters
        ----------
        None

        Returns
        -------
        out : Any
            ``dict`` with keys ``'rate'``, ``'start'``, ``'stop'``, and
            ``'origin'``. Values are Python ``float`` in Hz/ms public units;
            ``stop`` is ``inf`` when deactivation is disabled.
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
            lam=jnp.asarray(lam, dtype=jnp.float64),
            shape=self.varshape,
        ).astype(jnp.int64)

    def update(self):
        """Advance one simulation step and return spike multiplicities.

        Parameters
        ----------
        None

        Returns
        -------
        out : Any
            ``jax.Array`` with dtype ``int64`` and shape ``self.varshape``.
            When active and ``rate > 0``, entries are Poisson-distributed
            counts with mean ``rate * dt_ms / 1000``; otherwise all zeros.

        Raises
        ------
        ValueError
            If cached timing is refreshed and finite time parameters are not
            representable on the current simulation grid.
        KeyError
            If required simulation context values (notably ``dt``) are
            unavailable via ``brainstate.environ``.
        """
        if not hasattr(self, 'rng_key'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)):
            self._refresh_timing_cache(dt_ms)

        if self.rate <= 0.0:
            return jnp.zeros(self.varshape, dtype=jnp.int64)

        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)
        if self._is_active(curr_step):
            lam = self.rate * dt_ms / 1000.0
            return self._sample_poisson(lam)
        return jnp.zeros(self.varshape, dtype=jnp.int64)
