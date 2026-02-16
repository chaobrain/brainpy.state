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
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'mip_generator',
]


_UNSET = object()


class mip_generator(brainstate.nn.Dynamics):
    r"""Correlated spike trains from a Multiple Interaction Process (MIP).

    Description
    -----------
    ``mip_generator`` reproduces NEST's ``mip_generator`` device by combining
    one shared parent Poisson process with independent copy operations for
    each child output train.

    **1. Parent-child process model and derivation**

    Let ``rate = r`` in spikes/s and simulation step ``dt = \Delta t`` in ms.
    For each step ``n``:

    .. math::

       N_n \sim \mathrm{Poisson}(\lambda), \qquad
       \lambda = r \, \Delta t / 1000.

    For each child train :math:`i \in \{1,\dots,M\}` and each parent spike
    :math:`m \in \{1,\dots,N_n\}`, draw
    :math:`B_{i,m} \sim \mathrm{Bernoulli}(p_{\mathrm{copy}})` independently
    across ``i`` and ``m``. The emitted multiplicity is

    .. math::

       K_{i,n} = \sum_{m=1}^{N_n} B_{i,m}.

    Marginally, :math:`K_{i,n}` is Poisson with parameter
    :math:`p_{\mathrm{copy}}\lambda` (Poisson thinning), so each child has mean
    rate :math:`p_{\mathrm{copy}} r`. Shared parent fluctuations induce
    cross-child covariance:

    .. math::

       \mathrm{Cov}(K_{i,n}, K_{j,n}) = p_{\mathrm{copy}}^2 \lambda,\quad
       \mathrm{Var}(K_{i,n}) = p_{\mathrm{copy}}\lambda,\quad
       \rho_{ij} = p_{\mathrm{copy}} \ \ (i \neq j).

    **2. Source-equivalent sampling order and computational implications**

    The update path mirrors ``models/mip_generator.cpp``:

    1. Check whether the stimulation device is active at current step.
    2. Draw parent multiplicity from the parent Poisson process.
    3. For each output train, run explicit Bernoulli trials for each parent
       spike and count copied spikes.

    This implementation intentionally preserves NEST's explicit Bernoulli loop
    (instead of vectorized Binomial sampling). Runtime per active step is
    :math:`O(M N_n)` random comparisons in the general case, with fast paths
    for ``p_copy <= 0`` and ``p_copy >= 1``.

    **3. Timing semantics and grid constraints**

    Activity follows NEST stimulation-device semantics:

    .. math::

       t_{\min} < t \le t_{\max}, \qquad
       t_{\min} = \mathrm{origin} + \mathrm{start},\quad
       t_{\max} = \mathrm{origin} + \mathrm{stop}.

    Therefore ``start`` is exclusive and ``stop`` is inclusive. Internally,
    finite times are mapped to integer steps with
    :math:`\mathrm{round}(t / \Delta t)`. Finite ``origin``, ``start``, and
    ``stop`` must be on the simulation grid (absolute tolerance ``1e-12`` in
    ``time/dt`` ratio), otherwise :class:`ValueError` is raised.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification for :class:`brainstate.nn.Dynamics`.
        ``self.varshape`` is derived from ``in_size`` and determines the exact
        shape of arrays emitted by :meth:`update`. Each element of
        ``self.varshape`` corresponds to one child process. Default is ``1``.
    rate : ArrayLike, optional
        Scalar parent rate :math:`r` in spikes/s (Hz). Must be an
        ``ArrayLike`` with exactly one element, optionally a
        :class:`brainunit.Quantity` convertible to ``u.Hz``.
        Constraint: ``rate >= 0``. Default is ``0.0 * u.Hz``.
    p_copy : ArrayLike, optional
        Scalar copy probability :math:`p_{\mathrm{copy}}` for each parent spike
        and each child process. Must be scalar-convertible to ``float64`` and
        satisfy ``0 <= p_copy <= 1``. Default is ``1.0``.
    start : ArrayLike, optional
        Scalar relative start time in ms (exclusive lower bound after adding
        ``origin``). Must be scalar-convertible to ``float64`` and, when
        ``dt`` is available, representable on the simulation grid.
        Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative stop time in ms (inclusive upper bound after adding
        ``origin``). ``None`` maps to ``+inf``. If finite, must be
        scalar-convertible and grid-representable when ``dt`` is available.
        Must satisfy ``stop >= start`` after conversion. Default is ``None``.
    origin : ArrayLike, optional
        Scalar time offset in ms added to both ``start`` and ``stop``.
        Must be scalar-convertible and grid-representable when ``dt`` is
        available. Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed passed to :class:`numpy.random.SeedSequence` and split into two
        independent RNG streams (parent Poisson and child-copy Bernoulli).
        Default is ``0``.
    name : str or None, optional
        Optional dynamics node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 20 20 20 40

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``rate``
         - ``0.0 * u.Hz``
         - :math:`r`
         - Parent Poisson intensity in spikes/s.
       * - ``p_copy``
         - ``1.0``
         - :math:`p_{\mathrm{copy}}`
         - Copy probability per parent spike and per child train.
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower activity bound.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper activity bound; ``None`` maps to ``+\infty``.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Time offset added to ``start`` and ``stop``.
       * - ``in_size``
         - ``1``
         - :math:`M`
         - Number/shape of child processes (``M = prod(varshape)``).
       * - ``rng_seed``
         - ``0``
         - -
         - Entropy source for parent/child RNG stream initialization.

    Returns
    -------
    out : Any
        Dynamics node instance. Each call to :meth:`update` returns a
        NumPy ``int64`` array of shape ``self.varshape`` containing per-step
        spike multiplicities for each child process.

    Raises
    ------
    ValueError
        If ``rate < 0``; if ``p_copy`` is outside ``[0, 1]``; if ``stop < start``;
        if scalar conversion fails due non-scalar inputs; or if finite
        ``origin``/``start``/``stop`` are not multiples of ``dt`` when
        simulation resolution is available.
    TypeError
        If conversion of unitful inputs to ``u.Hz`` or ``u.ms`` is invalid.
    KeyError
        At update time, if the simulation environment does not provide
        required entries such as ``dt`` via ``brainstate.environ.get_dt()``.

    Notes
    -----
    - Outputs are multiplicities ``0, 1, 2, ...`` per discrete step, matching
      NEST ``SpikeEvent`` multiplicity semantics rather than binary spike flags.
    - ``init_state`` creates two independent RNG instances to mirror NEST's
      separation of parent and child stochastic paths.
    - ``set`` updates cached timing boundaries immediately when ``dt`` is
      already available in ``brainstate.environ``.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.mip_generator(
       ...         in_size=(2, 3),
       ...         rate=800.0 * u.Hz,
       ...         p_copy=0.25,
       ...         start=5.0 * u.ms,
       ...         stop=40.0 * u.ms,
       ...         rng_seed=7,
       ...     )
       ...     with brainstate.environ.context(t=10.0 * u.ms):
       ...         counts = gen.update()
       ...     _ = counts.shape, counts.dtype

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> gen = brainpy.state.mip_generator(rate=1200.0 * u.Hz, p_copy=0.1)
       >>> gen.set(start=2.0 * u.ms, stop=None, origin=1.0 * u.ms)
       >>> params = gen.get()
       >>> _ = params['rate'], params['p_copy'], params['stop']

    See Also
    --------
    poisson_generator : Independent Poisson trains without shared parent process.
    poisson_generator_ps : Precise-time Poisson generator with dead time.
    inhomogeneous_poisson_generator : Time-varying Poisson rate generator.

    References
    ----------
    .. [1] NEST source: ``models/mip_generator.h`` and
           ``models/mip_generator.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/mip_generator.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        p_copy: ArrayLike = 1.0,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.p_copy = self._to_scalar_float(p_copy, name='p_copy')
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate=self.rate,
            p_copy=self.p_copy,
            start=self.start,
            stop=self.stop,
        )

        self._num_targets = int(np.prod(self.varshape))
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
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _validate_parameters(
        *,
        rate: float,
        p_copy: float,
        start: float,
        stop: float,
    ):
        if rate < 0.0:
            raise ValueError('Rate must be non-negative.')
        if p_copy < 0.0 or p_copy > 1.0:
            raise ValueError('Copy probability must be in [0, 1].')
        if stop < start:
            raise ValueError('stop >= start required.')

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
        """Initialize RNG state for parent and child stochastic paths.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused placeholder for :class:`brainstate.nn.Dynamics` API
            compatibility. Ignored by this implementation.
        **kwargs
            Additional keyword arguments accepted for compatibility and ignored.

        Returns
        -------
        out : Any
            ``None``. The method sets ``self._rng_parent`` and
            ``self._rng_child`` as ``numpy.random.Generator`` instances.

        Raises
        ------
        ValueError
            If ``rng_seed`` cannot be consumed by
            :class:`numpy.random.SeedSequence`.
        TypeError
            If ``rng_seed`` has an invalid type for NumPy RNG initialization.
        """
        del batch_size, kwargs
        seed_seq = np.random.SeedSequence(self.rng_seed)
        parent_seed, child_seed = seed_seq.spawn(2)
        self._rng_parent = np.random.default_rng(parent_seed)
        self._rng_child = np.random.default_rng(child_seed)

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        p_copy: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        """Update public generator parameters using NEST-style semantics.

        Parameters
        ----------
        rate : ArrayLike or object, optional
            New scalar parent rate in Hz. If omitted (internal sentinel
            ``_UNSET``), keep current value. Must satisfy ``rate >= 0`` after
            scalar conversion.
        p_copy : ArrayLike or object, optional
            New scalar copy probability in ``[0, 1]``. If omitted, keep current
            value.
        start : ArrayLike or object, optional
            New scalar relative start time in ms. If omitted, keep current
            value.
        stop : ArrayLike, None, or object, optional
            New scalar relative stop time in ms. ``None`` maps to ``+inf``.
            If omitted, keep current value.
        origin : ArrayLike or object, optional
            New scalar time origin in ms. If omitted, keep current value.

        Returns
        -------
        out : Any
            ``None``. Parameters are updated in place. If ``dt`` is currently
            available in ``brainstate.environ``, cached step bounds are
            recomputed immediately.

        Raises
        ------
        ValueError
            If any provided parameter is non-scalar, violates parameter
            constraints (for example ``p_copy`` outside ``[0, 1]`` or
            ``stop < start``), or finite times are off the simulation grid
            when ``dt`` is available.
        TypeError
            If unit conversion or scalar coercion fails for provided values.
        """
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_p_copy = (
            self.p_copy
            if p_copy is _UNSET
            else self._to_scalar_float(p_copy, name='p_copy')
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
            p_copy=new_p_copy,
            start=new_start,
            stop=new_stop,
        )

        self.rate = new_rate
        self.p_copy = new_p_copy
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    def get(self) -> dict:
        """Return current public parameters as plain Python floats.

        Parameters
        ----------
        None
            This method has no input parameters.

        Returns
        -------
        out : Any
            Dictionary with keys ``'rate'``, ``'p_copy'``, ``'start'``,
            ``'stop'``, and ``'origin'``. Values are Python ``float`` in
            units Hz for ``rate`` and ms for time fields. ``stop`` is
            ``math.inf`` if unbounded.
        """
        return {
            'rate': float(self.rate),
            'p_copy': float(self.p_copy),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_parent_spikes(self, lam: float) -> int:
        return int(self._rng_parent.poisson(lam))

    def _sample_child_spikes(self, n_parent_spikes: int) -> np.ndarray:
        out = np.zeros(self._num_targets, dtype=np.int64)

        if n_parent_spikes <= 0 or self._num_targets == 0:
            return out
        if self.p_copy <= 0.0:
            return out
        if self.p_copy >= 1.0:
            out.fill(int(n_parent_spikes))
            return out

        for i in range(self._num_targets):
            copied = np.count_nonzero(self._rng_child.random(n_parent_spikes) < self.p_copy)
            out[i] = int(copied)
        return out

    def update(self):
        r"""Advance one simulation step and emit child spike multiplicities.

        The method executes the source-equivalent MIP sampling pipeline:
        validate/refresh timing cache, gate activity with
        :math:`t_{\min} < t \le t_{\max}`, draw parent spike multiplicity from
        :math:`\mathrm{Poisson}(r \Delta t / 1000)`, then copy parent spikes
        independently into each child with probability ``p_copy``.

        Parameters
        ----------
        None
            The method reads ``dt`` and optionally ``t`` from
            ``brainstate.environ`` and uses instance parameters.

        Returns
        -------
        out : Any
            NumPy array of dtype ``int64`` with shape ``self.varshape``.
            Entries are per-step spike multiplicities for each child train.
            If inactive, if ``rate <= 0``, or if parent multiplicity is zero,
            all entries are ``0``.

        Raises
        ------
        KeyError
            If the simulation context does not provide ``dt`` required by
            ``brainstate.environ.get_dt()``.
        ValueError
            If finite timing parameters are not aligned to the simulation grid
            after a ``dt`` change.
        TypeError
            If simulation-time values in the environment cannot be converted to
            scalar milliseconds.
        """
        if not hasattr(self, '_rng_parent'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_timing_cache(dt_ms)

        if self.rate <= 0.0:
            return np.zeros(self.varshape, dtype=np.int64)

        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)
        if not self._is_active(curr_step):
            return np.zeros(self.varshape, dtype=np.int64)

        lam = self.rate * dt_ms / 1000.0
        n_parent_spikes = self._sample_parent_spikes(lam)
        if n_parent_spikes <= 0:
            return np.zeros(self.varshape, dtype=np.int64)

        child_counts = self._sample_child_spikes(n_parent_spikes)
        return child_counts.reshape(self.varshape)
