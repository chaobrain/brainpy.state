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
import saiunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTDevice

__all__ = [
    'gamma_sup_generator',
]

_UNSET = object()


class gamma_sup_generator(NESTDevice):
    r"""Superposition of independent gamma processes (NEST-compatible).

    Description
    -----------
    ``gamma_sup_generator`` re-implements NEST's stimulation device of the
    same name. It emits, per output train and simulation step, the
    multiplicity of spikes produced by superimposing ``n_proc`` independent
    component renewal processes with gamma-distributed inter-spike intervals.

    **1. State-space model, derivation, and update equations**

    Let :math:`k = \mathrm{gamma\_shape}` and :math:`r = \mathrm{rate}` in Hz.
    Each component gamma process is represented as a cyclic chain of ``k``
    exponential phases. Using an occupation vector per output train,

    .. math::

       \mathbf{occ} = (occ_0, \dots, occ_{k-1}),
       \qquad \sum_{i=0}^{k-1} occ_i = n_{\mathrm{proc}},

    a process in phase ``i`` transitions to phase ``i+1`` (mod ``k``) with
    per-step probability

    .. math::

       p = r \cdot k \cdot \Delta t / 1000.

    This is the discrete-time hazard form used by NEST. For each bin ``i``,

    .. math::

       n_i \sim \mathrm{Binomial}(occ_i, p),

    except for NEST's sparse/high-count approximation branch:

    - if ``occ_i >= 100 and p <= 0.01``, or
    - if ``occ_i >= 500 and p * occ_i <= 0.1``,

    use ``Poisson(p * occ_i)`` and clip to ``occ_i``.

    After sampling, all ``n_i`` are moved simultaneously to preserve integer
    mass and avoid order-dependent updates. The emitted spike multiplicity for
    one train is

    .. math::

       K_n = n_{k-1},

    i.e., transitions leaving the last phase and re-entering phase 0. This
    allows per-step counts larger than 1, matching NEST ``SpikeEvent``
    multiplicity semantics.

    **2. Timing semantics and activity window**

    Activity follows NEST ``StimulationDevice::is_active`` for spike
    generators:

    .. math::

       t_{\min} < t \le t_{\max},
       \qquad
       t_{\min} = origin + start,\quad t_{\max} = origin + stop.

    Therefore ``start`` is exclusive and ``stop`` is inclusive. Internally,
    finite times are projected to integer steps with
    ``round(time_ms / dt_ms)`` and checked as
    ``t_min_step < curr_step <= t_max_step``.

    **3. Assumptions, constraints, and computational implications**

    Parameters are scalarized to ``float64``/``int`` before simulation.
    Enforced constraints are ``rate >= 0``, ``gamma_shape >= 1``,
    ``n_proc >= 1``, and ``stop >= start``. If ``dt`` is available, finite
    ``origin``, ``start``, and ``stop`` must lie on the simulation grid
    (absolute tolerance ``1e-12`` in ``time/dt`` ratio).

    Runtime complexity of :meth:`update` is
    :math:`O(\prod \mathrm{varshape} \cdot \mathrm{gamma\_shape})`, with state
    memory :math:`O(\prod \mathrm{varshape} \cdot \mathrm{gamma\_shape})` from
    the occupation matrix. RNG sampling uses ``numpy.random.Generator``
    (seeded by ``rng_seed``), so stochastic draws are CPU NumPy based rather
    than JAX-key based.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification consumed by
        :class:`brainstate.nn.Dynamics`. The exact shape returned by
        :meth:`update` is ``self.varshape`` derived from ``in_size``; each
        element corresponds to one independent output train. Default is ``1``.
    rate : ArrayLike, optional
        Scalar component-process rate in spikes/s (Hz), shape ``()`` after
        conversion. Accepts a single-element numeric ``ArrayLike`` or a
        :class:`saiunit.Quantity` convertible to ``u.Hz``.
        Must satisfy ``rate >= 0``. Default is ``0.0 * u.Hz``.
    gamma_shape : ArrayLike, optional
        Scalar integer gamma shape :math:`k`, shape ``()`` after conversion.
        Parsed via nearest-integer check with absolute tolerance ``1e-12``.
        Must satisfy ``gamma_shape >= 1``. Default is ``1``.
    n_proc : ArrayLike, optional
        Scalar integer number of independent component processes per output
        train, shape ``()`` after conversion. Parsed by nearest-integer check
        with absolute tolerance ``1e-12``. Must satisfy ``n_proc >= 1``.
        Default is ``1``.
    start : ArrayLike, optional
        Scalar relative activation time in ms, shape ``()`` after conversion.
        Effective lower activity bound is ``origin + start`` and is exclusive.
        Must be grid-representable when ``dt`` is available.
        Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative deactivation time in ms, shape ``()`` after
        conversion. Effective upper activity bound is ``origin + stop`` and is
        inclusive. ``None`` maps to ``+inf``. Must satisfy ``stop >= start``
        and be grid-representable when finite and ``dt`` is available.
        Default is ``None``.
    origin : ArrayLike, optional
        Scalar time-origin offset in ms, shape ``()`` after conversion, added
        to ``start`` and ``stop`` to compute absolute active bounds.
        Must be grid-representable when finite and ``dt`` is available.
        Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed used to initialize ``numpy.random.default_rng`` in
        :meth:`init_state`. Default is ``0``.
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
         - Component-process rate in spikes/s.
       * - ``gamma_shape``
         - ``1``
         - :math:`k`
         - Number of cyclic exponential phases per component process.
       * - ``n_proc``
         - ``1``
         - :math:`n_{\mathrm{proc}}`
         - Number of superimposed component processes per output train.
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
         - Defines ``self.varshape`` for independent output trains.
       * - ``rng_seed``
         - ``0``
         - -
         - Seed of the NumPy generator used for transition draws.

    Raises
    ------
    ValueError
        If scalar conversion fails due to non-scalar inputs; if ``rate < 0``;
        if ``gamma_shape < 1``; if ``n_proc < 1``; if ``stop < start``; if
        integer-valued inputs are non-integral beyond tolerance; or if finite
        ``origin``/``start``/``stop`` are not multiples of simulation
        resolution when ``dt`` is available.
    TypeError
        If unit conversion to ``u.Hz`` or ``u.ms`` fails for supplied inputs.
    KeyError
        At runtime, if required simulation-context fields (for example ``dt``
        used by ``brainstate.environ.get_dt()``) are unavailable.

    Notes
    -----
    - Initial occupation is the NEST equilibrium approximation used in
      ``pre_run_hook()``:
      ``floor(n_proc / gamma_shape)`` in all bins, with the remainder added
      to the last bin.
    - As in NEST, each output train maintains independent internal occupation
      states.
    - In the direct binomial branch, ``transition_prob`` is numerically
      clamped to ``[0, 1]`` before calling NumPy RNG to avoid invalid
      probability inputs in edge cases.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.gamma_sup_generator(
       ...         in_size=(2, 3),
       ...         rate=20.0 * u.Hz,
       ...         gamma_shape=3,
       ...         n_proc=50,
       ...         start=5.0 * u.ms,
       ...         stop=40.0 * u.ms,
       ...         rng_seed=7,
       ...     )
       ...     with brainstate.environ.context(t=12.0 * u.ms):
       ...         counts = gen.update()
       ...     _ = counts.shape

    .. code-block:: python

       >>> import brainpy
       >>> import saiunit as u
       >>> gen = brainpy.state.gamma_sup_generator(rate=15.0 * u.Hz, gamma_shape=2)
       >>> gen.set(n_proc=20, stop=None, origin=1.0 * u.ms)
       >>> params = gen.get()
       >>> _ = params['gamma_shape'], params['n_proc']

    See Also
    --------
    ppd_sup_generator : Superposition with dead-time component processes.
    sinusoidal_gamma_generator : Inhomogeneous gamma generator with sinusoidal rate.

    References
    ----------
    .. [1] NEST source: ``models/gamma_sup_generator.cpp`` and
           ``models/gamma_sup_generator.h``.
    .. [2] NEST model docs:
           https://nest-simulator.readthedocs.io/en/stable/models/gamma_sup_generator.html
    .. [3] Deger M, Helias M, Boucsein C, Rotter S (2012).
           Statistical properties of superimposed stationary spike trains.
           Journal of Computational Neuroscience.
           https://doi.org/10.1007/s10827-011-0362-8
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        gamma_shape: ArrayLike = 1,
        n_proc: ArrayLike = 1,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.gamma_shape = self._to_scalar_int(gamma_shape, name='gamma_shape')
        self.n_proc = self._to_scalar_int(n_proc, name='n_proc')
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate=self.rate,
            gamma_shape=self.gamma_shape,
            n_proc=self.n_proc,
            start=self.start,
            stop=self.stop,
        )

        self._num_targets = int(np.prod(self.varshape))
        self._transition_prob = 0.0
        self._dt_cache_ms = np.nan
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max
        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

    @staticmethod
    def _to_scalar_time_ms(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            dftype = brainstate.environ.dftype()
            arr = np.asarray(value.to_decimal(u.ms), dtype=dftype)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError('Time parameters must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_rate_hz(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            dftype = brainstate.environ.dftype()
            arr = np.asarray(value.to_decimal(u.Hz), dtype=dftype)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError('rate must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_int(value: ArrayLike, *, name: str) -> int:
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        scalar = float(arr.reshape(()))
        nearest = np.rint(scalar)
        if not math.isclose(scalar, nearest, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be an integer.')
        return int(nearest)

    @staticmethod
    def _validate_parameters(
        *,
        rate: float,
        gamma_shape: int,
        n_proc: int,
        start: float,
        stop: float,
    ):
        if gamma_shape < 1:
            raise ValueError('The shape must be larger or equal 1')
        if rate < 0.0:
            raise ValueError('The rate must be larger than 0.')
        if n_proc < 1:
            raise ValueError('The number of component processes cannot be smaller than one')
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

    def _refresh_runtime_cache(self, dt_ms: float):
        self._assert_grid_time('origin', self.origin, dt_ms)
        self._assert_grid_time('start', self.start, dt_ms)
        self._assert_grid_time('stop', self.stop, dt_ms)

        self._t_min_step = self._time_to_step(self.origin + self.start, dt_ms)
        if np.isfinite(self.stop):
            self._t_max_step = self._time_to_step(self.origin + self.stop, dt_ms)
        else:
            self._t_max_step = np.iinfo(np.int64).max

        self._transition_prob = self.rate * self.gamma_shape * dt_ms / 1000.0
        self._dt_cache_ms = float(dt_ms)

    def _is_active(self, curr_step: int) -> bool:
        return (self._t_min_step < curr_step) and (curr_step <= self._t_max_step)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize occupancy and RNG state for all output trains.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused API placeholder for compatibility with Dynamics interfaces.
            Ignored.
        **kwargs
            Additional unused keyword arguments. Ignored.

        """
        del batch_size, kwargs

        ini_occ_ref = int(self.n_proc // self.gamma_shape)
        ini_occ_act = int(self.n_proc - ini_occ_ref * self.gamma_shape)

        ditype = brainstate.environ.ditype()
        occ = np.full(
            (self._num_targets, self.gamma_shape),
            ini_occ_ref,
            dtype=ditype,
        )
        occ[:, -1] += ini_occ_act
        self.occ = brainstate.ShortTermState(occ)
        self._rng = np.random.default_rng(self.rng_seed)

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        gamma_shape: ArrayLike | object = _UNSET,
        n_proc: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        r"""Set public parameters with NEST-compatible semantics.

        Parameters
        ----------
        rate : ArrayLike or object, optional
            New scalar component rate in Hz. ``_UNSET`` keeps current value.
        gamma_shape : ArrayLike or object, optional
            New scalar integer gamma shape ``>= 1``. ``_UNSET`` keeps current
            value.
        n_proc : ArrayLike or object, optional
            New scalar integer number of component processes ``>= 1``.
            ``_UNSET`` keeps current value.
        start : ArrayLike or object, optional
            New scalar relative start time in ms. ``_UNSET`` keeps current
            value.
        stop : ArrayLike, object, or None, optional
            New scalar relative stop time in ms; ``None`` maps to ``+inf``.
            ``_UNSET`` keeps current value.
        origin : ArrayLike or object, optional
            New scalar origin time in ms. ``_UNSET`` keeps current value.


        Raises
        ------
        ValueError
            If converted values violate model constraints or are non-scalar.
        TypeError
            If unit conversion to Hz/ms fails.
        """
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_gamma_shape = (
            self.gamma_shape
            if gamma_shape is _UNSET
            else self._to_scalar_int(gamma_shape, name='gamma_shape')
        )
        new_n_proc = (
            self.n_proc
            if n_proc is _UNSET
            else self._to_scalar_int(n_proc, name='n_proc')
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
            gamma_shape=new_gamma_shape,
            n_proc=new_n_proc,
            start=new_start,
            stop=new_stop,
        )

        self.rate = new_rate
        self.gamma_shape = new_gamma_shape
        self.n_proc = new_n_proc
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

    def get(self) -> dict:
        r"""Return current public parameters as plain Python scalars.

        Returns
        -------
        out : dict
            ``dict`` with keys ``rate``, ``gamma_shape``, ``n_proc``,
            ``start``, ``stop``, and ``origin``. Values are ``float``/``int``
            in public units (Hz for ``rate``, ms for times).
        """
        return {
            'rate': float(self.rate),
            'gamma_shape': int(self.gamma_shape),
            'n_proc': int(self.n_proc),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_poisson(self, lam: float) -> int:
        return int(self._rng.poisson(lam))

    def _sample_binomial(self, n: int, p: float) -> int:
        return int(self._rng.binomial(n, p))

    def _update_internal_states(self, occ_row: np.ndarray, transition_prob: float) -> int:
        n_bins = occ_row.size
        ditype = brainstate.environ.ditype()
        n_trans = np.zeros(n_bins, dtype=ditype)

        for i in range(n_bins):
            occ_i = int(occ_row[i])
            if occ_i <= 0:
                continue

            use_poisson_approx = (
                (occ_i >= 100 and transition_prob <= 0.01)
                or (occ_i >= 500 and transition_prob * occ_i <= 0.1)
            )

            if use_poisson_approx:
                n_i = self._sample_poisson(transition_prob * occ_i)
                if n_i > occ_i:
                    n_i = occ_i
            else:
                # NEST uses std::binomial_distribution directly.
                # Clamp p numerically to avoid invalid values in Python RNG.
                if transition_prob <= 0.0:
                    n_i = 0
                elif transition_prob >= 1.0:
                    n_i = occ_i
                else:
                    n_i = self._sample_binomial(occ_i, transition_prob)
            n_trans[i] = int(n_i)

        for i in range(n_bins):
            n_i = int(n_trans[i])
            if n_i <= 0:
                continue
            occ_row[i] -= n_i
            if i == n_bins - 1:
                occ_row[0] += n_i
            else:
                occ_row[i + 1] += n_i

        return int(n_trans[-1])

    def update(self):
        r"""Advance one simulation step and return per-train spike multiplicity.

        The method lazily initializes state, refreshes timing/probability cache
        when ``dt`` changes, applies the active-window test, then updates each
        train's occupation vector using NEST-equivalent transition logic.

        Returns
        -------
        out : jax.Array
            JAX array with dtype ``int64`` and shape ``self.varshape``.
            Each element is the number of emitted spikes for one output train
            in the current step.

        Raises
        ------
        ValueError
            If cached times fail simulation-grid consistency checks during
            cache refresh.
        KeyError
            If required simulation context values (for example ``dt``) are
            unavailable in ``brainstate.environ``.
        """
        if not hasattr(self, 'occ'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_runtime_cache(dt_ms)

        if self.rate <= 0.0 or self._num_targets == 0:
            ditype = brainstate.environ.ditype()
            return jnp.zeros(self.varshape, dtype=ditype)

        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)
        if not self._is_active(curr_step):
            return jnp.zeros(self.varshape, dtype=ditype)

        occ = np.asarray(self.occ.value, dtype=ditype).copy()
        counts = np.zeros(self._num_targets, dtype=ditype)

        for idx in range(self._num_targets):
            counts[idx] = self._update_internal_states(occ[idx], self._transition_prob)

        self.occ.value = occ
        return jnp.asarray(counts.reshape(self.varshape), dtype=ditype)
