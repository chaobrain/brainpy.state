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
from typing import Sequence

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'inhomogeneous_poisson_generator',
]


_UNSET = object()


class inhomogeneous_poisson_generator(brainstate.nn.Dynamics):
    r"""Inhomogeneous Poisson spike generator with NEST-compatible scheduling.

    Emit Poisson-distributed spike multiplicities from a piecewise-constant
    rate schedule and replicate NEST update ordering for future rate changes.

    **1. Stochastic model and one-step-ahead schedule semantics**

    Let :math:`\Delta t` be the simulation resolution in ms and
    :math:`n \in \mathbb{N}` the current step index with
    :math:`t_n = n \Delta t`. The generator maintains an internal rate
    :math:`r_n` in spikes/s. For each configured pair
    :math:`(t_k, v_k) =` ``(rate_times[k], rate_values[k])``, the requested
    time is aligned to a grid step :math:`s_k`:

    .. math::

       s_k =
       \begin{cases}
         \mathrm{round}(t_k / \Delta t), & \text{if representable on grid}, \\
         \left\lceil t_k / \Delta t \right\rceil, &
         \text{if off-grid and ``allow_offgrid_times`` is True}.
       \end{cases}

    During :meth:`update`, entries with :math:`s_k \le n` are skipped as past
    events. The next unapplied entry is consumed exactly when
    :math:`s_k = n + 1`, i.e., one simulation step ahead of delivery. This
    one-step-ahead convention reproduces NEST device ordering and avoids
    retroactive rate jumps.

    For active steps with :math:`r_n > 0`, per-output spike multiplicities are
    sampled independently as

    .. math::

       K_n \sim \mathrm{Poisson}(\lambda_n), \quad
       \lambda_n = \frac{r_n \,\Delta t}{1000},

    where the factor of 1000 converts Hz × ms to a dimensionless Poisson mean.
    Returned values are non-negative integers and may exceed 1 for high firing
    rates or large time steps.

    **2. Activity window, assumptions, and constraints**

    Activity is gated by the NEST spike-device convention using a
    half-open-on-the-left interval:

    .. math::

       t_{\min} < t_n \le t_{\max}, \quad
       t_{\min} = t_0 + t_{\mathrm{start,rel}},\;
       t_{\max} = t_0 + t_{\mathrm{stop,rel}}.

    Therefore, ``start`` is an exclusive lower bound and ``stop`` is an
    inclusive upper bound in timestamp space. If ``stop is None``,
    :math:`t_{\max} = +\infty` and no upper cutoff is applied.

    The following schedule constraints are enforced at :meth:`set` call time:

    - ``rate_times`` and ``rate_values`` must always be provided together.
    - Flattened lengths of both arrays must match after conversion.
    - Aligned schedule steps :math:`s_k` must form a strictly increasing
      sequence; duplicate grid positions are rejected.
    - Each configured rate time must lie strictly in the future relative to
      the environment time reported by ``brainstate.environ`` at the moment
      :meth:`set` is called.

    **3. Computational implications**

    Schedule preprocessing in :meth:`set` is :math:`O(K)`, where :math:`K` is
    the number of configured change points. The per-step :meth:`update` cost is
    :math:`O(M + \prod \mathrm{varshape})`, where :math:`M` is the number of
    outdated entries skipped in that call (amortized :math:`O(1)` over a
    full simulation). Poisson sampling is vectorized over ``self.varshape``
    via ``jax.random.poisson``, yielding statistically independent output
    trains for each element in the output array.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification for :class:`brainstate.nn.Dynamics`.
        ``self.varshape`` derived from ``in_size`` gives the shape of the
        sampled multiplicity array returned by each :meth:`update` call.
        Default is ``1``.
    rate_times : Sequence[ArrayLike] or ArrayLike or None, optional
        Scheduled rate-change times with logical shape ``(K,)``. Entries are
        interpreted as milliseconds and stored internally as a flattened
        ``np.ndarray`` with dtype ``float64`` after grid alignment. ``None``
        means no schedule is configured at construction time. Must be provided
        together with ``rate_values``. Default is ``None``.
    rate_values : Sequence[ArrayLike] or ArrayLike or None, optional
        Scheduled firing rates in spikes/s (Hz) paired one-to-one with
        ``rate_times``, logical shape ``(K,)``. Stored as a flattened
        ``np.ndarray`` with dtype ``float64``. Must be provided together with
        ``rate_times``. Default is ``None``.
    allow_offgrid_times : bool, optional
        Grid-alignment policy for ``rate_times`` entries that do not fall
        exactly on a simulation time step. If ``False``, any off-grid time
        raises :class:`ValueError`. If ``True``, off-grid times are aligned
        upward (ceiling) to the nearest representable grid step, subject to a
        small absolute tolerance of ``1e-12`` to absorb floating-point round-
        off. Default is ``False``.
    start : ArrayLike, optional
        Scalar relative start time :math:`t_{\mathrm{start,rel}}` in ms.
        Added to ``origin`` to form the exclusive lower bound of the active
        interval. Unitless scalars are treated as ms; :class:`brainunit.Quantity`
        values are converted automatically. Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative stop time :math:`t_{\mathrm{stop,rel}}` in ms. Added
        to ``origin`` to form the inclusive upper bound of the active interval.
        ``None`` disables the upper bound (:math:`t_{\max} = +\infty`).
        Default is ``None``.
    origin : ArrayLike, optional
        Scalar time offset :math:`t_0` in ms applied to both ``start`` and
        ``stop``. Allows shifting the activity window without modifying the
        relative ``start``/``stop`` values. Default is ``0. * u.ms``.
    rng_seed : int, optional
        Integer seed used to initialize the ``jax.random.PRNGKey`` for Poisson
        sampling. Changing the seed produces a statistically independent output
        spike train for otherwise identical parameters. Default is ``0``.
    name : str or None, optional
        Optional human-readable name for the dynamics node passed to
        :class:`brainstate.nn.Dynamics`. Default is ``None``.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 24 18 20 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``rate_times``
         - ``None``
         - :math:`t_k`
         - Scheduled rate-change times, aligned to grid steps :math:`s_k`.
       * - ``rate_values``
         - ``None``
         - :math:`v_k`
         - Scheduled firing rates (spikes/s) applied when :math:`s_k = n + 1`.
       * - ``start``
         - ``0. * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower bound of the active interval.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound; ``None`` means no upper cutoff.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global time offset added to both ``start`` and ``stop``.
       * - ``allow_offgrid_times``
         - ``False``
         - —
         - Off-grid policy: strict grid validation or upward ceiling alignment.
       * - ``rng_seed``
         - ``0``
         - —
         - Seed for the JAX PRNG key used in Poisson sampling.

    Returns
    -------
    out : inhomogeneous_poisson_generator
        Configured dynamics node. Each :meth:`update` call returns a
        ``jax.Array`` with dtype ``int64`` and shape ``self.varshape``
        containing per-step Poisson spike multiplicities.

    Raises
    ------
    ValueError
        If ``stop < start`` at construction time; if ``rate_times`` and
        ``rate_values`` are not provided together; if their flattened lengths
        differ; if any configured time is not strictly in the future; if
        aligned grid steps are not strictly increasing; if an off-grid time
        is supplied while ``allow_offgrid_times`` is ``False``; or if any
        time-like parameter is not scalar-convertible.
    TypeError
        If unit conversion or numeric coercion fails for any time or rate
        input (e.g., incompatible ``brainunit.Quantity`` dimensions).
    KeyError
        At runtime during :meth:`update`, if the simulation context accessed
        via ``brainstate.environ`` is missing the required ``dt`` key.

    Notes
    -----
    - Output values are spike counts per step (``0, 1, 2, ...``), not binary
      spikes. High firing rates or large time steps may produce multiplicities
      greater than one.
    - Calling :meth:`set` with a new non-empty schedule atomically resets the
      internal schedule pointer to index 0, matching NEST setter semantics.
    - Calling :meth:`update` without a prior :meth:`init_state` call will
      lazily initialize state variables on the first invocation.
    - The ``rng_key`` state is split (not folded) at each call, so the Poisson
      samples are statistically independent across time steps and across
      different elements of ``self.varshape``.

    See Also
    --------
    poisson_generator : Homogeneous Poisson stimulation device.
    sinusoidal_poisson_generator : Sinusoidally modulated Poisson device.
    step_rate_generator : Piecewise-constant deterministic rate generator.

    References
    ----------
    .. [1] NEST Simulator model documentation: ``inhomogeneous_poisson_generator``.
           https://nest-simulator.readthedocs.io/en/stable/models/inhomogeneous_poisson_generator.html

    Examples
    --------
    Create a generator that fires at 800 Hz during ``(5, 20]`` ms then goes
    silent, and read out the per-neuron spike counts at step ``t = 6 ms``:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.inhomogeneous_poisson_generator(
       ...         in_size=4,
       ...         rate_times=[5.0 * u.ms, 20.0 * u.ms],
       ...         rate_values=[800.0 * u.Hz, 0.0 * u.Hz],
       ...         start=0.0 * u.ms,
       ...         stop=30.0 * u.ms,
       ...         rng_seed=7,
       ...     )
       ...     gen.init_state()
       ...     with brainstate.environ.context(t=6.0 * u.ms):
       ...         counts = gen.update()
       ...     _ = counts.shape  # (4,), dtype int64

    Allow off-grid rate times and inspect the aligned schedule via
    :meth:`get`:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.inhomogeneous_poisson_generator(
       ...         allow_offgrid_times=True,
       ...     )
       ...     gen.set(
       ...         rate_times=[1.23 * u.ms, 2.34 * u.ms],
       ...         rate_values=[10.0 * u.Hz, 20.0 * u.Hz],
       ...     )
       ...     params = gen.get()
       ...     # params['rate_times'] contains ceil-aligned ms values
       ...     _ = params['allow_offgrid_times']  # True
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate_times: Sequence[ArrayLike] | ArrayLike | None = None,
        rate_values: Sequence[ArrayLike] | ArrayLike | None = None,
        allow_offgrid_times: bool = False,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.allow_offgrid_times = bool(allow_offgrid_times)
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        if self.stop < self.start:
            raise ValueError('stop must be greater than or equal to start.')

        self._rate_times_ms = np.asarray([], dtype=np.float64)
        self._rate_values_hz = np.asarray([], dtype=np.float64)
        self._rate_steps = np.asarray([], dtype=np.int64)

        if (rate_times is None) ^ (rate_values is None):
            raise ValueError('Rate times and values must be reset together.')
        if rate_times is not None:
            self.set(rate_times=rate_times, rate_values=rate_values)

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
    def _to_time_array_ms(values: Sequence[ArrayLike] | ArrayLike) -> np.ndarray:
        if not isinstance(values, u.Quantity):
            arr0 = np.asarray(values)
            if arr0.size == 0:
                return np.asarray([], dtype=np.float64)
        if isinstance(values, u.Quantity):
            arr = values.to_decimal(u.ms)
        else:
            arr = u.math.asarray(values, dtype=jnp.float64)
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    @staticmethod
    def _to_rate_array_hz(values: Sequence[ArrayLike] | ArrayLike) -> np.ndarray:
        if not isinstance(values, u.Quantity):
            arr0 = np.asarray(values)
            if arr0.size == 0:
                return np.asarray([], dtype=np.float64)
        if isinstance(values, u.Quantity):
            arr = values.to_decimal(u.Hz)
        else:
            arr = u.math.asarray(values, dtype=jnp.float64)
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    @staticmethod
    def _array_to_public(value: np.ndarray):
        if value.size == 1:
            return float(value[0])
        return value.tolist()

    @staticmethod
    def _time_to_step(time_ms: float, dt_ms: float) -> int:
        return int(np.rint(time_ms / dt_ms))

    def _dt_ms(self) -> float:
        dt = brainstate.environ.get_dt()
        return self._to_scalar_time_ms(dt)

    def _current_time_ms(self) -> float:
        t = brainstate.environ.get('t', default=0. * u.ms)
        if t is None:
            return 0.0
        return self._to_scalar_time_ms(t)

    def _align_rate_time_to_grid(self, time_ms: float, dt_ms: float) -> tuple[int, float]:
        ratio = time_ms / dt_ms
        nearest = np.rint(ratio)

        if math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-12):
            step = int(nearest)
        elif self.allow_offgrid_times:
            step = int(math.ceil(ratio - 1e-12))
        else:
            raise ValueError(
                f'inhomogeneous_poisson_generator: Time point {time_ms} '
                f'is not representable in current resolution.'
            )

        return step, float(step) * dt_ms

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize transient schedule pointer and RNG state.

        Creates the three :class:`brainstate.ShortTermState` objects required
        by :meth:`update`: the schedule pointer ``_rate_idx`` (``int64``
        scalar), the currently active firing rate ``_rate_hz`` (``float64``
        scalar, initialized to ``0.0``), and the JAX PRNG key ``rng_key``
        seeded from ``self.rng_seed``.

        This method is idempotent with respect to the configured schedule: the
        existing ``_rate_times_ms``, ``_rate_values_hz``, and ``_rate_steps``
        arrays are left unchanged; only the runtime-mutable state variables are
        (re-)created.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused. Present only for :class:`brainstate.nn.Dynamics` API
            compatibility. Default is ``None``.
        **kwargs
            Additional keyword arguments accepted for API compatibility and
            silently ignored.

        """
        del batch_size, kwargs
        self._rate_idx = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int64))
        self._rate_hz = brainstate.ShortTermState(jnp.asarray(0.0, dtype=jnp.float64))
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))

    def set(
        self,
        *,
        rate_times: Sequence[ArrayLike] | ArrayLike | object = _UNSET,
        rate_values: Sequence[ArrayLike] | ArrayLike | object = _UNSET,
        allow_offgrid_times: bool | object = _UNSET,
    ):
        r"""Update the rate schedule and/or off-grid policy with NEST-compatible validation.

        Replaces the current piecewise-constant rate schedule with a new one,
        optionally updating the off-grid alignment policy at the same time.
        All provided times are validated against the current simulation clock
        (must be strictly in the future), aligned to the simulation grid, and
        checked for strict monotonicity.

        Passing ``rate_times=[]`` and ``rate_values=[]`` clears the schedule:
        internal arrays are set to empty and the schedule pointer is reset to 0.

        Parameters
        ----------
        rate_times : Sequence[ArrayLike] or ArrayLike, optional
            New rate-change times in ms. Inputs are flattened to shape ``(K,)``
            and stored as ``np.ndarray[float64]`` after grid alignment. Must be
            provided together with ``rate_values``; omitting one while supplying
            the other raises :class:`ValueError`. If omitted entirely (sentinel
            ``_UNSET``), the existing schedule is left unchanged.
        rate_values : Sequence[ArrayLike] or ArrayLike, optional
            New firing rates in spikes/s (Hz) paired one-to-one with
            ``rate_times``. Stored as ``np.ndarray[float64]``. Must have
            exactly the same flattened length as ``rate_times``.
        allow_offgrid_times : bool, optional
            If supplied, updates ``self.allow_offgrid_times``. Changing this
            flag is only permitted when ``rate_times`` is also being set in the
            same call, or when no schedule has been configured yet. Attempting
            to change the flag with an existing non-empty schedule and without
            new times raises :class:`ValueError`.

        Raises
        ------
        ValueError
            If exactly one of ``rate_times`` / ``rate_values`` is provided
            (must supply both or neither); if their flattened lengths differ;
            if ``allow_offgrid_times`` is changed while an existing non-empty
            schedule is in place without also providing new times; if any time
            value is not strictly greater than the current environment time; if
            any two adjacent aligned grid steps are not strictly increasing; or
            if a time is off-grid and ``allow_offgrid_times`` is ``False``.
        TypeError
            If unit conversion fails for ``rate_times`` or ``rate_values``
            inputs (e.g., incompatible ``brainunit.Quantity`` dimensions).
        """
        times_given = rate_times is not _UNSET
        rates_given = rate_values is not _UNSET

        if allow_offgrid_times is not _UNSET:
            new_flag = bool(allow_offgrid_times)
            if (
                new_flag != self.allow_offgrid_times
                and not (times_given or self._rate_times_ms.size == 0)
            ):
                raise ValueError(
                    'Option can only be set together with rate times '
                    'or if no rate times have been set.'
                )
            self.allow_offgrid_times = new_flag

        if times_given ^ rates_given:
            raise ValueError('Rate times and values must be reset together.')

        if not (times_given or rates_given):
            return

        times_ms = self._to_time_array_ms(rate_times)
        values_hz = self._to_rate_array_hz(rate_values)

        if times_ms.size != values_hz.size:
            raise ValueError('Rate times and values have to be the same size.')

        if times_ms.size == 0:
            self._rate_times_ms = np.asarray([], dtype=np.float64)
            self._rate_values_hz = np.asarray([], dtype=np.float64)
            self._rate_steps = np.asarray([], dtype=np.int64)
            if hasattr(self, '_rate_idx'):
                self._rate_idx.value = jnp.asarray(0, dtype=jnp.int64)
            return

        dt_ms = self._dt_ms()
        now_ms = self._current_time_ms()

        aligned_times = np.empty_like(times_ms, dtype=np.float64)
        aligned_steps = np.empty_like(times_ms, dtype=np.int64)

        for i, t_ms in enumerate(times_ms):
            if t_ms <= now_ms:
                raise ValueError('Time points must lie strictly in the future.')

            step, aligned_ms = self._align_rate_time_to_grid(float(t_ms), dt_ms)
            aligned_steps[i] = step
            aligned_times[i] = aligned_ms

            if i > 0 and aligned_steps[i - 1] >= aligned_steps[i]:
                raise ValueError('Rate times must be strictly increasing.')

        self._rate_times_ms = aligned_times
        self._rate_values_hz = values_hz
        self._rate_steps = aligned_steps

        # Match NEST setter semantics: schedule index is reset on new data.
        if hasattr(self, '_rate_idx'):
            self._rate_idx.value = jnp.asarray(0, dtype=jnp.int64)

    def get(self) -> dict:
        r"""Return current schedule and timing parameters in NEST-style format.

        Serializes all user-configurable generator parameters into a plain
        Python dict. This mirrors the ``nest.GetStatus`` interface so that
        parameter introspection and round-tripping via :meth:`set` / :meth:`get`
        work as expected.

        Returns
        -------
        params : dict
            Dictionary with the following keys:

            - ``'rate_times'`` (``float`` or ``list[float]``): Grid-aligned
              rate-change times in ms. A single-entry schedule is returned as
              a bare ``float``; a multi-entry schedule as a Python ``list``.
              An empty schedule returns an empty ``list``.
            - ``'rate_values'`` (``float`` or ``list[float]``): Corresponding
              firing rates in spikes/s (Hz), same shape convention as
              ``'rate_times'``.
            - ``'allow_offgrid_times'`` (``bool``): Current off-grid alignment
              policy.
            - ``'start'`` (``float``): Relative exclusive lower activity bound
              in ms.
            - ``'stop'`` (``float``): Inclusive upper activity bound in ms, or
              ``float('inf')`` if no upper bound was set.
            - ``'origin'`` (``float``): Global time offset in ms.
        """
        return {
            'rate_times': self._array_to_public(self._rate_times_ms),
            'rate_values': self._array_to_public(self._rate_values_hz),
            'allow_offgrid_times': bool(self.allow_offgrid_times),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _is_active(self, curr_step: int, dt_ms: float) -> bool:
        t_ms = curr_step * dt_ms
        t_min = self.origin + self.start
        t_max = self.origin + self.stop
        return (t_min < t_ms) and (t_ms <= t_max)

    def _sample_poisson(self, lam: float) -> jax.Array:
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.poisson(
            subkey,
            lam=jnp.asarray(lam, dtype=jnp.float64),
            shape=self.varshape,
        ).astype(jnp.int64)

    def update(self):
        r"""Advance one simulation step and emit Poisson spike multiplicities.

        Reads the current simulation time from ``brainstate.environ``, advances
        the schedule pointer past any entries whose grid step :math:`s_k \le n`,
        then applies the next scheduled rate change if :math:`s_k = n + 1`.
        When the generator is active (current time inside the activity window)
        and the current rate is positive, samples a Poisson multiplicity array
        over ``self.varshape``. Otherwise returns a zero array.

        Lazy initialization: if :meth:`init_state` has not been called, this
        method initializes state variables on the first invocation.

        Returns
        -------
        spikes : jax.Array, shape ``self.varshape``, dtype ``int64``
            Per-output Poisson spike multiplicity for the current time step.
            Each element :math:`K_i \sim \mathrm{Poisson}(\lambda_n)` where
            :math:`\lambda_n = r_n \Delta t / 1000`. Returns all-zero array
            when the generator is inactive or the current rate is zero.
        """
        if not hasattr(self, '_rate_idx'):
            self.init_state()

        dt_ms = self._dt_ms()
        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)

        idx = int(self._rate_idx.value)
        while idx < self._rate_steps.size and int(self._rate_steps[idx]) <= curr_step:
            idx += 1

        rate_hz = float(self._rate_hz.value)
        if idx < self._rate_steps.size and curr_step + 1 == int(self._rate_steps[idx]):
            rate_hz = float(self._rate_values_hz[idx])
            idx += 1

        self._rate_idx.value = jnp.asarray(idx, dtype=jnp.int64)
        self._rate_hz.value = jnp.asarray(rate_hz, dtype=jnp.float64)

        if rate_hz > 0.0 and self._is_active(curr_step, dt_ms):
            lam = rate_hz * dt_ms / 1000.0
            return self._sample_poisson(lam)

        return jnp.zeros(self.varshape, dtype=jnp.int64)
