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

    Let :math:`\Delta t` be simulation resolution in ms and
    :math:`n \in \mathbb{N}` the current step index with
    :math:`t_n = n \Delta t`. The generator keeps an internal rate
    :math:`r_n` in spikes/s. For each configured pair
    :math:`(t_k, v_k) =` ``(rate_times[k], rate_values[k])``, the time is
    aligned to a grid step :math:`s_k`:

    .. math::

       s_k =
       \begin{cases}
         \mathrm{round}(t_k / \Delta t), & \text{if representable on grid}, \\
         \left\lceil t_k / \Delta t \right\rceil, &
         \text{if off-grid and ``allow_offgrid_times`` is True}.
       \end{cases}

    During :meth:`update`, entries with :math:`s_k \le n` are skipped as past
    events, then the next unapplied entry is consumed exactly when
    :math:`s_k = n + 1`, i.e., one simulation step ahead of delivery. This
    reproduces NEST device ordering and avoids retroactive rate jumps.

    For active steps with :math:`r_n > 0`, multiplicities are sampled as

    .. math::

       K_n \sim \mathrm{Poisson}(\lambda_n), \quad
       \lambda_n = r_n \Delta t / 1000,

    where the ``1000`` factor converts Hz * ms to a dimensionless mean.
    Returned values are non-negative integers and may exceed 1.

    **2. Activity window, assumptions, and constraints**

    Activity is gated by NEST spike-device convention:

    .. math::

       t_{\min} < t_n \le t_{\max}, \quad
       t_{\min} = origin + start,\ t_{\max} = origin + stop.

    Therefore, ``start`` is exclusive and ``stop`` is inclusive in timestamp
    space. If ``stop is None``, :math:`t_{\max} = +\infty`.

    Enforced schedule constraints:

    - ``rate_times`` and ``rate_values`` must be provided together.
    - Lengths must match after flattening to 1-D arrays.
    - Aligned schedule steps must be strictly increasing.
    - Each configured rate time must be strictly in the future relative to
      current environment time at :meth:`set` call time.

    **3. Computational implications**

    Schedule preprocessing in :meth:`set` is :math:`O(K)`, where :math:`K` is
    number of configured change points. Per-step :meth:`update` cost is
    :math:`O(M + \prod \mathrm{varshape})`, where :math:`M` is number of
    skipped outdated entries in that call. Poisson sampling is vectorized over
    ``self.varshape``, yielding independent output trains per element.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification for :class:`brainstate.nn.Dynamics`.
        ``self.varshape`` derived from ``in_size`` is the shape of sampled
        multiplicity arrays. Default is ``1``.
    rate_times : Sequence[ArrayLike] or ArrayLike or None, optional
        Rate-change times with logical shape ``(K,)``. Entries are interpreted
        as milliseconds and converted to a flattened ``np.ndarray[float64]``.
        ``None`` means no schedule at construction. Default is ``None``.
    rate_values : Sequence[ArrayLike] or ArrayLike or None, optional
        Rate values paired one-to-one with ``rate_times``, logical shape
        ``(K,)``. Entries are interpreted as spikes/s (Hz) and converted to a
        flattened ``np.ndarray[float64]``. Default is ``None``.
    allow_offgrid_times : bool, optional
        Grid-alignment policy for non-representable ``rate_times``.
        If ``False``, off-grid times raise :class:`ValueError`.
        If ``True``, off-grid times are aligned upward to the end of the
        current step (``ceil`` policy with a small numerical tolerance).
        Default is ``False``.
    start : ArrayLike, optional
        Scalar relative start time in ms. ``start`` is added to ``origin`` and
        used as an exclusive lower activity bound. Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative stop time in ms. ``stop`` is added to ``origin`` and
        used as an inclusive upper activity bound. ``None`` means no upper
        bound. Default is ``None``.
    origin : ArrayLike, optional
        Scalar time offset in ms applied to ``start`` and ``stop``.
        Default is ``0. * u.ms``.
    rng_seed : int, optional
        Integer seed used to initialize ``jax.random.PRNGKey`` for Poisson
        sampling. Default is ``0``.
    name : str or None, optional
        Optional dynamics node name.

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
         - Scheduled rate-change times, aligned to grid steps ``s_k``.
       * - ``rate_values``
         - ``None``
         - :math:`v_k`
         - Scheduled rates (spikes/s) applied when ``s_k = n + 1``.
       * - ``start``
         - ``0. * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower bound of active interval.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound of active interval.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global offset added to ``start`` and ``stop``.
       * - ``allow_offgrid_times``
         - ``False``
         - -
         - Selects strict-grid validation vs upward off-grid alignment.

    Returns
    -------
    out : Any
        Dynamics node. Each :meth:`update` call returns an ``int64`` JAX array
        with shape ``self.varshape`` containing per-step spike multiplicities.

    Raises
    ------
    ValueError
        If ``stop < start`` at construction; if ``rate_times`` and
        ``rate_values`` are not set together; if schedule lengths differ; if
        configured times are not strictly in the future; if aligned times are
        not strictly increasing; if off-grid times are provided while
        ``allow_offgrid_times`` is ``False``; or if time parameters are not
        scalar-convertible.
    TypeError
        If unit conversion or numeric conversion fails for provided time/rate
        inputs.
    KeyError
        At runtime, if simulation context is missing required entries such as
        ``dt`` (depending on ``brainstate.environ`` behavior).

    Notes
    -----
    - Output values are spike counts per step (``0, 1, 2, ...``), not binary
      spikes.
    - Re-calling :meth:`set` with a new non-empty schedule resets the internal
      schedule index to match NEST setter semantics.
    - Calling :meth:`update` without prior :meth:`init_state` lazily
      initializes state variables.

    Examples
    --------
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
       ...     with brainstate.environ.context(t=6.0 * u.ms):
       ...         counts = gen.update()
       ...     _ = counts.shape

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
       ...     _ = params['rate_times']

    See Also
    --------
    poisson_generator : Homogeneous Poisson stimulation device.
    sinusoidal_poisson_generator : Sinusoidally modulated Poisson device.
    step_rate_generator : Piecewise-constant deterministic rate generator.

    References
    ----------
    .. [1] NEST Simulator model: ``inhomogeneous_poisson_generator``.
           https://nest-simulator.readthedocs.io/en/stable/models/inhomogeneous_poisson_generator.html
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
        """Initialize transient schedule and RNG state.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused. Present for framework API compatibility.
        **kwargs
            Unused keyword arguments for API compatibility.

        Returns
        -------
        out : Any
            ``None``. Side effect: creates ``_rate_idx``, ``_rate_hz``, and
            ``rng_key`` as :class:`brainstate.ShortTermState` objects.
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
        """Update schedule and off-grid policy with NEST-compatible checks.

        Parameters
        ----------
        rate_times : Sequence[ArrayLike] or ArrayLike or object, optional
            New rate-change times (ms). Must be provided together with
            ``rate_values`` unless omitted as ``_UNSET``.
            Inputs are flattened to shape ``(K,)`` and converted to
            ``float64`` ms.
        rate_values : Sequence[ArrayLike] or ArrayLike or object, optional
            New rate values (spikes/s) paired with ``rate_times``.
            Must have exactly the same flattened length.
        allow_offgrid_times : bool or object, optional
            Optional update for off-grid alignment policy. Changing this flag
            is only allowed when setting a schedule at the same call or when no
            schedule has been configured yet.

        Returns
        -------
        out : Any
            ``None``. Side effect: updates internal schedule arrays
            (``_rate_times_ms``, ``_rate_values_hz``, ``_rate_steps``) and may
            reset ``_rate_idx`` to ``0`` if state is initialized.

        Raises
        ------
        ValueError
            If ``rate_times`` and ``rate_values`` are not provided together;
            if lengths differ; if off-grid policy change is invalid for current
            state; if a schedule time is not in the future; if aligned times
            are not strictly increasing; or if off-grid time handling is
            disabled and a time is not representable on the simulation grid.
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
        """Return public schedule/timing parameters in NEST-style format.

        Parameters
        ----------
        None

        Returns
        -------
        out : Any
            ``dict`` with keys ``'rate_times'``, ``'rate_values'``,
            ``'allow_offgrid_times'``, ``'start'``, ``'stop'``, and
            ``'origin'``. Scalar schedules are returned as ``float``, multi-
            entry schedules as Python lists.
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
        """Advance one simulation step and emit spike multiplicities.

        Parameters
        ----------
        None

        Returns
        -------
        out : Any
            ``jax.Array`` with dtype ``int64`` and shape ``self.varshape``.
            Values are sampled Poisson multiplicities when active and
            ``rate_hz > 0``; otherwise zeros.
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
