# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Stimulation devices on the K0 ``Device`` contract (NEST timing semantics).

Activity windows follow NEST ``StimulationDevice::is_active``: with
``t_min = steps(origin + start)`` and ``t_max = steps(origin + stop)`` (NEST
round-up conversion, :func:`~brainspike._kernel.units.grid_steps`), a stamp
``S`` is active iff ``t_min < S <= t_max``.

* :class:`poisson_generator` -- tests the step index ``n`` itself (NEST tests
  the slice time ``origin + lag``).
* :class:`spike_generator` -- a spike time ``T`` has the stamp
  ``S = steps(origin + T)`` and is emitted at step ``S - 1``, so its NEST time
  stamp (``(n + 1) dt`` for an emission at step ``n``) is ``T``.
* :class:`dc_generator` -- tests ``n + 2`` (NEST current generators): with the
  NEST minimal delay of one step and a latched receiving ``current`` port (NEST
  models, ``latch=True``), the current acts exactly when it does in NEST.

Every device keeps its last output in a state (``spike`` / ``I``) that
``update`` returns, so observers can sample it like a neuron output.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.context import StepContext
from brainspike._kernel.model import Device
from brainspike._kernel.models._common import device_window
from brainspike._kernel.params import Init, Param, Params, Record, States
from brainspike._kernel.rng import draw_blocked
from brainspike._kernel.ports import Current, Event, Ports
from brainspike._kernel.units import grid_steps, to_mantissa

__all__ = ['poisson_generator', 'spike_generator', 'dc_generator', 'poisson_counts']

#: Below this mean a unit's count is drawn by CDF inversion; at or above it by
#: ``jax.random.poisson`` (computed only when some unit needs it).
_INVERSION_MAX_LAM = 10.0
#: CDF terms evaluated unconditionally (one fused element-wise pass); a ``while_loop``
#: finishes the rare units whose count exceeds it.
_INVERSION_UNROLL = 12


def _inversion(u_: jax.Array, lam: jax.Array, dtype: Any) -> jax.Array:
    """Smallest ``k`` with ``u < F(k)`` per element (element-wise in ``u``, ``lam``)."""
    p = jnp.exp(-lam)
    F = p
    k = jnp.zeros(u_.shape, dtype)
    for i in range(1, _INVERSION_UNROLL + 1):  # Python-unrolled: one fused pass
        k = k + (u_ >= F).astype(dtype)
        p = p * lam / i
        F = F + p

    def cond(c):
        _, k, p, F = c
        return jnp.any((u_ >= F) & (p > 0))

    def body(c):
        i, k, p, F = c
        k = k + ((u_ >= F) & (p > 0)).astype(dtype)
        p = p * lam / i.astype(jnp.float32)
        return i + 1, k, p, F + p

    i0 = jnp.asarray(_INVERSION_UNROLL + 1, dtype)
    return jax.lax.while_loop(cond, body, (i0, k, p, F))[1]


def poisson_counts(key: jax.Array, lam: Any, shape: tuple) -> jax.Array:
    """Poisson counts per unit, drawn per id block (CDF inversion for ``lam < 10``).

    Randomness is laid out per block of :data:`~brainspike._kernel.rng.GID_BLOCK`
    units (:func:`~brainspike._kernel.rng.draw_blocked`, D-g): unit ``i``'s count
    depends only on the key of its block, its lane and its own ``lam`` -- not on the
    population size or the other units' rates.

    A unit with ``lam < 10`` gets the smallest ``k`` with ``u < F(k)``, ``u ~ U[0,
    1)`` (one uniform per unit), ``F`` the Poisson CDF accumulated in float32 (``p_k
    = p_{k-1} lam / k``). The first :data:`_INVERSION_UNROLL` terms are one fused
    element-wise pass; a ``while_loop`` continues only while some unit is still
    searching (it stops when the next term underflows, so it always terminates).
    This draws one random number per unit instead of Knuth's ``count + 1``
    (``jax.random.poisson``), about 6x faster at the Brunel rate (``lam = 0.88``).
    A unit with ``lam >= 10`` gets ``jax.random.poisson`` from its block key folded
    with 1 (evaluated only if some unit needs it). Draws are float32 regardless of
    ``jax_enable_x64`` (same stream in both precisions).

    Parameters
    ----------
    key : jax.Array
        Stacked block keys (``StepContext.key`` in a plan) or one key (standalone
        use; block ``b`` then uses ``fold_in(key, b)``).
    lam : array_like
        Means, broadcastable to ``shape`` (non-negative).
    shape : tuple of int
        Output shape ``(n,)`` (one-dimensional).

    Returns
    -------
    jax.Array
        Integer counts (JAX's default int dtype).

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> from brainspike._kernel.models.devices import poisson_counts
        >>> k = poisson_counts(jax.random.key(0), 0.5, (100000,))
        >>> 0.49 < float(k.mean()) < 0.51
        True
    """
    shape = tuple(int(x) for x in shape)
    if len(shape) != 1:
        raise ValueError(f'poisson_counts draws one count per unit; shape must be (n,), got {shape}')
    n = shape[0]
    lam = jnp.broadcast_to(jnp.asarray(lam, jnp.float32), shape)
    dtype = jnp.zeros((), dtype=int).dtype
    small = lam < _INVERSION_MAX_LAM

    def uniforms():
        return draw_blocked(key, n, lambda k, s: jax.random.uniform(k, s, jnp.float32))

    # both branches draw the same uniforms; keeping the draw inside the branches lets
    # XLA fuse it with the inversion (as in K0) -- measurably faster on GPU
    def all_small(lam):
        return _inversion(uniforms(), lam, dtype)

    def mixed(lam):
        inv = _inversion(uniforms(), jnp.where(small, lam, 0.0), dtype)
        big = draw_blocked(key, n, lambda k, s, l: jax.random.poisson(
            jax.random.fold_in(k, 1), l, s, dtype=dtype), jnp.where(small, 0.0, lam))
        return jnp.where(small, inv, big)

    return jax.lax.cond(jnp.all(small), all_small, mixed, lam)


_WINDOW = dict(start=Param(0.0 * u.ms), stop=Param(float('inf') * u.ms), origin=Param(0.0 * u.ms))


def _dt_ms(ctx: StepContext) -> Any:
    return to_mantissa(ctx.dt, u.ms)


class poisson_generator(Device):
    r"""Poisson spike source: ``Poisson(rate * dt)`` spikes per unit and step.

    Parameters
    ----------
    n : int
        Number of independent units (NEST draws one train **per target**; give
        each target its own unit and connect ``one_to_one`` to reproduce that).
    params : mapping, optional
        ``rate`` (Hz, default 0), ``start`` (0 ms), ``stop`` (inf),
        ``origin`` (0 ms); per unit, traced.
    **kwargs
        ``precision``, ``name``.

    Notes
    -----
    Output ``spike`` -- ``event`` with ``max_multiplicity=None`` (counts > 1
    possible at high rates). Randomness comes only from ``ctx.key`` (required),
    drawn per id block (:func:`poisson_counts`), so unit ``i``'s stream is a
    function of ``(seed, name, i // GID_BLOCK, i % GID_BLOCK, step, trial)`` per the
    RNG contract (D-g).
    Emits at step ``n`` iff ``t_min < n <= t_max``.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainunit as u
        >>> from brainspike._kernel import StepContext
        >>> from brainspike._kernel.models import poisson_generator
        >>> pg = poisson_generator(1000, params=dict(rate=10000.0 * u.Hz))
        >>> ctx = StepContext.create(0.1 * u.ms, 0.1 * u.ms, step=1, key=jax.random.key(0))
        >>> counts = pg.update(ctx)['spike']
        >>> 0.8 < float(counts.mean()) < 1.2
        True
    """

    params = Params(rate=Param(0.0 * u.Hz), **_WINDOW)
    states = States(spike=Init(0.0))
    outputs = Ports(spike=Event(max_multiplicity=None))

    def output_states(self) -> dict[str, str]:
        return {'spike': 'spike'}

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Draw this step's counts (zero outside the activity window)."""
        if ctx.key is None:
            raise ValueError('poisson_generator needs ctx.key (a per-population, per-step key)')
        dtype = jnp.asarray(s.spike).dtype
        dt = _dt_ms(ctx)
        t_min, t_max = device_window(p, dt)
        n = jnp.asarray(ctx.step).astype(t_min.dtype)
        lam = jnp.maximum(to_mantissa(p.rate, u.kHz) * dt, 0.0)
        counts = poisson_counts(ctx.key, lam, (self.n,))
        active = (t_min < n) & (n <= t_max)
        spike = jnp.where(active, counts, 0).astype(dtype)
        s = s.replace(spike=spike)
        return s, {'spike': spike}


def _pad(trains: Sequence[np.ndarray], fill: float) -> np.ndarray:
    k = max((len(t) for t in trains), default=0)
    out = np.full((len(trains), k), fill, dtype=np.float64)
    for i, t in enumerate(trains):
        out[i, :len(t)] = t
    return out


def _times_ms(x: Any) -> np.ndarray:
    if isinstance(x, (list, tuple)):
        return np.asarray([to_mantissa(v, u.ms) for v in x], dtype=np.float64).reshape(-1)
    return np.asarray(to_mantissa(x, u.ms), dtype=np.float64)


def _layout(x: Any, conv: Any, n: int, what: str) -> tuple[list, bool]:
    """Split ``x`` into per-unit 1-D trains; returns ``(trains, per_unit)``."""
    if isinstance(x, (list, tuple)) and any(
            isinstance(t, (list, tuple)) or np.ndim(u.get_mantissa(t)) >= 1 for t in x):
        trains, per_unit = [np.asarray(conv(t)).reshape(-1) for t in x], True
    else:
        a = np.asarray(conv(x))
        if a.ndim > 2:
            raise ValueError(f'{what} must be 1-D (shared) or (n, K) / ragged per unit')
        trains, per_unit = (list(a), True) if a.ndim == 2 else ([a.reshape(-1)] * n, False)
    if per_unit and len(trains) != n:
        raise ValueError(f'{what} has {len(trains)} trains for {n} units')
    return trains, per_unit


def _spike_table(n: int, times: Any, mult: Any) -> tuple[Any, np.ndarray]:
    """Padded ``(n, K)`` spike times (inf-padded, ms) and multiplicities (0-padded)."""
    if times is None:
        if mult is not None:
            raise ValueError('spike_multiplicities given without spike_times')
        return np.zeros((n, 0)) * u.ms, np.zeros((n, 0))
    trains, _ = _layout(times, _times_ms, n, 'spike_times')
    if mult is None:
        mults = [np.ones_like(t) for t in trains]
    else:
        mults, _ = _layout(mult, lambda m: np.asarray(m, np.float64), n, 'spike_multiplicities')
        for t, m in zip(trains, mults):
            if m.shape != t.shape:
                raise ValueError(f'spike_multiplicities shape {m.shape} != spike_times shape {t.shape}')
            if np.any(m < 0) or np.any(m != np.round(m)):
                raise ValueError('spike_multiplicities must be non-negative integers')
    return _pad(trains, np.inf) * u.ms, _pad(mults, 0.0)


class spike_generator(Device):
    r"""Spikes at given times, with multiplicity (NEST ``spike_generator``).

    Duplicate times and ``spike_multiplicities`` add up: the step's output is the
    total count (e.g. ``[1, 1, 1] ms`` gives a count of 3 in the step ending at
    1 ms -- NEST delivers three events).

    Parameters
    ----------
    n : int
        Number of units.
    params : mapping, optional
        ``spike_times`` -- one train shared by all units (1-D Quantity or list of
        times) or one train per unit (list of ``n`` 1-D trains, ragged allowed,
        or a 2-D ``(n, K)`` Quantity); ``spike_multiplicities`` -- integer counts
        of the same layout (default 1 each); ``start`` (0 ms), ``stop`` (inf),
        ``origin`` (0 ms).
    **kwargs
        ``precision``, ``name``.

    Raises
    ------
    ValueError
        On inconsistent train/multiplicity shapes or non-integer multiplicities.

    Notes
    -----
    A time ``T`` is rounded up to the grid (stamp ``S = steps(origin + T)``)
    and emitted at step ``S - 1`` iff ``t_min < S <= t_max``; times with
    ``S <= 0`` never fire (NEST drops them). Output ``spike`` -- ``event`` with
    ``max_multiplicity=None`` (exact count payload path).

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainspike._kernel import StepContext
        >>> from brainspike._kernel.models import spike_generator
        >>> sg = spike_generator(1, params=dict(spike_times=[1.0, 1.0, 1.0] * u.ms))
        >>> ctx = StepContext.create(0.9 * u.ms, 0.1 * u.ms, step=9)
        >>> sg.update(ctx)['spike'].tolist()
        [3.0]
    """

    params = Params(spike_times=Param(None, u.ms, per_neuron=False),
                    spike_multiplicities=Param(None, None, per_neuron=False), **_WINDOW)
    states = States(spike=Init(0.0))
    outputs = Ports(spike=Event(max_multiplicity=None))

    def __init__(self, n: int, params: Mapping[str, Any] | None = None,
                 init: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        params = dict(params or {})
        times, mult = _spike_table(int(n), params.pop('spike_times', None),
                                   params.pop('spike_multiplicities', None))
        params.update(spike_times=times, spike_multiplicities=mult)
        super().__init__(n, params, init, **kwargs)

    def output_states(self) -> dict[str, str]:
        return {'spike': 'spike'}

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Count the spikes whose stamp ends this step."""
        dtype = jnp.asarray(s.spike).dtype
        dt = _dt_ms(ctx)
        t_min, t_max = device_window(p, dt)
        origin = to_mantissa(p.origin, u.ms)[:, None]
        stamps = grid_steps(origin + to_mantissa(p.spike_times, u.ms), dt)  # (n, K)
        n = jnp.asarray(ctx.step).astype(stamps.dtype)
        hit = (stamps - 1 == n) & (t_min[:, None] < stamps) & (stamps <= t_max[:, None])
        spike = jnp.sum(jnp.where(hit, jnp.asarray(p.spike_multiplicities), 0), axis=-1)
        spike = spike.astype(dtype)
        s = s.replace(spike=spike)
        return s, {'spike': spike}


class dc_generator(Device):
    r"""Constant current inside an activity window (NEST ``dc_generator``).

    Parameters
    ----------
    n : int
        Number of units.
    params : mapping, optional
        ``amplitude`` (pA, default 0), ``start`` (0 ms), ``stop`` (inf),
        ``origin`` (0 ms).
    **kwargs
        ``precision``, ``name``.

    Notes
    -----
    Output ``I`` -- ``current`` (pA): ``amplitude`` at step ``n`` iff
    ``t_min < n + 2 <= t_max`` (NEST's current-generator shift), else 0. Sent
    with a delay of one step to a neuron whose ``current`` port is latched (NEST
    models), it acts during the same steps as in NEST. The receiving port decides
    the latch (decision D-a): into a ``latch=False`` port it acts one step earlier.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainspike._kernel import StepContext
        >>> from brainspike._kernel.models import dc_generator
        >>> dc = dc_generator(1, params=dict(amplitude=100.0 * u.pA, start=1.0 * u.ms))
        >>> [float(dc.update(StepContext.create(k * 0.1 * u.ms, 0.1 * u.ms, step=k))['I'][0].to_decimal(u.pA))
        ...  for k in (7, 8, 9)]
        [0.0, 0.0, 100.0]
    """

    params = Params(amplitude=Param(0.0 * u.pA), **_WINDOW)
    states = States(I=Init(0.0 * u.pA))
    outputs = Ports(I=Current(u.pA))

    def output_states(self) -> dict[str, str]:
        return {'I': 'I'}

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Output ``amplitude`` inside the window."""
        dt = _dt_ms(ctx)
        t_min, t_max = device_window(p, dt)
        m = jnp.asarray(ctx.step).astype(t_min.dtype) + 2
        active = (t_min < m) & (m <= t_max)
        I = u.math.where(active, p.amplitude, 0.0 * u.pA)
        s = s.replace(I=I)
        return s, {'I': s.I}
