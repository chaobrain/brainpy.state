# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Integer-step delays and the two delay-buffer layouts (K0 spec §4.4).

Delays of ``event`` (and value) projections are whole steps, rounded to the grid as
NEST does (``Time::delay_ms_to_steps`` = ``floor(ms * steps_per_ms + 0.5)``, see
:func:`delay_steps`). A projection's per-edge delays are classified by
:func:`classify_delays` into one of three layouts, and the compiler places one of two
buffers:

* **pre-side ring** (:class:`PreRing`) -- one ``(L, n_src)`` ring per source port,
  shared by every projection leaving it, for **scalar** and **per-source** delays.
  The step-``n`` output is written to slot ``n % L`` at the end of step ``n``;
  delivery with delay ``d >= 1`` reads slot ``(n - d) % L``. ``L`` is the largest
  delay read from the ring, so the slot about to be overwritten is read first.
* **post-side arrival ring** (:class:`PostRing`) -- one ``(L, n_post)`` ring per
  projection with **per-edge** delays. At the end of step ``n`` the edges with delay
  ``k >= 1`` deposit ``payload @ W_k`` into slot ``(n + k) % L`` (after slot
  ``n % L``, consumed during step ``n``, has been cleared); the target reads slot
  ``n % L``. Edges are grouped by delay value, each group with its own storage, so
  delivery stays event-driven.

Delay ``0`` never touches a buffer: the target reads the source's output of the same
step, which the compiler's topological schedule makes available (decision D4a).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.units import check_unit

__all__ = [
    'DELAY_LAYOUTS',
    'delay_steps',
    'DelayLayout',
    'classify_delays',
    'PreRing',
    'PostRing',
]

#: Layouts of a projection's delays.
DELAY_LAYOUTS = ('scalar', 'per_source', 'per_edge')


def delay_steps(delay: Any, dt: Any) -> np.ndarray:
    """Delays in whole steps with NEST's rounding (host side).

    NEST converts a delay with ``Time::delay_ms_to_steps(ms) =
    ld_round(ms * STEPS_PER_MS)`` (round half up), unlike spike stamps, which round
    up. This function reproduces it in float64: ``floor(delay * (1 / dt) + 0.5)``.
    Values stored in float32 (x64 disabled) that lie within float32 resolution of
    a half step are taken as exact halves and rounded up, as NEST does for the
    decimal the user wrote (``0.35 ms`` is ``0.3499999940`` in float32).

    Parameters
    ----------
    delay : brainunit.Quantity
        Delay(s) (time dimension), any shape; must be finite and non-negative.
    dt : brainunit.Quantity
        Step size.

    Returns
    -------
    numpy.ndarray of int64
        Delay in steps, same shape as ``delay``.

    Raises
    ------
    UnitMismatchError
        If ``delay`` is not a time.
    ValueError
        On negative or non-finite delays.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.delays import delay_steps
        >>> delay_steps([0.0, 0.04, 0.05, 0.15, 1.5] * u.ms, 0.1 * u.ms).tolist()
        [0, 0, 1, 2, 15]
    """
    check_unit(delay, u.ms, 'delay')
    check_unit(dt, u.ms, 'dt')
    raw = np.asarray(u.Quantity(delay).to_decimal(u.ms))
    d = raw.astype(np.float64)
    if d.size and (not np.all(np.isfinite(d)) or np.min(d) < 0):
        raise ValueError(f'delays must be finite and non-negative, got min {np.min(d)} ms')
    steps_per_ms = 1.0 / float(np.asarray(u.Quantity(dt).to_decimal(u.ms)))
    k = np.rint(steps_per_ms)
    if k >= 1 and abs(steps_per_ms - k) <= 1e-6 * k:
        steps_per_ms = k  # NEST: an integer number of steps per ms
    y = d * steps_per_ms + 0.5
    out = np.floor(y)
    # Low-precision inputs (float32 when x64 is off) cannot hold decimals such as
    # 0.35 ms; NEST receives the float64 decimal (3.5 steps) and rounds half up. Treat
    # values within the input's resolution of a half step as exact halves.
    eps = np.finfo(raw.dtype).eps if np.issubdtype(raw.dtype, np.floating) else 0.0
    if eps > np.finfo(np.float64).eps:
        r = np.rint(y)
        out = np.where(np.abs(y - r) <= 8 * eps * np.maximum(y, 1.0), r, out)
    return out.astype(np.int64)


@dataclass(frozen=True)
class DelayLayout:
    """Classified delays of one projection.

    Attributes
    ----------
    kind : {'scalar', 'per_source', 'per_edge'}
        Layout.
    steps : int or numpy.ndarray
        ``int`` (scalar), ``(n_pre,)`` (per source) or ``(E,)`` (per edge, in
        :class:`~brainpy_state._kernel.connectivity.EdgeSet` order).
    min, max : int
        Smallest / largest delay over the edges (``0`` for a projection without
        edges).
    """

    kind: str
    steps: Any
    min: int
    max: int

    @property
    def values(self) -> np.ndarray:
        """Distinct delays (sorted) present on the edges."""
        return np.unique(np.asarray(self.steps)).astype(np.int64)

    @property
    def has_zero(self) -> bool:
        """Whether some edge has a zero delay (same-step delivery)."""
        return self.min == 0


def classify_delays(edge_steps: np.ndarray, pre_index: np.ndarray, n_pre: int,
                    force: str | None = None) -> DelayLayout:
    """Pick the layout of a projection's delays.

    Parameters
    ----------
    edge_steps : numpy.ndarray of int, shape (E,)
        Per-edge delays in steps (EdgeSet order).
    pre_index : numpy.ndarray of int, shape (E,)
        Local pre index of every edge.
    n_pre : int
        Size of the pre view.
    force : {'scalar', 'per_source', 'per_edge'}, optional
        Required layout; ``'per_edge'`` is always possible, ``'scalar'`` and
        ``'per_source'`` raise if the delays do not fit.

    Returns
    -------
    DelayLayout
        ``'scalar'`` if all edges share one delay, ``'per_source'`` if the delay
        depends only on the pre neuron, else ``'per_edge'``. Pre neurons without
        edges get the projection's minimum delay in the per-source vector.

    Raises
    ------
    ValueError
        If ``force`` names a layout the delays do not fit, or is unknown.
    """
    if force is not None and force not in DELAY_LAYOUTS:
        raise ValueError(f'unknown delay layout {force!r}; expected one of {DELAY_LAYOUTS}')
    e = np.asarray(edge_steps, np.int64).reshape(-1)
    pi = np.asarray(pre_index, np.int64).reshape(-1)
    if e.size == 0:
        kind = 'per_edge' if force == 'per_edge' else (force or 'scalar')
        steps = (np.zeros((0,), np.int64) if kind == 'per_edge'
                 else np.zeros((n_pre,), np.int64) if kind == 'per_source' else 0)
        return DelayLayout(kind, steps, 0, 0)
    lo, hi = int(e.min()), int(e.max())
    if force == 'per_edge':
        return DelayLayout('per_edge', e, lo, hi)
    if lo == hi and force in (None, 'scalar'):
        return DelayLayout('scalar', lo, lo, hi)
    if force == 'scalar':
        raise ValueError(f'delays are not uniform ({lo}..{hi} steps); a scalar layout is impossible')
    dmin = np.full((n_pre,), np.iinfo(np.int64).max, np.int64)
    dmax = np.full((n_pre,), -1, np.int64)
    np.minimum.at(dmin, pi, e)
    np.maximum.at(dmax, pi, e)
    used = dmax >= 0
    if np.array_equal(dmin[used], dmax[used]):
        return DelayLayout('per_source', np.where(used, dmin, lo), lo, hi)
    if force == 'per_source':
        raise ValueError('delays differ between edges of the same source; use the per-edge layout')
    return DelayLayout('per_edge', e, lo, hi)


def _index(step: Any, length: int) -> jax.Array:
    return jnp.remainder(jnp.asarray(step, jnp.int32), length)


class PreRing:
    """Pre-side ring buffer of one source port: ``(L, n)`` past outputs.

    Parameters
    ----------
    n : int
        Number of sources.
    length : int
        Number of slots ``L`` (the largest delay ``>= 1`` read from it).
    dtype : dtype
        Payload dtype.

    Notes
    -----
    All methods are pure: they take and return the buffer array (shape ``(L, n)``;
    under ``vmap`` the batch axis is added by JAX).
    """

    def __init__(self, n: int, length: int, dtype: Any) -> None:
        if int(length) < 1:
            raise ValueError(f'ring length must be >= 1, got {length}')
        self.n = int(n)
        self.length = int(length)
        self.dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return f'PreRing(n={self.n}, length={self.length}, dtype={self.dtype.name})'

    def init(self) -> jax.Array:
        """Zero buffer (no output before step 0)."""
        return jnp.zeros((self.length, self.n), self.dtype)

    def write(self, buf: jax.Array, step: Any, value: jax.Array) -> jax.Array:
        """Store the output of ``step`` in slot ``step % L``."""
        value = jnp.asarray(value, self.dtype)[None]
        return jax.lax.dynamic_update_slice_in_dim(buf, value, _index(step, self.length), axis=0)

    def read(self, buf: jax.Array, step: Any, delay: int) -> jax.Array:
        """Output emitted at ``step - delay`` (``1 <= delay <= L``), shape ``(n,)``."""
        if not 1 <= int(delay) <= self.length:
            raise ValueError(f'delay {delay} outside [1, {self.length}]')
        i = _index(jnp.asarray(step, jnp.int32) - int(delay), self.length)
        return jax.lax.dynamic_index_in_dim(buf, i, axis=0, keepdims=False)

    def read_per_source(self, buf: jax.Array, step: Any, delays: np.ndarray,
                        cols: np.ndarray) -> jax.Array:
        """``buf[(step - delays[i]) % L, cols[i]]`` for every ``i`` (delays ``>= 1``)."""
        delays = np.asarray(delays, np.int64)
        if delays.size and (delays.min() < 1 or delays.max() > self.length):
            raise ValueError(f'per-source delays outside [1, {self.length}]')
        rows = _index(jnp.asarray(step, jnp.int32) - jnp.asarray(delays, jnp.int32), self.length)
        return buf[rows, jnp.asarray(cols, jnp.int32)]

    def nbytes(self) -> int:
        """Device memory of the buffer (one trial)."""
        return self.length * self.n * self.dtype.itemsize


class PostRing:
    """Post-side arrival ring of one projection: ``(L, n_post)`` future inputs.

    Parameters
    ----------
    n : int
        Size of the post view.
    length : int
        Number of slots ``L`` (the largest per-edge delay).
    dtype : dtype
        Input dtype.
    """

    def __init__(self, n: int, length: int, dtype: Any) -> None:
        if int(length) < 1:
            raise ValueError(f'ring length must be >= 1, got {length}')
        self.n = int(n)
        self.length = int(length)
        self.dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return f'PostRing(n={self.n}, length={self.length}, dtype={self.dtype.name})'

    def init(self) -> jax.Array:
        """Zero buffer."""
        return jnp.zeros((self.length, self.n), self.dtype)

    def read(self, buf: jax.Array, step: Any) -> jax.Array:
        """Input arriving at ``step``, shape ``(n,)``."""
        return jax.lax.dynamic_index_in_dim(buf, _index(step, self.length), axis=0, keepdims=False)

    def clear(self, buf: jax.Array, step: Any) -> jax.Array:
        """Zero slot ``step % L`` (consumed during ``step``)."""
        z = jnp.zeros((1, self.n), self.dtype)
        return jax.lax.dynamic_update_slice_in_dim(buf, z, _index(step, self.length), axis=0)

    def deposit(self, buf: jax.Array, step: Any, delay: int, value: jax.Array) -> jax.Array:
        """Add ``value`` to the input arriving at ``step + delay`` (``1 <= delay <= L``)."""
        if not 1 <= int(delay) <= self.length:
            raise ValueError(f'delay {delay} outside [1, {self.length}]')
        i = _index(jnp.asarray(step, jnp.int32) + int(delay), self.length)
        row = jax.lax.dynamic_index_in_dim(buf, i, axis=0, keepdims=True)
        row = row + jnp.asarray(value, self.dtype)[None]
        return jax.lax.dynamic_update_slice_in_dim(buf, row, i, axis=0)

    def nbytes(self) -> int:
        """Device memory of the buffer (one trial)."""
        return self.length * self.n * self.dtype.itemsize
