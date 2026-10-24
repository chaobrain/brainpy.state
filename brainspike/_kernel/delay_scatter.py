# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Single-op delivery of per-edge delays into a post-side arrival ring (K1a-4).

A projection whose edges have different delays (``d_e >= 1`` steps) deposits, at the
end of step ``n``, the contribution ``payload[pre_e] * w_e`` of **every** edge into
the arrival ring slot ``(n + d_e) mod L`` of its target, in **one** scatter-add over
the flattened ``(L, n_post)`` ring (:class:`DelayScatter`). Step ``n`` reads (and
then clears) slot ``n mod L`` (:class:`~brainspike._kernel.delays.PostRing`), so
delivery is exactly NEST's: an event emitted in step ``n`` over an edge of delay
``d`` arrives in step ``n + d``. K0 instead grouped the edges by delay value, one
storage and one matmul per distinct delay; the compiler's cost model picks between
the two (:mod:`~brainspike._kernel.compiler`).

Layout
------
Edges are kept in row (pre) order, as in :class:`~brainspike._kernel.connectivity.storage.CSR`,
and by ring offset within a row.
Each edge stores ``rel_e = (d_e mod L) * n_post + post_e``; at step ``n`` its ring
position is ``rel_e + (n mod L) * n_post`` (minus ``L * n_post`` when it wraps). The
rows are cut into **segments** of at most ``M`` consecutive edges (``M`` = the largest
out-degree, capped at twice the mean), so one hub does not inflate the work of every
other source.

Strategies
----------
* ``'events'`` (event sources): only the segments of sources with a non-zero payload
  are visited, ``chunk`` segments per iteration of a ``lax.while_loop``; each segment
  is read with a contiguous ``dynamic_slice`` and scatter-added into the ring. The
  work is proportional to the active edges (plus ``M`` padding per segment). With
  ``capacity = K > 0`` the first ``K`` active segments are handled by one loop-free
  chunk and the loop runs only when more are active (behind a ``lax.cond``): on GPUs
  every ``while`` iteration costs a device-to-host sync of its predicate (measured
  ~30 us per iteration on the remote GPU), so the common case must be loop-free.
* ``'dense'`` (value sources, small projections on accelerators): every edge
  scatter-adds ``payload[pre_e] * w_e`` each step.

Gradients
---------
The deposit is a ``jax.custom_jvp`` whose tangent is the dense, linear scatter
``dring + scatter(dpayload[pre] * w + payload[pre] * dw)``: forward- and reverse-mode
derivatives with respect to the ring, the payload (surrogate spikes) and the weights
are exact -- the JVP/VJP of the scatter itself, also for sources whose payload is zero
in this step (the event-driven primal skips them, the tangent does not).

The fused event-driven brainevent kernel requested in
`brainevent#203 <https://github.com/chaobrain/brainevent/issues/203>`_ is the future
fast path behind the same interface.
"""
from __future__ import annotations

import functools
import math
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.connectivity.edgeset import EdgeSet
from brainspike._kernel.connectivity.storage import (
    Storage,
    _host_index,
    _split,
    _split_edges,
    _take,
    _wrap,
)
from brainspike._kernel.units import index_dtype

__all__ = ['SCATTER_STRATEGIES', 'DelayScatter', 'segment_length']

#: Strategies of :meth:`DelayScatter.deposit`.
SCATTER_STRATEGIES = ('events', 'dense')


def segment_length(counts: np.ndarray) -> int:
    """Segment length ``M`` for per-source edge counts (out-degrees).

    ``M`` is the largest out-degree, capped at twice the mean out-degree of the
    sources that have edges (and at least 1), so that a few hubs are split into
    several segments instead of padding every source to the hub's degree.

    Parameters
    ----------
    counts : numpy.ndarray of int
        Out-degree of every source.

    Returns
    -------
    int

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainspike._kernel.delay_scatter import segment_length
        >>> segment_length(np.array([3, 4, 5])), segment_length(np.array([1, 1, 1, 100]))
        (5, 52)
    """
    c = np.asarray(counts, np.int64)
    used = c[c > 0]
    if used.size == 0:
        return 1
    mean = used.sum() / used.size
    return int(max(1, min(int(used.max()), 2 * math.ceil(mean))))


class DelayScatter(Storage):
    """Edges with per-edge delays ``>= 1``, delivered by one scatter into the arrival ring.

    Parameters
    ----------
    edges : EdgeSet
        The edges (only their pre/post indices are used).
    steps : numpy.ndarray of int, shape (E,)
        Delay of every edge in steps (``EdgeSet`` order), ``1 <= d <= length``.
    length : int
        Number of ring slots ``L`` (the largest delay).
    strategy : {'events', 'dense'}, default 'events'
        See the module docstring.
    chunk : int, default 1
        Segments per ``while_loop`` iteration (``'events'``).
    capacity : int, default 0
        Active segments handled without a loop (``'events'``); ``0`` loops over all.
    seg_len : int, optional
        Segment length ``M`` (default :func:`segment_length` of the out-degrees).

    Attributes
    ----------
    order : numpy.ndarray
        Permutation from storage (row-major) position to ``EdgeSet`` position; the
        native weight layout is the per-edge vector in this order.
    rel, seg_pre, seg_start, seg_len, rows : jax.Array
        Device structure (:meth:`~brainspike._kernel.connectivity.storage.Storage.arrays`);
        ``rows`` (pre index per edge) only for the ``'dense'`` strategy.
    length, n_segments, M : int
        Ring length, number of segments and segment length.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from brainspike._kernel.connectivity.edgeset import EdgeSet
        >>> from brainspike._kernel.delay_scatter import DelayScatter
        >>> es = EdgeSet.create([0, 0, 1], [0, 1, 1], np.arange(2), np.arange(10, 12))
        >>> st = DelayScatter(es, np.array([1, 3, 2]), length=3)
        >>> ring = st.deposit(jnp.zeros((3, 2)), 0, jnp.array([1.0, 1.0]),
        ...                   st.weight_from_edges(jnp.array([0.5, 2.0, 4.0])))
        >>> ring.tolist()  # slot (0 + d) % 3: d=1 -> 1, d=2 -> 2, d=3 -> 0
        [[0.0, 2.0], [0.5, 0.0], [0.0, 4.0]]
    """

    kind: ClassVar[str] = 'scatter'
    array_names: ClassVar[tuple[str, ...]] = ('rel', 'seg_pre', 'seg_start', 'seg_len', 'rows')

    def __init__(self, edges: EdgeSet, steps: Any, length: int, strategy: str = 'events',
                 chunk: int = 1, seg_len: int | None = None, capacity: int = 0) -> None:
        super().__init__(edges)
        if strategy not in SCATTER_STRATEGIES:
            raise ValueError(f'strategy must be one of {SCATTER_STRATEGIES}, got {strategy!r}')
        if int(chunk) < 1:
            raise ValueError(f'chunk must be >= 1, got {chunk}')
        if int(capacity) < 0:
            raise ValueError(f'capacity must be >= 0, got {capacity}')
        L = int(length)
        if L < 1:
            raise ValueError(f'ring length must be >= 1, got {length}')
        d = np.asarray(steps, np.int64).reshape(-1)
        if d.shape != (self.n_edges,):
            raise ValueError(f'steps must have shape ({self.n_edges},), got {d.shape}')
        if d.size and (d.min() < 1 or d.max() > L):
            raise ValueError(f'per-edge delays must lie in [1, {L}] steps, got {d.min()}..{d.max()}')
        self.strategy = strategy
        self.chunk = int(chunk)
        self.length = L
        # absolute ring positions reach 2 * L * n_post - 1 before the wrap
        pos_dt = index_dtype(2 * L * max(self.n_post, 1))
        edge_dt = index_dtype(max(self.n_edges, 1))
        rel_e = (d % L) * self.n_post + self._post
        # rows in pre order; within a row ascending ring offset (memory locality of the
        # scatter); ties keep the EdgeSet order
        order = np.lexsort((rel_e, self._pre))
        self.order = _host_index(order)
        self._edge_dt = edge_dt
        rows = self._pre[order]
        rel = rel_e[order]
        self.rel = jnp.asarray(rel.astype(pos_dt))
        counts = np.bincount(self._pre, minlength=self.n_pre).astype(np.int64)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        M = int(seg_len) if seg_len is not None else segment_length(counts)
        if M < 1:
            raise ValueError(f'seg_len must be >= 1, got {seg_len}')
        self.M = min(M, max(self.n_edges, 1))
        nseg = -(-counts // self.M)
        seg_pre = np.repeat(np.arange(self.n_pre), nseg)
        k = np.arange(seg_pre.size) - np.repeat(np.cumsum(nseg) - nseg, nseg)
        seg_start = indptr[seg_pre] + k * self.M
        seg_l = np.minimum(self.M, counts[seg_pre] - k * self.M)
        self.n_segments = int(seg_pre.size)
        self.capacity = min(int(capacity), self.n_segments)
        self.seg_pre = jnp.asarray(seg_pre.astype(index_dtype(max(self.n_pre, 1))))
        self.seg_start = jnp.asarray(seg_start.astype(edge_dt))
        self.seg_len = jnp.asarray(seg_l.astype(edge_dt))
        self.rows = jnp.asarray(rows.astype(index_dtype(max(self.n_pre, 1)))) if strategy == 'dense' else None

    def __repr__(self) -> str:
        return (f'DelayScatter(n_pre={self.n_pre}, n_post={self.n_post}, n_edges={self.n_edges}, '
                f'length={self.length}, strategy={self.strategy!r}, segments={self.n_segments}, '
                f'M={self.M}, chunk={self.chunk}, capacity={self.capacity})')

    @property
    def _order_j(self) -> jax.Array:
        return jnp.asarray(self.order.astype(self._edge_dt, copy=False))

    # ------------------------------------------------------------- weights
    def weight_from_edges(self, edge_weight: Any) -> Any:
        """Permute per-edge weights (``EdgeSet`` order) into storage (row) order."""
        self._check_edge_weight(edge_weight)
        m, unit = _split_edges(edge_weight)
        return _wrap(_take(m, self.order, lambda: self._order_j), unit)

    def weight_to_edges(self, w: Any) -> Any:
        """Inverse permutation back to ``EdgeSet`` order (exact)."""
        m, unit = _split(w)
        if m.shape != (self.n_edges,):
            raise ValueError(f'scatter weights must have shape ({self.n_edges},), got {m.shape}')
        return _wrap(jnp.zeros_like(m).at[self._order_j].set(m), unit)

    # ------------------------------------------------------------ delivery
    def forward(self, payload: Any, w: Any, path: str = 'float') -> Any:
        """Not available: the edges have different delays (use :meth:`deposit`)."""
        raise TypeError('DelayScatter delivers into an arrival ring; use deposit(ring, step, '
                        'payload, w)')

    def forward_etrace(self, payload: Any, w: Any) -> Any:
        """Not available: per-edge-delay weights cannot be routed through braintrace."""
        raise TypeError('per-edge-delay (scatter) weights cannot be applied through a braintrace '
                        'primitive; online learners reject trainable per-edge-delay projections')

    def deposit(self, ring: Any, step: Any, payload: Any, w: Any) -> jax.Array:
        """Add every edge's contribution to the ring slot of its arrival step.

        Parameters
        ----------
        ring : jax.Array, shape (L, n_post)
            Arrival ring (one trial; under ``vmap`` the batch axis is added by JAX).
        step : int or jax.Array
            Current step ``n``; edge ``e`` writes slot ``(n + d_e) % L``.
        payload : jax.Array, shape (n_pre,)
            Source payload of step ``n`` (spike counts or values).
        w : jax.Array or Quantity, shape (E,)
            Weights in the native (row) order; a Quantity's mantissa is used.

        Returns
        -------
        jax.Array, shape (L, n_post)
            The new ring.
        """
        ring = jnp.asarray(ring)
        if ring.shape != (self.length, self.n_post):
            raise ValueError(f'ring must have shape ({self.length}, {self.n_post}), got {ring.shape}')
        m, _ = _split(w)
        if m.shape != (self.n_edges,):
            raise ValueError(f'scatter weights must have shape ({self.n_edges},), got {m.shape}')
        x = jnp.asarray(payload)
        if x.shape != (self.n_pre,):
            raise ValueError(f'payload must have shape ({self.n_pre},), got {x.shape}')
        if self.n_edges == 0:
            return ring
        cfg = (self.strategy, self.length, self.n_post, self.chunk, self.M, self.n_edges,
               self.capacity)
        aux = {k: getattr(self, k) for k in self.array_names if getattr(self, k) is not None}
        n = jnp.asarray(step, jnp.int32)
        flat = _scatter(cfg, ring.reshape(-1), x.astype(ring.dtype), m.astype(ring.dtype), n, aux)
        return flat.reshape(ring.shape)

    def nbytes(self, itemsize: int = 4) -> int:
        """Device memory of the structure plus native weights of ``itemsize`` bytes."""
        n = self.n_edges * (itemsize + self.rel.dtype.itemsize)
        n += self.n_segments * (self.seg_pre.dtype.itemsize + 2 * self.seg_start.dtype.itemsize)
        return n + (0 if self.rows is None else self.rows.size * self.rows.dtype.itemsize)


# ---------------------------------------------------------------------------
# the scatter (custom JVP: event-driven primal, dense linear tangent)
# ---------------------------------------------------------------------------

def _positions(cfg: tuple, rel: jax.Array, n: jax.Array) -> jax.Array:
    L, n_post = cfg[1], cfg[2]
    ln = L * n_post
    base = (jnp.remainder(n, L) * n_post).astype(rel.dtype)
    ix = rel + base
    return jnp.where(ix >= ln, ix - ln, ix)


def _chunk(cfg: tuple, buf: jax.Array, n: jax.Array, xs: jax.Array, w: jax.Array,
           aux: dict, sel: jax.Array, ok: jax.Array) -> jax.Array:
    """Scatter-add the segments ``sel`` (lanes with ``ok``) into the flat ring."""
    M, E = cfg[4], cfg[5]
    seg_start, seg_len, rel = aux['seg_start'], aux['seg_len'], aux['rel']
    s = seg_start[sel]
    ln = seg_len[sel]
    s_eff = jnp.minimum(s, E - M)  # keep the M-window inside the arrays
    lo = s - s_eff
    jj = jnp.arange(M, dtype=seg_start.dtype)
    ws = jax.vmap(lambda s0: jax.lax.dynamic_slice(w, (s0,), (M,)))(s_eff)
    rs = jax.vmap(lambda s0: jax.lax.dynamic_slice(rel, (s0,), (M,)))(s_eff)
    valid = ok[:, None] & (jj[None] >= lo[:, None]) & (jj[None] < (lo + ln)[:, None])
    val = jnp.where(valid, xs[sel][:, None] * ws, jnp.zeros((), buf.dtype))
    ix = _positions(cfg, rs, n)
    return buf.at[ix.reshape(-1)].add(val.reshape(-1), mode='promise_in_bounds')


def _primal(cfg: tuple, flat: jax.Array, x: jax.Array, w: jax.Array, n: jax.Array,
            aux: dict) -> jax.Array:
    strategy, L, n_post, C, M, E, K = cfg
    if strategy == 'dense':
        ix = _positions(cfg, aux['rel'], n)
        return flat.at[ix].add(x[aux['rows']] * w, mode='promise_in_bounds')
    seg_pre = aux['seg_pre']
    S = seg_pre.shape[0]
    xs = x[seg_pre]
    act = xs != 0
    count = jnp.sum(act, dtype=jnp.int32)
    lanes = jnp.arange(C, dtype=jnp.int32)

    def loop(buf, k0):
        # active segments k0, k0 + 1, ... in chunks of C; padded by C entries so that
        # the dynamic_slice below never clamps (k0 + k * C < count <= S)
        idx = jnp.nonzero(act, size=S + C, fill_value=0)[0].astype(jnp.int32)

        def body(carry):
            k, b = carry
            sel = jax.lax.dynamic_slice(idx, (k,), (C,))
            return k + C, _chunk(cfg, b, n, xs, w, aux, sel, k + lanes < count)

        return jax.lax.while_loop(lambda c: c[0] < count, body, (jnp.int32(k0), buf))[1]

    if K == 0:
        return loop(flat, 0)
    # the first K active segments without a loop; the rest (rare) behind a cond
    sel = jnp.nonzero(act, size=K, fill_value=0)[0].astype(jnp.int32)
    flat = _chunk(cfg, flat, n, xs, w, aux, sel, jnp.arange(K, dtype=jnp.int32) < count)
    return jax.lax.cond(count > K, lambda b: loop(b, K), lambda b: b, flat)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _scatter(cfg: tuple, flat: jax.Array, x: jax.Array, w: jax.Array, n: jax.Array,
             aux: dict) -> jax.Array:
    # upstream: brainevent#203 -- a fused event-driven per-edge-delay delivery kernel
    # would replace this JAX reference (same arguments, same result).
    return _primal(cfg, flat, x, w, n, aux)


@_scatter.defjvp
def _scatter_jvp(cfg: tuple, primals: tuple, tangents: tuple) -> tuple:
    flat, x, w, n, aux = primals
    dflat, dx, dw = tangents[:3]
    out = _scatter(cfg, flat, x, w, n, aux)
    E = cfg[5]
    if 'rows' in aux:
        rows = aux['rows']
    else:
        rows = jnp.repeat(aux['seg_pre'], aux['seg_len'], total_repeat_length=E)
    ix = _positions(cfg, aux['rel'], n)
    dval = dx[rows] * w + x[rows] * dw
    return out, dflat.at[ix].add(dval, mode='promise_in_bounds')
