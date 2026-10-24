# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Network delivery of ``timed_event`` payloads (K0 task T7).

Delays are whole steps rounded to the grid (NEST); the sub-step **offset** of every
event is preserved. Conventions: ``docs/specs/2026-10-22-k0-t7-precise-timing.md``.

* :class:`TimedRing` -- pre-side ring of one timed source port: ``(L, n, K)`` offsets
  (ms from the step start, ``inf`` in empty slots) and weights (multiplicities).
* :class:`TimedDelivery` -- compiled delivery of one projection from a timed source:
  per-edge gather of the source payload emitted ``d_e`` steps ago (or this step for
  ``d_e = 0``), weighted by the edge weight, gathered per target through a padded
  in-edge table and packed into the target port's capacity (sorted at the receiver;
  overflow counted). Into an ``event`` port the offsets are dropped and the counts
  delivered on the grid (NEST).
* Grid ``event`` sources feeding a timed port are converted by :func:`grid_to_timed`
  to one event per target at offset ``dt`` (the NEST stamp of a grid spike).
* :func:`accumulate` merges the payloads of several projections into one port;
  :func:`check_capacities` is the compile-time worst-case capacity check.

All runtime functions are pure, ``jit``/``vmap``-able and work on mantissas
(offsets in ms; weights in the projection's weight unit).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.ports import Event, PortError, TimedEvent, TimedEventPayload, check_capacity
from brainspike._kernel.units import from_mantissa, to_mantissa

__all__ = [
    'TimedRing',
    'TimedDelivery',
    'payload_mantissa',
    'payload_from_mantissa',
    'grid_to_timed',
    'accumulate',
    'with_weight_unit',
    'check_capacities',
]


def payload_mantissa(p: TimedEventPayload, dtype: Any) -> dict:
    """``{'offsets', 'weights'}`` mantissas of a source payload (``inf`` / 0 in empty slots)."""
    valid = jnp.asarray(p.valid, bool)
    off = to_mantissa(p.offsets, u.ms).astype(dtype)
    w = jnp.asarray(u.get_mantissa(p.weights)).astype(dtype)
    return {'offsets': jnp.where(valid, off, jnp.inf), 'weights': jnp.where(valid, w, 0)}


def payload_from_mantissa(m: dict, unit: Any = None) -> TimedEventPayload:
    """Inverse of :func:`payload_mantissa` (``overflow`` zero)."""
    off = m['offsets']
    valid = jnp.isfinite(off)
    return TimedEventPayload(offsets=from_mantissa(off, u.ms), weights=from_mantissa(m['weights'], unit),
                             valid=valid, overflow=jnp.zeros(off.shape[:1], jnp.int32))


class TimedRing:
    """Pre-side ring of one ``timed_event`` source port.

    Parameters
    ----------
    n : int
        Number of sources.
    length : int
        Slots ``L`` (largest delay ``>= 1`` read from it).
    capacity : int
        Events per source and step ``K``.
    dtype : dtype
        Float dtype.

    Notes
    -----
    Same interface as :class:`~brainspike._kernel.delays.PreRing` (``init``,
    ``write``, ``nbytes``) with a ``{'offsets', 'weights'}`` pytree as buffer.
    """

    def __init__(self, n: int, length: int, capacity: int, dtype: Any) -> None:
        if int(length) < 1:
            raise ValueError(f'ring length must be >= 1, got {length}')
        self.n = int(n)
        self.length = int(length)
        self.capacity = int(capacity)
        self.dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return (f'TimedRing(n={self.n}, length={self.length}, capacity={self.capacity}, '
                f'dtype={self.dtype.name})')

    def init(self) -> dict:
        """Empty buffer (no events before step 0)."""
        shape = (self.length, self.n, self.capacity)
        return {'offsets': jnp.full(shape, jnp.inf, self.dtype), 'weights': jnp.zeros(shape, self.dtype)}

    def write(self, buf: dict, step: Any, value: dict) -> dict:
        """Store the payload emitted at ``step`` in slot ``step % L``."""
        i = jnp.remainder(jnp.asarray(step, jnp.int32), self.length)
        return {k: jax.lax.dynamic_update_index_in_dim(buf[k], jnp.asarray(value[k], self.dtype), i, axis=0)
                for k in ('offsets', 'weights')}

    def read_edges(self, buf: dict, step: Any, delays: np.ndarray, cols: np.ndarray) -> tuple:
        """Payload rows ``cols[e]`` emitted at ``step - delays[e]`` (``1 <= delays <= L``).

        Returns
        -------
        offsets, weights : jax.Array
            Shape ``(E, K)``.
        """
        delays = np.asarray(delays, np.int64)
        if delays.size and (delays.min() < 1 or delays.max() > self.length):
            raise ValueError(f'delays outside [1, {self.length}]')
        rows = jnp.remainder(jnp.asarray(step, jnp.int32) - jnp.asarray(delays, jnp.int32), self.length)
        c = jnp.asarray(cols, jnp.int32)
        return buf['offsets'][rows, c], buf['weights'][rows, c]

    def nbytes(self) -> int:
        """Device memory of the buffer (one trial)."""
        return 2 * self.length * self.n * self.capacity * self.dtype.itemsize


@dataclass
class _SourceSegment:
    pop: int            # population index
    port: str           # output port
    edges: np.ndarray   # positions (EdgeSet order) of the edges leaving this segment
    cols: np.ndarray    # local source index of each edge
    delays: np.ndarray  # delay (steps) of each edge
    capacity: int       # source capacity K of the population's port


class TimedDelivery:
    """Compiled delivery of one projection whose source port is ``timed_event``.

    Parameters
    ----------
    segments : list
        Per pre-view segment: population, port, edge positions, local columns,
        delays and source capacity.
    post_index : numpy.ndarray
        Post position (in the post view) of every edge, ``(E,)``.
    n_post : int
        Size of the post view.
    target : {'timed', 'event'}
        Kind of the destination port.
    capacity : int
        Destination capacity (``target == 'timed'``).
    """

    def __init__(self, segments: list, post_index: np.ndarray, n_post: int, target: str,
                 capacity: int) -> None:
        self.segments = segments
        self.post_index = np.asarray(post_index, np.int64)
        self.n_post = int(n_post)
        self.n_edges = int(self.post_index.size)
        self.target = target
        self.capacity = int(capacity)
        self.src_capacity = max((s.capacity for s in segments), default=1)
        indeg = np.bincount(self.post_index, minlength=self.n_post) if self.n_edges else \
            np.zeros((self.n_post,), np.int64)
        self.indegree = indeg
        cap_e = np.zeros((self.n_edges,), np.int64)
        for s in segments:
            cap_e[s.edges] = s.capacity
        #: worst-case events per target and step (sum of the source capacities of its in-edges)
        self.max_events = (np.bincount(self.post_index, weights=cap_e, minlength=self.n_post).astype(np.int64)
                           if self.n_edges else np.zeros((self.n_post,), np.int64))
        width = int(indeg.max()) if indeg.size else 0
        table = np.full((self.n_post, width), self.n_edges, np.int64)  # sentinel = E
        order = np.argsort(self.post_index, kind='stable')
        starts = np.concatenate([[0], np.cumsum(indeg)[:-1]]) if indeg.size else np.zeros(0, np.int64)
        rank = np.arange(self.n_edges) - np.repeat(starts, indeg)
        table[self.post_index[order], rank] = order
        self.table = table

    @classmethod
    def build(cls, net: Any, proj: Any, es: Any, edge_steps: np.ndarray) -> 'TimedDelivery':
        """Build from a projection declaration and its realised edges (``net``: the Network)."""
        pre_index = np.asarray(es.pre_index, np.int64)
        segments = []
        for seg in proj.pre.segments:
            sel = np.nonzero((pre_index >= seg.pos.start) & (pre_index < seg.pos.stop))[0]
            local = seg.local_array()[pre_index[sel] - seg.pos.start]
            kind = net.populations[seg.pop].model.outputs[proj.source_port]
            segments.append(_SourceSegment(seg.pop, proj.source_port, sel, local,
                                           np.asarray(edge_steps, np.int64)[sel], kind.capacity))
        target = 'timed' if isinstance(proj.dst_kind, TimedEvent) else 'event'
        cap = proj.dst_kind.capacity if target == 'timed' else 1
        return cls(segments, np.asarray(es.post_index), es.n_post, target, cap)

    def ring_lengths(self) -> dict:
        """``{(population, port): largest delay >= 1}`` this projection reads."""
        out: dict[tuple[int, str], int] = {}
        for s in self.segments:
            if s.delays.size and s.delays.max() >= 1:
                k = (s.pop, s.port)
                out[k] = max(out.get(k, 0), int(s.delays.max()))
        return out

    def _edge_payload(self, rings: dict, ring_objs: dict, cur: dict, step: Any, dtype: Any) -> tuple:
        K = self.src_capacity
        off = jnp.full((self.n_edges, K), jnp.inf, dtype)
        w = jnp.zeros((self.n_edges, K), dtype)
        for s in self.segments:
            if s.edges.size == 0:
                continue
            zero = s.delays == 0
            now = cur[s.pop][s.port] if zero.any() else None  # only delay 0 reads this step
            if zero.all():
                assert now is not None  # non-empty segment: all() implies any()
                o = now['offsets'][jnp.asarray(s.cols, jnp.int32)]
                ww = now['weights'][jnp.asarray(s.cols, jnp.int32)]
            else:
                ring = ring_objs[(s.pop, s.port)]
                o, ww = ring.read_edges(rings[(s.pop, s.port)], step, np.where(zero, 1, s.delays), s.cols)
                if zero.any():
                    assert now is not None
                    z = jnp.asarray(zero)[:, None]
                    c = jnp.asarray(s.cols, jnp.int32)
                    o = jnp.where(z, now['offsets'][c], o)
                    ww = jnp.where(z, now['weights'][c], ww)
            if s.capacity < K:
                pad = ((0, 0), (0, K - s.capacity))
                o = jnp.pad(o, pad, constant_values=jnp.inf)
                ww = jnp.pad(ww, pad)
            if len(self.segments) == 1 and s.edges.size == self.n_edges:
                off, w = o.astype(dtype), ww.astype(dtype)
            else:
                pos = jnp.asarray(s.edges, jnp.int32)
                off = off.at[pos].set(o.astype(dtype))
                w = w.at[pos].set(ww.astype(dtype))
        return off, w

    def deliver(self, w_edges: Any, rings: dict, ring_objs: dict, cur: dict, step: Any,
                dtype: Any) -> Any:
        """Input of this step for the post view.

        Parameters
        ----------
        w_edges : jax.Array
            Edge weights ``(E,)`` (mantissas, EdgeSet order).
        rings : dict
            ``{(population, port): ring buffer}`` state.
        ring_objs : dict
            ``{(population, port): TimedRing}``.
        cur : dict
            This step's outputs ``{population: {port: mantissa payload}}``.
        step : jax.Array
            Global step index.
        dtype : dtype
            Float dtype.

        Returns
        -------
        TimedEventPayload or jax.Array
            ``(n_post, capacity)`` payload (weights as mantissas) for a timed target,
            ``(n_post,)`` summed counts × weights for an event target.
        """
        off, w_src = self._edge_payload(rings, ring_objs, cur, step, dtype)
        valid = jnp.isfinite(off)
        w_e = jnp.asarray(w_edges, dtype)
        if self.target == 'event':
            cnt = jnp.sum(jnp.where(valid, w_src, 0), axis=1) * w_e
            return jax.ops.segment_sum(cnt, jnp.asarray(self.post_index, jnp.int32),
                                       num_segments=self.n_post)
        w = w_src * w_e[:, None]
        off = jnp.concatenate([off, jnp.full((1, off.shape[1]), jnp.inf, dtype)])
        w = jnp.concatenate([w, jnp.zeros((1, w.shape[1]), dtype)])
        t = jnp.asarray(self.table, jnp.int32)
        off_t = off[t].reshape(self.n_post, -1)
        w_t = w[t].reshape(self.n_post, -1)
        valid_t = jnp.isfinite(off_t)
        return TimedEventPayload.pack(from_mantissa(off_t, u.ms), w_t, valid_t, self.capacity)


def grid_to_timed(y: Any, dt_ms: float, capacity: int, dtype: Any) -> TimedEventPayload:
    """One event per target at offset ``dt`` carrying the summed grid input ``y``.

    Targets with ``y == 0`` receive no event.
    """
    y = jnp.asarray(y, dtype)
    n = y.shape[0]
    valid = jnp.zeros((n, capacity), bool).at[:, 0].set(y != 0)
    off = jnp.where(valid, jnp.asarray(dt_ms, dtype), jnp.inf)
    w = jnp.zeros((n, capacity), dtype).at[:, 0].set(y)
    return TimedEventPayload(offsets=from_mantissa(off, u.ms), weights=jnp.where(valid, w, 0),
                             valid=valid, overflow=jnp.zeros((n,), jnp.int32))


def _rows(p: TimedEventPayload, rows: Any) -> TimedEventPayload:
    return TimedEventPayload(offsets=p.offsets[rows], weights=p.weights[rows], valid=p.valid[rows],
                             overflow=p.overflow[rows])


def accumulate(acc: TimedEventPayload | None, y: Any, seg: Any, n: int, scale: float,
               capacity: int, dt_ms: float, dtype: Any) -> TimedEventPayload:
    """Add the rows of one projection's delivery for one post segment to a port's payload.

    Parameters
    ----------
    acc : TimedEventPayload or None
        Payload accumulated so far for the population's port (``(n, capacity)``).
    y : TimedEventPayload or jax.Array
        Delivery of the projection for its whole post view (a grid ``(n_post,)``
        input is converted with :func:`grid_to_timed`).
    seg : Segment
        Post segment (view positions ``seg.pos``, population locals ``seg.local``).
    n : int
        Population size.
    scale : float
        Weight-unit conversion factor into the port unit.
    capacity : int
        Port capacity.
    dt_ms : float
        Step size (ms).
    dtype : dtype
        Float dtype.

    Returns
    -------
    TimedEventPayload
        ``(n, capacity)``, weights as mantissas in the port unit.
    """
    if not isinstance(y, TimedEventPayload):
        y = grid_to_timed(y, dt_ms, capacity, dtype)
    part = _rows(y, seg.pos)
    w = jnp.asarray(u.get_mantissa(part.weights), dtype)
    if scale != 1.0:
        w = w * scale
    off = to_mantissa(part.offsets, u.ms).astype(dtype)
    full = seg.full and seg.local == slice(0, n)
    if not full:
        idx = seg.local if seg.full else jnp.asarray(seg.local, jnp.int32)
        k = part.valid.shape[1]
        off = jnp.full((n, k), jnp.inf, dtype).at[idx].set(off)
        w = jnp.zeros((n, k), dtype).at[idx].set(w)
        valid = jnp.zeros((n, k), bool).at[idx].set(part.valid)
        ovf = jnp.zeros((n,), jnp.int32).at[idx].set(part.overflow)
    else:
        valid, ovf = part.valid, part.overflow
    new = TimedEventPayload.pack(from_mantissa(off, u.ms), w, valid, capacity, overflow=ovf)
    return new if acc is None else acc.merge(new)


def with_weight_unit(p: TimedEventPayload, unit: Any) -> TimedEventPayload:
    """``p`` with its weight mantissas wrapped in ``unit``."""
    return TimedEventPayload(offsets=p.offsets, weights=from_mantissa(u.get_mantissa(p.weights), unit),
                             valid=p.valid, overflow=p.overflow)


def check_capacities(plan: Any) -> None:
    """Compile-time worst-case check of every timed input port.

    For each population and ``timed_event`` input port the worst case per target is
    the sum over its incoming projections of in-degree × source capacity (timed
    sources) or 1 (grid sources, converted to one event per step). If it exceeds the
    port capacity and any of those projections was connected with
    ``strict_capacity=True``, a :class:`~brainspike._kernel.ports.PortError` is
    raised; otherwise the overflow is counted at run time.

    Raises
    ------
    PortError
        See above.
    """
    worst: dict = {}
    strict: dict = {}
    for pp in plan.projections:
        prj = pp.proj
        if pp.placement == 'none' or not isinstance(prj.dst_kind, TimedEvent):
            continue
        if pp.timed is not None:
            per_post = pp.timed.max_events
        else:
            per_post = (np.bincount(np.asarray(pp.edges.post_index), minlength=pp.edges.n_post) > 0
                        ).astype(np.int64)
        for seg in prj.post.segments:
            key = (seg.pop, prj.port)
            pop_n = plan.populations[seg.pop].pop.n
            arr = worst.setdefault(key, np.zeros((pop_n,), np.int64))
            np.add.at(arr, seg.local_array(), per_post[seg.pos])
            strict[key] = strict.get(key, False) or prj.strict_capacity
    for (a, port), arr in worst.items():
        if not arr.size:
            continue
        pplan = plan.populations[a]
        kind = pplan.pop.model.inputs[port]
        try:
            check_capacity(kind, int(arr.max()), strict=strict[(a, port)],
                           where=f'{pplan.name}.{port} (all incoming projections)')
        except PortError as e:
            raise PortError(f'{e} -- connect with strict_capacity=False to count overflow instead, '
                            f'or use {pplan.pop.model.__name__}.with_capacity(...)') from None

