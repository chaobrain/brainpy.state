# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""The realised edge list of a projection (K0 spec §4.4).

An :class:`EdgeSet` is the single source of truth about *which* edges exist: storages
(:mod:`~brainpy_state._kernel.connectivity.storage`) are derived from it, and
introspection (:meth:`EdgeSet.get_connections`) and weight assignment
(:meth:`EdgeSet.set`) operate on it, so zero-weight edges and multapses are never lost
(goal-01 bug 6).

Edges are stored as **local** indices into the pre and post views (what storages
need) together with the views' **global** ids (what autapse decisions, introspection
and the user see).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.params import Distribution
from brainpy_state._kernel.units import (
    check_unit,
    from_mantissa,
    index_dtype,
    to_mantissa,
    unit_of,
)

__all__ = ['EdgeSet', 'Connections', 'as_gids']


def as_gids(gids: Any, name: str = 'gids') -> np.ndarray:
    """Validate a view's global ids: 1-D, integer, non-negative, no duplicates.

    Parameters
    ----------
    gids : array_like of int
        Global ids of a (possibly concatenated) population view.
    name : str
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        ``int64`` host copy.

    Raises
    ------
    ValueError
        On a non-1-D input, negative ids or duplicates (a view must not contain a
        neuron twice; overlapping concatenations must be deduplicated by the caller).
    TypeError
        On non-integer ids.
    """
    a = np.asarray(gids)
    if a.ndim != 1:
        raise ValueError(f'{name} must be 1-D, got shape {a.shape}')
    if a.size == 0:
        return a.astype(np.int64)
    if not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f'{name} must be integers, got dtype {a.dtype}')
    a = a.astype(np.int64)
    if a.min() < 0:
        raise ValueError(f'{name} must be non-negative')
    if np.unique(a).size != a.size:
        raise ValueError(f'{name} contains duplicate global ids; a view must list each neuron once')
    return a


def _gid_dtype(*arrays: np.ndarray) -> np.dtype:
    hi = max((int(a.max()) + 1 for a in arrays if a.size), default=0)
    return index_dtype(hi)


@dataclass(frozen=True)
class Connections:
    """Result of :meth:`EdgeSet.get_connections` (host-side view of selected edges).

    Attributes
    ----------
    edge : numpy.ndarray
        Positions of the selected edges in the :class:`EdgeSet` (use them with
        :meth:`EdgeSet.set` to update exactly these edges).
    source, target : numpy.ndarray
        Global ids of the edges' pre and post neurons.
    weight, delay : Quantity or array or None
        Per-edge values of the selected edges (``None`` when not assigned).
    """

    edge: np.ndarray
    source: np.ndarray
    target: np.ndarray
    weight: Any
    delay: Any

    def __len__(self) -> int:
        return int(self.edge.size)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EdgeSet:
    """Edges of one projection over global ids, with optional per-edge values.

    Build with :meth:`create` / :meth:`from_gids` (validating) or a connectivity
    rule's ``build``; the raw constructor performs no validation so JAX can rebuild
    the pytree.

    Parameters
    ----------
    pre_index, post_index : jax.Array of int, shape (E,)
        Local indices of each edge's pre / post neuron in the views.
    pre_gids, post_gids : jax.Array of int, shapes (n_pre,), (n_post,)
        Global ids of the pre and post views (concatenations allowed).
    weight : Quantity or jax.Array, shape (E,), optional
        Per-edge weights.
    delay : Quantity, shape (E,), optional
        Per-edge delays (time dimension).
    """

    pre_index: Any
    post_index: Any
    pre_gids: Any
    post_gids: Any
    weight: Any = None
    delay: Any = None

    # ------------------------------------------------------------------ build
    @classmethod
    def create(cls, pre_index: Any, post_index: Any, pre_gids: Any, post_gids: Any,
               weight: Any = None, delay: Any = None) -> 'EdgeSet':
        """Validated constructor from local indices.

        Parameters
        ----------
        pre_index, post_index : array_like of int, shape (E,)
            Local indices into ``pre_gids`` / ``post_gids``.
        pre_gids, post_gids : array_like of int
            Global ids of the views (see :func:`as_gids`).
        weight, delay : optional
            Per-edge values; see :meth:`with_weight` / :meth:`with_delay`
            (distributions are not accepted here; use the methods with a key).

        Returns
        -------
        EdgeSet
            Arrays on device: local indices in
            ``index_dtype(max(n_pre, n_post))``, gids in ``index_dtype(max_gid + 1)``.

        Raises
        ------
        ValueError
            On shape mismatches or out-of-range indices.
        """
        pre_gids = as_gids(pre_gids, 'pre_gids')
        post_gids = as_gids(post_gids, 'post_gids')
        pi = np.asarray(pre_index)
        qi = np.asarray(post_index)
        if pi.ndim != 1 or qi.ndim != 1 or pi.shape != qi.shape:
            raise ValueError(f'pre_index and post_index must be 1-D of equal length, '
                             f'got {pi.shape} and {qi.shape}')
        if pi.size:
            if not (np.issubdtype(pi.dtype, np.integer) and np.issubdtype(qi.dtype, np.integer)):
                raise TypeError('pre_index and post_index must be integers')
            if pi.min() < 0 or pi.max() >= pre_gids.size:
                raise ValueError(f'pre_index out of range [0, {pre_gids.size})')
            if qi.min() < 0 or qi.max() >= post_gids.size:
                raise ValueError(f'post_index out of range [0, {post_gids.size})')
        idt = index_dtype(max(pre_gids.size, post_gids.size))
        gdt = _gid_dtype(pre_gids, post_gids)
        es = cls(
            pre_index=jnp.asarray(pi.astype(idt)),
            post_index=jnp.asarray(qi.astype(idt)),
            pre_gids=jnp.asarray(pre_gids.astype(gdt)),
            post_gids=jnp.asarray(post_gids.astype(gdt)),
        )
        if weight is not None:
            es = es.with_weight(weight)
        if delay is not None:
            es = es.with_delay(delay)
        return es

    @classmethod
    def from_gids(cls, pre: Any, post: Any, pre_gids: Any, post_gids: Any,
                  weight: Any = None, delay: Any = None) -> 'EdgeSet':
        """Validated constructor from per-edge **global** ids.

        Parameters
        ----------
        pre, post : array_like of int, shape (E,)
            Global ids of each edge's pre / post neuron; must belong to the views.
        pre_gids, post_gids : array_like of int
            Global ids of the views.
        weight, delay : optional
            As in :meth:`create`.

        Raises
        ------
        ValueError
            If an edge endpoint is not in its view.
        """
        pre_gids_np = as_gids(pre_gids, 'pre_gids')
        post_gids_np = as_gids(post_gids, 'post_gids')
        return cls.create(_locate(pre, pre_gids_np, 'pre'), _locate(post, post_gids_np, 'post'),
                          pre_gids_np, post_gids_np, weight=weight, delay=delay)

    # ------------------------------------------------------------- properties
    @property
    def n_pre(self) -> int:
        """Size of the pre view."""
        return int(self.pre_gids.shape[0])

    @property
    def n_post(self) -> int:
        """Size of the post view."""
        return int(self.post_gids.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of edges ``E`` (multapses counted)."""
        return int(self.pre_index.shape[0])

    def __len__(self) -> int:
        return self.n_edges

    @property
    def pre(self) -> jax.Array:
        """Global id of each edge's pre neuron, shape ``(E,)``."""
        return self.pre_gids[self.pre_index]

    @property
    def post(self) -> jax.Array:
        """Global id of each edge's post neuron, shape ``(E,)``."""
        return self.post_gids[self.post_index]

    def multiplicity(self) -> np.ndarray:
        """Number of edges sharing each edge's ``(pre, post)`` pair, shape ``(E,)``.

        ``1`` everywhere when there are no multapses.
        """
        if self.n_edges == 0:
            return np.zeros((0,), np.int64)
        flat = (np.asarray(self.pre_index, np.int64) * max(self.n_post, 1)
                + np.asarray(self.post_index, np.int64))
        _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
        return counts[inv.reshape(-1)]

    @property
    def max_multiplicity(self) -> int:
        """Largest number of parallel edges between one ``(pre, post)`` pair (0 if empty)."""
        m = self.multiplicity()
        return int(m.max()) if m.size else 0

    # ----------------------------------------------------------------- values
    def with_weight(self, weight: Any, key: jax.Array | None = None,
                    dtype: Any = None) -> 'EdgeSet':
        """Return a copy with per-edge weights assigned.

        Parameters
        ----------
        weight : scalar, array (E,), Quantity or Distribution
            Scalars broadcast to every edge; a :class:`~brainpy_state._kernel.params.Distribution`
            draws one value per edge (in edge order) from ``key``.
        key : jax.Array, optional
            Required for distributions.
        dtype : dtype, optional
            Float dtype of the mantissa.
        """
        return dataclasses.replace(self, weight=_edge_values(weight, self.n_edges, key, dtype, 'weight'))

    def with_delay(self, delay: Any, key: jax.Array | None = None, dtype: Any = None) -> 'EdgeSet':
        """Return a copy with per-edge delays (time dimension) assigned; see :meth:`with_weight`."""
        d = _edge_values(delay, self.n_edges, key, dtype, 'delay')
        check_unit(d, u.ms, 'delay')
        return dataclasses.replace(self, delay=d)

    # ---------------------------------------------------------- introspection
    def select(self, source: Any = None, target: Any = None) -> np.ndarray:
        """Positions of the edges whose pre gid is in ``source`` and post gid in ``target``.

        Parameters
        ----------
        source, target : array_like of int, optional
            Global ids to keep; ``None`` keeps all.

        Returns
        -------
        numpy.ndarray of int64
            Edge positions in increasing order.
        """
        keep = np.ones((self.n_edges,), bool)
        if source is not None:
            keep &= np.isin(np.asarray(self.pre), np.asarray(source).reshape(-1))
        if target is not None:
            keep &= np.isin(np.asarray(self.post), np.asarray(target).reshape(-1))
        return np.nonzero(keep)[0].astype(np.int64)

    def get_connections(self, source: Any = None, target: Any = None) -> Connections:
        """Edges from ``source`` to ``target`` (global ids), zero weights and multapses included.

        Parameters
        ----------
        source, target : array_like of int, optional
            Global ids to filter on; ``None`` keeps all.

        Returns
        -------
        Connections
            Edge positions, global endpoints and per-edge values, in edge order.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> from brainpy_state._kernel.connectivity.edgeset import EdgeSet
            >>> es = EdgeSet.from_gids([0, 0, 1], [2, 2, 2], [0, 1], [2], weight=0. * u.pA)
            >>> len(es.get_connections()), len(es.get_connections(source=[1]))
            (3, 1)
        """
        e = self.select(source, target)
        sel = lambda v: None if v is None else v[jnp.asarray(e)]
        return Connections(edge=e, source=np.asarray(self.pre)[e], target=np.asarray(self.post)[e],
                           weight=sel(self.weight), delay=sel(self.delay))

    def set(self, *, weight: Any = None, delay: Any = None, edges: Any = None,
            key: jax.Array | None = None) -> 'EdgeSet':
        """Return a copy with ``weight`` and/or ``delay`` replaced on ``edges``.

        Parameters
        ----------
        weight, delay : scalar, array, Quantity or Distribution, optional
            New values for the selected edges, aligned with them (one value per
            selected edge, in edge order) or broadcast from a scalar.
        edges : array_like of int, bool mask (E,), or Connections, optional
            Edges to update; ``None`` means all edges.
        key : jax.Array, optional
            Required when a value is a distribution.

        Raises
        ------
        ValueError
            If a value has the wrong length, or updating a subset of edges whose
            value has not been assigned yet.
        UnitMismatchError
            If the new values' dimension differs from the existing ones.
        """
        idx = self._edge_positions(edges)
        out = self
        for name, value in (('weight', weight), ('delay', delay)):
            if value is None:
                continue
            new = _edge_values(value, idx.size, key, None, name)
            if name == 'delay':
                check_unit(new, u.ms, 'delay')
            cur = getattr(out, name)
            if cur is not None:
                check_unit(new, unit_of(cur), name)
            if idx.size == self.n_edges and np.array_equal(idx, np.arange(self.n_edges)):
                full = new
            else:
                if cur is None:
                    raise ValueError(f'cannot set {name} on a subset of edges before it is assigned')
                unit = unit_of(cur)
                cur_m = jnp.asarray(u.get_mantissa(cur))
                new_m = to_mantissa(new, unit, cur_m.dtype)
                full = from_mantissa(cur_m.at[jnp.asarray(idx)].set(new_m), unit)
            out = dataclasses.replace(out, **{name: full})
        return out

    def _edge_positions(self, edges: Any) -> np.ndarray:
        if edges is None:
            return np.arange(self.n_edges, dtype=np.int64)
        if isinstance(edges, Connections):
            return edges.edge
        e = np.asarray(edges)
        if e.dtype == bool:
            if e.shape != (self.n_edges,):
                raise ValueError(f'edge mask must have shape ({self.n_edges},), got {e.shape}')
            return np.nonzero(e)[0].astype(np.int64)
        e = e.reshape(-1).astype(np.int64)
        if e.size and (e.min() < 0 or e.max() >= self.n_edges):
            raise ValueError(f'edge positions out of range [0, {self.n_edges})')
        return e

    # ------------------------------------------------------------- comparison
    def same_edges(self, other: 'EdgeSet') -> bool:
        """Whether both edge sets have the same views and the same edges in the same order."""
        return all(np.array_equal(np.asarray(getattr(self, f)), np.asarray(getattr(other, f)))
                   for f in ('pre_gids', 'post_gids', 'pre_index', 'post_index'))

    def __repr__(self) -> str:
        return (f'EdgeSet(n_pre={self.n_pre}, n_post={self.n_post}, n_edges={self.n_edges}, '
                f'weight={"set" if self.weight is not None else None}, '
                f'delay={"set" if self.delay is not None else None})')


def _locate(gids: Any, view: np.ndarray, name: str) -> np.ndarray:
    g = np.asarray(gids, dtype=np.int64).reshape(-1)
    if g.size == 0:
        return g
    order = np.argsort(view, kind='stable')
    pos = np.searchsorted(view[order], g)
    pos = np.clip(pos, 0, max(view.size - 1, 0))
    if view.size == 0 or not np.array_equal(view[order][pos], g):
        raise ValueError(f'some {name} gids are not in the {name} view')
    return order[pos]


def _edge_values(value: Any, n: int, key: Any, dtype: Any, name: str) -> Any:
    """Per-edge values of length ``n`` from a scalar, array, Quantity or Distribution."""
    if isinstance(value, Distribution):
        if key is None:
            raise ValueError(f'a key is required to sample the {name} distribution')
        return value.sample(key, (n,), dtype)
    unit = unit_of(value)
    m = jnp.asarray(u.get_mantissa(value))
    if dtype is not None:
        m = m.astype(dtype)
    elif not jnp.issubdtype(m.dtype, jnp.floating):
        m = m.astype(jnp.zeros((), dtype=float).dtype)
    if m.ndim == 0:
        m = jnp.full((n,), m, m.dtype)
    elif m.shape != (n,):
        raise ValueError(f'{name} must be a scalar or have shape ({n},), got {m.shape}')
    return m if unit.is_unitless else u.Quantity(m, unit=unit)
