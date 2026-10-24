# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Edge storages built from an :class:`EdgeSet`: :class:`Dense` and :class:`CSR` (K0 spec §4.3, §4.4).

A storage turns per-step source payloads into per-target sums
``y[j] = sum_{edges e: post(e) = j} payload[pre(e)] * w[e]``.

Weight layouts
--------------
Each storage has a *native* weight layout, the array a ``ParamState`` holds and that
:meth:`Storage.forward` consumes:

* :class:`Dense` — an ``(n_pre, n_post)`` matrix; multapses are summed into one entry
  and entries without an edge are masked to zero (their gradient is zero).
* :class:`CSR` — the per-edge vector permuted into row (pre) order, as
  ``brainevent.CSR`` stores it; multapses stay separate entries.

:meth:`Storage.weight_from_edges` / :meth:`Storage.weight_to_edges` convert between
the :class:`EdgeSet` per-edge order and the native layout (exact for CSR; for Dense
the weight of a multapse group is split equally among its edges).

Payload paths
-------------
``forward(payload, w, path)`` has two paths (spec §4.3):

* ``'binary'`` — event-driven ``brainevent.BinaryArray`` matmul. **Only** for sources
  with ``max_multiplicity == 1``: ``BinaryArray`` is a mask, not a value (brainevent
  >= 0.3: an entry is active iff ``x != 0``), so a count of 2 would be delivered as 1.
  The payload must therefore be a spike indicator in {0, 1} (the
  :class:`~brainspike._kernel.ports.Event` contract). Any float dtype is passed to
  brainevent as is (brainevent >= 0.3 reads float64 events correctly on GPU), so the
  payload gradient (straight-through ``ct @ W.T``) keeps the payload's dtype.
* ``'float'`` — exact float matmul; counts > 1 contribute ``count * w`` and both VJPs
  (weights, payload) are exact.

:func:`payload_path` picks the path from the source port. :meth:`Storage.forward_etrace`
routes through ``braintrace.matmul`` / ``braintrace.sparse_matmul`` so that
``braintrace.compile`` sees the weight (always the float path).
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable, ClassVar, Iterator, Mapping

import brainevent
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.connectivity.edgeset import EdgeSet
from brainspike._kernel.ports import Event
from brainspike._kernel.units import index_dtype

__all__ = [
    'PAYLOAD_PATHS',
    'STORAGE_KINDS',
    'payload_path',
    'csr_index_dtypes',
    'Storage',
    'Dense',
    'CSR',
    'make_storage',
    'AUTO_DENSE_THRESHOLDS',
    'auto_dense_threshold',
]

#: Payload paths of :meth:`Storage.forward`.
PAYLOAD_PATHS = ('binary', 'float')

#: Storage kinds accepted by :func:`make_storage`.
STORAGE_KINDS = ('dense', 'csr', 'auto')

#: Pair density from which ``make_storage(kind='auto')`` picks :class:`Dense`, per JAX
#: backend (``'cpu'`` is the fallback for other backends). Measured in K1a-9 with
#: ``benchmarks/k1a9_storage.py`` (Brunel A, orders 50-800, densities 0.1-1.0, float32
#: and float64). GPU: CSR step time is <= 1.06 x Dense in every cell (within noise for
#: N = 250, 0.22-0.89 x from N = 1000), and CSR peak memory is lower in every cell, so
#: ``'auto'`` never picks Dense there. CPU: Dense is 1.08-1.35 x faster for small
#: networks, so the K0 rule (density >= 0.1) is kept.
AUTO_DENSE_THRESHOLDS = {'cpu': 0.1, 'gpu': float('inf')}


def auto_dense_threshold(backend: str | None = None) -> float:
    """Density from which ``'auto'`` picks Dense on ``backend``.

    Parameters
    ----------
    backend : str, optional
        JAX backend name; default ``jax.default_backend()``. Backends without an entry
        in :data:`AUTO_DENSE_THRESHOLDS` use the ``'cpu'`` value.

    Returns
    -------
    float
        The threshold; ``inf`` means ``'auto'`` always picks CSR.
    """
    b = backend or jax.default_backend()
    return AUTO_DENSE_THRESHOLDS.get(b, AUTO_DENSE_THRESHOLDS['cpu'])


# upstream: brainevent#198 (fixed in 0.3.0) -- the rule stays: BinaryArray is a mask
# (active iff x != 0), not a value, so the 'binary' path is limited to sources with at
# most one spike per step. Count payloads need brainevent.WeightedArray (#201, K1b).
def payload_path(source: Any) -> str:
    """Payload path for a source: ``'binary'`` iff it emits at most one spike per step.

    Parameters
    ----------
    source : Event, int or None
        The source's output :class:`~brainspike._kernel.ports.Event` port, or its
        ``max_multiplicity`` (``None`` = unbounded).

    Returns
    -------
    str
        ``'binary'`` or ``'float'``.

    Examples
    --------
    .. code-block:: python

        >>> from brainspike._kernel.ports import Event
        >>> from brainspike._kernel.connectivity.storage import payload_path
        >>> payload_path(Event()), payload_path(Event(max_multiplicity=None)), payload_path(3)
        ('binary', 'float', 'float')
    """
    if isinstance(source, Event):
        source = source.max_multiplicity
    if source is not None and (isinstance(source, bool) or not isinstance(source, (int, np.integer))
                               or source < 1):
        raise TypeError(f'expected an Event port, an int >= 1 or None, got {source!r}')
    return 'binary' if source == 1 else 'float'


def csr_index_dtypes(n_pre: int, n_post: int, n_edges: int) -> tuple[np.dtype, np.dtype]:
    """Index dtypes of a CSR structure.

    Parameters
    ----------
    n_pre, n_post : int
        Matrix shape (rows = pre, columns = post).
    n_edges : int
        Stored entries.

    Returns
    -------
    (numpy.dtype, numpy.dtype)
        ``(indices_dtype, indptr_dtype)``: column indices are int32 unless the
        dimensions exceed int32; ``indptr`` is int32 below ``2**31`` edges and int64
        above.

    Raises
    ------
    RuntimeError
        If int64 is needed while ``jax_enable_x64`` is off.
    """
    return index_dtype(max(int(n_pre), int(n_post))), index_dtype(int(n_edges))


def _split(w: Any) -> tuple[jax.Array, u.Unit]:
    if isinstance(w, u.Quantity):
        return w.mantissa, w.unit
    return jnp.asarray(w), u.UNITLESS


def _split_edges(w: Any) -> tuple[Any, u.Unit]:
    """Like :func:`_split` but keeps host (numpy) per-edge weights on the host."""
    if isinstance(w, u.Quantity):
        return w.mantissa, w.unit
    if isinstance(w, np.ndarray):
        return w, u.UNITLESS
    return jnp.asarray(w), u.UNITLESS


def _take(m: Any, order: np.ndarray, order_j: Callable[[], jax.Array]) -> jax.Array:
    """``m[order]`` as a device array.

    Host weights are permuted on the host and copied once (K1a-8: no device copies
    of the edge-order weights and of the permutation at compile time); device
    weights are gathered on the device with ``order_j()``.
    """
    if isinstance(m, np.ndarray):
        return jnp.asarray(m[order])
    return m[order_j()]


def _wrap(m: Any, unit: u.Unit) -> Any:
    return m if unit.is_unitless else u.Quantity(m, unit=unit)


def _check_path(path: str) -> None:
    if path not in PAYLOAD_PATHS:
        raise ValueError(f'path must be one of {PAYLOAD_PATHS}, got {path!r}')


def _host_index(a: Any) -> np.ndarray:
    """Host index array in int32 when it fits (halves the memory of int64 copies)."""
    a = np.asarray(a)
    big = a.size and (int(a.max()) >= 2 ** 31 or int(a.min()) < -2 ** 31)
    return a.astype(np.int64 if big else np.int32, copy=False)


def _payload(payload: Any, n_pre: int) -> jax.Array:
    if isinstance(payload, u.Quantity):
        if not payload.unit.is_unitless:
            raise TypeError(f'event payloads are dimensionless counts, got unit {payload.unit}')
        payload = payload.mantissa
    x = jnp.asarray(payload)
    if x.ndim not in (1, 2) or x.shape[-1] != n_pre:
        raise ValueError(f'payload must have shape ({n_pre},) or (batch, {n_pre}), got {x.shape}')
    return x


class Storage:
    """Base class of edge storages (see the module docstring).

    Parameters
    ----------
    edges : EdgeSet
        Realised edges; the storage keeps only what it needs for the matmul.

    Attributes
    ----------
    n_pre, n_post, n_edges : int
        Shape and number of edges of the projection.
    """

    kind: ClassVar[str] = ''
    #: Names of the device arrays :meth:`forward` reads (see :meth:`arrays`).
    array_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, edges: EdgeSet) -> None:
        if not isinstance(edges, EdgeSet):
            raise TypeError(f'expected an EdgeSet, got {type(edges).__name__}')
        self.n_pre = edges.n_pre
        self.n_post = edges.n_post
        self.n_edges = edges.n_edges
        self._pre = _host_index(edges.pre_index)
        self._post = _host_index(edges.post_index)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(n_pre={self.n_pre}, n_post={self.n_post}, n_edges={self.n_edges})'

    def arrays(self) -> dict[str, jax.Array]:
        """Device structure arrays read by :meth:`forward` (index arrays, masks).

        A compiled program passes them as **arguments** (see :meth:`bound`) instead of
        closing over them: closed-over arrays are embedded in the lowered program as
        constants, which costs compile time and host memory proportional to the
        number of edges.

        Returns
        -------
        dict
            ``{attribute name: array}`` (absent arrays omitted).
        """
        return {k: getattr(self, k) for k in self.array_names if getattr(self, k) is not None}

    @contextlib.contextmanager
    def bound(self, arrays: Mapping[str, Any]) -> Iterator['Storage']:
        """Temporarily replace the structure arrays (e.g. by tracers of jit arguments).

        Parameters
        ----------
        arrays : mapping
            ``{attribute name: array}`` as returned by :meth:`arrays`.

        Raises
        ------
        KeyError
            If a name is not one of :attr:`array_names`.
        """
        unknown = set(arrays) - set(self.array_names)
        if unknown:
            raise KeyError(f'{type(self).__name__} has no structure arrays {sorted(unknown)}')
        old = {k: getattr(self, k) for k in arrays}
        try:
            for k, v in arrays.items():
                setattr(self, k, v)
            yield self
        finally:
            for k, v in old.items():
                setattr(self, k, v)

    def _check_edge_weight(self, w: Any) -> None:
        if tuple(np.shape(w)) != (self.n_edges,):
            raise ValueError(f'per-edge weights must have shape ({self.n_edges},), got {np.shape(w)}')

    def weight_from_edges(self, edge_weight: Any) -> Any:  # pragma: no cover
        """Native weight layout from per-edge weights (in :class:`EdgeSet` order)."""
        raise NotImplementedError

    def weight_to_edges(self, w: Any) -> Any:  # pragma: no cover
        """Per-edge weights (in :class:`EdgeSet` order) from the native layout."""
        raise NotImplementedError

    def forward(self, payload: Any, w: Any, path: str = 'float') -> Any:  # pragma: no cover
        """Per-target input ``payload @ W``.

        Parameters
        ----------
        payload : jax.Array, shape (n_pre,) or (batch, n_pre)
            Spike counts of the sources (float; surrogate gradients flow through it).
        w : jax.Array or Quantity
            Weights in the native layout.
        path : {'float', 'binary'}, default 'float'
            Payload path; ``'binary'`` only for sources with ``max_multiplicity == 1``
            (see :func:`payload_path`).

        Returns
        -------
        jax.Array or Quantity
            Shape ``(n_post,)`` or ``(batch, n_post)``, in the unit of ``w``.
        """
        raise NotImplementedError

    def forward_etrace(self, payload: Any, w: Any) -> Any:  # pragma: no cover
        """Like :meth:`forward` (float path) but routed through a braintrace primitive."""
        raise NotImplementedError

    def nbytes(self, itemsize: int = 4) -> int:  # pragma: no cover
        """Device memory of the structure plus native weights of ``itemsize`` bytes."""
        raise NotImplementedError


class Dense(Storage):
    """Dense ``(n_pre, n_post)`` weight matrix.

    Pairs without an edge are masked to zero inside :meth:`forward` and
    :meth:`forward_etrace` (the mask is skipped when every pair has an edge), so
    their weight entries receive zero gradient.
    """

    kind: ClassVar[str] = 'dense'
    array_names: ClassVar[tuple[str, ...]] = ('mask',)

    def __init__(self, edges: EdgeSet) -> None:
        super().__init__(edges)
        mask = np.zeros((self.n_pre, self.n_post), bool)
        mask[self._pre, self._post] = True
        self.full = bool(mask.all())
        self.mask = None if self.full else jnp.asarray(mask)
        self._mult = jnp.asarray(edges.multiplicity()) if self.n_edges else jnp.zeros((0,), jnp.int32)
        self._pre_j = jnp.asarray(self._pre, jnp.int32)
        self._post_j = jnp.asarray(self._post, jnp.int32)
        mask_j = self.mask  # concrete: braintrace binds weight_fn as a static parameter

        def weight_fn(W):
            return u.math.where(mask_j, W, u.math.zeros_like(W))

        # One function object per storage: braintrace binds it as a static parameter.
        self._weight_fn = None if self.full else weight_fn

    def weight_from_edges(self, edge_weight: Any) -> Any:
        """Scatter-add per-edge weights into the matrix (multapses summed)."""
        self._check_edge_weight(edge_weight)
        m, unit = _split(edge_weight)
        W = jnp.zeros((self.n_pre, self.n_post), m.dtype).at[self._pre_j, self._post_j].add(m)
        return _wrap(W, unit)

    def weight_to_edges(self, w: Any) -> Any:
        """Gather per-edge weights; a multapse group's entry is split equally among its edges."""
        m, unit = _split(w)
        if m.shape != (self.n_pre, self.n_post):
            raise ValueError(f'dense weights must have shape ({self.n_pre}, {self.n_post}), got {m.shape}')
        return _wrap(m[self._pre_j, self._post_j] / self._mult.astype(m.dtype), unit)

    def _masked(self, m: jax.Array) -> jax.Array:
        return m if self.full else jnp.where(self.mask, m, jnp.zeros((), m.dtype))

    def forward(self, payload: Any, w: Any, path: str = 'float') -> Any:
        _check_path(path)
        x = _payload(payload, self.n_pre)
        m, unit = _split(w)
        W = self._masked(m)
        if path == 'binary':
            y = brainevent.BinaryArray(x) @ W
        else:  # upstream: brainevent#201 -- no event-driven matmul for count payloads
            y = jnp.matmul(x.astype(jnp.result_type(x, W)), W)
        return _wrap(y, unit)

    def forward_etrace(self, payload: Any, w: Any) -> Any:
        x = _payload(payload, self.n_pre)
        return braintrace.matmul(x, w, weight_fn=self._weight_fn)

    def nbytes(self, itemsize: int = 4) -> int:
        n = self.n_pre * self.n_post
        return n * itemsize + (0 if self.full else n)


class CSR(Storage):
    """Compressed sparse rows (rows = pre, columns = post) via ``brainevent.CSR``.

    Edges are sorted stably by pre index; multapses are separate entries. Index
    dtypes follow :func:`csr_index_dtypes`.

    Projections with at most one edge per neuron on the larger side
    (``n_edges <= max(n_pre, n_post)``, e.g. ``one_to_one``) skip the sparse kernel:
    :meth:`forward` gathers the sources and scatter-adds into the targets
    (``payload * w`` when the edges are the identity map), which is exact on both
    payload paths and costs ``O(n_edges)`` element-wise work.

    Attributes
    ----------
    order : numpy.ndarray
        Permutation from CSR position to :class:`EdgeSet` edge position.
    indices, indptr : jax.Array
        CSR structure.
    rows : jax.Array or None
        Pre index of every CSR position (only on the gather path).
    gather : bool
        Whether :meth:`forward` uses the gather/scatter path.
    identity : bool
        Whether the edges are ``i -> i`` for every ``i`` (``payload * w``).
    """

    kind: ClassVar[str] = 'csr'
    # ``indptr`` stays concrete: brainevent sizes its binary-kernel workspace from it
    # on the host (and it is only ``n_pre + 1`` long).
    array_names: ClassVar[tuple[str, ...]] = ('indices', 'rows')

    def __init__(self, edges: EdgeSet) -> None:
        super().__init__(edges)
        idx_dt, ptr_dt = csr_index_dtypes(self.n_pre, self.n_post, self.n_edges)
        self._ptr_dt = ptr_dt
        self.order = _host_index(np.argsort(self._pre, kind='stable'))
        counts = np.bincount(self._pre, minlength=self.n_pre)
        cols = self._post[self.order]
        self.indices = jnp.asarray(cols.astype(idx_dt))
        self.indptr = jnp.asarray(np.concatenate([[0], np.cumsum(counts)]).astype(ptr_dt))
        # concrete copies for the braintrace template (never replaced by :meth:`bound`)
        self._indices0, self._indptr0 = self.indices, self.indptr
        self.gather = 0 < self.n_edges <= max(self.n_pre, self.n_post)
        rows = self._pre[self.order] if self.gather else None
        ar = np.arange(self.n_edges)
        # `rows is not None` <=> `self.gather`
        self.identity = bool(rows is not None and self.n_pre == self.n_post == self.n_edges
                             and np.array_equal(rows, ar) and np.array_equal(cols, ar))
        self.rows = jnp.asarray(rows.astype(idx_dt)) if rows is not None and not self.identity else None
        self._templates: dict = {}

    @property
    def _order_j(self) -> jax.Array:
        # built on demand (compile time / introspection only): keeps no resident copy
        return jnp.asarray(self.order.astype(self._ptr_dt, copy=False))

    def _matrix(self, m: jax.Array) -> brainevent.CSR:
        return brainevent.CSR((m, self.indices, self.indptr), shape=(self.n_pre, self.n_post))

    def template(self, dtype: Any) -> brainevent.CSR:
        """Cached structure-only, unitless ``brainevent.CSR`` with data of ``dtype``.

        braintrace binds it as a static parameter (identity hashed), so the same
        object is reused across traces. Built from the concrete structure even
        inside :meth:`bound`.
        """
        k = np.dtype(dtype).name
        if k not in self._templates:
            self._templates[k] = brainevent.CSR(
                (jnp.zeros((self.n_edges,), dtype), self._indices0, self._indptr0),
                shape=(self.n_pre, self.n_post))
        return self._templates[k]

    def weight_from_edges(self, edge_weight: Any) -> Any:
        """Permute per-edge weights into CSR order."""
        self._check_edge_weight(edge_weight)
        m, unit = _split_edges(edge_weight)
        return _wrap(_take(m, self.order, lambda: self._order_j), unit)

    def weight_to_edges(self, w: Any) -> Any:
        """Inverse permutation back to :class:`EdgeSet` order (exact)."""
        m, unit = _split(w)
        if m.shape != (self.n_edges,):
            raise ValueError(f'CSR weights must have shape ({self.n_edges},), got {m.shape}')
        return _wrap(jnp.zeros_like(m).at[self._order_j].set(m), unit)

    def forward(self, payload: Any, w: Any, path: str = 'float') -> Any:
        _check_path(path)
        x = _payload(payload, self.n_pre)
        m, unit = _split(w)
        if m.shape != (self.n_edges,):
            raise ValueError(f'CSR weights must have shape ({self.n_edges},), got {m.shape}')
        dt = jnp.result_type(x, m)
        if self.n_edges == 0:
            return _wrap(jnp.zeros(x.shape[:-1] + (self.n_post,), dt), unit)
        if self.gather:
            xm = x.astype(dt)
            if self.identity:
                y = xm * m
            else:
                y = jnp.zeros(x.shape[:-1] + (self.n_post,), dt).at[..., self.indices].add(
                    xm[..., self.rows] * m)
            return _wrap(y, unit)
        mat = self._matrix(m)
        if path == 'binary':
            y = brainevent.BinaryArray(x) @ mat
        else:  # upstream: brainevent#201 -- no event-driven matmul for count payloads
            y = x.astype(dt) @ mat
        return _wrap(y, unit)

    def forward_etrace(self, payload: Any, w: Any) -> Any:
        x = _payload(payload, self.n_pre)
        m, _ = _split(w)
        # braintrace splits the unit off before binding, so the template is unitless.
        return braintrace.sparse_matmul(x, w, sparse_mat=self.template(m.dtype))

    def nbytes(self, itemsize: int = 4) -> int:
        n = (self.n_edges * (itemsize + self.indices.dtype.itemsize)
             + self.indptr.size * self.indptr.dtype.itemsize)
        return n + (0 if self.rows is None else self.rows.size * self.rows.dtype.itemsize)


_KINDS = {'dense': Dense, 'csr': CSR}


def make_storage(edges: EdgeSet, kind: str = 'auto', dense_threshold: float | None = None,
                 dense_max_elems: int = 1 << 24, backend: str | None = None) -> Storage:
    """Build the storage of a projection.

    Parameters
    ----------
    edges : EdgeSet
        Realised edges.
    kind : {'auto', 'dense', 'csr'}, default 'auto'
        ``'auto'`` picks Dense when the pair density ``n_edges / (n_pre * n_post)``
        is at least ``dense_threshold`` and the matrix has at most
        ``dense_max_elems`` entries, else CSR.
    dense_threshold : float, optional
        Density from which ``'auto'`` prefers Dense. Default: the measured value of
        ``backend`` (:func:`auto_dense_threshold`): 0.1 on CPU; on GPU ``inf``
        (``'auto'`` always picks CSR, which is as fast or faster and smaller there).
    dense_max_elems : int, default 2**24
        Largest dense matrix ``'auto'`` will allocate.
    backend : str, optional
        JAX backend the default threshold is taken for; default
        ``jax.default_backend()`` at the call (compile time).

    Returns
    -------
    Storage
    """
    if kind not in STORAGE_KINDS:
        raise ValueError(f'kind must be one of {STORAGE_KINDS}, got {kind!r}')
    if kind == 'auto':
        thr = auto_dense_threshold(backend) if dense_threshold is None else dense_threshold
        n = edges.n_pre * edges.n_post
        dense = n > 0 and n <= dense_max_elems and edges.n_edges >= thr * n
        kind = 'dense' if dense else 'csr'
    return _KINDS[kind](edges)
