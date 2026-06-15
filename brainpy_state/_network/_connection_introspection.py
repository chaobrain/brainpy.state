# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Connection enumeration + introspection (NEST ``GetConnections`` analogue).

This module is a thin, **additive** convenience layer over the projections the
:class:`~brainpy_state._network._simulator.Simulator` already builds — it does not
change how any projection stores its edges. Each projection family exposes a
``realized_edges()`` accessor (in ``_event_proj.py`` / ``_event_plastic.py``)
returning a :class:`ProjEdges` view of its realized synapses; this module wraps a
filtered set of those across projections in a :class:`SynapseCollection`, the
object ``Simulator.get_connections`` returns.

Edge-order contract
-------------------
Within one projection, edges are enumerated grouped by **ascending
population-local source index** (stable within a group — the projection's native
CSR / row-major order). Across projections, results are concatenated in
**registration order** (the order ``connect`` built them). NEST orders
``GetConnections`` globally by ``(source, target, …)``; for a single-projection
query this coincides, and the two ported demos scatter into a ``W[src, trg] += w``
matrix (order-insensitive), so any cross-projection ordering difference is
immaterial (documented divergence).

Node indices are **population-local** — ``brainpy.state`` has no global node-id
space, so ``source`` / ``target`` are indices within the source / target
population (the matrix-building demos index their weight matrices with them
directly, without NEST's ``min(node_id)`` subtraction).
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import brainstate
import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._brainpy._delay import InputDelay

__all__ = ['ProjEdges', 'SynapseCollection']

_STATIC_PLASTIC_MODELS = ('static_synapse', 'static_synapse_hom_w')


@dataclasses.dataclass(frozen=True)
class ProjEdges:
    """A single projection's realized edges, in the documented canonical order.

    Returned by each projection's ``realized_edges()``. ``source`` / ``target``
    are population-local int arrays; ``weight`` / ``delay`` are brainunit
    Quantities read **live** (post-simulation evolved weights for a plastic
    projection). ``write_weight`` / ``write_delay`` are the guarded write-back
    hooks used by :meth:`SynapseCollection.set` (``None`` when the field is not
    settable).

    Parameters
    ----------
    source, target : numpy.ndarray
        Population-local pre / post index per edge, shape ``(E,)``.
    weight : brainunit.Quantity
        Per-edge weight, shape ``(E,)`` (a homogeneous projection broadcasts its
        shared scalar to ``E``).
    delay : brainunit.Quantity
        Per-edge axonal delay, shape ``(E,)`` (homogeneous broadcast; ``0 ms``
        when the projection has no delay).
    is_homogeneous_weight : bool
        ``True`` when the projection stores a single shared weight (a per-edge
        ``set`` is refused).
    is_plastic : bool
        ``True`` for an ``EventPlasticProj`` / ``VoltageCoupledPlasticProj``.
    model_name : str
        Synapse-model name (``'static_synapse'`` for the static event path, else
        the spec class name, e.g. ``'stdp_synapse'``).
    write_weight : Callable or None
        ``write_weight(local_edge_idx, mantissa_values)`` writes weight mantissas
        (in the projection's weight unit) at the given canonical edge positions;
        ``None`` if the projection refuses weight writes (a weight-evolving
        plastic rule).
    write_delay : Callable or None
        ``write_delay(quantity)`` sets the homogeneous axonal delay (rounded to
        the ``dt`` grid); ``None`` if unsettable.
    """
    source: np.ndarray
    target: np.ndarray
    weight: u.Quantity
    delay: u.Quantity
    is_homogeneous_weight: bool
    is_plastic: bool
    model_name: str
    write_weight: Optional[Callable] = None
    write_delay: Optional[Callable] = None


def canonical_order(source) -> np.ndarray:
    """Stable argsort grouping edges by ascending population-local source.

    Stable so that within a source group the projection's native storage order
    (CSR column order for the event/plastic CSR paths, row-major for dense) is
    preserved — the documented edge-order contract.
    """
    return np.argsort(np.asarray(source), kind='stable')


# ---------------------------------------------------------------------------
# Unit / delay helpers
# ---------------------------------------------------------------------------

def _split_weight(w):
    """Split a weight into ``(mantissa ndarray-like, unit)`` (``UNITLESS`` if plain)."""
    if isinstance(w, u.Quantity):
        return jnp.asarray(w.mantissa), w.unit
    return jnp.asarray(w), u.UNITLESS


def _as_quantity(mantissa, unit):
    """Re-attach ``unit`` to a mantissa (a plain array when ``unit`` is UNITLESS)."""
    return u.Quantity(mantissa, unit=unit) if unit is not u.UNITLESS else mantissa


def _broadcast_delay(delay, E: int) -> u.Quantity:
    """Broadcast a homogeneous axonal delay to an ``(E,)`` ms Quantity (``0 ms`` if none)."""
    if delay is None:
        d_ms = jnp.zeros(E)
    else:
        d = delay.to_decimal(u.ms) if isinstance(delay, u.Quantity) else jnp.asarray(delay)
        d_ms = jnp.broadcast_to(jnp.asarray(d), (E,))
    return u.Quantity(d_ms, unit=u.ms)


def _set_event_delay(proj, q) -> u.Quantity:
    """Set an EventProjection's homogeneous axonal delay, rounded to the ``dt`` grid.

    NEST stores delays as an integer multiple of the resolution; matching that, the
    requested delay is rounded to the nearest ``dt`` and the ``InputDelay`` seam is
    rebuilt (or cleared when the rounded delay is ``0``). Returns the grid-rounded
    delay actually applied.
    """
    dt = brainstate.environ.get_dt()
    dt_ms = float(u.Quantity(dt).to_decimal(u.ms))
    q_ms = float(u.Quantity(q).to_decimal(u.ms)) if isinstance(q, u.Quantity) else float(q)
    steps = int(round(q_ms / dt_ms))
    rounded = (steps * dt_ms) * u.ms
    if steps <= 0:
        proj._delay = None
        proj.delay_seam = None
    else:
        proj._delay = rounded
        proj.delay_seam = InputDelay((proj._n_pre_pop,), rounded)
        brainstate.nn.init_all_states(proj.delay_seam)
    return rounded


# ---------------------------------------------------------------------------
# Write-back closures (guarded by SynapseCollection.set)
# ---------------------------------------------------------------------------

def _make_event_writer(target, order):
    """Build a per-edge weight writer for an :class:`EventProjection` storage mode.

    ``order`` maps a canonical edge position (the order ``realized_edges`` returns)
    to a native storage position, so a write at canonical indices lands on the
    right ``_data`` / matrix cell / shared scalar.
    """
    kind = target[0]
    if kind == 'data':                     # sparse / receptor: per-edge _data State
        state = target[1]

        def write(local_idx, mantissa_values):
            pos = jnp.asarray(order[np.asarray(local_idx)])
            state.value = state.value.at[pos].set(jnp.asarray(mantissa_values))
        return write

    if kind == 'dense':                    # dense matrix cell (row, col) per edge
        wstate, rows, cols, w_unit = target[1], target[2], target[3], target[4]

        def write(local_idx, mantissa_values):
            pos = order[np.asarray(local_idx)]
            r = jnp.asarray(rows[pos]); c = jnp.asarray(cols[pos])
            mant, _ = _split_weight(wstate.value)
            mant = jnp.asarray(mant).at[r, c].set(jnp.asarray(mantissa_values))
            wstate.value = _as_quantity(mant, w_unit)
        return write

    # 'scalar' — one_to_one shares a single weight; only a uniform (scalar) set
    # reaches here (SynapseCollection refuses a per-edge set on a homogeneous proj).
    proj, w_unit = target[1], target[2]

    def write(local_idx, mantissa_values):
        v = np.asarray(mantissa_values).reshape(-1)
        scalar = jnp.asarray(v[0] if v.size else 0.0)
        proj._weight = _as_quantity(scalar, w_unit)
    return write


def _make_plastic_writer(proj, order):
    """Build a per-edge weight writer for a plastic projection (static rule only).

    Writes the live ``weight`` State when it has been allocated (post-``init_state``),
    else the pre-simulation ``_w_init`` array, so an introspection-time edit survives
    into the run. A homogeneous (0-d) weight takes the uniform scalar.
    """
    def write(local_idx, mantissa_values):
        pos = jnp.asarray(order[np.asarray(local_idx)])
        w_state = getattr(proj, 'weight', None)
        if isinstance(w_state, brainstate.State):
            cur = jnp.asarray(u.get_mantissa(w_state.value))
            if cur.ndim == 0:
                v = np.asarray(mantissa_values).reshape(-1)
                w_state.value = jnp.asarray(v[0] if v.size else 0.0)
            else:
                w_state.value = cur.at[pos].set(jnp.asarray(mantissa_values))
        else:
            init = jnp.asarray(proj._w_init)
            if init.ndim == 0:
                v = np.asarray(mantissa_values).reshape(-1)
                proj._w_init = jnp.asarray(v[0] if v.size else 0.0)
            else:
                proj._w_init = init.at[pos].set(jnp.asarray(mantissa_values))
    return write


# ---------------------------------------------------------------------------
# Per-projection edge enumeration
# ---------------------------------------------------------------------------

def event_proj_edges(proj) -> 'ProjEdges':
    """Enumerate an :class:`EventProjection`'s realized edges as a :class:`ProjEdges`.

    Handles all four comm modes (dense matrix, sparse CSR, per-receptor scatter,
    element-wise ``one_to_one``); maps segment-local storage indices to
    population-local ``source`` / ``target`` and sorts to the canonical edge order.
    """
    from brainpy_state._network._projections import _DenseMatMul

    pre_map = np.asarray(proj.pre_local_idx)
    post_map = np.asarray(proj.post_local_idx)

    if proj._receptor:
        comm = proj.comm
        pre_seg = np.asarray(comm._pre_idx)
        post_seg = np.asarray(comm._post_idx)
        w_mant = np.asarray(comm._data.value)
        w_unit = comm._unit
        is_homogeneous = False
        target = ('data', comm._data)
    elif proj._one_to_one:
        n = int(pre_map.shape[0])
        pre_seg = np.arange(n)
        post_seg = np.arange(n)
        w_m, w_unit = _split_weight(proj._weight)
        is_homogeneous = (w_m.ndim == 0)
        w_mant = np.broadcast_to(np.asarray(w_m), (n,))
        target = ('scalar', proj, w_unit)
    elif isinstance(proj.comm, _DenseMatMul):
        W = proj._W.value
        w_m, w_unit = _split_weight(W)
        Wm = np.asarray(w_m)
        rows, cols = np.nonzero(Wm)
        pre_seg = rows
        post_seg = cols
        w_mant = Wm[rows, cols]
        is_homogeneous = False
        target = ('dense', proj._W, rows, cols, w_unit)
    else:                                  # sparse CSR event matmul
        comm = proj.comm
        indptr = np.asarray(comm._indptr)
        pre_seg = np.repeat(np.arange(indptr.shape[0] - 1), np.diff(indptr))
        post_seg = np.asarray(comm._indices)
        w_mant = np.asarray(comm._data.value)
        w_unit = comm._unit
        is_homogeneous = False
        target = ('data', comm._data)

    source = pre_map[pre_seg]
    dst = post_map[post_seg]
    order = canonical_order(source)
    source = source[order]
    dst = dst[order]
    w_mant = np.asarray(w_mant)[order]
    E = int(source.shape[0])

    weight = _as_quantity(jnp.asarray(w_mant), w_unit)
    delay = _broadcast_delay(getattr(proj, '_delay', None), E)
    write_weight = _make_event_writer(target, order)
    write_delay = lambda q: _set_event_delay(proj, q)
    return ProjEdges(source, dst, weight, delay, is_homogeneous, False,
                     'static_synapse', write_weight, write_delay)


def plastic_proj_edges(proj) -> 'ProjEdges':
    """Enumerate a plastic projection's realized edges as a :class:`ProjEdges`.

    Reads the live ``weight`` State when allocated (post-simulation evolved weights),
    else the pre-simulation ``_w_init``; a weight-evolving rule (anything but the
    static family) exposes ``write_weight = None`` so :meth:`SynapseCollection.set`
    refuses to overwrite rule-managed weights.
    """
    pre_map = np.asarray(proj.pre_local_idx)
    post_map = np.asarray(proj.post_local_idx)
    pre_seg = np.asarray(proj._pre_idx)
    post_seg = np.asarray(proj._post_idx)
    rule = proj.rule
    w_unit = proj._w_unit

    w_state = getattr(proj, 'weight', None)
    if isinstance(w_state, brainstate.State):
        wm = np.asarray(u.get_mantissa(w_state.value))
    else:
        wm = np.asarray(proj._w_init)
    E = int(pre_seg.shape[0])
    w_mant = np.broadcast_to(wm, (E,)) if wm.ndim == 0 else wm

    source = pre_map[pre_seg]
    dst = post_map[post_seg]
    order = canonical_order(source)
    source = source[order]
    dst = dst[order]
    w_mant = np.asarray(w_mant)[order]

    weight = _as_quantity(jnp.asarray(w_mant), w_unit)
    delay = _broadcast_delay(getattr(rule, 'delay', None), E)
    model_name = type(rule).__name__
    evolves = model_name not in _STATIC_PLASTIC_MODELS
    write_weight = None if evolves else _make_plastic_writer(proj, order)
    return ProjEdges(source, dst, weight, delay,
                     bool(rule.is_homogeneous_weight), True, model_name,
                     write_weight, None)


# ---------------------------------------------------------------------------
# SynapseCollection — a filtered, lazy view across projections
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _SpannedProj:
    """One projection's contribution to a :class:`SynapseCollection`.

    Holds only the projection handle and the **canonical edge indices** kept by the
    query (plus the cached population-local ``source`` / ``target`` of those edges).
    Weight / delay are never copied — they are re-read live from ``proj`` on demand.
    """
    proj: object
    kept: np.ndarray
    source: np.ndarray
    target: np.ndarray


def _membership(view):
    """Build ``{id(population): set(local idx)}`` from a ``NodeView`` (``None`` -> no filter)."""
    if view is None:
        return None
    if hasattr(view, 'segments'):
        segments = view.segments
    else:                                  # a bare population module
        from brainpy_state._network._nodeview import NodeView
        segments = NodeView.of(view).segments
    members = {}
    for seg in segments:
        members.setdefault(id(seg.population), set()).update(
            int(i) for i in np.asarray(seg.indices).tolist())
    return members


def collect_connections(connections, source=None, target=None, synapse=None) -> 'SynapseCollection':
    """Build a :class:`SynapseCollection` over a Simulator's connection registry.

    Parameters
    ----------
    connections : list of tuple
        The Simulator's ``(pre_pop, post_pop, model_name, proj)`` registry, in
        registration order.
    source, target : NodeView or None
        Restrict to edges whose pre / post neuron lies in the given view (matched by
        population identity **and** population-local index); ``None`` keeps all.
    synapse : str or None
        Restrict to projections whose synapse-model name equals this string.

    Returns
    -------
    SynapseCollection
        A lazy, filtered view; empty when nothing matches.
    """
    src_m = _membership(source)
    tgt_m = _membership(target)
    spanned = []
    for pre_pop, post_pop, model_name, proj in connections:
        if synapse is not None and model_name != synapse:
            continue
        if src_m is not None and id(pre_pop) not in src_m:
            continue
        if tgt_m is not None and id(post_pop) not in tgt_m:
            continue
        edges = proj.realized_edges()
        src = np.asarray(edges.source)
        tgt = np.asarray(edges.target)
        keep = np.ones(src.shape[0], dtype=bool)
        if src_m is not None:
            keep &= np.isin(src, np.fromiter(src_m[id(pre_pop)], dtype=int))
        if tgt_m is not None:
            keep &= np.isin(tgt, np.fromiter(tgt_m[id(post_pop)], dtype=int))
        kept = np.nonzero(keep)[0]
        if kept.shape[0] == 0:
            continue
        spanned.append(_SpannedProj(proj, kept, src[kept], tgt[kept]))
    return SynapseCollection(spanned)


def _to_unit_mantissa(value, unit):
    """Convert ``value`` to a bare mantissa in ``unit`` (a Quantity is unit-checked)."""
    if isinstance(value, u.Quantity):
        return jnp.asarray(value.mantissa) if unit is u.UNITLESS else jnp.asarray(value.to_decimal(unit))
    return jnp.asarray(value)


class SynapseCollection:
    """A filtered, lazy view over realized synapses (NEST ``SynapseCollection``).

    Returned by :meth:`~brainpy_state._network._simulator.Simulator.get_connections`.
    It stores, per spanned projection, only the kept edge indices and their
    population-local ``source`` / ``target`` — **no copy** of weight or delay. Each
    :meth:`get` re-reads weight / delay from the live projection, so a query made
    before :meth:`~brainpy_state._network._simulator.Simulator.simulate` still
    reflects post-simulation evolved plastic weights.

    Methods mirror the NEST object: :meth:`get` (one key -> array, a list of keys
    -> dict), :meth:`set` (``'weight'`` / ``'delay'`` write-back, guarded), the
    :attr:`source` / :attr:`target` arrays, ``len()``, iteration over
    ``(source, target)`` pairs, and ``repr``.

    Notes
    -----
    ``source`` / ``target`` are **population-local** indices (``brainpy.state`` has
    no global node-id space). ``set('weight', value)`` accepts a scalar (broadcast
    to every edge) or a full per-edge array; a per-edge array is refused on a
    homogeneous-weight projection, and any write is refused on a projection whose
    plastic rule evolves the weight. ``set('delay', scalar)`` is homogeneous per
    projection (grid-rounded to ``dt``) and applies only to the static event path.
    """
    __module__ = 'brainpy.state'

    def __init__(self, spanned):
        self._spanned = list(spanned)

    def __len__(self) -> int:
        return int(sum(sp.source.shape[0] for sp in self._spanned))

    @property
    def source(self) -> np.ndarray:
        """Population-local source index per edge (``(N,)`` int array)."""
        if not self._spanned:
            return np.zeros(0, dtype=int)
        return np.concatenate([sp.source for sp in self._spanned])

    @property
    def target(self) -> np.ndarray:
        """Population-local target index per edge (``(N,)`` int array)."""
        if not self._spanned:
            return np.zeros(0, dtype=int)
        return np.concatenate([sp.target for sp in self._spanned])

    def get(self, key):
        """Read a connection attribute (or several).

        Parameters
        ----------
        key : str or list of str
            One of ``'source'``, ``'target'``, ``'weight'``, ``'delay'`` — or a list
            of them.

        Returns
        -------
        numpy.ndarray or brainunit.Quantity or dict
            A single array/Quantity for a string key (``weight`` / ``delay`` carry
            their unit and are read live), or a ``{key: value}`` dict for a list.
        """
        if isinstance(key, (list, tuple)):
            return {k: self.get(k) for k in key}
        if key == 'source':
            return self.source
        if key == 'target':
            return self.target
        if key == 'weight':
            return self._gather_live('weight', u.pA)
        if key == 'delay':
            return self._gather_live('delay', u.ms)
        raise KeyError(f"unknown connection key {key!r}; expected one of "
                       "'source', 'target', 'weight', 'delay'")

    def _gather_live(self, attr, default_unit):
        """Concatenate a live per-edge Quantity across spanned projections."""
        if not self._spanned:
            return u.Quantity(jnp.zeros(0), unit=default_unit)
        quantities = [getattr(sp.proj.realized_edges(), attr) for sp in self._spanned]
        unit = u.get_unit(quantities[0])
        parts = []
        for q, sp in zip(quantities, self._spanned):
            mant = (jnp.asarray(u.get_mantissa(q)) if unit is u.UNITLESS
                    else jnp.asarray(u.Quantity(q).to_decimal(unit)))
            parts.append(mant[sp.kept])
        mant = jnp.concatenate(parts)
        return mant if unit is u.UNITLESS else u.Quantity(mant, unit=unit)

    def set(self, key, value):
        """Write a connection attribute back into the live projections.

        Parameters
        ----------
        key : str
            ``'weight'`` or ``'delay'`` (``'source'`` / ``'target'`` are read-only).
        value : scalar, array, or brainunit.Quantity
            For ``'weight'``: a scalar broadcasts to every edge, a per-edge array
            sets each edge. For ``'delay'``: a scalar homogeneous delay per
            projection.

        Raises
        ------
        ValueError
            A per-edge weight on a homogeneous-weight projection; any weight write
            on a weight-evolving plastic projection; or a delay write on a plastic
            projection.
        KeyError
            A key other than ``'weight'`` / ``'delay'``.
        """
        if key == 'weight':
            self._set_weight(value)
        elif key == 'delay':
            self._set_delay(value)
        else:
            raise KeyError(f"cannot set {key!r}; only 'weight' and 'delay' are settable")

    def _set_weight(self, value):
        is_scalar = (jnp.asarray(u.get_mantissa(value)).ndim == 0)
        edges_list = [sp.proj.realized_edges() for sp in self._spanned]
        # Validate every spanned projection before writing any (all-or-nothing).
        for edges in edges_list:
            if edges.write_weight is None:
                raise ValueError(
                    f"weight of the '{edges.model_name}' projection is managed by its "
                    "plastic rule and is overwritten during simulate(); set it through "
                    "the synapse spec at connect() time, not SynapseCollection.set().")
            if (not is_scalar) and edges.is_homogeneous_weight:
                raise ValueError(
                    f"cannot set a per-edge weight on the homogeneous-weight projection "
                    f"'{edges.model_name}' (it shares a single scalar weight); pass a "
                    "scalar to set them all, or rebuild it with a per-edge rule.")
        offset = 0
        for sp, edges in zip(self._spanned, edges_list):
            m = int(sp.kept.shape[0])
            unit = u.get_unit(edges.weight)
            chunk = value if is_scalar else value[offset:offset + m]
            mant = jnp.broadcast_to(_to_unit_mantissa(chunk, unit), (m,))
            edges.write_weight(sp.kept, mant)
            offset += m

    def _set_delay(self, value):
        edges_list = [sp.proj.realized_edges() for sp in self._spanned]
        for edges in edges_list:
            if edges.write_delay is None:
                raise ValueError(
                    f"delay of the '{edges.model_name}' projection is not settable via "
                    "SynapseCollection (a plastic delay is fixed at connect() time).")
        for edges in edges_list:
            edges.write_delay(value)

    def __iter__(self):
        for s, t in zip(self.source.tolist(), self.target.tolist()):
            yield (s, t)

    def __repr__(self) -> str:
        return (f'SynapseCollection({len(self)} edges over '
                f'{len(self._spanned)} projection(s))')
