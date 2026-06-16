# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spatial query helpers (NEST ``FindCenterElement`` / ``Distance`` / target queries).

:func:`center_element` and :func:`Distance` are pure layer-level queries. :func:`target_nodes`
and :func:`target_positions` read the *realized* adjacency back out of a built network
(via :meth:`brainpy_state._network._simulator.Simulator.get_connections`), mirroring NEST's
``GetTargetNodes`` / ``GetTargetPositions``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._nest_spatial._distance import pairwise_distance
from brainpy_state._nest_spatial._layers import _LEN, _as_len

__all__ = ['center_element', 'Distance', 'nearest_element', 'select_nodes_by_mask',
           'dump_layer_nodes', 'dump_layer_connections', 'target_nodes', 'target_positions']


def _fmt(v) -> str:
    """Format a scalar at round-trip precision (clean ints, ``.12g`` floats)."""
    return f"{float(v):.12g}"


def center_element(layer) -> int:
    """Local index of the node nearest the layer centroid (NEST ``FindCenterElement``).

    Ties resolve to the lowest index (matching NEST).

    Parameters
    ----------
    layer : Layer
        A concrete position layer.

    Returns
    -------
    int
        The population-local index of the most central node.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> bp.spatial.center_element(bp.spatial.grid([4, 3], extent=[2.0, 1.5]))
        4
    """
    coords = layer.coords
    centroid = u.math.mean(coords, axis=0)
    d2 = u.math.sum((coords - centroid) ** 2, axis=-1)
    return int(jnp.argmin(u.get_magnitude(d2)))            # argmin -> first (lowest) on ties


def Distance(layer_a, layer_b):
    """Pairwise Euclidean distance between two layers (NEST ``Distance``).

    Parameters
    ----------
    layer_a, layer_b : Layer
        Concrete position layers.

    Returns
    -------
    Quantity
        ``(n_a, n_b)`` distances.
    """
    return pairwise_distance(layer_a.coords, layer_b.coords)


def nearest_element(layer, locations, find_all=False):
    """Local index of the node nearest each query location (NEST ``FindNearestElement``).

    Parameters
    ----------
    layer : Layer
        A concrete position layer.
    locations : sequence or Quantity
        A single coordinate ``(ndim,)`` or a list of coordinates ``(m, ndim)``. Bare floats
        are taken in micrometres.
    find_all : bool, optional
        When ``False`` (default), ties resolve to the lowest index. When ``True``, every node
        within ``|d - d_min| <= 1e-14 * d_min`` of the minimum is returned.

    Returns
    -------
    int or list
        For a single location: an ``int`` (``find_all=False``) or a list of ints
        (``find_all=True``). For a list of locations: a list with one such entry per location.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> bp.spatial.nearest_element(bp.spatial.grid([3, 1], extent=[3.0, 1.0]), [0.9, 0.0])
        2
    """
    coords = u.get_magnitude(layer.coords.to(_LEN))            # (n, ndim) bare micrometres
    locs = u.get_magnitude(_as_len(locations).to(_LEN))
    single = (locs.ndim == 1)
    queries = locs[None, :] if single else locs
    out = []
    for q in queries:
        d = np.sqrt(((coords - q[None, :]) ** 2).sum(axis=-1))
        dmin = float(d.min())
        if find_all:
            out.append([int(i) for i in np.nonzero(d - dmin <= 1e-14 * dmin)[0]])
        else:
            out.append(int(np.argmin(d)))                      # argmin -> lowest index on ties
    return out[0] if single else out


def select_nodes_by_mask(layer, anchor, mask):
    """Local indices of the nodes lying inside ``mask`` anchored at ``anchor`` (NEST ``SelectNodesByMask``).

    The mask is evaluated with ``anchor`` as the (single) source node and every layer node as a
    candidate target, so directional masks (``box`` / ``rectangular`` / rotated ellipses) respect
    the ``target - anchor`` displacement.

    Parameters
    ----------
    layer : Layer
        A concrete position layer.
    anchor : sequence or Quantity
        The reference point ``(ndim,)`` the mask is centred on. Bare floats are micrometres.
    mask : object
        Any spatial mask exposing ``contains(pre_pos, post_pos) -> bool (n_pre, n_post)``.

    Returns
    -------
    numpy.ndarray
        The population-local indices (ascending) of the selected nodes.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> layer = bp.spatial.grid([3, 3], extent=[2.0, 2.0])
        >>> bp.spatial.select_nodes_by_mask(layer, [0.0, 0.0], bp.spatial.circular(0.7)).tolist()
        [1, 3, 4, 5, 7]
    """
    anchor_pos = _as_len(anchor)[None, :]                      # (1, ndim) source
    inside = np.asarray(mask.contains(anchor_pos, layer.coords))[0]
    return np.nonzero(inside)[0]


def dump_layer_nodes(sim, pop, outname) -> str:
    """Write each node's local index + coordinates to ``outname`` (NEST ``DumpLayerNodes``).

    One whitespace-separated line per node: ``idx x y [z]`` (population-local index; coordinates
    in micrometre magnitude, ascending node order).

    Parameters
    ----------
    sim : Simulator
        The simulator holding the population's positions.
    pop : NodeView
        A population/view created with ``create(positions=...)``.
    outname : str
        Path of the text file to write.

    Returns
    -------
    str
        The written text (also persisted to ``outname``), for direct assertion.
    """
    coords = np.asarray(u.get_magnitude(sim.get_position(pop).to(_LEN)))
    lines = [' '.join([str(i)] + [_fmt(v) for v in row]) for i, row in enumerate(coords)]
    text = '\n'.join(lines) + '\n'
    with open(outname, 'w') as fh:
        fh.write(text)
    return text


def dump_layer_connections(sim, source, target, outname) -> str:
    """Write each realized edge's endpoints, weight, delay and displacement (NEST ``DumpLayerConnections``).

    One whitespace-separated line per edge: ``src tgt weight delay dx dy [dz]`` where
    ``(dx, dy[, dz]) = target_pos - source_pos`` (the source-anchored displacement). Weight is in
    pA magnitude, delay in ms magnitude, coordinates in micrometres.

    Parameters
    ----------
    sim : Simulator
        The simulator holding the realized connections and positions.
    source, target : NodeView
        The source and target populations/views (created with ``positions=...``).
    outname : str
        Path of the text file to write.

    Returns
    -------
    str
        The written text (also persisted to ``outname``), for direct assertion.
    """
    sc = sim.get_connections(source=source, target=target)
    src = np.asarray(sc.source)
    tgt = np.asarray(sc.target)
    weight = np.asarray(u.get_magnitude(sc.get('weight').to(u.pA)))
    delay = np.asarray(u.get_magnitude(sc.get('delay').to(u.ms)))
    spos = np.asarray(u.get_magnitude(sim.get_position(source).to(_LEN)))
    tpos = np.asarray(u.get_magnitude(sim.get_position(target).to(_LEN)))
    lines = []
    for s, t, w, d in zip(src, tgt, weight, delay):
        disp = tpos[t] - spos[s]
        lines.append(' '.join([str(int(s)), str(int(t)), _fmt(w), _fmt(d)]
                              + [_fmt(v) for v in disp]))
    text = ('\n'.join(lines) + '\n') if lines else ''
    with open(outname, 'w') as fh:
        fh.write(text)
    return text


def target_nodes(sim, source, target):
    r"""Realized target indices of each source node (NEST ``GetTargetNodes``).

    Reads the built network's adjacency back out (via
    :meth:`~brainpy_state._network._simulator.Simulator.get_connections`) and groups
    the realized target indices by source node.

    Parameters
    ----------
    sim : Simulator
        The simulator holding the realized connections.
    source : NodeView
        A single-segment source view; targets are grouped per node in this view's order.
    target : NodeView
        The candidate-target population view.

    Returns
    -------
    list of numpy.ndarray
        One entry per source node (in ``source`` order): the sorted unique
        population-local target indices that node connects to.

    See Also
    --------
    target_positions : the same query returning target coordinates.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> sim = bp.Simulator(dt=0.1 * u.ms)
        >>> pop = sim.create(bp.iaf_psc_alpha, positions=bp.spatial.grid([3, 1], extent=[3.0, 1.0]))
        >>> _ = sim.connect(pop, pop,
        ...     rule=bp.spatial.spatial_pairwise_bernoulli(p=1.0, mask=bp.spatial.circular(1.2)),
        ...     weight=1.0 * u.pA, delay=1.0 * u.ms)
        >>> [t.tolist() for t in bp.spatial.target_nodes(sim, pop, pop)]
        [[0, 1], [0, 1, 2], [1, 2]]
    """
    sc = sim.get_connections(source=source, target=target)
    src = np.asarray(sc.source)
    tgt = np.asarray(sc.target)
    return [np.unique(tgt[src == int(s)]) for s in source.segments[0].indices]


def target_positions(sim, source, target):
    r"""Coordinates of each source node's realized targets (NEST ``GetTargetPositions``).

    Parameters
    ----------
    sim : Simulator
        The simulator holding the realized connections and target positions.
    source : NodeView
        A single-segment source view (one entry is returned per node).
    target : NodeView
        The candidate-target population view (created with ``positions=``).

    Returns
    -------
    list of Quantity
        One ``(k_i, ndim)`` coordinate array per source node, in ``source`` order.

    See Also
    --------
    target_nodes : the underlying realized-target index query.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> sim = bp.Simulator(dt=0.1 * u.ms)
        >>> pop = sim.create(bp.iaf_psc_alpha, positions=bp.spatial.grid([3, 1], extent=[3.0, 1.0]))
        >>> _ = sim.connect(pop, pop,
        ...     rule=bp.spatial.spatial_pairwise_bernoulli(p=1.0, mask=bp.spatial.circular(1.2)),
        ...     weight=1.0 * u.pA, delay=1.0 * u.ms)
        >>> [tuple(p.shape) for p in bp.spatial.target_positions(sim, pop, pop)]
        [(2, 2), (3, 2), (2, 2)]
    """
    coords = sim._positions[id(target.segments[0].population)]
    return [coords[idx] for idx in target_nodes(sim, source, target)]
