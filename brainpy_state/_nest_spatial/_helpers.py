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

__all__ = ['center_element', 'Distance', 'target_nodes', 'target_positions']


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
