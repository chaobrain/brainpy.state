# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spatial query helpers (NEST ``FindCenterElement`` / ``Distance``).

``target_positions`` / ``target_nodes`` (which read realized adjacency back out of a
built network via :meth:`brainpy_state.Simulator.get_connections`) are added once the
Simulator coord-bind seam exists.
"""
from __future__ import annotations

import jax.numpy as jnp
import brainunit as u

from brainpy_state._nest_spatial._distance import pairwise_distance

__all__ = ['center_element', 'Distance']


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
