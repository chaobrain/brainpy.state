# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Spatially-structured connectivity for brainpy.state.

Mirrors NEST's ``nest.spatial`` / ``nest.spatial_distributions`` surface: position
layers (:func:`grid`, :func:`free`), a distance kernel (:func:`gaussian`) and the
:data:`distance` sentinel, spatial masks (:func:`circular`, :func:`spherical`,
:func:`box`), the :func:`spatial_pairwise_bernoulli` connection rule, and query
helpers (:func:`center_element`, :func:`Distance`). Re-exported as ``brainpy.state.spatial``.
"""
# ---------------------------------------------------------------------------
# Position layers
# ---------------------------------------------------------------------------
from ._layers import Layer, grid, free
# ---------------------------------------------------------------------------
# Distance / displacement
# ---------------------------------------------------------------------------
from ._distance import displacement, pairwise_distance
# ---------------------------------------------------------------------------
# Distance kernels + the distance sentinel
# ---------------------------------------------------------------------------
from ._kernels import distance, gaussian
# ---------------------------------------------------------------------------
# Spatial masks
# ---------------------------------------------------------------------------
from ._masks import circular, spherical, box
# ---------------------------------------------------------------------------
# Connection rule
# ---------------------------------------------------------------------------
from ._rule import SpatialConnRule, spatial_pairwise_bernoulli
# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
from ._helpers import center_element, Distance, target_nodes, target_positions

__all__ = [
    'Layer',
    'grid',
    'free',
    'displacement',
    'pairwise_distance',
    'distance',
    'gaussian',
    'circular',
    'spherical',
    'box',
    'SpatialConnRule',
    'spatial_pairwise_bernoulli',
    'center_element',
    'Distance',
    'target_nodes',
    'target_positions',
]
