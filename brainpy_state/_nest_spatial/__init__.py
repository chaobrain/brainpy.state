# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Spatially-structured connectivity for brainpy.state.

Mirrors NEST's ``nest.spatial`` / ``nest.spatial_distributions`` surface, re-exported as
``brainpy.state.spatial``:

- **Position layers** — :func:`grid`, :func:`free`.
- **Distance / position expressions** — the :data:`distance` sentinel (with per-axis
  :data:`distance.x` / ``.y`` / ``.z``) plus the :data:`pos` / :data:`source_pos` /
  :data:`target_pos` coordinate accessors.
- **Distance distributions** — :func:`gaussian`, :func:`exponential`, :func:`gamma`, and the
  anisotropic :func:`gabor` / :func:`gaussian2D`.
- **Masks** — :func:`circular`, :func:`spherical`, :func:`box`, :func:`rectangular`,
  :func:`doughnut`, :func:`elliptical`, :func:`ellipsoidal`.
- **Connection rule** — :func:`spatial_pairwise_bernoulli`.
- **Queries** — :func:`center_element`, :func:`nearest_element`, :func:`Distance`,
  :func:`select_nodes_by_mask`, :func:`target_nodes`, :func:`target_positions`.
- **Dump / plot** — :func:`dump_layer_nodes`, :func:`dump_layer_connections`, and the
  matplotlib-gated :func:`plot_layer` / :func:`plot_targets` / :func:`plot_sources` /
  :func:`plot_probability_parameter`.
"""
# ---------------------------------------------------------------------------
# Position layers
# ---------------------------------------------------------------------------
from .layers import Layer, grid, free
# ---------------------------------------------------------------------------
# Distance / displacement
# ---------------------------------------------------------------------------
from .distance import displacement, pairwise_distance
# ---------------------------------------------------------------------------
# Distance kernels + the distance sentinel / position expressions
# ---------------------------------------------------------------------------
from .kernels import (distance, pos, source_pos, target_pos,
                       gaussian, exponential, gamma, gabor, gaussian2D)
# ---------------------------------------------------------------------------
# Spatial masks
# ---------------------------------------------------------------------------
from .masks import (circular, spherical, box,
                     rectangular, doughnut, elliptical, ellipsoidal)
# ---------------------------------------------------------------------------
# Connection rule
# ---------------------------------------------------------------------------
from .rule import SpatialConnRule, spatial_pairwise_bernoulli
# ---------------------------------------------------------------------------
# Query / dump helpers
# ---------------------------------------------------------------------------
from .helpers import (center_element, Distance, nearest_element, select_nodes_by_mask,
                       dump_layer_nodes, dump_layer_connections, target_nodes, target_positions)
# ---------------------------------------------------------------------------
# Plot helpers (matplotlib lazily imported inside each function)
# ---------------------------------------------------------------------------
from .plot import plot_layer, plot_targets, plot_sources, plot_probability_parameter

__all__ = [
    'Layer',
    'grid',
    'free',
    'displacement',
    'pairwise_distance',
    'distance',
    'pos',
    'source_pos',
    'target_pos',
    'gaussian',
    'exponential',
    'gamma',
    'gabor',
    'gaussian2D',
    'circular',
    'spherical',
    'box',
    'rectangular',
    'doughnut',
    'elliptical',
    'ellipsoidal',
    'SpatialConnRule',
    'spatial_pairwise_bernoulli',
    'center_element',
    'Distance',
    'nearest_element',
    'select_nodes_by_mask',
    'dump_layer_nodes',
    'dump_layer_connections',
    'target_nodes',
    'target_positions',
    'plot_layer',
    'plot_targets',
    'plot_sources',
    'plot_probability_parameter',
]
