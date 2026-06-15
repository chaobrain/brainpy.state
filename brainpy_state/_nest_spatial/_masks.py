# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spatial connectivity masks.

A mask is a hard candidate cutoff anchored on the *source* node: ``contains`` returns
a boolean ``(n_pre, n_post)`` matrix selecting which target nodes are eligible. The
distance kernel ``p(d)`` then applies only within the mask. Mirrors NEST's
``{"circular": {"radius": r}}`` / ``{"spherical": {"radius": r}}`` /
``{"box": {"lower_left": ll, "upper_right": ur}}``.
"""
from __future__ import annotations

import brainunit as u

from brainpy_state._nest_spatial._distance import displacement, pairwise_distance
from brainpy_state._nest_spatial._layers import _as_len

__all__ = ['circular', 'spherical', 'box']


class _RadialMask:
    """Distance cutoff ``d <= radius`` (NEST ``circular`` in 2-D / ``spherical`` in 3-D)."""
    __module__ = 'brainpy.state'

    def __init__(self, radius):
        self.radius = _as_len(radius)

    def contains(self, pre_pos, post_pos):
        """Boolean ``(n_pre, n_post)``: target within ``radius`` of source (inclusive)."""
        return pairwise_distance(pre_pos, post_pos) <= self.radius


class _BoxMask:
    """Axis-aligned box on the displacement ``post - pre`` (NEST ``box``)."""
    __module__ = 'brainpy.state'

    def __init__(self, lower_left, upper_right):
        self.lower_left = _as_len(lower_left)
        self.upper_right = _as_len(upper_right)

    def contains(self, pre_pos, post_pos):
        """Boolean ``(n_pre, n_post)``: displacement within ``[lower_left, upper_right]``."""
        disp = displacement(pre_pos, post_pos)               # (n_pre, n_post, d)
        ge = u.math.all(disp >= self.lower_left, axis=-1)
        le = u.math.all(disp <= self.upper_right, axis=-1)
        return ge & le


def circular(radius) -> _RadialMask:
    """Circular mask (2-D): target within ``radius`` of source.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> mask = bp.spatial.circular(0.5)
    """
    return _RadialMask(radius)


def spherical(radius) -> _RadialMask:
    """Spherical mask (3-D): target within ``radius`` of source (same cutoff as circular)."""
    return _RadialMask(radius)


def box(lower_left, upper_right) -> _BoxMask:
    """Box mask (2-D/3-D): target displacement within ``[lower_left, upper_right]``.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> mask = bp.spatial.box([-0.75, -0.75, -0.75], [0.75, 0.75, 0.75])
    """
    return _BoxMask(lower_left, upper_right)
