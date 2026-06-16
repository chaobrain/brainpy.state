# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spatial connectivity masks.

A mask is a hard candidate cutoff anchored on the *source* node: ``contains`` returns
a boolean ``(n_pre, n_post)`` matrix selecting which target nodes are eligible. The
distance kernel ``p(d)`` then applies only within the mask. Mirrors NEST's
``{"circular": {"radius": r}}`` / ``{"spherical": {"radius": r}}`` /
``{"box": {"lower_left": ll, "upper_right": ur}}``.
"""
from __future__ import annotations

import math

import brainunit as u

from brainpy_state._nest_spatial._distance import displacement, pairwise_distance
from brainpy_state._nest_spatial._layers import _as_len

__all__ = ['circular', 'spherical', 'box', 'rectangular', 'doughnut']


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


def _rotate_disp_2d(disp, lower_left, upper_right, az_deg):
    """Rotate a 2-D displacement about the box center by ``R(-azimuth)`` (NEST ``BoxMask<2>``)."""
    c = (lower_left + upper_right) / 2.0
    az = math.radians(az_deg)
    cos, sin = math.cos(az), math.sin(az)
    rel_x = disp[..., 0] - c[0]
    rel_y = disp[..., 1] - c[1]
    new_x = rel_x * cos + rel_y * sin + c[0]
    new_y = -rel_x * sin + rel_y * cos + c[1]
    return u.math.stack([new_x, new_y], axis=-1)


class _DoughnutMask:
    """Annulus ``inner < d <= outer`` (NEST ``doughnut``: outer ball minus inner ball)."""
    __module__ = 'brainpy.state'

    def __init__(self, inner_radius, outer_radius):
        self.inner = _as_len(inner_radius)
        self.outer = _as_len(outer_radius)

    def contains(self, pre_pos, post_pos):
        """Boolean ``(n_pre, n_post)``: ``inner < d <= outer`` (inner exclusive, outer inclusive)."""
        d = pairwise_distance(pre_pos, post_pos)
        return (d > self.inner) & (d <= self.outer)


class _RectangularMask:
    """Axis-aligned (optionally rotated) box on the displacement ``post - pre`` (NEST ``rectangular``)."""
    __module__ = 'brainpy.state'

    def __init__(self, lower_left, upper_right, azimuth_angle=0.0):
        self.lower_left = _as_len(lower_left)
        self.upper_right = _as_len(upper_right)
        self.azimuth_angle = float(azimuth_angle)

    def contains(self, pre_pos, post_pos):
        """Boolean ``(n_pre, n_post)``: displacement within ``[lower_left, upper_right]`` (rotated)."""
        disp = displacement(pre_pos, post_pos)               # (n_pre, n_post, 2)
        if self.azimuth_angle != 0.0:
            disp = _rotate_disp_2d(disp, self.lower_left, self.upper_right, self.azimuth_angle)
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


def rectangular(lower_left, upper_right, azimuth_angle=0.0) -> _RectangularMask:
    """Rectangular mask (2-D): target displacement within ``[lower_left, upper_right]``.

    Parameters
    ----------
    lower_left, upper_right : sequence of float or Quantity
        The two corners of the (axis-aligned) rectangle on the source-anchored displacement.
    azimuth_angle : float, optional
        Rotation of the rectangle about its center, in degrees (NEST parity). Default ``0``.

    Returns
    -------
    _RectangularMask
        A hard-cutoff mask (the 2-D analogue of :func:`box`).

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> mask = bp.spatial.rectangular([-0.5, -0.5], [0.5, 0.5], azimuth_angle=30.0)
    """
    return _RectangularMask(lower_left, upper_right, azimuth_angle=azimuth_angle)


def doughnut(inner_radius, outer_radius) -> _DoughnutMask:
    """Doughnut (annulus) mask (2-D): ``inner_radius < d <= outer_radius``.

    The inner boundary is exclusive and the outer boundary inclusive (NEST's
    outer-ball-minus-inner-ball ``DifferenceMask``). ``inner_radius == outer_radius`` yields an
    empty mask; ``inner_radius == 0`` matches :func:`circular` except at the exact center.

    Parameters
    ----------
    inner_radius, outer_radius : float or Quantity
        Inner and outer radii (length); bare floats are taken in micrometres.

    Returns
    -------
    _DoughnutMask
        A hard-cutoff annulus mask.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> mask = bp.spatial.doughnut(0.3, 0.7)
    """
    return _DoughnutMask(inner_radius, outer_radius)
