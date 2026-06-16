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

from brainpy_state._nest_spatial.distance import displacement, pairwise_distance
from brainpy_state._nest_spatial.layers import _LEN, _as_len

__all__ = ['circular', 'spherical', 'box', 'rectangular', 'doughnut',
           'elliptical', 'ellipsoidal']


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


class _EllipseMask:
    """Rotated ellipse (2-D) / ellipsoid (3-D) on the displacement (NEST ``elliptical`` / ``ellipsoidal``).

    A target is included iff the source-anchored displacement, after rotation into the ellipse
    frame, satisfies ``sum_k (n_k / semi_k)**2 <= 1`` where ``semi = axis / 2``. Mirrors NEST's
    ``EllipseMask<2>``/``EllipseMask<3>`` (``mask.cpp``): the ``major``/``minor``/``polar`` arguments
    are the *full* axis lengths, the angles are in degrees, and the squared coordinates make the
    sign convention of the rotation immaterial.
    """
    __module__ = 'brainpy.state'

    def __init__(self, major_axis, minor_axis, polar_axis=None,
                 azimuth_angle=0.0, polar_angle=0.0, anchor=None):
        self.major = _as_len(major_axis)
        self.minor = _as_len(minor_axis)
        self.polar = _as_len(polar_axis) if polar_axis is not None else None
        self.azimuth_angle = float(azimuth_angle)
        self.polar_angle = float(polar_angle)
        self.anchor = _as_len(anchor) if anchor is not None else None
        self.ndim = 2 if polar_axis is None else 3

    def contains(self, pre_pos, post_pos):
        """Boolean ``(n_pre, n_post)``: displacement inside the (rotated) ellipse / ellipsoid."""
        disp = displacement(pre_pos, post_pos)               # (n_pre, n_post, d) Quantity
        d = u.get_magnitude(disp.to(_LEN))                   # bare micrometres
        if self.anchor is not None:
            d = d - u.get_magnitude(self.anchor.to(_LEN))
        dx, dy = d[..., 0], d[..., 1]
        az = math.radians(self.azimuth_angle)
        ac, asn = math.cos(az), math.sin(az)
        maj = float(u.get_magnitude(self.major.to(_LEN)))
        mn = float(u.get_magnitude(self.minor.to(_LEN)))
        xs, ys = 4.0 / maj ** 2, 4.0 / mn ** 2               # (n/semi)**2 = n**2 * 4/axis**2
        if self.ndim == 2:
            nx = dx * ac + dy * asn
            ny = dx * asn - dy * ac
            return (nx ** 2 * xs + ny ** 2 * ys) <= 1.0
        dz = d[..., 2]
        pol = math.radians(self.polar_angle)
        pc, ps = math.cos(pol), math.sin(pol)
        plr = float(u.get_magnitude(self.polar.to(_LEN)))
        zs = 4.0 / plr ** 2
        base = dx * ac + dy * asn
        nx = base * pc - dz * ps
        ny = dx * asn - dy * ac
        nz = base * ps + dz * pc
        return (nx ** 2 * xs + ny ** 2 * ys + nz ** 2 * zs) <= 1.0


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


def elliptical(major_axis, minor_axis, azimuth_angle=0.0, anchor=None) -> _EllipseMask:
    """Elliptical mask (2-D): displacement inside a rotated ellipse (NEST ``elliptical``).

    Parameters
    ----------
    major_axis, minor_axis : float or Quantity
        Full lengths of the two principal axes (not semi-axes). ``major_axis == minor_axis``
        degenerates to :func:`circular` of radius ``major_axis / 2``.
    azimuth_angle : float, optional
        Rotation of the ellipse about its anchor, in degrees. Default ``0``.
    anchor : sequence of float or Quantity, optional
        Offset of the ellipse centre from the source node (on the displacement). Default origin.

    Returns
    -------
    _EllipseMask
        A hard-cutoff elliptical mask.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> mask = bp.spatial.elliptical(4.0, 2.0, azimuth_angle=45.0)
    """
    return _EllipseMask(major_axis, minor_axis, azimuth_angle=azimuth_angle, anchor=anchor)


def ellipsoidal(major_axis, minor_axis, polar_axis, azimuth_angle=0.0,
                polar_angle=0.0, anchor=None) -> _EllipseMask:
    """Ellipsoidal mask (3-D): displacement inside a rotated ellipsoid (NEST ``ellipsoidal``).

    Parameters
    ----------
    major_axis, minor_axis, polar_axis : float or Quantity
        Full lengths of the three principal axes (not semi-axes). All three equal degenerates to
        :func:`spherical` of radius ``major_axis / 2``.
    azimuth_angle : float, optional
        Rotation about the polar (z) axis, in degrees. Default ``0``.
    polar_angle : float, optional
        Tilt of the polar axis away from z, in degrees (applied after the azimuthal rotation,
        NEST ``R_y(-polar) . R_z(-azimuth)``). Default ``0``.
    anchor : sequence of float or Quantity, optional
        Offset of the ellipsoid centre from the source node. Default origin.

    Returns
    -------
    _EllipseMask
        A hard-cutoff ellipsoidal mask.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> mask = bp.spatial.ellipsoidal(4.0, 2.0, 1.0, azimuth_angle=30.0, polar_angle=15.0)
    """
    return _EllipseMask(major_axis, minor_axis, polar_axis=polar_axis,
                        azimuth_angle=azimuth_angle, polar_angle=polar_angle, anchor=anchor)
