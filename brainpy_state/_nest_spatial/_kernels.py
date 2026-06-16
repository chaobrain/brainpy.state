# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Distance-dependent connection kernels (mirrors ``nest.spatial_distributions``).

A kernel is a callable ``p(d) -> probability`` over a (pairwise) distance. The
``distance`` sentinel mirrors NEST's ``nest.spatial.distance`` so kernels read
``gaussian(distance, std=...)``.

Kernels consume an *expression* -- by default the scalar :data:`distance`, but also the
per-axis ``distance.x/.y/.z`` or the ``source_pos``/``target_pos`` accessors. Every
expression evaluates, given the rule's bound sliced positions ``pre_pos (n_pre, d)`` /
``post_pos (n_post, d)``, to an ``(n_pre, n_post)`` grid; every kernel exposes
``_eval_pair(pre_pos, post_pos)`` returning the probability grid, which
:class:`~brainpy_state._nest_spatial._rule.SpatialConnRule` samples (zero seam change).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
from jax.scipy.special import gammaln
import brainunit as u

from brainpy_state._nest_spatial._distance import displacement, pairwise_distance
from brainpy_state._nest_spatial._layers import _LEN, _as_len

__all__ = ['distance', 'pos', 'source_pos', 'target_pos',
           'gaussian', 'exponential', 'gamma', 'gabor', 'gaussian2D']

_AXES = ('x', 'y', 'z')


# ---------------------------------------------------------------------------
# Expression family (axis / scalar values over the (n_pre, n_post) pair grid)
# ---------------------------------------------------------------------------
class _Expr:
    """Base spatial expression: evaluate to an ``(n_pre, n_post)`` grid from bound positions."""
    __module__ = 'brainpy.state'

    def _eval_pair(self, pre_pos, post_pos):
        raise NotImplementedError

    def _eval_nodes(self, coords):
        raise ValueError('this expression has no single-node value; it is only defined '
                         'over connected (source, target) pairs')


class _AxisDistance(_Expr):
    """Absolute per-axis distance ``|target_a - source_a|`` (NEST ``distance.x/.y/.z``)."""

    def __init__(self, axis):
        self.axis = int(axis)

    def _eval_pair(self, pre_pos, post_pos):
        ndim = pre_pos.shape[-1]
        if self.axis >= ndim:
            raise ValueError(
                f'distance.{_AXES[self.axis]} needs a {self.axis + 1}-D layer, got {ndim}-D')
        disp = displacement(pre_pos, post_pos)            # (n_pre, n_post, d)
        return u.math.abs(disp[..., self.axis])

    def __repr__(self):
        return f'spatial.distance.{_AXES[self.axis]}'


class _DistanceSentinel(_Expr):
    """Pairwise Euclidean distance in ``gaussian(distance, std=...)`` (NEST ``spatial.distance``)."""
    __module__ = 'brainpy.state'

    def _eval_pair(self, pre_pos, post_pos):
        return pairwise_distance(pre_pos, post_pos)

    @property
    def x(self):
        """Per-axis absolute distance on the x-axis (NEST ``distance.x``)."""
        return _AxisDistance(0)

    @property
    def y(self):
        """Per-axis absolute distance on the y-axis (NEST ``distance.y``)."""
        return _AxisDistance(1)

    @property
    def z(self):
        """Per-axis absolute distance on the z-axis (NEST ``distance.z``)."""
        return _AxisDistance(2)

    def __repr__(self):
        return 'spatial.distance'


#: Singleton representing the pairwise Euclidean distance between two nodes.
distance = _DistanceSentinel()


class _SourcePos(_Expr):
    """Source-node position on an axis, broadcast over targets (NEST ``source_pos.x/.y/.z``)."""

    def __init__(self, axis):
        self.axis = int(axis)

    def _eval_pair(self, pre_pos, post_pos):
        if self.axis >= pre_pos.shape[-1]:
            raise ValueError(f'source_pos.{_AXES[self.axis]} needs a {self.axis + 1}-D layer')
        col = pre_pos[:, self.axis][:, None]              # (n_pre, 1)
        return u.math.broadcast_to(col, (pre_pos.shape[0], post_pos.shape[0]))

    def __repr__(self):
        return f'spatial.source_pos.{_AXES[self.axis]}'


class _TargetPos(_Expr):
    """Target-node position on an axis, broadcast over sources (NEST ``target_pos.x/.y/.z``)."""

    def __init__(self, axis):
        self.axis = int(axis)

    def _eval_pair(self, pre_pos, post_pos):
        if self.axis >= post_pos.shape[-1]:
            raise ValueError(f'target_pos.{_AXES[self.axis]} needs a {self.axis + 1}-D layer')
        row = post_pos[:, self.axis][None, :]             # (1, n_post)
        return u.math.broadcast_to(row, (pre_pos.shape[0], post_pos.shape[0]))

    def __repr__(self):
        return f'spatial.target_pos.{_AXES[self.axis]}'


class _NodePos(_Expr):
    """Single-node position on an axis (NEST ``pos.x/.y/.z``); invalid in the connect path."""

    def __init__(self, axis):
        self.axis = int(axis)

    def _eval_pair(self, pre_pos, post_pos):
        raise ValueError(
            'pos.{a} is a single-node position parameter and cannot be used when connecting; '
            'use source_pos.{a} / target_pos.{a} (two-node) or distance.{a}'.format(
                a=_AXES[self.axis]))

    def _eval_nodes(self, coords):
        if self.axis >= coords.shape[-1]:
            raise ValueError(f'pos.{_AXES[self.axis]} needs a {self.axis + 1}-D layer')
        return coords[:, self.axis]

    def __repr__(self):
        return f'spatial.pos.{_AXES[self.axis]}'


class _AxisHolder:
    """Exposes ``.x/.y/.z`` building a given per-axis expression class (NEST ``pos`` etc.)."""
    __module__ = 'brainpy.state'

    def __init__(self, factory, name):
        self._factory = factory
        self._name = name

    @property
    def x(self):
        """Per-axis expression on the x-axis."""
        return self._factory(0)

    @property
    def y(self):
        """Per-axis expression on the y-axis."""
        return self._factory(1)

    @property
    def z(self):
        """Per-axis expression on the z-axis."""
        return self._factory(2)

    def __repr__(self):
        return f'spatial.{self._name}'


#: Per-node position accessors (NEST ``nest.spatial.pos`` / ``source_pos`` / ``target_pos``).
pos = _AxisHolder(_NodePos, 'pos')
source_pos = _AxisHolder(_SourcePos, 'source_pos')
target_pos = _AxisHolder(_TargetPos, 'target_pos')


def _as_input(x):
    """Validate a kernel input expression (anything evaluating over a pair grid)."""
    if not hasattr(x, '_eval_pair'):
        raise ValueError(
            'kernel input must be a spatial expression (spatial.distance, distance.x/.y/.z, '
            'source_pos/target_pos.x/.y/.z)')
    return x


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
class _GaussianKernel:
    r"""Gaussian distance kernel ``p(d) = \exp(-(d-\mu)^2 / (2\,\mathrm{std}^2))`` (peak 1 at ``d=\mu``)."""
    __module__ = 'brainpy.state'

    def __init__(self, std, mean=0.0, x=distance):
        self.std = _as_len(std)
        self.mean = _as_len(mean)
        self._input = _as_input(x)

    def __call__(self, d):
        r = (d - self.mean) / self.std                    # dimensionless ratio
        return u.math.exp(-(r ** 2) / 2.0)

    def _eval_pair(self, pre_pos, post_pos):
        return self(self._input._eval_pair(pre_pos, post_pos))


def gaussian(x=distance, mean=0.0, std=1.0) -> _GaussianKernel:
    r"""Gaussian distance-dependent connection probability.

    Returns a callable ``p(d) = exp(-(d-mean)^2 / (2 std^2))`` matching NEST's
    ``nest.spatial_distributions.gaussian(distance, mean, std)``.

    Parameters
    ----------
    x : object, optional
        The :data:`distance` sentinel (or a per-axis expression such as ``distance.x``).
    mean : float or Quantity, optional
        Distribution mean (length); bare floats are taken in micrometres. Default ``0``.
    std : float or Quantity, optional
        Standard deviation (length); bare floats are taken in micrometres. Default ``1``.

    Returns
    -------
    callable
        ``p(d)`` mapping a distance (Quantity) to a connection probability.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> p = bp.spatial.gaussian(bp.spatial.distance, std=0.5)
        >>> float(u.get_magnitude(p(0.0 * u.um)))
        1.0
    """
    return _GaussianKernel(std, mean=mean, x=x)


class _ExponentialKernel:
    r"""Exponential distance kernel ``p(d) = \exp(-d / \beta)`` (peak 1 at ``d=0``)."""
    __module__ = 'brainpy.state'

    def __init__(self, beta, x=distance):
        self.beta = _as_len(beta)
        self._input = _as_input(x)

    def __call__(self, d):
        return u.math.exp(-(d / self.beta))               # dimensionless ratio

    def _eval_pair(self, pre_pos, post_pos):
        return self(self._input._eval_pair(pre_pos, post_pos))


def exponential(x=distance, beta=1.0) -> _ExponentialKernel:
    r"""Exponential distance-dependent connection probability.

    Returns a callable ``p(d) = exp(-d / beta)`` matching NEST's
    ``nest.spatial_distributions.exponential(distance, beta)``.

    Parameters
    ----------
    x : object, optional
        The :data:`distance` sentinel (or a per-axis expression such as ``distance.x``).
    beta : float or Quantity, optional
        Decay length; bare floats are taken in micrometres. Default ``1``.

    Returns
    -------
    callable
        ``p(d)`` mapping a distance (Quantity) to a connection probability.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> p = bp.spatial.exponential(bp.spatial.distance, beta=2.0)
        >>> float(u.get_magnitude(p(0.0 * u.um)))
        1.0
    """
    return _ExponentialKernel(beta, x=x)


class _GammaKernel:
    r"""Gamma distance kernel ``p(d) = d^{\kappa-1} e^{-d/\theta} / (\theta^\kappa \Gamma(\kappa))``."""
    __module__ = 'brainpy.state'

    def __init__(self, kappa, theta, x=distance):
        self.kappa = float(kappa)
        self.theta = _as_len(theta)
        self._input = _as_input(x)

    def __call__(self, d):
        # Mirror NEST GammaParameter (bare magnitudes in the canonical length unit).
        x = u.get_magnitude(d.to(_LEN))
        th = float(u.get_magnitude(self.theta.to(_LEN)))
        delta = jnp.exp(-self.kappa * jnp.log(th) - gammaln(self.kappa))
        return x ** (self.kappa - 1.0) * jnp.exp(-x / th) * delta

    def _eval_pair(self, pre_pos, post_pos):
        return self(self._input._eval_pair(pre_pos, post_pos))


def gamma(x=distance, kappa=1.0, theta=1.0) -> _GammaKernel:
    r"""Gamma distance-dependent connection probability.

    Returns a callable ``p(d) = d^{kappa-1} exp(-d/theta) / (theta^kappa Gamma(kappa))``
    matching NEST's ``nest.spatial_distributions.gamma(distance, kappa, theta)``.

    Parameters
    ----------
    x : object, optional
        The :data:`distance` sentinel (or a per-axis expression such as ``distance.x``).
    kappa : float, optional
        Shape parameter. Default ``1``.
    theta : float or Quantity, optional
        Scale parameter (length); bare floats are taken in micrometres. Default ``1``.

    Returns
    -------
    callable
        ``p(d)`` mapping a distance (Quantity) to a connection probability.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> p = bp.spatial.gamma(bp.spatial.distance, kappa=2.0, theta=1.5)
        >>> float(u.get_magnitude(p(1.5 * u.um))) > 0.0
        True
    """
    return _GammaKernel(kappa, theta, x=x)


def _axis_mag(expr, pre_pos, post_pos):
    """Evaluate a two-node expression and return its bare magnitude in the canonical unit."""
    return u.get_magnitude(expr._eval_pair(pre_pos, post_pos).to(_LEN))


class _GaborKernel:
    r"""Rectified Gabor kernel on the (x, y) displacements (NEST ``gabor``)."""
    __module__ = 'brainpy.state'

    def __init__(self, x, y, theta=0.0, gamma=1.0, std=1.0, lam=1.0, psi=0.0):
        self._x = _as_input(x)
        self._y = _as_input(y)
        self.theta = float(theta)
        self.gamma = float(gamma)
        self.std = _as_len(std)
        self.lam = _as_len(lam)
        self.psi = float(psi)

    def _eval_pair(self, pre_pos, post_pos):
        X = _axis_mag(self._x, pre_pos, post_pos)
        Y = _axis_mag(self._y, pre_pos, post_pos)
        std = float(u.get_magnitude(self.std.to(_LEN)))
        lam = float(u.get_magnitude(self.lam.to(_LEN)))
        c, s = math.cos(math.radians(self.theta)), math.sin(math.radians(self.theta))
        xp = X * c + Y * s
        yp = -X * s + Y * c
        env = jnp.exp(-(self.gamma ** 2 * xp ** 2 + yp ** 2) / (2.0 * std ** 2))
        carrier = jnp.maximum(jnp.cos(2.0 * jnp.pi * yp / lam + math.radians(self.psi)), 0.0)
        return env * carrier


def gabor(x=None, y=None, theta=0.0, gamma=1.0, std=1.0, lam=1.0, psi=0.0) -> _GaborKernel:
    r"""Rectified-Gabor connection probability on the (x, y) displacements (NEST ``gabor``).

    ``p = [cos(2\pi y'/\lambda + \psi)]^+ \exp(-(\gamma^2 x'^2 + y'^2)/(2\,\mathrm{std}^2))`` with
    ``x' = x\cos\theta + y\sin\theta``, ``y' = -x\sin\theta + y\cos\theta`` (``\theta``, ``\psi``
    in degrees). The ``x`` / ``y`` inputs default to ``distance.x`` / ``distance.y`` (the absolute
    per-axis displacements), matching NEST.

    Parameters
    ----------
    x, y : object, optional
        Per-axis expressions for the x / y displacement. Default ``distance.x`` / ``distance.y``.
    theta : float, optional
        Orientation of the profile in degrees. Default ``0``.
    gamma : float, optional
        Spatial aspect ratio (major/minor axis). Default ``1``.
    std : float or Quantity, optional
        Envelope standard deviation (length). Default ``1``.
    lam : float or Quantity, optional
        Wavelength of the carrier (length). Default ``1``.
    psi : float, optional
        Carrier phase in degrees. Default ``0``.

    Returns
    -------
    callable
        A kernel with ``_eval_pair(pre, post)`` returning the ``(n_pre, n_post)`` probability grid.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> k = bp.spatial.gabor(bp.spatial.distance.x, bp.spatial.distance.y, theta=45.0, lam=2.0)
    """
    x = distance.x if x is None else x
    y = distance.y if y is None else y
    return _GaborKernel(x, y, theta=theta, gamma=gamma, std=std, lam=lam, psi=psi)


class _Gaussian2DKernel:
    r"""Bivariate Gaussian kernel on the (x, y) displacements (NEST ``gaussian2D``)."""
    __module__ = 'brainpy.state'

    def __init__(self, x, y, mean_x=0.0, mean_y=0.0, std_x=1.0, std_y=1.0, rho=0.0):
        self._x = _as_input(x)
        self._y = _as_input(y)
        self.mean_x = _as_len(mean_x)
        self.mean_y = _as_len(mean_y)
        self.std_x = _as_len(std_x)
        self.std_y = _as_len(std_y)
        self.rho = float(rho)

    def _eval_pair(self, pre_pos, post_pos):
        dx = _axis_mag(self._x, pre_pos, post_pos) - float(u.get_magnitude(self.mean_x.to(_LEN)))
        dy = _axis_mag(self._y, pre_pos, post_pos) - float(u.get_magnitude(self.mean_y.to(_LEN)))
        sx = float(u.get_magnitude(self.std_x.to(_LEN)))
        sy = float(u.get_magnitude(self.std_y.to(_LEN)))
        denom = 2.0 * (1.0 - self.rho ** 2)
        cx = 1.0 / (denom * sx ** 2)
        cy = 1.0 / (denom * sy ** 2)
        cxy = 2.0 * self.rho / (denom * sx * sy)
        return jnp.exp(-dx ** 2 * cx - dy ** 2 * cy + dx * dy * cxy)


def gaussian2D(x=None, y=None, mean_x=0.0, mean_y=0.0,
               std_x=1.0, std_y=1.0, rho=0.0) -> _Gaussian2DKernel:
    r"""Bivariate-Gaussian connection probability on the (x, y) displacements (NEST ``gaussian2D``).

    ``p = exp(-(u^2 - 2\rho u v + v^2)/(2(1-\rho^2)))`` with ``u=(x-mean_x)/std_x``,
    ``v=(y-mean_y)/std_y``. The ``x`` / ``y`` inputs default to ``distance.x`` / ``distance.y``.

    Parameters
    ----------
    x, y : object, optional
        Per-axis expressions for the x / y displacement. Default ``distance.x`` / ``distance.y``.
    mean_x, mean_y : float or Quantity, optional
        Means (length). Default ``0``.
    std_x, std_y : float or Quantity, optional
        Standard deviations (length). Default ``1``.
    rho : float, optional
        Correlation in ``(-1, 1)``. Default ``0``.

    Returns
    -------
    callable
        A kernel with ``_eval_pair(pre, post)`` returning the ``(n_pre, n_post)`` probability grid.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> k = bp.spatial.gaussian2D(std_x=0.5, std_y=1.0, rho=0.3)
    """
    x = distance.x if x is None else x
    y = distance.y if y is None else y
    return _Gaussian2DKernel(x, y, mean_x=mean_x, mean_y=mean_y,
                             std_x=std_x, std_y=std_y, rho=rho)
