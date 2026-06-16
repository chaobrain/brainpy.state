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

import brainunit as u

from brainpy_state._nest_spatial._distance import displacement, pairwise_distance
from brainpy_state._nest_spatial._layers import _as_len

__all__ = ['distance', 'gaussian']

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
