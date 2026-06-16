# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Distance-dependent connection kernels (mirrors ``nest.spatial_distributions``).

A kernel is a callable ``p(d) -> probability`` over a (pairwise) distance. The
``distance`` sentinel mirrors NEST's ``nest.spatial.distance`` so kernels read
``gaussian(distance, std=...)``.
"""
from __future__ import annotations

import brainunit as u

from brainpy_state._nest_spatial._layers import _as_len

__all__ = ['distance', 'gaussian']


class _DistanceSentinel:
    """Placeholder for the pairwise distance in ``gaussian(distance, std=...)`` (NEST parity)."""
    __module__ = 'brainpy.state'

    def __repr__(self):
        return 'spatial.distance'


#: Singleton representing the pairwise Euclidean distance between two nodes.
distance = _DistanceSentinel()


class _GaussianKernel:
    r"""Gaussian distance kernel ``p(d) = \exp(-d^2 / (2\,\mathrm{std}^2))`` (peak 1 at ``d=0``)."""
    __module__ = 'brainpy.state'

    def __init__(self, std):
        self.std = _as_len(std)

    def __call__(self, d):
        r = d / self.std                      # dimensionless ratio
        return u.math.exp(-(r ** 2) / 2.0)


def gaussian(x=distance, std=1.0) -> _GaussianKernel:
    r"""Gaussian distance-dependent connection probability.

    Returns a callable ``p(d) = exp(-d^2 / (2 std^2))`` matching NEST's
    ``nest.spatial_distributions.gaussian(distance, std)`` (mean 0, no normalization).

    Parameters
    ----------
    x : object, optional
        The :data:`distance` sentinel (for API parity with NEST). Other inputs raise.
    std : float or Quantity, optional
        Standard deviation (length); bare floats are taken in micrometres.

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
    if x is not distance:
        raise ValueError(
            'gaussian currently supports only the spatial.distance sentinel, '
            'e.g. gaussian(spatial.distance, std=0.5)'
        )
    return _GaussianKernel(std)
