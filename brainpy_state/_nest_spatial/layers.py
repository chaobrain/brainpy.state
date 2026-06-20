# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Position layers for spatial connectivity.

Mirrors NEST's ``nest.spatial.grid`` / ``nest.spatial.free``: a :class:`Layer`
carries node positions in 2-D or 3-D space (length units), which
:meth:`brainpy_state.Simulator.create` consumes through its ``positions=`` keyword.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import jax.numpy as jnp
import brainunit as u

from brainpy_state._dist import Distribution

__all__ = ['Layer', 'grid', 'free']

# Canonical length unit for bare-float coordinate / extent inputs.
_LEN = u.um


def _as_len(x):
    """Promote a bare float/array to the canonical length unit; pass a Quantity through."""
    if isinstance(x, u.Quantity):
        return x
    return jnp.asarray(x, dtype=float) * _LEN


@dataclass(frozen=True)
class Layer:
    """A set of node positions in 2-D or 3-D space.

    Parameters
    ----------
    coords : Quantity or None
        ``(n, d)`` length-unit positions, or ``None`` for a deferred ``free`` layer
        (positions drawn at :meth:`brainpy_state.Simulator.create`).
    ndim : int
        Number of spatial dimensions (2 or 3).
    shape, extent, center : tuple, Quantity, Quantity, optional
        Grid metadata (``None`` for free layers).
    sampler : callable, optional
        ``(n, key) -> Quantity (n, d)`` for a deferred free layer.
    """
    coords: u.Quantity | None
    ndim: int
    shape: tuple | None = None
    extent: u.Quantity | None = None
    center: u.Quantity | None = None
    sampler: Callable[..., u.Quantity] | None = None

    @property
    def is_deferred(self) -> bool:
        """Whether positions are drawn lazily (free layer built from a distribution)."""
        return self.coords is None

    @property
    def n(self) -> int:
        """Number of nodes (raises for a deferred layer — pass ``size`` to ``create``)."""
        if self.coords is None:
            raise ValueError(
                'a deferred free layer has no fixed node count; pass an explicit '
                'size to create(size, positions=free(distribution, ...)).'
            )
        return int(self.coords.shape[0])

    def sample(self, n: int, key) -> u.Quantity:
        """Resolve concrete coordinates.

        A concrete layer returns its stored ``coords``; a deferred free layer draws
        ``n`` positions from its distribution under ``key``.
        """
        if self.coords is not None:
            return self.coords
        # A deferred layer (coords is None) is always constructed with a sampler.
        assert self.sampler is not None, 'a deferred layer must carry a sampler'
        return self.sampler(n, key)


def grid(shape: Sequence[int], extent=None, center=None) -> Layer:
    """Build a regular cell-centered lattice (NEST ``nest.spatial.grid``).

    Node ``k`` runs with the first axis slowest (``k = col * ny [* nz] + ...``); the
    ``y`` axis decreases from top to bottom, matching NEST's raster order.

    Parameters
    ----------
    shape : sequence of int
        Two- or three-element grid shape ``[nx, ny]`` or ``[nx, ny, nz]``.
    extent : sequence or Quantity, optional
        Physical size per dimension; defaults to a unit box ``[1.0] * ndim``.
    center : sequence or Quantity, optional
        Layer center; defaults to the origin.

    Returns
    -------
    Layer
        A concrete layer of ``prod(shape)`` cell-centered positions.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> layer = bp.spatial.grid([4, 3], extent=[2.0, 1.5])
        >>> layer.n
        12
        >>> bool(u.math.allclose(layer.coords[0], [-0.75, 0.5] * u.um))
        True
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    if ndim not in (2, 3):
        raise ValueError(f'grid shape must be 2-D or 3-D, got shape={shape}')
    extent = _as_len([1.0] * ndim if extent is None else extent)
    center = _as_len([0.0] * ndim if center is None else center)
    L = u.get_magnitude(extent.to(_LEN))
    C = u.get_magnitude(center.to(_LEN))
    axes = []
    for a, n in enumerate(shape):
        step = L[a] / n
        if a == 1:                       # y axis: top (max) -> bottom (min)
            axes.append(C[a] + L[a] / 2 - (jnp.arange(n) + 0.5) * step)
        else:                            # x (and z) axis: min -> max
            axes.append(C[a] - L[a] / 2 + (jnp.arange(n) + 0.5) * step)
    mesh = jnp.meshgrid(*axes, indexing='ij')                  # first axis slowest
    coords = jnp.stack([m.reshape(-1) for m in mesh], axis=-1) * _LEN
    return Layer(coords=coords, ndim=ndim, shape=shape, extent=extent, center=center)


def free(positions, extent=None, num_dimensions=None) -> Layer:
    """Build a free-position layer (NEST ``nest.spatial.free``).

    Parameters
    ----------
    positions : array-like or Distribution
        An explicit ``(n, d)`` position array, or a :class:`brainpy_state._dist.Distribution`
        whose per-coordinate range positions are drawn from at create-time.
    extent : sequence or Quantity, optional
        Physical bounding box per dimension (its length sets ``ndim`` for a distribution).
    num_dimensions : int, optional
        Number of dimensions when a distribution is given without ``extent``.

    Returns
    -------
    Layer
        A concrete layer (explicit array) or a deferred layer (distribution).

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> layer = bp.spatial.free(bp.dist.Uniform(-0.5, 0.5), extent=[1.5, 1.5, 1.5])
        >>> layer.ndim
        3
    """
    if isinstance(positions, Distribution):
        if extent is not None and num_dimensions is not None:
            raise TypeError('give extent OR num_dimensions for a free distribution layer, not both')
        ndim = len(extent) if extent is not None else num_dimensions
        if ndim not in (2, 3):
            raise ValueError(
                'could not infer 2-D/3-D; set extent or num_dimensions when using a distribution'
            )

        def sampler(n, key):
            return _as_len(positions.sample((n, ndim), key))

        return Layer(coords=None, ndim=ndim, sampler=sampler,
                     extent=_as_len(extent) if extent is not None else None)

    coords = _as_len(positions)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError('free positions array must have shape (n, 2) or (n, 3)')
    return Layer(coords=coords, ndim=int(coords.shape[1]),
                 extent=_as_len(extent) if extent is not None else None)
