# examples/nest_like/spatial_gaussex.py
"""Spatial networks: a Gaussian probabilistic kernel -- NEST ``spatial/gaussex.py`` port.

Creates two ``iaf_psc_alpha`` populations on a shared grid and connects them with a
distance-dependent pairwise-Bernoulli rule whose probability is a Gaussian of the
source->target distance,

.. math::  p(d) = \\exp\\!\\big(-d^2 / (2\\,\\mathrm{std}^2)\\big),

clipped to a circular mask of ``radius``. This is
``spatial.spatial_pairwise_bernoulli(p=spatial.gaussian(spatial.distance, std=...),
mask=spatial.circular(...))`` riding the ordinary :meth:`~brainpy.state.Simulator.connect`
-- the same call any non-spatial rule uses. The realized footprint of the central source
neuron is read back with :func:`~brainpy.state.spatial.target_positions` (NEST
``GetTargetPositions``).

The per-distance connection probability is a fixed law, so the validation
(``brainpy_state/_nest_validation/spatial_gaussian_kernel_test.py``) asserts the empirical
``p(d)`` follows the Gaussian (NEST-free) and matches live NEST's empirical curve within a
distributional band (PRNG draws diverge sample-by-sample; the law does not).

Run:  PYTHONPATH=. python examples/nest_like/spatial_gaussex.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(shape=(30, 30), extent=(3.0, 3.0), std=0.5, radius=3.0, seed=0):
    """Connect two grid layers with a Gaussian kernel; return the central footprint.

    Parameters
    ----------
    shape : tuple of int, optional
        ``(n_columns, n_rows)`` of the shared grid. Default ``(30, 30)`` (NEST's value).
    extent : tuple of float, optional
        Physical extent (micrometres). Default ``(3.0, 3.0)``.
    std : float, optional
        Gaussian kernel standard deviation (micrometres). Default ``0.5``.
    radius : float, optional
        Circular-mask cutoff radius (micrometres). Default ``3.0``.
    seed : int, optional
        Connectivity-sampling seed. Default ``0``.

    Returns
    -------
    coords : brainunit.Quantity
        ``(n, 2)`` node coordinates of the (shared) grid.
    ctr : int
        Local index of the central source node.
    target_pos : brainunit.Quantity
        ``(k, 2)`` coordinates of the central node's realized targets.

    Examples
    --------
    .. code-block:: python

        >>> coords, ctr, tgt = run(shape=(10, 10))
        >>> coords.shape
        (100, 2)
        >>> bool(tgt.shape[0] >= 1)        # the centre connects to at least itself-at-distance-0
        True
    """
    sim = bp.Simulator(dt=0.1 * u.ms)
    pos = bp.spatial.grid(list(shape), extent=list(extent))
    a = sim.create(bp.iaf_psc_alpha, positions=pos)
    b = sim.create(bp.iaf_psc_alpha, positions=pos)
    sim.connect(
        a, b,
        rule=bp.spatial.spatial_pairwise_bernoulli(
            p=bp.spatial.gaussian(bp.spatial.distance, std=std),
            mask=bp.spatial.circular(radius)),
        weight=1.0 * u.pA, delay=1.0 * u.ms, seed=seed)
    ctr = bp.spatial.center_element(pos)
    target_pos = bp.spatial.target_positions(sim, a[ctr], b)[0]
    return sim.get_position(a), ctr, target_pos


def main():
    coords, ctr, target_pos = run()
    xy = np.asarray(u.get_magnitude(coords.to(u.um)))
    tgt = np.asarray(u.get_magnitude(target_pos.to(u.um)))
    print('spatial_gaussex (brainpy.state)')
    print(f'  {xy.shape[0]} nodes; centre node {ctr} has {tgt.shape[0]} realized targets')
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not installed; skipping plot)')
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(xy[:, 0], xy[:, 1], s=8, color='0.8', label='all nodes')
    plt.scatter(tgt[:, 0], tgt[:, 1], s=20, color='green', label='targets')
    plt.scatter([xy[ctr, 0]], [xy[ctr, 1]], s=80, color='blue', label='source')
    plt.gca().set_aspect('equal', 'box')
    plt.axis([-2.0, 2.0, -2.0, 2.0])
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('Connection targets, Gaussian kernel (brainpy.state)')
    plt.tight_layout()
    plt.savefig('examples/nest_like/spatial_gaussex.png', dpi=100)
    print('  wrote examples/nest_like/spatial_gaussex.png')


if __name__ == '__main__':
    main()
