# examples/nest_like/spatial_3d_gauss.py
"""Spatial networks in 3D with a Gaussian kernel -- NEST ``spatial/test_3d_gauss.py`` port.

Places 1000 ``iaf_psc_alpha`` neurons at uniformly-random positions in a 3D box (a
``spatial.free`` layer sampled from ``Uniform(-0.5, 0.5)`` with extent ``[1.5, 1.5, 1.5]``)
and connects the layer to itself with a Gaussian distance kernel
:math:`p(d)=\\exp(-d^2/(2\\,\\mathrm{std}^2))` (``std=0.25``), no autapses, clipped to a cubic
box mask ``[-0.75, 0.75]^3`` anchored on the source. The realized footprint of the central
node is read back with :func:`~brainpy.state.spatial.target_positions` and its target
distances are histogrammed (NEST's figure).

Random layer positions diverge sample-by-sample from NEST (independent PRNGs), so the
validation (``brainpy_state/_nest/_validation/spatial_3d_test.py``) checks the
position-independent **law**: the box-mask cutoff is hard, autapses are absent, and the
empirical ``p(d)`` and edge count match live NEST distributionally.

Run:  PYTHONPATH=. python examples/nest_like/spatial_3d_gauss.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(n=1000, extent=(1.5, 1.5, 1.5), std=0.25, box=0.75, seed=0):
    """Build a 3D free layer, connect with a Gaussian kernel, return the central footprint.

    Parameters
    ----------
    n : int, optional
        Number of neurons. Default ``1000`` (NEST's value).
    extent : tuple of float, optional
        3D bounding-box extent (micrometres). Default ``(1.5, 1.5, 1.5)``.
    std : float, optional
        Gaussian kernel standard deviation (micrometres). Default ``0.25``.
    box : float, optional
        Half-width of the cubic connection mask (micrometres). Default ``0.75``.
    seed : int, optional
        Connectivity-sampling seed. Default ``0``.

    Returns
    -------
    coords : brainunit.Quantity
        ``(n, 3)`` node coordinates.
    ctr : int
        Local index of the central node.
    target_pos : brainunit.Quantity
        ``(k, 3)`` coordinates of the central node's realized targets.
    target_dist : brainunit.Quantity
        ``(k,)`` distances from the central node to its targets.

    Examples
    --------
    .. code-block:: python

        >>> coords, ctr, tgt, dist = run(n=300)
        >>> coords.shape
        (300, 3)
        >>> bool(dist.shape[0] == tgt.shape[0])
        True
    """
    sim = bp.Simulator(dt=0.1 * u.ms)
    pos = bp.spatial.free(bp.dist.Uniform(-0.5, 0.5), extent=list(extent))
    l1 = sim.create(bp.iaf_psc_alpha, n, positions=pos)
    sim.connect(
        l1, l1,
        rule=bp.spatial.spatial_pairwise_bernoulli(
            p=bp.spatial.gaussian(bp.spatial.distance, std=std),
            mask=bp.spatial.box(lower_left=[-box, -box, -box],
                                upper_right=[box, box, box])),
        weight=1.0 * u.pA, delay=1.0 * u.ms, allow_autapses=False, seed=seed)
    coords = sim.get_position(l1)
    # the free layer is sampled from a distribution, so resolve the central node on the
    # realized coordinates (rebuild a concrete layer -> reuse center_element / FindCenterElement)
    ctr = bp.spatial.center_element(bp.spatial.free(coords))
    target_pos = bp.spatial.target_positions(sim, l1[ctr], l1)[0]
    target_dist = bp.spatial.pairwise_distance(coords[ctr][None], target_pos)[0]
    return coords, ctr, target_pos, target_dist


def main():
    coords, ctr, target_pos, target_dist = run()
    xyz = np.asarray(u.get_magnitude(coords.to(u.um)))
    dist = np.asarray(u.get_magnitude(target_dist.to(u.um)))
    print('spatial_3d_gauss (brainpy.state)')
    print(f'  {xyz.shape[0]} nodes in 3D; centre node {ctr} has {dist.shape[0]} targets')
    print(f'  target distance: min {dist.min():.3f}, max {dist.max():.3f}, mean {dist.mean():.3f}')
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print('  (matplotlib not installed; skipping plot)')
        return
    tgt = np.asarray(u.get_magnitude(target_pos.to(u.um)))
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=6, color='b')
    ax.scatter([xyz[ctr, 0]], [xyz[ctr, 1]], [xyz[ctr, 2]], s=50, color='r')
    ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], s=30, color='g')
    ax.set_title('3D layer + centre footprint')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(dist, 25)
    ax2.set_xlabel('distance from centre')
    ax2.set_ylabel('target count')
    ax2.set_title('Target-distance histogram')
    plt.tight_layout()
    plt.savefig('examples/nest_like/spatial_3d_gauss.png', dpi=100)
    print('  wrote examples/nest_like/spatial_3d_gauss.png')


if __name__ == '__main__':
    main()
