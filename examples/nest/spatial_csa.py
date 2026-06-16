# examples/nest/spatial_csa.py
"""Spatial connectivity without CSA -- NEST ``csa_spatial_example.py`` ported natively.

NEST's ``csa_spatial_example`` wires two 20x20 grids through the **Connection Set Algebra**
(``csa.random * (csa.gaussian(0.2, 0.5) * d)``), which requires NEST to be compiled against
``libneurosim``. The connectivity that expression *describes* is an ordinary
distance-dependent pairwise-Bernoulli draw: a Gaussian of the source->target distance with
``sigma = 0.2`` cut off at ``0.5``. brainpy.state expresses exactly that with the built-in
spatial rule -- no CSA, no external dependency::

    spatial.spatial_pairwise_bernoulli(
        p=spatial.gaussian(spatial.distance, std=0.2),
        mask=spatial.circular(0.5))

The CSA ``gaussian(sigma, cutoff)`` profile maps to ``gaussian(distance, std=sigma)`` (same
:math:`\\exp(-d^2/2\\sigma^2)` shape) with the hard ``cutoff`` realised by a
``circular(cutoff)`` mask, and the leading ``csa.random *`` is the Bernoulli draw itself.
The realized footprint of the central source neuron is read back with
:func:`~brainpy.state.spatial.target_positions` (NEST's ``PlotTargets``).

Run:  PYTHONPATH=. python examples/nest/spatial_csa.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(shape=(20, 20), std=0.2, cutoff=0.5, seed=0):
    """Connect two grids with the native Gaussian rule (CSA-free); return the central footprint.

    Parameters
    ----------
    shape : tuple of int, optional
        ``(n_columns, n_rows)`` of both grids (unit-square extent). Default ``(20, 20)``.
    std : float, optional
        Gaussian ``sigma`` of the connection profile. Default ``0.2`` (NEST's value).
    cutoff : float, optional
        Hard radial cutoff -> circular mask radius. Default ``0.5`` (NEST's value).
    seed : int, optional
        Connectivity-sampling seed. Default ``0``.

    Returns
    -------
    coords : brainunit.Quantity
        ``(n, 2)`` node coordinates of the source grid.
    ctr : int
        Local index of the central source node.
    target_pos : brainunit.Quantity
        ``(k, 2)`` coordinates of the central node's realized targets in ``pop2``.

    Examples
    --------
    .. code-block:: python

        >>> coords, ctr, tgt = run(shape=(8, 8))
        >>> coords.shape
        (64, 2)
        >>> bool(tgt.shape[0] >= 1)
        True
    """
    sim = bp.Simulator(dt=0.1 * u.ms)
    pos = bp.spatial.grid(list(shape))                       # default unit-square extent
    pop1 = sim.create(bp.iaf_psc_alpha, positions=pos)
    pop2 = sim.create(bp.iaf_psc_alpha, positions=pos)
    sim.connect(
        pop1, pop2,
        rule=bp.spatial.spatial_pairwise_bernoulli(
            p=bp.spatial.gaussian(bp.spatial.distance, std=std),
            mask=bp.spatial.circular(cutoff)),
        weight=10000.0 * u.pA, delay=1.0 * u.ms, seed=seed)
    ctr = bp.spatial.center_element(pos)
    target_pos = bp.spatial.target_positions(sim, pop1[ctr], pop2)[0]
    return sim.get_position(pop1), ctr, target_pos


def main():
    coords, ctr, target_pos = run()
    xy = np.asarray(u.get_magnitude(coords.to(u.um)))
    tgt = np.asarray(u.get_magnitude(target_pos.to(u.um)))
    print('spatial_csa (brainpy.state, CSA-free)')
    print(f'  two 20x20 grids; centre node {ctr} has {tgt.shape[0]} realized targets')
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not installed; skipping plot)')
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(xy[:, 0], xy[:, 1], s=8, color='0.8', label='all nodes')
    plt.scatter(tgt[:, 0], tgt[:, 1], s=20, color='green', label='targets')
    plt.scatter([xy[ctr, 0]], [xy[ctr, 1]], s=80, color='red', label='source')
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('Targets of centre neuron, Gaussian profile (brainpy.state)')
    plt.tight_layout()
    plt.savefig('examples/nest/spatial_csa.png', dpi=100)
    print('  wrote examples/nest/spatial_csa.png')


if __name__ == '__main__':
    main()
