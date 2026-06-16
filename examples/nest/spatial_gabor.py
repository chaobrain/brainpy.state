# examples/nest/spatial_gabor.py
r"""Spatial networks: an anisotropic Gabor kernel inside a tilted elliptical mask.

Demonstrates the cluster-27 additions to ``brainpy.state.spatial``: the per-axis distance
expressions :data:`distance.x` / :data:`distance.y`, the anisotropic :func:`gabor` distance
distribution, and the rotated :func:`elliptical` mask. Two ``iaf_psc_alpha`` populations share a
grid and are connected with a distance-dependent pairwise-Bernoulli rule whose probability is a
Gabor function of the source->target displacement,

.. math::

    p(x, y) = \max\!\big(\cos(2\pi y' / \lambda + \psi),\, 0\big)\,
              \exp\!\Big(-\frac{\gamma^2 x'^2 + y'^2}{2\,\mathrm{std}^2}\Big),

with :math:`(x', y')` the displacement rotated by ``theta`` and :math:`x = |dx|`, :math:`y = |dy|`
(NEST's per-axis ``distance.x`` / ``distance.y`` are absolute). The candidate set is clipped to an
ellipse tilted by the same angle, so the realized footprint is an oriented, striped blob. This is
just ``spatial.spatial_pairwise_bernoulli(p=spatial.gabor(...), mask=spatial.elliptical(...))``
riding the ordinary :meth:`~brainpy.state.Simulator.connect`. The central source neuron's realized
targets are read back with :func:`~brainpy.state.spatial.target_positions` (NEST
``GetTargetPositions``).

The per-displacement probability is a fixed law, so the validation
(``brainpy_state/_nest/_validation/spatial_gabor_test.py``) asserts the kernel matches live NEST's
``spatial_distributions.gabor`` element-by-element and that the realized footprint stays inside the
ellipse.

Run:  PYTHONPATH=. python examples/nest/spatial_gabor.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(shape=(30, 30), extent=(4.0, 4.0), theta=45.0, gamma=1.0, std=0.6,
        lam=1.0, psi=0.0, major=3.0, minor=1.0, seed=0):
    """Connect two grid layers with a Gabor kernel inside a tilted ellipse.

    Parameters
    ----------
    shape : tuple of int, optional
        ``(n_columns, n_rows)`` of the shared grid. Default ``(30, 30)``.
    extent : tuple of float, optional
        Physical extent (micrometres). Default ``(4.0, 4.0)``.
    theta : float, optional
        Orientation of both the Gabor carrier and the elliptical mask, degrees. Default ``45``.
    gamma : float, optional
        Spatial aspect ratio of the Gabor envelope. Default ``1.0``.
    std : float, optional
        Gabor envelope standard deviation (micrometres). Default ``0.6``.
    lam : float, optional
        Carrier wavelength (micrometres). Default ``1.0``.
    psi : float, optional
        Carrier phase offset, degrees. Default ``0``.
    major, minor : float, optional
        Full axis lengths of the elliptical mask (micrometres). Default ``3.0`` / ``1.0``.
    seed : int, optional
        Connectivity-sampling seed. Default ``0``.

    Returns
    -------
    coords : brainunit.Quantity
        ``(n, 2)`` node coordinates of the shared grid.
    ctr : int
        Local index of the central source node.
    target_pos : brainunit.Quantity
        ``(k, 2)`` coordinates of the central node's realized targets.

    Examples
    --------
    .. code-block:: python

        >>> coords, ctr, tgt = run(shape=(12, 12))
        >>> coords.shape
        (144, 2)
        >>> bool(tgt.shape[0] >= 1)
        True
    """
    sim = bp.Simulator(dt=0.1 * u.ms)
    pos = bp.spatial.grid(list(shape), extent=list(extent))
    a = sim.create(bp.iaf_psc_alpha, positions=pos)
    b = sim.create(bp.iaf_psc_alpha, positions=pos)
    sim.connect(
        a, b,
        rule=bp.spatial.spatial_pairwise_bernoulli(
            p=bp.spatial.gabor(bp.spatial.distance.x, bp.spatial.distance.y,
                               theta=theta, gamma=gamma, std=std, lam=lam, psi=psi),
            mask=bp.spatial.elliptical(major, minor, azimuth_angle=theta)),
        weight=1.0 * u.pA, delay=1.0 * u.ms, seed=seed)
    ctr = bp.spatial.center_element(pos)
    target_pos = bp.spatial.target_positions(sim, a[ctr], b)[0]
    return sim.get_position(a), ctr, target_pos


def main():
    coords, ctr, target_pos = run()
    xy = np.asarray(u.get_magnitude(coords.to(u.um)))
    tgt = np.asarray(u.get_magnitude(target_pos.to(u.um)))
    print('spatial_gabor (brainpy.state)')
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
    plt.axis([-2.5, 2.5, -2.5, 2.5])
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('Gabor kernel in a tilted ellipse (brainpy.state)')
    plt.tight_layout()
    plt.savefig('examples/nest/spatial_gabor.png', dpi=100)
    print('  wrote examples/nest/spatial_gabor.png')


if __name__ == '__main__':
    main()
