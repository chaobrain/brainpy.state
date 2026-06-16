# examples/nest_like/spatial_grid_iaf.py
"""Spatial networks: a 4x3 grid of iaf_psc_alpha -- NEST ``spatial/grid_iaf.py`` port.

Creates a population of ``iaf_psc_alpha`` neurons on a regular 4-column x 3-row grid of
extent ``[2.0, 1.5]`` (NEST default length units, taken here as micrometres) through
:meth:`~brainpy.state.Simulator.create` with ``positions=spatial.grid(...)``, then reads
the node coordinates back with :meth:`~brainpy.state.Simulator.get_position` (NEST
``GetPosition``).

The grid layout is *exactly* NEST's: column index is the slow axis and row index the fast
axis, ``x`` increases left->right and ``y`` decreases top->bottom, with node ``k`` at
column ``k // n_rows``, row ``k % n_rows`` and

.. math::

    x_k = c_x - L_x/2 + (\\mathrm{col} + 0.5)\\,L_x / n_x, \\qquad
    y_k = c_y + L_y/2 - (\\mathrm{row} + 0.5)\\,L_y / n_y .

The coordinates are asserted to match live NEST element-for-element in
``brainpy_state/_nest/_validation/spatial_grid_test.py``.

Run:  PYTHONPATH=. python examples/nest_like/spatial_grid_iaf.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(shape=(4, 3), extent=(2.0, 1.5)):
    """Build a grid layer of ``iaf_psc_alpha`` and return its node coordinates.

    Parameters
    ----------
    shape : tuple of int, optional
        ``(n_columns, n_rows)`` of the grid. Default ``(4, 3)`` (NEST's tutorial value).
    extent : tuple of float, optional
        Physical ``(L_x, L_y)`` extent (micrometres). Default ``(2.0, 1.5)``.

    Returns
    -------
    coords : brainunit.Quantity
        ``(n_columns * n_rows, 2)`` node coordinates (micrometres), in node-index order.

    Examples
    --------
    .. code-block:: python

        >>> coords = run()
        >>> coords.shape
        (12, 2)
        >>> import brainunit as u
        >>> [round(float(v), 2) for v in u.get_magnitude(coords.to(u.um))[0]]
        [-0.75, 0.5]
    """
    sim = bp.Simulator(dt=0.1 * u.ms)
    pop = sim.create(bp.iaf_psc_alpha,
                     positions=bp.spatial.grid(list(shape), extent=list(extent)))
    return sim.get_position(pop)


def main():
    coords = run()
    xy = np.asarray(u.get_magnitude(coords.to(u.um)))
    print('spatial_grid_iaf (brainpy.state)')
    print(f'  {xy.shape[0]} nodes on a 4x3 grid, extent [2.0, 1.5]')
    for k, (x, y) in enumerate(xy):
        print(f'  node {k:2d}: ({x:+.3f}, {y:+.3f})')
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not installed; skipping plot)')
        return
    plt.figure(figsize=(5, 4))
    plt.scatter(xy[:, 0], xy[:, 1], s=50)
    plt.axis([-1.0, 1.0, -0.75, 0.75])
    plt.gca().set_aspect('equal', 'box')
    plt.gca().set_xticks((-0.75, -0.25, 0.25, 0.75))
    plt.gca().set_yticks((-0.5, 0, 0.5))
    plt.grid(True)
    plt.xlabel('4 Columns, Extent: 1.5')
    plt.ylabel('3 Rows, Extent: 1.0')
    plt.title('Spatial grid of iaf_psc_alpha (brainpy.state)')
    plt.tight_layout()
    plt.savefig('examples/nest_like/spatial_grid_iaf.png', dpi=100)
    print('  wrote examples/nest_like/spatial_grid_iaf.png')


if __name__ == '__main__':
    main()
