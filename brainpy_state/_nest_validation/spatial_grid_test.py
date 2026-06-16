# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Exact structural parity for the spatial **grid** layer (goal 20, ``spatial/grid_iaf.py``).

Grid coordinates are deterministic, so the parity here is **exact** (not distributional):
node ``k`` sits at column ``k // n_rows``, row ``k % n_rows`` with

.. math::

    x_k = c_x - L_x/2 + (\mathrm{col} + 0.5)\,L_x/n_x, \qquad
    y_k = c_y + L_y/2 - (\mathrm{row} + 0.5)\,L_y/n_y .

Two classes (cluster-16 house style):

* ``TestSpatialGridStructure`` -- NEST-free, always runs: coordinates match the closed-form
  layout, the default extent is the unit square, the column-slow/row-fast ordering holds in
  2D and 3D, ``center_element`` lands on the central node, and the example runs standalone.
* ``TestSpatialGridNestParity`` (``@requires_nest``) -- the same layers vs live NEST
  (``nest.spatial.grid`` + ``nest.GetPosition`` / ``nest.FindCenterElement``): coordinates and
  the centre element match element-for-element.
"""
import unittest

import jax
import brainstate
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import brainunit as u

try:
    import nest
except Exception:
    nest = None

from brainpy_state import Simulator, iaf_psc_alpha
from brainpy_state._nest_spatial import grid, center_element
from brainpy_state._nest_validation.nest_compare import requires_nest


def _coords(shape, extent):
    """brainpy.state grid coordinates (micrometre magnitudes), in node order."""
    sim = Simulator(dt=0.1 * u.ms)
    pop = sim.create(iaf_psc_alpha, positions=grid(list(shape), extent=list(extent)))
    return np.asarray(u.get_magnitude(sim.get_position(pop).to(u.um)))


class TestSpatialGridStructure(unittest.TestCase):
    """The grid layer reproduces NEST's coordinate convention exactly (NEST-free)."""

    def test_4x3_coords_match_closed_form(self):
        xy = _coords((4, 3), (2.0, 1.5))
        self.assertEqual(xy.shape, (12, 2))
        expect = [[-1.0 + (col + 0.5) * 0.5, 0.75 - (row + 0.5) * 0.5]
                  for col in range(4) for row in range(3)]
        np.testing.assert_allclose(xy, expect, atol=1e-12)

    def test_default_extent_is_unit_square(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_alpha, positions=grid([2, 2]))   # default extent [1, 1]
        xy = np.asarray(u.get_magnitude(sim.get_position(pop).to(u.um)))
        np.testing.assert_allclose(np.sort(np.unique(xy[:, 0])), [-0.25, 0.25], atol=1e-12)
        np.testing.assert_allclose(np.sort(np.unique(xy[:, 1])), [-0.25, 0.25], atol=1e-12)

    def test_column_slow_row_fast_ordering(self):
        xy = _coords((4, 3), (2.0, 1.5))
        np.testing.assert_allclose(xy[:3, 0], -0.75, atol=1e-12)   # first column shares x
        self.assertTrue(xy[0, 1] > xy[1, 1] > xy[2, 1])            # y decreases top->bottom

    def test_3d_grid_shape_and_center(self):
        xy = _coords((3, 3, 3), (1.0, 1.0, 1.0))
        self.assertEqual(xy.shape, (27, 3))
        self.assertEqual(center_element(grid([3, 3, 3], extent=[1.0, 1.0, 1.0])), 13)

    def test_center_element_4x3(self):
        self.assertEqual(center_element(grid([4, 3], extent=[2.0, 1.5])), 4)

    def test_example_run_smoke(self):
        from examples.nest_like.spatial_grid_iaf import run
        coords = run()
        self.assertEqual(coords.shape, (12, 2))
        np.testing.assert_allclose(
            np.asarray(u.get_magnitude(coords.to(u.um)))[0], [-0.75, 0.5], atol=1e-12)


@requires_nest
class TestSpatialGridNestParity(unittest.TestCase):
    """Grid coordinates and centre element match live NEST element-for-element."""

    def test_grid_coords_match_nest_4x3(self):
        nest.ResetKernel()
        l1 = nest.Create('iaf_psc_alpha', positions=nest.spatial.grid(shape=[4, 3], extent=[2.0, 1.5]))
        np.testing.assert_allclose(_coords((4, 3), (2.0, 1.5)), np.array(nest.GetPosition(l1)), atol=1e-9)

    def test_grid_coords_match_nest_3d(self):
        nest.ResetKernel()
        l1 = nest.Create('iaf_psc_alpha',
                         positions=nest.spatial.grid(shape=[3, 3, 3], extent=[1.0, 1.0, 1.0]))
        np.testing.assert_allclose(_coords((3, 3, 3), (1.0, 1.0, 1.0)),
                                   np.array(nest.GetPosition(l1)), atol=1e-9)

    def test_center_element_matches_nest(self):
        nest.ResetKernel()
        l1 = nest.Create('iaf_psc_alpha', positions=nest.spatial.grid(shape=[4, 3], extent=[2.0, 1.5]))
        ctr = nest.FindCenterElement(l1)
        local = int(ctr.global_id - l1[0].global_id)
        self.assertEqual(local, center_element(grid([4, 3], extent=[2.0, 1.5])))


if __name__ == '__main__':
    unittest.main()
