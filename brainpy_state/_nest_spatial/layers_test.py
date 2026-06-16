# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for position layers (grid / free)."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._dist import Uniform
from brainpy_state._nest_spatial.layers import Layer, grid, free


class TestGrid(unittest.TestCase):
    def test_grid_4x3_coords_match_nest(self):
        # Pinned against live NEST 3.9.0: grid([4,3], extent=[2,1.5]).
        lay = grid([4, 3], extent=[2.0, 1.5])
        self.assertEqual(lay.n, 12)
        self.assertEqual(lay.ndim, 2)
        xy = u.get_magnitude(lay.coords.to(u.um))            # (12,2) mantissa in um
        np.testing.assert_allclose(xy[0], [-0.75, 0.5], atol=1e-12)   # node 0 = top-left
        np.testing.assert_allclose(xy[1], [-0.75, 0.0], atol=1e-12)   # y (row) varies fastest
        np.testing.assert_allclose(xy[2], [-0.75, -0.5], atol=1e-12)
        np.testing.assert_allclose(xy[3], [-0.25, 0.5], atol=1e-12)   # next column (x slowest)
        np.testing.assert_allclose(xy[11], [0.75, -0.5], atol=1e-12)

    def test_grid_default_extent_is_unit_box(self):
        xy = u.get_magnitude(grid([2, 2]).coords.to(u.um))
        np.testing.assert_allclose(sorted(xy[:, 0].tolist()), [-0.25, -0.25, 0.25, 0.25])
        np.testing.assert_allclose(sorted(xy[:, 1].tolist()), [-0.25, -0.25, 0.25, 0.25])

    def test_grid_degenerate_1xN(self):
        xy = u.get_magnitude(grid([1, 4]).coords.to(u.um))
        np.testing.assert_allclose(xy[:, 0], [0.0, 0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(xy[:, 1], [0.375, 0.125, -0.125, -0.375], atol=1e-12)

    def test_grid_3d_count_and_ndim(self):
        lay = grid([2, 2, 2], extent=[1.0, 1.0, 1.0])
        self.assertEqual((lay.n, lay.ndim), (8, 3))
        self.assertEqual(u.get_magnitude(lay.coords.to(u.um)).shape, (8, 3))

    def test_grid_quantity_extent(self):
        lay = grid([4, 3], extent=jnp.array([2.0, 1.5]) * u.um)
        self.assertEqual(lay.n, 12)
        xy = u.get_magnitude(lay.coords.to(u.um))
        np.testing.assert_allclose(xy[0], [-0.75, 0.5], atol=1e-12)

    def test_grid_center_offset(self):
        lay = grid([2, 2], extent=[1.0, 1.0], center=[10.0, 20.0])
        xy = u.get_magnitude(lay.coords.to(u.um))
        np.testing.assert_allclose(xy[0], [9.75, 20.25], atol=1e-12)

    def test_grid_bad_ndim_raises(self):
        with self.assertRaises(ValueError):
            grid([4])           # 1-D not allowed
        with self.assertRaises(ValueError):
            grid([2, 2, 2, 2])  # 4-D not allowed


class TestFree(unittest.TestCase):
    def test_free_explicit_array_2d(self):
        pos = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]]) * u.um
        lay = free(pos)
        self.assertEqual((lay.n, lay.ndim), (3, 2))
        self.assertFalse(lay.is_deferred)

    def test_free_deferred_from_distribution_3d(self):
        lay = free(Uniform(-0.5, 0.5), extent=[1.5, 1.5, 1.5])
        self.assertTrue(lay.is_deferred)
        self.assertEqual(lay.ndim, 3)
        coords = lay.sample(1000, jax.random.key(0))
        m = u.get_magnitude(coords.to(u.um))
        self.assertEqual(m.shape, (1000, 3))
        self.assertTrue(m.min() >= -0.5 - 1e-6 and m.max() <= 0.5 + 1e-6)

    def test_free_num_dimensions_2d(self):
        lay = free(Uniform(-1.0, 1.0), num_dimensions=2)
        self.assertEqual(lay.ndim, 2)
        self.assertTrue(lay.is_deferred)

    def test_free_extent_and_numdim_conflict_raises(self):
        with self.assertRaises(TypeError):
            free(Uniform(-1.0, 1.0), extent=[1.0, 1.0], num_dimensions=2)

    def test_free_bad_array_shape_raises(self):
        with self.assertRaises(ValueError):
            free(jnp.array([1.0, 2.0, 3.0]) * u.um)   # not (n, d)

    def test_deferred_layer_n_raises(self):
        lay = free(Uniform(-1.0, 1.0), num_dimensions=2)
        with self.assertRaises(ValueError):
            _ = lay.n

    def test_sample_on_concrete_layer_returns_stored_coords(self):
        # a concrete layer ignores (n, key) and returns its stored coordinates
        g = grid([2, 2], extent=[1.0, 1.0])
        out = g.sample(99, jax.random.key(0))
        np.testing.assert_array_equal(u.get_magnitude(out), u.get_magnitude(g.coords))

    def test_free_distribution_rejects_non_2d_3d(self):
        with self.assertRaises(ValueError):
            free(Uniform(-0.5, 0.5), num_dimensions=4)


if __name__ == '__main__':
    unittest.main()
