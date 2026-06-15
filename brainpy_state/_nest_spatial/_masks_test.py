# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for spatial masks (circular / spherical / box)."""
import unittest

import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._nest_spatial._masks import circular, spherical, box


class TestMasks(unittest.TestCase):
    def test_circular_cutoff_inclusive(self):
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.0, 0.0], [0.4, 0.0], [0.5, 0.0], [0.6, 0.0]]) * u.um
        m = np.asarray(circular(0.5).contains(pre, post))
        self.assertEqual(m.shape, (1, 4))
        np.testing.assert_array_equal(m[0], [True, True, True, False])

    def test_circular_radius_quantity(self):
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.49, 0.0]]) * u.um
        self.assertTrue(bool(circular(0.5 * u.um).contains(pre, post)[0, 0]))

    def test_box_anchored_on_source_3d(self):
        pre = jnp.array([[1.0, 1.0, 1.0]]) * u.um
        # displacement target-source: 0.5 (in) and 1.0 (out) along x.
        post = jnp.array([[1.5, 1.0, 1.0], [2.0, 1.0, 1.0]]) * u.um
        m = np.asarray(box([-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]).contains(pre, post))
        np.testing.assert_array_equal(m[0], [True, False])

    def test_box_2d(self):
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.2, 0.2], [0.2, -0.9]]) * u.um
        m = np.asarray(box([-0.5, -0.5], [0.5, 0.5]).contains(pre, post))
        np.testing.assert_array_equal(m[0], [True, False])

    def test_spherical_is_distance_cutoff_3d(self):
        pre = jnp.array([[0.0, 0.0, 0.0]]) * u.um
        post = jnp.array([[0.7, 0.0, 0.0], [0.8, 0.0, 0.0]]) * u.um
        m = np.asarray(spherical(0.75).contains(pre, post))
        np.testing.assert_array_equal(m[0], [True, False])


if __name__ == '__main__':
    unittest.main()
