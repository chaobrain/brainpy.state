# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for pairwise displacement / Euclidean distance."""
import unittest

import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._nest_spatial._distance import displacement, pairwise_distance


class TestDistance(unittest.TestCase):
    def test_pairwise_distance_2d(self):
        a = jnp.array([[0.0, 0.0], [3.0, 0.0]]) * u.um
        b = jnp.array([[0.0, 0.0], [0.0, 4.0]]) * u.um
        d = u.get_magnitude(pairwise_distance(a, b).to(u.um))
        self.assertEqual(d.shape, (2, 2))
        np.testing.assert_allclose(d, [[0.0, 4.0], [3.0, 5.0]], atol=1e-6)

    def test_displacement_sign_is_post_minus_pre(self):
        a = jnp.array([[1.0, 1.0]]) * u.um
        b = jnp.array([[4.0, 5.0]]) * u.um
        disp = u.get_magnitude(displacement(a, b).to(u.um))
        self.assertEqual(disp.shape, (1, 1, 2))
        np.testing.assert_allclose(disp[0, 0], [3.0, 4.0], atol=1e-6)

    def test_self_distance_zero_3d(self):
        a = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]]) * u.um
        d = u.get_magnitude(pairwise_distance(a, a).to(u.um))
        np.testing.assert_allclose(np.diag(d), [0.0, 0.0], atol=1e-6)

    def test_distance_symmetric_for_same_layer(self):
        a = jnp.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 1.0]]) * u.um
        d = u.get_magnitude(pairwise_distance(a, a).to(u.um))
        np.testing.assert_allclose(d, d.T, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
