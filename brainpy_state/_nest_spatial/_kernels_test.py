# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for distance kernels (gaussian) + the distance sentinel."""
import unittest

import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._nest_spatial._kernels import distance, gaussian


class TestGaussianKernel(unittest.TestCase):
    def test_peak_one_at_zero(self):
        k = gaussian(distance, std=0.5)
        self.assertAlmostEqual(float(k(0.0 * u.um)), 1.0, places=12)

    def test_value_matches_exp_formula(self):
        # p(d) = exp(-d^2 / (2 std^2)); std=0.5 -> exp(-2 d^2).
        k = gaussian(distance, std=0.5)
        d = jnp.array([0.5, 1.0]) * u.um
        got = np.asarray(u.get_magnitude(k(d)))
        np.testing.assert_allclose(got, np.exp(-2.0 * np.array([0.25, 1.0])), atol=1e-6)

    def test_std_as_quantity(self):
        k = gaussian(distance, std=0.5 * u.um)
        self.assertAlmostEqual(float(u.get_magnitude(k(0.5 * u.um))), float(np.exp(-0.5)), places=6)

    def test_matrix_input(self):
        k = gaussian(distance, std=1.0)
        d = jnp.array([[0.0, 1.0], [2.0, 3.0]]) * u.um
        got = np.asarray(u.get_magnitude(k(d)))
        self.assertEqual(got.shape, (2, 2))
        self.assertAlmostEqual(got[0, 0], 1.0, places=6)

    def test_gaussian_rejects_non_sentinel(self):
        with self.assertRaises(ValueError):
            gaussian(object(), std=0.5)

    def test_distance_sentinel_repr(self):
        self.assertIn('distance', repr(distance))


if __name__ == '__main__':
    unittest.main()
