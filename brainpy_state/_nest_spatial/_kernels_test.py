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


class TestExpressions(unittest.TestCase):
    """The axis/scalar expression family evaluated on bound (pre, post) positions."""

    def setUp(self):
        # disp[i, j] = post[j] - pre[i]
        self.pre = jnp.array([[0.0, 0.0], [1.0, 2.0]]) * u.um
        self.post = jnp.array([[3.0, 0.0], [0.0, 5.0]]) * u.um

    def test_scalar_distance_eval_pair(self):
        g = distance._eval_pair(self.pre, self.post)
        self.assertEqual(g.shape, (2, 2))
        gm = np.asarray(u.get_magnitude(g.to(u.um)))
        self.assertAlmostEqual(gm[0, 0], 3.0, places=6)          # (0,0)->(3,0)
        self.assertAlmostEqual(gm[0, 1], 5.0, places=6)          # (0,0)->(0,5)
        self.assertAlmostEqual(gm[1, 0], np.sqrt(8.0), places=6) # (1,2)->(3,0)

    def test_axis_distance_is_abs_per_axis(self):
        gx = np.asarray(u.get_magnitude(distance.x._eval_pair(self.pre, self.post).to(u.um)))
        gy = np.asarray(u.get_magnitude(distance.y._eval_pair(self.pre, self.post).to(u.um)))
        # |post_x - pre_x|: disp[1,0] = (2,-2) -> |x|=2 ; disp[0,0]=(3,0)->|x|=3
        self.assertAlmostEqual(gx[1, 0], 2.0, places=6)
        self.assertAlmostEqual(gx[0, 0], 3.0, places=6)
        # |post_y - pre_y|: disp[0,1] = (0,5) -> |y|=5
        self.assertAlmostEqual(gy[0, 1], 5.0, places=6)

    def test_axis_z_on_2d_layer_raises(self):
        with self.assertRaises(ValueError):
            distance.z._eval_pair(self.pre, self.post)

    def test_gaussian_consumes_axis_distance(self):
        # gaussian(distance.x, std=1) == exp(-|dx|^2 / 2)
        k = gaussian(distance.x, std=1.0)
        g = np.asarray(u.get_magnitude(k._eval_pair(self.pre, self.post)))
        dx = np.abs(np.asarray(u.get_magnitude(self.post.to(u.um)))[None, :, 0]
                    - np.asarray(u.get_magnitude(self.pre.to(u.um)))[:, None, 0])
        np.testing.assert_allclose(g, np.exp(-(dx ** 2) / 2.0), atol=1e-6)

    def test_source_pos_broadcasts_source_coord(self):
        from brainpy_state._nest_spatial._kernels import source_pos
        g = np.asarray(u.get_magnitude(source_pos.x._eval_pair(self.pre, self.post).to(u.um)))
        np.testing.assert_allclose(g, [[0.0, 0.0], [1.0, 1.0]], atol=1e-6)   # rows = pre_x

    def test_target_pos_broadcasts_target_coord(self):
        from brainpy_state._nest_spatial._kernels import target_pos
        g = np.asarray(u.get_magnitude(target_pos.y._eval_pair(self.pre, self.post).to(u.um)))
        np.testing.assert_allclose(g, [[0.0, 5.0], [0.0, 5.0]], atol=1e-6)   # cols = post_y

    def test_pos_in_connect_path_raises(self):
        from brainpy_state._nest_spatial._kernels import pos
        with self.assertRaises(ValueError):
            pos.x._eval_pair(self.pre, self.post)

    def test_pos_eval_nodes_returns_axis_column(self):
        from brainpy_state._nest_spatial._kernels import pos
        coords = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) * u.um
        gx = np.asarray(u.get_magnitude(pos.x._eval_nodes(coords).to(u.um)))
        gy = np.asarray(u.get_magnitude(pos.y._eval_nodes(coords).to(u.um)))
        np.testing.assert_allclose(gx, [1.0, 3.0, 5.0], atol=1e-6)
        np.testing.assert_allclose(gy, [2.0, 4.0, 6.0], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
