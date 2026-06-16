# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for distance kernels (gaussian) + the distance sentinel."""
import math
import unittest

import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._nest_spatial.kernels import distance, gaussian


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
        from brainpy_state._nest_spatial.kernels import source_pos
        g = np.asarray(u.get_magnitude(source_pos.x._eval_pair(self.pre, self.post).to(u.um)))
        np.testing.assert_allclose(g, [[0.0, 0.0], [1.0, 1.0]], atol=1e-6)   # rows = pre_x

    def test_target_pos_broadcasts_target_coord(self):
        from brainpy_state._nest_spatial.kernels import target_pos
        g = np.asarray(u.get_magnitude(target_pos.y._eval_pair(self.pre, self.post).to(u.um)))
        np.testing.assert_allclose(g, [[0.0, 5.0], [0.0, 5.0]], atol=1e-6)   # cols = post_y

    def test_pos_in_connect_path_raises(self):
        from brainpy_state._nest_spatial.kernels import pos
        with self.assertRaises(ValueError):
            pos.x._eval_pair(self.pre, self.post)

    def test_pos_eval_nodes_returns_axis_column(self):
        from brainpy_state._nest_spatial.kernels import pos
        coords = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) * u.um
        gx = np.asarray(u.get_magnitude(pos.x._eval_nodes(coords).to(u.um)))
        gy = np.asarray(u.get_magnitude(pos.y._eval_nodes(coords).to(u.um)))
        np.testing.assert_allclose(gx, [1.0, 3.0, 5.0], atol=1e-6)
        np.testing.assert_allclose(gy, [2.0, 4.0, 6.0], atol=1e-6)


class TestExpressionEdges(unittest.TestCase):
    """The per-axis guards, reprs and gamma's expression path (edge coverage)."""

    def setUp(self):
        self.pre = jnp.array([[0.0, 0.0], [1.0, 2.0]]) * u.um     # 2-D layer
        self.post = jnp.array([[3.0, 0.0], [0.0, 5.0]]) * u.um
        self.coords2d = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * u.um

    def test_source_pos_z_on_2d_raises(self):
        from brainpy_state._nest_spatial.kernels import source_pos
        with self.assertRaises(ValueError):
            source_pos.z._eval_pair(self.pre, self.post)

    def test_target_pos_z_on_2d_raises(self):
        from brainpy_state._nest_spatial.kernels import target_pos
        with self.assertRaises(ValueError):
            target_pos.z._eval_pair(self.pre, self.post)

    def test_pos_z_eval_nodes_on_2d_raises(self):
        from brainpy_state._nest_spatial.kernels import pos
        with self.assertRaises(ValueError):
            pos.z._eval_nodes(self.coords2d)

    def test_reprs_name_the_axis(self):
        from brainpy_state._nest_spatial.kernels import pos, source_pos, target_pos
        self.assertEqual(repr(distance.x), 'spatial.distance.x')
        self.assertEqual(repr(source_pos.y), 'spatial.source_pos.y')
        self.assertEqual(repr(target_pos.z), 'spatial.target_pos.z')
        self.assertEqual(repr(pos.x), 'spatial.pos.x')
        self.assertEqual(repr(pos), 'spatial.pos')

    def test_gamma_eval_pair_consumes_axis_expression(self):
        from brainpy_state._nest_spatial.kernels import gamma
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[3.0, 4.0]]) * u.um
        # gamma(distance.x, kappa=1, theta=2) on |dx|=3 == e^{-3/2}/2
        g = float(u.get_magnitude(gamma(distance.x, kappa=1.0, theta=2.0)._eval_pair(pre, post)[0, 0]))
        self.assertAlmostEqual(g, math.exp(-1.5) / 2.0, places=6)


class TestDistributions(unittest.TestCase):
    """exponential / gamma scalar-distance kernels vs the exact NEST formulas."""

    def test_exponential_value(self):
        from brainpy_state._nest_spatial.kernels import exponential
        k = exponential(distance, beta=2.0)
        d = jnp.array([0.0, 2.0, 4.0]) * u.um
        got = np.asarray(u.get_magnitude(k(d)))
        np.testing.assert_allclose(got, np.exp(-np.array([0.0, 1.0, 2.0])), atol=1e-6)

    def test_exponential_peak_one_at_zero(self):
        from brainpy_state._nest_spatial.kernels import exponential
        self.assertAlmostEqual(float(u.get_magnitude(exponential(distance, beta=0.7)(0.0 * u.um))),
                               1.0, places=10)

    def test_gamma_matches_pdf(self):
        from brainpy_state._nest_spatial.kernels import gamma
        kappa, theta = 2.0, 1.5
        k = gamma(distance, kappa=kappa, theta=theta)
        d = jnp.array([0.5, 1.0, 3.0]) * u.um
        got = np.asarray(u.get_magnitude(k(d)))
        ref = [x ** (kappa - 1) * math.exp(-x / theta) / (theta ** kappa * math.gamma(kappa))
               for x in (0.5, 1.0, 3.0)]
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_gamma_shape1_is_exponential(self):
        # kappa=1 -> gamma reduces to exponential/theta: x^0 e^{-x/θ}/(θ Γ(1)) = e^{-x/θ}/θ
        from brainpy_state._nest_spatial.kernels import gamma
        k = gamma(distance, kappa=1.0, theta=2.0)
        got = float(u.get_magnitude(k(2.0 * u.um)))
        self.assertAlmostEqual(got, math.exp(-1.0) / 2.0, places=6)

    def test_kernels_take_per_axis_input(self):
        # the distributions accept a per-axis expression, not just scalar distance
        from brainpy_state._nest_spatial.kernels import exponential
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[3.0, 4.0]]) * u.um
        # exponential(distance.x, beta=1) -> exp(-|dx|) = exp(-3)
        g = float(u.get_magnitude(exponential(distance.x, beta=1.0)._eval_pair(pre, post)[0, 0]))
        self.assertAlmostEqual(g, math.exp(-3.0), places=6)


class TestAnisotropicKernels(unittest.TestCase):
    """gabor / gaussian2D two-axis kernels vs the exact NEST formulas (X=|dx|, Y=|dy|)."""

    def test_gabor_matches_formula(self):
        from brainpy_state._nest_spatial.kernels import gabor
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[3.0, 0.0]]) * u.um
        k = gabor(distance.x, distance.y, theta=0.0, gamma=1.0, std=1.0, lam=2.0, psi=0.0)
        got = float(u.get_magnitude(k._eval_pair(pre, post)[0, 0]))
        X, Y = 3.0, 0.0
        ref = max(math.cos(2 * math.pi * Y / 2.0), 0.0) * math.exp(-(X ** 2 + Y ** 2) / 2.0)
        self.assertAlmostEqual(got, ref, places=6)

    def test_gabor_rectifies_to_zero(self):
        from brainpy_state._nest_spatial.kernels import gabor
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.0, 1.0]]) * u.um            # Y=1 -> cos(pi) = -1 -> rectified to 0
        k = gabor(distance.x, distance.y, lam=2.0)
        self.assertAlmostEqual(float(u.get_magnitude(k._eval_pair(pre, post)[0, 0])), 0.0, places=7)

    def test_gabor_rotated_and_phased(self):
        from brainpy_state._nest_spatial.kernels import gabor
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[1.0, 2.0]]) * u.um
        k = gabor(distance.x, distance.y, theta=90.0, gamma=2.0, std=1.5, lam=3.0, psi=30.0)
        got = float(u.get_magnitude(k._eval_pair(pre, post)[0, 0]))
        X, Y, th = 1.0, 2.0, math.radians(90.0)
        xp = X * math.cos(th) + Y * math.sin(th)
        yp = -X * math.sin(th) + Y * math.cos(th)
        ref = (max(math.cos(2 * math.pi * yp / 3.0 + math.radians(30.0)), 0.0)
               * math.exp(-(4.0 * xp ** 2 + yp ** 2) / (2 * 1.5 ** 2)))
        self.assertAlmostEqual(got, ref, places=6)

    def test_gaussian2D_reduces_to_product_when_rho0(self):
        from brainpy_state._nest_spatial.kernels import gaussian2D
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[2.0, 4.0]]) * u.um
        k = gaussian2D(distance.x, distance.y, std_x=1.0, std_y=2.0, rho=0.0)
        got = float(u.get_magnitude(k._eval_pair(pre, post)[0, 0]))
        ref = math.exp(-(2.0 ** 2) / 2.0) * math.exp(-(4.0 ** 2) / (2 * 2.0 ** 2))
        self.assertAlmostEqual(got, ref, places=6)

    def test_gaussian2D_with_correlation(self):
        from brainpy_state._nest_spatial.kernels import gaussian2D
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[1.0, 1.0]]) * u.um
        k = gaussian2D(distance.x, distance.y, std_x=1.0, std_y=1.0, rho=0.5)
        got = float(u.get_magnitude(k._eval_pair(pre, post)[0, 0]))
        X, Y, rho = 1.0, 1.0, 0.5
        Cx = 1 / (2 * (1 - rho ** 2)); Cy = Cx; Cxy = rho / ((1 - rho ** 2))
        ref = math.exp(-X ** 2 * Cx - Y ** 2 * Cy + X * Y * Cxy)
        self.assertAlmostEqual(got, ref, places=6)


if __name__ == '__main__':
    unittest.main()
