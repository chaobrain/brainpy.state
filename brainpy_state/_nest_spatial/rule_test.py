# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for the spatial_pairwise_bernoulli ConnRule."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._nest_network.rules import ConnRule
from brainpy_state._nest_spatial.layers import grid
from brainpy_state._nest_spatial.distance import pairwise_distance
from brainpy_state._nest_spatial.kernels import distance, gaussian
from brainpy_state._nest_spatial.masks import circular
from brainpy_state._nest_spatial.rule import SpatialConnRule, spatial_pairwise_bernoulli


def _bind(rule, lay_a, lay_b):
    return rule.with_coords(lay_a.coords, lay_b.coords)


def _sample(rule, n_pre, n_post, *, seed=0, pre_is_post=True, autapses=True, multapses=True):
    return rule.sample(n_pre, n_post, key=jax.random.key(seed), pre_is_post=pre_is_post,
                       allow_autapses=autapses, allow_multapses=multapses)


class TestSpatialRule(unittest.TestCase):
    def test_is_a_connrule(self):
        self.assertIsInstance(spatial_pairwise_bernoulli(p=0.5), ConnRule)

    def test_unbound_sample_raises(self):
        r = spatial_pairwise_bernoulli(p=0.5)
        with self.assertRaises(ValueError):
            _sample(r, 3, 3, pre_is_post=False)

    def test_circular_mask_hard_cutoff_no_edge_beyond_radius(self):
        a = grid([10, 10], extent=[3.0, 3.0])
        r = _bind(spatial_pairwise_bernoulli(p=1.0, mask=circular(0.5)), a, a)
        spec = _sample(r, 100, 100, seed=0)
        disp = u.get_magnitude((a.coords[spec.pre_idx] - a.coords[spec.post_idx]).to(u.um))
        dist = np.sqrt((disp ** 2).sum(-1))
        self.assertTrue(dist.max() <= 0.5 + 1e-6)          # NO realized edge beyond the radius
        self.assertGreater(spec.n_edges, 0)

    def test_gaussian_density_matches_analytic_expectation(self):
        a = grid([20, 20], extent=[3.0, 3.0])
        rule = spatial_pairwise_bernoulli(p=gaussian(distance, std=0.3), mask=circular(1.0))
        r = _bind(rule, a, a)
        dd = u.get_magnitude(pairwise_distance(a.coords, a.coords).to(u.um))
        prob = np.exp(-(dd / 0.3) ** 2 / 2.0)
        prob[dd > 1.0] = 0.0                                # circular(1.0) cutoff
        expected = prob.sum()
        spec = _sample(r, 400, 400, seed=1)
        self.assertAlmostEqual(spec.n_edges / expected, 1.0, delta=0.06)

    def test_float_p_flat_density(self):
        a = grid([20, 20], extent=[10.0, 10.0])            # large extent, no mask -> ~flat
        r = _bind(spatial_pairwise_bernoulli(p=0.3), a, a)
        spec = _sample(r, 400, 400, seed=2)
        self.assertAlmostEqual(spec.n_edges / (400 * 400), 0.3, delta=0.02)

    def test_allow_autapses_false_zeroes_diagonal(self):
        a = grid([5, 5], extent=[1.0, 1.0])
        r = _bind(spatial_pairwise_bernoulli(p=1.0), a, a)
        spec = _sample(r, 25, 25, seed=0, autapses=False)
        self.assertFalse(bool(np.any(np.asarray(spec.pre_idx) == np.asarray(spec.post_idx))))
        self.assertEqual(spec.n_edges, 25 * 25 - 25)        # full minus diagonal

    def test_rule_level_allow_autapses_false(self):
        a = grid([4, 4], extent=[1.0, 1.0])
        # rule says no autapses even when connect-level allow_autapses=True
        r = _bind(spatial_pairwise_bernoulli(p=1.0, allow_autapses=False), a, a)
        spec = _sample(r, 16, 16, seed=0, autapses=True)
        self.assertFalse(bool(np.any(np.asarray(spec.pre_idx) == np.asarray(spec.post_idx))))

    def test_std_to_zero_only_self_edges(self):
        a = grid([5, 5], extent=[1.0, 1.0])
        r = _bind(spatial_pairwise_bernoulli(p=gaussian(distance, std=1e-9)), a, a)
        spec = _sample(r, 25, 25, seed=0, autapses=True)
        self.assertEqual(spec.n_edges, 25)                  # only the d=0 self-pairs survive
        np.testing.assert_array_equal(np.asarray(spec.pre_idx), np.asarray(spec.post_idx))

    def test_empty_sample_for_zero_p(self):
        a = grid([6, 6], extent=[1.0, 1.0])
        r = _bind(spatial_pairwise_bernoulli(p=0.0), a, a)
        spec = _sample(r, 36, 36, seed=0)
        self.assertEqual(spec.n_edges, 0)

    def test_mask_larger_than_extent_is_all_to_all(self):
        a = grid([5, 5], extent=[1.0, 1.0])
        r = _bind(spatial_pairwise_bernoulli(p=1.0, mask=circular(100.0)), a, a)
        spec = _sample(r, 25, 25, seed=0, autapses=True)
        self.assertEqual(spec.n_edges, 25 * 25)

    def test_seeded_reproducible(self):
        a = grid([8, 8], extent=[2.0, 2.0])
        r = _bind(spatial_pairwise_bernoulli(p=gaussian(distance, std=0.4)), a, a)
        s1 = _sample(r, 64, 64, seed=7)
        s2 = _sample(r, 64, 64, seed=7)
        np.testing.assert_array_equal(s1.pre_idx, s2.pre_idx)
        np.testing.assert_array_equal(s1.post_idx, s2.post_idx)

    def test_different_seed_differs(self):
        a = grid([8, 8], extent=[2.0, 2.0])
        r = _bind(spatial_pairwise_bernoulli(p=gaussian(distance, std=0.4)), a, a)
        s1 = _sample(r, 64, 64, seed=1)
        s2 = _sample(r, 64, 64, seed=2)
        self.assertNotEqual(s1.n_edges, s2.n_edges)

    def test_per_axis_kernel_is_anisotropic(self):
        # A single x-column grid: every node shares x, so gaussian(distance.x, std->0)
        # connects EVERY pair (|dx|=0). A scalar-distance kernel would connect only the
        # 5 diagonal self-pairs. Proves _prob_matrix routes the per-axis expression.
        a = grid([1, 5], extent=[1.0, 1.0])
        r = _bind(spatial_pairwise_bernoulli(p=gaussian(distance.x, std=1e-9)), a, a)
        prob = np.asarray(r._prob_matrix())
        self.assertEqual(int((prob > 0.5).sum()), 25)

    def test_2d_vs_3d_free_shapes(self):
        a2 = grid([4, 4], extent=[1.0, 1.0])
        a3 = grid([3, 3, 3], extent=[1.0, 1.0, 1.0])
        r2 = _bind(spatial_pairwise_bernoulli(p=0.5), a2, a2)
        r3 = _bind(spatial_pairwise_bernoulli(p=0.5), a3, a3)
        self.assertGreater(_sample(r2, 16, 16, seed=0).n_edges, 0)
        self.assertGreater(_sample(r3, 27, 27, seed=0).n_edges, 0)


if __name__ == '__main__':
    unittest.main()
