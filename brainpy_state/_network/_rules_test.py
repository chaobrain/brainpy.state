# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax

from brainpy_state._network import (
    all_to_all, one_to_one, fixed_indegree, pairwise_bernoulli,
    fixed_total_number,
)


class TestRules(unittest.TestCase):
    def test_fixed_indegree_edge_count(self):
        spec = fixed_indegree(3).sample(
            10, 5, key=jax.random.key(0), pre_is_post=False,
            allow_autapses=True, allow_multapses=True)
        self.assertEqual(spec.n_edges, 15)        # K=3 per each of 5 post

    def test_all_to_all_count(self):
        spec = all_to_all.sample(
            4, 6, key=jax.random.key(0), pre_is_post=False,
            allow_autapses=True, allow_multapses=True)
        self.assertEqual(spec.n_edges, 24)

    def test_one_to_one_diagonal(self):
        spec = one_to_one.sample(
            5, 5, key=jax.random.key(0), pre_is_post=False,
            allow_autapses=True, allow_multapses=True)
        self.assertEqual(spec.n_edges, 5)

    def test_fixed_indegree_negative_K_rejected(self):
        with self.assertRaises(ValueError):
            fixed_indegree(-1)

    def test_fixed_total_number_edge_count(self):
        spec = fixed_total_number(5).sample(
            10, 5, key=jax.random.key(0), pre_is_post=False,
            allow_autapses=True, allow_multapses=True)
        self.assertEqual(spec.n_edges, 5)

    def test_fixed_total_number_negative_rejected(self):
        with self.assertRaises(ValueError):
            fixed_total_number(-1)


class TestPairwiseBernoulli(unittest.TestCase):
    def _sample(self, n_pre, n_post, p, *, key=0, pre_is_post=False,
                allow_autapses=True):
        return pairwise_bernoulli(p).sample(
            n_pre, n_post, key=jax.random.key(key), pre_is_post=pre_is_post,
            allow_autapses=allow_autapses, allow_multapses=True)

    def test_p1_connects_all(self):
        spec = self._sample(4, 6, 1.0)
        self.assertEqual(spec.n_edges, 24)

    def test_p0_connects_none(self):
        spec = self._sample(8, 8, 0.0)
        self.assertEqual(spec.n_edges, 0)

    def test_density_matches_p(self):
        # ~p * n_pre * n_post edges over a large pair (Bernoulli LLN).
        n_pre, n_post, p = 200, 200, 0.2
        spec = self._sample(n_pre, n_post, p)
        expected = p * n_pre * n_post
        self.assertLess(abs(spec.n_edges - expected) / expected, 0.05)

    def test_autapses_excluded_when_recurrent(self):
        # p=1 recurrent without autapses -> full minus the diagonal.
        spec = self._sample(10, 10, 1.0, pre_is_post=True, allow_autapses=False)
        self.assertEqual(spec.n_edges, 90)
        self.assertTrue(bool((spec.pre_idx != spec.post_idx).all()))

    def test_deterministic_under_key(self):
        a = self._sample(50, 50, 0.3, key=7)
        b = self._sample(50, 50, 0.3, key=7)
        self.assertEqual(a.n_edges, b.n_edges)

    def test_invalid_p_rejected(self):
        with self.assertRaises(ValueError):
            pairwise_bernoulli(-0.1)
        with self.assertRaises(ValueError):
            pairwise_bernoulli(1.5)
