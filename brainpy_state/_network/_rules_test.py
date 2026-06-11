# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax

from brainpy_state._network import all_to_all, one_to_one, fixed_indegree


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
