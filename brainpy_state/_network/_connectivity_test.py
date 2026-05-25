# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u

from brainpy_state._dist import Normal
from brainpy_state._network._connectivity import (
    sample_one_to_one,
    sample_all_to_all,
    sample_pairwise_bernoulli,
    sample_fixed_indegree,
    sample_fixed_outdegree,
    sample_fixed_total_number,
    sample_pairwise_poisson,
    resolve_param,
)


class TestConnectivitySamplers(unittest.TestCase):
    def test_one_to_one_requires_equal_sizes(self):
        with self.assertRaises(ValueError):
            sample_one_to_one(5, 4)

    def test_one_to_one_edges(self):
        spec = sample_one_to_one(4, 4)
        np.testing.assert_array_equal(spec.pre_idx, [0, 1, 2, 3])
        np.testing.assert_array_equal(spec.post_idx, [0, 1, 2, 3])

    def test_all_to_all_with_autapses(self):
        spec = sample_all_to_all(3, 3, pre_is_post=True, allow_autapses=True)
        self.assertEqual(spec.n_edges, 9)

    def test_all_to_all_without_autapses(self):
        spec = sample_all_to_all(3, 3, pre_is_post=True, allow_autapses=False)
        self.assertEqual(spec.n_edges, 6)
        pairs = set(zip(spec.pre_idx.tolist(), spec.post_idx.tolist()))
        for i in range(3):
            self.assertNotIn((i, i), pairs)

    def test_pairwise_bernoulli_density(self):
        key = jax.random.key(0)
        spec = sample_pairwise_bernoulli(
            100, 100, p=0.1, key=key,
            pre_is_post=False, allow_autapses=True, allow_multapses=True,
        )
        density = spec.n_edges / (100 * 100)
        self.assertAlmostEqual(density, 0.1, delta=0.02)

    def test_fixed_indegree_each_post_has_K(self):
        key = jax.random.key(1)
        spec = sample_fixed_indegree(
            n_pre=50, n_post=20, K=10, key=key,
            pre_is_post=False, allow_autapses=True, allow_multapses=False,
        )
        for j in range(20):
            self.assertEqual(int(jnp.sum(spec.post_idx == j)), 10)

    def test_fixed_outdegree_each_pre_has_K(self):
        key = jax.random.key(2)
        spec = sample_fixed_outdegree(
            n_pre=20, n_post=50, K=10, key=key,
            pre_is_post=False, allow_autapses=True, allow_multapses=False,
        )
        for i in range(20):
            self.assertEqual(int(jnp.sum(spec.pre_idx == i)), 10)

    def test_fixed_total_number(self):
        key = jax.random.key(3)
        spec = sample_fixed_total_number(
            n_pre=50, n_post=50, N=137, key=key,
            pre_is_post=False, allow_autapses=True, allow_multapses=True,
        )
        self.assertEqual(spec.n_edges, 137)

    def test_pairwise_poisson_mean(self):
        key = jax.random.key(4)
        spec = sample_pairwise_poisson(
            n_pre=100, n_post=100, mean=0.05, key=key,
            pre_is_post=False, allow_autapses=True,
        )
        expected = 100 * 100 * 0.05
        self.assertAlmostEqual(spec.n_edges, expected, delta=0.1 * expected)


class TestResolveParam(unittest.TestCase):
    def test_scalar_broadcast(self):
        key = jax.random.key(0)
        out = resolve_param(0.5, (10,), key)
        self.assertEqual(out.shape, (10,))
        self.assertTrue(bool(jnp.all(out == 0.5)))

    def test_array_passthrough(self):
        key = jax.random.key(0)
        arr = jnp.arange(10.0)
        out = resolve_param(arr, (10,), key)
        np.testing.assert_array_equal(out, arr)

    def test_distribution_sampled(self):
        key = jax.random.key(0)
        out = resolve_param(Normal(mean=0.0, std=1.0), (1000,), key)
        self.assertEqual(out.shape, (1000,))
        self.assertAlmostEqual(float(jnp.mean(out)), 0.0, delta=0.1)

    def test_array_shape_mismatch_raises(self):
        key = jax.random.key(0)
        with self.assertRaises(ValueError):
            resolve_param(jnp.zeros(7), (10,), key)
