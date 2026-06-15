# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._dist import Normal
from brainpy_state._network._connectivity import (
    ConnSpec,
    sample_one_to_one,
    sample_all_to_all,
    sample_pairwise_bernoulli,
    sample_fixed_indegree,
    sample_fixed_outdegree,
    sample_fixed_total_number,
    sample_pairwise_poisson,
    resolve_param,
    build_pool_map,
    sample_third_factor_pairing,
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


class TestBuildPoolMapBlock(unittest.TestCase):
    """`build_pool_map` block: deterministic contiguous non-overlapping blocks.

    Mirrors NEST ``get_first_pool_index_`` (conn_builder.cpp L933): pool_size>1 ->
    ``j*pool_size``; pool_size==1 -> ``j // (n_post//n_third)`` (integer division).
    """

    def test_pool_size_1_blocks_by_floor_div(self):
        # n_post=10, n_third=5 -> targets_per_third=2 -> astro_j = j//2.
        pool = build_pool_map(10, 5, pool_size=1, pool_type='block', key=jax.random.key(0))
        np.testing.assert_array_equal(
            np.asarray(pool).reshape(-1), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_pool_size_1_one_to_one_when_equal_sizes(self):
        # The N_astro == N_post case: astro_j = j (the goal's one-to-one map).
        pool = build_pool_map(6, 6, pool_size=1, pool_type='block', key=jax.random.key(0))
        np.testing.assert_array_equal(np.asarray(pool).reshape(-1), np.arange(6))

    def test_pool_size_gt_1_contiguous_unique_blocks(self):
        # n_post=3, pool_size=4, n_third=12: target j -> [4j, 4j+1, 4j+2, 4j+3].
        pool = build_pool_map(3, 12, pool_size=4, pool_type='block', key=jax.random.key(0))
        np.testing.assert_array_equal(
            np.asarray(pool), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

    def test_block_is_deterministic_independent_of_key(self):
        a = build_pool_map(10, 5, pool_size=1, pool_type='block', key=jax.random.key(1))
        b = build_pool_map(10, 5, pool_size=1, pool_type='block', key=jax.random.key(999))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestBuildPoolMapValidation(unittest.TestCase):
    """Constructor-equivalent validation, mirroring NEST ctor (conn_builder.cpp L800-838)."""

    def test_pool_size_below_1_rejected(self):
        with self.assertRaises(ValueError):
            build_pool_map(10, 5, pool_size=0, pool_type='block', key=jax.random.key(0))

    def test_pool_size_above_n_third_rejected(self):
        with self.assertRaises(ValueError):
            build_pool_map(10, 5, pool_size=6, pool_type='random', key=jax.random.key(0))

    def test_bad_pool_type_rejected(self):
        with self.assertRaises(ValueError):
            build_pool_map(10, 5, pool_size=1, pool_type='nonsense', key=jax.random.key(0))

    def test_block_pool_size_1_requires_divisible(self):
        # pool_size==1 + n_post % n_third != 0 -> incompatible (NEST BadProperty).
        with self.assertRaises(ValueError):
            build_pool_map(7, 5, pool_size=1, pool_type='block', key=jax.random.key(0))

    def test_block_pool_size_gt1_requires_exact_tiling(self):
        # pool_size>1 needs n_post*pool_size == n_third.
        with self.assertRaises(ValueError):
            build_pool_map(3, 10, pool_size=4, pool_type='block', key=jax.random.key(0))

    def test_random_has_no_size_compat_constraint(self):
        # random tolerates any (n_post, n_third, pool_size<=n_third).
        pool = build_pool_map(7, 5, pool_size=3, pool_type='random', key=jax.random.key(0))
        self.assertEqual(np.asarray(pool).shape, (7, 3))


class TestBuildPoolMapRandom(unittest.TestCase):
    """`build_pool_map` random: pool_size distinct astrocytes per target, seeded."""

    def test_shape_and_range(self):
        pool = np.asarray(build_pool_map(8, 5, pool_size=3, pool_type='random',
                                         key=jax.random.key(0)))
        self.assertEqual(pool.shape, (8, 3))
        self.assertTrue((pool >= 0).all() and (pool < 5).all())

    def test_each_pool_has_distinct_astrocytes(self):
        # NEST samples without replacement: each target's pool members are distinct.
        pool = np.asarray(build_pool_map(20, 6, pool_size=4, pool_type='random',
                                         key=jax.random.key(3)))
        for row in pool:
            self.assertEqual(len(set(row.tolist())), 4)

    def test_seeded_reproducible(self):
        a = build_pool_map(8, 5, pool_size=3, pool_type='random', key=jax.random.key(7))
        b = build_pool_map(8, 5, pool_size=3, pool_type='random', key=jax.random.key(7))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_different_keys_differ(self):
        a = build_pool_map(50, 20, pool_size=5, pool_type='random', key=jax.random.key(1))
        b = build_pool_map(50, 20, pool_size=5, pool_type='random', key=jax.random.key(2))
        self.assertFalse(np.array_equal(np.asarray(a), np.asarray(b)))


class TestSampleThirdFactorPairing(unittest.TestCase):
    """`sample_third_factor_pairing`: Bernoulli-pair primary edges, draw one astro/pool."""

    def _primary(self, n_pre, n_post):
        # all-to-all primary edges (deterministic order)
        pre = jnp.repeat(jnp.arange(n_pre), n_post)
        post = jnp.tile(jnp.arange(n_post), n_pre)
        return pre, post

    def test_p1_pairs_every_primary_edge(self):
        pre, post = self._primary(4, 6)
        pool = build_pool_map(6, 6, pool_size=1, pool_type='block', key=jax.random.key(0))
        tin_pre, astro, tout_post = sample_third_factor_pairing(
            pre, post, pool, p=1.0, key=jax.random.key(0))
        self.assertEqual(tin_pre.shape[0], pre.shape[0])      # every edge paired
        self.assertEqual(astro.shape[0], pre.shape[0])
        np.testing.assert_array_equal(np.asarray(tout_post), np.asarray(post))

    def test_p0_pairs_nothing(self):
        pre, post = self._primary(4, 6)
        pool = build_pool_map(6, 6, pool_size=1, pool_type='block', key=jax.random.key(0))
        tin_pre, astro, tout_post = sample_third_factor_pairing(
            pre, post, pool, p=0.0, key=jax.random.key(0))
        self.assertEqual(tin_pre.shape[0], 0)
        self.assertEqual(astro.shape[0], 0)

    def test_block_pool_size1_selects_floor_div_astro(self):
        # With block pool_size=1, the chosen astro for target j is j//targets_per_third.
        pre, post = self._primary(2, 10)
        pool = build_pool_map(10, 5, pool_size=1, pool_type='block', key=jax.random.key(0))
        tin_pre, astro, tout_post = sample_third_factor_pairing(
            pre, post, pool, p=1.0, key=jax.random.key(0))
        # astro selected == post//2 for every paired edge
        np.testing.assert_array_equal(np.asarray(astro), np.asarray(post) // 2)

    def test_third_in_pre_matches_primary_pre(self):
        # third_in source is the primary edge's pre (pre_i -> astro).
        pre, post = self._primary(3, 4)
        pool = build_pool_map(4, 4, pool_size=1, pool_type='block', key=jax.random.key(0))
        tin_pre, astro, tout_post = sample_third_factor_pairing(
            pre, post, pool, p=1.0, key=jax.random.key(0))
        np.testing.assert_array_equal(np.asarray(tin_pre), np.asarray(pre))

    def test_selected_astro_is_in_target_pool(self):
        # random pool: the drawn astro must belong to the target's pool.
        pre, post = self._primary(5, 8)
        pool = np.asarray(build_pool_map(8, 6, pool_size=3, pool_type='random',
                                         key=jax.random.key(2)))
        tin_pre, astro, tout_post = sample_third_factor_pairing(
            pre, post, jnp.asarray(pool), p=1.0, key=jax.random.key(5))
        astro = np.asarray(astro); tout_post = np.asarray(tout_post)
        for a, j in zip(astro, tout_post):
            self.assertIn(a, pool[j].tolist())

    def test_pairing_reproducible_under_seed(self):
        pre, post = self._primary(6, 6)
        pool = build_pool_map(6, 6, pool_size=2, pool_type='random', key=jax.random.key(0))
        out1 = sample_third_factor_pairing(pre, post, pool, p=0.5, key=jax.random.key(11))
        out2 = sample_third_factor_pairing(pre, post, pool, p=0.5, key=jax.random.key(11))
        for a, b in zip(out1, out2):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_p_out_of_range_rejected(self):
        # The pairing primitive guards its own p in [0, 1] (defensive: the public
        # factory validates too, but this function is callable directly).
        pre, post = self._primary(3, 4)
        pool = build_pool_map(4, 4, pool_size=1, pool_type='block', key=jax.random.key(0))
        for bad_p in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                sample_third_factor_pairing(pre, post, pool, p=bad_p, key=jax.random.key(0))

    def test_pairing_rate_in_binomial_band(self):
        # Over many edges, the realized pairing fraction tracks p.
        pre, post = self._primary(40, 40)  # 1600 edges
        pool = build_pool_map(40, 40, pool_size=1, pool_type='block', key=jax.random.key(0))
        _, astro, _ = sample_third_factor_pairing(pre, post, pool, p=0.5,
                                                  key=jax.random.key(0))
        frac = astro.shape[0] / pre.shape[0]
        self.assertAlmostEqual(frac, 0.5, delta=0.05)
