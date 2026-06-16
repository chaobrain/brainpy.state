# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import Simulator, iaf_psc_exp
from brainpy_state._nest_network import (
    all_to_all, one_to_one, fixed_indegree, pairwise_bernoulli,
    fixed_total_number, third_factor_bernoulli_with_pool, explicit_edges,
)
from brainpy_state._nest_network._connectivity import ConnSpec
from brainpy_state._nest_network._rules import _ExplicitEdges


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


class TestExplicitEdges(unittest.TestCase):
    """`_ExplicitEdges` returns a fixed ConnSpec, ignoring sampling args."""

    def test_returns_wrapped_spec(self):
        spec = ConnSpec(jnp.array([0, 1, 2]), jnp.array([3, 4, 5]), 3)
        rule = _ExplicitEdges(spec)
        out = rule.sample(9, 9, key=jax.random.key(0), pre_is_post=False,
                          allow_autapses=True, allow_multapses=True)
        self.assertIs(out, spec)

    def test_ignores_key_and_flags(self):
        spec = ConnSpec(jnp.array([0]), jnp.array([0]), 1)
        rule = _ExplicitEdges(spec)
        a = rule.sample(1, 1, key=jax.random.key(1), pre_is_post=True,
                        allow_autapses=False, allow_multapses=False)
        b = rule.sample(99, 99, key=jax.random.key(42), pre_is_post=False,
                        allow_autapses=True, allow_multapses=True)
        np.testing.assert_array_equal(np.asarray(a.pre_idx), np.asarray(b.pre_idx))


class TestExplicitEdgesPublic(unittest.TestCase):
    """Public ``explicit_edges(pre_idx, post_idx)`` factory + end-to-end wiring."""

    def test_factory_wraps_spec(self):
        rule = explicit_edges([0, 1, 2], [3, 4, 5])
        self.assertIsInstance(rule, _ExplicitEdges)
        spec = rule.sample(9, 9, key=jax.random.key(0), pre_is_post=False,
                           allow_autapses=True, allow_multapses=True)
        np.testing.assert_array_equal(np.asarray(spec.pre_idx), [0, 1, 2])
        np.testing.assert_array_equal(np.asarray(spec.post_idx), [3, 4, 5])
        self.assertEqual(spec.n_edges, 3)

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            explicit_edges(np.array([0, 1]), np.array([0, 1, 2]))

    def test_rejects_non_1d(self):
        with self.assertRaises(ValueError):
            explicit_edges(np.zeros((2, 2), int), np.zeros((2, 2), int))

    def test_rejects_non_integer(self):
        with self.assertRaises(ValueError):
            explicit_edges(np.array([0.0, 1.0]), np.array([1.0, 0.0]))

    def test_wires_exact_pairs_via_get_connections(self):
        # The linchpin for sudoku: one sparse explicit-edge projection, read back
        # exactly the (pre, post) pairs it was given (order-independent set equality).
        pre = np.array([0, 0, 2, 3], dtype=int)
        post = np.array([1, 4, 4, 0], dtype=int)
        sim = Simulator(dt=0.1 * u.ms)
        a = sim.create(iaf_psc_exp, 5)
        sim.connect(a, a, rule=explicit_edges(pre, post),
                    weight=-0.2 * u.pA, delay=1.0 * u.ms, comm='sparse')
        conns = sim.get_connections(source=a, target=a)
        got = set(zip(np.asarray(conns.get('source')).tolist(),
                      np.asarray(conns.get('target')).tolist()))
        self.assertEqual(got, set(zip(pre.tolist(), post.tolist())))

    def test_per_edge_weight_set_roundtrip(self):
        # De-risks the clue clamp: per-edge weights settable on an explicit-edge
        # sparse projection, keyed by the edge's target neuron (order-robust).
        pre = np.array([0, 0, 1, 1], dtype=int)
        post = np.array([2, 3, 4, 5], dtype=int)
        sim = Simulator(dt=0.1 * u.ms)
        a = sim.create(iaf_psc_exp, 6)
        sim.connect(a, a, rule=explicit_edges(pre, post),
                    weight=0.0 * u.pA, delay=1.0 * u.ms, comm='sparse')
        conns = sim.get_connections(source=a, target=a)
        tgt = np.asarray(conns.get('target'))
        new_w = np.where(tgt >= 4, 1.3, 0.0)          # weight keyed by target neuron
        conns.set('weight', new_w * u.pA)
        rt = sim.get_connections(source=a, target=a)
        got = u.Quantity(rt.get('weight')).to_decimal(u.pA)
        np.testing.assert_allclose(got, np.where(np.asarray(rt.get('target')) >= 4, 1.3, 0.0))


class TestThirdFactorBernoulliWithPool(unittest.TestCase):
    """`third_factor_bernoulli_with_pool` spec: validation + derived edge sampling."""

    def test_invalid_p_rejected(self):
        with self.assertRaises(ValueError):
            third_factor_bernoulli_with_pool(p=-0.1, pool_size=1, pool_type='block')
        with self.assertRaises(ValueError):
            third_factor_bernoulli_with_pool(p=1.5, pool_size=1, pool_type='block')

    def test_invalid_pool_size_rejected(self):
        with self.assertRaises(ValueError):
            third_factor_bernoulli_with_pool(p=1.0, pool_size=0, pool_type='block')

    def test_invalid_pool_type_rejected(self):
        with self.assertRaises(ValueError):
            third_factor_bernoulli_with_pool(p=1.0, pool_size=1, pool_type='nope')

    def test_sample_third_block_deterministic(self):
        # primary: all-to-all pre(2) x post(10); block pool_size=1, n_third=5, p=1.
        pre = jnp.repeat(jnp.arange(2), 10)
        post = jnp.tile(jnp.arange(10), 2)
        primary = ConnSpec(pre, post, 20)
        spec = third_factor_bernoulli_with_pool(p=1.0, pool_size=1, pool_type='block')
        tin, tout = spec.sample_third(primary, n_post=10, n_third=5, key=jax.random.key(0))
        # third_in: pre_i -> astro(post//2); third_out: astro(post//2) -> post.
        np.testing.assert_array_equal(np.asarray(tin.pre_idx), np.asarray(pre))
        np.testing.assert_array_equal(np.asarray(tin.post_idx), np.asarray(post) // 2)
        np.testing.assert_array_equal(np.asarray(tout.pre_idx), np.asarray(post) // 2)
        np.testing.assert_array_equal(np.asarray(tout.post_idx), np.asarray(post))
        self.assertEqual(tin.n_edges, 20)
        self.assertEqual(tout.n_edges, 20)

    def test_sample_third_p0_empty(self):
        pre = jnp.repeat(jnp.arange(2), 6)
        post = jnp.tile(jnp.arange(6), 2)
        primary = ConnSpec(pre, post, 12)
        spec = third_factor_bernoulli_with_pool(p=0.0, pool_size=1, pool_type='block')
        tin, tout = spec.sample_third(primary, n_post=6, n_third=6, key=jax.random.key(0))
        self.assertEqual(tin.n_edges, 0)
        self.assertEqual(tout.n_edges, 0)

    def test_sample_third_reproducible(self):
        pre = jnp.repeat(jnp.arange(4), 6)
        post = jnp.tile(jnp.arange(6), 4)
        primary = ConnSpec(pre, post, 24)
        spec = third_factor_bernoulli_with_pool(p=0.5, pool_size=2, pool_type='random')
        t1 = spec.sample_third(primary, n_post=6, n_third=6, key=jax.random.key(3))
        t2 = spec.sample_third(primary, n_post=6, n_third=6, key=jax.random.key(3))
        np.testing.assert_array_equal(np.asarray(t1[0].pre_idx), np.asarray(t2[0].pre_idx))
        np.testing.assert_array_equal(np.asarray(t1[1].post_idx), np.asarray(t2[1].post_idx))
