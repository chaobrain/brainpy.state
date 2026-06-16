# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import jax.numpy as jnp
import brainunit as u

from brainpy_state import LIF, Expon, COBA
from brainpy_state._nest_network._projections import _RuleProj, OneToOneProj


class TestRuleProjBase(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_stores_pre_and_post(self):
        pre = LIF(5)
        post = LIF(5)
        proj = OneToOneProj(
            pre, post,
            weight=0.1 * u.nS,
            syn=Expon.desc(5, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        self.assertIs(proj.pre, pre)
        self.assertIs(proj.post, post)

    def test_per_edge_weight_shape(self):
        pre = LIF(5)
        post = LIF(5)
        proj = OneToOneProj(
            pre, post,
            weight=0.1 * u.nS,
            syn=Expon.desc(5, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        w = proj._weight_matrix.value  # (n_pre, n_post)
        self.assertEqual(w.shape, (5, 5))
        # one-to-one: only diagonal has weight
        diag = jnp.diag(u.get_mantissa(w))
        off = u.get_mantissa(w) - jnp.diag(diag)
        self.assertTrue(bool(jnp.all(jnp.abs(off) < 1e-9)))

    def test_seed_determinism(self):
        # Determinism is delegated to rule classes that use randomness.
        # OneToOne has no randomness, so two builds with same args match.
        pre1, post1 = LIF(5), LIF(5)
        pre2, post2 = LIF(5), LIF(5)
        p1 = OneToOneProj(pre1, post1, weight=0.1*u.nS,
                          syn=Expon.desc(5, tau=5*u.ms),
                          out=COBA.desc(E=0*u.mV))
        p2 = OneToOneProj(pre2, post2, weight=0.1*u.nS,
                          syn=Expon.desc(5, tau=5*u.ms),
                          out=COBA.desc(E=0*u.mV))
        self.assertTrue(jnp.allclose(
            u.get_mantissa(p1._weight_matrix.value),
            u.get_mantissa(p2._weight_matrix.value)))

    def test_delay_raises_until_implemented(self):
        pre = LIF(5)
        post = LIF(5)
        with self.assertRaises(NotImplementedError):
            OneToOneProj(
                pre, post,
                weight=0.1*u.nS,
                delay=1.0*u.ms,
                syn=Expon.desc(5, tau=5*u.ms),
                out=COBA.desc(E=0*u.mV),
            )


from brainpy_state._nest_network._projections import (
    AllToAllProj,
    PairwiseBernoulliProj, SymmetricPairwiseBernoulliProj,
    FixedIndegreeProj, FixedOutdegreeProj,
    FixedTotalNumberProj, PairwisePoissonProj,
)


class TestAllToAllProj(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_dense_weight_with_autapses(self):
        pre = LIF(4); post = LIF(4)
        proj = AllToAllProj(
            pre, post, weight=0.5*u.nS,
            syn=Expon.desc(4, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        self.assertTrue(jnp.allclose(W, jnp.full((4, 4), 0.5)))

    def test_no_autapses_when_pre_is_post(self):
        pop = LIF(4)
        proj = AllToAllProj(
            pop, pop, weight=0.5*u.nS,
            syn=Expon.desc(4, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            allow_autapses=False,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        self.assertTrue(jnp.allclose(jnp.diag(W), 0.0))
        off = W - jnp.diag(jnp.diag(W))
        self.assertTrue(jnp.allclose(off, 0.5 * (1 - jnp.eye(4))))


class TestBernoulliProjs(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_pairwise_bernoulli_density_within_tolerance(self):
        pre = LIF(80); post = LIF(80)
        proj = PairwiseBernoulliProj(
            pre, post, p=0.1, weight=1.0*u.nS,
            syn=Expon.desc(80, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=42,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        density = float(jnp.mean(W > 0))
        self.assertAlmostEqual(density, 0.1, delta=0.025)

    def test_pairwise_bernoulli_seed_determinism(self):
        pre = LIF(40); post = LIF(40)
        kw = dict(p=0.2, weight=1.0*u.nS,
                  syn=Expon.desc(40, tau=5*u.ms),
                  out=COBA.desc(E=0*u.mV),
                  seed=7)
        a = u.get_mantissa(PairwiseBernoulliProj(pre, post, **kw)._weight_matrix.value)
        b = u.get_mantissa(PairwiseBernoulliProj(pre, post, **kw)._weight_matrix.value)
        self.assertTrue(jnp.allclose(a, b))

    def test_symmetric_pairwise_bernoulli_is_symmetric(self):
        pop = LIF(40)
        proj = SymmetricPairwiseBernoulliProj(
            pop, pop, p=0.2, weight=1.0*u.nS,
            syn=Expon.desc(40, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=11,
        )
        W = u.get_mantissa(proj._weight_matrix.value) > 0
        self.assertTrue(jnp.array_equal(W, W.T))


class TestFixedDegreeProjs(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_fixed_indegree_each_post_has_K(self):
        pre = LIF(50); post = LIF(20)
        proj = FixedIndegreeProj(
            pre, post, K=10, weight=1.0*u.nS,
            syn=Expon.desc(20, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=3, allow_multapses=False,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        for j in range(20):
            self.assertEqual(int(jnp.sum(W[:, j] > 0)), 10)

    def test_fixed_outdegree_each_pre_has_K(self):
        pre = LIF(20); post = LIF(50)
        proj = FixedOutdegreeProj(
            pre, post, K=8, weight=1.0*u.nS,
            syn=Expon.desc(50, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=4, allow_multapses=False,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        for i in range(20):
            self.assertEqual(int(jnp.sum(W[i, :] > 0)), 8)


class TestFixedTotalAndPoisson(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_fixed_total_number(self):
        pre = LIF(50); post = LIF(50)
        proj = FixedTotalNumberProj(
            pre, post, N=137, weight=1.0*u.nS,
            syn=Expon.desc(50, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=8,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        # With allow_multapses=True (default), the W.at[].add() accumulates
        # — count non-zero entries.
        self.assertGreaterEqual(int(jnp.sum(W > 0)), 130)

    def test_pairwise_poisson_mean(self):
        pre = LIF(50); post = LIF(50)
        proj = PairwisePoissonProj(
            pre, post, mean=0.1, weight=1.0*u.nS,
            syn=Expon.desc(50, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=9,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        # Expected count of edges ≈ 50*50*0.1 = 250.
        # W accumulates per-pair counts via add(); mean per pair ≈ 0.1.
        self.assertAlmostEqual(float(jnp.mean(W)), 0.1, delta=0.025)
