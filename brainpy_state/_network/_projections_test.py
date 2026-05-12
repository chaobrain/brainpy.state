# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import jax.numpy as jnp
import saiunit as u

from brainpy_state import LIF, Expon, COBA
from brainpy_state._network._projections import _RuleProj, OneToOneProj


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
