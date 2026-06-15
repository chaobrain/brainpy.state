# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax
import jax.numpy as jnp
import brainunit as u

from brainpy_state._dist import Distribution, Normal, LogNormal, Uniform


class TestDistributions(unittest.TestCase):
    def test_normal_shape_and_seed_determinism(self):
        d = Normal(mean=0.0, std=1.0)
        key = jax.random.key(0)
        a = d.sample((100,), key)
        b = d.sample((100,), key)
        self.assertEqual(a.shape, (100,))
        self.assertTrue(jnp.allclose(a, b))

    def test_normal_carries_units(self):
        d = Normal(mean=0.1 * u.nS, std=0.01 * u.nS)
        key = jax.random.key(1)
        x = d.sample((10,), key)
        self.assertTrue(u.get_unit(x).has_same_dim(u.nS))

    def test_uniform_bounds(self):
        d = Uniform(low=-1.0, high=2.0)
        key = jax.random.key(2)
        x = d.sample((1000,), key)
        self.assertGreaterEqual(float(jnp.min(x)), -1.0 - 1e-6)
        self.assertLessEqual(float(jnp.max(x)), 2.0 + 1e-6)

    def test_lognormal_positive(self):
        d = LogNormal(mean=0.0, std=1.0)
        key = jax.random.key(3)
        x = d.sample((100,), key)
        self.assertTrue(bool(jnp.all(x > 0)))

    def test_is_distribution(self):
        for cls in [Normal, LogNormal, Uniform]:
            self.assertTrue(issubclass(cls, Distribution))
