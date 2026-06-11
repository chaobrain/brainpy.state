# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import jax
import jax.numpy as jnp
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import iaf_psc_alpha
from brainpy_state._network import one_to_one
from brainpy_state._network._event_proj import EventProjection


class _Box:
    def __init__(self, val):
        self.val = val


class TestEventProjection(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_one_to_one_weighted_delta_after_delay(self):
        post = iaf_psc_alpha(1, tau_syn_ex=1.0 * u.ms)
        box = _Box(jnp.zeros(1))
        proj = EventProjection(
            pre_spike=lambda: box.val, n_pre_pop=1, pre_local_idx=jnp.arange(1),
            post=post, post_local_idx=jnp.arange(1), rule=one_to_one,
            weight=100.0 * u.pA, delay=0.5 * u.ms)
        brainstate.nn.init_all_states(post)
        brainstate.nn.init_all_states(proj)

        # No spike yet -> the delta summed by the neuron is ~0 pA.
        with brainstate.environ.context(t=0.0 * u.ms, i=0):
            proj.update()
            summed0 = post.sum_delta_inputs(0.0 * u.pA)
        self.assertAlmostEqual(float(u.get_mantissa(summed0 / u.pA)[0]), 0.0, places=6)

        # Emit one spike; after >= delay steps the neuron must see ~100 pA once.
        box.val = jnp.ones(1)
        seen = []
        for k in range(1, 12):
            with brainstate.environ.context(t=k * 0.1 * u.ms, i=k):
                proj.update()
                seen.append(float(u.get_mantissa(post.sum_delta_inputs(0.0 * u.pA) / u.pA)[0]))
            box.val = jnp.zeros(1)  # single spike only at step 0
        self.assertTrue(any(abs(v - 100.0) < 1e-3 for v in seen),
                        f'expected a ~100 pA delta once; saw {seen}')
