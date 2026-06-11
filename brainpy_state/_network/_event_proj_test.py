# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import iaf_psc_alpha
from brainpy_state._network import one_to_one, fixed_indegree
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

    def test_sparse_comm_matches_dense_fixed_indegree(self):
        # The sparse CSR path uses the SAME sampler + seed as the dense matmul,
        # so the per-step delta contributions must be bit-identical.
        n_pre, n_post, K = 30, 20, 5
        post_d = iaf_psc_alpha(n_post)
        post_s = iaf_psc_alpha(n_post)
        box = _Box(jnp.zeros(n_pre))
        common = dict(
            pre_spike=lambda: box.val, n_pre_pop=n_pre,
            pre_local_idx=jnp.arange(n_pre), post_local_idx=jnp.arange(n_post),
            rule=fixed_indegree(K), weight=7.0 * u.pA, delay=0.5 * u.ms,
            seed=3, allow_multapses=True)
        proj_d = EventProjection(post=post_d, comm='dense', **common)
        proj_s = EventProjection(post=post_s, comm='sparse', **common)
        for m in (post_d, post_s, proj_d, proj_s):
            brainstate.nn.init_all_states(m)

        rng = np.random.RandomState(0)
        saw_nonzero = False
        for k in range(15):
            box.val = jnp.asarray((rng.random(n_pre) < 0.3).astype(float))
            with brainstate.environ.context(t=k * 0.1 * u.ms, i=k):
                proj_d.update()
                proj_s.update()
                yd = np.asarray(u.get_mantissa(post_d.sum_delta_inputs(0. * u.pA) / u.pA))
                ys = np.asarray(u.get_mantissa(post_s.sum_delta_inputs(0. * u.pA) / u.pA))
            self.assertTrue(np.allclose(yd, ys, atol=1e-9),
                            f'step {k}: dense {yd} != sparse {ys}')
            saw_nonzero = saw_nonzero or bool(np.any(np.abs(yd) > 0))
        self.assertTrue(saw_nonzero, 'expected some non-zero delta contribution')
