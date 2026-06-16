# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Simulator orchestration of ``tripartite_connect`` (NEST ``TripartiteConnect``, 24).

NEST-free integration tests of
``Simulator.tripartite_connect(pre, post, third, conn_spec,
third_factor_conn_spec, syn_specs)``: one realized primary ``pre->post`` sample is
shared across three arms -- primary (``pre->post``), ``third_in`` (``pre->astro``,
delta IP3), and ``third_out`` (``astro->post``, the ``sic_connection``) -- via the
``third_factor_bernoulli_with_pool`` pool sampler. Reuses the merged static +
``sic_connection`` (15d) paths; no new deposit primitive.
"""
import gc
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import (aeif_cond_alpha_astro, astrocyte_lr_1994,
                           sic_connection, multimeter, pairwise_bernoulli,
                           all_to_all, third_factor_bernoulli_with_pool,
                           poisson_generator)
from brainpy_state._network import Simulator
from brainpy_state._network._event_proj import EventProjection


def _syn_specs(w_primary=1.0, w_third_in=1.0, w_a2n=1.0, delay=0.1, sic_delay_steps=10):
    """Static-arm syn_specs: conductance EPSP / delta IP3 / sic_connection."""
    return {
        'primary': {'weight': w_primary * u.nS, 'delay': delay * u.ms, 'receptor_type': 1},
        'third_in': {'weight': w_third_in, 'delay': delay * u.ms},
        'third_out': {'synapse': sic_connection(weight=w_a2n, delay_steps=sic_delay_steps)},
    }


def _edges(proj):
    """Unique (source, target) population-local pairs of a realized projection."""
    pe = proj.realized_edges()
    s = np.asarray(pe.source); t = np.asarray(pe.target)
    return set(zip(s.tolist(), t.tolist()))


class TestTripartiteConnectStructure(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_builds_three_projections(self):
        sim = Simulator(dt=0.1 * u.ms)
        pre = sim.create(aeif_cond_alpha_astro, 3)
        post = sim.create(aeif_cond_alpha_astro, 4)
        astro = sim.create(astrocyte_lr_1994, 2)
        prim, tin, tout = sim.tripartite_connect(
            pre, post, astro,
            conn_spec=all_to_all,
            third_factor_conn_spec=third_factor_bernoulli_with_pool(
                p=1.0, pool_size=1, pool_type='block'),
            syn_specs=_syn_specs(), seed=0)
        self.assertIsInstance(prim, EventProjection)
        self.assertIsInstance(tin, EventProjection)
        self.assertIsInstance(tout, EventProjection)

    def test_third_out_is_as_current_sic(self):
        sim = Simulator(dt=0.1 * u.ms)
        pre = sim.create(aeif_cond_alpha_astro, 3)
        post = sim.create(aeif_cond_alpha_astro, 4)
        astro = sim.create(astrocyte_lr_1994, 2)
        _, _, tout = sim.tripartite_connect(
            pre, post, astro, conn_spec=all_to_all,
            third_factor_conn_spec=third_factor_bernoulli_with_pool(
                p=1.0, pool_size=1, pool_type='block'),
            syn_specs=_syn_specs(), seed=0)
        self.assertTrue(tout._as_current)
        self.assertEqual(tout._channel_label, 'I_SIC')

    def test_registers_in_connections(self):
        sim = Simulator(dt=0.1 * u.ms)
        pre = sim.create(aeif_cond_alpha_astro, 3)
        post = sim.create(aeif_cond_alpha_astro, 4)
        astro = sim.create(astrocyte_lr_1994, 2)
        sim.tripartite_connect(
            pre, post, astro, conn_spec=all_to_all,
            third_factor_conn_spec=third_factor_bernoulli_with_pool(
                p=1.0, pool_size=1, pool_type='block'),
            syn_specs=_syn_specs(), seed=0)
        models = [m for (_, _, m, _) in sim._connections]
        self.assertIn('static_synapse', models)         # primary + third_in
        self.assertIn('sic_connection', models)          # third_out

    def test_block_shared_sample_deterministic_edges(self):
        # all-to-all primary (3x4); block pool_size=1, n_third=2 -> astro_j = j//2.
        # third_in: pre_i -> astro(j//2); third_out: astro(j//2) -> post_j.
        sim = Simulator(dt=0.1 * u.ms)
        pre = sim.create(aeif_cond_alpha_astro, 3)
        post = sim.create(aeif_cond_alpha_astro, 4)
        astro = sim.create(astrocyte_lr_1994, 2)
        prim, tin, tout = sim.tripartite_connect(
            pre, post, astro, conn_spec=all_to_all,
            third_factor_conn_spec=third_factor_bernoulli_with_pool(
                p=1.0, pool_size=1, pool_type='block'),
            syn_specs=_syn_specs(), seed=0)
        self.assertEqual(_edges(prim), {(i, j) for i in range(3) for j in range(4)})
        # third_out unique pairs: astro(j//2) -> post_j for all j
        self.assertEqual(_edges(tout), {(j // 2, j) for j in range(4)})
        # third_in unique pairs: pre_i -> astro(j//2) for all i, j
        self.assertEqual(_edges(tin), {(i, j // 2) for i in range(3) for j in range(4)})

    def test_p_third_zero_no_third_arms(self):
        sim = Simulator(dt=0.1 * u.ms)
        pre = sim.create(aeif_cond_alpha_astro, 3)
        post = sim.create(aeif_cond_alpha_astro, 4)
        astro = sim.create(astrocyte_lr_1994, 2)
        prim, tin, tout = sim.tripartite_connect(
            pre, post, astro, conn_spec=all_to_all,
            third_factor_conn_spec=third_factor_bernoulli_with_pool(
                p=0.0, pool_size=1, pool_type='block'),
            syn_specs=_syn_specs(), seed=0)
        self.assertIsInstance(prim, EventProjection)
        self.assertIsNone(tin)
        self.assertIsNone(tout)
        # no sic_connection registered when no edge is paired
        models = [m for (_, _, m, _) in sim._connections]
        self.assertNotIn('sic_connection', models)

    def test_multisegment_views_rejected(self):
        sim = Simulator(dt=0.1 * u.ms)
        a = sim.create(aeif_cond_alpha_astro, 3)
        b = sim.create(aeif_cond_alpha_astro, 4)
        astro = sim.create(astrocyte_lr_1994, 2)
        with self.assertRaises(NotImplementedError):
            sim.tripartite_connect(
                a + b, b, astro, conn_spec=all_to_all,
                third_factor_conn_spec=third_factor_bernoulli_with_pool(
                    p=1.0, pool_size=1, pool_type='block'),
                syn_specs=_syn_specs(), seed=0)

    def test_generator_views_rejected(self):
        # A deferred generator (e.g. poisson_generator) is not a created population
        # and cannot be a tripartite role; reject with a clear message.
        sim = Simulator(dt=0.1 * u.ms)
        gen = sim.create(poisson_generator, rate=100.0 * u.Hz)
        post = sim.create(aeif_cond_alpha_astro, 4)
        astro = sim.create(astrocyte_lr_1994, 2)
        with self.assertRaises(NotImplementedError):
            sim.tripartite_connect(
                gen, post, astro, conn_spec=all_to_all,
                third_factor_conn_spec=third_factor_bernoulli_with_pool(
                    p=1.0, pool_size=1, pool_type='block'),
                syn_specs=_syn_specs(), seed=0)

    def test_seeded_reproducible_edges(self):
        def build():
            sim = Simulator(dt=0.1 * u.ms)
            pre = sim.create(aeif_cond_alpha_astro, 4)
            post = sim.create(aeif_cond_alpha_astro, 6)
            astro = sim.create(astrocyte_lr_1994, 6)
            return sim.tripartite_connect(
                pre, post, astro, conn_spec=pairwise_bernoulli(0.6),
                third_factor_conn_spec=third_factor_bernoulli_with_pool(
                    p=0.7, pool_size=2, pool_type='random'),
                syn_specs=_syn_specs(), seed=123)
        p1, i1, o1 = build()
        p2, i2, o2 = build()
        self.assertEqual(_edges(p1), _edges(p2))
        self.assertEqual(_edges(o1), _edges(o2))


class TestTripartiteForLoopLowering(unittest.TestCase):
    """The assembled tripartite net runs under the Simulator's for_loop (cluster-12)."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_lowers_and_returns_stable_traces(self):
        brainstate.random.seed(0)
        sim = Simulator(dt=0.1 * u.ms)
        pre = sim.create(aeif_cond_alpha_astro, 3, params={'I_e': 1000.0 * u.pA})
        post = sim.create(aeif_cond_alpha_astro, 4, params={'I_e': 1000.0 * u.pA})
        astro = sim.create(astrocyte_lr_1994, 2, params={'delta_IP3': 1.0})
        sim.tripartite_connect(
            pre, post, astro, conn_spec=all_to_all,
            third_factor_conn_spec=third_factor_bernoulli_with_pool(
                p=1.0, pool_size=1, pool_type='block'),
            syn_specs=_syn_specs(w_a2n=10.0), seed=0)
        mm_a = sim.create(multimeter, record_from=['IP3', 'Ca'])
        mm_p = sim.create(multimeter, record_from=['I_SIC'])
        sim.connect(mm_a, astro)
        sim.connect(mm_p, post)
        T = 50.0
        res = sim.simulate(T * u.ms)
        n = int(round(T / 0.1))
        ip3 = np.asarray(u.get_mantissa(res.trace(mm_a, 'IP3')))
        isic = np.asarray(u.get_mantissa(res.trace(mm_p, 'I_SIC')))
        self.assertEqual(ip3.shape, (n, 2))
        self.assertEqual(isic.shape, (n, 4))
        self.assertTrue(np.all(np.isfinite(ip3)) and np.all(np.isfinite(isic)))


if __name__ == '__main__':
    unittest.main()
