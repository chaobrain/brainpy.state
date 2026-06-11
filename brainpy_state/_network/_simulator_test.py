# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import braintools
import brainstate
import jax
import jax.numpy as jnp
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import iaf_psc_alpha, poisson_generator, spike_recorder
from brainpy_state._network import Simulator, fixed_indegree, all_to_all


class TestNetworkPublicAPI(unittest.TestCase):
    def test_public_names_importable(self):
        from brainpy_state._network import (
            Simulator, SimulationResult, NodeView,
            all_to_all, one_to_one, fixed_indegree,
        )
        self.assertTrue(callable(fixed_indegree))
        self.assertIsNotNone(Simulator)
        self.assertIsNotNone(SimulationResult)
        self.assertIsNotNone(NodeView)
        self.assertIsNotNone(all_to_all)
        self.assertIsNotNone(one_to_one)


class TestSimulatorEndToEnd(unittest.TestCase):
    def test_two_population_network_runs(self):
        sim = Simulator(dt=0.1 * u.ms)
        npar = dict(C_m=250. * u.pF, tau_m=20. * u.ms, tau_syn_ex=0.5 * u.ms,
                    tau_syn_in=0.5 * u.ms, t_ref=2. * u.ms, E_L=0. * u.mV,
                    V_reset=0. * u.mV, V_th=20. * u.mV,
                    V_initializer=braintools.init.Constant(0. * u.mV))
        ne = sim.create(iaf_psc_alpha, 40, params=npar)
        ni = sim.create(iaf_psc_alpha, 10, params=npar)
        noise = sim.create(poisson_generator, rate=20000. * u.Hz)
        esr = sim.create(spike_recorder)

        sim.connect(noise, ne, weight=20. * u.pA, delay=1.5 * u.ms, rule=all_to_all)
        sim.connect(noise, ni, weight=20. * u.pA, delay=1.5 * u.ms, rule=all_to_all)
        sim.connect(ne, ne + ni, weight=20. * u.pA, delay=1.5 * u.ms,
                    rule=fixed_indegree(4), allow_multapses=False, seed=1)
        sim.connect(ni, ne + ni, weight=-100. * u.pA, delay=1.5 * u.ms,
                    rule=fixed_indegree(1), allow_multapses=False, seed=2)
        sim.connect(ne[:20], esr)

        res = sim.simulate(50. * u.ms)
        spk = res.spikes(esr)
        self.assertEqual(spk.shape, (500, 20))         # 50 ms / 0.1 ms, N_rec=20
        self.assertFalse(bool(jnp.any(jnp.isnan(spk))))
        self.assertGreaterEqual(res.n_events(esr), 0)
        self.assertGreater(float(jnp.sum(spk > 0)), 0.0)  # poisson drive -> some spikes
