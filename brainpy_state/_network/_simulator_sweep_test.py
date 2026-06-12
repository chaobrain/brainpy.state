# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Sweep / re-run ergonomics (Extension C): rebuild-per-trial + per-trial readouts.

The parameter-sweep demos (``balancedneuron``, ``if_curve``, ``testiaf``) build a
fresh network per trial and read a scalar back. This verifies the building blocks
that pattern relies on:

* ``res.rate(rec)`` / ``res.n_events(rec)`` are correct for a single recorded
  neuron (a ``spike_recorder`` tap),
* sweeping a device parameter across fresh ``Simulator``s changes the readout
  monotonically (the bisection/F-I-curve premise), and
* re-``simulate()``-ing one object fully resets state (deterministic repeat).
"""
import unittest

import braintools
import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import (iaf_psc_alpha, voltmeter, spike_recorder, dc_generator)
from brainpy_state._network import Simulator

NPAR = dict(C_m=250. * u.pF, tau_m=10. * u.ms, tau_syn_ex=2. * u.ms,
            tau_syn_in=2. * u.ms, t_ref=2. * u.ms, E_L=-70. * u.mV,
            V_reset=-70. * u.mV, V_th=-55. * u.mV,
            V_initializer=braintools.init.Constant(-70. * u.mV))


def _vm(res, rec):
    return np.asarray(u.get_mantissa(res.trace(rec, 'V_m') / u.mV)).reshape(-1)


def _rate_for_amplitude(amp_pA, T_ms=500.):
    """Rebuild-per-trial: fresh net driven by a dc current, return output rate (Hz)."""
    sim = Simulator(dt=0.1 * u.ms)
    neu = sim.create(iaf_psc_alpha, 1, params=NPAR)
    dc = sim.create(dc_generator, amplitude=amp_pA * u.pA)
    rec = sim.create(spike_recorder)
    sim.connect(dc, neu)
    sim.connect(neu, rec)
    return sim.simulate(T_ms * u.ms).rate(rec)


class TestSweepRerun(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_rate_and_n_events_consistent_single_neuron(self):
        # I_e well above rheobase (~375 pA) -> a regular spike train.
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params={**NPAR, 'I_e': 500. * u.pA})
        rec = sim.create(spike_recorder)
        sim.connect(neu, rec)
        res = sim.simulate(1000. * u.ms)
        n = res.n_events(rec)
        rate = res.rate(rec)
        self.assertGreater(n, 0)
        # rate == n_events / (1 neuron * 1 s)
        self.assertAlmostEqual(rate, n / 1.0, delta=1e-6)

    def test_sweep_is_monotonic_across_fresh_sims(self):
        rates = [_rate_for_amplitude(a) for a in (400., 550., 800.)]
        self.assertTrue(rates[0] < rates[1] < rates[2],
                        f"output rate not monotonic in drive: {rates}")
        self.assertGreater(rates[0], 0.)   # 400 pA > rheobase -> fires

    def test_resimulate_resets_state(self):
        # A deterministic (dc-driven) sim simulated twice must give identical
        # traces -- simulate() re-inits all state each run.
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params={**NPAR, 'I_e': 200. * u.pA})
        vm = sim.create(voltmeter)
        sim.connect(vm, neu)
        v1 = _vm(sim.simulate(30. * u.ms), vm)
        v2 = _vm(sim.simulate(30. * u.ms), vm)
        npt = float(np.max(np.abs(v1 - v2)))
        self.assertLess(npt, 1e-12, f"re-simulate not deterministic: {npt}")


if __name__ == '__main__':
    unittest.main()
