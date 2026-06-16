# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Current-injecting devices on the Simulator: dc_/step_/noise_generator."""
import unittest

import braintools
import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state import (iaf_psc_alpha, voltmeter, dc_generator,
                           step_current_generator, noise_generator)
from brainpy_state._nest_network import Simulator

NPAR = dict(C_m=250. * u.pF, tau_m=10. * u.ms, tau_syn_ex=2. * u.ms,
            tau_syn_in=2. * u.ms, t_ref=2. * u.ms, E_L=-70. * u.mV,
            V_reset=-70. * u.mV, V_th=-55. * u.mV,
            V_initializer=braintools.init.Constant(-70. * u.mV))


def _vm(res, rec):
    return np.asarray(u.get_mantissa(res.trace(rec, 'V_m') / u.mV))


class TestCurrentInjection(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_dc_generator_charges_to_analytic_steady_state(self):
        # V_inf = E_L + I * tau_m / C_m = -70 + 200*10/250 = -62 mV.
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params=NPAR)
        dc = sim.create(dc_generator, amplitude=200. * u.pA)
        vm = sim.create(voltmeter)
        sim.connect(dc, neu)
        sim.connect(vm, neu)
        res = sim.simulate(300. * u.ms)
        v = _vm(res, vm).reshape(-1)
        self.assertAlmostEqual(v[-1], -62.0, delta=0.05)

    def test_dc_equals_I_e_with_one_step_buffer_delay(self):
        # A dc_generator injects through the neuron's current ring buffer (one-step
        # delay), whereas I_e is added directly; so dc[1:] must match I_e[:-1].
        sim_ie = Simulator(dt=0.1 * u.ms)
        neu_ie = sim_ie.create(iaf_psc_alpha, 1, params={**NPAR, 'I_e': 200. * u.pA})
        vm_ie = sim_ie.create(voltmeter)
        sim_ie.connect(vm_ie, neu_ie)
        v_ie = _vm(sim_ie.simulate(40. * u.ms), vm_ie).reshape(-1)

        sim_dc = Simulator(dt=0.1 * u.ms)
        neu_dc = sim_dc.create(iaf_psc_alpha, 1, params=NPAR)
        dc = sim_dc.create(dc_generator, amplitude=200. * u.pA)
        vm_dc = sim_dc.create(voltmeter)
        sim_dc.connect(dc, neu_dc)
        sim_dc.connect(vm_dc, neu_dc)
        v_dc = _vm(sim_dc.simulate(40. * u.ms), vm_dc).reshape(-1)

        shifted = float(np.max(np.abs(v_dc[1:] - v_ie[:-1])))
        self.assertLess(shifted, 1e-9, f"dc vs I_e one-step buffer mismatch {shifted}")

    def test_connect_weight_scales_injected_current(self):
        # A current injector's connect ``weight`` scales the device current
        # (``cur * weight`` in the injection loop): dc(amplitude=A, weight=2)
        # must equal dc(amplitude=2A, weight=None). The weight is a dimensionless
        # multiplier here (the device already emits pA).
        def run(amplitude, weight):
            sim = Simulator(dt=0.1 * u.ms)
            neu = sim.create(iaf_psc_alpha, 1, params=NPAR)
            dc = sim.create(dc_generator, amplitude=amplitude * u.pA)
            vm = sim.create(voltmeter)
            sim.connect(dc, neu, weight=weight)
            sim.connect(vm, neu)
            return _vm(sim.simulate(50. * u.ms), vm).reshape(-1)

        v_weighted = run(100., 2.0)
        v_plain = run(200., None)
        d = float(np.max(np.abs(v_weighted - v_plain)))
        self.assertLess(d, 1e-9, f"weighted injector != scaled amplitude {d}")

    def test_noise_generator_drives_fluctuating_vm(self):
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params=NPAR)
        ng = sim.create(noise_generator, mean=0. * u.pA, std=150. * u.pA, seed=7)
        vm = sim.create(voltmeter)
        sim.connect(ng, neu)
        sim.connect(vm, neu)
        res = sim.simulate(200. * u.ms)
        v = _vm(res, vm).reshape(-1)
        self.assertGreater(float(np.std(v)), 0.1, "noise current produced no V_m fluctuation")

    def test_noise_fanout_is_independent_dc_fanout_is_identical(self):
        # noise -> 2 neurons: independent currents (different V_m).
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_alpha, 2, params=NPAR)
        ng = sim.create(noise_generator, mean=0. * u.pA, std=150. * u.pA, seed=3)
        vm = sim.create(voltmeter)
        sim.connect(ng, pop)
        sim.connect(vm, pop)
        v = _vm(sim.simulate(50. * u.ms), vm)
        self.assertFalse(np.allclose(v[:, 0], v[:, 1]), "noise fan-out not independent")

        # dc -> 2 neurons: deterministic, identical V_m.
        sim2 = Simulator(dt=0.1 * u.ms)
        pop2 = sim2.create(iaf_psc_alpha, 2, params=NPAR)
        dc = sim2.create(dc_generator, amplitude=120. * u.pA)
        vm2 = sim2.create(voltmeter)
        sim2.connect(dc, pop2)
        sim2.connect(vm2, pop2)
        v2 = _vm(sim2.simulate(50. * u.ms), vm2)
        self.assertTrue(np.allclose(v2[:, 0], v2[:, 1]), "dc fan-out not identical")

    def test_step_current_generator_changes_amplitude(self):
        # 0 pA until 10 ms, then 250 pA: V_m flat at rest, then charges.
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params=NPAR)
        step = sim.create(step_current_generator,
                          amplitude_times=[10.] * u.ms,
                          amplitude_values=[250.] * u.pA)
        vm = sim.create(voltmeter)
        sim.connect(step, neu)
        sim.connect(vm, neu)
        v = _vm(sim.simulate(60. * u.ms), vm).reshape(-1)
        self.assertAlmostEqual(v[50], -70.0, delta=1e-6)   # t=5 ms: still at rest
        self.assertGreater(v[-1], -66.0)                   # later: charged up


if __name__ == '__main__':
    unittest.main()
