# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Analog (State-tap) recording on the Simulator: voltmeter / multimeter."""
import unittest

import braintools
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import iaf_psc_alpha, voltmeter, multimeter
from brainpy_state._network import Simulator

NPAR = dict(C_m=250. * u.pF, tau_m=10. * u.ms, tau_syn_ex=2. * u.ms,
            tau_syn_in=2. * u.ms, t_ref=2. * u.ms, E_L=-70. * u.mV,
            V_reset=-70. * u.mV, V_th=-55. * u.mV,
            V_initializer=braintools.init.Constant(-70. * u.mV))


def _manual_vm(I_e, T_ms, dt_ms=0.1):
    """Step one iaf_psc_alpha by hand; return its V_m trace (mV)."""
    neu = brainstate.nn.init_all_states(
        iaf_psc_alpha(1, I_e=I_e * u.pA, **NPAR))
    n = int(round(T_ms / dt_ms))

    @brainstate.transform.jit
    def step(t, i):
        with brainstate.environ.context(t=t, i=i):
            neu.update()
            return neu.V.value[0]

    return np.asarray([float(u.get_mantissa(step(k * dt_ms * u.ms, k) / u.mV))
                       for k in range(n)])


class TestAnalogRecording(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_voltmeter_trace_shape_and_subthreshold_charge(self):
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params={**NPAR, 'I_e': 200. * u.pA})
        vm = sim.create(voltmeter)
        sim.connect(vm, neu)            # reversed direction: recorder observes neuron
        res = sim.simulate(50. * u.ms)

        trace = res.trace(vm, 'V_m')
        self.assertEqual(trace.shape, (500, 1))
        v = np.asarray(u.get_mantissa(trace / u.mV)).reshape(-1)
        # Sub-threshold (V_inf = -70 + 200*10/250 = -62 mV): monotone rise, no spike.
        self.assertAlmostEqual(v[0], -70.0, delta=0.5)
        self.assertTrue(np.all(np.diff(v) >= -1e-9), "V_m not monotonically rising")
        self.assertLess(v[-1], -55.0)         # stays below threshold
        self.assertGreater(v[-1], -64.0)      # but charged well past rest

    def test_times_shape_and_units(self):
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params={**NPAR, 'I_e': 200. * u.pA})
        vm = sim.create(voltmeter)
        sim.connect(vm, neu)
        res = sim.simulate(50. * u.ms)
        times = res.times
        self.assertEqual(times.shape, (500,))
        t_ms = np.asarray(u.get_mantissa(times / u.ms))
        self.assertAlmostEqual(t_ms[0], 0.0, places=9)
        self.assertAlmostEqual(t_ms[1] - t_ms[0], 0.1, places=9)

    def test_trace_bitmatches_manual_stepping(self):
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params={**NPAR, 'I_e': 200. * u.pA})
        vm = sim.create(voltmeter)
        sim.connect(vm, neu)
        res = sim.simulate(30. * u.ms)
        sim_v = np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV)).reshape(-1)
        man_v = _manual_vm(200., 30.)
        npt = np.max(np.abs(sim_v - man_v))
        self.assertLess(npt, 1e-9, f"analog tap diverged from manual stepping by {npt}")

    def test_multi_neuron_trace_columns(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_alpha, 3, params={**NPAR, 'I_e': 200. * u.pA})
        vm = sim.create(voltmeter)
        sim.connect(vm, pop)
        res = sim.simulate(20. * u.ms)
        self.assertEqual(res.trace(vm, 'V_m').shape, (200, 3))

    def test_multimeter_multiple_recordables(self):
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params={**NPAR, 'I_e': 200. * u.pA})
        mm = sim.create(multimeter, record_from=['V_m', 'I_syn_ex'])
        sim.connect(mm, neu)
        res = sim.simulate(20. * u.ms)
        self.assertEqual(res.trace(mm, 'V_m').shape, (200, 1))
        isyn = res.trace(mm, 'I_syn_ex')
        self.assertEqual(isyn.shape, (200, 1))
        # No synaptic input -> I_syn_ex stays zero throughout.
        self.assertTrue(np.allclose(np.asarray(u.get_mantissa(isyn / u.pA)), 0.0))

    def test_iaf_psc_exp_isyn_recordables_via_alias(self):
        # iaf_psc_exp stores synaptic currents as lowercase i_syn_ex/i_syn_in;
        # recording the NEST 'I_syn_ex'/'I_syn_in' names must resolve via alias,
        # while iaf_psc_alpha's capital I_syn_ex still resolves directly.
        from brainpy_state import iaf_psc_exp, spike_generator
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_exp, 1)
        s_ex = sim.create(spike_generator, spike_times=np.array([5.0]) * u.ms)
        s_in = sim.create(spike_generator, spike_times=np.array([10.0]) * u.ms)
        mm = sim.create(multimeter, record_from=['V_m', 'I_syn_ex', 'I_syn_in'])
        sim.connect(s_ex, neu, weight=50. * u.pA)
        sim.connect(s_in, neu, weight=-50. * u.pA)
        sim.connect(mm, neu)
        res = sim.simulate(20. * u.ms)
        self.assertEqual(res.trace(mm, 'V_m').shape, (200, 1))
        iex = np.asarray(u.get_mantissa(res.trace(mm, 'I_syn_ex') / u.pA)).reshape(-1)
        iin = np.asarray(u.get_mantissa(res.trace(mm, 'I_syn_in') / u.pA)).reshape(-1)
        self.assertGreater(iex.max(), 40.0)        # ex spike drives i_syn_ex positive
        self.assertLess(iin.min(), -40.0)          # in spike drives i_syn_in negative
        self.assertTrue(np.all(iex >= -1e-9))      # excitatory current never negative

    def test_unknown_recordable_raises(self):
        sim = Simulator(dt=0.1 * u.ms)
        neu = sim.create(iaf_psc_alpha, 1, params={**NPAR, 'I_e': 200. * u.pA})
        vm = sim.create(voltmeter)
        sim.connect(vm, neu)
        res = sim.simulate(5. * u.ms)
        with self.assertRaises(KeyError):
            res.trace(vm, 'does_not_exist')


if __name__ == '__main__':
    unittest.main()
