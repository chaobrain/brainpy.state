# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest/one_neuron.py``.

NEST's ``one_neuron`` drives an ``iaf_psc_alpha`` with a constant ``I_e`` and
records ``V_m`` with a ``voltmeter``. ``I_e = 376 pA`` lands just above rheobase
(V_inf ~ -54.96 mV vs V_th = -55 mV), so the membrane charges for ~59 ms, spikes,
and repeats. We compare the deterministic sub-threshold charge against live NEST
(category B, one-step recorder alignment) and confirm the drive is suprathreshold.
"""
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import CAT_B_ALIGNED


def _nest_vm(I_e, simtime):
    nest.ResetKernel()
    nest.resolution = 0.1
    n = nest.Create("iaf_psc_alpha", params={"I_e": I_e})
    vm = nest.Create("voltmeter", params={"interval": 0.1})
    nest.Connect(vm, n)
    nest.Simulate(simtime)
    return np.asarray(nest.GetStatus(vm, "events")[0]["V_m"])


@requires_nest
class TestOneNeuronParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_subthreshold_charge_matches_nest(self):
        from examples.nest.one_neuron import build
        sim, vm, _neuron, _simtime = build(I_e=376.0, simtime=80.0)
        res = sim.simulate(80.0 * u.ms)
        bp_v = np.asarray(u.get_mantissa(res.trace(vm, "V_m") / u.mV)).reshape(-1)
        nest_v = _nest_vm(376.0, 80.0)
        # First spike is ~59 ms; compare the clean sub-threshold charge before it.
        w = 500
        compare_trace(nest_v[:w], bp_v[:w], tol=CAT_B_ALIGNED,
                      metric="one_neuron V_m charge").assert_()

    def test_drive_is_suprathreshold(self):
        from examples.nest.one_neuron import build
        sim, vm, _neuron, _simtime = build(I_e=376.0, simtime=1000.0)
        res = sim.simulate(1000.0 * u.ms)
        bp_v = np.asarray(u.get_mantissa(res.trace(vm, "V_m") / u.mV)).reshape(-1)
        # A reset (sharp drop toward V_reset) proves at least one spike fired.
        self.assertTrue(bool(np.min(np.diff(bp_v)) < -10.0),
                        "no spike/reset seen: I_e=376 pA should be suprathreshold")


if __name__ == "__main__":
    unittest.main()
