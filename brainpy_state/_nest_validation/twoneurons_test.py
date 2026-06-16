# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/twoneurons.py``.

NEST's ``twoneurons`` drives ``neuron_1`` (``iaf_psc_alpha``, ``I_e = 376 pA``),
connects it to ``neuron_2`` through a static synapse (``w = 20 pA``, ``d = 1 ms``),
and records both membrane potentials. Both traces are deterministic: we compare
``neuron_1``'s sub-threshold charge and ``neuron_2``'s evoked PSP train against
live NEST (category B, one-step recorder alignment).
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import CAT_B_ALIGNED

SIMTIME = 150.0


def _nest_traces(simtime):
    nest.ResetKernel()
    nest.resolution = 0.1
    n1 = nest.Create("iaf_psc_alpha", params={"I_e": 376.0})
    n2 = nest.Create("iaf_psc_alpha")
    vm1 = nest.Create("voltmeter", params={"interval": 0.1})
    vm2 = nest.Create("voltmeter", params={"interval": 0.1})
    nest.Connect(n1, n2, syn_spec={"weight": 20.0, "delay": 1.0})
    nest.Connect(vm1, n1)
    nest.Connect(vm2, n2)
    nest.Simulate(simtime)
    v1 = np.asarray(nest.GetStatus(vm1, "events")[0]["V_m"])
    v2 = np.asarray(nest.GetStatus(vm2, "events")[0]["V_m"])
    return v1, v2


@requires_nest
class TestTwoNeuronsParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_both_traces_match_nest(self):
        from examples.nest_like.twoneurons import build
        sim, vm1, vm2, _n1, _n2, _t = build(simtime=SIMTIME)
        res = sim.simulate(SIMTIME * u.ms)
        bp_v1 = np.asarray(u.get_mantissa(res.trace(vm1, "V_m") / u.mV)).reshape(-1)
        bp_v2 = np.asarray(u.get_mantissa(res.trace(vm2, "V_m") / u.mV)).reshape(-1)
        nest_v1, nest_v2 = _nest_traces(SIMTIME)

        # neuron_1: clean sub-threshold charge before its first spike (~59 ms).
        compare_trace(nest_v1[:500], bp_v1[:500], tol=CAT_B_ALIGNED,
                      metric="neuron_1 V_m charge").assert_()
        # neuron_2: the full evoked-PSP train (stays sub-threshold).
        compare_trace(nest_v2, bp_v2, tol=CAT_B_ALIGNED,
                      metric="neuron_2 V_m PSP").assert_()
        # The PSP must be a real, positive deflection (not a trivially-flat match).
        self.assertGreater(float(np.max(bp_v2) - np.min(bp_v2)), 0.05)


if __name__ == "__main__":
    unittest.main()
