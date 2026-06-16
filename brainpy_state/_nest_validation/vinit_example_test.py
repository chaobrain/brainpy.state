# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/vinit_example.py``.

NEST's ``vinit_example`` runs the ``iaf_cond_exp_sfa_rr`` neuron with no input
from several initial membrane voltages and records the passive relaxation toward
E_L. We rebuild per initial voltage (sweep / rebuild-per-trial) and compare each
``V_m(t)`` relaxation against live NEST. ``iaf_cond_exp_sfa_rr`` is an adaptive
(RKF45/GSL) conductance neuron, so the bound is category A/B with a one-step
recorder alignment.
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

VINITS = (-100.0, -90.0, -80.0, -70.0, -60.0)
SIMTIME = 75.0


def _nest_vm(vinit, simtime):
    nest.ResetKernel()
    nest.resolution = 0.1
    cbn = nest.Create("iaf_cond_exp_sfa_rr")
    cbn.V_m = vinit
    vm = nest.Create("voltmeter", params={"interval": 0.1})
    nest.Connect(vm, cbn)
    nest.Simulate(simtime)
    return np.asarray(nest.GetStatus(vm, "events")[0]["V_m"])


@requires_nest
class TestVinitExampleParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_relaxation_matches_nest_for_each_vinit(self):
        from examples.nest_like.vinit_example import build
        worst = 0.0
        for vinit in VINITS:
            with self.subTest(vinit=vinit):
                sim, vm, _neuron, _t = build(vinit=vinit, simtime=SIMTIME)
                res = sim.simulate(SIMTIME * u.ms)
                bp_v = np.asarray(u.get_mantissa(res.trace(vm, "V_m") / u.mV)).reshape(-1)
                nest_v = _nest_vm(vinit, SIMTIME)
                r = compare_trace(nest_v, bp_v, tol=CAT_B_ALIGNED,
                                  metric=f"vinit={vinit} V_m relax")
                worst = max(worst, r.error)
                r.assert_()
        print(f"\n[vinit_example] worst max|Δ| across all vinit = {worst:.3e} mV")


if __name__ == "__main__":
    unittest.main()
