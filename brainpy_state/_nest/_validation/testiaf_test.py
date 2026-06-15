# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest/testiaf.py``.

NEST's ``testiaf`` injects a constant ``I_e = 376 pA`` into an ``iaf_psc_alpha``
and records ``V_m`` and spikes at three resolutions ``dt in {0.1, 0.5, 1.0}`` ms
(the resolution-sweep / rebuild-per-trial pattern). For each ``dt`` we compare the
sub-threshold charge (category B, one-step alignment) and the spike count over
1 s (category E, ``|dN| <= 2``) against live NEST.
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

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import CAT_B_ALIGNED, CAT_E

DTS = (0.1, 0.5, 1.0)
SIMTIME = 1000.0


def _nest_run(dt, simtime):
    nest.ResetKernel()
    nest.local_num_threads = 1
    nest.resolution = dt
    neuron = nest.Create("iaf_psc_alpha", params={"I_e": 376.0})
    vm = nest.Create("voltmeter", params={"interval": dt})
    sr = nest.Create("spike_recorder")
    nest.Connect(vm, neuron)
    nest.Connect(neuron, sr)
    nest.Simulate(simtime)
    v = np.asarray(nest.GetStatus(vm, "events")[0]["V_m"])
    return v, int(sr.n_events)


@requires_nest
class TestTestiafParity(unittest.TestCase):
    def test_vm_and_spike_count_match_nest_across_dt(self):
        from examples.nest.testiaf import build
        for dt in DTS:
            with self.subTest(dt=dt):
                brainstate.environ.set(dt=dt * u.ms)
                sim, vm, sr, _neuron, _t = build(dt=dt, simtime=SIMTIME)
                res = sim.simulate(SIMTIME * u.ms)
                bp_v = np.asarray(u.get_mantissa(res.trace(vm, "V_m") / u.mV)).reshape(-1)
                bp_n = res.n_events(sr)
                nest_v, nest_n = _nest_run(dt, SIMTIME)

                w = int(round(50.0 / dt))    # ~50 ms pre-spike charge window
                compare_trace(nest_v[:w], bp_v[:w], tol=CAT_B_ALIGNED,
                              metric=f"testiaf V_m charge dt={dt}").assert_()
                self.assertLessEqual(
                    abs(nest_n - bp_n), CAT_E.max_count_diff,
                    f"dt={dt}: spike count NEST={nest_n} brainpy={bp_n}")


if __name__ == "__main__":
    unittest.main()
