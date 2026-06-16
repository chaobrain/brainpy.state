# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Live-NEST parity for the Brette & Gerstner Fig 2C AdEx demo (§3.10).

The demo ``examples/nest_like/brette_gerstner_fig_2c.py`` drives one
``aeif_cond_alpha`` with a 500 pA sub-threshold pulse ``[0, 200) ms`` followed
by an 800 pA spiking pulse ``[500, 1000) ms`` and reproduces the
spike-frequency-adaptation figure.

Parity posture (the established RKF45-with-real-spikes recipe, cf. ``hh_psc_alpha``
/ ``glif_cond_neuron``):

* **Sub-threshold V_m trace (category A).** The two adaptive integrators agree to
  the float-noise floor *only* away from spikes — after the first spike the two
  RKF45 streams decorrelate, and at a current-step edge the recorded sample
  carries a one-step current-application phase difference (brainpy applies the
  DC at the generator window start; NEST's ``dc_generator`` defers it one
  ``min_delay`` step). So the trace is compared on the **clean sub-threshold
  charging window** ``[2, 198] ms`` (the 500 pA pulse response: no spikes, no
  current edges), where it matches NEST to ``< 1e-3 mV``. The NEST DC is
  connected at ``delay = dt`` to equalise the current onset (the 1.0 ms default
  is NEST's min-delay connection convention, not AdEx dynamics).
* **Spike pattern (category E).** The full-run spike *count* and the first-spike
  *step* match NEST within ``|dN| <= 2`` / ``|dstep| <= 1``.

The adaptation *law* is asserted NEST-free: the inter-spike intervals lengthen
(non-decreasing, last >> first) with adaptation on, and become regular when it is
switched off (``a = b = 0`` -> plain exponential I&F).
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

import examples.nest_like.brette_gerstner_fig_2c as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

# Category A, sub-threshold window, two-step align guard (recorder phase + the
# one-step current-onset offset). The 500 pA charging window matches to ~1e-6 mV.
CAT_A_SUB = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=2, label="A",
                           note="AdEx sub-threshold V_m, current-edge/spike-free window")

#: Sub-threshold comparison window (ms): the 500 pA pulse response, edges excluded.
WIN_MS = (2.0, 198.0)


def _nest_fig2c(simtime=demo.T_SIM, dt=demo.DT, a=demo.A_NS, b=demo.B_PA):
    """Live-NEST V_m trace and spike steps under the Fig 2C protocol.

    Every AdEx parameter is pinned explicitly (cluster-02 discipline); the DC is
    connected at ``delay = dt`` so the current onset matches the brainpy demo.
    """
    nest.ResetKernel()
    nest.resolution = dt
    nest.set_verbosity("M_ERROR")
    neuron = nest.Create("aeif_cond_alpha", params={
        "C_m": 281.0, "g_L": 30.0, "E_L": -70.6, "V_th": -50.4, "V_peak": 0.0,
        "V_reset": -60.0, "Delta_T": 2.0, "tau_w": 144.0, "a": a, "b": b,
        "t_ref": 0.0, "tau_syn_ex": 0.2, "tau_syn_in": 2.0, "E_ex": 0.0,
        "E_in": -85.0, "I_e": 0.0, "V_m": -70.6})
    dc = nest.Create("dc_generator", len(demo.DC_PULSES))
    dc.set(amplitude=[p[0] for p in demo.DC_PULSES],
           start=[p[1] for p in demo.DC_PULSES],
           stop=[p[2] for p in demo.DC_PULSES])
    nest.Connect(dc, neuron, "all_to_all", syn_spec={"delay": dt})
    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": dt})
    sr = nest.Create("spike_recorder")
    nest.Connect(mm, neuron)
    nest.Connect(neuron, sr)
    nest.Simulate(simtime)
    v = np.asarray(mm.get("events")["V_m"])
    spike_steps = np.round(np.asarray(sr.get("events")["times"]) / dt).astype(int)
    return v, spike_steps


@requires_nest
class TestFig2cParity(unittest.TestCase):
    """V_m sub-threshold trace and spike pattern match live NEST."""

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_subthreshold_window_matches_nest(self):
        nest_v, _ = _nest_fig2c()
        _t, bp_v, _spk = demo.run()
        i0, i1 = int(WIN_MS[0] / demo.DT), int(WIN_MS[1] / demo.DT)
        compare_trace(nest_v[i0:i1], bp_v[i0:i1], tol=CAT_A_SUB,
                      metric="fig2c sub-threshold V_m").assert_()

    def test_spike_pattern_matches_nest(self):
        nest_v, nest_spk = _nest_fig2c()
        _t, _v, bp_spk = demo.run()
        # Category E: count within +/-2, first spike within +/-1 step.
        self.assertLessEqual(abs(len(nest_spk) - len(bp_spk)), 2,
                             f"spike count NEST={len(nest_spk)} bp={len(bp_spk)}")
        self.assertTrue(len(nest_spk) and len(bp_spk), "both sides must spike")
        self.assertLessEqual(abs(int(nest_spk[0]) - int(bp_spk[0])), 1,
                             f"first spike NEST={nest_spk[0]} bp={bp_spk[0]}")


class TestFig2cLaw(unittest.TestCase):
    """NEST-free: the adaptation law and the for_loop-lowered Simulator run."""

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_adaptation_lengthens_isis(self):
        # The whole rollout lowers through one Simulator for_loop (no Python loop).
        _t, v, spikes = demo.run()
        self.assertGreaterEqual(len(spikes), 3, "the 800 pA pulse must drive a train")
        isis = np.diff(spikes) * demo.DT
        # ISIs lengthen then saturate: non-decreasing, and the last clearly
        # exceeds the first (spike-frequency adaptation).
        self.assertTrue(np.all(np.diff(isis) >= -1e-9), f"ISIs not monotone: {isis}")
        self.assertGreater(isis[-1], isis[0] * 1.5, f"no adaptation: ISIs={isis}")
        # The 500 pA pulse stays sub-threshold (first spike is in the 800 pA epoch).
        self.assertGreater(spikes[0] * demo.DT, 500.0, "500 pA pulse should not spike")

    def test_adaptation_off_is_regular(self):
        # a = b = 0 -> plain exponential I&F: ISIs become (near) constant.
        _t, _v, spikes = demo.run(a=0.0, b=0.0)
        self.assertGreaterEqual(len(spikes), 4, "exp-IF should fire repetitively")
        isis = np.diff(spikes) * demo.DT
        spread = (isis.max() - isis.min()) / isis.mean()
        self.assertLess(spread, 0.05, f"exp-IF ISIs should be regular: {isis}")


if __name__ == "__main__":
    unittest.main()
