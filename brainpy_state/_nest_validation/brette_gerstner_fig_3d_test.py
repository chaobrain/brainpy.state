# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Live-NEST parity for the Brette & Gerstner Fig 3D AdEx demo (§3.10).

The demo ``examples/nest/brette_gerstner_fig_3d.py`` hyperpolarises one
``aeif_cond_exp`` with an 800 pA inhibitory step over ``[0, 400) ms``; on release
the membrane rebounds through threshold and fires a short burst (post-inhibitory
rebound).

Parity posture (same RKF45-with-real-spikes recipe as ``brette_gerstner_fig_2c``):

* **Hyperpolarised plateau (category A).** The deterministic membrane trajectory
  while the inhibitory step is on, window ``[10, 395] ms`` (after the onset
  transient settles, before the rebound rise steepens), matches NEST to
  ``< 0.02 mV``. The NEST DC is connected at ``delay = dt`` to equalise the
  current onset.
* **Rebound (category E).** No spikes occur while the step is on; the first
  rebound spike *step* matches NEST within ``|dstep| <= 1`` and the burst counts
  agree within ``|dN| <= 2``.

The critical initial condition: the upstream demo sets ``E_L = -60 mV`` but leaves
``V_m`` at its ``-70.6 mV`` default — NEST does not move ``V_m`` to a freshly-set
``E_L``. The brainpy demo keeps the matching ``-70.6 mV`` default; this test pins
``V_m = -70.6`` explicitly on the NEST side too.
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

import examples.nest.brette_gerstner_fig_3d as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

CAT_A_SUB = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=2, label="A",
                           note="AdEx hyperpolarised plateau, transient-free window")

#: Plateau comparison window (ms): onset settled, before the rebound rise.
WIN_MS = (10.0, 395.0)


def _nest_fig3d(simtime=demo.T_SIM, dt=demo.DT, a=demo.A_NS, b=demo.B_PA,
                tau_w=demo.TAU_W_MS):
    """Live-NEST V_m trace and spike steps under the Fig 3D protocol.

    Every parameter is pinned explicitly, including ``V_m = -70.6 mV`` (the
    default NEST keeps even though ``E_L = -60 mV``); the DC is at ``delay = dt``.
    """
    nest.ResetKernel()
    nest.resolution = dt
    nest.set_verbosity("M_ERROR")
    neuron = nest.Create("aeif_cond_exp", params={
        "C_m": 281.0, "g_L": 30.0, "E_L": demo.E_L_MV, "V_th": -50.4,
        "V_peak": demo.V_PEAK_MV, "V_reset": -60.0, "Delta_T": 2.0, "tau_w": tau_w,
        "a": a, "b": b, "t_ref": 0.0, "tau_syn_ex": 0.2, "tau_syn_in": 2.0,
        "E_ex": 0.0, "E_in": -85.0, "I_e": 0.0, "V_m": -70.6})
    amplitude, start, stop = demo.DC_STEP
    dc = nest.Create("dc_generator")
    dc.set(amplitude=amplitude, start=start, stop=stop)
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
class TestFig3dParity(unittest.TestCase):
    """Hyperpolarised plateau and rebound spike match live NEST."""

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_plateau_matches_nest(self):
        nest_v, _ = _nest_fig3d()
        _t, bp_v, _spk = demo.run()
        i0, i1 = int(WIN_MS[0] / demo.DT), int(WIN_MS[1] / demo.DT)
        compare_trace(nest_v[i0:i1], bp_v[i0:i1], tol=CAT_A_SUB,
                      metric="fig3d hyperpolarised V_m").assert_()

    def test_rebound_matches_nest(self):
        _nv, nest_spk = _nest_fig3d()
        _t, _v, bp_spk = demo.run()
        self.assertLessEqual(abs(len(nest_spk) - len(bp_spk)), 2,
                             f"burst count NEST={len(nest_spk)} bp={len(bp_spk)}")
        self.assertTrue(len(nest_spk) and len(bp_spk), "both sides must rebound")
        self.assertLessEqual(abs(int(nest_spk[0]) - int(bp_spk[0])), 1,
                             f"first rebound NEST={nest_spk[0]} bp={bp_spk[0]}")


class TestFig3dLaw(unittest.TestCase):
    """NEST-free: the post-inhibitory-rebound law and the for_loop-lowered run."""

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_rebound_after_release(self):
        # The whole rollout lowers through one Simulator for_loop (no Python loop).
        _t, _v, spikes = demo.run()
        _amp, _start, stop = demo.DC_STEP
        during = spikes[spikes * demo.DT < stop]
        after = spikes[spikes * demo.DT >= stop]
        self.assertEqual(len(during), 0, "no spikes while the inhibitory step is on")
        self.assertGreaterEqual(len(after), 1, "the membrane must rebound after release")

    def test_membrane_hyperpolarises(self):
        # The 800 pA inhibitory step drives V_m well below the -70.6 mV start.
        _t, v, _spk = demo.run()
        self.assertLess(float(v.min()), -80.0, "inhibitory step should hyperpolarise")


if __name__ == "__main__":
    unittest.main()
