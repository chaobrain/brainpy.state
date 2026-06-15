# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest/hh_psc_alpha.py``.

NEST's ``hh_psc_alpha`` is a Hodgkin–Huxley neuron integrated with an adaptive
RKF45 step (category A). Two complementary checks:

* **Subthreshold trace** — a sub-rheobase bias current (``I_e=200 pA``) drives a
  smooth, non-chaotic depolarization; the ``V_m`` and gating (``Act_m``/``Inact_h``/
  ``Act_n``) traces must match live NEST per-sample (category A, with a one-step
  multimeter-offset alignment: brainpy samples the initial state at ``t=0`` while
  NEST's first sample is at ``t=dt``).
* **F–I curve** — suprathreshold spiking is phase-sensitive (the two RKF45 streams
  decorrelate after the first spike), so the *rate* over a 1 s window is compared
  as a deterministic fixed point within 5 % (category C-rate).
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import (
    TraceTolerance, CAT_C_RATE)
import brainunit as u

# Category A with a one-step recorder-offset alignment (HH RKF45 trace).
CAT_A_ALIGNED = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=1, label="A",
                               note="HH RKF45 V_m/gating with one-step recorder alignment")
# Gating variables are dimensionless (0..1); reuse the same numbers, bare float atol.
CAT_A_GATE = TraceTolerance(1e-3, 1e-3, align_steps=1, label="A",
                            note="HH gating variable trace, one-step aligned")

_NEST_PARAMS = {
    "E_L": -54.402, "C_m": 100.0, "g_Na": 12000.0, "g_K": 3600.0, "g_L": 30.0,
    "E_Na": 50.0, "E_K": -77.0, "t_ref": 2.0, "tau_syn_ex": 0.2, "tau_syn_in": 2.0,
    "V_m": -65.0,
}


def _nest_traces(I_e, dt=0.1, simtime=100.0):
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.resolution = dt
    neu = nest.Create("hh_psc_alpha", params={**_NEST_PARAMS, "I_e": I_e})
    mm = nest.Create("multimeter", params={
        "record_from": ["V_m", "Act_m", "Inact_h", "Act_n"], "interval": dt})
    sr = nest.Create("spike_recorder")
    nest.Connect(mm, neu)
    nest.Connect(neu, sr)
    nest.Simulate(simtime)
    ev = mm.get("events")
    return {
        "V_m": np.asarray(ev["V_m"]), "Act_m": np.asarray(ev["Act_m"]),
        "Inact_h": np.asarray(ev["Inact_h"]), "Act_n": np.asarray(ev["Act_n"]),
        "n_spikes": int(sr.n_events),
    }


def _nest_rate(I_e, dt=0.1, simtime=1000.0, warmup=200.0):
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.resolution = dt
    neu = nest.Create("hh_psc_alpha", params={**_NEST_PARAMS, "I_e": I_e})
    sr = nest.Create("spike_recorder")
    nest.Connect(neu, sr)
    nest.Simulate(simtime)
    times = np.asarray(sr.get("events")["times"])
    return int(np.sum(times >= warmup)) * 1000.0 / (simtime - warmup)


@requires_nest
class TestHHParity(unittest.TestCase):
    def test_subthreshold_trace_matches_nest(self):
        from examples.nest.hh_psc_alpha import run_traces
        I_e, dt, simtime = 200.0, 0.1, 100.0
        bp = run_traces(I_e=I_e, dt=dt, simtime=simtime)
        ns = _nest_traces(I_e, dt=dt, simtime=simtime)
        self.assertEqual(bp["n_spikes"], 0)
        self.assertEqual(ns["n_spikes"], 0)
        # brainpy includes the t=0 initial sample; drop it so both start at t=dt.
        compare_trace(ns["V_m"], bp["V_m"][1:], tol=CAT_A_ALIGNED, metric="hh V_m").assert_()
        for g in ("Act_m", "Inact_h", "Act_n"):
            compare_trace(ns[g], bp[g][1:], tol=CAT_A_GATE, metric=f"hh {g}").assert_()

    def test_fi_curve_matches_nest(self):
        from examples.nest.hh_psc_alpha import fi_curve
        # Amps in the repetitive-firing regime (below ~700 pA the neuron fires only
        # onset spikes then falls quiescent, so the post-warmup rate is 0 on both
        # sides — not a meaningful rate comparison).
        amps = [1000.0, 1500.0, 2000.0]
        dt, simtime, warmup = 0.1, 1000.0, 200.0
        bp = fi_curve(amps, dt=dt, simtime=simtime, warmup=warmup)
        for amp, r_bp in zip(amps, bp):
            ns = _nest_rate(amp, dt=dt, simtime=simtime, warmup=warmup)
            with self.subTest(I_e=amp):
                self.assertGreater(ns, 0.0)
                compare_trace(ns, r_bp, tol=CAT_C_RATE,
                              metric=f"hh F-I rate I_e={amp}").assert_()


if __name__ == "__main__":
    unittest.main()
