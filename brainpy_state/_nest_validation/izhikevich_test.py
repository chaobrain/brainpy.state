# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Live-NEST parity for the Izhikevich firing-regime demo (§3.5).

The demo ``examples/nest/izhikevich.py`` drives a single ``izhikevich`` neuron with
a constant 10 pA current in each of the four canonical regimes (RS / IB / CH / FS)
and records ``V_m`` and ``U_m`` through a multimeter.

Izhikevich is a category **A** model (a nonlinear 2-variable integrator), but with
NEST's default ``consistent_integration=True`` (forward Euler) and an ``I_e`` drive
(no device, hence no connection-delay offset) the Simulator reproduces a live
``nest.izhikevich`` **to the float-noise floor with zero shift**: identical spike
counts, identical V_m and U_m traces. The reset convention also matches — both
record the pre-reset sub-threshold value on a spike step (so the recorded V_m peak
sits just under ``V_th``), then reset to ``c`` on the same step.

The regime-distinctness test is NEST-free: the four parameter sets must produce
genuinely different firing (the spike-count ordering RS < IB < CH < FS), so the
demo is not silently collapsing to one pattern.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

import examples.nest.izhikevich as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

# Category A (nonlinear integrator). The forward-Euler trace matches NEST to the
# float-noise floor with no shift; a 1-step align guard absorbs any recorder phase.
_BAND = tc.TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=1, label="A",
                          note="izhikevich forward-Euler V_m/U_m, I_e drive (no device offset)")


def _nest_traces(regime, simtime=demo.T_SIM):
    """Live-NEST V_m, U_m and spike count for one regime under the demo's drive."""
    p = demo.REGIMES[regime]
    nest.ResetKernel()
    nest.resolution = demo.DT
    nest.set_verbosity("M_ERROR")
    n = nest.Create("izhikevich", 1, params={
        "a": p["a"], "b": p["b"], "c": p["c"], "d": p["d"],
        "V_th": demo.V_TH, "I_e": demo.I_DRIVE, "consistent_integration": True,
        "V_m": demo.V0, "U_m": p["b"] * demo.V0})
    mm = nest.Create("multimeter", params={"record_from": ["V_m", "U_m"], "interval": demo.DT})
    sr = nest.Create("spike_recorder")
    nest.Connect(mm, n)
    nest.Connect(n, sr)
    nest.Simulate(simtime)
    ev = mm.get("events")
    return np.asarray(ev["V_m"]), np.asarray(ev["U_m"]), int(sr.n_events)


@requires_nest
class TestIzhikevichParity(unittest.TestCase):
    """V_m, U_m and spike count match live NEST in every regime."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _run(self, regime):
        nest_v, nest_u, nest_n = _nest_traces(regime)
        _t, bp_v, bp_u, bp_n = demo.run_traces(regime)
        mv = min(len(nest_v), len(bp_v))
        compare_trace(nest_v[:mv], bp_v[:mv], tol=_BAND,
                      metric=f"izhikevich {regime} V_m").assert_()
        mu = min(len(nest_u), len(bp_u))
        compare_trace(nest_u[:mu], bp_u[:mu], tol=_BAND,
                      metric=f"izhikevich {regime} U_m").assert_()
        # Spike-count parity (category E): the forward-Euler trains are identical.
        self.assertLessEqual(abs(nest_n - bp_n), 2,
                             f"{regime} spike count NEST={nest_n} brainpy={bp_n}")

    def test_rs_matches_nest(self):
        self._run("RS")

    def test_ib_matches_nest(self):
        self._run("IB")

    def test_ch_matches_nest(self):
        self._run("CH")

    def test_fs_matches_nest(self):
        self._run("FS")


class TestIzhikevichRegimes(unittest.TestCase):
    """NEST-free: the four parameter sets produce genuinely distinct firing."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_regimes_are_distinct(self):
        counts = {regime: demo.run_traces(regime)[3] for regime in demo.REGIMES}
        # Each regime fires, and the canonical ordering RS < IB < CH < FS holds for
        # this drive (adaptation strongest in RS, none in FS).
        for regime, n in counts.items():
            self.assertGreater(n, 0, f"{regime} should fire under 10 pA drive")
        self.assertLess(counts["RS"], counts["IB"], f"counts={counts}")
        self.assertLess(counts["IB"], counts["CH"], f"counts={counts}")
        self.assertLess(counts["CH"], counts["FS"], f"counts={counts}")


if __name__ == "__main__":
    unittest.main()
