# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Live-NEST parity for the two ``iaf_tum_2000`` short-term-plasticity demos.

The §3.5 demos ``examples/nest_like/iaf_tum_2000_short_term_depression.py`` and
``…_facilitation.py`` port NEST's Figure-1A/1B reproduction: two ``iaf_tum_2000``
neurons, the presynaptic one driven by a ``dc_generator`` into a regular ~20 Hz
train, a ``static_synapse`` on ``receptor_type=1`` carrying the *graded released
efficacy* ``weight * (u * x)`` to the post, whose sub-threshold V_m is recorded.
A large ``U`` with ``tau_fac = 0`` depresses the successive EPSPs; a small ``U``
with a non-zero ``tau_fac`` facilitates them.

Parity result
-------------
The post V_m matches live NEST **to machine precision** in both regimes, modulo a
single *constant* integer-step delivery/recorder offset (≈8 steps at dt = 0.1 ms).
The offset is a fixed pipeline-latency convention difference, not drift:

* NEST's ``dc_generator`` carries the kernel's **default 1.0 ms connection delay**
  (10 steps) onto the presynaptic neuron, so NEST's whole train is delivered later
  than the Simulator's current-injector path; our presynaptic-emission seam in turn
  leads by ~2 steps (the emit-holder is captured the same step the spike is
  detected). The net is a constant ~8-step lag of the NEST trace, identical in both
  regimes and independent of the STP parameters.
* It is **not** a spike-time drift. NEST's ``iaf_tum_2000`` crosses threshold on the
  grid exactly as ours does (``cpp`` sets the spike time to ``origin + lag + 1`` — a
  grid step, no interpolation). The model *abuses* the ``SpikeEvent`` offset field to
  ship the synaptic efficacy ``delta_y_tsp = u*x`` to the ``static_synapse``
  (``s *= e.get_offset()``); a plain ``spike_recorder`` then misreads that efficacy
  as a sub-step *time* offset, so NEST's reported "spike times" (e.g. 97.736 ms) are
  bogus — the real grid times (98.0 ms …) and the per-spike efficacies match ours
  exactly (both converge to 0.0569).

The comparison therefore uses category **B** (analytic exact propagator, atol
1e-3 mV) with a generous integer-step alignment search that absorbs the constant
offset — once aligned, the residual is at the float-noise floor. This mirrors the
cluster-07 ``RELAY_D`` / cluster-03 recorder-offset conventions, just with a larger
(but still constant) offset.

The regime-signature tests are **NEST-free**: they assert the depression train's
EPSP increments *decrease* (first increment is the largest) and the facilitation
train's *increase then saturate* (last increment is the largest) directly on the
Simulator output, so the qualitative physics is checked even without NEST.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

import examples.nest_like.iaf_tum_2000_short_term_depression as dep_demo
import examples.nest_like.iaf_tum_2000_short_term_facilitation as fac_demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

_MODULES = {"depression": dep_demo, "facilitation": fac_demo}

# Category B (analytic exact propagator). The post V_m matches NEST to the float
# noise floor; the generous align search absorbs the constant device-delay/recorder
# offset (~8 steps at dt=0.1 ms, identical in both regimes — see the module docstring).
_BAND = tc.TraceTolerance(1e-3 * u.mV, 1e-6, align_steps=12, label="B",
                          note="iaf_tum_2000 STP post V_m; constant device+recorder offset")


def _nest_post_vm(mod):
    """Live-NEST post V_m (mV) for one demo module, mirroring the NEST §3.5 script.

    Two ``iaf_tum_2000`` neurons (all STP parameters on the *presynaptic* model), a
    ``dc_generator`` into the pre, a ``static_synapse`` on ``receptor_type=1`` to the
    post, and a multimeter at the simulation resolution so the trace lines up sample
    for sample with the Simulator's per-step voltmeter.
    """
    nest.ResetKernel()
    nest.resolution = mod.DT
    nest.set_verbosity("M_ERROR")
    dc = nest.Create("dc_generator", 1, params={
        "amplitude": mod.DC_AMP, "start": mod.STIM_START, "stop": mod.STIM_END})
    nrns = nest.Create("iaf_tum_2000", 2, params={
        "C_m": mod.C_M, "tau_m": mod.TAU_M, "tau_syn_ex": mod.TAU_PSC,
        "tau_syn_in": mod.TAU_PSC, "V_th": mod.V_TH, "V_reset": mod.V_RESET,
        "E_L": mod.V_RESET, "V_m": mod.V_RESET, "t_ref": mod.T_REF, "U": mod.U,
        "tau_psc": mod.TAU_PSC, "tau_rec": mod.TAU_REC, "tau_fac": mod.TAU_FAC,
        "x": mod.X0, "u": mod.U0})
    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": mod.DT})
    nest.Connect(dc, nrns[0])
    nest.Connect(nrns[0], nrns[1], syn_spec={
        "synapse_model": "static_synapse", "weight": mod.WEIGHT,
        "delay": mod.DELAY, "receptor_type": 1})
    nest.Connect(mm, nrns[1])
    nest.Simulate(mod.T_SIM)
    return np.asarray(mm.get("events")["V_m"])


def _epsp_increments(t, v, mod):
    """Per-presynaptic-spike EPSP increments of a post V_m trace (regime signature).

    The pre fires a regular ~20 Hz train, so segment the stimulation window into
    ISI-wide (50 ms) bins and take, in each, the rise from the bin's start to its peak
    -- the EPSP increment for that spike. A depressing train's increments decrease
    (first is largest); a facilitating train's increase then saturate (last is
    largest). Returns a 1-D float array of increments (mV), one per spike.
    """
    isi = 1000.0 / 20.0
    out = []
    for k in range(20):
        a = int(round((mod.STIM_START + k * isi) / mod.DT))
        b = int(round((mod.STIM_START + (k + 1) * isi) / mod.DT))
        seg = v[a:b]
        if seg.size:
            out.append(float(seg.max() - seg[0]))
    return np.asarray(out)


@requires_nest
class TestIafTum2000StpParity(unittest.TestCase):
    """Post V_m matches live NEST to the float-noise floor in both STP regimes."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _run(self, regime):
        mod = _MODULES[regime]
        nest_v = _nest_post_vm(mod)
        _, bp_v = mod.run_traces()
        m = min(len(nest_v), len(bp_v))
        compare_trace(nest_v[:m], bp_v[:m], tol=_BAND,
                      metric=f"iaf_tum_2000 {regime} post V_m").assert_()

    def test_depression_post_vm_matches_nest(self):
        self._run("depression")

    def test_facilitation_post_vm_matches_nest(self):
        self._run("facilitation")

    def test_peak_vm_matches_nest(self):
        # A sanity cross-check independent of alignment: the maximum sub-threshold
        # excursion is a single scalar, so it must agree regardless of the constant
        # delivery offset (the depression peak is tiny, the facilitation peak large).
        for regime, mod in _MODULES.items():
            nest_v = _nest_post_vm(mod)
            _, bp_v = mod.run_traces()
            compare_trace(float(nest_v.max()), float(bp_v.max()), tol=_BAND,
                          metric=f"iaf_tum_2000 {regime} peak V_m").assert_()


class TestIafTum2000StpSignature(unittest.TestCase):
    """NEST-free: the Simulator output shows the depression / facilitation signature."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_depression_epsps_decrease(self):
        mod = _MODULES["depression"]
        t, v = mod.run_traces()
        incr = _epsp_increments(t, v, mod)
        self.assertGreater(incr.size, 5)
        # depression: the first EPSP is the largest, and the train shrinks.
        self.assertEqual(int(np.argmax(incr)), 0,
                         f"depression: largest EPSP should be the first ({incr[:4]})")
        self.assertGreater(incr[0], incr[-1] * 2.0,
                           f"depression: train should shrink markedly ({incr[0]} vs {incr[-1]})")

    def test_facilitation_epsps_increase(self):
        mod = _MODULES["facilitation"]
        t, v = mod.run_traces()
        incr = _epsp_increments(t, v, mod)
        self.assertGreater(incr.size, 5)
        # facilitation: the last EPSP is the largest (increase then saturate).
        self.assertEqual(int(np.argmax(incr)), incr.size - 1,
                         f"facilitation: largest EPSP should be the last ({incr[-4:]})")
        self.assertGreater(incr[-1], incr[0] * 2.0,
                           f"facilitation: train should grow markedly ({incr[0]} vs {incr[-1]})")

    def test_regimes_are_distinct(self):
        # The two demos must select genuinely different dynamics: depression peaks
        # early (front-loaded), facilitation peaks late (back-loaded).
        t_dep, v_dep = dep_demo.run_traces()
        t_fac, v_fac = fac_demo.run_traces()
        dep_argmax_t = t_dep[int(np.argmax(v_dep))]
        fac_argmax_t = t_fac[int(np.argmax(v_fac))]
        self.assertLess(dep_argmax_t, 300.0,
                        f"depression peak should be early, got {dep_argmax_t} ms")
        self.assertGreater(fac_argmax_t, 800.0,
                           f"facilitation peak should be late, got {fac_argmax_t} ms")


if __name__ == "__main__":
    unittest.main()
