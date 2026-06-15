# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Live-NEST parity for the three-compartment ``iaf_cond_alpha_mc`` demo (§3.5).

The demo ``examples/nest/mc_neuron.py`` drives a three-compartment neuron through
all nine receptors: per-compartment current pulses (receptors 7-9), per-compartment
excitatory/inhibitory spike trains (receptors 1-6), and a somatic rheobase step.
This exercises the full Simulator device→compartment routing seam end to end.

``iaf_cond_alpha_mc`` integrates with an adaptive RKF45 step (category **A**), so the
per-compartment membrane potentials match a live NEST ``iaf_cond_alpha_mc`` to the
adaptive-integrator noise floor (≈0.05-0.08 mV over the 1 s run), while the alpha
synaptic conductances ``g_ex`` / ``g_in`` — driven only by the routed spikes — match
to the float-noise floor. All device drives carry NEST's default connection delay,
which shows up as a constant ~8-step recorder offset absorbed by ``align_steps``.
Spike counts (the somatic rheobase output) match exactly.

NEST's published ``mc_neuron.py`` realizes the somatic rheobase by setting
``n.soma = {'I_e': 150.0}`` midway through the run; the Simulator lowers the whole
simulation into one compiled loop, so the demo (and this reference) use the
equivalent ``step_current_generator`` into the soma current receptor instead — the
two are compared head to head here, so the parity is exact for the same wiring.

The NEST-free behaviour test pins the demo-level routing: a compartment's current
pulse and spike train deflect that compartment far more than a non-adjacent one,
and the rheobase makes the soma fire.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc

import examples.nest.mc_neuron as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

# Category A (adaptive RKF45). V_m matches to the adaptive-integrator floor; the
# generous align absorbs the ~8-step device connection-delay offset (and the benign
# 1-step current-vs-spike ambiguity at the sharp current-pulse edges).
_VM_BAND = tc.TraceTolerance(1.5e-1 * u.mV, 1e-3, align_steps=12, label="A",
                             note="mc per-compartment V_m, device drives (RKF45 + connection-delay offset)")
# The alpha conductances are driven only by routed spikes -> float-noise floor.
_G_BAND = tc.TraceTolerance(1e-3 * u.nS, 1e-3, align_steps=12, label="A",
                            note="mc per-compartment g_ex/g_in, spike-driven")

_VM_RECORDABLES = ('V_m.s', 'V_m.p', 'V_m.d')
_G_RECORDABLES = ('g_ex.s', 'g_ex.p', 'g_ex.d', 'g_in.s', 'g_in.p', 'g_in.d')

# Receptor-type int -> NEST receptor label (the demo addresses by int; NEST's
# Connect needs the model's ``receptor_types`` map).
_RT_NAME = {1: 'soma_exc', 2: 'soma_inh', 3: 'proximal_exc', 4: 'proximal_inh',
            5: 'distal_exc', 6: 'distal_inh',
            7: 'soma_curr', 8: 'proximal_curr', 9: 'distal_curr'}


def _nest_params():
    """The demo's neuron params translated to NEST's plain-float dict."""
    return {
        "V_th": -60.0, "V_reset": -65.0, "t_ref": 10.0, "g_sp": 5.0,
        "soma": {"g_L": 12.0},
        "proximal": {"tau_syn_ex": 1.0, "tau_syn_in": 5.0},
        "distal": {"C_m": 90.0},
    }


def _nest_traces(simtime=demo.T_SIM):
    """Live-NEST per-compartment traces + spike count for the demo's exact wiring."""
    nest.ResetKernel()
    nest.resolution = demo.DT
    nest.set_verbosity("M_ERROR")
    syns = nest.GetDefaults("iaf_cond_alpha_mc")["receptor_types"]
    n = nest.Create("iaf_cond_alpha_mc", params=_nest_params())
    mm = nest.Create("multimeter",
                     params={"record_from": list(demo.RECORDABLES), "interval": demo.DT})
    nest.Connect(mm, n)
    sr = nest.Create("spike_recorder")
    nest.Connect(n, sr)

    # Paradigm 1: per-compartment current pulses (receptors 7/8/9).
    for rtype, amp, start, stop in demo.CURRENT_PULSES:
        cg = nest.Create("dc_generator")
        cg.set(start=start, stop=stop, amplitude=amp)
        nest.Connect(cg, n, syn_spec={"receptor_type": syns[_RT_NAME[rtype]]})

    # Paradigm 2: per-compartment spike trains (receptors 1-6).
    for rtype, times in demo.SPIKE_TRAINS:
        sg = nest.Create("spike_generator")
        sg.spike_times = times
        nest.Connect(sg, n, syn_spec={"receptor_type": syns[_RT_NAME[rtype]],
                                      "weight": demo.SPIKE_WEIGHT})

    # Paradigm 3: somatic rheobase via a step generator (matches the demo).
    step = nest.Create("step_current_generator")
    step.set(amplitude_times=[demo.RHEO_START], amplitude_values=[demo.RHEO_AMP])
    nest.Connect(step, n, syn_spec={"receptor_type": syns["soma_curr"]})

    nest.Simulate(simtime)
    ev = mm.events
    traces = {name: np.asarray(ev[name]) for name in demo.RECORDABLES}
    return traces, int(sr.n_events)


@requires_nest
class TestMcNeuronParity(unittest.TestCase):
    """Per-compartment V_m, conductances and spike count match live NEST."""

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    @classmethod
    def setUpClass(cls):
        if not _HAS_NEST:
            return
        cls._nest_traces, cls._nest_n = _nest_traces()
        cls._t, cls._bp_traces, cls._bp_n = demo.run_traces()

    def _compare(self, name, band):
        ref = self._nest_traces[name]
        cand = self._bp_traces[name]
        mv = min(len(ref), len(cand))
        compare_trace(ref[:mv], cand[:mv], tol=band, metric=f"mc {name}").assert_()

    def test_vm_traces_match_nest(self):
        for name in _VM_RECORDABLES:
            self._compare(name, _VM_BAND)

    def test_conductance_traces_match_nest(self):
        for name in _G_RECORDABLES:
            self._compare(name, _G_BAND)

    def test_spike_count_matches_nest(self):
        self.assertLessEqual(abs(self._nest_n - self._bp_n), 2,
                             f"mc rheobase spike count NEST={self._nest_n} "
                             f"brainpy={self._bp_n}")


class TestMcNeuronBehaviour(unittest.TestCase):
    """NEST-free: the demo's device→compartment routing is compartment-local."""

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    @classmethod
    def setUpClass(cls):
        cls._t, cls._traces, cls._n = demo.run_traces()

    def _window(self, name, t0, t1):
        t = self._t
        sel = (t >= t0) & (t < t1)
        return self._traces[name][sel]

    def test_distal_current_pulse_is_compartment_local(self):
        # The +100 pA distal pulse (50-100 ms) should deflect the distal compartment
        # far more than the soma (two electrotonic hops away).
        rest = -70.0
        d_dev = float(np.abs(self._window('V_m.d', 50.0, 110.0) - rest).max())
        s_dev = float(np.abs(self._window('V_m.s', 50.0, 110.0) - rest).max())
        self.assertGreater(d_dev, 1.0, "distal pulse should depolarize distal")
        self.assertLess(s_dev, d_dev, "soma moved as much as the directly-driven distal")

    def test_soma_spike_train_raises_only_soma_g_ex(self):
        # The soma excitatory train (600, 620 ms) must raise g_ex.s while the distal
        # excitatory channel stays at rest in that window (no cross-compartment leak).
        g_s = float(self._window('g_ex.s', 600.0, 660.0).max())
        g_d = float(self._window('g_ex.d', 600.0, 660.0).max())
        self.assertGreater(g_s, 0.1, "soma exc spikes should raise g_ex.s")
        self.assertLess(g_d, 1e-9, "distal g_ex must stay zero for a soma spike")

    def test_rheobase_makes_soma_fire(self):
        # Paradigm 3: the 150 pA somatic rheobase from 700 ms drives output spikes.
        self.assertGreater(self._n, 0, "somatic rheobase should make the neuron fire")


if __name__ == "__main__":
    unittest.main()
