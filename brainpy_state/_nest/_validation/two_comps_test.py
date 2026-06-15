# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Live-NEST parity for the two-compartment ``cm_default`` demo (§3.10).

The demo ``examples/nest/two_comps.py`` builds the same soma+dendrite tree twice
— once with a **passive** dendrite (cm_pas), once with an **active** Na/K dendrite
(cm_act) — and drives both with identical somatic (10/13/16 ms, 5 nS) and dendritic
(70/73/76 ms, 2 nS) spike trains through ``AMPA_NMDA`` receptors. This exercises the
full Simulator device→compartment-receptor routing seam for the general
multi-compartment model end to end.

``cm_default`` integrates the cable with an **exact fixed-step Crank–Nicolson**
solver (category **C**), so against a live NEST ``cm_default`` with the same
morphology and drive the traces match very tightly once a constant recorder offset
is removed (NEST records 1595 samples to the demo's 1600 — the 0.5 ms connection
delay — and the optimal alignment is a constant −2-step shift, both absorbed by
``align_steps`` + min-length truncation):

* **Synaptic conductances** ``g_{r,d}_AN_{AMPA,NMDA}`` — deterministic
  double-exponential kernels with the same ``g_norm`` and ``dt`` on both sides, so
  they match to the **float-noise floor** (≈1e-15 nS).
* **Na/K gating** ``m/h/n`` — match to ≈2.5e-3 (dimensionless).
* **Compartment voltages** ``v_comp`` — unlike a reset-based neuron, ``cm_default``
  has **no reset**: the somatic action potential is a genuine Na/K upstroke that
  both integrators evolve identically, matching to ≈0.03 mV. The only larger
  residual (≈0.56 mV) is at the **tip of the active dendrite's sharp Na spike**,
  where sub-step timing makes the peak sample sensitive; the band tolerates it.

The NEST-free behaviour test pins the demo's pedagogical point: the soma fires a
real (overshooting) action potential, and the active dendrite amplifies the
dendritic response well beyond the passive cable.
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

import examples.nest.two_comps as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

# Category C (exact Crank–Nicolson cable). Each recordable self-aligns within
# +/-align_steps; the bands are the measured worst-case residual with ~2x headroom.
# v_comp tolerates the sub-step peak sensitivity at the active dendrite's Na spike
# tip (the soma AP itself matches to ~0.03 mV).
_V_BAND = tc.TraceTolerance(1.0 * u.mV, 2e-3, align_steps=6, label="C",
                            note="cm_default v_comp; Na/K spikes (no reset) -> sub-step peak tip sensitivity")
_GATE_BAND = tc.TraceTolerance(6e-3, 1e-3, align_steps=6, label="C",
                               note="cm_default Na/K gating m/h/n (dimensionless)")
_G_BAND = tc.TraceTolerance(1e-4 * u.nS, 1e-3, align_steps=6, label="C",
                            note="cm_default AMPA_NMDA rise/decay conductances (det. double-exp; ~machine precision)")

_V_RECORDABLES = ('v_comp0', 'v_comp1')
_GATE_RECORDABLES = ('m_Na_0', 'h_Na_0', 'n_K_0', 'm_Na_1', 'h_Na_1', 'n_K_1')
_G_RECORDABLES = ('g_r_AN_AMPA_1', 'g_d_AN_AMPA_1', 'g_r_AN_NMDA_1', 'g_d_AN_NMDA_1')


def _nest_traces(simtime=demo.T_SIM):
    """Live-NEST per-compartment traces for the demo's exact wiring.

    Builds the passive- and active-dendrite ``cm_default`` models in one kernel
    (as the demo builds both in one Simulator) and returns
    ``{'pas': {name: trace}, 'act': {name: trace}}`` over :data:`demo.RECORDABLES`.
    """
    nest.ResetKernel()
    nest.resolution = demo.DT
    nest.set_verbosity("M_ERROR")
    mms = {}
    for key, active in (('pas', False), ('act', True)):
        dend = demo.DEND_ACTIVE if active else demo.DEND_PASSIVE
        cm = nest.Create("cm_default")
        cm.compartments = [{"parent_idx": -1, "params": dict(demo.SOMA_PARAMS)},
                           {"parent_idx": 0, "params": dict(dend)}]
        cm.V_th = demo.V_TH
        cm.receptors = [{"comp_idx": 0, "receptor_type": "AMPA_NMDA"},
                        {"comp_idx": 1, "receptor_type": "AMPA_NMDA"}]
        sg_soma = nest.Create("spike_generator", 1, {"spike_times": demo.SG_SOMA_TIMES})
        sg_dend = nest.Create("spike_generator", 1, {"spike_times": demo.SG_DEND_TIMES})
        nest.Connect(sg_soma, cm, syn_spec={"synapse_model": "static_synapse",
                     "weight": demo.W_SOMA, "delay": demo.DELAY, "receptor_type": demo.SYN_SOMA})
        nest.Connect(sg_dend, cm, syn_spec={"synapse_model": "static_synapse",
                     "weight": demo.W_DEND, "delay": demo.DELAY, "receptor_type": demo.SYN_DEND})
        mm = nest.Create("multimeter", 1, {"record_from": list(demo.RECORDABLES), "interval": demo.DT})
        nest.Connect(mm, cm)
        mms[key] = mm
    nest.Simulate(simtime)
    return {key: {name: np.asarray(mm.events[name]) for name in demo.RECORDABLES}
            for key, mm in mms.items()}


@requires_nest
class TestTwoCompsParity(unittest.TestCase):
    """Per-compartment voltage, gating and synaptic conductances match live NEST."""

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    @classmethod
    def setUpClass(cls):
        if not _HAS_NEST:
            return
        cls._nest = _nest_traces()
        cls._t, cls._bp = demo.run_traces()

    def _compare(self, key, name, band):
        ref = np.asarray(self._nest[key][name]).reshape(-1)
        cand = np.asarray(self._bp[key][name]).reshape(-1)
        mv = min(len(ref), len(cand))
        compare_trace(ref[:mv], cand[:mv], tol=band, metric=f"two_comps {key} {name}").assert_()

    def test_voltage_traces_match_nest(self):
        for key in ('pas', 'act'):
            for name in _V_RECORDABLES:
                self._compare(key, name, _V_BAND)

    def test_gating_traces_match_nest(self):
        for key in ('pas', 'act'):
            for name in _GATE_RECORDABLES:
                self._compare(key, name, _GATE_BAND)

    def test_conductance_traces_match_nest(self):
        for key in ('pas', 'act'):
            for name in _G_RECORDABLES:
                self._compare(key, name, _G_BAND)


class TestTwoCompsBehaviour(unittest.TestCase):
    """NEST-free: the demo's active-vs-passive dendrite contrast holds."""

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    @classmethod
    def setUpClass(cls):
        cls._t, cls._traces = demo.run_traces()

    def _window(self, key, name, t0, t1):
        sel = (self._t >= t0) & (self._t < t1)
        return self._traces[key][name][sel]

    def test_soma_fires_action_potential(self):
        # cm_default has no reset: a somatic AP is a genuine Na/K upstroke that
        # overshoots 0 mV. Both models share the same (active) soma, so both fire.
        for key in ('pas', 'act'):
            vmax = float(np.max(self._traces[key]['v_comp0']))
            self.assertGreater(vmax, 0.0, f"{key}: soma should fire an overshooting AP")

    def test_active_dendrite_amplifies(self):
        # In the dendritic-input window the active dendrite (Na/K) must depolarize
        # markedly further than the passive cable driven by the identical 2 nS train.
        pas = float(np.max(self._window('pas', 'v_comp1', 70.0, 100.0)))
        act = float(np.max(self._window('act', 'v_comp1', 70.0, 100.0)))
        self.assertGreater(act, pas + 5.0, "active dendrite should amplify the dendritic response")

    def test_dendritic_conductance_is_causal(self):
        # The dendritic AMPA conductance (g_r + g_d) must be ~zero before the dend
        # spikes (70 ms + delay) and clearly positive after — a causal, shaped EPSC.
        for key in ('pas', 'act'):
            g = self._traces[key]['g_r_AN_AMPA_1'] + self._traces[key]['g_d_AN_AMPA_1']
            before = float(np.max(np.abs(g[self._t < 70.0])))
            after = float(np.max(g[(self._t >= 70.0) & (self._t < 100.0)]))
            self.assertLess(before, 1e-9, f"{key}: dend AMPA conductance must be causal")
            self.assertGreater(after, 0.1, f"{key}: dend spikes should open AMPA conductance")


if __name__ == "__main__":
    unittest.main()
