# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity + mechanism-subset checks for ``examples/nest/glif_psc_neuron.py``.

The current-based GLIF neuron is validated at all five mechanism levels. Unlike
its conductance sibling ``glif_cond`` (RKF45-integrated alpha conductances),
``glif_psc`` advances the whole system with **exact propagator matrices**, so the
parity is even sharper:

* **Mechanism subset (NEST-free).** Each ``(spike_dependent_threshold,
  after_spike_currents, adapting_threshold)`` flag gates exactly its own state:
  after-spike currents stay at machine zero unless enabled, the spike-triggered
  threshold component only grows when enabled, and the voltage-adapting component
  only moves when enabled. The summed post-synaptic current ``I_syn`` is a linear
  filter of the *external* excitatory/inhibitory spike trains alone, so it is
  bit-identical across all five mechanism levels. The Poisson window
  (``poisson_generator → parrot_neuron → receptor 2``) must add post-synaptic
  activity, exercising the parrot relay end to end.

* **Live-NEST parity (deterministic drive, no Poisson).** With the stochastic
  Poisson window dropped (brainpy and NEST draw from independent PRNG streams),
  the drive is fully deterministic (step current + excitatory/inhibitory receptor
  spikes). Then:

  - the summed post-synaptic current ``I_syn`` and the injected current ``I`` match
    NEST over the **whole 1 s trace** to machine precision — they are linear
    filters of the fixed external drive, independent of the neuron's own spikes;
  - ``V_m`` and the (possibly adapting) ``threshold`` match NEST per sample in the
    sub-threshold window before the first spike. ``V_m`` carries a ~0.03 mV
    residual there: brainpy computes ``V`` from the *pre*-propagation PSC while the
    multimeter samples the *post*-propagation ``y2`` as ``I_syn``, so ``V_m`` and the
    recorded ``I_syn`` sit one integration step apart relative to NEST (which reports
    both from the same ``y2``). The residual is ~0.1 % of the ~28 mV sub-threshold
    swing — hence ``CAT_B_ALIGNED`` (5e-2 mV, one-step aligned) rather than the
    machine-precision band the linear currents enjoy;
  - the spike **count** matches NEST within the category-E event tolerance (in
    practice exactly, across all five levels).

NEST records ``V_m = U + E_L`` and ``threshold = threshold_ + E_L`` (absolute mV)
while brainpy stores the GLIF threshold relative to ``E_L``; the test adds ``E_L``
to the brainpy threshold. As for the other multimeter demos, the brainpy ``t = 0``
sample is dropped and ``align_steps=1`` absorbs the residual one-step recorder
offset (each trace finds its own best shift: ``I_syn`` at 0, ``V_m``/``I`` at one step).
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

import saiunit as u

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import (
    TraceTolerance, CAT_B_ALIGNED)

from examples.nest.glif_psc_neuron import (
    run_traces, GLIF_LEVELS, TAU_SYN, ESPIKE_TIMES, ISPIKE_TIMES, ESPIKE_W,
    ISPIKE_W, STEP_AMP, STEP_START, STEP_STOP, RESOLUTION, SIMTIME, RECORD_FROM)

#: NEST/brainpy shared default resting potential (mV); the GLIF threshold frame.
E_L = -78.85
#: End the sub-threshold comparison this many ms before the first spike.
SUBTH_MARGIN = 1.0

#: V_m: exact-propagator membrane, carrying only the one-step V-vs-recorded-PSC
#: offset (~0.03 mV at the PSC peak). CAT_B_ALIGNED = 5e-2 mV, align_steps=1.
CAT_V = CAT_B_ALIGNED
#: threshold (+E_L), sub-threshold: <= 3.6e-4 mV on the adapting level, flat
#: otherwise. Bare-mV CAT_A band with the one-step recorder shift.
CAT_TH = TraceTolerance(1e-3, 1e-3, align_steps=1, label="A",
                        note="glif_psc threshold (+E_L), sub-threshold window")
#: I_syn / I are linear filters of the fixed external drive -> machine precision
#: (observed ~2e-15 pA / 0 pA). Bare-pA band, one-step aligned.
CAT_I = TraceTolerance(1e-6, 1e-6, align_steps=1, label="B",
                       note="glif_psc I_syn / I, full trace")
#: Category-E event-count tolerance (|ΔN| <= 2).
SPIKE_COUNT_TOL = 2


def _nest_run(mech, dt=RESOLUTION, simtime=SIMTIME):
    """Deterministic NEST ``glif_psc`` reference (no Poisson window).

    Mirrors NEST's ``glif_psc_neuron.py`` wiring exactly: the synaptic weights
    ride on the ``spike_generator`` (``spike_weights``), both excitatory and
    inhibitory trains target ``receptor_type=1``, and the connection weight is the
    default 1.0.
    """
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    n = nest.Create('glif_psc', params=dict(mech, tau_syn=list(TAU_SYN)))
    espk = nest.Create('spike_generator', params={
        'spike_times': list(ESPIKE_TIMES), 'spike_weights': [ESPIKE_W] * len(ESPIKE_TIMES)})
    ispk = nest.Create('spike_generator', params={
        'spike_times': list(ISPIKE_TIMES), 'spike_weights': [ISPIKE_W] * len(ISPIKE_TIMES)})
    cg = nest.Create('step_current_generator', params={
        'amplitude_values': [STEP_AMP], 'amplitude_times': [STEP_START],
        'start': STEP_START, 'stop': STEP_STOP})
    mm = nest.Create('multimeter', params={'interval': dt, 'record_from': list(RECORD_FROM)})
    sr = nest.Create('spike_recorder')
    nest.Connect(cg, n, syn_spec={'delay': dt})
    nest.Connect(espk, n, syn_spec={'delay': dt, 'receptor_type': 1})
    nest.Connect(ispk, n, syn_spec={'delay': dt, 'receptor_type': 1})
    nest.Connect(mm, n)
    nest.Connect(n, sr)
    nest.Simulate(simtime)
    ev = mm.events
    traces = {k: np.asarray(ev[k]) for k in RECORD_FROM}
    return traces, int(sr.n_events), np.asarray(sr.events['times'])


# Cache the (expensive) brainpy runs across test classes/methods, keyed by
# (level label, with_poisson) — each is a full jit-compiled 20k-step for_loop.
_BP_CACHE = {}


def _bp(label, mech, with_poisson):
    key = (label, with_poisson)
    if key not in _BP_CACHE:
        _BP_CACHE[key] = run_traces(mech, with_poisson=with_poisson)
    return _BP_CACHE[key]


class TestGlifPscMechanismSubset(unittest.TestCase):
    """NEST-free: each mechanism flag gates exactly its own state."""

    def test_after_spike_currents_active_only_when_enabled(self):
        for label, mech in GLIF_LEVELS:
            tr = _bp(label, mech, False)
            asc_peak = float(np.abs(tr['ASCurrents_sum']).max())
            with self.subTest(level=label):
                if mech['after_spike_currents']:
                    self.assertGreater(asc_peak, 1.0)      # ASC kicks after spikes
                else:
                    self.assertLess(asc_peak, 1e-9)        # exactly inert otherwise

    def test_threshold_components_track_flags(self):
        for label, mech in GLIF_LEVELS:
            tr = _bp(label, mech, False)
            tspk = float(np.abs(tr['threshold_spike']).max())
            tvlt = float(np.abs(tr['threshold_voltage']).max())
            with self.subTest(level=label):
                if mech['spike_dependent_threshold']:
                    self.assertGreater(tspk, 1e-3)         # spike-triggered bump grows
                else:
                    self.assertLess(tspk, 1e-9)
                if mech['adapting_threshold']:
                    self.assertGreater(tvlt, 1e-3)         # voltage-adapting term moves
                else:
                    self.assertLess(tvlt, 1e-9)

    def test_plain_lif_threshold_is_flat(self):
        # lif (no spike/voltage threshold mechanisms): total threshold is constant.
        label, mech = GLIF_LEVELS[0]
        self.assertEqual(label, 'lif')
        tr = _bp(label, mech, False)
        th = tr['threshold']
        self.assertLess(float(th.max() - th.min()), 1e-9)

    def test_i_syn_is_mechanism_independent(self):
        # I_syn is a linear filter of the external exc/inh trains and the fixed
        # tau_syn alone — the PSC dynamics never see V, the threshold, or the ASCs.
        # So the summed post-synaptic current must be bit-identical across levels.
        ref = _bp('lif', GLIF_LEVELS[0][1], False)['I_syn']
        for label, mech in GLIF_LEVELS[1:]:
            tr = _bp(label, mech, False)
            with self.subTest(level=label):
                self.assertLess(float(np.abs(tr['I_syn'] - ref).max()), 1e-9)

    def test_poisson_window_adds_activity_via_parrot(self):
        # The poisson_generator -> parrot_neuron -> receptor 2 chain must inject
        # extra spikes in the 600-900 ms window relative to the deterministic run.
        label, mech = GLIF_LEVELS[0]
        det = _bp(label, mech, False)
        wp = _bp(label, mech, True)
        self.assertGreater(wp['n_spikes'], det['n_spikes'])
        in_window = wp['spike_times'][(wp['spike_times'] >= 600.0) &
                                      (wp['spike_times'] <= 900.0)]
        det_window = det['spike_times'][(det['spike_times'] >= 600.0) &
                                        (det['spike_times'] <= 900.0)]
        self.assertGreater(in_window.size, det_window.size)


@requires_nest
class TestGlifPscNestParity(unittest.TestCase):
    """Deterministic (no-Poisson) live-NEST parity across all five levels."""

    @classmethod
    def setUpClass(cls):
        cls.nest = {label: _nest_run(mech) for label, mech in GLIF_LEVELS}

    def test_psc_and_injected_current_match_nest_full_trace(self):
        # I_syn (summed PSC) and I (step current) are linear filters of the fixed
        # external drive, so they match NEST over the whole 1 s trace to machine
        # precision regardless of neuronal spike jitter.
        for label, mech in GLIF_LEVELS:
            bp = _bp(label, mech, False)
            ns, _n, _t = self.nest[label]
            with self.subTest(level=label):
                compare_trace(ns['I_syn'], bp['I_syn'][1:], tol=CAT_I,
                              metric=f'glif_psc {label} I_syn').assert_()
                compare_trace(ns['I'], bp['I'][1:], tol=CAT_I,
                              metric=f'glif_psc {label} I').assert_()

    def test_subthreshold_Vm_and_threshold_match_nest(self):
        for label, mech in GLIF_LEVELS:
            bp = _bp(label, mech, False)
            ns, _n, ns_t = self.nest[label]
            first_spike = ns_t[0] if ns_t.size else SIMTIME
            w = int(max(first_spike - SUBTH_MARGIN, 5.0) / RESOLUTION)
            with self.subTest(level=label):
                compare_trace(ns['V_m'][:w], bp['V_m'][1:][:w], tol=CAT_V,
                              metric=f'glif_psc {label} V_m (subthreshold)').assert_()
                # NEST threshold is absolute (threshold_ + E_L); brainpy stores it
                # relative to E_L.
                compare_trace(ns['threshold'][:w], bp['threshold'][1:][:w] + E_L,
                              tol=CAT_TH,
                              metric=f'glif_psc {label} threshold (subthreshold)').assert_()

    def test_spike_counts_match_nest(self):
        for label, mech in GLIF_LEVELS:
            bp = _bp(label, mech, False)
            _ns, ns_n, _t = self.nest[label]
            with self.subTest(level=label):
                self.assertLessEqual(abs(bp['n_spikes'] - ns_n), SPIKE_COUNT_TOL)


if __name__ == '__main__':
    unittest.main()
