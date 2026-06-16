# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity + mechanism-subset checks for ``examples/nest_like/glif_cond_neuron.py``.

The conductance-based GLIF neuron is validated at all five mechanism levels on
several fronts:

* **Mechanism subset (NEST-free).** Each ``(spike_dependent_threshold,
  after_spike_currents, adapting_threshold)`` flag must gate exactly its own
  state: after-spike currents stay at machine zero unless enabled, the
  spike-triggered threshold component only grows when enabled, and the
  voltage-adapting component only moves when enabled. The Poisson window
  (``poisson_generator → parrot_neuron → receptor 1``) must add post-synaptic
  activity, exercising the parrot relay end to end.

* **Live-NEST parity (deterministic drive, no Poisson).** The stochastic Poisson
  window is dropped (brainpy and NEST draw from independent PRNG streams), leaving
  a fully deterministic 3-paradigm drive (step current + excitatory/inhibitory
  receptor spikes). Then:

  - the two per-port conductances ``g_1``/``g_2`` match NEST **over the whole
    1 s trace** to machine precision — they are linear filters of the fixed
    external spike trains, independent of the neuron's own (jitter-prone) spikes;
  - ``V_m`` and the (possibly adapting) ``threshold`` match NEST per sample in the
    sub-threshold window before the first spike (after a spike, sub-sample timing
    jitter between the two RKF45 integrators makes a per-sample voltage comparison
    meaningless — the spiking behaviour is checked by count instead);
  - the spike **count** matches NEST within the category-E event tolerance (in
    practice exactly, across all five levels).

NEST records ``V_m = y + E_L`` and ``threshold = threshold_ + E_L`` (absolute mV)
while brainpy stores the GLIF threshold relative to ``E_L``; the test adds ``E_L``
to the brainpy threshold. As for the other RKF45 multimeter demos, the brainpy
``t = 0`` sample is dropped and ``align_steps=1`` absorbs the residual one-step
recorder offset (a 2-sample shift overall).
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

import brainunit as u

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

from examples.nest_like.glif_cond_neuron import (
    run_traces, GLIF_LEVELS, ESPIKE_TIMES, ISPIKE_TIMES, ESPIKE_W, ISPIKE_W,
    STEP_AMP, STEP_START, STEP_STOP, RESOLUTION, SIMTIME, RECORD_FROM)

#: NEST/brainpy shared default resting potential (mV); the GLIF threshold frame.
E_L = -78.85
#: End the sub-threshold comparison this many ms before the first spike.
SUBTH_MARGIN = 1.0

# Sub-threshold V_m / threshold match NEST to ~1e-13; conductances are linear
# filters of the external trains and match over the full trace. The one-step
# recorder offset is absorbed by dropping bp[0] and align_steps=1.
CAT_A_V = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=1, label="A",
                         note="glif_cond V_m, sub-threshold window, one-step aligned")
CAT_A_TH = TraceTolerance(1e-3, 1e-3, align_steps=1, label="A",
                          note="glif_cond threshold (+E_L), sub-threshold window")
CAT_A_G = TraceTolerance(1e-3, 1e-3, align_steps=1, label="A",
                         note="glif_cond per-port conductance g_k, full trace")
#: Category-E event-count tolerance (|ΔN| <= 2).
SPIKE_COUNT_TOL = 2


def _nest_run(mech, dt=RESOLUTION, simtime=SIMTIME):
    """Deterministic NEST ``glif_cond`` reference (no Poisson window)."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    n = nest.Create('glif_cond', params=dict(mech))
    espk = nest.Create('spike_generator', params={'spike_times': list(ESPIKE_TIMES)})
    ispk = nest.Create('spike_generator', params={'spike_times': list(ISPIKE_TIMES)})
    cg = nest.Create('step_current_generator', params={
        'amplitude_values': [STEP_AMP], 'amplitude_times': [STEP_START],
        'start': STEP_START, 'stop': STEP_STOP})
    mm = nest.Create('multimeter', params={'interval': dt, 'record_from': list(RECORD_FROM)})
    sr = nest.Create('spike_recorder')
    nest.Connect(cg, n, syn_spec={'delay': dt})
    nest.Connect(espk, n, syn_spec={'delay': dt, 'receptor_type': 1, 'weight': ESPIKE_W})
    nest.Connect(ispk, n, syn_spec={'delay': dt, 'receptor_type': 2, 'weight': ISPIKE_W})
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


class TestGlifCondMechanismSubset(unittest.TestCase):
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

    def test_poisson_window_adds_activity_via_parrot(self):
        # The poisson_generator -> parrot_neuron -> receptor 1 chain must inject
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
class TestGlifCondNestParity(unittest.TestCase):
    """Deterministic (no-Poisson) live-NEST parity across all five levels."""

    @classmethod
    def setUpClass(cls):
        cls.nest = {label: _nest_run(mech) for label, mech in GLIF_LEVELS}

    def test_conductances_match_nest_full_trace(self):
        # g_1 / g_2 are linear filters of the fixed external spike trains, so they
        # match NEST over the whole 1 s trace regardless of neuronal spike jitter.
        for label, mech in GLIF_LEVELS:
            bp = _bp(label, mech, False)
            ns, _n, _t = self.nest[label]
            for k in (1, 2):
                compare_trace(ns[f'g_{k}'], bp[f'g_{k}'][1:], tol=CAT_A_G,
                              metric=f'glif_cond {label} g_{k}').assert_()

    def test_subthreshold_Vm_and_threshold_match_nest(self):
        for label, mech in GLIF_LEVELS:
            bp = _bp(label, mech, False)
            ns, _n, ns_t = self.nest[label]
            first_spike = ns_t[0] if ns_t.size else SIMTIME
            w = int(max(first_spike - SUBTH_MARGIN, 5.0) / RESOLUTION)
            with self.subTest(level=label):
                compare_trace(ns['V_m'][:w], bp['V_m'][1:][:w], tol=CAT_A_V,
                              metric=f'glif_cond {label} V_m (subthreshold)').assert_()
                # NEST threshold is absolute (threshold_ + E_L); brainpy stores it
                # relative to E_L.
                compare_trace(ns['threshold'][:w], bp['threshold'][1:][:w] + E_L,
                              tol=CAT_A_TH,
                              metric=f'glif_cond {label} threshold (subthreshold)').assert_()

    def test_spike_counts_match_nest(self):
        for label, mech in GLIF_LEVELS:
            bp = _bp(label, mech, False)
            _ns, ns_n, _t = self.nest[label]
            with self.subTest(level=label):
                self.assertLessEqual(abs(bp['n_spikes'] - ns_n), SPIKE_COUNT_TOL)


if __name__ == '__main__':
    unittest.main()
