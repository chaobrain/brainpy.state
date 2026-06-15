# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity + shape checks for ``examples/nest/glif_psc_double_alpha_neuron.py``.

``glif_psc_double_alpha`` is ``glif_psc`` with a *double* alpha synaptic kernel
(fast alpha + ``amp_slow`` x slow alpha) and is driven here with all spike
mechanisms off, so the demo isolates the synaptic-current shape. Validation:

* **Shape invariants (NEST-free).** The weak 20 pA single-spike inputs keep every
  neuron sub-threshold (no spikes). The slow component adds amplitude, so the
  double-alpha ``I_syn`` *peak* exceeds the single-alpha reference; and because the
  slow alpha decays more slowly, the double-alpha current has a markedly *heavier
  tail* (sampled 30 ms after the receptor-1 spike, before the receptor-2 input).

* **Live-NEST parity.** The drive is fully deterministic (three fixed
  excitatory spikes, one per receptor port). With no spikes and exact propagator
  matrices, both ``V_m`` and the summed synaptic current ``I_syn`` match NEST over
  the **whole 300 ms trace** to machine precision for all three configurations
  (single alpha, and the double-alpha timing / amplitude variations). As for the
  other multimeter demos, the brainpy ``t = 0`` sample is dropped and
  ``align_steps=1`` absorbs the one-step recorder offset.
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

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import TraceTolerance

from examples.nest.glif_psc_double_alpha_neuron import (
    run_traces, CONFIGS, MECH_OFF, TAU_PSC, ESPIKES, ESPIKE_W, RECORD_FROM,
    RESOLUTION, SIMTIME)

#: V_m / I_syn: no spikes + exact propagators -> machine precision (~1e-13 mV /
#: ~1e-15 pA observed). Tight bands with the one-step recorder shift.
CAT_V = TraceTolerance(1e-6 * u.mV, 1e-6, align_steps=1, label="B",
                       note="glif_psc_double_alpha V_m, full trace")
CAT_I = TraceTolerance(1e-6, 1e-6, align_steps=1, label="B",
                       note="glif_psc_double_alpha I_syn, full trace")
#: Sample the post-synaptic tail this long after the receptor-1 spike (10 ms),
#: still before the receptor-2 spike at 110 ms.
TAIL_T = 40.0


def _nest_run(cfg, dt=RESOLUTION, simtime=SIMTIME):
    """Deterministic NEST reference for one configuration.

    ``cfg`` is ``None`` for the single-alpha ``glif_psc`` (with :data:`TAU_PSC`),
    else a ``glif_psc_double_alpha`` keyword dict.
    """
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    if cfg is None:
        n = nest.Create('glif_psc', params=dict(MECH_OFF, tau_syn=list(TAU_PSC)))
    else:
        n = nest.Create('glif_psc_double_alpha', params=dict(
            MECH_OFF,
            tau_syn_fast=list(cfg['tau_syn_fast']),
            tau_syn_slow=list(cfg['tau_syn_slow']),
            amp_slow=list(cfg['amp_slow'])))
    mm = nest.Create('multimeter', params={'interval': dt, 'record_from': list(RECORD_FROM)})
    sr = nest.Create('spike_recorder')
    for t, rec in ESPIKES:
        g = nest.Create('spike_generator', params={'spike_times': [t], 'spike_weights': [ESPIKE_W]})
        nest.Connect(g, n, syn_spec={'delay': dt, 'receptor_type': rec})
    nest.Connect(mm, n)
    nest.Connect(n, sr)
    nest.Simulate(simtime)
    ev = mm.events
    traces = {k: np.asarray(ev[k]) for k in RECORD_FROM}
    return traces, int(sr.n_events)


# Cache the (expensive) brainpy runs across test classes/methods, keyed by label.
_BP_CACHE = {}


def _bp(label, cfg):
    if label not in _BP_CACHE:
        _BP_CACHE[label] = run_traces(cfg)
    return _BP_CACHE[label]


class TestDoubleAlphaShape(unittest.TestCase):
    """NEST-free: the double-alpha kernel reshapes the synaptic current."""

    def test_all_configs_stay_subthreshold(self):
        for label, cfg in CONFIGS:
            with self.subTest(config=label):
                self.assertEqual(_bp(label, cfg)['n_spikes'], 0)

    def test_double_alpha_peak_exceeds_single_alpha(self):
        # The slow component adds amplitude on top of the fast alpha, so both
        # double-alpha variants peak above the single-alpha reference.
        single_peak = float(np.abs(_bp('glif_psc', None)['I_syn']).max())
        for label, cfg in CONFIGS:
            if cfg is None:
                continue
            with self.subTest(config=label):
                self.assertGreater(float(np.abs(_bp(label, cfg)['I_syn']).max()),
                                   single_peak * 1.05)

    def test_double_alpha_has_heavier_tail(self):
        # 30 ms after the receptor-1 spike the fast (single) alpha has all but
        # decayed; the slow component keeps the double-alpha current well above it.
        ref = _bp('glif_psc', None)
        t = ref['times']
        i = int(np.argmin(np.abs(t - TAIL_T)))
        single_tail = float(ref['I_syn'][i])
        for label, cfg in CONFIGS:
            if cfg is None:
                continue
            with self.subTest(config=label):
                self.assertGreater(float(_bp(label, cfg)['I_syn'][i]),
                                   single_tail * 5.0)


@requires_nest
class TestDoubleAlphaNestParity(unittest.TestCase):
    """Deterministic live-NEST parity for all three configurations."""

    @classmethod
    def setUpClass(cls):
        cls.nest = {label: _nest_run(cfg) for label, cfg in CONFIGS}

    def test_Vm_and_Isyn_match_nest_full_trace(self):
        for label, cfg in CONFIGS:
            bp = _bp(label, cfg)
            ns, _n = self.nest[label]
            with self.subTest(config=label):
                compare_trace(ns['V_m'], bp['V_m'][1:], tol=CAT_V,
                              metric=f'{label} V_m').assert_()
                compare_trace(ns['I_syn'], bp['I_syn'][1:], tol=CAT_I,
                              metric=f'{label} I_syn').assert_()

    def test_no_spikes_match_nest(self):
        for label, cfg in CONFIGS:
            _ns, ns_n = self.nest[label]
            with self.subTest(config=label):
                self.assertEqual(_bp(label, cfg)['n_spikes'], ns_n)


if __name__ == '__main__':
    unittest.main()
