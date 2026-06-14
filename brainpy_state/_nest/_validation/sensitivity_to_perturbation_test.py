# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Sensitivity-to-perturbation parity: brainpy.state Simulator vs live NEST.

Ports NEST's ``sensitivity_to_perturbation.py`` — a Brunel-style sparse balanced
E/I network of ``iaf_psc_delta`` neurons, run in two trials identical except for
one extra input spike at ``t_stim`` (injected into the first neuron to fire after
``t_stim``). In the balanced regime the network is chaotic, so that single spike
can decorrelate the whole population (London et al. 2010).

Parity model — **distributional + qualitative, never per-sample** (the system is
chaotic and the PRNG streams diverge):

* **rate** — the AI-state population firing rate matches NEST within ``CAT_D``
  (5 %, multi-seed); the clean distributional observable.
* **sensitivity** — both simulators are *bit-identical before* ``t_stim``
  (``d_before == 0``) and exhibit the *probabilistic chaotic transition*: each
  perturbation either dies (~0) or decorrelates ~0.97–0.99 of the network, the
  split being realization-dependent. On the deterministically-chaotic seeds
  ``{7, 12}`` (for the locked scaled config) a perturbation decorrelates > 0.9 of
  the network on *both* sims. Per-seed divergence does **not** match across sims
  (different perturbed neuron / connectivity under independent PRNG).

The no-NEST companion runs always so CI exercises the importable surface
(``fix-brunel-es-import`` rule).
"""
import unittest

import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np

try:
    import nest
except Exception:
    nest = None

from examples.nest.sensitivity_to_perturbation import (
    network_rate, run_sensitivity, first_spike_after, divergence,
    NE, NI, KE, KI, J, G, JEXT, RATE_EXT, VMIN, VMAX,
)
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest._validation.tolerance_conventions import CAT_D

#: Locked scaled config — keeps the in-degree regime (K=100/25) but shrinks N so
#: the parity test is fast. Chaotic seeds {7, 12} are deterministic at this config.
SCALED = dict(ne=300, ni=75, ke=100, ki=25, t_stim=120.0, dt=0.1, T=250.0)
CHAOTIC_SEEDS = (7, 12)
RATE_SEEDS = (7, 8, 9, 11)


class TestSensitivityToPerturbation(unittest.TestCase):
    """No-NEST companion: identical-before-perturbation + chaotic divergence."""

    @classmethod
    def setUpClass(cls):
        cls.div7 = run_sensitivity(7, **SCALED)        # a chaotic realization

    def test_identical_before_perturbation(self):
        # The two trials share network, init and external drive → bit-identical
        # until the extra spike at t_stim.
        self.assertEqual(self.div7['d_before'], 0)

    def test_perturbation_triggers_divergence(self):
        # One extra spike decorrelates almost the whole network (chaotic regime).
        self.assertGreater(self.div7['frac_decorr'], 0.9)

    def test_divergence_onset_not_before_tstim(self):
        # No difference can appear before the perturbation is delivered.
        self.assertIsNotNone(self.div7['onset_ms'])
        self.assertGreaterEqual(self.div7['onset_ms'], SCALED['t_stim'])

    def test_ai_state_rate(self):
        # Balanced asynchronous-irregular state: a sane moderate rate, not silent
        # and not refractory-saturated (~500 Hz).
        self.assertGreater(self.div7['rate'], 5.0)
        self.assertLess(self.div7['rate'], 40.0)

    def test_determinism(self):
        # No hidden global PRNG: same seed reproduces the run bit-for-bit.
        again = run_sensitivity(7, **SCALED)
        self.assertEqual(again['d_after'], self.div7['d_after'])
        self.assertEqual(again['frac_decorr'], self.div7['frac_decorr'])

    def test_divergence_pure_helper(self):
        # divergence() is a pure function over two rasters: identical rasters have
        # zero divergence; a single differing spike after t_stim is counted.
        r = np.zeros((100, 4), dtype=bool)
        r[10, 1] = True
        d0 = divergence(r, r.copy(), t_stim=2.0, dt=0.1)
        self.assertEqual(d0['d_after'], 0)
        self.assertEqual(d0['frac_decorr'], 0.0)
        r2 = r.copy(); r2[50, 2] = True       # one extra spike at t=5 ms
        d1 = divergence(r, r2, t_stim=2.0, dt=0.1)
        self.assertEqual(d1['d_before'], 0)
        self.assertEqual(d1['d_after'], 1)

    def test_first_spike_after_picks_earliest(self):
        # Earliest step strictly after the split, lowest neuron id on ties.
        r = np.zeros((100, 5), dtype=bool)
        r[30, 3] = True                        # before split
        r[60, 4] = True; r[60, 2] = True       # earliest after split → id 2
        r[70, 0] = True
        self.assertEqual(first_spike_after(r, t_split=4.0, dt=0.1), 2)


@requires_nest
class TestSensitivityToPerturbationParity(unittest.TestCase):
    def _nest_two_trials(self, seed, perturb_id=None, *, ne, ni, ke, ki, t_stim, dt, T):
        n = ne + ni
        nest.ResetKernel()
        nest.resolution = dt
        ex = nest.Create('iaf_psc_delta', ne)
        inh = nest.Create('iaf_psc_delta', ni)
        alln = ex + inh
        nest.Connect(ex, alln, {'rule': 'fixed_indegree', 'indegree': ke},
                     {'weight': J, 'delay': dt})
        nest.Connect(inh, alln, {'rule': 'fixed_indegree', 'indegree': ki},
                     {'weight': -G * J, 'delay': dt})
        ext = nest.Create('poisson_generator', params={'rate': RATE_EXT, 'stop': T})
        nest.Connect(ext, alln, syn_spec={'weight': JEXT, 'delay': dt})
        sr = nest.Create('spike_recorder')
        nest.Connect(alln, sr)
        nest.rng_seed = seed
        rng = np.random.RandomState(seed)
        alln.V_m = list(VMIN + (VMAX - VMIN) * rng.rand(n))
        if perturb_id is not None:
            stim = nest.Create('spike_generator')
            stim.spike_times = [t_stim]
            nest.Connect(stim, alln[int(perturb_id):int(perturb_id) + 1],
                         syn_spec={'weight': JEXT, 'delay': dt})
        nest.Simulate(T)
        ev = sr.get('events')
        return (np.asarray(ev['senders']), np.asarray(ev['times']),
                alln[0].global_id, n)

    def _nest_grid(self, senders, times, gid0, n, *, dt, T):
        steps = int(round(T / dt))
        g = np.zeros((steps, n), dtype=bool)
        si = (np.round(times / dt).astype(int) - 1).clip(0, steps - 1)
        g[si, (senders - gid0).astype(int)] = True
        return g

    def _nest_sensitivity(self, seed):
        cfg = SCALED
        se0, ti0, g0, n = self._nest_two_trials(seed, **cfg)
        post = ti0 > cfg['t_stim']
        pid = int(se0[post][np.argmin(ti0[post])] - g0)
        r0 = self._nest_grid(se0, ti0, g0, n, dt=cfg['dt'], T=cfg['T'])
        se1, ti1, g1, _ = self._nest_two_trials(seed, perturb_id=pid, **cfg)
        r1 = self._nest_grid(se1, ti1, g1, n, dt=cfg['dt'], T=cfg['T'])
        d = divergence(r0, r1, t_stim=cfg['t_stim'], dt=cfg['dt'])
        d['rate'] = se0.size / n / (cfg['T'] / 1000.0)
        return d

    def _nest_rate(self, seed):
        cfg = SCALED
        se0, _, _, n = self._nest_two_trials(seed, **cfg)
        return se0.size / n / (cfg['T'] / 1000.0)

    def test_rate_matches_nest(self):
        bp = [network_rate(s, **SCALED) for s in RATE_SEEDS]
        ref = [self._nest_rate(s) for s in RATE_SEEDS]
        self.assertGreater(np.mean(ref), 0.0)
        compare_distributional(ref, bp, tol=CAT_D, metric='network rate (Hz)').assert_()

    def test_sensitivity_present_in_both(self):
        bp = [run_sensitivity(s, **SCALED) for s in CHAOTIC_SEEDS]
        ref = [self._nest_sensitivity(s) for s in CHAOTIC_SEEDS]
        for d in bp + ref:                       # bit-identical before perturbation
            self.assertEqual(d['d_before'], 0)
        # A perturbation decorrelates > 0.9 of the network on both simulators.
        self.assertGreater(max(d['frac_decorr'] for d in bp), 0.9)
        self.assertGreater(max(d['frac_decorr'] for d in ref), 0.9)
