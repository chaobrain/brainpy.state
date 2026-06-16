# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Repeated-stimulation parity: brainpy.state Simulator vs live NEST.

Ports NEST's ``repeated_stimulation.py`` — a ``poisson_generator`` gated to a
``[start, stop]`` window, repeated across trials. The headline statistic is the
per-trial spike count inside the active window (≈ ``rate · (stop−start)``); the
window-gating makes the rest of each trial silent. PRNG diverges (NEST per-thread
RNG vs JAX) → distributional parity (``CAT_D``). The no-NEST companion runs
always so CI exercises the importable surface (``fix-brunel-es-import`` rule).
"""
import unittest

import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

try:
    import nest
except Exception:
    nest = None

from examples.nest_like.repeated_stimulation import (
    build, run_trials, window_count, RATE, T_START, T_STOP, TRIAL_DURATION,
)
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest_validation.tolerance_conventions import CAT_D

EXPECTED = RATE * (T_STOP - T_START) / 1000.0   # ≈ 400 spikes per active window


class TestRepeatedStimulation(unittest.TestCase):
    """No-NEST companion: gating + repeat behaviour of our Simulator port."""

    def test_active_window_count_near_expected(self):
        trains = run_trials(num_trials=1, seed=0)
        n = window_count(trains[0], T_START, T_STOP)
        self.assertLess(abs(n - EXPECTED) / EXPECTED, 0.2)   # Poisson, 1 trial

    def test_spikes_confined_to_window(self):
        # The whole point of the demo: stimulation only inside [start, stop].
        trains = run_trials(num_trials=1, seed=1)
        before = window_count(trains[0], 0.0, T_START)
        after = window_count(trains[0], T_STOP, TRIAL_DURATION)
        self.assertEqual(before, 0)
        self.assertEqual(after, 0)

    def test_repeated_identical_stimulation(self):
        # Each trial reproduces the same windowed drive.
        trains = run_trials(num_trials=5, seed=0)
        self.assertEqual(len(trains), 5)
        for tr in trains:
            n = window_count(tr, T_START, T_STOP)
            self.assertLess(abs(n - EXPECTED) / EXPECTED, 0.2)

    def test_zero_rate_is_silent(self):
        # Edge case: empty external drive -> no spikes at all.
        trains = run_trials(num_trials=1, seed=0, rate=0.0)
        self.assertEqual(window_count(trains[0], 0.0, TRIAL_DURATION), 0)


@requires_nest
class TestRepeatedStimulationParity(unittest.TestCase):
    def _nest_active_counts(self, n_trials, seed0):
        counts = []
        for k in range(n_trials):
            nest.ResetKernel()
            nest.resolution = 0.1
            nest.rng_seed = seed0 + k + 1
            pg = nest.Create('poisson_generator',
                             params={'rate': RATE, 'start': T_START, 'stop': T_STOP})
            sr = nest.Create('spike_recorder')
            nest.Connect(pg, sr)
            nest.Simulate(TRIAL_DURATION)
            counts.append(sr.n_events)
        return counts

    def test_active_count_within_5pct_of_nest(self):
        n_trials = 6
        trains = run_trials(num_trials=n_trials, seed=0)
        bp_counts = [window_count(tr, T_START, T_STOP) for tr in trains]
        nest_counts = self._nest_active_counts(n_trials, seed0=0)
        self.assertGreater(np.mean(nest_counts), 0.0)
        compare_distributional(nest_counts, bp_counts, tol=CAT_D,
                               metric='active-window spike count').assert_()
