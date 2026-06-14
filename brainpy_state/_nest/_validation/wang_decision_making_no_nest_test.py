# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""CI-safe companion for the Wang (2002) decision network example.

Runs without NEST: it imports the ported example and exercises a *tiny*,
mean-field-preserving headless network plus the pure decision-readout helper.
The full distributional WTA parity against live NEST lives in
``wang_decision_making_test.py`` (``@requires_nest``).
"""
import unittest

import jax
import brainstate
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from examples.nest.wang_decision_making import (build, run_decision,
                                                decision_from_rates)


class TestWangNoNest(unittest.TestCase):
    """The ported network builds, runs headless, and reads out a decision."""

    def test_tiny_run_produces_valid_decision_and_aligned_rates(self):
        """A tiny network runs end-to-end and yields a well-formed result dict."""
        out = run_decision(102.4, seed=1, ne=80, ni=20, T=700.0)
        self.assertIn(out['winner'], ('A', 'B', None))
        self.assertEqual(out['rate_a'].shape, out['rate_b'].shape)
        # Background drive alone (signal starts at 1000 ms, run ends at 700 ms) keeps
        # both rates finite and non-negative.
        self.assertTrue(np.all(np.isfinite(out['rate_a'])))
        self.assertGreaterEqual(float(out['rate_a'].min()), 0.0)

    def test_build_slices_populations_by_fraction(self):
        """``build`` carves selA/selB at the f-fraction and wires a runnable sim."""
        sim, rec = build(25.6, seed=3, ne=80, ni=20, T=300.0)
        self.assertEqual(rec['nA'], int(0.15 * 80))
        self.assertEqual(rec['selA'].size, rec['nA'])
        self.assertEqual(rec['selB'].size, rec['nA'])
        res = sim.simulate(50.0 * u.ms)   # short run just to confirm it lowers + runs
        self.assertIsNotNone(res)

    def test_decision_helper_picks_first_supra_threshold_winner(self):
        """The pure readout returns the population that first crosses threshold."""
        a = np.r_[np.zeros(50), np.full(50, 30.0)]
        b = np.zeros(100)
        out = decision_from_rates(a, b, dt=0.1, start_ms=0.0, thr_hz=15.0)
        self.assertEqual(out['winner'], 'A')
        self.assertAlmostEqual(out['t_decision'], 5.0, places=6)

    def test_decision_helper_returns_none_when_silent(self):
        """No crossing -> no winner, no decision time."""
        out = decision_from_rates(np.zeros(100), np.zeros(100), dt=0.1,
                                  start_ms=0.0, thr_hz=15.0)
        self.assertIsNone(out['winner'])
        self.assertIsNone(out['t_decision'])


if __name__ == '__main__':
    unittest.main()
