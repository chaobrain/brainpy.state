# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for the A-E tolerance conventions (NEST-free)."""
import dataclasses
import unittest

import saiunit as u

from brainpy_state._nest._validation import tolerance_conventions as tc


class TestToleranceConstants(unittest.TestCase):
    def test_trace_categories_are_unit_aware_mV(self):
        for cat in (tc.CAT_A, tc.CAT_B, tc.CAT_B_ALIGNED, tc.CAT_C):
            self.assertIsInstance(cat.atol, u.Quantity)
            self.assertEqual(u.get_unit(cat.atol), u.mV)

    def test_sim_defaults_carry_time_units(self):
        self.assertEqual(u.get_unit(tc.T_DEFAULT), u.ms)
        self.assertEqual(u.get_unit(tc.DT_DEFAULT), u.ms)
        self.assertEqual(tc.N_SEEDS_DEFAULT, 5)

    def test_category_values_match_doc_sec6(self):
        self.assertEqual(u.get_mantissa(tc.CAT_A.atol.to(u.mV)), 1e-3)
        self.assertEqual(tc.CAT_A.rtol, 1e-3)
        self.assertEqual(u.get_mantissa(tc.CAT_B.atol.to(u.mV)), 1e-6)
        self.assertEqual(tc.CAT_B.rtol, 1e-6)
        self.assertEqual(tc.CAT_B_ALIGNED.align_steps, 1)

    def test_distributional_category_D(self):
        self.assertEqual(tc.CAT_D.rate_rtol, 5e-2)
        self.assertEqual(tc.CAT_D.mean_diff_pct, 2e-2)
        self.assertEqual(tc.CAT_D.autocorr_max_diff, 5e-2)
        self.assertGreaterEqual(tc.CAT_D.n_seeds, 4)

    def test_rate_and_spike_categories(self):
        self.assertEqual(tc.CAT_C_RATE.rtol, 5e-2)          # mean-field rate fixed point
        self.assertEqual(tc.CAT_C_RATE.atol, 0.0)           # pure-relative (rates are plain floats)
        self.assertEqual(tc.CAT_E.max_count_diff, 2)
        self.assertEqual(tc.CAT_E.max_peak_step_diff, 1)

    def test_tolerances_are_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            tc.CAT_A.rtol = 1.0

    def test_labels_present(self):
        self.assertEqual(tc.CAT_A.label, "A")
        self.assertEqual(tc.CAT_D.label, "D")
        self.assertEqual(tc.CAT_E.label, "E")


if __name__ == "__main__":
    unittest.main()
