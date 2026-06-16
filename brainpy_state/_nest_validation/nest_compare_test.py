# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for the comparison engine (NEST-free; drives >90% coverage)."""
import unittest
from unittest import mock

import numpy as np
import brainunit as u

from brainpy_state._nest_validation import nest_compare as nc
from brainpy_state._nest_validation import tolerance_conventions as tc


class TestCompareTrace(unittest.TestCase):
    def test_identical_traces_pass(self):
        x = np.linspace(0.0, 10.0, 50)
        r = nc.compare_trace(x, x.copy(), tol=tc.CAT_A, metric="V_m")
        self.assertTrue(r.passed)

    def test_unit_aware_vm_within_atol(self):
        a = np.array([1.0, 2.0, 3.0]) * u.mV
        b = (np.array([1.0, 2.0, 3.0]) + 5e-4) * u.mV    # 0.5 uV < 1e-3 mV atol
        self.assertTrue(nc.compare_trace(a, b, tol=tc.CAT_A, metric="V_m").passed)

    def test_unit_aware_quantity_converts_units(self):
        a = np.array([1.0, 2.0]) * u.mV
        b = np.array([0.001, 0.002]) * u.volt             # same value, different unit
        self.assertTrue(nc.compare_trace(a, b, tol=tc.CAT_A, metric="V_m").passed)

    def test_failing_trace_flags_clear_diff(self):
        a = np.zeros(10)
        b = np.zeros(10)
        b[7] = 5.0                                        # big spike at index 7
        r = nc.compare_trace(a, b, tol=tc.CAT_B, metric="V_m")
        self.assertFalse(r.passed)
        self.assertIn("V_m", r.detail)
        self.assertIn("7", r.detail)                      # names the offending index
        self.assertAlmostEqual(r.error, 5.0)
        with self.assertRaises(AssertionError):
            r.assert_()

    def test_passing_assert_is_noop(self):
        nc.compare_trace(np.zeros(3), np.zeros(3), tol=tc.CAT_A).assert_()   # no raise

    def test_alignment_absorbs_one_step_shift(self):
        base = np.sin(np.linspace(0, 6, 60))
        shifted = np.roll(base, 1)
        no_align = tc.TraceTolerance(1e-9, 0.0)
        aligned = tc.TraceTolerance(1e-9, 0.0, align_steps=1)
        self.assertFalse(nc.compare_trace(base, shifted, tol=no_align).passed)
        self.assertTrue(nc.compare_trace(base, shifted, tol=aligned).passed)

    def test_pure_relative_scalar(self):                  # reproduces Siegert assertion
        self.assertTrue(nc.compare_trace(100.0, 104.0, tol=tc.CAT_C_RATE, metric="rate").passed)
        self.assertFalse(nc.compare_trace(100.0, 106.0, tol=tc.CAT_C_RATE).passed)

    def test_zero_reference_is_safe(self):
        r = nc.compare_trace(0.0, 0.0, tol=tc.CAT_C_RATE)
        self.assertTrue(r.passed)
        self.assertFalse(np.isnan(r.error))

    def test_plain_tol_strips_quantity_metric(self):
        # a rate given as a Hz quantity, compared with a plain-atol (rate) tolerance
        self.assertTrue(nc.compare_trace(100.0 * u.Hz, 104.0 * u.Hz,
                                         tol=tc.CAT_C_RATE, metric="rate").passed)   # 4% < 5%
        self.assertFalse(nc.compare_trace(100.0 * u.Hz, 110.0 * u.Hz, tol=tc.CAT_C_RATE).passed)

    def test_alignment_skips_empty_overlap(self):
        # align_steps larger than the trace: the +/-1 shifts give an empty overlap
        # (skipped); the zero-shift comparison still decides the result.
        tol = tc.TraceTolerance(1e-9, 0.0, align_steps=1)
        self.assertTrue(nc.compare_trace(np.array([1.0]), np.array([1.0]), tol=tol).passed)


class TestCompareDistributional(unittest.TestCase):
    def test_mean_within_rate_rtol(self):
        ref = [10.0, 11.0, 9.0, 10.0]
        cand = [10.2, 10.8, 9.4, 10.1]
        self.assertTrue(nc.compare_distributional(ref, cand, tol=tc.CAT_D, metric="rate").passed)

    def test_compares_means_not_per_sample(self):
        ref = [0.0, 20.0, 0.0, 20.0]                      # same mean, opposite ordering
        cand = [10.0, 10.0, 10.0, 10.0]
        self.assertTrue(nc.compare_distributional(ref, cand, tol=tc.CAT_D).passed)

    def test_mean_outside_rtol_fails_with_detail(self):
        r = nc.compare_distributional([10.0, 10.0], [12.0, 12.0], tol=tc.CAT_D, metric="rate")
        self.assertFalse(r.passed)
        self.assertIn("rate", r.detail)

    def test_reproducible_aggregate(self):
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        cand = [1.1, 2.1, 2.9, 4.05, 4.95]
        r1 = nc.compare_distributional(ref, cand, tol=tc.CAT_D)
        r2 = nc.compare_distributional(list(ref), list(cand), tol=tc.CAT_D)
        self.assertEqual(r1.error, r2.error)

    def test_zero_variance_zero_mean_no_divide_by_zero(self):
        r = nc.compare_distributional([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], tol=tc.CAT_D)
        self.assertTrue(r.passed)
        self.assertFalse(np.isnan(r.error))

    def test_unit_aware_samples(self):
        ref = [10.0 * u.Hz, 11.0 * u.Hz]
        cand = [10.5 * u.Hz, 10.5 * u.Hz]
        self.assertTrue(nc.compare_distributional(ref, cand, tol=tc.CAT_D).passed)

    def test_unsupported_statistic_raises(self):
        with self.assertRaises(ValueError):
            nc.compare_distributional([1.0], [1.0], tol=tc.CAT_D, statistic="median")


class TestCompareDistributionalAutocorr(unittest.TestCase):
    """The autocorr/covariance-function path (correlation-detector parity, 16 reuse)."""

    def test_identical_functions_pass(self):
        f = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        r = nc.compare_distributional(f, f.copy(), tol=tc.CAT_D,
                                      statistic="autocorr", metric="xcorr")
        self.assertTrue(r.passed)
        self.assertEqual(r.error, 0.0)

    def test_within_autocorr_bound_passes(self):
        f = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        g = f + 0.01                                       # 0.01 < autocorr_max_diff 0.05
        self.assertTrue(nc.compare_distributional(
            f, g, tol=tc.CAT_D, statistic="autocorr").passed)

    def test_out_of_bound_fails_and_names_lag(self):
        f = np.zeros(7)
        g = np.zeros(7)
        g[4] = 0.2                                         # 0.2 > 0.05 bound, at lag 4
        r = nc.compare_distributional(f, g, tol=tc.CAT_D,
                                      statistic="autocorr", metric="cov")
        self.assertFalse(r.passed)
        self.assertIn("cov", r.detail)
        self.assertIn("4", r.detail)                       # names the offending lag
        self.assertAlmostEqual(r.error, 0.2)
        with self.assertRaises(AssertionError):
            r.assert_()

    def test_per_seed_functions_are_averaged(self):
        # mean of each side is [0, 1, 0]; per-sample they differ but the means match.
        ref = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        cand = [np.array([0.0, 1.2, 0.0]), np.array([0.0, 0.8, 0.0])]
        self.assertTrue(nc.compare_distributional(
            ref, cand, tol=tc.CAT_D, statistic="autocorr").passed)

    def test_unit_aware_functions_stripped(self):
        f = np.array([0.0, 1.0, 0.0]) * u.mV
        g = np.array([0.0, 1.0, 0.0]) * u.mV
        self.assertTrue(nc.compare_distributional(
            f, g, tol=tc.CAT_D, statistic="autocorr").passed)

    def test_empty_functions_safe(self):
        # an empty detector window (no events) -> zero-length function, no crash
        r = nc.compare_distributional(np.array([]), np.array([]),
                                      tol=tc.CAT_D, statistic="autocorr")
        self.assertTrue(r.passed)
        self.assertEqual(r.error, 0.0)

    def test_mismatched_lengths_compare_shared_prefix(self):
        f = np.array([0.0, 1.0, 0.0, 9.0])                # extra lag ignored
        g = np.array([0.0, 1.0, 0.0])
        self.assertTrue(nc.compare_distributional(
            f, g, tol=tc.CAT_D, statistic="autocorr").passed)


class TestCompareDistributionalCV(unittest.TestCase):
    """The CV-of-ISI path: a normalized statistic compared on the seed mean."""

    def test_cv_within_bound_passes(self):
        ref = [0.95, 1.0, 1.05]                            # mean 1.0
        cand = [0.96, 1.0, 1.02]                           # mean 0.9933 -> 0.67% < 2%
        self.assertTrue(nc.compare_distributional(
            ref, cand, tol=tc.CAT_D, statistic="cv").passed)

    def test_cv_outside_bound_fails(self):
        r = nc.compare_distributional([1.0, 1.0], [1.1, 1.1], tol=tc.CAT_D,
                                      statistic="cv", metric="ISI CV")
        self.assertFalse(r.passed)                         # 10% > 2% (mean_diff_pct)
        self.assertIn("CV", r.detail)

    def test_cv_zero_reference_safe(self):
        r = nc.compare_distributional([0.0, 0.0], [0.0, 0.0], tol=tc.CAT_D, statistic="cv")
        self.assertTrue(r.passed)
        self.assertFalse(np.isnan(r.error))


class TestNestCompareDispatch(unittest.TestCase):
    def test_trace_mode_runs_both_callables(self):
        r = nc.nest_compare(lambda: np.arange(5.0), lambda: np.arange(5.0),
                            mode="trace", tol=tc.CAT_A, metric="x")
        self.assertTrue(r.passed)

    def test_distributional_mode_loops_seeds(self):
        r = nc.nest_compare(lambda s: 10.0, lambda s: 10.1,
                            mode="distributional", tol=tc.CAT_D, seeds=range(4))
        self.assertTrue(r.passed)

    def test_distributional_requires_seeds(self):
        with self.assertRaises(ValueError):
            nc.nest_compare(lambda s: 1.0, lambda s: 1.0, mode="distributional", tol=tc.CAT_D)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            nc.nest_compare(lambda: 1.0, lambda: 1.0, mode="bogus", tol=tc.CAT_A)


class TestRequiresNest(unittest.TestCase):
    def test_skips_when_nest_absent(self):
        with mock.patch.object(nc, "HAS_NEST", False):
            @nc.requires_nest
            def dummy():
                pass
            self.assertTrue(getattr(dummy, "__unittest_skip__", False))

    def test_does_not_skip_when_nest_present(self):
        with mock.patch.object(nc, "HAS_NEST", True):
            @nc.requires_nest
            def dummy():
                pass
            self.assertFalse(getattr(dummy, "__unittest_skip__", False))

    def test_decorates_testcase_class(self):
        with mock.patch.object(nc, "HAS_NEST", False):
            @nc.requires_nest
            class C(unittest.TestCase):
                def test_x(self):
                    pass
            self.assertTrue(getattr(C, "__unittest_skip__", False))


if __name__ == "__main__":
    unittest.main()
