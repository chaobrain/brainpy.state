# Shared live-NEST validation harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Gitignored scratch (CLAUDE.md #7) — the
> plan/spec are NOT committed.

**Goal:** Build `brainpy_state/_nest/_validation/{tolerance_conventions.py, nest_compare.py, conftest.py,
README.md}` — the shared NEST-parity harness — and refactor the 5 existing Brunel/device parity tests
onto it (same assertions), proving the API and locking the tolerance convention.

**Architecture:** Pure value-comparators (`compare_trace`, `compare_distributional`) + a callable-running
umbrella (`nest_compare(mode=…)`) + a `requires_nest` skip/marker decorator. Tolerances live as
unit-aware frozen-dataclass constants (categories A–E) in `tolerance_conventions.py`. Pass decisions use
the division-free numpy-allclose form `|a−b| ≤ atol + rtol·|ref|` (zero-reference safe). Comparators are
NEST-free ⇒ unit-tested without NEST for >90% coverage; the 5 refactored tests are the live-NEST proof.

**Tech Stack:** Python 3.13, pytest 9, `unittest.TestCase`, `saiunit` (units), `numpy`, live NEST 3.x,
`jax`/`brainstate` (brainpy side), `pytest-cov` for coverage.

**Conventions:** test files are `<name>_test.py`; every public symbol gets a NumPy-style docstring with a
doctest `Examples` block in `.. code-block:: python`; file header `# Copyright 2026 BrainX Ecosystem
Limited. Apache 2.0.`

---

## File structure

```
brainpy_state/_nest/_validation/
  tolerance_conventions.py        NEW   TraceTolerance/DistributionalTolerance/SpikeTimeTolerance + CAT_* + defaults
  nest_compare.py                 NEW   HAS_NEST, requires_nest, ComparisonResult, compare_trace, compare_distributional, nest_compare
  conftest.py                     NEW   register `requires_nest` marker
  README.md                       NEW   how-to + carve-outs + run with/without nest
  tolerance_conventions_test.py   NEW   constants + unit-aware (NEST-free)
  nest_compare_test.py            NEW   edge cases 1-6 (NEST-free)
  brunel_alpha_test.py            EDIT  thin glue
  brunel_delta_test.py            EDIT  thin glue
  brunel_exp_multisynapse_test.py EDIT  thin glue
  brunel_siegert_test.py          EDIT  thin glue
  device_parity_test.py           EDIT  thin glue
```

---

## Task 1: `tolerance_conventions.py` (TDD)

**Files:** Create `brainpy_state/_nest/_validation/tolerance_conventions.py`,
`brainpy_state/_nest/_validation/tolerance_conventions_test.py`.

- [ ] **Step 1 — failing test** `tolerance_conventions_test.py`:

```python
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
        self.assertEqual(tc.CAT_E.max_count_diff, 2)
        self.assertEqual(tc.CAT_E.max_peak_step_diff, 1)

    def test_tolerances_are_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            tc.CAT_A.rtol = 1.0


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2 — run, expect fail** `python -m pytest brainpy_state/_nest/_validation/tolerance_conventions_test.py -q` → ImportError/AttributeError.

- [ ] **Step 3 — implement** `tolerance_conventions.py` (one-screen module docstring summarising A–E +
  multi-seed protocol; frozen dataclasses; constants). Key content:

```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tolerance conventions for live-NEST parity — the single source of truth.

What tolerance for what kind of model. Categories A-E span the comparison space
(V_m trace, firing rate, weight trajectory, PSC-amplitude train, F-I curve):

==== ============================== =================== ============= =============
Cat  Kind                           Metric              Tolerance     Mode
==== ============================== =================== ============= =============
A    adaptive numerical integrator  aeif/HH/izh V_m     atol 1e-3 mV  trace
B    analytic exact propagator      linear iaf_psc V_m  atol 1e-6 mV  trace
C    conductance / coupled          iaf_cond / Siegert  1e-3 mV / 5%  trace
D    distributional (PRNG-divergent network rate, ISI CV mean 5% (>=4 seeds) distributional
E    spike-time / event-count       *_ps, PSC timing    |dN|<=2,|dt|<=1 trace
==== ============================== =================== ============= =============

A/B/C numbers are from ``numerical-validation-gap.md`` Sec 6. D extends the
unnumbered "distributional" default; E covers precise spiking. Multi-seed
protocol (D): run N>=N_SEEDS_DEFAULT seeds/side, compare the seed-MEAN, never
per-sample (PRNG streams diverge between NEST and JAX).
"""
import dataclasses

import saiunit as u

__all__ = ["TraceTolerance", "DistributionalTolerance", "SpikeTimeTolerance",
           "CAT_A", "CAT_B", "CAT_B_ALIGNED", "CAT_C", "CAT_C_RATE", "CAT_D", "CAT_E",
           "T_DEFAULT", "DT_DEFAULT", "N_SEEDS_DEFAULT"]


@dataclasses.dataclass(frozen=True)
class TraceTolerance:
    """Per-sample / max-abs tolerance for a deterministic trace (categories A, B, C).

    Parameters
    ----------
    atol : float or saiunit.Quantity
        Absolute tolerance. Unit-aware for voltage traces (e.g. ``1e-3 * u.mV``);
        a plain float for dimensionless / rate metrics.
    rtol : float
        Relative tolerance (dimensionless). Pass test is the numpy-allclose form
        ``|a - b| <= atol + rtol * |reference|`` (division-free).
    align_steps : int, optional
        Allow a ``+/- align_steps`` integer-sample shift search to absorb a
        recorder one-step offset. Default ``0`` (exact alignment).
    label, note : str
        Category label and human description for diagnostics.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> from brainpy_state._nest._validation.tolerance_conventions import CAT_B
        >>> u.get_unit(CAT_B.atol) == u.mV
        True
        >>> CAT_B.rtol
        1e-06
    """
    atol: object
    rtol: float
    align_steps: int = 0
    label: str = ""
    note: str = ""


@dataclasses.dataclass(frozen=True)
class DistributionalTolerance:
    """Multi-seed statistical tolerance (category D). See module docstring.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._nest._validation.tolerance_conventions import CAT_D
        >>> CAT_D.rate_rtol, CAT_D.n_seeds >= 4
        (0.05, True)
    """
    rate_rtol: float
    mean_diff_pct: float
    autocorr_max_diff: float
    n_seeds: int
    label: str = "D"
    note: str = ""


@dataclasses.dataclass(frozen=True)
class SpikeTimeTolerance:
    """Event-count / spike-time tolerance (category E).

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._nest._validation.tolerance_conventions import CAT_E
        >>> CAT_E.max_count_diff, CAT_E.max_peak_step_diff
        (2, 1)
    """
    max_count_diff: int
    max_peak_step_diff: int
    label: str = "E"
    note: str = ""


T_DEFAULT = 1000.0 * u.ms
DT_DEFAULT = 0.1 * u.ms
N_SEEDS_DEFAULT = 5

CAT_A = TraceTolerance(1e-3 * u.mV, 1e-3, label="A", note="adaptive RKF45 integrator trace")
CAT_B = TraceTolerance(1e-6 * u.mV, 1e-6, label="B", note="analytic exact-propagator trace (near-exact)")
CAT_B_ALIGNED = TraceTolerance(5e-2 * u.mV, 1e-3, align_steps=1, label="B",
                               note="analytic trace with one-step recorder alignment")
CAT_C = TraceTolerance(1e-3 * u.mV, 1e-3, label="C", note="conductance/coupled deterministic trace")
CAT_C_RATE = TraceTolerance(0.0, 5e-2, label="C", note="mean-field rate fixed point (Hz), deterministic")
CAT_D = DistributionalTolerance(5e-2, 2e-2, 5e-2, N_SEEDS_DEFAULT,
                                note="PRNG-divergent multi-seed statistic; compare aggregate, never per-sample")
CAT_E = SpikeTimeTolerance(2, 1, note="spike-time/event-count; +/-1 step ~ dt")
```

- [ ] **Step 4 — run, expect pass** (and doctests): `python -m pytest
  brainpy_state/_nest/_validation/tolerance_conventions_test.py --doctest-modules
  brainpy_state/_nest/_validation/tolerance_conventions.py -q` → PASS.

---

## Task 2: `nest_compare.py` (TDD — the engine)

**Files:** Create `brainpy_state/_nest/_validation/nest_compare.py`,
`brainpy_state/_nest/_validation/nest_compare_test.py`.

- [ ] **Step 1 — failing tests** `nest_compare_test.py` (covers goal edge cases 1-6; NEST-free):

```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for the comparison engine (NEST-free)."""
import unittest
from unittest import mock

import numpy as np
import saiunit as u

from brainpy_state._nest._validation import nest_compare as nc
from brainpy_state._nest._validation import tolerance_conventions as tc


class TestCompareTrace(unittest.TestCase):
    def test_identical_traces_pass(self):
        x = np.linspace(0.0, 10.0, 50)
        r = nc.compare_trace(x, x.copy(), tol=tc.CAT_A, metric="V_m")
        self.assertTrue(r.passed)

    def test_unit_aware_vm_within_atol(self):
        a = np.array([1.0, 2.0, 3.0]) * u.mV
        b = (np.array([1.0, 2.0, 3.0]) + 5e-4) * u.mV    # 0.5 uV < 1 mV atol
        self.assertTrue(nc.compare_trace(a, b, tol=tc.CAT_A, metric="V_m").passed)

    def test_failing_trace_flags_clear_diff(self):
        a = np.zeros(10)
        b = np.zeros(10); b[7] = 5.0                     # big spike at index 7
        r = nc.compare_trace(a, b, tol=tc.CAT_B, metric="V_m")
        self.assertFalse(r.passed)
        self.assertIn("V_m", r.detail)
        self.assertIn("7", r.detail)                     # names the offending index
        self.assertAlmostEqual(r.error, 5.0)
        with self.assertRaises(AssertionError):
            r.assert_()

    def test_alignment_absorbs_one_step_shift(self):
        base = np.sin(np.linspace(0, 6, 60))
        shifted = np.roll(base, 1)
        tol = tc.TraceTolerance(1e-9, 0.0, align_steps=1)
        self.assertFalse(nc.compare_trace(base, shifted, tol=tc.TraceTolerance(1e-9, 0.0)).passed)
        self.assertTrue(nc.compare_trace(base, shifted, tol=tol).passed)

    def test_pure_relative_scalar(self):                 # reproduces Siegert assertion
        r = nc.compare_trace(100.0, 104.0, tol=tc.CAT_C_RATE, metric="rate")
        self.assertTrue(r.passed)                        # 4% < 5%
        self.assertFalse(nc.compare_trace(100.0, 106.0, tol=tc.CAT_C_RATE).passed)

    def test_zero_reference_is_safe(self):
        r = nc.compare_trace(0.0, 0.0, tol=tc.CAT_C_RATE)
        self.assertTrue(r.passed)
        self.assertFalse(np.isnan(r.error))


class TestCompareDistributional(unittest.TestCase):
    def test_mean_within_rate_rtol(self):
        ref = [10.0, 11.0, 9.0, 10.0]
        cand = [10.2, 10.8, 9.4, 10.1]
        self.assertTrue(nc.compare_distributional(ref, cand, tol=tc.CAT_D, metric="rate").passed)

    def test_compares_means_not_per_sample(self):
        # wildly different per-sample ordering, identical means -> passes
        ref = [0.0, 20.0, 0.0, 20.0]
        cand = [10.0, 10.0, 10.0, 10.0]
        self.assertTrue(nc.compare_distributional(ref, cand, tol=tc.CAT_D).passed)

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


class TestNestCompareDispatch(unittest.TestCase):
    def test_trace_mode_runs_both_callables(self):
        r = nc.nest_compare(lambda: np.arange(5.0), lambda: np.arange(5.0),
                            mode="trace", tol=tc.CAT_A, metric="x")
        self.assertTrue(r.passed)

    def test_distributional_mode_loops_seeds(self):
        r = nc.nest_compare(lambda s: 10.0 + 0.0 * s, lambda s: 10.1,
                            mode="distributional", tol=tc.CAT_D, seeds=range(4))
        self.assertTrue(r.passed)

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
```

- [ ] **Step 2 — run, expect fail** `python -m pytest brainpy_state/_nest/_validation/nest_compare_test.py -q` → import error.

- [ ] **Step 3 — implement** `nest_compare.py`. Core logic (full module; NumPy docstrings + doctest
  Examples on every public symbol; only the bodies shown compactly here):

```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Comparison engine for live-NEST parity tests. See README.md."""
import dataclasses
import unittest

import numpy as np
import saiunit as u

try:
    import nest  # noqa: F401
    HAS_NEST = True
except Exception:
    HAS_NEST = False

try:
    import pytest
    _HAS_PYTEST = True
except Exception:
    _HAS_PYTEST = False

_NO_NEST = "live NEST not importable (install `nest-simulator`)"
_TINY = 1e-30
__all__ = ["HAS_NEST", "requires_nest", "ComparisonResult",
           "compare_trace", "compare_distributional", "nest_compare"]


def requires_nest(obj):
    """Skip a TestCase/method when NEST is unavailable; tag ``requires_nest`` marker.

    Reads ``HAS_NEST`` at call time (patchable in tests). Works on classes and methods.

    Examples
    --------
    .. code-block:: python

        >>> import unittest
        >>> from brainpy_state._nest._validation.nest_compare import requires_nest
        >>> @requires_nest
        ... class T(unittest.TestCase):
        ...     def test_parity(self): ...
        >>> isinstance(T, type)
        True
    """
    if _HAS_PYTEST:
        obj = pytest.mark.requires_nest(obj)
    return unittest.skipUnless(HAS_NEST, _NO_NEST)(obj)


@dataclasses.dataclass
class ComparisonResult:
    """Outcome of a parity comparison.

    Attributes: ``passed`` (bool), ``error`` (realized error), ``bound`` (tolerance
    bound), ``metric`` (label), ``detail`` (human diff). Call :meth:`assert_` to
    raise ``AssertionError(detail)`` on failure.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._nest._validation.nest_compare import ComparisonResult
        >>> ComparisonResult(True, 0.0, 1.0, "x", "ok").passed
        True
    """
    passed: bool
    error: float
    bound: float
    metric: str
    detail: str

    def assert_(self):
        if not self.passed:
            raise AssertionError(self.detail)


def _mantissas(reference, candidate, atol):
    """Return (ref, cand, atol) as plain float arrays in a common unit."""
    if isinstance(atol, u.Quantity):
        unit = atol.unit
        ref = np.asarray(u.Quantity(reference).to(unit).mantissa, dtype=float)
        cand = np.asarray(u.Quantity(candidate).to(unit).mantissa, dtype=float)
        atol_m = float(atol.mantissa)
    else:
        ref = np.asarray(u.get_mantissa(reference) if isinstance(reference, u.Quantity)
                         else reference, dtype=float)
        cand = np.asarray(u.get_mantissa(candidate) if isinstance(candidate, u.Quantity)
                          else candidate, dtype=float)
        atol_m = float(atol)
    return np.atleast_1d(ref), np.atleast_1d(cand), atol_m


def compare_trace(reference, candidate, *, tol, metric="trace"):
    """Deterministic per-sample / max-abs comparison (categories A/B/C).

    Pass test: ``max|a-b| <= atol + rtol*max|ref|`` (division-free, zero-ref safe).
    Scalars are treated as length-1 traces. If ``tol.align_steps > 0``, search
    integer shifts in ``[-align_steps, +align_steps]`` and keep the best overlap.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainpy_state._nest._validation import nest_compare as nc, tolerance_conventions as tc
        >>> nc.compare_trace(np.zeros(5), np.zeros(5), tol=tc.CAT_A).passed
        True
    """
    ref, cand, atol_m = _mantissas(reference, candidate, tol.atol)
    best = None
    shifts = range(-tol.align_steps, tol.align_steps + 1) if tol.align_steps else (0,)
    for s in shifts:
        if s == 0:
            a, b = ref, cand
        elif s > 0:
            a, b = ref[s:], cand[:cand.size - s]
        else:
            a, b = ref[:ref.size + s], cand[-s:]
        m = min(a.size, b.size)
        if m == 0:
            continue
        diff = np.abs(a[:m] - b[:m])
        idx = int(np.argmax(diff))
        err = float(diff[idx])
        if best is None or err < best[0]:
            best = (err, idx, float(np.max(np.abs(a[:m]))), s)
    err, idx, max_ref, shift = best
    bound = atol_m + tol.rtol * max_ref
    passed = err <= bound
    detail = (f"[{metric}] max|Δ|={err:.6g} at index {idx} (shift {shift}) "
              f"> bound {bound:.6g} (atol={atol_m:.3g}, rtol={tol.rtol:.3g}, "
              f"max|ref|={max_ref:.6g})") if not passed else f"[{metric}] ok (err={err:.3g})"
    return ComparisonResult(passed, err, bound, metric, detail)


def compare_distributional(reference_samples, candidate_samples, *, tol, metric="rate",
                           statistic="mean"):
    """Multi-seed statistical comparison: aggregate each side, then compare.

    NEVER per-sample (PRNG streams diverge). Pass test on the seed-mean:
    ``|mean(cand) - mean(ref)| <= rate_rtol*|mean(ref)|`` (division-free, zero-safe).
    ``error`` reports the guarded relative difference for readability.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._nest._validation import nest_compare as nc, tolerance_conventions as tc
        >>> nc.compare_distributional([10., 11., 9.], [10.1, 10.9, 9.1], tol=tc.CAT_D).passed
        True
    """
    if statistic != "mean":
        raise ValueError(f"unsupported statistic {statistic!r}")
    ref = np.asarray([u.get_mantissa(x) if isinstance(x, u.Quantity) else x
                      for x in reference_samples], dtype=float)
    cand = np.asarray([u.get_mantissa(x) if isinstance(x, u.Quantity) else x
                       for x in candidate_samples], dtype=float)
    rmean, cmean = float(np.mean(ref)), float(np.mean(cand))
    bound = tol.rate_rtol * abs(rmean)
    passed = abs(cmean - rmean) <= bound
    rel = abs(cmean - rmean) / max(abs(rmean), _TINY)
    detail = (f"[{metric}] mean cand={cmean:.6g} ref={rmean:.6g} rel={rel:.4g} "
              f"> rate_rtol {tol.rate_rtol:.3g} (n={ref.size}/{cand.size})") if not passed \
             else f"[{metric}] ok (rel={rel:.4g}, n={ref.size})"
    return ComparisonResult(passed, rel, tol.rate_rtol, metric, detail)


def nest_compare(nest_fn, brainpy_fn, *, mode, tol, metric="metric", seeds=None,
                 statistic="mean"):
    """Run ``nest_fn`` (live NEST) and ``brainpy_fn`` (brainpy.state) and compare.

    ``mode='trace'`` calls each once → :func:`compare_trace`.
    ``mode='distributional'`` (needs ``seeds``) calls each per seed →
    :func:`compare_distributional`.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainpy_state._nest._validation import nest_compare as nc, tolerance_conventions as tc
        >>> nc.nest_compare(lambda: np.zeros(3), lambda: np.zeros(3),
        ...                 mode="trace", tol=tc.CAT_A).passed
        True
    """
    if mode == "trace":
        return compare_trace(nest_fn(), brainpy_fn(), tol=tol, metric=metric)
    if mode == "distributional":
        if seeds is None:
            raise ValueError("distributional mode requires `seeds`")
        seeds = list(seeds)
        ref = [nest_fn(s) for s in seeds]
        cand = [brainpy_fn(s) for s in seeds]
        return compare_distributional(ref, cand, tol=tol, metric=metric, statistic=statistic)
    raise ValueError(f"unknown mode {mode!r} (expected 'trace' or 'distributional')")
```

- [ ] **Step 4 — run, expect pass** `python -m pytest brainpy_state/_nest/_validation/nest_compare_test.py
  --doctest-modules brainpy_state/_nest/_validation/nest_compare.py -q` → PASS.

- [ ] **Step 5 — commit** (harness core):
  `git add brainpy_state/_nest/_validation/{tolerance_conventions,nest_compare}*.py && git commit -m
  "feat(validation): shared NEST-parity harness (tolerances A-E + compare engine)"`

---

## Task 3: `conftest.py` — register the marker

**Files:** Create `brainpy_state/_nest/_validation/conftest.py`.

- [ ] **Step 1 — implement:**

```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""pytest config for the validation harness: register the `requires_nest` marker."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_nest: test needs a live NEST install (skipped otherwise)")
```

- [ ] **Step 2 — verify no unknown-mark warning:** `python -m pytest
  brainpy_state/_nest/_validation/nest_compare_test.py -q -W error::pytest.PytestUnknownMarkWarning`
  → PASS (the `requires_nest` mark applied inside tests does not warn).

---

## Task 4: `../../README.md`

**Files:** Create `brainpy_state/_nest/_validation/README.md`.

- [ ] **Step 1 — write** the README covering: (a) what the harness is; (b) the two modes; (c) the A–E
  table (link to `tolerance_conventions.py`); (d) a copy-paste parity-test skeleton using
  `@requires_nest` + `compare_trace`/`compare_distributional`; (e) **carve-out rules**: distributional
  (PRNG-divergent → compare seed-mean, never per-sample), docs-only (doctest, no parity run), blocked →
  skipped placeholder; (f) run instructions:
  - all: `python -m pytest brainpy_state/_nest/_validation/ -q`
  - NEST-only: `python -m pytest brainpy_state/_nest/_validation/ -m requires_nest -q`
  - without NEST: those tests skip cleanly; harness unit tests still run.
  - coverage: `python -m pytest brainpy_state/_nest/_validation/{tolerance_conventions,nest_compare}_test.py
    --cov=brainpy_state/_nest/_validation/tolerance_conventions --cov=brainpy_state/_nest/_validation/nest_compare
    --cov-report=term-missing -q`

- [ ] **Step 2 — commit** README.

---

## Task 5: Refactor the 5 tests onto the harness (thin glue)

For each: keep the `_nest_*`/`build`/`run` run logic **verbatim**; replace the `try/except import nest` +
`_HAS_NEST` + `@unittest.skipUnless(...)` with `from .nest_compare import requires_nest` + `@requires_nest`,
and replace the final rel-error/assert block with a `compare_*().assert_()`. Run each with NEST after.

- [ ] **5a `brunel_alpha_test.py`** — drop `try/except`/`_HAS_NEST`/`TOL`; `@requires_nest` on the class;
  body tail:

```python
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest._validation.tolerance_conventions import CAT_D
...
        nest_rate = _nest_rates(ORDER, SIMTIME)
        self.assertGreater(nest_rate, 0.0)
        compare_distributional([nest_rate], [bp_rate], tol=CAT_D, metric="exc rate").assert_()
```
  Run: `python -m pytest brainpy_state/_nest/_validation/brunel_alpha_test.py -q` → PASS.

- [ ] **5b `brunel_delta_test.py`** — identical pattern. Run → PASS.

- [ ] **5c `brunel_exp_multisynapse_test.py`** — keep the 4-seed loop building `bp`/`ns` lists; replace
  tail with `compare_distributional(ns, bp, tol=CAT_D, metric="exc rate").assert_()`. Run → PASS.

- [ ] **5d `brunel_siegert_test.py`** — `@requires_nest`; replace the per-(exc,inh) loop with:

```python
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import CAT_C_RATE
...
        for label, bp, ns in (("exc", erate, nest_ex), ("inh", irate, nest_in)):
            self.assertGreater(ns, 0.0)
            compare_trace(ns, bp, tol=CAT_C_RATE, metric=f"{label} mean-field rate").assert_()
```
  Run → PASS.

- [ ] **5e `device_parity_test.py`** — `from .nest_compare import requires_nest, compare_distributional`;
  `from .tolerance_conventions import CAT_D`. (i) `TestPoissonRate` (brainpy-only) tail →
  `compare_distributional([rate_hz], [emp_rate], tol=CAT_D, metric="poisson rate").assert_()`.
  (ii) `TestPoissonRateVsNest` → `@requires_nest`, tail →
  `compare_distributional([1000.0], [nest_rate], tol=CAT_D, metric="nest poisson rate").assert_()`.
  (iii) `TestSpikeRecorderStamp` **unchanged**. Run: `python -m pytest
  brainpy_state/_nest/_validation/device_parity_test.py -q` → PASS.

- [ ] **5f — full directory run** `python -m pytest brainpy_state/_nest/_validation/ -q` → all PASS
  (incl. the 2 out-of-scope tests still green). **Commit** the 5 refactors:
  `git commit -am "refactor(validation): port 5 Brunel/device parity tests onto the shared harness"`

---

## Task 6: Coverage + skip verification

- [ ] **Step 1 — coverage** on the two harness modules (must be >90%):

Run: `python -m pytest brainpy_state/_nest/_validation/tolerance_conventions_test.py
brainpy_state/_nest/_validation/nest_compare_test.py
--cov=brainpy_state/_nest/_validation/tolerance_conventions
--cov=brainpy_state/_nest/_validation/nest_compare --cov-report=term-missing -q`
Expected: both files >90%. If any line uncovered (e.g. a branch in `_mantissas`/alignment), add a
targeted unit test and re-run.

- [ ] **Step 2 — skip path** (simulate NEST absent without uninstalling): the
  `TestRequiresNest.test_skips_when_nest_absent` already proves the decorator skips when
  `HAS_NEST=False`. Document in the Lessons entry that with NEST present the 5 tests run; without it they
  skip via the same decorator.

---

## Task 7: Lessons entry + git lifecycle

- [ ] **Step 1 — append** a Part-5 Lessons entry to `CONTEXT.md` (newest at top), per the template:
  Shipped (files, PR #), Parity (the 5 tests' metrics), API discovered (`compare_trace` /
  `compare_distributional` / `nest_compare(mode=)` / `requires_nest` signatures the next clusters import),
  Gotchas (division-free allclose form; `requires_nest` reads `HAS_NEST` at call time; rates are plain
  floats not Quantities; recorder one-step alignment), For next clusters (01 uses `compare_trace`+CAT_A for
  PSC-amplitude train; 02 uses CAT_B/CAT_B_ALIGNED for V_m + CAT_E for F-I/spike timing).

- [ ] **Step 2 — commit** `git commit -am "docs(context): Lessons entry for 00-validation-harness"`.

- [ ] **Step 3 — push** to a clean remote branch name (local worktree branch is harness-managed):
  `git push -u origin HEAD:nest-goal/00-validation-harness`.

- [ ] **Step 4 — PR** `gh pr create --base main --head nest-goal/00-validation-harness
  --title "feat(validation): shared live-NEST parity harness + refactor 5 Brunel/device tests"
  --body "<scope; the 5 tests still pass; >90% harness coverage>"` (no `Co-Authored-By`).

- [ ] **Step 5 — merge** `gh pr merge --squash --delete-branch`.

- [ ] **Step 6 — sync main + remove worktree** `git switch main` (in main checkout) `&& git fetch origin
  && git merge --ff-only origin/main`; then `ExitWorktree(action="remove")` (or `git worktree remove`).

---

## Self-review (writing-plans)

- **Spec coverage:** §4 API → Tasks 1-2; §4.3 conftest → Task 3; §5 A–E → Task 1 constants; §6 modes →
  Task 2 `compare_trace`/`compare_distributional`; §7 5-test mapping → Task 5; §8 edge cases 1-6 → Task 2
  tests; README → Task 4; coverage → Task 6; Lessons + git → Task 7. No gaps.
- **Placeholder scan:** README body bullets are specified (content listed); all code steps contain real
  code. No TBD/TODO.
- **Type consistency:** `ComparisonResult(passed, error, bound, metric, detail)` + `.assert_()` used
  identically across Task 2 impl, tests, and Task 5 call sites. `compare_distributional(ref, cand, *,
  tol, metric, statistic)` and `compare_trace(ref, cand, *, tol, metric)` signatures consistent. `CAT_*`
  names consistent with Task 1.
```
