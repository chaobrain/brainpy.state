# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Comparison engine for live-NEST parity tests.

Run a metric in **live NEST** and in **brainpy.state**, then compare it under an
explicit tolerance (:mod:`brainpy_state._nest._validation.tolerance_conventions`)
and an explicit comparison **mode**:

* ``trace`` — deterministic drive, compared per-sample / max-abs (with optional
  integer-step alignment for a recorder offset). Categories A/B/C, plus scalar
  fixed points such as ``siegert_neuron``.
* ``distributional`` — PRNG-divergent drive, compared as a seed-aggregated
  statistic (the mean), **never per-sample**. Category D.

The core comparators take already-computed metric values (plain floats / arrays
or :mod:`saiunit` quantities), so they are pure and unit-testable without NEST.
:func:`nest_compare` is the convenience that *runs* the two callables first.

See ``brainpy_state/_nest/_validation/README.md`` for how to write a parity test.
"""
import dataclasses
import unittest

import numpy as np
import saiunit as u

try:
    import nest  # noqa: F401
    HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    HAS_NEST = False

try:
    import pytest
    _HAS_PYTEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_PYTEST = False

_NO_NEST = "live NEST not importable (install `nest-simulator`)"
_TINY = 1e-30
__all__ = [
    "HAS_NEST", "requires_nest", "ComparisonResult",
    "compare_trace", "compare_distributional", "nest_compare",
]


def requires_nest(obj):
    """Skip a ``TestCase`` class/method when NEST is unavailable; tag the marker.

    The skip condition reads :data:`HAS_NEST` at call time (so tests can patch it),
    and the ``requires_nest`` pytest marker is attached for ``pytest -m requires_nest``
    selection. Works on both classes and methods.

    Parameters
    ----------
    obj : type or callable
        A ``unittest.TestCase`` subclass or a test method.

    Returns
    -------
    type or callable
        ``obj``, marked and conditionally skipped.

    Examples
    --------
    .. code-block:: python

        >>> import unittest
        >>> from brainpy_state._nest._validation.nest_compare import requires_nest
        >>> @requires_nest
        ... class T(unittest.TestCase):
        ...     def test_parity(self):
        ...         ...
        >>> isinstance(T, type)
        True
    """
    if _HAS_PYTEST:
        obj = pytest.mark.requires_nest(obj)
    return unittest.skipUnless(HAS_NEST, _NO_NEST)(obj)


@dataclasses.dataclass
class ComparisonResult:
    """Outcome of a parity comparison.

    Parameters
    ----------
    passed : bool
        Whether the comparison met its tolerance.
    error : float
        The realized error (max-abs for traces, guarded relative for distributions).
    bound : float
        The tolerance bound the error was tested against.
    metric : str
        Human label of the compared quantity (e.g. ``"V_m"``, ``"exc rate"``).
    detail : str
        Human-readable diff, used as the assertion message on failure.

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
        """Raise ``AssertionError(self.detail)`` when the comparison failed."""
        if not self.passed:
            raise AssertionError(self.detail)


def _to_unit(x, unit):
    """Return ``x`` as a plain float array in ``unit`` (plain input assumed in ``unit``)."""
    if isinstance(x, u.Quantity):
        return np.atleast_1d(np.asarray(x.to(unit).mantissa, dtype=float))
    return np.atleast_1d(np.asarray(x, dtype=float))


def _bare(x):
    """Return ``x`` as a plain float array, stripping any unit."""
    if isinstance(x, u.Quantity):
        x = u.get_mantissa(x)
    return np.atleast_1d(np.asarray(x, dtype=float))


def _samples_to_array(samples):
    """Strip units element-wise from a per-seed sample sequence to a float array."""
    return np.atleast_1d(np.asarray(
        [float(u.get_mantissa(x)) if isinstance(x, u.Quantity) else float(x)
         for x in samples], dtype=float))


def _stack_functions(x):
    """Coerce a correlation/covariance *function* to a 2-D ``(n_seeds, n_lags)`` array.

    Accepts a single function (a 1-D sequence / :mod:`saiunit` quantity over lags)
    -> ``(1, n_lags)``, or a sequence of per-seed functions -> ``(n_seeds, n_lags)``.
    Units are stripped element-wise (a covariance function is compared on its bare
    magnitude). Used by the ``autocorr`` statistic.
    """
    if isinstance(x, u.Quantity):
        m = np.atleast_1d(np.asarray(u.get_mantissa(x), dtype=float))
        return m[None, :] if m.ndim == 1 else m
    seq = list(x)
    if seq and isinstance(seq[0], (u.Quantity, np.ndarray, list, tuple)):
        return np.asarray([_bare(f) for f in seq], dtype=float)   # per-seed functions
    return _bare(seq)[None, :]                                     # single function


def _seed_mean_compare(reference_samples, candidate_samples, *, bound_rtol, metric):
    """Seed-aggregated relative comparison of two scalar samples (mean / CV paths).

    The pass test on the seed mean is ``|mean(cand) - mean(ref)| <= bound_rtol *
    |mean(ref)|`` (division-free, so an all-zero / zero-variance reference is safe);
    ``error`` is the guarded relative difference. Shared by the ``mean`` and ``cv``
    statistics, which differ only in the tolerance they pass as ``bound_rtol``.
    """
    ref = _samples_to_array(reference_samples)
    cand = _samples_to_array(candidate_samples)
    rmean, cmean = float(np.mean(ref)), float(np.mean(cand))
    bound = bound_rtol * abs(rmean)
    passed = bool(abs(cmean - rmean) <= bound)
    rel = abs(cmean - rmean) / max(abs(rmean), _TINY)
    if passed:
        detail = f"[{metric}] ok (rel={rel:.4g} <= {bound_rtol:.3g}, n={ref.size})"
    else:
        detail = (f"[{metric}] mean cand={cmean:.6g} ref={rmean:.6g} rel={rel:.4g} "
                  f"> rtol {bound_rtol:.3g} (n={ref.size}/{cand.size})")
    return ComparisonResult(passed, rel, bound_rtol, metric, detail)


def _mantissas(reference, candidate, atol):
    """Reduce (reference, candidate, atol) to plain float arrays in a common basis."""
    if isinstance(atol, u.Quantity):
        unit = atol.unit
        return _to_unit(reference, unit), _to_unit(candidate, unit), float(atol.mantissa)
    return _bare(reference), _bare(candidate), float(atol)


def compare_trace(reference, candidate, *, tol, metric="trace"):
    """Deterministic per-sample / max-abs comparison (categories A/B/C).

    The pass test is the division-free numpy-allclose form
    ``max|a - b| <= atol + rtol * max|reference|``, which reproduces both a
    pure-absolute (``rtol=0``) and a pure-relative (``atol=0``) assertion and never
    divides by a zero reference. Scalars are treated as length-1 traces. When
    ``tol.align_steps > 0`` the comparison searches integer shifts in
    ``[-align_steps, +align_steps]`` and keeps the best overlap (recorder offset).

    Parameters
    ----------
    reference, candidate : float, array, or saiunit.Quantity
        The metric from NEST (``reference``) and brainpy.state (``candidate``). A
        plain array paired with a unit-aware ``tol.atol`` is assumed already in that
        unit; a quantity is converted.
    tol : TraceTolerance
        Tolerance (e.g. ``tolerance_conventions.CAT_A``).
    metric : str, optional
        Label for diagnostics.

    Returns
    -------
    ComparisonResult

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainpy_state._nest._validation import nest_compare as nc
        >>> from brainpy_state._nest._validation import tolerance_conventions as tc
        >>> nc.compare_trace(np.zeros(5), np.zeros(5), tol=tc.CAT_A).passed
        True
        >>> nc.compare_trace(100.0, 104.0, tol=tc.CAT_C_RATE).passed   # 4 % < 5 %
        True
    """
    ref, cand, atol_m = _mantissas(reference, candidate, tol.atol)
    shifts = range(-tol.align_steps, tol.align_steps + 1) if tol.align_steps else (0,)
    best = None
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
        max_ref = float(np.max(np.abs(a[:m])))
        if best is None or err < best[0]:
            best = (err, idx, max_ref, s)
    err, idx, max_ref, shift = best
    bound = atol_m + tol.rtol * max_ref
    passed = bool(err <= bound)
    if passed:
        detail = f"[{metric}] ok (max|Δ|={err:.3g} <= {bound:.3g})"
    else:
        detail = (f"[{metric}] max|Δ|={err:.6g} at index {idx} (shift {shift}) "
                  f"> bound {bound:.6g} (atol={atol_m:.3g}, rtol={tol.rtol:.3g}, "
                  f"max|ref|={max_ref:.6g})")
    return ComparisonResult(passed, err, bound, metric, detail)


def compare_distributional(reference_samples, candidate_samples, *, tol, metric="rate",
                           statistic="mean"):
    """Multi-seed statistical comparison: aggregate each side, then compare.

    PRNG streams diverge between NEST and JAX, so this **never** compares
    per-sample. Three statistics are supported, selected by ``statistic``:

    * ``"mean"`` (default) — seed-mean firing-rate / scalar parity. Pass test
      ``|mean(cand) - mean(ref)| <= rate_rtol * |mean(ref)|`` (division-free, so an
      all-zero / zero-variance reference is safe); ``error`` is the guarded
      relative difference.
    * ``"cv"`` — coefficient-of-variation (e.g. of ISIs). Same seed-mean test as
      ``"mean"`` but against the tighter ``mean_diff_pct`` bound (CV is already a
      normalized statistic).
    * ``"autocorr"`` — auto-/cross-correlation or covariance **functions** (1-D
      over lags) from the correlation detectors. Each side is a single function or
      a per-seed sequence of functions; they are seed-averaged and compared
      element-wise, ``max_lag |cand - ref| <= autocorr_max_diff`` (absolute).

    Parameters
    ----------
    reference_samples, candidate_samples : sequence of float / saiunit.Quantity, or function(s)
        For ``"mean"``/``"cv"``: one metric value per seed, from NEST
        (``reference``) and brainpy.state (``candidate``). For ``"autocorr"``: a
        single correlation/covariance function (1-D over lags) or a sequence of
        per-seed functions.
    tol : DistributionalTolerance
        Tolerance (e.g. ``tolerance_conventions.CAT_D``). ``"mean"`` reads
        ``rate_rtol``, ``"cv"`` reads ``mean_diff_pct``, ``"autocorr"`` reads
        ``autocorr_max_diff``.
    metric : str, optional
        Label for diagnostics.
    statistic : {"mean", "cv", "autocorr"}, optional
        Aggregation statistic. Default ``"mean"``.

    Returns
    -------
    ComparisonResult

    Raises
    ------
    ValueError
        If ``statistic`` is not one of ``"mean"``, ``"cv"``, ``"autocorr"``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainpy_state._nest._validation import nest_compare as nc
        >>> from brainpy_state._nest._validation import tolerance_conventions as tc
        >>> nc.compare_distributional([10., 11., 9.], [10.1, 10.9, 9.1], tol=tc.CAT_D).passed
        True
        >>> f = np.array([0.0, 0.5, 1.0, 0.5, 0.0])     # a covariance function
        >>> nc.compare_distributional(f, f + 0.01, tol=tc.CAT_D, statistic="autocorr").passed
        True
    """
    if statistic == "mean":
        return _seed_mean_compare(reference_samples, candidate_samples,
                                  bound_rtol=tol.rate_rtol, metric=metric)
    if statistic == "cv":
        # Coefficient of variation is already a normalized statistic; compare the
        # seed mean within the (tighter) per-cell mean tolerance.
        return _seed_mean_compare(reference_samples, candidate_samples,
                                  bound_rtol=tol.mean_diff_pct, metric=metric)
    if statistic == "autocorr":
        # Correlation/covariance *functions* (1-D over lags), seed-averaged, then
        # compared element-wise against the absolute autocorrelation tolerance.
        ref = _stack_functions(reference_samples).mean(axis=0)
        cand = _stack_functions(candidate_samples).mean(axis=0)
        n = min(ref.size, cand.size)
        diff = np.abs(ref[:n] - cand[:n])
        idx = int(np.argmax(diff)) if n else 0
        err = float(diff[idx]) if n else 0.0
        bound = float(tol.autocorr_max_diff)
        passed = bool(err <= bound)
        if passed:
            detail = f"[{metric}] ok (max|Δ|={err:.4g} <= {bound:.3g}, lags={n})"
        else:
            detail = (f"[{metric}] max|Δ|={err:.6g} at lag {idx} "
                      f"> autocorr_max_diff {bound:.3g} (lags={n})")
        return ComparisonResult(passed, err, bound, metric, detail)
    raise ValueError(
        f"unsupported statistic {statistic!r} (expected 'mean', 'cv', or 'autocorr')")


def nest_compare(nest_fn, brainpy_fn, *, mode, tol, metric="metric", seeds=None,
                 statistic="mean"):
    """Run ``nest_fn`` (live NEST) and ``brainpy_fn`` (brainpy.state) and compare.

    Parameters
    ----------
    nest_fn, brainpy_fn : callable
        In ``trace`` mode, called with no arguments and returning the metric. In
        ``distributional`` mode, called once per seed (``fn(seed)``).
    mode : {"trace", "distributional"}
        Comparison mode.
    tol : TraceTolerance or DistributionalTolerance
        Tolerance matching the mode.
    metric : str, optional
        Label for diagnostics.
    seeds : iterable, optional
        Required for ``distributional`` mode: the seeds to average over.
    statistic : str, optional
        Distributional aggregation statistic (default ``"mean"``).

    Returns
    -------
    ComparisonResult

    Raises
    ------
    ValueError
        If ``mode`` is unknown, or ``distributional`` mode is missing ``seeds``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainpy_state._nest._validation import nest_compare as nc
        >>> from brainpy_state._nest._validation import tolerance_conventions as tc
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
