# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spike-train statistics and the multi-seed agreement rule of gates G2 and G3.

Every function takes spike events as ``(times_ms, senders)`` with ``senders`` local
neuron indices in ``[0, n)``, so NEST, Brian2 and the kernel feed the same code.
Definitions (spec ``docs/specs/2026-10-22-k0-t10-gates.md`` §2):

* :func:`rate` -- spikes per neuron and second;
* :func:`cv_isi` -- mean over neurons with at least 3 spikes of ``std(ISI) / mean(ISI)``;
* :func:`fano` -- per-neuron ``var / mean`` of the counts in ``bin_ms`` bins, mean over
  neurons with a non-zero mean;
* :func:`mean_correlation` -- mean pairwise Pearson correlation of the binned counts
  over pairs whose counts both vary;
* :func:`g2_rule` -- ``|Δmean| <= max(rel * |reference mean|, 2 * combined SEM)``.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np

__all__ = ['window', 'rate', 'cv_isi', 'binned_counts', 'fano', 'mean_correlation',
           'network_stats', 'G2Result', 'g2_rule', 'format_g2_table']


def window(times: Any, senders: Any, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Events with ``t0 <= t < t1`` (ms)."""
    t = np.asarray(times, float).reshape(-1)
    s = np.asarray(senders, np.int64).reshape(-1)
    if t.shape != s.shape:
        raise ValueError(f'times and senders differ in length: {t.shape} vs {s.shape}')
    keep = (t >= t0) & (t < t1)
    return t[keep], s[keep]


def rate(times: Any, senders: Any, n: int, duration_ms: float) -> float:
    """Mean rate (Hz) of ``n`` neurons over ``duration_ms``."""
    if n <= 0 or duration_ms <= 0:
        raise ValueError('n and duration_ms must be positive')
    return float(np.asarray(times).size) / n / (duration_ms * 1e-3)


def cv_isi(times: Any, senders: Any, n: int, min_spikes: int = 3) -> float:
    """Mean coefficient of variation of the inter-spike intervals (``nan`` if no neuron qualifies)."""
    t = np.asarray(times, float)
    s = np.asarray(senders, np.int64)
    order = np.lexsort((t, s))
    t, s = t[order], s[order]
    cvs = []
    starts = np.searchsorted(s, np.arange(n))
    stops = np.searchsorted(s, np.arange(n), side='right')
    for a, b in zip(starts, stops):
        if b - a >= min_spikes:
            isi = np.diff(t[a:b])
            m = isi.mean()
            if m > 0:
                cvs.append(isi.std() / m)
    return float(np.mean(cvs)) if cvs else float('nan')


def binned_counts(times: Any, senders: Any, n: int, t0: float, t1: float,
                  bin_ms: float = 10.0) -> np.ndarray:
    """Spike counts, shape ``(n, n_bins)``, in bins of ``bin_ms`` over ``[t0, t1)``."""
    n_bins = int(round((t1 - t0) / bin_ms))
    if n_bins < 1:
        raise ValueError('the window holds no complete bin')
    t, s = window(times, senders, t0, t0 + n_bins * bin_ms)
    b = np.minimum(((t - t0) / bin_ms).astype(np.int64), n_bins - 1)
    out = np.zeros((n, n_bins), np.int64)
    np.add.at(out, (s, b), 1)
    return out


def fano(counts: np.ndarray) -> float:
    """Mean per-neuron Fano factor of binned counts ``(n, n_bins)`` (``nan`` if all silent)."""
    c = np.asarray(counts, float)
    m = c.mean(axis=1)
    keep = m > 0
    if not keep.any():
        return float('nan')
    return float((c[keep].var(axis=1) / m[keep]).mean())


def mean_correlation(counts: np.ndarray) -> float:
    """Mean pairwise Pearson correlation of binned counts (``nan`` if < 2 neurons vary)."""
    c = np.asarray(counts, float)
    c = c[c.std(axis=1) > 0]
    if c.shape[0] < 2:
        return float('nan')
    r = np.corrcoef(c)
    iu = np.triu_indices(c.shape[0], k=1)
    return float(r[iu].mean())


def network_stats(times: Any, senders: Any, n_e: int, n_i: int, t0: float, t1: float,
                  n_corr: int = 50, bin_ms: float = 10.0) -> dict:
    """The G2/G3 statistics of one run (E = senders ``[0, n_e)``, I = ``[n_e, n_e + n_i)``).

    Returns
    -------
    dict
        ``rate_E``, ``rate_I`` (Hz), ``cv_isi_E``, ``fano_E``, ``corr_E`` (first
        ``n_corr`` E neurons).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainspike._kernel.validation.stats import network_stats
        >>> t = np.tile(np.arange(5.0, 1000.0, 10.0), 3)
        >>> s = np.repeat([0, 1, 2], 100)
        >>> st = network_stats(t, s, n_e=2, n_i=1, t0=0.0, t1=1000.0, n_corr=2)
        >>> st['rate_E'], st['rate_I'], st['cv_isi_E']
        (100.0, 100.0, 0.0)
    """
    t, s = window(times, senders, t0, t1)
    dur = t1 - t0
    e = s < n_e
    i = (s >= n_e) & (s < n_e + n_i)
    te, se = t[e], s[e]
    counts = binned_counts(te, se, n_e, t0, t1, bin_ms)
    return {
        'rate_E': rate(te, se, n_e, dur),
        'rate_I': rate(t[i], s[i] - n_e, n_i, dur),
        'cv_isi_E': cv_isi(te, se, n_e),
        'fano_E': fano(counts),
        'corr_E': mean_correlation(counts[:min(n_corr, n_e)]),
    }


@dataclasses.dataclass
class G2Result:
    """Outcome of :func:`g2_rule` for one statistic."""
    metric: str
    ours_mean: float
    ref_mean: float
    ours_sem: float
    ref_sem: float
    bound: float
    passed: bool

    @property
    def diff(self) -> float:
        return self.ours_mean - self.ref_mean


def g2_rule(ours: Sequence[float], reference: Sequence[float], metric: str = '',
            rel: float = 0.05) -> G2Result:
    """``|mean(ours) - mean(ref)| <= max(rel * |mean(ref)|, 2 * sqrt(SEM_ours^2 + SEM_ref^2))``.

    Parameters
    ----------
    ours, reference : sequence of float
        One value per seed (at least two each; ``nan`` values fail the rule).
    metric : str
        Label.
    rel : float, default 0.05
        Relative part of the bound.

    Returns
    -------
    G2Result

    Examples
    --------
    .. code-block:: python

        >>> from brainspike._kernel.validation.stats import g2_rule
        >>> g2_rule([10.0, 10.2, 9.9], [10.1, 9.8, 10.0]).passed
        True
        >>> g2_rule([12.0, 12.1, 11.9], [10.0, 10.1, 9.9]).passed
        False
    """
    a = np.asarray(ours, float)
    b = np.asarray(reference, float)
    if a.size < 2 or b.size < 2:
        raise ValueError('need at least two seeds on each side')
    sem_a = a.std(ddof=1) / np.sqrt(a.size)
    sem_b = b.std(ddof=1) / np.sqrt(b.size)
    bound = max(rel * abs(b.mean()), 2.0 * float(np.hypot(sem_a, sem_b)))
    diff = abs(a.mean() - b.mean())
    passed = bool(np.isfinite(diff) and diff <= bound)
    return G2Result(metric, float(a.mean()), float(b.mean()), float(sem_a), float(sem_b),
                    float(bound), passed)


def format_g2_table(results: Sequence[G2Result], ours: str = 'kernel', ref: str = 'NEST') -> str:
    """Markdown table of :func:`g2_rule` results."""
    lines = [f'| statistic | {ours} mean ± SEM | {ref} mean ± SEM | |Δ| | bound | pass |',
             '|---|---|---|---|---|---|']
    for r in results:
        lines.append(f'| {r.metric} | {r.ours_mean:.4g} ± {r.ours_sem:.2g} | '
                     f'{r.ref_mean:.4g} ± {r.ref_sem:.2g} | {abs(r.diff):.3g} | {r.bound:.3g} | '
                     f'{"pass" if r.passed else "FAIL"} |')
    return '\n'.join(lines)
