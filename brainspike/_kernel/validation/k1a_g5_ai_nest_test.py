# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate K1a-G5: G2 in the asynchronous-irregular regime vs live NEST (``requires_nest``).

Brunel ``iaf_psc_alpha`` (the network of :mod:`.g2_brunel_nest_test`) at the AI point
chosen by the lead on 2026-10-22 from the NEST-only scan
(``benchmarks/k1a_g5_ai_scan.py``, results ``benchmarks/results/2026-10-22-K1a-G5-ai-scan-*.json``):
**order 2500, g = 16, eta = 4** (NEST: rate ~10.6 Hz, CV-ISI ~0.65, Fano ~0.89, pairwise
correlation ~0.006). Order 2500 because no point at G2's order 200 reaches CV-ISI >= 0.5
by g / eta alone (best 0.43) and order 1000 is marginal (0.51).

Protocol as G2: every neuron recorded, ``T_WARM = 100 ms`` discarded, statistics over the
next ``T_STAT = 1000 ms``, 10 seeds each side, kernel float32, rule
``|Δmean| <= max(5 % of the NEST mean, 2 x combined SEM)`` on rate E/I, CV-ISI, Fano
factor and pairwise correlation. One test computes everything (one process: the run is
~10 min, it must not be repeated per parametrised case under ``pytest-xdist``).

Run as a script to print the table::

    python -m brainspike._kernel.validation.k1a_g5_ai_nest_test
"""
from __future__ import annotations

import functools

import numpy as np

from brainspike._kernel.validation.brunel import brunel_params
from brainspike._kernel.validation.g2_brunel_nest_test import kernel_brunel, nest_brunel
from brainspike._kernel.validation.stats import format_g2_table, g2_rule, network_stats
from brainspike._nest_validation.nest_compare import requires_nest

ORDER = 2500
G = 16.0
ETA = 4.0
SEEDS = tuple(range(1, 11))
T_WARM = 100.0
T_STAT = 1000.0
NEST_THREADS = 4
STATS = ('rate_E', 'rate_I', 'cv_isi_E', 'fano_E', 'corr_E')


def run_stats(sim: str, order: int = ORDER, seeds=SEEDS, g: float = G, eta: float = ETA) -> dict:
    """``{statistic: [value per seed]}`` for ``sim`` in {'nest', 'kernel'}."""
    p = brunel_params(order, g=g, eta=eta)
    out = {k: [] for k in STATS}
    for s in seeds:
        if sim == 'nest':
            t, snd = nest_brunel(order, s, T_WARM + T_STAT, g=g, eta=eta, threads=NEST_THREADS)
        else:
            t, snd = kernel_brunel(order, s, T_WARM + T_STAT, g=g, eta=eta)
        st = network_stats(t, snd, p['N_E'], p['N_I'], T_WARM, T_WARM + T_STAT)
        for k in STATS:
            out[k].append(st[k])
    return out


@functools.lru_cache(maxsize=None)
def g5_results() -> tuple:
    """G2 rule per statistic at the AI point (computed once per process)."""
    ref = run_stats('nest')
    ours = run_stats('kernel')
    return tuple(g2_rule(ours[k], ref[k], k) for k in STATS), ours, ref


@requires_nest
def test_k1a_g5_ai_regime_statistics_match_nest():
    results, ours, ref = g5_results()
    print('\n' + format_g2_table(results))
    for k in STATS:
        assert len(ours[k]) == len(ref[k]) == len(SEEDS)
    by = {r.metric: r for r in results}
    # the point is in the AI regime on both sides (the reason for the gate)
    assert by['cv_isi_E'].ref_mean >= 0.5 and by['cv_isi_E'].ours_mean >= 0.5
    failed = [r for r in results if not r.passed]
    assert not failed, format_g2_table(failed)


if __name__ == '__main__':  # pragma: no cover
    res, ours, ref = g5_results()
    print(format_g2_table(res))
    for k in STATS:
        print(k, 'kernel', np.round(ours[k], 4).tolist(), 'NEST', np.round(ref[k], 4).tolist())
