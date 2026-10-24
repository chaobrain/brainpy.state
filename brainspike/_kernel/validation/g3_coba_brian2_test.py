# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate G3: COBA (Vogels--Abbott) network and single neurons vs Brian2 (``requires_brian2``).

Model and simulator correspondence: :mod:`brainspike._kernel.validation.coba`.

* Network: 4000 neurons, 1000 ms, 10 seeds each side (kernel float32, the default
  tier); E and I rate, CV-ISI, Fano factor (10 ms bins) and mean pairwise spike-count
  correlation (50 E neurons) by the G2 rule. The initial voltages are random, so no
  transient is discarded.
* Single neuron: identical excitatory and inhibitory input spike trains, the same
  ``iaf_cond_exp`` neuron below and above rheobase; ``V_m`` traces ``CAT_A`` (and the
  conductances within ``1e-3 nS``), spike times equal.

Run as a script to print the network table::

    python -m brainspike._kernel.validation.g3_coba_brian2_test
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

from brainspike._kernel.validation import coba as C
from brainspike._kernel.validation.markers import requires_brian2
from brainspike._kernel.validation.stats import format_g2_table, g2_rule, network_stats
from brainspike._nest_validation import tolerance_conventions as tc
from brainspike._nest_validation.nest_compare import compare_trace

SEEDS = tuple(range(1, 11))
T_MS = 1000.0
STATS = ('rate_E', 'rate_I', 'cv_isi_E', 'fano_E', 'corr_E')


def run_stats(sim: str, seeds=SEEDS, precision: str = 'float32') -> dict:
    """``{statistic: [value per seed]}`` for ``sim`` in {'kernel', 'brian2'}."""
    out = {k: [] for k in STATS}
    for s in seeds:
        if sim == 'kernel':
            t, snd = C.kernel_coba(s, T_MS, precision)
        else:
            t, snd = C.brian2_coba(s, T_MS)
        st = network_stats(t, snd, C.COBA['N_E'], C.COBA['N_I'], 0.0, T_MS)
        for k in STATS:
            out[k].append(st[k])
    return out


@functools.lru_cache(maxsize=None)
def g3_results() -> tuple:
    """G2 rule per statistic, kernel vs Brian2 (computed once per process)."""
    ref = run_stats('brian2')
    ours = run_stats('kernel')
    return tuple(g2_rule(ours[k], ref[k], k) for k in STATS), ours, ref


def _inputs():
    rng = np.random.default_rng(5)
    ex = np.unique(np.round(rng.uniform(2.0, 295.0, 150), 1))
    inh = np.unique(np.round(rng.uniform(2.0, 295.0, 30), 1))
    return ex, inh


@requires_brian2
class TestSingleNeuronVsBrian2:
    @pytest.mark.parametrize('I_e', [0.0, 150.0])
    def test_traces_and_spikes(self, I_e):
        ex, inh = _inputs()
        ours = C.kernel_single(ex, inh, 300.0, I_e)
        ref = C.brian2_single(ex, inh, 300.0, I_e)
        r = compare_trace(ref['V_m'], ours['V_m'], tol=tc.CAT_A)
        assert r.passed, r
        assert np.abs(ours['V_m'] - ref['V_m']).max() <= 1e-3
        for g in ('g_ex', 'g_in'):
            assert np.abs(ours[g] - ref[g]).max() <= 1e-3
        np.testing.assert_allclose(ours['spikes'], ref['spikes'], atol=1e-9)
        assert ours['g_in'].max() > C.COBA['w_in'] * 0.5
        if I_e > 0:
            assert len(ref['spikes']) >= 5  # refractory clamps are part of the comparison


@requires_brian2
@pytest.mark.parametrize('stat', STATS)
def test_g3_coba_statistic_matches_brian2(stat):
    results, ours, ref = g3_results()
    print('\n' + format_g2_table(results, ref='Brian2'))
    assert len(ours[stat]) == len(ref[stat]) == len(SEEDS)
    r = results[STATS.index(stat)]
    assert r.passed, format_g2_table([r], ref='Brian2')


def test_small_coba_runs_and_has_no_autapses():
    net, info = C.build_coba(seed=3, scale=0.05)
    plan = net.compile()
    for name in ('E_P', 'I_P'):
        es = plan.edges(name)
        assert es.n_edges > 0 and not np.any(np.asarray(es.pre) == np.asarray(es.post))
    t, s = C.kernel_coba(3, 100.0, scale=0.05)
    assert t.size > 0 and s.max() < 200 and t.min() > 0.0


if __name__ == '__main__':  # pragma: no cover
    res, ours, ref = g3_results()
    print(format_g2_table(res, ref='Brian2'))
    for k in STATS:
        print(k, 'kernel', np.round(ours[k], 4).tolist(), 'Brian2', np.round(ref[k], 4).tolist())
