# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate G2: Brunel ``iaf_psc_alpha`` network statistics vs live NEST (``requires_nest``).

Order 200 (800 E + 200 I), the parameters of NEST's ``brunel_alpha_nest.py``
(:func:`brainpy_state._kernel.validation.brunel.brunel_params`), spikes of every
neuron recorded. ``T_WARM = 100 ms`` of initial transient (every neuron starts at
``V_m = 0``, so the onset is synchronous) is simulated and discarded; the statistics
of :func:`~brainpy_state._kernel.validation.stats.network_stats` use the following
``T_STAT = 1000 ms``. 10 seeds each side (NEST ``rng_seed``, kernel ``Network(seed)``;
connectivity and noise both change with the seed). Each statistic passes if
``|Δmean| <= max(5 % of the NEST mean, 2 x combined SEM)``.

Run as a script to print the table::

    python -m brainpy_state._kernel.validation.g2_brunel_nest_test
"""
from __future__ import annotations

import functools

import brainunit as u
import numpy as np
import pytest

from brainpy_state._kernel.validation.brunel import brunel_params, build_brunel
from brainpy_state._kernel.validation.stats import format_g2_table, g2_rule, network_stats
from brainpy_state._nest_validation.nest_compare import requires_nest

ORDER = 200
SEEDS = tuple(range(1, 11))
T_WARM = 100.0
T_STAT = 1000.0
DT = 0.1
STATS = ('rate_E', 'rate_I', 'cv_isi_E', 'fano_E', 'corr_E')


def nest_brunel(order: int, seed: int, t_total: float, g: float = 5.0, eta: float = 2.0,
                threads: int = 1):
    """One NEST run; returns ``(times ms, senders)`` with senders ``0..N-1`` (E first).

    ``g`` (relative inhibition) and ``eta`` (external drive / threshold rate) default to
    the NEST example values used by G2; K1a-G5 passes its asynchronous-irregular point.
    """
    import nest
    p = brunel_params(order, g=g, eta=eta)
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = DT
    nest.rng_seed = seed
    nest.local_num_threads = threads
    npar = {'C_m': p['c_mem'], 'tau_m': p['tau_mem'], 't_ref': p['t_ref'], 'E_L': 0.0,
            'V_reset': 0.0, 'V_m': 0.0, 'V_th': p['theta'], 'tau_syn_ex': p['tau_syn'],
            'tau_syn_in': p['tau_syn']}
    E = nest.Create('iaf_psc_alpha', p['N_E'], params=npar)
    I = nest.Create('iaf_psc_alpha', p['N_I'], params=npar)
    noise = nest.Create('poisson_generator', params={'rate': p['rate']})
    sr = nest.Create('spike_recorder')
    ex = {'weight': p['J_ex'], 'delay': p['delay']}
    inh = {'weight': p['J_in'], 'delay': p['delay']}
    nest.Connect(noise, E + I, syn_spec=ex)
    nest.Connect(E, E + I, {'rule': 'fixed_indegree', 'indegree': p['C_E']}, ex)
    nest.Connect(I, E + I, {'rule': 'fixed_indegree', 'indegree': p['C_I']}, inh)
    nest.Connect(E + I, sr)
    nest.Simulate(t_total)
    ev = sr.events
    return np.asarray(ev['times'], float), np.asarray(ev['senders']) - E[0].global_id


def kernel_brunel(order: int, seed: int, t_total: float, precision: str = 'float32',
                  g: float = 5.0, eta: float = 2.0):
    """One kernel run (float32 by default); returns ``(times ms, senders)`` like :func:`nest_brunel`."""
    p = brunel_params(order, g=g, eta=eta)
    net, info = build_brunel(order, seed=seed, n_rec=p['N_E'] + p['N_I'], g=g, eta=eta)
    res = net.compile(precision).runner().simulate(t_total * u.ms)
    ts, ss = [], []
    for key in ('rec_E', 'rec_I'):
        t, gids, counts = res[info[key]].spike_events()
        ts.append(np.repeat(np.asarray(t.to_decimal(u.ms)), np.asarray(counts, int)))
        ss.append(np.repeat(np.asarray(gids), np.asarray(counts, int)))
    return np.concatenate(ts), np.concatenate(ss)


def run_stats(sim, order: int = ORDER, seeds=SEEDS) -> dict:
    """``{statistic: [value per seed]}`` for ``sim`` in {'nest', 'kernel'}."""
    p = brunel_params(order)
    fn = nest_brunel if sim == 'nest' else kernel_brunel
    out = {k: [] for k in STATS}
    for s in seeds:
        t, snd = fn(order, s, T_WARM + T_STAT)
        st = network_stats(t, snd, p['N_E'], p['N_I'], T_WARM, T_WARM + T_STAT)
        for k in STATS:
            out[k].append(st[k])
    return out


@functools.lru_cache(maxsize=None)
def g2_results() -> tuple:
    """G2 rule per statistic (computed once per process)."""
    ref = run_stats('nest')
    ours = run_stats('kernel')
    return tuple(g2_rule(ours[k], ref[k], k) for k in STATS), ours, ref


@requires_nest
@pytest.mark.parametrize('stat', STATS)
def test_g2_brunel_statistic_matches_nest(stat):
    results, ours, ref = g2_results()
    r = results[STATS.index(stat)]
    print('\n' + format_g2_table(results))
    assert len(ours[stat]) == len(ref[stat]) == len(SEEDS)
    assert r.passed, format_g2_table([r])


@requires_nest
def test_g2_runs_are_in_the_expected_firing_regime():
    results, _, _ = g2_results()
    by = {r.metric: r for r in results}
    # NEST 3.9 / 3.10, order 200 (eta = 2, g = 5): ~57 Hz, regular oscillatory firing
    # (CV-ISI ~0.08, Fano ~0.43, positive correlations). A silent, saturated or broken
    # network would fail these loudly, on either side.
    for r in (by['rate_E'], by['rate_I']):
        assert 40.0 < r.ref_mean < 80.0 and 40.0 < r.ours_mean < 80.0
    assert 0.02 < by['cv_isi_E'].ref_mean < 0.5 and 0.02 < by['cv_isi_E'].ours_mean < 0.5
    assert 0.0 < by['fano_E'].ref_mean and 0.0 < by['fano_E'].ours_mean
    assert 0.0 < by['corr_E'].ref_mean and 0.0 < by['corr_E'].ours_mean

if __name__ == '__main__':  # pragma: no cover
    res, ours, ref = g2_results()
    print(format_g2_table(res))
    for k in STATS:
        print(k, 'kernel', np.round(ours[k], 4).tolist(), 'NEST', np.round(ref[k], 4).tolist())
