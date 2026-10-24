# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate G1, precise part: ``timed_event`` delivery and ``iaf_psc_exp_ps`` vs live NEST.

Both sides are networks with the same delays, recorded every step with no alignment
shift (our step ``n`` ends at ``(n + 1) dt`` = the NEST multimeter sample). Pass:
``V_m`` within ``CAT_B`` (1e-6 mV) and output spike times within 1e-6 ms (spec §6 G1).
Our ``poisson_generator_ps`` cannot reproduce NEST's random stream, so its spikes are
replayed in NEST through a precise ``spike_generator``; its statistics are compared
with NEST's ``poisson_generator_ps`` separately.
"""
from __future__ import annotations

import brainunit as u
import jax
import numpy as np
import pytest

from brainspike._kernel.connectivity import one_to_one
from brainspike._kernel.models import (
    iaf_psc_alpha,
    iaf_psc_exp_ps,
    poisson_generator_ps,
    spike_generator,
    spike_generator_ps,
)
from brainspike._kernel.network import Network
from brainspike._nest_validation import tolerance_conventions as tc
from brainspike._nest_validation.nest_compare import compare_trace, requires_nest

DT = 0.1
T_SIM = 80.0
N_STEPS = int(round(T_SIM / DT))
SPIKE_ATOL = 1e-6  # ms

#: three heterogeneous neurons (NEST parameter names)
NEURONS = [
    dict(I_e=250.0),
    dict(I_e=320.0, t_ref=0.25, tau_syn_in=5.0, V_reset=-68.0, V_min=-69.0),
    dict(I_e=150.0, tau_m=15.0, tau_syn_ex=0.5, C_m=200.0, V_reset=-65.0, V_th=-56.0),
]
_UNITS = dict(I_e=u.pA, t_ref=u.ms, tau_syn_in=u.ms, tau_syn_ex=u.ms, tau_m=u.ms, V_min=u.mV,
              C_m=u.pF, V_reset=u.mV, V_th=u.mV)


def _ours(params_list):
    keys = sorted({k for p in params_list for k in p})
    defaults = iaf_psc_exp_ps.params
    out = {}
    for k in keys:
        vals = [p.get(k, float(u.get_mantissa(defaults[k].default))) for p in params_list]
        out[k] = np.asarray(vals) * _UNITS[k]
    return out


def _trains(seed):
    rng = np.random.default_rng(seed)
    ex, inh = [], []
    for i in range(len(NEURONS)):
        t = np.sort(rng.uniform(0.01, T_SIM - 5, 300))
        t = np.concatenate([t, 20.0 + i + np.array([0.011, 0.012, 0.0123, 0.05, 0.0999])])  # one step
        t = np.unique(np.round(t, 7))
        is_ex = rng.random(t.size) < 0.75
        ex.append(t[is_ex])
        inh.append(t[~is_ex])
    return ex, inh


def _nest_record(nrn, n):
    import nest
    mm = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': DT})
    sr = nest.Create('spike_recorder')
    nest.Connect(mm, nrn)
    nest.Connect(nrn, sr)
    return mm, sr


def _nest_read(mm, sr, nrn, n):
    ev = mm.events
    steps = np.rint(np.asarray(ev['times']) / DT).astype(int) - 1
    senders = np.asarray(ev['senders']) - nrn[0].global_id
    V = np.full((N_STEPS, n), np.nan)
    keep = steps < N_STEPS
    V[steps[keep], senders[keep]] = np.asarray(ev['V_m'])[keep]
    spk = sr.events
    return V, np.asarray(spk['times']), np.asarray(spk['senders']) - nrn[0].global_id


def _nest_reset():
    import nest
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = DT
    return nest


def nest_precise_input(ex, inh, w_ex, w_in, delay, model='iaf_psc_exp_ps', params=NEURONS):
    nest = _nest_reset()
    n = len(ex)
    nrn = nest.Create(model, n, params=list(params) if params is not None else None)
    for i in range(n):
        for times, w in ((ex[i], w_ex), (inh[i], w_in)):
            sg = nest.Create('spike_generator', params={'spike_times': times, 'precise_times': True})
            nest.Connect(sg, nrn[i], syn_spec={'weight': w, 'delay': delay})
    mm, sr = _nest_record(nrn, n)
    nest.Simulate(T_SIM + DT)
    return _nest_read(mm, sr, nrn, n)


def our_precise_input(ex, inh, w_ex, w_in, delay, model=iaf_psc_exp_ps, params=None):
    net = Network(dt=DT * u.ms)
    n = len(ex)
    nrn = net.add(model, n, params=params, name='n')
    gE = net.add(spike_generator_ps, n, params=dict(spike_times=[t * u.ms for t in ex]))
    gI = net.add(spike_generator_ps, n, params=dict(spike_times=[t * u.ms for t in inh]))
    port = 'syn' if model is iaf_psc_exp_ps else 'ex'
    net.connect(gE, nrn, one_to_one(), port=port, weight=w_ex * u.pA, delay=delay * u.ms)
    net.connect(gI, nrn, one_to_one(), port=port if model is iaf_psc_exp_ps else 'in_',
                weight=w_in * u.pA, delay=delay * u.ms)
    return _our_read(net, nrn, precise=model is iaf_psc_exp_ps)


def _our_read(net, nrn, precise=True):
    names = ('V_m', 'spike', 'spike_offset') if precise else ('V_m', 'spike')
    rec = net.observe(nrn, *names)
    res = net.compile('float64').runner().simulate(T_SIM * u.ms)
    r = res[rec]
    V = np.asarray(r['V_m'].to_decimal(u.mV))
    sp = np.asarray(r['spike'])
    rows, cols = np.nonzero(sp)
    t = rows * DT + (np.asarray(r['spike_offset'].to_decimal(u.ms))[rows, cols] if precise else DT)
    order = np.lexsort((cols, t))
    return V, t[order], cols[order]


def _check(ref, ours, what):
    V_ref, t_ref, s_ref = ref
    V, t, s = ours
    assert not np.isnan(V_ref).any()
    for col in range(V.shape[1]):
        r = compare_trace(V_ref[:, col] * u.mV, V[:, col] * u.mV, tol=tc.CAT_B)
        assert r.passed, f'{what} neuron {col}: {r}'
    order = np.lexsort((s_ref, t_ref))
    t_ref, s_ref = t_ref[order], s_ref[order]
    assert len(t) == len(t_ref), f'{what}: {len(t)} vs NEST {len(t_ref)} spikes'
    np.testing.assert_array_equal(s, s_ref)
    np.testing.assert_allclose(t, t_ref, atol=SPIKE_ATOL, rtol=0)
    return len(t)


@requires_nest
class TestPreciseVsNest:
    @pytest.mark.parametrize('delay', [0.1, 1.0, 2.35])
    def test_spike_generator_ps_to_iaf_psc_exp_ps(self, delay):
        ex, inh = _trains(1)
        ref = nest_precise_input(ex, inh, 160.0, -120.0, delay)
        with jax.enable_x64(True):
            ours = our_precise_input(ex, inh, 160.0, -120.0, delay, params=_ours(NEURONS))
        n_spk = _check(ref, ours, f'delay {delay}')
        assert n_spk >= 10
        # refractory neuron 1 (t_ref 0.25 ms = 3 steps) spikes and its V_min clamp is hit
        assert np.any(ours[2] == 1) and abs(np.min(ours[0][:, 1]) + 69.0) < 1e-9

    def test_poisson_generator_ps_to_iaf_psc_exp_ps(self):
        n = 4
        with jax.enable_x64(True):
            net = Network(dt=DT * u.ms, seed=11)
            nrn = net.add(iaf_psc_exp_ps, n, params=dict(I_e=200.0 * u.pA), name='n')
            gE = net.add(poisson_generator_ps, n, params=dict(rate=6000.0 * u.Hz), name='gE')
            gI = net.add(poisson_generator_ps, n, params=dict(rate=1500.0 * u.Hz,
                                                               dead_time=0.2 * u.ms), name='gI')
            net.connect(gE, nrn, one_to_one(), port='syn', weight=45.0 * u.pA, delay=1.0 * u.ms)
            net.connect(gI, nrn, one_to_one(), port='syn', weight=-90.0 * u.pA, delay=1.0 * u.ms)
            rg = net.observe(gE + gI, 'spike_offsets', 'overflow')
            ours = _our_read(net, nrn)
            # replay the generated trains in NEST (the observer ran in the same simulation)
            res = net.compile('float64').runner().simulate(T_SIM * u.ms)
            off = np.asarray(res[rg]['spike_offsets'].to_decimal(u.ms))
            assert int(np.asarray(res[rg]['overflow']).max()) == 0
        t = np.arange(N_STEPS)[:, None, None] * DT + off
        trains = [np.sort(t[:, i][np.isfinite(t[:, i])]) for i in range(2 * n)]
        assert sum(len(x) for x in trains) > 1000
        ref = nest_precise_input(trains[:n], trains[n:], 45.0, -90.0, 1.0,
                                 params=[dict(I_e=200.0)] * n)
        assert _check(ref, ours, 'poisson') >= 4

    def test_iaf_psc_exp_ps_chain(self):
        ex, inh = _trains(2)
        nest = _nest_reset()
        a = nest.Create('iaf_psc_exp_ps', params={'I_e': 300.0})
        b = nest.Create('iaf_psc_exp_ps', params={'I_e': 100.0})
        sg = nest.Create('spike_generator', params={'spike_times': ex[0], 'precise_times': True})
        nest.Connect(sg, a, syn_spec={'weight': 150.0, 'delay': 0.5})
        nest.Connect(a, b, syn_spec={'weight': 900.0, 'delay': 1.3})
        mm, sr = _nest_record(a + b, 2)
        nest.Simulate(T_SIM + DT)
        ref = _nest_read(mm, sr, a + b, 2)
        with jax.enable_x64(True):
            net = Network(dt=DT * u.ms)
            nrn = net.add(iaf_psc_exp_ps, 2, params=dict(I_e=[300.0, 100.0] * u.pA))
            g = net.add(spike_generator_ps, 1, params=dict(spike_times=ex[0] * u.ms))
            net.connect(g, nrn[0], port='syn', weight=150.0 * u.pA, delay=0.5 * u.ms)
            net.connect(nrn[0], nrn[1], port='syn', weight=900.0 * u.pA, delay=1.3 * u.ms)
            ours = _our_read(net, nrn)
        _check(ref, ours, 'chain')
        assert np.sum(ours[2] == 1) >= 2  # b fires from a's precisely timed spikes

    def test_grid_spike_generator_into_iaf_psc_exp_ps(self):
        times = [1.0, 1.0, 5.3, 12.0, 12.1, 30.0]
        nest = _nest_reset()
        nrn = nest.Create('iaf_psc_exp_ps', params={'I_e': 330.0})
        sg = nest.Create('spike_generator', params={'spike_times': times})
        nest.Connect(sg, nrn, syn_spec={'weight': 250.0, 'delay': 0.4})
        mm, sr = _nest_record(nrn, 1)
        nest.Simulate(T_SIM + DT)
        ref = _nest_read(mm, sr, nrn, 1)
        with jax.enable_x64(True):
            net = Network(dt=DT * u.ms)
            n = net.add(iaf_psc_exp_ps, 1, params=dict(I_e=330.0 * u.pA))
            g = net.add(spike_generator, 1, params=dict(spike_times=times * u.ms))
            net.connect(g, n, port='syn', weight=250.0 * u.pA, delay=0.4 * u.ms)
            ours = _our_read(net, n)
        _check(ref, ours, 'grid -> ps')

    def test_precise_source_into_grid_neuron(self):
        ex, inh = _trains(3)
        params = [dict(I_e=200.0)] * len(ex)
        ref = nest_precise_input(ex, inh, 300.0, -200.0, 1.0, model='iaf_psc_alpha', params=params)
        with jax.enable_x64(True):
            ours = our_precise_input(ex, inh, 300.0, -200.0, 1.0, model=iaf_psc_alpha,
                                     params=dict(I_e=200.0 * u.pA))
        # grid neuron: spike times on the grid (no offset)
        V_ref, t_ref, s_ref = ref
        V, t, s = ours
        for col in range(V.shape[1]):
            r = compare_trace(V_ref[:, col] * u.mV, V[:, col] * u.mV, tol=tc.CAT_B)
            assert r.passed, f'neuron {col}: {r}'


@requires_nest
class TestPoissonStatisticsVsNest:
    def test_isi_distribution_matches_nest(self):
        stats = pytest.importorskip('scipy.stats')
        rate, dead, n, T = 250.0, 1.5, 200, 1000.0
        nest = _nest_reset()
        pg = nest.Create('poisson_generator_ps', params={'rate': rate, 'dead_time': dead})
        par = nest.Create('parrot_neuron_ps', n)
        sr = nest.Create('spike_recorder')
        nest.Connect(pg, par)
        nest.Connect(par, sr)
        nest.Simulate(T)
        ev = sr.events
        nest_isi = np.concatenate([np.diff(np.sort(ev['times'][ev['senders'] == g]))
                                   for g in par.global_id])
        with jax.enable_x64(True):
            net = Network(dt=DT * u.ms, seed=5)
            g = net.add(poisson_generator_ps, n, params=dict(rate=rate * u.Hz, dead_time=dead * u.ms))
            rec = net.observe(g, 'spike_offsets')
            res = net.compile('float64').runner().simulate(T * u.ms)
            off = np.asarray(res[rec]['spike_offsets'].to_decimal(u.ms))
        t = np.arange(off.shape[0])[:, None, None] * DT + off
        our_isi = np.concatenate([np.diff(np.sort(t[:, i][np.isfinite(t[:, i])])) for i in range(n)])
        assert our_isi.min() >= dead - 1e-9 and nest_isi.min() >= dead - 1e-9
        np.testing.assert_allclose(len(our_isi), len(nest_isi), rtol=0.05)
        assert stats.ks_2samp(our_isi, nest_isi).pvalue > 1e-3
