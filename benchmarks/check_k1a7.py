# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Checks for the K1a-7 benchmark drivers (no GPU, no NEST); run explicitly::

    python -m pytest benchmarks/check_k1a7.py
"""
import json
import os

import pytest

import k1a_gpu_g9 as g3
import k1a_microcircuit as mc

GIB = 2 ** 30


def _rec(impl, order, prec, ms, peak, compile_s=1.0, status='ok', backend='default'):
    return {'status': status, 'config': {'order': order, 'precision': prec},
            'steady': {'ms_per_step': {'median': ms}}, 'memory': {'peak_bytes': peak},
            'compile': {'compile_s': compile_s}, 'sanity': {'e_rate_hz': {'median': 28.8}},
            'g3': {'impl': impl, 'round': 0, 'backend': backend, 'load_start': 0.1}}


def _t0(order, prec, ms, peak, backend=None, device='gpu'):
    return {'status': 'ok', 'config': {'device': device, 'order': order, 'precision': prec,
                                       'brainevent_backend': backend},
            'steady': {'ms_per_step': {'median': ms}}, 'memory': {'peak_bytes': peak},
            'compile': {'compile_s': 0.8}}


def test_g3_compare_against_t0_and_old():
    runs = [_rec('old', 2500, 32, 0.50, 0.25 * GIB), _rec('new', 2500, 32, 0.10, 0.26 * GIB),
            _rec('new', 2500, 32, 0.12, 0.26 * GIB),
            _rec('new', 2500, 64, 1.20, 0.30 * GIB),
            _rec('new', 200, 32, 0.05, 1.0 * GIB, status='failed')]
    t0 = [_t0(2500, 32, 0.49, 0.232 * GIB), _t0(2500, 64, 0.88, 0.426 * GIB, 'cusparse'),
          _t0(2500, 32, 9.9, 9 * GIB, device='cpu')]
    rows = {(r['order'], r['precision']): r for r in g3.compare_runs(runs, t0)}
    r = rows[(2500, 32)]
    assert r['new']['n'] == 2 and r['new']['ms_per_step'] == pytest.approx(0.11)
    assert r['step_ratio_t0'] == pytest.approx(0.11 / 0.49)
    assert r['mem_ratio_t0'] == pytest.approx(0.26 / 0.232) and r['status'] == 'FAIL'  # > 1.1
    assert r['step_ratio_old'] == pytest.approx(0.11 / 0.50)
    r64 = rows[(2500, 64)]
    assert r64['backend'] == 'cusparse' and r64['status'] == 'FAIL'  # 1.2 / 0.88 > 1.1
    assert 'step_ratio_old' not in r64  # old not re-measured
    assert rows[(200, 32)]['status'] == 'not measured'  # failed runs are ignored
    table = g3.markdown(g3.compare_runs(runs, t0))
    assert table.count('\n') == 2 + 3 and '| 2500 | f32 | default |' in table


def test_g3_compare_cli(tmp_path, capsys):
    log = tmp_path / 'g3.log'
    log.write_text('noise\n' + g3.RUN_PREFIX + json.dumps(_rec('new', 200, 32, 0.03, 0.04 * GIB)) + '\n')
    t0 = tmp_path / 't0.json'
    t0.write_text(json.dumps({'records': [_t0(200, 32, 0.398, 0.048 * GIB)]}))
    out = tmp_path / 'o.json'
    g3.main(['compare', str(log), '--t0', str(t0), '--out', str(out)])
    rows = json.loads(out.read_text())['rows']
    assert [r['status'] for r in rows if r['order'] == 200 and r['precision'] == 32] == ['pass']
    assert 'pass' in capsys.readouterr().out


def _mc(sim, seed, scale, device='cpu', **kw):
    rates = {k: 2.0 + 0.01 * seed + (0.5 if sim == 'kernel' and scale else 0.0) for k in
             ('L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I')}
    rec = dict(sim=sim, seed=seed, rates=rates, build_s=1.0, peak_rss_gib=1.0, **kw)
    if sim == 'kernel':
        rec.update(device=device, compile_s=2.0, ms_per_step=3.0, n_edges=10)
        if device == 'gpu':
            rec['peak_device_gib'] = 0.5
    else:
        rec.update(sim_s=5.0, threads=16)
    return rec


def test_microcircuit_compare(tmp_path, capsys):
    os.environ.setdefault('BRAINPY_STATE_ALLOW_DIRECT_IMPORT', '1')
    lines = [_mc('nest', s, 0) for s in range(1, 4)]
    lines += [_mc('kernel', s, 0, 'cpu') for s in range(1, 4)]       # equal -> pass
    lines += [_mc('kernel', s, 1, 'gpu') for s in range(1, 4)]       # +0.5 Hz -> fail
    log = tmp_path / 'mc.log'
    log.write_text(''.join('RESULT_JSON ' + json.dumps(r) + '\n' for r in lines))
    out = tmp_path / 'mc.json'
    mc.main(['compare', str(log), '--out', str(out)])
    res = json.loads(out.read_text())
    assert res['gate']['cpu']['passed'] and not res['gate']['gpu']['passed']
    assert res['gate']['gpu']['perf']['peak_device_gib'] == 0.5
    text = capsys.readouterr().out
    assert 'NEST: 3 seeds' in text and 'FAIL' in text
