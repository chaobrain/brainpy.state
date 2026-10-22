# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Checks for ``benchmarks/g9_interleaved.py``; run explicitly::

    python -m pytest benchmarks/check_g9_interleaved.py
"""
import json
import types

import pytest

import _bench_utils as bu
import g9_interleaved as g9

STAT0 = """cpu  100 0 100 800 0 0 0 0 0 0
cpu0 10 0 10 80 0 0 0 0 0 0
cpu1 10 0 10 80 0 0 0 0 0 0
cpu2 10 0 10 80 0 0 0 0 0 0
cpu3 10 0 10 80 0 0 0 0 0 0
intr 1 2 3
"""
# cpu0 fully busy, cpu1 3 % busy (iowait counts as idle), cpu2 idle, cpu3 50 %
STAT1 = """cpu  0 0 0 0 0 0 0 0 0 0
cpu0 110 0 10 80 0 0 0 0 0 0
cpu1 12 0 11 177 20 0 0 0 0 0
cpu2 10 0 10 180 0 0 0 0 0 0
cpu3 35 0 35 130 0 0 0 0 0 0
"""


def test_proc_stat_and_utilisation():
    a, b = g9.read_proc_stat(STAT0), g9.read_proc_stat(STAT1)
    assert sorted(a) == [0, 1, 2, 3] and a[0] == (20, 100)
    u = g9.utilisation(a, b)
    assert u[0] == 100.0 and u[1] == pytest.approx(2.5) and u[2] == 0.0 and u[3] == 50.0
    assert g9.utilisation(b, b)[0] == 0.0  # no elapsed jiffies
    snaps = iter([a, b])
    assert g9.sample(0.0, reader=lambda: next(snaps), sleep=lambda s: None)[0] == 100.0


def test_parse_cpu_list_and_siblings(tmp_path):
    assert g9.parse_cpu_list('4-5\n') == [4, 5]
    assert g9.parse_cpu_list('0-1,8,') == [0, 1, 8]
    topo = tmp_path / 'cpu4' / 'topology'
    topo.mkdir(parents=True)
    (topo / 'thread_siblings_list').write_text('4-5\n')
    assert g9.siblings(4, root=str(tmp_path)) == [5]
    assert g9.siblings(7, root=str(tmp_path)) == []


def test_pick_core():
    sib = {0: [1], 1: [0], 2: [3], 3: [2], 4: [], 5: []}.get
    util = {0: 1.0, 1: 0.5, 2: 0.2, 3: 9.0, 4: 0.0, 5: 3.0}
    # preferred core idle with an idle sibling
    assert g9.pick_core(util, sib, prefer=0) == (0, [1])
    # preferred core's sibling busy: least-used hyperthreaded idle pair, then any
    assert g9.pick_core(util, sib, prefer=2) == (0, [1])
    assert g9.pick_core({**util, 0: 7.0}, sib, prefer=None) == (4, [])
    assert g9.pick_core({0: 50.0, 1: 50.0}, sib) is None


def test_check_run():
    assert g9.check_run({5: 1.0}, 0.99, 5.0, 0.85) == []
    r = g9.check_run({5: 7.0}, 0.5, 5.0, 0.85)
    assert len(r) == 2 and 'sibling' in r[0] and 'share' in r[1]


def _run(impl, order, prec, ms, peak, rnd=0, valid=True, status='ok'):
    r = {'status': status, 'config': {'device': 'cpu', 'order': order, 'precision': prec,
                                      'brainevent_backend': None, 'cpu_affinity': '4'},
         'g9': {'impl': impl, 'round': rnd, 'valid': valid}}
    if status == 'ok':
        r.update(steady={'ms_per_step': bu.summarize([ms])}, memory={'peak_bytes': peak},
                 compile={'compile_s': 1.0})
    return r


def _t0(order, prec, ms, peak, aff='4'):
    return {'status': 'ok', 'config': {'device': 'cpu', 'order': order, 'precision': prec,
                                       'brainevent_backend': None, 'cpu_affinity': aff},
            'steady': {'ms_per_step': bu.summarize([ms])}, 'memory': {'peak_bytes': peak},
            'compile': {'compile_s': 0.9}}


def test_compare_medians_ratios_and_status():
    runs = []
    for rnd, (o, n) in enumerate([(1.0, 1.05), (1.2, 1.0), (0.9, 5.0)]):
        runs += [_run('old', 2500, 32, o, 100, rnd), _run('new', 2500, 32, n, 105, rnd)]
    runs += [_run('old', 2500, 64, 1.0, 100), _run('new', 2500, 64, 1.0, 120),
             _run('new', 2500, 64, 0.1, 50, valid=False),        # discarded: ignored
             _run('old', 200, 32, 1.0, 100), _run('new', 200, 32, 3.0, 100),
             _run('old', 200, 64, 1.0, 100, status='failed')]
    rows = g9.compare(runs, [_t0(2500, 32, 0.5, 200), _t0(2500, 32, 9.0, 1, aff='all')])
    by = {(r['order'], r['precision']): r for r in rows}
    assert [(r['order'], r['precision']) for r in rows] == [(2500, 32), (2500, 64), (200, 32),
                                                             (200, 64)]
    r = by[2500, 32]
    assert r['old']['ms_per_step_median'] == 1.0 and r['new']['ms_per_step_median'] == 1.05
    assert r['step_ratio'] == pytest.approx(1.05) and r['mem_ratio'] == pytest.approx(1.05)
    assert r['status'] == 'pass' and r['t0']['ms_per_step'] == 0.5  # pinned T0 row only
    assert by[2500, 64]['status'] == 'fail' and by[2500, 64]['new']['n'] == 1  # memory 1.2x
    assert by[200, 32]['status'] == 'reported' and not by[200, 32]['gated']
    assert by[200, 64]['status'] == 'not measured' and by[200, 64]['t0'] is None
    assert not g9.gate_passed(rows)
    assert g9.gate_passed([by[2500, 32]]) and not g9.gate_passed([by[200, 32]])
    md = g9.markdown(rows)
    assert '| 2500 | f32 | yes | 3/3 | 1.0000 | 1.0500 | 1.050 |' in md
    assert '| 2500 | f32 | 0.5000 | 1.0500 | 2.100 |' in md and '| fail |' in md


def test_compare_main_and_parse(tmp_path, capsys):
    runs = [_run('old', 2500, 32, 1.0, 100), _run('new', 2500, 32, 1.0, 100)]
    log = tmp_path / 'g9.log'
    log.write_text('# core 4\n' + ''.join(g9.RUN_PREFIX + json.dumps(r) + '\n' for r in runs))
    assert len(g9.parse_runs(log.read_text())) == 2
    t0 = tmp_path / 't0.json'
    t0.write_text(json.dumps({'records': [_t0(2500, 32, 1.0, 100)]}))
    out = tmp_path / 'out.json'
    assert g9.main(['compare', str(log), '--t0', str(t0), '--out', str(out)]) == 0
    assert 'PASS' in capsys.readouterr().out
    saved = json.loads(out.read_text())
    assert saved['rows'][0]['status'] == 'pass' and len(saved['runs']) == 2
    assert g9.main(['compare', str(log), '--t0', str(tmp_path / 'missing.json'),
                    '--tol', '0.5']) == 1


def test_run_command_and_measure(monkeypatch):
    args = types.SimpleNamespace(commit='abc', repeats=3, window_ms=50.0, max_attempts=2,
                                 idle_pct=5.0, min_share=0.85, pre_sample_s=0.0, wait_s=0.0,
                                 max_waits=2, timeout=10.0)
    cmd = g9.run_command('new', 200, 64, 4, args)
    assert cmd[:3] == ['taskset', '-c', '4'] and cmd[4].endswith('brunel_kernel.py')
    assert cmd[-2:] == ['--window-ms', '50.0'] and '--commit' in cmd
    rec = _run('new', 200, 64, 1.0, 100)
    del rec['g9']
    outputs = iter([
        types.SimpleNamespace(returncode=1, stdout='', stderr='boom'),
        types.SimpleNamespace(returncode=0, stdout=bu.RESULT_PREFIX + json.dumps(rec) + '\n',
                              stderr=''),
    ])
    monkeypatch.setattr(g9.subprocess, 'run', lambda *a, **k: next(outputs))
    monkeypatch.setattr(g9, 'sample', lambda s: {4: 0.0, 5: 0.0})
    monkeypatch.setattr(g9, 'read_proc_stat', lambda: {4: (0, 0), 5: (0, 0)})
    monkeypatch.setattr(g9, 'check_run', lambda *a: [])
    logs = []
    got = g9.measure('new', 200, 64, 4, [5], args, 0, log=logs.append)
    assert got['status'] == 'ok' and got['g9']['valid'] and got['g9']['attempt'] == 2
    assert len(got['g9']['discarded']) == 1 and 'run failed' in got['g9']['discarded'][0][
        'reasons'][0]
    assert any('discard' in m for m in logs)
    # always failing -> failure record carrying the last attempt
    monkeypatch.setattr(g9.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout='', stderr='x'))
    bad = g9.measure('old', 200, 64, 4, [5], args, 1, log=logs.append)
    assert bad['status'] == 'failed' and len(bad['g9']['discarded']) == 1
    # a core that stays busy aborts
    monkeypatch.setattr(g9, 'sample', lambda s: {4: 90.0, 5: 0.0})
    with pytest.raises(RuntimeError):
        g9.wait_idle(4, [5], args, log=logs.append)
