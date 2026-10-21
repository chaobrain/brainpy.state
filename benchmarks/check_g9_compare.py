# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Checks for ``benchmarks/g9_compare.py``; run explicitly::

    python -m pytest benchmarks/check_g9_compare.py
"""
import json

import _bench_utils as bu
import g9_compare as g9


def _rec(device, precision, backend=None, affinity='all', ms=1.0, peak=2 ** 30, order=2500,
         status='ok'):
    r = {'schema_version': bu.SCHEMA_VERSION, 'benchmark': 'x', 'status': status,
         'config': {'device': device, 'order': order, 'precision': precision,
                    'brainevent_backend': backend, 'cpu_affinity': affinity}}
    if status == 'ok':
        r.update(steady={'ms_per_step': bu.summarize([ms])}, memory={'peak_bytes': peak},
                 compile={'compile_s': 1.0})
    return r


def _baseline():
    return [_rec('cpu', 32, affinity='4'), _rec('cpu', 64, affinity='4'),
            _rec('cpu', 32), _rec('gpu', 32), _rec('gpu', 64, status='failed'),
            _rec('gpu', 64, 'cusparse'), _rec('cpu', 32, order=200)]


def test_all_gated_pass_and_unpinned_reported():
    kern = [_rec('cpu', 32, affinity='4', ms=1.1), _rec('cpu', 64, affinity='4', ms=0.5),
            _rec('gpu', 32, peak=1.1 * 2 ** 30), _rec('gpu', 64, 'cusparse'),
            _rec('cpu', 32, ms=5.0)]
    rows = g9.compare(kern, _baseline())
    assert [r['status'] for r in rows] == ['pass'] * 4 + ['reported']
    assert g9.gate_passed(rows)
    assert abs(rows[0]['step_ratio'] - 1.1) < 1e-12 and rows[2]['mem_ratio'] == 1.1


def test_failures_and_missing_rows():
    kern = [_rec('cpu', 32, affinity='4', ms=1.2), _rec('cpu', 64, affinity='4', peak=2 ** 31),
            _rec('gpu', 32, status='failed')]
    rows = g9.compare(kern, _baseline())
    assert [r['status'] for r in rows] == ['fail', 'fail', 'not measured', 'not measured']
    assert not g9.gate_passed(rows)
    # gpu f64 default backend is not gated; its baseline failed: reported as no baseline
    rows = g9.compare([_rec('gpu', 64)], _baseline())
    assert len(rows) == 5 and not rows[4]['gated'] and rows[4]['status'] == 'no baseline'
    # a kernel configuration with no baseline row at all is not listed
    assert len(g9.compare([_rec('cpu', 32, affinity='0-3')], _baseline())) == 4


def test_no_baseline_and_tolerance():
    rows = g9.compare([_rec('cpu', 32, affinity='4', ms=1.2)], [])
    assert rows[0]['status'] == 'no baseline' and not g9.gate_passed(rows)
    rows = g9.compare([_rec('cpu', 32, affinity='4', ms=1.2)], _baseline(), tol=1.25)
    assert rows[0]['status'] == 'pass'


def test_markdown_and_main(tmp_path, capsys):
    kern = [_rec(k[0], k[2], k[3], k[4]) for k in g9.GATED_KEYS]
    kp, bp = tmp_path / 'k.json', tmp_path / 'b.json'
    kp.write_text(json.dumps({'records': kern}))
    bp.write_text(json.dumps({'records': _baseline()}))
    assert g9.main([str(kp), '--baseline', str(bp)]) == 0
    out = capsys.readouterr().out
    assert 'cpu pinned 4 order 2500 f32' in out and 'gpu cusparse order 2500 f64' in out
    assert 'G9: PASS' in out
    kp.write_text(json.dumps({'records': kern[:1]}))
    assert g9.main([str(kp), '--baseline', str(bp)]) == 1
    assert '| not measured |' in capsys.readouterr().out
