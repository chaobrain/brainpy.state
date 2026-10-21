# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Checks for ``benchmarks/_bench_utils.py``.

Named ``check_*`` so the default pytest collection (``test_*.py`` / ``*_test.py``)
never picks it up; run it explicitly::

    python -m pytest benchmarks/check_bench_utils.py
"""
import json

import pytest

import _bench_utils as bu


def _ok_record(device='cpu', order=200, precision=32, backend=None, median=1.0,
               affinity='all'):
    return {
        'schema_version': bu.SCHEMA_VERSION,
        'benchmark': 'brunel_simulator_baseline',
        'status': 'ok',
        'config': {'device': device, 'order': order, 'precision': precision,
                   'brainevent_backend': backend, 'cpu_affinity': affinity},
        'meta': {'loadavg_1min': 0.5},
        'build': {'build_s': 2.0},
        'compile': {'compile_s': 1.0, 'first_call_s': 1.5},
        'steady': {'s_per_sim_second': bu.summarize([median]),
                   'ms_per_step': bu.summarize([median / 10])},
        'memory': {'peak_bytes': 2 ** 30},
        'sanity': {'e_rate_hz': bu.summarize([30.0])},
    }


def test_summarize_median_even_and_odd():
    assert bu.summarize([3, 1, 2])['median'] == 2
    s = bu.summarize([4, 1, 3, 2])
    assert s['median'] == 2.5 and s['min'] == 1 and s['max'] == 4 and s['n'] == 4
    assert s['samples'] == [4.0, 1.0, 3.0, 2.0]


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        bu.summarize([])


def test_ru_maxrss_units():
    assert bu.ru_maxrss_bytes(10, platform='linux') == 10240
    assert bu.ru_maxrss_bytes(10, platform='darwin') == 10
    assert bu.peak_rss_bytes() > 0


def test_current_rss_parsing():
    text = 'Name:\tpython\nVmPeak:\t  999 kB\nVmRSS:\t  2048 kB\n'
    assert bu.current_rss_bytes(text) == 2048 * 1024
    assert bu.current_rss_bytes('Name: x\n') is None


def test_cpu_model_parsing():
    assert bu.cpu_model('processor\t: 0\nmodel name\t: Intel i7\n') == 'Intel i7'
    assert bu.cpu_model('processor\t: 0\n') is None


def test_validate_accepts_ok_and_failed():
    bu.validate_record(_ok_record())
    rec = bu.failure_record({'device': 'gpu', 'order': 2500, 'precision': 64}, {},
                            'OOM', returncode=1, stderr_tail='RESOURCE_EXHAUSTED')
    assert rec['status'] == 'failed' and rec['error'] == 'OOM'


@pytest.mark.parametrize('mutate, msg', [
    (lambda r: r.pop('meta'), 'misses keys'),
    (lambda r: r.update(schema_version=0), 'schema_version'),
    (lambda r: r.update(status='weird'), 'unknown status'),
    (lambda r: r['config'].pop('order'), "config misses 'order'"),
    (lambda r: r['config'].update(precision=16), 'precision'),
    (lambda r: r.pop('steady'), 'ok record misses'),
    (lambda r: r['steady'].pop('s_per_sim_second'), 's_per_sim_second'),
    (lambda r: r.update(status='failed'), "'error'"),
])
def test_validate_rejects(mutate, msg):
    rec = _ok_record()
    mutate(rec)
    with pytest.raises(ValueError, match=msg):
        bu.validate_record(rec)


@pytest.mark.parametrize('text, rc, label', [
    ('jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory', 1, 'OOM'),
    ('MemoryError', 1, 'OOM'),
    ('', -9, 'killed (SIGKILL)'),
    ('expects matching vector dtype', -6, 'abort (SIGABRT)'),
    ('', -11, 'signal 11'),
    ('ValueError: bloom room', 1, 'error'),   # no false 'oom' substring match
    (None, None, 'error'),
])
def test_classify_failure(text, rc, label):
    assert bu.classify_failure(text, rc) == label


def test_parse_result_lines_with_prefix_noise():
    rec = _ok_record()
    text = ('loadavg=0.1\n# sweep\n'
            'host$ ' + bu.RESULT_PREFIX + json.dumps(rec) + '\nother\n')
    assert bu.parse_result_lines(text) == [rec]
    assert bu.parse_result_lines('nothing here') == []


def test_merge_replaces_same_key_and_keeps_variants():
    a = _ok_record(median=1.0)
    b = _ok_record(precision=64)
    a2 = _ok_record(median=9.0)
    c = _ok_record(backend='cusparse')
    d = _ok_record(affinity='4')
    merged = bu.merge_records([a, b], [a2, c, d])
    assert len(merged) == 4
    assert merged[0]['steady']['s_per_sim_second']['median'] == 9.0
    assert merged[2]['config']['brainevent_backend'] == 'cusparse'
    assert merged[3]['config']['cpu_affinity'] == '4'


def test_config_key_defaults_affinity_to_all():
    rec = _ok_record()
    del rec['config']['cpu_affinity']
    assert bu.config_key(rec)[-1] == 'all'


@pytest.mark.parametrize('cpus, n, label', [
    (range(28), 28, 'all'),
    ([4], 28, '4'),
    (range(16), 28, '0-15'),
    ([0, 1, 2, 5, 7, 8], 28, '0-2,5,7-8'),
    ([3, 1, 2], 28, '1-3'),
])
def test_affinity_label(cpus, n, label):
    assert bu.affinity_label(cpus, n) == label


def test_sort_and_table():
    recs = [_ok_record(device='gpu', order=2500, precision=64),
            bu.failure_record({'device': 'gpu', 'order': 2500, 'precision': 64,
                               'brainevent_backend': 'x'}, {}, 'OOM'),
            _ok_record(device='cpu', order=2500),
            _ok_record(device='cpu', order=200, precision=64),
            _ok_record(device='cpu', order=200)]
    keys = [bu.config_key(r) for r in bu.sort_records(recs)]
    assert keys[:3] == [('cpu', 200, 32, None, 'all'), ('cpu', 200, 64, None, 'all'),
                        ('cpu', 2500, 32, None, 'all')]
    table = bu.markdown_table(recs).splitlines()
    assert len(table) == 2 + len(recs)
    assert all(line.count('|') == table[0].count('|') for line in table)
    assert 'FAILED (OOM)' in table[-1]
    assert '| 1.000 |' in table[2]   # peak memory in GiB
