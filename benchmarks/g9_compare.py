# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate G9: compare K0 kernel benchmark records with the T0 ``Simulator`` baseline.

Both files use the ``_bench_utils`` result schema. A kernel record is matched to the
baseline record with the same configuration key ``(device, order, precision,
brainevent backend, cpu affinity)``. The gated rows (spec §6 G9) are order 2500 at
float32 and float64:

* CPU pinned (``taskset -c 4``) against the T0 pinned rows;
* GPU float32, brainevent default backend, against the T0 default-backend row;
* GPU float64 with ``--brainevent-backend cusparse`` against the T0 cusparse row (the
  T0 default backend aborts at float64 on the GPU, brainevent 0.2.1).

A gated row passes when the median step time and the peak memory are both at most
``1.1x`` the baseline. Unpinned CPU rows and any other matched rows are reported but
not gated. Compile time is reported, not gated.

Usage::

    python benchmarks/g9_compare.py KERNEL.json [--baseline T0.json] [--tol 1.1]

Exit status 0 when every gated row that is present passes, 1 otherwise; gated rows
with no kernel record are listed as "not measured".
"""
import argparse
import json
import os
import sys

import _bench_utils as bu

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASELINE = os.path.join(HERE, 'results', '2026-10-21-T0-simulator-baseline.json')
GATED_ORDER = 2500
#: configuration keys of the gated rows (see the module docstring)
GATED_KEYS = (
    ('cpu', GATED_ORDER, 32, None, '4'),
    ('cpu', GATED_ORDER, 64, None, '4'),
    ('gpu', GATED_ORDER, 32, None, 'all'),
    ('gpu', GATED_ORDER, 64, 'cusparse', 'all'),
)


def _metrics(record):
    """``(ms/step median, peak bytes, compile s)`` of an ``ok`` record, else ``None``."""
    if record is None or record.get('status') != 'ok':
        return None
    return (float(record['steady']['ms_per_step']['median']),
            float(record['memory']['peak_bytes']),
            float(record['compile'].get('compile_s', float('nan'))))


def compare(kernel_records, baseline_records, tol=1.1):
    """Rows ``{key, gated, status, step, mem, ratios...}`` for every gated key and every
    other kernel record that has a baseline match.

    ``status`` is ``'pass'`` / ``'fail'`` for gated rows that were measured,
    ``'not measured'`` when the kernel record is missing or failed, ``'no baseline'``
    when the baseline row is missing, and ``'reported'`` for non-gated rows.
    """
    base = {bu.config_key(r): r for r in baseline_records}
    kern = {bu.config_key(r): r for r in kernel_records}
    keys = list(GATED_KEYS) + sorted((k for k in kern if k not in GATED_KEYS and k in base),
                                     key=lambda k: tuple(str(x) for x in k))
    rows = []
    for key in keys:
        gated = key in GATED_KEYS
        k, b = _metrics(kern.get(key)), _metrics(base.get(key))
        row = {'key': key, 'gated': gated, 'kernel': k, 'baseline': b,
               'step_ratio': None, 'mem_ratio': None}
        if k is None:
            row['status'] = 'not measured'
        elif b is None:
            row['status'] = 'no baseline'
        else:
            row['step_ratio'] = k[0] / b[0]
            row['mem_ratio'] = k[1] / b[1]
            if gated:
                ok = row['step_ratio'] <= tol and row['mem_ratio'] <= tol
                row['status'] = 'pass' if ok else 'fail'
            else:
                row['status'] = 'reported'
        rows.append(row)
    return rows


def gate_passed(rows):
    """``True`` when every gated row was measured and passed."""
    return all(r['status'] == 'pass' for r in rows if r['gated'])


def _label(key):
    dev, order, prec, backend, aff = key
    pin = '' if aff == 'all' else f' pinned {aff}'
    be = f' {backend}' if backend else ''
    return f'{dev}{pin}{be} order {order} f{prec}'


def markdown(rows):
    """Markdown table of :func:`compare` rows."""
    gib = 2.0 ** 30
    out = ['| config | gated | kernel ms/step | baseline ms/step | ratio | kernel peak GiB '
           '| baseline peak GiB | ratio | kernel compile s | baseline compile s | result |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        k, b = r['kernel'], r['baseline']
        cells = [_label(r['key']), 'yes' if r['gated'] else 'no',
                 bu._fmt(k[0], 4) if k else '-', bu._fmt(b[0], 4) if b else '-',
                 bu._fmt(r['step_ratio'], 3),
                 bu._fmt(k[1] / gib, 3) if k else '-', bu._fmt(b[1] / gib, 3) if b else '-',
                 bu._fmt(r['mem_ratio'], 3),
                 bu._fmt(k[2], 2) if k else '-', bu._fmt(b[2], 2) if b else '-',
                 r['status']]
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def _load(path):
    with open(path) as f:
        return json.load(f)['records']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('kernel')
    p.add_argument('--baseline', default=DEFAULT_BASELINE)
    p.add_argument('--tol', type=float, default=1.1)
    args = p.parse_args(argv)
    rows = compare(_load(args.kernel), _load(args.baseline), args.tol)
    print(markdown(rows))
    ok = gate_passed(rows)
    print(f'\nG9: {"PASS" if ok else "FAIL / incomplete"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
