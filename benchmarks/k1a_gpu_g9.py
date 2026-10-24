# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K1a-G3: gate G9 on the GPU -- Brunel alpha, old ``Simulator`` vs the kernel, GPU 1.

Configurations (K1a spec §K1a-7): orders 200 and 2500; float32 with brainevent's
default GPU backend, float64 with the ``cusparse`` backend (chosen in K1a-7 because
brainevent 0.2.1's default GPU kernel aborted on a float64 event vector, brainevent#199;
fixed in 0.3.0, but the f64 rows keep ``cusparse`` so they stay comparable with the
recorded K1a-7 baseline; moving them to the default backend is a K1b re-measurement).
The reference is the T0 GPU row of the
old ``Simulator`` (``results/2026-10-21-T0-simulator-baseline.json``); the old
``Simulator`` is also re-measured here, alternating with the kernel in fresh processes
(old, new, old, new, ...), with the T0 protocol inside every process
(``brunel_simulator_baseline.py run`` / ``brunel_kernel.py run``: warm-up until no
compilation, 7 timed 1000 ms windows, median ms/step, device ``peak_bytes_in_use``).

Gate: kernel median step time and peak device memory <= 1.1 x the old ``Simulator``
(T0 row; the re-measured old rows are reported next to it). Compile time is reported.

Usage (remote host, GPU 1 only)::

    GPU=1 BENCH=1 /tmp/k0/remote.sh K1a-7 'python benchmarks/k1a_gpu_g9.py run --rounds 3' > g3.log
    python benchmarks/k1a_gpu_g9.py compare g3.log --out benchmarks/results/<date>-K1a-G3-gpu-g9.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bench_utils as bu  # noqa: E402

T0_RESULTS = os.path.join(HERE, 'results', '2026-10-21-T0-simulator-baseline.json')
SCRIPTS = {'old': 'brunel_simulator_baseline.py', 'new': 'brunel_kernel.py'}
CONFIGS = ((2500, 32, None), (2500, 64, 'cusparse'), (200, 32, None), (200, 64, 'cusparse'))
RUN_PREFIX = 'G3_RUN '
TOL = 1.1


def _gpu1_busy() -> bool:
    out = subprocess.run(['nvidia-smi', '-i', os.environ.get('CUDA_VISIBLE_DEVICES', '1'),
                          '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def run(args) -> None:
    for rnd in range(args.rounds):
        for order, prec, backend in CONFIGS:
            for impl in ('old', 'new'):
                if _gpu1_busy():
                    print('ABORT: GPU busy', flush=True)
                    sys.exit(3)
                cmd = [sys.executable, os.path.join(HERE, SCRIPTS[impl]), 'run', '--order',
                       str(order), '--precision', str(prec), '--commit', str(args.commit),
                       '--repeats', str(args.repeats)]
                if backend:
                    cmd += ['--brainevent-backend', backend]
                load = os.getloadavg()[0]
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                recs = bu.parse_result_lines(p.stdout)
                rec = recs[-1] if (p.returncode == 0 and recs) else {
                    'status': 'failed', 'config': {'order': order, 'precision': prec},
                    'stderr_tail': p.stderr[-2000:]}
                rec['g3'] = dict(impl=impl, round=rnd, backend=backend or 'default',
                                 load_start=load)
                print(RUN_PREFIX + json.dumps(rec), flush=True)


def _key(r):
    return int(r['config']['order']), int(r['config']['precision'])


def compare_runs(runs, t0_records) -> list:
    t0 = {}
    for r in t0_records:
        c = r['config']
        if c['device'] == 'gpu' and r.get('status') == 'ok':
            be = c.get('brainevent_backend') or 'default'
            t0[(int(c['order']), int(c['precision']), be)] = r
    rows = []
    for order, prec, backend in CONFIGS:
        be = backend or 'default'
        row = dict(order=order, precision=prec, backend=be)
        for impl in ('old', 'new'):
            sel = [r for r in runs if r.get('status') == 'ok' and _key(r) == (order, prec)
                   and r['g3']['impl'] == impl]
            row[impl] = dict(
                n=len(sel),
                ms_per_step=statistics.median([r['steady']['ms_per_step']['median'] for r in sel])
                if sel else None,
                peak_bytes=statistics.median([r['memory']['peak_bytes'] for r in sel]) if sel else None,
                compile_s=statistics.median([r['compile']['compile_s'] for r in sel]) if sel else None,
                e_rate=statistics.median([r['sanity']['e_rate_hz']['median'] for r in sel])
                if sel else None)
        ref = t0.get((order, prec, be))
        row['t0'] = None if ref is None else dict(
            ms_per_step=ref['steady']['ms_per_step']['median'], peak_bytes=ref['memory']['peak_bytes'],
            compile_s=ref['compile']['compile_s'])
        n = row['new']
        if n['n'] and row['t0']:
            row['step_ratio_t0'] = n['ms_per_step'] / row['t0']['ms_per_step']
            row['mem_ratio_t0'] = n['peak_bytes'] / row['t0']['peak_bytes']
            row['status'] = 'pass' if (row['step_ratio_t0'] <= TOL and row['mem_ratio_t0'] <= TOL) else 'FAIL'
        else:
            row['status'] = 'not measured'
        if n['n'] and row['old']['n']:
            row['step_ratio_old'] = n['ms_per_step'] / row['old']['ms_per_step']
            row['mem_ratio_old'] = n['peak_bytes'] / row['old']['peak_bytes']
        rows.append(row)
    return rows


def markdown(rows) -> str:
    gib = 2.0 ** 30
    f = lambda x, nd=4: '-' if x is None else f'{x:.{nd}f}'
    lines = ['| order | prec | backend | T0 old ms/step | old (now) ms/step | new ms/step | ratio vs T0 | '
             'ratio vs old now | T0 peak GiB | old (now) peak GiB | new peak GiB | mem ratio vs T0 | '
             'mem ratio vs old now | old / new compile s | new E rate Hz | result |',
             '|' + '---|' * 16]
    for r in rows:
        t0, o, n = r['t0'] or {}, r['old'], r['new']
        lines.append(
            f"| {r['order']} | f{r['precision']} | {r['backend']} | {f(t0.get('ms_per_step'))} | "
            f"{f(o['ms_per_step'])} | {f(n['ms_per_step'])} | {f(r.get('step_ratio_t0'), 3)} | "
            f"{f(r.get('step_ratio_old'), 3)} | "
            f"{f(t0['peak_bytes'] / gib if t0 else None, 3)} | "
            f"{f(o['peak_bytes'] / gib if o['peak_bytes'] else None, 3)} | "
            f"{f(n['peak_bytes'] / gib if n['peak_bytes'] else None, 3)} | "
            f"{f(r.get('mem_ratio_t0'), 3)} | {f(r.get('mem_ratio_old'), 3)} | "
            f"{f(o['compile_s'], 2)} / {f(n['compile_s'], 2)} | {f(n['e_rate'], 2)} | {r['status']} |")
    return '\n'.join(lines)


def compare(args) -> None:
    runs = []
    for p in args.logs:
        with open(p) as fh:
            runs += [json.loads(line[len(RUN_PREFIX):]) for line in fh if line.startswith(RUN_PREFIX)]
    with open(args.t0) as fh:
        t0 = json.load(fh)['records']
    rows = compare_runs(runs, t0)
    print(markdown(rows))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(dict(rows=rows, runs=runs), fh, indent=1)
            fh.write('\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--rounds', type=int, default=3)
    r.add_argument('--repeats', type=int, default=7)
    r.add_argument('--commit', default=os.environ.get('BENCH_COMMIT', 'unknown'))
    r.add_argument('--timeout', type=float, default=3600)
    c = sub.add_parser('compare')
    c.add_argument('logs', nargs='+')
    c.add_argument('--t0', default=T0_RESULTS)
    c.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    (run if a.cmd == 'run' else compare)(a)


if __name__ == '__main__':
    main()
