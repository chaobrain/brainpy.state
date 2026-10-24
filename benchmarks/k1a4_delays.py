# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K1a-4 gate: per-edge delays (one scatter) vs the same network with a scalar delay.

Two networks:

* ``review`` -- the post-K0 review's setup (``/tmp/k0/review/delays_exp.py``): 2000
  ``iaf_psc_alpha`` neurons driven one-to-one by 2000 Poisson units (20 kHz,
  0.5 pA, delay 0.1 ms), recurrent ``fixed_indegree(100)``, 1 pA, delay 1.5 ms
  (scalar) or ``Uniform(0.1 ms, D ms)`` per edge;
* ``brunel`` -- Brunel model A (:func:`~brainspike._kernel.validation.brunel.build_brunel`)
  at ``--order``, recurrent delays 1.5 ms (scalar) or ``Uniform(0.1 ms, D ms)``.

``run`` measures one configuration in this process and prints one ``RESULT_JSON``
line: build time (``net.compile()`` + ``plan.runner()``, as in the review), first
call, steady ms/step (median of ``--repeats`` windows), delivery and group counts.
``sweep`` runs every configuration in a fresh process, interleaving the delay
settings round by round, each pinned to one CPU with ``taskset`` (``--cpu``, or
the least-used idle core picked as in ``g9_interleaved.py``), and prints a table
with the ratios to the scalar-delay run of the same network and precision
(gate: step <= 1.5x, build <= 2x).

``--delivery grouped|scatter`` overrides the cost model (calibration).

Usage::

    BENCH=1 /tmp/k0/remote.sh K1a-4 'python benchmarks/k1a4_delays.py sweep --net review \
        --delays scalar 2 5 20 --precisions 32 64 --rounds 3'
    GPU=1 /tmp/k0/remote.sh K1a-4 'python benchmarks/k1a4_delays.py sweep --no-pin ...'
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PREFIX = 'RESULT_JSON '


def _build(args):
    import brainunit as u
    from brainspike._kernel.connectivity import fixed_indegree, one_to_one
    from brainspike._kernel.models import iaf_psc_alpha, poisson_generator
    from brainspike._kernel.network import Network
    from brainspike._kernel.params import Uniform

    d = 1.5 * u.ms if args.delay == 'scalar' else Uniform(0.1 * u.ms, float(args.delay) * u.ms)
    if args.net == 'review':
        n = args.n
        net = Network(dt=0.1 * u.ms, seed=0)
        E = net.add(iaf_psc_alpha, n, name='E')
        P = net.add(poisson_generator, n, params=dict(rate=20000 * u.Hz), name='P')
        net.connect(P, E, one_to_one(), port='ex', weight=0.5 * u.pA, delay=0.1 * u.ms, name='PE')
        net.connect(E, E, fixed_indegree(args.indegree), port='ex', weight=1 * u.pA, delay=d,
                    name='EE')
        rec = net.observe(E[:50], 'spike', name='rec')
        return net, rec
    from brainspike._kernel.validation.brunel import build_brunel
    net, info = build_brunel(args.order, delay=d)
    return net, info['rec_E']


def run_one(args):
    os.environ.setdefault('BRAINSPIKE_ALLOW_DIRECT_IMPORT', '1')
    sys.path.insert(0, REPO_ROOT)
    import jax
    jax.config.update('jax_enable_x64', args.precision == 64)
    import brainunit as u
    import numpy as np
    from brainspike._kernel import compiler

    backend = jax.default_backend()
    if args.brainevent_backend:  # upstream: brainevent#199 (float64 GPU needs 'cusparse')
        import brainevent
        brainevent.config.set_backend('gpu', args.brainevent_backend)
    over = {k: v for k, v in (('chunk', args.chunk), ('capacity_factor', args.capacity_factor))
            if v is not None}
    if over:
        compiler.DELAY_COST_MODELS[backend] = dataclasses.replace(
            compiler.delay_cost_model(backend), **over)
    if args.delivery != 'auto':
        m = compiler.delay_cost_model(backend)
        if args.delivery == 'scatter':
            m = dataclasses.replace(m, max_groups=0)
        else:
            m = dataclasses.replace(m, max_groups=10 ** 9, group_step_us=0.0, group_edge_ns=0.0)
        compiler.DELAY_COST_MODELS[backend] = m
    net, rec = _build(args)
    t0 = time.perf_counter()
    plan = net.compile(f'float{args.precision}')
    run = plan.runner()
    jax.block_until_ready(run.state)
    build_s = time.perf_counter() - t0
    n_steps = int(round(args.window_ms / 0.1))
    t0 = time.perf_counter()
    r = run.cont(args.window_ms * u.ms)
    jax.block_until_ready(r[rec]['spike'])
    first_s = time.perf_counter() - t0
    walls, spikes = [], []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        r = run.cont(args.window_ms * u.ms)
        jax.block_until_ready(r[rec]['spike'])
        walls.append(time.perf_counter() - t0)
        spikes.append(float(np.asarray(r[rec]['spike']).sum()))
    rate = statistics.mean(spikes) / 50 / (args.window_ms / 1000.0)
    pps = [p for p in plan.projections if p.delivery_cost is not None]
    rec_out = {
        'net': args.net, 'order': args.order if args.net == 'brunel' else args.n,
        'delay': args.delay, 'precision': args.precision, 'device': backend,
        'delivery_arg': args.delivery, 'brainevent_backend': args.brainevent_backend,
        'chunk': compiler.delay_cost_model(backend).chunk,
        'capacity': {p.name: [getattr(g.storage, 'capacity', None) for g in p.groups] for p in pps},
        'delivery': {p.name: p.delivery for p in pps},
        'groups': {p.name: len(p.groups) for p in plan.projections},
        'n_delay_values': {p.name: p.delivery_cost['n_groups'] for p in pps},
        'build_s': build_s, 'first_call_s': first_s,
        'ms_per_step': statistics.median(walls) / n_steps * 1e3,
        'ms_per_step_all': [w / n_steps * 1e3 for w in walls],
        'rate_hz': rate, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'loadavg_end': os.getloadavg()[0],
    }
    print(PREFIX + json.dumps(rec_out), flush=True)


def _pick_cpu():
    sys.path.insert(0, HERE)
    import g9_interleaved as g9
    picked = g9.pick_core(g9.sample(2.0))
    if picked is None:
        raise SystemExit('no idle CPU (with idle hyperthread siblings) found; retry later')
    return picked[0]


def sweep(args):
    cpu = args.cpu
    if cpu is None and not args.no_pin:
        cpu = _pick_cpu()
    rows = []
    for rnd in range(args.rounds):
        for prec in args.precisions:
            for delay in args.delays:
                for delivery in args.deliveries:
                    if delay == 'scalar' and delivery != 'auto':
                        continue
                    cmd = [sys.executable, os.path.abspath(__file__), 'run', '--net', args.net,
                           '--order', str(args.order), '--n', str(args.n),
                           '--indegree', str(args.indegree), '--delay', delay,
                           '--precision', str(prec), '--delivery', delivery,
                           '--window-ms', str(args.window_ms), '--repeats', str(args.repeats)]
                    if args.brainevent_backend:
                        cmd += ['--brainevent-backend', args.brainevent_backend]
                    if args.chunk is not None:
                        cmd += ['--chunk', str(args.chunk)]
                    if args.capacity_factor is not None:
                        cmd += ['--capacity-factor', str(args.capacity_factor)]
                    if cpu is not None:
                        cmd = ['taskset', '-c', str(cpu)] + cmd
                    out = subprocess.run(cmd, capture_output=True, text=True)
                    recs = [json.loads(l[len(PREFIX):]) for l in out.stdout.splitlines()
                            if l.startswith(PREFIX)]
                    if out.returncode or not recs:
                        print(f'FAILED {cmd}\n{out.stderr[-3000:]}', flush=True)
                        continue
                    rec = recs[0]
                    rec['round'] = rnd
                    rows.append(rec)
                    print(f'round {rnd} f{prec} delay={delay:>6} {delivery:>8}: '
                          f'build {rec["build_s"]:.2f} s first {rec["first_call_s"]:.2f} s '
                          f'{rec["ms_per_step"]:.4f} ms/step rate {rec["rate_hz"]:.1f} Hz '
                          f'{rec["delivery"]} values={rec["n_delay_values"]} '
                          f'load {rec["loadavg_end"]:.2f}', flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=1)
    print(table(rows))


def table(rows):
    def med(sel, key):
        v = [r[key] for r in sel]
        return statistics.median(v) if v else float('nan')

    lines = ['| net | size | prec | device | delay | delivery | build s | ratio | ms/step | ratio '
             '| first call s | gate |', '|' + '---|' * 12]
    keys = sorted({(r['net'], r['order'], r['precision'], r['device']) for r in rows})
    for net, size, prec, dev in keys:
        sel = [r for r in rows if (r['net'], r['order'], r['precision'], r['device']) ==
               (net, size, prec, dev)]
        base = [r for r in sel if r['delay'] == 'scalar']
        b_build, b_step = med(base, 'build_s'), med(base, 'ms_per_step')
        for delay in dict.fromkeys(r['delay'] for r in sel):
            for dv in dict.fromkeys(r['delivery_arg'] for r in sel if r['delay'] == delay):
                s = [r for r in sel if r['delay'] == delay and r['delivery_arg'] == dv]
                bs, st = med(s, 'build_s'), med(s, 'ms_per_step')
                rb, rs = bs / b_build, st / b_step
                gate = '' if delay == 'scalar' else ('pass' if rb <= 2.0 and rs <= 1.5 else 'FAIL')
                chosen = ','.join(sorted(set(v for r in s for v in r['delivery'].values()))) or '-'
                lines.append(f'| {net} | {size} | f{prec} | {dev} | {delay} | {dv}:{chosen} | '
                             f'{bs:.2f} | {rb:.2f} | {st:.4f} | {rs:.2f} | '
                             f'{med(s, "first_call_s"):.2f} | {gate} |')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--net', choices=['review', 'brunel'], default='review')
        sp.add_argument('--order', type=int, default=2500)
        sp.add_argument('--n', type=int, default=2000)
        sp.add_argument('--indegree', type=int, default=100)
        sp.add_argument('--window-ms', type=float, default=100.0)
        sp.add_argument('--repeats', type=int, default=5)
        sp.add_argument('--brainevent-backend', default=None,
                        help="brainevent GPU backend (float64 on GPU needs 'cusparse', brainevent#199)")
        sp.add_argument('--chunk', type=int, default=None,
                        help='override the cost model chunk (segments per scatter iteration)')
        sp.add_argument('--capacity-factor', type=float, default=None,
                        help='override the cost model capacity_factor (loop-free capacity)')

    r = sub.add_parser('run')
    common(r)
    r.add_argument('--delay', default='scalar')
    r.add_argument('--precision', type=int, choices=[32, 64], default=32)
    r.add_argument('--delivery', choices=['auto', 'grouped', 'scatter'], default='auto')
    s = sub.add_parser('sweep')
    common(s)
    s.add_argument('--delays', nargs='+', default=['scalar', '2', '5', '20'])
    s.add_argument('--precisions', nargs='+', type=int, default=[32, 64])
    s.add_argument('--deliveries', nargs='+', default=['auto'])
    s.add_argument('--rounds', type=int, default=3)
    s.add_argument('--cpu', type=int, default=None)
    s.add_argument('--no-pin', action='store_true')
    s.add_argument('--out', default=None)
    t = sub.add_parser('table')
    t.add_argument('files', nargs='+')
    args = ap.parse_args(argv)
    if args.cmd == 'run':
        run_one(args)
    elif args.cmd == 'sweep':
        sweep(args)
    else:
        rows = []
        for f in args.files:
            with open(f) as fh:
                rows += json.load(fh)
        print(table(rows))


if __name__ == '__main__':
    main()
