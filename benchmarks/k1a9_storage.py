# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K1a-9: Dense vs CSR storage for the recurrent projections of a Brunel-like network.

Measures what ``make_storage('auto')`` should pick. The network is Brunel model A
(``iaf_psc_alpha``, ``N = 5 * order``, Poisson drive) with connection probability
``epsilon``; recurrent weights are scaled by ``0.1 / epsilon`` so the mean recurrent
input (and roughly the activity) does not depend on ``epsilon``. Every configuration
runs in a fresh process: build, compile, warm-up until no compilation, then
``--repeats`` timed windows (median ms/step) and the device ``peak_bytes_in_use``
(GPU) or peak RSS (CPU).

::

    python benchmarks/k1a9_storage.py sweep --orders 50 200 800 --epsilons 0.1 0.2 0.5 \
        --precisions 32 > storage_cpu.log
    python benchmarks/k1a9_storage.py table storage_cpu.log storage_gpu.log
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PREFIX = 'K1A9_RUN '


def run_one(args):
    os.environ.setdefault('BRAINSPIKE_ALLOW_DIRECT_IMPORT', '1')
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, HERE)
    import jax
    jax.config.update('jax_enable_x64', args.precision == 64)
    import brainevent
    import brainunit as u
    import _bench_utils as bu
    from brainspike._kernel.connectivity import fixed_indegree, one_to_one
    from brainspike._kernel.models import iaf_psc_alpha, poisson_generator
    from brainspike._kernel.network import Network
    from brainspike._kernel.validation.brunel import brunel_params

    if args.brainevent_backend:  # optional GPU backend override (e.g. 'cusparse')
        brainevent.config.set_backend('gpu', args.brainevent_backend)
    dev = jax.devices()[0]
    gpu = dev.platform in ('gpu', 'cuda', 'rocm')
    p = brunel_params(args.order, epsilon=args.epsilon)
    scale = 0.1 / args.epsilon
    npar = dict(C_m=p['c_mem'] * u.pF, tau_m=p['tau_mem'] * u.ms, tau_syn_ex=p['tau_syn'] * u.ms,
                tau_syn_in=p['tau_syn'] * u.ms, t_ref=p['t_ref'] * u.ms, E_L=0.0 * u.mV,
                V_reset=0.0 * u.mV, V_th=p['theta'] * u.mV)
    t0 = time.perf_counter()
    net = Network(dt=0.1 * u.ms, seed=0)
    E = net.add(iaf_psc_alpha, p['N_E'], params=npar, name='E')
    I = net.add(iaf_psc_alpha, p['N_I'], params=npar, name='I')
    noise = net.add(poisson_generator, p['N_E'] + p['N_I'], params=dict(rate=p['rate'] * u.Hz),
                    name='noise')
    d = p['delay'] * u.ms
    net.connect(noise, E + I, one_to_one(), port='ex', weight=p['J_ex'] * u.pA, delay=d, name='noise_EI')
    net.connect(E, E + I, fixed_indegree(p['C_E']), port='ex', weight=p['J_ex'] * scale * u.pA,
                delay=d, storage=args.storage, name='E_EI')
    net.connect(I, E + I, fixed_indegree(p['C_I']), port='in_', weight=p['J_in'] * scale * u.pA,
                delay=d, storage=args.storage, name='I_EI')
    rec = net.observe(E[:50], 'spike', name='rec_E')
    plan = net.compile(f'float{args.precision}')
    kinds = [g.storage.kind for pp in plan.projections if pp.proj.name != 'noise_EI' for g in pp.groups]
    run = plan.runner()
    jax.block_until_ready(run.state)
    build_s = time.perf_counter() - t0
    n_steps = int(round(args.window_ms / 0.1))

    def window():
        t = time.perf_counter()
        res = run.cont(args.window_ms * u.ms)
        x = jax.block_until_ready(res[rec]['spike'])
        return time.perf_counter() - t, float(x.sum()) / 50 / (args.window_ms / 1000.0)

    t = time.perf_counter()
    window()
    first_s = time.perf_counter() - t
    n0 = run.n_traces
    window()
    if run.n_traces != n0:
        raise RuntimeError('recompiled during warm-up')
    timed = [window() for _ in range(args.repeats)]
    ms = statistics.median(w for w, _ in timed) / n_steps * 1e3
    peak = int(dev.memory_stats()['peak_bytes_in_use']) if gpu else bu.peak_rss_bytes()
    rec_out = dict(device='gpu' if gpu else 'cpu', order=args.order, epsilon=args.epsilon,
                   precision=args.precision, storage=args.storage, kinds=kinds,
                   n=p['N_E'] + p['N_I'], build_s=build_s, first_s=first_s, ms_per_step=ms,
                   peak_bytes=peak, peak_kind='device' if gpu else 'rss',
                   rate_hz=statistics.median(r for _, r in timed),
                   xla_flags=os.environ.get('XLA_FLAGS'), load=os.getloadavg()[0])
    print(PREFIX + json.dumps(rec_out), flush=True)
    os._exit(0)


def sweep(args):
    for order in args.orders:
        for eps in args.epsilons:
            for prec in args.precisions:
                for storage in args.storages:
                    cmd = [sys.executable, os.path.abspath(__file__), 'run', '--order', str(order),
                           '--epsilon', str(eps), '--precision', str(prec), '--storage', storage,
                           '--repeats', str(args.repeats), '--window-ms', str(args.window_ms)]
                    if prec == 64 and args.f64_backend:
                        cmd += ['--brainevent-backend', args.f64_backend]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(PREFIX)]
                    if lines:
                        print(lines[-1], flush=True)
                    else:
                        tail = ' | '.join(proc.stderr.strip().splitlines()[-3:])
                        print(f'# FAILED order={order} eps={eps} f{prec} {storage}: {tail}', flush=True)


def table(args):
    recs = []
    for path in args.logs:
        with open(path) as f:
            recs += [json.loads(ln[len(PREFIX):]) for ln in f if ln.startswith(PREFIX)]
    by = {}
    for r in recs:
        by[(r['device'], r['order'], r['epsilon'], r['precision'], r['storage'])] = r
    print('| device | order | N | epsilon | prec | dense ms/step | csr ms/step | csr / dense | '
          'dense peak MiB | csr peak MiB | rate Hz |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for key in sorted({k[:4] for k in by}):
        d, c = by.get(key + ('dense',)), by.get(key + ('csr',))
        if not (d and c):
            continue
        print(f'| {key[0]} | {key[1]} | {d["n"]} | {key[2]} | f{key[3]} | {d["ms_per_step"]:.4f} | '
              f'{c["ms_per_step"]:.4f} | {c["ms_per_step"] / d["ms_per_step"]:.2f} | '
              f'{d["peak_bytes"] / 2 ** 20:.1f} | {c["peak_bytes"] / 2 ** 20:.1f} | '
              f'{d["rate_hz"]:.1f} / {c["rate_hz"]:.1f} |')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'benchmark': 'k1a9_storage', 'records': recs}, f, indent=1)
            f.write('\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--order', type=int, required=True)
    r.add_argument('--epsilon', type=float, default=0.1)
    r.add_argument('--precision', type=int, choices=(32, 64), default=32)
    r.add_argument('--storage', choices=('auto', 'dense', 'csr'), default='auto')
    r.add_argument('--repeats', type=int, default=5)
    r.add_argument('--window-ms', type=float, default=200.0)
    r.add_argument('--brainevent-backend', default=None)
    s = sub.add_parser('sweep')
    s.add_argument('--orders', type=int, nargs='+', default=[50, 200, 800])
    s.add_argument('--epsilons', type=float, nargs='+', default=[0.1, 0.2, 0.5])
    s.add_argument('--precisions', type=int, nargs='+', default=[32])
    s.add_argument('--storages', nargs='+', default=['dense', 'csr'])
    s.add_argument('--repeats', type=int, default=5)
    s.add_argument('--window-ms', type=float, default=200.0)
    s.add_argument('--f64-backend', default=None)
    s.add_argument('--timeout', type=float, default=1800)
    t = sub.add_parser('table')
    t.add_argument('logs', nargs='+')
    t.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    {'run': run_one, 'sweep': sweep, 'table': table}[args.cmd](args)


if __name__ == '__main__':
    main()
