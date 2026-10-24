# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K0 kernel runtime benchmark on the Brunel alpha network (T5 smoke, gate G9 in T10).

Same network, protocol and record schema as ``brunel_simulator_baseline.py``
(T0): 1000 ms windows through the jitted ``Runner.cont`` (compiled once, reused),
warm-up until a call triggers no compilation, then 7 timed windows; the median
ms/step is compared with the T0 baseline by `g9_compare.py` (gate G9).

::

    python benchmarks/brunel_kernel.py run --order 2500 --precision 32
    python benchmarks/brunel_kernel.py sweep --orders 200 2500 --precisions 32 64 \
        --out benchmarks/results/2026-10-22-T5-kernel-smoke.json
    # merge RESULT_JSON lines captured from remote logs (gate G9)
    python benchmarks/brunel_kernel.py collect cpu.log gpu.log \
        --out benchmarks/results/2026-10-22-T10-kernel-g9.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_utils as bu  # noqa: E402
from brunel_simulator_baseline import (  # noqa: E402
    DEFAULT_WINDOW_MS,
    _affinity,
    _device_label_from_env,
    _gpu_info,
    _versions,
    base_meta,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK = 'brunel_kernel'


def run_one(args):
    """Measure one (order, precision) configuration in this process."""
    meta = base_meta(args)
    os.environ.setdefault('BRAINSPIKE_ALLOW_DIRECT_IMPORT', '1')
    sys.path.insert(0, REPO_ROOT)

    import jax
    jax.config.update('jax_enable_x64', args.precision == 64)
    import brainunit as u
    import brainevent
    from brainspike._kernel.validation.brunel import build_brunel, rates

    if args.brainevent_backend:  # optional GPU backend override (e.g. 'cusparse')
        brainevent.config.set_backend('gpu', args.brainevent_backend)
    dev = jax.devices()[0]
    device = 'gpu' if dev.platform in ('gpu', 'cuda', 'rocm') else dev.platform
    is_gpu = device == 'gpu'
    meta.update(versions=_versions(), device_kind=dev.device_kind, device_platform=dev.platform,
                n_devices=jax.device_count(), gpu_info=_gpu_info() if is_gpu else None)

    def mem_in_use():
        return int(dev.memory_stats()['bytes_in_use']) if is_gpu else bu.current_rss_bytes()

    window_ms = args.window_ms or DEFAULT_WINDOW_MS
    n_steps = int(round(window_ms / 0.1))
    config = {'device': device, 'order': args.order, 'precision': args.precision,
              'window_ms': window_ms, 'steps_per_window': n_steps, 'repeats': args.repeats,
              'brainevent_backend': args.brainevent_backend, 'storage': args.storage,
              'cpu_affinity': _affinity()}

    compile_events = []

    def _listener(event, duration, **kw):
        if event.startswith('/jax/core/compile/'):
            compile_events.append((event, float(duration)))

    jax.monitoring.register_event_duration_secs_listener(_listener)

    mem_before = mem_in_use()
    t0 = time.perf_counter()
    net, info = build_brunel(args.order, storage=args.storage)
    plan = net.compile(f'float{args.precision}')
    run = plan.runner()
    jax.block_until_ready(run.state)
    build_s = time.perf_counter() - t0
    mem_after = mem_in_use()

    def timed_call():
        n0 = len(compile_events)
        t = time.perf_counter()
        res = run.cont(window_ms * u.ms)
        jax.block_until_ready(res[info['rec_E']]['spike'])
        wall = time.perf_counter() - t
        comp = sum(d for _, d in compile_events[n0:])
        return (wall, comp, len(compile_events) - n0) + tuple(rates(res, info))

    warm = []
    for k in range(max(args.max_warmup, 2)):
        warm.append(timed_call())
        if k >= 1 and warm[-1][2] == 0:
            break
    else:
        raise RuntimeError(f'still compiling after {args.max_warmup} warm-up calls')
    timed = [timed_call() for _ in range(args.repeats)]
    walls = [t[0] for t in timed]

    if is_gpu:
        stats = dev.memory_stats()
        mem = {'kind': 'gpu_peak_bytes_in_use', 'peak_bytes': int(stats['peak_bytes_in_use']),
               'before_build_bytes': mem_before, 'after_build_bytes': mem_after,
               'host_peak_rss_bytes': bu.peak_rss_bytes()}
    else:
        mem = {'kind': 'cpu_peak_rss', 'peak_bytes': bu.peak_rss_bytes(),
               'before_build_bytes': mem_before, 'after_build_bytes': mem_after}
    record = {
        'schema_version': bu.SCHEMA_VERSION,
        'benchmark': BENCHMARK,
        'status': 'ok',
        'config': config,
        'meta': meta,
        'network': {k: v for k, v in info['params'].items() if isinstance(v, (int, float, str))},
        'build': {'build_s': build_s, 'plan_memory_bytes': plan.memory()},
        'compile': {
            # first call = trace + lower + XLA compile + one window
            'compile_s': warm[0][0] - (sum(walls) / len(walls)),
            'first_call_s': warm[0][0],
            'warmup_xla_events_s': sum(w[1] for w in warm),
            'n_compiling_calls': sum(1 for w in warm if w[2] > 0),
            'recompiles_during_timing': sum(t[2] for t in timed),
            'n_traces': run.n_traces,
        },
        'steady': {
            'window_wall_s': bu.summarize(walls),
            's_per_sim_second': bu.summarize([w / (window_ms / 1000.0) for w in walls]),
            'ms_per_step': bu.summarize([w / n_steps * 1e3 for w in walls]),
        },
        'memory': mem,
        'sanity': {'e_rate_hz': bu.summarize([t[3] for t in timed]),
                   'i_rate_hz': bu.summarize([t[4] for t in timed])},
    }
    meta['loadavg_1min_end'] = os.getloadavg()[0]
    bu.validate_record(record)
    print(bu.RESULT_PREFIX + json.dumps(record), flush=True)
    return record


def sweep(args):
    """Each configuration in a fresh subprocess; prints records and a table."""
    records = []
    for order in args.orders:
        for prec in args.precisions:
            cmd = [sys.executable, os.path.abspath(__file__), 'run', '--order', str(order),
                   '--precision', str(prec), '--repeats', str(args.repeats),
                   '--max-warmup', str(args.max_warmup), '--commit', str(args.commit),
                   '--storage', args.storage]
            if args.window_ms:
                cmd += ['--window-ms', str(args.window_ms)]
            if args.brainevent_backend:
                cmd += ['--brainevent-backend', args.brainevent_backend]
            print(f'# sweep: order={order} precision={prec}', flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            found = bu.parse_result_lines(proc.stdout)
            if proc.returncode == 0 and found:
                rec = found[-1]
            else:
                tail = '\n'.join(proc.stderr.strip().splitlines()[-25:])
                cfg = {'device': _device_label_from_env(), 'order': order, 'precision': prec,
                       'brainevent_backend': args.brainevent_backend,
                       'cpu_affinity': _affinity()}
                rec = bu.failure_record(cfg, base_meta(args),
                                        bu.classify_failure(tail, proc.returncode),
                                        returncode=proc.returncode, stderr_tail=tail)
                rec['benchmark'] = BENCHMARK
            print(bu.RESULT_PREFIX + json.dumps(rec), flush=True)
            records.append(rec)
    print(bu.markdown_table(records))
    if args.out:
        write_results(args.out, records)


def write_results(path, records):
    """Merge ``records`` into the results JSON at ``path`` (created if absent)."""
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)['records']
    merged = bu.sort_records(bu.merge_records(existing, records))
    for r in merged:
        bu.validate_record(r)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'benchmark': BENCHMARK, 'schema_version': bu.SCHEMA_VERSION,
                   'records': merged}, f, indent=1)
        f.write('\n')
    return merged


def collect(args):
    """Parse RESULT_JSON lines from (remote) log files and merge them into ``--out``."""
    records = []
    for p in args.logs:
        with open(p) as f:
            records = bu.merge_records(records, bu.parse_result_lines(f.read()))
    print(bu.markdown_table(write_results(args.out, records)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--repeats', type=int, default=7)
        sp.add_argument('--window-ms', type=float, default=None)
        sp.add_argument('--max-warmup', type=int, default=4)
        sp.add_argument('--commit', default=os.environ.get('BENCH_COMMIT'))
        sp.add_argument('--brainevent-backend', default=None)
        sp.add_argument('--storage', default='auto', choices=('auto', 'dense', 'csr'))

    r = sub.add_parser('run')
    r.add_argument('--order', type=int, required=True)
    r.add_argument('--precision', type=int, choices=(32, 64), required=True)
    common(r)
    s = sub.add_parser('sweep')
    s.add_argument('--orders', type=int, nargs='+', default=[200, 2500])
    s.add_argument('--precisions', type=int, nargs='+', default=[32, 64])
    s.add_argument('--timeout', type=float, default=3600)
    s.add_argument('--out', default=None)
    common(s)
    c = sub.add_parser('collect')
    c.add_argument('logs', nargs='+')
    c.add_argument('--out', required=True)
    args = p.parse_args(argv)
    {'run': run_one, 'sweep': sweep, 'collect': collect}[args.cmd](args)


if __name__ == '__main__':
    main()
