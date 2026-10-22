# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate G9: old ``Simulator`` vs K0 kernel, interleaved on one idle pinned core.

Protocol (K0 T10, orchestrator decision 2026-10-22; the remote host is shared with
long-running jobs, so the 1-minute load average cannot be brought below 2, and the GPU
is not measured):

1. **Core choice.** Per-CPU utilisation is sampled from ``/proc/stat`` over
   ``--sample-s`` seconds; the chosen CPU and every hyperthread sibling
   (``/sys/devices/system/cpu/cpuN/topology/thread_siblings_list``) must be below
   ``--idle-pct`` (5 %). ``--prefer`` (default 4, the T0 core) is taken when idle, else
   the least-used CPU that has a hyperthread sibling (a P-core on the i7-14700KF), else
   the least-used idle CPU. Every run is pinned there with ``taskset -c N``.
2. **Interleaving.** For each round and each ``(order, precision)``: one old
   (``brunel_simulator_baseline.py run``) then one new (``brunel_kernel.py run``) run,
   each in a fresh process, same T0 measurement protocol inside the process.
3. **Validity.** Before every run the core and its siblings are re-sampled (must be
   idle, else wait and re-sample); during the run the siblings must stay below
   ``--idle-pct`` and the child's CPU time / wall time (its share of the pinned core)
   must be at least ``--min-share``. A run failing either check is discarded and redone
   (``--max-attempts``). The CPU, siblings, idle %, load average and checks are stored
   with every record (``g9`` key).
4. **Gate.** Per configuration, the median over rounds of each run's median ms/step
   and of its peak RSS; G9 rows (order 2500) pass when new <= ``--tol`` x old on both.
   Other orders are reported. The original T0 pinned rows are compared for reference.

Usage (remote, CPU)::

    python benchmarks/g9_interleaved.py run --commit <sha> --rounds 3 > g9.log
    python benchmarks/g9_interleaved.py compare g9.log [--out results.json]

Only the standard library and ``_bench_utils`` are imported (no JAX).
"""
import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_utils as bu  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T0_RESULTS = os.path.join(HERE, 'results', '2026-10-21-T0-simulator-baseline.json')
SCRIPTS = {'old': 'brunel_simulator_baseline.py', 'new': 'brunel_kernel.py'}
RUN_PREFIX = 'G9RUN_JSON '
GATED_ORDERS = (2500,)


# ----------------------------------------------------------------------------- cpu state

def read_proc_stat(text=None):
    """``{cpu: (busy_jiffies, total_jiffies)}`` from ``/proc/stat`` (per-CPU lines)."""
    if text is None:
        with open('/proc/stat') as f:
            text = f.read()
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith('cpu') or parts[0] == 'cpu':
            continue
        vals = [int(v) for v in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        total = sum(vals[:8])  # user..steal; guest is already counted in user
        out[int(parts[0][3:])] = (total - idle, total)
    return out


def utilisation(before, after):
    """Percent busy per CPU between two :func:`read_proc_stat` snapshots."""
    out = {}
    for cpu, (b1, t1) in after.items():
        b0, t0 = before.get(cpu, (b1, t1))
        dt = t1 - t0
        out[cpu] = 100.0 * (b1 - b0) / dt if dt > 0 else 0.0
    return out


def sample(seconds, reader=read_proc_stat, sleep=time.sleep):
    """Per-CPU utilisation over ``seconds``."""
    a = reader()
    sleep(seconds)
    return utilisation(a, reader())


def parse_cpu_list(text):
    """``'4-5'`` / ``'4,5'`` / ``'0-1,8'`` -> sorted list of ints."""
    cpus = set()
    for part in text.strip().split(','):
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-')
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def siblings(cpu, root='/sys/devices/system/cpu'):
    """Hyperthread siblings of ``cpu`` (excluding itself); ``[]`` when unknown."""
    path = os.path.join(root, f'cpu{cpu}', 'topology', 'thread_siblings_list')
    try:
        with open(path) as f:
            return [c for c in parse_cpu_list(f.read()) if c != cpu]
    except OSError:
        return []


def pick_core(util, sibling_fn=siblings, idle_pct=5.0, prefer=4):
    """``(cpu, siblings)`` of an idle CPU whose siblings are idle too, or ``None``.

    ``prefer`` wins when idle; then CPUs with a hyperthread sibling (least used first),
    then any idle CPU.
    """
    def idle(c):
        return util.get(c, 100.0) < idle_pct and all(util.get(s, 100.0) < idle_pct
                                                    for s in sibling_fn(c))

    if prefer is not None and prefer in util and idle(prefer):
        return prefer, sibling_fn(prefer)
    cands = [c for c in sorted(util) if idle(c)]
    if not cands:
        return None
    ht = [c for c in cands if sibling_fn(c)]
    pool = ht or cands
    best = min(pool, key=lambda c: (util[c] + sum(util.get(s, 0.0) for s in sibling_fn(c)), c))
    return best, sibling_fn(best)


def loadavg():
    try:
        with open('/proc/loadavg') as f:
            return [float(x) for x in f.read().split()[:3]]
    except OSError:  # pragma: no cover - not Linux
        return None


# ----------------------------------------------------------------------------- runs

def run_command(impl, order, precision, cpu, args):
    cmd = ['taskset', '-c', str(cpu), sys.executable, os.path.join(HERE, SCRIPTS[impl]), 'run',
           '--order', str(order), '--precision', str(precision), '--commit', str(args.commit),
           '--repeats', str(args.repeats)]
    if args.window_ms:
        cmd += ['--window-ms', str(args.window_ms)]
    return cmd


def check_run(sib_util, share, idle_pct, min_share):
    """Reasons a finished run is invalid (empty list: valid)."""
    reasons = []
    busy = {s: u for s, u in sib_util.items() if u >= idle_pct}
    if busy:
        reasons.append(f'sibling busy during run: {busy}')
    if share < min_share:
        reasons.append(f'core share {share:.3f} < {min_share}')
    return reasons


def measure(impl, order, precision, cpu, sibs, args, round_idx, log=print):
    """Run one configuration until valid (or ``max_attempts``); return the run dict."""
    attempts = []
    for attempt in range(1, args.max_attempts + 1):
        pre = wait_idle(cpu, sibs, args, log)
        snap0 = read_proc_stat()
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        la0, t0 = loadavg(), time.monotonic()
        proc = subprocess.run(run_command(impl, order, precision, cpu, args),
                              capture_output=True, text=True, timeout=args.timeout)
        wall = time.monotonic() - t0
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        during = utilisation(snap0, read_proc_stat())
        share = ((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)) / wall
        recs = bu.parse_result_lines(proc.stdout)
        rec = recs[-1] if recs else None
        reasons = check_run({s: during.get(s, 0.0) for s in sibs}, share,
                            args.idle_pct, args.min_share)
        if proc.returncode != 0 or rec is None or rec.get('status') != 'ok':
            reasons.append(f'run failed (rc {proc.returncode}): '
                           + '\n'.join(proc.stderr.strip().splitlines()[-5:]))
        info = {'impl': impl, 'round': round_idx, 'attempt': attempt, 'cpu': cpu,
                'siblings': sibs, 'pre_util_pct': pre,
                'during_util_pct': {str(c): round(during.get(c, 0.0), 2) for c in [cpu] + sibs},
                'core_share': round(share, 4), 'wall_s': round(wall, 2),
                'loadavg_before': la0, 'loadavg_after': loadavg(),
                'valid': not reasons, 'reasons': reasons}
        attempts.append(info)
        if not reasons:
            rec['g9'] = dict(info, discarded=attempts[:-1])
            return rec
        log(f'# discard {impl} order={order} f{precision} round {round_idx} '
            f'attempt {attempt}: {reasons}')
    return {'status': 'failed', 'g9': dict(attempts[-1], discarded=attempts[:-1]),
            'config': {'device': 'cpu', 'order': order, 'precision': precision,
                       'brainevent_backend': None, 'cpu_affinity': str(cpu)}}


def wait_idle(cpu, sibs, args, log=print):
    """Re-sample until the core and its siblings are idle; return their utilisation."""
    for _ in range(args.max_waits):
        util = sample(args.pre_sample_s)
        pre = {str(c): round(util.get(c, 100.0), 2) for c in [cpu] + sibs}
        if all(v < args.idle_pct for v in pre.values()):
            return pre
        log(f'# core {cpu} busy before run: {pre}; waiting')
        time.sleep(args.wait_s)
    raise RuntimeError(f'core {cpu} stayed busy')


def run(args, log=print):
    util = sample(args.sample_s)
    picked = pick_core(util, idle_pct=args.idle_pct, prefer=args.prefer)
    if picked is None:
        raise SystemExit(f'no idle core (util {util})')
    cpu, sibs = picked
    log(f'# core {cpu} siblings {sibs} util '
        + json.dumps({str(c): round(util[c], 2) for c in [cpu] + sibs})
        + f' loadavg {loadavg()}')
    configs = [(o, p) for o in args.orders for p in args.precisions]
    for r in range(args.rounds):
        for order, prec in configs:
            for impl in ('old', 'new'):
                rec = measure(impl, order, prec, cpu, sibs, args, r, log)
                log(RUN_PREFIX + json.dumps(rec), flush=True)
                m = rec.get('steady', {}).get('ms_per_step', {}).get('median')
                log(f'# {impl} order={order} f{prec} round {r}: ms/step {m} '
                    f'peak {rec.get("memory", {}).get("peak_bytes")}', flush=True)


# ----------------------------------------------------------------------------- analysis

def parse_runs(text):
    return [json.loads(line[len(RUN_PREFIX):]) for line in text.splitlines()
            if line.startswith(RUN_PREFIX)]


def _key(rec):
    c = rec['config']
    return int(c['order']), int(c['precision'])


def _impl(rec):
    return rec['g9']['impl']


def compare(runs, t0_records=(), tol=1.1, gated_orders=GATED_ORDERS):
    """Rows per ``(order, precision)``: medians over valid rounds, ratios and status."""
    ok = [r for r in runs if r.get('status') == 'ok' and r.get('g9', {}).get('valid')]
    t0 = {(int(r['config']['order']), int(r['config']['precision'])): r for r in t0_records
          if r['config']['device'] == 'cpu' and r['config'].get('cpu_affinity', 'all') != 'all'
          and r.get('status') == 'ok'}
    rows = []
    for key in sorted({_key(r) for r in runs}, key=lambda k: (-k[0], k[1])):
        row = {'order': key[0], 'precision': key[1], 'gated': key[0] in gated_orders}
        for impl in ('old', 'new'):
            sel = [r for r in ok if _key(r) == key and _impl(r) == impl]
            row[impl] = {
                'n': len(sel),
                'ms_per_step': [float(r['steady']['ms_per_step']['median']) for r in sel],
                'peak_bytes': [float(r['memory']['peak_bytes']) for r in sel],
                'compile_s': [float(r['compile'].get('compile_s', float('nan'))) for r in sel],
            }
            for f in ('ms_per_step', 'peak_bytes', 'compile_s'):
                v = row[impl][f]
                row[impl][f + '_median'] = statistics.median(v) if v else None
        o, n = row['old'], row['new']
        if o['n'] and n['n']:
            row['step_ratio'] = n['ms_per_step_median'] / o['ms_per_step_median']
            row['mem_ratio'] = n['peak_bytes_median'] / o['peak_bytes_median']
            if row['gated']:
                passed = row['step_ratio'] <= tol and row['mem_ratio'] <= tol
                row['status'] = 'pass' if passed else 'fail'
            else:
                row['status'] = 'reported'
        else:
            row['step_ratio'] = row['mem_ratio'] = None
            row['status'] = 'not measured'
        ref = t0.get(key)
        row['t0'] = None if ref is None else {
            'ms_per_step': float(ref['steady']['ms_per_step']['median']),
            'peak_bytes': float(ref['memory']['peak_bytes']),
            'compile_s': float(ref['compile'].get('compile_s', float('nan'))),
            'cpu_affinity': ref['config']['cpu_affinity']}
        rows.append(row)
    return rows


def gate_passed(rows):
    gated = [r for r in rows if r['gated']]
    return bool(gated) and all(r['status'] == 'pass' for r in gated)


def markdown(rows):
    """Interleaved table and the reference table against the T0 pinned rows."""
    gib = 2.0 ** 30
    f = bu._fmt

    def med(r, impl, field, scale=1.0, nd=4):
        v = r[impl][field + '_median']
        return f(None if v is None else v / scale, nd)

    out = ['| order | prec | gated | rounds old/new | old ms/step | new ms/step | ratio '
           '| old peak GiB | new peak GiB | ratio | old compile s | new compile s | result |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        out.append('| ' + ' | '.join([
            str(r['order']), f"f{r['precision']}", 'yes' if r['gated'] else 'no',
            f"{r['old']['n']}/{r['new']['n']}",
            med(r, 'old', 'ms_per_step'), med(r, 'new', 'ms_per_step'), f(r['step_ratio'], 3),
            med(r, 'old', 'peak_bytes', gib, 3), med(r, 'new', 'peak_bytes', gib, 3),
            f(r['mem_ratio'], 3),
            med(r, 'old', 'compile_s', 1.0, 2), med(r, 'new', 'compile_s', 1.0, 2),
            r['status']]) + ' |')
    out += ['', '| order | prec | T0 pinned ms/step | new ms/step | ratio | T0 peak GiB '
            '| new peak GiB | ratio | T0 compile s | new compile s | old (re-measured) vs T0 step |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        t = r['t0']
        if t is None:
            continue
        nm, nb = r['new']['ms_per_step_median'], r['new']['peak_bytes_median']
        om = r['old']['ms_per_step_median']
        out.append('| ' + ' | '.join([
            str(r['order']), f"f{r['precision']}", f(t['ms_per_step'], 4), f(nm, 4),
            f(None if nm is None else nm / t['ms_per_step'], 3), f(t['peak_bytes'] / gib, 3),
            f(None if nb is None else nb / gib, 3),
            f(None if nb is None else nb / t['peak_bytes'], 3), f(t['compile_s'], 2),
            f(r['new']['compile_s_median'], 2),
            f(None if om is None else om / t['ms_per_step'], 3)]) + ' |')
    return '\n'.join(out)


def compare_main(args):
    runs = []
    for p in args.logs:
        with open(p) as fh:
            runs += parse_runs(fh.read())
    t0 = []
    if args.t0 and os.path.exists(args.t0):
        with open(args.t0) as fh:
            t0 = json.load(fh)['records']
    rows = compare(runs, t0, args.tol)
    print(markdown(rows))
    ok = gate_passed(rows)
    print(f'\nG9 (interleaved, CPU): {"PASS" if ok else "FAIL / incomplete"}')
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'benchmark': 'k0_g9_interleaved', 'schema_version': bu.SCHEMA_VERSION,
                       'protocol': __doc__.split('Usage')[0].strip(), 'tol': args.tol,
                       'rows': rows, 'runs': runs}, fh, indent=1)
            fh.write('\n')
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--commit', required=True)
    r.add_argument('--rounds', type=int, default=3)
    r.add_argument('--orders', type=int, nargs='+', default=[2500, 200])
    r.add_argument('--precisions', type=int, nargs='+', default=[32, 64])
    r.add_argument('--repeats', type=int, default=7)
    r.add_argument('--window-ms', type=float, default=None)
    r.add_argument('--prefer', type=int, default=4)
    r.add_argument('--idle-pct', type=float, default=5.0)
    r.add_argument('--min-share', type=float, default=0.85)
    r.add_argument('--sample-s', type=float, default=5.0)
    r.add_argument('--pre-sample-s', type=float, default=2.0)
    r.add_argument('--wait-s', type=float, default=20.0)
    r.add_argument('--max-waits', type=int, default=90)
    r.add_argument('--max-attempts', type=int, default=3)
    r.add_argument('--timeout', type=float, default=3600.0)
    c = sub.add_parser('compare')
    c.add_argument('logs', nargs='+')
    c.add_argument('--t0', default=T0_RESULTS)
    c.add_argument('--tol', type=float, default=1.1)
    c.add_argument('--out', default=None)
    args = p.parse_args(argv)
    if args.cmd == 'run':
        run(args)
        return 0
    return compare_main(args)


if __name__ == '__main__':
    sys.exit(main())
