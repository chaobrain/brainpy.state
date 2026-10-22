# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K1a-G4: Potjans--Diesmann microcircuit at full scale, kernel vs NEST 3.10.

Network: :mod:`brainpy_state._kernel.validation.microcircuit` (77 169 ``iaf_psc_exp``
neurons, ~3e8 synapses, Poisson background). Protocol (lead decision 2026-10-22):
5 seeds each side, rates over 1000 ms after a discarded 100 ms transient; kernel in
float32 on the CPU and on GPU 1. Every run is a fresh process (one RESULT_JSON line).

Kernel numbers: ``build_s`` (network + ``compile()`` to a Plan + runner state),
``compile_s`` (first ``cont(100 ms)`` minus the median warm ``cont(100 ms)``),
``ms_per_step`` (median warm chunk / 1000 steps), peak memory (CPU: process peak
RSS; GPU: device ``peak_bytes_in_use`` and the process peak RSS).

Usage::

    python benchmarks/k1a_microcircuit.py kernel --seed 1            # one run, this process
    python benchmarks/k1a_microcircuit.py nest --seed 1 --threads 16 # one NEST run
    python benchmarks/k1a_microcircuit.py compare run.log ... [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_WARM = 100.0
T_STAT = 1000.0


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # KiB on Linux


def _emit(rec: dict) -> None:
    print('RESULT_JSON ' + json.dumps(rec), flush=True)


def run_kernel(a) -> None:
    os.environ.setdefault('BRAINPY_STATE_ALLOW_DIRECT_IMPORT', '1')
    sys.path.insert(0, REPO_ROOT)
    import jax
    from brainpy_state._kernel.validation.microcircuit import kernel_microcircuit

    dev = jax.devices()[0]
    is_gpu = dev.platform in ('gpu', 'cuda')
    t0 = time.perf_counter()
    r = kernel_microcircuit(a.seed, T_WARM, T_STAT, scale=a.scale, precision=f'float{a.precision}',
                            bg_input=a.bg_input)
    r.update(sim='kernel', seed=a.seed, scale=a.scale, precision=a.precision, bg_input=a.bg_input,
             device='gpu' if is_gpu else 'cpu', device_kind=dev.device_kind,
             wall_s=time.perf_counter() - t0, peak_rss_gib=_peak_rss_gib(),
             load=os.getloadavg()[0], cpu_affinity=sorted(os.sched_getaffinity(0)))
    if is_gpu:
        st = dev.memory_stats()
        r.update(peak_device_gib=st['peak_bytes_in_use'] / 2 ** 30,
                 device_in_use_gib=st['bytes_in_use'] / 2 ** 30)
    _emit(r)
    sys.stdout.flush()
    os._exit(0)  # see benchmarks/k1a_compile_time.py: interpreter exit may hang after a run


def run_nest(a) -> None:
    os.environ.setdefault('BRAINPY_STATE_ALLOW_DIRECT_IMPORT', '1')
    sys.path.insert(0, REPO_ROOT)
    from brainpy_state._kernel.validation.microcircuit import nest_microcircuit
    t0 = time.perf_counter()
    r = nest_microcircuit(a.seed, T_WARM, T_STAT, scale=a.scale, threads=a.threads,
                          bg_input=a.bg_input)
    r.update(sim='nest', seed=a.seed, scale=a.scale, threads=a.threads, bg_input=a.bg_input,
             wall_s=time.perf_counter() - t0, peak_rss_gib=_peak_rss_gib(),
             load=os.getloadavg()[0])
    _emit(r)


def _records(paths) -> list:
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.startswith('RESULT_JSON '):
                    out.append(json.loads(line[len('RESULT_JSON '):]))
    return out


def compare(a) -> None:
    import numpy as np
    sys.path.insert(0, REPO_ROOT)
    os.environ.setdefault('BRAINPY_STATE_ALLOW_DIRECT_IMPORT', '1')
    from brainpy_state._kernel.validation.microcircuit import POPULATIONS
    from brainpy_state._kernel.validation.stats import g2_rule

    recs = _records(a.logs)
    nest = [r for r in recs if r['sim'] == 'nest']
    groups = {}
    for r in recs:
        if r['sim'] == 'kernel':
            groups.setdefault(r['device'], []).append(r)
    out = dict(nest=nest, kernel=groups, gate={})
    print(f'NEST: {len(nest)} seeds')
    for dev, ks in sorted(groups.items()):
        print(f'\n### kernel {dev} f32 ({len(ks)} seeds) vs NEST ({len(nest)} seeds)\n')
        print('| population | kernel Hz mean ± SEM | NEST Hz mean ± SEM | |Δ| | bound | pass |')
        print('|---|---|---|---|---|---|')
        rows = []
        for k in POPULATIONS:
            res = g2_rule([r['rates'][k] for r in ks], [r['rates'][k] for r in nest], k)
            rows.append(res.__dict__)
            print(f'| {k} | {res.ours_mean:.3f} ± {res.ours_sem:.3f} | {res.ref_mean:.3f} ± '
                  f'{res.ref_sem:.3f} | {abs(res.diff):.3f} | {res.bound:.3f} | '
                  f'{"pass" if res.passed else "FAIL"} |')
        med = lambda key: float(np.median([r[key] for r in ks if r.get(key) is not None]))
        perf = dict(build_s=med('build_s'), compile_s=med('compile_s'),
                    ms_per_step=med('ms_per_step'), peak_rss_gib=med('peak_rss_gib'),
                    n_edges=ks[0]['n_edges'])
        if dev == 'gpu':
            perf['peak_device_gib'] = med('peak_device_gib')
        print('\nperf (median over seeds):', json.dumps(perf))
        out['gate'][dev] = dict(rows=rows, passed=all(r['passed'] for r in rows), perf=perf)
    if nest:
        print('\nNEST (median): build_s %.1f, sim_s %.1f (%.3f s per simulated s), peak RSS %.1f GiB, '
              'threads %s' % (np.median([r['build_s'] for r in nest]),
                              np.median([r['sim_s'] for r in nest]),
                              np.median([r['sim_s'] for r in nest]) / ((T_WARM + T_STAT) / 1e3),
                              np.median([r['peak_rss_gib'] for r in nest]), nest[0]['threads']))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('kernel')
    k.add_argument('--seed', type=int, required=True)
    k.add_argument('--precision', type=int, choices=(32, 64), default=32)
    k.add_argument('--scale', type=float, default=1.0)
    k.add_argument('--bg-input', default='poisson', choices=('poisson', 'dc'))
    n = sub.add_parser('nest')
    n.add_argument('--seed', type=int, required=True)
    n.add_argument('--threads', type=int, default=16)
    n.add_argument('--scale', type=float, default=1.0)
    n.add_argument('--bg-input', default='poisson', choices=('poisson', 'dc'))
    c = sub.add_parser('compare')
    c.add_argument('logs', nargs='+')
    c.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    {'kernel': run_kernel, 'nest': run_nest, 'compare': compare}[a.cmd](a)


if __name__ == '__main__':
    main()
