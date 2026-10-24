# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K1a-G5 step 1: NEST-only scan for an asynchronous-irregular (AI) Brunel point.

Brunel ``iaf_psc_alpha`` (NEST ``brunel_alpha_nest.py`` network,
:func:`brainspike._kernel.validation.brunel.brunel_params`) over a grid of
``g`` (relative inhibition) and ``eta`` (external drive), NEST 3.10 only. Every point
runs ``--seeds`` seeds of ``T_WARM + T_STAT`` (100 ms discarded + 1000 ms, as G2) and
reports the G2 statistics (rate E/I, CV-ISI, Fano factor, pairwise correlation of the
first 50 E neurons, 10 ms bins). A point qualifies for K1a-G5 if its mean CV-ISI is
``>= 0.5`` (spec §K1a-7). The scan only *proposes* points: the lead picks the point
before the kernel comparison runs.

Usage (remote host, NEST installed)::

    python benchmarks/k1a_g5_ai_scan.py --order 200 --g 4 5 6 7 8 --eta 0.9 1.5 2 3 4 \\
        --seeds 1 2 --jobs 8 --out benchmarks/results/2026-10-22-K1a-G5-ai-scan.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np

os.environ.setdefault('BRAINSPIKE_ALLOW_DIRECT_IMPORT', '1')

T_WARM = 100.0
T_STAT = 1000.0
STATS = ('rate_E', 'rate_I', 'cv_isi_E', 'fano_E', 'corr_E')


def _one(args):
    order, g, eta, seed = args
    from brainspike._kernel.validation.brunel import brunel_params
    from brainspike._kernel.validation.g2_brunel_nest_test import nest_brunel
    from brainspike._kernel.validation.stats import network_stats
    p = brunel_params(order, g=g, eta=eta)
    t0 = time.perf_counter()
    t, s = nest_brunel(order, seed, T_WARM + T_STAT, g=g, eta=eta)
    st = network_stats(t, s, p['N_E'], p['N_I'], T_WARM, T_WARM + T_STAT)
    st.update(order=order, g=g, eta=eta, seed=seed, wall_s=time.perf_counter() - t0)
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--order', type=int, default=200)
    ap.add_argument('--g', type=float, nargs='+', default=[4.0, 5.0, 6.0, 7.0, 8.0])
    ap.add_argument('--eta', type=float, nargs='+', default=[0.9, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    tasks = list(itertools.product([a.order], a.g, a.eta, a.seeds))
    with cf.ProcessPoolExecutor(a.jobs, mp_context=mp.get_context('spawn')) as ex:
        runs = list(ex.map(_one, tasks))
    rows = []
    for g, eta in itertools.product(a.g, a.eta):
        sel = [r for r in runs if r['g'] == g and r['eta'] == eta]
        row = dict(order=a.order, g=g, eta=eta, n_seeds=len(sel))
        for k in STATS:
            v = np.array([r[k] for r in sel], float)
            row[k] = float(np.nanmean(v))
            row[k + '_sd'] = float(np.nanstd(v))
        row['ai'] = bool(row['cv_isi_E'] >= 0.5)
        rows.append(row)
    print('| order | g | eta | rate_E Hz | rate_I Hz | CV-ISI | Fano | corr | CV >= 0.5 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['order']} | {r['g']:g} | {r['eta']:g} | {r['rate_E']:.2f} | {r['rate_I']:.2f} | "
              f"{r['cv_isi_E']:.3f} | {r['fano_E']:.3f} | {r['corr_E']:.4f} | "
              f"{'yes' if r['ai'] else 'no'} |")
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(dict(t_warm=T_WARM, t_stat=T_STAT, rows=rows, runs=runs), f, indent=1)


if __name__ == '__main__':
    main()
