# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K1a-1 compile-time gate and tables (K1a spec §3, K1a-1 "Gate (compile time)").

Compile time of a network = wall time of the first ``runner.simulate(T)`` (trace,
lower, XLA compile, run) minus the wall time of a warm ``runner.cont(T)`` of the same
length (run only), in a fresh process per measurement (``JAX_PLATFORMS=cpu``, single
process). For the BPTT case it is the first ``bptt.grad`` minus a warm one.

Usage::

    # one measurement in this process (prints one RESULT_JSON line)
    python benchmarks/k1a_compile_time.py one aeif --k 16

    # the gate networks at default substeps + the k / capacity tables, each in a
    # fresh subprocess with a timeout; prints markdown tables
    python benchmarks/k1a_compile_time.py all --timeout 900 --tag K1a-1

Cases: ``aeif`` / ``cond`` (``--k`` RK4 substeps; default = the model's default),
``ps`` (``--capacity``), ``brunel`` (order 2500), ``coba`` (4000 neurons), ``nmda``,
``stdp``, ``bptt_lif``. The script uses only the K0 kernel API, so the same file
measures the K0 tree (pre-K1a-1) for the before/after tables.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

GATE_S = 5.0
T_MS = 10.0
K_TABLE = (1, 2, 4, 8, 16, 32, 64)
CAP_TABLE = (8, 16, 64)
GATE_CASES = (('brunel', {}), ('coba', {}), ('aeif', {}), ('ps', {'capacity': 64}),
              ('nmda', {}), ('stdp', {}), ('bptt_lif', {}))


def _build(case: str, k: int | None, capacity: int | None):
    import brainunit as u
    import numpy as np

    from brainpy_state._kernel.connectivity import fixed_indegree, one_to_one
    from brainpy_state._kernel.integrators import RK
    from brainpy_state._kernel.network import Network

    dt = 0.1 * u.ms
    if case in ('aeif', 'cond'):
        from brainpy_state._kernel.models import aeif_cond_alpha, iaf_cond_exp, spike_generator
        base = aeif_cond_alpha if case == 'aeif' else iaf_cond_exp
        model = base
        if k is not None:
            model = type(base.__name__, (base,), {'integrator': RK(4, k),
                                                  '__module__': base.__module__})
        net = Network(dt=dt, seed=0)
        p = net.add(model, 4, params=dict(I_e=np.array([0.0, 500.0, 800.0, 1500.0]) * u.pA))
        sg = net.add(spike_generator, 1, params=dict(spike_times=np.arange(2.0, 100.0, 3.7) * u.ms))
        net.connect(sg, p, port='ex', weight=6.0 * u.nS, delay=0.1 * u.ms)
        net.connect(sg, p, port='in_', weight=4.0 * u.nS, delay=0.2 * u.ms)
        net.observe(p, 'V_m', 'spike')
        return net, dict(substeps=model.integrator.substeps)
    if case == 'ps':
        from brainpy_state._kernel.models.precise import iaf_psc_exp_ps, poisson_generator_ps
        K = capacity or 16
        n = 100
        net = Network(dt=dt, seed=11)
        nrn = net.add(iaf_psc_exp_ps.with_capacity(K), n, params=dict(I_e=200.0 * u.pA))
        gE = net.add(poisson_generator_ps, n, params=dict(rate=6000.0 * u.Hz))
        gI = net.add(poisson_generator_ps, n, params=dict(rate=1500.0 * u.Hz))
        kw = dict(delay=1.0 * u.ms, strict_capacity=False)
        net.connect(gE, nrn, one_to_one(), port='syn', weight=45.0 * u.pA, **kw)
        net.connect(gI, nrn, one_to_one(), port='syn', weight=-90.0 * u.pA, **kw)
        net.observe(nrn, 'V_m', 'spike')
        return net, dict(slots=K)
    if case == 'brunel':
        from brainpy_state._kernel.validation.brunel import build_brunel
        return build_brunel(2500, seed=1)[0], dict(order=2500)
    if case == 'coba':
        from brainpy_state._kernel.validation.coba import build_coba
        return build_coba(0, 1.0)[0], dict(neurons=4000)
    if case == 'nmda':
        from brainpy_state._kernel.gates import MgBlock
        from brainpy_state._kernel.model import with_conductance_ports
        from brainpy_state._kernel.models import NMDA, iaf_cond_exp, spike_generator
        from brainpy_state._kernel.ports import Conductance
        cell = with_conductance_ports(iaf_cond_exp, nmda=Conductance(u.nS, E_rev='E_nmda', gate=MgBlock()),
                                      params=dict(E_nmda=0.0 * u.mV))
        net = Network(dt=dt, seed=3)
        sg = net.add(spike_generator, 2, params=dict(spike_times=np.array([1.0, 1.2, 1.4, 20.0]) * u.ms))
        rec = net.add(NMDA, 2)
        post = net.add(cell, 2, params=dict(V_th=-50.0 * u.mV, I_e=150.0 * u.pA))
        net.connect(sg, rec, one_to_one(), port='spike')
        net.connect(rec, post, port='nmda', weight=30.0 * u.nS, delay=0.0 * u.ms)
        net.observe(post, 'V_m')
        return net, {}
    if case == 'stdp':
        from brainpy_state._kernel.models import LIF, spike_generator
        from brainpy_state._kernel.plasticity import stdp_synapse
        K = 8
        net = Network(dt=dt)
        pre = net.add(spike_generator, K, params=dict(spike_times=np.arange(1.0, 9.0) * u.ms))
        drv = net.add(spike_generator, K, params=dict(spike_times=np.arange(1.5, 9.5) * u.ms))
        post = net.add(LIF, K, params=dict(V_th=1e4 * u.mV), spk_reset='hard')
        net.connect(drv, post, one_to_one(), port='delta', weight=1e6 * u.mV, delay=dt)
        rule = stdp_synapse(tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms, lambda_=0.01, alpha=1.0,
                            mu_plus=1.0, mu_minus=1.0, Wmax=100.0)
        net.connect(pre, post, one_to_one(), port='delta', weight=1.0 * u.mV, delay=1.0 * u.ms,
                    plasticity=rule)
        net.observe(post, 'spike')
        return net, {}
    raise ValueError(f'unknown case {case!r}')


def _bptt_lif():
    import braintools
    import brainunit as u
    import jax
    import numpy as np

    from brainpy_state._kernel.connectivity import fixed_indegree
    from brainpy_state._kernel.learn import BPTT, ExternalInput, spike_count_mse
    from brainpy_state._kernel.models import LIF
    from brainpy_state._kernel.network import Network

    dt = 0.1 * u.ms
    sig = braintools.surrogate.Sigmoid()
    net = Network(dt=dt)
    x = net.add(ExternalInput, 6, name='x')
    A = net.add(LIF, 10, params=dict(tau=8.0 * u.ms), name='A', surrogate=sig, trainable=('tau',))
    B = net.add(LIF, 10, params=dict(tau=8.0 * u.ms), name='B', surrogate=sig)
    net.connect(x, A, port='delta', weight=0.35 * u.mV, delay=2 * dt, trainable=True, name='xA')
    net.connect(A, A, fixed_indegree(4), port='delta', weight=0.15 * u.mV, delay=2 * dt,
                trainable=True, name='AA')
    net.connect(A, B, fixed_indegree(5), port='delta', weight=0.3 * u.mV, delay=2 * dt,
                trainable=True, name='AB')
    net.observe(B, 'spike', name='s_B')
    plan = net.compile()
    n = int(round(T_MS / 0.1))
    xs = {'x': (jax.random.uniform(jax.random.key(0), (n, 6)) < 0.3).astype('float32')}
    loss = lambda r, y: spike_count_mse(r['s_B']['spike'], y)
    if hasattr(plan, 'program'):  # K1a-5 learner API: BPTT(program, loss).grad(state0, data, targets)
        prog = plan.program()
        bptt = BPTT(prog, loss)
        state0 = prog.init_state()

        def call():
            jax.block_until_ready(bptt.grad(state0, xs, 0.1))
    else:  # K0 API: learner attached to a runner
        bptt = BPTT(loss)
        plan.runner(learner=bptt)

        def call():
            jax.block_until_ready(bptt.grad(T_MS * u.ms, xs=xs, targets=0.1))

    t = time.perf_counter()
    call()
    first = time.perf_counter() - t
    t = time.perf_counter()
    call()
    warm = time.perf_counter() - t
    return first, warm, {}


def measure(case: str, k: int | None = None, capacity: int | None = None) -> dict:
    """Compile time of one case in this process."""
    import brainunit as u
    import jax

    t = time.perf_counter()
    if case == 'bptt_lif':
        first, warm, info = _bptt_lif()
        build = 0.0
    else:
        net, info = _build(case, k, capacity)
        run = net.compile().runner()
        build = time.perf_counter() - t
        t = time.perf_counter()
        res = run.simulate(T_MS * u.ms)
        jax.block_until_ready(jax.tree.leaves(res.recordings))
        first = time.perf_counter() - t
        t = time.perf_counter()
        res = run.cont(T_MS * u.ms)
        jax.block_until_ready(jax.tree.leaves(res.recordings))
        warm = time.perf_counter() - t
    return dict(case=case, k=k, capacity=capacity, build_s=round(build, 3), first_s=round(first, 3),
                warm_s=round(warm, 4), compile_s=round(first - warm, 3), **info)


def _sub(case: str, timeout: float, **kw) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), 'one', case]
    for key, v in kw.items():
        if v is not None:
            cmd += [f'--{key}', str(v)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(case=case, **kw, compile_s=None, status=f'timeout > {timeout:.0f} s')
    for line in out.stdout.splitlines():
        if line.startswith('RESULT_JSON '):
            return dict(json.loads(line[len('RESULT_JSON '):]), status='ok')
    return dict(case=case, **kw, compile_s=None, status='error: ' + out.stderr.strip()[-300:])


def _fmt(r: dict) -> str:
    return f"{r['compile_s']:.2f}" if r.get('compile_s') is not None else r['status']


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    one = sub.add_parser('one')
    one.add_argument('case')
    one.add_argument('--k', type=int)
    one.add_argument('--capacity', type=int)
    al = sub.add_parser('all')
    al.add_argument('--timeout', type=float, default=900.0)
    al.add_argument('--tag', default='')
    al.add_argument('--skip-tables', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'one':
        print('RESULT_JSON ' + json.dumps(measure(a.case, a.k, a.capacity)), flush=True)
        sys.stderr.flush()
        # the interpreter can hang at shutdown after a kernel run (observed on the remote
        # host: threads left in futex waits); the result is out, so leave immediately
        os._exit(0)
    rows = []
    print(f'## compile time [{a.tag}] (s; first call - warm call, {T_MS:g} ms rollout)\n', flush=True)
    print('| network | compile s | first s | warm s | gate <= 5 s |\n|---|---|---|---|---|', flush=True)
    for case, kw in GATE_CASES:
        r = _sub(case, a.timeout, **kw)
        rows.append(r)
        ok = r.get('compile_s') is not None and r['compile_s'] <= GATE_S
        name = case + ''.join(f' {k}={v}' for k, v in kw.items())
        print(f"| {name} | {_fmt(r)} | {r.get('first_s', '-')} | {r.get('warm_s', '-')} | "
              f"{'pass' if ok else 'FAIL'} |", flush=True)
    if not a.skip_tables:
        print('\n| k (RK4 substeps) | ' + ' | '.join(map(str, K_TABLE)) + ' |', flush=True)
        print('|---|' + '---|' * len(K_TABLE), flush=True)
        for case in ('aeif', 'cond'):
            rs = [_sub(case, a.timeout, k=k) for k in K_TABLE]
            rows += rs
            print(f'| {case} | ' + ' | '.join(_fmt(r) for r in rs) + ' |', flush=True)
        print('\n| capacity (iaf_psc_exp_ps) | ' + ' | '.join(map(str, CAP_TABLE)) + ' |', flush=True)
        print('|---|' + '---|' * len(CAP_TABLE), flush=True)
        rs = [_sub('ps', a.timeout, capacity=c) for c in CAP_TABLE]
        rows += rs
        print('| ps | ' + ' | '.join(_fmt(r) for r in rs) + ' |', flush=True)
    print('\nRESULTS_JSON ' + json.dumps(rows), flush=True)


if __name__ == '__main__':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('BRAINPY_STATE_ALLOW_DIRECT_IMPORT', '1')
    main()
