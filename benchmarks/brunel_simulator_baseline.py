# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Baseline benchmark of today's NEST-style ``Simulator`` on the Brunel network.

K0 spec task T0 (``docs/specs/2026-10-21-k0-kernel-spec.md`` §5); the numbers
are the reference for gate G9. See ``benchmarks/README.md`` for the protocol.

Three sub-commands::

    # one configuration, in this process; prints one ``RESULT_JSON:`` line
    python benchmarks/brunel_simulator_baseline.py run --order 200 --precision 32

    # every (order, precision) pair, each in a fresh subprocess
    python benchmarks/brunel_simulator_baseline.py sweep --orders 200 2500 \
        --precisions 32 64 --commit "$(git rev-parse HEAD)"

    # gather RESULT_JSON lines from captured logs into a results file + table
    python benchmarks/brunel_simulator_baseline.py collect cpu.log gpu.log \
        --out benchmarks/results/2026-10-21-T0-simulator-baseline.json

The device is whatever JAX picks in the run environment: ``JAX_PLATFORMS=cpu``
for CPU, ``CUDA_VISIBLE_DEVICES=<gpu>`` for GPU.
"""
import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_utils as bu  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_REC = 50
POISSON_STOP_STEPS = 2 ** 20  # 104857.6 ms at dt = 0.1 ms


#: Simulated milliseconds per timed window (see README, "Durations").
DEFAULT_WINDOW_MS = 1000.0


# --------------------------------------------------------------------------- #
# Network construction -- reused from examples/nest_like/brunel_alpha.py       #
# (Brunel 2000 model A, alpha PSCs, NEST brunel_alpha_nest.py parameters).     #
# --------------------------------------------------------------------------- #

def _psp_norm(tau_mem, c_mem, tau_syn):
    import numpy as np
    import scipy.special as sp
    a = tau_mem / tau_syn
    b = 1.0 / tau_syn - 1.0 / tau_mem
    x = -np.exp(-1.0 / a) / a
    lw = sp.lambertw(x, k=-1 if x < 0 else 0).real
    t_max = 1.0 / b * (-lw - 1.0 / a)
    return (np.exp(1.0) / (tau_syn * c_mem * b)
            * ((np.exp(-t_max / tau_mem) - np.exp(-t_max / tau_syn)) / b
               - t_max * np.exp(-t_max / tau_syn)))


def build_brunel(order, comm='sparse'):
    """Build the Brunel alpha network exactly as ``examples/nest_like/brunel_alpha.py``.

    Parameters
    ----------
    order : int
        ``N_E = 4 * order``, ``N_I = order``.
    comm : str
        Recurrent projection storage passed to ``Simulator.connect``.

    Returns
    -------
    tuple
        ``(sim, esr, isr, params)`` -- the simulator, the E / I spike recorders
        (first ``N_REC`` neurons each) and a dict of derived parameters.
    """
    import numpy as np
    import brainunit as u
    import braintools
    from brainpy_state import (Simulator, fixed_indegree, all_to_all,
                               iaf_psc_alpha, poisson_generator, spike_recorder)

    dt, delay = 0.1, 1.5
    g, eta, epsilon = 5.0, 2.0, 0.1
    ne_n, ni_n = 4 * order, 1 * order
    ce, ci = int(epsilon * ne_n), int(epsilon * ni_n)
    tau_syn, tau_mem, c_mem, theta, tref = 0.5, 20.0, 250.0, 20.0, 2.0
    j_ex = 0.1 / _psp_norm(tau_mem, c_mem, tau_syn)
    j_in = -g * j_ex
    nu_th = (theta * c_mem) / (j_ex * ce * np.exp(1) * tau_mem * tau_syn)
    p_rate = 1000.0 * (eta * nu_th) * ce

    npar = dict(C_m=c_mem * u.pF, tau_m=tau_mem * u.ms, tau_syn_ex=tau_syn * u.ms,
                tau_syn_in=tau_syn * u.ms, t_ref=tref * u.ms, E_L=0. * u.mV,
                V_reset=0. * u.mV, V_th=theta * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV))

    sim = Simulator(dt=dt * u.ms)
    ne = sim.create(iaf_psc_alpha, ne_n, params=npar)
    ni = sim.create(iaf_psc_alpha, ni_n, params=npar)
    # ``stop`` is finite only to dodge an existing float32 bug: the default
    # ``stop=inf`` stores ``t_max_step = int64 max``, which overflows the int32
    # comparison when x64 is disabled. 2**20 steps (~105 s) is beyond any measured
    # rollout, so the dynamics are identical to ``stop=inf``; a power-of-two step
    # count keeps ``stop / dt`` an exact integer in float32 too (the generator
    # rejects off-grid times).
    noise = sim.create(poisson_generator, rate=p_rate * u.Hz,
                       stop=POISSON_STOP_STEPS * dt * u.ms)
    esr = sim.create(spike_recorder)
    isr = sim.create(spike_recorder)
    sim.connect(noise, ne, weight=j_ex * u.pA, delay=delay * u.ms, rule=all_to_all)
    sim.connect(noise, ni, weight=j_ex * u.pA, delay=delay * u.ms, rule=all_to_all)
    sim.connect(ne, ne + ni, weight=j_ex * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(ce), comm=comm, allow_multapses=True, seed=1)
    sim.connect(ni, ne + ni, weight=j_in * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(ci), comm=comm, allow_multapses=True, seed=2)
    sim.connect(ne[:N_REC], esr)
    sim.connect(ni[:N_REC], isr)
    params = dict(dt_ms=dt, delay_ms=delay, g=g, eta=eta, epsilon=epsilon,
                  N_E=ne_n, N_I=ni_n, C_E=ce, C_I=ci, J_ex_pA=float(j_ex),
                  J_in_pA=float(j_in), poisson_rate_hz=float(p_rate), N_rec=N_REC,
                  comm=comm, poisson_stop_ms=POISSON_STOP_STEPS * dt,
                  n_synapses=int((ne_n + ni_n) * (ce + ci)))
    return sim, esr, isr, params


# --------------------------------------------------------------------------- #
# Single configuration                                                         #
# --------------------------------------------------------------------------- #

def _versions():
    from importlib import metadata
    out = {'python': platform.python_version()}
    for pkg in ('jax', 'jaxlib', 'brainstate', 'brainevent', 'brainunit', 'braintools',
                'numpy'):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    for pkg in ('jax-cuda12-plugin', 'jax-cuda13-plugin', 'jax-cuda12-pjrt',
                'jax-cuda13-pjrt'):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _gpu_info():
    try:
        q = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
        return q.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def base_meta(args):
    """Metadata known without importing JAX."""
    return {
        'commit': args.commit,
        'host': socket.gethostname(),
        'cpu_model': bu.cpu_model(),
        'n_cpus': os.cpu_count(),
        'loadavg_1min': os.getloadavg()[0],
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec='seconds'),
        'env': {k: os.environ.get(k) for k in (
            'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'XLA_PYTHON_CLIENT_PREALLOCATE',
            'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_FLAGS',
            'JAX_COMPILATION_CACHE_DIR')},
    }


def run_one(args):
    """Measure one (order, precision) configuration in this process."""
    meta = base_meta(args)
    os.environ.setdefault('BRAINPY_STATE_ALLOW_DIRECT_IMPORT', '1')
    sys.path.insert(0, REPO_ROOT)  # measure the local source tree

    import jax
    jax.config.update('jax_enable_x64', args.precision == 64)
    import brainstate
    brainstate.environ.set(precision=args.precision)
    import numpy as np
    import brainunit as u
    import brainevent
    import brainpy_state

    if args.brainevent_backend:
        # e.g. 'cusparse': opt-in work-around for the default GPU kernel's
        # float64 crash (see README); default = brainevent's own choice
        brainevent.config.set_backend('gpu', args.brainevent_backend)

    dev = jax.devices()[0]
    device = 'gpu' if dev.platform in ('gpu', 'cuda', 'rocm') else dev.platform
    is_gpu = device == 'gpu'
    meta.update(versions=_versions(), brainpy_state_file=brainpy_state.__file__,
                device_kind=dev.device_kind, device_platform=dev.platform,
                n_devices=jax.device_count(),
                gpu_info=_gpu_info() if is_gpu else None)

    def mem_in_use():
        if is_gpu:
            return int(dev.memory_stats()['bytes_in_use'])
        return bu.current_rss_bytes()

    window_ms = args.window_ms or DEFAULT_WINDOW_MS
    n_steps = int(round(window_ms / 0.1))
    config = {'device': device, 'order': args.order, 'precision': args.precision,
              'window_ms': window_ms, 'steps_per_window': n_steps,
              'repeats': args.repeats, 'brainevent_backend': args.brainevent_backend}

    mem_before_build = mem_in_use()
    t0 = time.perf_counter()
    sim, esr, isr, params = build_brunel(args.order)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    sim.reset_rollout()
    jax.block_until_ready([s.value for s in sim.states().values()])
    init_s = time.perf_counter() - t0
    mem_after_build = mem_in_use()

    # --- compile accounting via jax.monitoring --------------------------------
    compile_events = []

    def _listener(event, duration, **kw):
        if event.startswith('/jax/core/compile/'):
            compile_events.append((event, float(duration)))

    jax.monitoring.register_event_duration_secs_listener(_listener)

    e_key = id(esr.segments[0].population)
    i_key = id(isr.segments[0].population)

    # The Simulator's own rollout (`cont` -> `for_loop` over the window), wrapped
    # in brainstate.transform.jit so the compiled program is reused across calls.
    # (Plain `sim.cont` re-traces and re-compiles its scan on every call.)
    @brainstate.transform.jit
    def run_window():
        res = sim.cont(window_ms * u.ms)
        return res._rec[e_key], res._rec[i_key]

    def timed_call():
        n0 = len(compile_events)
        t = time.perf_counter()
        e_spk, i_spk = jax.block_until_ready(run_window())
        wall = time.perf_counter() - t
        comp = sum(d for _, d in compile_events[n0:])
        e_rate = float(np.asarray(e_spk).sum()) / N_REC / (window_ms / 1000.0)
        i_rate = float(np.asarray(i_spk).sum()) / N_REC / (window_ms / 1000.0)
        return wall, comp, len(compile_events) - n0, e_rate, i_rate

    # warm-up: call until one call triggers no compilation (the first call
    # compiles; a second compile may follow when initial weak-typed state
    # values are replaced by concrete ones)
    warm = []
    for _ in range(args.max_warmup):
        warm.append(timed_call())
        if warm[-1][2] == 0:
            break
    else:
        raise RuntimeError(f'still compiling after {args.max_warmup} warm-up calls')
    # the last warm-up call is compile-free; it is a steady sample of its own
    # but we keep it out of the timed set for a clean "post-warmup" protocol.
    timed = [timed_call() for _ in range(args.repeats)]
    recompiles = sum(t[2] for t in timed)
    walls = [t[0] for t in timed]
    steady_ref = sorted(walls)[len(walls) // 2]
    compiling_calls = [w for w in warm if w[2] > 0]
    compile_s = sum(w[1] for w in compiling_calls)
    compile_wall_overhead_s = sum(w[0] for w in compiling_calls) - steady_ref * len(
        compiling_calls)

    public = None
    if args.public_call:
        n0 = len(compile_events)
        t = time.perf_counter()
        res = sim.cont(window_ms * u.ms)
        jax.block_until_ready(res._rec[e_key])
        public = {'wall_s': time.perf_counter() - t,
                  'compile_events_s': sum(d for _, d in compile_events[n0:]),
                  'n_compile_events': len(compile_events) - n0}

    if is_gpu:
        stats = dev.memory_stats()
        peak = int(stats['peak_bytes_in_use'])
        mem = {'kind': 'gpu_peak_bytes_in_use', 'peak_bytes': peak,
               'bytes_limit': int(stats.get('bytes_limit', 0)) or None,
               'before_build_bytes': mem_before_build, 'after_build_bytes': mem_after_build,
               'host_peak_rss_bytes': bu.peak_rss_bytes()}
    else:
        mem = {'kind': 'cpu_peak_rss', 'peak_bytes': bu.peak_rss_bytes(),
               'before_build_bytes': mem_before_build, 'after_build_bytes': mem_after_build}

    sec_per_sim_s = [w / (window_ms / 1000.0) for w in walls]
    record = {
        'schema_version': bu.SCHEMA_VERSION,
        'benchmark': 'brunel_simulator_baseline',
        'status': 'ok',
        'config': config,
        'meta': meta,
        'network': params,
        'build': {'build_s': build_s, 'init_states_s': init_s},
        'compile': {
            'first_call_s': warm[0][0],
            'compile_s': compile_s,
            'compile_wall_overhead_s': compile_wall_overhead_s,
            'n_compiling_calls': len(compiling_calls),
            'n_compile_events': sum(w[2] for w in warm),
            'warmup_walls_s': [w[0] for w in warm],
            'recompiles_during_timing': recompiles,
            'public_cont_call': public,
        },
        'steady': {
            'window_wall_s': bu.summarize(walls),
            's_per_sim_second': bu.summarize(sec_per_sim_s),
            'ms_per_step': bu.summarize([w / n_steps * 1e3 for w in walls]),
        },
        'memory': mem,
        'sanity': {'e_rate_hz': bu.summarize([t[3] for t in timed]),
                   'i_rate_hz': bu.summarize([t[4] for t in timed])},
    }
    bu.validate_record(record)
    print(bu.RESULT_PREFIX + json.dumps(record), flush=True)
    return record


# --------------------------------------------------------------------------- #
# Sweep / collect                                                              #
# --------------------------------------------------------------------------- #

def _device_label_from_env():
    return 'cpu' if os.environ.get('JAX_PLATFORMS', '') == 'cpu' else 'gpu'


def sweep(args):
    """Run every configuration in its own subprocess and print/merge the records."""
    records = []
    for order in args.orders:
        for prec in args.precisions:
            cmd = [sys.executable, os.path.abspath(__file__), 'run', '--order', str(order),
                   '--precision', str(prec), '--repeats', str(args.repeats),
                   '--max-warmup', str(args.max_warmup), '--commit', str(args.commit)]
            if args.window_ms:
                cmd += ['--window-ms', str(args.window_ms)]
            if args.brainevent_backend:
                cmd += ['--brainevent-backend', args.brainevent_backend]
            if not args.public_call:
                cmd += ['--no-public-call']
            print(f'# sweep: order={order} precision={prec}', flush=True)
            t = time.perf_counter()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=args.timeout)
                out, err, rc = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as e:
                out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
                err = f'timeout after {args.timeout} s'
                rc = None
            found = bu.parse_result_lines(out)
            if rc == 0 and found:
                rec = found[-1]
            else:
                tail = '\n'.join((err or '').strip().splitlines()[-25:])
                label = 'timeout' if rc is None else bu.classify_failure(tail, rc)
                cfg = {'device': _device_label_from_env(), 'order': order, 'precision': prec,
                       'brainevent_backend': args.brainevent_backend}
                rec = bu.failure_record(cfg, base_meta(args), label, returncode=rc,
                                        stderr_tail=tail)
            print(bu.RESULT_PREFIX + json.dumps(rec), flush=True)
            print(f'# done in {time.perf_counter() - t:.1f} s: {rec["status"]}', flush=True)
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
        json.dump({'benchmark': 'brunel_simulator_baseline',
                   'schema_version': bu.SCHEMA_VERSION,
                   'records': merged}, f, indent=1)
        f.write('\n')
    return merged


def collect(args):
    """Parse RESULT_JSON lines from log files and merge them into ``--out``."""
    records = []
    for p in args.logs:
        with open(p) as f:
            records = bu.merge_records(records, bu.parse_result_lines(f.read()))
    merged = write_results(args.out, records)
    print(bu.markdown_table(merged))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--repeats', type=int, default=5,
                        help='timed windows after warm-up (median reported)')
        sp.add_argument('--window-ms', type=float, default=None,
                        help=f'simulated ms per timed window (default {DEFAULT_WINDOW_MS:g})')
        sp.add_argument('--max-warmup', type=int, default=4)
        sp.add_argument('--commit', default=os.environ.get('BENCH_COMMIT'),
                        help='commit sha being measured (remote copies have no .git)')
        sp.add_argument('--brainevent-backend', default=None,
                        help="global brainevent GPU backend override (e.g. 'cusparse')")
        sp.add_argument('--no-public-call', dest='public_call', action='store_false',
                        help='skip the extra un-jitted sim.cont() call')

    r = sub.add_parser('run', help='measure one configuration in this process')
    r.add_argument('--order', type=int, required=True)
    r.add_argument('--precision', type=int, choices=(32, 64), required=True)
    common(r)

    s = sub.add_parser('sweep', help='each configuration in a fresh subprocess')
    s.add_argument('--orders', type=int, nargs='+', default=[200, 2500])
    s.add_argument('--precisions', type=int, nargs='+', default=[32, 64])
    s.add_argument('--timeout', type=float, default=4 * 3600)
    s.add_argument('--out', default=None, help='results JSON to merge into')
    common(s)

    c = sub.add_parser('collect', help='merge RESULT_JSON lines from logs')
    c.add_argument('logs', nargs='+')
    c.add_argument('--out', required=True)

    args = p.parse_args(argv)
    {'run': run_one, 'sweep': sweep, 'collect': collect}[args.cmd](args)


if __name__ == '__main__':
    main()
