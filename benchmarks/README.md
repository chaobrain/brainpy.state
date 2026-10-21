# Benchmarks

Performance measurements that are too slow or too host-dependent for the test
suite. Nothing here is collected by pytest (`pytest brainpy_state/` is what CI
runs, and no file here is named `test_*.py` / `*_test.py`).

| file | purpose |
|---|---|
| `brunel_simulator_baseline.py` | K0 task T0: baseline of today's NEST-style `Simulator` on the Brunel network; the reference for K0 gate G9 |
| `_bench_utils.py` | JAX-free helpers: statistics, memory parsing, result schema, merging, markdown table |
| `check_bench_utils.py` | checks for the helpers; run explicitly: `python -m pytest benchmarks/check_bench_utils.py` |
| `results/` | committed result files (JSON) and a summary table (`results/README.md`) |

## Brunel `Simulator` baseline (T0)

### Model

Brunel (2000) model A with alpha-shaped PSCs, built with the existing
`Simulator` exactly as `examples/nest_like/brunel_alpha.py` does (NEST
`brunel_alpha_nest.py` parameters): `iaf_psc_alpha` neurons,
`N_E = 4 * order`, `N_I = order`, fixed in-degree `C_E = 0.1 N_E`,
`C_I = 0.1 N_I` (multapses allowed, `comm='sparse'`, i.e. brainevent CSR),
`g = 5`, `eta = 2`, `J = 0.1 mV` PSP amplitude, delay 1.5 ms, `dt = 0.1 ms`,
one `poisson_generator` connected all-to-all, 50 E and 50 I neurons recorded.

One deviation from the example: the Poisson generator gets a finite
`stop = 2**20 * dt` (about 105 s, longer than any measured rollout, so the
dynamics are unchanged). With the default `stop = inf` the generator stores
`int64` max as its last step, which overflows when x64 is off, so float32 runs
crash otherwise.

### Usage

```bash
# one configuration in the current process (prints one RESULT_JSON line)
python benchmarks/brunel_simulator_baseline.py run --order 200 --precision 32

# all (order, precision) pairs, each in a fresh subprocess, merged into a file
python benchmarks/brunel_simulator_baseline.py sweep --orders 200 2500 \
    --precisions 32 64 --commit "$(git rev-parse HEAD)" --out results/x.json

# merge RESULT_JSON lines captured from (remote) logs and print the table
python benchmarks/brunel_simulator_baseline.py collect cpu.log gpu.log \
    --out benchmarks/results/2026-10-21-T0-simulator-baseline.json
```

The device is whatever JAX selects: set `JAX_PLATFORMS=cpu` for CPU and
`CUDA_VISIBLE_DEVICES=<id>` for a GPU. `--brainevent-backend cusparse` sets
brainevent's global GPU backend (see "Known issues"). Pass `--commit` because
remote copies of the tree have no `.git`.

The K0 runs use the remote host of spec §10.3 via the task wrapper:

```bash
BENCH=1 /tmp/k0/remote.sh T0 'python benchmarks/brunel_simulator_baseline.py sweep --commit <sha>'
BENCH=1 /tmp/k0/remote.sh T0 'taskset -c 4 python benchmarks/brunel_simulator_baseline.py sweep --commit <sha>'
BENCH=1 GPU=1 /tmp/k0/remote.sh T0 'python benchmarks/brunel_simulator_baseline.py sweep --commit <sha>'
BENCH=1 GPU=1 /tmp/k0/remote.sh T0 'python benchmarks/brunel_simulator_baseline.py sweep --brainevent-backend cusparse --commit <sha>'
```

`BENCH=1` aborts when the 1-minute load average is 2 or more; every record also
stores the load average at its start and end (the sweep's own previous
subprocess contributes about 1 to the start value).

### Measurement protocol

Each configuration (device, order, precision, brainevent backend, CPU affinity)
runs in a fresh Python subprocess, so the peak-memory counters start clean and
no compiled program is shared. Inside it:

1. **Precision.** `jax_enable_x64` and `brainstate.environ.set(precision=...)`
   are set before the package is imported.
2. **Build time** (`build.build_s`): wall time of the network construction
   (`Simulator.create` / `connect`, including fixed-indegree sampling and CSR
   construction). `build.init_states_s` is `reset_rollout()` (state
   initialisation), blocked until ready.
3. **Rollout.** The Simulator's own rollout, `sim.cont(window)` (one
   `brainstate.transform.for_loop` over the window), wrapped in
   `brainstate.transform.jit` so the compiled program is reused across calls.
   A bare `sim.cont` re-traces and re-compiles its scan on every call (see
   "Known issues"); that cost is recorded separately as
   `compile.public_cont_call`.
4. **Compile time.** `compile.compile_s` is the wall time of an explicit
   ahead-of-time `run_window.compile()`: brainstate's Python trace of the
   Simulator step, lowering and XLA compilation, with no simulation. Warm-up
   calls follow (at least two, then until a call triggers no compilation):
   the second call recompiles once, because the initial weak-typed state
   values have been replaced, and the JAX compile-event time of that
   recompile is `compile.warmup_xla_events_s` ("recompile s" in the table).
   `compile.first_call_s` and `compile.warmup_walls_s` keep the raw wall
   times of the warm-up calls.
5. **Steady-state step time.** `--repeats` (default 7) further windows of
   `--window-ms` (default 1000 ms = 10 000 steps) simulated time, each timed
   with `jax.block_until_ready` on the outputs. The records report median, min,
   max and all samples of the window wall time, seconds per simulated second and
   milliseconds per step. `compile.recompiles_during_timing` must be 0.
6. **Peak memory.** GPU: `jax.devices()[0].memory_stats()['peak_bytes_in_use']`
   (allocator peak, independent of XLA's pre-allocation; host peak RSS is stored
   too). CPU: peak RSS of the process (`getrusage(RUSAGE_SELF).ru_maxrss`),
   which includes the Python/JAX runtime. `bytes after build` and `before build`
   are stored so the network's share can be estimated.
7. **Sanity.** Mean E and I rates of the 50 recorded neurons of each population
   per timed window (median reported). The timed windows follow the warm-up,
   so the initial transient is excluded.
8. **Metadata.** Commit, host, CPU model and count, device kind, GPU name and
   driver, package versions (python, jax, jaxlib, CUDA plugin, brainstate,
   brainevent, brainunit, braintools, numpy), relevant environment variables,
   load average and UTC time stamp.

A configuration that crashes, runs out of memory or times out is stored as a
`status: "failed"` record with its return code, a short label (`OOM`,
`killed (SIGKILL)`, `abort (SIGABRT)`, `timeout`, `error`) and the stderr tail.

### CPU affinity

On the K0 host (i7-14700KF: CPUs 0-15 are the 8 hyper-threaded P-cores,
16-27 the E-cores) the unpinned CPU run is both slow and noisy: at order 2500,
float32, 1000 ms windows spread from 11 s to 18 s inside one process. Pinned to
one P-core (`taskset -c 4`) the same window takes 5.1 s with under 1 % spread.
So the CPU sweep is run twice, unpinned (`cpus = all`, what a user gets by
default) and pinned to CPU 4 (`cpus = 4`). The affinity is part of each
record's configuration (`config.cpu_affinity`). For a ≤ 1.1× gate, compare
against the pinned row; only it is stable enough.

### Durations

At order 2500 one simulated second takes 5 to 30 s on the CPU of the K0 host,
and at order 200 about 4 s on either device, so 1000 ms windows last seconds to
tens of seconds: long against timer resolution and dispatch overhead, and
short enough that the full sweep stays around an hour. Seven timed windows
plus at least two warm-up calls and one bare `sim.cont` call make about 10
simulated seconds per configuration. The min/max spread and all samples are
stored to judge stability.

### Known issues found while building the baseline

- `poisson_generator` with the default `stop = inf` fails when x64 is off
  (`int64` max overflows the step comparison); worked around above.
- `Simulator.cont` documents "no recompile" but re-traces and re-compiles its
  `for_loop` on every call.
- On GPU in float64, the default brainevent backend aborts the process
  (`binary_csrmv_wat_hybrid_hetero_f64_float expects matching vector dtype`):
  the CUDA kernel picks the `_float` (float32) event variant for a float64 spike
  vector. The default-backend float64 GPU runs are therefore recorded as
  failures, and a second GPU sweep with `--brainevent-backend cusparse`
  supplies float32 and float64 numbers.
