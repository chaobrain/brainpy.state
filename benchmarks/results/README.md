# Benchmark results

## 2026-10-21 — T0: today's `Simulator`, Brunel alpha (K0 baseline for gate G9)

File: [`2026-10-21-T0-simulator-baseline.json`](2026-10-21-T0-simulator-baseline.json)
(all samples and metadata). Protocol: [`../README.md`](../README.md).

- Commit measured: `c8b46484b79b096eecbc78dc5736f26fd6d2609e`, the pre-rebase T0
  branch (later squashed). Its `brainpy_state/` is identical to `main` at `931c544`.
  The K0 commits that followed (T1, T2, T4) only add `brainpy_state/_kernel/` and
  docs, so this is the `Simulator` on `main` when T0 merged. The benchmark code is
  the same as in the merged commit, apart from documentation.
- Host: `wanglab3090b-01a0.chaobrain`, Intel i7-14700KF (28 CPUs), RTX 3090 (GPU 1,
  driver 580.178.04). Python 3.12.15, jax/jaxlib 0.12.0 (CUDA 13 plugin),
  brainstate 0.6.0, brainevent 0.2.1, brainunit 0.6.0, braintools 0.4.0, numpy 2.5.4.
- `dt = 0.1 ms`, 1000 ms windows, 7 timed windows per configuration (median
  shown), each configuration in a fresh process. `load` is the 1-minute load
  average at the start of the configuration; the previous configuration of the same
  sweep adds about 1 to it.
- `cpus = 4`: the process was pinned to one P-core (`taskset -c 4`). `all`: not pinned.
- `backend = cusparse`: `brainevent.config.set_backend('gpu', 'cusparse')`.
- Peak memory: on CPU, the process peak RSS (includes about 0.7 GiB of Python/JAX
  runtime); on GPU, the device `peak_bytes_in_use`.
- `compile s`: ahead-of-time trace + lower + XLA compile of the rollout.
  `recompile s`: the one recompile on the second call (weak-typed initial state).

| device | cpus | backend | order | N | prec | status | build s | compile s | recompile s | s / sim-s | ms / step | peak mem GiB | E rate Hz | load |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | all | default | 200 | 1000 | f32 | ok | 1.9 | 0.59 | 0.27 | 3.754 | 0.3754 | 1.088 | 57.16 | 0.22 |
| cpu | all | default | 200 | 1000 | f64 | ok | 2.8 | 0.59 | 0.28 | 4.352 | 0.4352 | 1.219 | 57.16 | 0.77 |
| cpu | all | default | 2500 | 12500 | f32 | ok | 41.7 | 0.68 | 0.41 | 16.225 | 1.6225 | 1.986 | 28.76 | 1.06 |
| cpu | all | default | 2500 | 12500 | f64 | ok | 42.0 | 0.76 | 0.43 | 27.991 | 2.7991 | 2.199 | 28.84 | 1.41 |
| cpu | 4 | default | 200 | 1000 | f32 | ok | 2.0 | 0.88 | 0.57 | 0.720 | 0.0720 | 1.014 | 57.16 | 0.32 |
| cpu | 4 | default | 200 | 1000 | f64 | ok | 2.8 | 0.91 | 0.60 | 0.745 | 0.0745 | 1.020 | 57.16 | 0.47 |
| cpu | 4 | default | 2500 | 12500 | f32 | ok | 41.2 | 0.99 | 0.73 | 5.092 | 0.5092 | 1.684 | 28.76 | 0.59 |
| cpu | 4 | default | 2500 | 12500 | f64 | ok | 43.5 | 1.09 | 0.75 | 5.755 | 0.5755 | 1.754 | 28.84 | 0.93 |
| gpu | all | default | 200 | 1000 | f32 | ok | 3.1 | 0.81 | 0.55 | 3.981 | 0.3981 | 0.048 | 57.14 | 1.16 |
| gpu | all | default | 200 | 1000 | f64 | FAILED (abort (SIGABRT)) | - | - | - | - | - | - | - | 1.77 |
| gpu | all | default | 2500 | 12500 | f32 | ok | 18.3 | 1.09 | 0.75 | 4.908 | 0.4908 | 0.232 | 28.84 | 1.77 |
| gpu | all | default | 2500 | 12500 | f64 | FAILED (abort (SIGABRT)) | - | - | - | - | - | - | - | 1.96 |
| gpu | all | cusparse | 200 | 1000 | f32 | ok | 3.5 | 0.83 | 0.55 | 4.179 | 0.4179 | 0.048 | 57.14 | 1.76 |
| gpu | all | cusparse | 200 | 1000 | f64 | ok | 3.2 | 0.84 | 0.52 | 4.215 | 0.4215 | 0.049 | 57.16 | 1.58 |
| gpu | all | cusparse | 2500 | 12500 | f32 | ok | 18.2 | 1.05 | 0.79 | 9.544 | 0.9544 | 0.232 | 28.84 | 1.23 |
| gpu | all | cusparse | 2500 | 12500 | f64 | ok | 19.1 | 1.12 | 0.78 | 8.843 | 0.8843 | 0.426 | 28.90 | 1.90 |

Spread of the timed windows (min to max, in s per simulated second). Unpinned CPU:
3.33–4.04 (order 200, f32), 4.27–4.41 (200, f64), 12.48–17.26 (2500, f32) and
24.80–30.36 (2500, f64). Every pinned-CPU and GPU row stays within 4 % of its median.

### Reading the numbers

- **Use the pinned CPU rows (`cpus = 4`) for the G9 CPU comparison.** Unpinned, the
  same program is 3–5× slower at order 2500. Its window times also spread over up
  to 30 % of the median, which is too wide for a ≤ 1.1× gate.
- **GPU float64 does not run with brainevent's default GPU kernel.** The process
  aborts in `binary_csrmv_wat_hybrid_hetero_f64_float` ("expects matching vector
  dtype"). That kernel expects a float32 event vector, but the Simulator passes a
  float64 spike vector. The `cusparse` rows give a GPU baseline in both precisions.
  For G9 on the GPU, either compare float32 with the default row and float64 with
  the `cusparse` row, or compare both precisions with `cusparse`.
- On the GPU the Simulator is launch-bound: a step takes about 0.4 ms even at 1000
  neurons. At order 2500 the default GPU kernel is only about as fast as one pinned
  CPU core, and `cusparse` is about 2× slower.
- A bare `sim.cont(1000 ms)` call (`compile.public_cont_call` in the JSON)
  re-traces and re-compiles on every call, so it costs the compile time plus one
  window.
- The E rate is about 57 Hz at order 200 and about 29 Hz at order 2500. It is the
  same in both precisions and on both devices.

## 2026-10-22 — T10: gate G9, old `Simulator` vs K0 kernel, interleaved

File: `2026-10-22-T10-g9-interleaved.json` (every run with its core checks, plus
the comparison rows). Protocol: `benchmarks/g9_interleaved.py` (user decision
2026-10-22: CPU only, no load gate, one idle pinned core; GPU skipped). Host
`wanglab3090b-01a0` (i7-14700KF), commit `d8dccce`, `JAX_PLATFORMS=cpu`,
jax 0.12.0, brainevent 0.2.1, brainstate 0.6.0, numpy 2.5.4. Core 2 (sibling 3),
both 0.0 % busy at selection, load average 2.98-3.90 during the session (the
user's own jobs, on other cores). Old, new, old, new ... in fresh processes, 3
rounds per configuration, T0 measurement protocol inside each process; all 24
runs valid on the first attempt (core share >= 0.999, sibling <= 0.01 % busy).
Medians over the rounds; compile time reported, not gated.

| order | prec | gated | old ms/step | new ms/step | ratio | old peak GiB | new peak GiB | ratio | old compile s | new compile s | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2500 | f32 | yes | 0.5435 | 0.2853 | 0.525 | 1.683 | 1.639 | 0.974 | 1.08 | 1.68 | pass |
| 2500 | f64 | yes | 0.6038 | 0.4660 | 0.772 | 1.742 | 1.758 | 1.009 | 1.06 | 1.66 | pass |
| 200 | f32 | no | 0.0770 | 0.0284 | 0.369 | 1.006 | 1.043 | 1.037 | 0.95 | 1.55 | reported |
| 200 | f64 | no | 0.0801 | 0.0374 | 0.467 | 1.033 | 1.059 | 1.026 | 0.93 | 1.60 | reported |

Reference, the new kernel against the original T0 pinned rows (core 4, quiet host):

| order | prec | T0 ms/step | new ms/step | ratio | T0 peak GiB | new peak GiB | ratio | old re-measured / T0 step |
|---|---|---|---|---|---|---|---|---|
| 2500 | f32 | 0.5092 | 0.2853 | 0.560 | 1.684 | 1.639 | 0.973 | 1.067 |
| 2500 | f64 | 0.5755 | 0.4660 | 0.810 | 1.754 | 1.758 | 1.002 | 1.049 |
| 200 | f32 | 0.0720 | 0.0284 | 0.394 | 1.014 | 1.043 | 1.029 | 1.068 |
| 200 | f64 | 0.0745 | 0.0374 | 0.502 | 1.020 | 1.059 | 1.038 | 1.075 |

The re-measured old `Simulator` is 5-8 % slower than in T0 (busier host, other
core), which is why the gate uses the interleaved comparison. GPU: skipped by user
decision.

## 2026-10-22 — K1a-4: per-edge delays (one scatter) vs a scalar delay

File: [`2026-10-22-K1a-4-delays.json`](2026-10-22-K1a-4-delays.json) (all runs and
metadata). Script: `benchmarks/k1a4_delays.py sweep` (3 rounds, 5 for `cpu5`). CPU:
every run in a fresh process pinned to one idle core, delay settings interleaved
round by round; GPU 1 unpinned. Ratios are per-edge / scalar delay (1.5 ms) of the
same network and precision; gate: step ≤ 1.5, build ≤ 2. Delays `Uniform(0.1 ms, D)`;
every per-edge projection was delivered by the scatter (cost model `auto`).
`review`: 2000 `iaf_psc_alpha`, Poisson drive, recurrent `fixed_indegree(100)`
(`/tmp/k0/review/delays_exp.py`; silent recurrent network). `brunel`: model A at
order 400 (2000 neurons) and 2500 (12 500 neurons). GPU float64 uses brainevent's
`cusparse` backend (the default backend aborts, brainevent#199).

| device | net | prec | scalar ms/step | D = 2 ms step / build | D = 5 ms | D = 20 ms |
|---|---|---|---|---|---|---|
| cpu | review 2000 | f32 | 0.0292 | 0.93 / 1.09 | 0.95 / 1.09 | 0.96 / 1.09 |
| cpu | review 2000 | f64 | 0.0414 | 0.87 / 1.17 | 0.87 / 1.18 | 0.89 / 1.18 |
| cpu | brunel 400 | f32 | 0.0443 | 0.78 / 1.01 | 0.80 / 1.02 | 0.86 / 1.01 |
| cpu | brunel 400 | f64 | 0.0717 | 0.72 / 1.10 | 0.73 / 1.13 | 0.80 / 1.09 |
| cpu | brunel 2500 (cpu5) | f32 | 0.2708 | 1.17 / 1.46 | 1.24 / 1.52 | 1.39 / 1.57 |
| cpu | brunel 2500 (cpu5) | f64 | 0.3964 | 1.13 / 1.46 | 1.20 / 1.52 | **1.55** / 1.58 |
| cpu | brunel 2500 (cpu4) | f32 | 0.2459 | 1.24 / 1.43 | 1.29 / 1.49 | **1.52** / 1.55 |
| cpu | brunel 2500 (cpu4) | f64 | 0.3959 | 1.13 / 1.46 | 1.19 / 1.52 | **1.54** / 1.58 |
| gpu | review 2000 | f32 | 0.0486 | 0.89 / 1.13 | 0.89 / 1.15 | 0.89 / 1.15 |
| gpu | review 2000 | f64 | 0.0737 | 0.88 / 1.26 | 0.90 / 1.23 | 0.94 / 1.28 |
| gpu | brunel 400 | f32 | 0.1026 | 0.57 / 1.08 | 0.58 / 1.07 | 0.61 / 1.07 |
| gpu | brunel 400 | f64 | 0.1780 | 0.53 / 1.16 | 0.53 / 1.17 | 0.62 / 1.14 |
| gpu | brunel 2500 | f32 | 0.0710 | **1.63** / 1.43 | **1.72** / 1.49 | **1.84** / 1.55 |
| gpu | brunel 2500 | f64 | 0.4179 | 0.40 / 1.44 | 0.41 / 1.52 | 0.42 / 1.57 |

The gate network of the spec (review, 2000 × 100, up to 20 ms) passes on CPU and
GPU in both precisions. At order 2500 with delays up to 20 ms the ring is
200 × 12 500 (10 MB in f32, 20 MB in f64) and the random per-edge writes leave the
cache: CPU step 1.39-1.55 × (round to round). On the GPU at order 2500 the scalar
path is one brainevent CSR kernel per projection while the JAX scatter is about ten
small kernels plus a `lax.cond` (device-to-host predicate sync): 1.6-1.8 ×; the
fused kernel of brainevent#203 is the fix. An earlier sweep (`cpu3`, before the
GPU capacity change, same CPU code path) measured 1.39 / 1.40 at order 2500, 20 ms.

**Correction (K1a-10, 2026-10-23).** The GPU f64 brunel 400 scalar-delay runs used
`Dense` storage (`auto`). brainevent#204 made them compute wrong values: 57.68 Hz on
GPU against 52.44 Hz on CPU. Their rate column is wrong. Their step times are kept,
because the binary Dense kernel's cost does not depend on the event values. Every
other row here uses CSR or the scatter, or runs on the CPU, and is unaffected. K1a-10
casts f64 binary events to f32, which fixes this.
