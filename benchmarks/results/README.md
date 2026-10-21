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
