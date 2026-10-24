# Upstream issues and workarounds — TODO list

_Date: 2026-10-22_ · Status: **live list** (append-only rows; update the status column
in place) · Parents: [K0 spec §9](2026-10-21-k0-kernel-spec.md) (status at filing),
[K1a spec](2026-10-22-k1a-kernel-hardening-spec.md), architecture doc §8.4 and §13.

Every bug or missing feature found in an upstream library (brainevent, braintrace,
brainstate, brainunit, braintools, jax) that the kernel works around gets one row
here. A workaround is removed only when a **released** upstream version contains the
fix and the dependency floor in `pyproject.toml` is raised to it. Line numbers refer
to `main` @ 7938574 (2026-10-22); K1a tasks that move the code update them.

## 1. Filed issues

Versions at filing: brainevent 0.2.1, braintrace 0.2.6 (latest releases on
2026-10-22). Status checked 2026-10-22 with `gh issue list`; updated 2026-10-25 (K1b-0)
after the upstream audit of brainevent 0.3.0 / braintrace 0.2.7 (all eleven issues
fixed or implemented). K1b-0 raised the floors to `brainevent>=0.3.0` (runtime) and
`braintrace>=0.2.7` (dev/CI only). The file:line cells keep their filing-time lines.

| repo | issue | summary | type | status (updated 2026-10-25, K1b-0; braintrace rows K1L-0) | our workaround (file:line in `brainpy_state/`) | TODO |
|---|---|---|---|---|---|---|
| brainevent | [#198](https://github.com/chaobrain/brainevent/issues/198) | reverse-mode weight gradient under `BinaryArray` uses the raw input values instead of the binarised events (CSR, dense, JITC; forward-mode correct) | bug | **fixed in brainevent 0.3.0** (floor since K1b-0, 2026-10-25). Rule kept with a new rationale: `BinaryArray` is a mask (0.3.0: active iff `x != 0`), not a value, so counts must not reach it; count payloads move to `WeightedArray` (#201, K1b). Was: open | binary payload path only for sources with `max_multiplicity == 1`: `_kernel/connectivity/storage.py:69-97` (`payload_path`), used at `storage.py:321,439` and `_kernel/compiler.py:586`; rationale in `storage.py:25-27` | when a release ≥ the version that fixes #198 is the floor, allow the binary path for count payloads (or keep the rule if #201 lands first) |
| brainevent | [#199](https://github.com/chaobrain/brainevent/issues/199) | GPU float64 `BinaryArray @ CSR` aborts the process in `binary_csrmv_wat_hybrid_hetero_f64_float` (default backend; `cusparse` works) | bug | **fixed in brainevent 0.3.0** (floor since K1b-0): float64 events on the default GPU backend no longer abort (K1b-0: Brunel order 2500 float64 on GPU 1, default backend, E rate 28.78 Hz = CPU, identical spike counts); K1b-0 removed the float32 event cast and the "needs cusparse" benchmark comments; the `k1a_gpu_g9.py` f64 rows still select `cusparse` (moving them is a K1b G3 re-measurement). Was: open; **still reproduces on brainevent 0.2.1** (latest release; K1a-7 re-test on GPU 1, 2026-10-22: kernel Brunel order 2500 float64, default backend, aborts with `binary_csrmv_wat_hybrid_hetero_f64_float expects matching vector dtype`); K1a-10 (2026-10-23): the float32 event cast for #204 also avoids this abort in `_kernel`, so float64 GPU runs no longer need `cusparse` | none in `_kernel`; float64 GPU benchmarks select the backend: `benchmarks/brunel_kernel.py` `run_one` (`--brainevent-backend cusparse`, `# upstream: brainevent#199`), `benchmarks/k1a_gpu_g9.py` and `benchmarks/k1a4_delays.py` (f64 rows with `cusparse`) | when a release with the fix is the floor, drop the cusparse-only f64 rows and gate the default backend |
| brainevent | [#200](https://github.com/chaobrain/brainevent/issues/200) | `csr_to_csc_index` int64 → int32 scatter `FutureWarning` under x64 | bug | **fixed in brainevent 0.3.0** (floor since K1b-0): the comment is removed and the plasticity fast-path tests turn this `FutureWarning` into an error. Was: fixed on main (e18f81c), unreleased | none (warning tolerated) at `_kernel/plasticity.py:443` | raise the floor to the release containing e18f81c; then make the warning an error in the plasticity tests |
| brainevent | [#201](https://github.com/chaobrain/brainevent/issues/201) | event-driven matmul for count-valued events (multiplicity > 1) with exact VJPs | enhancement | **implemented in brainevent 0.3.0** (`WeightedArray`); workaround kept in K1b-0, adoption is a later K1b task (summation-order numerics + benchmark). Was: open | exact dense float matmul (not event-driven) for count payloads: `_kernel/connectivity/storage.py:323` (Dense) and `storage.py:441` (CSR) | when released ≥ the version with #201, route `'float'`-path event payloads through it and benchmark |
| brainevent | [#202](https://github.com/chaobrain/brainevent/issues/202) | differentiable `update_csr_on_binary_pre/post`, or a generic per-edge plasticity kernel with arbitrary edge state | enhancement | **implemented in brainevent 0.3.0** (differentiable `update_csr_on_binary_pre/post`, `csr_apply_on_pre/post`); workaround kept in K1b-0, adoption is a later K1b task (FD gate through `w_min`/`w_max` needed). Was: open | forward-only fast path + differentiable JAX reference path: `_kernel/plasticity.py:449-467` (`_resolve_path`), fast-path calls at `plasticity.py:545-548` | when released, use the kernel for both paths where it fits; keep the reference path as the fallback for rules it does not cover |
| brainevent | [#203](https://github.com/chaobrain/brainevent/issues/203) | per-edge-delay delivery fused with the event-driven matmul | enhancement | **implemented in brainevent 0.3.0** (`csr_delay_deliver`, `CSRDelayLayout`); workaround kept in K1b-0, adoption is a later K1b task (brainevent requires `1 <= d < L`, our ring has `L = max delay`: ring-layout decision). Was: open | one storage per distinct delay value: `_kernel/compiler.py:606-616`; replaced by a single-op JAX scatter in K1a-4 | when released, add it as the fast path behind the K1a-4 cost model |
| brainevent | [#204](https://github.com/chaobrain/brainevent/issues/204) | GPU: `BinaryArray(float64 vector) @ dense W` returns wrong values silently (bool / float32 events and CPU are correct); found in K1a-7 G3 (order-200 float64 Dense rate 59.7 Hz vs 57.1 Hz CSR and old `Simulator`) | bug | **fixed in brainevent 0.3.0** (floor since K1b-0); K1b-0 removed `_events`, `_binary_dense` and `_binary_sparse` (7c3dfba): float64 payloads go to brainevent unchanged; regression test `storage_test.py::TestFloat64BinaryPayload` (CPU and GPU 1). Was: open (filed 2026-10-22 on 0.2.1) | K1a-10: binary-path float64 events are cast to float32 before brainevent (0/1 exact; output keeps W's dtype): `_kernel/connectivity/storage.py:163-190` (`_events`, `_binary_dense` with a `custom_jvp` that keeps the event tangent in float64), used at `storage.py:374-375` (Dense) and `:494-496` (CSR). The same cast avoids the #199 abort on the CSR default backend (GPU f64 order 200/800 runs on the default backend after K1a-10) | when a release ≥ the version that fixes #204 (and #199) is the floor, drop `_events` / `_binary_dense` and re-run `storage_test.py::TestFloat64BinaryPayload` on GPU |
| braintrace | [#184](https://github.com/chaobrain/braintrace/issues/184) | `KeyError` in `HiddenGroup.transition` when a `HiddenState`'s new value depends only on inputs | bug | **fixed in braintrace 0.2.7; workaround removed in K1L-0** by evidence: without `new + 0 * old` the one-step D-RTRL/e-prop gradients equal BPTT on the G7 nets (<= 2.1e-16, gate K1L-G7); with it the LIF `xh`/`hh` gradients were off by 9-10 % (the workaround made braintrace see the same-step `h -> o` path as a dropped bypass). Was: fixed upstream, workaround kept (K1b-0) | `new + 0 * old`: `_kernel/model.py:102-107` (`_keep_dependency`), used at `model.py:513`, `_kernel/models/_common.py:98`, `_kernel/learn/etrace.py:327-329,493,500`; checker rule 5 at `learn/etrace.py:278-293` (lines as of K1a-1) | done (K1L-0): `_keep_dependency` and checker rule 5 removed |
| braintrace | [#185](https://github.com/chaobrain/braintrace/issues/185) | silent all-zero online gradients when the output reads an ETP weight through a non-hidden intermediate value | bug | **fixed in braintrace 0.2.7; rule 3 narrowed in K1L-0**: bypass outputs (pre-reset reads, readout weights) are accepted and trained exactly; `check_learner` rejects them only under `learning_signal='modulatory'` (no bypass rewrite) and turns braintrace's `OUTPUT_BYPASS_DROPPED` diagnostic into an `EtraceError`; rule 1 accepts pure readout weights. Was: fixed upstream, rule 3 kept (K1b-0) | checker rule 3 (outputs must be read from new `HiddenState` values): `_kernel/learn/etrace.py:294-305` (`# upstream:` comment added in K1a-5) | done (K1L-0) |
| braintrace | [#186](https://github.com/chaobrain/braintrace/issues/186) | `etrace_grad` fails for a weight-free `while_loop` whose result reaches a `HiddenState` through further ops | bug | **fixed in braintrace 0.2.7**: weight-free `while` loops allowed (rule 2 docstring, K1L-0); rule 2 still rejects ETP weights inside `while` bodies (by design upstream). Was: fixed upstream, comments updated (K1b-0) | no `while_loop` in neuron steps (§7.3 rule 3); checker rule 2 rejects ETP primitives in `while` bodies only (K1a-5 relaxed it from any `scan`/`while` body; static `scan`/`fori_loop` bodies give the Python-unrolled gradients, `learn/etrace_test.py::test_checker_accepts_braintrace_weight_in_static_loops`): `_kernel/learn/etrace.py:60-62` (`_LOOP_PRIMITIVES`), `:247-254`. Note: an ETP weight inside a `while` body raises braintrace's own `NotImplementedError` ("Weight state used inside a while loop", by design: no fixed per-iteration Jacobian); #186 is the weight-free case | done (K1L-0); §7.3 rule 3 of the K0 spec ("no while_loop in neuron steps") is superseded for weight-free loops |
| braintrace | [#187](https://github.com/chaobrain/braintrace/issues/187) | document the "no `ParamState` writes inside the online step" rule; make `NotSupportedError` actionable | documentation | **done in braintrace 0.2.7**: rule 4 kept (checked first) with the decision-L4 wording and a short pointer to the two supported patterns (K1L-0). Was: done upstream (K1b-0) | checker rule 4 with an actionable message: `_kernel/learn/etrace.py:238-245` (`# upstream:` comment added in K1a-5) | done (K1L-0) |
| brainevent | [#203](https://github.com/chaobrain/brainevent/issues/203) (update, K1a-4) | measured motivation for the fused per-edge-delay kernel: the JAX scatter of K1a-4 meets the K1a gate on the spec network (2000 × 100, delays ≤ 20 ms: 0.87-0.96 × the scalar-delay step, CPU and GPU) but not at Brunel order 2500 with delays ≤ 20 ms: CPU 1.39-1.55 × (10-20 MB ring, cache-bound random writes), GPU f32 1.63-1.84 × (~10 small kernels + one `lax.cond` sync per projection vs one CSR kernel); numbers commented on the issue | enhancement | **implemented in brainevent 0.3.0** (see the #203 row above). Was: open; known gap accepted by the lead (2026-10-22) | `DelayScatter` (`_kernel/delay_scatter.py`, `# upstream: brainevent#203` at `_scatter`), results `benchmarks/results/2026-10-22-K1a-4-delays.json` | when released, put the fused kernel behind `DelayScatter.deposit` / the cost model and re-run `benchmarks/k1a4_delays.py` at order 2500 (target ≤ 1.5 × on CPU and GPU) |
| braintrace | [#197](https://github.com/chaobrain/braintrace/issues/197) | `output_bypass._is_blocking` builds a `Jaxpr` without `debug_info`, so jax emits `DeprecationWarning: core.Jaxpr is missing a DebugInfo object` once per equation (kernel suite: 28 -> ~6.5k warnings with 0.2.7); jax plans to make it an error | bug | open (filed 2026-10-25 on 0.2.7, K1b-0; repro `/tmp/k0/upstream/braintrace/repro5_debuginfo_warning.py`) | pytest `filterwarnings` ignores exactly this message: `pyproject.toml` `[tool.pytest.ini_options]` (`# upstream: braintrace#197`) | when a release with the fix is the dev floor, drop the filter |

### 1.1 Code comments to add (K1a)

This docs PR adds no code. The K1a task that next touches each location adds a
`# upstream: <repo>#<n>` comment at the workaround (agent brief, step 5):

| comment | location | task |
|---|---|---|
| `# upstream: brainevent#198` | `_kernel/connectivity/storage.py` `payload_path` | K1a-4 (done) |
| `# upstream: brainevent#201` | `_kernel/connectivity/storage.py` float-path matmuls (Dense and CSR `forward`) | K1a-4 (done) |
| `# upstream: brainevent#203` | `_kernel/compiler.py` per-delay grouping (or its K1a-4 replacement) | K1a-4 (done: `_kernel/delay_scatter.py` `_scatter`) |
| `# upstream: brainevent#200` | `_kernel/plasticity.py` `csr_to_csc_index` call | K1a-4 (done) |
| `# upstream: brainevent#202` | `_kernel/plasticity.py` `_resolve_path` | K1a-4 (done) |
| `# upstream: brainevent#199` | `benchmarks/brunel_kernel.py` backend flag | K1a-7 (done) |
| `# upstream: brainevent#204` | `_kernel/connectivity/storage.py` `_events` / `_binary_dense`, Dense and CSR binary `forward` | K1a-10 (done) |
| `# upstream: braintrace#184` | `_kernel/model.py` `_keep_dependency`, `_kernel/learn/etrace.py` `_keep_dependency` and rule 5 | K1a-1; removed in K1L-0 (no marker left) |
| `# upstream: braintrace#185` | `_kernel/learn/etrace.py` rule 3 | K1a-5 |
| `# upstream: braintrace#186` | `_kernel/learn/etrace.py` rule 2 | K1a-5 |
| `# upstream: braintrace#187` | `_kernel/learn/etrace.py` rule 4 | K1a-5 |

## 2. jax candidates (not filed)

jax is not a chaobrain repository: candidates are recorded here, not filed.

| summary | repro (`/tmp/k0/upstream/jax/...`) | jax version | our workaround (file:line) | found in |
|---|---|---|---|---|
| — | — | — | — | — |

## 2.1 Unattributed problems (not filed; library not identified)

| summary | observed | workaround (file) | found in | next step |
|---|---|---|---|---|
| the Python process sometimes hangs at interpreter exit after a kernel run (threads left in futex waits; seen in `subprocess` children on the remote host CPU) | `benchmarks/k1a_compile_time.py one <case>` run from `validation/compile_gate_test.py` | `os._exit(0)` after printing `RESULT_JSON` (`benchmarks/k1a_compile_time.py`) | K1a-1 | isolate (jax / XLA CPU thread pool vs brainevent numba FFI); minimal repro, then file upstream per §3 |

## 3. Procedure (from the agent brief, K1a onwards)

If a task hits a bug or missing feature in brainevent / braintrace / brainstate /
brainunit / braintools / jax:

1. Write a minimal self-contained repro under `/tmp/k0/upstream/<repo>/` and run it on
   the latest release.
2. Search existing issues (`gh issue list -R chaobrain/<repo> --state all --search ...`;
   jax: `-R jax-ml/jax`) and comment on an existing one instead of duplicating.
3. File the issue (`gh issue create`, labels bug/enhancement; sections Summary /
   Environment / Reproduction with real output / Actual vs expected / Likely cause /
   Workaround in brainpy.state; no Claude attribution lines). For jax, do **not** file:
   add a row to §2 instead.
4. Add a row to §1 in the task's PR: repo, issue link, summary, our workaround with
   file:line, and the TODO to remove the workaround when a release with the fix is the
   dependency floor.
5. Put a `# upstream: <repo>#<n>` comment at the workaround in code.
