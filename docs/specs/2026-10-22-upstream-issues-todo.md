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
2026-10-22). Status checked 2026-10-22 with `gh issue list`.

| repo | issue | summary | type | status (2026-10-22) | our workaround (file:line in `brainpy_state/`) | TODO |
|---|---|---|---|---|---|---|
| brainevent | [#198](https://github.com/chaobrain/brainevent/issues/198) | reverse-mode weight gradient under `BinaryArray` uses the raw input values instead of the binarised events (CSR, dense, JITC; forward-mode correct) | bug | open | binary payload path only for sources with `max_multiplicity == 1`: `_kernel/connectivity/storage.py:69-97` (`payload_path`), used at `storage.py:321,439` and `_kernel/compiler.py:586`; rationale in `storage.py:25-27` | when a release ≥ the version that fixes #198 is the floor, allow the binary path for count payloads (or keep the rule if #201 lands first) |
| brainevent | [#199](https://github.com/chaobrain/brainevent/issues/199) | GPU float64 `BinaryArray @ CSR` aborts the process in `binary_csrmv_wat_hybrid_hetero_f64_float` (default backend; `cusparse` works) | bug | open (main e18f81c likely fixes the CSR kernels; unverified on GPU) | none in `_kernel`; float64 GPU benchmarks select the backend: `benchmarks/brunel_kernel.py:52-53` (`--brainevent-backend cusparse`) | K1a-7 re-tests on GPU 1 with the latest release; when a release with the fix is the floor, drop the cusparse-only f64 rows and gate the default backend |
| brainevent | [#200](https://github.com/chaobrain/brainevent/issues/200) | `csr_to_csc_index` int64 → int32 scatter `FutureWarning` under x64 | bug | fixed on main (e18f81c), unreleased | none (warning tolerated) at `_kernel/plasticity.py:443` | raise the floor to the release containing e18f81c; then make the warning an error in the plasticity tests |
| brainevent | [#201](https://github.com/chaobrain/brainevent/issues/201) | event-driven matmul for count-valued events (multiplicity > 1) with exact VJPs | enhancement | open | exact dense float matmul (not event-driven) for count payloads: `_kernel/connectivity/storage.py:323` (Dense) and `storage.py:441` (CSR) | when released ≥ the version with #201, route `'float'`-path event payloads through it and benchmark |
| brainevent | [#202](https://github.com/chaobrain/brainevent/issues/202) | differentiable `update_csr_on_binary_pre/post`, or a generic per-edge plasticity kernel with arbitrary edge state | enhancement | open | forward-only fast path + differentiable JAX reference path: `_kernel/plasticity.py:449-467` (`_resolve_path`), fast-path calls at `plasticity.py:545-548` | when released, use the kernel for both paths where it fits; keep the reference path as the fallback for rules it does not cover |
| brainevent | [#203](https://github.com/chaobrain/brainevent/issues/203) | per-edge-delay delivery fused with the event-driven matmul | enhancement | open | one storage per distinct delay value: `_kernel/compiler.py:606-616`; replaced by a single-op JAX scatter in K1a-4 | when released, add it as the fast path behind the K1a-4 cost model |
| braintrace | [#184](https://github.com/chaobrain/braintrace/issues/184) | `KeyError` in `HiddenGroup.transition` when a `HiddenState`'s new value depends only on inputs | bug | open | `new + 0 * old`: `_kernel/model.py:102-106` (`_keep_dependency`), used at `model.py:511`, `_kernel/models/_common.py:97`, `_kernel/learn/etrace.py:315-316,438,445`; checker rule 5 at `learn/etrace.py:268-282` | when released ≥ the version that fixes #184, remove `_keep_dependency` and rule 5 |
| braintrace | [#185](https://github.com/chaobrain/braintrace/issues/185) | silent all-zero online gradients when the output reads an ETP weight through a non-hidden intermediate value | bug | open | checker rule 3 (outputs must be read from new `HiddenState` values): `_kernel/learn/etrace.py:284-294` | when released ≥ the fixing version, keep rule 3 only if braintrace still returns zeros (then as a warning) |
| braintrace | [#186](https://github.com/chaobrain/braintrace/issues/186) | `etrace_grad` fails for a weight-free `while_loop` whose result reaches a `HiddenState` through further ops | bug | open | no `while_loop` in neuron steps (§7.3 rule 3); checker rule 2 rejects ETP primitives in any `scan`/`while` body: `_kernel/learn/etrace.py:239-244` — **too strict**, K1a-5 relaxes it to `while` bodies only | when released ≥ the fixing version, drop the `while` restriction from rule 2 |
| braintrace | [#187](https://github.com/chaobrain/braintrace/issues/187) | document the "no `ParamState` writes inside the online step" rule; make `NotSupportedError` actionable | documentation | open (behaviour intended) | checker rule 4 with an actionable message: `_kernel/learn/etrace.py:232-237` | keep rule 4 (intended behaviour); shorten our message to a pointer once upstream documents it |

### 1.1 Code comments to add (K1a)

This docs PR adds no code. The K1a task that next touches each location adds a
`# upstream: <repo>#<n>` comment at the workaround (agent brief, step 5):

| comment | location | task |
|---|---|---|
| `# upstream: brainevent#198` | `_kernel/connectivity/storage.py` `payload_path` | K1a-4 |
| `# upstream: brainevent#201` | `_kernel/connectivity/storage.py` float-path matmuls (Dense and CSR `forward`) | K1a-4 |
| `# upstream: brainevent#203` | `_kernel/compiler.py` per-delay grouping (or its K1a-4 replacement) | K1a-4 |
| `# upstream: brainevent#200` | `_kernel/plasticity.py` `csr_to_csc_index` call | K1a-4 |
| `# upstream: brainevent#202` | `_kernel/plasticity.py` `_resolve_path` | K1a-4 |
| `# upstream: brainevent#199` | `benchmarks/brunel_kernel.py` backend flag | K1a-7 |
| `# upstream: braintrace#184` | `_kernel/model.py` `_keep_dependency`, `_kernel/learn/etrace.py` `_keep_dependency` and rule 5 | K1a-1 |
| `# upstream: braintrace#185` | `_kernel/learn/etrace.py` rule 3 | K1a-5 |
| `# upstream: braintrace#186` | `_kernel/learn/etrace.py` rule 2 | K1a-5 |
| `# upstream: braintrace#187` | `_kernel/learn/etrace.py` rule 4 | K1a-5 |

## 2. jax candidates (not filed)

jax is not a chaobrain repository: candidates are recorded here, not filed.

| summary | repro (`/tmp/k0/upstream/jax/...`) | jax version | our workaround (file:line) | found in |
|---|---|---|---|---|
| — | — | — | — | — |

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
