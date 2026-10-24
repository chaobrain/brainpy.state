# K1b — substrate completion (α)

Date: 2026-10-25. Base: `main` @ 7c3dfba (renamed tree, package `brainspike`).
Parent: [architecture doc](2026-10-21-architecture-review-and-redesign.md) §7 (L1/L2,
§7.1–§7.3), §11, §12 K1b row. Predecessors: [K1a spec](2026-10-22-k1a-kernel-hardening-spec.md),
[K1a report](2026-10-22-k1a-report.md) §6, [split](2026-10-24-brainpy-state-split.md).
Upstream list: [upstream TODO](2026-10-22-upstream-issues-todo.md).

Paths are relative to `brainspike/_kernel/` unless they start with `brainspike/`,
`benchmarks/` or a repo-root file.

## Goal

Finish the L1/L2 substrate contracts that K2 model ports build on: payload paths,
per-edge delays, pre-aligned synapses, STP, gap junctions (lag), spike localisation,
plastic weights in edge order, and the `timed_event` budget. Nothing is exported
(`_kernel` stays private until K3, D-d), and the old kernel stays frozen (D-f).
K1b ends at the α checkpoint: gates K1b-G0 … G8 green, exit gate K1b-G1. K2 model
ports may start after it.

**Invariants for every task.**
- `-m "k0_gate or k1a_gate"` stays green; `_kernel` coverage > 90 %; mypy clean.
- Every K0/K1a network compiles in ≤ 5 s on CPU (§7.3 rule 4).
- NEST parity of the existing models is bit-identical in f64, unless the task changes
  semantics on purpose and says so in its PR.
- No model-dependent Python loops in the traced step (§7.3); fixed iterations are
  `scan`/peeled.
- Tests and benchmarks run on the remote host (`/tmp/k0/remote.sh`), GPU runs on GPU 1 only.

## Moved to K1c

Lead decision (2026-10-24). These are a separate milestone and are not specified here:
procedural storage (JITC) and the missing rules; recording at scale with the
merge-pass veto; checkpoint/restore; stimulus schedules and sweeps; the
distributed-layout spec with block-invariant draws; gap-junction WFR (old K1b-5b);
the `_nest_network` inventory; the β gates and report.

`_nest_network` inventory, recounted on the renamed tree:
`brainspike/_nest_network/*_test.py` = **31 files, 350 test functions**.

## Decisions

The policy is K1a spec §7: task agents decide and log internal structure, helper names,
file splits, test design and tolerances already fixed here. Contract/API changes,
semantics or numerics changes vs NEST/Brian2/K1a, gate changes, dependency changes and
edits outside `_kernel` go to the lead as a draft PR. Task agents append their own rows
to the log at the end.

**Lead decisions** (2026-10-24 / 2026-10-25):

| # | decision | reason |
|---|---|---|
| L1 | K1b = α only (K1b-0 … K1b-8 + gates/report). β becomes K1c. | K2 ports need only α; β can be planned on the merged α kernel. |
| L2 | Start on the renamed `brainspike` tree; specs committed first. | Task agents branch from the tree they will merge into. |
| L3 | At most 3 task agents at a time (shared with K1-L). | Local WSL memory. |
| L4 | braintrace#184 `_keep_dependency` removal is **not** in K1b-0; it moves to K1L-0 and is decided by evidence. | Removing it changes online gradients on the recurrent G7 net by 1.8–8 %; no test pins them. |
| L5 | braintrace stays dev-only until K3 (K1b-0 raises only the dev floor). | Overrides the K1-L draft Q13(a). |
| L6 | K1b-2: ring length becomes `d_max + 1`, so delays lie in `[1, L)` (audit option a, not traced delays). | brainevent `csr_delay_deliver` validates `1 ≤ d < L`; bypassing the check with traced delays hides errors. One extra ring row; a state-layout change. |
| L7 | Gap junctions: lag in K1b-5a; WFR moves to K1c. | Lag unblocks K2; WFR waits for `hh_psc_alpha_gap` parity numbers. |

**Defaults** (each: *default per draft recommendation; lead may override*). Q numbers
are those of the K1b draft (`/tmp/k0/k1b/k1b-plan-draft.md` §9).

| Q | task | decision | reason |
|---|---|---|---|
| Q4 | 5a | No new `gap` port kind: source `V=Graded(u.mV)`, target `I_gap=Current(u.pA, latch=False)`, difference synapse `Gap()` with `net.gap(...)` sugar (arch §10; fix the §7 table). G5 numbers set at a measurement checkpoint. Port `hh_psc_alpha_gap` in K1b. (WFR interpolation, Q4b, moves to K1c.) | One encoding; NEST parity needs the real model, which also exercises the peak rule. |
| Q5 | 4 | STP is a pre-aligned `SynapseModel` (`filters=False`, O(N_pre), differentiable); per-edge heterogeneous STP goes through K1b-7's generic rule in K2. Composition by an ordered list `synapse=[TsodyksMarkram(...), Expon(...)]`. | One keyword, ordered; NEST `tsodyks2_synapse` is per-source. |
| Q6 | 1 | Declare `Event(weighted=True)` and `Graded(sparse=True)` (default `False`). | `weighted` changes event semantics and should be visible; `sparse` is a performance hint. |
| Q7 | 1 | Graded delays are rounded to the grid like events. | NEST rounds gap/rate delays; interpolation can be added later without breaking this. |
| Q10 | 3 | Remove the explicit `NMDA` population class; keep `nmda_kinetics`; the G3 test keeps a copy of the old population as its reference. | Private kernel; one way to express NMDA. |
| Q11 | 7 | Plastic state lives in a storage-native layout behind an exact plan-held permutation; `EdgeSet` order is the canonical interface (read/set/observe/checkpoint). One-line reword of arch §7 L1. | Same guarantees, no per-step permutation. |
| Q12 | 7 | `d_ax` scalar per rule, `d_den,e = d_e − d_ax` per edge. | Matches NEST `stdp_synapse` (dendritic delay). |
| Q13 | 6 | Localisation + peak rule in K1b-6; hazard rule in K1b-6 only if cheap, else first task of K2. | 5a's model needs the peak rule; hazard blocks only K2 models. |
| Q14 | 6 | Keep `CAT_A` vs NEST on every step as the target. Measure the f64 high-k reference vs NEST first; if NEST's floor is above `CAT_A`, escalate with numbers and fall back to `CAT_A` vs the reference + convergence to NEST's floor. Never raise default substeps to pass. | Raising substeps hides the problem in cost. |
| Q15 | 8 | `p_overflow = 1e-6` per target-step; `rate_hint=` required for neuron sources into `'auto'`; default overflow policy `'count'`. | No silent default rate; "never a silent drop" (arch §7 L1). |
| Q16 | 8 | `iaf_psc_exp_ps.syn`: two-state `into=('I_syn_ex', 'I_syn_in')` routed by weight sign. | Keeps NEST's single receptor and the one-encoding rule (D-b). |
| Q21 | 0 | braintrace#186 fixed: keep §7.3 rule 3 (no `while` in neuron steps); relax only the etrace checker's comment/docstring. | Reverse-mode BPTT still rejects `while`. |

## Step plan

### K1b-0 — upstream bump (brainevent 0.3.0, braintrace 0.2.7)

**Contract.** One numerics-neutral PR, per the audit's proposed task list
(`/tmp/k0/k1b/upstream-audit.md` §5):
- floors: `brainevent>=0.3.0` in `pyproject.toml:43` and `requirements.txt:5`;
  `braintrace>=0.2.7` in `requirements-dev.txt:18` and `.github/workflows/type-check.yml:58`
  (dev-only, L5);
- #204/#199 cast: main 7c3dfba already rewrote the CSR binary-path payload JVP
  (`_binary_sparse`) so the f64 gradient test passes on 0.3.0. What is left is the cast
  itself: `_events`, `_binary_dense` + its JVP and `_binary_sparse` (`connectivity/storage.py:192-243`),
  used at `:425-426` (Dense) and `:545-547` (CSR). Remove them all and call
  `brainevent.BinaryArray(x) @ W/mat` directly; in `TestFloat64BinaryPayload`
  (`connectivity/storage_test.py:415`) delete the two tests of the cast itself
  (`test_events_dtype`, `test_dense_forward_bit_identical_to_uncast_on_cpu`) and keep the
  other three as #204 / f64-gradient regression tests, run on GPU 1;
- comments: #198 rationale at `connectivity/storage.py:98` ("BinaryArray is a mask; counts need
  WeightedArray, K1b-1"); drop #200 at `plasticity.py:443` and make `FutureWarning` an
  error in the fast-path tests; #186 (`learn/etrace.py:61`, `:249`, Q21) and #187
  (`learn/etrace.py:239`) wording; #199 comments/help text in `benchmarks/brunel_kernel.py:52`,
  `benchmarks/k1a9_storage.py:47`, `benchmarks/k1a4_delays.py:79,223`, `benchmarks/k1a_gpu_g9.py:5`
  (keep the `--brainevent-backend` flag; the G9 f64 CONFIGS stay as they are);
- document that `Event(max_multiplicity=1)` payloads are in {0, 1} (`ports.py:156`);
- pytest `filterwarnings` for the braintrace/jax `core.Jaxpr is missing a DebugInfo`
  `DeprecationWarning` (28 → 6560 warnings), scoped to braintrace;
- update the upstream TODO rows. Not in scope: #184 (L4), rule 3 narrowing (K1L-0),
  `WeightedArray` / `csr_delay_deliver` / differentiable plasticity (K1b-1/2/7).

Separate, non-blocking follow-up (own small task, upstream procedure): the GPU-1 failures
of `plasticity_test.py::TestGradients::test_grad_wrt_rule_params_matches_finite_differences`
(`plasticity_test.py:485`) and `TestLearners::test_bptt_learns_rule_parameters`
(`gpusparseCreateCsr ... invalid value`, both envs).

**Gate K1b-G0 (neutrality).** The seven K1a-G1 networks vs `main` @ 7c3dfba: f64 max
rel ≤ 1e-12, f32 within `CAT_B`, identical spike counts. Brunel order 2500 step ≤ 1.05 ×
on CPU and GPU 1. GPU-1 subset (`xla_test.py`, `connectivity/storage_test.py`,
`delay_scatter_test.py`, `plasticity_test.py`) green except the two pre-existing failures.
Brunel order 2500 f64 on the GPU default backend runs.

**Depends on:** nothing. Blocks every other K1b task.

### K1b-1 — payload paths (`WeightedArray`) and graded emission

**Contract.** `PAYLOAD_PATHS = ('binary', 'weighted', 'float')` (`connectivity/storage.py:65`);
`payload_path(src, *, relaxed=False)` (`:100`) decides and `Plan.describe()` reports it.
`'weighted'` uses `brainevent.WeightedArray` for count events (`max_multiplicity > 1`)
and sparse graded outputs, replacing the dense float matmuls of the count path
(`connectivity/storage.py:427`, `:548`; #201). `'float'` stays for dense graded values and relaxed
spikes (`runner.py:443`). A `NeuronModel` may declare `Graded` outputs reading any
state; graded sources work with pre ring, post ring, `DelayScatter` `'dense'` and zero
delay, with unit conversion `dst.unit / src.unit` (`ports.py:646`). New fields (Q6):
`Event(..., weighted=True)` (non-negative real per-source amplitude) and
`Graded(unit, sparse=False)`. Graded delays round to the grid (Q7). Neutral refactor in
the same PR: move `Program._pre_payload/_delayed/_forward/_deliver*` (`runner.py:388-488`)
into a new `delivery.py`, which K1b-2/3/5a/7 then edit.

**Gate.** Weighted vs float, Dense and CSR, 1-D and batched: f64 value and VJP
≤ 1e-12, `jacfwd == jacrev`; `spike_generator` with duplicates into `iaf_psc_alpha`
identical to K1a; graded `V` through all four delay placements equals a hand-rolled
reference. Count-payload Brunel step ≤ 1.0 × K1a on CPU and GPU 1. K0 G6 BPTT FD green.

**Depends on:** K1b-0.

### K1b-2 — fused per-edge delays (brainevent#203) and `PreRing`

**Contract.** New `DelayScatter` strategy `'fused'` via `brainevent.csr_delay_deliver_layout`
with a `CSRDelayLayout` built at compile time, chosen by `DelayCostModel.choose`
(`compiler.py:162`, `:273`) with constants re-calibrated per backend; `'events'`,
`'dense'` and the JAX scatter stay as fallbacks/reference. **State-layout change (L6):**
the arrival ring becomes `d_max + 1` rows so delays are in `[1, L)` (today
`L = d_max`, `compiler.py:923`); edge order then coincides with the layout's `perm`, and
`weight_from_edges` / `weight_to_edges` stay exact. Weighted events use weighted delivery
(K1b-1). Gradients: the brainevent VJP, else the existing `custom_jvp`
(`delay_scatter.py:359`). `PreRing`: measure the suspected whole-ring copy per step
(`delays.py:237`) and fix it with the `PostRing.take` barrier pattern (`delays.py:291`)
or a carried slot index; record the measurement either way.

**Gate K1b-G2.** Brunel order 2500, delays uniform up to 2, 5, 20 ms: step ≤ 1.5 × the
scalar-delay network on pinned remote CPU and GPU 1, f32 and f64; build ≤ 2 ×. K1a-G2
spec network (2000 × 100) ≤ 1.05 × its K1a numbers. Fused vs scatter: counts exact,
summed weights within `CAT_B` (multapses, `d = d_max`, zero-delay group); BPTT FD (G6a)
through a fused projection; `PreRing` change bit-identical. Re-run
`benchmarks/k1a4_delays.py` and update brainevent#203.

**Depends on:** K1b-1.

### K1b-3 — pre-aligned `SynapseModel` (NMDA as `synapse=NMDA()`)

**Contract.** `SynapseModel.alignment ∈ {'post', 'pre'}` (`synapse.py:72-86`, today
post only, `:247`). A pre-aligned model has `init_state(n_pre, dtype)`,
`step(state, x_pre, ctx) -> (state, out_pre)` and `output_kind() -> Graded | Event(weighted=True)`.
The compiler lowers each distinct `(source segment, synapse.merge_key())` to one hidden
receptor pool: no gids, updated in the source's schedule slot right after the source,
driven by its spikes, emitting through the projection's own delay, storage and weight.
Equal parameters share one pool; pre-aligned models never enter the post merge pass.
Pool streams are keyed `"<proj>/syn"` (D-g). `models/nmda.py`: `NMDA` becomes a
pre-aligned model wrapping `nmda_kinetics` (`:39`); the population class (`:89`) is
removed (Q10). Per-neuron array synapse parameters are allowed (lifts `synapse.py:138`).

**Gate K1b-G3.** The K0 T8 NMDA network (explicit population) vs `synapse=NMDA()`:
f64 ≤ 1e-12, f32 `CAT_B`, spikes identical. One pool per (source, parameters), listed
with its memory in `Plan.describe()`; pool memory O(N_pre). BPTT through the pool.

**Depends on:** K1b-1.

### K1b-4 — Tsodyks–Markram STP

**Contract.** `TsodyksMarkram(U, tau_rec, tau_fac)` with NEST `tsodyks2_synapse`
semantics: a pre-aligned `SynapseModel` with `filters=False`, state `(u, x)` per source,
output `Event(weighted=True)` with efficacy `u⁺·x` per spike. The per-edge weight
multiplies the efficacy, so STP onto a neuron-owned receptor port is legal: only
`filters=True` models are rejected there (`check_synapse_connection`, `synapse.py:216`).
Chaining: `synapse=[TsodyksMarkram(...), Expon(5*u.ms)]` (Q5). Multiplicity k in one
step = k sequential spikes at the same time (closed form). Heterogeneous per-edge STP is
not pre-aligned (K2, through K1b-7's generic rule).

**Gate K1b-G4.** NEST 3.10 `tsodyks2_synapse`, one connection onto `iaf_psc_exp`,
20/50/100 Hz trains, facilitating and depressing sets of the NEST examples: `V_m`
within `CAT_B`, `I_syn` ≤ 1e-9 pA. `u`, `x` vs the closed-form recursion; delay
equivalence (update at emission + shift); BPTT FD through `U`.

**Depends on:** K1b-1, K1b-3.

### K1b-5a — gap junctions (lag) and the `hh_psc_alpha_gap` port

**Contract (Q4).** Encoding: source `V=Graded(u.mV)`, target
`I_gap=Current(u.pA, latch=False)`, difference synapse `Gap()`:
`I_i = Σ_j g_ij V_j − G_i V_i`, `G_i = Σ_j g_ij` computed at compile time. Sugar
`net.gap(view, rule, g=..., mode='lag')` enforces the NEST rules: symmetric edges,
scalar or per-edge conductance, recurrent view, no user delay. Lag: both `V` terms are
read one step back from the pre ring (NEST's WFR seed). Port NEST `hh_psc_alpha_gap`
into `models/` with K1b-6's peak rule. `mode='wfr'` raises "K1c". Fix the arch §7 L1
table (no `gap` kind; `ports.py:45`).

**Gate K1b-G5 (lag part; numbers confirmed at a measurement checkpoint, Q4).** Two
`hh_psc_alpha_gap` neurons with different initial `V` vs NEST (WFR on): spike times
within 2·dt for the first 10 spikes and phase-locking reached. NEST
`gap_junctions_inhibitory_network`: rates and CV-ISI by the G2 rule, 5 seeds each side.
Unit tests: 2-neuron and 3-chain arithmetic, equal `V` gives 0, g = 0 decouples, every
guard of `brainspike/_nest_network/simulator_gap_test.py`, BPTT FD through `g`.

**Depends on:** K1b-1 (graded `V`), K1b-6 (peak rule).

### K1b-6 — spike localisation and the peak rule

**Contract.** `SpikeRule(..., localize=None | 'hermite' | 'secant', iterations=3)`
(`spiking.py:227`). In the substep where the margin crosses zero: locate `τ ∈ (0, h]`
from the cubic Hermite dense output (RK4 ends), optionally refined by a fixed number of
secant/Illinois steps re-integrating one sub-step (no `while_loop`); apply the jump at
`τ`; integrate `h − τ`. All inside the substep `scan` (`integrators.py:228`, hook
`model.py:645-657`). `localize=None` is bit-identical to K1a. The located offset may be
declared as output `spike_offset` (a timed source; capability `precise_spike_times` for
RK models). Several spikes per step are still counted. `SpikeRule.peak(var='V_m', threshold=0*u.mV)`
detects a local maximum above threshold with no jump (NEST `hh_*`). Hazard rule per Q13.
`aeif_cond_alpha` defaults to the localised rule after the gate.

**Gate K1b-G1 (exit gate).** Spiking `aeif_cond_alpha` (`aeif_cond_alpha/dc` and a
Poisson-driven case) vs NEST 3.10 at default substeps: `V_m` within `CAT_A`
(`brainspike/_nest_validation/tolerance_conventions.py:169`) on **every** step, including
upswing and reset steps (today only off them, `models/nest_parity_test.py:323`); spike
times equal. Convergence err(k)/err(2k) ≥ 3 on spike steps for k ∈ {4, 8, 16} vs an f64
k = 1024 reference. Step ≤ 1.25 × K1a `aeif`; compile ≤ 5 s. Fallback per Q14.

**Depends on:** K1b-0.

### K1b-7 — plastic weights in edge order

**Contract (Q11, Q12).** Every read, write, introspection and observer of plastic
weights and edge state is in `EdgeSet` order; the working state stays storage-native
behind an exact plan-held permutation. Plastic projections may deliver through CSR,
scatter, fused (K1b-2), or Dense on request (`auto` never picks it); lifts
`network.py:553` and `compiler.py:949`. Per-edge delays: `d_e = d_ax + d_den,e`, post
events reach edge e at `d_den,e` (per-edge gather from the post-spike `PreRing`); lifts
`compiler.py:939`. Generic per-edge rules `on_pre/on_post(edge, pre_vals, post_vals, params)`
over an edge-state pytree through `brainevent.csr_apply_on_pre/post` (#202), replacing
the forward-only path (`plasticity.py:545-560`) where it is not faster; the reference
path stays. Post-state reads `PlasticityRule.reads = ('V_m', ...)` (strictly prior
values). `xs = (x > 0)` (`plasticity.py:539`) becomes `x != 0` with explicit multiplicity
(k sequential updates). Trainable plastic weights stay `NotImplementedError`
(`network.py:550`); K1L-5 builds on this task, and K1L-4 lands first in `plasticity.py`.

**Gate K1b-G6.** (a) STDP weights after 1 s identical across CSR, scatter, fused and
dense (f64 ≤ 1e-12, multapses). (b) NEST `stdp_synapse` with heterogeneous delays (100
synapses, U(1, 5) ms, 10 Hz Poisson pre/post, 10 s): final weights within `CAT_B`.
(c) step ≤ 1.2 × the K1a fast path on the G8 network. K0 G8 green; generic vs reference
path ≤ 1e-12 with gradients in rule parameters.

**Depends on:** K1b-2.

### K1b-8 — `timed_event` capacity as a statistical budget

**Contract (Q15, Q16).** `TimedEvent(capacity='auto' | int)` (`ports.py:213`). `'auto'`:
`λ = Σ_proj indegree × rate × dt`, capacity = Poisson quantile at `p_overflow = 1e-6`,
minimum 4; device sources declare their rate, neuron sources need `rate_hint=` on
`connect`. The run-time overflow counter per target is recordable;
`overflow='count' | 'raise' | 'worst_case'` on `connect` replaces `strict_capacity`
(`network.py:518`); `'raise'` checks after the run; `'worst_case'` keeps today's
compile-time check (`timed.py:356`). Timed sources with `Expon` (`SynapseModel.step_timed`,
exact sub-step integration; others raise) and with plasticity (sub-step trace offsets,
CSR and scatter). Timed projections may use the scatter for per-edge delays; Dense stays
rejected (`network.py:569`). K1a-3 follow-ups: sign-routed `iaf_psc_exp_ps.syn` (Q16),
`TimedEvent` receptor default jump, `into=` on `LIF.delta`.

**Gate K1b-G7.** Precise Brunel (`iaf_psc_exp_ps` + `poisson_generator_ps`, 10 000
neurons, indegree 1000), `'auto'`: compile ≤ 5 s; observed overflow fraction ≤ 10 ×
`p_overflow` over 1 s; step time and memory reported vs the worst-case run; rates by
the G2 rule vs NEST (5 seeds). Precise networks whose worst case fits stay bit-identical
in f64; timed + `Expon` vs exact integration ≤ 1e-12; `models/precise_nest_test.py` green.

**Depends on:** K1b-0.

### K1b-R — gates and report

`K1B_GATES` in `validation/gates.py` (next to `K1A_GATES`, `:144`), markers `k1b_gate` /
`k1b_g0` … `k1b_g8`, CI selection `-m "(k0_gate or k1a_gate or k1b_gate) and not requires_nest"`.
**Gate K1b-G8 (regressions):** K0 + K1a gates green; every K0/K1a/K1b gate network
compiles ≤ 5 s; coverage > 90 %; mypy clean. Report `docs/specs/<date>-k1b-report.md`:
gate table, protocol, decisions, upstream status, follow-ups for K1c and K2. The lead
decides K1c at this checkpoint.

**Depends on:** K1b-0 … K1b-8.

### Waves (≤ 3 task agents at a time, shared with K1-L)

| wave | tasks | hot files |
|---|---|---|
| W0 | K1b-0 | `pyproject.toml`, `connectivity/storage.py`, comments |
| W1 | K1b-1, K1b-6, K1b-8 | 1: `storage.py`, `ports.py` outputs, `compiler.py:881`, new `delivery.py`; 6: `spiking.py`, `integrators.py`, `model.py`, `models/aeif_cond_alpha.py`; 8: `timed.py`, `models/precise.py`, `ports.py` `TimedEvent` (rebases on 1) |
| W2 | K1b-2, K1b-3, K1b-5a | 2: `delay_scatter.py`, `delays.py`, cost model; 3: `synapse.py`, `models/nmda.py`, synapse lowering; 5a: new `gap.py`, `network.gap`, `models/hh_psc_alpha_gap.py` |
| W3 | K1b-4, K1b-7 | 4: new `synapses/stp.py`, owner check; 7: `plasticity.py`, `delay_scatter.py`, `delivery.py` plastic path (after K1L-4) |
| W4 | K1b-R | measures the merged tree |

`compiler.py` and `delivery.py` sections touched in the same wave are disjoint; the
later PR rebases.

## How to verify

- Each task: its gate above, run remotely (`/tmp/k0/remote.sh <task> '...'`), plus
  `-m "k0_gate or k1a_gate"`, full `_kernel` suite with
  `--cov=brainspike/_kernel` (> 90 %) and mypy; GPU rows on GPU 1 (`GPU=1`),
  benchmarks with `BENCH=1`. The PR body records the command and result.
- α checkpoint: `-m "k0_gate or k1a_gate or k1b_gate"` green remotely, NEST rows
  included; the K1b report lists every gate number.

## Decision log (task agents)

| task | decision | rationale | date |
|---|---|---|---|
| — | — | — | — |
