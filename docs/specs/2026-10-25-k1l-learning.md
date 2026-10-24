# K1-L — learning track

Date: 2026-10-25. Base: `main` @ 7c3dfba (renamed tree, package `brainspike`).
Parent: [architecture doc](2026-10-21-architecture-review-and-redesign.md) §8 (§8.3 K1-L
gates), §11 L1–L4 and D-c, §12 K1-L row. Predecessors: [K1a spec](2026-10-22-k1a-kernel-hardening-spec.md)
(K1a-5 learner decisions), [K1a report](2026-10-22-k1a-report.md). Runs in parallel with
[K1b](2026-10-25-k1b-substrate.md). Upstream list: [upstream TODO](2026-10-22-upstream-issues-todo.md).

Paths are relative to `brainspike/_kernel/` unless they start with `brainspike/` or a
repo-root file.

## Goal

Build the learning methods of arch §8 on the K1a learner substrate: measured update
quality of the online learners, per-step and batched online learning, three-factor
learning, differentiable plasticity and hybrid weights, ES, readouts, spike-time
gradients, and an EventProp prototype within the L2 scope. Nothing is exported (D-d);
the old kernel is not touched (D-f); online algorithms stay in braintrace (L1).

**K1a-5 rules every learner keeps:** built from a `Program`; each episode runs from an
explicit `state0` at global step `n0` (D-c); spike mode / `detach_reset` only through
`program.replace(...)`; episode length from `data` or `duration=`; `trials=` marks a
batched `state0`; learn phase after the observe phase of every step with
`(n + 1) % every == 0`; validation at construction with capability errors (§8.2-5).

**Invariants:** `-m "k0_gate or k1a_gate"` green remotely; `_kernel` coverage > 90 %;
NEST parity unchanged; forward numerics bit-identical unless a task says otherwise
(K1L-4 changes relaxed + plasticity on purpose); every learner jit goes through `xla.py`
and compiles in ≤ 5 s on its gate network.

## Decisions

Policy as in K1a spec §7 (task agents decide internal structure, names, test design and
fixed tolerances; contract/API/semantics/gate/dependency changes go to the lead as a
draft PR). Task agents append their rows to the log at the end.

**Lead decisions** (2026-10-24 / 2026-10-25):

| # | decision | reason |
|---|---|---|
| L1 | **EventProp is the continuous Wunderlich–Pehle adjoint** with interpolated spike times (eventax-like), evaluated on our step grid. It is *not* an exact discrete adjoint of the step forward. Overrides draft Q5(a). | Matches the published method and eventax; an exact discrete adjoint duplicates autodiff. |
| L2 | The K1L-10 gate is approximate: rtol ≈ 1e-3 vs FD and vs autodiff of the same forward. The exact numbers are confirmed by the lead at a measurement checkpoint. | The continuous adjoint differs from the discretised forward by the interpolation error. |
| L3 | K1L-9 model scope: LIF with exponential current synapse, no refractoriness, hard reset, homogeneous delays ≥ 1 step, recurrence allowed, ≤ 1 spike per neuron per step. | L2 scope; adaptation, refractory clamps, heterogeneous/zero or learnable delays and conductance models stay research. |
| L4 | braintrace stays a dev-only dependency until K3: K1L-0 does **not** declare it in `pyproject.toml`. Overrides draft Q13(a). K1b-0 raises the dev floor to 0.2.7 and brainevent to 0.3.0. | `brainspike/__init__.py` does not import `_kernel`; no user needs it before K3. |
| L5 | braintrace#184 `_keep_dependency` removal is decided by evidence in K1L-0: build the exactness gates first, then keep the variant that matches BPTT. | The audit found a 1.8–8 % change of online gradients on the recurrent G7 net, with no test pinning them and an inconclusive harness. |
| L6 | K1L-5 hard-depends on K1b-7 (plastic weights in edge order). | `w_slow` / `w_init` must live in the final plastic layout. |
| L7 | K1L-4 starts with a guard against the suspected relaxed + plasticity bug. | Under `spikes='relaxed'`, `x > 0` (`plasticity.py:539-540`) counts almost every step as an event. |
| L8 | At most 3 task agents at a time, shared with K1b. | Local WSL memory. |

**Defaults** (each: *default per draft recommendation; lead may override*). Q numbers
are those of the K1-L draft (`/tmp/k0/k1b/k1l-plan-draft.md` §6).

| Q | task | decision | reason |
|---|---|---|---|
| Q1 | all | Public shape `X(program, ..., optimizer).grad/run/fit(state0, data, targets, *, duration, n0, trials)` on a private `_EpisodeLearner` base (validation + capability checks, new `Capabilities.adjoint`). | Consistent with `BPTT`/`Online`; one place for checks. |
| Q2 | 4 | Rules gate with `ctx.gate(spike, new, old)`: hard = `where(s>0,new,old) + (s − sg(s))·(new − old)`; relaxed = `old + s·(new − old)`. | Exact in both modes; NEST forward unchanged. |
| Q3 | 5 | `connect(..., trainable=True, hybrid=Hybrid('decomposed' \| 'through_rule', alpha=, train=))`; through-the-rule writes a trainable `w_init` into `state0` when `reset_plastic=True`. | One object holds the mode; fits trial-structured meta-learning. |
| Q4 | 9 | New model (placeholder `lif_psc_exp_ep`); crossing linear by default plus `'root'`; rest as L3. | L2; the K1L-9 scope (L3) matches. |
| Q6 | 4 | brainevent 0.3.0 differentiable fast kernels for `FastForm` rules under grad, adopted only after a value-and-gradient equality test; reference path as fallback. Own adjoint over the rule trajectory is research. | Speed where proven; no `csr_apply_on_*` (no gain, different semantics). |
| Q7 | 6 | Generic `Program.vmap_rollout` primitive plus a thin `ES` estimator; `rng='common'` default; primitive shared with K1c sweeps. | Owned once. |
| Q8 | 7 | Offline ridge `Readout` from observer features (with `write` into a projection); FORCE with feedback through a stateless `Graded` relay. | A relay with no dynamics is not a rate neuron (S1a). If the lead reads S1a otherwise: RLS without feedback only. |
| Q9 | 2 | Batched `Online` = minibatch: shared weights, per-trial state and traces, mean gradient (braintrace `vmap=True`). Independent learners go through `vmap_rollout`. | Consistent with `Runner` batching. |
| Q10 | 2 | Learn phase on batched runs: shared params; `learn` sees the batched state (scan outside, vmap inside). | Same as Q9. |
| Q11 | 3 | `signal_generator` device and `ExternalSignal` relay, read with one step of latency by `Online(..., modulator=(pop, port))`; `modulatory` port kind deferred to K2. | braintrace reads the modulator at the top of `update()`. |
| Q12 | 0 | "OSTL exact on feed-forward" = single trainable layer, gated with `OSTLRecurrent`; two-layer first-layer error reported. | Multi-layer is not exact for D-RTRL by construction. |
| Q14 | 5 | `Online` keeps rejecting hybrid weights; decomposed + online is a K2 candidate. | L4 text. |
| Q15 | 11 | BPTT memory at scale: report only. | Set a gate in K2 from the numbers. |

(Q5 and Q13 are overridden by L1 and L4.)

## Step plan

### K1L-0 — exactness gates, checker update, #184 by evidence

**Contract.** New `learn/etrace_quality_test.py` with gates G6 and G7 below, validated
first on cases with a known answer (single element-wise layer, where D-RTRL equals BPTT;
the feed-forward net where both #184 variants are bit-identical). Then run both
variants (with and without `_keep_dependency`: `model.py:120,152`, `runner.py:643`,
`learn/etrace.py:344,510,517`, doc `models/_common.py:98`) on the recurrent G7 net
against BPTT and keep the one that meets G7; if neither does, escalate with numbers.
Add a test that pins online gradient values (none exists; `learn/etrace_test.py:64`
only checks non-zero). If removed, rule 5 (`learn/etrace.py:280`) goes too. Rule 3
(`:296`) narrows to `learning_signal='modulatory'` and braintrace's
`OUTPUT_BYPASS_DROPPED` diagnostic (made an `EtraceError`); rule 2 (`:249`) allows
weight-free `while`, still rejects ETP weights in `while`; rule 4 (`:239`) keeps the
rejection with a short pointer. No `pyproject.toml` change (L4).

**Gate K1L-G6 (OSTL exact, feed-forward).** `ExternalInput → LIF` and `→ iaf_psc_alpha`,
one trainable layer, zero delay, Dense and CSR, T = 50, per-step loss, both spike modes
with matched `detach_reset`: `max|g_ostl − g_bptt| / max|g_bptt| ≤ 1e-6` (f64).
**Gate K1L-G7 (D-RTRL vs BPTT, one step).** T = 1 from a warm `state0` (200 steps,
traces reset), the K0 G7 nets (`LIF`, `iaf_cond_exp`, `iaf_psc_alpha`), Dense and CSR,
outputs with pre-reset spike reads and a readout weight: ≤ 1e-6 (f64), ≤ 1e-4 (f32).
Multi-step on element-wise single-layer cells: D-RTRL and EProp equal BPTT at the same
bound. The `EtraceNetwork` forward equals `Runner.cont` bit for bit.

**Depends on:** K1b-0 (floors).

### K1L-1 — per-step online updates

**Contract.** `Online.run(..., update='episode' | 'step', every=k)` (`learn/etrace.py:677`).
`'episode'` keeps K1a. `'step'`: one compiled scan accumulates the single-step online
gradient; every k global steps the optimizer steps and the accumulator clears; traces
are not reset; optimizer state is carried by the scan. Documented: traces use the
pre-update weights (standard online-RTRL approximation).
**Gate.** Equals a carried Python driver of k-step episodes (rtol 1e-12, f64, T not a
multiple of k, unaligned `n0`); `lr = 0` equals `'episode'`; one trace per length; compile
≤ 5 s on the G7 net.
**Depends on:** K1L-0.

### K1L-2 — batched learning

**Contract (Q9, Q10).** `Program.rollout(..., batched=False)` (`runner.py:690`):
`batched=True` scans outside and vmaps the step inside, so params stay shared; the
learn phase sees the batched state. `Runner(batch=B).cont(learn=...)` no longer raises
(`runner.py:921`); without `learn` the path is unchanged. `Online(..., batch=B)` through
`braintrace.compile(vmap=True)`; `EtraceNetwork.load` (`learn/etrace.py:458`) accepts batched states
(today rejected at `:477`).
**Gate.** No-op learn equals the K1a vmapped rollout (1e-12, f64); trial-mean learn
equals B unbatched rollouts (1e-12); batched `Online.grad` equals the mean of B
unbatched calls (1e-10); B = 8 step ≤ 1.5 × the vmapped rollout without learn.
**Depends on:** K1L-1.

### K1L-3 — three-factor learning with a modulator source

**Contract (Q11).** `signal_generator` (`Graded` output `m` from a per-step table) and
`ExternalSignal` (stateless relay fed from `data`); `Online(program, 'three_factor', ...,
modulator=('reward', 'm'))` reads step n − 1's output (one-step latency). Scalar
populations broadcast; `(n,)` must match the trained layer's hidden shape. The modulator
replaces `dL/dh` (pass `−reward` to maximise). Rule 3 stays enforced for `three_factor`.
**Gate K1L-G2.** Per-step gradient = `expand(m_{t−1}) ⊙ e_t` (rtol 1e-8, f64);
`m_t := dL^t/dh^t` fed back reproduces `D_RTRL` (rtol 1e-8); LIF G7 net, Dense and CSR;
reward smoke task +0.3 mean reward in 200 episodes (proposed).
**Depends on:** K1L-0 (soft: K1L-1).

### K1L-4 — differentiable plasticity events and learned STDP

**Contract.** Step 1, guard: a test showing the relaxed + plasticity bug
(`plasticity.py:539-540`, `:572`), and `Program(spikes='relaxed')` raises on plastic
projections until step 2 lands. Step 2: rules gate with `ctx.gate` (Q2); trace
increments and lazy commit use it; multiplicity > 1 counts once for the trace (NEST).
Hard-mode forward stays bit-identical. Step 3: `FastForm` rules with trainable params
may keep `'fast'` (`plasticity.py:453`, Q6) only after a value + gradient equality test.
Scope: CSR, scalar delay (today's layout); land before K1b-7, which rebases. Note: two
rule-param gradient tests fail on GPU 1 with `gpusparseCreateCsr` (K1b-0 follow-up);
GPU rows of this task wait for that.
**Gate K1L-G3.** Relaxed recurrent net, post spikes from the network, T = 80 ms:
BPTT gradient w.r.t. `lambda_`, `alpha`, `tau_plus`, `tau_minus`, `Wmax` vs central FD
rel ≤ 1e-4 (f64). Hard mode finite and non-zero. G8 NEST parity unchanged. Saturated
relaxed spikes reproduce the hard trajectory. Meta-learning smoke −50 % in 100 steps
(proposed).
**Depends on:** K1a.

### K1L-5 — hybrid weights (L4)

**Contract (Q3, Q14).** `Hybrid('decomposed')`: `w_eff = w_slow + alpha · w_fast`
(`w_slow`, `alpha` trainable in edge order; `w_fast` edge state from 0). `Hybrid('through_rule')`:
one rule-mutated weight with trainable `w_init`, written into `state0` by
`BPTT(..., reset_plastic=True)`. Lifts `network.py:550`. `Plan.describe()` shows the
mode; `Online` rejects both.
**Gate (part of K1L-G3).** Both modes vs FD (relaxed, f64, rel ≤ 1e-4); gradient exactly
0 at a clip bound; `w_slow` round-trips through `get_connections` / `set('weight')`.
**Depends on:** **K1b-7 (hard)**, K1L-4.

### K1L-6 — evolution strategies

**Contract (Q7).** `program.vmap_rollout(params_P, state0, n0, trials_P, n_steps, *, chunk=64)`
(`jax.lax.map(batch_size=chunk)`, works inside `program.bound`), and
`ES(program, loss, optimizer, *, sigma, popsize, antithetic=True, params=None, rng='common', chunk)`.
**Gate K1L-G4.** `vmap_rollout` equals a loop over members (1e-12, f64) on Dense, CSR
and `DelayScatter` (first check of brainevent batching over batched weights; a slow
fallback is escalated); `chunk` invariance; cosine ≥ 0.95 vs relaxed BPTT on the G6 net
(popsize 8192, σ = 1e-3); hard spike-count task −50 % in 100 iterations; peak memory
at popsize 1024 ≤ 1.2 × popsize 64 (proposed).
**Depends on:** K1a.

### K1L-7 — ridge and FORCE readouts

**Contract (Q8).** `Readout(program, features=, filter=, method='ridge', alpha=, bias=)`
solved by Cholesky in f64 from observer features; `write` into a trainable projection.
`FORCE(program, readout=, rates=, target=, alpha=, every=)` runs RLS in the learn phase
with `P` carried by the scan; feedback through a stateless `Graded` relay.
**Gate K1L-G5.** Ridge vs `numpy.linalg.solve` ≤ 1e-8; RLS step vs NumPy ≤ 1e-9; 500-LIF
reservoir sine task NRMSE ≤ 0.1 (ridge), feedback FORCE ≤ 0.2 (proposed).
**Depends on:** K1a.

### K1L-8 — spike-time gradients through timed delivery

**Contract.** Losses in `learn/bptt.py`: `spike_times`, `first_spike_time`,
`spike_time_mse` (defined penalty for missing spikes). No stop-gradient on offsets or
weights in `TimedRing` reads (`timed.py:64`), `TimedDelivery.deliver` (`:137`),
`TimedEventPayload` pack/merge (`ports.py:468`), `accumulate` (`timed.py:300`). Zero-gradient cases documented and tested as zero
(event-port delivery, `grid_to_timed`, overflow drops, integer delays). Builds on the
implicit-function offsets (`models/precise.py:394`, tested at `models/precise_test.py:318`)
and `timed_test.py:261`.
**Gate.** `spike_generator_ps → iaf_psc_exp_ps → iaf_psc_exp_ps` (scalar and per-edge
delays): output spike-time gradients w.r.t. both layers' weights, `I_e`, `tau_m` vs
central FD rel ≤ 1e-5 (f64), 10 seeds, ≥ 8 valid points (FD bracket keeps counts and
in-step order); f32 ≤ 1e-3; vmap equals unbatched.
**Depends on:** K1a (re-run after K1b-8).

### K1L-9 — EventProp model (forward)

**Contract (L3, Q4).** `lif_psc_exp_ep` (placeholder name): states `V`, `I_syn`,
`spike_offset`; input `syn=TimedEvent(K, into='I_syn')`, optional `I=Current`; output
`spike=TimedEvent(1)`; params `tau_m, tau_s, C_m, E_L, V_th, V_reset, I_e`; no
refractoriness. Exact integration between sorted arrivals (reusing the `precise.py`
segment scan); crossing by linear interpolation per segment (`crossing='root'`:
Illinois + implicit-function correction); hard reset at `t*`; a second crossing in one
step increments `overflow` and is not emitted. `Capabilities(..., adjoint=True)` (new
field, `model.py:271`); `adjoint_spec()` gives the propagator, `g = V − V_th`, the reset
map and the input jump `I_syn += w`.
**Gate.** Constant-drive crossing vs analytic first passage (`'root'` ≤ 1e-9 ms; `'linear'`
O(h²) by a convergence test); counts equal an event-driven NumPy reference as dt → 0;
`jax.grad` of spike times vs FD rel ≤ 1e-5; relaxed mode rejected.
**Depends on:** K1L-8 (soft: K1b-6 crossing contract).

### K1L-10 — EventProp learner (continuous adjoint)

**Contract (L1, L2).** `EventProp(program, loss, optimizer=None)` with
`.grad/.run(state0, data, targets, *, duration, n0)`; forces `spikes='hard'`,
`detach_reset=False`. Validation: trainable populations have `adjoint=True`; their
projections have one scalar delay ≥ 1 step, Dense or CSR, no synapse model, no
plasticity; constant sources; losses read spike-time observers and optionally `V` at
observer steps; overflow > 0 in an episode is an error. Backward = the Wunderlich–Pehle
continuous adjoint, as in eventax (`/mnt/d/codes/githubs/snn/eventax/src/eventax/adjoint.py:260-279`)
but on our grid: a `jax.custom_vjp` over the episode; a reverse `scan` over steps carries
`(λ_V, λ_I)`, integrated in closed form between events (linear dynamics); at each
interpolated spike time `t*` the jump `ν = −(dL/dt* + λ·f) / (∂g/∂t + ∂g/∂y·f)` with the
reset map and a clamped `V̇(t*)` (counted in diagnostics); weight gradients `dL/dw_ij`
accumulated from `λ_I,j` at arrival times through the storage's transposed product;
homogeneous delays as an adjoint ring of length d. Residuals: spike offsets and the
state at `t*`; `V` only where the loss reads it.
**Gate K1L-G1 (approximate; numbers confirmed at a measurement checkpoint).** On a
feed-forward 2-layer (20-30-5) and a recurrent 30-neuron net, d ∈ {1, 5}, TTFS,
all-spike-time and V-at-time losses, f64: EventProp vs autodiff of the K1L-9 forward and
vs central FD within rtol ≈ 1e-3 (eventax level), at 20 seeds with ≥ 15 valid points
(FD bracket keeps counts and in-step order; rejected points reported). The error vs
step size is measured and reported first; the lead then fixes the numbers. Residual
memory vs BPTT reported.
**Depends on:** K1L-9.

### K1L-11 — gates and report

`K1L_GATES` in `validation/gates.py`, markers `k1l_gate`, `k1l_g1` … `k1l_g8`; CI
`kernel-gates.yml` adds `k1l_gate`. **K1L-G8 (regressions):** `-m "k0_gate or k1a_gate"`
green. Report only: BPTT memory at scale (Brunel 2500 and microcircuit 100 ms,
`checkpoint=None | ⌈√T⌉ | truncation`, CPU and GPU 1, f32; cf. `learn/bptt.py:453`) and
EventProp vs BPTT memory. Report `docs/specs/<date>-k1l-report.md`.
**Depends on:** all K1-L tasks.

### Waves (≤ 3 task agents at a time, shared with K1b)

| wave | tasks | notes |
|---|---|---|
| A | K1L-0, K1L-4, K1L-8, then K1L-6, K1L-7 as slots free | after K1b-0. K1L-0: `learn/etrace.py`, `model.py`, `runner.py:643`; K1L-4: `plasticity.py` (before K1b-7); K1L-8: `timed.py`, `learn/bptt.py`; K1L-6: new `learn/es.py`, `runner.py`; K1L-7: new `learn/readout.py` |
| B | K1L-1, K1L-3 (after K1L-0); K1L-9 (after K1L-8) | K1L-1 before K1L-3 (both edit `Online`) |
| C | K1L-2 (after K1L-1), K1L-10 (after K1L-9), K1L-5 (after K1b-7 and K1L-4) | |
| D | K1L-11 | measures the merged tree |

## How to verify

- Each task: its gate above, remotely (`/tmp/k0/remote.sh <task> '...'`), plus
  `-m "k0_gate or k1a_gate"` and the full `_kernel` suite with
  `--cov=brainspike/_kernel` (> 90 %); GPU rows on GPU 1; the PR body records the
  command and result.
- K1L-0 PR states which #184 variant was kept, with the G7 numbers of both.
- K1L-10 stops at a draft PR with the measured error-vs-dt table for the lead before
  the G1 numbers are fixed.
- End: `-m "k0_gate or k1a_gate or k1l_gate"` green remotely; the K1-L report lists
  every gate number.

## Decision log (task agents)

| task | decision | rationale | date |
|---|---|---|---|
| — | — | — | — |
