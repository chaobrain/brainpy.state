# K1a — kernel hardening: specification

_Date: 2026-10-22_ · Status: **authored; awaiting lead review** ·
Parent: [`2026-10-21-architecture-review-and-redesign.md`](2026-10-21-architecture-review-and-redesign.md)
(revision 2026-10-22: §6 principle 7, §7 L1/L2/L3, §7.2, §7.3, §8.2, §11 D-a…D-g, §12 K1a row) ·
Predecessor: [K0 spec](2026-10-21-k0-kernel-spec.md), [K0 report](2026-10-22-k0-report.md) (GO) ·
Upstream workarounds: [`2026-10-22-upstream-issues-todo.md`](2026-10-22-upstream-issues-todo.md).

K1a fixes what K0 showed to be structurally weak **before** K1b (substrate completion),
K1-L (learning track) and K2 (model library) build on it. It adds no new model family
and no public export (D-d: `_kernel` stays private until K3). The old kernel is frozen
(D-f): K1a does not touch it.

---

## 1. Context

K0 passed all ten gates (GO). A post-K0 architecture review (2026-10-22, evidence
scripts in `/tmp/k0/review/`, measured on a local CPU against `main` @ 7938574) found
contract gaps and structural costs that would multiply with every model ported in K2.
The lead decided D-a…D-g (architecture doc §11). The findings this spec acts on:

| # | finding | evidence | task |
|---|---|---|---|
| 1 | RK substeps are unrolled in Python; compile time grows with the substep count | `_kernel/integrators.py:243`; `aeif_cond_alpha` compiles in 17.3 s at k = 16; a `lax.scan` prototype gives identical traces and 1.7 s at k = 16 and k = 64, with identical braintrace D-RTRL gradients on an `iaf_cond_exp` net | K1a-1 |
| 2 | the substep hook mutates a Python side channel and branches on a Python int | `s_holder[0]` (`_kernel/model.py:552`); `rule.when == 'substep' or k == n_sub - 1` (`model.py:550`) | K1a-1 |
| 3 | precise-model in-step segments are unrolled (`capacity + 2` per step) | `_kernel/models/precise.py:17,347`; 100 neurons 4.8 ms/step; capacity 64 did not compile within 10 min | K1a-1 |
| 4 | every `Current`/`Graded` port is latched; a zero-delay chain through `LIF.I` has one step of latency per layer | `_LATCHED = (Current, Graded)` (`_kernel/model.py:94,508-511`); 4-layer zero-delay chain, first-spike steps: `event` port 9, 9, 9, 9; Expon → `LIF.I` 10, 11, 12, 13 | K1a-2 (D-a) |
| 5 | two encodings for neuron-owned filtering; the architecture example raises `PortError` | `iaf_psc_alpha` `ex=Event(unit=pA)` vs `iaf_cond_exp` `Conductance(filtered=True)` + special case `_kernel/ports.py:495-496` | K1a-3 (D-b) |
| 6 | one storage per distinct delay value | `_kernel/compiler.py:154,606-616`; 2000 neurons, indegree 100: scalar 1 group, build 1.2 s, 1.06 ms/step; ≤ 2 ms 20 groups, 5.3 s, 2.9 ms/step; ≤ 5 ms 50 groups, 11 s, 5.4 ms/step; ≤ 20 ms 200 groups, 26 s, 4.2 ms/step | K1a-4 |
| 7 | learners attach to the `Runner`; BPTT episodes start at step 0; no learn phase | `_kernel/compiler.py:391` (`plan.runner(learner=)`); `_kernel/learn/bptt.py:4-6` | K1a-5 (D-c) |
| 8 | relaxed spikes are a module `ContextVar` read at trace time | `_kernel/spiking.py:89,444` | K1a-5 |
| 9 | the etrace checker rejects braintrace primitives in any `scan`; only `while` bodies fail in braintrace 0.2.6 | `_kernel/learn/etrace.py:239-244`; [braintrace#186](https://github.com/chaobrain/braintrace/issues/186) | K1a-5 |
| 10 | object keys come from insertion order; dynamics noise is drawn for whole populations | `population_key(..., gid_block=0)` (`_kernel/rng.py:196-202`); `uid` = insertion index | K1a-6 (D-g) |
| 11 | GPU unmeasured; no network above order 2500; G2 ran a regular (CV 0.083) regime; no CI runs the gate marker or Brian2 | K0 report §1, §3.1; `.github/workflows/` | K1a-7 |

## 2. Scope

**In:** K1a-1 … K1a-8 below (one PR each; K1a-8 added from the K1a-7 G3 escalation), the CI job of v0.1.1 goal 02 that serves
the new kernel (D-f), and the K1a report.

**Out (K1b / K1-L / later):** in-substep spike localisation (`SpikeRule` contract,
K1b); `pre`-aligned `SynapseModel` lowering of NMDA (K1b); plastic weights in edge
order and per-edge delays on plastic projections (K1b); `timed_event` capacity as a
statistical budget (K1b — K1a-1 only turns the segment loop into a `scan`); merge-pass
veto (K1b); recording at scale (K1b); EventProp, three-factor, hybrid weights, ES,
readouts (K1-L); any change to the old kernel (D-f); any public export (D-d).

**Invariants for every task.** All K0 gates (G1–G8, G10; `-m k0_gate`) stay green on
the remote host; coverage of `_kernel` stays > 90 %; NEST parity of the K0 NEST models
is unchanged (NEST models keep their semantics; D-a keeps them latched).

---

## 3. Tasks and contracts

### K1a-1 — substeps and precise segments as `scan`; compile-time gate

**Contract.**

- `integrate(integrator, f, y0, t0, dt, hook, aux)` runs the substeps as **one
  `lax.scan`** (via `brainstate.transform.scan` where states are involved) over a
  static length. The carry holds **every** model state the hook can change — float
  states, integer/boolean states (refractory counters, flags) and the `SpikeRule`
  carry — explicitly. No Python object is mutated inside the loop body (remove
  `s_holder`).
- Per-substep vs per-step spike checks are expressed without a Python int on the loop
  index: for `rule.when == 'step'`, scan `k - 1` substeps without the check and peel
  the final substep with the check (or an equivalent traced form that keeps the cost
  of the per-step rule unchanged).
- The `iaf_psc_exp_ps` in-step segment loop (`capacity + 2` segments) is one `scan`
  over segments; the root finder keeps its fixed iteration count inside the body.
- Port inputs stay computed once per step, outside the scan (braintrace rule); the
  `_keep_dependency` workaround ([braintrace#184](https://github.com/chaobrain/braintrace/issues/184))
  stays and gets its `# upstream:` comment.

**Acceptance tests.**

- Traces of `iaf_cond_exp` and `aeif_cond_alpha` (k ∈ {1, 2, 4, 16, 64}) and of
  `iaf_psc_exp_ps` equal the K0 implementation ("numerically unchanged", lead decision
  2026-10-22: float64 max relative difference ≤ 1e-12; float32 within `CAT_B`), including spike counts per step.
  Exception for `iaf_psc_exp_ps` in float32 (lead decision 2026-10-22, §7): spike counts
  per step identical, `V_m` and `I_syn` within relative 1e-5, spike offsets within
  1e-4 ms; float64 keeps ≤ 1e-12 and NEST parity is unchanged.
- braintrace D-RTRL and e-prop gradients on the K0 `iaf_cond_exp` and `LIF` nets equal
  the K0 values (same bounds).
- BPTT G6 tests green.
- Edge cases: k = 1 (scan of length 0 + peeled step); refractory period shorter than
  one substep; several spikes in one step (`aeif`); batched (`vmap`) runs; zero-size
  population.

**Gate (compile time).** Every K0 network compiles in **≤ 5 s** at the default
substeps on the remote host CPU (`JAX_PLATFORMS=cpu`, single process, compile time =
first-call wall time minus a warm call): Brunel `iaf_psc_alpha` order 2500, COBA
`iaf_cond_exp` 4000, the `aeif_cond_alpha` network of G5, the `iaf_psc_exp_ps` +
`poisson_generator_ps` network of G1 at capacity 64, the NMDA network of T8, the
`stdp_synapse` network of G8, the `LIF` BPTT network of G6. Baseline to beat (review,
local CPU): `aeif_cond_alpha` 17.3 s → 1.7 s at k = 16. A test (`k1a_compile` marker)
records the table; the measured numbers go into the K1a report.

### K1a-2 — per-port latch (D-a)

**Contract.**

- `Current(unit, latch=True)` and `Graded(unit, latch=True)` declare whether the input
  acts in the next step (`latch=True`, NEST `CurrentEvent`) or in the step it arrives
  in (`latch=False`). **`latch` defaults to `True`** (K0 behaviour; lead decision
  2026-10-22): a port is NEST-style latched unless it opts out with `latch=False`.
  Only latched ports get a latch state (`latch_<port>`).
- Unlatched inputs of step `n` enter step `n`'s update: as a constant current over the
  step for RK/exp-Euler models (the same value at every substep), and through the
  propagator for exact models.
- K0 NEST models (`iaf_psc_alpha`, `iaf_cond_exp`, `aeif_cond_alpha`,
  `iaf_psc_exp_ps`) keep the default `latch=True` (may state it explicitly); the
  BrainPy-family `LIF` **must declare `latch=False` explicitly** (a test checks it). Devices feeding `current` ports are unaffected (the receiving port
  decides).
- Architecture doc §7 L2 (D4a as revised) is the semantic reference: a zero delay is
  same-step for `event`, `conductance` and unlatched ports; a latched port adds one
  step by port semantics.
- Schedule (as implemented, §7): zero-delay edges into same-step ports are hard
  ordering constraints (a cycle through them is a `ScheduleError`); zero-delay edges
  into latched ports keep the K0 order when the whole graph is acyclic, and otherwise
  the target port is **deferred** (`Plan.deferred`): its inputs are gathered after all
  updates of the step and only written into the latch, so a latched port breaks a
  zero-delay cycle with the same numbers as the source-first order.
  `Plan.describe()` shows the latch of every input port and the deferred ports.

**Acceptance tests.**

- **Zero-delay chain:** a 4-layer feed-forward chain, Expon → `LIF.I`, all delays
  zero, driven by one input spike train: the first spike of every layer is in the same
  step (today 10, 11, 12, 13). The same chain through `event` ports keeps 9, 9, 9, 9.
  The same chain through a latched port shows one step per layer (documented
  behaviour).
- NEST parity unchanged: G1 (`current` DC onset), G2, G3, G5, G8 green with the same
  numbers (bit-identical for latched models).
- `LIF` G6/G7 tests green after the change (the `LIF` step changes semantics; this
  is intended).
- Edge cases: unlatched port in a zero-delay cycle (still a `ScheduleError`); unlatched
  port behind a delay ≥ 1; external input (`ext`) into latched and unlatched ports;
  batched runs; `Graded` ports both ways.

### K1a-3 — one receptor-port encoding (D-b)

**Contract.**

- A neuron-owned receptor port is an `Event` input port that **names the state it
  filters into** with the keyword **`into=`** (lead decision 2026-10-22), e.g.
  `ex=Event(unit=u.nS, into='g_ex')`; `TimedEvent` receptor ports use the same keyword. A delivered weight (unit = the named state's unit per
  event) jumps that state after propagation; the neuron's `derivatives` filter it.
- `filtered=` is removed **everywhere** (`Current`, `Conductance`, `Graded`,
  `TimedEvent`; and the special case at
  `_kernel/ports.py:495-496`). Projection-owned kinetics (`synapse=Expon(...)`, …)
  connect only to unfiltered `Current`/`Conductance` ports; attaching a `SynapseModel`
  to a receptor `Event` port raises `PortError` naming the owner of the kinetics.
- Migrate `iaf_psc_alpha` (state `i_syn_ex`/`i_syn_in` path), `iaf_cond_exp` and
  `aeif_cond_alpha` (`g_ex`, `g_in`).

**Acceptance tests.**

- The architecture doc §7 L3 example is a test: the corrected example compiles and
  runs; the previous example (`synapse=Expon` into `iaf_cond_exp.ex`) raises
  `PortError`.
- NEST parity unchanged for the three models (G1, G3, G5 green, same numbers).
- Edge cases: weight unit mismatch on a receptor port (error at `connect`); negative
  weights on an excitatory receptor port with a signed distribution (validation per
  §7 L1); a source with `max_multiplicity > 1` into a receptor port (multiplicity
  carried); a receptor port naming an undeclared state (error at class creation).

### K1a-4 — single-op per-edge delays

**Contract.**

- Post-side delivery of a projection with per-edge delays is **one** operation per
  step: per-edge contributions `payload[pre_e] · w_e` are scatter-added into the
  arrival ring at `((n + d_e) mod L, post_e)`; the step reads and clears slot
  `n mod L`. Gradients with respect to weights and payload are exact (JAX VJP of the
  scatter).
- The compiler chooses between the scatter and K0's per-delay grouping with a **cost
  model** (per-step cost of the scatter ∝ edges; of grouping ∝ groups × per-group
  matmul cost; plus build time), caps the number of groups, and reports the choice and
  its memory in `plan.describe()`. The number of groups has a **hard ceiling recorded
  in the plan**; the cost-model constants come from this task's measurements.
- [brainevent#203](https://github.com/chaobrain/brainevent/issues/203) (fused
  per-edge-delay delivery) is the future fast path behind the same cost model.
- Plastic projections keep their K0 scalar-delay restriction (K1b).

**Acceptance tests.**

- Delivered input equals the K0 grouped implementation (counts exact; summed weights
  within `CAT_B`) for delays drawn per edge, including delay 0 on acyclic paths,
  multapses with different delays, and a delay equal to the ring length.
- BPTT FD check (G6 regime a) through a per-edge-delay projection.
- `get_connections()` / `set('weight')` still exact (edge order).

**Gate (per-edge delays).** Brunel with per-edge delays (2000 neurons, indegree 100,
delays uniform up to 20 ms — the review's worst case) on the remote CPU: **step time
≤ 1.5 × the same network with a scalar delay**, and **build time ≤ 2 × the scalar-delay build
time** (lead decision 2026-10-22).
Measured with `BENCH=1` or the interleaved protocol.

### K1a-5 — learner API (D-c), explicit spike mode, relaxed checker, learn phase

**Contract.**

- `Program(plan, spikes='hard'|'relaxed', ...)` is the L2 contract (architecture doc
  §7 L2): pure `init_state`, `read_params`, `step`, `rollout`. `spikes='hard'`
  (default) emits exact counts with a surrogate backward; `'relaxed'` emits smooth
  spikes with soft jumps and a non-detached reset. The `relaxed_spikes()` `ContextVar`
  (`_kernel/spiking.py:89,444`) is removed; the spike mode is a static field of the
  `Program`.
- Learners take **`(Program, state0)`** (D-c): `BPTT`, `Online` (and later learners) are
  built from a `Program` and run episodes from any `state0` (initial, warmed-up,
  checkpointed, or a `Runner`'s current state). The `Runner` is a client of a `Program`
  like the learners; `plan.runner(learner=…)` and the `attach(runner)` coupling are
  removed. Shape (lead decision 2026-10-22): learner objects,
  `BPTT(program, loss, optimizer, ...).run(state0, data)`; the exact signatures are
  shown in the K1a-5 PR and approved by the lead before merge.
- A **learn phase** in the rollout: an optional function applied inside the compiled
  loop at a chosen interval (architecture doc §8.2-6), as
  `Program.rollout(..., learn=fn, every=k)` (lead decision 2026-10-22; exact signature
  approved in the K1a-5 PR).
- The etrace checker's rule 2 allows braintrace primitives inside static
  `scan`/`fori_loop` bodies and rejects them only inside `while` bodies
  ([braintrace#186](https://github.com/chaobrain/braintrace/issues/186)); rules 2–4
  get their `# upstream:` comments.

**Acceptance tests.**

- G6 and G7 green on the new API (both spike modes).
- BPTT from a non-zero `state0` (after 200 warm-up steps): relative error vs central FD
  ≤ 1e-4 in the G6 regimes; the episode's forward trace equals `Runner.cont()` from the
  same state.
- `Program(plan, spikes='relaxed')` and `'hard'` coexist in one process (no global
  state); jit caches do not leak the mode between programs.
- The relaxed checker accepts a model with a braintrace weight inside a static `scan`
  (gradients equal to the unrolled loop and to BPTT) and rejects one inside a `while`
  body with a clear error.
- Learn phase: an optimizer step every `k` steps inside one compiled rollout equals the
  same updates applied between `cont()` calls.
- Edge cases: `state0` from a batched runner; learner on a plan with no trainable
  parameters (clear error); `spikes='relaxed'` on a model without a surrogate (clear
  error).

### K1a-6 — RNG layout (D-g)

**Contract.**

- Object keys are derived from a stable hash of the object's **name** (populations,
  projections, devices), not from its insertion index; names are unique within a
  `Network` (duplicate → error). Unnamed objects are auto-named `<model>_<k>`, counted
  per model class (lead decision 2026-10-22), so insertion order matters only among
  unnamed objects of the same class — documented in the docstrings.
- Per-step dynamics noise (Poisson generators, noise inputs, stochastic spiking) is
  drawn **per gid block** of 1024 gids (the same `GID_BLOCK` constant as connectivity,
  `_kernel/rng.py:61`; lead decision 2026-10-22): the draws for gids in block `b` come from
  `block_key(object_key, b)` folded with the step (and trial). Connectivity keeps its
  per-(projection, post gid block) rule with the new object keys. The distributed-layout
  spec (K1b) builds on this block size.
- This changes every realised stream relative to K0 (decided by D-g); statistical gates
  are re-run, exact-equality gates are re-baselined.

**Acceptance tests.**

- **Insertion-order invariance:** adding, removing or reordering an unrelated
  population or projection leaves every other object's realised connectivity and noise
  identical (exact equality).
- **Shard invariance (simulated):** drawing a population's noise as one array or as
  `k` contiguous block-aligned pieces (as a sharded runtime would) gives identical
  numbers; non-aligned split points are rejected or handled (documented).
- G4 invariances (storage, batch, RNG modes) still exact; `cont()` continues streams;
  G2/G3 statistics pass with the new streams.
- **Gate:** Brunel order 2500 step time ≤ 1.05 × K0 (lead decision 2026-10-22).
- Edge cases: population smaller than one block; size not a multiple of the block;
  renamed object (new streams, by design); name collisions; `vmap` common vs
  independent.

### K1a-7 — gates, benchmarks, CI, K1a report

**Benchmarks and gates** (remote host; GPU 1 only, idle check; `BENCH=1` or the
interleaved protocol of `benchmarks/g9_interleaved.py`):

| gate | measurement | criterion |
|---|---|---|
| K1a-G1 compile | compile time of every K0 network at default substeps (K1a-1 table), CPU | ≤ 5 s each |
| K1a-G2 per-edge delays | K1a-4 gate | step ≤ 1.5 × scalar-delay; build within the approved bound |
| K1a-G3 GPU G9 | Brunel order 200 and 2500 on GPU 1: float32 with the default brainevent backend, float64 with `cusparse` ([brainevent#199](https://github.com/chaobrain/brainevent/issues/199)); compared with the T0 GPU baseline rows of the old `Simulator` | step time and peak device memory ≤ 1.1 × old (the K0 G9 rule); compile time reported |
| K1a-G4 microcircuit | Potjans–Diesmann microcircuit at full scale (77 169 neurons, ≈ 3 × 10⁸ synapses), CPU and GPU 1, float32: memory, build, compile and step time; per-population rates vs NEST 3.10, **5 seeds each side, 1 s after a 100 ms transient** | rates by the G2 rule (\|Δmean\| ≤ max(5 % of NEST mean, 2 × combined SEM)); memory/build/compile/step reported |
| K1a-G5 G2 in the AI regime | Brunel `iaf_psc_alpha` at a parameter point in the asynchronous-irregular regime, chosen by a short NEST-only scan requiring CV-ISI ≥ 0.5 and **shown to the lead before the kernel comparison runs**; 10 seeds each side | the G2 rule on rate, CV-ISI, Fano factor and pairwise correlation |
| K1a-G6 regressions | all K0 gates (`-m k0_gate`) | green |

**CI (the D-f remainder of v0.1.1 goal 02).** A GitHub Actions job that installs the
package with the `validation` extra (`brian2`) and runs `pytest -m "k0_gate or
k1a_gate" brainpy_state/_kernel` (NEST-marked tests skip without NEST); the existing
NEST workflow (`.github/workflows/nest-comparison.yml`) also runs the kernel's
`requires_nest` gate tests. A `k1a_gate` marker is registered next to `k0_gate`.

**K1a report** (`docs/specs/<date>-k1a-report.md`): gate table with numbers, protocol
and host, decisions taken, upstream status (from the TODO file), follow-ups for K1b /
K1-L.

### K1a-8 — host-resident EdgeSet (added 2026-10-22, from the K1a-7 G3 escalation)

**Problem.** The first K1a-G3 measurement (pre K1a-8) failed on peak device memory
(Brunel order 2500 float32: 0.374 GiB vs the T0 0.232 GiB, 1.61 ×). The realised
`EdgeSet` of every projection (local indices, gids, per-edge weights and delays;
≈ 298 MB at order 2500) stayed on the device next to the storage arrays (119 MB), and
compilation held further device copies of the edge-order weights and of the storage
permutation.

**Contract.** After `Network.compile`, a `Plan` keeps no realised edge arrays on the
device. Every array of an `EdgeSet` (built by `create` / `from_gids`, the rules,
`with_weight` / `with_delay` / `set`, `Plan.edges`, `Runner.edges`, `get_connections`)
is a host `numpy` array (Quantities carry numpy mantissas); float dtypes follow JAX's
canonicalisation, so the values equal what a device array held before. Storages
(`Dense`, `CSR`, `DelayScatter`) copy what the compiled program reads to the device;
`weight_from_edges` permutes host weights on the host and copies only the result.
`get_connections` and every introspection result are unchanged in values, dtype,
unit and order.

**Tests.** Host-residency of every `EdgeSet` field and introspection result; dtype
canonicalisation (float64 input without x64 stored as float32); subset `set` on host
values leaves the original unchanged; a compiled plan holds at most the storage and
weight-state device arrays of edge length (no `EdgeSet` copy); the full `_kernel`
suite and `-m k0_gate` remotely; mypy clean.

**Gate.** Re-measure K1a-G3 with the same protocol and metric (raw
`peak_bytes_in_use`, ≤ 1.1 × T0); the metric is not changed.

### K1a-10 — float64 binary event payload (added 2026-10-23, from the K1a-7 G3 re-measure)

**Problem.** On GPU, brainevent 0.2.1 computes `BinaryArray(x) @ W` wrongly when `x`
is a float64 vector and `W` a dense matrix, and without an error
([brainevent#204](https://github.com/chaobrain/brainevent/issues/204)). bool and
float32 `x` give correct results, and so does the CPU. In float64 the kernel's binary
payload path passed float64 spikes, so GPU float64 runs with `Dense` storage were
wrong. For example, Brunel order 200 fired at 59.7 Hz, against 57.1 Hz with CSR and
with the old `Simulator`. The same float64 events make the default-backend CSR kernel
abort ([brainevent#199](https://github.com/chaobrain/brainevent/issues/199)).

**Contract.** On the binary payload path only, float64 event vectors are cast to
float32 before they reach brainevent. The values are 0/1, so the cast is exact. The
output keeps `W`'s dtype. `Dense` wraps the call in a `jax.custom_jvp` whose tangent
in the events is `dx @ W` in `W`'s dtype, so the event gradient stays in float64.
`CSR` uses the plain cast: its event VJP does not depend on the dtype of `x`. The
float payload path, float32 runs and the CPU are unchanged. Dense float64 on the CPU
is bit-identical to before.

**Tests.** `storage_test.py::TestFloat64BinaryPayload` covers:

- binary vs float path for Dense and CSR, 1-D and batched, float64 output dtype;
- Dense bit-identical to the uncast call on the CPU;
- value and weight/event gradients against the float path in float64 (1e-12);
- `jacfwd` vs `jacrev`.

The class runs on the CPU in the suite and on GPU 1 remotely. At network level on
GPU 1, Brunel ε = 0.1 at orders 200 and 800 in float64 gives the same rate with Dense
and with CSR, on both the default and the `cusparse` backend.

---

## 4. Dependency and wave plan

| wave | tasks | notes |
|---|---|---|
| A | K1a-1, K1a-4, K1a-5, K1a-6 in parallel | K1a-1 touches `integrators.py`, `model.py`, `models/precise.py`; K1a-4 `compiler.py`, `delays.py`, `storage.py`; K1a-5 `runner.py`, `learn/`, `spiking.py`, `compiler.py` (`runner()`); K1a-6 `rng.py`, `network.py`, `runner.py`, `models/devices.py`. Overlaps (`compiler.py`, `runner.py`) are small; later PRs rebase |
| B | K1a-2, then K1a-3 | both edit `ports.py`, `model.py` and the model files; K1a-2 after K1a-1 (same `model.py` step code) |
| C | K1a-7 | after all others; measures the merged kernel |
| C′ | K1a-8 | from the K1a-7 G3 escalation; separate PR merged before K1a-7, which re-measures G3 on it |
| C″ | K1a-10, then K1a-9 | K1a-10 (float64 binary payload, brainevent#204) is a separate PR merged before K1a-9 (small-network GPU footprint); K1a-7 rebases on both and re-measures G3 |

Each task: test-first (AGENTS.md #4), NumPy docstrings, > 90 % coverage of the modules
it touches, remote test run recorded in the PR, `# upstream:` comments per
[`2026-10-22-upstream-issues-todo.md`](2026-10-22-upstream-issues-todo.md) §1.1.

## 5. Risks specific to K1a

- **Semantic changes hidden as refactors.** K1a-1 must be numerically neutral; K1a-2,
  K1a-3 and K1a-6 change semantics on purpose. Mitigation: every PR states which
  numbers changed and why; NEST models are checked bit-identical where their semantics
  do not change.
- **Microcircuit cost.** Full-scale Potjans–Diesmann on NEST and the kernel, several
  seeds, may take hours on the shared host. Mitigation: fixed protocol (5 seeds each side,
  1 s after a 100 ms transient; lead decision 2026-10-22).
- **GPU availability.** GPU 1 may become busy again. Mitigation: the remote script
  aborts on a busy GPU; K1a-G3/G4 GPU rows wait rather than run on GPU 0.

---

## 6. Open questions for the lead (raised while writing this spec)

Each is a contract, semantics or gate choice that D-a…D-g do not settle. **All rows
were resolved by the lead on 2026-10-22**; the answers are in §7 and in the task
sections above.

| task | question | options | recommendation | status |
|---|---|---|---|---|
| K1a-1 | tolerance for "numerically neutral" `scan` substeps | (a) float64 max relative difference ≤ 1e-12 vs K0, float32 within `CAT_B`; (b) bit-identical | (a): XLA may reorder floating-point ops inside a `scan` | resolved (§7) |
| K1a-2 | default of `latch` on `Current`/`Graded` | (a) required keyword on model input ports (no default); (b) default `True` (K0 behaviour); (c) default `False` | (a): D-a says the latch is declared per port; a silent default hides a one-step semantic difference | resolved (§7) |
| K1a-3 | keyword naming the filtered state on a receptor `Event` port; fate of `filtered` on `TimedEvent`/`Graded` | `into=` / `state=` / `filters=`; `TimedEvent` gets the same keyword or keeps a flag | `into='g_ex'`; the same keyword on `TimedEvent` receptor ports (one encoding, as D-b intends) | resolved (§7) |
| K1a-4 | build-time bound of the per-edge-delay gate; group cap | build ≤ 2 × scalar-delay build, or an absolute bound (e.g. ≤ 3 s at 2000 × 100); cap fixed (e.g. 8) or derived from the cost model | ≤ 2 × scalar build; cap from the cost model, with a hard ceiling recorded in the plan | resolved (§7) |
| K1a-5 | learner and learn-phase signatures | (a) `BPTT(program, loss, optimizer, …).run(state0, data)`; (b) functional `bptt(program, state0, …)`; learn phase as `Program.rollout(..., learn=fn, every=k)` | (a), with the learn phase on `rollout`; proposed in the K1a-5 PR for approval | resolved (§7) |
| K1a-6 | gid block size for dynamics noise; default names of unnamed objects | block 1024 (= today's connectivity `GID_BLOCK`, `_kernel/rng.py:61`) or smaller; default names `<model>_<k>` counted per model class, or require names | block 1024 (one constant for connectivity and noise); default `<model>_<k>` per class (insertion order then matters only among same-class unnamed objects — documented) | resolved (§7) |
| K1a-6 | step-time bound for blocked draws | ≤ 1.05 × K0, or report only | ≤ 1.05 × K0 at Brunel order 2500 | resolved (§7) |
| K1a-7 | microcircuit protocol and AI-regime parameter point | seeds (e.g. 3 or 5 each side), duration (e.g. 1 s after a 100 ms transient); AI point from Brunel 2000 (e.g. `g = 5`, `η = 2` gave CV 0.083 at the NEST example's `J`, so a larger `g` / `η` is needed) | 5 seeds × 1 s for the microcircuit (cost-bound); AI point chosen by a short NEST-only scan, CV-ISI ≥ 0.5 required, approved before the kernel run | resolved (§7) |

## 7. Decisions

**Decision policy** (agent brief, updated 2026-10-22). Task agents decide and log only
internal structure, private helper names, file splits, test design, and tolerances
already fixed in this spec. **The lead decides** every choice that changes a contract
or user-facing API (ports, `NeuronModel`/`SynapseModel`/`PlasticityRule`,
`Network`/`Plan`/`Program`/`Runner`/learner signatures, observer/result formats) or
deviates from this spec; changes simulation semantics or numerics relative to
NEST/Brian2/the old kernel (timing, delays, RNG streams, integration, defaults);
changes a gate, criterion or tolerance, or finds a gate cannot be met; adds or removes
a dependency, exports anything publicly, removes a feature or deletes existing
non-`_kernel` code; or finds a premise of this spec false. For those, the task opens
its PR as a draft, reports options with a recommendation, and does not merge; the
orchestrator does not decide them. When unsure, escalate. Every decision — minor ones
too — is appended below (and stated in the PR body); lead decisions say so in the
decision column ("lead: …").

| task | decision | rationale | date |
|---|---|---|---|
| K1a-1 | lead: "numerically unchanged" = float64 max relative difference ≤ 1e-12 vs K0; float32 within `CAT_B` | lead decision | 2026-10-22 |
| K1a-1 | lead: option (a) — float32 criterion for `iaf_psc_exp_ps` vs K0: identical spike counts per step, `V_m` and `I_syn` within relative 1e-5, spike offsets within 1e-4 ms; float64 keeps ≤ 1e-12 (NEST parity unchanged) | lead decision on the PR #114 escalation (K0's own float32 precise result varies at the ulp level with the XLA fusion context; the root finder amplifies it) | 2026-10-22 |
| K1a-2 | lead: `latch` **defaults to `True`** (K0 behaviour; NEST-style one-step latch unless a port opts out with `latch=False`); the BrainPy-family `LIF` declares `latch=False` explicitly. (Differs from the spec author's recommendation of a required keyword.) | lead decision | 2026-10-22 |
| K1a-3 | lead: keyword `into=` (e.g. `ex=Event(unit=u.nS, into='g_ex')`); the same keyword on `TimedEvent` receptor ports; the `filtered` flag is removed everywhere | lead decision | 2026-10-22 |
| K1a-4 | lead: build time ≤ 2 × the scalar-delay build; step time ≤ 1.5 × scalar; the compiler picks scatter vs grouping by a cost model, with a hard ceiling on groups recorded in the plan | lead decision | 2026-10-22 |
| K1a-5 | lead: learner objects `BPTT(program, loss, optimizer, ...).run(state0, data)`; learn phase as `Program.rollout(..., learn=fn, every=k)`; exact signatures shown in the K1a-5 PR for lead approval before merge | lead decision | 2026-10-22 |
| K1a-6 | lead: gid block 1024 (same constant as connectivity); unnamed objects auto-named `<model>_<k>` per class (insertion order matters only among same-class unnamed objects — documented); gate: step time ≤ 1.05 × K0 at Brunel order 2500 | lead decision | 2026-10-22 |
| K1a-7 | lead: Potjans–Diesmann microcircuit with 5 seeds each side, 1 s after a 100 ms transient, CPU and GPU 1; AI Brunel point chosen by a short NEST-only scan requiring CV-ISI ≥ 0.5, shown to the lead before the kernel comparison runs | lead decision | 2026-10-22 |
| K1a-8 | lead: fix the K1a-G3 memory failure now and keep the metric (raw peak device memory, ≤ 1.1 × T0); separate PR (`k1a/K1a-8-edgeset-host`) merged before K1a-7, which then re-measures G3 with the same protocol; if order 200 still fails, investigate the kernel's share (Dense vs CSR, rollout temporaries) and report before merging K1a-7 — the metric is not changed | lead decision on the K1a-7 G3 escalation | 2026-10-22 |
| K1a-8 | host from creation (numpy everywhere in `EdgeSet`) instead of dropping and re-materialising on demand | simpler (no re-run of the rules, no cache), introspection results identical by construction, and it also removes the device copies held *during* compile, where the peak was | 2026-10-22 |
| K1a-8 | storages permute host per-edge weights on the host (`_take`) and copy only the native layout to the device; the compiler's per-edge weights stay on the host | the device copies of the edge-order weights and of the permutation set the compile-time peak (order 2500 f32: 287.5 MB with a host `EdgeSet` alone, 144.7 MB with host permutation) | 2026-10-22 |
| K1a-8 | on the CPU backend `numpy.asarray` of a device array is a zero-copy view, so host copies cost no extra RAM there; no explicit copy is forced | a forced copy would raise the CPU peak RSS by one weight and one delay array per projection during compile (≈ 2.4 GB on the full microcircuit) | 2026-10-22 |
| K1a-5 | lead decision 2026-10-22 (approved in PR #112): the spike mode and the program-level `detach_reset` reach the models as **static fields of `StepContext`** (`ctx.spikes`, `ctx.relaxed`, `ctx.detach_reset`; static pytree metadata, so they are part of the jit cache key); `SpikeRule.check(..., relaxed=False)` takes the mode explicitly; `relaxed_spikes()` / `relaxed_enabled()` are removed | removes the `ContextVar` (no global state, visible in signatures); two programs with different modes never share a trace | 2026-10-22 |
| K1a-5 | lead decision 2026-10-22 (approved in PR #112): `Program(plan, *, spikes='hard', detach_reset=None, etrace=False)` (keyword-only options), `Program.replace(**options)`, `Plan.program(...)`; `detach_reset` resolution: rule's own setting > program option > model's `detach_reset` > False | learners set `detach_reset` / `spikes` by deriving a program (architecture doc §8.2-2) instead of mutating the models (K0 BPTT wrote `m.detach_reset` on every model of the plan) | 2026-10-22 |
| K1a-5 | `spikes='relaxed'` uses the exact float payload path for every projection (the binary event path would binarise fractional spikes); construction fails with a list of populations whose spike rule has no surrogate with `surrogate_fun`, or that are precise-timing neurons; devices and relays are allowed | K0 tests set `pp.path = 'float'` by hand, which mutated the shared plan; the spec asks for a clear error on models without a surrogate | 2026-10-22 |
| K1a-5 | `Program` captures its initial state at construction (`replace` shares it); `init_state()` no longer reads the models' live `State` values | found while testing `Online` from any state: `EtraceNetwork` writes the models' states, after which K0's `init_state()` returned the current, not the initial, values | 2026-10-22 |
| K1a-5 | lead decision 2026-10-22 (approved in PR #112): `Runner(program_or_plan, batch=None, rng='independent')`; `Plan.runner(batch, rng)` loses `learner=`; `Runner.learner` and `learner.attach(runner)` are removed; new `Runner.trials`; `Program.init_state(batch=None)`, `Program.write_params(params)`, `trial_indices(batch, rng)` | D-c: the runner is just another client of the program | 2026-10-22 |
| K1a-5 | lead decision 2026-10-22 (approved in PR #112): `BPTT(program, loss, optimizer=None, *, checkpoint=None, truncation=None, detach_reset=True, spikes=None)` with `record / loss / grad / run / fit(state0=None, data=None, targets=None, *, duration=None, n0=0, trials=None)`; `run` returns `(loss, final_state)` (one optimizer step), `grad` returns `(loss, grads)`, `fit` takes `n_epochs=` keyword-only; `state0=None` is the program's initial state; the episode length comes from `data` (or `duration=`); a batched `state0` is given with `trials=` (e.g. `runner.trials`); validation (trainable parameters, observers, differentiability) happens at construction | lead decision D-c / K1a-5 shape; episodes from any state fix the step-0 limitation; `n0` keeps RNG streams and time aligned with `Runner.cont()` (tested equal) | 2026-10-22 |
| K1a-5 | lead decision 2026-10-22 (approved in PR #112): `Online(program, algorithm='eprop', *, inputs=None, outputs=None, loss=None, optimizer=None, **options)` with `load(state0=None, n0=0)`, `state()`, `grad(state0, data, targets, *, n0=0) -> (loss, grads)`, `run(...) -> (loss, final_state)`; `load` resets the eligibility traces; `EtraceNetwork` accepts a `Program` (keeps its `spikes` / `detach_reset`) and gains `read()` / `load(state, n)`; online episodes are unbatched (batched online learning stays K1-L) | same client shape as `BPTT`; `run` = braintrace `etrace_grad` over the episode, then one optimizer step (per-step updates inside the loop are K1-L) | 2026-10-22 |
| K1a-5 | lead decision 2026-10-22 (approved in PR #112): learn phase `Program.rollout(params, state, n0, trial, n_steps, *, learn=None, every=1)`; `learn(params, state, n_next) -> (params, state)` runs inside the scan after the observe phase of every step `n` with `(n + 1) % every == 0` (global step count, so consecutive rollouts line up), via `brainstate.transform.cond` (no cond for `every=1`); with `learn` the rollout returns `(state, recordings, params)`, without it `(state, recordings)` as before; `Runner.cont(duration, *, learn=None, every=1)` writes the final parameters back (`Program.write_params`); unbatched runners only (`NotImplementedError` when batched) | lead decision K1a-5 shape; the global-step rule makes "an update every k steps inside one rollout" equal "updates between `cont()` calls" (tested equal within rtol 1e-12 in float64, also across a rollout boundary not aligned with `k`); stateful `braintools` optimizers work inside `learn` (their states are carried by the scan) | 2026-10-22 |
| K1a-5 | etrace checker rule 2 rejects braintrace primitives only inside `while` bodies (`_LOOP_PRIMITIVES = ('while',)`); static `scan` / `fori_loop` bodies are accepted; `# upstream: braintrace#185/#186/#187` comments at rules 3/2/4 | verified on braintrace 0.2.6 (`/tmp/k0/k1a5/scan_etp.py`): D-RTRL and e-prop gradients with the weight inside `fori_loop` / `scan` equal the Python-unrolled loop, and equal BPTT on diagonal hidden dynamics (test); a weight inside a `while` body raises braintrace's own `NotImplementedError` | 2026-10-22 |
| K1a-6 | **lead decision 2026-10-22 (PR #113, option a; was escalated as a false spec premise):** connectivity keys come from the post neuron's population **name and local id**, not its global id: per post neuron `post_keys(fold_in(proj_key, name_id(post population)), local id)` (`rng.view_post_keys`), passed by the compiler to `ConnRule.build(..., post_keys=)` (new optional keyword; without it a rule derives keys from the view ids as before) | the contract kept "per-(projection, post gid block)" keys, but global ids shift by `n` for every population added before the target, so that rule cannot meet the acceptance test "adding … an unrelated population leaves … realised connectivity identical"; option (b), keep global-id keys and document that only appended populations / any projection are invariant, is a one-line revert (drop `post_keys=` in `compiler.py`) | 2026-10-22 |
| K1a-6 | **lead decision 2026-10-22 (PR #113, accepted as implemented):** inside a plan `StepContext.key` holds the stacked per-block keys, shape `(n_blocks, 2)`, `key[b] = fold_in(population_key(seed, name, n, b), trial)`; models draw per-neuron noise only through `rng.draw_blocked(key, n, sampler, *lane_args, tail=)`, which also accepts one key (standalone use: block `b` uses `fold_in(key, b)`) | the blocked layout has to reach the model; a single key in `ctx.key` would let models draw whole-population arrays again; a raw `jax.random.*(ctx.key, …)` on stacked keys fails loudly | 2026-10-22 |
| K1a-6 | noise blocks are **population-local** ids (`local // 1024`, lane `local % 1024`), one `GID_BLOCK` constant for noise and connectivity | global-id blocks shift with earlier populations (same issue as above) and do not align with the per-population state arrays a sharded runtime partitions | 2026-10-22 |
| K1a-6 | name hash: first 8 bytes of `blake2b(name)`, folded into the root key as two `uint32` words `(hi, lo)` (`rng.name_id`, `rng.object_key(seed, name)`); integers still accepted as raw counters for low-level use; a 64-bit collision between two names of one network raises | stable across processes and platforms (Python `hash` is salted); 64 bits make collisions negligible | 2026-10-22 |
| K1a-6 | every block draws a full 1024-lane block (padded, then sliced), so a neuron's numbers do not depend on the population size; lane arguments (e.g. `lam`) are padded with zeros | needed for extension invariance; cost ≤ one partial block per population | 2026-10-22 |
| K1a-6 | `poisson_counts` chooses inversion vs `jax.random.poisson` **per unit** (exact draws from `fold_in(block key, 1)`, computed only if some unit has `lam ≥ 10`); K0 used the exact sampler for the whole array when any unit had `lam ≥ 10` | a unit's count must not depend on other units' rates; distribution unchanged | 2026-10-22 |
| K1a-6 | `poisson_generator_ps` sub-streams are `fold_in(block key, j)`, `j = 0…4` (was `split(key, 5)`) | per-block layout | 2026-10-22 |
| K1a-6 | auto names: populations `<Model>_<k>` (e.g. `iaf_psc_alpha_0`), projections `<RuleClass>_<k>` (e.g. `FixedIndegree_0`; K0: `proj<k>`), observers `Observer_<k>` (K0: `obs<k>`); `k` counts the unnamed objects of that class, skipping names already taken explicitly; names are unique across populations, projections and observers | lead decision `<model>_<k>` per class; the rule class is the projection's "model" and keeps unnamed projections of different rules independent | 2026-10-22 |
| K1a-6 | `Population.uid`, `Projection.uid` and `rng.UidAllocator` removed (no RNG use left) | no dead insertion-index field that looks like an RNG id | 2026-10-22 |
| K1a-6 | not blocked (follow-up for the K1b distributed spec): initial-state distributions (`INIT_STREAM`) and weight/delay distributions keep whole-array draws, now from name-keyed streams | D-g covers per-step dynamics noise; init/weight draws are insertion-order invariant now but not extension/shard invariant | 2026-10-22 |
| K1a-6 | gate measured (remote host, Brunel order 2500 float32, `main` @ ae69409 (K0 kernel) vs K1a-6, 6 alternating rounds each, median of medians; load ≈ 1.5–3 from other jobs, so the idle-core protocol was used): **CPU** (core 27 pinned) 0.5228 → 0.5280 ms/step, **ratio 1.010**; **GPU 1** 0.0706 → 0.0649 ms/step, **ratio 0.920**; gate ≤ 1.05 met on both. First-call compile grows 2.6 → 3.0 s (CPU) and 0.96 → 1.09 s (GPU). A first version that drew the uniforms outside the `lax.cond` of `poisson_counts` ran at ratio 1.13 on GPU (fusion lost); the draw now sits inside both branches as in K0 | gate (lead decision) | 2026-10-22 |
| K1a-1 | `integrate(...)` runs substeps `0 .. k-2` as one `jax.lax.scan` and peels the last substep; the hook signature becomes `hook(k, t, y, aux, final)` with `final` a static Python bool (`True` only after the last substep) and `k` a traced int32 inside the scan; `k = 1` has no scan | the per-step spike check (`rule.when == 'step'`) runs only in the peeled substep, so its cost is unchanged and nothing branches on a traced index (§3 K1a-1 contract, option 1) | 2026-10-22 |
| K1a-1 | the generated step carries every state explicitly: float states in `y`, all other states (spike output, int/bool states, latches) as mantissas in `aux` next to the `SpikeCarry`; `s_holder` removed; `derivatives` keep seeing the step-start values of non-integrated states (K0 semantics) | a jump may change any state; carrying all of them is free (XLA drops unchanged carries) and needs no static analysis of the jump | 2026-10-22 |
| K1a-1 | `jax.lax.scan` (not `brainstate.transform.scan`) for the substeps and the precise segments | the bodies are pure functions of explicit carries with no `State` reads/writes; braintrace D-RTRL / e-prop gradients are unchanged (identical to the last bit on the review nets) | 2026-10-22 |
| K1a-1 | inside the substep scan the carry leaves are cast back to their input dtypes (`integrators._like`); the peeled last substep is not cast (K0 behaviour) | `scan` needs a type-stable carry; a Python-scalar `aux` or a jump to a parameter of another dtype would otherwise fail to trace | 2026-10-22 |
| K1a-1 | `iaf_psc_exp_ps`: one scan over the `capacity + 1` sorted slots (input events + refractory pseudo-event), final segment peeled; the scan driver is the module attribute `precise._segment_scan` so tests swap in the K0 unrolled loop as the reference | compile time flat in the capacity (0.71 / 0.70 / 0.73 s at 8 / 16 / 64 on the remote CPU; K0: 42.9 s at 64) | 2026-10-22 |
| K1a-1 | in-suite K0 reference = the unrolled loops swapped in with `monkeypatch` (`models/substep_scan_test.py`); `aeif` k = 64 and `iaf_cond_exp` `t_ref = 0` k = 64 are left to the cross-commit comparison (the unrolled reference compiles for minutes) | keeps the suite fast; the PR records the comparison against the K0 tree itself on all K0 models / networks, both precisions | 2026-10-22 |
| K1a-1 | compile-time gate measured by `benchmarks/k1a_compile_time.py` (first `simulate` minus a warm `cont`, fresh process per case; first minus warm `BPTT.grad` for the LIF net) and the `k1a_compile`-marked `validation/compile_gate_test.py` (skips unless selected with `-m k1a_compile`, being wall-clock) | spec: “a test (`k1a_compile` marker) records the table”; a wall-clock assertion must not run inside the ordinary parallel suite | 2026-10-22 |
| K1a-1 | not blocked (follow-up, tracked in the upstream TODO §2.1): the Python process sometimes hangs at interpreter exit after a kernel run on the remote host; `benchmarks/k1a_compile_time.py one` calls `os._exit(0)` after printing its result | cause not isolated (jax / XLA CPU thread pool or brainevent numba FFI); the gate test only needs the printed result | 2026-10-22 |
| K1a-1 | ESCALATED, resolved by lead decision (a) above: `iaf_psc_exp_ps` float32 is not within `CAT_B` of K0 on long recurrent / Poisson-driven runs (spike counts identical; V_m up to 1.3 × the `CAT_B` bound after 100 ms, max rel 1.15e-6; I_syn rel ≤ 1.3e-6; spike offsets ≤ 2.8e-5 ms). Cause: K0's own unrolled float32 result depends on the XLA fusion context at the ulp level (the same step evaluated standalone gives the scan's value), and ulp-level offset differences are amplified by the root finder's conditioning. float64 max rel 3.5e-13 (≤ 1e-12). Proposed: float32 precise criterion = spike counts per step identical, V_m / I_syn rel ≤ 1e-5, offsets ≤ 1e-4 ms | gate cannot be met as worded → lead decides (brief) | 2026-10-22 |
| K1a-2 | `latch` is a **keyword-only** argument of `Current(unit, filtered=False, *, latch=True)` and `Graded(unit, filtered=False, *, latch=True)` (a bool; frozen-dataclass field, so `Current(u.pA) != Current(u.pA, latch=False)`); `ports.is_latched(kind)` is the single test; the flag only matters on **input** ports (output kinds — `dc_generator.I`, `NMDA.s`, synapse output kinds — ignore it: the receiving port decides) | lead decision D-a / default True; keyword-only so the positional `filtered` (removed by K1a-3) cannot be confused with it | 2026-10-22 |
| K1a-2 | the global `_LATCHED = (Current, Graded)` rule is replaced by `model.latched_ports(inputs)` and `model.latch_inputs(inputs, s, inp) -> (acting, new_latches)`, used by the generated step and (through `models._common.latch(model, s, inp)`) by every hand-written `Exact()` step; only latched ports get a `latch_<port>` state | one implementation of the latch for both step kinds; an unlatched port's value enters the step directly (constant over the substeps for RK/exp-Euler, through the propagator for exact models) | 2026-10-22 |
| K1a-2 | per-model latch: `iaf_psc_alpha`, `iaf_cond_exp`, `aeif_cond_alpha`, `iaf_psc_exp_ps` declare `I=Current(u.pA, latch=True)` explicitly (NEST `CurrentEvent`; tested by source inspection); `LIF` declares `I=Current(u.mA, latch=False)` (tested); `NMDA.s` (Graded output), `dc_generator.I` (Current output), gated `Conductance` ports and `ExternalInput` (Event) have no input latch to decide — unchanged | D-a; no case was unclear, no NEST-model semantics changed | 2026-10-22 |
| K1a-2 | **schedule:** zero-delay edges into same-step ports (event, timed_event, conductance, unlatched current/graded) are hard constraints (cycle → `ScheduleError`, message now names the port kinds and "a latched port" as the alternative); zero-delay edges into latched ports are soft: `compiler.schedule_ports` keeps the K0 order (all edges) when that graph is acyclic, otherwise orders by the hard edges only and **defers** every latched port whose zero-delay source is not earlier (incl. a zero-delay self-connection): `Plan.deferred = {pop index: frozenset(ports)}`; `Program.step` gathers the deferred ports' deliveries, synapse groups and `ext` after all populations updated and writes only their latch (`new + 0 * old`, braintrace#184) | task instruction "latch=True ports break cycles"; the spec's edge case keeps unlatched zero-delay cycles a `ScheduleError`; deferral is exact because a latched payload never influences its own step (contract stated in `model.py`); tested equal (bit-for-bit, float64, with Expon synapse + ext on the deferred port) to the source-first order; every K0-valid network keeps its K0 order, so nothing is deferred and numerics are bit-identical | 2026-10-22 |
| K1a-2 | `Plan.describe()`: each population line ends with ` inputs: I=current(latch) delta=event` (`(latch)` / `(same-step)` for every current/graded input port), plus a `deferred latched ports (...)` schedule line when any port is deferred | spec: "update `Plan.describe()` to show latch per port" | 2026-10-22 |
| K1a-2 | braintrace#184 workaround (`new + 0 * old`) stays on latched inputs only; same-step inputs create no HiddenState whose new value depends only on inputs (they enter `V`'s update), so none is needed — G7 / etrace tests green | checked by the full suite and `-m k0_gate` | 2026-10-22 |
| K1a-2 | K0-vs-K1a-2 equality for every latch=True model, measured on the remote host against `origin/main` @ ca228d6 (`/tmp/k0/k1a2/k0_vs_k1a.py`: Brunel `iaf_psc_alpha` order 40 + `dc` → `I`, COBA `iaf_cond_exp` (scale 0.1) + `dc`, `aeif_cond_alpha` + Poisson + `dc`, `iaf_psc_exp_ps` + `poisson_generator_ps` + `dc`, and the K0 `LIF` vs a K1a `LIF` subclass with `latch=True` in a zero-delay Expon chain with recurrence): **bit-identical in float64 and float32** (max rel 0) | spec bound f64 ≤ 1e-12; NEST parity therefore unchanged | 2026-10-22 |
| K1a-2 | `LIF` timing changes as approved (D-a): the K0 LIF vs `_brainpy.LIF` parity test no longer shifts the input by one step (the kernel `LIF` now equals the BrainPy model step for step); `runner_test` dc → LIF onset moves one step earlier (a NEST-model latched variant was added); the Expon PSP closed form in `synapse_test` uses `N + 1` | intended semantic change of D-a / K1a-2 | 2026-10-22 |
| K1a-4 | new private module `_kernel/delay_scatter.py`: `DelayScatter(Storage)` (`kind='scatter'`) holds all edges of a projection with delay ≥ 1 in row (pre) order, sorted by ring offset within a row (`np.lexsort`, locality of the scatter); each edge stores `rel = (d mod L) * n_post + post`, position at step `n` = `rel + (n mod L) * n_post` (minus `L * n_post` on wrap); rows are cut into segments of `M` = max out-degree capped at 2 × mean, so a hub does not pad every row; `deposit(ring, step, payload, w)` replaces `forward` (which raises `TypeError`, as does `forward_etrace`) | one op per projection and step, as the contract asks; edge order kept for `get_connections` / `set('weight')` (permutation, tested exact) | 2026-10-22 |
| K1a-4 | two strategies: `'events'` (event sources: visit only segments of sources with a non-zero payload, `chunk` segments per `lax.while_loop` iteration, contiguous `dynamic_slice` of length `M`) and `'dense'` (value sources, and event projections ≤ `dense_max_edges` on GPU: every edge scatter-adds each step). `'events'` has an optional loop-free `capacity`: the first `K` active segments in one chunk, the loop only when more are active (behind `lax.cond`) | GPU `while` iterations cost a device-to-host predicate sync each (~30 us measured on GPU 1), the loop-free chunk took Brunel 2500 f32 GPU from 0.215 to 0.116-0.131 ms/step; on CPU loops are cheap (capacity 0) | 2026-10-22 |
| K1a-4 | the deposit is a `jax.custom_jvp`: event-driven primal, dense linear tangent `dring + scatter(dpayload[pre] * w + payload[pre] * dw)` | exact JVP/VJP of the scatter in weights, payload (also for silent sources in the step, which the event primal skips) and ring; BPTT FD check passes; vmap works (tested over trials and steps) | 2026-10-22 |
| K1a-4 | cost model `DelayCostModel` (`compiler.py`; per backend in `DELAY_COST_MODELS`, `'cpu'` fallback, chosen by `jax.default_backend()`): grouped step `G * group_step_us + a * E * group_edge_ns`, scatter step `scatter_step_us + a * E * scatter_edge_ns` (`'dense'`: `E * dense_edge_ns`), `a = activity = 0.003`; grouping only if `G ≤ max_groups` and not more expensive. Hard ceiling `max_groups = 8`, recorded as `Plan.max_delay_groups` and printed by `describe()` with the choice, both estimates, the strategy and capacity, and the storage memory per projection. CPU constants `group_step_us 6, group_edge_ns 1.5, scatter_step_us 8, scatter_edge_ns 3, dense_edge_ns 2`; GPU `40, 0.5, 30, 1.5, 0.05`, `chunk 64`, `capacity_factor 4`, `capacity_min 32`, `dense_max_edges 2^20` | calibrated on the remote host (Brunel 400 CPU f32: scatter 0.77-0.85 × scalar at any delay; grouped 1.48 × at G = 2, 4.4 × at G = 8, 14.2 × at G = 200 with 16 × build); with these constants every Brunel / review projection with ≥ 2 delay values is scattered, `G = 1` stays grouped | 2026-10-22 |
| K1a-4 | the zero-delay edges of a per-edge projection keep their own group (same-step delivery as in K0); the scatter group has `EdgeGroup.delay = None`; it ignores `Projection.storage` (own layout) | delay 0 is not a ring slot; acyclic same-step semantics unchanged | 2026-10-22 |
| K1a-4 | `PostRing.take(buf, step)`: read slot `n mod L`, `lax.optimization_barrier((row, buf))`, clear it; the runner reads and clears all post-side rings at the start of the step and emits into the cleared ring | without the barrier XLA copied the whole carried ring every step (0.06 ms/step at L = 200, 2000 post); changes the speed of K0's grouped post-side path too, not its numerics. Follow-up (not done): `PreRing` (read then write of the carry) likely has the same copy | 2026-10-22 |
| K1a-4 | braintrace: no capability lost. K0's `check_etrace` already rejected trainable per-edge (d ≥ 1) weights (they reach only the arrival ring, a `ShortTermState`); the error now names the per-edge projections and the options (one delay per projection or per source, non-trainable, or BPTT). Non-trainable scatter projections run under `Online` (tested with D-RTRL / e-prop) | the spec asks for a clear rejection; `check_plan` is unchanged (rejecting there would break forward use of `EtraceNetwork`) | 2026-10-22 |
| K1a-4 | `trainable_states` / `Plan.states(ParamState)`: a trainable per-edge-delay projection now has one weight state for its delay ≥ 1 edges (plus one for delay-0 edges) instead of one per delay value (keys `weights/<proj>/0`, `weights/<proj>/1`); existing tests updated (`compiler_test`, `runner_test`, `bptt_test`) | inherent in "one storage" (task statement); internal `_kernel` layout | 2026-10-22 |
| K1a-4 | additions: `ProjectionPlan.delivery` (`'grouped'`/`'scatter'`/`''`) and `delivery_cost` (estimates, `n_groups`, `strategy`, `capacity`), `Plan.max_delay_groups`, `compiler.DelayCostModel` / `DELAY_COST_MODELS` / `delay_cost_model`; `validation.brunel.build_brunel(..., delay=None)` overrides the recurrent delay (benchmark) | private `_kernel` API, nothing exported publicly | 2026-10-22 |
| K1a-4 | `# upstream: brainevent#203` on the scatter (fused event-driven per-edge-delay kernel is the fast path behind the same `deposit`). GPU float64 benchmarks use brainevent's `cusparse` backend (`k1a4_delays.py --brainevent-backend`): `BinaryArray(float64) @ CSR(float64)` aborts the process on GPU (brainevent#199, already in the upstream TODO) | existing upstream issues; no new row needed | 2026-10-22 |
| K1a-4 | gate measured (`benchmarks/results/2026-10-22-K1a-4-delays.json`, table in `benchmarks/results/README.md`): the spec's gate network (review: 2000 neurons, indegree 100, delays up to 2/5/20 ms) passes on CPU (pinned, interleaved) and GPU 1, f32 and f64: step 0.87-0.96 ×, build 1.09-1.28 ×; Brunel order 400 passes everywhere (step 0.53-0.86 ×). **Brunel order 2500 does not pass everywhere**: CPU up to 20 ms step 1.39-1.55 × (f32 1.39 / 1.52, f64 1.40 / 1.54 / 1.55 in three sweeps), build 1.43-1.58 ×; GPU f32 1.63 / 1.72 / 1.84 × at 2 / 5 / 20 ms; GPU f64 (cusparse) 0.40-0.42 ×. escalated; resolved by the lead decision below (gate as written, order 2500 a known gap) | the CPU excess at 20 ms is the 10 / 20 MB ring (cache-bound random writes); the GPU excess is kernel count (~10 small kernels + one `cond` sync per projection vs one CSR kernel) | 2026-10-22 |
| K1a-4 | lead decision 2026-10-22 (PR #115): the gate is met as written (spec network 2000 × 100, delays ≤ 20 ms, CPU and GPU, f32 and f64). **Known gap**, recorded here and in the architecture doc §12 (K1b row) / §13: Brunel order 2500 with delays ≤ 20 ms, CPU 1.39-1.55 × and GPU f32 1.63-1.84 × the scalar-delay step; tracked on [brainevent#203](https://github.com/chaobrain/brainevent/issues/203) (measured numbers commented on the issue; upstream TODO row added). Follow-up: `PreRing` whole-ring copy per step suspected (same read-then-write pattern as `PostRing` before `take`), unmeasured | lead decision | 2026-10-22 |
| K1a-4 | lead decision 2026-10-22 (PR #115): accept one weight state for a per-edge projection's delay ≥ 1 edges plus one for its delay-0 edges (`trainable_states` keys change from one per delay value) | lead decision | 2026-10-22 |
| K1a-4 | `# upstream:` comments assigned to K1a-4 by the upstream TODO §1.1 added: brainevent#198 (`connectivity/storage.py` `payload_path`), #201 (Dense / CSR float-path matmuls), #200 (`plasticity.py` `csr_to_csc_index`), #202 (`plasticity.py` `_resolve_path`), #203 (`delay_scatter.py` `_scatter`) | brief, upstream procedure | 2026-10-22 |
| K1a-2 | **lead decision 2026-10-22 (K1a-2 follow-up):** zero-delay cycles through a latched port are ALLOWED (as merged in #116); the compiler defers the latched port. Documented limitation: hand-written models must not read a latched port's current-step value (not enforced). | lead decision | 2026-10-22 |
| K1a-3 | `Event(max_multiplicity=1, unit=None, *, into=None)` and `TimedEvent(capacity, unit=None, *, into=None)`: `into` is **keyword-only** (an identifier, else `PortError`); `Current(unit, *, latch=True)`, `Graded(unit, *, latch=True)`, `Conductance(unit, E_rev, gate)` lose `filtered` (passing it is a `TypeError`); `PortKind.into` is `None` on every non-receptor kind; new `ports.is_receptor(kind)` (exported from `_kernel`) is the single test | lead decision `into=` / one encoding; keyword-only like `latch`, so `Event(u.nS, into=...)` (first positional = `max_multiplicity`) fails loudly — the architecture doc §7 example now writes `Event(unit=u.nS, into='g_ex')` | 2026-10-22 |
| K1a-3 | receptor validation at **class creation** (`NeuronModel._validate_receptors`, `PortError`): the named state must be declared, be a float state, and have the port unit's dimension (skipped when the state's `Init` is a callable without a declared unit); `into` on an **output** port has no effect (not validated), as `latch` on output ports | spec edge case "a receptor port naming an undeclared state (error at class creation)"; follows the K1a-2 precedent for `latch` on outputs | 2026-10-22 |
| K1a-3 | the default `NeuronModel.on_input` applies the receptor jumps: every `Event(..., into=s)` adds its payload to `s` (several ports into one state add up in port order); models without receptor ports are unchanged (same object returned). `iaf_cond_exp` drops its own `on_input` (identical ops); `aeif_cond_alpha` keeps its override (`dg += e / tau * w`), `iaf_psc_alpha` / precise models apply jumps in their hand-written `step`. `TimedEvent` receptor payloads are left to the model's `step` | spec contract "a delivered weight … jumps that state after propagation"; for alpha kernels the named state is the kernel output (`g_ex`, `I_syn_ex`, the weight is its peak) and the jump enters its derivative state — the spec itself names `g_ex` / `I_syn_ex` for these models | 2026-10-22 |
| K1a-3 | weight sign: a receptor port whose unit is a **conductance** takes non-negative weights only (`Event.nonnegative_weights` / `TimedEvent.nonnegative_weights` property, checked by the unchanged `check_weight_sign`, distributions by their clipped support); current receptors (`iaf_psc_alpha`, pA) keep signed weights | §7 L1 "an excitatory conductance cannot be drawn negative"; keeps K0's behaviour exactly (`Conductance(filtered=True)` ports were non-negative, `Event(unit=pA)` ports signed) without a new keyword | 2026-10-22 |
| K1a-3 | single owner: `check_synapse_connection` rejects **any** `SynapseModel` on a receptor port (`check_single_owner(dst, True)`), message `"<src> -> <Model>.<port>: the target is a neuron-owned receptor port: the neuron model owns its kinetics (it filters the input into state 'g_ex') …"`; the K0 special case in `check_kind_compatible` (events straight into `filtered` `Current`/`Conductance` ports) is removed, so `Event`/`TimedEvent` → `Current`/`Conductance` without a synapse is a `PortError` again | spec: "attaching a `SynapseModel` to a receptor Event port raises `PortError` naming the owner of the kinetics" | 2026-10-22 |
| K1a-3 | migrated: `iaf_cond_exp` / `aeif_cond_alpha` `ex=Event(unit=u.nS, into='g_ex')`, `in_=Event(unit=u.nS, into='g_in')`; `iaf_psc_alpha` `ex=Event(unit=u.pA, into='I_syn_ex')`, `in_=Event(unit=u.pA, into='I_syn_in')`; `precise._capacity_variant` keeps `into`. **Not** migrated (never used either encoding): `iaf_psc_exp_ps.syn` (one port sign-routed into `I_syn_ex` / `I_syn_in`, does not fit one `into` state), `LIF.delta` (jumps `V`, no kinetics), `NMDA.spike` (receptor population input), `with_conductance_ports` ports (plain `Conductance`) | spec migration list; the sign-routed precise port and a possible `into` for delta ports are follow-ups for K2 (lead) | 2026-10-22 |
| K1a-3 | `Plan.describe()` / `Ports.__repr__` show the filtered state of receptor ports (`ex=event(into g_ex)`, `Ports(ex=Event(into='g_ex'), …)`) | visibility of the encoding | 2026-10-22 |
| K1a-3 | numerics: `origin/main` @ 488f7d1 vs K1a-3 on the remote host (`/tmp/k0/k1a3/main_vs_k1a3.py`: Brunel `iaf_psc_alpha` order 40 + dc, `iaf_psc_alpha` doublets with signed weights, COBA `iaf_cond_exp` (scale 0.1) + dc, `iaf_cond_exp` from spike generators with doublets and delay 0, `aeif_cond_alpha` Poisson + recurrent, `iaf_psc_exp_ps.with_capacity(8)` + `poisson_generator_ps`, `iaf_cond_exp` + gated NMDA port + `ex` + Expon on `I`): **bit-identical in float64 and float32** (25/25 recorded arrays each, max rel 0) | spec: NEST parity unchanged; the change is declarative plus identical jump ops | 2026-10-22 |
| K1a-10 | cast float64 binary-path events to **float32** (not bool) before brainevent; Dense through `_binary_dense` (`jax.custom_jvp`, event tangent `dx @ W` in W's dtype), CSR with a plain cast; output keeps W's dtype (tested) | brainevent#204: float64 `x` @ dense W is wrong on GPU, bool/float32 are correct. bool was rejected: `astype(bool)` has a zero cotangent, so the event (input) gradient would silently become 0 (`/tmp/k0/k1a7/grad_cast.py`). A plain float32 cast keeps the forward and the W gradient identical but rounds the Dense event gradient to float32 (rel ~4e-8); the custom_jvp restores float64 (matches the float path to 1e-12). The CSR event gradient was already identical with the cast. The same cast avoids the #199 default-backend abort | orchestrator (delegated by lead), 2026-10-23 |
| K1a-10 | verification: GPU 1, Brunel ε = 0.1 float64, Dense vs CSR, default and `cusparse` backends: order 200 57.10 / 57.10 Hz (old `Simulator` 57.16 Hz; before the fix Dense 59.7 Hz), order 800 45.20 / 45.20 Hz (before the fix Dense 53.6 Hz) | regression evidence at network level for the #204 cast | orchestrator (delegated by lead), 2026-10-23 |
| K1a-10 | audit of merged results for float64 + GPU + Dense: K0 G9 / T10 are CPU-only (unaffected); T0 is the old `Simulator` (unaffected); K1a-4 `gpu64cs_brunel2500` and the per-edge-delay rows use CSR / `DelayScatter` (unaffected; rates equal the CPU rows); **affected**: K1a-4 `gpu64cs_brunel400` scalar-delay rows (`auto` chose Dense: 57.68 Hz vs 52.44 Hz on CPU), whose rates are wrong; their step times are kept as measured (the binary Dense kernel's cost does not depend on the payload values) but the rate column is superseded; `gpu64cs_review0` fires 0 Hz (no events, unaffected); the K1a-7 G3 float64 order-200 kernel row measured before K1a-10 is superseded (re-measured in K1a-7) | correctness note for the K1a report | orchestrator (delegated by lead), 2026-10-23 |
