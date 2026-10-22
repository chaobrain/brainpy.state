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

**In:** K1a-1 … K1a-7 below (one PR each), the CI job of v0.1.1 goal 02 that serves
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

---

## 4. Dependency and wave plan

| wave | tasks | notes |
|---|---|---|
| A | K1a-1, K1a-4, K1a-5, K1a-6 in parallel | K1a-1 touches `integrators.py`, `model.py`, `models/precise.py`; K1a-4 `compiler.py`, `delays.py`, `storage.py`; K1a-5 `runner.py`, `learn/`, `spiking.py`, `compiler.py` (`runner()`); K1a-6 `rng.py`, `network.py`, `runner.py`, `models/devices.py`. Overlaps (`compiler.py`, `runner.py`) are small; later PRs rebase |
| B | K1a-2, then K1a-3 | both edit `ports.py`, `model.py` and the model files; K1a-2 after K1a-1 (same `model.py` step code) |
| C | K1a-7 | after all others; measures the merged kernel |

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
| K1a-2 | lead: `latch` **defaults to `True`** (K0 behaviour; NEST-style one-step latch unless a port opts out with `latch=False`); the BrainPy-family `LIF` declares `latch=False` explicitly. (Differs from the spec author's recommendation of a required keyword.) | lead decision | 2026-10-22 |
| K1a-3 | lead: keyword `into=` (e.g. `ex=Event(unit=u.nS, into='g_ex')`); the same keyword on `TimedEvent` receptor ports; the `filtered` flag is removed everywhere | lead decision | 2026-10-22 |
| K1a-4 | lead: build time ≤ 2 × the scalar-delay build; step time ≤ 1.5 × scalar; the compiler picks scatter vs grouping by a cost model, with a hard ceiling on groups recorded in the plan | lead decision | 2026-10-22 |
| K1a-5 | lead: learner objects `BPTT(program, loss, optimizer, ...).run(state0, data)`; learn phase as `Program.rollout(..., learn=fn, every=k)`; exact signatures shown in the K1a-5 PR for lead approval before merge | lead decision | 2026-10-22 |
| K1a-6 | lead: gid block 1024 (same constant as connectivity); unnamed objects auto-named `<model>_<k>` per class (insertion order matters only among same-class unnamed objects — documented); gate: step time ≤ 1.05 × K0 at Brunel order 2500 | lead decision | 2026-10-22 |
| K1a-7 | lead: Potjans–Diesmann microcircuit with 5 seeds each side, 1 s after a 100 ms transient, CPU and GPU 1; AI Brunel point chosen by a short NEST-only scan requiring CV-ISI ≥ 0.5, shown to the lead before the kernel comparison runs | lead decision | 2026-10-22 |
| K1a-5 | **proposed, awaiting lead approval (PR draft):** the spike mode and the program-level `detach_reset` reach the models as **static fields of `StepContext`** (`ctx.spikes`, `ctx.relaxed`, `ctx.detach_reset`; static pytree metadata, so they are part of the jit cache key); `SpikeRule.check(..., relaxed=False)` takes the mode explicitly; `relaxed_spikes()` / `relaxed_enabled()` are removed | removes the `ContextVar` (no global state, visible in signatures); two programs with different modes never share a trace | 2026-10-22 |
| K1a-5 | **proposed:** `Program(plan, *, spikes='hard', detach_reset=None, etrace=False)` (keyword-only options), `Program.replace(**options)`, `Plan.program(...)`; `detach_reset` resolution: rule's own setting > program option > model's `detach_reset` > False | learners set `detach_reset` / `spikes` by deriving a program (architecture doc §8.2-2) instead of mutating the models (K0 BPTT wrote `m.detach_reset` on every model of the plan) | 2026-10-22 |
| K1a-5 | `spikes='relaxed'` uses the exact float payload path for every projection (the binary event path would binarise fractional spikes); construction fails with a list of populations whose spike rule has no surrogate with `surrogate_fun`, or that are precise-timing neurons; devices and relays are allowed | K0 tests set `pp.path = 'float'` by hand, which mutated the shared plan; the spec asks for a clear error on models without a surrogate | 2026-10-22 |
| K1a-5 | `Program` captures its initial state at construction (`replace` shares it); `init_state()` no longer reads the models' live `State` values | found while testing `Online` from any state: `EtraceNetwork` writes the models' states, after which K0's `init_state()` returned the current, not the initial, values | 2026-10-22 |
| K1a-5 | **proposed:** `Runner(program_or_plan, batch=None, rng='independent')`; `Plan.runner(batch, rng)` loses `learner=`; `Runner.learner` and `learner.attach(runner)` are removed; new `Runner.trials`; `Program.init_state(batch=None)`, `Program.write_params(params)`, `trial_indices(batch, rng)` | D-c: the runner is just another client of the program | 2026-10-22 |
| K1a-5 | **proposed:** `BPTT(program, loss, optimizer=None, *, checkpoint=None, truncation=None, detach_reset=True, spikes=None)` with `record / loss / grad / run / fit(state0=None, data=None, targets=None, *, duration=None, n0=0, trials=None)`; `run` returns `(loss, final_state)` (one optimizer step), `grad` returns `(loss, grads)`, `fit` takes `n_epochs=` keyword-only; `state0=None` is the program's initial state; the episode length comes from `data` (or `duration=`); a batched `state0` is given with `trials=` (e.g. `runner.trials`); validation (trainable parameters, observers, differentiability) happens at construction | lead decision D-c / K1a-5 shape; episodes from any state fix the step-0 limitation; `n0` keeps RNG streams and time aligned with `Runner.cont()` (tested equal) | 2026-10-22 |
| K1a-5 | **proposed:** `Online(program, algorithm='eprop', *, inputs=None, outputs=None, loss=None, optimizer=None, **options)` with `load(state0=None, n0=0)`, `state()`, `grad(state0, data, targets, *, n0=0) -> (loss, grads)`, `run(...) -> (loss, final_state)`; `load` resets the eligibility traces; `EtraceNetwork` accepts a `Program` (keeps its `spikes` / `detach_reset`) and gains `read()` / `load(state, n)`; online episodes are unbatched (batched online learning stays K1-L) | same client shape as `BPTT`; `run` = braintrace `etrace_grad` over the episode, then one optimizer step (per-step updates inside the loop are K1-L) | 2026-10-22 |
| K1a-5 | **proposed:** learn phase `Program.rollout(params, state, n0, trial, n_steps, *, learn=None, every=1)`; `learn(params, state, n_next) -> (params, state)` runs inside the scan after the observe phase of every step `n` with `(n + 1) % every == 0` (global step count, so consecutive rollouts line up), via `brainstate.transform.cond` (no cond for `every=1`); with `learn` the rollout returns `(state, recordings, params)`, without it `(state, recordings)` as before; `Runner.cont(duration, *, learn=None, every=1)` writes the final parameters back (`Program.write_params`); unbatched runners only (`NotImplementedError` when batched) | lead decision K1a-5 shape; the global-step rule makes "an update every k steps inside one rollout" equal "updates between `cont()` calls" (tested equal within rtol 1e-12 in float64, also across a rollout boundary not aligned with `k`); stateful `braintools` optimizers work inside `learn` (their states are carried by the scan) | 2026-10-22 |
| K1a-5 | etrace checker rule 2 rejects braintrace primitives only inside `while` bodies (`_LOOP_PRIMITIVES = ('while',)`); static `scan` / `fori_loop` bodies are accepted; `# upstream: braintrace#185/#186/#187` comments at rules 3/2/4 | verified on braintrace 0.2.6 (`/tmp/k0/k1a5/scan_etp.py`): D-RTRL and e-prop gradients with the weight inside `fori_loop` / `scan` equal the Python-unrolled loop, and equal BPTT on diagonal hidden dynamics (test); a weight inside a `while` body raises braintrace's own `NotImplementedError` | 2026-10-22 |
