# /goal 04 · STDP core family — approach + implementation plan

> Companion to `nest-goal-04-stdp-core.md` (the goal spec). Worktree
> `nest-goal/04-stdp-core` off `origin/main` (e815dbf). Approach-first: this
> doc is the design artifact; **no code until approved.**

## 0. Research outcome (what is now pinned)

Read in full: `CONTEXT.md` (Part 2 + the `01-event-plastic-substrate`,
`00-validation-harness`, `02/03` Lessons), `synapses-plasticity-gap.md` §5,
the substrate (`_event_plastic.py`: `EventPlasticProj` / `KernelContext` /
`init_state` / `update`), `_plastic_base.py` (`frozen`, unit helpers), the two
worked specs (`static_synapse.py`, `tsodyks2_synapse.py`), the legacy
`stdp_synapse.py` / `stdp_triplet_synapse.py`, and the **exact NEST C++ math
for all 7 models** (facilitate_/depress_ forms, clip sites, trace ownership +
taus, defaults, update ordering — line-referenced to nest-simulator/models).

Three findings that shape the plan:

1. **Only `stdp_triplet_synapse` needs >1 trace per side.** The other six fit
   the **already-frozen** substrate: `stdp_synapse`, `stdp_synapse_hom`,
   `stdp_pl_synapse_hom`, `vogels_sprekeler_synapse`, `jonke_synapse` each use
   exactly one pre trace (`tau_plus`/`tau`) and one post trace (`tau_minus`);
   **`ht_synapse` uses no trace at all** — it is a depression-only vesicle pool
   (`P`), pre-spike-only, no post history, no `Wmax` — structurally identical to
   `tsodyks2_synapse` (per-edge state + `frozen`-gated writeback).
2. **`tau_minus` lives on the post neuron in NEST** (ArchivingNode) for all six
   trace models — never a synapse param. We keep it a synapse-rule attr
   (standalone fidelity) and **document the delta in every docstring** + a
   `CONTEXT.md` Lessons line. Same for `tau_minus_triplet` (triplet `o2`).
3. **"hom" ≠ homogeneous weight.** `stdp_synapse_hom` / `stdp_pl_synapse_hom`
   share *parameters* (NEST common-properties) but keep **per-synapse plastic
   weight**. In our spec-on-substrate design params are *always* rule-level
   (shared across edges), so `is_homogeneous_weight=False` for **all 7** STDP
   models, and `stdp_synapse_hom` reuses `stdp_synapse`'s kernel verbatim.

## 1. The contract decision — **(a) extend the substrate**, minimally + additively

**Decision: option (a).** Generalize the per-side trace seam from *one* tau to
*a tuple* of taus, keeping single-tau and `None` as degenerate cases.
`stdp_triplet_synapse` is the only consumer this cluster, but the per-neuron
multi-trace seam is the correct foundation that **05 (`stdp_nn`) and 07
(Clopath) depend on**.

Why (a) over (b) (per-edge `edge_state_init` second trace pair):

- **07 Clopath forces a per-neuron seam anyway.** Clopath's two low-pass
  *voltage* filters (`u-`, `u+`) are per-**post-neuron**, driven by a continuous
  post variable, **not** reconstructable per-edge from `ctx.post_spike`. Option
  (b) solves triplet locally but leaves 07 to extend the substrate regardless —
  ending with two parallel mechanisms. (a) builds the seam once.
- **Per-neuron (N) beats per-edge (E).** Triplet's `r2`/`o2` are semantically
  per-pre / per-post (identical across a neuron's synapses). (b) stores them
  E-wide (redundant; E ≫ N in real STDP nets); (a) stores them N-wide, matching
  NEST's archive layout.
- **It is the documented extension point.** 01's Lessons: *"If a rule needs
  cross-edge or per-target reductions beyond the per-edge `update`, that is the
  first substrate extension point."* Multi-trace per neuron is exactly that, and
  the change is **purely additive** (below), so cluster-01's six specs + their
  tests are unaffected (regression-gated).

### Exact substrate change (additive, backward-compatible)

`_network/_event_plastic.py`:

- **Rule declaration widening.** `pre_trace_tau` / `post_trace_tau` accept
  `None` | `Quantity` (scalar, current) | **`tuple[Quantity, ...]`** (new). A
  one-line normalizer `_as_tau_tuple(x) -> tuple[float_ms, ...]` (`()` for
  `None`, `(x,)` for scalar, `tuple(x)` for a tuple).
- **`init_state`.** Allocate the per-neuron trace State as `(_n_pre_pop, k_pre)`
  (and `(_n_post_pop, k_post)`), `k = len(taus)`; allocate **only if `k >= 1`**
  (so `None` → no State, unchanged). 1-D single-trace State becomes a `(N, 1)`
  column — shape change is invisible to existing specs (they declare `None`).
- **`update`.** Build a `(1, k)` decay row `exp(-dt / taus)`, decay the `(N, k)`
  State columnwise, add the (delayed) spike vector to **every** column, gather
  `(E, k)` per edge. Keep `pre_trace_edge`/`post_trace_edge` as the **first
  column** `(E,)` (single-trace back-compat) and add `pre_traces_edge`/
  `post_traces_edge` `(E, k)`.
- **`KernelContext`.** Append two fields: `pre_traces: (E, k_pre)`,
  `post_traces: (E, k_post)` (empty `(E, 0)` when no trace). Existing fields and
  their order are unchanged → all current kernels keep working (they read
  `ctx.pre_trace` / `ctx.post_trace`). Triplet reads `ctx.pre_traces[:, 0]`
  (`r1`), `[:, 1]` (`r2`), `ctx.post_traces[:, 0]` (`o1`), `[:, 1]` (`o2`).

This is the *whole* substrate delta. No change to delivery, CSR, delay, weight,
or aux handling.

## 2. Online ↔ NEST-deferred equivalence (correctness foundation)

NEST `send()` is **deferred**: on a pre spike it replays every post spike since
the last pre spike (potentiation, each `facilitate_` compounding the running
weight), then depresses once, then delivers the post-depression weight; the
`weight_recorder` samples **at `send()` = pre-spike times**. The substrate is
**online** (clock-driven, per-neuron decay-then-add traces). Mapping:

- **Potentiation on a post-spike step**, using `K+` propagated to "now". The
  substrate's `pre_trace` decayed to a post step equals NEST's
  `Kplus_ · exp((t_lastpre - t_post)/tau_plus)` term exactly.
- **Depression on a pre-spike step**, using `K-` ("now").
- Between two pre spikes there is no depression, so the *set and order* of
  `facilitate_` ops is identical to NEST's batch at the next pre spike ⇒ **weight
  coincides at every pre-spike/send time** (diverging only mid-interval, which
  NEST never records). **Parity is asserted at send times.**

**Simultaneous pre&post on one step** (the ordering edge case): NEST's
potentiation window is half-open `(t_last-d, t_spike-d]` and its `get_K_value`
is **strictly** before `t_spike-d`; the synapse's own `Kplus_` is incremented
**last**. To mirror: potentiation reads `ctx.pre_traces - ctx.pre_spike[:,None]`
(exclude this step's pre +1), depression reads `ctx.post_trace - ctx.post_spike`
(exclude this step's post +1). For non-coincident steps the subtracted term is 0
(no-op). **This exact choice is verified against NEST in a dedicated test.**

Delay: the substrate already delays *pre* spikes (InputDelay) before both
trace-add and `pre_fired`, shifting the pre causal reference by `d` — the analog
of NEST's `t_spike - d` window. Validated by the `delay=1ms` single-pair + 5 s
runs; any divergence is a parity-test failure to diagnose, never patched in the
example.

## 3. Per-model kernel specs (NEST-verbatim)

All: `is_homogeneous_weight=False`, `stochastic=False`, `weight_unit=pA`,
per-edge `weight`; `fired_pre = ctx.pre_spike > 0`, `fired_post =
ctx.post_spike > 0`; writeback gated with `frozen(...)` **per side**
(depression/aux-pre on `fired_pre`, potentiation on `fired_post`). Clip sites
kept exactly where NEST clips (inside facilitate_/depress_). `kplus =
ctx.pre_trace - ctx.pre_spike`, `kminus = ctx.post_trace - ctx.post_spike`
(simultaneous-exclusion; ms-decayed by the substrate).

| model | pre tau(s) | post tau(s) | potentiation (on post) | depression (on pre) | clip |
|---|---|---|---|---|---|
| `stdp_synapse` | `tau_plus` | `tau_minus` | `ŵ += λ(1-ŵ)^μ+ · kplus` | `ŵ -= αλ ŵ^μ- · kminus` | `[0,Wmax]` in-helper |
| `stdp_synapse_hom` | = stdp_synapse (shared params; **same kernel**) | | | | |
| `stdp_pl_synapse_hom` | `tau_plus` | `tau_minus` | `w += λ w^μ · kplus` (no Wmax) | `w -= λα w · kminus` (linear) | lower-only |
| `stdp_triplet_synapse` | `(tau_plus, tau_plus_triplet)` | `(tau_minus, tau_minus_triplet)` | `|w| += r1(A2+ + A3+·(o2-1))` | `|w| -= o1(A2- + A3-·r2)` | copysign, both |
| `vogels_sprekeler_synapse` | `tau` | `tau_minus` | `|w| += η·kplus` **and** `+= η·kminus` at pre | const `|w| -= α·η` at pre | copysign, both |
| `jonke_synapse` | `tau_plus` | `tau_minus` | `w += λ(exp(μ+ w)·kplus - β)` | `w += λ(-α exp(μ- w)·kminus - β)` | `[0,Wmax]` in-helper |
| `ht_synapse` | — | — | — (none) | pool: `P=1-(1-P)e^{-h/τ_P}`, deliver `w·P`, `P*=(1-δ_P)` | bounds by construction |

Triplet trace ordering (FP-load-bearing, per NEST): potentiation uses `o2-1`
(value just before each post spike — the `o2` trace excludes its own +1 via the
`-ctx.post_spike`-style read on the second column); depression uses `r2`
**decayed but pre-increment** — replicated by reading `ctx.pre_traces[:,1] -
ctx.pre_spike` (r2 without this step's +1). `ht_synapse` carries
`{'P', 't_lastspike'}` per-edge aux (first spike skips decay via
`t_lastspike < 0`, like tsodyks2); `vogels` and the rest carry **no aux** (pure
weight + substrate traces).

Defaults reproduced verbatim from NEST source (e.g. `stdp`: tau_plus 20, λ 0.01,
α 1, μ± 1, Wmax 100; `pl`: λ 0.1, μ 0.4, no Wmax; `triplet`: tau 16.8/101/20/110,
A 5e-10/7e-3/6.2e-3/2.3e-4; `vogels`: weight 0.5, α 0.12, η 1e-3, Wmax 1;
`jonke`: λ 0.01, β 0, μ± 0; `ht`: tau_P 500, δ_P 0.125, P 1). Validation +
ranges done by `_plastic_base` helpers.

## 4. Weight-trajectory recording seam (`_network`)

Mirror `res.trace` (analog State-tap). Minimal, general (cluster 09 builds on it):

- `Simulator.connect(..., synapse=<spec>)` **returns a projection handle** (the
  `EventPlasticProj`; currently returns `None` — additive, existing callers
  ignore it).
- `Simulator.record_weight(proj)` registers a weight tap
  (`_weight_taps[id(proj)] = proj`), analogous to `_analog_taps`.
- The `simulate()` `for_loop` stacks `proj.weight.value` each step → `(T, E)`
  (or `(T,)` if homogeneous), stored as a pA `Quantity`.
- `SimulationResult.weight_trace(proj)` returns that `(T, E)` Quantity.

Unit-tested **NEST-free** (build pre→post plastic `connect`, `record_weight`,
run, assert shape `(T, E)` + weight evolves). The parity tests do **not** use
this seam — they hand-roll a `for_loop` tapping `proj.weight.value` directly (the
`stp_parity_test` drive pattern), so the seam is an independent deliverable.

## 5. Validation (`_validation/stdp_*_parity_test.py`, `requires_nest`)

Drive pattern (per 01 Lesson): NEST `spike_generator → parrot_neuron →
synapse → post`; post made to spike at chosen steps via a second
`gen → parrot` driving the post `parrot_neuron`; `weight_recorder` (or
`GetStatus`) reads weight at send times. brainpy side: single 0→0 edge via
`EventPlasticProj`, `box['v']` pre-spike closure + a post-spike closure,
`for_loop` injecting the **same** spike steps, tap `proj.weight.value`.

- **Single-spike-pair regression** per rule: one pre/post pair, sweep `Δt` over
  a window; assert post-update weight == NEST (`compare_trace`, **CAT_B-tight**,
  deterministic). This nails facilitate_/depress_ + trace decay + clip.
- **5 s fixed-train weight trajectory** per rule: pre & post driven by the
  **same precomputed spike trains** on both sides (deterministic; avoids
  NEST↔JAX PRNG divergence); compare weight-at-send-times with `compare_trace`
  + **CAT_A (1e-3)**. Exercises compounding, clamps, long-run trace dynamics.
- **`stdp_triplet`**: a post-pre-post protocol with `o2 ≠ 0`; assert the weight
  change includes the `A3+·o2` triplet term (differs from the `A3=0` pair
  limit) and matches NEST.

Edge cases (each a test): Wmax clamp from below/above; `μ± ∈ {0,1}`; simultaneous
pre&post (the §2 exclusion vs NEST); first spike (zero traces); per-edge vs
multapse weights; triplet 2nd-trace decay; vogels target-rate fixed point
(`α=2τρ₀` ⇒ stationary weight at rate ρ₀); `dt` invariance (0.1 vs 0.05);
`vmap`/`grad` smoke on each kernel.

## 6. TDD phase plan (Red→Green→Refactor throughout)

**A. Substrate multi-trace seam** (`_event_plastic.py` + `_event_plastic_test.py`)
  - RED: synthetic 2-tau pre+post rule → assert `ctx.pre_traces`/`post_traces`
    `(E,2)`, correct columnwise decay-then-add + per-edge gather; back-compat
    (scalar → `(E,1)` + `ctx.pre_trace` unchanged; `None` → `(E,0)` + zeros).
  - GREEN: the §1 change. Regression: cluster-01's 6 specs' tests +
    existing substrate tests all green.

**B. Weight recording seam** (`_simulator.py` + `_simulator` test) — §4, TDD.

**C. The 7 specs** (`_nest/<model>_synapse.py` + per-rule unit tests), in order:
  `stdp_synapse` → `stdp_synapse_hom` (thin reuse) → `stdp_pl_synapse_hom` →
  `jonke_synapse` → `vogels_sprekeler_synapse` → `ht_synapse` → `stdp_triplet`
  (uses A). Each: RED host-side formula test at a controlled spike pair (expected
  from the NEST closed form) → GREEN kernel → REFACTOR shared bits into
  `_plastic_base` (`stdp_normalised_facilitate/depress`, copysign clamp helper).

**D. Parity** (`_validation/stdp_*_parity_test.py`) — §5, `requires_nest`.

**E. Wire + retire + docs**:
  - Export the 7 specs from `_nest/__init__.py`; route the package STDP names to
    the new specs; **retire the imperative STDP** from the active path (they move
    to / stay behind `_legacy_imperative`, per the `static_synapse` precedent);
    replace old eager STDP unit tests with spec + parity tests.
  - `tau_minus` (+`tau_minus_triplet`) divergence note in **every** docstring;
    NumPy docstrings + doctest examples (`>>> spec.edge_state_init()` etc.).
  - Coverage > 90 % (measure by directory path — NEST-coredump workaround from
    the 00 Lessons).

**F. Close-out**: `CONTEXT.md` Lessons entry (contract decision + the substrate
  grew a multi-trace seam — **flag for 05/07**; weight-trace API; per-rule parity
  numbers; NEST traps: pairing order, clamp timing, triplet taus, tau_minus-on-
  post). Git lifecycle: commit (**no `Co-Authored-By`**) → push → PR (scope +
  parity numbers) → merge → delete remote branch → `git switch main &&
  git merge --ff-only origin/main` → remove worktree.

## 7. File inventory

- **Edit** `_network/_event_plastic.py` (multi-trace seam),
  `_network/_simulator.py` (weight recording), `_network/__init__` if needed.
- **New** `_nest/{stdp_synapse,stdp_synapse_hom,stdp_pl_synapse_hom,
  stdp_triplet_synapse,vogels_sprekeler_synapse,jonke_synapse,ht_synapse}.py`
  (rebuilt specs; replace the legacy imperative modules' active role).
- **Edit** `_nest/_plastic_base.py` (shared STDP helpers), `_nest/__init__.py`
  (exports).
- **New tests**: `_event_plastic_test.py` (extend), `_simulator` weight-trace
  test, 7 `_nest/<model>_synapse_test.py`, `_validation/stdp_*_parity_test.py`.
- **Edit** `CONTEXT.md` (Lessons).

## 8. DoD ↔ plan map

| DoD item | covered by |
|---|---|
| 7 specs rebuilt; imperative retired; eager tests replaced | C, E |
| contract question resolved + reasoned in Lessons | §1, F |
| `res.weight_trace` seam + unit test | B |
| per-rule single-pair + 5 s parity (live/skip) | D |
| `tau_minus` divergence in every docstring; >90 %; doctests | E |
| Lessons entry; git lifecycle | F |
