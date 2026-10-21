# /goal — 04 · STDP core family (spec-only, on the frozen `EventPlasticProj` contract)

> Self-contained prompt. Open at repo root. Stream: synapse. Depends on: 01 (merged). Critical path for 13 (plasticity demos).

## Session start

1. Create an isolated worktree branch **`nest-goal/04-stdp-core`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially Part 2 and the **`01-event-plastic-substrate` Lessons entry** (the frozen rule contract you build on) and the **`00-validation-harness` Lessons** (`compare_trace`+`CAT_A` for 5 s weight trajectories). Then read:
   - `docs/nest-status/internal/synapses-plasticity-gap.md` §5 (trace-storage divergence, pairing, Wmax clipping) + the `stdp_*` rows.
   - The substrate + contract: `brainpy_state/_network/_event_plastic.py` (`EventPlasticProj`, `KernelContext`), `brainpy_state/_nest/_plastic_base.py`, and the rebuilt specs from 01 (`tsodyks2_synapse.py`, `static_synapse.py`) as worked examples of "spec + `update(state, ctx)`".
   - The models you are rebuilding (study NEST math; their imperative impl is now in/redirected via `_nest/_legacy_imperative.py`): `stdp_synapse`, `stdp_synapse_hom`, `stdp_pl_synapse_hom`, `stdp_triplet_synapse`, `vogels_sprekeler_synapse`, `jonke_synapse`, `ht_synapse`.
   - NEST upstream model docs for each + the `clopath`/`stdp` math in the NEST source.
3. Write spec + plan under ``. **Approach-first; wait for approval before code.**

## Goal

Rebuild the **STDP core family** as pure parameter-spec + `update` rule kernels on the **already-frozen** `EventPlasticProj` contract (01) — no imperative code, validated by 5 s weight-trajectory parity vs live NEST. Per 01's Lessons, "build new plasticity by writing specs only": the per-pre/per-post online trace seams are wired and tested. Your job is the seven rules + their NEST-faithful parameters, **plus** resolving the one open contract question below.

## Scope

Seven `_nest/*_synapse.py` specs (attrs + `edge_state_init()` + `update(state, ctx)->(new_state, w_eff)`), each consumed via `sim.connect(pre, post, synapse=<spec>)`:

- `stdp_synapse` — canonical all-to-all (Morrison 2008): pre-trace `K+` (`pre_trace_tau=tau_plus`), post-trace `K-` (`post_trace_tau=tau_minus`); **depression on pre-spike** (uses `K-`), **potentiation on post-spike** (uses `K+`); weight clamped to `[0, Wmax]` in-update; `mu_plus/mu_minus` exponents.
- `stdp_synapse_hom` — homogeneous-param variant (params via common-properties, like 01's `tsodyks_synapse_hom`; set with `nest.SetDefaults` on the NEST side).
- `stdp_pl_synapse_hom` — power-law STDP.
- `stdp_triplet_synapse` — Pfister-Gerstner: **two pre traces (`r1,r2`) + two post traces (`o1,o2`)** with distinct taus. ← see the contract question.
- `vogels_sprekeler_synapse` — symmetric inhibitory STDP with target-rate offset `alpha`.
- `jonke_synapse` — STDP with additive factors.
- `ht_synapse` — Hill-Tononi depression.

### The one open contract question (resolve in approach-first, record in Lessons)

01 flagged: *"If a rule needs cross-edge or per-target reductions beyond the per-edge `update`, that is the first substrate extension point."* The `pre_trace_tau`/`post_trace_tau` seams give **one** trace per side. `stdp_triplet_synapse` needs **two per side**. Decide and justify:
- **(a) extend the substrate** to allow a list of per-side trace taus (cleanest if more rules will need it — triplet, and later clopath filters), or
- **(b) carry the second trace pair as per-edge `edge_state_init()` state**, updated manually in the kernel from `ctx` (no substrate change, slightly less efficient, per-edge not per-neuron).
Recommendation: **(a)** — a small, well-tested generalization (`pre_trace_taus: tuple`, gathered per edge) that 07's Clopath voltage-filters will also want; keep the single-tau attr working as the 1-tuple case. But measure against 01's "no substrate change" ideal and pick deliberately.

## Constraints (from `CONTEXT.md` Part 2 + 01 Lessons)

- brainevent-only; `_network` owns compute, `_nest` owns spec+rule. **NEST fidelity wins.**
- Kernel gates its **own** writeback: depression on `ctx.pre_spike`, potentiation on `ctx.post_spike` (the substrate gates only *delivery*). Use the `frozen(fired, new, old)` helper per side.
- **`tau_minus`-on-post divergence** (Part 2.4): in NEST `tau_minus` lives on the post neuron; here it's a synapse attr. Keep the synapse-attr design, document the delta in every docstring (cluster 10 writes the user-facing page; you write the docstring note + a `CONTEXT.md` Lessons line).
- jit/vmap/for_loop-safe; thread `t/i` as scanned args (01 gotcha); no host scalars in the hot path. Propagator/exponent forms kept verbatim per NEST source (FP-distinct).
- NumPy docstrings (doctest examples), >90 % coverage.

## Weight-trajectory recording (likely small `_network` seam)

Parity here compares the **weight over time**, not a V_m trace. 02 built analog State-tap recording for neuron attributes and noted "weight recording reuses the analog-recording (State-tap) mechanism." Add the minimal seam to record a plastic projection's weight `State` across the run (e.g. `res.weight_trace(proj)` / tapping the projection's `ParamState`), mirroring `res.trace`. Cluster 09 (weight-recorder audit) builds on this — keep it general.

## Validation (via the 00 harness)

`_validation/stdp_*_parity_test.py`, `requires_nest`:
- **Single-spike-pair regression** per rule: one pre/post pair at Δt, assert the post-update weight matches NEST (`compare_trace`, `CAT_B`-tight where deterministic).
- **5 s Poisson-driven weight trajectory** vs NEST: `compare_trace` + `CAT_A` (1e-3). Drive plastic synapses in NEST via `spike_generator → parrot_neuron → synapse` (01 Lesson); inject the same spike steps on the brainpy side.
- `stdp_triplet`: assert the triplet term (not just the pair limit).

## Edge cases to enumerate + test

`Wmax` clamp hit from below/above; `mu_plus/mu_minus ∈ {0,1}` (additive vs multiplicative limits); simultaneous pre&post spike (ordering convention vs NEST); first spike (uninitialized traces); homogeneous vs per-edge weight; triplet second-trace decay; vogels target-rate fixed point; `dt` invariance; multapses; `vmap`/`grad` smoke.

## Definition of done

- [ ] 7 STDP specs rebuilt as spec+rule; imperative versions retired from the active path; old eager tests replaced by spec + parity tests.
- [ ] Contract question resolved (substrate extension *or* per-edge state), with the decision + reasoning in the Lessons entry.
- [ ] `res.weight_trace`-style weight recording seam added + unit-tested.
- [ ] Per-rule single-pair + 5 s trajectory parity passing (live NEST) / skipping (without); tolerances documented.
- [ ] `tau_minus` divergence noted in every docstring; >90 % coverage; doctest examples.
- [ ] Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the contract decision (and whether the substrate grew a multi-trace seam — **05 and 07 depend on this answer**), the weight-trajectory recording API, per-rule parity numbers, and any NEST-fidelity trap (pairing order, clamp timing, triplet trace taus).
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
