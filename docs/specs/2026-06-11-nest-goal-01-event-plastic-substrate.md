# /goal — 01 · Event-plastic substrate + `EventPlasticProj` + static/STP rebuild

> Self-contained prompt. Open at repo root. Stream: synapse (foundation). Depends on: 00.

## Session start

1. Create an isolated worktree branch **`nest-goal/01-event-plastic-substrate`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — Part 2 (architecture) and Part 3 (repo map) are authoritative for this file. Then read:
   - `docs/nest-status/internal/synapses-plasticity-gap.md` §5 (semantic risks) + the STP rows.
   - **Substrate to build on:** `brainpy_state/_network/_projections.py:79-140` (`_SparseEventMatMul` — the `brainevent.CSR` event-matmul + per-edge weight `ParamState` pattern), `_event_proj.py` (delay seam + scatter), `brainpy_state/_brainpy/_delay.py` (`InputDelay`).
   - **Models to rebuild (study the NEST math, then discard the imperative impl):** `_nest/static_synapse.py`, `static_synapse_hom_w.py`, `tsodyks_synapse.py`, `tsodyks_synapse_hom.py`, `tsodyks2_synapse.py`, `quantal_stp_synapse.py`.
   - NEST upstream model docs for `tsodyks_synapse` / `tsodyks2_synapse` / `quantal_stp_synapse` and the example `evaluate_tsodyks2_synapse.py` (drive pattern).
3. Write spec + plan under ``. **Approach-first; wait for approval before code.**

## Goal

Build the **first of the three rebuilt primitives** (`CONTEXT.md` Part 2.5) and prove it end-to-end by rebuilding the simplest static synapse and the entire STP family on it — JAX-native, jit/vmap-safe, live-NEST-validated. This file establishes the pattern every later synapse cluster reuses; get the substrate right.

## Scope

### A. `_network/` substrate + `EventPlasticProj`

A new module in `brainpy_state/_network/` providing:

- **Shared substrate:** CSR edge layout (reuse the `_SparseEventMatMul` construction — sort edges by pre, `indices`/`indptr`), `InputDelay` for the homogeneous axonal delay, and a **rule-declared `State` allocator**: a rule announces the per-edge State it needs (e.g. STP `u, x`), the per-pre-neuron trace State, and the per-post-neuron trace State; the primitive allocates `brainstate.State` arrays of the right shapes and threads them through the jitted step.
- **`EventPlasticProj`** — each step: (1) read pre spikes (delayed) and post spikes from the Simulator's `_SpikeHolder`; (2) update per-neuron traces `trace ← trace·exp(-dt/τ) + spike`; (3) call the **rule kernel** `update(state, pre_spike, post_spike, dt) -> (state, effective_weight)`; (4) deliver `effective_weight` events to post via the `brainevent.CSR` event matmul + scatter. Per-edge weight is a mutable `State` (plastic rules write it; static rules leave it). No Python queue, no history list, no host scalars in the step.

### B. `_nest/` rebuilt models (spec + pure rule)

Rewrite each target down to: NEST-faithful **parameter spec** (names/defaults/units/init-validation, fidelity wins per `CONTEXT.md` 2.4) **+ a pure `update` rule kernel** consumed by `EventPlasticProj`. **Delete** the `defaultdict(list)` queue, the history lists, `send()`/`record_post_spike()`, and the host-scalar math.

- `static_synapse`, `static_synapse_hom_w` — trivial rule (constant weight); proves delivery + delay parity.
- `tsodyks_synapse`, `tsodyks_synapse_hom` — per-edge `u, x, y`; NEST update equations.
- `tsodyks2_synapse` — multiplicative scaling, `tau_fac == 0` special case (NEST convention).
- `quantal_stp_synapse` — probabilistic release; **distributional** parity (PRNG differs).

### C. Simulator wiring

Extend `sim.connect(...)` to accept `synapse=<model>` so a plastic projection is built (alongside the existing static `EventProjection` path). Keep the existing weight=/delay= static path working.

## Constraints (from `CONTEXT.md` Part 2)

- `_network/` owns compute; `_nest/*_synapse.py` owns spec + pure rule. brainevent-only — **do not** import `_brainpy/`'s `Projection`/`STP`/`AlignPostProj`.
- NEST fidelity wins; document any convention delta in the docstring **and** the `CONTEXT.md` Lessons entry.
- Everything in the step must be `jit`/`vmap`/`for_loop`-safe: fixed-shape `State`, no Python-list growth, no data-dependent host control flow, no `np`/`math`/`float()` in the hot path.

## Edge cases to enumerate + test

`tau_fac == 0` (tsodyks2); first spike (uninitialized traces); `dt` invariance; multapses (multiple edges same pre→post) summing correctly; delay = integer and the buffer wrap; zero-edge projection; static-weight conservation (no drift); `vmap` over a batch; a gradient flows through `effective_weight` (smoke `grad`).

## Definition of done

- [ ] `EventPlasticProj` + substrate in `_network/`, with unit tests (State shapes, trace decay, delivery == dense reference, jit/vmap/grad smoke).
- [ ] `static`(+`_hom_w`) and the 4 STP models rebuilt as spec+rule; imperative machinery deleted; old eager `*_test.py` replaced.
- [ ] `sim.connect(..., synapse=...)` builds plastic projections.
- [ ] Live-NEST parity in `_validation/` via the `00` harness: STP **PSC-amplitude train** vs NEST (`trace`); `quantal_stp` **distributional**; static **delivery** parity. Tolerances documented.
- [ ] >90% coverage; NumPy docstrings (doctest examples).
- [ ] Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the exact `EventPlasticProj` constructor + rule-kernel signature (the **contract `04`/`05`/`06` will reuse**), the State-declaration mechanism, STP parity numbers, and any NEST-fidelity gotcha (release timing, `y` reset, weight sign).
2. Verify all tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
