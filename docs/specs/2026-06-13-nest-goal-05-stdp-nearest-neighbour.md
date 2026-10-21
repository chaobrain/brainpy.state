# /goal — 05 · STDP nearest-neighbour family + facetshw (spec-only on `EventPlasticProj`)

> Self-contained prompt. Open at repo root. Stream: synapse. Depends on: 04 (merged).

## Session start

1. Create an isolated worktree branch **`nest-goal/05-stdp-nearest-neighbour`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially the **`04-stdp-core` Lessons** (the frozen multi-trace contract, the `_stdp_drive.py` parity harness, `res.weight_trace`, and the four hard gotchas you will hit again: parrot-relay archive pollution, dendritic-delay convention, simultaneous-spike exclusion via the trace's own `+1`, `tau_minus`-on-post). Then read:
   - `docs/nest-status/internal/synapses-plasticity-gap.md` §5 (per-`stdp_nn_*` pairing-convention rows, P1) + the `stdp_nn_*` / `stdp_facetshw` rows.
   - The frozen contract + a worked multi-trace example: `brainpy_state/_network/_event_plastic.py`, `brainpy_state/_nest/stdp_triplet_synapse.py` (multi-trace user), `_nest/stdp_synapse.py` (single-trace), `_nest/_plastic_base.py`, and `_validation/_stdp_drive.py`.
   - The models (study NEST pairing math; they currently sit on `_nest/_legacy_imperative.py`/`_legacy_stdp_synapse`): `stdp_nn_pre_centered_synapse`, `stdp_nn_restr_synapse`, `stdp_nn_symm_synapse`, `stdp_facetshw_synapse_hom`.
   - NEST upstream source for each NN pairing scheme (cite the exact source lines in each docstring).
3. Write spec + plan under ``. **Approach-first; wait for approval before code.**

## Goal

Port the **nearest-neighbour STDP variants** + the hardware-quantised `stdp_facetshw_synapse_hom` from the legacy imperative base onto the frozen `EventPlasticProj` spec+rule pattern, validated by single-spike-pair regression (the pairing-scheme proof `synapses-plasticity-gap.md` §5 P1 demands) + train-trajectory parity.

## Scope + the core design question

Three NN specs + one quantised spec (`attrs` + `edge_state_init()` + `update(state, ctx)->(new_state, w_eff)`):

- `stdp_nn_pre_centered_synapse`, `stdp_nn_restr_synapse`, `stdp_nn_symm_synapse` — three **distinct NEST nearest-neighbour pairing schemes**. Each is subtle; reproduce the exact scheme and assert it.
- `stdp_facetshw_synapse_hom` — FACETS hardware model: **quantised weights / lookup-table updates**. The odd one (discretisation, not a trace rule).

**The core design question (resolve in approach-first, record in Lessons):** `04`'s substrate trace is **decay-then-add** = *all-to-all* pairing. Nearest-neighbour pairing pairs each spike only with the *nearest* partner — i.e. the relevant trace is **reset/saturated to 1 on each spike**, not accumulated. Decide and justify:
- **(a)** add a per-side substrate **trace mode** (`'nearest'` → set-to-1-on-spike / saturate, vs the default `'all_to_all'` decay-then-add), a small generalisation analogous to `04`'s multi-trace seam (likely reused by other NN-pairing rules), **or**
- **(b)** carry per-edge last-spike state in `edge_state_init()` and compute the pairing in the kernel (no substrate change, per-edge).
Recommendation: **(a)** if the three schemes share one reset semantics cleanly; otherwise **(b)** for the schemes that differ. The three NEST schemes do **not** all reset identically — map each precisely before choosing; a hybrid (substrate `nearest` mode + per-scheme kernel masking) is acceptable. `stdp_facetshw` is weight-quantisation, orthogonal to pairing — keep its discretisation in the kernel/`edge_state_init`.

## Constraints (from `CONTEXT.md` Part 2 + 04 Lessons)

- brainevent-only; `_network` compute, `_nest` spec+rule; **NEST fidelity wins**; jit/vmap/for_loop-safe.
- Reuse `04`'s machinery verbatim: the `_stdp_drive.py` harness (strong-driven `iaf_psc_delta` post to avoid parrot-archive pollution; record actual pre/post fire steps and replay), the dendritic-delay convention (`rule.delay=None`, inject post `d` later), simultaneous-spike exclusion (`kplus = ctx.pre_trace - ctx.pre_spike`), `res.weight_trace(proj)` for trajectories.
- `*_hom` plasticity params are **common** (CopyModel/SetDefaults), not per-connection (04 gotcha).
- `tau_minus`-on-post divergence: document in each docstring; the parity drive sets the post node's `tau_minus` to match.
- NumPy docstrings (cite the NEST pairing source line per variant); >90 % coverage.

## Validation (via the 00 harness + `_stdp_drive`)

`_validation/stdp_nn_*_parity_test.py`, `requires_nest`:
- **Single-spike-pair regression per scheme** (the §5 P1 deliverable): assert the pairing convention exactly (`compare_trace`, `CAT_B`).
- **Multi-spike train** where all-to-all and nearest-neighbour *diverge* (≥3 spikes one side): assert the NN scheme, not the a2a limit (`CAT_A`).
- `stdp_facetshw`: quantised-weight trajectory + a lookup-table-boundary case.

## Edge cases to enumerate + test

Three consecutive pre spikes between one post (where schemes differ); coincident pre&post; first spike; `Wmax` clamp; facetshw quantisation boundary / saturation; per-edge vs per-neuron reset correctness; `dt` invariance; multapses; `vmap`/`grad` smoke.

## Definition of done

- [ ] 4 specs ported off the legacy base onto spec+rule; legacy versions retired from the active path; old eager tests replaced.
- [ ] Design question resolved (substrate `nearest` mode and/or per-edge), decision + reasoning in the Lessons entry.
- [ ] Single-pair pairing-scheme regression + divergent-train parity passing (live NEST) / skipping (without); each docstring cites the NEST pairing source.
- [ ] >90 % coverage; doctest examples; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: whether the substrate grew a `nearest` trace mode (and its API), the precise per-scheme pairing mapping, facetshw quantisation handling, parity numbers, and any remaining STDP variant still on the legacy base.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
