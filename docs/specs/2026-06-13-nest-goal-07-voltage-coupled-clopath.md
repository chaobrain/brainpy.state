# /goal — 07 · `VoltageCoupledPlasticProj` (primitive #2) + `clopath_synapse`

> Self-contained prompt. Open at repo root. Stream: synapse. Depends on: 04 (merged). Builds primitive #2 of the typed family; unblocks 08 (dopamine) and 13 (Clopath examples).

## Session start

1. Create an isolated worktree branch **`nest-goal/07-voltage-coupled-clopath`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — Part 2.5 (the three-primitive family; this builds #2) and the **`04-stdp-core` Lessons**, which explicitly teed this up: *"05-Clopath layers a per-post voltage filter on the same per-neuron multi-trace storage (voltage is a neuron State, not reconstructable per-edge)."* (In the index this is cluster **07**, not 05.) Then read:
   - `docs/nest-status/internal/synapses-plasticity-gap.md` (the `clopath_synapse` row; "needs a Clopath-capable postsynaptic neuron").
   - The substrate to extend: `brainpy_state/_network/_event_plastic.py` (`EventPlasticProj`, `KernelContext`, the multi-trace `(N,k)` per-neuron seam + per-edge gather), `_simulator.py` (`weight_trace`/`record_weight`, the `connect(synapse=)` path), `_validation/_stdp_drive.py`.
   - The Clopath-capable post neurons — **verify they are JAX-native and expose their filtered voltages as State** (`u_bar_minus`/`u_bar_plus` or equivalent): `brainpy_state/_nest/aeif_psc_delta_clopath.py`, `hh_psc_alpha_clopath.py` (both validated per the gap doc). Read how they compute the low-pass voltages.
   - `brainpy_state/_nest/clopath_synapse.py` (NEST math; currently on the legacy base) + NEST upstream `clopath_synapse` source + the example `clopath_synapse_spike_pairing.py` (the regression `13` will port; you validate the model).
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the primitive's post-state-reader shape is the key decision; get sign-off.

## Goal

Build the **second primitive**, `VoltageCoupledPlasticProj` — `EventPlasticProj` plus a **post-neuron analog-state reader** — and prove it on `clopath_synapse` against live NEST. Per `04`, the per-post storage is the foundation: the new capability is **sampling a post-neuron State (voltage / filtered voltage) per edge each step**, which a spike-driven trace cannot reconstruct.

## Scope

### A. `VoltageCoupledPlasticProj` (`_network/`)

Extend the `EventPlasticProj` substrate with a **post-state reader**: a rule declares which post-neuron State variables it samples (e.g. `post_state_reads = ('u_bar_minus', 'u_bar_plus', 'V')`); each step the primitive **gathers those per-post-neuron State columns into `ctx` per edge** (sorted-by-pre edge order, exactly like the post-trace gather), unit-stripped to mantissas. Reuse the multi-trace `(N,k)` per-neuron machinery and the `04` weight-recording/`_stdp_drive` seams unchanged. Keep `EventPlasticProj` (primitive #1) intact — `VoltageCoupledPlasticProj` is a superset, not a rewrite.

### B. `clopath_synapse` spec + rule (`_nest/`)

Rewrite as parameter-spec + pure `update(state, ctx)->(new_state, w_eff)`:
- **LTD** on pre-spike: `−A_LTD · (u_bar_minus − θ_minus)+` (post filtered-voltage read).
- **LTP** on post activity: `+A_LTP · x_bar · (V − θ_plus)+ · (u_bar_plus − θ_minus)+` (pre-trace `x_bar` via the multi-trace seam; `V` and `u_bar_plus` via the post-state reader).
- `Wmax`/`Wmin` clamp; map NEST's exact parameter names/defaults/units. Keep the formula **verbatim** to NEST (FP-distinct).

### C. Wire `sim.connect(pre, post, synapse=clopath_synapse, …)` to build a `VoltageCoupledPlasticProj` when the spec declares `post_state_reads`.

## Constraints (from `CONTEXT.md` Part 2 + 04 Lessons)

- brainevent-only; `_network` compute, `_nest` spec+rule; **NEST fidelity wins**; jit/vmap/for_loop-safe; no host scalars in the hot path.
- The post neuron's filtered voltages are **read-only State gathers** — do not mutate the neuron from the synapse. If a Clopath neuron does not already expose the filters as State, prefer **adding the filter to the neuron** (it is a post-neuron quantity in NEST's `ArchivingNodeClopath`) over filtering inside the synapse — but record the choice in Lessons.
- Reuse `04`'s parity scaffolding: strong-driven post to avoid parrot pollution (adapted — here the post is a *Clopath* neuron whose real voltage matters, so drive it to produce the intended `V(t)` and record fire steps), dendritic-delay convention, `res.weight_trace`.
- `tau_minus`/filter-time-constants are **post-neuron** params in NEST — document the divergence; the parity drive sets them on the post node.
- NumPy docstrings (doctest); >90 % coverage.

## Validation (via the 00 harness)

`_validation/clopath_synapse_parity_test.py`, `requires_nest`:
- **Spike-pairing regression** (mirrors `clopath_synapse_spike_pairing.py`): pre/post pairs at a range of Δt and frequencies; assert the weight change vs NEST (`compare_trace`, `CAT_A`/`CAT_B`). The post must be a Clopath-capable neuron whose `V(t)`/filtered voltages match NEST — verify the **neuron** voltage parity first (reuse `02`'s `res.trace`) so a synapse mismatch isn't masked by a neuron mismatch.
- **Voltage-clamp sanity:** hold post voltage at a constant and check LTD/LTP cross the `θ_minus`/`θ_plus` thresholds as NEST does.

## Edge cases to enumerate + test

`V` exactly at `θ_plus`/`θ_minus` (the `(·)+` rectifiers); no post spikes (pure LTD); `Wmax`/`Wmin` clamp; post-state reader with a multi-segment / sliced post population; gather order matches CSR edge order; neuron-voltage parity isolated from synapse parity; `dt` invariance; `vmap`/`grad` smoke.

## Definition of done

- [ ] `VoltageCoupledPlasticProj` in `_network/` (post-state reader on the multi-trace substrate), unit-tested (gather correctness, jit/vmap, primitive-#1 untouched).
- [ ] `clopath_synapse` rebuilt as spec+rule using the post-state reader; legacy version retired from the active path.
- [ ] `sim.connect(synapse=clopath_synapse)` builds the voltage-coupled projection.
- [ ] Clopath-neuron voltage parity verified, then synapse spike-pairing parity passing (live NEST) / skipping (without); tolerances documented.
- [ ] >90 % coverage; doctest examples; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the `VoltageCoupledPlasticProj` post-state-reader API (the exact `post_state_reads` declaration + `ctx` field names) that **08-dopamine and the Urbanczik cluster reuse**, whether filters live on the neuron or the synapse, the Clopath parity numbers, and any neuron-side fix needed to expose the filtered voltages.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
