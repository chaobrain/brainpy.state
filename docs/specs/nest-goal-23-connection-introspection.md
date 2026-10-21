# /goal — 23 · Connection introspection (`GetConnections`/`SynapseCollection`) → retire §3.4's blocked demos

> Self-contained prompt. Open at repo root. Stream: example (+ `_network` API seam). Depends on: 02/03 (Simulator + the 2 placeholders) — merged. Independent of 16. Builds the connection-enumeration/introspection API NEST users expect and retires cluster-03's two blocked §3.4 demos. Bucket-3-independent.

## Session start

1. Create an isolated worktree branch **`nest-goal/23-connection-introspection`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`03-recording-demos` Lessons** — the two blocked demos shipped as documented skipped placeholders (`plot_weight_matrices`, `synapsecollection`) with marker tests asserting the skip reason; the exact blocker recorded (no `GetConnections`/`SynapseCollection`).
   - the **`13-plasticity-demos` Lessons** — `record_weight(proj)`/`res.weight_trace(proj)` reads the realized edges from **`proj._pre_idx`/`proj._post_idx` in CSR sorted-by-pre order** (the enumeration substrate already exists per projection — this cluster surfaces it as a query API), and for `all_to_all` no-autapses that order is `[(i,j) for i in range(N) for j in range(N) if i!=j]`.
   - the **`14` Lessons** — the population-concat / `NodeView` idioms (`e+i`, slices) a `GetConnections(source, target)` query filters over.
   Then read:
   - `docs/nest-status/internal/network-api-gap.md` **§3.1 (`GetConnections` — missing)**, **§3.8 (`SynapseCollection` — missing)**, §3.9/§3.11 (named rules / `Parameter` weights), and the roadmap line ~325 (*enumeration + default-introspection helpers*). The §3.1 note: projections expose `.weight`/`.delay` directly; brainpy.state "requires holding a reference to the `Projection`" — this cluster removes that requirement.
   - `examples/nest/plot_weight_matrices.py` + `examples/nest/synapsecollection.py` (the placeholders + their `_BLOCKED_REASON` + marker tests) and the NEST upstream `plot_weight_matrices.py` / `synapsecollection.py` they mirror.
   - The substrate: `brainpy_state/_network/_simulator.py` (how the Simulator stores projections; `_pre_idx`/`_post_idx`/`weight`/`delay` on the projection objects), `_network/{_event_proj.py,_event_plastic.py,_projections.py}` (the projection types a query must span: static event, plastic, rule-built), `_network/_nodeview.py` (the `NodeView` a source/target filter uses).
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the `GetConnections`/`SynapseCollection` API shape is the decision to sign off.

## Goal

Add a connection-enumeration + introspection API (`sim.get_connections(...)` → a `SynapseCollection` over realized edges, with `.get`/`.set` of weight/delay/source/target), so a NEST user introspects connections **without** holding each `Projection` reference, and port the two blocked §3.4 demos onto it.

## Scope + the design question

### A. `GetConnections` + `SynapseCollection` (`_network/`)  ← **core design question**

- **`sim.get_connections(source=None, target=None, synapse=None)`** → enumerate the realized edges across **all** registered projections (static / plastic / rule-built), filtered by `source`/`target` `NodeView` membership and optional synapse-model/handle, returned as a **`SynapseCollection`**. Decide and justify (record in Lessons): the SynapseCollection's backing representation — a **light view** carrying `(projection, edge-index)` pairs (lazy, reads `proj._pre_idx/_post_idx/weight/delay` on demand) vs a **materialized** `(source, target, weight, delay)` table. Recommendation: the **lazy view** (no copy; `.get('weight')` reads the live projection State, so it reflects post-simulation plastic weights — exactly what `plot_weight_matrices` wants), with a `.get(...)`-materializes-to-arrays escape hatch.
- **`SynapseCollection.get(key)`** for `key ∈ {'source','target','weight','delay', …}` → arrays in a **stable, documented edge order** (CSR sorted-by-pre per projection, concatenated in registration order). **`.set('weight'/'delay', …)`** writes back to the projection State (guard: refuse to set a per-edge weight on a homogeneous-weight projection; refuse plastic-managed fields with a clear error). `len()`, iteration, and a `__repr__` mirroring NEST's table.
- Keep it **additive** — existing `proj.weight`/`.delay` direct access untouched; the API is a convenience layer over the same State.

### B. Port the two blocked §3.4 demos

- `plot_weight_matrices` → `sim.get_connections(pop_pre, pop_post)` → reshape `.get('weight')` (+ source/target) into the **dense weight matrix** per (E→E, E→I, …) block and plot; assert the realized matrix vs the constructed connectivity (and vs NEST's `GetConnections` weights where NEST is available). **Delete the placeholder + its skip.**
- `synapsecollection` → the introspection-idioms tour: query by source/target, read weight/delay, the **minimum** `Parameter`-weight / named-rule support the NEST script exercises (§3.9/§3.11 — implement only what the demo needs, document the boundary). **Delete the placeholder + its skip.**

## Constraints

- `_network` compute; additive convenience layer (no change to how projections store edges); the query is **pure indexing** over existing State (`vmap`/`for_loop`-irrelevant — it's a host-side introspection API, but `.get('weight')` must read the **live** State so post-sim plastic weights are correct).
- `Simulator` API only in the demos; `examples/nest/` house style; `saiunit` units.
- Reproduce NEST's `GetConnections` **semantics** (edge ordering, source/target filtering, the source-then-target enumeration) closely enough that the demos match; document any ordering divergence.
- TDD (`superpowers:test-driven-development`): the enumeration/order/`.set` guards get NEST-free unit tests first; the demos get live-NEST parity where NEST exposes the same query.
- NumPy docstrings; >90 % coverage on the new API. Coverage via `coverage run --source=brainpy_state` + the coverage API.

## Validation (via the 00 harness)

`_network/_connection_introspection_test.py` (NEST-free: enumeration correctness, edge order, source/target filter, `.get`/`.set` round-trip, the homogeneous/plastic `.set` guards, post-simulation live-weight read) + `_nest/_validation/{plot_weight_matrices,synapsecollection}_test.py`, `requires_nest` (the realized weight matrix / queried weights match NEST's `GetConnections`). The two former marker/skip tests are **replaced** by real parity tests. README §3.4 rows flip blocked→implemented.

## Edge cases to enumerate + test

Empty result (no matching connections → empty SynapseCollection, not an error); a query spanning multiple projections (order = registration order, CSR within); `source`/`target` as a sliced/concatenated `NodeView`; homogeneous-weight projection (`.get('weight')` broadcasts; `.set` per-edge refused); plastic projection after simulation (`.get('weight')` reflects the **evolved** weight, not the init); multapses (repeated `(i,j)` enumerated distinctly); `.set('delay')` where the substrate rounds to the grid (document); a projection with no delay.

## Definition of done

- [ ] `sim.get_connections(...)` + `SynapseCollection` (`.get`/`.set`/`len`/iterate/`__repr__`) landed in `_network/`, additive + TDD-covered (enumeration, order, filter, guards, live-weight read).
- [ ] `plot_weight_matrices` + `synapsecollection` ported onto it; both placeholders + skip markers deleted; README §3.4 flipped to implemented.
- [ ] Live-NEST parity where NEST exposes the same query; ordering divergence (if any) documented.
- [ ] >90 % coverage on the new API; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the `GetConnections`/`SynapseCollection` API shape (backing representation, edge-order contract, `.set` guards), the minimum `Parameter`/named-rule support added, how `.get('weight')` reads live plastic State, the parity outcome, and confirmation that **§3.4 is fully complete** (no remaining blocked recording demos). Note any `network-api-gap.md` items still open (full `Parameter`/`Mask`/spatial) for a future cluster.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
