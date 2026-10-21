# /goal — 20 · §3.9 spatial connectivity (`_nest_spatial/` submodule) + spatial/CSA demos

> Self-contained prompt. Open at repo root. Stream: example **+ a new substrate submodule**. **XL** — this builds the *only* genuinely-absent capability left in the backlog: a spatial-connectivity layer (`nest.spatial.*` has no equivalent yet). Per the architecture directive it lives in a **NEW `../../brainpy_state/_nest_spatial` submodule** (a sibling of `_nest/`/`_network/`, NOT inside `_nest/`). Depends on: `02` (Simulator `create`/`connect`) + the existing `ConnRule` machinery (`_network/_rules.py`, `_connectivity.py`). Independent of bucket-3 and of `19`/`24`.

## What's absent + the integration seams (read first)
A codebase audit (the connectivity architecture) establishes:
- **No spatial concept exists** — `NodeView`/`_Segment` (`_network/_nodeview.py`) carry only `(population, indices)`; there is no position/coordinate/layer notion anywhere (`grep position|coord|grid|spatial` → nothing relevant). This submodule introduces it.
- **The rule contract** (`_network/_rules.py:18-27`): a `ConnRule.sample(n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses) -> ConnSpec` where `ConnSpec(pre_idx, post_idx, n_edges)` (`_connectivity.py:23-27`). Existing rules (`pairwise_bernoulli`, `fixed_indegree`, `all_to_all`, …) all return this. **A spatial rule is just a `ConnRule` subclass** that additionally needs the pre/post coordinates — `Simulator.connect(..., rule=<spatial rule>)` needs **no signature change**; the only seam is *how positions reach the rule's `sample`*.
- **Module/namespace layout**: top-level `../../brainpy_state/__init__.py` re-exports `_network` + `_nest`; `_nest/__init__.py` re-exports one model per file. Mirror this for `_nest_spatial/__init__.py`. Per CLAUDE.md's two-namespace rule, **inner source/tests import `brainpy_state` directly**; the **examples use the public `brainpy.state` surface** (the spatial API must be re-exported onto `brainpy.state`).

## Session start
1. Create an isolated worktree branch **`nest-goal/20-spatial`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`14`-network-demos Lessons** — distributional parity (seed-mean/median band + the qualitative law): spatial connectivity is a *probabilistic sample*, so the parity target is the **connection-probability-vs-distance profile** + the **degree-vs-distance** law, not a per-edge identity.
   - the **`16`-generator-demos Lessons** — the demo-cluster house style (runnable `examples/nest/` script + one test module with a NEST-free structural class **and** a `skipUnless(nest)` class; `from examples.nest.<module>`; no-NEST companion).
   - the **`23` connection-introspection Lessons** + the **`15b` Lessons** — `get_connections()`/the projection edge-State view (to assert the realized spatial adjacency) and `comm='dense'` vs `'sparse'` deposit semantics.
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.9 rows + the P2 acceptance ("at least one spatial example ports cleanly"); `network-api-gap.md` §3.10 spatial roadmap (the recorded blocker this cluster clears).
   - The connectivity machinery: `brainpy_state/_network/{_rules.py,_connectivity.py,_connectivity_test.py,_nodeview.py}`; `_simulator.py` `create()` + `connect()` + `_connect_pair()`.
   - The module pattern: `brainpy_state/_nest/__init__.py` + top-level `../../brainpy_state/__init__.py` (the re-export sites).
   - NEST upstream (clone on this machine: **`/mnt/d/codes/githubs/computational_neuroscience/nest-simulator`**; the 17-Lessons `…/githubs/nest-simulator` path is machine-specific and **absent here** — `ls`-verify): `pynest/examples/spatial/{grid_iaf,grid_iaf_irr,gaussex,connex,connex_ew,test_3d,test_3d_gauss}.py`; `csa_example.py`; `csa_spatial_example.py`. The `nest.spatial` API surface: `spatial.grid(shape,extent)`, `spatial.free(pos|distribution,extent)` (2D/3D), `spatial.distance`, `spatial_distributions.gaussian(distance,std)`, masks `{"circular":{"radius"}}`/`{"box":{"lower_left","upper_right"}}`/rectangular, `Connect(a,b,{"rule":"pairwise_bernoulli","p":<kernel>,"mask":…,"allow_autapses":…})`, helpers `GetPosition`/`Distance`/`FindCenterElement`/`GetTargetPositions`/`GetTargetNodes`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the sign-off is **design questions A + B + the build/demo decomposition** below.

## Goal
Build a `../../brainpy_state/_nest_spatial` submodule providing spatially-embedded populations (grid/free positions, 2D & 3D), distance-dependent connectivity (gaussian kernel + spatial masks) as a `ConnRule` that plugs into the existing `Simulator.connect(rule=…)`, and port a representative `spatial/` subset + the CSA-spatial demo on it — with distributional (probability-vs-distance) + structural (mask cutoff, grid coords) parity vs live NEST `nest.spatial.*`.

## Scope + the core design questions

### A. The positions seam — how coordinates attach + reach the rule (**resolve FIRST**, record in Lessons)
**Try:** `Simulator.create(model, n, positions=<Layer>)` accepts a positions object and stores the resulting `coords[n, d]` (d∈{2,3}) in a per-population side-registry on the Simulator (positions live **on the population**, NEST-faithful — not on every `NodeView`). At `connect()`, when the `rule` is a spatial rule, `_connect_pair` looks up the pre/post populations' coords, slices them by the segment indices, and supplies them to the rule: `rule.sample(n_pre, n_post, *, key, pre_pos=…, post_pos=…, …)`. **Recommend (a):** positions-on-population + a spatial-rule branch that injects `pre_pos`/`post_pos` into `sample` — keeps non-spatial `connect` untouched and the rule pure. *(Alt (b): a `SpatialNodeView`/`Layer` wrapper carrying coords — heavier; prefer (a) unless a demo forces per-view coords.)*

### B. The `_nest_spatial/` primitive set (the minimum that ports the demos)
Build, mirroring `_network/_connectivity.py`+`_rules.py` style:
- **Position layers** — `grid(shape, extent)` (regular lattice coords; matches `spatial.grid`), `free(positions_or_distribution, extent)` (explicit array **or** a `brainstate`/`brainunit` random distribution; 2D **and** 3D). Return a `Layer`/coords object `create(positions=)` consumes.
- **Distance** — pairwise Euclidean `displacement`/`distance` (flat boundary first; `edge_wrap`/torus is a documented **non-goal v1** unless a demo needs it).
- **Kernel** — `gaussian(distance, std)` → callable `p(d)` (add `exponential`/`linear` only if a ported demo needs them; gaussian covers gaussex/connex/test_3d_gauss).
- **Masks** — `circular(radius)` (2D) + `spherical`/`box(lower_left, upper_right)` (2D/3D) hard cutoffs.
- **The rule** — `spatial_pairwise_bernoulli(p=<kernel>, mask=<mask>, allow_autapses=…)`: a `ConnRule` that, given `pre_pos`/`post_pos`, computes pairwise distances, applies `mask` (hard cutoff) × `p(d)` (probability), Bernoulli-samples under `key` → `ConnSpec`. Lower the pairwise distance + sampling with `jnp`/`brainstate.transform` (no Python double-loop; CLAUDE.md #10).
- **Helpers** (for demos/tests) — `distance(layer_a, layer_b)`, `center_element(layer)`, `target_positions`/`target_nodes` (read back the realized adjacency via `23`'s `get_connections`).

### C. Demo scope (the build/demo decomposition — **may split 20a layer / 20b demos**)
Port a **representative** `spatial/` subset (not all 14 — note which are dropped and why):
- **`grid_iaf`** → a `grid` layer; assert the grid coordinates exactly (structural).
- **`gaussex`** → two grid layers + `spatial_pairwise_bernoulli(p=gaussian(distance,std), mask=circular(radius))`; assert the **connection-probability-vs-distance** profile + the **mask hard-cutoff** (no edge beyond the radius) vs NEST, distributionally.
- **`test_3d_gauss`** → a `free` 3D layer + gaussian + `box` mask; assert the 3D target-distance histogram vs NEST.
- **`csa_spatial_example`** → **re-expressed natively** via the spatial gaussian rule (it is a Gaussian-distance connectivity; the `csa` library / `conngen` rule is **not** a dependency — document this).
- **`csa_example`** (pure random CSA) → re-express its *intent* as `pairwise_bernoulli(0.1)` with fixed weight/delay, **or** a documented skipped placeholder noting `csa`/`conngen`/libneurosim is an external optional dependency out of scope. **The literal CSA API stays a documented non-goal either way.**
- **Must-have** (the gap-doc acceptance floor): `grid_iaf` + `gaussex` port cleanly with parity. The 3D + box-mask + csa-spatial are the fuller set; **at approach-time you may split the `_nest_spatial/` layer + structural tests into `20a` and the demo ports + distributional parity into `20b`.**

## Constraints
- The spatial rule is an ordinary `ConnRule` → it rides the **existing** `Simulator.connect(rule=…)` with **no `connect` signature change** beyond the `create(positions=)` kwarg + the spatial-rule coord-injection branch (design A).
- `_nest_spatial/` is a **sibling submodule** (not under `_nest/`); inner source/tests `import brainpy_state` directly; the public spatial API is re-exported onto `brainpy.state`; examples use `from brainpy import state`. **brainunit** units (coordinates/extent carry length units).
- Distance + sampling lowered with `jnp`/`brainstate.transform` — **no Python pairwise double-loop** (CLAUDE.md #10). Seed via `connect(seed=)`→key for reproducible samples.
- Parity is **distributional for the sample** (probability-vs-distance band; degree-vs-distance law; the `14` posture) + **structural/exact** for deterministic pieces (grid coords; mask hard cutoff — zero edges beyond the boundary). Any fix **reproduce-first**.
- NumPy docstrings (with runnable Examples on the public spatial API); >90 % coverage on touched code. **Coverage (corrected by 17/17b):** `coverage run -m pytest <touched test files>` then `coverage report --include=…` — **drop `--source`** (both `--source=brainpy_state` and `--cov=pkg.mod` SIGABRT inside jaxlib/absl double-init) and let the run finish fully before reading `.coverage`.

## Validation (via the 00 harness)
`_nest/_validation/spatial_{grid,gaussian_kernel,3d}_test.py` (+ a `_nest_spatial/` unit-test module mirroring `_connectivity_test.py`), `requires_nest`:
- **structural** (always-run, no NEST): grid coords exact; `circular`/`box` mask → **no realized edge beyond the cutoff**; degree symmetry where expected; seeded reproducibility.
- **distributional** (`skipUnless(nest)`): the binned connection-probability-vs-distance profile + the target-distance histogram vs NEST `nest.spatial` (seed-mean band).
- Each runnable example runs standalone without NEST (parity skips; the demo still plots its layer / target field). README + examples-gap §3.9 + `network-api-gap.md` §3.10 rows updated (the spatial blocker cleared / scoped).

## Edge cases to enumerate + test
A 1×N grid (degenerate 1D); `std→0` gaussian (only self/nearest); `std→∞` (≈ uniform within mask); a mask larger than the extent (all-to-all within layer); `allow_autapses=False` on a layer-to-itself connection; 2D vs 3D `free` layers; a population created **without** positions used in a spatial rule (clear error); empty realized sample (very small `p`); seeded reproducibility; `for_loop`/`jnp` lowering of the distance+sample (no Python double-loop); standalone-without-NEST.

## Definition of done
- [ ] `../../brainpy_state/_nest_spatial` submodule: `grid`/`free` layers (2D/3D), `gaussian` kernel, `circular`/`box` masks, `spatial_pairwise_bernoulli` rule, distance helpers — re-exported onto `brainpy.state`; design A (positions seam) resolved + recorded.
- [ ] `grid_iaf` + `gaussex` ported with parity (the acceptance floor); `test_3d_gauss` + `csa_spatial_example` ported (or `20b`-deferred with a recorded reason); `csa_example` re-expressed-or-placeholdered with the CSA-non-dependency documented.
- [ ] structural (mask cutoff / grid coords, always-run) + distributional (probability-vs-distance, skipUnless) parity passing/skipping; seeded reproducibility; `jnp`/`for_loop` lowering (no Python double-loop).
- [ ] >90 % coverage on touched code; README + examples-gap §3.9 + `network-api-gap.md` §3.10 rows updated; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the spatial recipe (design A — positions-on-population + the spatial-rule coord-injection branch; the `_nest_spatial/` primitive set; the distributional probability-vs-distance + structural mask-cutoff parity tiers; the CSA-library non-dependency), any fix, whether it split `20a`/`20b`, and confirmation **§3.9 spatial is unblocked** (the one absent capability is now built). Note any `spatial/` demos intentionally dropped + why.
2. Verify tests pass / skip; coverage > 90 %; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
