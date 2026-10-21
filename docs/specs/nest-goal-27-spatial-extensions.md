# /goal — 27 · `nest.spatial.*` extensions (additive — per-axis pos, more distributions/masks, nearest-element/by-mask, dump/plot helpers)

> Self-contained prompt. Open at repo root. Stream: spatial. **Additive follow-up — NO seam change.** Extends the **`_nest_spatial/` submodule `20` shipped** (PR #72) with the residual `nest.spatial` surface. Every item rides the **already-fixed** spatial seam (`create(positions=)` → `_positions` registry → `_bind_spatial_coords` → `rule.with_coords()` pure-clone → `SpatialConnRule.sample(...)`; read-side `get_position`/`target_*`). **Do not touch `_simulator.py`/`_connect_pair`/the `ConnRule` contract.** Depends on: `20` only. Independent of everything else.

## Why this is purely additive — read first
`20`'s "For next clusters" Lessons (CONTEXT.md) hand this off explicitly: *"the primitives + seam are the substrate for the rest of `nest.spatial`. Queued (all additive, no seam change): per-axis `spatial.pos.x/y/z`/`source_pos`/`target_pos`; the `exponential`/`gabor`/`gamma` distance distributions (**mirror `_GaussianKernel`**); the other mask shapes (rectangular/doughnut/elliptical — **mirror `_RadialMask`/`_BoxMask`**); nearest-element/`SelectNodesByMask`; layer dump/plot helpers. The `_positions` registry + `get_position` + `target_*` helpers are the read-side substrate; `with_coords`-clone binding is the write-side substrate."* The precise residual is mapped in **`network-api-gap.md` §3.10** (rows 157–163, all `missing`/`partial`). This cluster closes that residual; it adds **new atoms to existing files**, never a new seam.

## The precise residual (network-api-gap.md §3.10 — what's still `missing`/`partial`)
| group | done (`20`) | **27 adds** (mirror) | parity tier |
|---|---|---|---|
| per-axis position | scalar `distance` sentinel | `spatial.pos.{x,y,z}`, `source_pos`/`target_pos.{x,y,z}`, `distance.{x,y,z}` — kernels/masks consume a single axis, not just the scalar (extend `_DistanceSentinel` in `_kernels.py`) | **exact** (deterministic geometry) |
| distance distributions | `gaussian` | `exponential`, `gabor`, `gamma` (+`gaussian2D` if cheap) — **mirror `_GaussianKernel`** (`_kernels.py:29`) | **distributional** (`p(d)` band, the `20` posture) |
| mask shapes | `circular`/`spherical`/`box` | `rectangular`, `doughnut`, `elliptical`, `ellipsoidal` — **mirror `_RadialMask`/`_BoxMask`** (`_masks.py:20/32`) | **exact** (hard in/out cutoff) |
| queries | `center_element`, `Distance`, `target_nodes`, `target_positions`, `get_position` | `nearest_element` (`FindNearestElement`), `select_nodes_by_mask` (`SelectNodesByMask`) — helper takes `(sim, source, target, …)` | **exact** |
| dump | — | `dump_layer_connections` (`DumpLayerConnections`), `dump_layer_nodes` (`DumpLayerNodes`) — read via `get_connections`/`get_position` | **exact** (vs NEST dump text) |
| plot | — | `plot_layer`, `plot_targets`, `plot_sources`, `plot_probability_parameter` — matplotlib, smoke-only (no NEST parity) | smoke / no-parity |

## Session start
1. Create an isolated worktree branch **`nest-goal/27-spatial-extensions`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially the **`20`-spatial Lessons**:
   - the **seam** (API-discovered section): `create(positions=)` → `_positions[id(pop)]`; `_bind_spatial_coords` at the top of `_connect_pair` gated on `getattr(rule, '_is_spatial', False)`; `with_coords` **pure clone** so every downstream path samples one coord-bound rule; the `ConnRule.sample(...)` contract is **unchanged**. *"any new distance-rule variant subclasses/feeds `SpatialConnRule`; any new query helper takes `(sim, source, target)`."*
   - the **five gotchas** (these recur for every new atom): (1) **diagonal/autapse trap** — for a self-connection with `allow_autapses=False`, exclude the `n` diagonal `(i,i)` `d=0` pairs from the candidate denominator or the `d≈0` bin dilutes (a≠b connects don't hit this); (2) **`free(distribution)` infers dim from `extent` XOR `num_dimensions`** — never both; (3) **units** — bare floats on coords/extent/std become `u.um` via `_as_len`, so `d/std` is dimensionless; never multiply a Quantity distance by a bare std; (4) **NEST grid convention** `x=c−L/2+(col+0.5)·L/n`, `y=c+L/2−(row+0.5)·L/n`, node `k`=col·n_rows+row; (5) **distributional tests use a fixed brainpy seed** so the NEST-free structural class is deterministic.
   - the **`16`-generator-demos Lessons** for the demo-cluster house style (runnable `examples/nest/` script + a structural NEST-free class **and** a `skipUnless(nest)` parity class).
   Then read:
   - the existing submodule to mirror: `brainpy_state/_nest_spatial/{_kernels,_masks,_distance,_helpers,_layers,_rule}.py` + `__init__.py` `__all__` (the export list you extend). Mirror `_GaussianKernel` (kernels), `_RadialMask`/`_BoxMask` (masks).
   - the existing tests/demos as templates: `_nest/_validation/spatial_{grid,gaussian_kernel,3d}_test.py`; `examples/nest/spatial_{grid_iaf,gaussex,3d_gauss,csa}.py`.
   - `docs/nest-status/internal/network-api-gap.md` §3.10 (rows 157–163 — you will flip them).
   - NEST upstream (clone on this machine: **`/mnt/d/codes/githubs/computational_neuroscience/nest-simulator`** — the CLAUDE.md path; the 17-Lessons `…/githubs/nest-simulator` note is machine-specific and **absent here** — `ls`-verify): `nest.spatial` / `nest.spatial_distributions` (the `pynest/nest/spatial*` modules) + `pynest/examples/spatial/*` for the exact `p(d)`/mask formulas and dump/plot text format.
3. Write spec + plan under `` (gitignored). **Approach-first; wait for approval before code** — the sign-off is the **scope cut** (which residual atoms ship now vs split to `27b`; plot helpers are smoke-only) + the **per-group parity tiers** (exact geometry vs distributional `p(d)` vs hard-cutoff masks).

## Goal
Extend `../../brainpy_state/_nest_spatial` (re-exported as `brainpy.state.spatial`) with the residual `nest.spatial` surface — per-axis position expressions, the `exponential`/`gabor`/`gamma` distributions, the rectangular/doughnut/elliptical(/ellipsoidal) masks, nearest-element + by-mask queries, and dump/plot helpers — **with zero seam change**, each with a live-NEST test (exact coords / distributional `p(d)` / hard-cutoff mask, pass-or-skip) + the structural NEST-free class. At least one new/extended demo exercising the per-axis + new-distribution + new-mask paths.

## Constraints
- **`_nest_spatial/` only** (extend `_kernels.py`/`_masks.py`/`_distance.py`/`_helpers.py`/`__init__.__all__`; add `_plot.py` if needed). **No `_simulator.py`, `_connect_pair`, or `ConnRule`-contract change** — if you think you need one, stop and re-read the `20` seam Lessons (you don't). New distance-rule variants subclass/feed `SpatialConnRule`; new query helpers take `(sim, source, target, …)`.
- Example code uses the **public** namespace (`from brainpy import state as bp`; `bp.state.spatial.*`) — never reach the inner package through `brainpy.state` from inside it; demo helpers import `from examples.nest.<module>`; keep a no-NEST companion.
- **brainunit** units throughout; respect the `_as_len`→`u.um` convention (gotcha 3); never multiply a Quantity distance by a bare std.
- Parity tiers: **exact** for per-axis coords + masks (deterministic geometry / hard in-out cutoff — assert node-set / coord equality vs NEST), **distributional** for the new `p(d)` distributions (empirical-fraction band per distance bin + a fixed brainpy seed for the structural class — the `20` posture, ≤~1.6 %), **smoke-only** for plot helpers (figure builds, no NEST parity). Mind the diagonal/autapse trap (gotcha 1) in every distributional binning test.
- Any fix **reproduce-first** (`superpowers:systematic-debugging`). Any Simulator-driven rollout in a demo lowers via `brainstate.transform.for_loop`/`scan` — **never a bare Python `for`/`while`** over the model (CLAUDE.md #10).
- NumPy docstrings; >90 % coverage on touched code. **Coverage (corrected by 17/17b):** `coverage run -m pytest <touched test files>` then `coverage report --include=…` — **drop `--source`** (both `--source=brainpy_state` and `--cov=pkg.mod` SIGABRT inside jaxlib/absl double-init); let the run finish before reading `.coverage`.

## Validation (via the 00 harness + the `_nest_spatial` unit tests)
- `_nest_spatial/_kernels_test.py` (extend) — `exponential`/`gabor`/`gamma` shape + per-axis sentinel; `_masks_test.py` (extend) — rectangular/doughnut/elliptical(/ellipsoidal) in/out cutoff; `_helpers_test.py` (extend) — nearest-element / by-mask; a `_plot_test.py` smoke test.
- `_nest/_validation/spatial_{exponential,gabor,gamma,rect_mask,nearest,…}_test.py`, `requires_nest` — distributional `p(d)` band for each new distribution; exact node-set vs NEST for each new mask + nearest-element + by-mask; dump-text vs `DumpLayer*`.
- Each runnable demo runs standalone (`PYTHONPATH=. python examples/nest/spatial_<name>.py`) **without** NEST (parity skips; the demo still plots its layer / connectivity).

## Edge cases to enumerate + test
A kernel consuming `pos.x` only (1-axis anisotropy) vs the scalar `distance`; `gamma`/`gabor` at parameter extremes (the band still holds); a doughnut mask with inner=0 (degenerates to circular) and inner=outer (empty); an elliptical mask rotated vs axis-aligned; nearest-element with ties (deterministic tiebreak); by-mask on a node exactly on the boundary (in/out convention matches NEST); 2-D vs 3-D for every mask (ellipsoidal = 3-D elliptical); the autapse/diagonal trap on every distributional test; a deferred-`free` layer (size drawn with a per-pop key) under a new mask; standalone-without-NEST for each demo.

## Definition of done
- [ ] Per-axis pos/distance + `exponential`/`gabor`/`gamma` + rectangular/doughnut/elliptical(/ellipsoidal) masks + nearest-element/by-mask + dump/plot helpers added to `_nest_spatial/` (or a documented `27b` split), `__all__` extended, **zero seam change**.
- [ ] Live-NEST tests: exact coords/masks/nearest/by-mask/dump; distributional `p(d)` bands; plot smoke. Structural NEST-free classes deterministic (fixed seed).
- [ ] ≥1 demo exercises the per-axis + new-distribution + new-mask paths; runnable standalone without NEST.
- [ ] >90 % coverage on touched code; `network-api-gap.md` §3.10 rows 157–163 flipped (missing/partial → done); Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: which residual atoms shipped (vs any `27b` split), the per-group parity tiers + bands, any new gotcha (anisotropic-axis binning, mask-boundary convention, dump-text format), and confirmation **the `20` seam absorbed all of it with zero `_simulator.py` change** (or the precise reason one was unavoidable). Note `network-api-gap.md` §3.10 is now **done** bar any documented residual.
2. Verify tests pass / skip; coverage > 90 %; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + per-group parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
