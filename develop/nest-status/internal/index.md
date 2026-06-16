# NEST Parity Gap Analysis — Internal Maintainer Index

**Last updated:** 2026-06-16
**NEST reference version:** 3.x (latest stable on nest-simulator.readthedocs.io)
**Audience:** brainpy.state maintainers. Not built into the public Sphinx site.

This index rolls up the seven per-axis gap analyses in this directory. Each per-axis
doc owns its own evidence table; this index owns the consolidated roadmap.

> **Status (2026-06-16):** The original 2026-05-11 snapshot below predated the
> port work. Cluster backlog 00–28 has since **all merged** (cluster 18, e-prop,
> was deliberately dropped — it now lives in the sibling `braintrace` package).
> The numbers and findings have been refreshed to reflect the shipped state:
> `brainpy_state/_nest/` carries 122 model/rule modules and 75 NEST-style port
> scripts under `examples/nest_like/`; the validation harness, network `Simulator`
> facade, spatial API, and the STDP-divergence guide are all live. Sections that
> were premised on work that has since shipped are rewritten; genuinely-open and
> permanently-unsupported items are preserved.

## Parity summary

After the cluster backlog merged, the dominant status across every axis flipped
from `unvalidated`/`divergent`/`missing` to `implemented` (model ported **and**
covered by a live-NEST parity test). The validation harness (`nest_compare.py`,
`tolerance_conventions.py`) is shipped, and `brainpy_state/_nest/_validation/`
holds 140 files, 120 of them marked `@requires_nest`. Remaining non-`implemented`
counts below are the genuinely-open items tracked in the per-axis docs; figures
are approximate and the per-axis docs own the exact rows.

| Axis | implemented | unvalidated | partial | divergent | missing | unsupported | Doc |
|---|---:|---:|---:|---:|---:|---:|---|
| Neurons               | ~71 | 0 | 0 | 0 | 1 | 2 | [neurons-gap.md](neurons-gap.md) |
| Synapses & plasticity | ~31 | 0 | 0 | 0 | 1 | 0 | [synapses-plasticity-gap.md](synapses-plasticity-gap.md) |
| Devices               | ~24 | 0 | 0 | 0 | 0 | 7 | [devices-gap.md](devices-gap.md) |
| Network API           | ~90 | 0 | ~5 | 0 | ~6 | ~8 | [network-api-gap.md](network-api-gap.md) |
| Examples              | 75 | 0 | 0 | 0 | 0 | ~5 | [examples-gap.md](examples-gap.md) |
| Docs portfolio        | 3 | 0 | 2 | 0 | 5 | 1 | [docs-portfolio-gap.md](docs-portfolio-gap.md) |
| Validation coverage   | 120 | 0 | 0 | 0 | ~2 | 0 | [numerical-validation-gap.md](numerical-validation-gap.md) |

- Neurons `missing` = `parrot_neuron_ps` (precise variant, absent);
  `unsupported` = `ht_neuron` intrinsic-currents (out of scope) plus the
  permanently-unsupported list below. `parrot_neuron` itself ships.
- Synapses `missing` reflects remaining edge naming/expression helpers, not models.
- Network API `missing` = `nest_compat` string-model naming, `Parameter` lazy
  expressions, `CopyModel`, `CollocatedSynapses`, `symmetric_pairwise_bernoulli`.
- Validation `missing` ≈ `pp_psc_delta` lacks a `_validation` parity test.

Catalog baseline: [nest-catalog-snapshot.md](nest-catalog-snapshot.md) — 74
NEST neurons, 32 synapses/plasticity, 15 generators, 3 recorders, 4 detectors,
2 other devices, 7 MUSIC proxies, 10 connection rules, ~70 PyNEST API entries,
~25 spatial/topology entities.

## Headline findings

1. **Models are now `implemented` in the strict sense.** The shared harness
   (`brainpy_state/_nest/_validation/nest_compare.py` +
   `tolerance_conventions.py`) gives every promoted model a documented tolerance
   category (A–E) + duration + dt convention. 120 of the 140 `_validation` files
   carry `@requires_nest` live-NEST parity tests; the implicit-convention era is
   over.
2. **A network API surface exists.** `brainpy_state/_nest/_validation/` exercises
   a `Simulator` facade (`create`/`connect`/`get_connections`/
   `tripartite_connect`/`cont`/`reset_rollout`) with 6 named connection rules,
   `SynapseCollection`, the full set of mask shapes, and `explicit_edges`.
   `GetConnections` landed in cluster 23. The remaining gaps are naming-shim and
   convenience surface (see roadmap), not the core programming model.
3. **Validation coverage is broad, not bimodal.** AdEx, rate models, all devices,
   synapses+plasticity, and the previously-empty families (IAF psc/cond/
   specialized, GIF, GLIF, HH, MAT, Izhikevich, binary, point-process, STP) now
   have NEST-comparison tests. The lone known hole is `pp_psc_delta`, which ships
   but lacks a `_validation` parity test.
4. **The `docs/nest-guide/` on-ramp has started.** `docs/nest-guide/index.rst`
   plus the `stdp-divergences.rst` page are live. Still missing: the full
   side-by-side porting tutorial, connection-management/recording/randomness
   guides, and the PyNEST API mapping reference.
5. **The e-prop family is out of scope here — it moved to `braintrace`.** The
   8 neurons + 4 synapses + `weight_optimizer` were deliberately dropped from the
   cluster backlog (cluster 18) and now live in the sibling `braintrace` package
   wired through brainpy.state's surrogate-gradient stack. Not a gap in this repo.
6. **Recording-device semantic divergence is now documented.** The STDP /
   trace-storage divergence has a canonical written treatment in
   `docs/nest-guide/stdp-divergences.rst`, and recorder parity is exercised in
   `_validation`. The public `nest-status/index.rst` caveat is backed by tests.

## Consolidated roadmap

Ordering applies the spec §6 prioritization principles: validation harness
unblocks everything else, then the network-API surface and porting guide unblock
user adoption, then per-family validation lands. **Status (2026-06-16):** the
cluster backlog 00–28 has merged, so the bulk of P0 and P1 below is **done**.
Done items are marked inline; the few that remain open are flagged
**STILL OPEN**.

### P0 (blocks family promotion or credible porting)

1. **Build shared NEST-comparison harness** [M] — `numerical-validation-gap.md`.
   **DONE:** `brainpy_state/_nest/_validation/` ships `nest_compare.py`
   (`requires_nest`, `compare_trace`, `compare_distributional`),
   `tolerance_conventions.py`, `conftest.py`, and `README.md`;
   `@requires_nest` registered; existing tests refactored onto the harness.

2. **Build network API surface** [XL] — `network-api-gap.md`.
   **DONE (mostly):** a `Simulator` facade
   (`create`/`connect`/`get_connections`/`tripartite_connect`/`cont`/
   `reset_rollout`) with 6 named connection rules, `SynapseCollection`, mask
   shapes, and `explicit_edges` ships; Brunel ports run on it. **STILL OPEN:**
   the `nest_compat` string-model naming layer, `Parameter` lazy expressions,
   `CopyModel`, `CollocatedSynapses`, `symmetric_pairwise_bernoulli`.

3. **Create `docs/nest-guide/` + porting tutorial** [L] — `docs-portfolio-gap.md`.
   **PARTIAL:** `docs/nest-guide/index.rst` + `stdp-divergences.rst` are live.
   **STILL OPEN:** the full side-by-side Create → Connect → Simulate → Plot
   porting tutorial.

4. **Document tolerance conventions in a single page** [S] — `numerical-validation-gap.md`.
   **DONE:** `tolerance_conventions.py` defines categories A–E (CAT_A 1e-3 mV,
   CAT_B 1e-6, CAT_B_ALIGNED align_steps, CAT_C, CAT_D distributional/N_SEEDS,
   CAT_E spike-time) with the multi-seed protocol; documented in the harness
   `README.md`.

5. **Validate IAF psc family** [L] — `neurons-gap.md` + `numerical-validation-gap.md`.
   **DONE:** `iaf_psc_*` variants carry live-NEST V_m parity tests in `_validation`.

6. **Validate IAF cond family** [L] — `neurons-gap.md` + `numerical-validation-gap.md`.
   **DONE:** `iaf_cond_*` validated; `iaf_cond_alpha_mc.py` ships with its
   compartment-tree topology equivalence covered.

7. **Validate STP family (`tsodyks*`, `quantal_stp_synapse`, `volume_transmitter`)** [M] — `synapses-plasticity-gap.md` + `numerical-validation-gap.md`.
   **DONE:** the STP plasticity rules + device validated on the harness.

8. **Validate `stdp_dopamine_synapse` + `volume_transmitter`** [M] — `synapses-plasticity-gap.md`.
   **DONE:** dopamine-modulated STDP weight-trajectory parity covered in `_validation`.

9. **Audit `weight_recorder` hookup per STDP variant** [M] — `synapses-plasticity-gap.md` + `devices-gap.md`.
   **DONE:** `stdp_*` tests attach `weight_recorder` and check event count + timing.

10. **Document STDP trace-storage divergence** [S] — `synapses-plasticity-gap.md`.
    **DONE:** `docs/nest-guide/stdp-divergences.rst` covers `tau_minus` placement
    side by side.

11. **Recording-device parity test** [M] — `devices-gap.md`.
    **DONE:** recorder/`multimeter` parity is exercised in `_validation`; backs
    the public `nest-status/index.rst` recorder caveat.

12. **Document stamp-step gating convention** [S] — `devices-gap.md`.
    **DONE:** the device-gating window convention is captured alongside the
    device parity tests (`(origin+start, origin+stop]`).

13. **Document the programming-model gap** [S] — `network-api-gap.md`.
    **PARTIAL:** the `Simulator` facade closes most of the gap; a polished
    PyNEST → brainpy.state cheatsheet page is **STILL OPEN**.

14. **Map named connection rules to brainstate primitives** [M] — `network-api-gap.md`.
    **DONE:** 6 named connection rules ship on the `Simulator` facade with
    matching NEST semantics (`one_to_one`, `all_to_all`, `fixed_indegree`,
    `pairwise_bernoulli`, `fixed_total_number`,
    `third_factor_bernoulli_with_pool`), plus the `explicit_edges` primitive.
    (`fixed_outdegree` exists only as an internal sampler, not a named rule.)

15. **PyNEST → brainpy.state cheatsheet** [S] — `docs-portfolio-gap.md`.
    **STILL OPEN:** a dedicated cheatsheet page mapping the PyNEST verbs to
    brainpy.state idioms is not yet written.

16. **Parameter-table render in API ref** [M] — `docs-portfolio-gap.md`.
    **STILL OPEN:** per-model "Defaults" mini-tables in the API reference.

17. **Port `brunel_alpha_nest.py` as flagship example** [L] — `examples-gap.md`.
    **DONE:** `examples/nest_like/brunel_alpha.py` ships (plus `brunel_delta`,
    `brunel_exp_multisynapse`, `brunel_siegert`, and astrocyte-Brunel variants).

18. **Port `one_neuron.py` + `one_neuron_with_noise.py`** [S] — `examples-gap.md`.
    **DONE:** both ship under `examples/nest_like/`.

19. **Port `multimeter_file.py` or in-memory equivalent** [M] — `examples-gap.md`.
    **DONE:** `examples/nest_like/multimeter_file.py` ships.

### P1 (parameter drift, common variants, flagship support)

Pulled from per-axis docs §7; full acceptance criteria live there. Most of this
tier landed with the cluster backlog — done items struck through, remainder
flagged **STILL OPEN**.

- ~~**Promote AdEx family (9) from `divergent` to `implemented`**~~ [M] — **DONE.**
- ~~**Promote rate models (10) from `divergent` to `implemented`**~~ [M] — **DONE.**
- ~~**Validate GIF (5), GLIF (3), HH (6), MAT (2), Izhikevich (1), point-process (2), binary (3)**~~ [L total] — **DONE** (parity tests in `_validation`).
- ~~**Validate `iaf_cond_alpha_mc` multi-compartment**~~ [M] — **DONE** (`iaf_cond_alpha_mc.py`).
- **Port `parrot_neuron` + `parrot_neuron_ps`** [S] — `parrot_neuron.py` **DONE**;
  `parrot_neuron_ps` (precise variant) **STILL OPEN** (absent).
- ~~**Document spike-pairing convention per `stdp_nn_*` variant**~~ [S] — **DONE** (`stdp-divergences.rst`).
- ~~**Promote 22 `divergent` synapses to `implemented`**~~ [M] — **DONE.**
- ~~**Boundary regression for `start`/`stop`/`origin`**~~ [S] — **DONE** (device parity tests).
- ~~**Noise generator dt-invariance test**~~ [S] — **DONE.**
- ~~**Correlation-detector window + normalization parity**~~ [M] — **DONE** (`correlation_detector` + `correlomatrix`/`correlospinmatrix` tests).
- ~~**`spike_generator` off-grid times convention**~~ [S] — **DONE.**
- ~~**Implement `pairwise_bernoulli` + `fixed_total_number` named helpers**~~ [M] — **DONE** (named connection rules on `Simulator`).
- **Implement `Parameter` runtime-evaluated expressions in `nest_compat`** [L] — **STILL OPEN.**
- ~~**Add `TripartiteConnect`**~~ [M] — network-api. **DONE (cluster 24):**
  `Simulator.tripartite_connect` + `third_factor_bernoulli_with_pool`, live-NEST
  parity (block bit-identical / random cat-D). Unblocked the 3 astrocyte demos.
- **`CollocatedSynapses` support** [M] — network-api. **STILL OPEN.**
- ~~**Port rest of Brunel family**~~ [L] — **DONE** (`brunel_delta`, `brunel_exp_multisynapse`, `brunel_siegert`, astrocyte-Brunel).
- ~~**Port Clopath, STP, Astrocyte-Brunel, pedagogical-singles examples**~~ [L total] — **DONE** (75 NEST-style ports under `examples/nest_like/`).
- **Recording-from-simulations guide** [M] — docs. **STILL OPEN.**
- **Connection-management guide** [M] — docs. **STILL OPEN.**
- **Randomness guide** [S] — docs. **STILL OPEN.**
- **PyNEST tutorials series (4-part)** [L] — docs. **STILL OPEN** (only the STDP page + index exist).
- **PyNEST API mapping reference** [M] — docs. **STILL OPEN.**
- **Define brainpy-extension parameter convention** [S] — neurons / docs. **STILL OPEN.**

> **Validation gap to preserve:** `pp_psc_delta.py` ships but has **no
> `_validation` parity test** yet — the one known coverage hole.

### P2 (edge cases, polish)

Summarized at index level — see per-axis docs §7 for full lists.

- ~~**e-prop family port (XL)**~~ — **out of scope here**: deliberately dropped from
  the cluster backlog (cluster 18) and ported to the sibling **`braintrace`**
  package instead. Not a gap in this repo.
- ~~**Spatial / topology surface (XL)**~~ — **DONE (clusters 20/27):**
  `brainpy.state.spatial.*` ships per-axis `pos`, gaussian/exponential/gabor/
  gamma distributions, circular/spherical/box/rectangular/doughnut/elliptical/
  ellipsoidal masks, `nearest_element`/`select_nodes_by_mask`, and dump/plot
  helpers.
- ~~**HH gap-junction parity (M)**~~ — **DONE:** `hh_psc_alpha_gap` (cluster-15b) and
  `hh_cond_beta_gap_traub` gap-parity both ship.
- ~~**`pong`/`sudoku` example ports (L)**~~ — **DONE** (both ported; see §3.10).
- **CI parity-check matrix (M)** + **validation progress badge (S)** —
  **STILL OPEN:** the 120 `@requires_nest` tests exist but are not yet wired into
  a CI parity matrix or surfaced as a badge.
- **Examples-gallery wiring** — **STILL OPEN:** the 75 `examples/nest_like/` ports are
  not yet surfaced in `gallery.rst`.
- **File-backed recording backends (L)** — out of scope (see Intentionally
  unsupported; ascii/sionlib/file backends are not supported).
- **Parallel-computing guide (M)** — **STILL OPEN** (would document JAX device
  sharding as the substitute for MPI multi-process).
- **Glossary (S)** — **STILL OPEN.**

`ht_neuron` (Hill–Tononi) intrinsic-currents remain out of scope.

## Intentionally unsupported

Per spec §7 — not gaps:

- MPI / multi-process distribution (JAX device sharding instead).
- MUSIC interface (real-time inter-simulator coupling). 7 NEST `music_*`
  proxies in catalog.
- NESTML and SLI modeling languages (brainpy.state authors models in Python).
- Real-time / hardware-in-the-loop devices.
- Bit-exact RNG parity (distributional in scope).
- NEST kernel internals (event scheduler, ring buffers, slice scheduling).
- SONATA file loader (file format is open; the NEST loader is NEST-internal).
- Structural plasticity (`structural_plasticity.py` example out of scope).
- HPC benchmark (`hpc_benchmark.py` requires MPI).

## Methodology and classification reference

- Status values: `implemented`, `unvalidated`, `partial`, `divergent`,
  `missing`, `unsupported`. Defined in spec §3.
- Catalog snapshot: [nest-catalog-snapshot.md](nest-catalog-snapshot.md).
