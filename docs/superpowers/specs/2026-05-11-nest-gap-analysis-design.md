---
title: NEST Gap Analysis — Design Spec
date: 2026-05-11
status: approved-for-planning
audience: brainpy.state maintainers
owner: Chaoming Wang
nest_reference_version: 3.x (latest stable on nest-simulator.readthedocs.io)
output_location: docs/nest-status/internal/
sphinx_visibility: excluded (maintainer-only)
---

# NEST Gap Analysis — Design Spec

This spec describes **how** to produce an internal gap analysis comparing
`brainpy_state/_nest` and surrounding network/example/doc infrastructure
against the upstream [NEST simulator](https://nest-simulator.readthedocs.io/).
The analysis itself is the *output* of executing this spec; the spec defines
its shape, taxonomy, methodology, and acceptance criteria so the work is
reproducible and the artifact stays maintainable.

## 1. Goal

Produce an actionable, prioritized internal roadmap for closing parity gaps
between `brainpy.state` and NEST. The deliverable is **for maintainers**, not
end users — the user-facing caveats already live in `docs/nest-status/index.rst`
and the gap analysis builds on those rather than restating them.

Concretely, executing this spec must yield, at minimum:

1. An evidence-backed classification of every NEST feature axis
   (neurons, synapses, plasticity, devices, generators, recorders, detectors,
   connection types, rate/binary models, multi-compartment models, parameters,
   units, defaults, state variables, delay handling, stochastic behavior,
   integration methods, network API, examples portfolio, docs portfolio,
   numerical validation coverage).
2. A list of features missing from `brainpy_state/_nest` (full upstream NEST
   catalog enumeration, not just self-disclosed gaps).
3. A list of semantic / numerical risks in the ported models.
4. A list of validation coverage gaps per family.
5. A single rolled-up prioritized roadmap at the index level, plus per-axis
   roadmap sections.

## 2. Deliverable shape

### 2.1 Layout

```
docs/nest-status/internal/
├── index.md                          # Rolled-up summary + consolidated roadmap
├── nest-catalog-snapshot.md          # Frozen NEST 3.x catalog (for re-diff)
├── neurons-gap.md
├── synapses-plasticity-gap.md
├── devices-gap.md
├── network-api-gap.md
├── examples-gap.md
├── docs-portfolio-gap.md
└── numerical-validation-gap.md
```

The `docs/nest-status/internal/` directory is **excluded from the public
Sphinx build** via `exclude_patterns` in `docs/conf.py` so the public
`docs/nest-status/index.rst` remains the only published NEST status page.
The directory is checked into git for maintainer visibility.

### 2.2 Index document (`index.md`)

Constraint: under ~150 lines. Contains:

- **Header**: NEST reference version studied; git SHA the analysis was
  performed against; last-updated date.
- **Parity summary table** with one row per axis: counts per classification
  bucket, link to the corresponding per-axis doc.
- **Consolidated prioritized roadmap**: all P0/P1 items from all per-axis
  docs pulled into one ranked list with T-shirt sizes and the family they
  unblock.
- **Methodology reference + classification taxonomy reference** (link, do
  not duplicate).
- **Unsupported list** (the intentionally-out-of-scope items from §6).

### 2.3 Per-axis documents

The seven per-axis docs follow an identical template so they are scannable
side-by-side:

```markdown
# <axis> — NEST parity gap

## 1. Scope
   What NEST surface this doc covers. Upstream NEST docs link.

## 2. Parity summary
   One paragraph + counts per classification bucket.

## 3. Evidence-backed mapping table
   | NEST feature | Status | brainpy.state location | NEST upstream | Tests | Notes |
   Status ∈ {implemented, unvalidated, partial, divergent, missing, unsupported}.

## 4. Missing or incomplete functionality
   Bulleted, with concrete evidence (file:line, NEST doc URL).

## 5. Semantic & numerical risks
   Where defaults/units/state vars/integration/stochasticity/delays diverge.

## 6. Validation gaps
   What tests don't exist; what NEST reference traces would be needed.

## 7. Prioritized roadmap
   - **P0 — <item>** [S/M/L/XL] — rationale + acceptance criteria.
   - **P1 — <item>** [S/M/L/XL] — ...
   - **P2 — <item>** [S/M/L/XL] — ...
```

### 2.4 Axis assignments

| Doc | Covers |
|---|---|
| `neurons-gap.md` | IAF (psc/cond/multisynapse/ps), AdEx, GIF, GLIF, HH, MAT, Izhikevich, rate, binary, point-process, multi-compartment, astrocyte, ignore_and_fire |
| `synapses-plasticity-gap.md` | Static synapses, STDP (9 variants), STP (tsodyks/quantal), Clopath, Urbanczik, Jonke, Vogels-Sprekeler, dopamine, volume_transmitter, gap junctions, SIC, diffusion, continuous-delay, bernoulli, ht_synapse |
| `devices-gap.md` | ac/dc/step/poisson/spike/inhomogeneous/sinusoidal/gamma/mip/pulsepacket/noise generators; multimeter, spike_recorder, weight_recorder; correlation/correlomatrix/correlospinmatrix/spin detectors; spike_dilutor, spike_train_injector |
| `network-api-gap.md` | `Connect`, `CopyModel`, `Create`, `GetStatus`/`SetStatus`, `GetConnections`, `NodeCollection`, connection rules (`one_to_one`, `all_to_all`, `fixed_indegree`, `fixed_outdegree`, `fixed_total_number`, `pairwise_bernoulli`, `pairwise_poisson`, symmetric variants), spatial/topology (Layer, Mask, Parameter), parameter expressions, model registry, kernel-status surface |
| `examples-gap.md` | PyNEST example portfolio (Brunel, balanced random network, microcircuit, hh demos, plasticity demos, gif demos, multimeter usage, etc.) vs. `docs/examples/gallery.rst` |
| `docs-portfolio-gap.md` | NEST user-doc tiers (getting started, model glossary, parameter-table conventions, connection management guide, recording-from-simulations guide, randomness guide) vs. `docs/`. Notable: there is no `docs/nest-guide/` porting tutorial today |
| `numerical-validation-gap.md` | Cross-cutting. Per-family inventory: which tests import `nest` for reference comparison; which only check self-consistency. Lists families that need a NEST-comparison harness before promotion |

## 3. Classification taxonomy

The only allowed status values in mapping tables:

- **implemented** — present, NEST-compatible parameters/defaults/state/units, and
  validated against NEST reference traces (a `nest.*` comparison test exists and
  passes within a documented tolerance).
- **unvalidated** — present and structurally NEST-compatible, but **no numerical
  validation** against NEST. Default bucket given the current `nest-status`
  self-disclosure.
- **partial** — present but missing a subset of NEST parameters, ports, or
  features (e.g., a multisynapse variant missing receptor types; a STDP rule
  missing weight-recorder hooks).
- **divergent** — present with intentionally different semantics (e.g.,
  precise-spike-time variants without lossless predicates; surrogate-gradient
  differentiability that NEST doesn't offer).
- **missing** — not present in `_nest/`; would need to be ported.
- **unsupported** — intentionally out of scope (see §6).

## 4. Evidence rules

Every row in a mapping table carries at minimum:

1. **Repo location**: file path with line range when relevant
   (e.g., `brainpy_state/_nest/iaf_psc_alpha.py:142-178`).
2. **NEST upstream doc URL**: link to the model/feature page on
   nest-simulator.readthedocs.io.
3. **Test reference**: paired `*_test.py` path + flag for whether the test
   imports `nest` and runs a numerical comparison.
4. **One-line note** on the specific gap or risk.

Risks and roadmap items must cite file:line where applicable. No hand-waving.

## 5. Methodology

**Structural sweep + targeted reads** (~1 day; family-level confidence;
explicitly noted where extrapolated).

1. **Snapshot the NEST catalog.** Fetch the NEST 3.x stable docs TOC for
   Models → Neurons / Synapses / Devices, Connection Management, and the
   PyNEST API reference. Save to `nest-catalog-snapshot.md` so future
   re-diffs are cheap. Record NEST version + retrieval date.
2. **Diff catalog vs. repo.** Compare against
   `brainpy_state/_nest/__init__.py` `__all__` to produce the missing-list
   per axis.
3. **Read 15-20 lead implementations** spanning all 5 integration categories
   + plasticity + devices. Initial reading slate:
   - Category A: `aeif_cond_alpha`, `aeif_psc_delta_clopath`, `gif_psc_exp`, `glif_psc`, `iaf_cond_alpha`, `iaf_cond_alpha_mc`
   - Category B: `iaf_psc_alpha`, `iaf_psc_exp_ps`, `iaf_psc_exp_ps_lossless`
   - Category C: `hh_psc_alpha`, `hh_cond_beta_gap_traub`, `ht_neuron`
   - Category D: `lin_rate`, `siegert_neuron`, `erfc_neuron`
   - Other neurons: `izhikevich`, `mat2_psc_exp`, `pp_psc_delta`, `cm_default`, `ignore_and_fire`
   - Synapses/plasticity: `stdp_synapse`, `stdp_dopamine_synapse`, `tsodyks2_synapse`, `clopath_synapse`, `urbanczik_synapse`, `vogels_sprekeler_synapse`, `volume_transmitter`
   - Devices: `multimeter`, `spike_recorder`, `weight_recorder`, `poisson_generator`, `noise_generator`, `inhomogeneous_poisson_generator`
   Extrapolate findings family-wide; record extrapolations explicitly in the
   per-axis docs so the basis is clear.
4. **For each read model, extract and diff**:
   - Parameter names, defaults, units (vs. NEST upstream docs).
   - State-variable layout (`DotDict` keys vs. NEST state names).
   - Integration method (Category A/B/C/D/E) + tolerance/step settings.
   - Delay handling (NEST uses min-delay-based slice scheduling;
     `brainpy.state` uses brainstate's delay infrastructure).
   - Stochasticity (PRNG source, distribution, parameterization).
   - Error handling (JIT-safe `jit_error_if` vs. Python `raise`).
   - Test file: does it `import nest`? Does it compare traces?
5. **Network-API sweep**: grep the repo to confirm `Connect`/`CopyModel`/etc.
   exist only in test comparison harnesses, then map each NEST PyNEST function
   onto its brainstate/brainpy equivalent (or `missing`).
6. **Examples sweep**: walk NEST's `pynest/examples/` listing (via the NEST
   GitHub repo or the docs example gallery), diff against
   `docs/examples/gallery.rst`. Produce a porting-target list.
7. **Docs portfolio sweep**: walk NEST's user-doc TOC, diff against repo
   `docs/` tree. Flag tier-level gaps (e.g., the absent `docs/nest-guide/`).

## 6. Prioritization principles

**P0** — blocks promotion of an entire family from Experimental → Beta, OR
blocks credible NEST porting at all. Candidates likely to land in P0:

- Missing NEST numerical-validation harness (cross-cutting).
- Recording-device semantic divergence vs. NEST device model.
- Absence of any NEST-style network API (`Connect`, `CopyModel`, connection rules).
- Absence of a `docs/nest-guide/` porting tutorial.

**P1** — parameter/default drift on individual models; missing widely-used
variants; missing flagship examples (`brunel_*`, cortical microcircuit,
balanced random network); missing common connection rules.

**P2** — edge-case models, niche connection rules, doc polish, parameter-table
aesthetics.

**T-shirt sizing**: S < 1 day, M = 1-3 days, L ≈ 1 week, XL = multi-week or
requires design.

Each roadmap item must include an **acceptance criterion** — a one-sentence
test for "done" (e.g., "Brunel reproduces NEST firing rate within 5% over a
1s window with matched seeds; test lives at
`brainpy_state/_nest/examples/brunel_validation_test.py`").

## 7. Intentionally unsupported

Document once in `index.md`; reference from per-axis docs. Not gaps, by design:

- **MPI / multi-process distribution** — NEST distributes via MPI;
  `brainpy.state` uses JAX device sharding.
- **MUSIC interface** — real-time inter-simulator coupling; not a goal.
- **NESTML and SLI** modeling languages — `brainpy.state` authors models in
  Python directly.
- **Real-time / hardware-in-the-loop devices.**
- **Bit-exact RNG parity** — NEST and JAX use independent PRNG streams;
  distributional equivalence is in scope, bitwise is not.
- **NEST kernel internals** (event scheduler, ring buffers, slice scheduling)
  — simulator-internal, irrelevant to a JAX rewrite.

## 8. Acceptance criteria for the gap analysis itself

The executed analysis is considered complete when:

1. All 7 per-axis docs + index exist under `docs/nest-status/internal/`.
2. Every row in every mapping table satisfies the evidence rules (§4).
3. `nest-catalog-snapshot.md` records the NEST version + retrieval date.
4. The index's consolidated roadmap is a strict superset of the P0/P1 items
   pulled from per-axis docs.
5. Every roadmap item has an acceptance criterion.
6. `docs/conf.py` `exclude_patterns` excludes the new internal directory.
7. The work is committed to git with a single commit/PR for reviewability.

## 9. Out-of-scope (for this spec; not for follow-up work)

- **Actually closing the gaps.** This spec scopes the analysis only.
  Roadmap items will be executed as separate plans/PRs after the analysis lands.
- **Bit-exact NEST validation.** See §7. Distributional / per-trace validation
  is in scope for individual roadmap items, not for this analysis.
- **A user-facing rewrite of `docs/nest-status/index.rst`.** The internal
  analysis may surface that the public page needs updates; that's a separate
  PR.

## 10. Open questions

None requiring user input at spec-approval time. Decisions deferred to
execution time:

- Final ordering of P0 items in the consolidated roadmap — depends on what the
  per-axis sweep surfaces. Initial heuristic: validation harness > network API
  > porting guide, but reorder if a per-axis sweep finds something more urgent.
- Whether `nest-catalog-snapshot.md` is generated by a script or written
  manually. Manual is fine for v1; if the analysis is re-run, consider a
  scrape script. Not a v1 requirement.
