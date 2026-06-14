# NEST Parity Gap Analysis — Internal Maintainer Index

**Last updated:** 2026-05-11
**Git SHA at analysis:** `3c575a34453ba08b22e078755ae3feeaa4151ee6`
**NEST reference version:** 3.x (latest stable on nest-simulator.readthedocs.io)
**Audience:** brainpy.state maintainers. Not built into the public Sphinx site.

This index rolls up the seven per-axis gap analyses in this directory. Each per-axis
doc owns its own evidence table; this index owns the consolidated roadmap.

## Parity summary

| Axis | implemented | unvalidated | partial | divergent | missing | unsupported | Doc |
|---|---:|---:|---:|---:|---:|---:|---|
| Neurons               | 0 | ~41 | 0 | 22 | 10 | 7 | [neurons-gap.md](neurons-gap.md) |
| Synapses & plasticity | 0 | 5 | 0 | 22 | 5 | 0 | [synapses-plasticity-gap.md](synapses-plasticity-gap.md) |
| Devices               | 0 | 0 | 0 | 24 | 0 | 7 | [devices-gap.md](devices-gap.md) |
| Network API           | 0 | 0 | ~5 | 0 | ~95 | ~8 | [network-api-gap.md](network-api-gap.md) |
| Examples              | 0 | 0 | ~4 | 0 | ~50 | ~5 | [examples-gap.md](examples-gap.md) |
| Docs portfolio        | 1 | 0 | 4 | 1 | 6 | 1 | [docs-portfolio-gap.md](docs-portfolio-gap.md) |
| Validation coverage   | 0 | 47 | 0 | 69 | 0 | 0 | [numerical-validation-gap.md](numerical-validation-gap.md) |

Catalog baseline: [nest-catalog-snapshot.md](nest-catalog-snapshot.md) — 73
NEST neurons, 32 synapses/plasticity, 15 generators, 3 recorders, 4 detectors,
2 other devices, 7 MUSIC proxies, 10 connection rules, ~70 PyNEST API entries,
~25 spatial/topology entities.

## Headline findings

1. **No model is `implemented`** in the strict sense — none have a documented
   tolerance + duration + dt convention in their NEST-comparison test header.
   59 % of ported modules have *some* NEST-comparison test code but the
   conventions are implicit.
2. **The PyNEST API surface is essentially absent.** No `Connect`, `Create`,
   `CopyModel`, `Simulate` at top level; users compose brainstate `Projection`
   objects directly. This is the single biggest porting obstacle.
3. **Validation coverage is bimodal.** AdEx, rate models, all devices, and
   most synapses+plasticity rules are validated; IAF (psc/cond/specialized),
   GIF, GLIF, HH, MAT, Izhikevich, binary, point-process, and STP family have
   zero NEST-comparison tests.
4. **No `docs/nest-guide/` porting tutorial exists.** NEST users have no
   guided on-ramp; the public `nest-status/index.rst` is caveats-only.
5. **The e-prop family (8 neurons + 4 synapses + `weight_optimizer`) is
   entirely missing.** Could either be a verbatim port or wired through
   brainpy.state's existing surrogate-gradient stack — strategic decision
   needed.
6. **Recording-device semantic divergence is real but undocumented in
   concrete terms.** `nest-status/index.rst:92-93` warns but the divergence
   needs a single canonical reproducer test.

## Consolidated roadmap

Ordering applies the spec §6 prioritization principles: validation harness
unblocks everything else, then the network-API shim and porting guide unblock
user adoption, then per-family validation lands.

### P0 (blocks family promotion or credible porting)

1. **Build shared NEST-comparison harness** [M] — `numerical-validation-gap.md`.
   Acceptance: `brainpy_state/_nest/_validation/` exists with `nest_compare.py`,
   `comparison_base.py`, `tolerance_conventions.py`, README;
   `@pytest.mark.requires_nest` registered; 3 existing tests refactored.

2. **Build `brainpy_state.nest_compat` shim package** [XL] — `network-api-gap.md`.
   Acceptance: minimum viable surface (`Create`, `Connect` with 4 connection
   rules, `CopyModel`, `GetStatus`, `SetStatus`, `Simulate`, `ResetKernel`,
   `SetKernelStatus`); Brunel example ports verbatim from NEST.

3. **Create `docs/nest-guide/` + porting tutorial** [L] — `docs-portfolio-gap.md`.
   Acceptance: side-by-side PyNEST + brainpy.state for Create → Connect →
   Simulate → Plot; linked from public Experimental warning.

4. **Document tolerance conventions in a single page** [S] — `numerical-validation-gap.md`.
   Acceptance: per-category defaults (A/B/C/D/E) + multi-seed protocol
   documented; linked from harness README.

5. **Validate IAF psc family** [L] — `neurons-gap.md` + `numerical-validation-gap.md`.
   Acceptance: all 10 `iaf_psc_*` variants run V_m traces matching NEST in
   harness over 1 s × 3 parameter sets.

6. **Validate IAF cond family** [L] — `neurons-gap.md` + `numerical-validation-gap.md`.
   Acceptance: all 5 `iaf_cond_*` variants validated; `iaf_cond_alpha_mc`
   documents compartment-tree topology equivalence.

7. **Validate STP family (`tsodyks*`, `quantal_stp_synapse`, `volume_transmitter`)** [M] — `synapses-plasticity-gap.md` + `numerical-validation-gap.md`.
   Acceptance: 5 plasticity rules + 1 device validated using harness.

8. **Validate `stdp_dopamine_synapse` + `volume_transmitter`** [M] — `synapses-plasticity-gap.md`.
   Acceptance: 3-neuron + 1-VT regression matches NEST weight trajectory
   within tolerance over 5 s.

9. **Audit `weight_recorder` hookup per STDP variant** [M] — `synapses-plasticity-gap.md` + `devices-gap.md`.
   Acceptance: each `stdp_*` test attaches `weight_recorder` and verifies
   event count + timing matches NEST.

10. **Document STDP trace-storage divergence** [S] — `synapses-plasticity-gap.md`.
    Acceptance: side-by-side `tau_minus` placement docs in `docs/nest-guide/`
    (or interim location); linked from every `stdp_*_synapse` docstring.

11. **Recording-device parity test** [M] — `devices-gap.md`.
    Acceptance: `iaf_psc_alpha` + `multimeter` 1 s comparison test exists
    at `brainpy_state/_nest/_validation/recorder_parity_test.py`; sharpens
    the language in public `nest-status/index.rst:92-93`.

12. **Document stamp-step gating convention** [S] — `devices-gap.md`.
    Acceptance: device-gating formula `(origin+start, origin+stop]` appears
    in `docs/api/nest-devices.rst` or `docs/nest-guide/`.

13. **Document the programming-model gap** [S] — `network-api-gap.md`.
    Acceptance: PyNEST → brainpy.state cheatsheet section in `docs/nest-guide/`.

14. **Map named connection rules to brainstate primitives** [M] — `network-api-gap.md`.
    Acceptance: mapping table + thin wrappers in `nest_compat` for
    `fixed_indegree`, `fixed_outdegree`, `pairwise_bernoulli`,
    `fixed_total_number` matching NEST semantics.

15. **PyNEST → brainpy.state cheatsheet** [S] — `docs-portfolio-gap.md`.
    Acceptance: `docs/nest-guide/cheatsheet.rst` maps `Create`, `Connect`,
    `Simulate`, `GetStatus`, `SetStatus`, `CopyModel`, `ResetKernel`,
    `SetKernelStatus`, 4 connection rules to brainpy.state idioms.

16. **Parameter-table render in API ref** [M] — `docs-portfolio-gap.md`.
    Acceptance: each model in `docs/api/nest-neurons.rst` gains a "Defaults"
    mini-table (parameter, default, unit, NEST upstream link).

17. **Port `brunel_alpha_nest.py` as flagship example** [L] — `examples-gap.md`.
    Acceptance: `examples/nest/brunel_alpha.py` exists; firing rate + CV
    within 5 % of NEST over 1 s; doubles as IAF psc family validation.

18. **Port `one_neuron.py` + `one_neuron_with_noise.py`** [S] — `examples-gap.md`.
    Acceptance: both ports live in `docs/nest-guide/examples/`; PyNEST and
    `nest_compat` shown side by side.

19. **Port `multimeter_file.py` or in-memory equivalent** [M] — `examples-gap.md`.
    Acceptance: ported example produces same V_m trace as NEST example.

### P1 (parameter drift, common variants, flagship support)

Pulled from per-axis docs §7; full acceptance criteria live there.

- **Promote AdEx family (9) from `divergent` to `implemented`** [M] — neurons / validation.
- **Promote rate models (10) from `divergent` to `implemented`** [M] — validation.
- **Validate GIF (5), GLIF (3), HH (6), MAT (2), Izhikevich (1), point-process (2), binary (3)** [L total] — neurons / validation.
- **Validate `iaf_cond_alpha_mc` multi-compartment** [M] — neurons / validation.
- **Port `parrot_neuron` + `parrot_neuron_ps`** [S] — neurons / synapses.
- **Document spike-pairing convention per `stdp_nn_*` variant** [S] — synapses.
- **Promote 22 `divergent` synapses to `implemented`** [M] — synapses.
- **Boundary regression for `start`/`stop`/`origin`** [S] — devices.
- **Noise generator dt-invariance test** [S] — devices.
- **Correlation-detector window + normalization parity** [M] — devices.
- **`spike_generator` off-grid times convention** [S] — devices.
- **Implement `pairwise_bernoulli` + `fixed_total_number` named helpers** [M] — network-api.
- **Implement `Parameter` runtime-evaluated expressions in `nest_compat`** [L] — network-api.
- **Add `TripartiteConnect`** [M] — network-api.
- **`CollocatedSynapses` support** [M] — network-api.
- **Port rest of Brunel family** [L] — examples.
- **Port Clopath, STP, Astrocyte-Brunel, pedagogical-singles examples** [L total] — examples.
- **Recording-from-simulations guide** [M] — docs.
- **Connection-management guide** [M] — docs.
- **Randomness guide** [S] — docs.
- **PyNEST tutorials series (4-part)** [L] — docs.
- **PyNEST API mapping reference** [M] — docs.
- **Define brainpy-extension parameter convention** [S] — neurons / docs.

### P2 (edge cases, polish)

Summarized at index level — see per-axis docs §7 for full lists. Highlights:
e-prop family port (XL), spatial / topology surface (XL), HH gap-junction
parity (M), `pong`/`sudoku` example ports (L), file-
backed recording backends (L), CI parity-check matrix (M), validation
progress badge (S), parallel-computing guide (M), glossary (S).

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
