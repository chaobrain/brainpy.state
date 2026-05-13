# Editorial Report — Network Specification DSL for brainpy.state

**Manuscript type:** Research/engineering RFC (`docs/rfc/network-spec-dsl.md`, 2,146 lines after revision 1)
**Reviewer role:** Senior editor, computational neuroscience / neuromorphic computing
**Date of original review:** 2026-05-13
**Date of revision-1 re-review:** 2026-05-13
**Verdict (preview, post-revision-1):** Still **major revision** before *Nature Communications*-grade research-contribution status. The novelty repositioning in the new §1.1 is a substantive improvement and is the *right* axis to claim; however, the load-bearing claim is now *asserted but not demonstrated*, and concerns C2–C11 from the original report are untouched. Suitable today for *Neuroinformatics* / *Frontiers in Neuroinformatics* / *JOSS*; not yet for NC.

---

## Revision 1 — re-review of the §1.1 / D28 rewrite

### What was changed

The revision is concentrated and surgical: a new **§1.1 "Novelty and prior art"** (~70 lines, three subsections) and a new **D28** in the decision log. The body of the document — goals (G1–G12), IR dataclasses (§5), backend protocols (§8), determinism contract (§13), validation catalog (§14), testing strategy (§16) — is unchanged.

The repositioning is from *"frozen IR + DSL surface"* (the prior framing) to:

> **The load-bearing novelty is that the same network description drives four mathematically distinct SNN training paradigms from a single IR — BPTT (with surrogate gradients), event-prop (Wunderlich & Pehle 2021), RTRL / forward-mode autodiff, and eligibility-trace methods (e-prop; Bellec et al. 2020).**

NIR export is now explicitly *demoted* as "fourth axis of pluralism (deployment), not the load-bearing novelty." A prior-art table contrasts snnTorch / Norse / BindsNET / Nengo / PyNN&Brian2 / Lava on the training-paradigm axis. D28 codifies the positioning as a tie-breaker for future scope decisions ("does this preserve neutrality across the four paradigms?").

### What this fixes

1. **Novelty is now a real, falsifiable, publishable claim.** "Single IR drives four mathematically distinct training paradigms" is concrete: it can be tested, refuted, and benchmarked. The original framing ("frozen IR + DSL") was not — every framework on the prior-art list has some flavor of that.
2. **The wedge is correctly identified.** Every existing SNN framework (snnTorch, Norse, Lava, Nengo, BindsNET, PyNN/Brian2) commits to *one* training paradigm at model-definition time. The training-paradigm-pluralism axis is the empty quadrant on the prior-art map. This is the right insight.
3. **The "JAX brought to autodiff / ONNX brought to inference" framing** is good positioning rhetoric — it tells a reviewer, in one sentence, what altitude this contribution claims.
4. **D28 as an architectural tie-breaker** is a strong bit of design discipline. It anchors future scope decisions to the central claim.

### What still blocks NC acceptance

The novelty *positioning* is now sufficient. The novelty *evidence* is not. The body of the spec must follow through on the §1.1 claim, and currently it does not. Specific, load-bearing gaps:

#### N1. The four-paradigm claim is asserted, not designed for

§8.2 names `bptt`, `eprop`, `event-prop` as shipped backends in a single table cell — but the spec gives **no design** for them, and the IR contains **no construct** that distinguishes their requirements:

- **Event-prop** demands exact gradients of spike times — i.e. differentiable thresholds and event-time implicit differentiation (Wunderlich & Pehle 2021; §3 of that paper). This is *not* the same surrogate-gradient signal BPTT uses; it requires the IR to carry threshold-crossing information. The current IR has no such construct.
- **RTRL / forward-mode** requires forward-mode JVPs through recurrent dynamics; a naive RTRL through arbitrary-topology recurrent surrogate-gradient computation has O(N⁴) memory and is intractable for the deep-SNN audience the spec also targets. Real RTRL backends (UORO, SnAp-k, online OE) impose IR constraints that the spec does not articulate.
- **Eligibility-trace / e-prop** needs cross-projection eligibility hooks — exactly the C8 plasticity-expressiveness gap from the original report, which the rewrite did not address. e-prop in Bellec et al. (2020) requires per-projection learning-signal pathways that the current `Projection` node does not expose.

A reviewer for NC will ask, on §1.1 page 1: *"show me, in the IR, what makes this claim mechanically possible."* The IR must surface backend-specific feature requirements (e.g., `requires_spike_time_differentiability`, `requires_third_factor_signal`); §5 currently does not.

#### N2. Per-training-backend capability matrix is missing

`BackendCapabilities` (§8.2) is a *uniform* schema across sim/train/export. But each training paradigm imposes radically different constraints on the IR:

| Paradigm | Common constraints (illustrative — must be made precise in §8.2) |
|---|---|
| BPTT (surrogate) | Any neuron model with a defined surrogate; arbitrary topology; bounded sequence length for memory |
| Event-prop | LIF / ALIF only (analytic threshold crossing); no plasticity during training; specific reset semantics |
| RTRL family | Bounded state size; specific recurrent topology constraints; forward-mode-friendly synapse models |
| e-prop | Local learning signals only; specific recurrent network topology; no global gradients |

Without this matrix written down per backend, "switch backend by changing one kwarg" collapses on first contact: the user who tries to swap `bptt` for `event-prop` on a Hodgkin–Huxley network will hit `BackendCapabilityError` and conclude the claim is hollow. The matrix is also load-bearing for the comparative-study story — the user needs to know *a priori* which IRs admit which paradigms.

#### N3. Comparative-study reproducibility requires C7 (now load-bearing)

The new framing's user story is *"compare event-prop vs BPTT on the same architecture."* For this to be reproducible — and reproducibility is what NC requires of methods contributions — the spec needs canonical **Protocol** (warm-up, epochs, reset semantics), **Dataset** (canonical references, splits, preprocessing), and **Optimizer / Loss / Schedule** abstractions. D21 explicitly defers loss/optimizer to the user. Combined with the absence of dataset and protocol abstractions, the spec does not deliver reproducibility for the very comparative study that §1.1 motivates. **C7 from the original report is now the central blocker, not a minor gap.**

#### N4. Prior-art table omits the most relevant comparative-study work

The §1.1.1 table is honest about the *frameworks* but misses the *literature* that already targets the training-paradigm-comparison wedge:

- **EXODUS** (Bauer, Lenz, Liu, Sheik 2023, *Frontiers in Neuroscience*) — explicitly compares SLAYER variants on the same architecture, very close to the "swap one kwarg" claim.
- **hxtorch.snn** (Pehle et al. 2022) — implements both surrogate-gradient and event-prop on a single substrate.
- **Norse** training-paradigm coverage extends to surrogate-gradient *and* SuperSpike, ADAM-based and Adjoint-based — beyond BPTT.
- **Lava-DL** (Intel) supports SLAYER on Loihi-targetable models — multi-paradigm in practice.
- **BrainCog** and **PySNN** also belong on this map.

For an NC submission, the table must engage these explicitly — they are the closest competing claims to the training-paradigm-pluralism wedge.

#### N5. Original C1 (prior art for the *DSL* axis) is only partially addressed

PyNN, Brian2, Nengo are now engaged at the framework-axis level. **NeuroML / LEMS, SONATA, NMODL, NESTML, and NIR itself** remain unengaged in the rewrite. NC has computational-neuroscience reviewers who will treat this as a fatal omission for any "declarative SNN spec" claim, even one whose primary novelty is on the training-paradigm axis. The Related Work section must engage them — particularly NeuroML/LEMS for the units + schema lineage, SONATA for the population/edge data-table form, and NIR for the canonical neuromorphic IR.

#### N6. The novelty claim is asymmetric — §1.1 vs the rest of the document

The new framing is not yet reflected in:

- **Goals (§2).** No G-line mentions training-paradigm pluralism; G7 mentions deep SNNs but not the training-strategy axis.
- **The IR (§5).** No node, no marker, no metadata block surfaces what each training paradigm requires. The IR cannot distinguish "this network is trainable under event-prop" from "this network is trainable under BPTT only."
- **Validation catalog (§14).** No SPEC-NNN code for "IR feature incompatible with chosen training backend"; SPEC-013 / SPEC-022 only handle plasticity-kind and layer-macro mismatches.
- **Testing strategy (§16).** No test category exercises the four-paradigm claim. The acceptance test should include: build IR; train under {BPTT, event-prop, e-prop} where each is supported; verify gradient signals are algorithmically distinct; verify the spec hash is bit-identical across the three runs.

The novelty claim feels grafted onto §1; it must be load-bearing through §2, §5, §8, §14, §16. As written, a reviewer who reads §1.1 then reads §5 will conclude "the IR was designed for BPTT and tagged for the others post hoc." That conclusion is currently defensible from the document.

### Specific, minimum revision to reach NC standard

The below is the *minimum* set of changes the document needs. Each maps to one or more concerns above.

1. **Add a new G-line** — *G13: Training-paradigm pluralism. The IR is neutral across BPTT, event-prop, RTRL/forward-mode, and eligibility-trace training. Switching paradigms is a backend-build kwarg; no IR rewrite is required. Backends declare per-paradigm constraints and surface incompatibilities at build time.* — and trace it through the rest of the document. (N1, N6)
2. **Per-training-backend capability matrix.** Write out, in §8.2, for each shipped training backend, what IR features it requires / forbids / approximates. (N2, N6)
3. **Add a §10.x training round-trip.** Walk through `bptt.build(ir) → train → trainer.parameters.diff() → ir.patch(...) → event-prop.build(...) → train` showing the comparison-study workflow end-to-end. (N3, N1)
4. **Demonstrate the claim on a published benchmark.** Reproduce one published result where event-prop and BPTT have been compared (e.g., Wunderlich & Pehle 2021 on YinYang or N-MNIST) by switching the backend on a single IR and matching published accuracy within reported error bars. This is the load-bearing experiment for NC; without it the claim is rhetoric. (N1, N3)
5. **Expand §1.1 prior-art engagement** to include EXODUS, hxtorch.snn, Lava-DL, BrainCog, and the comp-neuro axis (NeuroML/LEMS, SONATA, NMODL/NESTML, NIR). (N4, N5)
6. **Add SPEC-NNN error codes** for IR-feature × training-backend mismatches. (N6)
7. **Address C7 (Protocol / Dataset / Optimizer abstractions)** as load-bearing for the comparative-study story — at minimum a `Protocol(warmup, trial_structure, reset_policy)` and a thin `Dataset` reference. (N3, original C7)

The remaining concerns (C2 event semantics, C3 RNG contract, C4 training round-trip, C5 hardware honest-scoping, C6 biophysical scope, C8 plasticity, C9 DAG composability, C10 constraint vocabulary, C11 Builder/NetSpec coexistence) remain valid as *secondary* revision items but are not the blocker for NC suitability. The training-paradigm-pluralism claim's evidence — items 1–5 above — is the blocker.

### Updated final verdict for the §1.1 / D28 rewrite

- **Direction of the rewrite:** correct, sharp, and well-judged. The training-paradigm-pluralism axis is the right wedge; the prior-art table correctly identifies the empty quadrant; the JAX/ONNX framing is good positioning; D28 disciplines future scope decisions.
- **Sufficiency for *Nature Communications*:** **No.** The novelty is now correctly *positioned* but not yet correctly *demonstrated*. The IR (§5), goals (§2), validation (§14), and testing (§16) do not yet reflect the central claim, and the spec ships no worked comparative example.
- **Sufficiency for a methods venue with a lower evidence bar** (*Neuroinformatics*, *Frontiers in Neuroinformatics*, *JOSS*, *SoftwareX*): **Yes, with the original C2 / C3 / C4 / C7 fixes.** These venues accept the positioning + design + implementation without requiring a benchmarked comparative study.
- **Recommended path to NC:** complete revision items 1–7 above, ship a single working comparative-study notebook (event-prop vs BPTT on YinYang or similar), and resubmit. With those, the claim transitions from "asserted" to "demonstrated," and the contribution becomes one of the more interesting SNN-tooling submissions of the year.

---

## Executive Summary

The document specifies a frozen, content-hashable intermediate representation (`NetIR`) for spiking neural networks, with two equivalent frontends (a fluent Python builder and a YAML data DSL) and three backend families (simulation, training, export). The design is **engineering-mature**: error catalog, capability protocol, NIR export with a six-class lossiness taxonomy, dotted-path patch language for pre- and post-build parameter mutation, and a determinism contract over `(IR, backend, seed, dt)`.

However, as a *research contribution* it suffers from two strategic deficits. First, **prior art is conspicuously absent**: PyNN (the long-standing multi-simulator SNN DSL), NeuroML/LEMS, SONATA, NMODL, NESTML, Nengo's `Network`, snnTorch's IR, and even NIR itself are not cited or contrasted, despite all overlapping substantially with what is proposed. Second, **the semantic surface is uneven**: deep-SNN ergonomics are well covered, but biophysical, event-driven, plasticity, multi-compartment, and experiment-protocol semantics are either deferred to backends or omitted, weakening the claim that "the spec is the source of truth."

The work is publication-worthy as a *systems / methods* contribution after substantial revision. As shipped, it reads as an internal engineering specification rather than a research artifact.

---

## Major Strengths

1. **Frozen IR with content hash.** SHA-256 over canonical JSON, with explicit canonicalization of `u.Quantity`, `Trainable`, `DistRef`, `ConnRule`, `ModelRef`. This is a real reproducibility primitive — most SNN frameworks lack it.
2. **Two equivalent frontends.** The `NetSpec ↔ YAML ↔ NetIR` round-trip law is testable and gives both library users and archival workflows what they need.
3. **Physical units are first-class.** Carrying `saiunit` through the IR and validating dimensional consistency is the right call; the contrast with PyNN's mostly-numeric API is real.
4. **Unified path language for static and dynamic parameters.** `populations.exc.model.tau` addressing both pre-build edits and runtime `ParameterView` updates is a clean abstraction, and the three-class `LIVE / LIVE_RESET / REBUILD` taxonomy is a genuine engineering insight.
5. **Lossy-export taxonomy.** The six-class scheme (`LOSSLESS / RECORDED / APPROXIMATE / EXTENSION / DROPPED / UNSUPPORTED`) with stable `EXPORT-NIR-NNN` codes and strict/lenient modes is more rigorous than any existing NIR exporter the reviewer is aware of.
6. **Stable error code catalog (§14).** SPEC-NNN codes with construction/finalize/backend/mutation tiers make the spec actually documentable.
7. **Backend capability protocol.** `BackendCapabilities` with sets of supported kinds + a single validation pass is the right shape.
8. **Decision log (§18).** Twenty-seven captured decisions with rationale is exactly what an RFC should ship.

---

## Major Concerns

### C1. Novelty is asserted, not demonstrated (most consequential)

> **Status after revision 1: PARTIALLY ADDRESSED.** The new §1.1 reframes the contribution as training-paradigm pluralism over a single IR (BPTT / event-prop / RTRL / e-prop) rather than the DSL surface. PyNN, Brian2, Nengo, snnTorch, Norse, BindsNET, Lava are now engaged at the framework-axis level. **What is still missing:** (a) NeuroML/LEMS, SONATA, NMODL/NESTML, NIR are not engaged; (b) closer comparative-study prior art (EXODUS, hxtorch.snn, Lava-DL multi-paradigm coverage) is not addressed; (c) the new training-paradigm claim is asserted in §1.1 but not designed for in §5 / §8 / §14 / §16, and no worked comparative-study example demonstrates it. See **Revision 1 review, items N1–N6** at the top of this document.

The original observation, retained for context:

The introduction motivates the work against `brainpy_state._network` only. It does not engage **PyNN** (Davison 2009), **NeuroML/LEMS** (Cannon 2014, Gleeson 2010), **SONATA** (Dai 2020, Allen/BBP/Bluebrain), **NMODL/NESTML** (Plotnikov 2016, NEST community), **Nengo**'s declarative API, **snnTorch**, **Norse**, **BindsNET**, **Lava-DL**, **Rockpool**, or even **NIR** itself as a competing IR for the upstream layer. Many of the claimed contributions (declarative spec, multi-backend, hardware export, schema-validated YAML, sub-networks, recording) have existed for over a decade in this neighborhood.

The truly distinctive ideas — (a) saiunit physical-unit-first IR, (b) content-hash determinism contract, (c) trainability-as-a-marker that propagates from spec leaves to `brainstate.nn.Param`, (d) path-addressed runtime mutation with `LIVE/LIVE_RESET/REBUILD` typing, (e) end-to-end coexistence of biophysical and deep-SNN workloads in one IR, (f) auto-registration over `braintools.conn` rules with weight/delay as first-class rule attributes — must be *named, isolated, and contrasted*. As currently written, a reviewer cannot tell what is new.

### C2. The "spec is the source of truth" claim is partially honored

G1 says "users describe *what* the network is." Yet the IR delegates substantial semantics to backends:

- **Event-driven semantics.** Schedule discipline, queue order, zero-delay cycles, simultaneous-event handling — none in the IR. Two `event` backends could produce different spike trains from the same IR and still satisfy the protocol.
- **Plasticity semantics.** A `ModelRef.kind` for plasticity is registered, but the canonical update law (pre/post coupling, weight bounds, neuromodulator hooks, third-factor signals) is not in the spec. STDP, R-STDP, BCM, hebbian, and STP have radically different surfaces; the IR currently treats them all as opaque kinds.
- **Stochastic dynamics.** Stochastic neurons (noisy LIF, sigma-Wiener), stochastic synapses (release probability), stochastic plasticity — handled neither in `DistRef` (which is sample-once) nor in the determinism contract.
- **Recording cadence under event-driven simulation.** `Observable.every: u.ms` is dt-relative under `clock`; what is it under `event`?

Either these must enter the IR, or G1 must be softened to "spec is the source of truth for *structure*; semantics are jointly determined by `(IR, backend)`." The latter is honest and matches reality, but invalidates the bit-identity acceptance test across backend pairs.

### C3. Determinism contract has unstated dependencies

- §13 specifies `jax.random.fold_in` for connectivity sampling, but `braintools.conn.ConnectionResult.weights` is a `np.ndarray`. If `braintools.conn` rules internally use NumPy's `np.random` or a non-folded JAX path, determinism leaks. The spec must (a) require rules to accept a JAX key, (b) document the seed handoff, and (c) ban host-side RNG inside rules.
- "Bit-identical artifacts" depends on platform float repr (mentioned in §10), but cross-platform float canonicalization is not solved by `repr` in all Python builds. Specify the canonical float printer (e.g., `numpy.format_float_positional` with fixed precision, or a fixed-precision rational encoding for floats that round-trip).
- Content hash is over the IR. Trained weights live outside the IR. There is no contract for "hash of a trained model" — making provenance of training runs harder than provenance of *unbuilt* networks.

### C4. Training-to-inference round-trip is undefined

A user trains under `bptt`, gets parameters via `trainer.parameters`. Then they want to:

- Run inference in `clock` with the trained weights.
- Export to NIR for hardware.
- Archive a trained model alongside its spec.

§9.3.5 says "`Trainable` baked as constants at export time" but does not say *which* values — initial values from the IR, or current `ParamState` values from the trainer. §6.9.7 hints at `ParameterView.diff()` returning a `ParamPatch` list, but the round-trip pipeline (`trainer → diff → patches → spec.patch(*patches) → finalize → backends.nir.export`) is not made canonical. This is the most important user workflow and it is not in §10.

### C5. Hardware-mapping story is thin

G11 names Loihi, SpiNNaker, Nengo. §9 maps to NIR, then delegates everything to NIR-consuming toolchains. Missing:

- **Quantization** is "per-export-backend" (D25). For neuromorphic deployment this is the single most important transformation; absence of a canonical quantization contract makes the export story brittle.
- **Time-step alignment.** Loihi/SpiNNaker run discrete time-step / cycle-based dynamics; continuous-time HH or AdEx must be temporally discretized. The spec is silent on this transformation.
- **Routing constraints.** Loihi has fan-in/fan-out limits per neuron; the spec offers no way for an export backend to refuse a graph that exceeds them at IR-validation time (it would surface only at deploy time).
- **Per-core / per-chip placement.** No abstraction for partitioning, even as a hint.
- **Power / area annotations.** Not even as metadata.

Without these, "Neuromorphic-IR export" is a graph-rewrite, not a hardware-mapping contribution.

### C6. Spatial and biophysical scope is narrow

- **Neuron positions** are required by `DistanceDependent` and `Gaussian` connectivity but the IR has no canonical home for them (presumably in `PopulationNode.init`?). Make this explicit.
- **Multi-compartment / morphological models** are out of scope without comment. This forecloses NEURON / Arbor interoperation and biophysically detailed models — an explicit non-goal would be honest but should be stated.
- **Ion channels / Hodgkin–Huxley gating variables** are exposed only via `ModelRef.params`; HH is mapped to "UNSUPPORTED" in NIR. There is no path for biophysical interoperation analogous to NeuroML/LEMS.

### C7. Experiment, protocol, and dataset abstractions are missing

For both communities the spec serves (computational neuroscience and brain-inspired ML):

- **Protocols.** Warm-up windows, baseline epochs, stimulus blocks, ITI structure, reset semantics between trials.
- **Datasets.** Train/val/test references; canonical preprocessing; DVS-Gesture-style temporal streams.
- **Optimizer / loss / scheduler.** Deferred to user (D21). Combined with the absence of dataset and protocol abstractions, the spec does *not* deliver reproducibility for the deep-SNN community despite G4.

### C8. Plasticity expressiveness is shallow

- No third-factor / neuromodulator channels.
- No cross-projection eligibility-trace coupling.
- No plasticity scheduling (phase-based, gated).
- Plasticity vs. structural learning (synaptogenesis, pruning) not addressed.

### C9. Composability is restricted to chains

`Sequential` is sugar for chains; `Group` is organizational; `SubNetwork` is opaque-with-exports. There is no first-class:

- Skip / residual / DAG connection at the layer-macro level.
- Parallel branches with merge.
- Tag-driven selection (`spec.where(tag="excitatory")` style).
- Predicate-driven subpopulations (e.g., subset by a learned parameter).

For brain-inspired ML this matters because modern SNNs increasingly resemble transformer-shaped DAGs.

### C10. `Trainable` constraint vocabulary is too coarse for biophysics

`"positive" | "unit_norm" | "clip:lo,hi"` covers ML use, not biophysical priors (parameter coupling such as τ_m = R_m C_m, monotonicity along ion-channel kinetics, ratio constraints between rest/threshold/reset). The spec should either widen the vocabulary or hand off to a constraint registry.

### C11. Co-existence of `Builder` and `NetSpec` is a maintenance liability

§17 keeps both as user-facing. Two ways to describe a network multiply the test surface, divide documentation, and confuse new users. A clear deprecation path (or a sharp positioning: "Builder is internal substrate; do not import directly") would strengthen the design.

---

## Novelty Assessment

| Claimed novelty | Reviewer assessment |
|---|---|
| Declarative SNN spec with multi-backend dispatch | **Not new** — PyNN (since 2008), NeuroML, SONATA, Brian standalone all do this. |
| YAML/JSON archival form with schema validation | **Not new** — NeuroML/LEMS XML+schema is the prior art. |
| NIR export with lossy-mapping taxonomy | **Partially new** — NIR export exists in Norse/snnTorch/Sinabs/Rockpool. The *six-class lossiness taxonomy with strict mode + sidecar* is a genuine refinement. |
| Frozen, content-hashable IR | **Largely new for SNN DSLs.** PyTorch FX and JAX's jaxpr offer analogues for ML, but not in the SNN/neuro space at this granularity. |
| Physical-units-first IR | **New in this combination.** NeuroML carries units in XML; saiunit-bound IR with dimensional validation at construction is incrementally novel. |
| Trainability as a value-level marker propagating to `brainstate.nn.Param` | **New** — most ML SNN libraries (snnTorch, Norse, BindsNET) make every weight trainable by default; few have a typed marker that flows through an IR. |
| Path-addressed runtime mutation with `LIVE/LIVE_RESET/REBUILD` typing | **Novel and well-designed.** This is the most distinctive engineering idea in the document. |
| Auto-registration over `braintools.conn` with weight/delay as rule attributes | **Incremental but useful.** |
| End-to-end biophysical + deep-SNN coverage in one DSL | **Novel in degree, not in kind.** PyNN has tried similar but with weaker deep-SNN ergonomics; the depth of this proposal's deep-SNN coverage is real. |

**Net:** Two-to-three genuinely novel ideas (`ParamPatch` + `ParameterView` typing, content-hash IR with patch round-trip, lossy-taxonomy export). The rest is competent engineering on well-trodden ground. The novel ideas must be foregrounded.

---

## Significance Assessment

The proposal addresses a real need: the JAX-native SNN tooling stack (`brainstate` / `brainpy.state` / `braintools` / `brainevent`) currently lacks an upstream declarative layer comparable to PyNN-for-NEST. For the brainpy.state user base this is high-value engineering and would be the right next step.

For the *broader* computational-neuroscience and neuromorphic communities, significance is limited unless:

- (a) the spec is positioned as a *bridge* to existing standards (NeuroML, SONATA, PyNN) rather than a replacement;
- (b) the hardware-mapping story includes at least one concrete platform (Loihi or SpiNNaker) end-to-end, not just NIR-as-handoff;
- (c) the deep-SNN training story includes optimizer/dataset/protocol abstractions enabling reproducible benchmark reporting.

Without these, the contribution is significant *within* the brainpy ecosystem and minor outside it.

---

## Conceptual and Semantic Gaps

1. **Time-step semantics on cross-backend translation.** `dt` is a backend kwarg (D1), yet event-driven, fixed-step, and adaptive-step backends interpret `dt` incompatibly. The IR should at minimum carry `time_resolution_hint` for archival.
2. **Random-state hand-off across host/device boundaries.** Documented intent is JAX-key-based, but `braintools.conn.ConnectionResult` is host (NumPy). Spec the boundary.
3. **Population indexing under merge views.** `MergedViewHandle` denormalizes to one `ProjectionNode` per member at finalize, which means a single user projection lowers to N projections at the IR level. Observables, gradients, and `ParamPatch` paths must address these consistently — the spec does not show what happens to `projections[i]` paths when `i` indexes a denormalized projection.
4. **Wildcard semantics for `ParamPatch`.** `projections[*].rule.weight` is mentioned; the matching against denormalized merge-view projections, sequential-lowered projections, and sub-network-inlined projections is unspecified.
5. **Sub-network parameterization with shared trainables.** When the same `column_spec` is instantiated four times with different `N`, are their `Trainable` weights shared or independent? The spec does not say (probably independent, but parameter tying is a normal request for weight-sharing CNNs and recurrent cores).
6. **`Trainable` over `DistRef`** — does the trainable parameterize the distribution's hyperparameters (mean, std), or freeze the distribution sample and train the resulting tensor? The example `sp.train(init.Normal(...))` suggests the latter, but a user wanting to train `mean` itself has no syntax.
7. **`init` semantics on re-build.** §6.9.5 says `LIVE_RESET` resets corresponding state variables. For Trainable initial states, does `sim.reset()` resample from the (now possibly stale) distribution, or reuse the last sample? Specify.
8. **Versioning policy.** `netir/1.0` — no schema-evolution policy. State the compatibility rules: which changes are minor (additive fields, defaults), which are major.
9. **Float canonicalization across platforms.** The hash law depends on float repr stability; specify a precise encoder.
10. **Concurrency / re-entrancy of `ParameterView`.** Is `ParameterView.batch()` thread-safe? Re-entrant? What if a training step is mid-flight? Document the synchronization contract.

---

## Missing or Underdeveloped Features

| Category | What's missing |
|---|---|
| Spatial primitives | Canonical 3D position field; spatial kernels beyond Conv2d; distance-dependent connectivity grounded in stored positions. |
| Morphology | Compartmental / cable-equation models; integration with NEURON/Arbor. (Acceptable as explicit non-goal, but state it.) |
| Plasticity | Third-factor / neuromodulator channels; cross-projection eligibility traces; scheduled plasticity phases; structural plasticity. |
| Stochastic dynamics | Noise terms in neuron / synapse equations; not just stochastic inputs. |
| Experiment protocol | Trial structure, ITI, baselines, warm-up, multi-condition randomization. |
| Datasets | Canonical references, splits, preprocessing. |
| Optimizer / loss / schedule | Currently deferred to user; reproducibility regresses. Either canonicalize or document the trade-off explicitly. |
| DAG composability | Skip connections, parallel branches, merge points at the layer-macro level. |
| Tag-driven and predicate-driven views | `spec.where(tag=...)`, `spec.filter(...)`. |
| Constraint vocabulary | Biophysical priors (parameter coupling, ratios, monotonicity). |
| Hardware constraints | Fan-in/out, core / chip placement hints, quantization vocabulary, time-discretization. |
| Sweep strategies | Random / Sobol / Bayesian sweep beyond cartesian; resume / early stop. |
| Streaming recording | Disk-backed observables; downsampling reducers beyond mean/sum (quantiles, custom callables). |
| Provenance for trained artifacts | A canonical (IR, training-run, trained-parameter-set) bundle with its own hash. |
| Schema evolution | Migration tooling, deprecation policy, version-skew warnings. |
| Profiling / cost models | Memory and compute estimates from the IR (population × density × dt × duration). |

---

## Comparison With Existing Frameworks

The manuscript should add a Related Work section organized along the axes below. Reviewer's quick read across the relevant landscape:

### Capability matrix

| Property | PyNN | NeuroML / LEMS | SONATA | Nengo | snnTorch / Norse | NIR (alone) | **This proposal** |
|---|---|---|---|---|---|---|---|
| Declarative, framework-agnostic | yes | yes | yes (data) | partial | no (Pythonic) | yes (graph) | **yes** |
| Schema-validated archival | partial (XML v1.0) | yes (XSD / Schematron) | yes (HDF5+JSON) | no | no | yes (JSON) | **yes (JSON Schema)** |
| Physical units first-class | partial | yes | partial | yes (SI) | no | no | **yes (saiunit)** |
| Content-hashable IR | no | no | partial | no | no | no | **yes** |
| Trainability as IR concept | no | no | no | partial | yes (Python) | no | **yes (typed marker)** |
| Path-addressed runtime mutation | no | no | no | no | no | no | **yes (`ParamPatch`)** |
| Deep-SNN layer macros | no | no | no | partial | yes | partial | **yes** |
| Biophysical / multi-compartment | partial | yes | yes | no | no | no | no |
| Multi-simulator (NEST / NEURON / Brian) | **yes** | yes | yes | no | no | yes (via consumers) | partial (clock / event only) |
| Hardware export | partial | partial | no | yes (Loihi) | yes (via NIR) | **yes (canonical)** | yes (via NIR) |
| Protocol / experiment abstractions | no | no | partial | no | no | no | no |

### Audience and overlap

| Framework | Primary audience | Strongest overlap with this proposal | Key differentiator the proposal must claim |
|---|---|---|---|
| **PyNN** | Computational neuroscience, multi-simulator interop | Populations, projections, recording, sub-networks, schema | Physical-unit-first IR; content-hash determinism; trainability marker; deep-SNN layer macros |
| **NeuroML / LEMS** | Biophysical modeling, archival | Schema-validated declarative form, units, multi-tool consumption | JAX-native runtime substrate; trainability + ML-style backends; ParamPatch language |
| **SONATA** | Large-scale circuit data (Allen / BBP) | Population/edge tables, file-format archival | In-memory IR + Pythonic builder; pre/post-build mutation; export-backend protocol |
| **Nengo** | Neural-engineering framework, Loihi deployment | Declarative `Network`, hardware backends | Multi-paradigm support (biophysical + deep-SNN); frozen content-hashable IR |
| **snnTorch / Norse** | Deep-SNN / brain-inspired ML | Layer macros, trainable weights, NIR export | Spec layer above the model code; YAML archival; ParamPatch + LIVE/LIVE_RESET/REBUILD |
| **BindsNET / Lava-DL / Rockpool** | Brain-inspired ML, neuromorphic | Layer macros, plasticity, NIR export (Rockpool/Sinabs) | Declarative IR rather than imperative Module; cross-paradigm coverage |
| **NIR** | Neuromorphic IR for deployment | Graph IR, hardware consumer ecosystem | The proposal *produces* NIR; the differentiator is the rich upstream spec, the unit / trainability / seed sidecar, and the six-class lossiness taxonomy |
| **NMODL / NESTML** | Biophysical model description, compilation | Declarative model description with units | Network-level (not single-cell) scope; ML-friendly trainability |

### Bottom line on differentiation

The proposal's strongest dimensions are **physical units in the IR**, **content-hash IR**, **trainability marker propagating to `brainstate.nn.Param`**, and the **ParamPatch / ParameterView path language**. Its weakest are **multi-simulator interop** (NEST / NEURON / Brian are not reachable backends), **biophysical / multi-compartment coverage**, and **experiment-protocol abstractions**.

---

## Concrete Improvement Suggestions

### Essential (block research-contribution status)

1. **§1 must engage prior art.** Add a 1–2 page Related Work section contrasting with PyNN, NeuroML/LEMS, SONATA, Nengo, snnTorch, Norse, BindsNET, Lava-DL, Rockpool, and NIR. Identify (and label) the genuinely novel contributions explicitly.
2. **Soften G1 or strengthen the IR.** Either move event-scheduling, plasticity update laws, recording cadence, and stochastic dynamics into the IR, or rewrite G1/G4 to acknowledge that semantics are jointly determined by `(IR, backend)`. The current text overclaims.
3. **Document the train → infer → export round-trip.** Add §10.x showing how `Trainer.parameters.diff()` becomes a `ParamPatch` list → `spec.patch(...)` → `nir.export` with trained values. This is the single most important user workflow and is currently implicit.
4. **Specify the RNG contract end-to-end.** `braintools.conn` rules must accept a JAX key; ban host RNG; document float canonicalization; specify the trained-artifact hash.
5. **Add experiment / protocol primitives.** At minimum: `Protocol(warmup, epochs, reset_policy)` and `Trial` abstractions. Without these, reproducibility for deep-SNN benchmarks is user-discipline.
6. **Position the hardware story honestly.** Either deliver a Loihi-or-SpiNNaker end-to-end mapping (quantization vocabulary, time discretization, routing constraints) or scope G11 to "graph-level NIR export; deployment is consumer-toolchain responsibility."
7. **Specify versioning and schema evolution.** Document compatibility rules between `netir/1.0` → `1.x` → `2.0`, the migration tool surface, and version-skew handling at load time.

### High-priority (improve rigor and adoption)

8. **Multi-compartment / morphology:** state explicitly as a non-goal in §2.2 (it currently is not listed), or sketch a future extension path.
9. **Plasticity vocabulary:** at minimum, formalize STDP / STP / R-STDP / homeostatic plasticity as registered protocols with documented pre/post coupling and third-factor inputs.
10. **DAG composability:** add `parallel(...)`, skip-connection sugar, and predicate-driven views (`spec.where`).
11. **Constraint registry:** widen `Trainable.constraint` to a registry with documented options, including biophysical priors.
12. **Recording reducers as a registry:** support quantiles, rolling statistics, custom callables.
13. **Spatial positions as a canonical IR field** on `PopulationNode` rather than `init`.
14. **Define `MergedView` projection semantics rigorously** so wildcards in `ParamPatch` paths and trainable name resolution are unambiguous after denormalization.
15. **Concrete deprecation plan for `Builder`** or a hard "internal substrate" boundary.

### Optional (adoption polish)

16. **Add a profiling / cost model:** memory + flops estimate from the IR alone.
17. **Sweep strategies beyond cartesian:** Sobol, random, Bayesian, with resume / early-stop hooks.
18. **Streaming recording back-end:** disk-backed `TraceBundle` for long simulations.
19. **Patch portability across renames:** ship the `bp-spec patch migrate` helper proposed as an open question.
20. **NIR extension namespace coordination:** make the upstream NIR coordination a release blocker, not an open question.
21. **Trace-level provenance:** include the `(ir_hash, trainer_log_hash, param_diff_hash)` bundle as a canonical archive object.

---

## Final Editorial Recommendation (post-revision-1)

**Decision: Major revision required for *Nature Communications*. The revision-1 rewrite is a substantive improvement on novelty positioning but is not yet sufficient on novelty evidence.**

- **As an engineering specification for internal `brainpy.state` adoption:** the document is approximately ready. Address C2 (semantic surface), C3 (RNG contract), and C4 (training round-trip) before implementation; the rest can iterate with the codebase.
- **As a research contribution for *Neuroinformatics* / *Frontiers in Neuroinformatics* / *JOSS* / *SoftwareX*:** the §1.1 rewrite plus C2 / C3 / C4 / C7 fixes likely clears the bar. The training-paradigm-pluralism positioning is publishable at this tier on positioning + design alone.
- **As a research contribution for *Nature Communications*:** **not yet suitable.** The §1.1 rewrite correctly identifies the empty quadrant on the prior-art map (every existing SNN framework commits to one training paradigm) and pivots the claim to that wedge — this is the right move and is necessary. But the body of the spec must follow through: the IR (§5) must surface what each paradigm requires, the per-training-backend capability matrix (§8.2) must be written out, validation (§14) must catch IR × backend mismatches, testing (§16) must exercise the four-paradigm claim, and **at least one published comparative-study result must be reproduced by switching backend on a single IR** (e.g., event-prop vs BPTT on YinYang or N-MNIST, matching Wunderlich & Pehle 2021 within reported error bars).

A *Nature Communications*-grade revision needs all seven items in **Revision 1 review § "Specific, minimum revision to reach NC standard"** above:

1. New G13 for training-paradigm pluralism, traced through goals / IR / validation.
2. Per-training-backend capability matrix in §8.2.
3. Worked §10.x training round-trip across backends.
4. Reproduced published benchmark via single-line backend swap.
5. Expanded §1.1 prior art including EXODUS, hxtorch.snn, Lava-DL, NeuroML/LEMS, SONATA, NMODL/NESTML, NIR.
6. New SPEC-NNN codes for IR × training-backend mismatches.
7. Protocol / Dataset / minimal Optimizer abstractions to make comparative-study reproducibility achievable (the load-bearing form of original C7).

The remaining concerns from the original report (C2, C3, C4, C5, C6, C8, C9, C10, C11) are valid as secondary revision items but are not the NC blocker. The training-paradigm-pluralism claim's evidence is the blocker.

The technical work shown here remains competent and frequently elegant, and the §1.1 rewrite is sharp and well-judged in direction. The remaining gap is no longer editorial positioning — that has been substantially fixed — it is *evidence*: the IR and the backends must visibly mechanize the claim that §1.1 makes, and a single end-to-end comparative study must demonstrate it. With those in place, this becomes a credible *Nature Communications* methods/systems submission and arguably one of the more important SNN-tooling contributions of the year.
