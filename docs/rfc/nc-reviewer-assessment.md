# Editorial Report — Network Specification DSL for brainpy.state

**Manuscript type:** Research/engineering RFC (`docs/rfc/network-spec-dsl.md`, 2,076 lines)
**Reviewer role:** Senior editor, computational neuroscience / neuromorphic computing
**Date:** 2026-05-13
**Verdict (preview):** Major revision before research-contribution status; currently a strong engineering specification with insufficient differentiation from prior art.

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

## Final Editorial Recommendation

**Decision: Major revision required.**

- **As an engineering specification for internal `brainpy.state` adoption:** the document is approximately ready. Address C2 (semantic surface), C3 (RNG contract), and C4 (training round-trip) before implementation; the rest can iterate with the codebase.
- **As a research contribution for a venue like *Nature Communications*, *PLOS Computational Biology*, or *Neuroinformatics*:** not yet suitable. The novelty is real but unisolated; the prior-art engagement is absent; the hardware story underdelivers on its claims; the spec/runtime semantic boundary is not honestly drawn.

A revised manuscript that (a) clearly differentiates from PyNN / NeuroML / SONATA / NIR, (b) foregrounds the genuinely novel ideas (content-hash IR + patch language + typed trainability + LIVE / LIVE_RESET / REBUILD taxonomy), (c) delivers either end-to-end hardware mapping for one platform or scopes G11 down honestly, and (d) commits to experiment-protocol primitives for reproducibility, would be a strong methods / systems contribution to the brain-inspired computing literature.

The technical work shown here is competent and frequently elegant. The remaining gap is editorial: name what is new, contrast it with what is not, and either deliver or scope the load-bearing claims (semantic determinism, hardware mapping). Done well, this becomes one of the better SNN DSL contributions of the year.
