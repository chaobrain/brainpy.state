# Novelty Assessment

> Part of the editorial report on [`../network-spec-dsl/`](../network-spec-dsl/). See [README](./README.md) for navigation.

## Per-claim assessment

| Claim                                                                                                  | Reviewer assessment |
|--------------------------------------------------------------------------------------------------------|---------------------|
| Declarative SNN spec with multi-backend dispatch                                                       | **Not new.** PyNN (since 2008), NeuroML, SONATA, Brian standalone all do this. |
| YAML/JSON archival form with schema validation                                                         | **Not new.** NeuroML/LEMS XML+schema is the prior art. |
| NIR export with six-class lossiness taxonomy and `nir.brainx.*` extension namespace                    | **Partially new.** NIR export exists in Norse / snnTorch / Sinabs / Rockpool; the *taxonomy with strict mode + metadata sidecar + reserved extension namespace* is a genuine refinement. |
| Frozen, content-hashable IR (units / `Trainable` / `DistRef` / `Noise` / `VariableRef` canonicalized)  | **Largely new for SNN DSLs.** PyTorch FX and JAX jaxpr offer analogues for ML, but not in the SNN / neuro space at this granularity. |
| Physical-units-first IR                                                                                | **New in this combination.** NeuroML carries units in XML; `saiunit`-bound IR with dimensional validation at construction is incrementally novel. |
| Trainability as a value-level marker propagating to `brainstate.nn.Param`                              | **New.** Most ML SNN libraries (snnTorch, Norse, BindsNET) make every weight trainable by default; few have a typed marker flowing through an IR. |
| Immutable IR + declared build-time variables (`net.variable` / `VariableRef`)                          | **Novel in this combination.** The closest prior art is Hydra / OmegaConf overrides (ML config layer, not IR-level) and jaxpr's lack of mutation (structural, not parameterizable-by-design). Declaring the variation surface up front, binding by name at build, and keeping the IR's `content_hash` independent of the binding supports sweep deduplication on structure. The `Trainable[VariableRef]` rejection rule is a sharp invariant. |
| Auto-registration over `braintools.conn` with weight / delay as rule attributes                        | **Incremental but useful.** Supplementary-rule path (`FixedIndegree`, `FixedOutdegree`, `FixedTotalNumber`, `PairwisePoisson`, `SymmetricPairwiseBernoulli`) ships with explicit upstreaming target to `braintools`. |
| End-to-end biophysical + deep-SNN coverage in one DSL                                                  | **Novel in degree.** PyNN has tried similar; this proposal's deep-SNN depth (DAG composability, layer macros, sequential stacks) plus multi-compartment cells (`spec.models.Cell` with `morphology` / `paint` / `place` / `cv_policy`) in the same IR is a real advance. |
| Training-paradigm pluralism over a single IR (BPTT / event-prop / RTRL / e-prop)                       | **Largely new at this granularity** — the §1.1 thesis. Every existing SNN framework commits to one training paradigm at model-definition time (snnTorch / Norse: BPTT-surrogate; BindsNET: BPTT + Hebbian; Nengo: NEF/PES; PyNN / Brian2: plasticity-only; Lava: on-chip plasticity). Lava-DL and hxtorch.snn are the closest comparators — they belong in the §1.1 prior-art table and currently are not. **As an *asserted* claim, well-positioned; as a *demonstrated* claim, not yet — see C2 / C3 / C4 in [`04-concerns.md`](./04-concerns.md).** |
| Six-layer concentric plasticity surface (per-proj → modulated → cross-projection eligibility → phased → structural → homeostatic) | **Novel.** No existing SNN DSL offers this stack at this granularity. Most pick either "rule registry with per-projection slot" (PyNN, Brian2) or "embedded in module" (snnTorch, Norse). Cross-projection eligibility via `EligibilitySource` / `EligibilityConsumer` with SPEC-041 enforcement is a real first. |
| Schedule grammar (`Phase` / `Phases` / `Trial`) as IR nodes                                            | **Novel in this combination.** Experiment protocols (warmup / ITI / stim / response / trial randomization) are typically user-side in SNN frameworks. Lifting them into the IR — with two consumers (plasticity gates, observable windows) sharing one schedule node — is substantive. |
| Signals as explicit modulator graph                                                                    | **Novel.** `SignalNode` with `source: ModelRef` (`External`, `PopulationRate`, `EligibilityTrace`, `FromState`) makes the third-factor / neuromodulator graph inspectable and visualizable. |
| Noise as composable value wrapper alongside `Trainable` / `DistRef`                                    | **Novel in this combination.** SDE-friendly noise terms attachable to neuron / synapse / plasticity-weight / input / signal parameters, with SDE integrator auto-selection at backend build, fold-in seed streams per noise term, and `meta["noise_terms"]` enumeration. Composition rules (`Trainable[DistRef[...]]` allowed; `Trainable[Noise]` rejected; `Trainable[VariableRef]` rejected; `Noise[..., VariableRef, ...]` allowed) are well-thought-out. |
| Tag- and predicate-driven views with set algebra                                                       | **Incremental.** Similar patterns in PyTorch FX (select-by-type) and Brian2 (tag-style group filtering). The typed query handle with `&` / `\|` / `-` and `pairwise` / `cross` / `per_pre` / `per_post` / `merged` broadcast modes is a useful refinement, not landmark novelty. |
| DAG composability with `temporal_offset` for cycles                                                    | **Incremental.** DAG composition is well-trodden (Keras Functional, PyTorch hooks, NetworkX). The `temporal_offset` (`same_step` / `next_step`) distinction with SPEC-042 same-step-cycle detection is a genuine refinement for spiking DAGs where the unrolled-time-axis question is load-bearing. |

## Headline novelty claim

The §1.1 framing is correct: **training-paradigm pluralism over a single IR** is the empty quadrant on the prior-art map and the right wedge. The supporting novel ideas — in descending order of distinctiveness — are:

1. Six-layer concentric plasticity surface with explicit cross-projection eligibility hooks.
2. Immutable IR + declared build-time variables with content-hash invariant under binding.
3. Schedule grammar as IR nodes + signals as explicit modulator graph.
4. Noise as composable value wrapper alongside `Trainable` / `DistRef`.
5. Lossy-taxonomy NIR export with `nir.brainx.*` extension namespace.
6. Content-hashable, units-first, frozen IR.

For an NC submission, claim 1 (the §1.1 thesis) is the headline; claims above are supporting evidence and should be foregrounded *as* supporting evidence, not as competing theses. The §1.1 framing handles this correctly; what is missing is the *demonstration* — see C2 / C3 / C4 in [`04-concerns.md`](./04-concerns.md).
