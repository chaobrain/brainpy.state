# Major Strengths

> Part of the editorial report on [`../network-spec-dsl/`](../network-spec-dsl/). See [README](./README.md) for navigation.

1. **Frozen, content-hashable IR.** SHA-256 over canonical JSON, with explicit canonicalization of `u.Quantity`, `Trainable`, `DistRef`, `Noise`, `VariableRef`, `ConnRule`, `ModelRef`. The hash law is the right reproducibility primitive — most SNN frameworks lack one — and the content hash is independent of the build-time variable binding (§9.1), which is what makes sweeps share upstream caches correctly.

2. **Two equivalent frontends with round-trip law.** `NetSpec ↔ YAML/JSON ↔ NetIR` is a testable equivalence (§5.2), and every new value-wrapper / view-kind / composition form has a documented YAML surface (§4, §11.2 cheat sheet). Archival workflows and library workflows share one source of truth.

3. **Physical units are first-class.** `saiunit` flows through the IR; SPEC-006 catches wrong-unit-dimension parameters at construction; the NIR exporter strips to canonical SI and records the original units in the metadata sidecar (`EXPORT-NIR-008`). The dimensional contract is end-to-end.

4. **Immutable IR with declared build-time variables.** The IR has *no* path-addressed mutation API; cross-run variation is declared up front via `net.variable(name, default, *, constraint, required)` and bound by name at `backend.build(ir, ..., variables={...})` (§3.14). The `Trainable[VariableRef]` rejection rule ("a leaf is either trained or bound at build, not both") is a sharp invariant. The variable-binding determinism contract (§9.1) plus content-hash invariance under binding is exactly the shape sweep deduplication needs.

5. **Six-layer concentric plasticity surface (§3.12).** Per-projection → modulated (third-factor via signal handles bound to rule-declared roles) → cross-projection eligibility (`EligibilitySource` / `EligibilityConsumer`; SPEC-041 enforces trace resolution) → phased / trial-structured (`plasticity_schedule=`) → structural (`REBUILD` default; `live_topology` capability-gated) → homeostatic / meta-plasticity. Each layer is strictly more expressive than the previous; users pay for what they use; the e-prop substrate (cross-projection eligibility hooks) is explicitly present rather than re-invented per backend.

6. **Lossy-export taxonomy with `nir.brainx.*` extension namespace (§6).** Six classes (`LOSSLESS / RECORDED / APPROXIMATE / EXTENSION / DROPPED / UNSUPPORTED`) with stable `EXPORT-NIR-NNN` codes (§9.2.2), strict/lenient modes, a metadata sidecar preserving units / trainable names / seeds / stochastic-input parameters, and an opt-in extension namespace for constructs that cannot map to NIR core (`SpikeTimes`, `Concat`, future `ALIF` adaptation node). This is more rigorous than any existing NIR exporter the reviewer is aware of.

7. **Stable error-code catalog (§9.2).** `SPEC-001…026` (construction / finalize / backend / build) plus `SPEC-040…043` (spatial / eligibility / cycle / empty-query) plus `EXPORT-NIR-001…010`. The catalog is comprehensive enough to be documentable and gives the spec a debuggable error surface from day one.

8. **Backend capability protocol (§5.1.3).** `BackendCapabilities` with `supports_delay`, `supports_plasticity`, `supports_distributions`, `supports_training`, `supports_batch`, `supports_positions`, `supports_morphology`, `supports_noise`, `supports_signals`, `supports_schedules`, `supports_structural_plasticity`, `supports_graphs`, plus the kind-sets (`supported_neuron_kinds`, `supported_synapse_kinds`, `supported_output_kinds`, `supported_rules`, `supported_layer_macros`, `supported_input_kinds`). The schema is the right shape for the §1.1 paradigm-pluralism claim; the only gap is the per-shipped-backend instantiation.

9. **Decision log (§11.1).** Twenty-nine captured decisions with rationale. D26 / D27 (mutation-surface collapse), D28 (training-paradigm-pluralism positioning as architectural tie-breaker), and D29 (backend module location) are particularly sharp pieces of design discipline.

10. **Spatial and morphological coverage.** §3.5.2 ships `spec.geometry.{Grid1d, Grid2d, Grid3d, HexGrid2d, Free, Layered}` with explicit `positions` carried on `PopulationNode`, the full `braintools.conn._spatial` family, and a `spec.kernel.*` × `spec.mask.*` decomposition vocabulary. §3.5.3 ships `spec.models.Cell` with morphology / `paint` / `place` / `cv_policy` / `solver` / `spike_threshold`. §3.6.3 ships compartment-targeted projections; §3.9.4 / §3.9.5 ship spatial and compartmental views. Per-backend capability table is explicit.

11. **Schedule grammar as IR nodes (§3.7.3).** `spec.schedule.{Phase, Phases, Trial}` is a small, focused grammar that closes the experiment-protocol gap (warmup / ITI / stim / response, trial randomization, phase-gated learning and observables) without ballooning into a general experiment framework. Two consumers (plasticity gates, observable windows) sharing one schedule node is the right shape.

12. **Signals as explicit modulator graph (§3.7.2).** `SignalNode` with a typed `source: ModelRef` (`External`, `PopulationRate`, `EligibilityTrace`, `FromState`) makes the third-factor / neuromodulator graph inspectable and visualizable. Reward-modulated STDP without hidden globals is the canonical use case.

13. **Noise as a composable value wrapper (§3.10.3).** `Noise(kind, params, seed_tag)` slots into `ModelRef.params["noise"]` (or `["weight_noise"]` for plasticity) alongside `Trainable` and `DistRef`. SDE integrator auto-selection at backend build with `InfoNotice("SDEIntegratorChosen", ...)` preserves G1 ("describe what, not how to step"). Per-noise-term `fold_in(build_seed, "noise/<scope>/<var>")` keeps determinism end-to-end.

14. **DAG composability (§3.11.3).** `net.graph` as the explicit DAG form, merge-arity layer macros (`Add` / `Concat` / `Mul` / `Gate` / `Split` / `Tee`), `fork()` for parallel branches with a merge layer, the `|` operator for chain sugar, and `temporal_offset` (`same_step` / `next_step`) for cyclic graphs with SPEC-042 same-step-cycle detection. Covers modern deep-SNN architectures (ResNet, skip, transformer-shaped DAGs, top-down feedback).

15. **Backend module location (§5.1.1; D29).** Backends are peer top-level modules under `brainpy.state` (`clock`, `event`, `bptt`, `eprop`, `eventprop`, `ppprop`, `nir`, `onnxspike`). The spec module stays paradigm-neutral and dependency-light; switching gradient flavors is a one-line `from brainpy.state import <backend>` change, mirroring the load-bearing §1.1 novelty.

16. **Comprehensive worked example (§3.15).** The cortex-striatum loop in §3.15 touches *every* extension axis — spatial populations, morphological cells, reward / dopamine signals, DAG encoder with skip, plasticity with reward + schedule + eligibility, predicate-driven homeostatic scaling, phase-gated observables — and lowers cleanly to `eprop.build(...)`. It is the canonical demonstration that the surface composes.
