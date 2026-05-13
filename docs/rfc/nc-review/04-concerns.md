# Major Concerns

> Part of the editorial report on [`../network-spec-dsl.md`](../network-spec-dsl.md). See [README](./README.md) for navigation. C1 is partially addressed by revision 1 (see note below); C2–C11 remain valid as written. New concerns introduced by revision 1 (N1–N6) are in [`01-revision-1-review.md`](./01-revision-1-review.md).

## C1. Novelty is asserted, not demonstrated (most consequential)

> **Status after revision 1: PARTIALLY ADDRESSED.** The new §1.1 reframes the contribution as training-paradigm pluralism over a single IR (BPTT / event-prop / RTRL / e-prop) rather than the DSL surface. PyNN, Brian2, Nengo, snnTorch, Norse, BindsNET, Lava are now engaged at the framework-axis level. **What is still missing:** (a) NeuroML/LEMS, SONATA, NMODL/NESTML, NIR are not engaged; (b) closer comparative-study prior art (EXODUS, hxtorch.snn, Lava-DL multi-paradigm coverage) is not addressed; (c) the new training-paradigm claim is asserted in §1.1 but not designed for in §5 / §8 / §14 / §16, and no worked comparative-study example demonstrates it. See [`01-revision-1-review.md`](./01-revision-1-review.md), items N1–N6.

The original observation, retained for context:

The introduction motivates the work against `brainpy_state._network` only. It does not engage **PyNN** (Davison 2009), **NeuroML/LEMS** (Cannon 2014, Gleeson 2010), **SONATA** (Dai 2020, Allen/BBP/Bluebrain), **NMODL/NESTML** (Plotnikov 2016, NEST community), **Nengo**'s declarative API, **snnTorch**, **Norse**, **BindsNET**, **Lava-DL**, **Rockpool**, or even **NIR** itself as a competing IR for the upstream layer. Many of the claimed contributions (declarative spec, multi-backend, hardware export, schema-validated YAML, sub-networks, recording) have existed for over a decade in this neighborhood.

The truly distinctive ideas — (a) saiunit physical-unit-first IR, (b) content-hash determinism contract, (c) trainability-as-a-marker that propagates from spec leaves to `brainstate.nn.Param`, (d) path-addressed runtime mutation with `LIVE/LIVE_RESET/REBUILD` typing, (e) end-to-end coexistence of biophysical and deep-SNN workloads in one IR, (f) auto-registration over `braintools.conn` rules with weight/delay as first-class rule attributes — must be *named, isolated, and contrasted*. As currently written, a reviewer cannot tell what is new.

## C2. The "spec is the source of truth" claim is partially honored

G1 says "users describe *what* the network is." Yet the IR delegates substantial semantics to backends:

- **Event-driven semantics.** Schedule discipline, queue order, zero-delay cycles, simultaneous-event handling — none in the IR. Two `event` backends could produce different spike trains from the same IR and still satisfy the protocol.
- **Plasticity semantics.** A `ModelRef.kind` for plasticity is registered, but the canonical update law (pre/post coupling, weight bounds, neuromodulator hooks, third-factor signals) is not in the spec. STDP, R-STDP, BCM, hebbian, and STP have radically different surfaces; the IR currently treats them all as opaque kinds.
- **Stochastic dynamics.** Stochastic neurons (noisy LIF, sigma-Wiener), stochastic synapses (release probability), stochastic plasticity — handled neither in `DistRef` (which is sample-once) nor in the determinism contract.
- **Recording cadence under event-driven simulation.** `Observable.every: u.ms` is dt-relative under `clock`; what is it under `event`?

Either these must enter the IR, or G1 must be softened to "spec is the source of truth for *structure*; semantics are jointly determined by `(IR, backend)`." The latter is honest and matches reality, but invalidates the bit-identity acceptance test across backend pairs.

## C3. Determinism contract has unstated dependencies

- §13 specifies `jax.random.fold_in` for connectivity sampling, but `braintools.conn.ConnectionResult.weights` is a `np.ndarray`. If `braintools.conn` rules internally use NumPy's `np.random` or a non-folded JAX path, determinism leaks. The spec must (a) require rules to accept a JAX key, (b) document the seed handoff, and (c) ban host-side RNG inside rules.
- "Bit-identical artifacts" depends on platform float repr (mentioned in §10), but cross-platform float canonicalization is not solved by `repr` in all Python builds. Specify the canonical float printer (e.g., `numpy.format_float_positional` with fixed precision, or a fixed-precision rational encoding for floats that round-trip).
- Content hash is over the IR. Trained weights live outside the IR. There is no contract for "hash of a trained model" — making provenance of training runs harder than provenance of *unbuilt* networks.

## C4. Training-to-inference round-trip is undefined

A user trains under `bptt`, gets parameters via `trainer.parameters`. Then they want to:

- Run inference in `clock` with the trained weights.
- Export to NIR for hardware.
- Archive a trained model alongside its spec.

§9.3.5 says "`Trainable` baked as constants at export time" but does not say *which* values — initial values from the IR, or current `ParamState` values from the trainer. §6.9.7 hints at `ParameterView.diff()` returning a `ParamPatch` list, but the round-trip pipeline (`trainer → diff → patches → spec.patch(*patches) → finalize → backends.nir.export`) is not made canonical. This is the most important user workflow and it is not in §10.

## C5. Hardware-mapping story is thin

G11 names Loihi, SpiNNaker, Nengo. §9 maps to NIR, then delegates everything to NIR-consuming toolchains. Missing:

- **Quantization** is "per-export-backend" (D25). For neuromorphic deployment this is the single most important transformation; absence of a canonical quantization contract makes the export story brittle.
- **Time-step alignment.** Loihi/SpiNNaker run discrete time-step / cycle-based dynamics; continuous-time HH or AdEx must be temporally discretized. The spec is silent on this transformation.
- **Routing constraints.** Loihi has fan-in/fan-out limits per neuron; the spec offers no way for an export backend to refuse a graph that exceeds them at IR-validation time (it would surface only at deploy time).
- **Per-core / per-chip placement.** No abstraction for partitioning, even as a hint.
- **Power / area annotations.** Not even as metadata.

Without these, "Neuromorphic-IR export" is a graph-rewrite, not a hardware-mapping contribution.

## C6. Spatial and biophysical scope is narrow

- **Neuron positions** are required by `DistanceDependent` and `Gaussian` connectivity but the IR has no canonical home for them (presumably in `PopulationNode.init`?). Make this explicit.
- **Multi-compartment / morphological models** are out of scope without comment. This forecloses NEURON / Arbor interoperation and biophysically detailed models — an explicit non-goal would be honest but should be stated.
- **Ion channels / Hodgkin–Huxley gating variables** are exposed only via `ModelRef.params`; HH is mapped to "UNSUPPORTED" in NIR. There is no path for biophysical interoperation analogous to NeuroML/LEMS.

## C7. Experiment, protocol, and dataset abstractions are missing

For both communities the spec serves (computational neuroscience and brain-inspired ML):

- **Protocols.** Warm-up windows, baseline epochs, stimulus blocks, ITI structure, reset semantics between trials.
- **Datasets.** Train/val/test references; canonical preprocessing; DVS-Gesture-style temporal streams.
- **Optimizer / loss / scheduler.** Deferred to user (D21). Combined with the absence of dataset and protocol abstractions, the spec does *not* deliver reproducibility for the deep-SNN community despite G4.

## C8. Plasticity expressiveness is shallow

- No third-factor / neuromodulator channels.
- No cross-projection eligibility-trace coupling.
- No plasticity scheduling (phase-based, gated).
- Plasticity vs. structural learning (synaptogenesis, pruning) not addressed.

## C9. Composability is restricted to chains

`Sequential` is sugar for chains; `Group` is organizational; `SubNetwork` is opaque-with-exports. There is no first-class:

- Skip / residual / DAG connection at the layer-macro level.
- Parallel branches with merge.
- Tag-driven selection (`spec.where(tag="excitatory")` style).
- Predicate-driven subpopulations (e.g., subset by a learned parameter).

For brain-inspired ML this matters because modern SNNs increasingly resemble transformer-shaped DAGs.

## C10. `Trainable` constraint vocabulary is too coarse for biophysics

`"positive" | "unit_norm" | "clip:lo,hi"` covers ML use, not biophysical priors (parameter coupling such as τ_m = R_m C_m, monotonicity along ion-channel kinetics, ratio constraints between rest/threshold/reset). The spec should either widen the vocabulary or hand off to a constraint registry.

## C11. Co-existence of `Builder` and `NetSpec` is a maintenance liability

§17 keeps both as user-facing. Two ways to describe a network multiply the test surface, divide documentation, and confuse new users. A clear deprecation path (or a sharp positioning: "Builder is internal substrate; do not import directly") would strengthen the design.
