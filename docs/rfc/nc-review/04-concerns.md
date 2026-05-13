# Open Concerns

> Part of the editorial report on [`../network-spec-dsl/`](../network-spec-dsl/). See [README](./README.md) for navigation. This file lists only currently open concerns. Resolved concerns are removed.

## C1. Novelty engagement is incomplete

The §1.1 framing — *training-paradigm pluralism over a single IR* — is the right wedge, and the prior-art table covers the framework axis (snnTorch, Norse, BindsNET, Nengo, PyNN, Brian2, Lava). What is still missing:

- **The declarative-SNN axis is under-engaged.** NeuroML / LEMS, SONATA, NMODL / NESTML, and NIR-as-upstream-IR (rather than as the export target) belong in the prior-art table. NC computational-neuroscience reviewers will treat this as a fatal omission for a "declarative SNN spec" claim.
- **The closest multi-paradigm prior art is missing.** EXODUS (Bauer, Lenz, Liu, Sheik 2023) compares SLAYER variants on the same architecture; hxtorch.snn (Pehle et al. 2022) implements surrogate-gradient and event-prop on a single substrate; Lava-DL supports SLAYER on Loihi-targetable models. These are the closest claims to the wedge §1.1 names and must be engaged.
- **The novelty claim is *expressively supported but not mechanically demonstrated*.** The IR has the surface area the comparative study would need (signals, schedules, eligibility traces, DAG, noise, build-time variables), but no IR construct distinguishes which training paradigm a given network admits — see C2 and C3 below.

## C2. The four-paradigm claim is not mechanized in the IR

§1.1 asserts that BPTT / event-prop / RTRL / e-prop can swap by changing a backend kwarg, but the IR (§2) carries no node, marker, or metadata block that distinguishes their requirements. The only paradigm-aware machinery is the `Trainable` value wrapper and the `BackendCapabilities` schema; the former is paradigm-neutral and the latter checks at build time, not at spec time.

What is missing:

- **`requires_spike_time_differentiability`** capability on neuron signatures — so event-prop's "LIF / ALIF only" precondition is declarative.
- **`requires_third_factor_signal`** annotation on plasticity rules — so e-prop's local-learning-signal requirement is declarative (the eligibility-trace surface in §3.12.3 has the *structure* but no marker saying "this rule needs a third-factor signal").
- **`supports_recurrent_state_size`** bound on RTRL-family backends — so the IR can be rejected at build for forward-mode-intractable topologies.

A reader of §1.1 cannot identify what in the IR makes the event-prop / BPTT swap mechanically possible vs merely asserted.

## C3. Per-shipped-training-backend capability matrix is not written

`BackendCapabilities` (§5.1.3) is substantial as a *schema*, but §5.1.1's table only names the shipped backends (`bptt`, `eprop`, `eventprop`, `ppprop`) — it does not show what each declares in its `capabilities` field. Without that, "switch paradigm by changing one line" is a promise rather than a contract.

The minimum table for §5.1 should look approximately like:

| Paradigm   | `supports_plasticity` | `supports_noise` | `supports_morphology` | `supports_structural_plasticity` | Threshold semantics | Topology constraint | Notes                                                 |
|------------|-----------------------|------------------|------------------------|----------------------------------|---------------------|---------------------|--------------------------------------------------------|
| `clock`    | yes                   | yes              | yes                    | yes                              | hard reset          | arbitrary           | reference substrate                                   |
| `event`    | yes                   | partial          | no                     | live_topology only               | exact threshold     | no zero-delay cycle | SPEC-014 enforced                                     |
| `bptt`     | yes                   | yes              | differentiable solvers | rebuild only                     | surrogate gradient  | bounded T memory    | trainables become `brainstate.nn.Param`               |
| `eprop`    | yes                   | partial          | no                     | rebuild only                     | surrogate gradient  | local learning sig. | requires `EligibilitySource` or third-factor signal   |
| `eventprop`| **no**                | **no**           | no                     | no                               | exact spike time    | LIF/ALIF only       | analytic adjoint                                      |
| `ppprop`   | TBD                   | TBD              | TBD                    | TBD                              | TBD                 | TBD                 | see `/mnt/d/codes/projects/braintrace`                |

Each row is a concrete pre-build contract; the absence of these rows is the load-bearing gap for the §1.1 claim.

## C4. No worked comparative-study benchmark

For an NC submission, the load-bearing acceptance test is *reproduce one published result where event-prop and BPTT (or any two of the four paradigms) have been compared* by switching the backend on a single IR — e.g. Wunderlich & Pehle (2021) on YinYang or N-MNIST. Without this, the §1.1 claim remains rhetorically positioned but empirically undemonstrated.

The §10.2 testing strategy covers determinism, backend equivalence (Brunel on clock vs event), NIR round-trip, capability mismatches, and variable binding rigorously — but not the four-paradigm comparison. A notebook in `docs/examples/` plus a CI smoke test is the minimum.

## C5. Spec / runtime semantic boundary remains soft on event-driven semantics

Most of the original semantic-boundary concerns are now in the IR (schedules, signals, noise terms, structural-plasticity flag, modulator graph, temporal-offset, reducer vocabulary). What remains backend-determined: **queue order, simultaneous-event handling, and observable cadence under event-driven simulation**. Two `event` backends could produce different spike trains from the same IR and still satisfy the protocol. SPEC-014 catches zero-delay recurrent cycles but does not pin queue order.

Either pin these in the IR (and lose some `event` backend implementation freedom) or state in §1 / G1 that *semantics are jointly determined by `(IR, backend)`* for event-driven cases. The current text overclaims slightly on G1.

## C6. Determinism contract still relies on `repr` for float canonicalization

§9.1 spells out the fold-in chain, per-projection seeds, variable-binding determinism, export-determinism, and per-mode visualization-determinism semantics — substantively. But §5.2 still says *"floats formatted with `repr` (no trailing zeros)"*, and Python's `repr` is not guaranteed bit-stable across builds in all corner cases (subnormals, very-small numbers near format-crossover thresholds). Specify a precise encoder — `numpy.format_float_positional(precision=17)`, or a fixed-precision rational, or an explicit byte-level encoder — or the content-hash law has a corner-case failure mode.

Also still open: **the trained-artifact hash is not contracted.** `bound_variables` is recorded on the runtime artifact and the IR's `content_hash` covers structure + variable declarations / defaults, but there is no canonical `(ir_hash, bound_variables_hash, trained_param_hash, training_log_hash)` archive object. The pieces are present; the bundler is not.

## C7. Training-to-inference round-trip is missing

A user trains under `bptt`, calls `trainer.parameters()`, and wants to (a) run inference under `clock` with the trained weights, (b) export to NIR with trained weights baked in, or (c) archive a trained model alongside its spec. The IR-mutation surface is correctly closed (`net.variable` handles non-trained cross-run variation), but **the trained-weight-to-frozen-IR pipeline is undocumented**.

For the §1.1 comparative-study claim — *"compare event-prop vs BPTT on the same architecture"* — the user needs `bptt.build → train → archive-trained-IR → eventprop.build → train` to be canonical. Currently §10 does not show this. Recommend §10.x with the full pipeline, including how trained values are bound (either as new variable values or via a dedicated "freeze trainables" API).

## C8. Hardware-mapping story is thin

D25 scopes quantization / fan-in / placement per export-backend. The NIR exporter ships notice codes (`EXPORT-NIR-005` for densified large matrices, `EXPORT-NIR-007` for extension nodes, `EXPORT-NIR-010` for stochastic input parameters in sidecar) but no platform-specific transformations. This may be the right call — the alternative is for the spec to encode every neuromorphic-hardware quirk — but the manuscript should pick a position: either "graph-level NIR export; deployment is consumer-toolchain responsibility" should be stated explicitly in G11, or one platform (Loihi or SpiNNaker) should land end-to-end as a worked exemplar. The current "thin and undeclared" middle position weakens the contribution.

## C9. Datasets and optimizer / loss are still deferred

D21 keeps optimizer / loss / scheduler user-side; §3.17 explicitly defers datasets. For the §1.1 comparative-study claim, this still bites: the schedule grammar (§3.7.3) closes the protocol half of the original C7 reproducibility gap (warmup / ITI / stim / response / trial randomization), but the dataset / loss / optimizer half remains user-discipline rather than spec-enforced.

Recommendation: ship at least a thin dataset reference convention (defer to `tonic` / `nengo_loihi` data conventions for DVS streams; document the recommended interop pattern) and a canonical trained-artifact bundle helper.

## C10. `Trainable.constraint` vocabulary is too coarse for biophysics

`"positive" | "unit_norm" | "clip:lo,hi"` covers ML use, not biophysical priors (parameter coupling such as τ_m = R_m · C_m, monotonicity along ion-channel kinetics, ratio constraints between rest / threshold / reset). Widen to a constraint registry with documented options including biophysical priors, or hand off to a third-party constraint package and document the seam.

## C11. `Builder` co-existence has no deprecation timeline

§10.3 makes `Builder` explicitly the substrate of `brainpy.state.clock`; `NetSpec` is the recommended user-facing entry point; documentation and examples migrate. Both surfaces remain importable — `Builder` keeps working unchanged — but no formal deprecation date is given. Recommend stating an EOL for the "import `Builder` directly" path, with a migration cookbook from `Builder.add` / `Builder.connect*` patterns to `NetSpec.population` / `NetSpec.project`.
