# Improvement Suggestions

> Part of the editorial report on [`../network-spec-dsl/`](../network-spec-dsl/). See [README](./README.md) for navigation. This file lists only currently open suggestions. Suggestions that have been incorporated into the spec are removed.

## Essential (block research-contribution status)

1. **Add `G13` to §1.2** — *Training-paradigm pluralism. The IR is neutral across BPTT, event-prop, RTRL / forward-mode, and eligibility-trace training. Switching paradigms is a backend-build kwarg; no IR rewrite is required. Backends declare per-paradigm constraints and surface incompatibilities at build time.* — and trace it through Chapter 2 (IR additions for paradigm-requirement markers: `requires_spike_time_differentiability`, `requires_third_factor_signal`, `supports_recurrent_state_size`), Chapter 6 (per-paradigm `BackendCapabilities` instantiations), and Chapter 10 (SPEC-NNN code for "IR feature × training-backend mismatch" beyond the existing SPEC-013 / SPEC-015 / SPEC-021).

2. **Write the per-shipped-training-backend capability matrix in §6.1.** One row per shipped backend (`clock`, `event`, `bptt`, `eprop`, `eventprop`, `ppprop`), columns covering `supports_plasticity`, `supports_noise`, `supports_morphology`, `supports_structural_plasticity`, threshold semantics, topology constraint, plus any backend-specific notes. The schema is already in place (`BackendCapabilities`); the missing piece is the concrete instantiation. See the sketch in [`04-concerns.md`](./04-concerns.md) C3.

3. **Reproduce one published comparative-study benchmark** by switching the backend on a single IR. Event-prop vs BPTT on YinYang or N-MNIST (Wunderlich & Pehle 2021) is the canonical target; matching the published accuracy within reported error bars is the acceptance gate. Ship the notebook under `docs/examples/` and add a CI smoke test. This is the load-bearing experiment for the §1.1 claim; without it, the claim is rhetorical.

4. **Engage the missing prior art in §1.1.** One paragraph each for **Lava-DL**, **hxtorch.snn**, **EXODUS** (closest multi-paradigm prior art). One paragraph for **NeuroML / LEMS**, **SONATA**, **NMODL / NESTML**, **NIR-as-upstream-IR** (the declarative-SNN axis). The §1.1.1 prior-art table currently covers only the framework axis at the BPTT / NEF level.

5. **Add a §5.x training-round-trip walkthrough.** Show `bptt.build → train → archive trained IR → eventprop.build → train` end-to-end, including how trained weights are bound (either as new variable values via `backend.build(ir, variables={...})` or via a dedicated "freeze trainables" API). This is the load-bearing user workflow for the §1.1 comparative-study claim and is currently undocumented.

6. **Position the hardware story honestly in G11.** Either deliver a Loihi-or-SpiNNaker end-to-end mapping (quantization vocabulary, time discretization, routing constraints) as a worked exemplar, or scope G11 to "graph-level NIR export; deployment is consumer-toolchain responsibility." The current undeclared middle position weakens the contribution.

7. **Specify versioning and schema-evolution policy.** Document compatibility rules between `netir/1.0` → `1.x` → `2.0`, the migration-tool surface, and version-skew warnings at load time.

## High-priority (improve rigor and adoption)

8. **Ship at least a thin dataset reference convention.** Defer to `tonic` / `nengo_loihi` data conventions for DVS streams; document the recommended interop pattern. The schedule grammar closes the protocol half of the reproducibility story; this closes the dataset half.

9. **Widen `Trainable.constraint` to a constraint registry** with documented options, including biophysical priors (parameter coupling, ratios, monotonicity along ion-channel kinetics).

10. **Specify a precise float canonicalization encoder.** Replace `repr` in §6.2 with `numpy.format_float_positional(precision=17)`, a fixed-precision rational, or an explicit byte-level encoder. Without this the content-hash law has a corner-case failure mode.

11. **Document `braintools.conn` rule-side RNG invariant.** §10.1 #5 bans backend-side random consumption outside the seed tree; the corresponding rule-side requirement (rules must accept a JAX key; no host `np.random`) should be promoted from implicit to explicit in §8.1.

12. **State sub-network trainable-sharing semantics in §3.11.1.** When the same `column_spec` is instantiated multiple times, are `Trainable` weights shared or independent? (Probably independent.) If parameter-tying is desirable, sketch a future `ShareWith(other)` value wrapper.

13. **State a concrete deprecation plan for the "import `Builder` directly" path.** §5.3 makes `Builder` the substrate of `brainpy.state.clock`; `NetSpec` is the recommended user-facing entry point. Set an EOL for direct `Builder` import, with a migration cookbook from `Builder.add` / `Builder.connect*` patterns to `NetSpec.population` / `NetSpec.project`.

## Optional (adoption polish)

14. **Trained-artifact provenance bundle.** Canonical `(ir_hash, bound_variables_hash, trained_param_hash, training_log_hash)` archive object. The pieces are present; the bundler is not.

15. **Profiling / cost-model CLI.** `brainpy estimate` reporting memory + compute estimates from the IR alone (population × density × dt × duration). Already scoped in §3.17.

16. **Sweep strategies beyond cartesian.** Sobol, random, Bayesian, with resume / early-stop hooks, in `brainpy sweep`.

17. **Streaming / disk-backed `TraceBundle`.** For long simulations beyond memory.

18. **NIR extension namespace coordination.** Make the upstream NIR coordination (currently §11.3 open question) a release blocker so the `nir.brainx.*` namespace is reserved and documented in the NIR extension registry.

19. **Observable cadence under event-driven backends.** Either lift `every:` to a backend-uniform "nearest event after T elapses" rule, or document the per-backend interpretation in `BackendCapabilities` (§6.1.3).
