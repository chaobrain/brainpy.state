# Concrete Improvement Suggestions

> Part of the editorial report on [`../network-spec-dsl.md`](../network-spec-dsl.md). See [README](./README.md) for navigation. The revision-1 rewrite addresses item 1 below in part (PyNN / Brian2 / Nengo engagement) but not in full; for the *minimum revision to reach NC standard* under the new training-paradigm framing, see [`01-revision-1-review.md`](./01-revision-1-review.md) "Specific, minimum revision to reach NC standard."

## Essential (block research-contribution status)

1. **§1 must engage prior art.** Add a 1–2 page Related Work section contrasting with PyNN, NeuroML/LEMS, SONATA, Nengo, snnTorch, Norse, BindsNET, Lava-DL, Rockpool, and NIR. Identify (and label) the genuinely novel contributions explicitly.
2. **Soften G1 or strengthen the IR.** Either move event-scheduling, plasticity update laws, recording cadence, and stochastic dynamics into the IR, or rewrite G1/G4 to acknowledge that semantics are jointly determined by `(IR, backend)`. The current text overclaims.
3. **Document the train → infer → export round-trip.** Add §10.x showing how `Trainer.parameters.diff()` becomes a `ParamPatch` list → `spec.patch(...)` → `nir.export` with trained values. This is the single most important user workflow and is currently implicit.
4. **Specify the RNG contract end-to-end.** `braintools.conn` rules must accept a JAX key; ban host RNG; document float canonicalization; specify the trained-artifact hash.
5. **Add experiment / protocol primitives.** At minimum: `Protocol(warmup, epochs, reset_policy)` and `Trial` abstractions. Without these, reproducibility for deep-SNN benchmarks is user-discipline.
6. **Position the hardware story honestly.** Either deliver a Loihi-or-SpiNNaker end-to-end mapping (quantization vocabulary, time discretization, routing constraints) or scope G11 to "graph-level NIR export; deployment is consumer-toolchain responsibility."
7. **Specify versioning and schema evolution.** Document compatibility rules between `netir/1.0` → `1.x` → `2.0`, the migration tool surface, and version-skew handling at load time.

## High-priority (improve rigor and adoption)

8. **Multi-compartment / morphology:** state explicitly as a non-goal in §2.2 (it currently is not listed), or sketch a future extension path.
9. **Plasticity vocabulary:** at minimum, formalize STDP / STP / R-STDP / homeostatic plasticity as registered protocols with documented pre/post coupling and third-factor inputs.
10. **DAG composability:** add `parallel(...)`, skip-connection sugar, and predicate-driven views (`spec.where`).
11. **Constraint registry:** widen `Trainable.constraint` to a registry with documented options, including biophysical priors.
12. **Recording reducers as a registry:** support quantiles, rolling statistics, custom callables.
13. **Spatial positions as a canonical IR field** on `PopulationNode` rather than `init`.
14. **Define `MergedView` projection semantics rigorously** so wildcards in `ParamPatch` paths and trainable name resolution are unambiguous after denormalization.
15. **Concrete deprecation plan for `Builder`** or a hard "internal substrate" boundary.

## Optional (adoption polish)

16. **Add a profiling / cost model:** memory + flops estimate from the IR alone.
17. **Sweep strategies beyond cartesian:** Sobol, random, Bayesian, with resume / early-stop hooks.
18. **Streaming recording back-end:** disk-backed `TraceBundle` for long simulations.
19. **Patch portability across renames:** ship the `bp-spec patch migrate` helper proposed as an open question.
20. **NIR extension namespace coordination:** make the upstream NIR coordination a release blocker, not an open question.
21. **Trace-level provenance:** include the `(ir_hash, trainer_log_hash, param_diff_hash)` bundle as a canonical archive object.
