# Conceptual and Semantic Gaps

> Part of the editorial report on [`../network-spec-dsl.md`](../network-spec-dsl.md). See [README](./README.md) for navigation.

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
