# Open Conceptual and Semantic Gaps

> Part of the editorial report on [`../network-spec-dsl/`](../network-spec-dsl/). See [README](./README.md) for navigation. This file lists only currently open gaps. Resolved gaps are removed.

1. **Time-step semantics on cross-backend translation.** `dt` is a backend kwarg (D1), yet event-driven, fixed-step, and adaptive-step backends interpret `dt` incompatibly. The IR should at minimum carry a `time_resolution_hint` (advisory; non-binding for backends that ignore it) for archival reproducibility across backend swaps.

2. **Random-state hand-off across the host/device boundary.** §9.1 #5 explicitly bans backend-side random consumption outside the seed tree, and the spec requires `braintools.conn` rules to accept a JAX key. The contract should be promoted from "spec requirement" to a documented `braintools.conn.Connectivity` rule-side invariant — currently the rule-side enforcement is implicit.

3. **Sub-network parameterization with shared trainables.** When the same `column_spec` is instantiated four times with different `N`, are their `Trainable` weights shared or independent? The spec does not say (probably independent, but parameter tying is a normal request for weight-sharing CNNs and recurrent cores). Recommend an explicit statement in §3.11.1, with a possible future extension via a `ShareWith(other)` value wrapper.

4. **Versioning policy.** `netir/1.0` is fixed; the forward-compatibility invariant in §3.16 ("every new field defaults to 'absent' and round-trips unchanged") is the right shape but does not constitute a versioning policy. State what triggers a minor bump (additive optional fields, new value-wrapper kinds) vs a major bump (breaking field changes, removed fields, semantic redefinition), and the deprecation policy for retired field names.

5. **Float canonicalization across platforms.** §5.2 says floats are formatted with `repr` (no trailing zeros). Python's `repr` is not bit-stable across builds for all corner cases (subnormals, format-crossover thresholds). Specify a precise encoder — `numpy.format_float_positional(precision=17)`, a fixed-precision rational encoding, or an explicit byte-level encoder. Without this the content-hash law has a corner-case failure mode.

6. **Trained-artifact provenance hash.** `bound_variables` is recorded on the runtime artifact and the IR's `content_hash` covers structure + variable declarations / defaults, but there is no canonical `(ir_hash, bound_variables_hash, trained_param_hash, training_log_hash)` archive object. The pieces are present; the bundler is not.

7. **Observable cadence under event-driven backends.** `Observable.every: u.ms` is dt-relative under `clock`; what does it mean under `event`? Either (a) lift to a backend-uniform "nearest event after T elapses" rule, or (b) document the per-backend interpretation explicitly in §5.1.3 capability declarations.
