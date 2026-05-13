# Major Strengths

> Part of the editorial report on [`../network-spec-dsl.md`](../network-spec-dsl.md). See [README](./README.md) for navigation.

1. **Frozen IR with content hash.** SHA-256 over canonical JSON, with explicit canonicalization of `u.Quantity`, `Trainable`, `DistRef`, `ConnRule`, `ModelRef`. This is a real reproducibility primitive — most SNN frameworks lack it.
2. **Two equivalent frontends.** The `NetSpec ↔ YAML ↔ NetIR` round-trip law is testable and gives both library users and archival workflows what they need.
3. **Physical units are first-class.** Carrying `saiunit` through the IR and validating dimensional consistency is the right call; the contrast with PyNN's mostly-numeric API is real.
4. **Unified path language for static and dynamic parameters.** `populations.exc.model.tau` addressing both pre-build edits and runtime `ParameterView` updates is a clean abstraction, and the three-class `LIVE / LIVE_RESET / REBUILD` taxonomy is a genuine engineering insight.
5. **Lossy-export taxonomy.** The six-class scheme (`LOSSLESS / RECORDED / APPROXIMATE / EXTENSION / DROPPED / UNSUPPORTED`) with stable `EXPORT-NIR-NNN` codes and strict/lenient modes is more rigorous than any existing NIR exporter the reviewer is aware of.
6. **Stable error code catalog (§14).** SPEC-NNN codes with construction/finalize/backend/mutation tiers make the spec actually documentable.
7. **Backend capability protocol.** `BackendCapabilities` with sets of supported kinds + a single validation pass is the right shape.
8. **Decision log (§18).** Twenty-seven captured decisions with rationale is exactly what an RFC should ship.
