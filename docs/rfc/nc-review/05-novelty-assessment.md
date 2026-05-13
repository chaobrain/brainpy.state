# Novelty Assessment

> Part of the editorial report on [`../network-spec-dsl.md`](../network-spec-dsl.md). See [README](./README.md) for navigation. This file scores the original document's per-claim novelty. The revision-1 rewrite repositions the load-bearing claim onto a different axis (training-paradigm pluralism) — see [`01-revision-1-review.md`](./01-revision-1-review.md) for the updated central-claim analysis.

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

**Net (original-report perspective):** Two-to-three genuinely novel ideas (`ParamPatch` + `ParameterView` typing, content-hash IR with patch round-trip, lossy-taxonomy export). The rest is competent engineering on well-trodden ground. The novel ideas must be foregrounded.

**Net after revision 1:** the rewrite chooses *not* to foreground the engineering ideas above, and instead pivots to a different load-bearing claim — *training-paradigm pluralism over a single IR* — that is more publishable and is correctly identified as the empty quadrant on the prior-art map. The engineering ideas in the table above remain real but are now supporting evidence rather than the central thesis. See [`01-revision-1-review.md`](./01-revision-1-review.md).
