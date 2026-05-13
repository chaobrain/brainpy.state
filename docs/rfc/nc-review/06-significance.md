# Significance

> Part of the editorial report on [`../network-spec-dsl/`](../network-spec-dsl/). See [README](./README.md) for navigation.

The proposal addresses a real need: the JAX-native SNN tooling stack (`brainstate` / `brainpy.state` / `braintools` / `brainevent`) currently lacks an upstream declarative layer comparable to PyNN-for-NEST. For the `brainpy.state` user base this is high-value engineering and would be the right next step.

For the *broader* computational-neuroscience and neuromorphic communities, significance is limited unless:

- (a) the spec is positioned as a *bridge* to existing standards (NeuroML, SONATA, PyNN) rather than a replacement;
- (b) the hardware-mapping story includes at least one concrete platform (Loihi or SpiNNaker) end-to-end, not just NIR-as-handoff (or alternatively, scope G11 explicitly to "graph-level NIR export; deployment is consumer-toolchain responsibility");
- (c) the deep-SNN training story includes optimizer / dataset / protocol abstractions enabling reproducible benchmark reporting.

Item (c) is load-bearing for the §1.1 training-paradigm-pluralism claim: the canonical user story is "compare event-prop vs BPTT on the same architecture." For this to be reproducible — and reproducibility is what NC requires of methods contributions — the spec needs canonical **Protocol** (warm-up, epochs, reset semantics; addressed by §3.7.3 schedules), **Dataset** (canonical references, splits, preprocessing; still deferred), and **Optimizer / Loss / Schedule** abstractions (still deferred). The protocol half is closed; the dataset / optimizer half is not.
