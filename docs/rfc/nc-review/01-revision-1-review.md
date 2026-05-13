# Revision 1 — re-review of the §1.1 / D28 rewrite

> See [README](./README.md) for navigation. This file is the current re-review; the original report sections that follow ([02–11](./README.md#reading-order)) are retained as reference. The updated final verdict lives in [`11-final-recommendation.md`](./11-final-recommendation.md).

## What was changed

The revision is concentrated and surgical: a new **§1.1 "Novelty and prior art"** (~70 lines, three subsections) and a new **D28** in the decision log. The body of the document — goals (G1–G12), IR dataclasses (§5), backend protocols (§8), determinism contract (§13), validation catalog (§14), testing strategy (§16) — is unchanged.

The repositioning is from *"frozen IR + DSL surface"* (the prior framing) to:

> **The load-bearing novelty is that the same network description drives four mathematically distinct SNN training paradigms from a single IR — BPTT (with surrogate gradients), event-prop (Wunderlich & Pehle 2021), RTRL / forward-mode autodiff, and eligibility-trace methods (e-prop; Bellec et al. 2020).**

NIR export is now explicitly *demoted* as "fourth axis of pluralism (deployment), not the load-bearing novelty." A prior-art table contrasts snnTorch / Norse / BindsNET / Nengo / PyNN&Brian2 / Lava on the training-paradigm axis. D28 codifies the positioning as a tie-breaker for future scope decisions ("does this preserve neutrality across the four paradigms?").

## What this fixes

1. **Novelty is now a real, falsifiable, publishable claim.** "Single IR drives four mathematically distinct training paradigms" is concrete: it can be tested, refuted, and benchmarked. The original framing ("frozen IR + DSL") was not — every framework on the prior-art list has some flavor of that.
2. **The wedge is correctly identified.** Every existing SNN framework (snnTorch, Norse, Lava, Nengo, BindsNET, PyNN/Brian2) commits to *one* training paradigm at model-definition time. The training-paradigm-pluralism axis is the empty quadrant on the prior-art map. This is the right insight.
3. **The "JAX brought to autodiff / ONNX brought to inference" framing** is good positioning rhetoric — it tells a reviewer, in one sentence, what altitude this contribution claims.
4. **D28 as an architectural tie-breaker** is a strong bit of design discipline. It anchors future scope decisions to the central claim.

## What still blocks NC acceptance

The novelty *positioning* is now sufficient. The novelty *evidence* is not. The body of the spec must follow through on the §1.1 claim, and currently it does not. Specific, load-bearing gaps:

### N1. The four-paradigm claim is asserted, not designed for

§8.2 names `bptt`, `eprop`, `event-prop` as shipped backends in a single table cell — but the spec gives **no design** for them, and the IR contains **no construct** that distinguishes their requirements:

- **Event-prop** demands exact gradients of spike times — i.e. differentiable thresholds and event-time implicit differentiation (Wunderlich & Pehle 2021; §3 of that paper). This is *not* the same surrogate-gradient signal BPTT uses; it requires the IR to carry threshold-crossing information. The current IR has no such construct.
- **RTRL / forward-mode** requires forward-mode JVPs through recurrent dynamics; a naive RTRL through arbitrary-topology recurrent surrogate-gradient computation has O(N⁴) memory and is intractable for the deep-SNN audience the spec also targets. Real RTRL backends (UORO, SnAp-k, online OE) impose IR constraints that the spec does not articulate.
- **Eligibility-trace / e-prop** needs cross-projection eligibility hooks — exactly the C8 plasticity-expressiveness gap from the original report ([`04-concerns.md`](./04-concerns.md)), which the rewrite did not address. e-prop in Bellec et al. (2020) requires per-projection learning-signal pathways that the current `Projection` node does not expose.

A reviewer for NC will ask, on §1.1 page 1: *"show me, in the IR, what makes this claim mechanically possible."* The IR must surface backend-specific feature requirements (e.g., `requires_spike_time_differentiability`, `requires_third_factor_signal`); §5 currently does not.

### N2. Per-training-backend capability matrix is missing

`BackendCapabilities` (§8.2) is a *uniform* schema across sim/train/export. But each training paradigm imposes radically different constraints on the IR:

| Paradigm | Common constraints (illustrative — must be made precise in §8.2) |
|---|---|
| BPTT (surrogate) | Any neuron model with a defined surrogate; arbitrary topology; bounded sequence length for memory |
| Event-prop | LIF / ALIF only (analytic threshold crossing); no plasticity during training; specific reset semantics |
| RTRL family | Bounded state size; specific recurrent topology constraints; forward-mode-friendly synapse models |
| e-prop | Local learning signals only; specific recurrent network topology; no global gradients |

Without this matrix written down per backend, "switch backend by changing one kwarg" collapses on first contact: the user who tries to swap `bptt` for `event-prop` on a Hodgkin–Huxley network will hit `BackendCapabilityError` and conclude the claim is hollow. The matrix is also load-bearing for the comparative-study story — the user needs to know *a priori* which IRs admit which paradigms.

### N3. Comparative-study reproducibility requires C7 (now load-bearing)

The new framing's user story is *"compare event-prop vs BPTT on the same architecture."* For this to be reproducible — and reproducibility is what NC requires of methods contributions — the spec needs canonical **Protocol** (warm-up, epochs, reset semantics), **Dataset** (canonical references, splits, preprocessing), and **Optimizer / Loss / Schedule** abstractions. D21 explicitly defers loss/optimizer to the user. Combined with the absence of dataset and protocol abstractions, the spec does not deliver reproducibility for the very comparative study that §1.1 motivates. **C7 from the original report ([`04-concerns.md`](./04-concerns.md)) is now the central blocker, not a minor gap.**

### N4. Prior-art table omits the most relevant comparative-study work

The §1.1.1 table is honest about the *frameworks* but misses the *literature* that already targets the training-paradigm-comparison wedge:

- **EXODUS** (Bauer, Lenz, Liu, Sheik 2023, *Frontiers in Neuroscience*) — explicitly compares SLAYER variants on the same architecture, very close to the "swap one kwarg" claim.
- **hxtorch.snn** (Pehle et al. 2022) — implements both surrogate-gradient and event-prop on a single substrate.
- **Norse** training-paradigm coverage extends to surrogate-gradient *and* SuperSpike, ADAM-based and Adjoint-based — beyond BPTT.
- **Lava-DL** (Intel) supports SLAYER on Loihi-targetable models — multi-paradigm in practice.
- **BrainCog** and **PySNN** also belong on this map.

For an NC submission, the table must engage these explicitly — they are the closest competing claims to the training-paradigm-pluralism wedge.

### N5. Original C1 (prior art for the *DSL* axis) is only partially addressed

PyNN, Brian2, Nengo are now engaged at the framework-axis level. **NeuroML / LEMS, SONATA, NMODL, NESTML, and NIR itself** remain unengaged in the rewrite. NC has computational-neuroscience reviewers who will treat this as a fatal omission for any "declarative SNN spec" claim, even one whose primary novelty is on the training-paradigm axis. The Related Work section must engage them — particularly NeuroML/LEMS for the units + schema lineage, SONATA for the population/edge data-table form, and NIR for the canonical neuromorphic IR.

### N6. The novelty claim is asymmetric — §1.1 vs the rest of the document

The new framing is not yet reflected in:

- **Goals (§2).** No G-line mentions training-paradigm pluralism; G7 mentions deep SNNs but not the training-strategy axis.
- **The IR (§5).** No node, no marker, no metadata block surfaces what each training paradigm requires. The IR cannot distinguish "this network is trainable under event-prop" from "this network is trainable under BPTT only."
- **Validation catalog (§14).** No SPEC-NNN code for "IR feature incompatible with chosen training backend"; SPEC-013 / SPEC-022 only handle plasticity-kind and layer-macro mismatches.
- **Testing strategy (§16).** No test category exercises the four-paradigm claim. The acceptance test should include: build IR; train under {BPTT, event-prop, e-prop} where each is supported; verify gradient signals are algorithmically distinct; verify the spec hash is bit-identical across the three runs.

The novelty claim feels grafted onto §1; it must be load-bearing through §2, §5, §8, §14, §16. As written, a reviewer who reads §1.1 then reads §5 will conclude "the IR was designed for BPTT and tagged for the others post hoc." That conclusion is currently defensible from the document.

## Specific, minimum revision to reach NC standard

The below is the *minimum* set of changes the document needs. Each maps to one or more concerns above.

1. **Add a new G-line** — *G13: Training-paradigm pluralism. The IR is neutral across BPTT, event-prop, RTRL/forward-mode, and eligibility-trace training. Switching paradigms is a backend-build kwarg; no IR rewrite is required. Backends declare per-paradigm constraints and surface incompatibilities at build time.* — and trace it through the rest of the document. (N1, N6)
2. **Per-training-backend capability matrix.** Write out, in §8.2, for each shipped training backend, what IR features it requires / forbids / approximates. (N2, N6)
3. **Add a §10.x training round-trip.** Walk through `bptt.build(ir) → train → trainer.parameters.diff() → ir.patch(...) → event-prop.build(...) → train` showing the comparison-study workflow end-to-end. (N3, N1)
4. **Demonstrate the claim on a published benchmark.** Reproduce one published result where event-prop and BPTT have been compared (e.g., Wunderlich & Pehle 2021 on YinYang or N-MNIST) by switching the backend on a single IR and matching published accuracy within reported error bars. This is the load-bearing experiment for NC; without it the claim is rhetoric. (N1, N3)
5. **Expand §1.1 prior-art engagement** to include EXODUS, hxtorch.snn, Lava-DL, BrainCog, and the comp-neuro axis (NeuroML/LEMS, SONATA, NMODL/NESTML, NIR). (N4, N5)
6. **Add SPEC-NNN error codes** for IR-feature × training-backend mismatches. (N6)
7. **Address C7 (Protocol / Dataset / Optimizer abstractions)** as load-bearing for the comparative-study story — at minimum a `Protocol(warmup, trial_structure, reset_policy)` and a thin `Dataset` reference. (N3, original C7)

The remaining concerns ([`04-concerns.md`](./04-concerns.md): C2 event semantics, C3 RNG contract, C4 training round-trip, C5 hardware honest-scoping, C6 biophysical scope, C8 plasticity, C9 DAG composability, C10 constraint vocabulary, C11 Builder/NetSpec coexistence) remain valid as *secondary* revision items but are not the blocker for NC suitability. The training-paradigm-pluralism claim's evidence — items 1–5 above — is the blocker.

## Updated final verdict for the §1.1 / D28 rewrite

- **Direction of the rewrite:** correct, sharp, and well-judged. The training-paradigm-pluralism axis is the right wedge; the prior-art table correctly identifies the empty quadrant; the JAX/ONNX framing is good positioning; D28 disciplines future scope decisions.
- **Sufficiency for *Nature Communications*:** **No.** The novelty is now correctly *positioned* but not yet correctly *demonstrated*. The IR (§5), goals (§2), validation (§14), and testing (§16) do not yet reflect the central claim, and the spec ships no worked comparative example.
- **Sufficiency for a methods venue with a lower evidence bar** (*Neuroinformatics*, *Frontiers in Neuroinformatics*, *JOSS*, *SoftwareX*): **Yes, with the original C2 / C3 / C4 / C7 fixes.** These venues accept the positioning + design + implementation without requiring a benchmarked comparative study.
- **Recommended path to NC:** complete revision items 1–7 above, ship a single working comparative-study notebook (event-prop vs BPTT on YinYang or similar), and resubmit. With those, the claim transitions from "asserted" to "demonstrated," and the contribution becomes one of the more interesting SNN-tooling submissions of the year.
