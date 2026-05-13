# Comparison With Existing Frameworks

> Part of the editorial report on [`../network-spec-dsl.md`](../network-spec-dsl.md). See [README](./README.md) for navigation. The revision-1 rewrite added a §1.1 prior-art table at the framework / training-paradigm axis; this file adds the deeper *capability-matrix* axis for the original DSL claim. Both are needed for an NC submission ([`01-revision-1-review.md`](./01-revision-1-review.md), N4 / N5).

The manuscript should add a Related Work section organized along the axes below. Reviewer's quick read across the relevant landscape:

## Capability matrix

| Property | PyNN | NeuroML / LEMS | SONATA | Nengo | snnTorch / Norse | NIR (alone) | **This proposal** |
|---|---|---|---|---|---|---|---|
| Declarative, framework-agnostic | yes | yes | yes (data) | partial | no (Pythonic) | yes (graph) | **yes** |
| Schema-validated archival | partial (XML v1.0) | yes (XSD / Schematron) | yes (HDF5+JSON) | no | no | yes (JSON) | **yes (JSON Schema)** |
| Physical units first-class | partial | yes | partial | yes (SI) | no | no | **yes (saiunit)** |
| Content-hashable IR | no | no | partial | no | no | no | **yes** |
| Trainability as IR concept | no | no | no | partial | yes (Python) | no | **yes (typed marker)** |
| Path-addressed runtime mutation | no | no | no | no | no | no | **yes (`ParamPatch`)** |
| Deep-SNN layer macros | no | no | no | partial | yes | partial | **yes** |
| Biophysical / multi-compartment | partial | yes | yes | no | no | no | no |
| Multi-simulator (NEST / NEURON / Brian) | **yes** | yes | yes | no | no | yes (via consumers) | partial (clock / event only) |
| Hardware export | partial | partial | no | yes (Loihi) | yes (via NIR) | **yes (canonical)** | yes (via NIR) |
| Protocol / experiment abstractions | no | no | partial | no | no | no | no |

## Audience and overlap

| Framework | Primary audience | Strongest overlap with this proposal | Key differentiator the proposal must claim |
|---|---|---|---|
| **PyNN** | Computational neuroscience, multi-simulator interop | Populations, projections, recording, sub-networks, schema | Physical-unit-first IR; content-hash determinism; trainability marker; deep-SNN layer macros |
| **NeuroML / LEMS** | Biophysical modeling, archival | Schema-validated declarative form, units, multi-tool consumption | JAX-native runtime substrate; trainability + ML-style backends; ParamPatch language |
| **SONATA** | Large-scale circuit data (Allen / BBP) | Population/edge tables, file-format archival | In-memory IR + Pythonic builder; pre/post-build mutation; export-backend protocol |
| **Nengo** | Neural-engineering framework, Loihi deployment | Declarative `Network`, hardware backends | Multi-paradigm support (biophysical + deep-SNN); frozen content-hashable IR |
| **snnTorch / Norse** | Deep-SNN / brain-inspired ML | Layer macros, trainable weights, NIR export | Spec layer above the model code; YAML archival; ParamPatch + LIVE/LIVE_RESET/REBUILD |
| **BindsNET / Lava-DL / Rockpool** | Brain-inspired ML, neuromorphic | Layer macros, plasticity, NIR export (Rockpool/Sinabs) | Declarative IR rather than imperative Module; cross-paradigm coverage |
| **NIR** | Neuromorphic IR for deployment | Graph IR, hardware consumer ecosystem | The proposal *produces* NIR; the differentiator is the rich upstream spec, the unit / trainability / seed sidecar, and the six-class lossiness taxonomy |
| **NMODL / NESTML** | Biophysical model description, compilation | Declarative model description with units | Network-level (not single-cell) scope; ML-friendly trainability |

## Bottom line on differentiation

The proposal's strongest dimensions are **physical units in the IR**, **content-hash IR**, **trainability marker propagating to `brainstate.nn.Param`**, and the **ParamPatch / ParameterView path language**. Its weakest are **multi-simulator interop** (NEST / NEURON / Brian are not reachable backends), **biophysical / multi-compartment coverage**, and **experiment-protocol abstractions**.

After revision 1, the central differentiation claim shifts: the load-bearing dimension becomes **training-paradigm pluralism** (BPTT / event-prop / RTRL / e-prop). The dimensions above remain real engineering wins but are no longer the central thesis. See [`01-revision-1-review.md`](./01-revision-1-review.md) for the updated framing and the additional comparative-study prior art (EXODUS, hxtorch.snn, Lava-DL multi-paradigm coverage) that must be engaged.
