# Comparison With Existing Frameworks

> Part of the editorial report on [`../network-spec-dsl/`](../network-spec-dsl/). See [README](./README.md) for navigation.

The manuscript should add a Related Work section organized along the axes below. Reviewer's read across the relevant landscape:

## Capability matrix (declarative-spec axis)

| Property                                | PyNN          | NeuroML / LEMS  | SONATA            | Nengo        | snnTorch / Norse | NIR (alone)     | **This proposal**                              |
|------------------------------------------|---------------|------------------|--------------------|--------------|--------------------|-------------------|------------------------------------------------|
| Declarative, framework-agnostic          | yes           | yes              | yes (data)         | partial      | no (Pythonic)      | yes (graph)       | **yes**                                        |
| Schema-validated archival                | partial (XML) | yes (XSD)        | yes (HDF5+JSON)    | no           | no                 | yes (JSON)        | **yes (JSON Schema)**                          |
| Physical units first-class               | partial       | yes              | partial            | yes (SI)     | no                 | no                | **yes (saiunit)**                              |
| Content-hashable IR                      | no            | no               | partial            | no           | no                 | no                | **yes**                                        |
| Trainability as IR concept               | no            | no               | no                 | partial      | yes (Python)       | no                | **yes (typed marker)**                         |
| Immutable IR + declared build-time vars  | no            | no               | no                 | no           | no                 | no                | **yes (`net.variable` / `VariableRef`)**       |
| Deep-SNN layer macros                    | no            | no               | no                 | partial      | yes                | partial           | **yes**                                        |
| DAG composability                        | no            | no               | no                 | partial      | yes                | yes               | **yes (`net.graph`, `fork`, merge layers)**    |
| Spatial / position-aware connectivity    | partial       | partial          | yes (positions)    | partial      | no                 | no                | **yes (`spec.geometry.*` + kernel × mask)**    |
| Biophysical / multi-compartment          | partial       | yes              | yes                | no           | no                 | no                | **yes (`spec.models.Cell`)**                   |
| Stochastic dynamics (in-equation noise)  | partial       | partial          | no                 | partial      | no                 | no                | **yes (`Noise` value wrapper, SDE auto-pick)** |
| Plasticity expressiveness                | shallow       | rules-only       | no                 | NEF / PES    | embedded           | inference-only    | **6-layer surface (per-proj → homeostatic)**   |
| Experiment-protocol abstractions         | no            | no               | partial            | no           | no                 | no                | **yes (`spec.schedule.*` Phase / Trial)**      |
| Multi-simulator (NEST / NEURON / Brian)  | **yes**       | yes              | yes                | no           | no                 | yes (consumers)   | partial (clock / event only)                   |
| Hardware export                          | partial       | partial          | no                 | yes (Loihi)  | yes (via NIR)      | **yes (canonical)** | yes (via NIR; one platform end-to-end TBD)   |

## Capability matrix (training-paradigm axis)

The §1.1 wedge — every existing framework commits to one training paradigm at model-definition time:

| Framework         | Modeling surface           | Training paradigm(s)                                | Deployment              |
|-------------------|----------------------------|------------------------------------------------------|-------------------------|
| snnTorch          | PyTorch modules            | BPTT (surrogate grad)                                | PyTorch                 |
| Norse             | PyTorch modules            | BPTT (surrogate grad + SuperSpike)                   | PyTorch                 |
| BindsNET          | PyTorch modules            | BPTT + Hebbian / STDP                                | PyTorch                 |
| Nengo             | NEF Network DSL            | NEF / PES                                            | Nengo, Loihi            |
| PyNN / Brian2     | Declarative DSL            | Plasticity rules only (no global gradient)           | NEST / NEURON / GPU     |
| Lava (Intel)      | Process graph              | On-chip plasticity                                   | Loihi 2                 |
| **Lava-DL**       | PyTorch modules            | SLAYER (surrogate-gradient family); Loihi-targetable | Loihi via Lava          |
| **hxtorch.snn**   | PyTorch modules            | Surrogate-gradient + event-prop on a single substrate | BrainScaleS-2 / GPU    |
| **EXODUS**        | PyTorch + custom training  | SLAYER variants compared on the same architecture    | GPU                     |
| **brainpy.state** | **Declarative IR**         | **BPTT + event-prop + RTRL + eligibility-trace**     | **clock/event + NIR**   |

Lava-DL, hxtorch.snn, and EXODUS are the closest comparators on the multi-paradigm axis and **must** be engaged in §1.1.1 of the manuscript — currently they are not. Each is multi-paradigm in practice on a single substrate; the proposal's differentiator is the declarative IR (paradigm-neutral spec module, backends as peer top-level modules under `brainpy.state`) rather than the multi-paradigm claim alone.

## Audience and differentiator

| Framework               | Primary audience                                    | Strongest overlap with this proposal                       | Key differentiator the proposal must claim                                                              |
|-------------------------|-----------------------------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **PyNN**                | Computational neuroscience, multi-simulator interop | Populations, projections, recording, sub-networks, schema  | Physical-unit-first IR; content-hash determinism; trainability marker; deep-SNN layer macros            |
| **NeuroML / LEMS**      | Biophysical modeling, archival                      | Schema-validated declarative form, units, multi-tool consumption | JAX-native runtime substrate; trainability + ML-style backends; declared build-time variables       |
| **SONATA**              | Large-scale circuit data (Allen / BBP)              | Population/edge tables, file-format archival                | In-memory IR + Pythonic builder; export-backend protocol; six-layer plasticity                          |
| **Nengo**               | Neural-engineering framework, Loihi deployment      | Declarative `Network`, hardware backends                    | Multi-paradigm support (biophysical + deep-SNN); frozen content-hashable IR; immutable IR + variables   |
| **snnTorch / Norse**    | Deep-SNN / brain-inspired ML                        | Layer macros, trainable weights, NIR export                 | Spec layer above the model code; YAML archival; backend-neutral training-paradigm pluralism             |
| **BindsNET / Rockpool** | Brain-inspired ML, neuromorphic                     | Layer macros, plasticity, NIR export                        | Declarative IR rather than imperative Module; cross-paradigm coverage; six-layer plasticity             |
| **Lava-DL / hxtorch.snn** | Multi-paradigm SNN training                       | Multi-paradigm on single substrate                          | Declarative IR (paradigm-neutral spec); paradigm-swap is one-line backend change; spec is JAX-native    |
| **NIR**                 | Neuromorphic IR for deployment                      | Graph IR, hardware consumer ecosystem                       | The proposal *produces* NIR; differentiator is the rich upstream spec, the sidecar, and the six-class lossiness taxonomy |
| **NMODL / NESTML**      | Biophysical model description, compilation          | Declarative model description with units                    | Network-level (not single-cell) scope; ML-friendly trainability                                          |

## Bottom line

The strongest differentiating dimensions are: **training-paradigm pluralism over a single IR** (the §1.1 thesis), the **six-layer plasticity surface**, **immutable IR + declared build-time variables**, **physical units in the IR**, **content-hash determinism**, the **schedule grammar** + **signals modulator graph**, and the **NIR export taxonomy**. The weakest dimensions are: **multi-simulator interop** (NEST / NEURON / Brian are not reachable backends — explicit non-goal candidate), **end-to-end hardware story** (G11 needs honest-scoping or one platform landed), and **§1.1 prior-art engagement on the comp-neuro and multi-paradigm axes** (NeuroML / LEMS, SONATA, NMODL / NESTML, Lava-DL, hxtorch.snn, EXODUS are not engaged).
