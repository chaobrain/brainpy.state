# Network Specification DSL for brainpy.state — Requirements

**Status:** Requirements specification (single integrated design)
**Owner:** TBD
**Date:** 2026-05-13
**Scope:** New `brainpy_state.spec` module; existing `brainpy_state._network` becomes the substrate of the `clock` simulation backend.

---

## What this RFC proposes

A new `brainpy_state.spec` module: a declarative DSL that describes spiking
neural networks once and lowers to multiple simulation, training, and
export backends without the model being rewritten. Two frontends (fluent
Python builder and YAML/JSON data DSL) produce one canonical, frozen,
content-hashable IR (`NetIR`). Backends consume the IR.

## The novelty (TL;DR)

The novelty is **not** the specification surface — PyNN, NESTML, Brian2,
and Nengo have shown for over a decade that an SNN can be described
declaratively. The novelty is that **the same network description drives
four mathematically distinct SNN training paradigms from a single IR**:

- **BPTT** with surrogate gradients,
- **Event-prop** (Wunderlich & Pehle 2021) — exact gradients of spike times,
- **RTRL / forward-mode autodiff** — online gradient estimation, and
- **Eligibility-trace methods** (e-prop, Bellec 2020).

Every existing SNN framework commits to exactly one paradigm at
model-definition time. Here, the choice of gradient flow is a backend
kwarg — researchers A/B paradigms by changing one line while the spec,
seed, and connectivity rules stay bit-identical. NIR export (deployment
to Loihi, SpiNNaker, Nengo) is a fourth axis of pluralism but is not
the load-bearing novelty — NIR is a community standard we adopt.

See [§1.1.1 in Chapter 1](./01-overview.md#111-novelty-and-prior-art) for the
prior-art comparison table.

---

## Table of contents

Section numbering follows file numbering: chapter N's content is §N (with sub-sections §N.x, §N.x.y, …).

| Chapter                                                               | What's in it                                                                                                  |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [1 — Overview](./01-overview.md)                                      | Problem statement, novelty pitch and prior-art table, goals (G1–G11), user populations, non-goals, primitive node kinds, architecture diagram. |
| [2 — The IR (`NetIR`)](./02-ir.md)                                    | Canonical frozen-dataclass IR: value wrappers (`Trainable`, `DistRef`, `ModelRef`, `ConnRule`), topological nodes, root container, connectivity/weight/delay semantics. |
| [3 — Frontend A: Python `NetSpec` builder](./03-frontend-python.md)   | Fluent builder API, handles, view algebra (incl. spatial / compartmental / tag-predicate), value wrappers (`Trainable` / `DistRef` / `Noise` / `VariableRef`), populations (point / spatial / morphological), projections with spatial and compartment-targeted rules, inputs / signals / schedules, observables, composition forms (subnetwork / sequential / DAG), plasticity (per-projection through structural and homeostatic), construction-time errors, build-time variables (`net.variable`, `variables=` build kwarg). |
| [4 — Frontend B: YAML/JSON DSL](./04-frontend-yaml.md)                | Top-level YAML schema, lexical conventions, JSON Schema, parameter sweeps.                                    |
| [5 — Backend protocol & round-trip](./05-backends.md)                 | `Backend` protocol at `brainpy.state.backend`; backend implementations as top-level modules (`brainpy.state.clock` / `event` / `bptt` / `eprop` / `eventprop` / `ppprop` / `nir` / `onnxspike`); third-party backends; capability declarations; IR round-trip and equivalence guarantees. |
| [6 — Export backends: Neuromorphic IR](./06-export-nir.md)            | Why NIR export, the export workflow, mapping NetIR → NIR (neurons, projections, inputs, topology, units), lossy taxonomy, strict mode, metadata sidecar, other export targets. |
| [7 — Registry](./07-registry.md)                                      | Connectivity, initializer, neuron/synapse/output/input/plasticity, and layer registries. Third-party registration via entry points. |
| [8 — CLI & visualization](./08-cli-and-viz.md)                        | `brainpy` CLI commands, visualization modes (graph, layers, matrix, params), renderers (Mermaid, Graphviz, HTML, Matplotlib). |
| [9 — Determinism & validation](./09-determinism-validation.md)        | Determinism contract (G4), validation rule catalog (spec-level errors `SPEC-NNN`, NIR export notices `EXPORT-NIR-NNN`). |
| [10 — Implementation](./10-implementation.md)                         | Mapping to the existing codebase, testing strategy, relationship to the existing `_network` API.              |
| [11 — Appendix](./11-appendix.md)                                     | Decision log (D1–D29), Python ↔ YAML cheat sheet, open questions.                                             |

---

## Goal map

Each goal (G1–G11 in [Chapter 1](./01-overview.md#12-goals)) has its primary chapter and its supporting chapters:

| Goal | Headline                                  | Primary chapter                             | Also relevant                                                |
|------|-------------------------------------------|---------------------------------------------|--------------------------------------------------------------|
| G1   | Declarative spec                          | [1](./01-overview.md), [3](./03-frontend-python.md) | [4](./04-frontend-yaml.md)                                   |
| G2   | Backend pluralism                         | [5](./05-backends.md)                       | [6](./06-export-nir.md)                                      |
| G3   | Physical-units-first                      | [2](./02-ir.md)                             | [6](./06-export-nir.md) (units stripped on export)            |
| G4   | Deterministic lowering                    | [9](./09-determinism-validation.md)         | [2](./02-ir.md), [5](./05-backends.md)                       |
| G5   | Composable specs (subnetworks)            | [3](./03-frontend-python.md), [4](./04-frontend-yaml.md) | —                                                |
| G6   | Inspectable IR                            | [2](./02-ir.md)                             | [8](./08-cli-and-viz.md)                                      |
| G7   | Deep / neuromorphic SNNs                  | [3](./03-frontend-python.md) (§3.11)        | [7](./07-registry.md) (layer registry)                       |
| G8   | View algebra                              | [3](./03-frontend-python.md) (§3.9)         | —                                                            |
| G9   | Trainable declarations                    | [3](./03-frontend-python.md) (§3.10)        | [2](./02-ir.md), [5](./05-backends.md)                       |
| G10  | Visualization                             | [8](./08-cli-and-viz.md)                    | —                                                            |
| G11  | Neuromorphic-IR export                    | [6](./06-export-nir.md)                     | [9](./09-determinism-validation.md) (`EXPORT-NIR-*` notices) |

The spec is immutable after `.finalize()`. Parameters that need to
vary across runs are declared as build-time variables
(`net.variable(...)`, see [Chapter 3 §3.14](./03-frontend-python.md))
and bound by name at `backend.build(ir, ..., variables={...})`. There
is no post-definition mutation API on the IR or on built artifacts
(decision log [D26 / D27](./11-appendix.md)).

---

## How to read this RFC

- **Reviewers / decision-makers:** start with [Chapter 1](./01-overview.md) (problem, novelty, goals, architecture), then [Chapter 11](./11-appendix.md) (decision log).
- **DSL users:** [Chapter 3](./03-frontend-python.md) (Python) and [Chapter 4](./04-frontend-yaml.md) (YAML) are the user-facing surface; [Chapter 11](./11-appendix.md) has the Python ↔ YAML cheat sheet.
- **Backend authors:** [Chapter 2](./02-ir.md) (IR contract), [Chapter 5](./05-backends.md) (protocol), [Chapter 7](./07-registry.md) (registration), [Chapter 9](./09-determinism-validation.md) (determinism guarantees).
- **NIR / neuromorphic-hardware integrators:** [Chapter 6](./06-export-nir.md) end to end.
- **Implementers landing this in `brainpy_state`:** [Chapter 10](./10-implementation.md) (codebase mapping, tests, relationship to existing `_network`).

Cross-references in chapter bodies use `§N.M.P` notation; the top-level
component `N` is the chapter (= file) number.
