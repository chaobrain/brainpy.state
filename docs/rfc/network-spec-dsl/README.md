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

See [§1.1 in Chapter 1](./01-overview.md#11-novelty-and-prior-art) for the
prior-art comparison table.

---

## Table of contents

| Chapter                                                               | Source sections        | What's in it                                                                                                  |
|-----------------------------------------------------------------------|------------------------|---------------------------------------------------------------------------------------------------------------|
| [1 — Overview](./01-overview.md)                                      | §1 · §1.1 · §2 · §3 · §4 | Problem statement, novelty pitch and prior-art table, goals (G1–G12), user populations, non-goals, primitive node kinds, architecture diagram. |
| [2 — The IR (`NetIR`)](./02-ir.md)                                    | §5                     | Canonical frozen-dataclass IR: value wrappers (`Trainable`, `DistRef`, `ModelRef`, `ConnRule`), topological nodes, root container, connectivity/weight/delay semantics. |
| [3 — Frontend A: Python `NetSpec` builder](./03-frontend-python.md)   | §6                     | Fluent builder API, handles, Brunel example, subnetworks, view algebra, trainable parameters, deep-SNN sequential composition, construction-time errors, post-definition parameter modification (G12) with path language, `ParamPatch`, `ParameterView`. |
| [4 — Frontend B: YAML/JSON DSL](./04-frontend-yaml.md)                | §7                     | Top-level YAML schema, lexical conventions, JSON Schema, parameter sweeps.                                    |
| [5 — Backend protocol & round-trip](./05-backends.md)                 | §8 · §10               | `Backend` protocol, third-party backends, capability declarations, IR round-trip and equivalence guarantees.   |
| [6 — Export backends: Neuromorphic IR](./06-export-nir.md)            | §9                     | Why NIR export, the export workflow, mapping NetIR → NIR (neurons, projections, inputs, topology, units), lossy taxonomy, strict mode, metadata sidecar, other export targets. |
| [7 — Registry](./07-registry.md)                                      | §11                    | Connectivity, initializer, neuron/synapse/output/input/plasticity, and layer registries. Third-party registration via entry points. |
| [8 — CLI & visualization](./08-cli-and-viz.md)                        | §12                    | `brainpy` CLI commands, visualization modes (graph, layers, matrix, params), renderers (Mermaid, Graphviz, HTML, Matplotlib). |
| [9 — Determinism & validation](./09-determinism-validation.md)        | §13 · §14              | Determinism contract (G4), validation rule catalog (spec-level errors `SPEC-NNN`, NIR export notices `EXPORT-NIR-NNN`). |
| [10 — Implementation](./10-implementation.md)                         | §15 · §16 · §17        | Mapping to the existing codebase, testing strategy, relationship to the existing `_network` API.              |
| [11 — Appendix](./11-appendix.md)                                     | §18 · §19 · §20        | Decision log (D1–D28), Python ↔ YAML cheat sheet, open questions.                                             |

---

## Goal map

Each goal (G1–G12 in [Chapter 1](./01-overview.md#2-goals)) has its primary chapter and its supporting chapters:

| Goal | Headline                                  | Primary chapter                             | Also relevant                                                |
|------|-------------------------------------------|---------------------------------------------|--------------------------------------------------------------|
| G1   | Declarative spec                          | [1](./01-overview.md), [3](./03-frontend-python.md) | [4](./04-frontend-yaml.md)                                   |
| G2   | Backend pluralism                         | [5](./05-backends.md)                       | [6](./06-export-nir.md)                                      |
| G3   | Physical-units-first                      | [2](./02-ir.md)                             | [6](./06-export-nir.md) (units stripped on export)            |
| G4   | Deterministic lowering                    | [9](./09-determinism-validation.md)         | [2](./02-ir.md), [5](./05-backends.md)                       |
| G5   | Composable specs (subnetworks)            | [3](./03-frontend-python.md), [4](./04-frontend-yaml.md) | —                                                |
| G6   | Inspectable IR                            | [2](./02-ir.md)                             | [8](./08-cli-and-viz.md)                                      |
| G7   | Deep / neuromorphic SNNs                  | [3](./03-frontend-python.md) (§6.7)         | [7](./07-registry.md) (layer registry)                       |
| G8   | View algebra                              | [3](./03-frontend-python.md) (§6.5)         | —                                                            |
| G9   | Trainable declarations                    | [3](./03-frontend-python.md) (§6.6)         | [2](./02-ir.md), [5](./05-backends.md)                       |
| G10  | Visualization                             | [8](./08-cli-and-viz.md)                    | —                                                            |
| G11  | Neuromorphic-IR export                    | [6](./06-export-nir.md)                     | [9](./09-determinism-validation.md) (`EXPORT-NIR-*` notices) |
| G12  | Post-definition parameter modification    | [3](./03-frontend-python.md) (§6.9)         | [11](./11-appendix.md) (D26, D27)                            |

---

## How to read this RFC

- **Reviewers / decision-makers:** start with [Chapter 1](./01-overview.md) (problem, novelty, goals, architecture), then [Chapter 11](./11-appendix.md) (decision log).
- **DSL users:** [Chapter 3](./03-frontend-python.md) (Python) and [Chapter 4](./04-frontend-yaml.md) (YAML) are the user-facing surface; [Chapter 11](./11-appendix.md) has the Python ↔ YAML cheat sheet.
- **Backend authors:** [Chapter 2](./02-ir.md) (IR contract), [Chapter 5](./05-backends.md) (protocol), [Chapter 7](./07-registry.md) (registration), [Chapter 9](./09-determinism-validation.md) (determinism guarantees).
- **NIR / neuromorphic-hardware integrators:** [Chapter 6](./06-export-nir.md) end to end.
- **Implementers landing this in `brainpy_state`:** [Chapter 10](./10-implementation.md) (codebase mapping, tests, relationship to existing `_network`).

Cross-references in chapter bodies use the original `§N.M` notation; the
table above maps each `§N` to the chapter that contains it.
