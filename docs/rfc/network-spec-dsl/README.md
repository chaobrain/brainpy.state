# Network Specification DSL for brainpy.state — Requirements

**Status:** Requirements specification (single integrated design)
**Owner:** TBD
**Date:** 2026-05-13
**Scope:** New `brainpy_state.spec` module; existing `brainpy_state._network` becomes the substrate of the `clock` simulation backend.

---

## What this RFC proposes

A new `brainpy_state.spec` module: a declarative DSL that describes spiking
neural networks once and lowers to multiple simulation and training
backends without the model being rewritten. Two frontends (fluent
Python builder and YAML/JSON data DSL) produce one canonical, frozen,
content-hashable IR (`NetIR`). `NetIR` is the standard exchange format
for `brainpy.state` networks; backends consume it directly.

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
seed, and connectivity rules stay bit-identical.

See [§1.1.1 in Chapter 1](./01-overview.md#111-novelty-and-prior-art) for the
prior-art comparison table.

---

## Table of contents

Section numbering follows file numbering: chapter N's content is §N (with sub-sections §N.x, §N.x.y, …).

| Chapter                                                               | What's in it                                                                                                  |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [1 — Overview](./01-overview.md)                                      | Problem statement, novelty pitch and prior-art table, goals (G1–G10), user populations, non-goals, primitive node kinds, architecture diagram. |
| [2 — The IR (`NetIR`)](./02-ir.md)                                    | Canonical frozen-dataclass IR: value wrappers (`Trainable`, `DistRef`, `ModelRef`, `ConnRule`), topological nodes, root container, connectivity/weight/delay semantics. |
| [3 — Frontend A: Python `NetSpec` builder](./03-frontend-python.md)   | Fluent builder API, handles, view algebra (incl. spatial / compartmental / tag-predicate), value wrappers (`Trainable` / `DistRef` / `Noise` / `VariableRef`), populations (point / spatial / morphological), projections with spatial and compartment-targeted rules, inputs / signals / schedules, observables, composition forms (subnetwork / sequential / DAG), plasticity (per-projection through structural and homeostatic), construction-time errors, build-time variables (`net.variable`, `variables=` build kwarg). |
| [4 — Frontend B: YAML/JSON DSL](./04-frontend-yaml.md)                | Top-level YAML schema, lexical conventions, JSON Schema, parameter sweeps.                                    |
| [5 — Backend protocol & round-trip](./05-backends.md)                 | `Backend` protocol at `brainpy.state.backend`; backend implementations as top-level modules (`brainpy.state.clock` / `event` / `bptt` / `eprop` / `eventprop` / `ppprop`); third-party backends; capability declarations; IR round-trip and equivalence guarantees. |
| [6 — Registry](./06-registry.md)                                      | Connectivity, initializer, neuron/synapse/output/input/plasticity, and layer registries. Third-party registration via entry points. |
| [7 — CLI & visualization](./07-cli-and-viz.md)                        | `brainpy` CLI commands, visualization modes (graph, layers, matrix, params), renderers (Mermaid, Graphviz, HTML, Matplotlib). |
| [8 — Determinism & validation](./08-determinism-validation.md)        | Determinism contract (G4), validation rule catalog (spec-level errors `SPEC-NNN`). |
| [9 — Implementation](./09-implementation.md)                          | Mapping to the existing codebase (`_network/`, `_brainpy/`, `_nest/`), testing strategy, relationship to the existing module-level APIs. |
| [10 — Appendix](./10-appendix.md)                                     | Decision log (D1–D24), Python ↔ YAML cheat sheet, open questions.                                             |

---

## Goal map

Each goal (G1–G10 in [Chapter 1](./01-overview.md#12-goals)) has its primary chapter and its supporting chapters:

| Goal | Headline                                  | Primary chapter                             | Also relevant                                                |
|------|-------------------------------------------|---------------------------------------------|--------------------------------------------------------------|
| G1   | Declarative spec                          | [1](./01-overview.md), [3](./03-frontend-python.md) | [4](./04-frontend-yaml.md)                                   |
| G2   | Backend pluralism                         | [5](./05-backends.md)                       | —                                                            |
| G3   | Physical-units-first                      | [2](./02-ir.md)                             | —                                                            |
| G4   | Deterministic lowering                    | [8](./08-determinism-validation.md)         | [2](./02-ir.md), [5](./05-backends.md)                       |
| G5   | Composable specs (subnetworks)            | [3](./03-frontend-python.md), [4](./04-frontend-yaml.md) | —                                                |
| G6   | Inspectable IR                            | [2](./02-ir.md)                             | [7](./07-cli-and-viz.md)                                     |
| G7   | Deep / neuromorphic SNNs                  | [3](./03-frontend-python.md) (§3.11)        | [6](./06-registry.md) (layer registry)                       |
| G8   | View algebra                              | [3](./03-frontend-python.md) (§3.9)         | —                                                            |
| G9   | Trainable declarations                    | [3](./03-frontend-python.md) (§3.10)        | [2](./02-ir.md), [5](./05-backends.md)                       |
| G10  | Visualization                             | [7](./07-cli-and-viz.md)                    | —                                                            |

The spec is immutable after `.finalize()`. Parameters that need to
vary across runs are declared as build-time variables
(`net.variable(...)`, see [Chapter 3 §3.14](./03-frontend-python.md))
and bound by name at `backend.build(ir, ..., variables={...})`. There
is no post-definition mutation API on the IR or on built artifacts
(decision log [D19 / D20](./10-appendix.md)).

---

## How to read this RFC

- **Reviewers / decision-makers:** start with [Chapter 1](./01-overview.md) (problem, novelty, goals, architecture), then [Chapter 10](./10-appendix.md) (decision log).
- **DSL users:** [Chapter 3](./03-frontend-python.md) (Python) and [Chapter 4](./04-frontend-yaml.md) (YAML) are the user-facing surface; [Chapter 10](./10-appendix.md) has the Python ↔ YAML cheat sheet.
- **Backend authors:** [Chapter 2](./02-ir.md) (IR contract), [Chapter 5](./05-backends.md) (protocol), [Chapter 6](./06-registry.md) (registration), [Chapter 8](./08-determinism-validation.md) (determinism guarantees).
- **Implementers landing this in `brainpy_state`:** [Chapter 9](./09-implementation.md) (codebase mapping, tests, relationship to the existing `_network` / `_brainpy` / `_nest` modules).

Cross-references in chapter bodies use `§N.M.P` notation; the top-level
component `N` is the chapter (= file) number.
