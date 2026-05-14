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
| [1 — Overview](./01-overview.md)                                      | Problem statement, novelty pitch and prior-art table, goals (G1–G10), user populations, non-goals, **the description/implementation separation principle (spec describes the model; backend chooses the realization)**, primitive node kinds, architecture diagram. |
| [2 — The IR (`NetIR`)](./02-ir.md)                                    | Canonical frozen-dataclass IR: value wrappers (`Trainable`, `DistRef`, `ModelRef`, `ConnRule`), topological nodes, root container, connectivity/weight/delay semantics. |
| [3 — Frontend A: Python `NetSpec` builder](./03-frontend-python.md)   | Fluent builder API, handles, view algebra (incl. spatial / compartmental / tag-predicate), value wrappers (`Trainable` / `DistRef` / `Noise` / `VariableRef`), populations (point / spatial / morphological), projections with spatial and compartment-targeted rules, inputs / signals / schedules, observables, composition forms (subnetwork / sequential / DAG), plasticity (per-projection through structural and homeostatic), construction-time errors, build-time variables (`net.variable`, `variables=` build kwarg). |
| [4 — Frontend B: YAML/JSON DSL](./04-frontend-yaml.md)                | Top-level YAML schema, lexical conventions, JSON Schema, parameter sweeps.                                    |
| [5 — Domain extensions](05-frontend-domain-extensions.md)                    | The DSL substrate: **`IRNode`, `ViewHandle`, builder-verb, codec, and backend-dispatch protocols** that adjacent domains apply. Full code interfaces for the [`braincell`](https://github.com/chaobrain/braincell) and [`brainmass`](https://github.com/chaobrain/brainmass) extensions — node kinds, view handles, builder verbs, backend handlers, and user-facing examples. |
| [6 — Backend design: principles, protocol, lowering substrate](./06-backends.md) | **The substrate every backend stands on.** Four design principles (P1–P4: paradigm-neutral IR, pure lowering, open handlers / closed backends, typed capability mismatches). `Backend` protocol at `brainpy.state.backend`; top-level backend modules (`clock` / `clock_joint` / `clock_scan` / `event` / `bptt` / `eprop` / `eventprop` / `ppprop`); third-party backends via entry points; shared `brainpy_state/lowering/` package; node-kind handler protocol (`NodeHandler`, `LoweredNode`, `LoweredNet`, `Feature` set); capability declaration & three-class mismatch policy (hard miss / soft miss / informational); variable + Trainable resolution; determinism contract; IR round-trip equivalence. |
| [7 — Registry](./07-registry.md)                                      | Connectivity, initializer, neuron/synapse/output/input/plasticity, and layer registries. Third-party registration via entry points. |
| [8 — CLI & visualization](./08-cli-and-viz.md)                        | `brainpy` CLI commands, visualization modes (graph, layers, matrix, params), renderers (Mermaid, Graphviz, HTML, Matplotlib). |
| [9 — Determinism & validation](./09-determinism-validation.md)        | Determinism contract (G4), validation rule catalog (spec-level errors `SPEC-NNN`). |
| [10 — Implementation](./10-implementation.md)                         | Mapping to the existing codebase (`_network/`, `_brainpy/`, `_nest/`), testing strategy, relationship to the existing module-level APIs. |
| [11 — Appendix](./11-appendix.md)                                     | Decision log (D1–D29), Python ↔ YAML cheat sheet, open questions.                                             |
| [12 — Backend implementation (sim + BPTT)](./12-backend-impl-sim-bptt.md) | Execution architecture for `clock`, `clock_joint`, `clock_scan`, `bptt`: shared `lowering/` substrate, handler protocol, step × time composition matrix, capability cards, fallback policy. Replaces Chapter 10's `clock=adapter-over-_network` view with a fresh substrate. |

---

## Goal map

Each goal (G1–G10 in [Chapter 1](./01-overview.md#12-goals)) has its primary chapter and its supporting chapters:

| Goal | Headline                                  | Primary chapter                             | Also relevant                                                |
|------|-------------------------------------------|---------------------------------------------|--------------------------------------------------------------|
| G1   | Declarative spec                          | [1](./01-overview.md), [3](./03-frontend-python.md) | [4](./04-frontend-yaml.md)                                   |
| G2   | Backend pluralism                         | [6](./06-backends.md)                       | —                                                            |
| G3   | Physical-units-first                      | [2](./02-ir.md)                             | —                                                            |
| G4   | Deterministic lowering                    | [9](./09-determinism-validation.md)         | [2](./02-ir.md), [6](./06-backends.md)                       |
| G5   | Composable specs (subnetworks)            | [3](./03-frontend-python.md), [4](./04-frontend-yaml.md) | —                                                |
| G6   | Inspectable IR                            | [2](./02-ir.md)                             | [8](./08-cli-and-viz.md)                                     |
| G7   | Deep / neuromorphic SNNs                  | [3](./03-frontend-python.md) (§3.11)        | [7](./07-registry.md) (layer registry)                       |
| G8   | View algebra                              | [3](./03-frontend-python.md) (§3.9)         | —                                                            |
| G9   | Trainable declarations                    | [3](./03-frontend-python.md) (§3.10)        | [2](./02-ir.md), [6](./06-backends.md)                       |
| G10  | Visualization                             | [8](./08-cli-and-viz.md)                    | —                                                            |

The spec is immutable after `.finalize()`. Parameters that need to
vary across runs are declared as build-time variables
(`net.variable(...)`, see [Chapter 3 §3.14](./03-frontend-python.md))
and bound by name at `backend.build(ir, ..., variables={...})`. There
is no post-definition mutation API on the IR or on built artifacts
(decision log [D19 / D20](./11-appendix.md)).

---

## How to read this RFC

- **Reviewers / decision-makers:** start with [Chapter 1](./01-overview.md) (problem, novelty, goals, architecture), then [Chapter 11](./11-appendix.md) (decision log).
- **DSL users:** [Chapter 3](./03-frontend-python.md) (Python) and [Chapter 4](./04-frontend-yaml.md) (YAML) are the user-facing surface; [Chapter 11](./11-appendix.md) has the Python ↔ YAML cheat sheet.
- **Backend authors:** [Chapter 2](./02-ir.md) (IR contract), [Chapter 6](./06-backends.md) (protocol), [Chapter 7](./07-registry.md) (registration), [Chapter 9](./09-determinism-validation.md) (determinism guarantees).
- **Implementers landing this in `brainpy_state`:** [Chapter 10](./10-implementation.md) (codebase mapping, tests, relationship to the existing `_network` / `_brainpy` / `_nest` modules).
- **Domain-extension authors (out-of-tree braincell / brainmass / …):** [Chapter 5](05-frontend-domain-extensions.md) is the substrate contract — `IRNode`, `ViewHandle`, builder verbs, codecs, backend dispatch — with full worked code for both braincell and brainmass. [Chapter 7](./07-registry.md) and [Chapter 6](./06-backends.md) cover the registry and backend protocols you plug into.

Cross-references in chapter bodies use `§N.M.P` notation; the top-level
component `N` is the chapter (= file) number.
