# Chapter 1 — Overview, problem statement, and goals

> Part of the [Network Specification DSL RFC](./README.md).

## 1. Problem statement

Today, a `brainpy.state` model commits to a *runtime* at definition time:

- The class body chooses ODE integrators (`AdaptiveRungeKuttaStep`, exact propagators).
- The update schedule is clock-driven via `update(t)`.
- The gradient story is implicit: autodiff flows through surrogate spikes.

Switching a model from clock-driven simulation to event-driven simulation,
from BPTT to e-prop / event-prop, or exporting to a neuromorphic-hardware
toolchain requires **rewriting the model**. The current
`brainpy_state._network.Network` / `Builder` is an imperative
`brainstate.nn.Module`; populations and projections store JAX state in-place
and step in lockstep with `brainstate.environ['t']`.

We need a layer **above** the existing modules that lets users:

1. **Describe the network once** — populations, synapses, projections, inputs,
   recorders, parameters with physical units, trainable markers, layer
   structure for deep SNNs.
2. **Pick a runtime later** — choose a simulation backend (clock / event), a
   training backend (BPTT / e-prop / event-prop), or an export backend
   (NIR / ONNX-Spike / Nengo / …) without touching the spec.

### 1.1 Novelty and prior art

The novelty of `brainpy.state.spec` is **not** the specification surface.
PyNN, NESTML, Brian2, and Nengo have shown for over a decade that an SNN
can be described declaratively, and we deliberately borrow conventions
from that lineage (units-first parameters, frontend-agnostic IR, sparse
projection rules). Treating the DSL itself as the contribution would be
reinventing a well-known wheel.

The novelty is that **the same network description drives four
mathematically distinct SNN training paradigms from a single IR**:

- **BPTT** with surrogate gradients (the snnTorch / Norse default),
- **Event-prop** (Wunderlich & Pehle 2021) — exact gradients of the
  spike times for clock-free training,
- **RTRL / forward-mode autodiff** — online gradient estimation that
  does not store the full unrolled graph,
- **Eligibility-trace methods** — e-prop (Bellec et al. 2020) and
  related local online learning rules.

Every existing SNN framework commits to exactly one of these paradigms
at model-definition time. Switching paradigms (e.g. comparing event-prop
to BPTT on the same architecture) means rewriting the model in a
different framework, with all of the unit, topology, and initialization
drift that implies. In `brainpy.state.spec`, the gradient story is a
backend kwarg: a researcher A/Bs paradigms by changing one line, while
the spec, the seed, and the connectivity rules are bit-identical.

NIR export (G11) is a fourth axis of pluralism — deployment — but is
**not** the load-bearing novelty. NIR is a community standard
(Neuromorphic Intermediate Representation, neuromorphs/NIR); we adopt
it rather than invent it, and several frameworks above also ship a NIR
exporter. The training-paradigm axis is what is genuinely new.

#### 1.1.1 Prior-art comparison

| Framework         | Modeling surface           | Training paradigm(s)                                    | Deployment              |
|-------------------|----------------------------|---------------------------------------------------------|-------------------------|
| snnTorch          | PyTorch modules            | BPTT (surrogate grad)                                   | PyTorch                 |
| Norse             | PyTorch modules            | BPTT (surrogate grad)                                   | PyTorch                 |
| BindsNET          | PyTorch modules            | BPTT + Hebbian / STDP                                   | PyTorch                 |
| Nengo             | NEF Network DSL            | NEF / PES                                               | Nengo, Loihi            |
| PyNN / Brian2     | Declarative DSL            | Plasticity rules only (no global gradient)              | NEST / NEURON / GPU     |
| Lava (Intel)      | Process graph              | On-chip plasticity                                      | Loihi 2                 |
| **brainpy.state** | **Declarative IR**         | **BPTT + event-prop + RTRL + eligibility-trace**        | **clock/event + NIR**   |

Every row except the last commits to one column-2 entry. The bold row
is the wedge: a single IR, four gradient flavors, deployment plurality
on top.

#### 1.1.2 Why this matters

The load-bearing user story is the comparative study. SNN training is a
moving target — event-prop is recent, RTRL variants are an active
research area, and eligibility-trace methods are increasingly important
for neuromorphic hardware that cannot afford BPTT's memory footprint.
Researchers who currently want to compare these methods either
re-implement their model in three frameworks (introducing drift) or
pick one paradigm and never benchmark the others. The spec collapses
the comparison into a backend swap, which is the same value
proposition that JAX brought to autodiff and that ONNX brought to
inference: **separate the model from the execution strategy**.

This positioning also informs scope decisions throughout the rest of
this document. Whenever a feature could land in either the spec or a
specific training backend, the tie-breaker is: *does this feature
preserve the spec's neutrality across the four paradigms?* If yes,
it belongs in the spec; if no, it belongs in a backend.

---

## 2. Goals

| ID  | Goal                                                                                     |
|-----|------------------------------------------------------------------------------------------|
| G1  | **Declarative spec.** Users describe *what* the network is, not *how* it steps. No `update()`, no integrator picks, no `jax.grad` calls. |
| G2  | **Backend pluralism.** Arbitrary number of simulation, training, and export backends behind a stable protocol. Third-party backends register via Python entry points. |
| G3  | **Physical-units-first.** All parameters carry `saiunit` units; the spec is the source of truth for dimensionality. |
| G4  | **Deterministic lowering.** `(spec, backend, seed) → artifact` is pure; re-running yields bit-identical results. |
| G5  | **Composable.** Specs nest — a sub-network is itself a spec node. |
| G6  | **Inspectable.** A built spec is serializable IR (JSON / YAML / pickle / dataclass tree). Tools can lint, diff, visualize, persist. |
| G7  | **Deep / neuromorphic SNNs.** First-class support for feedforward and recurrent deep SNNs used in brain-inspired computing: sequential composition, dense / conv / pool connectivity, layer macros, batch dimension on populations. |
| G8  | **View algebra.** Slice, index, merge, concat, split populations. Views are first-class targets of projections, inputs, and observables. |
| G9  | **Trainable declarations.** Any spec value (model parameter, weight, delay, initial state) can be marked trainable. The spec is the source of truth for *what* is learnable; backends decide *how* to gradient through it. Trainables materialize as `brainstate.nn.Param` at backend build. |
| G10 | **Visualization.** The IR is the source for graph, layer-stack, connectivity-matrix, and parameter-summary visualizations. Python API + CLI, multiple renderers (Graphviz, Mermaid, Matplotlib, HTML). |
| G11 | **Neuromorphic-IR export.** A spec lowers to the [Neuromorphic Intermediate Representation (NIR)](https://github.com/neuromorphs/NIR) for deployment on Loihi, SpiNNaker, Nengo, and other NIR-consuming platforms. The mapping is documented, deterministic, and surfaces lossy transformations explicitly. |
| G12 | **Post-definition parameter modification.** After a spec is built (or after a backend has materialized it), users can read and write parameter values — both static (e.g. `tau`, `V_th`) and dynamic (e.g. synaptic weights changing during training) — through one uniform path-addressed interface. Modifications propagate consistently to the IR and to any running backend artifact. |

### 2.1 User populations and example workloads

The DSL targets two communities; the spec serves both without forking the language:

| Community                       | Canonical example                                                            | Dominant patterns |
|---------------------------------|------------------------------------------------------------------------------|-------------------|
| Computational neuroscience      | Brunel 2000; cortical microcircuit; multi-area cortex                        | Sparse Bernoulli / fixed-indegree connectivity, biophysical units, recording-heavy, simulation-only. |
| Brain-inspired ML / neuromorphic| Spiking MLP / CNN on MNIST, DVS-Gesture; spiking RNN / LSM; deployment to Loihi or SpiNNaker | Dense / Conv / Pool layers stacked sequentially; mini-batch axis; trainable weights and (optionally) neuron parameters; loss + optimizer; classification or regression readout; NIR export for hardware deployment. |

Both communities share the IR, the registry, validation, visualization, and
the determinism contract.

### 2.2 Non-goals

- A GUI / visual editor (the IR enables one; this spec ships the IR + CLI only).
- Distributed multi-host execution (backend concern; the spec is host-agnostic).
- Mixed clock-event hybrid scheduling within a single backend (a future
  hybrid backend may consume the IR; the spec does not encode scheduling).
- Reverse (NIR-import) is **not** in scope. Specs lower to NIR; NIR does
  not lift back to specs because NIR loses unit and randomness information.

---

## 3. Primitive node kinds

The spec is a tree (containment) plus an edge set (connectivity) of typed nodes:

| Node           | Purpose                                                                    |
|----------------|----------------------------------------------------------------------------|
| `Population`   | N units of a `NeuronModel`, with init-state distribution.                  |
| `Projection`   | `(pre, post, Connectivity, SynapseModel, OutputModel, plasticity?)`. The `Connectivity` rule (from `braintools.conn`) owns per-edge `weight` and `delay`. |
| `InputSource`  | Poisson, spike-times, step current, DC, AC, image stream, …                |
| `Observable`   | What to record (spikes, voltage trace, weight snapshots, summary stats).   |
| `SubNetwork`   | A named, parameterizable spec embedded in another spec.                    |

Every node has: stable id, kind tag, frozen parameter dict (units carried),
optional children. Nodes are **values**, not modules — they do not own JAX
state. State is materialized by the backend at lowering time.

### 3.1 Compound forms (sugar that lowers to primitives)

| Compound form | Lowers to                                                                                   |
|---------------|---------------------------------------------------------------------------------------------|
| `Sequential`  | Ordered list of `(Population, Projection)` pairs. The output of each layer is the `pre` of the next. |
| `Layer`       | One `Population` (or a stateless functional layer; §6.7) plus an inbound `Projection` configured from the previous layer's output. |
| `MergedView`  | A `ViewRef` whose `population` is a synthesized id over multiple base populations; the backend de-references it as `concat`/`union` at materialization. |
| `Trainable[…]`| A value wrapper, not a node. Carries a learnability marker through the IR; the backend chooses the storage (`brainstate.nn.Param`, frozen constant, …). |
| `Group`       | A named, labelled bundle of populations / views — purely for organization and visualization. |

Compound forms appear in the IR as their lowered primitive shape **plus** a
`compounds: {...}` block on the root `NetIR` (§5) that records user intent.
Tools (viz, diff, describe, NIR export) use it; backends ignore it.

---

## 4. Architecture overview

Two frontends, one IR, three backend families:

```
  Frontend A (Python)            Frontend B (YAML/JSON)
  ──────────────────             ──────────────────────
  spec = NetSpec("brunel")       brunel.netspec.yaml
  exc = spec.population(...)     populations:
  spec.project(exc, inh, ...)      exc: ...
  ir = spec.finalize()           ir = sp.spec.load("brunel.netspec.yaml")
            │                               │
            ▼                               ▼
                  ┌──────────────────────────┐
                  │         NetIR            │   canonical, frozen,
                  │   (frozen dataclass      │   JSON-able, content-hashable
                  │    + version tag)        │
                  └──────────┬───────────────┘
            ┌────────────────┼─────────────────────┬─────────────────┐
            ▼                ▼                     ▼                 ▼
       sim backends     train backends       export backends     visualization
       (clock, event)   (bptt, eprop,        (nir, onnx-spike,   (graph, layers,
                         event-prop)          nengo, …)           matrix, params)
```

The IR is the contract. Frontends produce it; backends consume it.

---


---

**Previous:** [README](./README.md)  
**Next:** [Chapter 2 — The IR (NetIR)](./02-ir.md)
