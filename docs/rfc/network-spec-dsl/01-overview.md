# Chapter 1 — Overview, problem statement, and goals

> Part of the [Network Specification DSL RFC](./README.md).

## 1.1 Problem statement

Today, a `brainpy.state` model commits to a *runtime* at definition time:

- The class body chooses ODE integrators (`AdaptiveRungeKuttaStep`, exact propagators).
- The update schedule is clock-driven via `update(t)`.
- The gradient story is implicit: autodiff flows through surrogate spikes.

Switching a model from clock-driven simulation to event-driven simulation,
or from BPTT to e-prop / event-prop, requires **rewriting the model**. The
current `brainpy_state._network.Network` / `Builder` is an imperative
`brainstate.nn.Module`; populations and projections store JAX state in-place
and step in lockstep with `brainstate.environ['t']`.

We need a layer **above** the existing modules that lets users:

1. **Describe the network once** — populations, synapses, projections, inputs,
   recorders, parameters with physical units, trainable markers, layer
   structure for deep SNNs.
2. **Pick a runtime later** — choose a simulation backend (clock / event)
   or a training backend (BPTT / e-prop / event-prop) without touching
   the spec. `NetIR` is the canonical exchange format; backends consume
   it directly.

### 1.1.1 Novelty and prior art

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

`NetIR` is `brainpy.state`'s canonical IR — the single contract every
backend consumes. We deliberately do not adopt a foreign neuromorphic
IR as the standard exchange format; the spec module owns its IR so
unit-aware parameters, distributions, trainable markers, and seed
provenance survive across simulation and training paradigms without
the lossy conversion that a deployment-oriented IR would impose.

#### 1.1.1.1 Prior-art comparison

| Framework         | Modeling surface           | Training paradigm(s)                                    | Runtime                 |
|-------------------|----------------------------|---------------------------------------------------------|-------------------------|
| snnTorch          | PyTorch modules            | BPTT (surrogate grad)                                   | PyTorch                 |
| Norse             | PyTorch modules            | BPTT (surrogate grad)                                   | PyTorch                 |
| BindsNET          | PyTorch modules            | BPTT + Hebbian / STDP                                   | PyTorch                 |
| Nengo             | NEF Network DSL            | NEF / PES                                               | Nengo, Loihi            |
| PyNN / Brian2     | Declarative DSL            | Plasticity rules only (no global gradient)              | NEST / NEURON / GPU     |
| Lava (Intel)      | Process graph              | On-chip plasticity                                      | Loihi 2                 |
| **brainpy.state** | **Declarative IR**         | **BPTT + event-prop + RTRL + eligibility-trace**        | **clock + event**       |

Every row except the last commits to one column-2 entry. The bold row
is the wedge: a single IR, four gradient flavors, plus a clock/event
runtime pair under the same description.

#### 1.1.1.2 Why this matters

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

## 1.2 Goals

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

The spec is **immutable after `.finalize()`**: the IR is a frozen, content-hashable value. There is no path-addressed mutation API for either the IR or a built backend artifact. Values that need to vary across runs — sweeps, A/B comparisons, hyperparameter binding — are declared up front as **variables** (§3.14) and bound by name at `backend.build(...)`. Gradient-trained parameters remain declared via `Trainable` (G9) and are updated by the trainer's optimizer; that is an internal training-state concern, not user-facing IR mutation.

### 1.2.1 User populations and example workloads

The DSL targets two communities; the spec serves both without forking the language:

| Community                       | Canonical example                                                            | Dominant patterns |
|---------------------------------|------------------------------------------------------------------------------|-------------------|
| Computational neuroscience      | Brunel 2000; cortical microcircuit; multi-area cortex                        | Sparse Bernoulli / fixed-indegree connectivity, biophysical units, recording-heavy, simulation-only. |
| Brain-inspired ML / neuromorphic| Spiking MLP / CNN on MNIST, DVS-Gesture; spiking RNN / LSM                  | Dense / Conv / Pool layers stacked sequentially; mini-batch axis; trainable weights and (optionally) neuron parameters; loss + optimizer; classification or regression readout. |

Both communities share the IR, the registry, validation, visualization, and
the determinism contract.

### 1.2.2 Non-goals

- A GUI / visual editor (the IR enables one; this spec ships the IR + CLI only).
- Distributed multi-host execution (backend concern; the spec is host-agnostic).
- Mixed clock-event hybrid scheduling within a single backend (a future
  hybrid backend may consume the IR; the spec does not encode scheduling).
- Adopting a foreign neuromorphic IR as `brainpy.state`'s exchange format
  is **not** in scope. `NetIR` is the standard; converters from `NetIR`
  to third-party formats may exist as out-of-tree tools but are not part
  of this spec.

---

## 1.3 Description vs. implementation: the spec/backend boundary

G1 ("describe *what* the network is, not *how* it steps") is not a
slogan — it is a hard architectural rule that runs through every
chapter of this RFC, every backend, and every domain extension
(Chapter 5). This section makes it explicit.

### 1.3.1 The rule

The specification language describes **the model**: dynamics,
topology, parameters with physical units, initial conditions,
connectivity, inputs, observables. It does **not** describe **how to
compute it**: numerical integrator, time-step discretization,
compartmentalization policy, ring-buffer sizes, accelerator placement,
gradient flavor, memory layout. Those are backend-side realization
choices, and the same `NetIR` flows through every backend without
encoding any of them.

| In the spec (`NetIR` fields)                                      | In the backend (`build()` kwargs)                                  |
|-------------------------------------------------------------------|---------------------------------------------------------------------|
| Model kind and its parameters (with `saiunit` units)              | Solver / integrator (`"dopri5"`, `"staggered"`, `"exact"`, …)       |
| Topology (`size`, connectivity rules, coupling matrices)          | Time step `dt`, integration tolerance, adaptive-step controllers    |
| Dynamics terms — including noise / stochastic forcing             | Discretization (number of compartments, `cv_policy`, mesh)          |
| Initial conditions and state-init distributions                   | Delay-buffer sizes, ring-buffer policy, memory layout               |
| Spike threshold, reset rule, refractoriness (as model)            | JIT compilation flags, accelerator placement (CPU / GPU / TPU)      |
| Inputs, schedules, what to observe                                | Gradient flavor (BPTT / event-prop / RTRL / eligibility)            |
| Trainable markers (which leaves can be learned, §3.10)            | Optimizer, loss, batch sampling — supplied by user, never in the IR |

A useful test when classifying a candidate field: ask whether two
researchers running the *same biological model* could legitimately
disagree on the value while both publishing it as "the same model."
If yes (different solver, different `dt`, different `cv_policy`),
the field belongs in `build()` kwargs. If no (different `tau_m`,
different connectivity probability, different noise variance), the
field belongs in the IR.

### 1.3.2 Why this rule is load-bearing

Three properties of the RFC stand or fall on it:

1. **Training-paradigm pluralism (§1.1.1).** The novelty pitch — one
   spec, four mathematically distinct gradient flavors — only works
   if the gradient flavor is not baked into the model description.
   The moment the spec carries a solver name, "switching to
   event-prop" stops being a backend swap and becomes a model
   rewrite.
2. **Content-hash determinism (G4).** `(spec, backend, seed)` →
   bit-identical artifact requires that the *spec* commit only to the
   model. Numerical knobs change between runs of the same study; if
   they were in the IR, every change would reshape the content hash
   and break build-cache reuse, golden-IR fixtures, and sweep
   deduplication.
3. **Reproducibility across hardware.** A spec written on a laptop
   and run on a multi-GPU cluster should produce the same model,
   even if the cluster build picks a different solver and JIT layout.
   That is only true when the spec is hardware- and realization-free.

### 1.3.3 How numerical knobs reach the backend

Backends accept two mappings as `build()` kwargs:

```python
sim = clock.build(
    ir, seed=0, dt=0.1*u.ms,
    # Per-node-kind defaults — apply to every node of this kind:
    kind_options={
        "brainpy.population":             dict(solver="exact"),
        "braincell.morph_population":     dict(solver="staggered",
                                              cv_policy="per_branch"),
    },
    # Per-node overrides keyed by IR node id:
    node_options={
        "L5_pyr": dict(solver="exp_euler"),
    },
)
```

Backends document their own option vocabulary and defaults; handlers
read the merged view through `BuildContext.options_for(node)` (Chapter
10 §5.7). Two builds of the same IR with different option mappings
produce two artifacts with the same `content_hash` but different
runtime behaviour — that is precisely the property the rule is
designed to preserve.

### 1.3.4 What this means for domain extensions

The rule binds **every** node kind, built-in or contributed by a
domain extension (Chapter 5). braincell's `MorphPopulation` does
not carry a solver or a `cv_policy`; brainmass's `MassPopulation`
does not carry an SDE solver or a delay-buffer size. Both pass those
values through the backend's `kind_options` / `node_options` channel.
A domain author who is tempted to put a numerical knob on a node
should ask the same publication test (§1.3.1) and almost always come
out on the backend side.

The full enforcement of this rule for extensions lives in
[Chapter 5 §5.1.1](05-frontend-domain-extensions.md#511-what-the-spec-describes--and-what-it-does-not) (table, examples, decision log entry D24a).

---

## 1.4 Primitive node kinds

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

### 1.4.1 Compound forms (sugar that lowers to primitives)

| Compound form | Lowers to                                                                                   |
|---------------|---------------------------------------------------------------------------------------------|
| `Sequential`  | Ordered list of `(Population, Projection)` pairs. The output of each layer is the `pre` of the next. |
| `Layer`       | One `Population` (or a stateless functional layer; §3.7) plus an inbound `Projection` configured from the previous layer's output. |
| `MergedView`  | A `ViewRef` whose `population` is a synthesized id over multiple base populations; the backend de-references it as `concat`/`union` at materialization. |
| `Trainable[…]`| A value wrapper, not a node. Carries a learnability marker through the IR; the backend chooses the storage (`brainstate.nn.Param`, frozen constant, …). |
| `Group`       | A named, labelled bundle of populations / views — purely for organization and visualization. |

Compound forms appear in the IR as their lowered primitive shape **plus** a
`compounds: {...}` block on the root `NetIR` (§2) that records user intent.
Tools (viz, diff, describe) use it; backends ignore it.

---

## 1.5 Architecture overview

Two frontends, one IR, two backend families plus visualization:

```
  Frontend A (Python)            Frontend B (YAML/JSON)
  ──────────────────             ──────────────────────
  net = spec.NetSpec("brunel")   brunel.netspec.yaml
  exc = net.population(...)      populations:
  net.project(exc, inh, ...)       exc: ...
  ir = net.finalize()            ir = spec.load("brunel.netspec.yaml")
            │                               │
            ▼                               ▼
                  ┌──────────────────────────┐
                  │         NetIR            │   canonical, frozen,
                  │   (frozen dataclass      │   JSON-able, content-hashable
                  │    + version tag)        │   — the standard exchange format
                  └──────────┬───────────────┘
                  ┌──────────┼──────────────────────┐
                  ▼          ▼                      ▼
            sim backends     train backends     visualization
            (clock, event)   (bptt, eprop,      (graph, layers,
                              event-prop)        matrix, params)
```

The IR is the contract. Frontends produce it; backends consume it.

---


---

**Previous:** [README](./README.md)  
**Next:** [Chapter 2 — The IR (NetIR)](./02-ir.md)
