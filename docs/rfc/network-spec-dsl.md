# Network Specification DSL for brainpy.state — Requirements

**Status:** Requirements specification (single integrated design)
**Owner:** TBD
**Date:** 2026-05-13
**Scope:** New `brainpy_state.spec` module; existing `brainpy_state._network` becomes the substrate of the `clock` simulation backend.

---

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

## 5. The IR (`NetIR`)

Frozen dataclasses; pytree-registered for `jax.tree_util`; pickleable;
serializable to/from JSON / YAML via a stable encoder.

```python
# brainpy_state/spec/ir.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Optional, Tuple, Union, Any
import saiunit as u

IR_VERSION = "netir/1.0"

# ── Value wrappers ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Trainable:
    """Marker wrapping any spec value that should be learnable.

    The wrapped ``value`` is the initial value (scalar, ``u.Quantity``,
    ``DistRef``, or array). Backends that support training materialize each
    ``Trainable`` as a ``brainstate.nn.Param``; backends that do not
    (e.g. ``clock`` simulation-only, ``nir`` export) treat it as a
    constant and emit a ``TrainableIgnored`` notice unless ``required=True``.
    """
    value: Any
    name: Optional[str] = None
    constraint: Optional[str] = None       # "positive" | "unit_norm" | "clip:lo,hi"
    required: bool = False

@dataclass(frozen=True)
class DistRef:
    """A parameter drawn from a distribution / initializer at lowering time.

    Resolves to a `braintools.init.Initialization` at backend build.
    `kind` matches a class in `braintools.init` (Normal, LogNormal, Uniform,
    TruncatedNormal, Constant, KaimingNormal, XavierNormal, …).
    """
    kind: str
    params: Mapping[str, Any]

@dataclass(frozen=True)
class ModelRef:
    """Reference to a registered model class (neuron / synapse / output / plasticity / input source / layer macro)."""
    kind: str
    params: Mapping[str, Any]              # scalars, u.Quantity, DistRef, or Trainable

@dataclass(frozen=True)
class ConnRule:
    """Reference to a `braintools.conn.Connectivity` rule.

    `kind` is the PascalCase class name in ``braintools.conn`` or a
    supplementary rule registered under the same protocol. `params` carries
    the rule constructor kwargs, including the canonical `weight` and
    `delay` (which are first-class on ``braintools.conn.Connectivity``).
    """
    kind: str
    params: Mapping[str, Any]

# ── Topological nodes ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PopulationNode:
    id: str
    model: ModelRef
    size: Union[int, Tuple[int, ...]]      # int or shape tuple (e.g. (C, H, W))
    batch: Optional[int] = None            # leading batch axis for deep SNNs
    init: Optional[Mapping[str, Union[Any, DistRef, Trainable]]] = None
    tags: Tuple[str, ...] = ()

@dataclass(frozen=True)
class ViewRef:
    """A view into one or more populations.

    Exactly one of {whole, slice_, indices, merge} is materially set.
    A merge view references multiple populations; all members must have
    compatible per-unit shape (the leading element count may differ).
    """
    population: str                        # primary id; for merge views, synthesized
    slice_: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None
    indices: Optional[Tuple[int, ...]] = None
    merge: Optional[Tuple[str, ...]] = None
    reshape: Optional[Tuple[int, ...]] = None

    @classmethod
    def whole(cls, pop: str) -> "ViewRef":
        return cls(population=pop)

    @classmethod
    def merged(cls, ids: Sequence[str]) -> "ViewRef":
        return cls(population="merge[" + ",".join(ids) + "]", merge=tuple(ids))

@dataclass(frozen=True)
class ProjectionNode:
    id: str
    pre: ViewRef
    post: ViewRef
    rule: ConnRule
    synapse: ModelRef
    output: ModelRef
    plasticity: Optional[ModelRef] = None
    seed: Optional[int] = None             # if None, inherit from build-time seed

@dataclass(frozen=True)
class InputNode:
    id: str
    target: ViewRef
    source: ModelRef                       # "Poisson", "SpikeTimes", "DC", "Step",
                                           # "AC", "LayerImage", "DataStream"
    weight: Optional[Union[Any, DistRef, Trainable]] = None
    delay: Optional[Any] = None

@dataclass(frozen=True)
class ObservableNode:
    id: str
    target: ViewRef
    quantity: str                          # "spike", "V", "current", "weight", "rate"
    projection: Optional[str] = None       # only when quantity="weight"
    every: Optional[Any] = None            # downsample period; None = every step
    reducer: Optional[str] = None          # "mean", "sum", or None for full trace

@dataclass(frozen=True)
class SubNetworkNode:
    id: str
    inner: "NetIR"
    exports: Mapping[str, str]             # local_id -> exported handle name
    params: Mapping[str, Any] = field(default_factory=dict)

# ── Compound metadata (preserved for tooling) ────────────────────────────

@dataclass(frozen=True)
class SequentialMeta:
    name: str
    layer_ids: Tuple[str, ...]             # population ids in order
    proj_ids: Tuple[str, ...]              # inter-layer projection ids in order

@dataclass(frozen=True)
class GroupMeta:
    name: str
    members: Tuple[str, ...]               # population ids
    tags: Tuple[str, ...] = ()

@dataclass(frozen=True)
class CompoundMeta:
    sequentials: Tuple[SequentialMeta, ...] = ()
    groups: Tuple[GroupMeta, ...] = ()

# ── Root ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NetIR:
    version: str
    name: str
    populations: Tuple[PopulationNode, ...]
    projections: Tuple[ProjectionNode, ...]
    inputs:      Tuple[InputNode, ...]
    observables: Tuple[ObservableNode, ...]
    subnetworks: Tuple[SubNetworkNode, ...] = ()
    compounds:   CompoundMeta = field(default_factory=CompoundMeta)
    meta:        Mapping[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str: ...     # SHA-256 over canonical JSON
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "NetIR": ...

    # Post-definition modification (see §6.9). All return a NEW NetIR;
    # the original is not mutated.
    def update(self, path_or_mapping, value=_MISSING, /) -> "NetIR": ...
    def patch(self, *patches: "ParamPatch") -> "NetIR": ...
    def select(self, path: str) -> Any: ...     # read by dotted path
```

**Invariants enforced at construction**

1. `version == IR_VERSION`.
2. Every `ViewRef.population` resolves to a `PopulationNode.id`, a
   `SubNetworkNode.exports` entry, or a synthesized merge id whose members
   are all known populations.
3. Ids are unique within the enclosing `NetIR` and within each list.
4. `ModelRef.kind` is registered (§11).
5. `ConnRule.kind` is registered (§11.1).
6. Units on each parameter match the declared signature of `ModelRef.kind`.
7. `ConnRule.params['weight']` and `['delay']` (if present) are either
   dimensional `u.Quantity` of the right dimension, scalar-without-unit
   (for unitless models), a `DistRef` whose sample dimension matches, or
   a `Trainable` wrapping any of the above.
8. Merged-view members share neuron model kind and per-unit shape.
9. Sequential layer-shape inference (§6.7) passes.

Violations raise `SpecError` with a node id and a precise message (§14).

### 5.1 Connectivity, weights, and delays

We adopt `braintools.conn.Connectivity` as the canonical connectivity contract.
A rule is a strategy object whose `.generate(pre_size, post_size, ...)` returns
a `braintools.conn.ConnectionResult`:

```python
class ConnectionResult:
    pre_indices:  np.ndarray
    post_indices: np.ndarray
    pre_size:     int | tuple[int, ...] | None
    post_size:    int | tuple[int, ...] | None
    weights:      np.ndarray | u.Quantity | None
    delays:       np.ndarray | u.Quantity | None
    model_type:   str
    metadata:     dict[str, Any] | None
```

Consequences for the spec layer:

- **Single source of truth for per-edge attributes.** `weight`, `delay`,
  `allow_autapses` / `include_self_connections`, and other per-rule kwargs
  live inside `ConnRule.params`. `ProjectionNode` has no `weight` / `delay`
  / `allow_*` fields.
- **Frontend sugar.** `NetSpec.project(...)` accepts `weight=`, `delay=`,
  `allow_autapses=`, `allow_multapses=` as top-level kwargs. At
  `finalize()` they are merged into `ConnRule.params`. Conflicting
  declarations raise SPEC-016 / SPEC-017.
- **Distributions on weight/delay.** Any `DistRef` (or
  `braintools.init.Initialization` instance, accepted in B) becomes a
  concrete `Initialization` when the rule is materialized. Backends receive
  a concrete `weights` / `delays` array on `ConnectionResult`.
- **Autapses vs multapses.** `braintools.conn` uses
  `allow_self_connections` (`FixedProb`, `Random`) or
  `include_self_connections` (`AllToAll`). The spec accepts both plus the
  legacy `allow_autapses` alias; finalize canonicalizes to the rule's
  required name, raising SPEC-017 on conflicting values.
- **Seeding.** Rules expose a `seed` kwarg. If unset, the projection's
  `seed` (or, failing that, the build-time seed; §13) is folded in via
  `jax.random.fold_in(build_seed, projection_index)`.

---

## 6. Frontend A — Fluent `NetSpec` builder

Calls register node descriptions; nothing executes until `.finalize()`
returns a `NetIR`. Handles are typed symbolic references holding an id and
a back-pointer to the builder; they hold no JAX state.

### 6.1 `NetSpec` API

```python
class NetSpec:
    def __init__(self, name: str, *, meta: Mapping[str, Any] | None = None): ...

    # ── Building blocks ────────────────────────────────────────────────
    def population(
        self,
        name: str,
        model: ModelLike,
        size: int | Sequence[int],
        *,
        batch: int | None = None,
        init: Mapping[str, Any | DistLike | Trainable] | None = None,
        tags: Sequence[str] = (),
    ) -> PopulationHandle: ...

    def project(
        self,
        pre: PopulationHandle | ViewHandle,
        post: PopulationHandle | ViewHandle,
        *,
        rule: ConnRuleLike,                # braintools.conn.Connectivity or supplementary
        synapse: ModelLike,
        output: ModelLike,
        # sugar (merged into rule.params at finalize):
        weight: Any | DistLike | Trainable | None = None,
        delay: Any | DistLike | Trainable | None = None,
        allow_autapses: bool | None = None,
        allow_multapses: bool | None = None,
        # projection-level fields:
        plasticity: ModelLike | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> ProjectionHandle: ...

    def input(
        self,
        target: PopulationHandle | ViewHandle,
        source: InputSourceLike,
        *,
        weight: Any | DistLike | Trainable | None = None,
        delay: Any | DistLike | None = None,
        name: str | None = None,
    ) -> InputHandle: ...

    def observe(
        self,
        observable: ObservableLike,
        *,
        every: Any | None = None,
        reducer: str | None = None,
        name: str | None = None,
    ) -> ObservableHandle: ...

    def subnetwork(
        self,
        name: str,
        factory: Callable[..., "NetSpec"] | "NetSpec",
        **params: Any,
    ) -> SubNetworkHandle: ...

    def sequential(
        self,
        name: str,
        layers: Sequence["LayerLike"],
    ) -> SequentialHandle: ...

    def group(
        self,
        name: str,
        members: Sequence[PopulationHandle | ViewHandle],
        *,
        tags: Sequence[str] = (),
    ) -> GroupHandle: ...

    def export(self, **handles: PopulationHandle | ViewHandle) -> None: ...

    # ── Finalization & I/O ────────────────────────────────────────────
    def finalize(self) -> NetIR: ...
    def to_yaml(self, path: str | os.PathLike) -> None: ...
    def to_json(self, path: str | os.PathLike) -> None: ...

    @classmethod
    def from_ir(cls, ir: NetIR) -> "NetSpec": ...

    # ── Post-definition modification (see §6.9) ───────────────────────
    def update(self, path_or_mapping, value=_MISSING, /) -> "NetSpec": ...
    def with_(self, **subtrees: Any) -> "NetSpec": ...
    def patch(self, *patches: "ParamPatch") -> "NetSpec": ...
```

### 6.2 Handles

```python
class PopulationHandle:
    name: str
    size: int | tuple[int, ...]
    model: ModelRef

    # Slicing / indexing → ViewHandle
    def __getitem__(self, key: slice | Sequence[int]) -> "ViewHandle": ...
    def reshape(self, *shape: int) -> "ViewHandle": ...
    def concat(self, *others: "PopulationHandle | ViewHandle") -> "MergedViewHandle": ...

    # Observable factories
    @property
    def spikes(self) -> ObservableLike: ...
    @property
    def voltage(self) -> ObservableLike: ...
    @property
    def current(self) -> ObservableLike: ...
    @property
    def rate(self) -> ObservableLike: ...
    def state(self, name: str) -> ObservableLike: ...

class ViewHandle:                          # base for slice, indices, reshape views
    population: PopulationHandle
    # Same observable properties as PopulationHandle.

class MergedViewHandle(ViewHandle):
    members: tuple[PopulationHandle | ViewHandle, ...]

class ProjectionHandle:
    name: str
    pre: ViewHandle
    post: ViewHandle
    rule: ConnRule

    @property
    def weights(self) -> ObservableLike: ...

class SubNetworkHandle:
    name: str
    # exposes the inner spec's exported handles as attributes

class SequentialHandle:
    name: str
    layers: tuple[PopulationHandle, ...]
    @property
    def output(self) -> PopulationHandle: ...   # last layer's population

class GroupHandle:
    name: str
    members: tuple[PopulationHandle | ViewHandle, ...]
```

Module-level helpers mirror the most common handle methods:

```python
import brainpy.state.spec as sp

sp.merge(*handles) -> MergedViewHandle
sp.split(handle, sizes) -> tuple[ViewHandle, ...]
sp.concat(*handles, axis=0) -> MergedViewHandle
sp.train(value, *, name=None, constraint=None, required=False) -> Trainable
```

### 6.3 Example — Brunel network

```python
import brainpy.state.spec as sp
import braintools.conn as conn
import braintools.init as init
import saiunit as u

spec = sp.NetSpec("brunel")

neuron = sp.models.LIF(
    tau=20*u.ms, V_th=-50*u.mV, V_reset=-60*u.mV, V_rest=-65*u.mV,
)

exc = spec.population("exc", neuron, size=8000, tags=("excitatory",))
inh = spec.population("inh", neuron, size=2000, tags=("inhibitory",))

syn  = sp.models.Expon(tau=5*u.ms)
coba = sp.models.COBA(E=0*u.mV)

# Canonical: weight on the rule.
spec.project(exc, exc, synapse=syn, output=coba,
             rule=conn.FixedProb(prob=0.1, allow_self_connections=False,
                                 weight=0.10*u.nS))
# Sugar: weight as projection-level kwarg (merged at finalize).
spec.project(exc, inh, synapse=syn, output=coba,
             rule=conn.FixedProb(prob=0.1), weight=0.10*u.nS)
spec.project(inh, exc, synapse=syn, output=coba,
             rule=conn.FixedProb(prob=0.1), weight=-0.50*u.nS)
spec.project(inh, inh, synapse=syn, output=coba,
             rule=conn.FixedProb(prob=0.1, allow_self_connections=False),
             weight=-0.50*u.nS)

spec.input(exc, sp.input.Poisson(rate=20*u.Hz), weight=0.2*u.nS)
spec.observe(exc.spikes)
spec.observe(exc[:50].voltage)

ir  = spec.finalize()
sim = sp.backends.clock.build(ir, seed=0, dt=0.1*u.ms)
out = sim.run(1*u.second)
```

### 6.4 Subnetwork composition

```python
def column_spec(N: int, *, name: str) -> sp.NetSpec:
    s = sp.NetSpec(name)
    E = s.population("E", sp.models.LIF(...), size=int(0.8*N))
    I = s.population("I", sp.models.LIF(...), size=int(0.2*N))
    s.project(E, I, ...)
    s.project(I, E, ...)
    s.export(E=E, I=I)
    return s

net = sp.NetSpec("multi_column")
cols = [net.subnetwork(f"col_{k}", column_spec, N=1000) for k in range(4)]
for a, b in zip(cols, cols[1:]):
    net.project(a.E, b.E, rule=conn.FixedProb(prob=0.01, weight=...),
                synapse=..., output=...)
```

### 6.5 View algebra — slicing, merging, splitting, concatenating

```python
# Slicing
view = exc[:50]
view = exc[100:200]
view = exc[::2]
view = exc[[0, 1, 5, 42]]                  # explicit index set

# Reshape (only when size is a shape tuple, e.g. (C, H, W))
view = conv1.reshape(-1)

# Merging
all_neurons = sp.merge(exc, inh)
spec.project(all_neurons, readout,
             rule=conn.AllToAll(weight=...), synapse=..., output=...)

# Convenience alias on a handle:
all_neurons = exc.concat(inh)

# Splitting (inverse of merge)
e_part, i_part = sp.split(combined, sizes=[8000, 2000])

# Groups (organization / viz only; no semantic effect)
spec.group("recurrent_core", [exc, inh], tags=("balanced_eI",))
```

Properties of merged views:

- A `MergedViewHandle` is allowed as `pre`, `post`, `target`, or
  observable source.
- All members must share neuron model kind and per-unit shape
  (SPEC-019 on mismatch).
- Projections from a merged view materialize as one `ProjectionNode` per
  member, all sharing the same `synapse` / `output` / `plasticity` / `rule`
  template.
- Observing a merge concatenates underlying values at record time;
  `TraceBundle` returns a single array.

### 6.6 Trainable parameters and states

Any spec value — neuron parameter, synapse parameter, weight, delay,
initial state — can be marked **trainable**.

```python
# 1. Wrap a value with sp.train(...):
neuron = sp.models.LIF(
    tau=sp.train(20*u.ms, constraint="positive"),
    V_th=-50*u.mV, V_reset=-60*u.mV, V_rest=-65*u.mV,
)

# 2. Trainable weight tensor via rule.
rule = conn.FixedProb(
    prob=0.1,
    weight=sp.train(init.Normal(mean=0.1*u.nS, std=0.05*u.nS),
                     name="exc_to_inh.W"),
)

# 3. Learnable initial state.
exc = spec.population("exc", neuron, size=1024,
    init={"V": sp.train(init.Uniform(low=-65*u.mV, high=-55*u.mV))})

# 4. Sugar on spec.project:
spec.project(exc, inh, rule=conn.FixedProb(prob=0.1),
             synapse=..., output=..., weight=sp.train(0.1*u.nS))
```

**Mapping to `brainstate.nn.Param`.** At backend build, every `Trainable`
value materializes as a `brainstate.nn.Param`. The training-capable
backends use brainstate's parameter system uniformly:

```python
import brainstate as bs

class _MaterializedLIF(bs.nn.Module):
    def __init__(self, ir_model: ModelRef, ...):
        super().__init__()
        for name, value in ir_model.params.items():
            if isinstance(value, Trainable):
                setattr(self, name, bs.nn.Param(_init(value),
                                                name=value.name or name))
            else:
                setattr(self, name, _init(value))
```

| Backend  | Behavior for `Trainable`                                                                                    |
|----------|--------------------------------------------------------------------------------------------------------------|
| `clock`  | Treated as a constant initial value. Logs one `TrainableIgnored` notice per build. With `required=True`, raises SPEC-018. |
| `event`  | Same as `clock`.                                                                                             |
| `bptt`   | Becomes a `brainstate.nn.Param`. Collected via `Trainer.parameters()`.                                       |
| `eprop`  | Honors the trainable kinds supported by the algorithm (recurrent / output weights, optionally neuron params). Unsupported trainables raise `BackendCapabilityError` (SPEC-013). |
| `nir`    | Same as `clock`: trainables baked as constants at export time. Original `Trainable.name` recorded in the metadata sidecar (§9.4). |

**The `parameters()` view:**

```python
trainer = sp.backends.bptt.build(ir, seed=0, loss=loss_fn, dt=1*u.ms)
params  = trainer.parameters()
# {"exc_to_inh.W": ParamState(...), "exc.LIF.tau": ParamState(...), ...}

trainer.freeze("exc.LIF.tau")
trainer.unfreeze("exc.LIF.tau")
```

Parameter names follow a stable dotted convention:

```
<projection-id>.W                  # weight on a projection
<projection-id>.delay              # delay on a projection
<population-id>.<model-kind>.<p>   # neuron / synapse parameter
<population-id>.init.<state-var>   # learnable initial state
```

`Trainable.name`, when set, overrides this for that one value.

### 6.7 Deep SNNs — sequential composition and layer macros

```python
spec = sp.NetSpec("spiking_mnist", meta={"batch": 64})

stack = spec.sequential(
    "encoder",
    [
        sp.input.LayerImage(shape=(1, 28, 28)),
        sp.layer.Conv2d(out_channels=16, kernel=3,
                        neuron=sp.models.LIF(tau=10*u.ms,
                            V_th=-50*u.mV, V_reset=-60*u.mV, V_rest=-65*u.mV),
                        weight=sp.train(init.KaimingNormal())),
        sp.layer.MaxPool2d(kernel=2),
        sp.layer.Conv2d(out_channels=32, kernel=3,
                        neuron=sp.models.LIF(tau=10*u.ms,
                            V_th=-50*u.mV, V_reset=-60*u.mV, V_rest=-65*u.mV),
                        weight=sp.train(init.KaimingNormal())),
        sp.layer.Flatten(),
        sp.layer.Linear(out=10,
                        neuron=sp.models.LeakyRateReadout(),
                        weight=sp.train(init.XavierNormal())),
    ],
)

spec.observe(stack.output.rate)
ir = spec.finalize()
trainer = sp.backends.bptt.build(ir, seed=0, dt=1*u.ms,
                                  loss=sp.loss.cross_entropy)
```

`spec.sequential(name, layers)` returns a `SequentialHandle` whose
`.output` is the `PopulationHandle` of the last layer. Each entry is one of:

- A `LayerSpec` macro (Conv2d, Linear, MaxPool2d, AvgPool2d, Flatten,
  BatchNorm, Dropout, LeakyRateReadout, …).
- A `PopulationHandle` declared earlier — inserted verbatim.
- A bare callable returning a `LayerSpec` (late binding).

Stateless layers (Flatten, MaxPool2d, AvgPool2d, Dropout) materialize as
`ProjectionNode`s with a synapse of kind `"Identity"` and a parameterized
connectivity rule (`Pool2d`, `Reshape`, …) but no `Population` of their own.

**Recurrent connections.** A self-projection on a layer's output is the
canonical way to express recurrence:

```python
core = spec.sequential("rsnn", [
    sp.layer.Linear(out=512, neuron=sp.models.ALIF(...),
                    weight=sp.train(init.XavierNormal())),
])
spec.project(core.output, core.output,
             rule=conn.Random(prob=0.1,
                              weight=sp.train(init.Normal(std=0.05))),
             synapse=sp.models.Expon(tau=5*u.ms),
             output=sp.models.CUBA())
```

**Layer macro registry.**

| Macro                 | Connectivity rule used internally       | Stateful? |
|-----------------------|------------------------------------------|-----------|
| `Linear(out)`         | `braintools.conn.AllToAll`              | yes (neuron pop) |
| `Conv2d(...)`         | `braintools.conn.Conv2dKernel`          | yes              |
| `Conv1d(...)`         | supplementary `Conv1dKernel`            | yes              |
| `MaxPool2d(...)`      | supplementary `Pool2d(kind="max")`      | no               |
| `AvgPool2d(...)`      | supplementary `Pool2d(kind="avg")`      | no               |
| `Flatten()`           | supplementary `Reshape(target=-1)`      | no               |
| `BatchNorm()`         | supplementary `BatchNorm`               | yes (running stats) |
| `Dropout(p)`          | supplementary `Dropout(p)`              | no (rng state)   |
| `LeakyRateReadout(out)` | `AllToAll`, neuron=`LeakyRateReadout`  | yes              |

Third-party macros register via the `brainpy_state.spec.layers` entry point
(§11.5).

### 6.8 Construction-time errors

`NetSpec` raises eagerly on:

- Duplicate population / projection / observable name.
- Population name already used as a Python attribute on the builder.
- Sliced / indexed / reshaped view referencing a population not yet declared.
- `rule` not a `braintools.conn.Connectivity` instance (or registered
  supplementary rule).
- Pre and post sizes incompatible with `rule` (e.g. `OneToOne` and
  `n_pre != n_post`).
- `weight` / `delay` / autapse-flag set both as projection sugar and on the
  rule with conflicting values (SPEC-016 / SPEC-017).
- Unit dimension mismatch between `weight`, `synapse` input, and `output`
  expected dimensions.
- Merged view with incompatible member shapes / models (SPEC-019).
- Sequential layer shape mismatch (SPEC-020).
- `Trainable.required=True` on a non-trainable slot (SPEC-018).

Errors point at the offending Python source line.

### 6.9 Post-definition parameter modification (G12)

The spec is the source of truth for *what* a network is; users often need to
change *values* without rewriting the spec. There are two settings:

- **Pre-build (offline)** — modify a `NetSpec` or a `NetIR` before any
  backend has materialized it. The IR stays frozen; mutations return a
  new IR. Same content hash whenever the same edits are applied.
- **Post-build (live)** — modify parameter values on an already-built
  `Simulator` / `Trainer` without rebuilding. Trainers write to these
  during gradient descent; users can also read and write them imperatively.

Both settings share one **path language** and one **patch type**.

#### 6.9.1 Path language

A path is a dotted / indexed string that addresses any leaf in the IR.
Grammar:

```
path  = segment ("." segment)*
segment = name | name "[" index "]"
name   = identifier              # alphanumeric + underscore
index  = integer | string         # integer = list index; string = dict key
```

Examples:

```
populations.exc.size
populations.exc.model.tau
populations.exc.init.V
projections[0].rule.weight
projections[0].rule.prob
projections[2].synapse.tau
inputs[0].source.rate
sequentials.encoder.layers[1].weight
subnetworks.col_0.params.N
meta.author
```

`NetIR.select(path)` reads any addressable leaf; `NetIR.update(path, value)`
returns a new `NetIR` with that leaf replaced. Wildcards (`projections[*].rule.weight`)
are supported in `update` to broadcast a single value to many leaves.

#### 6.9.2 The `ParamPatch` type

A patch is a value-level description of a change. It is JSON-serializable
and round-trips through YAML/JSON, so the same patch can be applied to a
spec, an IR, or a running backend.

```python
@dataclass(frozen=True)
class ParamPatch:
    path: str                          # dotted path; wildcards allowed
    value: Any                         # scalar | u.Quantity | DistRef | Trainable | array
    op:    str = "set"                 # "set" | "scale" | "add" | "replace_with_trainable"
    label: Optional[str] = None        # free-form annotation for logs / sweeps
```

`op` lets a patch describe a transformation rather than a literal value:

- `"set"` — replace the value at `path`.
- `"scale"` — multiply (for numeric / Quantity leaves).
- `"add"` — additive shift (for numeric / Quantity leaves).
- `"replace_with_trainable"` — wrap a static value with `Trainable[...]`
  (or unwrap one if applied to a `Trainable`).

#### 6.9.3 Pre-build mutation — three equivalent forms

```python
# 1. Single-path .update(path, value):
spec2 = spec.update("populations.exc.model.tau", 25*u.ms)

# 2. Mapping .update({path: value, ...}):
spec2 = spec.update({
    "populations.exc.model.tau": 25*u.ms,
    "projections[0].rule.weight": 0.2*u.nS,
})

# 3. .with_() — Pythonic, builds the mapping for you (use when the path
#    naturally maps to nested kwargs):
spec2 = spec.with_(populations={"exc": {"model": {"tau": 25*u.ms}}})

# 4. .patch(*patches) — when you want richer operations than "set":
spec2 = spec.patch(
    sp.spec.ParamPatch("populations.exc.model.tau", 25*u.ms),
    sp.spec.ParamPatch("projections[*].rule.weight", 2.0, op="scale"),
    sp.spec.ParamPatch("populations.inh.model.tau", None,
                        op="replace_with_trainable"),
)
```

All four forms are immutable — `spec` is unchanged; `spec2` is a new
builder with the edits applied. The corresponding methods on `NetIR`
have identical semantics. YAML loader's `overrides=` kwarg (§7.4) is
exactly `NetIR.update(mapping)` after schema parsing.

#### 6.9.4 Post-build mutation — `ParameterView`

A built `Simulator` or `Trainer` exposes a `parameters` attribute of
type `ParameterView`. The view is the single read/write interface for
both static and dynamic parameters.

```python
class ParameterView(Protocol):
    # ── Read ────────────────────────────────────────────────────────
    def get(self, path: str) -> Any: ...
    def tree(self) -> Mapping[str, Any]: ...                  # full dict
    def trainable(self) -> Mapping[str, "brainstate.ParamState"]: ...
    def static(self) -> Mapping[str, Any]: ...

    # ── Write ───────────────────────────────────────────────────────
    def set(self, path: str, value: Any) -> None: ...
    def apply(self, *patches: ParamPatch) -> None: ...

    # ── Batching & rebuild policy ───────────────────────────────────
    def batch(self) -> "ContextManager[ParameterView]": ...
    def reset(self) -> None: ...                              # restore IR values
    def diff(self) -> tuple[ParamPatch, ...]: ...             # current vs IR
```

Usage:

```python
sim = sp.backends.clock.build(ir, seed=0, dt=0.1*u.ms)

sim.parameters.get("populations.exc.model.tau")            # 20 ms
sim.parameters.set("populations.exc.model.tau", 25*u.ms)   # live update
sim.parameters.apply(
    sp.spec.ParamPatch("projections[*].rule.weight", 1.5, op="scale"),
)

with sim.parameters.batch():
    sim.parameters.set("populations.exc.model.tau", 30*u.ms)
    sim.parameters.set("populations.inh.model.tau", 35*u.ms)
# any state rebuild happens once on context exit
```

For trainers, the same interface returns gradient-bearing
`brainstate.ParamState` instances when `path` resolves to a
`Trainable`; static parameters return plain values.

```python
trainer = sp.backends.bptt.build(ir, seed=0, loss=loss_fn, dt=1*u.ms)
W = trainer.parameters.get("projections[0].rule.weight")   # a ParamState
trainer.parameters.set("projections[0].rule.weight", new_W_array)
```

#### 6.9.5 Live vs rebuild changes

Not every parameter can be changed in-place. The backend classifies each
leaf as one of:

| Class       | Examples                                                                          | Behavior on `.set()`                              |
|-------------|-----------------------------------------------------------------------------------|---------------------------------------------------|
| `LIVE`      | Scalar model parameters (`tau`, `V_th`, `R`); per-edge weight / delay arrays; `Trainable` values. | Update the underlying state in place; cheap. |
| `LIVE_RESET`| Initial-state distributions (`init.V`); RNG-seeded leaves.                        | Update + reset the corresponding state variables; cheap. |
| `REBUILD`   | Population `size`; connectivity rule kind or hyperparameters (`prob`, `K`); synapse / output / plasticity `kind`; sequential layer membership; merge-view structure. | Raise `ParameterChangeRequiresRebuild(path)` (SPEC-024). User must edit the IR and call `backend.build(new_ir, ...)` again. |

The `ParameterView.set()` method consults the backend's parameter-class
table (declared next to its `BackendCapabilities`). A `REBUILD` change
never silently re-samples connectivity or reallocates state.

A convenience helper covers the common rebuild path:

```python
sim2 = sim.rebuild_with(spec.update("populations.exc.size", 16000))
# Equivalent to:
#   ir2  = sim.ir.update("populations.exc.size", 16000)
#   sim2 = sp.backends.clock.build(ir2, seed=sim.seed, dt=sim.dt)
# State is migrated where shapes are unchanged; reset otherwise.
```

#### 6.9.6 YAML / CLI ergonomics

Patches load from YAML through the same lexical conventions as the
spec (units, distributions, trainables):

```yaml
# brunel.patch.yaml
- { path: "populations.exc.model.tau",     value: "25 ms" }
- { path: "projections[*].rule.weight",    value: 1.5, op: scale }
- { path: "populations.inh.model.tau",     value: null,
    op: replace_with_trainable, label: "explore-trainable-tau" }
```

```sh
bp-spec patch brunel.netspec.yaml --from brunel.patch.yaml -o brunel-v2.yaml
bp-spec run   brunel.netspec.yaml --patch brunel.patch.yaml --backend clock --duration "1 s"
bp-spec build brunel.netspec.yaml --patch brunel.patch.yaml --backend nir -o brunel.nir
```

#### 6.9.7 Determinism and round-trip with patches

- Applying the same patch list to identical `(NetSpec, NetIR)` inputs
  produces identical content hashes (G4).
- Patches are commutative within `set` operations on disjoint paths;
  ordering matters when paths overlap or when `scale`/`add` mix with
  `set`. The IR records the applied patch list under
  `NetIR.meta["applied_patches"]` for archival.
- `ParameterView.diff()` returns the list of patches that would replay
  the current runtime state back from the original IR. This is the
  inverse of `apply` and is used by the CLI to print a "configuration
  diff" after a training run.

---

## 7. Frontend B — YAML/JSON data DSL

Spec is data. A YAML/JSON file is the canonical archival form; Python loads
it with `sp.spec.load(path) -> NetIR`. Same IR, same backends.

### 7.1 Top-level schema (informal)

```yaml
version: "netir/1.0"
name: "<spec-name>"

defaults:
  lif: &lif
    kind: LIF
    tau:     "20 ms"
    V_th:   "-50 mV"
    V_reset:"-60 mV"
    V_rest: "-65 mV"

populations:
  exc:
    model: *lif
    size: 8000
    init:
      V: { kind: Uniform, low: "-65 mV", high: "-55 mV" }
    tags: [excitatory]

projections:
  - { pre: exc, post: inh,
      rule: { kind: FixedProb, prob: 0.1, allow_self_connections: false,
              weight: "0.10 nS" },
      synapse: { kind: Expon, tau: "5 ms" },
      output:  { kind: COBA, E: "0 mV" } }

inputs:
  - { target: exc, source: { kind: Poisson, rate: "20 Hz" }, weight: "0.2 nS" }

observables:
  - { target: exc, quantity: spike }
  - { target: "exc[:50]", quantity: V }
  - { target: exc, quantity: V, every: "1 ms", reducer: mean }

subnetworks:
  column:
    !include "column.netspec.yaml"
    params: { N: 1000 }

sequentials:
  encoder:
    layers:
      - { kind: LayerImage, shape: [1, 28, 28] }
      - { kind: Conv2d, out_channels: 16, kernel: 3,
          neuron: *lif, weight: !train { kind: KaimingNormal } }
      # …

groups:
  recurrent_core:
    members: [exc, inh]
    tags: [balanced_eI]

meta:
  author: "Chaoming Wang"
  citation: "Brunel 2000"
```

### 7.2 Lexical conventions

- **Unit strings.** A quantity is `"<number><whitespace><unit>"`, where
  `<unit>` is anything `saiunit` parses (`mV`, `ms`, `nS`, `Hz`, `pA*ms`,
  …). Whitespace required. Negative numbers allowed (`"-50 mV"`).
- **References.** Bare strings name populations (`"exc"`). Bracketed strings
  name views: `"exc[:50]"`, `"exc[100:200]"`, `"exc[[0,1,5]]"`.
- **Merged views.** Object form `{ merge: [exc, inh] }` or string sugar
  `"exc | inh"`.
- **Connectivity rules.** A mapping with a PascalCase `kind` naming a
  `braintools.conn` class (`FixedProb`, `OneToOne`, `AllToAll`, `Random`,
  `Gaussian`, `Exponential`, `Ring`, `SmallWorld`, `ScaleFree`,
  `DistanceDependent`, `ExcitatoryInhibitory`, …) or a registered
  supplementary rule (`FixedIndegree`, `FixedOutdegree`, `FixedTotalNumber`,
  `PairwisePoisson`, `SymmetricPairwiseBernoulli`).
- **Distributions / initializers.** A mapping with a `kind` naming a
  `braintools.init` class (`Normal`, `LogNormal`, `Uniform`,
  `TruncatedNormal`, `Constant`, `KaimingNormal`, `XavierNormal`, …).
  Lower-case aliases accepted; canonicalized in the IR.
- **Trainables.** Object form `{ train: true, value: ..., constraint: ..., name: ... }`
  or shorthand tag `!train <value>`.
- **Includes.** `!include "<relative path>"` inlines another YAML mapping.
  Cycles are detected and rejected.
- **Anchors and aliases.** Standard YAML `&` / `*` is supported; resolved
  before schema validation.

### 7.3 JSON Schema

A full schema lives at `brainpy_state/spec/schema/netir-1.0.json`. Sketch:

```json
{
  "$id": "https://brainx.chaobrain.com/schema/netir-1.0.json",
  "type": "object",
  "required": ["version", "name", "populations"],
  "properties": {
    "version": { "const": "netir/1.0" },
    "name":    { "type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_-]*$" },
    "populations": {
      "type": "object",
      "additionalProperties": { "$ref": "#/$defs/Population" }
    },
    "projections":  { "type": "array", "items": { "$ref": "#/$defs/Projection" } },
    "inputs":       { "type": "array", "items": { "$ref": "#/$defs/Input" } },
    "observables":  { "type": "array", "items": { "$ref": "#/$defs/Observable" } },
    "sequentials":  { "type": "object" },
    "groups":       { "type": "object" }
  },
  "$defs": {
    "Quantity":     { "type": "string",
                      "pattern": "^-?\\d+(\\.\\d+)?([eE][+-]?\\d+)?\\s+[A-Za-z*/0-9]+$" },
    "Distribution": { "type": "object", "required": ["kind"] },
    "ViewRef":      { "oneOf": [
                        { "type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*(\\[[^\\]]+\\])?$" },
                        { "type": "object" }
                      ] },
    "Trainable":    { "type": "object", "required": ["train"] },
    "ModelRef":     { "type": "object", "required": ["kind"] }
  }
}
```

The schema is used by `bp-spec lint`, IDE integrations (YAML Language
Server via `yaml.schemas`), and the loader's pre-validation pass.

### 7.4 Parameter sweeps

Two supported patterns:

1. **Python overrides** — keep the YAML, override at load time:

   ```python
   for g in [4.0, 4.5, 5.0]:
       ir = sp.spec.load("brunel.netspec.yaml",
                         overrides={"projections[2].rule.weight": f"-{0.1*g} nS"})
       sim = sp.backends.clock.build(ir, seed=0, dt=0.1*u.ms)
   ```

2. **Sweep file** — a side file listing patches; the CLI expands the
   cartesian product:

   ```yaml
   # brunel.sweep.yaml
   base: brunel.netspec.yaml
   axes:
     g:    [4.0, 4.5, 5.0]
     seed: [0, 1, 2]
   patches:
     - path: "projections[2].rule.weight"
       value: "${-0.1 * g} nS"
   ```

   ```sh
   bp-spec sweep brunel.sweep.yaml --backend clock --out runs/
   ```

The patch language is intentionally minimal (dotted/indexed path + value).
Anything more complex stays in Python.

---

## 8. Backend protocol

Three backend families share the same registry plumbing but have distinct
contracts:

```python
# brainpy_state/spec/backend.py
from typing import Protocol, Mapping, Any, Iterable

class SimBackend(Protocol):
    name: str
    capabilities: "BackendCapabilities"

    def build(self, ir: NetIR, *,
              seed: int,
              dt: u.Quantity | None = None,
              **opts: Any) -> "Simulator": ...

class TrainBackend(Protocol):
    name: str
    capabilities: "BackendCapabilities"

    def build(self, ir: NetIR, *,
              seed: int,
              loss: Callable,
              dt: u.Quantity | None = None,
              **opts: Any) -> "Trainer": ...

class ExportBackend(Protocol):
    name: str
    capabilities: "BackendCapabilities"
    artifact_kind: str                     # "nir.NIRGraph", "onnx.ModelProto", ...

    def export(self, ir: NetIR, *,
               seed: int = 0,
               strict: bool = False,
               **opts: Any) -> "ExportResult": ...

class Simulator(Protocol):
    ir: NetIR
    seed: int
    parameters: "ParameterView"            # §6.9
    def run(self, duration: u.Quantity) -> "TraceBundle": ...
    def reset(self) -> None: ...
    def state(self) -> Mapping[str, Any]: ...
    def rebuild_with(self, new_ir: NetIR) -> "Simulator": ...

class Trainer(Protocol):
    ir: NetIR
    seed: int
    parameters: "ParameterView"            # §6.9; returns ParamState for Trainables
    def step(self, batch) -> "StepReport": ...
    def freeze(self, *paths: str) -> None: ...
    def unfreeze(self, *paths: str) -> None: ...
    def rebuild_with(self, new_ir: NetIR) -> "Trainer": ...

@dataclass(frozen=True)
class ExportResult:
    artifact: Any                          # backend-specific (e.g. nir.NIRGraph)
    notices: tuple["ExportNotice", ...]    # lossy/skipped mappings
    sidecar: Mapping[str, Any]             # round-trip metadata (units, names, seeds)
    content_hash: str

@dataclass(frozen=True)
class ExportNotice:
    code: str                              # "EXPORT-NIR-002", ...
    node_id: str
    message: str
    severity: str                          # "info" | "warning" | "error"
```

### 8.1 Third-party backends

Entry points group all three families:

```toml
[project.entry-points."brainpy_state.backends.sim"]
my_sim = "mypkg.backend:MySimBackend"

[project.entry-points."brainpy_state.backends.train"]
my_train = "mypkg.backend:MyTrainBackend"

[project.entry-points."brainpy_state.backends.export"]
my_export = "mypkg.backend:MyExportBackend"
```

`sp.backends.list(kind=None)` enumerates registered backends;
`sp.backends.get(name)` resolves one regardless of family.

### 8.2 Backend capabilities

Each backend declares a `capabilities` mapping. The loader validates the IR
against the chosen backend's capabilities and raises
`BackendCapabilityError` with the responsible node id when the IR uses a
feature the backend doesn't support.

```python
@dataclass(frozen=True)
class BackendCapabilities:
    supports_delay: bool
    supports_plasticity: bool
    supports_distributions: bool
    supports_nested_subnetworks: bool
    supports_training: bool                # for sim/export, always False
    supports_batch: bool
    supported_neuron_kinds: frozenset[str]
    supported_synapse_kinds: frozenset[str]
    supported_output_kinds: frozenset[str]
    supported_rules: frozenset[str]
    supported_layer_macros: frozenset[str]
    supported_input_kinds: frozenset[str]
```

Shipped backends:

| Family   | Backend     | Notes                                                                 |
|----------|-------------|-----------------------------------------------------------------------|
| sim      | `clock`     | Adapter to existing `_network.Network`/`Builder`.                     |
| sim      | `event`     | Event-driven simulator; depends on `brainevent`.                      |
| train    | `bptt`      | Autodiff through surrogate spikes; uses `brainstate.nn.Param`.        |
| train    | `eprop`     | Synaptic-eligibility-trace training; gradient-free recurrent updates. |
| train    | `event-prop`| Event-based exact gradients.                                          |
| export   | `nir`       | Neuromorphic IR (§9).                                                 |
| export   | `onnx-spike`| ONNX with the spiking extension ops (future, behind same protocol).   |
| export   | `nengo`     | Direct Nengo `Network` artifact (future).                             |

---

## 9. Export backends — Neuromorphic IR (G11)

### 9.1 Why an Export backend family

Sim and train backends produce trajectories or trained parameters; an
**export** backend produces an artifact in a foreign IR format suitable
for deployment to a third-party toolchain. The Neuromorphic IR (NIR) is
the lingua franca for spiking-network deployment across Loihi, SpiNNaker,
Nengo, Norse, Rockpool, Lava, Sinabs, and others — a single export path
into NIR transitively reaches all of them.

### 9.2 The export workflow

```python
import brainpy.state.spec as sp
import nir

ir = sp.spec.load("brunel.netspec.yaml")
result = sp.backends.nir.export(ir, seed=0, strict=False)

# result.artifact is a nir.NIRGraph
nir.write("brunel.nir", result.artifact)

# result.notices documents lossy mappings
for n in result.notices:
    print(f"[{n.severity}] {n.code} on {n.node_id}: {n.message}")

# result.sidecar carries metadata that NIR drops (units, seeds, trainable names)
import json
with open("brunel.nir.meta.json", "w") as f:
    json.dump(result.sidecar, f)
```

CLI:

```sh
bp-spec export brunel.netspec.yaml --backend nir --strict -o brunel.nir
```

### 9.3 Mapping: `NetIR` → NIR

NIR is a directed graph of typed nodes connected by edges. Each NIR node
type is a concrete dataclass (`nir.LIF`, `nir.Linear`, `nir.Conv2d`, …).
The exporter walks `NetIR` once and emits a `nir.NIRGraph`.

#### 9.3.1 Neuron model mapping

| `brainpy.state` neuron model                | NIR node                   | Notes                                                                                           |
|---------------------------------------------|----------------------------|-------------------------------------------------------------------------------------------------|
| `LIF(tau, V_th, V_rest, V_reset, R=1)`      | `nir.LIF(tau, r, v_leak, v_threshold)`   | `tau` in seconds, voltages in volts. `R` → `r`. `V_reset == V_rest` enforced or recorded.       |
| `IF(V_th, R=1)`                             | `nir.IF(r, v_threshold)`                 | Direct.                                                                                          |
| `LeakyRateReadout(tau, R=1)`                | `nir.LI(tau, r, v_leak)`                 | Rate-coded output, no spike threshold.                                                          |
| `LIF` + `Expon` synapse (CUBA)              | `nir.CubaLIF(tau_mem, tau_syn, r, v_leak, v_threshold, w_in)` | Synapse fused into the post-synaptic neuron. Detected at export time when a single inbound projection has `Expon` + `CUBA`. |
| `ALIF(tau, tau_adapt, ...)`                 | `nir.LIF` + custom adaptation node       | EXPORT-NIR-001: NIR has no canonical adaptive-threshold node; exported as a `nir.LIF` plus a custom-typed companion node. Strict mode raises. |
| `HH(...)`                                   | —                                        | EXPORT-NIR-002: no NIR equivalent. Strict mode raises; lenient mode skips with notice.          |
| `Izhikevich(...)`                           | —                                        | EXPORT-NIR-002 (same as HH).                                                                    |

#### 9.3.2 Connectivity / projection mapping

| `brainpy.state` shape                                     | NIR node                                              | Notes                                                  |
|-----------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------|
| `AllToAll(weight)` + `Identity` synapse + `CUBA`          | `nir.Linear(weight)`                                  | Sample, materialize, strip units.                      |
| Above + bias term on output                                | `nir.Affine(weight, bias)`                            |                                                         |
| `Conv2dKernel(weight, stride, padding, ...)`              | `nir.Conv2d(weight, stride, padding, dilation, groups, bias)` | Direct.                                                |
| `Conv1dKernel(...)`                                       | `nir.Conv1d(...)`                                     | Direct.                                                |
| `Pool2d(kind="avg", ...)`                                 | `nir.AvgPool2d(kernel_size, stride, padding)`         | Direct.                                                |
| `Pool2d(kind="sum", ...)`                                 | `nir.SumPool2d(...)`                                  | Direct.                                                |
| `Pool2d(kind="max", ...)`                                 | —                                                     | EXPORT-NIR-003: no NIR MaxPool. Lenient mode replaces with `AvgPool2d` and emits a warning notice. Strict mode raises. |
| `Reshape(target=-1)`                                      | `nir.Flatten(start_dim, end_dim, input_type)`         | Direct.                                                |
| `Delay` (per-projection or per-edge)                      | `nir.Delay(delay)` inserted as a pass-through node     | Per-edge delays require expansion into one Delay node per delay-value group (NIR delay is per-tensor).|
| Sparse `FixedProb` / `Random` / `FixedIndegree` / etc.    | `nir.Linear(weight=dense_W)` with zeros at non-edges  | Sparse rules densify at export. EXPORT-NIR-005 notice for matrices > 10⁷ entries to warn about size. |
| `STDP`, `STP`, other plasticity                           | —                                                     | EXPORT-NIR-004: NIR is inference-only at present. Stripped with notice in lenient mode; strict mode raises. |

#### 9.3.3 Input and output mapping

| `brainpy.state` node                            | NIR node                                                    | Notes                                                                                            |
|-------------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `InputNode(source=Poisson, rate=R)`             | `nir.Input(input_type=spike)` + sidecar metadata            | NIR does not model stochastic input sources; rate recorded in sidecar.                           |
| `InputNode(source=SpikeTimes, times=...)`       | `nir.brainx.SpikeTimes(times=...)` extension node           | Spike times are deployment-critical (Lava / Norse / Nengo need the table at build time, not only as sidecar). Emitted as a NIR extension node; the sidecar mirrors the table for consumers that ignore extensions. |
| `InputNode(source=LayerImage, shape=S)`         | `nir.Input(input_type=tensor with shape S)`                 | Direct.                                                                                          |
| `InputNode(source=DC, current=I)`               | `nir.Input(input_type=current)` + constant sidecar          |                                                                                                  |
| `ObservableNode(quantity=spike)`                | `nir.Output(output_type=spike)`                             | Connected from the source's output edge.                                                         |
| `ObservableNode(quantity=V)`                    | `nir.Output(output_type=voltage)`                           |                                                                                                  |
| `ObservableNode(quantity=rate)`                 | `nir.Output(output_type=rate)`                              | Used for `LeakyRateReadout` outputs.                                                             |
| `ObservableNode(quantity=weight, every=...)`    | —                                                           | EXPORT-NIR-006: weight tracing is a simulation concern, not deployable. Stripped with notice.    |

#### 9.3.4 Topology mapping

| `brainpy.state` topology         | NIR encoding                                                                                |
|----------------------------------|---------------------------------------------------------------------------------------------|
| `ProjectionNode(pre, post, ...)` | Edge `(pre_node, syn_or_neuron_node)` + `(syn_or_neuron_node, post_node)` as needed.        |
| Merged view `sp.merge(a, b)`     | Concat node: NIR currently has no native concat. The exporter emits a synthesized custom node `nir.brainx.Concat(axis=0)` under our reserved `nir.brainx.*` extension namespace. EXPORT-NIR-007 notice. |
| Recurrent self-projection         | Edge from post-neuron output back to its own input. NIR supports cycles.                    |
| `SubNetworkNode`                  | Inlined into the parent graph; the export preserves `id` namespacing.                       |
| `SequentialMeta` ordering         | The exporter walks layers in declared order; NIR edge list reflects the sequential chain.   |
| `GroupMeta`                       | Recorded in sidecar only; NIR has no group concept.                                         |

#### 9.3.5 Parameter and unit handling

NIR is unit-agnostic. Parameters are floats / tensors. The exporter:

1. **Strips units** by reducing every `u.Quantity` to its mantissa in a
   canonical SI base (`s` for time, `V` for voltage, `A` for current,
   `S` for conductance, `Hz` for rate).
2. **Records the original units** in `sidecar.units[<node_id>.<param>]`
   so a round-trip through the metadata sidecar (§9.4) restores them.
3. **Materializes distributions** by sampling them with the build-time
   seed before stripping units.
4. **Bakes `Trainable` values as constants**: the wrapper is unwrapped
   to its current value; `Trainable.name` is recorded in
   `sidecar.trainables`.

#### 9.3.6 The exporter algorithm (sketch)

```python
def to_nir(ir: NetIR, *, seed: int = 0, strict: bool = False) -> ExportResult:
    notices: list[ExportNotice] = []
    nodes: dict[str, nir.NIRNode] = {}
    edges: list[tuple[str, str]] = []
    sidecar: dict[str, Any] = {"units": {}, "trainables": {}, "seeds": {seed: True}}

    # 1. Inline subnetworks.
    flat = _inline_subnetworks(ir)

    # 2. Emit Population → neuron / readout NIR node.
    for pop in flat.populations:
        nodes[pop.id] = _emit_neuron(pop, sidecar, notices, strict)

    # 3. Emit Projection → Linear / Affine / Conv* / Pool* (+ optional Delay).
    for proj in flat.projections:
        result = _sample_connectivity(proj, seed)             # ConnectionResult
        edge_ids = _emit_projection(proj, result, nodes, edges, sidecar,
                                     notices, strict)

    # 4. Emit Input / Output nodes; wire to the corresponding internal nodes.
    for inp in flat.inputs:
        _emit_input(inp, nodes, edges, sidecar, notices)
    for obs in flat.observables:
        _emit_output(obs, nodes, edges, sidecar, notices)

    # 5. Fuse synapse + post-neuron into CubaLIF where the pattern matches.
    _fuse_cuba_lif(nodes, edges, notices)

    graph = nir.NIRGraph(nodes=nodes, edges=edges)
    return ExportResult(artifact=graph, notices=tuple(notices), sidecar=sidecar,
                         content_hash=_canonical_hash(graph, sidecar))
```

### 9.4 Lossy mappings, strict mode, and the metadata sidecar

NIR is intentionally a *minimum-common-denominator* IR. Some `NetIR`
constructs have no direct NIR equivalent (table in §9.3); the exporter
classifies each into one of:

| Class    | Meaning                                                                                  | Strict mode | Lenient mode                                |
|----------|------------------------------------------------------------------------------------------|-------------|---------------------------------------------|
| `LOSSLESS`| Direct one-to-one mapping. Sidecar restores units, names, seeds.                        | always      | always                                      |
| `RECORDED`| Mapping drops information NIR can't carry (units, names, seeds, stochastic params).     | always      | always (information lives in sidecar)       |
| `APPROXIMATE`| Mapping replaces the construct with the nearest NIR equivalent (MaxPool → AvgPool).   | **raises** EXPORT-NIR-003 | replaces with notice |
| `EXTENSION`| Emits a custom NIR-extension node (merged view → `nir.brainx.Concat`).                 | **raises** EXPORT-NIR-007 if `--no-extensions` | extension node emitted |
| `DROPPED` | Construct stripped (plasticity, weight observables).                                    | **raises** EXPORT-NIR-004 / EXPORT-NIR-006 | stripped with notice                       |
| `UNSUPPORTED`| No mapping exists for the node kind (HH, Izhikevich).                                | **raises** EXPORT-NIR-002 | node skipped, downstream edges rerouted; if rerouting is impossible, raises regardless |

Every transformation in classes `RECORDED`–`UNSUPPORTED` emits an
`ExportNotice` with a stable code (§14).

**The sidecar** is a plain-Python dict / JSON document recording:

- `units` — original `saiunit` units for every parameter that NIR strips.
- `trainables` — original `Trainable.name`, `constraint`, `required` flags.
- `seeds` — build-time and per-projection seeds used during connectivity sampling.
- `stochastic_inputs` — Poisson rates, spike-time tables, DC currents.
- `compounds` — the original `CompoundMeta` block (sequentials, groups).
- `tags` — population tags.
- `notices` — the same list returned in `ExportResult.notices`, for archival.

The sidecar is written next to the `.nir` file as `<name>.nir.meta.json`
by default. A loader (`sp.spec.import_.nir.load(nir_path, sidecar_path)`)
is provided to reconstruct as much of the original `NetIR` as is
recoverable — useful for round-trip testing but not for production use,
since `UNSUPPORTED` and `DROPPED` losses are not recoverable.

### 9.5 Other export targets

`nir` is the canonical export backend; the same `ExportBackend` protocol
hosts further targets:

- `onnx-spike` — ONNX with the spiking ops extension.
- `nengo` — direct `nengo.Network` artifact.
- `lava` — Intel Lava graph artifact (alternative to NIR-via-Loihi).

Each declares its own capability matrix and notice codes; they are not
required to ship with the spec library and can live in third-party
packages registered through the `brainpy_state.backends.export` entry
point.

---

## 10. Round-trip and equivalence

The two frontends are interchangeable:

```
NetSpec   ──finalize──►   NetIR   ──to_yaml──►   .netspec.yaml
   ▲                        │                       │
   │                     to_dict                    │
NetSpec.from_ir   ◄───── NetIR   ◄───── load ──────┘
```

**Equivalence law.** For any spec `s`:

```python
ir1 = s.finalize()
s.to_yaml("x.yaml")
ir2 = sp.spec.load("x.yaml")
assert ir1.content_hash() == ir2.content_hash()
```

The content hash is a SHA-256 over the IR rendered to its canonical JSON form:

- keys sorted lexicographically,
- floats formatted with `repr` (no trailing zeros),
- `u.Quantity` rendered as `{"_q": [mantissa, unit_str]}`,
- list order preserved (it is semantic for projections / observables),
- `Trainable`, `DistRef`, `ConnRule`, `ModelRef` rendered with `_t`,
  `_d`, `_c`, `_m` discriminators.

Content hash is used for: build cache keys, golden-IR test fixtures,
export determinism, and sweep deduplication.

---

## 11. Registry

Every model and rule is referenced by `kind` string. The registry maps
each `kind` to its Python implementation and a parameter signature
(names, units, defaults, trainability metadata).

### 11.1 Connectivity registry

The canonical source is **`braintools.conn`**. At import time, every
public subclass of `braintools.conn.Connectivity` is registered, keyed by
its PascalCase class name:

```python
import braintools.conn as _bt_conn

for _name in _bt_conn.__all__:
    _cls = getattr(_bt_conn, _name)
    if isinstance(_cls, type) and issubclass(_cls, _bt_conn.Connectivity):
        register_connectivity(_name, source=_cls)
```

Supplementary rules live in `brainpy_state/spec/connect/supplementary.py`
as `braintools.conn.PointConnectivity` subclasses, registered under the
same protocol. The legacy `brainpy_state._network._connectivity` module
is removed (§17).

| Supplementary rule              | Status                                        |
|---------------------------------|-----------------------------------------------|
| `FixedIndegree`                 | shipped here; upstream PR target: `braintools`|
| `FixedOutdegree`                | shipped here; upstream PR target: `braintools`|
| `FixedTotalNumber`              | shipped here; upstream PR target: `braintools`|
| `PairwisePoisson`               | shipped here; upstream PR target: `braintools`|
| `SymmetricPairwiseBernoulli`    | shipped here; upstream PR target: `braintools`|

`brainpy_state.spec.connect` re-exports the full registered set.

### 11.2 Initializer registry

Distributions and weight/delay initializers are sourced from
**`braintools.init`** with the same auto-registration mechanism. Every
`braintools.init.Initialization` subclass is keyed by PascalCase class name
(`Normal`, `LogNormal`, `Uniform`, `TruncatedNormal`, `Constant`,
`KaimingNormal`, `XavierNormal`, …). Lower-case aliases are accepted by
the YAML loader and canonicalized.

### 11.3 Neuron / synapse / output / input / plasticity registries

```python
@register_neuron("LIF", source="brainpy_state._brainpy.lif.LIF")
class LIFSignature:
    tau:     Annotated[u.ms, Trainability.OK]
    V_th:    Annotated[u.mV, Trainability.OK]
    V_reset: Annotated[u.mV, Trainability.OK]
    V_rest:  Annotated[u.mV, Trainability.OK]

@register_input("Poisson", source="brainpy_state._brainpy.inputs.PoissonSpike")
class PoissonSignature:
    rate: Annotated[u.Hz, Trainability.OK]   # learnable rate is allowed
```

Trainability annotations:

- `Trainability.OK` — `Trainable[...]` accepted.
- `Trainability.NEVER` — `Trainable[...]` raises SPEC-018.
- `Trainability.BACKEND` — accepted, but specific backends may reject
  (raises SPEC-021 on that backend).

### 11.4 Layer registry (for deep SNNs)

The v1 set is the table in §6.7. Each macro declares:

- `in_kind` — accepted view shape (`flat`, `2d`, `3d`).
- `out_kind` — produced view shape.
- Whether it materializes a Population (stateful) or only a Projection.

`spec.sequential(...)` checks `layer[k].out_kind == layer[k+1].in_kind` and
that numeric shapes broadcast; otherwise SPEC-020.

### 11.5 Third-party registration

```toml
[project.entry-points."brainpy_state.spec.neurons"]
my_neuron = "mypkg.models:MyNeuron"

[project.entry-points."brainpy_state.spec.connectivity"]
my_rule = "mypkg.conn:MyRule"

[project.entry-points."brainpy_state.spec.layers"]
my_layer = "mypkg.layers:MyLayer"

[project.entry-points."brainpy_state.spec.inputs"]
my_input = "mypkg.inputs:MyInputSource"

[project.entry-points."brainpy_state.spec.initializers"]
my_init = "mypkg.init:MyInit"
```

---

## 12. CLI tooling and visualization (G10)

### 12.1 `bp-spec` CLI

```
bp-spec lint     <path.yaml>                 # JSON Schema + IR validation
bp-spec describe <path.yaml>                 # counts + parameter summary (--json available)
bp-spec diff     <a.yaml> <b.yaml>           # structural diff at the IR level
bp-spec viz      <path.yaml> -o net.svg      # see §12.2
bp-spec build    <path.yaml> --backend NAME [--seed N] [--dt T] [--dry-run]
bp-spec run      <path.yaml> --backend clock --duration "1 s" --out runs/<hash>/
bp-spec sweep    <sweep.yaml> --backend clock --out runs/
bp-spec export   <path.yaml> --backend nir [--strict] [--seed N] -o brunel.nir
bp-spec patch    <path.yaml> --from patch.yaml -o new.yaml
bp-spec run      <path.yaml> --patch patch.yaml --backend clock --duration "1 s"
```

`build --dry-run` performs full IR construction + backend capability check
without running. `export` writes both the artifact and the sidecar (§9.4).

### 12.2 Visualization

Visualization reads the IR (not the runtime) so the same view is available
before any backend is built.

```sh
bp-spec viz <path>                                  \
    --mode    {graph,layers,matrix,params,nir}      \
    --renderer {graphviz,mermaid,matplotlib,html}   \
    --collapse-subnetworks                          \
    --color-by {tag,size,trainable,kind}            \
    -o net.svg
```

```python
import brainpy.state.spec as sp
ir = sp.spec.load("brunel.netspec.yaml")
sp.spec.viz(ir, mode="graph", renderer="graphviz", out="brunel.svg")
sp.spec.viz(ir, mode="layers", renderer="matplotlib")
fig = sp.spec.viz(ir, mode="matrix", return_figure=True)
sp.spec.viz(ir, mode="nir", out="brunel.nir.svg")    # post-export graph view
```

**Modes**

| Mode    | What it shows                                                                                                       | Best for                            |
|---------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| `graph` | Populations as nodes, projections as directed edges. Edge thickness ∝ #edges; edge color = sign. Inputs / observables drawn as squares / chevrons. Subnetworks rendered as collapsible clusters. | Sparse biophysical networks.        |
| `layers`| Vertical stack of layer macros (from `CompoundMeta.sequentials`). Each layer shows shape, neuron model, parameter count. Recurrent edges drawn as side loops. | Deep / neuromorphic SNNs.           |
| `matrix`| Block-structured connectivity matrix per projection. Dense layers as dot density. Conv/Pool layers as kernel previews. | Topology check.                     |
| `params`| Bar-chart of trainable vs frozen parameter counts per population / projection. Total parameter count for the whole IR. | Sanity-checking a deep model.       |
| `nir`   | The NIR graph that `sp.backends.nir.export(...)` would produce, with lossy mappings highlighted. | Verifying export shape pre-deployment. |

**Renderers**

| Renderer    | Output formats                  | Dependency                                |
|-------------|---------------------------------|-------------------------------------------|
| `graphviz`  | `.svg`, `.png`, `.pdf`, `.dot`  | `graphviz` (optional dep)                 |
| `mermaid`   | `.md`, `.mmd`                   | none                                       |
| `matplotlib`| `.png`, `.svg`, interactive     | `matplotlib`                              |
| `html`      | self-contained HTML (D3 + pan-zoom) | template ships in package              |

Default renderer is `graphviz` if installed, else `mermaid`. Visualization
is deterministic in `(ir, mode, renderer, seed)` (G4).

---

## 13. Determinism contract (G4)

Given a fixed `(NetIR, backend, seed, dt)`:

1. **Connectivity sampling** uses
   `jax.random.fold_in(jax.random.key(seed), proj_index)` per projection.
   A projection's own `seed` overrides this.
2. **Weight / delay distributions** use a derived key:
   `jax.random.fold_in(proj_key, _SUBKEY_WEIGHT)` with stable constants.
3. **Init-state distributions** use `fold_in(pop_key, _SUBKEY_INIT)`.
4. **Input sources** (e.g. Poisson) use `fold_in(input_key, step)`.
5. **Backends must not consume randomness outside the seed tree.**
6. **Visualization** (mode-dependent): `graph` / `layers` / `params` /
   `nir` are deterministic in `(ir, mode, renderer)` alone; `matrix`
   additionally takes `seed` (since it samples a `ConnectionResult`).
7. **Export** is deterministic in `(ir, seed, strict)`: same inputs ⇒
   identical NIR artifact bytes and identical sidecar. The default
   `seed` for export inherits the simulator's default seed
   (`sp.spec.DEFAULT_SEED`, currently `0`); it can be overridden via
   the `seed=` kwarg in Python or the `--seed N` flag on the CLI.
8. **Post-build mutation** (§6.9) is deterministic: applying the same
   `ParamPatch` list to identical `(NetSpec, NetIR)` inputs yields the
   same content hash; applying it to a built `Simulator` / `Trainer`
   yields the same in-memory parameter values.

Acceptance test: for each backend, two builds with identical
`(NetIR, backend, seed, dt)` produce bit-identical artifacts.

---

## 14. Validation rules catalog

Every error has a stable code for documentation cross-reference. Codes
are partitioned into spec-level (`SPEC-NNN`), backend-capability
(`SPEC-021`+), and per-export-backend (`EXPORT-<KIND>-NNN`).

### 14.1 Spec-level errors

| Code     | Tier        | Rule                                                                 |
|----------|-------------|----------------------------------------------------------------------|
| SPEC-001 | construction| Duplicate id in `populations` / `projections` / `observables`.       |
| SPEC-002 | construction| Reference to unknown population (`pre`, `post`, `target`).           |
| SPEC-003 | construction| Slice / index out of range for the referenced population.            |
| SPEC-004 | construction| `ModelRef.kind` not in registry.                                     |
| SPEC-005 | construction| Required parameter for `kind` is missing.                            |
| SPEC-006 | construction| Parameter has wrong unit dimension.                                  |
| SPEC-007 | construction| Distribution sample dimension does not match parameter dimension.    |
| SPEC-008 | finalize    | Connectivity rule precondition failed (delegated to `braintools.conn`). |
| SPEC-009 | finalize    | Connectivity rule rejected by constructor / `.generate()`.           |
| SPEC-010 | finalize    | `ConnRule.kind` not registered.                                      |
| SPEC-011 | finalize    | Subnetwork export name collides with parent population.              |
| SPEC-012 | backend     | Backend does not support `delay` in `ConnRule.params`.               |
| SPEC-013 | backend     | Backend does not support `plasticity` of this kind.                  |
| SPEC-014 | backend     | Instantaneous (zero-delay) recurrent cycle detected on event backend.|
| SPEC-015 | backend     | Backend rejects neuron / synapse / connectivity kind.                |
| SPEC-016 | construction| `weight` set both as projection sugar and on the rule with conflicting values. |
| SPEC-017 | construction| Conflicting alias and canonical kwarg on the same rule.              |
| SPEC-018 | construction| `Trainable` on a parameter slot annotated `Trainability.NEVER`.      |
| SPEC-019 | construction| Merged view with incompatible member shapes / neuron-model kinds.    |
| SPEC-020 | construction| Sequential layer-shape mismatch.                                     |
| SPEC-021 | backend     | Backend declares no training support but the IR contains `Trainable(required=True)`. |
| SPEC-022 | backend     | Backend rejects a layer macro kind.                                  |
| SPEC-023 | mutation    | `ParamPatch.path` does not resolve to a valid IR leaf (or wildcard matches nothing). |
| SPEC-024 | mutation    | `ParameterView.set(path, ...)` on a `REBUILD`-class leaf (§6.9.5). Raised as `ParameterChangeRequiresRebuild`. Hint to use `Simulator.rebuild_with(new_ir)`. |
| SPEC-025 | mutation    | `ParamPatch.op` not valid for the leaf type (e.g. `scale` on a categorical `kind` field). |

### 14.2 NIR export notices (`EXPORT-NIR-NNN`)

| Code            | Class       | Trigger                                                                              |
|-----------------|-------------|---------------------------------------------------------------------------------------|
| EXPORT-NIR-001  | APPROXIMATE | `ALIF` exported as `nir.LIF` + custom adaptation node.                                |
| EXPORT-NIR-002  | UNSUPPORTED | `HH`, `Izhikevich`, or other no-NIR-equivalent neuron model.                          |
| EXPORT-NIR-003  | APPROXIMATE | `MaxPool2d` → `AvgPool2d` in lenient mode.                                            |
| EXPORT-NIR-004  | DROPPED     | Plasticity (STDP / STP / …) stripped — NIR is inference-only.                         |
| EXPORT-NIR-005  | RECORDED    | Sparse rule densified to a large `nir.Linear` matrix (> 10⁷ entries).                |
| EXPORT-NIR-006  | DROPPED     | Weight observable stripped — not deployable.                                           |
| EXPORT-NIR-007  | EXTENSION   | Merged view emitted as custom `nir.brainx.Concat` extension node.                     |
| EXPORT-NIR-008  | RECORDED    | Physical units stripped; original units placed in sidecar.                            |
| EXPORT-NIR-009  | RECORDED    | `Trainable` baked as constant; original `Trainable.name` placed in sidecar.           |
| EXPORT-NIR-010  | RECORDED    | Stochastic input source's parameters placed in sidecar; NIR sees a placeholder Input. |

Strict mode (`--strict`) elevates `APPROXIMATE`, `EXTENSION`, `DROPPED`,
and `UNSUPPORTED` notices to errors.

---

## 15. Mapping to the existing codebase

```
brainpy_state/spec/                          (NEW)
├── __init__.py                              re-exports: NetSpec, load, NetIR, SpecError,
│                                            train, merge, split, concat, ParamPatch
├── ir.py                                    NetIR + all node dataclasses
├── netspec.py                               Frontend A: NetSpec + handles
├── yaml_loader.py                           Frontend B: load(), to_yaml()
├── schema/
│   └── netir-1.0.json                       JSON Schema
├── registry.py                              neuron / synapse / output / input / layer /
│                                            connectivity / initializer registries
├── connect/
│   ├── __init__.py                          re-exports: braintools.conn.* + supplementary
│   └── supplementary.py                     FixedIndegree, FixedOutdegree,
│                                            FixedTotalNumber, PairwisePoisson,
│                                            SymmetricPairwiseBernoulli
├── params.py                                ParamPatch, ParameterView (§6.9)
├── backend.py                               Protocols (SimBackend, TrainBackend,
│                                            ExportBackend), entry-point loader
├── viz/
│   ├── graph.py
│   ├── layers.py
│   ├── matrix.py
│   ├── params.py
│   └── nir.py
├── cli.py                                   bp-spec entry point
├── backends/
│   ├── clock.py                             Adapter over Network/Builder
│   ├── event.py                             Event-driven simulator
│   ├── bptt.py                              Autodiff training
│   ├── eprop.py                             E-prop training
│   ├── event_prop.py                        Event-prop training
│   └── nir.py                               NIR export (§9)
└── export_/
    ├── notices.py                           ExportNotice + code registry
    ├── sidecar.py                           Sidecar serialization
    └── nir_extensions.py                    nir.brainx.* namespace
                                             (SpikeTimes, Concat, …)

brainpy_state/_network/
├── _base.py                                 (unchanged) Network base class
├── _builder.py                              (unchanged + connect_from_result helper)
├── _projections.py                          rewritten as thin facade over
│                                            brainpy_state.spec.connect rules
├── _recorders.py                            (unchanged)
└── _connectivity.py                         REMOVED — rules moved to spec/connect/

    The _network module remains the runtime substrate of backends.clock.
    Public symbols stay importable; users may keep using `Builder` directly.
```

`backends.clock` does roughly:

```python
def build(ir: NetIR, *, seed: int, dt: u.Quantity, **_) -> Simulator:
    b = brainpy_state._network.Builder()
    for pop in ir.populations:
        b.add(pop.id, _instantiate_neuron(pop))

    for idx, proj in enumerate(ir.projections):
        pre_pop  = b._pop(proj.pre.population)
        post_pop = b._pop(proj.post.population)

        # 1. Materialize the braintools.conn rule from the IR.
        rule = _instantiate_rule(proj.rule, seed=_seed_for(seed, proj, idx))

        # 2. Sample edges + per-edge weights/delays.
        result = rule.generate(
            pre_size=_size(pre_pop, proj.pre),
            post_size=_size(post_pop, proj.post),
        )  # -> braintools.conn.ConnectionResult

        # 3. Wire synapse + output around the sampled edges.
        b.connect_from_result(
            pre_pop, post_pop, result=result,
            syn=_instantiate_syn(proj.synapse, post=post_pop),
            out=_instantiate_out(proj.output),
            plasticity=_instantiate_plasticity(proj.plasticity),
        )

    for inp in ir.inputs:
        _wire_input(b, inp)
    for obs in ir.observables:
        _wire_observable(b, obs)
    brainstate.nn.init_all_states(b)
    return _ClockSimulator(b, ir=ir, seed=seed, dt=dt)
```

`Builder.connect_from_result` is a new helper added by this work that
consumes a `ConnectionResult` directly — today's rule-specific `*Proj`
classes become thin facades over it.

---

## 16. Testing strategy

- **Unit** — every node dataclass: construction, repr, content hash
  stability, frozen-mutation rejection.
- **Frontend A** — every `NetSpec` method: success and the catalog of
  construction-time errors. Round-trip `B → IR → B` is identity on content
  hash.
- **Frontend B** — schema-positive examples (Brunel, COBA E/I,
  multi-area, spiking MLP, spiking CNN, RSNN), schema-negative examples
  (one per SPEC-NNN code), `!include` cycle detection.
- **Connectivity registry coverage** — parametrized test iterates every
  registered `braintools.conn` rule, builds a 2-population, 100-unit
  spec using it, calls `spec.finalize()` and `clock.build(...)`, and
  asserts the `ConnectionResult` is non-empty and has expected
  dtypes / units. Supplementary rules tested by the same parametrization.
- **View algebra** — slicing, indexing, merging, and reshape: each form
  constructed, lowered to IR, and asserted to address the right unit
  indices in a small (≤ 16-unit) toy network. Mixed-model merge raises
  SPEC-019; merged-view projection produces one `ProjectionNode` per
  member.
- **Trainable round-trip** — for each registered neuron / synapse, build
  a spec marking every trainable-capable parameter with `Trainable`,
  finalize, lower to `bptt` backend, assert `trainer.parameters()`
  contains all of them with the expected dotted names and that each is a
  `brainstate.nn.Param`. `Trainable` on a non-trainable slot raises
  SPEC-018.
- **Deep-SNN sequential** — golden test: a 3-layer spiking MLP and a
  2-layer spiking CNN built via `spec.sequential(...)` finalize to an IR
  whose `compounds.sequentials` recovers the layer order. Spiking MLP on
  MNIST trains under `bptt` to ≥ 90% test accuracy in CI.
- **Visualization determinism** — `sp.spec.viz(ir, mode=M, renderer=R, seed=S)`
  produces byte-identical output for `(M, R, S)` triples across Python
  versions in CI; golden artifacts checked in for one Mermaid and one
  Graphviz example.
- **Backend equivalence** — Brunel runs on `clock` and `event` with
  population firing rates within 2 σ over 1 s @ 10 trials.
- **NIR export** — every example in `docs/examples/` exports to NIR;
  the resulting `nir.NIRGraph` round-trips through `nir.write` →
  `nir.read` with byte equality on the file (artifact + sidecar). For
  examples with `APPROXIMATE` / `EXTENSION` / `DROPPED` / `UNSUPPORTED`
  notices, the strict-mode test asserts those codes are raised.
- **NIR-to-platform smoke** (optional, gated by env var) — when a
  NIR-consuming platform (e.g. `lava-nc`, `nengo`, `norse`) is
  available, load the exported `.nir` into that platform and verify it
  builds without error.
- **Capability mismatch** — every backend declares `capabilities`; tests
  assert that a known-unsupported feature on each backend raises
  `BackendCapabilityError` with the expected node id.
- **Parameter modification (G12)** — for each `op` in `ParamPatch`:
  applying it to a `NetSpec`, the resulting `NetIR`, and a built
  `Simulator`/`Trainer` all yield consistent state.
  `NetSpec.update(...) → finalize → content_hash` is order-independent
  for disjoint `set` paths and order-dependent (with documented order
  semantics) otherwise. `Simulator.parameters.diff()` round-trips
  through `apply(*diff())` to recover original values.
- **Live vs rebuild classification** — for each leaf class in §6.9.5,
  the corresponding backend reports the expected classification.
  `REBUILD` writes raise SPEC-024 with a path-accurate message;
  `LIVE`/`LIVE_RESET` writes propagate to a subsequent `sim.run(...)`.

Test file layout follows the existing convention: colocated `*_test.py`.

---

## 17. Relationship to the existing `_network` API

The new `brainpy_state.spec` module ships alongside today's
`brainpy_state._network.Builder` and the rule-based `*Proj` classes,
with two deliberate changes to the existing tree:

- **`brainpy_state/_network/_connectivity.py` is removed.** The samplers it
  provides (`sample_one_to_one`, `sample_pairwise_bernoulli`,
  `sample_fixed_indegree`, …) are duplicates of `braintools.conn` rules or
  candidates for upstreaming. After this change:
  - Rules already in `braintools.conn` (`OneToOne`, `AllToAll`,
    `FixedProb`, `Random`, `Gaussian`, …) are used directly.
  - Rules not yet in `braintools.conn` (`FixedIndegree`, `FixedOutdegree`,
    `FixedTotalNumber`, `PairwisePoisson`, `SymmetricPairwiseBernoulli`)
    move to `brainpy_state/spec/connect/supplementary.py` as
    `braintools.conn.PointConnectivity` subclasses. Upstreaming them
    eventually turns this file into a thin import-shim. The old
    `brainpy_state._network._connectivity` import path is dropped — code
    that referenced it must now import from `brainpy_state.spec.connect`
    or `braintools.conn`.
- **`brainpy_state/_network/_projections.py` is rewritten** as a thin
  facade over `braintools.conn` rules + `Builder.connect_from_result`.
  Public class signatures (`OneToOneProj`, `FixedIndegreeProj`,
  `PairwiseBernoulliProj`, …) are preserved; their internals delegate to
  the canonical rule of the same shape.

What stays:

- `Builder` keeps working unchanged — it is now the substrate of
  `backends.clock.build()`. User code that imports `Builder` directly
  needs no changes.
- Documentation and examples migrate to `NetSpec` as the recommended
  entry point.
- The `brainpy.state` top-level namespace gains `spec`, `NetSpec`,
  `load`, `train`, `merge`, `split`, `concat`, `backends`, `viz`, and
  `ParamPatch`. Existing symbols (`LIF`, `Expon`, `COBA`, `Builder`,
  `OneToOneProj`, `FixedIndegreeProj`, …) keep their current paths.

---

## 18. Decision log

| ID  | Decision                                                          | Resolution |
|-----|-------------------------------------------------------------------|-----------|
| D1  | `dt` placement                                                    | Backend.build kwarg. Pinning in the spec leaks runtime choice into G1; event backends ignore it. |
| D2  | `seed` placement                                                  | Backend.build kwarg, with per-projection `seed` override allowed in the IR for reproducibility of partial sub-graphs. |
| D3  | Module path                                                       | `brainpy_state.spec`, exported as `brainpy.state.spec`. Coexists with `brainpy.state.Builder`. |
| D4  | CLI name                                                          | `bp-spec`. |
| D5  | Cross-ref style                                                   | IR uses strings (population ids, dotted paths); B uses handles; D uses string ids. |
| D6  | Custom user models                                                | Registry decorators + entry-point groups (§11.5). |
| D7  | YAML subnetwork parameterization                                  | `!include` + an explicit `params:` map per instance. Templating (Jinja / Hydra) is opt-in via `bp-spec sweep`. |
| D8  | Mutable parameters for training                                   | IR stays frozen. Trainers expose a `parameters()` view; updates happen in trainer state, not in the IR. |
| D9  | View granularity                                                  | Slice / index / merge / reshape are fields on `ViewRef`. No separate `View` node type. Merge views denormalize into one `ProjectionNode` per member at finalize. |
| D10 | Connectivity source library                                       | `braintools.conn` is canonical. Auto-registers every `Connectivity` subclass; weight/delay live on the rule. Supplementary rules ship from `brainpy_state` and are tracked for upstreaming. |
| D11 | Initializer source library                                        | `braintools.init` is canonical. Auto-registers every `Initialization` subclass; `DistRef` lowers to a concrete `Initialization` at backend build. |
| D12 | Weight/delay precedence                                           | Canonical home is `ConnRule.params`. Projection-level `weight=` / `delay=` are sugar merged at finalize. Conflicts raise SPEC-016. |
| D13 | Trainable surface                                                 | `Trainable` is a value-level wrapper, applicable to any leaf in `ModelRef.params`, `ConnRule.params`, `PopulationNode.init`, and `InputNode.weight` / `InputNode.source.params`. Trainability metadata lives in registry signatures. |
| D14 | Trainable storage                                                 | Every `Trainable` materializes as `brainstate.nn.Param` on the synthesized `brainstate.nn.Module`. Trainers collect via `state.tree_states(brainstate.ParamState)`. |
| D15 | Layer macros                                                      | Ship the §6.7 set covering deep-SNN essentials. Third-party macros via entry points. |
| D16 | Visualization default renderer                                    | Mermaid (no runtime deps) when Graphviz is not installed; Graphviz when available. HTML renderer for interactive `bp-spec viz --renderer html`. |
| D17 | NIR as the canonical export target                                | Yes. Other export targets (`onnx-spike`, `nengo`, `lava`) implement the same `ExportBackend` protocol but are not required to ship with the spec library. |
| D18 | Lossy-export policy                                               | Six-class taxonomy (§9.4). Strict mode is opt-in (`--strict`) and elevates classes `APPROXIMATE`, `EXTENSION`, `DROPPED`, `UNSUPPORTED` to errors. Lenient mode is default and ships notices. |
| D19 | NIR units                                                         | NIR is unit-agnostic. The exporter strips to canonical SI and writes a sidecar (`<name>.nir.meta.json`) preserving the original units and trainable / seed / compound metadata. |
| D20 | NIR import                                                        | Not in scope. The sidecar enables partial reconstruction for round-trip testing only; production round-trip is not supported because `UNSUPPORTED` and `DROPPED` losses are unrecoverable. |
| D21 | Loss / optimizer integration                                      | **User-side.** Losses and optimizers are plain Python callables passed to `bptt.build(..., loss=...)`. No `brainpy.state.spec.loss` / `…optim` modules ship with the spec. |
| D22 | NIR export default seed                                           | Defaults to the simulator's default seed (`sp.spec.DEFAULT_SEED`, currently `0`). Overridable in Python via `export(..., seed=N)` and on the CLI via `bp-spec export --seed N`. |
| D23 | Reverse-compatibility shim for `_network/_connectivity.py`        | **Removed.** Supplementary rules move to `brainpy_state/spec/connect/supplementary.py`. The legacy import path is dropped. |
| D24 | Spike-time input encoding in NIR                                  | Emitted as a NIR extension node `nir.brainx.SpikeTimes(times=...)` under our reserved `nir.brainx.*` namespace. The sidecar mirrors the table for consumers that ignore extensions. |
| D25 | Quantized weights for hardware                                    | **Per-export-backend concern.** The core spec does not carry a `quantize` flag. Each export backend (e.g. a future Loihi-targeted exporter) may consume `Trainable.constraint` strings (e.g. `"int8"`, `"fixed:Q4.4"`) or apply its own quantization config. |
| D26 | Post-definition parameter modification — interface               | One path language (§6.9.1) and one `ParamPatch` type (§6.9.2). Pre-build: `NetSpec.update` / `.with_` / `.patch` (immutable; new spec returned). Post-build: `Simulator.parameters` / `Trainer.parameters` of type `ParameterView` (mutating; in-place on the runtime). |
| D27 | Live vs rebuild policy                                            | Three-class taxonomy `LIVE` / `LIVE_RESET` / `REBUILD` (§6.9.5). Live writes propagate in place; rebuild writes raise SPEC-024 with a hint to call `Simulator.rebuild_with(new_ir)`. Connectivity is never silently re-sampled. |
| D28 | Novelty positioning                                               | The load-bearing novelty is **training-paradigm pluralism over a single IR** (BPTT, event-prop, RTRL/forward-mode, eligibility-trace), not the DSL surface (which intentionally inherits from PyNN/NESTML/Brian2/Nengo) and not NIR export (an adopted community standard). See §1.1. Scope ties are broken in favor of preserving spec neutrality across the four training paradigms. |

---

## 19. Cheat sheet — Python ↔ YAML

| Python (B)                                                                                             | YAML (D)                                                                                                          |
|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `spec.population("exc", LIF(tau=20*u.ms), size=8000)`                                                  | `populations: { exc: { model: { kind: LIF, tau: "20 ms" }, size: 8000 } }`                                        |
| `spec.project(exc, inh, rule=conn.FixedProb(prob=0.1, weight=0.1*u.nS), synapse=..., output=...)`      | `{ pre: exc, post: inh, rule: { kind: FixedProb, prob: 0.1, weight: "0.10 nS" }, synapse: ..., output: ... }`     |
| `spec.project(exc, inh, rule=conn.FixedProb(prob=0.1), weight=0.1*u.nS, ...)` *(sugar)*                | `{ pre: exc, post: inh, rule: { kind: FixedProb, prob: 0.1 }, weight: "0.10 nS", synapse: ..., output: ... }`     |
| `spec.input(exc, Poisson(rate=20*u.Hz), weight=0.2*u.nS)`                                              | `{ target: exc, source: { kind: Poisson, rate: "20 Hz" }, weight: "0.2 nS" }`                                     |
| `spec.observe(exc.spikes)`                                                                             | `{ target: exc, quantity: spike }`                                                                                |
| `spec.observe(exc[:50].voltage, every=1*u.ms, reducer="mean")`                                         | `{ target: "exc[:50]", quantity: V, every: "1 ms", reducer: mean }`                                               |
| `rule=conn.FixedProb(prob=0.1, weight=init.LogNormal(mean=0.1*u.nS, std=0.05*u.nS))`                   | `rule: { kind: FixedProb, prob: 0.1, weight: { kind: LogNormal, mean: "0.1 nS", std: "0.05 nS" } }`               |
| `cols = [spec.subnetwork(f"col_{k}", column_spec, N=1000) for k in range(4)]`                          | `subnetworks: { col_0: { !include "column.netspec.yaml", params: { N: 1000 } }, col_1: {...}, ... }`              |
| `all_neurons = sp.merge(exc, inh)` *(merged view)*                                                     | `target: { merge: [exc, inh] }`   *or*   `target: "exc \| inh"`                                                   |
| `view = exc[[0, 1, 5, 42]]`                                                                            | `target: "exc[[0,1,5,42]]"`                                                                                       |
| `view = conv1.reshape(-1)`                                                                             | `target: { population: conv1, reshape: [-1] }`                                                                    |
| `sp.train(20*u.ms, constraint="positive")`                                                             | `{ train: true, value: "20 ms", constraint: positive }`   *or*   `!train "20 ms"`                                 |
| `sp.train(init.XavierNormal(), name="W")`                                                              | `!train { kind: XavierNormal }`   *or*   `{ train: true, init: { kind: XavierNormal }, name: "W" }`               |
| `spec.sequential("enc", [sp.layer.Conv2d(...), sp.layer.MaxPool2d(2), ...])`                           | `sequentials: { enc: { layers: [ { kind: Conv2d, ... }, { kind: MaxPool2d, kernel: 2 }, ... ] } }`                |
| `sp.layer.Linear(out=10, neuron=sp.models.LeakyRateReadout(), weight=sp.train(init.XavierNormal()))`   | `{ kind: Linear, out: 10, neuron: { kind: LeakyRateReadout }, weight: !train { kind: XavierNormal } }`            |
| `sp.spec.viz(ir, mode="layers", renderer="mermaid", out="net.md")`                                     | (CLI) `bp-spec viz path.yaml --mode layers --renderer mermaid -o net.md`                                          |
| `sp.backends.nir.export(ir, seed=0, strict=False)`                                                     | (CLI) `bp-spec export path.yaml --backend nir -o net.nir`                                                         |
| `spec2 = spec.update("populations.exc.model.tau", 25*u.ms)`                                            | (CLI) `bp-spec patch path.yaml --from patches.yaml -o new.yaml`                                                   |
| `spec.patch(ParamPatch("projections[*].rule.weight", 1.5, op="scale"))`                                | `patches: - { path: "projections[*].rule.weight", value: 1.5, op: scale }`                                        |
| `sim.parameters.get("populations.exc.model.tau")`   *(read live)*                                       | n/a (runtime)                                                                                                     |
| `sim.parameters.set("populations.exc.model.tau", 25*u.ms)`   *(write live)*                             | n/a (runtime)                                                                                                     |
| `sim2 = sim.rebuild_with(spec.update("populations.exc.size", 16000))`                                  | n/a (runtime)                                                                                                     |
| `trainer.parameters.get("projections[0].rule.weight")` *(returns ParamState)*                          | n/a (runtime)                                                                                                     |

---

## 20. Open questions

- **NIR extension namespace coordination.** We claim a `nir.brainx.*`
  namespace for our custom extension nodes (`SpikeTimes`, `Concat`, and
  potentially an `ALIF`-adaptation node). Coordinate with the NIR
  maintainers before publishing so the namespace is reserved upstream
  and the schema is documented in NIR's own extension registry.
- **Patch order semantics for overlapping paths.** When two
  `ParamPatch` entries target overlapping wildcards or mix `set` and
  `scale`/`add`, the documented order is "last patch wins for `set`;
  cumulative for `scale`/`add`." Confirm this matches user expectations
  on the first sweep example that exercises overlap.
- **Patch portability across spec edits.** A patch stored in YAML
  references paths by id. If the underlying spec renames a population,
  every stored patch needs to be migrated. Should we ship a
  `bp-spec patch migrate <old.yaml> <new.yaml>` helper, or treat patches
  as session-local artifacts that don't survive schema renames?
