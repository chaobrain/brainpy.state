# Chapter 5 — Domain extensions: applying the DSL to new domains

> Part of the [Network Specification DSL RFC](./README.md).

The previous chapters describe **one application** of the spec
substrate: point-neuron SNN. This chapter formalizes the substrate
itself — the protocols a node kind, a view handle, a builder verb, a
codec, and a backend conform to — so that other domains can apply the
same language to their own primitives.

Two concrete domains designed against here:

- **`braincell`** ([repo](https://github.com/chaobrain/braincell)) —
  biophysically detailed cells: morphology trees, ion channels,
  paint / place mechanisms, compartmental views, compartment-targeted
  projections.
- **`brainmass`** ([repo](https://github.com/chaobrain/brainmass)) —
  neural-mass / whole-brain modeling: rate and oscillator dynamics
  per parcel, structural-connectivity coupling, forward-model
  observables (BOLD / EEG / MEG).

Neither will ship code under `brainpy_state/`. This chapter defines
the **protocols** they implement and the **interfaces** their
extension modules expose. Both worked designs include the full code
each repository would publish.

## Section map

| §     | Topic                                          | Purpose                                                                                       |
|-------|------------------------------------------------|-----------------------------------------------------------------------------------------------|
| 10.1  | The DSL is a substrate, not a schema           | Reframe Chapters 2 – 9 as the SNN instance of a more general substrate.                       |
| 10.2  | The five extension protocols                   | The minimum surface a domain implements.                                                      |
| 10.3  | `IRNode` — defining a node kind                | Frozen-dataclass contract, content-hash, units, validation.                                   |
| 10.4  | `ViewHandle` — sub-population references       | How a domain defines a new way to address a slice of a node.                                  |
| 10.5  | Builder verbs — extending `NetSpec`            | The decorator that adds methods to the frontend.                                              |
| 10.6  | Codec — round-trip and content hash            | JSON canonicalization for domain-defined nodes and handles.                                   |
| 10.7  | Backend dispatch — handlers per node kind      | How a backend opts in to support a domain's node kinds.                                       |
| 10.8  | Discovery, activation, conflict                | Entry points; import-time registration; name collisions.                                      |
| 10.9  | Worked extension — `braincell`                 | Full code: IR nodes, view handles, builder verbs, backend dispatch, user-facing example.      |
| 10.10 | Worked extension — `brainmass`                 | Full code: IR nodes, builder verbs, dedicated backend, user-facing example.                   |
| 10.11 | Constraints and invariants                     | What extensions may not do.                                                                   |
| 10.12 | Decision log additions                         | D25 – D29.                                                                                    |

---

## 5.1 The DSL is a substrate, not a schema

Chapter 2 defines a `NetIR` with five typed node tuples
(`populations`, `projections`, `inputs`, `observables`,
`subnetworks`) and a fixed set of value wrappers (`Trainable`,
`DistRef`, `ModelRef`, `ConnRule`, `VariableRef`). That description is
**complete** for the point-neuron SNN domain — the one the rest of
the RFC is built around. It is **not** a closed description of the
DSL itself.

### 5.1.1 What the spec describes — and what it does not

The specification language describes **the model**: dynamics,
topology, parameters, initial conditions, connectivity, inputs,
observables. It does **not** describe **how to compute it**:
numerical integrator, time-step discretization, compartmentalization
policy, ring-buffer sizes, accelerator placement. Those are
backend-side realization choices, and they vary per backend even for
the same spec.

The boundary is load-bearing for the four-paradigm pitch (§1.1.1):
the same `NetIR` running through `clock`, `event`, `bptt`, `eprop`,
and `eventprop` produces five mathematically distinct artifacts
precisely because the spec does not commit to a numerical scheme.
Domain extensions must respect the same line. Concretely:

| In the spec (IR fields)                                 | In the backend (`build()` kwargs)                        |
|---------------------------------------------------------|-----------------------------------------------------------|
| Model kind and its parameters (with `saiunit` units)    | Solver / integrator (`"dopri5"`, `"staggered"`, …)        |
| Topology (`size`, connectivity rules, coupling matrix)  | Time step `dt`, jitter, tolerance settings                |
| Dynamics terms (noise model, stochastic forcing)        | Number of compartments / `cv_policy`                       |
| Initial conditions / state init distributions           | Delay-buffer sizes, ring-buffer policy, memory layout     |
| Spike threshold, reset rule, refractoriness (as model)  | Adaptive-step controllers, error tolerances               |
| Inputs and what to observe                              | Sampling resolution of recorders (downsample is spec; how it is stored is backend) |

Numerical knobs reach the backend through `backend.build(...,
node_options=...)` (per-node by id) and the backend's own top-level
kwargs (defaults applicable to every node). The IR carries no such
knob; two builds of the same IR with different `node_options` produce
two artifacts with the same `content_hash` but different runtime
behaviour. That separation is what makes the spec a portable
description of the model rather than of a particular integration.

The worked extensions in §5.9 and §5.10 obey this rule strictly:
neither `MorphPopulation` nor `MassPopulation` declares a solver,
a discretization policy, or a delay-buffer size. Those are passed
to `clock.build(...)` and `brainmass.build(...)` respectively.

The DSL itself is the set of mechanisms that make the SNN nodes work:

- A **frozen-dataclass IR** with a stable canonicalization rule
  (Chapter 2 §2.1).
- **Content-hash determinism** (G4, Chapter 6 §6.2).
- A **typed handle system** for symbolic references during building
  (Chapter 3 §3.3).
- A **registry** that maps `kind` strings to implementations
  (Chapter 7).
- A **builder** (`NetSpec`) that emits node values, not modules
  (Chapter 3 §3.1).
- A **backend protocol** that consumes the IR (Chapter 6 §6.1).
- **Variables** for build-time parameter binding (Chapter 3 §3.14).

Every one of these mechanisms is **node-kind-generic**: a frozen
dataclass with units and a content-hash codec is a frozen dataclass
with units and a content-hash codec, whether it carries one spike
trace or a forward-model output. The fact that `PopulationNode`,
`ProjectionNode`, and so on are special-cased in `NetIR` is an
ergonomic choice for the SNN domain, not a property of the substrate.

This chapter promotes the substrate to a first-class extension
surface. A domain extension is a Python package that:

1. Defines new **IR node kinds** as frozen dataclasses conforming to
   the `IRNode` protocol.
2. Optionally defines new **view-handle subtypes** for sub-references
   to those nodes.
3. Registers new **builder verbs** on `NetSpec` so the frontend is
   domain-flavored.
4. Provides **codecs** so its nodes and handles round-trip and
   participate in the content hash.
5. Either ships a **backend** that handles its node kinds, or asks
   an existing backend to register a **dispatch handler** for them.

The SNN node kinds (`PopulationNode`, `ProjectionNode`,
`InputNode`, `ObservableNode`, `SubNetworkNode`) are themselves
applications of this protocol — pre-registered by
`brainpy_state.spec` itself. There is no special-casing required for
domain extensions to look and feel as first-class as the built-ins.

---

## 5.2 The five extension protocols

```
┌──────────────────────────────────────────────────────────────────────────┐
│  brainpy.state.spec — the substrate                                      │
│                                                                          │
│   IRNode      ViewHandle    builder verb     Codec     Backend dispatch  │
│   ─────       ──────────    ────────────     ─────     ────────────────  │
│   protocol    protocol      decorator        protocol  protocol          │
└──────┬─────────────┬──────────────┬──────────────┬──────────────┬────────┘
       │             │              │              │              │
       ▼             ▼              ▼              ▼              ▼
   PopulationNode  ViewHandle    net.population   built-in     clock/event/
   ProjectionNode  +slice/index  net.project      JSON codec   bptt/eprop
   InputNode       /merge/...    net.input        for built-   dispatch on
   ObservableNode                net.observe      ins          built-in
   SubNetworkNode                ...                           kinds
       △             △              △              △              △
       │             │              │              │              │
       └─────────────┴──────────────┴──────────────┴──────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
         braincell.dsl                       brainmass.dsl
         MorphPopulation                     MassPopulation
         CompartmentProjection               CouplingMatrix
         CompartmentView                     ForwardModelObservable
         net.morph_population(...)           net.mass_population(...)
         net.compartment_project(...)        net.couple(...)
         pyr.compartments(region)            net.observe_forward(...)
         clock dispatch handler              brainmass backend
```

A domain extension implements as much of this surface as it needs.
The smallest useful extension is "one new `IRNode` subclass plus a
builder verb plus a clock-backend handler" — for a single new node
kind. A full domain (brainmass) implements all five.

---

## 5.3 `IRNode` — defining a node kind

The substrate exposes one abstract base class:

```python
# brainpy_state/spec/ir.py — NEW additions

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import ClassVar, Iterable, Mapping, Any

class IRNode(ABC):
    """Every node in a NetIR is an IRNode.

    Subclasses must be *frozen dataclasses* registered through
    ``@register_node_kind``. The dataclass shape is the canonical
    schema; the runtime never reflects beyond declared fields.
    """

    # Unique kind tag — used in canonical JSON, content hash, and the
    # backend dispatch table. Set by the subclass.
    KIND: ClassVar[str]

    # Every node has an id. The dataclass declares it as the first field.
    id: str

    # ── Required overrides ────────────────────────────────────────────

    @abstractmethod
    def referenced_ids(self) -> Iterable[str]:
        """Every node id this node refers to (for topology checks)."""

    @abstractmethod
    def validate(self, ir: "NetIR") -> None:
        """Domain-local validation; raises SpecError on failure.

        Called by ``NetSpec.finalize()`` after all nodes are present.
        Should verify only invariants intrinsic to this node kind —
        cross-node consistency that the substrate already enforces
        (id uniqueness, referenced ids exist) is automatic.
        """

    # ── Default implementations the substrate provides ────────────────

    @classmethod
    def kind_tag(cls) -> str:
        return cls.KIND

    def fields_for_hash(self) -> Iterable[tuple[str, Any]]:
        """Yield (name, value) pairs in canonical order.

        Default: dataclass field order, which equals declaration
        order. Override only when the canonical form is something
        other than the raw dataclass fields.
        """
        for f in fields(self):
            yield f.name, getattr(self, f.name)
```

The substrate also exposes a registration decorator:

```python
# brainpy_state/spec/registry.py — NEW

_NODE_KINDS: dict[str, type[IRNode]] = {}

def register_node_kind(cls: type[IRNode]) -> type[IRNode]:
    """Register an IRNode subclass.

    Enforces:
      - ``cls`` is a frozen dataclass.
      - ``cls.KIND`` is set and globally unique.
      - The first dataclass field is ``id: str``.
    """
    if cls.KIND in _NODE_KINDS and _NODE_KINDS[cls.KIND] is not cls:
        raise DomainError(
            f"node kind {cls.KIND!r} already registered by "
            f"{_NODE_KINDS[cls.KIND].__module__}"
        )
    _NODE_KINDS[cls.KIND] = cls
    return cls

def node_kind(tag: str) -> type[IRNode]:
    """Resolve a tag to its registered class."""
    return _NODE_KINDS[tag]
```

### 5.3.1 The built-in node kinds are pre-registered

`brainpy_state.spec.ir` re-exports the five SNN node dataclasses
already documented in Chapter 2 with the `IRNode` base added and
`KIND` set:

```python
# brainpy_state/spec/ir.py

@register_node_kind
@dataclass(frozen=True)
class PopulationNode(IRNode):
    KIND: ClassVar[str] = "brainpy.population"
    id: str
    model: ModelRef
    size: Union[int, Tuple[int, ...]]
    batch: Optional[int] = None
    init: Optional[Mapping[str, Any]] = None
    tags: Tuple[str, ...] = ()
    positions: Optional[PositionsRef] = None       # spatial; §3.5.2

    def referenced_ids(self) -> Iterable[str]:
        return ()

    def validate(self, ir: "NetIR") -> None:
        _validate_model_kind(self.model, expected_role="neuron")
        _validate_units(self.model, expected=NEURON_SIGNATURE[self.model.kind])
        # ... existing SPEC-NNN checks
```

(Analogous for `ProjectionNode`, `InputNode`, `ObservableNode`,
`SubNetworkNode`.) From this chapter on, *any* `IRNode` is treated
uniformly by the substrate.

### 5.3.2 The `NetIR` root sees nodes uniformly

```python
@dataclass(frozen=True)
class NetIR:
    version: str
    name: str

    # Built-in tuples — populated by the built-in builder verbs:
    populations: Tuple[PopulationNode, ...]
    projections: Tuple[ProjectionNode, ...]
    inputs:      Tuple[InputNode,      ...]
    observables: Tuple[ObservableNode, ...]
    subnetworks: Tuple[SubNetworkNode, ...] = ()

    # Domain-extension nodes — every IRNode subclass not in the
    # built-in set lands here, in finalization order:
    extension_nodes: Tuple[IRNode, ...] = ()

    variables: Tuple[VariableDecl, ...] = ()
    compounds: CompoundMeta              = field(default_factory=CompoundMeta)
    meta:      Mapping[str, Any]         = field(default_factory=dict)

    def nodes(self) -> Iterable[IRNode]:
        """All nodes in canonical iteration order."""
        yield from self.populations
        yield from self.projections
        yield from self.inputs
        yield from self.observables
        yield from self.subnetworks
        yield from self.extension_nodes

    def nodes_of(self, cls: type[IRNode]) -> Iterable[IRNode]:
        return (n for n in self.nodes() if isinstance(n, cls))

    def content_hash(self) -> str: ...     # SHA-256 over canonical JSON
```

Domain-extension nodes get a typed home (`extension_nodes`) only as
an implementation detail of the dataclass — there is no semantic
difference between a built-in `PopulationNode` and a
`braincell.MorphPopulation` from the IR consumer's point of view.
Both are `IRNode` subclasses; both are iterated over by `ir.nodes()`;
both are validated; both contribute to the content hash.

### 5.3.3 What an `IRNode` subclass looks like in practice

The full template a domain author writes:

```python
@register_node_kind
@dataclass(frozen=True)
class MyNewNode(IRNode):
    KIND: ClassVar[str] = "mydomain.my_node"

    # First field is always ``id: str``.
    id: str

    # Then the domain-specific fields. Use existing value wrappers
    # (ModelRef, DistRef, Trainable, VariableRef, ConnRule) wherever
    # they fit — they already participate in the content hash and
    # the variable-binding pass.
    model: ModelRef
    size: int
    extra_param: u.Quantity

    # Frozen tuples and Mappings, never lists or dicts at the leaves.
    children: Tuple[str, ...] = ()

    def referenced_ids(self) -> Iterable[str]:
        return self.children

    def validate(self, ir: NetIR) -> None:
        if self.size <= 0:
            raise SpecError(
                f"DOMAIN-101 [mydomain]: size must be positive on "
                f"node {self.id!r}, got {self.size}"
            )
        # ... any other intrinsic checks
```

The substrate handles everything else: pytree registration, JSON
round-trip, content-hash inclusion, id uniqueness, variable
resolution, frozen-dataclass equality.

---

## 5.4 `ViewHandle` — sub-population references

The same protocol pattern, scaled down, for view handles:

```python
# brainpy_state/spec/handles.py — NEW additions

class ViewHandle(ABC):
    """A typed symbolic reference returned by builder verbs."""
    KIND: ClassVar[str]
    id: str
    _spec: "NetSpec"

    @abstractmethod
    def to_view_ref(self) -> "ViewRef":
        """Lower this handle to a serializable ViewRef."""

def register_view_kind(cls: type[ViewHandle]) -> type[ViewHandle]: ...
```

The four built-in handles — whole-population, slice, indices, merge
— are pre-registered. A domain extension registers new ones (e.g.,
`braincell.CompartmentViewHandle`) the same way. ViewHandle methods
are not the protocol surface — domains add methods freely; the only
shared contract is `to_view_ref()`.

A concrete `ViewRef` subtype is the IR-side counterpart and uses the
same registration pattern (`register_view_ref_kind`). The encode /
decode is supplied by the domain and called by the canonical-JSON
serializer dispatching on the `KIND` tag.

---

## 5.5 Builder verbs — extending `NetSpec`

`NetSpec` is open for extension via a decorator that attaches a
function to the class as a method:

```python
# brainpy_state/spec/builder.py — NEW

def register_builder_verb(name: str | None = None):
    """Attach ``fn`` to ``NetSpec`` as a method.

    Signature contract: the first positional argument is ``self``;
    keyword-only arguments are encouraged for everything except the
    handful of "main" positional ones (handles being projected, etc.).
    The function appends one or more nodes to the spec via
    ``self._append_node(node)`` and returns a handle.
    """
    def decorator(fn):
        verb_name = name or fn.__name__
        if hasattr(NetSpec, verb_name):
            raise DomainError(
                f"NetSpec already has verb {verb_name!r} "
                f"({type(getattr(NetSpec, verb_name)).__name__})"
            )
        setattr(NetSpec, verb_name, fn)
        _REGISTERED_VERBS[verb_name] = fn
        return fn
    return decorator
```

The internal helpers a verb is expected to use are also exposed:

```python
class NetSpec:
    ...
    def _append_node(self, node: IRNode) -> None: ...
    def _mint_id(self, base: str | None) -> str: ...
    def _to_model_ref(self, m: ModelLike) -> ModelRef: ...
    def _to_conn_rule(self, r: ConnRuleLike) -> ConnRule: ...
    def _resolve_view(self, h: ViewHandle | str) -> ViewRef: ...
    def _resolve_value(self, v) -> Any:
        """Pass through scalars/units; lift DistLike → DistRef;
        leave Trainable / VariableRef / Noise wrappers alone."""
```

`_append_node` routes a node into `extension_nodes` if its class is
not one of the five built-ins; otherwise into the matching typed
tuple. The verb author does not branch on this.

### 5.5.1 Why a decorator, not a mixin?

Mixins force every domain to instantiate `NetSpec` through a
domain-specific subclass (`MassNetSpec(NetSpec)`), which is hostile to
composition: a multi-domain spec using both braincell and brainmass
would need `MultiDomainNetSpec(BrainCellMixin, BrainMassMixin,
NetSpec)`. The decorator approach lets the user write
`spec.NetSpec(...)` regardless of which domains are active, and each
imported extension contributes verbs to the same class. Domains do
not interact through inheritance; they interact through the registry.

---

## 5.6 Codec — round-trip and content hash

For every new `IRNode` subclass, the substrate generates a default
codec from the dataclass shape:

```python
# brainpy_state/spec/codec.py — NEW

def to_canonical(node: IRNode) -> dict:
    """Default codec — works for any frozen dataclass whose leaves
    are scalars, u.Quantity, frozen tuples/mappings, or already-known
    IR value wrappers (ModelRef, ConnRule, DistRef, Trainable,
    Noise, VariableRef).

    Domains override by registering a custom codec for their KIND.
    """
    out = {"_k": node.KIND}
    for name, value in node.fields_for_hash():
        out[name] = _encode_leaf(value)
    return out

def from_canonical(payload: dict) -> IRNode:
    cls = node_kind(payload["_k"])
    return cls(**{k: _decode_leaf(v) for k, v in payload.items() if k != "_k"})

def register_codec(kind: str, encode, decode) -> None: ...
```

The default codec covers virtually every domain use case. A domain
needs a custom codec only when its node carries values the substrate
does not know how to encode — e.g., a morphology tree object, or a
large numpy connectivity matrix that wants a more compact
representation than nested lists.

Domains contribute custom codecs the same way they contribute node
kinds:

```python
@register_codec("braincell.morph_population")
def _encode_morph_pop(node: MorphPopulation) -> dict: ...
```

Canonical JSON rules from Chapter 6 §6.2 are unchanged: keys sort
lexicographically, floats use `repr`, `u.Quantity` renders as
`{"_q": [mantissa, unit_str]}`, list order is semantic, the leading
`_k` tag identifies the node kind. The content hash is SHA-256 over
this canonical form.

---

## 5.7 Backend dispatch — handlers per node kind

A backend is a registered `SimBackend` / `TrainBackend` (Chapter 6
§6.1) whose `build()` walks `ir.nodes()` and dispatches each node to
a handler keyed on its `KIND`:

```python
# brainpy_state/backend.py — NEW additions

class Backend(Protocol):
    name: str
    capabilities: BackendCapabilities
    handlers: Mapping[str, "NodeHandler"]      # KIND -> handler

class NodeHandler(Protocol):
    """Wires one IRNode into the backend's runtime."""
    def __call__(self, ctx: "BuildContext", node: IRNode) -> None: ...

class BuildContext(Protocol):
    """Per-build state passed to every NodeHandler.

    Exposes — among other things — the user-supplied numerical
    realization options. Per §5.1.1 those live on the backend, not on
    the IR; handlers query them through this surface.
    """
    ir: NetIR
    seed: int
    dt: u.Quantity | None
    variables: Mapping[str, Any]               # bound build-time variables (§3.14)

    def kind_options(self, kind: str) -> Mapping[str, Any]:
        """Backend-build kwarg ``kind_options.get(kind, {})``.
        Defaults that apply to every node of this KIND."""

    def node_options(self, node_id: str) -> Mapping[str, Any]:
        """Backend-build kwarg ``node_options.get(node_id, {})``.
        Per-node overrides keyed by IR node id."""

    def options_for(self, node: IRNode) -> Mapping[str, Any]:
        """Merge of ``kind_options(node.KIND)`` and
        ``node_options(node.id)`` — node-level overrides win."""
```

Backends accept both `kind_options` and `node_options` as `build()`
kwargs. The substrate's `BuildContext` provides the lookup so handlers
never reach back into the IR for solver / discretization / buffer
choices:

```python
sim = clock.build(
    ir, seed=0, dt=0.025*u.ms,
    kind_options={"braincell.morph_population":
                  dict(solver="staggered", cv_policy="per_branch")},
    node_options={"L5_pyr": dict(solver="exp_euler")},
)
```

Built-in backends pre-register handlers for the five built-in
node kinds. Two patterns let a domain extend a backend:

### 5.7.1 Pattern A — extend an existing backend

The domain ships a small module that registers handlers on a core
backend it wants to support. Example: braincell registers handlers
for its morphological node kinds on the `clock` backend:

```python
# braincell/dsl/clock_handlers.py
from brainpy.state import clock
from braincell.dsl import MorphPopulation, CompartmentTargetedProjection

@clock.register_handler(MorphPopulation.KIND)
def _build_morph_population(ctx, node: MorphPopulation) -> None:
    # Numerical realization knobs come from the backend's options
    # mapping, never from the IR node (§5.1.1).
    opts = ctx.options_for(node)         # merges kind_options + node_options
    cell = braincell.Cell(
        morphology=_load_morph(node.morphology),
        cv_policy=opts.get("cv_policy", "per_branch"),
        solver=opts.get("solver", "staggered"),
    )
    for region_pred, mech_ref in node.paint:
        cell.paint(_resolve_region(region_pred),
                   _instantiate_mech(mech_ref))
    for region_pred, point_ref in node.place:
        cell.place(_resolve_region(region_pred),
                   _instantiate_point(point_ref))
    ctx.builder.add(node.id, cell)


@clock.register_handler(CompartmentTargetedProjection.KIND)
def _build_compartment_projection(ctx, node) -> None: ...
```

The clock backend's `build()` walks `ir.nodes()` and looks up each
node's handler. A node whose `KIND` has no handler raises
`BackendCapabilityError` pointing at the responsible id — the same
error class §6.1.3 uses today.

### 5.7.2 Pattern B — ship a dedicated backend

The domain ships a top-level peer of `brainpy.state.clock` / `.bptt`
because its runtime is fundamentally different. brainmass does this:
continuous-time SDE solver, parcel-as-unit semantics, dense coupling
matrices, forward-model post-processing — none of which fit the
clock loop.

```python
# brainmass/dsl/backend.py
from brainpy_state.backend import SimBackend, BackendCapabilities, register_handler

class BrainmassSim:
    name = "brainmass"
    capabilities = BackendCapabilities(
        supports_delay=True, supports_plasticity=False,
        supports_distributions=True, supports_nested_subnetworks=True,
        supports_training=False, supports_batch=True,
        supports_positions=False, supports_morphology=False,
        supports_noise=True, supports_signals=True,
        supports_schedules=True, supports_structural_plasticity=False,
        supports_graphs=False,
        supported_neuron_kinds=frozenset({"brainmass.mass_population"}),
        supported_synapse_kinds=frozenset(),
        supported_output_kinds=frozenset(),
        supported_rules=frozenset(),
        supported_layer_macros=frozenset(),
        supported_input_kinds=frozenset({"brainmass.mass_input"}),
    )

    def build(self, ir, *, seed, dt=None, variables=None, **opts):
        ctx = BuildContext(ir=ir, seed=seed, dt=dt, variables=variables)
        for node in ir.nodes():
            handler = self.handlers.get(node.KIND)
            if handler is None:
                raise BackendCapabilityError(
                    f"backend {self.name!r} has no handler for node "
                    f"kind {node.KIND!r} (node id={node.id!r})"
                )
            handler(ctx, node)
        return BrainmassSimulator(ctx)

    handlers = {}    # populated by @register_handler below

BACKEND = BrainmassSim()

build = BACKEND.build      # so user code can `from brainmass.dsl import build`
```

A backend's handler table is open: third parties can register
handlers on a brainmass backend the same way braincell registers on
the clock backend. Backends advertise the set of node kinds they
handle through `capabilities.supported_neuron_kinds` etc. — the
existing capability dataclass already supports this directly.

---

## 5.8 Discovery, activation, conflict

The same entry-point group used in §7.5 covers extensions:

```toml
[project.entry-points."brainpy_state.spec.extensions"]
braincell = "braincell.dsl:_register"
brainmass = "brainmass.dsl:_register"

[project.entry-points."brainpy_state.backends.sim"]
brainmass = "brainmass.dsl.backend:BACKEND"
```

The `_register` callable is executed once at first import of
`brainpy_state.spec`. It imports the domain's `dsl` module, which
runs every `@register_node_kind` / `@register_builder_verb` /
`@register_handler` decorator at module-import time. After
`_register` returns, the substrate has the domain's nodes, verbs,
handles, codecs, and handlers in its tables.

An import directly from the domain module — `import brainmass.dsl` —
also activates the extension (decorators self-register). Entry-point
registration covers the case where the user wrote
`import brainpy.state.spec as spec` without importing the domain;
verbs and nodes are still available because the entry point already
ran.

### 5.8.1 Conflict resolution

| Situation                                                     | Resolution                                                    |
|---------------------------------------------------------------|---------------------------------------------------------------|
| Two extensions register the same node `KIND`                  | `DomainError` at activation, naming both modules.             |
| An extension's builder verb collides with an existing method  | `DomainError` at decoration. Domains namespace verbs by convention (`net.mass_population`, not `net.population_mass`). |
| A spec uses a verb whose extension is not installed           | `AttributeError` from the Python attribute lookup — the user gets a stack-traced message telling them which extension is missing. |
| An IR loaded from JSON uses a node KIND not registered        | `DomainError` at `NetIR.from_dict`, naming the unknown KIND and the suspected owning extension (from the namespace prefix). |

Conflicts are detected at activation / decoration time, not at
`finalize()` — extensions fail loudly the moment they are installed
incompatibly, before the user has a chance to construct a spec.

---

## 5.9 Worked extension — `braincell`

This subsection is the **complete public interface** of the
`braincell.dsl` module: every node kind, view handle, builder verb,
and backend handler that braincell would ship to extend the spec.

### 5.9.1 Node kinds

```python
# braincell/dsl/nodes.py
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Iterable, Mapping, Optional, Tuple, Any
import saiunit as u
from brainpy_state.spec import (
    IRNode, ModelRef, ConnRule, DistRef, Trainable, SpecError,
    register_node_kind,
)

# A region predicate is a frozen mapping describing a sub-set of
# compartments on a Cell. Examples:
#   {"kind": "SomaRegion"}
#   {"kind": "ApicalDendrite"}
#   {"kind": "DistanceFromSoma", "min": "200 um", "max": "400 um"}
#   {"op": "and", "args": [...]}
RegionPred = Mapping[str, Any]


@register_node_kind
@dataclass(frozen=True)
class MorphPopulation(IRNode):
    """A population of multi-compartment cells with a shared morphology.

    Describes the *model*: morphology, painted mechanisms, placed
    point processes, spike threshold. Numerical realization
    (compartmentalization policy, ODE solver, time step) is supplied
    at backend build time via ``node_options`` — see §5.1.1.
    """

    KIND: ClassVar[str] = "braincell.morph_population"

    id: str
    size: int
    morphology: ModelRef                    # e.g. ModelRef("braincell.MorphFromSWC", {"path": "..."})
    paint: Tuple[Tuple[RegionPred, ModelRef], ...]    # (region, mechanism)
    place: Tuple[Tuple[RegionPred, ModelRef], ...] = ()   # (region, point process)
    state_init: Optional[Mapping[str, Any]] = None        # initial conditions per state variable
    spike_threshold: u.Quantity = 0 * u.mV
    tags: Tuple[str, ...] = ()

    def referenced_ids(self) -> Iterable[str]:
        return ()

    def validate(self, ir) -> None:
        if self.size <= 0:
            raise SpecError(
                f"DOMAIN-101 [braincell]: MorphPopulation {self.id!r} "
                f"requires size > 0; got {self.size}"
            )
        if not self.paint:
            raise SpecError(
                f"DOMAIN-102 [braincell]: MorphPopulation {self.id!r} "
                f"must paint at least cable properties on AllRegion"
            )
        u.fail_for_dimension_mismatch(
            self.spike_threshold, u.mV,
            error_message=f"MorphPopulation {self.id!r} spike_threshold must be a voltage",
        )


@register_node_kind
@dataclass(frozen=True)
class CompartmentTargetedProjection(IRNode):
    """A projection whose post side targets a region predicate on a MorphPopulation."""

    KIND: ClassVar[str] = "braincell.compartment_projection"

    id: str
    pre: "ViewRef"                          # any ViewRef
    post_population: str                    # id of the target MorphPopulation
    post_region: RegionPred                 # which compartments
    rule: ConnRule                          # compartment-targeted rule from braincell.dsl.conn
    synapse: ModelRef
    output: ModelRef
    plasticity: Optional[ModelRef] = None
    seed: Optional[int] = None

    def referenced_ids(self) -> Iterable[str]:
        yield self.post_population
        yield from _refs_in_view(self.pre)

    def validate(self, ir) -> None:
        pop = next((n for n in ir.nodes_of(MorphPopulation)
                    if n.id == self.post_population), None)
        if pop is None:
            raise SpecError(
                f"DOMAIN-105 [braincell]: CompartmentTargetedProjection "
                f"{self.id!r} targets {self.post_population!r}, "
                f"which is not a MorphPopulation"
            )
        if not self.rule.kind.startswith("braincell."):
            raise SpecError(
                f"DOMAIN-106 [braincell]: CompartmentTargetedProjection "
                f"{self.id!r} must use a braincell.* connectivity rule"
            )
```

### 5.9.2 View handle

```python
# braincell/dsl/handles.py
from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Any
from brainpy_state.spec import (
    ViewHandle, ViewRef, register_view_kind, register_view_ref_kind,
)

@register_view_ref_kind
@dataclass(frozen=True)
class CompartmentViewRef(ViewRef):
    KIND: ClassVar[str] = "braincell.compartment"
    population: str                          # the MorphPopulation id
    region: Mapping[str, Any] = field(default_factory=dict)   # RegionPred


@register_view_kind
class CompartmentViewHandle(ViewHandle):
    KIND = "braincell.compartment"

    def __init__(self, spec, pop_id: str, region):
        self._spec = spec
        self.id = pop_id
        self.region = region

    def to_view_ref(self) -> ViewRef:
        return CompartmentViewRef(
            population=self.id,
            region=_freeze_region(self.region),
        )

    # User-facing voltages / states / spikes on a compartment subset:
    @property
    def voltage(self): return _ObservableTarget(self, quantity="V")
    @property
    def spikes(self):  return _ObservableTarget(self, quantity="spike")
    def state(self, name: str): return _ObservableTarget(self, quantity=name)
```

The `PopulationHandle` already returned by `net.morph_population(...)`
exposes `.compartments(region)` that calls this constructor:

```python
class MorphPopulationHandle(PopulationHandle):
    def compartments(self, region) -> CompartmentViewHandle:
        return CompartmentViewHandle(self._spec, self.id, region)
```

### 5.9.3 Builder verbs

```python
# braincell/dsl/verbs.py
from brainpy_state.spec import register_builder_verb
from .nodes import MorphPopulation, CompartmentTargetedProjection
from .handles import MorphPopulationHandle, CompartmentViewHandle

@register_builder_verb("morph_population")
def morph_population(
    self,                                   # NetSpec
    name: str,
    *,
    size: int,
    morphology: "ModelLike",
    paint: list[tuple["RegionLike", "ModelLike"]],
    place: list[tuple["RegionLike", "ModelLike"]] | None = None,
    state_init: "Mapping[str, Any] | None" = None,
    spike_threshold: u.Quantity = 0 * u.mV,
    tags: tuple[str, ...] = (),
) -> MorphPopulationHandle:
    node = MorphPopulation(
        id=self._mint_id(name),
        size=size,
        morphology=self._to_model_ref(morphology),
        paint=tuple((_freeze_region(r), self._to_model_ref(m))
                    for r, m in paint),
        place=tuple((_freeze_region(r), self._to_model_ref(m))
                    for r, m in (place or ())),
        state_init=_freeze_init(state_init),
        spike_threshold=spike_threshold,
        tags=tuple(tags),
    )
    self._append_node(node)
    return MorphPopulationHandle(spec=self, id=node.id)


@register_builder_verb("compartment_project")
def compartment_project(
    self,
    pre,                                    # PopulationHandle | ViewHandle | QueryHandle
    post,                                   # CompartmentViewHandle
    *,
    rule: "ConnRuleLike",
    synapse: "ModelLike",
    output: "ModelLike",
    plasticity: "ModelLike | None" = None,
    seed: int | None = None,
    name: str | None = None,
) -> "ProjectionHandle":
    if not isinstance(post, CompartmentViewHandle):
        raise SpecError(
            "DOMAIN-110 [braincell]: compartment_project requires a "
            "CompartmentViewHandle as post"
        )
    node = CompartmentTargetedProjection(
        id=self._mint_id(name or f"{pre.id}__to__{post.id}_{post.region}"),
        pre=self._resolve_view(pre),
        post_population=post.id,
        post_region=_freeze_region(post.region),
        rule=self._to_conn_rule(rule),
        synapse=self._to_model_ref(synapse),
        output=self._to_model_ref(output),
        plasticity=self._to_model_ref(plasticity) if plasticity else None,
        seed=seed,
    )
    self._append_node(node)
    return _make_projection_handle(self, node.id)
```

`net.project(pre, post, ...)` (the built-in verb from §3.6) keeps
working unchanged when `post` is a plain `PopulationHandle` or
`ViewHandle`. The braincell extension chooses to expose
`compartment_project` as a *separate* verb for clarity, but it could
equally well dispatch from inside the built-in `project` verb on
`isinstance(post, CompartmentViewHandle)`. The pattern is up to the
extension author.

### 5.9.4 Backend dispatch (clock + bptt)

```python
# braincell/dsl/backends/clock.py
from brainpy.state import clock
from braincell.dsl import MorphPopulation, CompartmentTargetedProjection
import braincell

_BRAINCELL_DEFAULTS = dict(cv_policy="per_branch", solver="staggered")

@clock.register_handler(MorphPopulation.KIND)
def _build_morph_population(ctx, node):
    # Implementation knobs live on the backend, not on the IR node.
    # `ctx.node_options(node.id)` returns the dict the user passed to
    # `clock.build(ir, ..., node_options={node.id: {...}})`, merged
    # over the backend's per-kind defaults.
    opts = {**_BRAINCELL_DEFAULTS,
            **ctx.kind_options(MorphPopulation.KIND),
            **ctx.node_options(node.id)}
    cv_policy = opts["cv_policy"]
    solver    = opts["solver"]
    if solver not in {"staggered", "exp_euler", "rk4", "dopri5"}:
        raise BackendCapabilityError(
            f"clock backend: unknown solver {solver!r} on MorphPopulation "
            f"{node.id!r}; supported: staggered, exp_euler, rk4, dopri5"
        )

    cell = braincell.Cell(
        morphology=_load_morphology(node.morphology),
        cv_policy=cv_policy,
        solver=solver,
        spike_threshold=node.spike_threshold,
    )
    for region_pred, mech in node.paint:
        cell.paint(_resolve_region(region_pred), _instantiate_mech(mech, ctx))
    for region_pred, point in node.place:
        cell.place(_resolve_region(region_pred), _instantiate_point(point, ctx))
    cell = cell.build(size=node.size)
    if node.state_init:
        cell.initialize(**node.state_init)
    ctx.builder.add(node.id, cell)


@clock.register_handler(CompartmentTargetedProjection.KIND)
def _build_compartment_projection(ctx, node):
    pre = ctx.resolve(node.pre)
    post_pop = ctx.population(node.post_population)
    compartment_mask = _resolve_region_to_mask(
        post_pop, node.post_region
    )
    rule = _instantiate_braincell_rule(
        node.rule, post_mask=compartment_mask,
        seed=ctx.seed_for(node, "rule"),
    )
    result = rule.generate(pre_size=pre.size, post=post_pop)
    ctx.builder.connect_from_result(
        pre, post_pop, result=result,
        syn=_instantiate_syn(node.synapse, post=post_pop),
        out=_instantiate_out(node.output),
        plasticity=_instantiate_plasticity(node.plasticity),
    )


# braincell/dsl/backends/bptt.py — identical shape, but registers on
# brainpy.state.bptt and only accepts a differentiable solver from
# ctx.node_options(...); otherwise raises BackendCapabilityError with
# the offending node.id. Allowed: {"staggered", "exp_euler"}.
```

braincell ships *no* dedicated backend — its node kinds are
adapted into existing core backends through handler registration.
The IR node itself contains no numerical knobs; each handler reads
its solver, `cv_policy`, and any other realization choice from the
backend's `kind_options` / `node_options` mapping. Backend-specific
constraints (e.g. bptt's "differentiable solvers only") are enforced
inside the handler with `BackendCapabilityError`.

### 5.9.5 Capability declaration

The clock backend already publishes `capabilities` (§6.1.3). The
braincell extension adds entries:

```python
# braincell/dsl/__init__.py — top-level _register
from brainpy.state import clock, bptt

def _register():
    from . import nodes, handles, verbs        # decorators self-register
    from .backends import clock as _bc_clock   # handler registration
    from .backends import bptt  as _bc_bptt

    clock.capabilities = clock.capabilities.with_(
        supported_neuron_kinds=clock.capabilities.supported_neuron_kinds | {
            "braincell.morph_population",
        },
        supported_rules=clock.capabilities.supported_rules | {
            "braincell.SomaToDendrite", "braincell.ApicalDendriteTargeting",
            "braincell.BasalDendriteTargeting", "braincell.MorphologyDistance",
            # ...
        },
        supports_morphology=True,
    )
    bptt.capabilities = bptt.capabilities.with_(...)
```

`BackendCapabilities.with_(...)` is a frozen-dataclass copy helper
that returns a new capabilities object with the listed fields
overridden — extensions never mutate built-in state.

### 5.9.6 User-facing example

```python
import saiunit as u
import brainpy.state.spec as spec
import braincell.dsl as bc          # registers extension on import

import brainpy.state as bs          # gives us clock backend

net = spec.NetSpec("L5_microcircuit")

# Standard point populations from the SNN domain:
pv  = net.population("PV",  spec.models.LIF(tau=10*u.ms, V_th=-50*u.mV),
                     size=200, tags=("interneuron",))
sst = net.population("SST", spec.models.LIF(tau=15*u.ms, V_th=-52*u.mV),
                     size=200, tags=("interneuron",))

# Morphological L5 pyramidals via the braincell extension verb.
# Note: no solver, no cv_policy here — those are implementation
# choices supplied to clock.build(...) below.
pyr = net.morph_population(
    "L5_pyr",
    size=500,
    morphology=bc.MorphFromSWC("l5_pyr.swc"),
    paint=[
        (bc.AllRegion(),
         bc.CableProperty(Cm=1*u.uF/u.cm**2, Ra=100*u.ohm*u.cm,
                          Em=-70*u.mV)),
        (bc.SomaRegion(),
         bc.INa_HH(g_max=0.12*u.S/u.cm**2)),
        (bc.SomaRegion(),
         bc.IKDR_HH(g_max=0.036*u.S/u.cm**2)),
        (bc.ApicalDendrite(),
         bc.Ih(g_max=spec.train(
             spec.init.LogNormal(mean=5e-3*u.S/u.cm**2,
                                 std=1e-3*u.S/u.cm**2)))),
        (bc.BasalDendrite(),
         bc.ICaT(g_max=2*u.mS/u.cm**2)),
    ],
    place=[
        (bc.SomaRegion(), bc.CurrentClamp(amp=0*u.nA)),
    ],
    spike_threshold=0*u.mV,
    tags=("cortex", "L5", "pyramidal"),
)

# Compartment-resolved targets:
apical = pyr.compartments(bc.ApicalDendrite())
basal  = pyr.compartments(bc.BasalDendrite())
soma   = pyr.compartments(bc.SomaRegion())

# Point → compartment projection via the braincell extension verb:
net.compartment_project(
    pv, soma,
    rule=bc.ProximalTargeting(
        prob=0.2,
        weight=spec.train(0.5*u.nS, constraint="positive"),
    ),
    synapse=spec.models.GABAa(tau=8*u.ms),
    output=spec.models.COBA(E=-80*u.mV),
)

# Point → point projection — built-in verb, unchanged:
net.project(
    pv, sst,
    rule=spec.conn.FixedProb(prob=0.1, weight=0.3*u.nS),
    synapse=spec.models.GABAa(tau=6*u.ms),
    output=spec.models.COBA(E=-75*u.mV),
)

# Inputs and observables — built-in verbs, no special-casing needed.
# A built-in input.target referencing a CompartmentViewHandle goes
# through CompartmentViewRef in the IR, which the clock handler
# already knows how to resolve.
net.input(apical, spec.models.PoissonSpike(rate=5*u.Hz), weight=0.1*u.nS)
net.observe(pv.spikes)
net.observe(soma.voltage, every=1*u.ms)
net.observe(apical.state("Ca_concentration"), every=5*u.ms)

ir = net.finalize()                 # validates every IRNode + cross-node ids

# Build on the clock backend (which now has braincell handlers).
# Numerical realization choices live here, not on the IR:
sim = bs.clock.build(
    ir, seed=0, dt=0.025*u.ms,
    # Per-kind defaults that apply to every node of this kind:
    kind_options={
        "braincell.morph_population": dict(
            cv_policy="per_branch",
            solver="staggered",
        ),
    },
    # Per-node overrides (by id) — for cells that want a different solver:
    node_options={
        "L5_pyr": dict(solver="exp_euler"),
    },
)
trace = sim.run(1000*u.ms)
```

Nothing in this example uses an `extras` field, a namespaced opt-in,
or a pack-discovery hook from the user's side. The morphological
populations are first-class nodes; the compartment view is a
first-class handle; the verb that creates them lives on
`net.morph_population` next to `net.population`. The fact that all
of this is supplied by an out-of-tree package is invisible at the
call site.

---

## 5.10 Worked extension — `brainmass`

`brainmass` is the harder example: it introduces continuous-state
populations, a different connectivity primitive (coupling matrix),
and forward-model observables. It ships its own backend rather than
extending the clock backend.

### 5.10.1 Node kinds

```python
# brainmass/dsl/nodes.py
from dataclasses import dataclass
from typing import ClassVar, Iterable, Optional, Tuple, Mapping, Any
import saiunit as u
from brainpy_state.spec import (
    IRNode, ViewRef, ModelRef, ConnRule, DistRef, Trainable, SpecError,
    register_node_kind,
)


@register_node_kind
@dataclass(frozen=True)
class MassPopulation(IRNode):
    """A population of mass-model nodes (parcels / regions).

    Describes the *model and its dynamics*: which mass model, its
    parameters (with units), the noise term in the SDE, the initial
    conditions, and the number of parcels. Numerical realization
    (ODE / SDE solver, time step, delay-buffer sizes) is supplied
    at backend build time via ``node_options`` — see §5.1.1.

    Each unit along ``size`` is one mass-model evaluation point —
    typically one parcel in a brain atlas. State variables are
    continuous; there are no spikes.
    """

    KIND: ClassVar[str] = "brainmass.mass_population"

    id: str
    model: ModelRef                         # e.g. "brainmass.WilsonCowan", "brainmass.JansenRit"
    size: int                               # number of parcels
    noise: Optional[ModelRef] = None        # noise is a *dynamics* term, not a realization choice
    state_init: Optional[Mapping[str, Any]] = None
    tags: Tuple[str, ...] = ()

    def referenced_ids(self) -> Iterable[str]:
        return ()

    def validate(self, ir) -> None:
        if not self.model.kind.startswith("brainmass."):
            raise SpecError(
                f"DOMAIN-201 [brainmass]: MassPopulation {self.id!r} "
                f"model must be brainmass.*; got {self.model.kind!r}"
            )
        if self.size <= 0:
            raise SpecError(
                f"DOMAIN-202 [brainmass]: MassPopulation {self.id!r} "
                f"requires size > 0"
            )
        if self.noise is not None and not self.noise.kind.startswith("brainmass."):
            raise SpecError(
                f"DOMAIN-203 [brainmass]: noise must be brainmass.*; "
                f"got {self.noise.kind!r}"
            )


@register_node_kind
@dataclass(frozen=True)
class CouplingNode(IRNode):
    """Structural-connectivity coupling between mass populations.

    Replaces the built-in ProjectionNode for mass dynamics: the
    connectivity is a (post, pre) coupling matrix, not a sparse
    edge sample.
    """

    KIND: ClassVar[str] = "brainmass.coupling"

    id: str
    pre: ViewRef
    post: ViewRef
    matrix: ConnRule                        # ConnRule(kind="brainmass.CouplingMatrix",
                                            #          params={"W": <array>, "delays": <array>, ...})
    kernel: ModelRef                        # "brainmass.DiffusiveCoupling" | "brainmass.AdditiveCoupling"
    scale: Optional[Any] = None             # scalar | DistRef | Trainable | VariableRef
    seed: Optional[int] = None

    def referenced_ids(self) -> Iterable[str]:
        yield from _refs_in_view(self.pre)
        yield from _refs_in_view(self.post)

    def validate(self, ir) -> None:
        if not self.matrix.kind.startswith("brainmass."):
            raise SpecError(
                f"DOMAIN-210 [brainmass]: CouplingNode {self.id!r} must "
                f"use a brainmass.* coupling rule; got {self.matrix.kind!r}"
            )
        if not self.kernel.kind.startswith("brainmass."):
            raise SpecError(
                f"DOMAIN-211 [brainmass]: CouplingNode {self.id!r} kernel "
                f"must be brainmass.*"
            )
        for v in (self.pre, self.post):
            target = _resolve_population(v, ir)
            if not isinstance(target, MassPopulation):
                raise SpecError(
                    f"DOMAIN-212 [brainmass]: CouplingNode {self.id!r} "
                    f"endpoints must be MassPopulations"
                )


@register_node_kind
@dataclass(frozen=True)
class ForwardModelObservable(IRNode):
    """An observable that applies a forward model to mass state."""

    KIND: ClassVar[str] = "brainmass.forward_model"

    id: str
    target: ViewRef                         # MassPopulation view
    state_var: str                          # which state variable feeds the forward model
    forward_model: ModelRef                 # "brainmass.BOLDSignal", "brainmass.EEGLeadFieldModel", ...
    lead_field: Optional[ModelRef] = None   # required for EEG/MEG
    every: Optional[u.Quantity] = None      # downsample period
    reducer: Optional[str] = None

    def referenced_ids(self) -> Iterable[str]:
        return _refs_in_view(self.target)

    def validate(self, ir) -> None:
        if not self.forward_model.kind.startswith("brainmass."):
            raise SpecError(
                f"DOMAIN-220 [brainmass]: ForwardModelObservable {self.id!r} "
                f"forward_model must be brainmass.*"
            )
        needs_lead_field = self.forward_model.kind in {
            "brainmass.EEGLeadFieldModel",
            "brainmass.MEGLeadFieldModel",
        }
        if needs_lead_field and self.lead_field is None:
            raise SpecError(
                f"DOMAIN-221 [brainmass]: {self.forward_model.kind} "
                f"requires a lead_field model on {self.id!r}"
            )


@register_node_kind
@dataclass(frozen=True)
class MassInput(IRNode):
    """Time-series stimulus into a MassPopulation."""

    KIND: ClassVar[str] = "brainmass.mass_input"

    id: str
    target: ViewRef
    source: ModelRef                        # "brainmass.OUProcess", "brainmass.PiecewiseConstant", ...
    state_var: str = "I_ext"                # which state variable receives the input

    def referenced_ids(self) -> Iterable[str]:
        return _refs_in_view(self.target)

    def validate(self, ir) -> None:
        if not self.source.kind.startswith("brainmass."):
            raise SpecError(
                f"DOMAIN-230 [brainmass]: MassInput {self.id!r} "
                f"source must be brainmass.*"
            )
```

### 5.10.2 Builder verbs

```python
# brainmass/dsl/verbs.py
from brainpy_state.spec import register_builder_verb
from .nodes import MassPopulation, CouplingNode, ForwardModelObservable, MassInput
from .handles import MassPopulationHandle


@register_builder_verb("mass_population")
def mass_population(
    self,
    name: str,
    model: "ModelLike",
    *,
    size: int,
    noise: "ModelLike | None" = None,
    state_init: "Mapping[str, Any] | None" = None,
    tags: tuple[str, ...] = (),
) -> MassPopulationHandle:
    # No solver / dt / delay-buffer kwargs — those are realization
    # choices passed to brainmass.dsl.build(..., node_options=...).
    node = MassPopulation(
        id=self._mint_id(name),
        model=self._to_model_ref(model),
        size=size,
        noise=self._to_model_ref(noise) if noise else None,
        state_init=_freeze_init(state_init),
        tags=tuple(tags),
    )
    self._append_node(node)
    return MassPopulationHandle(spec=self, id=node.id)


@register_builder_verb("couple")
def couple(
    self,
    pre,
    post,
    *,
    matrix: "ConnRuleLike",                 # brainmass.CouplingMatrix(W=..., delays=...)
    kernel: "ModelLike",                    # brainmass.DiffusiveCoupling() | AdditiveCoupling()
    scale: Any | None = None,
    seed: int | None = None,
    name: str | None = None,
) -> "CouplingHandle":
    node = CouplingNode(
        id=self._mint_id(name or f"{pre.id}__couples__{post.id}"),
        pre=self._resolve_view(pre),
        post=self._resolve_view(post),
        matrix=self._to_conn_rule(matrix),
        kernel=self._to_model_ref(kernel),
        scale=self._resolve_value(scale),
        seed=seed,
    )
    self._append_node(node)
    return _make_coupling_handle(self, node.id)


@register_builder_verb("observe_forward")
def observe_forward(
    self,
    target,                                 # MassPopulationHandle | ViewHandle
    *,
    state_var: str,
    forward_model: "ModelLike",
    lead_field: "ModelLike | None" = None,
    every: u.Quantity | None = None,
    reducer: str | None = None,
    name: str | None = None,
) -> "ObservableHandle":
    node = ForwardModelObservable(
        id=self._mint_id(name or f"fwd_{target.id}"),
        target=self._resolve_view(target),
        state_var=state_var,
        forward_model=self._to_model_ref(forward_model),
        lead_field=self._to_model_ref(lead_field) if lead_field else None,
        every=every,
        reducer=reducer,
    )
    self._append_node(node)
    return _make_observable_handle(self, node.id)


@register_builder_verb("mass_input")
def mass_input(
    self,
    target,
    source: "ModelLike",
    *,
    state_var: str = "I_ext",
    name: str | None = None,
) -> "InputHandle":
    node = MassInput(
        id=self._mint_id(name or f"in_{target.id}"),
        target=self._resolve_view(target),
        source=self._to_model_ref(source),
        state_var=state_var,
    )
    self._append_node(node)
    return _make_input_handle(self, node.id)
```

### 5.10.3 Dedicated backend

```python
# brainmass/dsl/backend.py
from brainpy_state.backend import (
    SimBackend, BackendCapabilities, BuildContext, BackendCapabilityError,
    register_handler,
)
from .nodes import (
    MassPopulation, CouplingNode, ForwardModelObservable, MassInput,
)
import brainmass as bm
import jax.numpy as jnp


class BrainmassSim(SimBackend):
    name = "brainmass"
    capabilities = BackendCapabilities(
        supports_delay=True,
        supports_plasticity=False,
        supports_distributions=True,
        supports_nested_subnetworks=True,
        supports_training=False,
        supports_batch=True,
        supports_positions=False,
        supports_morphology=False,
        supports_noise=True,
        supports_signals=True,
        supports_schedules=True,
        supports_structural_plasticity=False,
        supports_graphs=False,
        supported_neuron_kinds=frozenset({"brainmass.mass_population"}),
        supported_synapse_kinds=frozenset(),
        supported_output_kinds=frozenset({
            "brainmass.DiffusiveCoupling", "brainmass.AdditiveCoupling",
        }),
        supported_rules=frozenset({
            "brainmass.CouplingMatrix", "brainmass.LaplacianCoupling",
        }),
        supported_layer_macros=frozenset(),
        supported_input_kinds=frozenset({"brainmass.mass_input"}),
    )

    handlers: dict[str, "NodeHandler"] = {}

    def build(self, ir, *, seed, dt=None, variables=None, **opts):
        ctx = BuildContext(
            ir=ir, seed=seed, dt=dt, variables=variables, opts=opts,
        )
        for node in ir.nodes():
            handler = self.handlers.get(node.KIND)
            if handler is None:
                raise BackendCapabilityError(
                    f"brainmass backend has no handler for node kind "
                    f"{node.KIND!r} (id={node.id!r}); use the brainmass "
                    f"DSL verbs (mass_population, couple, observe_forward, "
                    f"mass_input) instead of the SNN ones for whole-brain "
                    f"models"
                )
            handler(ctx, node)
        return ctx.finalize_simulator()


BACKEND = BrainmassSim()
build = BACKEND.build


_BRAINMASS_DEFAULTS = dict(solver="dopri5", delay_buffer_size=None)
_BRAINMASS_SOLVERS  = {"dopri5", "tsit5", "heun", "euler", "exp_euler"}

@register_handler(BACKEND, MassPopulation.KIND)
def _build_mass_population(ctx, node):
    # Numerical-realization choices come from node_options, not from
    # the IR node itself (the IR carries only model & dynamics).
    opts = {**_BRAINMASS_DEFAULTS,
            **ctx.kind_options(MassPopulation.KIND),
            **ctx.node_options(node.id)}
    solver       = opts["solver"]
    buffer_size  = opts["delay_buffer_size"]
    if solver not in _BRAINMASS_SOLVERS:
        raise BackendCapabilityError(
            f"brainmass backend: unknown solver {solver!r} on "
            f"MassPopulation {node.id!r}; supported: {sorted(_BRAINMASS_SOLVERS)}"
        )

    model_cls = ctx.resolve_kind(node.model.kind)      # e.g. bm.WilsonCowanStep
    model = model_cls(
        size=node.size,
        solver=solver,
        **ctx.resolve_params(node.model.params),
    )
    if node.noise is not None:
        noise_cls = ctx.resolve_kind(node.noise.kind)
        model.attach_noise(noise_cls(**ctx.resolve_params(node.noise.params),
                                     seed=ctx.seed_for(node, "noise")))
    if node.state_init:
        model.initialize(**node.state_init)
    if buffer_size is not None:
        model.allocate_delay_buffer(buffer_size)
    ctx.register_mass_pop(node.id, model)


@register_handler(BACKEND, CouplingNode.KIND)
def _build_coupling(ctx, node):
    pre_pop  = ctx.mass_pop(node.pre)
    post_pop = ctx.mass_pop(node.post)
    matrix_params = ctx.resolve_params(node.matrix.params)
    W      = jnp.asarray(matrix_params["W"])
    delays = matrix_params.get("delays")
    kernel = ctx.resolve_kind(node.kernel.kind)(
        **ctx.resolve_params(node.kernel.params)
    )
    scale = ctx.resolve_value(node.scale, default=1.0)
    ctx.register_coupling(node.id,
        bm.coupling.build(pre_pop, post_pop, W=W, delays=delays,
                          kernel=kernel, scale=scale))


@register_handler(BACKEND, ForwardModelObservable.KIND)
def _build_forward(ctx, node):
    target_pop = ctx.mass_pop(node.target)
    fwd_cls = ctx.resolve_kind(node.forward_model.kind)
    fwd = fwd_cls(**ctx.resolve_params(node.forward_model.params))
    lead = None
    if node.lead_field is not None:
        lf_cls = ctx.resolve_kind(node.lead_field.kind)
        lead = lf_cls(**ctx.resolve_params(node.lead_field.params))
    ctx.register_observable(node.id,
        bm.forward_model.attach(target_pop, state_var=node.state_var,
                                model=fwd, lead_field=lead,
                                every=node.every, reducer=node.reducer))


@register_handler(BACKEND, MassInput.KIND)
def _build_mass_input(ctx, node):
    target_pop = ctx.mass_pop(node.target)
    src_cls = ctx.resolve_kind(node.source.kind)
    src = src_cls(**ctx.resolve_params(node.source.params),
                  seed=ctx.seed_for(node, "input"))
    target_pop.bind_input(state_var=node.state_var, source=src)
```

The backend also registers handlers for built-in `SubNetworkNode` and
`VariableRef` resolution by delegating to the substrate (those are
not domain-specific). Built-in `PopulationNode` / `ProjectionNode` /
`InputNode` / `ObservableNode` have **no** handler on the brainmass
backend; a spec that uses them gets `BackendCapabilityError` with a
clear message. The same is true the other way: a `MassPopulation`
node has no handler on the SNN-domain `clock` backend.

### 5.10.4 Registration plumbing

```python
# brainmass/dsl/__init__.py
"""Importing this module activates the brainmass extension."""

from . import nodes           # @register_node_kind decorators run
from . import handles         # @register_view_kind decorators run
from . import verbs           # @register_builder_verb decorators run
from .backend import BACKEND, build  # backend handler decorators run

# Re-export user-facing constructors:
from .models import (
    WilsonCowan, JansenRit, Kuramoto, Hopf, StuartLandau,
    FitzHughNagumo, WongWang, MontbrioPazoRoxin, VanDerPol,
)
from .coupling import CouplingMatrix, LaplacianCoupling
from .coupling import DiffusiveCoupling, AdditiveCoupling
from .noise    import OUProcess, GaussianNoise, ColoredNoise, WhiteNoise
from .forward  import BOLDSignal, EEGLeadFieldModel, MEGLeadFieldModel
```

`WilsonCowan`, `CouplingMatrix`, etc. are thin constructor classes
that return `ModelRef("brainmass.WilsonCowan", {...})` /
`ConnRule("brainmass.CouplingMatrix", {...})` — the same pattern
the SNN domain uses with `spec.models.LIF(...)` →
`ModelRef("LIF", {...})`. Each constructor is also registered through
the existing `brainpy_state.spec.neurons` / `connectivity` /
`outputs` / `inputs` / `initializers` entry-point groups so the YAML
frontend (Chapter 4) can resolve them by name.

### 5.10.5 Entry points

```toml
# brainmass/pyproject.toml
[project.entry-points."brainpy_state.spec.extensions"]
brainmass = "brainmass.dsl:_register"

[project.entry-points."brainpy_state.backends.sim"]
brainmass = "brainmass.dsl.backend:BACKEND"

[project.entry-points."brainpy_state.spec.neurons"]
"brainmass.WilsonCowan"    = "brainmass.wilson_cowan:WilsonCowanStep"
"brainmass.JansenRit"      = "brainmass.jansen_rit:JansenRitStep"
"brainmass.Kuramoto"       = "brainmass.kuramoto:KuramotoNetwork"
"brainmass.Hopf"           = "brainmass.hopf:HopfStep"
"brainmass.StuartLandau"   = "brainmass.sl:StuartLandauStep"
"brainmass.FitzHughNagumo" = "brainmass.fhn:FitzHughNagumoStep"
"brainmass.WongWang"       = "brainmass.wong_wang:WongWangStep"
"brainmass.MontbrioPazoRoxin" = "brainmass.qif:MontbrioPazoRoxinStep"
"brainmass.VanDerPol"      = "brainmass.vdp:VanDerPolStep"

[project.entry-points."brainpy_state.spec.connectivity"]
"brainmass.CouplingMatrix"    = "brainmass.coupling:CouplingMatrixRule"
"brainmass.LaplacianCoupling" = "brainmass.coupling:LaplacianRule"

[project.entry-points."brainpy_state.spec.outputs"]
"brainmass.DiffusiveCoupling" = "brainmass.coupling:DiffusiveCoupling"
"brainmass.AdditiveCoupling"  = "brainmass.coupling:AdditiveCoupling"

[project.entry-points."brainpy_state.spec.inputs"]
"brainmass.OUProcess"     = "brainmass.noise:OUProcess"
"brainmass.GaussianNoise" = "brainmass.noise:GaussianNoise"
"brainmass.ColoredNoise"  = "brainmass.noise:ColoredNoise"
"brainmass.WhiteNoise"    = "brainmass.noise:WhiteNoise"
```

### 5.10.6 User-facing example

```python
import numpy as np
import saiunit as u

import brainpy.state.spec as spec
import brainmass.dsl as bm                  # registers extension on import

# Read structural connectivity and delay matrices:
SC     = np.load("sc_100.npy")              # (100, 100)
delays = np.load("delays_100.npy") * u.ms

net = spec.NetSpec("whole_brain_wilson_cowan")

# 100 parcels, each a Wilson–Cowan mass model.
# Noise is part of the dynamics (it enters the SDE), so it lives on
# the IR node. The solver and time step are realization choices and
# are passed to bm.build(...) below — not declared here.
regions = net.mass_population(
    "regions",
    bm.WilsonCowan(
        tau_e=10*u.ms, tau_i=20*u.ms,
        c_ee=16., c_ei=12., c_ie=15., c_ii=3.,
        # global gain is a build-time variable:
        gain=net.variable("gain", default=1.0),
    ),
    size=100,
    noise=bm.OUProcess(sigma=spec.train(0.01), tau=5*u.ms),
)

# Structural-connectivity coupling — a single first-class node, not
# a sparse projection. Trainable scale is OK; W is fixed.
net.couple(
    regions, regions,
    matrix=bm.CouplingMatrix(W=SC, delays=delays),
    kernel=bm.DiffusiveCoupling(),
    scale=spec.train(0.3, constraint="positive"),
)

# Time-series stimulus into a subset of parcels (visual cortex, 10 parcels):
visual = regions[80:90]
net.mass_input(
    visual,
    bm.PiecewiseConstant(values=stim_protocol, t_breaks=t_breaks),
    state_var="I_ext",
)

# Continuous-state observable — built-in verb, works on mass populations:
net.observe(regions.state("E"), every=10*u.ms)

# Forward-model observable — brainmass extension verb:
net.observe_forward(
    regions,
    state_var="E",
    forward_model=bm.BOLDSignal(TR=2*u.s),
    every=2*u.s,
)
net.observe_forward(
    regions,
    state_var="E",
    forward_model=bm.EEGLeadFieldModel(electrodes=64),
    lead_field=bm.LeadField(path="dk_eeg_lf.npy"),
    every=1*u.ms,
)

ir = net.finalize()                          # validates every IRNode

# Build on the brainmass backend. Numerical knobs live here, not on
# the IR — the same IR runs through different solvers without
# rebuilding the spec.
sim = bm.build(
    ir, seed=0, dt=0.1*u.ms,
    variables={"gain": 1.2},
    # Per-kind default: integrate every MassPopulation with dopri5
    # adaptive RK; allocate a 200-step delay buffer:
    kind_options={
        "brainmass.mass_population": dict(
            solver="dopri5", delay_buffer_size=200,
        ),
    },
)
trace = sim.run(60*u.s)

bold = trace.observable("fwd_regions")       # the BOLD trace
eeg  = trace.observable("fwd_regions_eeg")   # the EEG trace
```

Same observation as in §5.9.6: no `extras`, no opt-in flags. The
user writes mass-domain verbs alongside built-in verbs (`observe`,
`net.variable`, `spec.train`) and the spec finalizes to an IR that
the dedicated brainmass backend consumes. The IR is content-hashable
and round-trips through YAML the same way an SNN spec does, because
every node — `MassPopulation`, `CouplingNode`,
`ForwardModelObservable`, `MassInput` — is an `IRNode` and the
default codec covers their shapes.

### 5.10.7 Mixed-domain specs are rejected at validate time

A spec that mixes mass and spike populations finalizes — the IR
substrate is happy to hold both — but at backend `build()` the
chosen backend rejects whichever kinds it does not handle. There is
no cross-domain coupling primitive in v1: a mass-to-spike or
spike-to-mass projection is a future-RFC concern (D27). Until then,
the conservative path is per-domain isolation: brainmass models live
in brainmass-only specs, point-neuron and morphological models share
clock-compatible specs.

---

## 5.11 Constraints and invariants

### 5.11.1 Hard constraints on extensions

- **All IR nodes are `IRNode` subclasses.** No domain may add fields
  to existing node dataclasses, subclass them, or smuggle data
  through `meta`.
- **All builder verbs go through `@register_builder_verb`.** No
  monkey-patching `NetSpec` directly; no subclassing it.
- **Nodes are frozen dataclasses with `KIND` set and `id` first.**
  Default codec is generated from the dataclass shape; custom codecs
  are opt-in.
- **`validate()` sees its own node and the surrounding IR.** It must
  not read other nodes' private attributes; it interacts with the IR
  through public iteration (`ir.nodes_of`, `ir.populations`, etc.).
- **`Trainable`, `DistRef`, `VariableRef`, and `Noise` are the only
  value wrappers.** Domains use them; they do not add new ones.
  Wrapper shape changes are spec-level decisions, not extensions.
- **No mutation of built-in registries or backend handler tables for
  same-name entries.** Extensions add; they do not replace. Same-name
  collisions raise `DomainError` at activation.

### 5.11.2 Non-goals for this chapter

- **Cross-domain coupling primitives** (mass-to-spike, spike-to-mass).
  Deferred — D27. Per-backend validation rejects mixed specs until
  there is a designed cross-boundary node kind.
- **Multi-version coexistence for a single extension in one Python
  process.** A single installed version per extension per process.
- **A new value-wrapper axis** for domain-specific markers (e.g.,
  `MassParameter` to flag continuous-state parameters). Treat them
  as ordinary `ModelRef.params` leaves and let the domain's
  `validate()` enforce dimensional / structural rules.
- **A way for one extension to read another's node payloads.**
  Extensions are isolated; their nodes appear in `ir.nodes_of` but
  their `validate()` should not depend on another domain's presence.

---

## 5.12 Decision log additions

| ID  | Decision                                                                                                                                                  | Resolution |
|-----|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| D24a| Spec/backend boundary for numerical-realization knobs.                                                                                                    | **Hard rule.** The IR carries the model, dynamics, topology, and architecture — never the integrator, time-step policy, discretization, ring-buffer sizes, or accelerator placement. Those reach backends through `backend.build(..., kind_options=..., node_options=...)` and are looked up by handlers through `BuildContext.kind_options(KIND)` / `BuildContext.node_options(id)` / `BuildContext.options_for(node)`. The same `NetIR` content-hash routes through different solvers without re-`finalize()`. This rule applies equally to core nodes and to every domain extension (D25). |
| D25 | How adjacent domains (braincell, brainmass, …) extend the DSL.                                                                                            | The spec module exposes five extension protocols — `IRNode`, `ViewHandle`, builder-verb decorator, codec, backend-dispatch — and **every node in a NetIR is an `IRNode`**, built-in or otherwise. Domains register new node kinds, view handles, builder verbs, codecs, and backend handlers. Built-in SNN node kinds are themselves pre-registered instances of the same protocol; there is no special-casing between core and extension nodes once they are in the IR. The substrate is the contribution; the SNN content is one application. |
| D26 | Where braincell- and brainmass-specific code lives.                                                                                                        | **Out of tree.** No `brainpy.state.spec.morph` / `.mech`. The planned morphological surface in §3.5.3, §3.6.3, §3.9.5 is supplied by the [`braincell`](https://github.com/chaobrain/braincell) repository as `braincell.dsl`. The brainmass surface ships from [`brainmass`](https://github.com/chaobrain/brainmass) as `brainmass.dsl`. Both register through the entry-point groups in §7.5 plus `brainpy_state.spec.extensions` for the top-level activation hook. |
| D27 | Cross-domain composition (spike ↔ mass, point ↔ morphology) in one IR.                                                                                    | **Deferred.** v1 allows both to coexist in a single IR — the substrate is uniform — but no cross-domain projection / coupling node kind is shipped. Backends reject what they can't handle at `build()`. A future RFC defines a `cross_project` node kind once the semantics (rate-to-spike, spike-to-rate) are settled. |
| D28 | Should `extras` exist on built-in IR nodes?                                                                                                                | **No.** Domains add full node kinds, not opt-in side fields. The earlier proposal of namespaced `extras` is withdrawn: it forces domain semantics into core dataclasses and erodes the contract that every IR node is a typed value with its own validator. Domains that need cross-cutting metadata on a built-in population (e.g. attaching cell-level diagnostics to a `PopulationNode`) define a separate `IRNode` subclass that references the population by id. |
| D29 | What to do with the existing §3.5.3 / §3.6.3 / §3.9.5 morphology examples in Chapter 3.                                                                   | They remain — they are user-facing examples and the call-site syntax is unchanged. The introductory paragraphs note that the verbs (`net.morph_population`, `compartments`) and classes (`bc.Cell`, `bc.MorphFromSWC`, `bc.ApicalDendrite`, etc.) are contributed by the `braincell` extension. The classes do not move to `brainpy.state.spec.morph` / `.mech`; they were never going to ship from this repository under the revised design. |

---

**Previous:** [Chapter 10 — Implementation](./10-implementation.md)
**Next:** [Chapter 11 — Appendix: decisions, cheat sheet, open questions](./11-appendix.md)
