# Chapter 3 — Frontend A: Fluent NetSpec builder (Python)

> Part of the [Network Specification DSL RFC](./README.md).

## Section map

| §     | Topic                                                            | Purpose                                                       |
|-------|------------------------------------------------------------------|---------------------------------------------------------------|
| 6.1   | Design and extension model                                       | The seams along which extensions are added.                   |
| 6.2   | The `NetSpec` API                                                | The builder's full surface.                                   |
| 6.3   | Handles                                                          | Typed symbolic references; one subtype per view kind.         |
| 6.4   | Grounding example — Brunel                                       | The minimal end-to-end shape of a spec.                       |
| 6.5   | Populations                                                      | Point, spatial, and morphological populations.                |
| 6.6   | Projections, connectivity, and synapses                          | Rule sources, spatial rules, compartment-targeted rules, per-edge attributes. |
| 6.7   | Inputs, signals, schedules                                       | External drivers, third-factor / eligibility signals, learning phases / trial structure. |
| 6.8   | Observables                                                      | Recording channels with windowing and reducers.               |
| 6.9   | View algebra                                                     | Index / slice / merge / spatial / compartmental / tag / predicate views. |
| 6.10  | Value wrappers                                                   | `Trainable`, `DistRef`, `Noise` — and how they compose.       |
| 6.11  | Composition forms                                                | Subnetwork, sequential, DAG; temporal semantics; layer macros.|
| 6.12  | Plasticity                                                       | Per-projection → modulated → eligibility → scheduled → structural → homeostatic. |
| 6.13  | Construction-time errors                                         | Eager validation.                                             |
| 6.14  | Post-definition parameter modification (G12)                     | Path language, `ParamPatch`, `ParameterView`, live vs rebuild.|
| 6.15  | End-to-end example                                               | A cortex–striatum loop touching every extension.              |
| 6.16  | IR delta and forward compatibility                               | Per-extension summary of IR additions and parameter class.    |
| 6.17  | Intentionally out of scope                                       | What this chapter deliberately does not cover.                |

---

## 3.1 Design and extension model

`NetSpec` is a value-only builder. Calls register node descriptions;
nothing executes until `.finalize()` returns a frozen `NetIR`. Handles
are typed symbolic references holding an id and a back-pointer to the
builder; they hold no JAX state.

**Extension seams.** Every new feature in this chapter lands along one
of four explicit axes:

| Axis             | What it adds                                                 | Where it appears                       |
|------------------|--------------------------------------------------------------|----------------------------------------|
| **Block kind**   | A new top-level node kind (`Population`, `Projection`, `Input`, `Observable`, `Signal`, `Schedule`, …). | §3.5 – §3.8                            |
| **Value wrapper**| A new transparent wrapper around a leaf value (`Trainable`, `DistRef`, `Noise`). | §3.10                                  |
| **View kind**    | A new way to reference a sub-set of an existing node (`slice`, `merge`, `reshape`, `within(mask)`, `compartments(region)`, `where(tag)`, `filter(pred)`). | §3.9                                   |
| **Composition form**| A new aggregation primitive (`subnetwork`, `sequential`, `graph`, future `loop` / `branchstack` / …). | §3.11                                  |

A future extension belongs to *exactly one* of these axes; that
constrains where it lands in this chapter, what IR fields it may add,
and what backward-compatibility burden it carries.

**Invariants every extension must preserve.**

1. **Additive IR.** Existing fields keep their semantics; new fields
   default to "absent" and round-trip unchanged on older specs.
2. **Deterministic lowering (G4).** Same `(spec, backend, seed)` →
   bit-identical artifact, including over any newly introduced
   randomness source.
3. **Unit discipline (G3).** Every numeric leaf carries `saiunit`
   units; new value wrappers preserve them.
4. **Backend declarative classification.** Every new IR leaf has an
   explicit `LIVE` / `LIVE_RESET` / `REBUILD` class (§3.14.5). No leaf
   is "rebuilds silently."
5. **Domain-specific features are opt-in.** Spatial positions,
   morphology, and signals appear only when the user asks for them;
   non-spatial point-neuron specs are unchanged.

§3.16 summarizes the IR additions and parameter-class map for every
extension in this chapter.

---

## 3.2 The `NetSpec` API

```python
class NetSpec:
    def __init__(self, name: str, *, meta: Mapping[str, Any] | None = None): ...

    # ── Block-kind constructors ────────────────────────────────────────
    def population(
        self,
        name: str,
        model: ModelLike,
        size: int | Sequence[int] | None = None,
        *,
        batch: int | None = None,
        positions: "GeometryLike | None" = None,          # §3.5.2
        init: Mapping[str, Any | DistLike | Trainable] | None = None,
        tags: Sequence[str] = (),
    ) -> PopulationHandle: ...

    def project(
        self,
        pre: "PopulationHandle | ViewHandle | QueryHandle",
        post: "PopulationHandle | ViewHandle | QueryHandle",
        *,
        rule: ConnRuleLike,                # braintools.conn.Connectivity or supplementary
        synapse: ModelLike,
        output: ModelLike,
        # per-edge sugar (merged into rule.params at finalize):
        weight: Any | DistLike | Trainable | None = None,
        delay: Any | DistLike | Trainable | None = None,
        allow_autapses: bool | None = None,
        allow_multapses: bool | None = None,
        # plasticity:
        plasticity: ModelLike | None = None,
        structural_plasticity: ModelLike | None = None,   # §3.12.5
        plasticity_schedule: "ScheduleHandle | None" = None,  # §3.12.4
        # graph / temporal semantics:
        temporal_offset: str = "same_step",                # §3.11.4
        # bulk projection over queries (§3.9.6):
        broadcast: str = "pairwise",
        seed: int | None = None,
        name: str | None = None,
    ) -> "ProjectionHandle | tuple[ProjectionHandle, ...]": ...

    def input(
        self,
        target: "PopulationHandle | ViewHandle | QueryHandle",
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
        during: "ScheduleWindow | None" = None,            # §3.8.2
        name: str | None = None,
    ) -> ObservableHandle: ...

    def signal(                                            # §3.7.2
        self,
        name: str,
        *,
        source: "SignalSourceLike",
    ) -> SignalHandle: ...

    def schedule(                                          # §3.7.3
        self,
        name: str,
        body: "ScheduleBody",
    ) -> ScheduleHandle: ...

    def attach_schedule(self, schedule: ScheduleHandle) -> None: ...

    def attach_plasticity(                                 # §3.12.6
        self,
        target: "PopulationHandle | ViewHandle | QueryHandle",
        plasticity: ModelLike,
    ) -> None: ...

    # ── Composition forms (§3.11) ──────────────────────────────────────
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

    def graph(
        self,
        name: str,
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> "GraphBuilder": ...

    def fork(
        self,
        source: "LayerHandle | PopulationHandle",
        *,
        branches: Sequence["SequentialLike"],
        merge: "LayerLike",
    ) -> "LayerHandle": ...

    def group(
        self,
        name: str,
        members: Sequence["PopulationHandle | ViewHandle"],
        *,
        tags: Sequence[str] = (),
    ) -> GroupHandle: ...

    # ── View queries (§3.9.6) ──────────────────────────────────────────
    def where(self, *, tag: str | Sequence[str], mode: str = "any",
              require_nonempty: bool = True) -> "TagQueryHandle": ...
    def filter(self, predicate: Callable[..., bool], *,
               require_nonempty: bool = True) -> "PredicateQueryHandle": ...

    # ── Tag management (§3.9.6) ───────────────────────────────────────
    def tag(self, target, *tags: str) -> None: ...
    def untag(self, target, *tags: str) -> None: ...
    def tag_where(self, query: "QueryHandle", *tags: str) -> None: ...

    def export(self, **handles: "PopulationHandle | ViewHandle") -> None: ...

    # ── Finalization & I/O ────────────────────────────────────────────
    def finalize(self) -> NetIR: ...
    def to_yaml(self, path: str | os.PathLike) -> None: ...
    def to_json(self, path: str | os.PathLike) -> None: ...

    @classmethod
    def from_ir(cls, ir: NetIR) -> "NetSpec": ...

    # ── Post-definition modification (§3.14) ──────────────────────────
    def update(self, path_or_mapping, value=_MISSING, /) -> "NetSpec": ...
    def with_(self, **subtrees: Any) -> "NetSpec": ...
    def patch(self, *patches: "ParamPatch") -> "NetSpec": ...
```

Module-level helpers mirror the most common handle methods:

```python
import brainpy.state.spec as spec

spec.merge(*handles)            -> MergedViewHandle
spec.split(handle, sizes)       -> tuple[ViewHandle, ...]
spec.concat(*handles, axis=0)   -> MergedViewHandle
spec.train(value, *, name=None, constraint=None, required=False) -> Trainable
spec.noise.Wiener(sigma)        -> Noise                     # §3.10.3
spec.noise.OU(sigma, tau)       -> Noise
```

---

## 3.3 Handles

```python
class PopulationHandle:
    name: str
    size: int | tuple[int, ...]
    model: ModelRef
    tags: tuple[str, ...]

    # Slicing / indexing → ViewHandle
    def __getitem__(self, key: slice | Sequence[int]) -> "ViewHandle": ...
    def reshape(self, *shape: int) -> "ViewHandle": ...
    def concat(self, *others: "PopulationHandle | ViewHandle") -> "MergedViewHandle": ...

    # Spatial sub-view (only when positions= was set; §3.9.4)
    def within(self, mask: "MaskLike") -> "SpatialViewHandle": ...

    # Compartmental sub-view (only on Cell models; §3.9.5)
    def compartments(self, region: "RegionExprLike | str") -> "CompartmentViewHandle": ...

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


class ViewHandle:                          # base for slice / indices / reshape views
    population: PopulationHandle
    # Same observable properties as PopulationHandle.


class MergedViewHandle(ViewHandle):
    members: tuple[PopulationHandle | ViewHandle, ...]


class SpatialViewHandle(ViewHandle):        # §3.9.4
    population: PopulationHandle
    mask: "MaskRef"


class CompartmentViewHandle(ViewHandle):    # §3.9.5
    population: PopulationHandle
    region: "RegionRef"


class TagQueryHandle:                       # §3.9.6
    tags: tuple[str, ...]
    mode: str                               # "any" | "all"
    def __iter__(self) -> Iterator[PopulationHandle]: ...
    def __and__(self, other): ...           # set-algebra on queries
    def __or__(self, other): ...
    def __sub__(self, other): ...


class PredicateQueryHandle:                 # §3.9.6
    predicate: Callable[..., bool]
    # same set-algebra as TagQueryHandle


class ProjectionHandle:
    name: str
    pre: ViewHandle
    post: ViewHandle
    rule: ConnRule

    @property
    def weights(self) -> ObservableLike: ...


class SignalHandle:                         # §3.7.2
    name: str
    source: ModelRef


class ScheduleHandle:                       # §3.7.3
    name: str
    body: "ScheduleBody"
    def gated_on(self, phases: Sequence[str]) -> "ScheduleHandle": ...
    def window(self, phase: str) -> "ScheduleWindow": ...


class SubNetworkHandle:
    name: str
    # exposes the inner spec's exported handles as attributes


class SequentialHandle:
    name: str
    layers: tuple[PopulationHandle, ...]
    @property
    def output(self) -> PopulationHandle: ...   # last layer's population


class GraphBuilder:                         # §3.11.3
    def input(self, source: "InputSourceLike") -> "LayerHandle": ...
    def input_handle(self, pop: "PopulationHandle | LayerHandle") -> "LayerHandle": ...
    def node(self,
             name: str,
             layer: "LayerLike",
             *,
             inputs: "LayerHandle | Sequence[LayerHandle]") -> "LayerHandle": ...
    def export(self, **outputs: "LayerHandle") -> None: ...


class GroupHandle:
    name: str
    members: tuple[PopulationHandle | ViewHandle, ...]
```

A handle hierarchy summary, ordered by extension axis:

```
                  Handle
                    │
   ┌────────────────┼─────────────────┬─────────────────┬──────────────┐
   ▼                ▼                 ▼                 ▼              ▼
PopulationHandle  ViewHandle      QueryHandle       SignalHandle  ScheduleHandle
                    │                 │
   ┌────────────────┼──────────────┐  ├── TagQueryHandle
   ▼                ▼              ▼  └── PredicateQueryHandle
MergedView   SpatialView     CompartmentView
```

Every future view subtype slots in as another `ViewHandle` leaf; every
future top-level node kind adds a new sibling handle.

---

## 3.4 Grounding example — Brunel

```python
import brainpy.state.spec as spec
import brainpy.state.clock as clock          # backend lives at brainpy.state.clock
import braintools.conn as conn
import braintools.init as init
import saiunit as u

net = spec.NetSpec("brunel")

neuron = spec.models.LIF(tau=20*u.ms, V_th=-50*u.mV,
                       V_reset=-60*u.mV, V_rest=-65*u.mV)
syn  = spec.models.Expon(tau=5*u.ms)
coba = spec.models.COBA(E=0*u.mV)

exc = net.population("exc", neuron, size=8000, tags=("excitatory",))
inh = net.population("inh", neuron, size=2000, tags=("inhibitory",))

# Canonical: weight on the rule.
net.project(exc, exc, synapse=syn, output=coba,
             rule=conn.FixedProb(prob=0.1, allow_self_connections=False,
                                 weight=0.10*u.nS))
# Sugar: weight as projection-level kwarg (merged at finalize).
net.project(exc, inh, synapse=syn, output=coba,
             rule=conn.FixedProb(prob=0.1), weight=0.10*u.nS)
net.project(inh, exc, synapse=syn, output=coba,
             rule=conn.FixedProb(prob=0.1), weight=-0.50*u.nS)
net.project(inh, inh, synapse=syn, output=coba,
             rule=conn.FixedProb(prob=0.1, allow_self_connections=False),
             weight=-0.50*u.nS)

net.input(exc, spec.input.Poisson(rate=20*u.Hz), weight=0.2*u.nS)
net.observe(exc.spikes)
net.observe(exc[:50].voltage)

ir  = net.finalize()
sim = clock.build(ir, seed=0, dt=0.1*u.ms)
out = sim.run(1*u.second)
```

Backends live as **top-level modules under `brainpy.state`**, never under
`brainpy.state.spec` — the spec module is the DSL surface and contains
no execution code. The full backend module list is in
[Chapter 5 §5.1.1](./05-backends.md#511-module-location); the rest of
this chapter imports them as `from brainpy.state import <backend>`
(e.g. `clock`, `bptt`, `eprop`, `eventprop`, `nir`).

This is the **minimal** shape: point populations, random connectivity,
Poisson drive, two observables. The rest of the chapter is what
becomes possible once each extension axis is opened.

---

## 3.5 Populations

A `population` describes `N` units of one model kind. The model kind
and the optional `positions=` / morphology determine which extensions
are reachable on the resulting handle.

### 3.5.1 Point populations

The default. `size` is an `int` or a shape tuple (used by deep-SNN
layers, e.g. `(C, H, W)`).

```python
exc = net.population("exc", spec.models.LIF(...), size=8000)
conv1 = net.population("conv1", spec.models.LIF(...), size=(16, 28, 28))
```

Initial state distributions and per-unit parameter variation come
from the value-wrapper layer (§3.10):

```python
exc = net.population("exc", neuron, size=1024,
    init={"V": spec.train(init.Uniform(low=-65*u.mV, high=-55*u.mV))})
```

### 3.5.2 Spatial populations

A spatial population carries an explicit `positions` field. This
unlocks **all distance-, kernel-, and mask-based connectivity rules**
in `braintools.conn._spatial` (`DistanceDependent`, `Gaussian`,
`Exponential`, `Ring`, `Grid2d`, `RadialPatches`) and the spatial
views in §3.9.4.

```python
# brainpy.state.spec.geometry
class Geometry:
    ndim: int                       # 1, 2, or 3
    extent: tuple[u.Quantity, ...]  # bounding box per dim
    periodic: tuple[bool, ...]      # PBC flag per dim
    def positions(self) -> u.Quantity: ...   # (N, ndim) array

# Built-in geometries:
spec.geometry.Grid1d(n, spacing, origin, periodic)
spec.geometry.Grid2d(rows, cols, spacing, origin, periodic)
spec.geometry.Grid3d(shape, spacing, origin, periodic)        # shape=(D,H,W)
spec.geometry.HexGrid2d(rows, cols, spacing, periodic)
spec.geometry.Free(positions, extent, periodic)               # arbitrary, from data
spec.geometry.Layered(plane, layers)                          # 2D-plane × layer slabs
```

Usage:

```python
v1 = net.population(
    "V1", neuron,
    positions=spec.geometry.Grid2d(rows=50, cols=50, spacing=10*u.um,
                                 periodic=(True, True)),
    tags=("visual_cortex", "L4"),
)
# size is inferred from positions.positions().shape[0]; passing both an
# explicit size and incompatible positions raises SPEC-040.

ca1 = net.population(
    "CA1", neuron,
    positions=spec.geometry.Free(positions=np.load("ca1_xyz.npy")*u.um,
                               extent=(2*u.mm, 0.5*u.mm, 0.5*u.mm)),
)
```

`Geometry` is deterministic — same kind+params yield identical position
arrays, included in the content hash.

### 3.5.3 Morphological populations

For biophysically detailed networks, the model itself carries a
morphology and a list of paint / place rules. This integrates the
`braincell.Cell` ecosystem into the spec layer.

```python
import brainpy.state.spec.morph as morph
import brainpy.state.spec.mech as mech

pyr_cell = spec.models.Cell(
    morphology=morph.from_swc("l5_pyr.swc"),     # or .from_asc, .from_neuroml
    paint=[
        (morph.AllRegion(),       mech.CableProperty(Cm=1*u.uF/u.cm**2,
                                                     Ra=100*u.ohm*u.cm,
                                                     Em=-70*u.mV)),
        (morph.SomaRegion(),      mech.Channel("INa_HH",  g_max=0.12*u.S/u.cm**2)),
        (morph.SomaRegion(),      mech.Channel("IKDR_HH", g_max=0.036*u.S/u.cm**2)),
        (morph.ApicalDendrite(),  mech.Channel("Ih",
                                  g_max=spec.train(
                                      init.LogNormal(mean=0.005*u.S/u.cm**2,
                                                      std=0.001*u.S/u.cm**2)))),
        (morph.BasalDendrite(),   mech.Channel("ICaT",
                                  g_max=2*u.mS/u.cm**2)),
    ],
    place=[
        # Persistent point processes (e.g. an intracellular electrode):
        (morph.SomaRegion(), mech.CurrentClamp(amp=0*u.nA)),
    ],
    cv_policy="per_branch",      # or "fixed_segments" with arguments
    solver="staggered",
    spike_threshold=0*u.mV,
)

pyr = net.population("L5_pyr", pyr_cell, size=500,
                      tags=("cortex", "L5", "pyramidal"))
```

A `Cell` population has two state axes — unit (`0..N-1`) and
compartment (per-cell, dictated by `cv_policy`). Per-unit parameter
variation is the same `Trainable[Initialization]` mechanism as point
populations; the backend samples one value per unit. All cells in one
population share the same morphology tree (identical compartment count
and adjacency). To mix morphologies in one projection target, declare
separate populations and use `spec.merge(...)`.

The `Cell` model has its own reachability constraints:

| Backend          | Cell-model support                                              |
|------------------|-----------------------------------------------------------------|
| `clock`          | Full. Delegates to `braincell.Cell.run`.                        |
| `event`          | Not supported. Multi-compartment is not event-driven (capability error). |
| `bptt`           | Supported when `solver` is differentiable (`staggered`, `exp_euler`); otherwise capability error. |
| `eprop`, `event-prop` | Not supported (out of paradigm scope).                     |
| `nir`            | Stripped to point with `EXPORT-NIR-LOSSY` notice.               |

§3.9.5 covers compartment-resolved views on this population.

---

## 3.6 Projections, connectivity, and synapses

A projection is `(pre, post, rule, synapse, output[, plasticity, …])`.
The rule is a `braintools.conn.Connectivity` instance and is the
single source of truth for per-edge attributes (weight, delay,
autapses, multapses) — see §2.1 in [Chapter 2](./02-ir.md). The
frontend lets you set those as `project(...)` kwargs as sugar;
conflicting values raise `SPEC-016` / `SPEC-017` at finalize.

### 3.6.1 Standard connectivity rules

Every public subclass of `braintools.conn.Connectivity` is registered
by PascalCase name (§7.1) and consumable as `rule=`:

```python
# random / regular / topological — point connectivity, no positions needed
net.project(pre, post, rule=conn.FixedProb(prob=0.1, weight=0.1*u.nS),
             synapse=syn, output=coba)
net.project(pre, post, rule=conn.AllToAll(weight=init.XavierNormal()),
             synapse=syn, output=coba)
net.project(pre, post, rule=conn.SmallWorld(k=4, p=0.1, weight=...),
             synapse=syn, output=coba)
```

### 3.6.2 Spatial rules (kernel × mask)

When both populations have `positions=`, the rule's `generate()` is
fed `pre_positions` / `post_positions` via
`braintools.conn.ConnectionResult`, and the spatial rule family
becomes available. Two styles, both supported:

```python
# (a) Use a packaged spatial rule directly.
net.project(v1, v1,
    rule=conn.Gaussian(
        sigma=80*u.um, prob_max=0.15, cutoff=240*u.um,
        weight=init.LogNormal(mean=0.1*u.nS, std=0.03*u.nS),
        delay=spec.kernel.Linear(rate=1*u.ms / (100*u.um), base=0.5*u.ms),
        allow_self_connections=False,
    ),
    synapse=syn, output=coba)

# (b) Decompose into kernel × mask (NEST-readable).
net.project(v1, v1,
    rule=conn.DistanceDependent(
        prob=spec.kernel.Gaussian(sigma=80*u.um, scale=0.15),
        weight=spec.kernel.Exponential(decay=120*u.um, peak=0.2*u.nS),
        delay=spec.kernel.Linear(rate=1*u.ms / (100*u.um), base=0.5*u.ms),
        mask=spec.mask.Annular(inner=20*u.um, outer=240*u.um, periodic=True),
    ),
    synapse=syn, output=coba)
```

Two new value-object namespaces, both registered like initializers
and connectivities:

| Module        | Members                                                                                         |
|---------------|-------------------------------------------------------------------------------------------------|
| `spec.kernel` | `Gaussian`, `Exponential`, `MexicanHat`, `Linear`, `Constant`, `Step`, `Custom(fn)`. Carry units. |
| `spec.mask`   | `Circular(radius)`, `Annular(inner, outer)`, `Rectangular(width, height)`, `Wedge(angle_lo, angle_hi)`, `Grid(rows, cols, spacing)`, `PeriodicWrap(of=mask)`. |

The same kernels and masks are reused for spatial inputs and spatial
views (§3.7.1, §3.9.4).

### 3.6.3 Compartment-targeted rules

When the post-side handle is a `CompartmentViewHandle` (§3.9.5), the
rules from `braintools.conn._compartment` route synapses onto the
selected compartments:

```python
apical = pyr.compartments(morph.ApicalDendrite())
basal  = pyr.compartments(morph.BasalDendrite())

# Apical-tuft-targeted cortico-cortical input:
net.project(other_area, apical,
    rule=conn.ApicalDendriteTargeting(
        prob=0.05,
        distance_pref=spec.kernel.Gaussian(mu=400*u.um, sigma=80*u.um),
        weight=0.2*u.nS),
    synapse=syn_AMPA, output=coba)

# Basal-dendrite-targeted thalamic input:
net.project(thal, basal,
    rule=conn.BasalDendriteTargeting(prob=0.1, weight=0.5*u.nS),
    synapse=syn_AMPA, output=coba)

# Morphology-distance rule:
net.project(pyr, pyr,
    rule=conn.MorphologyDistance(
        source_region="axon_terminal",
        target_region="basal_dendrite",
        prob_kernel=spec.kernel.Gaussian(sigma=200*u.um, scale=0.1)),
    synapse=syn, output=coba)
```

Available rules include `SomaToDendrite`, `AxonToSoma`,
`AxonToDendrite`, `DendriteToDendrite`, `ProximalTargeting`,
`DistalTargeting`, `BranchSpecific`, `MorphologyDistance`,
`DendriticTree`, `AxonalProjection`, `AxonalBranching`,
`AxonalArborization`, `TopographicProjection`, `SynapticPlacement`,
`SynapticClustering`, and the `Basal*` / `Apical*` targeting variants.

### 3.6.4 Per-edge attributes

- `weight=` and `delay=` may be scalars, `u.Quantity`, `DistRef`, an
  `Initialization` instance from `braintools.init`, or wrapped in
  `Trainable[...]`. Distributions resolve to concrete arrays at
  `ConnectionResult` materialization.
- `allow_autapses` / `allow_multapses` map to whichever flag the rule
  declares (`allow_self_connections`, `include_self_connections`, …);
  the loader canonicalizes (§2.1).
- `seed=` on `project` overrides the projection's slot in the per-build
  seed fold-in.

### 3.6.5 Plasticity slot

The `plasticity=` kwarg attaches a single per-projection rule (STDP,
Hebbian, BCM, …). §3.12 describes the full plasticity surface —
including third-factor modulators, cross-projection eligibility
traces, learning schedules, and structural plasticity.

---

## 3.7 Inputs, signals, schedules

These three node kinds drive a simulation from the outside, carry
control / modulation values across nodes, and gate learning in time.

### 3.7.1 External input sources

```python
net.input(exc, spec.input.Poisson(rate=20*u.Hz), weight=0.2*u.nS)
net.input(exc, spec.input.SpikeTimes(times=spike_table, units=u.ms))
net.input(exc, spec.input.DC(amplitude=0.5*u.nA))
net.input(exc, spec.input.Step(amplitudes=..., times=...))
net.input(exc, spec.input.AC(amp=0.2*u.nA, freq=10*u.Hz, phase=0))
net.input(image_layer, spec.input.LayerImage(shape=(1, 28, 28)))
net.input(any_pop, spec.input.DataStream(...))
```

When the target population has `positions=`, **spatial input sources**
are reachable. They mirror the kernel × mask vocabulary:

```python
net.input(v1, spec.input.SpatialDrive(
    kernel=spec.kernel.Gaussian(sigma=30*u.um, peak=0.5*u.nA),
    trajectory=spec.trajectory.Constant(center=(250*u.um, 250*u.um)),
))

net.input(v1, spec.input.SpatialPoisson(
    rate_field=spec.kernel.MexicanHat(center=(250*u.um, 250*u.um),
                                    sigma_pos=30*u.um, sigma_neg=120*u.um,
                                    peak=50*u.Hz),
))
```

### 3.7.2 Signals (third-factor and eligibility)

A **signal** is a scalar (or per-unit / per-batch) time-varying value
produced by one node and read by zero or more plasticity rules. It is
the explicit modulator graph that makes reward-modulated learning,
neuromodulation, and cross-projection eligibility expressible without
hidden global variables.

```python
# Externally driven reward (RL loop):
reward = net.signal("reward",
                     source=spec.signal.External(unit=u.dimensionless))

# Neuromodulator derived from a population's firing rate:
da = net.signal("dopamine",
                 source=spec.signal.PopulationRate(vta, tau=200*u.ms,
                                                 transfer=spec.fn.Sigmoid(beta=5)))

# Shared eligibility trace consumed by many projections:
etr = net.signal("etr",
                  source=spec.signal.EligibilityTrace(
                      kind="eprop", pre=hidden, post=hidden, tau=20*u.ms))

# Custom signal — a user-supplied callable on existing state:
ctrl = net.signal("ctrl",
                   source=spec.signal.FromState(of=ctrl_pop.rate,
                                              transform=spec.fn.Tanh()))
```

Signals appear as `SignalNode` in the IR (§3.16) and can carry
their own `Noise` value (§3.10.3) for sensor noise on a measured
reward.

### 3.7.3 Schedules (phases and trial structure)

A **schedule** is a small grammar for time-gated behaviour. It serves
two consumers: plasticity rules (gate learning on/off, §3.12.4) and
observables (record only inside a named phase, §3.8.2). This also
closes the experiment-protocol gap — ITI / stim / response windows
and multi-condition randomization are expressible directly.

```python
schedule = net.schedule("epoch",
    spec.schedule.Phases([
        spec.schedule.Phase("warmup",   duration=200*u.ms,    learning=False),
        spec.schedule.Phase("train",    duration=10*u.second, learning=True),
        spec.schedule.Phase("probe",    duration=1*u.second,  learning=False),
        spec.schedule.Phase("recovery", duration=500*u.ms,    learning=False),
    ]))

# Trial-structured (alternating ITI / stim / response):
trial = net.schedule("trial",
    spec.schedule.Trial(
        iti  = spec.schedule.Phase("iti",  duration=500*u.ms, learning=False),
        stim = spec.schedule.Phase("stim", duration=200*u.ms, learning=True),
        resp = spec.schedule.Phase("resp", duration=300*u.ms, learning=True),
        n_trials=200,
        randomize="stim_id",
    ))

net.attach_schedule(trial)                 # global default for all plasticity

# Per-projection override:
net.project(..., plasticity=stdp,
             plasticity_schedule=schedule.gated_on(["train"]))
```

---

## 3.8 Observables

### 3.8.1 Standard quantities

```python
net.observe(exc.spikes)
net.observe(exc[:50].voltage)
net.observe(exc.current)
net.observe(exc.rate, every=10*u.ms, reducer="mean")
net.observe(exc.state("g_ampa"))          # arbitrary registered state variable
net.observe(projection_handle.weights, every=1*u.second)
```

Observables also accept compartmental and query handles directly:

```python
net.observe(pyr.compartments(morph.SomaRegion()).voltage)
net.observe(pyr.compartments(morph.ApicalDendrite()).state("Ca_concentration"),
             every=1*u.ms)
net.observe(net.where(tag="excitatory").spikes, name="exc_spikes")
```

### 3.8.2 Windowing and reducers

`during=` restricts recording to a schedule phase; `reducer=` collapses
the time axis at write time.

```python
net.observe(it.voltage, every=0.5*u.ms,
             during=trial.window("stim"))                # phase-gated
net.observe(exc.spikes, reducer="rate", every=10*u.ms)  # spikes → rate
net.observe(exc.voltage, reducer="quantiles:0.1,0.5,0.9")
```

The reducer vocabulary is extensible — `mean`, `sum`, `max`, `min`,
`rate`, `quantiles:p1,p2,…`, `last`, plus user callables via
`reducer=spec.reduce.Custom(fn)`.

---

## 3.9 View algebra

A view is a typed lens onto an existing block. Every view subtype is
allowed as `pre`, `post`, `target`, or observable source. Each new
view kind in this section is reached by a constructor on a handle or
on the builder.

### 3.9.1 Slicing, indexing, reshape

```python
view = exc[:50]
view = exc[100:200]
view = exc[::2]
view = exc[[0, 1, 5, 42]]                  # explicit index set

# Reshape (only when size is a shape tuple, e.g. (C, H, W)):
view = conv1.reshape(-1)
```

### 3.9.2 Merging, splitting, concatenating

```python
all_neurons = spec.merge(exc, inh)
net.project(all_neurons, readout,
             rule=conn.AllToAll(weight=...), synapse=..., output=...)

all_neurons = exc.concat(inh)              # equivalent
e_part, i_part = spec.split(combined, sizes=[8000, 2000])
```

Properties of merged views:

- A `MergedViewHandle` is allowed as `pre`, `post`, `target`, or
  observable source.
- All members must share neuron model kind and per-unit shape
  (`SPEC-019` on mismatch).
- Projections from a merged view materialize as one `ProjectionNode`
  per member, all sharing the same `synapse` / `output` / `plasticity`
  / `rule` template.
- Observing a merge concatenates underlying values at record time;
  `TraceBundle` returns a single array.

### 3.9.3 Groups

Groups are organizational labels with no semantic effect. They live in
`compounds.groups` (§2) for tools.

```python
net.group("recurrent_core", [exc, inh], tags=("balanced_eI",))
```

### 3.9.4 Spatial views — `within(mask)`

A spatial view is "the indices of `pop` whose positions lie inside
`mask`." It requires the population to carry `positions=`.

```python
patch = v1.within(spec.mask.Circular(radius=100*u.um,
                                    center=(250*u.um, 250*u.um)))
net.observe(patch.voltage, every=1*u.ms, reducer="mean")

# As projection target:
net.project(driver, v1.within(spec.mask.Wedge(0, jnp.pi/4)),
             rule=conn.FixedProb(prob=0.5, weight=...),
             synapse=..., output=...)
```

Internally materializes as a `SpatialViewRef` (§3.16) carrying a
serialized mask; resolution to an index array happens at finalize.

### 3.9.5 Compartmental views — `compartments(region)`

A compartmental view restricts an action to one or more compartments
of a `Cell` population. Region selectors compose with `&`, `|`, `~`:

```python
soma   = pyr.compartments(morph.SomaRegion())
apical = pyr.compartments(morph.ApicalDendrite())
basal  = pyr.compartments(morph.BasalDendrite())
distal = pyr.compartments(morph.DistanceFromSoma(min=300*u.um))
combo  = pyr.compartments(morph.ApicalDendrite() &
                          morph.DistanceFromSoma(max=200*u.um))
```

Use as projection target (§3.6.3), observable source (§3.8), or
target of `attach_plasticity` for compartment-local rules.

### 3.9.6 Tag-driven and predicate-driven views — `where`, `filter`

```python
# Tag-based (single tag, any-of, all-of):
excitatory = net.where(tag="excitatory")
deep_e     = net.where(tag=("cortex", "L5", "excitatory"), mode="all")
either     = net.where(tag=("cortex", "thalamus"),         mode="any")

# Predicate-based:
big_pops = net.filter(lambda p: p.size > 1000)
lif_only = net.filter(lambda p: p.model.kind == "LIF")
named    = net.filter(lambda p: p.id.startswith("aud_"))

# Set-algebra on queries:
big_exc  = excitatory & big_pops
big_or_e = excitatory | big_pops
not_lif  = excitatory - lif_only

# Iteration — get the matched handles back:
for pop in net.where(tag="excitatory"):
    ...
```

Queries resolve **at finalize**. A population declared later that
matches the tag is captured retroactively by any projection that
referenced the query — this is the predicate analogue of merged
views.

**Tag management** (declaratively, after the fact):

```python
net.tag(pyr_pop, "cortex", "L5", "pyramidal")
net.untag(pyr_pop, "draft")
net.tag_where(net.filter(lambda p: p.size > 5000), "large")
```

**Caveats.** Predicate callables must be picklable for IR
serialization; the builder records the source text (`inspect.getsource`)
and a hash. YAML uses a small predicate sub-grammar (`tag in [...]
and size > 1000`). Empty matches raise **SPEC-043** by default; pass
`allow_empty=True` for opt-in lenience.

### 3.9.7 Broadcast semantics on queries

When `pre` or `post` is a query, `broadcast=` on `project` decides
fan-out. Each mode has a deterministic id rule for materialized
`ProjectionNode`s (`<base>__<pre>__<post>`), preserving content-hash
determinism (G4).

| Mode         | Meaning                                                                  |
|--------------|--------------------------------------------------------------------------|
| `"pairwise"` | One projection per matching pair (default for same query on both sides).  |
| `"cross"`    | Cartesian product of pre × post matches.                                  |
| `"per_pre"`  | One node per pre member; post is the merged view.                         |
| `"per_post"` | One node per post member; pre is the merged view.                         |
| `"merged"`   | Materialize merge first, then one projection (falls back to §3.9.2 semantics). |

```python
# Same rule fans out over every excitatory–excitatory pair:
net.project(excitatory, excitatory,
             rule=conn.FixedProb(prob=0.05, weight=0.05*u.nS),
             synapse=syn, output=coba,
             broadcast="pairwise")

# Cartesian thalamic × cortex routing:
net.project(net.where(tag="thalamic"), net.where(tag="cortex"),
             rule=conn.Gaussian(sigma=200*u.um, prob_max=0.1, ...),
             synapse=syn, output=coba,
             broadcast="cross")
```

---

## 3.10 Value wrappers

A value wrapper is a transparent annotation around any leaf in
`ModelRef.params`, `ConnRule.params`, `PopulationNode.init`, or input
parameters. Wrappers compose: a leaf can be `Trainable[DistRef[...]]`,
or `Trainable[<scalar with Noise>]`.

### 3.10.1 `Trainable` — gradient-bearing values

```python
neuron = spec.models.LIF(
    tau=spec.train(20*u.ms, constraint="positive"),
    V_th=-50*u.mV, V_reset=-60*u.mV, V_rest=-65*u.mV,
)

rule = conn.FixedProb(
    prob=0.1,
    weight=spec.train(init.Normal(mean=0.1*u.nS, std=0.05*u.nS),
                     name="exc_to_inh.W"),
)

exc = net.population("exc", neuron, size=1024,
    init={"V": spec.train(init.Uniform(low=-65*u.mV, high=-55*u.mV))})

net.project(exc, inh, rule=conn.FixedProb(prob=0.1),
             synapse=..., output=..., weight=spec.train(0.1*u.nS))
```

**Mapping to `brainstate.nn.Param`.** At backend build, every
`Trainable` value materializes as a `brainstate.nn.Param`:

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
| `nir`    | Same as `clock`: trainables baked as constants at export time. Original `Trainable.name` recorded in the metadata sidecar (§6.4). |

`parameters()` view:

```python
from brainpy.state import bptt
trainer = bptt.build(ir, seed=0, loss=loss_fn, dt=1*u.ms)
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

### 3.10.2 `DistRef` — initialized-at-build values

`DistRef` carries a `braintools.init.Initialization` kind + params.
The builder accepts an `Initialization` instance directly and wraps it
as `DistRef`:

```python
weight=init.LogNormal(mean=0.1*u.nS, std=0.05*u.nS)   # accepted directly
weight=spec.train(init.LogNormal(...))                   # trainable + distributed
```

`DistRef` is materialized as a concrete array (with units) at backend
build, with seed derived from the projection's fold-in chain (§9.1).

### 3.10.3 `Noise` — stochastic values

`Noise` adds in-equation noise to neuron / synapse / plasticity /
input / signal parameters. It is the answer to the "noise terms in
equations, not just stochastic inputs" gap.

```python
# brainpy.state.spec.noise
class Noise:
    kind: str                   # "Wiener" | "OU" | "Multiplicative" | "Poisson" | "Custom"
    params: Mapping[str, Any]
    seed_tag: Optional[str] = None     # override default fold-in tag

spec.noise.Wiener(sigma)              # dW = sigma * sqrt(dt) * eta
spec.noise.OU(sigma, tau)             # Ornstein–Uhlenbeck colored noise
spec.noise.Multiplicative(sigma, of)  # x' += sigma * x * eta (used on weights)
spec.noise.Poisson(rate, amp)         # shot noise — discrete kicks
spec.noise.Custom(fn)                 # user callable; must be backend-pure
```

`Noise` lives inside a model's `params` under the conventional key
`"noise": { <state_var>: Noise(...), ... }`:

```python
neuron = spec.models.LIF(
    tau=20*u.ms, V_th=-50*u.mV, V_reset=-60*u.mV, V_rest=-65*u.mV,
    noise={"V": spec.noise.OU(sigma=2*u.mV, tau=5*u.ms)},
)

adex = spec.models.AdEx(
    ...,
    noise={
        "V": spec.noise.Wiener(sigma=0.5*u.mV / u.ms**0.5),
        "w": spec.noise.OU(sigma=10*u.pA, tau=50*u.ms),
    },
)

plast = spec.plasticity.STDP(
    A_pre=0.01, A_post=-0.0105, tau_pre=20*u.ms, tau_post=20*u.ms,
    weight_noise=spec.noise.Multiplicative(sigma=0.005, of="w"),
)
```

**Where it can attach.** Any registered model declaring noise-OK on
specific state variables.

| Slot                                          | Example                                                  |
|-----------------------------------------------|----------------------------------------------------------|
| `Population.model.params["noise"]`            | Voltage / current / adaptation noise on neuron equations |
| `Synapse.params["noise"]`                     | Conductance noise (synaptic vesicle release variability) |
| `Plasticity.params["weight_noise"]`           | Weight-drift noise during learning                       |
| `InputNode.source.params["noise"]`            | Jitter on a deterministic stimulus                       |
| `Signal.params["noise"]`                      | Sensor noise on a measured signal (e.g. reward)          |

**Determinism.** Each noise term gets a seed stream
`fold_in(build_seed, "noise/<scope>/<var>")` so runs are bit-identical
given the same build seed (G4). `meta["noise_terms"]` enumerates every
insertion point for the determinism contract.

**SDE integrator selection.** When any noise term is present, the
backend automatically picks an SDE-compatible integrator
(Euler-Maruyama, stochastic Heun, or Milstein depending on capability)
and logs `InfoNotice("SDEIntegratorChosen", scheme="euler_maruyama")`.
The choice is not user-configurable from the spec — that would
violate G1 ("describe what, not how to step").

**Backend classification.**

- **Live:** `sigma`, `tau` parameters of any `Noise`.
- **Live-reset:** `seed_tag` (re-folds the stream).
- **Rebuild:** noise `kind` (changes integrator selection).
- **NIR export:** stripped with `EXPORT-NIR-LOSSY` notice; sidecar
  records parameters so hardware-specific noise sources can be wired
  manually.

### 3.10.4 Composition rules

- `Trainable[DistRef[...]]` — learnable parameter initialized from a
  distribution. Common for weights.
- `Trainable[<scalar>]` — learnable scalar parameter.
- `<scalar with Noise>` — deterministic mean with intrinsic noise.
- `Trainable` cannot wrap `Noise` directly; noise is a runtime
  modification, not a learnable leaf. (Hyperparameters of `Noise`,
  e.g. `sigma`, may themselves be `Trainable`.)

---

## 3.11 Composition forms

### 3.11.1 Subnetwork — parameterized inner spec

```python
def column_spec(N: int, *, name: str) -> spec.NetSpec:
    s = spec.NetSpec(name)
    E = s.population("E", spec.models.LIF(...), size=int(0.8*N))
    I = s.population("I", spec.models.LIF(...), size=int(0.2*N))
    s.project(E, I, ...)
    s.project(I, E, ...)
    s.export(E=E, I=I)
    return s

net = spec.NetSpec("multi_column")
cols = [net.subnetwork(f"col_{k}", column_spec, N=1000) for k in range(4)]
for a, b in zip(cols, cols[1:]):
    net.project(a.E, b.E,
                rule=conn.FixedProb(prob=0.01, weight=...),
                synapse=..., output=...)
```

A `SubNetworkHandle` exposes the inner spec's exported handles
(`a.E`, `a.I`) as attributes. Each `subnetwork` call materializes its
own `SubNetworkNode` with the supplied params.

### 3.11.2 Sequential — linear stacks for deep SNNs

```python
net = spec.NetSpec("spiking_mnist", meta={"batch": 64})

stack = net.sequential(
    "encoder",
    [
        spec.input.LayerImage(shape=(1, 28, 28)),
        spec.layer.Conv2d(out_channels=16, kernel=3,
                        neuron=spec.models.LIF(tau=10*u.ms,
                            V_th=-50*u.mV, V_reset=-60*u.mV, V_rest=-65*u.mV),
                        weight=spec.train(init.KaimingNormal())),
        spec.layer.MaxPool2d(kernel=2),
        spec.layer.Conv2d(out_channels=32, kernel=3,
                        neuron=spec.models.LIF(tau=10*u.ms,
                            V_th=-50*u.mV, V_reset=-60*u.mV, V_rest=-65*u.mV),
                        weight=spec.train(init.KaimingNormal())),
        spec.layer.Flatten(),
        spec.layer.Linear(out=10,
                        neuron=spec.models.LeakyRateReadout(),
                        weight=spec.train(init.XavierNormal())),
    ],
)

net.observe(stack.output.rate)
ir = net.finalize()

from brainpy.state import bptt
trainer = bptt.build(ir, seed=0, dt=1*u.ms,
                     loss=spec.loss.cross_entropy)
```

`net.sequential(name, layers)` returns a `SequentialHandle` whose
`.output` is the `PopulationHandle` of the last layer. Each entry is
one of:

- A `LayerSpec` macro (`Conv2d`, `Linear`, `MaxPool2d`, `AvgPool2d`,
  `Flatten`, `BatchNorm`, `Dropout`, `LeakyRateReadout`, `Add`,
  `Concat`, …; §3.11.5).
- A `PopulationHandle` declared earlier — inserted verbatim.
- A bare callable returning a `LayerSpec` (late binding).

Stateless layers (`Flatten`, `MaxPool2d`, `AvgPool2d`, `Dropout`)
materialize as `ProjectionNode`s with a synapse of kind `"Identity"`
and a parameterized connectivity rule (`Pool2d`, `Reshape`, …) but no
`Population` of their own.

**Recurrence via self-projection.** A self-projection on a layer's
output is the canonical way to express recurrence in a linear stack:

```python
core = net.sequential("rsnn", [
    spec.layer.Linear(out=512, neuron=spec.models.ALIF(...),
                    weight=spec.train(init.XavierNormal())),
])
net.project(core.output, core.output,
             rule=conn.Random(prob=0.1,
                              weight=spec.train(init.Normal(std=0.05))),
             synapse=spec.models.Expon(tau=5*u.ms),
             output=spec.models.CUBA())
```

### 3.11.3 DAG — `net.graph` and merge layers

Modern deep SNNs have skip connections, parallel branches, and
multi-input ops. `net.graph` is the explicit DAG form:

```python
g = net.graph("encoder", meta={"batch": 64})

img = g.input(spec.input.LayerImage(shape=(1, 28, 28)))
c1  = g.node("conv1", spec.layer.Conv2d(out=32, kernel=3, neuron=lif), inputs=img)
p1  = g.node("pool1", spec.layer.MaxPool2d(2),                          inputs=c1)
c2  = g.node("conv2", spec.layer.Conv2d(out=64, kernel=3, neuron=lif), inputs=p1)
c3  = g.node("conv3", spec.layer.Conv2d(out=64, kernel=3, neuron=lif), inputs=c2)

# Skip — 1x1 projection of p1 added to c3:
proj = g.node("skip_proj", spec.layer.Conv2d(out=64, kernel=1, neuron=identity),
              inputs=p1)
skip = g.node("skip_add", spec.layer.Add(), inputs=[c3, proj])

flat = g.node("flat",   spec.layer.Flatten(),                    inputs=skip)
out  = g.node("logits", spec.layer.Linear(out=10, neuron=readout),
              inputs=flat)
g.export(output=out)
```

`g.node(name, layer, inputs)` accepts:

- A single `LayerHandle` → single-input layer macro.
- A list of `LayerHandle` → multi-input merge layer (`Add`, `Concat`,
  `Mul`, `Gate`).

**Merge-arity layer macros:**

| Macro            | Arity | Semantics                                                  |
|------------------|-------|------------------------------------------------------------|
| `Add()`          | N→1   | Elementwise sum (shapes must match).                       |
| `Concat(axis=-1)`| N→1   | Concatenate along axis.                                    |
| `Mul()`          | N→1   | Elementwise product (gating).                              |
| `Gate(activation=Sigmoid())` | 2→1 | Multiplicative gate: `x * activation(y)`.        |
| `Split(sizes)`   | 1→N   | Split along axis. Returns N handles.                       |
| `Tee()`          | 1→N   | Identity broadcast: send same handle to many consumers.    |

`Split` and `Tee` keep fan-out explicit — the graph stays a directed
acyclic graph, not implicit broadcast.

**Sugar — `fork` and `|`:**

```python
# fork: one input, multiple parallel branches, merge at the end
block = net.fork(
    p1,
    branches=[
        spec.sequential(layers=[spec.layer.Conv2d(out=64, kernel=3, neuron=lif),
                              spec.layer.Conv2d(out=64, kernel=3, neuron=lif)]),
        spec.sequential(layers=[spec.layer.Conv2d(out=64, kernel=1, neuron=identity)]),
    ],
    merge=spec.layer.Add(),
)

# Operator sugar (returns a graph node):
out = (c1 | spec.layer.Conv2d(out=64, kernel=3, neuron=lif)
          | spec.layer.MaxPool2d(2))
```

### 3.11.4 Temporal semantics

`ProjectionNode.temporal_offset` makes the time-discretization
explicit for cyclic graphs:

| Value           | Meaning                                                            |
|-----------------|--------------------------------------------------------------------|
| `"same_step"`   | Default. Allowed only when the projection is acyclic in the unrolled DAG. |
| `"next_step"`   | One-timestep delay; required for cycles.                           |

```python
# Top-down feedback that closes a loop — must be next-step:
net.project(c3.output, c1.output,
             rule=conn.AllToAll(weight=spec.train(init.XavierNormal())),
             synapse=spec.models.Expon(tau=10*u.ms),
             output=spec.models.CUBA(),
             temporal_offset="next_step")
```

A same-step cycle raises `SPEC-042` at finalize.

### 3.11.5 Layer macro registry

The v1 set covers deep-SNN essentials. Third-party macros register
via the `brainpy_state.spec.layers` entry point (§7.5).

| Macro                  | Connectivity rule used internally       | Stateful? |
|------------------------|------------------------------------------|-----------|
| `Linear(out)`          | `braintools.conn.AllToAll`              | yes (neuron pop) |
| `Conv1d(...)`          | supplementary `Conv1dKernel`            | yes              |
| `Conv2d(...)`          | `braintools.conn.Conv2dKernel`          | yes              |
| `MaxPool2d(...)`       | supplementary `Pool2d(kind="max")`      | no               |
| `AvgPool2d(...)`       | supplementary `Pool2d(kind="avg")`      | no               |
| `Flatten()`            | supplementary `Reshape(target=-1)`      | no               |
| `BatchNorm()`          | supplementary `BatchNorm`               | yes (running stats) |
| `Dropout(p)`           | supplementary `Dropout(p)`              | no (rng state)   |
| `LeakyRateReadout(out)`| `AllToAll`, neuron=`LeakyRateReadout`   | yes              |
| `Add` / `Concat` / `Mul` / `Gate` / `Split` / `Tee` | merge layers (§3.11.3) | no |

---

## 3.12 Plasticity

Plasticity in this DSL has six concentric layers, each strictly more
expressive than the previous. Most users only need 6.12.1; the rest
exist so that no class of learning rule forces a backend rewrite.

### 3.12.1 Per-projection rules

The default. Pass any registered plasticity model as `plasticity=`:

```python
stdp = spec.plasticity.STDP(tau_pre=20*u.ms, tau_post=20*u.ms,
                          A_pre=0.01, A_post=-0.0105,
                          w_min=0*u.nS, w_max=1*u.nS)

net.project(pre, post,
             rule=conn.FixedProb(prob=0.1, weight=...),
             synapse=syn, output=coba,
             plasticity=stdp)
```

The same slot accepts `Hebbian`, `BCM`, `Oja`, and any user-registered
plasticity rule.

### 3.12.2 Modulated plasticity (third-factor)

A plasticity rule's `modulators=` mapping binds signal handles
(§3.7.2) to role names declared by the rule kind:

```python
reward = net.signal("reward",
                     source=spec.signal.External(unit=u.dimensionless))
da     = net.signal("dopamine",
                     source=spec.signal.PopulationRate(vta, tau=200*u.ms))

stdp_da = spec.plasticity.RewardModulatedSTDP(
    tau_pre=20*u.ms, tau_post=20*u.ms,
    A_pre=0.01, A_post=-0.0105,
    modulators={"reward": reward, "baseline": 1.0},
)

net.project(cortex, striatum,
             rule=conn.FixedProb(prob=0.1,
                                 weight=init.Uniform(0.05, 0.15)*u.nS),
             synapse=syn, output=coba,
             plasticity=stdp_da)
```

The modulator graph is **explicit** in the IR — every signal is
declared, every read is named — so backends and visualization tools
can render it.

### 3.12.3 Cross-projection eligibility traces

A trace produced by one pair of populations can gate updates on a
different projection. Two roles:

```python
# Eligibility owned by one projection (the trace producer):
recurrent = net.project(hidden, hidden, ...,
    plasticity=spec.plasticity.EligibilitySource(
        name="etr", kind="eprop", tau=20*u.ms))

# Read by another (the consumer):
readout = net.project(hidden, out, ...,
    plasticity=spec.plasticity.EligibilityConsumer(
        trace="etr",                          # name reference
        lr=spec.train(1e-3, constraint="positive"),
        modulators={"reward": reward}))
```

`EligibilityConsumer(trace=name)` must resolve to exactly one
`EligibilitySource(name=name)` — otherwise **SPEC-041**.

### 3.12.4 Phased and trial-structured learning

A schedule (§3.7.3) is attached globally with `spec.attach_schedule`,
or per-projection via `plasticity_schedule=`:

```python
trial = net.schedule("trial", spec.schedule.Trial(
    iti  = spec.schedule.Phase("iti",  duration=500*u.ms, learning=False),
    stim = spec.schedule.Phase("stim", duration=200*u.ms, learning=True),
    resp = spec.schedule.Phase("resp", duration=300*u.ms, learning=True),
    n_trials=2000, randomize="stim_id"))

net.attach_schedule(trial)   # global default for all plasticity

# Per-projection override (only learn during "stim"):
net.project(it, str_d1, ...,
             plasticity=stdp_da,
             plasticity_schedule=trial.gated_on(["stim"]))
```

### 3.12.5 Structural plasticity

Edge sets become time-varying via a dedicated `structural_plasticity=`
slot:

```python
net.project(exc, exc,
    rule=conn.FixedProb(prob=0.1, weight=...),
    synapse=..., output=...,
    structural_plasticity=spec.plasticity.Structural(
        kind="rewire_random",
        period=1*u.second,            # evaluate every 1s of sim time
        activity_target=10*u.Hz,      # homeostatic target rate
        rate_window=500*u.ms,
        max_prob_change=0.01,
        prune_below=0.005*u.nS,
        grow_to_prob=0.1,
        mode="rebuild",               # or "live_topology" (requires capability)
    ),
)
```

Structural plasticity is **`REBUILD` by default**. A rule with
`mode="live_topology"` asks the backend to support in-place edge
mutation; only backends declaring `STRUCTURAL_PLASTICITY` accept it.
This isolates the cost — most users still benefit from the
fixed-topology fast path.

### 3.12.6 Homeostatic and meta-plasticity

Both expressible as plasticity rules consuming signals, attachable
post-hoc (per-population, no projection target needed):

```python
# Synaptic scaling on every excitatory population:
for pop in net.where(tag="excitatory"):
    net.attach_plasticity(pop,
        spec.plasticity.SynapticScaling(
            target_rate=5*u.Hz, tau=1*u.minute,
            activity=spec.signal.PopulationRate(pop, tau=10*u.second)))

# BCM with sliding threshold (meta-plasticity):
bcm = spec.plasticity.BCM(
    threshold=spec.signal.RunningMean(of=post.rate, tau=10*u.second),
    lr=1e-3)
net.project(..., plasticity=bcm)
```

---

## 3.13 Construction-time errors

`NetSpec` raises eagerly on:

- Duplicate population / projection / observable / signal / schedule name.
- Population name already used as a Python attribute on the builder.
- Sliced / indexed / reshaped view referencing a population not yet
  declared.
- `rule` not a `braintools.conn.Connectivity` instance (or registered
  supplementary rule).
- Pre and post sizes incompatible with `rule` (e.g. `OneToOne` and
  `n_pre != n_post`).
- `weight` / `delay` / autapse-flag set both as projection sugar and
  on the rule with conflicting values (SPEC-016 / SPEC-017).
- Unit dimension mismatch between `weight`, `synapse` input, and
  `output` expected dimensions.
- Merged view with incompatible member shapes / models (SPEC-019).
- Sequential layer shape mismatch (SPEC-020).
- `Trainable.required=True` on a non-trainable slot (SPEC-018).
- Spatial rule used on a population without `positions=` (SPEC-040).
- `EligibilityConsumer` referring to a missing trace (SPEC-041).
- Same-step projection forming a cycle (SPEC-042).
- Empty tag / predicate query when `require_nonempty=True` (SPEC-043).

Errors point at the offending Python source line.

---

## 3.14 Post-definition parameter modification (G12)

The spec is the source of truth for *what* a network is; users often
need to change *values* without rewriting the spec. There are two
settings:

- **Pre-build (offline)** — modify a `NetSpec` or a `NetIR` before
  any backend has materialized it. The IR stays frozen; mutations
  return a new IR. Same content hash whenever the same edits are
  applied.
- **Post-build (live)** — modify parameter values on an already-built
  `Simulator` / `Trainer` without rebuilding. Trainers write to these
  during gradient descent; users can also read and write them
  imperatively.

Both settings share one **path language** and one **patch type**.

### 3.14.1 Path language

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
populations.exc.model.noise.V.sigma           # leaf inside a Noise wrapper
populations.exc.init.V
populations.V1.positions.params.spacing       # leaf inside a Geometry
projections[0].rule.weight
projections[0].rule.prob
projections[0].temporal_offset
projections[2].synapse.tau
inputs[0].source.rate
signals.dopamine.source.tau
schedules.trial.body.phases[1].duration
sequentials.encoder.layers[1].weight
graphs.encoder.nodes[3].layer.kernel
subnetworks.col_0.params.N
meta.author
```

`NetIR.select(path)` reads any addressable leaf; `NetIR.update(path,
value)` returns a new `NetIR` with that leaf replaced. Wildcards
(`projections[*].rule.weight`) are supported in `update` to broadcast
a single value to many leaves.

### 3.14.2 The `ParamPatch` type

A patch is a value-level description of a change. It is
JSON-serializable and round-trips through YAML/JSON, so the same
patch can be applied to a spec, an IR, or a running backend.

```python
@dataclass(frozen=True)
class ParamPatch:
    path: str                          # dotted path; wildcards allowed
    value: Any                         # scalar | u.Quantity | DistRef | Trainable | Noise | array
    op:    str = "set"                 # "set" | "scale" | "add" | "replace_with_trainable"
    where: Optional[Callable[..., bool]] = None    # predicate-driven (§3.14.8)
    label: Optional[str] = None        # free-form annotation for logs / sweeps
```

`op` lets a patch describe a transformation rather than a literal
value:

- `"set"` — replace the value at `path`.
- `"scale"` — multiply (for numeric / Quantity leaves).
- `"add"` — additive shift (for numeric / Quantity leaves).
- `"replace_with_trainable"` — wrap a static value with `Trainable[...]`
  (or unwrap one if applied to a `Trainable`).

### 3.14.3 Pre-build mutation — four equivalent forms

```python
# 1.1 Single-path .update(path, value):
net2 = net.update("populations.exc.model.tau", 25*u.ms)

# 1.2 Mapping .update({path: value, ...}):
net2 = net.update({
    "populations.exc.model.tau": 25*u.ms,
    "projections[0].rule.weight": 0.2*u.nS,
})

# 1.3 .with_() — Pythonic, builds the mapping for you:
net2 = net.with_(populations={"exc": {"model": {"tau": 25*u.ms}}})

# 1.4 .patch(*patches) — richer operations:
net2 = net.patch(
    spec.ParamPatch("populations.exc.model.tau", 25*u.ms),
    spec.ParamPatch("projections[*].rule.weight", 2.0, op="scale"),
    spec.ParamPatch("populations.inh.model.tau", None,
                        op="replace_with_trainable"),
)
```

All four are immutable — `net` is unchanged, `net2` is a new builder
with the edits applied. The corresponding methods on `NetIR` have
identical semantics. The YAML loader's `overrides=` kwarg (§4.4) is
exactly `NetIR.update(mapping)` after schema parsing.

### 3.14.4 Post-build mutation — `ParameterView`

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
from brainpy.state import clock
sim = clock.build(ir, seed=0, dt=0.1*u.ms)

sim.parameters.get("populations.exc.model.tau")            # 20 ms
sim.parameters.set("populations.exc.model.tau", 25*u.ms)   # live update
sim.parameters.apply(
    spec.ParamPatch("projections[*].rule.weight", 1.5, op="scale"),
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
from brainpy.state import bptt
trainer = bptt.build(ir, seed=0, loss=loss_fn, dt=1*u.ms)
W = trainer.parameters.get("projections[0].rule.weight")   # a ParamState
trainer.parameters.set("projections[0].rule.weight", new_W_array)
```

### 3.14.5 Live vs rebuild changes

Not every parameter can be changed in-place. The backend classifies
each leaf as one of:

| Class       | Examples                                                                          | Behavior on `.set()`                              |
|-------------|-----------------------------------------------------------------------------------|---------------------------------------------------|
| `LIVE`      | Scalar model parameters (`tau`, `V_th`, `R`); per-edge weight / delay arrays; `Trainable` values; `Noise` `sigma` / `tau`; schedule phase durations; tag metadata. | Update the underlying state in place; cheap. |
| `LIVE_RESET`| Initial-state distributions (`init.V`); RNG-seeded leaves; positions array; schedule attach / detach. | Update + reset the corresponding state variables; cheap. |
| `REBUILD`   | Population `size`; connectivity rule kind or hyperparameters (`prob`, `K`); synapse / output / plasticity `kind`; sequential layer membership; merge-view structure; modulator binding; morphology / paint / place; `Noise` kind; structural-plasticity on/off; `temporal_offset`. | Raise `ParameterChangeRequiresRebuild(path)` (SPEC-024). User must edit the IR and call `backend.build(new_ir, ...)` again. |

The `ParameterView.set()` method consults the backend's
parameter-class table (declared next to its `BackendCapabilities`). A
`REBUILD` change never silently re-samples connectivity or reallocates
state.

A convenience helper covers the common rebuild path:

```python
sim2 = sim.rebuild_with(net.update("populations.exc.size", 16000))
# Equivalent to:
#   ir2  = sim.ir.update("populations.exc.size", 16000)
#   sim2 = clock.build(ir2, seed=sim.seed, dt=sim.dt)
# State is migrated where shapes are unchanged; reset otherwise.
```

### 3.14.6 YAML / CLI ergonomics

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
brainpy patch brunel.netspec.yaml --from brunel.patch.yaml -o brunel-v2.yaml
brainpy run   brunel.netspec.yaml --patch brunel.patch.yaml --backend clock --duration "1 s"
brainpy build brunel.netspec.yaml --patch brunel.patch.yaml --backend nir -o brunel.nir
```

### 3.14.7 Determinism and round-trip with patches

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

### 3.14.8 Predicate-driven patches

`ParamPatch.where=` extends the path language with a predicate over
the containing node. The `path` is then applied to every leaf whose
node matches:

```python
# Make all LIF populations faster:
net.patch(spec.ParamPatch(
    where=lambda node: getattr(node.model, "kind", None) == "LIF",
    path="tau",
    value=15*u.ms,
    label="speed_up_all_LIFs",
))

# Scale every learnable weight by 0.5 across all projections:
net.patch(spec.ParamPatch(
    where=lambda node: isinstance(node.rule.params.get("weight"), Trainable),
    path="rule.weight.value",
    value=0.5, op="scale",
))
```

Predicates have the same picklability constraints as
`net.filter(...)` (§3.9.6). YAML uses the same predicate sub-grammar.

---

## 3.15 End-to-end example

A 4-area cortex–striatum loop with:

- **Spatial layout** — V1 is a 2D grid; V4 has free positions from a
  connectome file; striatum is a 3D volume.
- **Morphological cells** — L5 pyramidals in IT are multi-compartment
  with apical-tuft-targeted feedback.
- **Plasticity** — cortico-striatal projections use reward-modulated
  STDP with cross-projection eligibility, gated by trial structure.
- **Noise** — V1 has membrane-voltage OU noise.
- **DAG** — visual stream uses a ResNet-like encoder.
- **Predicate-driven views** — homeostatic scaling applies to every
  excitatory population by tag.

```python
import brainpy.state.spec as spec
import brainpy.state.spec.morph as morph
import brainpy.state.spec.mech as mech
import braintools.conn as conn
import braintools.init as init
import saiunit as u

net = spec.NetSpec("cortex_striatum_loop", meta={"batch": 8})

# ── 1. Populations with spatial layout ──────────────────────────────
lif = spec.models.LIF(tau=20*u.ms, V_th=-50*u.mV,
                    V_reset=-60*u.mV, V_rest=-65*u.mV,
                    noise={"V": spec.noise.OU(sigma=1.5*u.mV, tau=5*u.ms)})

v1 = net.population("V1", lif,
        positions=spec.geometry.Grid2d(rows=50, cols=50, spacing=10*u.um,
                                     periodic=(True, True)),
        tags=("cortex", "V1", "excitatory"))

v4 = net.population("V4", lif,
        positions=spec.geometry.Free(positions=load_v4_xyz(),
                                    extent=(1*u.mm,)*2),
        tags=("cortex", "V4", "excitatory"))

# ── 2. Morphological L5 pyramidals in IT ────────────────────────────
pyr_cell = spec.models.Cell(
    morphology=morph.from_swc("l5_pyr.swc"),
    paint=[
        (morph.AllRegion(),      mech.CableProperty(Cm=1*u.uF/u.cm**2,
                                                     Ra=100*u.ohm*u.cm)),
        (morph.SomaRegion(),     mech.Channel("INa_HH",  g_max=0.12*u.S/u.cm**2)),
        (morph.SomaRegion(),     mech.Channel("IKDR_HH", g_max=0.036*u.S/u.cm**2)),
        (morph.ApicalDendrite(), mech.Channel("Ih",
                                  g_max=spec.train(
                                      init.LogNormal(mean=0.005*u.S/u.cm**2,
                                                      std=0.001*u.S/u.cm**2)))),
    ],
    solver="staggered")

it = net.population("IT", pyr_cell, size=500,
                     tags=("cortex", "IT", "excitatory", "L5_pyramidal"))

# ── 3. Striatum, 3D ─────────────────────────────────────────────────
msn = spec.models.LIF(tau=10*u.ms, V_th=-45*u.mV,
                    V_reset=-55*u.mV, V_rest=-70*u.mV)
str_d1 = net.population("Str_D1", msn,
            positions=spec.geometry.Grid3d(shape=(10, 20, 20), spacing=20*u.um),
            tags=("striatum", "D1", "inhibitory"))

# ── 4. Reward / dopamine signals ────────────────────────────────────
reward = net.signal("reward",
                     source=spec.signal.External(unit=u.dimensionless))
da = net.signal("dopamine",
                 source=spec.signal.PopulationRate(net.where(tag="VTA"),
                                                  tau=200*u.ms,
                                                  transfer=spec.fn.Sigmoid(beta=5)))

# ── 5. Visual stream — DAG encoder with skip ────────────────────────
net.input(v1, spec.input.LayerImage(shape=(1, 28, 28)),
           weight=spec.train(init.KaimingNormal()))

enc = net.graph("ResNet_encoder")
x  = enc.input_handle(v1)
c1 = enc.node("c1",    spec.layer.Conv2d(out=64,  kernel=3, neuron=lif), inputs=x)
p1 = enc.node("p1",    spec.layer.MaxPool2d(2),                          inputs=c1)
c2 = enc.node("c2",    spec.layer.Conv2d(out=128, kernel=3, neuron=lif), inputs=p1)
c3 = enc.node("c3",    spec.layer.Conv2d(out=128, kernel=3, neuron=lif), inputs=c2)
sk = enc.node("sk",    spec.layer.Conv2d(out=128, kernel=1, neuron=lif), inputs=p1)
mg = enc.node("merge", spec.layer.Add(),                                  inputs=[c3, sk])
enc.export(output=mg)
net.project(enc.output, v4,
             rule=conn.FixedProb(prob=0.1,
                                 weight=spec.train(init.XavierNormal())),
             synapse=spec.models.Expon(tau=5*u.ms),
             output=spec.models.COBA(E=0*u.mV))

# ── 6. Spatial / compartmental projections ──────────────────────────
net.project(v1, v1,
    rule=conn.Gaussian(sigma=80*u.um, prob_max=0.15, cutoff=240*u.um,
                       weight=0.05*u.nS, allow_self_connections=False),
    synapse=spec.models.Expon(tau=5*u.ms),
    output=spec.models.COBA(E=0*u.mV))

net.project(v4, it.compartments(morph.ApicalDendrite()),
    rule=conn.ApicalDendriteTargeting(prob=0.05,
        distance_pref=spec.kernel.Gaussian(mu=400*u.um, sigma=80*u.um),
        weight=spec.train(0.2*u.nS)),
    synapse=spec.models.Expon(tau=5*u.ms),
    output=spec.models.COBA(E=0*u.mV))

# ── 7. Plasticity with reward, schedule, eligibility ────────────────
trial = net.schedule("trial", spec.schedule.Trial(
    iti  = spec.schedule.Phase("iti",  duration=500*u.ms, learning=False),
    stim = spec.schedule.Phase("stim", duration=200*u.ms, learning=True),
    resp = spec.schedule.Phase("resp", duration=300*u.ms, learning=True),
    n_trials=2000, randomize="stim_id"))
net.attach_schedule(trial)

net.project(it, str_d1,
    rule=conn.FixedProb(prob=0.1,
                        weight=spec.train(init.LogNormal(mean=0.05*u.nS,
                                                        std=0.02*u.nS))),
    synapse=spec.models.Expon(tau=5*u.ms),
    output=spec.models.COBA(E=0*u.mV),
    plasticity=spec.plasticity.EligibilitySource(
        name="ctx_str", kind="exp", tau=200*u.ms))

net.project(it, str_d1,
    rule=conn.FixedProb(prob=0.1),
    synapse=spec.models.Expon(tau=5*u.ms),
    output=spec.models.COBA(E=0*u.mV),
    plasticity=spec.plasticity.EligibilityConsumer(
        trace="ctx_str",
        lr=spec.train(1e-3, constraint="positive"),
        modulators={"reward": reward, "dopamine": da}),
    plasticity_schedule=trial.gated_on(["stim", "resp"]))

# ── 8. Bulk homeostatic scaling on every excitatory pop ─────────────
for pop in net.where(tag="excitatory"):
    net.attach_plasticity(pop,
        spec.plasticity.SynapticScaling(
            target_rate=5*u.Hz,
            tau=30*u.second,
            activity=spec.signal.PopulationRate(pop, tau=10*u.second)))

# ── 9. Observations ─────────────────────────────────────────────────
net.observe(net.where(tag="excitatory").spikes,
             name="exc_spikes", every=1*u.ms)
net.observe(it.compartments(morph.SomaRegion()).voltage,
             every=0.5*u.ms, during=trial.window("stim"))
net.observe(str_d1.rate, reducer="mean", every=10*u.ms)

# ── 10. Build ───────────────────────────────────────────────────────
from brainpy.state import eprop          # top-level backend module

ir = net.finalize()
trainer = eprop.build(ir, seed=0, dt=0.5*u.ms,
                      reward_signal="reward",
                      loss=spec.loss.policy_gradient)
```

Every line is reachable from one of §3.5 – §3.12. Nothing is bespoke.

---

## 3.16 IR delta and forward compatibility

This section summarizes what each extension axis adds to the IR (the
canonical surface in [Chapter 2](./02-ir.md)), with its parameter
class. The IR remains forward-compatible: every new field defaults to
"absent" and round-trips unchanged on specs that don't use it.

### 3.16.1 New value wrappers (additive to `ModelRef.params` leaves)

```python
@dataclass(frozen=True)
class Noise:                    # §3.10.3 — sibling of Trainable / DistRef
    kind: str                   # "Wiener" | "OU" | "Multiplicative" | "Poisson" | "Custom"
    params: Mapping[str, Any]
    seed_tag: Optional[str] = None
```

### 3.16.2 New `PopulationNode` fields

```python
@dataclass(frozen=True)
class PopulationNode:
    ...
    positions: Optional["PositionsRef"] = None     # §3.5.2

@dataclass(frozen=True)
class PositionsRef:
    kind: str                                       # "Grid1d" | "Grid2d" | ... | "Free"
    params: Mapping[str, Any]
    extent: Tuple[u.Quantity, ...]
    periodic: Tuple[bool, ...]
```

`ModelRef` of kind `"Cell"` may carry `morphology`, `paint`, `place`,
`cv_policy`, `solver`, `spike_threshold` in `params` (§3.5.3).

### 3.16.3 New `ViewRef` subtypes

```python
@dataclass(frozen=True)
class SpatialViewRef(ViewRef):       # §3.9.4
    mask: Mapping[str, Any]

@dataclass(frozen=True)
class CompartmentViewRef(ViewRef):   # §3.9.5
    region: Mapping[str, Any]
```

### 3.16.4 New top-level node kinds

```python
@dataclass(frozen=True)
class SignalNode:                    # §3.7.2
    id: str
    source: ModelRef

@dataclass(frozen=True)
class ScheduleNode:                  # §3.7.3
    id: str
    kind: str                        # "Phases" | "Trial" | "OneShot"
    params: Mapping[str, Any]
```

### 3.16.5 New `ProjectionNode` fields

```python
@dataclass(frozen=True)
class ProjectionNode:
    ...
    structural_plasticity: Optional[ModelRef] = None     # §3.12.5
    plasticity_schedule:   Optional[str]      = None     # ScheduleNode id; §3.12.4
    modulators:            Mapping[str, str]  = field(default_factory=dict)   # role -> SignalNode id
    temporal_offset:       str                = "same_step"   # §3.11.4
```

### 3.16.6 New `CompoundMeta` blocks

```python
@dataclass(frozen=True)
class GraphMeta:                     # §3.11.3
    name:      str
    node_ids:  Tuple[str, ...]
    edge_ids:  Tuple[str, ...]
    inputs:    Tuple[str, ...]
    outputs:   Tuple[str, ...]

@dataclass(frozen=True)
class QueryMeta:                     # §3.9.6
    id:          str
    kind:        str                 # "Tag" | "Predicate" | "Combined"
    spec:        Mapping[str, Any]
    matched_ids: Tuple[str, ...]

@dataclass(frozen=True)
class CompoundMeta:
    sequentials: Tuple[SequentialMeta, ...] = ()
    groups:      Tuple[GroupMeta,      ...] = ()
    graphs:      Tuple[GraphMeta,      ...] = ()
    queries:     Tuple[QueryMeta,      ...] = ()
```

### 3.16.7 New `NetIR` root fields

```python
@dataclass(frozen=True)
class NetIR:
    ...
    signals:   Tuple[SignalNode,   ...] = ()
    schedules: Tuple[ScheduleNode, ...] = ()
```

`meta["noise_terms"]` enumerates every noise insertion point for the
determinism contract; `meta["applied_patches"]` (existing) lists
applied `ParamPatch`es.

### 3.16.8 Parameter-class map

| Extension                                | LIVE                                       | LIVE_RESET                       | REBUILD                                |
|------------------------------------------|--------------------------------------------|----------------------------------|----------------------------------------|
| Spatial positions (§3.5.2)               | —                                          | positions array                  | geometry kind                          |
| Morphology (§3.5.3)                      | mechanism params (`g_max`, `Cm`, `Em`)     | place params                     | morphology, cv_policy, solver, region exprs |
| Noise (§3.10.3)                          | `sigma`, `tau`                             | `seed_tag`                       | `kind`                                 |
| Signals (§3.7.2)                         | source scalar params                       | bound trace state                | source `kind`                          |
| Schedules (§3.7.3)                       | phase durations                            | schedule attach / detach         | schedule structure                     |
| Plasticity (§3.12.1)                     | rule scalars                               | —                                | rule `kind`                            |
| Modulators binding (§3.12.2)             | —                                          | —                                | `modulators={role: signal_id}`         |
| Eligibility (§3.12.3)                    | `lr`                                       | trace state                      | trace `kind`                           |
| Structural plasticity (§3.12.5)          | rule params (when enabled)                 | —                                | on/off, `mode`                         |
| Graph (§3.11.3)                          | layer params                               | —                                | node membership, edges                 |
| Temporal offset (§3.11.4)                | —                                          | —                                | `same_step` / `next_step`              |
| Tags (§3.9.6)                            | tag metadata                               | —                                | —                                      |
| Predicate-driven view (§3.9.6)           | —                                          | matched-id resolution            | predicate identity                     |

---

## 3.17 Intentionally out of scope

The following categories from
[nc-review §5.1](../nc-review/08-missing-features.md) are deferred:

| Deferred                                     | Reason                                                                                   |
|----------------------------------------------|-------------------------------------------------------------------------------------------|
| Datasets, optimizers, losses                 | D21 keeps these user-side; the schedule grammar in §3.7.3 closes the experiment-protocol gap, which is the more load-bearing slice. |
| Hardware constraints (fan-in/out, placement) | Per-export-backend concern (D25 analogue). Loihi / SpiNNaker exporters consume them; the core spec does not encode them. |
| Sweep strategies (Sobol, Bayesian, resume)   | Belongs in `brainpy sweep` (§4.x of Chapter 4) — not a spec-language concern. |
| Streaming-recording reducers                 | Observable surface extension, parallel to but smaller than §3.7.3 schedules. Suggest separate addendum. |
| Trained-artifact provenance bundle           | Already partially served by `NetIR.meta["applied_patches"]` (§3.14.7) + content hash. A bundle helper is implementation, not spec. |
| Schema evolution / migration tooling         | Implementation surface; spec covers version tag and round-trip determinism (G4). |
| Profiling / cost models                      | Build-time analysis on the finalized IR. Belongs in CLI (§5.1) — `brainpy estimate`. |

Of the six extension axes covered in §3.5 – §3.12, three (spatial,
plasticity, DAG) require **only additive surface changes** — no
breaking edits. Two (morphology, predicate-views) require **one new
IR field each** and one new view-handle subtype. One (stochastic
dynamics) requires only a new value wrapper (`Noise`) alongside
`Trainable` and `DistRef`, plus a `meta["noise_terms"]` list. The
migration is forward-compatible: existing specs round-trip unchanged.

---


---

**Previous:** [Chapter 2 — The IR](./02-ir.md)  
**Next:** [Chapter 4 — Frontend B: YAML/JSON DSL](./04-frontend-yaml.md)
