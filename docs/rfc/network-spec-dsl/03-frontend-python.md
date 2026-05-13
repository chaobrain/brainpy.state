# Chapter 3 — Frontend A: Fluent NetSpec builder (Python)

> Part of the [Network Specification DSL RFC](./README.md).

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
brainpy patch brunel.netspec.yaml --from brunel.patch.yaml -o brunel-v2.yaml
brainpy run   brunel.netspec.yaml --patch brunel.patch.yaml --backend clock --duration "1 s"
brainpy build brunel.netspec.yaml --patch brunel.patch.yaml --backend nir -o brunel.nir
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


---

**Previous:** [Chapter 2 — The IR](./02-ir.md)  
**Next:** [Chapter 4 — Frontend B: YAML/JSON DSL](./04-frontend-yaml.md)
