# Chapter 5 — Backend protocol and round-trip equivalence

> Part of the [Network Specification DSL RFC](./README.md).

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

| Family   | Backend      | Notes                                                                 |
|----------|--------------|-----------------------------------------------------------------------|
| sim      | `clock`      | Adapter to existing `_network.Network`/`Builder`.                     |
| train    | `bptt`       | Autodiff through surrogate spikes; uses `brainstate.nn.Param`.        |
| train    | `eprop`      | Synaptic-eligibility-trace training; gradient-free recurrent updates. |
| train    | `event-prop` | Event-based exact gradients. see `/mnt/d/codes/githubs/snn/eventax`   |
| train    | `pp-prop`    | see `/mnt/d/codes/projects/braintrace`                                |
| export   | `nir`        | Neuromorphic IR (§9).                                                 |
| export   | `onnx-spike` | ONNX with the spiking extension ops (future, behind same protocol).   |

---


## 10. Round-trip and equivalence

The two frontends are interchangeable:

```
NetSpec   ──finalize──►   NetIR   ──to_yaml──►   .netspec.yaml
   ▲                        │                       │
   │                     to_dict                    │
NetSpec.from_ir   ◄───── NetIR   ◄───── load ───────┘
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


---

**Previous:** [Chapter 4 — Frontend B: YAML/JSON DSL](./04-frontend-yaml.md)  
**Next:** [Chapter 6 — Export backends: Neuromorphic IR](./06-export-nir.md)
