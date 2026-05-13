# Chapter 5 — Backend protocol and round-trip equivalence

> Part of the [Network Specification DSL RFC](./README.md).

## 5.1 Backend protocol

Two backend families (sim, train) share the same registry plumbing but have
distinct contracts. The protocols live at **`brainpy_state.backend`**
(top-level, **not** under `brainpy_state.spec`). The `spec` module is the
DSL surface and contains no execution code — it only knows about the IR.
`NetIR` is the canonical, content-hashable contract every backend
consumes; the spec module does not ship an exporter to any foreign IR.

```python
# brainpy_state/backend.py                              (TOP-LEVEL module)
from typing import Protocol, Mapping, Any, Iterable

class SimBackend(Protocol):
    name: str
    capabilities: "BackendCapabilities"

    def build(self, ir: NetIR, *,
              seed: int,
              dt: u.Quantity | None = None,
              variables: Mapping[str, Any] | None = None,    # §3.14
              **opts: Any) -> "Simulator": ...

class TrainBackend(Protocol):
    name: str
    capabilities: "BackendCapabilities"

    def build(self, ir: NetIR, *,
              seed: int,
              loss: Callable,
              dt: u.Quantity | None = None,
              variables: Mapping[str, Any] | None = None,    # §3.14
              **opts: Any) -> "Trainer": ...

class Simulator(Protocol):
    ir: NetIR
    seed: int
    bound_variables: Mapping[str, Any]     # §3.14 — concrete values used at build
    def run(self, duration: u.Quantity) -> "TraceBundle": ...
    def reset(self) -> None: ...
    def state(self) -> Mapping[str, Any]: ...

class Trainer(Protocol):
    ir: NetIR
    seed: int
    bound_variables: Mapping[str, Any]     # §3.14 — concrete values used at build
    def step(self, batch) -> "StepReport": ...
    def freeze(self, *paths: str) -> None: ...
    def unfreeze(self, *paths: str) -> None: ...
    def parameters(self) -> Mapping[str, "brainstate.ParamState"]: ...
    # ^ Read-only enumeration of trainable leaves by dotted name. The
    #   trainer's optimizer updates these as internal training state;
    #   they are NOT a path-addressed mutation surface — there is no
    #   .set / .apply / .rebuild_with. To run with different
    #   non-trainable values, call backend.build again with a new
    #   variables={...} mapping.

```

### 5.1.1 Module location

Backend implementations are **top-level modules under `brainpy.state`**,
one per backend. The `brainpy.state.spec` module deliberately contains
no backend implementations — only the IR, frontends, registry, view
algebra, and variable-declaration machinery (D22). The IR is
immutable after `.finalize()`; cross-run variation is supplied
through each backend's `variables=` build kwarg (§3.14).

| Family | Backend      | Module path                  | Notes                                                                |
|--------|--------------|------------------------------|----------------------------------------------------------------------|
| sim    | `clock`      | `brainpy.state.clock`        | Adapter over the existing `_network.Network`/`Builder`.              |
| sim    | `event`      | `brainpy.state.event`        | Event-driven simulator.                                              |
| train  | `bptt`       | `brainpy.state.bptt`         | Autodiff through surrogate spikes; uses `brainstate.nn.Param`.       |
| train  | `eprop`      | `brainpy.state.eprop`        | Synaptic-eligibility-trace training; gradient-free recurrent updates.|
| train  | `eventprop`  | `brainpy.state.eventprop`    | Event-based exact gradients. See `/mnt/d/codes/githubs/snn/eventax`. |
| train  | `ppprop`     | `brainpy.state.ppprop`       | See `/mnt/d/codes/projects/braintrace`.                              |

User code calls them directly:

```python
from brainpy.state import clock
sim = clock.build(ir, seed=0, dt=0.1*u.ms,
                  variables={"tau_exc": 25*u.ms})

from brainpy.state import bptt
trainer = bptt.build(ir, seed=0, loss=loss_fn, dt=1*u.ms,
                     variables={"W_init_std": 0.05})

from brainpy.state import eprop
trainer = eprop.build(ir, seed=0, dt=0.5*u.ms, reward_signal="reward")

from brainpy.state import eventprop
trainer = eventprop.build(ir, seed=0, dt=1*u.ms, loss=loss_fn)
```

Why not nest under `brainpy.state.spec.backends.*`? Two reasons:

1. **The spec is paradigm-neutral; the backends are not.** Keeping
   `spec` free of backend imports preserves the property that
   importing the DSL surface does not transitively pull in JAX
   training runtimes or event-prop tooling.
2. **One symbol per backend at the top.** Switching gradient flavors
   is a one-line `from brainpy.state import <backend>` change, which
   mirrors the load-bearing novelty pitched in
   [§1.1.1](./01-overview.md#111-novelty-and-prior-art).

### 5.1.2 Third-party backends

Entry points group all three families. Entry-point group names sit
under `brainpy_state.backends.*` (the registry uses `backends` plural
since it routes across many implementations):

```toml
[project.entry-points."brainpy_state.backends.sim"]
my_sim = "mypkg.backend:MySimBackend"

[project.entry-points."brainpy_state.backends.train"]
my_train = "mypkg.backend:MyTrainBackend"
```

A third-party backend ships its implementation at whatever import path
it likes (no requirement to live under `brainpy.state.*`); the entry
point makes it discoverable. By convention, a third-party backend
named `my_backend` exposes a top-level alias module
`brainpy_state_my_backend` so users can write
`import brainpy_state_my_backend as my_backend; my_backend.build(...)`.

**Discovery** lives next to the protocol, at the top level:

```python
import brainpy.state.backend as backend

backend.list(kind=None)         # -> tuple[BackendInfo, ...]
backend.list(kind="train")      # filter by family
backend.get("eprop")            # resolve one by name, regardless of family
```

`backend.list()` enumerates every registered backend (shipped + third
party + entry-point loaded). `backend.get(name)` returns the module
object — equivalent to `from brainpy.state import <name>` for shipped
backends.

### 5.1.3 Backend capabilities

Each backend declares a `capabilities` mapping. The loader validates
the IR against the chosen backend's capabilities and raises
`BackendCapabilityError` with the responsible node id when the IR
uses a feature the backend doesn't support.

```python
@dataclass(frozen=True)
class BackendCapabilities:
    supports_delay: bool
    supports_plasticity: bool
    supports_distributions: bool
    supports_nested_subnetworks: bool
    supports_training: bool                # for sim/export, always False
    supports_batch: bool
    supports_positions: bool               # spatial populations (§3.5.2)
    supports_morphology: bool              # multi-compartment Cell models (§3.5.3)
    supports_noise: bool                   # in-equation noise (§3.10.3)
    supports_signals: bool                 # signal nodes (§3.7.2)
    supports_schedules: bool               # schedules (§3.7.3)
    supports_structural_plasticity: bool   # (§3.12.5)
    supports_graphs: bool                  # DAG composition (§3.11.3)
    supported_neuron_kinds: frozenset[str]
    supported_synapse_kinds: frozenset[str]
    supported_output_kinds: frozenset[str]
    supported_rules: frozenset[str]
    supported_layer_macros: frozenset[str]
    supported_input_kinds: frozenset[str]
```

`supported_neuron_kinds`, `supported_rules`, etc. are the load-bearing
fields when domain extensions are in play (Chapter 10). A backend
that handles `braincell.morph_population` lists that string in
`supported_neuron_kinds`; a backend that handles
`brainmass.CouplingMatrix` lists it in `supported_rules`. The boolean
`supports_*` flags continue to describe substrate features (delays,
plasticity, distributions, …) and are extension-agnostic —
extensions that need new boolean flags do not modify this dataclass,
they validate their requirements inside their own node
`validate()` methods and through the per-kind frozensets above.

---


## 5.2 Round-trip and equivalence

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
ir2 = spec.load("x.yaml")
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
**Next:** [Chapter 6 — Registry](./06-registry.md)
