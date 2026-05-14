# Chapter 6 — Backend design: principles, protocol, and lowering substrate

> Part of the [Network Specification DSL RFC](./README.md).
>
> **Scope.** This chapter defines (a) the load-bearing design
> principles every backend obeys, (b) the protocol surface backends
> implement, (c) the shared `lowering/` substrate every backend
> consumes, (d) the node-kind handler contract that lets new model
> families enter the system without touching any backend, (e) the
> capability declaration and mismatch policy, and (f) the determinism
> and round-trip contract.
>
> Backend-specific execution architectures (per_model vs joint stepping,
> py-loop vs scan time advancement, the BPTT autodiff path, the event
> scheduler) are out of scope here — they live in [Chapter
> 12](./12-backend-impl-sim-bptt.md) (simulation + BPTT) and forthcoming
> companions for event-prop, e-prop, and pp-prop.

---

## 6.1 Design principles

The four principles below are the design's axioms; every later decision
in this RFC and in any future backend is a corollary. They are stated
once here and referenced by short name (P1–P4) throughout.

### P1 — The IR is paradigm-neutral; the backend chooses physics, not the model

A `NetIR` node says *what* a population is (LIF with τ = 20 ms, COBA
output, 1000 units, Bernoulli connectivity at probability 0.1). It
says nothing about Euler vs RK4 vs exact propagator, surrogate-gradient
slope, scan vs while-loop, JIT boundary, accelerator placement, or
delay-buffer representation. Those are backend-side realization
choices, passed through `kind_options` / `node_options` build kwargs
(§1.3 of the overview).

**Operational consequence:** a node-kind handler MAY NOT read any field
that is not on the IR node it lowers. If a value influences
*correctness* of the biological model, it belongs in the IR; if it
influences *how the computation is performed*, it belongs in build
kwargs. Two researchers running the same biological model can
legitimately disagree on the value while still publishing it as "the
same model" — that is the test.

This is the property that makes the load-bearing novelty of
[§1.1.1](./01-overview.md#111-novelty-and-prior-art) work: one spec,
four mathematically distinct gradient flavors, swapped with one import
line.

### P2 — Lowering is a pure function; execution is the only stateful step

`lower(ir, capabilities, ctx) → LoweredNet` is a pure value-to-value
transform. Re-running it with the same `(ir, capabilities, seed,
kind_options, node_options, variables)` returns a structurally
identical `LoweredNet`. **State** — membrane potentials, refractory
clocks, synapse currents, plasticity traces, recorder buffers — only
materializes when a sibling backend instantiates an *execution
artifact* (`Simulator`, `Trainer`) from a `LoweredNet`.

**Operational consequences:**

- `LoweredNet` is cheap to test, cache, diff, golden-fixture, and
  share across sibling backends.
- The same `LoweredNet` can power a `clock` simulator and a `bptt`
  trainer in the same Python session without re-lowering.
- Backends that need different parameter realizations (e.g.
  `Trainable` → `Param` for `bptt`, `Trainable` → frozen constant for
  `clock`) call a post-lowering *materialization* step that takes a
  `LoweredNet` and produces an artifact; lowering itself does not
  know which.

### P3 — Node-kind handlers are open; backends are closed; the lowering protocol is the contract

Anyone — built-ins, BrainPy-family models, NEST-family models,
`braincell`, `brainmass`, third-party domain extensions ([Chapter
5](05-frontend-domain-extensions.md)) — can register a new
**node-kind handler** that supplies a small, fixed interface
(§6.5). **Backends are closed**: they consume handlers but do not
extend them.

**Operational consequence:** adding a new neuron family does not
require touching any backend; adding a new backend does not require
touching any model. The handler interface — not any per-backend
adapter — is the cross-cutting contract.

This is the architectural reason the spec module has no backend
imports (Chapter 10): the substrate that handlers plug into is in
`brainpy_state/lowering/`, which depends only on `spec/` and on
third-party numerics (`braintools.conn`, `braintools.init`,
`saiunit`).

### P4 — Capability mismatch is a typed, locatable error — never silent

Every handler declares a `features: FrozenSet[Feature]` set. Every
backend declares `required_features`, `optional_features`, and per-IR
constraints ("must have a `drift` OR an `exact_propagator`"). The
lowerer cross-references them at `build()` time and raises
`BackendCapabilityError(node_id, missing_features, suggested_backend)`
with actionable text. Fallbacks (e.g., "this backend has no exact
propagator for `LIF`; using `exp_euler` instead") are surfaced as
`BackendNotice` objects on the built artifact — never as Python
warnings, so determinism (G4) and provenance survive across runs.

**Operational consequence:** there are no surprise behaviors. Either
the artifact builds with a documented set of notices, or `build()`
fails before any state is allocated.

---

## 6.2 Architecture: three layers

```
        ┌──────────────────────────────────────────────────────────────┐
        │   spec/                  (Chapters 2–4 — DSL surface)        │
        │   ──────                                                     │
        │   NetSpec  •  load  •  NetIR  •  registry  •  variables      │
        │                                                              │
        │   produces:  NetIR  (immutable, content-hashable value)      │
        └──────────────────────┬───────────────────────────────────────┘
                               │  NetIR
                               ▼
        ┌──────────────────────────────────────────────────────────────┐
        │   lowering/             (NEW — shared substrate; pure)       │
        │   ─────────                                                  │
        │   lower(ir, capabilities, ctx)  ──►  LoweredNet              │
        │     • node-kind handler dispatch (open registry)             │
        │     • view resolution                                        │
        │     • connectivity rule sampling (sparse layout)             │
        │     • seed plan derivation                                   │
        │     • variable binding (§3.14)                               │
        │     • capability cross-check                                 │
        │                                                              │
        │   compose_step_kernel(net, step_mode, integrator, dt)        │
        │   compose_time_loop(step_fn, time_mode, recorder)            │
        │   trainable.materialize(net, mode)                           │
        └──────────────────────┬───────────────────────────────────────┘
                               │  LoweredNet
                               ▼
        ┌──────────────────────────────────────────────────────────────┐
        │   Sibling backends      (top-level modules; closed)          │
        │   ─────────────────                                          │
        │   clock         clock_joint    clock_scan    bptt            │
        │   event         eventprop      eprop         ppprop          │
        │                                                              │
        │   Each is a short build() pipeline that composes pieces      │
        │   from lowering/ into its execution strategy.                │
        └──────────────────────────────────────────────────────────────┘
```

**The three layers correspond directly to P1, P2, P3.** The spec layer
is paradigm-neutral (P1); the lowering layer is the pure value-to-value
transform (P2); the backend layer is where new execution strategies
plug in without touching the model definitions, and where new model
families plug in without touching the execution strategies (P3).

The `lowering/` module is new with this design. Earlier drafts of the
RFC sketched `clock` as a thin adapter over the existing
`brainpy_state._network.Network`/`Builder`. That sketch is superseded
here: the execution layer is rebuilt on top of `lowering/` so the
same substrate hosts simulation, BPTT, and the upcoming event-prop,
e-prop, and pp-prop backends without per-backend duplication.

---

## 6.3 Backend protocol

Two backend families (sim, train) share the same registry plumbing but
have distinct contracts. The protocols live at
**`brainpy_state.backend`** (top-level, **not** under
`brainpy_state.spec`). The `spec` module is the DSL surface and
contains no execution code — it only knows about the IR. `NetIR` is
the canonical, content-hashable contract every backend consumes; the
spec module does not ship an exporter to any foreign IR.

```python
# brainpy_state/backend.py                              (TOP-LEVEL module)
from typing import Protocol, Mapping, Any, Callable

class SimBackend(Protocol):
    name: str
    capabilities: "BackendCapabilities"

    def build(self, ir: NetIR, *,
              seed: int,
              dt: u.Quantity | None = None,
              kind_options: Mapping[str, Mapping[str, Any]] | None = None,
              node_options: Mapping[str, Mapping[str, Any]] | None = None,
              variables: Mapping[str, Any] | None = None,
              **opts: Any) -> "Simulator": ...

class TrainBackend(Protocol):
    name: str
    capabilities: "BackendCapabilities"

    def build(self, ir: NetIR, *,
              seed: int,
              loss: Callable,
              dt: u.Quantity | None = None,
              kind_options: Mapping[str, Mapping[str, Any]] | None = None,
              node_options: Mapping[str, Mapping[str, Any]] | None = None,
              variables: Mapping[str, Any] | None = None,
              **opts: Any) -> "Trainer": ...

class Simulator(Protocol):
    ir: NetIR
    seed: int
    bound_variables: Mapping[str, Any]     # §3.14 — concrete values used at build
    notices: tuple["BackendNotice", ...]   # capability-fallback record (§6.6)
    def run(self, duration: u.Quantity) -> "TraceBundle": ...
    def reset(self) -> None: ...
    def state(self) -> Mapping[str, Any]: ...

class Trainer(Protocol):
    ir: NetIR
    seed: int
    bound_variables: Mapping[str, Any]
    notices: tuple["BackendNotice", ...]
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

### 6.3.1 Module location

Backend implementations are **top-level modules under `brainpy.state`**,
one per backend. The `brainpy.state.spec` module deliberately contains
no backend implementations — only the IR, frontends, registry, view
algebra, and variable-declaration machinery (D22). The IR is
immutable after `.finalize()`; cross-run variation is supplied
through each backend's `variables=` build kwarg (§3.14).

| Family | Backend       | Module path                  | Notes                                                                |
|--------|---------------|------------------------------|----------------------------------------------------------------------|
| sim    | `clock`       | `brainpy.state.clock`        | per_model × py_loop — default sim (Chapter 12).                       |
| sim    | `clock_joint` | `brainpy.state.clock_joint`  | joint × py_loop — coupled biophysical ODEs (Chapter 12).              |
| sim    | `clock_scan`  | `brainpy.state.clock_scan`   | per_model × scan — fastest forward for uniform-shape SNNs.            |
| sim    | `event`       | `brainpy.state.event`        | Event-driven simulator (future chapter).                              |
| train  | `bptt`        | `brainpy.state.bptt`         | clock_scan + grad (Chapter 12).                                       |
| train  | `eprop`       | `brainpy.state.eprop`        | Eligibility-trace training (future chapter).                          |
| train  | `eventprop`   | `brainpy.state.eventprop`    | Event-based exact gradients (future chapter).                         |
| train  | `ppprop`      | `brainpy.state.ppprop`       | RTRL / forward-mode autodiff (future chapter).                        |

User code calls them directly:

```python
from brainpy.state import clock
sim = clock.build(ir, seed=0, dt=0.1*u.ms,
                  variables={"tau_exc": 25*u.ms})

from brainpy.state import bptt
trainer = bptt.build(ir, seed=0, loss=loss_fn, dt=1*u.ms,
                     variables={"W_init_std": 0.05})
```

**Why backends are top-level, not nested under `brainpy.state.spec.backends.*`:**

1. **The spec is paradigm-neutral; the backends are not.** Keeping
   `spec/` free of backend imports preserves the property that
   importing the DSL surface does not transitively pull in JAX
   training runtimes or event-prop tooling.
2. **One symbol per backend at the top.** Switching gradient flavors
   is a one-line `from brainpy.state import <backend>` change, which
   is the load-bearing ergonomic of the comparative-study workflow
   ([§1.1.1.2](./01-overview.md#1112-why-this-matters)).

### 6.3.2 Third-party backends

Entry points group all three families. Entry-point group names sit
under `brainpy_state.backends.*` (plural — the registry routes across
many implementations):

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

backend.list(kind=None)              # -> tuple[BackendInfo, ...]
backend.list(kind="train")           # filter by family
backend.get("eprop")                 # resolve one by name, regardless of family
```

`backend.list()` enumerates every registered backend (shipped + third
party + entry-point loaded). `backend.get(name)` returns the module
object — equivalent to `from brainpy.state import <name>` for shipped
backends.

---

## 6.4 The shared lowering substrate

`brainpy_state/lowering/` is the substrate every backend consumes. It
is a pure, paradigm-neutral library — no JAX training imports, no
event-scheduler imports, no surrogate-gradient choices baked in. Its
contents are stable across all backend families.

```
brainpy_state/lowering/
├── protocol.py        NodeHandler protocol, LoweredNode, LoweredNet,
│                      BuildContext, Feature, BackendCapabilities,
│                      BackendCapabilityError, BackendNotice.
├── lower.py           lower(ir, capabilities, ctx) -> LoweredNet
├── registry.py        Handler registration & lookup by IR `kind`.
├── connectivity.py    Materialize ConnRule -> sampled edges + sparse
│                      layout (CSR / COO / dense, chosen by edge density
│                      and the backend's preferred layout hint).
├── views.py           Resolve ViewRef into integer index arrays / slices.
├── variables.py       Bind §3.14 variables into concrete IR leaves.
├── distributions.py   DistRef -> braintools.init.Initialization instance.
├── trainable.py       Trainable -> {param leaf, frozen leaf} based on
│                      the backend's TrainableMode.
├── seed.py            fold_in scheme for per-population / per-projection
│                      seeds; deterministic given the build seed.
├── compose.py         compose_step_kernel(net, step_mode, integrator, dt)
│                      compose_time_loop(step_fn, time_mode, recorder)
└── handlers/          First-party handlers (LIF, Expon, COBA, Poisson, ...).
```

The substrate exposes three kinds of API:

1. **The lowering entry point** — `lower(ir, capabilities, ctx) →
   LoweredNet`. Every backend calls this exactly once per `build()`.
2. **Composer functions** — `compose_step_kernel` and
   `compose_time_loop`. These are how sibling backends differ: each
   picks a `step_mode` (per_model / joint) and a `time_mode`
   (py_loop / scan / while_loop). Chapter 12 describes the four
   first-party cells.
3. **Cross-cutting helpers** — `variables.bind`,
   `trainable.materialize`, `seed.plan`. Backends call these in
   short, fixed pipelines (Chapter 12 §12.7).

**Sibling backends are thin.** Each is "assemble a `LoweredNet` into MY
kind of step kernel, wrap in MY loop, expose MY runtime surface." All
the heavy lifting (node-kind dispatch, view resolution, edge sampling,
seed derivation, unit checking, capability cross-check) is in
`lowering/` and is exercised once per IR regardless of which backend
is chosen.

---

## 6.5 The node-kind handler protocol

This is the single contract a model family must satisfy to participate
in the system. One handler per IR `kind` (`LIF`, `Expon`, `COBA`,
`iaf_psc_alpha`, …). A handler is a *pure value-producer* — it does
NOT step state and does NOT touch JAX random state directly. State
materialization and time advancement are the backend's job.

```python
# brainpy_state/lowering/protocol.py
from typing import Protocol, FrozenSet, Optional, Callable, Mapping, Any
from dataclasses import dataclass, field

Feature = str
# Canonical first-party features (extensions may add their own):
#   "ode_drift_fn"          handler exposes drift(state, inputs, t, params) -> dstate/dt
#   "exact_propagator"      handler exposes exact_step(state, inputs, dt, params)
#   "discrete_spike"        handler exposes spike(state, params) -> bool
#   "surrogate_grad"        handler exposes surrogate(state, params) -> array
#   "discontinuous_reset"   handler exposes jump(state, params) -> state'
#   "refractoriness"        jump uses a refractory-clock state field
#   "stochastic_term"       handler's drift consumes a per-step PRNG key
#   "delta_input"           handler accepts delta-style spike inputs
#   "current_input"         handler accepts continuous-current inputs
#   "multi_compartment"     handler's state is per-compartment (braincell)
#   "adaptive_dt_friendly"  drift is smooth enough for adaptive integrators
#   "next_spike_time"       (for event backend) handler exposes a closed-form
#                           or root-finding next-spike-time function

class NodeHandler(Protocol):
    kind: str
    role: str                      # "neuron" | "synapse" | "output" | "plasticity"
                                   # | "input" | "observable"
    features: FrozenSet[Feature]

    def signature(self) -> "ParamSignature":
        """Parameter names, expected saiunit dimensions, defaults, and
        trainability annotations. Consumed by the spec validator
        (Chapter 9) and by viz (Chapter 8)."""

    def lower(self,
              node: "ModelRef",
              target_shape: tuple[int, ...],
              ctx: "BuildContext") -> "LoweredNode":
        """Pure function of (node, target_shape, ctx.kind_options,
        ctx.node_options, ctx.seed_for(node))."""
```

A handler's output is a frozen value the backend interprets according
to its execution strategy:

```python
@dataclass(frozen=True)
class LoweredNode:
    kind: str
    role: str
    features: FrozenSet[Feature]
    state_spec: "StateSpec"               # PyTree skeleton + init fns + units

    # Dynamics. A handler populates a subset; `features` declares which.
    drift:      Optional[Callable] = None
    jump:       Optional[Callable] = None
    spike:      Optional[Callable] = None
    surrogate:  Optional[Callable] = None
    exact_step: Optional[Callable] = None
    next_spike_time: Optional[Callable] = None    # (state, params) -> u.Quantity

    observables: Mapping[str, Callable] = field(default_factory=dict)
    params:      Mapping[str, Any]      = field(default_factory=dict)
    notices:     tuple[str, ...]        = ()
```

A handler may populate any subset of `(drift, jump, exact_step, spike,
surrogate, next_spike_time)`; the `features` set declares which ones
are real. A backend reads the relevant fields and treats the rest as
absent:

- `clock` (per_model): prefer `exact_step` if available, else
  `drift` + integrator; apply `jump` after each step.
- `clock_joint` (joint): require `drift`; refuse `exact_step`-only
  handlers because joint integration would skip cross-population
  coupling.
- `bptt`: require `surrogate` on neuron handlers (consumed by the
  custom-VJP path inside `spike_detect` — Chapter 12 §12.7.4).
- `event`: require `next_spike_time` on neuron handlers.

`LoweredNet` aggregates handler outputs plus connectivity:

```python
@dataclass(frozen=True)
class LoweredNet:
    populations: tuple[LoweredPopulation, ...]
    projections: tuple[LoweredProjection, ...]   # sparse layout + lowered syn/out
    inputs:      tuple[LoweredInput, ...]
    observables: tuple[LoweredObservable, ...]
    seed_plan:   "SeedPlan"
    notices:     tuple[BackendNotice, ...]
```

### 6.5.1 Handler registration

Handlers are registered against the IR `kind` they handle. First-party
handlers live in `brainpy_state/lowering/handlers/` and self-register
at import. Third-party handlers register via Python entry points:

```toml
[project.entry-points."brainpy_state.lowering.handlers"]
my_neuron = "mypkg.handlers:MyNeuronHandler"
```

The registry is read once per process; a handler's `kind` collides
loudly (`HandlerRegistrationError`) if two packages claim the same
name. Domain extensions (Chapter 5) use the same mechanism — there is
no separate "extension" surface, only handlers.

### 6.5.2 What handlers must not do

To preserve P1 and P2:

- A handler MUST NOT call `jax.random.*` or read process-level RNG;
  randomness comes from `ctx.seed_for(node)`, which the substrate
  derives from the build seed via `jax.random.fold_in`.
- A handler MUST NOT mutate any input. `LoweredNode` is frozen; the
  handler returns a fresh value.
- A handler MUST NOT call into another handler. Composition is the
  backend's job, via `compose_step_kernel`. A handler does not know
  what its neighbours are.
- A handler MUST NOT depend on `dt`, the time-loop strategy, or
  whether the surrounding artifact will be differentiated. It declares
  features; backends select among them.

---

## 6.6 Capability declaration and mismatch policy

### 6.6.1 The capability dataclass

Each backend declares a `capabilities` value. The lowerer validates
the IR against this declaration and raises `BackendCapabilityError`
with the responsible node id when the IR uses a feature the backend
doesn't support.

```python
@dataclass(frozen=True)
class BackendCapabilities:
    # Substrate-level booleans (extension-agnostic):
    supports_delay: bool
    supports_plasticity: bool
    supports_distributions: bool
    supports_nested_subnetworks: bool
    supports_training: bool                # sim backends always False
    supports_batch: bool
    supports_positions: bool               # spatial populations (§3.5.2)
    supports_morphology: bool              # multi-compartment Cell models (§3.5.3)
    supports_noise: bool                   # in-equation noise (§3.10.3)
    supports_signals: bool                 # signal nodes (§3.7.2)
    supports_schedules: bool               # schedules (§3.7.3)
    supports_structural_plasticity: bool   # (§3.12.5)
    supports_graphs: bool                  # DAG composition (§3.11.3)

    # Feature-level: required and optional handler features.
    required_features:  FrozenSet[Feature]
    optional_features:  FrozenSet[Feature]
    requires_any_of:    FrozenSet[FrozenSet[Feature]]   # "X OR Y" requirements

    # Kind-level: which kinds this backend has wiring for (load-bearing
    # for domain extensions; see Chapter 5).
    supported_neuron_kinds: FrozenSet[str]
    supported_synapse_kinds: FrozenSet[str]
    supported_output_kinds:  FrozenSet[str]
    supported_input_kinds:   FrozenSet[str]
    supported_rules:         FrozenSet[str]
    supported_layer_macros:  FrozenSet[str]

    # How the backend materializes Trainable wrappers:
    trainable_mode: Literal["constant", "param"]
```

`supported_neuron_kinds`, `supported_rules`, etc. are the load-bearing
fields when domain extensions are in play (Chapter 5). A backend that
handles `braincell.morph_population` lists that string in
`supported_neuron_kinds`; a backend that handles
`brainmass.CouplingMatrix` lists it in `supported_rules`. The boolean
`supports_*` flags continue to describe substrate features (delays,
plasticity, distributions, …) and are extension-agnostic — extensions
that need new feature flags do not modify this dataclass, they
validate their requirements inside their own handler `signature()`
and through the `Feature` strings in `required_features`.

### 6.6.2 Three classes of mismatch

**1. Hard miss — a required feature is absent, or a required kind is
not in the supported set.**
The lowerer raises `BackendCapabilityError(node_id,
missing_features, suggested_backend)` at `build()` (never at `run()` /
`step()`). The error names the IR node id and includes a concrete
fix:

```
BackendCapabilityError:
  Node 'population.weird_neuron' (kind 'MyNeuron') is missing required
  features for backend 'bptt': {'ode_drift_fn', 'surrogate_grad'}.
  Suggested fixes:
    - attach a `drift` function to MyNeuron's handler;
    - attach a surrogate (e.g. braintools.surrogate.fast_sigmoid())
      to MyNeuron;
    - or switch to `clock` if BPTT training is not required.
```

**2. Soft miss — an optional feature is absent and a documented
fallback exists.**
The lowerer emits a `BackendNotice` carrying the node id and the
fallback choice. Notices are attached to `LoweredNet.notices` and
flow through to the built artifact (`sim.notices`, `trainer.notices`).
They are NOT Python warnings, so determinism (G4) is preserved.

```python
BackendNotice(
    node_id="population.exc",
    code="LOW-FALLBACK-001",
    message=("clock: 'LIF' has no exact_step; "
             "using integrator='exp_euler' for this population."),
)
```

**3. Soft mismatch with no fallback — accepted but informational.**
E.g., a `Trainable` marker reaching `clock` resolves to a frozen
constant; the backend emits `TrainableIgnored(path)` and proceeds
unless the `Trainable` was declared `required=True`, in which case it
becomes a hard miss.

### 6.6.3 Pre-flight check (no state allocation)

The capabilities object exposes a dry-run:

```python
caps = backend.get("bptt").capabilities
report = caps.report_for(ir)
# -> CapabilityReport(missing=[...], fallbacks=[...], notices=[...])
```

`report_for(ir)` walks the IR with the same logic `build()` uses, but
without materializing any state. It is the substrate for sweep plans
(skip configurations that won't build), CI guards, the
`brainpy explain` CLI subcommand, and IR linters.

---

## 6.7 Variable and Trainable resolution

These two value wrappers (§2, §3.14) are the only IR leaves that
require build-time interpretation. Both are resolved inside
`lowering/`, before sibling-backend-specific work begins.

### 6.7.1 Variable binding

Build-time variables (§3.14) are resolved **before** lowering, by
`lowering.variables.bind(ir, variables) → bound_ir`. The bind step:

1. Validates that every declared `VariableDecl.name` is bound or has a
   default (else `SPEC-023`).
2. Validates units / dimensions against the declared default
   (`SPEC-024`).
3. Applies constraints (`positive`, `unit_norm`, `clip:lo,hi`;
   `SPEC-025`).
4. Rejects unknown keys in the user-supplied mapping (`SPEC-026`).
5. Substitutes `VariableRef` leaves with their concrete values,
   producing a new `NetIR` with the same `content_hash`. **Variables
   do not enter the hash** — they are build inputs, not model
   identity.

After bind, the IR has no `VariableRef` leaves left. The bound IR is
what `lower()` consumes.

### 6.7.2 Trainable materialization

`lowering.trainable.materialize(net, mode) → LoweredNet` operates on
the `LoweredNet`, not the IR. `mode` is set by the backend:

- `mode="constant"` (clock, clock_joint, clock_scan, event):
  every `Trainable` leaf becomes a frozen constant carrying the
  wrapped initial value; a `TrainableIgnored(path)` notice is
  emitted unless `required=True`.
- `mode="param"` (bptt, eprop, eventprop, ppprop): every
  `Trainable` leaf becomes a `brainstate.nn.Param` whose initial
  value is the wrapper's `value` (after `DistRef` materialization,
  if any). The flat `Mapping[dotted_name, ParamState]` is exposed
  via `trainer.parameters()`.

The dotted-name convention is `<role>.<node_id>.<state_or_param_name>`,
e.g. `population.exc.tau_m`, `projection.exc__to__inh.weight`. The
name is derived from the IR — it is stable across re-lowerings of the
same IR, which is the property that makes parameter checkpoints
portable across backend swaps.

---

## 6.8 Determinism contract

Each backend commits to a multi-layered determinism guarantee, all of
which derive from the principles above:

**G4(a) — Lowering determinism.**
For any IR and any `variables` mapping, two `bind()` + `lower()` runs
produce structurally identical `LoweredNet` values, including notice
ordering. Notices are sorted by IR node id, not by handler visit
order.

**G4(b) — Seed determinism.**
`lowering.seed.plan(build_seed, ir)` returns a deterministic
`SeedPlan`. Each population, projection, and stochastic input gets a
key via `jax.random.fold_in(build_seed, h(node_id))`, where `h` is
SHA-256 over the canonical node id string. The plan is independent of
backend choice.

**G4(c) — Connectivity determinism.**
Sparse-layout choices (CSR vs COO vs dense) are made by edge density
thresholds — not by handler visit order — so two builds of the same
IR with the same seed produce byte-identical sparse layouts.

**G4(d) — Artifact determinism.**
Two `build()` calls with the same `(ir, seed, kind_options,
node_options, variables)` produce execution artifacts that, given the
same input batch, return bit-identical outputs (modulo non-deterministic
GPU reductions, which users opt out of with the `xla_deterministic`
build flag).

**G4(e) — Training determinism (train backends).**
Under the same `(seed, optimizer, batch sequence)`, parameter values
after `N` steps are bit-identical across runs (subject to the same
GPU-reduction caveat).

The content hash defined in §6.9 is the cross-cutting verifier:
identical content hash + identical build inputs ⇒ identical artifact
behavior.

---

## 6.9 Round-trip and equivalence

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

The content hash is a SHA-256 over the IR rendered to its canonical
JSON form:

- keys sorted lexicographically,
- floats formatted with `repr` (no trailing zeros),
- `u.Quantity` rendered as `{"_q": [mantissa, unit_str]}`,
- list order preserved (it is semantic for projections / observables),
- `Trainable`, `DistRef`, `ConnRule`, `ModelRef` rendered with `_t`,
  `_d`, `_c`, `_m` discriminators,
- `VariableRef` leaves rendered with their declared `default` and
  `name` but NOT with any user-supplied binding (binding happens at
  build time, not at IR construction — §6.7.1).

Content hash is used for: build cache keys, golden-IR test fixtures,
export determinism, sweep deduplication, and the
`BackendCapabilities.report_for(ir)` pre-flight cache.

---

## 6.10 Pointers to specific backend architectures

Backend-specific execution architectures — the per_model vs joint
stepping choice, the py-loop vs scan time-loop choice, the BPTT
autodiff path, the surrogate-gradient `custom_vjp`, the event
scheduler, the eligibility-trace composer, the RTRL update — live in
chapters of their own. This chapter is the *substrate* they all stand
on.

| Backend family               | Chapter                                              |
|------------------------------|------------------------------------------------------|
| `clock`, `clock_joint`, `clock_scan`, `bptt` | [Chapter 12](./12-backend-impl-sim-bptt.md) |
| `event`, `eventprop`         | (forthcoming) Chapter 13                              |
| `eprop`                      | (forthcoming) Chapter 14                              |
| `ppprop`                     | (forthcoming) Chapter 15                              |

Each backend chapter follows the same template: capability card,
`build()` pipeline, composition strategy (which `compose_*` calls it
makes from `lowering/`), execution-artifact runtime contract, and
backend-specific kwargs. None of those chapters re-derive the
principles in §6.1 or the protocol in §6.3; they extend, not replace,
this chapter.

---

**Previous:** [Chapter 5 — Domain extensions](05-frontend-domain-extensions.md)
**Next:** [Chapter 7 — Registry](./07-registry.md)
