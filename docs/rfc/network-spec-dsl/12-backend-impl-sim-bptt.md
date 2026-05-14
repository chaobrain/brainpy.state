# Chapter 12 — Backend implementation architecture (simulation + BPTT)

> Part of the [Network Specification DSL RFC](./README.md).
>
> **Status:** Design specification.
> **Scope:** Execution architecture for the **simulation** backends
> (`clock`, `clock_joint`, `clock_scan`) and the **BPTT training** backend
> (`bptt`). Event-prop, e-prop, and pp-prop backends share the same
> substrate; their backend-specific machinery is deferred to a separate
> chapter and is **not** described here.
> **Date:** 2026-05-14.
>
> This chapter complements [Chapter 6](./06-backends.md) (protocol surface)
> and [Chapter 10](./10-implementation.md) (codebase mapping). Where this
> chapter and Chapter 10 disagree on internal organization, this chapter
> takes precedence: Chapter 10 was written when `clock` was conceived as
> an adapter over the existing `_network.Network`/`Builder`. This chapter
> rebuilds the execution layer from scratch on a paradigm-neutral
> substrate, so that the same machinery can later host event-prop, e-prop,
> and pp-prop without reshaping `clock` or `bptt`.

---

## 12.1 Four principles

These four are the design axioms; every later decision is a corollary.

**P1 — The IR is paradigm-neutral; the backend chooses physics, not the model.**
A `NetIR` node says *what* a population is (LIF with τ = 20 ms, COBA output,
1000 units). It says nothing about Euler vs RK4 vs exact propagator,
surrogate-gradient slope, scan vs while-loop, JIT boundary, or device
placement. Those are backend-side realization choices passed through
`kind_options` / `node_options` / build kwargs (see §1.3 of the RFC).
A **node-kind handler MAY NOT read any field that is not on the IR node
it lowers**.

**P2 — Lowering is a pure function. Execution is the only stateful step.**
`lower(ir, ctx) → LoweredNet` is a pure value-to-value transform.
Re-running it with the same `(ir, seed, kind_options, node_options,
variables)` returns a structurally identical `LoweredNet`. State
(membrane potentials, refractory clocks, synapse currents, plasticity
traces) only materializes when a backend instantiates an *execution
artifact* from a `LoweredNet`. This makes the lowering layer cheap to
test, cache, diff, and share across sibling backends.

**P3 — Node-kind handlers are open; backends are closed; the lowering
protocol is the contract.**
Anyone (built-ins, BrainPy-family models, NEST-family models,
`braincell`, `brainmass`, third-party domain extensions) can register a
new node-kind handler that supplies a small, fixed interface
(§12.4). **Backends are closed**: they consume handlers but do not
extend them. Adding a new neuron family does not require touching any
backend; adding a new backend does not require touching any model.

**P4 — Capability mismatch is a typed, locatable error — never silent.**
Every handler declares a `features: FrozenSet[Feature]`. Every backend
declares `required_features` and `optional_features`. The lowerer
cross-references them and raises `BackendCapabilityError(node_id,
missing_features, suggested_backend)` with actionable text. Fallbacks
(e.g., "this backend has no exact propagator for `LIF`; will use
`exp_euler` instead") are surfaced as `BackendNotice` objects on the
built artifact — never as Python warnings, so determinism (G4) and
provenance survive across runs.

---

## 12.2 Layered module map

```
brainpy_state/
├── spec/                       (Chapters 2–4 — unchanged surface)
├── lowering/                   (NEW — the shared substrate; pure)
│   ├── protocol.py             Handler protocol, LoweredNet, NodeBundle,
│   │                           BuildContext, Feature, BackendCapabilityError,
│   │                           BackendNotice.
│   ├── lower.py                lower(ir, capabilities, ctx) -> LoweredNet
│   ├── registry.py             Handler registration & lookup by IR kind.
│   ├── connectivity.py         Materialize ConnRule -> sampled edges + sparse
│   │                           layout (CSR / COO / dense, picked from edge
│   │                           density at lowering time).
│   ├── views.py                Resolve ViewRef into integer index arrays / slices.
│   ├── variables.py            Bind §3.14 variables into concrete IR leaves.
│   ├── distributions.py        DistRef -> braintools.init.Initialization instance.
│   ├── trainable.py            Trainable -> {param leaf, frozen leaf} based on
│   │                           the backend's TrainableMode.
│   ├── seed.py                 fold_in scheme for per-population / per-projection
│   │                           seeds; deterministic given the build seed.
│   ├── compose.py              compose_step_kernel(net, step_mode, integrator, dt)
│   │                           compose_time_loop(step_fn, time_mode, recorder)
│   └── handlers/               First-party handlers (LIF, Expon, Poisson, ...).
│
├── backend.py                  Protocols (SimBackend, TrainBackend),
│                               BackendCapabilities, entry-point loader,
│                               backend.list() / backend.get(name).
├── clock.py                    sibling sim backend — per_model × py_loop (default)
├── clock_joint.py              sibling sim backend — joint × py_loop
├── clock_scan.py               sibling sim backend — per_model × scan
├── bptt.py                     sibling train backend — per_model × scan × grad
└── …
```

**Sibling backends are thin.** Each is a build pipeline that composes
shared pieces in a specific way; none owns a private re-implementation
of lowering, view algebra, edge sampling, or seed derivation.

---

## 12.3 The node-kind handler protocol

One handler per IR `kind` (`LIF`, `Expon`, `COBA`, `iaf_psc_alpha`, …).
A handler is a *pure value-producer* — it does NOT step state and does
NOT touch JAX random state directly.

```python
# brainpy_state/lowering/protocol.py
from typing import Protocol, FrozenSet, Optional, Callable, Mapping, Any
from dataclasses import dataclass, field
import saiunit as u

Feature = str
# Canonical first-party features (others may be added by extensions):
#   "ode_drift_fn"            handler exposes f(state, inputs, t, params) -> dstate/dt
#   "exact_propagator"        handler exposes step(state, inputs, dt, params)
#   "discrete_spike"          handler exposes spike(state, params) -> bool
#   "surrogate_grad"          handler exposes surrogate(state, params) -> array
#   "discontinuous_reset"     handler exposes jump(state, params) -> state'
#   "stochastic_term"         handler's drift consumes a per-step PRNG key
#   "refractoriness"          jump uses a refractory-clock state field
#   "delta_input"             handler accepts delta-style spike inputs
#   "current_input"           handler accepts continuous-current inputs
#   "multi_compartment"       handler's state is per-compartment (braincell)
#   "adaptive_dt_friendly"    drift is smooth enough for adaptive integrators

class NodeHandler(Protocol):
    kind: str
    role: str                      # "neuron" | "synapse" | "output" | "plasticity"
                                   # | "input" | "observable"
    features: FrozenSet[Feature]

    def signature(self) -> "ParamSignature":
        """Names, expected saiunit dimensions, defaults, and trainability
        annotations for the model's constructor parameters. Consumed by the
        spec validator (Chapter 9) and by viz (Chapter 8)."""

    def lower(self,
              node: "ModelRef",
              target_shape: tuple[int, ...],
              ctx: "BuildContext") -> "LoweredNode":
        """Pure function of (node, target_shape, ctx.kind_options,
        ctx.node_options, ctx.seed_for(node))."""
```

`LoweredNode` is the handler's output — a frozen value the backend
interprets according to its execution strategy:

```python
@dataclass(frozen=True)
class LoweredNode:
    kind: str
    role: str
    features: FrozenSet[Feature]
    state_spec: "StateSpec"               # PyTree skeleton + init fns + units

    # Dynamics. A handler populates a subset of these; `features` declares which.
    drift:      Optional[Callable] = None  # (state, inputs, t, params) -> dstate/dt
    jump:       Optional[Callable] = None  # (state, inputs, t, params) -> state'
                                           #   instantaneous: reset + refractory
    spike:      Optional[Callable] = None  # (state, params) -> spike_bool
    surrogate:  Optional[Callable] = None  # (state, params) -> array  (for VJP)
    exact_step: Optional[Callable] = None  # (state, inputs, dt, params) -> state'

    observables: Mapping[str, Callable] = field(default_factory=dict)
    params:      Mapping[str, Any]      = field(default_factory=dict)  # concrete
    notices:     tuple[str, ...]        = ()
```

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

A clock backend in `per_model` mode prefers `exact_step` if available
and falls back to `drift` + the configured integrator; `clock_joint`
refuses `exact_step` (because it would skip cross-population coupling)
and requires `drift`; `bptt` additionally requires `surrogate` on
neuron handlers.

---

## 12.4 Two orthogonal axes

What previous chapters called "execution semantics" is actually a 2 × 2
matrix of two independent decisions.

| Axis | Variants | What it controls |
|---|---|---|
| **Step semantics** | `per_model` \| `joint` | How a single `dt` is computed: each population integrates with its own integrator (per_model), or all `drift` functions are stacked and a single integrator advances the union state (joint). |
| **Time semantics** | `py_loop` \| `scan` *(\| `while_loop`, future)* | How time is advanced: a Python-controlled loop with side-effecting recorders, or `jax.lax.scan` with stacked outputs and a pure carry. |

Sibling backends are specific cells of this matrix:

| Backend | Step | Time | Differentiable | Use case |
|---|---|---|---|---|
| `clock` | per_model | py_loop | No | General simulation; biggest IR-validity envelope; debugger-friendly. |
| `clock_joint` | joint | py_loop | No | Tight biophysical coupling — populations whose dynamics must share a single integrator (gap junctions, conductance-based mean field). |
| `clock_scan` | per_model | scan | No | Uniform-shape deep / recurrent SNN with a batch axis; JIT-friendly. |
| `bptt` | per_model | scan | **Yes** | Same kernel as `clock_scan` plus `Trainable` materialization, surrogate-gradient enforcement, and loss/optimizer. |

The matrix is forward-compatible: a future `clock_adaptive`
(per_model × while_loop) or `event_scan` (event-driven step × scan)
slots in without rearchitecture.

---

## 12.5 The step composer

The shared lowering library exposes one function that turns a
`LoweredNet` into a pure step kernel:

```python
# brainpy_state/lowering/compose.py
def compose_step_kernel(
    net: LoweredNet,
    *,
    step_mode: Literal["per_model", "joint"],
    integrator: IntegratorSpec,
    dt: u.Quantity,
) -> StepFn:
    """Return a pure function
        step(state, t, exogenous_inputs, params) -> (state', step_outputs)
    `step_outputs` is a fixed-shape dict of observable values plus the
    network-wide spike record. The function is JAX-traceable."""
```

The composer's per-`dt` schedule is fixed and explicit (because P1
forbids backends from rearranging physics):

```
For one dt:
  1. project_spikes:   sparse delivery of t-1 spikes through every projection
                       -> per-post-population synaptic input bundle.
  2. synapse_step:     advance each synapse state (exact_step if available,
                       else drift + jump under the chosen integrator).
  3. mix_inputs:       sum (a) synaptic currents/conductances from §1-§2,
                       (b) exogenous_inputs[t] from InputNodes,
                       (c) noise term if features include "stochastic_term".
  4. neuron_step:
       per_model:      each population: exact_step if available, else
                       integrator(drift, dt). `exact_if_available` falls
                       back to `exp_euler` per-handler.
       joint:          stack all `drift` fns into one flat drift on a union
                       state vector; advance with a single integrator step.
                       Reject any handler missing `drift` (raise
                       BackendCapabilityError listing the responsible
                       populations and suggesting `clock`).
  5. spike_detect:     call each neuron handler's `spike(state, params)`.
                       Under bptt, this op carries the custom-VJP that
                       uses `surrogate` (§12.7).
  6. reset_jump:       apply each handler's `jump` (reset + refractory) on
                       spiking units.
  7. plasticity_step:  advance plasticity traces (separate handlers;
                       eligibility traces, STDP windows, STP states).
  8. emit_observables: compute every ObservableNode's quantity from current
                       state (downsampling / reducers applied here).
```

Step 1 uses the sparse layout the lowerer baked from
`ConnRule.generate(...)`; it is identical for all four backends.
Steps 2/4/7 differ between `per_model` and `joint`; everything else is
shared. The returned `step` function is the *same shape* in both modes
— only its internals change. That keeps the time composer
backend-agnostic.

**Integrators.** A first-party set lives in `lowering/compose.py` and
is selectable by name: `"euler"`, `"exp_euler"`, `"rk2"`, `"rk4"`,
`"rk45_adaptive"`, plus the meta-choice `"exact_if_available"` (only
valid under `per_model`). Each integrator is a single function
`step(drift, state, inputs, dt) -> state'`. Adding an integrator
does not require touching backends or handlers.

---

## 12.6 The time composer

```python
def compose_time_loop(
    step_fn: StepFn,
    *,
    time_mode: Literal["py_loop", "scan"],
    record_strategy: RecordStrategy,
) -> RunFn: ...
```

**`py_loop`** (clock / clock_joint).
A Python `for i in range(num_steps): state, out = step_fn(...);
recorder.append(out)`. Recorders are imperative
`brainstate.Variable`-backed buffers (or Python lists for off-device
traces). Benefits: drop a `breakpoint()` mid-simulation, mutate from
outside, swap recorders without rebuilding. Cost: cannot be JIT'd as a
single artifact; Python overhead per step.

**`scan`** (clock_scan / bptt).
The whole time axis runs through `jax.lax.scan(step_fn, initial_state,
(t_array, inputs_array))`. The carry is `state`; the stacked output is
`step_outputs` per `dt`. Benefits: single JIT, autodiff-amenable,
fastest. Cost: `num_steps` is baked into the trace; observable
reducers must be pure; no mid-run Python intervention. Long traces use
`scan_chunks=K` (chunked scan, carry preserved).

---

## 12.7 The four backends

Each sibling backend is a `build()` pipeline that composes
`lowering/` pieces in a specific way. The pipelines are short and
mostly differ in three lines.

### 12.7.1 `clock` — per_model × py_loop

```python
# brainpy_state/clock.py
from brainpy_state import lowering

CLOCK_CAPS = BackendCapabilities(
    required_features=frozenset({"discrete_spike"}),
    requires_any_of={("ode_drift_fn", "exact_propagator")},
    trainable_mode="constant",        # Trainable -> frozen constant + TrainableIgnored
    supports_batch=True,
    supports_training=False,
    …
)

def build(ir, *, seed, dt,
          kind_options=None, node_options=None,
          variables=None, integrator="exact_if_available",
          **_) -> ClockSimulator:
    ctx       = BuildContext(seed=seed, dt=dt,
                             kind_options=kind_options or {},
                             node_options=node_options or {},
                             capabilities=CLOCK_CAPS)
    bound_ir  = lowering.variables.bind(ir, variables)
    net       = lowering.lower(bound_ir, ctx)
    lowering.trainable.freeze(net, notice="TrainableIgnored")
    step_fn   = lowering.compose_step_kernel(net,
                       step_mode="per_model", integrator=integrator, dt=dt)
    run_fn    = lowering.compose_time_loop(step_fn,
                       time_mode="py_loop",
                       record_strategy=ImperativeRecorders(net.observables))
    return ClockSimulator(ir=bound_ir, net=net, run_fn=run_fn,
                          seed=seed, dt=dt, bound_variables=…)
```

`ClockSimulator.run(duration)` translates `duration` to a step count and
calls `run_fn`. `ClockSimulator.reset()` re-initializes state from
`net.populations[*].state_spec.init(seed)`. `ClockSimulator.state()`
returns a frozen snapshot of the canonical state PyTree (§12.9).

### 12.7.2 `clock_joint` — joint × py_loop

Identical pipeline; one line changes:

```python
step_fn = lowering.compose_step_kernel(net,
              step_mode="joint", integrator="rk45_adaptive", dt=dt)
```

Capabilities: `required_features = {"ode_drift_fn", "discrete_spike"}`.
Refuses any handler whose features advertise `exact_propagator` but no
`ode_drift_fn` (raising `BackendCapabilityError` that names the
responsible populations and suggests `clock`). The joint-state vector
is one flat `jnp.ndarray` plus a static `StateMap`
`(population_id, field) → slice`, reified once at build and shared
across all readers (spike detector, observables, plasticity).

### 12.7.3 `clock_scan` — per_model × scan

```python
def build(ir, *, seed, dt, num_steps, batch_size=None,
          kind_options=None, node_options=None,
          variables=None, integrator="exact_if_available",
          online_inputs=False, **_) -> ScanSimulator:
    ctx       = BuildContext(…, capabilities=CLOCK_SCAN_CAPS)
    bound_ir  = lowering.variables.bind(ir, variables)
    net       = lowering.lower(bound_ir, ctx)
    lowering.trainable.freeze(net, notice="TrainableIgnored")
    step_fn   = lowering.compose_step_kernel(net,
                       step_mode="per_model", integrator=integrator, dt=dt)
    if batch_size is not None:
        step_fn = lowering.batching.vmap_step(step_fn, batch_size)
    inputs    = lowering.inputs.stage(net.inputs, num_steps,
                                       online=online_inputs)
    run_fn    = lowering.compose_time_loop(step_fn,
                       time_mode="scan",
                       record_strategy=ScannedRecorders(net.observables))
    return ScanSimulator(ir=bound_ir, net=net, run_fn=run_fn,
                         num_steps=num_steps, batch_size=batch_size, …)
```

`ScanSimulator.run()` (no `duration` arg — fixed at build) returns a
`TraceBundle` of `(T, [batch, ] …)` stacked arrays.

`lowering.inputs.stage` materializes every `InputNode` into a
`(T, [batch, ] …)` array eagerly, or — when `online_inputs=True` —
into a `jax.random.PRNGKey`-threading source held in the scan carry
(used for Poisson sources whose realizations should differ across
runs without re-tracing).

### 12.7.4 `bptt` — clock_scan + grad

Builds on `clock_scan`'s kernel, with three additions:

**(1) Trainable materialization.**
`lowering.trainable.materialize(net, mode="param")` replaces every
`Trainable` leaf with a `brainstate.nn.Param`; non-trainable leaves
stay constants. `params` is a flat `Mapping[dotted_name, ParamState]`
exposed via `trainer.parameters()`.

**(2) Surrogate enforcement.**
Every neuron handler used by the IR must publish the `surrogate_grad`
feature (else `BackendCapabilityError` naming the population, suggesting
the user attach a surrogate to the model or switch to `clock`). The
composer threads the surrogate into `spike_detect` via `jax.custom_vjp`:

```python
@jax.custom_vjp
def spike_detect(state, params, handler):
    return handler.spike(state, params)

def _fwd(state, params, handler):
    return spike_detect(state, params, handler), (state, params)

def _bwd(handler, residuals, g):
    state, params = residuals
    return (handler.surrogate(state, params) * g, None, None)

spike_detect.defvjp(_fwd, _bwd)
```

The forward pass returns the hard boolean spike; the backward pass
returns the surrogate-gradient pseudo-derivative. The surrogate is a
property of the **handler**, not the backend — a model author who
ships a new neuron family chooses the slope shape once, and every
training backend uses it consistently.

**(3) Loss / grad / optimizer.**

```python
class Trainer:
    def step(self, batch) -> StepReport:
        def loss_fn(params):
            state0 = self._init_state(batch)
            state_T, outputs = self.scan_run_fn(state0, params,
                                                batch.inputs)
            return self.loss(outputs, batch.targets), (state_T, outputs)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                                                self.params)
        self.params = self.optimizer.update(grads, self.params)
        return StepReport(loss=loss, grads=grads, aux=aux)
```

`Trainer.freeze(*paths)` / `Trainer.unfreeze(*paths)` toggle leaves in
and out of the optimizer's mask. There is no path-addressed write API
to the IR (RFC §1).

**Memory controls.** Three build kwargs on `bptt.build` — all are
*backend kwargs*, never in the IR:

- `truncated_bptt=K`: chunked scan with `jax.lax.stop_gradient` on the
  carry between chunks. Each chunk is differentiable; long horizons
  stay tractable.
- `checkpoint_every=N`: `jax.checkpoint` applied around every `N`-th
  step. Trades activation memory for recomputation.
- `precision="fwd_fp32_bwd_fp32" | "fwd_bf16_bwd_fp32" | …`: dtype
  policy on the kernel.

---

## 12.8 State PyTree contract (the cross-backend invariant)

`LoweredNet` defines the canonical state PyTree once; every backend
uses the same shape:

```
state = {
    "populations": {
        pop_id: <handler-defined leaf set,
                 units carried via jnp arrays whose dimension is
                 recorded in the handler's StateSpec>,
        …
    },
    "synapses":   { proj_id: { … } },
    "plasticity": { proj_id: { … } },
    "rng": <jax.random.PRNGKey, threaded through scan when any
            handler has the "stochastic_term" feature; absent otherwise>,
}
```

**Why uniform.** A state captured from `clock` at `t = 200 ms` can be
reloaded as the initial state of `bptt` (warm-start training from an
equilibrated simulation), and vice versa. The lowering library owns
the schema; backends are forbidden from inventing their own. This is
the strongest expression of P3: **the substrate, not the backend,
defines what "the network's state" means.**

---

## 12.9 Capability features and mismatch policy

### 12.9.1 Per-backend capability cards

| Required feature | clock | clock_joint | clock_scan | bptt |
|---|:---:|:---:|:---:|:---:|
| `discrete_spike` on each neuron handler | ✓ | ✓ | ✓ | ✓ |
| `ode_drift_fn` OR `exact_propagator` per population | ✓ | drift only | ✓ | ✓ |
| `surrogate_grad` on each neuron handler | – | – | – | ✓ |
| `discontinuous_reset` (if model has reset/refractory) | ✓ | ✓ | ✓ | ✓ |
| Inputs staged-array-compatible | – | – | ✓ | ✓ |
| `Trainable` materializes as | constant | constant | constant | Param |
| Adaptive dt | optional | ✓ | – | – |

### 12.9.2 Three classes of mismatch

1. **Hard miss — a required feature is absent.** The lowerer raises
   `BackendCapabilityError(node_id, missing_features, suggested_backend)`
   at `build()` (never at `run()` / `step()`). The error includes the
   IR node id and a concrete suggestion ("switch to `clock`", "attach a
   `surrogate` to your LIF model", "rewrite this neuron's dynamics with
   a `drift` term").
2. **Soft miss — an optional feature is absent and a documented
   fallback exists.** The lowerer emits a `BackendNotice` carrying the
   node id and the fallback choice (e.g., `"clock: 'LIF.exact_step'
   unavailable; using integrator='exp_euler' for this population."`).
   Notices are attached to `LoweredNet.notices` and to the built
   artifact (`sim.notices`, `trainer.notices`); they are NOT Python
   warnings, so determinism (G4) is preserved.
3. **Soft mismatch with no fallback — accepted but informational.**
   E.g., a `Trainable` marker reaching `clock` resolves to a frozen
   constant; the backend emits `TrainableIgnored(path)` and proceeds
   unless the `Trainable` was declared `required=True`, in which case
   it becomes a hard miss.

### 12.9.3 Discovery surface

```python
import brainpy.state.backend as backend

backend.list()                        # all registered backends
backend.list(kind="sim")              # filter by family
backend.get("clock_scan")             # resolve a module by name

caps = backend.get("bptt").capabilities
caps.required_features                # frozenset[...]
caps.report_for(ir)                   # -> (missing, fallbacks)  pre-flight check
```

`caps.report_for(ir)` runs the same capability check as `build()` but
without materializing state — useful for tooling (sweep plans, the
`brainpy explain` CLI subcommand, IR linters).

---

## 12.10 Variable and Trainable resolution

### 12.10.1 Variables (§3.14)

Build-time variables are resolved **before** lowering, by
`lowering.variables.bind(ir, variables)`. The bind step:

1. Validates that every declared `VariableDecl.name` is bound or has a
   default (else `SPEC-023`).
2. Validates units / dimensions against the declared default
   (`SPEC-024`).
3. Applies constraints (`positive`, `unit_norm`, `clip:lo,hi`;
   `SPEC-025`).
4. Rejects unknown keys in the user-supplied mapping (`SPEC-026`).
5. Substitutes `VariableRef` leaves with their concrete values, producing
   a new `NetIR` with the same `content_hash` (variables do not enter
   the hash — they are build inputs, not model identity).

After bind, the IR has no `VariableRef` leaves left.

### 12.10.2 Trainable

`lowering.trainable.materialize(net, mode)` operates on the `LoweredNet`,
not the IR. `mode` is set by the backend:

- `mode="constant"` (clock, clock_joint, clock_scan): every `Trainable`
  leaf becomes a frozen constant carrying the wrapped initial value;
  a `TrainableIgnored(path)` notice is emitted unless `required=True`.
- `mode="param"` (bptt): every `Trainable` leaf becomes a
  `brainstate.nn.Param` whose initial value is the wrapper's `value`
  (after `DistRef` materialization, if any). The flat
  `Mapping[dotted_name, ParamState]` is exposed via
  `trainer.parameters()`.

The dotted-name convention is `<role>.<node_id>.<state_or_param_name>`,
e.g., `population.exc.tau_m`, `projection.exc__to__inh.weight`. The
name is derived from the IR — it is stable across re-lowerings of the
same IR.

---

## 12.11 Determinism contract

Each backend commits to:

```
content_hash(bound_ir) ==
        content_hash(bind(ir, variables_A))
        == content_hash(bind(ir, variables_B))                    # G4(a)
artifact = backend.build(ir, seed, kind_options, node_options,
                         variables)
                                                                  # G4(b)
artifact'.run(...) == artifact.run(...)   ↔   same build inputs
```

Concretely:

- **Seed plan** (`lowering.seed.plan`) is deterministic given the build
  seed. Each population, projection, and stochastic input gets a
  derived key via `jax.random.fold_in(build_seed, h(node_id))`.
- **Connectivity sampling** uses the derived projection key. Sparse
  layouts (CSR/COO/dense) are chosen by edge density, not by the
  iteration order of the connectivity rule.
- **Notices ordering** is by IR-node-id sort, not by lowering visit
  order, so two builds of the same IR emit identical notices in the
  same order.

The `bptt` backend additionally commits that under the same `seed`,
optimizer, and batch sequence, the trained parameter values after
`N` steps are bit-identical across runs (modulo non-deterministic
GPU reductions, which users opt out of with the `xla_deterministic`
build flag).

---

## 12.12 Examples

### 12.12.1 Simulation

```python
import brainpy.state as bs
from brainpy.state import clock

net = bs.NetSpec("brunel")
exc = net.population("exc", "LIF", size=8000, params={"tau_m": 20*u.ms})
inh = net.population("inh", "LIF", size=2000, params={"tau_m": 10*u.ms})
net.project(exc, exc, rule="FixedProb", weight=0.5*u.mV, prob=0.1)
net.project(exc, inh, rule="FixedProb", weight=0.5*u.mV, prob=0.1)
net.project(inh, exc, rule="FixedProb", weight=-2.5*u.mV, prob=0.1)
net.observe(exc, "spike")
ir = net.finalize()

sim = clock.build(ir, seed=0, dt=0.1*u.ms)
traces = sim.run(1.0*u.s)
print(sim.notices)             # any fallbacks the lowerer chose
```

### 12.12.2 Training (BPTT) — same IR, one-line backend swap

```python
from brainpy.state import bptt

# Same IR; mark per-projection weight as Trainable in the spec.
net.project(exc, exc, rule="FixedProb",
            weight=bs.Trainable(0.5*u.mV), prob=0.1)
ir = net.finalize()

trainer = bptt.build(ir, seed=0, dt=1.0*u.ms, num_steps=200,
                     batch_size=32,
                     loss=my_loss_fn,
                     truncated_bptt=50,
                     checkpoint_every=10)
for batch in loader:
    report = trainer.step(batch)
```

### 12.12.3 Capability mismatch

```python
# A user-defined neuron with neither exact_step nor a drift function.
trainer = bptt.build(ir, seed=0, ...)
# raises:
# BackendCapabilityError:
#   Node 'population.weird_neuron' (kind 'MyNeuron') is missing required
#   features for backend 'bptt': {'ode_drift_fn', 'surrogate_grad'}.
#   Suggested fixes:
#     - attach a `drift` function to MyNeuron's handler;
#     - attach a surrogate (e.g. braintools.surrogate.fast_sigmoid()) to MyNeuron;
#     - or switch to `clock` if BPTT training is not required.
```

---

## 12.13 What this chapter intentionally leaves open

These items are deferred to follow-up chapters and are not load-bearing
for the sim + BPTT scope:

- **Event-driven backend (`event`)** — consumes the same `LoweredNet`
  but replaces `compose_step_kernel` with an event scheduler. Its
  capability card will require `exact_propagator` and a
  `next_spike_time(state, params)` feature on neuron handlers.
- **Event-prop training (`eventprop`)** — like `bptt`, but the
  surrogate-gradient path is replaced by exact spike-time gradients.
  Composable on top of `event`'s kernel.
- **E-prop (`eprop`)** — adds a local-learning-rule composer to the
  step kernel; trains without an outer `jax.grad`. Will require new
  features on neuron handlers (`eligibility_kernel`,
  `pseudo_derivative`).
- **pp-prop (`ppprop`)** — see `braintrace`; uses RTRL / forward-mode
  through the same kernel.

Each of these is a sibling backend that consumes the same handler
protocol and the same `LoweredNet`; none of them require changes to
the `lowering/` substrate or to the sim and bptt backends specified
here.

---

**Previous:** [Chapter 11 — Appendix](./11-appendix.md)
**Next:** (forthcoming) Chapter 13 — Event-driven and event-prop backends
