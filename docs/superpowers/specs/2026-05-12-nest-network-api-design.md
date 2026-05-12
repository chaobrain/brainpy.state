# Network API for brainpy.state NEST-style models — design

**Date:** 2026-05-12
**Status:** design approved, pending implementation plan
**Related:** `docs/nest-status/internal/network-api-gap.md`,
`docs/nest-status/internal/index.md` (consolidated roadmap)

## 1. Goal

Provide a brainpy.state-native API for assembling NEST-style neuron, synapse,
plasticity, and device modules into networks, running their dynamics, and
recording their outputs. Two public entry points (declarative subclass and
imperative builder) over a single underlying object.

This design is the *foundation*. A future PyNEST-style `nest_compat` shim
(separate effort, P0 in `network-api-gap.md`) will sit on top of this layer.

## 2. Non-goals

- A PyNEST `Create`/`Connect`/`Simulate` facade — covered by the deferred
  `nest_compat` shim.
- `TripartiteConnect`, `CollocatedSynapses` (P1 in the gap roadmap).
- Spatial / topology surface (`nest.spatial.*`) — P2.
- Lazy NEST-style `Parameter` expressions (handled by `nest_compat`).
- MPI / SONATA / structural plasticity — out of scope per `network-api-gap.md` §7.

## 3. Architecture

### 3.1 Two entry points, one object

- `brainpy.state.Network` — base class, subclasses
  `brainstate.nn.Module`. Authors subclass it to define reusable,
  parameterizable networks. **Canonical style.**
- `brainpy.state.Builder` — subclass of `Network` that adds two imperative
  methods, `add()` and `connect()`. Use it directly for scripts and notebooks.
  Instances are first-class `Network`s, so `simulate()`, JIT, vmap behave
  identically.

Both produce a brainstate module tree. The choice is purely ergonomic; there
is no mode flag, no precedence rule, no parallel registry.

### 3.2 Execution model

- `Network.update(t=None)` runs one timestep. It walks the module tree in the
  existing brainpy.state execution order — projections first, then dynamics —
  matching the convention documented at
  `brainpy_state/_brainpy/projection.py:46-51`.
- `Network.simulate(duration, *, dt=None, monitor=None)` is documented sugar:
  it JIT-wraps a `brainstate.compile.for_loop` over `update`. It does **not**
  introduce a separate stepping abstraction — power users can ignore it and
  call `update` from their own loop with identical results.

### 3.3 Module placement

```
brainpy_state/
└── _network/
    ├── __init__.py
    ├── _base.py           # Network
    ├── _builder.py        # Builder
    ├── _projections.py    # OneToOneProj, AllToAllProj, FixedIndegreeProj, ...
    ├── _connectivity.py   # internal index/mask generators
    ├── _recorders.py      # Recorder helper that wires NESTDevice recorders
    └── *_test.py
```

Top-level exports (all set `__module__ = 'brainpy.state'`):
`Network`, `Builder`, `Recorder`, and every `*Proj` class listed in §5.

## 4. `Network` and `Builder` interfaces

```python
class Network(brainstate.nn.Module):
    def update(self, t=None) -> None: ...
    def simulate(self,
                 duration,                # saiunit.Quantity
                 *,
                 dt=None,                 # optional override; defaults to brainstate.environ
                 monitor=None,            # None | list[str] | dict[str, Callable[[Network], Array]]
                 ) -> dict[str, Array]: ...
    def reset_state(self, *batch_shape) -> None: ...

    @property
    def populations(self) -> dict[str, NESTNeuron]: ...
    @property
    def projections(self) -> dict[str, Projection]: ...
    @property
    def devices(self) -> dict[str, NESTDevice]: ...

class Builder(Network):
    def add(self, name: str, module: Module) -> Module:
        """Register `module` as self.<name>; return the module instance."""
    def connect(self, pre, post, *, rule: type[Projection], **kwargs) -> Projection:
        """Instantiate `rule(pre, post, **kwargs)` and register it as an
        auto-named attribute. `pre`/`post` accept module references or
        string names previously registered via `add()`."""
```

**Naming rules.** `add(name, module)` requires an explicit name so attributes
are stable across reruns and JIT recompiles. `connect(...)` auto-names
projections (`_proj_0`, `_proj_1`, …). To give a projection a stable name,
construct it explicitly and use `add('e2i', FixedIndegreeProj(...))`.

**`rule=` is a class, not a string.** Symmetry with how subclass-style authors
write `self.e2i = FixedIndegreeProj(...)`. Strings remain reserved for the
future `nest_compat` shim, which is the layer that adopts NEST's string-based
naming.

**Introspection properties** are derived from the module tree by filtering
on base class (`NESTNeuron`, `Projection`, `NESTDevice`). No parallel
bookkeeping.

## 5. Projection class inventory

One class per NEST connection rule. Uniform constructor:

```python
RuleProj(pre, post, *,
         weight,                  # scalar | array | Distribution
         delay=None,              # scalar | array | None
         syn,                     # ParamDescriber[NESTSynapse]
         out,                     # ParamDescriber[SynOut]
         allow_autapses=True,
         allow_multapses=True,
         seed=None,
         **rule_specific)
```

Initial inventory (parity with `network-api-gap.md` §3.9):

| Class | Rule-specific kwargs | NEST rule |
|---|---|---|
| `OneToOneProj` | — | `one_to_one` |
| `AllToAllProj` | — | `all_to_all` |
| `PairwiseBernoulliProj` | `p: float` | `pairwise_bernoulli` |
| `SymmetricPairwiseBernoulliProj` | `p: float` | `symmetric_pairwise_bernoulli` |
| `FixedIndegreeProj` | `K: int` | `fixed_indegree` |
| `FixedOutdegreeProj` | `K: int` | `fixed_outdegree` |
| `FixedTotalNumberProj` | `N: int` | `fixed_total_number` |
| `PairwisePoissonProj` | `mean: float` | `pairwise_poisson` |

Each subclass composes the existing brainpy.state pipeline (`AlignPostProj` /
`DeltaProj` / `CurrentProj`). The rule lives in a private `Connectivity`
helper (`_network/_connectivity.py`) that constructs the index/mask array at
`__init__` and stores it as a `brainstate.State` (so JIT traces it as data,
not Python).

`allow_autapses=False` removes diagonal edges when `pre is post`.
`allow_multapses=False` enforces at-most-one edge per (pre, post) pair.
These flags follow NEST semantics so a future `nest_compat` shim can pass
them through unchanged.

Out of scope for this design: `TripartiteConnect` (P1), `CollocatedSynapses`
(P1), CSA/`conngen`, spatial-distance rules.

## 6. Synapse / output composition

`syn` and `out` are `ParamDescriber` instances, using the existing pattern
from `brainpy_state/_brainpy/projection.py`. Example:

```python
class Brunel(bp.state.Network):
    def __init__(self):
        super().__init__()
        self.exc = LIF(8000)
        self.inh = LIF(2000)
        self.e2i = bp.state.FixedIndegreeProj(
            self.exc, self.inh, K=800,
            weight=0.1 * u.nS,
            delay=1.5 * u.ms,
            syn=bp.state.Expon.desc(tau=5*u.ms),
            out=bp.state.COBA.desc(E=0*u.mV),
        )

Brunel().simulate(1 * u.second)
```

The descriptors are instantiated once per projection, aligned to the post-
population. This is the same mechanism used today by `AlignPostProj`; the new
classes are sugar that pick the connectivity matrix.

## 7. Weights and delays

- **Weight**: scalar (broadcast), array shaped to the connectivity, or a
  `bp.state.dist.{Normal, LogNormal, Uniform, …}` object that is sampled
  **once** at projection `__init__` (not lazily at runtime). All values
  carry `saiunit` units.
- **Delay**: scalar (uniform) or per-edge array. Backed by brainstate's
  existing delay container. Distributions sampled at init using the same
  pattern.
- **Seed**: a single `seed=` kwarg on the projection drives both the
  connectivity draw and the weight/delay distributions (when used).
  Deterministic across reruns for a fixed seed.

This deliberately diverges from NEST's lazy `Parameter`: simpler semantics,
JIT-compatible, deterministic given a seed. NEST-style lazy expressions are
deferred to the future `nest_compat` shim.

## 8. Devices

Two flavors of `NESTDevice` subclass exist in `brainpy_state/_nest/` today,
and the network layer treats them differently because their existing APIs
differ:

### 8.1 Generator devices

Two sub-flavors with different wiring semantics:

- **Spike-emitting generators** (`spike_generator`, `poisson_generator`,
  `gamma_sup_generator`, …) expose a `spike`-like interface and fit
  directly as `pre` in any `*Proj` constructor — they flow through the
  synapse pipeline like any other source population.
- **Current-emitting generators** (`ac_generator`, `dc_generator`,
  `step_current_generator`, …) emit continuous current with no spike
  interface. They wire to a target population via the existing
  `add_current_input` mechanism: the user holds the generator as a
  Network attribute, and the target neuron pulls its current each step.
  The precise binding API (constructor `target=` kwarg vs. a tiny
  `CurrentGeneratorProj` wrapper) is finalized in the implementation
  plan after reading the existing `ac_generator` / `dc_generator` test
  usage — both options preserve the existing generator class signatures.

### 8.2 Recorder devices (`spike_recorder`, `multimeter`, …)

The existing `spike_recorder.update(spikes, senders, offsets)` signature
expects the user to pass data in explicitly — recorders are *passive*. The
network layer adds a small wiring helper:

```python
bp.state.Recorder(source, attr='spike', device=<NESTDevice instance>)
```

`Recorder` is a `Module` whose `update()` reads `getattr(source, attr).value`
and dispatches it to the underlying device. After `simulate()` returns, the
device's recorded data is accessible the same way as a standalone NEST
device — e.g., `self.rec.device.events`.

This keeps the existing `_nest/` recorder classes unchanged (passive,
NEST-faithful signatures) and gives Network authors a one-line wiring
primitive. Authors who prefer to wire recorders by hand can do so by
overriding `update()`.

### 8.3 `monitor=` kwarg on `simulate()`

For lightweight recording without instantiating a NESTDevice, `simulate()`
accepts a `monitor` argument:

- `monitor=None` (default) — no monitoring, only NESTDevice recorders capture
  data.
- `monitor=["exc.spike", "inh.V"]` — list of attribute paths; returned dict
  has one stacked-across-time array per path.
- `monitor={"r_exc": lambda net: net.exc.spike.value.mean()}` — dict of
  callables; returned dict applies each callable per timestep and stacks.

`monitor` is implemented inside `simulate()` only — it does not affect
`update()` semantics. It is sugar for the common case
`for_loop(lambda t: (net.update(t), net.<attr>.value), times)`.

## 9. Error handling

- Constructor validation runs eagerly (no tracer): pre/post shape mismatches,
  invalid rule kwargs, unit mismatches via `saiunit`, autapse/multapse flag
  feasibility.
- Inside `update()`, runtime guards use `brainstate.transform.jit_error_if`
  per project convention from `CLAUDE.md`.
- `simulate()` validates that `dt` is resolvable (either via `dt=` kwarg or
  `brainstate.environ`) before compilation.

## 10. Testing strategy

Tests live colocated as `*_test.py` per project convention, run by pytest
with `unittest.TestCase` classes.

- **Per-rule unit tests** (`brainpy_state/_network/<rule>_proj_test.py`):
  edge-count statistics, autapse/multapse flag enforcement, seed
  determinism, weight/delay broadcasting.
- **`Network` / `Builder` parity test**: build the same Brunel network both
  ways; assert identical module trees and identical simulated output for the
  same seed.
- **`update()` vs `simulate()` parity test**: confirm a custom `for_loop`
  over `update()` produces output bit-identical to `simulate()` for the same
  duration, dt, seed, and monitor specification.
- **Brunel flagship example** (`examples/brunel.py`): uses
  `FixedIndegreeProj` for the recurrent connectivity and the `Recorder`
  helper for spike output. Doubles as the IAF-psc-family flagship per
  `nest-status/internal/index.md` P0 #17, and is the first network-level
  acceptance test in the validation harness.

## 11. Documentation deliverables

- Module-level docstring on `_network/_base.py` summarizing the two-entry-
  point design and the projection-rules inventory.
- NumPy-style docstrings on every public class (`Network`, `Builder`,
  `Recorder`, every `*Proj`), per project convention.
- A new page in `docs/brainpy-guide/` introducing the Network API with the
  Brunel example side-by-side in both styles.
- A link from `docs/nest-status/internal/network-api-gap.md` §7 noting that
  the P0 foundation is in place (once shipped) and the `nest_compat` shim
  can now be built on top.

## 12. Open questions deferred to the implementation plan

- Exact projection-execution order when a Network contains both projections
  whose `post` is another projection's `pre` (chained dependencies). Existing
  `brainstate.nn.Module` traversal handles flat dependency graphs; chained
  cases need a documented contract before implementation.
- Whether `Recorder` should support attribute paths (`source.attr` with
  dots) for nested attributes, or strictly one-level lookup. Default to one
  level until a real example demands more.
- Pickling / state save-load story for `Network` (likely deferred — falls
  out of brainstate's existing serialization).
