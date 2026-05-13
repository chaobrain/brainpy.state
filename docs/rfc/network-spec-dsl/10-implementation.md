# Chapter 10 — Implementation: codebase mapping, testing, and relationship to existing modules

> Part of the [Network Specification DSL RFC](./README.md).

## 10.1 Mapping to the existing codebase

```
brainpy_state/                               (TOP-LEVEL package)
├── __init__.py                              re-exports: spec, NetSpec, load, NetIR,
│                                            VariableRef, viz; and every backend
│                                            (clock, event, bptt, eprop, eventprop,
│                                             ppprop)
├── backend.py                               (NEW) Protocols (SimBackend,
│                                            TrainBackend),
│                                            BackendCapabilities, entry-point loader,
│                                            backend.list() / backend.get(name)
│                                            — top-level, NOT under spec/
│
├── clock.py                                 sim backend — adapter over Network/Builder
├── event.py                                 sim backend — event-driven simulator
├── bptt.py                                  train backend — autodiff through surrogates
├── eprop.py                                 train backend — eligibility-trace training
├── eventprop.py                             train backend — event-based exact gradients
├── ppprop.py                                train backend — see braintrace
│
├── spec/                                    DSL surface ONLY — no execution code
│   ├── __init__.py                          re-exports: NetSpec, load, NetIR, SpecError,
│   │                                        train, merge, split, concat, VariableRef.
│   │                                        Does NOT re-export any backend.
│   ├── ir.py                                NetIR + all node dataclasses
│   ├── netspec.py                           Frontend A: NetSpec + handles
│   ├── yaml_loader.py                       Frontend B: load(), to_yaml()
│   ├── schema/
│   │   └── netir-1.0.json                   JSON Schema
│   ├── registry.py                          neuron / synapse / output / input / layer /
│   │                                        connectivity / initializer registries
│   ├── connect/
│   │   ├── __init__.py                      re-exports: braintools.conn.* + supplementary
│   │   └── supplementary.py                 FixedIndegree, FixedOutdegree,
│   │                                        FixedTotalNumber, PairwisePoisson,
│   │                                        SymmetricPairwiseBernoulli
│   └── variables.py                         VariableRef, VariableDecl, and the
│                                            build-time resolver that substitutes
│                                            bound values during backend.build (§3.14)
│
├── viz/
│   ├── graph.py
│   ├── layers.py
│   ├── matrix.py
│   └── params.py
│
└── cli.py                                   brainpy entry point

brainpy_state/_network/
├── _base.py                                 (unchanged) Network base class
├── _builder.py                              (unchanged + connect_from_result helper)
├── _projections.py                          rewritten as thin facade over
│                                            brainpy_state.spec.connect rules
├── _recorders.py                            (unchanged)
└── _connectivity.py                         REMOVED — rules moved to spec/connect/

    The _network module remains the runtime substrate of brainpy_state.clock.
    Public symbols stay importable; users may keep using `Builder` directly.

brainpy_state/_brainpy/                      (unchanged)
brainpy_state/_nest/                         (unchanged)
```

**Why backends are top-level, not nested under `spec/`.** The spec
module is the paradigm-neutral DSL surface; backends commit to a
specific runtime / gradient flavor and pull in heavyweight
dependencies (JAX training, event-prop machinery). Keeping them out
of `spec/` means `import brainpy.state.spec` stays lightweight, and
switching gradient paradigms is a one-line
`from brainpy.state import <backend>` change (D22).

`brainpy_state.clock` does roughly:

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

## 10.2 Testing strategy

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
  spec using it, calls `net.finalize()` and `clock.build(...)`, and
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
  2-layer spiking CNN built via `net.sequential(...)` finalize to an IR
  whose `compounds.sequentials` recovers the layer order. Spiking MLP on
  MNIST trains under `bptt` to ≥ 90% test accuracy in CI.
- **Visualization determinism** — `spec.viz(ir, mode=M, renderer=R, seed=S)`
  produces byte-identical output for `(M, R, S)` triples across Python
  versions in CI; golden artifacts checked in for one Mermaid and one
  Graphviz example.
- **Backend equivalence** — Brunel runs on `clock` and `event` with
  population firing rates within 2 σ over 1 s @ 10 trials.
- **Cross-module model coverage** — every neuron / synapse class
  exported from `brainpy_state._brainpy` and `brainpy_state._nest` is
  exercised through the spec surface: a `Population` referencing the
  model by `kind` finalizes, builds on `clock`, and matches a direct
  `_brainpy` / `_nest` construction byte-for-byte in trace output.
- **Capability mismatch** — every backend declares `capabilities`; tests
  assert that a known-unsupported feature on each backend raises
  `BackendCapabilityError` with the expected node id.
- **Variable binding (§3.14)** — for every backend: building the same
  IR with the same `variables={...}` mapping twice produces
  bit-identical artifacts; building with two distinct mappings against
  the same IR yields the same `content_hash`. Required variables not
  supplied raise SPEC-023; wrong-dimension values raise SPEC-024;
  constraint violations raise SPEC-025; unknown keys raise SPEC-026.
  Each error names the offending variable.
- **Spec immutability** — `NetIR` and `NetSpec` instances reject all
  attribute writes after construction (`frozen=True` enforced); there
  is no `update` / `patch` / `parameters.set` surface to test the
  *absence* of, but the missing-method tests assert these names are
  not present on either type.

Test file layout follows the existing convention: colocated `*_test.py`.

---

## 10.3 Relationship to the existing `_network` / `_brainpy` / `_nest` APIs

The spec module is a new layer **above** the existing modules; nothing
under `brainpy_state._network`, `brainpy_state._brainpy`, or
`brainpy_state._nest` is replaced. The relationship per subpackage:

### 10.3.1 `_network/` — wiring layer

This subpackage is the imperative wiring substrate. The spec module
treats it as the runtime of the `clock` backend:

- **`Network`** (`_network/_base.py`) — unchanged. `brainpy.state.clock`
  builds a `Network` from the IR and steps it via
  `brainstate.environ['t']`. Direct `Network` users are unaffected.
- **`Builder`** (`_network/_builder.py`) — kept; gains one helper,
  `Builder.connect_from_result(...)`, that consumes a
  `braintools.conn.ConnectionResult` directly. Existing
  `Builder.add(...)` / `Builder.connect(...)` calls are unchanged.
- **`*Proj` rule classes** (`_network/_projections.py`) — public class
  signatures are preserved (`OneToOneProj`, `AllToAllProj`,
  `PairwiseBernoulliProj`, `SymmetricPairwiseBernoulliProj`,
  `FixedIndegreeProj`, `FixedOutdegreeProj`, `FixedTotalNumberProj`,
  `PairwisePoissonProj`); their internals are rewritten as thin facades
  over the canonical `braintools.conn` rule of the same shape plus
  `Builder.connect_from_result`. User code importing these names
  continues to work.
- **`Recorder`** (`_network/_recorders.py`) — unchanged. The spec
  observable nodes (§3.8) lower to `Recorder` invocations under
  `clock`.
- **`_connectivity.py`** — **removed**. The bare samplers
  (`sample_one_to_one`, `sample_pairwise_bernoulli`,
  `sample_fixed_indegree`, …) are duplicates of `braintools.conn`
  rules. Anything not yet upstream moves to
  `brainpy_state/spec/connect/supplementary.py` as
  `braintools.conn.PointConnectivity` subclasses. The legacy import
  path `brainpy_state._network._connectivity` is dropped — code that
  imported it must now import from `brainpy_state.spec.connect` or
  `braintools.conn`. (D16)

### 10.3.2 `_brainpy/` — BrainPy-style point models

This subpackage is the model library for the BrainPy-style lineage
(LIF / ALIF / ExpIF / AdExIF / HH / Izhikevich / …, plus `Expon` /
`Alpha` / `AMPA` / `GABAa` / `BioNMDA` synapses, `COBA` / `CUBA` /
`MgBlock` outputs, `STP` / `STD` plasticity, and the input / readout
generators). The spec module does **not** redefine any of these
models; it references them by `kind` string through the registry
(§7.3):

- Each public class is auto-registered at import time. The PascalCase
  class name becomes the IR `kind`: `LIF` → `kind="LIF"`,
  `Expon` → `kind="Expon"`, `LeakyRateReadout` → `kind="LeakyRateReadout"`,
  and so on.
- The registry holds the parameter signature (names, units,
  trainability metadata) alongside a `source=` pointer back to the
  concrete class. `_instantiate_neuron`, `_instantiate_syn`, etc.
  resolve the `kind` and call the underlying constructor with the
  unit-bearing parameters from the IR.
- `brainstate`-level state objects, `add_current_input` /
  `add_delta_input` composition, surrogate gradients, and the existing
  `update(x)` contract remain authoritative. The spec does not impose
  a parallel implementation — it is metadata over the same classes.
- Direct `brainpy_state._brainpy` users are unaffected: the spec layer
  is opt-in, and existing scripts that construct `LIF(...)` /
  `PoissonSpike(...)` / `Projection(...)` directly continue to work.

### 10.3.3 `_nest/` — NEST-compatible models

This subpackage is the model library for NEST-compatible neurons,
synapses, plasticity rules, and devices (`iaf_psc_alpha`,
`aeif_cond_exp`, `hh_psc_alpha`, `stdp_synapse`,
`spike_recorder`, …). It plugs into the spec module on exactly the
same terms as `_brainpy/`:

- Each public class registers under its lowercase NEST-style `kind`
  string (`iaf_psc_alpha`, `aeif_cond_exp`, …). The registry signature
  declares unit-bearing parameters (`saiunit` quantities) and
  trainability annotations.
- `NESTDevice`, `NESTNeuron`, `NESTSynapse`, `NESTPlasticity`
  (defined in `_nest/_base.py`) are the marker base classes the
  registry uses to route a `Population` / `Projection` / `InputSource`
  / `Observable` node to the right wiring code under `clock`.
- The spec module reuses (does not duplicate) the existing NEST-side
  conventions documented in [CLAUDE.md](../../../CLAUDE.md):
  `saiunit` for units, `AdaptiveRungeKuttaStep` for ODE integration,
  `DotDict` for state packing, `is_tracer()`-guarded validation,
  `brainstate.transform.jit_error_if()` for JIT-safe errors.
- A spec built around NEST-compatible kinds runs through the same
  `clock` backend as a BrainPy-style spec — there is no separate
  "NEST" backend. The two model families are interchangeable from the
  spec's point of view; only the registered kind strings differ.

What stays at the top level of `brainpy.state`:

- DSL surface: `spec`, `NetSpec`, `load`, `train`, `merge`, `split`,
  `concat`, `VariableRef`, `viz`.
- Backend protocol and discovery: `backend` (singular —
  `SimBackend`, `TrainBackend`, `backend.list`, `backend.get`).
- Backend implementations as **peer top-level modules**: `clock`,
  `event`, `bptt`, `eprop`, `eventprop`, `ppprop`. None of these live
  under `brainpy.state.spec`.
- Existing symbols (`LIF`, `Expon`, `COBA`, `Builder`, `OneToOneProj`,
  `FixedIndegreeProj`, every `_brainpy` model, every `_nest` model)
  keep their current import paths.

---


---

**Previous:** [Chapter 9 — Determinism and validation](./09-determinism-validation.md)  
**Next:** [Chapter 5 — Domain extensions: the domain-pack contract](./05-domain-extensions.md)
