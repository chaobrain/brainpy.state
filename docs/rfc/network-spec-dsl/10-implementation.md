# Chapter 10 — Implementation: codebase mapping, testing, and _network relationship

> Part of the [Network Specification DSL RFC](./README.md).

## 10.1 Mapping to the existing codebase

```
brainpy_state/                               (TOP-LEVEL package)
├── __init__.py                              re-exports: spec, NetSpec, load, NetIR,
│                                            ParamPatch, viz; and every backend
│                                            (clock, event, bptt, eprop, eventprop,
│                                             ppprop, nir, onnxspike)
├── backend.py                               (NEW) Protocols (SimBackend,
│                                            TrainBackend, ExportBackend),
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
├── nir.py                                   export backend — Neuromorphic IR (§6)
├── onnxspike.py                             export backend — ONNX-spike (future)
│
├── spec/                                    DSL surface ONLY — no execution code
│   ├── __init__.py                          re-exports: NetSpec, load, NetIR, SpecError,
│   │                                        train, merge, split, concat, ParamPatch.
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
│   └── params.py                            ParamPatch, ParameterView (§3.14)
│
├── viz/
│   ├── graph.py
│   ├── layers.py
│   ├── matrix.py
│   ├── params.py
│   └── nir.py
│
├── cli.py                                   brainpy entry point
│
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

    The _network module remains the runtime substrate of brainpy_state.clock.
    Public symbols stay importable; users may keep using `Builder` directly.
```

**Why backends are top-level, not nested under `spec/`.** The spec
module is the paradigm-neutral DSL surface; backends commit to a
specific runtime / gradient flavor and pull in heavyweight
dependencies (JAX training, NIR libraries, event-prop machinery).
Keeping them out of `spec/` means `import brainpy.state.spec` stays
lightweight, and switching gradient paradigms is a one-line
`from brainpy.state import <backend>` change (D29).

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
- **Live vs rebuild classification** — for each leaf class in §3.14.5,
  the corresponding backend reports the expected classification.
  `REBUILD` writes raise SPEC-024 with a path-accurate message;
  `LIVE`/`LIVE_RESET` writes propagate to a subsequent `sim.run(...)`.

Test file layout follows the existing convention: colocated `*_test.py`.

---

## 10.3 Relationship to the existing `_network` API

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
  `brainpy.state.clock.build()`. User code that imports `Builder`
  directly needs no changes.
- Documentation and examples migrate to `NetSpec` as the recommended
  entry point.
- The `brainpy.state` top-level namespace gains:
  - DSL surface: `spec`, `NetSpec`, `load`, `train`, `merge`, `split`,
    `concat`, `ParamPatch`, `viz`.
  - Backend protocol and discovery: `backend` (singular —
    `SimBackend`, `TrainBackend`, `ExportBackend`, `backend.list`,
    `backend.get`).
  - Backend implementations as **peer top-level modules**: `clock`,
    `event`, `bptt`, `eprop`, `eventprop`, `ppprop`, `nir`,
    `onnxspike`. None of these live under `brainpy.state.spec`.
  - Existing symbols (`LIF`, `Expon`, `COBA`, `Builder`, `OneToOneProj`,
    `FixedIndegreeProj`, …) keep their current paths.

---


---

**Previous:** [Chapter 9 — Determinism and validation](./09-determinism-validation.md)  
**Next:** [Chapter 11 — Appendix: decisions, cheat sheet, open questions](./11-appendix.md)
