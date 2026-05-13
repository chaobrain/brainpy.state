# Chapter 11 — Appendix: decision log, cheat sheet, open questions

> Part of the [Network Specification DSL RFC](./README.md).

## 18. Decision log

| ID  | Decision                                                          | Resolution |
|-----|-------------------------------------------------------------------|-----------|
| D1  | `dt` placement                                                    | Backend.build kwarg. Pinning in the spec leaks runtime choice into G1; event backends ignore it. |
| D2  | `seed` placement                                                  | Backend.build kwarg, with per-projection `seed` override allowed in the IR for reproducibility of partial sub-graphs. |
| D3  | Module path                                                       | `brainpy_state.spec`, exported as `brainpy.state.spec`. Coexists with `brainpy.state.Builder`. |
| D4  | CLI name                                                          | `brainpy`. |
| D5  | Cross-ref style                                                   | IR uses strings (population ids, dotted paths); B uses handles; D uses string ids. |
| D6  | Custom user models                                                | Registry decorators + entry-point groups (§11.5). |
| D7  | YAML subnetwork parameterization                                  | `!include` + an explicit `params:` map per instance. Templating (Jinja / Hydra) is opt-in via `brainpy sweep`. |
| D8  | Mutable parameters for training                                   | IR stays frozen. Trainers expose a `parameters()` view; updates happen in trainer state, not in the IR. |
| D9  | View granularity                                                  | Slice / index / merge / reshape are fields on `ViewRef`. No separate `View` node type. Merge views denormalize into one `ProjectionNode` per member at finalize. |
| D10 | Connectivity source library                                       | `braintools.conn` is canonical. Auto-registers every `Connectivity` subclass; weight/delay live on the rule. Supplementary rules ship from `brainpy_state` and are tracked for upstreaming. |
| D11 | Initializer source library                                        | `braintools.init` is canonical. Auto-registers every `Initialization` subclass; `DistRef` lowers to a concrete `Initialization` at backend build. |
| D12 | Weight/delay precedence                                           | Canonical home is `ConnRule.params`. Projection-level `weight=` / `delay=` are sugar merged at finalize. Conflicts raise SPEC-016. |
| D13 | Trainable surface                                                 | `Trainable` is a value-level wrapper, applicable to any leaf in `ModelRef.params`, `ConnRule.params`, `PopulationNode.init`, and `InputNode.weight` / `InputNode.source.params`. Trainability metadata lives in registry signatures. |
| D14 | Trainable storage                                                 | Every `Trainable` materializes as `brainstate.nn.Param` on the synthesized `brainstate.nn.Module`. Trainers collect via `state.tree_states(brainstate.ParamState)`. |
| D15 | Layer macros                                                      | Ship the §6.7 set covering deep-SNN essentials. Third-party macros via entry points. |
| D16 | Visualization default renderer                                    | Mermaid (no runtime deps) when Graphviz is not installed; Graphviz when available. HTML renderer for interactive `brainpy viz --renderer html`. |
| D17 | NIR as the canonical export target                                | Yes. Other export targets (`onnx-spike`, `nengo`, `lava`) implement the same `ExportBackend` protocol but are not required to ship with the spec library. |
| D18 | Lossy-export policy                                               | Six-class taxonomy (§9.4). Strict mode is opt-in (`--strict`) and elevates classes `APPROXIMATE`, `EXTENSION`, `DROPPED`, `UNSUPPORTED` to errors. Lenient mode is default and ships notices. |
| D19 | NIR units                                                         | NIR is unit-agnostic. The exporter strips to canonical SI and writes a sidecar (`<name>.nir.meta.json`) preserving the original units and trainable / seed / compound metadata. |
| D20 | NIR import                                                        | Not in scope. The sidecar enables partial reconstruction for round-trip testing only; production round-trip is not supported because `UNSUPPORTED` and `DROPPED` losses are unrecoverable. |
| D21 | Loss / optimizer integration                                      | **User-side.** Losses and optimizers are plain Python callables passed to `bptt.build(..., loss=...)`. No `brainpy.state.spec.loss` / `…optim` modules ship with the spec. |
| D22 | NIR export default seed                                           | Defaults to the simulator's default seed (`sp.spec.DEFAULT_SEED`, currently `0`). Overridable in Python via `export(..., seed=N)` and on the CLI via `brainpy export --seed N`. |
| D23 | Reverse-compatibility shim for `_network/_connectivity.py`        | **Removed.** Supplementary rules move to `brainpy_state/spec/connect/supplementary.py`. The legacy import path is dropped. |
| D24 | Spike-time input encoding in NIR                                  | Emitted as a NIR extension node `nir.brainx.SpikeTimes(times=...)` under our reserved `nir.brainx.*` namespace. The sidecar mirrors the table for consumers that ignore extensions. |
| D25 | Quantized weights for hardware                                    | **Per-export-backend concern.** The core spec does not carry a `quantize` flag. Each export backend (e.g. a future Loihi-targeted exporter) may consume `Trainable.constraint` strings (e.g. `"int8"`, `"fixed:Q4.4"`) or apply its own quantization config. |
| D26 | Post-definition parameter modification — interface               | One path language (§6.9.1) and one `ParamPatch` type (§6.9.2). Pre-build: `NetSpec.update` / `.with_` / `.patch` (immutable; new spec returned). Post-build: `Simulator.parameters` / `Trainer.parameters` of type `ParameterView` (mutating; in-place on the runtime). |
| D27 | Live vs rebuild policy                                            | Three-class taxonomy `LIVE` / `LIVE_RESET` / `REBUILD` (§6.9.5). Live writes propagate in place; rebuild writes raise SPEC-024 with a hint to call `Simulator.rebuild_with(new_ir)`. Connectivity is never silently re-sampled. |
| D28 | Novelty positioning                                               | The load-bearing novelty is **training-paradigm pluralism over a single IR** (BPTT, event-prop, RTRL/forward-mode, eligibility-trace), not the DSL surface (which intentionally inherits from PyNN/NESTML/Brian2/Nengo) and not NIR export (an adopted community standard). See §1.1. Scope ties are broken in favor of preserving spec neutrality across the four training paradigms. |

---

## 19. Cheat sheet — Python ↔ YAML

| Python (B)                                                                                             | YAML (D)                                                                                                          |
|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `spec.population("exc", LIF(tau=20*u.ms), size=8000)`                                                  | `populations: { exc: { model: { kind: LIF, tau: "20 ms" }, size: 8000 } }`                                        |
| `spec.project(exc, inh, rule=conn.FixedProb(prob=0.1, weight=0.1*u.nS), synapse=..., output=...)`      | `{ pre: exc, post: inh, rule: { kind: FixedProb, prob: 0.1, weight: "0.10 nS" }, synapse: ..., output: ... }`     |
| `spec.project(exc, inh, rule=conn.FixedProb(prob=0.1), weight=0.1*u.nS, ...)` *(sugar)*                | `{ pre: exc, post: inh, rule: { kind: FixedProb, prob: 0.1 }, weight: "0.10 nS", synapse: ..., output: ... }`     |
| `spec.input(exc, Poisson(rate=20*u.Hz), weight=0.2*u.nS)`                                              | `{ target: exc, source: { kind: Poisson, rate: "20 Hz" }, weight: "0.2 nS" }`                                     |
| `spec.observe(exc.spikes)`                                                                             | `{ target: exc, quantity: spike }`                                                                                |
| `spec.observe(exc[:50].voltage, every=1*u.ms, reducer="mean")`                                         | `{ target: "exc[:50]", quantity: V, every: "1 ms", reducer: mean }`                                               |
| `rule=conn.FixedProb(prob=0.1, weight=init.LogNormal(mean=0.1*u.nS, std=0.05*u.nS))`                   | `rule: { kind: FixedProb, prob: 0.1, weight: { kind: LogNormal, mean: "0.1 nS", std: "0.05 nS" } }`               |
| `cols = [spec.subnetwork(f"col_{k}", column_spec, N=1000) for k in range(4)]`                          | `subnetworks: { col_0: { !include "column.netspec.yaml", params: { N: 1000 } }, col_1: {...}, ... }`              |
| `all_neurons = sp.merge(exc, inh)` *(merged view)*                                                     | `target: { merge: [exc, inh] }`   *or*   `target: "exc \| inh"`                                                   |
| `view = exc[[0, 1, 5, 42]]`                                                                            | `target: "exc[[0,1,5,42]]"`                                                                                       |
| `view = conv1.reshape(-1)`                                                                             | `target: { population: conv1, reshape: [-1] }`                                                                    |
| `sp.train(20*u.ms, constraint="positive")`                                                             | `{ train: true, value: "20 ms", constraint: positive }`   *or*   `!train "20 ms"`                                 |
| `sp.train(init.XavierNormal(), name="W")`                                                              | `!train { kind: XavierNormal }`   *or*   `{ train: true, init: { kind: XavierNormal }, name: "W" }`               |
| `spec.sequential("enc", [sp.layer.Conv2d(...), sp.layer.MaxPool2d(2), ...])`                           | `sequentials: { enc: { layers: [ { kind: Conv2d, ... }, { kind: MaxPool2d, kernel: 2 }, ... ] } }`                |
| `sp.layer.Linear(out=10, neuron=sp.models.LeakyRateReadout(), weight=sp.train(init.XavierNormal()))`   | `{ kind: Linear, out: 10, neuron: { kind: LeakyRateReadout }, weight: !train { kind: XavierNormal } }`            |
| `sp.spec.viz(ir, mode="layers", renderer="mermaid", out="net.md")`                                     | (CLI) `brainpy viz path.yaml --mode layers --renderer mermaid -o net.md`                                          |
| `sp.backends.nir.export(ir, seed=0, strict=False)`                                                     | (CLI) `brainpy export path.yaml --backend nir -o net.nir`                                                         |
| `spec2 = spec.update("populations.exc.model.tau", 25*u.ms)`                                            | (CLI) `brainpy patch path.yaml --from patches.yaml -o new.yaml`                                                   |
| `spec.patch(ParamPatch("projections[*].rule.weight", 1.5, op="scale"))`                                | `patches: - { path: "projections[*].rule.weight", value: 1.5, op: scale }`                                        |
| `sim.parameters.get("populations.exc.model.tau")`   *(read live)*                                       | n/a (runtime)                                                                                                     |
| `sim.parameters.set("populations.exc.model.tau", 25*u.ms)`   *(write live)*                             | n/a (runtime)                                                                                                     |
| `sim2 = sim.rebuild_with(spec.update("populations.exc.size", 16000))`                                  | n/a (runtime)                                                                                                     |
| `trainer.parameters.get("projections[0].rule.weight")` *(returns ParamState)*                          | n/a (runtime)                                                                                                     |

---

## 20. Open questions

- **NIR extension namespace coordination.** We claim a `nir.brainx.*`
  namespace for our custom extension nodes (`SpikeTimes`, `Concat`, and
  potentially an `ALIF`-adaptation node). Coordinate with the NIR
  maintainers before publishing so the namespace is reserved upstream
  and the schema is documented in NIR's own extension registry.
- **Patch order semantics for overlapping paths.** When two
  `ParamPatch` entries target overlapping wildcards or mix `set` and
  `scale`/`add`, the documented order is "last patch wins for `set`;
  cumulative for `scale`/`add`." Confirm this matches user expectations
  on the first sweep example that exercises overlap.
- **Patch portability across spec edits.** A patch stored in YAML
  references paths by id. If the underlying spec renames a population,
  every stored patch needs to be migrated. Should we ship a
  `brainpy patch migrate <old.yaml> <new.yaml>` helper, or treat patches
  as session-local artifacts that don't survive schema renames?

---

**Previous:** [Chapter 10 — Implementation](./10-implementation.md)  
**Next:** [README](./README.md)
