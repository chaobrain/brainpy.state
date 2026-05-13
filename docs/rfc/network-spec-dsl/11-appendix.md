# Chapter 11 — Appendix: decision log, cheat sheet, open questions

> Part of the [Network Specification DSL RFC](./README.md).

## 11.1 Decision log

| ID  | Decision                                                          | Resolution |
|-----|-------------------------------------------------------------------|-----------|
| D1  | `dt` placement                                                    | Backend.build kwarg. Pinning in the spec leaks runtime choice into G1; event backends ignore it. |
| D2  | `seed` placement                                                  | Backend.build kwarg, with per-projection `seed` override allowed in the IR for reproducibility of partial sub-graphs. |
| D3  | Module path                                                       | `brainpy_state.spec`, exported as `brainpy.state.spec`. Coexists with `brainpy.state.Builder`. |
| D4  | CLI name                                                          | `brainpy`. |
| D5  | Cross-ref style                                                   | IR uses strings (population ids, dotted paths); B uses handles; D uses string ids. |
| D6  | Custom user models                                                | Registry decorators + entry-point groups (§7.5). |
| D7  | YAML subnetwork parameterization                                  | `!include` + an explicit `params:` map per instance. Templating (Jinja / Hydra) is opt-in via `brainpy sweep`. |
| D8  | Mutable parameters for training                                   | IR stays frozen. `Trainable`-wrapped leaves materialize as `brainstate.nn.Param` on the synthesized module and are updated by the trainer's optimizer as internal training state. There is no user-facing path-addressed write surface on either the IR or the built `Trainer`. |
| D9  | View granularity                                                  | Slice / index / merge / reshape are fields on `ViewRef`. No separate `View` node type. Merge views denormalize into one `ProjectionNode` per member at finalize. |
| D10 | Connectivity source library                                       | `braintools.conn` is canonical. Auto-registers every `Connectivity` subclass; weight/delay live on the rule. Supplementary rules ship from `brainpy_state` and are tracked for upstreaming. |
| D11 | Initializer source library                                        | `braintools.init` is canonical. Auto-registers every `Initialization` subclass; `DistRef` lowers to a concrete `Initialization` at backend build. |
| D12 | Weight/delay precedence                                           | Canonical home is `ConnRule.params`. Projection-level `weight=` / `delay=` are sugar merged at finalize. Conflicts raise SPEC-016. |
| D13 | Trainable surface                                                 | `Trainable` is a value-level wrapper, applicable to any leaf in `ModelRef.params`, `ConnRule.params`, `PopulationNode.init`, and `InputNode.weight` / `InputNode.source.params`. Trainability metadata lives in registry signatures. |
| D14 | Trainable storage                                                 | Every `Trainable` materializes as `brainstate.nn.Param` on the synthesized `brainstate.nn.Module`. Trainers collect via `state.tree_states(brainstate.ParamState)`. |
| D15 | Layer macros                                                      | Ship the §3.11.5 set covering deep-SNN essentials. Third-party macros via entry points. |
| D16 | Visualization default renderer                                    | Mermaid (no runtime deps) when Graphviz is not installed; Graphviz when available. HTML renderer for interactive `brainpy viz --renderer html`. |
| D17 | NIR as the canonical export target                                | Yes. Other export targets (`onnx-spike`, `nengo`, `lava`) implement the same `ExportBackend` protocol but are not required to ship with the spec library. |
| D18 | Lossy-export policy                                               | Six-class taxonomy (§6.4). Strict mode is opt-in (`--strict`) and elevates classes `APPROXIMATE`, `EXTENSION`, `DROPPED`, `UNSUPPORTED` to errors. Lenient mode is default and ships notices. |
| D19 | NIR units                                                         | NIR is unit-agnostic. The exporter strips to canonical SI and writes a sidecar (`<name>.nir.meta.json`) preserving the original units and trainable / seed / compound metadata. |
| D20 | NIR import                                                        | Not in scope. The sidecar enables partial reconstruction for round-trip testing only; production round-trip is not supported because `UNSUPPORTED` and `DROPPED` losses are unrecoverable. |
| D21 | Loss / optimizer integration                                      | **User-side.** Losses and optimizers are plain Python callables passed to `bptt.build(..., loss=...)`. No `brainpy.state.spec.loss` / `…optim` modules ship with the spec. |
| D22 | NIR export default seed                                           | Defaults to the simulator's default seed (`spec.DEFAULT_SEED`, currently `0`). Overridable in Python via `export(..., seed=N)` and on the CLI via `brainpy export --seed N`. |
| D23 | Reverse-compatibility shim for `_network/_connectivity.py`        | **Removed.** Supplementary rules move to `brainpy_state/spec/connect/supplementary.py`. The legacy import path is dropped. |
| D24 | Spike-time input encoding in NIR                                  | Emitted as a NIR extension node `nir.brainx.SpikeTimes(times=...)` under our reserved `nir.brainx.*` namespace. The sidecar mirrors the table for consumers that ignore extensions. |
| D25 | Quantized weights for hardware                                    | **Per-export-backend concern.** The core spec does not carry a `quantize` flag. Each export backend (e.g. a future Loihi-targeted exporter) may consume `Trainable.constraint` strings (e.g. `"int8"`, `"fixed:Q4.4"`) or apply its own quantization config. |
| D26 | Post-definition parameter modification                            | **Not supported.** The IR is immutable after `.finalize()`; built `Simulator` / `Trainer` artifacts expose no path-addressed parameter writeback. Cross-run variation is declared up front via `net.variable(name, default, ...)` (§3.14) and bound at `backend.build(ir, ..., variables={...})`. To change anything else, edit the source spec and re-`finalize` to produce a new IR with a new content hash. Rationale: post-definition mutation makes reproducibility, content-hash caching, and the determinism contract (G4) fragile in ways that are not worth the convenience. |
| D27 | Variable declaration ergonomics                                   | One declaration surface: `net.variable(name, default, *, constraint, required)` on the Python builder, mirrored by a top-level `variables:` block + `!variable <name>` tag in YAML. A `VariableRef` is a value wrapper alongside `Trainable` / `DistRef` / `Noise` (§3.10). `Trainable[VariableRef]` is rejected; `DistRef[..., VariableRef, ...]` and `Noise[..., VariableRef, ...]` are allowed. |
| D28 | Novelty positioning                                               | The load-bearing novelty is **training-paradigm pluralism over a single IR** (BPTT, event-prop, RTRL/forward-mode, eligibility-trace), not the DSL surface (which intentionally inherits from PyNN/NESTML/Brian2/Nengo) and not NIR export (an adopted community standard). See §1.1.1. Scope ties are broken in favor of preserving spec neutrality across the four training paradigms. |
| D29 | Backend module location                                           | Backends live as **top-level modules under `brainpy.state`**: `brainpy.state.clock`, `brainpy.state.event`, `brainpy.state.bptt`, `brainpy.state.eprop`, `brainpy.state.eventprop`, `brainpy.state.ppprop`, `brainpy.state.nir`, `brainpy.state.onnxspike`. The spec module (`brainpy.state.spec`) does **not** contain any backend implementation — only the IR, frontends, registry, view algebra, and parameter-modification machinery. Backend protocols and discovery (`SimBackend`, `TrainBackend`, `ExportBackend`, `backend.list`, `backend.get`) live at `brainpy.state.backend` (singular, also top-level). Entry-point group names (`brainpy_state.backends.sim/.train/.export`) are unchanged. See [Chapter 5 §5.1.1](./05-backends.md#511-module-location). |

---

## 11.2 Cheat sheet — Python ↔ YAML

| Python (B)                                                                                             | YAML (D)                                                                                                          |
|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `net.population("exc", LIF(tau=20*u.ms), size=8000)`                                                   | `populations: { exc: { model: { kind: LIF, tau: "20 ms" }, size: 8000 } }`                                        |
| `net.project(exc, inh, rule=conn.FixedProb(prob=0.1, weight=0.1*u.nS), synapse=..., output=...)`       | `{ pre: exc, post: inh, rule: { kind: FixedProb, prob: 0.1, weight: "0.10 nS" }, synapse: ..., output: ... }`     |
| `net.project(exc, inh, rule=conn.FixedProb(prob=0.1), weight=0.1*u.nS, ...)` *(sugar)*                 | `{ pre: exc, post: inh, rule: { kind: FixedProb, prob: 0.1 }, weight: "0.10 nS", synapse: ..., output: ... }`     |
| `net.input(exc, Poisson(rate=20*u.Hz), weight=0.2*u.nS)`                                               | `{ target: exc, source: { kind: Poisson, rate: "20 Hz" }, weight: "0.2 nS" }`                                     |
| `net.observe(exc.spikes)`                                                                              | `{ target: exc, quantity: spike }`                                                                                |
| `net.observe(exc[:50].voltage, every=1*u.ms, reducer="mean")`                                          | `{ target: "exc[:50]", quantity: V, every: "1 ms", reducer: mean }`                                               |
| `rule=conn.FixedProb(prob=0.1, weight=init.LogNormal(mean=0.1*u.nS, std=0.05*u.nS))`                   | `rule: { kind: FixedProb, prob: 0.1, weight: { kind: LogNormal, mean: "0.1 nS", std: "0.05 nS" } }`               |
| `cols = [net.subnetwork(f"col_{k}", column_spec, N=1000) for k in range(4)]`                           | `subnetworks: { col_0: { !include "column.netspec.yaml", params: { N: 1000 } }, col_1: {...}, ... }`              |
| `all_neurons = spec.merge(exc, inh)` *(merged view)*                                                   | `target: { merge: [exc, inh] }`   *or*   `target: "exc \| inh"`                                                   |
| `view = exc[[0, 1, 5, 42]]`                                                                            | `target: "exc[[0,1,5,42]]"`                                                                                       |
| `view = conv1.reshape(-1)`                                                                             | `target: { population: conv1, reshape: [-1] }`                                                                    |
| `spec.train(20*u.ms, constraint="positive")`                                                           | `{ train: true, value: "20 ms", constraint: positive }`   *or*   `!train "20 ms"`                                 |
| `spec.train(init.XavierNormal(), name="W")`                                                            | `!train { kind: XavierNormal }`   *or*   `{ train: true, init: { kind: XavierNormal }, name: "W" }`               |
| `net.sequential("enc", [spec.layer.Conv2d(...), spec.layer.MaxPool2d(2), ...])`                        | `sequentials: { enc: { layers: [ { kind: Conv2d, ... }, { kind: MaxPool2d, kernel: 2 }, ... ] } }`                |
| `spec.layer.Linear(out=10, neuron=spec.models.LeakyRateReadout(), weight=spec.train(init.XavierNormal()))` | `{ kind: Linear, out: 10, neuron: { kind: LeakyRateReadout }, weight: !train { kind: XavierNormal } }`        |
| `spec.viz(ir, mode="layers", renderer="mermaid", out="net.md")`                                        | (CLI) `brainpy viz path.yaml --mode layers --renderer mermaid -o net.md`                                          |
| `from brainpy.state import nir; nir.export(ir, seed=0, strict=False)`                                  | (CLI) `brainpy export path.yaml --backend nir -o net.nir`                                                         |
| `tau_exc = net.variable("tau_exc", 20*u.ms)` *(declare)*                                                | `variables: { tau_exc: { default: "20 ms" } }`                                                                    |
| `lif = spec.models.LIF(tau=tau_exc, ...)` *(reference at value site)*                                   | `model: { kind: LIF, tau: !variable tau_exc, ... }`                                                               |
| `clock.build(ir, seed=0, dt=0.1*u.ms, variables={"tau_exc": 25*u.ms})` *(bind at build)*                | (CLI) `brainpy run path.yaml --backend clock --var tau_exc="25 ms"`                                               |

---

## 11.3 Open questions

- **NIR extension namespace coordination.** We claim a `nir.brainx.*`
  namespace for our custom extension nodes (`SpikeTimes`, `Concat`, and
  potentially an `ALIF`-adaptation node). Coordinate with the NIR
  maintainers before publishing so the namespace is reserved upstream
  and the schema is documented in NIR's own extension registry.
- **Variable defaults vs. required flag for sweep ergonomics.** A
  declaration is either `default=<concrete>` (sweep-friendly: omitting
  the variable in a build call just reuses the default) or `required=True`
  (forces every build to supply a value, useful for seeds and
  experiment-defining knobs). Confirm with users running real sweeps
  whether the right ergonomic default is "always-required" or
  "default-with-fallback," and whether we should ship a `brainpy lint`
  rule that flags missing `default=` on numeric variables.
- **Sweep-file expression grammar.** §4.4 sweep files allow
  `"${-0.1 * g} nS"` to compose axis values with units. The grammar is
  intentionally minimal — confirm before the first real sweep whether
  this covers the common cases (linear axes, log axes, paired axes) or
  whether the file should just call into a Python expression.

---

**Previous:** [Chapter 10 — Implementation](./10-implementation.md)  
**Next:** [README](./README.md)
