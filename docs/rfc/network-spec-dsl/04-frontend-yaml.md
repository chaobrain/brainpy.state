# Chapter 4 — Frontend B: YAML/JSON data DSL

> Part of the [Network Specification DSL RFC](./README.md).

## 4. Frontend B — YAML/JSON data DSL

Spec is data. A YAML/JSON file is the canonical archival form; Python loads
it with `spec.load(path) -> NetIR`. Same IR, same backends.

### 4.1 Top-level schema (informal)

```yaml
version: "netir/1.0"
name: "<spec-name>"

defaults:
  lif: &lif
    kind: LIF
    tau:     "20 ms"
    V_th:   "-50 mV"
    V_reset:"-60 mV"
    V_rest: "-65 mV"

populations:
  exc:
    model: *lif
    size: 8000
    init:
      V: { kind: Uniform, low: "-65 mV", high: "-55 mV" }
    tags: [excitatory]

projections:
  - { pre: exc, post: inh,
      rule: { kind: FixedProb, prob: 0.1, allow_self_connections: false,
              weight: "0.10 nS" },
      synapse: { kind: Expon, tau: "5 ms" },
      output:  { kind: COBA, E: "0 mV" } }

inputs:
  - { target: exc, source: { kind: Poisson, rate: "20 Hz" }, weight: "0.2 nS" }

observables:
  - { target: exc, quantity: spike }
  - { target: "exc[:50]", quantity: V }
  - { target: exc, quantity: V, every: "1 ms", reducer: mean }

subnetworks:
  column:
    !include "column.netspec.yaml"
    params: { N: 1000 }

sequentials:
  encoder:
    layers:
      - { kind: LayerImage, shape: [1, 28, 28] }
      - { kind: Conv2d, out_channels: 16, kernel: 3,
          neuron: *lif, weight: !train { kind: KaimingNormal } }
      # …

groups:
  recurrent_core:
    members: [exc, inh]
    tags: [balanced_eI]

meta:
  author: "Chaoming Wang"
  citation: "Brunel 2000"
```

### 4.2 Lexical conventions

- **Unit strings.** A quantity is `"<number><whitespace><unit>"`, where
  `<unit>` is anything `saiunit` parses (`mV`, `ms`, `nS`, `Hz`, `pA*ms`,
  …). Whitespace required. Negative numbers allowed (`"-50 mV"`).
- **References.** Bare strings name populations (`"exc"`). Bracketed strings
  name views: `"exc[:50]"`, `"exc[100:200]"`, `"exc[[0,1,5]]"`.
- **Merged views.** Object form `{ merge: [exc, inh] }` or string sugar
  `"exc | inh"`.
- **Connectivity rules.** A mapping with a PascalCase `kind` naming a
  `braintools.conn` class (`FixedProb`, `OneToOne`, `AllToAll`, `Random`,
  `Gaussian`, `Exponential`, `Ring`, `SmallWorld`, `ScaleFree`,
  `DistanceDependent`, `ExcitatoryInhibitory`, …) or a registered
  supplementary rule (`FixedIndegree`, `FixedOutdegree`, `FixedTotalNumber`,
  `PairwisePoisson`, `SymmetricPairwiseBernoulli`).
- **Distributions / initializers.** A mapping with a `kind` naming a
  `braintools.init` class (`Normal`, `LogNormal`, `Uniform`,
  `TruncatedNormal`, `Constant`, `KaimingNormal`, `XavierNormal`, …).
  Lower-case aliases accepted; canonicalized in the IR.
- **Trainables.** Object form `{ train: true, value: ..., constraint: ..., name: ... }`
  or shorthand tag `!train <value>`.
- **Includes.** `!include "<relative path>"` inlines another YAML mapping.
  Cycles are detected and rejected.
- **Anchors and aliases.** Standard YAML `&` / `*` is supported; resolved
  before schema validation.

### 4.3 JSON Schema

A full schema lives at `brainpy_state/spec/schema/netir-1.0.json`. Sketch:

```json
{
  "$id": "https://brainx.chaobrain.com/schema/netir-1.0.json",
  "type": "object",
  "required": ["version", "name", "populations"],
  "properties": {
    "version": { "const": "netir/1.0" },
    "name":    { "type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_-]*$" },
    "populations": {
      "type": "object",
      "additionalProperties": { "$ref": "#/$defs/Population" }
    },
    "projections":  { "type": "array", "items": { "$ref": "#/$defs/Projection" } },
    "inputs":       { "type": "array", "items": { "$ref": "#/$defs/Input" } },
    "observables":  { "type": "array", "items": { "$ref": "#/$defs/Observable" } },
    "sequentials":  { "type": "object" },
    "groups":       { "type": "object" }
  },
  "$defs": {
    "Quantity":     { "type": "string",
                      "pattern": "^-?\\d+(\\.\\d+)?([eE][+-]?\\d+)?\\s+[A-Za-z*/0-9]+$" },
    "Distribution": { "type": "object", "required": ["kind"] },
    "ViewRef":      { "oneOf": [
                        { "type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*(\\[[^\\]]+\\])?$" },
                        { "type": "object" }
                      ] },
    "Trainable":    { "type": "object", "required": ["train"] },
    "ModelRef":     { "type": "object", "required": ["kind"] }
  }
}
```

The schema is used by `brainpy lint`, IDE integrations (YAML Language
Server via `yaml.schemas`), and the loader's pre-validation pass.

### 4.4 Parameter sweeps

Sweeps bind **declared variables** (§3.14). The spec must declare
every value the sweep varies under a `variables:` block; sites that
should use a variable reference it with `!variable <name>`. The IR
itself is loaded once and reused across the sweep.

Two supported patterns:

1. **Python binding** — keep the YAML, supply `variables=` at build:

   ```python
   import brainpy.state.spec as spec
   from brainpy.state import clock        # backend lives at brainpy.state.clock
   ir = spec.load("brunel.netspec.yaml")
   for g in [4.0, 4.5, 5.0]:
       sim = clock.build(ir, seed=0, dt=0.1*u.ms,
                         variables={"W_inh": -0.10*g*u.nS})
   ```

2. **Sweep file** — a side file listing the axes and the variable
   bindings to compute per cell; the CLI expands the cartesian product:

   ```yaml
   # brunel.sweep.yaml
   base: brunel.netspec.yaml
   axes:
     g:    [4.0, 4.5, 5.0]
     seed: [0, 1, 2]
   variables:
     W_inh:  "${-0.1 * g} nS"
     W_seed: "${seed}"
   ```

   ```sh
   brainpy sweep brunel.sweep.yaml --backend clock --out runs/
   ```

Only values declared as variables in the spec can be swept. To sweep
a value that wasn't declared, edit the source spec to add a
declaration and re-`finalize`. The sweep-file expression language is
intentionally minimal (axis interpolation + unit suffix); anything
more complex stays in Python.

---


---

**Previous:** [Chapter 3 — Frontend A: Python NetSpec builder](./03-frontend-python.md)  
**Next:** [Chapter 5 — Backend protocol and round-trip](./05-backends.md)
