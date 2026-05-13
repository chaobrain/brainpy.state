# Chapter 7 — Registry (connectivity, initializers, models, layers)

> Part of the [Network Specification DSL RFC](./README.md).

## 7. Registry

Every model and rule is referenced by `kind` string. The registry maps
each `kind` to its Python implementation and a parameter signature
(names, units, defaults, trainability metadata).

### 7.1 Connectivity registry

The canonical source is **`braintools.conn`**. At import time, every
public subclass of `braintools.conn.Connectivity` is registered, keyed by
its PascalCase class name:

```python
import braintools.conn as _bt_conn

for _name in _bt_conn.__all__:
    _cls = getattr(_bt_conn, _name)
    if isinstance(_cls, type) and issubclass(_cls, _bt_conn.Connectivity):
        register_connectivity(_name, source=_cls)
```

Supplementary rules live in `brainpy_state/spec/connect/supplementary.py`
as `braintools.conn.PointConnectivity` subclasses, registered under the
same protocol. The legacy `brainpy_state._network._connectivity` module
is removed (§10.3).

| Supplementary rule              | Status                                        |
|---------------------------------|-----------------------------------------------|
| `FixedIndegree`                 | shipped here; upstream PR target: `braintools`|
| `FixedOutdegree`                | shipped here; upstream PR target: `braintools`|
| `FixedTotalNumber`              | shipped here; upstream PR target: `braintools`|
| `PairwisePoisson`               | shipped here; upstream PR target: `braintools`|
| `SymmetricPairwiseBernoulli`    | shipped here; upstream PR target: `braintools`|

`brainpy_state.spec.connect` re-exports the full registered set.

### 7.2 Initializer registry

Distributions and weight/delay initializers are sourced from
**`braintools.init`** with the same auto-registration mechanism. Every
`braintools.init.Initialization` subclass is keyed by PascalCase class name
(`Normal`, `LogNormal`, `Uniform`, `TruncatedNormal`, `Constant`,
`KaimingNormal`, `XavierNormal`, …). Lower-case aliases are accepted by
the YAML loader and canonicalized.

### 7.3 Neuron / synapse / output / input / plasticity registries

```python
@register_neuron("LIF", source="brainpy_state._brainpy.lif.LIF")
class LIFSignature:
    tau:     Annotated[u.ms, Trainability.OK]
    V_th:    Annotated[u.mV, Trainability.OK]
    V_reset: Annotated[u.mV, Trainability.OK]
    V_rest:  Annotated[u.mV, Trainability.OK]

@register_input("Poisson", source="brainpy_state._brainpy.inputs.PoissonSpike")
class PoissonSignature:
    rate: Annotated[u.Hz, Trainability.OK]   # learnable rate is allowed
```

Trainability annotations:

- `Trainability.OK` — `Trainable[...]` accepted.
- `Trainability.NEVER` — `Trainable[...]` raises SPEC-018.
- `Trainability.BACKEND` — accepted, but specific backends may reject
  (raises SPEC-021 on that backend).

### 7.4 Layer registry (for deep SNNs)

The v1 set is the table in §3.11.5. Each macro declares:

- `in_kind` — accepted view shape (`flat`, `2d`, `3d`).
- `out_kind` — produced view shape.
- Whether it materializes a Population (stateful) or only a Projection.

`spec.sequential(...)` checks `layer[k].out_kind == layer[k+1].in_kind` and
that numeric shapes broadcast; otherwise SPEC-020.

### 7.5 Third-party registration

```toml
[project.entry-points."brainpy_state.spec.neurons"]
my_neuron = "mypkg.models:MyNeuron"

[project.entry-points."brainpy_state.spec.connectivity"]
my_rule = "mypkg.conn:MyRule"

[project.entry-points."brainpy_state.spec.layers"]
my_layer = "mypkg.layers:MyLayer"

[project.entry-points."brainpy_state.spec.inputs"]
my_input = "mypkg.inputs:MyInputSource"

[project.entry-points."brainpy_state.spec.initializers"]
my_init = "mypkg.init:MyInit"
```

---


---

**Previous:** [Chapter 6 — Export backends](./06-export-nir.md)  
**Next:** [Chapter 8 — CLI and visualization](./08-cli-and-viz.md)
