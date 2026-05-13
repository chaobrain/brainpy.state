# Chapter 6 — Export backends: Neuromorphic IR (G11)

> Part of the [Network Specification DSL RFC](./README.md).

## 6. Export backends — Neuromorphic IR (G11)

### 6.1 Why an Export backend family

Sim and train backends produce trajectories or trained parameters; an
**export** backend produces an artifact in a foreign IR format suitable
for deployment to a third-party toolchain. The Neuromorphic IR (NIR) is
the lingua franca for spiking-network deployment across Loihi, SpiNNaker,
Nengo, Norse, Rockpool, Lava, Sinabs, and others — a single export path
into NIR transitively reaches all of them.

### 6.2 The export workflow

```python
import brainpy.state.spec as sp
from brainpy.state import nir as bp_nir          # NIR export backend (top-level)
import nir

ir = sp.spec.load("brunel.netspec.yaml")
result = bp_nir.export(ir, seed=0, strict=False)

# result.artifact is a nir.NIRGraph
nir.write("brunel.nir", result.artifact)

# result.notices documents lossy mappings
for n in result.notices:
    print(f"[{n.severity}] {n.code} on {n.node_id}: {n.message}")

# result.sidecar carries metadata that NIR drops (units, seeds, trainable names)
import json
with open("brunel.nir.meta.json", "w") as f:
    json.dump(result.sidecar, f)
```

CLI:

```sh
brainpy export brunel.netspec.yaml --backend nir --strict -o brunel.nir
```

### 6.3 Mapping: `NetIR` → NIR

NIR is a directed graph of typed nodes connected by edges. Each NIR node
type is a concrete dataclass (`nir.LIF`, `nir.Linear`, `nir.Conv2d`, …).
The exporter walks `NetIR` once and emits a `nir.NIRGraph`.

#### 6.3.1 Neuron model mapping

| `brainpy.state` neuron model                | NIR node                   | Notes                                                                                           |
|---------------------------------------------|----------------------------|-------------------------------------------------------------------------------------------------|
| `LIF(tau, V_th, V_rest, V_reset, R=1)`      | `nir.LIF(tau, r, v_leak, v_threshold)`   | `tau` in seconds, voltages in volts. `R` → `r`. `V_reset == V_rest` enforced or recorded.       |
| `IF(V_th, R=1)`                             | `nir.IF(r, v_threshold)`                 | Direct.                                                                                          |
| `LeakyRateReadout(tau, R=1)`                | `nir.LI(tau, r, v_leak)`                 | Rate-coded output, no spike threshold.                                                          |
| `LIF` + `Expon` synapse (CUBA)              | `nir.CubaLIF(tau_mem, tau_syn, r, v_leak, v_threshold, w_in)` | Synapse fused into the post-synaptic neuron. Detected at export time when a single inbound projection has `Expon` + `CUBA`. |
| `ALIF(tau, tau_adapt, ...)`                 | `nir.LIF` + custom adaptation node       | EXPORT-NIR-001: NIR has no canonical adaptive-threshold node; exported as a `nir.LIF` plus a custom-typed companion node. Strict mode raises. |
| `HH(...)`                                   | —                                        | EXPORT-NIR-002: no NIR equivalent. Strict mode raises; lenient mode skips with notice.          |
| `Izhikevich(...)`                           | —                                        | EXPORT-NIR-002 (same as HH).                                                                    |

#### 6.3.2 Connectivity / projection mapping

| `brainpy.state` shape                                     | NIR node                                              | Notes                                                  |
|-----------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------|
| `AllToAll(weight)` + `Identity` synapse + `CUBA`          | `nir.Linear(weight)`                                  | Sample, materialize, strip units.                      |
| Above + bias term on output                                | `nir.Affine(weight, bias)`                            |                                                         |
| `Conv2dKernel(weight, stride, padding, ...)`              | `nir.Conv2d(weight, stride, padding, dilation, groups, bias)` | Direct.                                                |
| `Conv1dKernel(...)`                                       | `nir.Conv1d(...)`                                     | Direct.                                                |
| `Pool2d(kind="avg", ...)`                                 | `nir.AvgPool2d(kernel_size, stride, padding)`         | Direct.                                                |
| `Pool2d(kind="sum", ...)`                                 | `nir.SumPool2d(...)`                                  | Direct.                                                |
| `Pool2d(kind="max", ...)`                                 | —                                                     | EXPORT-NIR-003: no NIR MaxPool. Lenient mode replaces with `AvgPool2d` and emits a warning notice. Strict mode raises. |
| `Reshape(target=-1)`                                      | `nir.Flatten(start_dim, end_dim, input_type)`         | Direct.                                                |
| `Delay` (per-projection or per-edge)                      | `nir.Delay(delay)` inserted as a pass-through node     | Per-edge delays require expansion into one Delay node per delay-value group (NIR delay is per-tensor).|
| Sparse `FixedProb` / `Random` / `FixedIndegree` / etc.    | `nir.Linear(weight=dense_W)` with zeros at non-edges  | Sparse rules densify at export. EXPORT-NIR-005 notice for matrices > 10⁷ entries to warn about size. |
| `STDP`, `STP`, other plasticity                           | —                                                     | EXPORT-NIR-004: NIR is inference-only at present. Stripped with notice in lenient mode; strict mode raises. |

#### 6.3.3 Input and output mapping

| `brainpy.state` node                            | NIR node                                                    | Notes                                                                                            |
|-------------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `InputNode(source=Poisson, rate=R)`             | `nir.Input(input_type=spike)` + sidecar metadata            | NIR does not model stochastic input sources; rate recorded in sidecar.                           |
| `InputNode(source=SpikeTimes, times=...)`       | `nir.brainx.SpikeTimes(times=...)` extension node           | Spike times are deployment-critical (Lava / Norse / Nengo need the table at build time, not only as sidecar). Emitted as a NIR extension node; the sidecar mirrors the table for consumers that ignore extensions. |
| `InputNode(source=LayerImage, shape=S)`         | `nir.Input(input_type=tensor with shape S)`                 | Direct.                                                                                          |
| `InputNode(source=DC, current=I)`               | `nir.Input(input_type=current)` + constant sidecar          |                                                                                                  |
| `ObservableNode(quantity=spike)`                | `nir.Output(output_type=spike)`                             | Connected from the source's output edge.                                                         |
| `ObservableNode(quantity=V)`                    | `nir.Output(output_type=voltage)`                           |                                                                                                  |
| `ObservableNode(quantity=rate)`                 | `nir.Output(output_type=rate)`                              | Used for `LeakyRateReadout` outputs.                                                             |
| `ObservableNode(quantity=weight, every=...)`    | —                                                           | EXPORT-NIR-006: weight tracing is a simulation concern, not deployable. Stripped with notice.    |

#### 6.3.4 Topology mapping

| `brainpy.state` topology         | NIR encoding                                                                                |
|----------------------------------|---------------------------------------------------------------------------------------------|
| `ProjectionNode(pre, post, ...)` | Edge `(pre_node, syn_or_neuron_node)` + `(syn_or_neuron_node, post_node)` as needed.        |
| Merged view `sp.merge(a, b)`     | Concat node: NIR currently has no native concat. The exporter emits a synthesized custom node `nir.brainx.Concat(axis=0)` under our reserved `nir.brainx.*` extension namespace. EXPORT-NIR-007 notice. |
| Recurrent self-projection         | Edge from post-neuron output back to its own input. NIR supports cycles.                    |
| `SubNetworkNode`                  | Inlined into the parent graph; the export preserves `id` namespacing.                       |
| `SequentialMeta` ordering         | The exporter walks layers in declared order; NIR edge list reflects the sequential chain.   |
| `GroupMeta`                       | Recorded in sidecar only; NIR has no group concept.                                         |

#### 6.3.5 Parameter and unit handling

NIR is unit-agnostic. Parameters are floats / tensors. The exporter:

1. **Strips units** by reducing every `u.Quantity` to its mantissa in a
   canonical SI base (`s` for time, `V` for voltage, `A` for current,
   `S` for conductance, `Hz` for rate).
2. **Records the original units** in `sidecar.units[<node_id>.<param>]`
   so a round-trip through the metadata sidecar (§6.4) restores them.
3. **Materializes distributions** by sampling them with the build-time
   seed before stripping units.
4. **Bakes `Trainable` values as constants**: the wrapper is unwrapped
   to its current value; `Trainable.name` is recorded in
   `sidecar.trainables`.

#### 6.3.6 The exporter algorithm (sketch)

```python
def to_nir(ir: NetIR, *, seed: int = 0, strict: bool = False) -> ExportResult:
    notices: list[ExportNotice] = []
    nodes: dict[str, nir.NIRNode] = {}
    edges: list[tuple[str, str]] = []
    sidecar: dict[str, Any] = {"units": {}, "trainables": {}, "seeds": {seed: True}}

    # 1. Inline subnetworks.
    flat = _inline_subnetworks(ir)

    # 2. Emit Population → neuron / readout NIR node.
    for pop in flat.populations:
        nodes[pop.id] = _emit_neuron(pop, sidecar, notices, strict)

    # 3. Emit Projection → Linear / Affine / Conv* / Pool* (+ optional Delay).
    for proj in flat.projections:
        result = _sample_connectivity(proj, seed)             # ConnectionResult
        edge_ids = _emit_projection(proj, result, nodes, edges, sidecar,
                                     notices, strict)

    # 4. Emit Input / Output nodes; wire to the corresponding internal nodes.
    for inp in flat.inputs:
        _emit_input(inp, nodes, edges, sidecar, notices)
    for obs in flat.observables:
        _emit_output(obs, nodes, edges, sidecar, notices)

    # 5. Fuse synapse + post-neuron into CubaLIF where the pattern matches.
    _fuse_cuba_lif(nodes, edges, notices)

    graph = nir.NIRGraph(nodes=nodes, edges=edges)
    return ExportResult(artifact=graph, notices=tuple(notices), sidecar=sidecar,
                         content_hash=_canonical_hash(graph, sidecar))
```

### 6.4 Lossy mappings, strict mode, and the metadata sidecar

NIR is intentionally a *minimum-common-denominator* IR. Some `NetIR`
constructs have no direct NIR equivalent (table in §6.3); the exporter
classifies each into one of:

| Class    | Meaning                                                                                  | Strict mode | Lenient mode                                |
|----------|------------------------------------------------------------------------------------------|-------------|---------------------------------------------|
| `LOSSLESS`| Direct one-to-one mapping. Sidecar restores units, names, seeds.                        | always      | always                                      |
| `RECORDED`| Mapping drops information NIR can't carry (units, names, seeds, stochastic params).     | always      | always (information lives in sidecar)       |
| `APPROXIMATE`| Mapping replaces the construct with the nearest NIR equivalent (MaxPool → AvgPool).   | **raises** EXPORT-NIR-003 | replaces with notice |
| `EXTENSION`| Emits a custom NIR-extension node (merged view → `nir.brainx.Concat`).                 | **raises** EXPORT-NIR-007 if `--no-extensions` | extension node emitted |
| `DROPPED` | Construct stripped (plasticity, weight observables).                                    | **raises** EXPORT-NIR-004 / EXPORT-NIR-006 | stripped with notice                       |
| `UNSUPPORTED`| No mapping exists for the node kind (HH, Izhikevich).                                | **raises** EXPORT-NIR-002 | node skipped, downstream edges rerouted; if rerouting is impossible, raises regardless |

Every transformation in classes `RECORDED`–`UNSUPPORTED` emits an
`ExportNotice` with a stable code (§9.2).

**The sidecar** is a plain-Python dict / JSON document recording:

- `units` — original `saiunit` units for every parameter that NIR strips.
- `trainables` — original `Trainable.name`, `constraint`, `required` flags.
- `seeds` — build-time and per-projection seeds used during connectivity sampling.
- `stochastic_inputs` — Poisson rates, spike-time tables, DC currents.
- `compounds` — the original `CompoundMeta` block (sequentials, groups).
- `tags` — population tags.
- `notices` — the same list returned in `ExportResult.notices`, for archival.

The sidecar is written next to the `.nir` file as `<name>.nir.meta.json`
by default. A loader (`sp.spec.import_.nir.load(nir_path, sidecar_path)`)
is provided to reconstruct as much of the original `NetIR` as is
recoverable — useful for round-trip testing but not for production use,
since `UNSUPPORTED` and `DROPPED` losses are not recoverable.

### 6.5 Other export targets

`nir` is the canonical export backend; the same `ExportBackend` protocol
hosts further targets:

- `onnx-spike` — ONNX with the spiking ops extension.
- `nengo` — direct `nengo.Network` artifact.
- `lava` — Intel Lava graph artifact (alternative to NIR-via-Loihi).

Each declares its own capability matrix and notice codes; they are not
required to ship with the spec library and can live in third-party
packages registered through the `brainpy_state.backends.export` entry
point.

---


---

**Previous:** [Chapter 5 — Backends and round-trip](./05-backends.md)  
**Next:** [Chapter 7 — Registry](./07-registry.md)
