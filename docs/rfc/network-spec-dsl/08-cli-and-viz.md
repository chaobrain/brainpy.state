# Chapter 8 — CLI tooling and visualization (G10)

> Part of the [Network Specification DSL RFC](./README.md).

## 12. CLI tooling and visualization (G10)

### 12.1 `bp-spec` CLI

```
bp-spec lint     <path.yaml>                 # JSON Schema + IR validation
bp-spec describe <path.yaml>                 # counts + parameter summary (--json available)
bp-spec diff     <a.yaml> <b.yaml>           # structural diff at the IR level
bp-spec viz      <path.yaml> -o net.svg      # see §12.2
bp-spec build    <path.yaml> --backend NAME [--seed N] [--dt T] [--dry-run]
bp-spec run      <path.yaml> --backend clock --duration "1 s" --out runs/<hash>/
bp-spec sweep    <sweep.yaml> --backend clock --out runs/
bp-spec export   <path.yaml> --backend nir [--strict] [--seed N] -o brunel.nir
bp-spec patch    <path.yaml> --from patch.yaml -o new.yaml
bp-spec run      <path.yaml> --patch patch.yaml --backend clock --duration "1 s"
```

`build --dry-run` performs full IR construction + backend capability check
without running. `export` writes both the artifact and the sidecar (§9.4).

### 12.2 Visualization

Visualization reads the IR (not the runtime) so the same view is available
before any backend is built.

```sh
bp-spec viz <path>                                  \
    --mode    {graph,layers,matrix,params,nir}      \
    --renderer {graphviz,mermaid,matplotlib,html}   \
    --collapse-subnetworks                          \
    --color-by {tag,size,trainable,kind}            \
    -o net.svg
```

```python
import brainpy.state.spec as sp
ir = sp.spec.load("brunel.netspec.yaml")
sp.spec.viz(ir, mode="graph", renderer="graphviz", out="brunel.svg")
sp.spec.viz(ir, mode="layers", renderer="matplotlib")
fig = sp.spec.viz(ir, mode="matrix", return_figure=True)
sp.spec.viz(ir, mode="nir", out="brunel.nir.svg")    # post-export graph view
```

**Modes**

| Mode    | What it shows                                                                                                       | Best for                            |
|---------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| `graph` | Populations as nodes, projections as directed edges. Edge thickness ∝ #edges; edge color = sign. Inputs / observables drawn as squares / chevrons. Subnetworks rendered as collapsible clusters. | Sparse biophysical networks.        |
| `layers`| Vertical stack of layer macros (from `CompoundMeta.sequentials`). Each layer shows shape, neuron model, parameter count. Recurrent edges drawn as side loops. | Deep / neuromorphic SNNs.           |
| `matrix`| Block-structured connectivity matrix per projection. Dense layers as dot density. Conv/Pool layers as kernel previews. | Topology check.                     |
| `params`| Bar-chart of trainable vs frozen parameter counts per population / projection. Total parameter count for the whole IR. | Sanity-checking a deep model.       |
| `nir`   | The NIR graph that `sp.backends.nir.export(...)` would produce, with lossy mappings highlighted. | Verifying export shape pre-deployment. |

**Renderers**

| Renderer    | Output formats                  | Dependency                                |
|-------------|---------------------------------|-------------------------------------------|
| `graphviz`  | `.svg`, `.png`, `.pdf`, `.dot`  | `graphviz` (optional dep)                 |
| `mermaid`   | `.md`, `.mmd`                   | none                                       |
| `matplotlib`| `.png`, `.svg`, interactive     | `matplotlib`                              |
| `html`      | self-contained HTML (D3 + pan-zoom) | template ships in package              |

Default renderer is `graphviz` if installed, else `mermaid`. Visualization
is deterministic in `(ir, mode, renderer, seed)` (G4).

---


---

**Previous:** [Chapter 7 — Registry](./07-registry.md)  
**Next:** [Chapter 9 — Determinism and validation](./09-determinism-validation.md)
