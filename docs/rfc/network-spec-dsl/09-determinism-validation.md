# Chapter 9 — Determinism contract and validation rules

> Part of the [Network Specification DSL RFC](./README.md).

## 9.1 Determinism contract (G4)

Given a fixed `(NetIR, backend, seed, dt)`:

1. **Connectivity sampling** uses
   `jax.random.fold_in(jax.random.key(seed), proj_index)` per projection.
   A projection's own `seed` overrides this.
2. **Weight / delay distributions** use a derived key:
   `jax.random.fold_in(proj_key, _SUBKEY_WEIGHT)` with stable constants.
3. **Init-state distributions** use `fold_in(pop_key, _SUBKEY_INIT)`.
4. **Input sources** (e.g. Poisson) use `fold_in(input_key, step)`.
5. **Backends must not consume randomness outside the seed tree.**
6. **Visualization** (mode-dependent): `graph` / `layers` / `params` /
   `nir` are deterministic in `(ir, mode, renderer)` alone; `matrix`
   additionally takes `seed` (since it samples a `ConnectionResult`).
7. **Export** is deterministic in `(ir, seed, strict)`: same inputs ⇒
   identical NIR artifact bytes and identical sidecar. The default
   `seed` for export inherits the simulator's default seed
   (`sp.spec.DEFAULT_SEED`, currently `0`); it can be overridden via
   the `seed=` kwarg in Python or the `--seed N` flag on the CLI.
8. **Post-build mutation** (§3.14) is deterministic: applying the same
   `ParamPatch` list to identical `(NetSpec, NetIR)` inputs yields the
   same content hash; applying it to a built `Simulator` / `Trainer`
   yields the same in-memory parameter values.

Acceptance test: for each backend, two builds with identical
`(NetIR, backend, seed, dt)` produce bit-identical artifacts.

---

## 9.2 Validation rules catalog

Every error has a stable code for documentation cross-reference. Codes
are partitioned into spec-level (`SPEC-NNN`), backend-capability
(`SPEC-021`+), and per-export-backend (`EXPORT-<KIND>-NNN`).

### 9.2.1 Spec-level errors

| Code     | Tier        | Rule                                                                 |
|----------|-------------|----------------------------------------------------------------------|
| SPEC-001 | construction| Duplicate id in `populations` / `projections` / `observables`.       |
| SPEC-002 | construction| Reference to unknown population (`pre`, `post`, `target`).           |
| SPEC-003 | construction| Slice / index out of range for the referenced population.            |
| SPEC-004 | construction| `ModelRef.kind` not in registry.                                     |
| SPEC-005 | construction| Required parameter for `kind` is missing.                            |
| SPEC-006 | construction| Parameter has wrong unit dimension.                                  |
| SPEC-007 | construction| Distribution sample dimension does not match parameter dimension.    |
| SPEC-008 | finalize    | Connectivity rule precondition failed (delegated to `braintools.conn`). |
| SPEC-009 | finalize    | Connectivity rule rejected by constructor / `.generate()`.           |
| SPEC-010 | finalize    | `ConnRule.kind` not registered.                                      |
| SPEC-011 | finalize    | Subnetwork export name collides with parent population.              |
| SPEC-012 | backend     | Backend does not support `delay` in `ConnRule.params`.               |
| SPEC-013 | backend     | Backend does not support `plasticity` of this kind.                  |
| SPEC-014 | backend     | Instantaneous (zero-delay) recurrent cycle detected on event backend.|
| SPEC-015 | backend     | Backend rejects neuron / synapse / connectivity kind.                |
| SPEC-016 | construction| `weight` set both as projection sugar and on the rule with conflicting values. |
| SPEC-017 | construction| Conflicting alias and canonical kwarg on the same rule.              |
| SPEC-018 | construction| `Trainable` on a parameter slot annotated `Trainability.NEVER`.      |
| SPEC-019 | construction| Merged view with incompatible member shapes / neuron-model kinds.    |
| SPEC-020 | construction| Sequential layer-shape mismatch.                                     |
| SPEC-021 | backend     | Backend declares no training support but the IR contains `Trainable(required=True)`. |
| SPEC-022 | backend     | Backend rejects a layer macro kind.                                  |
| SPEC-023 | mutation    | `ParamPatch.path` does not resolve to a valid IR leaf (or wildcard matches nothing). |
| SPEC-024 | mutation    | `ParameterView.set(path, ...)` on a `REBUILD`-class leaf (§3.14.5). Raised as `ParameterChangeRequiresRebuild`. Hint to use `Simulator.rebuild_with(new_ir)`. |
| SPEC-025 | mutation    | `ParamPatch.op` not valid for the leaf type (e.g. `scale` on a categorical `kind` field). |

### 9.2.2 NIR export notices (`EXPORT-NIR-NNN`)

| Code            | Class       | Trigger                                                                              |
|-----------------|-------------|---------------------------------------------------------------------------------------|
| EXPORT-NIR-001  | APPROXIMATE | `ALIF` exported as `nir.LIF` + custom adaptation node.                                |
| EXPORT-NIR-002  | UNSUPPORTED | `HH`, `Izhikevich`, or other no-NIR-equivalent neuron model.                          |
| EXPORT-NIR-003  | APPROXIMATE | `MaxPool2d` → `AvgPool2d` in lenient mode.                                            |
| EXPORT-NIR-004  | DROPPED     | Plasticity (STDP / STP / …) stripped — NIR is inference-only.                         |
| EXPORT-NIR-005  | RECORDED    | Sparse rule densified to a large `nir.Linear` matrix (> 10⁷ entries).                |
| EXPORT-NIR-006  | DROPPED     | Weight observable stripped — not deployable.                                           |
| EXPORT-NIR-007  | EXTENSION   | Merged view emitted as custom `nir.brainx.Concat` extension node.                     |
| EXPORT-NIR-008  | RECORDED    | Physical units stripped; original units placed in sidecar.                            |
| EXPORT-NIR-009  | RECORDED    | `Trainable` baked as constant; original `Trainable.name` placed in sidecar.           |
| EXPORT-NIR-010  | RECORDED    | Stochastic input source's parameters placed in sidecar; NIR sees a placeholder Input. |

Strict mode (`--strict`) elevates `APPROXIMATE`, `EXTENSION`, `DROPPED`,
and `UNSUPPORTED` notices to errors.

---


---

**Previous:** [Chapter 8 — CLI and visualization](./08-cli-and-viz.md)  
**Next:** [Chapter 10 — Implementation](./10-implementation.md)
