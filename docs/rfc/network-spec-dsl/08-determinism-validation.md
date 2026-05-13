# Chapter 8 — Determinism contract and validation rules

> Part of the [Network Specification DSL RFC](./README.md).

## 8.1 Determinism contract (G4)

Given a fixed `(NetIR, variables, backend, seed, dt)`:

1. **Connectivity sampling** uses
   `jax.random.fold_in(jax.random.key(seed), proj_index)` per projection.
   A projection's own `seed` overrides this.
2. **Weight / delay distributions** use a derived key:
   `jax.random.fold_in(proj_key, _SUBKEY_WEIGHT)` with stable constants.
3. **Init-state distributions** use `fold_in(pop_key, _SUBKEY_INIT)`.
4. **Input sources** (e.g. Poisson) use `fold_in(input_key, step)`.
5. **Backends must not consume randomness outside the seed tree.**
6. **Visualization** (mode-dependent): `graph` / `layers` / `params` are
   deterministic in `(ir, mode, renderer)` alone; `matrix` additionally
   takes `seed` (since it samples a `ConnectionResult`).
7. **Variable binding** (§3.14) is deterministic: building twice with
   the same `variables=` mapping yields bit-identical artifacts. The
   IR's `content_hash` covers the declared variables and their defaults
   but is independent of any particular binding; two distinct bindings
   against the same IR therefore share a content hash, which lets
   sweeps reuse upstream structure-keyed caches.

Acceptance test: for each backend, two builds with identical
`(NetIR, variables, backend, seed, dt)` produce bit-identical artifacts.

---

## 8.2 Validation rules catalog

Every error has a stable code for documentation cross-reference. Codes
are partitioned into spec-level (`SPEC-NNN`) and backend-capability
(`SPEC-021`+).

### 8.2.1 Spec-level errors

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
| SPEC-023 | build       | Required variable not supplied. A leaf was declared `net.variable(name, ..., required=True)` (or `default=None`) but `backend.build(ir, ..., variables={...})` did not bind it. |
| SPEC-024 | build       | Supplied variable value has the wrong unit dimension for the declared default (or wrong shape for a `DistRef` default). |
| SPEC-025 | build       | Supplied variable value violates the declared `constraint` (e.g. `"positive"`, `"unit_norm"`, `"clip:lo,hi"`). |
| SPEC-026 | build       | Unknown key in `variables=` — no matching `net.variable(...)` declaration in the IR. |

---


---

**Previous:** [Chapter 7 — CLI and visualization](./07-cli-and-viz.md)  
**Next:** [Chapter 9 — Implementation](./09-implementation.md)
