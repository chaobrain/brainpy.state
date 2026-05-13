# Spike-Driven Event Backend for `brainpy.state.spec`

**Status:** Requirements specification
**Owner:** TBD
**Date:** 2026-05-13
**Scope:** Implementation contract for the `event` backend listed in `network-spec-dsl.md` §8.2. Defines the paradigm (spike-driven, not asynchronous), the sub-`dt` timing semantics, the delay-scheduling structure, the neuron-adapter protocol, the equivalence relationship with the `clock` backend, and the differentiability scope.

---

## 1. Problem statement

The parent RFC promises an `event` simulation backend distinct from `clock`, but does not pin down what "event-driven" means. Pure asynchronous event-driven simulation — pop-min from a global priority queue, advance only the firing neuron, root-find the next crossing — is **incompatible with JAX at scale**:

- Priority-queue dispatch is inherently sequential; GPU/TPU parallelism is lost.
- Nonlinear neurons (HH, AdEx, Izhikevich, ALIF) have no closed-form inter-event dynamics; predicting the next crossing requires a nested `while_loop` of numerical root-finding inside an outer event loop.
- `jax.lax.while_loop` does not support reverse-mode autodiff cleanly, blocking BPTT through events.
- Dynamic event counts fight JAX's static-shape discipline.

A pure async backend would therefore have to (a) restrict to LIF/IF, (b) run on CPU only, and (c) forfeit autodiff — leaving it without a clear use case the `clock` backend does not already cover.

The realistic and shippable design is **spike-driven event simulation**:

> Fixed-step, vectorized ODE integration for neuron and synapse state (like `clock`), combined with **exact sub-`dt` spike times** and **event-scheduled spike delivery** through sparse event operators (`brainevent`). The state scaffold is clock-like; the spike layer is event-like.

This is what Brian2's standalone mode, NEST's hybrid scheduler, and Lava all ship in practice. None are fully asynchronous on accelerators.

---

## 2. Goals

| ID  | Goal |
|-----|------|
| E1  | **Exact spike timing.** Spike emission and delivery resolve to sub-`dt` precision; delays are not quantized to integer multiples of `dt`. |
| E2  | **GPU/TPU-friendly.** All state updates remain vectorized `jax.lax.scan` bodies; the only sparse / data-dependent ops are the spike emission and delivery, expressed through `brainevent`. |
| E3  | **Same IR as clock.** The backend consumes the `NetIR` defined in the parent RFC unchanged; no new node kinds, no new IR fields. Scheduling is backend-internal. |
| E4  | **Wide model coverage.** Every neuron model the `clock` backend supports is also supported by `event`. Spike-driven semantics do not restrict the neuron family. |
| E5  | **Numerical relationship with `clock` is documented and testable.** Equivalence holds in well-defined limits (zero delay, integer-`dt` delays); divergence is explained and bounded elsewhere. |
| E6  | **Differentiability through surrogate gradients.** BPTT and surrogate-gradient training work through the spike-driven scaffold; the sub-`dt` crossing function is smooth in the state. |
| E7  | **Event-prop opt-in.** Exact-gradient event-prop is a v2 capability behind a `gradient="exact"` flag, not required for v1. |
| E8  | **Determinism.** Given `(NetIR, seed, dt, max_delay, queue_capacity)`, output is bit-identical across runs and platforms (subject to the parent RFC's float-canonicalization rules). |
| E9  | **Plasticity at spike events.** STDP and other spike-timing-dependent rules update on exact pre/post event times rather than on integer-`dt` boundaries. |
| E10 | **Predictable capability surface.** The backend's `BackendCapabilities` is conservative and precise; unsupported feature combinations raise `BackendCapabilityError` at `build()`, never at `run()`. |

### 2.1 Non-goals

- **Asynchronous "advance until next event" execution.** State is always advanced lockstep per `dt`. Backends that want true async semantics may register separately, are out of scope here.
- **Adaptive step within a `dt`.** Fixed-step integration only in v1; nested adaptive substepping is a future extension.
- **Hardware deployment.** Spike-driven is a simulation strategy, not a hardware contract. NIR export remains the responsibility of `backends.nir` (parent §9).
- **Spike-prediction root-finding for nonlinear neurons.** Threshold crossings are detected *after* the step via interpolation, not predicted ahead of the step via root-finding.

---

## 3. Conceptual model

For each macro-step `[t, t+dt]`:

```
                 ┌───────────────────────────────┐
   state(t)  ───►│ 1. Apply incoming events       │
                 │    scheduled for [t, t+dt]     │  (sparse, brainevent)
                 │    — each carries sub-dt offset│
                 └───────────────────────────────┘
                                │
                                ▼
                 ┌───────────────────────────────┐
                 │ 2. Integrate ODEs over dt      │  (vectorized,
                 │    (same integrator as clock)  │   brainstate)
                 └───────────────────────────────┘
                                │
                                ▼
                 ┌───────────────────────────────┐
                 │ 3. Detect threshold crossings  │
                 │    via linear interpolation of │
                 │    V_n → V_{n+1}; reset V      │
                 └───────────────────────────────┘
                                │
                                ▼
                 ┌───────────────────────────────┐
                 │ 4. Schedule outbound events at │
                 │    t* + delay into the         │
                 │    event store                 │
                 └───────────────────────────────┘
                                │
                                ▼
                          state(t+dt)
```

Step 1 dominates synaptic correctness; step 3 dominates spike timing precision; step 4 owns delay accuracy. Steps 2 and 3 are JAX `vmap`-friendly per population; steps 1 and 4 are sparse / event-shaped and live in `brainevent`.

The user's mental model: **"State evolves on a clock; spikes live on a timeline."**

---

## 4. Backend lifecycle

```python
# brainpy_state/spec/backends/event.py

class SpikeDrivenSimulator:
    ir: NetIR
    seed: int
    dt: u.Quantity
    parameters: ParameterView

    def __init__(self, ir, *, seed, dt, max_delay=None,
                 queue_capacity=None, crossing="linear",
                 gradient="surrogate"):
        ...

    def run(self, duration: u.Quantity) -> TraceBundle: ...
    def reset(self) -> None: ...
    def state(self) -> Mapping[str, Any]: ...
    def rebuild_with(self, new_ir: NetIR) -> "SpikeDrivenSimulator": ...
```

`build()` options (in addition to those required by the parent RFC):

| Option            | Type             | Default                                        | Meaning |
|-------------------|------------------|------------------------------------------------|---------|
| `max_delay`       | `u.Quantity`     | inferred from IR (max of all `ConnRule` delays)| Ring-buffer sizing; events scheduled beyond this fall to the spillover heap. |
| `queue_capacity`  | `int`            | `4 × E_expected_per_step`                      | Per-step sparse event buffer width. SPEC-EV-002 on overflow. |
| `crossing`        | `"linear" \| "exp"` | `"linear"`                                  | Sub-`dt` crossing interpolation. `"exp"` available for models that publish an exponential interpolant (LIF closed form). |
| `gradient`        | `"surrogate" \| "exact"` | `"surrogate"`                          | `"exact"` enables event-prop adjoint (v2; raises NotImplementedError in v1). |
| `spillover`       | `"heap" \| "raise"` | `"heap"`                                    | Behavior for events with delay > `max_delay`. |

### 4.1 Build-time discovery

`build()` performs, in order:

1. **Capability validation.** Walk the IR; for each node, check against §11. Reject early.
2. **Delay analysis.** Compute `max_delay` and per-projection delay distribution. Decide ring-buffer length `K = ceil(max_delay / dt)`.
3. **Queue sizing.** Estimate `E_expected_per_step` from `firing_rate_hint` in `NetIR.meta` (optional) or default to `0.05 × Σ pop.size`. User can override via `queue_capacity`.
4. **Fast-path detection.** Identify projection chains of the form `(linear neuron) ← (linear synapse) ← (linear output)` and route through the fused kernel (§10).
5. **State allocation.** Materialize neuron / synapse `brainstate.Variable`s, event store, and observable buffers.
6. **`vmap` batching.** If `NetIR.meta["batch"]` is set or any population has `batch=B`, vectorize over the batch axis.

---

## 5. Sub-`dt` spike-time semantics

After step 2 integrates `V_n → V_{n+1}`, step 3 detects a spike when `V_n < V_th ≤ V_{n+1}`. The sub-step crossing time is computed by interpolation:

### 5.1 Linear interpolation (default)

```
t* = t_n + dt * (V_th - V_n) / (V_{n+1} - V_n)
```

Differentiable in `V_n` and `V_{n+1}`. First-order accurate. Cheap. The default for all neuron models.

### 5.2 Exponential interpolation (LIF fast-path)

For LIF with `tau_m`, the closed-form trajectory between events is exponential. The crossing time is:

```
t* = t_n + tau_m * log((V_n - V_inf) / (V_th - V_inf))
```

where `V_inf` is the asymptotic value over `[t_n, t_n+dt]`. Used by the fused fast-path (§10) where the analytical propagator is already in scope.

### 5.3 Multiple crossings per step

If a neuron crosses, fires, resets, and crosses again within one `dt` (possible at small `dt` and high drive), v1 records only the first crossing and defers the second to the next step. SPEC-EV-005 notice is emitted at `build()` when the integrator's `dt` is large relative to `tau_m` (heuristic: `dt > 0.5 * min(tau_m)`).

### 5.4 Reset semantics with sub-`dt` timing

After detecting a crossing at `t*`, the neuron's `V` is reset to `V_reset` **and integrated forward from `t*` to `t_n + dt`** with the post-reset dynamics. The default is "instantaneous reset, no further integration in this step" — i.e., `V(t_{n+1}) = V_reset`. This matches `clock` exactly. The full sub-step re-integration is an opt-in `reset_mode="continue"` in v2.

---

## 6. Delay representation and event store

### 6.1 Where delays live

Per the parent RFC §5.1, delays live on `ConnRule.params["delay"]` — per-edge if a distribution or array, per-projection if scalar. No new IR fields are introduced.

At backend build, each `ProjectionNode` produces a `ConnectionResult` with `.delays` as a NumPy / JAX array of shape `(n_edges,)`. The backend converts each delay to a `(integer_slot, fractional_offset)` pair:

```
slot   = floor(delay / dt)
offset = delay - slot * dt          # in [0, dt)
```

### 6.2 Event store

The event store is a **ring buffer** of length `K = ceil(max_delay / dt) + 1` slots. Each slot holds a sparse event tensor — a dense COO triple of `(post_idx, weight, sub_dt_offset)` columns over all spikes scheduled to deliver during that future step.

```
slot[(now + 0) mod K]     ← events delivering in current step
slot[(now + 1) mod K]     ← events delivering one step ahead
...
slot[(now + K-1) mod K]   ← events delivering K-1 steps ahead
```

Each slot has a fixed capacity (`queue_capacity` from §4). The capacity is overcommit-by-default (`4 × E_expected_per_step`); overflow raises `SPEC-EV-002` with a hint to raise `queue_capacity` or increase `dt`.

### 6.3 Spillover heap

Delays exceeding `K * dt` fall through to a small heap (fixed-size, `O(log H)` insertion). On every step the heap is peeked; events whose `deliver_time ≤ t + dt` are popped into the current slot. In practice the heap is empty for nearly all SNN workloads; it exists so the backend does not silently truncate delays.

Behavior when `spillover="raise"`: any event scheduled beyond `K * dt` raises `SPEC-EV-003` at scheduling time — useful for hardware-mapping workflows where unbounded delays are forbidden.

### 6.4 Per-step delivery

The current slot is popped at the start of step 1 (Section 3). The contained events are applied to post-synaptic state with their `sub_dt_offset` taken into account:

- For **CUBA / instantaneous synapses**: `I_syn(t* + delay)` adds `weight` to the post-synaptic input integrator with no further timing care (the receiver is integrating over `[t, t+dt]`; the offset only affects the membrane integral, computed analytically for the linear fast-path or approximated as midpoint for non-fast-path).
- For **COBA / conductance synapses**: same, with conductance gating applied.
- For **Expon / Alpha / DualExpon synapses**: the synapse state has its own `tau_syn`; the incoming event is added to the synapse state at `t* + delay`, and the synapse is then integrated from that sub-`dt` time to `t + dt` analytically (closed form available for these).

### 6.5 Plasticity hooks

When a projection has a `plasticity` `ModelRef`, the event store records both pre- and post-spike events (the latter at step 3) tagged with the projection id. At end-of-step, the plasticity rule consumes the matched event pairs and updates per-edge weights. STDP, R-STDP, and other timing rules see exact sub-`dt` timestamps.

Cross-projection eligibility traces (parent C8 concern) are out of scope for v1; flagged as Open Question OQ-3.

---

## 7. Neuron-model adapter protocol

A neuron model qualifies for the `event` backend if it satisfies:

```python
class EventCompatibleNeuron(Protocol):
    # Identical to the clock-side adapter for state integration.
    def init_state(self, batch_shape) -> Mapping[str, Variable]: ...
    def update(self, x, V_prev) -> Mapping[str, Variable]: ...

    # Required for spike-driven semantics.
    threshold: Callable[[State], jnp.ndarray]      # V_th expression
    reset:     Callable[[State, jnp.ndarray], State]  # post-spike reset
    V_var:     str                                 # name of the membrane state variable
```

The `threshold` and `reset` slots are already present on every `Neuron` subclass in `brainpy_state` — the spike-driven backend simply requires them to be Python-level descriptors (so it can construct the crossing-time interpolant), not opaque methods.

| Neuron family    | Support | Notes |
|------------------|---------|-------|
| `LIF`, `IF`      | full + fast-path | Linear; analytical propagator + exponential crossing. |
| `ALIF`           | full | Linear state; threshold adapts; sub-`dt` crossing uses linear interpolation on V. |
| `ExpIF`, `AdExIF`| full | Non-linear `V` but threshold detection is still cheap; linear interpolation is first-order accurate, which is fine when `dt << tau_m`. |
| `Izhikevich`     | full | Same. |
| `AdEx`           | full | Same. |
| `HH`, `MorrisLecar`, `WangBuzsakiHH` | full but with caveat | Spike "threshold" is convention-dependent (V > some value); linear interpolation around a steep upstroke is approximate. SPEC-EV-004 notice; users may pin a `V_th` consistent with their analysis. |
| `LeakyRateReadout` | full | No threshold; passes state through; behaves identically to clock. |

The capability matrix (§11) lists the registered kinds explicitly.

---

## 8. Synapse and output evaluation

Synapse and output kinds are evaluated **analytically between events when possible**:

| Synapse        | Analytical inter-event propagator | v1 path |
|----------------|-----------------------------------|---------|
| `Expon(tau)`   | `s(t) = s(t0) * exp(-(t - t0)/tau)`| fast |
| `Alpha(tau)`   | closed form (two-state)            | fast |
| `DualExpon(tau_r, tau_d)` | closed form (two-state) | fast |
| `AMPA`, `GABAa`, `BioNMDA` | kinetic; clock-step integration | standard |

Output kinds (`CUBA`, `COBA`, `MgBlock`) are pure functions of state and incoming events; they do not introduce additional state.

**Within-step event application order.** Events arriving in the same slot are applied in `(deliver_time, projection_index, pre_idx)` lex order. This ordering is part of the determinism contract (§13).

---

## 9. Linear fast-path (CubaLIF-style fusion)

The most common pattern in deep SNNs is:

```
LIF / IF neuron  ←  Expon / Alpha synapse  ←  CUBA / COBA output  ←  AllToAll / FixedProb projection
```

For this chain the entire `(synapse, output, neuron)` triple has a closed-form propagator over `[t, t+dt]` parameterized by:

- Initial state `(V_n, s_n)`,
- The bag of incoming events for the step `{(weight_i, sub_dt_offset_i)}`,
- Step size `dt`.

The fused kernel computes `(V_{n+1}, s_{n+1}, spike?, t*)` in one fused JAX op without per-event integration. Same fusion idea as NIR's `CubaLIF` node (parent §9.3.1) and as Norse / snnTorch's `LIFParams` step.

**Detection at build.** Walk each projection; if `synapse.kind ∈ {Expon, Alpha, DualExpon}` and `output.kind ∈ {CUBA, COBA}` and `neuron.kind ∈ {LIF, IF, ALIF}`, route to the fast-path. Otherwise, the standard path (§3 steps 1–4) runs.

**Determinism.** The fast-path and the standard path are required to be numerically equivalent up to `1e-6` relative tolerance for any IR they both support — verified by `tests/event_fastpath_equivalence_test.py`.

---

## 10. Backend capabilities (`BackendCapabilities`)

```python
BackendCapabilities(
    supports_delay=True,                       # native sub-dt
    supports_plasticity=True,                  # event-time plasticity rules
    supports_distributions=True,
    supports_nested_subnetworks=True,
    supports_training=True,                    # via surrogate gradient
    supports_batch=True,
    supported_neuron_kinds=frozenset({
        "LIF", "IF", "ALIF", "ExpIF", "AdExIF",
        "Izhikevich", "AdEx",
        "HH", "MorrisLecar", "WangBuzsakiHH",
        "LeakyRateReadout",
    }),
    supported_synapse_kinds=frozenset({
        "Expon", "Alpha", "DualExpon",
        "AMPA", "GABAa", "BioNMDA",
        "Identity",
    }),
    supported_output_kinds=frozenset({
        "CUBA", "COBA", "MgBlock",
    }),
    supported_rules=frozenset({
        "AllToAll", "OneToOne", "FixedProb", "Random",
        "FixedIndegree", "FixedOutdegree", "FixedTotalNumber",
        "PairwisePoisson", "SymmetricPairwiseBernoulli",
        "Gaussian", "Exponential", "DistanceDependent",
        "Ring", "SmallWorld", "ScaleFree",
        "ExcitatoryInhibitory",
        "Conv1dKernel", "Conv2dKernel",
    }),
    supported_layer_macros=frozenset({
        "Linear", "Conv2d", "Conv1d", "LeakyRateReadout",
    }),
    supported_input_kinds=frozenset({
        "Poisson", "SpikeTimes", "DC", "Step", "AC",
        "LayerImage", "DataStream",
    }),
)
```

Explicit non-support (raises `BackendCapabilityError` at `build()`):

- Stateless pooling / reshape layers (`MaxPool2d`, `AvgPool2d`, `Flatten`, `Dropout`, `BatchNorm`) — these are not event-shaped operations. They can be expressed by composing the event backend with a clock-driven scaffold for the stateless part; v2.
- Zero-delay recurrent cycles (already SPEC-014 in parent RFC; surfaces here at build time, not run time).

---

## 11. Numerical equivalence with `clock`

The `event` and `clock` backends are **not** required to produce bit-identical trajectories — they have genuinely different spike-timing semantics. Equivalence holds in well-defined limits:

| Condition | Equivalence |
|---|---|
| All delays are integer multiples of `dt` AND `crossing="linear"` AND `dt → 0` | spike trains converge in trajectory norm. |
| All delays are integer multiples of `dt` AND `crossing="linear"` at finite `dt` | spike *counts* per neuron agree exactly; spike *times* agree up to `dt` quantization (clock loses, event keeps). |
| Sub-`dt` delays present | event-driven trajectories diverge from clock; this is the point of the backend. |

**Acceptance test.** Brunel network with integer-`dt` delays at `dt = 0.1 ms`: per-neuron spike counts over 1 s agree exactly across `clock` and `event` (no statistical tolerance); spike-time distributions agree within 2σ over 10 trials.

---

## 12. Determinism contract

In addition to the parent RFC §13:

1. **Within-slot event ordering** is `(deliver_time, projection_index, pre_idx)` lex.
2. **Sub-`dt` offsets are stored as float32 by default** (sufficient for `dt ≥ 1 µs`); a `precision="float64"` build flag selects double for stricter equivalence.
3. **Spillover heap insertion order** does not affect outcomes: the heap is keyed on `(deliver_time, projection_index, pre_idx)` and ties broken by the same lex.
4. **Ring buffer is zeroed on `reset()`**; spillover heap is cleared.
5. **JAX key fold-in.** Events generated by stochastic inputs (Poisson) use `jax.random.fold_in(input_key, step_index)`; this matches `clock` exactly when `dt` is identical.

Acceptance: two runs of `(ir, seed, dt, max_delay, queue_capacity, crossing, precision)` produce bit-identical `TraceBundle` outputs.

---

## 13. Differentiability

### 13.1 Surrogate-gradient BPTT (v1, default)

The state-integration scan body is identical to `clock`'s; spike emission uses a configurable surrogate gradient (`braintools.surrogate.*`) applied to `(V_{n+1} - V_th) / V_scale`. Sub-`dt` crossing time `t*` is a smooth function of `V_n` and `V_{n+1}`, so it differentiates naturally.

Sparse event delivery uses `brainevent`'s registered VJP rules. Gradients flow:

```
loss → trace → spike(post)  → V(post)  → event-delivery  → weight + V(pre) → spike(pre) → ...
```

`bptt` backend's `Trainable` materialization protocol (parent §6.6) is reused unchanged. The `event` backend simply provides a different forward scan body and a different event-delivery primitive.

### 13.2 Event-prop / exact gradient (v2, opt-in)

When `gradient="exact"`, the backend records the adjoint state at each spike event and propagates gradients in reverse along the event timeline rather than through the state scan. Requires:

- A custom VJP for the sub-`dt` crossing detector (closed-form derivative is known).
- A custom VJP for event delivery that handles backward event propagation.
- A persistent event log for the backward pass.

Out of scope for v1. RAISE `NotImplementedError("event-prop gradient is v2")` at `build()` when `gradient="exact"`.

### 13.3 No gradient (`gradient="none"`)

For pure simulation workloads, skip surrogate-gradient bookkeeping. Cheaper memory profile. Equivalent to `clock` simulation use case.

---

## 14. Mapping to `brainevent` and `brainstate` primitives

| Backend operation        | Implementation primitive |
|--------------------------|--------------------------|
| State integration         | `brainstate.transform.scan` over the macro-step body, identical to `clock`. |
| Sparse event delivery     | `brainevent.csr_event_mv` / `brainevent.coo_event_mv` (existing). |
| Event store ring buffer   | Three `brainstate.Variable`s shaped `(K, queue_capacity, ...)`: `post_idx`, `weight`, `offset`. Rotated by an integer `now` cursor. |
| Spillover heap            | Fixed-size `(H, 4)` array `(deliver_time, proj_idx, pre_idx, weight)` with a `heap_size` cursor; `jax.lax.dynamic_update_slice`-based sift-up/sift-down. |
| Threshold crossing        | Vectorized comparison + linear interpolation; no scan needed within step. |
| Plasticity update         | Pre/post event arrays scanned with the registered plasticity rule; weights updated via `brainevent` scatter. |
| Recording                 | Observable bundles populated from per-step spike-time arrays (sub-`dt` accurate) and downsampled per `Observable.every`. |

The existing `_network` runtime substrate (parent §17) is **not** reused. `backends.event` builds directly on `brainstate` / `brainevent` and shares the registry, the IR, and the `ParameterView` but not the imperative `Builder`.

---

## 15. Validation and testing

- **Unit:** ring-buffer arithmetic, sub-`dt` interpolation arithmetic, spillover-heap correctness, event-ordering tie-breakers.
- **Fast-path equivalence:** for each `(linear neuron, linear synapse, linear output)` triple, fast-path and standard-path agree to `1e-6` rtol over 1 s of simulation.
- **Clock equivalence:** Brunel at `dt = 0.1 ms` with integer-`dt` delays — spike counts exact, spike times agree to `dt`.
- **Sub-`dt` divergence sanity:** Brunel with sub-`dt` delays — spike trains differ from `clock`; agreement converges as `dt → 0`.
- **Determinism:** two runs of the same configuration produce bit-identical traces; cross-platform (CPU / GPU) traces match within `1e-6` rtol.
- **Surrogate-gradient training:** spiking MLP on MNIST under `bptt`-over-`event` matches `bptt`-over-`clock` test accuracy to ±0.5%.
- **Capability mismatch:** every entry in §10's non-support list raises `BackendCapabilityError` at `build()`.
- **Plasticity:** STDP pair rule with exact pre/post times yields the analytical weight change for hand-computed pre/post pairs.
- **Heap overflow:** events scheduled beyond `max_delay` with `spillover="heap"` deliver correctly; with `spillover="raise"` raise SPEC-EV-003.
- **Queue overflow:** synthetic high-rate scenario triggers SPEC-EV-002 with a clear remediation hint.

Tests live colocated as `brainpy_state/spec/backends/event_*_test.py`.

---

## 16. Error codes

Spike-driven-specific codes (in addition to parent SPEC-NNN):

| Code         | Tier     | Trigger |
|--------------|----------|---------|
| SPEC-EV-001  | build    | Neuron model lacks a `threshold` or `reset` descriptor required for sub-`dt` detection. |
| SPEC-EV-002  | run      | Per-slot `queue_capacity` exceeded during a step. Hint: raise `queue_capacity` or coarsen `dt`. |
| SPEC-EV-003  | run      | Event scheduled beyond `max_delay` with `spillover="raise"`. |
| SPEC-EV-004  | build    | HH-family neuron used with default `crossing="linear"` and no explicit `V_th`; first-order interpolation around the action-potential upstroke is approximate. Notice, not error. |
| SPEC-EV-005  | build    | `dt > 0.5 × min(tau_m)`; multiple-crossing-per-step risk. Notice. |
| SPEC-EV-006  | build    | `gradient="exact"` requested in v1. Raises `NotImplementedError`. |
| SPEC-EV-007  | build    | Zero-delay recurrent cycle detected (mirrors parent SPEC-014; surfaced at build instead of run). |

---

## 17. Mapping to the parent RFC

| Parent RFC reference | This RFC's contribution |
|---|---|
| §8 Backend protocol | This RFC is the concrete contract for `SimBackend(name="event")`. |
| §8.2 Capabilities (event) | Filled in by §10 here. |
| §13 Determinism contract | Extended by §12 here. |
| §5.1 Delay representation | Consumed; this RFC adds `ConnectionResult.delays` decomposition into `(slot, offset)`. |
| §6.9 ParameterView | Reused as-is; live/rebuild classification unchanged. Per-edge weight updates land in `LIVE`. |
| §14 Error codes | Extended with SPEC-EV-NNN range. |
| §9 NIR export | Independent. Export still walks the IR; the event backend's internal scheduling is irrelevant to NIR. |

No new IR fields are added. The parent RFC's `NetIR` is sufficient.

---

## 18. Decision log

| ID  | Decision | Resolution |
|-----|----------|-----------|
| EV1 | Async event-driven vs spike-driven | **Spike-driven.** Async fights JAX, restricts to LIF/IF, blocks autodiff, has no use case `clock` does not cover. |
| EV2 | Backend name | Keep parent's `event`. The paradigm is "spike-driven event simulation"; the public name stays short. |
| EV3 | Where delays live | `ConnRule.params["delay"]` in the IR (parent §5.1, unchanged). Backend decomposes at build. |
| EV4 | Event store data structure | Ring buffer keyed by integer slot, with a fixed-size spillover heap for long-tail delays. Both expressed in `brainstate.Variable`s for JIT-friendliness. |
| EV5 | Sub-`dt` crossing method | Linear interpolation default; exponential available for the LIF fast-path. Higher-order is v2. |
| EV6 | Multiple crossings per step | First crossing only in v1; emit SPEC-EV-005 when `dt` is large relative to `tau_m`. |
| EV7 | Reset after sub-`dt` crossing | `V(t_{n+1}) = V_reset` (matches `clock`). Continue-from-`t*` integration is v2. |
| EV8 | Fast-path detection | Build-time pattern match on `(linear neuron, linear synapse, linear output)`. Falls back to standard path otherwise. |
| EV9 | Differentiability | Surrogate-gradient BPTT in v1; event-prop exact gradient is v2. |
| EV10 | Layer-macro support | Linear and Conv only; pooling / reshape / dropout / batchnorm are not event-shaped and are explicitly rejected at build. |
| EV11 | Determinism precision | float32 sub-`dt` offsets by default; float64 via `precision="float64"` build flag. |
| EV12 | Event ordering within a slot | `(deliver_time, projection_index, pre_idx)` lex. |
| EV13 | Spillover policy | Heap by default; `spillover="raise"` available for hardware-mapping workflows that forbid unbounded delays. |
| EV14 | HH-family threshold detection | Supported but first-order; SPEC-EV-004 notice. |
| EV15 | Reuse of `_network.Builder` substrate | **Not reused.** `backends.event` builds directly on `brainstate` / `brainevent`. `clock` keeps `Builder`. |
| EV16 | Adaptive substep within `dt` | Out of scope for v1. The integrator is fixed-step. |
| EV17 | Cross-projection eligibility traces | Out of scope for v1. Open Question OQ-3. |
| EV18 | Capability surface | Conservative: every supported `kind` is listed explicitly in §10; unsupported combinations raise at `build()`, never at `run()`. |

---

## 19. Open questions

- **OQ-1: Default `queue_capacity` heuristic.** Should the default scale with mean firing-rate hint, with worst-case `Σ pop.size`, or via a measured "warm-up" pass that records actual peak per-step event counts? Current default `4 × E_expected_per_step` with `E_expected_per_step = 0.05 × Σ pop.size` is a guess from cortical microcircuit statistics.
- **OQ-2: Heap representation.** A `(H, 4)` sift-up/sift-down array works but is sequential. Would a sorted segmented array updated via segment scans be faster on GPU? Benchmark needed.
- **OQ-3: Cross-projection eligibility traces.** Some plasticity rules (R-STDP with a shared dopamine signal across all projections) need cross-projection coupling. Where does the shared signal live in the IR? Possible addition: a `NetIR.signals` table indexed by name, referenced from `ModelRef` plasticity params. Defer to a follow-up RFC.
- **OQ-4: Mixed-precision recording.** Sub-`dt` offsets at float32 risk drift across multi-hour simulations. Should observables auto-promote to float64 when `duration > some_threshold`?
- **OQ-5: Streaming event log for event-prop.** v2 will need to persist the event log across the forward pass to replay in the backward pass. Disk-backed vs in-memory bounded-buffer — defer to the event-prop RFC.
- **OQ-6: Interaction with `Trainable` over delays.** Per-edge delays as `Trainable[init.LogNormal(...)]` interact with the integer-slot/fractional-offset decomposition: a gradient on `delay` flows through the slot index (discrete) and the offset (continuous). Standard re-parametrization (delay = slot * dt + offset, with slot frozen during backward) likely works but needs verification.
