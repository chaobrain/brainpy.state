# Spec — 08 · `stdp_dopamine_synapse` + `volume_transmitter` (broadcast-modulator seam)

> Frozen spec. Companion: `2026-06-13-nest-goal-08-dopamine-vt-plan.md`. Authoritative NEST
> sources: `models/stdp_dopamine_synapse.{h,cpp}`, `models/volume_transmitter.{h,cpp}`.
> All formulas below are quoted/derived **from the C++**, not the legacy port.

---

## 0. What this cluster adds (and does NOT add)

- **Capability:** a *global broadcast modulatory signal* `n(t)` (dopamine
  concentration) that **every edge of a projection reads each step** — a superset
  of 07's per-edge post gather. Per-edge eligibility trace `c` (05 machinery) +
  STDP pairing (04 machinery) supply the rest.
- **Not a 4th primitive.** `VoltageCoupledPlasticProj` (#2, the *reader* primitive)
  is generalised: it already reads per-edge post-State (`post_state_reads`); it now
  *also* reads a broadcast scalar (`signal_reads`). Both default-empty → primitives
  #1 and existing #2 users (clopath) untouched.
- **Three new files** (`volume_transmitter` rebuild, dopamine spec, drive) + **two
  small additive seams** (`_event_plastic.py` signal reader, `_simulator.py` VT
  node + `vt=` binding).

---

## 1. The NEST algorithm (authoritative, from the C++)

### 1a. Continuous model (derived from `update_weight_`, `.h:427-448`)

```cpp
taus_ = (tau_c_ + tau_n_) / (tau_c_ * tau_n_);                       // = 1/tau_c + 1/tau_n
weight_ -= c0 * ( n0/taus_ * expm1(taus_*minus_dt) - b*tau_c_*expm1(minus_dt/tau_c_) );
clip(weight_, Wmin_, Wmax_);
```
With `minus_dt = t0 - t1 < 0` (a negative interval), integrating
`c(s)=c0·exp(-(s-t0)/tau_c)` and `n(s)=n0·exp(-(s-t0)/tau_n)` over `(t0,t1]` gives
`weight += ∫ c(s)·(n(s) − b) ds`. **So the ODE is `dw/dt = c(t)·(n(t) − b)`.** The
time dimension is **baked into `expm1(taus·minus_dt)`** — there is *no* separate
`* dt` in NEST's event-time form. The `b` baseline's sign-flip behaviour
(facilitation↔depression when `n<b`) falls out of the `(n−b)` factor for free; the
verbatim formula needs no special-casing.

### 1b. Eligibility trace `c` (the 04 STDP pairing) — `.h:507-519, 526-578`

`c` is driven by **exactly standard STDP pairing**, but accumulated into `c`
(not directly into `w`):
- **facilitate** on post-after-pre: `c += A_plus · Kplus(t_post)`   (`facilitate_`, `.h:511`)
- **depress** on pre: `c -= A_minus · Kminus(t_pre)`                 (`depress_`, `.h:518`)
- **decay**: `c = c · exp((t0−t1)/tau_c)` over each sub-interval     (`.h:504`)

where `Kplus` is the **pre-trace** (`tau_plus`, `+1` per pre spike — `Kplus_ = Kplus_*exp(Δ/tau_plus)+1`, `.h:573`) and `Kminus` is the **post-neuron** `K⁻` trace
(`tau_minus`, read via `target->get_K_value()`, `.h:565`). **`tau_minus` is a
post-neuron param in NEST** (the 04 gotcha) — kept on the synapse spec for
standalone fidelity, set identically on both sides in the parity drive.

### 1c. Dopamine trace `n` (`update_dopamine_`, `.h:419-425`)

```cpp
minus_dt = dopa[idx].t − dopa[idx+1].t;          // < 0
n_ = n_*exp(minus_dt/tau_n) + dopa[idx+1].multiplicity / tau_n;
```
**Increment is `+multiplicity/tau_n` per dopa spike; decay `exp(−Δ/tau_n)`. No
`* dt`.** In NEST the *synapse* integrates `n`; the VT only buffers `(t,mult)` and
relays (`volume_transmitter.cpp:102-137`), stamping each dopa spike at
`t = (slice_origin + lag + 1)·dt` (**+1-step latency**, `.cpp:113`). VT triggers
`trigger_update_weight` every `deliver_interval·min_delay` steps
(`deliver_interval` default **1**, `.cpp:50`).

### 1d. Event-time integration (`process_dopa_spikes_`, `.h:450-505`)

NEST changes the weight **lazily**, only at pre-spikes (`send`) and VT triggers
(`trigger_update_weight`), integrating `c·(n−b)` piecewise around dopa-spike times
inside each interval. The `weight_recorder` samples the delivered weight at
pre-spike (send) times. **This is the same "online ↔ deferred equality" the 04
cluster relied on: the trajectories coincide at send/trigger times.**

### 1e. Defaults (common props, `stdp_dopamine_synapse.cpp:45-56`)

`A_plus=1.0`, `A_minus=1.5`, `tau_plus=20 ms`, `tau_c=1000 ms`, `tau_n=200 ms`,
`b=0.0`, `Wmin=0.0`, `Wmax=200.0`. Individual: `weight=1.0`, `Kplus=0.0`,
`c=0.0`, `n=0.0`.

---

## 2. Design decision #1 — the broadcast-modulator reader seam (`_network/`)

**RESOLVED: option (a), refined.** Add `signal_reads`/`ctx.signals` as a clean
superset of `post_state_reads`, with the **name→node binding split out of the
frozen spec** (the spec carries names only; the projection resolves the source —
exactly as `post_state_reads` names resolve against `self.post`).

**Why not the goal's literal `signal_reads = (('n', vt),)` (binding embedded in the
class attr)?** A `vt` is a *runtime node handle*; embedding it in a frozen spec
class attr breaks the spec contract every prior cluster established (specs carry
*parameters only*, never runtime handles — `_resolve_synapse` shallow-copies them).
Splitting keeps the spec frozen/serialisable and makes the binding the Simulator's
job (it owns node creation + `connect`).

### The seam (additive, default-empty)

1. **`KernelContext`** gains `signals: dict = None` (after `post_states`). Unit-
   stripped mantissas; `{name: scalar}` broadcast to all edges (JAX broadcasts
   `()`/`(1,)` against `(E,)`). Existing rules ignore it.
2. **Base `EventPlasticProj`** gains a no-op `_gather_signals(self) -> None`
   (exact mirror of `_gather_post_states`); `update()` passes
   `self._gather_signals()` into `ctx.signals`. Primitive #1 unaffected.
3. **`VoltageCoupledPlasticProj`** (the reader primitive) is generalised:
   - new ctor kwarg `signal_sources: dict | None` = `{name: (node, attr)}`;
   - its "must declare a reader" guard relaxes to **`post_state_reads` non-empty
     OR `signal_reads` non-empty** (was: `post_state_reads` only);
   - overrides `_gather_signals()` →
     `{name: u.get_mantissa(getattr(node, attr).value) for name,(node,attr) in self._signal_sources.items()}`
     (or `None` when empty, so clopath — `signal_reads=()`/`signal_sources={}` — is untouched);
   - `_gather_post_states()` returns `None` when `post_state_reads` is empty (so the
     signal-only dopamine proj skips the post gather; clopath unaffected).
4. **Read happens** in `update()` while building `ctx` (phase 1), **read-only**
   (`getattr(node, attr).value`, no writeback). State exactly: the dopamine kernel
   reads `ctx.signals['n']` once, uses it in the weight integral; never writes the VT.

**Rejected (goal option b):** treating the VT as a size-1 "post population" routed
through `post_state_reads` — the broadcast (1→E) vs per-edge gather (N→E) shapes
differ; conflating them muddies the reader contract.

---

## 3. Design decision #2 — VT-latency / integration scheme (`_nest/`)

**RESOLVED: option (a) — online integration on the fixed grid, using NEST's exact
analytic per-step kernel.** Reproduce the `c·(n−b)` integral **every step** with
the broadcast `n(t)` (one-step lag), document the divergence from NEST's event-time
integration — mirroring exactly how 07 handled `delay_u_bars`. Fall back to (b)
(weight update only on dopa-delivery steps) **only if** (a)'s residual exceeds a
defensible band on the P0 trajectory.

### 3a. Why (a) is faithful, not crude

The per-step weight change uses **NEST's own `update_weight_` formula** with
`minus_dt = −dt` (i.e. `update_weight_(c, n, −dt)` once per grid step):

```
taus = (tau_c + tau_n) / (tau_c * tau_n)
dw   = -c * ( n/taus * jnp.expm1(-taus*dt) - b*tau_c * jnp.expm1(-dt/tau_c) )
w    = clip(w + dw, Wmin, Wmax)
```

Because `c` and `n` decay analytically *within* a step (all impulses — dopa jumps
via the VT, STDP pairings — land on grid points), this per-step integral is **exact
for the grid-sampled dynamics**. The only divergences from NEST are:
1. **Grid placement of events** (dopa/pre/post land on the nearest grid point) —
   the standard sub-`dt` discretisation every model carries.
2. **One-step `n` lag.** The proj reads `ctx.signals['n'] = n(t_k)`; NEST's
   sub-interval integral uses `n(t_{k-1})`. With `tau_n=200 ms ≫ dt`, the per-step
   difference is `O(dt/tau_n)≈0.05%`. **Documented**, mirroring 07.

### 3b. VT timing matches NEST's +1 dopa stamp

The VT updates in a new Simulator **phase 0** (before projections), reading the
dopa pool's spike holder (the substrate's intrinsic one-step lag): a dopa neuron
firing at step `j` lands in `n` at step `j+1` — **the same `t_{lag}+1` stamp NEST's
VT applies** (`.cpp:113`). So `n(t)`'s *shape* matches NEST; only its use in the
weight integral carries the documented sub-step lag.

### 3c. Parity posture (07)

Assert **direction + ordering exact** across pulse timings + a **documented band**
(start from 07's 5 %; tighten/loosen to the measured residual), **not 1e-7**.
Record the realised residual + final band in Lessons.

---

## 4. `volume_transmitter` — JAX-native broadcast node (`_nest/volume_transmitter.py`)

Rebuilt as a `brainstate.nn.Module` (retire the eager dict/list/host-scalar
machinery). Maintains `n` as State; computed **once**, broadcast to all bound edges.

- **Params:** `tau_n=200 ms` (decay of `n`), `deliver_interval=1` (accepted for
  NEST API parity; **no-op in the online scheme** — documented). Optional `size=1`
  (number of independent VTs; `n` shape `(size,)`).
- **State:** `n : HiddenState (size,)`, init `0.0`.
- **Dopa source binding:** `bind_dopa(reader, local_idx)` appends a
  `(reader, local_idx)` pair; `connect(dopa_pool, vt)` registers it (no projection
  built — special-cased like recorder taps).
- **`update()`** (called in Simulator phase 0):
  ```
  count = Σ over bound sources of reader()[local_idx]   # # dopa spikes this step (one-step-lagged)
  n = n * jnp.exp(-dt/tau_n) + count / tau_n             # NEST update_dopamine_ increment
  ```
- **Divergence from NEST (documented):** NEST's VT *relays* spikes and the *synapse*
  computes `n`; we move `n` onto the VT (the broadcast generalisation). Valid because
  `n`'s dynamics depend only on `tau_n` + the dopa train (not the synapse) — requires
  `tau_n` shared across all synapses on the VT (it is — common property). The VT's
  `tau_n` and the synapse spec's `tau_n` must match (the parity drive sets both to
  the one NEST value).

---

## 5. `stdp_dopamine_synapse` — spec + rule (`_nest/stdp_dopamine_synapse.py`)

Frozen spec + pure `update(state, ctx) -> (new_state, w_eff)`. Mirrors
`clopath_synapse.py` structure (uses `_plastic_base` helpers, `__module__='brainpy.state'`).

### 5a. Class attributes
```
is_homogeneous_weight = False
stochastic            = False
pre_trace_tau         = tau_plus        # Kplus: +1/spike, decays tau_plus  (cumulative)
post_trace_tau        = tau_minus       # Kminus: post-neuron K⁻, decays tau_minus
post_state_reads      = ()              # no post-voltage gather
signal_reads          = ('n',)          # broadcast dopamine (NEW seam)
weight_unit           = u.pA            # current synapse (preserved by _resolve_synapse)
```

### 5b. `__init__` params (NEST defaults, §1e)
`weight=1.0→pA`, `delay=1.0*ms`, `receptor_type=0`, `A_plus=1.0`, `A_minus=1.5`,
`tau_plus=20*ms`, `tau_minus=20*ms`, `tau_c=1000*ms`, `tau_n=200*ms`, `b=0.0`,
`Wmin=0.0`, `Wmax=200.0`, `c=0.0` (initial per-edge eligibility).
- `n` is **not** a synapse param (lives on the VT); document the divergence.
- `tau_minus` documented as a post-neuron param (04 convention).
- `tau_n` documented as shared with the VT (§4).

### 5c. `edge_state_init`
```
return {'c': float(self.c_init)}        # per-edge eligibility trace (05 machinery)
```

### 5d. `update(state, ctx)` — the kernel (decision #2a)
```
w = state['weight']; c = state['c']
n = ctx.signals['n']                                   # broadcast scalar
# (i) integrate weight over the step (c at t_{k-1}, n at t_k) — NEST update_weight_(c, n, -dt)
taus = (tau_c + tau_n)/(tau_c*tau_n)
dw   = -c * ( n/taus*jnp.expm1(-taus*dt) - b*tau_c*jnp.expm1(-dt/tau_c) )
w    = jnp.clip(w + dw, Wmin, Wmax)
# (ii) decay eligibility c over the step
c    = c * jnp.exp(-dt/tau_c)
# (iii) STDP impulses to c at t_k (04 strictly-prior idiom: subtract current spike)
Kplus_prior  = ctx.pre_trace  - ctx.pre_spike          # pre-trace just before t_k
Kminus_prior = ctx.post_trace - ctx.post_spike         # post K⁻ just before t_k
c = c + A_plus  * jnp.where(ctx.post_spike > 0, Kplus_prior,  0.0)   # facilitate on post
c = c - A_minus * jnp.where(ctx.pre_spike  > 0, Kminus_prior, 0.0)   # depress on pre
return {'weight': w, 'c': c}, w
```
- **Order = NEST's**: integrate `w` with pre-impulse `c` (left-continuous), then
  decay `c`, then apply impulses (impulses affect future intervals).
- **`w_eff = w`** (post-integral stored weight); the substrate gates delivery by
  `ctx.pre_spike`. Weight changes only via the `c·(n−b)` integral, so delivered
  amplitude = stored weight (no instantaneous jump at pre spikes — unlike core STDP).
- No `frozen()` on `w` needed: `w` legitimately drifts every step (the integral runs
  continuously, matching NEST's `process_dopa_spikes_` running on every interval).

---

## 6. Simulator wiring (`_network/_simulator.py`, additive)

- `__init__`: `self._vt_nodes = []`.
- `create()`: `if isinstance(mod, volume_transmitter): self._vt_nodes.append(mod); setattr(self, f'_node_{id(mod)}', mod); return NodeView.of(mod)` — **no holder** (the
  VT emits no spikes into a holder; it exposes `n` as State).
- `connect()` top: branch — if `post` is a single VT segment, call
  `vt.bind_dopa(_holder_reader(dopa_holder), dopa_seg.indices)` and return `None`
  (like recorder taps). Supports multiple dopa pools → one VT.
- `connect()` signature: add `vt=None`. Thread to `_connect_pair`.
- `_plastic_proj_cls`: route to `VoltageCoupledPlasticProj` when
  `post_state_reads` **or** `signal_reads` is non-empty.
- `_connect_pair`: when `synapse.signal_reads` is non-empty, build
  `signal_sources = {name: (vt_module, name) for name in synapse.signal_reads}`
  (one bound VT) and pass it; **raise a clear error if `vt is None`**.
- `update()`: new **phase 0** before the projection loop —
  `for vt in self._vt_nodes: vt.update()` — so projections read fresh `n`.
- `record_weight`/`weight_trace`: reused unchanged for the dopamine proj.

Non-dopamine paths are fully untouched: no `signal_reads`, no `vt`, empty
`_vt_nodes` → phase 0 is a no-op, `_gather_signals` returns `None`, `ctx.signals`
is `None`.

---

## 7. Validation (`_nest/_validation/`, `requires_nest`)

New drive `_stdp_dopamine_drive.py` (mirrors `_clopath_drive.py`/`_stdp_drive.py`),
two parity tests, one NEST-free rule test.

### 7a. VT first, in isolation — `volume_transmitter_parity_test.py`
Drive a known dopaminergic spike pattern; assert the brainpy VT `n(t)` matches NEST's
dopamine concentration. **NEST reference:** create `volume_transmitter` + a
`stdp_dopamine_synapse`, simulate in chunks and `GetStatus(conn, 'n')` at chunk
boundaries (NEST's *own* `n`, the live cross-check); secondary sanity vs the
analytic `update_dopamine_` recursion. `compare_trace`, small documented band.
(Same "verify the upstream node first" discipline 07 used for Clopath voltage.)

### 7b. P0 — `stdp_dopamine_synapse_parity_test.py`
3-neuron + 1-VT topology, **known dopamine-pulse pattern**, weight trajectory over
**~5 s** via `res.weight_trace(proj)` vs NEST's `weight_recorder` (sampled at
send/trigger times), within the documented band; **direction + ordering exact**
across pulse timings (07 posture). NEST side uses the 04 strong-driven
`iaf_psc_delta` post (`V_th=1e4`, driver weight `1e6`) to avoid parrot-relay
pollution; dendritic-delay convention; `parrot`-relayed dopa pool → VT;
`CopyModel(stdp_dopamine_synapse, …, {volume_transmitter: vt, …})`.

### 7c. Eligibility-trace timing sweep
Vary the dopa-spike time relative to the pre/post pair (the gap-doc "high-risk
surface"); assert sign + magnitude of `Δw` tracks NEST.

### 7d. Rule test — `stdp_dopamine_synapse_rule_test.py` (NEST-free)
Hand-built `KernelContext` (incl. `signals={'n': …}`); closed-form host reference
for `c` + `w` integral; jit/vmap/grad smoke; all §8 edge cases; coverage via the
05 untraced-import + `coverage.start()`+`reload` workaround.

---

## 8. Edge cases to enumerate + test

1. **No dopamine spikes** → `n→0`, pure eligibility decay, weight integral → 0 (b=0).
2. **Dopa spike with no eligibility** (`c=0`) → no weight change.
3. **`Wmin`/`Wmax` clamp** on the integral (both signs of `n−b`).
4. **Broadcast correctness:** ≥2 synapses on one VT see the **same** `n`.
5. **VT independence:** two VTs in one sim stay independent.
6. **Dopa spike coincident with a pre/post pair** (ordering at the step boundary).
7. **`b≠0`** → facilitation↔depression sign-flip falls out of `(n−b)`.
8. **VT bound to a sliced/multi-segment dopa pool** (sum across bound sources).
9. **`dt` invariance** of the documented-band result (sweep `dt∈{0.1,0.5,1.0}`).
10. **`vmap`/`grad` smoke** on the kernel.
11. **Facilitation/depression simultaneity** (`−ctx.pre_spike`/`−ctx.post_spike` idiom).
12. **No `vt` given** to a dopamine `connect` → clear error.

---

## 9. Open question for sign-off

Both design decisions (§2, §3) are resolved with the recommended options and
justified above. The one judgement call deferred to measurement: the **final
documented band** in §3c (start 5 %, set to the realised P0 residual). If (a)'s
residual is indefensible, fall back to §3 option (b) (weight update gated to
dopa-delivery steps) — more machinery, closer to NEST's event-time integral.
