# Design: Synaptic Delays & Missing Features — `brainpy.state` (`_brainpy` layer)

Status: **Proposal** · Scope: `brainpy_state/_brainpy` · Author: architecture review · Date: 2026-06

This document accompanies the correctness-bug fixes (B1–B14, plus the HH-family
rising-edge fix). Those changes repaired *existing* behaviour. This document
covers **missing capabilities** and a focused **architecture assessment**, so the
maintainers can decide what to build next and in what order.

---

## 0. Executive summary

| Area | Today | Gap | Recommended priority |
|------|-------|-----|----------------------|
| **Synaptic delays** | Only `DeltaProj` can take a low-level `PrefetchDelayAt`. `AlignPostProj`, `CurrentProj` and the gap junctions expose **no `delay=`**. | First-class, unit-carrying `delay=` on every projection at all three granularities — global, axonal (per-pre-neuron) **and heterogeneous per-connection** — via the diagonal gather `brainstate.nn.Delay` already supports. | **P1 — highest** |
| **Input generators** | `SpikeTime`, `PoissonSpike`, `PoissonEncoder`, `PoissonInput`, `poisson_input`. | No constant/section, step, ramp, sinusoidal, Ornstein–Uhlenbeck, or Wiener current sources. | **P2** |
| **Neuron models** | LIF family (incl. QuaIF/AdQuaIF/Gif), HH/MorrisLecar/WangBuzsaki, Izhikevich. | No FitzHugh–Nagumo, Hindmarsh–Rose, or a one-line conductance-based LIF (`CobaLIF`/`CubaLIF`). | **P3** |
| **Architecture** | Input summation (`add_*_input`/`sum_*_inputs`) and projection patterns are sound. | The **delay seam** is the one genuine structural gap; everything else is additive. | see §4 |

**Bottom line:** the `_brainpy` layer does **not** need a rewrite. It needs one
new *deep* seam — a unified **delay** abstraction — plus additive library
content (inputs, models). The delay seam is the only item with architectural
weight; the rest are leaf additions that fit the current patterns.

---

## 1. Synaptic delays (P1)

### 1.1 Why this matters

Conduction/synaptic delay is not a nicety — it is *required* for the dynamics
the library is built to study: oscillation phase, synfire chains, E/I balance,
polychronization, and any network whose behaviour depends on spike timing.
Today a user cannot write `delay=1.5 * u.ms` on a standard projection.

### 1.2 What already exists (build on this, don't reinvent)

`brainstate` ships the delay machinery, and — importantly — it already supports
**per-element (heterogeneous) reads**, so we do not need to build a buffer of our
own:

- `brainstate.nn.Delay` — rolling history buffer of a variable. Two read APIs
  carry an optional index argument:
  - `retrieve_at_step(delay_steps, *indices)` — integer-step read.
  - `retrieve_at_time(delay_time, *indices)` — time read (needs `t` in `environ`),
    with sub-`dt` delays resolved by `interp_method` (default `'linear_interp'`).
  - `register_entry(name, time, *idx)` + `at(name)` — pre-register a fixed access
    pattern (including a per-connection index vector).
- `brainstate.nn.StateWithDelay` — a `State` that also maintains a delay buffer.
- `brainstate.nn.PrefetchDelayAt` / `PrefetchDelay` / `DelayAccess` — higher-level
  read of a module's state at `t − delay` (the homogeneous case).

The key behaviour (verified against the installed `brainstate`) is how
`retrieve_at_step` combines a delay vector with indices:

```text
retrieve_at_step(steps)            -> outer product: buffer[steps[a], :]      shape (len(steps), N)
retrieve_at_step(steps, idx)       -> diagonal gather: buffer[steps[k], idx[k]] shape (len(idx),)
```

The **diagonal gather** is exactly a heterogeneous, per-connection delayed read:
connection `k` reads source neuron `idx[k]` at its own offset `steps[k]`.

`DeltaProj` already accepts the homogeneous wrapper:

```python
# brainpy_state/_brainpy/projection.py  (DeltaProj.__init__)
self.prefetch = prefetch        # last element may be a PrefetchDelayAt
```

So the primitive exists (including heterogeneous support), but it is (a) only
wired into `DeltaProj`, (b) exposed as a raw positional `*prefetch` rather than an
ergonomic `delay=`, and (c) the heterogeneous gather is not surfaced at all.

### 1.3 Three delay granularities — all supported

The delay attaches to one shared history buffer of the pre-synaptic source; the
granularities differ only in *how that buffer is read each step*. All three reduce
to one `retrieve_at_step(delay_steps, indices)` call, so a single implementation
covers them:

| Granularity | `delay` value | Read | Buffer | Per-step read cost |
|-------------|---------------|------|--------|--------------------|
| **Global / homogeneous** | scalar `Quantity` (`1.5*u.ms`) | `retrieve_at_step(k)` | depth × `N_pre` | none (slice) |
| **Axonal (per-pre-neuron)** | array `(N_pre,)` | `retrieve_at_step(steps, arange(N_pre))` | depth × `N_pre` | `O(N_pre)` gather |
| **Synaptic (per-connection, heterogeneous)** | array `(N_syn,)` | `retrieve_at_step(steps, pre_ids)` | depth × `N_pre` | `O(N_syn)` gather |

The buffer is always over the pre-synaptic neurons (`N_pre`), so adding
heterogeneity does **not** grow memory — only the read pattern changes. This is
the property that lets us promote heterogeneous delays from "expensive, maybe
later" to "supported from day one": NEST-style `delay` arrays and BrainPy 2.x
heterogeneous synapses map onto the per-connection row directly.

Heterogeneity is naturally **per-connection**, so it pairs with explicit /
sparse connectivity (`pre_ids`/`post_ids`, event-driven `brainevent` operators)
where a connection list exists. For a dense `comm` (e.g. `Linear`) a true
per-(pre,post) delay would need an `N_pre × N_post` buffer and is intentionally
**not** offered; the practical granularity for dense projections is axonal
(per-pre-neuron). The API enforces this by accepting per-connection `delay` only
on projections that own a connection index.

### 1.4 Proposed interface

Add an optional, unit-carrying `delay=` to the projection classes, backed by one
delay seam. The buffer and the per-step read are encapsulated in a small module so
every projection (and the gap junctions, later) shares one implementation.

```python
# new: brainpy_state/_brainpy/_delay.py
class DelayedSource(brainstate.nn.Module):
    """Maintain a delay buffer over ``source.<state>`` and read it each step.

    delay is None        -> no buffer; direct read (current behaviour, zero cost)
    delay is scalar      -> global homogeneous delay
    delay is (N_pre,)    -> axonal, per-pre-neuron
    delay is (N_syn,)    -> synaptic, per-connection (requires ``indices`` = pre_ids)
    """
    def __init__(self, source, state, delay=None, indices=None):
        super().__init__()
        self.source, self.state, self.delay, self.indices = source, state, delay, indices
        self._buf = None        # a brainstate.nn.Delay, sized at init_state
        self._read_idx = None   # gather indices for the per-step read

    def init_state(self, *args, **kwargs):
        if self.delay is None:
            return
        value = getattr(self.source, self.state).value
        dt = brainstate.environ.get_dt()
        max_steps = int(u.math.ceil(u.math.max(self.delay) / dt))      # static at trace time
        self._buf = brainstate.nn.Delay(value, time=max_steps * dt,
                                        interp_method='linear_interp')
        # Resolve the read pattern once (see §1.3):
        #   scalar delay            -> no indices  (whole-population slice)
        #   (N_pre,) axonal         -> arange(N_pre) so the read is the *diagonal*
        #   (N_syn,) per-connection -> the supplied pre_ids
        if u.math.ndim(self.delay) == 0:
            self._read_idx = None
        elif self.indices is not None:
            self._read_idx = self.indices
        else:
            self._read_idx = jnp.arange(value.shape[-1])

    def update(self):
        value = getattr(self.source, self.state).value
        if self.delay is None:
            return value
        self._buf.update(value)
        steps = u.math.asarray(self.delay) / brainstate.environ.get_dt()   # per-element
        # NOTE: a bare retrieve_at_step(vector) is the OUTER product, not what we
        # want; the diagonal gather requires the index argument.
        return (self._buf.retrieve_at_step(steps) if self._read_idx is None
                else self._buf.retrieve_at_step(steps, self._read_idx))
```

```python
# AlignPostProj / CurrentProj gain a keyword:
proj = brainpy.state.AlignPostProj(comm, syn, out, post, delay=1.5 * u.ms)   # global
proj = brainpy.state.AlignPostProj(comm, syn, out, post, delay=axonal_ms)    # (N_pre,)

# Sparse / explicit-connectivity projection carries per-connection delays:
proj = brainpy.state.SparseProj(conn=conn, ..., delay=syn_delays_ms)         # (N_syn,)
# -> DelayedSource(pre, 'spike', delay=syn_delays_ms, indices=conn.pre_ids)
```

When `delay is None` the projection takes its current code path verbatim — zero
overhead, full backward compatibility. Fractional (sub-`dt`) delays are honoured
through the buffer's `linear_interp`, so `delay` need not be an integer multiple
of `dt`.

### 1.5 Phasing

Heterogeneous support is part of the **design** from the start (same keyword, same
seam); the phases below are purely an *implementation* ordering, not a capability
gate.

- **P1a — homogeneous + axonal.** Scalar and `(N_pre,)` `delay=` on `AlignPostProj`
  and `CurrentProj`. Smallest change; covers most networks.
- **P1b — heterogeneous (per-connection).** `(N_syn,)` `delay=` on sparse /
  explicit-connectivity projections, reading `retrieve_at_step(steps, pre_ids)`.
  No extra buffer memory over P1a; only the gather index changes. Add a
  `max_delay` guard and validate `len(delay) == N_syn`.
- **P1c — gap-junction delay.** Electrical coupling is near-instantaneous, so this
  stays out of scope; a passed `delay` must raise rather than be silently ignored.

### 1.6 Trade-offs & risks

- **Memory** is set by buffer *depth* = `ceil(max_delay / dt)` × `N_pre`, and is
  **independent of the delay granularity** (homogeneous, axonal and per-connection
  share one `N_pre` buffer). A 20 ms max delay at `dt = 0.01 ms` is 2000 frames ×
  `N_pre`. `delay` is given in time units (`brainunit`) and converted with the
  *current* `dt`; buffer depth is sized from `max(delay)` at `init_state`.
- **Compute:** per-connection delay adds an `O(N_syn)` gather per step on top of
  the existing comm; homogeneous adds nothing (a slice).
- **JIT:** buffer depth must be a static Python int at trace time — derive it from
  `max(delay)`/`dt` in `init_state`, never from a traced value. The per-element
  `delay_steps` passed to `retrieve_at_step` *may* be traced.
- **`dt` changes:** the buffer is sized once; changing `dt` afterwards must fail
  loudly (a stale buffer would silently misalign delays).
- **Backward compatibility:** `delay=None` default ⇒ no behavioural change.

### 1.7 Test plan

- **Homogeneous:** a single spike through `delay=k·dt` arrives at post exactly `k`
  steps later.
- **Axonal:** an `(N_pre,)` delay delays each pre-neuron's contribution
  independently; pre-neuron `j` arrives after `round(delay[j]/dt)` steps.
- **Heterogeneous:** a 3-connection projection with delays `[1, 5, 10]·dt` from a
  single impulse delivers three pulses offset by 1, 5, 10 steps — directly
  asserting the `retrieve_at_step(steps, pre_ids)` diagonal gather.
- **Fractional:** `delay=0.5·dt` interpolates between adjacent frames
  (`linear_interp`), value ≈ mean of the two bracketing samples.
- **Regression:** `delay=None` reproduces current outputs bit-for-bit.
- **JIT:** buffer depth is static under `jit`; changing `dt` after sizing raises;
  per-connection `delay` of wrong length raises at construction.

---

## 2. Input generators (P2)

`inputs.py` currently covers *spiking* sources well (Poisson, spike-time) but has
**no analog current generators**. These are standard stimulation primitives and
each is a small, self-contained function/Module that fits the existing style
(unit-carrying, `brainstate.environ.get_dt()`-aware, JIT-safe).

Proposed additions (names follow `braintools`/BrainPy convention):

| Generator | Signature sketch | Notes |
|-----------|------------------|-------|
| `section_input` | `(values, durations)` | Piecewise-constant current; the building block for the rest. |
| `constant_input` | `(value, duration)` | Thin wrapper over `section_input`. |
| `step_input` | `(amplitudes, step_times)` | Staircase stimulus. |
| `ramp_input` | `(c_start, c_end, duration, t_start=0)` | Linear ramp. |
| `sinusoidal_input` | `(amplitude, frequency, duration, bias=0)` | Periodic drive. |
| `ou_input` (Ornstein–Uhlenbeck) | `(mean, sigma, tau, duration)` | Coloured noise; integrate `dx = (mean−x)/tau·dt + sigma·√(2/tau)·dW`. Stateful `Module`. |
| `wiener_input` | `(sigma, duration)` | White-noise/Brownian drive; `sigma·√dt·N(0,1)` per step. |

**Convention to enforce** (learned from bugs B10/B13): every stochastic generator
must (a) build randomness through `brainstate.random`, (b) scale noise by `√dt`
(not `dt`) for the diffusion term, and (c) carry units so `add_current_input`
sees a current. OU and Wiener are `Module`s with state; the deterministic ones
can be pure functions returning a precomputed array (like `SpikeTime`) or a
`Module` that emits per-step (like `PoissonSpike`) — prefer the per-step `Module`
form to avoid materialising long arrays under `jit`.

---

## 3. Neuron models (P3)

The IF/HH/Izhikevich coverage is broad. The notable absences are two classic
2-D reduced models and a convenience LIF:

- **FitzHugh–Nagumo** — 2-D excitable system (`v`, `w`); no reset, continuous
  spikes. *Caveat:* it has the same no-reset property as HH, so its spike output
  must use the **rising-edge** detector introduced for the HH family (the B4
  fix), not a per-step threshold. This is the one place a new model could
  re-introduce the B4 bug — call it out in the model template.
- **Hindmarsh–Rose** — 3-D bursting model (`x`, `y`, `z`); valuable because no
  current model produces bursts. Same rising-edge spike caveat.
- **`CobaLIF` / `CubaLIF`** — a LIF with a built-in conductance-/current-based
  synaptic term, i.e. the common case bundled so users don't wire `LIF + Expon +
  COBA` by hand. Pure convenience; composes existing parts.

Each is additive and uses the existing `Neuron` base + `exp_euler_step`
integration; none requires architectural change.

---

## 4. Architecture assessment

**Does `_brainpy` need an upgrade? — No rewrite; one new seam.**

What is already deep and should be left alone:

- **Input summation** (`add_current_input` / `add_delta_input` /
  `sum_current_inputs` / `sum_delta_inputs`). Small interface, high leverage:
  callers register named contributions and the base class composes them. The B14
  fix reinforced the one rule this seam needs — *contributions are keyed by a
  stable per-instance name* — which is now consistent across all projections.
- **`Neuron` / `Synapse` / surrogate-gradient** split. Clear abstract methods
  (`get_spike`, `update`), configurable `spk_fun` / `spk_reset`.
- **Projection patterns** (`AlignPostProj` align-post merging, `DeltaProj`,
  `CurrentProj`). The descriptor-based sharing is the right amount of machinery.

The one **shallow / missing seam** is **delay**. Right now delay leaks through as
a raw `*prefetch` argument on a single projection class — an interface nearly as
complex as wiring `PrefetchDelayAt` by hand, with no locality (every call site
re-derives it, and the heterogeneous per-connection case is unreachable). A
`delay=` keyword backed by one `DelayedSource` seam (§1.4) turns that into a deep
module: trivial interface (`delay=1.5*u.ms` or a per-connection array), one
implementation covering all three granularities, every projection benefits. By
the *deletion test*, removing the seam would scatter `Delay`/`PrefetchDelayAt`
plumbing — and a hand-rolled gather — across every projection and every user
network; i.e. it earns its keep.

**Secondary, optional hardening (not required):**

- The `BindCondData` contract (`bind_cond` with no paired `unbind`) is correct
  for the current "rebind every step" usage (verified during this review), but
  the invariant lives only in prose. If a future model reads current inputs more
  than once per step with *conditional* binding, the stale-conductance guard
  would not catch a missed bind. This is a latent foot-gun, not a bug; leave as
  documented unless that usage appears.

---

## 5. Recommended sequencing

1. **P1a** global + axonal `delay=` on `AlignPostProj`/`CurrentProj` via the
   `DelayedSource` seam (+ tests). Highest value, smallest change, unblocks
   timing-dependent networks.
2. **P1b** heterogeneous per-connection `delay=` on sparse/explicit-connectivity
   projections. A small delta over P1a — same buffer, same seam; only the read
   becomes `retrieve_at_step(steps, pre_ids)` and `delay` is `(N_syn,)`. No extra
   memory. Lands right after P1a because it reuses the same code.
3. **P2** `section`/`constant`/`step`/`ramp`/`sinusoidal` inputs (pure, easy),
   then `ou`/`wiener` (stateful, apply the √dt + `brainstate.random` rules).
4. **P3** `CobaLIF`/`CubaLIF` (compose existing parts), then FitzHugh–Nagumo and
   Hindmarsh–Rose (reuse the rising-edge spike detector — do **not** regress B4).

Each item is independently shippable and testable; none blocks the others except
P1b building on P1a (shared `DelayedSource` seam).
