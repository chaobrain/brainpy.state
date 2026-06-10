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
| **Synaptic delays** | Only `DeltaProj` can take a low-level `PrefetchDelayAt`. `AlignPostProj`, `CurrentProj` and the gap junctions expose **no `delay=`**. | First-class, unit-carrying `delay=` on every projection; heterogeneous (per-connection) delays. | **P1 — highest** |
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

`brainstate` ships the delay machinery:

- `brainstate.nn.Delay` — rolling history buffer of a variable.
- `brainstate.nn.StateWithDelay` — a `State` that also maintains a delay buffer.
- `brainstate.nn.PrefetchDelayAt` / `PrefetchDelay` / `DelayAccess` — read a
  module's state at `t − delay`.

`DeltaProj` already accepts these:

```python
# brainpy_state/_brainpy/projection.py  (DeltaProj.__init__)
self.prefetch = prefetch        # last element may be a PrefetchDelayAt
```

So the primitive exists, but it is (a) only wired into `DeltaProj`, and (b)
exposed as a raw positional `*prefetch` rather than an ergonomic `delay=`.

### 1.3 Two distinct delay semantics

These are genuinely different and should be named, not conflated:

1. **Axonal / output delay** — one delay per *pre-synaptic neuron* (or one
   scalar for the whole projection). The pre-synaptic spike train is delayed
   *before* the communication module. One shared buffer of depth
   `ceil(max_delay / dt)`. Cheap: `O(N_pre)` memory, no gather.

2. **Synaptic delay** — one delay per *connection* (heterogeneous). Each synapse
   reads the pre-synaptic history at its own offset. Requires a gather over the
   delay buffer at per-connection indices. `O(N_pre · max_steps)` buffer +
   `O(N_syn)` gather per step.

Most models need only (1). (2) is what NEST's `delay` array and BrainPy 2.x's
heterogeneous-delay synapses provide, and is the harder, memory-heavier case.

### 1.4 Proposed interface

Add an optional, unit-carrying `delay=` to the projection classes. The seam is
a single helper that turns `(source_module, state_name, delay)` into a delayed
read, so every projection shares one implementation.

```python
# new: brainpy_state/_brainpy/_delay.py
def delayed_prefetch(source, state: str, delay=None):
    """Return a read of ``source.<state>`` at ``t - delay``.

    delay is None         -> direct prefetch (no buffer)
    delay is a scalar     -> homogeneous axonal delay (one Delay buffer)
    delay is an array     -> per-pre-neuron axonal delay (gather on read)
    """
    if delay is None:
        return brainstate.nn.Prefetch(source, state)
    return brainstate.nn.PrefetchDelayAt(source, state, delay)
```

```python
# AlignPostProj / CurrentProj gain a keyword:
proj = brainpy.state.AlignPostProj(
    comm=comm, syn=syn, out=out, post=post,
    delay=1.5 * u.ms,          # NEW — homogeneous axonal delay
)
```

Internally the projection wraps its pre-synaptic source access in
`delayed_prefetch(...)`; when `delay is None` the behaviour is byte-for-byte the
current path (zero overhead, full backward compatibility).

### 1.5 Phasing

- **P1a — homogeneous axonal delay.** Scalar `delay=` (and per-pre-neuron array)
  on `AlignPostProj` and `CurrentProj`, delegating to `PrefetchDelayAt`. Smallest
  change, covers the majority of use-cases. The buffer depth is derived from
  `delay` and the environment `dt` at `init_state`.
- **P1b — heterogeneous synaptic delay.** Per-connection delay vector, read via a
  gather over the history buffer. Land behind the same `delay=` keyword
  (an array sized to the connection count selects this path). Document the memory
  cost and provide a `max_delay` guard.
- **P1c — gap-junction delay.** Out of scope initially; electrical coupling is
  near-instantaneous. Document the omission rather than silently ignoring a
  passed `delay`.

### 1.6 Trade-offs & risks

- **Memory:** buffer depth = `ceil(max_delay / dt)`. A 20 ms max delay at
  `dt = 0.01 ms` is 2000 frames × `N_pre`. The API should accept `delay` in time
  units (via `saiunit`) and convert with the *current* `dt`, failing loudly if
  `dt` changes after the buffer is sized.
- **JIT:** buffer depth must be static (Python int) at trace time — derive it
  from `delay`/`dt` at `init_state`, not from a traced value.
- **Backward compatibility:** `delay=None` default ⇒ no behavioural change.

### 1.7 Test plan

- A single spike through a projection with `delay=k·dt` appears at the post
  population exactly `k` steps later (homogeneous).
- Heterogeneous: a 3-connection projection with delays `[1, 5, 10]·dt` delivers
  three offset pulses.
- `delay=None` reproduces the current outputs bit-for-bit (regression guard).
- Buffer depth is static under `jit`; changing `dt` after sizing raises.

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
re-derives it). A `delay=` keyword backed by one `delayed_prefetch` helper (§1.4)
turns that into a deep seam: trivial interface (`delay=1.5*u.ms`), one
implementation, all projections benefit. By the *deletion test*, removing the
helper would scatter `PrefetchDelayAt` plumbing across every projection and every
user network — i.e. it earns its keep.

**Secondary, optional hardening (not required):**

- The `BindCondData` contract (`bind_cond` with no paired `unbind`) is correct
  for the current "rebind every step" usage (verified during this review), but
  the invariant lives only in prose. If a future model reads current inputs more
  than once per step with *conditional* binding, the stale-conductance guard
  would not catch a missed bind. This is a latent foot-gun, not a bug; leave as
  documented unless that usage appears.

---

## 5. Recommended sequencing

1. **P1a** homogeneous axonal `delay=` on `AlignPostProj`/`CurrentProj` (+ tests).
   Highest value, smallest change, unblocks timing-dependent networks.
2. **P2** `section`/`constant`/`step`/`ramp`/`sinusoidal` inputs (pure, easy),
   then `ou`/`wiener` (stateful, apply the √dt + `brainstate.random` rules).
3. **P1b** heterogeneous per-connection delay (memory-aware, behind the same
   keyword).
4. **P3** `CobaLIF`/`CubaLIF` (compose existing parts), then FitzHugh–Nagumo and
   Hindmarsh–Rose (reuse the rising-edge spike detector — do **not** regress B4).

Each item is independently shippable and testable; none blocks the others except
P1b building on P1a.
