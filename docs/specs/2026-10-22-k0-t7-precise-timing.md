# K0 T7 — `timed_event` ports: precise spike timing in JAX

_Date: 2026-10-22_ · Parent: [`2026-10-21-k0-kernel-spec.md`](2026-10-21-k0-kernel-spec.md)
(§2 model table, §4.2 `TimedEvent(capacity)`, §5 row T7, §6 G1 precise part, §7 edge
cases, §10 decision 2) and the architecture doc §7 L1 (`timed_event` port) and §7.1.

T7 builds the precise-timing pair in JAX (`iaf_psc_exp_ps`, `poisson_generator_ps`, plus a
precise `spike_generator_ps` for tests) and network delivery for `TimedEvent` ports. The
existing host models (`_nest_neuron/iaf_psc_exp_ps.py`, `_nest_device/poisson_generator_ps.py`)
are untouched; only their algorithms (= NEST's) are reused.

## 1. Time convention

Step `n` covers `(n dt, (n + 1) dt]`. A timed event carries an **offset from the step
start** in `(0, dt]`; its time is `n dt + offset`. NEST describes the same event by the
stamp `n + 1` and the backward offset `dt - offset`. A grid `event` emitted at step `n`
(NEST stamp `(n + 1) dt`) seen by a timed port has offset `dt`.

`TimedEventPayload` (T1) is the per-step container: `offsets (n, K)`, `weights (n, K)`,
`valid (n, K)`, `overflow (n,)`; valid events first, sorted by offset, invalid slots at
`+inf`.

## 2. Models (`brainpy_state/_kernel/models/precise.py`)

| class | ports | states | notes |
|---|---|---|---|
| `iaf_psc_exp_ps` | in `syn`: TimedEvent(16, pA), `I`: Current(pA, latched); out `spike`: TimedEvent(1) | `V_m`, `I_syn_ex`, `I_syn_in`, `latch_I`, `spike` (count), `spike_offset` (ms, inf if none), `is_refractory`, `ref_count`, `last_spike_offset`, `overflow` | NEST `iaf_psc_exp_ps` |
| `poisson_generator_ps` | out `spike`: TimedEvent(8) | `spike`, `spike_offsets` (n, K), `next_spike`, `started`, `overflow` | NEST `poisson_generator_ps` incl. `dead_time` |
| `spike_generator_ps` | out `spike`: TimedEvent(K) | `spike`, `spike_offsets` (n, K) | NEST `spike_generator` with `precise_times=True` |

Capacities are class-level port declarations (the network validates against classes);
`Model.with_capacity(K)` returns a cached subclass with another capacity.
`spike_generator_ps` counts spikes beyond its per-step capacity in `overflow` (never a silent
drop). Its step stamps use an exact ceil (`ms_stamp` as in NEST, snapping only within a few
ulps), not the grid `GRID_SNAP` rounding, so a time 4e-5 ms after a grid point stays in the
next step.

### 2.1 `iaf_psc_exp_ps` step (vectorised, fixed iteration counts)

1. Instant spike if `V_m >= V_th` at the step start (initialisation only; offset 0).
2. Refractory bookkeeping: a neuron that spiked at step `m` with offset `o` returns from
   refractoriness at step `m + R` at offset `o`, `R = max(1, ceil(t_ref / dt))` (NEST
   requires `t_ref >= dt`; we clamp instead of raising because parameters are traced).
   The return is a pseudo-event merged into the step's event list (NEST
   `SliceRingBuffer::add_refractory`; it sorts before a coinciding spike).
3. The `K + 1` events (inputs sorted by offset at the receiver + the pseudo-event) split
   the step into `K + 2` Python-unrolled segments. Each segment propagates exactly
   (`expm1`/`exp` propagators, float32-safe `propagator_exp` from T2); `V` is frozen
   while refractory. After a segment of positive length the threshold is checked; a
   crossing stores the bracket (segment start state + length) and resets `V`. Inputs:
   `w >= 0` adds to `I_syn_ex`, `w < 0` to `I_syn_in` (NEST sign routing).
4. Steps without any event use NEST's fixed-step formula (`expm1(-h/tau) * y + y`) and the
   `V_min` clamp (NEST applies `V_min` only on that path).
5. Spike time: at most one spike per step (refractoriness >= 1 step), so the root finder
   runs once per step, after the segment loop, only if some neuron crossed
   (`lax.cond`): Illinois regula falsi (NEST's `regula_falsi`) with a fixed 40
   iterations, then an implicit-function correction (value unchanged, gradient
   `dt*/dθ = -∂f/∂θ / f'(t*)`).

Spike-time gradients: the output offset is differentiable w.r.t. parameters, weights,
input offsets and states (implicit function theorem); the spike *count* is a hard
threshold (no surrogate). Offsets travel to targets through the payload, so gradients
flow through timed delivery (EventProp-style, K1).

### 2.2 `poisson_generator_ps`

Per unit (= per target, NEST draws one train per target): `next_spike` relative to the
step start. Active window `(max(n dt, origin + start), min((n + 1) dt, origin + stop)]`.
First activation draws from NEST's equilibrium initialisation (uniform in
`[0, dead_time)` with probability `dead_time * rate`, else `dead_time + Exp * inv_rate`),
then `K` inter-spike intervals `dead_time + Exp * (1/rate - dead_time)` are drawn at once
(vectorised cumulative sum, no loop). Events beyond `K` in one step increment `overflow`
(1 plus the expected remaining count); the next spike is then redrawn memorylessly from
the step end (exact for `dead_time = 0`, approximate otherwise). Randomness comes from
`ctx.key` only.

## 3. Network delivery

* `connect`: kinds `timed_event -> timed_event`, `timed_event -> event` (offset dropped,
  count delivered on the grid, NEST) and `event -> timed_event` (offset `dt`). Delays are
  rounded to the grid (NEST); offsets are preserved.
* Projections from a timed source use the **timed** placement: per-edge gather from the
  source's `TimedRing` (`(L, n, K)` offsets + weights; `L` = largest delay) or from the
  current step's output for delay 0, times the edge weight, gathered per target through a
  padded in-edge table and `TimedEventPayload.pack`-ed into the port capacity (sorted at
  the receiver). Storage is always CSR (exact per-edge weights; multapses are separate
  events as in NEST). Event sources into timed ports use the normal grid path and are
  converted to one event per target at offset `dt`.
* Several projections into one timed port are merged (`TimedEventPayload.merge`).
* Capacity: at connect time a single source's capacity must fit the port; at compile time
  the worst case per target (in-degree × source capacity, summed over projections into
  the port; 1 per grid projection) must fit. Both raise `PortError` unless the projection
  was created with `strict_capacity=False`; at run time events beyond the capacity are
  dropped **and counted** in the receiver's `overflow` state (never silent).

## 4. Tests and gate

* G1 precise (NEST 3.10, `requires_nest`): `spike_generator_ps -> iaf_psc_exp_ps` and
  `poisson_generator_ps -> iaf_psc_exp_ps` (NEST gets the same spike times through a
  precise `spike_generator`), several delays, inhibitory input, `V_min`, refractoriness,
  multiple events in one step: `V_m` `CAT_B` (1e-6 mV), output spike times within 1e-6 ms,
  `align_steps = 0`.
* `poisson_generator_ps` statistics: rate, exponential ISIs (KS), uniform offsets (KS),
  dead time (no ISI below it, mean ISI preserved), start/stop window.
* Overflow (source and receiver), refractory edge cases (`t_ref < dt`, `t_ref` not a
  multiple of `dt`, return mid-step), float32 vs float64 (measured: V within 1e-4 mV, spike
  times within 3e-5 ms; tested at 1e-3 mV / 1e-4 ms), gradients vs finite
  differences.
