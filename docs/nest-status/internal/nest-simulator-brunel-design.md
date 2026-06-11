# Design — NEST-style `Simulator` API + `brunel_alpha` flagship port

- **Date:** 2026-06-11
- **Status:** approved (brainstorming) — pending spec review
- **Owner:** Chaoming Wang
- **Cross-links:** `docs/nest-status/internal/examples-gap.md` §3.1 (Flagship
  benchmarks), §7 P0; `docs/nest-status/internal/network-api-gap.md` §7 P0;
  `DESIGN_delays_and_missing_features.md`.

## 1. Problem & motivation

The repo ships 167+ NEST-compatible models in `brainpy_state/_nest`, but **zero**
NEST examples are ported, and the only network-construction surface is the
brainpy-style compositional API (`AlignPostProj`, `LIF`/`Expon`/`COBA`, the
`examples/1xx_*` gallery). NEST users have nothing that reads like PyNEST, and —
more importantly — the `_nest` models have **never been exercised end-to-end in a
network against live NEST**. Per `examples-gap.md` §5, *each ported example is
also a validation harness*: porting `brunel_alpha_nest.py` is simultaneously the
flagship network example and the first network-level parity test for
`iaf_psc_alpha` + `poisson_generator` + `spike_recorder` + static synapse +
delays.

**This effort delivers that flagship port and, through it, fixes whatever
`_nest` model bugs the port surfaces.** The network-construction syntax is
deliberately *not* the existing brainpy API; it is a new, explicit, NEST-flavored
`Simulator` object.

NEST **3.9.0 is importable in this environment**, so the validation harness
compares against live NEST rather than hard-coded reference numbers.

## 2. Goals / non-goals

### Goals
1. A new, explicit **NEST-flavored network API** in `brainpy_state/_network`
   (`Simulator` object; `create` / `connect` / `simulate`; `NodeView` set
   algebra; named connection rules; weight + delay). No global kernel; JAX-clean.
2. A faithful **`brunel_alpha` port** in `examples/nest/` driving the real
   `iaf_psc_alpha` / `poisson_generator` / `spike_recorder` models.
3. A **validation harness** asserting firing-rate parity with live NEST within
   5 % (statistical, not trajectory-exact), skipped when NEST is unavailable.
4. **Bug fixes in `brainpy_state/_nest`** for every discrepancy the port
   surfaces, each preceded by a failing unit test (project rule 5).

### Non-goals (this cycle)
- The faithful global-kernel PyNEST facade (`import nest as` drop-in). Rejected
  in favor of the explicit object API.
- The other four Brunel variants (`delta`, `exp_multisynapse`, `siegert`,
  `evolution_strategies`) — documented as later phases (§9), not built now.
- Spatial / CSA / `CopyModel` registry / `GetConnections` / file recording
  backends.
- Any change to the existing brainpy-style `Network` / `Builder` / `*Proj`
  classes or the `examples/1xx_*` / `examples/brunel.py` gallery. They are left
  untouched.

## 3. The target API (explicit `Simulator`)

```python
from brainpy.state.network import Simulator, fixed_indegree, all_to_all
from brainpy.state import iaf_psc_alpha, poisson_generator, spike_recorder
import saiunit as u

sim   = Simulator(dt=0.1 * u.ms)
ne    = sim.create(iaf_psc_alpha, NE, params=neuron_params)   # NodeView
ni    = sim.create(iaf_psc_alpha, NI, params=neuron_params)
noise = sim.create(poisson_generator, rate=p_rate * u.Hz)
esr   = sim.create(spike_recorder)

sim.connect(noise, ne,        weight=J_ex, delay=1.5*u.ms, rule=all_to_all)
sim.connect(ne, ne + ni,      weight=J_ex, delay=1.5*u.ms, rule=fixed_indegree(CE))
sim.connect(ni, ne + ni,      weight=J_in, delay=1.5*u.ms, rule=fixed_indegree(CI))
sim.connect(ne[:N_rec], esr)
res = sim.simulate(1000 * u.ms)
print('exc rate =', res.rate(esr), 'Hz')
```

NEST vocabulary preserved: `create` (≈ `Create`), `connect` (≈ `Connect` with
`rule`/`weight`/`delay` in place of `conn_spec`/`syn_spec`), `NodeView` `+` and
slicing (≈ `NodeCollection` algebra), `simulate` (≈ `Simulate`). Differences from
PyNEST: real model classes instead of string names; explicit `Simulator` instead
of a global kernel; `saiunit` quantities on parameters.

## 4. Architecture

`Simulator` builds a `brainstate` module graph and runs **one**
`brainstate.transform.for_loop` over a step function. Per-step order matches the
brainstate convention already used by `Network.update` (projections before
dynamics):

```
step(t, i):
  1. projections: read pre spike (delayed), scatter weighted-pA events
                  into post.add_delta_input(...)
  2. populations: spk = pop.update();  capture hard(spk) into Simulator state
  3. devices:     generators emit; recorded spikes are returned from the loop
                  as a stacked array (see §5)
```

### 4.1 Reuse map (no reinvention)

| Need | Reused primitive | Location |
|---|---|---|
| Synaptic delay (homogeneous + axonal, sub-dt) | `InputDelay` seam (`InputDelay(in_size, delay)`, `.update(x)`) | `brainpy_state/_brainpy/_delay.py` |
| Event routing into a current-based neuron | `add_delta_input` / `sum_delta_inputs` (pA, sign-split ex/in inside `iaf_psc_alpha`) | `_base.py`, `_nest/iaf_psc_alpha.py` |
| Connectivity sampling | `sample_fixed_indegree`, `sample_all_to_all`, … + `resolve_param` | `_network/_connectivity.py` |
| Delta-projection shape | `DeltaProj.update` pattern (`post.add_delta_input(name, comm(x))`) | `_brainpy/projection.py` |

### 4.2 New components (all in `brainpy_state/_network`)

- **`_simulator.py` — `Simulator`, `SimulationResult`.**
  - `create(model_cls, size=1, *, params=None, **kw)` → instantiates the model,
    registers it, returns a `NodeView` over it. `params` is a NEST-style dict
    (unit-bearing or mapped to the model's units).
  - `connect(pre, post, *, rule=all_to_all, weight=None, delay=None, ...)` →
    builds an event projection (§4.3) and registers it. `pre`/`post` are
    `NodeView`s. A `spike_recorder` target is wired as a recording tap, not a
    synapse.
  - `simulate(duration, *, dt=None)` → `init_all_states`, runs the for_loop,
    returns `SimulationResult`.
  - `SimulationResult.rate(node)`, `.n_events(node)`, `.spikes(node)`,
    `.raster(node)` — computed from the stacked per-step spike arrays; may also
    populate the real `spike_recorder` device post-hoc for fidelity.

- **`_nodeview.py` — `NodeView`.** Holds `(population, index_array)`. `a + b`
  concatenates index ranges over a shared/aggregate population view; `a[sl]`
  slices. Connection rules consume `NodeView.indices` so `ne[:N_rec]` and
  `ne + ni` work like `NodeCollection`. Sizing for `_size()` comes from the
  index array length.

- **`_rules.py` — connection rules as values.** `all_to_all`,
  `one_to_one`, `fixed_indegree(K)`, `fixed_outdegree(K)`,
  `pairwise_bernoulli(p)`, … Thin callables/objects wrapping the existing
  samplers; each yields a `ConnSpec` over the pre/post index sets.

- **`_event_proj.py` — delta+delay event projection.** The piece the current
  `_RuleProj` lacks. Given a `ConnSpec`, a per-edge `weight` (pA), and a
  `delay`: builds a weighted sparse/dense connection (reusing the
  `_RuleProj` weight-matrix scatter), wraps the **pre spike** in an
  `InputDelay` seam, and routes `comm(delayed_spike)` into
  `post.add_delta_input(name, …)`. No conductance `AlignPostProj`, no brainpy
  `Expon`/`COBA` — the alpha shaping lives inside `iaf_psc_alpha`.

### 4.3 Spike capture — **approach (A), Simulator-managed**

`iaf_psc_alpha.update()` returns a (surrogate) spike but persists no `.spike`
State, and projections must read the **pre** spike (delayed ≥ 1 step) *before*
neurons integrate. Decision: the **Simulator** owns a per-population
`ShortTermState` holding the latest hard spike (`0/1`, or integer multiplicity
for generators); the step function writes it after each `pop.update()`, and
event projections read it as their pre-input (then `InputDelay` applies the
remaining delay so total emission→delivery = `round(delay/dt)` steps, NEST
min-delay = 1 step).

Rejected: adding `self.spike` to every NEST neuron (churns 100+ models for
plumbing). Approach (A) keeps `_nest` edits limited to genuine numerical fixes.

### 4.4 `poisson_generator` fan-out semantics

In NEST a single `poisson_generator` node sends an **independent** train to each
target, so `nest.Connect(noise, nodes_ex, …)` (default `all_to_all`) delivers
`NE` independent realisations. The brainpy `poisson_generator(in_size=N)` already
produces `N` independent trains. Therefore, when the `pre` of a `connect` is a
generator device, the `Simulator` **realises `all_to_all` (or the default) as `N`
independent per-target trains** — the generator's effective train count is sized
from the target population at connect/build time and wired one-to-one (× `weight`,
+ `delay`) — rather than broadcasting one shared train. The user-facing call stays
NEST-faithful (`rule=all_to_all`); the Simulator does the right thing underneath.
Verifying this matches NEST is an explicit validation point (§7).

## 5. Recording & the JIT boundary

`spike_recorder.update()` mutates Python lists ⇒ **not** traceable inside the
jitted `for_loop`. Therefore recorded populations are collected as a **stacked
JAX array** returned from the loop (the existing `Network.simulate` monitor
mechanism already does this), and `SimulationResult` computes
rate / n_events / raster from it; the real `spike_recorder` device may be
populated post-loop for API fidelity. The recording tap respects the recorder's
`start`/`stop`/`origin` window and the `N_rec` slice.

## 6. Example & validation harness

- **`examples/nest/brunel_alpha.py`** — full `order = 2500` script
  (`NE = 4*order`, `NI = order`, `epsilon = 0.1`, `g = 5`, `eta = 2`,
  `delay = 1.5 ms`, `J = 0.1 mV`, `tauSyn = 0.5 ms`, `tauMem = 20 ms`,
  `CMem = 250 pF`, `theta = 20 mV`, `t_ref = 2 ms`, `E_L = V_reset = V_m = 0`).
  Computes `J_ex` via the same PSP normalisation (`ComputePSPnorm` /
  Lambert-W) as upstream, `J_in = -g·J_ex`, `p_rate = 1000·η·ν_th·CE`. Reads
  like the §3 snippet; prints exc/inh rates; draws a raster.
- **`brainpy_state/_nest/_validation/brunel_alpha_test.py`** —
  `unittest.TestCase`, small `order` (~200 ⇒ 1000 neurons) for CI speed. Builds
  the same network in **live NEST** and in brainpy.state, asserts **mean
  excitatory firing-rate parity within 5 %** over a 1 s window and a sane
  CV-of-ISI. Guarded by `@unittest.skipUnless(<nest importable>, …)` so the
  no-NEST CI stays green. Parity is **statistical** — RNG realisations differ
  between NEST and JAX, so we compare population statistics, never per-neuron
  trajectories.

## 7. Model-fix workflow (`brainpy_state/_nest`)

Every discrepancy the port surfaces is fixed **failing-test-first** (project
rule 5): a focused unit test in the model's colocated `*_test.py` reproduces the
NEST-vs-brainpy gap, then the fix lands. Prime suspects the alpha port probes:

1. **`poisson_generator`** — independent-train fan-out (§4.4); `lam = r·dt/1000`
   bin-count vs NEST; start-exclusive/stop-inclusive window.
2. **`iaf_psc_alpha`** — weight-in-pA semantics, PSP normalisation parity
   (`J_unit`), alpha propagator `P31/P32` near `tau_m ≈ tau_syn`, delay
   quantisation, one-step current buffering (`y0`).
3. **`spike_recorder`** — stacked-array ingestion path; stamp step `n+1`;
   window gating; `n_events`.
4. **static-synapse weight/delay** carrier semantics as realised by the event
   projection (weight × multiplicity, integer-step delay ≥ 1).

If a discrepancy turns out to be a *network-layer* bug (projection/delay/capture)
rather than a model bug, it is fixed in `_network` instead — but the failing
test still comes first.

## 8. File layout

```
brainpy_state/_network/
  _simulator.py        # NEW — Simulator, SimulationResult
  _nodeview.py         # NEW — NodeView (+ / slicing)
  _rules.py            # NEW — all_to_all, fixed_indegree(K), ...
  _event_proj.py       # NEW — delta + delay event projection
  __init__.py          # export the above (additive; existing exports kept)
  _connectivity.py     # reused unchanged
  _base.py / _builder.py / _projections.py / _recorders.py   # untouched
examples/nest/
  brunel_alpha.py      # NEW
brainpy_state/_nest/_validation/
  __init__.py          # NEW (if absent)
  brunel_alpha_test.py # NEW — live-NEST parity, skip-if-unavailable
brainpy_state/_nest/*  # surgical model fixes as surfaced, each with a failing test first
```

Top-level re-export: `brainpy.state.network` exposes `Simulator` and the rules;
the models are already exported from `brainpy.state`. `changelog.md` updated on
completion (project convention 2).

## 9. Phases (build Phase 1 now)

1. **Phase 1 (this cycle):** `Simulator` foundation + `brunel_alpha` port +
   validation harness + alpha-family model fixes.
2. **Phase 2:** `brunel_delta` (`iaf_psc_delta`).
3. **Phase 3:** `brunel_exp_multisynapse` (multi-receptor ports).
4. **Phase 4:** `brunel_siegert` (`siegert_neuron`, mean-field rate).
5. **Phase 5 (optional):** `brunel_alpha_evolution_strategies` (ES optimiser on
   top of the alpha network).

## 10. Risks

- **Spike-timing edge cases.** NEST schedules on a global ring buffer with
  min-delay slicing; brainpy.state evaluates per-step via `InputDelay`.
  Semantically equivalent at matched `dt` + grid-aligned delays, but the
  emission→delivery step count must be verified to equal `round(delay/dt)`
  exactly (off-by-one is the classic failure). The validation harness is the
  guard.
- **PSP normalisation.** The `J_ex` calibration must match NEST's
  `ComputePSPnorm` exactly, or rates diverge well beyond 5 %.
- **Generator fan-out** (§4.4) — wrong train-sharing would change the input
  drive and the rate.
- **Recorder JIT boundary** (§5) — must not force the whole loop eager.

## 11. Acceptance criteria

- `examples/nest/brunel_alpha.py` runs, prints exc/inh rates, writes a raster.
- `brunel_alpha_test.py` passes against live NEST 3.9: mean exc firing-rate
  within 5 % over 1 s; skipped cleanly where NEST is absent.
- Every `_nest` fix has a preceding failing test now passing.
- No change to existing brainpy-style network classes / gallery; full
  `pytest brainpy_state/` stays green.
