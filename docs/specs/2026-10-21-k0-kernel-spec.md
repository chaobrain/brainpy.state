# K0 — kernel spike: specification

_Date: 2026-10-21_ · Status: **decisions recorded (§10); ready for T0** ·
Parent: [`2026-10-21-architecture-review-and-redesign.md`](2026-10-21-architecture-review-and-redesign.md)
(§7 layers, §7.1 timing, §7.2 RNG, §8 learning, §12 K0 row).

K0 builds the smallest kernel that exercises every risky contract of the redesign, and
ends in a **go/no-go decision** on the design. It is not a feature release: nothing is
exported to `brainpy.state` in K0.

---

## 1. Facts this spec rests on (verified 2026-10-21)

Probed in `/tmp/k0_probe/` and by code reading of the current tree.

| fact | consequence for K0 |
|---|---|
| `BinaryArray(x) @ CSR` binarises (`x=[0,2,1]` → counts 2 treated as 1); exact counts need a plain `x @ CSR`, which does not skip zeros | event payload has two paths (§4.3) |
| Under `BinaryArray`, the weight gradient for CSR / dense / JITC uses the **raw** `x` (CSR gives `[0,0,2,1]`, correct is `[0,0,1,1]`) | K0 never feeds counts > 1 to `BinaryArray`; upstream bug report (§9) |
| `vmap` of `binary_csrmv` lowers to `binary_csrmm` (real batching rule) | batched trials are event-driven |
| `update_csr_on_binary_{pre,post}` = "add trace, clip"; not differentiable (`NotImplementedError` under `grad`); no per-edge state | STDP fast path is forward-only; a JAX reference path serves gradients (§4.6) |
| `csr_to_csc_index` gives a CSC transpose; `csrmv_dt2t` gathers neuron values to edges | post-spike-driven column updates and per-edge gathers are available |
| No per-edge-delay primitive in brainevent | K0 implements delay buffers in JAX (§4.4) |
| brainevent indices int32 by default; `indptr_dtype=jnp.int64` keeps indptr int64 under x64 | index-dtype policy (§4.2) |
| `jax.grad` through `while_loop` raises | fixed-substep integrators only (I1) |
| braintrace 0.2.6: online learning works with `ParamState` weights applied via `braintrace.matmul` / `sparse_matmul(x, w, sparse_mat=CSR)`; units supported; **the step output must be read from a `HiddenState` or all gradients are silently 0**; writing a `ParamState` inside the traced update raises; weights used inside a `fori_loop` lose their eligibility trace | §4.7 braintrace compatibility rules |
| `iaf_psc_exp_ps`, `poisson_generator_ps` and all propagator coefficients are host NumPy (float64 only) | the precise pair is a **new JAX implementation**; propagators get a float32-safe JAX port |
| No benchmark harness exists; "~1.9 GB at order=2500" is text only | K0 first builds the baseline measurement for today's Simulator |
| No test differentiates through an LIF network | gradient checks are new |
| Brunel parity tests assert only the E rate, one realisation, order 200 | K0 adds the statistics of §6 G2 to the harness |
| NEST 3.9.0 and Brian2 2.10.1 importable (Brian2 installed 2026-10-21) | COBA gate G3 compares against Brian2; `brian2` becomes a validation extra |
| JAX 0.12 runs on CPU only: an NVIDIA GPU may be present but no CUDA jaxlib is installed | GPU benchmarks run on the remote RTX 3090 host (§10.3) |

---

## 2. Scope

**In:** the L1 contracts, `Network` graph, compiler plan, step schedule, delay buffers,
RNG contract, runner, observers, BPTT learner and braintrace compatibility, and these
models:

| model | role in K0 | numerics source |
|---|---|---|
| `iaf_psc_alpha` | exact propagator, `event` + `current` ports, Brunel | `_nest_neuron/iaf_psc_alpha.py:451-680`, `_nest_base/utils.py:126,177` |
| `LIF` (BrainPy) | surrogate-gradient workhorse | `_brainpy/lif.py:162-293` |
| `iaf_cond_exp` | `conductance` port, fixed-substep RK | `_nest_neuron/iaf_cond_exp.py:456-629` |
| `aeif_cond_alpha` | in-substep spikes, multi-state jump (`w += b`), stiffness | `_nest_neuron/aeif_cond_alpha.py:554-729` |
| `iaf_psc_exp_ps` | `timed_event` port — **new JAX implementation** | algorithm from `_nest_neuron/iaf_psc_exp_ps.py:503-881` |
| `poisson_generator_ps` | `timed_event` source — **new JAX implementation** | `_nest_device/poisson_generator_ps.py` |
| `poisson_generator`, `spike_generator` (with multiplicity), `dc_generator` | devices for gates | existing files |
| NMDA conductance with Mg block | voltage-gated `conductance` port, forward only | `_brainpy/synouts.py:206-301`, `_brainpy/synapse.py:377-548` |
| `Expon` (post-aligned, mergeable) | `SynapseModel` + merge pass | `_brainpy/exponential.py:34-123` |
| `static_synapse`, `stdp_synapse` | `PlasticityRule` | `_nest_plasticity/stdp_synapse.py:163-179` |

**Out (K1+):** `gap`, `modulatory` and `graded` emission beyond what the gates need;
STP; `procedural`/`conv` storage; sparse recording buffers; checkpoint/restore;
sharding; EventProp; NIR; the NEST-vocabulary `Simulator` layer; porting the rest of
the library.

---

## 3. Package layout

New private package, not imported by `brainpy_state/__init__.py`, no
`__module__ = 'brainpy.state'` on its classes (so nothing becomes public):

```
brainpy_state/_kernel/
  __init__.py          # internal exports only
  units.py             # dtype/unit policy helpers (float32/float64 tiers)
  context.py           # StepContext
  ports.py             # Port kinds, payload types, connect-time checks
  params.py            # Param, Params, States, Init (+ distributions)
  spiking.py           # SpikeRule, Clamp, Block, count_ste
  integrators.py       # Exact, ExpEuler, RK(order, substeps); in-substep event check
  propagators.py       # JAX, float32-safe ports of propagator_exp / alpha P31-P32
  model.py             # NeuronModel, Device, generated step()
  synapse.py           # SynapseModel (+ Expon), alignment, merge key
  plasticity.py        # PlasticityRule protocol, KernelContext, traces (lazy NEST timing)
  rng.py               # counter-based key derivation (seed, uid, gid, step)
  connectivity/
    rules.py           # all_to_all, one_to_one, fixed_indegree, fixed_total_number, pairwise_bernoulli
    edgeset.py         # EdgeSet over global ids
    storage.py         # Dense, CSR (forward, VJP, braintrace registration)
  delays.py            # pre-side ring, post-side arrival ring
  network.py           # Network, PopulationView (gid ranges, slicing, concatenation)
  compiler.py          # Plan: validation, schedule, storage, buffers, merges; Plan.describe()
  runner.py            # Runner: simulate / cont / step; vmap batching
  observers.py         # spikes, state traces (dense in K0)
  learn/
    bptt.py            # BPTT learner (+ checkpointed option)
    etrace.py          # braintrace compatibility checks
  models/              # K0 models listed in §2
  validation/          # parity + gate tests (requires_nest reused from _nest_validation)
benchmarks/            # new, repo root: baseline + K0 benchmark scripts and results
```

Tests are `<module>_test.py` next to each module, importing `brainpy_state._kernel`
directly (CLAUDE.md #9). Gate tests live in `_kernel/validation/`.

---

## 4. Contracts

### 4.1 Context, parameters, states

```python
@dataclass(frozen=True)
class StepContext:            # passed explicitly; models never read brainstate.environ
    t: Quantity               # time at step start
    dt: Quantity
    step: jax.Array           # int step index
    key: jax.Array            # per-population key for this step (§4.5)
```

- `Params(name=Param(default, unit, trainable=False, per_neuron=True, dist=None))`.
  All parameters are **traced arrays** held in `State`s: `ParamState` when trainable,
  otherwise a `ConstParamState(State)` subclass — so parameter sweeps and `vmap` never
  recompile, and heterogeneity is the default.
- `States(name=Init(value | distribution, unit))` → `HiddenState`s.
- Derived constants (propagators, `ref_count`) are computed **inside `step` from traced
  parameters and `ctx.dt`**, with float32-safe singular fallbacks, never at construction.

### 4.2 Ports and payloads

```python
Event(max_multiplicity=1)          # payload: float array (n,) = count × spike
TimedEvent(capacity=K)             # payload: offsets (n,K), weights (n,K), valid (n,K), overflow counter
Current(unit=u.pA)                 # summed, latched for the next step
Conductance(unit=u.nS, E_rev, gate=None)   # gate(V) evaluated inside derivatives at every substep
Graded(unit)                       # K0: only as needed by gates
```

Connect-time validation: kind compatibility, unit dimension, port existence, single
owner of synaptic kinetics, signed weight distributions, `TimedEvent` capacity.

### 4.3 Neuron model

```python
class NeuronModel(brainstate.nn.Module):
    params: Params; states: States; inputs: Ports; outputs: Ports
    spiking: SpikeRule | None
    integrator: Exact | ExpEuler | RK
    capabilities = Capabilities(jit_safe=True, differentiable=True, etrace=True)

    def derivatives(self, s, p, inp, ctx) -> dict: ...    # continuous part
    def on_input(self, s, p, inp, ctx) -> s: ...          # jumps (applied per §7.1 timing)
    def step(self, s, p, inp, ctx) -> (s, outputs): ...   # generated; Exact models override
```

Generated `step` (per §7.1): latch currents from the previous step → integrate
`derivatives` with `RK(order, substeps)` (Python-unrolled substeps; synaptic port
inputs are computed **once, outside** the substep loop — braintrace rule) → check
`SpikeRule.condition` after every substep, apply `jump` and refractory clamp in-substep,
**count** spikes → apply `event` jumps after propagation → write outputs from the new
`HiddenState` values (braintrace rule).

`SpikeRule(condition, jump, refractory=Clamp(var, t_ref) | Block(t_ref) | None,
surrogate=braintools.surrogate.*, detach_reset=None)`; `detach_reset` defaults from the
learner. Output spike = `count_ste(count, surrogate(v_scaled))` where
`count_ste = stop_gradient(c - s) + s`.

**Event payload paths.** Sources declaring `max_multiplicity=1` use the event-driven
`BinaryArray` path; sources that can emit more (generators, `aeif` with several
in-step spikes) use the exact float path. The compiler picks per projection and
reports it in the plan. Gradients w.r.t. weights are taken from the float path's VJP,
never from `BinaryArray` with counts > 1 (§1).

### 4.4 Connectivity, storage, delays

- `rule.build(pre_gids, post_gids, key) -> EdgeSet(pre, post, n_pre, n_post)`:
  vectorised, autapse/multapse on **global ids**, NEST-exact counts;
  `fixed_total_number` mirrors NEST (refuses `allow_multapses=False`). Concatenated
  views are sampled as one population (fixes doubled indegree).
- Storage `Dense` and `CSR`; both implement `forward(payload, w)`, a VJP, and
  braintrace routing (`braintrace.matmul` / `braintrace.sparse_matmul`). Index dtype
  int32, int64 indptr above 2³¹ edges (x64 required; checked at compile).
- `EdgeSet` retained on device; `get_connections` / `set('weight')` operate on it.
- Delays: integer steps for `event` (rounded, NEST); pre-side ring for scalar or
  per-source delays; post-side arrival ring for per-edge delays; zero delay allowed on
  acyclic paths (compiler topologically orders them; zero-delay cycles are errors).

### 4.5 RNG contract

`key(seed, uid, gid, step) = fold_in(fold_in(fold_in(root(seed), uid), gid_block), step)`
where `uid` is the object's **insertion index** in the `Network` (deterministic, not
`id()`). Connectivity is generated per `(projection uid, post gid)` block, so the
realised `EdgeSet` is independent of storage and batch size. Under `vmap` the user
selects `independent` or `common` random numbers per trial.

### 4.6 Plasticity

`PlasticityRule` keeps today's protocol (`edge_state_init`, `update(state, ctx) ->
(state, w_eff)`, declared traces/modes/reads). K0 implements NEST **lazy** timing with
the dendritic/axonal delay split (today's substrate lacks the dendritic seam and the
parity drive fakes it, `_nest_validation/_stdp_drive.py:20-53`). Two paths: a fast
forward path via `update_csr_on_binary_{pre,post}` when the rule fits "add trace,
clip"; a JAX reference path over the `(nse,)` edge arrays that is differentiable
(learned STDP parameters, hybrid weights in through-the-rule mode).

### 4.7 Learning

- `BPTT(loss, optimizer, checkpoint=None, truncation=None, detach_reset=True)` over a
  `Runner`; uses `brainstate.transform.for_loop` / `checkpointed_for_loop`.
- braintrace compatibility (checked by `learn/etrace.py` at compile, clear errors):
  weights are `ParamState`s applied through braintrace primitives; port inputs computed
  outside substep loops; outputs read from `HiddenState`; no `ParamState` writes inside
  the traced step (so local plasticity + `Online` is rejected, per L4).

### 4.8 Network, compiler, runner

```python
net = Network(dt=0.1 * u.ms, seed=0)
E = net.add(iaf_psc_alpha, 800, params=dict(...))         # PopulationView, gids [0, 800)
noise = net.add(poisson_generator, 1, params=dict(rate=...))
net.connect(E, E, rule=fixed_indegree(80), port='in', weight=..., delay=1.5 * u.ms,
            storage='auto', plasticity=None)
rec = net.observe(E[:50], 'spike', 'V_m', interval=None)
plan = net.compile(precision='float32')                    # validates; plan.describe()
run = plan.runner(batch=None)
res = run.simulate(1000 * u.ms); res2 = run.cont(500 * u.ms)
```

`plan.describe()` prints the schedule, storages, payload paths, buffer placement and
memory, merges, and the braintrace/learner capability summary.

---

## 5. Work breakdown (order of implementation)

| # | task | depends |
|---|---|---|
| T0 | `benchmarks/` baseline: today's Simulator, Brunel order 200 and 2500, float32/64, compile time, step time, peak memory; CPU and GPU 1 on the RTX 3090 host (§10.3) | — |
| T1 | context, params, states, ports, units policy | — |
| T2 | integrators + propagators (float32-safe) + SpikeRule | T1 |
| T3 | models: LIF, `iaf_psc_alpha`, `iaf_cond_exp`, `aeif_cond_alpha` | T2 |
| T4 | rng, rules, EdgeSet, Dense/CSR storage with VJP + braintrace routing | T1 |
| T5 | delays (pre-/post-side), network, compiler, runner, observers | T3, T4 |
| T6 | Expon synapse + merge pass; static + stdp (lazy timing, both paths) | T5 |
| T7 | `timed_event`: JAX `poisson_generator_ps`, `iaf_psc_exp_ps` | T5 |
| T8 | NMDA/Mg-gated conductance port | T5 |
| T9 | learners: BPTT, braintrace checks | T5 |
| T10 | gate suite (§6) + benchmarks; `requires_brian2` marker and `brian2` validation extra; K0 report | all |

Each task: test-first, NumPy docstrings, >90 % coverage of `_kernel` (CLAUDE.md #8).

---

## 6. Go/no-go gates

All run in CI except those marked *(NEST)* (run in the NEST workflow), *(Brian2)* (`requires_brian2` marker, skipped when absent) and *(bench)*.

| gate | test | pass criterion |
|---|---|---|
| G1 timing *(NEST)* | per port kind: single-spike PSP (`event`, `iaf_psc_alpha`), DC step onset (`current`), conductance PSP (`iaf_cond_exp`), precise input (`timed_event`, `iaf_psc_exp_ps`) | traces match NEST with **`align_steps=0`**; exact models `CAT_B` (1e-6); precise model spike times within 1e-6 ms |
| G2 Brunel *(NEST)* | `iaf_psc_alpha` Brunel order 200, 1000 ms, 10 seeds each side | E and I rate, CV-ISI, Fano factor (10 ms bins), mean pairwise spike-count correlation (50 neurons): \|Δmean\| ≤ max(5 % of NEST mean, 2 × combined SEM) |
| G3 COBA *(Brian2)* | COBA E/I (Vogels–Abbott, 4000 neurons, 1000 ms) with `iaf_cond_exp`-equivalent dynamics, 10 seeds each side, vs the Brian2 reference; plus single-neuron conductance traces from identical input spikes | traces `CAT_A`; network statistics by the G2 rule |
| G4 invariance | realised `EdgeSet` identical for `Dense` vs `CSR`, batch 1 vs 4 (trial 0), `independent` vs `common` RNG semantics as specified; goal-01 repros (global-id autapses on slices/overlap/concatenation, zero-weight edges in `get_connections`, `set('weight')` alignment, `fixed_total_number`, `spike_generator` multiplicity end to end) | exact equality |
| G5 integrators *(NEST)* | `aeif_cond_alpha` and `iaf_cond_exp` vs NEST at substeps {1,2,4,8,16} | `iaf_cond_exp` and subthreshold traces: error decreases at the integrator's order; default substeps chosen to meet `CAT_A` (1e-3 mV). Spiking `aeif_cond_alpha`: spike times equal NEST (same step) at the default substeps, and `V_m` within `CAT_A` on every step except where either trace is above `V_th` (the exponential upswing) and the step after (the reset). Table recorded. (Decision after T3: fixed substeps localise the threshold crossing only to first order, so the full-trace error on spike steps stays large -- measured 2.26 mV at 16 and 0.87 mV at 64 substeps; see [`2026-10-21-k0-t3-models.md`](2026-10-21-k0-t3-models.md).) |
| G6 gradients | BPTT on 20-neuron LIF and `iaf_psc_alpha` nets, Dense and CSR, delay 0 and 2 steps, float64, smooth surrogate, in the two regimes where BPTT is the exact derivative of the simulated loss: (a) hard forward, feed-forward net, `detach_reset=True`, membrane loss; (b) recurrent nets under `relaxed_spikes()` (smooth spikes, soft jumps, non-detached reset) | relative error vs central finite differences ≤ 1e-4 |
| G7 braintrace | `braintrace.compile(net, 'eprop')` and `'d_rtrl'` on a K0 LIF net, Dense and CSR | compiles; gradients non-zero for every weight; the K0 checker rejects a deliberately non-compliant model with a clear error |
| G8 STDP *(NEST)* | `stdp_synapse` weight trajectory, pre/post pairing protocol with dendritic delay > 0 | `CAT_B` without the current drive workaround |
| G9 performance *(bench)* | Brunel order 2500, float32 and float64, vs T0 baseline, same host (§10.3), on CPU and on RTX 3090 GPU 1 | step time ≤ 1.1 × baseline; peak memory ≤ 1.1 × baseline; compile time reported |
| G10 quality | coverage of `_kernel` | > 90 % |

(Decision after T9, accepted by the orchestrator: with a hard forward pass and a surrogate
backward pass the BPTT gradient is not the derivative of the simulated loss whenever a
gradient crosses a spike, so a literal FD check fails by design (relative errors 3–500
measured on recurrent nets); G6 is checked in regimes (a) and (b) above. See
[`2026-10-22-k0-t9-learners.md`](2026-10-22-k0-t9-learners.md) §2.)

**Decision rule.** All gates pass → go (K1 starts). G9 fails by > 10 % or any of G1,
G4, G6 fails structurally → stop and revise the architecture doc before K1.

---

## 7. Edge cases the tests must cover

Zero-size populations; slices that overlap partially; concatenated views on both
sides; zero-delay acyclic chains vs zero-delay cycles (error); delays not multiple of
`dt` (rounded for `event`); `TimedEvent` capacity overflow (counter increments, no
silent drop); several spikes in one step from `aeif`; refractory period shorter than
`dt`; parameters given as distributions with unit-aware clipping; float32 vs float64
propagator fallbacks near `tau_m == tau_syn`; `cont()` after `simulate()` continues RNG
streams; batched trials with `common` vs `independent` RNG.

---

## 8. Risks specific to K0

- **Precise models are new code**, not ports: the JAX `iaf_psc_exp_ps` needs a
  vectorised, fixed-iteration root finder and sorted per-step event lists. Largest
  single task; bounded by keeping `poisson_generator_ps` + `spike_generator` as the
  only `timed_event` sources.
- **Benchmarks run remotely** (§10.3): the remote checkout must match the commit being
  measured; every benchmark result records the commit, host, device and package
  versions.
- **brainevent gradient bug** (§1): mitigated by the payload-path rule; reported
  upstream.

---

## 9. Upstream items (proposals, not built here)

| repo | item | type |
|---|---|---|
| brainevent | weight gradient under `BinaryArray` uses raw `x` (CSR, dense, JITC) | bug report with repro |
| brainevent | event-driven matmul for count-valued events with VJPs | feature proposal |
| brainevent | differentiable `update_csr_on_binary_*`, or a generic per-edge rule kernel with arbitrary edge state | feature proposal |
| brainevent | per-edge-delay delivery fused with the matmul | feature proposal |
| braintrace | `fori_loop` bodies containing weights lose eligibility traces; zero gradients without warning when the output is not read from a `HiddenState` | issue (diagnostics) |
| braintrace | deprecate `braintrace.nn` neuron/synapse duplicates in favour of brainpy.state (decision L3) | proposal |
| braintrace | 0.2.6: `KeyError` in `braintrace.compile` when a `HiddenState` depends only on inputs (no hidden-to-hidden path), found in T2 | bug report with repro |
| brainevent | 0.2.1: GPU float64 abort in `binary_csrmv_wat_hybrid_hetero_f64_float` (default backend; `cusparse` works), found in T0 | bug report with repro |
| brainevent | `csr_to_csc_index` casts int64 indices to int32 with a `FutureWarning`, found in T6 | issue |

**Existing-code bugs found during K0** (outside `_kernel`; not fixed in K0, to be
filed or fixed separately):

| where | bug | found in |
|---|---|---|
| old NumPy `alpha_propagator_p31_p32` | when `exp(-h / tau_syn)` underflows (very small `tau_syn`, e.g. `tau_syn <~ h / 700` in float64) `P31` is NaN and `P32` takes a singular fallback that is not the limit of the exact propagator | T2 |
| old `poisson_generator` | fails with the default `stop = inf` when x64 is off (the `int64` max overflows the step comparison) | T0 |
| `Simulator.cont` | documents "no recompile" but re-traces and re-compiles its `for_loop` on every call | T0 |

---

## 10. Decisions (2026-10-21)

1. **Package location**: `brainpy_state/_kernel/` (private until K3); benchmarks in a
   new top-level `benchmarks/`.
2. **Precise timing**: both `iaf_psc_exp_ps` and `poisson_generator_ps` are built in
   JAX in K0. The `timed_event` contract is one of the risks K0 must retire, and gate G1
   needs a real receiving model to test it.
3. **Hardware for G9**: all benchmarks, CPU and GPU, run on
   `wanglab3090b-01a0.chaobrain` (Intel i7-14700KF, 28 threads, 188 GB RAM;
   2 × RTX 3090 24 GB, driver 580.178, CUDA 13.0; Python 3.12.3, `uv`). CPU runs set
   `JAX_PLATFORMS=cpu`, first check that the 1-minute load average is below 2 (abort
   otherwise), and record it with each result. Code lives under `~/data/brainpy-state-k0/` (a checkout of
   the K0 branch with a `uv` venv using a CUDA 13 jaxlib). Runs use **GPU 1 only**
   (`CUDA_VISIBLE_DEVICES=1`), and every run first checks that GPU 1 has no compute
   processes (`nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader` is
   empty) and aborts otherwise. NEST for remote parity runs comes from the conda env
   `brainpy-state-k0` (conda-forge `nest-simulator`, via the Tsinghua mirror: direct
   conda.anaconda.org access is reset from that host) NEST there is **3.10.0**; locally it is 3.9.0. Parity
   results record the NEST version, and a gate counts only when it passes on the version
   it was run against. Baseline (T0) and K0 are measured on the same host and
   GPU, in float32 and float64.
4. **Rate neurons (S1a)**: single-unit rate neurons are not supported (see
   architecture doc §11). The `graded` port stays for NMDA emission and SIC.
