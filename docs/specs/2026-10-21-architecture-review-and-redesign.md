# Architecture review and kernel redesign

_Date: 2026-10-21_ · Status: **direction agreed (D1, D2, D4, D5, L1–L4 decided 2026-10-21); expert review pending; K0 spec next** ·
Breaking changes: **allowed** (no 0.1 compatibility constraint).

Supersedes §2 ("Two families, one kernel") and Phases 1–4 of
[`2026-10-21-roadmap.md`](2026-10-21-roadmap.md). Phase 0 bug findings
([`2026-10-21-v011-goal-01-correctness.md`](2026-10-21-v011-goal-01-correctness.md)) are
folded in as acceptance tests (§7).

Evidence base: four code deep-dives (BrainPy-style kernel, NEST kernel, the unmerged
`spec` branch, goal-01 bug clusters verified against live NEST 3.9) and a survey of
NEST/NESTML, Brian2, GeNN 5, NEST GPU, Arbor, Lava, Rockpool, Norse/snnTorch/Spyx,
jaxley, BrainPy 2.x, NIR, SONATA, PyNN (sources at the end).

---

## Part I — Review of the current architecture

### 1. What exists

Two model families on one runtime (brainstate `Module`/`State` + JAX), with **three**
network entry points:

| entry point | lives in | wiring | runs NEST models? |
|---|---|---|---|
| bare BrainPy style (`AlignPostProj`, `CurrentProj`, …) | `_brainpy/projection.py` | projections mutate `post`'s hook/input dicts at construction | partly |
| `Network` / `Builder` | `_nest_network/base.py`, `builder.py`, `projections.py` | dense-only `_RuleProj` → `AlignPostProj`; reads `pre.spike` | **no** (NEST models store no `spike`) |
| `Simulator` (NEST vocabulary) | `_nest_network/simulator.py` (1883 lines) | flat attributes keyed by `id()` | yes |

`Builder`/`Network` duplicates the Simulator's size helpers, `for_loop` driver,
recorders and samplers while interoperating with neither the Simulator nor NEST models.

### 2. Execution model today

**BrainPy style** — one step, order implicit (projections must run before their post or
their input silently becomes a one-step delay):

```
spk = pre.get_spike()                         # recomputed from V[t-1]; spikes are not State
proj.update(spk): [InputDelay] -> comm -> shared_syn.add_delta_input(name, W·spk)   # Python dict
post(x):  before-hooks: g = syn.update(); out.bind_cond(g)       # plain attribute, not State
          update: reset(spike(V[t-1])); V = exp_euler(dv) where dv calls sum_current_inputs(x, v)
                  V = sum_delta_inputs(V)                         # array inputs are POPPED
```

**NEST Simulator** — one compiled `for_loop`, explicit phases:

```
0 volume transmitters   1 EventProjection/EventPlasticProj (prev-step holders -> InputDelay
-> gather -> comm|rule -> deposit)   1b current injectors   1c gap couplers
2 every neuron: m.update() [or update(w_by_rec=...)] -> holder   taps -> SimulationResult
```

Every edge carries an implicit one-step lag on top of its configured delay.

### 3. Where the abstraction leaks

1. **The kernel probes models instead of reading a contract.** `simulator.py` has ~60
   `isinstance`/`hasattr`/class-name/`inspect.signature` branches (device classification
   by substring of the class name, `_emission_attr`, `_relays_multiplicity`,
   `n_receptors`, `'w_by_rec' in signature`, `REQUIRES_SYMMETRIC`, `_IS_DIFFUSION`,
   `'rng_seed'` via signature, …). `connect` is a nine-way routing chain. Every new
   model family has needed a new seam (the design docs name seams A–I).
2. **No uniform neuron contract.** Six `update` signatures (`x`; `x, w_by_rec`;
   `x, spike_events` (host NumPy, not JIT-safe); `x, x_filtered, spike_events`;
   `spike_weights, J_ext`; `x, noise`) and five receptor-input conventions.
   Recordables resolve through an alias table plus five functions that reverse-engineer
   model storage (`_g_port`, `_asc_sum`, `_psc_sum`, …). State names disagree
   (`i_syn_ex`/`I_syn_ex`, `V` vs NEST `V_m`).
3. **Connectivity is not first-class.** There are no global neuron ids and no retained
   edge list: edges exist only as per-segment-pair samples or a dense `W`. Verified
   consequences (live NEST 3.9): autapse exclusion wrong on slices (20 vs 25 edges),
   overlapping slices (20 vs 22) and concatenated views (80 vs 90); concatenated views
   double indegree; dense `get_connections` drops zero-weight edges, merges multapses,
   and after `set('weight', …)` shifts weights onto the wrong edges.
4. **Events are binarised in places.** Sparse comm and `EventPlasticProj` use
   `BinaryArray`; `spike_generator` keeps only the last duplicate; `n_events` counts
   steps-with-spikes. Multiplicity is not a type, so it is lost silently.
5. **Side channels invisible to the state graph.** Inputs popped from Python dicts during
   tracing; `SynOut._conductance` as a plain attribute; global `environ` `t`/`dt` read
   deep inside models and frozen at construction in places (`LeakyRateReadout`,
   `ref_count`).
6. **Plasticity only in one family.** The NEST side has a good pure-rule substrate; the
   BrainPy side has none (`align_pre_ltp`/`align_post_ltp` are empty stubs) and hides
   weights inside `comm` objects.
7. **Host-only models in a JIT kernel.** Four `*_ps` models, `gif_pop_psc_exp`, several
   generators and the `spike_events` paths are NumPy loops that cannot run inside the
   Simulator's compiled loop, and nothing marks them.
8. **Dead or broken surface.** `align_pre_projection`/`align_post_projection` never worked
   (their own examples fail; `brainevent.MaskedFloat` no longer exists) and had no tests.
   Legacy host-object connections (`rate_connection_*`, `gap_junction` WFR objects) are
   unreferenced or only read for marker attributes.
9. **Dependency drift.** brainstate 0.6.0 removed the current/delta input API from
   `nn.Dynamics`; `_base.py` already re-implements it, so this package owns that half of
   the kernel whether it wants to or not.

### 4. What works and must be preserved

| keep | where today |
|---|---|
| Plasticity rule as a pure function `rule.update(edge_state, ctx) -> (edge_state, w_eff)` with declared needs (traces/taus/modes, `post_state_reads`, `signal_reads`, `stochastic`, `fractional_delay`). It absorbed every NEST rule, incl. dopamine and Urbanczik, with no substrate change. | `_nest_network/event_plastic.py`, `_nest_plasticity/` |
| NEST-faithful numerics: exact propagators with NEST's singular fallbacks; `AdaptiveRungeKuttaStep` (RKF45 + in-substep `event_fn`). | `_nest_base/utils.py` |
| Live-NEST parity harness (`nest_compare`, trace/distributional modes, `align_steps`), 131 validation test files. | `_nest_validation/` |
| O(N_post) merged post-aligned synapses (the ICLR-2024 AlignPost result). | `_brainpy/projection.py` |
| Event-driven sparse kernels (brainevent), procedural connectivity (`JITC*`). | comm objects |
| End-to-end unit safety (brainunit). Surrogate spikes; `stop_gradient` hard reset. | everywhere |
| One compiled `for_loop` per run; `cont()` + host drive for closed loops. | `simulator.py` |
| `InputDelay`: scalar / per-pre / per-edge delays, sub-dt interpolation. | `_brainpy/delay.py` |
| NEST user surface: `NodeView` algebra, lazy `SynapseCollection` with write-back, spatial layers/masks/kernels. | `_nest_network/`, `_nest_spatial/` |

### 5. The `spec` branch

A well-documented skeleton (~3k code lines + ~7k planning docs, all dated 2026-05-12),
not a kernel: ~70 % of the advertised surface raises `NotImplementedError`; synapse,
plasticity and network IR are never constructed; the working path strips units
(`V_th` in V vs `V` in mV compares wrong), runs a Python time loop, and the BPTT
backend leaks tracers. **Archive it as a tag; borrow ideas, not code**: backend-neutral
IR with `supports()` capability negotiation, linearity tagging routed to exact
propagators, explicit event priorities, structured diagnostics with source locations,
separate runtime vs training backends.

---

## Part II — Target architecture

### 6. Design principles

1. **One graph, one execution path.** Every front-end builds the same `Network` graph;
   only that graph runs. NEST parity therefore validates what users execute.
2. **Contracts, not probing.** Models *declare* ports, states, parameters, recordables
   and capabilities. The kernel never inspects class names, attributes or signatures.
3. **Connectivity is data.** Edge sets over global ids are first-class, retained (or
   procedurally regenerable), inspectable and writable.
4. **Explicit time and ordering.** `t`, `dt`, step index arrive in a context argument;
   the step schedule and delay semantics are a written contract.
5. **Functional inside, stateful outside.** Model `step` functions are pure in their
   inputs (no input dict popping, no attribute side channels); brainstate `State`s hold
   the carried values so jit / `for_loop` / grad / vmap compose.
6. **Description separate from execution.** Integrator, connectivity storage, delay
   implementation and gradient strategy are chosen at compile/run time, not hard-wired
   into models.

### 7. Layers

```
 L4  Front-ends       Composable Python API          NEST vocabulary (Simulator)     Interop (NIR, SONATA*)
                              \                              |                          /
 L3  Network graph    Network: Populations(gid ranges) + Projections + Devices + Observers
                      validate (ports/units/delays) · inspect · serialise
                              |
 L2  Compiler/runtime  plan: schedule, delay buffers, edge storage, merged synapses
                      Runner: run / cont / step · batching (vmap) · host loop
                      Learning: surrogate BPTT · checkpointed · online (braintrace) · plasticity
                              |
 L1  Model contracts  NeuronModel · SynapseModel · PlasticityRule · ConnRule/EdgeSet · Device · Observer
                              |
 L0  Substrate        brainstate (State, transforms) · brainunit · brainevent kernels
```

#### L1 — model contracts

**Ports.** Every connection goes from an output port to an input port. Port kinds:

| kind | payload | examples |
|---|---|---|
| `event` | spike multiplicity (int-valued, not binarised) and optional per-event weight | spike output; `spike_generator`; parrot relay |
| `current` | current (pA), summed | static PSC input, `dc_generator`, SIC |
| `conductance` | conductance increment (nS), routed per receptor | `iaf_cond_*`, multisynapse ports |
| `graded` | continuous value with declared unit | rate (Hz), `V` for gap junctions, NMDA emission |

Input ports are named (`'AMPA'`, `'GABA'`, `'rec_3'`); NEST's integer `receptor_type`
maps to names in the NEST front-end. Units and kinds are checked at `connect` time.

**`NeuronModel`** (a brainstate `Module`):

```python
class iaf_cond_exp(NeuronModel):
    params  = Params(C_m=250*u.pF, g_L=16.6*u.nS, E_ex=0*u.mV, ..., t_ref=2*u.ms)
    states  = States(V_m=Init(-70*u.mV), g_ex=Init(0*u.nS), g_in=Init(0*u.nS))
    inputs  = Ports(ex=Conductance(u.nS), in_=Conductance(u.nS), I=Current(u.pA))
    outputs = Ports(spike=Event())
    spiking = Threshold('V_m', V_th=-55*u.mV, reset=Hard(V_reset=-60*u.mV), refractory='t_ref')
    integrator = 'rkf45'            # or 'exact', 'exp_euler', 'euler'; overridable at compile

    def derivatives(self, s, inp, ctx): ...        # continuous part
    def on_input(self, s, inp, ctx): ...           # optional jump part (e.g. g += Δg)
```

- `step(state, inputs, ctx) -> (state, outputs)` is generated from `derivatives` +
  `on_input` + `spiking` + `integrator`; models with hand-written exact propagators
  override `step` directly. One shared `Threshold/Reset/Refractory` implementation
  replaces ~13 copied `get_spike` and refractory blocks (and fixes the `*Ref`
  soft-reset bug by construction).
- `recordables` are the declared states plus declared derived quantities — no alias
  table.
- `capabilities`: `jit_safe` (false for host-only models → compile error, not silent
  breakage), `differentiable`, `precise_spike_times`.

**`SynapseModel`** — dynamics between edge delivery and port deposit (Expon, Alpha,
DualExpon, AMPA/NMDA kinetics, STP). Declares its alignment: `post` (state per target
neuron, mergeable when linear with identical params — the AlignPost result becomes an
automatic compiler optimisation), `pre` (per source neuron), or `edge` (per edge).

**`PlasticityRule`** — the existing pure-rule protocol, unchanged in spirit, available
to every projection regardless of front-end: declares traces (pre/post, taus, modes),
state reads from the post population, broadcast signals, stochasticity; implements
`init_edge_state()` and `update(edge_state, ctx) -> (edge_state, w_eff)`. Per-edge
parameters are allowed (lifts today's scalar-only limit). Rule parameters may be
`ParamState`s, so plasticity itself is learnable.

**`ConnRule` → `EdgeSet`.** `rule.build(pre_gids, post_gids, key) -> EdgeSet(pre, post,
n_pre, n_post)` with autapse/multapse semantics on **global ids** and NEST-exact
counts. Storage strategy is a compile-time choice per projection: `dense`, `csr`,
`one_to_one`, `procedural` (regenerated from a key, never materialised), `conv`
(spatial Toeplitz). The edge list is retained (or regenerable), so introspection and
`set('weight')` are exact.

**`Device`** — generators are populations with no inputs and declared output ports
(`event`, `current`, `graded`); recorders are **observers**, not modules: they declare
what to sample and return arrays (no Python lists, no tap identities keyed by `id()`).

#### L3 — `Network` graph

```python
net = Network(dt=0.1*u.ms)                       # no global environ mutation
E = net.add(iaf_cond_exp, 3200, params=...)      # -> PopulationView (gid range)
I = net.add(iaf_cond_exp, 800)
net.connect(E, E + I, FixedProb(0.02), syn=Expon(5*u.ms), port='ex',
            weight=Normal(0.6*u.nS, 0.1*u.nS), delay=1.5*u.ms, plasticity=stdp(...))
rec = net.observe(E[:100], 'spike', 'V_m')
run = net.compile(storage='auto')                # plan; validates ports/units/delays
res = run.simulate(1000*u.ms)                    # one compiled for_loop
```

`Network` is a brainstate `Module`, so it nests inside training code. It is inspectable
(`net.populations`, `net.projections`, `net.get_connections(...)`) and serialisable to a
plain description (basis for NIR export).

#### L2 — schedule, delays, runtime

**Step contract** (written, tested, identical for every front-end):

```
for each step n:
  1. deliver   every projection reads source outputs emitted at step n - d (d >= 1 step),
               applies synapse/plasticity, deposits into target input ports
  2. update    every population: (state, outputs) = step(state, port_inputs, ctx)
  3. emit      outputs written to per-source delay buffers (one buffer per source port,
               sized max delay — shared by all projections from that source)
  4. observe   observers sample
```

- Minimum delay is one step, as in NEST; the hidden extra lag of today's Simulator
  disappears (delays mean what the user wrote). Sub-step delays use the existing
  interpolation (`fractional_delay`). Continuous same-step coupling (gap junctions,
  rate networks) is explicit-lag by default with an optional iterative (WFR) mode.
- Delay buffers are owned by the *source*, not per projection (today each projection
  keeps its own `InputDelay` over the full pre population).
- Port inputs are passed as a value into `step` — no dict popping.
- Runner: `simulate`, `cont`, `step`; host drive for closed loops; batching by `vmap`
  over the state pytree (trials, parameter sweeps); optional min-delay epoch chunking
  (NEST/Arbor model) for long runs.
- Learning: see §8 — learners are chosen per run over the same graph.

#### L4 — front-ends

- **Composable Python API** (default): `Network` above, plus model classes usable
  standalone inside any brainstate module for ML-style work.
- **NEST vocabulary**: `Simulator.create/connect/simulate`, NEST parameter names,
  integer `receptor_type`, `NodeView`, `SynapseCollection` — a thin layer that builds a
  `Network`. No routing logic of its own.
- **Interop** (later): NIR import/export; SONATA import.

### 8. Learning architecture

The kernel must serve local plasticity and every major learning method over **one**
graph. Each method puts concrete requirements on the kernel; several conflict with naive
simulation choices (e.g. binarised spikes kill gradients), so they are design inputs,
not add-ons.

#### 8.1 Requirements per method

| method | needs from the kernel | today |
|---|---|---|
| **Surrogate BPTT** | every op in the step differentiable; spikes carried as floats with a surrogate gradient; VJPs for every edge storage (dense, CSR, procedural, conv) and for delay buffers; learnables as `ParamState` | works in the BrainPy family; NEST holders binarise with `>= 0.5` (`simulator.py:1774`) and `BinaryArray` paths, which cut gradients; event-kernel VJP coverage unverified |
| **Checkpointed / truncated BPTT** | runner accepts `checkpointed_for_loop` (`base`), truncation windows, detach/reset points | user-assembled |
| **EventProp** (exact gradients at spike times via an adjoint system) | dynamics as explicit ODE (`derivatives`), threshold function, reset/jump map; spike times recorded (in-step offsets); backward adjoint integration as a `custom_vjp` over the rollout; delays as time shifts | impossible: dynamics are opaque `update()` code |
| **Online learning via `braintrace`** (e-prop, D-RTRL, ES-D-RTRL, OSTL, SnAp, UORO, ThreeFactor, …) | braintrace ≥ 0.2: learnables are `brainstate.ParamState`, dynamics live in `HiddenState`, and every weight application goes through braintrace ETP primitives (`matmul`, `sparse_matmul`, `conv`, `element_wise`, or `register_primitive`); `braintrace.compile(model, algorithm)` traces one step to find hidden↔parameter relations | NEST projections apply weights through raw dense/brainevent ops that braintrace cannot see |
| **Local plasticity** (STDP family, STP, Clopath, Urbanczik, Vogels–Sprekeler, …) | forward-only pure rule over edge state; pre/post traces; post-state reads | exists (NEST family only) |
| **Three-factor / reward-modulated** | eligibility traces in edge state + a broadcast modulatory signal from a device, population or host | partial (`stdp_dopamine` + volume transmitter) |
| **Learned plasticity (meta-learning)** | gradients through rule parameters over a rollout: rule is pure and differentiable, its parameters are `ParamState` | not supported (rule params are frozen scalars) |
| **Gradient-free** (evolution strategies; FORCE / RLS; ridge readouts) | `vmap` over parameter populations; observer outputs available to closed-form or recursive readout fits | ES exists as one example; no readout trainers |
| **Structural plasticity** | fixed-capacity edge storage with an active mask (static shapes under JIT); a connectivity-update phase | absent |

#### 8.2 Consequences for the kernel

1. **State taxonomy, declared by every contract.** `ParamState` = trainable (weights,
   neuron/synapse/rule parameters); `HiddenState` = carried dynamics (what braintrace
   and BPTT differentiate through); `EdgeState` = per-edge values updated by local rules
   (plastic weights, eligibility traces); `ShortTermState` = buffers, counters, RNG keys.
   A weight declares whether it is gradient-trained, locally plastic, or both (hybrid).
2. **Differentiable event payload.** An `event` port carries `multiplicity × spike`,
   where `spike` holds the surrogate gradient: exact integer counts forward, surrogate
   backward. No binarisation anywhere in the kernel; an optional in-step time offset
   serves EventProp and precise-timing models alike.
3. **One synaptic-op layer.** Every edge storage implements forward, a VJP for weights
   and inputs, and a braintrace ETP-primitive registration. Projections never call raw
   matmuls, so every learner sees every weight.
4. **Declared dynamics enable derived learners.** From `derivatives` + `spiking` (D1) the
   kernel builds the surrogate step (BPTT), the adjoint system (EventProp), and exposes
   the step to braintrace's graph analysis. Models that only hand-write `step` declare
   which learners they support.
5. **`Learner` interface at L2 with capability negotiation.** `BPTT(loss, optimizer,
   checkpoint=…, truncation=…)`, `EventProp(…)`, `Online(algorithm=braintrace.EProp,
   learning_signal=…)`, `ES(…)`, `Readout('ridge' | 'force')`; local plasticity runs
   inside the deliver phase. A learner checks model capabilities (`differentiable`,
   `adjoint`, `etrace_compatible`, `jit_safe`) and fails with a clear message rather
   than silently mis-training.
6. **A learning phase in the schedule.** After *observe*, an optional *learn* phase runs
   on-demand updates at chosen intervals: optimizer steps, weight normalisation,
   learning-signal broadcast, trial resets.
7. **Ownership.** braintrace owns online algorithms; brainpy.state owns models and
   guarantees they are etrace-compatible. braintrace currently also ships `nn.LIF`,
   `ALIF`, `Expon`, `STP`, … — which library is canonical for basic models is an
   ecosystem decision, resolved by L3: **brainpy.state is canonical** for neuron, synapse and plasticity models; braintrace keeps algorithms and its ETP primitives, and its duplicate `nn` models are deprecated upstream.

#### 8.3 Learning gates per milestone

- **K0:** surrogate-BPTT gradients match finite differences on a small `LIF` and
  `iaf_psc_alpha` network through dense *and* sparse storage *and* a delay;
  `braintrace.compile(net, EProp)` succeeds on a K0 network and an update lowers the
  loss; `stdp_synapse` NEST parity.
- **K1:** EventProp prototype on LIF (spike-time-loss gradient vs finite differences);
  ThreeFactor through braintrace with a modulatory device; learned-STDP-parameter
  meta-gradient; ES via `vmap`; ridge readout.
- **K2:** every ported model declares its learner capabilities; a capability matrix is
  generated into the docs.

#### 8.4 Kernel needs from brainevent (verify, then propose upstream)

New CUDA/Pallas kernels are **not** built in this package: where the kernel needs one,
it is proposed and implemented in `brainevent`, with a JAX reference path here until it
lands. brainevent 0.2.1 already provides event matmuls over `BinaryArray`
(CSR/COO/CSC/dense/fixed-number/procedural `jitc*`), bit-packed and compact event
formats, and fixed-shape plastic updates (`update_{csr,csc,coo,dense}_on_binary_{pre,post}`).
Each item below is first checked against brainevent; only real gaps become proposals.

| need | why | likely status |
|---|---|---|
| Event matmul over **count/weighted events** (skip zeros, carry multiplicity), with VJPs w.r.t. weights and inputs for every storage | differentiable event payload (§8.2-2) without binarisation | verify — current fast paths are binary |
| Weight gradients for **procedural** (`jitc*`) connectivity parameters | BPTT / meta-learning on never-materialised connectivity | verify |
| **Generic per-edge rule kernel** over CSR/CSC: apply a pure rule to edges touched by pre or post events, with arbitrary per-edge state (aux dict, eligibility traces) | today's built-in updates cover fixed STDP shapes only; rules like triplet, Clopath, three-factor need custom per-edge state | likely gap → propose (fused templates for common rule shapes; JAX fallback for the rest) |
| **Per-edge delay delivery** fused with the matmul (gather from a source ring buffer by edge delay) | heterogeneous delays without one matmul per delay value | verify |
| **Fixed-capacity masked CSR** (active-edge mask, in-place insert/remove) | structural plasticity under static JIT shapes | likely gap → propose |
| **braintrace ETP-primitive registration** of brainevent ops | online learning must see every weight (§8.2-3) | coordinate across brainevent / braintrace |
| Event-driven **transpose at spike times** for the EventProp adjoint | exact gradients without dense backward passes | CSC paths exist; verify fit |

### 9. How today's special cases map onto the design

| today | becomes |
|---|---|
| `w_by_rec` bridge, `receptor_k` labels, `delta_label_for_receptor`, `spike_events`, `x_filtered` | named input ports |
| `_emission_attr` / seam (H), rate connections, SIC, diffusion dual channel | `graded` output port → projection → port |
| gap couplers (`G@V - D*V`) | projection from `graded` `V` port with a difference synapse; explicit lag or WFR |
| `_relays_multiplicity`, binarised `BinaryArray` paths | `event` payload carries multiplicity |
| `_RECORDABLE_ALIAS` + resolver functions | declared recordables |
| `_volume_transmitter`, dopamine `signals` | device with a `graded` output read as a broadcast signal by the rule |
| `_injects_current`, `_CURRENT_GENERATORS`, class-name device classification | `Device` with `current` output port |
| `AlignPostProj` merging via descriptor `repr` | compiler merges `post`-aligned linear synapses with equal params |
| `Builder`/`Network` + `_RuleProj` | removed (subsumed by `Network`) |
| `align_pre/post_projection`, `align_*_ltp` stubs | removed |
| host-only NumPy models | `jit_safe=False` capability → clear compile error; JAX ports tracked as model work |

### 10. Decisions

| # | decision | outcome |
|---|---|---|
| D1 | Model authoring | ✅ **Python classes** with declared `params/states/ports` + `derivatives`; a DSL may come later as a second front-end lowering to the same contract |
| D2 | State substrate | ✅ **brainstate `Module`/`State`**, functional `step` inside |
| D3 | Front-ends | proposed: composable `Network` default + thin NEST-vocabulary layer |
| D4 | Delivery semantics | ✅ **NEST-faithful**: delay ≥ 1 step, no hidden lag (BrainPy-style timing shifts by one step vs 0.1) |
| D5 | Rewrite strategy | ✅ **new kernel alongside**, port families, delete old kernel in one 0.2/1.0-alpha break |
| D6 | `spec` branch | proposed: archive as tag, borrow ideas |
| L1 | Online-learning algorithms | ✅ **live in braintrace**; brainpy.state guarantees etrace compatibility |
| L2 | EventProp | ✅ **K1 prototype** for models with declared ODEs |
| L3 | Canonical home of basic models (LIF/ALIF/Expon/STP exist in both braintrace.nn and brainpy.state) | ✅ **brainpy.state is canonical**; braintrace.nn duplicates to be deprecated upstream in favour of brainpy.state models |
| L4 | Hybrid weights (local plasticity + gradient on the same weight) | ✅ **supported**, declared per weight |

### 11. Migration plan

| milestone | content | exit gate |
|---|---|---|
| **K0 — kernel spike** | L1 contracts (ports, `NeuronModel`, `SynapseModel`, `PlasticityRule`, `EdgeSet`), `Network` graph, step schedule, source-owned delay buffers, `Runner`. Three neurons (`iaf_psc_alpha`, LIF, `iaf_cond_exp`), Expon synapse, `static` + `stdp_synapse`. | Brunel (NEST parity, existing harness) **and** COBA E/I pass; learning gates of §8.3 (BPTT vs finite differences, braintrace EProp, STDP parity) pass; step time and memory ≤ today's Simulator at `order=2500`. **Go/no-go on the design.** |
| K1 — substrate complete | all port kinds, graded emission, gap (lag + WFR), STP, all storage strategies incl. procedural, observers, `cont`/host drive, batching. | every existing `_nest_network` behaviour test re-expressed and green |
| K2 — model library | port neuron/synapse/plasticity/device families onto the contracts (mostly mechanical; numerics reused); goal-01 repro tests are acceptance tests; host-only models either ported to JAX or flagged. | all 131 NEST validation tests green on the new kernel |
| K3 — front-ends + interop | NEST `Simulator` layer, `NodeView`/`SynapseCollection`, spatial; NIR export. | examples + docs migrated |
| K4 — removal + release | delete old kernel, `Builder`, dead projections; migration guide; 1.0-alpha. | full suite + docs build + benchmarks |

Goal-01 findings map as follows: autapse ids, edge retention, multiplicity,
`fixed_total_number` (mirror NEST: refuse `allow_multapses=False`) are fixed **by
construction** in K0/K1; `*Ref` soft reset, `MgBlock`, AMPA/GABAa saturation (make `g`
a dimensionless open fraction with an explicit `g_max`), `OUProcessInput` (exact
discretisation, σ = stationary std), `LeakyRateReadout`, `LogNormal` (`unit=`),
receptor-type validation and the namespace env var are fixed **while porting** (K2),
with the verified repro tests as acceptance criteria.

### 12. Risks

- **Scope.** 66 NEST neuron modules + the BrainPy family must be ported. Mitigation: K0
  go/no-go gate; numerics reused verbatim; mechanical ports per family.
- **Performance regression** from generality (ports, schedule). Mitigation: K0 gate
  benchmarks against today's Simulator.
- **Parity drift** from the delay-semantics change (D4). Mitigation: NEST is already
  delay ≥ 1 step; the harness's `align_steps` offsets shrink, and every change is
  visible in parity tests.
- **Upstream kernel gaps** (brainevent, braintrace). Mitigation: JAX reference paths in
  this package; kernels proposed and landed upstream (§8.4); gates measure both.
- **brainstate churn** (0.6.0 removed the input API mid-cycle). Mitigation: the kernel
  owns ports/inputs; depend only on `State`, `Module`, transforms.

### Sources

NEST simulation/min-delay: https://nest-simulator.readthedocs.io/en/v3.1/guides/running_simulations.html ·
NESTML synapses: https://nestml.readthedocs.io/en/latest/nestml_language/synapses_in_nestml.html ·
Brian2: https://elifesciences.org/articles/47314 ·
GeNN 5: https://genn-team.github.io/genn/documentation/5/building_networks.html ·
procedural connectivity: https://www.nature.com/articles/s43588-020-00022-7 ·
NEST GPU construction: https://www.mdpi.com/2076-3417/13/17/9598 ·
Arbor: https://docs.arbor-sim.org/en/latest/concepts/interconnectivity.html ·
Lava: https://lava-nc.org/lava_architecture_overview.html ·
Rockpool: https://rockpool.ai/reference/_autosummary/rockpool.nn.modules.html ·
Spyx: https://arxiv.org/abs/2402.18994 ·
jaxley: https://www.nature.com/articles/s41592-025-02895-w ·
BrainPy: https://elifesciences.org/articles/86365 ·
NIR: https://www.nature.com/articles/s41467-024-52259-9 ·
SONATA: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007696
