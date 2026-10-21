# Architecture review and kernel redesign

_Date: 2026-10-21_ · Status: **direction agreed (D1, D2, D4/D4a, D5, I1, S1, S1a, L1–L4 decided 2026-10-21); expert review incorporated (Part III); K0 spec: [`2026-10-21-k0-kernel-spec.md`](2026-10-21-k0-kernel-spec.md)** ·
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
| NEST-faithful numerics: exact propagators with NEST's singular fallbacks; in-substep spike/reset handling (`event_fn`) — kept, moved onto fixed substeps (I1). | `_nest_base/utils.py` |
| Live-NEST parity harness (`nest_compare`, trace/distributional modes, `align_steps`), 131 validation test files. | `_nest_validation/` |
| O(N_post) merged post-aligned synapses (the ICLR-2024 AlignPost result). | `_brainpy/projection.py` |
| Event-driven sparse kernels (brainevent), procedural connectivity (`JITC*`). | comm objects |
| End-to-end unit safety (brainunit). Surrogate spikes; detachable hard reset (becomes a learner option, §8.2). | everywhere |
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

| kind | payload | application point (see §7.1) | examples |
|---|---|---|---|
| `event` | spike multiplicity × weight per target per step; delays rounded to the grid (NEST); never interpolated | jump after propagation | spike output; `spike_generator`; parrot relay |
| `timed_event` | fixed-capacity list of `(offset, weight)` per target per step, with declared capacity and an overflow counter; sorted by offset at the receiver | integrated between events | `*_ps` models, `poisson_generator_ps` |
| `current` | current (pA), summed | latched; acts during the next step | static PSC input, `dc_generator`, SIC |
| `conductance` | conductance (nS) with `E_rev` and an optional voltage gate `gate(V)` evaluated inside `derivatives` at every substep | inside the ODE | `iaf_cond_*`, multisynapse ports, NMDA Mg block (replaces `SynOut` COBA/MgBlock) |
| `graded` | continuous value with declared unit; sub-step delays may interpolate | latched | NMDA emission, astrocyte SIC |
| `gap` | symmetric pairing of two `V` ports with a per-edge conductance | explicit lag or fixed-iteration WFR | `hh_psc_alpha_gap` |
| `modulatory` | broadcast scalar/vector (volume transmission) | read by rules and models | dopamine via `volume_transmitter` |

Input ports are named (`'AMPA'`, `'GABA'`, `'rec_3'`); NEST's integer `receptor_type`
maps to names in the NEST front-end. Units and kinds are checked at `connect` time.
Synaptic kinetics have exactly one owner per port: either the neuron (state such as
`g_ex` filtered inside the neuron, NEST style) or a `SynapseModel` on the projection;
connecting a filtering synapse to a port the neuron already filters is a validation
error. Weight distributions declare sign/clipping (an excitatory conductance cannot be
drawn negative).

**`NeuronModel`** (a brainstate `Module`):

```python
class iaf_cond_exp(NeuronModel):
    params  = Params(C_m=250*u.pF, g_L=16.6*u.nS, E_ex=0*u.mV, ..., t_ref=2*u.ms)
    states  = States(V_m=Init(-70*u.mV), g_ex=Init(0*u.nS), g_in=Init(0*u.nS))
    inputs  = Ports(ex=Conductance(u.nS), in_=Conductance(u.nS), I=Current(u.pA))
    outputs = Ports(spike=Event())
    spiking = SpikeRule(condition=lambda s, p: s.V_m >= p.V_th,
                        jump=lambda s, p: s.replace(V_m=p.V_reset),
                        refractory=Clamp('V_m', 't_ref'))
    integrator = RK(order=4, substeps=4)   # fixed substeps; 'exact' only for linear models

    def derivatives(self, s, inp, ctx): ...        # continuous part
    def on_input(self, s, inp, ctx): ...           # optional jump part (e.g. g += Δg)
```

- `step(state, inputs, ctx) -> (state, outputs)` is generated from `derivatives` +
  `on_input` + `spiking` + `integrator`; models with hand-written exact propagators
  override `step` directly. One shared `SpikeRule` implementation replaces ~13 copied
  `get_spike` and refractory blocks (and fixes the `*Ref` soft-reset bug by
  construction). `SpikeRule` = condition function + jump map over **all** states
  (`aeif` `w += b`, Izhikevich `u += d`) + refractory semantics (clamp vs block) +
  optional stochastic hazard (escape noise for `pp_psc_delta`, `gif_*`, `erfc_neuron`;
  consumes RNG) + a peak-detection variant for HH-type models with no reset. The
  spike condition is checked at every integrator substep, so in-substep spikes and
  resets (today's `event_fn` in RKF45) are preserved; multiple spikes in one step
  are counted, not collapsed to a boolean.
- **Integrators are fixed-substep only** (decision I1). Adaptive RKF45
  (`AdaptiveRungeKuttaStep`, a `while_loop`) is removed: `jax.grad` through
  `while_loop` raises, and one stiff neuron sets the iteration count for the whole
  population. Options: `exact` (linear models only — rejected for nonlinear ones),
  `exp_euler`, `RK(order, substeps)`, exponential-Rosenbrock for stiff models; the
  substep count is static per model and overridable. NEST parity for formerly-GSL
  models is shown by a convergence study (error vs substeps against NEST) that fixes
  each model's default substep count — see §13 risks.
- **Parameters are traced arrays**, per-neuron heterogeneous by default, so sweeps and
  `vmap` over parameters do not recompile. `Init` accepts distributions.
- `recordables` are the declared states plus declared derived quantities — no alias
  table.
- `capabilities`: `jit_safe` (false for host-only models → compile error, not silent
  breakage), `differentiable`, `precise_spike_times`.

**`SynapseModel`** — dynamics between edge delivery and port deposit (Expon, Alpha,
DualExpon, AMPA/NMDA kinetics, STP). Declares its alignment: `post` (state per target
neuron), `pre` (per source neuron), or `edge` (per edge). Merging `post`-aligned
synapses (the AlignPost O(N_post) result) is a compiler **optimisation pass**: only for
linear kinetics with identical parameters, never for nonlinear kinetics (NMDA
saturation) or `timed_event` ports; any observer or learner that needs per-projection
values (per-source `I_syn`, per-projection gradients) vetoes it; the plan reports which
projections were merged; users can opt out.

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
`set('weight')` are exact. Retention policy: edges live on device for materialised
storages and are regenerated lazily for `procedural` ones, which are read-only (no
`set('weight')` without materialising). Index dtypes are int32 below 2³¹ edges and
int64 above (CSR `indptr` overflows otherwise); the plan states the memory per
projection (≈ 5 GB for 3×10⁸ edges with pre/post/weight/delay).

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
  1. deliver   every projection reads source outputs emitted at step n - d,
               applies synapse/plasticity, deposits into target input ports
               (d >= 1 step on cycles; d = 0 allowed on acyclic paths, which are then
               updated in topological order within the step)
  2. update    every population: (state, outputs) = step(state, port_inputs, ctx)
  3. emit      outputs written to per-source delay buffers (one buffer per source port,
               sized max delay — shared by all projections from that source)
  4. observe   observers sample
```

- Delays mean what the user wrote; the hidden extra lag of today's Simulator
  disappears. On cycles the minimum delay is one step (NEST). On acyclic paths delay
  0 is allowed (decision D4a), so an L-layer feed-forward SNN has no L-step latency;
  the compiler topologically orders zero-delay paths and rejects zero-delay cycles.
- Event delays are rounded to the grid (NEST) or carried as precise offsets on
  `timed_event` ports; interpolation is allowed only on `graded` ports. Continuous
  same-step coupling (gap junctions) is explicit-lag by default with an
  optional WFR mode that runs a **fixed** iteration count (differentiable).
- Delay-buffer placement is chosen by the compiler per projection: **pre-side** ring
  (`N_src × D_max`, shared by all projections from that source) for scalar or
  per-source delays; **post-side** arrival ring (`N_tgt × D_max`, scatter on spike —
  NEST/GeNN) for per-edge delays, which keeps delivery event-driven. The plan reports
  the choice and its memory.
- Port inputs are passed as a value into `step` — no dict popping.
- Runner: `simulate`, `cont`, `step`; host drive for closed loops; batching by `vmap`
  over the state pytree (trials, parameter sweeps); optional min-delay epoch chunking
  (NEST/Arbor model) for long runs.
- Learning: see §8 — learners are chosen per run over the same graph.

#### 7.1 Within-step timing contract

Written per port kind (table above) and tested against NEST per kind in K0, so parity
offsets cannot hide a one-step shift: `event` inputs are a jump applied **after** the
continuous propagation of the step they arrive in (NEST ring-buffer semantics);
`current`/`graded` inputs are latched and act during the **next** step; `conductance`
inputs enter the ODE and are seen at every substep; `timed_event` inputs are sorted by
offset and the model integrates between them. Plasticity follows NEST's **lazy**
semantics by default (weights updated on pre-spike delivery from the post history,
with dendritic/axonal delay split as in NEST); an eager mode is available for
learning, and observers see the weight of the semantics in use.

#### 7.2 RNG and reproducibility contract

Counter-based keys derived from `(seed, object id, gid, step)`: every population,
edge set and step gets an independent stream that does not depend on evaluation
order. Required, tested invariances: the realised connectivity is identical across
storages (`dense`/`csr`/`procedural`), batch layouts and shard counts; under `vmap`
the user chooses independent or common random numbers per trial; `cont()` and
checkpoint/restore continue the same streams.

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
   A weight declares whether it is gradient-trained, locally plastic, or hybrid. Hybrid
   supports **both modes, chosen by the user** (decision L4): *decomposed*
   `w = w_slow (ParamState) + α ⊙ w_fast (EdgeState)`, where each learner states which
   parts receive gradients; or *through-the-rule*, a single weight mutated by the rule
   in the forward pass with BPTT differentiating the rule trajectory (zero gradient at
   clipping bounds is documented). Online learning on plastic weights is rejected until
   braintrace supports time-varying parameters.
2. **Differentiable event payload.** An `event` port carries `multiplicity × spike`,
   where `spike` holds the surrogate gradient: exact integer counts forward, surrogate
   backward, defined as `count_ste = stop_gradient(c - s) + s` so counts > 1 keep a
   well-defined backward pass. No binarisation anywhere in the kernel. Precise times
   travel on `timed_event` ports. Detaching the reset from the graph (today's
   `stop_gradient` hard reset) is a **learner option**, not a model property — EventProp
   needs the reset Jacobian.
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
  `braintrace.compile(net, EProp)` succeeds on a K0 network (update-quality checks move
  to K1: OSTL exact on a feed-forward net, D-RTRL vs BPTT on one step); `stdp_synapse` NEST parity.
- **K1:** EventProp prototype scoped to LIF with interpolated threshold-crossing times,
  no refractoriness, homogeneous delays (spike-time-loss gradient vs finite
  differences); resets with adaptation jumps, refractory clamps, heterogeneous delays
  and learnable time constants through propagator fallbacks are research items;
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

### 9. Cross-cutting requirements

Added after the expert review (Part III). Each item gets a section in the K0 or K1
spec and a gate in its milestone.

| area | requirement | milestone |
|---|---|---|
| Precision | float32 parity tier with per-model tolerances alongside the float64 tier (129 validation tests force float64 today); float32-safe propagator fallbacks; mixed-precision guidance | K0 (policy), K2 (per model) |
| Stochastic semantics | noise ports with defined discretisation (Euler–Maruyama vs exact OU); hazard spiking; distributional parity with stated statistics | K1 |
| Initial conditions & heterogeneity | `Init` distributions; per-neuron parameter distributions with unit-aware clipping; warm-up / steady-state initialisation | K0 |
| Checkpoint / restore | full state incl. RNG counters, delay buffers, edge state, optimiser state; storage-independent canonical serialisation of edge state | K1 |
| Distributed layout | gid → shard map, target-side partitioning, min-delay communication epochs, shard-invariant connectivity construction — specified now because buffer and edge layouts depend on it; `shard_map` prototype later | K1 (spec), K3 (prototype) |
| Connectivity construction | on-device, blocked, vectorised builders (no `(n_pre, n_post)` draws, no per-neuron Python loops); exact NEST counts for deterministic rules, distributional parity otherwise | K1 |
| Recording at scale | sparse spike recording into fixed-capacity buffers with overflow counters; downsampled multimeters; online reductions (rates, PSTH, CV, correlations); streaming to host | K1 |
| Experiments & sweeps | stimulus schedules, trial structure and resets, sweep API (`vmap`/`pmap` with chunking), per-trial seeds | K1 |
| Validation beyond NEST | analytical cases (LIF f–I, Siegert rates), Brian2 references for non-NEST models, convergence-order tests (error vs dt and substeps) | K0 (framework), K2 |
| Interop | NIR import **and** export; PyNN front-end scope; SONATA import; NESTML as the reference for a future DSL | K3 |
| Hardware deployment | quantisation-aware training (fixed-point weights/state); per-target constraint checks (fan-in, delay range, precision) at `compile`; ANN-SNN conversion workflow | after K4 (scoped in K3) |
| Learning surface | loss library (spike count, spike time, van Rossum, rate regularisers), regularisation hooks, readout trainers, gradient diagnostics | K1–K2 |
| Diagnostics | printable plan (schedule, buffers, storage, merges); NaN/instability flags; event-buffer overflow reporting; structured errors with source locations (from the `spec` branch) | K0 |
| Benchmarks | cost model per storage; compile time as a metric; fixed hardware/precision/protocol | K0 |
| Testing strategy | property-based tests for connection-rule invariants; metamorphic tests (storage/batch/shard invariance); convergence-order tests; gradient checks per learner × storage × delay | K0 onward |
| API stability & docs | public-API freeze list and deprecation policy (from the roadmap); migration guide skeleton from K1; tutorial tracks for computational neuroscience, simulation infrastructure and neuromorphic users | K1–K4 |

### 10. How today's special cases map onto the design

| today | becomes |
|---|---|
| `w_by_rec` bridge, `receptor_k` labels, `delta_label_for_receptor`, `spike_events`, `x_filtered` | named input ports |
| `_emission_attr` / seam (H), SIC (rate connections and diffusion removed, S1/S1a) | `graded` output port → projection → port |
| gap couplers (`G@V - D*V`) | projection from `graded` `V` port with a difference synapse; explicit lag or WFR |
| `_relays_multiplicity`, binarised `BinaryArray` paths | `event` payload carries multiplicity |
| `_RECORDABLE_ALIAS` + resolver functions | declared recordables |
| `_volume_transmitter`, dopamine `signals` | device with a `graded` output read as a broadcast signal by the rule |
| `_injects_current`, `_CURRENT_GENERATORS`, class-name device classification | `Device` with `current` output port |
| `AlignPostProj` merging via descriptor `repr` | compiler merges `post`-aligned linear synapses with equal params |
| `Builder`/`Network` + `_RuleProj` | removed (subsumed by `Network`) |
| `align_pre/post_projection`, `align_*_ltp` stubs | removed |
| host-only NumPy models | `jit_safe=False` capability → clear compile error; JAX ports tracked as model work |

### 11. Decisions

| # | decision | outcome |
|---|---|---|
| D1 | Model authoring | ✅ **Python classes** with declared `params/states/ports` + `derivatives`; a DSL may come later as a second front-end lowering to the same contract |
| D2 | State substrate | ✅ **brainstate `Module`/`State`**, functional `step` inside |
| D3 | Front-ends | proposed: composable `Network` default + thin NEST-vocabulary layer |
| D4 | Delivery semantics | ✅ **NEST-faithful**: no hidden lag; delay ≥ 1 step on cycles |
| D4a | Zero delay on acyclic paths | ✅ **allowed** (topological order within a step); zero-delay cycles rejected |
| I1 | Integrators | ✅ **fixed-substep only**; adaptive RKF45 removed; per-model substep defaults set by convergence against NEST |
| S1 | Model scope | ✅ **point neurons only**: multi-compartment and population/mean-field models are not ported and are removed at K4 (list in §13) |
| S1a | Single-unit rate neurons | ✅ **not supported**: `lin_rate`, `tanh_rate`, `threshold_lin_rate`, `gauss_rate`, `sigmoid_rate`, `sigmoid_rate_gg_1998`, `rate_neuron_ipn/opn`, `rate_transformer_node`, with `rate_connection_instantaneous/delayed` and `step_rate_generator` are not ported and are removed at K4 |
| D5 | Rewrite strategy | ✅ **new kernel alongside**, port families, delete old kernel in one 0.2/1.0-alpha break |
| D6 | `spec` branch | proposed: archive as tag, borrow ideas |
| L1 | Online-learning algorithms | ✅ **live in braintrace**; brainpy.state guarantees etrace compatibility |
| L2 | EventProp | ✅ **K1 prototype**, scoped to interpolated-threshold LIF without refractoriness, homogeneous delays |
| L3 | Canonical home of basic models (LIF/ALIF/Expon/STP exist in both braintrace.nn and brainpy.state) | ✅ **brainpy.state is canonical**; braintrace.nn duplicates to be deprecated upstream in favour of brainpy.state models |
| L4 | Hybrid weights (local plasticity + gradient on the same weight) | ✅ **both modes, user-configured**: decomposed `w_slow + α·w_fast`, or one weight differentiated through the rule |

### 12. Migration plan

| milestone | content | exit gate |
|---|---|---|
| **K0 — kernel spike** | L1 contracts (ports, `NeuronModel`, `SynapseModel`, `PlasticityRule`, `EdgeSet`), `Network` graph, step schedule, source-owned delay buffers, `Runner`. Neurons chosen to exercise the risks: `iaf_psc_alpha`, LIF, `iaf_cond_exp`, **`aeif_cond_alpha`** (in-substep spikes, multi-state jump, fixed-substep integrator), **`iaf_psc_exp_ps` + `poisson_generator_ps`** (`timed_event`), and an **NMDA/MgBlock conductance** (voltage-gated port); Expon synapse; `static` + `stdp_synapse`; RNG contract. | Per-port-kind timing tests vs NEST (§7.1); Brunel vs NEST with named statistics and tolerances (rate, CV-ISI, Fano factor, pairwise correlation, with confidence intervals over seeds); COBA E/I vs Brian2; storage/batch invariance of realised connectivity; `aeif_cond_alpha` substep-convergence vs NEST; learning gates of §8.3; step time, compile time and memory ≤ today's Simulator at `order=2500` on named hardware, in float32 and float64. **Go/no-go on the design.** |
| K1 — substrate complete | all port kinds, graded emission, gap (lag + WFR), STP, all storage strategies incl. procedural, observers, `cont`/host drive, batching, in-substep event localisation for fixed-substep integrators. | every existing `_nest_network` behaviour test re-expressed and green |
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

### 13. Risks

- **Scope.** ~60 NEST neuron modules + the BrainPy family must be ported. Mitigation:
  K0 go/no-go gate; numerics reused verbatim; mechanical ports per family. Removed
  under S1 (point neurons only), not ported: multi-compartment `cm_default`,
  `iaf_cond_alpha_mc`, `pp_cond_exp_mc_urbanczik` (and therefore `urbanczik_synapse`);
  population/mean-field `gif_pop_psc_exp`, `siegert_neuron` (and
  `diffusion_connection`). Under S1a, single-unit rate neurons are removed as well:
  `lin_rate`, `tanh_rate`, `threshold_lin_rate`, `gauss_rate`, `sigmoid_rate`, `sigmoid_rate_gg_1998`, `rate_neuron_ipn/opn`, `rate_transformer_node`, with `rate_connection_instantaneous/delayed` and `step_rate_generator`.
- **Parity loss from fixed substeps (I1).** Formerly-GSL models (`aeif_*`,
  `iaf_cond_*`, `hh_*`, `ht_neuron`) lose NEST's adaptive error control; stiff models
  may need many substeps or exponential-Rosenbrock to stay within tolerance.
  Mitigation: per-model convergence study against NEST sets default substeps; parity
  tolerances for these models are re-derived and documented, not silently loosened.
  K0 measured this for `aeif_cond_alpha` (T3, `2026-10-21-k0-t3-models.md`): fixed-substep
  RK4 locates the threshold crossing only to within one substep (first order), giving
  ~1 mV errors on spike steps (2.26 mV at 16, 0.87 mV at 64 substeps) while spike times
  and off-spike traces match NEST. Mitigation planned for K1: in-substep threshold-crossing
  localisation (root finding / interpolation within the crossing substep, then reset and
  integration of the remainder of the substep).
- **Performance regression** from generality (ports, schedule). Mitigation: K0 gate
  benchmarks against today's Simulator.
- **Parity drift** from the delay-semantics change (D4). Mitigation: NEST is already
  delay ≥ 1 step; the per-port timing contract (§7.1) is tested per kind in K0, so
  `align_steps` offsets cannot hide a one-step shift.
- **Upstream kernel gaps** (brainevent, braintrace). Mitigation: JAX reference paths in
  this package; kernels proposed and landed upstream (§8.4); gates measure both.
- **brainstate churn** (0.6.0 removed the input API mid-cycle). Mitigation: the kernel
  owns ports/inputs; depend only on `State`, `Module`, transforms.

---

## Part III — Expert review (2026-10-21) and disposition

A three-perspective review (computational neuroscience, simulator infrastructure,
neuromorphic/SNN learning) found 19 issues (A1–A19) and 17 missing aspects (B1–B17).
Verified in this session: `jax.grad` through `brainstate.transform.while_loop` raises
(A1; `_nest_base/utils.py:715`); `InputDelay` interpolates fractional event delays
(A3; `_brainpy/delay.py:118-127`); 129 validation tests force float64 (A19).

| item | disposition |
|---|---|
| A1 RKF45 not reverse-differentiable | I1: fixed-substep integrators only (§7 L1) |
| A2 within-step timing | §7.1 contract, tested per port kind in K0 |
| A3 interpolation vs integer events | event delays rounded; interpolation only on `graded` (§7 L2) |
| A4 precise spike times | `timed_event` port kind (§7 L1); `*_ps` model in K0 |
| A5 port taxonomy | `conductance(E_rev, gate)`, `gap`, `modulatory` ports; compartments out of scope (S1) |
| A6 example contradictions | single owner of synaptic kinetics; signed weight distributions (§7 L1) |
| A7 spike rule too narrow | `SpikeRule` (§7 L1) |
| A8 RNG contract | §7.2 |
| A9 delay-buffer placement | compiler chooses pre- vs post-side (§7 L2) |
| A10 hybrid weights | L4: both modes, user-configured (§8.2) |
| A11 surrogate counts | `count_ste`; detach-reset as learner option (§8.2) |
| A12 EventProp scope | narrowed (L2, §8.3) |
| A13 plasticity timing | lazy NEST semantics by default, eager option (§7.1) |
| A14 edge retention at scale | retention policy and index dtypes (§7 L1) |
| A15 merging | optimisation pass with veto/opt-out (§7 L1) |
| A16 batching & parameters | traced per-neuron parameters (§7 L1); brainevent batching rules to verify (§8.4) |
| A17 feed-forward latency | D4a: zero delay on acyclic paths |
| A18 state shapes vs storage | storage-independent edge-state serialisation (§9) |
| A19 float64-only parity | float32 tier (§9) |
| B1–B17 | §9 cross-cutting requirements |

Not yet verified (first checks under §8.4): braintrace's handling of brainunit
quantities and its `ControlFlowPolicy`; batching rules of brainevent kernels; VJP
coverage for count-valued events.

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
