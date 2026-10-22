# Architecture review and kernel redesign

_Date: 2026-10-21 · revised 2026-10-22 (post-K0)_ · Status: **direction agreed (D1, D2, D4/D4a, D5, I1, S1, S1a, L1–L4 decided 2026-10-21; D3, D6, D-a…D-g decided 2026-10-22); expert review incorporated (Part III); K0 done — [report](2026-10-22-k0-report.md) (go); K1a spec: [`2026-10-22-k1a-kernel-hardening-spec.md`](2026-10-22-k1a-kernel-hardening-spec.md)** ·
Breaking changes: **allowed** (no 0.1 compatibility constraint).

> **Revision 2026-10-22 (post-K0).** A post-K0 architecture review compared this
> document with the merged kernel (`brainpy_state/_kernel/`, `main` @ 7938574) and the
> lead took decisions D-a…D-g (§11). Changes in this revision:
>
> 1. **Current/graded latch is per port (D-a).** `latch=True|False` on `Current` and
>    `Graded` ports; NEST models latch (one step, NEST `CurrentEvent`), BrainPy-family
>    models act in the same step. D4a rewritten to say which port kinds a zero delay
>    covers (§7 L1 ports, §7 L2, §7.1, §11).
> 2. **One encoding for neuron-owned receptor ports (D-b).** An `Event` port that names
>    the state it filters into replaces `filtered=True` on `Current`/`Conductance`;
>    the §7 `iaf_cond_exp` / `Network` examples are fixed (the old example raises
>    `PortError` on the K0 kernel).
> 3. **Pre-aligned synapses** are lowered by the compiler to hidden receptor
>    populations (§7 L1 `SynapseModel`).
> 4. **`SpikeRule` in-substep localisation** is part of the contract; parity target
>    restated as convergence to the reference within NEST tolerance (§7 L1).
> 5. **Plastic weights live in edge order**; Dense is a derived compute layout (§7 L1).
> 6. **`timed_event` capacity** is a statistical budget checked at run time; the segment
>    loop is a `scan`; precise timing is today a parity demo, not network scale (§7 L1).
> 7. **`Program` is the L2 contract** (D-c): learners take `(Program, state0)`, the
>    `Runner` is one client among others, and the *learn* phase is real (§7 L2, §8.2).
> 8. **Explicit `spikes='hard'|'relaxed'`** on the `Program` (§8.2).
> 9. **Step contract as built**: per population in schedule order deliver → update, then
>    emit; plasticity runs in emit (§7 L2).
> 10. **`Network` is not a `Module`; `Plan` is** (§7 L3).
> 11. **RNG layout (D-g)**: name-hashed object keys + gid-blocked dynamics noise (§7.2).
> 12. **New §7.3 compile-time model** (principle 7): no model-length Python loops in the
>     step; compile-time budget as a gate.
> 13. **BrainPy-style API deprecated (D-e)**: standalone modules and AlignPost/AlignPre
>     projections; the "usable standalone inside any brainstate module" promise is
>     removed; `Network` → `Program` is the only execution path (§6, §7 L4, §10).
> 14. **`_kernel` private until K3 (D-d)**; new §7.4 public naming/export plan.
> 15. **Old kernel frozen until K4 deletion (D-f)** (§12).
> 16. §11 decisions table (D-a…D-g, D3, D6), §12 migration plan (K0 done; K1a, K1b,
>     K1-L, K2, K3, K4) and §13 risks (compile time, GPU, per-edge delays, gate
>     strength) rewritten.

Superseded the pre-K0 roadmap's §2 ("Two families, one kernel") and Phases 1–4;
[`2026-10-21-roadmap.md`](2026-10-21-roadmap.md) now indexes milestones K0–K4 of §12
(rewritten 2026-10-22). Phase 0 bug findings
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
   only that graph runs, and only as `Network` → `Plan` → `Program` (D-e: there is no
   second, standalone-module execution path). NEST parity therefore validates what
   users execute.
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
7. **Compile time is part of the design** (added 2026-10-22, §7.3). The traced step
   contains no Python loop whose trip count depends on the model (substeps, in-step
   event segments, delay groups, populations of one kind); such loops are `scan`s over
   static lengths. Compile time is measured and gated like step time.

### 7. Layers

```
 L4  Front-ends       Network (Python API)           NEST vocabulary (Simulator, thin)   Interop (NIR, SONATA import*)
                              \                              |                          /
 L3  Network graph    Network: Populations(gid ranges) + Projections + Devices + Observers
                      validate (ports/units/delays) · inspect · serialise   (plain declaration, not a Module)
                              |  compile()
 L2  Compiler/runtime  Plan (brainstate Module): schedule, delay buffers, edge storage, merged synapses, States
                      Program(plan, spikes=…): pure init_state / step / rollout over explicit pytrees  <- the L2 contract
                      clients of a Program: Runner (simulate / cont / step · vmap batching · host loop)
                                            learners (BPTT · checkpointed · EventProp · online (braintrace) · ES · readout)
                      local plasticity runs inside the step (emit phase)
                              |
 L1  Model contracts  NeuronModel · SynapseModel · PlasticityRule · ConnRule/EdgeSet · Device · Observer
                              |
 L0  Substrate        brainstate (State, transforms) · brainunit · brainevent kernels
```

#### L1 — model contracts

**Ports.** Every connection goes from an output port to an input port. Port kinds:

| kind | payload | application point (see §7.1) | examples |
|---|---|---|---|
| `event` | spike multiplicity × weight per target per step; delays rounded to the grid (NEST); never interpolated. An input `event` port may **name the state it filters into** (`Event(u.nS, into='g_ex')`, D-b): that is a neuron-owned receptor port | jump after propagation (onto the named state, or onto the port's jump target) | spike output; `spike_generator`; parrot relay; receptor ports `ex`/`in` of `iaf_psc_alpha`, `iaf_cond_exp`, `aeif_cond_alpha` |
| `timed_event` | fixed-capacity list of `(offset, weight)` per target per step, with a capacity **budget** and an overflow counter checked at run time; sorted by offset at the receiver | integrated between events | `*_ps` models, `poisson_generator_ps` |
| `current` | current (pA), summed, unfiltered | **per port** (D-a): `latch=True` (default) acts during the next step (NEST `CurrentEvent`); `latch=False` acts in the same step | static PSC input, `dc_generator`, SIC; `I` of NEST models (latched), `I` of BrainPy-family `LIF` (same step) |
| `conductance` | conductance (nS) with `E_rev` and an optional voltage gate `gate(V)` evaluated inside `derivatives` at every substep; unfiltered (a projection-owned `SynapseModel` supplies `g`) | inside the ODE | multisynapse ports, NMDA Mg block (replaces `SynOut` COBA/MgBlock) |
| `graded` | continuous value with declared unit; sub-step delays may interpolate | per port, as `current` (`latch=True|False`) | NMDA emission, astrocyte SIC |
| `gap` | symmetric pairing of two `V` ports with a per-edge conductance | explicit lag or fixed-iteration WFR | `hh_psc_alpha_gap` |
| `modulatory` | broadcast scalar/vector (volume transmission) | read by rules and models | dopamine via `volume_transmitter` |

Input ports are named (`'AMPA'`, `'GABA'`, `'rec_3'`); NEST's integer `receptor_type`
maps to names in the NEST front-end. Units and kinds are checked at `connect` time.
Synaptic kinetics have exactly one owner per port, and there is **one encoding** for
each owner (D-b, 2026-10-22):

- **neuron-owned** (NEST style): an `Event` input port that names the state it filters
  into (`ex=Event(u.nS, into='g_ex')`); a delivered weight jumps that state, and the
  neuron's own `derivatives` filter it. Incoming projections carry no `SynapseModel`.
- **projection-owned**: a `SynapseModel` on the projection (Expon, Alpha, NMDA, …)
  deposits into an unfiltered `Current` or `Conductance` port.

Attaching a `SynapseModel` to a neuron-owned receptor port is a validation error. The
K0 encoding `Current/Conductance(filtered=True)` and its special case in the
port-compatibility check (`_kernel/ports.py:495-496`) are removed in K1a-3. Weight
distributions declare sign/clipping (an excitatory conductance cannot be drawn
negative).

**`NeuronModel`** (a brainstate `Module`):

```python
class iaf_cond_exp(NeuronModel):
    params  = Params(C_m=250*u.pF, g_L=16.6*u.nS, E_ex=0*u.mV, ..., t_ref=2*u.ms)
    states  = States(V_m=Init(-70*u.mV), g_ex=Init(0*u.nS), g_in=Init(0*u.nS))
    inputs  = Ports(ex=Event(u.nS, into='g_ex'),      # neuron-owned receptor ports (D-b)
                    in_=Event(u.nS, into='g_in'),
                    I=Current(u.pA, latch=True))      # NEST CurrentEvent: next step (D-a)
    outputs = Ports(spike=Event())
    spiking = SpikeRule(condition=lambda s, p: s.V_m >= p.V_th,
                        jump=lambda s, p: s.replace(V_m=p.V_reset),
                        refractory=Clamp('V_m', 't_ref'))
    integrator = RK(order=4, substeps=2)   # fixed substeps; 'exact' only for linear models

    def derivatives(self, s, inp, ctx): ...        # continuous part (g_ex, g_in decay here)
    def on_input(self, s, inp, ctx): ...           # optional jump part (e.g. g += Δg)
```

(`into=` is the keyword, lead decision 2026-10-22; `TimedEvent` receptor ports use it
too, and the `filtered` flag is removed from every port kind.)

- `step(state, inputs, ctx) -> (state, outputs)` is generated from `derivatives` +
  `on_input` + `spiking` + `integrator`; models with hand-written exact propagators
  override `step` directly. One shared `SpikeRule` implementation replaces ~13 copied
  `get_spike` and refractory blocks (and fixes the `*Ref` soft-reset bug by
  construction). `SpikeRule` = condition function + jump map over **all** states
  (`aeif` `w += b`, Izhikevich `u += d`) + refractory semantics (clamp vs block) +
  optional stochastic hazard (escape noise for `pp_psc_delta`, `gif_*`, `erfc_neuron`;
  consumes RNG) + a peak-detection variant for HH-type models with no reset (NEST
  `hh_*`; the BrainPy-family `HH`, `Izhikevich`, `FitzHughNagumo` and
  `HindmarshRose` need it when they are ported onto `Network`, D-e). The spike condition is checked at every integrator substep, so in-substep
  spikes and resets (today's `event_fn` in RKF45) are preserved; multiple spikes in
  one step are counted, not collapsed to a boolean.
- **In-substep localisation is part of the `SpikeRule` contract** (revised
  2026-10-22). In the substep where the condition becomes true, the rule locates the
  crossing time inside the substep — from the integrator's dense output or by a
  **fixed** number of root-finding iterations (no `while_loop`) — applies the jump at
  that time, and integrates the remainder of the substep. The located offset is an
  output of the step, so RK models get precise spike offsets (usable on `timed_event`
  ports and by EventProp). **Parity target:** as substeps grow, traces *including the
  spike steps* converge to the reference (NEST) and fall within the NEST tolerance
  band (`CAT_A`) at the model's default substeps. K0 met only "spike times equal,
  `CAT_A` off the spike steps" for `aeif_cond_alpha` (first-order localisation, §13);
  the localisation lands in K1b.
- **Integrators are fixed-substep only** (decision I1). Adaptive RKF45
  (`AdaptiveRungeKuttaStep`, a `while_loop`) is removed: `jax.grad` through
  `while_loop` raises, and one stiff neuron sets the iteration count for the whole
  population. Options: `exact` (linear models only — rejected for nonlinear ones),
  `exp_euler`, `RK(order, substeps)`, exponential-Rosenbrock for stiff models; the
  substep count is static per model and overridable. Substeps run as one `lax.scan`
  with every state carried explicitly (§7.3; K0 unrolled them in Python, compile time
  grew with the count). NEST parity for formerly-GSL
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
neuron), `pre` (per source neuron), or `edge` (per edge). **`pre`-aligned synapses are
lowered by the compiler to a hidden receptor population** (revised 2026-10-22): one
population per (source, synapse) with no gids, not visible to observers by gid, not
part of the merge pass, driven by the source's spikes and emitting a `graded` output
into the target port; `connect(..., synapse=NMDA())` is sugar for that lowering. (K0
built NMDA as an explicit receptor population, `_kernel/models/nmda.py`; K1b turns it
into the `SynapseModel` lowering.) Merging `post`-aligned
synapses (the AlignPost O(N_post) result) is a compiler **optimisation pass**: only for
linear kinetics with identical parameters, never for nonlinear kinetics (NMDA
saturation) or `timed_event` ports; any observer or learner that needs per-projection
values (per-source `I_syn`, per-projection gradients) vetoes it; the plan reports which
projections were merged; users can opt out. (K0's merge pass has no observer/learner
veto yet; it arrives with synapse observables in K1b.)

**`PlasticityRule`** — the existing pure-rule protocol, unchanged in spirit, available
to every projection regardless of front-end: declares traces (pre/post, taus, modes),
state reads from the post population, broadcast signals, stochasticity; implements
`init_edge_state()` and `update(edge_state, ctx) -> (edge_state, w_eff)`. Per-edge
parameters are allowed (lifts today's scalar-only limit). Rule parameters may be
`ParamState`s, so plasticity itself is learnable. **Plastic weights and edge state
live in edge (`EdgeSet`) order**; Dense, CSR and CSC are derived compute layouts
(index maps over the edge arrays), so a storage choice can never lose multapses or
reorder weights. K0 supports plastic projections on CSR only, with one scalar delay
and no trainable plastic weights (`_kernel/network.py:525,528`); the edge-order rule
lifts the CSR-only restriction. Hybrid weights in *through-the-rule* mode (L4) are
cheap on the differentiable reference path and come in K1-L; *decomposed* mode later.

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

**`timed_event` capacity** (revised 2026-10-22). The per-target capacity is a
**statistical budget**, not a worst-case bound: the compiler sizes it from the
expected arrivals per step (source rate × indegree × dt, plus a margin for the tail)
and the runtime counts overflowing events per target and step (never a silent drop;
a strict mode raises after the run). K0 checked a worst-case capacity at compile time
(`_kernel/network.py:279-302`) and unrolled the in-step event segments in Python
(`capacity + 2` per step, `_kernel/models/precise.py`), so at 100 neurons a step took
4.8 ms and capacity 64 did not compile within 10 minutes. The segment loop is a `scan`
over the capacity (§7.3). **Status: precise timing is a parity demonstration today,
not a network-scale feature**; K0 also restricts it to CSR storage and event sources.

#### L3 — `Network` graph

```python
net = Network(dt=0.1*u.ms, seed=0)               # no global environ mutation
E = net.add(iaf_cond_exp, 3200, params=...)      # -> PopulationView (gid range)
I = net.add(iaf_cond_exp, 800)
# neuron-owned receptor port 'ex' (filters into g_ex): no synapse model on the projection
net.connect(E, E + I, pairwise_bernoulli(0.02), port='ex',
            weight=Normal(0.6*u.nS, 0.1*u.nS), delay=1.5*u.ms, plasticity=stdp_synapse(...))
# projection-owned kinetics into an unfiltered current port of a BrainPy-family LIF
R = net.add(LIF, 100)
net.connect(E, R, fixed_indegree(50), synapse=Expon(5*u.ms), port='I',
            weight=0.5*u.pA, delay=1.0*u.ms)
rec = net.observe(E[:100], 'spike', 'V_m')
plan = net.compile(precision='float32')          # Plan: validates ports/units/delays
prog = Program(plan, spikes='hard')              # pure L2 contract
res = Runner(prog).simulate(1000*u.ms)           # one compiled scan; Runner is a client
```

(Attaching `synapse=Expon(...)` to `port='ex'` of `iaf_cond_exp` is a validation
error: that port already filters. The previous revision of this example did exactly
that; K1a-3 turns this example into a test.)

`Network` is a **plain declaration**, not a brainstate `Module` (as built in K0): it
holds populations, projections and observers and has no `State`s. `net.compile()`
returns a `Plan`, which **is** a brainstate `Module` (it owns the `State`s: weights as
`ParamState`, dynamics as `HiddenState`, buffers), so the compiled network nests inside
training code. `Network` is inspectable (`net.populations`, `net.projections`,
`net.get_connections(...)`) and serialisable to a plain description (basis for NIR
export).

#### L2 — schedule, delays, runtime

**Step contract** (written, tested, identical for every front-end; revised 2026-10-22
to the contract as built in K0, `_kernel/runner.py`):

```
for each step n:
  for each population P in schedule order
      (insertion order, constrained so that every zero-delay source precedes its target):
    1. deliver   every projection into P reads its source output of step n - d
                 (pre-side ring, d >= 1; or the arrival-ring slot of step n for
                 post-side placement; or, for d = 0, the source output of this step,
                 already updated), applies its synapse model, deposits into P's ports
    2. update    (state, outputs) = P.step(state, port_inputs, ctx)
  3. emit        plastic projections run their rule (NEST lazy timing: pre events d_ax,
                 post events d_den steps late, read before the rings are written) and
                 deposit into their arrival ring; outputs are written to the per-source
                 delay buffers (one per source port, sized max delay, shared by all
                 projections from that source)
  4. observe     observers sample
  5. learn       (optional, §8.2-6) clients of the Program run on-demand updates
```

Deliver/update interleave per population (not "all deliver, then all update") because
zero-delay paths need the source's output of the same step; plasticity runs in the
emit phase because the lazy NEST timing reads the post-synaptic history before the
rings advance.

- Delays mean what the user wrote; the hidden extra lag of today's Simulator
  disappears. On cycles the minimum delay is one step (NEST). On acyclic paths delay
  0 is allowed (decision D4a), and the compiler orders zero-delay paths topologically
  and rejects zero-delay cycles. **What a zero delay covers** (D4a as revised by D-a):
  `event` ports (the jump lands after propagation in the same step), `conductance`
  ports (fed by a same-step synapse model), and `current`/`graded` ports declared
  `latch=False`. A `latch=True` port adds one step by its *port semantics* (NEST
  `CurrentEvent`), not by its delay, so a zero-delay chain through latched ports has
  one step of latency per layer. For the same reason a latched port **breaks** a
  zero-delay cycle (K1a-2): only zero-delay cycles through same-step ports are
  rejected; when the order cannot update the source of a zero-delay edge into a
  latched port first, the compiler *defers* that port (its inputs are gathered after
  every population has updated and only fill the latch), which gives exactly the
  numbers of the source-first order. Measured on K0 (latched everywhere): a 4-layer
  zero-delay chain fires its first spikes at steps 9, 9, 9, 9 through `event` ports but
  10, 11, 12, 13 through Expon → `LIF.I`; after K1a-2 the BrainPy-family `LIF.I`
  (`latch=False`) gives one step for all layers.
- **Program is the L2 contract** (D-c). `Program(plan, spikes='hard'|'relaxed')`
  exposes pure functions over explicit pytrees — `init_state()`, `read_params()`,
  `step(params, state, n, trial, ext)`, `rollout(params, state, n0, trial, n_steps)` —
  and nothing else is needed to run, batch, differentiate or train the network. The
  `Runner` is one client of a `Program` (it keeps the state between calls); learners
  are other clients and take `(Program, state0)` (§8.2-5), so an episode can start from
  any state (K0's BPTT always started at step 0, and learners attached to a `Runner`
  through `plan.runner(learner=…)`).
- Event delays are rounded to the grid (NEST) or carried as precise offsets on
  `timed_event` ports; interpolation is allowed only on `graded` ports. Continuous
  same-step coupling (gap junctions) is explicit-lag by default with an
  optional WFR mode that runs a **fixed** iteration count (differentiable).
- Delay-buffer placement is chosen by the compiler per projection: **pre-side** ring
  (`N_src × D_max`, shared by all projections from that source) for scalar or
  per-source delays; **post-side** arrival ring (`N_tgt × D_max`, scatter on spike —
  NEST/GeNN) for per-edge delays, which keeps delivery event-driven. The plan reports
  the choice and its memory. Per-edge delays are delivered by **one** scatter into the
  arrival ring indexed by (delay slot, post) — not one storage and one matmul per
  distinct delay value as in K0 (`_kernel/compiler.py:154`, build and step time grew
  with the number of distinct delays, §13). The compiler chooses between the scatter
  and delay grouping with a cost model and caps the number of groups (K1a-4); a fused
  brainevent kernel ([brainevent#203](https://github.com/chaobrain/brainevent/issues/203))
  is the future fast path.
- Port inputs are passed as a value into `step` — no dict popping.
- Runner (a `Program` client): `simulate`, `cont`, `step`; host drive for closed loops;
  batching by `vmap` over the state pytree (trials, parameter sweeps); optional
  min-delay epoch chunking (NEST/Arbor model) for long runs.
- Learning: see §8 — learners are chosen per run over the same `Program`.

#### 7.1 Within-step timing contract

Written per port kind (table above) and tested against NEST per kind in K0, so parity
offsets cannot hide a one-step shift: `event` inputs are a jump applied **after** the
continuous propagation of the step they arrive in (NEST ring-buffer semantics);
`current`/`graded` inputs follow their port's `latch` flag (D-a): latched ports act
during the **next** step (NEST models), unlatched ports during the step they arrive in
(BrainPy-family models); `conductance`
inputs enter the ODE and are seen at every substep; `timed_event` inputs are sorted by
offset and the model integrates between them. Plasticity follows NEST's **lazy**
semantics by default (weights updated on pre-spike delivery from the post history,
with dendritic/axonal delay split as in NEST); an eager mode is available for
learning, and observers see the weight of the semantics in use.

#### 7.2 RNG and reproducibility contract

Counter-based keys derived from `(seed, object key, gid block, step)`: every
population, edge set and step gets an independent stream that does not depend on
evaluation order. Required, tested invariances: the realised connectivity is identical
across storages (`dense`/`csr`/`procedural`), batch layouts and shard counts; under
`vmap` the user chooses independent or common random numbers per trial; `cont()` and
checkpoint/restore continue the same streams.

**Layout (D-g, lead decision 2026-10-22; implemented in K1a-6).**

- **Object keys are hashed from the object's name**, not from its insertion index:
  adding, removing or reordering an unrelated population or projection leaves every
  other object's streams unchanged. (K0 folds in the insertion index `uid`, so adding
  an early population reshuffles all later streams.)
- **Dynamics noise is drawn per gid block**: a population's per-step draws (Poisson
  generators, noise ports, stochastic spiking) are generated block by block over fixed
  blocks of 1024 *population-local* ids, so a shard owning some blocks draws exactly the
  numbers it would draw in a single-device run, and a population extended beyond a
  block keeps its neurons' numbers. Inside a plan `StepContext.key` holds the stacked
  per-block keys, shape `(n_blocks, 2)`, `key[b] = fold_in(population_key(seed, name,
  n, b), trial)`; models draw only through `rng.draw_blocked` (lead decision
  2026-10-22, PR #113). (K0 drew whole populations from one key.)
- **Connectivity is keyed by the target population's name and local id** (lead
  decision 2026-10-22, PR #113): the per-target key is `post_keys(fold_in(proj_key,
  name_id(target population)), local id)` (`rng.view_post_keys`, passed as
  `ConnRule.build(..., post_keys=)`), with the same block/lane structure on local ids.
  Global ids shift when an earlier population is added, so global-id keys could not
  give insertion-order invariance.
- **Initial-state, weight and delay distributions** still draw whole arrays, from
  name-keyed streams (insertion-order invariant, not extension/shard invariant);
  block invariance for them is deferred to the K1b distributed spec.
- Block size 1024, the same `GID_BLOCK` constant as connectivity; unnamed objects
  are auto-named `<model>_<k>` per class, so insertion order matters only among
  unnamed objects of the same class (lead, 2026-10-22). The distributed-layout spec
  (K1b) fixes the gid → shard map on these blocks before K2 model ports rely on it.

#### L4 — front-ends

(Revised 2026-10-22, D3 and D-e.)

- **`Network`** (the Python API): `Network` → `Plan` → `Program` is the **only**
  execution path. Model classes are *not* promised to be usable standalone inside an
  arbitrary brainstate module; ML-style work embeds the compiled `Plan`/`Program` (a
  brainstate `Module` and pure functions, respectively) in training code.
- **NEST vocabulary**: `Simulator.create/connect/simulate`, NEST parameter names,
  integer `receptor_type`, `NodeView`, `SynapseCollection` (over `EdgeSet`) — a thin
  layer that builds a `Network`. No routing logic of its own. Built next to K2, when
  the NEST model library is ported.
- **BrainPy-style API: deprecated (D-e).** The standalone modules (`LIF(...)(x)` driven
  by hand inside a brainstate module) and the `AlignPostProj`/`AlignPreProj`-family
  projections get deprecation warnings when the new kernel is public (K3) and are
  removed at K4. Their models are ported onto the `NeuronModel` contract and used
  through `Network`; the AlignPost O(N_post) idea survives as the compiler's merge pass.
  `HH`, `Izhikevich`, `FitzHughNagumo` and `HindmarshRose` need the peak-detection
  `SpikeRule` when ported.
- **Interop** (later): NIR import/export; SONATA import (file-format import only;
  co-simulation is out of scope).

#### 7.3 Compile-time model

Added 2026-10-22 (principle 7). K0 measured compile time growing with model size and
model parameters because model-dependent loops were unrolled in Python inside the
traced step: `aeif_cond_alpha` at 16 RK substeps compiled in 17.3 s (7.8 s at 16 and
52 s at 32 substeps in T3's measurement); `iaf_psc_exp_ps` with `timed_event` capacity
64 did not compile within 10 minutes; per-edge delays built one storage per distinct
delay. Rules:

1. **No Python loop whose trip count depends on the model** inside the traced step:
   substeps, in-step event segments, root-finder iterations, delay groups. Each is a
   `lax.scan` (or `fori_loop` lowered to `scan`) over a static length with the state
   carried explicitly — no Python side channels mutated from inside the loop body
   (K0's `s_holder`, `_kernel/model.py:552`), no Python-int branches on the loop index
   (`k == n_sub - 1`, `_kernel/model.py:550`; peel the last iteration instead).
   Measured: substeps as `scan` give identical `aeif` traces and compile in 1.7 s at
   k = 16 and k = 64, with braintrace D-RTRL gradients unchanged.
2. **Loops over structure are allowed but bounded**: one traced block per population
   and per projection is fine (the graph is static); loops over *distinct parameter
   values* (delay values, receptor indices) are not — they become data.
3. **`while_loop` only where no gradient flows and the cost is bounded**, documented at
   the call site (e.g. the Poisson CDF-inversion tail); never in a neuron step
   (reverse-mode and braintrace both reject it).
4. **Compile time is a gate**: every K0/K1 network compiles in ≤ 5 s at default
   substeps on the reference CPU (K1a-1, K1a-7); compile time is reported next to step
   time in every benchmark.

#### 7.4 Public API, naming and export plan

Added 2026-10-22 (D-d). `brainpy_state._kernel` stays **private until K3**: no
experimental export, no `brainpy.state.kernel` alias, no `__module__ =
'brainpy.state'` on kernel classes. Reasons: the contracts are still moving (K1a
changes ports, `Program`, learners, RNG), and the kernel's names collide with today's
public exports.

**Collisions.** `brainpy_state.__all__` has 212 entries today (210 public symbols:
199 classes, 6 functions, 3 modules, 2 instances). These kernel names already exist
there with different classes behind them (18, counted 2026-10-22): `LIF`, `Expon`,
`MgBlock`, `Network`, `Projection`, `aeif_cond_alpha`, `iaf_cond_exp`,
`iaf_psc_alpha`, `iaf_psc_exp_ps`, `dc_generator`, `poisson_generator`,
`poisson_generator_ps`, `spike_generator`, `static_synapse`, `stdp_synapse`,
`fixed_indegree`, `fixed_total_number`, `pairwise_bernoulli`. Every model ported in K2
adds more.

**Staging.**

| milestone | public surface |
|---|---|
| K1a–K2 | unchanged: today's exports stay (frozen old kernel, D-f); the kernel is reachable only as `brainpy_state._kernel` from tests, benchmarks and specs |
| K3 | the kernel becomes public; BrainPy-style names without a kernel port get `DeprecationWarning`s (D-e); a public-API freeze list (stable / experimental) is published |
| K4 | the old kernel and every deprecated name are deleted |

**Open (lead decision in the K3 spec): how colliding names are staged at K3.**
Options: (a) kernel classes take over the existing names in `brainpy.state` (NEST
names keep NEST spelling) and the old classes move to a deprecated namespace for the
K3→K4 window; (b) the kernel is exported under a new sub-namespace
(e.g. `brainpy.state.kernel`) during K3 and moved to the top level at K4;
(c) kernel names get a suffix. Recommendation: (a) — one public name per concept, and
D-e already deprecates the old surface; (b) doubles the migration for users.

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
   needs the reset Jacobian. Likewise the **forward spike mode is explicit**:
   `Program(plan, spikes='hard'|'relaxed')` — `'hard'` (default) emits exact counts
   with a surrogate backward pass; `'relaxed'` emits smooth spikes with soft jumps and
   a non-detached reset, the regime in which BPTT is the exact derivative of the
   simulated loss on recurrent nets (K0 gate G6). K0 selected this with a module-level
   `ContextVar` read at trace time (`relaxed_spikes()`, `_kernel/spiking.py:89,444`),
   which is invisible in the `Program`'s signature; K1a-5 replaces it. `detach_reset`
   and `spikes` are the two forward/backward options a learner may set.
3. **One synaptic-op layer.** Every edge storage implements forward, a VJP for weights
   and inputs, and a braintrace ETP-primitive registration. Projections never call raw
   matmuls, so every learner sees every weight.
4. **Declared dynamics enable derived learners.** From `derivatives` + `spiking` (D1) the
   kernel builds the surrogate step (BPTT), the adjoint system (EventProp), and exposes
   the step to braintrace's graph analysis. Models that only hand-write `step` declare
   which learners they support.
5. **`Learner` interface at L2 with capability negotiation; learners take
   `(Program, state0)`** (D-c, lead decision 2026-10-22). `BPTT(loss, optimizer,
   checkpoint=…, truncation=…)`, `EventProp(…)`, `Online(algorithm=braintrace.EProp,
   learning_signal=…)`, `ES(…)`, `Readout('ridge' | 'force')` are constructed with a
   `Program` and run episodes from an arbitrary `state0` (a warmed-up state, a
   checkpoint, the state a `Runner` reached), so truncated and continuing training need
   no special path. The `Runner` is just another client of the same `Program`; there is
   no `plan.runner(learner=…)` attach coupling (K0, `_kernel/compiler.py:391`). Local
   plasticity runs inside the step (emit phase, §7 L2). A learner checks model
   capabilities (`differentiable`, `adjoint`, `etrace_compatible`, `jit_safe`) and fails
   with a clear message rather than silently mis-training. The braintrace
   compatibility checker allows braintrace primitives inside static `scan`/`fori_loop`
   bodies (braintrace 0.2.6 supports them) and rejects them only inside `while` bodies
   ([braintrace#186](https://github.com/chaobrain/braintrace/issues/186)).
6. **A learning phase in the schedule.** After *observe*, an optional *learn* phase runs
   on-demand updates at chosen intervals: optimizer steps, weight normalisation,
   learning-signal broadcast, trial resets. It is a real phase of the `Program`'s
   rollout, run inside the compiled loop, not host code between `cont()` calls. K0 has
   none; K1a-5 adds it as `Program.rollout(..., learn=fn, every=k)` (lead, 2026-10-22;
   exact signature approved in the K1a-5 PR).
7. **Ownership.** braintrace owns online algorithms; brainpy.state owns models and
   guarantees they are etrace-compatible. braintrace currently also ships `nn.LIF`,
   `ALIF`, `Expon`, `STP`, … — which library is canonical for basic models is an
   ecosystem decision, resolved by L3: **brainpy.state is canonical** for neuron, synapse and plasticity models; braintrace keeps algorithms and its ETP primitives, and its duplicate `nn` models are deprecated upstream.

#### 8.3 Learning gates per milestone

- **K0 (done, passed — [report](2026-10-22-k0-report.md) G6–G8):** surrogate-BPTT
  gradients match finite differences on a small `LIF` and
  `iaf_psc_alpha` network through dense *and* sparse storage *and* a delay (in the two
  regimes where BPTT is the exact derivative, K0 spec §6 G6);
  `braintrace.compile(net, EProp)` succeeds on a K0 network (update-quality checks move
  to K1-L: OSTL exact on a feed-forward net, D-RTRL vs BPTT on one step); `stdp_synapse` NEST parity.
- **K1a:** learners on `(Program, state0)`; BPTT from a non-zero `state0` matches FD;
  explicit `spikes=`; relaxed etrace checker (§12).
- **K1-L (learning track):** EventProp prototype scoped to LIF with interpolated threshold-crossing times,
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
| **Per-edge delay delivery** fused with the matmul (gather from a source ring buffer by edge delay) | heterogeneous delays without one matmul per delay value | gap — filed as [brainevent#203](https://github.com/chaobrain/brainevent/issues/203); K1a-4 uses a single-op JAX scatter meanwhile |
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
| Compile time | §7.3 rules; ≤ 5 s compile for every K0 network at default substeps; reported in every benchmark | K1a (gate) |
| Performance at scale | GPU measured (K0 measured CPU only); Potjans–Diesmann microcircuit at full scale (memory, build, compile, step; rates vs NEST); sparse recording instead of dense `(T, n)` observers | K1a (GPU, microcircuit), K1b (recording) |
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
| BrainPy-style standalone modules (`LIF(...)(x)` inside a user `Module`) and `AlignPostProj`/`AlignPreProj`/`CurrentProj` projections | deprecated (D-e): models ported onto `NeuronModel`/`SynapseModel` and used through `Network`; old API removed at K4 |
| `Current/Conductance(filtered=True)` (K0) | `Event` port naming the filtered state (D-b) |
| `Builder`/`Network` + `_RuleProj` | removed (subsumed by `Network`) |
| `align_pre/post_projection`, `align_*_ltp` stubs | removed |
| host-only NumPy models | `jit_safe=False` capability → clear compile error; JAX ports tracked as model work |

### 11. Decisions

| # | decision | outcome |
|---|---|---|
| D1 | Model authoring | ✅ **Python classes** with declared `params/states/ports` + `derivatives`; a DSL may come later as a second front-end lowering to the same contract |
| D2 | State substrate | ✅ **brainstate `Module`/`State`**, functional `step` inside |
| D3 | Front-ends | ✅ (2026-10-22, with D-e) **`Network` is the only Python API**; the NEST `Simulator` vocabulary is a thin layer over `Network` (built next to K2); the BrainPy-style front-end is deprecated (D-e) |
| D4 | Delivery semantics | ✅ **NEST-faithful**: no hidden lag; delay ≥ 1 step on cycles |
| D4a | Zero delay on acyclic paths | ✅ **allowed** (topological order within a step); zero-delay cycles rejected. *Revised 2026-10-22 (D-a):* a zero delay is same-step for `event`, `conductance` and unlatched `current`/`graded` ports; a latched port adds one step by port semantics (§7 L2), so only zero-delay cycles through same-step ports are rejected (a latched port breaks the cycle; K1a-2) |
| I1 | Integrators | ✅ **fixed-substep only**; adaptive RKF45 removed; per-model substep defaults set by convergence against NEST |
| S1 | Model scope | ✅ **point neurons only**: multi-compartment and population/mean-field models are not ported and are removed at K4 (list in §13) |
| S1a | Single-unit rate neurons | ✅ **not supported**: `lin_rate`, `tanh_rate`, `threshold_lin_rate`, `gauss_rate`, `sigmoid_rate`, `sigmoid_rate_gg_1998`, `rate_neuron_ipn/opn`, `rate_transformer_node`, with `rate_connection_instantaneous/delayed` and `step_rate_generator` are not ported and are removed at K4 |
| D5 | Rewrite strategy | ✅ **new kernel alongside**, port families, delete old kernel in one 0.2/1.0-alpha break |
| D6 | `spec` branch | ✅ (lead, 2026-10-22): **borrow ideas, not code** — K0 used none of its code; the branch is **archived as a tag at K4** (not now) |
| L1 | Online-learning algorithms | ✅ **live in braintrace**; brainpy.state guarantees etrace compatibility |
| L2 | EventProp | ✅ **K1 prototype**, scoped to interpolated-threshold LIF without refractoriness, homogeneous delays |
| L3 | Canonical home of basic models (LIF/ALIF/Expon/STP exist in both braintrace.nn and brainpy.state) | ✅ **brainpy.state is canonical**; braintrace.nn duplicates to be deprecated upstream in favour of brainpy.state models |
| L4 | Hybrid weights (local plasticity + gradient on the same weight) | ✅ **both modes, user-configured**: decomposed `w_slow + α·w_fast`, or one weight differentiated through the rule |

Post-K0 decisions (lead, 2026-10-22), from the post-K0 architecture review:

| # | decision | date | outcome |
|---|---|---|---|
| D-a | Latch of `current`/`graded` input ports | 2026-10-22 | ✅ **per port**: `latch=True|False` on `Current`/`Graded`, **default `True`** (K0 behaviour; lead, 2026-10-22); NEST models latched (one step, NEST `CurrentEvent`), BrainPy-family models same-step. D4a rewritten to name the port kinds a zero delay covers. K1a-2 |
| D-b | Encoding of neuron-owned receptor ports | 2026-10-22 | ✅ **one encoding**: an `Event` input port that names the state it filters into; replaces `filtered=True` on `Current`/`Conductance` and the special case in `_kernel/ports.py`. K1a-3 |
| D-c | Learner API | 2026-10-22 | ✅ **learners take `(Program, state0)`**; the `Runner` is just another client; no `plan.runner(learner=)` attach coupling (also removes BPTT's step-0-only episodes). K1a-5 |
| D-d | Public API of the new kernel | 2026-10-22 | ✅ **`_kernel` private until K3**; no experimental export (§7.4) |
| D-e | BrainPy-style API | 2026-10-22 | ✅ **deprecate all of it**: standalone modules and AlignPost/AlignPre projections; the "usable standalone inside any brainstate module" promise is dropped; `Network` → `Program` is the only execution path. Ported models used through `Network`; removal at K4 |
| D-f | Old kernel | 2026-10-22 | ✅ **frozen until deleted at K4**: no 0.1.x releases, no old-kernel fixes; v0.1.1 goals 01/03/04 dropped, goal 02 reduced to the CI that serves the new kernel |
| D-g | RNG layout | 2026-10-22 | ✅ **name-hashed object keys + gid-blocked dynamics noise** (shard- and insertion-order-invariant, §7.2); implemented in K1a-6, confirmed by the distributed spec (K1b) before model ports rely on it |

### 12. Migration plan

Rewritten 2026-10-22 after K0. The old kernel is **frozen** until it is deleted at K4
(D-f): no 0.1.x releases and no fixes to it; its tests keep running as the reference
for porting. The new kernel stays private until K3 (D-d).

| milestone | content | exit gate | status |
|---|---|---|---|
| **K0 — kernel spike** | L1 contracts, `Network`, compiler `Plan`, step schedule, delay buffers, `Program`/`Runner`, RNG contract; `iaf_psc_alpha`, `LIF`, `iaf_cond_exp`, `aeif_cond_alpha`, `iaf_psc_exp_ps` + `poisson_generator_ps`, NMDA/Mg-gated conductance, `Expon`, `static`/`stdp_synapse`; BPTT learner and braintrace checks ([K0 spec](2026-10-21-k0-kernel-spec.md)) | gates G1–G10 (K0 spec §6) | ✅ **done — GO** ([K0 report](2026-10-22-k0-report.md)); G9 measured on CPU only, G5/G6 under restated criteria |
| **K1a — kernel hardening** | fix what K0 showed to be structurally weak before more is built on it ([K1a spec](2026-10-22-k1a-kernel-hardening-spec.md)): substeps and precise-model segments as `scan` (§7.3); per-port latch (D-a); one receptor-port encoding (D-b); single-op per-edge delays; learner API on `(Program, state0)` (D-c), explicit `spikes='hard'|'relaxed'`, relaxed etrace checker, learn phase; RNG layout (D-g); CI job for the gate marker and the `brian2` extra | every K0 network compiles in ≤ 5 s at default substeps; per-edge-delay Brunel ≤ 1.5 × scalar-delay step time and ≤ 2 × scalar-delay build time; blocked RNG ≤ 1.05 × K0 step time; GPU G9 rerun (f32 default backend, f64 cusparse) vs the T0 rows; Potjans–Diesmann microcircuit at full scale on CPU and GPU (memory, build, compile, step; rates vs NEST 3.10, 5 seeds each side); G2 rerun in the asynchronous-irregular regime; all K0 gates still green; K1a report | **next** |
| **K1b — substrate completion** | graded emission; gap junctions (lag + fixed-iteration WFR); STP; procedural (and conv) storage; `pre`-aligned `SynapseModel` lowering (NMDA as `synapse=NMDA()`); in-substep spike localisation (`SpikeRule` contract, §7 L1); merge-pass veto with synapse observables; plastic weights in edge order (lifts CSR-only plasticity; per-edge delays on plastic projections); `timed_event` capacity as a statistical budget with run-time overflow (timed sources with synapse models and plasticity); recording at scale (sparse spike buffers, downsampled multimeters, online reductions); checkpoint/restore; stimulus schedules and sweeps; distributed-layout spec incl. the D-g RNG block layout; fused per-edge-delay delivery (brainevent#203) behind the K1a-4 cost model, closing the K1a-4 known gap (Brunel order 2500, delays ≤ 20 ms: CPU 1.39-1.55 ×, GPU f32 1.63-1.84 × the scalar-delay step; target ≤ 1.5 ×); check `PreRing` for the whole-ring copy per step (§13) | every old `_nest_network` behaviour test re-expressed on `Network` and green, or listed as removed/exempted (inventory as in K2); spiking `aeif_cond_alpha` within `CAT_A` on spike steps at default substeps | planned (after K1a) |
| **K1-L — learning track** (parallel to K1b; depends on K1a only) | EventProp prototype (L2 scope); three-factor learning through braintrace with a modulatory device; hybrid weights through the rule (L4) and learned STDP parameters; ES via `vmap`; ridge/FORCE readouts; batched online learning; spike-time gradients through timed delivery; deferred from K1a-5: per-step online updates inside the compiled loop (`Online.run` is one optimizer step per episode), learn phase on batched runs (`Runner.cont(learn=)` / batched `Program.rollout` learn raises `NotImplementedError`) | §8.3 K1 gates (EventProp vs FD on its scope; ThreeFactor; meta-gradient; ES; readout); OSTL exact on a feed-forward net; D-RTRL vs BPTT on one step | planned (after K1a) |
| **K2 — model library + NEST layer** | port neuron/synapse/plasticity/device families onto the contracts (numerics reused; goal-01 repro tests as acceptance tests; host-only models ported to JAX or flagged); BrainPy-family models ported for use through `Network` (D-e; peak-detection `SpikeRule` for `HH`, `Izhikevich`, `FitzHughNagumo`, `HindmarshRose`); **NEST `Simulator` vocabulary as a thin layer over `Network`** (`NodeView`, `SynapseCollection` over `EdgeSet`, integer `receptor_type`) — moved here from K3 because the validation tests drive it | **every one of the 131 NEST validation test files is re-expressed on the new kernel and green, or listed as removed (S1/S1a) or exempted with a written reason**. Inventory first (one table, committed with the K2 spec): ~75 drive the old `Simulator` (re-expressed through the NEST layer), ~23 cover models removed under S1/S1a (removed), 8 `spatial_*` need K3 (exempted until K3). Tolerances of formerly-GSL models are **re-derived** from substep-convergence studies against NEST and documented, never silently loosened | planned |
| **K3 — spatial, interop, public export** | spatial layers/masks/kernels on `Network` (`conv` storage); NIR export (and import); SONATA import; **public export of the kernel** with the naming/staging plan of §7.4; deprecation warnings on the BrainPy-style API (D-e); public-API freeze list; examples and docs migrated | the 8 `spatial_*` validation tests green; NIR round trip; docs build with every example on the new kernel | planned |
| **K4 — removal + release** | delete the old kernel (`_nest_network` `Simulator`, `Builder`, dead projections) **and the BrainPy-style API** (standalone modules, AlignPost/AlignPre projections; D-e); remove S1/S1a models; archive the `spec` branch as a tag (D6); migration guide; 1.0-alpha | full suite + docs build + benchmarks | planned |

Goal-01 findings map as follows: autapse ids, edge retention, multiplicity,
`fixed_total_number` (mirror NEST: refuse `allow_multapses=False`) were fixed **by
construction** in K0 (gate G4); `*Ref` soft reset, `MgBlock`, AMPA/GABAa saturation
(make `g` a dimensionless open fraction with an explicit `g_max`), `OUProcessInput`
(exact discretisation, σ = stationary std), `LeakyRateReadout`, `LogNormal` (`unit=`),
receptor-type validation and the namespace env var are fixed **while porting** (K2),
with the verified repro tests as acceptance criteria. They are not fixed in the
frozen old kernel (D-f).

### 13. Risks

- **Scope.** ~60 NEST neuron modules + the BrainPy family must be ported. Mitigation:
  K0 go/no-go gate (passed); numerics reused verbatim; mechanical ports per family.
  Removed under S1 (point neurons only), not ported: multi-compartment `cm_default`,
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
  and off-spike traces match NEST. Mitigation: in-substep localisation as part of the
  `SpikeRule` contract (§7 L1), K1b.
- **Compile time** (added 2026-10-22). Model-dependent Python loops in the traced step
  made compile time grow with substeps (`aeif` 17.3 s at k = 16), `timed_event`
  capacity (capacity 64 did not compile in 10 min) and the number of distinct delays.
  Mitigation: §7.3 rules; `scan` over substeps measured at 1.7 s for k = 16 and 64;
  ≤ 5 s compile gate in K1a.
- **GPU performance unmeasured** (added 2026-10-22). K0's G9 ran on CPU only (user
  decision); the old `Simulator` was launch-bound on GPU (≈ 0.4 ms/step even at 1000
  neurons) and float64 with the default brainevent backend aborts
  ([brainevent#199](https://github.com/chaobrain/brainevent/issues/199)). No GPU claim
  is made until K1a-7 measures it (f32 default backend, f64 cusparse) together with the
  Potjans–Diesmann microcircuit at full scale. Memory above order 2500 is likewise
  unmeasured (dense `(T, n)` observers).
- **Per-edge delays** (added 2026-10-22). K0 builds one storage per distinct delay
  value: at 2000 neurons and indegree 100, scalar delay = 1 group, 1.2 s build,
  1.06 ms/step; delays up to 2 ms = 20 groups, 5.3 s, 2.9 ms/step; up to 5 ms = 50
  groups, 11 s, 5.4 ms/step; up to 20 ms = 200 groups, 26 s, 4.2 ms/step. Mitigation:
  single-op scatter with a cost model and a group cap (K1a-4); fused kernel upstream
  ([brainevent#203](https://github.com/chaobrain/brainevent/issues/203)).
  K1a-4 result (lead decision 2026-10-22: gate met as written): on the gate network
  the scatter is 0.87-0.96 × the scalar-delay step (CPU and GPU, f32/f64). **Known
  gap:** Brunel order 2500 with delays up to 20 ms is CPU 1.39-1.55 × (10-20 MB ring,
  cache-bound random writes) and GPU f32 1.63-1.84 × (about ten small kernels and one
  `lax.cond` sync per projection vs one CSR kernel); tracked on brainevent#203, picked
  up in K1b. Follow-up: `PreRing` (read then write of the carried ring) is suspected of
  the same whole-ring copy per step that K1a-4 removed from `PostRing` with an
  `optimization_barrier`; unmeasured.
- **Gate strength** (added 2026-10-22). Several K0 gates passed on weak evidence: G2
  ran Brunel in a regular oscillatory regime (CV-ISI 0.083), not the asynchronous-
  irregular one; G3's correlation passed by a narrow margin (Δ 0.00506 vs bound
  0.00569); G6 does not cover Expon, conductance ports or plasticity; G7 covers only
  `LIF` with event ports; G9 has no GPU row. Mitigation: K1a-7 reruns G2 in the AI
  regime and adds GPU and microcircuit gates; K1-L and K2 widen G6/G7 to the new port
  kinds and models.
- **Performance regression** from generality (ports, schedule). Mitigation: G9 passed
  on CPU (step 0.53–0.77 × old at order 2500); GPU and scale gates in K1a.
- **Parity drift** from the delay-semantics change (D4) and the per-port latch (D-a).
  Mitigation: NEST is already delay ≥ 1 step; the per-port timing contract (§7.1) is
  tested per kind (G1), so `align_steps` offsets cannot hide a one-step shift; NEST
  models stay latched, so NEST parity is unchanged by D-a.
- **Upstream kernel gaps** (brainevent, braintrace). Mitigation: JAX reference paths in
  this package; kernels proposed and landed upstream (§8.4); every workaround is listed
  with its removal condition in
  [`2026-10-22-upstream-issues-todo.md`](2026-10-22-upstream-issues-todo.md).
- **brainstate churn** (0.6.0 removed the input API mid-cycle). Mitigation: the kernel
  owns ports/inputs; depend only on `State`, `Module`, transforms.
- **Users of the BrainPy-style API** (added 2026-10-22, D-e). Deprecating the whole
  API affects every existing user script and the docs/examples built on it.
  Mitigation: the API keeps working (frozen) until K4; deprecation warnings only once
  the replacement is public (K3); migration guide and migrated examples in K3/K4.

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
