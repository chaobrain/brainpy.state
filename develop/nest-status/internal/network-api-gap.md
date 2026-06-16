# Network API — NEST parity gap

## 1. Scope

PyNEST top-level functions exposed by `import nest`: connection handling
(`Connect`, `TripartiteConnect`, `Disconnect`, `GetConnections`), node handling
(`Create`, `GetNodes`, `NodeCollection`), model management (`CopyModel`,
`Models`, `GetDefaults`, `SetDefaults`, `ConnectionRules`), simulation control
(`Simulate`, `Prepare`, `Run`, `Cleanup`, `ResetKernel`, `RunManager`,
`EnableStructuralPlasticity`), kernel configuration (`SetKernelStatus`,
`GetKernelStatus`, `Install`), status/info (`GetStatus`, `SetStatus`,
`get_verbosity`, …), parallel computing (`NumProcesses`, `Rank`, …), data
types (`NodeCollection`, `SynapseCollection`, `Parameter`, `Mask`,
`CollocatedSynapses`, `Compartments`, `Receptors`), random-number factories
(`nest.random.*`), parameter operators (`nest.math.*`, `nest.logic.*`),
spatial/topology (`nest.spatial.*`, `nest.spatial_distributions.*`, masks,
spatial query/inspection, visualization), SONATA networks (`SonataNetwork`).
Plus the 10 connection rules used in `nest.Connect(... conn_spec={'rule': '...'} )`.

Upstream reference:
- <https://nest-simulator.readthedocs.io/en/stable/ref_material/pynest_api/index.html>
- <https://nest-simulator.readthedocs.io/en/stable/synapses/connectivity_concepts.html>
- <https://nest-simulator.readthedocs.io/en/stable/networks/spatially_structured_networks.html>

Catalog references: `nest-catalog-snapshot.md` §§8-12.

Evidence basis (as of 2026-06-16): the PyNEST-flavored facade now ships as the
`brainpy.state.Simulator` builder/runner (`brainpy_state/_network/_simulator.py`)
plus the named connection rules in `brainpy_state/_network/_rules.py`. The
imperative entry points are exposed as `Simulator` methods rather than module-level
functions — `sim.create(...)` (NEST `Create`), `sim.connect(...)` (`Connect`),
`sim.tripartite_connect(...)` (`TripartiteConnect`), `sim.get_connections(...)`
(`GetConnections`, returning a `SynapseCollection`), `sim.simulate(...)` /
`sim.cont(...)` / `sim.reset_rollout()` (the simulation rollout), and per-population
`spikes()`/`trace()` readouts. PyNEST references inside `_test.py` files still act as
live-NEST comparison harnesses (e.g. `aeif_cond_alpha_test.py:448` calls
`nest.Create`/`nest.Connect`/`nest.Simulate` to validate against the repo's own model).
The lower-level compositional primitives remain available too:
`brainpy_state/_brainpy/projection.py` (`AlignPostProj`, `DeltaProj`, `CurrentProj`,
`align_pre_projection`, `align_post_projection`) and brainstate (`FixedNumConn`,
`EventFixedNumConn`, `SparseLinear`).

## 2. Parity summary

The core PyNEST connection/simulation surface **ships** as the
`brainpy.state.Simulator` facade. The imperative `Create → Connect → Simulate`
flow that PyNEST users expect is exposed as `Simulator` methods
(`create`/`connect`/`simulate`/`cont`/`get_connections`/`tripartite_connect`),
layered over the compositional JAX/brainstate primitives. The named connection
rules, the `SynapseCollection` introspection view, arbitrary explicit edge lists,
and the full spatial/topology surface are all in place and validated against live
NEST. What remains is a **named residual** — chiefly the `nest_compat`
string-model-name idiom, lazy `Parameter` expressions, `CopyModel`, and
`CollocatedSynapses` — plus the permanently-unsupported items (MPI / MUSIC /
SONATA / structural plasticity / kernel-state serialization).

Design note: `brainpy.state` still has no global mutable kernel. The facade is a
per-`Simulator` builder, so the imperative flow is reproduced object-locally
rather than via NEST's global `ResetKernel`/`SetKernelStatus` state. This is the
intended design, not a gap.

| Bucket | Count | Notes |
|---|---:|---|
| implemented | ~75 | The `Simulator` facade (`create`/`connect`/`tripartite_connect`/`get_connections`/`simulate`/`cont`), named connection rules, `SynapseCollection`, `explicit_edges`, and the full spatial/topology surface (masks, distributions, queries, dump/plot) |
| partial | ~6 | `Mask`/`Compartments`/`Receptors` exposed but not as top-level NEST-named classes; `GetStatus`/`SetStatus` covered by direct attribute access rather than dict API |
| missing | ~12 | The named residual: `nest_compat` string-model naming, lazy `Parameter` expressions, `CopyModel`, `CollocatedSynapses`, `symmetric_pairwise_bernoulli`, `conngen` (CSA), a few enumeration/info conveniences |
| unsupported | ~8 | MPI parallel (`NumProcesses`, `Rank`, `GetLocalVPs`, `SetAcceptableLatency`, `SetMaxBuffered`, `SyncProcesses`, `Install` for dynamic modules), SONATA file loader (`SonataNetwork`), structural plasticity, `store_restore_network` |
| **total PyNEST API + spatial surveyed** | **≈ 105** | per snapshot §§8-12 |

## 3. Evidence-backed mapping table

### 3.1 Connection handling

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `Connect(pre, post, conn_spec, syn_spec)` | **done** | `Simulator.connect(pre, post, rule=, weight=, delay=, synapse=, ...)` (`brainpy_state/_network/_simulator.py`) | <https://nest-simulator.readthedocs.io/en/stable/ref_material/pynest_api/index.html> | `_network/_simulator_test.py`, `_network/_rules_test.py` | Facade method on a `Simulator` builder; `rule=` takes the named rules (§3.9), `synapse=` a plastic spec. Lower-level `AlignPostProj` subclasses remain available |
| `TripartiteConnect(pre, post, third, conn_spec, third_factor_conn_spec, syn_specs)` | **done** | `Simulator.tripartite_connect` (cluster 24) | upstream | `tripartite_connect_test.py`, `astrocyte_small_network_test.py`, `astrocyte_brunel_test.py` | One shared primary `pre→post` sample feeds three arms (primary / `third_in` / `third_out`-`sic_connection`); single-population views. Live-NEST parity: block edge sets bit-identical, random pools match seed-mean counts (cat D) |
| `Disconnect(...)` | missing | none | upstream | — | No first-class disconnect; user removes the `Projection` object |
| `GetConnections(source, target, synapse_model, synapse_label)` | **done** | `Simulator.get_connections(source=, target=, synapse=)` → `SynapseCollection` (`brainpy_state/_network/_connection_introspection.py`, cluster 23) | upstream | `_network/_connection_introspection_test.py` | Lazy filtered edge view; `.get('weight'/'delay')` reads live (post-sim) values, `.set('weight', arr)` writes per-edge — no need to hold each `Projection` handle |

### 3.2 Node handling

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `Create(model, n=1, params=None, positions=None)` | partial | `Simulator.create(model_cls, size, params=, positions=)` (`brainpy_state/_network/_simulator.py`); also `Module()` constructors directly, e.g. `iaf_psc_alpha(n, **params)` | upstream | `_network/_simulator_test.py`, `_network/_simulator_spatial_test.py` | The facade method ships incl. `positions=spatial.*` (goal 20). Only residual: NEST passes a **string** model name; `create` takes the model *class*. The string-name idiom is the `nest_compat` residual |
| `GetNodes(properties)` | missing | brainstate iteration via `.nodes()` | upstream | — | brainstate's tree traversal exposes nodes but not by NEST property dict |
| `GetLocalNodeCollection(nc)` | missing | n/a | upstream | — | MPI-specific (spec §7 unsupported) |
| `PrintNodes()` | missing | brainstate `Module.print_brief()`/`__repr__` | upstream | — | informational only |
| `NodeCollection` | missing | brainstate `Module` instances are themselves the population container | upstream | — | NEST `NodeCollection` supports `+`, slicing, GetStatus/SetStatus; brainpy.state populations don't expose set algebra natively |

### 3.3 Model management

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `CopyModel(existing, new, params)` | missing | Python subclass + parameter override | upstream | — | NEST users do `CopyModel('stdp_synapse', 'layer1_stdp', {'Wmax': 90.})`; brainpy.state users subclass + set defaults |
| `Models(mtype)` | missing | `dir(brainpy_state)` filtered | upstream | — | enumeration helper missing |
| `GetDefaults(model)` / `SetDefaults(model, params)` | missing | class-attribute mutation | upstream | — | NEST's default-mutation pattern has no brainpy.state equivalent — defaults live in `__init__` signature |
| `ConnectionRules()` | missing | n/a — rules are named values in `brainpy.state` (`all_to_all`, `fixed_indegree`, …) | upstream | — | The rules themselves ship (§3.9); only the string-keyed *enumeration registry* is absent (named residual) |

### 3.4 Simulation control

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `Simulate(t)` | **done** | `Simulator.simulate(duration)` (`brainpy_state/_network/_simulator.py`) | upstream | `_network/_simulator_test.py`, `_network/_simulator_analog_test.py` | The facade runs the internal stepping loop (`brainstate.transform.for_loop`) and returns a `SimulationResult` (`spikes()`/`trace()`); the user no longer hand-rolls the loop |
| `Prepare()` / `Run(t)` / `Cleanup()` / `RunManager()` | partial | `Simulator.reset_rollout()` + repeated `Simulator.cont(duration)` | upstream | `_network/_simulator_cont_test.py` | `cont` is a persistent chunked rollout: state persists across calls so host-side work (rewrite drives, overwrite weights) can interleave between windows. No separate `Prepare`/`Cleanup` ceremony |
| `ResetKernel()` | n/a (by design) | construct a new `Simulator` | upstream | — | brainpy.state has no global kernel; lifecycle is per-`Simulator` object, and `reset_rollout()` restarts a rollout cleanly |
| `EnableStructuralPlasticity()` / `DisableStructuralPlasticity()` | missing | n/a | upstream | — | structural plasticity not supported |

### 3.5 Kernel configuration

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `SetKernelStatus(params)` | missing | `brainstate.environ.set(...)` for some params (e.g. `dt`); others unsupported | upstream | — | brainpy.state config is split between brainstate environ and per-model args |
| `GetKernelStatus(keys)` | missing | `brainstate.environ.get(...)` | upstream | — | partial overlap |
| `Install(module)` | unsupported | n/a | upstream | — | dynamic NEST-module loading not in scope for brainpy.state |

### 3.6 Status / info

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `GetStatus(nodes, keys)` | missing | direct attribute access on the Module | upstream | — | brainpy.state uses Python attributes (e.g. `neuron.V_m`) not status dicts |
| `SetStatus(nodes, params)` | missing | direct attribute assignment | upstream | — | NEST users expect `nest.SetStatus(nc, {'V_m': v})`; brainpy.state users do `neuron.V_m.value = v` |
| `get_verbosity()` / `set_verbosity()` | missing | Python logging | upstream | — | trivial to add a shim |
| `authors()` / `help()` / `helpdesk()` / `sysinfo()` / `get_argv()` | missing | n/a | upstream | — | NEST-only conveniences |

### 3.7 Parallel computing

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `NumProcesses()` / `Rank()` / `GetLocalVPs()` / `SyncProcesses()` | unsupported | JAX `jax.device_count()`, `jax.process_index()` | upstream | — | MPI model not in scope (spec §7); JAX device sharding is the design intent |
| `SetAcceptableLatency()` / `SetMaxBuffered()` | unsupported | n/a | upstream | — | MUSIC-specific |

### 3.8 Data types / classes

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `NodeCollection` | missing | Module instance | upstream | — | concatenation (`+`), slicing, position lookup missing |
| `SynapseCollection` | **done** | `SynapseCollection` returned by `Simulator.get_connections` (`brainpy_state/_network/_connection_introspection.py`, cluster 23) | upstream | `_network/_connection_introspection_test.py` | Lazy filtered edge view with `.source`/`.target`/`.get(key)`/`.set(key, value)`; reads live (post-sim) weights/delays without holding each `Projection` |
| `Parameter` (runtime-evaluated expressions) | missing | none | upstream | — | NEST's `Parameter` allows `weight=nest.random.normal()` at `Connect` time; brainpy.state expects concrete values at projection-instantiation time. Part of the named residual |
| `CreateParameter(type, params)` | missing | none | upstream | — | parameter-distribution factory (tied to `Parameter` above) |
| `Mask` (spatial) | **done** | `spatial.circular` / `spherical` / `box` / `rectangular` / `doughnut` / `elliptical` / `ellipsoidal` | upstream | `_nest_spatial/_masks_test.py`, `_validation/spatial_masks_test.py` | all seven mask shapes done with live-NEST node-set parity (incl. rotated + 3-D); see §3.10 |
| `CollocatedSynapses(...)` | missing | multiple `Projection` instances on the same pre/post pair | upstream | — | NEST users compose multiple synapse types on one edge at once; brainpy.state requires separate Projection objects |
| `Compartments` / `Receptors` | partial | `brainpy_state/_nest/cm_default.py` exposes compartment+receptor spec internally | upstream | — | not a top-level class — users compose by passing dicts to `cm_default` constructor |
| `SonataNetwork(config)` | unsupported (per spec §7) | none | upstream | — | declarative SONATA loading out of scope; the SONATA HDF5 format itself could be imported by user code, but the NEST loader semantics are NEST-internal |
| `serialize_data` / `to_json` | missing | brainstate state PyTree serialization | upstream | — | rough equivalent for state, not for the full network |

### 3.9 Connection rules

| NEST rule | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `one_to_one` | **done** | `brainpy.state.one_to_one` (`brainpy_state/_network/_rules.py`) | <https://nest-simulator.readthedocs.io/en/stable/synapses/connectivity_concepts.html> | `_network/_rules_test.py`, `_network/_connectivity_test.py` | Named rule value passed as `connect(rule=)` |
| `all_to_all` | **done** | `brainpy.state.all_to_all` (`_network/_rules.py`) | upstream | `_network/_rules_test.py`, `_network/_connectivity_test.py` | The default `connect(rule=)`; honours `allow_autapses` for `pre is post` |
| `pairwise_bernoulli(p)` | **done** | `brainpy.state.pairwise_bernoulli(p)` (`_network/_rules.py`) | upstream | `_network/_rules_test.py`, `_network/_connectivity_test.py` | Per-ordered-pair Bernoulli(`p`); honours `allow_autapses`/`allow_multapses` |
| `symmetric_pairwise_bernoulli(p)` | missing | none | upstream | — | bidirectional Bernoulli has no analog (named residual) |
| `pairwise_poisson(pairwise_avg_num_conns)` | missing | none | upstream | — | Poisson-distributed counts per pair (named residual) |
| `fixed_total_number(N)` | **done** | `brainpy.state.fixed_total_number(N)` (`_network/_rules.py`) | upstream | `_network/_rules_test.py`, `_network/_connectivity_test.py` | Exactly `N` edges drawn uniformly over the `(pre, post)` grid |
| `fixed_indegree(indegree)` | **done** | `brainpy.state.fixed_indegree(K)` (`_network/_rules.py`) | upstream | `_network/_rules_test.py`, `_network/_connectivity_test.py` | Each post neuron gets exactly `K` edges; named rule, honours autapse/multapse flags. (`brainstate.nn.FixedNumConn` remains the lower-level primitive) |
| `fixed_outdegree(outdegree)` | partial | internal `sample_fixed_outdegree` (`_network/_connectivity.py`), used by `_projections.py` | upstream | `_network/_projections_test.py` | The out-degree sampler exists and is wired through the projection builders, but is **not** exposed as a named top-level `connect(rule=)` value the way the in-degree rule is |
| `conngen` (CSA) | missing | none | upstream | — | Connection Set Algebra integration absent (named residual). `explicit_edges(pre_idx, post_idx)` covers arbitrary precomputed edge lists as one sparse projection (cluster 26) |
| `third_factor_bernoulli_with_pool` | **done** | `brainpy.state.third_factor_bernoulli_with_pool(p, pool_size, pool_type)` (cluster 24) | upstream | `_network/_rules_test.py`, `_network/_connectivity_test.py`, `_validation/tripartite_connect_test.py` | `tripartite_connect` astrocyte-pool rule; `block` + `random` pools. Live-NEST parity in `_validation/tripartite_connect_test.py` |

### 3.10 Spatial / topology

| NEST entity | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `nest.spatial.grid(shape, extent, ...)` | **done** | `brainpy.state.spatial.grid` | <https://nest-simulator.readthedocs.io/en/stable/networks/spatially_structured_networks.html> | `_nest_spatial/_layers_test.py`, `_validation/spatial_grid_test.py` | exact coord parity vs NEST (2-D + 3-D) |
| `nest.spatial.free(pos, ...)` | **done** | `brainpy.state.spatial.free` | upstream | `_nest_spatial/_layers_test.py`, `_validation/spatial_3d_test.py` | array- or distribution-backed (2-D/3-D) |
| `nest.spatial.pos.{x,y,z}` / `source_pos.{x,y,z}` / `target_pos.{x,y,z}` | **done** | `spatial.pos` / `spatial.source_pos` / `spatial.target_pos` (`.x/.y/.z`) | upstream | `_nest_spatial/_kernels_test.py` | per-axis position expressions consumable by kernels; `source_pos`/`target_pos` bind in Connect, `pos` is a per-node accessor |
| `nest.spatial.distance` / `.x/.y/.z` | **done** | `brainpy.state.spatial.distance` (+ `.x/.y/.z`, `displacement`, `pairwise_distance`) | upstream | `_nest_spatial/_distance_test.py`, `_kernels_test.py` | scalar sentinel + per-axis `.x/.y/.z` (absolute, NEST convention); read-back parity vs live NEST |
| `nest.spatial_distributions.exponential / gaussian / gaussian2D / gabor / gamma` | **done** | `spatial.gaussian` / `exponential` / `gamma` / `gabor` / `gaussian2D` | upstream | `_nest_spatial/_kernels_test.py`, `_validation/spatial_{gaussian_kernel,exponential,gamma,gabor}_test.py` | all five match live NEST element-wise to machine precision (weight read-back) |
| `Mask` types (rectangular, circular, doughnut, elliptical, grid, box, spherical, ellipsoidal) | **done** | `spatial.circular` / `spherical` / `box` / `rectangular` / `doughnut` / `elliptical` / `ellipsoidal` | upstream | `_nest_spatial/_masks_test.py`, `_validation/spatial_masks_test.py` | hard-cutoff node-set parity vs live NEST incl. rotated (`azimuth`/`polar`) + 3-D; `box` is 2-D/3-D here vs NEST 3-D-only (`rectangular` is the 2-D box) |
| `GetPosition`, `GetSourcePositions`, `GetTargetPositions`, `FindNearestElement`, `FindCenterElement`, `Displacement`, `Distance`, `SelectNodesByMask` | **done** | `Simulator.get_position`, `spatial.target_positions` / `target_nodes` / `center_element` / `nearest_element` / `Distance` / `select_nodes_by_mask` | upstream | `_nest_spatial/_helpers_test.py`, `_validation/spatial_{grid,queries}_test.py` | `nearest_element`/`select_nodes_by_mask` match live NEST exactly (node-index/node-set) |
| `DumpLayerConnections`, `DumpLayerNodes` | **done** | `spatial.dump_layer_nodes` / `spatial.dump_layer_connections` | upstream | `_nest_spatial/_helpers_test.py`, `_validation/spatial_queries_test.py` | text parity vs `DumpLayer*` (coords/weight/delay/displacement identical; local 0-based vs NEST 1-based ids) |
| `PlotLayer`, `PlotTargets`, `PlotSources`, `PlotProbabilityParameter` | **done** | `spatial.plot_layer` / `plot_targets` / `plot_sources` / `plot_probability_parameter` | upstream | `_nest_spatial/_plot_test.py` | matplotlib lazily imported; smoke-tested (returns a `Figure`) |
| `Create(model, positions=spatial.*)` + `spatial` pairwise-Bernoulli rule | **done** | `Simulator.create(positions=)` + `spatial.spatial_pairwise_bernoulli` | upstream | `_network/_simulator_spatial_test.py`, `_nest_spatial/_rule_test.py` | coords attach to a population; spatial rule rides `connect(rule=)` unchanged |

### 3.11 Random / math / logic Parameter operators

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `nest.random.uniform/uniform_int/normal/lognormal/exponential` | missing | `jax.random.*` / `numpy.random.*` | upstream | — | concrete RNG factories exist in JAX/NumPy; NEST's Parameter-binding semantics absent |
| `nest.math.exp/sin/cos/min/max/redraw` | missing | `jnp.*` | upstream | — | usable directly but only on concrete arrays, not on Parameter trees |
| `nest.logic.conditional` | missing | `jnp.where` | upstream | — | likewise |

## 4. Missing or incomplete functionality

**Programming-model gap — resolved by the `Simulator` facade.** NEST is an
imperative simulator controlled by global state
(`ResetKernel`/`SetKernelStatus`/`Simulate`); `brainpy.state` is a compositional
JAX library. The `brainpy.state.Simulator`
(`brainpy_state/_network/_simulator.py`) reconciles the two: it exposes the
imperative `create → connect → simulate`/`cont` flow as methods on a
per-`Simulator` builder, running the internal stepping loop and returning a
`SimulationResult`. The remaining migration work is **convenience, not
capability**:

1. A thin `brainpy_state.nest_compat` layer over the facade for the last
   string-keyed idioms (model-by-name `Create('iaf_psc_alpha', ...)`,
   lazy `Parameter` expressions, `CopyModel`) — the named residual below.
2. A **porting guide** mapping NEST scripts onto `Simulator` calls.

**Connection rules — named values now ship.** The single-population NEST rules
`all_to_all`, `one_to_one`, `fixed_indegree`, `pairwise_bernoulli`, and
`fixed_total_number` are exposed as named values in `brainpy.state` (passed as
`connect(rule=)`); `explicit_edges(pre_idx, post_idx)` covers arbitrary
precomputed edge lists (cluster 26), and `third_factor_bernoulli_with_pool`
drives the tripartite arm (cluster 24). All honour `allow_autapses`/
`allow_multapses`. Residual rules with no named analog:
`symmetric_pairwise_bernoulli`, `pairwise_poisson`, and `conngen` (CSA).
`fixed_outdegree` exists only as an internal sampler (`_connectivity.py`), not
yet as a named top-level rule.

**Parameter expressions.** NEST's `Parameter` type lets `Connect` accept
runtime-evaluated expressions like `weight=nest.random.normal(mean=1.0, std=0.5)`
or `weight=2 * nest.spatial.distance`. `brainpy.state` requires concrete arrays
at projection-instantiation time. This breaks NEST examples that use
`Parameter` arithmetic.

**Spatial / topology.** *Complete (goals 20 + 27).* The full `nest.spatial` +
`nest.spatial_distributions` surface lives under `brainpy.state.spatial`, validated against live
NEST 3.9.0: `grid`/`free` layers (2-D/3-D); the `distance` sentinel **with per-axis `.x/.y/.z`**
and the `pos`/`source_pos`/`target_pos` position accessors; all five distance distributions
(`gaussian`/`exponential`/`gamma`/`gabor`/`gaussian2D`, machine-precision weight-read-back parity);
all seven masks (`circular`/`spherical`/`box`/`rectangular`/`doughnut`/`elliptical`/`ellipsoidal`,
exact node-set parity incl. rotated + 3-D); the `spatial_pairwise_bernoulli` rule (riding the
existing `Simulator.connect` unchanged); the query helpers
`GetPosition`/`GetTargetPositions`/`GetTargetNodes`/`FindCenterElement`/`Distance` plus
`nearest_element` (`FindNearestElement`) and `select_nodes_by_mask` (`SelectNodesByMask`); the
`dump_layer_nodes`/`dump_layer_connections` exporters (text parity vs `DumpLayer*`); and the
matplotlib-gated `plot_layer`/`plot_targets`/`plot_sources`/`plot_probability_parameter` helpers
(smoke-tested). The cluster-27 additions were **purely additive** — zero change to the cluster-20
binding seam.

**`CollocatedSynapses`.** No support for multiple synapse types on the same
edge pair at once. NEST users wire AMPA + NMDA on the same pre→post pair in
one `Connect` call; brainpy.state requires multiple `Projection` instances.

**`TripartiteConnect`.** *Implemented (cluster 24).*
`Simulator.tripartite_connect(pre, post, third, conn_spec, third_factor_conn_spec,
syn_specs)` is the top-level helper for the astrocyte triad pattern (used by
`aeif_cond_alpha_astro`). It samples the primary `pre→post` connectivity **once** and
shares that one realization across the three arms — primary, `third_in` (`pre→astro`)
and `third_out` (`astro→post`, a `sic_connection`) — via an internal `_ExplicitEdges`
rule, so no new deposit primitive is needed. The pool sampler
`third_factor_bernoulli_with_pool(p, pool_size, pool_type)` supports `block` and
`random` pools. `pre`/`post`/`third` must be single-population views (the Brunel ports
use one sliced neuron population). Live-NEST parity (`tripartite_connect_test.py`):
deterministic `block` edge sets are bit-identical; `random` pools match seed-mean
distinct-edge counts within category D.

## 5. Semantic & numerical risks

- **Imperative-vs-compositional semantics.** `nest.Connect(pre, post, ...)`
  mutates global kernel state; subsequent calls accumulate. brainpy.state's
  `Projection.__init__` is local to the projection object. Direct port of a
  NEST script that calls `Connect` in a loop won't compile in brainpy.state's
  paradigm.
- **Stepping-loop control.** `nest.Simulate(t)` includes its own time-loop;
  `brainpy.state` users wrap their network in a `brainstate.compile.for_loop`.
  Behavioral difference: NEST schedules events on a global ring buffer with
  min-delay slicing; brainpy.state evaluates per-step with explicit delays via
  brainstate's delay containers. *Potentially semantically equivalent at
  matched dt + matched delays but timing-edge-cases can differ.*
- **Parameter expressions evaluated lazily.** In NEST, `weight=nest.random.normal()`
  draws a different value per edge at Connect-time; the draws are deterministic
  for a given `rng_seed`. A naive brainpy.state translation that uses
  `numpy.random.normal(size=N)` *outside* the projection has different draw
  semantics (e.g., draw ordering, seeding).
- **NodeCollection set operations.** NEST allows `nc1 + nc2`, `nc[1:3]`,
  `nc.position`. brainpy.state populations have no equivalent — users compose
  via brainstate's Module tree.
- **`CopyModel` parameter inheritance.** NEST's `CopyModel(parent, child, overrides)`
  creates a *new model name* with bound overrides usable in subsequent
  `Connect(syn_spec={'synapse_model': 'child'})`. brainpy.state users subclass
  in Python — different reflection semantics (e.g., can't enumerate via
  `Models()`).
- **`ResetKernel` destroys CopyModel models.** Per NEST docs. Documented for
  completeness; not a brainpy.state concern since brainpy.state has no kernel.
- **`waveform-relaxation` (`use_wfr`).** NEST kernel attribute controlling
  iteration for instantaneous rate connections + gap junctions. brainpy.state has
  **no** waveform relaxation: both seams reproduce NEST's `use_wfr=False` regime via
  the substrate's explicit **one-step pipeline lag** — rate connections (cluster 15a)
  and gap junctions (cluster 15b, the `(G−diag(D))@V[n−1]` difference current). The
  equivalence to NEST's non-iterated mode is validated (rate FP parity; gap 2-neuron
  micro-parity to machine precision between spikes). Cross-link
  `synapses-plasticity-gap.md` cluster-15b Update + `numerical-validation-gap.md` §4.

## 6. Validation gaps

- The `Simulator` facade **is** validated end-to-end against live NEST: the
  network tests build the same network through `Simulator` and compare output
  (`_network/_simulator_*_test.py`, `_network/_brunel_test.py`, the astrocyte /
  tripartite validation suites). What is still missing is a **verbatim
  same-script** port — every `_test.py` builds the network in each library
  independently rather than running one identical script through both. This
  awaits the `nest_compat` string-name shim.
- No test asserts **Parameter-expression equivalence** between
  `weight=nest.random.normal()` at `Connect` and brainpy.state's
  jax.random-based weight initialization (lazy `Parameter` is part of the named
  residual, so there is nothing to assert against yet).

## 7. Prioritized roadmap

- **P0 — Build a `brainpy_state.nest_compat` string-name shim.** [L] —
  **mostly delivered by the `Simulator` facade.** The `create → connect →
  simulate`/`cont` flow, the named rules, `get_connections`/`SynapseCollection`,
  and `tripartite_connect` all ship (`brainpy_state/_network/_simulator.py`,
  `_rules.py`). The remaining shim work is the thin string-keyed layer on top:
  model-by-name `Create('iaf_psc_alpha', ...)`, `CopyModel`, dict-style
  `GetStatus`/`SetStatus`, and `SetKernelStatus`. Acceptance: the Brunel example
  ports verbatim from NEST source with only the import line changed, within 5 %
  firing rate. (The facade already runs Brunel — `_network/_brunel_test.py`.)

- **P0 — Document the programming-model gap.** [S]
  Cross-link `docs-portfolio-gap.md`. Acceptance: a side-by-side "PyNEST →
  brainpy.state" cheatsheet exists in `docs/nest-guide/` (or until that
  exists, in `nest-status/index.rst`); covers `Create`, `Connect`,
  `Simulate`, `GetStatus`, `SetStatus`, `CopyModel`. Lives near the
  Experimental warning so users find it.

- **P0 — Map named connection rules to brainstate primitives + document.** [M]
  — **DONE.** The single-population rules ship as named values
  (`brainpy_state/_network/_rules.py`: `all_to_all`, `one_to_one`,
  `fixed_indegree`, `pairwise_bernoulli`, `fixed_total_number`,
  `explicit_edges`, `third_factor_bernoulli_with_pool`), each honouring
  `allow_autapses`/`allow_multapses` (`_network/_rules_test.py`,
  `_network/_connectivity_test.py`). Residual: `fixed_outdegree` is wired as an
  internal sampler but not yet a named top-level rule.

- **P1 — Implement `pairwise_bernoulli` and `fixed_total_number` as named
  helpers.** [M] — **DONE.** Both ship as named rules
  (`brainpy.state.pairwise_bernoulli(p)` / `fixed_total_number(N)`,
  `_network/_rules.py`) and are exercised in `_network/_rules_test.py` /
  `_network/_connectivity_test.py`. The string-keyed `conn_spec={'rule': ...}`
  spelling awaits the `nest_compat` shim.

- **P1 — Implement `Parameter` runtime-evaluated expressions in `nest_compat`.** [L]
  Rationale: needed for NEST examples that use `weight=nest.random.normal()`.
  Acceptance: `nest_compat.Parameter`, `nest_compat.random.normal`,
  `nest_compat.random.uniform`, `nest_compat.math.exp` available; a test
  draws connection weights from a normal distribution at Connect time and
  matches the resulting weight histogram with NEST's.

- **P1 — Add `TripartiteConnect` for astrocyte triads.** [M] — **DONE (cluster 24).**
  `Simulator.tripartite_connect(pre, post, third, conn_spec,
  third_factor_conn_spec, syn_specs)` with the
  `third_factor_bernoulli_with_pool(p, pool_size, pool_type)` rule shares one primary
  sample across the three arms and matches live-NEST connectivity statistics
  (`tripartite_connect_test.py`: block bit-identical, random within cat D;
  `astrocyte_small_network_test.py`, `astrocyte_brunel_test.py`).

- **P1 — `CollocatedSynapses` support.** [M]
  Acceptance: AMPA+NMDA collocation on one pair (used in NEST Brunel-Wang
  example) expressible in `nest_compat` in one call.

- **P2 — Spatial / topology surface.** [XL] — **DONE (clusters 20 + 27).**
  The full `nest.spatial` + `nest.spatial_distributions` surface ships under
  `brainpy.state.spatial` (grid/free layers, all seven masks, all distance
  distributions, queries, dump/plot), validated against live NEST 3.9.0. See
  §3.10 and the §4 Spatial / topology note.

- **P2 — Named operators (`fixed_indegree`/`outdegree` with full
  param-parity).** [S]
  Acceptance: every `conn_spec` boolean (`allow_autapses`, `allow_multapses`,
  `make_symmetric`) respected.

- **P2 — `GetStatus`/`SetStatus` dict-style API.** [M]
  Rationale: NEST examples use `nest.SetStatus(neurons, {'V_m': v0})` to
  initialize state. brainpy.state direct attribute assignment is more
  idiomatic but less portable. Acceptance: `nest_compat` provides both with
  a deprecation note pointing to the attribute style for new code.

- **P2 — `Models()`, `ConnectionRules()`, `GetDefaults`, `SetDefaults`.** [S]
  Acceptance: enumeration + default-introspection helpers for the
  `nest_compat`-exposed models.

- **P2 — Visualization shims (`PlotLayer`, `PlotTargets`,
  `nest.raster_plot.from_device`).** [M]
  Acceptance: thin wrappers around matplotlib in `nest_compat.viz`.

**Status note (as of 2026-06-16):** the brainpy.state-native foundation
(`Network`, `Builder`, eight `*Proj` rule classes, `Recorder`,
`brainpy.state.dist`) shipped on 2026-05-12, and the PyNEST-flavored facade now
ships on top of it as `brainpy.state.Simulator`
(`brainpy_state/_network/_simulator.py`) plus the named rules
(`_network/_rules.py`): `create`/`connect`/`tripartite_connect`/
`get_connections` (`SynapseCollection`)/`simulate`/`cont`, with the full
spatial/topology surface and `explicit_edges`. Cluster backlog 00–28 is all
merged. The only remaining `brainpy_state.nest_compat` work is the thin
string-name shim (model-by-name `Create`, lazy `Parameter`, `CopyModel`,
`CollocatedSynapses`) — see §7 P0. Permanently unsupported: MPI / MUSIC /
SONATA / structural plasticity / `store_restore_network` / bit-exact RNG. See
`docs/brainpy-guide/network-api.md` for the user guide and
`docs/superpowers/specs/2026-05-12-nest-network-api-design.md` for the design
spec.
