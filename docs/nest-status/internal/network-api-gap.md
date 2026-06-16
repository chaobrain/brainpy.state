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

Evidence basis: `grep -rE "^def (Connect|Create|...)" brainpy_state/ --include="*.py" | grep -v _test`
returns **zero matches** (run 2026-05-11). All PyNEST references in the repo are
inside `_test.py` files acting as comparison harnesses (e.g.,
`aeif_cond_alpha_test.py:448` calls `nest.Create`/`nest.Connect`/
`nest.Simulate` to compare against the repo's own model). The repo's
connection primitives instead live in `brainpy_state/_brainpy/projection.py`
(`AlignPostProj`, `DeltaProj`, `CurrentProj`, `align_pre_projection`,
`align_post_projection`) and brainstate (`FixedNumConn`, `EventFixedNumConn`,
`SparseLinear`).

## 2. Parity summary

The PyNEST API surface is **essentially entirely absent** from `brainpy.state`.
`brainpy.state` exposes a compositional JAX/brainstate API based on
`Projection` subclasses and direct attribute mutation, not the imperative
`Create → Connect → Simulate` flow that PyNEST users expect. This is the
single biggest porting obstacle for NEST users.

Mitigating factor: the underlying *primitives* exist in brainstate
(connectivity masks, sparse linear ops, delays, projections). What's missing
is the NEST-style facade.

| Bucket | Count | Notes |
|---|---:|---|
| implemented | 0 | None of the NEST PyNEST API is exposed with NEST-style naming |
| unvalidated | 0 | n/a — surface is absent |
| partial | ~5 | Connection primitives exist in brainstate (`FixedNumConn`, etc.) but with different shape than NEST `Connect` |
| divergent | 0 | n/a — surface is absent |
| missing | ~95 | All PyNEST top-level functions and spatial/topology entities |
| unsupported | ~8 | MPI parallel (`NumProcesses`, `Rank`, `GetLocalVPs`, `SetAcceptableLatency`, `SetMaxBuffered`, `SyncProcesses`, `Install` for dynamic modules), SONATA file loader (`SonataNetwork`) |
| **total PyNEST API + spatial surveyed** | **≈ 105** | per snapshot §§8-12 |

## 3. Evidence-backed mapping table

### 3.1 Connection handling

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `Connect(pre, post, conn_spec, syn_spec)` | missing | `brainpy_state/_brainpy/projection.py:AlignPostProj` (different programming model) | <https://nest-simulator.readthedocs.io/en/stable/ref_material/pynest_api/index.html> | comparison only (in `*_test.py`) | NEST users compose `Connect(...)` calls; brainpy.state users instantiate `AlignPostProj` subclasses |
| `TripartiteConnect(pre, post, third, conn_spec, third_factor_conn_spec, syn_specs)` | **implemented** | `Simulator.tripartite_connect` (24) | upstream | `tripartite_connect_test.py`, `astrocyte_small_network_test.py`, `astrocyte_brunel_test.py` | One shared primary `pre→post` sample feeds three arms (primary / `third_in` / `third_out`-`sic_connection`); single-population views. Live-NEST parity: block edge sets bit-identical, random pools match seed-mean counts (cat D) |
| `Disconnect(...)` | missing | none | upstream | — | No first-class disconnect; user removes the `Projection` object |
| `GetConnections(source, target, synapse_model, synapse_label)` | missing | none — projections expose `.weight`, `.delay` attributes directly | upstream | — | NEST users introspect via SynapseCollection; brainpy.state requires holding a reference to the `Projection` |

### 3.2 Node handling

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `Create(model, n=1, params=None, positions=None)` | partial | `Module()` constructors directly, e.g. `iaf_psc_alpha(n, **params)`; `Simulator.create(...)` mirrors it (incl. `positions=`) | upstream | `_network/_simulator_spatial_test.py` | `brainpy.state` style instantiates classes; NEST style passes a string model name. `Simulator.create(positions=spatial.*)` done (goal 20) |
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
| `ConnectionRules()` | missing | n/a | upstream | — | the connection-rule registry is absent (since `Connect` is absent) |

### 3.4 Simulation control

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `Simulate(t)` | missing | `brainstate.environ.set(dt=...)` + manual stepping loop, typically via `brainstate.compile.for_loop` over `Network.update()` | upstream | — | Programming-model gap. NEST handles the stepping loop internally; brainpy.state expects user to manage it |
| `Prepare()` / `Run(t)` / `Cleanup()` / `RunManager()` | missing | n/a | upstream | — | Stepwise simulation context isn't a separate API in brainpy.state |
| `ResetKernel()` | missing | construct a new Network | upstream | — | brainpy.state lifecycle is per-object, not per-kernel |
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
| `SynapseCollection` | missing | `Projection` object | upstream | — | introspection idioms differ |
| `Parameter` (runtime-evaluated expressions) | missing | none | upstream | — | NEST's `Parameter` allows `weight=nest.random.normal()` at `Connect` time; brainpy.state expects concrete values at projection-instantiation time |
| `CreateParameter(type, params)` | missing | none | upstream | — | parameter-distribution factory |
| `Mask` (spatial) | partial | `spatial.circular` / `spatial.spherical` / `spatial.box` | upstream | `_nest_spatial/_masks_test.py` | circular/spherical/box done; other mask shapes queued |
| `CollocatedSynapses(...)` | missing | multiple `Projection` instances on the same pre/post pair | upstream | — | NEST users compose multiple synapse types on one edge at once; brainpy.state requires separate Projection objects |
| `Compartments` / `Receptors` | partial | `brainpy_state/_nest/cm_default.py` exposes compartment+receptor spec internally | upstream | — | not a top-level class — users compose by passing dicts to `cm_default` constructor |
| `SonataNetwork(config)` | unsupported (per spec §7) | none | upstream | — | declarative SONATA loading out of scope; the SONATA HDF5 format itself could be imported by user code, but the NEST loader semantics are NEST-internal |
| `serialize_data` / `to_json` | missing | brainstate state PyTree serialization | upstream | — | rough equivalent for state, not for the full network |

### 3.9 Connection rules

| NEST rule | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `one_to_one` | missing (as named primitive) | manual `jnp.eye`-style mask | <https://nest-simulator.readthedocs.io/en/stable/synapses/connectivity_concepts.html> | — | trivial to express but no named helper |
| `all_to_all` | missing (as named primitive) | `jnp.ones((N,M))` mask or default dense projection | upstream | — | trivial to express |
| `pairwise_bernoulli(p)` | missing (as named primitive) | `brainstate.nn.SparseLinear` with random mask | upstream | — | composable but not named |
| `symmetric_pairwise_bernoulli(p)` | missing | none | upstream | — | bidirectional Bernoulli has no analog |
| `pairwise_poisson(pairwise_avg_num_conns)` | missing | none | upstream | — | Poisson-distributed counts per pair |
| `fixed_total_number(N)` | missing | none | upstream | — | exactly N total edges |
| `fixed_indegree(indegree)` | partial | `brainstate.nn.FixedNumConn` / `EventFixedNumConn` (in-degree-style) | upstream | — | brainstate `FixedNumConn` is close but rule semantics + param-name parity not documented |
| `fixed_outdegree(outdegree)` | partial | brainstate variant — verify direction | upstream | — | likely available; not named the same |
| `conngen` (CSA) | missing | none | upstream | — | Connection Set Algebra integration absent |
| `third_factor_bernoulli_with_pool` | **implemented** | `brainpy_state.third_factor_bernoulli_with_pool(p, pool_size, pool_type)` (24) | upstream | `_rules_test.py`, `_connectivity_test.py`, `tripartite_connect_test.py` | `tripartite_connect` astrocyte-pool rule; `block` + `random` pools. Live-NEST parity in `tripartite_connect_test.py` |

### 3.10 Spatial / topology

| NEST entity | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `nest.spatial.grid(shape, extent, ...)` | **done** | `brainpy.state.spatial.grid` | <https://nest-simulator.readthedocs.io/en/stable/networks/spatially_structured_networks.html> | `_nest_spatial/_layers_test.py`, `_validation/spatial_grid_test.py` | exact coord parity vs NEST (2-D + 3-D) |
| `nest.spatial.free(pos, ...)` | **done** | `brainpy.state.spatial.free` | upstream | `_nest_spatial/_layers_test.py`, `_validation/spatial_3d_test.py` | array- or distribution-backed (2-D/3-D) |
| `nest.spatial.pos.{x,y,z}` / `source_pos.{x,y,z}` / `target_pos.{x,y,z}` | missing | none | upstream | — | per-axis position expressions for use in Connect (kernel only consumes `distance` so far) |
| `nest.spatial.distance` / `.x/.y/.z` | partial | `brainpy.state.spatial.distance` (+ `displacement`, `pairwise_distance`) | upstream | `_nest_spatial/_distance_test.py`, `_kernels_test.py` | scalar `distance` sentinel done; per-axis `.x/.y/.z` not yet |
| `nest.spatial_distributions.exponential / gaussian / gaussian2D / gabor / gamma` | partial | `brainpy.state.spatial.gaussian` | upstream | `_nest_spatial/_kernels_test.py`, `_validation/spatial_gaussian_kernel_test.py` | `gaussian` done (distributional parity); exponential/gabor/gamma queued |
| `Mask` types (rectangular, circular, doughnut, elliptical, grid, box, spherical, ellipsoidal) | partial | `spatial.circular` / `spatial.spherical` / `spatial.box` | upstream | `_nest_spatial/_masks_test.py` | circular/spherical/box done (hard cutoff parity); rectangular/doughnut/elliptical queued |
| `GetPosition`, `GetSourcePositions`, `GetTargetPositions`, `FindNearestElement`, `FindCenterElement`, `Displacement`, `Distance`, `SelectNodesByMask` | partial | `Simulator.get_position`, `spatial.target_positions`, `spatial.target_nodes`, `spatial.center_element`, `spatial.Distance` | upstream | `_nest_spatial/_helpers_test.py`, `_validation/spatial_grid_test.py` | GetPosition/GetTargetPositions/GetTargetNodes/FindCenterElement/Distance done; nearest-element/by-mask queued |
| `DumpLayerConnections`, `DumpLayerNodes` | missing | none (use `get_connections` / `get_position`) | upstream | — | spatial export helpers |
| `PlotLayer`, `PlotTargets`, `PlotSources`, `PlotProbabilityParameter` | missing | none (matplotlib used directly in demos) | upstream | — | spatial visualization helpers |
| `Create(model, positions=spatial.*)` + `spatial` pairwise-Bernoulli rule | **done** | `Simulator.create(positions=)` + `spatial.spatial_pairwise_bernoulli` | upstream | `_network/_simulator_spatial_test.py`, `_nest_spatial/_rule_test.py` | coords attach to a population; spatial rule rides `connect(rule=)` unchanged |

### 3.11 Random / math / logic Parameter operators

| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `nest.random.uniform/uniform_int/normal/lognormal/exponential` | missing | `jax.random.*` / `numpy.random.*` | upstream | — | concrete RNG factories exist in JAX/NumPy; NEST's Parameter-binding semantics absent |
| `nest.math.exp/sin/cos/min/max/redraw` | missing | `jnp.*` | upstream | — | usable directly but only on concrete arrays, not on Parameter trees |
| `nest.logic.conditional` | missing | `jnp.where` | upstream | — | likewise |

## 4. Missing or incomplete functionality

**Programming-model gap (the headline).** NEST is an imperative simulator
controlled by global state (`ResetKernel`/`SetKernelStatus`/`Simulate`).
`brainpy.state` is a compositional JAX library. The two are reconcilable but
require either:

1. A **NEST-compat shim package** (`brainpy_state.nest_compat`) that exposes
   `Create`/`Connect`/`Simulate` as a thin facade over the existing
   compositional primitives — the most NEST-user-friendly option.
2. A **porting guide** that shows how to translate NEST-style imperative
   scripts to the compositional style — less work, less migration support.

**Connection rules.** Eight of the ten NEST connection rules have **no named
analog** in `brainpy.state`. `fixed_indegree` and `fixed_outdegree` are
*close* via `brainstate.nn.FixedNumConn` but parameter naming and semantics
need verification.

**Parameter expressions.** NEST's `Parameter` type lets `Connect` accept
runtime-evaluated expressions like `weight=nest.random.normal(mean=1.0, std=0.5)`
or `weight=2 * nest.spatial.distance`. `brainpy.state` requires concrete arrays
at projection-instantiation time. This breaks NEST examples that use
`Parameter` arithmetic.

**Spatial / topology.** The core of `nest.spatial` + `nest.spatial_distributions` **landed**
(goal 20): `grid`/`free` layers (2-D/3-D), the `distance` sentinel + `gaussian` kernel,
`circular`/`spherical`/`box` masks, the `spatial_pairwise_bernoulli` rule (riding the existing
`Simulator.connect`), and the `GetPosition`/`GetTargetPositions`/`GetTargetNodes`/
`FindCenterElement`/`Distance` query helpers — all under `brainpy.state.spatial`, validated
against live NEST 3.9.0 (exact grid coords; distributional kernel parity). Remaining surface
(per-axis `pos.x/y/z` expressions, the exponential/gabor/gamma distributions, the other mask
shapes, nearest-element / by-mask selection, layer dump/plot helpers) is queued; the
primitives and seam are in place to add them incrementally.

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

- No regression test demonstrates **end-to-end NEST script → brainpy.state
  port** parity. Each `_test.py` builds the same network in both libraries
  independently and compares output, but no test asserts "the same NEST
  script runs in both libraries."
- No test asserts **Parameter-expression equivalence** between
  `weight=nest.random.normal()` at `Connect` and brainpy.state's
  jax.random-based weight initialization.
- No `nest_compat` shim exists to test against.

## 7. Prioritized roadmap

- **P0 — Build a `brainpy_state.nest_compat` shim package.** [XL]
  Rationale: the absence of NEST-style API is the single biggest porting
  obstacle for NEST users. A thin façade — even one that covers only the
  10% of PyNEST that gets used in 90% of examples — unblocks the entire
  Examples roadmap (`examples-gap.md`). Minimum viable surface: `Create`,
  `Connect` (with `one_to_one`, `all_to_all`, `fixed_indegree`,
  `pairwise_bernoulli` rules), `CopyModel`, `GetStatus`, `SetStatus`,
  `Simulate`, `ResetKernel`, `SetKernelStatus`. Acceptance: at least the
  Brunel example ports verbatim from NEST source to `nest_compat`-using
  brainpy.state script with only the import line changed, and produces a
  firing-rate result within 5 % of NEST.

- **P0 — Document the programming-model gap.** [S]
  Cross-link `docs-portfolio-gap.md`. Acceptance: a side-by-side "PyNEST →
  brainpy.state" cheatsheet exists in `docs/nest-guide/` (or until that
  exists, in `nest-status/index.rst`); covers `Create`, `Connect`,
  `Simulate`, `GetStatus`, `SetStatus`, `CopyModel`. Lives near the
  Experimental warning so users find it.

- **P0 — Map named connection rules to brainstate primitives + document.** [M]
  Rationale: `fixed_indegree` and `fixed_outdegree` are the two most-used
  NEST rules in network examples; `brainstate.nn.FixedNumConn` is close.
  Acceptance: a section in `network-api.md` (new or in `nest-guide/`) shows
  the mapping; if the brainstate primitive's semantics don't exactly match
  NEST's (allow_multapses / allow_autapses), a thin wrapper is added to
  `nest_compat`.

- **P1 — Implement `pairwise_bernoulli` and `fixed_total_number` as named
  helpers.** [M]
  Acceptance: `nest_compat.connect(..., conn_spec={'rule': 'pairwise_bernoulli',
  'p': 0.1})` produces an edge set whose density matches NEST's `pairwise_bernoulli`
  within Monte Carlo tolerance over a 1000×1000 connection.

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

- **P2 — Spatial / topology surface.** [XL]
  Rationale: blocks NEST examples that use grid layers (visual cortex,
  retinotopic maps). Acceptance: `nest_compat.spatial.grid`,
  `nest_compat.spatial.free`, at least 2D rectangular and circular masks,
  `nest_compat.spatial.distance` available; matches NEST connectivity
  statistics for a distance-Gaussian connection in a 50×50 layer.

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

**Status note (post-Network-API):** the brainpy.state-native foundation
(`Network`, `Builder`, eight `*Proj` rule classes, `Recorder`,
`brainpy.state.dist`) shipped on 2026-05-12. The `brainpy_state.nest_compat`
PyNEST-style facade tracked above can now be built as a thin shim over
this layer. See `docs/brainpy-guide/network-api.md` for the user guide
and `docs/superpowers/specs/2026-05-12-nest-network-api-design.md` for
the design spec.
