# CONTEXT.md — NEST-parity goal ledger

> **READ THIS FIRST, every session.** It is the cross-session memory for the
> multi-prompt effort to reach NEST parity across `examples-gap.md` and
> `synapses-plasticity-gap.md`. It is a **committed** file (not gitignored) so it
> survives across worktrees via `main`. Each `/goal` prompt file reads it before
> starting and **appends a Lessons entry** (Part 5) before finishing.

---

## Part 1 — The goal

Two parallel streams, executed cluster-per-file (one worktree → one PR → one
merge per prompt file), in dependency order. Master plan + ordering:
`dev/superpowers/nest-goal-index.md`.

1. **Synapse-rebuild stream.** Take every synapse/plasticity model in
   `docs/nest-status/internal/synapses-plasticity-gap.md` from
   `divergent`/`unvalidated` → `implemented` (documented tolerance + passing
   live-NEST parity), and port the `missing` ones. The current implementations
   are **not JAX-conformant** and are being rebuilt (Part 2).
2. **Example-port stream.** Port every non-`unsupported` example in
   `docs/nest-status/internal/examples-gap.md` §3.1–3.10 to `examples/nest/<name>.py`,
   each as a live-NEST parity harness that drives model fixes.

`unsupported` items are out of scope (MUSIC, SONATA, structural plasticity,
HPC/MPI). `blocked` items (spatial, e-prop) ship a **skipped placeholder** until
their API lands.

## Part 2 — Architecture decisions (AUTHORITATIVE — do not relitigate)

These were settled deliberately. Follow them; if one looks wrong, write the
objection into a Lessons entry, do **not** silently diverge.

1. **The synapse layer is being rebuilt, not patched.** The legacy
   `_nest/*_synapse.py` models emulate NEST's kernel in Python — a
   `defaultdict(list)` event queue (`static_synapse.py:618,1041`), unbounded
   post-history lists (`stdp_synapse.py:514-523,622-637`), host-scalar
   `np`/`math`/`float` math, one object per connection. **None of it runs under
   `jit`/`vmap`/`grad`/`for_loop`.** A separate world (`_network/EventProjection`)
   does static delivery in JAX but never uses these models, so **plastic synapses
   are currently unusable in any real network.**
2. **Rebuild lives in `_network/`, on its own JAX primitives, sharing only
   `brainevent`.** Do **not** build on `_brainpy/`'s `Projection`/`Synapse`/`STP`/
   `align_post_ltp` stack. The substrate is the `brainevent.CSR` event-matmul
   pattern already in `_network/_projections.py:79-107` (`_SparseEventMatMul`),
   plus `InputDelay` for delays.
3. **Division of responsibility.** `_network/` owns the **compute primitive**
   (CSR edge layout, per-edge weight `State`, per-edge/per-neuron trace `State`,
   delay, the event matmul, the update loop). `_nest/*_synapse.py` owns the
   **NEST-faithful spec** (parameter names/defaults/units/init-validation) **+ a
   pure, vectorized `update(state, pre_spike, post_spike, dt) -> state` rule
   kernel** the primitive calls inside the jitted loop. No queues, no history
   lists, no host scalars in the hot path.
4. **NEST fidelity wins.** Where NEST semantics and a brainpy.state convention
   conflict (e.g. NEST's `tau_minus`-on-post vs. trace-on-synapse), the NEST
   adapter overrides and the delta is documented (docstring + `CONTEXT.md`).
5. **Typed family of three primitives** over one shared substrate (CSR edges +
   `InputDelay` + rule-declared `State` allocation):
   - **`EventPlasticProj`** — pre-spike in → weighted event out; per-edge weight +
     per-edge aux State (e.g. STP `u,x`) + per-pre/post-neuron trace State, the
     **rule declares exactly which State it needs**. Per-neuron online traces
     reproduce NEST all-to-all STDP; `stdp_nn_*` declare per-edge/reset state.
     Covers ~19 models (static, STP, STDP core+NN, vogels, jonke, ht, bernoulli,
     cont_delay).
   - **`VoltageCoupledPlasticProj`** — extends the above with a **post-state
     reader** (rule declares which post variables it samples each step). Covers
     `clopath_synapse`, `urbanczik_synapse`, `stdp_dopamine_synapse` (+
     `volume_transmitter` broadcast).
   - **`ContinuousCoupledProj`** — bidirectional/instantaneous + waveform
     relaxation. Covers `gap_junction`, `rate_connection_*`, `diffusion_connection`,
     `sic_connection`. **This is a co-design with the rate *neurons*** (see Part 3)
     and **gets its own grilling** before implementation.
6. **Imperative path is deleted.** The `syn.send()`/`record_post_spike()`/event-
   queue/`set()/get()` machinery is removed (not kept as a shim). The ~28 eager
   `*_test.py` files are **replaced** by vectorized network-level parity tests.
7. **Integration with the Simulator.** Plastic projections read pre **and** post
   spikes from the Simulator's `_SpikeHolder` (post spikes drive K⁻). Weight
   recording reuses the analog-recording (State-tap) mechanism, not an imperative
   hook.

## Part 3 — Repo reality map

- **JAX-native already (safe to build on):** spiking neurons `iaf_*`/`aeif_*`
  (Brunel runs them in `transform.for_loop`, `_network/_simulator.py:268`); the
  `Simulator` API (`_network/`: `Simulator`, `create`, `connect`, `NodeView`
  algebra, rules, `simulate`); `EventProjection` static delivery; `brainevent`
  0.1.0.
- **Imperative / NOT JAX-native (needs rebuild or bypass):**
  - Synapses — all of `_nest/*_synapse.py` (rebuild, Part 2).
  - **Rate/continuous models** — were the `_queue`-pattern **bucket 3**. Rebuilt on the
    seam-(H) substrate: `rate_neuron_ipn/opn`, `lin_rate`, `gauss_rate`, `sigmoid_rate*`,
    `siegert_neuron`, `step_rate_generator`, `rate_transformer_node` (15a);
    `aeif_cond_alpha_astro` + `astrocyte_lr_1994` + `sic_connection` (15d). **Only the
    Siegert `diffusion_connection` still carries `_queue`** — deferred to 15c.
  - Recording — `multimeter` is imperative; the Simulator **bypasses** it by
    tapping `State` into the `for_loop` output (analog recording, built in `02`).
    Do not gut multimeter; tap State.
- **Already done (not a cluster):** §3.1 flagship Brunel family —
  `examples/nest/brunel_{alpha,delta,exp_multisynapse,siegert,alpha_evolution_strategies}.py`
  exist and pass live-NEST parity. Reuse their patterns.
- **NEST upstream sources (read before porting):**
  `/mnt/d/codes/githubs/computational_neuroscience/nest-simulator/pynest/examples/`
  and `.../models/`.

## Part 4 — Conventions (every prompt file)

- **Session protocol** (canonical text in the master index; each prompt embeds a
  self-contained copy):
  1. Create worktree branch `nest-goal/<NN>-<slug>` from `origin/main`
     (`superpowers:using-git-worktrees`).
  2. **Read `CONTEXT.md` fully**, then the named gap-doc sections + relevant source.
  3. Do the work: `superpowers:brainstorming`/`writing-plans` → spec+plan under
     `dev/superpowers/` (gitignored) → TDD (`superpowers:test-driven-development`).
  4. **Live-NEST parity** test in `brainpy_state/_nest/_validation/<x>_test.py`,
     `skipUnless(nest)`. Carve-outs: distributional (not per-sample) where PRNG
     diverges; docs-only items exempt; blocked items ship a skipped placeholder.
  5. NumPy-style docstrings (doctest `Examples` in `.. code-block:: python`);
     tests > 90% coverage on touched code.
  6. **Append a Lessons entry to `CONTEXT.md`** (Part 5 template).
  7. Git lifecycle: commit (**no `Co-Authored-By`**), push, open PR, merge,
     delete remote branch, `git switch main && git merge --ff-only origin/main`,
     remove the worktree.
- **Working agreement (`CLAUDE.md`):** approach-first then wait for approval;
  ask when ambiguous; reflect on every correction; specs/plans gitignored under
  `dev/superpowers/`.

## Part 5 — Lessons learned (APPEND-ONLY; newest at top)

> Each session appends one entry before finishing. Template:
>
> ```
> ### <NN>-<slug> — <YYYY-MM-DD>
> - **Shipped:** <what landed; files; PR #>.
> - **Parity:** <metric: brainpy vs NEST, rel/abs error>.
> - **API discovered/changed:** <new _network primitive shapes, signatures the
>   next cluster should reuse>.
> - **Gotchas:** <NEST-fidelity traps, numerical pitfalls, dt/delay conventions>.
> - **For next clusters:** <advice, blockers found, scope adjustments>.
> ```

### cleanup-remove-legacy-synapses — 2026-06-16

- **Shipped:** removed the 6 retired `_legacy_*` modules under `_nest/` (~6,100 lines):
  `_legacy_imperative` (`ImperativeSynapseBase`), `_legacy_clopath_synapse`,
  `_legacy_stdp_synapse`, `_legacy_urbanczik_synapse`, and the two legacy test files. These
  were the pre-substrate imperative ports, retired from the active path when the STDP-core
  (`04`), Clopath (`07`), and Urbanczik (`21`) synapses were rebuilt as spec+rule kernels.
  Nothing in production imported them (private modules, not in `__init__`); only their own
  tests did. Also scrubbed 14 now-dangling `:mod:` docstring cross-references in the live
  synapse modules. Branch `worktree-remove-legacy-nest-synapses`.
- **Parity:** n/a (dead-code removal). Verified repo-wide grep clean, all 14 edited modules
  import cleanly, and **189** NEST-free rule tests pass across the affected synapses.
- **Gotchas:** the `04-stdp-core` entry below read as if `stdp_nn_*` / `bernoulli` /
  `cont_delay` "route to" the legacy modules — they never did (docstring mentions only); the
  rebuilt specs were already self-contained, so removal was a no-op for the active path.
- **For next clusters:** retired reference implementations don't need to live in the tree —
  git history preserves them. Keep an oracle only if a *current* test actually drives it.

### 19-pedagogical — 2026-06-16

- **Shipped:** the **§3.10 pedagogical group** — AdEx figures, compartmental dendrites, and the
  pong RL demo — closing §3.10 bar the `sudoku/` TODO. (1) `brette_gerstner_fig_2c.py`
  (spike-frequency adaptation, `aeif_cond_alpha`) + `fig_3d.py` (post-inhibitory rebound,
  `aeif_cond_exp`), each with a live-NEST `V_m` parity test (`CAT_A` sub-threshold window +
  `CAT_E` spike pattern) and a NEST-free law class. (2) `two_comps.py` (active vs passive
  dendrite) + `receptors_and_current.py` (per-compartment AMPA / NMDA / GABA + DC), both on the
  committed **`cm_default` Simulator seam** (multi-compartment state + multi-receptor routing +
  device→compartment wiring). (3) the **pong** RL demo: `pong.py` (pure-numpy game),
  `pong_networks.py` (`PongNetRSTDP` host-STDP on static synapses + `PongNetDopa` dopaminergic
  actor–critic), `pong_run.py` (`AIPong` head-to-head harness), built on two **new reusable
  substrate primitives** — `Simulator.cont()` (persistent rollout) and `host_spike_drive` /
  `host_current_drive` (State-clamped per-step input). Branch `worktree-nest-goal+19-pedagogical`;
  feature commits `a153e2d`…`0624353`. **`sudoku/` deferred** as a documented TODO (below).
- **Parity (vs live NEST 3.9.0):** *deterministic* ports compared per-sample — `fig_2c`/`fig_3d`
  sub-threshold `V_m` < 1e-3 mV (`CAT_A`), spike pattern within `CAT_E` (count ≤ 2, first spike
  ≤ 1 step); `two_comps` soma AP ~0.03 mV / Na-K gating 6e-3 / conductance ~1e-15 nS (active-
  dendrite *tip* residual ~0.56 mV — a sub-`dt` Na/K peak with no `cm_default` reset to re-anchor
  it, so it rides a 1.0 mV band); `receptors_and_current` DC-only ~1e-13 mV, synaptic-drive
  ~1e-2 mV (`CAT_C`). **Pong RL is NOT per-sample** (see Gotchas): `calculate_stdp` reproduces
  NEST's own method bit-for-bit (pinned literals, e.g. 67.6377405226, places=8); the dopamine
  reward→potentiation pathway is sign-checked vs a zero-reward control; behaviour is a bounded
  well-formedness assertion (weights inside `[Wmin, Wmax]`, paddles in-field, reward baseline off
  zero). `cont()` itself is oracle-tested — chunked `cont()` == one long `simulate()` (`CAT_B`) +
  chunked live-NEST accumulation. Full §3.10 suite green (substrate+pong **41 passed / 2 skipped**,
  brette+compartmental **19 passed**); new-module coverage host_drive 97 % / pong.py 99 % /
  pong_networks 99 % / pong_run 97 %.
- **API discovered/changed:** **`Simulator.cont(duration)`** — a non-re-init sibling of
  `simulate()` (both now share `_run_window`; `simulate` = `reset_rollout` + `_run_window`).
  `cont()` keeps `_base_t` / `_base_i` so biological time, device counters, and the `environ` step
  index continue across chunks; the compiled per-chunk `for_loop` is reused as long as only State
  *contents* change. `reset_rollout()` restarts at `t=0`; `cont()` lazily inits on first call.
  **`host_spike_drive` / `host_current_drive`** (`brainpy_state/_nest/host_drive.py`, exported
  inner + `brainpy.state`) — a `(window, n)` `ShortTermState` schedule read one row/step via a
  wrapping counter; `set_schedule(arr)` overwrites contents (fixed shape → no retrace). The spike
  role is a holder-backed source (wire `one_to_one`→`parrot`, weight 1.0); the current role
  declares `_injects_current` and routes through the `dc_generator` ring-buffer path. The host
  keeps a live handle via `view.segments[0].population`. **Per-edge static-weight overwrite**:
  `get_connections(src, tgt).set('weight', arr*u.pA)` mutates the live weight State in place (no
  recompile) — allowed on `static_synapse`, **refused** on rule-managed plastic weights
  (`PongNetDopa`'s dopamine edges raise `ValueError`), which is the right guard.
- **Gotchas:** (1) **RL parity is PRNG-divergent by construction** — the pong policy is a
  stochastic argmax over noisy motor spikes, so the game trajectory cannot match NEST per-sample.
  The faithful split is *component-deterministic* (the translation-invariant `calculate_stdp`
  matches NEST bit-for-bit; the dopamine pathway sign-matches) **plus** *behavioural* (bounded
  well-formedness), never a learning-curve compare — the same posture as `wang_decision_making`.
  (2) **Subthreshold motor neurons are not a bug** — 20×1300 pA input alone is sub-threshold for
  `iaf_psc_exp`; NEST relies on the `noise_generator` (noisy variant) or scaled-up weights (clean)
  to reach threshold. Reproduce both faithfully rather than "fixing" the silence. (3) **The
  recompile guard is the whole game** — every per-turn change must be a fixed-shape `State.value`
  write (`set_schedule`, `set('weight')`), never a retraced Python constant; the
  recompile-invariant tests (steady turn ≪ first turn) catch a regression but **self-skip under
  machine load** (timing inconclusive), so isolate the timed region. (4) **Noisy variants are
  ~5× slower** (per-step Gaussian RNG, *not* a recompile — flat per-turn times confirm it); keep
  CI smoke runs on the clean variants + tiny turn counts. (5) **`sudoku/` intractable** — its
  noise-driven WTA constraint network did not reach NEST's solve rate within a practical substrate
  config; deferred as a TODO (`examples-gap.md` §3.10) rather than shipped half-working.
- **For next clusters:** `cont()` + `host_drive` are the reusable **closed-loop / clamped-input**
  seam — any host-in-the-loop demo (online learning, neuromorphic control, interactive stimulus)
  should build on them rather than re-deriving a `for_loop` driver. `sudoku/` is the one §3.10
  item left (needs a tractable WTA-relaxation formulation). The `cm_default` seam (Phase 3) now has
  two demos exercising it; the sibling conductance-bridge sweep (from 17b) remains the other open
  substrate follow-up.

### 25-conductance-bridge-sweep — 2026-06-15

- **Shipped:** the **eight remaining conductance neurons** enrolled into the 17b
  multi-receptor spike→conductance bridge, closing the "silent dead conductance path"
  gap with **zero substrate change**. `aeif_cond_alpha`, `aeif_cond_exp`, `iaf_cond_beta`
  (the alpha·exp·beta **micro-parity gate** that first resolved design A), then
  `iaf_cond_alpha`, `iaf_cond_exp_sfa_rr`, `gif_cond_exp`, `hh_cond_exp_traub`,
  `iaf_chxk_2008` (a **migration**). Each gets `n_receptors=2` /
  `receptor_input_unit=u.nS` + a source-only `w_by_rec` dual-path arm in `update()` (the
  legacy self-pull is the `else` branch) and a per-model parity test
  `_validation/<m>_conductance_test.py` (4 law + 2 live-NEST). Shared bridge edge cases
  (weight=0, convergent scatter-add→2×, seam-zero, `receptor_type` out-of-range) fold
  into `aeif_cond_exp_conductance_test.py`. Branch
  `worktree-nest-goal+25-conductance-bridge-sweep` (11 commits). `neurons-gap.md` §4
  follow-up **cleared** + §3 eight rows flipped; `examples/nest/README.md` `multimeter_file`
  noted now-portable.
- **Parity (vs live NEST 3.9.0):** every model's `V_m` within `VM_TOL`
  (1e-3 mV + `align_steps=3`) and `g_ex`/`g_in` within `COND_TOL` (1e-3), for exc
  (`receptor_type=1` / NEST `+W`) and inh (`=2` / NEST `−W`). Micro-parity residuals (the
  design-A gate): alpha `V_m` **1.72e-6 mV** / `g_ex` 3.3e-7; exp 1.72e-6 / 1.6e-7; beta
  6.95e-6 / 1.2e-4 — all far inside the bands. Stiff/subthreshold cells match to ~**1e-12**
  in the non-stiff regime (HH `g_ex`/`g_in` are autonomous-linear → exact). No-regression:
  all eight existing `I_e` suites stay byte-identical (the self-pull `else` supplies the
  same guaranteed 0). Touched-model coverage **94–98 %**.
- **API discovered/changed:** **design A resolved — the bridge is source-only /
  kinetics-agnostic.** Declaring `n_receptors` + the `w_by_rec` arm enrolls *any*
  conductance neuron with **zero `_simulator.py` change**, regardless of synapse class
  (alpha `dg += (e/τ)·w`, exp `g += w`, beta `dg += pscon_β·w`) — only the *source* of
  `w_ex`/`w_in` swaps (blob column `k-1` vs `sum_delta_inputs(label=…)`). The micro-parity
  trio proved this before the mechanical sweep. **`iaf_chxk_2008` migration:** its bespoke
  `update(w_ex=, w_in=)` kwargs → canonical `w_by_rec` (the two NEST-reference step tests
  re-pointed to stacked `w_by_rec`, numbers unchanged — behaviour-preserving).
- **Gotchas:** (1) **stiff cells** (`gif_cond_exp`, `hh_cond_exp_traub`, `iaf_chxk_2008`)
  need `jax.clear_caches()` + x64 (`precision=64`) or the float32 trace-cache collides
  across collection order (21-Lessons). (2) **`hh_cond_exp_traub` has no stable rest** — at
  defaults it is an autonomous oscillator (spontaneous AP ~11 ms), and hyperpolarising to
  `V_m=-80` makes it *more* excitable (rebound: a 1 nS EPSP fires). Held quiescent at
  `V_m_init=-75 mV, I_e=0`, subthreshold, with the law tests comparing **driven-vs-baseline**
  (relaxation alone clears `E_L+1`). A full AP splits the stiff Dormand-Prince vs GSL solvers
  by volts — only the subthreshold regime matches tightly. (3) `gif_cond_exp` fires
  **stochastically**; subthreshold drive keeps the hazard `λ₀·exp((V−V_T*)/Δ_V)` ~0 so both
  sims stay silent. (4) `iaf_chxk_2008` *does* rest at `E_L` (proper IAF) — standard template
  applies; keep subthreshold so the intrinsic AHP stays inert. (5) no newly-activated
  demo/test assertion path exists (grep-confirmed) — nothing to update.
- **For next clusters:** the **conductance family now has no silent dead spike path** — any
  `connect(spikes, neuron, receptor_type=1/2)` drives `g_ex`/`g_in` with NEST-sign routing.
  Still-open sibling: the **current-based** `aeif_psc_*` / `hh_psc_alpha_clopath` self-pull
  `label='w_ex'/'w_in'` too but take `pA`, not the `nS` conductance bridge — a related but
  distinct seam (no `receptor_input_unit` scaling), not touched here.

### 20-spatial — 2026-06-15

- **Shipped:** the **spatial-connectivity API** (`nest.spatial.*` was the last genuinely-absent
  capability). A new **`brainpy_state/_nest_spatial/`** submodule (sibling of `_nest`/`_network`,
  **not** inside `_nest`) holds position layers (`grid`, `free`, 2-D/3-D), the `distance`
  sentinel + `displacement`/`pairwise_distance`, the `gaussian` kernel, `circular`/`spherical`/
  `box` masks, the `spatial_pairwise_bernoulli` rule, and query helpers (`center_element`,
  `Distance`, `target_nodes`, `target_positions`). Re-exported as **`brainpy.state.spatial`**.
  The Simulator seam: **`create(positions=spatial.*)`** (coords attach to a population) +
  **`get_position`** (NEST `GetPosition`). Four demos — `spatial_grid_iaf`, `spatial_gaussex`,
  `spatial_3d_gauss`, `spatial_csa` (native CSA) — plus a `csa_example` documented placeholder,
  with three validation files (`_validation/spatial_{grid,gaussian_kernel,3d}_test.py`). Branch
  `worktree-nest-goal+20-spatial`.
- **Parity (vs live NEST 3.9.0):** **grid coordinates exact**, element-for-element
  (`GetPosition`, 2-D + 3-D) and the centre element (`FindCenterElement`). **Gaussian kernel
  distributional:** empirical `p(d)` binned by distance matches NEST bin-by-bin —
  gaussex max\|bp−NEST\| ≈ **0.016** (21083 vs 20980 edges, 0.5 %); 3-D max\|bp−NEST\| ≈ **0.008**
  (129437 vs 130758 edges, ≈1 %), both also tracking the analytic Gaussian. Box mask is a hard
  per-axis cutoff; `allow_autapses=False` removes every self-edge. **87 spatial tests, 100 %
  `_nest_spatial` coverage.**
- **API discovered/changed:** the spatial rule rides the **existing** `connect(rule=)` with
  **no signature change** beyond `create(positions=)`. `create` stores coords in
  `self._positions[id(pop)]` (grid/concrete-`free` derive size from coords; deferred `free`
  draws `size` with a per-pop key). The bind happens at the **top of `_connect_pair`**, gated on
  `getattr(rule, '_is_spatial', False)`: `_bind_spatial_coords` slices each side's coords by the
  segment's local indices and returns `rule.with_coords(pre, post)` — a **pure clone**, so every
  downstream path (static, plastic, diffusion, gap, sic) samples one coordinate-bound rule and
  the `ConnRule.sample(n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses)`
  contract is unchanged. `target_nodes`/`target_positions` read realized adjacency back via
  `get_connections` (population-local indices). **Reuse this pattern**: any new distance-rule
  variant subclasses/feeds `SpatialConnRule`; any new query helper takes `(sim, source, target)`.
- **Gotchas:** (1) **Diagonal/autapse trap in distributional binning** — for a self-connection
  with `allow_autapses=False`, the `n` diagonal `(i,i)` pairs have `d=0` but are *unconnectable*;
  they must be **excluded from the candidate-pair denominator** or the `d≈0` bin's empirical
  fraction is diluted toward 0 (cost me a RED on the 3-D law test). a≠b connects (gaussex) don't
  hit this — `a_i→b_i` at `d=0` is a real, allowed edge. (2) **`free(distribution)` infers
  dimensionality from `extent` XOR `num_dimensions`** — passing both raises; the 3-D demo extent
  `[1.5,1.5,1.5]` already implies 3-D, so don't also pass `num_dimensions=3`. (3) **Units:** bare
  floats on coords/extent/std become `u.um` via `_as_len`, so `gaussian(std=0.5)` is 0.5 µm and
  `d/std` is dimensionless — never multiply a Quantity distance by a bare std. (4) **NEST grid
  convention** (now NEST-confirmed, not just pinned): `x = c−L/2+(col+0.5)·L/n`,
  `y = c+L/2−(row+0.5)·L/n`, node `k`=col·n_rows+row (column slow, row fast); default extent is
  the unit hypercube. (5) **Distributional tests use a fixed brainpy seed** so the structural
  (NEST-free) class is deterministic; only the bp-vs-NEST band tolerates PRNG divergence.
- **For next clusters:** the **primitives + seam are the substrate** for the rest of
  `nest.spatial`. Queued (all additive, no seam change): per-axis `spatial.pos.x/y/z` /
  `source_pos`/`target_pos` expressions (kernels currently consume only the `distance` scalar);
  the `exponential`/`gabor`/`gamma` distance distributions (mirror `_GaussianKernel`); the other
  mask shapes (rectangular/doughnut/elliptical — mirror `_RadialMask`/`_BoxMask`); nearest-element
  / `SelectNodesByMask`; layer dump/plot helpers. The `_positions` registry + `get_position` +
  `target_*` helpers are the read-side substrate; `with_coords`-clone binding is the write-side
  substrate. See `network-api-gap.md` §3.10 (now mostly **done/partial**) for the precise residual.

### 24-tripartite-connect — 2026-06-15

- **Shipped:** the `Simulator`-level **`tripartite_connect`** + **`third_factor_bernoulli_with_pool`**
  astrocyte-pool rule (NEST's `TripartiteConnect`), and the **three §3.8 pool-rule demos**
  promoted from skipped placeholders to real ports. One realized primary `pre→post` sample is
  shared across three arms — primary, `third_in` (`pre→astro`, delta IP3) and `third_out`
  (`astro→post`, the 15d `sic_connection`) — reusing the merged static + SIC paths with **no new
  deposit primitive**. New: `_network/_rules.py` `_ExplicitEdges` (precomputed-edge rule) +
  `_ThirdFactorBernoulliWithPool` + `third_factor_bernoulli_with_pool` factory;
  `_network/_connectivity.py` `build_pool_map` + `sample_third_factor_pairing`; `_simulator.py`
  `tripartite_connect`/`_connect_tripartite_arm`/`_tripartite_segment`. Tests:
  `_network/_simulator_tripartite_test.py` (NEST-free structural),
  `_nest/_validation/tripartite_connect_test.py` (live-NEST GATE, 7), and the three demo parity
  suites (`astrocyte_small_network_test.py` 3, `astrocyte_brunel_test.py` 5) — **79 touched
  tests pass**. Examples: `astrocyte_small_network.py`, `astrocyte_brunel_{bernoulli,fixed_indegree}.py`.
  Branch `worktree-nest-goal+24-tripartite-connect`. **§3.8 complete; `network-api-gap`
  TripartiteConnect + third-factor rows flip to implemented.**
- **Parity (vs live NEST 3.9.0):** the GATE validates Design A two ways — **block** (`p=1`,
  `pool_type='block'`) realized edge-sets are **bit-identical** to NEST across all three arms;
  **random** pools match seed-by-seed on `n2n`/`n2a`/`a2n` counts (category D, 5 %), with the hard
  **pool invariant** (distinct astrocytes per target ≤ `pool_size`) asserted on the NEST side too.
  `astrocyte_small_network` is deterministic per-sample parity: IP3 `~3e-6` / Ca `~6e-5` / `V_pre`
  `~0.01` (CAT_A) / `I_SIC` `~0.7 %` (loosened `SIC_TOL`, log-onset ×10) under `ASTRO_TOL` align.
  The two `astrocyte_brunel_*` ports are **connectivity-distributional** (n2n/n2a/a2n/inh seed-mean
  counts, CAT_D) for both primary rules — not rate parity.
- **API discovered/changed:** `tripartite_connect(pre, post, third, *, conn_spec,
  third_factor_conn_spec, syn_specs={'primary'|'third_in'|'third_out': {...}}, seed, comm,
  allow_autapses, allow_multapses)` returns `(primary, third_in, third_out)` projections (`third_*`
  are `None` when no edge pairs, e.g. `p_third=0`). The **shared-sample** mechanism is
  `_ExplicitEdges(ConnSpec)` — a `ConnRule` whose `sample()` returns its precomputed `ConnSpec`, so
  `_connect_pair` wires the *same* realized edges on every arm instead of re-drawing.
  `_ThirdFactorBernoulliWithPool.sample_third(primary_spec, n_post, n_third, *, key)` →
  `(third_in_spec, third_out_spec)`. **Constraints the next cluster must respect:** each role must
  be a **single-population, single-segment** view (a prefix slice like `neurons[:N_ex]` is fine;
  `a + b` and deferred generators raise `NotImplementedError`); `third_out` is forced onto
  `comm='dense'` (the `sic_connection` `as_current` path). The Brunel ports satisfy the single-
  segment rule by making **one** neuron pop and slicing `ex = neurons[:N_ex]`, `post = neurons`.
- **Gotchas:** (1) **NEST `rng_seed` must be in `(0, 2^32-1)`** — it rejects 0 with `BadProperty`,
  while brainpy/jax accept `seed=0`. The block-parity test seeds **both** sides with `1` (block is
  deterministic, seed-irrelevant); never pass 0 to a NEST kernel. (2) **A synaptically-driven,
  near-critical spiking `V_m` is unsuitable for sample-wise parity** — `astrocyte_small_network`'s
  `V_post` diverged ~25 mV from a single spike-timing flip even though IP3/Ca/`V_pre` matched to
  ~1e-5; validate the *loop coupling* (IP3→Ca→`I_SIC`) and the *driver* `V_pre`, drop the driven
  `V_post`. (3) **Astrocyte SIC ignition has a ~250–300 ms latency** (slow Ca integrator) — a
  dynamics-law window shorter than that sees `I_SIC.max()==0` and reads as "loop dead"; the Brunel
  law uses a 400 ms window + an IP3-climb assertion. (4) **Dense-merge sums duplicate-edge weights**
  = NEST's multigraph dynamics (N edges of weight w ≡ one merged edge of weight N·w);
  `realized_edges()` reports the *unique* edge-set. Confirmed empirically (`3.59 = 3.59` on a 2×2×1
  minimal net) — no multiplicity bug. (5) **Coverage under JAX SIGABRTs** if any trace core is
  active during `jax/__init__`; the working recipe is *pre-import jax/jaxlib/brainstate before
  `cov.start()`* + `COVERAGE_CORE=sysmon` + `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (else the `jaxtyping`
  pytest plugin re-imports numpy under assertion-rewrite → "cannot load module more than once").
  **Touched-code coverage = 100 %** (90/90 new statements across `_rules`/`_connectivity`/`_simulator`).
- **For next clusters:** `sic_connection` is now consumed twice (15d bidirectional loop + cluster-24
  `third_out`); reuse `tripartite_connect`'s single-segment + `_ExplicitEdges` shared-sample pattern
  for any "one sample → many arms" rule (e.g. `CollocatedSynapses`). The static-vs-`tsodyks`
  divergence on the primary/`third_in` arms is connectivity-neutral and documented; a future
  STP-on-tripartite pass would swap the arm synapse only. Bucket-3 model clusters fully closed bar
  the Siegert `diffusion_connection` (→ 15c).

### 17b-astro-demos — 2026-06-15

- **Shipped:** the two substrate-ready **§3.8 astrocyte demos** on the `Simulator` API +
  the latent **`aeif_cond_alpha_astro` spike→conductance fix** they surfaced.
  `examples/nest/astrocyte_single.py` (one Poisson-driven `astrocyte_lr_1994` → IP3/Ca + a
  downstream `aeif_cond_alpha_astro` SIC) and `astrocyte_interaction.py` (the tripartite
  loop) each ship with a live-NEST parity test
  (`_validation/astrocyte_single_test.py`, `astrocyte_interaction_test.py`) and a NEST-free
  law class. The three pool-rule demos (`astrocyte_small_network.py`,
  `astrocyte_brunel_bernoulli.py`, `astrocyte_brunel_fixed_indegree.py`) ship as
  **documented skipped placeholders** (each a `BLOCKED_REASON` + `main()` that raises
  `NotImplementedError`, guarded by marker tests). **The fix:** `aeif_cond_alpha_astro` now
  exposes the `iaf_cond_exp` multi-receptor bridge (`n_receptors=2`,
  `receptor_input_unit=u.nS`, a `w_by_rec` arm in `update()`) + a dedicated conductance
  parity test (`_validation/aeif_cond_alpha_astro_test.py`, 4 law + 2 parity). Branch
  `worktree-nest-goal+17b-astro-demos`. **§3.8 is now closed bar the pool-rule
  placeholders → bucket-3 fully closed.**
- **Parity (vs live NEST 3.9.0):** the new spike-driven conductance path — `V_m` within
  `VM_TOL` (1e-3 mV + `align_steps=3`; residual after the clean 2-sample shift ~**1e-6 mV**),
  `g_ex`/`g_in` within `COND_TOL` (1e-3), for both an excitatory train (`receptor_type=1` /
  NEST `+W`) and an inhibitory train (`receptor_type=2` / NEST `−W`). `astrocyte_single`
  IP3/Ca/`I_SIC` and `astrocyte_interaction` `V_pre`(`CAT_A`)+IP3/Ca/`I_SIC` track NEST
  within `ASTRO_TOL` (1e-3, `align_steps=3`) under the deterministic `spike_generator` /
  constant-`I_e` drives. The demo at its NEST-faithful defaults (1.0 nS, 1500 Hz, 60 s) gives
  peak IP3 **0.653** / Ca **0.666** / I_SIC **6.151** vs NEST's own demo 0.565 / 0.676 /
  6.173 (Poisson PRNG-diverges, so this is a *demo-faithfulness* check, not per-sample
  parity). Full astro suite **30 passed / 2 skipped**; example coverage **95 %**.
- **API discovered/changed:** the Simulator's multi-receptor bridge
  (`_simulator.py:1486-1496`) fires for any neuron that declares **both** `n_receptors` and a
  `w_by_rec` param in `update()`'s signature — it then calls
  `m.update(w_by_rec=get_mantissa(m.sum_delta_inputs(zeros((*varshape, n_receptors)) *
  receptor_input_unit) / receptor_input_unit))`, column `k-1` = `receptor_type=k`. For a
  conductance neuron, column 0 → `g_ex`, column 1 → `g_in`, with **positive nS** weights —
  the brainpy expression of NEST's weight-**sign** routing (`aeif_cond_alpha_astro.cpp`
  `handle()`: `weight>0 → spike_exc_`, else `spike_inh_`). The alpha-derivative scaling
  (`dg_ex += (e/τ_syn_ex)·w_ex`) already matched NEST's `DG_EXC += spike_exc·g0_ex`; only the
  *source* of `w_ex`/`w_in` changed (bridge vs self-pull). **No-regression argument:** adding
  `n_receptors` only flips dispatch to `update(w_by_rec=…)`; since nothing previously
  deposited `w_ex`/`w_in`, the self-pull already returned 0, so the bridge supplies the same
  0 → byte-identical for every existing astro test (they act as regression guards).
- **Gotchas:** (1) **Silent dead conductance path** — a conductance neuron whose `update()`
  *only* `self.sum_delta_inputs(label='w_ex'/'w_in')` receives **nothing** from
  `sim.connect(spikes, neuron, weight=…)`: the Simulator never populates those labels without
  the bridge, so a presynaptic spike leaves `V_m` pinned at `E_L` with **no error**. The
  committed I_e-driven parity tests passed straight through it (they never compared a
  conductance-dependent trace). (2) **`align_steps` absorbs a clean 2-sample integer offset**
  (synaptic-input-applied-after-integration + multimeter buffer alignment) — measured by a
  shift sweep (shift −2 → V_m Δ~1e-6, g_ex Δ~0); use `align_steps=3` (the existing astro
  band), *not* a loosened atol, to keep the per-sample precision. (3) **Import resolution
  trap:** `python /tmp/probe.py` puts `/tmp` on `sys.path` and picks up an **installed**
  `brainpy_state`, not the worktree edits → ran against stale code and crashed on the
  missing `n_receptors`. Use `PYTHONPATH=<worktree>` or `python -c` (cwd on path); `pytest`
  always uses the worktree rootdir. (4) **Coverage:** `coverage run -m pytest <files>` then
  `coverage report --include=…` — dropping `--source` (it SIGABRTs inside jaxlib); and **wait
  for the run to fully finish** before reading, a partial `.coverage` reports misleadingly low.
- **For next clusters:** the **sibling conductance-bridge sweep** is the tracked follow-up
  (recorded in `neurons-gap.md` §4): `aeif_cond_alpha`, `aeif_cond_exp`, `iaf_cond_alpha`,
  `iaf_cond_beta`, `iaf_cond_exp_sfa_rr`, `iaf_chxk_2008`, `gif_cond_exp`, `hh_cond_exp_traub`
  each still self-pull (grep-verified `label='w_ex'` present, `w_by_rec` absent) and need the
  same bridge + a conductance parity test (their tests are `I_e`-only, so the gap is
  uncaught). The three pool-rule demos remain blocked on NEST's `TripartiteConnect` +
  `third_factor_bernoulli_with_pool` astrocyte-pool rule (`network-api-gap.md`) — the next
  astrocyte-network cluster.

### 17-rate-demos — 2026-06-15

- **Shipped:** the two remaining §3.6 **rate-network demos** on the `Simulator` API, each a
  runnable script + live-NEST parity test + NEST-free companion (cluster-16 house style):
  `examples/nest/lin_rate_ipn_network.py` (E/I `lin_rate_ipn`, delayed-E + instantaneous-I)
  and `examples/nest/rate_neuron_dm.py` (two-unit rectified WTA decision). Tests:
  `_nest/_validation/lin_rate_ipn_network_test.py` (8), `rate_neuron_dm_test.py` (8). Docs:
  `examples/nest/README.md` §3.6 + `docs/nest-status/internal/examples-gap.md` rows flipped.
  **No substrate change** (the 15a rate core sufficed). Branch
  `worktree-nest-goal+17-rate-demos`, **PR #68**. §3.6 is now complete bar `ht_neuron`.
- **Parity (vs live NEST 3.9, `use_wfr=False`):**
  - lin_rate_ipn_network: a deterministic (`sigma=0`) instrument net's fixed point == closed
    form `(λI−W)⁻¹μ` == NEST to **atol 1e-3**; per-neuron trajectory == NEST with
    **`align_steps=12`** (`TraceTolerance` 1e-4/1e-4). FP is delay- and dt-invariant; the
    random demo net is covered by a smoke run.
  - rate_neuron_dm: deterministic strong-bias winner **11.0 / loser 0 on both sims exactly**;
    strong ±bias decision direction **5/5 and 0/5 identical**; zero-bias both-win balance
    (brainpy 1:4, NEST 2:3 over 5 seeds; seed-mean < 0.85·gap).
- **API discovered/changed:** none (no substrate edit). Contracts the next rate cluster reuses:
  - **`rectify_output` is correct in a *recurrent* rate loop** (the goal R1 arbiter): the
    rectified `rate` is what the seam holder emits, so a loser clamped to 0 deposits 0 into its
    partner — clean WTA, no special-casing.
  - **Two-phase μ-protocols set `mu` at construction, not by mutating the node.** `NodeView`
    (the `create()` return) does **not** forward attribute writes to the wrapped module, so
    `d1.mu = 1+dE` sets a dead attribute on the *view* and the dynamics never see it. Because
    `simulate()` re-inits state to `rate_init` each call, building a fresh per-phase net with
    that phase's `mu` is **numerically identical** to NEST's mutate-and-continue (both evidence
    phases start from `rate=0`, sharing one RNG stream) — and idiomatic.
- **Gotchas:**
  - **Few-seed unbias bands must reject only the *fully one-sided* case.** With 5 seeds an
    unbiased WTA can only split 3:2 / 4:1 / 5:0 → seed-mean |D1−D2| ∈ {2,6,10}·(winner/5). A
    4:1 sampling lean (≈6) is normal, so a `< 0.5·gap` band wrongly fails it; the robust test
    is **both units win ≥1** (rejects 5:0) + a magnitude guard `< 0.85·gap` (rejects only ≈10).
    The plan's original `mean(|mean(r1)−mean(r2)|)` measured per-seed *contrast* (~10), not
    bias — wrong statistic; average the **signed** rates over seeds first.
  - **The `sigma=0` anchor is the tight cross-sim arbiter; noisy runs are distributional.**
    Random connectivity (lin_rate net) and the WTA attractor (dm) both PRNG-diverge, so a
    per-sample NEST match is meaningless there — anchor on the closed form / `sigma=0` and
    compare seeds otherwise (same discipline as clusters 14 / 22).
  - **`fixed_outdegree` has no brainpy rule → map to `fixed_indegree`** with
    `K_in = N_src·K_out/N_tgt` (same expected in-degree, hence the same mean-field input).
  - **dm uses `dt=0.1` ms (not upstream `1e-3`)** for runtime; both sims share `dt`, so the
    decision parity is unaffected (the converged winner rate is `dt`-independent).
  - **NEST repo actually lives at `/mnt/d/codes/githubs/nest-simulator`**, NOT the
    `…/computational_neuroscience/nest-simulator` path in CLAUDE.md (stale; do not edit theirs).
  - **Coverage:** both example modules **97 %** (only the `if __name__=='__main__'` line is
    uncovered; `main()` plotting carries `# pragma: no cover`). Use `coverage run -m pytest`
    + report-time `--include=` — `coverage run --source=…` still SIGABRTs on absl double-init.
- **For next clusters:** the 15a rate substrate + these ports leave only `ht_neuron`
  intrinsic-currents demos in §3.6 (they need a single-neuron intrinsic-currents primitive,
  not continuous coupling). Reuse: per-phase reconstruction for any "change a scalar param
  mid-run" protocol; the both-win + magnitude-guard recipe for few-seed unbias. Still deferred
  from 15a/15c: **sparse graded/rate emission** (only `comm='dense'` is wired).

### 15c-siegert-diffusion — 2026-06-15

- **Shipped:** the **Siegert mean-field node + dual-channel `diffusion_connection`** on the
  JAX substrate, completing the 15a deferral. **(B)** `siegert_neuron`'s transfer Φ(μ,σ²)
  ported to jnp (leggauss-64 + `erfcx`/Dawson + asymptotics) so `update()` **lowers under
  `for_loop`** — the 15a eager exception is retired; **(A)** `diffusion_connection` de-queued to
  a thin NEST-parity status spec the Simulator routes as a **dual-channel seam deposit**. Files:
  `_nest/siegert_neuron.py` (jnp `_siegert_phi_jax`/`_siegert_phi_core` + `_erfcx_jax`/
  `_dawsn_jax`/`_integral_erfcx_jax`; `_emission_continuous`/`_emission_attr='rate'`;
  dual-channel `update()` reading `'diffusion_mu'`/`'diffusion_sigma2'`), `_nest/
  diffusion_connection.py` (6 dict-queue methods deleted → status spec, `_IS_DIFFUSION`),
  `_network/_simulator.py` (`_build_siegert_diffusion` + dispatch). Validation:
  `siegert_diffusion_test.py` (23), `brunel_siegert_test.py` (2, broken import fixed) + the
  `examples/nest/brunel_siegert.py` Simulator rewrite. Branch
  `worktree-nest-goal+15c-siegert-diffusion`.
- **Parity (vs live NEST `use_wfr=False`):** jnp Φ == SciPy oracle ≤**1e-6** across the (μ,σ²)
  grid + colored-noise (`tau_syn>0`) + NEST ref `27.1095934379`; live two-siegert micro-parity
  (drift+diffusion) == NEST to **machine precision (max|Δ|~1e-15, shift=0)**; Brunel mean-field
  (`order=2500`) relaxes to the closed-form Siegert FP to **~3e-13** and to live NEST **0.00 %**
  (32.03 vs 32.03 spks/s), `erate==irate` exactly by symmetry.
- **API discovered/changed:** `_build_siegert_diffusion` fans ONE `diffusion_connection` into
  TWO labeled rate projections off the same seam-(H) `rate` emission
  (`drift_factor`→`'diffusion_mu'`, `diffusion_factor`→`'diffusion_sigma2'`) sharing one
  connectivity sample; rejects sparse / weight / delay / generator / non-continuous source.
  Convergent deposits **accumulate**: each EventProjection gets a unique `_delta_key`, so N
  edges into one target SUM under `sum_delta_inputs(label=…)`. Siegert is now a normal
  `for_loop`-lowering neuron — **no eager driver** (supersedes the 15a note).
- **Gotchas:**
  - **`sum_delta_inputs(label=None)` catches ALL keys — labeled ones included** (`filter_fn =
    lambda k: True`). This **refutes the plan's "drift → default channel, σ² → labeled"**: a
    default read would sum μ+σ². So **BOTH** channels must carry distinct labels
    (`'diffusion_mu'`, `'diffusion_sigma2'`) and the post reads each with its own
    `sum_delta_inputs(label=…)`. (Namespacing: `_input_label_repr(name,'L')='L // name'`; a
    labeled read filters `startswith('L // ')`, so distinct labels never cross.)
  - **The seam-holder lag IS NEST `min_delay=1` — micro-parity needs NO `align_steps`.** Unlike
    15a's instantaneous rate coupling (which needed `align_steps` to absorb a uniform offset on
    the transient), the two-siegert diffusion trace matches NEST at **shift=0** to machine
    precision: phase-1 projections read the *previous* step's `rate` holder, exactly NEST's
    one-step diffusion delivery.
  - **The host numpy asymptotics carry wrong signs — undiscovered because they are dead code.**
    SciPy is preferred, so the no-SciPy `_erfcx_pos_scalar` poly (all `+`; erfcx is
    **alternating** `(-1)^k(2k-1)!!/(2x²)^k`) and `_integral_erfcx_asympt` odd terms (k=1,3 were
    `-`, should be `+`) were never validated. The jnp port fixed them; **Dawson's asymptotic is
    all-positive** (do not alternate it). The host fallback is further guarded by the
    `(θ−μ)>6σ` deep-subthreshold fast-path, so it is unreachable in practice — but its scalar
    sign bugs (mirroring the ones fixed in jnp) should be repaired if SciPy-absent operation is
    ever relied upon.
  - **siegert relaxation `tau` defaults to 1 ms** (not `tau_m`): 50 ms = 50τ → fully converged,
    so the Brunel closed-form check is tight; the NEST micro-parity side uses the same default.
  - **Coverage:** `diffusion_connection.py` **100 %**, `siegert_neuron.py` **95 %** (residual =
    the dead no-SciPy host asymptotic fallback above); `_build_siegert_diffusion`'s touched
    `_simulator.py` lines fully covered (whole-file % is dominated by pre-existing infra).
- **For next clusters:** the dual-channel **labeled-deposit + convergent-accumulation** pattern
  generalizes any multi-quantity graded coupling — deposit each quantity under its own label,
  read each back by label, and **never rely on a default `sum_delta_inputs` read when labeled
  deposits coexist**. Still deferred: **sparse graded/diffusion emission** (only `comm='dense'`
  is wired).
### 15b-gap-junctions — 2026-06-15

- **Shipped:** **explicit-lag gap-junction coupling** for the two gap-capable HH neurons,
  reusing cluster 15a's seam-(H) emission substrate verbatim. The `Simulator` realizes
  NEST's `gap_junction` as a recurrent **difference-current** coupler
  `I_gap,i[n] = Σ_j g_ij (V_j[n−1] − V_i[n−1]) = (G − diag(D)) @ V[n−1]` deposited into the
  post's **current** channel, under the substrate's one-step pipeline lag (NEST's
  `use_wfr=False` regime — **no** waveform relaxation). Files: `_nest/{hh_psc_alpha_gap,
  hh_cond_beta_gap_traub}.py` (declare `_emission_attr='V'`; per-neuron gating-init fix),
  `_network/_simulator.py` (`_gap_conductance`, `_gap_current`, `_build_gap_coupling`,
  `_is_gap_synapse` dispatch in `_connect_pair`, `_gap_couplers` phase-loop injection).
  Tests: `_network/_simulator_gap_test.py` (20 NEST-free seam/guard/behavior),
  `_nest/{hh_psc_alpha_gap,hh_cond_beta_gap_traub}_test.py` (+ heterogeneous-init regression),
  `_nest/_validation/{gap_junction_parity, gap_junction_inhibitory_network_parity,
  gap_junction_no_nest}_test.py`, `examples/nest/gap_junctions_{two_neurons,
  inhibitory_network}.py`. The reference `_nest/gap_junction.py` WFR class is left
  **untouched and unused**. Branch `worktree-nest-goal-15b-gap`.
- **Parity (vs live NEST `use_wfr=False`):**
  - **2-neuron micro-parity** (g=0.5 nS, resting gating, I_e=100 pA, T=351 ms): membrane
    matches NEST to **machine precision between spikes** — median ~1e-3 mV, p95 ~0.1 mV
    after a ≤4-sample alignment; synchronization identical (last-20 ms RMS gap **0.530 mV
    BP vs 0.539 NEST**; g=5 nS: 0.005 vs 0.006). The ONLY divergence is an **O(dt) AP-edge
    timing jitter** (< 0.5 % of samples > 5 mV apart) — expected from the one-step lag at
    the spike's near-vertical upstroke.
  - **Inhibitory network** (N=200, ~24 gap edges/neuron, 4 seeds): Golomb-Rinzel coherence
    χ matches NEST distributionally at **async 0.137 BP / 0.139 NEST** (g=0) and **sync
    0.362 / 0.352** (g=0.7) — a ~2.6× synchrony rise on **both** sims.
- **A resolved → option (a) (full-lag difference deposit).** The 2-neuron micro-parity GATE
  confirmed option (a) reproduces `use_wfr=False` to machine precision between spikes, so the
  fallback option (b) (off-diagonal seam-H emission + neuron-side self-leak split) was **not
  needed**. Both terms lag: `G@V[n−1]` (off-diagonal, via the V emission holder) AND
  `−D·V[n−1]` (self term, the same lagged V). Lagging **both** keeps the rest balance exact
  (`I_gap ≡ 0` when all V equal) and matches NEST; lagging only the off-diagonal would inject
  a spurious self-bias.
- **Gap = negated graph Laplacian on the current channel.** `G` is the dense **symmetric**
  hollow gap-conductance matrix (nS), `D = rowsum(G)`. `connect(synapse=gap_junction)`
  dispatches **before** the plastic/static paths to `_build_gap_coupling`, which materializes
  BOTH edge directions (`A = (A | A.T) & ~eye` — NEST's `make_symmetric` / all-to-all
  bidirectionality), scales by one scalar `g`, caches `(G, D, V-reader, post, key)`, and the
  phase loop deposits `_gap_current(...) * u.pA` via `add_current_input` (the
  `sum_current_inputs(x, V)` seam the gap neurons already read). `nS·mV = pA` exactly. No
  EventProjection, no rule kernel, no WFR.
- **NEST freezes gating on `SetStatus(V_m)`; the port equilibrates per neuron — reconcile by
  overriding to resting.** The gap demos perturb a cell's `V_m` AFTER `Create`; NEST's
  `SetStatus` does **not** recompute gating, so the perturbed cell keeps **resting** gating
  (`eq(-69.604 mV)`). The port's convention is `eq(V_m_init)` **per neuron** — faithful but
  different. For parity the validation/example helpers override gating to the resting
  equilibrium (`_resting_gating()` → `Act_m_init=…`), reproducing NEST's construct-then-
  perturb workflow exactly. (A g=0 pair == two independent lone runs to 1e-9 confirms the gap
  itself is innocent of any IC effect — it was the diagnostic that isolated the init bug below.)
- **Latent per-neuron gating-init bug found + fixed (both gap neurons).** Surfaced by the gap
  demo's heterogeneous `V_m_init`: `init_state` equilibrated `m/h/n[/p]` at a SINGLE broadcast
  voltage (`get_mantissa(V).flat[0]`) instead of per neuron, so a population with heterogeneous
  initial voltages started with WRONG and IDENTICAL gating (a −65 mV cell in a pair plunged to
  −88 mV vs −64 mV lone). Fix: equilibrate at EACH neuron's own `V_m_init` (vectorized
  `_hh_*_equilibrium(V_arr)`). **Backward-compatible** — scalar `V_m_init` is unchanged; all
  prior neuron tests pass. Guarded by a per-neuron heterogeneous-gating regression in each
  `*_test.py` (the Traub `m` gate barely varies over [−70,−55,−45] mV — assert on the `n` gate
  instead, ptp > 1e-5).
- **Symmetry (B) enforced at connect; out-of-scope shapes rejected loudly.** NEST
  `gap_junction` is `REQUIRES_SYMMETRIC=True`. The coupler rejects, each with a NEST-free guard
  test: a non-recurrent connect (pre≠post), a `delay` (gap is instantaneous), `comm='sparse'`
  (the `G@V` is a dense matmul), a callable **or** non-scalar conductance (one scalar `g` per
  graph), and a post that doesn't emit `V`.
- **Coverage: the cluster-22 `--source` SIGABRT bites here too; behavioral fallback for
  `_simulator.py`.** `coverage run --source=<pkg>` SIGABRTs at **collection** (jaxlib absl
  double-init, entry 22). The documented `--source`-drop workaround (`coverage run -m pytest`
  then report-time `--include`) instruments the two **standalone neuron modules** cleanly:
  **`hh_psc_alpha_gap.py` 99 %** (2/202 miss), **`hh_cond_beta_gap_traub.py` 96 %** (8/226 miss)
  — the misses are pre-existing non-gap branches. `_simulator.py` still can't be
  line-instrumented (entry 22), so the gap seam relies on **behavioral coverage**: every gap
  branch (the 3 `_gap_conductance` reject modes, the 4 `_build_gap_coupling` guards, the
  difference-current arithmetic, the symmetric-matrix build, the dispatch, the phase injection)
  has an exercising NEST-free test. `clear_caches()` + x64 per stiff-HH test class (entry 21).
- **For next clusters:** the seam-(H) `_emission_attr` holder now carries BOTH a
  **continuous-rate** deposit (15a, delta channel) AND a **voltage difference-current** deposit
  (15b, current channel) — it is the single substrate for any state-gated coupling; declare
  `_emission_attr` + deposit `comm='dense'`. The gap coupler is **dense-only by design**; a
  sparse `G@V` for large nets is the natural follow-up. `hh_cond_beta_gap_traub` rides the SAME
  seam but only `hh_psc_alpha_gap` has a live-NEST gap-parity test — a cond_beta gap regression
  is the small remaining gap (noted in `neurons-gap.md` / `numerical-validation-gap.md`).

### 15d-astro — 2026-06-15

- **Shipped:** the **bidirectional neuron↔astrocyte SIC loop on the JAX substrate** — the
  **last bucket-3 *model* cluster** (the one remaining bucket-3 item is the Siegert
  `diffusion_connection`, still deferred to 15c). `astrocyte_lr_1994` emits
  its slow-inward current as seam-(H) continuous graded **current** emission; a one-way
  `sic_connection` deposits `weight·SIC` into `aeif_cond_alpha_astro`'s labelled `'I_SIC'`
  current channel through a new `as_current` `EventProjection` mode; the neuron→astro arm
  stays the ordinary delta path (`Δ_IP3·w` IP3 via `sum_delta_inputs`). The host-side
  `_sic_queue` event-emulator on the neuron and `sic_connection`'s host-queue coeff-array
  API are **deleted** (the de-queue), so the whole loop lowers under `Simulator.simulate`.
  Files: `_nest/astrocyte_lr_1994.py` (`_emission_attr='SIC'`, `_emission_continuous`,
  `_emission_current`, `_emission_current_label='I_SIC'`; spike read via `sum_delta_inputs`),
  `_nest/aeif_cond_alpha_astro.py` (de-queue: 6 host-queue helpers + `_sic_queue`/`_sic_step`
  removed, `sic_events` kwarg dropped; labelled `I_SIC` current read before the unlabelled
  `I_stim` read), `_nest/sic_connection.py` (host-queue API trimmed 991→664 lines → thin
  NEST-parity status spec), `_network/_event_proj.py` (`as_current` deposit mode),
  `_network/_simulator.py` (`_connect_sic` dispatch + `'I_SIC':('I_sic',)` recordable alias).
  Tests: `_event_proj_current_test`, `astrocyte_lr_1994_test` (+emission), `aeif_cond_alpha_astro_test`
  (+SIC channel), `_simulator_sic_test`, `sic_connection_test` (de-queue + validators),
  `_nest/_validation/astrocyte_sic_test.py` (live-NEST parity). Branch
  `worktree-nest-goal+15d-astro`.
- **Parity (vs live NEST 3.9.0, dt=0.1ms; exact-after-`align_steps`):**
  - **SIC-response micro** (spike_generator → astro → sic → post): IP3 `2.4e-5`, Ca
    `1.9e-4`, I_SIC `2.3e-4` aligned max|Δ|.
  - **Driven loop** (aeif `I_e=1000pA` → astro → sic → post, Ca crosses SIC_th): IP3 `0.0`
    (exact), Ca `9.9e-7`, I_SIC `6.0e-4`, V_pre `3.7e-4 mV`.
  - **Astro-network distributional** (N post `I_e=700pA`; Poisson → astro → sic → post):
    seed-mean post rate `NEST 9.0 = BP 9.0` (SIC off) → `14.0 = 14.0` (SIC on) — the
    SIC-raises-firing law (+5 Hz), **identical** on both sims, near-zero seed variance.
- **A resolved → option (a) `synapse=sic_connection(w)`.** The SIC crosses the seam as a
  **continuous graded current**, not a discrete event. `Simulator.connect(astro, neuron,
  synapse=sic_connection(weight=w))` dispatches (via `isinstance`) to `_connect_sic`, which
  enforces the NEST sender/receiver contract (`sic_connection.check_connection`), reads the
  astrocyte's `_emit_holder`, and builds an `as_current` `EventProjection` into the post's
  `'I_SIC'` channel. No bespoke SIC primitive — the existing seam-(H) emitter + one new
  deposit-mode flag suffice.
- **API discovered/changed (reuse downstream):**
  - **`EventProjection(..., as_current=False)`** — when `True`, deposits the dense graded
    contribution via `post.add_current_input(key, contrib, label=...)` instead of
    `add_delta_input`. The route for any **pA current that enters `dV/dt`** (vs a
    delta/conductance). Rejects `comm='sparse'` (binarises the presynaptic value).
  - **`_emission_current=True` + `_emission_current_label='I_SIC'`** on an emitter pop tells
    `_connect_sic` to route into a labelled current channel; the generic
    `create()`/phase-2 holder capture already handles any `_emission_attr` pop (no change).
  - **`sic_connection.delay_steps`**: NEST's default `delay=1.0 ms` is **10 steps** at
    dt=0.1, so `delay_steps=10` rides it; `delay_steps=1` (min delay) rides only the
    intrinsic one-step pipeline lag. `(delay_steps−1)` extra steps = `InputDelay` (mirrors
    the deleted queue's `base_offset`).
- **Gotchas:**
  - **Read-order collision — labelled before unlabelled.** The post's labelled `I_SIC`
    current read (`sum_current_inputs(0, V, label='I_SIC', pop=True)`) **must precede** the
    unlabelled `I_stim` read (`sum_current_inputs(x, V)` with `label=None`, which sums **all**
    current channels). Reversed, the device `I_stim` read would consume + double-count the
    SIC deposit. This was the Phase-3 RED (`units do not match: pA != 1`).
  - **Current, not delta.** SIC is a pA current; depositing it on the default delta channel
    is silently wrong. `as_current=True` is mandatory; `comm='dense'` is mandatory (graded).
  - **`sum_delta_inputs(spike_weights)` is backward-compatible**: returns its arg unchanged
    when no delta inputs are registered, so the astrocyte's standalone `update(spike_weights=…)`
    still works while the Simulator's neuron→astro deposits now also reach IP3.
  - **Recordable alias gap.** The Simulator's multimeter needs `_RECORDABLE_ALIAS['I_SIC'] =
    ('I_sic',)` to map the NEST recordable name to the State; the astro cluster had no
    Simulator integration before, so this was a genuine pre-existing hole.
  - **Coverage on touched lines** (whole-file is misleading on the shared `_event_proj.py`
    60 % / `_simulator.py` 53 %): touched lines covered; `sic_connection` 98 %, aeif 94 %,
    astrocyte 91 %.
- **For next clusters:** every bucket-3 **model** is now on the substrate; the sole
  remaining bucket-3 item is the Siegert `diffusion_connection` (still carries the host
  `_queue`; supplies network `(μ, σ²)`) — **15c**. **§3.8 astrocyte demos are unblocked
  (→ 17b):** `astrocyte_single` /
  `astrocyte_interaction` are substrate-ready (the exact loop is validated); `small_network`
  / `astrocyte_brunel_*` additionally need NEST's `TripartiteConnect` astrocyte-pool rule
  (`third_factor_bernoulli_with_pool`) — **out of scope** (no new connectivity rule). Still
  unwired everywhere: **sparse graded emission** (only `comm='dense'` graded deposit exists).

### 15a-rate-core — 2026-06-15

- **Shipped:** the **rate-neuron core on the JAX substrate** — 11 rate neurons +
  2 rate connections rebuilt on seam-(H) continuous graded emission, the host
  dict-queue event emulator **deleted**, `mult_coupling` dual-channel coupling, and a
  pre/post φ-homogeneity guard. Per the cluster decisions the de-queue also covered the
  3 extra files (`rate_transformer_node`, `siegert_neuron`, `step_rate_generator`).
  Files: `_nest/{lin_rate,gauss_rate,sigmoid_rate,sigmoid_rate_gg_1998,tanh_rate,
  threshold_lin_rate,rate_neuron_ipn,rate_neuron_opn,rate_transformer_node,
  siegert_neuron,step_rate_generator}.py` (seam-(H) markers + JAX φ; connection I/O
  removed), `_nest/rate_connection_{instantaneous,delayed}.py` (host-queue API trimmed →
  pure NEST-parity status specs), `_network/_simulator.py` (`_is_continuous_rate`,
  `_check_rate_phi_homogeneity`, `_sign_split_weight`, `_build_rate_dual_channel`),
  `_network/_event_proj.py` (`channel_label` kwarg). Validation (`_nest/_validation/`):
  `rate_coupling_micro_parity`, `rate_network_parity`, `rate_delayed_connection_parity`,
  `rate_mult_coupling_parity`, `rate_nonlinearity`, `rate_core_forloop`,
  `rate_core_no_nest`, `rate_core_edge_cases`, `rate_generator_source`,
  `rate_transformer_node_substrate`, `siegert_substrate` (110 tests + 127 subtests
  green). Branch `worktree-nest-goal+15a-rate-core`.
- **Parity (all vs live NEST `use_wfr=False` unless noted):**
  - **Linear-rate network FP** relaxes to the analytic `r* = (I − gC)⁻¹μ` to **1e-4**
    against the **closed form AND** NEST, for a 2-neuron mutual loop, a 3-neuron
    mixed-sign net, and a zero-coupling control (`r*=μ`). The recurrent loop solution is
    genuinely ≠ the one-pass feed-forward estimate — the substrate finds the loop FP.
  - **All 5 φ formulas == NEST `nonlinearities_*::input()` C++ exactly** (lin `g·h`;
    gauss `g·exp(−(h−μ)²/2σ²)`; sigmoid `g/(1+e^{−β(h−θ)})`; tanh `tanh(g(h−θ))`;
    threshold `min(max(g(h−θ),0),α)`; sigmoid_gg `(g·h)⁴/(0.1⁴+(g·h)⁴)`); nonlinear
    recurrent steady-states match NEST.
  - **`mult_coupling` dual-channel** matches the closed form `r*=(μ+g_ex θ_ex A+g_in θ_in
    B)/(1+g_ex A−g_in B)` and NEST; **siegert** Siegert-rate == NEST ref `27.1095934379`.
- **A resolved — receptor-optional continuous seam-(H).** Rate coupling crosses the seam
  as a **continuous** graded emission, not a discrete event: a rate neuron declares
  `_emission_continuous=True` + `_emission_attr='rate'` (`linear_summation=True`) or
  `'phi_rate'` (False); the connection deposits `weight·rate` into the post's **default**
  delta channel (`comm='dense'`, `receptor_type=None`), which the post reads via
  `sum_delta_inputs(0.0)`. No receptor routing, no host queue. For `mult_coupling` the
  weight is **sign-split** (`W_ex=max(W,0)`, `W_in=min(W,0)`) into two labeled
  EventProjections → `'rate_ex'`/`'rate_in'` channels, recombined as
  `P2·(H_ex·φ(Σ_ex) + H_in·φ(Σ_in))` with `H` from `_mult_factors`.
- **`linear_summation` is receiver-side, and that forces the homogeneity guard.** True →
  sender emits the raw `rate`, the receiver applies **its own** φ to the summed input (so
  heterogeneous φ across a projection is fine). False → the sender emits `phi_rate=φ_pre(rate)`
  and the receiver adds it directly — but NEST applies the **receiver's** φ in its event
  handler, so brainpy↔NEST agree only when pre and post share φ. Hence
  `_check_rate_phi_homogeneity` raises at `connect()` when the modes mismatch, or when both
  sides are `False` and `_phi_signature` differs (`(class, linear_summation, per-param
  distinct-value sets)`; the template appends `('input_nonlinearity', fn)` compared by
  identity).
- **De-queue: I/O seam swapped, dynamics untouched.** The deletion removed only the
  host-side event emulator (`prepare_secondary_event` / `to_rate_event` /
  `coeffarray_to_step_events` + `_to_coeff_array` / `_to_rate_value` helpers) from the two
  connections; the JAX update `τ Ẋ = −λX + μ + φ(h)`, `h = sum_delta_inputs(0)`,
  `μ = sum_current_inputs(x, rate)` was **unchanged**. The connection objects survive as
  thin NEST-parity status/param specs (`weight`, `delay`, `get/set_status`, delay
  rejection); the Simulator builds the actual routing from them.
- **Instantaneous lag = the WFR seed.** The substrate's one-step pipeline latency **is**
  NEST's `use_wfr=False` instantaneous seed: it **diverges on the transient** but
  **preserves the fixed point** — it breaks the otherwise-algebraic feedback loop
  `r0 ← r1 ← r0`. Trajectory parity needs `align_steps` to absorb the uniform integer
  offset; the FP needs nothing. **Delayed** = `InputDelay(delay_steps·dt)` on the same
  seam; `step_rate_generator` only emits `DelayedRateConnectionEvent` (no instantaneous
  output), so its NEST counterpart is `rate_connection_delayed` at the minimum delay,
  which equals the generator path's intrinsic one-step lag.
- **Gotchas:**
  - **gauss reuses `mu`/`sigma` as the φ centre/width AND the noise** — it cannot be
    driven noise-free (`sigma=0` is a divide-by-zero in φ), so its parity/edge tests keep
    `sigma>0` and assert distributionally / finiteness.
  - **`rate_neuron_opn` has no `lambda_` and no `rectify_output`** (output-noise template:
    `τ Ẋ = −X + μ + φ(h) + σξ`) — it is excluded from the zero-λ / rectify edge branches.
  - **siegert's step is EAGER, a documented `for_loop` exception** — its Siegert transfer
    is a host-side SciPy/Gauss-Legendre special-function integral on concrete values, so
    `update()` does **not** lower under `for_loop`/`jit`; drive it with an eager host loop.
    Its `_HAVE_SCIPY` is a **module global shadowed by the class re-export** — patch it via
    `sys.modules['brainpy_state._nest.siegert_neuron']`, not the imported class.
  - **Coverage on shared `_simulator.py` must be measured on the *touched* lines**
    (git-diff added lines = **90.7 %**), not whole-file (61 % — the rest is pre-existing
    spiking/device infra). Every rate-owned `_nest` file is **>90 %** (siegert 92 %; the
    residual is the no-SciPy quadrature split-boundary branches + 15c diffusion).
- **For next clusters:** the seam-(H) emitter + sign-split dual-channel + homogeneity
  guard is the reusable substrate for **15b/c/d** (any graded-state-gated coupling — declare
  `_emission_continuous`/`_emission_attr`, deposit `comm='dense'`). Still deferred: **siegert
  network diffusion → 15c** (the `DiffusionConnection` that supplies `(μ, σ²)`; in 15a these
  ride `update()`'s direct args), and **sparse graded emission** (only `comm='dense'` graded
  deposit is wired).

### 16-generator-demos — 2026-06-14

- **Shipped:** the **three §3.7 generator-pattern demos**, each a runnable
  `examples/nest/` script + a single test module carrying **both** a NEST-free
  structural class (always runs) and a live-NEST **distributional** parity class
  (`@requires_nest`):
  - `sinusoidal_poisson_generator.py` — inhomogeneous-Poisson drive
    `λ(t)=max(0,dc+ac·sin(2πf t+φ))` relayed through parrots; eager-`for_loop` drive.
  - `sinusoidal_gamma_generator.py` — gamma-process (order *m*) rate-modulated drive;
    the headline is ISI regularization **CV → 1/√m**.
  - `pulsepacket.py` — `pulsepacket_generator` emits Gaussian-jittered synchronous
    spike packets; packet width ∝ `sdev`, with the neuron-averaged membrane excursion
    checked against the **Diesmann analytical** Gaussian⊛PSP convolution.
  Files: `examples/nest/{sinusoidal_poisson_generator,sinusoidal_gamma_generator,pulsepacket}.py`,
  `brainpy_state/_nest/_validation/{sinusoidal_poisson_generator,sinusoidal_gamma_generator,pulsepacket}_test.py`,
  docs (`examples/nest/README.md` §3.7, `docs/nest-status/internal/examples-gap.md`).
  **§3.8 astrocytes are explicitly deferred** (need bucket-3 `sic_connection` + an
  astrocyte rate model — out of scope here). Branch `worktree-nest-goal+16-generator-demos`.
- **Parity (all distributional, category D — NEST per-thread RNG vs JAX/NumPy diverge
  sample-by-sample, so compare seed-aggregated statistics, never per-sample):**
  - **Poisson:** the seed-averaged per-bin population spike-count **autocorrelation**
    (which carries the modulation period) matches NEST element-wise within
    `CAT_D.autocorr_max_diff`. Standalone PSTH-vs-`λ(t)` corr > 0.85; the
    individual-vs-shared `individual_spike_trains` modes are both exercised.
  - **Gamma:** measured CV `[1.012, 0.708, 0.409, 0.316]` for orders `(1,2,6,10)`
    tracks `1/√m` `[1.000, 0.707, 0.408, 0.316]`; the order-3 modulated rate correlates
    0.968 with `λ(t)`. Both the CV law and the modulated rate match NEST.
  - **Pulsepacket:** the pooled spike-time std (packet **width**) tracks `sdev`
    (`compare_distributional(..., statistic='mean')`), and the per-step count **profile**
    matches NEST (smoothed corr > 0.93, ±8 % windowed mass, centroid aligned < 1 ms after
    a +1-step recorder align). The membrane excursion is verified **NEST-free** vs the
    analytical: windowed corr **0.9998**, peak 6.287 mV @ 511.8 ms vs 6.451 mV @ 511.2 ms.
- **API discovered/reusable:**
  - **Eager `for_loop` drive for a single multi-channel generator.** Build the generator
    once inside `environ.context(dt=…)`, then `transform.for_loop(step, times, idx)` with
    `step` closing over `environ.context(t=t, i=i)` → **one** trace, stacked `(n_steps, N)`
    output. This drives the `in_size=N` *individual-train* path directly (the Simulator
    fan-out demo does not use it); a `test_update_lowers_under_for_loop` regression pins
    the single-trace contract (cluster-12 discipline).
  - **`SpikeTime` population as a passive membrane drive.** Replay a host-generated spike
    matrix through the JAX `Simulator`: `create(SpikeTime, N, indices=, times=, weights=counts)`
    → `connect(one_to_one, weight, delay)` → `iaf_psc_alpha(V_th=1e9 mV)` (integrates without
    firing) → `voltmeter` → `res.trace(vm, 'V_m')`. Multiplicity rides in `weights`. This is
    the same `SpikeTime`-as-population-source seam cluster 21 found, reused here for a passive
    membrane readout rather than a plastic edge.
- **Gotchas:**
  - **JAX compile-cache accumulation reads as a hang (the big one).** Each `run_spikes`
    builds a *fresh* generator (new `State`s) in a *new* `environ.context`, so every call is
    a fresh JAX trace+compile. Gamma's `update()` lowers a **`while_loop`-in-`scan`**
    (rejection sampling) that is **costly to compile**; uncleared, those artifacts accumulate
    across tests in one pytest process until JAX cache lookups degrade into an apparent hang
    (gamma module > 400 s, faulthandler stack parked in scan compilation). **Fix:**
    `jax.clear_caches()` + `gc.collect()` in `tearDown` of *both* classes → gamma 23.7 s.
    Poisson lowers a **cheap `scan` (no `while_loop`)** so it merely *slowed* (71 s) rather
    than hanging — the same guard took it to 13 s. **Any parity test that rebuilds a fresh
    traced model per call needs this `tearDown`, especially anything lowering a `while_loop`.**
    (The earlier "parity-body-present ⇒ hang" bisection was a red herring: import-time
    reference retention merely nudged accumulation past threshold.)
  - **`pulsepacket_generator` is host-side** (NumPy `default_rng` + per-train `deque`
    queues + Python control flow), so `update()` is **not** JAX-traceable — CLAUDE.md rule
    10's premise (the model lowers into one XLA program) does **not** hold. The **eager host
    loop is the contract**; there is no `for_loop`-lowering test for it, only the membrane
    replay through `SpikeTime` is traced.
  - **`SpikeTime` needs JAX-backed inputs.** `SpikeTime.__init__` calls `brainunit.lax.sort`,
    which requires a JAX backend → pass `jnp.asarray(indices)` and `jnp.asarray(times) * u.ms`;
    plain NumPy inputs raise a `BackendError`.
  - **Centroid, not argmax, for distributional center alignment.** The argmax (mode) of a
    finite-sample Gaussian histogram is noise-dominated (wobbles ±1.3–3.8 ms across seeds);
    the **count-weighted centroid** is stable to ≤ 0.4 ms. Assert packet-center parity via the
    centroid moment, never the mode (the principled fix for a 1.3 ms argmax flake, not a
    loosened tolerance).
  - **`iaf_psc_alpha` warm-up transient.** `V_m` inits ~−66 mV and relaxes to `E_L` over
    ~100–150 ms; the analytical excursion is 0 there, so a *full-trace* correlation is
    swamped by the transient (corr 0.20). **Window around the pulse** (place it at t=500 ms,
    well past warm-up) → corr 0.9998. NEST's own demo only plots a window.
- **For next clusters:** **§3.7 generator demos are complete; §3.8 astrocytes remain**,
  blocked on bucket-3 (`sic_connection`) plus an astrocyte rate model — revisit post-bucket-3.
  The `clear_caches()` + `gc.collect()` `tearDown` is now the standing remedy for
  per-process compile-cache growth in fresh-model-per-call parity suites (it manifests as a
  hang, not a clean failure, so it is easy to misdiagnose).

### 21-urbanczik-dendritic — 2026-06-14

- **Shipped:** the **last plastic synapse on the legacy base** — `urbanczik_synapse`
  rebuilt as a frozen spec + pure `update(state, ctx) -> (new_state, w_eff)` kernel on
  `VoltageCoupledPlasticProj` (primitive #2; its **third user** after Clopath and
  dopamine), reading the post neuron's **dendritic** prediction error δΠ per edge. With it,
  the **§3.3 plasticity demos are complete (all 5)**: ported
  `examples/nest/urbanczik_synapse_example.py` (Urbanczik-Senn 2014 Fig 1B) and retired
  cluster-13's `NotImplementedError` placeholder. Files: `_nest/urbanczik_synapse.py` (spec
  + rule), `_nest/pp_cond_exp_mc_urbanczik.py` (δΠ/`V_W_star` States + `delta_label_for_receptor`
  routing seam), `_nest/spike_generator.py` (device-backed `spike_weights` fix),
  `_nest/_validation/{_urbanczik_drive.py,urbanczik_synapse_parity_test.py,urbanczik_synapse_example_test.py}`,
  docs (`examples-gap.md` §3.3, `synapses-plasticity-gap.md` §3, `neurons-gap.md`,
  `numerical-validation-gap.md`, `index.rst`, `examples/nest/README.md`). Branch
  `worktree-nest-goal-21-urbanczik-dendritic`.
- **Parity (deterministic chain, soma clamped silent so δΠ is the closed-form branch):**
  - dendritic `V_d` (the rule's only neuron input) matches NEST **sample-for-sample** (passive
    RC; `align_steps=2` absorbs a 2-step recorder offset on the synaptic onset; band 2e-2 mV).
  - recorded `V_W_star` and `delta_Pi` **== the closed-form NEST functions of `V_d`** to
    machine precision (`atol 1e-6`) → with `V_d`, they equal NEST transitively.
  - **weight trajectory** depresses and matches NEST **at every `weight_recorder` send step**
    (max |Δ| **~0.016 pA** over the 36-send curve; band 0.1 pA). The soma conductance teacher
    `V_s` matches NEST to **1.4e-14 mV**.
  - **demo learning** (upstream scale n_pg=200, 100 reps): rate prediction error
    `|φ(U)−φ(V_W*)|` **0.0239 → 0.0133 (ratio 0.557)**, RMS `|U_M−V_W*|` 6.30 → 4.11 mV,
    weights adapt bidirectionally (mean 90 → 54.6 pA, max 248).
- **API discovered/changed (reusable):**
  - **Named post-state reader on primitive #2.** A plastic rule declares
    `post_state_reads = ('<state>',)` and the `VoltageCoupledPlasticProj` substrate pulls that
    **named** post State per edge each step (`ctx.post_states['<state>']`). Primitive #2 now
    serves a **somatic** reader (Clopath's analog `V`) *and* a **dendritic-compartment** reader
    (Urbanczik's `delta_Pi`) **with no substrate change** — the rule names what it needs. Any
    future rule reading a *derived* neuron signal reuses this.
  - **Routing seam:** the neuron exposes `delta_label_for_receptor(receptor_type)` and
    `connect(..., synapse=<plastic>, receptor_type=k)` threads `k` to the right delta-input
    channel (dendritic exc = 3). Minimal + reused by the demo's soma teacher (receptor 1/2).
  - **Two-column pre-trace seam:** `pre_trace_tau = (tau_L, tau_s)` → the substrate keeps two
    per-pre traces, gathered as `ctx.pre_traces[:, 0]` / `[:, 1]`.
  - **`SpikeTime(n, indices=, times=)` as a population spike source** (JAX-backed inputs
    required — `brainunit.lax.sort`) → **one** `n→1` plastic projection drives `n` dendritic
    edges, vs `n` projections. The substrate lets a *device* drive a plastic edge directly (NEST
    needs a `parrot_neuron` relay).
  - **`spike_generator` `spike_weights` must be device-backed** (`jnp.asarray(get_mantissa(...))`):
    the per-step gather `spike_weights[idx]` with a traced index raises under `for_loop` if the
    array stays numpy (`u.math.asarray` of a numpy input stays host). This is the seam the
    time-varying somatic conductance teacher relies on (RED→GREEN regression added).
- **Gotchas:**
  - **float32 divergence + import-time cache contamination (the big one).** The *driven* soma
    (oscillating conductance teacher) is a **stiff** coupled ODE that **diverges in float32**
    (`V_s` → 7641 mV). `brainpy_state` traces some kernels at **import**; under pytest that
    happens during **collection, before any test enables x64**, so those kernels are cached in
    **float32**. Flipping `jax_enable_x64=True` later returns `True` and makes *fresh* arrays
    float64, **but the cached float32 kernels are still reused** → the sim diverges. Fix:
    `jax.clear_caches()` in `setUpClass` (+ pin `precision=64`) forces a float64 re-trace. The
    silent-soma **parity** test is float32-*stable in isolation* but **fails in the mixed state**
    (x64 flag on + float32 cache) when run *after* any test that enables x64 (e.g. the neuron
    unit test or the new demo test) — so it too had to pin float64 + `clear_caches()`.
    **Precision-fragility is collection-order-dependent**; any stiff/multi-compartment parity
    test needs this guard, not just a module-top `config.update`.
  - **Online vs event-driven weight.** The every-step kernel weight coincides with NEST's
    event-driven weight **only at presynaptic-send steps**; between/after spikes it drifts
    continuously and re-synchronises at the next spike. Assert parity **at send steps**
    (`sample_at_send_steps`), the Clopath precedent — a t=200 "13.6 % over-depression" was a
    false alarm from comparing my drifted weight to NEST's frozen-since-185 weight.
  - **Soma decoupling enables deterministic δΠ.** With `g_ps=0` the dendrite gets no somatic
    current, so `V_d`/`V_W_star`/`delta_Pi`/weight are **invariant to `soma_I_e`**; a strong
    hyperpolarisation (`SOMA_IE=-12000 pA`) pins **0 somatic spikes** → δΠ is the deterministic
    `-φ(V_W*)·dt·h` branch. **Potentiation needs somatic spikes (stochastic** point process), so
    the positive-δΠ branch is covered by the kernel unit tests + the demo smoke test, **not**
    deterministic live-NEST.
  - **Dendritic params are synapse-spec constructor args** (frozen before `connect`), defaulting
    to the `pp_cond_exp_mc_urbanczik` dendrite (`C_m=300 pF`, `g_L=30 nS`, `tau_syn=3 ms`);
    `tau_L=C_m/g_L`, and **`tau_s` is selected by the initial weight sign** (NEST `send()`), with
    sign-consistency constraints keeping the weight from crossing zero.
- **For next clusters:** **§3.3 is done** and **every plastic synapse is off the legacy base**.
  The **named post-state reader** is the pattern for any rule needing a derived neuron signal;
  the **`clear_caches()` + float64 guard** is mandatory for any test exercising a stiff or
  multi-compartment neuron under pytest (collection imports `brainpy_state` in float32 first). If
  many parity tests start sharing sessions, consider a root-level conftest that enables x64 +
  clears caches **once** before `brainpy_state` is imported — the per-test guard works but is
  repeated.

### 22-wang-nmda-network — 2026-06-14

- **Shipped:** **recurrent NMDA coupling through the `Simulator` API + the Wang (2002)
  winner-take-all decision network** — completing §3.6. The cluster-14 entry below
  deferred this as "needs a new offset-aware NMDA event projection"; **that turned out
  to be unnecessary** — generalizing the existing presynaptic-emission seam (H) was
  enough (design A → **option (a)**). Files: generalized
  `brainpy_state/_network/_simulator.py::_resolve_stp_emission`; the ported
  `examples/nest/wang_decision_making.py` (the deferred placeholder retired);
  `_network/_simulator_bw_receptor_test.py` (NEST-free seam unit tests),
  `_nest/_validation/iaf_bw_2001_recurrent_nmda_parity_test.py` (the design-A arbiter,
  live NEST), `_nest/_validation/wang_decision_making_test.py` (distributional WTA
  parity, live NEST), `_nest/_validation/wang_decision_making_no_nest_test.py` (CI
  companion). Docs: README §3.6 (Wang now the 6th port + the recurrent-NMDA seam +
  2 parity rows), `examples-gap.md` §3.6 (deferred→implemented, P2 marked DONE),
  `index.md`. Branch `worktree-nest-goal+22-wang-nmda-network`.
- **Parity:**
  - **Recurrent-NMDA micro-parity (the design-A arbiter, deterministic).** An
    *asymmetric* 3-cell pool (each neuron AMPA-fired at a distinct time, no autapses)
    wired `connect(pool, pool, receptor_type=NMDA, comm='dense')` reproduces NEST's
    recurrent `Connect(pool, pool, {receptor_type: 3})` **per-neuron `s_NMDA` to
    machine precision (max|Δ| ~ 5e-15 over every column)** and `V_m` to ~1e-3 mV. Every
    column compared (not just neuron 0) so a transposed/mis-routed weight matrix can't
    hide. ⇒ **option (a) confirmed: no bespoke offset-aware `EventProjection` needed.**
  - **Wang WTA decision (distributional — the attractor amplifies integrator/PRNG
    divergence, never per-sample).** Reduced mean-field-preserving net (ne=200, ni=50,
    weights ∝ N_full/N), seeds 1-3. **+102.4 coherence → A wins 3/3 on both sims**
    (late-window brainpy A~12 / B~1.4 Hz; NEST A~7 / B~1.2 Hz), mirror image at -102.4,
    zero coherence unbiased (|bias| < ½ the biased gap) on both. Asserted invariants:
    direction (±coh→A/B ≥2/3 seeds), WTA contrast (winner > 2.5× loser, loser < 4 Hz),
    zero-coherence unbias. The winner's **absolute** rate legitimately differs
    (~30-70 %) — genuine attractor amplification, not a wiring/coupling error (the
    coupling itself is the 5e-15 micro-parity above).
- **API discovered/changed (reusable):**
  - **Generalized presynaptic-emission seam (H) → multi-port graded routing.**
    `_resolve_stp_emission(pre, post, receptor_type, holder, comm)` now keys off two
    class attrs any emitting neuron declares: **`_emission_attr`** (the State holding
    the per-step graded emission, 0 off-spike) and **`_emission_receptor`** (the one
    NEST receptor that carries it). Over that receptor a static `connect` delivers
    `weight · emission` instead of `weight · spike`; **every other receptor and every
    non-emitting pre stays binary on its own channel.** Two post shapes: a **multi-port**
    post (`hasattr delta_label_for_receptor`/`n_receptors`, e.g. `iaf_bw_2001`) keeps
    the `receptor_type` so the graded value lands in the named NMDA delta channel; a
    **single-port** post (`iaf_tum_2000`) collapses it to `None` (the efficacy *is* the
    PSC) and requires the same model. `iaf_bw_2001` declares
    `_emission_attr='spike_offset'`, `_emission_receptor=NMDA`.
  - **Graded emission must ride `comm='dense'` (`x @ W`).** `comm='sparse'` binarizes
    the presynaptic value (it's an event/CSR path), so it is **rejected** for an
    emitting receptor with a clear error. Dense is required for any presynaptic-state-
    gated synapse.
  - **Recurrent realization works under `for_loop`/`scan` unchanged** — the emission
    holder is allocated in `create()` for any pop with `_emission_attr` and captured
    after the pre's `update()`; the carry shape is stable across steps.
- **Gotchas:**
  - **Pipeline-latency offset is uniform + benign.** The Simulator captures a pop's
    emission holder *after* its `update()`, so a recorded signal sits a fixed integer
    number of steps later than NEST's multimeter phase:
    **`brainpy_step = NEST_step + 1 (global multimeter phase) + 1 per projection hop`**.
    `V_m` (one hop, generator→cell) aligns at **shift 2**; `s_NMDA` (two hops,
    generator→sender→receiver) at **shift 3**. Absorb with `align_steps` (sub-ms lag on
    a 100 ms NMDA timescale is immaterial); the first probe failed at shift 0 (max|Δ|
    0.78) and matched at -3 (5e-15).
  - **Don't compare `I_NMDA` per-sample.** It is `s_NMDA · (V_m − E_ex) / Mg-block` — a
    product whose two factors carry *different* recorder phases, so no single shift
    aligns it (a comparison artifact, not a dynamics difference; both factors match NEST
    at their own shift). Compare `s_NMDA` and `V_m` separately.
  - **The SILENT-network bug: external Poisson needs `receptor_type=AMPA`.** First Wang
    smoke run gave all-zero rates because the background/signal `poisson_generator`
    connects omitted `receptor_type`, so the deposits never reached `s_AMPA` (an
    `iaf_bw_2001` reads only its named receptor channels — a connect with no receptor
    delivers nothing). Every external drive into a multi-receptor conductance neuron
    must name its receptor.
  - **`np.asarray(state.value)` raises on a unit `Quantity`** ("Only dimensionless
    quantities can be converted") — a false "it didn't accumulate" red herring (the
    recurrent NMDA *had* worked, s_NMDA=3.28 nS). Strip units first:
    `float((state.value / u.nS).max())`.
  - **`V_m` recorded just *below* `V_th` even on a spike.** NEST resets V before the
    multimeter samples, so the recurrent-NMDA `V_m` sanity bound is `> E_L + 5` (not
    `> V_th`); spiking is confirmed by the `s_NMDA` accumulation, not the recorded peak.
  - **coverage.py cannot instrument `brainpy_state._network._simulator`.** Measuring its
    import **SIGABRTs the interpreter inside jaxlib's native `xla_client`** (both the C
    tracer and `COVERAGE_CORE=sysmon`). The pure-Python example module measures fine
    (testable API 100 % after a B-winner edge test + `# pragma: no cover` on the
    full-scale `main()` demo driver); the simulator seam's branches are verified
    **behaviourally** by the passing NEST-free seam suite, not by a line-coverage number.
  - **CPU platform.** All test/example headers pin `brainstate.environ.set(platform=
    'cpu')` + `jax_enable_x64` (per the session directive); the 18-run live-NEST WTA grid
    takes ~9 min on CPU.
- **For next clusters:** **§3.6 is complete.** The generalized emission seam is the
  reusable substrate for **any presynaptic-state-gated synapse** (declare
  `_emission_attr` + `_emission_receptor`, wire `comm='dense'`); reuse the
  pipeline-latency `align_steps` convention and the distributional (direction +
  contrast, not absolute rate) pattern for any attractor/WTA network. Still open and
  *separate*: `nest_compat.CollocatedSynapses` (express AMPA+NMDA on one pair in a
  single call — the Wang port uses two ordinary `connect()` calls, which is fine).

### 23-connection-introspection — 2026-06-14

- **Shipped:** **post-hoc connection enumeration + introspection** —
  `Simulator.get_connections(source, target, synapse) -> SynapseCollection`, NEST's
  `GetConnections` / `SynapseCollection` idiom: enumerate realized edges and read or
  write per-edge `weight`/`delay`/`source`/`target` **without holding each
  `Projection` handle**. An additive convenience layer — **no edge-storage change**.
  With it the **§3.4 §-introspection demos are done**: ported
  `examples/nest/plot_weight_matrices.py` (four `E→E/E→I/I→E/I→I` weight matrices via
  `get_connections(src,tgt)` + `np.add.at`) and `examples/nest/synapsecollection.py`
  (the full idiom tour), retiring both cluster-deferred `NotImplementedError`
  placeholders. Files: `_network/_connection_introspection.py` (the core, 280 stmts:
  `ProjEdges`, `SynapseCollection`, per-family enumeration), `_connection_introspection_test.py`
  (29 NEST-free), `_network/_event_proj.py` + `_event_plastic.py` (added
  `realized_edges()`), `_network/_simulator.py` (`_connections` registry +
  `get_connections()`), `_network/__init__.py` + `brainpy_state/__init__.py` (export
  `SynapseCollection`); `_rules.py` + `_rules_test.py` (Phase-0 `fixed_total_number`
  rule); rewritten validations `_nest/_validation/{plot_weight_matrices,synapsecollection}_test.py`
  (each 6 NEST-free + 3–4 live-NEST). Docs: README §3.4 (blocked→implemented prose,
  **seam F**, 2 parity rows + prose). Branch `worktree-nest-goal+23-connection-introspection`.
- **Parity:**
  - **`plot_weight_matrices` (5 seeds, live NEST).** Connectivity is **exact** on both
    sides — `fixed_indegree(K)` gives every post exactly `K` inputs, so edge counts
    (`K·n_post`) and per-target in-degrees match bit-for-bit (`np.bincount == CE/CI`).
    The `Normal(20,0.5)` / `−g·Normal` weight **draws** match NEST only as a seed-mean
    (`CAT_D`): `w_ex` mean ≈ 20 pA, `w_in` mean ≈ −100 pA, and `mean(w_in) ≈ −g·mean(w_ex)`.
  - **`synapsecollection` (5 seeds, live NEST).** Deterministic-count rules are exact:
    `one_to_one` per-edge `set` round-trips to **identical `[1..10]` pA on both sims**,
    `all_to_all` count == `n_pre·n_post`, the `stdp` model-filter count == **65**
    (`one_to_one 5 + all_to_all 5×12`) on both. The `Uniform(0.5,4.5)` weight mean
    matches NEST distributionally (`CAT_D`). The random topology of
    `pairwise_bernoulli`/`fixed_total_number` PRNG-diverges, so only their counts are
    compared (not realized pairs). **49 NEST-free unit tests, 99 %** line coverage on
    the API module.
- **API discovered/changed (reusable):**
  - **`realized_edges()` — one additive accessor per projection family.** Returns a
    frozen `ProjEdges(source, target, weight, delay, is_homogeneous_weight, is_plastic,
    model_name, write_weight, write_delay)`. `EventProjection` implements **all four
    comm modes** (dense `_W`, sparse CSR, per-receptor scatter `_ReceptorScatter`,
    `one_to_one` scalar); `EventPlasticProj`/`VoltageCoupledPlasticProj` enumerate
    their CSR `_pre_idx`/`_post_idx`. Centralized in `_connection_introspection.py`
    (lazy-imports `_DenseMatMul` to dodge the import cycle). **Reuse for any future
    `nest_compat.GetConnections` facade** — a thin global-node-id translator wraps this
    with zero edge-storage change.
  - **`Simulator._connections` registry** `(pre_pop, post_pop, model_name, proj)`
    recorded at every `connect()` — *required* because neither projection family
    stores `self.pre`. `model_name` = `type(proj.rule).__name__` for plastic, else
    `'static_synapse'`.
  - **Lazy `SynapseCollection`** stores per-proj `(proj, kept_idx, source_local,
    target_local)`; `weight`/`delay` are **re-read live** on each `get` (so a
    post-`simulate` read reflects evolved weights). Canonical edge order = stable
    argsort of population-local source (deterministic → cached `kept` stays valid).
  - **`set` guards (all-or-nothing, validated before any write):** per-edge weight on a
    homogeneous-weight projection → `ValueError`; any weight write on a weight-evolving
    plastic (`model_name ∉ {static_synapse, static_synapse_hom_w}`) → `ValueError`
    ("rule-managed"); delay on a plastic → `ValueError`; a set delay **grid-rounds to
    `dt`** and rebuilds/clears the `InputDelay` seam (NEST stores delays as integer
    resolution multiples).
- **Gotchas:**
  - **The cluster-22 coverage SIGABRT has a workaround.** The abort is jaxlib's absl
    (`SetTimeZone() has already been called`), and it is triggered by coverage's
    **`--source=<pkg>` PRE-IMPORT** (a second absl init), *not* by tracing. Drop
    `--source`: `coverage run -m pytest <files>` then `coverage report
    --include="*_connection_introspection.py"` (report-time scoping never imports) →
    **99 % on the pure-Python API module**, no crash, any tracer. (The `_simulator`
    seam itself still can't be line-instrumented, consistent with entry 22 — but the
    introspection module is standalone and measures cleanly.) `pytest-cov`'s `--cov`
    plugin still SIGABRTs at collection; use bare `coverage run` + `pytest`.
  - **Plastic projections reject `dist.*` Parameter weights** — plastic `__init__`
    does `jnp.asarray(rule.weight)` → `TypeError` on a `Normal`/`Uniform`. Only the
    **static** path samples distributions eagerly at connect. So distributional weights
    ride the static `all_to_all`; plastic `stdp` connects take concrete scalars (the
    rule evolves them at `simulate` anyway).
  - **`one_to_one` is homogeneous** (one shared scalar EventProjection) → refuses
    per-edge `set`. NEST's `one_to_one` uses the per-edge `static_synapse`; to keep that
    per-edge pedagogy, route the demo's `one_to_one` block through the **`static_synapse`
    plastic path** (`rule=one_to_one, synapse=static_synapse(...)`) — same realized
    connectivity, per-edge weights.
  - **Pre-sim vs post-sim weight backing.** Plastic `weight` is a `brainstate.State`
    allocated in `init_state` (runs at `simulate`); **before** simulate a `set`/`get`
    must target `_w_init`, **after** it the live `weight.value`. `realized_edges()` and
    the writer both branch on `isinstance(getattr(proj,'weight'), State)`.
  - **Population-local source/target** — brainpy has no global node-id space, so the
    matrices index directly with **no `−min(node_id)` offset** (NEST's). Membership
    filtering is by `id(population)` **and** local index.
- **For next clusters:** **§3.4 is complete** (all 7 demos: 5 recording/device + 2
  introspection). The `realized_edges()` + `SynapseCollection` pair is the substrate
  the planned `nest_compat` facade should wrap (add `GetConnections` global-node-id
  translation on top, no storage change). Still open and *separate*:
  `SynapseCollection.set('source'/'target')` (rewiring) is deliberately unsupported —
  the layer is introspective, not a graph editor.

### 14-network-demos — 2026-06-14

- **Shipped:** the **§3.6 spiking network-demo cluster** — 5 ports on the `Simulator`
  API, each with a live-NEST **distributional** parity test + a no-NEST CI companion,
  plus the Wang decision neuron validated and its network deferred. Files:
  `examples/nest/{artificial_synchrony,repeated_stimulation,sensitivity_to_perturbation,
  ei_clustered_network,brette_et_al_2007,wang_decision_making}.py` and matching
  `_nest/_validation/*_test.py` (+ `iaf_bw_2001_nest_parity_test.py`). One neuron seam
  fix (`iaf_cond_exp` multi-receptor), one connection rule (`pairwise_bernoulli`,
  Phase 0), and a cross-link from `examples/brainpy_like/106_COBA_HH_2007.py` (the HH
  benchmark-3 sibling) to the new IF benchmarks 1&2. Docs: `examples/nest/README.md`
  §3.6 demos+validation sections, `examples-gap.md` §3.6 table. Branch
  `worktree-nest-goal+14-network-demos`.
- **Parity (all distributional — chaotic/balanced/metastable nets never per-sample):**
  - `repeated_stimulation` — per-trial active-window spike count vs NEST, **`CAT_D` 5 %**
    (≈ `rate·(stop−start)`); zero-rate → silent (0).
  - `artificial_synchrony` — Golomb–Rinzel Σ = var_t(mean_n V)/mean_n(var_t V): uncoupled
    baseline matches NEST exactly, coupling lifts Σ on **both** sims (monotone law); the
    sensitive synchronized strengths compared in a **documented ~10 %** distributional band
    (grid-degenerate volleys flip cell-assignment under a sub-ULP integrator difference).
  - `sensitivity_to_perturbation` — AI-state rate **brainpy 14.95 vs NEST 15.17 Hz
    (1.45 %)**, `CAT_D`; the 1-spike perturbation decorrelates **> 0.9** of the network on
    *both* sims after `t_stim` and is **0** before (chaotic divergence is qualitative — the
    perturbed neuron/connectivity PRNG-differ, so per-seed divergence is not matched).
  - `ei_clustered_network` — rep=1 homogeneous **median** E/I rate within **12 %**
    (measured ~1–3 %) and median ISI-CV within **8 %** (measured < 4 %); rep=6 clustering
    **signature** required in both sims (`std6 > 3·std1`, `CV6 > CV1`). Median, not mean —
    1-in-N seeds falls into a globally synchronized state (CV≈0) that NEST does not share at
    that seed, and the median is immune to that outlier while the mean is not.
  - `brette_et_al_2007` — **second-half (steady-state) population rate**, seed-mean over
    3 seeds: COBA (`iaf_cond_exp`) bp 13.77/14.03 vs NEST 15.12/14.44 Hz (E **8.9 %** / I
    2.8 %, band **15 %**); CUBA (`iaf_psc_exp`) bp 3.97/3.99 vs NEST 4.03/4.01 Hz (E
    **1.5 %** / I 0.5 %, band **12 %**); both self-sustain (late E > 1 Hz) after the kick.
  - `iaf_bw_2001` (Wang neuron) — single-cell AMPA+GABA `V_m`/`s_AMPA`/`s_GABA`/`I_*` and
    the two-neuron NMDA presynaptic-offset coupling (`s_NMDA`/`I_NMDA`/`V_m`) both match
    **live NEST to machine precision** (direct alignment `bp[i]==nest[i]`).
- **API discovered/changed (reusable):**
  - **`iaf_cond_exp` multi-receptor seam (extends cluster-11 seam F to a conductance LIF).**
    Added `n_receptors=2`, `receptor_input_unit=u.nS`, and a `w_by_rec` branch in `update`:
    receptor **1 → `g_ex`**, **2 → `g_in`**. Before this `iaf_cond_exp` only read the legacy
    `sum_delta_inputs(label='w_ex'/'w_in')` deltas — *invisible* to `sim.connect()`, so any
    Simulator connection into it silently delivered nothing. The Simulator blob-bridge
    (`_simulator.py:764-771`, `inspect`-detects `n_receptors` + `'w_by_rec'` in the
    signature) needed **no change** — the model just had to opt in. **In brainpy the
    inhibitory conductance weight is a *positive* magnitude** (`COBA_W_I = 67 nS`), reversal
    handled by `E_in`; NEST's own `iaf_cond_exp` instead splits ex/in by weight *sign* and
    has no receptor ports — so the demo routes by `receptor_type` on our side and by sign on
    NEST's. (`iaf_psc_exp`/CUBA splits by sign on *both* sides — inhibitory weight negative,
    no receptor.)
  - **`pairwise_bernoulli(p)` ConnRule** (Phase 0, TDD) — exposes the existing Bernoulli
    sampler as a named rule; used by `ei_clustered` / `sensitivity` and broadly reusable.
  - **Population-rate idioms (Brunel family, now reused for E/I spiking nets):**
    `sim.connect(ne, ne+ni, rule=fixed_indegree(CE), comm='sparse', allow_multapses=True,
    seed=...)` (population-concat target, CSR comm for N=4000); a `poisson_generator(Nstim,
    …)` kick wired `one_to_one` into `e[:Nstim]` reproduces NEST's independent-per-target
    Poisson; `res.rate(esr.segments[0].population)` / `res.spikes(node)` for rate/raster.
- **Gotchas:**
  - **NEST `V_m` default ≠ `E_L` (the −70 trap).** NEST defaults `iaf_cond_exp`/`iaf_bw_2001`
    `V_m` to the **model default −70 mV**, not to `E_L`. Setting `E_L=-60` alone leaves the
    trace starting at −70 and relaxing toward −60 (this surfaced as a 9.9 mV single-cell
    mismatch at index 0). **Pin NEST `V_m` to brainpy's `V_initializer`** for any single-cell
    parity. Network *rate*-band parity is IC-insensitive (kick + recurrence wash the start
    out), so the demo `main()` may leave NEST's default.
  - **`iaf_bw_2001` aligns *directly* (`bp[i]==nest[i]`, `align_steps=0`)** — unlike the RKF45
    conductance models (`iaf_cond_exp`, `aeif`) that need the candidate's `t=0` dropped +
    `align_steps=1`. A first probe that dropped `bp[0]` falsely showed a 0.54 mV mismatch;
    sample-by-sample the streams line up with no drop.
  - **NEST forbids `spike_generator → NMDA`** (`IllegalConnection`: the NMDA *sender* must be
    an `iaf_bw_2001`, because the deposit reads the sender's `spike_offset`). Validate NMDA
    with a **two-neuron** setup (force a sender to fire, project NMDA to a receiver); brainpy
    replays it feed-forward by reading the sender's per-step `spike_offset` and depositing
    `weight·offset` one delay step later.
  - **NMDA deposits `weight · sender_spike_offset`, not `weight · spike`** — a *presynaptic*-
    state-gated synapse. The generic Simulator event projection deposits a uniform
    `weight·spike` and has no path to fold a per-presynaptic-neuron state into the deposit, so
    **recurrent NMDA needs a new offset-aware event projection.** This is the genuine
    blocker for the Wang *network* (its attractor lives on recurrent NMDA); the neuron and the
    feed-forward coupling are fully validated, so the demo ships the building block + the
    precise seam gap (goal-sanctioned deferral).
  - **Steady-state (2nd-half) rate is the kick-independent parity observable.** Brette's
    full-window rate is dominated by the ignition transient whose *magnitude* differs between
    sims (a kick-response detail, not the benchmark's point); compare the rate over
    `[simtime/2, simtime]` (NEST via `spike_recorder(start=simtime/2)`).
  - **Sensitive / chaotic / metastable observables → distributional, never per-sample.** Σ
    (synchrony), divergence (perturbation), cluster occupancy (ei_clustered) all PRNG-diverge;
    assert a seed-**mean** or **median** in a documented band + a qualitative law (Σ monotone↑,
    divergence ≈0-then->0.9, clustering signature). `ei_clustered` additionally needs the
    **median** (mean is poisoned by the rare synchronized seed) and **`rep < Q`** to keep the
    out-cluster weight `J-` positive.
  - **External CPU contention.** A parallel agent's `/tmp/gif_driver.py` (a `gif_population`
    test loop) thrashed the shared 20-core WSL2 box (~18× slowdown + swap) during Phase 2;
    not mine to kill. Worked around by confirming the *identical*-harness CUBA parity GREEN +
    arithmetic COBA probe, deferring the in-pytest COBA confirmation to the Phase-4 full run.
- **For next clusters:** the **Wang network is the one §3.6 blocker** — it needs the
  **offset-aware (presynaptic-state-gated) NMDA event projection** above; when that lands the
  network ports on the now-validated `iaf_bw_2001` + the `receptor_type` routing seam. The
  remaining §3.6 items are out of scope by spec: `lin_rate_ipn_network` / `rate_neuron_dm`
  (rate-neuron primitives), `gap_junctions_*` (gap-junction coupling), `intrinsic_currents_*`
  (`ht_neuron`). Reuse the `iaf_cond_exp` receptor seam for any conductance E/I net and the
  steady-state-rate + seed-mean/median distributional pattern for any balanced/metastable net.
### 12-single-neuron-models-2 — 2026-06-14

- **Shipped:** the **§3.5 single-neuron model-demo cluster, part 2** — 9 more
  `examples/nest/` ports completing NEST §3.5 (now **16** total), each with a
  live-NEST trace-parity test or a documented carve-out in `_nest/_validation/`:
  `iaf_tum_2000_short_term_{depression,facilitation}` (presynaptically-integrated
  Tsodyks–Markram STP), `izhikevich` (RS/IB/CH/FS), `mat_psc_exp`
  (`mat2_psc_exp` + an active-`V_th_v` `amat2_psc_exp` config), `mc_neuron`
  (`iaf_cond_alpha_mc`, full device→compartment-receptor routing), `CampbellSiegert`
  (analytic mean/var/rate cross-check), `BrodyHopfield` (sinusoidal-drive
  phase-locking), `gif_population` (microscopic GIF network), `gif_pop_psc_exp`
  (mesoscopic population-rate model vs microscopic network). Two new seams: **(H)**
  presynaptic-STP emission and **(I)** mc device→compartment-receptor routing.
  Foundation: callable recordable aliases (`U_m`, composite `V_th`, per-compartment
  `V_m.{s,p,d}` / `g_ex/g_in.*`). Docs: README §3.5 prose + parity/carve-out tables;
  this entry. Branch `worktree-nest-goal+12-single-neuron-models-2`; PR #58.
- **Parity:**
  - deterministic (trace, generous constant integer-step align where a device
    carries NEST's 1 ms delay): `izhikevich` `V_m`/`U_m` `CAT_A` (~1e-3 mV) **zero
    shift** + exact counts (RS<IB<CH<FS) under an `I_e` drive; `mat_psc_exp`
    non-resetting `V_m` + composite `V_th` float-noise floor zero-shift + exact
    counts (`amat2` run with `beta=0.2/ms` so `V_th_v` is active); `iaf_tum_2000`
    post `V_m` machine precision after a **constant ~8-step** delivery/recorder
    offset (depression EPSPs decrease, facilitation increase-then-saturate; efficacy
    →0.0569); `mc_neuron` per-compartment `V_m.*` ~0.05–0.08 mV (RKF45), `g_*.*`
    float-noise floor, somatic-rheobase spike count exact.
  - carve-outs (distributional / analytic): `CampbellSiegert` Campbell μ <0.01 mV,
    σ² ~2–3 %, Siegert rate ~35 % (low count); `BrodyHopfield` seed-mean vector
    strength rel ~2 %, rate ~0.3 %, phase-hist max|Δ| ~0.006 (5 seeds);
    `gif_population` seed-mean rate ~1.4 %, binned-rate autocorr max|Δ| ~0.024
    (`CAT_D`); `gif_pop_psc_exp` meso-vs-micro window-mean rate ~11 % (mean 0.3 %),
    step jump ×3.36/×2.73, + meso-driver vs NEST uncoupled (`CAT_D`).
- **API discovered/changed (reusable seams):**
  - **(H) presynaptic-STP emission.** A presynaptic model declaring class attr
    `_emission_attr` (e.g. `iaf_tum_2000._emission_attr='spike_offset'`) wired with
    `connect(pre, post, receptor_type=pre.RECEPTOR_TYPES['TSODYKS'])` makes
    `Simulator._connect_pair` build a plain `EventProjection` that reads the pre's
    per-step emission holder — delivering the **released efficacy** `weight·(u·x)` as
    a pA delta, **not** the binary spike, and with **no** post-port routing
    (`iaf_tum_2000` has no `n_receptors`). A plain (receptor-0) connection from the
    same pre still delivers the binary spike — the per-connection distinction is the
    point. Reuse for any presynaptically-integrated plasticity.
  - **(I) mc device→compartment-receptor routing.** `connect(device, mc,
    receptor_type=k)`, 1-based `k∈1..6` = per-compartment exc/inh spike ports,
    `k∈7..9` = per-compartment current ports; per-compartment recordables resolve via
    `_mc_comp(attr, idx)` aliases (`V_m.{s,p,d}`, `g_ex/g_in.{s,p,d}`, last-axis
    compartment index). Pairs with the Urbanczik dendritic reader the deferred
    plasticity cluster needs.
  - **`gif_pop_psc_exp` is host-side** (NumPy `RandomState`, plain-int `N`,
    `update(x)->int n_spikes`, `add_delta_input(key, val)` sign-split into the
    tau_syn-filtered exc/inh channels, `update(x=)` → direct `y0` current; props
    `n_spikes`/`V_m`). Driven by a host Python loop — the **documented rule-#10
    carve-out** for untraceable models (it is not a lowerable `brainstate` module).
    Recurrent coupling = a delayed sign-split `add_delta_input` (synaptic); the step
    current = `update(x=)` (direct). The **microscopic** half ports onto the
    Simulator (one `for_loop`): `pconn` realized as `fixed_indegree(round(pconn·N))`,
    weight sign routing inhibition, `step_current_generator` for the jump.
- **Gotchas:**
  - **`gif_psc_exp.I_stim` collapsed `(N,)`→`()` under `for_loop`/`scan`.** With no
    current input registered, `sum_current_inputs(x, V)` returns the scalar `x`, so
    the write-back stored a scalar — changing the scan carry type between iterations
    (`carry input and carry output must have equal types`). Eager step-by-step tests
    (all the model had) never lowered it, so it was invisible until the Simulator ran
    the population in one `for_loop`. Fix = broadcast-on-write `I_stim =
    (i_stim + zeros(v_shape))·pA` (the mc idiom). **Lesson: every model the Simulator
    runs needs a `for_loop`-lowering regression test — eager stepping hides
    carry-shape bugs.** (Wrote the failing test first, then fixed.)
  - **`noise_generator` refresh interval (BrodyHopfield).** NEST's default
    `noise_generator` refresh is **1.0 ms**, not the sim `dt`; membrane-noise
    variance ∝ refresh interval, so leaving it at 0.1 ms inflated the noise and
    over-locked the population (R 0.44 vs NEST 0.27). Pass `noise_dt=1.0*ms`.
  - **No `pairwise_bernoulli` connectivity rule** — use `fixed_indegree(round(p·N))`
    (same mean in-degree); applied to both gif demos and matched NEST-side.
  - **Simulator compile cost dominates gif test wall-clock.** Each fresh
    `Simulator(...).simulate()` JIT-compiles the gif `for_loop` (minutes); the
    `gif_population` validation (4 distinct builds) took ~34 min. Amortize by sharing
    one expensive run across tests in `setUpClass` (`gif_pop_psc_exp` does **one**
    micro run); keep multi-seed `@requires_nest` work on the cheap host-side path.
    XLA's persistent cache makes a re-run of the same graph fast.
  - **meso-vs-micro autocorrelation dip is method-specific.** The cleaner mesoscopic
    rate model anti-correlates (adaptation dip ~26 ms); the noisier microscopic
    network's autocorrelation stays positive. Assert the *shared* feature (slow
    positive correlation at 60 ms), not the dip.
- **For next clusters:** §3.5 single-neuron demos are **complete (16 ports)**. Reuse
  seam (H) for any presynaptically-integrated plasticity, seam (I) + the
  per-compartment aliases for multi-compartment models (and the Urbanczik reader),
  and the host-side-loop + distributional-carve-out pattern for any untraceable /
  population-density model. The `for_loop`-lowering regression-test discipline
  applies to **every** Simulator-run model.

### 11-single-neuron-models-1 — 2026-06-13

- **Shipped:** the **§3.5 single-neuron model-demo cluster** — 7 ports under
  `examples/nest/` each with a live-NEST trace-parity test in `_nest/_validation/`:
  `hh_psc_alpha`, `hh_phaseplane`, `aeif_cond_beta_multisynapse`,
  `gif_cond_exp_multisynapse`, and the three GLIF demos
  (`glif_cond_neuron`, `glif_psc_neuron`, `glif_psc_double_alpha_neuron`). Cross-cutting
  reusable seams: **(F)** a multi-receptor `connect(receptor_type=k)` deposit seam in
  `_network/_event_proj.py` + a Simulator bridge in `_network/_simulator.py`; **(G)** a new
  `parrot_neuron` model (`_nest/parrot_neuron.py`) with a spike-multiplicity relay honoured by
  the substrate; and a batch of callable **recordable aliases** (HH gating, GLIF threshold
  components, per-port `g_k`, `ASCurrents_sum`, summed PSC `I_syn`, injected `I`). Docs:
  `examples/nest/README.md` §3.5 section + parity table, `examples-gap.md` §3.5 marked
  implemented + `gif_pop` deferral + aeif `n>1` limitation. Branch
  `worktree-nest-goal+11-single-neuron-models-1`; PR #57.
- **Parity (one-step recorder align: drop bp[0], `align_steps=1`):**
  - `hh_psc_alpha` — subthreshold `V_m`+`Act_m`/`Inact_h`/`Act_n` `CAT_A` (~1e-3 mV); F–I counts match.
  - `hh_phaseplane` — NEST-free: `n`-nullcline within one grid step of analytic `n_inf(V)`.
  - `aeif_cond_beta_multisynapse` — `V_m` ~1e-6 mV; `g_1..g_4` machine precision.
  - `gif_cond_exp_multisynapse` — subthreshold `V_m` machine precision.
  - `glif_cond_neuron` (5 levels) — `g_1`/`g_2` full-trace machine precision; `V_m`/`threshold`
    ~1e-13 mV; spike counts exact.
  - `glif_psc_neuron` (5 levels) — `I_syn`/`I` full-trace ~2e-15 pA + counts exact; `V_m`
    ~0.03 mV (`CAT_B_ALIGNED`); 150 kHz Poisson window matches NEST in aggregate (59/88/27/26/26
    vs 59/89/27/27/26) once the parrot relays multiplicity.
  - `glif_psc_double_alpha_neuron` (3 cfgs) — sub-threshold `V_m`/`I_syn` full-trace ~1e-13 mV /
    ~1e-15 pA.
- **API discovered/changed (reusable by all multi-receptor / derived-recordable models):**
  - **`connect(..., receptor_type=k)` is dual-mode, auto-detected by `inspect`-ing the post's
    `update` signature.** Models exposing **`w_by_rec`** (`iaf`/`aeif`/`gif_cond_exp_multisynapse`)
    take the **blob** path: one `add_delta_input(key, (N, n_receptors))` deposit, assembled by the
    Simulator bridge, which scales the gathered mantissa by the post's **`receptor_input_unit`**
    class attr (nS for conductance, pA for current). Models without it (the 3 GLIF models) take the
    **label-keyed** path: one `add_delta_input(key, col_k, label='receptor_k')` per port; the model
    self-pulls `sum_delta_inputs(label='receptor_k')`. 1-based `k` → internal port `k-1`;
    `'uniform'` keeps the random draw; out-of-range int and non-`'uniform'` string both raise.
  - **Callable `_RECORDABLE_ALIAS` entries.** `_read_recordable(pop, name)` invokes a callable
    alias as `entry(pop)`: `_g_port(k)` (list / single-State-last-axis / `g_syn` fallback, else
    clear `KeyError`), `_asc_sum`, `_psc_sum` (prefers `get_I_syn()`, else `Σ y2`), plus tuple
    aliases (`Act_m`→`m`, `threshold*`, `'I'`→`I_stim`). Add per-model recordables here, not in the
    neuron.
  - **`_relays_multiplicity` substrate flag.** The phase-2 capture binarises `Neuron` outputs at
    `>=0.5` **unless** the model sets `_relays_multiplicity=True` (then the raw per-step count is
    kept). `parrot_neuron` uses it to relay `get_mantissa(inp)` verbatim. Because the relay reads
    the *summed* delta input as the count, `EventProjection.__init__` **enforces the unit gate**:
    a connection into a `_relays_multiplicity` post raises `ValueError` unless its weight is the
    unitless `1.0` (NEST ignores weights into a parrot; a non-unit weight would silently scale the
    relayed multiplicity).
- **Gotchas:**
  - **NEST `poisson_generator` emits Poisson *counts* (multiplicity >1/step):** `rate·dt`
    (150000 Hz · 0.05 ms = 7.5). Two masked bugs collapsed this: the parrot's `get_spike` returned
    a **binary** `(arrived != 0)`, *and* the substrate binarises **all** Neuron outputs at `>=0.5`
    — so even a count-returning parrot was re-collapsed. Fix both (relay raw count + honour
    `_relays_multiplicity`). **Lesson:** exercise the *general* input (count >1 per step), not just
    1-spike-per-step, and trace the full substrate capture path — a unit test that only sends single
    spikes hides multiplicity collapse. **The same gap recurred:** the pre-merge code review caught
    that *every* multiplicity test used `weight=1.0`, hiding that a non-unit weight into the parrot
    silently scaled the relayed count (it reads the weighted sum, not a raw multiplicity field). Fix
    = the connect-time unit-gate guard above. Reinforced lesson: **test the adversarial value, not
    the convenient one** — for any "weight is ignored / must be X" contract, assert the off-nominal
    weight is actually rejected.
  - **`glif_psc` `V_m` is one step off, not machine-precise (`glif_cond` is exact).** brainpy
    computes `V` from the **pre**-propagation PSC (`y2_old`) while the recorded `I_syn` is the
    **post**-propagation `y2` — so `V_m` and recorded `I_syn` sit one step apart vs NEST (which
    reports both from the same `y2`). Not a bug; accept via `CAT_B_ALIGNED` (5e-2 mV). General
    pattern: **linear filters of the fixed input** (`g_k`, `I_syn`) match full-trace regardless of
    the neuron's own spike jitter → compare those over the whole trace; compare `V_m` only
    **sub-threshold** and spiking behaviour by **exact count**.
  - **GLIF threshold recordables live in the `E_L` frame** — compare as `bp['threshold'] + E_L`.
  - **`aeif_cond_beta_multisynapse` has an `n>1`×multi-receptor broadcasting bug** (receptor axis
    collides with the population axis); single-neuron multi-receptor traces are exact, so all demos
    use `n=1`. Fix belongs with the model (examples-gap §5).
  - **`hh_phaseplane` is an analysis demo, NEST-free.** Clamping `m`,`h` makes the reduced `(V,n)`
    system non-excitable (relaxes to rest, no AP). NEST's own script has two latent nullcline
    indexing bugs (`V_matrix[:][i]`, `index != len(n_vec)`) the port fixes — so it is *not* a
    line-by-line port and there is nothing to trace-compare.
  - **Touched-line coverage needs edge tests the `n=1` demos skip.** Parity demos only hit the
    full-population fast path + happy routing; the `test(nest): cover receptor-seam edge paths`
    commit pins the partial-population `_scatter_receptor` lift, `parrot.get_spike(None)`, the
    invalid-string / one-to-one receptor branches, and the `g_k` `KeyError` (parrot 100%,
    `_event_proj` 93%, both `_simulator` hunks 100%).
- **For next clusters:** `gif_pop_psc_exp` / `gif_population` are **deferred to cluster 12/14** —
  population/mean-field GIF needs a population-density update, out of scope for the single-neuron
  parity harness. Remaining §3.5 singles: `iaf_tum_2000_short_term_{depression,facilitation}` (LIF
  + integrated STP), `mc_neuron` (multi-compartment, `iaf_cond_alpha_mc` experimental),
  `BrodyHopfield`, `CampbellSiegert`. Reuse the `connect(receptor_type=k)` seam + `receptor_input_unit`
  + callable recordable aliases for any future multi-receptor or derived-recordable model, and
  `parrot_neuron` + `_relays_multiplicity` for any count-faithful relay (e.g. feeding a shared
  Poisson train to many targets).

### 13-plasticity-demos — 2026-06-13

- **Shipped:** the **four implementable §3.3 plasticity demos** on the `Simulator` API,
  each with a live-NEST parity test, + Urbanczik as a documented skipped placeholder, +
  README §3.3 / examples-gap §3.3 docs. Files: `examples/nest/{clopath_synapse_spike_pairing,
  clopath_synapse_small_network,evaluate_tsodyks2_synapse,evaluate_quantal_stp_synapse,
  urbanczik_synapse_example}.py` and matching `_nest/_validation/*_test.py`. Reused the
  cluster-07 `_clopath_drive` (generalized `_our_clopath_neuron` with an `n` param, default
  1 so all existing callers are unchanged) and the cluster-01 STP drives verbatim. One
  `_network` seam fix: `connect(seed=)` now threads into the plastic projection's runtime
  release `rng` (see below). Branch `worktree-nest-goal+13-plasticity-demos`.
- **Parity:**
  - `evaluate_tsodyks2_synapse` (deterministic): post-`V_m` PSC train, **max|Δ| ≈ 9.4e-16 mV**
    (`CAT_B`, 2-step align) both regimes — machine precision once delivery-aligned.
  - `evaluate_quantal_stp_synapse` (stochastic): seed-mean `V_m` `CAT_D`, **dep 1.8 % / fac
    2.9 %** at 8 seeds (ours vs NEST 2.08 v 2.11 / 2.57 v 2.49 mV).
  - `clopath_synapse_spike_pairing`: stored weight per train, in the frozen clopath band —
    **LTP ≤ 3.3 %, LTD |Δ| ≤ 0.0022 mV**.
  - `clopath_synapse_small_network`: per-edge final recurrent weight, **LTP ≤ 2.0 %, LTD
    |Δ| ≤ 0.0007 mV**; forward edges potentiate (mean 0.527), backward depress (mean 0.493),
    matching NEST's sign on every edge.
- **API discovered/changed (reusable):**
  - **`connect(seed=)` → runtime release `rng` seam.** `_event_plastic.py` now stores
    `self._rng_seed = seed` in `__init__` and `init_state` keys
    `self.rng = State(jax.random.key(seed or 0))` from it — so the seed **survives
    `simulate`'s `init_all_states`** (which re-runs `init_state`). Before this, a stochastic
    rule's seed set on the proj was wiped by `simulate`. Reproduced with a regression test
    *first* (`_simulator_plastic_test.py`: seed reproducible + distinct, and threads to
    `key(7)`), per working-agreement #4. Stochastic plastic rules are now reproducible
    through the Simulator.
  - **Weight-matrix recording for networks.** `record_weight(proj)` + `res.weight_trace(proj)`
    → `(T, E)` in **CSR sorted-by-pre** order; for `all_to_all` no-autapses that is exactly
    `[(i,j) for i in range(N) for j in range(N) if i!=j]`. Read the actual `(pre,post)` from
    `proj._pre_idx`/`_post_idx` rather than assuming. Clopath dispatches to
    `VoltageCoupledPlasticProj` (post-state reader); STP rules to `EventPlasticProj`.
- **Gotchas:**
  - **The `RELAY_D` holder-lag convention extends to every device→plastic-synapse port.** In
    NEST a device can't drive a plastic synapse (a `parrot` relays); the `Simulator`
    `spike_generator` injects with a **one-step (0.1 ms) holder lag**. So the NEST parrot
    relay delay must be **`RELAY_D = 0.1`**, NOT the 1.0 ms default — otherwise the whole train
    delivery-shifts ~8–9 steps and parity fails at any sane tolerance (tsodyks2 first showed
    this: `_D1=1.0` → best align shift 8; `_D1=0.1` → shift ≈0).
  - **NEST `quantal_stp_synapse` has a *second* `set_status` footgun: `u`, sibling of `a`.**
    `set_status` only re-derives `u_` if `'u'` is in the dict; the constructor set `u_(U_)`
    with the **old 0.5 default `U_`**, so `U=0.15` with no explicit `u` silently starts release
    at u=0.5 — biasing facilitation ~4 %. Depression (U=0.5) matched only by *coincidence* with
    the stuck default. Fix: pin **both `a=n` AND `u=U`** on the NEST side. This was a latent bug
    in the *existing* cluster-01 `quantal_stp_parity_test.py` too (passed only by small-sample
    luck); fixed there as well.
  - **Quantal variance reduction: `n_sites=100`, `n·w` fixed.** The stochastic seed-mean's CV
    drops ~7 %→4.3 % going 30→100 sites (envelope `x·u·(n·w)` unchanged), so 8 seeds reliably
    clears `CAT_D`. The kernel's seed-mean depends on the *absolute* rng-stream position, so
    Simulator-path and low-level seed-means differ by pure sampling jitter — not a bug.
  - **Standalone examples need `PYTHONPATH=. python examples/nest/<f>.py`.** Running a script
    puts the *script's* dir on `sys.path` (→ resolves the **installed** `brainpy_state`), not
    the cwd. `brainpy.state` ALSO resolves to site-packages, so during development examples
    import `brainpy_state` and are run with `PYTHONPATH=.` to exercise the worktree. Tests get
    this free (pytest rootdir on path).
- **For next clusters:** **Urbanczik is the one §3.3 blocker** — `urbanczik_synapse` is still on
  the legacy `NESTSynapse` base and the Simulator-API plastic post-state reader
  (`VoltageCoupledPlasticProj`) exposes only the **somatic `V`**; the rule needs a **named
  dendritic-compartment reader** plus a validated multi-compartment point-process post
  (`pp_cond_exp_mc_urbanczik`). When those land it ports like the other four. The
  `connect(seed=)`→`rng` seam unblocks any future **stochastic** plastic rule
  (`stdp_synapse` Hebbian-noise variants, etc.). §3.3 is otherwise complete.

### fix-brunel-es-import — 2026-06-13

- **Shipped:** repaired `brunel_alpha_evolution_strategies_test.py` (the lone red on `main`
  across many merges). Its ES helpers (`optimize`, `simulate`, `cut_warmup_time`,
  `compute_rate`, `compute_cv`, `sort_spikes`) were imported `from brainpy` — names that do
  not exist in the installed `brainpy` package; they live in
  `examples/nest/brunel_alpha_evolution_strategies.py`. Repointed both imports to
  `from examples.nest.brunel_alpha_evolution_strategies import ...` (the house pattern its
  siblings `balancedneuron_test` / `correlospinmatrix_detector_two_neuron_test` already use)
  and **restored** the `TestEvolutionStrategiesOptimizer` optimizer-math test that PR #54
  (`70080d0`) had *deleted* to silence CI. Branch `fix/brunel-es-test-import`; test-only (+20/−1).
- **Parity:** the NEST-gated `TestBrunelAlphaESNetworkParity` had **never run to completion**
  (it ImportError'd before the comparison); with the import fixed it now runs a live 1000 ms
  Brunel-alpha NEST run vs brainpy.state and passes the **5 %** rate band. Optimizer test
  converges (`|μ−opt| < 0.1`, seed 0).
- **API discovered/changed:** none (test wiring only).
- **Gotchas:**
  - **A NEST-gated test hides import bugs in CI.** CI has no NEST, so `@skipUnless(_HAS_NEST)`
    tests are skipped — a broken `from brainpy import …` inside one stays green in CI yet fails
    in any NEST-equipped env (local). Only the *always-run* sibling exposed it; deleting that
    sibling (PR #54's band-aid) turned CI green while leaving the latent bug **and** dropping the
    optimizer-math coverage CI actually exercised.
  - **`examples` is importable from tests via pytest prepend-mode.** Repo root has no
    `__init__.py`, so the basedir for `brainpy_state/...` tests is the repo root → it lands on
    `sys.path` → `import examples.nest.<m>` resolves (`examples` is a PEP-420 namespace pkg;
    `examples/nest/__init__.py` exists). Verified under both `python -m pytest` and the bare
    `pytest` entrypoint CI uses.
  - **Band-aid vs fix:** when a port's test fails on a bad import, repoint it to the real source —
    do not delete the test.
- **For next clusters:** NEST-example ports import helpers `from examples.nest.<module>` (the
  established pattern), never `from brainpy`. Keep a no-NEST companion test alongside any
  NEST-gated one so CI still exercises the importable surface. `main` was red for many merges on
  this single test — watch `gh run list --branch main` after merging.

### 10-stdp-docs — 2026-06-13

- **Shipped:** the **STDP parity reference page** — the discrete-synapse divergences
  discovered/frozen in 04/05/07/08, consolidated into user-facing docs. A new
  `docs/nest-guide/` Sphinx section (`index.rst` + `stdp-divergences.rst`) wired into
  `docs/index.rst` under a new hidden **"NEST Porting Guide"** caption (master lists
  `nest-guide/index`; that landing page's own toctree pulls in `stdp-divergences` — the
  nested pattern `brainpy-guide/` uses, so the child is not double-included). The page
  documents **(A)** the `tau_minus` trace-storage move + a family parameter-location
  table (`tau_minus`/`tau_minus_triplet`; clopath `A_LTP/A_LTD/theta_*` moved vs
  `tau_u_bar_*` staying on the neuron + `delay_u_bars`; dopamine `n`/`tau_n` on the VT)
  + the three numerical bands (clopath 5 %, dopamine 0.2 %, NN phantom-pre-at-0), and
  **(B)** the symm / restr / pre_centered / facetshw pairing conventions with the NEST
  source-line citations **lifted from the 05 docstrings**. A one-line `**Parity note.**`
  paragraph (`:doc:` + `:ref:` links) was appended to the **Notes** section of all **10**
  `stdp_*`/`clopath_`/`stdp_dopamine_` specs. NEST-free guard test
  `brainpy_state/_nest/stdp_docs_crosslink_test.py` (3 tests / 10 subtests). Docs-only
  carve-out: doctest, **no live-NEST run**. Branch `nest-goal/10-stdp-docs`.
- **Parity:** nothing re-measured — this cluster **cites** 04/05/07/08 (single source of
  truth: every number pulled from the Lessons + existing docstrings). The page's runnable
  `brainpy.state` doctests pass (`tau_minus=20 ms` on the synapse; `('nearest','nearest')`;
  `pre_trace_tau is None`; clopath `theta_plus=-45.3 mV`); the documented bands are quoted
  verbatim from the source Lessons.
- **API discovered/changed (reusable by all future model docs):**
  - **Docstring-link introspection test.** A NEST-free test under `brainpy_state/`
    (collected by CI's `pytest brainpy_state/`) that (a) asserts every targeted spec's
    `__doc__` carries the stable ``:doc:`/nest-guide/stdp-divergences``` marker, (b) parses
    the page's `.. _label:` defs and asserts each spec's cited `:ref:` resolves (no dangling),
    and (c) runs the page doctests via `doctest.testfile`. Reuse to keep any docstring→docs
    cross-link from rotting.
  - **doctest-in-`.rst` is the workaround for the `__module__='brainpy.state'` filter** (the
    01/04/06/07/08 gotcha): spec-class `>>>` examples are skipped by `--doctest-modules`, so
    the *runnable* examples live in the `.rst`, executed by `doctest.testfile` /
    `pytest --doctest-glob='*.rst'`. **There is no repo docs-test command**, so the guard test
    runs them in-CI itself.
  - **Public-vs-inner import split (CLAUDE.md rule 9) holds for docs.** Page examples use
    `brainpy.state`; the guard test uses `brainpy_state`. `brainpy/state/__init__.py` does
    `from brainpy_state import *`, so under PYTHONPATH=worktree (and CI's editable install)
    `brainpy.state.X is brainpy_state.X` — the doctests reflect worktree code.
- **Gotchas:**
  - **`conf.py exclude_patterns` hides `nest-status/internal/**`** — the gap-doc is **not**
    built, so a user-facing page cannot live there; `docs/nest-guide/` (built) is the home.
  - **Docs deploy uses `sphinx-build` without `-W`, only on release (not PR CI)** — so
    "builds clean" is a local check. The build emits **156 pre-existing `[docutils]` "Field
    list ends without a blank line" warnings** spanning the whole API (untouched BrainPy-style
    models included); my new files + **Notes-only** docstring inserts add **zero** (no
    link-integrity warnings; cross-links render in the built HTML, e.g. the `stdp_synapse`
    autosummary page → `#stdp-tau-minus`). Do **not** widen scope to fix the codebase-wide
    field-list quirk here.
  - **FP-noisy attrs stay out of doctests** — clopath `A_LTP`/`A_LTD` print as `7.9999…e-05`;
    the table quotes them, the doctests use only clean outputs.
- **For next clusters:** this **closes the discrete-synapse stream's docs + validation**
  (04/05/07/08 implemented and now documented). The documented STDP divergence map is
  complete: trace-storage (`tau_minus`), clopath param-location + `delay_u_bars` 5 %, dopamine
  `n`/`tau_n`-on-VT + 0.2 %, NN phantom-pre-at-0. The next synapse work is the **bucket-3
  continuous re-grill** (`gap_junction` / `rate_connection_*` / `diffusion_connection` /
  `sic_connection` + the rate neurons, `ContinuousCoupledProj`). Reuse the introspection-test
  + doctest-in-`.rst` pattern for its docs.

### 09-weight-recorder-audit — 2026-06-13

- **Shipped:** the **send-view `weight_recorder` seam** — two pure `_network` functions
  (`brainpy_state/_network/_weight_recorder_view.py`): `send_steps_from_pre(pre_spikes,
  pre_of_edge=None, *, lag=0)` derives the per-edge send mask from a pre spike train
  (single-pre, multi-pre CSR, relay `lag`), and `weight_recorder_events(weight_trace,
  send_steps)` masks the per-step weight trajectory to those steps (1-D, 2-D shared,
  per-edge list). No new device, **no Simulator/substrate change** — weight recording
  reuses the analog State-tap (CONTEXT Part 2.7), masked at the send steps. Plus the
  family-wide **live audit** (`_validation/weight_recorder_audit_test.py`): emitted event
  **count + timing + value** vs NEST for all **13 plastic rules** (11 STDP-family via
  `_stdp_drive`, `stdp_dopamine_synapse` via `_stdp_dopamine_drive`, `clopath_synapse`
  via two additive `_clopath_drive` captures `nest_pairing_weights_full` /
  `our_pairing_weight_trace`) + 5 edge-case methods. NEST-free: `_weight_recorder_view_test.py`
  (17 + 2 doctests, **100 % branch**). Closed `synapses-plasticity-gap.md` §2/§6 + P0
  weight-recorder, `devices-gap.md` weight_recorder row + §5 + P1. Part B (STP) was
  already shipped by cluster-01 — confirmed live (12 passed), not re-implemented.
  Branch `nest-goal/09-weight-recorder-audit`.
- **Parity (live NEST):** **8 audit methods + 11 subtests** pass. Every rule's recorder
  series reproduced: **count** == #pre sends == NEST event count; **timing** ==
  `steps(wr_times)` == pre-emission steps (`e.get_stamp()`, **no** delay offset);
  **value** within each rule's existing band — STDP family `CAT_B` (near-exact, ~1e-6),
  `ht_synapse` delivered `w·P` `CAT_B`, `clopath` 5 %, `stdp_dopamine` ~0.2 %. The goal's
  flagged "most likely bug" (before/after-update ordering) was **de-risked by C++ reading
  and RED-proven**: NEST logs the **post-send** weight, so masking our **post-update**
  `weight_trace` at the send steps is the apples-to-apples value (the seam unit test
  `test_samples_post_update_value_at_send_step` fails if you read `trace[step-1]`).
- **API discovered/changed (reuse verbatim):**
  - **Send-view is the `weight_recorder` analogue — a mask, not a hook.** `send_steps_from_pre`
    → `weight_recorder_events` compose; both are pure array indexing (`vmap`/`grad`-safe on
    the trace-generation path, ragged per-edge extraction host-side). Add a plastic rule to
    the audit by appending one `Row` (model, fresh-rule factory, per_conn/common,
    post_tau_minus, pre/post/T, band) — the `_audit` helper does count/timing/value.
  - **The `lag` parameter is the relay-offset knob.** Direct-feed drives (`_stdp_drive`,
    `_stdp_dopamine_drive` feed the projection in-step) → `lag=0`. A drive whose pre relays
    `sg→parrot` one step *without* pre-offsetting the generator (clopath's `RELAY_D`) stamps
    the send at `steps(s_pre)+1` → `lag=1`. `send_steps_from_pre(pre_arr, lag=1)` reproduces
    `steps(wr_times)` exactly — the live cross-check of the lag alignment.
- **Gotchas (NEST fidelity):**
  - **Per-rule logged value differs.** STDP family + clopath + dopamine log the **stored**
    weight (`delivered=False`); `ht_synapse` logs the **delivered** amplitude `w·P`
    (`delivered=True`, the depression pool's `w_eff`). Dopamine logs the weight integrated
    to `t_spike` *before* this pre's impulse mutates `c_` — our post-update trace at the send
    step still matches (the impulse touches only the future `c`).
  - **A change strictly after the last send is invisible to the recorder** (NEST logs only at
    sends, so it misses a post-last-send LTP too) — it lives in `weight_trace[-1]`, not the
    event series. Asserted live (`test_change_after_last_send_is_invisible`).
  - **nn_* phantom-pre-at-0 dodge.** A **causal** audit protocol (post after pre, so no post
    precedes the first pre send) sidesteps the symm/restr phantom-pre-at-0 facilitation the
    substrate doesn't model — no large `P0` needed when the pairing is causal.
  - **Coverage × JAX/NEST coredumps** (C tracer, sysmon, and `--timid` all abort once
    `import brainpy_state`/`nest` runs under coverage). The pure-numpy seam has **no** JAX
    dependency → measure it by loading the file via `importlib` (bypassing the package
    `__init__`) under `coverage.Coverage(include=[file], branch=True)` → 100 %. The live
    drive additions are straight-line code fully executed by the GREEN audit.
  - **NumPy-2 scalar repr in doctests:** `list(np_int_array)` prints `[np.int64(2)]`; use
    `.tolist()` for clean `[2, 5]` output.
- **For next clusters:** any new plastic rule gets recorder parity for free — append a `Row`
  to the audit and (if its drive relays the pre) set the matching `lag`. The send-view seam
  is the canonical way to compare event-emitting devices against a per-step State-tap; the
  same mask-at-event-steps shape fits `spike_recorder` / `multimeter` gating audits.

### 08-dopamine-vt — 2026-06-13

- **Shipped:** the **broadcast-modulator seam** — primitive #2's `VoltageCoupledPlasticProj`
  gains a `signal_reads` reader (a **1→E broadcast scalar** from a bound node, the superset
  of `post_states`' N→E gather); a JAX-native **`volume_transmitter`** maintaining `n(t)` as
  a broadcast `HiddenState` (`_nest/volume_transmitter.py`, ring-buffer port retired); and the
  rebuilt **`stdp_dopamine_synapse`** as a frozen spec + pure `update(state, ctx)` kernel
  (per-edge eligibility `c` + broadcast `n`; `_nest/stdp_dopamine_synapse.py`, eager port
  retired). `Simulator.connect(dopa, vt)` binds a dopa source; `connect(pre, post, vt=…)`
  wires the `n` signal source and routes to `VoltageCoupledPlasticProj` (`_network/_simulator.py`,
  `_event_plastic.py`). NEST-free: `volume_transmitter_rule_test.py` (16), `stdp_dopamine_synapse_rule_test.py`
  (27), `_voltage_coupled_plastic_test.py` (+signal-source guard), `_simulator_dopamine_test.py`
  (9); live-NEST `_validation/volume_transmitter_parity_test.py` (6) + `stdp_dopamine_synapse_parity_test.py`
  (9) + `_stdp_dopamine_drive.py`. Coverage: VT / dopamine-rule / drive **100 %**, substrate **98.5 %**.
  Branch `nest-goal/08-dopamine-vt`.
- **Parity (live NEST 3.9.0):** **VT `n(t)`** is the pure `update_dopamine_` recursion on both
  sides → **near-exact** (vs the closed form **0.0**, vs NEST **4.7e-16**; validated **upstream
  first**, the dopamine analogue of cluster-07's neuron-voltage precondition). **Weight trajectory**
  under dopamine modulation tracks NEST's `weight_recorder` **send-for-send**: realised
  **max|Δw| < 8e-3 pA** over LTP / LTD / clamp / window sweeps spanning 50→200 pA (**~0.2 % of Δw**,
  ~25× tighter than Clopath's 5 % — the clean scalar `n` carries **no** analog-history divergence),
  with **direction and ordering exact**. **STDP eligibility window** (single pair + dopa read-out):
  pre-before-post potentiates, post-before-pre depresses, `|Δw|` grows as the pairing tightens, the
  `A_minus>A_plus` asymmetry reproduces — sign, ordering, and magnitude all match.
- **API discovered/changed (next neuromodulator clusters reuse verbatim):**
  - **Broadcast-signal reader.** A rule declares `signal_reads = (names…)`; the substrate reads each
    from a bound node into **`ctx.signals = {name: scalar}`** — unit-stripped scalars **broadcast**
    against the `(E,)` per-edge arrays (1→E, a superset of `post_states`' N→E gather). `EventPlasticProj`
    keeps a no-op `_gather_signals()->None` (primitive #1 sees `ctx.signals is None`);
    `VoltageCoupledPlasticProj` overrides it and **raises** if a declared signal has no source. Sources
    are `signal_sources={name: (node, attr)}` (the Simulator wires them from `connect(…, vt=…)`); a spec
    may declare `post_state_reads`, `signal_reads`, or **both** with no substrate change.
  - **`volume_transmitter` is a broadcast node, not a spike holder.** `Simulator.create` registers it in
    `_vt_nodes` (it emits no spikes → no holder); `update()` **phase 0** advances every VT before the
    projections. `bind_dopa(reader, local_idx)` registers a dopa source; `n` lives **on the VT** (moved
    off the synapse — valid because `n` depends only on `tau_n` + the shared dopa train, a common
    property), so **`tau_n` must match** transmitter↔spec.
- **Gotchas (NEST fidelity):**
  - **`n`'s increment is `* dt`-free; the decay carries the step** — the mirror image of cluster-07's
    LTP trap. NEST `update_dopamine_`: `n += multiplicity/tau_n` per dopa spike, then `n *= exp(-dt/tau_n)`.
    The jump on a dopa-arrival step is exactly `1/tau_n` with **no decay that step**; `_advance(n, count,
    dt, tau_n) = n*exp(-dt/tau_n) + count/tau_n` reproduces NEST **bit-for-bit**. Verify against the C++,
    not the nominal `1/ms` unit.
  - **`n`-relay timing + recorder sampling.** A dopa `spike_generator` spike at `s` relayed
    `sg→parrot→vt` (two `dt` steps) makes `n` jump at `s + 0.2`; our `connect(dopa, vt)` holder lag is one
    step, so the drive places our generator one `dt` earlier — both jump on the **same** wall-clock step.
    NEST's `weight_recorder` logs **only at pre `send`**, so a weight read-out needs the pre to **keep
    firing**; a delayed dopa pulse *after* the last send is invisible to the recorder — read the final
    `GetConnections` weight (our last step) instead.
  - **Online ↔ deferred, but far tighter than Clopath.** NEST integrates the weight lazily (at
    `send`/`trigger_update_weight`; `e.set_weight(weight_)` is logged **before** the impulse's
    facilitate/depress, which mutate only `c_`); our kernel integrates **every step** with broadcast `n`
    (one-step lag). With `n` a clean scalar (no ring-buffered analog history), the residual is just the
    one-step-`n`-lag + per-step-vs-deferred integral → **~0.2 %**, vs Clopath's structural 5 %. Still a
    documented band + exact direction/ordering, not 1e-7.
  - **Param location.** NEST: `A_plus/A_minus/tau_plus/tau_c/tau_n/b/Wmin/Wmax` are **common** props
    (`CopyModel`) with the VT bound there; `tau_minus` is a **post-neuron** param (`get_K_value`); `c`/`n`
    are per-synapse state. The self-contained spec moves `tau_minus` onto the synapse (the `stdp_synapse`
    convention) and `n` onto the VT; parity sets identical values across neuron, synapse, and transmitter.
  - **Doctest `__module__` filter (cluster-07 redux) + coverage.** Spec/node classes set
    `__module__='brainpy.state'`, so `DocTestFinder` skips them — verify by restoring `cls.__module__`
    before `DocTestFinder().find(cls)` (VT 5 + dopamine 6 examples pass). For coverage, the rtk stdout
    proxy swallows pytest-cov's term-table; `coverage run --source=brainpy_state -m pytest …` + the
    coverage API (`Coverage().load(); cov.analysis2(file)`) reports per-file cleanly.
- **For next clusters:** the `KernelContext` now carries **two reader shapes** — `post_states` (N→E
  per-edge gather) and `signals` (1→E broadcast) — both flowing through the same `update(state, ctx)`
  kernel. A voltage-gated **and** neuromodulated synapse needs **no new substrate**: declare both
  `post_state_reads` and `signal_reads` and pass `signal_sources`. The `volume_transmitter` pattern (one
  node, `n`-like broadcast State, `bind_*` sources, phase-0 advance) generalises to any shared modulator
  (ACh, NE) or a global gain/reward channel.

### 06-static-stochastic — 2026-06-13

- **Shipped:** **`bernoulli_synapse`** (static delivery + stochastic transmission) and
  **`cont_delay_synapse`** (static delivery + sub-dt delay) rebuilt as frozen NEST specs +
  pure `update(state, ctx) -> (new_state, w_eff)` kernels on the cluster-01 `EventPlasticProj`
  (primitive #1), retiring their eager `ImperativeSynapseBase` impls from the active path
  (`_nest/{bernoulli_synapse,cont_delay_synapse}.py`). One **additive, default-off** substrate
  seam landed (`fractional_delay`, below). Each ships a NEST-free `*_rule_test.py` (**both
  100 %** line coverage) + a live-NEST `_validation/*_parity_test.py`; the old eager
  `*_test.py` are deleted. Both are **delivery-semantics** models (no learned weight evolves) —
  this **closes the non-plastic `EventPlasticProj` delivery family**. Branch
  `nest-goal/06-static-stochastic`.
- **Parity (live NEST 3.9.0):**
  - **bernoulli:** `p_transmit=1` ≡ `static_synapse` **exact** (CAT_B V_m trace);
    `p_transmit=0` flat at rest; `0<p<1` seed-mean `V_m` within **CAT_D 5 %** over 6 seeds.
    Driven through **E=20 multapses** (one pre→post pair, repeated CSR edges) → ~500 Bernoulli
    arrivals/seed → per-seed variance ~2 % ≪ bound.
  - **cont_delay:** integer/grid delay ≡ static **exact** (CAT_B, verified first). Sub-dt
    `d∈{1.33,1.35,1.37} ms` (frac {0.3,0.5,0.7}) vs NEST **precise `iaf_psc_exp_ps`**:
    integrated `∫V_m` **and** EPSP **peak amplitude** match **~1e-5..4e-5 rel** (charge +
    first-moment exact), peak **timing** within **±1 step** (CAT_E); the sole residual is a
    ~0.25 mV **onset transient** (frac 0.3 worst, on a ~4.3 mV EPSP) — documented, intrinsic.
- **API discovered/changed (reusable by 08+):**
  - **Design A — per-edge stochastic independence is free.** A rule that sets
    `stochastic=True` gets one 0-d step key `ctx.key`; drawing
    `jax.random.uniform(ctx.key, (E,)) < p` yields **edge-axis-independent** gates (JAX threefry
    counter PRNG: output position `j` ← counter `j`), so two edges from one pre and multapses
    gate independently. **No per-edge `split`/`fold_in` seam is needed** (the goal's fallback is
    unnecessary, and cheaper) — the shape-`(E,)` draw already supplies it. Same idiom
    `quantal_stp` uses. Multapses (repeated CSR `(i,j)`) **sum** in the event-matmul.
  - **Design B — `fractional_delay` output-carry seam** (additive, default-off via
    `getattr(rule,'fractional_delay',False)`). When a rule sets `fractional_delay=True` the
    substrate, **in `init_state`** (where `dt` is known), decomposes the homogeneous delay into
    an integer floor `k_lo=⌊d/dt⌋` → `InputDelay(k_lo·dt)` (clean binary floor frame) + a
    `(n_post,)` `delay_carry` HiddenState, and `update` applies a 1-step FIR `[1−frac, frac]` on
    the **post amplitude after the matmul**. Charge + arrival-centroid exact; `frac==0` collapses
    byte-identical to a grid delay. Reusable by any precise/off-grid demo.
- **Gotchas (NEST fidelity / substrate):**
  - **`BinaryArray` binarizes the pre vector** (`y[j]=Σ_{i:s[i]≠0} W[i,j]`, magnitude ignored),
    so a sub-dt delay **cannot** ride on a fractional *pre* weight — it would double-deliver. The
    split **must** live on the delivered (post) amplitude → output-carry, not the `InputDelay`
    linear-interp (which BinaryArray defeats). Phase-2 RED captured exactly this double-delivery.
  - **`cont_delay` `delay >= dt` is best-effort.** `__init__` reads
    `brainstate.environ.get_dt()`, but that **raises `KeyError` when no `dt` is set**, so the
    floor check is *deferred* (the Simulator/tests set `dt` before building synapses). Don't
    assume construction always validates the resolution floor.
  - **The sub-dt residual is first-order, not `O((dt/τ)²)`.** The exp-PSC onset is a C¹ kink, so
    the split-vs-precise error is `O(frac·dt/τ)` **at the onset** (a ~6 %-of-peak ripple on the
    rising edge), not the second-order I first wrote. Charge-exactness makes `∫V_m` and peak
    amplitude match to ~1e-4 regardless — **assert those + peak-step**, document the transient,
    never force a tight per-sample trace bound on a grid integrator. Corrected the docstring
    after measuring vs live NEST.
  - **`brainpy_state._nest.<model>` resolves to the CLASS, not the submodule** (the
    `_nest/__init__` `from .x import x` rebind shadows it). To monkeypatch a spec's module-level
    dependency in a test, patch the **imported object directly** (`brainstate.environ.get_dt`),
    not `mod.brainstate...`. (Same root cause as 07's doctest-finder note.)
  - **Doctests:** specs set `__module__='brainpy.state'` → `DocTestFinder` skips them when
    scanning the file; verify via `DocTestFinder()._from_module=lambda *_:True` + `.find(cls)`
    (bernoulli 6 / cont_delay 8 examples, 0 failed).
  - **Coverage:** the local pytest proxy **strips `--cov` flags** (emits only `Pytest: N
    passed`) → use `python -m coverage run … && coverage json -o … && read the JSON`
    (the directory-dotted `--cov=pkg.mod` form still coredumps per 04/05).
- **For next clusters:** the **non-plastic delivery family is complete** (`static`,
  `static_synapse_hom_w`, `bernoulli`, `cont_delay`). The per-edge `ctx.key` shape-`(E,)` idiom
  and the `fractional_delay` output-carry seam are both proven and reusable. The stochastic
  `ctx.key` seam is now exercised by **both** `quantal_stp` (binomial) and `bernoulli` (uniform
  gate) — 08's neuromodulated/stochastic rules can lean on it. Remaining: plastic clusters 08
  (dopamine) + 09/10 + bucket-3.

### 07-voltage-coupled-clopath — 2026-06-13

- **Shipped:** **primitive #2 `VoltageCoupledPlasticProj`** (the post-state reader) +
  the rebuilt voltage-based **`clopath_synapse`** spec+rule proven on it. The substrate
  gains a post-neuron analog-State reader (`_network/_event_plastic.py`); `clopath_synapse`
  becomes a frozen spec + pure `update(state, ctx) -> (new_state, w_eff)` kernel
  (`_nest/clopath_synapse.py`, legacy history-buffer port moved to
  `_legacy_clopath_synapse.py` — *removed 2026-06-16; see `cleanup-remove-legacy-synapses`*);
  `Simulator.connect` dispatches to
  `VoltageCoupledPlasticProj` when the spec declares `post_state_reads`. Ships a NEST-free
  `clopath_synapse_rule_test.py` (**100 %** line coverage, 19 tests) and a live-NEST
  `_validation/clopath_synapse_parity_test.py` (+ `_clopath_drive.py`). Branch
  `nest-goal/07-voltage-coupled-clopath`.
- **Parity (live NEST 3.9.0):** **neuron analog states** (`V`/`u_bar_plus`/`u_bar_minus`)
  under a subthreshold dc drive match **sample-for-sample** (CAT_A, realised
  `< 5e-7` mV — the Simulator+multimeter aligns to NEST's recorder with **no** index
  shift). **Spike-pairing** (canonical `clopath_synapse_spike_pairing.py`, 10 trains
  10–50 Hz): **stored weight within 3.31 %** of NEST (pure-LTD train **0.002 %**; LTP grows
  with frequency), with **direction and frequency-ordering exact**. Voltage-clamp LTD
  (sustained sub-θ₊ depolarization) near-exact.
- **API discovered/changed (08-dopamine + Urbanczik reuse this verbatim):**
  - **Post-state reader.** A rule declares `post_state_reads = (names…)`; the substrate
    gathers those post-neuron State columns per edge each step into
    **`ctx.post_states = {name: (E,)}`** — a named dict of **unit-stripped mantissas** in
    sorted-by-pre (CSR) edge order, via `{n: u.get_mantissa(getattr(self.post, n).value)
    [self._post_gather]}` with `_post_gather = post_local_idx[_post_idx]`. `EventPlasticProj`
    keeps a no-op `_gather_post_states()->None` seam (primitive #1 sees `ctx.post_states is
    None`); `VoltageCoupledPlasticProj` overrides it and **raises** without a `post`
    population or a non-empty `post_state_reads`. Reads are **read-only gathers**.
  - **Filters live on the neuron already.** Both Clopath neurons expose
    `u_bar_plus`/`u_bar_minus`/`u_bar_bar` as `brainstate.HiddenState` — **no neuron change**
    to surface them as reads (record once, reuse).
  - **`x_bar = ctx.pre_trace / tau_x`** (declare `pre_trace_tau = tau_x`): NEST increments
    `x_bar` by `1/tau_x` per spike, the substrate by 1 — divide to match.
  - **mV weight for the delta neuron.** `aeif_psc_delta_clopath` is a *delta* model (input
    seam is a **mV voltage jump**), so a bare `clopath_synapse` weight defaults to **mV**
    (not pA); `connect`'s bare-weight override preserves the spec's unit. `hh_psc_alpha_clopath`
    still wants explicit pA.
- **Gotchas (NEST fidelity):**
  - **LTP carries a `* dt`; LTD does not.** NEST `ClopathArchivingNode::write_LTP_history`
    multiplies its per-step `dw` by `Time::get_resolution().get_ms()` (so the accumulated
    potentiation is a resolution-independent **integral**); `write_LTD_history` has no such
    factor. **Verify against the C++ source, not nominal unit docs** — `A_LTP` reads as
    `1/mV²` which *looks* dt-free; dropping the factor made LTP ~10–20× too large. Caught by
    measuring vs live NEST.
  - **`A_LTD`/`A_LTP`/`theta_plus`/`theta_minus` are NEST *neuron* params** (the archiving
    node precomputes Δw); the self-contained spec+rule moves them onto the **synapse spec**
    (mirrors `stdp_synapse`'s `tau_minus` move). The filter constants `tau_u_bar_*` stay on
    the neuron. Parity sets identical values on both.
  - **`delay_u_bars` aligned to one step.** NEST evaluates LTP/LTD against ring-buffered post
    voltages delayed by `delay_u_bars` (default **4.0 ms**); the online reader has no analog
    ring buffer and reads the post State with the substrate's intrinsic **one-step** lag
    (projections run before neurons), so parity sets `delay_u_bars = 0.1 ms`. The 4.0 ms
    default is **not reproducible** online — documented divergence.
  - **The residual LTP gap is structural, not a bug.** Our formula on NEST's *own* recorded
    voltages (with the one-step u_bar delay) reproduces NEST's Δw to ~4.6 %; the rest is the
    online-instantaneous-read vs NEST-deferred-history (decayed-`x_bar` per-LTP-entry sum) +
    a one-step **event-delivery** lag on the post driver. It grows with pairing frequency, so
    the parity test asserts a **documented 5 % band + exact direction/ordering**, not 1e-7.
  - **Spec doctests + `--doctest-modules`.** A spec class sets `__module__='brainpy.state'`,
    so `DocTestFinder` (filtering by `__module__`) **skips it** when scanning the file —
    `--doctest-modules` collects 0. The `>>>` examples are still valid (verify by running
    `DocTestFinder().find(cls)` directly); keep the 8-space code-block indent for consistency.
  - **Coverage:** use the **directory-path** form `--cov=brainpy_state/_nest` (per cluster-04/05);
    the module-dotted form coredumps.

### 05-stdp-nearest-neighbour — 2026-06-13

- **Shipped:** the **4 remaining nearest-neighbour / hardware STDP models** rebuilt as
  frozen parameter spec + pure `update(state, ctx) -> (new_state, w_eff)` rule kernels on
  the cluster-01 `EventPlasticProj` substrate, retiring their legacy imperative
  subclasses from the active path (`_nest/{stdp_nn_symm_synapse,stdp_nn_restr_synapse,
  stdp_nn_pre_centered_synapse,stdp_facetshw_synapse_hom}.py`). Each ships a NEST-free
  `*_rule_test.py` (closed-form/host kernel + jit/vmap/grad) and a live-NEST
  `_validation/*_parity_test.py`; the old eager `*_test.py` are deleted. One substrate
  extension landed (the `nearest` trace mode, below). All 4 spec files at **100 %**
  line+branch coverage. This closes the 04 entry's "a future cluster ports them" note.
  Branch `nest-goal/05-stdp-nearest-neighbour`.
- **Parity (live NEST 3.9.0):** all 4 match NEST to **machine precision** — single
  spike-pair Δt sweeps + divergent-train scenarios at **CAT_B** (atol 1e-6, realised abs
  err ~1e-13–1e-15) and 5 s trains at **CAT_A** (atol 1e-3). facetshw is **CAT_B exact**
  throughout: its weight is discrete (`k * weight_per_lut_entry`), so once the charge
  evaluations align both sides agree to 0 ULP.
- **Core design — RESOLVED (do not relitigate):** hybrid. A reusable substrate
  **`nearest` trace mode** supplies the nearest-neighbour K±; each scheme's divergence
  (eligibility gating / accumulation / reset / readout) lives as **per-edge kernel state**
  in `edge_state_init()`. Rejected: a bespoke per-edge trace per model (re-implements the
  substrate decay and loses the unified-gather idiom below).
- **API discovered/changed:**
  - **Substrate `nearest` trace mode** (`_network/_event_plastic.py`): `pre_trace_mode` /
    `post_trace_mode` ∈ `{'cumulative' (default), 'nearest'}`. `nearest` **stores**
    `where(spike,1,decayed)` (reset-to-1) but still **gathers `decayed+spike`**
    (accumulated), so the kernel's `k = ctx.trace - ctx.spike` recovers the
    **second-latest** partner on an exactly-coinciding step *for free* — the same
    +1-subtraction idiom cluster-04 used for simultaneity now also does the
    nearest-neighbour reset. Default stays cumulative; cluster-01 + the 7 core specs untouched.
  - **Per-edge eligibility flags** (restr): `pre_avail` / `post_avail`; each spike makes
    its own side available and **consumes** the opposite — reproduces NEST's
    `start != finish` one-pair-per-spike gate with no history scan.
  - **Per-edge accumulate-then-reset trace** (pre_centered): `Kplus` carried in
    `edge_state_init` (NOT a substrate per-neuron trace) because the **post** spike resets
    it, so two edges from one pre reset at different times. `pre_trace_tau = None`; decay
    in-kernel (`*exp(-dt/tau)` **before** the +1).
  - **Charge+readout kernel** (facetshw): per-edge `{a_causal, a_acausal, causal_pending,
    pre_seen, post_seen, next_readout}`; a readout (first pre past each
    `readout_cycle_duration`) = quantise → 2× eval_function → LUT → reset → advance-clock.
- **Gotchas (NEST fidelity):**
  - **Phantom-pre-at-0 (symm & restr only).** NEST's first `send()` (`t_lastspike_=0`)
    facilitates a post preceding the first pre against a *virtual* pre at t=0; the substrate
    seeds traces/flags at 0 → models the physically-correct "no pre" and does NOT reproduce
    it. pre_centered & facetshw are immune (their facilitation scales by a trace/flag that
    starts 0). Parity dodges it via a leading pre or P0 ≥ 500 so the `exp(-(q+d)/tau)` term
    sits below atol.
  - **facetshw deferred accumulation is OBSERVABLE** (unlike the pair models). NEST runs
    the readout BEFORE folding the triggering pair, so a readout never sees its own pair's
    charge. Capture the causal term at the first post (`causal_pending`) and fold it + the
    acausal term only at the next pre, **after** that pre's readout. Folding at the post
    step (as the pair models harmlessly do) shifts one pairing across the boundary → flips a
    threshold → visible weight divergence.
  - **facetshw weight footgun:** `wple = Wmax/15 ≈ 6.667`; the default `weight=1.0`
    quantises to index 0 and the first readout — which re-quantises **even on `(F,F)`** —
    zeroes it. Use on-grid weights (`5*wple ≈ 33.33`).
  - **facetshw NEST keys/scope:** common `tau_minus` is exposed as **`tau_minus_stdp`**;
    `a_thresh_th/tl` are PER-SYNAPSE (syn_spec); `no_synapses`/`readout_cycle_duration`
    auto-compute (don't set). A DENSE symmetric train drives BOTH charges over threshold →
    `(T,T)` identity (no learning) — clean one-pair-per-cycle trains at a lowered threshold
    separate the LUT branches (the threshold *value* doesn't affect NEST↔bp agreement).
    **Single-driver scope only** (E ≤ `synapses_per_driver`).
  - **Coverage:** the cluster-04 `--cov` segfault now also bites the directory/module forms
    — tracing `install_exp_euler_patch()`'s C-path at import coredumps. Workaround: import
    `brainpy_state` **untraced**, then `coverage.start()` + `importlib.reload` the spec
    modules (covers their module-level lines) before running the rule tests in-process;
    coverage is by file-line so the cached classes still trace method bodies.
- **For next clusters:** the `nearest` mode + eligibility-flag + per-edge-trace +
  charge/readout idioms now cover every nearest-neighbour and hardware-LUT shape by
  **writing specs only**. Remaining legacy-imperative STDP: `stdp_dopamine` (needs a global
  dopamine / eligibility-trace seam) and the non-STDP `bernoulli` / `cont_delay`. The one
  deferred sub-feature is facetshw **multi-driver** round-robin (E > `synapses_per_driver`,
  staggered per-synapse readout offsets).

### 04-stdp-core — 2026-06-12

- **Shipped:** the **7 STDP-core synapse models** rebuilt as frozen parameter
  spec + pure `update(state, ctx) -> (new_state, w_eff)` rule kernels on the
  cluster-01 `EventPlasticProj` substrate — no imperative code in the active path
  (`_nest/{stdp_synapse,stdp_synapse_hom,stdp_pl_synapse_hom,jonke_synapse,
  vogels_sprekeler_synapse,ht_synapse,stdp_triplet_synapse}.py`, shared
  `_nest/_plastic_base.py`). Each ships a NEST-free `*_rule_test.py` (kernel/host
  closed-form, **110** rule tests) + a live-NEST `_validation/*_parity_test.py`
  (**24** tests / **38** subtests). Two substrate/seam extensions landed
  (below). The legacy imperative STDP stays **only** for out-of-cluster models
  (`stdp_nn_*`, `stdp_dopamine`, `stdp_facetshw`, `bernoulli`, `cont_delay` →
  `_legacy_imperative`/`_legacy_stdp_synapse`); the 7 core names now route to the
  new specs. *(Update 2026-06-16: `_legacy_imperative` / `_legacy_stdp_synapse` removed — the
  listed models only referenced them in docstrings, never imported them; see
  `cleanup-remove-legacy-synapses`.)* Spec files + `_plastic_base` at **100 %** line coverage. Branch
  `nest-goal/04-stdp-core`.
- **Parity (live NEST 3.9.0):** all 7 match NEST to **machine precision** —
  single spike-pair Δt sweep (both signs) at **CAT_B** (atol 1e-6) and 5 s
  weight-trajectory at **CAT_A** (atol 1e-3), with realised abs errors ~1e-13–1e-15.
  - `stdp_synapse` (+`_hom` thin reuse): Δt sweep, coincident, Wmax clamp exact.
  - `stdp_pl_synapse_hom` (power-law), `jonke_synapse` (exp weight-dep + `beta`):
    trajectories step-for-step.
  - `vogels_sprekeler_synapse` (symmetric inhibitory, constant per-pre depression):
    pre-only + paired trains exact.
  - `ht_synapse` (vesicle-pool depression): delivered `w·P` max|Δ| **1.4e-14**,
    closed-form **0.0**.
  - `stdp_triplet_synapse` (Pfister-Gerstner, multi-trace): single pairs **0.0**,
    50 Hz + 5 s trains **2.7e-15**.
- **API discovered/changed** — what 05/06/07 reuse:
  - **Multi-trace seam (the substrate extension 01 anticipated).** `pre_trace_tau`/
    `post_trace_tau` now accept `None | Quantity | tuple[Quantity,...]`. A tuple
    allocates a per-neuron `(N, k)` trace State, decays each column by its own tau,
    adds the (delayed) spike to every column, and gathers `(E, k)` into
    `ctx.pre_traces` / `ctx.post_traces`; `ctx.pre_trace`/`post_trace` alias column 0
    (**back-compat — cluster-01's 6 specs untouched**). `stdp_triplet_synapse` is the
    first user (fast+slow per side). **Per-neuron storage (N·k) beats per-edge; this
    is the foundation for 05-Clopath's per-post voltage filters** (a per-edge
    `ctx.post_spike` cannot reconstruct them).
  - **Weight-recording API** (`_network/_simulator.py`): `connect(synapse=spec)`
    now **returns the proj handle**; `sim.record_weight(proj)` registers a tap;
    `res.weight_trace(proj)` → `(T, E)` pA trajectory in CSR (sorted-by-pre) edge
    order. Mirrors the analog-recorder State tap; independent of the parity harness.
  - **`_stdp_drive.py` reusable parity harness** (`_validation/`): the decoupled-iaf
    drive + dendritic-delay shift + common/per-conn routing for any pair/triplet/
    pool STDP model. `bp_weight_trace(..., delivered=True)` samples the delivered
    `w_eff` (for pool models whose stored weight is static); `nest_pair_run(...,
    post_params=…)` sets extra post-node constants (e.g. `tau_minus_triplet`).
- **Gotchas (NEST fidelity + JAX):**
  - **Online ↔ NEST-deferred equality is the whole correctness argument.** NEST
    `send()` defers potentiation to the next pre spike where the `weight_recorder`
    samples; the online substrate potentiates **eagerly on post steps** + depresses
    **on pre steps**. With no depression between pre spikes the cumulative op set/order
    is identical ⇒ weight **coincides at every send (pre-spike) time**. Parity is
    asserted there, not step-by-step.
  - **`tau_minus` (and `tau_minus_triplet`) is a POST-NEURON param in NEST**, not a
    synapse param — the `ArchivingNode` owns `K-`. Kept as a synapse-rule attr for
    standalone fidelity; **documented in every trace-model docstring** (ht is
    trace-free → explicitly exempt). The parity drive sets the post node's
    `tau_minus`(+`tau_minus_triplet`) to match.
  - **Decoupled archiving post — parrot relay pollution.** A `parrot_neuron` post
    **relays the STDP-delivered pre spikes into its own archive** → phantom posts →
    phantom facilitation (proven: 2-pre/no-post gave 5→13.47). Fix: a strong-driven
    `iaf_psc_delta` post (`V_th=1e4`, driver weight `1e6`, STDP EPSP subthreshold),
    record the actual pre/post fire times (spike_recorders) and replay them on brainpy.
  - **Dendritic-delay convention:** a NEST STDP synapse of delay `d` behaves as if
    each post spike's effect occurs at `q+d`. Reproduce by injecting each post spike
    `d` later on the brainpy timeline **and** disabling the substrate's axonal
    `InputDelay` (`rule.delay = None`; the kernel never reads it).
  - **Simultaneous-spike exclusion via the trace's own +1.** The substrate's
    decay-then-add feeds traces **including the current step's spike**, so the firing
    side's trace is always `>= 1`; the kernel subtracts the spike
    (`kplus = ctx.pre_trace - ctx.pre_spike`, and for triplet `r = pre_traces -
    pre_spike`, `o = post_traces - post_spike`) to recover the strictly-prior
    (`t-ε`) value. Rule-test inputs **must** keep the firing side `>= 1` or exclusion
    goes negative (caught three red tests).
  - **`ht_synapse` delivered-vs-stored observable.** Its stored weight is **static**
    — depression lives in the vesicle pool `P`; NEST's `weight_recorder` logs the
    **delivered** `w·P_send`. Sample the delivered amplitude (`delivered=True`), not
    `proj.weight.value`. Inits `t_lastspike = 0.0` (**not** tsodyks2's `-1.0` skip)
    so a *partial* initial `P` recovers from `t=0`, matching NEST (`P=1` makes the
    first-spike recovery a natural no-op).
  - **common vs per-connection params:** `stdp_synapse`, `jonke`, `ht`, `triplet`
    accept plasticity params **per-connection**; `*_hom` and `vogels` require them as
    **common** (CopyModel/SetDefaults). Using `common` is universally safe.
  - **pytest runs unittest methods ALPHABETICALLY** — `-x` stops at the
    alphabetically-first failing method (here `test_fixed_train...` masked passing
    single-pair tests). Run `-v` **without** `-x` to see the real picture.
  - **Coverage segfaults under the module-form `--cov`.** Measure by **directory
    path** (`--cov=brainpy_state/_nest`), never `--cov=brainpy_state._nest.<mod>` —
    the module form coredumps importing the package under instrumentation in this env.
- **For next clusters:** build new plasticity by **writing specs only** — the
  multi-trace seam + `_stdp_drive` harness + weight-recording API now cover
  single-trace, multi-trace, stochastic (`ctx.key`), and depression-pool shapes.
  **05-Clopath** layers a per-post *voltage* filter on the same per-neuron multi-trace
  storage (voltage is a neuron State, not reconstructable per-edge). The remaining
  STDP variants (`stdp_nn_*` nearest-neighbour, `stdp_dopamine`, `stdp_facetshw`)
  still sit on the legacy imperative base — a future cluster ports them onto the
  spec+rule pattern (each its own file; legacy bases stay for the non-STDP models).

### 03-recording-demos — 2026-06-12

- **Shipped:** the 5 NEST §3.4 recording/device demos on the `Simulator` API
  (`examples/nest/{multimeter_file,recording_demo,cross_check_mip_corrdet,
  correlospinmatrix_detector_two_neuron,precise_spiking}.py`), each with a
  live-NEST parity test (`_nest/_validation/<name>_test.py`). Drove **one reusable
  pattern (E — eager imperative devices)** and **one validation-helper extension**
  (`compare_distributional` `autocorr`/`cv` statistics). The 2 blocked demos
  (`plot_weight_matrices`, `synapsecollection`) ship as **skipped placeholders**
  that raise `NotImplementedError` with the gap reason (need `GetConnections`/
  `SynapseCollection`, network-api-gap.md §3.1/§3.8). **One `_nest` model fix was
  needed:** `mcculloch_pitts_neuron` now self-manages its PRNG (reproduced with a
  unit test first — see Gotchas). NEST-free unit suites: the detector self-check in
  `cross_check_mip_corrdet_test.py`, the structural tours in
  `recording_demo_test.py` / `precise_spiking_test.py` /
  `correlospinmatrix_..._test.py`, and the 2 placeholder marker tests. Branch
  `nest-goal/03-recording-demos`.
- **Parity (live NEST):**
  - `multimeter_file`: `V_m`/`I_syn_ex`/`I_syn_in` traces match to **machine
    precision** (`CAT_B_GEN` = `TraceTolerance` with `align_steps=2`).
  - `recording_demo`: the 1 MHz drive pins `iaf_psc_exp` to its refractory-
    saturated rate → **identical** across 4 seeds (`CAT_D`).
  - `cross_check_mip_corrdet`: seed-mean normalized cross-correlogram within
    `CAT_D.autocorr_max_diff` (5e-2); the built-in `correlation_detector`
    reproduces the hand-written `corr_spikes_sorted` reference **exactly for
    lag ≥ 0** (NEST's asymmetric half-bin pruning differs <2 % rel-L1 only at the
    extreme negative lags — a boundary effect).
  - `correlospinmatrix_..._two_neuron`: per-channel mean activities **|Δ| ≤ 0.013**
    (bp `[0.500, 0.506]` vs NEST `[0.497, 0.493]`); 2×2 covariance **max|Δ| ≈
    0.014** (`CAT_D`, 5 seeds).
  - `precise_spiking`: spike **count exact** (`10 = 10`) for both grid and precise
    across `dt ∈ {0.1, 0.5, 1.0}`; onset-aligned relative spike times within
    `CAT_E` (≤ 1 step — grid exact, precise ~1e-4 ms). Grid first spikes
    `7.7/8.0/8.0`; precise `7.77/8.17/8.67` (off-grid).
- **API discovered/changed** — what later imperative-detector clusters reuse:
  - **E. Eager imperative devices.** `mip_generator`, `correlation_detector`, and
    `correlospinmatrix_detector` are NumPy-RNG / Python-loop host devices that
    **cannot enter a JAX `for_loop`**. The pattern: obtain the spike data *first*
    — a device's own `device.simulate(n_steps)` `(n_steps, k)` multiplicity matrix,
    or a binary spin train State-tapped from a single `for_loop` — then drive the
    detector **post-hoc**, feeding only event-carrying steps and stamping each at
    `step + 1` (NEST's one-step delivery latency, which cancels in the lag/cross
    difference). **Nothing imperative runs inside the `for_loop`** (the standing
    constraint). `correlation_detector.update(spikes, receptor_ports, weights,
    multiplicities, stamp_steps)` + `.get('count_histogram'|'histogram'|
    'n_events')`; `correlospinmatrix_detector` consumes spike pairs incrementally
    and exposes the `cc[i,j,τ]` tensor (mean activity = `cc[i,i,center]·dt/T`,
    covariance = `cc[i,j]·dt/T − m_i·m_j`).
  - **`compare_distributional` gained `statistic="autocorr"`** (seed-averages 1-D
    functions, element-wise `max|Δ| ≤ autocorr_max_diff`) **and `"cv"`** (uses
    `mean_diff_pct`) on top of `"mean"` (`rate_rtol`). Reuse `autocorr` for any
    seed-mean correlogram / PSTH / kernel comparison.
  - **Analog recordable aliases `I_syn_ex` / `I_syn_in`** on the State tap (a
    `Simulator` seam): a `multimeter` records `iaf_psc_exp`'s two synaptic-current
    ports, and a signed delta event splits by sign exactly as NEST routes a signed
    weight to its ex/in port (so one `spike_generator` with `+w`/`−w` drives both).
- **Gotchas (NEST fidelity + JAX):**
  - **Binary/stochastic neurons must self-manage `rng_key` from `rng_seed`.**
    `environ.get('key')` returns `None` inside a brainstate `for_loop`, so the old
    `environ`-key path silently degenerated to **no stochasticity** (`mcculloch_
    pitts` fired deterministically → n2 never Poisson-gated). Fix: store
    `self.rng_key` (a `State` seeded from `rng_seed`) and split it in `update()`.
    Reproduced with a unit test first; **fixed in `_nest/mcculloch_pitts_neuron.py`,
    never worked around in the example.** This is the fix that makes n2 match NEST.
  - **`correlospinmatrix` up-transitions must be fed as TWO multiplicity-1 spikes
    at the same stamp**, not one multiplicity-2 event (down = one mult-1 spike).
    NEST delivers a real spike *pair*; a single mult-2 event corrupts the detector's
    `_last_change` self-confirmation (a channel's down gets confirmed by its own
    next up), dragging ch0's mean to 0.38 vs the correct 0.51. This is an
    **example-feeding** convention — the detector itself mirrors NEST bit-for-bit
    (verified by `correlation_detector_test`); diagnosed via the detector's own
    unit test, not patched in the device.
  - **Binary coupling `h = w·y1` ⇒ read the *pre-update* spin** in the shared
    `for_loop` step: `y1_prev = n1.y.value` **before** `n2.update(x=w*y1_prev)`,
    then `y1 = n1.update()`. In brainpy the coupling current `x` is *transient*
    (consumed that step) while delta inputs persist in `h`; n2's gain is
    `H(h + c − θ)` with `c = w·y1_prev`. Seed each neuron's `rng_seed` distinctly
    (offset the global `brainstate.random.seed`).
  - **`precise_spiking` carries a constant dc-delay onset offset.** NEST's
    `dc_generator` default *connection* delay is 1.0 ms; the eager precise model is
    driven from t=0. The constant shift (11 steps at dt=0.1) exceeds the `CAT_E`
    absolute bound, so compare **onset-aligned relative** spike times (subtract the
    first spike) — exact for grid, ~1e-4 ms for precise. Off-grid time is read from
    `neuron.last_spike_time` (also `last_spike_step`/`last_spike_offset`); the
    precise model runs **eagerly** (plain Python loop, `update(x=…)` under
    `environ.context(t=k*dt)`, spike via `bool(all(spk>0))`) because sub-step spike
    timing is a host-side property, not a `for_loop` array.
  - **Generator→neuron→multimeter analog traces carry a *two*-step delivery
    offset** vs NEST (generator spike-holder one step + recorder one step), unlike
    the one-step offset of an `I_e`/current-injected trace — use `align_steps=2`
    (`CAT_B_GEN`) for any `spike_generator`-driven analog recording.
  - **`multimeter_file` conductance gap.** The upstream records `g_ex`/`g_in` from
    a conductance `iaf_cond_alpha`; driving conductance synapses from a spike source
    needs a `w_ex`/`w_in` labelled-delta routing seam the explicit `Simulator`
    lacks. Kept the demo *structure* on current-based `iaf_psc_exp`
    (`V_m`/`I_syn_ex`/`I_syn_in`) — a documented follow-up.
- **For next clusters:** the **E (eager-device)** pattern is the vocabulary for
  every imperative detector/generator port (correlation, correlospinmatrix, spike-
  train statistics) — tap State or run the device's own `.simulate()` first, then
  post-process; never inside the `for_loop`. `compare_distributional` now covers
  `mean`/`autocorr`/`cv` — reuse `autocorr` for any correlogram/PSTH/kernel. Two
  demos stay **blocked** on connection introspection (`GetConnections`/
  `SynapseCollection`, network-api-gap.md §3.1/§3.8) — a `nest_compat` facade
  unblocks `plot_weight_matrices` + `synapsecollection` together. A conductance-
  recordable seam (spike→`g_ex`/`g_in` on `iaf_cond_alpha`) would let
  `multimeter_file` port to its exact upstream model.

### 02-single-neuron-demos — 2026-06-12

- **Shipped:** the 7 NEST §3.2 single-/few-neuron demos on the `Simulator` API
  (`examples/nest/{one_neuron,one_neuron_with_noise,twoneurons,testiaf,
  balancedneuron,if_curve,vinit_example}.py`), each with a live-NEST parity test
  (`_nest/_validation/<name>_test.py`). Drove **4 reusable `Simulator`
  extensions** (in `_network/_simulator.py` + new `_nest/voltmeter.py`):
  **A** analog State-tap recording, **B** current-injecting devices, **C**
  sweep/rebuild-per-trial, **D** per-generator weight vectors. NEST-free unit
  suites: `_simulator_analog_test.py` (6), `_simulator_current_test.py` (5),
  `_simulator_sweep_test.py` (3), `_simulator_weightvec_test.py` (3),
  `voltmeter_test.py` (4). **No `_nest` model fixes were needed** —
  `iaf_psc_alpha`/`aeif_cond_exp`/`iaf_cond_exp_sfa_rr` matched NEST out of the
  box; the work was entirely the Simulator seams. Branch
  `nest-goal/02-single-neuron-demos`.
- **Parity (live NEST):** deterministic single-neuron `V_m` traces match to
  **~1e-14 mV** (`CAT_B_ALIGNED`): one_neuron 2.8e-14, twoneurons 2.8e-14 /
  5.7e-14, testiaf 2.8e-14 (spikes 16=16 across dt∈{0.1,0.5,1.0}), vinit worst
  1.4e-14. Stochastic: one_neuron_with_noise 45.8 vs 46.0 spks/s (0.5 %, `CAT_D`
  4 seeds); if_curve det. point 9.0=9.0 (`CAT_C_RATE`), noisy points <5 %
  (`CAT_D`); balancedneuron bisected inhibitory rate 20.81 = 20.81 Hz.
- **API discovered/changed** — what clusters 03/11/12/14 reuse on the `Simulator`:
  - **A. Analog recording.** `voltmeter` = `multimeter` preset (`record_from=
    ('V_m',)`), exported top-level. Connect *reversed*: `connect(voltmeter, pop)`
    registers an analog State tap `{id(rec) -> (id(pop), idx, recordables)}`; the
    `for_loop` step reads `getattr(pop, attr).value[idx]` after `update()` and
    stacks `(T, N)`. Read with `res.trace(rec, 'V_m')` and `res.times`. Recordable
    alias `{'V_m': 'V'}`, else `getattr(pop, name)` (so `g_ex`/`w`/`I_syn_ex` work).
    NEST recorder one-step offset absorbed by `CAT_B_ALIGNED` (align_steps=1).
  - **B. Current devices.** `_CURRENT_GENERATORS = (noise_/dc_/step_/ac_generator)`
    classify as *current* sources; `connect(gen, pop)` realises the device at
    `n=n_post`, registers `(device, pop, idx, weight, key)` in `_current_injectors`,
    and `update()` does `pop.add_current_input(key, scatter(device.update()*weight))`
    **before** neuron drive. The neuron's `sum_current_inputs` consumes it into its
    `y0` one-step buffer = NEST current ring buffer (so a dc current shows the same
    one-step lag vs `I_e`, which is added directly). Re-add every step (input is
    popped on consume). Spike generators still take the EventProjection delta path.
  - **C. Sweep.** No new API: `simulate()` calls `init_all_states(self)` so a fresh
    `build()` + `simulate()` per trial resets everything (incl. device/RNG State).
    `res.rate(node)` / `res.n_events(node)` work on a single-neuron `spike_recorder`
    tap. Rebuild-per-trial mirrors NEST `ResetKernel`-in-loop.
  - **D. Per-generator weight vectors.** `create(poisson_generator, k, rate=[…])`
    returns a **k-segment** NodeView (one scalar-param `_GenSegment` per channel;
    vector params split, scalars broadcast — `_index_channel`). `connect(gen, pop,
    weight=[w0..w_{k-1}])` applies `weight[i]` to segment `i` (`_segment_weights`:
    a length-`n_seg` vector indexes per segment, anything else broadcasts). Signed
    weights sum in the neuron's delta input → `w0·train0 + w1·train1`. Works
    because the per-segment one-to-one EventProjection already accepts a scalar
    weight; **no `poisson_generator` model change** (it still forces scalar rate).
- **Gotchas (NEST fidelity + JAX):**
  - **Spiking V_m traces are brittle to per-sample compare** (reset discontinuity
    → ~15 mV at a 1-step misalignment). Compare the **sub-threshold charge window
    before the first spike** (+ spike count via `CAT_E`), not the sawtooth. NEST's
    default `voltmeter.interval=1.0 ms` ≠ sim dt — set `interval=dt` (0.1) in the
    NEST reference so sample counts line up.
  - **Current-neuron sign-split == NEST separate ports only when
    `tau_syn_ex==tau_syn_in`.** `iaf_psc_alpha.update` sums all delta inputs then
    splits by sign (`w_ex=max(w,0)`, `w_in=min(w,0)`); NEST routes `+w` to the ex
    port and `−w` to the in port separately. With equal synaptic taus the alpha
    kernels are identical so linear superposition makes them bit-equal (demo 2 holds
    exactly); **with unequal taus they differ** — revisit if a demo needs split taus.
  - **brainpy neuron defaults ≠ NEST defaults for some models.** `iaf_psc_alpha`
    matches NEST exactly, but `aeif_cond_exp` differs (V_peak 0 vs −40, V_reset −60
    vs −70.6, E_in −85 vs −70, t_ref 0 vs 5, b 80.5 vs 80.8) → **set every NEST
    param explicitly** in a port. Initial `V_m` via `V_initializer=Constant(v*mV)`.
  - **noise_generator fan-out is independent per target** (one device at `n=n_post`
    draws `randn(n)`/step); `dc/step` fan-out is identical. The per-connect derived
    seed keeps separate `connect`s independent. `noise_generator`'s seed kwarg is
    `seed` (not `rng_seed` like the spike generators).
  - **Worktree import shadowing:** bare `python examples/nest/foo.py` picks up the
    *installed* `brainpy_state` (site-packages), not the worktree — run demos with
    `PYTHONPATH=$(pwd)`; pytest already prepends the worktree root.
- **For next clusters:** the 4 seams are the device/recording vocabulary for all
  later single-cell and network ports — reuse `voltmeter`+`res.trace` for any V_m
  validation, the current-injector path for any `*_generator` current device, and
  the multi-channel-generator+weight-vector pattern for any signed multi-source
  Poisson drive. Keep parity grids/bisections modest (each `build()` recompiles
  the `for_loop`). Compare deterministic V_m on a pre-spike window; compare
  anything PRNG-driven as a seed-mean (`CAT_D`), never per-sample.

### 01-event-plastic-substrate — 2026-06-11

- **Shipped:** the JAX-native event-driven plastic projection substrate
  `brainpy_state/_network/_event_plastic.py` (`EventPlasticProj` + the
  `KernelContext` / `PlasticSynapse` rule contract), the shared spec helpers
  `_nest/_plastic_base.py`, and **6 NEST-faithful synapse specs rebuilt as
  pure parameter-spec + rule kernel**: `static_synapse`, `static_synapse_hom_w`,
  `tsodyks_synapse`, `tsodyks_synapse_hom`, `tsodyks2_synapse`,
  `quantal_stp_synapse`. The old imperative base was relocated to
  `_nest/_legacy_imperative.py` (`ImperativeSynapseBase`) and the 7 not-yet-ported
  models redirected onto it. *(Update 2026-06-16: `_legacy_imperative` removed; see
  `cleanup-remove-legacy-synapses`.)* `sim.connect(pre, post, synapse=<spec>, …)` now
  builds an `EventPlasticProj` (Scope C). Tests: substrate unit suite
  (`_event_plastic_test.py`), 6 NEST-free rule tests, `_simulator_plastic_test.py`,
  and 3 live-NEST parity tests (`_validation/{static_synapse,stp,quantal_stp}_parity_test.py`).
  **100 % line coverage** on the substrate, all 6 specs, and `_plastic_base`.
  Branch `nest-goal/01-event-plastic-substrate`.
- **Parity (live NEST, post `V_m` through `iaf_psc_exp`, `tau_syn_ex=tau_psc=3 ms`):**
  - static delivery: max|Δ| ≈ **1e-15 mV** (`CAT_B_ALIGNED`).
  - tsodyks / tsodyks_hom / tsodyks2 depression: **~1e-15 mV** (machine precision);
    facilitation: **~1.8e-4 mV** (`CAT_B_ALIGNED`, 5e-2 mV bound).
  - quantal_stp (6-seed distributional, `CAT_D`, 5 % bound): depression **2.1 %**,
    facilitation **4.3 %** rel. PRNG streams differ (NEST per-site Bernoulli vs one
    `jax.random.binomial`) → aggregate-mean only, never per-sample.
- **API discovered/changed** — the substrate + contract STDP clusters (04/05/06) reuse:
  - `EventPlasticProj(*, pre_spike, n_pre_pop, pre_local_idx, post, post_local_idx,
    rule, conn=None, pre_idx=None, post_idx=None, n_post_pop=None, post_spike=None,
    pre_is_post=False, allow_autapses=True, allow_multapses=True, seed=None,
    delta_key=None)`. Edges are sorted by pre into CSR order once; **all per-edge
    State is stored in that order**. Delivery = `BinaryArray(pre_seg) @ CSR((w_eff,
    indices, indptr))` into `post.add_delta_input`, gated by the actual pre spikes.
  - `KernelContext(pre_spike, post_spike, pre_trace, post_trace, t_now, dt, key)` —
    per-edge `(E,)` arrays (sorted-by-pre) + 0-d `t_now`/`dt`/`key`; all unit-free
    mantissas, the substrate re-attaches pA on delivery.
  - **Rule contract:** a spec exposes class attrs `is_homogeneous_weight`,
    `stochastic`, `pre_trace_tau`, `post_trace_tau`, `weight_unit`, instance
    `weight`/`delay`, plus `edge_state_init() -> dict` and
    `update(state, ctx) -> (new_state, w_eff)`. **State is rule-declared, the
    substrate allocates it:** `is_homogeneous_weight` → 0-d vs `(E,)` `ParamState`;
    `edge_state_init()` keys → per-edge `HiddenState`s; `pre/post_trace_tau` →
    per-neuron trace `HiddenState`s (decay-then-add, gathered per edge);
    `stochastic` → a PRNG `State` split each step into `ctx.key`. The kernel must
    gate its own writeback by `ctx.pre_spike` (helper `frozen(fired,new,old)`); the
    substrate only gates *delivery*.
  - To add a model: write the spec (attrs + `edge_state_init` + `update`) — **no
    substrate change**. `sim.connect(synapse=…)` resolves via shallow-copy override
    of weight/delay (`Simulator._resolve_synapse`).
- **Gotchas (NEST fidelity + JAX):**
  - **jit bakes `environ` `t`/`i`** at first trace (`jit(mod.update)` returns a
    constant time). Thread t/i as args: drive inside `transform.for_loop(step,
    times, indices, spikes)` with the per-step spike a **scanned argument** (the
    efficient, compile-once pattern used in every parity drive).
  - **quantal_stp `a` footgun:** NEST's `set_status` leaves `a_` at its constructor
    default (`a_(n_)`, `n_=1`) unless `'a'` is passed — setting `n=30` alone gives
    `a_=1`. With strong depression the first spike dominates, so a missing `a` blew
    parity to 41 %. Set `a=n` explicitly (the rebuilt kernel defaults `a=n`).
  - **`tsodyks_synapse_hom` common props:** `weight` (via `CommonPropertiesHomW`) +
    `tau_psc/tau_rec/tau_fac/U` are *homogeneous* → set with `nest.SetDefaults`,
    **not** per-connection; only `delay/x/u` go in `syn_spec`.
  - **Plastic synapses can't be device-driven** in NEST → relay through a
    `parrot_neuron` (`spike_generator → parrot → synapse`); the relay shifts time
    by one delay but preserves the ISIs that drive the STP state. The brainpy side
    injects at the parrot fire steps so `h = t_now - t_lastspike` matches.
  - **Propagator forms are FP-distinct, keep each verbatim:** `tsodyks_synapse`
    uses `Pzz=expm1(-h/τrec)` with `x += Pxy·y − Pzz·z`; `tsodyks_synapse_hom`
    uses plain `Pzz=exp(-h/τrec)` with `x += Pxy·y + Pxz·z`. Algebraically equal,
    not bit-equal. `tsodyks2` uses the **old** `u` in the `x` update but the **new**
    `x·u` for the weight, with a `t_lastspike ≥ 0` first-spike guard.
  - **`t_lastspike` init differs:** `0.0` for tsodyks/tsodyks_hom (interval-invariant
    first spike when `x=1,y=0`), `-1.0` for tsodyks2/quantal (first spike skips
    decay/recovery).
  - **`__module__='brainpy.state'` hides doctests** from `--doctest-modules`
    (codebase convention); examples still must be correct — verify with a
    `DocTestFinder` overriding `_from_module`. `init_all_states` returns the module,
    whose repr leaks into doctests → assign to `_`.
- **For next clusters:** the contract is frozen and reusable — STDP (04/05/06) get
  the `pre_trace`/`post_trace` seams (decay-then-add per-neuron, gathered per edge)
  already wired **and tested** (synthetic trace rules in `_event_plastic_test.py`),
  and the `ctx.key` stochastic seam is proven by quantal. Build new plasticity by
  writing specs only. If a rule needs cross-edge or per-target reductions beyond the
  current per-edge `update`, that is the first substrate extension point.

### 00-validation-harness — 2026-06-11

- **Shipped:** the shared parity harness in `brainpy_state/_nest/_validation/`:
  `tolerance_conventions.py` (A–E constants), `nest_compare.py` (compare engine),
  `conftest.py` (`requires_nest` marker), `README.md`, + `tolerance_conventions_test.py`
  / `nest_compare_test.py` (31 NEST-free unit tests, **100 %** line coverage on both
  harness modules). Refactored the 5 existing parity tests onto it (thin glue: NEST
  run logic + Simulator drive kept verbatim, only skip + compare/assert moved).
  Branch `nest-goal/00-validation-harness`. Full dir: **43 passed** with live NEST.
- **Parity:** all 5 pass live (brunel alpha/delta/multisynapse exc-rate within 5 % of
  NEST; siegert mean-field exc+inh within 5 %; device Poisson within 5 % of configured
  1000 Hz). Assertions are byte-for-byte the originals (5 %), now via documented tols.
- **API discovered/changed** — the surface the next clusters import
  (`from brainpy_state._nest._validation.nest_compare import …` /
  `… .tolerance_conventions import CAT_*`):
  - `compare_trace(reference, candidate, *, tol, metric="trace") -> ComparisonResult`
    — deterministic; pass = division-free allclose `|a−b| ≤ atol + rtol·|ref|`; optional
    `tol.align_steps` integer-shift search (recorder offset). Scalars are 0-d traces.
  - `compare_distributional(reference_samples, candidate_samples, *, tol, metric="rate",
    statistic="mean") -> ComparisonResult` — multi-seed **mean**, never per-sample;
    zero-variance/zero-mean safe.
  - `nest_compare(nest_fn, brainpy_fn, *, mode, tol, metric, seeds=None)` — runs both
    callables (`mode='trace'` calls each once; `'distributional'` calls each per seed).
  - `ComparisonResult(passed, error, bound, metric, detail)` + `.assert_()` (raises
    `AssertionError(detail)`). `requires_nest` decorator (skip + pytest marker; reads
    `HAS_NEST` at call time, so patchable/testable). `reference`=NEST (or analytic
    ground-truth), `candidate`=brainpy.
  - Tolerances: `CAT_A` adaptive trace (1e-3 mV), `CAT_B` analytic (1e-6 mV) +
    `CAT_B_ALIGNED` (5e-2 mV, `align_steps=1`), `CAT_C` conductance/coupled (1e-3 mV) +
    `CAT_C_RATE` mean-field rate (5 % rel), `CAT_D` distributional (5 % rate, ≥4 seeds),
    `CAT_E` spike-time (|ΔN|≤2, |Δstep|≤1). `T_DEFAULT`/`DT_DEFAULT`/`N_SEEDS_DEFAULT`.
    A/B/C numbers are from `numerical-validation-gap.md` §6; **D/E were referenced
    (index.md P0 #4) but never defined — synthesized here** (by model/compare kind).
- **Gotchas:**
  - **Coverage + NEST:** a dotted `--cov=…nest_compare` pre-imports NEST under
    coverage's C tracer and **core-dumps**. Measure by directory path
    (`--cov=brainpy_state/_nest/_validation`) instead.
  - `requires_nest` reads `HAS_NEST` at decoration time — patch the module global
    *before* defining the class to test the skip path.
  - Rates here are plain floats (not brainunit Quantities); V_m traces may be either.
    The comparator strips units to the tolerance's unit (mV) or takes the mantissa —
    pick the category whose unit matches the metric (mV for V_m, plain/Hz for rates).
  - Division-free allclose reproduces both pure-abs (V_m) and pure-rel (rate) and is
    zero-reference safe — do not reintroduce `|a−b|/ref`.
  - `brainunit` has no `u.uV`; use `u.volt`/`u.mV`. NEST multimeter carries a one-step
    recorder offset → `CAT_B_ALIGNED`/`align_steps` absorbs it (for 02's V_m traces).
- **For next clusters:**
  - **01** (PSC-amplitude train, static delivery): `compare_trace` + `CAT_A`/`CAT_B`;
    `bernoulli` (06) is a distributional carve-out → `compare_distributional`/`CAT_D`.
  - **02** (single-neuron demos, V_m, F-I): `compare_trace` + `CAT_B`/`CAT_B_ALIGNED`
    for V_m (recorder alignment), `CAT_E` for spike-count/F-I. `iaf_psc_alpha_parity_test.py`
    already shows the V_m max-abs + ±1-step pattern (kept un-refactored as a reference).
  - **04/08** (weight trajectory over 5 s): `compare_trace` + `CAT_A`.
  - Thin-glue convention: keep each test's own `_nest_*` fn (with its `ResetKernel`).
    A shared ResetKernel/`setUp` helper was deferred (YAGNI) — add when a cluster needs it.
  - `compare_distributional`'s autocorr/CV path is a stub (constants recorded, mean-only
    comparator) — flesh out for binary-neuron / generator distributional tests (16).
