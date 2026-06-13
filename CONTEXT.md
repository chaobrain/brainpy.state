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
  - **Rate/continuous models** — `rate_neuron_ipn/opn`, `lin_rate`, `gauss_rate`,
    `sigmoid_rate*`, `siegert_neuron`, `aeif_cond_alpha_astro`,
    `diffusion_connection` carry the same `_queue` pattern. These are **bucket 3**
    and co-designed with `ContinuousCoupledProj`.
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

### 07-voltage-coupled-clopath — 2026-06-13

- **Shipped:** **primitive #2 `VoltageCoupledPlasticProj`** (the post-state reader) +
  the rebuilt voltage-based **`clopath_synapse`** spec+rule proven on it. The substrate
  gains a post-neuron analog-State reader (`_network/_event_plastic.py`); `clopath_synapse`
  becomes a frozen spec + pure `update(state, ctx) -> (new_state, w_eff)` kernel
  (`_nest/clopath_synapse.py`, legacy history-buffer port moved to
  `_legacy_clopath_synapse.py`); `Simulator.connect` dispatches to
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
  new specs. Spec files + `_plastic_base` at **100 %** line coverage. Branch
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
  models redirected onto it. `sim.connect(pre, post, synapse=<spec>, …)` now
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
  - Rates here are plain floats (not saiunit Quantities); V_m traces may be either.
    The comparator strips units to the tolerance's unit (mV) or takes the mantissa —
    pick the category whose unit matches the metric (mV for V_m, plain/Hz for rates).
  - Division-free allclose reproduces both pure-abs (V_m) and pure-rel (rate) and is
    zero-reference safe — do not reintroduce `|a−b|/ref`.
  - `saiunit` has no `u.uV`; use `u.volt`/`u.mV`. NEST multimeter carries a one-step
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
