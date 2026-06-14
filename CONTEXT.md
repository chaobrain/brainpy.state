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
    required — `saiunit.lax.sort`) → **one** `n→1` plastic projection drives `n` dendritic
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
