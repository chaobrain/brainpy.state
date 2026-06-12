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
