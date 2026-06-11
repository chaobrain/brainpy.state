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
