# /goal — 17b · §3.8 astrocyte demos (`astrocyte_single`, `astrocyte_interaction`) + the pool-rule boundary

> Self-contained prompt. Open at repo root. Stream: example. **Bucket-3 demo cluster** (the astro half) — the **last bucket-3 cluster**. Depends on: `15d` (the neuron↔astrocyte SIC loop on the substrate; `sic_connection`; the `I_SIC` recordable alias) — **MERGED (#65)**. Independent of `17` (rate).

## What 15d established + the scope boundary (read first)
15d shipped + validated the **exact** neuron↔astrocyte SIC loop: `Simulator.connect(astro, neuron, synapse=sic_connection(weight=w))` → `_connect_sic`; the astrocyte emits SIC as a continuous graded **current** into the neuron's `'I_SIC'` channel (`as_current` `EventProjection`); the neuron→astro arm is the ordinary delta path (`Δ_IP3·w` IP3); the multimeter records `I_SIC`→`I_sic`. **In scope (substrate-ready — the loop is validated to live NEST):** `astrocyte_single` + `astrocyte_interaction`. **Out of scope (need NEST's `TripartiteConnect` / `third_factor_bernoulli_with_pool` astrocyte-pool connectivity rule — no new connectivity rule this cluster):** `astrocyte_small_network` + `astrocyte_brunel_{bernoulli,fixed_indegree}` → **documented skipped placeholders**.

## Session start
1. Create an isolated worktree branch **`nest-goal/17b-astro-demos`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`15d-astro` Lessons** — the SIC-loop API (above); the parity numbers (SIC-response micro IP3 `2.4e-5`/Ca `1.9e-4`/I_SIC `2.3e-4`; driven loop; astro-network distributional **SIC-raises-firing +5 Hz**, exact-after-`align_steps`); `sic_connection.delay_steps` (NEST default `delay=1.0 ms` = **10 steps** at dt=0.1; `delay_steps=10` rides it, `=1` rides the intrinsic one-step lag); the **labelled `I_SIC` read precedes the unlabelled `I_stim` read** (else double-count); the `TripartiteConnect` pool-rule boundary.
   - the **`16-generator-demos` Lessons** — the demo-cluster house style (runnable `examples/nest/` script + one test module with a NEST-free structural class + a live-NEST distributional class `skipUnless`; `interval=dt`; `from examples.nest.<module>`; no-NEST companion); and the **skipped-placeholder pattern** for a blocked demo (`_BLOCKED_REASON` + a marker test asserting the skip — as cluster 03 shipped for `plot_weight_matrices`/`synapsecollection`).
   - the **`14-network-demos` Lessons** — distributional parity (seed-mean/median band + the qualitative law).
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.8 astro rows (+ status markers), `synapses-plasticity-gap.md` (sic/astro notes), `network-api-gap.md` (the `TripartiteConnect`/astrocyte-pool entry — the placeholder's recorded blocker).
   - The merged substrate: `brainpy_state/_nest/{astrocyte_lr_1994,aeif_cond_alpha_astro,sic_connection}.py`; `_network/_simulator.py` `_connect_sic` + the `I_SIC` recordable alias; `_network/_event_proj.py` `as_current` mode.
   - The house style: `examples/nest/{brunel_siegert,recording_demo}.py`, the §3.4 blocked-demo placeholders (for the skip pattern), + `../../README.md`.
   - NEST upstream: `pynest/examples/astrocytes/{astrocyte_single,astrocyte_interaction,astrocyte_small_network,astrocyte_brunel_bernoulli}.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the in-scope/placeholder split is the sign-off.

## Goal
Port the two substrate-ready §3.8 astrocyte demos to `examples/nest/<name>.py` on the `Simulator` API, each with a live-NEST parity test, and ship `astrocyte_small_network`/`astrocyte_brunel_*` as documented skipped placeholders recording the `TripartiteConnect` blocker.

## Scope
- **`astrocyte_single`** → one `astrocyte_lr_1994` driven by a `poisson_generator` spike train; record IP3/Ca/SIC; assert the IP3/Ca/SIC waveform vs NEST (tight where deterministic — mirrors 15d's SIC-response micro-parity).
- **`astrocyte_interaction`** → the bidirectional `aeif_cond_alpha_astro` ↔ `astrocyte_lr_1994` loop (neuron spikes → IP3; Ca → SIC → neuron); assert the SIC-modulated V/Ca/I_SIC vs NEST (`align_steps`; tight where deterministic).
- **`astrocyte_small_network` + `astrocyte_brunel_{bernoulli,fixed_indegree}`** → **skipped placeholders** (`_BLOCKED_REASON` = "needs `nest.TripartiteConnect` / `third_factor_bernoulli_with_pool` astrocyte-pool connectivity — no equivalent rule yet") + a marker test asserting the skip; record the blocker in `network-api-gap.md`.

## Constraints
- `Simulator` API only; `examples/nest/` house style; `brainunit` units; `from examples.nest.<module>`; no-NEST companion.
- Parity vs live NEST 3.9.0 (dt=0.1 ms): exact-after-`align_steps` for the deterministic loop; distributional (seed-mean band) for any Poisson-driven population. Set the recorder `interval=dt`. The **labelled `I_SIC` read precedes the unlabelled `I_stim` read** (15d) — the demos must not double-count.
- Any astro-demo fix **reproduce-first** + a `for_loop`-lowering regression test (12) if Simulator-run. Drive multi-step runs with `brainstate.transform.for_loop` (CLAUDE.md #10). **`clear_caches()` + x64** for the stiff Ca dynamics (21).
- NumPy docstrings; >90 % coverage on touched code. Coverage via `coverage run --source=brainpy_state` + the coverage API.

## Validation (via the 00 harness)
`_nest/_validation/{astrocyte_single,astrocyte_interaction}_test.py`, `requires_nest`: the IP3/Ca/SIC waveform (single) + the SIC-modulated loop (interaction), each vs NEST in a documented band. `{astrocyte_small_network,astrocyte_brunel}_test.py` = marker/skip tests asserting the `_BLOCKED_REASON`. Each runnable example runs standalone without NEST (parity skips; demo still plots its Ca/SIC traces). README §3.8 rows.

## Edge cases to enumerate + test
Sub-threshold Ca (no SIC → `I_SIC=0`); strong spike drive (Ca crosses `SIC_th` → SIC); the neuron→astro delta arm alone (IP3 without return SIC); `weight=0` SIC (decoupled); `delay_steps=10` NEST-default vs `1` min-delay; standalone without NEST; `for_loop` lowering of the loop; the blocked-demo skip markers fire.

## Definition of done
- [ ] `astrocyte_single` + `astrocyte_interaction` ported + runnable standalone, each with a live-NEST parity test (pass/skip) + a no-NEST companion.
- [ ] `astrocyte_small_network` + `astrocyte_brunel_*` shipped as documented skipped placeholders + marker tests; the `TripartiteConnect` blocker recorded in `network-api-gap.md`.
- [ ] Any astro-demo fix reproduce-first + (if Simulator-run) `for_loop`-lowering-tested; `clear_caches()`+x64.
- [ ] >90 % coverage on touched code; README §3.8 rows updated; Lessons entry in `CONTEXT.md`; git lifecycle done. **§3.7/§3.8 then complete bar the pool-rule placeholders.**

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the astro-demo parity recipe (the validated loop packaged as examples; the in-scope vs `TripartiteConnect`-blocked split), any fix, and confirmation that **§3.8 single/interaction are complete** and the pool-rule demos are the one documented deferral (a future connectivity-rule cluster). Note **bucket-3 is now fully closed** (all models + all unblocked demos).
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
