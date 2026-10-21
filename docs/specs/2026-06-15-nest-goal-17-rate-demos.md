# /goal — 17 · §3.6 rate-network demos (`lin_rate_ipn_network`, `rate_neuron_dm`)

> Self-contained prompt. Open at repo root. Stream: example. **Bucket-3 demo cluster** (the rate half). Depends on: `15a` (rate-core: seam-(H) continuous emission, `rate_connection_*`, the φ-homogeneity guard) — **MERGED (#64)**. **The gap demos (`gap_junctions_*`) already shipped with `15b` (#67); `brunel_siegert` with `15c` (#66)** — so this cluster is the **rate-network** example ports only. Independent of `17b` (astro).

## What's already done (don't re-port)
`examples/nest/` already carries `gap_junctions_two_neurons.py` + `gap_junctions_inhibitory_network.py` (15b) and `brunel_siegert.py` (15c). This cluster ports the **two remaining §3.6 rate examples**: `lin_rate_ipn_network` + `rate_neuron_dm`. (`ht_neuron`/`intrinsic_currents_*` stay **out** — single-neuron intrinsic currents, not continuous coupling.)

## Session start
1. Create an isolated worktree branch **`nest-goal/17-rate-demos`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`15a-rate-core` Lessons** — the rate API the demos drive: a rate neuron declares `_emission_continuous=True` + `_emission_attr='rate'` (`linear_summation=True`) or `'phi_rate'` (False); `Simulator.connect(pre, post, synapse=rate_connection_instantaneous(weight=w))` deposits `weight·rate` into the post's **default delta channel** (`comm='dense'`), read via `sum_delta_inputs(0.0)`; `rate_connection_delayed(weight, delay)` rides `InputDelay`. **Instantaneous lag diverges on the transient but preserves the fixed point** — trajectory parity needs `align_steps`, the **FP needs nothing** (it breaks the algebraic `r0←r1←r0` loop). The `linear_summation` **homogeneity guard** (`_check_rate_phi_homogeneity`) raises at `connect()` when φ-modes mismatch (or both-False with differing `_phi_signature`). The 5 φ formulas == NEST C++ exactly; `mult_coupling` sign-splits into `'rate_ex'`/`'rate_in'`. **gauss can't be driven noise-free** (`sigma>0`); **`rate_neuron_opn` has no `lambda_`/`rectify_output`**.
   - the **`16-generator-demos` Lessons** — the demo-cluster house style: each example a runnable `examples/nest/` script + **one** test module carrying **both** a NEST-free structural class (always runs) **and** a live-NEST distributional class (`skipUnless`); set the NEST recorder `interval=dt`; import helpers `from examples.nest.<module>` (never `from brainpy`); keep a no-NEST companion.
   - the **`14-network-demos` Lessons** — distributional/FP network parity (seed-mean/median band + the qualitative law; steady-state on the 2nd half).
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.6 rate rows (`lin_rate_ipn_network` / `rate_neuron_dm` + any status markers).
   - The merged rate substrate: `brainpy_state/_nest/{lin_rate,rate_neuron_ipn,rate_neuron_opn}.py` + `rate_connection_{instantaneous,delayed}.py`; `_network/_simulator.py` `_is_continuous_rate`/`_build_rate_dual_channel`/`_check_rate_phi_homogeneity`/`_sign_split_weight`; `_network/_event_proj.py` `channel_label`.
   - The house style: `examples/nest/{brunel_siegert,gap_junctions_two_neurons,recording_demo}.py` + `../../README.md`.
   - NEST upstream: `pynest/examples/lin_rate_ipn_network.py` + `rate_neuron_dm.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code.**

## Goal
Port the two §3.6 rate-network demos to `examples/nest/<name>.py` on the `Simulator` API, each with a live-NEST parity test (FP/distributional), driving any rate-demo fix back into the merged substrate.

## Scope
- **`lin_rate_ipn_network`** → a linear-rate network (`lin_rate`/`rate_neuron_ipn`) with instantaneous + delayed rate connections; assert the **steady-state fixed point** `r* = (I − gC)⁻¹μ` vs the closed form **AND** NEST (tight, `use_wfr=False`), and the trajectory vs NEST with `align_steps` (the instantaneous-lag transient offset).
- **`rate_neuron_dm`** → the rate-neuron decision-making net (two mutually-coupled rate pops + bias); assert the **decision behaviour** (which pop wins vs bias; steady-state rates) distributionally vs NEST (bias-driven WTA; seed-mean/median band).

## Constraints
- `Simulator` API only; `examples/nest/` house style; `brainunit` units; import helpers `from examples.nest.<module>`; keep a no-NEST companion.
- Parity is **FP/distributional** (NEST `use_wfr=False` vs the substrate's one-step lag): assert the fixed point (needs **no** alignment) + the trajectory **with** `align_steps`; never per-sample without alignment. Set the NEST recorder `interval=dt`.
- Respect the rate gotchas (gauss `sigma>0`; `rate_neuron_opn` exclusions; keep φ **homogeneous** across a `linear_summation=False` projection so the guard passes).
- Any rate-demo fix **reproduce-first** (`superpowers:systematic-debugging`) + a `for_loop`-lowering regression test (12) if the Simulator runs it. Drive multi-step runs with `brainstate.transform.for_loop` (CLAUDE.md #10).
- NumPy docstrings; >90 % coverage on touched code. Coverage via `coverage run --source=brainpy_state` + the coverage API (the `--cov=pkg.mod` form coredumps; rtk proxy strips `--cov`).

## Validation (via the 00 harness)
`_nest/_validation/{lin_rate_ipn_network,rate_neuron_dm}_test.py`, `requires_nest`: the linear-rate FP (vs closed form + NEST) + the dm decision (distributional band). Each example runs standalone (`PYTHONPATH=. python …`) without NEST (parity skips, the demo still plots its rate trajectory / decision). README §3.6 rate rows.

## Edge cases to enumerate + test
Zero-coupling control (`r*=μ`); a mutual-coupling loop (the FP ≠ the feed-forward one-pass estimate — the substrate finds the loop FP); strong bias (deterministic dm winner) vs zero bias (assert the *distribution*); delayed vs instantaneous connection (the delay shifts the response by `delay_steps`); `dt`-invariance of the FP; standalone without NEST; `for_loop` lowering.

## Definition of done
- [ ] `lin_rate_ipn_network` + `rate_neuron_dm` ported + runnable standalone, each with a live-NEST FP/distributional parity test (pass/skip) + a no-NEST companion.
- [ ] Any rate-demo fix reproduce-first + (if Simulator-run) `for_loop`-lowering-tested.
- [ ] >90 % coverage on touched code; README §3.6 rate rows updated; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the rate-demo parity recipe (FP vs closed-form+NEST; the `align_steps` transient offset; the dm decision band), any rate-demo fix, and confirmation the **§3.6 rate demos are complete** (gap + siegert already shipped by 15b/15c). Note `ht_neuron`/`intrinsic_currents_*` as still out (single-neuron, not coupling).
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
