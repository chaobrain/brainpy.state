# /goal — 19 · §3.10 pedagogical demos (`brette_gerstner_fig_*`, `compartmental_model/`, `sudoku/`, `pong/`)

> Self-contained prompt. Open at repo root. Stream: example. **Routine demo cluster — bucket-3-independent.** Rides only the §3.2/§3.5 Simulator substrate + already-merged models. Depends on: `02` (analog `multimeter` recording + current generators), the merged `aeif_cond_alpha.py` + `cm_default.py` (§3.5 single-neuron stream), and `08` (`stdp_dopamine_synapse` + `volume_transmitter`). Independent of bucket-3 (`15*`/`17*`), of `20` (spatial), and of `24` (tripartite).

## Substrate check — 19 has NO blocked sub-demo (read first)
A capability audit of `brainpy_state/_nest/` confirms every model the four §3.10 groups need is **already merged**:
- `aeif_cond_alpha.py` → `brette_gerstner_fig_2c/3d` (AdEx single-neuron figures).
- **`cm_default.py` → `compartmental_model/` is NOT blocked** — the general compartment-tree model (soma+dendrite, AMPA/NMDA receptors, active/passive channels) is implemented. (The earlier examples-gap "blocked on `cm_default`" note is stale.)
- `stdp_dopamine_synapse.py` + `volume_transmitter` (cluster `08`) → `pong/`'s reward-modulated (R-STDP) learning.
- The §3.2 Simulator recording + current-generator substrate → `sudoku/`'s structured stochastic iaf network.

So 19 is **routine ports**, ordered easiest→hardest. The only real judgement is the **functional-parity band** for the two structured/RL nets (sudoku solve-rate, pong ball-tracking) — distributional, the `14` posture.

## Session start
1. Create an isolated worktree branch **`nest-goal/19-pedagogical`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`02` / §3.5 single-neuron Lessons** — the Simulator demo house style + analog `multimeter`/current-generator recording (`res.trace`); the tight deterministic V-trace parity posture for single-neuron figures.
   - the **`12` / `cm_default` Lessons** + the **`21` Lessons** — the multi-compartment recording + the **`clear_caches()` + float64 (x64) guard** mandatory for stiff multi-compartment parity, and `align_steps` for pipeline latency.
   - the **`08` dopamine-vt Lessons** — `stdp_dopamine_synapse` + `volume_transmitter` broadcast/relay timing (the substrate pong's R-STDP rides).
   - the **`14`-network-demos Lessons** — distributional functional parity (seed-mean/median band + the qualitative law); steady-state on the 2nd half.
   - the **`16`-generator-demos Lessons** — the demo-cluster house style (a runnable `examples/nest/` script + one test module carrying a NEST-free structural class **and** a `skipUnless(nest)` parity class; `interval=dt`; import helpers `from examples.nest.<module>`; a no-NEST companion).
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.10 rows (+ the P2 pong/sudoku acceptance text: "both ported; functional behavior — pong: ball-tracking accuracy; sudoku: solve rate — matches within the documented band").
   - The merged substrate: `brainpy_state/_nest/{aeif_cond_alpha,cm_default,iaf_cond_alpha_mc,stdp_dopamine_synapse}.py`; `_network/_simulator.py` recording + current-injection + dopamine/VT paths.
   - The house style: `examples/nest/{recording_demo,brunel_siegert}.py` + `../../README.md`.
   - NEST upstream (clone on this machine: **`/mnt/d/codes/githubs/computational_neuroscience/nest-simulator`** — the CLAUDE.md path; the 17-Lessons `…/githubs/nest-simulator` note is machine-specific and **absent here**, `ls`-verify before chasing it): `pynest/examples/brette_gerstner_fig_2c.py` + `_fig_3d.py`; `compartmental_model/{two_comps,receptors_and_current}.py`; `sudoku/{sudoku_net,sudoku_solver,helpers_sudoku}.py`; `pong/{networks,pong,run_simulations}.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the sign-off is the **four-group scope + the functional-parity bands** + the pong decomposition decision (below).

## Goal
Port the four §3.10 pedagogical demo groups to `examples/nest/` on the `Simulator` API — deterministic V/state-trace parity for the single-neuron + compartmental figures, distributional functional parity (solve-rate / ball-tracking band) for the structured/RL nets — each with a live-NEST test (pass/skip) + a no-NEST companion.

## Scope (ordered easiest → hardest)
- **`brette_gerstner_fig_2c` + `_fig_3d`** → one `aeif_cond_alpha` driven by `dc_generator`/`step_current_generator` (fig 2c: two DC epochs `amplitude=[500,800] pA`; fig 3d: the spike-adaptation protocol); record V at `interval=dt`; assert the V-trace + spike times vs NEST **tight** (deterministic; `a`/`b` from the paper, `b` in pA). The §3.5 posture.
- **`compartmental_model/two_comps` + `receptors_and_current`** → a `cm_default` two-compartment model (soma+dendrite; passive + active Na/K; AMPA+NMDA receptors on both compartments); record the per-compartment V + receptor states; assert vs NEST **tight where deterministic** with **`clear_caches()` + x64** (stiff multi-compartment — the `21` discipline) and `align_steps`.
- **`sudoku/`** → the structured stochastic-iaf constraint-satisfaction net (one `iaf_psc_*` cell per (row,col,digit); fixed inhibitory WTA connectivity encoding the row/col/box constraints + noise drive); run the solver; assert the **solve rate** over seeds distributionally vs NEST (documented band; the `14` posture). Keep the puzzle/IO helpers plain-Python (port faithfully from `helpers_sudoku.py`).
- **`pong/`** *(the stretch — split to a `19b` follow-up at approach-time if the env port balloons)* → the R-STDP Pong demo: two paddle networks (one `stdp_dopamine_synapse`/`volume_transmitter` learner, one fixed/baseline) driven by a ball-position input layer; reward modulates the learner's STDP. Port the spiking networks + the per-game reward loop; keep the game environment (ball, paddles, scoring) plain-Python (faithful to `pong.py`/`networks.py`). Assert the **learner's ball-tracking accuracy improves with training and matches NEST** distributionally (seed-mean band) — not per-spike.

## Constraints
- `Simulator` API only; `examples/nest/` house style; **brainunit** units; import helpers `from examples.nest.<module>` (never `from brainpy`); keep a no-NEST companion per example.
- Parity tiers: **tight deterministic** (brette_gerstner V-trace; compartmental per-compartment state where deterministic) vs **distributional functional** (sudoku solve-rate, pong ball-tracking — seed-mean/median band + the qualitative law). Set the NEST recorder `interval=dt`. `align_steps` for any pipeline-latency offset.
- **`clear_caches()` + x64** for the stiff `cm_default` compartmental parity (the `21` discipline).
- **Conductance + mid-run-param notes (from 17b/17):** `aeif_cond_alpha` here is **DC/`I_e`-driven** (brette_gerstner) so it dodges the spike→conductance path 17b found latent on the conductance family (`I_e`-only tests don't catch it). If *any* 19 demo drives a conductance neuron with **spikes**, assert a non-trivial `g_ex`/`g_in`/`V_m` response — 17b's **"silent dead conductance path"**: a missing multi-receptor bridge pins `V_m` at `E_L` with **no error**. For any mid-run scalar-param change (a protocol phase), **build a fresh net per phase** — `NodeView` (the `create()` return) does **not** forward attribute writes to the wrapped module (17), so `view.param = …` sets a dead attribute and `simulate()` re-inits from the constructor value anyway.
- Any demo fix **reproduce-first** (`superpowers:systematic-debugging`) + a **`for_loop`-lowering regression test** (`12`) if the Simulator runs it. Drive every multi-step run with `brainstate.transform.for_loop`/`scan` — **never a bare Python `for`/`while`** over the model (CLAUDE.md #10); the env/IO host loop (pong frames, sudoku display) stays plain-Python but the network rollout per frame/step lowers via `for_loop`.
- NumPy docstrings; >90 % coverage on touched code. **Coverage (corrected by 17/17b):** `coverage run -m pytest <touched test files>` then `coverage report --include=…` — **drop `--source`** (both `--source=brainpy_state` and `--cov=pkg.mod` SIGABRT inside jaxlib/absl double-init) and let the run finish fully before reading `.coverage` (a partial file reports misleadingly low).

## Validation (via the 00 harness)
`_nest/_validation/{brette_gerstner_fig_2c,brette_gerstner_fig_3d,compartmental_two_comps,sudoku_solve,pong_learning}_test.py`, `requires_nest`:
- brette_gerstner: V-trace + spike times vs NEST (tight band).
- compartmental: per-compartment V/receptor state vs NEST (tight where deterministic; x64).
- sudoku: solve-rate over N seeds vs NEST (documented distributional band).
- pong: pre-vs-post-training ball-tracking accuracy (learning occurred) + the seed-mean band vs NEST.
Each runnable example runs standalone (`PYTHONPATH=. python examples/nest/<name>.py`) **without** NEST (parity skips; the demo still plots its trace / solve / game).

## Edge cases to enumerate + test
AdEx with adaptation off (`a=b=0` → plain exp-IF); compartmental with the dendrite passive-only vs active (the two `two_comps` variants); a receptor with AMPA-only vs AMPA+NMDA; sudoku with a trivially-constrained board (fast solve) vs a hard board (band); pong with learning **disabled** (no improvement — the control) vs enabled; standalone-without-NEST for each; `for_loop` lowering of every Simulator rollout; `clear_caches()`/x64 on the compartmental run.

## Definition of done
- [ ] brette_gerstner figs + compartmental_model ported + runnable standalone, each with a tight live-NEST parity test (pass/skip) + a no-NEST companion.
- [ ] sudoku ported with a distributional solve-rate parity test; pong ported (or split to `19b` with a recorded reason) with a distributional ball-tracking parity test.
- [ ] Any demo fix reproduce-first + (if Simulator-run) `for_loop`-lowering-tested; `clear_caches()`+x64 on the compartmental parity.
- [ ] >90 % coverage on touched code; README + examples-gap §3.10 rows updated; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the pedagogical-demo recipe (which groups are tight-deterministic vs distributional-functional; the `cm_default` compartmental parity recipe + x64; the pong R-STDP env-loop posture and whether it split to `19b`), any demo fix, and confirmation **§3.10 is complete** (or the one documented `19b` deferral). Note that the audit found **`cm_default` already shipped** (the stale "blocked" note retired).
2. Verify tests pass / skip; coverage > 90 %; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
