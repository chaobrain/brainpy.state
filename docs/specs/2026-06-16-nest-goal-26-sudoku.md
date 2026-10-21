# /goal — 26 · §3.10 `sudoku/` — stochastic-WTA constraint-satisfaction solver (the one §3.10 item 19 deferred)

> Self-contained prompt. Open at repo root. Stream: example. **Single-demo cluster — the deferred §3.10 stretch.** Rides the §3.2/§3.5 Simulator substrate + the **closed-loop seam `19` shipped** (`Simulator.cont()` + `host_*_drive` + per-edge `get_connections().set('weight')` + `res.spikes(recorder)` readout). Depends on: `02` (recording + generators), the merged `iaf_psc_exp.py`, `poisson_generator.py`/`noise_generator.py`, `spike_recorder`, and **`19` (PR #75) — the pong host-in-the-loop demo is the working precedent to copy**. Independent of `20` (spatial), `24` (tripartite), `25`/`28` (conductance/PSC sweeps).

## Why this was deferred — read first (the whole point of cluster 26)
`19` ported the other three §3.10 groups but **deferred `sudoku/`** (CONTEXT.md 19-Lessons, verbatim): *"its noise-driven WTA constraint network did not reach NEST's solve rate within a practical substrate config; deferred as a TODO … rather than shipped half-working,"* and *"`sudoku/` is the one §3.10 item left (needs a **tractable WTA-relaxation formulation**)."* The gap doc agrees (`examples-gap.md` §3.10, P2 [L]): *"Investigated and found intractable to bring to solve-rate parity on the current substrate — recorded TODO."*

**The thesis of this cluster:** "intractable within a practical substrate config" almost certainly means the *driving pattern*, not the dynamics. The substrate now has the exact primitives that make a 729-population, multi-chunk, multi-puzzle relaxation practical — and `19` itself shipped and proved them on **`pong/`**, a structurally identical host-in-the-loop demo:
- **`Simulator.cont(sim_time)`** (`_simulator.py:1818`) — persistent chunked rollout reusing **one** compiled `for_loop` as long as only State *contents* change. This is the direct analog of NEST's `while not valid: nest.Simulate(100 ms)`.
- **`get_connections(stim, cell).set('weight', w*u.pA)`** (pong `pong_networks.py:185`) — per-puzzle clue stimulation is a **live State write**, no recompile.
- **`res.spikes(recorder)` → `(poll_steps, n)`** + a per-chunk count sum (pong `get_spike_counts`, `pong_networks.py:210-213`) — the per-population spike readout, reset each chunk.
- **`reset_rollout()`** (`_simulator.py:1804`) — restart at t=0 to re-seed `V_m`/clear counters per puzzle.

So the **tractable formulation = build SudokuNet *once* (fixed topology, fixed shapes) and run the solver as a pong-style host loop** — never rebuild the net or retrace per chunk/per puzzle. If `19` hit "intractable" by rebuilding/retracing each `Simulate`, *that* is the fix and most of this cluster is a faithful port. **Diagnose which root you hit before tuning dynamics** (§ Diagnose below).

## Session start
1. Create an isolated worktree branch **`nest-goal/26-sudoku`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`19`-pedagogical Lessons** — the `cont()` + `host_*_drive` closed-loop seam contract, and the exact sudoku-deferral wording (this cluster's mandate).
   - the **`14`-network-demos Lessons** — **distributional functional parity** (seed-mean/median band + the qualitative law); steady-state on the 2nd half. This is the acceptance posture for a *stochastic* solver — there is no per-sample parity.
   - the **`16`-generator-demos Lessons** — the demo-cluster house style (a runnable `examples/nest/` script + one test module carrying a NEST-free structural class **and** a `skipUnless(nest)` parity class; `interval=dt`; import helpers `from examples.nest.<module>`; a no-NEST companion).
   - any **`08`/`02` recording Lessons** touching `spike_recorder` count readout.
   Then read:
   - **`examples/nest/pong_networks.py` + `pong_run.py` in full — the working precedent.** Copy its idiom: net built once in `__init__`; `spike_recorder` tapping a layer; `res.spikes(recorder)` → counts; `get_connections().set('weight', …*u.pA)`; `cont()` per turn; `reset_rollout()` to seed; `precision=64, platform='cpu'`.
   - The merged substrate: `brainpy_state/_nest/{iaf_psc_exp,poisson_generator,noise_generator,spike_recorder}.py`; `_network/_simulator.py` (`cont` 1818, `reset_rollout` 1804, `get_connections` 828, recording/`res.spikes`).
   - `docs/nest-status/internal/examples-gap.md` §3.10 (the `sudoku/` row 339 + P2 item 494) — you will flip this row.
   - NEST upstream (clone on this machine: **`/mnt/d/codes/githubs/computational_neuroscience/nest-simulator`** — the CLAUDE.md path; the 17-Lessons `…/githubs/nest-simulator` note is machine-specific and **absent here** — `ls`-verify before chasing it): `pynest/examples/sudoku/{sudoku_net,sudoku_solver,helpers_sudoku}.py` + `README.rst`.
3. Write spec + plan under `` (gitignored). **Approach-first; wait for approval before code** — the sign-off is: the **diagnosis** (runtime-pattern vs dynamics root), the **solve-rate parity band + puzzle/seed scope**, and the **documented-partial fallback** boundary (below).

## Goal
Port `sudoku/` to `examples/nest/` on the `Simulator` API as a **single compiled net driven by a host-side relaxation loop** (the `cont()` + `set('weight')` + `res.spikes` seam `19`/pong shipped), reaching **NEST's solve rate within a documented distributional band** on a stated puzzle set — or, if convergence stays short after honest tuning, shipping the faithful mechanism + a **measured, documented partial** (upgrading the TODO from "missing/intractable" to "ported, here is the band + diagnosis"). One live-NEST parity test (pass/skip) + a no-NEST companion + convergence-independent component tests.

## Faithful architecture port (from `sudoku_net.py` / `sudoku_solver.py` / `helpers_sudoku.py`)
- **Populations:** 729 = 9 rows × 9 cols × 9 digits, `pop_size=5` `iaf_psc_exp` each → 3645 neurons. `neuron_indices = arange(3645).reshape(9,9,9,pop_size)`. Params: `C_m=0.25 nF, I_e=0.5 nA, tau_m=20, t_ref=2, tau_syn_ex=tau_syn_in=5, V_reset=-70 mV, E_L=-65 mV, V_th=-50 mV`, `V_m ~ uniform(-65,-55) mV` init. `delay=1.0 ms` everywhere. **Use brainunit units** (`mV`, `nF`, `nA`, `ms`, `pA`).
- **Inhibitory WTA + constraint topology** (`inter_neuron_weight = -0.2`): all-to-all (at `pop_size` resolution) between populations coding *the same digit in the same row*, *…same col*, *…same box*, **and** *different digits in the same cell* — `unique`, minus self. Build the **explicit edge list** from the same index logic NEST uses (port `__init__`'s `row_targets/col_targets/box_targets/digit_targets`) and wire **one** inhibitory projection. `iaf_psc_exp` is current-based PSC → a **negative weight is inhibitory current** (confirm the sign convention in `iaf_psc_exp.py`; no conductance bridge here — this is the PSC family, not the §4 conductance one).
- **Background noise:** `poisson_generator` at `noise_rate=350 Hz`, `weight_noise=1.6`, to **all** 3645 neurons (a `noise_generator`/`poisson_generator` broadcast, as pong does).
- **Per-population stim:** 729 `poisson_generator`s at `stim_rate=200 Hz`, one→its population, `weight_stim=1.3` for **clue** cells and **0** otherwise. Set the clue pattern via **`get_connections(stim, cell).set('weight', …*u.pA)`** (the pong idiom) — *not* by rebuilding the net. `set_input_config(puzzle)` (a 9×9 array, nonzero = clue digit) maps to which (cell,digit) stim edges get `1.3` vs `0`.
- **Readout:** one `spike_recorder` per population (or one wide recorder tapping all, columns = populations — prefer the wide form, as pong does). Per chunk: `counts = res.spikes(recorder).sum(axis=0)` reshaped `(9,9,9)`; per cell `winning_digit = argmax over digits` with **random tiebreak +1** (port faithfully). Build the 9×9 solution; `validate_solution(puzzle, solution) → (valid, cells, rows, cols)`; `ratio_correct = (cells.sum()+rows.sum()+cols.sum())/27`.
- **Solver loop (host-side):** `while not valid and run < max_iterations`: clear counters → `cont(sim_time=100 ms)` → readout → validate. `max_iterations = max_sim_time/sim_time = 10000/100 = 100`. Keep all puzzle/IO/validation helpers **plain-Python** (port `get_puzzle` — 8 puzzles, index 0–7; `validate_solution`; skip `plot_field`/matplotlib in tests).

## The host-loop nuance of CLAUDE.md #10 (read carefully — do not over-apply it)
- The **outer solve loop** (chunk → readout → validate → re-stimulate → repeat) is **host-side Python** — this is correct and is *exactly* the closed-loop seam (NEST's own `while not valid: Simulate(100ms)`; pong's per-turn loop). Do **not** try to fuse the whole solver into one giant `scan`.
- The **inner per-step dynamics** (the 100 ms chunk) **must** lower via `cont()` (a compiled `for_loop`) — **never** a bare Python step loop over the network. Add a `for_loop`-lowering regression test (`12`) for the chunk rollout.

## Diagnose before you tune (the crux — record this in the spec)
Measure **live NEST's own solve rate** on your chosen puzzle set first — that is the target band, not a paper number. Then port faithfully and run. Two roots of "intractable", treat in order:
1. **Runtime/scale (most likely).** If a faithful single-compiled-net port (build once, `cont()` per chunk, `set('weight')` per puzzle) runs the full solver in practical wall-clock, the original "intractable" was the *driving pattern* (rebuild/retrace per `Simulate`) and you are essentially done — proceed to the band. Reuse pong's proof that this scales.
2. **Dynamics convergence.** If it runs fast but **under-solves** vs NEST, the effective stochastic WTA differs (Poisson statistics, PSC kinetics, `V_m`-init spread, or the `noise/inhibition/stim` balance). Re-tune **the balance, not the topology** — `noise_rate`/`weight_noise` (exploration), `inter_neuron_weight` (WTA sharpness), `weight_stim` (clue clamp) — to recover NEST's convergence on the *same* puzzles. Document every deviation from NEST's constants and why.

## Acceptance — distributional, the `14` posture (no per-sample parity)
- **Primary (functional) band:** solve rate (and median time-to-solution, in chunks) over a **stated puzzle set × N seeds**, within a **documented band of live NEST** measured on the same puzzles/seed-policy/`sim_time`/`max_iterations`. Pick a representative set given runtime (e.g. an easy + a medium + the default index-4 board; index-6 "world's hardest" and index-0 "dream" as documented stretch/holdout). **No silent caps:** `log`/document exactly which puzzles, how many seeds, and any board you dropped for runtime + why.
- **Convergence-independent component tests** (these gate the port even if the band is partial):
  - **Topology edge-set equality:** the inhibitory edge set == an independent reference builder over the row/col/box/cell index sets (exact set equality; this is the faithful-WTA guarantee).
  - **Clue clamp:** with `set('weight')` marking a clue cell, that cell's clue digit wins its WTA within a chunk; an un-clued cell relaxes (no spurious lock).
  - **Rates:** measured Poisson noise/stim rates match `350/200 Hz` within sampling tolerance; `weight_stim` is `1.3` on clue edges / `0` elsewhere after `set_input_config`.
  - **`for_loop` lowering** of the chunk rollout; **standalone-without-NEST** runs (solver still relaxes + prints a board).
- **Documented-partial fallback (explicit, like `19` did for this very demo):** if convergence stays short of the band after honest tuning, **ship** the faithful mechanism + component tests green, and record the **measured** solve rate vs NEST, **which** puzzles converge / don't, and the suspected cause. That upgrades the gap-doc row from *missing/intractable* to *ported; converges on easy/medium; gap on hard; band = X vs NEST = Y; diagnosis = Z* — a real improvement over the current state, not a half-working ship.

## Constraints
- `Simulator` API only; `examples/nest/` house style; **brainunit** units; example code uses the **public** namespace (`from brainpy import state as …`) — never reach the inner package through `brainpy.state` from inside it; import demo helpers `from examples.nest.<module>`; keep a no-NEST companion.
- Build the net **once** (fixed shapes) and drive it host-side via `cont()`/`set('weight')`/`res.spikes` — the pong precedent. `precision=64, platform='cpu'`.
- Any fix **reproduce-first** (`superpowers:systematic-debugging`) + the `for_loop`-lowering regression test.
- NumPy docstrings; >90 % coverage on touched code. **Coverage (corrected by 17/17b):** `coverage run -m pytest <touched test files>` then `coverage report --include=…` — **drop `--source`** (both `--source=brainpy_state` and `--cov=pkg.mod` SIGABRT inside jaxlib/absl double-init); let the run finish before reading `.coverage` (a partial file reports misleadingly low). The 24-robust variant if needed: pre-import `jax`/`jaxlib`/`brainstate` before `cov.start()` + `COVERAGE_CORE=sysmon` + `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## Validation (via the 00 harness)
`_nest/_validation/sudoku_*_test.py`, `requires_nest`:
- `sudoku_topology_test.py` — inhibitory edge-set == reference builder (no NEST needed; pure structure).
- `sudoku_clue_clamp_test.py` — clue cell locks; un-clued relaxes (small board / short chunk).
- `sudoku_solve_test.py` (`skipUnless(nest)`) — solve rate over the stated puzzle set × seeds within the documented band vs live NEST; records median time-to-solution.
- The runnable example runs standalone (`PYTHONPATH=. python examples/nest/sudoku.py`) **without** NEST (parity skips; the solver still relaxes a board and prints it).

## Edge cases to enumerate + test
Index-0 "dream" board (all-zeros — net must produce *any* valid sudoku, no clue clamp); a trivially-constrained near-solved board (fast convergence, few chunks); the default index-4 board (the band); index-6 "world's hardest" (documented stretch/holdout — may be in the partial); a board with **all** clues (every cell clamped → should validate ~immediately, a clue-path sanity check); random-tiebreak determinism under a fixed seed; `max_iterations` reached without solve (graceful report, not a hang); standalone-without-NEST; `for_loop` lowering of the chunk.

## Definition of done
- [ ] `examples/nest/sudoku.py` (+ a `sudoku_net`-style helper module if it keeps the example clean) ported as a **single compiled net + host relaxation loop**; runnable standalone without NEST.
- [ ] Component tests green (topology edge-set, clue clamp, rates, `for_loop` lowering); solve-rate parity test passes the **documented band** vs live NEST — **or** the recorded documented-partial with the measured band + diagnosis.
- [ ] Diagnosis (runtime-pattern vs dynamics root) recorded; every deviation from NEST's constants justified.
- [ ] >90 % coverage on touched code; README + `examples-gap.md` §3.10 `sudoku/` row updated (missing → ported / documented-partial); Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the tractable-WTA-relaxation recipe (single compiled net + `cont()`/`set('weight')`/`res.spikes` host loop — and **which root** "intractable" actually was), the **measured** solve-rate band vs NEST + puzzle/seed scope, any dynamics re-tuning + justification, and whether §3.10 `sudoku/` shipped to band or as a **documented partial** (which puzzles converge/don't). Note that this **closes the one §3.10 item `19` left** (or records the residual precisely).
2. Verify tests pass / skip; coverage > 90 %; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + the measured band) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
