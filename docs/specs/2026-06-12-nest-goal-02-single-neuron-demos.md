# /goal — 02 · Single- & few-neuron demos (§3.2) + Simulator extensions

> Self-contained prompt. Open at repo root. Stream: example (foundation). Depends on: 00.
> Supersedes `nest-3.2-demos-prompt.md`.

## Session start

1. Create an isolated worktree branch **`nest-goal/02-single-neuron-demos`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full.** Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.2.
   - The established idiom: `examples/nest/brunel_alpha.py`, `examples/nest/README.md`, and `brainpy_state/_network/_simulator.py` (the `Simulator`/`SimulationResult`/`_taps` you will extend — today it records **spikes only** and treats every generator as a **spike** source).
   - The 7 NEST upstream scripts at `/mnt/d/codes/githubs/computational_neuroscience/nest-simulator/pynest/examples/`: `one_neuron.py`, `one_neuron_with_noise.py`, `twoneurons.py`, `testiaf.py`, `balancedneuron.py`, `if_curve.py`, `vinit_example.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code.**

## Goal

Port the 7 §3.2 demos to `examples/nest/<name>.py` on the **`Simulator` API** (NEST-flavored — **not** brainpy.state's native compositional API), each backed by a live-NEST parity test that drives `_nest` model fixes. These demos force the **example-stream Simulator extensions** the rest of the example clusters reuse.

| # | port | model(s) under test | asserts |
|---|---|---|---|
|1|`one_neuron`|`iaf_psc_alpha`,`voltmeter`|`I_e=376 pA` charges `V_m`|
|2|`one_neuron_with_noise`|`iaf_psc_alpha`,`poisson_generator`×2|two drives (80/15 kHz), weights `[1.2,-1.0]`|
|3|`twoneurons`|`iaf_psc_alpha`,`static_synapse`|`I_e` driver, `w=20,delay=1`; PSC shape in post|
|4|`testiaf`|`iaf_psc_alpha`,`voltmeter`,`spike_recorder`|charge→spike→**refractory**→recover, `dt∈{0.1,0.5,1.0}`|
|5|`balancedneuron`|`iaf_psc_alpha`,`poisson_generator`×2|scipy `bisect` finds inhib rate → out=5 Hz; repeated re-sim|
|6|`if_curve`|`aeif_cond_exp`,`noise_generator`|F-I curve over `(I_mean,I_std)`; **noise current**|
|7|`vinit_example`|`iaf_cond_exp_sfa_rr`,`voltmeter`|sub-threshold relaxation from several initial `V_m`|

## Scope — Simulator extensions (build in `brainpy_state/_network/`)

The current `Simulator` can't express §3.2. Add (TDD, each with `_network/*_test.py`):

- **A. Analog/state recording** — `voltmeter`/`multimeter`: `connect(voltmeter, neuron)` (NEST's reversed direction — recorder is *pre*, *observes* post) captures `V_m`/recordables per step in the `for_loop`, exposed as `res.trace(rec,'V_m') -> (T,N)` + `res.times`. (Multimeter stays imperative; **tap `State`**, per `CONTEXT.md` Part 3.) Needed by 6/7 demos.
- **B. Current-injecting devices** — `I_e` (neuron param; demos 1/3/4), `noise_generator` (mean+std white-noise **current**; demo 6), `dc_/step_current_generator`. These inject **current**, not spike events — the delta-event path is the wrong model. Add a current-injection wiring summed into the neuron's input each step.
- **C. Sweep / re-run ergonomics** — demos 5/6 re-simulate with a changed device param and reset counts between trials. Support cheap rebuild-per-trial (mirrors NEST `ResetKernel`-in-loop) or mutable-param + re-`simulate`, plus per-trial `n_events`/`rate`.
- **D. Per-generator weight vectors** — two generators → one neuron with signed weights `[1.2,-1.0]` / `[epsc,ipsc]` (demos 2/5).

## Constraints

- **`Simulator` API only** — no `brainstate.nn.*`/`Projection`/`Network` native style. Match `examples/nest/brunel_alpha.py` house style (module docstring with `Run:`, a `build(...)` returning sim+handles, a `main()` printing headline numbers + `*.png` when matplotlib present, `if __name__=='__main__'`).
- Model bugs are fixed in `_nest/<model>.py`, never worked around in the example or `_network`. When parity fails, `superpowers:systematic-debugging`: a focused reproducing unit test in `_nest/<model>_test.py`, then fix until both it and the parity test pass.
- Units in `saiunit` (pA / mV / Hz / ms). NumPy docstrings.

## Validation bar (via the `00` harness)

Live-NEST parity per demo, `skipUnless(nest)`, in `brainpy_state/_nest/_validation/`:
1,2,3,7 → `V_m(t)` trace (max-abs mV; **distributional** where Poisson/noise drives, compare V_m stats over seeds). 4 → spike count + `V_m` at each `dt`. 5 → bisected inhib rate. 6 → F-I curve per grid point. Document each tolerance.

## Edge cases

`dt` sensitivity (4); bisection non-convergence (5); zero-rate grid points (6); `V_m` init at/above threshold (7); two-generator signed-weight summation (2/5); analog recording length vs `dt`.

## Definition of done

- [ ] 7 ports in `examples/nest/`, `Simulator`-API-only, runnable standalone.
- [ ] `_network` gained analog recording + current-injecting devices + sweep + per-generator weight vectors, each TDD-covered + docstringed.
- [ ] Each `_nest` bug surfaced has a reproducing unit test + fix.
- [ ] 7 parity tests pass with `nest` / skip without; rows added to `examples/nest/README.md`'s validation table.
- [ ] >90% coverage on touched code; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the new Simulator surfaces (`res.trace`, current-device wiring, sweep API) that `03`/`11`/`12`/`14` reuse, every `_nest` fix + root cause, parity numbers.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity table) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
