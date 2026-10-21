# /goal — 12 · Single-neuron model demos §3.5 (STP-LIF / adaptive-threshold / multi-compartment / mean-field)

> Self-contained prompt. Open at repo root. Stream: example. Depends on: 02 (Simulator seams) + 11 (the §3.5 pt-1 seams: `connect(receptor_type=k)`, callable recordable aliases, `parrot_neuron`/`_relays_multiplicity`) — all merged. Independent of 14. Finishes the §3.5 single-neuron table.

## Session start

1. Create an isolated worktree branch **`nest-goal/12-single-neuron-models-2`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`11-single-neuron-models-1` Lessons** — the seams you reuse verbatim: callable `_RECORDABLE_ALIAS` entries (`_read_recordable`, add per-model recordables **here**, not in the neuron), `connect(receptor_type=k)` + `receptor_input_unit`, `parrot_neuron`+`_relays_multiplicity` (count-faithful Poisson relay, unit-gated at connect), the **one-step recorder align** (`drop bp[0]`, `align_steps=1`), and the key discipline: **linear filters of the fixed input** (`I_syn`, adaptive currents) match **full-trace**; compare `V_m` only **sub-threshold** + spiking by **exact count** (the `glif_psc` `y2_old` vs `y2` one-step `CAT_B_ALIGNED` pattern).
   - the **`02` Lessons** (analog `res.trace`/`res.times`, current devices, **sweep-by-rebuild**, `testiaf`/`if_curve` templates) and **`00`** (`compare_trace`/`CAT_*`; `hh_phaseplane`'s **analysis carve-out** — the model for `CampbellSiegert`).
   - the **`13` Lessons** for the STP background `iaf_tum_2000` shares with tsodyks (integrated short-term depression/facilitation), and the `PYTHONPATH=. python examples/nest/<f>.py` standalone-run convention.
   Then read:
   - `docs/nest-status/internal/examples-gap.md` **§3.5** (the remaining rows: `iaf_tum_2000_short_term_{depression,facilitation}`, `mc_neuron`, `BrodyHopfield`, `CampbellSiegert`) + the §3.5 demos my index also assigns here: **`izhikevich`**, **MAT** (`mat2_psc_exp`/`amat2_psc_exp`), and the **deferred population-GIF** (`gif_pop_psc_exp`/`gif_population`, 11's "needs a population-density update" tail).
   - The models (all present): `brainpy_state/_nest/{iaf_tum_2000,mat2_psc_exp,amat2_psc_exp,izhikevich,iaf_cond_alpha_mc,iaf_psc_alpha,gif_pop_psc_exp}.py`.
   - The house style: `examples/nest/{testiaf,if_curve,one_neuron}.py` + `../../README.md` §3.5 section (append rows).
   - NEST upstream scripts: `iaf_tum_2000_short_term_depression.py`, `…_facilitation.py`, `mc_neuron.py`, `BrodyHopfield.py`, `CampbellSiegert.py`, plus the `izhikevich`/MAT example scripts.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — confirm the demo grouping, the mean-field/analysis carve-outs, and the population-GIF defer-or-port call.

## Goal

Port the remaining §3.5 single-neuron demos to `examples/nest/<name>.py` on the `Simulator` API, each with a live-NEST trace-parity test (or a documented analysis/mean-field carve-out where the demo is analytic), driving any model fix back into `_nest/<model>.py`.

## Scope — grouped (land coherent groups; split the tail, 09/11 pattern)

**Group 1 — integrated-STP LIF (rides 02; STP is *inside* the neuron, no plastic synapse):**
- `iaf_tum_2000_short_term_depression`, `iaf_tum_2000_short_term_facilitation` → spike-train drive of `iaf_tum_2000`; `V_m`/PSC parity across the two regimes (`compare_trace`, `CAT_B`). Mirror 13's tsodyks regimes but the dynamics live in the neuron.

**Group 2 — adaptive-threshold + nonlinear (rides 02 + 11 recordable aliases):**
- `izhikevich` → current-drive across the canonical firing regimes (RS/IB/CH/FS/…); `V_m`/`U_m` trace parity sub-threshold + **exact spike count** per regime (`izhikevich` has a reset nonlinearity → use the 11 discipline). NEST's `izhikevich` has a documented Euler-step convention — match it.
- **MAT** (`mat2_psc_exp`, and `amat2_psc_exp` if the script covers it) → multi-timescale adaptive threshold; expose the threshold components as recordable aliases (11 pattern); parity on the adaptive threshold + spike count.

**Group 3 — multi-compartment (experimental model; coordinate with the Urbanczik need):**
- `mc_neuron` → `iaf_cond_alpha_mc` multi-compartment demo: current into distal/proximal/soma compartments, per-compartment `V_m` recording. The model is flagged **experimental** — verify it under the Simulator first; if a per-compartment recordable/seam is missing, add it (this **dendritic-compartment reader** is exactly what the deferred **Urbanczik cluster** needs — note the overlap explicitly). If the model can't reach parity this session, ship a **documented skipped placeholder** + the precise blocker (don't force it).

**Group 4 — mean-field cross-checks (analysis carve-out, like `hh_phaseplane`):**
- `CampbellSiegert` → compare the simulated firing rate/`V_m` statistics against the **Campbell / Siegert analytic prediction** (and NEST). This is an analytic cross-check, not a per-sample trace port — document the carve-out; assert the simulation matches the closed form within its statistical band.
- `BrodyHopfield` → sinusoidally-driven iaf population spike-coding (phase-locking). Distributional/phase parity (`compare_distributional`).

**Splittable tail:** `gif_pop_psc_exp` / `gif_population` — population-density GIF needs a genuine **population-density update** (not the single-neuron harness). Port **only if** the model already implements the density dynamics; otherwise leave a documented defer note (a dedicated population/mean-field cluster) rather than half-porting.

## Constraints

- `Simulator` API only; `examples/nest/` house style; `saiunit` units.
- Model bugs fixed in `_nest/<model>.py` (reproduce with a unit test first, `superpowers:systematic-debugging`); recordables added as **callable aliases** in `_read_recordable` (11 rule), never bloating the neuron.
- Reuse 11's compare discipline: full-trace on linear-filter recordables (`I_syn`, threshold, adaptive currents), `V_m` sub-threshold + **exact spike count** for any model with a reset nonlinearity (`izhikevich`, MAT, iaf_tum_2000). Recorder one-step offset → `CAT_B_ALIGNED`; NEST recorder `interval = dt`.
- Analysis/mean-field demos (`CampbellSiegert`, `BrodyHopfield`) document their carve-out (analytic/distributional, not per-sample trace) per `hh_phaseplane`'s precedent.
- NumPy docstrings; >90 % coverage on touched `_nest`/`_network` code. Coverage via `coverage run --source=brainpy_state` + the coverage API (06/08/09/11 — `--cov=pkg.mod` coredumps; rtk proxy strips `--cov`).

## Validation (via the 00 harness)

`_nest/_validation/<name>_test.py`, `requires_nest`, per demo: trace parity (`compare_trace`, `CAT_B`/`CAT_B_ALIGNED`) for Groups 1–2; per-compartment `V_m` for `mc_neuron` (or a skip placeholder); analytic cross-check for `CampbellSiegert` (NEST-free where the formula suffices); distributional/phase for `BrodyHopfield`. Each example runs standalone (`PYTHONPATH=. python …`) without NEST (parity test skips). Append a README §3.5 row per demo.

## Edge cases to enumerate + test

iaf_tum_2000: depression vs facilitation regime params; first spike (no prior adaptation). izhikevich: each firing regime; the reset/`U_m` jump; the Euler convention boundary. MAT: threshold recovery between spikes; multi-timescale superposition. mc_neuron: current into each compartment routes correctly; per-compartment recording order. CampbellSiegert: rate at the analytic limit; sub/supra-threshold mean drive. All: `dt` invariance; standalone run without NEST.

## Definition of done

- [ ] §3.5 remainder ported + runnable standalone (iaf_tum_2000 ×2, izhikevich, MAT, BrodyHopfield, CampbellSiegert); `mc_neuron` ported or documented-skip; population-GIF ported or explicitly deferred.
- [ ] Each demo has a live-NEST parity test (pass / skip) or a documented analysis/mean-field carve-out.
- [ ] Any model fix + recordable alias TDD-covered; README §3.5 completed; >90 % coverage on touched code.
- [ ] Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the integrated-STP vs plastic-synapse distinction for `iaf_tum_2000`, the reset-nonlinearity compare recipe for `izhikevich`/MAT, the `mc_neuron` per-compartment recorder seam (and **exactly how it does/doesn't unblock Urbanczik**), the mean-field analysis-carve-out pattern, the population-GIF disposition, and confirmation that §3.5 is complete (or its precise remainder).
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity table) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
