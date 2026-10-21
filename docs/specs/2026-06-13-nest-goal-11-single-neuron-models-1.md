# /goal — 11 · Single-neuron model demos §3.5 (HH / AdEx-multisynapse / GIF / GLIF)

> Self-contained prompt. Open at repo root. Stream: example. Depends on: 02 (merged — the Simulator analog-recording + current-device seams). Independent of 13. Exercises the §3.5 single-neuron models as parity demos; its one new substrate seam is **multi-receptor `connect(receptor_type=…)`**.

## Session start

1. Create an isolated worktree branch **`nest-goal/11-single-neuron-models-1`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`02-single-neuron-demos` Lessons** (the 4 reusable seams: analog `res.trace`/`res.times`, current-injecting devices `I_e`/`noise_generator`/`dc_`/`step_current_generator`, **sweep-by-rebuild**, multi-channel generators; the `if_curve`/`testiaf`/`balancedneuron` patterns; the recorder one-step offset `CAT_B_ALIGNED`),
   - the **`00` Lessons** (`compare_trace`, `CAT_*`),
   - the **`03` Lessons** (eager-device "bypass don't gut" rule; `getattr(pop, recordable)` alias map — add aliases if a recordable is missing).
   Then read:
   - `docs/nest-status/internal/examples-gap.md` **§3.5** (the rows below) and `docs/nest-status/internal/synapses-plasticity-gap.md` **§4 "Multisynapse ports"** (*receptor weighting + port-ID is an NEST `Connect`-time `receptor_type` convention — coordinate with `network-api-gap.md`*) — **this is 11's key seam.**
   - The existing multi-receptor precedent already in the repo: `examples/nest/brunel_exp_multisynapse.py` (the cluster-that-landed-#41 port) + how it wires receptors — **reuse/extend that path, don't reinvent.**
   - The models (all present): `brainpy_state/_nest/{hh_psc_alpha,aeif_cond_beta_multisynapse,gif_cond_exp,gif_cond_exp_multisynapse,gif_psc_exp,glif_cond,glif_psc,glif_psc_double_alpha}.py`.
   - The NEST upstream scripts: `hh_psc_alpha.py`, `hh_phaseplane.py`, `aeif_cond_beta_multisynapse.py`, `gif_cond_exp_multisynapse.py`, `gif_pop_psc_exp.py`, `glif_cond_neuron.py`, `glif_psc_neuron.py`, `glif_psc_double_alpha_neuron.py`.
   - The house style: `examples/nest/{testiaf,if_curve,balancedneuron}.py` + `../../README.md`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the `receptor_type` seam shape (below) needs sign-off, and confirm the demo grouping / splittable tail.

## Goal

Port the §3.5 single-neuron model demos to `examples/nest/<name>.py` on the `Simulator` API, each with a live-NEST trace-parity test, driving any model fixes back into `_nest/<model>.py`. The cross-cutting deliverable is a **multi-receptor connection seam** so the `*_multisynapse` / multi-receptor AdEx + GIF demos can route inputs to distinct receptor ports as NEST does.

## Scope — grouped (land coherent groups; split the tail if it balloons, per 09's pattern)

**Group 1 — HH (no new seam; ride 02 directly):**
- `hh_psc_alpha` → step/dc-current drive of `hh_psc_alpha`; `V_m` (+ gating vars if recordable) trace parity (`compare_trace`, `CAT_B`). The canonical HH single-neuron demo.
- `hh_phaseplane` → **pedagogical phase-plane** (V–n nullclines + a trajectory). This is an *analysis* script, not a spike-train parity demo: validate by reproducing the trajectory/nullcline values (a pure-`brainpy.state` numeric check, or `CAT_B` on the V(t),n(t) trajectory vs NEST if NEST can emit `Act_n`). Document the analysis-vs-parity framing.

**Group 2 — multi-receptor (BUILDS THE SEAM):**
- `aeif_cond_beta_multisynapse` → AdEx with several conductance receptors (distinct `tau_rise`/`tau_decay` per port); inputs routed by `receptor_type`. **Core design question (resolve approach-first, record in Lessons):** how `sim.connect` names the target receptor port — `connect(pre, post, receptor_type=k, …)` threading to the neuron's per-receptor input seam — generalising/​reusing whatever `brunel_exp_multisynapse` established. Parity: per-receptor PSP shape + summed `V_m` (`CAT_B`).
- `gif_cond_exp_multisynapse` → same seam on the GIF multi-receptor neuron (the seam, once built for AdEx, should drop in).

**Group 3 — GIF / GLIF families (ride 02; GIF adaptation + GLIF mechanism subsets):**
- `gif_pop_psc_exp` *(population GIF — if this reads more like a network demo, it's the splittable tail → note for 12/14 rather than forcing it here)*.
- `glif_cond_neuron`, `glif_psc_neuron`, `glif_psc_double_alpha_neuron` → GLIF mechanism variants; current-drive `V_m`/threshold-dynamics trace parity (`CAT_B`). GLIF has configurable mechanisms — exercise the demo's mechanism subset and assert it.

## Constraints

- `Simulator` API only; `examples/nest/` house style; `saiunit` units.
- Model bugs fixed in `_nest/<model>.py` (reproduce with a unit test first, `superpowers:systematic-debugging`); the `receptor_type` routing lives in `_network` (`connect`) — **never** worked around in the example.
- The `receptor_type` seam is **additive** (default = single receptor, existing demos untouched) + unit-tested in isolation before the demos use it. If full Connect-side port parity needs `network-api-gap.md` work beyond this cluster's reach, implement the **minimum** the demos need and document the boundary (don't scope-creep the whole Connect API).
- Recording rides 02's `res.trace`; add a recordable alias (03 rule) if a gating var / threshold var is missing. Recorder one-step offset → `CAT_B_ALIGNED` (02 Lesson); set NEST recorder `interval = dt`.
- NumPy docstrings; >90 % coverage on touched `_nest`/`_network` code. Coverage via `coverage run --source=brainpy_state` + the coverage API (06/08/09 — `--cov=pkg.mod` coredumps; rtk proxy strips `--cov`).

## Validation (via the 00 harness)

`_validation/<name>_test.py`, `requires_nest`, per demo: `V_m` (+ model-specific recordables) trace parity (`compare_trace`, `CAT_B`/`CAT_B_ALIGNED`); multi-receptor demos additionally assert **per-receptor** PSP separation. `hh_phaseplane`: trajectory/nullcline numeric check (documented analysis carve-out). The `receptor_type` seam gets its own NEST-free unit test (routing correctness, default-port back-compat). Add a README row per demo.

## Edge cases to enumerate + test

Multi-receptor: input to port `k` affects only receptor `k`; `receptor_type` out of range (clear error); default (no `receptor_type`) ≡ existing single-receptor path byte-identical; two receptors with different kinetics summing in `V_m`. HH: strong drive (spiking) vs subthreshold; gating-var recordables. GLIF: each mechanism-subset variant; threshold/after-spike-current reset. `dt` invariance; the demo runs standalone without NEST (parity test skips, demo still plots).

## Definition of done

- [ ] HH (×2), multi-receptor (AdEx + GIF), GLIF (×3) demos ported + runnable standalone; population-GIF either ported or explicitly deferred to 12/14 with a note.
- [ ] `connect(receptor_type=…)` seam landed in `_network`, additive + default-back-compatible + unit-tested; multisynapse demos route through it.
- [ ] Each demo has a live-NEST parity test (pass / skip); `hh_phaseplane` analysis carve-out documented.
- [ ] README updated; model fixes TDD-covered; >90 % coverage on touched code; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the **`receptor_type` connect-seam API** (exact signature + how it routes to the neuron's per-receptor input — reused by 12/14 and any `*_multisynapse` neuron), the recordable aliases added, the `hh_phaseplane` analysis-demo framing, which §3.5 demos remain for 12, and any `network-api-gap.md` boundary hit.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity table) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
