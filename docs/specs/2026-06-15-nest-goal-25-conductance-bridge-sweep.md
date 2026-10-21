# /goal — 25 · Spike→conductance bridge sweep (8 self-pulling conductance neurons)

> Self-contained prompt. Open at repo root. Stream: neuron (substrate **reuse only**). **Follow-up cluster** spawned by the `17b` Lessons. **No new substrate, no new primitive** — the Simulator's multi-receptor conductance bridge already exists (`_simulator.py:1486-1494`); this cluster only *enrolls* eight more neurons into it (declare `n_receptors`+`receptor_input_unit`+a `w_by_rec` arm in `update()`) and adds the **spike-driven** conductance parity test each currently lacks. Depends on: `17b` (PR #70 — the bridge + the reference model/test). Independent of `19`/`20`/`24`.

## What 17b established + the exact gap (read first)
`17b` fixed a **latent** bug: a conductance neuron whose `update()` only `self.sum_delta_inputs(label='w_ex'/'w_in')` receives **nothing** from `sim.connect(spikes, neuron, weight=…)` — the Simulator never populates those labels without the bridge, so a presynaptic spike leaves `V_m` pinned at `E_L` with **no error** (the *"silent dead conductance path"*). 17b fixed it for `aeif_cond_alpha_astro` by routing spike weights through the **multi-receptor bridge**:
- `_simulator.py:1486-1494` — for any `Neuron` that declares **both** `n_receptors` **and** a `w_by_rec` param in `update()`, the Simulator pre-gathers per-receptor jumps and calls `m.update(w_by_rec=get_mantissa(m.sum_delta_inputs(zeros((*varshape, n_receptors)) * receptor_input_unit) / receptor_input_unit))` — **column `k-1` = `receptor_type=k`**; col 0 → `g_ex`, col 1 → `g_in`, **positive-nS** weights (the brainpy expression of NEST's weight-**sign** routing: `weight>0 → g_ex`, else `g_in`). `_simulator.py:862` auto-enrolls the same model into connect-time receptor routing. **Declaring the two is the whole hookup — no Simulator edit.**
- The reference dual-path arm (`aeif_cond_alpha_astro.py:776-788`):
  ```python
  if w_by_rec is not None:                       # Simulator bridge path
      w_by_rec = jnp.asarray(w_by_rec, dtype=dftype)
      w_ex = w_by_rec[..., 0] * u.nS
      w_in = w_by_rec[..., 1] * u.nS
  else:                                          # legacy self-pull (no-arg seam, numpy)
      w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
      w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')
  ```
  plus `receptor_input_unit = u.nS` (class attr), `self.n_receptors = 2` (in `__init__`), `def update(self, x=…, w_by_rec=None)`. **The bridge changes only the *source* of `w_ex`/`w_in`; each model's existing synaptic kinetics consume them unchanged.**

**Audit (this repo, today):** the eight siblings each have the `else` self-pull but **no `n_receptors`/`w_by_rec`** → all unbridged, all 2-channel (ex/in) conductance cells, and **none** of their tests touch `receptor_type` (grep: `receptor_type`=0 across all eight `_test.py`) → the connect→conductance path is **uncaught**. Recorded in `neurons-gap.md §4` ("Spike→conductance bridge — tracked follow-up (from 17b)").

**The eight targets** (by synaptic kinetics — the only per-model variation):
- **alpha** (`dg += (e/τ_syn)·w`): `aeif_cond_alpha`, `iaf_cond_alpha`, `iaf_chxk_2008` (+ intrinsic AHP conductance, *not* a synaptic receptor).
- **exp** (`g += w`): `aeif_cond_exp`, `iaf_cond_exp_sfa_rr`, `gif_cond_exp`, `hh_cond_exp_traub`.
- **beta** (dual-exp rise/decay): `iaf_cond_beta`.

## Session start
1. Create an isolated worktree branch **`nest-goal/25-conductance-bridge-sweep`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`17b`-astro-demos Lessons** — the bridge mechanics (`_simulator.py:1486-1496`), the **no-regression argument** (adding `n_receptors` only flips dispatch to `update(w_by_rec=…)`; since nothing previously deposited `w_ex`/`w_in`, the self-pull already returned 0, so the bridge supplies the **same 0** → byte-identical for every existing test), the **"silent dead conductance path"** trap, and the parity recipe (`VM_TOL` 1e-3 mV + `align_steps=3` → residual ~1e-6 mV after the clean 2-sample shift; `COND_TOL` 1e-3; an **excitatory** train `receptor_type=1`/NEST `+W` and an **inhibitory** train `receptor_type=2`/NEST `−W`).
   - the **`11`/`12` single-neuron Lessons** + the **`21` Lessons** — the §3.5 conductance-neuron parity posture; **`clear_caches()` + float64 (x64)** for the stiff cells (HH/GIF/chxk); `align_steps` for the synaptic-input-after-integration + multimeter offset.
   Then read:
   - `docs/nest-status/internal/neurons-gap.md §4` (the recorded follow-up + the per-model rows / validation `(Y/N)` markers).
   - **The reference (copy this):** `brainpy_state/_nest/aeif_cond_alpha_astro.py` (`:399` `receptor_input_unit`, `:461` `n_receptors=2`, `:671` `update(…, w_by_rec=None)`, `:776-788` the dual-path arm) + its parity test `brainpy_state/_nest/_validation/aeif_cond_alpha_astro_test.py` (the 4-law + 2-parity structure, tolerances, exc/inh trains).
   - The bridge: `brainpy_state/_network/_simulator.py:1486-1494` (update dispatch) + `:862` (connect-time receptor routing).
   - **The eight targets + their current tests:** `brainpy_state/_nest/{aeif_cond_alpha,aeif_cond_exp,iaf_cond_alpha,iaf_cond_beta,iaf_cond_exp_sfa_rr,iaf_chxk_2008,gif_cond_exp,hh_cond_exp_traub}.py` (find each `else` self-pull in `update()`) + their `_test.py` (the no-regression law guards).
   - NEST upstream (clone on this machine: **`/mnt/d/codes/githubs/computational_neuroscience/nest-simulator`**; the 17-Lessons `…/githubs/nest-simulator` path is machine-specific and **absent here** — `ls`-verify): `models/<neuron>.cpp` `handle(SpikeEvent&)` (the `weight>0 → spike_exc_` sign routing each bridge reproduces).
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the sign-off is **design question A (the bridge is source-only/kinetics-agnostic), resolved by a per-kinetics-class micro-parity** below.

## Goal
Enroll the eight self-pulling conductance neurons into the existing multi-receptor bridge (the `17b` dual-path pattern) so `sim.connect(spikes, neuron, receptor_type=…)` actually drives their `g_ex`/`g_in`, and add the **spike-driven** conductance parity test each lacks — closing the silent-dead-path gap across the conductance family with **no substrate or primitive change**.

## Scope + the core design question

### A. The bridge is source-only — **confirm per kinetics class via micro-parity FIRST** (record in Lessons)
**Try:** for each target, add `receptor_input_unit = u.nS` + `self.n_receptors = 2` + the `update(self, …, w_by_rec=None)` dual-path arm — the `if w_by_rec is not None: w_ex=col0·nS; w_in=col1·nS` branch in front of the **existing** self-pull `else` (verbatim from `aeif_cond_alpha_astro:776-782`), leaving each model's own kinetics (`dg += pscon·w` / `g += w` / beta rise) **untouched**. Build a **micro-parity per kinetics class** (one alpha = `aeif_cond_alpha`, one exp = `aeif_cond_exp`, the beta = `iaf_cond_beta`) **before** sweeping the rest: a deterministic `spike_generator` exc train (`receptor_type=1`) and inh train (`receptor_type=2`) → `g_ex`/`g_in`/`V_m` vs live NEST (`+W`/`−W`). Decide:
- **(a)** the source-only arm **suffices** for every kinetics class (recommended — 17b proved it for alpha; exp/beta only change *consumption*, which already lives in the `else` path) → mechanical sweep over the remaining five. **(check this first.)**
- **(b)** a kinetics class needs more than the source swap → the minimal per-class variant, documented. **Prefer (a).**

### B. The sweep + the no-regression guard
Apply the confirmed pattern to all eight; **run each model's existing `_test.py` law suite unchanged as the no-regression guard** (the 17b argument: the bridge supplies the same 0 when nothing connects → byte-identical). **Watch for newly-activated paths:** if any existing *test or demo* `sim.connect`s spikes onto one of these eight and silently relied on the dead path (e.g. asserted `V_m` stayed at `E_L`, or a network demo whose conductance input was inert), the fix will make it respond — that is the bug being fixed; update the assertion to the now-correct conductance response (note it in Lessons), don't paper over it.

## Constraints
- **Reuse only**: declare `n_receptors`/`receptor_input_unit` + the dual-path `update` arm per model — **no `_simulator.py` edit** (declaring the two auto-enrolls via `:1486` + `:862`). Keep the `else` self-pull as the no-arg numpy seam. **brainunit** units (`receptor_input_unit = u.nS`; positive-nS weights).
- Parity vs live NEST 3.9.0 (dt=0.1 ms): **tight** — `V_m` within `VM_TOL` 1e-3 mV with `align_steps=3` (the synaptic-input-after-integration + multimeter 2-sample offset; do **not** loosen atol to absorb it), `g_ex`/`g_in` within `COND_TOL` 1e-3, for both an exc train (`receptor_type=1`) and an inh train (`receptor_type=2`). Deterministic `spike_generator` drive; recorder `interval=dt`.
- **`clear_caches()` + x64** for the stiff cells (`hh_cond_exp_traub`, `gif_cond_exp`, `iaf_chxk_2008`; the `21` discipline). Multi-step runs via `brainstate.transform.for_loop` — **never a bare Python loop** (CLAUDE.md #10); any fix **reproduce-first** + a `for_loop`-lowering regression test (`12`).
- NumPy docstrings (document the new `w_by_rec`/`n_receptors`/`receptor_input_unit` on each touched model, mirroring `aeif_cond_alpha_astro`); >90 % coverage on touched code. **Coverage (corrected by 17/17b):** `coverage run -m pytest <touched test files>` then `coverage report --include=…` — **drop `--source`** (both `--source=brainpy_state` and `--cov=pkg.mod` SIGABRT inside jaxlib/absl double-init) and let the run finish fully before reading `.coverage`.

## Validation (via the 00 harness)
Per target, `brainpy_state/_nest/_validation/<neuron>_conductance_test.py`, `requires_nest` (mirror `aeif_cond_alpha_astro_test.py`'s 2 parity cases): the exc-train `g_ex`/`V_m` and the inh-train `g_in`/`V_m` vs live NEST in the `VM_TOL`/`COND_TOL` band with `align_steps=3`. **Plus** each model's existing `_nest/<neuron>_test.py` law suite re-run as the no-regression guard (must stay green). A NEST-free structural class per model: declaring `n_receptors=2` + a `sim.connect(spikes, …, receptor_type=1/2)` deposits a non-trivial `g_ex`/`g_in` (asserts the dead path is gone — runs without NEST). Update `neurons-gap.md §4` rows (the eight flip toward `validated` / the spike→conductance follow-up cleared).

## Edge cases to enumerate + test
The dead-path regression itself (pre-fix: `connect(spikes)`→`V_m≡E_L`; post-fix: responds) as an explicit guard; exc-only vs inh-only vs both; `receptor_type` out of `{1,2}` (clear error); `weight=0` (no jump); the no-arg/self-pull seam still returns 0 (no-regression); the stiff cells under x64 (no instability); convergent spikes into one receptor **sum**; `for_loop` lowering; standalone without NEST; the beta dual-exp normalization (`iaf_cond_beta`) and the alpha `e/τ_syn` scaling (`aeif_cond_alpha`/`iaf_cond_alpha`) consume `w_by_rec` identically to NEST.

## Definition of done
- [ ] Design A resolved by the per-kinetics-class micro-parity (source-only arm suffices, **or** the minimal per-class variant); recorded in Lessons.
- [ ] All eight neurons declare `n_receptors=2` + `receptor_input_unit=u.nS` + the dual-path `w_by_rec` arm; **no `_simulator.py` change**; the `else` self-pull seam retained.
- [ ] Per-neuron spike-driven conductance parity test (exc + inh trains, `VM_TOL`/`COND_TOL`, `align_steps=3`) passing/skipping; each existing `_test.py` law suite still green (no-regression); any newly-activated test/demo path updated + noted.
- [ ] Stiff cells under `clear_caches()`+x64; `for_loop`-lowering tested; >90 % coverage on touched code.
- [ ] `neurons-gap.md §4` rows + README updated (spike→conductance follow-up cleared); Lessons entry in `CONTEXT.md`; git lifecycle done. **The conductance family no longer has a silent dead spike path.**

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the **A resolution** (the bridge is source-only — declaring `n_receptors`+the dual-path arm enrolls a model with zero Simulator change; kinetics class is irrelevant to the source swap), the per-class micro-parity numbers, any newly-activated test/demo path the fix exposed, and confirmation the **spike→conductance bridge is now uniform across the conductance family** (the 17b follow-up closed). Note any neuron that needed a per-class variant (design B).
2. Verify tests pass / skip; coverage > 90 %; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + per-neuron parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
