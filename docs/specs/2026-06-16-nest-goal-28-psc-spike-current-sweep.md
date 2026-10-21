# /goal — 28 · PSC spike→current sweep (audit + bridge the current-based `label='w_ex'/'w_in'` family)

> Self-contained prompt. Open at repo root. Stream: neuron. **Audit-first follow-up to `25`.** The conductance siblings of this exact seam were found **silently dead** (spike input pinned `V_m` at rest with no error) and fixed by `17b`/`25`. This cluster does the same audit + (if confirmed) the same fix for the **current-based** family that self-pulls in **pA**. Depends on: `25` (the design-A bridge template + the "silent dead path" framing) and `17b` (the multi-receptor bridge). Independent of `26`/`27`.

## Why this exists — the `25` handoff (read first)
`25`'s closing Lessons (CONTEXT.md, verbatim): *"the conductance family now has no silent dead spike path … Still-open sibling: the **current-based** `aeif_psc_*` / `hh_psc_alpha_clopath` self-pull `label='w_ex'/'w_in'` too but take `pA`, not the `nS` conductance bridge — a related but distinct seam (no `receptor_input_unit` scaling), not touched here."*

Grounding confirms the exact scope and the smoking gun:
- **Affected family (labeled self-pull, current-based, NOT bridge-enrolled):** `aeif_psc_alpha` (`label='w_ex'/'w_in'` → `I_ex`/`I_in`, `aeif_psc_alpha.py:815`), `aeif_psc_exp` (same), `hh_psc_alpha_clopath` (→ `I_syn_ex`/`I_syn_in`, `hh_psc_alpha_clopath.py:1079`). All three: **0** hits for `n_receptors`/`w_by_rec` → the Simulator's connect-time bridge (`_simulator.py:1758`, fires only on `n_receptors` + a `w_by_rec` param) **does not enroll them.**
- **The contrast that localizes the bug:** the known-working `iaf_psc_exp` (pong/sudoku/brunel) does **not** use labeled self-pull — it uses **unlabeled, sign-split** `sum_delta_inputs(...)` (`iaf_psc_exp.py:641`, doc line 323) plus `sum_current_inputs(x, V)`. The Simulator routing populates the unlabeled/sign-split channel (that's why those demos fire). It is the **labeled `w_ex`/`w_in` channel** — the *same* mechanism the conductance neurons used before `25` — that is unproven for spike drive on current-based neurons. So the audit is sharply scoped: **does `connect(spikes, m, receptor_type=k)` reach the labeled `w_ex`/`w_in` self-pull on these three?**
- **Likely latent, like the conductance ones:** their existing `_test.py` are `I_e`/DC-driven (neurons-gap §4 rows) — exactly the blind spot `17b` named ("`I_e`-only tests don't catch the dead spike path"). The fix, if needed, is mechanical: the bridge default `runit` is **already `u.pA`** (`_simulator.py:1764` `getattr(m,'receptor_input_unit',u.pA)`), so enrolling a current-based neuron needs **no `receptor_input_unit` override** — just `n_receptors=2` + the `w_by_rec` arm.

## Session start
1. Create an isolated worktree branch **`nest-goal/28-psc-spike-current-sweep`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`25`-conductance-bridge-sweep Lessons** — **design A** (source-only / kinetics-agnostic: declaring `n_receptors` + the `w_by_rec` arm enrolls *any* neuron with **zero `_simulator.py` change**; only the *source* of `w_ex`/`w_in` swaps — blob column `k-1` vs `sum_delta_inputs(label=…)`); the dual-path template; the shared edge-case tests (weight=0, convergent scatter-add→2×, seam-zero, `receptor_type` out-of-range); the stiff-cell `clear_caches()`+x64 rule; the I_e-suite no-regression guarantee (the self-pull `else` supplies the same guaranteed 0).
   - the **`17b` Lessons** — the **multi-receptor bridge** mechanics (col `k-1` = `receptor_type=k`; col0→exc, col1→inh; NEST weight-sign routing) and the **"silent dead path"** failure mode (no error, `V_m` pinned at rest).
   Then read:
   - the three targets: `brainpy_state/_nest/{aeif_psc_alpha,aeif_psc_exp,hh_psc_alpha_clopath}.py` (the `label='w_ex'/'w_in'` self-pull blocks + their `update()` signatures + the `I_ex`/`I_in` / `I_syn_ex`/`I_syn_in` States).
   - a **shipped pA-vs-nS reference**: a `25` conductance model's dual-path arm (e.g. `aeif_cond_exp.py`) + its `_validation/<m>_conductance_test.py` (4 law + 2 live-NEST) — the template to mirror with `pA` instead of `nS`.
   - the bridge enroll site `_simulator.py:1758` and the `runit` default `:1764`.
   - `docs/nest-status/internal/neurons-gap.md` §3 rows 75/78/103 (the three you will flip) + §4 follow-up.
   - NEST upstream (clone on this machine: **`/mnt/d/codes/githubs/computational_neuroscience/nest-simulator`** — the CLAUDE.md path; the 17-Lessons `…/githubs/nest-simulator` note is machine-specific and **absent here** — `ls`-verify): the three model `.cpp`/`.h` for the `I_ex`/`I_in` (`I_syn_ex`/`I_syn_in`) PSC kinetics + the weight-sign/`receptor_type` current routing.
3. Write spec + plan under `` (gitignored). **Approach-first; wait for approval before code** — the sign-off gates on the **audit result** (§ Audit below): if dead → the bridge-enroll fix + parity scope; if alive → the documentation + the spike→current parity tests you add anyway.

## Audit first (the reproduce gate — do this before any fix)
For each of `aeif_psc_alpha`, `aeif_psc_exp`, `hh_psc_alpha_clopath`: build a `Simulator`, `connect(spike_source, m, receptor_type=1)` (excitatory), drive a spike train, run via `for_loop`, and **assert `I_ex`/`I_syn_ex` and `V_m` move** off baseline. Repeat `receptor_type=2` → `I_in`/`I_syn_in`. Two outcomes:
- **Dead (expected):** `I_ex`/`V_m` stay at rest with **no error** → the current-based "silent dead path" is confirmed. This reproduce test is the regression guard; **fix** (below).
- **Alive (document):** if the labeled self-pull *is* routed for current-based neurons (unlike conductance), record exactly **why** the seam differs, and still add the missing spike→current parity tests (the gap was untested, even if not dead).

## The fix (if dead) — `25` design A, pA variant
Mirror the `25` dual-path arm in each model's `update()`, scaling in **pA** (the bridge default — **no `receptor_input_unit` override**):
```python
def update(self, x=0. * u.pA, w_by_rec=None):
    ...
    if w_by_rec is not None:                       # Simulator bridge path
        w_by_rec = jnp.asarray(w_by_rec, dtype=dftype)
        w_ex = w_by_rec[..., 0] * u.pA
        w_in = w_by_rec[..., 1] * u.pA
    else:                                          # legacy self-pull (unchanged → guaranteed 0)
        w_ex = self.sum_delta_inputs(u.math.zeros_like(self.I_ex.value), label='w_ex')
        w_in = self.sum_delta_inputs(u.math.zeros_like(self.I_in.value), label='w_in')
```
plus `self.n_receptors = 2` (enrolls it at `_simulator.py:1758`). `hh_psc_alpha_clopath` uses `I_syn_ex`/`I_syn_in` (not `I_ex`/`I_in`) — match the model's own State names. Keep the `else` branch byte-identical so every existing `I_e` suite stays unchanged (the `25` no-regression guarantee).

## Parity (vs live NEST) — the `25` posture
- `V_m` within **`VM_TOL` (1e-3 mV + `align_steps=3`)** and `I_ex`/`I_in` (the current analog of `25`'s `g_ex`/`g_in`) within a `CUR_TOL`, for **exc** (`receptor_type=1` / NEST `+W`) and **inh** (`=2` / NEST `−W`) spike trains.
- A per-model `_validation/<m>_psc_current_test.py` (≈4 law-class + 2 live-NEST, mirroring `25`'s `<m>_conductance_test.py`). Shared bridge edge cases (weight=0, convergent scatter-add→2×, seam-zero, `receptor_type` out-of-range) fold into **one** of them (as `25` folded into `aeif_cond_exp_conductance_test.py`).
- **No-regression:** every existing `aeif_psc_alpha`/`aeif_psc_exp`/`hh_psc_alpha_clopath` `I_e` suite stays byte-identical.

## Constraints
- **Source-only / kinetics-agnostic** (design A) — touch only the three models' `update()` (+ `n_receptors`); **zero `_simulator.py` change**. If you think the substrate needs a change, re-read `25`'s "design A resolved" Lessons (it doesn't — pA is already the bridge default).
- Tests/inner source import `from brainpy_state import …` (never via `brainpy.state` from inside the package — circular import). Any example code (if a demo is added) uses public `from brainpy import state`.
- **brainunit** units (`pA`, `mV`); reproduce-first (`superpowers:systematic-debugging`); drive every multi-step rollout with `brainstate.transform.for_loop`/`scan` — **never a bare Python `for`/`while`** over the model (CLAUDE.md #10).
- **Stiff cell:** `hh_psc_alpha_clopath` (HH kinetics) needs `jax.clear_caches()` + x64 (`precision=64`) and a held-quiescent / subthreshold regime so a full AP doesn't split the stiff solvers (the `25` HH discipline); compare driven-vs-baseline if it has no stable rest.
- NumPy docstrings; >90 % coverage on touched code. **Coverage (corrected by 17/17b):** `coverage run -m pytest <touched test files>` then `coverage report --include=…` — **drop `--source`** (SIGABRT in jaxlib/absl double-init); let the run finish before reading `.coverage`. The `24`-robust variant if needed: pre-import `jax`/`jaxlib`/`brainstate` before `cov.start()` + `COVERAGE_CORE=sysmon` + `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## Edge cases to enumerate + test
The audit reproduce (dead-path regression) for each of the three; exc-only / inh-only / both; weight=0 (no current); two convergent edges onto one target (scatter-add → 2×); `receptor_type` out-of-range (graceful, not a silent miss); the `I_e`-driven legacy path unchanged (no-regression); `hh_psc_alpha_clopath` held subthreshold (no AP) + driven-vs-baseline; stiff-cell `clear_caches()`+x64; `aeif_psc_delta`/`aeif_psc_delta_clopath` confirmed **out of scope** (delta synapses inject directly — they do **not** use the labeled `w_ex`/`w_in` seam; grep-verify and record).

## Definition of done
- [ ] Audit reproduce test for all three (dead-path confirmed or aliveness documented).
- [ ] If dead: `aeif_psc_alpha`, `aeif_psc_exp`, `hh_psc_alpha_clopath` enrolled into the bridge (design A, pA — `n_receptors=2` + `w_by_rec` arm, `else` branch byte-identical); zero `_simulator.py` change.
- [ ] Per-model live-NEST `V_m` + `I_ex`/`I_in` parity (exc + inh) within the `25` bands; shared bridge edge cases folded into one test; all existing `I_e` suites byte-identical.
- [ ] >90 % coverage on touched code; `neurons-gap.md` §3 rows 75/78/103 + §4 follow-up flipped (spike→current bridged); Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the audit result (was the current-based labeled self-pull **dead** like the conductance one, or routed — and why), the design-A pA enrollment (confirming the bridge default `runit=pA` needed no override), the parity bands + any stiff-`hh_psc_alpha_clopath` regime, and confirmation the **current-based spike→current path is now live** (or the precise residual). Note `aeif_psc_delta*` are out of scope (delta-direct, no `w_ex`/`w_in` seam). State whether `neurons-gap.md` §4 is now **cleared**.
2. Verify tests pass / skip; coverage > 90 %; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (audit result + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
