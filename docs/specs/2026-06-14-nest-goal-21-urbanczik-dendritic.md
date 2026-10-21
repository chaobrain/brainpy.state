# /goal — 21 · `urbanczik_synapse` (dendritic prediction-error plasticity) on `VoltageCoupledPlasticProj`

> Self-contained prompt. Open at repo root. Stream: synapse. Depends on: 07 (`VoltageCoupledPlasticProj`), 04 (multi-trace seam), 12 (seam I + per-compartment recordables), 13 (the Urbanczik placeholder) — all merged. The **last plastic synapse**; retires 13's `urbanczik_synapse_example` placeholder and completes §3.3. **Not a new primitive** — primitive #2's third user (after Clopath, dopamine).

## Session start

1. Create an isolated worktree branch **`nest-goal/21-urbanczik-dendritic`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — Part 2.5 (typed family; this is primitive #2 again), and especially:
   - the **`07-voltage-coupled-clopath` Lessons** — `post_state_reads → ctx.post_states = {name:(E,)}`, the read-only per-edge gather, `connect(synapse=…)→VoltageCoupledPlasticProj` dispatch, the **`*dt` integral trap** (verify against C++, not unit docs), and the **online-read-vs-NEST-deferred-history** posture (a post-state quantity read with the substrate's intrinsic one-step lag → assert a **documented band + exact direction**, the `delay_u_bars` precedent — Urbanczik's **dendritic-delay history window** is the same shape).
   - the **`04-stdp-core` Lessons** — the **multi-trace seam** (`pre_trace_tau = (τ, …)` → per-neuron `(N,k)`, `ctx.pre_traces`): Urbanczik needs **two** presynaptic traces (membrane `τ_L`, synaptic `τ_s`).
   - the **`12-single-neuron-models-2` Lessons** — seam **(I)** + per-compartment aliases (`V_m.{s,p,d}`), and the **`for_loop`-lowering regression-test discipline** (*every* Simulator-run model needs one; eager stepping hides carry-shape bugs).
   - the **`13-plasticity-demos` Lessons** — the precise Urbanczik blocker it recorded: *"`urbanczik_synapse` still on the legacy `NESTSynapse` base; the plastic post-state reader exposes only the somatic `V` — the rule needs a named dendritic reader + a validated multi-compartment point-process post."* + `_clopath_drive`/weight-matrix recording/`RELAY_D=0.1`.
   Then read:
   - `docs/nest-status/internal/synapses-plasticity-gap.md` (the `urbanczik_synapse` + `pp_cond_exp_mc_urbanczik` rows).
   - `brainpy_state/_nest/pp_cond_exp_mc_urbanczik.py` — the post neuron: it **computes** the learning signal `δΠ(t) = (n_spikes(t) − φ(V*_W(t))·dt)·h(V*_W(t))` but currently exposes only `V_s`/`V_d`/conductances as `HiddenState` (**δΠ is not a readable State** — that's the neuron-side fix). Read how φ, h, V*_W, δΠ are computed each step.
   - `brainpy_state/_nest/urbanczik_synapse.py` (legacy rule: the two presynaptic traces `τ_L^tr`/`τ_s^tr`, the `Π_int`/`Π_exp` low-pass integrals over the dendritic-delay window, the `15·C_m·τ_s·η/(g_L(τ_L−τ_s))` prefactor, the clip) + `_network/_event_plastic.py` (`VoltageCoupledPlasticProj`, `post_state_reads`).
   - NEST upstream `models/urbanczik_synapse.{h,cpp}` (the `Pi`/integral math + dendritic-delay history loop) + `pp_cond_exp_mc_urbanczik.{h,cpp}` (the `δΠ` archive) + `pynest/examples/urbanczik_synapse_example.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the δΠ-State exposure + the dendritic-delay-window divergence posture are the decisions to sign off.

## Goal

Rebuild `urbanczik_synapse` as a frozen spec + pure `update(state, ctx) -> (new_state, w_eff)` on `VoltageCoupledPlasticProj`, reading the post-neuron's **dendritic prediction error** δΠ per edge, validated against live NEST, and port `urbanczik_synapse_example` (retiring 13's placeholder).

## Scope + the design questions

### A. Expose δΠ (and `V*_W`) as a readable State on `pp_cond_exp_mc_urbanczik`

The Urbanczik-Senn rule reads the post's **dendritic prediction error** δΠ (a derived learning signal), not a raw voltage. Surface δΠ — and the dendritic-prediction membrane `V*_W` if the rule's presynaptic-trace scaling needs it — as a per-step `HiddenState`/recordable so the post-state reader can gather it. **Add a `for_loop`-lowering regression test** for the neuron (12 discipline). If δΠ is most naturally a `ShortTermState` (recomputed each step, not carried), justify the choice.

### B. `urbanczik_synapse` spec + rule (`_nest/`)

Frozen spec + pure kernel:
- **Two presynaptic traces** via the 04 multi-trace seam: `pre_trace_tau = (τ_L, τ_s)` (membrane + synaptic), `ctx.pre_traces` supplies `τ_L^tr`, `τ_s^tr`.
- **Read δΠ** via `post_state_reads = ('delta_Pi', …)` → `ctx.post_states`.
- Reproduce NEST's `Π_int` (running integral) + `Π_exp` (low-pass at `τ_Δ`) and the weight update `w ← clip(w₀ + 15·C_m·τ_s·η/(g_L(τ_L−τ_s))·(Π_int − Π_exp), Wmin, Wmax)` **verbatim** (FP-distinct). Map NEST param names/defaults/units exactly; `C_m`/`g_L` are the **dendritic** compartment's.

### C. Dendritic-delay history window → online divergence

NEST integrates δΠ over `(t_last − d, t − d]` (dendritic delay `d`). The online reader has no analog history buffer — it reads δΠ with the substrate's **one-step lag**. Document the divergence (the Clopath `delay_u_bars` precedent); the parity drive sets `d` to the minimal grid step; assert a **documented band + exact direction/sign**, not 1e-7.

### D. Wire `connect(pre, post, synapse=urbanczik_synapse, …)` → `VoltageCoupledPlasticProj` (the `post_state_reads` dispatch already exists; confirm δΠ routing).

## Constraints (from CONTEXT Part 2 + 04/07/12/13 Lessons)

- brainevent-only; `_network` compute, `_nest` spec+rule; **NEST fidelity wins**, document deltas; jit/vmap/for_loop-safe; **no host scalars** in the hot path.
- Reuse, don't rebuild: `VoltageCoupledPlasticProj` (07), the multi-trace seam (04), `_clopath_drive`/weight-matrix recording/`RELAY_D=0.1` (13). The only substrate-adjacent change is **neuron-side** (expose δΠ) — record it in Lessons. Keep primitives #1/#2 intact.
- Watch the **`*dt` factor** on the integral terms (07's hardest trap); verify against `urbanczik_synapse.cpp`, not nominal units.
- Every Simulator-run model gets a `for_loop`-lowering regression test (12); RED-first for any neuron fix (`superpowers:test-driven-development`).
- NumPy docstrings citing the NEST source lines; >90 % coverage (`coverage run --source=brainpy_state` + the coverage API — `--cov=pkg.mod` coredumps, rtk proxy strips `--cov`).

## Validation (via the 00 harness + `_clopath_drive`)

`_nest/_validation/urbanczik_synapse_parity_test.py`, `requires_nest`:
- **Post neuron first:** δΠ / `V_d` / `V*_W` / somatic rate `φ` parity vs NEST (`compare_trace`, reuse `res.trace`) — so a synapse mismatch isn't masked by a neuron mismatch (the 07/14 "validate the upstream node first" discipline).
- **Synapse weight trajectory:** supervised pre→dendrite drive with a teacher somatic current; assert the weight trajectory vs NEST within the documented dendritic-delay band + correct sign (potentiate when somatic spikes exceed the dendritic prediction, depress when below).
- Then the **demo:** port `examples/nest/urbanczik_synapse_example.py` (the small supervised-learning task) + its parity test; **delete 13's skipped placeholder**.

## Edge cases to enumerate + test

δΠ exactly 0 (prediction matches → no weight change); `Wmax`/`Wmin` clamp; the `τ_L = τ_s` degenerate prefactor (guard the `1/(τ_L−τ_s)`); excitatory vs inhibitory trace `τ_s` selection (`tau_syn_ex` vs `tau_syn_in`); no presynaptic spikes; a sliced/multi-segment post population (gather order = CSR); `dt` invariance of the documented-band result; `for_loop`/`vmap`/`grad` smoke.

## Definition of done

- [ ] `pp_cond_exp_mc_urbanczik` exposes δΠ (+ `V*_W` if needed) as State, with a `for_loop`-lowering regression test; neuron δΠ/`V_d` parity verified.
- [ ] `urbanczik_synapse` rebuilt as spec+rule on `VoltageCoupledPlasticProj` (two pre-traces + δΠ post-read); legacy retired from the active path; old eager test replaced.
- [ ] `connect(synapse=urbanczik_synapse)` builds the voltage-coupled projection; weight-trajectory parity passing (live NEST) / skipping (without), band documented.
- [ ] `urbanczik_synapse_example` ported + runnable standalone; 13's placeholder removed; §3.3 complete.
- [ ] >90 % coverage; doctest examples; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the **derived-signal post-state reader** pattern (reading a computed learning signal δΠ rather than a raw voltage — the third `VoltageCoupledPlasticProj` user, reusable by any prediction-error rule), the neuron-side δΠ-State exposure, the dendritic-delay-window divergence + band, the parity numbers, and confirmation that the **plastic-synapse family is complete** (only bucket-3 continuous remains on the synapse side).
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
