# /goal — 22 · `wang_decision_making` — recurrent NMDA coupling + the WTA decision network

> Self-contained prompt. Open at repo root. Stream: example (+ one `_network` seam if needed). Depends on: 14 (`iaf_bw_2001` validated; the blocker recorded), 12 (seam H `_emission_attr`), 11 (`receptor_type` routing) — all merged. Completes §3.6 (the one deferred spiking network). Retires 14's Wang placeholder.

## Session start

1. Create an isolated worktree branch **`nest-goal/22-wang-nmda-network`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`14-network-demos` Lessons** — `iaf_bw_2001` is validated **to machine precision** (single-cell AMPA+GABA + two-neuron feed-forward NMDA via the sender's `spike_offset`); the recorded blocker: *"NMDA deposits `weight·sender_spike_offset`, not `weight·spike` — a presynaptic-state-gated synapse; the generic event projection has no path to fold a per-presynaptic-neuron state into the deposit, so recurrent NMDA needs a new offset-aware event projection."* Plus the **population-rate idioms** (`fixed_indegree`+`comm='sparse'`, the Poisson kick `one_to_one` into `e[:Nstim]`, `res.rate`/`res.spikes`), the **−70 `V_m` IC trap**, the **steady-state (2nd-half) rate** observable, and **distributional-not-per-sample** for metastable nets.
   - the **`12` Lessons — seam (H)** — *"a presynaptic model declaring class attr `_emission_attr` … makes `_connect_pair` build a plain `EventProjection` that reads the pre's per-step emission holder — delivering `weight·emission`, not the binary spike."* **This may already be the offset-aware projection 14 asked for** (deposit `weight·spike_offset`) — resolve in approach-first (design question A).
   - the **`11` Lessons** — `connect(receptor_type=k)` dual-mode routing (the NMDA deposit lands in the post's `s_NMDA` receptor channel) + `receptor_input_unit`.
   - the **`fix-brunel-es-import` Lessons** — NEST-example ports import helpers `from examples.nest.<module>` (never `from brainpy`); keep a **no-NEST companion** test; watch `gh run list --branch main` after merge.
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.6 (`wang_decision_making` row).
   - `brainpy_state/_nest/iaf_bw_2001.py` — `spike_offset` (`Δs_NMDA = k₀ + k₁·s_pre`), the `s_NMDA` channel, the **voltage-dependent Mg²⁺ block** applied **post-side** (`I_NMDA = (V−E_ex)·s_NMDA / (1+[Mg]·exp(−0.062V)/3.57)`); `iaf_bw_2001_exact.py`; `_nest/_validation/iaf_bw_2001_nest_parity_test.py` (the validated feed-forward NMDA replay).
   - `brainpy_state/_network/_event_proj.py` (`EventProjection` deposit path) + `_network/_simulator.py` `_connect_pair` (seam H `_emission_attr` branch + the `receptor_type` bridge).
   - NEST upstream `pynest/examples/wang_decision_making.py` (two selective + nonselective excitatory pops + inhibitory pop; recurrent AMPA/NMDA/GABA; two coherence-modulated Poisson stimuli; the decision read-out).
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — design question A (seam H vs a new projection) is the crux; resolve it with a recurrent-NMDA micro-parity test before building the full network.

## Goal

Enable **recurrent NMDA** coupling among `iaf_bw_2001` neurons and port `wang_decision_making` on the validated neuron, with the winner-take-all decision attractor validated **distributionally** against live NEST.

## Scope + the core design question

### A. Recurrent NMDA coupling — **seam (H) or a new projection?** (resolve FIRST, record in Lessons)

14 concluded "a new offset-aware projection is needed," but seam (H) (`_emission_attr`, possibly landed concurrently) already deposits `weight·emission`. **Try seam (H) first:** declare `iaf_bw_2001._emission_attr = 'spike_offset'` and route the deposit into the post's `s_NMDA` receptor via `receptor_type`. Build a **recurrent-NMDA micro-parity test** (a small pool of `iaf_bw_2001` with recurrent NMDA only; compare `s_NMDA`/`I_NMDA`/`V_m` accumulation vs live NEST). Decide:
- **(a)** seam (H) + `receptor_type` **suffices** → no new substrate; document that 14's "new projection" was superseded by the concurrently-landed seam H. **(check this first.)**
- **(b)** seam (H) **cannot** express it (e.g. the emission-holder timing under recurrence, or the NMDA delay step, or the Mg block needs the deposit pre-block) → build the **minimal presynaptic-state-gated `EventProjection`** that folds the per-pre `spike_offset` into the CSR deposit, additive + unit-tested, primitives intact.
Pick the smaller correct option; the micro-parity test is the arbiter, not the prior Lessons text.

### B. Port `wang_decision_making` (`examples/nest/wang_decision_making.py`)

Two selective excitatory populations (A/B) + nonselective excitatory + inhibitory, recurrent **AMPA + NMDA + GABA** (NMDA via the resolved seam, AMPA/GABA via the standard event path + `receptor_type`), two coherence/bias-modulated Poisson stimuli, and the decision read-out (which selective pop crosses threshold first). Reuse 14's population-rate idioms.

## Constraints (from CONTEXT Part 2 + 11/12/14 Lessons)

- `Simulator` API only; `examples/nest/` house style; `saiunit` units. Import helpers `from examples.nest.<module>`; keep a **no-NEST companion** test.
- Network parity is **distributional** — the WTA decision is metastable/stochastic: assert the **decision behaviour** (P(choose A) vs coherence/bias is monotone; mean decision time; the attractor exists on **both** sims) in a documented band + the qualitative law, **never** per-sample spike matching (14 posture). The recurrent-NMDA **micro**-parity (precondition) may be tighter (`s_NMDA` accumulation is deterministic given identical input).
- Pin NEST `V_m` IC to brainpy's `V_initializer` for the micro-parity (the −70 trap, 14); compare steady-state where the transient differs.
- Any neuron/seam fix RED-first + a `for_loop`-lowering regression test (12); NMDA deposit math verified against `iaf_bw_2001.cpp`'s `SpikeEvent` handling.
- NumPy docstrings; >90 % coverage on touched `_nest`/`_network` code. Coverage via `coverage run --source=brainpy_state` + the coverage API (06/08/09/11/14).

## Validation (via the 00 harness)

`_nest/_validation/wang_decision_making_test.py` + a recurrent-NMDA micro-parity test, `requires_nest`:
- **Micro-parity (precondition):** small recurrent `iaf_bw_2001` pool, `s_NMDA`/`I_NMDA`/`V_m` accumulation vs NEST (tight band; deterministic given matched input + IC).
- **Decision network:** P(choose A|coherence) monotone + the WTA attractor present on both sims (seed-mean/median, documented band); mean decision time in band. No-NEST companion: imports + a short headless run produces a decision raster.

## Edge cases to enumerate + test

Zero-coherence (50/50 choice, both sims; assert the *distribution*, not the winner); strong bias (deterministic winner); the NMDA saturation regime; recurrent pool of size 1 (self-coupling / no autapse per the network spec); AMPA+NMDA+GABA all routed to distinct receptors without cross-talk; `s_NMDA` deposit timing (one delay step, matching the feed-forward replay); standalone run without NEST.

## Definition of done

- [ ] Design question A resolved by the micro-parity test (seam H suffices, **or** a minimal offset-aware projection landed additive + unit-tested); decision + reasoning in Lessons.
- [ ] Recurrent-NMDA micro-parity passing (live NEST) / skipping (without).
- [ ] `wang_decision_making` ported + runnable standalone; WTA-decision distributional parity (pass / skip) + no-NEST companion; 14's Wang placeholder retired.
- [ ] >90 % coverage on touched code; README §3.6 row updated (Wang now implemented); Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the **A resolution** (whether seam H expresses recurrent presynaptic-state-gated coupling, or the new projection's API), the recurrent-NMDA wiring (`_emission_attr` + `receptor_type` → `s_NMDA`), the Wang WTA parity numbers + the decision observable/band, and confirmation that **§3.6 spiking networks are complete** (only the rate/gap/`ht_neuron` rows remain → bucket-3/`17`).
2. Verify tests pass / skip (and `gh run list --branch main` green after merge); coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
