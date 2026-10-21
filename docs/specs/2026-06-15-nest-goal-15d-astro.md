# /goal — 15d · Astrocytes: one-way SIC seam-H (`astrocyte_lr_1994` → `aeif_cond_alpha_astro`) + de-queue

> Self-contained prompt. Open at repo root. Stream: synapse-substrate + `_nest` astrocyte/neuron. **Bucket-3 child** of `15a` (rate-core); **last bucket-3 model cluster**. Adds **one** new deposit variant — the *one-way SIC current deposit* `w·I_SIC` — and closes the neuron↔astrocyte bidirectional loop. Depends on: `15a` (seam-H continuous emission + de-queue + lag template), `12`/`22` (seam H) — **run `15a` first**.

## What 15a established (build on it)
`15a` resolved seam-(H) continuous emission + the de-queue + the 1-step lag. The astrocyte emits a **graded SIC current** `I_SIC` (when intracellular Ca crosses threshold); `sic_connection` deposits `w·I_SIC` one-way into the AdEx neuron's SIC current channel — per-edge graded emission exactly like `22`'s NMDA `spike_offset` (`_emission_attr` → `weight·emission`). The neuron→astrocyte direction is the **standard delta-input** (presynaptic spikes increment astrocytic IP3). Both directions are 1-step lagged — a fully explicit bidirectional loop (the Ca/SIC dynamics are slow, so the lag is negligible vs the timescales).

## Session start
1. Create an isolated worktree branch **`nest-goal/15d-astro`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`15a` Lessons** — seam-H continuous emission (receptor-optional), the de-queue template, the 1-step-lag posture.
   - the **`22` Lessons (seam H generalized)** — `_emission_attr`(+`_emission_receptor`) deposits `weight·emission` for continuous presynaptic state (SIC is per-edge graded emission like NMDA's `spike_offset`); external drive into a multi-receptor cell **must name `receptor_type`**.
   - the **`12` Lessons (seam H)** — the emission-holder timing + the **`for_loop`-lowering regression-test discipline**.
   - the **`14` Lessons** — distributional network parity (seed-mean/median band + the qualitative law).
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.8 astro rows + `synapses-plasticity-gap.md` §5 (sic / astro notes).
   - `brainpy_state/_nest/astrocyte_lr_1994.py` (the IP3/Ca/h_IP3R Lazaridis-Rinzel dynamics; spikes → IP3 increment via `Δ_IP3·w`; SIC generation at the Ca threshold `SIC_th`), `sic_connection.py` (`VALID_SENDERS={astrocyte_lr_1994}` / `VALID_RECEIVERS={aeif_cond_alpha_astro}`; one-way), `aeif_cond_alpha_astro.py` (the AdEx dynamics + the `_sic_queue` dict event emulator to delete + the SIC current input).
   - The substrate: `brainpy_state/_network/_simulator.py` `_connect_pair` (the seam-H branch + the `receptor_type`/current bridge), `_network/_event_proj.py`.
   - NEST upstream: `models/astrocyte_lr_1994.{h,cpp}`, `models/sic_connection.h`, `models/aeif_cond_alpha_astro.cpp` (`handle(SICEvent&)`), the examples `pynest/examples/astrocytes/astrocyte_single.py` / `astrocyte_small_network.py` / `astrocyte_brunel.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — design question A (the one-way SIC deposit + de-queue) is the sign-off; resolve it with a neuron↔astrocyte micro-parity.

## Goal
Wire the bidirectional neuron↔astrocyte loop: `astrocyte_lr_1994` emits `I_SIC` (seam-H, `_emission_attr='I_SIC'`), `sic_connection` deposits `w·I_SIC` one-way into `aeif_cond_alpha_astro` (de-queued), with the neuron→astrocyte spike path via standard delta input — validated by a SIC response + a neuron-astrocyte network against live NEST.

## Scope + the core design question

### A. The one-way SIC deposit + de-queue — **resolve FIRST via micro-parity** (record in Lessons)
**Try:** declare `astrocyte_lr_1994._emission_attr='I_SIC'` (the graded SIC current it generates each step); `sic_connection` (one-way, `VALID_SENDERS`/`VALID_RECEIVERS` enforced) deposits `weight·I_SIC[n−1]` into the AdEx neuron's **SIC current** channel via seam-H. **Delete** `aeif_cond_alpha_astro._sic_queue` (+ its `_queue_add`/event machinery); the neuron reads SIC via the seam-H current deposit. The neuron→astrocyte direction stays the **standard delta-input** (spikes → `Δ_IP3·w` IP3 increment). Build a **neuron↔astrocyte micro-parity** (one `aeif_cond_alpha_astro` + one `astrocyte_lr_1994` + one `sic_connection`; drive the neuron, watch IP3/Ca rise then the SIC current return; compare Ca/IP3/I_SIC/V vs live NEST). Decide:
- **(a)** seam-H one-way into the SIC current channel **suffices** → the minimal additive variant (likely just an `_emission_attr` declaration + a SIC receptor/current target, like `22`'s NMDA-into-receptor — no new projection). **(check this first.)**
- **(b)** the SIC threshold/timing forces a dedicated path → the minimal SIC variant. **Prefer (a)**.

### B. Sender/receiver enforcement
`sic_connection` only connects `astrocyte_lr_1994` → `aeif_cond_alpha_astro` (honor `VALID_SENDERS`/`VALID_RECEIVERS`; clear error otherwise). De-queue the astrocyte too if it carries the same dict-queue pattern.

## Constraints
- seam-(H) `comm='dense'` (astro nets are small/structured); the SIC deposit is a **current** (pA), not a spike.
- Both loop directions are 1-step lagged — **document** the explicit-lag posture (consistent with 15a–c; the slow Ca/SIC dynamics make the lag negligible).
- **RED-first** + a **`for_loop`-lowering regression test** (12); `brainstate.transform.for_loop`, never a Python loop (CLAUDE.md #10).
- **`clear_caches()` + float64 guard** (the astrocyte Ca dynamics are stiff; 21 discipline).
- NumPy docstrings; >90 % coverage. Coverage via `coverage run --source=brainpy_state` + the coverage API.

## Validation (via the 00 harness)
`_nest/_validation/astrocyte_sic_test.py` + the micro-parity, `requires_nest`:
- **SIC-response micro-parity (precondition):** one astrocyte driven by a spike train → IP3/Ca/I_SIC waveform vs NEST (deterministic given input); the SIC threshold-crossing timing.
- **Neuron↔astrocyte loop:** one `aeif_cond_alpha_astro` + one `astrocyte_lr_1994` + `sic_connection` (both directions); the SIC-modulated V/firing vs NEST (tight where deterministic).
- **Astrocyte network:** the `astrocyte_small_network`/`astrocyte_brunel` topology — assert the SIC-modulated population firing distributionally (seed-mean/median, documented band; 14 posture).
- No-NEST companion: imports + a headless neuron-astrocyte run shows the SIC modulation.

## Edge cases to enumerate + test
Sub-threshold Ca (no SIC emitted → `I_SIC=0`); a strong spike drive (Ca crosses threshold → SIC); `sic_connection` with an invalid sender/receiver (errors); the neuron→astro delta path (IP3 increment) without the return SIC; `weight=0` SIC (decoupled); the bidirectional loop's stability (slow Ca dynamics — no runaway); `clear_caches()`/x64; standalone without NEST; `for_loop` lowering of the loop.

## Definition of done
- [ ] Design question A resolved by the micro-parity (seam-H one-way SIC into the current channel suffices, **or** the minimal SIC variant); the de-queue + the explicit-lag-loop posture recorded in Lessons.
- [ ] `astrocyte_lr_1994` emits `I_SIC`; `sic_connection` one-way (sender/receiver enforced); `aeif_cond_alpha_astro` de-queued (`_sic_queue` deleted).
- [ ] SIC-response + neuron↔astro-loop + astro-network parity passing (live NEST) / skipping; `for_loop`-lowering test; `clear_caches()`+x64.
- [ ] >90 % coverage; examples-gap §3.8 astro rows + README updated; Lessons entry in `CONTEXT.md`; **§3.8 ready for the `17b` demos**; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the **A resolution** (the one-way `w·I_SIC` current deposit via seam-H; the `aeif_cond_alpha_astro` de-queue), the sender/receiver enforcement, the bidirectional-loop explicit-lag posture, and confirmation that **bucket-3 substrate is complete** (rate/gap/siegert+diffusion/astro all on seam-H + explicit lag; no new primitive) and **§3.8 astro demos (`17b`) are unblocked**.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
