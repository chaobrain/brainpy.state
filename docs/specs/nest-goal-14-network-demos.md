# /goal — 14 · Network demos §3.6 (the spiking networks; rate/gap/ht defer to bucket-3)

> Self-contained prompt. Open at repo root. Stream: example. Depends on: 02/03 (Simulator + recording seams) — merged; the Brunel ports (PRs #39–#41) are the house pattern. Independent of 12. Ports the §3.6 **spiking** network demos; the **rate / gap-junction / `ht_neuron`** rows are explicitly out of scope → bucket-3 cluster `17`.

## Session start

1. Create an isolated worktree branch **`nest-goal/14-network-demos`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`02`/`03` Lessons** (Simulator network construction, devices, `spike_recorder`/analog recording, `res.rate`/`res.n_events`, the eager-detector "bypass don't gut" rule) and **`00`** (the **distributional** comparison vocabulary — network parity is rates/CV/correlations, not per-sample traces: `compare_distributional`, the **5 % rate band** the Brunel tests use, `CAT_D`).
   - the **`11`/`13` Lessons** for any neuron/synapse seam a network needs (`connect(receptor_type=k)` for multi-receptor cells, weight-matrix recording, `RELAY_D=0.1` device→synapse lag).
   - the **`fix-brunel-es-import` Lessons** — the house rule: NEST-example ports import helpers `from examples.nest.<module>`, **never** `from brainpy`; keep a **no-NEST companion** test alongside any `@skipUnless(nest)` one so CI exercises the importable surface; watch `gh run list --branch main` after merge.
   Then read:
   - `docs/nest-status/internal/examples-gap.md` **§3.6** — port the **spiking** rows (`EI_clustered_network/`, `brette_et_al_2007/`, `wang_decision_making`, `artificial_synchrony`, `repeated_stimulation`, `sensitivity_to_perturbation`); the **`lin_rate_ipn_network` / `rate_neuron_dm` / `gap_junctions_*` / `intrinsic_currents_*` rows are OUT of scope** (rate neurons + gap junctions + `ht_neuron` ⇒ bucket-3 / cluster `17`) — note the deferral, do not port them here.
   - The house pattern: `examples/nest/{brunel_alpha,brunel_delta,brunel_exp_multisynapse}.py` (network construction + parity-test structure) and the conceptually-adjacent brainpy-native refs the gap-doc cites (`examples/106_COBA_HH_2007.py`, the `110-113_*` Susin-Destexhe gamma series).
   - The models: `brainpy_state/_nest/{iaf_psc_exp,iaf_psc_delta,iaf_cond_exp,iaf_cond_alpha,iaf_bw_2001,iaf_bw_2001_exact}.py`.
   - NEST upstream scripts/subdirs: `EI_clustered_network/`, `brette_et_al_2007/`, `wang_decision_making.py`, `artificial_synchrony.py`, `repeated_stimulation.py`, `sensitivity_to_perturbation.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the subdir-demo scope (port the core network + its analysis, not every helper file) and the `iaf_bw_2001` validation precondition need sign-off.

## Goal

Port the §3.6 **spiking** network demos to `examples/nest/<name>.py` (or a subpackage mirroring the NEST subdir) on the `Simulator` API, each with a live-NEST **distributional** parity test (population rate / CV / correlation within a documented band), driving any neuron/seam fix back into `_nest`/`_network`.

## Scope — grouped (land coherent groups; split the tail, 09/11 pattern)

**Group 1 — classic E/I spiking networks (ride the Brunel substrate directly):**
- `artificial_synchrony`, `repeated_stimulation`, `sensitivity_to_perturbation` → iaf E/I networks with structured stimulation; parity on population rate + the demo's headline statistic (synchrony index / evoked response / perturbation divergence) within a documented band.
- `EI_clustered_network/` → clustered E/I network (Rostami/Schmidt). Port the core network + its rate/CV analysis (not every plotting helper). Distributional parity on per-cluster rates.

**Group 2 — benchmark family:**
- `brette_et_al_2007/` → the Brette benchmark networks (COBA/CUBA variants). Port the representative member(s); cross-link the existing `106_COBA_HH_2007` overlap. Rate-band parity.

**Group 3 — NMDA decision network (validate the neuron first):**
- `wang_decision_making` → classic Wang 2002 NMDA network on `iaf_bw_2001`. **`iaf_bw_2001` is currently unvalidated** — verify the **neuron** against NEST first (single-cell `V_m`/NMDA-conductance trace parity, reusing 11's discipline; `iaf_bw_2001_exact` is the reference variant) **before** the network, so a network mismatch isn't masked by a neuron mismatch. If the neuron can't reach parity this session, ship the network demo as a **documented skipped placeholder** + the neuron blocker (don't force the network on an unvalidated cell). This is **Group 3 = the splittable tail.**

**Out of scope (defer to bucket-3 / `17`, note in README + Lessons):** `lin_rate_ipn_network`, `rate_neuron_dm` (rate neurons), `gap_junctions_two_neurons`, `gap_junctions_inhibitory_network` (gap junctions), `intrinsic_currents_spiking`, `intrinsic_currents_subthreshold` (`ht_neuron`).

## Constraints

- `Simulator` API only; `examples/nest/` house style; `saiunit` units. Subdir demos → a tidy `examples/nest/<name>/` or a single consolidated script (decide in approach-first); import shared helpers `from examples.nest.<module>` (never `from brainpy`).
- Network parity is **distributional**: population rate, CV(ISI), pairwise/cross-correlation, synchrony — `compare_distributional` + the documented **5 % rate band** (Brunel precedent), **not** per-sample `V_m`. State each test's statistic + band; PRNG divergence is expected (NEST per-thread RNG vs JAX) → distributional only.
- Any neuron/seam bug fixed in `_nest`/`_network` (reproduce first, `superpowers:systematic-debugging`); `iaf_bw_2001` validated as a **neuron** before its network.
- Keep a **no-NEST companion** check (import + short headless run) beside every `@skipUnless(nest)` network test, per `fix-brunel-es-import` — CI has no NEST.
- NumPy docstrings; >90 % coverage on touched `_nest`/`_network` code (examples exercised by their tests). Coverage via `coverage run --source=brainpy_state` + the coverage API (06/08/09/11).

## Validation (via the 00 harness)

`_nest/_validation/<name>_test.py`, `requires_nest`, per demo: the network's headline statistic (rate / CV / synchrony / correlation) within its documented band vs a matched-seed NEST run; `wang_decision_making` additionally asserts the **neuron** trace parity precondition (or a skip). Companion no-NEST test asserts the demo imports + runs a short window. README §3.6 rows + the explicit bucket-3 deferral note.

## Edge cases to enumerate + test

E/I balance (inhibition-dominated → asynchronous irregular; assert the regime, not exact spikes); empty/zero external drive; a single perturbed spike diverging the network (`sensitivity_to_perturbation` — assert divergence growth, chaotic so distributional only); clustered network with 1 cluster ≡ homogeneous; `iaf_bw_2001` NMDA saturation; matched-seed determinism of *our* run (reproducible under the Simulator). Standalone run without NEST (parity skips, demo still produces its raster/plot).

## Definition of done

- [ ] Group 1 + Group 2 spiking-network demos ported + runnable standalone, each with a distributional live-NEST parity test (pass / skip) + a no-NEST companion.
- [ ] `wang_decision_making`: `iaf_bw_2001` neuron-parity verified then the network ported, **or** a documented skipped placeholder + the neuron blocker.
- [ ] The rate/gap/`ht_neuron` §3.6 rows explicitly deferred to bucket-3/`17` in README + Lessons (not silently dropped).
- [ ] >90 % coverage on touched code; README §3.6 updated; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the network-demo parity recipe (which statistic + band per demo), the subdir-demo layout chosen, the `iaf_bw_2001` validation outcome (parity numbers or the precise blocker), any Simulator network-construction seam added, and the exact §3.6 remainder routed to bucket-3/`17`.
2. Verify tests pass / skip (and `gh run list --branch main` is green after merge); coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity table) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
