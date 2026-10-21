# /goal — 09 · `weight_recorder` hookup audit + STP-family live-NEST validation

> Self-contained prompt. Open at repo root. Stream: synapse. Depends on: 04, 05, 08 (all merged). Independent of 10. Closes the synapses-plasticity-gap.md **§6 validation gaps** — the send-triggered weight recorder across every plastic rule, plus the 4 STP modules that still lack a live-NEST test. No new neuron/synapse *models*: this is a validation-and-wiring cluster.

## Session start

1. Create an isolated worktree branch **`nest-goal/09-weight-recorder-audit`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — Part 2 (architecture, no relitigation), and especially:
   - the **`08-dopamine-vt` Lessons** — the weight-recording reality you must honour: *"NEST's `weight_recorder` logs **only at pre `send`**"*, so a delayed change after the last send is invisible → read the final `GetConnections` weight; and `e.set_weight(weight_)` is logged **before** the impulse's facilitate/depress. This is the exact send-time semantics 09 audits.
   - the **`04-stdp-core` Lessons** — `res.weight_trace(proj)` / `Simulator.record_weight` (the per-step State-tap weight trajectory) and `_validation/_stdp_drive.py` (the strong-driven post, dendritic-delay convention, record-actual-fire-steps-and-replay harness) — 09's recorder is the **send-event view** of this same weight signal.
   - the **`01-event-plastic-substrate` Lessons** — the STP family (`tsodyks_synapse`,`_hom`,`tsodyks2_synapse`,`quantal_stp_synapse`) rebuilt as spec+rule (these are the 4 modules needing the live-NEST backfill); and the **`06`/`08` distributional** patterns (`compare_distributional`, seed-mean CAT_D) for `quantal_stp`.
   Then read:
   - `docs/nest-status/internal/synapses-plasticity-gap.md` **§6 Validation gaps** (the 5 no-`import nest` modules — `volume_transmitter` already done in 08, leaving the 4 STP) and **§7 roadmap P0 "Audit weight-recorder hookup per STDP variant"** + **P0 "Validate the STP family"**; and `devices-gap.md` line ~75 (`weight_recorder` row: *"requires hook from plastic synapse; verify per-rule wiring"*).
   - `brainpy_state/_nest/weight_recorder.py` (+ `_test.py`) — currently `divergent`; decide rebuild-vs-thin-view (below).
   - `brainpy_state/_network/_simulator.py` — `record_weight`/`weight_trace` (lines ~228, ~416), the projection update loop, the `_SpikeHolder`/pre-spike signal (the send mask).
   - NEST upstream: `models/weight_recorder.{h,cpp}` and a plastic `*_connection.h` `send()` to see exactly **when** a weight event is emitted (per send, value = weight at send-time **before** the post-pairing update).
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the recorder seam shape (thin view vs device) is the decision to sign off.

## Goal

Give every plastic synapse a documented **live-NEST validation**: (A) a send-triggered `weight_recorder` whose emitted **event count + timing + value** match NEST across the whole STDP family, and (B) the 4 remaining STP rules a live-NEST parity test — closing §6.

## Scope + the core design question

### A. `weight_recorder` send-event audit (`_network` seam + `_nest` wiring)  ← **core design question**

NEST's `weight_recorder` records one event **per plastic `send`** (i.e. per pre-spike on that connection), logging `(sender, target, weight, t)` with the weight **as of send-time, before** that spike's pairing update. The repo already has the **dense per-step** `res.weight_trace` (State-tap). Decide and justify (record in Lessons):
- **(a) thin send-view over `weight_trace`** — mask the per-step weight trajectory to the steps where each edge's **pre fires** (the send mask is already available from the `_SpikeHolder`/pre vector), emitting the `(edge, t, w)` event series. No new device; reuses 04/08 machinery. **(recommended** if the send mask aligns one-step-cleanly with the logged weight — verify the **before-vs-after-update** ordering matches NEST, the 08 `set_weight`-before-impulse gotcha.**)**
- **(b) dedicated recorder device** — a `weight_recorder` node that taps the projection on send. Choose only if the value-at-send-time ordering can't be expressed as a clean mask over `weight_trace`.
Then **audit across the full plastic family**: `stdp_synapse`,`_hom`,`stdp_pl_synapse_hom`,`stdp_triplet_synapse`,`vogels_sprekeler_synapse`,`jonke_synapse`,`ht_synapse`,`stdp_nn_{symm,restr,pre_centered}_synapse`,`stdp_facetshw_synapse_hom`,`clopath_synapse`,`stdp_dopamine_synapse`. Per rule: attach the recorder both sides, assert **emitted event count** (= number of sends) and **timing** (= pre-spike steps) exact, and **value** within the rule's already-established parity band (machine-precision for the NN/core families; the documented 5 %/0.2 % bands for clopath/dopamine).

### B. STP-family live-NEST backfill

`tsodyks_synapse`, `tsodyks_synapse_hom`, `tsodyks2_synapse`, `quantal_stp_synapse` currently have **no `import nest`** test. Add a `_validation/*_parity_test.py` each, `requires_nest`: PSC-amplitude train vs NEST (`compare_trace`; deterministic STP → `CAT_A`/`CAT_B`), and `quantal_stp` distributional (`compare_distributional`, seed-mean CAT_D, per 06/08). Reuse `_stdp_drive.py`'s drive where it fits.

> **Splittable tail:** if the family-wide audit (A) balloons past one session, land A first and split B (the 4 STP tests) into a `09b` — note it in Lessons rather than half-finishing both.

## Constraints (from `CONTEXT.md` Part 2 + 04/06/08 Lessons)

- brainevent-only; `_network` compute, `_nest` spec+wiring; **NEST fidelity wins**, document deltas; jit/vmap/for_loop-safe; no host scalars in the hot path.
- **Reuse, don't rebuild:** the recorder rides `weight_trace`/`record_weight` + the `_SpikeHolder` send mask; the parity tests ride the 00 harness + `_stdp_drive.py`. Any seam added is additive + unit-tested.
- Honour 08's recorder semantics: log **only at send**; a change after the last send is read from the final weight, not the recorder; value is **pre-pairing-update**. Mismatching the before/after ordering is the most likely bug — write the RED test for it first (`superpowers:test-driven-development`).
- `*_hom` params common, per-edge weights per-connection (04/05); set identical values across neuron/synapse/transmitter in the drives.
- NumPy docstrings; >90 % coverage (use `coverage run --source=brainpy_state -m pytest` + the coverage API / `coverage json`, per 06/08 — the `--cov=pkg.mod` form coredumps and the rtk proxy strips `--cov`).

## Validation (via the 00 harness + `_stdp_drive`)

`_validation/weight_recorder_audit_test.py` (parametrised over the plastic family) + the 4 `_validation/<stp>_parity_test.py`, `requires_nest`:
- recorder: per rule, **event count** + **timing** exact vs NEST's `weight_recorder`; **value** within the rule's band. Empty train → zero events. A change after the last send → absent from the recorder, present in the final weight (assert both).
- STP: PSC-amplitude train parity; `quantal_stp` distributional.

## Edge cases to enumerate + test

No pre spikes (zero recorded events); a single send; weight change strictly **between** the last send and sim end (recorder misses it — that's correct, assert it); send coincident with the post pairing (before/after ordering); multapses (each edge records independently in CSR order); a rule whose weight is clamped at `Wmax` (recorded value clamped); `dt` invariance of event timing; `vmap`/`grad` smoke on the masked-view path.

## Definition of done

- [ ] `weight_recorder` send-event seam (thin view or device) landed, additive + unit-tested; ordering matches NEST (RED test first).
- [ ] Family-wide audit: event count + timing exact, value within each rule's band, across all ~13 plastic rules (live NEST) / skipping (without).
- [ ] 4 STP rules gain live-NEST parity tests (`quantal_stp` distributional); §6's no-`import nest` gap closed.
- [ ] >90 % coverage; doctest examples; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the recorder seam chosen (thin-view vs device) and its **send-time / before-vs-after-update** semantics, the per-rule event count/timing/value parity table, the STP-family parity numbers, whether B was split to `09b`, and confirmation that §6's validation gaps are closed (only the docs cluster 10 + bucket-3 remain on the discrete-synapse side).
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity table) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
