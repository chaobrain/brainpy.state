# /goal — 13 · Plasticity demos (§3.3) on the Simulator API

> Self-contained prompt. Open at repo root. Stream: example. Depends on: 01 (STP), 07 (Clopath), 08 (dopamine/VT) — all merged. The user-facing payoff of the now-complete discrete-synapse stream: port the §3.3 plasticity demos that exercise the rebuilt clopath / tsodyks2 / quantal_stp models.

## Session start

1. Create an isolated worktree branch **`nest-goal/13-plasticity-demos`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — the synapse models these demos drive are implemented + validated + documented (04/05/07/08/09/10). Pull the wiring facts from:
   - **`07-voltage-coupled-clopath` Lessons** — `connect(pre, post, synapse=clopath_synapse, …)` builds `VoltageCoupledPlasticProj`; `aeif_psc_delta_clopath` is a **delta** neuron so a bare clopath weight defaults to **mV** (not pA); the spike-pairing regression already lives in `_validation/clopath_synapse_parity_test.py` + `_clopath_drive.py` (the demo is the *user-facing* version of that proof); `delay_u_bars` online-read 5 % band.
   - **`01-event-plastic-substrate` Lessons** — how `tsodyks2_synapse` / `quantal_stp_synapse` are built as spec+rule and wired via `connect(synapse=…)`; the STP PSC-amplitude semantics.
   - **`09-weight-recorder-audit` Lessons** — `res.weight_trace(proj)` + the send-view recorder (for showing weight evolution); `quantal_stp` is **distributional** (`compare_distributional`, seed-mean `CAT_D`); STP family confirmed live-validated.
   - **`02`/`03` Lessons** — the Simulator example seams these ride: analog `res.trace`/`res.times` recording, current-injecting devices, sweep-by-rebuild, `examples/nest/` house style, the `compare_trace`/`CAT_*` tolerance vocabulary.
   Then read:
   - `docs/nest-status/internal/examples-gap.md` **§3.3** (the 5-row table) + the note *"Plasticity examples need volume-transmitter parity"* (~214 — **N/A** for these 5: none use dopamine; VT parity already landed in 08, note it).
   - The house style: `examples/nest/{one_neuron,testiaf,recording_demo}.py` + `examples/nest/README.md` (the per-demo table you append to).
   - The models: `brainpy_state/_nest/{clopath_synapse,aeif_psc_delta_clopath,tsodyks2_synapse,quantal_stp_synapse}.py`.
   - The NEST upstream scripts (port their structure faithfully): `clopath_synapse_small_network.py`, `clopath_synapse_spike_pairing.py`, `evaluate_quantal_stp_synapse.py`, `evaluate_tsodyks2_synapse.py` (+ glance at `urbanczik_synapse_example.py` to scope the blocked placeholder).
3. Write spec + plan under ``. **Approach-first; wait for approval before code.**

## Goal

Port the §3.3 plasticity demos to `examples/nest/<name>.py` on the `Simulator` API, each backed by a live-NEST parity test via the 00 harness — the demos are the user-facing exercise of the rebuilt plastic synapses. Four port now; Urbanczik ships a skipped placeholder (its synapse isn't rebuilt yet).

## Scope — by feasibility

**Port now (models implemented + validated):**
- `clopath_synapse_spike_pairing` → pre/post pairing across Δt and frequency on `aeif_psc_delta_clopath` + `clopath_synapse`; show the weight change vs pairing. The parity is **already proven** in `_validation/clopath_synapse_parity_test.py` — reuse its drive/numbers; the example presents it (assert the published 5 % band + direction/ordering, not 1e-7).
- `clopath_synapse_small_network` → the small recurrent Clopath network; record the weight matrix evolution (`res.weight_trace`). Parity on the weight trajectory within the documented clopath band.
- `evaluate_tsodyks2_synapse` → `tsodyks2_synapse` PSC-amplitude train (facilitation + depression regimes); `compare_trace` `CAT_B` (deterministic STP).
- `evaluate_quantal_stp_synapse` → `quantal_stp_synapse` stochastic release; **distributional** (`compare_distributional`, seed-mean `CAT_D`, per 06/09 — bit-exact unsupported).

**Blocked → skipped placeholder (clear `skip("blocked on <X>")`):**
- `urbanczik_synapse_example` → `urbanczik_synapse` is **still on the legacy `NESTSynapse` base** (07 deferred it; it needs a `VoltageCoupledPlasticProj` rebuild with a **dendritic** post-compartment state read on `pp_cond_exp_mc_urbanczik`). Ship the placeholder; note the dependency explicitly (this is the trigger to author the deferred **Urbanczik cluster**).

## Constraints

- `Simulator` API only; `examples/nest/` house style; units in `saiunit`.
- Any model/wiring bug is fixed in `_nest/<model>.py` or the `_network` seam (reproduce with a unit test first, `superpowers:systematic-debugging`), **never** worked around in the example.
- Reuse 07/01/09 drives and bands verbatim — do **not** re-derive a divergence the Lessons already froze (if the demo and a Lesson disagree, the Lesson wins).
- The clopath demo's post is a **delta** neuron → bare weight is **mV**; set it explicitly and document (07 gotcha). STP demos show **PSC amplitude** evolution, not a stored weight.
- NumPy docstrings; >90 % coverage on any touched `_nest`/`_network` code (the examples themselves are exercised by their parity tests). Coverage via `coverage run --source=brainpy_state -m pytest` + the coverage API (per 06/08/09 — the `--cov=pkg.mod` form coredumps, the rtk proxy strips `--cov`).

## Validation (via the 00 harness)

`_validation/<name>_test.py`, `requires_nest`, one per ported demo:
- clopath spike-pairing / small-network: weight-trajectory parity within the **documented clopath band** (5 %) + direction/ordering exact; reuse `_clopath_drive`.
- tsodyks2: PSC-amplitude train `compare_trace` `CAT_B`.
- quantal_stp: `compare_distributional` seed-mean `CAT_D`.
- urbanczik: a skipped-placeholder test asserting the skip reason.
- Add a row per demo to `examples/nest/README.md`.

## Edge cases to enumerate + test

clopath: pairing at Δt=0 (the rectifier boundary); pure-LTD train (no post spikes); `Wmax`/`Wmin` clamp; the small network's recurrent weight matrix gather order (CSR). tsodyks2: facilitation vs depression regime selection (`U`,`tau_fac`,`tau_rec`); first spike (no prior facilitation). quantal_stp: release probability 0 and 1 end-points; seed independence; empty train. All: the example **runs standalone** (`python examples/nest/<name>.py`) without NEST installed (parity test skips, demo still plots/prints).

## Definition of done

- [ ] 4 plasticity demos ported + runnable standalone; Urbanczik as a documented skipped placeholder.
- [ ] Each ported demo has a live-NEST parity test (pass with NEST / skip without); tolerances reuse the frozen bands.
- [ ] README table updated; any `_nest`/`_network` fix is TDD-covered + docstringed.
- [ ] >90 % coverage on touched code; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the demo-side wiring patterns (clopath mV-weight + weight-matrix recording, STP PSC-amplitude display, quantal distributional), the precise Urbanczik blocker (what the deferred cluster must build), and confirmation that §3.3 is ported bar Urbanczik + e-prop.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity table) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
