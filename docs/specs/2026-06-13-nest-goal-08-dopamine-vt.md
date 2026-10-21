# /goal — 08 · `stdp_dopamine_synapse` + `volume_transmitter` (the broadcast-modulator seam)

> Self-contained prompt. Open at repo root. Stream: synapse. Depends on: 07 (merged — reuses its State-reader seam, generalised to a broadcast signal) and 05 (merged — per-edge `edge_state_init` eligibility trace). Independent of 06. This is the **last major plastic seam** before bucket-3; it does **not** add a 4th primitive — it generalises the 07 reader from a per-edge post gather to a **global modulatory scalar**.

## Session start

1. Create an isolated worktree branch **`nest-goal/08-dopamine-vt`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — Part 2.5 (the typed family of three; confirm dopamine is **not** a new primitive but a reader generalisation), and especially:
   - the **`07-voltage-coupled-clopath` Lessons** — the **post-state reader API** (`post_state_reads` → `ctx.post_states = {name:(E,)}`, the `_gather_post_states()` seam, read-only gathers) that 07 said *"08-dopamine + Urbanczik reuse verbatim"*; the `* dt` integral gotcha (NEST multiplies per-step `dw` by the resolution — **check the C++, not unit docs**); the one-step online-read lag + the **documented-band parity** posture (assert direction/ordering exact + a documented % band, not 1e-7).
   - the **`05-stdp-nearest-neighbour` Lessons** — the **per-edge `edge_state_init()`** machinery (the eligibility trace `c` lives here), the `_stdp_drive.py` harness, `res.weight_trace`, the **`*_hom`/common-param** convention, and the **phantom-pre-at-0** + `tau_minus`-on-post gotchas.
   - the **`04-stdp-core` Lessons** — the multi-trace seam and the STDP pairing kernel that the eligibility trace `c` is built from.
   Then read:
   - `docs/nest-status/internal/synapses-plasticity-gap.md`: the `stdp_dopamine_synapse` row (~60), the `volume_transmitter` row (~74), the **"Volume transmitter latency"** risk note (~122: *deliver-on-VT-update semantics, not read-VT-state-at-synapse-update-time*) — **this is the cluster's high-risk surface** — and the **P0** deliverable (~181: known dopamine-pulse pattern → NEST weight trajectory within tolerance).
   - The substrate: `brainpy_state/_network/_event_plastic.py` (`EventPlasticProj`/`VoltageCoupledPlasticProj`, `_gather_post_states`, `KernelContext`, `edge_state_init`) and `_simulator.py` (`connect(synapse=)` dispatch, `record_weight`/`weight_trace`, how nodes/devices are created).
   - The models (currently legacy/eager): `brainpy_state/_nest/stdp_dopamine_synapse.py`, `volume_transmitter.py` (+ `_test.py`).
   - NEST upstream: `models/stdp_dopamine_connection.h` — **read `process_dopa_spikes_`, `facilitate_`, `depress_`, `update_dopamine_`** (the between-consecutive-dopamine-spikes integration of `c` and `n`) — and `models/volume_transmitter.{h,cpp}` (how it buffers spikes over a deliver interval and relays the dopa-spike vector to subscribed synapses).
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the modulator-signal seam shape **and** the VT-latency fidelity decision both need sign-off.

## Goal

Rebuild `stdp_dopamine_synapse` as frozen spec + pure `update(state, ctx) -> (new_state, w_eff)`, driven by a JAX-native `volume_transmitter` node, validated against live NEST. The new capability is a **global broadcast modulatory signal** `n(t)` (dopamine concentration) that **every edge of the projection reads each step** — a superset of 07's per-edge post gather. Per-edge eligibility trace `c` (05 machinery) + STDP pairing (04 machinery) supply the rest.

## Scope + the two design questions

### A. `volume_transmitter` as a JAX-native broadcast node (`_nest/` + Simulator seam)

A VT subscribes to a presynaptic **pool of (dopaminergic) neurons** and produces a decaying dopamine concentration `n(t)` shared by all synapses bound to it. Rebuild it to maintain `n` as `brainstate` State, advanced each step from its input pool's spikes (`n` decays with `tau_n`, jumps `+1/tau_n`-style per dopa spike — map NEST's exact increment). Wire it through the Simulator: `sim.create(volume_transmitter, …)`, connect the dopaminergic pool into it, and let a plastic `connect(... synapse=stdp_dopamine_synapse, vt=<handle>)` **bind the projection to that VT**.

### B. The broadcast-modulator reader seam (`_network/`)  ← **core design question #1**

Generalise 07's State reader from a **per-edge post-neuron gather** to a **single broadcast scalar** read from a third-party node (the VT). Decide and justify (record in Lessons):
- **(a)** add `signal_reads = (('n', vt),)` (a rule/projection declaration that reads named State from a bound node and exposes it as **`ctx.signals = {name: scalar}`**, broadcast to all edges) — a clean superset of `post_state_reads`, default-empty so primitives #1/#2 are untouched; the natural home for any future neuromodulator. **(recommended)**
- **(b)** treat the VT as a degenerate "post population" of size 1 and route through `post_state_reads` — rejected unless it falls out trivially, because the broadcast (1→E) vs per-edge gather (N→E) shapes differ and conflating them muddies the reader contract.
State exactly where the read happens in `update()` and that it is **read-only**.

### C. `stdp_dopamine_synapse` spec + rule (`_nest/`)  ← **core design question #2 (VT latency)**

Per-edge eligibility trace `c` (in `edge_state_init`) updated by STDP coincidence (facilitate on post-after-pre via `post_trace`, depress on pre-after-post via `pre_trace` — the 04 pairing); weight integrated as NEST's `process_dopa_spikes_` does **between consecutive dopamine spikes**, gated by `n`. Map NEST's exact param names/defaults/units (`A_plus`,`A_minus`,`tau_plus`,`tau_c`,`tau_n`,`b`,`Wmin`,`Wmax`); keep the formula **verbatim** (FP-distinct). **The high-risk decision:** NEST integrates `c`,`n` and applies weight changes **at dopa-spike delivery times** (deliver-on-VT-update), not by sampling `n` at each synapse step. Decide:
- **(a)** reproduce the between-dopa-spike integral **online** on the fixed grid (accumulate the `c·n` contribution every step using the broadcast `n(t)` with the substrate's intrinsic one-step lag), and **document the divergence** from NEST's exact event-time integration — mirrors exactly how 07 handled `delay_u_bars` (online instantaneous read vs NEST deferred-history → assert a **documented band + exact direction/ordering**, not 1e-7). **(recommended starting point)**
- **(b)** reproduce NEST's event-time integration faithfully by triggering the weight update only on the steps where a dopa spike is delivered (closer to NEST, more machinery). Choose (b) only if (a)'s residual exceeds a defensible band on the P0 trajectory.
Record the choice, the residual, and the band in Lessons.

## Constraints (from `CONTEXT.md` Part 2 + 04/05/07 Lessons)

- brainevent-only; `_network` compute, `_nest` spec+rule; **NEST fidelity wins**, document deltas; jit/vmap/for_loop-safe; no host scalars in the hot path.
- **Reuse, don't re-add:** the 07 reader seam (generalised per B), the 05 per-edge `edge_state_init`/`_stdp_drive.py`/`res.weight_trace`, the 04 pairing kernel. Keep primitives #1/#2 intact; the `signal_reads` seam is additive + default-empty + unit-tested in isolation.
- `*_hom`-style plasticity params are **common** (SetDefaults/CopyModel), not per-connection; `tau_plus`/`tau_minus` trace constants follow the established on-spec vs on-post split — document and set identically on both sides in the parity drive (04/05/07 gotcha).
- Watch the **`* dt` resolution factor** on the weight integral (07's hardest bug) — verify against `stdp_dopamine_connection.h`, not nominal units.
- NumPy docstrings citing the NEST source lines (`process_dopa_spikes_`, VT relay); >90 % coverage (directory-path `--cov`, or 05's untraced-import + `coverage.start()`+`reload` workaround if the import C-path coredumps).

## Validation (via the 00 harness + `_stdp_drive`)

`_validation/stdp_dopamine_synapse_parity_test.py` + `volume_transmitter_parity_test.py`, `requires_nest`:
- **VT first, in isolation:** drive a known dopaminergic spike pattern into the VT and assert `n(t)` matches NEST's concentration trace (`compare_trace`) — so a synapse mismatch isn't masked by a VT mismatch (same "verify the upstream node first" discipline 07 used for Clopath voltage).
- **P0 deliverable:** 3-neuron + 1-VT topology, a **known dopamine-pulse pattern**, assert the **weight trajectory over ~5 s** vs NEST (`res.weight_trace`, `compare_trace`) within the documented band; assert **direction + ordering exact** across pulse timings (07 posture).
- **Eligibility-trace timing sweep:** vary the dopa-spike time relative to the pre/post pair (the gap-doc's "high-risk surface") and assert the sign/magnitude of the resulting `Δw` tracks NEST.

## Edge cases to enumerate + test

No dopamine spikes (pure eligibility decay, `n→0`); dopamine spike with no eligibility (`c=0`); `Wmin`/`Wmax` clamp; multiple synapses sharing one VT all see the **same** `n` (broadcast correctness); two VTs in one sim stay independent; dopa spike coincident with a pre/post pair (ordering at the step boundary); `b` (baseline dopamine) non-zero; VT bound to a sliced/multi-segment pool; `dt` invariance of the documented-band result; `vmap`/`grad` smoke.

## Definition of done

- [ ] `volume_transmitter` rebuilt as a JAX-native broadcast node maintaining `n(t)` as State; created + bound via the Simulator (`connect(... vt=…)`).
- [ ] Broadcast-modulator reader seam (`signal_reads` → `ctx.signals`) added to `_network/`, additive + default-empty + unit-tested; primitives #1/#2 untouched.
- [ ] `stdp_dopamine_synapse` rebuilt as spec+rule (per-edge eligibility `c` + broadcast `n`); legacy/eager retired from the active path; old eager tests replaced.
- [ ] Both design questions resolved with sign-off (reader-seam shape; VT-latency scheme), decisions + residual band in Lessons.
- [ ] VT-`n(t)` parity verified, then dopamine weight-trajectory P0 parity passing (live NEST) / skipping (without); band + carve-outs documented.
- [ ] >90 % coverage; doctest examples; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the **`signal_reads`/`ctx.signals` broadcast-reader API** (exact declaration + ctx field — the seam any future neuromodulator reuses), the **VT-latency scheme chosen** + its residual band vs NEST, the VT `n(t)` increment/decay mapping, the dopamine parity numbers, and the remaining legacy-imperative STDP after this (none in the event/voltage family → only bucket-3 continuous + deferred e-prop remain).
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
