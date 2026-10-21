# /goal — 06 · static-stochastic synapses: `bernoulli_synapse` + `cont_delay_synapse`

> Self-contained prompt. Open at repo root. Stream: synapse. Depends on: 01 (merged). Independent of 05/07/08. Finishes the non-plastic EventPlasticProj delivery family — the last two synapses whose *delivery* (not weight) diverges from NEST.

## Session start

1. Create an isolated worktree branch **`nest-goal/06-static-stochastic`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — Part 2 (architecture, no relitigation), and especially:
   - the **`01-event-plastic-substrate` Lessons** (the spec+rule contract, `EventPlasticProj`, the static rebuild),
   - the **`05-stdp-nearest-neighbour` Lessons** — they explicitly name these two as *"the non-STDP `bernoulli` / `cont_delay`"* remainder, and established the **per-edge `edge_state_init()`** idiom + the **`stochastic`/`ctx.key`** observation you will exercise here,
   - the **`03-recording-demos` Lessons** for `compare_distributional` (autocorr/CV path now implemented) — bernoulli is distributional-only.
   Then read:
   - `docs/nest-status/internal/synapses-plasticity-gap.md` rows for `bernoulli_synapse` (line ~47) and `cont_delay_synapse` (line ~49), plus the risk notes **"Continuous delays"** (~131) and **"Bernoulli synapse stochastic transmission"** (~134), and the P2 row "cont_delay sub-dt delay verification" (~217).
   - The substrate you extend: `brainpy_state/_network/_event_plastic.py` — the `KernelContext` (`key`, `pre_spike`, `dt`, `t_now`), the `stochastic` attr + `ctx.key` seam (declared in the `PlasticSynapse` Protocol, lines ~119), `edge_state_init()`, `update()`/`_scatter`, and **how `InputDelay`/the delay grid is applied** (the cont-delay decision hinges on this).
   - The models (currently divergent, on the legacy/eager path): `brainpy_state/_nest/bernoulli_synapse.py`, `cont_delay_synapse.py` (+ their `_test.py`).
   - NEST upstream source for both: `models/bernoulli_synapse.h` (the `p_transmit` Bernoulli draw in `send()`) and `models/cont_delay_synapse.h`/`.cpp` (the sub-resolution `delay_offset_` and how `send()` splits delivery across the bracketing grid steps).
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the cont-delay sub-dt scheme (below) is the decision that needs sign-off.

## Goal

Rebuild both as frozen parameter-spec + pure `update(state, ctx) -> (new_state, w_eff)` on `EventPlasticProj`, retiring their eager/legacy code, validated against live NEST. Neither changes a learned weight — both are **delivery-semantics** models, so they exercise two substrate seams that the plastic clusters declared but never drove hard: **stochastic per-edge masking** (`ctx.key`) and **sub-dt delay**.

## Scope + the two design questions

### A. `bernoulli_synapse` — stochastic transmission (`ctx.key`)

Each presynaptic spike is transmitted **independently per connection** with probability `p_transmit`; weight is otherwise static. Map to the rule contract:
- `stochastic = True` (the Protocol flag that makes the substrate pass a non-`None` `ctx.key`).
- `update` emits `w_eff = where(pre_spike & (uniform(key_per_edge) < p_transmit), weight, 0)` — a **per-edge Bernoulli gate on the delivered amplitude**, no weight state mutated.
- **Design question (resolve approach-first):** confirm the substrate's `ctx.key` is **split per-edge per-step** so each connection draws independently (NEST draws per-connection in `send()`); if `EventPlasticProj` currently exposes only a single step key, add a documented **per-edge key-split** seam (`jax.random.split`/`fold_in` over the edge axis) — minimal, reused by any future stochastic synapse. State exactly where the split happens.
- **Parity is distributional only** (gap-doc: NEST per-thread RNG vs JAX PRNG — bit-exact unsupported). Use `compare_distributional`: over many trials / a long train, assert the **transmitted fraction → `p_transmit`** and that delivered-event count is Binomial(`n_pre_spikes`, `p`) within tolerance. Document the carve-out.

### B. `cont_delay_synapse` — sub-dt (non-grid) delay  ← **the core design question**

NEST allows a delay that is **not** a multiple of the resolution: it carries `delay_offset_ ∈ [0, dt)` and, in `send()`, delivers the event split/shifted to land at the exact off-grid time. The brainstate `InputDelay` grid rounds to `dt`. Decide and justify (record in Lessons):
- **(a) Linear two-step split** — deliver fraction `(1 − f)` at grid step `k` and `f` at step `k+1`, where `f = delay_offset/dt`. Reproduces the first moment of the off-grid arrival on a fixed grid; faithful for the alpha/exp PSC integral. Needs the substrate to scatter a delivery across **two** adjacent delay slots (a small generalisation of the single-slot `InputDelay`).
- **(b) Round-to-nearest grid + documented divergence** — snap `delay_offset` to the nearest `dt`, assert parity only at grid-resolvable delays, and document the sub-dt residual as a known divergence (`CAT_E`, sub-step).
Recommendation: **(a)** if the substrate's delay path can carry a per-edge fractional split cleanly (it's the NEST-faithful choice and matches how `precise_spiking`/`CAT_E` framed off-grid timing in cluster 03); fall back to **(b)** for any case where the split can't be expressed without breaking the event-matmul vectorisation. Map NEST's `delay_offset_` semantics exactly before choosing; cite the `cont_delay_synapse.cpp` `send()` lines.

## Constraints (from `CONTEXT.md` Part 2 + 01/05 Lessons)

- brainevent-only; `_network` compute, `_nest` spec+rule; **NEST fidelity wins**, document deltas; jit/vmap/for_loop-safe; **no host scalars / no Python RNG** in the hot path (the Bernoulli draw is `jax.random`, keyed off `ctx.key`).
- Keep `EventPlasticProj` (primitive #1) intact — both are plain rules on it; any seam you add (per-edge key split; two-slot delay scatter) is an **additive** substrate generalisation with its own unit test, default-off for existing rules.
- Retire the eager/legacy implementations from the active path (mirror 05's retirement of the legacy subclasses); replace the old eager `*_test.py` with a NEST-free `*_rule_test.py` (closed-form / distributional + jit/vmap smoke) **and** a live-NEST `_validation/*_parity_test.py`.
- NumPy docstrings citing the NEST `send()` source lines; >90 % coverage (use the **directory-path** `--cov=brainpy_state/_nest` form, or 05's untraced-import + `coverage.start()`+`reload` workaround if the import C-path coredumps).

## Validation (via the 00 harness)

`_validation/bernoulli_synapse_parity_test.py` + `cont_delay_synapse_parity_test.py`, `requires_nest`:
- **bernoulli:** `compare_distributional` — transmitted-fraction convergence to `p_transmit` across seeds/long train; Binomial count within tolerance; `p=0` (no delivery) and `p=1` (≡ static_synapse, can be `compare_trace` exact) end-points.
- **cont_delay:** for scheme (a), `compare_trace` of the post PSC against NEST at several `delay_offset` values (`CAT_B`/`CAT_E`); for scheme (b), grid-delay parity exact + a documented sub-dt divergence test. Verify against an exact grid delay first (offset 0 ≡ `static_synapse`) so a sub-dt mismatch isn't masked.

## Edge cases to enumerate + test

bernoulli: `p_transmit` 0 and 1; per-edge independence (two edges from one pre transmit independently — not a shared draw); key-split determinism (same key → same mask under `jit`); empty pre train. cont_delay: `delay_offset = 0` (≡ static), offset → `dt` (lands fully on next slot), offset straddling the last buffer slot, multapses with different offsets, delay below one `dt`. Both: `vmap`/`grad` smoke; `dt` invariance where defined; multi-segment / sliced populations preserve CSR edge order.

## Definition of done

- [ ] `bernoulli_synapse` + `cont_delay_synapse` rebuilt as spec+rule on `EventPlasticProj`; eager/legacy retired from the active path; old eager tests replaced.
- [ ] Both design questions resolved with sign-off (per-edge key split; cont-delay sub-dt scheme), decision + reasoning in the Lessons entry; any substrate seam added is additive + unit-tested + default-off.
- [ ] Distributional bernoulli parity + cont_delay (grid-exact and, for scheme (a), sub-dt) parity passing (live NEST) / skipping (without); tolerances + carve-outs documented.
- [ ] >90 % coverage; doctest examples; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the **per-edge key-split** API (where the split lives — reused by any future stochastic synapse), the **cont-delay sub-dt scheme chosen** (and the two-slot delay-scatter seam if added, which `precise`/off-grid demos may reuse), the distributional tolerances, and confirmation that the EventPlasticProj non-plastic delivery family is now **complete** (only plastic clusters 08/09/10 + bucket-3 remain).
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
