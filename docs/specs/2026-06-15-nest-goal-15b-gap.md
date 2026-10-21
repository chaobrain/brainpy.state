# /goal — 15b · Gap junctions: explicit-lag `w·(pre_V − post_V)` difference deposit (seam-H 12 + post-V reader 21)

> Self-contained prompt. Open at repo root. Stream: synapse-substrate + `_nest` gap neurons. **Bucket-3 child** of `15a` (rate-core). Adds **one** new deposit variant — the *difference deposit* `w·(pre_emission − post_state)` — combining seam-(H) pre-V emission (`12`) with the named post-V reader (`21`), depositing the gap current into the post's **current** channel under explicit one-step-lag (**no WFR**). Depends on: `15a` (seam-H continuous emission + de-queue + lag template), `12` (seam H `_emission_attr`), `21` (`VoltageCoupledPlasticProj` named post-state reader), `14` (distributional) — **run `15a` first**.

## What 15a established (build on it, don't relitigate)

`15a`'s Lessons resolved: seam-(H) emits a **continuous** per-step State (`comm='dense'`), `_emission_receptor` optional → default channel; phase-1 reads it **one-step-lagged** (= NEST's WFR seed, fixed-point-preserving); the de-queue template (delete the host-side dict event emulator, read via the substrate). Gap junctions reuse all of it — except the pre emits its **voltage** `V` (not `rate`), the post-V is read via the `21` named reader, and the per-edge deposit is the **difference** `g·(V_pre − V_post)` into the **current** channel.

## The gap decomposition (locked in the bucket-3 re-grill)

```
I_gap,i[n] = Σ_j g_ij·(V_j[n-1] − V_i[n-1])  =  seam-(H) emit V_pre  −  named post-V read (21)
```

Confirmed in the grill: `hh_psc_alpha_gap` / `hh_cond_beta_gap_traub` are JAX-native RK45 integrators that read `I_gap` via the **ordinary `sum_current_inputs(x, V)` path** — **no** WFR buffers in the neuron (the docstring is explicit: *"`I_gap` must be computed externally and provided via the `x` parameter"*). The host-side NumPy `gap_junction` class (`begin_wfr_cycle`/`evaluate_gap_current`) is a faithful `gap_junction.cpp` **reference port that is NOT wired into the Simulator** and cannot be (eager NumPy). It **stays as documented reference**; the Simulator path is the explicit-lag difference deposit. Explicit-lag is stable in the canonical regime (`Σⱼgᵢⱼ/C_m·dt ≈ 0.04 ≪ 2`); divergence from NEST's WFR is **O(dt) AP-timing**, not a blow-up.

## Session start
1. Create an isolated worktree branch **`nest-goal/15b-gap`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`15a` Lessons** — the seam-H-continuous-emission resolution (receptor-optional), the de-queue template, the instantaneous-lag = WFR-seed posture (gap reuses the lag).
   - the **`12` Lessons (seam H)** — `_emission_attr` → `_connect_pair` deposits `weight·emission`; the **`for_loop`-lowering regression-test discipline**.
   - the **`21` Lessons (named post-state reader)** — a rule declares `post_state_reads=('<state>',)` → `ctx.post_states['<state>']` gathers the named post State per edge; it served somatic V (Clopath) and dendritic δΠ (Urbanczik) with no substrate change — here it reads post **`V`**.
   - the **`14` Lessons** — distributional / synchrony network parity (seed-mean/median band + the qualitative law).
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.6 gap rows + `synapses-plasticity-gap.md` §5 (gap / WFR risk notes).
   - `brainpy_state/_nest/gap_junction.py` (the host-side WFR **reference** — confirm it stays unused by the Simulator), `hh_psc_alpha_gap.py` (reads `I_gap` via `sum_current_inputs`; no WFR buffers), `hh_cond_beta_gap_traub.py`.
   - The substrate: `brainpy_state/_network/_simulator.py` `_connect_pair` (the seam-H `_emission_attr` branch + the phase-1 previous-state read), the `21` machinery (`post_state_reads`/`ctx.post_states`, in `_network/_voltage_coupled_plastic.py` or wherever it landed), `_network/_event_proj.py`.
   - NEST upstream: `models/gap_junction.{h,cpp}`, `models/hh_psc_alpha_gap.cpp` (the `SecondaryEvent`/gap-current handling), the examples `pynest/examples/gap_junctions_two_neurons.py` + `gap_junctions_inhibitory_network.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — design question A (the difference-deposit variant) is the sign-off; resolve it with a 2-neuron micro-parity test before the inhibitory network.

## Goal

Enable explicit-lag gap-junction coupling among `hh_psc_alpha_gap` / `hh_cond_beta_gap_traub`: a per-edge **difference deposit** `g·(V_pre − V_post)` (seam-H pre-V + `21` post-V reader) into the post's current channel, with connect-time **symmetry**, validated by 2-neuron synchronization + an inhibitory gap network against live NEST.

## Scope + the core design question

### A. The difference-deposit variant — **resolve FIRST via micro-parity** (record in Lessons)
seam-(H) deposits `weight·pre_emission`; `21` reads a named post State. Gap needs **both** in one per-edge deposit: `g·(V_pre[n−1] − V_post[n−1])` → the post's **current** channel (`sum_current_inputs`, where the gap neurons read `I_gap`). **Try:** a projection declaring `_emission_attr='V'` (pre) **and** `post_state_reads=('V',)` (post), depositing the per-edge difference `weight·(pre_V − post_V)` as a current. Build a **2-neuron gap micro-parity test** (two `hh_psc_alpha_gap` started at **different** V, one symmetric gap pair; compare V trajectories / phase-convergence vs live NEST). Decide:
- **(a)** the combined emit-V + read-post-V difference deposit **suffices** → the minimal additive variant in `_connect_pair`/`_event_proj`. **(check this first.)**
- **(b)** a timing/units mismatch forces a split (off-diagonal seam-H `+g·V_pre` + a neuron-side static self-leak `−G_i·V_i`, the row-sum `G_i` pushed into the neuron at connect time) → the hybrid. **Prefer (a)** (the grill chose the full-difference explicit-lag); (b) only if the micro-parity demands it.

### B. Symmetry enforcement
NEST's `gap_junction` is `REQUIRES_SYMMETRIC=True` (connected with `make_symmetric=True`). At connect time, materialize **both** directions of each gap pair (or require the user-supplied connectivity to be symmetric and error clearly if not); validate the realized conductance matrix `G` is symmetric. Gap connections carry **no delay** (raise on delay, per the model).

### C. Neurons
`hh_psc_alpha_gap` / `hh_cond_beta_gap_traub` already read `I_gap` via `sum_current_inputs(x, V)` — minimal/no neuron change (de-queue only if they carry a dict-queue). Keep the NumPy `gap_junction` reference class as-is (documented host-side reference; not wired into the Simulator).

## Constraints
- seam-(H) rides `comm='dense'` (the gap `G` is a dense symmetric matmul `G @ V`); note sparse-gap as out-of-scope.
- Explicit-lag is the **chosen posture** (no WFR). **Document the O(dt) AP-timing divergence** from NEST's WFR-refined solution; the synchronization phenomenon is preserved. Tighten `dt` for the micro-parity if needed.
- Any substrate/neuron change **RED-first** (`superpowers:test-driven-development`) + a **`for_loop`-lowering regression test** (12). Drive every multi-step run with `brainstate.transform.for_loop` (CLAUDE.md #10).
- **`clear_caches()` + float64 guard** for the stiff HH-gap parity tests (21 discipline — the collection imports float32 first; a later x64 flip otherwise diverges).
- NumPy docstrings; >90 % coverage on touched code. Coverage via `coverage run --source=brainpy_state` + the coverage API.

## Validation (via the 00 harness)
`_nest/_validation/gap_junction_test.py` + the 2-neuron micro-parity, `requires_nest`:
- **Micro-parity (precondition):** two `hh_psc_alpha_gap` started at **different** V, one symmetric gap pair; assert **phase-convergence to synchrony** on both sims, V-trajectory in a documented band (expect O(dt) timing drift vs NEST's WFR — document; tighten dt). Identical-IC neurons stay identical (trivial guard).
- **Inhibitory gap network:** the `gap_junctions_inhibitory_network` topology — a population of gap-coupled inhibitory HH neurons; assert the **population synchrony measure** (order parameter / spike-time coherence) on both sims distributionally (seed-mean/median, documented band; 14 posture).
- No-NEST companion: imports + a short headless 2-neuron gap run shows V-convergence.

## Edge cases to enumerate + test
Identical-IC pair (stays synchronized); anti-phase start (converges); `g=0` (decoupled — independent trajectories); strong `g` (still stable per the estimate — assert no blow-up); asymmetric connectivity request (errored or symmetrized per B); gap connection + a delay (raises); a 3-neuron chain vs all-to-all gap; the `clear_caches()`/x64 path; standalone without NEST; `for_loop` lowering of the coupled pair.

## Definition of done
- [ ] Design question A resolved by the 2-neuron micro-parity (combined difference deposit suffices, **or** the off-diagonal+self-leak split); decision + the explicit-lag O(dt) divergence recorded in Lessons.
- [ ] The difference-deposit variant landed additive + unit-tested; connect-time symmetry enforced; gap delay rejected; NumPy `gap_junction` reference class confirmed unused-by-Simulator.
- [ ] 2-neuron sync micro-parity + inhibitory-gap-network synchrony parity passing (live NEST) / skipping (without); `for_loop`-lowering test; `clear_caches()`+x64 guard.
- [ ] >90 % coverage; examples-gap §3.6 gap rows + README updated; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the **A resolution** (the difference-deposit `w·(pre_V−post_V)` variant: seam-H pre-V + `21` post-V reader → current channel, or the self-leak split), the **symmetry** enforcement, the **explicit-lag stability** result (canonical regime stable; the documented O(dt) AP-timing divergence vs NEST's WFR; the NumPy `gap_junction` class stays a host-side reference), and confirmation that **gap junctions need no WFR machinery**. Note the sparse-gap extension as deferred.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
