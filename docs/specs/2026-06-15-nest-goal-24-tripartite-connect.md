# /goal — 24 · TripartiteConnect-equivalent: `third_factor_bernoulli_with_pool` + the astro-pool network demos

> Self-contained prompt. Open at repo root. Stream: example (+ a connectivity-rule seam). **Follow-up cluster** (next after `21`/`22`/`23`) — builds the one connectivity rule the `17b` astro placeholders are blocked on, then flips them to real ports. **Reuses** the merged SIC path (`15d`) + the existing static/STP projections; the only new substrate is a tripartite **orchestration + a pool sampler** (no new deposit primitive). Depends on: `15d` (`sic_connection` / `_connect_sic` / `as_current` `'I_SIC'` deposit — **MERGED #65**) + `17b` (the `astrocyte_small_network` / `astrocyte_brunel_*` skipped placeholders to replace). Independent of `19`/`20`.

## What's blocked + the exact NEST rule (read first)
`17b` shipped `astrocyte_single` + `astrocyte_interaction` (the neuron↔astrocyte SIC loop) but left **three demos as documented skipped placeholders** — `astrocyte_small_network`, `astrocyte_brunel_bernoulli`, `astrocyte_brunel_fixed_indegree` — all blocked on NEST's `nest.TripartiteConnect(...)` with the `third_factor_bernoulli_with_pool` rule. The confirmed NEST signature + semantics (from the upstream examples):

```python
nest.TripartiteConnect(
    pre, post, third,                         # 3 populations (neurons_ex, neurons_ex+in, astrocytes)
    conn_spec={"rule": "pairwise_bernoulli", "p": p_primary},   # OR {"rule":"fixed_indegree","indegree":CE}
    third_factor_conn_spec={"rule": "third_factor_bernoulli_with_pool",
                            "p": p_third_if_primary, "pool_size": K, "pool_type": "random"|"block"},
    syn_specs={"primary":  <pre→post  syn>,    # e.g. tsodyks_synapse
               "third_in": <pre→astro syn>,    # pre spikes → astrocyte IP3 (delta path)
               "third_out":<astro→post syn>},  # sic_connection (the 15d w·SIC current deposit)
)
```

**Semantics** (the rule is the deliverable):
1. Sample the **primary** pre→post edges via `conn_spec` (any one-directional rule — `pairwise_bernoulli(p)` or `fixed_indegree(K)`).
2. For **each realized primary edge** `(pre_i → post_j)`, a Bernoulli(`p_third_if_primary`) trial decides whether it is **paired with one astrocyte**.
3. The paired astrocyte is drawn from **post_j's pool** — `pool_size` astrocytes assigned per target neuron:
   - `pool_type="random"` → `pool_size` randomly-chosen astrocytes per target (any size).
   - `pool_type="block"` → contiguous **non-overlapping** blocks by index (deterministic; the common case `N_astro == N_post`, `pool_size=1` gives the one-to-one `astro_j ↔ post_j` map).
4. A paired edge creates **two** derived connections: `third_in` (pre_i → astro) and `third_out` (astro → post_j, the `sic_connection`). The astrocyte thus bridges the pre→post pair (it sees the presynaptic spike via `third_in`→IP3 and returns SIC to the postsynaptic neuron via `third_out`).

## Session start
1. Create an isolated worktree branch **`nest-goal/24-tripartite-connect`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`15d`-astro Lessons** — the SIC loop API: `sic_connection` (one-way; `VALID_SENDERS={astrocyte_lr_1994}`/`VALID_RECEIVERS={aeif_cond_alpha_astro}`); `_connect_sic`; the `as_current` `EventProjection` depositing `w·SIC` into `'I_SIC'`; the neuron→astro **delta** arm (`Δ_IP3·w` IP3); `delay_steps`; the labelled-`I_SIC`-before-unlabelled-`I_stim` read order; the parity numbers (SIC-raises-firing **+5 Hz**, exact-after-`align_steps`); **and the recorded `TripartiteConnect` blocker this cluster clears.**
   - the **`17b`-astro-demos Lessons** — the three placeholders' `BLOCKED_REASON` + `NotImplementedError` `main()` + marker tests (to replace); the **multi-receptor conductance bridge** 17b added to `aeif_cond_alpha_astro` (`n_receptors`+`w_by_rec`; `receptor_type=k`→column `k-1`, positive-nS) + the **"silent dead conductance path"** trap; the astro-demo house style + the `delay_steps=10` NEST default.
   - the **connectivity Lessons** (`_connectivity.py`/`_rules.py` + `15c`'s labeled-deposit / `23`'s edge-State view) — the `ConnRule.sample → ConnSpec(pre_idx, post_idx, n_edges)` contract a rule realizes, and how `_connect_pair` builds projections.
   - the **`14`-network-demos Lessons** — distributional network parity (seed-mean/median band + the qualitative law).
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.8 astro rows (the three placeholders); `network-api-gap.md` (the `TripartiteConnect`/astrocyte-pool entry — the blocker to clear); `synapses-plasticity-gap.md` sic/astro notes.
   - The substrate: `brainpy_state/_network/{_simulator.py (`connect`/`_connect_pair`/`_connect_sic`),_rules.py,_connectivity.py,_event_proj.py}`; `brainpy_state/_nest/{astrocyte_lr_1994,aeif_cond_alpha_astro,sic_connection}.py`; the `17b` placeholder modules + tests.
   - NEST upstream (clone on this machine: **`/mnt/d/codes/githubs/computational_neuroscience/nest-simulator`**; the 17-Lessons `…/githubs/nest-simulator` path is machine-specific and **absent here** — `ls`-verify): `pynest/examples/astrocytes/{astrocyte_small_network,astrocyte_brunel_bernoulli,astrocyte_brunel_fixed_indegree}.py` (the `TripartiteConnect` calls + params: `p_primary`, `p_third_if_primary`, `pool_size`, `pool_type`, `w_a2n`); the NEST docs `tripartite_connectivity` / `third_factor_bernoulli_with_pool`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — the sign-off is **design question A (the orchestration + pool sampler), resolved by a tripartite micro-parity** below.

## Goal
Build a `Simulator`-level TripartiteConnect equivalent — the `third_factor_bernoulli_with_pool` pool sampler + a `tripartite_connect(pre, post, third, …)` orchestration that shares **one** realized primary edge sample across the primary / third_in / third_out arms (reusing the merged static-STP + `sic_connection` paths) — validated by a tripartite micro-parity, then **replace the three `17b` astro-pool placeholders with real distributional ports**.

## Scope + the core design question

### A. The tripartite orchestration + pool sampler — **resolve FIRST via micro-parity** (record in Lessons)
**Try:** a dedicated `Simulator.tripartite_connect(pre, post, third, *, conn_spec, third_factor_conn_spec, syn_specs, seed=None)` (a separate method mirroring NEST's `TripartiteConnect` — the two-population `connect` signature does not carry a third population + three `syn_specs`). It:
1. Samples the **primary** pre→post edges by delegating to the existing rule (`pairwise_bernoulli`/`fixed_indegree`) → realized `(pre_idx, post_idx)`.
2. Builds the **pool** map (`third_factor_bernoulli_with_pool`): per target neuron, a `pool_size`-astrocyte pool (`"random"` sample under `key`, or `"block"` contiguous non-overlapping assignment).
3. For each realized primary edge, a Bernoulli(`p_third_if_primary`) trial under `key`; if paired, draw one astrocyte from that target's pool → derived `third_in` `(pre_i → astro)` and `third_out` `(astro → post_j)` edge lists.
4. Wires **three projections from the shared sample**: `primary` via the existing static/STP path; `third_in` via the existing static/STP **delta** path (pre spikes → astrocyte IP3, exactly `15d`'s neuron→astro arm); `third_out` via the merged `sic_connection`/`_connect_sic` (`w·SIC` `as_current` into `'I_SIC'`). No new deposit variant.

**New substrate = only the sampler + the orchestration** (steps 2-3 and the 3-arm wiring). Build a **tripartite micro-parity FIRST** (a tiny seeded network: a handful of pre/post neurons + a small astro pool): assert the realized **primary edge set**, the **pool assignment**, and the derived **third_in/third_out edge sets** match NEST's `TripartiteConnect` — **exact for `pool_type="block"`** (deterministic) and **distributional for `"random"`** (degree/pairing-rate band). Decide:
- **(a)** the sampler + orchestration over the existing paths **suffices** → the minimal additive rule (recommended — `15d` already proved the SIC arm; this only adds the *which-edges* sampling). **(check this first.)**
- **(b)** edge-sharing/pool indexing forces a dedicated helper → the minimal such helper. **Prefer (a).**

### B. Pool-type correctness
`"block"`: contiguous non-overlapping astrocyte blocks by target index (assert the one-to-one `astro_j↔post_j` when `N_astro==N_post`, `pool_size=1` — the `astrocyte_small_network` case). `"random"`: `pool_size` distinct astrocytes per target under the seed (assert distinctness + the distributional pool composition). Honor `allow_autapses`/`allow_multapses` on the primary sample.

## Then: flip the three `17b` placeholders to real ports
- **`astrocyte_small_network`** → `pool_type="block"`, `p_primary=p_third=1.0`, `pool_size=1` (deterministic pairing) → assert the IP3/Ca/SIC dynamics + the SIC-modulated firing **tight where deterministic**, distributional otherwise.
- **`astrocyte_brunel_bernoulli`** → primary `pairwise_bernoulli(p_primary)`, `third_factor` `p_third_if_primary`, `pool_size=10`, `pool_type="random"` → distributional SIC-modulated population firing vs NEST (seed-mean band; the `14`/`15d` posture; SIC raises firing).
- **`astrocyte_brunel_fixed_indegree`** → identical but primary `fixed_indegree(CE)` → same distributional parity.
Each placeholder, as 17b shipped it, is a module-level `BLOCKED_REASON` string + a `main()` that raises `NotImplementedError`, guarded by a marker test — **replace** each with a real parity test (pass/skip) + a no-NEST companion, and remove `BLOCKED_REASON`/the `NotImplementedError`.

## Constraints
- **Reuse only**: primary + third_in ride the existing static/STP projection paths; third_out rides the merged `sic_connection` (`15d`). The *only* new code is the pool sampler + the `tripartite_connect` orchestration that shares the primary sample. seam-(H) `comm='dense'` for the SIC arm (the `15d` posture).
- **Route E/I onto `aeif_cond_alpha_astro` via the 17b multi-receptor conductance bridge, NOT weight sign:** the brunel post-neuron is a conductance cell, and 17b shipped the bridge (`n_receptors`+`w_by_rec` in `update()`; `receptor_type=k`→column `k-1`). Wire excitatory primary edges as `receptor_type=1` (→`g_ex`) and inhibitory as `receptor_type=2` (→`g_in`) with **positive-nS** magnitudes (the bridge expresses NEST's weight-sign routing as receptor routing). Heed 17b's **"silent dead conductance path"** — a conductance neuron wired without the bridge / wrong `receptor_type` leaves `V_m` pinned at `E_L` with **no error**; so a parity test must assert a non-trivial `g_ex`/`g_in`/`V_m` (+SIC) response, not merely that the network runs.
- `Simulator` API; `examples/nest/` house style; **brainunit** units; `from examples.nest.<module>`; no-NEST companion. Set the NEST recorder `interval=dt`; **labelled `I_SIC` read precedes the unlabelled `I_stim`** (`15d` — no double-count). `align_steps` for the loop latency; `delay_steps` per the NEST default.
- Sampling lowered with `jnp`/`brainstate.transform` and seeded via the `seed=`→key (reproducible) — **no Python per-edge loop** (CLAUDE.md #10). Multi-step runs via `brainstate.transform.for_loop`; any fix **reproduce-first** + a `for_loop`-lowering regression test (`12`). **`clear_caches()` + x64** for the stiff Ca dynamics (`21`).
- NumPy docstrings (runnable Examples on `tripartite_connect`); >90 % coverage on touched code. **Coverage (corrected by 17/17b):** `coverage run -m pytest <touched test files>` then `coverage report --include=…` — **drop `--source`** (both `--source=brainpy_state` and `--cov=pkg.mod` SIGABRT inside jaxlib/absl double-init) and let the run finish fully before reading `.coverage`.

## Validation (via the 00 harness)
`_nest/_validation/tripartite_connect_test.py` (the rule micro-parity) + `{astrocyte_small_network,astrocyte_brunel_bernoulli,astrocyte_brunel_fixed_indegree}_test.py`, `requires_nest`:
- **micro-parity** (rule): realized primary + derived third_in/third_out edge sets + pool assignment vs NEST — **exact for `"block"`**, distributional (degree/pairing-rate band) for `"random"`; seeded reproducibility.
- **network parity**: SIC-modulated population firing vs NEST — tight for the deterministic `small_network`, distributional (seed-mean band) for the two brunel variants.
- Each runnable example runs standalone without NEST (parity skips; the demo still plots its IP3/Ca/SIC + firing). README + examples-gap §3.8 + `network-api-gap.md` rows updated (the `TripartiteConnect` blocker cleared).

## Edge cases to enumerate + test
`p_third_if_primary=0` (no astrocyte pairing → plain primary network, SIC=0); `p_third=1`+`pool_size=1`+`block` (deterministic one-to-one — `small_network`); `pool_size > N_astro` (random, clamp/error); `pool_type="block"` with `N_astro % N_post ≠ 0` (the boundary handling NEST defines); a primary edge **not** paired (no third arm for it); `fixed_indegree` vs `pairwise_bernoulli` primary; `allow_autapses=False`; seeded reproducibility of the whole tripartite sample; `for_loop` lowering; `clear_caches()`/x64; standalone-without-NEST.

## Definition of done
- [ ] Design A resolved by the tripartite micro-parity (the sampler + orchestration over the existing paths suffices, **or** the minimal helper); the pool-type semantics (`block`/`random`) recorded in Lessons.
- [ ] `Simulator.tripartite_connect(...)` + the `third_factor_bernoulli_with_pool` pool sampler built (reusing static/STP + `sic_connection`); micro-parity passing (exact block / distributional random).
- [ ] The three `17b` placeholders (each a `BLOCKED_REASON` + `NotImplementedError` `main()` + marker test) **replaced** with real ports + parity tests (pass/skip) + no-NEST companions.
- [ ] Any fix reproduce-first + `for_loop`-lowering-tested; `clear_caches()`+x64; >90 % coverage on touched code.
- [ ] README + examples-gap §3.8 + `network-api-gap.md` rows updated (blocker cleared); Lessons entry in `CONTEXT.md`; **§3.8 astro fully complete**; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the **A resolution** (the `tripartite_connect` orchestration + `third_factor_bernoulli_with_pool` sampler sharing one primary sample across three arms; `block` vs `random` pool semantics; reuse of static/STP + `15d` `sic_connection` with no new deposit primitive), the three demos' parity numbers, and confirmation that **§3.8 astro is fully complete** (the `TripartiteConnect` blocker is cleared — the last astro deferral closed). Note whether tripartite is a dedicated method vs a `connect` branch.
2. Verify tests pass / skip; coverage > 90 %; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
