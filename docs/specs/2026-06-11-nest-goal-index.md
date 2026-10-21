# NEST-parity goal — master index (prompt-file ordering)

Drives the two streams in `CONTEXT.md` Part 1 as a sequence of self-contained
`/goal` prompt files. Run them **top to bottom**; each is one worktree → one PR →
one merge. Read `CONTEXT.md` (committed, repo root) before every one.

**Progress (2026-06-13):**
- ✅ **`00`–`10` MERGED** (PRs #44–#54). The **discrete-synapse stream is COMPLETE** —
  implemented + validated + documented. The substrate carries every shape a discrete
  plastic rule needs (multi-trace `(N,k)` `04`; `nearest` mode `05`; per-edge
  `edge_state_init()` `05`; N→E post-state reader `ctx.post_states` `07`; 1→E
  broadcast-signal reader `ctx.signals` `08`; per-edge `ctx.key` stochastic `06`;
  `fractional_delay` `06`). `09` shipped the send-view `weight_recorder`
  (`_network/_weight_recorder_view.py`) auditing all 13 plastic rules + closed the
  STP `import nest` gap; `10` shipped the `docs/nest-guide/` STDP-divergence reference
  + the docstring-link introspection-test pattern. `08`'s Lessons: *"a voltage-gated
  **and** neuromodulated synapse needs no new substrate."* Frozen APIs in Part 5.
- ✅ **`13`,`11`,`12`,`14` MERGED** (PRs #56–#59). **§3.3 (bar Urbanczik), §3.4 (bar 2
  blocked), §3.5 (complete, 16 ports), §3.6 spiking (bar Wang) all done.** Reusable
  seams landed: `connect(receptor_type=k)` (dual-mode), callable `_RECORDABLE_ALIAS`,
  `parrot_neuron`+`_relays_multiplicity`, `connect(seed=)`→rng, weight-matrix
  recording, `RELAY_D=0.1`, **seam (H)** presynaptic-STP emission (`_emission_attr` →
  `weight·emission`), **seam (I)** mc device→compartment-receptor routing + per-
  compartment aliases, `iaf_cond_exp` multi-receptor, `pairwise_bernoulli(p)`,
  the steady-state-rate + seed-mean/median distributional pattern. `iaf_bw_2001`
  validated to machine precision. (Plus `fix-brunel-es-import`.)
- ✅ **`21`,`22` MERGED** (PRs #61, #60). **§3.3 complete (5/5)** — `21` rebuilt
  `urbanczik_synapse` on `VoltageCoupledPlasticProj` reading the post's dendritic δΠ
  (primitive #2's 3rd user; the **named post-state reader** now serves somatic *and*
  dendritic-derived signals with no substrate change; **every plastic synapse is off
  the legacy base**). **§3.6 complete** — `22` confirmed **seam (H) generalized**
  (`_emission_attr`+`_emission_receptor`, `comm='dense'`) **expresses recurrent NMDA**
  (14's "new projection" was unnecessary; micro-parity 5e-15) + ported the Wang WTA
  net (distributional). Key disciplines banked: `clear_caches()`+float64 guard for
  stiff/mc parity tests; pipeline-latency `align_steps`; external drive into a
  multi-receptor cell **must name `receptor_type`**.
- ✅ **`16`,`23` MERGED** (PRs #63, #62). `16` shipped the §3.7 generator demos
  (`sinusoidal_poisson`/`sinusoidal_gamma`/`pulsepacket`, distributional); `23` shipped
  `sim.get_connections(...)`→`SynapseCollection` (lazy view over the projection edge State),
  retiring §3.4's `plot_weight_matrices`+`synapsecollection` placeholders → **§3.4 complete**.
- ✅ **bucket-3 RE-GRILLED (2026-06-14) — the "primitive #3" framing is dropped.**
  A live design session + codebase audit established: the rate neurons are already
  JAX-native (only a host-side dict-queue to delete); a rate connection **is** the
  seam-(H) graded-emission path `22` generalized; the substrate is already explicit
  one-step-lagged (= NEST's WFR seed, fixed-point-preserving). **No `ContinuousCoupledProj`
  primitive.** `15` splits into **4 model + 2 demo** clusters, each reusing seam-(H) +
  `InputDelay` + the de-queue template and adding ≤1 small additive deposit variant
  proven by a micro-parity test (the `22` discipline): gap = explicit-lag
  `w·(pre_V−post_V)` (seam-H `12` + reader `21`), **no WFR**; Siegert = leggauss-64
  ported to JAX (scipy as oracle); diffusion = dual-channel seam-H; sic = one-way seam-H.
- ✅ **`15a`–`15d` MERGED** (PRs #64, #67, #66, #65) — **bucket-3 model stream COMPLETE**;
  the "`ContinuousCoupledProj` primitive #3" never existed (seam-(H) reuse + explicit lag +
  de-queue throughout). `15a` rate-core (11 rate neurons + 2 connections de-queued onto
  receptor-optional continuous seam-(H); linear-rate FP `r*=(I−gC)⁻¹μ` to 1e-4 vs closed-form
  AND NEST; all 5 φ == NEST C++ exactly); `15b` gap (difference-current `(G−diag(D))@V[n−1]`;
  2-neuron membrane parity to machine precision between spikes, only O(dt) AP-edge jitter; a
  latent per-neuron gating-init bug found+fixed); `15c` siegert+diffusion (Φ(μ,σ²) ported to
  jnp so siegert lowers under `for_loop` — the 15a eager exception retired; dual-channel
  **labeled** deposit `'diffusion_mu'`/`'diffusion_sigma2'`; Brunel mean-field 0.00 % vs NEST);
  `15d` astro (one-way `as_current` SIC deposit into `'I_SIC'`; the neuron↔astro loop exact
  after `align_steps`; SIC-raises-firing +5 Hz on both sims). New reusable substrate:
  `EventProjection(as_current=…)` + `channel_label`/labeled convergent-accumulation deposits;
  **sparse graded emission still deferred** (only `comm='dense'`).
- ✅ **`17`,`17b` MERGED** (PRs #68, #70) — **bucket-3 demo stream COMPLETE.** `17` = the **rate**
  ports (`lin_rate_ipn_network`, `rate_neuron_dm`; FP == closed form `(λI−W)⁻¹μ` == NEST 1e-3,
  trajectory `align_steps=12`; **no substrate change** — 15a sufficed; §3.6 complete bar `ht_neuron`).
  `17b` = the **astro** ports (`astrocyte_single`/`astrocyte_interaction` on live NEST within
  `ASTRO_TOL` 1e-3 + `align_steps=3`) **plus a latent `aeif_cond_alpha_astro` spike→conductance fix**
  via a new **multi-receptor conductance bridge** (`_simulator.py:1486-1496`: a neuron declaring both
  `n_receptors` + a `w_by_rec` arg in `update()` gets `receptor_type=k`→column `k-1`, **positive-nS**
  weights; col0→`g_ex`, col1→`g_in`). The three pool-rule demos (`astrocyte_small_network`,
  `astrocyte_brunel_{bernoulli,fixed_indegree}`) ship as **`BLOCKED_REASON` placeholders** (`main()`
  raises `NotImplementedError`, marker-tested) → the **`24`** target. **bucket-3 fully closed.**
- ✅ **`24`,`19`,`20`,`25` MERGED (2026-06-16; PRs #74, #75, #72, #73); `18` DROPPED** (in `braintrace`).
  **The whole `00`–`25` sequence is now merged.** **`24`** TripartiteConnect — `tripartite_connect` +
  `third_factor_bernoulli_with_pool`; one realized primary sample shared across primary/`third_in`/`third_out`
  via `_ExplicitEdges`; the three §3.8 pool-rule demos promoted from placeholders (block edge-sets
  **bit-identical** to NEST, random pools seed-matched) → **§3.8 complete**. **`25`** swept all 8 conductance
  neurons into the 17b bridge (design A confirmed **source-only / kinetics-agnostic**; micro-parity `V_m`
  ~1.7e-6 mV) → `neurons-gap §4` cleared. **`20`** shipped the **`../../brainpy_state/_nest_spatial`** submodule
  (`grid`/`free` 2-D/3-D, gaussian kernel, masks, `spatial_pairwise_bernoulli` riding the **unchanged**
  `connect(rule=)` via a `with_coords` clone; `create(positions=)`+`get_position`) — grid coords exact,
  gaussian `p(d)` distributional ≤1.6 %. **`19`** shipped brette_gerstner figs + compartmental (on `cm_default`)
  + **pong** RL (new reusable **`Simulator.cont()`** + **`host_spike_drive`/`host_current_drive`** closed-loop
  seam). **The follow-up tail `26`–`28` is now ALL ✅ MERGED:** ① **`sudoku/`** (#81) — one compiled net +
  pong-style `cont()`/`set('weight')` host loop; the "intractable" verdict was a **pF/pA unit-comment bug**, not a
  substrate limit; new public **`explicit_edges`** primitive; ② **spatial extensions** (#80) — full residual
  `nest.spatial.*`, **exact** read-back parity (~1e-16), new `_eval_pair` expression protocol (≤6-line seam touch);
  ③ **PSC spike→current** (#79) — audit confirmed the dead path, `25` design-A at the **pA** default; the bridge is
  now **unit-agnostic** across `nS`+`pA`. **The numbered backlog `00`–`28` is COMPLETE** (see status note at end).

> Numbering note: `04`'s Lessons call Clopath "05"; the index keeps Clopath at
> `07` (primitive-#2 cluster) and `05` as the STDP-NN family. The multi-trace
> runway `04` built serves both.

## The session protocol (every prompt file embeds a self-contained copy)

```
SESSION START
  1. Create worktree branch  nest-goal/<NN>-<slug>  from origin/main
     (superpowers:using-git-worktrees).
  2. Read CONTEXT.md in full. Read the gap-doc section(s) named below and the
     relevant source under brainpy_state/ and the NEST upstream script(s).
  3. Write spec+plan under dev/superpowers/ (gitignored). Approach-first: state
     the approach and WAIT for approval before code (CLAUDE.md #1).

DO THE WORK  (TDD; NumPy docstrings; >90% coverage on touched code)
  - Architecture per CONTEXT.md Part 2 (no relitigation).
  - Live-NEST parity test in brainpy_state/_nest/_validation/<x>_test.py,
    skipUnless(nest). Carve-outs: distributional where PRNG diverges; docs-only
    exempt; blocked → skipped placeholder.

SESSION END
  4. Append a Lessons entry to CONTEXT.md (Part 5 template).
  5. Commit (NO Co-Authored-By). Push. Open PR (body: scope + parity numbers).
     Merge PR. Delete remote branch. git switch main && git merge --ff-only
     origin/main. Remove the worktree.
```

## Foundational batch (authored)

| NN | slug | stream | scope | depends |
|---|---|---|---|---|
| 00 | validation-harness | both | `brainpy_state/_nest/_validation/`: `nest_compare.py` (build-in-both + compare), `tolerance_conventions.py` (A–E categories + multi-seed), `requires_nest` marker, README. Refactor the 5 existing Brunel parity tests onto it. | — |
| 01 | event-plastic-substrate | synapse | `_network`: the shared CSR+`InputDelay`+rule-declared-`State` substrate; **`EventPlasticProj`**. `_nest`: rebuild `static_synapse`(+`_hom_w`) and the **STP family** (`tsodyks_synapse`,`_hom`,`tsodyks2_synapse`,`quantal_stp_synapse`) as spec+pure-rule. Gut their imperative code. Parity: PSC-amplitude train vs NEST (STP), event delivery (static). | 00 |
| 02 | single-neuron-demos | example | `_network` Simulator extensions: **analog/`multimeter` recording** (`res.trace`), **current-injecting devices** (`I_e`, `noise_generator`, `dc_/step_current_generator`), **sweep/re-run** ergonomics. Port `examples/nest/` §3.2: `one_neuron`, `one_neuron_with_noise`, `twoneurons`, `testiaf`, `balancedneuron`, `if_curve`, `vinit_example`. (Supersedes `nest-3.2-demos-prompt.md`.) | 00 |

## Synapse stream (outlined)

| NN | slug | scope | primitive | depends |
|---|---|---|---|---|
| 04 | stdp-core | `stdp_synapse`,`_hom`,`stdp_pl_synapse_hom`,`stdp_triplet_synapse`,`vogels_sprekeler_synapse`,`jonke_synapse`,`ht_synapse` → spec+rule; per-neuron online traces. Parity: weight trajectory over 5 s. | EventPlasticProj | 01 |
| 05 | stdp-nn | `stdp_nn_pre_centered`,`stdp_nn_restr`,`stdp_nn_symm`,`stdp_facetshw_synapse_hom`; per-edge/reset traces; single-spike-pair regression per pairing scheme. | EventPlasticProj | 04 |
| 06 | static-stochastic | finish `bernoulli_synapse` (distributional), `cont_delay_synapse` (sub-dt delay). | EventPlasticProj | 01 |
| 07 | voltage-coupled | **`VoltageCoupledPlasticProj`** (post-state reader) + `clopath_synapse` (post V; `aeif_psc_delta_clopath`/`hh_psc_alpha_clopath`) + `urbanczik_synapse` (`pp_cond_exp_mc_urbanczik`). | VoltageCoupledPlasticProj | 01 |
| 08 | dopamine-vt | `stdp_dopamine_synapse` + `volume_transmitter` broadcast; 3-neuron+1-VT weight-trajectory parity over 5 s (VT relay timing is the risk). | VoltageCoupledPlasticProj | 07 |
| 09 | weight-recorder-audit | per-STDP-variant `weight_recorder` hookup: emitted-event count + timing vs NEST (reuses State-tap weight recording). | — | 04,05,08 |
| 10 | stdp-docs | document STDP `tau_minus` trace-storage divergence (side-by-side) + `stdp_nn_*` pairing-convention docs; linked from each docstring. **Docs carve-out** (doctest, no parity run). | — | 04,05 |
| 15 | continuous-coupling | ✅ **RE-GRILLED → no new primitive** (seam-(H) reuse + explicit one-step-lag + de-queue). Split into **15a/b/c/d** model clusters + **17/17b** demos — see the *Bucket-3* section below. | seam (H) + `InputDelay` | 12,22,23 |
| ~~18~~ | eprop *(DROPPED)* | ~~`eprop_synapse*`,`eprop_learning_signal_connection*`,`weight_optimizer` + e-prop neurons~~ → **implemented in the `braintrace` package** (verbatim e-prop); out of this backlog. | — | n/a |

## Example stream (outlined)

| NN | slug | gap-doc | scope | depends |
|---|---|---|---|---|
| 03 | recording-demos | §3.4 | `multimeter_file`/in-memory, `recording_demo`, `cross_check_mip_corrdet`, `correlospinmatrix_detector_two_neuron`, `precise_spiking`, `plot_weight_matrices`, `synapsecollection`. | 02 |
| 11 | single-neuron-models-1 | §3.5 | `hh_psc_alpha`, `hh_phaseplane`, `aeif_cond_beta_multisynapse`, GIF (`gif_*`), GLIF (`glif_*`). | 02 |
| 12 | single-neuron-models-2 | §3.5 | MAT, `iaf_tum_2000_*`, `BrodyHopfield`, `CampbellSiegert`, `mc_neuron`, `izhikevich`. | 02 |
| 13 | plasticity-demos | §3.3 | `clopath_synapse_small_network`, `clopath_synapse_spike_pairing`, `evaluate_quantal_stp_synapse`, `evaluate_tsodyks2_synapse`, `urbanczik_synapse_example`. | 01,07 |
| 14 | network-demos | §3.6 | `wang_decision_making` (`iaf_bw_2001`), `EI_clustered_network/`, `brette_et_al_2007/`, `repeated_stimulation`, `sensitivity_to_perturbation`, `artificial_synchrony`. | 02 |
| 16 | generators *(→ done as a follow-up)* | §3.7 | `sinusoidal_poisson`/`sinusoidal_gamma`/`pulsepacket` — authored standalone (see follow-up table); §3.8 astro demos moved to **17b** (need `15d`'s `sic_connection`). | 02,03 |
| 17 | rate-demos | §3.6 | ✅ **MERGED (#68)** — `lin_rate_ipn_network`, `rate_neuron_dm` (no substrate change; FP == closed-form == NEST 1e-3). `17b` (#70) = the astro half. `ht_neuron`/`intrinsic_currents_*` stay out. | 15a |
| 19 | pedagogical *(P2)* | §3.10 | ✅ **MERGED (#75)** — brette_gerstner figs + `compartmental_model/` (on `cm_default`) + **pong** RL (new `Simulator.cont()`+`host_drive` seam). **`sudoku/` deferred** → cluster `26` (needs tractable WTA). | 02, `cm_default`, 08 |
| 20 | spatial | §3.9 | ✅ **MERGED (#72)** — `_nest_spatial/` submodule (`grid`/`free` 2D/3D, gaussian kernel, masks, `spatial_pairwise_bernoulli`, `create(positions=)`/`get_position`) + 4 demos; grid coords exact, `p(d)` ≤1.6 %. Extensions → cluster `27`. | 02 |

## Follow-up clusters (blocker-spawned — not in the original 00–20 numbering)

These emerged from deferrals recorded in the Lessons; authored on-demand once their
prerequisites landed. Both **reuse** existing primitives/seams (no new primitive).

| NN | slug | stream | scope | depends |
|---|---|---|---|---|
| 21 | urbanczik-dendritic | synapse | ✅ **MERGED (#61).** the **last plastic synapse**: rebuilt `urbanczik_synapse` on `VoltageCoupledPlasticProj` reading the post's dendritic δΠ (δΠ/`V_W_star` States on `pp_cond_exp_mc_urbanczik`; two pre-traces via the 04 seam). Retired 13's placeholder → §3.3 complete. | 07,04,12,13 |
| 22 | wang-nmda-network | example (+seam) | ✅ **MERGED (#60).** recurrent NMDA via the **generalized seam (H)** (`_emission_attr`+`_emission_receptor`, `comm='dense'`; no new projection needed — micro-parity 5e-15) + the `wang_decision_making` WTA network. Retired 14's placeholder → §3.6 complete. | 14,12,11 |
| 16 | generator-demos | example | §3.7 `sinusoidal_poisson`/`sinusoidal_gamma`/`pulsepacket`, distributional parity. §3.8 astro deferred (bucket-3 `sic_connection`). | 02,03 |
| 23 | connection-introspection | example (+API seam) | `sim.get_connections(...)`→`SynapseCollection` (`.get`/`.set`, lazy view over the projection edge State); retires §3.4's `plot_weight_matrices` + `synapsecollection` placeholders. network-api-gap §3.1/§3.8. | 02,03 |
| 24 | tripartite-connect | example (+conn-rule seam) | ✅ **MERGED (#74)** — `tripartite_connect` + `third_factor_bernoulli_with_pool` + `_ExplicitEdges` shared-sample; the three §3.8 pool-rule demos promoted from placeholders (block edge-sets **bit-identical** to NEST; random seed-matched) → **§3.8 complete**. | 15d, 17b |
| 25 | conductance-bridge-sweep | neuron (bridge reuse) | ✅ **MERGED (#73)** — all 8 conductance neurons enrolled into the 17b bridge (design A: **source-only / kinetics-agnostic**; micro-parity `V_m` ~1.7e-6 mV); `neurons-gap §4` cleared. Sibling **PSC `pA`** path still open → cluster `28`. | 17b |
| 26 | sudoku | example | ✅ **MERGED (#81)** — `examples/nest/{sudoku,sudoku_net,sudoku_puzzles}.py`: 3645 `iaf_psc_exp` (729 pops×5), 510 300-edge row/col/box/cell inhibition as **ONE** sparse projection, parrot-relayed live-toggleable clue stim, host `cont(100 ms)` relaxation. **The "intractable" verdict was a UNIT BUG** — `iaf_psc_exp` takes pF/pA but the source's `nF`/`nA` comments were WRONG (scaled PSPs ~1000× below bias → tonic spiking); native pF/pA → healthy WTA. Parity **distributional documented-partial** (easy board = NEST rate; hard puzzle-4 neither completes, brainpy tracks NEST's plateau ≥ −0.3). **New public primitive `brainpy.state.explicit_edges`**. §3.10 pedagogical now fully complete. | 19 |
| 27 | spatial-extensions | spatial | ✅ **MERGED (#80)** — full residual `nest.spatial.*` (api-gap §3.10 rows 157–163 all done): per-axis `distance.{x,y,z}` + `pos`/`source_pos`/`target_pos` expression family, 4 distributions (`exponential`/`gamma`/`gabor`/`gaussian2D`), 4 masks (`rectangular`/`doughnut`/`elliptical`/`ellipsoidal`), `nearest_element`/`select_nodes_by_mask`, `dump_layer_*`, smoke plot helpers. **Parity EXACT, not distributional** — a *read-back* trick (weight=spatial-param on `p=1`, read `GetConnections`) gives element-wise ~1e-16; all 7 masks sym-diff 0. New **`_eval_pair` expression protocol** (a **≤6-line** `_rule.py` dispatch — the only seam touch; `with_coords`/`sample` untouched). §3.10 spatial **fully closed** (20+27). | 20 |
| 28 | psc-spike-current-sweep | neuron | ✅ **MERGED (#79)** — `aeif_psc_alpha`/`aeif_psc_exp`/`hh_psc_alpha_clopath` enrolled into the bridge, zero substrate change. **Audit confirmed DEAD** for all three (spike drive left `V_m` pinned; `receptor_type=k` raised `AttributeError`) — same failure 17b/25 found on the conductance side. Fix = `25` design-A at the **pA** default (no `receptor_input_unit` override). Parity aligned (`align_steps=3`): AdEx `V_m` 1.72e-6 mV, HH 4.96e-5; I_e suites byte-identical. **The bridge is unit-agnostic** (one mechanism spans `nS`+`pA`); `n_receptors` must live in `__init__` (read at connect time). With 17b+25+28 **every audited self-pull neuron is bridged**; `aeif_psc_delta*` out of scope (delta-direct `w_delta`). | 25 |

## Bucket-3 — continuous coupling (re-grilled 2026-06-14, replaces "primitive #3 `15`")

**Design (locked):** *no new primitive.* Every continuous coupling reuses **seam-(H)** graded
emission (`12`/`22`, `comm='dense'`) + the existing **`InputDelay`** (`23`) + the **de-queue**
of the per-model host-side dict event emulator, read **explicit one-step-lagged** (phase-1 reads
previous-step analog state = NEST's WFR seed — fixed-point-preserving, transient-divergent; the
two WFR-flagged couplings, instantaneous-rate + gap, get this same lag, **no WFR machinery**).
Each model cluster adds **≤1 small additive deposit variant**, proven by a **micro-parity test
before the full network** (the `22` discipline — the test is the arbiter, not prior text).
**All four model clusters `15a`–`15d` are MERGED** (#64/#67/#66/#65) — the design held: each
reused the seam-(H) emitter + de-queue + ≤1 small deposit variant, no new primitive. The demo
clusters `17`/`17b` are **MERGED (#68/#70)** (rescoped from the 15a–d Lessons: the gap demos shipped
inside `15b`, `brunel_siegert` inside `15c`); 17b also landed a latent `aeif_cond_alpha_astro`
spike→conductance fix + the multi-receptor conductance bridge. `ht_neuron` stays **out of scope**
here (single-neuron intrinsic currents, not coupling).

| NN | slug | seam-(H) deposit variant landed | headline parity (achieved) | status |
|---|---|---|---|---|
| **15a** | rate-core | continuous `rate`/`phi_rate` → default delta channel (receptor-optional); sign-split `mult_coupling` | linear-rate FP `r*=(I−gC)⁻¹μ` **1e-4** vs closed-form+NEST; 5 φ == NEST C++ | ✅ #64 |
| **15b** | gap | **difference-current** `(G−diag(D))@V[n−1]` → current channel (seam-H pre-V + post-V reader); connect-time symmetry | 2-neuron membrane machine-precision between spikes (only O(dt) AP jitter); χ-coherence distributional | ✅ #67 |
| **15c** | siegert-diffusion | **dual-channel labeled** `drift→'diffusion_mu'`, `diffusion→'diffusion_sigma2'`; Φ(μ,σ²) → jnp (siegert now `for_loop`-able) | jnp-Φ == scipy ≤**1e-6**; two-siegert ~1e-15; Brunel mean-field **0.00 %** vs NEST | ✅ #66 |
| **15d** | astro | **one-way** `as_current` `w·SIC` → `'I_SIC'`; de-queue `aeif_cond_alpha_astro` | loop exact after `align_steps`; SIC-raises-firing **+5 Hz** both sims | ✅ #65 |
| **17** | rate-demos | — *(gap + siegert demos already shipped by 15b/15c)* | `lin_rate_ipn_network`, `rate_neuron_dm`; FP == closed-form == NEST 1e-3, traj `align_steps=12` | ✅ #68 |
| **17b** | astro-demos | *(latent fix: multi-receptor conductance bridge)* | `astrocyte_single`/`astrocyte_interaction` within `ASTRO_TOL` 1e-3; `small_network`/`astrocyte_brunel_*` → `BLOCKED_REASON` placeholders (→ `24`) | ✅ #70 |

## Recommended linear order

`00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 13 → 11 → 12 → 14 →`
`21 → 22 → 16 → 23 → 15a → 15b → 15c → 15d → 17 → 17b → 24 → 19 → 20 → 25 → 26 → 27 → 28` — **ALL ✅ MERGED** *(`18` dropped — e-prop lives in `braintrace`)*. The entire numbered backlog is complete.

Rationale: harness first; then the two stream foundations (`01` synapse substrate,
`02` example/Simulator extensions); validate STDP/STP and their docs before the
plasticity *examples* that depend on them (`13`); single-neuron + network demos;
the two blocker-spawned finishers (`21` Urbanczik → §3.3 done, `22` Wang → §3.6
spiking done); generators (`16`, bucket-3-independent); then **bucket-3** (re-grilled,
no new primitive) — `15a` rate-core foundation, then `15b` gap ∥ `15c` siegert+diffusion
∥ `15d` astro (independent children, each fresh from `origin/main`), then the demos `17`
(rate) + `17b` (§3.8 astro); then `24` (TripartiteConnect — flips the `17b` astro-pool
placeholders), pedagogical `19`, the `20` spatial submodule, and `25` (conductance-bridge
sweep — the 17b spike→conductance fix generalized to 8 neurons) last. `18` (e-prop) is
**dropped** — implemented in the `braintrace` package.

> **Status (2026-06-16):** `00`–`14`, `16`, `21`–`23` **and the entire bucket-3 model stream
> `15a`–`15d` MERGED** (PRs #44–#67) **+ `17`,`17b` MERGED** (#68, #70 — bucket-3 fully closed).
> **`24`,`19`,`20`,`25` MERGED** (PRs #74/#75/#72/#73) — **the entire `00`–`25` sequence is merged;** `18`
> dropped (in the `braintrace` package). **`24`** built the TripartiteConnect-equivalent (`third_factor_bernoulli_with_pool`
> + `tripartite_connect`) and **flips the three `17b` `BLOCKED_REASON` astro-pool placeholders** to real
> ports → §3.8 astro complete; **`19`** shipped the §3.10 pedagogical group (brette_gerstner + compartmental
> on `cm_default` + pong RL, new `cont()`/`host_drive` seam); **`20`** built the **`_nest_spatial/`** submodule
> (the one genuinely-absent capability — `nest.spatial.*`); **`25`** enrolled all 8 conductance neurons into
> the spike→conductance bridge; **`26`** ported `sudoku/` (#81, the deferred "intractable" item — a pF/pA
> unit-comment bug, not a substrate limit; new public `explicit_edges`); **`27`** closed the residual
> `nest.spatial.*` (#80, exact read-back parity + the `_eval_pair` protocol); **`28`** bridged the current-based
> `pA` spike→current family (#79, audit-confirmed dead path; the bridge is now unit-agnostic across `nS`+`pA`).
> **The entire numbered backlog `00`–`28` (examples-gap + synapses-plasticity + neuron-conductance/current +
> spatial) is COMPLETE — every cluster merged.** `18` (e-prop) is deliberately dropped (lives in `braintrace`).
>
> **Only residual = documentation hygiene, not implementation.** A spot-audit found `examples-gap.md` still marks
> ~13 demos `missing`/`blocked` that are in fact **already ported with parity tests** — the whole Brunel family
> (`brunel_alpha`/`delta`/`exp_multisynapse`/`evolution_strategies`/`siegert`), `CampbellSiegert`, `BrodyHopfield`,
> `mc_neuron`, both `iaf_tum_2000_*`, and the two introspection demos (`plot_weight_matrices`/`synapsecollection`,
> unblocked by `23`). These rows just need flipping. Genuinely out of scope (by spec/decision): e-prop +
> `weight_optimizer` (braintrace), `ht_neuron` intrinsic-currents, MPI/SONATA, and the optional `spatial/` tutorial
> variants (`conncon_*`/`connex*`/`grid_iaf_oc`/`test_3d` — pure demo variants over now-present primitives).
