# Examples portfolio — NEST parity gap

## 1. Scope

PyNEST example portfolio at `nest/nest-simulator/pynest/examples/` vs.
`brainpy.state`'s `docs/examples/gallery.rst` (and the colocated `examples/`
folder it links to). Goal: a porting-target list ordered by priority.

Upstream reference:
<https://nest-simulator.readthedocs.io/en/stable/examples/index.html>
(listing also obtained from GitHub: `gh api repos/nest/nest-simulator/contents/pynest/examples`).

Evidence basis:
- `find docs/examples/ -type f` shows only `gallery.rst` (5.7 KB) — no notebooks,
  no scripts.
- `cat docs/examples/gallery.rst` (read in this analysis) lists 14 examples
  under three rubrics: classical network models, oscillations, and SNN
  training. **None of these are NEST-style ports.** They use brainpy.state's
  native compositional API.
- Upstream NEST examples list (~70 top-level scripts + ~10 subdirectories,
  retrieved 2026-05-11) covers: Brunel family, balanced random networks,
  COBA/CUBA, gap junctions, GIF/GLIF demos, HH demos, IAF Tum 2000 STP
  demos, plasticity demos (Clopath, STDP, STP, Urbanczik), correlation
  demos, multimeter recording demos, spatial subdir, astrocyte subdir,
  e-prop subdir, SONATA subdir, HPC benchmark, pong, sudoku.

## 2. Parity summary

The example portfolio gap is total: **zero NEST examples are ported into the
repo's example gallery**. The 14 examples present are brainpy-style E-I
networks and SNN training scripts — useful in their own right but not
demonstrating the NEST-compat surface to NEST users. This is the most direct
documentation-side blocker for NEST porting.

| Bucket | Count | Notes |
|---|---:|---|
| implemented (ported + reproducing NEST result) | 0 | |
| partial (concept covered by a non-NEST-style example) | ~4 | COBA / CUBA / E-I balanced / HH-COBA in `gallery.rst` overlap conceptually with Brunel-style examples |
| missing | ~50+ | flagship Brunel family + microcircuit + most plasticity / recording / spatial / astrocyte / e-prop / SONATA demos |
| unsupported | ~5 | MUSIC examples (`music_cont_out_proxy_example/`), SONATA (`sonata_example/`), structural plasticity (`structural_plasticity.py`), HPC benchmark (`hpc_benchmark.py` requires MPI), `store_restore_network.py` (kernel-state serialization) |
| **upstream NEST example scripts surveyed** | **≈ 70 + 10 subdirs** | per the gh-api listing 2026-05-11 |

## 3. Evidence-backed mapping table

Status legend: `missing` = no port in repo, `partial` = a conceptually similar
brainpy-style example exists, `implemented` = a port reproducing NEST's result
exists.

### 3.1 Flagship benchmarks (porting priority order)

| NEST example | Status | brainpy.state equivalent | NEST upstream | Notes |
|---|---|---|---|---|
| `brunel_alpha_nest.py` | missing | partial: `examples/102_EI_net_1996.py`, `103_COBA_2005.py`, `104_CUBA_2005.py` are similar in spirit | <https://nest-simulator.readthedocs.io/en/stable/auto_examples/brunel_alpha_nest.html> | The single most-cited NEST example. Uses `iaf_psc_alpha`, `poisson_generator`, `spike_recorder`, all present in repo. P0 port target. |
| `brunel_delta_nest.py` | missing | partial: same as above | upstream | Delta-current variant of Brunel |
| `brunel_exp_multisynapse_nest.py` | missing | none | upstream | Exercises multi-port AdEx; cross-link `neurons-gap.md` multisynapse |
| `brunel_siegert_nest.py` | missing | none | upstream | Mean-field comparison vs. spiking — uses `siegert_neuron` (already validated) |
| `brunel_alpha_evolution_strategies.py` | missing | none | upstream | Optimizer-tuned Brunel |

### 3.2 Single- and few-neuron demos

All seven ported in cluster 02 (`Simulator` API, live-NEST parity). The ports
forced four `Simulator` extensions reused downstream: **A** analog State-tap
recording (`voltmeter`/`multimeter` + `res.trace`/`res.times`), **B**
current-injecting devices (`noise_/dc_/step_/ac_generator` via the neuron's
current ring buffer), **C** rebuild-per-trial sweep ergonomics, **D**
per-generator weight vectors (multi-channel generator view).

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `one_neuron.py` | implemented | `examples/nest/one_neuron.py` | `iaf_psc_alpha` + `I_e` + `voltmeter`; V_m charge `CAT_B_ALIGNED` (ext. A) |
| `one_neuron_with_noise.py` | implemented | `examples/nest/one_neuron_with_noise.py` | 2-channel `poisson_generator`, signed weights `[1.2,-1.0]`; rate `CAT_D` 5 % (ext. A, D) |
| `twoneurons.py` | implemented | `examples/nest/twoneurons.py` | static synapse `w=20 pA, d=1 ms`; both V_m traces `CAT_B_ALIGNED` (ext. A) |
| `testiaf.py` | implemented | `examples/nest/testiaf.py` | charge→spike→refractory over `dt∈{0.1,0.5,1.0}`; V_m `CAT_B_ALIGNED` + count `CAT_E` (ext. C) |
| `balancedneuron.py` | implemented | `examples/nest/balancedneuron.py` | SciPy `bisect` inhib rate→5 Hz; root matches NEST (ext. C, D) |
| `if_curve.py` | implemented | `examples/nest/if_curve.py` | `aeif_cond_exp` + `noise_generator` F-I curve; rate `CAT_C_RATE`/`CAT_D` (ext. B, C) |
| `vinit_example.py` | implemented | `examples/nest/vinit_example.py` | `iaf_cond_exp_sfa_rr` V_m-init sweep; relaxation ~1e-14 mV (ext. A) |

### 3.3 Plasticity demos

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `clopath_synapse_small_network.py` | missing | none | uses `clopath_synapse` + `aeif_psc_delta_clopath` (both ported, both validated) |
| `clopath_synapse_spike_pairing.py` | missing | none | spike-pair regression; coordinates with `synapses-plasticity-gap.md` P1 |
| `evaluate_quantal_stp_synapse.py` | missing | none | exercises `quantal_stp_synapse` (currently unvalidated per `synapses-plasticity-gap.md`) |
| `evaluate_tsodyks2_synapse.py` | missing | none | exercises `tsodyks2_synapse` (currently unvalidated) |
| `urbanczik_synapse_example.py` | missing | none | uses `pp_cond_exp_mc_urbanczik` + `urbanczik_synapse` |
| eprop_plasticity/ (subdir) | unsupported until ported | none | requires e-prop neuron + synapse port (cross-link `neurons-gap.md` P2, `synapses-plasticity-gap.md` P2) |

### 3.4 Recording / device demos

Five ported in cluster 03 (`Simulator` API, live-NEST parity); two are **blocked**
and ship as skipped placeholders. The ports added one reusable extension — **E**
eager imperative devices (`mip_generator`/`correlation_detector`/
`correlospinmatrix_detector` driven post-hoc from State-tapped or
`device.simulate()` spike data, never inside the `for_loop`) — and one
validation-helper extension (`compare_distributional` `autocorr`/`cv` statistics).
One `_nest` model fix was required: `mcculloch_pitts_neuron` now self-manages its
PRNG (`environ.get('key')` is `None` inside a `for_loop`). The two blocked demos
need connection-weight introspection (`GetConnections`/`SynapseCollection`,
`network-api-gap.md` §3.1, §3.8).

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `multimeter_file.py` | implemented | `examples/nest/multimeter_file.py` | in-memory `iaf_psc_exp`; `V_m`/`I_syn_ex`/`I_syn_in` `CAT_B_GEN` (ext. E); conductance recordables a follow-up |
| `recording_demo.py` | implemented | `examples/nest/recording_demo.py` | `poisson_generator`→`iaf_psc_exp` recording tour; rate `CAT_D` (refractory-saturated) |
| `cross_check_mip_corrdet.py` | implemented | `examples/nest/cross_check_mip_corrdet.py` | eager `mip_generator`+`correlation_detector`; cross-correlogram `CAT_D` autocorr (ext. E) |
| `correlospinmatrix_detector_two_neuron.py` | implemented | `examples/nest/correlospinmatrix_detector_two_neuron.py` | `ginzburg`→`mcculloch_pitts` binary correlator; means/cov `CAT_D` (ext. E; fixed `mcculloch_pitts` rng) |
| `precise_spiking.py` | implemented | `examples/nest/precise_spiking.py` | grid `iaf_psc_exp` vs precise `iaf_psc_exp_ps`; onset-aligned spikes `CAT_E` |
| `plot_weight_matrices.py` | blocked | skipped placeholder | needs `GetConnections`/`SynapseCollection` (`network-api-gap.md` §3.1, §3.8) |
| `synapsecollection.py` | blocked | skipped placeholder | needs `SynapseCollection` + named rules + `Parameter` weights (`network-api-gap.md` §3.8, §3.1, §3.9, §3.11) |

### 3.5 Single-neuron model demos

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `hh_psc_alpha.py` | missing | partial: `examples/106_COBA_HH_2007.py` uses HH but in a network | |
| `hh_phaseplane.py` | missing | none | HH phase-plane analysis — pedagogical |
| `aeif_cond_beta_multisynapse.py` | missing | none | AdEx multi-receptor demo |
| `gif_cond_exp_multisynapse.py` | missing | none | |
| `gif_pop_psc_exp.py` | missing | none | population GIF |
| `gif_population.py` | missing | none | |
| `glif_cond_neuron.py` | missing | none | |
| `glif_psc_neuron.py` | missing | none | |
| `glif_psc_double_alpha_neuron.py` | missing | none | |
| `iaf_tum_2000_short_term_depression.py` | missing | none | LIF + integrated STP, depression regime |
| `iaf_tum_2000_short_term_facilitation.py` | missing | none | LIF + integrated STP, facilitation regime |
| `mc_neuron.py` | missing | none | multi-compartment demo — exercises `iaf_cond_alpha_mc` (flagged experimental) |
| `BrodyHopfield.py` | missing | none | spike-coding network |
| `CampbellSiegert.py` | missing | none | mean-field cross-check |

### 3.6 Network demos

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `EI_clustered_network/` (subdir) | missing | partial: `examples/110-113_*` Susin-Destexhe gamma series is conceptually adjacent | |
| `brette_et_al_2007/` (subdir) | missing | partial: `examples/106_COBA_HH_2007.py` overlaps | the Brette benchmark family |
| `lin_rate_ipn_network.py` | missing | none | `lin_rate` + rate connections |
| `rate_neuron_dm.py` | missing | none | rate-network decision-making |
| `wang_decision_making.py` | missing | none | classic Wang 2002 NMDA network — uses `iaf_bw_2001` (currently unvalidated) |
| `artificial_synchrony.py` | missing | none | |
| `repeated_stimulation.py` | missing | none | |
| `sensitivity_to_perturbation.py` | missing | none | |
| `gap_junctions_two_neurons.py` | missing | none | uses `hh_psc_alpha_gap` + `gap_junction` |
| `gap_junctions_inhibitory_network.py` | missing | none | |
| `intrinsic_currents_spiking.py` | missing | none | uses `ht_neuron` |
| `intrinsic_currents_subthreshold.py` | missing | none | uses `ht_neuron` |

### 3.7 Generator-pattern demos

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `sinusoidal_gamma_generator.py` | missing | none | |
| `sinusoidal_poisson_generator.py` | missing | none | |
| `pulsepacket.py` | missing | none | uses `pulsepacket_generator` |

### 3.8 Astrocyte demos

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `astrocytes/` (subdir, contains `astrocyte_brunel_*` series) | missing | none | uses `astrocyte_lr_1994` + `aeif_cond_alpha_astro` (both validated). Astrocyte Brunel variant is the natural P0 port target for the astrocyte family. |

### 3.9 Spatial demos

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `spatial/` (subdir) | missing | none | depends on `nest.spatial.*` which is absent (`network-api-gap.md` §3.10). Blocked by that P2. |
| `csa_example.py` | missing | none | Connection Set Algebra — `conngen` rule, also absent |
| `csa_spatial_example.py` | missing | none | |

### 3.10 Pedagogical / advanced

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `compartmental_model/` (subdir) | missing | none | dendritic-tree models on `cm_default` |
| `pong/` (subdir) | missing | none | reinforcement-learning demo |
| `sudoku/` (subdir) | missing | none | constraint-satisfaction with stochastic neurons |
| `structural_plasticity.py` | unsupported | none | spec §7 (no structural plasticity) |
| `store_restore_network.py` | unsupported | none | kernel-state serialization is NEST-internal |
| `music_cont_out_proxy_example/` | unsupported | none | spec §7 (MUSIC) |
| `sonata_example/` | unsupported | none | spec §7 (SONATA) |
| `brette_gerstner_fig_2c.py`, `brette_gerstner_fig_3d.py` | missing | none | reproduces figures from AdEx paper |

## 4. Missing or incomplete functionality

The entire NEST examples portfolio is missing from the repo (except where
conceptually-overlapping brainpy-style examples exist for E-I balanced
networks, COBA, and HH-COBA). Concretely:

- **Flagship benchmarks**: Brunel family (5 variants), HPC benchmark.
- **Single-neuron pedagogy**: `one_neuron.py`, `one_neuron_with_noise.py`,
  `if_curve.py`, `balancedneuron.py`, `testiaf.py`, `vinit_example.py`.
- **Plasticity demos**: every Clopath, STDP, STP, Urbanczik demo.
- **Recording demos**: full multimeter / spike_recorder / weight_recorder
  pedagogy + correlation-detector regressions.
- **Single-neuron model demos**: HH, AdEx, GIF, GLIF, MAT, IAF Tum 2000, Brody-
  Hopfield, Campbell-Siegert, multi-compartment, intrinsic-currents.
- **Network demos**: gap junctions, rate networks, Wang 2002, Brette 2007,
  EI-clustered, perturbation sensitivity.
- **Generator demos**: sinusoidal Poisson + gamma, pulse packets.
- **Astrocyte demos**: the `astrocytes/` Brunel-variant series.
- **Spatial and SONATA**: blocked by API absence; classified `unsupported`
  until the underlying API lands.

## 5. Semantic & numerical risks

- **Each port is also a validation harness.** A ported Brunel example *is*
  an end-to-end NEST-comparison test (firing rate, mean ISI, CV of ISI).
  Skipping the example means losing the regression — and losing the
  promotion-blocking parity check.
- **Multimeter file-backend gap.** `multimeter_file.py` and `recording_demo.py`
  use NEST's `ascii` / `sionlib` backends; the repo doesn't have these
  (`devices-gap.md` P2). Ports either skip the file-backend step or rely on
  the P2 work landing first.
- **Plasticity examples need volume-transmitter parity.** Several plasticity
  demos require `stdp_dopamine_synapse` + `volume_transmitter` which are
  unvalidated (`synapses-plasticity-gap.md`).
- **Spatial examples are blocked.** Every example in `spatial/`,
  `csa_example.py`, and `csa_spatial_example.py` depends on the absent
  spatial API.
- **`structural_plasticity.py` is genuinely unsupported.** Not a gap, by
  design.
- **HPC benchmark requires MPI.** Not a gap, by design.

## 6. Validation gaps

The examples-as-validation point above is the headline: porting NEST examples
*is* the most rigorous form of validation. Specific gaps:

- No port of any Brunel variant → no flagship-level network parity test.
- No port of any plasticity demo → STDP / STP / Clopath learning-curve
  regression is absent end-to-end.
- No port of `cross_check_mip_corrdet.py` → no detector regression at
  network level.
- Per-example pass/fail is undefined because no examples are ported. Define
  in the roadmap below.

## 7. Prioritized roadmap

- **P0 — Port `brunel_alpha_nest.py` as the flagship example.** [L]
  Rationale: most-cited NEST example, exercises `iaf_psc_alpha` +
  `poisson_generator` + `spike_recorder` — all three present, IAF psc family
  P0-priority validation per `neurons-gap.md`. Acceptance:
  `examples/nest/brunel_alpha.py` exists; produces population firing-rate +
  CV of ISI matching NEST's example within 5 % over a 1 s window; uses
  `nest_compat` shim from `network-api-gap.md` P0; a CI test in
  `brainpy_state/_nest/_validation/brunel_test.py` runs it (skipped by
  default) and asserts the comparison.

- **P0 — Port `one_neuron.py` + `one_neuron_with_noise.py` as the first-day
  pedagogy.** [S]
  Rationale: simplest possible PyNEST script — paired with a
  `nest_compat`-using equivalent it makes the porting story visible to new
  users. Acceptance: both ports live in `docs/nest-guide/examples/`
  (cross-link `docs-portfolio-gap.md`) and render with comments comparing
  PyNEST and `nest_compat` calls side by side.

- **P0 — Port `multimeter_file.py` or an in-memory equivalent.** [M]
  Rationale: serves as the recording-device parity test that
  `devices-gap.md` P0 already prescribes. Acceptance: example port produces
  the same per-step `V_m` trace as the NEST example (modulo recording
  backend); test in `_validation/recorder_parity_test.py` references it.

- **P1 — Port the rest of the Brunel family.** [L]
  `brunel_delta_nest.py`, `brunel_exp_multisynapse_nest.py`,
  `brunel_siegert_nest.py`. Acceptance: each ported, firing-rate stat
  matches NEST.

- **P1 — Port `clopath_synapse_small_network.py` +
  `clopath_synapse_spike_pairing.py`.** [M]
  Rationale: validates Clopath end-to-end. Acceptance: weight trajectory
  matches NEST over 5 s within tolerance.

- **P1 — Port `evaluate_quantal_stp_synapse.py` and
  `evaluate_tsodyks2_synapse.py`.** [M]
  Rationale: doubles as STP validation called out in
  `synapses-plasticity-gap.md` P0. Acceptance: PSC amplitude train matches
  NEST.

- **P1 — Port `astrocytes/astrocyte_brunel_*` (at least the
  `fixed_indegree` variant).** [L]
  Rationale: the astrocyte family is already validated at unit level; the
  network-level demo proves it composes. Acceptance: astrocyte-modulated
  firing rate matches NEST.

- **P1 — Port pedagogical singles: `if_curve.py`, `balancedneuron.py`,
  `testiaf.py`, `vinit_example.py`.** [M]
  Acceptance: all four ported; each produces the same plotted curve / final
  state as NEST.

- **P2 — Port HH demos (`hh_psc_alpha.py`, `hh_phaseplane.py`).** [M]
  Acceptance: phase-plane plot matches NEST.

- **P2 — Port GIF / GLIF / MAT / IAF Tum 2000 single-neuron demos.** [M]
  Acceptance: each ported; trace matches NEST.

- **P2 — Port `wang_decision_making.py`.** [L]
  Blocked by `iaf_bw_2001*` validation (currently unvalidated). Acceptance:
  decision dynamics match NEST in a single-trial trace.

- **P2 — Port gap-junction demos (`gap_junctions_*`).** [M]
  Blocked by gap-junction synapse parity audit (`synapses-plasticity-gap.md`).
  Acceptance: synchronization metric matches NEST.

- **P2 — Port correlation-demos (`cross_check_mip_corrdet.py`,
  `correlospinmatrix_detector_two_neuron.py`).** [M]
  Acceptance: covariance matrix matches NEST.

- **P2 — Spatial examples.** [XL]
  Blocked by `network-api-gap.md` spatial roadmap. Acceptance: at least one
  spatial example ports cleanly via `nest_compat.spatial.*`.

- **P2 — `pong/` and `sudoku/` demos.** [L]
  Rationale: demonstrates RL / constraint-satisfaction patterns. Lower
  priority because they're niche but high-profile when present.
  Acceptance: both ported; functional behavior (pong: ball-tracking accuracy;
  sudoku: solve rate) matches the NEST examples within the documented
  stochastic variance.
