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
| `brunel_siegert_nest.py` | implemented | `examples/nest/brunel_siegert.py` | upstream | Mean-field Brunel: `siegert_neuron` + dual-channel `diffusion_connection`, relaxed end-to-end through the `Simulator`; 32.03 vs NEST 32.03 spks/s (0.00 %) and matches the closed-form Siegert fixed point to ~3e-13 (cluster 15c; §3.6) |
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

All five ported on the `Simulator` API with live-NEST parity — four in cluster 13,
the fifth (`urbanczik_synapse_example`) in cluster 21. The cluster-13 ports added
two reusable extensions —
**F** plastic projections + weight recording (`connect(synapse=<plastic rule>)`
dispatches to `EventPlasticProj` / `VoltageCoupledPlasticProj`; `record_weight` +
`res.weight_trace(proj)` → `(T, E)` CSR order) and **G** stochastic seed threading
(`connect(seed=)` keys the per-edge release PRNG, surviving `simulate`'s
`init_all_states`). One `_network` seam fix was required: the `connect(seed=)`
integer now threads into the plastic projection's runtime release `rng` (reproduced
with a regression test first), so stochastic rules are reproducible through
`simulate`. The deterministic ports also pinned the **`RELAY_D` holder-lag
convention** (NEST's parrot relay delay set to the `Simulator` generator's 0.1 ms
holder lag) and surfaced the NEST `quantal_stp_synapse` `set_status` footgun (`u`
stays at the 0.5 constructor default unless pinned, sibling of the known `a`
footgun). Cluster 21 then rebuilt `urbanczik_synapse` as a frozen spec + pure
`update` kernel on `VoltageCoupledPlasticProj`, extending that post-state reader to
pull a **named dendritic** post State — the prediction error `delta_Pi` — via the
rule's `post_state_reads`, so the same primitive #2 serves both a somatic (Clopath)
and a dendritic-compartment reader without change. It also validated
`pp_cond_exp_mc_urbanczik` against live NEST and retired the placeholder
(`synapses-plasticity-gap.md` §3, `neurons-gap.md` §3).

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `clopath_synapse_spike_pairing.py` | implemented | `examples/nest/clopath_synapse_spike_pairing.py` | voltage-based STDP 10–50 Hz; stored weight `res.weight_trace` in clopath band (LTP ≤ 3.3 %, LTD near-exact) (ext. F) |
| `clopath_synapse_small_network.py` | implemented | `examples/nest/clopath_synapse_small_network.py` | recurrent Clopath weight matrix; per-edge final weight in clopath band (LTP ≤ 2.0 %, LTD near-exact) (ext. F) |
| `evaluate_tsodyks2_synapse.py` | implemented | `examples/nest/evaluate_tsodyks2_synapse.py` | deterministic Tsodyks-Markram; PSC-train post `V_m` `CAT_B` ~9e-16 mV (ext. F via post `V_m`) |
| `evaluate_quantal_stp_synapse.py` | implemented | `examples/nest/evaluate_quantal_stp_synapse.py` | stochastic quantal STP; seed-mean `V_m` `CAT_D` (dep 1.8 %, fac 2.9 %, 8 seeds) (ext. F, G) |
| `urbanczik_synapse_example.py` | implemented | `examples/nest/urbanczik_synapse_example.py` | Urbanczik-Senn dendritic prediction error; soma conductance teacher + plastic dendrite, learning asserted (rate err ratio ~0.56); rule reads post `delta_Pi`, neuron-side `urbanczik_synapse_parity_test.py` (ext. F, dendritic reader) |
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

Seven ported in cluster 11 (`Simulator` API, live-NEST parity). The ports added
two reusable connection seams plus a batch of recordable aliases. **F**
multi-receptor routing: `connect(..., receptor_type=k)` selects a synaptic port;
conductance models bridge it through a per-port `w_by_rec` weight (unit from
`receptor_input_unit`), current-based GLIF models pull a keyed
`sum_delta_inputs(label=f'receptor_{k}')`. **G** spike-multiplicity relay: the new
`parrot_neuron` repeats every incoming spike *including* multiplicity, and the
spike substrate honours a `_relays_multiplicity` flag so the relayed count is
captured raw instead of binarised. The ports also extended `_RECORDABLE_ALIAS`
with HH gating (`Act_m`/`Inact_h`/`Act_n`), GLIF threshold components
(`threshold`/`threshold_spike`/`threshold_voltage`), per-port conductance (`g_k`,
resolved from a list, a last-axis index, or a flat `g_syn`), `ASCurrents_sum`,
summed PSC `I_syn`, and injected current `I`.

`gif_pop_psc_exp.py` / `gif_population.py` are **deferred to cluster 12/14**: both
are *population* (mean-field) GIF models, not single-neuron traces — they need a
population-density update that is out of scope for the single-neuron parity
harness this cluster builds.

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `hh_psc_alpha.py` | implemented | `examples/nest/hh_psc_alpha.py` | step-current + F–I sweep; subthreshold `V_m`+gating `CAT_A` (~1e-3 mV), spike counts match (seam A/B) |
| `hh_phaseplane.py` | implemented | `examples/nest/hh_phaseplane.py` | V–n phase plane; `n`-nullcline within one grid step of analytic `n_inf(V)` (NEST-free) |
| `aeif_cond_beta_multisynapse.py` | implemented | `examples/nest/aeif_cond_beta_multisynapse.py` | 4-receptor AdEx; `V_m` ~1e-6 mV, `g_1..g_4` machine precision (seam F). **`n=1` only** — see §5 broadcasting limitation |
| `gif_cond_exp_multisynapse.py` | implemented | `examples/nest/gif_cond_exp_multisynapse.py` | multi-receptor GIF; subthreshold `V_m` machine precision (seam F) |
| `gif_pop_psc_exp.py` | deferred | none | population GIF — deferred to cluster 12/14 (mean-field, not single-neuron) |
| `gif_population.py` | deferred | none | population GIF — deferred to cluster 12/14 |
| `glif_cond_neuron.py` | implemented | `examples/nest/glif_cond_neuron.py` | 5 mechanism levels; `g_1`/`g_2` full-trace + spike counts exact, subthreshold `V_m`/`threshold` ~1e-13 mV (seam F) |
| `glif_psc_neuron.py` | implemented | `examples/nest/glif_psc_neuron.py` | 5 levels, current-based; `I_syn`/`I` full-trace ~2e-15 pA + counts exact, `V_m` ~0.03 mV (`CAT_B_ALIGNED`); Poisson window via parrot (seam F/G) |
| `glif_psc_double_alpha_neuron.py` | implemented | `examples/nest/glif_psc_double_alpha_neuron.py` | 3 kernel configs; subthreshold `V_m`/`I_syn` full-trace ~1e-13 mV / ~1e-15 pA (seam F) |
| `iaf_tum_2000_short_term_depression.py` | missing | none | LIF + integrated STP, depression regime |
| `iaf_tum_2000_short_term_facilitation.py` | missing | none | LIF + integrated STP, facilitation regime |
| `mc_neuron.py` | missing | none | multi-compartment demo — exercises `iaf_cond_alpha_mc` (flagged experimental) |
| `BrodyHopfield.py` | missing | none | spike-coding network |
| `CampbellSiegert.py` | missing | none | mean-field cross-check |

### 3.6 Network demos

Six spiking-network demos: five ported in cluster 14 plus Wang's decision network in
cluster 22 (`Simulator` API, live-NEST **distributional** parity). Network parity is
distributional by construction — chaotic / balanced / metastable nets PRNG-diverge from
NEST, so each test compares a seed-**mean** (or, for `ei_clustered`, a seed-**median**
robust to a rare globally-synchronized seed) of a population observable within a
documented band, plus the qualitative law the demo exists to show. Bands are wider than
the single-neuron 5 % because a *balanced* rate sits on a near-cancellation of large E/I
currents → hypersensitive to sub-percent scatter, and a winner-take-all attractor
amplifies them further (so Wang's parity is the *direction* and *contrast* of the
decision, not the winner's absolute rate). The ports added two neuron seams — **the
`iaf_cond_exp` multi-receptor routing** (`n_receptors=2`, `receptor_input_unit=u.nS`, a
`w_by_rec` branch: receptor 1→`g_ex`, 2→`g_in`) so COBA excitation/inhibition route
through `connect(receptor_type=k)` (extends the §3.5 seam F to a conductance LIF), and
**recurrent presynaptic-gated NMDA** (cluster 22: `iaf_bw_2001` emits a graded
`spike_offset = k0 + k1·s_NMDA_pre` routed by `connect(receptor_type=NMDA, comm='dense')`
into the post's NMDA channel; reproduces NEST's recurrent gate to machine precision, so
no bespoke offset-aware event projection is needed) — and one connection rule,
**`pairwise_bernoulli(p)`** (Phase 0).

The rate-neuron (`lin_rate_ipn_network`, `rate_neuron_dm`), gap-junction
(`gap_junctions_*`), and `ht_neuron` (`intrinsic_currents_*`) demos were **out of scope
for the spiking network-demo cluster (14)** — each needs a primitive that harness does not
build. The **rate-neuron substrate has since landed (cluster 15a)**: `lin_rate`/`rate_neuron`
+ instantaneous/delayed rate connections now run on the seam-(H) continuous-emission path
(`τ Ẋ = −λX + μ + φ(h)`, linear-rate FP `(I−gC)⁻¹μ` parity vs NEST), so the two rate-network
demos are **substrate-ready** and only await a demo-port cluster. Gap-junction and
`ht_neuron` demos remain blocked on their own primitives.

The **Siegert mean-field network demo has landed (cluster 15c)**. `siegert_neuron`'s
transfer Φ(μ,σ²) now lowers under `for_loop` — a jnp leggauss-64 quadrature port of the
SciPy oracle (matched ≤1e-6 across the (μ,σ²) grid; the 15a eager exception is retired) —
and `diffusion_connection` is a thin NEST-parity status spec that the `Simulator` routes
as a **dual-channel seam deposit**: drift·rate → the target's `'diffusion_mu'` delta
channel, diffusion·rate → `'diffusion_sigma2'`, read back with
`sum_delta_inputs(label=…)`. The flagship `brunel_siegert.py` (§3.1) is rewritten to relax
three rate nodes end-to-end through the `Simulator` — six convergent `diffusion_connection`
edges (drive/ex/in into each of ex/in, incl. the ex→ex and in→in population self-coupling)
whose μ/σ² deposits **accumulate per target** — reproducing the closed-form self-consistent
Siegert fixed point to ~3e-13 and a live two-population NEST run to 0.00 % (32.03 vs 32.03
spks/s). The host dict-queue secondary-event emulator is retired; the new path is
JAX-native end to end (`siegert_diffusion_test.py`, `brunel_siegert_test.py`).

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `brette_et_al_2007/` (subdir) | implemented | `examples/nest/brette_et_al_2007.py` | IF benchmarks 1 (COBA `iaf_cond_exp`) + 2 (CUBA `iaf_psc_exp`), one consolidated script; steady-state E/I rate band 15 %/12 % (COBA E 8.9 %, CUBA E 1.5 %); needed the COBA receptor seam. HH benchmark-3 sibling `examples/brainpy_like/106_COBA_HH_2007.py` (cross-linked) |
| `EI_clustered_network/` (subdir) | implemented | `examples/nest/ei_clustered_network.py` | Litwin-Kumar/Rostami clustered RBN; rep=1 median E/I rate `CAT_D` 12 % (meas. ~1–3 %) + ISI-CV 8 % (meas. <4 %); rep=6 clustering signature (`std6>3·std1`, `CV6>CV1`) both sims. Uses `pairwise_bernoulli` |
| `artificial_synchrony.py` | implemented | `examples/nest/artificial_synchrony.py` | Golomb–Rinzel Σ vs coupling; uncoupled baseline exact, Σ↑ monotone both sims, sensitive strengths ~10 % band (grid branch) |
| `repeated_stimulation.py` | implemented | `examples/nest/repeated_stimulation.py` | gated `poisson_generator` over repeated trials; active-window spike count `CAT_D` 5 %; zero-rate → silent |
| `sensitivity_to_perturbation.py` | implemented | `examples/nest/sensitivity_to_perturbation.py` | Brunel-style balanced net; AI rate 14.95 vs 15.17 Hz (1.45 %, `CAT_D`); 1-spike perturbation decorrelates >0.9 of net after `t_stim`, 0 before (both sims) |
| `wang_decision_making.py` | implemented | `examples/nest/wang_decision_making.py` | Wang (2002) WTA decision network on `iaf_bw_2001`. Recurrent NMDA via `connect(receptor_type=NMDA, comm='dense')` matches live NEST to machine precision (~5e-15; `iaf_bw_2001_recurrent_nmda_parity_test.py`) — **design A resolved to option (a)** (generalize the presynaptic-emission seam; no offset-aware event projection needed). Decision parity is distributional (`wang_decision_making_test.py`): ±coherence→A/B both sims, winner > 2.5× loser (< 4 Hz), unbiased at 0; the WTA attractor amplifies integrator/PRNG differences so the winner's absolute rate differs (BP A~12 vs NEST A~7 Hz). No-NEST companion `wang_decision_making_no_nest_test.py` |
| `lin_rate_ipn_network.py` | missing | none | substrate-ready (cluster 15a): `lin_rate` + rate connections on seam-(H); demo port pending |
| `rate_neuron_dm.py` | missing | none | substrate-ready (cluster 15a): rate-network decision-making on the rate core; demo port pending |
| `gap_junctions_two_neurons.py` | missing | none | out of scope (cluster 14): `hh_psc_alpha_gap` + `gap_junction` coupling |
| `gap_junctions_inhibitory_network.py` | missing | none | out of scope (cluster 14): gap-junction coupling |
| `intrinsic_currents_spiking.py` | missing | none | out of scope (cluster 14): `ht_neuron` intrinsic currents |
| `intrinsic_currents_subthreshold.py` | missing | none | out of scope (cluster 14): `ht_neuron` |

### 3.7 Generator-pattern demos

All three ported in cluster 16 with live-NEST **distributional** parity tests plus
no-NEST companions. The two sinusoidal generators are `for_loop`-traceable, so each is
driven directly by `brainstate.transform.for_loop` over a single `in_size=N` instance
(the loop primitive `Simulator.simulate` uses internally) — which both exercises the
`individual_spike_trains` flag and lowers the rollout into one compiled program. The
host-side `pulsepacket_generator` (NumPy RNG + `deque` queues, not JAX-traceable) is
driven by an eager host loop, then its packet is replayed through the `Simulator` as a
`SpikeTime` population (`one_to_one`) for the membrane drive. Parity is distributional
because NEST's per-thread RNG diverges from the JAX/NumPy streams: each test compares a
seed-aggregated statistic within a documented band, plus the qualitative law the demo
exists to show (PSTH tracks `λ(t)`; `CV → 1/√m`; packet width ∝ `sdev`).

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `sinusoidal_gamma_generator.py` | implemented | `examples/nest/sinusoidal_gamma_generator.py` | gamma renewal order `m`; ISI-CV → 1/√m (seed-mean `CAT_D` width 6 %, qualitative law within 12 %) + modulated-rate autocorr `CAT_D`; eager `for_loop` drive. Tests clear JAX caches per case (the gamma `update()` lowers a costly `while_loop`-in-`scan` that otherwise accumulates) |
| `sinusoidal_poisson_generator.py` | implemented | `examples/nest/sinusoidal_poisson_generator.py` | inhomogeneous Poisson; PSTH tracks `λ(t)` + spike-count autocorr `CAT_D`; `individual_spike_trains` (independent vs shared) modes; eager `for_loop` drive |
| `pulsepacket.py` | implemented | `examples/nest/pulsepacket.py` | host-side `pulsepacket_generator` via eager host loop; packet width = `sdev` (seed-mean `CAT_D`) + per-step profile (smoothed corr > 0.93, mass 8 %, centroid < 1 ms); averaged-V_m vs analytical Gaussian⊛PSP (NEST-free, windowed corr 0.9998) through a `SpikeTime → iaf_psc_alpha` Simulator drive |

### 3.8 Astrocyte demos

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `astrocytes/` (subdir, contains `astrocyte_brunel_*` series) | missing (deferred) | none | uses `astrocyte_lr_1994` + `aeif_cond_alpha_astro` (both validated) but also the **bucket-3 `sic_connection`** + an astrocyte rate model. Explicitly deferred past cluster 16 (generator demos) to the post-bucket-3 window; the astrocyte Brunel variant is then the natural P0 port target. |

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
- **Network demos**: gap junctions and rate networks only (both out of scope —
  gap-junction coupling / rate-neuron connections). Wang 2002, Brette 2007,
  EI-clustered, perturbation-sensitivity, artificial-synchrony and
  repeated-stimulation are ported (§3.6).
- **Generator demos**: sinusoidal Poisson + gamma and pulse packets are ported (§3.7).
- **Astrocyte demos**: the `astrocytes/` Brunel-variant series (deferred to post-bucket-3,
  pending `sic_connection` + an astrocyte rate model).
- **Spatial and SONATA**: blocked by API absence; classified `unsupported`
  until the underlying API lands.

## 5. Semantic & numerical risks

- **Each port is also a validation harness.** A ported Brunel example *is*
  an end-to-end NEST-comparison test (firing rate, mean ISI, CV of ISI).
  Skipping the example means losing the regression — and losing the
  promotion-blocking parity check.
- **`aeif_cond_beta_multisynapse` multi-receptor broadcasting (`n>1`).** The
  `aeif_cond_beta_multisynapse.py` port (§3.5) drives a **single** neuron
  (`n=1`). With `n>1` neurons *and* multiple receptor ports, the per-port
  conductance state does not broadcast correctly against the population axis
  (the receptor axis and the neuron axis collide), so a multi-neuron
  multi-receptor AdEx population is not yet trustworthy. Single-neuron
  multi-receptor traces are exact; the limitation is purely the `n>1 ×
  receptors` shape. Fix belongs with the `aeif` model, not the example.
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

- **P2 — Port `wang_decision_making.py`.** [L] — **DONE (cluster 22).** The
  `iaf_bw_2001` recurrent-NMDA seam was generalized (design A resolved to option
  (a): the graded presynaptic emission over `connect(receptor_type=NMDA,
  comm='dense')` matches live NEST to ~5e-15), and the Wang WTA decision network was
  ported with distributional live-NEST parity (±coherence→A/B, winner ≫ loser,
  unbiased at 0). See §3.6.

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
