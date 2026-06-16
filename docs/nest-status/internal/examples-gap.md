# Examples portfolio — NEST parity gap

## 1. Scope

PyNEST example portfolio at `nest/nest-simulator/pynest/examples/` vs.
`brainpy.state`'s `docs/examples/gallery.rst` (and the colocated `examples/`
folder it links to). Goal: a porting-target list ordered by priority.

Upstream reference:
<https://nest-simulator.readthedocs.io/en/stable/examples/index.html>
(listing also obtained from GitHub: `gh api repos/nest/nest-simulator/contents/pynest/examples`).

Evidence basis:
- `examples/nest/` now holds **~75 NEST-style port scripts** (clusters 00–28),
  each paired with a live-NEST (or analytic) parity test under
  `brainpy_state/_nest/_validation/`. This is the porting-target list, realized.
- `docs/examples/gallery.rst` (5.7 KB) still lists only the original 14
  brainpy-style examples (classical network models, oscillations, SNN
  training) under their native compositional API — the `examples/nest/` ports
  are **not yet wired into the rendered gallery** (the residual; see
  `docs-portfolio-gap.md`).
- Upstream NEST examples list (~70 top-level scripts + ~10 subdirectories,
  retrieved 2026-05-11) covers: Brunel family, balanced random networks,
  COBA/CUBA, gap junctions, GIF/GLIF demos, HH demos, IAF Tum 2000 STP
  demos, plasticity demos (Clopath, STDP, STP, Urbanczik), correlation
  demos, multimeter recording demos, spatial subdir, astrocyte subdir,
  e-prop subdir, SONATA subdir, HPC benchmark, pong, sudoku.

## 2. Parity summary

The example-porting work is essentially complete: **~75 NEST-style scripts now
live in `examples/nest/`**, spanning the flagship Brunel family, single-neuron
pedagogy, plasticity, recording, generator, spatial, astrocyte, rate, and
compartmental demos, plus pong and sudoku — each backed by a live-NEST (or
analytic) parity test. The residual is twofold: (1) **wiring these ports into
the rendered `gallery.rst`** so NEST users can discover them (a docs task, not
a porting task — `docs-portfolio-gap.md`), and (2) a small **out-of-scope
tail** (e-prop → `braintrace`; `ht_neuron` intrinsic-currents; MUSIC / SONATA /
HPC / structural-plasticity by design).

| Bucket | Count | Notes |
|---|---:|---|
| implemented (ported + reproducing NEST result) | ~53 | clusters 00–28: flagship Brunel family + single-neuron / plasticity / recording / generator / spatial / astrocyte / rate / compartmental / sudoku / pong demos, each with a live-NEST (or analytic) parity test |
| partial (concept covered by a non-NEST-style example) | ~0 | the former `gallery.rst` COBA / CUBA / E-I overlaps are now superseded by the direct Brunel ports below |
| missing | ~10 | out-of-scope tail only: the **e-prop** subdir (ported in `braintrace`), `ht_neuron` intrinsic-currents (cluster-14 deferral), and the optional `spatial/` tutorial variants (`conncon_*` / `connex*` / `grid_iaf_oc` / `test_3d`) over now-present primitives |
| unsupported | ~5 | MUSIC examples (`music_cont_out_proxy_example/`), SONATA (`sonata_example/`), structural plasticity (`structural_plasticity.py`), HPC benchmark (`hpc_benchmark.py` requires MPI), `store_restore_network.py` (kernel-state serialization) |
| **upstream NEST example scripts surveyed** | **≈ 70 + 10 subdirs** | per the gh-api listing 2026-05-11 |

## 3. Evidence-backed mapping table

Status legend: `missing` = no port in repo, `partial` = a conceptually similar
brainpy-style example exists, `implemented` = a port reproducing NEST's result
exists.

### 3.1 Flagship benchmarks (porting priority order)

| NEST example | Status | brainpy.state equivalent | NEST upstream | Notes |
|---|---|---|---|---|
| `brunel_alpha_nest.py` | implemented | `examples/nest/brunel_alpha.py` (+ `_validation/brunel_alpha_test.py`) | <https://nest-simulator.readthedocs.io/en/stable/auto_examples/brunel_alpha_nest.html> | The single most-cited NEST example (`iaf_psc_alpha` + `poisson_generator` + `spike_recorder`); the Phase-1 flagship port with live-NEST parity (sparse CSR comm path at `order=2500`, PR #39/#40) |
| `brunel_delta_nest.py` | implemented | `examples/nest/brunel_delta.py` (+ `_validation/brunel_delta_test.py`) | upstream | Delta-current variant of Brunel, live-NEST parity |
| `brunel_exp_multisynapse_nest.py` | implemented | `examples/nest/brunel_exp_multisynapse.py` (+ `_validation/brunel_exp_multisynapse_test.py`) | upstream | Multi-port AdEx variant (cross-link `neurons-gap.md` multisynapse), live-NEST parity |
| `brunel_siegert_nest.py` | implemented | `examples/nest/brunel_siegert.py` | upstream | Mean-field Brunel: `siegert_neuron` + dual-channel `diffusion_connection`, relaxed end-to-end through the `Simulator`; 32.03 vs NEST 32.03 spks/s (0.00 %) and matches the closed-form Siegert fixed point to ~3e-13 (cluster 15c; §3.6) |
| `brunel_alpha_evolution_strategies.py` | implemented | `examples/nest/brunel_alpha_evolution_strategies.py` (+ `_validation/brunel_alpha_evolution_strategies_test.py`) | upstream | Optimizer-tuned Brunel, ported with parity test |

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
| `plot_weight_matrices.py` | implemented | `examples/nest/plot_weight_matrices.py` (+ `_validation/plot_weight_matrices_test.py`) | unblocked by cluster `23` (`get_connections`→`SynapseCollection` `.get`/`.set`); the only runtime skip is a benign matplotlib guard |
| `synapsecollection.py` | implemented | `examples/nest/synapsecollection.py` (+ `_validation/synapsecollection_test.py`) | unblocked by cluster `23` (`SynapseCollection` lazy view); `.get`/`.set` over the projection edge State |

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
| `iaf_tum_2000_short_term_depression.py` | implemented | `examples/nest/iaf_tum_2000_short_term_depression.py` (+ `_validation/iaf_tum_2000_stp_test.py`) | LIF + integrated STP, depression regime; parity via the shared `iaf_tum_2000_stp` test |
| `iaf_tum_2000_short_term_facilitation.py` | implemented | `examples/nest/iaf_tum_2000_short_term_facilitation.py` (+ `_validation/iaf_tum_2000_stp_test.py`) | LIF + integrated STP, facilitation regime; parity via the shared `iaf_tum_2000_stp` test |
| `mc_neuron.py` | implemented | `examples/nest/mc_neuron.py` (+ `_validation/mc_neuron_test.py`) | three-compartment demo (`iaf_cond_alpha_mc`), live-NEST parity |
| `BrodyHopfield.py` | implemented | `examples/nest/BrodyHopfield.py` (+ `_validation/BrodyHopfield_test.py`) | spike-coding network, parity test |
| `CampbellSiegert.py` | implemented | `examples/nest/CampbellSiegert.py` (+ `_validation/CampbellSiegert_test.py`) | mean-field cross-check (analytic carve-out) |

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
(`τ Ẋ = −λX + μ + φ(h)`, linear-rate FP `(I−gC)⁻¹μ` parity vs NEST), and the two
rate-network demos are now **ported with live-NEST parity (cluster 17)**:
`lin_rate_ipn_network` (delayed-E/instantaneous-I linear-rate net; closed-form **and** NEST
fixed point `(λI−W)⁻¹μ`, trajectory with `align_steps`) and `rate_neuron_dm` (two-unit
rectified winner-take-all decision; deterministic winner `10·μ_win` matched to NEST, plus
distributional decision-direction/contrast/zero-bias parity) — rows below. The **gap-junction
substrate has since landed too (cluster 15b)**: both `gap_junctions_*` demos are now ported
with live-NEST parity (rows below), realized as an explicit one-step-lagged **difference
current** `I_gap,i = Σ_j g_ij (V_j[n−1] − V_i[n−1])` deposited into the post current channel
on the SAME seam-(H) emission path (the V emission holder), NEST's `use_wfr=False` regime —
no waveform relaxation. With the rate, gap-junction, and Siegert ports landed, **§3.6 is
now complete except the `ht_neuron` (`intrinsic_currents_*`) demos**, which remain blocked
on their own intrinsic-currents primitive (single-neuron, not continuous network coupling).

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
| `lin_rate_ipn_network.py` | implemented | `examples/nest/lin_rate_ipn_network.py` | E/I `lin_rate_ipn` net, delayed-E + instantaneous-I (`fixed_outdegree`→mean-field `fixed_indegree`). Deterministic FP arbiter matches closed form **and** NEST `(λI−W)⁻¹μ` (atol 1e-3, `use_wfr=False`); per-neuron trajectory matches NEST with `align_steps=12` (`lin_rate_ipn_network_test.py`, cluster 17) |
| `rate_neuron_dm.py` | implemented | `examples/nest/rate_neuron_dm.py` | two mutually-inhibiting `lin_rate_ipn` units → rectified WTA decision. Deterministic winner `10·μ_win` (11.0) + loser 0 match NEST exactly (R1 arbiter for `rectify_output` in the recurrent path); distributional parity — strong-bias direction 5/5 & 0/5 both sims, winner-loser contrast, zero-bias both-win balance (`rate_neuron_dm_test.py`, cluster 17) |
| `gap_junctions_two_neurons.py` | implemented | `examples/nest/gap_junctions_two_neurons.py` | gap-coupled `hh_psc_alpha_gap` pair synchronizes (g=0.5 nS, resting-gating ICs). Explicit-lag difference deposit (`use_wfr=False`); 2-neuron micro-parity matches live NEST to machine precision between spikes (median ~1e-3 mV, p95 ~0.1 mV), only O(dt) AP-edge jitter (`gap_junction_parity_test.py`). No-NEST companion `gap_junction_no_nest_test.py` (cluster 15b) |
| `gap_junctions_inhibitory_network.py` | implemented | `examples/nest/gap_junctions_inhibitory_network.py` | inhibitory `hh_psc_alpha_gap` net + random symmetric gap graph; Golomb-Rinzel coherence χ rises with gap weight (async ~0.14 → sync ~0.36, ~2.6× on BOTH sims), distributional live-NEST parity (`gap_junction_inhibitory_network_parity_test.py`, 4 seeds, χ within a few %). `fixed_indegree` gap graph (cluster 15b) |
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

The **neuron↔astrocyte SIC loop substrate landed in cluster 15d** (the last bucket-3
*model* cluster; only the Siegert `diffusion_connection` remains queued → 15c).
`astrocyte_lr_1994` now emits its slow-inward current as seam-(H)
continuous graded emission (`_emission_attr='SIC'`, `_emission_current=True`); a one-way
`sic_connection` (sender/receiver-enforced `astrocyte_lr_1994 → aeif_cond_alpha_astro`)
deposits `weight·SIC` into the neuron's labelled `'I_SIC'` current channel through an
`as_current` `EventProjection`; the neuron→astrocyte arm is the ordinary delta path
(spikes → `Δ_IP3·w` IP3 via `sum_delta_inputs`). The host-side `_sic_queue` event-emulator
and `sic_connection`'s host-queue coeff-array API were **deleted** (the bucket-3 de-queue),
so the whole bidirectional loop lowers under `Simulator.simulate`'s `for_loop`. Live-NEST
parity is near-exact (`astrocyte_sic_test.py`): SIC-response micro IP3 `2.4e-5`/Ca
`1.9e-4`/I_SIC `2.3e-4`; driven loop IP3 `0`/Ca `1e-6`/I_SIC `6e-4`/V_pre `3.7e-4 mV`;
astro-network seed-mean post rate `9.0=9.0` (SIC off) → `14.0=14.0` (SIC on). The
`sic_connection` default delay `1.0 ms` maps to `delay_steps=10`.

The single-astrocyte and tripartite-interaction demos are now **ported** (cluster 17b)
to `examples/nest/`, each with a live-NEST parity test. Porting `astrocyte_interaction`
surfaced — and **fixed** — a latent gap: `aeif_cond_alpha_astro` could not receive
ordinary excitatory/inhibitory **spike** input into its synaptic conductance through the
`Simulator` (it self-pulls `label='w_ex'/'w_in'` but had no `n_receptors`/`w_by_rec`
bridge to populate them), so the demo's Poisson drive left the presynaptic neuron at
`E_L`. The model now exposes the multi-receptor bridge (`receptor_type=1`→`g_ex`,
`=2`→`g_in`, positive nS — NEST's weight-sign routing), parity-validated in
`aeif_cond_alpha_astro_test.py` (V_m/g_ex/g_in ~1e-6 vs live NEST). The same self-pull-only
gap remains in sibling conductance neurons (`aeif_cond_alpha`, `aeif_cond_exp`,
`iaf_cond_alpha`, `iaf_cond_beta`, `gif_cond_exp`, …) — tracked in `neurons-gap.md` as a
follow-up.

The `small_network` / `astrocyte_brunel_*` variants needed NEST's `TripartiteConnect`
astrocyte-pool rule (`third_factor_bernoulli_with_pool`), which **landed in cluster 24**:
`Simulator.tripartite_connect(pre, post, third, conn_spec, third_factor_conn_spec,
syn_specs)` samples the primary `pre→post` edges **once** and shares that realization
across all three arms — primary (`pre→post`), `third_in` (`pre→astro`), `third_out`
(`astro→post`, the `sic_connection`) — reusing the existing static + `sic_connection`
(15d) paths with **no new deposit primitive** (an internal `_ExplicitEdges` rule feeds
the shared sample into the ordinary `_connect_pair`/`_connect_sic`). The pool sampler
`third_factor_bernoulli_with_pool(p, pool_size, pool_type)` supports both `block`
(non-overlapping) and `random` pools. Design was arbitrated by a live-NEST micro-parity
(`tripartite_connect_test.py`): the deterministic **block** edge sets are *bit-identical*
to `nest.TripartiteConnect` (n2n/n2a/a2n), and **random** pools match seed-mean
distinct-edge counts within category D. All three demos are now **ported** with live-NEST
parity. Synapse divergence: the ports use **static** synapses on the primary/third_in arms
(the 15d-validated SIC loop) where NEST's demos use `tsodyks_synapse`; the connectivity is
identical, so the parity is unaffected.

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `astrocyte_single.py` | **implemented** | `examples/nest/astrocyte_single.py` | one `astrocyte_lr_1994` + Poisson → IP3/Ca, plus a downstream `aeif_cond_alpha_astro` to expose `I_SIC`. Live-NEST parity IP3/Ca/I_SIC (`astrocyte_single_test.py`, 17b) |
| `astrocyte_interaction.py` | **implemented** | `examples/nest/astrocyte_interaction.py` | tripartite two-neuron + one-astrocyte SIC loop; faithful Poisson drive (spike→conductance fix). Live-NEST parity V_pre/IP3/Ca/I_SIC (`astrocyte_interaction_test.py`, 17b) |
| `astrocyte_small_network.py` | **implemented** | `examples/nest/astrocyte_small_network.py` | `tripartite_connect` (`third_factor_bernoulli_with_pool`, pool_size=1/block, 24). Deterministic; live-NEST loop-trace parity V_pre/IP3/Ca/I_SIC (`astrocyte_small_network_test.py`) |
| `astrocyte_brunel_bernoulli.py` | **implemented** | `examples/nest/astrocyte_brunel_bernoulli.py` | Brunel + astrocytes via `tripartite_connect` (pool_size=10/random, `pairwise_bernoulli` primary, 24); one sliced neuron pop (E=`neurons[:N_ex]`). Live-NEST connectivity-distributional parity (`astrocyte_brunel_test.py`) |
| `astrocyte_brunel_fixed_indegree.py` | **implemented** | `examples/nest/astrocyte_brunel_fixed_indegree.py` | as above, `fixed_indegree` primary rule (24). Shares `astrocyte_brunel_bernoulli.build`; live-NEST connectivity-distributional parity |

### 3.9 Spatial demos

The spatial API landed (goal 20): `brainpy.state.spatial.*` (layers, distance, `gaussian`
kernel, masks, `spatial_pairwise_bernoulli`, query helpers) + `Simulator.create(positions=)`
and `get_position`. Four representative demos are ported with live-NEST parity (NEST 3.9.0);
the remaining `spatial/` tutorials are variants over the same primitives. See `network-api-gap.md`
§3.10 for the API-surface status and `examples/nest/README.md` §3.9.

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `spatial/grid_iaf.py` | done | `examples/nest/spatial_grid_iaf.py` | 4×3 grid; exact coord + centre parity vs NEST `GetPosition`/`FindCenterElement` |
| `spatial/gaussex.py` | done | `examples/nest/spatial_gaussex.py` | Gaussian kernel + circular mask; empirical `p(d)` matches NEST bin-by-bin (max\|Δ\|≈0.016) |
| `spatial/test_3d_gauss.py` | done | `examples/nest/spatial_3d_gauss.py` | 3-D free layer + box mask + no autapses; curve parity (max\|Δ\|≈0.008) |
| `csa_spatial_example.py` | done (native) | `examples/nest/spatial_csa.py` | CSA Gaussian re-expressed as `spatial_pairwise_bernoulli(gaussian, circular)` — no libneurosim |
| `csa_example.py` | placeholder | `examples/nest/csa_example.py` | CSA/`conngen` mechanism not ported; documents `csa.random(0.1)` → `pairwise_bernoulli(0.1)` |
| `spatial/` (other tutorials) | missing | none | `conncon_*`, `connex*`, `grid_iaf_oc`, `test_3d`, … — variants over the now-present primitives |

### 3.10 Pedagogical / advanced

| NEST example | Status | brainpy.state equivalent | Notes |
|---|---|---|---|
| `compartmental_model/` (subdir) | implemented | `examples/nest/two_comps.py`, `examples/nest/receptors_and_current.py` | dendritic-tree models on `cm_default` (active vs passive dendrite; per-compartment AMPA / NMDA / GABA + DC). Live-NEST `v_comp` / gating / `g_r,g_d` parity + NEST-free dendritic-amplification laws |
| `pong/` (subdir) | implemented | `examples/nest/pong.py`, `pong_networks.py`, `pong_run.py` | RL demo: `PongNetRSTDP` (host R-STDP on static synapses) + `PongNetDopa` (dopaminergic actor–critic) on the `Simulator.cont` / `host_drive` persistent-rollout substrate. Component-deterministic parity (`calculate_stdp` vs live NEST; dopamine reward→potentiation pathway) + bounded behavioural learning |
| `sudoku/` (subdir) | implemented | `examples/nest/sudoku.py`, `sudoku_net.py`, `sudoku_puzzles.py` | stochastic WTA constraint-satisfaction: 3645 `iaf_psc_exp` neurons (729 pops × 5), row/col/box/cell inhibition (510 300 edges) as **one** sparse `explicit_edges` projection, 350 Hz background noise + per-clue 200 Hz parrot-relay stimulation, host `cont(100 ms)` relaxation loop. **Distributional (documented-partial) parity** (§3.14 posture): matches live NEST's solve rate on a near-complete board (both ~100 % over seeds); on a hard board (puzzle 4, NEST's default) neither solver completes within a practical chunk budget and brainpy tracks NEST's best fraction-correct. The earlier "intractable" verdict was a unit bug (pF/pA vs nF/nA), not a substrate limit |
| `structural_plasticity.py` | unsupported | none | spec §7 (no structural plasticity) |
| `store_restore_network.py` | unsupported | none | kernel-state serialization is NEST-internal |
| `music_cont_out_proxy_example/` | unsupported | none | spec §7 (MUSIC) |
| `sonata_example/` | unsupported | none | spec §7 (SONATA) |
| `brette_gerstner_fig_2c.py`, `brette_gerstner_fig_3d.py` | implemented | `examples/nest/brette_gerstner_fig_2c.py`, `brette_gerstner_fig_3d.py` | AdEx pedagogical figures: spike-frequency adaptation (`aeif_cond_alpha`) + post-inhibitory rebound (`aeif_cond_exp`). Sub-threshold `V_m` parity (CAT_A, < 1e-3 mV) + spike-pattern (CAT_E) |

## 4. Missing or incomplete functionality

The portfolio is substantially ported (~75 scripts, §3). The residual is a
short tail:

- **`ht_neuron` intrinsic-currents demos** — out of scope: the Hill–Tononi
  primitive is not ported (cluster-14 deferral). Other intrinsic-currents
  pedagogy is covered.
- **Optional `spatial/` tutorial variants** (`conncon_*`, `connex*`,
  `grid_iaf_oc`, `test_3d`) — the underlying spatial primitives all ship
  (clusters 20/27, §3.9); these are extra tutorial scripts over them, not new
  capability.
- **By design unsupported** (not gaps): the HPC benchmark (`hpc_benchmark.py`,
  requires MPI), `structural_plasticity.py`, the MUSIC examples, the SONATA
  subdir, and `store_restore_network.py` (kernel-state serialization).

The e-prop subdir is ported in the sibling **`braintrace`** package, not here.

## 5. Semantic & numerical risks

- **Each port is also a validation harness.** A ported Brunel example *is*
  an end-to-end NEST-comparison test (firing rate, mean ISI, CV of ISI); the
  parity tests live in `brainpy_state/_nest/_validation/` (140 files, 120
  live-NEST). The risk is upkeep: a dropped port is a dropped regression.
- **`aeif_cond_beta_multisynapse` multi-receptor broadcasting (`n>1`).** The
  `aeif_cond_beta_multisynapse.py` port (§3.5) drives a **single** neuron
  (`n=1`). With `n>1` neurons *and* multiple receptor ports, the per-port
  conductance state does not broadcast cleanly against the population axis
  (receptor axis × neuron axis collide), so a multi-neuron multi-receptor AdEx
  population is not yet trustworthy. Single-neuron multi-receptor traces are
  exact; the limitation is the `n>1 × receptors` shape and belongs with the
  `aeif` model, not the example.
- **Multimeter file-backend step.** `multimeter_file.py` and `recording_demo.py`
  are ported, but NEST's `ascii` / `sionlib` recording backends are not
  implemented (`devices-gap.md` P2); those ports skip the file-backend step and
  record in-memory instead.

Resolved risks (previously listed here, now closed):

- **Plasticity volume-transmitter parity** — `stdp_dopamine_synapse` +
  `volume_transmitter` are now validated against live NEST
  (`_validation/{stdp_dopamine_synapse,volume_transmitter}_parity_test.py`).
- **Spatial examples** — no longer blocked: the spatial API shipped (clusters
  20/27) and the spatial demos are ported (§3.9).
- **`structural_plasticity.py` / HPC benchmark** — genuinely unsupported by
  design (structural-plasticity primitive / MPI), not gaps.

## 6. Validation gaps

The examples-as-validation principle is now realized: the NEST ports are backed
by `brainpy_state/_nest/_validation/` (140 files, 120 `requires_nest` live-NEST
tests + analytic checks), covering the flagship Brunel family, plasticity
learning curves, the correlation/detector regression (`cross_check_mip_corrdet.py`
is ported, §3.4), and the generator / spatial / astrocyte demos.

Residual validation gaps (full list in `numerical-validation-gap.md`):

- **`pp_psc_delta`** — model present (`pp_psc_delta.py`), but no dedicated
  `_validation` parity test yet.

The astrocyte path is fully covered: `sic_connection` + `astrocyte_lr_1994` +
`aeif_cond_alpha_astro` each have `_validation` parity tests
(`astrocyte_{sic,brunel,interaction,single,small_network}_test.py`).
