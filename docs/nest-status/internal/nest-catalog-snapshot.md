# NEST 3.x Catalog Snapshot

**Retrieved:** 2026-05-11
**NEST version target:** 3.x (latest stable per `https://nest-simulator.readthedocs.io/en/stable/`)
**Sources:**
- Models index: <https://nest-simulator.readthedocs.io/en/stable/models/index.html>
- Connectivity concepts: <https://nest-simulator.readthedocs.io/en/stable/synapses/connectivity_concepts.html>
- PyNEST API listing: <https://nest-simulator.readthedocs.io/en/stable/ref_material/pynest_api/index.html>
- Spatially structured networks: <https://nest-simulator.readthedocs.io/en/stable/networks/spatially_structured_networks.html>

**Purpose:** Frozen baseline for the gap analysis in this directory. See
`./index.md` for the gap-analysis overview and refresh procedure.

---

## 1. Neurons

| Model | Class | Description | Upstream doc |
|---|---|---|---|
| `aeif_cond_alpha` | AdEx conductance, alpha PSC | Conductance-based exponential integrate-and-fire | https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_alpha.html |
| `aeif_cond_alpha_astro` | AdEx + astrocyte | Conductance-based AdEx with neuron-astrocyte interactions | https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_alpha_astro.html |
| `aeif_cond_alpha_multisynapse` | AdEx conductance, multi-port | AdEx with multiple synaptic receptor channels (alpha) | https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_alpha_multisynapse.html |
| `aeif_cond_beta_multisynapse` | AdEx conductance, multi-port | AdEx with multiple receptor channels (beta) | https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_beta_multisynapse.html |
| `aeif_cond_exp` | AdEx conductance, exp PSC | Conductance-based AdEx | https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_exp.html |
| `aeif_psc_alpha` | AdEx current, alpha PSC | Current-based AdEx | https://nest-simulator.readthedocs.io/en/stable/models/aeif_psc_alpha.html |
| `aeif_psc_delta` | AdEx current, delta PSC | AdEx with delta-shaped PSCs | https://nest-simulator.readthedocs.io/en/stable/models/aeif_psc_delta.html |
| `aeif_psc_delta_clopath` | AdEx + Clopath traces | AdEx supporting Clopath voltage-based STDP | https://nest-simulator.readthedocs.io/en/stable/models/aeif_psc_delta_clopath.html |
| `aeif_psc_exp` | AdEx current, exp PSC | Current-based AdEx | https://nest-simulator.readthedocs.io/en/stable/models/aeif_psc_exp.html |
| `amat2_psc_exp` | Multi-timescale adaptive threshold | Non-resetting LIF with adaptive threshold | https://nest-simulator.readthedocs.io/en/stable/models/amat2_psc_exp.html |
| `cm_default` | Multi-compartment | User-defined dendrites with AMPA / GABA / AMPA+NMDA receptors | https://nest-simulator.readthedocs.io/en/stable/models/cm_default.html |
| `eprop_iaf` | e-prop LIF | LIF for e-prop plasticity (delta PSC) | https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf.html |
| `eprop_iaf_adapt` | e-prop LIF + adapt | LIF with threshold adaptation for e-prop | https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_adapt.html |
| `eprop_iaf_adapt_bsshslm_2020` | e-prop (Bellec 2020) | LIF with adapt for the Bellec et al. 2020 e-prop formulation | https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_adapt_bsshslm_2020.html |
| `eprop_iaf_bsshslm_2020` | e-prop (Bellec 2020) | LIF for the Bellec et al. 2020 e-prop formulation | https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_bsshslm_2020.html |
| `eprop_iaf_psc_delta` | e-prop LIF | LIF (delta PSC) for e-prop | https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_psc_delta.html |
| `eprop_iaf_psc_delta_adapt` | e-prop LIF + adapt | LIF (delta PSC) with threshold adaptation for e-prop | https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_psc_delta_adapt.html |
| `eprop_readout` | e-prop readout | Leaky integrator readout for e-prop | https://nest-simulator.readthedocs.io/en/stable/models/eprop_readout.html |
| `eprop_readout_bsshslm_2020` | e-prop readout (Bellec 2020) | Readout for the Bellec et al. 2020 formulation | https://nest-simulator.readthedocs.io/en/stable/models/eprop_readout_bsshslm_2020.html |
| `erfc_neuron` | Binary stochastic | erfc activation function | https://nest-simulator.readthedocs.io/en/stable/models/erfc_neuron.html |
| `gauss_rate` | Rate | Gaussian gain function | https://nest-simulator.readthedocs.io/en/stable/models/gauss_rate.html |
| `gif_cond_exp` | GIF conductance | Conductance-based generalized IF | https://nest-simulator.readthedocs.io/en/stable/models/gif_cond_exp.html |
| `gif_cond_exp_multisynapse` | GIF conductance, multi-port | GIF with multiple synaptic time constants | https://nest-simulator.readthedocs.io/en/stable/models/gif_cond_exp_multisynapse.html |
| `gif_pop_psc_exp` | GIF population | Population of GIF neurons with exp PSCs and adaptation | https://nest-simulator.readthedocs.io/en/stable/models/gif_pop_psc_exp.html |
| `gif_psc_exp` | GIF current | Current-based generalized IF | https://nest-simulator.readthedocs.io/en/stable/models/gif_psc_exp.html |
| `gif_psc_exp_multisynapse` | GIF current, multi-port | GIF with multiple synaptic time constants | https://nest-simulator.readthedocs.io/en/stable/models/gif_psc_exp_multisynapse.html |
| `ginzburg_neuron` | Binary stochastic | Sigmoidal activation | https://nest-simulator.readthedocs.io/en/stable/models/ginzburg_neuron.html |
| `glif_cond` | GLIF conductance | Conductance-based generalized leaky IF | https://nest-simulator.readthedocs.io/en/stable/models/glif_cond.html |
| `glif_psc` | GLIF current | Current-based generalized leaky IF | https://nest-simulator.readthedocs.io/en/stable/models/glif_psc.html |
| `glif_psc_double_alpha` | GLIF current, double alpha | GLIF with double alpha-function PSC | https://nest-simulator.readthedocs.io/en/stable/models/glif_psc_double_alpha.html |
| `hh_cond_beta_gap_traub` | Hodgkin-Huxley conductance | HH with gap junction + beta-function conductances | https://nest-simulator.readthedocs.io/en/stable/models/hh_cond_beta_gap_traub.html |
| `hh_cond_exp_traub` | Hodgkin-Huxley conductance | HH for Brette et al. (2007) review | https://nest-simulator.readthedocs.io/en/stable/models/hh_cond_exp_traub.html |
| `hh_psc_alpha` | Hodgkin-Huxley current | HH neuron, alpha PSC | https://nest-simulator.readthedocs.io/en/stable/models/hh_psc_alpha.html |
| `hh_psc_alpha_clopath` | HH + Clopath | HH with Clopath plasticity traces | https://nest-simulator.readthedocs.io/en/stable/models/hh_psc_alpha_clopath.html |
| `hh_psc_alpha_gap` | HH + gap junctions | HH with gap-junction support | https://nest-simulator.readthedocs.io/en/stable/models/hh_psc_alpha_gap.html |
| `ht_neuron` | Hill-Tononi (2005) | Hill-Tononi thalamocortical model | https://nest-simulator.readthedocs.io/en/stable/models/ht_neuron.html |
| `iaf_bw_2001` | Brunel-Wang (2001), simplified | LIF + conductance + NMDA, simplified dynamics | https://nest-simulator.readthedocs.io/en/stable/models/iaf_bw_2001.html |
| `iaf_bw_2001_exact` | Brunel-Wang (2001), exact | LIF + conductance + extended NMDA | https://nest-simulator.readthedocs.io/en/stable/models/iaf_bw_2001_exact.html |
| `iaf_chs_2007` | Carandini-Heeger spike response | Spike-response model (Carandini 2007) | https://nest-simulator.readthedocs.io/en/stable/models/iaf_chs_2007.html |
| `iaf_chxk_2008` | LIF conductance, precise | Conductance LIF with precise spike times | https://nest-simulator.readthedocs.io/en/stable/models/iaf_chxk_2008.html |
| `iaf_cond_alpha` | LIF conductance, alpha | Simple conductance LIF | https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_alpha.html |
| `iaf_cond_alpha_mc` | LIF conductance, multi-compartment | Multi-compartment conductance LIF | https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_alpha_mc.html |
| `iaf_cond_beta` | LIF conductance, beta | Conductance LIF with beta-function synapses | https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_beta.html |
| `iaf_cond_exp` | LIF conductance, exp | Conductance LIF, exponential PSCs | https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_exp.html |
| `iaf_cond_exp_sfa_rr` | LIF conductance + SFA + RR | Conductance LIF with spike-frequency adaptation + relative refractory | https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_exp_sfa_rr.html |
| `iaf_psc_alpha` | LIF current, alpha | LIF with alpha-shaped PSCs | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_alpha.html |
| `iaf_psc_alpha_multisynapse` | LIF current, multi-port | LIF with multiple receptor ports (alpha) | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_alpha_multisynapse.html |
| `iaf_psc_alpha_ps` | LIF current, precise spike | LIF alpha with regula-falsi precise spike timing | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_alpha_ps.html |
| `iaf_psc_delta` | LIF current, delta | LIF with delta-shaped PSCs | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_delta.html |
| `iaf_psc_delta_ps` | LIF current, precise spike | LIF delta, precise spike timing | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_delta_ps.html |
| `iaf_psc_exp` | LIF current, exp | LIF with exponential PSCs | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp.html |
| `iaf_psc_exp_htum` | LIF current, htum | LIF with separate relative + absolute refractory | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_htum.html |
| `iaf_psc_exp_multisynapse` | LIF current, multi-port | LIF exp with multiple receptor ports | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_multisynapse.html |
| `iaf_psc_exp_ps` | LIF current, precise spike | LIF exp with regula-falsi precise spike timing | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_ps.html |
| `iaf_psc_exp_ps_lossless` | LIF current, lossless | LIF exp with exact spike-count prediction | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_ps_lossless.html |
| `iaf_tum_2000` | LIF current + integrated STP | LIF exp with integrated short-term plasticity | https://nest-simulator.readthedocs.io/en/stable/models/iaf_tum_2000.html |
| `ignore_and_fire` | Toy / debugging | Fixed-interval spiking irrespective of input | https://nest-simulator.readthedocs.io/en/stable/models/ignore_and_fire.html |
| `izhikevich` | Izhikevich | Izhikevich neuron model | https://nest-simulator.readthedocs.io/en/stable/models/izhikevich.html |
| `lin_rate` | Rate | Linear rate model | https://nest-simulator.readthedocs.io/en/stable/models/lin_rate.html |
| `mat2_psc_exp` | Multi-timescale adaptive threshold | Non-resetting LIF with adaptive threshold (exp PSC) | https://nest-simulator.readthedocs.io/en/stable/models/mat2_psc_exp.html |
| `mcculloch_pitts_neuron` | Binary deterministic | Heaviside activation | https://nest-simulator.readthedocs.io/en/stable/models/mcculloch_pitts_neuron.html |
| `parrot_neuron` | Repeater | Repeats incoming spikes verbatim | https://nest-simulator.readthedocs.io/en/stable/models/parrot_neuron.html |
| `parrot_neuron_ps` | Repeater, precise spike | Repeats incoming spikes with precise timing | https://nest-simulator.readthedocs.io/en/stable/models/parrot_neuron_ps.html |
| `pp_cond_exp_mc_urbanczik` | Point process, multi-compartment | Two-compartment point-process w/ conductance synapses | https://nest-simulator.readthedocs.io/en/stable/models/pp_cond_exp_mc_urbanczik.html |
| `pp_psc_delta` | Point process | Leaky integrate of delta PSCs, point process | https://nest-simulator.readthedocs.io/en/stable/models/pp_psc_delta.html |
| `rate_neuron_ipn` | Rate base | Rate with input noise (base class) | https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_ipn.html |
| `rate_neuron_opn` | Rate base | Rate with output noise (base class) | https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_opn.html |
| `rate_transformer_node` | Rate | Sums incoming rates + nonlinearity | https://nest-simulator.readthedocs.io/en/stable/models/rate_transformer_node.html |
| `siegert_neuron` | Rate / mean-field | Mean-field analysis of spiking networks | https://nest-simulator.readthedocs.io/en/stable/models/siegert_neuron.html |
| `sigmoid_rate` | Rate | Sigmoidal gain | https://nest-simulator.readthedocs.io/en/stable/models/sigmoid_rate.html |
| `sigmoid_rate_gg_1998` | Rate | Sigmoidal gain (Gerstner & Gerstner 1998 variant) | https://nest-simulator.readthedocs.io/en/stable/models/sigmoid_rate_gg_1998.html |
| `spike_train_injector` | Injector | Emits prescribed spike trains as a neuron | https://nest-simulator.readthedocs.io/en/stable/models/spike_train_injector.html |
| `tanh_rate` | Rate | tanh gain | https://nest-simulator.readthedocs.io/en/stable/models/tanh_rate.html |
| `threshold_lin_rate` | Rate | Threshold-linear gain | https://nest-simulator.readthedocs.io/en/stable/models/threshold_lin_rate.html |

**Total NEST neurons in this snapshot:** 74.

## 2. Synapses and plasticity

| Model | Class | Description | Upstream doc |
|---|---|---|---|
| `bernoulli_synapse` | Static, stochastic | Static synapse with stochastic transmission | https://nest-simulator.readthedocs.io/en/stable/models/bernoulli_synapse.html |
| `clopath_synapse` | Voltage-based STDP | Clopath voltage-based STDP | https://nest-simulator.readthedocs.io/en/stable/models/clopath_synapse.html |
| `cont_delay_synapse` | Static | Continuous delays | https://nest-simulator.readthedocs.io/en/stable/models/cont_delay_synapse.html |
| `diffusion_connection` | Rate | Instantaneous rate between `siegert_neuron`s | https://nest-simulator.readthedocs.io/en/stable/models/diffusion_connection.html |
| `eprop_learning_signal_connection` | e-prop | Feedback learning signal | https://nest-simulator.readthedocs.io/en/stable/models/eprop_learning_signal_connection.html |
| `eprop_learning_signal_connection_bsshslm_2020` | e-prop (Bellec) | Feedback learning signal (Bellec formulation) | https://nest-simulator.readthedocs.io/en/stable/models/eprop_learning_signal_connection_bsshslm_2020.html |
| `eprop_synapse` | e-prop plasticity | e-prop plasticity rule | https://nest-simulator.readthedocs.io/en/stable/models/eprop_synapse.html |
| `eprop_synapse_bsshslm_2020` | e-prop (Bellec) | e-prop plasticity rule (Bellec formulation) | https://nest-simulator.readthedocs.io/en/stable/models/eprop_synapse_bsshslm_2020.html |
| `gap_junction` | Gap junction | Bidirectional voltage coupling | https://nest-simulator.readthedocs.io/en/stable/models/gap_junction.html |
| `ht_synapse` | Hill-Tononi | Synapse with depression (Hill & Tononi 2005) | https://nest-simulator.readthedocs.io/en/stable/models/ht_synapse.html |
| `jonke_synapse` | STDP variant | STDP with additional additive factors | https://nest-simulator.readthedocs.io/en/stable/models/jonke_synapse.html |
| `quantal_stp_synapse` | STP | Probabilistic short-term plasticity | https://nest-simulator.readthedocs.io/en/stable/models/quantal_stp_synapse.html |
| `rate_connection_delayed` | Rate | Rate connection with delay | https://nest-simulator.readthedocs.io/en/stable/models/rate_connection_delayed.html |
| `rate_connection_instantaneous` | Rate | Instantaneous rate connection | https://nest-simulator.readthedocs.io/en/stable/models/rate_connection_instantaneous.html |
| `sic_connection` | Astrocyte | Astrocyte-neuron slow-inward current | https://nest-simulator.readthedocs.io/en/stable/models/sic_connection.html |
| `static_synapse` | Static | Default static connection | https://nest-simulator.readthedocs.io/en/stable/models/static_synapse.html |
| `static_synapse_hom_w` | Static | Static with homogeneous weight | https://nest-simulator.readthedocs.io/en/stable/models/static_synapse_hom_w.html |
| `stdp_dopamine_synapse` | Neuromodulated STDP | Dopamine-modulated STDP | https://nest-simulator.readthedocs.io/en/stable/models/stdp_dopamine_synapse.html |
| `stdp_facetshw_synapse_hom` | Hardware-style STDP | STDP using homogeneous parameters (FACETS HW) | https://nest-simulator.readthedocs.io/en/stable/models/stdp_facetshw_synapse_hom.html |
| `stdp_nn_pre_centered_synapse` | STDP NN | Presynaptic-centered nearest-neighbour pairing | https://nest-simulator.readthedocs.io/en/stable/models/stdp_nn_pre_centered_synapse.html |
| `stdp_nn_restr_synapse` | STDP NN | Restricted symmetric NN pairing | https://nest-simulator.readthedocs.io/en/stable/models/stdp_nn_restr_synapse.html |
| `stdp_nn_symm_synapse` | STDP NN | Symmetric NN pairing | https://nest-simulator.readthedocs.io/en/stable/models/stdp_nn_symm_synapse.html |
| `stdp_pl_synapse_hom` | STDP power-law | Power-law STDP, homogeneous params | https://nest-simulator.readthedocs.io/en/stable/models/stdp_pl_synapse_hom.html |
| `stdp_synapse` | STDP (canonical) | Canonical STDP synapse | https://nest-simulator.readthedocs.io/en/stable/models/stdp_synapse.html |
| `stdp_synapse_hom` | STDP, homogeneous | Canonical STDP, homogeneous params | https://nest-simulator.readthedocs.io/en/stable/models/stdp_synapse_hom.html |
| `stdp_triplet_synapse` | STDP triplet | Pfister-Gerstner triplet STDP | https://nest-simulator.readthedocs.io/en/stable/models/stdp_triplet_synapse.html |
| `tsodyks2_synapse` | STP | Tsodyks-Markram, v2 | https://nest-simulator.readthedocs.io/en/stable/models/tsodyks2_synapse.html |
| `tsodyks_synapse` | STP | Tsodyks-Markram, v1 | https://nest-simulator.readthedocs.io/en/stable/models/tsodyks_synapse.html |
| `tsodyks_synapse_hom` | STP, homogeneous | Tsodyks-Markram with homogeneous params | https://nest-simulator.readthedocs.io/en/stable/models/tsodyks_synapse_hom.html |
| `urbanczik_synapse` | Voltage-based, dendritic | Urbanczik-Senn plasticity | https://nest-simulator.readthedocs.io/en/stable/models/urbanczik_synapse.html |
| `vogels_sprekeler_synapse` | Inhibitory STDP | Symmetric STDP with constant depression | https://nest-simulator.readthedocs.io/en/stable/models/vogels_sprekeler_synapse.html |
| `weight_optimizer` | e-prop optimizer | Selection of weight optimizers (Adam, SGD…) | https://nest-simulator.readthedocs.io/en/stable/models/weight_optimizer.html |

**Total NEST synapses/plasticity in this snapshot:** 32.

## 3. Devices — stimulation / generators

| Model | Description | Upstream doc |
|---|---|---|
| `ac_generator` | Alternating-current input | https://nest-simulator.readthedocs.io/en/stable/models/ac_generator.html |
| `dc_generator` | Direct-current input | https://nest-simulator.readthedocs.io/en/stable/models/dc_generator.html |
| `gamma_sup_generator` | Superimposed gamma spike train | https://nest-simulator.readthedocs.io/en/stable/models/gamma_sup_generator.html |
| `inhomogeneous_poisson_generator` | Poisson with piecewise-constant rate | https://nest-simulator.readthedocs.io/en/stable/models/inhomogeneous_poisson_generator.html |
| `mip_generator` | Multiple-interaction-process spike train | https://nest-simulator.readthedocs.io/en/stable/models/mip_generator.html |
| `noise_generator` | Gaussian white-noise current | https://nest-simulator.readthedocs.io/en/stable/models/noise_generator.html |
| `poisson_generator` | Poisson spike train | https://nest-simulator.readthedocs.io/en/stable/models/poisson_generator.html |
| `poisson_generator_ps` | Poisson with precise spike timing + dead time | https://nest-simulator.readthedocs.io/en/stable/models/poisson_generator_ps.html |
| `ppd_sup_generator` | Superimposed Poisson with dead time | https://nest-simulator.readthedocs.io/en/stable/models/ppd_sup_generator.html |
| `pulsepacket_generator` | Sequence of Gaussian pulse packets | https://nest-simulator.readthedocs.io/en/stable/models/pulsepacket_generator.html |
| `sinusoidal_gamma_generator` | Sinusoidally-modulated gamma spike train | https://nest-simulator.readthedocs.io/en/stable/models/sinusoidal_gamma_generator.html |
| `sinusoidal_poisson_generator` | Sinusoidally-modulated Poisson spike train | https://nest-simulator.readthedocs.io/en/stable/models/sinusoidal_poisson_generator.html |
| `spike_generator` | Emits spikes from a prescribed array | https://nest-simulator.readthedocs.io/en/stable/models/spike_generator.html |
| `step_current_generator` | Piecewise-constant DC current | https://nest-simulator.readthedocs.io/en/stable/models/step_current_generator.html |
| `step_rate_generator` | Piecewise-constant input rate (for rate models) | https://nest-simulator.readthedocs.io/en/stable/models/step_rate_generator.html |

## 4. Devices — recorders

| Model | Description | Upstream doc |
|---|---|---|
| `multimeter` | Sample continuous quantities from neurons (configurable `record_from`) | https://nest-simulator.readthedocs.io/en/stable/models/multimeter.html |
| `spike_recorder` | Collect spikes from neurons | https://nest-simulator.readthedocs.io/en/stable/models/spike_recorder.html |
| `weight_recorder` | Record synaptic weights over time | https://nest-simulator.readthedocs.io/en/stable/models/weight_recorder.html |

## 5. Devices — detectors

| Model | Description | Upstream doc |
|---|---|---|
| `correlation_detector` | Cross-correlation between two spike sources | https://nest-simulator.readthedocs.io/en/stable/models/correlation_detector.html |
| `correlomatrix_detector` | Covariance matrix over multiple inputs | https://nest-simulator.readthedocs.io/en/stable/models/correlomatrix_detector.html |
| `correlospinmatrix_detector` | Covariance matrix from binary states | https://nest-simulator.readthedocs.io/en/stable/models/correlospinmatrix_detector.html |
| `spin_detector` | Detect binary states in neurons | https://nest-simulator.readthedocs.io/en/stable/models/spin_detector.html |

## 6. Devices — other

| Model | Description | Upstream doc |
|---|---|---|
| `spike_dilutor` | Repeat incoming spikes with a probability | https://nest-simulator.readthedocs.io/en/stable/models/spike_dilutor.html |
| `volume_transmitter` | Neuromodulatory broadcast for plastic synapses | https://nest-simulator.readthedocs.io/en/stable/models/volume_transmitter.html |

## 7. MUSIC proxies (intentionally unsupported)

Per spec §7, MUSIC real-time inter-simulator coupling is out of scope:

`music_cont_in_proxy`, `music_cont_out_proxy`, `music_event_in_proxy`,
`music_event_out_proxy`, `music_message_in_proxy`, `music_rate_in_proxy`,
`music_rate_out_proxy`.

Catalogued here only for completeness; not counted as gaps.

## 8. Connection rules — `nest.Connect(... conn_spec={'rule': '...'} )`

| Rule | Required params | Optional params | Description |
|---|---|---|---|
| `one_to_one` | – | `allow_autapses`, `allow_multapses` | i-th source ↔ i-th target |
| `all_to_all` | – | `allow_autapses`, `allow_multapses` | Every source connects to every target |
| `pairwise_bernoulli` | `p` | `allow_autapses`, `allow_multapses`, `mask`, `use_on_source` | Per-pair Bernoulli(p) inspection |
| `symmetric_pairwise_bernoulli` | `p` | `make_symmetric=True`, `allow_autapses=False` | Bidirectional Bernoulli |
| `pairwise_poisson` | `pairwise_avg_num_conns` | `allow_multapses` (must remain True) | Poisson-distributed counts per pair |
| `fixed_total_number` | `N` | `allow_autapses`, `allow_multapses` | Exactly N edges total, randomly distributed |
| `fixed_indegree` | `indegree` | `allow_autapses`, `allow_multapses`, `mask` | Each target has exactly N inbound edges |
| `fixed_outdegree` | `outdegree` | `allow_autapses`, `allow_multapses`, `mask` | Each source has exactly N outbound edges |
| `conngen` | – | (via CSA) | Connection Set Algebra interface |
| `third_factor_bernoulli_with_pool` | – | – | For `TripartiteConnect()` (neuron + astrocyte triads) |

## 9. PyNEST top-level API

### 9.1 Connection handling

| Function | Purpose |
|---|---|
| `Connect(pre, post, conn_spec, syn_spec)` | Create connections between NodeCollections per a rule + synapse spec |
| `TripartiteConnect(pre, post, third, conn_spec, syn_spec)` | Connect with a third-factor pool (e.g., astrocytes) |
| `Disconnect(pre, post, conn_spec, syn_spec)` / `Disconnect(SynapseCollection)` | Remove connections |
| `GetConnections(source, target, synapse_model, synapse_label)` | Return `SynapseCollection` matching the query |

### 9.2 Node handling

| Function | Purpose |
|---|---|
| `Create(model, n=1, params=None, positions=None)` | Instantiate `n` nodes of `model`, optionally spatially positioned |
| `GetNodes(properties, local_only)` | Query `NodeCollection` by property dict |
| `GetLocalNodeCollection(node_collection)` | Subset of NodeCollection on local MPI rank |
| `PrintNodes()` | Show node-ID ranges + model names |

### 9.3 Model management

| Function | Purpose |
|---|---|
| `CopyModel(existing, new, params=None)` | Duplicate a model under a new name with parameter overrides |
| `Models(mtype, sel=None)` | List built-in models (`mtype` ∈ {nodes, synapses}) |
| `GetDefaults(model, keys=None)` | Read default parameter dict for a model |
| `SetDefaults(model, params)` | Mutate default parameter dict for a model |
| `ConnectionRules()` | List supported connection-rule names |

### 9.4 Simulation control

| Function | Purpose |
|---|---|
| `Simulate(t)` | Prepare + Run(t) + Cleanup |
| `Prepare()` | Initialize kernel for simulation |
| `Run(t)` | Advance biological time by `t` ms |
| `Cleanup()` | Release post-Run resources |
| `ResetKernel()` | Reset all kernel state (destroys CopyModel models) |
| `RunManager()` | Context manager wrapping Prepare/Run/Cleanup |
| `EnableStructuralPlasticity()` / `DisableStructuralPlasticity()` | Toggle structural plasticity |

### 9.5 Kernel configuration

| Function | Purpose |
|---|---|
| `SetKernelStatus(params)` | Configure kernel (`resolution`, `total_num_virtual_procs`, `rng_seed`, …) |
| `GetKernelStatus(keys=None)` | Read kernel-status dict |
| `Install(module)` | Load dynamic NEST module |

### 9.6 Status / info

| Function | Purpose |
|---|---|
| `GetStatus(nodes_or_synapses, keys=None)` | Read parameter dicts |
| `SetStatus(nodes_or_synapses, params)` | Mutate parameter dicts |
| `get_verbosity()` / `set_verbosity(level)` | Message verbosity |
| `authors()` / `help()` / `helpdesk()` / `message()` / `sysinfo()` / `get_argv()` | Misc info |

### 9.7 Parallel computing

| Function | Purpose |
|---|---|
| `NumProcesses()` | Number of MPI processes |
| `Rank()` | This process's MPI rank |
| `GetLocalVPs()` | Virtual processes local to this rank |
| `SetAcceptableLatency(port, latency)` / `SetMaxBuffered(port, n)` | MUSIC port config |
| `SyncProcesses()` | MPI barrier |

### 9.8 Data types / classes

| Class | Purpose |
|---|---|
| `NodeCollection` | Container for node IDs (supports `+`, slicing, iteration, GetStatus/SetStatus) |
| `SynapseCollection` | Container for connection identifiers (GetStatus/SetStatus, Disconnect) |
| `Parameter` | Runtime-evaluated parameter expression (composable, sampled per-pair at connect time) |
| `Mask` | Spatial connectivity mask |
| `CreateParameter(type, params)` | Build a Parameter for a distribution |
| `CollocatedSynapses(...)` | Multiple synapses created on the same edge pair |
| `Compartments` / `Receptors` | Multi-compartment dendrite + receptor specifications |
| `SonataNetwork` | Build + simulate SONATA-format networks |

### 9.9 Random-number factories (for use in params + Parameters)

`nest.random.uniform`, `nest.random.uniform_int`, `nest.random.normal`,
`nest.random.lognormal`, `nest.random.exponential`.

### 9.10 Mathematical / logical Parameter operators

`nest.math.exp/sin/cos/min/max/redraw`, `nest.logic.conditional`.

## 10. Spatial / topology API (`nest.spatial`, `nest.spatial_distributions`)

### 10.1 Layer placement

| Entity | Purpose |
|---|---|
| `nest.spatial.grid(shape, extent, center, edge_wrap)` | Regular grid-positioned NodeCollection |
| `nest.spatial.free(pos, extent, edge_wrap, num_dimensions)` | Free-positioned NodeCollection |

### 10.2 Position expressions (usable when connecting)

| Entity | Purpose |
|---|---|
| `nest.spatial.pos.{x,y,z}` | Position of node being set |
| `nest.spatial.source_pos.{x,y,z}` | Source position during `Connect` |
| `nest.spatial.target_pos.{x,y,z}` | Target position during `Connect` |
| `nest.spatial.distance` | Shortest-distance scalar (with periodic BC) |
| `nest.spatial.distance.{x,y,z}` | Per-axis displacement |

### 10.3 Spatial-distribution factories (Parameter generators)

`nest.spatial_distributions.exponential`,
`nest.spatial_distributions.gaussian`,
`nest.spatial_distributions.gaussian2D`,
`nest.spatial_distributions.gabor`,
`nest.spatial_distributions.gamma`.

### 10.4 Masks

2D: `rectangular`, `circular`, `doughnut`, `elliptical`, `grid`.
3D: `box`, `spherical`, `ellipsoidal`.
Constructed via `nest.CreateMask(masktype, specs, anchor=None)`.

### 10.5 Spatial query / inspection

`GetPosition`, `GetSourcePositions`, `GetTargetPositions`,
`FindNearestElement`, `FindCenterElement`, `Displacement`, `Distance`,
`SelectNodesByMask`, `DumpLayerNodes`, `DumpLayerConnections`.

### 10.6 Visualization

`PlotLayer`, `PlotTargets`, `PlotSources`, `PlotProbabilityParameter`,
plus `nest.raster_plot.*` and `nest.voltage_trace.*` for spike/voltage
visualization.

## 11. Kernel attributes (read/write via `SetKernelStatus`/`GetKernelStatus` or `nest.<attr>`)

`kernel_status`, `resolution`, `biological_time`, `to_do`, `max_delay`,
`min_delay`, `ms_per_tic`, `tics_per_ms`, `tics_per_step`, `T_max`,
`T_min`, `rng_types`, `rng_type`, `rng_seed`, `total_num_virtual_procs`,
`local_num_threads`, `num_processes`, `off_grid_spiking`,
`adaptive_target_buffers`, `use_wfr`, `wfr_comm_interval`, `wfr_tol`,
`wfr_max_iterations`, `wfr_interpolation_order`, `network_size`,
`num_connections`, `recording_backends`, `stimulation_backends`.

## 12. SONATA-format network builder

`nest.SonataNetwork(config_file, sim_config=None)` — declarative
network construction from SONATA HDF5/CSV/JSON files.

---

## Quick-reference totals

| Category | Count |
|---|---|
| Neurons (excl. MUSIC proxies) | 74 |
| Synapses + plasticity | 32 |
| Stimulation generators | 15 |
| Recorders | 3 |
| Detectors | 4 |
| Other devices | 2 |
| MUSIC proxies (unsupported) | 7 |
| Connection rules | 10 (incl. `conngen`, `third_factor_bernoulli_with_pool`) |
| PyNEST top-level functions/classes | ≈ 70 |
| Spatial/topology entities | ≈ 25 |
