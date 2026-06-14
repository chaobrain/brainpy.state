# Neurons — NEST parity gap

## 1. Scope

Covers IAF (psc/cond/multisynapse/ps/lossless), AdEx (`aeif_*`), GIF, GLIF, HH,
MAT, Izhikevich, rate (`lin_rate`, `siegert_neuron`, `tanh_rate`, `sigmoid_rate*`,
`threshold_lin_rate`, `gauss_rate`), binary (`erfc_neuron`, `ginzburg_neuron`,
`mcculloch_pitts_neuron`), point-process (`pp_psc_delta`,
`pp_cond_exp_mc_urbanczik`), multi-compartment (`cm_default`,
`iaf_cond_alpha_mc`), astrocyte coupling (`astrocyte_lr_1994`,
`aeif_cond_alpha_astro`), spike-injection neurons (`spike_train_injector`,
`ignore_and_fire`), Brunel-Wang NMDA models (`iaf_bw_2001*`), and Hill-Tononi
(`ht_neuron`).

Upstream reference: <https://nest-simulator.readthedocs.io/en/stable/models/index.html>
(neuron section of catalog snapshot: 73 entries).

Lead implementations actually read for this analysis:
`brainpy_state/_nest/iaf_psc_alpha.py` (lines 1-274 confirm full NEST
parameter mapping table: `E_L=-70 mV`, `C_m=250 pF`, `tau_m=10 ms`,
`t_ref=2 ms`, `V_th=-55 mV`, `V_reset=-70 mV`, `tau_syn_ex=tau_syn_in=2 ms`,
`I_e=0 pA` — exactly NEST defaults). Family-wide pattern extrapolated from this
plus structural-signature checks (`grep` for `def __init__`, `init_state`,
`update`, integration helper imports) across the remaining 59 ported neuron
modules. Where extrapolation was used, it is flagged in §5.

## 2. Parity summary

Most ported neurons present NEST-compatible parameter surfaces but lack
numerical validation against upstream NEST traces. The AdEx and rate families
have NEST-comparison tests; the IAF, GIF, GLIF, HH, MAT, Izhikevich, binary,
point-process, multi-compartment, and Brunel-Wang families do not. Ten NEST
neurons are entirely missing from `_nest/`: the e-prop family (8 models) and
both `parrot_neuron` variants. MUSIC proxies are intentionally unsupported per
spec §7.

| Bucket | Count | Notes |
|---|---:|---|
| implemented | 0 | (no neuron has a passing NEST-trace comparison documented in CI) |
| unvalidated | ~40 | Present and parameter-compatible, no NEST-trace test |
| partial | 0 known | (none identified at family level — see §5 for per-model risks) |
| divergent | 23 | AdEx and rate families: parameter-compatible *and* nest-comparison test exists; classified divergent because surrogate-gradient (`spk_fun`) extensions are additive to NEST and intentional. Plus `pp_cond_exp_mc_urbanczik` (cluster-21): per-compartment live-NEST parity in the `_validation` harness |
| missing | 10 | e-prop (8) + `parrot_neuron`, `parrot_neuron_ps` |
| unsupported | 7 | MUSIC proxies (catalogued in `nest-catalog-snapshot.md` §7) |
| **total NEST neurons surveyed** | **73** | per snapshot §1 |

Reading note: "divergent" here means the model exists, matches NEST parameters,
*and* has a `nest`-importing comparison test in its `*_test.py`. The divergence
is the additive brainpy-extension (surrogate gradient, differentiability) that
NEST itself doesn't offer. Promotion to `implemented` requires documenting
trace-tolerance in the test and confirming pass.

## 3. Evidence-backed mapping table

| NEST model | Status | brainpy.state location | NEST upstream | Tests (import nest?) | Notes |
|---|---|---|---|---|---|
| `aeif_cond_alpha` | divergent | `brainpy_state/_nest/aeif_cond_alpha.py` | <https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_alpha.html> | `aeif_cond_alpha_test.py` (Y, line 448) | nest-comparison present — but tolerance + duration not documented in header |
| `aeif_cond_alpha_astro` | divergent | `brainpy_state/_nest/aeif_cond_alpha_astro.py` | <https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_alpha_astro.html> | `aeif_cond_alpha_astro_test.py` (Y) | astrocyte coupling exercised; `sic_connection` skip-if-not-available pattern in test |
| `aeif_cond_alpha_multisynapse` | divergent | `brainpy_state/_nest/aeif_cond_alpha_multisynapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_alpha_multisynapse.html> | `aeif_cond_alpha_multisynapse_test.py` (Y) | multi-receptor port semantics covered |
| `aeif_cond_beta_multisynapse` | divergent | `brainpy_state/_nest/aeif_cond_beta_multisynapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_beta_multisynapse.html> | `aeif_cond_beta_multisynapse_test.py` (Y) | |
| `aeif_cond_exp` | divergent | `brainpy_state/_nest/aeif_cond_exp.py` | <https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_exp.html> | `aeif_cond_exp_test.py` (Y) | |
| `aeif_psc_alpha` | divergent | `brainpy_state/_nest/aeif_psc_alpha.py` | <https://nest-simulator.readthedocs.io/en/stable/models/aeif_psc_alpha.html> | `aeif_psc_alpha_test.py` (Y) | |
| `aeif_psc_delta` | divergent | `brainpy_state/_nest/aeif_psc_delta.py` | <https://nest-simulator.readthedocs.io/en/stable/models/aeif_psc_delta.html> | `aeif_psc_delta_test.py` (Y) | |
| `aeif_psc_delta_clopath` | divergent | `brainpy_state/_nest/aeif_psc_delta_clopath.py` | <https://nest-simulator.readthedocs.io/en/stable/models/aeif_psc_delta_clopath.html> | `aeif_psc_delta_clopath_test.py` (Y) | Clopath voltage-trace plasticity exercised |
| `aeif_psc_exp` | divergent | `brainpy_state/_nest/aeif_psc_exp.py` | <https://nest-simulator.readthedocs.io/en/stable/models/aeif_psc_exp.html> | `aeif_psc_exp_test.py` (Y) | |
| `amat2_psc_exp` | unvalidated | `brainpy_state/_nest/amat2_psc_exp.py` | <https://nest-simulator.readthedocs.io/en/stable/models/amat2_psc_exp.html> | `amat2_psc_exp_test.py` (N) | self-consistency only |
| `cm_default` | divergent | `brainpy_state/_nest/cm_default.py` | <https://nest-simulator.readthedocs.io/en/stable/models/cm_default.html> | `cm_default_test.py` (Y) | multi-compartment; high-risk surface (per `nest-status/index.rst`) |
| `eprop_iaf` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf.html> | — | not ported |
| `eprop_iaf_adapt` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_adapt.html> | — | not ported |
| `eprop_iaf_adapt_bsshslm_2020` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_adapt_bsshslm_2020.html> | — | Bellec et al. 2020 variant; not ported |
| `eprop_iaf_bsshslm_2020` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_bsshslm_2020.html> | — | Bellec et al. 2020 variant; not ported |
| `eprop_iaf_psc_delta` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_psc_delta.html> | — | not ported |
| `eprop_iaf_psc_delta_adapt` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_psc_delta_adapt.html> | — | not ported |
| `eprop_readout` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/eprop_readout.html> | — | not ported |
| `eprop_readout_bsshslm_2020` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/eprop_readout_bsshslm_2020.html> | — | Bellec et al. 2020 variant; not ported |
| `erfc_neuron` | unvalidated | `brainpy_state/_nest/erfc_neuron.py` | <https://nest-simulator.readthedocs.io/en/stable/models/erfc_neuron.html> | `erfc_neuron_test.py` (N) | binary stochastic; PRNG parity caveat |
| `gauss_rate` | divergent | `brainpy_state/_nest/gauss_rate.py` | <https://nest-simulator.readthedocs.io/en/stable/models/gauss_rate.html> | `gauss_rate_test.py` (Y) | |
| `gif_cond_exp` | unvalidated | `brainpy_state/_nest/gif_cond_exp.py` | <https://nest-simulator.readthedocs.io/en/stable/models/gif_cond_exp.html> | `gif_cond_exp_test.py` (N) | category-A RKF45 integration; no NEST trace check |
| `gif_cond_exp_multisynapse` | unvalidated | `brainpy_state/_nest/gif_cond_exp_multisynapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/gif_cond_exp_multisynapse.html> | `gif_cond_exp_multisynapse_test.py` (N) | |
| `gif_pop_psc_exp` | unvalidated | `brainpy_state/_nest/gif_pop_psc_exp.py` | <https://nest-simulator.readthedocs.io/en/stable/models/gif_pop_psc_exp.html> | `gif_pop_psc_exp_test.py` (N) | population model — vectorised differently |
| `gif_psc_exp` | unvalidated | `brainpy_state/_nest/gif_psc_exp.py` | <https://nest-simulator.readthedocs.io/en/stable/models/gif_psc_exp.html> | `gif_psc_exp_test.py` (N) | |
| `gif_psc_exp_multisynapse` | unvalidated | `brainpy_state/_nest/gif_psc_exp_multisynapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/gif_psc_exp_multisynapse.html> | `gif_psc_exp_multisynapse_test.py` (N) | |
| `ginzburg_neuron` | unvalidated | `brainpy_state/_nest/ginzburg_neuron.py` | <https://nest-simulator.readthedocs.io/en/stable/models/ginzburg_neuron.html> | `ginzburg_neuron_test.py` (N) | binary stochastic |
| `glif_cond` | unvalidated | `brainpy_state/_nest/glif_cond.py` | <https://nest-simulator.readthedocs.io/en/stable/models/glif_cond.html> | `glif_cond_test.py` (N) | |
| `glif_psc` | unvalidated | `brainpy_state/_nest/glif_psc.py` | <https://nest-simulator.readthedocs.io/en/stable/models/glif_psc.html> | `glif_psc_test.py` (N) | |
| `glif_psc_double_alpha` | unvalidated | `brainpy_state/_nest/glif_psc_double_alpha.py` | <https://nest-simulator.readthedocs.io/en/stable/models/glif_psc_double_alpha.html> | `glif_psc_double_alpha_test.py` (N) | |
| `hh_cond_beta_gap_traub` | unvalidated | `brainpy_state/_nest/hh_cond_beta_gap_traub.py` | <https://nest-simulator.readthedocs.io/en/stable/models/hh_cond_beta_gap_traub.html> | `hh_cond_beta_gap_traub_test.py` (N) | gap-junction-capable HH — couples to `gap_junction` synapse |
| `hh_cond_exp_traub` | unvalidated | `brainpy_state/_nest/hh_cond_exp_traub.py` | <https://nest-simulator.readthedocs.io/en/stable/models/hh_cond_exp_traub.html> | `hh_cond_exp_traub_test.py` (N) | |
| `hh_psc_alpha` | unvalidated | `brainpy_state/_nest/hh_psc_alpha.py` | <https://nest-simulator.readthedocs.io/en/stable/models/hh_psc_alpha.html> | `hh_psc_alpha_test.py` (N) | |
| `hh_psc_alpha_clopath` | unvalidated | `brainpy_state/_nest/hh_psc_alpha_clopath.py` | <https://nest-simulator.readthedocs.io/en/stable/models/hh_psc_alpha_clopath.html> | `hh_psc_alpha_clopath_test.py` (N) | |
| `hh_psc_alpha_gap` | unvalidated | `brainpy_state/_nest/hh_psc_alpha_gap.py` | <https://nest-simulator.readthedocs.io/en/stable/models/hh_psc_alpha_gap.html> | `hh_psc_alpha_gap_test.py` (N) | |
| `ht_neuron` | unvalidated | `brainpy_state/_nest/ht_neuron.py` | <https://nest-simulator.readthedocs.io/en/stable/models/ht_neuron.html> | `ht_neuron_test.py` (N) | Hill-Tononi 2005; ~70 KB implementation suggests complex state |
| `iaf_bw_2001` | unvalidated | `brainpy_state/_nest/iaf_bw_2001.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_bw_2001.html> | `iaf_bw_2001_test.py` (N) | NMDA channels (simplified) |
| `iaf_bw_2001_exact` | unvalidated | `brainpy_state/_nest/iaf_bw_2001_exact.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_bw_2001_exact.html> | `iaf_bw_2001_exact_test.py` (N) | NMDA channels (exact) |
| `iaf_chs_2007` | unvalidated | `brainpy_state/_nest/iaf_chs_2007.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_chs_2007.html> | `iaf_chs_2007_test.py` (N) | spike-response form |
| `iaf_chxk_2008` | unvalidated | `brainpy_state/_nest/iaf_chxk_2008.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_chxk_2008.html> | `iaf_chxk_2008_test.py` (N) | precise-spike-time conductance LIF |
| `iaf_cond_alpha` | unvalidated | `brainpy_state/_nest/iaf_cond_alpha.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_alpha.html> | `iaf_cond_alpha_test.py` (N) | |
| `iaf_cond_alpha_mc` | unvalidated | `brainpy_state/_nest/iaf_cond_alpha_mc.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_alpha_mc.html> | `iaf_cond_alpha_mc_test.py` (N) | multi-compartment; flagged experimental in `nest-status/index.rst` |
| `iaf_cond_beta` | unvalidated | `brainpy_state/_nest/iaf_cond_beta.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_beta.html> | `iaf_cond_beta_test.py` (N) | |
| `iaf_cond_exp` | unvalidated | `brainpy_state/_nest/iaf_cond_exp.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_exp.html> | `iaf_cond_exp_test.py` (N) | |
| `iaf_cond_exp_sfa_rr` | unvalidated | `brainpy_state/_nest/iaf_cond_exp_sfa_rr.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_cond_exp_sfa_rr.html> | `iaf_cond_exp_sfa_rr_test.py` (N) | spike-frequency adaptation + relative refractory |
| `iaf_psc_alpha` | unvalidated | `brainpy_state/_nest/iaf_psc_alpha.py:36-274` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_alpha.html> | `iaf_psc_alpha_test.py` (N) | Exact NEST defaults present (E_L=-70mV, C_m=250pF, tau_m=10ms, t_ref=2ms, V_th=-55mV, V_reset=-70mV); analytical propagator. No NEST-trace check. |
| `iaf_psc_alpha_multisynapse` | unvalidated | `brainpy_state/_nest/iaf_psc_alpha_multisynapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_alpha_multisynapse.html> | `iaf_psc_alpha_multisynapse_test.py` (N) | 5KB test file — sparse coverage |
| `iaf_psc_alpha_ps` | unvalidated | `brainpy_state/_nest/iaf_psc_alpha_ps.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_alpha_ps.html> | `iaf_psc_alpha_ps_test.py` (N) | precise spike timing |
| `iaf_psc_delta` | unvalidated | `brainpy_state/_nest/iaf_psc_delta.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_delta.html> | `iaf_psc_delta_test.py` (N) | |
| `iaf_psc_delta_ps` | unvalidated | `brainpy_state/_nest/iaf_psc_delta_ps.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_delta_ps.html> | `iaf_psc_delta_ps_test.py` (N) | |
| `iaf_psc_exp` | unvalidated | `brainpy_state/_nest/iaf_psc_exp.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp.html> | `iaf_psc_exp_test.py` (N) | |
| `iaf_psc_exp_htum` | unvalidated | `brainpy_state/_nest/iaf_psc_exp_htum.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_htum.html> | `iaf_psc_exp_htum_test.py` (N) | separate relative + absolute refractory |
| `iaf_psc_exp_multisynapse` | unvalidated | `brainpy_state/_nest/iaf_psc_exp_multisynapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_multisynapse.html> | `iaf_psc_exp_multisynapse_test.py` (N) | |
| `iaf_psc_exp_ps` | unvalidated | `brainpy_state/_nest/iaf_psc_exp_ps.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_ps.html> | `iaf_psc_exp_ps_test.py` (N) | |
| `iaf_psc_exp_ps_lossless` | divergent | `brainpy_state/_nest/iaf_psc_exp_ps_lossless.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_ps_lossless.html> | `iaf_psc_exp_ps_lossless_test.py` (N) | NEST's lossless predicate (Krishnan et al.) — verify implementation matches |
| `iaf_tum_2000` | unvalidated | `brainpy_state/_nest/iaf_tum_2000.py` | <https://nest-simulator.readthedocs.io/en/stable/models/iaf_tum_2000.html> | `iaf_tum_2000_test.py` (N) | LIF + integrated STP — coupling semantics to validate |
| `ignore_and_fire` | divergent | `brainpy_state/_nest/ignore_and_fire.py` | <https://nest-simulator.readthedocs.io/en/stable/models/ignore_and_fire.html> | `ignore_and_fire_test.py` (Y) | toy model |
| `izhikevich` | unvalidated | `brainpy_state/_nest/izhikevich.py` | <https://nest-simulator.readthedocs.io/en/stable/models/izhikevich.html> | `izhikevich_test.py` (N) | 41KB test file but no nest-comparison |
| `lin_rate` | divergent | `brainpy_state/_nest/lin_rate.py` | <https://nest-simulator.readthedocs.io/en/stable/models/lin_rate.html> | `lin_rate_test.py` (Y) | |
| `mat2_psc_exp` | unvalidated | `brainpy_state/_nest/mat2_psc_exp.py` | <https://nest-simulator.readthedocs.io/en/stable/models/mat2_psc_exp.html> | `mat2_psc_exp_test.py` (N) | multi-timescale adaptive threshold |
| `mcculloch_pitts_neuron` | unvalidated | `brainpy_state/_nest/mcculloch_pitts_neuron.py` | <https://nest-simulator.readthedocs.io/en/stable/models/mcculloch_pitts_neuron.html> | `mcculloch_pitts_neuron_test.py` (N) | binary deterministic |
| `parrot_neuron` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/parrot_neuron.html> | — | spike-repeater; used pervasively in NEST examples as fan-out fan-in glue |
| `parrot_neuron_ps` | missing | — | <https://nest-simulator.readthedocs.io/en/stable/models/parrot_neuron_ps.html> | — | precise-spike variant |
| `pp_cond_exp_mc_urbanczik` | divergent | `brainpy_state/_nest/pp_cond_exp_mc_urbanczik.py` | <https://nest-simulator.readthedocs.io/en/stable/models/pp_cond_exp_mc_urbanczik.html> | `pp_cond_exp_mc_urbanczik_test.py` (N) + `_validation/urbanczik_synapse_parity_test.py` (Y) | multi-compartment + point-process; per-compartment parity vs live NEST (cluster-21): dendritic `V_d` exact, somatic `V_s`, and `V_W_star`/`delta_Pi` == closed-form on `V_d` |
| `pp_psc_delta` | unvalidated | `brainpy_state/_nest/pp_psc_delta.py` | <https://nest-simulator.readthedocs.io/en/stable/models/pp_psc_delta.html> | `pp_psc_delta_test.py` (N) | point process |
| `rate_neuron_ipn` | divergent | `brainpy_state/_nest/rate_neuron_ipn.py` | <https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_ipn.html> | `rate_neuron_ipn_test.py` (Y) | |
| `rate_neuron_opn` | divergent | `brainpy_state/_nest/rate_neuron_opn.py` | <https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_opn.html> | `rate_neuron_opn_test.py` (Y) | |
| `rate_transformer_node` | divergent | `brainpy_state/_nest/rate_transformer_node.py` | <https://nest-simulator.readthedocs.io/en/stable/models/rate_transformer_node.html> | `rate_transformer_node_test.py` (Y) | |
| `siegert_neuron` | divergent | `brainpy_state/_nest/siegert_neuron.py` | <https://nest-simulator.readthedocs.io/en/stable/models/siegert_neuron.html> | `siegert_neuron_test.py` (Y) | mean-field |
| `sigmoid_rate` | divergent | `brainpy_state/_nest/sigmoid_rate.py` | <https://nest-simulator.readthedocs.io/en/stable/models/sigmoid_rate.html> | `sigmoid_rate_test.py` (Y) | |
| `sigmoid_rate_gg_1998` | divergent | `brainpy_state/_nest/sigmoid_rate_gg_1998.py` | <https://nest-simulator.readthedocs.io/en/stable/models/sigmoid_rate_gg_1998.html> | `sigmoid_rate_gg_1998_test.py` (Y) | |
| `spike_train_injector` | divergent | `brainpy_state/_nest/spike_train_injector.py` | <https://nest-simulator.readthedocs.io/en/stable/models/spike_train_injector.html> | `spike_train_injector_test.py` (Y) | injector neuron — see also `devices-gap.md` |
| `tanh_rate` | divergent | `brainpy_state/_nest/tanh_rate.py` | <https://nest-simulator.readthedocs.io/en/stable/models/tanh_rate.html> | `tanh_rate_test.py` (Y) | |
| `threshold_lin_rate` | divergent | `brainpy_state/_nest/threshold_lin_rate.py` | <https://nest-simulator.readthedocs.io/en/stable/models/threshold_lin_rate.html> | `threshold_lin_rate_test.py` (Y) | |

Astrocyte / coupling models (overlap with neurons category — `astrocyte_lr_1994`
is documented under neurons in upstream):

| `astrocyte_lr_1994` | divergent | `brainpy_state/_nest/astrocyte_lr_1994.py` | <https://nest-simulator.readthedocs.io/en/stable/models/astrocyte_lr_1994.html> | `astrocyte_lr_1994_test.py` (Y) | astrocyte model used with `aeif_cond_alpha_astro` |

## 4. Missing or incomplete functionality

**Entirely missing (10):**

- `eprop_iaf`, `eprop_iaf_adapt`, `eprop_iaf_adapt_bsshslm_2020`,
  `eprop_iaf_bsshslm_2020`, `eprop_iaf_psc_delta`, `eprop_iaf_psc_delta_adapt`,
  `eprop_readout`, `eprop_readout_bsshslm_2020` — e-prop family (Bellec et al.
  2020 and follow-ups). Not ported. Because `brainpy.state` already provides
  surrogate-gradient training natively via brainstate/braintools, the cleanest
  port may be to *interoperate* with the existing surrogate-gradient stack
  rather than copy NEST's e-prop ODEs. Coordination with `synapses-plasticity-
  gap.md` needed because the e-prop family ships its own `eprop_synapse` and
  `eprop_learning_signal_connection`.
- `parrot_neuron`, `parrot_neuron_ps` — small but pervasive in NEST examples
  (Brunel, microcircuit, sinusoidal Poisson demos). Used as fan-out / fan-in
  glue. Easy to port (under one screen of code in NEST itself).

**Variants potentially missing (extrapolation flagged):** None confirmed within
families that are at least partially ported. The IAF, AdEx, GIF, GLIF, HH, MAT,
rate, and binary families look complete at the model-name level.

## 5. Semantic & numerical risks

This section captures risks identified or extrapolated; per the methodology
many are *not* per-file verified — flagged where so.

- **Defaults drift — IAF psc family.** `iaf_psc_alpha.py:209-243` shows
  NEST-identical defaults (E_L=-70 mV, C_m=250 pF, tau_m=10 ms, t_ref=2 ms,
  V_th=-55 mV, V_reset=-70 mV, tau_syn=2 ms, I_e=0 pA). Family extrapolation:
  other `iaf_psc_*` files were spot-checked structurally (similar parameter
  table convention); per-default audit pending in roadmap.
- **Brainpy-extension parameters added to every NEST neuron.** Every neuron in
  the repo carries `V_initializer`, `spk_fun`, `spk_reset`, `ref_var` in
  addition to the NEST parameters (`iaf_psc_alpha.py:178-189`). These are
  additive — they do not change NEST behavior when left at defaults. Document
  this convention in `docs/api/nest-neurons.rst`. Classified `divergent` where
  these extensions are exposed; not a numerical risk if defaults are unchanged.
- **Integration method — IAF psc family uses analytical propagators
  (Category B).** Per spec §1: `iaf_psc_alpha.py` documents NEST-style exact
  linear propagators (lines 74-97). This is the right approach for NEST parity;
  numerical comparison with NEST should be near-exact at fixed dt.
- **Integration method — AdEx / GIF / GLIF / IAF cond (Category A).** Spec §1
  describes vectorised `AdaptiveRungeKuttaStep` (RKF45). NEST itself uses GSL
  embedded Runge-Kutta-Fehlberg for these models. *Tolerances may differ.*
  Where NEST defaults `rtol=1e-3, atol=1e-3` (model-dependent), the brainpy.state
  RKF45 tolerance must match for trace comparison to converge — verify before
  promoting any A-category model to `implemented`.
- **Integration method — Hodgkin-Huxley (Category C).** Spec §1 swaps
  scipy `solve_ivp` for `AdaptiveRungeKuttaStep`. Conformance test still needed.
- **Refractory rounding.** NEST rounds `t_ref` to a multiple of `dt`. The
  `iaf_psc_alpha.py` docstring describes a step-counter approach (`refractory`
  state — `iaf_psc_alpha.py:188`). Whether the round-toward-zero / round-up
  convention matches NEST is unverified.
- **Spike threshold timing.** NEST checks `V_m >= V_th` *after* the propagator
  step. Repo implementation should match — extrapolation, not verified per
  model. Flag P1 audit.
- **Multi-compartment models.** `cm_default` and `iaf_cond_alpha_mc` are
  flagged `experimental` in `docs/nest-status/index.rst:89-91`. Compartment
  topology, receptor wiring, and recording semantics are particularly likely to
  diverge from NEST. `cm_default_test.py` does import `nest` — promote to
  `divergent` only after documenting the compartment-tree comparison protocol.
- **`iaf_psc_exp_ps_lossless`.** NEST implements Krishnan et al. (2018)'s
  lossless spike-time predicate. The repo file is `43.8 KB` — comparable size to
  NEST's reference, but predicate-equivalence is non-trivial to verify. Flag P1.
- **`iaf_tum_2000`.** Integrates STP into a LIF neuron; coupling to the
  synapse-side `tsodyks*` family must be consistent — coordinate with
  `synapses-plasticity-gap.md`.
- **Binary stochastic models (`erfc_neuron`, `ginzburg_neuron`,
  `mcculloch_pitts_neuron`).** PRNG divergence (spec §7: bitwise unsupported).
  Validate distributional equivalence (mean firing rate, autocorrelation) over a
  long window with multiple seeds.
- **`ht_neuron` (Hill-Tononi 2005).** 70.6 KB implementation; complex state
  with multiple ion channels and intrinsic plasticity. High-risk for numerical
  drift. No `import nest` reference test.
- **`iaf_bw_2001` / `iaf_bw_2001_exact`.** Brunel-Wang NMDA dynamics — known to
  be sensitive to integration step. `iaf_bw_2001_exact` claims "exact NMDA";
  protocol for comparison needs to specify NMDA saturation regime explicitly.
- **`gif_pop_psc_exp`.** Population-level model; vectorisation strategy in the
  repo may differ from NEST's per-population accumulation. Validation requires
  agreement on population summary statistics rather than per-neuron traces.

## 6. Validation gaps

- **48 of the 117 ported modules in `_nest/` have no `import nest` in their
  `*_test.py`** (`/tmp/unvalidated.txt`-style listing in this analysis). The
  neuron subset of that 48 lacks NEST-trace comparison entirely.
- **22 neurons do have `import nest` in their test** (AdEx family, rate family,
  astrocyte, `cm_default`, `ignore_and_fire`, `spike_train_injector`), but
  *no documented per-family tolerance + duration convention*. Even where
  comparison code exists, the test header should state "compared quantity =
  V_m, duration = T ms, dt = X ms, atol = Y, rtol = Z, max observed diff = …".
  Currently this is implicit — `aeif_cond_alpha_test.py:448` imports nest but
  the test header does not document tolerance.
- **No shared validation harness.** Each `*_test.py` re-implements its own NEST
  comparison glue. A shared helper would standardise: ResetKernel, parameter
  marshalling, multimeter setup, trace alignment, tolerance assertions.
  Cross-link: `numerical-validation-gap.md` Task 8.

## 7. Prioritized roadmap

- **P0 — Build a shared NEST-trace comparison harness.** [M]
  Rationale: the same comparison glue is repeated across ~22 test files, and
  ~48 ported modules lack any comparison at all. A reusable
  `brainpy_state/_nest/_validation/nest_compare.py` with documented tolerance
  conventions unblocks promoting families to `implemented`. Acceptance: at
  least 5 neuron tests use it; tolerance, duration, dt, and PRNG-seeding
  protocol are documented in the harness module's docstring; the harness skips
  by default under `pytest -m "not requires_nest"`.

- **P0 — Promote IAF psc family to `implemented` with the new harness.** [L]
  Rationale: IAF psc is the most-used base family in NEST examples and is
  currently `unvalidated` across all variants. Acceptance: `iaf_psc_alpha`,
  `iaf_psc_exp`, `iaf_psc_delta`, `iaf_psc_alpha_multisynapse`,
  `iaf_psc_exp_multisynapse`, `iaf_psc_alpha_ps`, `iaf_psc_exp_ps` all run V_m
  traces matching NEST within harness tolerance over a 1 s window with at least
  3 different parameter sets each. CI passes on a NEST-installed runner.

- **P0 — Promote IAF cond family to `implemented`.** [L]
  Acceptance: `iaf_cond_alpha`, `iaf_cond_beta`, `iaf_cond_exp`,
  `iaf_cond_exp_sfa_rr`, `iaf_chxk_2008` all run V_m + conductance traces
  matching NEST within harness tolerance over a 1 s window with at least 3
  parameter sets each. `iaf_cond_alpha_mc` separately documents compartment-
  tree topology equivalence (coordinate with `nest-status/index.rst:89`
  self-disclosure on multi-compartment experimental status).

- **P1 — Port `parrot_neuron` and `parrot_neuron_ps`.** [S]
  Rationale: needed for direct ports of NEST examples (Brunel uses
  `parrot_neuron` as a relay). Acceptance: both models present in
  `brainpy_state/_nest/`, with NEST-comparison tests verifying spike-time
  preservation (precise variant: sub-dt accuracy).

- **P1 — Promote AdEx family from `divergent` to `implemented`.** [M]
  Tests already import `nest`; missing is the per-test documented tolerance
  and an audit that AdEx parameters / state vars match NEST exactly.
  Acceptance: each `aeif_*` test header documents tolerance + duration + dt.

- **P1 — GIF / GLIF / HH / MAT / Izhikevich validation pass.** [L]
  Rationale: these are still in `unvalidated`. Acceptance: at least one
  representative variant per family promoted to `implemented` using the
  shared harness.

- **P1 — Define and document the brainpy-extension parameter convention.** [S]
  Rationale: `V_initializer`, `spk_fun`, `spk_reset`, `ref_var` appear on every
  NEST neuron. Document the convention once in `docs/api/nest-neurons.rst` (or
  a new `docs/nest-guide/`) so users porting from NEST know which parameters
  to ignore. Acceptance: convention page exists and is linked from every NEST
  neuron's docstring.

- **P2 — Port the e-prop family.** [XL]
  Rationale: brainpy.state already supports surrogate-gradient training; e-prop
  is an alternative formulation. Strategic decision: port verbatim from NEST,
  or wire e-prop semantics through the existing surrogate-gradient stack?
  Either path is multi-week. Acceptance: at minimum `eprop_iaf` + `eprop_readout`
  + `eprop_synapse` work end-to-end on a small classification task with a
  documented learning curve comparable to NEST's e-prop example.

- **P2 — Audit precise-spike-time variants (`*_ps`, `iaf_psc_exp_ps_lossless`).** [M]
  Rationale: NEST's regula-falsi root-finding and Krishnan's lossless predicate
  are non-trivial. Acceptance: sub-dt spike-time agreement with NEST within
  ε = `dt/10` over a 1 s window for the matched-input case.

- **P2 — Audit binary stochastic neurons (PRNG distributional check).** [S]
  Rationale: `erfc_neuron`, `ginzburg_neuron`, `mcculloch_pitts_neuron` —
  bit-exact RNG comparison is impossible (spec §7). Acceptance: mean firing
  rate within 2% of NEST over a 10 s window for 3 input regimes.

- **P2 — Audit `ht_neuron` and `iaf_bw_2001*`.** [M]
  Rationale: complex state, high drift risk. Acceptance: V_m + key
  channel-state traces within harness tolerance.

- **P2 — Audit `gif_pop_psc_exp` population semantics.** [M]
  Acceptance: population firing-rate statistics within tolerance.
