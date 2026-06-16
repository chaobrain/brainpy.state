# Synapses and plasticity — NEST parity gap

_Status as of 2026-06-16._

## 1. Scope

Static synapses, STDP family (9 variants), STP family (4 variants), Clopath
voltage-based STDP, Urbanczik dendritic, Jonke, Vogels-Sprekeler symmetric
inhibitory STDP, dopamine-modulated STDP + `volume_transmitter`, gap junctions,
`sic_connection` (astrocyte→neuron), `diffusion_connection` (Siegert mean-field),
`cont_delay_synapse` (continuous delays), `bernoulli_synapse` (stochastic
transmission), `ht_synapse` (Hill-Tononi depression), rate connections
(instantaneous and delayed).

Upstream reference: <https://nest-simulator.readthedocs.io/en/stable/models/index.html>
(synapse section of catalog snapshot: 32 entries).

Lead implementations actually read for this analysis:
- `brainpy_state/_nest/stdp_synapse.py` (177 lines; confirms NEST-canonical
  parameters `tau_plus`, `tau_minus`, `lambda_`, `alpha`, `mu_plus`, `mu_minus`,
  `Wmax`, `Kplus`; full STDP update equations match NEST Morrison et al. 2008).
- `brainpy_state/_nest/tsodyks2_synapse.py` (132 lines; confirms NEST STP
  parameters `U`, `tau_rec`, `tau_fac`; supports `tau_fac == 0` special case
  per NEST convention).
- Family extrapolation from these two leads + structural-signature checks
  across the remaining 26 ported synapse modules.

## 2. Parity summary

Most static and STDP synapses have NEST-comparison tests. The e-prop synapse
family (4 variants) plus `weight_optimizer` are **out of scope here** — they live
in the sibling `braintrace` package (see §4), not in `brainpy_state._nest`.

> **Update (cluster-09).** The STP family and `volume_transmitter` are no longer
> unvalidated: live-NEST parity now lives in the `_validation` harness, not the
> per-module law tests `<module>_rule_test.py` (which stay NEST-free rule tests).
> Closed:
> `tsodyks*`/`tsodyks2` via `_validation/stp_parity_test.py`, `quantal_stp_synapse`
> via `_validation/quantal_stp_parity_test.py`, `volume_transmitter` +
> `stdp_dopamine_synapse` via `_validation/volume_transmitter_parity_test.py` /
> `_validation/stdp_dopamine_synapse_parity_test.py` (cluster-07/08). The family-wide
> **`weight_recorder` send-event audit** (`_validation/weight_recorder_audit_test.py`)
> asserts event count + timing + value against NEST for all 13 plastic rules.

> **Update (cluster-21).** `urbanczik_synapse` — the last plastic synapse on the
> legacy base — is rebuilt as a frozen spec + pure `update(state, ctx) -> (state, w)`
> kernel on `VoltageCoupledPlasticProj` (primitive #2, the post-state reader). It is
> the primitive's third user (after `clopath_synapse` and `stdp_dopamine_synapse`)
> and the first to read a **named dendritic** post State: the rule declares
> `post_state_reads = ('delta_Pi',)`, so the substrate pulls the
> `pp_cond_exp_mc_urbanczik` dendritic prediction error δΠ per edge each step (no
> change to the somatic readers). Live-NEST parity in
> `_validation/urbanczik_synapse_parity_test.py` (dendritic `V_d` exact; recorded
> `V_W_star`/`delta_Pi` == closed-form on `V_d`; depression weight trajectory matches
> NEST at every `weight_recorder` send step); the `urbanczik_synapse_example` demo
> exercises the potentiating branch end-to-end (`examples-gap.md` §3.3).

> **Update (cluster-15b).** `gap_junction` is no longer a divergent unknown: it is
> validated against live NEST through the `Simulator` gap-coupling seam. The port does
> **not** use NEST's waveform relaxation — it reproduces NEST's `use_wfr=False` regime
> with an explicit **one-step-lagged difference current** deposited into the post's
> *current* channel: `I_gap,i[n] = Σ_j g_ij (V_j[n−1] − V_i[n−1])` = `(G − diag(D)) @ V[n−1]`,
> the negated graph Laplacian of the symmetric gap graph (`D = rowsum(G)`, `nS·mV = pA`).
> The off-diagonal `G @ V_pre` rides the SAME seam-(H) continuous-emission machinery as the
> rate/Wang seams (the post's `V` emission holder, `_emission_attr='V'`); the `−D·V_post`
> self term keeps the rest balance (`I_gap ≡ 0` when all `V` are equal). `REQUIRES_SYMMETRIC`
> is enforced at connect time (both directions materialized, hollow diagonal, scalar `g`,
> no delay, `comm='dense'`). **Design A resolved to option (a)** (full-lag difference deposit;
> no off-diagonal-seam + neuron-side self-leak split was needed). Parity: the 2-neuron
> micro-parity (`_validation/gap_junction_parity_test.py`) matches NEST to machine precision
> between spikes (median ~1e-3 mV, p95 ~0.1 mV), the only divergence an O(dt) AP-edge timing
> jitter (< 0.5 % of samples); the inhibitory-network Golomb coherence matches
> distributionally at async/sync gap weights (`_validation/gap_junction_inhibitory_network_parity_test.py`).
> The reference `gap_junction.py` WFR class (`begin_wfr_cycle`/`evaluate_gap_current`) stays
> in the tree but is **never invoked** on the simulation path (asserted by a mock guard).
> Per-edge / random / `comm='sparse'` gap weights remain out of scope. See `examples-gap.md` §3.6.

> **Update (cluster-15d) — last bucket-3 model on the substrate.** `sic_connection`
> (astrocyte→neuron slow-inward current) closes the last bucket-3 *model* cluster (only the
> Siegert `diffusion_connection` remains queued → 15c) and is now a wired,
> parity-validated tripartite loop on the `Simulator`. `astrocyte_lr_1994` emits
> its graded `SIC` through the seam-(H) continuous-emission path
> (`_emission_attr='SIC'`, `_emission_current=True`); the one-way `sic_connection`
> (sender/receiver-enforced `astrocyte_lr_1994 → aeif_cond_alpha_astro` via
> `check_connection`) deposits `weight·SIC` into the neuron's labelled `'I_SIC'`
> **current** channel through a new `as_current` `EventProjection` mode (a pA current
> entering `dV/dt`, not a delta/conductance — requires `comm='dense'`). The neuron→astro
> arm stays the ordinary delta path (`Δ_IP3·w` IP3 via `sum_delta_inputs`). The
> **bucket-3 de-queue**: the host-side `_sic_queue` event-emulator on the neuron and
> `sic_connection`'s host-queue coeff-array API (`prepare_secondary_event` /
> `to_aeif_sic_event` / `to_sic_event` / `coeffarray_to_step_events`) are **deleted**, so
> the whole bidirectional loop lowers under `Simulator.simulate`'s `for_loop`. Explicit
> one-step pipeline lag both directions; `sic_connection`'s NEST default delay `1.0 ms`
> maps to `delay_steps=10`. Live-NEST parity (`_validation/astrocyte_sic_test.py`) is
> near-exact: SIC-response IP3 `2.4e-5` / Ca `1.9e-4` / I_SIC `2.3e-4`; driven loop IP3
> `0` / Ca `1e-6` / I_SIC `6e-4` / V_pre `3.7e-4 mV`; astro-network seed-mean post rate
> `9.0=9.0` (SIC off) → `14.0=14.0` (SIC on, the SIC-raises-firing law, identical on both
> sims). `sic_connection_test.py` stays NEST-free rule tests.
>
> **Cluster 24 (`TripartiteConnect`).** `sic_connection` is now also consumed by
> `Simulator.tripartite_connect` as the **`third_out`** (astrocyte→neuron) arm: the
> orchestrator shares one realized primary sample across three arms and wires the
> astrocyte→post arm through this same merged static-STP + SIC deposit path — **no new
> deposit primitive**. Validated end-to-end against live NEST in the cluster-24 micro-
> parity GATE (`_validation/tripartite_connect_test.py`) and the three astrocyte demos
> (`astrocyte_small_network`, `astrocyte_brunel_{bernoulli,fixed_indegree}`).

| Bucket | Count | Notes |
|---|---:|---|
| implemented | 0 | No tolerance/duration documented in test headers |
| unvalidated | 0 | STP family + `volume_transmitter` now carry live-NEST `_validation` parity (cluster-01/07) — see §3 |
| partial | 0 known | (no missing parameters identified at family level — see §5) |
| nest_validated / divergent | 28 | Every in-scope §3 row (incl. the `volume_transmitter` support node) has a live-NEST `_validation/<module>_parity_test.py`; some are PRNG-distributional or trace-storage-divergent (see §5) |
| missing | 0 | — |
| out of scope → `braintrace` | 5 | `eprop_synapse`, `eprop_synapse_bsshslm_2020`, `eprop_learning_signal_connection`, `eprop_learning_signal_connection_bsshslm_2020`, `weight_optimizer` — e-prop family lives in the sibling `braintrace` package, not in `brainpy_state._nest` (see §4) |
| unsupported | 0 | — |
| **total NEST synapses/plasticity surveyed** | **32** | per snapshot §2 (in-scope synapses + the e-prop family routed to `braintrace`) |

## 3. Evidence-backed mapping table

| NEST model | Status | brainpy.state location | NEST upstream | Tests (import nest?) | Notes |
|---|---|---|---|---|---|
| `bernoulli_synapse` | divergent | `brainpy_state/_nest/bernoulli_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/bernoulli_synapse.html> | `bernoulli_synapse_rule_test.py` (N, rule) + `_validation/bernoulli_synapse_parity_test.py` (Y) | stochastic transmission; PRNG distributional only (bit-exact unsupported) |
| `clopath_synapse` | divergent | `brainpy_state/_nest/clopath_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/clopath_synapse.html> | `clopath_synapse_rule_test.py` (N, rule) + `_validation/clopath_synapse_parity_test.py` (Y) | voltage-based STDP; needs Clopath-capable postsynaptic neuron (`aeif_psc_delta_clopath`, `hh_psc_alpha_clopath`) |
| `cont_delay_synapse` | divergent | `brainpy_state/_nest/cont_delay_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/cont_delay_synapse.html> | `cont_delay_synapse_rule_test.py` (N, rule) + `_validation/cont_delay_synapse_parity_test.py` (Y) | continuous (non-grid) delays |
| `diffusion_connection` | divergent | `brainpy_state/_nest/diffusion_connection.py` | <https://nest-simulator.readthedocs.io/en/stable/models/diffusion_connection.html> | `diffusion_connection_test.py` (Y) + `_validation/siegert_diffusion_test.py` (Y) | Siegert-only rate connection |
| `gap_junction` | divergent | `brainpy_state/_nest/gap_junction.py` | <https://nest-simulator.readthedocs.io/en/stable/models/gap_junction.html> | `_validation/gap_junction_parity_test.py` (Y) + `_validation/gap_junction_inhibitory_network_parity_test.py` (Y) | gap-junction-capable neurons (`hh_psc_alpha_gap`, `hh_cond_beta_gap_traub`). **Validated (cluster-15b):** the `Simulator` realizes the gap as an explicit one-step-lagged difference current `I_gap = (G−diag(D))@V[n−1]` into the post current channel (NEST's `use_wfr=False` regime; **no** waveform relaxation, reference WFR class unused) on the seam-(H) V-emission path. 2-neuron micro-parity to machine precision between spikes (only O(dt) AP-edge jitter) + distributional network-coherence parity. `REQUIRES_SYMMETRIC` enforced; sparse / per-edge weights out of scope. See the cluster-15b Update above + `examples-gap.md` §3.6 |
| `ht_synapse` | divergent | `brainpy_state/_nest/ht_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/ht_synapse.html> | `ht_synapse_rule_test.py` (N, rule) + `_validation/ht_synapse_parity_test.py` (Y) | Hill-Tononi depression |
| `jonke_synapse` | divergent | `brainpy_state/_nest/jonke_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/jonke_synapse.html> | `jonke_synapse_rule_test.py` (N, rule) + `_validation/jonke_synapse_parity_test.py` (Y) | STDP with additive factors |
| `quantal_stp_synapse` | nest_validated | `brainpy_state/_nest/quantal_stp_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/quantal_stp_synapse.html> | `quantal_stp_synapse_rule_test.py` (N, rule) + `_validation/quantal_stp_parity_test.py` (Y) | probabilistic STP — PRNG distributional (6 seeds); validated cluster-01 |
| `rate_connection_delayed` | divergent | `brainpy_state/_nest/rate_connection_delayed.py` | <https://nest-simulator.readthedocs.io/en/stable/models/rate_connection_delayed.html> | `rate_connection_delayed_test.py` (Y) + `_validation/rate_delayed_connection_parity_test.py` (Y) | |
| `rate_connection_instantaneous` | divergent | `brainpy_state/_nest/rate_connection_instantaneous.py` | <https://nest-simulator.readthedocs.io/en/stable/models/rate_connection_instantaneous.html> | `rate_connection_instantaneous_test.py` (Y) + `_validation/rate_coupling_micro_parity_test.py` (Y) | requires waveform-relaxation (`use_wfr`) in NEST — brainpy.state semantics for instantaneous-rate loops needs verification |
| `sic_connection` | nest_validated | `brainpy_state/_nest/sic_connection.py` | <https://nest-simulator.readthedocs.io/en/stable/models/sic_connection.html> | `sic_connection_test.py` (N, rule) + `_validation/astrocyte_sic_test.py` (Y) | astrocyte slow-inward current; one-way `as_current` SIC loop wired & validated (cluster 15d, §2 update). Bucket-3 de-queue done |
| `static_synapse` | divergent | `brainpy_state/_nest/static_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/static_synapse.html> | `static_synapse_rule_test.py` (N, rule) + `_validation/static_synapse_parity_test.py` (Y) | default synapse |
| `static_synapse_hom_w` | divergent | `brainpy_state/_nest/static_synapse_hom_w.py` | <https://nest-simulator.readthedocs.io/en/stable/models/static_synapse_hom_w.html> | `static_synapse_hom_w_rule_test.py` (N, rule) + `_validation/static_synapse_parity_test.py` (Y) | homogeneous weight variant |
| `stdp_dopamine_synapse` | divergent | `brainpy_state/_nest/stdp_dopamine_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/stdp_dopamine_synapse.html> | `stdp_dopamine_synapse_rule_test.py` (N, rule) + `_validation/stdp_dopamine_synapse_parity_test.py` (Y) | needs `volume_transmitter`; eligibility-trace timing relative to dopamine signal is the high-risk surface |
| `stdp_facetshw_synapse_hom` | divergent | `brainpy_state/_nest/stdp_facetshw_synapse_hom.py` | <https://nest-simulator.readthedocs.io/en/stable/models/stdp_facetshw_synapse_hom.html> | `stdp_facetshw_synapse_hom_rule_test.py` (N, rule) + `_validation/stdp_facetshw_synapse_hom_parity_test.py` (Y) | hardware-style quantized weights |
| `stdp_nn_pre_centered_synapse` | divergent | `brainpy_state/_nest/stdp_nn_pre_centered_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/stdp_nn_pre_centered_synapse.html> | `stdp_nn_pre_centered_synapse_rule_test.py` (N, rule) + `_validation/stdp_nn_pre_centered_synapse_parity_test.py` (Y) | NN pairing — verify pairing convention exact |
| `stdp_nn_restr_synapse` | divergent | `brainpy_state/_nest/stdp_nn_restr_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/stdp_nn_restr_synapse.html> | `stdp_nn_restr_synapse_rule_test.py` (N, rule) + `_validation/stdp_nn_restr_synapse_parity_test.py` (Y) | restricted symmetric NN |
| `stdp_nn_symm_synapse` | divergent | `brainpy_state/_nest/stdp_nn_symm_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/stdp_nn_symm_synapse.html> | `stdp_nn_symm_synapse_rule_test.py` (N, rule) + `_validation/stdp_nn_symm_synapse_parity_test.py` (Y) | symmetric NN |
| `stdp_pl_synapse_hom` | divergent | `brainpy_state/_nest/stdp_pl_synapse_hom.py` | <https://nest-simulator.readthedocs.io/en/stable/models/stdp_pl_synapse_hom.html> | `stdp_pl_synapse_hom_rule_test.py` (N, rule) + `_validation/stdp_pl_synapse_hom_parity_test.py` (Y) | power-law STDP |
| `stdp_synapse` | divergent | `brainpy_state/_nest/stdp_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/stdp_synapse.html> | `stdp_synapse_rule_test.py` (N, rule) + `_validation/stdp_synapse_parity_test.py` (Y) | **divergent trace storage**: NEST stores `tau_minus` in postsynaptic `ArchivingNode`; repo stores the post `K-` trace inside the synapse (documented `stdp_synapse.py` §"NEST divergence — `tau_minus` location", ~lines 80-85). Pairing convention is canonical/all-to-all per Morrison 2008. See `docs/nest-guide/stdp-divergences.rst`. |
| `stdp_synapse_hom` | divergent | `brainpy_state/_nest/stdp_synapse_hom.py` | <https://nest-simulator.readthedocs.io/en/stable/models/stdp_synapse_hom.html> | `stdp_synapse_hom_rule_test.py` (N, rule) + `_validation/stdp_synapse_hom_parity_test.py` (Y) | homogeneous-param variant |
| `stdp_triplet_synapse` | divergent | `brainpy_state/_nest/stdp_triplet_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/stdp_triplet_synapse.html> | `stdp_triplet_synapse_rule_test.py` (N, rule) + `_validation/stdp_triplet_synapse_parity_test.py` (Y) | Pfister-Gerstner triplet rule |
| `tsodyks_synapse` | nest_validated | `brainpy_state/_nest/tsodyks_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/tsodyks_synapse.html> | `tsodyks_synapse_rule_test.py` (N, rule) + `_validation/stp_parity_test.py` (Y) | STP base; per-neuron `u,x,y` triple; validated cluster-01 |
| `tsodyks_synapse_hom` | nest_validated | `brainpy_state/_nest/tsodyks_synapse_hom.py` | <https://nest-simulator.readthedocs.io/en/stable/models/tsodyks_synapse_hom.html> | `tsodyks_synapse_hom_rule_test.py` (N, rule) + `_validation/stp_parity_test.py` (Y) | homogeneous; validated cluster-01 |
| `tsodyks2_synapse` | nest_validated | `brainpy_state/_nest/tsodyks2_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/tsodyks2_synapse.html> | `tsodyks2_synapse_rule_test.py` (N, rule) + `_validation/stp_parity_test.py` (Y) | v2 (multiplicative scaling, `tau_fac==0` special case); NEST defaults present; validated cluster-01 |
| `urbanczik_synapse` | divergent | `brainpy_state/_nest/urbanczik_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/urbanczik_synapse.html> | `urbanczik_synapse_test.py` (N, rule) + `_validation/urbanczik_synapse_parity_test.py` (Y) | dendritic prediction-error plasticity; **rebuilt** (cluster-21) as a frozen spec + pure `update` on `VoltageCoupledPlasticProj` reading the post-neuron `delta_Pi` per edge; live-NEST parity validated |
| `vogels_sprekeler_synapse` | divergent | `brainpy_state/_nest/vogels_sprekeler_synapse.py` | <https://nest-simulator.readthedocs.io/en/stable/models/vogels_sprekeler_synapse.html> | `vogels_sprekeler_synapse_rule_test.py` (N, rule) + `_validation/vogels_sprekeler_synapse_parity_test.py` (Y) | inhibitory STDP |
| `volume_transmitter` | nest_validated | `brainpy_state/_nest/volume_transmitter.py` | <https://nest-simulator.readthedocs.io/en/stable/models/volume_transmitter.html> | `volume_transmitter_rule_test.py` (N, rule) + `_validation/volume_transmitter_parity_test.py` (Y) | dopamine broadcast support node — required by `stdp_dopamine_synapse`; validated cluster-07 |
| `eprop_synapse` family + `weight_optimizer` | out of scope → `braintrace` | — (sibling `braintrace` package) | <https://nest-simulator.readthedocs.io/en/stable/models/weight_optimizer.html> | — | e-prop synapses + the Adam/SGD `weight_optimizer` selector are deliberately NOT ported into `brainpy_state._nest`; they live in the sibling `braintrace` package (see §4) |

## 4. Out-of-scope and incomplete functionality

**Out of scope here → sibling `braintrace` package (5):**

- `eprop_synapse`, `eprop_synapse_bsshslm_2020`,
  `eprop_learning_signal_connection`,
  `eprop_learning_signal_connection_bsshslm_2020` — the e-prop synapse family,
  together with the e-prop neurons. These are **not** a `brainpy_state._nest`
  gap: the whole e-prop family lives in the sibling **`braintrace`** package and
  is tracked there (the cluster-18 e-prop backlog was routed to `braintrace`, not
  this repo). Cross-reference `neurons-gap.md` for the matching neuron-side note.
- `weight_optimizer` — the Adam/SGD selector used by e-prop. Out of scope here for
  the same reason (e-prop → `braintrace`); in `brainpy.state`'s ecosystem the
  optax/brainstate optimizers cover the optimizer role.

**Not missing but not in repo as a separate concept:** NEST's `CollocatedSynapses`
(specifying multiple synapse types on the same pair of nodes simultaneously)
has no `brainpy.state` analog. Tracked in `network-api-gap.md`.

## 5. Semantic & numerical risks

- **Trace storage divergence (STDP family).** `stdp_synapse.py` documents this in
  its docstring section *"NEST divergence — `tau_minus` location"* (~lines 80-85):
  in NEST `tau_minus` is a parameter of the *postsynaptic neuron* (`ArchivingNode`),
  not the synapse; `brainpy.state` stores the post-side `K-` trace inside the synapse
  so JAX simulation doesn't require postsynaptic neurons to implement archiving APIs.
  This is a **load-bearing semantic difference**:
  - NEST's `tau_minus` lives on the post neuron, so setting it on the synapse
    has no effect; in `brainpy.state` it's a synapse parameter.
  - When porting NEST code, users may set `tau_minus` on the wrong object
    relative to NEST and get different behavior.
  - **Resolved (P0):** documented user-facing in `docs/nest-guide/stdp-divergences.rst`
    with the side-by-side of where `tau_minus` is set in NEST vs. `brainpy.state`.
- **Spike-pairing convention.** `stdp_synapse.py` applies canonical all-to-all
  pairing per the NEST default (the `K+ = pre_trace - pre_spike` /
  `K- = post_trace - post_spike` exclusion documented in its docstring). The
  `stdp_nn_*` variants implement three different NEST nearest-neighbor pairing
  schemes. Each pairing scheme is subtle; promotion to `implemented` requires a
  single-spike-pair regression test per variant against NEST.
- **Weight clipping.** NEST clamps in the update with `Wmax` (or `0` floor); the
  repo clamps `w` to `[0, Wmax]` inside each update (matching NEST
  `facilitate_`/`depress_`). The init-time sign check raises a Python `ValueError`
  ("Weight and Wmax must have same sign.", `stdp_synapse.py` ~line 149) — acceptable
  at init time but flag any in-update raises (spec §5).
- **STP family — `tsodyks*` + `quantal_stp_synapse` validated.** ✅ The five STP
  rules now carry live-NEST `_validation` parity (`_validation/stp_parity_test.py`
  for `tsodyks_synapse`/`_hom`/`tsodyks2_synapse`; `_validation/quantal_stp_parity_test.py`
  for `quantal_stp_synapse`). STP is widely used in cortical circuit models
  (Mongillo-Barak-Tsodyks working memory, etc.). Resolved cluster-01.
- **Volume transmitter latency.** `stdp_dopamine_synapse` reads from
  `volume_transmitter` with NEST's specific buffer-relay timing. Repo
  implementation needs to match NEST's deliver-on-VT-update semantics, not the
  simpler "read VT state at synapse-update time" pattern.
- **Weight recorder integration.** NEST's `weight_recorder` snapshots weights
  emitted by plastic synapses through a hook in the synapse's `send()` method.
  `brainpy.state`'s `weight_recorder.py` exists and is in `nest_validated`
  (see `devices-gap.md`); but verify the per-plasticity-rule "emit on update"
  hook is wired — particularly the `stdp_*` family.
- **Continuous delays.** `cont_delay_synapse` in NEST allows non-grid delays;
  repo's brainstate delay infrastructure rounds to dt grid by default. Verify
  the continuous-delay model in repo actually supports sub-dt delays.
- **Bernoulli synapse stochastic transmission.** Per-spike Bernoulli draw uses
  NEST's per-thread RNG; repo uses JAX PRNG keys. Distributional only.
- **Rate connections — waveform relaxation.** NEST's instantaneous rate
  connections require waveform-relaxation iteration controlled by
  `use_wfr`/`wfr_*` kernel attributes. The repo's equivalent loop semantics
  need verification; if absent, this is a P1 partial — but per the catalog
  snapshot the test imports nest, so the test should expose the divergence if
  it is structural.
- **Quantal STP.** Probabilistic vesicle release — distributional comparison
  only.
- **Multisynapse ports.** Multi-receptor models (`*_multisynapse`) appear as
  neurons in `neurons-gap.md`, but the *receptor weighting and port-ID
  semantics* are an NEST `Connect`-time convention (`receptor_type` in
  syn_spec). Coordinate with `network-api-gap.md` because Connect-side parity is
  needed for these neurons to be used as NEST does.
- **SIC loop — current deposit, not delta (RESOLVED cluster-15d).** The
  astrocyte slow-inward current is a **pA current entering `dV/dt`**, not a
  delta/conductance event. Routing it through the default delta channel would be
  wrong; `sic_connection` deposits via a new `as_current` `EventProjection` mode
  into the neuron's labelled `'I_SIC'` *current* channel. Two read-order rules
  guard correctness: the labelled `I_SIC` current read must precede the
  unlabelled `I_stim` read (whose `label=None` sums *all* current channels, else
  it would double-count SIC), and graded current requires `comm='dense'` (sparse
  binarises the presynaptic value). Both lags (neuron→astro spikes, astro→neuron
  SIC) carry the substrate's intrinsic one-step pipeline latency; `delay_steps=1`
  rides it with no `InputDelay`, larger values add `(delay_steps−1)` buffered
  steps (mirrors the deleted host queue's `base_offset`). NEST's default
  `delay=1.0 ms` ↔ `delay_steps=10`. Residual offsets are absorbed by
  `align_steps` in parity — exact-after-alignment (§2 cluster-15d).

## 6. Validation gaps

- The per-module law tests `<module>_rule_test.py` for `quantal_stp_synapse`,
  `tsodyks_synapse`, `tsodyks_synapse_hom`, `tsodyks2_synapse`, `volume_transmitter`
  have **no** `import nest` — but this no longer means "unvalidated": live-NEST
  parity for each moved to the `_validation` harness
  (`_validation/stp_parity_test.py`, `_validation/quantal_stp_parity_test.py`,
  `_validation/volume_transmitter_parity_test.py`). Resolved cluster-01/07.
- The `_validation/<module>_parity_test.py` tests that import `nest`
  (`@requires_nest`) still lack a uniformly documented tolerance + duration
  convention in their headers.
- ~~No NEST-comparison test exists for the **weight-recorder hook** wiring per
  plasticity rule.~~ **Resolved (cluster-09).** `_validation/weight_recorder_audit_test.py`
  reproduces NEST's `weight_recorder` send-event series through a thin send-view
  over the per-step weight trajectory (`brainpy_state._network.weight_recorder_events`
  / `send_steps_from_pre`) and asserts emitted event **count + timing + value**
  against NEST for all 13 plastic rules (the STDP family, `ht_synapse`,
  `clopath_synapse`, `stdp_dopamine_synapse`).
- ~~No NEST-comparison test stresses the **eligibility trace + volume-transmitter
  latency** of `stdp_dopamine_synapse`.~~ **Resolved (cluster-08/09):**
  `_validation/stdp_dopamine_synapse_parity_test.py` (eligibility window + sustained
  trajectory + clamp) and the dopamine row of the weight-recorder audit.

## 7. Prioritized roadmap

- **P0 — Document the STDP trace-storage divergence.** ✅ **Done.**
  `docs/nest-guide/stdp-divergences.rst` is the user-facing write-up: it shows the
  side-by-side of where `tau_minus` is set in NEST (post neuron / `ArchivingNode`)
  vs. `brainpy.state` (synapse parameter), with examples. The source-side note lives
  in the `stdp_synapse.py` docstring section *"NEST divergence — `tau_minus` location"*.

- **P0 — Validate the STP family.** ✅ **Done (cluster-01).**
  `_validation/stp_parity_test.py` (`tsodyks_synapse` / `_hom` / `tsodyks2_synapse`,
  EPSP-amplitude train vs NEST) and `_validation/quantal_stp_parity_test.py`
  (`quantal_stp_synapse`, distributional, 6 seeds); `volume_transmitter` via
  `_validation/volume_transmitter_parity_test.py`. Tolerances documented per test.

- **P0 — Validate `stdp_dopamine_synapse` + `volume_transmitter` against NEST.** ✅ **Done (cluster-08).**
  `_validation/stdp_dopamine_synapse_parity_test.py` reproduces the NEST weight
  trajectory (eligibility window + multi-second sustained LTP/LTD + clamp) within
  the documented one-step-`n`-lag band; layered on the `volume_transmitter` `n(t)`
  precondition.

- **P0 — Audit weight-recorder hookup per STDP variant.** ✅ **Done (cluster-09).**
  `_validation/weight_recorder_audit_test.py` asserts emitted event **count +
  timing + value** against NEST for all 13 plastic rules via the send-view seam
  (`brainpy_state._network.weight_recorder_events` / `send_steps_from_pre`). The
  seam masks the per-step weight trajectory to its send (pre-spike) steps — no
  imperative device, honouring the analog-recording State-tap posture.

- **P1 — Document spike-pairing-convention parity per `stdp_nn_*` variant.** [S]
  Acceptance: docstring of each NN variant cites the NEST source line / paper
  equation; a single-pair regression test asserts the convention.

- **P1 — Promote ported synapses from `divergent` to `implemented`.** [M]
  Apply the shared harness convention. Acceptance: 22 currently-`divergent`
  synapse tests grow tolerance + duration headers and pass.

- **P1 — Port `parrot_neuron`.** ✅ **Done.**
  `brainpy_state/_nest/parrot_neuron.py` exists with `parrot_neuron_test.py`
  verifying spike-time preservation; called out here because STDP and
  spike-injection examples rely on `parrot_neuron` as fan-out glue. Cross-cutting
  with `neurons-gap.md` P1.

- **Out of scope here — e-prop family → `braintrace`.**
  The e-prop synapse family + `weight_optimizer` are **not** a `brainpy_state._nest`
  roadmap item: they live in the sibling **`braintrace`** package and are tracked
  there (§4). Cross-listed with `neurons-gap.md`.

- **P2 — Validate rate connections under waveform-relaxation.** [M]
  Rationale: `rate_connection_instantaneous` semantics depend on NEST's
  waveform-relaxation. Acceptance: a 2-neuron Siegert demo reaches the same
  fixed point as NEST within tolerance.

- **P2 — `cont_delay_synapse` sub-dt delay verification.** [S]
  Acceptance: a single-spike delay test with `delay = 1.7 * dt` produces an
  EPSP at the same time NEST does (within ε = dt/100).

- **P2 — Audit STDP weight-clipping convention per variant.** [S]
  Acceptance: bounded-weight regression test (input drives weight against
  `Wmax`) shows the same clamp value as NEST.
