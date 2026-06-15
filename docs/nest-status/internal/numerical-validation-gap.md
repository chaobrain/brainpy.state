# Numerical validation — NEST parity gap (cross-cutting)

## 1. Scope

Cross-cutting inventory of NEST-comparison test coverage across all ported
modules in `brainpy_state/_nest/`. Per-family validation-coverage rollup,
identification of families that need a NEST-comparison harness before
promotion from Experimental → Beta, and a shared-infrastructure proposal.

Upstream reference: not applicable — this is a repo-internal classification.

Evidence basis:
- `grep -l "import nest" brainpy_state/_nest/*_test.py | wc -l` → **69**
  (run 2026-05-11).
- Total ported module files: **117** in `brainpy_state/_nest/__init__.py`
  exports (one of which is `_base`, not a model — so 116 user-visible models).
- Tests grep `import nest` inside individual test methods (lazy import,
  skipped via `try/except ImportError`), not at module top. So the count is
  authoritative only if checked with the full file scan, which this analysis
  does.
- 69 modules have at least one `import nest` test method.
- 48 modules (47 models + `_base`) have no `import nest` reference test.

## 2. Parity summary

NEST-comparison test coverage is **bimodal** by family. AdEx, rate models,
all devices (generators / recorders / detectors), and most synapses+plasticity
rules have NEST-comparison tests. The entire spiking-neuron set apart from
AdEx (IAF psc/cond/specialized, GIF, GLIF, HH, MAT, Izhikevich, binary,
point-process) and the entire STP family have **zero** NEST-comparison
coverage.

Even where tests exist, **no test header documents tolerance, duration, dt,
or PRNG-seeding protocol**. Promotion to `implemented` (per the taxonomy in
the spec §3) requires that documentation step to land per-family.

| Bucket | Count | % | Notes |
|---|---:|---:|---|
| modules with `import nest` test | 69 | 59 % | High-level coverage; tolerance unspecified |
| modules without `import nest` test | 48 | 41 % | Includes `_base` (1 module, not a model) — effective gap is 47 |
| modules with documented tolerance + duration | 0 | 0 % | No `*_test.py` header documents these |
| **total ported modules** | **117** | | |

## 3. Evidence-backed mapping table — per-family rollup

| Family | n_models | n_with_nest_test | coverage % | Lead test sample | Compared quantity (heuristic) | Notes |
|---|---:|---:|---:|---|---|---|
| AdEx (`aeif_*`) | 9 | 9 | 100 % | `aeif_cond_alpha_test.py:448-491` | V_m trace via multimeter | full family covered |
| IAF psc (`iaf_psc_*` incl. multisynapse + ps + lossless) | 10 | 0 | **0 %** | — | n/a | **largest gap** — base family of NEST |
| IAF cond (`iaf_cond_*` incl. mc + beta + sfa_rr) | 5 | 0 | **0 %** | — | n/a | **gap** |
| IAF specialized (`iaf_bw_2001*`, `iaf_chs_2007`, `iaf_chxk_2008`, `iaf_tum_2000`) | 5 | 0 | **0 %** | — | n/a | **gap** |
| MAT / AMAT (`mat2_psc_exp`, `amat2_psc_exp`) | 2 | 0 | **0 %** | — | n/a | **gap** |
| GIF (`gif_*`) | 5 | 0 | **0 %** | — | n/a | **gap** |
| GIF population (`gif_pop_psc_exp`) | (1 of 5 above) | 0 | **0 %** | — | n/a | population statistics needed |
| GLIF (`glif_*`) | 3 | 0 | **0 %** | — | n/a | **gap** |
| HH (`hh_*`, `ht_neuron`) | 6 | 0 | **0 %** | — | n/a | **gap** — includes Hill-Tononi |
| Izhikevich (`izhikevich`) | 1 | 0 | **0 %** | — | n/a | **gap** |
| Binary (`erfc_neuron`, `ginzburg_neuron`, `mcculloch_pitts_neuron`) | 3 | 0 | **0 %** | — | n/a | PRNG distributional |
| Point process (`pp_psc_delta`, `pp_cond_exp_mc_urbanczik`) | 2 | 1 | 50 % | `_validation/urbanczik_synapse_parity_test.py` (Y) | dendritic/somatic V + `V_W_star`/`delta_Pi` | `pp_cond_exp_mc_urbanczik` validated (cluster-21); `pp_psc_delta` unvalidated |
| Multi-compartment (`cm_default`, `iaf_cond_alpha_mc`) | 2 | 1 | 50 % | `cm_default_test.py` (Y) | V_m + compartment traces | `iaf_cond_alpha_mc` unvalidated |
| Astrocyte (`astrocyte_lr_1994`) | 1 | 1 | 100 % | `astrocyte_lr_1994_test.py` (Y) | astrocyte state | covered |
| Other neurons (`ignore_and_fire`, `spike_train_injector`) | 2 | 2 | 100 % | `ignore_and_fire_test.py` (Y), `spike_train_injector_test.py` (Y) | spike times | covered |
| Rate models (`lin_rate`, `tanh_rate`, `sigmoid_rate*`, `threshold_lin_rate`, `gauss_rate`, `siegert_neuron`, `rate_neuron_ipn/opn`, `rate_transformer_node`) | 10 | 10 | 100 % | `lin_rate_test.py` (Y) | rate state | full family covered |
| **Neurons total** | **66** | **24** | **36 %** | | | |
| Static + bernoulli + cont_delay + diffusion + gap + sic + rate-connection + ht_synapse | 9 | 9 | 100 % | `static_synapse_test.py` (Y) | weight / current | covered |
| STDP family (`stdp_*` × 9) | 9 | 9 | 100 % | `stdp_synapse_test.py` (Y) | weight trajectory | covered |
| STP family (`tsodyks*` × 3, `quantal_stp_synapse`) | 4 | 0 | **0 %** | — | n/a | **gap** |
| Clopath / Urbanczik / Jonke / Vogels-Sprekeler | 4 | 4 | 100 % | `clopath_synapse_test.py` (Y) | weight trajectory | covered |
| `volume_transmitter` | 1 | 0 | **0 %** | — | n/a | **gap** — couples to dopamine STDP |
| **Synapses + plasticity total** | **27** | **22** | **81 %** | | | |
| Generators (15: ac, dc, step current/rate, Poisson 4×, sinusoidal 2×, gamma sup, ppd sup, mip, pulsepacket, noise, spike) | 15 | 15 | 100 % | `poisson_generator_test.py` (Y) | spike times / current | covered |
| Recorders (multimeter, spike_recorder, weight_recorder) | 3 | 3 | 100 % | `multimeter_test.py` (Y) | event lists | covered |
| Detectors (4: correlation, correlomatrix, correlospinmatrix, spin) | 4 | 4 | 100 % | `correlation_detector_test.py` (Y) | covariance | covered |
| Spike utilities (`spike_dilutor`, `spike_train_injector`) | 2 | 2 | 100 % | `spike_dilutor_test.py` (Y) | spike times | covered (spike_train_injector also classed as neuron) |
| **Devices total** | **24** | **24** | **100 %** | | | |

(Numbers above sum across overlapping categories — `_base` excluded; `spike_train_injector` and `volume_transmitter` counted in their primary categories.)

## 4. Missing or incomplete functionality

**Families with zero NEST-comparison coverage:**

- IAF psc (10 models)
- IAF cond (5 models)
- IAF specialized (5 models)
- MAT (2 models)
- GIF (5 models)
- GLIF (3 models)
- HH (6 models)
- Izhikevich (1 model)
- Binary (3 models)
- Point process (2 models)
- STP family (4 plasticity rules)
- `volume_transmitter`
- `iaf_cond_alpha_mc` (the multi-compartment with no NEST test, alongside the
  validated `cm_default`)

**No documented tolerance/duration/dt convention.** All 69 modules with NEST
imports lack a per-test or per-family tolerance + duration + dt convention.
This is the structural reason no module is currently classified `implemented`
(spec §3).

**No shared validation harness.** Each `*_test.py` re-implements its own NEST
comparison glue: `nest.ResetKernel()`, parameter marshalling, multimeter
setup, trace alignment, tolerance assertions. This works but invites drift
(different tests use different conventions) and is a barrier to onboarding
new model validations.

**No reusable test infrastructure** for:
- multi-seed PRNG distributional comparisons (binary neurons, STP, Bernoulli
  synapses)
- weight-trajectory comparisons (STDP, STP — needed for the synapse
  promotions)
- spike-time precise comparisons (`*_ps` variants, `parrot_neuron_ps` when
  ported)
- ~~gap-junction waveform-relaxation convergence~~ **Resolved (cluster-15b)** — the port
  reproduces NEST's `use_wfr=False` regime with an explicit one-step-lagged difference
  current (no waveform relaxation to converge); parity is the 2-neuron micro-parity +
  distributional network coherence in `_validation/gap_junction_*parity_test.py`
- spatial / topology connectivity statistics (when spatial lands)

## 5. Semantic & numerical risks

- **PRNG divergence (cross-cutting).** NEST and JAX use independent PRNG
  streams. Spec §7 establishes bit-exact unsupported, distributional in
  scope. Concrete implication: validation tests for binary neurons, MIP
  generator, Poisson generators, Bernoulli synapses, quantal STP, etc., must
  use distributional metrics (mean firing rate, ISI CV, covariance) not
  per-event equality. Documented tolerance widens accordingly.
- **Integration-step coupling.** NEST uses fixed dt with min-delay-based
  slice scheduling and ring buffers. `brainpy.state` uses adaptive RKF45 for
  category A/C models and analytical propagators for category B. At matched
  dt these *should* converge but at the level of round-off; promoting any
  cat-A model to `implemented` requires verifying the RKF45 tolerance
  (`atol`, `rtol`) matches whatever NEST GSL settings the model uses.
- **Refractory rounding.** NEST rounds `t_ref` to a multiple of `dt`. Verify
  round-toward-zero vs. round-up convention per family.
- **Spike-threshold timing.** NEST checks `V_m >= V_th` after the propagator
  step. Most repo families likely match this; needs per-family confirmation.
- **Delay handling.** NEST uses min-delay-based scheduling; brainpy.state
  uses brainstate delay containers. *Behavior should match at matched delays
  but timing-edge cases (delay exactly at min_delay boundary, delay across
  Run/Cleanup) may differ.* Test harness should pin a specific delay regime.
- **Recording-device semantic divergence.** Cross-link `devices-gap.md` §5.
  Trace comparison harnesses must define whether they compare in-memory
  state arrays or backend-flushed events — these may differ in stamping.
- **Reset semantics.** `nest.ResetKernel()` destroys everything;
  brainpy.state has no kernel. The shared harness must wrap NEST-side
  ResetKernel + brainpy-side fresh-state construction in a single setUp
  helper.
- **Multi-seed averaging.** For PRNG-distributional tests, 1 seed is not
  enough. The harness should default to N ≥ 5 seeds and accept a tighter
  per-seed tolerance plus a looser distributional tolerance.

## 6. Validation infrastructure gaps

There is **no reusable harness** in
`brainpy_state/_nest/_validation/` (the directory itself does not exist).
Concretely, the harness should provide:

- `nest_compare(model_factory, params, dt, T, record, seeds, tol)` — runs the
  brainpy.state model and the matching NEST model under matched seeds, records
  the same recordable(s) on both, and asserts trace agreement within
  `tol = (atol, rtol)`.
- `pytest` marker `@pytest.mark.requires_nest` and conftest skipping (so the
  harness is opt-in: `pytest -m requires_nest`).
- Per-family base classes (`NESTNeuronComparisonCase`,
  `NESTSynapseComparisonCase`, `NESTDeviceComparisonCase`) that subclass
  `unittest.TestCase` and provide the boilerplate.
- A documented tolerance-naming convention:
  `cat_A_default = (atol=1e-3 mV, rtol=1e-3)`,
  `cat_B_default = (atol=1e-6 mV, rtol=1e-6)` (analytical propagator should
  be near-exact),
  `cat_C_default = (atol=1e-3 mV, rtol=1e-3)`,
  `distributional_default = mean_diff_pct=2 %, autocorr_max_diff=0.05`.
- A documented `T_default = 1000 ms`, `dt_default = 0.1 ms`, `n_seeds_default = 5`.

## 7. Prioritized roadmap

- **P0 — Build the shared NEST-comparison harness.** [M]
  Rationale: prerequisite for every other validation-related P0 across the
  per-axis docs (cross-link `neurons-gap.md` P0, `synapses-plasticity-gap.md`
  P0, `devices-gap.md` P0). Acceptance: `brainpy_state/_nest/_validation/`
  exists with `nest_compare.py`, `comparison_base.py`,
  `tolerance_conventions.py`, and a README documenting the harness.
  `@pytest.mark.requires_nest` is registered in `conftest.py`. At least 3
  existing tests (e.g. `aeif_cond_alpha_test.py`, `iaf_psc_alpha_test.py`,
  `multimeter_test.py`) are refactored to use it, *demonstrating no
  behavioral regression*.

- **P0 — Promote IAF psc family from 0 % validation coverage.** [L]
  Rationale: highest-priority family (most-used in NEST benchmarks).
  Cross-link `neurons-gap.md` P0. Acceptance: all 10 `iaf_psc_*` variants
  have NEST-comparison tests using the harness; each documents tolerance,
  duration, dt; CI green on a NEST-installed runner.

- **P0 — Promote IAF cond family from 0 % validation coverage.** [L]
  Cross-link `neurons-gap.md` P0. Acceptance: all 5 `iaf_cond_*` variants
  validated with the harness.

- **P0 — Validate the STP family (`tsodyks*`, `quantal_stp_synapse`,
  `volume_transmitter`).** [M]
  Cross-link `synapses-plasticity-gap.md` P0. Acceptance: 5 plasticity
  rules + 1 device validated with the harness.

- **P0 — Document tolerance conventions in a single page.** [S]
  Acceptance: `docs/nest-status/internal/numerical-validation-gap.md`
  becomes the canonical reference for tolerance defaults (linked from the
  harness README); per-category defaults (A/B/C/D/E) documented.

- **P1 — Promote AdEx family from `divergent` to `implemented`.** [M]
  Cross-link `neurons-gap.md` P1. Acceptance: all 9 `aeif_*` tests refactored
  to use the harness with documented tolerance.

- **P1 — Promote rate models from `divergent` to `implemented`.** [M]
  Acceptance: all 10 rate models refactored to use the harness with
  documented tolerance; `siegert_neuron` adds a mean-field-equivalence test
  separate from the trace test.

- **P1 — Validate GIF family.** [L]
  Acceptance: at least `gif_psc_exp`, `gif_cond_exp` validated; population
  variant (`gif_pop_psc_exp`) has its own population-statistics test.

- **P1 — Validate GLIF family.** [M]
  Acceptance: `glif_psc`, `glif_cond`, `glif_psc_double_alpha` validated.

- **P1 — Validate HH family.** [L]
  Acceptance: `hh_psc_alpha`, `hh_cond_exp_traub`, `hh_cond_beta_gap_traub`,
  `hh_psc_alpha_clopath`, `hh_psc_alpha_gap`, `ht_neuron` validated. Gap-junction
  variants are validated against NEST's `use_wfr=False` regime (explicit one-step
  lag, **not** waveform relaxation): `hh_psc_alpha_gap` ✓ done in cluster-15b
  (`_validation/gap_junction_parity_test.py` + the inhibitory-network parity);
  `hh_cond_beta_gap_traub` shares the seam (gap-capable) but its own gap-parity test
  is still pending. The remaining four HH models keep their single-neuron parity gap.

- **P1 — Validate MAT, Izhikevich, point-process families.** [M]
  Acceptance: `mat2_psc_exp`, `amat2_psc_exp`, `izhikevich`, `pp_psc_delta`
  validated (`pp_cond_exp_mc_urbanczik` ✓ done in cluster-21 via
  `_validation/urbanczik_synapse_parity_test.py`).

- **P1 — Validate binary stochastic family (distributional).** [S]
  Acceptance: `erfc_neuron`, `ginzburg_neuron`, `mcculloch_pitts_neuron`
  validated with distributional metrics (mean firing rate, autocorrelation).

- **P1 — Validate `iaf_cond_alpha_mc` (multi-compartment).** [M]
  Acceptance: compartment-tree topology and V_m traces match NEST.

- **P2 — Validate IAF specialized models (`iaf_bw_2001*`, `iaf_chs_2007`,
  `iaf_chxk_2008`, `iaf_tum_2000`).** [M]
  `iaf_bw_2001*` are NMDA-sensitive; document NMDA saturation regime in the
  test. Acceptance: each validated; `iaf_tum_2000` STP coupling to the
  synapse-side `tsodyks*` family is also exercised.

- **P2 — Add gap-junction parity test.** [M] — **DONE for `hh_psc_alpha_gap` (cluster-15b).**
  Validated against NEST's `use_wfr=False` regime (explicit one-step lag, not waveform
  relaxation): the 2-`hh_psc_alpha_gap` micro-parity matches NEST to machine precision
  between spikes (`_validation/gap_junction_parity_test.py`) and the inhibitory-network
  Golomb coherence matches distributionally
  (`_validation/gap_junction_inhibitory_network_parity_test.py`). Remaining: an analogous
  `hh_cond_beta_gap_traub` gap-parity regression (shares the seam; not yet covered).

- **P2 — CI parity-check matrix.** [M]
  Acceptance: GitHub Actions workflow runs the harness with NEST installed
  (separate from the default CI which doesn't require NEST); failures
  produce a regression report.

- **P2 — Validation-progress badge.** [S]
  Acceptance: per-family validation coverage is auto-computed from the
  harness's test markers and displayed in a generated table in
  `nest-status/index.rst`.
