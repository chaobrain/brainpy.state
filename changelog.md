# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions,
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Unreleased

### Fixed — Correctness bugs in BrainPy-style (`_brainpy`) models

- **HH-family spike output (`HH`, `MorrisLecar`, `WangBuzsakiHH`).** These models
  do not reset the membrane potential after a spike, so `V` stays above threshold
  for the whole action potential. `update()` previously returned a spike on *every*
  step above threshold, counting a single action potential as dozens of spikes
  (~55× over-count at `dt = 0.01 ms`). It now emits one spike per upward threshold
  crossing (rising-edge detection), with no new state and a still-differentiable
  surrogate gradient.
- **Short-term plasticity (`STP`, `STD`).** Transmitted output is now computed from
  the resources available *before* depletion, so the first spike releases the
  correct amount instead of the already-depleted value.
- **`WangBuzsakiHH` n-gate kinetics.** Corrected the K⁺ activation opening rate
  `αₙ`, which was 10× too large.
- **`SpikeTime`.** Out-of-bounds time indices no longer read invalid rows; steps
  outside the defined window return zeros.
- **`ExpIF` / `AdExIF` (and refractory variants).** The exponential spike-initiation
  term is now guarded against overflow (clamped argument), preventing `inf`/`NaN`
  for membrane potentials above the threshold slope factor.
- **Poisson inputs (`PoissonSpike`, `PoissonEncoder`, `poisson_input`).** Use the
  exact per-bin spike probability `1 − exp(−rate·dt)`; the Gaussian-approximation
  branch no longer yields negative, fractional, or `NaN` spike counts when
  `rate·dt` is large.
- **`LeakyRateReadout`.** `tau` is now sized to `out_size` (the readout-state
  dimension), so a per-unit `tau` works when `in_size != out_size`.
- **Gap junctions (`SymmetryGapJunction`, `AsymmetryGapJunction`).** Current inputs
  are registered under a stable per-instance key instead of one derived from the
  live input count; repeated updates no longer leak new input slots or accumulate
  stale currents.
- **`Neuron` base class.** `spk_reset` is validated at construction (must be
  `'soft'` or `'hard'`).

### Added — Synaptic delays on projections (`_brainpy`)

- `AlignPostProj` and `CurrentProj` now accept an optional, unit-carrying
  `delay=` keyword. A scalar (e.g. `delay=1.5 * u.ms`) applies a global,
  homogeneous conduction delay; a `(N_pre,)` array applies an axonal,
  per-pre-neuron delay. Sub-`dt` delays are linearly interpolated, so the delay
  need not be an integer multiple of `dt`. `delay=None` (the default) keeps the
  original code path with zero overhead and bit-for-bit identical output. The
  delay is backed by a single shared `brainstate.nn.Delay` buffer over the
  pre-synaptic input (new internal `InputDelay` seam), sized once from
  `ceil(max(delay) / dt)` at `init_state`.
- The `InputDelay` seam also implements the heterogeneous, per-connection read
  (`delay` of shape `(N_syn,)` plus `indices=pre_ids`) via the diagonal gather
  `retrieve_at_step(steps, pre_ids)` — the same buffer, only the gather index
  changes — with `len(delay) == len(indices)` validated at `init_state`. This is
  the mechanism intended to back explicit-connectivity projections; the
  comm-callable `AlignPostProj`/`CurrentProj` expose the global and axonal
  granularities. Gap junctions take no `delay=` (electrical coupling is treated
  as instantaneous), so a passed delay raises rather than being silently ignored.

### Added — Analog current generators (`_brainpy`)

- Per-step analog current sources (`brainpy.state` namespace): `SectionInput`,
  `ConstantInput`, `StepInput`, `RampInput`, `SinusoidalInput`,
  `WienerProcessInput`, and `OUProcessInput`. Unlike `braintools.input` (which
  precomputes an offline array for a whole run), these are stateful `Module`s
  that emit the instantaneous current at the network's current time, so they
  compose with `add_current_input` and stay synchronized with the simulation
  clock without materializing long arrays under `jit`. Scalar parameters
  broadcast to `in_size`; the stochastic generators draw independent per-element
  samples (noise scaled by `√dt`), and `OUProcessInput` matches the
  `braintools.input.ou_process` discretization.

### Added — Reduced neuron models (`_brainpy`)

- `FitzHughNagumo` (2-D excitable) and `HindmarshRose` (3-D bursting) reduced
  models. Both are dimensionless systems with physical time constants and,
  like the Hodgkin-Huxley family, **no reset** — so their spike output uses the
  rising-edge detector (one spike per upward threshold crossing), not a per-step
  threshold test.
- `CubaLIF` and `CobaLIF`: convenience LIF neurons with a bundled exponential
  synaptic term (current-based and conductance-based, respectively), so the
  common `LIF + Expon + CUBA/COBA` wiring is available as a single model. Delta
  inputs feed the built-in synapse; current inputs / the `x` argument are the
  external membrane current.

### Added — Design notes

- `DESIGN_delays_and_missing_features.md` — proposal for first-class synaptic
  delays (`delay=` on projections, built on `brainstate` delay primitives), plus
  missing input generators (section/step/ramp/sinusoidal/Ornstein–Uhlenbeck/Wiener)
  and reduced neuron models (FitzHugh–Nagumo, Hindmarsh–Rose, `CobaLIF`/`CubaLIF`),
  with an architecture assessment.

### Changed — JIT-safe parameter validation for NEST neurons

- Added `brainpy_state._nest._utils.cond_any`, a shared tracer-aware reduction
  helper: it returns `False` when its condition is a JAX tracer (so `if`
  validation checks are skipped during `jit`/`vmap`/`grad` tracing) and
  `bool(np.any(...))` otherwise. All `NESTNeuron` parameter-validation checks
  (`if np.any(...)` / `if u.math.any(...)`) now route through it.
- `erfc_neuron` and `ginzburg_neuron`: removed a Python `if bool(any(...))`
  branch inside `update()` that broke under `jax.jit`; the per-neuron update is
  now always computed and masked with `where`, making both models JIT-compatible.
- Added `brainpy_state/_nest/jit_compat_test.py` verifying every public
  `NESTNeuron` subclass traces under `jit` (57 models), with architecturally
  NumPy-scalar models (precise-spiking, mean-field, delay-queue) documented.

### Added — Network API for NEST-style models

- `brainpy.state.Network` — `brainstate.nn.Module` subclass with
  projection-first `update()` traversal and JIT-wrapped
  `simulate(duration, monitor=...)`.
- `brainpy.state.Builder` — imperative subclass exposing `add()` and
  `connect()`; produces the same underlying module tree as a subclassed
  `Network`.
- Rule-based projections: `OneToOneProj`, `AllToAllProj`,
  `PairwiseBernoulliProj`, `SymmetricPairwiseBernoulliProj`,
  `FixedIndegreeProj`, `FixedOutdegreeProj`, `FixedTotalNumberProj`,
  `PairwisePoissonProj`. Uniform constructor `(pre, post, *, weight,
  delay=None, syn, out, allow_autapses, allow_multapses, seed,
  **rule_kwargs)`. `delay=` support is deferred to a follow-up — v1
  accepts `delay=None` only.
- `brainpy.state.Recorder` — helper that wires a passive `NESTDevice`
  recorder to a source population (string attribute or callable).
- `brainpy.state.dist.{Normal, LogNormal, Uniform}` — distribution
  objects sampled once at projection `__init__`.
- Brunel flagship example at `examples/brunel.py`.

See `docs/superpowers/specs/2026-05-12-nest-network-api-design.md` for
the design and `docs/superpowers/plans/2026-05-12-nest-network-api.md`
for the implementation plan.

---

## [0.0.4] – 2025-02-21

### Highlights

Version 0.0.4 is a major feature release that introduces a comprehensive library of
NEST-compatible neural models (initial version), reorganizes the public API into dedicated `_brainpy` and
`_nest` submodules, and transitions the project to the Apache 2.0 license.

### Added

#### NEST-Compatible Model Library

A complete port of the NEST simulator model catalogue, covering more than 250 models
across all major categories:

**Neuron Models**
- *Integrate-and-Fire (IAF)*: `iaf_psc_alpha`, `iaf_psc_alpha_multisynapse`,
  `iaf_psc_alpha_ps`, `iaf_psc_delta`, `iaf_psc_delta_ps`, `iaf_psc_exp`,
  `iaf_psc_exp_htum`, `iaf_psc_exp_multisynapse`, `iaf_psc_exp_ps`,
  `iaf_psc_exp_ps_lossless`, `iaf_cond_alpha`, `iaf_cond_alpha_mc`,
  `iaf_cond_beta`, `iaf_cond_exp`, `iaf_cond_exp_sfa_rr`, `iaf_bw_2001`,
  `iaf_bw_2001_exact`, `iaf_chs_2007`, `iaf_chxk_2008`, `iaf_tum_2000`
- *Adaptive Exponential IF (AdEx / aeif)*: `aeif_cond_alpha`,
  `aeif_cond_alpha_astro`, `aeif_cond_alpha_multisynapse`,
  `aeif_cond_beta_multisynapse`, `aeif_cond_exp`, `aeif_psc_alpha`,
  `aeif_psc_delta`, `aeif_psc_delta_clopath`, `aeif_psc_exp`
- *Generalized IF (GIF)*: `gif_cond_exp`, `gif_cond_exp_multisynapse`,
  `gif_pop_psc_exp`, `gif_psc_exp`, `gif_psc_exp_multisynapse`
- *Multi-timescale Adaptive Threshold (MAT)*: `amat2_psc_exp`, `mat2_psc_exp`
- *Generalized LIF (GLIF)*: `glif_cond`, `glif_psc`, `glif_psc_double_alpha`
- *Hodgkin-Huxley family*: `hh_cond_beta_gap_traub`, `hh_cond_exp_traub`,
  `hh_psc_alpha`, `hh_psc_alpha_clopath`, `hh_psc_alpha_gap`, `ht_neuron`
- *Izhikevich*: `izhikevich`
- *Point-process neurons*: `pp_cond_exp_mc_urbanczik`, `pp_psc_delta`
- *Binary neurons*: `erfc_neuron`, `ginzburg_neuron`, `mcculloch_pitts_neuron`
- *Rate neurons*: `gauss_rate_ipn`, `lin_rate_ipn`, `lin_rate_opn`,
  `rate_neuron_ipn`, `rate_neuron_opn`, `rate_transformer_node`,
  `siegert_neuron`, `sigmoid_rate_ipn`, `sigmoid_rate_gg_1998_ipn`,
  `tanh_rate_ipn`, `tanh_rate_opn`, `threshold_lin_rate_ipn`,
  `threshold_lin_rate_opn`
- *Miscellaneous*: `ignore_and_fire`

**Synapse Models**
- *Static*: `static_synapse`, `static_synapse_hom_w`, `cont_delay_synapse`,
  `bernoulli_synapse`
- *Short-term plasticity*: `tsodyks_synapse`, `tsodyks_synapse_hom`,
  `tsodyks2_synapse`, `quantal_stp_synapse`
- *STDP*: `stdp_synapse`, `stdp_synapse_hom`, `stdp_dopamine_synapse`,
  `stdp_facetshw_synapse_hom`, `stdp_nn_pre_centered_synapse`,
  `stdp_nn_restr_synapse`, `stdp_nn_symm_synapse`, `stdp_pl_synapse_hom`,
  `stdp_triplet_synapse`
- *Voltage-based / specialized*: `clopath_synapse`, `ht_synapse`,
  `jonke_synapse`, `urbanczik_synapse`, `vogels_sprekeler_synapse`
- *Structural connections*: `diffusion_connection`, `gap_junction`,
  `rate_connection_delayed`, `rate_connection_instantaneous`, `sic_connection`

**Stimulation Devices**
- *Current generators*: `ac_generator`, `dc_generator`, `noise_generator`,
  `step_current_generator`, `step_rate_generator`
- *Spike generators*: `spike_generator`, `spike_train_injector`, `spike_dilutor`
- *Poisson generators*: `poisson_generator`, `poisson_generator_ps`,
  `inhomogeneous_poisson_generator`, `sinusoidal_poisson_generator`
- *Other generators*: `gamma_sup_generator`, `mip_generator`,
  `ppd_sup_generator`, `pulsepacket_generator`, `sinusoidal_gamma_generator`

**Recording Devices**
- `correlation_detector`, `correlomatrix_detector`,
  `correlospinmatrix_detector`, `multimeter`, `spike_recorder`,
  `spin_detector`, `volume_transmitter`, `weight_recorder`

**Specialised Models**
- `astrocyte_lr_1994`: Leaky integrator astrocyte model
- `cm_default`: Multi-compartment neuron model

#### NEST Base Infrastructure
- `NESTNeuron`, `NESTSynapse`, `NESTDevice`: abstract base classes for all
  NEST-compatible models, providing shared parameter management and state
  initialisation utilities (`_nest/_base.py`, `_nest/_utils.py`)

#### BrainPy-style Model Enhancements
- `SpikeTime`: added `weight` parameter and time-rounding option
- `AlignPostProj`, `DeltaProj`, `CurrentProj`: new projection variants
- `align_pre_projection`, `align_post_projection`: projection utility functions
- `SymmetryGapJunction`, `AsymmetryGapJunction`: gap junction projection types
- `PoissonEncoder`, `PoissonInput`, `poisson_input`: additional input generators
- `LeakyRateReadout`, `LeakySpikeReadout`: renamed and expanded readout classes

### Changed

- **API layout**: public models reorganised into `brainpy_state._brainpy`
  (BrainPy-style models) and `brainpy_state._nest` (NEST-compatible models)
  subpackages; all symbols remain importable from the top-level namespace
- **License**: changed from GNU GPLv3 to Apache License 2.0
- **Data types**: model state variables now use `brainstate.environ.dftype()`
  for consistent default floating-point precision across the ecosystem
- **Dependency**: minimum `brainpy` requirement raised to `>= 2.7.6`
- **NEST models**: all NEST models refactored onto shared base classes and
  utility helpers, eliminating duplicated boilerplate across model files
- **`aeif_cond_alpha`**: streamlined initialisation of `integration_step` and
  `I_stim` fields
- Documentation: mathematical equations and parameter descriptions expanded
  and standardised across all BrainPy-style and NEST-compatible model files

### Fixed

- Documentation URLs updated in `CONTRIBUTING.md`, `config.yml`, and
  `pyproject.toml` to point to the correct hosted locations
- Ecosystem cross-references in `README.md` and `index.rst` corrected

---

## [0.0.3] – 2025-01-01

### Highlights

Version 0.0.3 consolidates the package rename from `brainpy.state` to
`brainpy_state`, adds `brainpy` as a declared runtime dependency, and
tightens internal state initialisation.

### Added

- `brainpy` added as an explicit runtime dependency in `requirements.txt`
  and `pyproject.toml`

### Changed

- Package renamed from `brainpy.state` to `brainpy_state`; all public import
  paths updated accordingly
- `HiddenState` initialisation refactored for correctness and clarity
- Function names updated and a simulation example added to the main script
- Directory structure reorganised in preparation for the `_brainpy` /
  `_nest` split introduced in 0.0.4
- Minimum `brainpy` requirement formalised

### Fixed

- Import statements that still referenced the old `brainpy.state` namespace
  corrected throughout the codebase

---

## [0.0.1] – 2024-12-01

*Initial release of `brainpy_state`.*

`brainpy_state` modernises the [BrainPy](https://github.com/brainpy/BrainPy)
spiking neural network simulator by adopting the state-based programming model
introduced in [brainstate](https://github.com/chaobrain/brainstate).

### Added

#### Neuron Models

- **Integrate-and-Fire (LIF) family**
  - `IF`: basic integrate-and-fire neuron
  - `LIF`, `LIFRef`: leaky integrate-and-fire (with optional refractory period)
  - `ExpIF`, `ExpIFRef`: exponential integrate-and-fire
  - `AdExIF`, `AdExIFRef`: adaptive exponential integrate-and-fire
  - `ALIF`: adaptive leaky integrate-and-fire
  - `QuaIF`: quadratic integrate-and-fire
  - `AdQuaIF`, `AdQuaIFRef`: adaptive quadratic integrate-and-fire
  - `Gif`, `GifRef`: generalized integrate-and-fire

- **Hodgkin-Huxley family**
  - `HH`: classic Hodgkin-Huxley conductance-based neuron
  - `MorrisLecar`: Morris-Lecar neuron
  - `WangBuzsakiHH`: Wang-Buzsaki modified Hodgkin-Huxley neuron

- **Izhikevich family**
  - `Izhikevich`, `IzhikevichRef`: Izhikevich neuron (with optional refractory
    period)

#### Synapse Models

- **Exponential synapses**: `Expon` (single exponential decay), `DualExpon`
  (dual exponential rise-and-decay)
- **Receptor-based synapses**: `Alpha`, `AMPA`, `GABAa`, `BioNMDA`
- **Short-term plasticity**: `STP` (facilitation and depression), `STD`
  (pure depression)

#### Infrastructure

- `Neuron`, `Synapse`, `Dynamics`: abstract base classes for custom model
  development
- `Projection`, `AlignPostProj`: network projection utilities
- `COBA`, `CUBA`, `MgBlock`: synaptic output current handlers
- `SpikeTime`, `PoissonSpike`: spike-train input generators
- `Readout`, `LeakyReadout`, `WeightedReadout`: readout layer implementations
- Runtime compatibility check: raises an informative error when an
  incompatible `brainpy` version (< 2.7.4) is detected

#### Dependencies

| Package | Minimum version |
|---------|----------------|
| Python | 3.10 |
| jax | latest |
| brainstate | 0.2.0 |
| saiunit | latest |
| brainevent | 0.0.4 |
| braintools | 0.0.9 |
| numpy | 1.15 |

---

[0.0.4]: https://github.com/chaobrain/brainpy.state/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/chaobrain/brainpy.state/compare/v0.0.1...v0.0.3
[0.0.1]: https://github.com/chaobrain/brainpy.state/releases/tag/v0.0.1
