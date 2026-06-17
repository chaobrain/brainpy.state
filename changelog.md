# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions,
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] – 2026-06-17

### Highlights

Version 0.1.0 is the first release in which `brainpy.state` is the single public
entry point for the package, and it consolidates both model families into a
coherent, documented whole. It introduces two complementary ways to build
networks — a high-level `Network` / `Builder` API and an explicit, NEST-vocabulary
`Simulator` — together with first-class synaptic delays, new analog input
generators and reduced neuron models, fully JIT-safe NEST-compatible models, and a
live-NEST validation harness that reproduces published Brunel benchmarks to
sub-percent accuracy. It also corrects several numerical bugs in the BrainPy-style
models.

### Added

#### Network construction

- **`Network` and `Builder`** — a high-level API for assembling models into a
  runnable network. `Network` is a `brainstate.nn.Module` with a projection-first
  update traversal and a JIT-compiled `simulate(duration, monitor=...)`; `Builder`
  produces the same module graph through an imperative `add()` / `connect()`
  interface. Includes rule-based projections (`OneToOneProj`, `AllToAllProj`,
  `PairwiseBernoulliProj`, `SymmetricPairwiseBernoulliProj`, `FixedIndegreeProj`,
  `FixedOutdegreeProj`, `FixedTotalNumberProj`, `PairwisePoissonProj`), a
  `Recorder` helper, and weight/parameter distributions (`dist.Normal`,
  `dist.LogNormal`, `dist.Uniform`).
- **`Simulator`** — an explicit network builder that mirrors NEST's vocabulary:
  `create(model, size, params=...)`, `connect(pre, post, *, rule, weight, delay,
  ...)`, and `simulate(duration)` returning a `SimulationResult` with `rate()`,
  `n_events()`, and `spikes()`. There is no global kernel; populations, devices,
  and projections form a flat module graph compiled into a single rollout.
  - **Population algebra** — `NodeView` supports concatenation (`ne + ni`) and
    slicing (`ne[:n]`), so one `connect` can target a combined or partial
    population.
  - **Connection rules** — `all_to_all`, `one_to_one`, `fixed_indegree`,
    `pairwise_bernoulli`, `fixed_total_number`, and more.
  - **Sparse or dense connectivity** — `connect(..., comm='sparse')` routes
    connectivity through a memory-light sparse event matmul (~1.9 GB at NEST's
    native `order=2500`); `comm='dense'` remains the default for small networks.
    Both paths share one sampler and produce identical results.
  - **Multi-receptor routing** — `connect(..., receptor_type=...)` delivers input
    to a specific receptor port of a multi-receptor neuron.

#### Synaptic delays (BrainPy-style)

- `AlignPostProj` and `CurrentProj` accept an optional, unit-carrying `delay=`
  keyword: a scalar applies a homogeneous conduction delay; a `(N_pre,)` array
  applies a per-presynaptic-neuron axonal delay. Sub-`dt` delays are linearly
  interpolated. The default `delay=None` is unchanged and adds no overhead.

#### Analog current generators (BrainPy-style)

- `SectionInput`, `ConstantInput`, `StepInput`, `RampInput`, `SinusoidalInput`,
  `WienerProcessInput`, and `OUProcessInput` — stateful current sources that emit
  the instantaneous current at the network's current time, staying synchronized
  with the simulation clock and composing under `jit` without materializing long
  offline arrays. Scalar parameters broadcast to `in_size`; the stochastic
  generators draw independent per-element samples.

#### Reduced neuron models (BrainPy-style)

- `FitzHughNagumo` (2-D excitable) and `HindmarshRose` (3-D bursting) reduced
  models, plus `CubaLIF` / `CobaLIF` convenience neurons that bundle an exponential
  synaptic term (current- and conductance-based, respectively).

#### Validation against live NEST

- A parity harness reproduces published benchmarks in both `brainpy.state` and a
  live NEST 3.x run, comparing population statistics (the random-number streams
  differ between simulators, so per-neuron matching is not meaningful). The tests
  skip automatically when NEST is not installed. Representative Brunel figures:

  | Benchmark (vs live NEST) | `brainpy.state` vs NEST | rel. error |
  |---|---|---|
  | Brunel alpha (`order=200`) | 56.9 vs 57.0 spks/s | 0.21 % |
  | Brunel alpha (`order=2500`) | 28.8 vs 28.5 spks/s | 0.91 % |
  | Brunel delta (`order=200`) | 58.5 vs 58.2 spks/s | 0.55 % |
  | Brunel exp-multisynapse (`order=200`, 4 seeds) | 25.8 vs 24.8 spks/s | ≈ 4 % |
  | Brunel siegert (mean-field) | 32.03 vs 32.03 spks/s | 0.00 % |
  | Brunel + evolution strategies | 51.5 vs 51.5 spks/s | 0.08 % |

#### Examples

- An expanded example library covering both families: BrainPy-style E/I networks
  and surrogate-gradient training, and faithful NEST ports including the Brunel
  variants (alpha, delta, exp-multisynapse, siegert, evolution strategies),
  single-neuron and plasticity demos, recording/device tours, spiking and
  rate-based networks, spatially-structured networks, and pedagogical models
  (AdEx figures, multi-compartment neurons, a reinforcement-learning Pong, and a
  Sudoku constraint solver). The Brunel flagship lives at
  `examples/brainpy_like/brunel.py`.

### Changed

- **Public API is `brainpy.state` only.** The package is now used exclusively
  through the `brainpy.state` namespace; a direct `import brainpy_state` from user
  code raises `ImportError` pointing to the supported path (an escape hatch remains
  for tooling). `brainpy.state.__version__` and `brainpy.state.__version_info__`
  are now exposed, and all examples, docs, and docstrings use the public namespace.
- **NEST-compatible models are JIT-safe.** Parameter validation is now tracer-aware
  (skipped during `jit` / `vmap` / `grad` tracing), and `erfc_neuron` /
  `ginzburg_neuron` no longer break under `jax.jit`. All 57 NEST neuron models are
  verified to trace under `jit`.
- **Supported Python is now ≥ 3.11**, and the declared dependency floors were
  aligned across the packaging metadata (notably `brainstate >= 0.5.0`).

### Fixed

- **BrainPy-style model correctness:**
  - *Hodgkin-Huxley family (`HH`, `MorrisLecar`, `WangBuzsakiHH`).* These models do
    not reset after a spike, so the previous per-step threshold test counted a
    single action potential as dozens of spikes (~55× over-count at `dt = 0.01 ms`).
    Spikes are now emitted once per upward threshold crossing, preserving a
    differentiable surrogate gradient.
  - *Short-term plasticity (`STP`, `STD`).* Transmitted output is computed from the
    resources available *before* depletion, so the first spike releases the correct
    amount.
  - *`WangBuzsakiHH`.* Corrected the K⁺ activation rate `αₙ`, which was 10× too large.
  - *`SpikeTime`.* Out-of-bounds time indices no longer read invalid rows; steps
    outside the defined window return zeros.
  - *`ExpIF` / `AdExIF` (and refractory variants).* The exponential spike-initiation
    term is clamped against overflow, preventing `inf` / `NaN` above the slope factor.
  - *Poisson inputs (`PoissonSpike`, `PoissonEncoder`, `poisson_input`).* Use the
    exact per-bin spike probability `1 − exp(−rate·dt)`, eliminating negative,
    fractional, or `NaN` counts at large `rate·dt`.
  - *`LeakyRateReadout`.* `tau` is sized to `out_size`, so a per-unit `tau` works
    when `in_size != out_size`.
  - *Gap junctions (`SymmetryGapJunction`, `AsymmetryGapJunction`).* Current inputs
    use a stable per-instance key, so repeated updates no longer leak input slots or
    accumulate stale currents.
  - *`Neuron` base class.* `spk_reset` is validated at construction (must be
    `'soft'` or `'hard'`).
- **Network simulation.** Projections and generators fanned out to several targets
  now derive a distinct, reproducible seed per target; previously a shared base
  seed gave every target identical connectivity and external drive.

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
| brainunit | latest |
| brainevent | 0.0.4 |
| braintools | 0.0.9 |
| numpy | 1.15 |

[0.1.0]: https://github.com/chaobrain/brainpy.state/compare/v0.0.4...v0.1.0
[0.0.4]: https://github.com/chaobrain/brainpy.state/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/chaobrain/brainpy.state/compare/v0.0.1...v0.0.3
[0.0.1]: https://github.com/chaobrain/brainpy.state/releases/tag/v0.0.1
