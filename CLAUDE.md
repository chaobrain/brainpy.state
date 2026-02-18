# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**brainpy.state** provides stateful spiking neural network models for [BrainPy](https://github.com/brainpy/BrainPy), built on JAX. It re-implements classic neuron models (LIF, HH, Izhikevich), synapse models, projections, and input generators using the [brainstate](https://github.com/chaobrain/brainstate) state management system. All classes are exposed under the `brainpy.state` namespace.

**brainpy.state** is benchmarked against the **NEST simulator**  (https://github.com/nest/nest-simulator) and aims to implement the vast majority of neuron, synapse, and plasticity models available in NEST. 

Part of the BrainX ecosystem: `brainpy`, `brainstate`, `brainunit`, `braintools`, `brainevent`.

Reference of NEST documentation: https://nest-simulator.readthedocs.io/en/stable/models/index.html

Reference of NEST-GPU documentation: https://nest-gpu.readthedocs.io/en/latest/contents.html


## Commands

```bash
# Install for development (requires flit)
pip install flit && pip install -e .

# Run all tests
pytest brainpy_state/

# Run a single test file
pytest brainpy_state/lif_test.py -v

# Run tests in parallel
pytest brainpy_state/ -n auto

# Lint (pre-commit hooks: flake8, trailing-whitespace, end-of-file-fixer, debug-statements)
pre-commit run --all-files

# Build package
flit build
```

## Architecture

### Package Layout

Flat structure under `brainpy_state/` — all modules are prefixed with `_` (private convention). Tests are co-located as `_*_test.py` files.

### Class Hierarchy

```
brainstate.nn.Dynamics
  └─ Dynamics (_base.py)        # Adds current/delta input dicts, align_pre()
      ├─ Neuron (_base.py)      # Adds spk_fun, spk_reset, get_spike()
      │    ├─ IF, LIF, LIFRef, ALIF, ExpIF, AdExIF, QuaIF, ... (_lif.py)
      │    ├─ HH, MorrisLecar, WangBuzsakiHH (_hh.py)
      │    ├─ Izhikevich, IzhikevichRef (_izhikevich.py)
      │    └─ LeakySpikeReadout (_readout.py)
      └─ Synapse (_base.py)     # Base for synapse dynamics
           ├─ Expon, DualExpon (_exponential.py)
           ├─ Alpha, AMPA, GABAa, BioNMDA (_synapse.py)
           └─ STP, STD (_stp.py)

brainstate.nn.Module
  ├─ Projection (_projection.py)
  │    ├─ AlignPostProj, DeltaProj, CurrentProj
  │    ├─ align_pre_projection, align_post_projection
  │    └─ SymmetryGapJunction, AsymmetryGapJunction (_synaptic_projection.py)
  ├─ SynOut (_synouts.py)  →  COBA, CUBA, MgBlock
  └─ LeakyRateReadout (_readout.py)
```

### Key Design Patterns

- **Input management**: `Dynamics` base class maintains `current_inputs` and `delta_inputs` dicts. Projections register inputs via `add_current_input()`/`add_delta_input()` with string labels.
- **Physical units**: All parameters use `brainunit` quantities (`u.mV`, `u.ms`, `u.mS`, etc.). Use `brainunit` for any new parameters.
- **State types**: Models declare states as `brainstate.HiddenState` (membrane potential, etc.), `brainstate.ShortTermState` (spike output), or `brainstate.ParamState`.
- **Numerical integration**: ODEs integrated with `brainstate.nn.exp_euler_step`.
- **Surrogate gradients**: Neurons use `spk_fun` (from `braintools.surrogate`) for differentiable spike generation. Spike reset supports `'soft'` (V -= V_th) and `'hard'` (stop_gradient) modes.
- **Mixins** (`_mixin.py`): `AlignPost` for post-synaptic input alignment; `BindCondData` for temporary conductance storage between computation steps.
- **Module naming**: All public classes set `__module__ = 'brainpy.state'` via `@set_module_as('brainpy.state')` decorator, so they appear under `brainpy.state.*` in docs and imports.

### Test Conventions

- Tests use `unittest.TestCase` style with `setUp`/`test_*` methods.
- Simulation context is set via `brainstate.environ.context(dt=0.1 * u.ms)`.
- State initialization uses `model.init_all_states()`.

---

## Gap Analysis: brainpy.state vs NEST Simulator

*Generated 2026-02-15 from source analysis of brainpy.state v0.0.3 and NEST 3.9+*

This section catalogs the complete model inventories of both frameworks, identifies missing
models in brainpy.state, and provides a prioritized implementation roadmap.

### Current brainpy.state Model Inventory

#### Neuron Models (18 models)

| Model | File | NEST Equivalent(s) |
|-------|------|---------------------|
| IF | `_lif.py` | (perfect integrator, no direct NEST equiv) |
| LIF | `_lif.py` | `iaf_psc_exp` / `iaf_psc_alpha` |
| LIFRef | `_lif.py` | `iaf_psc_exp` (with t_ref) |
| ALIF | `_lif.py` | `iaf_cond_exp_sfa_rr` (partial) |
| ExpIF | `_lif.py` | `aeif_psc_exp` (without adaptation) |
| ExpIFRef | `_lif.py` | `aeif_psc_exp` (without adaptation, with t_ref) |
| AdExIF | `_lif.py` | `aeif_psc_exp` / `aeif_cond_exp` |
| AdExIFRef | `_lif.py` | `aeif_psc_exp` (with t_ref) |
| QuaIF | `_lif.py` | (no NEST equiv; quadratic IF) |
| AdQuaIF | `_lif.py` | (no NEST equiv; adaptive quadratic IF) |
| AdQuaIFRef | `_lif.py` | (no NEST equiv) |
| Gif | `_lif.py` | `gif_psc_exp` |
| GifRef | `_lif.py` | `gif_psc_exp` (with refractory) |
| HH | `_hh.py` | `hh_psc_alpha` |
| MorrisLecar | `_hh.py` | (no NEST equiv) |
| WangBuzsakiHH | `_hh.py` | (no NEST equiv) |
| Izhikevich | `_izhikevich.py` | `izhikevich` |
| IzhikevichRef | `_izhikevich.py` | `izhikevich` (NEST has no ref variant) |

#### Synapse Models (6 models)

| Model | File | NEST Equivalent |
|-------|------|-----------------|
| Expon | `_exponential.py` | exponential PSC kernel |
| DualExpon | `_exponential.py` | beta/dual-exp kernel |
| Alpha | `_synapse.py` | alpha PSC kernel |
| AMPA | `_synapse.py` | AMPA receptor kinetics |
| GABAa | `_synapse.py` | GABAa receptor kinetics |
| BioNMDA | `_synapse.py` | NMDA with Mg²⁺ block |

#### Short-Term Plasticity (2 models)

| Model | File | NEST Equivalent |
|-------|------|-----------------|
| STP | `_stp.py` | `tsodyks_synapse` / `tsodyks2_synapse` |
| STD | `_stp.py` | (depression-only subset) |

#### Synaptic Output / Projections / Other

| Category | Models |
|----------|--------|
| Synaptic output | COBA, CUBA, MgBlock |
| Projections | AlignPostProj, DeltaProj, CurrentProj, align_pre_projection, align_post_projection |
| Gap junctions | SymmetryGapJunction, AsymmetryGapJunction |
| Readouts | LeakyRateReadout, LeakySpikeReadout |
| Input generators | SpikeTime, PoissonSpike, PoissonEncoder, PoissonInput |

### Complete NEST Model Inventory (for comparison)

#### NEST Spiking Neuron Models (60+ models)

**IAF current-based (psc):**
`iaf_psc_delta`, `iaf_psc_delta_ps`, `iaf_psc_alpha`, `iaf_psc_alpha_multisynapse`,
`iaf_psc_alpha_ps`, `iaf_psc_exp`, `iaf_psc_exp_multisynapse`, `iaf_psc_exp_ps`,
`iaf_psc_exp_ps_lossless`, `iaf_psc_exp_htum`

**IAF conductance-based (cond):**
`iaf_cond_alpha`, `iaf_cond_alpha_mc` (multi-compartment), `iaf_cond_beta`, `iaf_cond_exp`,
`iaf_cond_exp_sfa_rr` (spike-frequency adaptation + relative refractory)

**IAF specialized variants:**
`iaf_bw_2001` / `iaf_bw_2001_exact` (Brunel-Wang with NMDA),
`iaf_chs_2007` (Carandini spike-response), `iaf_chxk_2008` (Casti precise timing),
`iaf_tum_2000` (with integrated STP)

**Adaptive Exponential IF (aeif):**
`aeif_cond_alpha`, `aeif_cond_alpha_astro` (astrocyte interaction),
`aeif_cond_alpha_multisynapse`, `aeif_cond_beta_multisynapse`, `aeif_cond_exp`,
`aeif_psc_alpha`, `aeif_psc_delta`, `aeif_psc_delta_clopath`, `aeif_psc_exp`

**Generalized IF (gif):**
`gif_cond_exp`, `gif_cond_exp_multisynapse`, `gif_pop_psc_exp` (population-level),
`gif_psc_exp`, `gif_psc_exp_multisynapse`

**Multi-timescale Adaptive Threshold (mat):**
`mat2_psc_exp`, `amat2_psc_exp`

**Generalized LIF – Allen Institute (glif):**
`glif_cond`, `glif_psc`, `glif_psc_double_alpha`

**Hodgkin-Huxley family:**
`hh_psc_alpha`, `hh_psc_alpha_clopath`, `hh_psc_alpha_gap`,
`hh_cond_exp_traub`, `hh_cond_beta_gap_traub`, `ht_neuron` (Hill-Tononi)

**Point process / other spiking:**
`izhikevich`, `pp_psc_delta`, `pp_cond_exp_mc_urbanczik`, `ignore_and_fire`

**Binary neurons:**
`mcculloch_pitts_neuron`, `ginzburg_neuron`, `erfc_neuron`

**Rate neurons:**
`lin_rate`, `tanh_rate`, `sigmoid_rate`, `sigmoid_rate_gg_1998`, `gauss_rate`,
`threshold_lin_rate`, `rate_neuron_ipn`, `rate_neuron_opn`, `rate_transformer_node`,
`siegert_neuron` (mean-field)






**Multi-compartment:** `cm_default`

**e-prop (eligibility propagation):**
`eprop_iaf`, `eprop_iaf_adapt`, `eprop_readout`,
`eprop_iaf_bsshslm_2020`, `eprop_iaf_adapt_bsshslm_2020`, `eprop_readout_bsshslm_2020`,
`eprop_iaf_psc_delta`, `eprop_iaf_psc_delta_adapt`

**Astrocyte:** `astrocyte_lr_1994`

#### NEST Synapse / Plasticity Models (30+ models)

**Static connections:**
`static_synapse`, `static_synapse_hom_w`, `bernoulli_synapse` (stochastic),
`cont_delay_synapse`

**Short-term plasticity:**
`tsodyks_synapse`, `tsodyks_synapse_hom`, `tsodyks2_synapse`, `quantal_stp_synapse`

**Spike-timing dependent plasticity (STDP):**
`stdp_synapse` (classical pair-based), `stdp_synapse_hom`,
`stdp_pl_synapse_hom` (power-law STDP),
`stdp_nn_pre_centered_synapse`, `stdp_nn_restr_synapse`, `stdp_nn_symm_synapse`
(nearest-neighbor variants),
`stdp_triplet_synapse` (Pfister-Gerstner triplet rule),
`stdp_dopamine_synapse` (reward-modulated),
`stdp_facetshw_synapse_hom` (hardware emulation)

**Voltage-based / advanced STDP:**
`clopath_synapse` (voltage-based STDP), `jonke_synapse` (STDP with additive factors)

**Supervised / other plasticity:**
`urbanczik_synapse` (two-compartment supervised), `vogels_sprekeler_synapse` (inhibitory STDP),
`eprop_synapse`, `eprop_synapse_bsshslm_2020`,
`eprop_learning_signal_connection`, `eprop_learning_signal_connection_bsshslm_2020`

**Gap junctions / special connections:**
`gap_junction`, `diffusion_connection`, `rate_connection_delayed`,
`rate_connection_instantaneous`, `sic_connection` (slow inward current, astrocyte),
`ht_synapse` (Hill-Tononi depression)





#### NEST Devices (30+ models)

**Stimulation:** `dc_generator`, `ac_generator`, `noise_generator`, `step_current_generator`,
`step_rate_generator`, `spike_generator`, `spike_train_injector`,
`inhomogeneous_poisson_generator`, `poisson_generator`, `poisson_generator_ps`,
`sinusoidal_poisson_generator`, `sinusoidal_gamma_generator`,
`gamma_sup_generator`, `ppd_sup_generator`, `pulsepacket_generator`,
`mip_generator`, `spike_dilutor`

**Recording:** `multimeter`, `spike_recorder`, `weight_recorder`,
`spin_detector`, `correlation_detector`, `correlomatrix_detector`,
`correlospinmatrix_detector`, `volume_transmitter`




### Missing Models in brainpy.state (Gap Summary)

#### PRIORITY 1 — Core Neuron Models (High Impact, Widely Used)

These are the most commonly used NEST models with no brainpy.state equivalent:

| Missing Model | Description | Complexity | Impact |
|---------------|-------------|------------|--------|
| **IAF-cond variants** | `iaf_cond_alpha`, `iaf_cond_exp`, `iaf_cond_beta` — conductance-based LIF with alpha/exp/beta synapses | Low | **Critical** — most NEST tutorials and benchmarks use these |
| **IAF-psc-delta** | `iaf_psc_delta` — LIF with delta-function PSCs | Low | **High** — simplest/fastest model, used in large-scale sims |
| **IAF-psc-alpha** | `iaf_psc_alpha` — LIF with alpha-function PSCs | Low | **High** — NEST's default tutorial model |
| **Multisynapse variants** | `*_multisynapse` — neurons with multiple receptor ports | Medium | **High** — essential for multi-receptor networks |
| **AdEx conductance** | `aeif_cond_alpha`, `aeif_cond_exp`, `aeif_cond_beta_multisynapse` — conductance-based AdEx | Low | **High** — standard cortical models |

#### PRIORITY 2 — Important Plasticity Models (No STDP in brainpy.state)

brainpy.state currently has **zero** long-term plasticity / STDP models. This is the largest gap.

| Missing Model | Description | Complexity | Impact |
|---------------|-------------|------------|--------|
| **STDP (pair-based)** | `stdp_synapse` — classical asymmetric STDP | Medium | **Critical** — most basic learning rule |
| **STDP (triplet)** | `stdp_triplet_synapse` — Pfister-Gerstner triplet rule | Medium | **High** — better fit to experimental data |
| **STDP (dopamine)** | `stdp_dopamine_synapse` — reward-modulated STDP | High | **High** — reinforcement learning |
| **Vogels-Sprekeler** | `vogels_sprekeler_synapse` — inhibitory STDP for E/I balance | Medium | **High** — widely used for balanced networks |
| **Clopath** | `clopath_synapse` — voltage-based STDP | High | **Medium** — voltage-dependent learning |
| **STDP variants** | power-law, nearest-neighbor, homogeneous | Medium | **Medium** — complete STDP coverage |

#### PRIORITY 3 — Specialized Neuron Models

| Missing Model | Description | Complexity | Impact |
|---------------|-------------|------------|--------|
| **GLIF** | `glif_psc`, `glif_cond`, `glif_psc_double_alpha` — Allen Institute GLIF 1-5 | High | **High** — data-driven modeling |
| **MAT2** | `mat2_psc_exp`, `amat2_psc_exp` — multi-timescale adaptive threshold | Medium | **Medium** — good biological accuracy |
| **Brunel-Wang** | `iaf_bw_2001` / `iaf_bw_2001_exact` — LIF with NMDA for working memory | Medium | **High** — decision-making models |
| **IAF-SFA-RR** | `iaf_cond_exp_sfa_rr` — spike-frequency adaptation + relative refractory | Medium | **Medium** — adaptation modeling |
| **HH with gap junctions** | `hh_psc_alpha_gap`, `hh_cond_beta_gap_traub` | Medium | **Medium** — oscillation studies |
| **HH Traub** | `hh_cond_exp_traub` — Brette et al. benchmark model | Low | **Medium** — standard benchmark |
| **Hill-Tononi** | `ht_neuron` + `ht_synapse` — intrinsic currents, sleep/wake | High | **Medium** — thalamocortical modeling |
| **GIF conductance** | `gif_cond_exp`, `gif_cond_exp_multisynapse` | Low | **Medium** — conductance-based GIF variants |
| **GIF population** | `gif_pop_psc_exp` — population-level GIF | High | **Medium** — mesoscale modeling |
| **Point process** | `pp_psc_delta`, `pp_cond_exp_mc_urbanczik` | Medium | **Low** — point process neurons |

#### PRIORITY 4 — Rate / Binary / Mean-Field / Special Models

| Missing Model | Description | Complexity | Impact |
|---------------|-------------|------------|--------|
| **Rate neurons** | `lin_rate`, `tanh_rate`, `sigmoid_rate`, `gauss_rate`, `threshold_lin_rate` | Low | **Medium** — rate-based network support |
| **Siegert neuron** | `siegert_neuron` — mean-field analysis | Medium | **Medium** — analytical tools |
| **Binary neurons** | `mcculloch_pitts_neuron`, `ginzburg_neuron`, `erfc_neuron` | Low | **Low** — niche use |
| **Rate connections** | `rate_connection_delayed`, `rate_connection_instantaneous`, `diffusion_connection` | Low | **Low** — rate infrastructure |
| **Parrot neuron** | `parrot_neuron` — spike repeater | Low | **Low** — utility model |
| **Ignore-and-fire** | `ignore_and_fire` — fixed-rate spiking | Low | **Low** — benchmarking |

#### PRIORITY 5 — Advanced / Emerging Models

| Missing Model | Description | Complexity | Impact |
|---------------|-------------|------------|--------|
| **e-prop system** | `eprop_iaf`, `eprop_iaf_adapt`, `eprop_readout`, `eprop_synapse` — eligibility propagation | Very High | **High** — online learning (but covered by braintrace) |
| **Astrocyte** | `astrocyte_lr_1994`, `aeif_cond_alpha_astro`, `sic_connection` — neuron-astrocyte interaction | High | **Medium** — emerging neuroscience |
| **Multi-compartment** | `cm_default`, `iaf_cond_alpha_mc` — dendritic computation | Very High | **Medium** — but may be better served by BrainCell |
| **Quantal STP** | `quantal_stp_synapse` — probabilistic vesicle-based STP | Medium | **Low** — biologically detailed STP |
| **Bernoulli synapse** | `bernoulli_synapse` — stochastic transmission | Low | **Low** — stochastic networks |
| **Urbanczik synapse** | `urbanczik_synapse` — two-compartment supervised learning | High | **Low** — specialized |

#### PRIORITY 6 — Device / Generator Models

brainpy.state currently has: `SpikeTime`, `PoissonSpike`, `PoissonEncoder`, `PoissonInput`.

| Missing Device | NEST Model | Priority |
|----------------|-----------|----------|
| DC current source | `dc_generator` | **High** |
| AC current source | `ac_generator` | Medium |
| Noise generator | `noise_generator` | **High** |
| Step current | `step_current_generator` | Medium |
| Sinusoidal Poisson | `sinusoidal_poisson_generator` | Medium |
| Inhomogeneous Poisson | `inhomogeneous_poisson_generator` | Medium |
| Gamma process | `gamma_sup_generator`, `sinusoidal_gamma_generator` | Low |
| Pulse packet | `pulsepacket_generator` | Low |

### Summary Statistics

| Category | NEST Count | brainpy.state Count | Gap |
|----------|-----------|--------------------|----|
| Spiking neuron models | ~50 | 18 | ~32 |
| Rate/binary neuron models | ~12 | 0 | ~12 |
| Synapse kinetics models | built into neurons | 6 | — |
| Short-term plasticity | 4 | 2 | 2 |
| **Long-term plasticity (STDP)** | **~15** | **0** | **~15** |
| Gap junctions | 1 (+ diffusion) | 2 | ✓ |
| Input/stimulation devices | ~17 | 4 | ~13 |
| Recording devices | ~8 | 0 | ~8 |
| Astrocyte/glia | 3 | 0 | 3 |
| e-prop system | ~8 | 0 (see braintrace) | ~8 |

### Design Difference Notes

The comparison is not one-to-one because the frameworks differ architecturally:

1. **PSC shape vs. synapse model separation**: NEST bakes PSC shapes (delta, alpha, exp) into
   neuron model names (e.g., `iaf_psc_alpha` vs. `iaf_psc_exp`). brainpy.state separates this:
   a single `LIF` neuron works with any synapse model (`Expon`, `Alpha`, `DualExpon`). This
   means brainpy.state needs fewer neuron classes to cover the same functionality.

2. **Conductance vs. current**: NEST has separate `*_cond_*` and `*_psc_*` models. brainpy.state
   uses `COBA` / `CUBA` / `MgBlock` synaptic output modules to switch between conductance-based
   and current-based modes with the same neuron. This is more modular.

3. **Multisynapse**: NEST `*_multisynapse` variants support multiple receptor ports. brainpy.state
   achieves this through multiple projection objects. No separate model is needed, but explicit
   multi-receptor neuron support may improve usability.

4. **Precise spike timing**: NEST's `*_ps` variants use exact spike time computation. brainpy.state
   uses fixed-step integration with surrogate gradients (different design goal: differentiability).

5. **Plasticity location**: NEST implements plasticity in synapse objects. brainpy.state should
   add plasticity as composable modules that wrap or extend projections.

---

## Development Roadmap

### TODO list

- [x] iaf_psc_delta
- [x] iaf_cond_exp
- [x] iaf_psc_delta_ps
- [ ] iaf_psc_alpha





- [ ] dc_generator



### Implementation Guidelines

When implementing new models, follow these conventions:

1. **Parameter naming**: Use NEST-compatible parameter names where possible (e.g., `V_th`, `V_reset`,
   `t_ref`, `tau_m`, `tau_syn_ex`, `tau_syn_in`, `g_L`, `C_m`, `E_L`, `E_ex`, `E_in`) to ease migration.

2. **Physical units**: All parameters must use `brainunit` quantities. Verify default values match
   NEST defaults (converted to SI units).

3. **Test validation**: For each new model, create a NEST comparison test:
   - Run identical network in NEST and brainpy.state
   - Compare membrane potential traces within numerical tolerance
   - Compare spike times within dt tolerance
   - Template: `_*_test.py` co-located with implementation
   - If nest is not available, for example in windows, skip. Use `pytest.skip()` with a clear message.

4. **Plasticity design pattern**: STDP and other plasticity rules should be implemented as
   composable modules that can wrap any projection type:
   ```python
   # Target API:
   proj = STDPProjection(
       pre=exc_neurons, 
       post=all_neurons,
       comm=brainevent.nn.FixedProb(...),
       syn=Expon.desc(...),
       plasticity=STDP(tau_plus=20*u.ms, tau_minus=20*u.ms, A_plus=0.01, A_minus=0.012),
   )
   ```

5. **Benchmarking**: Each model should include a performance benchmark against NEST for networks
   of 1K, 10K, and 100K neurons (where applicable).

6. **Documentation**: Each model class needs a docstring with: mathematical equations, parameter
   table with units and defaults, references to original papers, and a minimal usage example.

---

## Lessons Learned (Claude Coding Mistakes to Avoid)

### 1. Keep `brainunit` Quantities as Quantities — Never Convert to Plain Float

**Mistake**: Converting each `Quantity` entry to `float` before storing, e.g.:
```python
# WRONG
for t in amplitude_times:
    if isinstance(t, u.Quantity):
        amp_times_ms.append(float(u.math.asarray(t / u.ms)))
    else:
        amp_times_ms.append(float(t))
self._amp_times = jnp.asarray(amp_times_ms)
```

**Correct pattern**: Pass the sequence directly to `u.math.asarray`, which validates unit
consistency across all entries and returns a properly-typed Quantity array:
```python
# CORRECT
self.amplitude_times = u.math.asarray(amplitude_times)
```

---

### 2. Prefer `u.math` over `jnp`

**Mistake**: Importing and using `jax.numpy as jnp` directly when `brainunit.math`
provides unit-aware equivalents:
```python
# WRONG
import jax.numpy as jnp
idx = jnp.searchsorted(times, t, side='right') - 1
safe_idx = jnp.clip(idx, 0, n - 1)
val = jnp.where(idx >= 0, arr[safe_idx], 0.0)
```

**Correct pattern**: Use `u.math` throughout so that Quantity units are preserved:
```python
# CORRECT
idx = u.math.searchsorted(times, t, side='right') - 1
safe_idx = u.math.clip(idx, 0, n - 1)
val = u.math.where(idx >= 0, arr[safe_idx], zeros)
```
Only fall back to `jnp` if `u.math` genuinely does not provide the needed function.

---

### 5. JAX JIT Compatibility — No Python Control-Flow over Traced Values

**Mistake**: Using Python `for`/`if` loops that branch on a traced JAX value `t`:
```python
# WRONG — Python if over traced t breaks jit
for i in range(len(self.amplitude_times)):
    if t >= self.amplitude_times[i]:
        rate = self._amp_values[i]
    else:
        break
```

**Correct pattern**: Use `searchsorted` + `clip` + `where` to perform the same
piecewise-constant lookup in a JIT-compatible way:
```python
# CORRECT
t_dimless = u.math.asarray(t / u.ms)
times_dimless = u.math.asarray(self.amplitude_times / u.ms)
idx = u.math.searchsorted(times_dimless, t_dimless, side='right') - 1
safe_idx = u.math.clip(idx, 0, self.amplitude_values.shape[0] - 1)
amplitude = u.math.where(idx >= 0, self.amplitude_values[safe_idx], zeros)
```

---

### Cross-Ecosystem Coordination

Some functionality gaps are better addressed in sibling packages:

| Gap | Recommended Package | Notes |
|-----|--------------------|----|
| e-prop online learning | **braintrace** | Already implements eligibility trace learning |
| Multi-compartment neurons | **braincell** | Morphological neuron simulation |
| Event-driven sparse operators | **brainevent** | GPU-accelerated spike event processing |
| Whole-brain mean-field | **brainmass** | Neural mass models |
| Recording/analysis tools | **braintools** | Checkpointing, analysis utilities |
