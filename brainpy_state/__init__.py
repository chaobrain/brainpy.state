# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# Compatibility check: abort import if a stale brainpy would shadow brainpy.state.
from ._compat import check_brainpy_compatibility

check_brainpy_compatibility()
del check_brainpy_compatibility

# Namespace guard: this package is private; users must reach it through brainpy.state.
from ._namespace import enforce_namespace_access

enforce_namespace_access()
del enforce_namespace_access

from ._version import __version_info__, __version__

# =============================================================================
# Base Models
# =============================================================================

from ._base import Dynamics, Neuron, Synapse

# =============================================================================
# BrainPy-style Models
# =============================================================================

from ._brainpy import (
    # Neuron models - LIF family
    IF, LIF, LIFRef, ALIF,
    ExpIF, ExpIFRef,
    AdExIF, AdExIFRef,
    QuaIF, AdQuaIF, AdQuaIFRef,
    Gif, GifRef,
    CubaLIF, CobaLIF,

    # Neuron models - Hodgkin-Huxley family
    HH, MorrisLecar, WangBuzsakiHH,

    # Neuron models - Izhikevich family
    Izhikevich, IzhikevichRef,

    # Neuron models - reduced (FitzHugh-Nagumo, Hindmarsh-Rose)
    FitzHughNagumo, HindmarshRose,

    # Synapse models
    Expon, DualExpon,
    Alpha, AMPA, GABAa, BioNMDA,

    # Synaptic outputs
    SynOut, COBA, CUBA, MgBlock,

    # Short-term plasticity
    STP, STD,

    # Projections
    Projection, AlignPostProj, DeltaProj, CurrentProj,
    align_pre_projection, align_post_projection,
    SymmetryGapJunction, AsymmetryGapJunction,

    # Readouts
    LeakyRateReadout,

    # Input generators
    SpikeTime, PoissonSpike, PoissonEncoder, PoissonInput, poisson_input,
    SectionInput, ConstantInput, StepInput, RampInput,
    SinusoidalInput, OUProcessInput, WienerProcessInput,
)

# =============================================================================
# Network API
# =============================================================================

from ._nest_network import (
    Network,
    Builder,
    Recorder,
    OneToOneProj,
    AllToAllProj,
    PairwiseBernoulliProj,
    SymmetricPairwiseBernoulliProj,
    FixedIndegreeProj,
    FixedOutdegreeProj,
    FixedTotalNumberProj,
    PairwisePoissonProj,
    # NEST-flavored explicit Simulator API
    Simulator,
    SimulationResult,
    SynapseCollection,
    NodeView,
    all_to_all,
    one_to_one,
    fixed_indegree,
    pairwise_bernoulli,
    fixed_total_number,
    third_factor_bernoulli_with_pool,
    explicit_edges,
)

from . import _nest_network as network  # noqa: F401  -- exposed as brainpy.state.network
from . import _dist as dist  # noqa: F401  -- exposed as brainpy.state.dist
from . import _nest_spatial as spatial  # noqa: F401  -- exposed as brainpy.state.spatial

# =============================================================================
# NEST-Compatible Models
# =============================================================================

from ._nest_base import (
    NESTDevice,
    NESTNeuron,
    NESTSynapse,
    NESTPlasticity,
)
from ._nest_neuron import (
    iaf_psc_alpha,
    iaf_psc_alpha_multisynapse,
    iaf_psc_alpha_ps,
    iaf_psc_delta,
    iaf_psc_delta_ps,
    iaf_psc_exp,
    iaf_psc_exp_htum,
    iaf_psc_exp_multisynapse,
    iaf_psc_exp_ps,
    iaf_psc_exp_ps_lossless,
    iaf_cond_alpha,
    iaf_cond_alpha_mc,
    iaf_cond_beta,
    iaf_cond_exp,
    iaf_cond_exp_sfa_rr,
    iaf_bw_2001,
    iaf_bw_2001_exact,
    iaf_chs_2007,
    iaf_chxk_2008,
    iaf_tum_2000,
    aeif_cond_alpha,
    aeif_cond_alpha_astro,
    aeif_cond_alpha_multisynapse,
    aeif_cond_beta_multisynapse,
    aeif_cond_exp,
    aeif_psc_alpha,
    aeif_psc_delta,
    aeif_psc_delta_clopath,
    aeif_psc_exp,
    gif_cond_exp,
    gif_cond_exp_multisynapse,
    gif_pop_psc_exp,
    gif_psc_exp,
    gif_psc_exp_multisynapse,
    amat2_psc_exp,
    mat2_psc_exp,
    glif_cond,
    glif_psc,
    glif_psc_double_alpha,
    hh_cond_beta_gap_traub,
    hh_cond_exp_traub,
    hh_psc_alpha,
    hh_psc_alpha_clopath,
    hh_psc_alpha_gap,
    ht_neuron,
    izhikevich,
    pp_cond_exp_mc_urbanczik,
    pp_psc_delta,
    erfc_neuron,
    ginzburg_neuron,
    mcculloch_pitts_neuron,
    gauss_rate_ipn,
    lin_rate_ipn,
    lin_rate_opn,
    rate_neuron_ipn,
    rate_neuron_opn,
    rate_transformer_node,
    siegert_neuron,
    sigmoid_rate_ipn,
    sigmoid_rate_gg_1998_ipn,
    tanh_rate_ipn,
    tanh_rate_opn,
    threshold_lin_rate_ipn,
    threshold_lin_rate_opn,
    astrocyte_lr_1994,
    cm_default,
    ignore_and_fire,
    parrot_neuron,
)
from ._nest_synapse import (
    bernoulli_synapse,
    cont_delay_synapse,
    static_synapse,
    static_synapse_hom_w,
    quantal_stp_synapse,
    tsodyks2_synapse,
    tsodyks_synapse,
    tsodyks_synapse_hom,
    diffusion_connection,
    gap_junction,
    rate_connection_delayed,
    rate_connection_instantaneous,
    sic_connection,
)
from ._nest_plasticity import (
    stdp_dopamine_synapse,
    stdp_facetshw_synapse_hom,
    stdp_nn_pre_centered_synapse,
    stdp_nn_restr_synapse,
    stdp_nn_symm_synapse,
    stdp_pl_synapse_hom,
    stdp_synapse,
    stdp_synapse_hom,
    stdp_triplet_synapse,
    clopath_synapse,
    ht_synapse,
    jonke_synapse,
    urbanczik_synapse,
    vogels_sprekeler_synapse,
)
from ._nest_device import (
    ac_generator,
    dc_generator,
    noise_generator,
    step_current_generator,
    host_spike_drive,
    host_current_drive,
    step_rate_generator,
    spike_generator,
    spike_train_injector,
    spike_dilutor,
    inhomogeneous_poisson_generator,
    poisson_generator,
    poisson_generator_ps,
    sinusoidal_poisson_generator,
    gamma_sup_generator,
    mip_generator,
    ppd_sup_generator,
    pulsepacket_generator,
    sinusoidal_gamma_generator,
    correlation_detector,
    correlomatrix_detector,
    correlospinmatrix_detector,
    multimeter,
    voltmeter,
    spike_recorder,
    spin_detector,
    volume_transmitter,
    weight_recorder,
)

__all__ = [
    # =========================================================================
    # Package metadata (re-exported through the brainpy.state namespace)
    # =========================================================================
    '__version__', '__version_info__',

    # =========================================================================
    # Base Models
    # =========================================================================
    'Dynamics', 'Neuron', 'Synapse',

    # =========================================================================
    # BrainPy-style Models
    # =========================================================================

    # Neuron models - LIF family
    'IF', 'LIF', 'LIFRef', 'ALIF',
    'ExpIF', 'ExpIFRef',
    'AdExIF', 'AdExIFRef',
    'QuaIF', 'AdQuaIF', 'AdQuaIFRef',
    'Gif', 'GifRef',
    'CubaLIF', 'CobaLIF',

    # Neuron models - Hodgkin-Huxley family
    'HH', 'MorrisLecar', 'WangBuzsakiHH',

    # Neuron models - Izhikevich family
    'Izhikevich', 'IzhikevichRef',

    # Neuron models - reduced (FitzHugh-Nagumo, Hindmarsh-Rose)
    'FitzHughNagumo', 'HindmarshRose',

    # Synapse models
    'Expon', 'DualExpon',
    'Alpha', 'AMPA', 'GABAa', 'BioNMDA',

    # Synaptic outputs
    'SynOut', 'COBA', 'CUBA', 'MgBlock',

    # Short-term plasticity
    'STP', 'STD',

    # Projections
    'Projection', 'AlignPostProj', 'DeltaProj', 'CurrentProj',
    'align_pre_projection', 'align_post_projection',
    'SymmetryGapJunction', 'AsymmetryGapJunction',

    # Readouts
    'LeakyRateReadout',

    # Input generators
    'SpikeTime', 'PoissonSpike', 'PoissonEncoder', 'PoissonInput', 'poisson_input',
    'SectionInput', 'ConstantInput', 'StepInput', 'RampInput',
    'SinusoidalInput', 'OUProcessInput', 'WienerProcessInput',

    # =========================================================================
    # NEST-Compatible Models
    # =========================================================================

    # Base classes
    'NESTDevice', 'NESTNeuron', 'NESTSynapse', 'NESTPlasticity',

    # Stimulation devices - Current generators
    'ac_generator',
    'dc_generator',
    'noise_generator',
    'step_current_generator',
    # Host-clamped input devices (closed-loop / Simulator.cont rollouts)
    'host_spike_drive',
    'host_current_drive',
    'step_rate_generator',

    # Stimulation devices - Spike generators
    'spike_generator',
    'spike_train_injector',
    'spike_dilutor',

    # Stimulation devices - Poisson generators
    'inhomogeneous_poisson_generator',
    'poisson_generator',
    'poisson_generator_ps',
    'sinusoidal_poisson_generator',

    # Stimulation devices - Other generators
    'gamma_sup_generator',
    'mip_generator',
    'ppd_sup_generator',
    'pulsepacket_generator',
    'sinusoidal_gamma_generator',

    # Recording devices
    'correlation_detector',
    'correlomatrix_detector',
    'correlospinmatrix_detector',
    'multimeter',
    'voltmeter',
    'spike_recorder',
    'spin_detector',
    'volume_transmitter',
    'weight_recorder',

    # IAF neurons - current-based (psc)
    'iaf_psc_alpha',
    'iaf_psc_alpha_multisynapse',
    'iaf_psc_alpha_ps',
    'iaf_psc_delta',
    'iaf_psc_delta_ps',
    'iaf_psc_exp',
    'iaf_psc_exp_htum',
    'iaf_psc_exp_multisynapse',
    'iaf_psc_exp_ps',
    'iaf_psc_exp_ps_lossless',

    # IAF neurons - conductance-based (cond)
    'iaf_cond_alpha',
    'iaf_cond_alpha_mc',
    'iaf_cond_beta',
    'iaf_cond_exp',
    'iaf_cond_exp_sfa_rr',

    # IAF neurons - specialized variants
    'iaf_bw_2001',
    'iaf_bw_2001_exact',
    'iaf_chs_2007',
    'iaf_chxk_2008',
    'iaf_tum_2000',

    # Adaptive Exponential IF (aeif) neurons
    'aeif_cond_alpha',
    'aeif_cond_alpha_astro',
    'aeif_cond_alpha_multisynapse',
    'aeif_cond_beta_multisynapse',
    'aeif_cond_exp',
    'aeif_psc_alpha',
    'aeif_psc_delta',
    'aeif_psc_delta_clopath',
    'aeif_psc_exp',

    # Generalized IF (gif) neurons
    'gif_cond_exp',
    'gif_cond_exp_multisynapse',
    'gif_pop_psc_exp',
    'gif_psc_exp',
    'gif_psc_exp_multisynapse',

    # Multi-timescale Adaptive Threshold (mat) neurons
    'amat2_psc_exp',
    'mat2_psc_exp',

    # Generalized LIF (glif) neurons
    'glif_cond',
    'glif_psc',
    'glif_psc_double_alpha',

    # Hodgkin-Huxley family
    'hh_cond_beta_gap_traub',
    'hh_cond_exp_traub',
    'hh_psc_alpha',
    'hh_psc_alpha_clopath',
    'hh_psc_alpha_gap',
    'ht_neuron',

    # Izhikevich neuron
    'izhikevich',

    # Point process neurons
    'pp_cond_exp_mc_urbanczik',
    'pp_psc_delta',

    # Binary neurons
    'erfc_neuron',
    'ginzburg_neuron',
    'mcculloch_pitts_neuron',

    # Rate neurons
    'gauss_rate_ipn',
    'lin_rate_ipn',
    'lin_rate_opn',
    'rate_neuron_ipn',
    'rate_neuron_opn',
    'rate_transformer_node',
    'siegert_neuron',
    'sigmoid_rate_ipn',
    'sigmoid_rate_gg_1998_ipn',
    'tanh_rate_ipn',
    'tanh_rate_opn',
    'threshold_lin_rate_ipn',
    'threshold_lin_rate_opn',

    # Astrocyte models
    'astrocyte_lr_1994',

    # Other spiking neurons
    'ignore_and_fire',
    'parrot_neuron',

    # Static synapses
    'bernoulli_synapse',
    'cont_delay_synapse',
    'static_synapse',
    'static_synapse_hom_w',

    # Short-term plasticity synapses
    'quantal_stp_synapse',
    'tsodyks2_synapse',
    'tsodyks_synapse',
    'tsodyks_synapse_hom',

    # STDP synapses
    'stdp_dopamine_synapse',
    'stdp_facetshw_synapse_hom',
    'stdp_nn_pre_centered_synapse',
    'stdp_nn_restr_synapse',
    'stdp_nn_symm_synapse',
    'stdp_pl_synapse_hom',
    'stdp_synapse',
    'stdp_synapse_hom',
    'stdp_triplet_synapse',

    # Voltage-based / specialized synapses
    'clopath_synapse',
    'ht_synapse',
    'jonke_synapse',
    'urbanczik_synapse',
    'vogels_sprekeler_synapse',

    # Gap junctions and special connections
    'diffusion_connection',
    'gap_junction',
    'rate_connection_delayed',
    'rate_connection_instantaneous',
    'sic_connection',

    # multi-compartment models
    'cm_default',

    # Network API
    'Network',
    'Builder',
    'Recorder',
    'OneToOneProj',
    'AllToAllProj',
    'PairwiseBernoulliProj',
    'SymmetricPairwiseBernoulliProj',
    'FixedIndegreeProj',
    'FixedOutdegreeProj',
    'FixedTotalNumberProj',
    'PairwisePoissonProj',
    # NEST-flavored explicit Simulator API
    'Simulator',
    'SimulationResult',
    'SynapseCollection',
    'NodeView',
    'all_to_all',
    'one_to_one',
    'fixed_indegree',
    'pairwise_bernoulli',
    'fixed_total_number',
    'third_factor_bernoulli_with_pool',
    'explicit_edges',
    'network',
    'dist',
    'spatial',
]
