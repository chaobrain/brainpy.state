# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

r"""
NEST-compatible models for brainpy.state.

This subpackage provides implementations of neuron, synapse, and device models
that are compatible with the NEST simulator (https://nest-simulator.readthedocs.io/).
"""

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------
from ._base import NESTDevice, NESTNeuron, NESTSynapse, NESTPlasticity
# ---------------------------------------------------------------------------
# Stimulation devices - Current generators
# ---------------------------------------------------------------------------
from .ac_generator import ac_generator
# ---------------------------------------------------------------------------
# Adaptive Exponential IF (aeif) neurons
# ---------------------------------------------------------------------------
from .aeif_cond_alpha import aeif_cond_alpha
from .aeif_cond_alpha_astro import aeif_cond_alpha_astro
from .aeif_cond_alpha_multisynapse import aeif_cond_alpha_multisynapse
from .aeif_cond_beta_multisynapse import aeif_cond_beta_multisynapse
from .aeif_cond_exp import aeif_cond_exp
from .aeif_psc_alpha import aeif_psc_alpha
from .aeif_psc_delta import aeif_psc_delta
from .aeif_psc_delta_clopath import aeif_psc_delta_clopath
from .aeif_psc_exp import aeif_psc_exp
# ---------------------------------------------------------------------------
# Multi-timescale Adaptive Threshold (mat) neurons
# ---------------------------------------------------------------------------
from .amat2_psc_exp import amat2_psc_exp
# ---------------------------------------------------------------------------
# Astrocyte models
# ---------------------------------------------------------------------------
from .astrocyte_lr_1994 import astrocyte_lr_1994
# ---------------------------------------------------------------------------
# Static synapses
# ---------------------------------------------------------------------------
from .bernoulli_synapse import bernoulli_synapse
# ---------------------------------------------------------------------------
# Voltage-based / specialized synapses
# ---------------------------------------------------------------------------
from .clopath_synapse import clopath_synapse
# ---------------------------------------------------------------------------
# Multi-compartment models
# ---------------------------------------------------------------------------
from .cm_default import cm_default
from .cont_delay_synapse import cont_delay_synapse
# ---------------------------------------------------------------------------
# Recording devices
# ---------------------------------------------------------------------------
from .correlation_detector import correlation_detector
from .correlomatrix_detector import correlomatrix_detector
from .correlospinmatrix_detector import correlospinmatrix_detector
from .dc_generator import dc_generator
from .host_drive import host_spike_drive, host_current_drive
# ---------------------------------------------------------------------------
# Gap junctions and special connections
# ---------------------------------------------------------------------------
from .diffusion_connection import diffusion_connection
# ---------------------------------------------------------------------------
# Binary neurons
# ---------------------------------------------------------------------------
from .erfc_neuron import erfc_neuron
# ---------------------------------------------------------------------------
# Stimulation devices - Other generators
# ---------------------------------------------------------------------------
from .gamma_sup_generator import gamma_sup_generator
from .gap_junction import gap_junction
# ---------------------------------------------------------------------------
# Rate neurons
# ---------------------------------------------------------------------------
from .gauss_rate import gauss_rate_ipn
# ---------------------------------------------------------------------------
# Generalized IF (gif) neurons
# ---------------------------------------------------------------------------
from .gif_cond_exp import gif_cond_exp
from .gif_cond_exp_multisynapse import gif_cond_exp_multisynapse
from .gif_pop_psc_exp import gif_pop_psc_exp
from .gif_psc_exp import gif_psc_exp
from .gif_psc_exp_multisynapse import gif_psc_exp_multisynapse
from .ginzburg_neuron import ginzburg_neuron
# ---------------------------------------------------------------------------
# Generalized LIF (glif) neurons - Allen Institute
# ---------------------------------------------------------------------------
from .glif_cond import glif_cond
from .glif_psc import glif_psc
from .glif_psc_double_alpha import glif_psc_double_alpha
# ---------------------------------------------------------------------------
# Hodgkin-Huxley family
# ---------------------------------------------------------------------------
from .hh_cond_beta_gap_traub import hh_cond_beta_gap_traub
from .hh_cond_exp_traub import hh_cond_exp_traub
from .hh_psc_alpha import hh_psc_alpha
from .hh_psc_alpha_clopath import hh_psc_alpha_clopath
from .hh_psc_alpha_gap import hh_psc_alpha_gap
from .ht_neuron import ht_neuron
from .ht_synapse import ht_synapse
# ---------------------------------------------------------------------------
# IAF neurons - specialized variants
# ---------------------------------------------------------------------------
from .iaf_bw_2001 import iaf_bw_2001
from .iaf_bw_2001_exact import iaf_bw_2001_exact
from .iaf_chs_2007 import iaf_chs_2007
from .iaf_chxk_2008 import iaf_chxk_2008
# ---------------------------------------------------------------------------
# IAF neurons - conductance-based (cond)
# ---------------------------------------------------------------------------
from .iaf_cond_alpha import iaf_cond_alpha
from .iaf_cond_alpha_mc import iaf_cond_alpha_mc
from .iaf_cond_beta import iaf_cond_beta
from .iaf_cond_exp import iaf_cond_exp
from .iaf_cond_exp_sfa_rr import iaf_cond_exp_sfa_rr
# ---------------------------------------------------------------------------
# IAF neurons - current-based (psc)
# ---------------------------------------------------------------------------
from .iaf_psc_alpha import iaf_psc_alpha
from .iaf_psc_alpha_multisynapse import iaf_psc_alpha_multisynapse
from .iaf_psc_alpha_ps import iaf_psc_alpha_ps
from .iaf_psc_delta import iaf_psc_delta
from .iaf_psc_delta_ps import iaf_psc_delta_ps
from .iaf_psc_exp import iaf_psc_exp
from .iaf_psc_exp_htum import iaf_psc_exp_htum
from .iaf_psc_exp_multisynapse import iaf_psc_exp_multisynapse
from .iaf_psc_exp_ps import iaf_psc_exp_ps
from .iaf_psc_exp_ps_lossless import iaf_psc_exp_ps_lossless
from .iaf_tum_2000 import iaf_tum_2000
# ---------------------------------------------------------------------------
# Other spiking neurons
# ---------------------------------------------------------------------------
from .ignore_and_fire import ignore_and_fire
from .parrot_neuron import parrot_neuron
# ---------------------------------------------------------------------------
# Stimulation devices - Poisson generators
# ---------------------------------------------------------------------------
from .inhomogeneous_poisson_generator import inhomogeneous_poisson_generator
# ---------------------------------------------------------------------------
# Izhikevich neuron
# ---------------------------------------------------------------------------
from .izhikevich import izhikevich
from .jonke_synapse import jonke_synapse
from .lin_rate import lin_rate_ipn, lin_rate_opn
from .mat2_psc_exp import mat2_psc_exp
from .mcculloch_pitts_neuron import mcculloch_pitts_neuron
from .mip_generator import mip_generator
from .multimeter import multimeter
from .voltmeter import voltmeter
from .noise_generator import noise_generator
from .poisson_generator import poisson_generator
from .poisson_generator_ps import poisson_generator_ps
# ---------------------------------------------------------------------------
# Point process neurons
# ---------------------------------------------------------------------------
from .pp_cond_exp_mc_urbanczik import pp_cond_exp_mc_urbanczik
from .pp_psc_delta import pp_psc_delta
from .ppd_sup_generator import ppd_sup_generator
from .pulsepacket_generator import pulsepacket_generator
# ---------------------------------------------------------------------------
# Short-term plasticity synapses
# ---------------------------------------------------------------------------
from .quantal_stp_synapse import quantal_stp_synapse
from .rate_connection_delayed import rate_connection_delayed
from .rate_connection_instantaneous import rate_connection_instantaneous
from .rate_neuron_ipn import rate_neuron_ipn
from .rate_neuron_opn import rate_neuron_opn
from .rate_transformer_node import rate_transformer_node
from .sic_connection import sic_connection
from .siegert_neuron import siegert_neuron
from .sigmoid_rate import sigmoid_rate_ipn
from .sigmoid_rate_gg_1998 import sigmoid_rate_gg_1998_ipn
from .sinusoidal_gamma_generator import sinusoidal_gamma_generator
from .sinusoidal_poisson_generator import sinusoidal_poisson_generator
from .spike_dilutor import spike_dilutor
# ---------------------------------------------------------------------------
# Stimulation devices - Spike generators
# ---------------------------------------------------------------------------
from .spike_generator import spike_generator
from .spike_recorder import spike_recorder
from .spike_train_injector import spike_train_injector
from .spin_detector import spin_detector
from .static_synapse import static_synapse
from .static_synapse_hom_w import static_synapse_hom_w
# ---------------------------------------------------------------------------
# STDP synapses
# ---------------------------------------------------------------------------
from .stdp_dopamine_synapse import stdp_dopamine_synapse
from .stdp_facetshw_synapse_hom import stdp_facetshw_synapse_hom
from .stdp_nn_pre_centered_synapse import stdp_nn_pre_centered_synapse
from .stdp_nn_restr_synapse import stdp_nn_restr_synapse
from .stdp_nn_symm_synapse import stdp_nn_symm_synapse
from .stdp_pl_synapse_hom import stdp_pl_synapse_hom
from .stdp_synapse import stdp_synapse
from .stdp_synapse_hom import stdp_synapse_hom
from .stdp_triplet_synapse import stdp_triplet_synapse
from .step_current_generator import step_current_generator
from .step_rate_generator import step_rate_generator
from .tanh_rate import tanh_rate_ipn, tanh_rate_opn
from .threshold_lin_rate import threshold_lin_rate_ipn, threshold_lin_rate_opn
from .tsodyks2_synapse import tsodyks2_synapse
from .tsodyks_synapse import tsodyks_synapse
from .tsodyks_synapse_hom import tsodyks_synapse_hom
from .urbanczik_synapse import urbanczik_synapse
from .vogels_sprekeler_synapse import vogels_sprekeler_synapse
from .volume_transmitter import volume_transmitter
from .weight_recorder import weight_recorder

__all__ = [
    # Base classes
    'NESTDevice',
    'NESTNeuron',
    'NESTSynapse',
    'NESTPlasticity',

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

    # Multi-compartment models
    'cm_default',

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
]
