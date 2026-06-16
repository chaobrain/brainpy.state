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
Abstract base classes for NEST-compatible models.

Four marker base classes that categorise every model in
``brainpy_state._nest`` by its NEST model type:

- :class:`NESTDevice`      -- stimulation and recording devices.
- :class:`NESTNeuron`      -- point-neuron and population-neuron models.
- :class:`NESTSynapse`     -- static synapse, short-term-plasticity, and
  connection models.
- :class:`NESTPlasticity`  -- long-term / activity-history plasticity synapse
  models (STDP and voltage-based learning rules).

Each class is intentionally kept empty; all behaviour is inherited from
the BrainPy / BrainState parent classes.
"""

from brainpy_state._base import Dynamics, Neuron

__all__ = [
    'NESTDevice',
    'NESTNeuron',
    'NESTSynapse',
    'NESTPlasticity',
]


class NESTDevice(Dynamics):
    """Abstract base class for all NEST-compatible device models.

    Covers stimulation devices (current generators, spike generators,
    Poisson generators, …) and recording devices (multimeter, spike
    recorder, correlation detectors, …).
    """
    __module__ = 'brainpy.state'


class NESTNeuron(Neuron):
    """Abstract base class for all NEST-compatible neuron models.

    Covers spiking neuron families (IAF, AdEx, GIF, Hodgkin-Huxley, …),
    rate-coded neurons, binary neurons, multi-compartment models, and
    astrocyte models.
    """
    __module__ = 'brainpy.state'


class NESTSynapse(Dynamics):
    """Abstract base class for all NEST-compatible synapse models.

    Covers static synapses (``static_synapse``, ``static_synapse_hom_w``,
    ``bernoulli_synapse``, ``cont_delay_synapse``), short-term-plasticity
    synapses (``tsodyks_synapse``, ``tsodyks_synapse_hom``, ``tsodyks2_synapse``,
    ``quantal_stp_synapse``), gap junctions (``gap_junction``,
    ``diffusion_connection``), rate connections
    (``rate_connection_instantaneous``, ``rate_connection_delayed``),
    and other connection models (``sic_connection``).

    Long-term / activity-history plasticity synapse models (STDP, voltage-based
    learning rules) inherit from the more specific :class:`NESTPlasticity`
    subclass.
    """
    __module__ = 'brainpy.state'


class NESTPlasticity(NESTSynapse):
    """Abstract base class for all NEST-compatible plasticity synapse models.

    Subclass of :class:`NESTSynapse` that marks models whose synaptic
    weights change as a long-term function of neural activity history.  Two
    broad families are covered (short-term plasticity is a plain
    :class:`NESTSynapse`):

    **Spike-Timing Dependent Plasticity (STDP)**
        Long-term weight changes driven by pre/post spike timing:
        ``stdp_synapse``, ``stdp_synapse_hom``, ``stdp_pl_synapse_hom``,
        ``stdp_facetshw_synapse_hom``, ``stdp_nn_pre_centered_synapse``,
        ``stdp_nn_restr_synapse``, ``stdp_nn_symm_synapse``,
        ``stdp_triplet_synapse``, ``stdp_dopamine_synapse``.

    **Voltage-Based / Specialised Learning Rules**
        Weight updates that depend on membrane voltage or supervised
        signals: ``clopath_synapse``, ``jonke_synapse``,
        ``urbanczik_synapse``, ``vogels_sprekeler_synapse``,
        ``ht_synapse``.
    """
    __module__ = 'brainpy.state'
