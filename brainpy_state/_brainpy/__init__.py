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
BrainPy-style models for brainpy.state.

This subpackage provides high-level, composable neuron, synapse, projection,
and input models that are previously designed in the BrainPy simulator.
"""

# ---------------------------------------------------------------------------
# Neuron models
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Synapse models
# ---------------------------------------------------------------------------
from .exponential import Expon, DualExpon
# Hodgkin-Huxley family
from .hh import HH, MorrisLecar, WangBuzsakiHH
# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------
from .inputs import SpikeTime, PoissonSpike, PoissonEncoder, PoissonInput, poisson_input
# Izhikevich family
from .izhikevich import Izhikevich, IzhikevichRef
# LIF family
from .lif import (
    IF, LIF, LIFRef, ALIF,
    ExpIF, ExpIFRef,
    AdExIF, AdExIFRef,
    QuaIF, AdQuaIF, AdQuaIFRef,
    Gif, GifRef,
)
# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------
from .projection import (
    Projection, AlignPostProj, DeltaProj, CurrentProj,
    align_pre_projection, align_post_projection,
)
# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------
from .readout import LeakyRateReadout, LeakySpikeReadout
# ---------------------------------------------------------------------------
# Short-term plasticity
# ---------------------------------------------------------------------------
from .stp import STP, STD
from .synapse import Alpha, AMPA, GABAa, BioNMDA
from .synaptic_projection import SymmetryGapJunction, AsymmetryGapJunction
# ---------------------------------------------------------------------------
# Synaptic outputs
# ---------------------------------------------------------------------------
from .synouts import SynOut, COBA, CUBA, MgBlock

__all__ = [
    # Neuron models - LIF family
    'IF', 'LIF', 'LIFRef', 'ALIF',
    'ExpIF', 'ExpIFRef',
    'AdExIF', 'AdExIFRef',
    'QuaIF', 'AdQuaIF', 'AdQuaIFRef',
    'Gif', 'GifRef',

    # Neuron models - Hodgkin-Huxley family
    'HH', 'MorrisLecar', 'WangBuzsakiHH',

    # Neuron models - Izhikevich family
    'Izhikevich', 'IzhikevichRef',

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
    'LeakyRateReadout', 'LeakySpikeReadout',

    # Input generators
    'SpikeTime', 'PoissonSpike', 'PoissonEncoder', 'PoissonInput', 'poisson_input',
]
