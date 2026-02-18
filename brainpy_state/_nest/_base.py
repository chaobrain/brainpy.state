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

Three marker base classes that categorise every model in
``brainpy_state._nest`` by its NEST model type:

- :class:`NESTDevice`  -- stimulation and recording devices.
- :class:`NESTNeuron`  -- point-neuron and population-neuron models.
- :class:`NESTSynapse` -- synapse and connection models.

Each class is intentionally kept empty; all behaviour is inherited from
the BrainPy / BrainState parent classes.
"""

from brainpy_state._base import Dynamics, Neuron

__all__ = [
    'NESTDevice',
    'NESTNeuron',
    'NESTSynapse',
]


class NESTDevice(Dynamics):
    """Abstract base class for all NEST-compatible device models.

    Covers stimulation devices (current generators, spike generators,
    Poisson generators, …) and recording devices (multimeter, spike
    recorder, correlation detectors, …).
    """


class NESTNeuron(Neuron):
    """Abstract base class for all NEST-compatible neuron models.

    Covers spiking neuron families (IAF, AdEx, GIF, Hodgkin-Huxley, …),
    rate-coded neurons, binary neurons, multi-compartment models, and
    astrocyte models.
    """


class NESTSynapse(Dynamics):
    """Abstract base class for all NEST-compatible synapse models.

    Covers static synapses, short-term plasticity synapses, STDP synapses,
    gap junctions, rate connections, and other connection models.
    """
