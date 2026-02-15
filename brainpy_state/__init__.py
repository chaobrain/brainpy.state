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


# Compatibility check: ensure no conflicting old brainpy version is installed
def _check_brainpy_compatibility():
    try:
        from importlib.metadata import version, PackageNotFoundError

        brainpy_version = version("brainpy")
        # Parse version string (handle versions like "2.7.3.post1")
        version_parts = brainpy_version.split(".")[:3]
        major, minor = int(version_parts[0]), int(version_parts[1])
        patch = int(version_parts[2].split("+")[0].split("post")[0].split("a")[0].split("b")[0].split("rc")[0])

        if (major, minor, patch) < (2, 7, 5):
            raise RuntimeError(
                f"Incompatible brainpy version detected: {brainpy_version}. \n"
                f"brainpy.state requires brainpy >= 2.7.5 or no brainpy installed. "
                f"Please upgrade brainpy with 'pip install brainpy>=2.7.5' or "
                f"uninstall it with 'pip uninstall brainpy'."
            )
    except:
        # brainpy is not installed, which is fine
        pass


_check_brainpy_compatibility()
del _check_brainpy_compatibility

__version__ = "0.0.4"
__version_info__ = tuple(map(int, __version__.split(".")))

from ._base import Dynamics, Neuron, Synapse
from ._exponential import Expon, DualExpon
from ._hh import HH, MorrisLecar, WangBuzsakiHH
from ._inputs import SpikeTime, PoissonSpike, PoissonEncoder, PoissonInput, poisson_input
from ._izhikevich import Izhikevich, IzhikevichRef
from ._lif import (
    IF, LIF, ExpIF, ExpIFRef, AdExIF, AdExIFRef, LIFRef, ALIF,
    QuaIF, AdQuaIF, AdQuaIFRef, Gif, GifRef
)
from ._nest.iaf_psc_delta import iaf_psc_delta
from ._projection import (Projection, AlignPostProj, DeltaProj, CurrentProj,
                          align_pre_projection, align_post_projection)
from ._readout import LeakyRateReadout, LeakySpikeReadout
from ._stp import STP, STD
from ._synapse import Alpha, AMPA, GABAa, BioNMDA
from ._synaptic_projection import SymmetryGapJunction, AsymmetryGapJunction
from ._synouts import SynOut, COBA, CUBA, MgBlock

__all__ = [
    # _base
    'Dynamics', 'Neuron', 'Synapse',
    # _exponential
    'Expon', 'DualExpon',
    # _hh
    'HH', 'MorrisLecar', 'WangBuzsakiHH',
    # _inputs
    'SpikeTime', 'PoissonSpike', 'PoissonEncoder', 'PoissonInput', 'poisson_input',

    # _izhikevich
    'Izhikevich', 'IzhikevichRef',
    # _lif
    'IF', 'LIF', 'ExpIF', 'ExpIFRef', 'AdExIF', 'AdExIFRef', 'LIFRef', 'ALIF',
    'QuaIF', 'AdQuaIF', 'AdQuaIFRef', 'Gif', 'GifRef',
    # _projection
    'Projection', 'AlignPostProj', 'DeltaProj', 'CurrentProj',
    'align_pre_projection', 'align_post_projection',
    # _readout
    'LeakyRateReadout', 'LeakySpikeReadout',
    # _stp
    'STP', 'STD',
    # _synapse
    'Alpha', 'AMPA', 'GABAa', 'BioNMDA',
    # _synaptic_projection
    'SymmetryGapJunction', 'AsymmetryGapJunction',
    # _synouts
    'SynOut', 'COBA', 'CUBA', 'MgBlock',

    # iaf_psc_delta
    'iaf_psc_delta',
]
