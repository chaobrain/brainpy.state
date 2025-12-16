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
    except ImportError:
        from importlib_metadata import version, PackageNotFoundError

    try:
        brainpy_version = version("brainpy")
        # Parse version string (handle versions like "2.7.3.post1")
        version_parts = brainpy_version.split(".")[:3]
        major, minor = int(version_parts[0]), int(version_parts[1])
        patch = int(version_parts[2].split("+")[0].split("post")[0].split("a")[0].split("b")[0].split("rc")[0])

        if (major, minor, patch) < (2, 7, 4):
            raise RuntimeError(
                f"Incompatible brainpy version detected: {brainpy_version}. \n"
                f"brainpy.state requires brainpy >= 2.7.4 or no brainpy installed. "
                f"Please upgrade brainpy with 'pip install brainpy>=2.7.4' or "
                f"uninstall it with 'pip uninstall brainpy'."
            )
    except PackageNotFoundError:
        # brainpy is not installed, which is fine
        pass


_check_brainpy_compatibility()
del _check_brainpy_compatibility

__version__ = "0.0.1"
__version_info__ = tuple(map(int, __version__.split(".")))

from ._base import *
from ._base import __all__ as base_all
from ._exponential import *
from ._exponential import __all__ as exp_all
from ._hh import *
from ._hh import __all__ as hh_all
from ._inputs import *
from ._inputs import __all__ as inputs_all
from ._izhikevich import *
from ._izhikevich import __all__ as izh_all
from ._lif import *
from ._lif import __all__ as neuron_all
from ._projection import *
from ._projection import __all__ as proj_all
from ._readout import *
from ._readout import __all__ as readout_all
from ._stp import *
from ._stp import __all__ as stp_all
from ._synapse import *
from ._synapse import __all__ as synapse_all
from ._synaptic_projection import *
from ._synaptic_projection import __all__ as synproj_all
from ._synouts import *
from ._synouts import __all__ as synout_all

__main__ = inputs_all + neuron_all + izh_all + hh_all + readout_all + stp_all + synapse_all
__main__ = __main__ + synout_all + base_all + exp_all + proj_all + synproj_all
del inputs_all, neuron_all, izh_all, hh_all, readout_all, stp_all, synapse_all, synout_all, base_all
del exp_all, proj_all, synproj_all
