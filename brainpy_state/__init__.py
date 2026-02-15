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
from ._nest.dc_generator import dc_generator
from ._nest.multimeter import multimeter
from ._nest.correlation_detector import correlation_detector
from ._nest.correlomatrix_detector import correlomatrix_detector
from ._nest.correlospinmatrix_detector import correlospinmatrix_detector
from ._nest.spike_recorder import spike_recorder
from ._nest.spin_detector import spin_detector
from ._nest.weight_recorder import weight_recorder
from ._nest.gap_junction import gap_junction
from ._nest.diffusion_connection import diffusion_connection
from ._nest.rate_connection_instantaneous import rate_connection_instantaneous
from ._nest.rate_connection_delayed import rate_connection_delayed
from ._nest.sic_connection import sic_connection
from ._nest.ht_synapse import ht_synapse
from ._nest.clopath_synapse import clopath_synapse
from ._nest.jonke_synapse import jonke_synapse
from ._nest.urbanczik_synapse import urbanczik_synapse
from ._nest.vogels_sprekeler_synapse import vogels_sprekeler_synapse
from ._nest.volume_transmitter import volume_transmitter
from ._nest.aeif_cond_alpha import aeif_cond_alpha
from ._nest.aeif_cond_exp import aeif_cond_exp
from ._nest.aeif_psc_alpha import aeif_psc_alpha
from ._nest.aeif_psc_exp import aeif_psc_exp
from ._nest.aeif_psc_delta import aeif_psc_delta
from ._nest.aeif_psc_delta_clopath import aeif_psc_delta_clopath
from ._nest.aeif_cond_alpha_multisynapse import aeif_cond_alpha_multisynapse
from ._nest.aeif_cond_beta_multisynapse import aeif_cond_beta_multisynapse
from ._nest.aeif_cond_alpha_astro import aeif_cond_alpha_astro
from ._nest.iaf_psc_delta import iaf_psc_delta
from ._nest.iaf_psc_delta_ps import iaf_psc_delta_ps
from ._nest.iaf_cond_alpha import iaf_cond_alpha
from ._nest.iaf_cond_alpha_mc import iaf_cond_alpha_mc
from ._nest.iaf_cond_beta import iaf_cond_beta
from ._nest.iaf_cond_exp import iaf_cond_exp
from ._nest.iaf_cond_exp_sfa_rr import iaf_cond_exp_sfa_rr
from ._nest.iaf_chs_2007 import iaf_chs_2007
from ._nest.iaf_chxk_2008 import iaf_chxk_2008
from ._nest.iaf_psc_alpha import iaf_psc_alpha
from ._nest.iaf_psc_exp import iaf_psc_exp
from ._nest.iaf_psc_exp_multisynapse import iaf_psc_exp_multisynapse
from ._nest.iaf_psc_alpha_multisynapse import iaf_psc_alpha_multisynapse
from ._nest.iaf_psc_exp_htum import iaf_psc_exp_htum
from ._nest.iaf_tum_2000 import iaf_tum_2000
from ._nest.iaf_psc_exp_ps import iaf_psc_exp_ps
from ._nest.iaf_psc_exp_ps_lossless import iaf_psc_exp_ps_lossless
from ._nest.iaf_psc_alpha_ps import iaf_psc_alpha_ps
from ._nest.iaf_bw_2001 import iaf_bw_2001
from ._nest.iaf_bw_2001_exact import iaf_bw_2001_exact
from ._nest.rate_neuron_ipn import rate_neuron_ipn
from ._nest.rate_neuron_opn import rate_neuron_opn
from ._nest.lin_rate import lin_rate_ipn, lin_rate_opn
from ._nest.gauss_rate import gauss_rate_ipn
from ._nest.sigmoid_rate import sigmoid_rate_ipn
from ._nest.sigmoid_rate_gg_1998 import sigmoid_rate_gg_1998_ipn
from ._nest.tanh_rate import tanh_rate_ipn, tanh_rate_opn
from ._nest.threshold_lin_rate import threshold_lin_rate_ipn, threshold_lin_rate_opn
from ._nest.rate_transformer_node import rate_transformer_node
from ._nest.siegert_neuron import siegert_neuron
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

    # NEST-compatible models
    'dc_generator',
    'multimeter',
    'correlation_detector',
    'correlomatrix_detector',
    'correlospinmatrix_detector',
    'spike_recorder',
    'spin_detector',
    'weight_recorder',
    'gap_junction',
    'diffusion_connection',
    'rate_connection_instantaneous',
    'rate_connection_delayed',
    'sic_connection',
    'ht_synapse',
    'clopath_synapse',
    'jonke_synapse',
    'urbanczik_synapse',
    'vogels_sprekeler_synapse',
    'volume_transmitter',
    'aeif_cond_alpha',
    'aeif_cond_exp',
    'aeif_psc_alpha',
    'aeif_psc_exp',
    'aeif_psc_delta',
    'aeif_psc_delta_clopath',
    'aeif_cond_alpha_multisynapse',
    'aeif_cond_beta_multisynapse',
    'aeif_cond_alpha_astro',
    'iaf_psc_delta',
    'iaf_psc_delta_ps',
    'iaf_cond_alpha',
    'iaf_cond_alpha_mc',
    'iaf_cond_beta',
    'iaf_cond_exp',
    'iaf_cond_exp_sfa_rr',
    'iaf_chs_2007',
    'iaf_chxk_2008',
    'iaf_psc_alpha',
    'iaf_psc_exp',
    'iaf_psc_exp_multisynapse',
    'iaf_psc_alpha_multisynapse',
    'iaf_psc_exp_htum',
    'iaf_tum_2000',
    'iaf_psc_exp_ps',
    'iaf_psc_exp_ps_lossless',
    'iaf_psc_alpha_ps',
    'iaf_bw_2001',
    'iaf_bw_2001_exact',
    'rate_neuron_ipn',
    'rate_neuron_opn',
    'lin_rate_ipn',
    'lin_rate_opn',
    'gauss_rate_ipn',
    'sigmoid_rate_ipn',
    'sigmoid_rate_gg_1998_ipn',
    'tanh_rate_ipn',
    'tanh_rate_opn',
    'threshold_lin_rate_ipn',
    'threshold_lin_rate_opn',
    'rate_transformer_node',
    'siegert_neuron',
]
