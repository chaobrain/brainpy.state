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

from __future__ import annotations

from collections.abc import Mapping

from .stdp_synapse import stdp_synapse

__all__ = [
    'stdp_synapse_hom',
]


class stdp_synapse_hom(stdp_synapse):
    r"""NEST-compatible ``stdp_synapse_hom`` connection model.

    Short description
    -----------------

    Synapse type for spike-timing dependent plasticity with homogeneous
    plasticity parameters.

    Description
    -----------

    ``stdp_synapse_hom`` mirrors NEST ``models/stdp_synapse_hom.h``.
    The STDP update itself is identical to :class:`stdp_synapse` and follows
    the same send ordering:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\, t_{pre}-d]`.
    2. Apply potentiation for each postsynaptic spike in that interval.
    3. Apply depression from postsynaptic trace
       :math:`K^{-}(t_{pre}-d)`.
    4. Send the event with updated weight.
    5. Update presynaptic trace ``Kplus``.
    6. Set ``t_lastspike = t_pre``.

    Homogeneous-property semantics
    ------------------------------

    In NEST, ``tau_plus``, ``lambda``, ``alpha``, ``mu_plus``, ``mu_minus``
    and ``Wmax`` are common model properties shared by all synapses of this
    model. Per-connection state remains ``weight`` and ``Kplus``.

    This implementation mirrors those connect-time semantics by rejecting
    common-property keys in :meth:`check_synapse_params`.

    Validation semantics
    --------------------

    Unlike NEST ``stdp_synapse``, NEST ``stdp_synapse_hom`` does not enforce
    the explicit ``weight``/``Wmax`` sign check or ``Kplus >= 0`` check in
    its ``set_status`` path. This class follows that behavior.

    Notes
    -----
    - Like :class:`stdp_synapse`, this backend stores ``tau_minus`` on the
      synapse for standalone compatibility, while in NEST it belongs to the
      postsynaptic archiving neuron.
    - Event timing uses on-grid spike stamps and ignores sub-step offsets.

    References
    ----------
    .. [1] NEST source: ``models/stdp_synapse_hom.h`` and
           ``models/stdp_synapse_hom.cpp``.
    .. [2] Guetig R, Aharonov R, Rotter S, Sompolinsky H (2003).
           Learning input correlations through nonlinear temporally asymmetric
           Hebbian plasticity. Journal of Neuroscience, 23(9):3697-3714.
    """

    __module__ = 'brainpy.state'

    @staticmethod
    def _validate_non_negative(value: float, *, name: str):
        # NEST stdp_synapse_hom::set_status does not enforce Kplus >= 0.
        del value, name

    @classmethod
    def _validate_weight_wmax_sign(cls, weight: float, Wmax: float):
        # NEST stdp_synapse_hom::set_status does not enforce equal sign.
        del cls, weight, Wmax

    def get(self) -> dict:
        """Return current public parameters and mutable state."""
        params = super().get()
        params['synapse_model'] = 'stdp_synapse_hom'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        """Reject common-property assignments in connect-time synapse specs."""
        if syn_spec is None:
            return
        disallowed = ('tau_plus', 'lambda', 'alpha', 'mu_plus', 'mu_minus', 'Wmax')
        for key in disallowed:
            if key in syn_spec:
                raise ValueError(
                    f'{key} cannot be specified in connect-time synapse parameters '
                    'for stdp_synapse_hom; set common properties on the model '
                    'itself (for example via CopyModel()/SetDefaults()).'
                )
