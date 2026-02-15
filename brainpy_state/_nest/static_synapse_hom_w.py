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

from brainstate.typing import ArrayLike

from .static_synapse import static_synapse

__all__ = [
    'static_synapse_hom_w',
]


class static_synapse_hom_w(static_synapse):
    r"""NEST-compatible ``static_synapse_hom_w`` connection model.

    Short description
    -----------------

    Synapse type for static connections with homogeneous weight.

    Description
    -----------

    ``static_synapse_hom_w`` mirrors NEST
    ``models/static_synapse_hom_w.h``. It behaves like
    :class:`static_synapse` for event transmission and delay scheduling,
    but differs in one key semantic:

    - The synaptic weight is a *common model property* shared across
      connections of the same model (NEST ``CommonPropertiesHomW``),
      not an individual connection parameter.

    Event send ordering
    -------------------

    NEST applies event fields in this order:

    1. ``e.set_weight(cp.get_weight())``
    2. ``e.set_delay_steps(get_delay_steps())``
    3. ``e.set_receiver(*get_target(tid))``
    4. ``e.set_rport(get_rport())``
    5. ``e()`` (deliver event)

    This implementation preserves the same ordering semantics by inheriting
    :class:`static_synapse` scheduling logic unchanged.

    Homogeneous-weight semantics
    ----------------------------

    - ``set(weight=...)`` updates the model's common weight.
    - ``set_weight(...)`` is intentionally forbidden to mirror NEST's
      per-connection weight restriction.
    - ``check_synapse_params(...)`` rejects ``weight`` in connection-level
      specifications.

    Parameters
    ----------
    weight : ArrayLike, optional
        Common synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    post : object, optional
        Default receiver object.
    event_type : str, optional
        Event transmission type. Same supported values as
        :class:`static_synapse`. Default: ``'spike'``.
    name : str, optional
        Object name.

    References
    ----------
    .. [1] NEST source: ``models/static_synapse_hom_w.h`` and
           ``nestkernel/common_properties_hom_w.h``.
    """

    __module__ = 'brainpy.state'

    def get(self) -> dict:
        """Return current public parameters."""
        params = super().get()
        params['synapse_model'] = 'static_synapse_hom_w'
        return params

    def set_weight(self, weight: ArrayLike):
        """Mirror NEST restriction on per-connection weight setting."""
        del weight
        raise ValueError(
            'Setting of individual weights is not possible! The common weights '
            'can be changed via CopyModel().'
        )

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        """Reject per-connection ``weight`` in synapse specifications."""
        if syn_spec is None:
            return
        if 'weight' in syn_spec:
            raise ValueError(
                'Weight cannot be specified since it needs to be equal for all '
                'connections when static_synapse_hom_w is used.'
            )
