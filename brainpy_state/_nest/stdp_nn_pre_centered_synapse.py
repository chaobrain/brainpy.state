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

import math

from brainstate.typing import ArrayLike

from .stdp_synapse import stdp_synapse

__all__ = [
    'stdp_nn_pre_centered_synapse',
]


_STDP_EPS = 1.0e-6


class stdp_nn_pre_centered_synapse(stdp_synapse):
    r"""NEST-compatible ``stdp_nn_pre_centered_synapse`` connection model.

    Short description
    -----------------

    Synapse type for spike-timing dependent plasticity with presynaptic-
    centered nearest-neighbour spike pairing.

    Description
    -----------

    ``stdp_nn_pre_centered_synapse`` mirrors NEST
    ``models/stdp_nn_pre_centered_synapse.h`` and implements the pairing
    scheme described by Izhikevich and Desai (2003) and Morrison et al. (2008):

    - each presynaptic spike is depressed by the nearest preceding
      postsynaptic spike,
    - each postsynaptic spike facilitates all presynaptic spikes that occurred
      after the previous postsynaptic spike.

    Compared with :class:`stdp_synapse`, this model introduces nearest-neighbor
    postsynaptic depression and a presynaptic trace reset behavior:

    - ``Kplus`` decays with ``tau_plus``, increments by ``1`` per pre-spike,
      and is reset to ``0`` when any postsynaptic spike occurred in
      :math:`(t_{\mathrm{last}}-d,\, t_{pre}-d]`.
    - the depression trace term is nearest-neighbor only:
      :math:`\exp((t_{post}^{\mathrm{last}}-t)/\tau_{-})`, where
      :math:`t_{post}^{\mathrm{last}} < t`.

    Update order (NEST source equivalent)
    -------------------------------------

    For a presynaptic spike at :math:`t_{pre}` with dendritic delay :math:`d`,
    NEST ``stdp_nn_pre_centered_synapse::send`` performs:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\, t_{pre}-d]`.
    2. If non-empty, use only the first postsynaptic spike in this interval for
       facilitation with
       :math:`Kplus \exp((t_{\mathrm{last}}-(t_{post}+d))/\tau_+)`.
    3. If step 2 happened, reset ``Kplus = 0``.
    4. Apply depression from nearest-neighbor postsynaptic trace at
       :math:`t_{pre}-d`.
    5. Update ``Kplus`` as
       :math:`Kplus \leftarrow Kplus \exp((t_{\mathrm{last}}-t_{pre})/\tau_+) + 1`.
    6. Send event with updated ``weight``.
    7. Set ``t_lastspike = t_pre``.

    This implementation preserves the same ordering.

    Coincidence semantics
    ---------------------

    Pairs with exact coincidence are discarded by strict time comparisons
    (NEST ``stdp_eps`` behavior). If
    ``presynaptic_spike == postsynaptic_spike + dendritic_delay``,
    the coincident postsynaptic spike is not used for depression/facilitation;
    earlier valid nearest neighbors are used instead.

    Parameters
    ----------
    Parameters and defaults match :class:`stdp_synapse`:
    ``weight``, ``delay``, ``receptor_type``, ``tau_plus``, ``tau_minus``,
    ``lambda_``, ``alpha``, ``mu_plus``, ``mu_minus``, ``Wmax``, ``Kplus``,
    ``post``, ``name``.

    Notes
    -----
    - In NEST, ``tau_minus`` belongs to the postsynaptic archiving neuron.
      This backend stores equivalent state locally for standalone
      compatibility, while preserving update semantics.
    - As in NEST, the model uses on-grid spike stamps and ignores sub-step
      precise spike offsets for STDP updates.

    References
    ----------
    .. [1] NEST source: ``models/stdp_nn_pre_centered_synapse.h`` and
           ``models/stdp_nn_pre_centered_synapse.cpp``.
    .. [2] Izhikevich EM, Desai NS (2003). Relating STDP to BCM.
           Neural Computation, 15:1511-1523.
    .. [3] Morrison A, Diesmann M, Gerstner W (2008).
           Phenomenological models of synaptic plasticity based on spike timing.
           Biological Cybernetics, 98:459-478.
    """

    __module__ = 'brainpy.state'

    def _get_nearest_neighbor_K_value(self, t_ms: float) -> float:
        # Match ArchivingNode::get_K_values nearest-neighbor behavior:
        # use latest post spike strictly before t and decay a unit trace.
        for idx in range(len(self._post_hist_t) - 1, -1, -1):
            t_post = self._post_hist_t[idx]
            if (t_ms - t_post) > _STDP_EPS:
                return math.exp((t_post - t_ms) / self.tau_minus)
        return 0.0

    def get(self) -> dict:
        """Return current public parameters and mutable state."""
        params = super().get()
        params['synapse_model'] = 'stdp_nn_pre_centered_synapse'
        return params

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        """Schedule one outgoing event with NEST ``stdp_nn_pre_centered_synapse`` dynamics."""
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST uses on-grid event stamps in this model.
        t_spike = self._current_time_ms() + dt_ms
        dendritic_delay = float(self.delay)

        # Read postsynaptic history in (t_lastspike - d, t_spike - d].
        t1 = self.t_lastspike - dendritic_delay
        t2 = t_spike - dendritic_delay
        history = self._get_post_history_times(t1, t2)

        # Facilitation from the first postsynaptic spike in the interval.
        if history:
            minus_dt = self.t_lastspike - (history[0] + dendritic_delay)
            assert minus_dt < (-1.0 * _STDP_EPS)
            kplus_term = self.Kplus * math.exp(minus_dt / self.tau_plus)
            self.weight = float(self._facilitate(float(self.weight), float(kplus_term)))

            # Pre-centered nearest-neighbor scheme forgets previous pre spikes
            # once a post spike happened between current and previous pre spike.
            self.Kplus = 0.0

        # Depression from nearest preceding postsynaptic spike.
        kminus_value = self._get_nearest_neighbor_K_value(t_spike - dendritic_delay)
        self.weight = float(self._depress(float(self.weight), float(kminus_value)))

        self.Kplus = float(self.Kplus * math.exp((self.t_lastspike - t_spike) / self.tau_plus) + 1.0)

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True
