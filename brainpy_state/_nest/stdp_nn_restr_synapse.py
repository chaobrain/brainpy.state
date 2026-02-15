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

import brainunit as u
from brainstate.typing import ArrayLike

from .stdp_synapse import _STDP_EPS, stdp_synapse

__all__ = [
    'stdp_nn_restr_synapse',
]


class stdp_nn_restr_synapse(stdp_synapse):
    r"""NEST-compatible ``stdp_nn_restr_synapse`` connection model.

    Short description
    -----------------

    Synapse type for spike-timing dependent plasticity with restricted
    symmetric nearest-neighbour spike pairing.

    Description
    -----------

    ``stdp_nn_restr_synapse`` mirrors NEST
    ``models/stdp_nn_restr_synapse.h`` and implements the restricted
    nearest-neighbor pairing scheme from Morrison et al. (2008, fig. 7C):

    - on a presynaptic spike, depression uses the nearest preceding
      postsynaptic spike only if that postsynaptic spike occurred after the
      previous presynaptic spike,
    - on postsynaptic spikes, facilitation pairs only with the nearest
      preceding presynaptic spike that has not already been used for
      facilitation.

    A spike therefore participates in at most one depression pair and at most
    one facilitation pair.

    Compared with :class:`stdp_synapse`, this model changes two core STDP
    mechanisms:

    - no running presynaptic ``Kplus`` trace is used,
    - depression is nearest-neighbor and restricted to intervals where at
      least one postsynaptic spike occurred since the last presynaptic spike.

    Update order (NEST source equivalent)
    -------------------------------------

    For a presynaptic spike at :math:`t_{pre}` with dendritic delay :math:`d`,
    NEST ``stdp_nn_restr_synapse::send`` performs:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\, t_{pre}-d]`.
    2. If history is non-empty, facilitate once using the first postsynaptic
       spike in that interval:
       :math:`\exp((t_{\mathrm{last}}-(t_{post}+d))/\tau_+)`.
    3. If history is non-empty, depress once using nearest-neighbor
       postsynaptic trace at :math:`t_{pre}-d`:
       :math:`\exp((t_{post}^{\mathrm{nn}}-(t_{pre}-d))/\tau_-)`.
    4. Send event with updated ``weight``.
    5. Set ``t_lastspike = t_pre``.

    This implementation preserves that exact ordering.

    Coincidence semantics
    ---------------------

    Pairs with exact coincidence are discarded by strict time comparisons
    (NEST ``stdp_eps`` behavior). If
    ``presynaptic_spike == postsynaptic_spike + dendritic_delay``,
    :math:`\Delta t = 0` is not used; the nearest strictly earlier valid
    post-spike is used instead.

    Parameters
    ----------
    weight : ArrayLike, optional
        Initial synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    tau_plus : ArrayLike, optional
        Potentiation time constant ``tau_plus`` in ms. Default: ``20.0 * u.ms``.
    tau_minus : ArrayLike, optional
        Depression trace time constant ``tau_minus`` in ms.
        In NEST this belongs to the postsynaptic archiving neuron; here it is
        stored on the synapse for standalone compatibility.
        Default: ``20.0 * u.ms``.
    lambda_ : ArrayLike, optional
        Learning-rate parameter ``lambda``. Default: ``0.01``.
    alpha : ArrayLike, optional
        Depression scaling parameter. Default: ``1.0``.
    mu_plus : ArrayLike, optional
        Potentiation exponent. Default: ``1.0``.
    mu_minus : ArrayLike, optional
        Depression exponent. Default: ``1.0``.
    Wmax : ArrayLike, optional
        Maximum weight bound. Must have same sign as ``weight``.
        Default: ``100.0``.
    post : object, optional
        Default receiver object.
    name : str, optional
        Object name.

    Notes
    -----
    - In NEST, ``tau_minus`` is a postsynaptic-neuron parameter.
    - As in NEST, STDP updates are based on on-grid spike stamps and ignore
      sub-step precise offsets.

    References
    ----------
    .. [1] NEST source: ``models/stdp_nn_restr_synapse.h`` and
           ``models/stdp_nn_restr_synapse.cpp``.
    .. [2] Morrison A, Diesmann M, Gerstner W (2008).
           Phenomenological models of synaptic plasticity based on spike timing.
           Biological Cybernetics, 98:459-478.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        lambda_: ArrayLike = 0.01,
        alpha: ArrayLike = 1.0,
        mu_plus: ArrayLike = 1.0,
        mu_minus: ArrayLike = 1.0,
        Wmax: ArrayLike = 100.0,
        post=None,
        name: str | None = None,
    ):
        super().__init__(
            weight=weight,
            delay=delay,
            receptor_type=receptor_type,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            lambda_=lambda_,
            alpha=alpha,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            Wmax=Wmax,
            Kplus=0.0,
            post=post,
            name=name,
        )

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
        params.pop('Kplus', None)
        params['synapse_model'] = 'stdp_nn_restr_synapse'
        return params

    def set(self, **kwargs):
        """Set NEST-style public parameters and mutable state."""
        if 'Kplus' in kwargs:
            raise ValueError('Kplus is not a parameter of stdp_nn_restr_synapse.')
        super().set(**kwargs)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        """Schedule one outgoing event with NEST ``stdp_nn_restr_synapse`` dynamics."""
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

        # Restricted nearest-neighbor rule: both facilitation and depression
        # are applied only if there was at least one post spike between the
        # previous and current pre spike.
        if history:
            minus_dt = self.t_lastspike - (history[0] + dendritic_delay)
            assert minus_dt < (-1.0 * _STDP_EPS)
            self.weight = float(self._facilitate(float(self.weight), math.exp(minus_dt / self.tau_plus)))

            kminus_value = self._get_nearest_neighbor_K_value(t_spike - dendritic_delay)
            self.weight = float(self._depress(float(self.weight), float(kminus_value)))

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True
