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

from .static_synapse import _UNSET, static_synapse
from .stdp_synapse import _STDP_EPS, stdp_synapse

__all__ = [
    'stdp_triplet_synapse',
]


class stdp_triplet_synapse(stdp_synapse):
    r"""NEST-compatible ``stdp_triplet_synapse`` connection model.

    Short description
    -----------------

    Synapse type with spike-timing dependent plasticity (triplets).

    Description
    -----------

    ``stdp_triplet_synapse`` mirrors NEST ``models/stdp_triplet_synapse.h`` and
    implements the triplet STDP rule of Pfister and Gerstner (2006).

    Per-connection state:

    - ``weight``: synaptic efficacy,
    - ``Kplus``: short presynaptic trace ``r_1``,
    - ``Kplus_triplet``: long presynaptic trace ``r_2``,
    - ``t_lastspike``: previous presynaptic spike stamp.

    In NEST, postsynaptic traces live in the postsynaptic archiving neuron:
    ``Kminus`` (``o_1``) with ``tau_minus``, and ``Kminus_triplet`` (``o_2``)
    with ``tau_minus_triplet``. This backend reproduces that behavior with a
    local post-spike history buffer for standalone compatibility.

    Update order (NEST source equivalent)
    -------------------------------------

    For a presynaptic spike at stamp :math:`t_{pre}` and dendritic delay
    :math:`d`, NEST ``stdp_triplet_synapse::send`` performs:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\, t_{pre}-d]`.
    2. For each postsynaptic spike :math:`t_{post}` in that range, facilitate
       with:
       :math:`Kplus \exp((t_{\mathrm{last}}-(t_{post}+d))/\tau_{plus})`
       and :math:`ky = Kminus\_triplet(t_{post}^+) - 1`.
    3. Decay ``Kplus_triplet`` to current pre-spike stamp.
    4. Depress using postsynaptic pair trace at :math:`t_{pre}-d` and the
       decayed ``Kplus_triplet``.
    5. Increment ``Kplus_triplet`` by ``1``.
    6. Decay/increment ``Kplus``.
    7. Send event with updated weight.
    8. Set ``t_lastspike = t_pre``.

    This implementation preserves the same ordering.

    Weight update functions
    -----------------------

    Matching NEST C++ implementation:

    .. math::
       w \leftarrow \operatorname{copysign}\left(
       \min\left(\left|w\right| + k_+\left(A_+ + A_{3+} k_y\right), \left|W_{max}\right|\right),
       W_{max}
       \right)

    .. math::
       w \leftarrow \operatorname{copysign}\left(
       \max\left(\left|w\right| - k_-\left(A_- + A_{3-} Kplus\_{triplet}\right), 0\right),
       W_{max}
       \right)

    Event timing semantics
    ----------------------

    As in NEST, this model uses on-grid spike stamps and ignores precise
    sub-step offsets during STDP updates.

    Parameters
    ----------
    weight : ArrayLike, optional
        Initial synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    tau_plus : ArrayLike, optional
        Time constant of short presynaptic trace in ms. Default: ``16.8 * u.ms``.
    tau_plus_triplet : ArrayLike, optional
        Time constant of long presynaptic trace in ms. Default: ``101.0 * u.ms``.
    tau_minus : ArrayLike, optional
        Time constant of postsynaptic pair trace in ms.
        In NEST this belongs to the postsynaptic archiving neuron.
        Default: ``20.0 * u.ms``.
    tau_minus_triplet : ArrayLike, optional
        Time constant of postsynaptic triplet trace in ms.
        In NEST this belongs to the postsynaptic archiving neuron.
        Default: ``110.0 * u.ms``.
    Aplus : ArrayLike, optional
        Pair potentiation coefficient. Default: ``5e-10``.
    Aminus : ArrayLike, optional
        Pair depression coefficient. Default: ``7e-3``.
    Aplus_triplet : ArrayLike, optional
        Triplet potentiation coefficient. Default: ``6.2e-3``.
    Aminus_triplet : ArrayLike, optional
        Triplet depression coefficient. Default: ``2.3e-4``.
    Wmax : ArrayLike, optional
        Maximum absolute weight bound. Must have same sign as ``weight``.
        Default: ``100.0``.
    Kplus : ArrayLike, optional
        Initial short presynaptic trace. Must be non-negative.
        Default: ``0.0``.
    Kplus_triplet : ArrayLike, optional
        Initial long presynaptic trace. Must be non-negative.
        Default: ``0.0``.
    post : object, optional
        Default receiver object.
    name : str, optional
        Object name.

    Notes
    -----
    - The model transmits spike-like events only.
    - ``update(pre_spike=..., post_spike=...)`` supports integer multiplicities
      for standalone STDP simulations.

    References
    ----------
    .. [1] NEST source: ``models/stdp_triplet_synapse.h`` and
           ``models/stdp_triplet_synapse.cpp``.
    .. [2] Pfister JP, Gerstner W (2006). Triplets of spikes in a model of
           spike timing-dependent plasticity. Journal of Neuroscience, 26(38),
           9673-9682. https://doi.org/10.1523/JNEUROSCI.1425-06.2006
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 16.8 * u.ms,
        tau_plus_triplet: ArrayLike = 101.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        tau_minus_triplet: ArrayLike = 110.0 * u.ms,
        Aplus: ArrayLike = 5e-10,
        Aminus: ArrayLike = 7e-3,
        Aplus_triplet: ArrayLike = 6.2e-3,
        Aminus_triplet: ArrayLike = 2.3e-4,
        Wmax: ArrayLike = 100.0,
        Kplus: ArrayLike = 0.0,
        Kplus_triplet: ArrayLike = 0.0,
        post=None,
        name: str | None = None,
    ):
        super().__init__(
            weight=weight,
            delay=delay,
            receptor_type=receptor_type,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            Wmax=Wmax,
            Kplus=Kplus,
            post=post,
            name=name,
        )

        self.tau_plus_triplet = self._to_scalar_time_ms(tau_plus_triplet, name='tau_plus_triplet')
        self.tau_minus_triplet = self._to_scalar_time_ms(tau_minus_triplet, name='tau_minus_triplet')
        self.Aplus = self._to_scalar_float(Aplus, name='Aplus')
        self.Aminus = self._to_scalar_float(Aminus, name='Aminus')
        self.Aplus_triplet = self._to_scalar_float(Aplus_triplet, name='Aplus_triplet')
        self.Aminus_triplet = self._to_scalar_float(Aminus_triplet, name='Aminus_triplet')
        self.Kplus_triplet = self._to_scalar_float(Kplus_triplet, name='Kplus_triplet')

        self._validate_non_negative(self.Kplus_triplet, name='Kplus_triplet')

        self._Kplus_triplet0 = float(self.Kplus_triplet)

        self._post_kminus_triplet = 0.0
        self._post_hist_kminus_triplet: list[float] = []

    def _facilitate(self, w: float, kplus: float, ky: float) -> float:
        new_w = abs(w) + kplus * (self.Aplus + self.Aplus_triplet * ky)
        w_abs_max = abs(self.Wmax)
        return math.copysign(new_w if new_w < w_abs_max else w_abs_max, self.Wmax)

    def _depress(self, w: float, kminus: float, kplus_triplet: float) -> float:
        new_w = abs(w) - kminus * (self.Aminus + self.Aminus_triplet * kplus_triplet)
        return math.copysign(new_w if new_w > 0.0 else 0.0, self.Wmax)

    def clear_post_history(self):
        """Clear internal postsynaptic STDP history state."""
        self._post_kminus = 0.0
        self._post_kminus_triplet = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t = []
        self._post_hist_kminus = []
        self._post_hist_kminus_triplet = []

    def _record_post_spike_at(self, t_spike_ms: float):
        self._post_kminus = (
            self._post_kminus * math.exp((self._last_post_spike - t_spike_ms) / self.tau_minus) + 1.0
        )
        self._post_kminus_triplet = (
            self._post_kminus_triplet * math.exp((self._last_post_spike - t_spike_ms) / self.tau_minus_triplet) + 1.0
        )
        self._last_post_spike = float(t_spike_ms)
        self._post_hist_t.append(float(t_spike_ms))
        self._post_hist_kminus.append(float(self._post_kminus))
        self._post_hist_kminus_triplet.append(float(self._post_kminus_triplet))

    def _get_post_history_entries(self, t1_ms: float, t2_ms: float) -> list[tuple[float, float]]:
        t1_lim = float(t1_ms + _STDP_EPS)
        t2_lim = float(t2_ms + _STDP_EPS)
        selected: list[tuple[float, float]] = []
        for t_post, kminus_triplet in zip(self._post_hist_t, self._post_hist_kminus_triplet):
            if t_post >= t1_lim and t_post < t2_lim:
                selected.append((float(t_post), float(kminus_triplet)))
        return selected

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        super().init_state()
        self.Kplus_triplet = float(self._Kplus_triplet0)

    def get(self) -> dict:
        """Return current public parameters and mutable state."""
        params = static_synapse.get(self)
        params['tau_plus'] = float(self.tau_plus)
        params['tau_plus_triplet'] = float(self.tau_plus_triplet)
        params['tau_minus'] = float(self.tau_minus)
        params['tau_minus_triplet'] = float(self.tau_minus_triplet)
        params['Aplus'] = float(self.Aplus)
        params['Aminus'] = float(self.Aminus)
        params['Aplus_triplet'] = float(self.Aplus_triplet)
        params['Aminus_triplet'] = float(self.Aminus_triplet)
        params['Wmax'] = float(self.Wmax)
        params['Kplus'] = float(self.Kplus)
        params['Kplus_triplet'] = float(self.Kplus_triplet)
        params['synapse_model'] = 'stdp_triplet_synapse'
        return params

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        tau_plus: ArrayLike | object = _UNSET,
        tau_plus_triplet: ArrayLike | object = _UNSET,
        tau_minus: ArrayLike | object = _UNSET,
        tau_minus_triplet: ArrayLike | object = _UNSET,
        Aplus: ArrayLike | object = _UNSET,
        Aminus: ArrayLike | object = _UNSET,
        Aplus_triplet: ArrayLike | object = _UNSET,
        Aminus_triplet: ArrayLike | object = _UNSET,
        Wmax: ArrayLike | object = _UNSET,
        Kplus: ArrayLike | object = _UNSET,
        Kplus_triplet: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        """Set NEST-style public parameters and mutable state."""
        new_weight = self.weight if weight is _UNSET else self._to_scalar_float(weight, name='weight')
        new_tau_plus = (
            self.tau_plus
            if tau_plus is _UNSET
            else self._to_scalar_time_ms(tau_plus, name='tau_plus')
        )
        new_tau_plus_triplet = (
            self.tau_plus_triplet
            if tau_plus_triplet is _UNSET
            else self._to_scalar_time_ms(tau_plus_triplet, name='tau_plus_triplet')
        )
        new_tau_minus = (
            self.tau_minus
            if tau_minus is _UNSET
            else self._to_scalar_time_ms(tau_minus, name='tau_minus')
        )
        new_tau_minus_triplet = (
            self.tau_minus_triplet
            if tau_minus_triplet is _UNSET
            else self._to_scalar_time_ms(tau_minus_triplet, name='tau_minus_triplet')
        )
        new_Aplus = self.Aplus if Aplus is _UNSET else self._to_scalar_float(Aplus, name='Aplus')
        new_Aminus = self.Aminus if Aminus is _UNSET else self._to_scalar_float(Aminus, name='Aminus')
        new_Aplus_triplet = (
            self.Aplus_triplet
            if Aplus_triplet is _UNSET
            else self._to_scalar_float(Aplus_triplet, name='Aplus_triplet')
        )
        new_Aminus_triplet = (
            self.Aminus_triplet
            if Aminus_triplet is _UNSET
            else self._to_scalar_float(Aminus_triplet, name='Aminus_triplet')
        )
        new_Wmax = self.Wmax if Wmax is _UNSET else self._to_scalar_float(Wmax, name='Wmax')
        new_Kplus = self.Kplus if Kplus is _UNSET else self._to_scalar_float(Kplus, name='Kplus')
        new_Kplus_triplet = (
            self.Kplus_triplet
            if Kplus_triplet is _UNSET
            else self._to_scalar_float(Kplus_triplet, name='Kplus_triplet')
        )

        self._validate_weight_wmax_sign(float(new_weight), float(new_Wmax))
        self._validate_non_negative(float(new_Kplus), name='Kplus')
        self._validate_non_negative(float(new_Kplus_triplet), name='Kplus_triplet')

        super_kwargs = {}
        if weight is not _UNSET:
            super_kwargs['weight'] = float(new_weight)
        if delay is not _UNSET:
            super_kwargs['delay'] = delay
        if receptor_type is not _UNSET:
            super_kwargs['receptor_type'] = receptor_type
        if post is not _UNSET:
            super_kwargs['post'] = post
        if super_kwargs:
            static_synapse.set(self, **super_kwargs)

        self.tau_plus = float(new_tau_plus)
        self.tau_plus_triplet = float(new_tau_plus_triplet)
        self.tau_minus = float(new_tau_minus)
        self.tau_minus_triplet = float(new_tau_minus_triplet)
        self.Aplus = float(new_Aplus)
        self.Aminus = float(new_Aminus)
        self.Aplus_triplet = float(new_Aplus_triplet)
        self.Aminus_triplet = float(new_Aminus_triplet)
        self.Wmax = float(new_Wmax)
        self.Kplus = float(new_Kplus)
        self.Kplus_triplet = float(new_Kplus_triplet)

        self._Kplus0 = float(self.Kplus)
        self._Kplus_triplet0 = float(self.Kplus_triplet)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        """Schedule one outgoing event with NEST ``stdp_triplet_synapse`` dynamics."""
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST uses on-grid event stamps in this model.
        t_spike = self._current_time_ms() + dt_ms
        dendritic_delay = float(self.delay)

        # Facilitation due to postsynaptic spikes in
        # (t_lastspike - dendritic_delay, t_spike - dendritic_delay].
        t1 = self.t_lastspike - dendritic_delay
        t2 = t_spike - dendritic_delay
        for t_post, kminus_triplet_at_post in self._get_post_history_entries(t1, t2):
            minus_dt = self.t_lastspike - (t_post + dendritic_delay)
            assert minus_dt < (-1.0 * _STDP_EPS)
            ky = kminus_triplet_at_post - 1.0
            kplus_term = self.Kplus * math.exp(minus_dt / self.tau_plus)
            self.weight = float(self._facilitate(float(self.weight), float(kplus_term), float(ky)))

        # Depression due to current presynaptic spike.
        self.Kplus_triplet = float(
            self.Kplus_triplet * math.exp((self.t_lastspike - t_spike) / self.tau_plus_triplet)
        )
        kminus_value = self._get_K_value(t_spike - dendritic_delay)
        self.weight = float(
            self._depress(float(self.weight), float(kminus_value), float(self.Kplus_triplet))
        )

        self.Kplus_triplet = float(self.Kplus_triplet + 1.0)
        self.Kplus = float(self.Kplus * math.exp((self.t_lastspike - t_spike) / self.tau_plus) + 1.0)

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True
