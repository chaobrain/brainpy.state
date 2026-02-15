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
from collections.abc import Mapping

import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike

from .static_synapse import static_synapse

__all__ = [
    'stdp_pl_synapse_hom',
]


_UNSET = object()
_STDP_EPS = 1.0e-6


class stdp_pl_synapse_hom(static_synapse):
    r"""NEST-compatible ``stdp_pl_synapse_hom`` connection model.

    Short description
    -----------------

    Synapse type for spike-timing dependent plasticity with power-law
    potentiation and homogeneous plasticity parameters.

    Description
    -----------

    ``stdp_pl_synapse_hom`` mirrors NEST ``models/stdp_pl_synapse_hom.h``.
    It implements the Morrison et al. (2007) power-law STDP rule with
    per-connection state:

    - ``weight``: synaptic efficacy,
    - ``Kplus``: presynaptic eligibility trace,
    - ``t_lastspike``: timestamp of the previous presynaptic spike.

    In NEST, ``tau_minus`` belongs to the postsynaptic archiving neuron.
    For standalone compatibility in this backend, equivalent behavior is
    implemented via an internal post-spike history buffer parameterized by
    ``tau_minus``.

    Update order (NEST source equivalent)
    -------------------------------------

    For a presynaptic spike at stamp :math:`t_{pre}` and dendritic delay
    :math:`d`, NEST ``stdp_pl_synapse_hom::send`` applies:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\, t_{pre}-d]`.
    2. For each postsynaptic spike :math:`t_{post}` in that interval, apply
       facilitation with
       :math:`Kplus \exp((t_{\mathrm{last}}-(t_{post}+d))/\tau_+)`.
    3. Apply depression using postsynaptic trace
       :math:`K^{-}(t_{pre}-d)`.
    4. Send event with updated ``weight``.
    5. Update presynaptic trace:
       :math:`Kplus \leftarrow Kplus \exp((t_{\mathrm{last}}-t_{pre})/\tau_+) + 1`.
    6. Set ``t_lastspike = t_pre``.

    This implementation preserves the same ordering.

    Plasticity functions
    --------------------

    Potentiation:

    .. math::
       w \leftarrow w + \lambda w^{\mu} k_+

    Depression:

    .. math::
       w \leftarrow w - \alpha \lambda w k_-

    with depression clipped to non-negative values:
    :math:`w \leftarrow \max(w, 0)`.

    Homogeneous-property semantics
    ------------------------------

    In NEST, ``tau_plus``, ``lambda``, ``alpha`` and ``mu`` are common model
    properties shared by all synapses of this type, while ``weight`` and
    ``Kplus`` are per-connection.

    This implementation mirrors NEST connect-time semantics by rejecting
    common-property keys in :meth:`check_synapse_params`.

    Event timing semantics
    ----------------------

    As in NEST, this model uses on-grid spike stamps and ignores precise
    sub-step offsets. At simulation time ``t`` and step size ``dt``, spikes
    handled in the current step are stamped at ``t + dt``.

    Parameters
    ----------
    weight : ArrayLike, optional
        Initial synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    tau_plus : ArrayLike, optional
        Potentiation time constant in ms. Must be ``> 0``.
        Default: ``20.0 * u.ms``.
    tau_minus : ArrayLike, optional
        Depression trace time constant in ms.
        In NEST this belongs to the postsynaptic archiving neuron; here it is
        stored on the synapse for standalone compatibility.
        Default: ``20.0 * u.ms``.
    lambda_ : ArrayLike, optional
        Learning-rate parameter ``lambda``. Default: ``0.1``.
    alpha : ArrayLike, optional
        Depression scaling parameter. Default: ``1.0``.
    mu : ArrayLike, optional
        Power-law potentiation exponent. Default: ``0.4``.
    Kplus : ArrayLike, optional
        Initial presynaptic trace value. Default: ``0.0``.
    post : object, optional
        Default receiver object.
    name : str, optional
        Object name.

    Notes
    -----
    - The model transmits spike-like events only.
    - ``update(pre_spike=..., post_spike=...)`` accepts both pre- and post-
      synaptic spike multiplicities for standalone STDP simulation.
    - ``record_post_spike(...)`` can be used to manually feed postsynaptic
      spikes when the postsynaptic model does not expose NEST archiver APIs.

    References
    ----------
    .. [1] NEST source: ``models/stdp_pl_synapse_hom.h`` and
           ``models/stdp_pl_synapse_hom.cpp``.
    .. [2] Morrison A, Aertsen A, Diesmann M (2007). Spike-timing dependent
           plasticity in balanced random networks.
           Neural Computation, 19(6):1437-1467.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        lambda_: ArrayLike = 0.1,
        alpha: ArrayLike = 1.0,
        mu: ArrayLike = 0.4,
        Kplus: ArrayLike = 0.0,
        post=None,
        name: str | None = None,
    ):
        super().__init__(
            weight=weight,
            delay=delay,
            receptor_type=receptor_type,
            post=post,
            event_type='spike',
            name=name,
        )

        self.tau_plus = self._to_scalar_time_ms(tau_plus, name='tau_plus')
        self.tau_minus = self._to_scalar_time_ms(tau_minus, name='tau_minus')
        self.lambda_ = self._to_scalar_float(lambda_, name='lambda')
        self.alpha = self._to_scalar_float(alpha, name='alpha')
        self.mu = self._to_scalar_float(mu, name='mu')
        self.Kplus = self._to_scalar_float(Kplus, name='Kplus')

        self._validate_tau_plus(self.tau_plus)

        self._Kplus0 = float(self.Kplus)
        self._t_lastspike0 = 0.0

        self.t_lastspike = float(self._t_lastspike0)
        self._post_kminus = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t: list[float] = []
        self._post_hist_kminus: list[float] = []

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        if isinstance(value, u.Quantity):
            unit = u.get_unit(value)
            arr = np.asarray(value.to_decimal(unit), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr.reshape(()))
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        return v

    @staticmethod
    def _to_non_negative_int_count(value: ArrayLike, *, name: str) -> int:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr.reshape(()))
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        if v < 0.0:
            raise ValueError(f'{name} must be non-negative.')
        rounded = int(round(v))
        if not math.isclose(v, float(rounded), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be an integer spike count.')
        return rounded

    @staticmethod
    def _validate_tau_plus(value: float):
        if value <= 0.0:
            raise ValueError('tau_plus must be > 0.')

    def _facilitate(self, w: float, kplus: float) -> float:
        power_term = float(np.power(np.float64(w), np.float64(self.mu)))
        return w + (self.lambda_ * power_term * kplus)

    def _depress(self, w: float, kminus: float) -> float:
        new_w = w - (self.lambda_ * self.alpha * w * kminus)
        return new_w if new_w > 0.0 else 0.0

    def clear_post_history(self):
        """Clear internal postsynaptic STDP history state."""
        self._post_kminus = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t = []
        self._post_hist_kminus = []

    def _record_post_spike_at(self, t_spike_ms: float):
        self._post_kminus = (
            self._post_kminus * math.exp((self._last_post_spike - t_spike_ms) / self.tau_minus) + 1.0
        )
        self._last_post_spike = float(t_spike_ms)
        self._post_hist_t.append(float(t_spike_ms))
        self._post_hist_kminus.append(float(self._post_kminus))

    def record_post_spike(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        t_spike_ms: ArrayLike | None = None,
    ) -> int:
        """Record postsynaptic spikes into the internal STDP history."""
        count = self._to_non_negative_int_count(multiplicity, name='post_spike')
        if count == 0:
            return 0

        if t_spike_ms is None:
            dt_ms = self._refresh_delay_if_needed()
            t_value = self._current_time_ms() + dt_ms
        else:
            t_value = self._to_scalar_float(t_spike_ms, name='t_spike_ms')

        for _ in range(count):
            self._record_post_spike_at(float(t_value))
        return count

    def _get_post_history_times(self, t1_ms: float, t2_ms: float) -> list[float]:
        t1_lim = float(t1_ms + _STDP_EPS)
        t2_lim = float(t2_ms + _STDP_EPS)
        selected = []
        for t_post in self._post_hist_t:
            if t_post >= t1_lim and t_post < t2_lim:
                selected.append(t_post)
        return selected

    def _get_K_value(self, t_ms: float) -> float:
        # Return trace strictly before t, matching ArchivingNode::get_K_value.
        for idx in range(len(self._post_hist_t) - 1, -1, -1):
            t_post = self._post_hist_t[idx]
            if (t_ms - t_post) > _STDP_EPS:
                return self._post_hist_kminus[idx] * math.exp((t_post - t_ms) / self.tau_minus)
        return 0.0

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        super().init_state()
        self.Kplus = float(self._Kplus0)
        self.t_lastspike = float(self._t_lastspike0)
        self.clear_post_history()

    def get(self) -> dict:
        """Return current public parameters and mutable state."""
        params = super().get()
        params['tau_plus'] = float(self.tau_plus)
        params['tau_minus'] = float(self.tau_minus)
        params['lambda'] = float(self.lambda_)
        params['alpha'] = float(self.alpha)
        params['mu'] = float(self.mu)
        params['Kplus'] = float(self.Kplus)
        params['synapse_model'] = 'stdp_pl_synapse_hom'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        """Reject common-property assignments in connect-time synapse specs."""
        if syn_spec is None:
            return
        disallowed = ('tau_plus', 'lambda', 'alpha', 'mu')
        for key in disallowed:
            if key in syn_spec:
                raise ValueError(
                    f'{key} cannot be specified in connect-time synapse parameters '
                    'for stdp_pl_synapse_hom; set common properties on the model '
                    'itself (for example via CopyModel()/SetDefaults()).'
                )

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        tau_plus: ArrayLike | object = _UNSET,
        tau_minus: ArrayLike | object = _UNSET,
        lambda_: ArrayLike | object = _UNSET,
        alpha: ArrayLike | object = _UNSET,
        mu: ArrayLike | object = _UNSET,
        Kplus: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        """Set NEST-style public parameters and mutable state."""
        new_tau_plus = (
            self.tau_plus
            if tau_plus is _UNSET
            else self._to_scalar_time_ms(tau_plus, name='tau_plus')
        )
        self._validate_tau_plus(float(new_tau_plus))

        new_tau_minus = (
            self.tau_minus
            if tau_minus is _UNSET
            else self._to_scalar_time_ms(tau_minus, name='tau_minus')
        )
        new_lambda = (
            self.lambda_
            if lambda_ is _UNSET
            else self._to_scalar_float(lambda_, name='lambda')
        )
        new_alpha = self.alpha if alpha is _UNSET else self._to_scalar_float(alpha, name='alpha')
        new_mu = self.mu if mu is _UNSET else self._to_scalar_float(mu, name='mu')
        new_Kplus = self.Kplus if Kplus is _UNSET else self._to_scalar_float(Kplus, name='Kplus')

        super_kwargs = {}
        if weight is not _UNSET:
            super_kwargs['weight'] = self._normalize_scalar_weight(weight)
        if delay is not _UNSET:
            super_kwargs['delay'] = delay
        if receptor_type is not _UNSET:
            super_kwargs['receptor_type'] = receptor_type
        if post is not _UNSET:
            super_kwargs['post'] = post
        if super_kwargs:
            super().set(**super_kwargs)

        self.tau_plus = float(new_tau_plus)
        self.tau_minus = float(new_tau_minus)
        self.lambda_ = float(new_lambda)
        self.alpha = float(new_alpha)
        self.mu = float(new_mu)
        self.Kplus = float(new_Kplus)

        self._Kplus0 = float(self.Kplus)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        """Schedule one outgoing event with NEST ``stdp_pl_synapse_hom`` dynamics."""
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
        for t_post in self._get_post_history_times(t1, t2):
            minus_dt = self.t_lastspike - (t_post + dendritic_delay)
            assert minus_dt < (-1.0 * _STDP_EPS)
            kplus_term = self.Kplus * math.exp(minus_dt / self.tau_plus)
            self.weight = float(self._facilitate(float(self.weight), float(kplus_term)))

        # Depression due to current presynaptic spike.
        kminus_value = self._get_K_value(t_spike - dendritic_delay)
        self.weight = float(self._depress(float(self.weight), float(kminus_value)))

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.Kplus = float(self.Kplus * math.exp((self.t_lastspike - t_spike) / self.tau_plus) + 1.0)
        self.t_lastspike = float(t_spike)
        return True

    def update(
        self,
        pre_spike: ArrayLike = 0.0,
        *,
        post_spike: ArrayLike = 0.0,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> int:
        """Deliver due events, update post history, then process pre spikes."""
        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        post_count = self._to_non_negative_int_count(post_spike, name='post_spike')
        if post_count > 0:
            t_post = self._current_time_ms() + dt_ms
            for _ in range(post_count):
                self._record_post_spike_at(float(t_post))

        total_pre = self.sum_current_inputs(pre_spike)
        total_pre = self.sum_delta_inputs(total_pre)
        if self._is_nonzero(total_pre):
            self.send(total_pre, post=post, receptor_type=receptor_type)

        return delivered
