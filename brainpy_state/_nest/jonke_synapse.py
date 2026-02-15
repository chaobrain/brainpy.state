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
from typing import Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

__all__ = [
    'jonke_synapse',
]


class jonke_synapse:
    r"""NEST-compatible ``jonke_synapse`` connection model.

    Short description
    -----------------
    STDP synapse with additive offsets and exponential weight factors.

    Description
    -----------
    This class reproduces connection-level semantics of NEST
    ``models/jonke_synapse.{h,cpp}``.

    For presynaptic spike time :math:`t`, postsynaptic spike trace values
    :math:`K_+(t), K_-(t)`, and current weight :math:`w`, the update kernels are

    .. math::

       K_+(w) &= \exp(\mu_+ w), \\
       K_-(w) &= \exp(\mu_- w),

    .. math::

       \Delta w_+ &= \lambda\left(K_+(w)\,F_+ - \beta\right), \\
       \Delta w_- &= \lambda\left(-\alpha K_-(w)\,F_- - \beta\right),

    with hard bounds ``[0, Wmax]`` applied after each update.

    NEST send-ordering in ``jonke_synapse::send(...)`` is:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\; t-d]`.
    2. For each postsynaptic history spike :math:`t_j` in this interval:

       .. math::
          w \leftarrow \mathrm{facilitate}\!\left(
          w,\; K_+\exp((t_{\mathrm{last}}-(t_j+d))/\tau_+)\right)

    3. Apply depression with postsynaptic ``Kminus`` at :math:`t-d`:

       .. math::
          w \leftarrow \mathrm{depress}(w,\; K_-(t-d))

    4. Emit spike event using updated ``weight``.
    5. Update presynaptic trace:

       .. math::
          K_+ \leftarrow K_+\exp((t_{\mathrm{last}}-t)/\tau_+) + 1

    6. Set :math:`t_{\mathrm{last}} \leftarrow t`.

    Parameters
    ----------
    weight : float, optional
        Synaptic weight. Default ``1.0``.
    delay : float, optional
        Dendritic delay in milliseconds used for history lookups.
        Default ``1.0``.
    delay_steps : int, optional
        Event delivery delay in integer simulation steps. Default ``1``.
    Kplus : float, optional
        Presynaptic trace value. Must be non-negative. Default ``0.0``.
    t_last_spike_ms : float, optional
        Last presynaptic spike time in ms. Default ``0.0``.
    alpha : float, optional
        Depression scaling constant. Default ``1.0``.
    beta : float, optional
        Additive negative offset applied to both update branches.
        Default ``0.0``.
    lambda_ : float, optional
        Learning step size (:math:`\lambda`). Default ``0.01``.
    mu_plus : float, optional
        Weight dependence of facilitation branch. Default ``0.0``.
    mu_minus : float, optional
        Weight dependence of depression branch. Default ``0.0``.
    tau_plus : float, optional
        Potentiation time constant in ms. Default ``20.0``.
    Wmax : float, optional
        Maximum allowed weight. Default ``100.0``.
    name : str, optional
        Optional model instance name.

    Target interface
    ----------------
    ``send()`` requires the target object to expose:

    - ``get_history(t1, t2)`` returning postsynaptic spike history entries in
      ``(t1, t2]``. Each entry may be:
      attribute object with ``t_`` or ``t``, mapping with ``t_``/``t``,
      or tuple/list where the first item is time.
    - ``get_K_value(t)`` (or ``get_k_value(t)``) returning postsynaptic
      ``Kminus`` value at time ``t``.

    Notes
    -----
    - As in NEST, the precise (sub-grid) timestamp component is ignored when
      computing plasticity updates.
    - ``Kplus`` non-negativity check follows NEST ``set_status`` behavior.

    References
    ----------
    .. [1] NEST source: ``models/jonke_synapse.h`` and
           ``models/jonke_synapse.cpp``.
    .. [2] Jonke Z et al. (2017). Feedback inhibition shapes emergent
           computational properties of cortical microcircuit motifs.
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    IS_PRIMARY = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True
    SUPPORTS_WFR = False

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        Kplus: ArrayLike = 0.0,
        t_last_spike_ms: ArrayLike = 0.0,
        alpha: ArrayLike = 1.0,
        beta: ArrayLike = 0.0,
        lambda_: ArrayLike = 0.01,
        mu_plus: ArrayLike = 0.0,
        mu_minus: ArrayLike = 0.0,
        tau_plus: ArrayLike = 20.0,
        Wmax: ArrayLike = 100.0,
        name: str | None = None,
    ):
        self.name = name

        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay = self._validate_positive_delay(delay)
        self.delay_steps = self._validate_delay_steps(delay_steps)

        self.Kplus = self._to_float_scalar(Kplus, name='Kplus')
        if self.Kplus < 0.0:
            raise ValueError('Kplus must be non-negative.')

        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

        self.alpha = self._to_float_scalar(alpha, name='alpha')
        self.beta = self._to_float_scalar(beta, name='beta')
        self.lambda_ = self._to_float_scalar(lambda_, name='lambda_')
        self.mu_plus = self._to_float_scalar(mu_plus, name='mu_plus')
        self.mu_minus = self._to_float_scalar(mu_minus, name='mu_minus')
        self.tau_plus = self._to_float_scalar(tau_plus, name='tau_plus')
        self.Wmax = self._to_float_scalar(Wmax, name='Wmax')

    @property
    def properties(self) -> dict[str, Any]:
        return {
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            'weight': float(self.weight),
            'delay': float(self.delay),
            'delay_steps': int(self.delay_steps),
            'Kplus': float(self.Kplus),
            't_last_spike_ms': float(self.t_last_spike_ms),
            'alpha': float(self.alpha),
            'beta': float(self.beta),
            'lambda': float(self.lambda_),
            'mu_plus': float(self.mu_plus),
            'mu_minus': float(self.mu_minus),
            'tau_plus': float(self.tau_plus),
            'Wmax': float(self.Wmax),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        if 'lambda' in updates and 'lambda_' in updates:
            lv = self._to_float_scalar(updates['lambda'], name='lambda')
            lvv = self._to_float_scalar(updates['lambda_'], name='lambda_')
            if lv != lvv:
                raise ValueError('lambda and lambda_ must be identical when both are provided.')

        if 'weight' in updates:
            self.weight = self._to_float_scalar(updates['weight'], name='weight')
        if 'delay' in updates:
            self.delay = self._validate_positive_delay(updates['delay'])
        if 'delay_steps' in updates:
            self.delay_steps = self._validate_delay_steps(updates['delay_steps'])
        if 'Kplus' in updates:
            self.Kplus = self._to_float_scalar(updates['Kplus'], name='Kplus')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')
        if 'alpha' in updates:
            self.alpha = self._to_float_scalar(updates['alpha'], name='alpha')
        if 'beta' in updates:
            self.beta = self._to_float_scalar(updates['beta'], name='beta')
        if 'lambda' in updates:
            self.lambda_ = self._to_float_scalar(updates['lambda'], name='lambda')
        if 'lambda_' in updates:
            self.lambda_ = self._to_float_scalar(updates['lambda_'], name='lambda_')
        if 'mu_plus' in updates:
            self.mu_plus = self._to_float_scalar(updates['mu_plus'], name='mu_plus')
        if 'mu_minus' in updates:
            self.mu_minus = self._to_float_scalar(updates['mu_minus'], name='mu_minus')
        if 'tau_plus' in updates:
            self.tau_plus = self._to_float_scalar(updates['tau_plus'], name='tau_plus')
        if 'Wmax' in updates:
            self.Wmax = self._to_float_scalar(updates['Wmax'], name='Wmax')

        if self.Kplus < 0.0:
            raise ValueError('Kplus must be non-negative.')

    def get(self, key: str = 'status'):
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for jonke_synapse.get().')

    def set_weight(self, weight: ArrayLike):
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, delay: ArrayLike):
        self.delay = self._validate_positive_delay(delay)

    def set_delay_steps(self, delay_steps: ArrayLike):
        self.delay_steps = self._validate_delay_steps(delay_steps)

    def send(
        self,
        t_spike_ms: ArrayLike,
        target: Any,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay: ArrayLike | None = None,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Process one presynaptic spike and return emitted SpikeEvent payload."""
        t_spike = self._to_float_scalar(t_spike_ms, name='t_spike_ms')
        dendritic_delay = self.delay if delay is None else self._validate_positive_delay(delay)
        event_delay_steps = (
            self.delay_steps
            if delay_steps is None
            else self._validate_delay_steps(delay_steps)
        )

        history_entries = self._get_history(
            target,
            self.t_last_spike_ms - dendritic_delay,
            t_spike - dendritic_delay,
        )

        for entry in history_entries:
            t_hist = self._extract_history_time(entry)
            minus_dt = self.t_last_spike_ms - (t_hist + dendritic_delay)
            kplus_t = self.Kplus * math.exp(minus_dt / self.tau_plus)
            self.weight = self._facilitate(self.weight, kplus_t)

        kminus = self._get_k_value(target, t_spike - dendritic_delay)
        self.weight = self._depress(self.weight, kminus)

        event = {
            'weight': float(self.weight),
            'delay': float(dendritic_delay),
            'delay_steps': int(event_delay_steps),
            'receptor_type': self._to_int_scalar(receptor_type, name='receptor_type'),
            'multiplicity': self._validate_multiplicity(multiplicity),
            't_spike_ms': float(t_spike),
            'Kminus': float(kminus),
            'Kplus_pre': float(self.Kplus),
        }

        self.Kplus = self.Kplus * math.exp((self.t_last_spike_ms - t_spike) / self.tau_plus) + 1.0
        self.t_last_spike_ms = t_spike

        event['Kplus_post'] = float(self.Kplus)
        return event

    def to_spike_event(
        self,
        t_spike_ms: ArrayLike,
        target: Any,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay: ArrayLike | None = None,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        return self.send(
            t_spike_ms=t_spike_ms,
            target=target,
            receptor_type=receptor_type,
            multiplicity=multiplicity,
            delay=delay,
            delay_steps=delay_steps,
        )

    def simulate_pre_spike_train(
        self,
        pre_spike_times_ms: ArrayLike,
        target: Any,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay: ArrayLike | None = None,
        delay_steps: ArrayLike | None = None,
    ) -> list[dict[str, Any]]:
        times = np.asarray(u.math.asarray(pre_spike_times_ms), dtype=np.float64).reshape(-1)
        events = []
        for t in times:
            events.append(
                self.send(
                    t_spike_ms=float(t),
                    target=target,
                    receptor_type=receptor_type,
                    multiplicity=multiplicity,
                    delay=delay,
                    delay_steps=delay_steps,
                )
            )
        return events

    def _facilitate(self, w: float, kplus: float) -> float:
        if self.lambda_ == 0.0:
            return w
        k_w = math.exp(self.mu_plus * w)
        dw = self.lambda_ * (k_w * kplus - self.beta)
        new_w = w + dw
        return new_w if new_w < self.Wmax else self.Wmax

    def _depress(self, w: float, kminus: float) -> float:
        if self.lambda_ == 0.0:
            return w
        k_w = math.exp(self.mu_minus * w)
        dw = self.lambda_ * (-self.alpha * k_w * kminus - self.beta)
        new_w = w + dw
        return new_w if new_w > 0.0 else 0.0

    @staticmethod
    def _get_history(target: Any, t1: float, t2: float):
        if hasattr(target, 'get_history'):
            return target.get_history(float(t1), float(t2))
        raise AttributeError(
            'Target must provide get_history(t1, t2) for jonke_synapse.'
        )

    @staticmethod
    def _extract_history_time(entry: Any) -> float:
        if hasattr(entry, 't_'):
            return float(entry.t_)
        if hasattr(entry, 't'):
            return float(entry.t)
        if isinstance(entry, dict):
            if 't_' in entry:
                return float(entry['t_'])
            if 't' in entry:
                return float(entry['t'])
        if isinstance(entry, (tuple, list)) and len(entry) >= 1:
            return float(entry[0])
        raise TypeError(
            'History entry must expose a time as attribute t_/t, mapping key t_/t, or first tuple element.'
        )

    @staticmethod
    def _get_k_value(target: Any, t: float) -> float:
        if hasattr(target, 'get_K_value'):
            return float(target.get_K_value(float(t)))
        if hasattr(target, 'get_k_value'):
            return float(target.get_k_value(float(t)))
        raise AttributeError(
            'Target must provide get_K_value(t) or get_k_value(t) for jonke_synapse.'
        )

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr[0])
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        return v

    @staticmethod
    def _to_int_scalar(value: ArrayLike, name: str) -> int:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr[0])
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        iv = int(round(v))
        if abs(v - iv) > 1e-12:
            raise ValueError(f'{name} must be an integer value.')
        return iv

    @classmethod
    def _validate_positive_delay(cls, value: ArrayLike) -> float:
        d = cls._to_float_scalar(value, name='delay')
        if d <= 0.0:
            raise ValueError('delay must be > 0.')
        return d

    @classmethod
    def _validate_delay_steps(cls, value: ArrayLike) -> int:
        d = cls._to_int_scalar(value, name='delay_steps')
        if d < 1:
            raise ValueError('delay_steps must be >= 1.')
        return d

    @classmethod
    def _validate_multiplicity(cls, value: ArrayLike) -> float:
        m = cls._to_float_scalar(value, name='multiplicity')
        if m < 0.0:
            raise ValueError('multiplicity must be >= 0.')
        return m
