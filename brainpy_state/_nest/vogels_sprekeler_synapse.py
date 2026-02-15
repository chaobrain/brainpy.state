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
    'vogels_sprekeler_synapse',
]


class vogels_sprekeler_synapse:
    r"""NEST-compatible ``vogels_sprekeler_synapse`` connection model.

    Short description
    -----------------
    Symmetric STDP with constant presynaptic depression.

    Description
    -----------
    This class reproduces connection-level semantics of NEST
    ``models/vogels_sprekeler_synapse.{h,cpp}``.

    Plasticity follows the NEST send ordering exactly for one presynaptic spike
    at time :math:`t` (dendritic delay :math:`d`):

    1. Read postsynaptic history in ``(t_last - d, t - d]``.
    2. For each postsynaptic history spike :math:`t_j`:

       .. math::
          w \leftarrow \operatorname{facilitate}\!\left(
          w, K_+ \exp\left(\frac{t_{last} - (t_j + d)}{\tau}\right)\right)

    3. Apply facilitation with postsynaptic trace at ``t - d``:

       .. math::
          w \leftarrow \operatorname{facilitate}(w, K_-(t - d))

    4. Apply constant depression:

       .. math::
          w \leftarrow \operatorname{depress}(w)

    5. Emit spike event using updated ``weight``.
    6. Update presynaptic trace:

       .. math::
          K_+ \leftarrow K_+ \exp((t_{last}-t)/\tau) + 1

    7. Set ``t_last = t``.

    NEST ``facilitate_``/``depress_`` are sign-aware via ``Wmax``:

    .. math::
       \operatorname{facilitate}(w, k) =
       \operatorname{copysign}\left(\min(|w| + \eta k,\ |Wmax|), Wmax\right)

    .. math::
       \operatorname{depress}(w) =
       \operatorname{copysign}\left(\max(|w| - \alpha\eta,\ 0), Wmax\right)

    Parameters
    ----------
    weight : float, optional
        Synaptic weight. Default ``0.5``.
    delay : float, optional
        Dendritic delay in milliseconds used for history lookup.
        Default ``1.0``.
    delay_steps : int, optional
        Event delivery delay in integer simulation steps. Default ``1``.
    tau : float, optional
        STDP time constant in ms. Default ``20.0``.
    alpha : float, optional
        Constant depression factor. Default ``0.12``.
    eta : float, optional
        Learning rate. Default ``0.001``.
    Wmax : float, optional
        Signed maximum absolute weight. Default ``1.0``.
    Kplus : float, optional
        Presynaptic STDP trace. Must be non-negative. Default ``0.0``.
    t_last_spike_ms : float, optional
        Last presynaptic spike time in ms. Default ``0.0``.
    name : str, optional
        Optional model instance name.

    Target interface
    ----------------
    ``send()`` requires target methods:

    - ``get_history(t1, t2)`` returning postsynaptic spike history entries in
      ``(t1, t2]``.
    - ``get_K_value(t)`` (or ``get_k_value(t)``) returning postsynaptic STDP
      trace at time ``t``.

    Notes
    -----
    - As in NEST, precise sub-step timestamps are ignored for plasticity.
    - NEST sign constraint is reproduced: if ``weight != 0``, ``weight`` and
      ``Wmax`` must have the same sign.
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    IS_PRIMARY = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True
    SUPPORTS_WFR = True

    def __init__(
        self,
        weight: ArrayLike = 0.5,
        delay: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        tau: ArrayLike = 20.0,
        alpha: ArrayLike = 0.12,
        eta: ArrayLike = 0.001,
        Wmax: ArrayLike = 1.0,
        Kplus: ArrayLike = 0.0,
        t_last_spike_ms: ArrayLike = 0.0,
        name: str | None = None,
    ):
        self.name = name

        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay = self._validate_positive_delay(delay)
        self.delay_steps = self._validate_delay_steps(delay_steps)
        self.tau = self._validate_positive_tau(tau)
        self.alpha = self._to_float_scalar(alpha, name='alpha')
        self.eta = self._to_float_scalar(eta, name='eta')
        self.Wmax = self._to_float_scalar(Wmax, name='Wmax')
        self.Kplus = self._to_float_scalar(Kplus, name='Kplus')
        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

        self._check_constraints()

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
            'tau': float(self.tau),
            'alpha': float(self.alpha),
            'eta': float(self.eta),
            'Wmax': float(self.Wmax),
            'Kplus': float(self.Kplus),
            't_last_spike_ms': float(self.t_last_spike_ms),
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

        if 'weight' in updates:
            self.weight = self._to_float_scalar(updates['weight'], name='weight')
        if 'delay' in updates:
            self.delay = self._validate_positive_delay(updates['delay'])
        if 'delay_steps' in updates:
            self.delay_steps = self._validate_delay_steps(updates['delay_steps'])
        if 'tau' in updates:
            self.tau = self._validate_positive_tau(updates['tau'])
        if 'alpha' in updates:
            self.alpha = self._to_float_scalar(updates['alpha'], name='alpha')
        if 'eta' in updates:
            self.eta = self._to_float_scalar(updates['eta'], name='eta')
        if 'Wmax' in updates:
            self.Wmax = self._to_float_scalar(updates['Wmax'], name='Wmax')
        if 'Kplus' in updates:
            self.Kplus = self._to_float_scalar(updates['Kplus'], name='Kplus')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')

        self._check_constraints()

    def get(self, key: str = 'status'):
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for vogels_sprekeler_synapse.get().')

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
            self.weight = self._facilitate(self.weight, self.Kplus * math.exp(minus_dt / self.tau))

        kminus = self._get_k_value(target, t_spike - dendritic_delay)
        self.weight = self._facilitate(self.weight, kminus)
        self.weight = self._depress(self.weight)

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

        self.Kplus = self.Kplus * math.exp((self.t_last_spike_ms - t_spike) / self.tau) + 1.0
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
        new_w = abs(w) + self.eta * kplus
        return math.copysign(min(new_w, abs(self.Wmax)), self.Wmax)

    def _depress(self, w: float) -> float:
        new_w = abs(w) - self.alpha * self.eta
        return math.copysign(max(new_w, 0.0), self.Wmax)

    def _check_constraints(self):
        if self.Kplus < 0.0:
            raise ValueError('State Kplus must be positive.')
        if self.weight != 0.0 and (math.copysign(1.0, self.weight) != math.copysign(1.0, self.Wmax)):
            raise ValueError('Weight and Wmax must have same sign.')

    @staticmethod
    def _get_history(target: Any, t1: float, t2: float):
        if hasattr(target, 'get_history'):
            return target.get_history(float(t1), float(t2))
        raise AttributeError(
            'Target must provide get_history(t1, t2) for vogels_sprekeler_synapse.'
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
            'Target must provide get_K_value(t) or get_k_value(t) for vogels_sprekeler_synapse.'
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
    def _validate_positive_tau(cls, value: ArrayLike) -> float:
        tau = cls._to_float_scalar(value, name='tau')
        if tau <= 0.0:
            raise ValueError('tau must be > 0.')
        return tau

    @classmethod
    def _validate_multiplicity(cls, value: ArrayLike) -> float:
        m = cls._to_float_scalar(value, name='multiplicity')
        if m < 0.0:
            raise ValueError('multiplicity must be >= 0.')
        return m
