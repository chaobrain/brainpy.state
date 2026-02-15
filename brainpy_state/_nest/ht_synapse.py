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
    'ht_synapse',
]


class ht_synapse:
    r"""NEST-compatible ``ht_synapse`` depression connection model.

    Short description
    -----------------
    Synapse model with vesicle-pool depression after Hill and Tononi (2005).

    Description
    -----------
    This class implements connection-level semantics of NEST
    ``models/ht_synapse.{h,cpp}``.

    Connection state consists of:

    - ``weight``: baseline synaptic weight.
    - ``tau_P`` [ms]: recovery time constant of vesicle pool.
    - ``delta_P``: fractional depletion per emitted spike.
    - ``P``: current pool size in ``[0, 1]``.
    - ``t_last_spike_ms``: last processed spike timestamp (ms), initialized to
      ``0.0`` as in NEST.

    Dynamics follow NEST exactly:

    .. math::

       \frac{dP}{dt} = \frac{1-P}{\tau_P},
       \quad
       P(T^+) = (1-\delta_P)P(T^-).

    For a spike at time :math:`t`, with previous spike time ``t_last``:

    1. **Recovery**

       .. math::
          P_{send} = 1 - (1 - P)\exp(-(t - t_{last})/\tau_P)

    2. **Emit spike event with effective weight**

       .. math::
          w_{eff} = w \cdot P_{send}

    3. **Deplete pool after send**

       .. math::
          P \leftarrow (1-\delta_P)P_{send}

    4. **Update last spike time**

       .. math::
          t_{last} \leftarrow t

    This ordering matters and matches NEST ``ht_synapse::send(...)``.

    Parameters
    ----------
    weight : float, optional
        Baseline synaptic weight. Default ``1.0``.
    delay_steps : int, optional
        Delivery delay in simulation steps. Must be ``>= 1``. Default ``1``.
    tau_P : float, optional
        Pool recovery time constant in milliseconds. Must be ``> 0``.
        Default ``500.0``.
    delta_P : float, optional
        Fractional pool depletion per spike, in ``[0, 1]``.
        Default ``0.125``.
    P : float, optional
        Initial pool size, in ``[0, 1]``. Default ``1.0``.
    name : str, optional
        Optional model instance name.

    Notes
    -----
    - NEST warning about precise spike timing applies: this model updates from
      event stamps on the simulation grid and ignores sub-grid offsets.
    - ``delay`` is accepted as alias of ``delay_steps`` in ``set_status``,
      consistent with other NEST-compatible connection classes in this package.

    References
    ----------
    .. [1] NEST source: ``models/ht_synapse.h`` and ``models/ht_synapse.cpp``.
    .. [2] Hill S, Tononi G (2005), Journal of Neurophysiology 93:1671-1698.
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    SUPPORTS_WFR = False
    IS_PRIMARY = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        tau_P: ArrayLike = 500.0,
        delta_P: ArrayLike = 0.125,
        P: ArrayLike = 1.0,
        name: str | None = None,
    ):
        self.name = name
        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay_steps = self._validate_delay_steps(delay_steps)
        self.tau_P = self._validate_tau_P(tau_P)
        self.delta_P = self._validate_fraction(delta_P, name='delta_P')
        self.P = self._validate_fraction(P, name='P')

        # NEST default initialization: t_lastspike_ = 0.0
        self.t_last_spike_ms = 0.0

    @property
    def properties(self) -> dict[str, Any]:
        return {
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            'weight': float(self.weight),
            'delay_steps': int(self.delay_steps),
            'delay': int(self.delay_steps),
            'tau_P': float(self.tau_P),
            'delta_P': float(self.delta_P),
            'P': float(self.P),
            't_last_spike_ms': float(self.t_last_spike_ms),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        if 'weight' in updates:
            self.set_weight(updates['weight'])

        has_delay = 'delay' in updates
        has_delay_steps = 'delay_steps' in updates
        if has_delay and has_delay_steps:
            d = self._to_int_scalar(updates['delay'], name='delay')
            ds = self._to_int_scalar(updates['delay_steps'], name='delay_steps')
            if d != ds:
                raise ValueError('delay and delay_steps must be identical when both are provided.')
            self.set_delay_steps(ds)
        elif has_delay_steps:
            self.set_delay_steps(updates['delay_steps'])
        elif has_delay:
            self.set_delay(updates['delay'])

        if 'tau_P' in updates:
            self.tau_P = self._validate_tau_P(updates['tau_P'])
        if 'delta_P' in updates:
            self.delta_P = self._validate_fraction(updates['delta_P'], name='delta_P')
        if 'P' in updates:
            self.P = self._validate_fraction(updates['P'], name='P')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')

    def get(self, key: str = 'status'):
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for ht_synapse.get().')

    def set_weight(self, weight: ArrayLike):
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, delay: ArrayLike):
        self.delay_steps = self._validate_delay_steps(delay, name='delay')

    def set_delay_steps(self, delay_steps: ArrayLike):
        self.delay_steps = self._validate_delay_steps(delay_steps, name='delay_steps')

    def reset_state(
        self,
        P: ArrayLike = 1.0,
        t_last_spike_ms: ArrayLike = 0.0,
    ):
        self.P = self._validate_fraction(P, name='P')
        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

    def recover_pool(self, t_spike_ms: ArrayLike) -> float:
        """Propagate pool state to ``t_spike_ms`` without depletion."""
        t = self._to_float_scalar(t_spike_ms, name='t_spike_ms')
        h = t - self.t_last_spike_ms
        self.P = 1.0 - (1.0 - self.P) * math.exp(-h / self.tau_P)
        return float(self.P)

    def send(
        self,
        t_spike_ms: ArrayLike,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Process one incoming spike and return emitted SpikeEvent payload.

        Ordering matches NEST exactly:
        recover pool -> emit weighted spike -> deplete pool -> update last time.
        """
        t = self._to_float_scalar(t_spike_ms, name='t_spike_ms')
        p_send = self.recover_pool(t)
        eff_weight = self.weight * p_send

        self.P *= (1.0 - self.delta_P)
        self.t_last_spike_ms = t

        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        return {
            'weight': float(eff_weight),
            'delay_steps': int(d),
            'delay': int(d),
            'receptor_type': self._to_int_scalar(receptor_type, name='receptor_type'),
            'multiplicity': self._validate_multiplicity(multiplicity),
            't_spike_ms': float(t),
            'P_send': float(p_send),
            'P_post': float(self.P),
        }

    def to_spike_event(
        self,
        t_spike_ms: ArrayLike,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Alias of :meth:`send` for event-style APIs."""
        return self.send(
            t_spike_ms=t_spike_ms,
            receptor_type=receptor_type,
            multiplicity=multiplicity,
            delay_steps=delay_steps,
        )

    def simulate_spike_train(
        self,
        spike_times_ms: ArrayLike,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> list[dict[str, Any]]:
        """Apply a sequence of spikes and return emitted events."""
        times = np.asarray(u.math.asarray(spike_times_ms), dtype=np.float64).reshape(-1)
        events = []
        for t in times:
            events.append(
                self.send(
                    t_spike_ms=float(t),
                    receptor_type=receptor_type,
                    multiplicity=multiplicity,
                    delay_steps=delay_steps,
                )
            )
        return events

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
        vr = int(round(v))
        if abs(v - vr) > 1e-12:
            raise ValueError(f'{name} must be integer-valued.')
        return vr

    def _validate_delay_steps(self, delay_steps: ArrayLike, name: str = 'delay_steps') -> int:
        d = self._to_int_scalar(delay_steps, name=name)
        if d < 1:
            raise ValueError(f'{name} must be >= 1.')
        return d

    def _validate_tau_P(self, tau_P: ArrayLike) -> float:
        v = self._to_float_scalar(tau_P, name='tau_P')
        if v <= 0.0:
            raise ValueError('tau_P > 0 required.')
        return v

    def _validate_fraction(self, value: ArrayLike, name: str) -> float:
        v = self._to_float_scalar(value, name=name)
        if v < 0.0 or v > 1.0:
            raise ValueError(f'0 <= {name} <= 1 required.')
        return v

    def _validate_multiplicity(self, multiplicity: ArrayLike) -> float:
        m = self._to_float_scalar(multiplicity, name='multiplicity')
        if m < 0.0:
            raise ValueError('multiplicity must be >= 0.')
        return m
