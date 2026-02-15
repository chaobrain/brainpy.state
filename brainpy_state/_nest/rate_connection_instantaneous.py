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

from typing import Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

__all__ = [
    'rate_connection_instantaneous',
]


class rate_connection_instantaneous:
    r"""NEST-compatible ``rate_connection_instantaneous`` connection model.

    Short description
    -----------------
    Synapse model for instantaneous rate connections.

    Description
    -----------
    This class implements connection-level semantics of NEST
    ``models/rate_connection_instantaneous.{h,cpp}``.

    In NEST, ``rate_connection_instantaneous``:

    - stores a scalar ``weight``,
    - transmits ``InstantaneousRateConnectionEvent``,
    - supports waveform relaxation (WFR),
    - does not support configurable delay; setting ``delay`` raises
      ``BadProperty``.

    This implementation follows those semantics in step-based form.

    Transmits
    ---------
    Instantaneous rate secondary-event payloads represented as dictionaries
    carrying ``coeffarray`` and ``weight``.

    Update ordering and event mapping
    ---------------------------------
    In NEST ``rate_neuron_*_impl.h``, targets handle instantaneous events by
    iterating coefficient index :math:`i` and adding weighted contribution to
    instantaneous input buffer slot :math:`i`.

    The helper :meth:`coeffarray_to_step_events` reproduces this ordering by
    mapping ``coeffarray[i]`` to a per-step event with
    ``delay_steps = first_delay_steps + i``.

    Notes
    -----
    - ``set_status(delay=...)`` and ``set_delay(...)`` raise with NEST-matching
      message:
      ``"rate_connection_instantaneous has no delay. Please use rate_connection_delayed."``
    - A compatibility ``delay`` field is exposed in ``get_status`` but ignored
      by the model, matching NEST behavior where delay can be queried but not
      set.

    References
    ----------
    .. [1] NEST source: ``models/rate_connection_instantaneous.h`` and
           ``models/rate_connection_instantaneous.cpp``.
    .. [2] NEST receiver handling:
           ``models/rate_neuron_ipn_impl.h`` and
           ``models/rate_neuron_opn_impl.h``.
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = False
    SUPPORTS_WFR = True

    _DELAY_ERROR = (
        'rate_connection_instantaneous has no delay. Please use '
        'rate_connection_delayed.'
    )

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        name: str | None = None,
    ):
        self.name = name
        self.weight = self._to_float_scalar(weight, name='weight')
        # Kept for status parity with NEST; not used in transmission logic.
        self.delay = 1

    @property
    def properties(self) -> dict[str, Any]:
        return {
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            'weight': float(self.weight),
            'delay': int(self.delay),
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        # Match NEST behavior: reject delay updates before applying any weight.
        if 'delay' in updates or 'delay_steps' in updates:
            raise ValueError(self._DELAY_ERROR)

        if 'weight' in updates:
            self.set_weight(updates['weight'])

    def get(self, key: str = 'status'):
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for rate_connection_instantaneous.get().')

    def set_weight(self, weight: ArrayLike):
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, _):
        raise ValueError(self._DELAY_ERROR)

    def set_delay_steps(self, _):
        raise ValueError(self._DELAY_ERROR)

    def prepare_secondary_event(self, coeffarray: ArrayLike) -> dict[str, Any]:
        """Create an instantaneous secondary-event payload."""
        return {
            'coeffarray': self._to_coeff_array(coeffarray),
            'weight': float(self.weight),
        }

    def to_rate_event(
        self,
        rate: ArrayLike,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike = 0,
    ) -> dict[str, Any]:
        """Create an instantaneous rate-event payload for local step APIs."""
        d = self._to_int_scalar(delay_steps, name='delay_steps')
        if d != 0:
            raise ValueError('delay_steps for rate_connection_instantaneous must be 0.')
        return {
            'rate': self._to_rate_value(rate),
            'weight': float(self.weight),
            'delay_steps': 0,
            'multiplicity': self._to_float_scalar(multiplicity, name='multiplicity'),
        }

    def coeffarray_to_step_events(
        self,
        coeffarray: ArrayLike,
        first_delay_steps: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
    ) -> list[dict[str, Any]]:
        """Map lag-indexed coeffarray to per-step events.

        For each coefficient ``coeffarray[i]``, this returns an event with
        ``delay_steps = first_delay_steps + i``.
        """
        coeff = self._to_coeff_array(coeffarray)
        d0 = self._to_int_scalar(first_delay_steps, name='first_delay_steps')
        if d0 < 0:
            raise ValueError('first_delay_steps must be >= 0.')
        mult = self._to_float_scalar(multiplicity, name='multiplicity')

        events = []
        for i, c in enumerate(coeff):
            events.append(
                {
                    'rate': float(c),
                    'weight': float(self.weight),
                    'delay_steps': int(d0 + i),
                    'multiplicity': float(mult),
                }
            )
        return events

    @staticmethod
    def _to_coeff_array(value: ArrayLike) -> np.ndarray:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError('Coefficient array must not be empty.')
        return arr

    @staticmethod
    def _to_rate_value(value: ArrayLike):
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return arr

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr[0])

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
