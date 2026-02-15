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
    'rate_connection_delayed',
]


class rate_connection_delayed:
    r"""NEST-compatible ``rate_connection_delayed`` connection model.

    Short description
    -----------------
    Synapse model for delayed rate connections.

    Description
    -----------
    This class implements connection-level semantics of NEST
    ``models/rate_connection_delayed.{h,cpp}``.

    In NEST, ``rate_connection_delayed``:

    - stores scalar ``weight`` and ``delay``,
    - transmits ``DelayedRateConnectionEvent``,
    - sends delayed secondary events (no instantaneous/WFR-only behavior).

    This implementation follows those semantics in step-based form and exposes:

    - ``weight``: connection gain.
    - ``delay_steps``: integer delay in simulation steps (must be ``>= 1``).

    Transmits
    ---------
    Delayed rate secondary-event payloads, represented here as dictionaries
    carrying ``coeffarray``/``rate``, ``weight``, and ``delay_steps``.

    Update ordering and event mapping
    ---------------------------------
    NEST target-side handling for delayed rate events uses:

    .. math::

       \text{buffer\_offset} = \text{event.delay\_steps} - \text{min\_delay\_steps},
       \qquad
       \text{target\_slot} = \text{buffer\_offset} + i

    for coefficient index :math:`i` in a lag-indexed ``coeffarray``.

    The helper :meth:`coeffarray_to_step_events` reproduces this ordering:
    each coefficient ``coeffarray[i]`` is mapped to a per-step event with
    ``delay_steps = (delay_steps - min_delay_steps) + i``.

    Notes
    -----
    - NEST stores delay in time units (ms). This model uses discrete
      ``delay_steps`` to match this package's event APIs.
    - ``delay`` is accepted as an alias of ``delay_steps`` in ``set_status``.

    References
    ----------
    .. [1] NEST source: ``models/rate_connection_delayed.h`` and
           ``models/rate_connection_delayed.cpp``.
    .. [2] NEST delayed-rate receiver handling:
           ``models/rate_neuron_ipn_impl.h`` and
           ``models/rate_neuron_opn_impl.h``.
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    SUPPORTS_WFR = False

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        name: str | None = None,
    ):
        self.name = name
        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay_steps = self._validate_delay_steps(delay_steps)

    @property
    def properties(self) -> dict[str, Any]:
        return {
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            'weight': float(self.weight),
            'delay_steps': int(self.delay_steps),
            'delay': int(self.delay_steps),
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
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

    def get(self, key: str = 'status'):
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for rate_connection_delayed.get().')

    def set_weight(self, weight: ArrayLike):
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, delay: ArrayLike):
        self.delay_steps = self._validate_delay_steps(delay, name='delay')

    def set_delay_steps(self, delay_steps: ArrayLike):
        self.delay_steps = self._validate_delay_steps(delay_steps, name='delay_steps')

    def prepare_secondary_event(self, coeffarray: ArrayLike) -> dict[str, Any]:
        """Create a delayed secondary-event payload."""
        return {
            'coeffarray': self._to_coeff_array(coeffarray),
            'weight': float(self.weight),
            'delay_steps': int(self.delay_steps),
        }

    def to_rate_event(
        self,
        rate: ArrayLike,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Create a delayed-rate event payload for local step-based APIs."""
        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        return {
            'rate': self._to_rate_value(rate),
            'weight': float(self.weight),
            'delay_steps': int(d),
            'multiplicity': self._to_float_scalar(multiplicity, name='multiplicity'),
        }

    def coeffarray_to_step_events(
        self,
        coeffarray: ArrayLike,
        min_delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
    ) -> list[dict[str, Any]]:
        """Map lag-indexed coeffarray to per-step delayed-rate events.

        For each coefficient ``coeffarray[i]``, this method returns an event
        with

        ``delay_steps = (self.delay_steps - min_delay_steps) + i``.

        This matches NEST delayed-rate receiver logic in
        ``rate_neuron_*_impl.h``.
        """
        coeff = self._to_coeff_array(coeffarray)
        min_delay = self._validate_delay_steps(min_delay_steps, name='min_delay_steps')
        base_delay = int(self.delay_steps - min_delay)
        if base_delay < 0:
            raise ValueError('delay_steps must be >= min_delay_steps.')
        mult = self._to_float_scalar(multiplicity, name='multiplicity')

        events = []
        for i, c in enumerate(coeff):
            events.append(
                {
                    'rate': float(c),
                    'weight': float(self.weight),
                    'delay_steps': int(base_delay + i),
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

    def _validate_delay_steps(self, delay_steps: ArrayLike, name: str = 'delay_steps') -> int:
        d = self._to_int_scalar(delay_steps, name=name)
        if d < 1:
            raise ValueError(f'{name} must be >= 1.')
        return d
