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
    'diffusion_connection',
]


class diffusion_connection:
    r"""NEST-compatible ``diffusion_connection`` connection model.

    Short description
    -----------------
    Synapse model for instantaneous diffusion coupling between
    ``siegert_neuron`` populations.

    Description
    -----------
    This class implements connection-level semantics of NEST
    ``models/diffusion_connection.{h,cpp}``.

    ``diffusion_connection`` is equivalent to an instantaneous rate connection
    with two per-connection factors replacing the standard weight:

    - ``drift_factor``: scales presynaptic rate contribution to drift input
      :math:`\mu`.
    - ``diffusion_factor``: scales presynaptic rate contribution to diffusion
      input :math:`\sigma^2`.

    As in NEST, this connection:

    - supports waveform relaxation (WFR) secondary events,
    - does not support configurable delay,
    - does not allow ``weight`` to be set.

    Update ordering (matching NEST)
    --------------------------------
    1. Sender emits a diffusion secondary event with coefficient array
       ``coeffarray`` (typically sender rate values).
    2. Connection copies its factors into the event:
       ``drift_factor``, ``diffusion_factor``.
    3. Target ``siegert_neuron`` accumulates per-lag inputs:

       .. math::

          \mu_i \leftarrow \mu_i + \text{drift\_factor}\cdot c_i,
          \qquad
          \sigma^2_i \leftarrow \sigma^2_i + \text{diffusion\_factor}\cdot c_i

       for each coefficient :math:`c_i` in ``coeffarray``.

    Parameters
    ----------
    drift_factor : float, optional
        Drift scaling factor. Default ``1.0``.
    diffusion_factor : float, optional
        Diffusion scaling factor. Default ``1.0``.
    name : str, optional
        Optional model instance name.

    Notes
    -----
    - ``weight`` and ``delay`` can be queried in status for compatibility, but
      setting either raises an error, matching NEST ``BadProperty`` behavior.
    - The NEST error message for weight contains the original typo
      ``specifiy``; this implementation keeps it intentionally for parity.

    References
    ----------
    .. [1] NEST source: ``models/diffusion_connection.h`` and
           ``models/diffusion_connection.cpp``.
    .. [2] NEST target-side accumulation:
           ``models/siegert_neuron.cpp`` (``handle(DiffusionConnectionEvent&)``).
    """

    __module__ = 'brainpy.state'

    SUPPORTS_WFR = True
    HAS_DELAY = False

    _WEIGHT_ERROR = (
        'Please use the parameters drift_factor and diffusion_factor to specifiy the weights.'
    )
    _DELAY_ERROR = 'diffusion_connection has no delay.'

    def __init__(
        self,
        drift_factor: ArrayLike = 1.0,
        diffusion_factor: ArrayLike = 1.0,
        name: str | None = None,
    ):
        self.name = name
        # Keep a status ``weight`` field for parity with NEST model status.
        self.weight = 1.0
        self.drift_factor = self._to_float_scalar(drift_factor, name='drift_factor')
        self.diffusion_factor = self._to_float_scalar(diffusion_factor, name='diffusion_factor')

    @property
    def properties(self) -> dict[str, Any]:
        return {
            'supports_wfr': self.SUPPORTS_WFR,
            'has_delay': self.HAS_DELAY,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            'weight': float(self.weight),
            'delay': None,
            'drift_factor': float(self.drift_factor),
            'diffusion_factor': float(self.diffusion_factor),
            'supports_wfr': self.SUPPORTS_WFR,
            'has_delay': self.HAS_DELAY,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        if 'delay' in updates:
            raise ValueError(self._DELAY_ERROR)
        if 'weight' in updates:
            raise ValueError(self._WEIGHT_ERROR)
        if 'drift_factor' in updates:
            self.set_drift_factor(updates['drift_factor'])
        if 'diffusion_factor' in updates:
            self.set_diffusion_factor(updates['diffusion_factor'])

    def get(self, key: str = 'status'):
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for diffusion_connection.get().')

    def set_drift_factor(self, drift_factor: ArrayLike):
        self.drift_factor = self._to_float_scalar(drift_factor, name='drift_factor')

    def set_diffusion_factor(self, diffusion_factor: ArrayLike):
        self.diffusion_factor = self._to_float_scalar(diffusion_factor, name='diffusion_factor')

    def set_weight(self, _):
        raise ValueError(self._WEIGHT_ERROR)

    def set_delay(self, _):
        raise ValueError(self._DELAY_ERROR)

    def prepare_secondary_event(self, coeffarray: ArrayLike) -> dict[str, Any]:
        coeff_np = self._to_coeff_array(coeffarray)
        return {
            'coeffarray': coeff_np,
            'drift_factor': float(self.drift_factor),
            'diffusion_factor': float(self.diffusion_factor),
        }

    def project_coeffarray(self, coeffarray: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Project diffusion event coefficients to drift/diffusion lag inputs."""
        coeff_np = self._to_coeff_array(coeffarray)
        return self.drift_factor * coeff_np, self.diffusion_factor * coeff_np

    def to_siegert_event(
        self,
        coeff: ArrayLike,
        delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
    ) -> dict[str, Any]:
        """Create a single-step event payload consumable by ``siegert_neuron``."""
        return {
            'coeff': self._to_float_scalar(coeff, name='coeff'),
            'drift_factor': float(self.drift_factor),
            'diffusion_factor': float(self.diffusion_factor),
            'delay_steps': self._to_int_scalar(delay_steps, name='delay_steps'),
            'multiplicity': self._to_float_scalar(multiplicity, name='multiplicity'),
        }

    def coeffarray_to_step_events(
        self,
        coeffarray: ArrayLike,
        first_delay_steps: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
    ) -> list[dict[str, Any]]:
        """Map a NEST lag-indexed coeffarray to per-step delayed events.

        For a coefficient array ``c[i]``, this returns events with
        ``delay_steps = first_delay_steps + i`` and ``coeff = c[i]``.
        """
        coeff_np = self._to_coeff_array(coeffarray)
        d0 = self._to_int_scalar(first_delay_steps, name='first_delay_steps')
        mult = self._to_float_scalar(multiplicity, name='multiplicity')
        if d0 < 0:
            raise ValueError('first_delay_steps must be >= 0.')

        events = []
        for i, c in enumerate(coeff_np):
            events.append(
                {
                    'coeff': float(c),
                    'drift_factor': float(self.drift_factor),
                    'diffusion_factor': float(self.diffusion_factor),
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
        return int(arr[0])
