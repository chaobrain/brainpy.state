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
    'gap_junction',
]


class gap_junction:
    r"""NEST-compatible ``gap_junction`` connection model.

    Short description
    -----------------
    Electrical coupling connection that transmits ``GapJunctionEvent`` data and
    contributes a current proportional to membrane-voltage differences.

    Description
    -----------
    This class implements the connection-level semantics of NEST
    ``models/gap_junction.{h,cpp}``.

    In NEST, a ``gap_junction`` connection:

    - stores a scalar ``weight``,
    - has no configurable delay (setting ``delay`` is an error),
    - requires symmetric connectivity,
    - supports waveform-relaxation (WFR) secondary-event coupling.

    During WFR updates, each target neuron accumulates weighted interpolation
    coefficients from incoming ``GapJunctionEvent`` objects:

    .. math::

       \Sigma g_{ij} \leftarrow \Sigma g_{ij} + w_{ij},
       \quad
       c_k \leftarrow c_k + w_{ij}\,a_k,

    where :math:`a_k` are source interpolation coefficients and :math:`c_k` are
    target-side summed coefficients.

    For interpolation order ``0``, ``1``, or ``3``, NEST evaluates the gap term
    in the target membrane RHS as:

    .. math::

       I_\mathrm{gap}
       =
       -\Sigma g_{ij}\,V
       + P_\mathrm{interp}(t),

    where :math:`P_\mathrm{interp}(t)` is the polynomial built from
    ``interpolation_coefficients`` at the current lag.

    Update ordering (matching NEST)
    --------------------------------
    1. Start WFR window: clear ``sumj_g_ij`` and summed interpolation
       coefficients.
    2. For each arriving secondary event: add weight and weighted coefficients.
    3. During ODE substeps: evaluate ``I_gap`` from current ``V``, ``lag``,
       interpolation order, and normalized substep time ``t``.
    4. End WFR window: reset runtime buffers for the next window.

    Parameters
    ----------
    weight : float, optional
        Gap-junction conductance weight. Default ``1.0``.
    name : str, optional
        Optional model instance name.

    Notes
    -----
    - Delay is intentionally unsupported, matching NEST:
      ``"gap_junction connection has no delay"``.
    - This class represents connection semantics only. Neuron-specific ODE
      dynamics remain in the neuron model.

    References
    ----------
    .. [1] NEST source: ``models/gap_junction.h`` and
           ``models/gap_junction.cpp``.
    .. [2] NEST gap-current RHS evaluation in
           ``models/hh_psc_alpha_gap.cpp`` and
           ``models/hh_cond_beta_gap_traub.cpp``.
    """

    __module__ = 'brainpy.state'

    REQUIRES_SYMMETRIC = True
    SUPPORTS_WFR = True
    SUPPORTED_WFR_INTERPOLATION_ORDERS = (0, 1, 3)

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        name: str | None = None,
    ):
        self.name = name
        self.weight = self._to_float_scalar(weight, name='weight')

        # Runtime state accumulated from incoming GapJunctionEvent payloads.
        self.sumj_g_ij: float = 0.0
        self.interpolation_coefficients = np.zeros((0,), dtype=np.float64)
        self._interpolation_order = 0

    @property
    def properties(self) -> dict[str, Any]:
        return {
            'requires_symmetric': self.REQUIRES_SYMMETRIC,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        # Keep "delay" present (as None) for API parity with model status.
        return {
            'weight': float(self.weight),
            'delay': None,
            'requires_symmetric': self.REQUIRES_SYMMETRIC,
            'supports_wfr': self.SUPPORTS_WFR,
            'supported_wfr_interpolation_orders': self.SUPPORTED_WFR_INTERPOLATION_ORDERS,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)
        if 'delay' in updates:
            raise ValueError('gap_junction connection has no delay')
        if 'weight' in updates:
            self.set_weight(updates['weight'])

    def get(self, key: str = 'status'):
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for gap_junction.get().')

    def set_weight(self, weight: ArrayLike):
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, _):
        raise ValueError('gap_junction connection has no delay')

    def begin_wfr_cycle(self, min_delay_steps: ArrayLike, interpolation_order: int = 0):
        min_delay_steps = self._to_int_scalar(min_delay_steps, name='min_delay_steps')
        if min_delay_steps <= 0:
            raise ValueError('min_delay_steps must be > 0.')

        interpolation_order = self._validate_interpolation_order(interpolation_order)
        coeff_len = min_delay_steps * (interpolation_order + 1)

        self._interpolation_order = interpolation_order
        self.sumj_g_ij = 0.0
        self.interpolation_coefficients = np.zeros((coeff_len,), dtype=np.float64)

    def reset_runtime_state(self):
        self.sumj_g_ij = 0.0
        if self.interpolation_coefficients.size > 0:
            self.interpolation_coefficients.fill(0.0)

    def prepare_secondary_event(self, coeffarray: ArrayLike) -> dict[str, Any]:
        coeff_np = self._to_coeff_array(coeffarray)
        return {
            'weight': float(self.weight),
            'coeffarray': coeff_np,
        }

    def handle_gap_event(self, coeffarray: ArrayLike, weight: ArrayLike | None = None):
        coeff_np = self._to_coeff_array(coeffarray)

        if self.interpolation_coefficients.size == 0:
            self.interpolation_coefficients = np.zeros_like(coeff_np, dtype=np.float64)
        if coeff_np.size != self.interpolation_coefficients.size:
            raise ValueError(
                f'Coefficient size mismatch: got {coeff_np.size}, expected {self.interpolation_coefficients.size}.'
            )

        w = self.weight if weight is None else self._to_float_scalar(weight, name='weight')
        self.sumj_g_ij += w

        # Matches NEST handle(GapJunctionEvent&): interpolation_coefficients += weight * coeffarray.
        self.interpolation_coefficients += w * coeff_np

    def evaluate_gap_current(
        self,
        V_m: ArrayLike,
        lag: int,
        t: ArrayLike = 0.0,
        interpolation_order: int | None = None,
    ):
        if self.interpolation_coefficients.size == 0:
            raise ValueError('No interpolation coefficients available. Call begin_wfr_cycle() first.')

        order = self._interpolation_order if interpolation_order is None else interpolation_order
        order = self._validate_interpolation_order(order)

        n_per_lag = order + 1
        if self.interpolation_coefficients.size % n_per_lag != 0:
            raise ValueError('Interpolation coefficient buffer size is incompatible with interpolation order.')

        lag = self._to_int_scalar(lag, name='lag')
        n_lags = self.interpolation_coefficients.size // n_per_lag
        if lag < 0 or lag >= n_lags:
            raise ValueError(f'lag {lag} is out of bounds for {n_lags} lag slots.')

        v = np.asarray(u.math.asarray(V_m), dtype=np.float64)
        t = np.asarray(u.math.asarray(t), dtype=np.float64)
        base = -self.sumj_g_ij * v

        if order == 0:
            return base + self.interpolation_coefficients[lag]
        if order == 1:
            i0 = lag * 2
            return base + self.interpolation_coefficients[i0] + self.interpolation_coefficients[i0 + 1] * t

        i0 = lag * 4
        t2 = t * t
        return (
            base
            + self.interpolation_coefficients[i0]
            + self.interpolation_coefficients[i0 + 1] * t
            + self.interpolation_coefficients[i0 + 2] * t2
            + self.interpolation_coefficients[i0 + 3] * t2 * t
        )

    @classmethod
    def _validate_interpolation_order(cls, order: int) -> int:
        order = cls._to_int_scalar(order, name='interpolation_order')
        if order not in cls.SUPPORTED_WFR_INTERPOLATION_ORDERS:
            raise ValueError('Interpolation order must be 0, 1, or 3.')
        return order

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
