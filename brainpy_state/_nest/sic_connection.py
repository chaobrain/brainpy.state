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
    'sic_connection',
]


class sic_connection:
    r"""NEST-compatible ``sic_connection`` synapse model.

    Short description
    -----------------
    Synapse model for astrocyte-to-neuron slow inward current (SIC) coupling.

    Description
    -----------
    This class implements connection-level semantics of NEST
    ``models/sic_connection.{h,cpp}``.

    In NEST, ``sic_connection``:

    - stores scalar ``weight`` and ``delay``,
    - transmits ``SICEvent`` secondary events,
    - multiplies the emitted SIC coefficient stream by connection weight,
    - supports delayed delivery.

    Supported model pairing in NEST
    --------------------------------
    The source must emit ``SICEvent`` and the target must handle it.
    In standard NEST model sets this means:

    - source: ``astrocyte_lr_1994``
    - target: ``aeif_cond_alpha_astro``

    Methods :meth:`supports_connection` and :meth:`check_connection` mirror this
    validation at the model-name level.

    Delay/index mapping to target SIC buffer
    ----------------------------------------
    NEST target-side SIC handling in
    ``aeif_cond_alpha_astro::handle(SICEvent&)`` uses:

    .. math::

       \text{offset} = \text{event.delay\_steps} - \text{min\_delay\_steps},
       \qquad
       \text{slot}_i = \text{offset} + i

    for coefficient index :math:`i` in ``coeffarray``.

    Local step-based APIs in this package consume SIC events with
    ``delay_steps`` interpreted as an offset ``delay_steps - 1``. Therefore, for
    an absolute NEST delay ``d`` and ``min_delay_steps = m``, this class maps to
    local delay:

    .. math::

       d_{\text{local}} = (d - m) + 1.

    Helper methods :meth:`to_aeif_sic_event` and
    :meth:`coeffarray_to_step_events` apply this mapping.

    Parameters
    ----------
    weight : float, optional
        Synaptic weight multiplying SIC coefficients. Default ``1.0``.
    delay_steps : int, optional
        Absolute event delay in simulation steps (NEST-style). Must be
        ``>= 1``. Default ``1``.
    name : str, optional
        Optional model instance name.

    Notes
    -----
    - This class represents synapse/connection semantics only.
    - SIC coefficient generation (astrocyte dynamics) is handled by the source
      model (e.g., NEST ``astrocyte_lr_1994`` or an equivalent local source).

    References
    ----------
    .. [1] NEST source: ``models/sic_connection.h`` and
           ``models/sic_connection.cpp``.
    .. [2] NEST receiver logic:
           ``models/aeif_cond_alpha_astro.cpp`` (``handle(SICEvent&)``).
    .. [3] NEST tests:
           ``testsuite/pytests/test_sic_connection.py``.
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    SUPPORTS_WFR = False
    SUPPORTED_SOURCES = ('astrocyte_lr_1994',)
    SUPPORTED_TARGETS = ('aeif_cond_alpha_astro',)

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
            'supported_sources': self.SUPPORTED_SOURCES,
            'supported_targets': self.SUPPORTED_TARGETS,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            'weight': float(self.weight),
            'delay_steps': int(self.delay_steps),
            'delay': int(self.delay_steps),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
            'supported_sources': self.SUPPORTED_SOURCES,
            'supported_targets': self.SUPPORTED_TARGETS,
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
        raise KeyError(f'Unsupported key "{key}" for sic_connection.get().')

    def set_weight(self, weight: ArrayLike):
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, delay: ArrayLike):
        self.delay_steps = self._validate_delay_steps(delay, name='delay')

    def set_delay_steps(self, delay_steps: ArrayLike):
        self.delay_steps = self._validate_delay_steps(delay_steps, name='delay_steps')

    @classmethod
    def _model_name(cls, model: Any) -> str:
        if isinstance(model, str):
            return model
        if hasattr(model, '__name__'):
            return str(model.__name__)
        if hasattr(model, '__class__') and hasattr(model.__class__, '__name__'):
            return str(model.__class__.__name__)
        return str(model)

    @classmethod
    def supports_connection(cls, source_model: Any, target_model: Any) -> bool:
        src = cls._model_name(source_model)
        tgt = cls._model_name(target_model)
        return src in cls.SUPPORTED_SOURCES and tgt in cls.SUPPORTED_TARGETS

    @classmethod
    def check_connection(cls, source_model: Any, target_model: Any) -> bool:
        ok = cls.supports_connection(source_model, target_model)
        if not ok:
            src = cls._model_name(source_model)
            tgt = cls._model_name(target_model)
            raise ValueError(
                f'Unsupported sic_connection pair: source={src}, target={tgt}. '
                f'Expected source in {cls.SUPPORTED_SOURCES} and target in {cls.SUPPORTED_TARGETS}.'
            )
        return True

    def prepare_secondary_event(
        self,
        coeffarray: ArrayLike,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Create a NEST-style ``SICEvent`` payload."""
        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        return {
            'coeffarray': self._to_coeff_array(coeffarray),
            'weight': float(self.weight),
            'delay_steps': int(d),
        }

    def to_aeif_sic_event(
        self,
        coeffarray: ArrayLike,
        min_delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Create SIC event payload consumable by ``aeif_cond_alpha_astro.update``."""
        coeff = self._to_coeff_array(coeffarray)
        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        local_delay = self._to_local_delay_steps(d, min_delay_steps=min_delay_steps)
        mult = self._to_float_scalar(multiplicity, name='multiplicity')
        return {
            'coeffs': coeff,
            'weight': float(self.weight * mult),
            'delay_steps': int(local_delay),
            'multiplicity': 1.0,
        }

    def to_sic_event(
        self,
        coeff: ArrayLike,
        min_delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Create single-coefficient SIC event for local step-based APIs."""
        return self.to_aeif_sic_event(
            coeffarray=coeff,
            min_delay_steps=min_delay_steps,
            multiplicity=multiplicity,
            delay_steps=delay_steps,
        )

    def coeffarray_to_step_events(
        self,
        coeffarray: ArrayLike,
        min_delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> list[dict[str, Any]]:
        """Map lag-indexed SIC coefficients to one event per future step."""
        coeff = self._to_coeff_array(coeffarray)
        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        local_delay = self._to_local_delay_steps(d, min_delay_steps=min_delay_steps)
        mult = self._to_float_scalar(multiplicity, name='multiplicity')
        weight = float(self.weight * mult)

        events = []
        for i, c in enumerate(coeff):
            events.append(
                {
                    'coeffs': float(c),
                    'weight': weight,
                    'delay_steps': int(local_delay + i),
                    'multiplicity': 1.0,
                }
            )
        return events

    @classmethod
    def _to_local_delay_steps(
        cls,
        delay_steps: ArrayLike,
        min_delay_steps: ArrayLike = 1,
    ) -> int:
        delay = cls._validate_delay_steps(delay_steps, name='delay_steps')
        min_delay = cls._validate_delay_steps(min_delay_steps, name='min_delay_steps')
        if delay < min_delay:
            raise ValueError('delay_steps must be >= min_delay_steps.')
        return int(delay - min_delay + 1)

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
        v = float(arr[0])
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        vr = int(round(v))
        if abs(v - vr) > 1e-12:
            raise ValueError(f'{name} must be integer-valued.')
        return vr

    @classmethod
    def _validate_delay_steps(cls, delay_steps: ArrayLike, name: str = 'delay_steps') -> int:
        d = cls._to_int_scalar(delay_steps, name=name)
        if d < 1:
            raise ValueError(f'{name} must be >= 1.')
        return d
