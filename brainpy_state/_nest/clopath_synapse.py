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
    'clopath_synapse',
]


class clopath_synapse:
    r"""NEST-compatible ``clopath_synapse`` connection model.

    Short description
    -----------------
    Voltage-based STDP synapse after Clopath.

    Description
    -----------
    This class implements connection-level semantics of NEST
    ``models/clopath_synapse.{h,cpp}``.

    In contrast to standard pair-based STDP, the weight update depends on
    postsynaptic voltage-derived traces archived by the target neuron
    (for example ``aeif_psc_delta_clopath`` in NEST).

    Connection state consists of:

    - ``weight``: current synaptic weight.
    - ``x_bar``: presynaptic trace value.
    - ``tau_x`` [ms]: presynaptic trace time constant.
    - ``Wmin``/``Wmax``: hard lower/upper bounds on ``weight``.
    - ``t_last_spike_ms``: timestamp of previous presynaptic spike,
      initialized to ``0.0`` as in NEST.

    For a presynaptic spike at time :math:`t`, delay :math:`d`, and previous
    presynaptic spike time :math:`t_\mathrm{last}`, NEST update order is:

    1. Retrieve postsynaptic LTP history entries in
       :math:`(t_\mathrm{last}-d,\; t-d]`.
    2. For each entry with timestamp :math:`t_i` and amplitude ``dw_i``,
       facilitate weight:

       .. math::

          w \leftarrow \min\left(W_\max,
          w + dw_i\,x_\mathrm{bar}\exp\left(\frac{t_\mathrm{last}-(t_i+d)}{\tau_x}\right)\right).

    3. Apply LTD from target at time :math:`t-d`:

       .. math::

          w \leftarrow \max(W_\min,\; w - dw_\mathrm{LTD}(t-d)).

    4. Emit spike event with updated ``weight``.
    5. Update presynaptic trace:

       .. math::

          x_\mathrm{bar} \leftarrow
          x_\mathrm{bar}\exp\left(\frac{t_\mathrm{last}-t}{\tau_x}\right)
          + \frac{1}{\tau_x}.

    6. Set :math:`t_\mathrm{last} \leftarrow t`.

    The update ordering above is the critical NEST behavior in
    ``clopath_synapse::send(...)``.

    Parameters
    ----------
    weight : float, optional
        Synaptic weight. Default ``1.0``.
    delay : float, optional
        Connection delay in milliseconds. Default ``1.0``.
    delay_steps : int, optional
        Integer delay in simulation steps for event payloads. Default ``1``.
    x_bar : float, optional
        Initial presynaptic trace value. Default ``0.0``.
    tau_x : float, optional
        Presynaptic trace time constant in ms. Default ``15.0``.
    Wmin : float, optional
        Minimum allowed weight. Default ``0.0``.
    Wmax : float, optional
        Maximum allowed weight. Default ``100.0``.
    t_last_spike_ms : float, optional
        Last presynaptic spike timestamp in ms. Default ``0.0``.
    name : str, optional
        Optional model instance name.

    Target interface
    ----------------
    ``send()`` expects the postsynaptic target object to provide:

    - ``get_LTP_history(t1, t2)`` or ``get_ltp_history(t1, t2)`` returning an
      iterable of entries with time and amplitude fields, and
    - ``get_LTD_value(t)`` or ``get_ltd_value(t)`` returning a scalar LTD
      amplitude.

    Each LTP entry may be any of:

    - object with attributes ``t_`` and ``dw_``,
    - object with attributes ``t`` and ``dw``,
    - mapping containing ``t``/``t_`` and ``dw``/``dw_``,
    - 2-tuple ``(t, dw)``.

    Notes
    -----
    - As in NEST, precise (sub-grid) spike offsets are ignored in Clopath
      plasticity updates.
    - Sign constraints follow NEST checks exactly:
      ``weight`` must have same sign as ``Wmin`` and ``Wmax`` according to
      NEST's internal sign tests.

    References
    ----------
    .. [1] Clopath C et al. (2010). Connectivity reflects coding:
           a model of voltage-based STDP with homeostasis.
           Nature Neuroscience 13(3):344-352.
    .. [2] NEST source: ``models/clopath_synapse.h`` and
           ``models/clopath_synapse.cpp``.
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    IS_PRIMARY = True
    REQUIRES_CLOPATH_ARCHIVING = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True
    SUPPORTS_WFR = True

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        x_bar: ArrayLike = 0.0,
        tau_x: ArrayLike = 15.0,
        Wmin: ArrayLike = 0.0,
        Wmax: ArrayLike = 100.0,
        t_last_spike_ms: ArrayLike = 0.0,
        name: str | None = None,
    ):
        self.name = name

        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay = self._validate_positive_delay(delay)
        self.delay_steps = self._validate_delay_steps(delay_steps)

        self.x_bar = self._to_float_scalar(x_bar, name='x_bar')
        self.tau_x = self._to_float_scalar(tau_x, name='tau_x')
        self.Wmin = self._to_float_scalar(Wmin, name='Wmin')
        self.Wmax = self._to_float_scalar(Wmax, name='Wmax')
        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

        self._check_weight_sign_constraints()

    @property
    def properties(self) -> dict[str, Any]:
        return {
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'requires_clopath_archiving': self.REQUIRES_CLOPATH_ARCHIVING,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            'weight': float(self.weight),
            'delay': float(self.delay),
            'delay_steps': int(self.delay_steps),
            'x_bar': float(self.x_bar),
            'tau_x': float(self.tau_x),
            'Wmin': float(self.Wmin),
            'Wmax': float(self.Wmax),
            't_last_spike_ms': float(self.t_last_spike_ms),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'requires_clopath_archiving': self.REQUIRES_CLOPATH_ARCHIVING,
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
        if 'x_bar' in updates:
            self.x_bar = self._to_float_scalar(updates['x_bar'], name='x_bar')
        if 'tau_x' in updates:
            self.tau_x = self._to_float_scalar(updates['tau_x'], name='tau_x')
        if 'Wmin' in updates:
            self.Wmin = self._to_float_scalar(updates['Wmin'], name='Wmin')
        if 'Wmax' in updates:
            self.Wmax = self._to_float_scalar(updates['Wmax'], name='Wmax')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')

        self._check_weight_sign_constraints()

    def get(self, key: str = 'status'):
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for clopath_synapse.get().')

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
        if self.tau_x == 0.0:
            raise ValueError('tau_x must be non-zero.')

        dendritic_delay = self.delay if delay is None else self._validate_positive_delay(delay)
        event_delay_steps = (
            self.delay_steps
            if delay_steps is None
            else self._validate_delay_steps(delay_steps)
        )

        ltp_entries = self._get_ltp_history(
            target,
            self.t_last_spike_ms - dendritic_delay,
            t_spike - dendritic_delay,
        )

        for entry in ltp_entries:
            t_hist, dw = self._extract_history_entry(entry)
            minus_dt = self.t_last_spike_ms - (t_hist + dendritic_delay)

            self.weight = self._facilitate(
                self.weight,
                dw,
                self.x_bar * math.exp(minus_dt / self.tau_x),
            )

        ltd_dw = self._get_ltd_value(target, t_spike - dendritic_delay)
        self.weight = self._depress(self.weight, ltd_dw)

        event = {
            'weight': float(self.weight),
            'delay': float(dendritic_delay),
            'delay_steps': int(event_delay_steps),
            'receptor_type': self._to_int_scalar(receptor_type, name='receptor_type'),
            'multiplicity': self._validate_multiplicity(multiplicity),
            't_spike_ms': float(t_spike),
        }

        self.x_bar = self.x_bar * math.exp((self.t_last_spike_ms - t_spike) / self.tau_x) + 1.0 / self.tau_x
        self.t_last_spike_ms = t_spike

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
        spike_times_ms: ArrayLike,
        target: Any,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay: ArrayLike | None = None,
        delay_steps: ArrayLike | None = None,
    ) -> list[dict[str, Any]]:
        times = np.asarray(u.math.asarray(spike_times_ms), dtype=np.float64).reshape(-1)
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

    def _check_weight_sign_constraints(self):
        # Keep sign checks exactly as in NEST clopath_synapse::set_status.
        if self._sign_like_wmin(self.weight) != self._sign_like_wmin(self.Wmin):
            raise ValueError('Weight and Wmin must have same sign.')

        if self._sign_like_wmax(self.weight) != self._sign_like_wmax(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')

    @staticmethod
    def _sign_like_wmin(x: float) -> int:
        return int((x >= 0.0) - (x < 0.0))

    @staticmethod
    def _sign_like_wmax(x: float) -> int:
        return int((x > 0.0) - (x <= 0.0))

    def _depress(self, w: float, dw: float) -> float:
        w_new = w - float(dw)
        return w_new if w_new > self.Wmin else self.Wmin

    def _facilitate(self, w: float, dw: float, x_trace: float) -> float:
        w_new = w + float(dw) * float(x_trace)
        return w_new if w_new < self.Wmax else self.Wmax

    def _get_ltp_history(self, target: Any, t1: float, t2: float):
        fn = getattr(target, 'get_LTP_history', None)
        if fn is None:
            fn = getattr(target, 'get_ltp_history', None)
        if fn is None or not callable(fn):
            raise AttributeError(
                'Target must provide get_LTP_history(t1, t2) or get_ltp_history(t1, t2).'
            )
        history = fn(float(t1), float(t2))
        if history is None:
            return []
        return history

    def _get_ltd_value(self, target: Any, t: float) -> float:
        fn = getattr(target, 'get_LTD_value', None)
        if fn is None:
            fn = getattr(target, 'get_ltd_value', None)
        if fn is None or not callable(fn):
            raise AttributeError(
                'Target must provide get_LTD_value(t) or get_ltd_value(t).'
            )
        return float(fn(float(t)))

    @staticmethod
    def _extract_history_entry(entry: Any) -> tuple[float, float]:
        t = None
        dw = None

        if isinstance(entry, dict):
            t = entry.get('t_', entry.get('t', entry.get('time_ms', entry.get('time', None))))
            dw = entry.get('dw_', entry.get('dw', entry.get('delta_w', entry.get('weight_change', None))))
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            t, dw = entry[0], entry[1]
        else:
            t = getattr(entry, 't_', getattr(entry, 't', None))
            dw = getattr(entry, 'dw_', getattr(entry, 'dw', None))

        if t is None or dw is None:
            raise ValueError('Each LTP history entry must provide both time and dw values.')

        return float(t), float(dw)

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

    def _validate_positive_delay(self, value: ArrayLike) -> float:
        d = self._to_float_scalar(value, name='delay')
        if d <= 0.0:
            raise ValueError('delay must be > 0.')
        return d

    def _validate_delay_steps(self, value: ArrayLike) -> int:
        d = self._to_int_scalar(value, name='delay_steps')
        if d < 1:
            raise ValueError('delay_steps must be >= 1.')
        return d

    def _validate_multiplicity(self, value: ArrayLike) -> float:
        m = self._to_float_scalar(value, name='multiplicity')
        if m < 0.0:
            raise ValueError('multiplicity must be >= 0.')
        return m
