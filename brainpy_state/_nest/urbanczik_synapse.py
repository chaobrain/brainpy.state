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
    'urbanczik_synapse',
]


class urbanczik_synapse:
    r"""NEST-compatible ``urbanczik_synapse`` connection model.

    Short description
    -----------------
    Plastic synapse after Urbanczik and Senn with dendritic prediction error.

    Description
    -----------
    This class reproduces connection-level semantics of NEST
    ``models/urbanczik_synapse.{h,cpp}``.

    In NEST, this synapse can only connect to targets supporting Urbanczik
    archiving (for example ``pp_cond_exp_mc_urbanczik``). Plasticity depends
    on archived postsynaptic dendritic prediction-error entries in
    ``(t_last - d, t_spike - d]`` and on presynaptic traces updated per spike.

    Connection state follows NEST member variables:

    - ``weight``: current synaptic weight.
    - ``init_weight``: baseline weight captured by ``set_status``.
    - ``tau_Delta`` [ms]: low-pass time constant of weight change.
    - ``eta``: learning rate.
    - ``Wmin``/``Wmax``: lower/upper hard bounds.
    - ``PI_integral`` and ``PI_exp_integral``: accumulated internal integrals.
    - ``tau_L_trace`` and ``tau_s_trace``: presynaptic traces.
    - ``t_last_spike_ms``: previous presynaptic spike time, default ``-1.0``.

    For a presynaptic spike at time :math:`t`, delay :math:`d`, and history
    entries :math:`(t_i, \Delta w_i)` from the target:

    .. math::
       \Pi_i =
       \left[
       \tau_L^\mathrm{tr}\exp\left(\frac{t_{last}-(t_i+d)}{\tau_L}\right)
       - \tau_s^\mathrm{tr}\exp\left(\frac{t_{last}-(t_i+d)}{\tau_s}\right)
       \right]\Delta w_i

    .. math::
       \Pi_\mathrm{int} \leftarrow \Pi_\mathrm{int} + \sum_i \Pi_i

    .. math::
       \Pi_\mathrm{exp} \leftarrow
       \exp\left(\frac{t_{last}-t}{\tau_\Delta}\right)\Pi_\mathrm{exp}
       + \sum_i \exp\left(\frac{(t_i+d)-t}{\tau_\Delta}\right)\Pi_i

    .. math::
       w \leftarrow \mathrm{clip}\left(
       w_0 +
       (\Pi_\mathrm{int}-\Pi_\mathrm{exp})
       \frac{15\,C_m\,\tau_s\,\eta}{g_L(\tau_L-\tau_s)},
       W_{min}, W_{max}
       \right)

    NEST send-ordering in ``urbanczik_synapse::send(...)`` is preserved:

    1. Read archived history in ``(t_last - delay, t_spike - delay]``.
    2. Update integrals and compute new weight.
    3. Emit spike event with updated ``weight``.
    4. Update ``tau_L_trace`` and ``tau_s_trace``.
    5. Set ``t_last_spike_ms = t_spike``.

    The synaptic time constant branch is NEST-exact:
    ``tau_syn_ex`` is used when current ``weight > 0`` else ``tau_syn_in``.

    Parameters
    ----------
    weight : float, optional
        Synaptic weight. Default ``1.0``.
    delay : float, optional
        Dendritic delay used for history lookup. Must be ``> 0``.
        Default ``1.0``.
    delay_steps : int, optional
        Event delivery delay in simulation steps. Must be ``>= 1``.
        Default ``1``.
    tau_Delta : float, optional
        Time constant (ms) of low-pass filtered weight change.
        Default ``100.0``.
    eta : float, optional
        Learning rate. Default ``0.07``.
    Wmin : float, optional
        Lower bound of synaptic weight. Default ``0.0``.
    Wmax : float, optional
        Upper bound of synaptic weight. Default ``100.0``.
    PI_integral : float, optional
        Initial value of accumulated integral. Default ``0.0``.
    PI_exp_integral : float, optional
        Initial value of exponentially filtered integral. Default ``0.0``.
    tau_L_trace : float, optional
        Initial :math:`\tau_L` trace state. Default ``0.0``.
    tau_s_trace : float, optional
        Initial :math:`\tau_s` trace state. Default ``0.0``.
    t_last_spike_ms : float, optional
        Last presynaptic spike time in ms. Default ``-1.0``.
    name : str, optional
        Optional instance name.

    Target interface
    ----------------
    ``send()`` expects target methods:

    - ``get_urbanczik_history(t1, t2, comp)`` (or ``get_urbanczik_history(t1, t2)``)
      returning history entries in ``(t1, t2]``.
    - ``get_g_L(comp)``, ``get_tau_L(comp)``, ``get_C_m(comp)``,
      ``get_tau_syn_ex(comp)``, ``get_tau_syn_in(comp)``.

    History entries can be objects with ``t_``/``dw_`` (NEST-like), objects
    with ``t``/``dw``, mappings with those keys, or 2-tuples ``(t, dw)``.

    Notes
    -----
    - As in NEST, precise (sub-grid) timestamp offsets are ignored in this
      plasticity rule.
    - ``set_status()`` reproduces NEST behavior where ``init_weight`` is reset
      to current ``weight`` at each call.
    - Sign checks intentionally mirror current NEST implementation, including
      the original status-message wording.

    References
    ----------
    .. [1] Urbanczik R, Senn W (2014). Learning by the dendritic prediction
           of somatic spiking. Neuron 81:521-528.
    .. [2] NEST source: ``models/urbanczik_synapse.h`` and
           ``models/urbanczik_synapse.cpp``.
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    IS_PRIMARY = True
    REQUIRES_URBANCZIK_ARCHIVING = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True
    SUPPORTS_WFR = True

    DENDRITIC_COMPARTMENT = 1

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        tau_Delta: ArrayLike = 100.0,
        eta: ArrayLike = 0.07,
        Wmin: ArrayLike = 0.0,
        Wmax: ArrayLike = 100.0,
        PI_integral: ArrayLike = 0.0,
        PI_exp_integral: ArrayLike = 0.0,
        tau_L_trace: ArrayLike = 0.0,
        tau_s_trace: ArrayLike = 0.0,
        t_last_spike_ms: ArrayLike = -1.0,
        name: str | None = None,
    ):
        self.name = name

        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay = self._validate_positive_delay(delay)
        self.delay_steps = self._validate_delay_steps(delay_steps)

        self.tau_Delta = self._to_float_scalar(tau_Delta, name='tau_Delta')
        self.eta = self._to_float_scalar(eta, name='eta')
        self.Wmin = self._to_float_scalar(Wmin, name='Wmin')
        self.Wmax = self._to_float_scalar(Wmax, name='Wmax')

        self.PI_integral = self._to_float_scalar(PI_integral, name='PI_integral')
        self.PI_exp_integral = self._to_float_scalar(PI_exp_integral, name='PI_exp_integral')
        self.tau_L_trace = self._to_float_scalar(tau_L_trace, name='tau_L_trace')
        self.tau_s_trace = self._to_float_scalar(tau_s_trace, name='tau_s_trace')
        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

        # NEST initializes init_weight_ from weight_.
        self.init_weight = float(self.weight)

        self._check_weight_sign_constraints()

    @property
    def properties(self) -> dict[str, Any]:
        return {
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'requires_urbanczik_archiving': self.REQUIRES_URBANCZIK_ARCHIVING,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            'weight': float(self.weight),
            'delay': float(self.delay),
            'delay_steps': int(self.delay_steps),
            'tau_Delta': float(self.tau_Delta),
            'eta': float(self.eta),
            'Wmin': float(self.Wmin),
            'Wmax': float(self.Wmax),
            'init_weight': float(self.init_weight),
            'PI_integral': float(self.PI_integral),
            'PI_exp_integral': float(self.PI_exp_integral),
            'tau_L_trace': float(self.tau_L_trace),
            'tau_s_trace': float(self.tau_s_trace),
            't_last_spike_ms': float(self.t_last_spike_ms),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'requires_urbanczik_archiving': self.REQUIRES_URBANCZIK_ARCHIVING,
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
        if 'tau_Delta' in updates:
            self.tau_Delta = self._to_float_scalar(updates['tau_Delta'], name='tau_Delta')
        if 'eta' in updates:
            self.eta = self._to_float_scalar(updates['eta'], name='eta')
        if 'Wmin' in updates:
            self.Wmin = self._to_float_scalar(updates['Wmin'], name='Wmin')
        if 'Wmax' in updates:
            self.Wmax = self._to_float_scalar(updates['Wmax'], name='Wmax')
        if 'PI_integral' in updates:
            self.PI_integral = self._to_float_scalar(updates['PI_integral'], name='PI_integral')
        if 'PI_exp_integral' in updates:
            self.PI_exp_integral = self._to_float_scalar(updates['PI_exp_integral'], name='PI_exp_integral')
        if 'tau_L_trace' in updates:
            self.tau_L_trace = self._to_float_scalar(updates['tau_L_trace'], name='tau_L_trace')
        if 'tau_s_trace' in updates:
            self.tau_s_trace = self._to_float_scalar(updates['tau_s_trace'], name='tau_s_trace')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')

        if 'init_weight' in updates:
            self.init_weight = self._to_float_scalar(updates['init_weight'], name='init_weight')
        else:
            # NEST set_status() always syncs init_weight_ to current weight_.
            self.init_weight = float(self.weight)

        self._check_weight_sign_constraints()

    def get(self, key: str = 'status'):
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for urbanczik_synapse.get().')

    def set_weight(self, weight: ArrayLike):
        self.weight = self._to_float_scalar(weight, name='weight')
        self._check_weight_sign_constraints()

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

        comp = self.DENDRITIC_COMPARTMENT
        history_entries = self._get_urbanczik_history(
            target,
            self.t_last_spike_ms - dendritic_delay,
            t_spike - dendritic_delay,
            comp=comp,
        )

        g_L = self._get_compartment_value(target, ['get_g_L', 'get_g_l'], comp=comp, field='g_L')
        tau_L = self._get_tau_L(target, comp=comp)
        C_m = self._get_compartment_value(target, ['get_C_m', 'get_c_m'], comp=comp, field='C_m')
        tau_syn_ex = self._get_compartment_value(
            target,
            ['get_tau_syn_ex', 'get_tau_syn_exc'],
            comp=comp,
            field='tau_syn_ex',
        )
        tau_syn_in = self._get_compartment_value(
            target,
            ['get_tau_syn_in', 'get_tau_syn_inh'],
            comp=comp,
            field='tau_syn_in',
        )
        tau_s = tau_syn_ex if self.weight > 0.0 else tau_syn_in

        dPI_exp_integral = 0.0
        for entry in history_entries:
            t_hist, dw = self._extract_history_entry(entry)

            t_up = t_hist + dendritic_delay
            minus_delta_t_up = self.t_last_spike_ms - t_up
            minus_t_down = t_up - t_spike

            PI = (
                self.tau_L_trace * math.exp(minus_delta_t_up / tau_L)
                - self.tau_s_trace * math.exp(minus_delta_t_up / tau_s)
            ) * dw
            self.PI_integral += PI
            dPI_exp_integral += math.exp(minus_t_down / self.tau_Delta) * PI

        self.PI_exp_integral = (
            math.exp((self.t_last_spike_ms - t_spike) / self.tau_Delta) * self.PI_exp_integral
            + dPI_exp_integral
        )

        self.weight = self.PI_integral - self.PI_exp_integral
        self.weight = self.init_weight + (
            self.weight * 15.0 * C_m * tau_s * self.eta / (g_L * (tau_L - tau_s))
        )
        if self.weight > self.Wmax:
            self.weight = self.Wmax
        elif self.weight < self.Wmin:
            self.weight = self.Wmin

        event = {
            'weight': float(self.weight),
            'delay': float(dendritic_delay),
            'delay_steps': int(event_delay_steps),
            'receptor_type': self._to_int_scalar(receptor_type, name='receptor_type'),
            'multiplicity': self._validate_multiplicity(multiplicity),
            't_spike_ms': float(t_spike),
            'tau_s_ms': float(tau_s),
            'PI_integral': float(self.PI_integral),
            'PI_exp_integral': float(self.PI_exp_integral),
        }

        self.tau_L_trace = self.tau_L_trace * math.exp((self.t_last_spike_ms - t_spike) / tau_L) + 1.0
        self.tau_s_trace = self.tau_s_trace * math.exp((self.t_last_spike_ms - t_spike) / tau_s) + 1.0
        self.t_last_spike_ms = t_spike

        event['tau_L_trace_post'] = float(self.tau_L_trace)
        event['tau_s_trace_post'] = float(self.tau_s_trace)
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

    def _check_weight_sign_constraints(self):
        # Keep sign checks/message text aligned with NEST urbanczik_synapse::set_status.
        if bool(np.signbit(self.weight)) != bool(np.signbit(self.Wmax)):
            raise ValueError('Weight and Wmin must have same sign.')

        if self._sign_like_wmax(self.weight) != self._sign_like_wmax(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')

    @staticmethod
    def _sign_like_wmax(x: float) -> int:
        return int((x > 0.0) - (x <= 0.0))

    @staticmethod
    def _get_urbanczik_history(target: Any, t1: float, t2: float, comp: int):
        fn = getattr(target, 'get_urbanczik_history', None)
        if fn is None or not callable(fn):
            raise AttributeError(
                'Target must provide get_urbanczik_history(t1, t2, comp) for urbanczik_synapse.'
            )

        try:
            history = fn(float(t1), float(t2), int(comp))
        except TypeError:
            history = fn(float(t1), float(t2))

        if history is None:
            return []
        return history

    @classmethod
    def _get_tau_L(cls, target: Any, comp: int) -> float:
        fn = getattr(target, 'get_tau_L', None)
        if fn is None:
            fn = getattr(target, 'get_tau_l', None)
        if fn is not None and callable(fn):
            try:
                return float(fn(int(comp)))
            except TypeError:
                return float(fn())

        c_m = cls._get_compartment_value(target, ['get_C_m', 'get_c_m'], comp=comp, field='C_m')
        g_l = cls._get_compartment_value(target, ['get_g_L', 'get_g_l'], comp=comp, field='g_L')
        return float(c_m / g_l)

    @staticmethod
    def _get_compartment_value(target: Any, names: list[str], comp: int, field: str) -> float:
        for name in names:
            fn = getattr(target, name, None)
            if fn is None or not callable(fn):
                continue
            try:
                return float(fn(int(comp)))
            except TypeError:
                return float(fn())
        raise AttributeError(
            f'Target must provide {"/".join(names)}(comp) for urbanczik_synapse ({field}).'
        )

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
            raise ValueError('Each Urbanczik history entry must provide both time and dw values.')

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
