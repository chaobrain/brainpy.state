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

import math
from typing import Callable, Hashable, Iterable

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'iaf_bw_2001_exact',
]


class iaf_bw_2001_exact(Neuron):
    r"""NEST-compatible ``iaf_bw_2001_exact`` neuron model.

    Short description
    -----------------

    Conductance-based leaky integrate-and-fire neuron with AMPA, GABA and
    exact per-synapse NMDA dynamics, equivalent to NEST
    ``models/iaf_bw_2001_exact.{h,cpp}``.

    Description
    -----------

    ``iaf_bw_2001_exact`` implements the Brunel-Wang style NMDA kinetics
    without the presynaptic-jump approximation used by :class:`iaf_bw_2001`.
    For each NMDA connection, the model stores explicit NMDA rise/decay states.

    The model follows NEST update ordering:

    1. Integrate ODE state on :math:`(t, t+dt]` with adaptive RKF45 and
       persistent internal step size.
    2. Add AMPA/GABA spike jumps to ``s_AMPA/s_GABA`` and NMDA jumps to each
       NMDA rise variable ``x_j``.
    3. Apply refractory countdown or threshold/reset/spike emission.
    4. Store external current in one-step delayed buffer ``I_stim``.

    Neuron and synaptic dynamics
    ............................

    Let :math:`j` index NMDA synapses. The continuous-time dynamics are

    .. math::

       C_m \frac{dV_m}{dt} = -g_L(V_m - E_L) - I_{syn} + I_{stim},

    .. math::

       I_{syn} = I_{AMPA} + I_{GABA} + I_{NMDA},

    .. math::

       I_{AMPA} = (V_m - E_{ex}) s_{AMPA},
       \quad
       I_{GABA} = (V_m - E_{in}) s_{GABA},

    .. math::

       I_{NMDA} = \frac{(V_m - E_{ex})}
       {1 + [Mg^{2+}]\exp(-0.062V_m)/3.57}
       \sum_j w_j s_j,

    .. math::

       \frac{ds_{AMPA}}{dt} = -\frac{s_{AMPA}}{\tau_{AMPA}},
       \qquad
       \frac{ds_{GABA}}{dt} = -\frac{s_{GABA}}{\tau_{GABA}},

    .. math::

       \frac{dx_j}{dt} = -\frac{x_j}{\tau_{NMDA,rise}},
       \qquad
       \frac{ds_j}{dt} =
       -\frac{s_j}{\tau_{NMDA,decay}} + \alpha x_j (1-s_j).

    Here :math:`w_j` is the connection weight associated with NMDA port
    :math:`j`.

    NMDA event and port semantics
    .............................

    NEST gives every NMDA connection a unique receptor port at connect time and
    forbids adding new NMDA connections after the first simulation call.
    This implementation mirrors that behavior:

    - Each NMDA event uses a ``port`` identifier (hashable value).
    - The first NMDA event seen for a new ``port`` registers that port with a
      fixed weight.
    - The same ``port`` must always use the same weight.
    - New NMDA ports are allowed only before the first :meth:`update` call.

    Accepted ``spike_events`` formats:

    - ``(receptor, weight)``
    - ``(receptor, weight, third)``
      where ``third`` is multiplicity for AMPA/GABA and ``port`` for NMDA
    - ``(receptor, weight, port, multiplicity)`` for NMDA
    - ``dict`` with keys:
      ``receptor_type``/``receptor``, ``weight``, ``multiplicity``, and
      optional ``port``/``rport``/``synapse_id``.

    AMPA and GABA increments are ``weight * multiplicity``.
    NMDA increments are multiplicity-only jumps of ``x_j`` with
    fixed per-port weights.

    Parameters
    ----------

    ==================== ================== ===========================================================
    **Parameter**        **Default**        **Description**
    ==================== ================== ===========================================================
    ``in_size``          (required)         Population shape
    ``E_L``              -70 mV             Leak reversal potential
    ``E_ex``             0 mV               Excitatory reversal potential
    ``E_in``             -70 mV             Inhibitory reversal potential
    ``V_th``             -55 mV             Spike threshold
    ``V_reset``          -60 mV             Reset potential
    ``C_m``              500 pF             Membrane capacitance
    ``g_L``              25 nS              Leak conductance
    ``t_ref``            2 ms               Absolute refractory duration
    ``tau_AMPA``         2 ms               AMPA decay time constant
    ``tau_GABA``         5 ms               GABA decay time constant
    ``tau_rise_NMDA``    2 ms               NMDA rise time constant
    ``tau_decay_NMDA``   100 ms             NMDA decay time constant
    ``alpha``            0.5 / ms           NMDA rise coupling strength
    ``conc_Mg2``         1 mM               Extracellular magnesium concentration
    ``gsl_error_tol``    1e-3               RKF45 local error tolerance
    ``V_initializer``    Constant(-70 mV)   Membrane initializer
    ``s_AMPA_initializer`` Constant(0 nS)   AMPA state initializer
    ``s_GABA_initializer`` Constant(0 nS)   GABA state initializer
    ``spk_fun``          ReluGrad()         Surrogate spike function
    ``spk_reset``        ``'hard'``         Reset mode; hard reset matches NEST behavior
    ``ref_var``          ``False``          If True, expose boolean refractory indicator
    ==================== ================== ===========================================================

    Recordables
    -----------

    - ``V_m``
    - ``s_AMPA``
    - ``s_GABA``
    - ``s_NMDA`` (weighted NMDA sum, :math:`\sum_j w_j s_j`)
    - ``I_NMDA``
    - ``I_AMPA``
    - ``I_GABA``

    Additional state
    ----------------

    - ``x_NMDA``: NMDA rise variables for each registered NMDA port.
    - ``s_NMDA_components``: NMDA decay/gating variables for each NMDA port.
    - ``nmda_weights``: fixed per-port NMDA weights.
    - ``refractory_step_count``: absolute refractory countdown.
    - ``integration_step``: persistent adaptive RKF45 step state.
    - ``I_stim``: one-step delayed external current buffer.

    References
    ----------
    .. [1] Wang X-J (1999). Synaptic basis of cortical persistent activity:
           The importance of NMDA receptors to working memory.
           Journal of Neuroscience, 19(21):9587-9603.
           DOI: https://doi.org/10.1523/JNEUROSCI.19-21-09587.1999
    .. [2] Brunel N, Wang X-J (2001). Effects of neuromodulation in a cortical
           network model of object working memory dominated by recurrent
           inhibition. Journal of Computational Neuroscience, 11(1):63-85.
           DOI: https://doi.org/10.1023/A:1011204814320
    .. [3] Wang X-J (2002). Probabilistic decision making by slow
           reverberation in cortical circuits. Neuron, 36(5):955-968.
           DOI: https://doi.org/10.1016/S0896-6273(02)01092-9
    .. [4] NEST source: ``models/iaf_bw_2001_exact.h`` and
           ``models/iaf_bw_2001_exact.cpp``.
    """

    __module__ = 'brainpy.state'

    AMPA = 1
    GABA = 2
    NMDA = 3

    RECEPTOR_TYPES = {
        'AMPA': AMPA,
        'GABA': GABA,
        'NMDA': NMDA,
    }

    RECORDABLES = (
        'V_m',
        's_AMPA',
        's_GABA',
        's_NMDA',
        'I_NMDA',
        'I_AMPA',
        'I_GABA',
    )

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 10000

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        E_ex: ArrayLike = 0. * u.mV,
        E_in: ArrayLike = -70. * u.mV,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -60. * u.mV,
        C_m: ArrayLike = 500. * u.pF,
        g_L: ArrayLike = 25. * u.nS,
        t_ref: ArrayLike = 2. * u.ms,
        tau_AMPA: ArrayLike = 2. * u.ms,
        tau_GABA: ArrayLike = 5. * u.ms,
        tau_rise_NMDA: ArrayLike = 2. * u.ms,
        tau_decay_NMDA: ArrayLike = 100. * u.ms,
        alpha: ArrayLike = 0.5 / u.ms,
        conc_Mg2: ArrayLike = 1.0 * u.mM,
        gsl_error_tol: ArrayLike = 1e-3,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        s_AMPA_initializer: Callable = braintools.init.Constant(0. * u.nS),
        s_GABA_initializer: Callable = braintools.init.Constant(0. * u.nS),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)

        self.tau_AMPA = braintools.init.param(tau_AMPA, self.varshape)
        self.tau_GABA = braintools.init.param(tau_GABA, self.varshape)
        self.tau_rise_NMDA = braintools.init.param(tau_rise_NMDA, self.varshape)
        self.tau_decay_NMDA = braintools.init.param(tau_decay_NMDA, self.varshape)
        self.alpha = braintools.init.param(alpha, self.varshape)
        self.conc_Mg2 = braintools.init.param(conc_Mg2, self.varshape)
        self.gsl_error_tol = braintools.init.param(gsl_error_tol, self.varshape)

        self.V_initializer = V_initializer
        self.s_AMPA_initializer = s_AMPA_initializer
        self.s_GABA_initializer = s_GABA_initializer
        self.ref_var = ref_var

        self._nmda_port_index = {}
        self._updates_started = False

        self._validate_parameters()

    @property
    def receptor_types(self):
        return dict(self.RECEPTOR_TYPES)

    @property
    def recordables(self):
        return list(self.RECORDABLES)

    @staticmethod
    def _value_to_float(x, unit=None):
        if unit is None:
            return np.asarray(u.math.asarray(x), dtype=np.float64)
        try:
            return np.asarray(u.math.asarray(x / unit), dtype=np.float64)
        except Exception:
            return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @classmethod
    def _normalize_spike_receptor(cls, receptor):
        if isinstance(receptor, str):
            key = receptor.strip()
            if key in cls.RECEPTOR_TYPES:
                return cls.RECEPTOR_TYPES[key]
            if key.isdigit():
                receptor = int(key)
            else:
                raise ValueError(f'Unknown receptor label: {receptor}')
        receptor = int(receptor)
        if receptor < cls.AMPA or receptor > cls.NMDA:
            raise ValueError(f'Receptor type must be in [1, 3], got {receptor}.')
        return receptor

    @staticmethod
    def _normalize_nmda_port(port) -> Hashable:
        if port is None:
            return 0
        if isinstance(port, str):
            p = port.strip()
            if p.isdigit():
                return int(p)
            return p
        try:
            hash(port)
        except TypeError as e:
            raise ValueError(f'NMDA port must be hashable, got {type(port)}.') from e
        return port

    def _validate_parameters(self):
        if np.any(self._value_to_float(self.V_reset, u.mV) >= self._value_to_float(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._value_to_float(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._value_to_float(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._value_to_float(self.tau_AMPA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._value_to_float(self.tau_GABA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._value_to_float(self.tau_rise_NMDA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._value_to_float(self.tau_decay_NMDA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._value_to_float(self.alpha, 1 / u.ms) <= 0.0):
            raise ValueError('alpha > 0 required.')
        if np.any(self._value_to_float(self.conc_Mg2, u.mM) <= 0.0):
            raise ValueError('Mg2 concentration must be strictly positive.')
        if np.any(self._value_to_float(self.gsl_error_tol, None) <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def _nmda_num_ports(self):
        if hasattr(self, 'x_NMDA'):
            return int(np.asarray(self.x_NMDA.value).shape[-1])
        return 0

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        s_ampa = braintools.init.param(self.s_AMPA_initializer, self.varshape, batch_size)
        s_gaba = braintools.init.param(self.s_GABA_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.s_AMPA = brainstate.HiddenState(s_ampa)
        self.s_GABA = brainstate.HiddenState(s_gaba)

        state_shape = self.V.value.shape
        zeros = np.zeros(state_shape, dtype=np.float64)

        self.s_NMDA = brainstate.ShortTermState(zeros * u.nS)
        self.I_NMDA = brainstate.ShortTermState(zeros * u.pA)
        self.I_AMPA = brainstate.ShortTermState(zeros * u.pA)
        self.I_GABA = brainstate.ShortTermState(zeros * u.pA)

        self.x_NMDA = brainstate.ShortTermState(np.zeros(state_shape + (0,), dtype=np.float64))
        self.s_NMDA_components = brainstate.ShortTermState(np.zeros(state_shape + (0,), dtype=np.float64))
        self.nmda_weights = brainstate.ShortTermState(np.zeros(state_shape + (0,), dtype=np.float64))

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = self._safe_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0. * u.pA), self.varshape, batch_size)
        )

        self._nmda_port_index = {}
        self._updates_started = False

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.s_AMPA.value = braintools.init.param(self.s_AMPA_initializer, self.varshape, batch_size)
        self.s_GABA.value = braintools.init.param(self.s_GABA_initializer, self.varshape, batch_size)

        state_shape = self.V.value.shape
        zeros = np.zeros(state_shape, dtype=np.float64)
        self.s_NMDA.value = zeros * u.nS
        self.I_NMDA.value = zeros * u.pA
        self.I_AMPA.value = zeros * u.pA
        self.I_GABA.value = zeros * u.pA

        n_ports = self._nmda_num_ports()
        self.x_NMDA.value = np.zeros(state_shape + (n_ports,), dtype=np.float64)
        self.s_NMDA_components.value = np.zeros(state_shape + (n_ports,), dtype=np.float64)

        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)

        dt = self._safe_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0. * u.pA), self.varshape, batch_size
        )

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _ensure_nmda_port(self, port: Hashable, weight_np: np.ndarray, state_shape):
        if port in self._nmda_port_index:
            idx = self._nmda_port_index[port]
            current_weight = np.asarray(self.nmda_weights.value[..., idx], dtype=np.float64)
            if np.any(current_weight != weight_np):
                raise ValueError('iaf_bw_2001_exact requires constant weights per NMDA port.')
            return idx

        if self._updates_started:
            raise ValueError('NMDA ports can only be added before the first call to update().')

        idx = self._nmda_num_ports()
        self._nmda_port_index[port] = idx

        zero_channel = np.zeros(state_shape + (1,), dtype=np.float64)
        x_old = np.asarray(self.x_NMDA.value, dtype=np.float64)
        s_old = np.asarray(self.s_NMDA_components.value, dtype=np.float64)
        w_old = np.asarray(self.nmda_weights.value, dtype=np.float64)

        self.x_NMDA.value = np.concatenate([x_old, zero_channel], axis=-1)
        self.s_NMDA_components.value = np.concatenate([s_old, zero_channel], axis=-1)
        self.nmda_weights.value = np.concatenate([w_old, np.expand_dims(weight_np, axis=-1)], axis=-1)
        return idx

    def _parse_spike_events(self, spike_events: Iterable, state_shape):
        ds_ampa = np.zeros(state_shape, dtype=np.float64)
        ds_gaba = np.zeros(state_shape, dtype=np.float64)
        nmda_mult = np.zeros(state_shape + (self._nmda_num_ports(),), dtype=np.float64)

        if spike_events is None:
            return ds_ampa, ds_gaba, nmda_mult

        for ev in spike_events:
            receptor = 'AMPA'
            weight = 0.0 * u.nS
            multiplicity = 1.0
            port = None

            if isinstance(ev, dict):
                receptor = ev.get('receptor_type', ev.get('receptor', 'AMPA'))
                weight = ev.get('weight', 0.0 * u.nS)
                multiplicity = ev.get('multiplicity', 1.0)
                port = ev.get('port', ev.get('rport', ev.get('synapse_id', None)))
            else:
                if len(ev) == 2:
                    receptor, weight = ev
                elif len(ev) == 3:
                    receptor, weight, third = ev
                    receptor_id = self._normalize_spike_receptor(receptor)
                    if receptor_id == self.NMDA:
                        port = third
                    else:
                        multiplicity = third
                elif len(ev) == 4:
                    receptor, weight, port, multiplicity = ev
                else:
                    raise ValueError('Spike event tuples must have length 2, 3, or 4.')

            receptor_id = self._normalize_spike_receptor(receptor)
            weight_np = self._value_to_float(weight, u.nS)
            weight_np = np.broadcast_to(weight_np, state_shape)
            mult_np = self._value_to_float(multiplicity, None)
            mult_np = np.broadcast_to(mult_np, state_shape)

            if receptor_id == self.AMPA:
                ds_ampa = ds_ampa + weight_np * mult_np
            elif receptor_id == self.GABA:
                ds_gaba = ds_gaba + weight_np * mult_np
            else:
                nmda_port = self._normalize_nmda_port(port)
                nmda_idx = self._ensure_nmda_port(nmda_port, weight_np, state_shape)
                if nmda_idx >= nmda_mult.shape[-1]:
                    pad = np.zeros(state_shape + (nmda_idx + 1 - nmda_mult.shape[-1],), dtype=np.float64)
                    nmda_mult = np.concatenate([nmda_mult, pad], axis=-1)
                nmda_mult[..., nmda_idx] = nmda_mult[..., nmda_idx] + mult_np

        return ds_ampa, ds_gaba, nmda_mult

    def _parse_registered_spike_inputs(self, state_shape):
        ds_ampa = np.zeros(state_shape, dtype=np.float64)
        ds_gaba = np.zeros(state_shape, dtype=np.float64)
        if self.delta_inputs is None:
            return ds_ampa, ds_gaba

        for key in tuple(self.delta_inputs.keys()):
            val = self.delta_inputs[key]
            if callable(val):
                val = val()
            else:
                self.delta_inputs.pop(key)

            label = None
            if ' // ' in key:
                label, _ = key.split(' // ', maxsplit=1)

            if label is None:
                receptor = self.AMPA
            else:
                receptor = self._normalize_spike_receptor(label)

            if receptor == self.NMDA:
                raise ValueError('Use spike_events with NMDA port specification for iaf_bw_2001_exact.')

            val_np = self._value_to_float(val, u.nS)
            val_np = np.broadcast_to(val_np, state_shape)
            if receptor == self.AMPA:
                ds_ampa = ds_ampa + val_np
            else:
                ds_gaba = ds_gaba + val_np

        return ds_ampa, ds_gaba

    @staticmethod
    def _nmda_currents_scalar(v, s_ampa, s_gaba, s_nmda_sum, p):
        i_ampa = (v - p['E_ex']) * s_ampa
        i_gaba = (v - p['E_in']) * s_gaba
        denom = 1.0 + p['conc_Mg2'] * math.exp(-0.062 * v) / 3.57
        i_nmda = (v - p['E_ex']) / denom * s_nmda_sum
        return i_ampa, i_gaba, i_nmda

    @classmethod
    def _dynamics_scalar(cls, y, i_stim, p, nmda_weights):
        n_nmda = int(nmda_weights.shape[0])
        v = float(y[0])
        s_ampa = float(y[1])
        s_gaba = float(y[2])

        if n_nmda > 0:
            x_nmda = y[3:3 + n_nmda]
            s_nmda = y[3 + n_nmda:]
            s_nmda_sum = float(np.dot(s_nmda, nmda_weights))
        else:
            x_nmda = np.zeros((0,), dtype=np.float64)
            s_nmda = np.zeros((0,), dtype=np.float64)
            s_nmda_sum = 0.0

        i_ampa, i_gaba, i_nmda = cls._nmda_currents_scalar(v, s_ampa, s_gaba, s_nmda_sum, p)
        i_syn = i_ampa + i_gaba + i_nmda

        dy = np.zeros_like(y, dtype=np.float64)
        dy[0] = (-p['g_L'] * (v - p['E_L']) - i_syn + i_stim) / p['C_m']
        dy[1] = -s_ampa / p['tau_AMPA']
        dy[2] = -s_gaba / p['tau_GABA']

        if n_nmda > 0:
            dy[3:3 + n_nmda] = -x_nmda / p['tau_rise_NMDA']
            dy[3 + n_nmda:] = -s_nmda / p['tau_decay_NMDA'] + p['alpha'] * x_nmda * (1.0 - s_nmda)

        return dy, i_ampa, i_gaba, i_nmda, s_nmda_sum

    def _rkf45_integrate_scalar(self, y0, i_stim, h0, dt, p, nmda_weights, atol):
        t = 0.0
        h = max(h0, self._MIN_H)
        y = np.asarray(y0, dtype=np.float64)
        iters = 0

        i_ampa = 0.0
        i_gaba = 0.0
        i_nmda = 0.0
        s_nmda_sum = 0.0

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = max(self._MIN_H, min(h, dt - t))

            k1, *_ = self._dynamics_scalar(y, i_stim, p, nmda_weights)
            k2, *_ = self._dynamics_scalar(y + h * (1.0 / 4.0) * k1, i_stim, p, nmda_weights)
            k3, *_ = self._dynamics_scalar(
                y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0),
                i_stim,
                p,
                nmda_weights,
            )
            k4, *_ = self._dynamics_scalar(
                y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
                i_stim,
                p,
                nmda_weights,
            )
            k5, *_ = self._dynamics_scalar(
                y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
                i_stim,
                p,
                nmda_weights,
            )
            k6, *_ = self._dynamics_scalar(
                y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0),
                i_stim,
                p,
                nmda_weights,
            )

            y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
            y5 = y + h * (
                16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0
            )
            err = float(np.max(np.abs(y5 - y4)))

            if err <= atol or h <= self._MIN_H:
                y = y5
                t += h
                fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
                h = max(self._MIN_H, h * fac)
            else:
                fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
                h = max(self._MIN_H, h * fac)

            _, i_ampa, i_gaba, i_nmda, s_nmda_sum = self._dynamics_scalar(y, i_stim, p, nmda_weights)

        return y, h, i_ampa, i_gaba, i_nmda, s_nmda_sum

    def update(self, x=0. * u.pA, spike_events=None):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        state_shape = self.V.value.shape

        V = self._broadcast_to_state(self._value_to_float(self.V.value, u.mV), state_shape)
        s_ampa = self._broadcast_to_state(self._value_to_float(self.s_AMPA.value, u.nS), state_shape)
        s_gaba = self._broadcast_to_state(self._value_to_float(self.s_GABA.value, u.nS), state_shape)

        i_ampa_prev = self._broadcast_to_state(self._value_to_float(self.I_AMPA.value, u.pA), state_shape)
        i_gaba_prev = self._broadcast_to_state(self._value_to_float(self.I_GABA.value, u.pA), state_shape)
        i_nmda_prev = self._broadcast_to_state(self._value_to_float(self.I_NMDA.value, u.pA), state_shape)

        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), state_shape
        )
        i_stim = self._broadcast_to_state(self._value_to_float(self.I_stim.value, u.pA), state_shape)
        h_int = self._broadcast_to_state(self._value_to_float(self.integration_step.value, u.ms), state_shape)

        p = {
            'E_L': self._broadcast_to_state(self._value_to_float(self.E_L, u.mV), state_shape),
            'E_ex': self._broadcast_to_state(self._value_to_float(self.E_ex, u.mV), state_shape),
            'E_in': self._broadcast_to_state(self._value_to_float(self.E_in, u.mV), state_shape),
            'V_th': self._broadcast_to_state(self._value_to_float(self.V_th, u.mV), state_shape),
            'V_reset': self._broadcast_to_state(self._value_to_float(self.V_reset, u.mV), state_shape),
            'C_m': self._broadcast_to_state(self._value_to_float(self.C_m, u.pF), state_shape),
            'g_L': self._broadcast_to_state(self._value_to_float(self.g_L, u.nS), state_shape),
            'tau_AMPA': self._broadcast_to_state(self._value_to_float(self.tau_AMPA, u.ms), state_shape),
            'tau_GABA': self._broadcast_to_state(self._value_to_float(self.tau_GABA, u.ms), state_shape),
            'tau_rise_NMDA': self._broadcast_to_state(self._value_to_float(self.tau_rise_NMDA, u.ms), state_shape),
            'tau_decay_NMDA': self._broadcast_to_state(self._value_to_float(self.tau_decay_NMDA, u.ms), state_shape),
            'alpha': self._broadcast_to_state(self._value_to_float(self.alpha, 1 / u.ms), state_shape),
            'conc_Mg2': self._broadcast_to_state(self._value_to_float(self.conc_Mg2, u.mM), state_shape),
            'gsl_error_tol': self._broadcast_to_state(self._value_to_float(self.gsl_error_tol, None), state_shape),
        }

        ds_ampa_ev, ds_gaba_ev, dx_nmda_ev = self._parse_spike_events(spike_events, state_shape)
        ds_ampa_reg, ds_gaba_reg = self._parse_registered_spike_inputs(state_shape)
        ds_ampa = ds_ampa_ev + ds_ampa_reg
        ds_gaba = ds_gaba_ev + ds_gaba_reg

        x_nmda = np.asarray(self.x_NMDA.value, dtype=np.float64)
        s_nmda_components = np.asarray(self.s_NMDA_components.value, dtype=np.float64)
        nmda_weights = np.asarray(self.nmda_weights.value, dtype=np.float64)
        n_nmda = int(x_nmda.shape[-1])

        if dx_nmda_ev.shape[-1] != n_nmda:
            if dx_nmda_ev.shape[-1] < n_nmda:
                pad = np.zeros(state_shape + (n_nmda - dx_nmda_ev.shape[-1],), dtype=np.float64)
                dx_nmda_ev = np.concatenate([dx_nmda_ev, pad], axis=-1)
            else:
                dx_nmda_ev = dx_nmda_ev[..., :n_nmda]

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._value_to_float(new_i_stim_q, u.pA), state_shape)

        v_for_spike = np.empty_like(V)
        V_next = np.empty_like(V)
        s_ampa_next = np.empty_like(s_ampa)
        s_gaba_next = np.empty_like(s_gaba)
        x_nmda_next = np.empty_like(x_nmda)
        s_nmda_next = np.empty_like(s_nmda_components)
        s_nmda_sum_next = np.zeros_like(V)

        i_ampa_next = np.empty_like(i_ampa_prev)
        i_gaba_next = np.empty_like(i_gaba_prev)
        i_nmda_next = np.empty_like(i_nmda_prev)

        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)
        last_spike = self._broadcast_to_state(self._value_to_float(self.last_spike_time.value, u.ms), state_shape)
        last_spike_next = np.empty_like(last_spike)
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            state_shape,
        )

        for idx in np.ndindex(state_shape):
            local_p = {k: p[k][idx] for k in p}
            w_nmda_i = np.asarray(nmda_weights[idx], dtype=np.float64)
            x_i = np.asarray(x_nmda[idx], dtype=np.float64)
            s_i = np.asarray(s_nmda_components[idx], dtype=np.float64)
            y0 = np.concatenate([
                np.asarray([V[idx], s_ampa[idx], s_gaba[idx]], dtype=np.float64),
                x_i,
                s_i,
            ])
            y_i, h_i, ia_i, ig_i, in_i, s_nmda_sum_i = self._rkf45_integrate_scalar(
                y0=y0,
                i_stim=i_stim[idx],
                h0=h_int[idx],
                dt=dt,
                p=local_p,
                nmda_weights=w_nmda_i,
                atol=local_p['gsl_error_tol'],
            )

            sa_i = y_i[1] + ds_ampa[idx]
            sg_i = y_i[2] + ds_gaba[idx]
            if n_nmda > 0:
                x_i = y_i[3:3 + n_nmda] + dx_nmda_ev[idx]
                s_i = y_i[3 + n_nmda:]
            else:
                x_i = np.zeros((0,), dtype=np.float64)
                s_i = np.zeros((0,), dtype=np.float64)

            v_i = y_i[0]
            if r[idx] > 0:
                v_for_spike[idx] = local_p['V_reset']
                v_i = local_p['V_reset']
                r_i = r[idx] - 1
                t_last_i = last_spike[idx]
            else:
                v_for_spike[idx] = v_i
                if v_i >= local_p['V_th']:
                    v_i = local_p['V_reset']
                    r_i = int(refr_counts[idx])
                    t_last_i = float(u.math.asarray((t + dt_q) / u.ms))
                else:
                    r_i = 0
                    t_last_i = last_spike[idx]

            V_next[idx] = v_i
            s_ampa_next[idx] = sa_i
            s_gaba_next[idx] = sg_i
            if n_nmda > 0:
                x_nmda_next[idx] = x_i
                s_nmda_next[idx] = s_i
            s_nmda_sum_next[idx] = s_nmda_sum_i

            i_ampa_next[idx] = ia_i
            i_gaba_next[idx] = ig_i
            i_nmda_next[idx] = in_i

            r_next[idx] = r_i
            h_next[idx] = h_i
            last_spike_next[idx] = t_last_i

        self.V.value = V_next * u.mV
        self.s_AMPA.value = s_ampa_next * u.nS
        self.s_GABA.value = s_gaba_next * u.nS
        self.s_NMDA.value = s_nmda_sum_next * u.nS

        self.x_NMDA.value = x_nmda_next
        self.s_NMDA_components.value = s_nmda_next

        self.I_AMPA.value = i_ampa_next * u.pA
        self.I_GABA.value = i_gaba_next * u.pA
        self.I_NMDA.value = i_nmda_next * u.pA

        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_next * u.ms)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        self._updates_started = True
        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float64) * u.mV)
