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

from typing import Callable, Iterable

import numpy as np

import brainstate
import braintools
import brainunit as bu
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron
from .iaf_psc_exp import iaf_psc_exp

__all__ = [
    'iaf_tum_2000',
]


class iaf_tum_2000(Neuron):
    r"""NEST-compatible ``iaf_tum_2000`` neuron model.

    Short description
    -----------------

    Leaky integrate-and-fire neuron with exponential PSCs and integrated
    Tsodyks-Markram short-term synaptic plasticity on spike emission.

    Description
    -----------

    ``iaf_tum_2000`` extends :class:`iaf_psc_exp` by carrying presynaptic
    short-term plasticity states ``x``, ``y``, and ``u`` and by emitting a
    per-spike ``spike_offset`` that corresponds to the jump in ``y`` at spike
    time.

    The implementation follows NEST ``models/iaf_tum_2000.{h,cpp}`` update
    ordering and event semantics:

    1. Propagate membrane potential when not refractory.
    2. Decay synaptic currents.
    3. Add receptor-1 filtered current contribution to ``I_syn_ex``.
    4. Add arriving spike inputs (excitatory or inhibitory by sign).
    5. Perform threshold test and reset/refractory assignment.
    6. On emitted spike, update Tsodyks states exactly in NEST order and set
       ``spike_offset``.
    7. Buffer current inputs ``i_0`` and ``i_1`` for the next step.

    Membrane and synaptic dynamics
    ..............................

    Subthreshold dynamics:

    .. math::

       \frac{dV_m}{dt} =
       -\frac{V_m - E_L}{\tau_m} +
       \frac{I_{syn,ex} + I_{syn,in} + I_e + I_0}{C_m},

    .. math::

       \frac{dI_{syn,ex}}{dt} = -\frac{I_{syn,ex}}{\tau_{syn,ex}},
       \quad
       \frac{dI_{syn,in}}{dt} = -\frac{I_{syn,in}}{\tau_{syn,in}}.

    Receptor-1 current input ``I_1`` is filtered through the excitatory kernel:

    .. math::

       I_{syn,ex} \leftarrow I_{syn,ex} + (1 - e^{-h/\tau_{syn,ex}}) I_1.

    Tsodyks short-term dynamics on spike
    ....................................

    Let ``t_last`` be the previous spike time (with NEST-compatible first-spike
    convention ``t_last = 0`` if internal state is negative), ``t_spike`` the
    current spike time, and ``h_ts = t_spike - t_last``.

    Define:

    .. math::

       P_{uu} = \begin{cases}
       0, & \tau_{fac}=0 \\
       e^{-h_{ts}/\tau_{fac}}, & \text{otherwise}
       \end{cases},
       \quad
       P_{yy} = e^{-h_{ts}/\tau_{psc}},

    .. math::

       P_{zz} = \exp_m1(-h_{ts}/\tau_{rec}),
       \quad
       P_{xy} =
       \frac{P_{zz}\tau_{rec} - (P_{yy}-1)\tau_{psc}}{\tau_{psc}-\tau_{rec}}.

    With :math:`z = 1 - x - y`, NEST order is:

    .. math::

       u \leftarrow u P_{uu},
       \quad
       x \leftarrow x + P_{xy}y - P_{zz}z,
       \quad
       y \leftarrow y P_{yy},

    .. math::

       u \leftarrow u + U(1-u),
       \quad
       \Delta y = u x,
       \quad
       x \leftarrow x - \Delta y,
       \quad
       y \leftarrow y + \Delta y.

    ``spike_offset`` is set to :math:`\Delta y` on spike, else zero.

    Event semantics
    ...............

    ``spike_events`` items accepted by :meth:`update`:

    - ``(receptor_type, weight)``
    - ``(receptor_type, weight, offset)``
    - ``(receptor_type, weight, offset, multiplicity)``
    - ``(receptor_type, weight, offset, multiplicity, sender_model)``
    - dict with keys ``receptor_type``/``receptor``, ``weight``,
      ``offset``, ``multiplicity``, ``sender_model``.

    Receptors:

    - ``0``: regular spike input, effective weight ``weight * multiplicity``
    - ``1``: Tsodyks-coupled input, effective weight
      ``weight * multiplicity * offset``

    For receptor ``1``, ``sender_model`` must be ``"iaf_tum_2000"``;
    otherwise a ``ValueError`` is raised (mirrors NEST illegal-connection
    constraints).

    Positive effective weights are routed to excitatory channel, non-positive
    to inhibitory channel, matching NEST buffer routing.

    Parameters
    ----------

    ==================== ================== ==========================================================
    **Parameter**        **Default**        **Description**
    ==================== ================== ==========================================================
    ``E_L``              -70 mV             Resting potential
    ``C_m``              250 pF             Membrane capacitance
    ``tau_m``            10 ms              Membrane time constant
    ``t_ref``            2 ms               Absolute refractory duration
    ``V_th``             -55 mV             Spike threshold
    ``V_reset``          -70 mV             Reset potential
    ``tau_syn_ex``       2 ms               Excitatory synaptic time constant
    ``tau_syn_in``       2 ms               Inhibitory synaptic time constant
    ``I_e``              0 pA               Constant external current
    ``rho``              0.01 1/s           Escape-noise base rate at threshold
    ``delta``            0 mV               Escape-noise width (0 => deterministic threshold)
    ``tau_fac``          1000 ms            Facilitation decay time constant
    ``tau_psc``          2 ms               Synaptic current time constant for Tsodyks update
    ``tau_rec``          400 ms             Recovery time constant
    ``U``                0.5                Utilization increment factor
    ``x``                0.0                Initial readily-releasable fraction
    ``y``                0.0                Initial cleft fraction
    ``u``                0.0                Initial release probability
    ``V_initializer``    Constant(-70 mV)   Initial membrane potential
    ``ref_var``          ``False``          If True, expose boolean refractory flag
    ==================== ================== ==========================================================

    Notes
    -----

    - Uses the same exact exponential propagator implementation as
      :class:`iaf_psc_exp`.
    - ``x`` and ``y`` must satisfy ``x + y <= 1`` and ``u`` must be in ``[0,1]``.
    - The model is grid-based and follows NEST-style one-step input buffering.
    """

    __module__ = 'brainpy.state'

    RECEPTOR_TYPES = {
        'DEFAULT': 0,
        'TSODYKS': 1,
    }

    RECORDABLES = (
        'V_m',
        'I_syn_ex',
        'I_syn_in',
        'x',
        'y',
        'u',
        'spike_offset',
    )

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * bu.mV,
        C_m: ArrayLike = 250. * bu.pF,
        tau_m: ArrayLike = 10. * bu.ms,
        t_ref: ArrayLike = 2. * bu.ms,
        V_th: ArrayLike = -55. * bu.mV,
        V_reset: ArrayLike = -70. * bu.mV,
        tau_syn_ex: ArrayLike = 2. * bu.ms,
        tau_syn_in: ArrayLike = 2. * bu.ms,
        I_e: ArrayLike = 0. * bu.pA,
        rho: ArrayLike = 0.01 / bu.second,
        delta: ArrayLike = 0. * bu.mV,
        tau_fac: ArrayLike = 1000. * bu.ms,
        tau_psc: ArrayLike = 2. * bu.ms,
        tau_rec: ArrayLike = 400. * bu.ms,
        U: ArrayLike = 0.5,
        x: ArrayLike = 0.0,
        y: ArrayLike = 0.0,
        u: ArrayLike = 0.0,
        V_initializer: Callable = braintools.init.Constant(-70. * bu.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.rho = braintools.init.param(rho, self.varshape)
        self.delta = braintools.init.param(delta, self.varshape)

        self.tau_fac = braintools.init.param(tau_fac, self.varshape)
        self.tau_psc = braintools.init.param(tau_psc, self.varshape)
        self.tau_rec = braintools.init.param(tau_rec, self.varshape)
        self.U = braintools.init.param(U, self.varshape)
        self.x_init = braintools.init.param(x, self.varshape)
        self.y_init = braintools.init.param(y, self.varshape)
        self.u_init = braintools.init.param(u, self.varshape)

        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @property
    def receptor_types(self):
        return dict(self.RECEPTOR_TYPES)

    @property
    def recordables(self):
        return list(self.RECORDABLES)

    @staticmethod
    def _to_numpy(x, unit=None):
        if unit is None:
            return np.asarray(bu.math.asarray(x), dtype=np.float64)
        try:
            return np.asarray(bu.math.asarray(x / unit), dtype=np.float64)
        except Exception:
            return np.asarray(bu.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @classmethod
    def _normalize_spike_receptor(cls, receptor):
        if isinstance(receptor, str):
            key = receptor.strip().upper()
            if key in ('DEFAULT', 'R0', 'RECEPTOR0', '0'):
                return 0
            if key in ('TSODYKS', 'R1', 'RECEPTOR1', '1'):
                return 1
            if key.isdigit():
                receptor = int(key)
            else:
                raise ValueError(f'Unknown receptor label: {receptor}')

        receptor = int(receptor)
        if receptor not in (0, 1):
            raise ValueError(f'Receptor type must be 0 or 1, got {receptor}.')
        return receptor

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.V_reset, bu.mV) >= self._to_numpy(self.V_th, bu.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._to_numpy(self.C_m, bu.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.tau_m, bu.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_ex, bu.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, bu.ms) <= 0.0):
            raise ValueError('Synaptic time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_psc, bu.ms) <= 0.0) or np.any(self._to_numpy(self.tau_rec, bu.ms) <= 0.0):
            raise ValueError('Tsodyks time constants tau_psc and tau_rec must be strictly positive.')
        if np.any(self._to_numpy(self.tau_fac, bu.ms) < 0.0):
            raise ValueError("'tau_fac' must be >= 0.")
        if np.any(self._to_numpy(self.t_ref, bu.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')
        if np.any(self._to_numpy(self.U, None) < 0.0) or np.any(self._to_numpy(self.U, None) > 1.0):
            raise ValueError("'U' must be in [0,1].")
        if np.any(self._to_numpy(self.rho, 1 / bu.second) < 0.0):
            raise ValueError('Stochastic firing intensity rho must not be negative.')
        if np.any(self._to_numpy(self.delta, bu.mV) < 0.0):
            raise ValueError('Threshold width delta must not be negative.')

        x0 = self._to_numpy(self.x_init, None)
        y0 = self._to_numpy(self.y_init, None)
        u0 = self._to_numpy(self.u_init, None)
        if np.any(x0 + y0 > 1.0):
            raise ValueError('x + y must be <= 1.0.')
        if np.any((u0 < 0.0) | (u0 > 1.0)):
            raise ValueError("'u' must be in [0,1].")

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        state_shape = V.shape
        zeros = np.zeros(state_shape, dtype=np.float64)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * bu.ms), self.varshape, batch_size)

        x0 = np.broadcast_to(self._to_numpy(self.x_init, None), state_shape).copy()
        y0 = np.broadcast_to(self._to_numpy(self.y_init, None), state_shape).copy()
        u0 = np.broadcast_to(self._to_numpy(self.u_init, None), state_shape).copy()

        self.V = brainstate.HiddenState(V)
        self.i_syn_ex = brainstate.ShortTermState(zeros * bu.pA)
        self.i_syn_in = brainstate.ShortTermState(zeros * bu.pA)
        self.i_0 = brainstate.ShortTermState(zeros * bu.pA)
        self.i_1 = brainstate.ShortTermState(zeros * bu.pA)
        self.refractory_step_count = brainstate.ShortTermState(bu.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        self.x = brainstate.ShortTermState(x0)
        self.y = brainstate.ShortTermState(y0)
        self.u = brainstate.ShortTermState(u0)
        self.spike_offset = brainstate.ShortTermState(zeros.copy())

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(bu.math.asarray(ref_steps > 0, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return bu.math.asarray(bu.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _parse_spike_events(self, spike_events: Iterable, state_shape):
        w_ex = np.zeros(state_shape, dtype=np.float64)
        w_in = np.zeros(state_shape, dtype=np.float64)

        if spike_events is None:
            return w_ex, w_in

        for ev in spike_events:
            sender_model = 'iaf_tum_2000'
            multiplicity = 1.0
            offset = 1.0

            if isinstance(ev, dict):
                receptor = ev.get('receptor_type', ev.get('receptor', 0))
                weight = ev.get('weight', 0.0 * bu.pA)
                offset = ev.get('offset', 1.0)
                multiplicity = ev.get('multiplicity', 1.0)
                sender_model = ev.get('sender_model', 'iaf_tum_2000')
            else:
                if len(ev) == 2:
                    receptor, weight = ev
                elif len(ev) == 3:
                    receptor, weight, offset = ev
                elif len(ev) == 4:
                    receptor, weight, offset, multiplicity = ev
                elif len(ev) == 5:
                    receptor, weight, offset, multiplicity, sender_model = ev
                else:
                    raise ValueError('Spike event tuples must have length 2, 3, 4, or 5.')

            receptor_id = self._normalize_spike_receptor(receptor)
            s = np.broadcast_to(self._to_numpy(weight, bu.pA), state_shape)
            s = s * np.broadcast_to(self._to_numpy(multiplicity, None), state_shape)

            if receptor_id == 1:
                if sender_model != 'iaf_tum_2000':
                    raise ValueError(
                        'For receptor_type 1 in iaf_tum_2000, pre-synaptic neuron must also be of type iaf_tum_2000.'
                    )
                s = s * np.broadcast_to(self._to_numpy(offset, None), state_shape)

            w_ex += np.where(s > 0.0, s, 0.0)
            w_in += np.where(s > 0.0, 0.0, s)

        return w_ex, w_in

    def _parse_registered_spike_inputs(self, state_shape):
        w_ex = np.zeros(state_shape, dtype=np.float64)
        w_in = np.zeros(state_shape, dtype=np.float64)

        if self.delta_inputs is None:
            return w_ex, w_in

        for key in tuple(self.delta_inputs.keys()):
            val = self.delta_inputs[key]
            if callable(val):
                val = val()
            else:
                self.delta_inputs.pop(key)

            label = None
            if ' // ' in key:
                label, _ = key.split(' // ', maxsplit=1)
            receptor = 0 if label is None else self._normalize_spike_receptor(label)

            s = np.broadcast_to(self._to_numpy(val, bu.pA), state_shape)
            if receptor == 0:
                w_ex += np.where(s > 0.0, s, 0.0)
                w_in += np.where(s > 0.0, 0.0, s)
            else:
                w_ex += np.where(s > 0.0, s, 0.0)
                w_in += np.where(s > 0.0, 0.0, s)

        return w_ex, w_in

    def update(self, x=0. * bu.pA, x_filtered=0. * bu.pA, spike_events=None):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(bu.math.asarray(dt_q / bu.ms))
        t_ms = float(bu.math.asarray(t / bu.ms))

        state_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, bu.mV), state_shape)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, bu.mV), state_shape) - E_L
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, bu.pF), state_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, bu.ms), state_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, bu.ms), state_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, bu.ms), state_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, bu.pA), state_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, bu.mV), state_shape)
        V_reset_rel = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, bu.mV), state_shape)
        rho = self._broadcast_to_state(self._to_numpy(self.rho, 1 / bu.second), state_shape)
        delta = self._broadcast_to_state(self._to_numpy(self.delta, bu.mV), state_shape)

        tau_fac = self._broadcast_to_state(self._to_numpy(self.tau_fac, bu.ms), state_shape)
        tau_psc = self._broadcast_to_state(self._to_numpy(self.tau_psc, bu.ms), state_shape)
        tau_rec = self._broadcast_to_state(self._to_numpy(self.tau_rec, bu.ms), state_shape)
        U = self._broadcast_to_state(self._to_numpy(self.U, None), state_shape)

        i_0 = self._broadcast_to_state(self._to_numpy(self.i_0.value, bu.pA), state_shape)
        i_1 = self._broadcast_to_state(self._to_numpy(self.i_1.value, bu.pA), state_shape)
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value, bu.pA), state_shape)
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.i_syn_in.value, bu.pA), state_shape)
        r = self._broadcast_to_state(
            np.asarray(bu.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            state_shape,
        )

        x_state = self._broadcast_to_state(self._to_numpy(self.x.value, None), state_shape)
        y_state = self._broadcast_to_state(self._to_numpy(self.y.value, None), state_shape)
        u_state = self._broadcast_to_state(self._to_numpy(self.u.value, None), state_shape)
        last_spike_prev = self._broadcast_to_state(self._to_numpy(self.last_spike_time.value, bu.ms), state_shape)

        P11_ex = np.exp(-h / tau_ex)
        P11_in = np.exp(-h / tau_in)
        P22 = np.exp(-h / tau_m)
        P21_ex = iaf_psc_exp._propagator_exp(tau_ex, tau_m, C_m, h)
        P21_in = iaf_psc_exp._propagator_exp(tau_in, tau_m, C_m, h)
        P20 = tau_m / C_m * (1.0 - P22)

        ev_ex, ev_in = self._parse_spike_events(spike_events, state_shape)
        reg_ex, reg_in = self._parse_registered_spike_inputs(state_shape)
        w_ex = ev_ex + reg_ex
        w_in = ev_in + reg_in

        i_0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), bu.pA), state_shape)
        i_1_next = self._broadcast_to_state(self._to_numpy(x_filtered, bu.pA), state_shape)

        not_refractory = r == 0
        V_candidate = V_rel * P22 + i_syn_ex * P21_ex + i_syn_in * P21_in + (I_e + i_0) * P20
        V_rel = np.where(not_refractory, V_candidate, V_rel)
        r = np.where(not_refractory, r, r - 1)

        i_syn_ex = i_syn_ex * P11_ex
        i_syn_in = i_syn_in * P11_in
        i_syn_ex = i_syn_ex + (1.0 - P11_ex) * i_1
        i_syn_ex = i_syn_ex + w_ex
        i_syn_in = i_syn_in + w_in

        deterministic = delta < 1e-10
        det_spike = V_rel >= theta
        phi = rho * np.exp((V_rel - theta) / np.where(deterministic, 1.0, delta))
        stoch_spike = np.random.random(size=state_shape) < phi * h * 1e-3
        spike_cond = np.where(deterministic, det_spike, stoch_spike)

        refr_counts = self._broadcast_to_state(
            np.asarray(bu.math.asarray(self._refractory_counts()), dtype=np.int32),
            state_shape,
        )
        r = np.where(spike_cond, refr_counts, r)
        V_before_reset = V_rel
        V_rel = np.where(spike_cond, V_reset_rel, V_rel)

        t_last = np.where(last_spike_prev < 0.0, 0.0, last_spike_prev)
        t_spike = t_ms + h
        h_tsodyks = t_spike - t_last

        tau_fac_safe = np.where(tau_fac == 0.0, 1.0, tau_fac)
        Puu = np.where(tau_fac == 0.0, 0.0, np.exp(-h_tsodyks / tau_fac_safe))
        Pyy = np.exp(-h_tsodyks / tau_psc)
        Pzz = np.expm1(-h_tsodyks / tau_rec)
        Pxy = (Pzz * tau_rec - (Pyy - 1.0) * tau_psc) / (tau_psc - tau_rec)

        z_state = 1.0 - x_state - y_state
        u_prop = u_state * Puu
        x_prop = x_state + Pxy * y_state - Pzz * z_state
        y_prop = y_state * Pyy

        u_jump = u_prop + U * (1.0 - u_prop)
        delta_y_tsp = u_jump * x_prop
        x_new = x_prop - delta_y_tsp
        y_new = y_prop + delta_y_tsp

        x_state = np.where(spike_cond, x_new, x_state)
        y_state = np.where(spike_cond, y_new, y_state)
        u_state = np.where(spike_cond, u_jump, u_state)
        spike_offset = np.where(spike_cond, delta_y_tsp, 0.0)
        last_spike_next = np.where(spike_cond, t_spike, last_spike_prev)

        self.V.value = (V_rel + E_L) * bu.mV
        self.i_syn_ex.value = i_syn_ex * bu.pA
        self.i_syn_in.value = i_syn_in * bu.pA
        self.i_0.value = i_0_next * bu.pA
        self.i_1.value = i_1_next * bu.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_next * bu.ms)

        self.x.value = x_state
        self.y.value = y_state
        self.u.value = u_state
        self.spike_offset.value = spike_offset

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        V_out = np.where(spike_cond, theta + E_L + 1e-12, V_before_reset + E_L)
        return self.get_spike(V_out * bu.mV)
