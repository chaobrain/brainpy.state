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

import numpy as np
from typing import Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'iaf_cond_exp',
]


class iaf_cond_exp(Neuron):
    r"""Leaky integrate-and-fire model with exponential conductance synapses.

    Description
    -----------

    ``iaf_cond_exp`` is a conductance-based leaky integrate-and-fire neuron with

    * hard threshold,
    * fixed absolute refractory period,
    * exponentially decaying excitatory and inhibitory synaptic conductances,
    * no adaptation variables.

    This implementation follows NEST ``iaf_cond_exp`` dynamics and update order,
    using NEST C++ model behavior as the source of truth.

    Membrane potential, synaptic conductances, and spiking
    ......................................................

    The membrane potential evolves according to

    .. math::

       \frac{dV_\mathrm{m}}{dt} =
       \frac{-g_\mathrm{L}(V_\mathrm{m}-E_\mathrm{L})
             - I_\mathrm{syn}
             + I_\mathrm{e}
             + I_\mathrm{stim}}
            {C_\mathrm{m}}

    with

    .. math::

       I_\mathrm{syn}
       = I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}}
       = g_\mathrm{ex}(V_\mathrm{m}-E_\mathrm{ex})
       + g_\mathrm{in}(V_\mathrm{m}-E_\mathrm{in}) .

    Synaptic conductances decay exponentially:

    .. math::

       \frac{dg_\mathrm{ex}}{dt} = -\frac{g_\mathrm{ex}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{dg_\mathrm{in}}{dt} = -\frac{g_\mathrm{in}}{\tau_{\mathrm{syn,in}}}.

    A presynaptic spike with weight :math:`w` causes an instantaneous jump at
    the end of the simulation step:

    .. math::

       w > 0 \Rightarrow g_\mathrm{ex} \leftarrow g_\mathrm{ex} + w,
       \qquad
       w < 0 \Rightarrow g_\mathrm{in} \leftarrow g_\mathrm{in} + |w|.

    Spike emission and refractory mechanism
    .......................................

    A spike is emitted when :math:`V_\mathrm{m} \ge V_\mathrm{th}` at the end of
    a simulation step. On spike:

    * :math:`V_\mathrm{m}` is reset to :math:`V_\mathrm{reset}`,
    * refractory counter is set to :math:`\lceil t_\mathrm{ref}/dt \rceil`,
    * spike time is recorded as :math:`t + dt`.

    During absolute refractory period:

    * membrane potential is clamped to :math:`V_\mathrm{reset}`,
    * :math:`dV_\mathrm{m}/dt = 0`,
    * conductances continue to decay.

    Numerical integration and update order
    ......................................

    NEST integrates this model with adaptive RKF45. This implementation mirrors
    that behavior with an RKF45(4,5) integrator and persistent internal step size.
    The discrete-time update order is:

    1. Integrate continuous dynamics on :math:`(t, t+dt]`.
    2. Add synaptic conductance jumps from spike inputs arriving this step.
    3. Apply refractory countdown / threshold check / reset and spike emission.
    4. Store external current input as :math:`I_\mathrm{stim}` for the next step.

    The one-step delayed application of current input (``I_stim`` buffer) is
    intentional and matches NEST's ring-buffer update semantics.

    Parameters
    ----------

    ==================== ================== ========================================== =====================================================
    **Parameter**        **Default**        **Math equivalent**                        **Description**
    ==================== ================== ========================================== =====================================================
    ``in_size``          (required)                                                    Population shape
    ``E_L``              -70 mV             :math:`E_\mathrm{L}`                       Leak reversal potential
    ``C_m``              250 pF             :math:`C_\mathrm{m}`                       Membrane capacitance
    ``t_ref``            2 ms               :math:`t_\mathrm{ref}`                     Absolute refractory duration
    ``V_th``             -55 mV             :math:`V_\mathrm{th}`                      Spike threshold
    ``V_reset``          -60 mV             :math:`V_\mathrm{reset}`                   Reset potential
    ``E_ex``             0 mV               :math:`E_\mathrm{ex}`                      Excitatory reversal potential
    ``E_in``             -85 mV             :math:`E_\mathrm{in}`                      Inhibitory reversal potential
    ``g_L``              16.6667 nS         :math:`g_\mathrm{L}`                       Leak conductance
    ``tau_syn_ex``       0.2 ms             :math:`\tau_{\mathrm{syn,ex}}`            Excitatory conductance time constant
    ``tau_syn_in``       2.0 ms             :math:`\tau_{\mathrm{syn,in}}`            Inhibitory conductance time constant
    ``I_e``              0 pA               :math:`I_\mathrm{e}`                       Constant external current
    ``V_initializer``    Constant(-70 mV)                                              Initializer for membrane potential
    ``g_ex_initializer`` Constant(0 nS)                                                Initializer for excitatory conductance
    ``g_in_initializer`` Constant(0 nS)                                                Initializer for inhibitory conductance
    ``spk_fun``          ReluGrad()                                                    Surrogate spike function
    ``spk_reset``        ``'hard'``                                                    Reset mode; hard reset matches NEST behavior
    ``ref_var``          ``False``                                                     If True, expose boolean refractory state
    ==================== ================== ========================================== =====================================================

    State Variables
    ---------------

    ======================== ===========================================
    **State variable**       **Description**
    ======================== ===========================================
    ``V``                    Membrane potential :math:`V_\mathrm{m}`
    ``g_ex``                 Excitatory conductance :math:`g_\mathrm{ex}`
    ``g_in``                 Inhibitory conductance :math:`g_\mathrm{in}`
    ``last_spike_time``      Last spike time
    ``refractory_step_count``Remaining refractory grid steps
    ``integration_step``     Internal RKF45 step-size state
    ``I_stim``               Buffered current applied in next step
    ``refractory``           Optional boolean refractory indicator (if ``ref_var=True``)
    ======================== ===========================================

    Notes
    -----

    - Defaults follow NEST C++ source for ``iaf_cond_exp``.
    - In NEST docs, some printed default values may differ from the source for
      specific releases; source code behavior is used here for parity.
    - Synaptic spike weights are interpreted in conductance units (nS), with
      positive/negative sign selecting excitatory/inhibitory channel.

    References
    ----------
    .. [1] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. Journal of Computational Neuroscience
           16:159-175. DOI: https://doi.org/10.1023/B:JCNS.0000014108.03012.81
    .. [2] NEST Simulator ``iaf_cond_exp`` model documentation and C++ source:
           ``models/iaf_cond_exp.h`` and ``models/iaf_cond_exp.cpp``.

    See Also
    --------
    iaf_psc_delta
    LIF
    LIFRef
    """
    __module__ = 'brainpy.state'

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 10000

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 250. * u.pF,
        t_ref: ArrayLike = 2. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -60. * u.mV,
        E_ex: ArrayLike = 0. * u.mV,
        E_in: ArrayLike = -85. * u.mV,
        g_L: ArrayLike = 16.6667 * u.nS,
        tau_syn_ex: ArrayLike = 0.2 * u.ms,
        tau_syn_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        g_ex_initializer: Callable = braintools.init.Constant(0. * u.nS),
        g_in_initializer: Callable = braintools.init.Constant(0. * u.nS),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive.')

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)

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

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.g_ex.value = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        self.g_in.value = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
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

    def _sum_signed_delta_inputs(self):
        g_ex = u.math.zeros_like(self.g_ex.value)
        g_in = u.math.zeros_like(self.g_in.value)
        if self.delta_inputs is None:
            return g_ex, g_in

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            zero = u.math.zeros_like(out)
            g_ex = g_ex + u.math.maximum(out, zero)
            g_in = g_in + u.math.maximum(-out, zero)
        return g_ex, g_in

    @staticmethod
    def _dynamics_scalar(v, g_ex, g_in, is_refractory, i_stim, p):
        if is_refractory:
            return 0.0, -g_ex / p['tau_syn_ex'], -g_in / p['tau_syn_in']

        v_eff = min(v, p['V_th'])
        i_syn_exc = g_ex * (v_eff - p['E_ex'])
        i_syn_inh = g_in * (v_eff - p['E_in'])
        i_l = p['g_L'] * (v_eff - p['E_L'])

        dv = (-i_l + i_stim + p['I_e'] - i_syn_exc - i_syn_inh) / p['C_m']
        dg_ex = -g_ex / p['tau_syn_ex']
        dg_in = -g_in / p['tau_syn_in']
        return dv, dg_ex, dg_in

    def _rkf45_integrate_scalar(self, v0, ge0, gi0, is_refractory, i_stim, h0, dt, p):
        t = 0.0
        h = max(h0, self._MIN_H)
        v, ge, gi = v0, ge0, gi0
        iters = 0

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = min(h, dt - t)
            h = max(h, self._MIN_H)

            def f(y1, y2, y3):
                return self._dynamics_scalar(y1, y2, y3, is_refractory, i_stim, p)

            k1 = f(v, ge, gi)
            y2 = (v + h * k1[0] / 4.0, ge + h * k1[1] / 4.0, gi + h * k1[2] / 4.0)
            k2 = f(*y2)
            y3 = (
                v + h * (3.0 * k1[0] / 32.0 + 9.0 * k2[0] / 32.0),
                ge + h * (3.0 * k1[1] / 32.0 + 9.0 * k2[1] / 32.0),
                gi + h * (3.0 * k1[2] / 32.0 + 9.0 * k2[2] / 32.0),
            )
            k3 = f(*y3)
            y4 = (
                v + h * (1932.0 * k1[0] / 2197.0 - 7200.0 * k2[0] / 2197.0 + 7296.0 * k3[0] / 2197.0),
                ge + h * (1932.0 * k1[1] / 2197.0 - 7200.0 * k2[1] / 2197.0 + 7296.0 * k3[1] / 2197.0),
                gi + h * (1932.0 * k1[2] / 2197.0 - 7200.0 * k2[2] / 2197.0 + 7296.0 * k3[2] / 2197.0),
            )
            k4 = f(*y4)
            y5 = (
                v + h * (439.0 * k1[0] / 216.0 - 8.0 * k2[0] + 3680.0 * k3[0] / 513.0 - 845.0 * k4[0] / 4104.0),
                ge + h * (439.0 * k1[1] / 216.0 - 8.0 * k2[1] + 3680.0 * k3[1] / 513.0 - 845.0 * k4[1] / 4104.0),
                gi + h * (439.0 * k1[2] / 216.0 - 8.0 * k2[2] + 3680.0 * k3[2] / 513.0 - 845.0 * k4[2] / 4104.0),
            )
            k5 = f(*y5)
            y6 = (
                v + h * (-8.0 * k1[0] / 27.0 + 2.0 * k2[0] - 3544.0 * k3[0] / 2565.0 + 1859.0 * k4[0] / 4104.0 - 11.0 * k5[0] / 40.0),
                ge + h * (-8.0 * k1[1] / 27.0 + 2.0 * k2[1] - 3544.0 * k3[1] / 2565.0 + 1859.0 * k4[1] / 4104.0 - 11.0 * k5[1] / 40.0),
                gi + h * (-8.0 * k1[2] / 27.0 + 2.0 * k2[2] - 3544.0 * k3[2] / 2565.0 + 1859.0 * k4[2] / 4104.0 - 11.0 * k5[2] / 40.0),
            )
            k6 = f(*y6)

            y4_sol = (
                v + h * (25.0 * k1[0] / 216.0 + 1408.0 * k3[0] / 2565.0 + 2197.0 * k4[0] / 4104.0 - k5[0] / 5.0),
                ge + h * (25.0 * k1[1] / 216.0 + 1408.0 * k3[1] / 2565.0 + 2197.0 * k4[1] / 4104.0 - k5[1] / 5.0),
                gi + h * (25.0 * k1[2] / 216.0 + 1408.0 * k3[2] / 2565.0 + 2197.0 * k4[2] / 4104.0 - k5[2] / 5.0),
            )
            y5_sol = (
                v + h * (16.0 * k1[0] / 135.0 + 6656.0 * k3[0] / 12825.0 + 28561.0 * k4[0] / 56430.0 - 9.0 * k5[0] / 50.0 + 2.0 * k6[0] / 55.0),
                ge + h * (16.0 * k1[1] / 135.0 + 6656.0 * k3[1] / 12825.0 + 28561.0 * k4[1] / 56430.0 - 9.0 * k5[1] / 50.0 + 2.0 * k6[1] / 55.0),
                gi + h * (16.0 * k1[2] / 135.0 + 6656.0 * k3[2] / 12825.0 + 28561.0 * k4[2] / 56430.0 - 9.0 * k5[2] / 50.0 + 2.0 * k6[2] / 55.0),
            )

            err = max(
                abs(y5_sol[0] - y4_sol[0]),
                abs(y5_sol[1] - y4_sol[1]),
                abs(y5_sol[2] - y4_sol[2]),
            )

            if err <= self._ATOL or h <= self._MIN_H:
                v, ge, gi = y5_sol
                t += h
                if err == 0.0:
                    fac = 5.0
                else:
                    fac = 0.9 * (self._ATOL / err) ** 0.2
                    fac = min(5.0, max(0.2, fac))
                h = max(self._MIN_H, h * fac)
            else:
                fac = 0.9 * (self._ATOL / err) ** 0.25
                fac = min(1.0, max(0.2, fac))
                h = max(self._MIN_H, h * fac)

        return v, ge, gi, h

    def update(self, x=0. * u.pA):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        ge = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape)
        gi = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape)

        p = {
            'V_th': self._broadcast_to_state(self._to_numpy(self.V_th, u.mV), v_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'E_ex': self._broadcast_to_state(self._to_numpy(self.E_ex, u.mV), v_shape),
            'E_in': self._broadcast_to_state(self._to_numpy(self.E_in, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'tau_syn_ex': self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape),
            'tau_syn_in': self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
        }
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )

        dg_ex_q, dg_in_q = self._sum_signed_delta_inputs()
        dg_ex = self._broadcast_to_state(self._to_numpy(dg_ex_q, u.nS), v_shape)
        dg_in = self._broadcast_to_state(self._to_numpy(dg_in_q, u.nS), v_shape)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        v_for_spike = np.empty_like(V)
        spike_mask = np.zeros_like(V, dtype=bool)
        V_next = np.empty_like(V)
        ge_next = np.empty_like(ge)
        gi_next = np.empty_like(gi)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            is_refractory = r[idx] > 0
            v_i, ge_i, gi_i, h_i = self._rkf45_integrate_scalar(
                V[idx], ge[idx], gi[idx], is_refractory, i_stim[idx], h_int[idx], dt, local_p
            )

            ge_i += dg_ex[idx]
            gi_i += dg_in[idx]

            if is_refractory:
                v_for_spike[idx] = local_p['V_reset']
                v_i = local_p['V_reset']
                r_i = r[idx] - 1
            else:
                v_for_spike[idx] = v_i
                if v_i >= local_p['V_th']:
                    spike_mask[idx] = True
                    v_i = local_p['V_reset']
                    r_i = refr_counts[idx]
                else:
                    r_i = 0

            V_next[idx] = v_i
            ge_next[idx] = ge_i
            gi_next[idx] = gi_i
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.g_ex.value = ge_next * u.nS
        self.g_in.value = gi_next * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float32) * u.mV)
