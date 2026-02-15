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

from typing import Callable

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'iaf_cond_alpha',
]


class iaf_cond_alpha(Neuron):
    r"""NEST-compatible ``iaf_cond_alpha`` neuron model.

    Short description
    -----------------

    Conductance-based leaky integrate-and-fire neuron with alpha-shaped
    synaptic conductances.

    Description
    -----------

    ``iaf_cond_alpha`` follows NEST ``models/iaf_cond_alpha.{h,cpp}`` with:

    - hard threshold crossing,
    - fixed absolute refractory period,
    - alpha-shaped excitatory and inhibitory conductance dynamics,
    - one-step delayed external current buffering.

    Membrane and synaptic dynamics
    ..............................

    With membrane voltage :math:`V_m`, the subthreshold dynamics are:

    .. math::

       \frac{dV_m}{dt} =
       \frac{-I_\mathrm{leak} - I_{\mathrm{syn,ex}} - I_{\mathrm{syn,in}}
             + I_e + I_\mathrm{stim}}{C_m},

    where

    .. math::

       I_\mathrm{leak} = g_L (V - E_L),\quad
       I_{\mathrm{syn,ex}} = g_\mathrm{ex}(V - E_\mathrm{ex}),\quad
       I_{\mathrm{syn,in}} = g_\mathrm{in}(V - E_\mathrm{in}).

    During refractory period, NEST clamps the effective voltage to
    :math:`V_\mathrm{reset}` and sets :math:`dV_m/dt = 0`.
    Otherwise, voltage is bounded by :math:`\min(V_m, V_\mathrm{th})` in
    the RHS.

    Alpha conductance states use two coupled variables per channel:

    .. math::

       \frac{d\,dg_\mathrm{ex}}{dt} = -\frac{dg_\mathrm{ex}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{dg_\mathrm{in}}{dt} = -\frac{dg_\mathrm{in}}{\tau_{\mathrm{syn,in}}},

    .. math::

       \frac{d g_\mathrm{ex}}{dt}
       = dg_\mathrm{ex} - \frac{g_\mathrm{ex}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{d g_\mathrm{in}}{dt}
       = dg_\mathrm{in} - \frac{g_\mathrm{in}}{\tau_{\mathrm{syn,in}}}.

    Incoming spike weights are interpreted in nS (conductance peak units).
    Positive/negative weights map to excitatory/inhibitory channels and are
    converted to :math:`dg` jumps by NEST's normalization:

    .. math::

       dg_\mathrm{ex} \leftarrow dg_\mathrm{ex} + \frac{e}{\tau_{\mathrm{syn,ex}}} w_+,
       \qquad
       dg_\mathrm{in} \leftarrow dg_\mathrm{in} + \frac{e}{\tau_{\mathrm{syn,in}}} |w_-|.

    Update order (NEST semantics)
    .............................

    Per simulation step:

    1. Integrate ODEs on :math:`(t, t+dt]` using RKF45 with adaptive substeps.
    2. Apply refractory countdown / threshold test / reset / spike emission.
    3. Apply arriving spike weights to ``dg_ex`` / ``dg_in``.
    4. Store external current input ``x`` into delayed buffer ``I_stim``.

    This matches NEST ring-buffer semantics: current input affects the next step.

    Parameters
    ----------

    ==================== ================== ======================================== ================================================
    **Parameter**        **Default**        **Math equivalent**                      **Description**
    ==================== ================== ======================================== ================================================
    ``in_size``          (required)                                                   Population shape
    ``E_L``              -70 mV             :math:`E_\mathrm{L}`                     Leak reversal potential
    ``C_m``              250 pF             :math:`C_\mathrm{m}`                     Membrane capacitance
    ``t_ref``            2 ms               :math:`t_\mathrm{ref}`                   Absolute refractory duration
    ``V_th``             -55 mV             :math:`V_\mathrm{th}`                    Spike threshold
    ``V_reset``          -60 mV             :math:`V_\mathrm{reset}`                 Reset potential
    ``E_ex``             0 mV               :math:`E_\mathrm{ex}`                    Excitatory reversal potential
    ``E_in``             -85 mV             :math:`E_\mathrm{in}`                    Inhibitory reversal potential
    ``g_L``              16.6667 nS         :math:`g_\mathrm{L}`                     Leak conductance
    ``tau_syn_ex``       0.2 ms             :math:`\tau_{\mathrm{syn,ex}}`          Excitatory alpha time constant
    ``tau_syn_in``       2.0 ms             :math:`\tau_{\mathrm{syn,in}}`          Inhibitory alpha time constant
    ``I_e``              0 pA               :math:`I_\mathrm{e}`                     Constant external current
    ``V_initializer``    Constant(-70 mV)                                             Membrane initializer
    ``g_ex_initializer`` Constant(0 nS)                                               Excitatory conductance initializer
    ``g_in_initializer`` Constant(0 nS)                                               Inhibitory conductance initializer
    ``spk_fun``          ReluGrad()                                                   Surrogate spike function
    ``spk_reset``        ``'hard'``                                                   Reset mode; hard reset matches NEST behavior
    ``ref_var``          ``False``                                                    If True, expose boolean refractory indicator
    ==================== ================== ======================================== ================================================

    State variables
    ---------------

    - ``V``: membrane potential :math:`V_m`.
    - ``dg_ex`` / ``dg_in``: alpha auxiliary conductance states.
    - ``g_ex`` / ``g_in``: excitatory/inhibitory conductances.
    - ``refractory_step_count``: remaining refractory grid steps.
    - ``integration_step``: persistent RKF45 internal step size.
    - ``I_stim``: one-step delayed current buffer.
    - ``last_spike_time``: last emitted spike time (:math:`t+dt` on spike).
    - ``refractory``: optional boolean refractory indicator.

    References
    ----------
    .. [1] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. Journal of Computational Neuroscience,
           16:159-175. DOI: https://doi.org/10.1023/B:JCNS.0000014108.03012.81
    .. [2] Bernander O, Douglas RJ, Martin KAC, Koch C (1991). Synaptic
           background activity influences spatiotemporal integration in single
           pyramidal cells. PNAS, 88(24):11569-11573.
           DOI: https://doi.org/10.1073/pnas.88.24.11569
    .. [3] Kuhn A, Rotter S (2004). Neuronal integration of synaptic input in
           the fluctuation-driven regime. Journal of Neuroscience, 24(10):2345-2356.
           DOI: https://doi.org/10.1523/JNEUROSCI.3349-03.2004
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
            raise ValueError('All time constants must be strictly positive.')

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))

        self.V = brainstate.HiddenState(V)
        self.dg_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.g_ex = brainstate.HiddenState(g_ex)
        self.dg_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
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
        zeros = u.math.zeros_like(u.math.asarray(self.V.value / u.mV))
        self.dg_ex.value = np.asarray(zeros, dtype=np.float64)
        self.dg_in.value = np.asarray(zeros, dtype=np.float64)
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
        w_ex = u.math.zeros_like(self.g_ex.value)
        w_in = u.math.zeros_like(self.g_in.value)
        if self.delta_inputs is None:
            return w_ex, w_in

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            zero = u.math.zeros_like(out)
            w_ex = w_ex + u.math.maximum(out, zero)
            w_in = w_in + u.math.maximum(-out, zero)
        return w_ex, w_in

    @staticmethod
    def _dynamics_scalar(v, dg_ex, g_ex, dg_in, g_in, is_refractory, i_stim, p):
        v_eff = p['V_reset'] if is_refractory else min(v, p['V_th'])

        i_syn_exc = g_ex * (v_eff - p['E_ex'])
        i_syn_inh = g_in * (v_eff - p['E_in'])
        i_leak = p['g_L'] * (v_eff - p['E_L'])
        dv = 0.0 if is_refractory else (
            -i_leak - i_syn_exc - i_syn_inh + i_stim + p['I_e']
        ) / p['C_m']

        ddg_ex = -dg_ex / p['tau_syn_ex']
        dg_ex_dt = dg_ex - (g_ex / p['tau_syn_ex'])
        ddg_in = -dg_in / p['tau_syn_in']
        dg_in_dt = dg_in - (g_in / p['tau_syn_in'])
        return dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt

    def _rkf45_integrate_scalar(self, v0, dg_ex0, g_ex0, dg_in0, g_in0, is_refractory, i_stim, h0, dt, p):
        t = 0.0
        h = max(h0, self._MIN_H)
        y = np.asarray([v0, dg_ex0, g_ex0, dg_in0, g_in0], dtype=np.float64)
        iters = 0

        def f(y_):
            return np.asarray(
                self._dynamics_scalar(
                    y_[0], y_[1], y_[2], y_[3], y_[4], is_refractory, i_stim, p
                ),
                dtype=np.float64
            )

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = max(self._MIN_H, min(h, dt - t))

            k1 = f(y)
            k2 = f(y + h * (1.0 / 4.0) * k1)
            k3 = f(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
            k4 = f(y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0))
            k5 = f(y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0))
            k6 = f(y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0))

            y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
            y5 = y + h * (16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0)
            err = float(np.max(np.abs(y5 - y4)))

            if err <= self._ATOL or h <= self._MIN_H:
                y = y5
                t += h
                fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (self._ATOL / err) ** 0.2))
                h = max(self._MIN_H, h * fac)
            else:
                fac = min(1.0, max(0.2, 0.9 * (self._ATOL / err) ** 0.25))
                h = max(self._MIN_H, h * fac)

        return y[0], y[1], y[2], y[3], y[4], h

    def update(self, x=0. * u.pA):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        dg_ex = self._broadcast_to_state(np.asarray(self.dg_ex.value, dtype=np.float64), v_shape)
        g_ex = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape)
        dg_in = self._broadcast_to_state(np.asarray(self.dg_in.value, dtype=np.float64), v_shape)
        g_in = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            v_shape
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
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )

        w_ex_q, w_in_q = self._sum_signed_delta_inputs()
        w_ex = self._broadcast_to_state(self._to_numpy(w_ex_q, u.nS), v_shape)
        w_in = self._broadcast_to_state(self._to_numpy(w_in_q, u.nS), v_shape)
        pscon_ex = self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        pscon_in = self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        v_for_spike = np.empty_like(V)
        spike_mask = np.zeros_like(V, dtype=bool)
        V_next = np.empty_like(V)
        dg_ex_next = np.empty_like(dg_ex)
        g_ex_next = np.empty_like(g_ex)
        dg_in_next = np.empty_like(dg_in)
        g_in_next = np.empty_like(g_in)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            is_refractory = r[idx] > 0
            v_i, dg_ex_i, g_ex_i, dg_in_i, g_in_i, h_i = self._rkf45_integrate_scalar(
                V[idx], dg_ex[idx], g_ex[idx], dg_in[idx], g_in[idx], is_refractory,
                i_stim[idx], h_int[idx], dt, local_p
            )

            # NEST ordering: refractory/spike handling before applying spike inputs.
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

            dg_ex_i += pscon_ex[idx] * w_ex[idx]
            dg_in_i += pscon_in[idx] * w_in[idx]

            V_next[idx] = v_i
            dg_ex_next[idx] = dg_ex_i
            g_ex_next[idx] = g_ex_i
            dg_in_next[idx] = dg_in_i
            g_in_next[idx] = g_in_i
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.dg_ex.value = dg_ex_next
        self.g_ex.value = g_ex_next * u.nS
        self.dg_in.value = dg_in_next
        self.g_in.value = g_in_next * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float32) * u.mV)
