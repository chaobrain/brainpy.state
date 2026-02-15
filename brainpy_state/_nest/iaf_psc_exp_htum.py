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
from .iaf_psc_exp import iaf_psc_exp

__all__ = [
    'iaf_psc_exp_htum',
]


class iaf_psc_exp_htum(Neuron):
    r"""NEST-compatible ``iaf_psc_exp_htum``.

    Exponential-PSC LIF neuron with two refractory times, equivalent to
    NEST ``iaf_psc_exp_htum``:

    - ``t_ref_abs``: absolute refractory period where the membrane is clamped,
    - ``t_ref_tot``: total refractory period where new spikes are suppressed.

    During ``t_ref_abs`` membrane integration is skipped; during the remaining
    ``t_ref_tot - t_ref_abs`` interval, membrane integration resumes but
    threshold crossings are ignored.

    **Dynamics**

    Synaptic currents follow exponential decay for excitatory and inhibitory
    channels. Membrane integration uses the exact discrete propagator, with
    one-step delayed continuous current buffering, matching NEST's update order.

    **Per-step update order**

    1. Update membrane only if absolute refractory counter is zero.
    2. Decay synaptic currents and add arriving spikes.
    3. Evaluate threshold only if total refractory counter is zero.
    4. On spike, set both refractory counters and reset membrane.
    5. Buffer current input for the next step.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 250. * u.pF,
        tau_m: ArrayLike = 10. * u.ms,
        t_ref_abs: ArrayLike = 2. * u.ms,
        t_ref_tot: ArrayLike = 2. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -70. * u.mV,
        tau_syn_ex: ArrayLike = 2. * u.ms,
        tau_syn_in: ArrayLike = 2. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.t_ref_abs = braintools.init.param(t_ref_abs, self.varshape)
        self.t_ref_tot = braintools.init.param(t_ref_tot, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_initializer = V_initializer
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
        if np.any(self._to_numpy(self.t_ref_abs, u.ms) <= 0.0) or np.any(self._to_numpy(self.t_ref_tot, u.ms) <= 0.0):
            raise ValueError('All refractory time constants must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref_abs, u.ms) > self._to_numpy(self.t_ref_tot, u.ms)):
            raise ValueError('Total refractory period must be >= absolute refractory period.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('Synaptic time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.i_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.i_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.i_0 = brainstate.ShortTermState(zeros * u.pA)
        self.refractory_abs_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.refractory_tot_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(u.math.asarray(ref_steps > 0, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_abs_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref_abs / dt), dtype=jnp.int32)

    def _refractory_tot_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref_tot / dt), dtype=jnp.int32)

    def update(self, x=0. * u.pA):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))
        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        V_reset_rel = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)

        i_0 = self._broadcast_to_state(self._to_numpy(self.i_0.value, u.pA), v_shape)
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value, u.pA), v_shape)
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.i_syn_in.value, u.pA), v_shape)
        r_abs = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_abs_step_count.value), dtype=np.int32), v_shape
        )
        r_tot = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_tot_step_count.value), dtype=np.int32), v_shape
        )

        P11_ex = np.exp(-h / tau_ex)
        P11_in = np.exp(-h / tau_in)
        P22 = np.exp(-h / tau_m)
        P21_ex = iaf_psc_exp._propagator_exp(tau_ex, tau_m, C_m, h)
        P21_in = iaf_psc_exp._propagator_exp(tau_in, tau_m, C_m, h)
        P20 = tau_m / C_m * (1.0 - P22)

        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all >= 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)
        i_0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        if np.any(r_abs == 0):
            V_candidate = V_rel * P22 + i_syn_ex * P21_ex + i_syn_in * P21_in + (I_e + i_0) * P20
            V_rel = np.where(r_abs == 0, V_candidate, V_rel)
        r_abs = np.where(r_abs == 0, r_abs, r_abs - 1)

        i_syn_ex = i_syn_ex * P11_ex + w_ex
        i_syn_in = i_syn_in * P11_in + w_in

        can_spike = r_tot == 0
        spike_cond = can_spike & (V_rel >= theta)
        refr_abs = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_abs_counts()), dtype=np.int32), v_shape
        )
        refr_tot = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_tot_counts()), dtype=np.int32), v_shape
        )
        r_abs = np.where(spike_cond, refr_abs, r_abs)
        r_tot = np.where(spike_cond, refr_tot, np.where(r_tot > 0, r_tot - 1, r_tot))
        V_before_reset = V_rel
        V_rel = np.where(spike_cond, V_reset_rel, V_rel)

        self.V.value = (V_rel + E_L) * u.mV
        self.i_syn_ex.value = i_syn_ex * u.pA
        self.i_syn_in.value = i_syn_in * u.pA
        self.i_0.value = i_0_next * u.pA
        self.refractory_abs_step_count.value = jnp.asarray(r_abs, dtype=jnp.int32)
        self.refractory_tot_step_count.value = jnp.asarray(r_tot, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_tot_step_count.value > 0)

        # Emit spikes only on actual threshold events (respecting total refractory).
        V_nospike = np.minimum(V_before_reset, theta - 1e-12)
        V_out = np.where(spike_cond, theta + E_L + 1e-12, V_nospike + E_L)
        return self.get_spike(V_out * u.mV)
