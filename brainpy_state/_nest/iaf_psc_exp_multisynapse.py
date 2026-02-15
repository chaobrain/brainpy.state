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
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron
from .iaf_psc_exp import iaf_psc_exp

__all__ = [
    'iaf_psc_exp_multisynapse',
]


class iaf_psc_exp_multisynapse(Neuron):
    r"""NEST-compatible ``iaf_psc_exp_multisynapse``.

    Leaky integrate-and-fire neuron with exponential PSCs and an arbitrary
    number of receptor ports. This is the multisynapse extension of
    :class:`iaf_psc_exp`, equivalent to NEST ``iaf_psc_exp_multisynapse``.

    **State equations**

    For each receptor :math:`k`, synaptic current follows

    .. math::

       \frac{dI_k}{dt} = -\frac{I_k}{\tau_{\mathrm{syn},k}}.

    The membrane equation is

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m}
       + \frac{\sum_k I_k + I_e + I_0}{C_m},

    with hard reset and a fixed refractory period.

    **NEST event semantics**

    - ``tau_syn`` defines receptor count and time constants.
    - ``spike_events`` are passed as ``(receptor_type, weight)`` with
      1-based receptor indexing (same as NEST).
    - Default delta-input stream is mapped to receptor 1.
    - ``update(x=...)`` uses one-step delayed current buffering, matching
      NEST current-event ordering.

    **Per-step update order**

    1. Propagate membrane (if not refractory) using exact exponential kernels.
    2. Decay receptor currents.
    3. Add incoming receptor-weighted spikes.
    4. Test threshold, apply reset/refractory, store spike time.
    5. Buffer continuous current input for the next step.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 250. * u.pF,
        tau_m: ArrayLike = 10. * u.ms,
        t_ref: ArrayLike = 2. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -70. * u.mV,
        tau_syn: ArrayLike = (2.0,) * u.ms,
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
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.tau_syn = np.asarray(u.math.asarray(tau_syn / u.ms), dtype=np.float64).reshape(-1)
        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @property
    def n_receptors(self):
        return int(self.tau_syn.size)

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
            raise ValueError('Capacitance must be > 0.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self.tau_syn <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive.')
        if np.any(np.isclose(self.tau_syn, self._to_numpy(self.tau_m, u.ms))):
            raise ValueError('Membrane and synapse time constants must differ.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = np.zeros(V.shape + (self.n_receptors,), dtype=np.float64)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.i_syn = brainstate.ShortTermState(zeros * u.pA)
        self.i_const = brainstate.ShortTermState(np.zeros(V.shape, dtype=np.float64) * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(u.math.asarray(ref_steps > 0, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        out = np.zeros(v_shape + (self.n_receptors,), dtype=np.float64)
        if spike_events is None:
            return out
        for ev in spike_events:
            if isinstance(ev, dict):
                receptor = int(ev.get('receptor_type', ev.get('receptor', 1)))
                weight = ev.get('weight', 0.0)
            else:
                receptor, weight = ev
                receptor = int(receptor)
            if receptor < 1 or receptor > self.n_receptors:
                raise ValueError(f'Receptor type {receptor} out of range [1, {self.n_receptors}].')
            w_np = np.asarray(u.math.asarray(weight / u.pA), dtype=np.float64)
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    def update(self, x=0. * u.pA, spike_events=None):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))
        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        V_reset_rel = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)

        i_syn = np.asarray(u.math.asarray(self.i_syn.value / u.pA), dtype=np.float64)
        i_const = self._broadcast_to_state(self._to_numpy(self.i_const.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )

        P22 = np.exp(-h / tau_m)
        P20 = tau_m / C_m * (1.0 - P22)
        P11_syn = np.exp(-h / self.tau_syn)
        P21_syn = np.stack([
            iaf_psc_exp._propagator_exp(tau_s * np.ones(v_shape), tau_m, C_m, h) for tau_s in self.tau_syn
        ], axis=-1)

        w_by_rec = self._parse_spike_events(spike_events, v_shape)
        w_default = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        if self.n_receptors > 0:
            w_by_rec[..., 0] += w_default
        i_const_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        if np.any(r == 0):
            V_candidate = V_rel * P22 + (I_e + i_const) * P20 + np.sum(P21_syn * i_syn, axis=-1)
            V_rel = np.where(r == 0, V_candidate, V_rel)
        r = np.where(r == 0, r, r - 1)

        i_syn = i_syn * P11_syn
        i_syn = i_syn + w_by_rec

        spike_cond = V_rel >= theta
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )
        r = np.where(spike_cond, refr_counts, r)
        V_before_reset = V_rel
        V_rel = np.where(spike_cond, V_reset_rel, V_rel)

        self.V.value = (V_rel + E_L) * u.mV
        self.i_syn.value = i_syn * u.pA
        self.i_const.value = i_const_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        V_out = np.where(spike_cond, theta + E_L + 1e-12, V_before_reset + E_L)
        return self.get_spike(V_out * u.mV)
