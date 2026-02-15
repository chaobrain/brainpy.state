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
from typing import Callable, Sequence

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'iaf_chs_2007',
]


class iaf_chs_2007(Neuron):
    r"""NEST-compatible ``iaf_chs_2007`` spike-response neuron model.

    Short description
    -----------------

    Spike-response model used in Carandini, Horton, and Sincich (2007).

    Description
    -----------

    ``iaf_chs_2007`` is a discrete-time linear spike-response model where the
    normalized membrane potential is the sum of:

    - alpha-shaped postsynaptic contribution ``V_syn``,
    - post-spike reset / after-hyperpolarization contribution ``V_spike``,
    - optional externally prepared Gaussian-noise trace.

    This implementation mirrors the NEST C++ model
    ``models/iaf_chs_2007.{h,cpp}`` as closely as possible.

    Model equations and exact integration
    .....................................

    Let :math:`h = dt` in ms, and denote state at step :math:`k` by superscript.
    NEST precomputes:

    .. math::

       P_{11} = e^{-h / \tau_{\mathrm{epsp}}},
       \quad
       P_{22} = e^{-h / \tau_{\mathrm{epsp}}},
       \quad
       P_{30} = e^{-h / \tau_{\mathrm{reset}}},

    .. math::

       P_{21} = U_{\mathrm{epsp}} \, e \, P_{11}\, \frac{h}{\tau_{\mathrm{epsp}}}.

    Then each step updates:

    .. math::

       V_{\mathrm{syn}}^{k+1} = P_{22} V_{\mathrm{syn}}^k + P_{21} i_{\mathrm{syn}}^k

    .. math::

       i_{\mathrm{syn}}^{k+1} = P_{11} i_{\mathrm{syn}}^k + w_k^+,
       \quad w_k^+ = \max(w_k, 0)

    .. math::

       V_{\mathrm{spike}}^{k+1} = P_{30} V_{\mathrm{spike}}^k

    .. math::

       V_m^{k+1} = V_{\mathrm{syn}}^{k+1} + V_{\mathrm{spike}}^{k+1}
                  + U_{\mathrm{noise}} \, \eta_k.

    Spike emission uses hard threshold :math:`U_{\mathrm{th}} = 1`:

    .. math::

       V_m^{k+1} \ge 1 \Rightarrow
       V_{\mathrm{spike}}^{k+1} \leftarrow V_{\mathrm{spike}}^{k+1} - U_{\mathrm{reset}},
       \quad
       V_m^{k+1} \leftarrow V_m^{k+1} - U_{\mathrm{reset}}.

    Update ordering (NEST semantics)
    ................................

    The per-step order is identical to NEST:

    1. Update ``V_syn`` from previous ``i_syn_ex``.
    2. Decay ``i_syn_ex``.
    3. Add arriving excitatory spike weights (negative weights are ignored).
    4. Decay ``V_spike``.
    5. Add optional noise sample.
    6. Compute ``V_m`` and apply threshold/reset/spike emission.

    A key consequence is that a spike arriving in the current step updates
    ``i_syn_ex`` immediately, but affects ``V_m`` only from the next step via
    ``V_syn``.

    Noise semantics
    ...............

    NEST expects noise to be externally prepared and at least as long as the
    simulation. Here:

    - noise is used only if ``V_noise > 0`` and ``noise`` is non-empty,
    - one sample is consumed per step and per neuron state entry,
    - if the noise sequence is exhausted, an ``IndexError`` is raised.

    Parameters
    ----------

    ==================== ================== ============================================ ============================================================
    **Parameter**        **Default**        **Math equivalent**                           **Description**
    ==================== ================== ============================================ ============================================================
    ``in_size``          (required)                                                       Population shape
    ``tau_epsp``         8.5 ms             :math:`\tau_{\mathrm{epsp}}`                 EPSP time constant
    ``tau_reset``        15.4 ms            :math:`\tau_{\mathrm{reset}}`                Post-spike reset recovery time constant
    ``V_epsp``           0.77               :math:`U_{\mathrm{epsp}}`                    Normalized maximal EPSP amplitude
    ``V_reset``          2.31               :math:`U_{\mathrm{reset}}`                   Normalized reset/AHP magnitude
    ``V_noise``          0.0                :math:`U_{\mathrm{noise}}`                   Noise scale
    ``noise``            ``None``           :math:`\eta_k`                               Externally prepared noise samples
    ``V_initializer``    Constant(0.0)      :math:`V_m(0)`                               Initial membrane potential
    ``spk_fun``          ReluGrad()                                                       Surrogate spike function
    ``spk_reset``        ``'hard'``                                                       Reset mode; hard reset matches NEST behavior
    ==================== ================== ============================================ ============================================================

    State variables
    ---------------

    - ``i_syn_ex``: excitatory synaptic state.
    - ``V_syn``: EPSP waveform state.
    - ``V_spike``: post-spike reset/AHP waveform state.
    - ``V``: normalized membrane potential.
    - ``position``: current index into ``noise``.
    - ``last_spike_time``: last spike time (:math:`t + dt` on spike).

    Notes
    -----

    - Only non-negative incoming spike weights are used (NEST ignores negative
      ``SpikeEvent`` weights in this model).
    - Unlike most LIF models in this package, ``iaf_chs_2007`` has no
      refractory state and no current-event dynamics in NEST C++.
      Therefore, ``update(x=...)`` ignores ``x`` by design.

    References
    ----------
    .. [1] Carandini M, Horton JC, Sincich LC (2007). Thalamic filtering of
           retinal spike trains by postsynaptic summation. Journal of Vision
           7(14):20, 1-11. DOI: https://doi.org/10.1167/7.14.20
    .. [2] Rotter S, Diesmann M (1999). Exact simulation of time-invariant
           linear systems with applications to neuronal modeling.
           Biological Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    """

    __module__ = 'brainpy.state'

    _U_TH = 1.0  # NEST hard-coded normalized threshold.
    _E_L = 0.0   # NEST hard-coded normalized rest potential.

    def __init__(
        self,
        in_size: Size,
        tau_epsp: ArrayLike = 8.5 * u.ms,
        tau_reset: ArrayLike = 15.4 * u.ms,
        V_epsp: ArrayLike = 0.77,
        V_reset: ArrayLike = 2.31,
        V_noise: ArrayLike = 0.0,
        noise: Sequence[float] | np.ndarray | None = None,
        V_initializer: Callable = braintools.init.Constant(0.0),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.tau_epsp = braintools.init.param(tau_epsp, self.varshape)
        self.tau_reset = braintools.init.param(tau_reset, self.varshape)
        self.V_epsp = braintools.init.param(V_epsp, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_noise = braintools.init.param(V_noise, self.varshape)

        self.noise = np.asarray([] if noise is None else u.math.asarray(noise), dtype=np.float64).reshape(-1)
        self.V_initializer = V_initializer

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x):
        return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _to_numpy_ms(x):
        return np.asarray(u.math.asarray(x / u.ms), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.V_epsp) < 0.0):
            raise ValueError('EPSP amplitude V_epsp cannot be negative.')
        if np.any(self._to_numpy(self.V_reset) < 0.0):
            raise ValueError('Reset magnitude V_reset cannot be negative.')
        if np.any(self._to_numpy_ms(self.tau_epsp) <= 0.0) or np.any(self._to_numpy_ms(self.tau_reset) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')

    def _sum_excitatory_delta_inputs(self, state_shape):
        w_ex = np.zeros(state_shape, dtype=np.float64)
        if self.delta_inputs is None:
            return w_ex

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)
            out_np = self._broadcast_to_state(self._to_numpy(out), state_shape)
            w_ex = w_ex + np.maximum(out_np, 0.0)
        return w_ex

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = np.zeros_like(np.asarray(u.math.asarray(V), dtype=np.float64))
        idx0 = np.zeros_like(zeros, dtype=np.int64)

        self.i_syn_ex = brainstate.ShortTermState(jnp.asarray(zeros, dtype=jnp.float64))
        self.V_syn = brainstate.ShortTermState(jnp.asarray(zeros, dtype=jnp.float64))
        self.V_spike = brainstate.ShortTermState(jnp.asarray(zeros, dtype=jnp.float64))
        self.V = brainstate.HiddenState(jnp.asarray(u.math.asarray(V), dtype=jnp.float64))
        self.position = brainstate.ShortTermState(jnp.asarray(idx0, dtype=jnp.int64))

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)

    def reset_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = np.zeros_like(np.asarray(u.math.asarray(V), dtype=np.float64))
        idx0 = np.zeros_like(zeros, dtype=np.int64)

        self.i_syn_ex.value = jnp.asarray(zeros, dtype=jnp.float64)
        self.V_syn.value = jnp.asarray(zeros, dtype=jnp.float64)
        self.V_spike.value = jnp.asarray(zeros, dtype=jnp.float64)
        self.V.value = jnp.asarray(u.math.asarray(V), dtype=jnp.float64)
        self.position.value = jnp.asarray(idx0, dtype=jnp.int64)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        # Unlike typical LIF models, here U_reset > U_th by default.
        # Therefore, we scale directly with threshold crossing sign.
        v_scaled = V - self._U_TH
        return self.spk_fun(v_scaled)

    def update(self, x=0.0):
        # NEST iaf_chs_2007 has no CurrentEvent handler; x is intentionally unused.
        del x

        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        state_shape = self.V.value.shape

        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value), state_shape).copy()
        V_syn = self._broadcast_to_state(self._to_numpy(self.V_syn.value), state_shape).copy()
        V_spike = self._broadcast_to_state(self._to_numpy(self.V_spike.value), state_shape).copy()
        pos = self._broadcast_to_state(np.asarray(u.math.asarray(self.position.value), dtype=np.int64), state_shape).copy()

        tau_epsp = self._broadcast_to_state(self._to_numpy_ms(self.tau_epsp), state_shape)
        tau_reset = self._broadcast_to_state(self._to_numpy_ms(self.tau_reset), state_shape)
        U_epsp = self._broadcast_to_state(self._to_numpy(self.V_epsp), state_shape)
        U_reset = self._broadcast_to_state(self._to_numpy(self.V_reset), state_shape)
        U_noise = self._broadcast_to_state(self._to_numpy(self.V_noise), state_shape)

        # NEST pre_run_hook propagators.
        P11 = np.exp(-h / tau_epsp)
        P22 = P11
        P30 = np.exp(-h / tau_reset)
        P21 = U_epsp * math.e * P11 * h / tau_epsp

        # Spike input ring-buffer contribution for this simulation step.
        w_ex = self._sum_excitatory_delta_inputs(state_shape)

        # NEST update order in models/iaf_chs_2007.cpp::update().
        V_syn = V_syn * P22 + i_syn_ex * P21
        i_syn_ex = i_syn_ex * P11
        i_syn_ex = i_syn_ex + w_ex
        V_spike = V_spike * P30

        noise_term = np.zeros(state_shape, dtype=np.float64)
        if self.noise.size > 0:
            use_noise = U_noise > 0.0
            if np.any(use_noise):
                if np.any(pos[use_noise] >= self.noise.size):
                    raise IndexError(
                        'Noise signal exhausted before end of simulation. '
                        'Provide a noise vector at least as long as all simulated steps.'
                    )
                sampled_noise = np.zeros(state_shape, dtype=np.float64)
                sampled_noise[use_noise] = self.noise[pos[use_noise]]
                noise_term = U_noise * sampled_noise
                pos = np.where(use_noise, pos + 1, pos)

        V_m = V_syn + V_spike + noise_term

        spike_cond = V_m >= self._U_TH
        V_for_spike = V_m
        V_spike = np.where(spike_cond, V_spike - U_reset, V_spike)
        V_m = np.where(spike_cond, V_m - U_reset, V_m)

        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        self.i_syn_ex.value = jnp.asarray(i_syn_ex, dtype=jnp.float64)
        self.V_syn.value = jnp.asarray(V_syn, dtype=jnp.float64)
        self.V_spike.value = jnp.asarray(V_spike, dtype=jnp.float64)
        self.V.value = jnp.asarray(V_m, dtype=jnp.float64)
        self.position.value = jnp.asarray(pos, dtype=jnp.int64)

        V_out = np.where(spike_cond, self._U_TH + 1e-12, V_for_spike)
        return self.get_spike(jnp.asarray(V_out, dtype=jnp.float64))
