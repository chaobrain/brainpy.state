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
    'iaf_psc_alpha',
]


class iaf_psc_alpha(Neuron):
    r"""NEST-compatible ``iaf_psc_alpha`` neuron model.

    Short description
    -----------------

    Leaky integrate-and-fire model with alpha-shaped synaptic input currents.

    Description
    -----------

    ``iaf_psc_alpha`` is a current-based leaky integrate-and-fire model with:

    - hard threshold crossing,
    - fixed absolute refractory period,
    - no adaptation variables,
    - alpha-shaped excitatory and inhibitory synaptic currents.

    This implementation follows the update order and parameterization used by
    NEST ``models/iaf_psc_alpha.{h,cpp}``.

    Membrane dynamics
    .................

    The membrane potential evolves as

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m} + \frac{I_\text{syn} + I_e}{C_m}

    with

    .. math::

       I_\text{syn}(t) = I_{\text{syn,ex}}(t) + I_{\text{syn,in}}(t).

    Each incoming spike contributes an alpha-shaped postsynaptic current:

    .. math::

       i_{\text{syn,X}}(t) = \frac{e}{\tau_{\text{syn,X}}}
       t e^{-t/\tau_{\text{syn,X}}}\Theta(t), \quad X \in \{\text{ex},\text{in}\},

    normalized to peak 1 at :math:`t = \tau_{\text{syn,X}}`.

    Update scheme and state variables
    .................................

    Following NEST, the model is integrated exactly for fixed simulation step
    :math:`h = dt` using precomputed propagator coefficients.

    Internal state (NEST notation):

    - :math:`y_0`: buffered external current for next step,
    - :math:`dI_{ex}, I_{ex}`: excitatory alpha-kernel states,
    - :math:`dI_{in}, I_{in}`: inhibitory alpha-kernel states,
    - :math:`y_3 = V_m - E_L`,
    - :math:`r`: refractory countdown in grid steps.

    For each step, NEST-compatible order is:

    1. Update membrane potential if not refractory.
    2. Update synaptic alpha states.
    3. Add arriving spike input to :math:`dI_{ex}` / :math:`dI_{in}`.
    4. Perform threshold test, reset, refractory assignment, spike emission.
    5. Store buffered external current for the next step.

    Numerical stability near singularity
    ....................................

    NEST uses ``IAFPropagatorAlpha`` to avoid instability when
    :math:`\tau_m \approx \tau_{\text{syn}}`. This implementation mirrors that
    behavior with the same singular fallback formulas for propagator elements
    :math:`P_{31}` and :math:`P_{32}`.

    Parameters
    ----------

    ==================== ================== =============================== ======================================================
    **Parameter**        **Default**        **Math equivalent**             **Description**
    ==================== ================== =============================== ======================================================
    ``in_size``          (required)                                         Population shape
    ``E_L``              -70 mV             :math:`E_L`                     Resting membrane potential
    ``C_m``              250 pF             :math:`C_m`                     Membrane capacitance
    ``tau_m``            10 ms              :math:`\tau_m`                  Membrane time constant
    ``t_ref``            2 ms               :math:`t_{ref}`                 Absolute refractory period
    ``V_th``             -55 mV             :math:`V_{th}`                  Spike threshold
    ``V_reset``          -70 mV             :math:`V_{reset}`               Reset potential
    ``tau_syn_ex``       2 ms               :math:`\tau_{\text{syn,ex}}`    Excitatory alpha rise time
    ``tau_syn_in``       2 ms               :math:`\tau_{\text{syn,in}}`    Inhibitory alpha rise time
    ``I_e``              0 pA               :math:`I_e`                     Constant external current
    ``V_min``            None               :math:`V_{min}`                 Optional lower bound (None -> :math:`-\infty`)
    ``V_initializer``    Constant(-70 mV)                                   Membrane initializer
    ``spk_fun``          ReluGrad()                                         Surrogate spike function
    ``spk_reset``        ``'hard'``                                         Reset mode (hard reset matches NEST)
    ``ref_var``          ``False``                                          If True, expose boolean refractory state
    ==================== ================== =============================== ======================================================

    State variables
    ---------------

    - ``V``: membrane potential :math:`V_m`.
    - ``I_syn_ex``: excitatory synaptic current.
    - ``I_syn_in``: inhibitory synaptic current.
    - ``dI_syn_ex`` / ``dI_syn_in``: alpha auxiliary states.
    - ``y0``: one-step delayed external current buffer.
    - ``refractory_step_count``: refractory countdown in grid steps.
    - ``last_spike_time``: last emitted spike time (:math:`t + dt` on spike).
    - ``refractory``: optional boolean refractory flag.

    Notes
    -----

    - Spike weights are interpreted in current units (pA).
      Positive weights are excitatory, negative weights are inhibitory.
    - ``update(x=...)`` applies ``x`` one step later (ring-buffer semantics),
      matching NEST current-event handling.

    References
    ----------
    .. [1] Rotter S, Diesmann M (1999). Exact simulation of time-invariant linear
           systems with applications to neuronal modeling. Biological Cybernetics
           81:381-402. DOI: https://doi.org/10.1007/s004220050570
    .. [2] Diesmann M, Gewaltig M-O, Rotter S, Aertsen A (2001). State space
           analysis of synchronous spiking in cortical neural networks.
           Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X
    .. [3] Morrison A, Straube S, Plesser HE, Diesmann M (2007). Exact
           subthreshold integration with continuous spike times in discrete time
           neural network simulations. Neural Computation 19(1):47-79.
           DOI: https://doi.org/10.1162/neco.2007.19.1.47
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
        tau_syn_ex: ArrayLike = 2. * u.ms,
        tau_syn_in: ArrayLike = 2. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        V_min: ArrayLike = None,
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
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_min = None if V_min is None else braintools.init.param(V_min, self.varshape)
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
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be > 0.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be > 0.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All synaptic time constants must be > 0.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError("The refractory time t_ref can't be negative.")
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.dI_syn_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.dI_syn_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.y0 = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
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

    @staticmethod
    def _alpha_propagator_p31_p32(tau_syn: np.ndarray, tau_m: np.ndarray, c_m: np.ndarray, h_ms: float):
        # Mirrors NEST IAFPropagatorAlpha and singular fallback behavior.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            beta = tau_syn * tau_m / (tau_m - tau_syn)
            gamma = beta / c_m
            inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)

            exp_h_tau_syn = np.exp(-h_ms / tau_syn)
            expm1_h_tau = np.expm1(h_ms * inv_beta)

            p32_raw = gamma * exp_h_tau_syn * expm1_h_tau
            exp_h_tau_m = np.exp(-h_ms / tau_m)
            p32_singular = h_ms / c_m * exp_h_tau_m

            # NEST checks "isnormal && > 0". Approximate isnormal in NumPy.
            normal_min = np.finfo(np.float64).tiny
            p32_regular_mask = np.isfinite(p32_raw) & (np.abs(p32_raw) >= normal_min) & (p32_raw > 0.0)
            p32 = np.where(p32_regular_mask, p32_raw, p32_singular)

            h_min_regular = 1e-7 * tau_m * tau_m / np.abs(tau_m - tau_syn)
            p31_regular_mask = np.isfinite(h_min_regular) & (h_ms > h_min_regular)

            p31_regular = gamma * exp_h_tau_syn * (beta * expm1_h_tau - h_ms)
            p31_singular = 0.5 * h_ms * h_ms / c_m * exp_h_tau_m
            p31 = np.where(p31_regular_mask, p31_regular, p31_singular)

        return p31, p32

    def update(self, x=0. * u.pA):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        v_abs = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        y3 = v_abs - E_L

        c_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        i_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)

        theta_rel = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        v_reset_rel = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)
        lower_rel = -np.inf * np.ones(v_shape, dtype=np.float64)
        if self.V_min is not None:
            lower_rel = self._broadcast_to_state(self._to_numpy(self.V_min - self.E_L, u.mV), v_shape)

        y0 = self._broadcast_to_state(np.asarray(self.y0.value, dtype=np.float64), v_shape)
        dI_ex = self._broadcast_to_state(np.asarray(self.dI_syn_ex.value, dtype=np.float64), v_shape)
        I_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape)
        dI_in = self._broadcast_to_state(np.asarray(self.dI_syn_in.value, dtype=np.float64), v_shape)
        I_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            v_shape,
        )

        # Coefficients from NEST pre_run_hook.
        P11_ex = np.exp(-h / tau_ex)
        P22_ex = P11_ex
        P21_ex = h * P11_ex

        P11_in = np.exp(-h / tau_in)
        P22_in = P11_in
        P21_in = h * P11_in

        expm1_tau_m = np.expm1(-h / tau_m)
        P30 = -tau_m / c_m * expm1_tau_m
        P31_ex, P32_ex = self._alpha_propagator_p31_p32(tau_ex, tau_m, c_m, h)
        P31_in, P32_in = self._alpha_propagator_p31_p32(tau_in, tau_m, c_m, h)

        epsc_init = math.e / tau_ex
        ipsc_init = math.e / tau_in

        # Spike/current buffers for next step.
        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all > 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)
        y0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        # 1) membrane update
        not_refractory = r == 0
        y3_candidate = (
            P30 * (y0 + i_e)
            + P31_ex * dI_ex
            + P32_ex * I_ex
            + P31_in * dI_in
            + P32_in * I_in
            + expm1_tau_m * y3
            + y3
        )
        y3_candidate = np.maximum(y3_candidate, lower_rel)
        y3 = np.where(not_refractory, y3_candidate, y3)
        r = np.where(not_refractory, r, r - 1)

        # 2) synaptic alpha updates
        I_ex = P21_ex * dI_ex + P22_ex * I_ex
        dI_ex = dI_ex * P11_ex
        dI_ex = dI_ex + epsc_init * w_ex

        I_in = P21_in * dI_in + P22_in * I_in
        dI_in = dI_in * P11_in
        dI_in = dI_in + ipsc_init * w_in

        # 3) threshold + reset
        spike_cond = y3 >= theta_rel
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )
        r = np.where(spike_cond, refr_counts, r)
        y3_for_spike = y3
        y3 = np.where(spike_cond, v_reset_rel, y3)

        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        # write-back
        self.V.value = (y3 + E_L) * u.mV
        self.I_syn_ex.value = I_ex * u.pA
        self.I_syn_in.value = I_in * u.pA
        self.dI_syn_ex.value = dI_ex
        self.dI_syn_in.value = dI_in
        self.y0.value = y0_next
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        v_out = np.where(spike_cond, theta_rel + E_L + 1e-12, y3_for_spike + E_L)
        return self.get_spike(v_out * u.mV)
