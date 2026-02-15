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
    'mat2_psc_exp',
]


class mat2_psc_exp(Neuron):
    r"""NEST-compatible ``mat2_psc_exp`` neuron model.

    Short description
    -----------------

    Non-resetting leaky integrate-and-fire neuron model with exponential
    postsynaptic currents and a two-timescale adaptive threshold.

    Description
    -----------

    ``mat2_psc_exp`` is an implementation of a leaky integrate-and-fire model
    with exponential shaped postsynaptic currents (PSCs).  Thus, postsynaptic
    currents have an infinitely short rise time.

    The threshold is lifted when the neuron fires and then decreases in a
    fixed time scale toward a fixed level [3]_.

    The threshold crossing is followed by a total refractory period during
    which the neuron is not allowed to fire, even if the membrane potential
    exceeds the threshold.  **The membrane potential is NOT reset**, but
    continuously integrated.

    The linear subthreshold dynamics is integrated by the Exact Integration
    scheme [1]_.  The neuron dynamics is solved on the time grid given by the
    computation step size.  Incoming as well as emitted spikes are forced to
    that grid.

    An additional state variable and the corresponding differential equation
    represents a piecewise constant external current.

    The implementation requires ``tau_m != tau_syn_ex`` and
    ``tau_m != tau_syn_in`` to avoid a degenerate case of the ODE describing
    the model [1]_.  For very similar values, numerics will be unstable.

    State equations
    ...............

    Subthreshold membrane dynamics:

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m}
       + \frac{I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}} + I_e + I_0}{C_m}

    Exponentially decaying synaptic currents:

    .. math::

       \frac{dI_{\mathrm{syn,ex}}}{dt} = -\frac{I_{\mathrm{syn,ex}}}{\tau_{\mathrm{syn,ex}}}
       \qquad
       \frac{dI_{\mathrm{syn,in}}}{dt} = -\frac{I_{\mathrm{syn,in}}}{\tau_{\mathrm{syn,in}}}

    Two-timescale adaptive threshold:

    .. math::

       V_{th}(t) = \omega + V_{th,1}(t) + V_{th,2}(t)

       \frac{dV_{th,1}}{dt} = -\frac{V_{th,1}}{\tau_1}
       \qquad
       \frac{dV_{th,2}}{dt} = -\frac{V_{th,2}}{\tau_2}

    On each spike, the threshold components are incremented:

    .. math::

       V_{th,1} \leftarrow V_{th,1} + \alpha_1
       \qquad
       V_{th,2} \leftarrow V_{th,2} + \alpha_2

    Update order
    ............

    For each simulation step (NEST order):

    1. Evolve membrane potential (exact integration).
    2. Decay adaptive threshold components.
    3. Decay synaptic currents and add incoming spikes.
    4. Spike detection: if not refractory and
       :math:`V_m \geq \omega + V_{th,1} + V_{th,2}`,
       fire a spike, jump thresholds, set refractory counter.
    5. Otherwise, if refractory, decrement refractory counter.
    6. Store buffered currents for next step.

    Parameters
    ----------

    ==================== ================== =============================== ==========================================================
    **Parameter**        **Default**        **Math equivalent**             **Description**
    ==================== ================== =============================== ==========================================================
    ``in_size``          (required)                                         Population shape
    ``E_L``              -70 mV             :math:`E_L`                     Resting membrane potential
    ``C_m``              100 pF             :math:`C_m`                     Membrane capacitance
    ``tau_m``            5 ms               :math:`\tau_m`                  Membrane time constant
    ``t_ref``            2 ms               :math:`t_{ref}`                 Duration of absolute refractory period (no spiking)
    ``tau_syn_ex``       1 ms               :math:`\tau_{\mathrm{syn,ex}}`  Time constant of excitatory postsynaptic current
    ``tau_syn_in``       3 ms               :math:`\tau_{\mathrm{syn,in}}`  Time constant of inhibitory postsynaptic current
    ``I_e``              0 pA               :math:`I_e`                     Constant external input current
    ``tau_1``            10 ms              :math:`\tau_1`                  Short time constant of adaptive threshold
    ``tau_2``            200 ms             :math:`\tau_2`                  Long time constant of adaptive threshold
    ``alpha_1``          37 mV              :math:`\alpha_1`                Amplitude of short time threshold adaption
    ``alpha_2``          2 mV               :math:`\alpha_2`                Amplitude of long time threshold adaption
    ``omega``            -51 mV             :math:`\omega`                  Resting spike threshold (absolute value, not relative to E_L)
    ``V_initializer``    Constant(-70 mV)                                   Membrane potential initializer
    ``spk_fun``          ReluGrad()                                         Surrogate spike function
    ``spk_reset``        ``'hard'``                                         Reset mode (not used for voltage; used in ``get_spike``)
    ``ref_var``          ``False``                                          If True, expose boolean refractory state
    ==================== ================== =============================== ==========================================================

    State variables
    ---------------

    ========================= ===================== ====================================================
    **Variable**              **Type**              **Description**
    ========================= ===================== ====================================================
    ``V``                     ``HiddenState`` (mV)  Membrane potential (absolute)
    ``V_th_1``                ``ShortTermState``    Short-timescale adaptive threshold component (mV, relative to omega)
    ``V_th_2``                ``ShortTermState``    Long-timescale adaptive threshold component (mV, relative to omega)
    ``i_syn_ex``              ``ShortTermState``    Excitatory postsynaptic current (pA)
    ``i_syn_in``              ``ShortTermState``    Inhibitory postsynaptic current (pA)
    ``i_0``                   ``ShortTermState``    DC input current (pA)
    ``refractory_step_count`` ``ShortTermState``    Refractory countdown (integer steps)
    ``last_spike_time``       ``ShortTermState``    Time of last spike (ms)
    ========================= ===================== ====================================================

    References
    ----------

    .. [1] Rotter S and Diesmann M (1999). Exact simulation of
           time-invariant linear systems with applications to neuronal
           modeling. Biological Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    .. [2] Diesmann M, Gewaltig M-O, Rotter S, Aertsen A (2001). State
           space analysis of synchronous spiking in cortical neural
           networks. Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X
    .. [3] Kobayashi R, Tsubo Y and Shinomoto S (2009). Made-to-order
           spiking neuron model equipped with a multi-timescale adaptive
           threshold. Frontiers in Computational Neuroscience 3:9.
           DOI: https://doi.org/10.3389/neuro.10.009.2009
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 100. * u.pF,
        tau_m: ArrayLike = 5. * u.ms,
        t_ref: ArrayLike = 2. * u.ms,
        tau_syn_ex: ArrayLike = 1. * u.ms,
        tau_syn_in: ArrayLike = 3. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        tau_1: ArrayLike = 10. * u.ms,
        tau_2: ArrayLike = 200. * u.ms,
        alpha_1: ArrayLike = 37. * u.mV,
        alpha_2: ArrayLike = 2. * u.mV,
        omega: ArrayLike = -51. * u.mV,
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
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.tau_1 = braintools.init.param(tau_1, self.varshape)
        self.tau_2 = braintools.init.param(tau_2, self.varshape)
        self.alpha_1 = braintools.init.param(alpha_1, self.varshape)
        self.alpha_2 = braintools.init.param(alpha_2, self.varshape)
        self.omega = braintools.init.param(omega, self.varshape)

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
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('Synaptic time constants must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) <= 0.0):
            raise ValueError('Refractory time must be strictly positive.')
        if np.any(self._to_numpy(self.tau_1, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_2, u.ms) <= 0.0):
            raise ValueError('Adaptive threshold time constants must be strictly positive.')
        tau_m_np = self._to_numpy(self.tau_m, u.ms)
        tau_ex_np = self._to_numpy(self.tau_syn_ex, u.ms)
        tau_in_np = self._to_numpy(self.tau_syn_in, u.ms)
        if np.any(tau_m_np == tau_ex_np) or np.any(tau_m_np == tau_in_np):
            raise ValueError(
                'Membrane and synapse time constant(s) must differ. '
                'See note in documentation.'
            )

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.V_th_1 = brainstate.ShortTermState(zeros * u.mV)
        self.V_th_2 = brainstate.ShortTermState(zeros * u.mV)
        self.i_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.i_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.i_0 = brainstate.ShortTermState(zeros * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(u.math.asarray(ref_steps > 0, dtype=bool))

    def get_spike(self, V: ArrayLike = None, V_th: ArrayLike = None):
        V = self.V.value if V is None else V
        if V_th is None:
            V_th = self.omega + self.V_th_1.value + self.V_th_2.value
        # Scale relative to the effective adaptive threshold.
        v_scaled = (V - V_th) / u.math.abs(self.omega - self.E_L)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def update(self, x=0. * u.pA):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        # Extract all parameters as plain float64 numpy arrays
        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        tau_1 = self._broadcast_to_state(self._to_numpy(self.tau_1, u.ms), v_shape)
        tau_2 = self._broadcast_to_state(self._to_numpy(self.tau_2, u.ms), v_shape)
        alpha_1 = self._broadcast_to_state(self._to_numpy(self.alpha_1, u.mV), v_shape)
        alpha_2 = self._broadcast_to_state(self._to_numpy(self.alpha_2, u.mV), v_shape)
        # omega is stored as absolute mV; convert to relative to E_L
        omega_rel = self._broadcast_to_state(self._to_numpy(self.omega - self.E_L, u.mV), v_shape)

        # State variables (all relative to E_L for V_m, relative to omega for thresholds)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        V_th_1 = self._broadcast_to_state(self._to_numpy(self.V_th_1.value, u.mV), v_shape)
        V_th_2 = self._broadcast_to_state(self._to_numpy(self.V_th_2.value, u.mV), v_shape)
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value, u.pA), v_shape)
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.i_syn_in.value, u.pA), v_shape)
        i_0 = self._broadcast_to_state(self._to_numpy(self.i_0.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )

        # --- Compute propagator coefficients ---
        # Membrane potential propagators (exact integration)
        P11ex = np.exp(-h / tau_ex)
        P11in = np.exp(-h / tau_in)
        P22_expm1 = np.expm1(-h / tau_m)

        P21ex = -tau_m / (C_m * (1.0 - tau_m / tau_ex)) * P11ex * np.expm1(h * (1.0 / tau_ex - 1.0 / tau_m))
        P21in = -tau_m / (C_m * (1.0 - tau_m / tau_in)) * P11in * np.expm1(h * (1.0 / tau_in - 1.0 / tau_m))
        P20 = -tau_m / C_m * P22_expm1

        # Adaptive threshold propagators
        P11th = np.exp(-h / tau_1)
        P22th = np.exp(-h / tau_2)

        # --- Get spike inputs ---
        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all > 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)

        # --- Get current inputs (one-step delayed, stored for next step) ---
        i_0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        # === NEST update ordering (mat2_psc_exp.cpp lines 316-358) ===

        # Step 1: Evolve membrane potential
        V_rel = V_rel * P22_expm1 + V_rel + i_syn_ex * P21ex + i_syn_in * P21in + (I_e + i_0) * P20

        # Step 2: Evolve adaptive threshold
        V_th_1 = V_th_1 * P11th
        V_th_2 = V_th_2 * P22th

        # Step 3: Decay synaptic currents and add incoming spikes
        i_syn_ex = i_syn_ex * P11ex
        i_syn_in = i_syn_in * P11in
        i_syn_ex = i_syn_ex + w_ex
        i_syn_in = i_syn_in + w_in

        # Step 4-5: Spike detection (no voltage reset!)
        not_refractory = r == 0
        spike_cond = not_refractory & (V_rel >= omega_rel + V_th_1 + V_th_2)

        # On spike: jump threshold components, set refractory counter
        V_th_1 = np.where(spike_cond, V_th_1 + alpha_1, V_th_1)
        V_th_2 = np.where(spike_cond, V_th_2 + alpha_2, V_th_2)
        r = np.where(
            spike_cond,
            self._broadcast_to_state(
                np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
            ),
            np.where(not_refractory, r, r - 1),
        )

        # Step 6: Store buffered currents for next step
        # (i_0 is updated with the new current input)

        # --- Write back state variables ---
        self.V.value = (V_rel + E_L) * u.mV
        self.V_th_1.value = V_th_1 * u.mV
        self.V_th_2.value = V_th_2 * u.mV
        self.i_syn_ex.value = i_syn_ex * u.pA
        self.i_syn_in.value = i_syn_in * u.pA
        self.i_0.value = i_0_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        # Return spike output: for surrogate gradient, emit above-threshold signal.
        # The effective threshold (absolute) is E_L + omega_rel + V_th_1 + V_th_2.
        # When spiking, produce V slightly above threshold; otherwise, slightly below.
        V_th_abs = omega_rel + V_th_1 + V_th_2 + E_L
        V_out = np.where(spike_cond, V_th_abs + 1e-12, V_th_abs - 1e-12)
        return self.get_spike(V_out * u.mV, V_th_abs * u.mV)
