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
    'amat2_psc_exp',
]


class amat2_psc_exp(Neuron):
    r"""NEST-compatible ``amat2_psc_exp`` neuron model.

    Short description
    -----------------

    Non-resetting leaky integrate-and-fire neuron model with exponential
    postsynaptic currents and adaptive threshold (including a voltage-dependent
    threshold component).

    Description
    -----------

    ``amat2_psc_exp`` is an implementation of a leaky integrate-and-fire model
    with exponential shaped postsynaptic currents (PSCs).  Thus, postsynaptic
    currents have an infinitely short rise time.

    The threshold is lifted when the neuron fires and then decreases in a
    fixed time scale toward a fixed level [3]_.

    The threshold crossing is followed by a total refractory period during
    which the neuron is not allowed to fire, even if the membrane potential
    exceeds the threshold.  **The membrane potential is NOT reset**, but
    continuously integrated.

    Compared to ``mat2_psc_exp``, this model adds a **voltage-dependent
    threshold component** :math:`V_{th,v}` that tracks the low-pass filtered
    derivative of the membrane potential, scaled by the parameter ``beta``.
    When ``beta = 0``, the model behaves identically to ``mat2_psc_exp``
    (given matching parameter values).

    The linear subthreshold dynamics is integrated by the Exact Integration
    scheme [1]_.  The neuron dynamics is solved on the time grid given by the
    computation step size.  Incoming as well as emitted spikes are forced to
    that grid.

    An additional state variable and the corresponding differential equation
    represents a piecewise constant external current.

    The general framework for the consistent formulation of systems with
    neuron like dynamics interacting by point events is described in
    [1]_.  A flow chart can be found in [2]_.

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

    Two-timescale adaptive threshold with voltage-dependent component:

    .. math::

       V_{th}(t) = \omega + V_{th,1}(t) + V_{th,2}(t) + V_{th,v}(t)

       \frac{dV_{th,1}}{dt} = -\frac{V_{th,1}}{\tau_1}
       \qquad
       \frac{dV_{th,2}}{dt} = -\frac{V_{th,2}}{\tau_2}

    The voltage-dependent threshold component satisfies [3]_, Eqs. 16-17:

    .. math::

       V_{th,v}(t) = \beta \int_0^t \frac{s}{\tau_v}
       \exp\!\left(-\frac{s}{\tau_v}\right)
       \frac{dV_m}{dt}(t-s)\, ds

    This is implemented via two auxiliary variables :math:`V_{th,v}` and
    :math:`V_{th,dv}` (see NEST source for the exact propagator coefficients).

    On each spike, the threshold components are incremented:

    .. math::

       V_{th,1} \leftarrow V_{th,1} + \alpha_1
       \qquad
       V_{th,2} \leftarrow V_{th,2} + \alpha_2

    Update order
    ............

    For each simulation step (NEST order):

    1. Evolve voltage-dependent threshold component (``V_th_v``, ``V_th_dv``)
       via exact integration propagators.
    2. Evolve membrane potential (exact integration).
    3. Decay adaptive threshold components (``V_th_1``, ``V_th_2``).
    4. Decay synaptic currents and add incoming spikes.
    5. Spike detection: if not refractory and
       :math:`V_m \geq \omega + V_{th,1} + V_{th,2} + V_{th,v}`,
       fire a spike, jump thresholds, set refractory counter.
    6. Otherwise, if refractory, decrement refractory counter.
    7. Store buffered currents for next step.

    .. note::

       - The time constants must fulfill the following conditions:
         ``tau_m != tau_syn_ex``, ``tau_m != tau_syn_in``,
         ``tau_m != tau_v``, ``tau_v != tau_syn_ex``, ``tau_v != tau_syn_in``.
         This is required to avoid singularities in the exact integration
         propagators.
       - Expect unstable numerics if time constants that are required to be
         different are very close.
       - Some parameter values given in Table 1 of [4]_ are incorrect.
         For correct values, see Table 4 of [5]_.

    Parameters
    ----------

    ==================== ================== =============================== ==========================================================
    **Parameter**        **Default**        **Math equivalent**             **Description**
    ==================== ================== =============================== ==========================================================
    ``in_size``          (required)                                         Population shape
    ``E_L``              -70 mV             :math:`E_L`                     Resting membrane potential
    ``C_m``              200 pF             :math:`C_m`                     Membrane capacitance
    ``tau_m``            10 ms              :math:`\tau_m`                  Membrane time constant
    ``t_ref``            2 ms               :math:`t_{ref}`                 Duration of absolute refractory period (no spiking)
    ``tau_syn_ex``       1 ms               :math:`\tau_{\mathrm{syn,ex}}`  Time constant of excitatory postsynaptic current
    ``tau_syn_in``       3 ms               :math:`\tau_{\mathrm{syn,in}}`  Time constant of inhibitory postsynaptic current
    ``I_e``              0 pA               :math:`I_e`                     Constant external input current
    ``tau_1``            10 ms              :math:`\tau_1`                  Short time constant of adaptive threshold
    ``tau_2``            200 ms             :math:`\tau_2`                  Long time constant of adaptive threshold
    ``alpha_1``          10 mV              :math:`\alpha_1`                Amplitude of short time threshold adaption
    ``alpha_2``          0 mV               :math:`\alpha_2`                Amplitude of long time threshold adaption
    ``beta``             0 1/ms             :math:`\beta`                   Scaling coefficient for voltage-dependent threshold
    ``tau_v``            5 ms               :math:`\tau_v`                  Time constant for voltage-dependent threshold component
    ``omega``            -65 mV             :math:`\omega`                  Resting spike threshold (absolute value, not relative to E_L)
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
    ``V_th_v``                ``ShortTermState``    Voltage-dependent threshold component (mV)
    ``V_th_dv``               ``ShortTermState``    Derivative of voltage-dependent threshold (mV)
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
    .. [4] Yamauchi S, Kim H, Shinomoto S (2011). Elemental spiking neuron
           model for reproducing diverse firing patterns and predicting precise
           firing times. Frontiers in Computational Neuroscience 5:42.
           DOI: https://doi.org/10.3389/fncom.2011.00042
    .. [5] Heiberg T, Kriener B, Tetzlaff T, Einevoll GT, Plesser HE (2018).
           Firing-rate model for neurons with a broad repertoire of spiking
           behaviors. J Comput Neurosci 45:103.
           DOI: https://doi.org/10.1007/s10827-018-0693-9

    See Also
    --------
    mat2_psc_exp : Same model without voltage-dependent threshold component.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 200. * u.pF,
        tau_m: ArrayLike = 10. * u.ms,
        t_ref: ArrayLike = 2. * u.ms,
        tau_syn_ex: ArrayLike = 1. * u.ms,
        tau_syn_in: ArrayLike = 3. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        tau_1: ArrayLike = 10. * u.ms,
        tau_2: ArrayLike = 200. * u.ms,
        alpha_1: ArrayLike = 10. * u.mV,
        alpha_2: ArrayLike = 0. * u.mV,
        beta: ArrayLike = 0. / u.ms,
        tau_v: ArrayLike = 5. * u.ms,
        omega: ArrayLike = -65. * u.mV,
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
        self.beta = braintools.init.param(beta, self.varshape)
        self.tau_v = braintools.init.param(tau_v, self.varshape)
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
        tau_m_np = self._to_numpy(self.tau_m, u.ms)
        tau_ex_np = self._to_numpy(self.tau_syn_ex, u.ms)
        tau_in_np = self._to_numpy(self.tau_syn_in, u.ms)
        tau_v_np = self._to_numpy(self.tau_v, u.ms)
        if np.any(tau_m_np <= 0.0) or np.any(tau_ex_np <= 0.0) or np.any(tau_in_np <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) <= 0.0):
            raise ValueError('Refractory time must be strictly positive.')
        tau_1_np = self._to_numpy(self.tau_1, u.ms)
        tau_2_np = self._to_numpy(self.tau_2, u.ms)
        if np.any(tau_1_np <= 0.0) or np.any(tau_2_np <= 0.0):
            raise ValueError('Adaptive threshold time constants must be strictly positive.')
        if np.any(tau_v_np <= 0.0):
            raise ValueError('tau_v must be strictly positive.')
        if np.any(tau_m_np == tau_ex_np) or np.any(tau_m_np == tau_in_np) or np.any(tau_m_np == tau_v_np):
            raise ValueError(
                'tau_m must differ from tau_syn_ex, tau_syn_in and tau_v. '
                'See note in documentation.'
            )
        if np.any(tau_v_np == tau_ex_np) or np.any(tau_v_np == tau_in_np):
            raise ValueError(
                'tau_v must differ from tau_syn_ex, tau_syn_in and tau_m. '
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
        self.V_th_v = brainstate.ShortTermState(zeros * u.mV)
        self.V_th_dv = brainstate.ShortTermState(zeros * u.mV)
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
            V_th = self.omega + self.V_th_1.value + self.V_th_2.value + self.V_th_v.value
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
        taum = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tauE = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tauI = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        tau_1 = self._broadcast_to_state(self._to_numpy(self.tau_1, u.ms), v_shape)
        tau_2 = self._broadcast_to_state(self._to_numpy(self.tau_2, u.ms), v_shape)
        alpha_1 = self._broadcast_to_state(self._to_numpy(self.alpha_1, u.mV), v_shape)
        alpha_2 = self._broadcast_to_state(self._to_numpy(self.alpha_2, u.mV), v_shape)
        beta = self._broadcast_to_state(self._to_numpy(self.beta, 1.0 / u.ms), v_shape)
        tauV = self._broadcast_to_state(self._to_numpy(self.tau_v, u.ms), v_shape)
        # omega is stored as absolute mV; convert to relative to E_L
        omega_rel = self._broadcast_to_state(self._to_numpy(self.omega - self.E_L, u.mV), v_shape)

        # State variables (V_m relative to E_L)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        V_th_1 = self._broadcast_to_state(self._to_numpy(self.V_th_1.value, u.mV), v_shape)
        V_th_2 = self._broadcast_to_state(self._to_numpy(self.V_th_2.value, u.mV), v_shape)
        V_th_v = self._broadcast_to_state(self._to_numpy(self.V_th_v.value, u.mV), v_shape)
        V_th_dv = self._broadcast_to_state(self._to_numpy(self.V_th_dv.value, u.mV), v_shape)
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value, u.pA), v_shape)
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.i_syn_in.value, u.pA), v_shape)
        i_0 = self._broadcast_to_state(self._to_numpy(self.i_0.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )

        # --- Compute propagator coefficients (NEST pre_run_hook) ---
        c = C_m

        eE = np.exp(-h / tauE)
        eI = np.exp(-h / tauI)
        em = np.exp(-h / taum)
        e1 = np.exp(-h / tau_1)
        e2 = np.exp(-h / tau_2)
        eV = np.exp(-h / tauV)

        # Independent propagators
        P11 = eE
        P22 = eI
        P33 = em
        P44 = e1
        P55 = e2
        P66 = eV
        P77 = eV

        # Membrane potential propagators
        P30 = (taum - em * taum) / c
        P31 = ((eE - em) * tauE * taum) / (c * (tauE - taum))
        P32 = ((eI - em) * tauI * taum) / (c * (tauI - taum))

        # Voltage-dependent threshold propagators (V_th_dv -> row 6)
        P60 = (beta * (em - eV) * taum * tauV) / (c * (taum - tauV))
        P61 = (beta * tauE * taum * tauV * (eV * (-tauE + taum) + em * (tauE - tauV) + eE * (-taum + tauV))) \
            / (c * (tauE - taum) * (tauE - tauV) * (taum - tauV))
        P62 = (beta * tauI * taum * tauV * (eV * (-tauI + taum) + em * (tauI - tauV) + eI * (-taum + tauV))) \
            / (c * (tauI - taum) * (tauI - tauV) * (taum - tauV))
        P63 = (beta * (-em + eV) * tauV) / (taum - tauV)

        # Voltage-dependent threshold propagators (V_th_v -> row 7)
        P70 = (beta * taum * tauV * (em * taum * tauV - eV * (h * (taum - tauV) + taum * tauV))) \
            / (c * (taum - tauV) ** 2)
        P71 = (beta * tauE * taum * tauV
               * ((em * taum * (tauE - tauV) ** 2 - eE * tauE * (taum - tauV) ** 2) * tauV
                  - eV * (tauE - taum)
                  * (h * (tauE - tauV) * (taum - tauV) + tauE * taum * tauV - tauV ** 3))) \
            / (c * (tauE - taum) * (tauE - tauV) ** 2 * (taum - tauV) ** 2)
        P72 = (beta * tauI * taum * tauV
               * ((em * taum * (tauI - tauV) ** 2 - eI * tauI * (taum - tauV) ** 2) * tauV
                  - eV * (tauI - taum)
                  * (h * (tauI - tauV) * (taum - tauV) + tauI * taum * tauV - tauV ** 3))) \
            / (c * (tauI - taum) * (tauI - tauV) ** 2 * (taum - tauV) ** 2)
        P73 = (beta * tauV * (-(em * taum * tauV) + eV * (h * (taum - tauV) + taum * tauV))) \
            / (taum - tauV) ** 2
        P76 = eV * h

        # --- Get spike inputs ---
        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all > 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)

        # --- Get current inputs (one-step delayed, stored for next step) ---
        i_0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        # === NEST update ordering (amat2_psc_exp.cpp update() lines 375-421) ===

        # Step 1: Evolve voltage-dependent threshold (V_th_v and V_th_dv)
        # IMPORTANT: V_th_v must be computed BEFORE V_th_dv is updated,
        # because V_th_v uses the OLD V_th_dv value.
        V_th_v_new = (I_e + i_0) * P70 + i_syn_ex * P71 + i_syn_in * P72 \
            + V_rel * P73 + V_th_dv * P76 + V_th_v * P77
        V_th_dv_new = (I_e + i_0) * P60 + i_syn_ex * P61 + i_syn_in * P62 \
            + V_rel * P63 + V_th_dv * P66
        V_th_v = V_th_v_new
        V_th_dv = V_th_dv_new

        # Step 2: Evolve membrane potential
        V_rel = (I_e + i_0) * P30 + i_syn_ex * P31 + i_syn_in * P32 + V_rel * P33

        # Step 3: Decay adaptive threshold components
        V_th_1 = V_th_1 * P44
        V_th_2 = V_th_2 * P55

        # Step 4: Decay synaptic currents and add incoming spikes
        i_syn_ex = i_syn_ex * P11
        i_syn_in = i_syn_in * P22
        i_syn_ex = i_syn_ex + w_ex
        i_syn_in = i_syn_in + w_in

        # Step 5-6: Spike detection (no voltage reset!)
        not_refractory = r == 0
        spike_cond = not_refractory & (V_rel >= omega_rel + V_th_1 + V_th_2 + V_th_v)

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

        # Step 7: Store buffered currents for next step

        # --- Write back state variables ---
        self.V.value = (V_rel + E_L) * u.mV
        self.V_th_1.value = V_th_1 * u.mV
        self.V_th_2.value = V_th_2 * u.mV
        self.V_th_v.value = V_th_v * u.mV
        self.V_th_dv.value = V_th_dv * u.mV
        self.i_syn_ex.value = i_syn_ex * u.pA
        self.i_syn_in.value = i_syn_in * u.pA
        self.i_0.value = i_0_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        # Return spike output via surrogate gradient.
        V_th_abs = omega_rel + V_th_1 + V_th_2 + V_th_v + E_L
        V_out = np.where(spike_cond, V_th_abs + 1e-12, V_th_abs - 1e-12)
        return self.get_spike(V_out * u.mV, V_th_abs * u.mV)
