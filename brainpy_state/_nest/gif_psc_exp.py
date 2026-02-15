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
from typing import Callable, Optional, Sequence

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'gif_psc_exp',
]


class gif_psc_exp(Neuron):
    r"""Current-based generalized integrate-and-fire neuron (GIF) model.

    Description
    -----------

    ``gif_psc_exp`` is the generalized integrate-and-fire neuron according to
    Mensi et al. (2012) [1]_ and Pozzorini et al. (2015) [2]_, with exponential
    shaped postsynaptic currents.

    This is a brainpy.state re-implementation of the NEST simulator model of the
    same name, using NEST-standard parameterization and exact integration.

    This model features both an adaptation current and a dynamic threshold for
    spike-frequency adaptation. The membrane potential :math:`V` is described by
    the differential equation:

    .. math::

       C_\mathrm{m} \frac{dV(t)}{dt} = -g_\mathrm{L}(V(t) - E_\mathrm{L})
           - \eta_1(t) - \eta_2(t) - \ldots - \eta_n(t)
           + I(t)

    where each :math:`\eta_i` is a spike-triggered current (stc), and the neuron
    model can have an arbitrary number of them.

    Synaptic currents decay exponentially:

    .. math::

       \frac{dI_{\mathrm{syn,ex}}}{dt} = -\frac{I_{\mathrm{syn,ex}}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{dI_{\mathrm{syn,in}}}{dt} = -\frac{I_{\mathrm{syn,in}}}{\tau_{\mathrm{syn,in}}}.

    Spike-triggered currents
    ........................

    Dynamic of each :math:`\eta_i` is described by:

    .. math::

       \tau_{\eta_i} \cdot \frac{d\eta_i}{dt} = -\eta_i

    and in case of spike emission, its value is increased by a constant:

    .. math::

       \eta_i = \eta_i + q_{\eta_i} \quad \text{(on spike emission)}

    Spike-frequency adaptation
    ..........................

    The neuron produces spikes stochastically according to a point process with
    the firing intensity:

    .. math::

       \lambda(t) = \lambda_0 \cdot \exp\left(\frac{V(t) - V_T(t)}{\Delta_V}\right)

    where :math:`V_T(t)` is a time-dependent firing threshold:

    .. math::

       V_T(t) = V_{T^*} + \gamma_1(t) + \gamma_2(t) + \ldots + \gamma_m(t)

    where :math:`\gamma_i` is a kernel of spike-frequency adaptation (sfa).
    Dynamic of each :math:`\gamma_i` is described by:

    .. math::

       \tau_{\gamma_i} \cdot \frac{d\gamma_i}{dt} = -\gamma_i

    and in case of spike emission, its value is increased by a constant:

    .. math::

       \gamma_i = \gamma_i + q_{\gamma_i} \quad \text{(on spike emission)}

    Stochastic spiking
    ..................

    The probability of firing within a time step :math:`dt` is computed using
    the hazard function:

    .. math::

       P(\text{spike}) = 1 - \exp(-\lambda(t) \cdot dt)

    A random number is drawn each (non-refractory) time step and compared to
    this probability to determine whether a spike occurs.

    Refractory mechanism
    ....................

    After a spike, the neuron enters an absolute refractory period of duration
    :math:`t_\mathrm{ref}`. During this period:

    * the refractory counter decrements each step,
    * :math:`V_\mathrm{m}` is clamped to :math:`V_\mathrm{reset}`,
    * synaptic currents continue to decay and receive inputs.

    Numerical integration and update order
    ......................................

    NEST integrates this model with exact (analytic) propagators for the linear
    subthreshold dynamics. The discrete-time update order per simulation step is:

    1. Compute total stc (sum of stc elements) and sfa threshold (V_T_star + sum
       of sfa elements). Then decay all stc and sfa elements by their respective
       exponential factors.
    2. Decay synaptic currents: :math:`I_{\mathrm{syn}} \leftarrow I_{\mathrm{syn}} \cdot P_{11}`.
    3. Add synaptic weight jumps from spike inputs arriving this step.
    4. If not refractory: update membrane potential via exact propagator
       :math:`V \leftarrow P_{30}(I_\mathrm{stim} + I_e - \mathrm{stc})
       + P_{33} V + P_{31} E_L + I_{\mathrm{syn,ex}} P_{21,\mathrm{ex}}
       + I_{\mathrm{syn,in}} P_{21,\mathrm{in}}`.
       Compute firing intensity, draw random number, potentially emit spike
       (update stc/sfa elements, set refractory counter).
       If refractory: decrement counter, clamp V to V_reset.
    5. Store external current input as :math:`I_\mathrm{stim}` for the next step.

    .. note::

       In the NEST implementation, the stc and sfa element jumps occur immediately
       after spike emission. The GIF toolbox uses a different convention where
       jumps occur after the refractory period. Conversion:

       .. math::

          q_{\eta,\text{toolbox}} = q_{\eta,\text{NEST}} \cdot
              (1 - \exp(-t_\mathrm{ref} / \tau_\eta))

    .. note::

       If ``tau_m`` is very close to ``tau_syn_ex`` or ``tau_syn_in``, the model
       will numerically behave as if ``tau_m`` is equal to ``tau_syn_ex`` or
       ``tau_syn_in``, respectively, to avoid numerical instabilities.

    .. note::

       Because spiking is stochastic (random number drawn each step), exact
       spike-time reproducibility requires matching the random number generator
       state. For deterministic testing, set ``rng_key`` explicitly.

    Parameters
    ----------

    ==================== =================== =================================== =====================================================
    **Parameter**        **Default**         **Math equivalent**                 **Description**
    ==================== =================== =================================== =====================================================
    ``in_size``          (required)                                              Population shape
    ``g_L``              4.0 nS              :math:`g_\mathrm{L}`               Leak conductance
    ``E_L``              -70.0 mV            :math:`E_\mathrm{L}`               Leak reversal potential
    ``C_m``              80.0 pF             :math:`C_\mathrm{m}`               Membrane capacitance
    ``V_reset``          -55.0 mV            :math:`V_\mathrm{reset}`           Reset potential
    ``Delta_V``          0.5 mV              :math:`\Delta_V`                   Stochasticity level
    ``V_T_star``         -35.0 mV            :math:`V_{T^*}`                    Base firing threshold
    ``lambda_0``         1.0 /s              :math:`\lambda_0`                  Stochastic intensity at threshold
    ``t_ref``            4.0 ms              :math:`t_\mathrm{ref}`             Absolute refractory period
    ``tau_syn_ex``       2.0 ms              :math:`\tau_{\mathrm{syn,ex}}`     Excitatory synaptic time constant
    ``tau_syn_in``       2.0 ms              :math:`\tau_{\mathrm{syn,in}}`     Inhibitory synaptic time constant
    ``I_e``              0.0 pA              :math:`I_\mathrm{e}`               Constant external current
    ``tau_sfa``          () ms               :math:`\tau_{\gamma_i}`            SFA time constants (tuple/list)
    ``q_sfa``            () mV               :math:`q_{\gamma_i}`              SFA jump values (tuple/list)
    ``tau_stc``          () ms               :math:`\tau_{\eta_i}`              STC time constants (tuple/list)
    ``q_stc``            () nA               :math:`q_{\eta_i}`                STC jump values (tuple/list)
    ``rng_key``          None                                                    JAX PRNG key for stochastic spiking
    ``V_initializer``    Constant(-70 mV)                                        Initializer for membrane potential
    ``spk_fun``          ReluGrad()                                              Surrogate spike function
    ``spk_reset``        ``'hard'``                                              Reset mode; hard reset matches NEST
    ==================== =================== =================================== =====================================================

    State Variables
    ---------------

    ========================== ===========================================
    **State variable**         **Description**
    ========================== ===========================================
    ``V``                      Membrane potential :math:`V_\mathrm{m}`
    ``I_syn_ex``               Excitatory synaptic current
    ``I_syn_in``               Inhibitory synaptic current
    ``stc``                    Total spike-triggered current
    ``sfa``                    Adaptive threshold :math:`V_T(t)`
    ``stc_elems``              Individual stc adaptation elements
    ``sfa_elems``              Individual sfa adaptation elements
    ``refractory_step_count``  Remaining refractory grid steps
    ``I_stim``                 Buffered current applied in next step
    ``last_spike_time``        Last spike time
    ========================== ===========================================

    Notes
    -----

    - Defaults follow NEST C++ source for ``gif_psc_exp``.
    - ``lambda_0`` is specified in 1/s (as in NEST's Python interface) and is
      internally converted to 1/ms for computation.
    - Synaptic spike weights are interpreted in current units (pA), with
      positive/negative sign selecting excitatory/inhibitory channel.
    - The subthreshold dynamics use exact (analytic) integration via propagator
      coefficients, matching NEST's integration scheme. This is different from
      ``gif_cond_exp`` which uses RKF45 adaptive integration.

    References
    ----------
    .. [1] Mensi S, Naud R, Pozzorini C, Avermann M, Petersen CC, Gerstner W
           (2012). Parameter extraction and classification of three cortical
           neuron types reveals two distinct adaptation mechanisms. Journal of
           Neurophysiology, 107(6):1756-1775.
           DOI: https://doi.org/10.1152/jn.00408.2011
    .. [2] Pozzorini C, Mensi S, Hagens O, Naud R, Koch C, Gerstner W (2015).
           Automated high-throughput characterization of single neurons by means
           of simplified spiking models. PLoS Computational Biology, 11(6),
           e1004275.
           DOI: https://doi.org/10.1371/journal.pcbi.1004275
    .. [3] NEST Simulator ``gif_psc_exp`` model documentation and C++ source:
           ``models/gif_psc_exp.h`` and ``models/gif_psc_exp.cpp``.

    See Also
    --------
    gif_cond_exp, iaf_psc_exp
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        g_L: ArrayLike = 4.0 * u.nS,
        E_L: ArrayLike = -70.0 * u.mV,
        C_m: ArrayLike = 80.0 * u.pF,
        V_reset: ArrayLike = -55.0 * u.mV,
        Delta_V: ArrayLike = 0.5 * u.mV,
        V_T_star: ArrayLike = -35.0 * u.mV,
        lambda_0: float = 1.0,  # 1/s, as in NEST Python interface
        t_ref: ArrayLike = 4.0 * u.ms,
        tau_syn_ex: ArrayLike = 2.0 * u.ms,
        tau_syn_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0.0 * u.pA,
        tau_sfa: Sequence[float] = (),  # ms values
        q_sfa: Sequence[float] = (),  # mV values
        tau_stc: Sequence[float] = (),  # ms values
        q_stc: Sequence[float] = (),  # nA values
        rng_key: Optional[jax.Array] = None,
        V_initializer: Callable = braintools.init.Constant(-70.0 * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Membrane parameters
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.Delta_V = braintools.init.param(Delta_V, self.varshape)
        self.V_T_star = braintools.init.param(V_T_star, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        # Synaptic parameters
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)

        # Stochastic spiking: lambda_0 in 1/s, store as 1/ms internally
        self.lambda_0 = lambda_0 / 1000.0  # convert from 1/s to 1/ms

        # Adaptation parameters (stored as plain Python tuples of floats in ms/mV/nA)
        self.tau_sfa = tuple(float(x) for x in tau_sfa)
        self.q_sfa = tuple(float(x) for x in q_sfa)
        self.tau_stc = tuple(float(x) for x in tau_stc)
        self.q_stc = tuple(float(x) for x in q_stc)

        if len(self.tau_sfa) != len(self.q_sfa):
            raise ValueError(
                f"'tau_sfa' and 'q_sfa' must have the same length. "
                f"Got {len(self.tau_sfa)} and {len(self.q_sfa)}."
            )
        if len(self.tau_stc) != len(self.q_stc):
            raise ValueError(
                f"'tau_stc' and 'q_stc' must have the same length. "
                f"Got {len(self.tau_stc)} and {len(self.q_stc)}."
            )

        # RNG key for stochastic spiking
        self._rng_key = rng_key

        # Initializers
        self.V_initializer = V_initializer

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _sum_signed_delta_inputs(self):
        """Route delta inputs by sign: positive -> excitatory, negative -> inhibitory.

        This matches NEST's spike routing where each spike event is individually
        directed to the excitatory or inhibitory buffer based on weight sign.
        """
        w_ex = u.math.zeros_like(self.I_syn_ex.value)
        w_in = u.math.zeros_like(self.I_syn_in.value)
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
            w_in = w_in + u.math.minimum(out, zero)
        return w_ex, w_in

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.g_L, u.nS) <= 0.0):
            raise ValueError('Membrane conductance must be strictly positive.')
        if np.any(self._to_numpy(self.Delta_V, u.mV) <= 0.0):
            raise ValueError('Delta_V must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')
        if self.lambda_0 < 0.0:
            raise ValueError('lambda_0 must not be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or \
           np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('Synapse time constants must be strictly positive.')
        for tau in self.tau_sfa:
            if tau <= 0.0:
                raise ValueError('All SFA time constants must be strictly positive.')
        for tau in self.tau_stc:
            if tau <= 0.0:
                raise ValueError('All STC time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        # Adaptation state: stc and sfa element arrays (unitless floats in nA and mV respectively)
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)  # total stc current (nA)
        self._sfa_val = np.full(v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=np.float64)

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(self.V.value / u.mV))
        self.I_syn_ex.value = zeros * u.pA
        self.I_syn_in.value = zeros * u.pA
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)
        self._sfa_val = np.full(v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=np.float64)

        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_reset) / (self.Delta_V)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    @staticmethod
    def _propagator_exp(tau_syn: np.ndarray, tau_m: np.ndarray, c_m: np.ndarray, h_ms: float):
        """Compute the propagator coefficient P21 (I_syn -> V_m) for exact integration.

        This matches NEST's IAFPropagatorExp::evaluate() with singularity handling.

        Parameters
        ----------
        tau_syn : float or ndarray
            Synaptic time constant in ms.
        tau_m : float or ndarray
            Membrane time constant in ms.
        c_m : float or ndarray
            Membrane capacitance in pF.
        h_ms : float
            Time step in ms.

        Returns
        -------
        P21 : float or ndarray
            Propagator coefficient.
        """
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            beta = tau_syn * tau_m / (tau_m - tau_syn)
            gamma = beta / c_m
            inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)
            exp_h_tau_syn = np.exp(-h_ms / tau_syn)
            expm1_h_tau = np.expm1(h_ms * inv_beta)
            p32_raw = gamma * exp_h_tau_syn * expm1_h_tau

            normal_min = np.finfo(np.float64).tiny
            regular_mask = np.isfinite(p32_raw) & (np.abs(p32_raw) >= normal_min) & (p32_raw > 0.0)
            p32_singular = h_ms / c_m * np.exp(-h_ms / tau_m)
            return np.where(regular_mask, p32_raw, p32_singular)

    def update(self, x=0.0 * u.pA):
        """Update neuron state for one simulation step.

        Parameters
        ----------
        x : Quantity, optional
            External current input (pA). Default is 0.

        Returns
        -------
        spike : array
            Spike output (float, via surrogate gradient function).
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))  # dt in ms as float

        v_shape = self.V.value.shape

        # Extract state variables as numpy arrays
        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape).copy()
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape).copy()
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape).copy()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        ).copy()
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape).copy()

        # Extract parameters as numpy arrays
        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        g_L = self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        V_reset = self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape)
        tau_syn_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_syn_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        V_T_star = float(self._to_numpy(self.V_T_star, u.mV))
        Delta_V = float(self._to_numpy(self.Delta_V, u.mV))
        lambda_0 = self.lambda_0  # 1/ms

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )

        # Compute propagator coefficients (exact integration)
        tau_m = C_m / g_L  # membrane time constant in ms
        P33 = np.exp(-h / tau_m)
        P30 = -1.0 / C_m * np.expm1(-h / tau_m) * tau_m  # = tau_m/C_m * (1 - exp(-h/tau_m))
        P31 = -np.expm1(-h / tau_m)  # = 1 - exp(-h/tau_m)
        P11_ex = np.exp(-h / tau_syn_ex)
        P11_in = np.exp(-h / tau_syn_in)
        P21_ex = self._propagator_exp(tau_syn_ex, tau_m, C_m, h)
        P21_in = self._propagator_exp(tau_syn_in, tau_m, C_m, h)

        # Compute exponential decay factors for adaptation
        P_stc = [math.exp(-h / tau) for tau in self.tau_stc]
        P_sfa = [math.exp(-h / tau) for tau in self.tau_sfa]

        # Get synaptic inputs (spike weights: positive -> excitatory, negative -> inhibitory)
        w_ex_q, w_in_q = self._sum_signed_delta_inputs()
        w_ex = self._broadcast_to_state(self._to_numpy(w_ex_q, u.pA), v_shape)
        w_in = self._broadcast_to_state(self._to_numpy(w_in_q, u.pA), v_shape)

        # Get external current for NEXT step (NEST ring buffer semantics)
        new_i_stim = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        # Advance RNG state for this step
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(jax.random.uniform(subkey, shape=v_shape), dtype=np.float64)

        spike_mask = np.zeros_like(V, dtype=bool)

        for idx in np.ndindex(v_shape):
            # ---- Step 1: Decay stc/sfa elements and compute totals ----
            stc_total = 0.0
            if self._stc_elems is not None:
                for i in range(len(self.tau_stc)):
                    stc_total += self._stc_elems[i][idx]
                    self._stc_elems[i][idx] *= P_stc[i]

            sfa_total = V_T_star
            if self._sfa_elems is not None:
                for i in range(len(self.tau_sfa)):
                    sfa_total += self._sfa_elems[i][idx]
                    self._sfa_elems[i][idx] *= P_sfa[i]

            self._stc_val[idx] = stc_total
            self._sfa_val[idx] = sfa_total

            # ---- Step 2: Decay synaptic currents ----
            i_syn_ex[idx] *= P11_ex[idx]
            i_syn_in[idx] *= P11_in[idx]

            # ---- Step 3: Add synaptic weight jumps ----
            i_syn_ex[idx] += w_ex[idx]
            i_syn_in[idx] += w_in[idx]

            # ---- Step 4: Refractory / membrane update / spike check ----
            if r[idx] == 0:
                # Not refractory: update membrane potential via exact propagator
                V[idx] = (P30[idx] * (i_stim[idx] + I_e[idx] - stc_total)
                          + P33[idx] * V[idx]
                          + P31[idx] * E_L[idx]
                          + i_syn_ex[idx] * P21_ex[idx]
                          + i_syn_in[idx] * P21_in[idx])

                # Stochastic spike check
                lam = lambda_0 * math.exp((V[idx] - sfa_total) / Delta_V)
                if lam > 0.0:
                    spike_prob = -math.expm1(-lam * h)
                    if rand_vals[idx] < spike_prob:
                        # Spike!
                        spike_mask[idx] = True

                        # Jump stc elements
                        if self._stc_elems is not None:
                            for i in range(len(self.q_stc)):
                                self._stc_elems[i][idx] += self.q_stc[i]

                        # Jump sfa elements
                        if self._sfa_elems is not None:
                            for i in range(len(self.q_sfa)):
                                self._sfa_elems[i][idx] += self.q_sfa[i]

                        r[idx] = refr_counts[idx]
            else:
                # Refractory: decrement counter, clamp V to V_reset
                r[idx] -= 1
                V[idx] = V_reset[idx]

        # ---- Step 5: Store new I_stim for next step, update state ----
        self.V.value = V * u.mV
        self.I_syn_ex.value = i_syn_ex * u.pA
        self.I_syn_in.value = i_syn_in * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)
