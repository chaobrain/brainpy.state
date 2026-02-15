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
from scipy.integrate import solve_ivp

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'hh_psc_alpha',
]


def _hh_psc_alpha_equilibrium(V):
    """Compute HH gating variable equilibrium values at voltage V (mV).

    Returns (m_inf, h_inf, n_inf) at the given membrane potential, using the
    same alpha/beta rate functions as NEST ``hh_psc_alpha``.
    """
    alpha_n = (0.01 * (V + 55.0)) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
    alpha_m = (0.1 * (V + 40.0)) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
    alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    m_inf = alpha_m / (alpha_m + beta_m)
    h_inf = alpha_h / (alpha_h + beta_h)
    n_inf = alpha_n / (alpha_n + beta_n)
    return m_inf, h_inf, n_inf


class hh_psc_alpha(Neuron):
    r"""NEST-compatible ``hh_psc_alpha`` neuron model.

    Short description
    -----------------

    Hodgkin-Huxley neuron model with alpha-shaped postsynaptic currents.

    Description
    -----------

    ``hh_psc_alpha`` is a spiking neuron using the Hodgkin-Huxley formalism
    with:

    - sodium (Na), potassium (K), and leak (L) conductances,
    - alpha-function shaped postsynaptic currents (PSCs),
    - combined threshold-and-local-maximum spike detection,
    - explicit refractory period (suppresses spike emission only; dynamics
      evolve freely during refractoriness).

    This implementation mirrors the NEST ``models/hh_psc_alpha.{h,cpp}``
    update ordering and parameterization, using an adaptive Runge-Kutta
    integrator (RK45, Dormand-Prince) to match NEST's GSL RKF45.

    Membrane and ionic current dynamics
    ....................................

    The membrane potential evolves as

    .. math::

       C_m \frac{dV_m}{dt} = -(I_{Na} + I_K + I_L) + I_{stim} + I_e
                              + I_{syn,ex} + I_{syn,in}

    where

    .. math::

       I_{Na} &= g_{Na}\, m^3\, h\, (V_m - E_{Na})  \\
       I_K    &= g_K\,   n^4\,     (V_m - E_K)       \\
       I_L    &= g_L\,             (V_m - E_L)

    Gating variables :math:`m`, :math:`h`, :math:`n` obey

    .. math::

       \frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)\,x

    with rate functions (voltage :math:`V` in mV, rates in 1/ms):

    .. math::

       \alpha_n &= \frac{0.01\,(V + 55)}{1 - e^{-(V+55)/10}}, \quad
       \beta_n  = 0.125\,e^{-(V+65)/80}                                   \\
       \alpha_m &= \frac{0.1\,(V + 40)}{1 - e^{-(V+40)/10}}, \quad
       \beta_m  = 4\,e^{-(V+65)/18}                                       \\
       \alpha_h &= 0.07\,e^{-(V+65)/20}, \quad
       \beta_h  = \frac{1}{1 + e^{-(V+35)/10}}

    Alpha-function synaptic currents
    .................................

    Each synapse type (excitatory / inhibitory) is modelled as a
    second-order system producing an alpha-shaped postsynaptic current:

    .. math::

       \frac{dI_{syn}}{dt}  &= dI_{syn} - \frac{I_{syn}}{\tau_{syn}} \\
       \frac{d(dI_{syn})}{dt} &= -\frac{dI_{syn}}{\tau_{syn}}

    A spike arriving with weight :math:`w` adds
    :math:`w \cdot e / \tau_{syn}` to :math:`dI_{syn}`, normalizing the
    peak current to :math:`w` pA for :math:`w = 1`.

    Spike detection
    ...............

    A spike is detected when the membrane potential crosses 0 mV from
    below **and** a local maximum is detected (i.e. the potential starts
    decreasing). Formally, a spike is emitted when:

    1. ``r == 0`` (not in refractory period), **and**
    2. ``V_m >= 0 mV``, **and**
    3. ``V_old > V_m`` (local maximum, the potential is now falling).

    Unlike integrate-and-fire models, no voltage reset occurs -- the
    potassium current naturally repolarizes the membrane.

    Numerical integration
    .....................

    NEST uses GSL RKF45 (Runge-Kutta-Fehlberg 4/5) with adaptive step-size
    control (relative tolerance 1e-3, absolute tolerance 0). This
    implementation uses ``scipy.integrate.solve_ivp`` with method ``'RK45'``
    (Dormand-Prince) at the same tolerances for matching numerical results.

    Parameters
    ----------

    ==================== ================== =============================== ====================================================
    **Parameter**        **Default**        **Math equivalent**             **Description**
    ==================== ================== =============================== ====================================================
    ``in_size``          (required)                                         Population shape
    ``E_L``              -54.402 mV         :math:`E_L`                     Leak reversal potential (resting potential)
    ``C_m``              100 pF             :math:`C_m`                     Membrane capacitance
    ``g_Na``             12000 nS           :math:`g_{Na}`                  Sodium peak conductance
    ``g_K``              3600 nS            :math:`g_K`                     Potassium peak conductance
    ``g_L``              30 nS              :math:`g_L`                     Leak conductance
    ``E_Na``             50 mV              :math:`E_{Na}`                  Sodium reversal potential
    ``E_K``              -77 mV             :math:`E_K`                     Potassium reversal potential
    ``t_ref``            2 ms               :math:`t_{ref}`                 Duration of refractory period
    ``tau_syn_ex``       0.2 ms             :math:`\tau_{syn,ex}`           Excitatory synaptic time constant
    ``tau_syn_in``       2 ms               :math:`\tau_{syn,in}`           Inhibitory synaptic time constant
    ``I_e``              0 pA               :math:`I_e`                     Constant external input current
    ``V_m_init``         -65 mV                                             Initial membrane potential
    ``Act_m_init``       None                                               Initial Na activation (None -> equilibrium at V_m_init)
    ``Inact_h_init``     None                                               Initial Na inactivation (None -> equilibrium at V_m_init)
    ``Act_n_init``       None                                               Initial K activation (None -> equilibrium at V_m_init)
    ``spk_fun``          ReluGrad()                                         Surrogate spike function
    ``spk_reset``        ``'hard'``                                         Reset mode
    ``rtol``             1e-3                                               Relative tolerance for ODE solver
    ``atol``             1e-9                                               Absolute tolerance for ODE solver
    ==================== ================== =============================== ====================================================

    State variables
    ---------------

    - ``V``: membrane potential :math:`V_m` (mV).
    - ``m``: Na activation gating variable.
    - ``h``: Na inactivation gating variable.
    - ``n``: K activation gating variable.
    - ``I_syn_ex``: excitatory postsynaptic current (pA).
    - ``I_syn_in``: inhibitory postsynaptic current (pA).
    - ``dI_syn_ex``: excitatory alpha-kernel derivative state.
    - ``dI_syn_in``: inhibitory alpha-kernel derivative state.
    - ``I_stim``: stimulation current buffer (pA).
    - ``refractory_step_count``: refractory countdown in grid steps.
    - ``last_spike_time``: time of most recent spike.

    Notes
    -----

    - Unlike IAF models, the HH model does **not** reset the membrane
      potential after a spike. Repolarization occurs naturally through
      the potassium current.
    - During the refractory period, the neuron's subthreshold dynamics
      continue to evolve freely; only spike emission is suppressed.
    - Spike weights are interpreted as current amplitudes (pA).
      Positive weights are excitatory; negative weights are inhibitory.

    References
    ----------
    .. [1] Hodgkin AL, Huxley AF (1952). A quantitative description of
           membrane current and its application to conduction and excitation
           in nerve. The Journal of Physiology 117:500-544.
           DOI: https://doi.org/10.1113/jphysiol.1952.sp004764
    .. [2] Gerstner W, Kistler W (2002). Spiking neuron models: Single
           neurons, populations, plasticity. Cambridge University Press.
    .. [3] Dayan P, Abbott LF (2001). Theoretical neuroscience: Computational
           and mathematical modeling of neural systems. MIT Press.

    See also
    --------
    iaf_psc_alpha : Leaky integrate-and-fire with alpha-shaped PSCs.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -54.402 * u.mV,
        C_m: ArrayLike = 100. * u.pF,
        g_Na: ArrayLike = 12000. * u.nS,
        g_K: ArrayLike = 3600. * u.nS,
        g_L: ArrayLike = 30. * u.nS,
        E_Na: ArrayLike = 50. * u.mV,
        E_K: ArrayLike = -77. * u.mV,
        t_ref: ArrayLike = 2. * u.ms,
        tau_syn_ex: ArrayLike = 0.2 * u.ms,
        tau_syn_in: ArrayLike = 2. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        V_m_init: ArrayLike = -65. * u.mV,
        Act_m_init: ArrayLike = None,
        Inact_h_init: ArrayLike = None,
        Act_n_init: ArrayLike = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        rtol: float = 1e-3,
        atol: float = 1e-9,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.g_Na = braintools.init.param(g_Na, self.varshape)
        self.g_K = braintools.init.param(g_K, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.E_Na = braintools.init.param(E_Na, self.varshape)
        self.E_K = braintools.init.param(E_K, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_m_init = V_m_init
        self.Act_m_init = Act_m_init
        self.Inact_h_init = Inact_h_init
        self.Act_n_init = Act_n_init
        self.rtol = rtol
        self.atol = atol

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
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.g_Na, u.nS) < 0.0) or np.any(self._to_numpy(self.g_K, u.nS) < 0.0) or np.any(self._to_numpy(self.g_L, u.nS) < 0.0):
            raise ValueError('All conductances must be non-negative.')

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def init_state(self, batch_size: int = None, **kwargs):
        V_init_mV = self._to_numpy(self.V_m_init, u.mV)
        V_init_scalar = float(V_init_mV.flat[0]) if V_init_mV.ndim > 0 else float(V_init_mV)

        # Compute equilibrium gating variables at initial V
        m_eq, h_eq, n_eq = _hh_psc_alpha_equilibrium(V_init_scalar)

        V = braintools.init.param(braintools.init.Constant(self.V_m_init), self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        if self.Act_m_init is not None:
            m_init = self._to_numpy(self.Act_m_init, u.UNITLESS).item()
        else:
            m_init = m_eq
        if self.Inact_h_init is not None:
            h_init = self._to_numpy(self.Inact_h_init, u.UNITLESS).item()
        else:
            h_init = h_eq
        if self.Act_n_init is not None:
            n_init = self._to_numpy(self.Act_n_init, u.UNITLESS).item()
        else:
            n_init = n_eq

        self.V = brainstate.HiddenState(V)
        self.m = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(m_init), self.varshape, batch_size)
        )
        self.h = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(h_init), self.varshape, batch_size)
        )
        self.n = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(n_init), self.varshape, batch_size)
        )
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.dI_syn_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.dI_syn_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.I_stim = brainstate.ShortTermState(zeros * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        # For HH neurons, spike threshold is 0 mV. Scale relative to 0 mV.
        v_scaled = V / (1. * u.mV)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.pA):
        r"""Update neuron state for one simulation step.

        The update follows the NEST ``hh_psc_alpha`` update order:

        1. Record pre-integration membrane potential (``V_old``).
        2. Integrate the full 8-dimensional ODE system over one time step
           using an adaptive RK45 solver.
        3. Add arriving synaptic spike inputs to ``dI_syn_ex`` / ``dI_syn_in``.
        4. Check spike condition: ``V_m >= 0 and V_old > V_m`` (threshold +
           local maximum).
        5. Update refractory counter and record spike time.
        6. Store buffered stimulation current for the next step.

        Parameters
        ----------
        x : ArrayLike, default 0 pA
            External stimulation current input (in addition to ``I_e``).

        Returns
        -------
        ArrayLike
            Spike output with shape ``(batch_size, *in_size)``.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        # Extract parameters as numpy float64
        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        g_Na = self._broadcast_to_state(self._to_numpy(self.g_Na, u.nS), v_shape)
        g_K = self._broadcast_to_state(self._to_numpy(self.g_K, u.nS), v_shape)
        g_L = self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape)
        E_Na = self._broadcast_to_state(self._to_numpy(self.E_Na, u.mV), v_shape)
        E_K = self._broadcast_to_state(self._to_numpy(self.E_K, u.mV), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)

        # Current state
        V_m = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        m_val = self._broadcast_to_state(np.asarray(self.m.value, dtype=np.float64), v_shape)
        h_val = self._broadcast_to_state(np.asarray(self.h.value, dtype=np.float64), v_shape)
        n_val = self._broadcast_to_state(np.asarray(self.n.value, dtype=np.float64), v_shape)
        dI_ex = self._broadcast_to_state(np.asarray(self.dI_syn_ex.value, dtype=np.float64), v_shape)
        I_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape)
        dI_in = self._broadcast_to_state(np.asarray(self.dI_syn_in.value, dtype=np.float64), v_shape)
        I_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape)
        I_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )

        # PSC normalization: e / tau ensures peak current = weight for weight=1.
        psc_init_ex = math.e / tau_ex
        psc_init_in = math.e / tau_in

        # Collect spike/current inputs
        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all > 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)
        I_stim_next = self._broadcast_to_state(
            self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape
        )

        # Record V before integration for spike detection
        V_old = V_m.copy()

        # Integrate ODE for each neuron independently
        flat_size = int(np.prod(v_shape)) if len(v_shape) > 0 else 1
        V_new = np.empty(flat_size, dtype=np.float64)
        m_new = np.empty(flat_size, dtype=np.float64)
        h_new = np.empty(flat_size, dtype=np.float64)
        n_new = np.empty(flat_size, dtype=np.float64)
        dI_ex_new = np.empty(flat_size, dtype=np.float64)
        I_ex_new = np.empty(flat_size, dtype=np.float64)
        dI_in_new = np.empty(flat_size, dtype=np.float64)
        I_in_new = np.empty(flat_size, dtype=np.float64)

        V_m_flat = V_m.ravel()
        m_flat = m_val.ravel()
        h_flat = h_val.ravel()
        n_flat = n_val.ravel()
        dI_ex_flat = dI_ex.ravel()
        I_ex_flat = I_ex.ravel()
        dI_in_flat = dI_in.ravel()
        I_in_flat = I_in.ravel()
        I_stim_flat = I_stim.ravel()
        g_Na_flat = g_Na.ravel()
        g_K_flat = g_K.ravel()
        g_L_flat = g_L.ravel()
        E_Na_flat = E_Na.ravel()
        E_K_flat = E_K.ravel()
        E_L_flat = E_L.ravel()
        C_m_flat = C_m.ravel()
        I_e_flat = I_e.ravel()
        tau_ex_flat = tau_ex.ravel()
        tau_in_flat = tau_in.ravel()

        for i in range(flat_size):
            y0 = np.array([
                V_m_flat[i], m_flat[i], h_flat[i], n_flat[i],
                dI_ex_flat[i], I_ex_flat[i], dI_in_flat[i], I_in_flat[i]
            ])

            # Capture per-neuron parameters for closure
            _g_Na = g_Na_flat[i]
            _g_K = g_K_flat[i]
            _g_L = g_L_flat[i]
            _E_Na = E_Na_flat[i]
            _E_K = E_K_flat[i]
            _E_L = E_L_flat[i]
            _C_m = C_m_flat[i]
            _I_e = I_e_flat[i]
            _I_stim = I_stim_flat[i]
            _tau_ex = tau_ex_flat[i]
            _tau_in = tau_in_flat[i]

            def rhs(t_local, y,
                    _g_Na=_g_Na, _g_K=_g_K, _g_L=_g_L,
                    _E_Na=_E_Na, _E_K=_E_K, _E_L=_E_L,
                    _C_m=_C_m, _I_e=_I_e, _I_stim=_I_stim,
                    _tau_ex=_tau_ex, _tau_in=_tau_in):
                V = y[0]
                m_ = y[1]
                h_ = y[2]
                n_ = y[3]
                dI_e = y[4]
                I_e_ = y[5]
                dI_i = y[6]
                I_i_ = y[7]

                alpha_n = (0.01 * (V + 55.0)) / (1.0 - math.exp(-(V + 55.0) / 10.0))
                beta_n = 0.125 * math.exp(-(V + 65.0) / 80.0)
                alpha_m = (0.1 * (V + 40.0)) / (1.0 - math.exp(-(V + 40.0) / 10.0))
                beta_m = 4.0 * math.exp(-(V + 65.0) / 18.0)
                alpha_h = 0.07 * math.exp(-(V + 65.0) / 20.0)
                beta_h = 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))

                I_Na = _g_Na * m_ * m_ * m_ * h_ * (V - _E_Na)
                I_K = _g_K * n_ * n_ * n_ * n_ * (V - _E_K)
                I_L = _g_L * (V - _E_L)

                f = np.empty(8)
                f[0] = (-(I_Na + I_K + I_L) + _I_stim + _I_e + I_e_ + I_i_) / _C_m
                f[1] = alpha_m * (1.0 - m_) - beta_m * m_
                f[2] = alpha_h * (1.0 - h_) - beta_h * h_
                f[3] = alpha_n * (1.0 - n_) - beta_n * n_
                f[4] = -dI_e / _tau_ex
                f[5] = dI_e - (I_e_ / _tau_ex)
                f[6] = -dI_i / _tau_in
                f[7] = dI_i - (I_i_ / _tau_in)
                return f

            sol = solve_ivp(
                rhs,
                [0.0, h],
                y0,
                method='RK45',
                rtol=self.rtol,
                atol=self.atol,
                dense_output=False,
            )
            yf = sol.y[:, -1]
            V_new[i] = yf[0]
            m_new[i] = yf[1]
            h_new[i] = yf[2]
            n_new[i] = yf[3]
            dI_ex_new[i] = yf[4]
            I_ex_new[i] = yf[5]
            dI_in_new[i] = yf[6]
            I_in_new[i] = yf[7]

        V_m = V_new.reshape(v_shape)
        m_val = m_new.reshape(v_shape)
        h_val = h_new.reshape(v_shape)
        n_val = n_new.reshape(v_shape)
        dI_ex = dI_ex_new.reshape(v_shape)
        I_ex = I_ex_new.reshape(v_shape)
        dI_in = dI_in_new.reshape(v_shape)
        I_in = I_in_new.reshape(v_shape)

        # Add arriving spike inputs to dI (after ODE integration, matching NEST)
        dI_ex = dI_ex + w_ex * psc_init_ex
        dI_in = dI_in + w_in * psc_init_in

        # Spike detection: threshold crossing + local maximum
        V_old_flat = V_old.ravel()
        V_m_flat = V_m.ravel()

        not_refractory = r == 0
        crossed_threshold = V_m >= 0.0
        local_max = V_old > V_m
        spike_cond = not_refractory & crossed_threshold & local_max

        # Refractory update
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )
        r_new = np.where(spike_cond, refr_counts, np.where(r > 0, r - 1, r))

        # Write back state
        self.V.value = V_m * u.mV
        self.m.value = m_val
        self.h.value = h_val
        self.n.value = n_val
        self.I_syn_ex.value = I_ex * u.pA
        self.I_syn_in.value = I_in * u.pA
        self.dI_syn_ex.value = dI_ex
        self.dI_syn_in.value = dI_in
        self.I_stim.value = I_stim_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r_new, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        # Return spike output: only signal a spike when spike_cond is True
        V_out = np.where(spike_cond, 1e-12, -1.0)
        return self.get_spike(V_out * u.mV)
