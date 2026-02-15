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
from typing import Callable, Optional

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
    'pp_cond_exp_mc_urbanczik',
]

# Compartment indices
SOMA = 0
DEND = 1
NCOMP = 2

# State vector element indices (per compartment)
_V_M = 0
_G_EXC = 1
_G_INH = 2
_I_EXC = 3
_I_INH = 4
_STATE_VEC_COMPS = 5

# Total state vector size
_STATE_VEC_SIZE = _STATE_VEC_COMPS * NCOMP


def _idx(comp, elem):
    """Compute linear index into state array from compartment and element."""
    return comp * _STATE_VEC_COMPS + elem


def _phi(u_val, phi_max, rate_slope, beta, theta):
    r"""Rate function.

    .. math::

        \phi(u) = \frac{\phi_\mathrm{max}}{1 + k \cdot \exp(\beta \cdot (\theta - u))}

    Parameters
    ----------
    u_val : float
        Membrane potential (mV).
    phi_max : float
        Maximum firing rate (kHz).
    rate_slope : float
        Rate function slope parameter (k in the paper).
    beta : float
        Rate function steepness (1/mV).
    theta : float
        Rate function threshold (mV).

    Returns
    -------
    float
        Firing rate in kHz.
    """
    return phi_max / (1.0 + rate_slope * math.exp(beta * (theta - u_val)))


def _h_func(u_val, rate_slope, beta, theta):
    r"""Learning signal modulation function h(u).

    .. math::

        h(u) = \frac{15 \cdot \beta}{1 + \frac{1}{k} \cdot \exp(-\beta \cdot (\theta - u))}

    Parameters
    ----------
    u_val : float
        Membrane potential (mV).
    rate_slope : float
        Rate function slope parameter (k in the paper).
    beta : float
        Rate function steepness (1/mV).
    theta : float
        Rate function threshold (mV).

    Returns
    -------
    float
        h(u) value.
    """
    return 15.0 * beta / (1.0 + (1.0 / rate_slope) * math.exp(-beta * (theta - u_val)))


class pp_cond_exp_mc_urbanczik(Neuron):
    r"""Two-compartment point process neuron with conductance-based synapses.

    Description
    -----------

    ``pp_cond_exp_mc_urbanczik`` is an implementation of a two-compartment spiking
    point process neuron with conductance-based synapses as described in
    Urbanczik & Senn (2014) [1]_. It is the neuron model designed to be used with
    the Urbanczik-Senn learning rule.

    This is a brainpy.state re-implementation of the NEST simulator model of the
    same name, using NEST-standard parameterization.

    The model has two compartments: soma and dendrite, labeled as s and p,
    respectively. Each compartment can receive spike events and current input
    from a current generator. Additionally, an external (rheobase) current can
    be set for each compartment.

    Compartment structure
    .....................

    The neuron has a somatic and a dendritic compartment coupled by conductances.
    The soma uses **conductance-based** synapses (excitatory and inhibitory
    conductances multiplied by driving force), while the dendrite uses
    **current-based** synaptic inputs (excitatory and inhibitory currents injected
    directly). This asymmetry is by design in the Urbanczik-Senn formulation.

    Somatic dynamics
    ................

    .. math::

        C_\mathrm{m}^s \frac{dV^s}{dt} = -g_\mathrm{L}^s (V^s - E_\mathrm{L}^s)
            - g_\mathrm{ex}^s (V^s - E_\mathrm{ex}^s)
            - g_\mathrm{in}^s (V^s - E_\mathrm{in}^s)
            + g_\mathrm{sp} (V^p - V^s)
            + I_\mathrm{stim}^s + I_\mathrm{e}^s

    .. math::

        \frac{dg_\mathrm{ex}^s}{dt} = -\frac{g_\mathrm{ex}^s}{\tau_\mathrm{syn,ex}^s},
        \qquad
        \frac{dg_\mathrm{in}^s}{dt} = -\frac{g_\mathrm{in}^s}{\tau_\mathrm{syn,in}^s}

    Dendritic dynamics
    ..................

    .. math::

        C_\mathrm{m}^p \frac{dV^p}{dt} = -g_\mathrm{L}^p (V^p - E_\mathrm{L}^p)
            + I_\mathrm{ex}^p + I_\mathrm{in}^p
            + g_\mathrm{ps} (V^s - V^p)

    .. math::

        \frac{dI_\mathrm{ex}^p}{dt} = -\frac{I_\mathrm{ex}^p}{\tau_\mathrm{syn,ex}^p},
        \qquad
        \frac{dI_\mathrm{in}^p}{dt} = -\frac{I_\mathrm{in}^p}{\tau_\mathrm{syn,in}^p}

    Note that the dendritic compartment uses *current-based* synaptic inputs,
    not conductance-based.

    Stochastic spike generation
    ...........................

    Spikes are generated stochastically according to the instantaneous rate
    function evaluated at the somatic membrane potential:

    .. math::

        \text{rate} = 1000 \cdot \phi(V^s) \quad [\text{Hz}]

    where:

    .. math::

        \phi(u) = \frac{\phi_\mathrm{max}}{1 + k \cdot \exp(\beta (\theta - u))}

    * **With refractory period** (``t_ref > 0``): At most one spike per step.
      A uniform random number is compared to
      :math:`P(\text{spike}) = 1 - \exp(-\text{rate} \cdot h \cdot 10^{-3})`.
    * **Without refractory period** (``t_ref == 0``): Multiple spikes per step
      drawn from a Poisson distribution.

    There is **no membrane potential reset** after a spike. After spiking, the
    neuron enters a refractory period of ``t_ref`` ms during which no further
    spikes can occur.

    Receptor types
    ..............

    Synaptic inputs are addressed through labeled receptor types:

    =================== ====== ============================================
    Receptor type        Value  Description
    =================== ====== ============================================
    ``soma_exc``         1      Excitatory conductance input to soma
    ``soma_inh``         2      Inhibitory conductance input to soma
    ``dendritic_exc``    3      Excitatory current input to dendrite
    ``dendritic_inh``    4      Inhibitory current input to dendrite
    ``soma_curr``        5      Current injection to soma
    ``dendritic_curr``   6      Current injection to dendrite
    =================== ====== ============================================

    All synaptic weights must be positive. The distinction between excitatory
    and inhibitory is made by the receptor type.

    In this implementation, synaptic spike inputs are provided via the
    ``add_delta_input()`` mechanism with labels ``'soma_exc'``, ``'soma_inh'``,
    ``'dend_exc'``, ``'dend_inh'``. Current inputs are provided via
    ``add_current_input()`` with labels ``'soma'`` and ``'dend'``.

    Urbanczik-Senn learning signal
    ..............................

    At each time step, the model computes and stores the learning signal
    for the Urbanczik-Senn plasticity rule. The dendritic prediction of
    the somatic membrane potential is:

    .. math::

        V^*_W = \frac{E_\mathrm{L}^s \cdot g_\mathrm{L}^s + V^p \cdot g_\mathrm{sp}}{g_\mathrm{sp} + g_\mathrm{L}^s}

    The error signal contribution at each step is:

    .. math::

        \delta\Pi = \left(n_\mathrm{spikes} - \phi(V^*_W) \cdot dt\right) \cdot h(V^*_W)

    where:

    .. math::

        h(u) = \frac{15 \cdot \beta}{1 + \frac{1}{k} \cdot \exp(-\beta (\theta - u))}

    The history of :math:`\delta\Pi` values is stored for use by connecting
    Urbanczik synapses.

    Numerical integration
    .....................

    NEST uses the GSL RKF45 adaptive ODE solver. This implementation uses
    ``scipy.integrate.solve_ivp`` with method ``'RK45'`` and matching tolerances
    (rtol=0.0, atol=1e-3) to reproduce the same integration behavior.

    Update order per simulation step:

    1. Integrate the 10-dimensional ODE system over :math:`(t, t+dt]`.
    2. Add arriving synaptic spike inputs (conductance/current jumps).
    3. If not refractory: compute rate, draw random, potentially emit spike.
       If refractory: decrement counter.
    4. Write Urbanczik history (dPI).
    5. Store external current inputs for next step.

    Parameters
    ----------

    ==================== =================== =====================================================
    **Parameter**        **Default**         **Description**
    ==================== =================== =====================================================
    ``in_size``          (required)          Population shape
    ``t_ref``            3.0 ms              Duration of refractory period
    ``phi_max``          0.15                Maximum firing rate (kHz)
    ``rate_slope``       0.5                 Rate function slope (k in the paper)
    ``beta``             1/3 (1/mV)          Rate function steepness
    ``theta``            -55.0 mV            Rate function threshold
    ``g_sp``             600.0 nS            Soma-dendrite coupling conductance
    ``g_ps``             0.0 nS              Dendrite-soma coupling conductance
    ``soma_g_L``         30.0 nS             Somatic leak conductance
    ``soma_C_m``         300.0 pF            Somatic membrane capacitance
    ``soma_E_L``         -70.0 mV            Somatic leak reversal potential
    ``soma_E_ex``        0.0 mV              Somatic excitatory reversal potential
    ``soma_E_in``        -75.0 mV            Somatic inhibitory reversal potential
    ``soma_tau_syn_ex``  3.0 ms              Somatic excitatory synaptic time constant
    ``soma_tau_syn_in``  3.0 ms              Somatic inhibitory synaptic time constant
    ``soma_I_e``         0.0 pA              Somatic constant external current
    ``dend_g_L``         30.0 nS             Dendritic leak conductance
    ``dend_C_m``         300.0 pF            Dendritic membrane capacitance
    ``dend_E_L``         -70.0 mV            Dendritic leak reversal potential
    ``dend_E_ex``        0.0 mV              Dendritic excitatory reversal potential
    ``dend_E_in``        0.0 mV              Dendritic inhibitory reversal potential
    ``dend_tau_syn_ex``  3.0 ms              Dendritic excitatory synaptic time constant
    ``dend_tau_syn_in``  3.0 ms              Dendritic inhibitory synaptic time constant
    ``dend_I_e``         0.0 pA              Dendritic constant external current
    ``rng_key``          None                JAX PRNG key for stochastic spiking
    ``spk_fun``          ReluGrad()          Surrogate spike function
    ``spk_reset``        'hard'              Reset mode
    ==================== =================== =====================================================

    State Variables
    ---------------

    ============================== ===========================================
    **State variable**             **Description**
    ============================== ===========================================
    ``V_s``                        Somatic membrane potential (mV)
    ``g_ex_s``                     Somatic excitatory conductance (nS)
    ``g_in_s``                     Somatic inhibitory conductance (nS)
    ``V_d``                        Dendritic membrane potential (mV)
    ``I_ex_d``                     Dendritic excitatory current (pA)
    ``I_in_d``                     Dendritic inhibitory current (pA)
    ``refractory_step_count``      Remaining refractory grid steps
    ``I_stim_soma``                Buffered soma current for next step (pA)
    ``I_stim_dend``                Buffered dendrite current for next step (pA)
    ``last_spike_time``            Last spike time
    ============================== ===========================================

    Notes
    -----

    - All parameters match NEST C++ source defaults for ``pp_cond_exp_mc_urbanczik``.
    - The dendritic inhibitory reversal potential defaults to 0.0 mV (not -75.0 mV),
      matching NEST.
    - Because spiking is stochastic, exact spike-time reproducibility requires
      matching the random number generator state via ``rng_key``.
    - There is NO membrane potential reset after a spike.

    References
    ----------

    .. [1] Urbanczik R, Senn W (2014). Learning by the Dendritic Prediction of
           Somatic Spiking. Neuron, 81:521-528.
           DOI: https://doi.org/10.1016/j.neuron.2013.11.030
    .. [2] NEST Simulator ``pp_cond_exp_mc_urbanczik`` model documentation and
           C++ source: ``models/pp_cond_exp_mc_urbanczik.h`` and
           ``models/pp_cond_exp_mc_urbanczik.cpp``.

    See Also
    --------
    gif_cond_exp, pp_psc_delta
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        # Global parameters
        t_ref: ArrayLike = 3.0 * u.ms,
        phi_max: float = 0.15,      # kHz
        rate_slope: float = 0.5,     # dimensionless
        beta: float = 1.0 / 3.0,    # 1/mV
        theta: float = -55.0,        # mV
        g_sp: ArrayLike = 600.0 * u.nS,  # soma-dendrite coupling
        g_ps: ArrayLike = 0.0 * u.nS,    # dendrite-soma coupling
        # Soma compartment parameters
        soma_g_L: ArrayLike = 30.0 * u.nS,
        soma_C_m: ArrayLike = 300.0 * u.pF,
        soma_E_L: ArrayLike = -70.0 * u.mV,
        soma_E_ex: ArrayLike = 0.0 * u.mV,
        soma_E_in: ArrayLike = -75.0 * u.mV,
        soma_tau_syn_ex: ArrayLike = 3.0 * u.ms,
        soma_tau_syn_in: ArrayLike = 3.0 * u.ms,
        soma_I_e: ArrayLike = 0.0 * u.pA,
        # Dendritic compartment parameters
        dend_g_L: ArrayLike = 30.0 * u.nS,
        dend_C_m: ArrayLike = 300.0 * u.pF,
        dend_E_L: ArrayLike = -70.0 * u.mV,
        dend_E_ex: ArrayLike = 0.0 * u.mV,
        dend_E_in: ArrayLike = 0.0 * u.mV,
        dend_tau_syn_ex: ArrayLike = 3.0 * u.ms,
        dend_tau_syn_in: ArrayLike = 3.0 * u.ms,
        dend_I_e: ArrayLike = 0.0 * u.pA,
        # RNG and surrogate
        rng_key: Optional[jax.Array] = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Global parameters
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.phi_max = float(phi_max)
        self.rate_slope = float(rate_slope)
        self.beta = float(beta)
        self.theta = float(theta)
        self.g_sp = braintools.init.param(g_sp, self.varshape)
        self.g_ps = braintools.init.param(g_ps, self.varshape)

        # Soma parameters
        self.soma_g_L = braintools.init.param(soma_g_L, self.varshape)
        self.soma_C_m = braintools.init.param(soma_C_m, self.varshape)
        self.soma_E_L = braintools.init.param(soma_E_L, self.varshape)
        self.soma_E_ex = braintools.init.param(soma_E_ex, self.varshape)
        self.soma_E_in = braintools.init.param(soma_E_in, self.varshape)
        self.soma_tau_syn_ex = braintools.init.param(soma_tau_syn_ex, self.varshape)
        self.soma_tau_syn_in = braintools.init.param(soma_tau_syn_in, self.varshape)
        self.soma_I_e = braintools.init.param(soma_I_e, self.varshape)

        # Dendritic parameters
        self.dend_g_L = braintools.init.param(dend_g_L, self.varshape)
        self.dend_C_m = braintools.init.param(dend_C_m, self.varshape)
        self.dend_E_L = braintools.init.param(dend_E_L, self.varshape)
        self.dend_E_ex = braintools.init.param(dend_E_ex, self.varshape)
        self.dend_E_in = braintools.init.param(dend_E_in, self.varshape)
        self.dend_tau_syn_ex = braintools.init.param(dend_tau_syn_ex, self.varshape)
        self.dend_tau_syn_in = braintools.init.param(dend_tau_syn_in, self.varshape)
        self.dend_I_e = braintools.init.param(dend_I_e, self.varshape)

        # RNG
        self._rng_key = rng_key

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if self.rate_slope < 0:
            raise ValueError('Rate slope cannot be negative.')
        if self.phi_max < 0:
            raise ValueError('Maximum rate cannot be negative.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        for label, C_m in [('soma', self.soma_C_m), ('dendritic', self.dend_C_m)]:
            if np.any(self._to_numpy(C_m, u.pF) <= 0.0):
                raise ValueError(f'Capacitance ({label}) must be strictly positive.')
        for label, tse, tsi in [
            ('soma', self.soma_tau_syn_ex, self.soma_tau_syn_in),
            ('dendritic', self.dend_tau_syn_ex, self.dend_tau_syn_in),
        ]:
            if np.any(self._to_numpy(tse, u.ms) <= 0.0) or np.any(self._to_numpy(tsi, u.ms) <= 0.0):
                raise ValueError('All time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        # Membrane potentials initialized to E_L
        soma_E_L = self._to_numpy(self.soma_E_L, u.mV)
        dend_E_L = self._to_numpy(self.dend_E_L, u.mV)

        self.V_s = brainstate.HiddenState(
            np.broadcast_to(soma_E_L, v_shape).copy() * u.mV
        )
        self.V_d = brainstate.HiddenState(
            np.broadcast_to(dend_E_L, v_shape).copy() * u.mV
        )

        # Somatic conductances
        self.g_ex_s = brainstate.HiddenState(np.zeros(v_shape, dtype=np.float64) * u.nS)
        self.g_in_s = brainstate.HiddenState(np.zeros(v_shape, dtype=np.float64) * u.nS)

        # Dendritic currents
        self.I_ex_d = brainstate.HiddenState(np.zeros(v_shape, dtype=np.float64) * u.pA)
        self.I_in_d = brainstate.HiddenState(np.zeros(v_shape, dtype=np.float64) * u.pA)

        # Refractory counter
        self.refractory_step_count = brainstate.ShortTermState(
            jnp.zeros(v_shape, dtype=jnp.int32)
        )

        # Buffered stimulus currents (per compartment)
        self.I_stim_soma = brainstate.ShortTermState(np.zeros(v_shape, dtype=np.float64) * u.pA)
        self.I_stim_dend = brainstate.ShortTermState(np.zeros(v_shape, dtype=np.float64) * u.pA)

        # Last spike time
        self.last_spike_time = brainstate.ShortTermState(
            np.full(v_shape, -1e7, dtype=np.float64) * u.ms
        )

        # Urbanczik history: list of (t_ms, dPI) tuples per neuron element
        self._urbanczik_history = {}  # key: flat index -> list of (t, dPI)

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def reset_state(self, batch_size: int = None, **kwargs):
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        soma_E_L = self._to_numpy(self.soma_E_L, u.mV)
        dend_E_L = self._to_numpy(self.dend_E_L, u.mV)

        self.V_s.value = np.broadcast_to(soma_E_L, v_shape).copy() * u.mV
        self.V_d.value = np.broadcast_to(dend_E_L, v_shape).copy() * u.mV
        self.g_ex_s.value = np.zeros(v_shape, dtype=np.float64) * u.nS
        self.g_in_s.value = np.zeros(v_shape, dtype=np.float64) * u.nS
        self.I_ex_d.value = np.zeros(v_shape, dtype=np.float64) * u.pA
        self.I_in_d.value = np.zeros(v_shape, dtype=np.float64) * u.pA
        self.refractory_step_count.value = jnp.zeros(v_shape, dtype=jnp.int32)
        self.I_stim_soma.value = np.zeros(v_shape, dtype=np.float64) * u.pA
        self.I_stim_dend.value = np.zeros(v_shape, dtype=np.float64) * u.pA
        self.last_spike_time.value = np.full(v_shape, -1e7, dtype=np.float64) * u.ms
        self._urbanczik_history = {}

        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def get_spike(self, V: ArrayLike = None):
        V = self.V_s.value if V is None else V
        v_scaled = V / (1.0 * u.mV)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.round(self.t_ref / dt), dtype=jnp.int32)

    def _collect_receptor_delta_inputs(self):
        """Collect delta inputs labeled by receptor type.

        Expected labels: 'soma_exc', 'soma_inh', 'dend_exc', 'dend_inh'.

        Returns
        -------
        soma_exc, soma_inh, dend_exc, dend_inh : Quantity arrays (nS or pA)
        """
        v_shape = self.V_s.value.shape

        soma_exc = u.math.zeros(v_shape) * u.nS
        soma_inh = u.math.zeros(v_shape) * u.nS
        dend_exc = u.math.zeros(v_shape) * u.pA
        dend_inh = u.math.zeros(v_shape) * u.pA

        if self.delta_inputs is None:
            return soma_exc, soma_inh, dend_exc, dend_inh

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            label = key if isinstance(key, str) else str(key)

            if 'soma_exc' in label:
                soma_exc = soma_exc + out
            elif 'soma_inh' in label:
                soma_inh = soma_inh + out
            elif 'dend_exc' in label:
                dend_exc = dend_exc + out
            elif 'dend_inh' in label:
                dend_inh = dend_inh + out

        return soma_exc, soma_inh, dend_exc, dend_inh

    @staticmethod
    def _dynamics(t, y, p):
        """ODE right-hand side for the 10-dimensional state vector.

        Matches NEST's ``pp_cond_exp_mc_urbanczik_dynamics()`` exactly.

        Parameters
        ----------
        t : float
            Time (unused, autonomous system).
        y : array of float, shape (10,)
            State vector [V_s, G_ex_s, G_in_s, I_ex_s(=0), I_in_s(=0),
                          V_d, G_ex_d(=0), G_in_d(=0), I_ex_d, I_in_d].
        p : dict
            Parameters dict.

        Returns
        -------
        f : array of float, shape (10,)
            Derivatives.
        """
        f = np.zeros(10)

        # Soma membrane potential
        V_s = y[_idx(SOMA, _V_M)]

        # Soma leak current
        I_L_s = p['g_L_soma'] * (V_s - p['E_L_soma'])

        # Soma excitatory synaptic current (conductance-based)
        I_syn_exc = y[_idx(SOMA, _G_EXC)] * (V_s - p['E_ex_soma'])

        # Soma inhibitory synaptic current (conductance-based)
        I_syn_inh = y[_idx(SOMA, _G_INH)] * (V_s - p['E_in_soma'])

        # Coupling from dendrites to soma
        I_conn_d_s = 0.0

        # Dendrite (n=1, DEND)
        V_d = y[_idx(DEND, _V_M)]

        # Coupling current dendrite -> soma
        I_conn_d_s += p['g_conn_soma'] * (V_d - V_s)

        # Coupling current soma -> dendrite
        I_conn_s_d = p['g_conn_dend'] * (V_s - V_d)

        # Dendritic synaptic currents (current-based)
        I_syn_ex_d = y[_idx(DEND, _I_EXC)]
        I_syn_in_d = y[_idx(DEND, _I_INH)]

        # Dendrite membrane potential derivative
        f[_idx(DEND, _V_M)] = (
            -p['g_L_dend'] * (V_d - p['E_L_dend'])
            + I_syn_ex_d + I_syn_in_d + I_conn_s_d
        ) / p['C_m_dend']

        # Dendrite current derivatives
        f[_idx(DEND, _I_EXC)] = -I_syn_ex_d / p['tau_syn_ex_dend']
        f[_idx(DEND, _I_INH)] = -I_syn_in_d / p['tau_syn_in_dend']

        # Dendrite unused channels
        f[_idx(DEND, _G_EXC)] = 0.0
        f[_idx(DEND, _G_INH)] = 0.0

        # Soma membrane potential derivative
        f[_idx(SOMA, _V_M)] = (
            -I_L_s - I_syn_exc - I_syn_inh + I_conn_d_s
            + p['I_stim_soma'] + p['I_e_soma']
        ) / p['C_m_soma']

        # Soma conductance derivatives
        f[_idx(SOMA, _G_EXC)] = -y[_idx(SOMA, _G_EXC)] / p['tau_syn_ex_soma']
        f[_idx(SOMA, _G_INH)] = -y[_idx(SOMA, _G_INH)] / p['tau_syn_in_soma']

        # Soma unused channels
        f[_idx(SOMA, _I_EXC)] = 0.0
        f[_idx(SOMA, _I_INH)] = 0.0

        return f

    def update(self, x=0.0 * u.pA):
        """Update neuron state for one simulation step.

        Parameters
        ----------
        x : Quantity, optional
            External current input (pA), applied to soma. Default is 0.

        Returns
        -------
        spike : array
            Spike output (float array; >0 indicates spike).
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))  # dt in ms

        v_shape = self.V_s.value.shape

        # Extract state as numpy arrays
        V_s = self._broadcast_to_state(self._to_numpy(self.V_s.value, u.mV), v_shape).copy()
        V_d = self._broadcast_to_state(self._to_numpy(self.V_d.value, u.mV), v_shape).copy()
        g_ex_s = self._broadcast_to_state(self._to_numpy(self.g_ex_s.value, u.nS), v_shape).copy()
        g_in_s = self._broadcast_to_state(self._to_numpy(self.g_in_s.value, u.nS), v_shape).copy()
        I_ex_d = self._broadcast_to_state(self._to_numpy(self.I_ex_d.value, u.pA), v_shape).copy()
        I_in_d = self._broadcast_to_state(self._to_numpy(self.I_in_d.value, u.pA), v_shape).copy()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        ).copy()
        i_stim_soma = self._broadcast_to_state(
            self._to_numpy(self.I_stim_soma.value, u.pA), v_shape
        ).copy()
        i_stim_dend = self._broadcast_to_state(
            self._to_numpy(self.I_stim_dend.value, u.pA), v_shape
        ).copy()

        # Extract parameters
        p_base = {
            'g_L_soma': self._broadcast_to_state(self._to_numpy(self.soma_g_L, u.nS), v_shape),
            'C_m_soma': self._broadcast_to_state(self._to_numpy(self.soma_C_m, u.pF), v_shape),
            'E_L_soma': self._broadcast_to_state(self._to_numpy(self.soma_E_L, u.mV), v_shape),
            'E_ex_soma': self._broadcast_to_state(self._to_numpy(self.soma_E_ex, u.mV), v_shape),
            'E_in_soma': self._broadcast_to_state(self._to_numpy(self.soma_E_in, u.mV), v_shape),
            'tau_syn_ex_soma': self._broadcast_to_state(self._to_numpy(self.soma_tau_syn_ex, u.ms), v_shape),
            'tau_syn_in_soma': self._broadcast_to_state(self._to_numpy(self.soma_tau_syn_in, u.ms), v_shape),
            'I_e_soma': self._broadcast_to_state(self._to_numpy(self.soma_I_e, u.pA), v_shape),
            'g_L_dend': self._broadcast_to_state(self._to_numpy(self.dend_g_L, u.nS), v_shape),
            'C_m_dend': self._broadcast_to_state(self._to_numpy(self.dend_C_m, u.pF), v_shape),
            'E_L_dend': self._broadcast_to_state(self._to_numpy(self.dend_E_L, u.mV), v_shape),
            'tau_syn_ex_dend': self._broadcast_to_state(self._to_numpy(self.dend_tau_syn_ex, u.ms), v_shape),
            'tau_syn_in_dend': self._broadcast_to_state(self._to_numpy(self.dend_tau_syn_in, u.ms), v_shape),
            'g_conn_soma': self._broadcast_to_state(self._to_numpy(self.g_sp, u.nS), v_shape),
            'g_conn_dend': self._broadcast_to_state(self._to_numpy(self.g_ps, u.nS), v_shape),
        }

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )

        # Collect synaptic spike inputs
        d_soma_exc, d_soma_inh, d_dend_exc, d_dend_inh = self._collect_receptor_delta_inputs()
        d_soma_exc_np = self._broadcast_to_state(self._to_numpy(d_soma_exc, u.nS), v_shape)
        d_soma_inh_np = self._broadcast_to_state(self._to_numpy(d_soma_inh, u.nS), v_shape)
        d_dend_exc_np = self._broadcast_to_state(self._to_numpy(d_dend_exc, u.pA), v_shape)
        d_dend_inh_np = self._broadcast_to_state(self._to_numpy(d_dend_inh, u.pA), v_shape)

        # Collect current inputs for next step
        new_i_stim_soma = self._broadcast_to_state(
            self._to_numpy(self.sum_current_inputs(x, self.V_s.value), u.pA), v_shape
        )
        # Note: dendritic current inputs are zero by default unless explicitly provided
        new_i_stim_dend = np.zeros(v_shape, dtype=np.float64)

        # Advance RNG
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(jax.random.uniform(subkey, shape=v_shape), dtype=np.float64)

        spike_mask = np.zeros(v_shape, dtype=bool)

        # Compute step time for urbanczik history
        t_ms = float(u.math.asarray(t / u.ms)) + dt

        for idx in np.ndindex(v_shape):
            # Build per-element parameter dict
            p = {k: p_base[k][idx] for k in p_base}
            p['I_stim_soma'] = i_stim_soma[idx]

            # Build state vector
            y0 = np.zeros(_STATE_VEC_SIZE)
            y0[_idx(SOMA, _V_M)] = V_s[idx]
            y0[_idx(SOMA, _G_EXC)] = g_ex_s[idx]
            y0[_idx(SOMA, _G_INH)] = g_in_s[idx]
            y0[_idx(SOMA, _I_EXC)] = 0.0
            y0[_idx(SOMA, _I_INH)] = 0.0
            y0[_idx(DEND, _V_M)] = V_d[idx]
            y0[_idx(DEND, _G_EXC)] = 0.0
            y0[_idx(DEND, _G_INH)] = 0.0
            y0[_idx(DEND, _I_EXC)] = I_ex_d[idx]
            y0[_idx(DEND, _I_INH)] = I_in_d[idx]

            # ---- Step 1: Integrate ODE ----
            sol = solve_ivp(
                lambda t_ode, y_ode: self._dynamics(t_ode, y_ode, p),
                [0.0, dt],
                y0,
                method='RK45',
                rtol=0.0,
                atol=1e-3,
            )
            yf = sol.y[:, -1]

            # ---- Step 2: Add arriving synaptic spike inputs ----
            # Soma: conductance jumps
            yf[_idx(SOMA, _G_EXC)] += d_soma_exc_np[idx]
            yf[_idx(SOMA, _G_INH)] += d_soma_inh_np[idx]

            # Dendrite: current jumps (note: inhibitory is SUBTRACTED, matching NEST)
            yf[_idx(DEND, _I_EXC)] += d_dend_exc_np[idx]
            yf[_idx(DEND, _I_INH)] -= d_dend_inh_np[idx]

            # ---- Step 3: Spike check / refractory ----
            n_spikes = 0

            if r[idx] == 0:
                # Neuron not refractory
                # No V_m reset after spike
                rate = 1000.0 * _phi(
                    yf[_idx(SOMA, _V_M)],
                    self.phi_max, self.rate_slope, self.beta, self.theta
                )

                if rate > 0.0:
                    t_ref_val = float(self._to_numpy(self.t_ref, u.ms).flat[0]) if \
                        np.ndim(self._to_numpy(self.t_ref, u.ms)) > 0 else \
                        float(self._to_numpy(self.t_ref, u.ms))

                    if t_ref_val > 0.0:
                        # With dead time: at most 1 spike
                        if rand_vals[idx] <= -math.expm1(-rate * dt * 1e-3):
                            n_spikes = 1
                    else:
                        # No dead time: Poisson spikes
                        lam = rate * dt * 1e-3
                        n_spikes = int(np.random.RandomState(
                            int(rand_vals[idx] * 2**31)
                        ).poisson(lam))

                    if n_spikes > 0:
                        spike_mask[idx] = True
                        r[idx] = refr_counts[idx]
            else:
                # Refractory: decrement
                r[idx] -= 1

            # ---- Step 4: Write Urbanczik history ----
            V_d_current = yf[_idx(DEND, _V_M)]
            g_D = p['g_conn_soma']  # g_sp
            g_L_s = p['g_L_soma']
            E_L_s = p['E_L_soma']
            V_W_star = (E_L_s * g_L_s + V_d_current * g_D) / (g_D + g_L_s)

            dPI = (n_spikes - _phi(V_W_star, self.phi_max, self.rate_slope,
                                   self.beta, self.theta) * dt) * \
                  _h_func(V_W_star, self.rate_slope, self.beta, self.theta)

            flat_idx = np.ravel_multi_index(idx, v_shape) if len(idx) > 0 else 0
            if flat_idx not in self._urbanczik_history:
                self._urbanczik_history[flat_idx] = []
            self._urbanczik_history[flat_idx].append((t_ms, dPI))

            # ---- Write back state ----
            V_s[idx] = yf[_idx(SOMA, _V_M)]
            g_ex_s[idx] = yf[_idx(SOMA, _G_EXC)]
            g_in_s[idx] = yf[_idx(SOMA, _G_INH)]
            V_d[idx] = yf[_idx(DEND, _V_M)]
            I_ex_d[idx] = yf[_idx(DEND, _I_EXC)]
            I_in_d[idx] = yf[_idx(DEND, _I_INH)]

        # ---- Step 5: Store new I_stim for next step ----
        self.V_s.value = V_s * u.mV
        self.V_d.value = V_d * u.mV
        self.g_ex_s.value = g_ex_s * u.nS
        self.g_in_s.value = g_in_s * u.nS
        self.I_ex_d.value = I_ex_d * u.pA
        self.I_in_d.value = I_in_d * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.I_stim_soma.value = new_i_stim_soma * u.pA
        self.I_stim_dend.value = new_i_stim_dend * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)

    def get_urbanczik_history(self, neuron_idx=0):
        """Get the Urbanczik learning signal history for a neuron.

        Parameters
        ----------
        neuron_idx : int
            Flat index of the neuron.

        Returns
        -------
        history : list of (t_ms, dPI) tuples
        """
        return self._urbanczik_history.get(neuron_idx, [])
