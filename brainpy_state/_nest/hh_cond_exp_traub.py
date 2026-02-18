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

from ._base import NESTNeuron

__all__ = [
    'hh_cond_exp_traub',
]


def _hh_cond_exp_traub_equilibrium(V):
    r"""Compute Traub HH gating variable equilibrium values at voltage V (mV).

    This matches NEST's ``State_::State_(const Parameters_&)`` initialization,
    which applies the Traub rate equations **without** the V_T offset.  The
    dynamics function uses ``V - V_T`` in its rate equations, but the
    equilibrium initialization in NEST uses the raw voltage ``y_[0]`` (= E_L).

    Parameters
    ----------
    V : float
        Membrane potential in mV (unitless scalar). Must be a finite value.

    Returns
    -------
    m_inf : float
        Sodium activation equilibrium value (0 ≤ m_inf ≤ 1).
    h_inf : float
        Sodium inactivation equilibrium value (0 ≤ h_inf ≤ 1).
    n_inf : float
        Potassium activation equilibrium value (0 ≤ n_inf ≤ 1).

    Notes
    -----
    The function evaluates the steady-state gating variables using Traub-Miles
    rate equations at the unshifted voltage V (not V - V_T). This is the
    correct initialization procedure for NEST compatibility, where initial
    gating states are computed at the raw E_L value, not shifted by V_T.

    The alpha/beta rate functions may encounter division by zero at special
    voltage values (e.g., V = 15 mV for alpha_n). These singularities are
    removable via L'Hospital's rule but may cause numerical issues if V is
    exactly at these points.

    Mathematical Formulation
    ------------------------
    Equilibrium values are computed from the Traub-Miles rate equations:

    .. math::

       x_{\infty}(V) = \frac{\alpha_x(V)}{\alpha_x(V) + \beta_x(V)}

    **1. Potassium Activation (n)**

    .. math::

       \alpha_n &= \frac{0.032(15 - V)}{e^{(15-V)/5} - 1} \\
       \beta_n  &= 0.5 \, e^{(10-V)/40}

    **2. Sodium Activation (m)**

    .. math::

       \alpha_m &= \frac{0.32(13 - V)}{e^{(13-V)/4} - 1} \\
       \beta_m  &= \frac{0.28(V - 40)}{e^{(V-40)/5} - 1}

    **3. Sodium Inactivation (h)**

    .. math::

       \alpha_h &= 0.128 \, e^{(17-V)/18} \\
       \beta_h  &= \frac{4}{1 + e^{(40-V)/5}}

    Examples
    --------
    .. code-block:: python

       >>> from brainpy_state._nest.hh_cond_exp_traub import _hh_cond_exp_traub_equilibrium
       >>> m_inf, h_inf, n_inf = _hh_cond_exp_traub_equilibrium(-60.0)
       >>> print(f"m={m_inf:.4f}, h={h_inf:.4f}, n={n_inf:.4f}")
       m=0.0529, h=0.5961, n=0.3177

    See Also
    --------
    hh_cond_exp_traub : The neuron model class that uses these equilibrium values.
    """
    alpha_n = 0.032 * (15.0 - V) / (math.exp((15.0 - V) / 5.0) - 1.0)
    beta_n = 0.5 * math.exp((10.0 - V) / 40.0)
    alpha_m = 0.32 * (13.0 - V) / (math.exp((13.0 - V) / 4.0) - 1.0)
    beta_m = 0.28 * (V - 40.0) / (math.exp((V - 40.0) / 5.0) - 1.0)
    alpha_h = 0.128 * math.exp((17.0 - V) / 18.0)
    beta_h = 4.0 / (1.0 + math.exp((40.0 - V) / 5.0))
    m_inf = alpha_m / (alpha_m + beta_m)
    h_inf = alpha_h / (alpha_h + beta_h)
    n_inf = alpha_n / (alpha_n + beta_n)
    return m_inf, h_inf, n_inf


class hh_cond_exp_traub(NESTNeuron):
    r"""NEST-compatible ``hh_cond_exp_traub`` neuron model.

    Hodgkin-Huxley model for Brette et al. (2007) review, based on Traub and
    Miles (1991) hippocampal pyramidal cell model.

    This is a modified Hodgkin-Huxley neuron model specifically developed for
    the Brette et al. (2007) simulator review, based on a model of hippocampal
    pyramidal cells by Traub and Miles (1991). Key differences from the original
    Traub-Miles model:

    - This is a point neuron, not a compartmental model.
    - Only ``I_Na`` and ``I_K`` ionic currents are included (no calcium dynamics),
      with simplified ``I_K`` dynamics giving three gating variables instead of
      eight.
    - Incoming spikes induce an instantaneous conductance change followed by
      exponential decay (conductance-based synapses), not activation over time.

    Parameters
    ----------
    in_size : int, tuple of int
        Population shape (number of neurons or spatial dimensions).
    E_L : ArrayLike, default -60 mV
        Leak reversal potential. Must be finite.
    C_m : ArrayLike, default 200 pF
        Membrane capacitance. Must be strictly positive.
    g_Na : ArrayLike, default 20000 nS
        Sodium peak conductance. Must be non-negative.
    g_K : ArrayLike, default 6000 nS
        Potassium peak conductance. Must be non-negative.
    g_L : ArrayLike, default 10 nS
        Leak conductance. Must be non-negative.
    E_Na : ArrayLike, default 50 mV
        Sodium reversal potential. Must be finite.
    E_K : ArrayLike, default -90 mV
        Potassium reversal potential. Must be finite.
    V_T : ArrayLike, default -63 mV
        Voltage offset for gating dynamics. Shifts the effective threshold
        to approximately V_T + 30 mV.
    E_ex : ArrayLike, default 0 mV
        Excitatory synaptic reversal potential. Must be finite.
    E_in : ArrayLike, default -80 mV
        Inhibitory synaptic reversal potential. Must be finite.
    t_ref : ArrayLike, default 2 ms
        Duration of refractory period. Must be non-negative. Traub and Miles
        used 3 ms; NEST default is 2 ms.
    tau_syn_ex : ArrayLike, default 5 ms
        Excitatory synaptic time constant. Must be strictly positive.
    tau_syn_in : ArrayLike, default 10 ms
        Inhibitory synaptic time constant. Must be strictly positive.
    I_e : ArrayLike, default 0 pA
        Constant external input current. Can be positive or negative.
    V_m_init : ArrayLike, optional
        Initial membrane potential. If None, defaults to E_L.
    Act_m_init : ArrayLike, optional
        Initial sodium activation gating variable (0 ≤ m ≤ 1). If None,
        computed from equilibrium at V_m_init.
    Inact_h_init : ArrayLike, optional
        Initial sodium inactivation gating variable (0 ≤ h ≤ 1). If None,
        computed from equilibrium at V_m_init.
    Act_n_init : ArrayLike, optional
        Initial potassium activation gating variable (0 ≤ n ≤ 1). If None,
        computed from equilibrium at V_m_init.
    spk_fun : Callable, default braintools.surrogate.ReluGrad()
        Surrogate spike function for differentiable spike generation.
    spk_reset : str, default 'hard'
        Reset mode ('hard' or 'soft'). Note: HH models do not reset voltage
        after spikes; this parameter affects gradient computation only.
    rtol : float, default 1e-3
        Relative tolerance for ODE solver (matches NEST GSL RKF45).
    atol : float, default 1e-9
        Absolute tolerance for ODE solver.
    name : str, optional
        Name of the neuron population.

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential with shape (\*in_size,) in mV.
    m : brainstate.HiddenState
        Sodium activation gating variable (0 ≤ m ≤ 1), shape (\*in_size,).
    h : brainstate.HiddenState
        Sodium inactivation gating variable (0 ≤ h ≤ 1), shape (\*in_size,).
    n : brainstate.HiddenState
        Potassium activation gating variable (0 ≤ n ≤ 1), shape (\*in_size,).
    g_ex : brainstate.HiddenState
        Excitatory synaptic conductance in nS, shape (\*in_size,).
    g_in : brainstate.HiddenState
        Inhibitory synaptic conductance in nS, shape (\*in_size,).
    I_stim : brainstate.ShortTermState
        Stimulation current buffer in pA, shape (\*in_size,).
    refractory_step_count : brainstate.ShortTermState
        Refractory countdown in grid steps, shape (\*in_size,), dtype int32.
    last_spike_time : brainstate.ShortTermState
        Time of most recent spike in ms, shape (\*in_size,).

    Raises
    ------
    ValueError
        If C_m ≤ 0, t_ref < 0, tau_syn_ex ≤ 0, or tau_syn_in ≤ 0.

    Notes
    -----
    - Unlike IAF models, the HH model does **not** reset the membrane
      potential after a spike. Repolarization occurs naturally through
      the potassium current.
    - During the refractory period, subthreshold dynamics continue to
      evolve freely; only spike emission is suppressed.
    - Synaptic spike weights are interpreted in conductance units (nS).
      Positive weights drive excitatory synapses; negative weights drive
      inhibitory synapses (sign is flipped, i.e. ``g_in += |w|``).
    - The numerical integration uses scipy's RK45 (Dormand-Prince) solver
      with adaptive step-size control. This is equivalent to NEST's GSL
      RKF45 implementation for numerical correspondence.

    Mathematical Formulation
    -------------------------

    **1. Membrane and Ionic Current Dynamics**

    The membrane potential evolves as:

    .. math::

       C_m \frac{dV_m}{dt} = -(I_{Na} + I_K + I_L + I_{syn,ex} + I_{syn,in})
                              + I_{stim} + I_e

    where the currents are:

    .. math::

       I_{Na}     &= g_{Na}\, m^3\, h\, (V_m - E_{Na})  \\
       I_K        &= g_K\,   n^4\,     (V_m - E_K)       \\
       I_L        &= g_L\,             (V_m - E_L)        \\
       I_{syn,ex} &= g_{ex}\,          (V_m - E_{ex})     \\
       I_{syn,in} &= g_{in}\,          (V_m - E_{in})

    **2. Channel Gating Variables**

    Gating variables :math:`m`, :math:`h`, :math:`n` obey first-order kinetics:

    .. math::

       \frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)\,x
                     = \alpha_x - (\alpha_x + \beta_x)\, x

    with Traub-Miles rate functions using shifted voltage
    :math:`V = V_m - V_T` (voltage in mV, rates in 1/ms):

    .. math::

       \alpha_n &= \frac{0.032\,(15 - V)}{e^{(15 - V)/5} - 1}, \quad
       \beta_n  = 0.5\,e^{(10 - V)/40}                                \\
       \alpha_m &= \frac{0.32\,(13 - V)}{e^{(13 - V)/4} - 1}, \quad
       \beta_m  = \frac{0.28\,(V - 40)}{e^{(V - 40)/5} - 1}          \\
       \alpha_h &= 0.128\,e^{(17 - V)/18}, \quad
       \beta_h  = \frac{4}{1 + e^{(40 - V)/5}}

    The voltage offset :math:`V_T` (default -63 mV) shifts the effective
    threshold to approximately -50 mV.

    **3. Exponential Conductance Synapses**

    Synaptic conductances decay exponentially:

    .. math::

       \frac{dg_{ex}}{dt} &= -g_{ex} / \tau_{syn,ex} \\
       \frac{dg_{in}}{dt} &= -g_{in} / \tau_{syn,in}

    A presynaptic spike with weight :math:`w` causes an instantaneous
    conductance jump:

    - :math:`w > 0` -- :math:`g_{ex} \leftarrow g_{ex} + w`
    - :math:`w < 0` -- :math:`g_{in} \leftarrow g_{in} + |w|`

    **4. Spike Detection**

    A spike is emitted when all three conditions are satisfied:

    1. ``r == 0`` (not in refractory period), **and**
    2. ``V_m >= V_T + 30`` mV (threshold crossing), **and**
    3. ``V_old > V_m`` (local maximum, the potential is now falling).

    Unlike integrate-and-fire models, no voltage reset occurs -- the
    potassium current naturally repolarizes the membrane.

    .. warning::

       To avoid multiple spikes during the falling flank of a spike, it is
       essential to choose a sufficiently long refractory period.
       Traub and Miles used :math:`t_{ref} = 3` ms, while the default here
       is :math:`t_{ref} = 2` ms (matching NEST).

    **5. Numerical Integration**

    NEST uses GSL RKF45 (Runge-Kutta-Fehlberg 4/5) with adaptive step-size
    control (relative tolerance 1e-3, absolute tolerance 0). This
    implementation uses ``scipy.integrate.solve_ivp`` with method ``'RK45'``
    (Dormand-Prince) at matching tolerances for numerical correspondence.

    The ODE system is 6-dimensional per neuron:
    :math:`[V_m, m, h, n, g_{ex}, g_{in}]`.

    Parameter Mapping
    -----------------

    The following table shows the correspondence between brainpy.state parameters
    and NEST/mathematical notation:

    ==================== ================== =============================== ====================================================
    **Parameter**        **Default**        **Math equivalent**             **Description**
    ==================== ================== =============================== ====================================================
    ``in_size``          (required)         —                               Population shape
    ``E_L``              -60 mV             :math:`E_L`                     Leak reversal potential
    ``C_m``              200 pF             :math:`C_m`                     Membrane capacitance
    ``g_Na``             20000 nS           :math:`g_{Na}`                  Sodium peak conductance
    ``g_K``              6000 nS            :math:`g_K`                     Potassium peak conductance
    ``g_L``              10 nS              :math:`g_L`                     Leak conductance
    ``E_Na``             50 mV              :math:`E_{Na}`                  Sodium reversal potential
    ``E_K``              -90 mV             :math:`E_K`                     Potassium reversal potential
    ``V_T``              -63 mV             :math:`V_T`                     Voltage offset for gating dynamics
    ``E_ex``             0 mV               :math:`E_{ex}`                  Excitatory synaptic reversal potential
    ``E_in``             -80 mV             :math:`E_{in}`                  Inhibitory synaptic reversal potential
    ``t_ref``            2 ms               :math:`t_{ref}`                 Duration of refractory period
    ``tau_syn_ex``       5 ms               :math:`\tau_{syn,ex}`           Excitatory synaptic time constant
    ``tau_syn_in``       10 ms              :math:`\tau_{syn,in}`           Inhibitory synaptic time constant
    ``I_e``              0 pA               :math:`I_e`                     Constant external input current
    ``V_m_init``         None               —                               Initial V_m (None → E_L)
    ``Act_m_init``       None               —                               Initial Na activation (None → equilibrium)
    ``Inact_h_init``     None               —                               Initial Na inactivation (None → equilibrium)
    ``Act_n_init``       None               —                               Initial K activation (None → equilibrium)
    ``spk_fun``          ReluGrad()         —                               Surrogate spike function
    ``spk_reset``        ``'hard'``         —                               Reset mode
    ``rtol``             1e-3               —                               Relative tolerance for ODE solver
    ``atol``             1e-9               —                               Absolute tolerance for ODE solver
    ==================== ================== =============================== ====================================================

    Examples
    --------
    .. code-block:: python

       >>> import brainstate as bst
       >>> import brainunit as u
       >>> from brainpy_state import hh_cond_exp_traub
       >>>
       >>> # Create a population of 100 Traub HH neurons
       >>> neurons = hh_cond_exp_traub(100)
       >>> neurons.init_all_states()
       >>>
       >>> # Run a simulation with constant current injection
       >>> with bst.environ.context(dt=0.1*u.ms):
       ...     for i in range(1000):
       ...         spikes = neurons.update(I_e=200*u.pA)

    .. code-block:: python

       >>> # Compare with NEST default parameters
       >>> import nest
       >>> nest_neuron = nest.Create('hh_cond_exp_traub')
       >>> nest.GetStatus(nest_neuron, ['V_m', 'E_L', 'C_m', 'g_Na', 'g_K'])
       [(-60.0, -60.0, 200.0, 20000.0, 6000.0)]
       >>>
       >>> # Match in brainpy.state
       >>> bp_neuron = hh_cond_exp_traub(1, E_L=-60*u.mV, C_m=200*u.pF,
       ...                               g_Na=20000*u.nS, g_K=6000*u.nS)

    References
    ----------
    .. [1] Brette R et al. (2007). Simulation of networks of spiking neurons:
           A review of tools and strategies. Journal of Computational
           Neuroscience 23:349-98.
           DOI: https://doi.org/10.1007/s10827-007-0038-6
    .. [2] Traub RD and Miles R (1991). Neuronal networks of the hippocampus.
           Cambridge University Press, Cambridge UK.
    .. [3] ModelDB entry: http://modeldb.yale.edu/83319

    See Also
    --------
    hh_psc_alpha : Hodgkin-Huxley with alpha-shaped postsynaptic currents.
    iaf_cond_exp : Leaky integrate-and-fire with conductance-based synapses.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -60. * u.mV,
        C_m: ArrayLike = 200. * u.pF,
        g_Na: ArrayLike = 20000. * u.nS,
        g_K: ArrayLike = 6000. * u.nS,
        g_L: ArrayLike = 10. * u.nS,
        E_Na: ArrayLike = 50. * u.mV,
        E_K: ArrayLike = -90. * u.mV,
        V_T: ArrayLike = -63. * u.mV,
        E_ex: ArrayLike = 0. * u.mV,
        E_in: ArrayLike = -80. * u.mV,
        t_ref: ArrayLike = 2. * u.ms,
        tau_syn_ex: ArrayLike = 5. * u.ms,
        tau_syn_in: ArrayLike = 10. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        V_m_init: ArrayLike = None,
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
        self.V_T = braintools.init.param(V_T, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
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
        r"""Convert brainunit quantity to numpy float64 array.

        Parameters
        ----------
        x : ArrayLike
            Input value with brainunit units.
        unit : brainunit.Unit
            Unit to divide by for conversion.

        Returns
        -------
        np.ndarray
            Unitless numpy array with dtype=float64.

        Examples
        --------
        .. code-block:: python

           >>> import brainunit as u
           >>> from brainpy_state._nest.hh_cond_exp_traub import hh_cond_exp_traub
           >>> result = hh_cond_exp_traub._to_numpy(200*u.pF, u.pF)
           >>> print(result)
           200.0
        """
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast numpy array to target shape.

        Parameters
        ----------
        x_np : np.ndarray
            Input numpy array to broadcast.
        shape : tuple
            Target shape for broadcasting.

        Returns
        -------
        np.ndarray
            Broadcasted array with the specified shape.

        Notes
        -----
        Uses numpy's broadcasting rules. Raises ValueError if shapes are
        incompatible.
        """
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        r"""Validate parameter constraints.

        Raises
        ------
        ValueError
            If capacitance C_m ≤ 0, refractory time t_ref < 0, or any synaptic
            time constant (tau_syn_ex, tau_syn_in) ≤ 0.

        Notes
        -----
        This is called during __init__ to ensure physical validity of parameters.
        Conductances (g_L, g_Na, g_K) are not validated for positivity since
        zero conductance is physically meaningful (though unusual).
        """
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')

    def _refractory_counts(self):
        r"""Compute refractory period length in simulation time steps.

        Returns
        -------
        jnp.ndarray
            Number of time steps for refractory period, shape (\*in_size,),
            dtype=int32. Computed as ceil(t_ref / dt).

        Notes
        -----
        The refractory period is discretized to the simulation grid. If t_ref
        is not an exact multiple of dt, the period is rounded up to ensure
        the minimum refractory time is respected.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate as bst
           >>> import brainunit as u
           >>> from brainpy_state import hh_cond_exp_traub
           >>>
           >>> neurons = hh_cond_exp_traub(10, t_ref=2.0*u.ms)
           >>> with bst.environ.context(dt=0.1*u.ms):
           ...     ref_steps = neurons._refractory_counts()
           >>> print(ref_steps)  # 2.0 / 0.1 = 20 steps
           20
        """
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Initializes membrane potential, gating variables, synaptic conductances,
        stimulation current buffer, refractory counter, and last spike time. If
        initial values are not explicitly provided, they are computed as follows:

        - ``V``: defaults to ``E_L``
        - ``m, h, n``: computed from equilibrium at initial ``V`` using Traub-Miles
          rate equations (without V_T offset, matching NEST initialization)
        - ``g_ex, g_in``: initialized to zero
        - ``I_stim``: initialized to zero
        - ``refractory_step_count``: initialized to zero (not refractory)
        - ``last_spike_time``: initialized to -1e7 ms (far in the past)

        Parameters
        ----------
        batch_size : int, optional
            Number of batches for parallel simulation. If None, no batch dimension
            is added. State shapes will be (batch_size, \*in_size) if provided,
            otherwise (\*in_size,).
        **kwargs
            Additional keyword arguments (reserved for future use).

        Notes
        -----
        The equilibrium gating variable computation uses the raw voltage V (not
        V - V_T) to match NEST's initialization procedure. During dynamics, the
        rate equations use the shifted voltage V - V_T, but initialization uses
        the unshifted value for consistency with NEST's ``State_::State_``
        constructor.

        This initialization ensures the neuron starts in a stable resting state
        when V_m_init = E_L (default). For custom initial voltages, gating
        variables are automatically adjusted to the corresponding equilibrium.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate as bst
           >>> import brainunit as u
           >>> from brainpy_state import hh_cond_exp_traub
           >>>
           >>> # Initialize with default rest state
           >>> neurons = hh_cond_exp_traub(100)
           >>> neurons.init_state()
           >>> print(neurons.V.value[0])  # Should be E_L = -60 mV
           -60.0 mV
           >>>
           >>> # Initialize with custom voltage
           >>> neurons = hh_cond_exp_traub(100, V_m_init=-65*u.mV)
           >>> neurons.init_state()
           >>> print(neurons.V.value[0])
           -65.0 mV
           >>>
           >>> # Initialize with batch dimension
           >>> neurons = hh_cond_exp_traub(50)
           >>> neurons.init_state(batch_size=10)
           >>> print(neurons.V.value.shape)  # (10, 50)
           (10, 50)

        See Also
        --------
        _hh_cond_exp_traub_equilibrium : Computes equilibrium gating values.
        """
        # Default V_m_init to E_L (matching NEST: y_[0] = p.E_L)
        if self.V_m_init is not None:
            V_init_val = self.V_m_init
        else:
            V_init_val = self.E_L

        V_init_mV = self._to_numpy(V_init_val, u.mV)
        V_init_scalar = float(V_init_mV.flat[0]) if V_init_mV.ndim > 0 else float(V_init_mV)

        # Compute equilibrium gating variables at initial V.
        # NEST uses raw V_m (not V_m - V_T) for equilibrium initialization.
        m_eq, h_eq, n_eq = _hh_cond_exp_traub_equilibrium(V_init_scalar)

        V = braintools.init.param(braintools.init.Constant(V_init_val), self.varshape, batch_size)
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
        self.g_ex = brainstate.HiddenState(zeros * u.nS)
        self.g_in = brainstate.HiddenState(zeros * u.nS)
        self.I_stim = brainstate.ShortTermState(zeros * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output using surrogate gradient function.

        Applies the surrogate spike function to the membrane potential. This is
        used for gradient-based learning; actual spike detection in the update
        method uses discrete threshold crossing logic (V >= V_T + 30 and local
        maximum).

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential in mV, shape (\*in_size,) or (batch_size, \*in_size).
            If None, uses the current state ``self.V.value``.

        Returns
        -------
        ArrayLike
            Differentiable spike output with the same shape as input V. Values are
            approximately 0 (no spike) or 1 (spike) with smooth gradients for
            backpropagation.

        Notes
        -----
        The voltage is scaled to unitless values (mV) before applying the surrogate
        function. For Hodgkin-Huxley neurons, the actual spike threshold is
        V_T + 30 mV (default: -33 mV), but the surrogate function operates on
        the raw scaled voltage for gradient computation.

        This method is primarily used for surrogate gradient learning. The discrete
        spike detection logic in the update method is independent and uses the
        three-condition test (refractory, threshold, local maximum).

        Examples
        --------
        .. code-block:: python

           >>> import brainunit as u
           >>> import jax.numpy as jnp
           >>> from brainpy_state import hh_cond_exp_traub
           >>>
           >>> neurons = hh_cond_exp_traub(10)
           >>> neurons.init_state()
           >>>
           >>> # Get spike output for current state
           >>> spikes = neurons.get_spike()
           >>> print(spikes.shape)
           (10,)
           >>>
           >>> # Get spike output for custom voltage
           >>> V_custom = jnp.array([-60., -50., -40.]) * u.mV
           >>> neurons_3 = hh_cond_exp_traub(3)
           >>> neurons_3.init_state()
           >>> spikes_custom = neurons_3.get_spike(V_custom)

        See Also
        --------
        update : Main update method with discrete spike detection logic.
        """
        V = self.V.value if V is None else V
        # For HH neurons with Traub threshold: spike at V_T + 30.
        # Scale relative to 0 mV for the surrogate function.
        v_scaled = V / (1. * u.mV)
        return self.spk_fun(v_scaled)

    def _sum_signed_delta_inputs(self):
        r"""Split delta inputs into excitatory (positive) and inhibitory (negative).

        Processes all incoming synaptic spike events (delta inputs) and separates
        them by sign into excitatory and inhibitory conductance jumps. Positive
        weights increase excitatory conductance; negative weights increase
        inhibitory conductance (with sign flipped).

        Returns
        -------
        g_ex : ArrayLike
            Total excitatory conductance jump in nS, shape (\*in_size,) or
            (batch_size, \*in_size). Sum of all positive delta inputs.
        g_in : ArrayLike
            Total inhibitory conductance jump in nS, shape (\*in_size,) or
            (batch_size, \*in_size). Sum of absolute values of all negative
            delta inputs.

        Notes
        -----
        This method iterates over ``self.delta_inputs``, evaluating callable
        entries and removing them after use (transient spike events). For each
        input value:

        - If ``value > 0``: added to ``g_ex``
        - If ``value < 0``: absolute value added to ``g_in``

        The conductance units are assumed to match the model's conductance scale
        (nS). If no delta inputs are registered, returns zero arrays.

        This is an internal helper method called during the update step to collect
        all synaptic inputs before ODE integration.

        Examples
        --------
        .. code-block:: python

           >>> import brainunit as u
           >>> from brainpy_state import hh_cond_exp_traub
           >>>
           >>> neurons = hh_cond_exp_traub(10)
           >>> neurons.init_state()
           >>>
           >>> # Register excitatory and inhibitory inputs
           >>> neurons.add_delta_input('syn1', lambda: 5.0 * u.nS)
           >>> neurons.add_delta_input('syn2', lambda: -3.0 * u.nS)
           >>>
           >>> g_ex, g_in = neurons._sum_signed_delta_inputs()
           >>> print(g_ex, g_in)  # g_ex = 5.0 nS, g_in = 3.0 nS
           5.0 nS 3.0 nS

        See Also
        --------
        update : Main update method that calls this helper.
        add_delta_input : Method to register delta spike inputs.
        """
        g_ex = u.math.zeros_like(self.g_ex.value)
        g_in = u.math.zeros_like(self.g_in.value)
        if self.delta_inputs is None:
            return g_ex, g_in

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            zero = u.math.zeros_like(out)
            g_ex = g_ex + u.math.maximum(out, zero)
            # Inhibitory: negative weight -> positive conductance (sign flipped)
            g_in = g_in + u.math.maximum(-out, zero)
        return g_ex, g_in

    def update(self, x=0. * u.pA):
        r"""Update neuron state for one simulation step.

        Integrates the 6-dimensional ODE system for one time step using adaptive
        RK45 solver, processes incoming synaptic inputs, detects spikes based on
        threshold crossing and local maximum, and updates refractory state.

        The update follows the NEST ``hh_cond_exp_traub`` update order:

        1. Record pre-integration membrane potential (``V_old``).
        2. Integrate the full 6-dimensional ODE system over one time step
           using an adaptive RK45 solver (scipy's Dormand-Prince).
        3. Add arriving synaptic conductance jumps to ``g_ex`` / ``g_in``.
        4. Check spike condition: ``V_m >= V_T + 30 and V_old > V_m``
           (threshold + local maximum).
        5. Update refractory counter and record spike time.
        6. Store buffered stimulation current for the next step.

        Parameters
        ----------
        x : ArrayLike, default 0 pA
            External stimulation current input (in addition to ``I_e``), shape
            () or (\*in_size,) or (batch_size, \*in_size). This current is added
            to the constant ``I_e`` parameter and any registered current inputs
            via ``add_current_input()``.

        Returns
        -------
        ArrayLike
            Spike output with shape (batch_size, \*in_size) if batch_size is set,
            otherwise (\*in_size,). Values are computed using the surrogate spike
            function for differentiability. Spikes occur only when the discrete
            spike condition is satisfied (not refractory, threshold crossed, and
            local maximum detected).

        Notes
        -----
        **Integration Details:**

        Each neuron's state is integrated independently using scipy's
        ``solve_ivp`` with method='RK45' (Dormand-Prince). This matches NEST's
        GSL RKF45 solver for numerical equivalence. The ODE system is:

        .. math::

           \frac{d}{dt}\begin{bmatrix} V_m \\ m \\ h \\ n \\ g_{ex} \\ g_{in} \end{bmatrix}
           = \begin{bmatrix}
               (-I_{Na} - I_K - I_L - I_{syn,ex} - I_{syn,in} + I_{stim} + I_e) / C_m \\
               \alpha_m - (\alpha_m + \beta_m) m \\
               \alpha_h - (\alpha_h + \beta_h) h \\
               \alpha_n - (\alpha_n + \beta_n) n \\
               -g_{ex} / \tau_{syn,ex} \\
               -g_{in} / \tau_{syn,in}
           \end{bmatrix}

        **Spike Detection Logic:**

        A spike is detected when all three conditions are met:

        1. ``refractory_step_count == 0`` (not in refractory period)
        2. ``V_m >= V_T + 30`` (threshold crossing)
        3. ``V_old > V_m`` (local maximum - voltage falling)

        No voltage reset occurs; repolarization is handled by intrinsic currents.

        **Synaptic Input Processing:**

        Delta inputs (spike events) are collected and split by sign:

        - Positive weights → excitatory conductance (``g_ex += w``)
        - Negative weights → inhibitory conductance (``g_in += |w|``)

        Conductance jumps are applied **after** ODE integration, matching NEST's
        update sequence.

        **Computational Complexity**


        The method performs per-neuron integration, so runtime scales linearly
        with the total number of neurons (``prod(in_size)``). For large populations,
        consider vectorized implementations if numerical accuracy can be relaxed.

        **Failure Modes**


        - If ODE solver fails to converge (stiff dynamics), ``solve_ivp`` may
          return partial integration. Check ``sol.success`` if debugging.
        - Extreme parameter values (very large conductances, very small time
          constants) may cause numerical instability.
        - Division by zero in rate equations at special voltages (e.g., V = 15 mV
          for alpha_n) is mitigated by L'Hospital's rule in the analytical form,
          but numerical issues may still arise near singularities.

        See Also
        --------
        init_state : Initialize state variables.
        get_spike : Compute surrogate spike output.
        _sum_signed_delta_inputs : Process synaptic spike inputs.
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
        V_T = self._broadcast_to_state(self._to_numpy(self.V_T, u.mV), v_shape)
        E_ex = self._broadcast_to_state(self._to_numpy(self.E_ex, u.mV), v_shape)
        E_in = self._broadcast_to_state(self._to_numpy(self.E_in, u.mV), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)

        # Current state
        V_m = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        m_val = self._broadcast_to_state(np.asarray(self.m.value, dtype=np.float64), v_shape)
        h_val = self._broadcast_to_state(np.asarray(self.h.value, dtype=np.float64), v_shape)
        n_val = self._broadcast_to_state(np.asarray(self.n.value, dtype=np.float64), v_shape)
        g_ex_val = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape)
        g_in_val = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape)
        I_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )

        # Collect spike/current inputs
        dg_ex_q, dg_in_q = self._sum_signed_delta_inputs()
        dg_ex = self._broadcast_to_state(self._to_numpy(dg_ex_q, u.nS), v_shape)
        dg_in = self._broadcast_to_state(self._to_numpy(dg_in_q, u.nS), v_shape)
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
        g_ex_new = np.empty(flat_size, dtype=np.float64)
        g_in_new = np.empty(flat_size, dtype=np.float64)

        V_m_flat = V_m.ravel()
        m_flat = m_val.ravel()
        h_flat = h_val.ravel()
        n_flat = n_val.ravel()
        g_ex_flat = g_ex_val.ravel()
        g_in_flat = g_in_val.ravel()
        I_stim_flat = I_stim.ravel()
        g_Na_flat = g_Na.ravel()
        g_K_flat = g_K.ravel()
        g_L_flat = g_L.ravel()
        E_Na_flat = E_Na.ravel()
        E_K_flat = E_K.ravel()
        E_L_flat = E_L.ravel()
        V_T_flat = V_T.ravel()
        E_ex_flat = E_ex.ravel()
        E_in_flat = E_in.ravel()
        C_m_flat = C_m.ravel()
        I_e_flat = I_e.ravel()
        tau_ex_flat = tau_ex.ravel()
        tau_in_flat = tau_in.ravel()

        for i in range(flat_size):
            y0 = np.array([
                V_m_flat[i], m_flat[i], h_flat[i], n_flat[i],
                g_ex_flat[i], g_in_flat[i]
            ])

            # Capture per-neuron parameters for closure
            _g_Na = g_Na_flat[i]
            _g_K = g_K_flat[i]
            _g_L = g_L_flat[i]
            _E_Na = E_Na_flat[i]
            _E_K = E_K_flat[i]
            _E_L = E_L_flat[i]
            _V_T = V_T_flat[i]
            _E_ex = E_ex_flat[i]
            _E_in = E_in_flat[i]
            _C_m = C_m_flat[i]
            _I_e = I_e_flat[i]
            _I_stim = I_stim_flat[i]
            _tau_ex = tau_ex_flat[i]
            _tau_in = tau_in_flat[i]

            def rhs(t_local, y,
                    _g_Na=_g_Na, _g_K=_g_K, _g_L=_g_L,
                    _E_Na=_E_Na, _E_K=_E_K, _E_L=_E_L,
                    _V_T=_V_T, _E_ex=_E_ex, _E_in=_E_in,
                    _C_m=_C_m, _I_e=_I_e, _I_stim=_I_stim,
                    _tau_ex=_tau_ex, _tau_in=_tau_in):
                V_m_ = y[0]
                m_ = y[1]
                h_ = y[2]
                n_ = y[3]
                g_e = y[4]
                g_i = y[5]

                # Ionic currents
                I_Na = _g_Na * m_ * m_ * m_ * h_ * (V_m_ - _E_Na)
                I_K = _g_K * n_ * n_ * n_ * n_ * (V_m_ - _E_K)
                I_L = _g_L * (V_m_ - _E_L)

                # Synaptic currents (conductance-based)
                I_syn_exc = g_e * (V_m_ - _E_ex)
                I_syn_inh = g_i * (V_m_ - _E_in)

                # Shifted voltage for gating variable rate equations
                V = V_m_ - _V_T

                alpha_n = 0.032 * (15.0 - V) / (math.exp((15.0 - V) / 5.0) - 1.0)
                beta_n = 0.5 * math.exp((10.0 - V) / 40.0)
                alpha_m = 0.32 * (13.0 - V) / (math.exp((13.0 - V) / 4.0) - 1.0)
                beta_m = 0.28 * (V - 40.0) / (math.exp((V - 40.0) / 5.0) - 1.0)
                alpha_h = 0.128 * math.exp((17.0 - V) / 18.0)
                beta_h = 4.0 / (1.0 + math.exp((40.0 - V) / 5.0))

                f = np.empty(6)
                f[0] = (-I_Na - I_K - I_L - I_syn_exc - I_syn_inh + _I_stim + _I_e) / _C_m
                f[1] = alpha_m - (alpha_m + beta_m) * m_
                f[2] = alpha_h - (alpha_h + beta_h) * h_
                f[3] = alpha_n - (alpha_n + beta_n) * n_
                f[4] = -g_e / _tau_ex
                f[5] = -g_i / _tau_in
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
            g_ex_new[i] = yf[4]
            g_in_new[i] = yf[5]

        V_m = V_new.reshape(v_shape)
        m_val = m_new.reshape(v_shape)
        h_val = h_new.reshape(v_shape)
        n_val = n_new.reshape(v_shape)
        g_ex_val = g_ex_new.reshape(v_shape)
        g_in_val = g_in_new.reshape(v_shape)

        # Add arriving spike conductance inputs (after ODE integration, matching NEST)
        g_ex_val = g_ex_val + dg_ex
        g_in_val = g_in_val + dg_in

        # Spike detection: threshold crossing + local maximum
        not_refractory = r == 0
        V_T_arr = self._broadcast_to_state(self._to_numpy(self.V_T, u.mV), v_shape)
        crossed_threshold = V_m >= (V_T_arr + 30.0)
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
        self.g_ex.value = g_ex_val * u.nS
        self.g_in.value = g_in_val * u.nS
        self.I_stim.value = I_stim_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r_new, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        # Return spike output: only signal a spike when spike_cond is True
        V_out = np.where(spike_cond, 1e-12, -1.0)
        return self.get_spike(V_out * u.mV)
