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

"""NEST-compatible ``ht_neuron`` model (Hill & Tononi, 2005).

This module implements the neuron model described in:

    Hill S, Tononi G (2005). Modeling sleep and wakefulness in the
    thalamocortical system. Journal of Neurophysiology, 93:1671-1698.
    DOI: https://doi.org/10.1152/jn.00915.2004

The implementation follows the NEST ``models/ht_neuron.{h,cpp}`` source
exactly, including:

- Integrate-and-fire with adaptive (dynamic) threshold.
- Repolarizing potassium current instead of hard reset.
- AMPA, NMDA, GABA_A, and GABA_B conductance-based synapses with
  beta-function (difference of exponentials) time course.
- Voltage-dependent NMDA with instantaneous or two-stage unblocking.
- Intrinsic currents I_h, I_T, I_Na(p), and I_KNa.
- GSL RKF45 adaptive ODE integration (mapped to scipy RK45).
"""

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
    'ht_neuron',
]


# ---------------------------------------------------------------------------
# Equilibrium / steady-state helper functions (module-level, pure NumPy)
# ---------------------------------------------------------------------------

def _m_eq_h(V):
    """Equilibrium activation for I_h (hyperpolarization-activated current).

    Parameters
    ----------
    V : float
        Membrane potential in mV.

    Returns
    -------
    float
        Equilibrium value of m_Ih at voltage *V*.
    """
    I_h_Vthreshold = -75.0
    return 1.0 / (1.0 + math.exp((V - I_h_Vthreshold) / 5.5))


def _h_eq_T(V):
    """Equilibrium inactivation for I_T (low-threshold Ca current).

    Parameters
    ----------
    V : float
        Membrane potential in mV.

    Returns
    -------
    float
        Equilibrium value of h_IT at voltage *V*.
    """
    return 1.0 / (1.0 + math.exp((V + 83.0) / 4.0))


def _m_eq_T(V):
    """Equilibrium activation for I_T (low-threshold Ca current).

    Parameters
    ----------
    V : float
        Membrane potential in mV.

    Returns
    -------
    float
        Equilibrium value of m_IT at voltage *V*.
    """
    return 1.0 / (1.0 + math.exp(-(V + 59.0) / 6.2))


def _D_eq_KNa(V, tau_D_KNa):
    """Steady-state D value for I_KNa (depolarization-activated K current).

    Parameters
    ----------
    V : float
        Membrane potential in mV.
    tau_D_KNa : float
        Relaxation time constant in ms.

    Returns
    -------
    float
        Equilibrium value of D_IKNa at voltage *V*.
    """
    D_influx_peak = 0.025
    D_thresh = -10.0
    D_slope = 5.0
    D_eq = 0.001
    D_influx = D_influx_peak / (1.0 + math.exp(-(V - D_thresh) / D_slope))
    return tau_D_KNa * D_influx + D_eq


def _m_eq_NMDA(V, S_act_NMDA, V_act_NMDA):
    """Steady-state magnesium unblock ratio for NMDA channels.

    Parameters
    ----------
    V : float
        Membrane potential in mV.
    S_act_NMDA : float
        Slope parameter for NMDA unblocking sigmoid.
    V_act_NMDA : float
        Voltage at inflection point of NMDA unblocking sigmoid in mV.

    Returns
    -------
    float
        Equilibrium Mg-unblock fraction at voltage *V*.
    """
    return 1.0 / (1.0 + math.exp(-S_act_NMDA * (V - V_act_NMDA)))


def _m_NMDA(V, m_eq, m_fast, m_slow, instant_unblock_NMDA):
    """Effective NMDA activation combining fast and slow unblocking.

    Parameters
    ----------
    V : float
        Membrane potential in mV.
    m_eq : float
        Equilibrium Mg-unblock fraction.
    m_fast : float
        Fast unblocking variable.
    m_slow : float
        Slow unblocking variable.
    instant_unblock_NMDA : bool
        If True, use instantaneous unblocking (return m_eq directly).

    Returns
    -------
    float
        Effective NMDA channel activation.
    """
    if instant_unblock_NMDA:
        return m_eq
    A1 = 0.51 - 0.0028 * V
    A2 = 1.0 - A1
    return A1 * m_fast + A2 * m_slow


def _beta_normalization_factor(tau_rise, tau_decay):
    """Compute the normalization constant for the beta (difference-of-exponentials) synapse.

    This matches NEST's ``beta_normalization_factor()`` from
    ``libnestutil/beta_normalization_factor.h``.  The factor ensures that a
    unit spike produces a peak conductance of exactly ``g_peak``.

    Parameters
    ----------
    tau_rise : float
        Synaptic rise time constant in ms.
    tau_decay : float
        Synaptic decay time constant in ms.

    Returns
    -------
    float
        Normalization constant.
    """
    eps = np.finfo(np.float64).eps
    tau_difference = tau_decay - tau_rise
    peak_value = 0.0
    if abs(tau_difference) > eps:
        t_peak = tau_decay * tau_rise * math.log(tau_decay / tau_rise) / tau_difference
        peak_value = math.exp(-t_peak / tau_decay) - math.exp(-t_peak / tau_rise)
    if abs(peak_value) < eps:
        # alpha-function limit
        return math.e / tau_decay
    else:
        return (1.0 / tau_rise - 1.0 / tau_decay) / peak_value


# ---------------------------------------------------------------------------
# State vector indices (matching NEST enum StateVecElems_)
# ---------------------------------------------------------------------------
_V_M = 0
_THETA = 1
_DG_AMPA = 2
_G_AMPA = 3
_DG_NMDA_TIMECOURSE = 4
_G_NMDA_TIMECOURSE = 5
_DG_GABA_A = 6
_G_GABA_A = 7
_DG_GABA_B = 8
_G_GABA_B = 9
_m_fast_NMDA = 10
_m_slow_NMDA = 11
_m_Ih = 12
_D_IKNa = 13
_m_IT = 14
_h_IT = 15
_STATE_VEC_SIZE = 16


class ht_neuron(Neuron):
    r"""NEST-compatible ``ht_neuron`` model (Hill & Tononi, 2005).

    Short description
    -----------------

    Neuron model after Hill & Tononi (2005) with adaptive threshold,
    repolarizing potassium current, four conductance-based synapse types
    (AMPA, NMDA, GABA_A, GABA_B), and four intrinsic currents
    (I_NaP, I_KNa, I_T, I_h).

    Description
    -----------

    This model neuron implements a slightly modified version of the
    neuron model described in [1]_. The most important properties are:

    - Integrate-and-fire with adaptive threshold.
    - Repolarizing potassium current instead of hard reset.
    - AMPA, NMDA, GABA_A, and GABA_B conductance-based synapses with
      beta-function (difference of exponentials) time course.
    - Voltage-dependent NMDA with instantaneous or two-stage unblocking
      [1]_, [2]_.
    - Intrinsic currents I_h, I_T, I_Na(p), and I_KNa.

    Membrane dynamics
    .................

    The membrane potential evolves as:

    .. math::

       \frac{dV_m}{dt} = \frac{I_{Na} + I_K + I_{syn} + I_{NaP}
         + I_{KNa} + I_T + I_h + I_{stim}}{\tau_m} + I_{spike}

    where the leak currents are:

    .. math::

       I_{Na} &= -g_{NaL}\,(V_m - E_{Na}) \\
       I_K    &= -g_{KL}\,(V_m - E_K)

    and the post-spike potassium current (only during refractory period) is:

    .. math::

       I_{spike} = \begin{cases}
         -(V_m - E_K) / \tau_{spike} & \text{if refractory} \\
         0 & \text{otherwise}
       \end{cases}

    Dynamic threshold
    .................

    .. math::

       \frac{d\theta}{dt} = -\frac{\theta - \theta_{eq}}{\tau_\theta}

    Synaptic currents
    .................

    Each synapse type (AMPA, NMDA, GABA_A, GABA_B) uses a beta-function
    conductance kernel:

    .. math::

       \frac{dg'}{dt} &= -g' / \tau_{rise} \\
       \frac{dg}{dt}  &= g' - g / \tau_{decay}

    The total synaptic current is:

    .. math::

       I_{syn} = -g_{AMPA}(V - E_{AMPA})
                 -g_{NMDA} \cdot m_{NMDA}(V)(V - E_{NMDA})
                 -g_{GABA\_A}(V - E_{GABA\_A})
                 -g_{GABA\_B}(V - E_{GABA\_B})

    NMDA voltage dependence uses a two-stage Mg²⁺ unblocking model
    (or instantaneous, controlled by ``instant_unblock_NMDA``).

    Intrinsic currents
    ..................

    - **I_NaP** (persistent sodium):
      :math:`I_{NaP} = -g_{NaP} \cdot m_\infty^{N_{NaP}} \cdot (V - E_{NaP})`
      where :math:`m_\infty = 1 / (1 + \exp(-(V + 55.7) / 7.7))`

    - **I_KNa** (depolarization-activated K):
      :math:`I_{KNa} = -g_{KNa} \cdot m_\infty \cdot (V - E_{KNa})`
      where :math:`m_\infty = 1 / (1 + (0.25 / D)^{3.5})`

    - **I_T** (low-threshold Ca):
      :math:`I_T = -g_T \cdot m^{N_T} \cdot h \cdot (V - E_T)`

    - **I_h** (hyperpolarization-activated):
      :math:`I_h = -g_h \cdot m \cdot (V - E_h)`

    Spike detection and reset
    .........................

    A spike is generated when the neuron is not refractory **and**
    :math:`V_m \ge \theta`. On spike:

    - :math:`V_m` and :math:`\theta` are set to :math:`E_{Na}`.
    - The refractory counter is activated for ``t_ref`` ms, during which
      the post-spike potassium current drives repolarization.

    Numerical integration
    .....................

    NEST uses GSL RKF45 (Runge-Kutta-Fehlberg 4/5) with adaptive step-size
    control (relative tolerance 1e-3, absolute tolerance 0). This
    implementation uses ``scipy.integrate.solve_ivp`` with method ``'RK45'``
    (Dormand-Prince) at matching tolerances.

    Parameters
    ----------

    ========================= ============ ====== ==============================================
    **Parameter**             **Default**  **Unit** **Description**
    ========================= ============ ====== ==============================================
    ``in_size``               (required)          Population shape
    ``E_Na``                  30.0         mV     Sodium reversal potential
    ``E_K``                   -90.0        mV     Potassium reversal potential
    ``g_NaL``                 0.2                 Sodium leak conductance (unitless)
    ``g_KL``                  1.0                 Potassium leak conductance (unitless)
    ``tau_m``                 16.0         ms     Membrane time constant
    ``theta_eq``              -51.0        mV     Equilibrium threshold value
    ``tau_theta``             2.0          ms     Threshold time constant
    ``tau_spike``             1.75         ms     Membrane time constant for post-spike K current
    ``t_ref``                 2.0          ms     Refractory time / duration of K current
    ``g_peak_AMPA``           0.1                 Peak AMPA conductance (unitless)
    ``tau_rise_AMPA``         0.5          ms     AMPA rise time constant
    ``tau_decay_AMPA``        2.4          ms     AMPA decay time constant
    ``E_rev_AMPA``            0.0          mV     AMPA reversal potential
    ``g_peak_NMDA``           0.075                Peak NMDA conductance (unitless)
    ``tau_rise_NMDA``         4.0          ms     NMDA rise time constant
    ``tau_decay_NMDA``        40.0         ms     NMDA decay time constant
    ``E_rev_NMDA``            0.0          mV     NMDA reversal potential
    ``V_act_NMDA``            -25.57       mV     NMDA unblocking sigmoid inflection
    ``S_act_NMDA``            0.081        1/mV   NMDA unblocking sigmoid slope
    ``tau_Mg_slow_NMDA``      22.7         ms     Slow Mg²⁺ unblocking time constant
    ``tau_Mg_fast_NMDA``      0.68         ms     Fast Mg²⁺ unblocking time constant
    ``instant_unblock_NMDA``  False                Use instantaneous NMDA unblocking
    ``g_peak_GABA_A``         0.33                 Peak GABA_A conductance (unitless)
    ``tau_rise_GABA_A``       1.0          ms     GABA_A rise time constant
    ``tau_decay_GABA_A``      7.0          ms     GABA_A decay time constant
    ``E_rev_GABA_A``          -70.0        mV     GABA_A reversal potential
    ``g_peak_GABA_B``         0.0132               Peak GABA_B conductance (unitless)
    ``tau_rise_GABA_B``       60.0         ms     GABA_B rise time constant
    ``tau_decay_GABA_B``      200.0        ms     GABA_B decay time constant
    ``E_rev_GABA_B``          -90.0        mV     GABA_B reversal potential
    ``g_peak_NaP``            1.0                  Peak persistent Na conductance (unitless)
    ``E_rev_NaP``             30.0         mV     I_NaP reversal potential
    ``N_NaP``                 3.0                  I_NaP activation exponent
    ``g_peak_KNa``            1.0                  Peak I_KNa conductance (unitless)
    ``E_rev_KNa``             -90.0        mV     I_KNa reversal potential
    ``tau_D_KNa``             1250.0       ms     I_KNa relaxation time constant
    ``g_peak_T``              1.0                  Peak I_T conductance (unitless)
    ``E_rev_T``               0.0          mV     I_T reversal potential
    ``N_T``                   2.0                  I_T activation exponent
    ``g_peak_h``              1.0                  Peak I_h conductance (unitless)
    ``E_rev_h``               -40.0        mV     I_h reversal potential
    ``voltage_clamp``         False                Clamp voltage at initial value (testing)
    ``rtol``                  1e-3                 ODE solver relative tolerance
    ``atol``                  1e-9                 ODE solver absolute tolerance
    ========================= ============ ====== ==============================================

    .. note::

       Conductances are **unitless** in this model. All currents are
       expressed in mV (i.e. divided by membrane capacitance already).

    State variables
    ---------------

    - ``V``:  membrane potential (:math:`V_m`, mV)
    - ``theta``:  dynamic threshold (mV)
    - ``DG_AMPA``, ``G_AMPA``:  AMPA beta-function state variables
    - ``DG_NMDA``, ``G_NMDA``:  NMDA beta-function state variables
    - ``DG_GABA_A``, ``G_GABA_A``:  GABA_A beta-function state variables
    - ``DG_GABA_B``, ``G_GABA_B``:  GABA_B beta-function state variables
    - ``m_fast_NMDA``, ``m_slow_NMDA``:  NMDA Mg²⁺ unblocking variables
    - ``m_Ih``:  I_h activation
    - ``D_IKNa``:  I_KNa activation variable
    - ``m_IT``:  I_T activation
    - ``h_IT``:  I_T inactivation
    - ``I_NaP``, ``I_KNa``, ``I_T``, ``I_h``:  intrinsic current values
    - ``ref_steps``:  refractory counter (integer steps)

    References
    ----------

    .. [1] Hill S, Tononi G (2005). Modeling sleep and wakefulness in the
           thalamocortical system. Journal of Neurophysiology, 93:1671-1698.
           DOI: https://doi.org/10.1152/jn.00915.2004
    .. [2] Vargas-Caballero M, Robinson HPC (2003). A slow fraction of Mg2+
           unblock of NMDA receptors limits their contribution to spike
           generation in cortical pyramidal neurons. Journal of
           Neurophysiology, 89:2778-2783.
           DOI: https://doi.org/10.1152/jn.01038.2002

    See also
    --------
    hh_psc_alpha : Hodgkin-Huxley model with alpha-shaped PSCs.
    """

    __module__ = 'brainpy.state'

    # Synapse receptor type constants (matching NEST enum)
    AMPA = 1
    NMDA = 2
    GABA_A = 3
    GABA_B = 4

    def __init__(
        self,
        in_size: Size,
        # Leak / reversal
        E_Na: float = 30.0,
        E_K: float = -90.0,
        g_NaL: float = 0.2,
        g_KL: float = 1.0,
        tau_m: float = 16.0,
        # Dynamic threshold
        theta_eq: float = -51.0,
        tau_theta: float = 2.0,
        # Post-spike potassium current
        tau_spike: float = 1.75,
        t_ref: float = 2.0,
        # AMPA synapse
        g_peak_AMPA: float = 0.1,
        tau_rise_AMPA: float = 0.5,
        tau_decay_AMPA: float = 2.4,
        E_rev_AMPA: float = 0.0,
        # NMDA synapse
        g_peak_NMDA: float = 0.075,
        tau_rise_NMDA: float = 4.0,
        tau_decay_NMDA: float = 40.0,
        E_rev_NMDA: float = 0.0,
        V_act_NMDA: float = -25.57,
        S_act_NMDA: float = 0.081,
        tau_Mg_slow_NMDA: float = 22.7,
        tau_Mg_fast_NMDA: float = 0.68,
        instant_unblock_NMDA: bool = False,
        # GABA_A synapse
        g_peak_GABA_A: float = 0.33,
        tau_rise_GABA_A: float = 1.0,
        tau_decay_GABA_A: float = 7.0,
        E_rev_GABA_A: float = -70.0,
        # GABA_B synapse
        g_peak_GABA_B: float = 0.0132,
        tau_rise_GABA_B: float = 60.0,
        tau_decay_GABA_B: float = 200.0,
        E_rev_GABA_B: float = -90.0,
        # Intrinsic: I_NaP
        g_peak_NaP: float = 1.0,
        E_rev_NaP: float = 30.0,
        N_NaP: float = 3.0,
        # Intrinsic: I_KNa
        g_peak_KNa: float = 1.0,
        E_rev_KNa: float = -90.0,
        tau_D_KNa: float = 1250.0,
        # Intrinsic: I_T
        g_peak_T: float = 1.0,
        E_rev_T: float = 0.0,
        N_T: float = 2.0,
        # Intrinsic: I_h
        g_peak_h: float = 1.0,
        E_rev_h: float = -40.0,
        # Testing
        voltage_clamp: bool = False,
        # Solver
        rtol: float = 1e-3,
        atol: float = 1e-9,
        # Base class
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Store all parameters as plain floats (unitless model, NEST convention)
        self.E_Na = E_Na
        self.E_K = E_K
        self.g_NaL = g_NaL
        self.g_KL = g_KL
        self.tau_m = tau_m
        self.theta_eq = theta_eq
        self.tau_theta = tau_theta
        self.tau_spike = tau_spike
        self.t_ref = t_ref

        self.g_peak_AMPA = g_peak_AMPA
        self.tau_rise_AMPA = tau_rise_AMPA
        self.tau_decay_AMPA = tau_decay_AMPA
        self.E_rev_AMPA = E_rev_AMPA

        self.g_peak_NMDA = g_peak_NMDA
        self.tau_rise_NMDA = tau_rise_NMDA
        self.tau_decay_NMDA = tau_decay_NMDA
        self.E_rev_NMDA = E_rev_NMDA
        self.V_act_NMDA = V_act_NMDA
        self.S_act_NMDA = S_act_NMDA
        self.tau_Mg_slow_NMDA = tau_Mg_slow_NMDA
        self.tau_Mg_fast_NMDA = tau_Mg_fast_NMDA
        self.instant_unblock_NMDA = instant_unblock_NMDA

        self.g_peak_GABA_A = g_peak_GABA_A
        self.tau_rise_GABA_A = tau_rise_GABA_A
        self.tau_decay_GABA_A = tau_decay_GABA_A
        self.E_rev_GABA_A = E_rev_GABA_A

        self.g_peak_GABA_B = g_peak_GABA_B
        self.tau_rise_GABA_B = tau_rise_GABA_B
        self.tau_decay_GABA_B = tau_decay_GABA_B
        self.E_rev_GABA_B = E_rev_GABA_B

        self.g_peak_NaP = g_peak_NaP
        self.E_rev_NaP = E_rev_NaP
        self.N_NaP = N_NaP

        self.g_peak_KNa = g_peak_KNa
        self.E_rev_KNa = E_rev_KNa
        self.tau_D_KNa = tau_D_KNa

        self.g_peak_T = g_peak_T
        self.E_rev_T = E_rev_T
        self.N_T = N_T

        self.g_peak_h = g_peak_h
        self.E_rev_h = E_rev_h

        self.voltage_clamp = voltage_clamp
        self.rtol = rtol
        self.atol = atol

        self._validate_parameters()

        # Pre-compute synaptic conductance step sizes
        self._cond_step_AMPA = g_peak_AMPA * _beta_normalization_factor(tau_rise_AMPA, tau_decay_AMPA)
        self._cond_step_NMDA = g_peak_NMDA * _beta_normalization_factor(tau_rise_NMDA, tau_decay_NMDA)
        self._cond_step_GABA_A = g_peak_GABA_A * _beta_normalization_factor(tau_rise_GABA_A, tau_decay_GABA_A)
        self._cond_step_GABA_B = g_peak_GABA_B * _beta_normalization_factor(tau_rise_GABA_B, tau_decay_GABA_B)

    def _validate_parameters(self):
        """Validate parameter constraints matching NEST's ``Parameters_::set()``."""
        # Non-negative peak conductances
        for name in ('g_peak_AMPA', 'g_peak_NMDA', 'g_peak_GABA_A', 'g_peak_GABA_B',
                     'g_peak_NaP', 'g_peak_KNa', 'g_peak_T', 'g_peak_h',
                     'g_NaL', 'g_KL'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} >= 0 required.')

        if self.S_act_NMDA < 0:
            raise ValueError('S_act_NMDA >= 0 required.')
        if self.t_ref < 0:
            raise ValueError('t_ref >= 0 required.')

        # Strictly positive time constants
        for name in ('tau_rise_AMPA', 'tau_decay_AMPA',
                     'tau_rise_NMDA', 'tau_decay_NMDA',
                     'tau_rise_GABA_A', 'tau_decay_GABA_A',
                     'tau_rise_GABA_B', 'tau_decay_GABA_B',
                     'tau_Mg_fast_NMDA', 'tau_Mg_slow_NMDA',
                     'tau_spike', 'tau_theta', 'tau_m', 'tau_D_KNa'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} > 0 required.')

        # Rise < decay constraints
        if self.tau_rise_AMPA >= self.tau_decay_AMPA:
            raise ValueError('tau_rise_AMPA < tau_decay_AMPA required.')
        if self.tau_rise_GABA_A >= self.tau_decay_GABA_A:
            raise ValueError('tau_rise_GABA_A < tau_decay_GABA_A required.')
        if self.tau_rise_GABA_B >= self.tau_decay_GABA_B:
            raise ValueError('tau_rise_GABA_B < tau_decay_GABA_B required.')
        if self.tau_rise_NMDA >= self.tau_decay_NMDA:
            raise ValueError('tau_rise_NMDA < tau_decay_NMDA required.')
        if self.tau_Mg_fast_NMDA >= self.tau_Mg_slow_NMDA:
            raise ValueError('tau_Mg_fast_NMDA < tau_Mg_slow_NMDA required.')

    def _refractory_counts(self, dt_ms):
        """Convert refractory time in ms to integer step count.

        Parameters
        ----------
        dt_ms : float
            Time step in ms.

        Returns
        -------
        int
            Number of refractory steps.
        """
        return int(round(self.t_ref / dt_ms))

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize all state variables.

        The membrane potential is set to the leak equilibrium:

        .. math::

           V_m = \\frac{g_{NaL} \\cdot E_{Na} + g_{KL} \\cdot E_K}{g_{NaL} + g_{KL}}

        All intrinsic gating variables are set to their equilibrium values
        at this initial voltage. Synaptic variables are zero.
        """
        # Compute initial membrane potential (leak equilibrium)
        V_init = (self.g_NaL * self.E_Na + self.g_KL * self.E_K) / (self.g_NaL + self.g_KL)

        # Build initial state vector
        y0 = np.zeros(_STATE_VEC_SIZE, dtype=np.float64)
        y0[_V_M] = V_init
        y0[_THETA] = self.theta_eq
        # Synaptic variables: all zero (indices 2-9)
        # Intrinsic gating at equilibrium
        y0[_m_fast_NMDA] = _m_eq_NMDA(V_init, self.S_act_NMDA, self.V_act_NMDA)
        y0[_m_slow_NMDA] = _m_eq_NMDA(V_init, self.S_act_NMDA, self.V_act_NMDA)
        y0[_m_Ih] = _m_eq_h(V_init)
        y0[_D_IKNa] = _D_eq_KNa(V_init, self.tau_D_KNa)
        y0[_m_IT] = _m_eq_T(V_init)
        y0[_h_IT] = _h_eq_T(V_init)

        # Allocate state vectors (shape = varshape, possibly batched)
        zeros_like = braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7), self.varshape, batch_size)

        # ODE state: 16-element vector per neuron, stored as HiddenState
        # We store each component separately for clarity
        self.V = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(y0[_V_M]), self.varshape, batch_size)
        )
        self.theta = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(y0[_THETA]), self.varshape, batch_size)
        )
        self.DG_AMPA = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.G_AMPA = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.DG_NMDA = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.G_NMDA = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.DG_GABA_A = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.G_GABA_A = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.DG_GABA_B = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.G_GABA_B = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.m_fast_NMDA_state = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(y0[_m_fast_NMDA]), self.varshape, batch_size)
        )
        self.m_slow_NMDA_state = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(y0[_m_slow_NMDA]), self.varshape, batch_size)
        )
        self.m_Ih_state = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(y0[_m_Ih]), self.varshape, batch_size)
        )
        self.D_IKNa_state = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(y0[_D_IKNa]), self.varshape, batch_size)
        )
        self.m_IT_state = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(y0[_m_IT]), self.varshape, batch_size)
        )
        self.h_IT_state = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(y0[_h_IT]), self.varshape, batch_size)
        )

        # Intrinsic current values (for recording)
        self.I_NaP_val = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.I_KNa_val = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.I_T_val = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )
        self.I_h_val = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.0), self.varshape, batch_size)
        )

        # Refractory counter
        self.ref_steps = brainstate.ShortTermState(
            np.zeros(self.varshape if batch_size is None else (batch_size, *self.varshape), dtype=np.int32)
        )

        # Stimulation current buffer
        self.I_stim = brainstate.ShortTermState(
            np.zeros(self.varshape if batch_size is None else (batch_size, *self.varshape), dtype=np.float64)
        )

        # Spike time tracking
        self.last_spike_time = brainstate.ShortTermState(spk_time * u.ms)

        # Voltage clamp value
        self._V_clamp = V_init

    def get_spike(self, V: ArrayLike = None):
        """Generate spike output using surrogate gradient.

        For ht_neuron, a spike occurs when V_m >= theta. We scale the
        voltage relative to the threshold for the surrogate function.
        """
        V = np.asarray(self.V.value, dtype=np.float64) if V is None else V
        theta = np.asarray(self.theta.value, dtype=np.float64)
        # Scale: positive when V >= theta
        v_scaled = (V - theta) / max(abs(self.theta_eq), 1.0)
        return self.spk_fun(jnp.asarray(v_scaled))

    def update(self, x=0.0):
        r"""Update neuron state for one simulation step.

        The update follows the NEST ``ht_neuron::update()`` ordering:

        1. Integrate the 16-dimensional ODE system over one time step
           using an adaptive RK45 solver, applying voltage clamp and
           instantaneous NMDA blocking constraints at each sub-step.
        2. Check spike condition (not refractory **and** V >= theta) and
           apply reset if spiking.
        3. Decrement refractory counter.
        4. Add arriving synaptic spike inputs to DG variables.
        5. Store stimulation current for the next step.

        Parameters
        ----------
        x : float or array, default 0.0
            External stimulation current (in mV/ms, since conductances are
            unitless in this model — effectively I/C_m). This is added to
            ``I_stim`` which enters the ODE as an additive term to dV/dt.

        Returns
        -------
        ArrayLike
            Spike output (surrogate gradient compatible).
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape
        flat_size = int(np.prod(v_shape)) if len(v_shape) > 0 else 1

        # Collect synaptic spike inputs (via delta inputs)
        # Spikes arrive as weighted conductance changes to specific receptor types.
        # In this model, we expect delta inputs formatted as a dict with keys
        # 'AMPA', 'NMDA', 'GABA_A', 'GABA_B', or a single aggregated value.
        spk_ampa = np.zeros(v_shape, dtype=np.float64)
        spk_nmda = np.zeros(v_shape, dtype=np.float64)
        spk_gaba_a = np.zeros(v_shape, dtype=np.float64)
        spk_gaba_b = np.zeros(v_shape, dtype=np.float64)

        # Handle labeled delta inputs for each receptor type
        for label, target in [('AMPA', 'ampa'), ('NMDA', 'nmda'),
                              ('GABA_A', 'gaba_a'), ('GABA_B', 'gaba_b')]:
            val = self.sum_delta_inputs(0.0, label=label)
            if isinstance(val, (int, float)):
                if val != 0.0:
                    arr = np.broadcast_to(np.float64(val), v_shape).copy()
                    if target == 'ampa':
                        spk_ampa = arr
                    elif target == 'nmda':
                        spk_nmda = arr
                    elif target == 'gaba_a':
                        spk_gaba_a = arr
                    elif target == 'gaba_b':
                        spk_gaba_b = arr
            else:
                arr = np.broadcast_to(np.asarray(val, dtype=np.float64), v_shape).copy()
                if target == 'ampa':
                    spk_ampa = arr
                elif target == 'nmda':
                    spk_nmda = arr
                elif target == 'gaba_a':
                    spk_gaba_a = arr
                elif target == 'gaba_b':
                    spk_gaba_b = arr

        # Also collect unlabeled delta inputs (generic spikes go to AMPA by default)
        unlabeled = self.sum_delta_inputs(0.0)
        if not isinstance(unlabeled, (int, float)) or unlabeled != 0.0:
            spk_ampa = spk_ampa + np.broadcast_to(
                np.asarray(unlabeled, dtype=np.float64) if not isinstance(unlabeled, (int, float))
                else np.full(v_shape, unlabeled, dtype=np.float64),
                v_shape
            )

        # Collect stimulation current input
        I_stim_next = float(x) if isinstance(x, (int, float)) else np.asarray(x, dtype=np.float64)
        I_stim_next = np.broadcast_to(
            np.asarray(I_stim_next, dtype=np.float64), v_shape
        ).copy()

        # Extract current state as flat numpy arrays
        V_m = np.asarray(self.V.value, dtype=np.float64).ravel()
        theta_val = np.asarray(self.theta.value, dtype=np.float64).ravel()
        DG_AMPA = np.asarray(self.DG_AMPA.value, dtype=np.float64).ravel()
        G_AMPA = np.asarray(self.G_AMPA.value, dtype=np.float64).ravel()
        DG_NMDA = np.asarray(self.DG_NMDA.value, dtype=np.float64).ravel()
        G_NMDA = np.asarray(self.G_NMDA.value, dtype=np.float64).ravel()
        DG_GABA_A = np.asarray(self.DG_GABA_A.value, dtype=np.float64).ravel()
        G_GABA_A = np.asarray(self.G_GABA_A.value, dtype=np.float64).ravel()
        DG_GABA_B = np.asarray(self.DG_GABA_B.value, dtype=np.float64).ravel()
        G_GABA_B = np.asarray(self.G_GABA_B.value, dtype=np.float64).ravel()
        m_fast = np.asarray(self.m_fast_NMDA_state.value, dtype=np.float64).ravel()
        m_slow = np.asarray(self.m_slow_NMDA_state.value, dtype=np.float64).ravel()
        m_Ih = np.asarray(self.m_Ih_state.value, dtype=np.float64).ravel()
        D_IKNa = np.asarray(self.D_IKNa_state.value, dtype=np.float64).ravel()
        m_IT = np.asarray(self.m_IT_state.value, dtype=np.float64).ravel()
        h_IT = np.asarray(self.h_IT_state.value, dtype=np.float64).ravel()
        ref = np.asarray(self.ref_steps.value, dtype=np.int32).ravel()
        I_stim_cur = np.asarray(self.I_stim.value, dtype=np.float64).ravel()

        # Pre-compute refractory step count
        potassium_refr_counts = self._refractory_counts(h)

        # Output arrays
        V_out = np.empty(flat_size, dtype=np.float64)
        theta_out = np.empty(flat_size, dtype=np.float64)
        ref_out = np.empty(flat_size, dtype=np.int32)
        spike_flags = np.zeros(flat_size, dtype=bool)

        # State output arrays for all 16 variables
        state_out = np.empty((flat_size, _STATE_VEC_SIZE), dtype=np.float64)
        I_NaP_out = np.empty(flat_size, dtype=np.float64)
        I_KNa_out = np.empty(flat_size, dtype=np.float64)
        I_T_out = np.empty(flat_size, dtype=np.float64)
        I_h_out = np.empty(flat_size, dtype=np.float64)

        # Flatten spike inputs
        spk_ampa_flat = spk_ampa.ravel()
        spk_nmda_flat = spk_nmda.ravel()
        spk_gaba_a_flat = spk_gaba_a.ravel()
        spk_gaba_b_flat = spk_gaba_b.ravel()
        I_stim_next_flat = I_stim_next.ravel()

        # Cache parameters as local variables for the inner loop
        _E_Na = self.E_Na
        _E_K = self.E_K
        _g_NaL = self.g_NaL
        _g_KL = self.g_KL
        _tau_m = self.tau_m
        _theta_eq = self.theta_eq
        _tau_theta = self.tau_theta
        _tau_spike = self.tau_spike
        _E_rev_AMPA = self.E_rev_AMPA
        _tau_rise_AMPA = self.tau_rise_AMPA
        _tau_decay_AMPA = self.tau_decay_AMPA
        _E_rev_NMDA = self.E_rev_NMDA
        _tau_rise_NMDA = self.tau_rise_NMDA
        _tau_decay_NMDA = self.tau_decay_NMDA
        _S_act_NMDA = self.S_act_NMDA
        _V_act_NMDA = self.V_act_NMDA
        _tau_Mg_fast_NMDA = self.tau_Mg_fast_NMDA
        _tau_Mg_slow_NMDA = self.tau_Mg_slow_NMDA
        _instant_unblock = self.instant_unblock_NMDA
        _E_rev_GABA_A = self.E_rev_GABA_A
        _tau_rise_GABA_A = self.tau_rise_GABA_A
        _tau_decay_GABA_A = self.tau_decay_GABA_A
        _E_rev_GABA_B = self.E_rev_GABA_B
        _tau_rise_GABA_B = self.tau_rise_GABA_B
        _tau_decay_GABA_B = self.tau_decay_GABA_B
        _g_peak_NaP = self.g_peak_NaP
        _E_rev_NaP = self.E_rev_NaP
        _N_NaP = self.N_NaP
        _g_peak_KNa = self.g_peak_KNa
        _E_rev_KNa = self.E_rev_KNa
        _tau_D_KNa = self.tau_D_KNa
        _g_peak_T = self.g_peak_T
        _E_rev_T = self.E_rev_T
        _N_T = self.N_T
        _g_peak_h = self.g_peak_h
        _E_rev_h = self.E_rev_h
        _voltage_clamp = self.voltage_clamp
        _V_clamp = self._V_clamp
        _cond_AMPA = self._cond_step_AMPA
        _cond_NMDA = self._cond_step_NMDA
        _cond_GABA_A = self._cond_step_GABA_A
        _cond_GABA_B = self._cond_step_GABA_B

        for i in range(flat_size):
            # Build state vector for this neuron
            y = np.array([
                V_m[i], theta_val[i],
                DG_AMPA[i], G_AMPA[i],
                DG_NMDA[i], G_NMDA[i],
                DG_GABA_A[i], G_GABA_A[i],
                DG_GABA_B[i], G_GABA_B[i],
                m_fast[i], m_slow[i],
                m_Ih[i], D_IKNa[i],
                m_IT[i], h_IT[i],
            ], dtype=np.float64)

            _ref_i = int(ref[i])
            _I_stim_i = I_stim_cur[i]

            # Intrinsic current accumulators (updated inside dynamics)
            _I_NaP = [0.0]
            _I_KNa = [0.0]
            _I_T = [0.0]
            _I_h = [0.0]

            def rhs(t_local, y,
                    _ref=_ref_i,
                    _I_stim=_I_stim_i,
                    _I_NaP=_I_NaP, _I_KNa=_I_KNa, _I_T=_I_T, _I_h=_I_h):
                """Right-hand side of the ODE system (matches ht_neuron_dynamics)."""
                V = _V_clamp if _voltage_clamp else y[0]

                # NMDA conductance with instantaneous blocking
                m_eq_nmda = 1.0 / (1.0 + math.exp(-_S_act_NMDA * (V - _V_act_NMDA)))
                mf = min(m_eq_nmda, y[10])
                ms = min(m_eq_nmda, y[11])
                if _instant_unblock:
                    m_nmda = m_eq_nmda
                else:
                    A1 = 0.51 - 0.0028 * V
                    A2 = 1.0 - A1
                    m_nmda = A1 * mf + A2 * ms

                # Synaptic currents: I = -g * (V - E)
                I_syn = (
                    -y[3] * (V - _E_rev_AMPA)
                    - y[5] * m_nmda * (V - _E_rev_NMDA)
                    - y[7] * (V - _E_rev_GABA_A)
                    - y[9] * (V - _E_rev_GABA_B)
                )

                # Post-spike K current (only during refractory)
                I_spike = -(V - _E_K) / _tau_spike if _ref > 0 else 0.0

                # Leak currents
                I_Na = -_g_NaL * (V - _E_Na)
                I_K_leak = -_g_KL * (V - _E_K)

                # I_NaP (persistent sodium)
                INaP_thresh = -55.7
                INaP_slope = 7.7
                m_inf_NaP = 1.0 / (1.0 + math.exp(-(V - INaP_thresh) / INaP_slope))
                i_NaP = -_g_peak_NaP * (m_inf_NaP ** _N_NaP) * (V - _E_rev_NaP)
                _I_NaP[0] = i_NaP

                # I_KNa (depolarization-activated K)
                d_half = 0.25
                d_val = y[13]
                if d_val > 0:
                    m_inf_KNa = 1.0 / (1.0 + (d_half / d_val) ** 3.5)
                else:
                    m_inf_KNa = 0.0
                i_KNa = -_g_peak_KNa * m_inf_KNa * (V - _E_rev_KNa)
                _I_KNa[0] = i_KNa

                # I_T (low-threshold Ca)
                i_T = -_g_peak_T * (y[14] ** _N_T) * y[15] * (V - _E_rev_T)
                _I_T[0] = i_T

                # I_h (hyperpolarization-activated)
                i_h = -_g_peak_h * y[12] * (V - _E_rev_h)
                _I_h[0] = i_h

                # Derivatives
                f = np.empty(16)

                # dV/dt
                f[0] = (I_Na + I_K_leak + I_syn + i_NaP + i_KNa + i_T + i_h + _I_stim) / _tau_m + I_spike

                # d(theta)/dt
                f[1] = -(y[1] - _theta_eq) / _tau_theta

                # AMPA synapse
                f[2] = -y[2] / _tau_rise_AMPA
                f[3] = y[2] - y[3] / _tau_decay_AMPA

                # NMDA synapse
                f[4] = -y[4] / _tau_rise_NMDA
                f[5] = y[4] - y[5] / _tau_decay_NMDA
                f[10] = (m_eq_nmda - mf) / _tau_Mg_fast_NMDA
                f[11] = (m_eq_nmda - ms) / _tau_Mg_slow_NMDA

                # GABA_A synapse
                f[6] = -y[6] / _tau_rise_GABA_A
                f[7] = y[6] - y[7] / _tau_decay_GABA_A

                # GABA_B synapse
                f[8] = -y[8] / _tau_rise_GABA_B
                f[9] = y[8] - y[9] / _tau_decay_GABA_B

                # I_KNa D variable
                D_influx_peak = 0.025
                D_thresh = -10.0
                D_slope = 5.0
                D_eq = 0.001
                D_influx = D_influx_peak / (1.0 + math.exp(-(V - D_thresh) / D_slope))
                D_eq_val = _tau_D_KNa * D_influx + D_eq
                f[13] = (D_eq_val - y[13]) / _tau_D_KNa

                # I_T gating
                tau_m_T = 0.22 / (math.exp(-(V + 132.0) / 16.7) + math.exp((V + 16.8) / 18.2)) + 0.13
                tau_h_T = 8.2 + (56.6 + 0.27 * math.exp((V + 115.2) / 5.0)) / (1.0 + math.exp((V + 86.0) / 3.2))
                m_eq_t = 1.0 / (1.0 + math.exp(-(V + 59.0) / 6.2))
                h_eq_t = 1.0 / (1.0 + math.exp((V + 83.0) / 4.0))
                f[14] = (m_eq_t - y[14]) / tau_m_T
                f[15] = (h_eq_t - y[15]) / tau_h_T

                # I_h gating
                tau_m_h = 1.0 / (math.exp(-14.59 - 0.086 * V) + math.exp(-1.87 + 0.0701 * V))
                I_h_Vthreshold = -75.0
                m_eq_ih = 1.0 / (1.0 + math.exp((V - I_h_Vthreshold) / 5.5))
                f[12] = (m_eq_ih - y[12]) / tau_m_h

                return f

            # --- ODE integration ---
            sol = solve_ivp(
                rhs,
                [0.0, h],
                y,
                method='RK45',
                rtol=self.rtol,
                atol=self.atol,
                dense_output=False,
            )
            yf = sol.y[:, -1]

            # Enforce voltage clamp
            if _voltage_clamp:
                yf[_V_M] = _V_clamp

            # Enforce instantaneous NMDA blocking
            m_eq_nmda_final = _m_eq_NMDA(yf[_V_M], _S_act_NMDA, _V_act_NMDA)
            yf[_m_fast_NMDA] = min(m_eq_nmda_final, yf[_m_fast_NMDA])
            yf[_m_slow_NMDA] = min(m_eq_nmda_final, yf[_m_slow_NMDA])

            # --- Spike detection (inside integration loop in NEST) ---
            # In NEST, spike detection happens at each adaptive sub-step inside
            # the while loop. Here we check once after the full step, matching
            # the final-state check.
            spiked = False
            if _ref_i == 0 and yf[_V_M] >= yf[_THETA]:
                # Spike!
                yf[_V_M] = _E_Na
                yf[_THETA] = _E_Na
                _ref_i = potassium_refr_counts + 1
                spiked = True

            # Decrement refractory counter (after integration loop)
            if _ref_i > 0:
                _ref_i -= 1

            # Add arriving spike inputs
            # Position 2 + 2*j is the DG variable for synapse type j
            yf[_DG_AMPA] += _cond_AMPA * spk_ampa_flat[i]
            yf[_DG_NMDA_TIMECOURSE] += _cond_NMDA * spk_nmda_flat[i]
            yf[_DG_GABA_A] += _cond_GABA_A * spk_gaba_a_flat[i]
            yf[_DG_GABA_B] += _cond_GABA_B * spk_gaba_b_flat[i]

            # Store results
            state_out[i, :] = yf
            ref_out[i] = _ref_i
            spike_flags[i] = spiked
            I_NaP_out[i] = _I_NaP[0]
            I_KNa_out[i] = _I_KNa[0]
            I_T_out[i] = _I_T[0]
            I_h_out[i] = _I_h[0]

        # Reshape and write back state
        state_out = state_out.reshape((*v_shape, _STATE_VEC_SIZE)) if len(v_shape) > 0 else state_out[0]
        if len(v_shape) > 0:
            self.V.value = state_out[..., _V_M]
            self.theta.value = state_out[..., _THETA]
            self.DG_AMPA.value = state_out[..., _DG_AMPA]
            self.G_AMPA.value = state_out[..., _G_AMPA]
            self.DG_NMDA.value = state_out[..., _DG_NMDA_TIMECOURSE]
            self.G_NMDA.value = state_out[..., _G_NMDA_TIMECOURSE]
            self.DG_GABA_A.value = state_out[..., _DG_GABA_A]
            self.G_GABA_A.value = state_out[..., _G_GABA_A]
            self.DG_GABA_B.value = state_out[..., _DG_GABA_B]
            self.G_GABA_B.value = state_out[..., _G_GABA_B]
            self.m_fast_NMDA_state.value = state_out[..., _m_fast_NMDA]
            self.m_slow_NMDA_state.value = state_out[..., _m_slow_NMDA]
            self.m_Ih_state.value = state_out[..., _m_Ih]
            self.D_IKNa_state.value = state_out[..., _D_IKNa]
            self.m_IT_state.value = state_out[..., _m_IT]
            self.h_IT_state.value = state_out[..., _h_IT]
        else:
            self.V.value = state_out[_V_M]
            self.theta.value = state_out[_THETA]
            self.DG_AMPA.value = state_out[_DG_AMPA]
            self.G_AMPA.value = state_out[_G_AMPA]
            self.DG_NMDA.value = state_out[_DG_NMDA_TIMECOURSE]
            self.G_NMDA.value = state_out[_G_NMDA_TIMECOURSE]
            self.DG_GABA_A.value = state_out[_DG_GABA_A]
            self.G_GABA_A.value = state_out[_G_GABA_A]
            self.DG_GABA_B.value = state_out[_DG_GABA_B]
            self.G_GABA_B.value = state_out[_G_GABA_B]
            self.m_fast_NMDA_state.value = state_out[_m_fast_NMDA]
            self.m_slow_NMDA_state.value = state_out[_m_slow_NMDA]
            self.m_Ih_state.value = state_out[_m_Ih]
            self.D_IKNa_state.value = state_out[_D_IKNa]
            self.m_IT_state.value = state_out[_m_IT]
            self.h_IT_state.value = state_out[_h_IT]

        # Intrinsic currents
        self.I_NaP_val.value = I_NaP_out.reshape(v_shape)
        self.I_KNa_val.value = I_KNa_out.reshape(v_shape)
        self.I_T_val.value = I_T_out.reshape(v_shape)
        self.I_h_val.value = I_h_out.reshape(v_shape)

        # Refractory counter
        self.ref_steps.value = jnp.asarray(ref_out.reshape(v_shape), dtype=jnp.int32)

        # Stimulation current for next step
        self.I_stim.value = I_stim_next

        # Spike time update
        spike_mask = spike_flags.reshape(v_shape)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        # Return spike output
        V_spike = np.where(spike_flags, 1e-12, -1.0).reshape(v_shape)
        return self.get_spike(V_spike)
