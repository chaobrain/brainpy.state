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

r"""NEST-compatible ``ht_neuron`` model (Hill & Tononi, 2005).

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

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size
from scipy.integrate import solve_ivp

from ._base import NESTNeuron
from ._utils import is_tracer

__all__ = [
    'ht_neuron',
]


# ---------------------------------------------------------------------------
# Equilibrium / steady-state helper functions (module-level, pure NumPy)
# ---------------------------------------------------------------------------

def _m_eq_h(V):
    r"""Compute equilibrium activation for I_h hyperpolarization-activated current.

    Implements the steady-state activation function for the h-current, which activates
    with hyperpolarization and provides a depolarizing inward current that contributes
    to resonance and rebound excitation in thalamocortical neurons.

    Parameters
    ----------
    V : float
        Membrane potential in mV. Typical physiological range: -90 to +30 mV.

    Returns
    -------
    float
        Equilibrium activation value m_∞^Ih ∈ [0, 1]. Approaches 1 at hyperpolarized
        potentials (V < -75 mV), and 0 at depolarized potentials.

    Notes
    -----
    The activation follows a Boltzmann sigmoid with inflection at V = -75 mV and
    slope factor 5.5 mV:

    .. math::

        m_\infty^{I_h}(V) = \frac{1}{1 + \exp\left(\frac{V + 75}{5.5}\right)}

    This function is used to initialize the m_Ih state variable and is not voltage-
    dependent during simulation (actual dynamics are governed by tau_m_h).
    """
    I_h_Vthreshold = -75.0
    return 1.0 / (1.0 + math.exp((V - I_h_Vthreshold) / 5.5))


def _h_eq_T(V):
    r"""Compute equilibrium inactivation for I_T low-threshold calcium current.

    Calculates the steady-state inactivation gate for the T-type Ca²⁺ channel,
    which is responsible for burst firing and oscillatory behavior in thalamic neurons.
    Inactivation is voltage-dependent and de-inactivates at hyperpolarized potentials.

    Parameters
    ----------
    V : float
        Membrane potential in mV. Typical physiological range: -90 to +30 mV.

    Returns
    -------
    float
        Equilibrium inactivation value h_∞^IT ∈ [0, 1]. Approaches 1 at hyperpolarized
        potentials (V < -83 mV) where the channel is deinactivated, and 0 at
        depolarized potentials where the channel is inactivated.

    Notes
    -----
    The inactivation follows a Boltzmann sigmoid with inflection at V = -83 mV and
    slope factor 4.0 mV:

    .. math::

        h_\infty^{I_T}(V) = \frac{1}{1 + \exp\left(\frac{V + 83}{4}\right)}

    This steep voltage dependence ensures that T-channels recover from inactivation
    only after sufficient hyperpolarization, enabling rebound burst firing.
    """
    return 1.0 / (1.0 + math.exp((V + 83.0) / 4.0))


def _m_eq_T(V):
    r"""Compute equilibrium activation for I_T low-threshold calcium current.

    Calculates the steady-state activation gate for the T-type Ca²⁺ channel. This
    channel activates at relatively hyperpolarized potentials (hence "low-threshold")
    and mediates burst firing when combined with de-inactivation.

    Parameters
    ----------
    V : float
        Membrane potential in mV. Typical physiological range: -90 to +30 mV.

    Returns
    -------
    float
        Equilibrium activation value m_∞^IT ∈ [0, 1]. Approaches 1 at depolarized
        potentials (V > -59 mV), and 0 at hyperpolarized potentials.

    Notes
    -----
    The activation follows a Boltzmann sigmoid with inflection at V = -59 mV and
    slope factor 6.2 mV:

    .. math::

        m_\infty^{I_T}(V) = \frac{1}{1 + \exp\left(-\frac{V + 59}{6.2}\right)}

    In the full I_T current, this activation is raised to the power N_T (typically 2)
    and multiplied by the inactivation variable h_IT, giving the current a transient
    character essential for burst generation.
    """
    return 1.0 / (1.0 + math.exp(-(V + 59.0) / 6.2))


def _D_eq_KNa(V, tau_D_KNa):
    r"""Compute steady-state D value for I_KNa depolarization-activated potassium current.

    The D variable represents an internal concentration-like quantity that accumulates
    during sustained depolarization and drives the slow activation of I_KNa. This
    provides spike-frequency adaptation on a longer timescale than typical AHP currents.

    Parameters
    ----------
    V : float
        Membrane potential in mV. Typical physiological range: -90 to +30 mV.
    tau_D_KNa : float
        Relaxation time constant in ms. Controls the rate at which D approaches its
        steady-state value. Typical value: 1250 ms (slow adaptation).

    Returns
    -------
    float
        Equilibrium D value (dimensionless, positive). At rest (~-70 mV), D ≈ 0.001;
        at depolarized potentials (>-10 mV), D can reach ~0.03.

    Notes
    -----
    The steady-state D is computed from a voltage-dependent influx term:

    .. math::

        D_{influx}(V) &= \frac{0.025}{1 + \exp\left(-\frac{V + 10}{5}\right)} \\
        D_\infty(V) &= \tau_{D,KNa} \cdot D_{influx}(V) + 0.001

    The influx is a sigmoid centered at V = -10 mV with slope 5 mV, multiplied by
    tau_D_KNa to yield the equilibrium value. The additive constant 0.001 ensures
    a minimum baseline D value even at hyperpolarized potentials.

    This equilibrium function is used only for initialization; the full dynamics
    during simulation include time-dependent relaxation toward D_∞.
    """
    D_influx_peak = 0.025
    D_thresh = -10.0
    D_slope = 5.0
    D_eq = 0.001
    D_influx = D_influx_peak / (1.0 + math.exp(-(V - D_thresh) / D_slope))
    return tau_D_KNa * D_influx + D_eq


def _m_eq_NMDA(V, S_act_NMDA, V_act_NMDA):
    r"""Compute steady-state magnesium unblock ratio for NMDA receptor channels.

    NMDA receptors are blocked by extracellular Mg²⁺ at hyperpolarized potentials
    and unblock with depolarization, providing voltage-dependent gain and enabling
    coincidence detection of pre- and post-synaptic activity.

    Parameters
    ----------
    V : float
        Membrane potential in mV. Typical physiological range: -90 to +30 mV.
    S_act_NMDA : float
        Slope parameter for the NMDA unblocking sigmoid in 1/mV. Default: 0.081 mV⁻¹.
        Higher values produce steeper voltage dependence.
    V_act_NMDA : float
        Voltage at inflection point of the NMDA unblocking sigmoid in mV. Default:
        -25.57 mV. This is the potential at which 50% of channels are unblocked.

    Returns
    -------
    float
        Equilibrium Mg²⁺ unblock fraction m_∞^NMDA ∈ [0, 1]. At V = V_act_NMDA,
        m_∞ = 0.5. Approaches 1 at depolarized potentials (full unblock) and 0 at
        hyperpolarized potentials (full block).

    Notes
    -----
    The unblock fraction follows a Boltzmann sigmoid:

    .. math::

        m_\infty^{NMDA}(V) = \frac{1}{1 + \exp\left(-S_{act} \cdot (V - V_{act})\right)}

    When ``instant_unblock_NMDA`` is True, this equilibrium value is used directly
    for the NMDA conductance calculation. When False, the model uses two-stage
    kinetics with fast and slow unblocking time constants (tau_Mg_fast_NMDA,
    tau_Mg_slow_NMDA) as described in Vargas-Caballero & Robinson (2003).
    """
    return 1.0 / (1.0 + math.exp(-S_act_NMDA * (V - V_act_NMDA)))


def _m_NMDA(V, m_eq, m_fast, m_slow, instant_unblock_NMDA):
    r"""Compute effective NMDA channel activation combining fast and slow unblocking kinetics.

    NMDA receptors exhibit two-stage Mg²⁺ unblocking kinetics: a fast component
    (tau ~0.68 ms) and a slow component (tau ~22.7 ms). The relative contributions
    of these components are voltage-dependent, with the fast component dominating
    at more depolarized potentials.

    Parameters
    ----------
    V : float
        Membrane potential in mV. Typical physiological range: -90 to +30 mV.
    m_eq : float
        Equilibrium Mg²⁺ unblock fraction ∈ [0, 1] at voltage V, computed from
        _m_eq_NMDA.
    m_fast : float
        Fast unblocking state variable ∈ [0, 1]. Approaches m_eq with time constant
        tau_Mg_fast_NMDA (~0.68 ms).
    m_slow : float
        Slow unblocking state variable ∈ [0, 1]. Approaches m_eq with time constant
        tau_Mg_slow_NMDA (~22.7 ms).
    instant_unblock_NMDA : bool
        If True, assume instantaneous unblocking and return m_eq directly, ignoring
        the two-stage kinetics. If False, compute weighted average of fast and slow
        components.

    Returns
    -------
    float
        Effective NMDA channel activation m^NMDA ∈ [0, 1]. This value multiplies the
        NMDA conductance to give the voltage-dependent NMDA current:
        I_NMDA = -g_NMDA * m^NMDA * (V - E_NMDA).

    Notes
    -----
    **1. Instantaneous Unblocking Mode**

    When ``instant_unblock_NMDA = True``:

    .. math::

        m^{NMDA} = m_\infty^{NMDA}(V)

    **2. Two-Stage Kinetics Mode**

    When ``instant_unblock_NMDA = False``, the effective activation is a weighted
    sum of fast and slow components with voltage-dependent coefficients:

    .. math::

        A_1(V) &= 0.51 - 0.0028 \cdot V \\
        A_2(V) &= 1 - A_1(V) \\
        m^{NMDA} &= A_1(V) \cdot m_{fast} + A_2(V) \cdot m_{slow}

    At rest (V ≈ -70 mV), A₁ ≈ 0.71 and A₂ ≈ 0.29, so the fast component dominates.
    At depolarized potentials (V = 0 mV), A₁ ≈ 0.51 and A₂ ≈ 0.49, giving more
    equal weighting.

    This two-stage model is based on Vargas-Caballero & Robinson (2003), who showed
    that slow Mg²⁺ unblocking limits NMDA contribution to spike generation.

    References
    ----------
    Vargas-Caballero M, Robinson HPC (2003). A slow fraction of Mg²⁺ unblock of
    NMDA receptors limits their contribution to spike generation in cortical
    pyramidal neurons. Journal of Neurophysiology, 89:2778-2783.
    """
    if instant_unblock_NMDA:
        return m_eq
    A1 = 0.51 - 0.0028 * V
    A2 = 1.0 - A1
    return A1 * m_fast + A2 * m_slow


def _beta_normalization_factor(tau_rise, tau_decay):
    r"""Compute normalization constant for beta-function (difference-of-exponentials) synapse.

    The beta function describes a synaptic conductance that rises and decays with two
    different time constants. This normalization factor ensures that a unit synaptic
    input produces a peak conductance of exactly g_peak, independent of the specific
    tau_rise and tau_decay values.

    This implementation matches NEST's ``beta_normalization_factor()`` from
    ``libnestutil/beta_normalization_factor.h``.

    Parameters
    ----------
    tau_rise : float
        Synaptic rise time constant in ms. Must be positive and less than tau_decay
        for proper beta-function behavior.
    tau_decay : float
        Synaptic decay time constant in ms. Must be positive and greater than tau_rise.

    Returns
    -------
    float
        Normalization constant (positive, unitless). Multiply this by g_peak and the
        synaptic spike count to get the conductance step added to the DG variable.

    Notes
    -----
    **1. Mathematical Derivation**

    The unnormalized beta-function conductance kernel is:

    .. math::

        g(t) = \exp(-t/\tau_{decay}) - \exp(-t/\tau_{rise})

    The peak occurs at time:

    .. math::

        t_{peak} = \frac{\tau_{rise} \cdot \tau_{decay}}{\tau_{decay} - \tau_{rise}}
                   \ln\left(\frac{\tau_{decay}}{\tau_{rise}}\right)

    Evaluating g(t_peak) gives the peak amplitude. The normalization factor is:

    .. math::

        \text{norm} = \frac{1/\tau_{rise} - 1/\tau_{decay}}{g(t_{peak})}

    **2. Alpha-Function Limit**

    When tau_rise → tau_decay, the beta function becomes an alpha function:

    .. math::

        g(t) = \frac{e \cdot t}{\tau} \exp(-t/\tau)

    with normalization factor e / tau_decay.

    **3. Numerical Stability**

    The function uses machine epsilon to detect near-equality of time constants and
    avoid division by zero, ensuring stable computation across all parameter regimes.

    **4. Usage in ht_neuron**

    For each synapse type (AMPA, NMDA, GABA_A, GABA_B), the normalization factor is
    precomputed during initialization and stored as _cond_step_*. When a spike arrives,
    the DG variable is incremented by:

    .. math::

        \Delta DG = g_{peak} \cdot \text{norm} \cdot N_{spikes}

    where N_spikes is the weighted spike count delivered to that receptor type.
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


class ht_neuron(NESTNeuron):
    r"""NEST-compatible Hill-Tononi thalamocortical neuron model with intrinsic currents.

    Implements the conductance-based integrate-and-fire neuron model from Hill & Tononi
    (2005) designed to simulate sleep-wake dynamics in thalamocortical networks. Features
    adaptive threshold, repolarizing post-spike potassium current, four receptor types
    (AMPA, NMDA, GABA_A, GABA_B), and four intrinsic currents (I_NaP, I_KNa, I_T, I_h)
    that mediate burst firing, adaptation, and oscillatory behavior.

    This implementation replicates NEST's ``ht_neuron`` (models/ht_neuron.{h,cpp})
    using JAX-compatible adaptive ODE integration with scipy RK45.

    Parameters
    ----------
    in_size : int or tuple of int
        Population shape (e.g., 100 or (10, 10)). Determines the number of neurons
        in this layer.
    E_Na : float, default=30.0
        Sodium reversal potential in mV. Sets the depolarized reset level after spike.
    E_K : float, default=-90.0
        Potassium reversal potential in mV. Sets the hyperpolarized target for
        repolarization during refractory period.
    g_NaL : float, default=0.2
        Sodium leak conductance (unitless). Contributes depolarizing leak current.
    g_KL : float, default=1.0
        Potassium leak conductance (unitless). Contributes hyperpolarizing leak current.
    tau_m : float, default=16.0
        Membrane time constant in ms. Governs the rate of subthreshold voltage changes.
    theta_eq : float, default=-51.0
        Equilibrium spike threshold in mV. The threshold relaxes to this value with
        time constant tau_theta.
    tau_theta : float, default=2.0
        Threshold relaxation time constant in ms. Controls adaptation timescale.
    tau_spike : float, default=1.75
        Repolarization time constant for post-spike potassium current in ms. Governs
        the speed of voltage recovery during refractory period.
    t_ref : float, default=2.0
        Absolute refractory period in ms. Duration of post-spike potassium current.
    g_peak_AMPA : float, default=0.1
        Peak AMPA conductance (unitless). Scaled by spike inputs to produce excitatory
        synaptic current.
    tau_rise_AMPA : float, default=0.5
        AMPA conductance rise time constant in ms. Must be < tau_decay_AMPA.
    tau_decay_AMPA : float, default=2.4
        AMPA conductance decay time constant in ms.
    E_rev_AMPA : float, default=0.0
        AMPA reversal potential in mV.
    g_peak_NMDA : float, default=0.075
        Peak NMDA conductance (unitless). Subject to voltage-dependent Mg²⁺ block.
    tau_rise_NMDA : float, default=4.0
        NMDA conductance rise time constant in ms. Must be < tau_decay_NMDA.
    tau_decay_NMDA : float, default=40.0
        NMDA conductance decay time constant in ms.
    E_rev_NMDA : float, default=0.0
        NMDA reversal potential in mV.
    V_act_NMDA : float, default=-25.57
        Voltage at 50% NMDA Mg²⁺ unblock in mV.
    S_act_NMDA : float, default=0.081
        Slope of NMDA Mg²⁺ unblocking sigmoid in 1/mV.
    tau_Mg_slow_NMDA : float, default=22.7
        Slow Mg²⁺ unblocking time constant in ms. Must be > tau_Mg_fast_NMDA.
    tau_Mg_fast_NMDA : float, default=0.68
        Fast Mg²⁺ unblocking time constant in ms.
    instant_unblock_NMDA : bool, default=False
        If True, use instantaneous Mg²⁺ unblocking (m^NMDA = m_∞). If False, use
        two-stage kinetics with fast and slow unblocking components.
    g_peak_GABA_A : float, default=0.33
        Peak GABA_A conductance (unitless). Fast inhibitory synaptic current.
    tau_rise_GABA_A : float, default=1.0
        GABA_A rise time constant in ms. Must be < tau_decay_GABA_A.
    tau_decay_GABA_A : float, default=7.0
        GABA_A decay time constant in ms.
    E_rev_GABA_A : float, default=-70.0
        GABA_A reversal potential in mV.
    g_peak_GABA_B : float, default=0.0132
        Peak GABA_B conductance (unitless). Slow inhibitory synaptic current.
    tau_rise_GABA_B : float, default=60.0
        GABA_B rise time constant in ms. Must be < tau_decay_GABA_B.
    tau_decay_GABA_B : float, default=200.0
        GABA_B decay time constant in ms.
    E_rev_GABA_B : float, default=-90.0
        GABA_B reversal potential in mV.
    g_peak_NaP : float, default=1.0
        Peak persistent sodium current conductance (unitless). Mediates subthreshold
        depolarization and bistability.
    E_rev_NaP : float, default=30.0
        I_NaP reversal potential in mV.
    N_NaP : float, default=3.0
        I_NaP activation exponent (power to which m_∞ is raised).
    g_peak_KNa : float, default=1.0
        Peak I_KNa conductance (unitless). Provides slow spike-frequency adaptation.
    E_rev_KNa : float, default=-90.0
        I_KNa reversal potential in mV.
    tau_D_KNa : float, default=1250.0
        I_KNa D-variable relaxation time constant in ms. Large value produces very
        slow adaptation (~seconds).
    g_peak_T : float, default=1.0
        Peak low-threshold Ca²⁺ current conductance (unitless). Mediates rebound
        bursts and oscillations.
    E_rev_T : float, default=0.0
        I_T reversal potential in mV.
    N_T : float, default=2.0
        I_T activation exponent (power to which m_IT is raised).
    g_peak_h : float, default=1.0
        Peak hyperpolarization-activated current conductance (unitless). Contributes
        to rebound excitation and resonance.
    E_rev_h : float, default=-40.0
        I_h reversal potential in mV.
    voltage_clamp : bool, default=False
        If True, clamp membrane potential at its initial value throughout simulation.
        Used for testing intrinsic current dynamics in isolation.
    rtol : float, default=1e-3
        Relative tolerance for ODE solver. Matches NEST's GSL RKF45 default.
    atol : float, default=1e-9
        Absolute tolerance for ODE solver.
    spk_fun : Callable, default=braintools.surrogate.ReluGrad()
        Surrogate gradient function for differentiable spike generation.
    spk_reset : str, default='hard'
        Spike reset mode: 'hard' (stop gradient) or 'soft' (V -= V_th).
    name : str or None, default=None
        Name of this neuron population.

    Parameter Mapping
    -----------------

    The table below maps brainpy.state parameter names to NEST equivalents:

    ========================= ====================== ======== ==============
    brainpy.state Parameter   NEST Parameter         Default  Units
    ========================= ====================== ======== ==============
    ``in_size``               (N/A)                  ---      ---
    ``E_Na``                  ``E_Na``               30.0     mV
    ``E_K``                   ``E_K``                -90.0    mV
    ``g_NaL``                 ``g_NaL``              0.2      (unitless)
    ``g_KL``                  ``g_KL``               1.0      (unitless)
    ``tau_m``                 ``tau_m``              16.0     ms
    ``theta_eq``              ``theta_eq``           -51.0    mV
    ``tau_theta``             ``tau_theta``          2.0      ms
    ``tau_spike``             ``tau_spike``          1.75     ms
    ``t_ref``                 ``t_ref``              2.0      ms
    ``g_peak_AMPA``           ``g_peak_AMPA``        0.1      (unitless)
    ``tau_rise_AMPA``         ``tau_rise_AMPA``      0.5      ms
    ``tau_decay_AMPA``        ``tau_decay_AMPA``     2.4      ms
    ``E_rev_AMPA``            ``E_rev_AMPA``         0.0      mV
    ``g_peak_NMDA``           ``g_peak_NMDA``        0.075    (unitless)
    ``tau_rise_NMDA``         ``tau_rise_NMDA``      4.0      ms
    ``tau_decay_NMDA``        ``tau_decay_NMDA``     40.0     ms
    ``E_rev_NMDA``            ``E_rev_NMDA``         0.0      mV
    ``V_act_NMDA``            ``V_act_NMDA``         -25.57   mV
    ``S_act_NMDA``            ``S_act_NMDA``         0.081    1/mV
    ``tau_Mg_slow_NMDA``      ``tau_Mg_slow_NMDA``   22.7     ms
    ``tau_Mg_fast_NMDA``      ``tau_Mg_fast_NMDA``   0.68     ms
    ``instant_unblock_NMDA``  ``instant_unblock``    False    ---
    ``g_peak_GABA_A``         ``g_peak_GABA_A``      0.33     (unitless)
    ``tau_rise_GABA_A``       ``tau_rise_GABA_A``    1.0      ms
    ``tau_decay_GABA_A``      ``tau_decay_GABA_A``   7.0      ms
    ``E_rev_GABA_A``          ``E_rev_GABA_A``       -70.0    mV
    ``g_peak_GABA_B``         ``g_peak_GABA_B``      0.0132   (unitless)
    ``tau_rise_GABA_B``       ``tau_rise_GABA_B``    60.0     ms
    ``tau_decay_GABA_B``      ``tau_decay_GABA_B``   200.0    ms
    ``E_rev_GABA_B``          ``E_rev_GABA_B``       -90.0    mV
    ``g_peak_NaP``            ``g_peak_NaP``         1.0      (unitless)
    ``E_rev_NaP``             ``E_rev_NaP``          30.0     mV
    ``N_NaP``                 ``NaP_N``              3.0      ---
    ``g_peak_KNa``            ``g_peak_KNa``         1.0      (unitless)
    ``E_rev_KNa``             ``E_rev_KNa``          -90.0    mV
    ``tau_D_KNa``             ``tau_D_KNa``          1250.0   ms
    ``g_peak_T``              ``g_peak_T``           1.0      (unitless)
    ``E_rev_T``               ``E_rev_T``            0.0      mV
    ``N_T``                   ``T_N``                2.0      ---
    ``g_peak_h``              ``g_peak_h``           1.0      (unitless)
    ``E_rev_h``               ``E_rev_h``            -40.0    mV
    ``voltage_clamp``         ``voltage_clamp``      False    ---
    ``rtol``                  (GSL tolerance)        1e-3     ---
    ``atol``                  (GSL tolerance)        0.0      ---
    ========================= ====================== ======== ==============

    Notes
    -----
    **1. Model Architecture**

    The ht_neuron is an integrate-and-fire model with:

    - **Adaptive threshold**: Threshold increases transiently after spike, then relaxes
      to theta_eq, providing spike-frequency adaptation on ~ms timescale.
    - **Soft reset**: No hard voltage reset. Instead, V and theta are set to E_Na, and
      a repolarizing K⁺ current drives voltage back toward E_K during t_ref.
    - **Four synaptic receptor types**: AMPA (fast excitation), NMDA (slow excitation
      with voltage-dependent Mg²⁺ block), GABA_A (fast inhibition), GABA_B (slow
      inhibition). Each uses beta-function (biexponential) conductance time course.
    - **Four intrinsic currents**:

      * **I_NaP** (persistent Na⁺): Subthreshold depolarizing current; enables
        bistability and up-states.
      * **I_KNa** (depolarization-activated K⁺): Very slow adaptation (~1 s timescale).
      * **I_T** (low-threshold Ca²⁺): Mediates rebound bursts; deinactivates during
        hyperpolarization and activates rapidly on depolarization.
      * **I_h** (hyperpolarization-activated cation current): Sag current; contributes
        to rebound and resonance.

    **2. Membrane Dynamics**

    The membrane potential obeys:

    .. math::

        \frac{dV}{dt} = \frac{I_{leak} + I_{syn} + I_{intrinsic} + I_{stim}}{\tau_m} + I_{spike}

    where:

    .. math::

        I_{leak} &= -g_{NaL}(V - E_{Na}) - g_{KL}(V - E_K) \\
        I_{syn} &= -g_{AMPA}(V - E_{AMPA}) - g_{NMDA} m^{NMDA}(V - E_{NMDA}) \\
                &\quad - g_{GABA_A}(V - E_{GABA_A}) - g_{GABA_B}(V - E_{GABA_B}) \\
        I_{intrinsic} &= I_{NaP} + I_{KNa} + I_T + I_h \\
        I_{spike} &= \begin{cases}
          -(V - E_K) / \tau_{spike} & \text{if refractory} \\
          0 & \text{otherwise}
        \end{cases}

    **3. Dynamic Threshold**

    .. math::

        \frac{d\theta}{dt} = -\frac{\theta - \theta_{eq}}{\tau_\theta}

    On spike, theta is reset to E_Na (along with V), then decays back to theta_eq.
    This provides fast spike-frequency adaptation.

    **4. Beta-Function Synapses**

    Each synapse type uses a two-variable beta function (difference of exponentials):

    .. math::

        \frac{dg'}{dt} &= -\frac{g'}{\tau_{rise}} \\
        \frac{dg}{dt} &= g' - \frac{g}{\tau_{decay}}

    On arrival of a spike, the DG variable (g') is incremented by:

    .. math::

        \Delta g' = g_{peak} \cdot \text{norm}(\tau_{rise}, \tau_{decay}) \cdot w

    where norm is the beta normalization factor and w is the synaptic weight.

    **5. NMDA Voltage Dependence**

    NMDA channels are blocked by Mg²⁺ at hyperpolarized potentials and unblock with
    depolarization. Two modes are available:

    - **Instantaneous unblocking** (instant_unblock_NMDA=True):

      .. math::

          m^{NMDA} = \frac{1}{1 + \exp(-S_{act}(V - V_{act}))}

    - **Two-stage kinetics** (instant_unblock_NMDA=False):

      .. math::

          \frac{dm_{fast}}{dt} &= \frac{m_\infty - m_{fast}}{\tau_{Mg,fast}} \\
          \frac{dm_{slow}}{dt} &= \frac{m_\infty - m_{slow}}{\tau_{Mg,slow}} \\
          m^{NMDA} &= A_1(V) m_{fast} + A_2(V) m_{slow}

      where A₁(V) = 0.51 - 0.0028V and A₂ = 1 - A₁. This captures the experimentally
      observed slow Mg²⁺ unblocking kinetics (Vargas-Caballero & Robinson, 2003).

    **6. Intrinsic Currents**

    - **I_NaP** (persistent sodium):

      .. math::

          m_\infty &= \frac{1}{1 + \exp(-(V + 55.7)/7.7)} \\
          I_{NaP} &= -g_{NaP} \cdot (m_\infty)^{N_{NaP}} \cdot (V - E_{NaP})

      No inactivation; provides tonic depolarizing drive.

    - **I_KNa** (depolarization-activated potassium):

      .. math::

          D_{influx} &= \frac{0.025}{1 + \exp(-(V + 10)/5)} \\
          \frac{dD}{dt} &= \frac{\tau_{D,KNa} \cdot D_{influx} + 0.001 - D}{\tau_{D,KNa}} \\
          m_\infty &= \frac{1}{1 + (0.25/D)^{3.5}} \\
          I_{KNa} &= -g_{KNa} \cdot m_\infty \cdot (V - E_{KNa})

      D accumulates slowly during depolarization; provides adaptation on ~second timescale.

    - **I_T** (low-threshold Ca²⁺):

      .. math::

          m_\infty &= \frac{1}{1 + \exp(-(V + 59)/6.2)} \\
          h_\infty &= \frac{1}{1 + \exp((V + 83)/4)} \\
          \tau_m &= 0.22 / (\exp(-(V+132)/16.7) + \exp((V+16.8)/18.2)) + 0.13 \\
          \tau_h &= 8.2 + \frac{56.6 + 0.27 \exp((V+115.2)/5)}{1 + \exp((V+86)/3.2)} \\
          I_T &= -g_T \cdot m^{N_T} \cdot h \cdot (V - E_T)

      Activation is fast; inactivation is slower. Channel deinactivates during
      hyperpolarization, enabling rebound bursts.

    - **I_h** (hyperpolarization-activated cation current):

      .. math::

          m_\infty &= \frac{1}{1 + \exp((V + 75)/5.5)} \\
          \tau_m &= \frac{1}{\exp(-14.59 - 0.086V) + \exp(-1.87 + 0.0701V)} \\
          I_h &= -g_h \cdot m \cdot (V - E_h)

      Activates slowly at hyperpolarized potentials; provides depolarizing sag and
      contributes to rebound.

    **7. Spike Detection and Reset**

    A spike occurs when ``ref_steps == 0`` and ``V >= theta``. On spike:

    - V → E_Na (≈ +30 mV)
    - theta → E_Na
    - ref_steps → ceil(t_ref / dt) + 1

    During the refractory period, I_spike drives V back toward E_K.

    **8. Numerical Integration**

    The model uses scipy's ``solve_ivp`` with method='RK45' (Dormand-Prince 5(4) adaptive
    Runge-Kutta). This matches NEST's GSL RKF45 integrator in terms of order and adaptive
    step-size control. Default tolerances (rtol=1e-3, atol=1e-9) match NEST defaults.

    **9. Conductance Units**

    All conductances are **unitless** in this model. The membrane equation is written
    as dV/dt = I/tau_m, meaning currents have units of mV/ms (i.e., they are already
    divided by capacitance). Peak conductances g_peak_* scale the synaptic currents.

    **10. Sleep-Wake Transitions**

    The ht_neuron was designed to model thalamocortical neurons that exhibit two
    distinct firing modes:

    - **Tonic firing** (awake/depolarized): Regular spiking driven by excitatory input
      and I_NaP.
    - **Burst firing** (sleep/hyperpolarized): Rebound bursts mediated by I_T
      deinactivation and I_h rebound.

    By varying the balance of intrinsic current conductances (g_peak_T, g_peak_h,
    g_peak_NaP) and background synaptic input, the model can transition between these
    modes, reproducing the sleep-wake dynamics observed in thalamocortical circuits.

    Examples
    --------
    **Example 1: Single neuron with injected current**

    .. code-block:: python

        >>> import brainpy as bp
        >>> import brainpy.state as bps
        >>> import saiunit as u
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Create a single ht_neuron
        >>> neuron = bps.ht_neuron(1, g_peak_T=0.0, g_peak_h=0.0)
        >>>
        >>> # Initialize state
        >>> with bp.environ.context(dt=0.1 * u.ms):
        ...     neuron.init_all_states()
        ...
        ...     # Simulate 500 ms with step current
        ...     currents = np.concatenate([
        ...         np.zeros(1000),
        ...         np.ones(3000) * 2.0,  # 2 mV/ms injected current
        ...         np.zeros(1000)
        ...     ])
        ...     voltages = []
        ...     for I in currents:
        ...         neuron.update(I)
        ...         voltages.append(neuron.V.value[0])
        >>>
        >>> # Plot membrane potential
        >>> times = np.arange(len(voltages)) * 0.1
        >>> plt.figure(figsize=(10, 4))
        >>> plt.plot(times, voltages)
        >>> plt.xlabel('Time (ms)')
        >>> plt.ylabel('Membrane potential (mV)')
        >>> plt.title('ht_neuron response to step current')
        >>> plt.show()

    **Example 2: Rebound burst with I_T**

    .. code-block:: python

        >>> # Enable I_T for burst firing
        >>> neuron = bps.ht_neuron(1, g_peak_T=1.0, g_peak_h=0.5)
        >>>
        >>> with bp.environ.context(dt=0.1 * u.ms):
        ...     neuron.init_all_states()
        ...
        ...     # Hyperpolarize, then release
        ...     currents = np.concatenate([
        ...         np.zeros(500),
        ...         -np.ones(1000) * 3.0,  # hyperpolarizing current
        ...         np.zeros(1500)
        ...     ])
        ...     voltages = []
        ...     for I in currents:
        ...         neuron.update(I)
        ...         voltages.append(neuron.V.value[0])
        >>>
        >>> # Observe rebound burst after hyperpolarization ends
        >>> plt.figure(figsize=(10, 4))
        >>> plt.plot(np.arange(len(voltages)) * 0.1, voltages)
        >>> plt.xlabel('Time (ms)')
        >>> plt.ylabel('V (mV)')
        >>> plt.title('Rebound burst mediated by I_T')
        >>> plt.show()

    **Example 3: Multi-receptor synaptic input**

    .. code-block:: python

        >>> # Network with AMPA and NMDA receptors
        >>> pre = bps.LIF(100, V_rest=-70*u.mV, V_th=-50*u.mV, V_reset=-70*u.mV,
        ...               tau=20*u.ms, R=1*u.ohm, V_initializer=bp.init.Normal(-70, 5))
        >>> post = bps.ht_neuron(10, g_peak_AMPA=0.1, g_peak_NMDA=0.05,
        ...                      instant_unblock_NMDA=False)
        >>>
        >>> # Create projections for AMPA and NMDA
        >>> ampa_proj = bps.AlignPostProj(
        ...     pre=pre, post=post,
        ...     comm=bp.event.FixedProb(0.1, weight=1.0),
        ...     syn=bps.Expon.desc(tau=2.4 * u.ms),
        ...     label='AMPA'
        ... )
        >>> nmda_proj = bps.AlignPostProj(
        ...     pre=pre, post=post,
        ...     comm=bp.event.FixedProb(0.1, weight=1.0),
        ...     syn=bps.Expon.desc(tau=40 * u.ms),
        ...     label='NMDA'
        ... )
        >>>
        >>> # Simulate network dynamics
        >>> # (implementation depends on BrainPy network API)

    See Also
    --------
    hh_psc_alpha : Hodgkin-Huxley neuron with alpha-shaped PSCs
    iaf_cond_exp : Simple IAF with exponential conductance synapses
    aeif_cond_alpha : Adaptive exponential IAF with alpha conductances

    References
    ----------
    .. [1] Hill S, Tononi G (2005). Modeling sleep and wakefulness in the
           thalamocortical system. Journal of Neurophysiology, 93:1671-1698.
           DOI: https://doi.org/10.1152/jn.00915.2004
    .. [2] Vargas-Caballero M, Robinson HPC (2003). A slow fraction of Mg²⁺
           unblock of NMDA receptors limits their contribution to spike
           generation in cortical pyramidal neurons. Journal of Neurophysiology,
           89:2778-2783. DOI: https://doi.org/10.1152/jn.01038.2002
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
        r"""Validate parameter constraints to ensure physiological consistency.

        Checks all parameter values against the same constraints enforced by NEST's
        ``ht_neuron::Parameters_::set()`` method. Raises ValueError if any constraint
        is violated.

        Raises
        ------
        ValueError
            If any conductance is negative, if any time constant is non-positive,
            if S_act_NMDA < 0, if t_ref < 0, or if rise time >= decay time for
            any synapse or NMDA Mg²⁺ unblocking kinetics.

        Notes
        -----
        Enforced constraints:

        1. **Non-negative conductances**: All g_peak_* and g_*L parameters must be >= 0.
        2. **Positive time constants**: All tau_* parameters must be > 0.
        3. **Rise < decay ordering**: For beta-function synapses and NMDA Mg²⁺ kinetics,
           tau_rise must be strictly less than tau_decay to ensure proper biexponential
           shape.
        4. **Non-negative slope**: S_act_NMDA >= 0 (zero slope would disable voltage
           dependence).
        5. **Non-negative refractory period**: t_ref >= 0 (zero is allowed, disabling
           refractory period).

        This validation is called automatically during ``__init__`` to catch
        configuration errors early.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.g_peak_AMPA, self.tau_m, self.S_act_NMDA, self.t_ref)):
            return

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
        r"""Convert refractory period from milliseconds to integer step count.

        Computes the number of discrete simulation steps corresponding to the
        refractory period t_ref. Uses rounding to nearest integer to minimize
        discretization error.

        Parameters
        ----------
        dt_ms : float
            Simulation time step in ms. Typically obtained from brainstate.environ.get_dt().

        Returns
        -------
        int
            Number of refractory steps (non-negative). If t_ref = 0, returns 0
            (no refractory period). For t_ref = 2.0 ms and dt = 0.1 ms, returns 20.

        Notes
        -----
        The refractory counter is incremented by this value + 1 when a spike occurs,
        then decremented by 1 each step. During the refractory period (ref_steps > 0),
        the post-spike potassium current I_spike is active and spike detection is
        disabled.

        Rounding ensures that fractional steps are handled consistently: e.g., if
        t_ref=2.0 ms and dt=0.3 ms, we get round(2.0/0.3) = round(6.67) = 7 steps.
        """
        return int(round(self.t_ref / dt_ms))

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables to physiologically consistent equilibrium values.

        Sets the membrane potential to the leak reversal potential (weighted average of
        E_Na and E_K based on leak conductances), threshold to theta_eq, all synaptic
        variables to zero, and all intrinsic gating variables to their voltage-dependent
        equilibrium values at the initial membrane potential.

        Parameters
        ----------
        batch_size : int or None, default=None
            If provided, creates a batch of independent neuron states with leading
            dimension batch_size. Useful for batched simulation or data-parallel training.
        **kwargs
            Additional keyword arguments (currently unused; reserved for future extensions).

        Notes
        -----
        **1. Initial Membrane Potential**

        Computed from leak conductance balance:

        .. math::

            V_{init} = \frac{g_{NaL} \cdot E_{Na} + g_{KL} \cdot E_K}{g_{NaL} + g_{KL}}

        With default parameters (g_NaL=0.2, g_KL=1.0, E_Na=30 mV, E_K=-90 mV):

        .. math::

            V_{init} = \frac{0.2 \cdot 30 + 1.0 \cdot (-90)}{0.2 + 1.0} = -70\ \text{mV}

        **2. Threshold Initialization**

        theta is set to theta_eq (default -51 mV).

        **3. Synaptic Variables**

        All beta-function state variables are initialized to zero:

        - DG_AMPA, G_AMPA = 0
        - DG_NMDA, G_NMDA = 0
        - DG_GABA_A, G_GABA_A = 0
        - DG_GABA_B, G_GABA_B = 0

        **4. Intrinsic Gating Variables**

        All gating variables are set to their steady-state values at V_init:

        - m_fast_NMDA = m_slow_NMDA = m_∞^NMDA(V_init)
        - m_Ih = m_∞^Ih(V_init)
        - D_IKNa = D_∞(V_init)
        - m_IT = m_∞^IT(V_init)
        - h_IT = h_∞^IT(V_init)

        At V_init ≈ -70 mV (resting potential):

        - m_Ih ≈ 0.4 (partially activated, since I_h activates at hyperpolarization)
        - m_IT ≈ 0.05 (mostly deactivated)
        - h_IT ≈ 0.9 (mostly deinactivated, ready to support burst)
        - D_IKNa ≈ 0.001 (minimal adaptation at rest)
        - m_NMDA ≈ 0.01 (strongly blocked by Mg²⁺ at rest)

        **5. Refractory Counter**

        ref_steps = 0 (neuron is not refractory at initialization).

        **6. Stimulation Current**

        I_stim = 0 (no external current at t=0).

        **7. Spike Time**

        last_spike_time = -1e7 ms (far in the past, ensures no artificial refractory
        period at simulation start).

        **8. Voltage Clamp Value**

        If voltage_clamp=True, _V_clamp is set to V_init and will be enforced during
        all subsequent updates.

        This initialization matches NEST's ``ht_neuron::State_::set()`` and
        ``calibrate()`` methods, ensuring consistent starting conditions for
        simulation comparisons.
        """
        # Compute initial membrane potential (leak equilibrium)
        V_init = (self.g_NaL * self.E_Na + self.g_KL * self.E_K) / (self.g_NaL + self.g_KL)

        # Build initial state vector
        dftype = brainstate.environ.dftype()
        y0 = np.zeros(_STATE_VEC_SIZE, dtype=dftype)
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
        ditype = brainstate.environ.ditype()
        self.ref_steps = brainstate.ShortTermState(
            np.zeros(self.varshape if batch_size is None else (batch_size, *self.varshape), dtype=ditype)
        )

        # Stimulation current buffer
        self.I_stim = brainstate.ShortTermState(
            np.zeros(self.varshape if batch_size is None else (batch_size, *self.varshape), dtype=dftype)
        )

        # Spike time tracking
        self.last_spike_time = brainstate.ShortTermState(spk_time * u.ms)

        # Voltage clamp value
        self._V_clamp = V_init

    def get_spike(self, V: ArrayLike = None):
        r"""Generate differentiable spike output using surrogate gradient function.

        Converts the discrete spike condition (V >= theta) into a continuous,
        differentiable output suitable for gradient-based optimization. The voltage
        is scaled relative to the dynamic threshold before passing through the
        surrogate function.

        Parameters
        ----------
        V : ArrayLike or None, default=None
            Membrane potential in mV. If None, uses self.V.value. Can be a scalar,
            1D, or multi-dimensional array matching the neuron population shape.
            For explicit spike injection (e.g., after reset), pass a manually set
            value (e.g., large positive for spike, large negative for no spike).

        Returns
        -------
        ArrayLike
            Surrogate spike output with the same shape as V. Values near 1.0 indicate
            spike, values near 0.0 indicate no spike. Gradients flow through the
            surrogate function (e.g., ReluGrad, sigmoid, etc.) rather than being zero.

        Notes
        -----
        **1. Voltage Scaling**

        The input voltage is scaled by the threshold magnitude to normalize the
        surrogate function input:

        .. math::

            v_{scaled} = \frac{V - \theta}{\max(|\theta_{eq}|, 1)}

        This ensures that v_scaled ≈ 0 when V ≈ theta, and v_scaled > 0 when spiking.
        The denominator prevents numerical issues if theta_eq is very small.

        **2. Surrogate Function**

        The scaled voltage is passed through the surrogate gradient function specified
        during initialization (default: braintools.surrogate.ReluGrad()):

        .. math::

            s = \text{spk\_fun}(v_{scaled})

        During forward pass, this typically produces a Heaviside-like step (0 or 1).
        During backward pass, the gradient is replaced by a smooth approximation
        (e.g., d/dv max(0, v) = 1 if v > 0 else 0 for ReluGrad).

        **3. Spike Detection vs. Spike Output**

        This method generates the *output* for surrogate gradient learning. The actual
        spike detection (threshold crossing, reset, refractory logic) happens in the
        ``update()`` method using discrete logic. The two are synchronized: when
        ``update()`` detects a spike, it calls ``get_spike()`` with a manually set
        V value to ensure the output is 1.0.

        **4. Gradient Flow**

        Unlike a true Heaviside function (which has zero gradient everywhere except
        at the discontinuity), the surrogate function provides non-zero gradients in
        a neighborhood around the threshold, enabling backpropagation through spiking
        networks.

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bps
            >>> import saiunit as u
            >>> import brainstate
            >>>
            >>> neuron = bps.ht_neuron(1)
            >>> with brainstate.environ.context(dt=0.1 * u.ms):
            ...     neuron.init_all_states()
            ...
            ...     # Check spike output at rest (V ≈ -70 mV, theta ≈ -51 mV)
            ...     # V < theta, so no spike
            ...     spike = neuron.get_spike()
            ...     print(spike)  # ≈ 0.0
            ...
            ...     # Manually set V above threshold
            ...     neuron.V.value = -45.0  # > theta
            ...     spike = neuron.get_spike()
            ...     print(spike)  # ≈ 1.0 (depends on surrogate function)
        """
        dftype = brainstate.environ.dftype()
        V = np.asarray(self.V.value, dtype=dftype) if V is None else V
        theta = np.asarray(self.theta.value, dtype=dftype)
        # Scale: positive when V >= theta
        v_scaled = (V - theta) / max(abs(self.theta_eq), 1.0)
        return self.spk_fun(jnp.asarray(v_scaled))

    def update(self, x=0.0):
        r"""Advance neuron state by one simulation time step with adaptive ODE integration.

        Performs a complete update cycle for the ht_neuron model, including: (1) adaptive
        RK45 integration of the 16-dimensional ODE system, (2) spike detection and reset,
        (3) refractory period management, (4) synaptic input processing, and (5) external
        current buffering. The update sequence matches NEST's ``ht_neuron::update()``
        implementation for numerical consistency.

        Parameters
        ----------
        x : float or ArrayLike, default=0.0
            External stimulation current in mV/ms (since conductances are unitless in
            this model, currents are expressed as I/C_m). Can be:

            - **Scalar**: Applied uniformly to all neurons in the population.
            - **Array**: Shape must broadcast to the neuron population shape (e.g., for
              spatially varying input or per-neuron stimulation protocols).

            This current is added to the membrane equation as I_stim and affects dV/dt
            during the next integration step.

        Returns
        -------
        ArrayLike
            Differentiable spike output with shape matching the neuron population. Values
            near 1.0 indicate spike, near 0.0 indicate no spike. Compatible with surrogate
            gradient-based learning.

        Notes
        -----
        **1. Update Sequence**

        The update follows this precise ordering (matching NEST):

        **Step 1: ODE Integration**

        Integrate the 16-dimensional state vector from t to t+dt using scipy's RK45
        (Dormand-Prince 5th order adaptive Runge-Kutta). The state vector contains:

        .. math::

            \mathbf{y} = [V, \theta, DG_{AMPA}, G_{AMPA}, DG_{NMDA}, G_{NMDA},
                          DG_{GABA_A}, G_{GABA_A}, DG_{GABA_B}, G_{GABA_B},
                          m_{fast}^{NMDA}, m_{slow}^{NMDA}, m_{Ih}, D_{IKNa},
                          m_{IT}, h_{IT}]

        The ODE right-hand side (defined in the inner ``rhs`` function) computes:

        - Membrane potential derivative from leak, synaptic, intrinsic, and stimulation
          currents, plus post-spike repolarization if refractory
        - Threshold relaxation: dθ/dt = -(θ - θ_eq)/tau_θ
        - Beta-function synaptic conductance dynamics (4 receptor types)
        - NMDA Mg²⁺ unblocking kinetics (fast and slow components)
        - Intrinsic gating variable dynamics (I_h, I_T, I_KNa)

        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        The solver uses adaptive step-size control with rtol=1e-3, atol=1e-9 (matching
        NEST's GSL tolerances).

        **Step 2: Post-Integration Constraints**

        After integration, enforce:

        - **Voltage clamp**: If voltage_clamp=True, reset V to _V_clamp.
        - **Instantaneous NMDA blocking**: Clamp m_fast_NMDA and m_slow_NMDA to not
          exceed m_∞^NMDA(V), ensuring the Mg²⁺ block cannot be "overshot" during
          adaptive time steps.

        **Step 3: Spike Detection and Reset**

        If ``ref_steps == 0`` and ``V >= theta``, a spike is generated:

        - V → E_Na (≈ +30 mV)
        - θ → E_Na
        - ref_steps → ceil(t_ref / dt) + 1
        - spike_flag = True

        **Step 4: Refractory Counter Decrement**

        If ref_steps > 0, decrement by 1. This happens *after* spike detection, so a
        neuron that just spiked will spend t_ref ms refractory.

        **Step 5: Synaptic Spike Input Delivery**

        Add arriving spikes to the DG (derivative of conductance) variables. Inputs are
        retrieved from delta_inputs with labels 'AMPA', 'NMDA', 'GABA_A', 'GABA_B':

        .. math::

            DG_{receptor} \mathrel{+}= g_{peak,receptor} \cdot \text{norm} \cdot w \cdot N_{spikes}

        Unlabeled delta inputs default to AMPA.

        **Step 6: Stimulation Current Buffering**

        Store the input current ``x`` in ``I_stim`` for use in the *next* update cycle.
        This matches NEST's one-step delay for external currents.

        **2. Refractory Dynamics**

        During the refractory period (ref_steps > 0), the neuron cannot spike, and the
        post-spike potassium current is active:

        .. math::

            I_{spike} = -\frac{V - E_K}{\tau_{spike}}

        This drives V toward E_K (hyperpolarization) with time constant tau_spike.

        **3. Synaptic Input Routing**

        The ht_neuron expects delta inputs to be labeled by receptor type. Projections
        should add inputs via:

        .. code-block:: python

            post.add_delta_input(weight * pre_spike, label='AMPA')

        If no label is provided, inputs accumulate in the generic delta_inputs and are
        routed to AMPA by default.

        **4. Numerical Considerations**

        - **Adaptive integration**: The RK45 solver uses variable step sizes to maintain
          accuracy. Typical internal steps are ~0.01-0.1 ms depending on voltage dynamics.
        - **Per-neuron integration**: Each neuron in the population is integrated
          independently to handle heterogeneous parameters or inputs (though this is
          slower than vectorized fixed-step methods).
        - **Intrinsic current caching**: Intrinsic currents (I_NaP, I_KNa, I_T, I_h) are
          computed during integration and stored in separate state variables for recording.

        **5. Performance**

        The per-neuron adaptive integration loop is slower than vectorized Euler methods
        used in simpler models (e.g., LIF, AdEx). For large networks (>1000 neurons),
        consider:

        - Using haiku model for this agent if speed is critical
        - Reducing solver tolerance (rtol=1e-2) for faster but less accurate integration
        - Implementing a vectorized fixed-step version if adaptive steps are not required

        **6. Gradient Compatibility**

        The integration loop uses numpy/scipy (not JAX) for numerical ODE solving, so
        automatic differentiation does *not* flow through the integration. However, the
        surrogate gradient spike output enables backpropagation through the network for
        learning tasks. For fully differentiable ODE solving, consider using JAX-based
        ODE solvers (e.g., diffrax), though this requires significant refactoring.

        Warnings
        --------
        - **Computational cost**: The per-neuron adaptive integration is significantly
          slower than vectorized Euler methods. For large populations, this can dominate
          simulation time.
        - **No JAX gradients through integration**: The scipy ODE solver is not JAX-
          differentiable. Only the spike output (via surrogate gradients) supports
          backpropagation.
        - **Unlabeled inputs default to AMPA**: If you send synaptic inputs without
          specifying a receptor label, they will be routed to AMPA receptors by default.
          This may produce unexpected results if you intended NMDA, GABA_A, or GABA_B.
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
        spk_ampa = np.zeros(v_shape, dtype=dftype)
        spk_nmda = np.zeros(v_shape, dtype=dftype)
        spk_gaba_a = np.zeros(v_shape, dtype=dftype)
        spk_gaba_b = np.zeros(v_shape, dtype=dftype)

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
                arr = np.broadcast_to(np.asarray(val, dtype=dftype), v_shape).copy()
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
                np.asarray(unlabeled, dtype=dftype) if not isinstance(unlabeled, (int, float))
                else np.full(v_shape, unlabeled, dtype=dftype),
                v_shape
            )

        # Collect stimulation current input
        I_stim_next = float(x) if isinstance(x, (int, float)) else np.asarray(x, dtype=dftype)
        I_stim_next = np.broadcast_to(
            np.asarray(I_stim_next, dtype=dftype), v_shape
        ).copy()

        # Extract current state as flat numpy arrays
        V_m = np.asarray(self.V.value, dtype=dftype).ravel()
        theta_val = np.asarray(self.theta.value, dtype=dftype).ravel()
        DG_AMPA = np.asarray(self.DG_AMPA.value, dtype=dftype).ravel()
        G_AMPA = np.asarray(self.G_AMPA.value, dtype=dftype).ravel()
        DG_NMDA = np.asarray(self.DG_NMDA.value, dtype=dftype).ravel()
        G_NMDA = np.asarray(self.G_NMDA.value, dtype=dftype).ravel()
        DG_GABA_A = np.asarray(self.DG_GABA_A.value, dtype=dftype).ravel()
        G_GABA_A = np.asarray(self.G_GABA_A.value, dtype=dftype).ravel()
        DG_GABA_B = np.asarray(self.DG_GABA_B.value, dtype=dftype).ravel()
        G_GABA_B = np.asarray(self.G_GABA_B.value, dtype=dftype).ravel()
        m_fast = np.asarray(self.m_fast_NMDA_state.value, dtype=dftype).ravel()
        m_slow = np.asarray(self.m_slow_NMDA_state.value, dtype=dftype).ravel()
        m_Ih = np.asarray(self.m_Ih_state.value, dtype=dftype).ravel()
        D_IKNa = np.asarray(self.D_IKNa_state.value, dtype=dftype).ravel()
        m_IT = np.asarray(self.m_IT_state.value, dtype=dftype).ravel()
        h_IT = np.asarray(self.h_IT_state.value, dtype=dftype).ravel()
        ref = np.asarray(self.ref_steps.value, dtype=ditype).ravel()
        I_stim_cur = np.asarray(self.I_stim.value, dtype=dftype).ravel()

        # Pre-compute refractory step count
        potassium_refr_counts = self._refractory_counts(h)

        # Output arrays
        V_out = np.empty(flat_size, dtype=dftype)
        theta_out = np.empty(flat_size, dtype=dftype)
        ref_out = np.empty(flat_size, dtype=ditype)
        spike_flags = np.zeros(flat_size, dtype=bool)

        # State output arrays for all 16 variables
        state_out = np.empty((flat_size, _STATE_VEC_SIZE), dtype=dftype)
        I_NaP_out = np.empty(flat_size, dtype=dftype)
        I_KNa_out = np.empty(flat_size, dtype=dftype)
        I_T_out = np.empty(flat_size, dtype=dftype)
        I_h_out = np.empty(flat_size, dtype=dftype)

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
            ], dtype=dftype)

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
                r"""Right-hand side of the ODE system (matches ht_neuron_dynamics)."""
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
        self.ref_steps.value = jnp.asarray(ref_out.reshape(v_shape), dtype=ditype)

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
