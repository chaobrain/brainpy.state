from brainpy_state._nest_base._base import NESTNeuron

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

r"""
Multi-compartment neuron model (``cm_default``), compatible with NEST.

This module implements a compartmental neuron model with a user-defined
dendritic tree structure. Each compartment supports optional Na/K ion
channels and AMPA/GABA/NMDA/AMPA_NMDA synaptic receptors.

The numerical integration uses the Crank-Nicolson scheme with an O(n)
tree-based matrix solver, matching NEST's implementation exactly.
"""

import math
from typing import Optional, Dict, List

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'cm_default',
]


# ---------------------------------------------------------------------------
# Ion channel classes
# ---------------------------------------------------------------------------

class _NaChannel:
    r"""Sodium channel with Hodgkin-Huxley style activation and inactivation kinetics.

    This class implements voltage-dependent sodium channels based on the kinetics from
    ModelDB entry 140828 (Branco 2010), originally from Huguenard et al. (1988) and
    Hamill et al. (1991). The channel uses m³h gating with exact exponential integration
    for state variables.

    Parameters
    ----------
    v_comp : float
        Initial compartment voltage in millivolts, used to initialize gating variables
        at their steady-state values.
    gbar_Na : float, optional
        Maximal sodium conductance in microsiemens (µS). When set to 0.0, the channel
        is inactive and contributes no current. Default: 0.0.
    e_Na : float, optional
        Sodium reversal potential in millivolts. Default: 50.0 mV.

    Attributes
    ----------
    gbar_Na : float
        Maximal conductance.
    e_Na : float
        Reversal potential.
    q10 : float
        Temperature correction factor (1.0 / 3.21), scales time constants.
    m_Na : float
        Activation gating variable (0 to 1).
    h_Na : float
        Inactivation gating variable (0 to 1).

    Notes
    -----
    **1. Gating Kinetics**

    The sodium current is given by:

    .. math::

        I_{Na} = \bar{g}_{Na} \, m^3 \, h \, (e_{Na} - V)

    where m is the activation variable and h is the inactivation variable.

    **2. State Variable Dynamics**

    Both gating variables evolve according to first-order kinetics:

    .. math::

        \frac{dm}{dt} = \frac{m_\infty(V) - m}{\tau_m(V)}

        \frac{dh}{dt} = \frac{h_\infty(V) - h}{\tau_h(V)}

    **3. Voltage-Dependent Rate Functions**

    For activation (m):

    .. math::

        \alpha_m(V) = \frac{0.182 (V + 35.013)}{\exp((V + 35.013)/9) - 1}

        \beta_m(V) = \frac{0.124 (V + 35.013)}{1 - \exp((V + 35.013)/9)}

    For inactivation (h):

    .. math::

        \alpha_h(V) = \frac{0.024 (V + 50.013)}{1 - \exp(-0.2(V + 50.013))}

        \beta_h(V) = \frac{-0.0091 (V + 75.013)}{1 - \exp(0.2(V + 75.013))}

        h_\infty(V) = \frac{1}{1 + \exp((V + 65.0)/6.2)}

    **4. Numerical Integration**

    The model uses exact exponential integration with the Crank-Nicolson scheme,
    returning conductance and current contributions linearized around the midpoint
    voltage for implicit time stepping.

    References
    ----------
    .. [1] Branco T, Clark BA, Häusser M (2010). Dendritic discrimination of temporal
           input sequences in cortical neurons. Science 329(5999):1671-1675.
    .. [2] Huguenard JR, Hamill OP, Prince DA (1988). Developmental changes in Na+
           conductances in rat neocortical neurons. Journal of Neurophysiology
           59(3):778-795.
    .. [3] Hamill OP, Huguenard JR, Prince DA (1991). Patch-clamp studies of
           voltage-gated currents in identified neurons of the rat cerebral cortex.
           Cerebral Cortex 1(1):48-61.
    """

    def __init__(self, v_comp: float, gbar_Na: float = 0.0, e_Na: float = 50.0):
        self.gbar_Na = gbar_Na
        self.e_Na = e_Na
        self.q10 = 1.0 / 3.21
        # State variables
        self.m_Na = 0.0
        self.h_Na = 0.0
        self._init_statevars(v_comp)

    def _init_statevars(self, v_init: float):
        m_inf, _ = self._compute_statevar_m(v_init)
        self.m_Na = m_inf
        h_inf, _ = self._compute_statevar_h(v_init)
        self.h_Na = h_inf

    def _compute_statevar_m(self, v_comp: float):
        v = v_comp + 35.013
        if abs(v) > 1e-5:
            exp_v_div_9 = math.exp(v / 9.0)
            frac = 1.0 / (exp_v_div_9 - 1.0)
            alpha_m = 0.182 * v * exp_v_div_9 * frac
            beta_m = 0.124 * v * frac
            frac_ab = 1.0 / (alpha_m + beta_m)
        else:
            alpha_m = 1.638
            frac_ab = 1.0 / (alpha_m + 1.116)
        tau_m = self.q10 * frac_ab
        m_inf = alpha_m * frac_ab
        return m_inf, tau_m

    def _compute_statevar_h(self, v_comp: float):
        v1 = v_comp + 50.013
        v2 = v_comp + 75.013
        if abs(v1) > 1e-5:
            alpha_h = 0.024 * v1 / (1.0 - math.exp(-0.2 * v1))
        else:
            alpha_h = 0.12
        if abs(v2) > 1e-9:
            beta_h = -0.0091 * v2 / (1.0 - math.exp(0.2 * v2))
        else:
            beta_h = 0.0455
        tau_h = self.q10 / (alpha_h + beta_h)
        h_inf = 1.0 / (1.0 + math.exp((v_comp + 65.0) / 6.2))
        return h_inf, tau_h

    def f_numstep(self, v_comp: float, dt: float):
        r"""Advance channel state by one timestep and compute Crank-Nicolson contributions.

        This method updates the gating variables m_Na and h_Na using exact exponential
        integration, then computes the linearized conductance and current terms needed
        for the Crank-Nicolson implicit time stepping scheme.

        Parameters
        ----------
        v_comp : float
            Current compartment voltage in millivolts.
        dt : float
            Integration timestep in milliseconds.

        Returns
        -------
        g_val : float
            Conductance contribution to diagonal matrix element (µS), computed as
            g_Na / 2 for Crank-Nicolson midpoint.
        i_val : float
            Current contribution to right-hand side (nA), computed as
            g_Na * (e_Na - V/2) for linearization around midpoint voltage.

        Notes
        -----
        When gbar_Na ≤ 1e-9, the channel is considered inactive and returns (0.0, 0.0).

        The exact exponential integration formula for gating variable x is:

        .. math::

            x(t + \Delta t) = x(t) \cdot e^{-\Delta t / \tau_x} +
                              x_\infty \cdot (1 - e^{-\Delta t / \tau_x})
        """
        if self.gbar_Na <= 1e-9:
            return 0.0, 0.0

        m_inf, tau_m = self._compute_statevar_m(v_comp)
        h_inf, tau_h = self._compute_statevar_h(v_comp)

        # Exact exponential integration
        p_m = math.exp(-dt / tau_m)
        self.m_Na = self.m_Na * p_m + (1.0 - p_m) * m_inf

        p_h = math.exp(-dt / tau_h)
        self.h_Na = self.h_Na * p_h + (1.0 - p_h) * h_inf

        g_Na = self.gbar_Na * (self.m_Na ** 3) * self.h_Na

        g_val = g_Na / 2.0
        i_val = g_Na * (self.e_Na - v_comp / 2.0)
        return g_val, i_val


class _KChannel:
    r"""Potassium channel with simplified Hodgkin-Huxley style activation kinetics.

    This class implements voltage-dependent potassium channels based on the kinetics from
    ModelDB entry 140828 (Branco 2010), originally from Sah et al. and Hamill et al. (1991).
    Unlike the classic Hodgkin-Huxley K channel (n⁴), this implementation uses linear
    dependence on the activation variable n.

    Parameters
    ----------
    v_comp : float
        Initial compartment voltage in millivolts, used to initialize the activation
        variable at its steady-state value.
    gbar_K : float, optional
        Maximal potassium conductance in microsiemens (µS). When set to 0.0, the channel
        is inactive and contributes no current. Default: 0.0.
    e_K : float, optional
        Potassium reversal potential in millivolts. Default: -85.0 mV.

    Attributes
    ----------
    gbar_K : float
        Maximal conductance.
    e_K : float
        Reversal potential.
    q10 : float
        Temperature correction factor (1.0 / 3.21), scales time constants.
    n_K : float
        Activation gating variable (0 to 1).

    Notes
    -----
    **1. Gating Kinetics**

    The potassium current is given by:

    .. math::

        I_K = \bar{g}_K \, n \, (e_K - V)

    Note the **linear** dependence on n, not n⁴ as in the original Hodgkin-Huxley model.

    **2. State Variable Dynamics**

    The activation variable evolves according to first-order kinetics:

    .. math::

        \frac{dn}{dt} = \frac{n_\infty(V) - n}{\tau_n(V)}

    **3. Voltage-Dependent Rate Functions**

    .. math::

        \alpha_n(V) = \frac{0.02 (V - 25)}{\exp((V - 25)/9) - 1}

        \beta_n(V) = \frac{0.002 (V - 25)}{1 - \exp((V - 25)/9)}

    The steady-state and time constant are:

    .. math::

        n_\infty = \frac{\alpha_n}{\alpha_n + \beta_n}

        \tau_n = \frac{q_{10}}{\alpha_n + \beta_n}

    **4. Numerical Integration**

    Identical to the sodium channel, this model uses exact exponential integration
    within the Crank-Nicolson framework for implicit time stepping.

    References
    ----------
    .. [1] Branco T, Clark BA, Häusser M (2010). Dendritic discrimination of temporal
           input sequences in cortical neurons. Science 329(5999):1671-1675.
    .. [2] Sah P, Gibb AJ, Gage PW (1988). The sodium current underlying action
           potentials in guinea pig hippocampal CA1 neurons. Journal of General
           Physiology 91(3):373-398.
    .. [3] Hamill OP, Huguenard JR, Prince DA (1991). Patch-clamp studies of
           voltage-gated currents in identified neurons of the rat cerebral cortex.
           Cerebral Cortex 1(1):48-61.
    """

    def __init__(self, v_comp: float, gbar_K: float = 0.0, e_K: float = -85.0):
        self.gbar_K = gbar_K
        self.e_K = e_K
        self.q10 = 1.0 / 3.21
        self.n_K = 0.0
        self._init_statevars(v_comp)

    def _init_statevars(self, v_init: float):
        n_inf, _ = self._compute_statevar_n(v_init)
        self.n_K = n_inf

    def _compute_statevar_n(self, v_comp: float):
        v = v_comp - 25.0
        if abs(v) > 1e-5:
            exp_v_div_9 = math.exp(v / 9.0)
            frac = 1.0 / (exp_v_div_9 - 1.0)
            alpha_n = 0.02 * v * exp_v_div_9 * frac
            beta_n = 0.002 * v * frac
            frac_ab = 1.0 / (alpha_n + beta_n)
        else:
            alpha_n = 0.18
            beta_n = 0.018
            frac_ab = 1.0 / (alpha_n + beta_n)
        tau_n = self.q10 * frac_ab
        n_inf = alpha_n * frac_ab
        return n_inf, tau_n

    def f_numstep(self, v_comp: float, dt: float):
        r"""Advance channel state by one timestep and compute Crank-Nicolson contributions.

        This method updates the activation variable n_K using exact exponential integration,
        then computes the linearized conductance and current terms needed for the
        Crank-Nicolson implicit time stepping scheme.

        Parameters
        ----------
        v_comp : float
            Current compartment voltage in millivolts.
        dt : float
            Integration timestep in milliseconds.

        Returns
        -------
        g_val : float
            Conductance contribution to diagonal matrix element (µS), computed as
            g_K / 2 for Crank-Nicolson midpoint.
        i_val : float
            Current contribution to right-hand side (nA), computed as
            g_K * (e_K - V/2) for linearization around midpoint voltage.

        Notes
        -----
        When gbar_K ≤ 1e-9, the channel is considered inactive and returns (0.0, 0.0).
        """
        if self.gbar_K <= 1e-9:
            return 0.0, 0.0

        n_inf, tau_n = self._compute_statevar_n(v_comp)

        p_n = math.exp(-dt / tau_n)
        self.n_K = self.n_K * p_n + (1.0 - p_n) * n_inf

        g_K = self.gbar_K * self.n_K

        g_val = g_K / 2.0
        i_val = g_K * (self.e_K - v_comp / 2.0)
        return g_val, i_val


# ---------------------------------------------------------------------------
# Synaptic receptor classes
# ---------------------------------------------------------------------------

def _compute_g_norm(tau_r: float, tau_d: float) -> float:
    r"""Compute normalization constant for dual-exponential conductance waveforms.

    The normalization ensures that the peak conductance of the dual-exponential
    waveform equals 1.0 when the synaptic weight is 1.0.

    Parameters
    ----------
    tau_r : float
        Rise time constant in milliseconds, must be less than tau_d.
    tau_d : float
        Decay time constant in milliseconds, must be greater than tau_r.

    Returns
    -------
    float
        Normalization constant that scales the conductance waveform such that
        its peak amplitude is unity.

    Notes
    -----
    The dual-exponential conductance waveform is:

    .. math::

        g(t) = g_{norm} \cdot (e^{-t/\tau_d} - e^{-t/\tau_r})

    The peak occurs at time:

    .. math::

        t_p = \frac{\tau_r \tau_d}{\tau_d - \tau_r} \ln\left(\frac{\tau_d}{\tau_r}\right)

    The normalization factor is computed as:

    .. math::

        g_{norm} = \frac{1}{e^{-t_p/\tau_d} - e^{-t_p/\tau_r}}
    """
    tp = (tau_r * tau_d) / (tau_d - tau_r) * math.log(tau_d / tau_r)
    return 1.0 / (-math.exp(-tp / tau_r) + math.exp(-tp / tau_d))


class _AMPAReceptor:
    r"""AMPA receptor with dual-exponential conductance kinetics for fast excitatory transmission.

    This class models AMPA (α-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid) glutamate
    receptors, which mediate fast excitatory synaptic transmission in the central nervous system.
    The conductance follows a dual-exponential time course with separate rise and decay phases.

    Parameters
    ----------
    e_AMPA : float, optional
        Reversal potential in millivolts. Default: 0.0 mV (excitatory).
    tau_r_AMPA : float, optional
        Rise time constant in milliseconds, controls the speed of conductance onset.
        Default: 0.2 ms (fast activation).
    tau_d_AMPA : float, optional
        Decay time constant in milliseconds, controls the duration of the conductance.
        Default: 3.0 ms (fast deactivation).

    Attributes
    ----------
    e_rev : float
        Reversal potential.
    tau_r : float
        Rise time constant.
    tau_d : float
        Decay time constant.
    g_norm : float
        Normalization constant ensuring peak conductance equals synaptic weight.
    g_r : float
        Rising phase conductance state variable (µS).
    g_d : float
        Decaying phase conductance state variable (µS).
    prop_r : float
        Precomputed exponential decay factor for rise phase (exp(-dt/tau_r)).
    prop_d : float
        Precomputed exponential decay factor for decay phase (exp(-dt/tau_d)).

    Notes
    -----
    **1. Conductance Kinetics**

    The AMPA conductance evolves as:

    .. math::

        g_{AMPA}(t) = g_{norm} \cdot (g_d(t) + g_r(t))

    where g_d represents the decay phase and g_r the rise phase (stored as negative).

    **2. Synaptic Current**

    The total current through AMPA receptors is:

    .. math::

        I_{AMPA} = g_{AMPA} \cdot (e_{AMPA} - V)

    **3. Linearization for Implicit Integration**

    For the Crank-Nicolson scheme, the current is linearized as:

    .. math::

        I = I_{tot} + g_{val} \cdot V

    where g_val = g_AMPA / 2 and i_val incorporates the driving force.

    **4. Spike Processing**

    Each presynaptic spike adds weight × g_norm to both conductance components
    (subtracts from g_r, adds to g_d) to produce the dual-exponential waveform.
    """

    def __init__(self, e_AMPA: float = 0.0, tau_r_AMPA: float = 0.2,
                 tau_d_AMPA: float = 3.0):
        self.e_rev = e_AMPA
        self.tau_r = tau_r_AMPA
        self.tau_d = tau_d_AMPA
        self.g_norm = _compute_g_norm(self.tau_r, self.tau_d)
        self.g_r = 0.0
        self.g_d = 0.0
        self.prop_r = 0.0
        self.prop_d = 0.0

    def pre_run_hook(self, dt: float):
        r"""Precompute time-step-dependent exponential decay factors for efficient simulation.

        Parameters
        ----------
        dt : float
            Integration timestep in milliseconds.

        Notes
        -----
        This method must be called once before simulation starts. It computes and caches
        the exponential decay factors exp(-dt/tau) for both rise and decay phases, avoiding
        repeated exponential function calls during the simulation loop.
        """
        self.prop_r = math.exp(-dt / self.tau_r)
        self.prop_d = math.exp(-dt / self.tau_d)

    def f_numstep(self, v_comp: float, spike_weight: float):
        r"""Advance receptor state by one timestep and compute Crank-Nicolson contributions.

        Updates the dual-exponential conductance state variables and computes the linearized
        conductance and current terms for implicit integration.

        Parameters
        ----------
        v_comp : float
            Current compartment voltage in millivolts.
        spike_weight : float
            Total synaptic weight from presynaptic spikes arriving at this timestep (µS).
            If no spikes arrive, this should be 0.0.

        Returns
        -------
        g_val : float
            Conductance contribution to diagonal matrix element (µS), computed as
            g_AMPA / 2 for Crank-Nicolson midpoint linearization.
        i_val : float
            Current contribution to right-hand side (nA), linearized around midpoint voltage.

        Notes
        -----
        The method performs three steps:
        1. Decay existing conductance states by multiplication with precomputed factors
        2. Add new spike contributions scaled by normalization constant
        3. Compute total conductance and linearized current/conductance for matrix assembly
        """
        self.g_r *= self.prop_r
        self.g_d *= self.prop_d

        s_val = spike_weight * self.g_norm
        self.g_r -= s_val
        self.g_d += s_val

        g_total = self.g_r + self.g_d

        i_tot = g_total * (self.e_rev - v_comp)
        d_i_tot_dv = -g_total

        g_val = -d_i_tot_dv / 2.0
        i_val = i_tot + g_val * v_comp
        return g_val, i_val


class _GABAReceptor:
    r"""GABA receptor with dual-exponential conductance kinetics for inhibitory transmission.

    This class models GABA_A (γ-aminobutyric acid type A) receptors, which mediate fast
    inhibitory synaptic transmission in the central nervous system. The conductance follows
    a dual-exponential time course with separate rise and decay phases.

    Parameters
    ----------
    e_GABA : float, optional
        Reversal potential in millivolts. Default: -80.0 mV (inhibitory, typically
        between -80 mV and -70 mV depending on chloride concentration).
    tau_r_GABA : float, optional
        Rise time constant in milliseconds, controls the speed of conductance onset.
        Default: 0.2 ms (fast activation).
    tau_d_GABA : float, optional
        Decay time constant in milliseconds, controls the duration of inhibition.
        Default: 10.0 ms (slower than AMPA, producing longer-lasting inhibition).

    Attributes
    ----------
    e_rev : float
        Reversal potential.
    tau_r : float
        Rise time constant.
    tau_d : float
        Decay time constant.
    g_norm : float
        Normalization constant ensuring peak conductance equals synaptic weight.
    g_r : float
        Rising phase conductance state variable (µS).
    g_d : float
        Decaying phase conductance state variable (µS).
    prop_r : float
        Precomputed exponential decay factor for rise phase (exp(-dt/tau_r)).
    prop_d : float
        Precomputed exponential decay factor for decay phase (exp(-dt/tau_d)).

    Notes
    -----
    **1. Conductance Kinetics**

    The GABA conductance evolves identically to AMPA receptors:

    .. math::

        g_{GABA}(t) = g_{norm} \cdot (g_d(t) + g_r(t))

    **2. Synaptic Current**

    The inhibitory current through GABA receptors is:

    .. math::

        I_{GABA} = g_{GABA} \cdot (e_{GABA} - V)

    Since e_GABA is typically more negative than the resting potential, this current
    hyperpolarizes the neuron when the conductance is activated.

    **3. Temporal Profile**

    The decay time constant (10 ms default) is slower than AMPA (3 ms), reflecting
    the longer-lasting inhibitory postsynaptic potentials (IPSPs) observed in
    cortical neurons.

    **4. Implementation**

    Computationally identical to _AMPAReceptor but with different default parameters
    reflecting the distinct biophysical properties of GABA_A receptors.
    """

    def __init__(self, e_GABA: float = -80.0, tau_r_GABA: float = 0.2,
                 tau_d_GABA: float = 10.0):
        self.e_rev = e_GABA
        self.tau_r = tau_r_GABA
        self.tau_d = tau_d_GABA
        self.g_norm = _compute_g_norm(self.tau_r, self.tau_d)
        self.g_r = 0.0
        self.g_d = 0.0
        self.prop_r = 0.0
        self.prop_d = 0.0

    def pre_run_hook(self, dt: float):
        r"""Precompute time-step-dependent exponential decay factors for efficient simulation.

        Parameters
        ----------
        dt : float
            Integration timestep in milliseconds.

        Notes
        -----
        This method must be called once before simulation starts. It computes and caches
        the exponential decay factors exp(-dt/tau) for both rise and decay phases.
        """
        self.prop_r = math.exp(-dt / self.tau_r)
        self.prop_d = math.exp(-dt / self.tau_d)

    def f_numstep(self, v_comp: float, spike_weight: float):
        r"""Advance receptor state by one timestep and compute Crank-Nicolson contributions.

        Updates the dual-exponential conductance state variables and computes the linearized
        conductance and current terms for implicit integration.

        Parameters
        ----------
        v_comp : float
            Current compartment voltage in millivolts.
        spike_weight : float
            Total synaptic weight from presynaptic spikes arriving at this timestep (µS).

        Returns
        -------
        g_val : float
            Conductance contribution to diagonal matrix element (µS).
        i_val : float
            Current contribution to right-hand side (nA).

        Notes
        -----
        Identical computational structure to _AMPAReceptor.f_numstep() but acts with
        the inhibitory reversal potential e_GABA.
        """
        self.g_r *= self.prop_r
        self.g_d *= self.prop_d

        s_val = spike_weight * self.g_norm
        self.g_r -= s_val
        self.g_d += s_val

        g_total = self.g_r + self.g_d

        i_tot = g_total * (self.e_rev - v_comp)
        d_i_tot_dv = -g_total

        g_val = -d_i_tot_dv / 2.0
        i_val = i_tot + g_val * v_comp
        return g_val, i_val


def _nmda_sigmoid(v_comp: float):
    r"""Compute voltage-dependent Mg²⁺ block function for NMDA receptors and its derivative.

    This function implements the sigmoidal voltage dependence of magnesium ion block
    in NMDA receptor channels. At hyperpolarized potentials, Mg²⁺ ions block the channel
    pore; depolarization relieves the block, allowing current flow.

    Parameters
    ----------
    v_comp : float
        Compartment voltage in millivolts.

    Returns
    -------
    B : float
        Mg²⁺ unblock factor (0 to 1), representing the fraction of channels not blocked.
        At V = 0 mV, B ≈ 0.77. At V = -80 mV, B ≈ 0.33 (strong block).
    dB_dv : float
        Voltage derivative of the unblock factor (1/mV), needed for linearization in
        the Crank-Nicolson scheme.

    Notes
    -----
    The Mg²⁺ block function is modeled as:

    .. math::

        B(V) = \frac{1}{1 + 0.3 \, e^{-0.1 V}}

    Its voltage derivative is:

    .. math::

        \frac{dB}{dV} = \frac{0.03 \, e^{-0.1 V}}{(1 + 0.3 \, e^{-0.1 V})^2}

    This formulation is a simplified version of the Jahr & Stevens (1990) model,
    commonly used in computational neuroscience for its computational efficiency
    while capturing the essential voltage-dependent properties of NMDA receptors.

    References
    ----------
    .. [1] Jahr CE, Stevens CF (1990). Voltage dependence of NMDA-activated
           macroscopic conductances predicted by single-channel kinetics.
           Journal of Neuroscience 10(9):3178-3182.
    """
    exp_v = math.exp(-0.1 * v_comp)
    denom = 1.0 + 0.3 * exp_v
    B = 1.0 / denom
    dB_dv = 0.03 * exp_v / (denom ** 2)
    return B, dB_dv


class _NMDAReceptor:
    r"""NMDA receptor with dual-exponential kinetics and voltage-dependent Mg²⁺ block.

    This class models NMDA (N-methyl-D-aspartate) glutamate receptors, which mediate
    slow excitatory synaptic transmission with unique voltage-dependent properties.
    NMDA receptors are critical for synaptic plasticity, learning, and memory due to
    their calcium permeability and coincidence detection properties.

    Parameters
    ----------
    e_NMDA : float, optional
        Reversal potential in millivolts. Default: 0.0 mV (excitatory, similar to AMPA).
    tau_r_NMDA : float, optional
        Rise time constant in milliseconds. Default: 0.2 ms.
    tau_d_NMDA : float, optional
        Decay time constant in milliseconds. Default: 43.0 ms (much slower than AMPA,
        producing long-lasting excitatory postsynaptic currents).

    Attributes
    ----------
    e_rev : float
        Reversal potential.
    tau_r : float
        Rise time constant.
    tau_d : float
        Decay time constant.
    g_norm : float
        Normalization constant for peak conductance.
    g_r : float
        Rising phase conductance state variable (µS).
    g_d : float
        Decaying phase conductance state variable (µS).
    prop_r : float
        Precomputed exponential decay factor for rise phase.
    prop_d : float
        Precomputed exponential decay factor for decay phase.

    Notes
    -----
    **1. Voltage-Dependent Conductance**

    Unlike AMPA and GABA receptors, NMDA receptors exhibit voltage-dependent Mg²⁺ block:

    .. math::

        g_{NMDA}(V, t) = \bar{g}_{NMDA}(t) \cdot B(V)

    where B(V) is the Mg²⁺ unblock factor (see _nmda_sigmoid).

    **2. Synaptic Current**

    The NMDA current is:

    .. math::

        I_{NMDA} = g_{NMDA}(V, t) \cdot (e_{NMDA} - V)

    This current is maximal at depolarized potentials (Mg²⁺ block relieved) and when
    glutamate is bound (high conductance).

    **3. Coincidence Detection**

    The voltage dependence makes NMDA receptors act as coincidence detectors: they
    conduct significant current only when (1) glutamate is released presynaptically
    AND (2) the postsynaptic cell is sufficiently depolarized. This property is
    fundamental to Hebbian learning mechanisms.

    **4. Calcium Permeability**

    While not explicitly modeled here (the reversal potential treats all cations equally),
    real NMDA receptors are highly permeable to Ca²⁺, which triggers intracellular
    signaling cascades underlying synaptic plasticity.

    **5. Temporal Profile**

    The slow decay (43 ms default) produces long-lasting excitatory postsynaptic
    currents (EPSCs), enabling temporal summation and integration of inputs over
    100+ milliseconds.

    References
    ----------
    .. [1] Jahr CE, Stevens CF (1990). Voltage dependence of NMDA-activated
           macroscopic conductances predicted by single-channel kinetics.
           Journal of Neuroscience 10(9):3178-3182.
    .. [2] Destexhe A, Mainen ZF, Sejnowski TJ (1998). Kinetic models of
           synaptic transmission. Methods in Neuronal Modeling 2:1-25.
    """

    def __init__(self, e_NMDA: float = 0.0, tau_r_NMDA: float = 0.2,
                 tau_d_NMDA: float = 43.0):
        self.e_rev = e_NMDA
        self.tau_r = tau_r_NMDA
        self.tau_d = tau_d_NMDA
        self.g_norm = _compute_g_norm(self.tau_r, self.tau_d)
        self.g_r = 0.0
        self.g_d = 0.0
        self.prop_r = 0.0
        self.prop_d = 0.0

    def pre_run_hook(self, dt: float):
        r"""Precompute time-step-dependent exponential decay factors for efficient simulation.

        Parameters
        ----------
        dt : float
            Integration timestep in milliseconds.
        """
        self.prop_r = math.exp(-dt / self.tau_r)
        self.prop_d = math.exp(-dt / self.tau_d)

    def f_numstep(self, v_comp: float, spike_weight: float):
        r"""Advance receptor state by one timestep and compute Crank-Nicolson contributions.

        Updates the dual-exponential conductance state variables and computes the linearized
        conductance and current terms, including the voltage-dependent Mg²⁺ block.

        Parameters
        ----------
        v_comp : float
            Current compartment voltage in millivolts.
        spike_weight : float
            Total synaptic weight from presynaptic spikes arriving at this timestep (µS).

        Returns
        -------
        g_val : float
            Conductance contribution to diagonal matrix element (µS), accounting for
            both the Mg²⁺ block B(V) and its voltage derivative dB/dV.
        i_val : float
            Current contribution to right-hand side (nA), linearized for the
            Crank-Nicolson scheme.

        Notes
        -----
        The linearization for the NMDA current includes the voltage dependence of the
        Mg²⁺ block:

        .. math::

            \frac{\partial I_{NMDA}}{\partial V} = g_{NMDA} \left(
                \frac{dB}{dV} (e_{NMDA} - V) - B
            \right)

        This ensures accurate implicit integration despite the nonlinear voltage dependence.
        """
        self.g_r *= self.prop_r
        self.g_d *= self.prop_d

        s_val = spike_weight * self.g_norm
        self.g_r -= s_val
        self.g_d += s_val

        g_total = self.g_r + self.g_d
        B, dB_dv = _nmda_sigmoid(v_comp)

        i_tot = g_total * B * (self.e_rev - v_comp)
        d_i_tot_dv = g_total * (dB_dv * (self.e_rev - v_comp) - B)

        g_val = -d_i_tot_dv / 2.0
        i_val = i_tot + g_val * v_comp
        return g_val, i_val


class _AMPA_NMDAReceptor:
    r"""Combined AMPA and NMDA receptor with shared reversal potential and dual kinetics.

    This class models colocalized AMPA and NMDA receptors at the same synapse, a common
    configuration at excitatory synapses in the brain. A single presynaptic spike activates
    both receptor types simultaneously, producing a fast AMPA component and a slow,
    voltage-dependent NMDA component.

    Parameters
    ----------
    e_AMPA_NMDA : float, optional
        Shared reversal potential in millivolts for both receptor types. Default: 0.0 mV.
    tau_r_AMPA : float, optional
        AMPA rise time constant in milliseconds. Default: 0.2 ms.
    tau_d_AMPA : float, optional
        AMPA decay time constant in milliseconds. Default: 3.0 ms.
    tau_r_NMDA : float, optional
        NMDA rise time constant in milliseconds. Default: 0.2 ms.
    tau_d_NMDA : float, optional
        NMDA decay time constant in milliseconds. Default: 43.0 ms.
    NMDA_ratio : float, optional
        Scaling ratio of NMDA conductance relative to AMPA conductance. A value of 2.0
        means the peak NMDA conductance is twice the peak AMPA conductance for the same
        synaptic weight. Default: 2.0.

    Attributes
    ----------
    e_rev : float
        Shared reversal potential.
    tau_r_AMPA, tau_d_AMPA : float
        AMPA rise and decay time constants.
    tau_r_NMDA, tau_d_NMDA : float
        NMDA rise and decay time constants.
    NMDA_ratio : float
        Conductance ratio parameter.
    g_norm_AMPA, g_norm_NMDA : float
        Normalization constants for each component.
    g_r_AMPA, g_d_AMPA : float
        AMPA conductance state variables (µS).
    g_r_NMDA, g_d_NMDA : float
        NMDA conductance state variables (µS).
    prop_r_AMPA, prop_d_AMPA : float
        Precomputed exponential factors for AMPA.
    prop_r_NMDA, prop_d_NMDA : float
        Precomputed exponential factors for NMDA.

    Notes
    -----
    **1. Dual-Component Conductance**

    The total synaptic conductance is:

    .. math::

        g_{syn} = g_{AMPA}(t) + R_{NMDA} \cdot g_{NMDA}(t) \cdot B(V)

    where R_NMDA is the NMDA_ratio parameter and B(V) is the Mg²⁺ unblock factor.

    **2. Synaptic Current**

    The total current is:

    .. math::

        I_{syn} = g_{syn} \cdot (e_{AMPA\_NMDA} - V)

    **3. Temporal Dynamics**

    The combination produces a biphasic EPSC:
    - Fast initial peak dominated by AMPA (peaks around 1-2 ms)
    - Slow tail dominated by NMDA (decays over 50-100 ms)

    **4. Functional Significance**

    This receptor configuration enables both fast signal transmission (AMPA) and
    slow integration for coincidence detection and plasticity induction (NMDA).
    The NMDA_ratio parameter controls the relative contribution of each component
    and varies across brain regions and developmental stages.

    **5. Computational Efficiency**

    Although more complex than single-receptor models, this combined implementation
    is more efficient than maintaining separate AMPA and NMDA receptor objects when
    they are always coactivated at the same synapse.

    References
    ----------
    .. [1] Hestrin S, Nicoll RA, Perkel DJ, Sah P (1990). Analysis of excitatory
           synaptic action in pyramidal cells using whole-cell recording from
           rat hippocampal slices. Journal of Physiology 422:203-225.
    .. [2] Destexhe A, Mainen ZF, Sejnowski TJ (1998). Kinetic models of
           synaptic transmission. Methods in Neuronal Modeling 2:1-25.
    """

    def __init__(self, e_AMPA_NMDA: float = 0.0,
                 tau_r_AMPA: float = 0.2, tau_d_AMPA: float = 3.0,
                 tau_r_NMDA: float = 0.2, tau_d_NMDA: float = 43.0,
                 NMDA_ratio: float = 2.0):
        self.e_rev = e_AMPA_NMDA
        self.tau_r_AMPA = tau_r_AMPA
        self.tau_d_AMPA = tau_d_AMPA
        self.tau_r_NMDA = tau_r_NMDA
        self.tau_d_NMDA = tau_d_NMDA
        self.NMDA_ratio = NMDA_ratio

        self.g_norm_AMPA = _compute_g_norm(self.tau_r_AMPA, self.tau_d_AMPA)
        self.g_norm_NMDA = _compute_g_norm(self.tau_r_NMDA, self.tau_d_NMDA)

        self.g_r_AMPA = 0.0
        self.g_d_AMPA = 0.0
        self.g_r_NMDA = 0.0
        self.g_d_NMDA = 0.0

        self.prop_r_AMPA = 0.0
        self.prop_d_AMPA = 0.0
        self.prop_r_NMDA = 0.0
        self.prop_d_NMDA = 0.0

    def pre_run_hook(self, dt: float):
        r"""Precompute time-step-dependent exponential decay factors for both receptor types.

        Parameters
        ----------
        dt : float
            Integration timestep in milliseconds.

        Notes
        -----
        Computes and caches four exponential factors (two for AMPA, two for NMDA) to
        avoid repeated exponential function calls during simulation.
        """
        self.prop_r_AMPA = math.exp(-dt / self.tau_r_AMPA)
        self.prop_d_AMPA = math.exp(-dt / self.tau_d_AMPA)
        self.prop_r_NMDA = math.exp(-dt / self.tau_r_NMDA)
        self.prop_d_NMDA = math.exp(-dt / self.tau_d_NMDA)

    def f_numstep(self, v_comp: float, spike_weight: float):
        r"""Advance receptor state by one timestep and compute Crank-Nicolson contributions.

        Updates both AMPA and NMDA conductance state variables and computes the linearized
        conductance and current terms for the combined receptor complex.

        Parameters
        ----------
        v_comp : float
            Current compartment voltage in millivolts.
        spike_weight : float
            Total synaptic weight from presynaptic spikes arriving at this timestep (µS).
            This weight is distributed to both AMPA and NMDA components according to
            their respective normalization constants and the NMDA_ratio parameter.

        Returns
        -------
        g_val : float
            Combined conductance contribution to diagonal matrix element (µS), accounting
            for both AMPA (voltage-independent) and NMDA (voltage-dependent with Mg²⁺ block)
            components.
        i_val : float
            Combined current contribution to right-hand side (nA), linearized for the
            Crank-Nicolson scheme.

        Notes
        -----
        The linearization accounts for the different voltage dependencies:

        - AMPA component: d(g_AMPA * (E - V))/dV = -g_AMPA
        - NMDA component: d(g_NMDA * B(V) * (E - V))/dV includes both dB/dV and -B terms

        The total derivative is:

        .. math::

            \frac{\partial I}{\partial V} = -g_{AMPA} +
                R_{NMDA} \cdot g_{NMDA} \left(
                    \frac{dB}{dV} (e - V) - B
                \right)

        where the NMDA term has opposite sign due to the product rule and voltage dependence.
        """
        self.g_r_AMPA *= self.prop_r_AMPA
        self.g_d_AMPA *= self.prop_d_AMPA
        self.g_r_NMDA *= self.prop_r_NMDA
        self.g_d_NMDA *= self.prop_d_NMDA

        s_val = spike_weight * self.g_norm_AMPA
        self.g_r_AMPA -= s_val
        self.g_d_AMPA += s_val

        s_val = spike_weight * self.g_norm_NMDA
        self.g_r_NMDA -= s_val
        self.g_d_NMDA += s_val

        g_AMPA = self.g_r_AMPA + self.g_d_AMPA
        g_NMDA = self.g_r_NMDA + self.g_d_NMDA
        B, dB_dv = _nmda_sigmoid(v_comp)

        i_tot = (g_AMPA + self.NMDA_ratio * g_NMDA * B) * (self.e_rev - v_comp)
        d_i_tot_dv = -g_AMPA + self.NMDA_ratio * g_NMDA * (
            dB_dv * (self.e_rev - v_comp) - B
        )

        g_val = -d_i_tot_dv / 2.0
        i_val = i_tot + g_val * v_comp
        return g_val, i_val


# ---------------------------------------------------------------------------
# Compartment class
# ---------------------------------------------------------------------------

class _Compartment:
    r"""A single compartment in the multi-compartment neuron's dendritic tree structure.

    This class represents one cylindrical segment of dendrite, soma, or axon in a
    morphologically realistic neuron model. Compartments are connected in a tree
    structure, with each compartment (except the root) having exactly one parent and
    zero or more children. The compartment implements the cable equation with optional
    active conductances (Na/K channels) and synaptic receptors.

    Parameters
    ----------
    comp_index : int
        Unique identifier for this compartment within the neuron, assigned sequentially
        starting from 0 (typically soma is index 0).
    parent_index : int
        Index of the parent compartment in the tree. Use -1 for the root compartment
        (soma), which has no parent.
    params : dict, optional
        Dictionary of electrical and biophysical parameters. If None, default values
        are used. Supported keys:

        - 'C_m' : Membrane capacitance in nanofarads (default: 1.0 nF)
        - 'g_C' : Axial coupling conductance to parent in microsiemens (default: 0.01 µS)
        - 'g_L' : Leak conductance in microsiemens (default: 0.1 µS)
        - 'e_L' : Leak reversal potential in millivolts (default: -70.0 mV)
        - 'v_comp' : Initial membrane voltage in millivolts (default: e_L)
        - 'gbar_Na' : Sodium channel maximal conductance in microsiemens (default: 0.0 µS, inactive)
        - 'e_Na' : Sodium reversal potential in millivolts (default: 50.0 mV)
        - 'gbar_K' : Potassium channel maximal conductance in microsiemens (default: 0.0 µS, inactive)
        - 'e_K' : Potassium reversal potential in millivolts (default: -85.0 mV)

    Attributes
    ----------
    comp_index : int
        Compartment identifier.
    p_index : int
        Parent compartment index.
    parent : _Compartment or None
        Reference to parent compartment object.
    children : list of _Compartment
        List of child compartment objects.
    ca, gc, gl, el : float
        Electrical parameters (capacitance, coupling, leak conductance, leak reversal).
    v_comp : float
        Current membrane voltage in millivolts.
    na_chan : _NaChannel
        Sodium channel object.
    k_chan : _KChannel
        Potassium channel object.
    receptors : list
        List of synaptic receptor objects attached to this compartment.
    ff, gg, hh : float
        Crank-Nicolson matrix elements computed during each timestep.
    _current_buffer : list of float
        Queue of external current injections (nA) for upcoming timesteps.

    Notes
    -----
    **1. Cable Equation**

    Each compartment obeys the cable equation:

    .. math::

        C_m \frac{dV}{dt} = -g_L (V - e_L) - g_C (V - V_{parent})
                             - \sum_{j \in \text{children}} g_C^{(j)} (V - V_j)
                             + I_{Na} + I_K + I_{syn} + I_{ext}

    **2. Tree Structure**

    Compartments form a directed tree with information flow:
    - Coupling conductances g_C are properties of child compartments
    - During matrix construction, each compartment sees its parent and all children
    - Root compartment (soma) has no parent coupling term

    **3. Crank-Nicolson Matrix Elements**

    The implicit time stepping produces a system of equations:

    .. math::

        gg_i \cdot V_i^{new} + hh_i \cdot V_{parent}^{new} + \sum_j terms_j = ff_i

    where:

    - gg: diagonal element (self-conductance + coupling conductances + channel conductances)
    - hh: off-diagonal coupling to parent
    - ff: right-hand side (incorporates old voltages and driving currents)

    **4. Spike Detection**

    Only the root compartment (soma) is monitored for spike detection. Dendritic
    compartments can generate action potentials if Na/K channels are activated, but
    these are not counted as neuron output spikes.

    **5. Receptor Distribution**

    Synaptic receptors can be placed on any compartment, enabling spatially distributed
    synaptic input patterns. Multiple receptors of different types can coexist on the
    same compartment.
    """

    def __init__(self, comp_index: int, parent_index: int, params: Optional[Dict] = None):
        self.comp_index = comp_index
        self.p_index = parent_index
        self.parent: Optional['_Compartment'] = None
        self.children: List['_Compartment'] = []

        # Electrical parameters (defaults match NEST)
        self.ca = 1.0  # C_m [nF]
        self.gc = 0.01  # g_C [uS]
        self.gl = 0.1  # g_L [uS]
        self.el = -70.0  # e_L [mV]
        self.v_comp = self.el  # voltage [mV]

        # Ion channel parameters
        gbar_Na = 0.0
        e_Na = 50.0
        gbar_K = 0.0
        e_K = -85.0

        if params is not None:
            self.ca = params.get('C_m', self.ca)
            self.gc = params.get('g_C', self.gc)
            self.gl = params.get('g_L', self.gl)
            self.el = params.get('e_L', self.el)
            self.v_comp = params.get('v_comp', self.el)
            gbar_Na = params.get('gbar_Na', gbar_Na)
            e_Na = params.get('e_Na', e_Na)
            gbar_K = params.get('gbar_K', gbar_K)
            e_K = params.get('e_K', e_K)

        # Ion channels
        self.na_chan = _NaChannel(self.v_comp, gbar_Na, e_Na)
        self.k_chan = _KChannel(self.v_comp, gbar_K, e_K)

        # Synaptic receptors (list of (receptor, spike_buffer_index) tuples)
        self.receptors: List = []

        # Pre-computed constants (set in pre_run_hook)
        self.ca__div__dt = 0.0
        self.gl__div__2 = 0.0
        self.gg0 = 0.0
        self.gc__div__2 = 0.0
        self.gl__times__el = 0.0

        # Matrix elements
        self.ff = 0.0
        self.gg = 0.0
        self.hh = 0.0

        # Aggregators for tree solver
        self._xx = 0.0
        self._yy = 0.0
        self.n_passed = 0

        # External current buffer (list indexed by lag)
        self._current_buffer: List[float] = []

    def add_receptor(self, receptor):
        r"""Add a synaptic receptor to this compartment.

        Parameters
        ----------
        receptor : _AMPAReceptor or _GABAReceptor or _NMDAReceptor or _AMPA_NMDAReceptor
            Receptor object to attach to this compartment. Multiple receptors can be
            added to the same compartment for multi-receptor synaptic input.

        Notes
        -----
        Receptors are stored in a list and processed during each simulation timestep.
        Their conductance contributions are added to the matrix elements during the
        construct_matrix_element() call.
        """
        self.receptors.append(receptor)

    def pre_run_hook(self, dt: float):
        r"""Precompute time-step-dependent constants for efficient numerical integration.

        Parameters
        ----------
        dt : float
            Integration timestep in milliseconds.

        Notes
        -----
        This method computes and caches several constants used repeatedly during
        the simulation loop:

        - ca__div__dt: C_m / dt (capacitance-timestep factor)
        - gl__div__2: g_L / 2 (half leak conductance for Crank-Nicolson)
        - gg0: baseline diagonal element = C_m/dt + g_L/2
        - gc__div__2: g_C / 2 (half coupling conductance)
        - gl__times__el: g_L * e_L (leak current constant)

        Also calls pre_run_hook() on all attached receptors to initialize their
        time-step-dependent exponential factors.
        """
        self.ca__div__dt = self.ca / dt
        self.gl__div__2 = self.gl / 2.0
        self.gg0 = self.ca__div__dt + self.gl__div__2
        self.gc__div__2 = self.gc / 2.0
        self.gl__times__el = self.gl * self.el

        for rec in self.receptors:
            rec.pre_run_hook(dt)

    def construct_matrix_element(self, dt: float, spike_buffers: Dict[int, float]):
        r"""Build Crank-Nicolson matrix row elements for this compartment.

        This method constructs the implicit equation for this compartment's voltage
        update, incorporating passive membrane properties, coupling to parent and children,
        active ion channels, synaptic receptors, and external current inputs.

        Parameters
        ----------
        dt : float
            Integration timestep in milliseconds.
        spike_buffers : dict
            Dictionary mapping receptor object IDs to accumulated synaptic weights
            (in µS) for this timestep. Receptors with no incoming spikes have weight 0.0.

        Notes
        -----
        **Matrix Structure**

        For the implicit Crank-Nicolson scheme, the equation for this compartment is:

        .. math::

            gg \cdot V^{new} - hh \cdot V_{parent}^{new}
            - \sum_{j \in \text{children}} hh_j \cdot V_j^{new} = ff

        where:
        - gg (diagonal): total conductance seen by this compartment
        - hh (parent coupling): -g_C/2
        - ff (RHS): incorporates old voltages and driving currents

        **Computation Sequence**

        1. Initialize diagonal (gg) with baseline capacitance and leak terms
        2. Add coupling conductance contributions from parent and children
        3. Incorporate passive membrane properties from previous timestep
        4. Add ion channel contributions (Na, K) via their f_numstep() methods
        5. Add synaptic receptor contributions via their f_numstep() methods
        6. Add external current injection if present in buffer

        **Linearization**

        All voltage-dependent terms (ion channels, NMDA receptors) are linearized
        around the current voltage using their derivatives, enabling stable implicit
        integration despite nonlinearities.
        """
        # Diagonal element
        self.gg = self.gg0

        if self.parent is not None:
            self.gg += self.gc__div__2
            self.hh = -self.gc__div__2

        for child in self.children:
            self.gg += child.gc__div__2

        # Right-hand side
        self.ff = (self.ca__div__dt - self.gl__div__2) * self.v_comp + self.gl__times__el

        if self.parent is not None:
            self.ff -= self.gc__div__2 * (self.v_comp - self.parent.v_comp)

        for child in self.children:
            self.ff -= child.gc__div__2 * (self.v_comp - child.v_comp)

        # Ion channel contributions
        g_val, i_val = self.na_chan.f_numstep(self.v_comp, dt)
        self.gg += g_val
        self.ff += i_val

        g_val, i_val = self.k_chan.f_numstep(self.v_comp, dt)
        self.gg += g_val
        self.ff += i_val

        # Receptor contributions
        for rec in self.receptors:
            sw = spike_buffers.get(id(rec), 0.0)
            g_val, i_val = rec.f_numstep(self.v_comp, sw)
            self.gg += g_val
            self.ff += i_val

        # External input current
        if self._current_buffer:
            self.ff += self._current_buffer.pop(0)

    def gather_input(self, g_val: float, f_val: float):
        r"""Accumulate input from a child compartment during down-sweep."""
        self._xx += g_val
        self._yy += f_val

    def io(self):
        r"""Compute input-output transformation for down-sweep.

        Returns (g_val, f_val) to pass to parent.
        """
        self.gg -= self._xx
        self.ff -= self._yy

        g_val = self.hh * self.hh / self.gg
        f_val = self.ff * self.hh / self.gg
        return g_val, f_val

    def calc_v(self, v_in: float) -> float:
        r"""Compute new voltage during up-sweep.

        Parameters
        ----------
        v_in : float
            Parent's new voltage (0 for root).

        Returns
        -------
        float
            New compartment voltage.
        """
        self._xx = 0.0
        self._yy = 0.0
        self.v_comp = (self.ff - v_in * self.hh) / self.gg
        return self.v_comp


# ---------------------------------------------------------------------------
# Vectorised (JAX) channel/receptor kinetics for the State-based update() path
# ---------------------------------------------------------------------------
#
# These functions are exact, array-valued replicas of the per-compartment host
# formulas above (``_NaChannel._compute_statevar_*``, ``_KChannel._compute_statevar_n``,
# ``_nmda_sigmoid``).  The host guards the alpha/beta singularities with a Python
# ``if abs(v) > eps`` branch; under JAX we evaluate the general branch with a
# *safe* denominator and select with :func:`jnp.where` (the double-``where`` trick),
# which avoids a 0/0 NaN polluting the gradient while reproducing the host's
# analytic limit at the singular point.  ``Q10 = 1/3.21`` matches the host.

_Q10 = 1.0 / 3.21


def _na_gates_vec(v):
    r"""Vectorised Na gating steady-states/time-constants ``(m_inf, tau_m, h_inf, tau_h)``."""
    # --- activation m (singularity at v + 35.013 == 0) ---
    vm = v + 35.013
    safe_m = jnp.abs(vm) > 1e-5
    e_m = jnp.exp(jnp.where(safe_m, vm, 0.0) / 9.0)
    denom_m = jnp.where(safe_m, e_m - 1.0, 1.0)
    frac_m = 1.0 / denom_m
    alpha_m = jnp.where(safe_m, 0.182 * vm * e_m * frac_m, 1.638)
    beta_m = jnp.where(safe_m, 0.124 * vm * frac_m, 1.116)
    frac_ab_m = 1.0 / (alpha_m + beta_m)
    tau_m = _Q10 * frac_ab_m
    m_inf = alpha_m * frac_ab_m

    # --- inactivation h (two singularities) ---
    v1 = v + 50.013
    v2 = v + 75.013
    safe_h1 = jnp.abs(v1) > 1e-5
    alpha_h = jnp.where(
        safe_h1,
        0.024 * v1 / (1.0 - jnp.exp(-0.2 * jnp.where(safe_h1, v1, 0.0))),
        0.12,
    )
    safe_h2 = jnp.abs(v2) > 1e-9
    beta_h = jnp.where(
        safe_h2,
        -0.0091 * v2 / (1.0 - jnp.exp(0.2 * jnp.where(safe_h2, v2, 0.0))),
        0.0455,
    )
    tau_h = _Q10 / (alpha_h + beta_h)
    h_inf = 1.0 / (1.0 + jnp.exp((v + 65.0) / 6.2))
    return m_inf, tau_m, h_inf, tau_h


def _k_gate_vec(v):
    r"""Vectorised K activation steady-state/time-constant ``(n_inf, tau_n)``."""
    vn = v - 25.0
    safe = jnp.abs(vn) > 1e-5
    e_n = jnp.exp(jnp.where(safe, vn, 0.0) / 9.0)
    denom = jnp.where(safe, e_n - 1.0, 1.0)
    frac = 1.0 / denom
    alpha_n = jnp.where(safe, 0.02 * vn * e_n * frac, 0.18)
    beta_n = jnp.where(safe, 0.002 * vn * frac, 0.018)
    frac_ab = 1.0 / (alpha_n + beta_n)
    return alpha_n * frac_ab, _Q10 * frac_ab


def _nmda_sigmoid_vec(v):
    r"""Vectorised NMDA Mg-block factor and derivative ``(B, dB/dV)``."""
    exp_v = jnp.exp(-0.1 * v)
    denom = 1.0 + 0.3 * exp_v
    return 1.0 / denom, 0.03 * exp_v / (denom ** 2)


def _to_nS(w):
    r"""Strip a synaptic weight to a plain ``nS`` magnitude (Quantity or float)."""
    return w.to_decimal(u.nS) if isinstance(w, u.Quantity) else w


def _to_pA(c):
    r"""Strip an injected current to a plain ``pA`` magnitude (Quantity or float)."""
    return c.to_decimal(u.pA) if isinstance(c, u.Quantity) else c


def _diag_embed(d):
    r"""Embed a ``(*batch, n)`` vector as the diagonal of a ``(*batch, n, n)`` matrix."""
    eye = jnp.eye(d.shape[-1], dtype=d.dtype)
    return d[..., None] * eye


# ---------------------------------------------------------------------------
# Main cm_default class
# ---------------------------------------------------------------------------

class cm_default(NESTNeuron):
    r"""Multi-compartment neuron model with user-defined morphological structure and flexible synapse placement.

    This class implements a compartmental neuron model with arbitrary dendritic tree
    topology, combining cable theory, optional Hodgkin-Huxley style active conductances,
    and multiple synaptic receptor types. The model faithfully replicates the NEST
    simulator's ``cm_default`` implementation, using the Crank-Nicolson implicit
    integration scheme with an O(n) tree-based matrix solver.

    Parameters
    ----------
    V_th : float, optional
        Spike detection threshold in millivolts, applied to the root (soma) compartment.
        A spike is registered when the soma voltage crosses V_th from below.
        Default: -55.0 mV.

    Attributes
    ----------
    V_th : float
        Spike threshold.
    _root : _Compartment or None
        Root compartment (typically soma), serves as spike detection site.
    _compartments : list of _Compartment
        Flat list of all compartments in index order.
    _receptors : list
        All receptor objects across all compartments.
    _spike_buffer : dict
        Accumulator for incoming spikes, maps receptor ID to total weight for current timestep.
    _dt : float or None
        Integration timestep, set by pre_run_hook().
    _t : float
        Current simulation time in milliseconds.
    _step_count : int
        Number of timesteps executed.

    Parameter Mapping
    -----------------

    **Compartment Parameters** (passed to ``add_compartment``):

    =========== ======= ====================================================
    Parameter   Unit    Description
    =========== ======= ====================================================
    ``C_m``     nF      Membrane capacitance (default: 1.0)
    ``g_C``     µS      Axial coupling conductance to parent (default: 0.01)
    ``g_L``     µS      Passive leak conductance (default: 0.1)
    ``e_L``     mV      Leak reversal potential (default: -70.0)
    ``v_comp``  mV      Initial membrane voltage (default: e_L)
    ``gbar_Na`` µS      Na channel maximal conductance (default: 0.0, inactive)
    ``e_Na``    mV      Na reversal potential (default: 50.0)
    ``gbar_K``  µS      K channel maximal conductance (default: 0.0, inactive)
    ``e_K``     mV      K reversal potential (default: -85.0)
    =========== ======= ====================================================

    **Receptor Parameters** (passed to ``add_receptor``):

    *AMPA receptor:*
        - ``e_AMPA``: Reversal potential (default: 0.0 mV)
        - ``tau_r_AMPA``: Rise time (default: 0.2 ms)
        - ``tau_d_AMPA``: Decay time (default: 3.0 ms)

    *GABA receptor:*
        - ``e_GABA``: Reversal potential (default: -80.0 mV)
        - ``tau_r_GABA``: Rise time (default: 0.2 ms)
        - ``tau_d_GABA``: Decay time (default: 10.0 ms)

    *NMDA receptor:*
        - ``e_NMDA``: Reversal potential (default: 0.0 mV)
        - ``tau_r_NMDA``: Rise time (default: 0.2 ms)
        - ``tau_d_NMDA``: Decay time (default: 43.0 ms)

    *AMPA_NMDA combined receptor:*
        - ``e_AMPA_NMDA``: Shared reversal potential (default: 0.0 mV)
        - ``tau_r_AMPA``, ``tau_d_AMPA``: AMPA rise/decay times (default: 0.2, 3.0 ms)
        - ``tau_r_NMDA``, ``tau_d_NMDA``: NMDA rise/decay times (default: 0.2, 43.0 ms)
        - ``NMDA_ratio``: NMDA-to-AMPA conductance ratio (default: 2.0)
    
    Raises
    ------
    ValueError
        - If pre_run_hook() is called before any compartments have been added
        - If add_compartment() attempts to add a second root (parent_idx=-1)
        - If add_compartment() or add_receptor() reference a non-existent parent or compartment
        - If add_receptor() is called with an unrecognized receptor_type

    See Also
    --------
    hh_psc_alpha : Single-compartment Hodgkin-Huxley neuron with alpha-function PSCs
    iaf_cond_alpha : Single-compartment conductance-based integrate-and-fire neuron
    iaf_cond_exp : Conductance-based LIF with exponential PSCs

    Notes
    -----
    **1. Model Overview**

    ``cm_default`` is a flexible compartmental neuron model where the morphological
    structure (soma, dendrites, axon) is constructed programmatically at runtime by
    adding compartments. Each compartment can host multiple synaptic receptors
    (AMPA, GABA, NMDA, or combined AMPA_NMDA) for spatially distributed synaptic input.

    This implementation replicates the NEST simulator's ``cm_default`` model, preserving:
    - Crank-Nicolson implicit integration for numerical stability
    - O(n) tree-based matrix solver (two-pass down-sweep/up-sweep algorithm)
    - Exact exponential integration for ion channel gating variables
    - Dual-exponential conductance kinetics for all synaptic receptors
    - Spike detection via threshold crossing at the soma (root compartment)

    The model is **passive by default**. Active conductances are enabled by setting
    non-zero ``gbar_Na`` and/or ``gbar_K`` in compartment parameters.

    **2. Cable Equation**

    Each compartment :math:`i` obeys the cable equation:

    .. math::

        C_m^{(i)} \frac{dV^{(i)}}{dt} =
            -g_L^{(i)} (V^{(i)} - e_L^{(i)})
            - g_C^{(i)} (V^{(i)} - V^{(\text{parent})})
            - \sum_{j \in \text{children}} g_C^{(j)} (V^{(i)} - V^{(j)})
            + I_{\text{Na}}^{(i)} + I_{\text{K}}^{(i)}
            + I_{\text{syn}}^{(i)} + I_{\text{ext}}^{(i)}

    where:

    - First term: passive leak current
    - Second term: axial current to parent (absent for root)
    - Third term: axial currents from all children
    - Remaining terms: active conductances, synaptic input, and external current

    **3. Crank-Nicolson Discretization**

    The cable equation is integrated using the implicit trapezoidal (Crank-Nicolson) rule:

    .. math::

        \frac{V^{(i)}_{\text{new}} - V^{(i)}_{\text{old}}}{\Delta t}
        = \frac{1}{2} \left( F(V_{\text{new}}) + F(V_{\text{old}}) \right)

    This second-order accurate scheme is unconditionally stable, enabling larger
    timesteps than explicit methods. The resulting linear system has a tridiagonal-like
    structure on the tree topology, solved in O(n) time via:

    1. **Down-sweep** (leaves to root): Compute input-output relations for each subtree
    2. **Up-sweep** (root to leaves): Back-substitute to compute all voltages

    **4. Spike Detection and Reset**

    A spike is registered when the root (soma) compartment voltage crosses V_th from below:

    .. math::

        V_{\text{soma}}^{\text{old}} < V_{\text{th}} \leq V_{\text{soma}}^{\text{new}}

    **No explicit voltage reset is performed.** Repolarization occurs naturally through
    the activation of K channels and inactivation of Na channels (if active conductances
    are present), or through passive decay (if only passive membrane properties are used).

    **5. Ion Channel Models**

    *Sodium channel* (Hodgkin-Huxley m³h kinetics):

    .. math::

        I_{\text{Na}} = \bar{g}_{\text{Na}} \, m^3 h \, (e_{\text{Na}} - V)

    *Potassium channel* (simplified linear n kinetics, not n⁴):

    .. math::

        I_{\text{K}} = \bar{g}_{\text{K}} \, n \, (e_{\text{K}} - V)

    Gating variables are integrated exactly using exponential integration:

    .. math::

        x(t + \Delta t) = x(t) e^{-\Delta t / \tau_x} + x_\infty (1 - e^{-\Delta t / \tau_x})

    **6. Synaptic Receptor Models**

    All receptors use dual-exponential conductance kinetics:

    .. math::

        g_{\text{syn}}(t) = g_{\text{norm}} (e^{-t/\tau_d} - e^{-t/\tau_r})

    where g_norm ensures peak conductance equals the synaptic weight. NMDA receptors
    include voltage-dependent Mg²⁺ block:

    .. math::

        B(V) = \frac{1}{1 + 0.3 \, e^{-0.1 V}}

    so the NMDA current is:

    .. math::

        I_{\text{NMDA}} = g_{\text{NMDA}} \, B(V) \, (e_{\text{NMDA}} - V)

    **7. Unit Conventions and Consistency**

    The model uses the following unit system (matching NEST conventions):
    - Voltages: millivolts (mV)
    - Time: milliseconds (ms)
    - Conductances: microsiemens (µS)
    - Capacitances: nanofarads (nF)
    - Currents: nanoamperes (nA)
    - Synaptic weights: microsiemens (µS)

    **Consistency requirement:** If conductances are in µS, then capacitances MUST be
    in nF to maintain C_m/dt units compatible with conductance units. The model does
    not enforce unit checking; incorrect units will produce nonsensical results.

    **8. Computational Performance**

    The O(n) matrix solver scales linearly with compartment count, making this model
    efficient even for morphologically detailed neurons with hundreds of compartments.
    However, the Crank-Nicolson implicit scheme requires solving a linear system every
    timestep, which is slower than explicit methods for very simple morphologies.
    
    References
    ----------
    
    .. [1] Wybo WAM, Jordan J, Ellenberger B, Mengual UM, Nevian T, Senn W
           (2021). Data-driven reduction of dendritic morphologies with
           preserved dendro-somatic responses. eLife 10:e60936.
           https://doi.org/10.7554/eLife.60936
    .. [2] Branco T, Clark BA, Häusser M (2010). Dendritic discrimination
           of temporal input sequences in cortical neurons. Science
           329(5999):1671-1675.
    
    Examples
    --------
    
    .. code-block:: python
    
       >>> model = cm_default()
       >>> model.add_compartment(-1, {'C_m': 89.245, 'g_L': 8.925, 'e_L': -75.0,
       ...                            'gbar_Na': 4608.7, 'e_Na': 60.0,
       ...                            'gbar_K': 956.1, 'e_K': -90.0})
       >>> model.add_compartment(0, {'C_m': 1.93, 'g_C': 1.255, 'g_L': 0.193,
       ...                           'e_L': -75.0})
       >>> r_idx = model.add_receptor(1, 'AMPA', {'e_AMPA': 0.0, 'tau_d_AMPA': 3.0})
       >>> model.pre_run_hook(dt=0.1)
       >>> # Inject a spike at receptor r_idx
       >>> model.add_spike(r_idx, weight=5.0)
       >>> # Step the simulation
       >>> spike = model.step()
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size=1,
        V_th: float = -55.0,
        *,
        compartments: Optional[List[Dict]] = None,
        receptors: Optional[List[Dict]] = None,
        name: Optional[str] = None,
        spk_fun=None,
        spk_reset: str = 'hard',
    ):
        # Initialise the brainstate ``Neuron`` base so the model is a first-class
        # State module (``varshape``, delta/current input registries, spike fn).
        # The legacy host-loop API never did this; supplying it is what lets the
        # Simulator drive ``update()`` through a compiled ``for_loop``.
        super().__init__(
            in_size,
            name=name,
            spk_fun=braintools.surrogate.ReluGrad() if spk_fun is None else spk_fun,
            spk_reset=spk_reset,
        )
        self.V_th = float(V_th)

        # --- Host-loop tree structure (also the source of truth from which the
        #     State topology is built when a morphology is supplied) ---
        self._root: Optional[_Compartment] = None
        self._compartments: List[_Compartment] = []
        self._compartment_indices: List[int] = []
        self._leafs: List[_Compartment] = []
        self._size = 0

        # Receptor bookkeeping
        # Maps receptor global index -> receptor object
        self._receptors: List = []
        # Spike buffers: maps receptor id -> accumulated weight for current lag
        self._spike_buffer: Dict[int, float] = {}

        # Simulation state
        self._dt: Optional[float] = None
        self._spike_times: List[float] = []
        self._t = 0.0
        self._step_count = 0

        # Recording
        self._v_history: List[List[float]] = []

        # State-based execution path: enabled only when a morphology is supplied
        # at construction (the ``Simulator.create()`` path).  Bare ``cm_default()``
        # / ``cm_default(V_th=...)`` stay in pure host-loop mode (no State vars),
        # preserving the legacy API and its test suite.
        self._state_ready = False
        if compartments is not None:
            for comp in compartments:
                self.add_compartment(comp['parent_idx'], comp.get('params'))
            for rec in (receptors or []):
                self.add_receptor(rec['comp_idx'], rec['receptor_type'], rec.get('params'))
            self._compute_topology_constants()
            self.init_state()

    # ======================================================================
    # State-based (Simulator) execution path
    # ======================================================================
    #
    # The methods below turn the host-built compartment tree into a vectorised,
    # ``for_loop``-lowerable JAX model.  They reuse the exact Crank-Nicolson math
    # of the host ``step()`` (which remains the correctness oracle) but assemble
    # the tridiagonal-on-tree system as a dense batched ``linalg.solve`` over the
    # whole population in one XLA op.

    def _compute_topology_constants(self):
        r"""Snapshot the host morphology into constant (non-State) JAX arrays.

        Builds the parent map, per-compartment passive/channel parameters, the
        symmetric coupling Laplacian ``L`` (so the CN matrix is ``L + diag(...)``
        and the old-voltage coupling on the RHS is ``-L @ v``), the receptor->
        compartment scatter matrix ``S``, and per-receptor kinetic constants.
        Receptors are flattened to a single two-component form: component 1 is the
        voltage-independent dual-exp (AMPA/GABA, and the AMPA part of AMPA_NMDA);
        component 2 is the NMDA dual-exp (standalone NMDA, and the NMDA part of
        AMPA_NMDA), gated by ``nmda_scale`` and the Mg-block sigmoid.
        """
        comps = self._compartments  # index order; parent index < child index
        n = len(comps)
        self._n_comp = n

        parent = np.array([c.p_index for c in comps], dtype=np.int64)
        self._parent = parent
        self._has_parent = jnp.asarray(parent >= 0)

        self._ca = jnp.asarray(np.array([c.ca for c in comps], dtype=np.float64))
        self._gl = jnp.asarray(np.array([c.gl for c in comps], dtype=np.float64))
        self._el = jnp.asarray(np.array([c.el for c in comps], dtype=np.float64))
        self._v_init = jnp.asarray(np.array([c.v_comp for c in comps], dtype=np.float64))
        gc = np.array([c.gc for c in comps], dtype=np.float64)
        gbar_Na = np.array([c.na_chan.gbar_Na for c in comps], dtype=np.float64)
        gbar_K = np.array([c.k_chan.gbar_K for c in comps], dtype=np.float64)
        self._gbar_Na = jnp.asarray(gbar_Na)
        self._e_Na = jnp.asarray(np.array([c.na_chan.e_Na for c in comps], dtype=np.float64))
        self._gbar_K = jnp.asarray(gbar_K)
        self._e_K = jnp.asarray(np.array([c.k_chan.e_K for c in comps], dtype=np.float64))
        self._na_active = jnp.asarray(gbar_Na > 1e-9)
        self._k_active = jnp.asarray(gbar_K > 1e-9)

        # Symmetric coupling Laplacian L (n x n).  Each non-root child i couples
        # to its parent p with conductance gc[i]/2 (host ``gc__div__2``).
        L = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            p = int(parent[i])
            if p >= 0:
                gc2 = gc[i] / 2.0
                L[i, i] += gc2
                L[p, p] += gc2
                L[i, p] -= gc2
                L[p, i] -= gc2
        self._L = jnp.asarray(L)

        # Receptors: per-receptor kinetic constants + scatter to compartments.
        recs = self._receptors
        nr = len(recs)
        self._n_rec = nr
        rec_comp = np.zeros(nr, dtype=np.int64)
        e_rev = np.zeros(nr, dtype=np.float64)
        nmda_scale = np.zeros(nr, dtype=np.float64)
        gnorm1 = np.zeros(nr, dtype=np.float64)
        gnorm2 = np.zeros(nr, dtype=np.float64)
        tau_r1 = np.ones(nr, dtype=np.float64)
        tau_d1 = np.ones(nr, dtype=np.float64)
        tau_r2 = np.ones(nr, dtype=np.float64)
        tau_d2 = np.ones(nr, dtype=np.float64)
        self._rec_kind = []
        rec_to_comp = {}
        for ci, c in enumerate(comps):
            for rec in c.receptors:
                rec_to_comp[id(rec)] = ci
        for k, rec in enumerate(recs):
            ci = rec_to_comp[id(rec)]
            rec_comp[k] = ci
            e_rev[k] = rec.e_rev
            if isinstance(rec, _AMPA_NMDAReceptor):
                self._rec_kind.append('AMPA_NMDA')
                nmda_scale[k] = rec.NMDA_ratio
                gnorm1[k] = rec.g_norm_AMPA
                tau_r1[k] = rec.tau_r_AMPA
                tau_d1[k] = rec.tau_d_AMPA
                gnorm2[k] = rec.g_norm_NMDA
                tau_r2[k] = rec.tau_r_NMDA
                tau_d2[k] = rec.tau_d_NMDA
            elif isinstance(rec, _NMDAReceptor):
                self._rec_kind.append('NMDA')
                nmda_scale[k] = 1.0
                gnorm2[k] = rec.g_norm
                tau_r2[k] = rec.tau_r
                tau_d2[k] = rec.tau_d
            else:  # AMPA or GABA: voltage-independent dual-exp (component 1)
                self._rec_kind.append('GABA' if isinstance(rec, _GABAReceptor) else 'AMPA')
                gnorm1[k] = rec.g_norm
                tau_r1[k] = rec.tau_r
                tau_d1[k] = rec.tau_d
        self._rec_comp = jnp.asarray(rec_comp)
        self._rec_e_rev = jnp.asarray(e_rev)
        self._rec_nmda_scale = jnp.asarray(nmda_scale)
        self._rec_gnorm1 = jnp.asarray(gnorm1)
        self._rec_gnorm2 = jnp.asarray(gnorm2)
        self._rec_tau_r1 = jnp.asarray(tau_r1)
        self._rec_tau_d1 = jnp.asarray(tau_d1)
        self._rec_tau_r2 = jnp.asarray(tau_r2)
        self._rec_tau_d2 = jnp.asarray(tau_d2)
        # Scatter matrix S (n_comp x n_rec): S[c, k] = 1 if receptor k is on comp c.
        S = np.zeros((n, nr), dtype=np.float64)
        for k in range(nr):
            S[rec_comp[k], k] = 1.0
        self._S = jnp.asarray(S)
        # First receptor index on each compartment (for recordable name resolution).
        self._comp_to_rec = {}
        for k in range(nr):
            self._comp_to_rec.setdefault(int(rec_comp[k]), k)

        self._state_ready = True

    def init_state(self, **kwargs):
        r"""Allocate/reset the vectorised State variables (no-op in host-loop mode).

        Compartment voltages start at each compartment's ``v_comp``; the Na/K
        gating variables start at their steady-state values for that voltage
        (matching the host ``_init_statevars``); receptor conductances start at
        zero.
        """
        if not getattr(self, '_state_ready', False):
            return  # pure host-loop instance: no State to allocate
        n = self._n_comp
        nr = self._n_rec
        vs = self.varshape
        dftype = brainstate.environ.dftype()

        v0 = jnp.broadcast_to(self._v_init.astype(dftype), vs + (n,))
        self.V = brainstate.HiddenState(v0)
        m_inf, _, h_inf, _ = _na_gates_vec(self._v_init)
        n_inf, _ = _k_gate_vec(self._v_init)
        self.m_Na = brainstate.HiddenState(jnp.broadcast_to(m_inf.astype(dftype), vs + (n,)))
        self.h_Na = brainstate.HiddenState(jnp.broadcast_to(h_inf.astype(dftype), vs + (n,)))
        self.n_K = brainstate.HiddenState(jnp.broadcast_to(n_inf.astype(dftype), vs + (n,)))
        self.syn_gr1 = brainstate.HiddenState(jnp.zeros(vs + (nr,), dtype=dftype))
        self.syn_gd1 = brainstate.HiddenState(jnp.zeros(vs + (nr,), dtype=dftype))
        self.syn_gr2 = brainstate.HiddenState(jnp.zeros(vs + (nr,), dtype=dftype))
        self.syn_gd2 = brainstate.HiddenState(jnp.zeros(vs + (nr,), dtype=dftype))
        # One-step buffer for Simulator-injected device current (dc/step/noise/ac):
        # captured from ``sum_current_inputs`` this step, applied next step, so the
        # injection carries NEST's current ring-buffer delay (= the dc connection's
        # 0.1 ms / 1-step delay in the receptors_and_current demo).
        self.I_stim = brainstate.ShortTermState(jnp.zeros(vs + (n,), dtype=dftype))

    def delta_label_for_receptor(self, receptor_type):
        r"""Map a receptor index (NEST ``syn_idx``, add-order) to its delta-input label.

        The Simulator's :class:`EventProjection` calls this to route a
        ``connect(spike_source, cm, receptor_type=k)`` deposit into the single
        channel that :meth:`update` reads back with ``sum_delta_inputs(label=...)``.
        The trailing ``#`` keeps label ``cm_rcpt1#`` from prefix-colliding with
        ``cm_rcpt10#`` (deposits are selected by label prefix).
        """
        try:
            rt = int(receptor_type)
        except (TypeError, ValueError):
            raise ValueError(
                f'cm_default receptor_type must be an integer receptor index, got {receptor_type!r}.'
            )
        if 0 <= rt < self._n_rec:
            return f'cm_rcpt{rt}#'
        raise ValueError(
            f'invalid receptor_type {rt}; cm_default has {self._n_rec} receptor(s) '
            f'(valid indices 0..{self._n_rec - 1}).'
        )

    @property
    def NCOMP(self):
        r"""Compartment count (the Simulator reads this to scatter device current)."""
        return self._n_comp

    def current_compartment_for_receptor(self, receptor_type):
        r"""Map a current generator's ``receptor_type`` to its target compartment index.

        Unlike spike receptors (routed by :meth:`delta_label_for_receptor`), NEST's
        ``cm_default`` addresses a **current** generator's ``receptor_type`` directly
        as the *compartment index* (see the model docstring: "the receptor type is
        the compartment index").  The Simulator calls this so
        ``connect(dc_generator, cm, receptor_type=c)`` injects into compartment ``c``.
        """
        try:
            rt = int(receptor_type)
        except (TypeError, ValueError):
            raise ValueError(
                f'cm_default current receptor_type must be an integer compartment index, '
                f'got {receptor_type!r}.'
            )
        if 0 <= rt < self._n_comp:
            return rt
        raise ValueError(
            f'invalid current receptor_type {rt}; cm_default current generators address a '
            f'compartment index 0..{self._n_comp - 1}.'
        )

    def update(self, x=0. * u.pA, spike_events=None, current_events=None):
        r"""Advance all compartments by one Crank-Nicolson timestep (vectorised).

        Parameters
        ----------
        x : ArrayLike or Quantity, optional
            External current (``pA``).  A ``(*in_size, n_comp)`` array is applied
            per compartment; a scalar / ``(*in_size,)`` value is applied to the
            soma (compartment 0).  Default ``0 pA``.
        spike_events : list, optional
            Direct synaptic spikes as ``(receptor_index, weight)`` tuples or
            ``{'receptor_type', 'weight'}`` dicts (weight in ``nS``).  Used by the
            host-vs-State oracle test; the Simulator instead deposits via the
            delta-channel routing read here with ``sum_delta_inputs``.
        current_events : list, optional
            Direct current injections as ``(comp_index, current)`` tuples or
            ``{'receptor_type'/'comp_idx', 'current'}`` dicts (current in ``pA``).

        Returns
        -------
        spike : Array
            Per-neuron somatic threshold crossing this step (diagnostic; the
            cable model performs no voltage reset), as a float array.
        """
        dt = brainstate.environ.get_dt().to_decimal(u.ms)
        v = self.V.value                       # (*vs, n_comp), mV magnitude
        dtype = v.dtype
        vs = self.varshape
        n = self._n_comp
        nr = self._n_rec

        # ---- incoming synaptic weights per receptor (nS) ----
        sw = jnp.zeros(vs + (nr,), dtype=dtype)
        if spike_events:
            for ev in spike_events:
                if isinstance(ev, dict):
                    ridx = int(ev.get('receptor_type', ev.get('receptor')))
                    w = ev.get('weight')
                else:
                    ridx, w = ev
                    ridx = int(ridx)
                sw = sw.at[..., ridx].add(_to_nS(w))
        for k in range(nr):
            dep = self.sum_delta_inputs(0. * u.nS, label=f'cm_rcpt{k}#')
            sw = sw.at[..., k].add(_to_nS(dep))

        # ---- external current per compartment (pA) ----
        i_ext = jnp.zeros(vs + (n,), dtype=dtype)
        xv = jnp.asarray(_to_pA(x), dtype=dtype)
        if xv.ndim >= 1 and xv.shape[-1] == n:
            i_ext = i_ext + jnp.broadcast_to(xv, vs + (n,))
        else:
            i_ext = i_ext.at[..., 0].add(jnp.broadcast_to(xv, vs))
        if current_events:
            for ev in current_events:
                if isinstance(ev, dict):
                    cidx = int(ev.get('comp_idx', ev.get('receptor_type', ev.get('receptor'))))
                    cur = ev.get('current', ev.get('weight'))
                else:
                    cidx, cur = ev
                    cidx = int(cidx)
                i_ext = i_ext.at[..., cidx].add(_to_pA(cur))
        # Simulator-injected device current rides the one-step ``I_stim`` buffer:
        # apply the previous step's captured current, then capture this step's deposit
        # (``x``/``current_events`` above are the immediate direct/oracle path and are
        # NOT buffered, so the host-vs-State oracle stays bit-exact).
        i_ext = i_ext + self.I_stim.value
        new_i_stim = _to_pA(self.sum_current_inputs(jnp.zeros(vs + (n,), dtype=dtype) * u.pA))

        # ---- ion channels: exact-exp gating, frozen where the channel is off ----
        m_inf, tau_m, h_inf, tau_h = _na_gates_vec(v)
        n_inf, tau_n = _k_gate_vec(v)
        m_new = self.m_Na.value * jnp.exp(-dt / tau_m) + (1.0 - jnp.exp(-dt / tau_m)) * m_inf
        h_new = self.h_Na.value * jnp.exp(-dt / tau_h) + (1.0 - jnp.exp(-dt / tau_h)) * h_inf
        n_new = self.n_K.value * jnp.exp(-dt / tau_n) + (1.0 - jnp.exp(-dt / tau_n)) * n_inf
        m_new = jnp.where(self._na_active, m_new, self.m_Na.value)
        h_new = jnp.where(self._na_active, h_new, self.h_Na.value)
        n_new = jnp.where(self._k_active, n_new, self.n_K.value)
        g_Na = jnp.where(self._na_active, self._gbar_Na * m_new ** 3 * h_new, 0.0)
        g_K = jnp.where(self._k_active, self._gbar_K * n_new, 0.0)
        g_chan = (g_Na + g_K) / 2.0
        i_chan = g_Na * (self._e_Na - v / 2.0) + g_K * (self._e_K - v / 2.0)

        # ---- receptors: one vectorised formula covers AMPA/GABA/NMDA/AMPA_NMDA ----
        if nr:
            gr1 = self.syn_gr1.value * jnp.exp(-dt / self._rec_tau_r1)
            gd1 = self.syn_gd1.value * jnp.exp(-dt / self._rec_tau_d1)
            gr2 = self.syn_gr2.value * jnp.exp(-dt / self._rec_tau_r2)
            gd2 = self.syn_gd2.value * jnp.exp(-dt / self._rec_tau_d2)
            s1 = sw * self._rec_gnorm1
            gr1 = gr1 - s1
            gd1 = gd1 + s1
            s2 = sw * self._rec_gnorm2
            gr2 = gr2 - s2
            gd2 = gd2 + s2
            g1 = gr1 + gd1
            g2 = gr2 + gd2
            v_rec = v[..., self._rec_comp]                     # (*vs, nr)
            B, dB = _nmda_sigmoid_vec(v_rec)
            drive = self._rec_e_rev - v_rec
            i_tot = (g1 + self._rec_nmda_scale * g2 * B) * drive
            di_dv = -g1 + self._rec_nmda_scale * g2 * (dB * drive - B)
            g_val = -di_dv / 2.0
            i_val = i_tot + g_val * v_rec
            g_rec = jnp.einsum('cr,...r->...c', self._S, g_val)
            i_rec = jnp.einsum('cr,...r->...c', self._S, i_val)
        else:
            gr1 = gd1 = gr2 = gd2 = None
            g_rec = jnp.zeros(vs + (n,), dtype=dtype)
            i_rec = jnp.zeros(vs + (n,), dtype=dtype)

        # ---- assemble & solve the Crank-Nicolson system  A v_new = rhs ----
        ca_dt = self._ca / dt
        gl_2 = self._gl / 2.0
        diag = ca_dt + gl_2 + g_chan + g_rec
        A = self._L + _diag_embed(diag)
        Lv = jnp.einsum('ij,...j->...i', self._L, v)
        rhs = (ca_dt - gl_2) * v + self._gl * self._el + i_chan + i_rec + i_ext - Lv
        v_new = jnp.linalg.solve(A, rhs[..., None])[..., 0]

        # ---- spike detection at the root (threshold crossing; no reset) ----
        spike = (v_new[..., 0] >= self.V_th) & (v[..., 0] < self.V_th)

        # ---- write back ----
        self.V.value = v_new
        self.m_Na.value = m_new
        self.h_Na.value = h_new
        self.n_K.value = n_new
        if nr:
            self.syn_gr1.value = gr1
            self.syn_gd1.value = gd1
            self.syn_gr2.value = gr2
            self.syn_gd2.value = gd2
        self.I_stim.value = new_i_stim
        return u.math.asarray(spike, dtype=brainstate.environ.dftype())

    def read_recordable(self, name):
        r"""Resolve a NEST multimeter recordable name to a State slice.

        Supports the per-compartment names the §3.10 demos record: ``v_comp{c}``
        (mV), gating ``m_Na_{c}``/``h_Na_{c}``/``n_K_{c}``, AMPA_NMDA conductance
        components ``g_{r,d}_AN_{AMPA,NMDA}_{c}``, and totals ``g_AMPA_{c}`` /
        ``g_GABA_{c}`` / ``g_NMDA_{c}``.  Receptor recordables are keyed by the
        compartment index (NEST convention), resolved via the first receptor on
        that compartment.
        """
        if name.startswith('v_comp'):
            return self.V.value[..., int(name[len('v_comp'):])] * u.mV
        for pref, state in (('m_Na_', 'm_Na'), ('h_Na_', 'h_Na'), ('n_K_', 'n_K')):
            if name.startswith(pref):
                return getattr(self, state).value[..., int(name[len(pref):])]
        comp_pairs = (
            ('g_r_AN_AMPA_', 'syn_gr1'), ('g_d_AN_AMPA_', 'syn_gd1'),
            ('g_r_AN_NMDA_', 'syn_gr2'), ('g_d_AN_NMDA_', 'syn_gd2'),
        )
        for pref, state in comp_pairs:
            if name.startswith(pref):
                k = self._comp_to_rec[int(name[len(pref):])]
                return getattr(self, state).value[..., k]
        for pref, rise, decay in (
            ('g_AMPA_', 'syn_gr1', 'syn_gd1'),
            ('g_GABA_', 'syn_gr1', 'syn_gd1'),
            ('g_NMDA_', 'syn_gr2', 'syn_gd2'),
        ):
            if name.startswith(pref):
                k = self._comp_to_rec[int(name[len(pref):])]
                return (getattr(self, rise).value + getattr(self, decay).value)[..., k]
        raise KeyError(f'cm_default has no recordable {name!r}')

    def add_compartment(self, parent_idx: int, params: Optional[Dict] = None) -> int:
        r"""Add a compartment to the neuron's morphological tree structure.

        Compartments are added incrementally to build the dendritic tree. The first
        compartment should have parent_idx=-1 (root/soma), and subsequent compartments
        reference their parent by index. Compartments are assigned sequential integer
        indices starting from 0.

        Parameters
        ----------
        parent_idx : int
            Index of the parent compartment in the tree. Use -1 for the root compartment
            (typically the soma). For all other compartments, this must be the index of
            a previously added compartment.
        params : dict, optional
            Dictionary of electrical and biophysical parameters. If None or if specific
            keys are absent, default values are used. Supported keys:
            'C_m' (1.0 nF), 'g_C' (0.01 µS), 'g_L' (0.1 µS), 'e_L' (-70.0 mV),
            'v_comp' (e_L), 'gbar_Na' (0.0 µS), 'e_Na' (50.0 mV),
            'gbar_K' (0.0 µS), 'e_K' (-85.0 mV).

        Returns
        -------
        int
            Index of the newly created compartment. This index should be used when
            adding child compartments or attaching receptors.

        Raises
        ------
        ValueError
            If parent_idx=-1 but a root compartment already exists, or if parent_idx
            references a non-existent compartment.

        Notes
        -----
        **Tree Construction Order**

        Compartments can be added in any order (depth-first, breadth-first, arbitrary)
        as long as each compartment's parent exists before the compartment is added.

        **Typical Construction Pattern** ::

            model = cm_default()
            soma = model.add_compartment(-1, {'C_m': 100.0, 'gbar_Na': 5000.0})  # idx 0
            dend1 = model.add_compartment(soma, {'C_m': 10.0, 'g_C': 1.0})       # idx 1
            dend2 = model.add_compartment(soma, {'C_m': 10.0, 'g_C': 1.0})       # idx 2
            dend1a = model.add_compartment(dend1, {'C_m': 5.0, 'g_C': 0.5})      # idx 3

        **Parameter Guidelines**

        - **C_m**: Scales with compartment surface area (typically 1-100 nF for dendrites,
          10-300 nF for soma)
        - **g_C**: Axial coupling conductance; higher values produce tighter electrical
          coupling to parent (typical range: 0.01-10 µS)
        - **g_L**: Passive leak conductance; scales with surface area (typical: 0.01-1 µS)
        - **gbar_Na, gbar_K**: Set to non-zero values to enable action potential generation
          in that compartment (typical soma values: gbar_Na=1000-5000 µS, gbar_K=200-1000 µS)
        """
        comp_index = self._size
        comp = _Compartment(comp_index, parent_idx, params)

        if parent_idx < 0:
            if self._root is not None:
                raise ValueError("Root compartment already exists.")
            self._root = comp
        else:
            parent = self._get_compartment(parent_idx)
            if parent is None:
                raise ValueError(
                    f"Parent compartment {parent_idx} does not exist."
                )
            parent.children.append(comp)
            comp.parent = parent

        self._size += 1
        self._compartment_indices.append(comp_index)
        self._update_compartment_list()

        return comp_index

    def add_receptor(self, comp_idx: int, receptor_type: str,
                     params: Optional[Dict] = None) -> int:
        r"""Add a synaptic receptor to a specific compartment.

        Receptors define the postsynaptic response to presynaptic spikes. Multiple
        receptors of the same or different types can be added to a single compartment
        to model multiple synaptic contacts or colocalized receptor populations.

        Parameters
        ----------
        comp_idx : int
            Index of the compartment where this receptor should be placed (must be a
            previously created compartment index returned by add_compartment).
        receptor_type : str
            Type of receptor to add. Must be one of:
            - 'AMPA': Fast excitatory glutamate receptor
            - 'GABA': Fast inhibitory GABAergic receptor
            - 'NMDA': Slow excitatory glutamate receptor with Mg²⁺ block
            - 'AMPA_NMDA': Combined AMPA and NMDA receptor at the same synapse
        params : dict, optional
            Dictionary of receptor-specific parameters. If None or if specific keys
            are absent, default values are used. See class docstring for parameter
            names and defaults for each receptor type.

        Returns
        -------
        int
            Global receptor index (unique across all compartments). This index must be
            used when calling add_spike() to deliver presynaptic spikes to this receptor.

        Raises
        ------
        ValueError
            If comp_idx references a non-existent compartment or if receptor_type is
            not one of the recognized types.

        Notes
        -----
        **Receptor Placement Strategies**

        - **Proximal excitation**: Place AMPA/NMDA receptors on soma and proximal dendrites
          for direct drive of action potentials
        - **Distal excitation**: Place receptors on distal dendrites for spatiotemporal
          integration and nonlinear dendritic processing
        - **Inhibition placement**: Somatic GABA receptors provide strong divisive inhibition;
          dendritic GABA receptors enable compartment-specific gating
        - **Colocalization**: Use 'AMPA_NMDA' type to efficiently model colocalized receptors
          at the same synapse

        **Receptor Index Usage** ::

            soma = model.add_compartment(-1, {...})
            dend = model.add_compartment(soma, {...})

            # Add receptors
            r_exc_soma = model.add_receptor(soma, 'AMPA')       # returns 0
            r_exc_dend = model.add_receptor(dend, 'AMPA_NMDA')  # returns 1
            r_inh_soma = model.add_receptor(soma, 'GABA')       # returns 2

            # Later, deliver spikes to specific receptors
            model.add_spike(r_exc_soma, weight=2.5)  # excitatory input to soma
            model.add_spike(r_inh_soma, weight=5.0)  # inhibitory input to soma

        **Performance Considerations**

        Each receptor adds computational overhead for conductance updates. For large-scale
        simulations, group synapses with identical parameters and sum their weights rather
        than creating individual receptor objects for each synapse.
        """
        comp = self._get_compartment(comp_idx)
        if comp is None:
            raise ValueError(f"Compartment {comp_idx} does not exist.")

        if params is None:
            params = {}

        if receptor_type == 'AMPA':
            rec = _AMPAReceptor(**{
                k: v for k, v in params.items()
                if k in ('e_AMPA', 'tau_r_AMPA', 'tau_d_AMPA')
            })
        elif receptor_type == 'GABA':
            rec = _GABAReceptor(**{
                k: v for k, v in params.items()
                if k in ('e_GABA', 'tau_r_GABA', 'tau_d_GABA')
            })
        elif receptor_type == 'NMDA':
            rec = _NMDAReceptor(**{
                k: v for k, v in params.items()
                if k in ('e_NMDA', 'tau_r_NMDA', 'tau_d_NMDA')
            })
        elif receptor_type == 'AMPA_NMDA':
            rec = _AMPA_NMDAReceptor(**{
                k: v for k, v in params.items()
                if k in ('e_AMPA_NMDA', 'tau_r_AMPA', 'tau_d_AMPA',
                         'tau_r_NMDA', 'tau_d_NMDA', 'NMDA_ratio')
            })
        else:
            raise ValueError(
                f"Unknown receptor type: {receptor_type}. "
                f"Must be one of AMPA, GABA, NMDA, AMPA_NMDA."
            )

        receptor_idx = len(self._receptors)
        self._receptors.append(rec)
        comp.add_receptor(rec)
        return receptor_idx

    def pre_run_hook(self, dt: float):
        r"""Initialize the model for simulation after construction is complete.

        This method performs essential preprocessing that must occur after all
        compartments and receptors have been added but before the first simulation
        step. It finalizes the tree structure, precomputes integration constants,
        and initializes receptor exponential factors.

        Parameters
        ----------
        dt : float
            Integration timestep in milliseconds. This value is stored and used for
            all subsequent simulation steps until a new pre_run_hook() call.

        Raises
        ------
        ValueError
            If no compartments have been added to the model (i.e., _root is None).

        Notes
        -----
        **Preprocessing Steps**

        1. Validates that at least one compartment exists (root must be defined)
        2. Rebuilds the flat compartment list in index order
        3. Resolves parent-child relationships (pointer updates)
        4. Identifies leaf compartments (needed for tree traversal algorithm)
        5. Calls each compartment's pre_run_hook() to compute integration constants
        6. Calls each receptor's pre_run_hook() to compute exponential decay factors

        **Simulation Workflow** ::

            # 1. Construct model
            model = cm_default(V_th=-55.0)
            soma = model.add_compartment(-1, {...})
            dend = model.add_compartment(soma, {...})
            r_exc = model.add_receptor(dend, 'AMPA', {...})

            # 2. Initialize simulation (REQUIRED before stepping)
            model.pre_run_hook(dt=0.1)

            # 3. Run simulation loop
            for i in range(num_steps):
                model.add_spike(r_exc, weight=...)  # optional: deliver spikes
                spike = model.step()                 # advance one timestep
                v_soma = model.get_voltage(soma)    # read voltages

        **Changing Timestep**

        To change the integration timestep mid-simulation, call pre_run_hook() again
        with the new dt value. All precomputed constants will be recalculated. However,
        this does not reset state variables (voltages, conductances, gating variables);
        the simulation continues from the current state with the new timestep.

        **Performance Impact**

        This method performs O(n) operations where n is the number of compartments.
        It should be called once at the start of simulation, not in the inner loop.
        """
        if self._root is None:
            raise ValueError("No compartments have been added.")

        self._dt = dt
        self._update_compartment_list()
        self._set_parents()
        self._set_leafs()

        for comp in self._compartments:
            comp.pre_run_hook(dt)

    def add_spike(self, receptor_idx: int, weight: float):
        r"""Schedule a presynaptic spike for delivery to a specific receptor.

        Spikes added via this method will be processed during the next call to step().
        Multiple spikes to the same receptor within a single timestep are accumulated
        (weights are summed).

        Parameters
        ----------
        receptor_idx : int
            Global receptor index returned by add_receptor(). This identifies which
            receptor receives the spike.
        weight : float
            Synaptic weight in microsiemens (µS). This scales the conductance waveform
            amplitude. For a normalized receptor, weight=1.0 produces a peak conductance
            of 1.0 µS. Typical weights range from 0.1 to 10 µS depending on synapse
            strength and receptor type.

        Notes
        -----
        **Spike Timing and Delivery**

        - Spikes are queued in an internal buffer (_spike_buffer) when add_spike() is called
        - During step(), all buffered spikes are processed simultaneously
        - The buffer is cleared after each step()
        - To deliver a spike at simulation time t, call add_spike() before calling step()
          at time t

        **Weight Accumulation**

        If multiple spikes are added to the same receptor before step() is called, their
        weights are summed. This allows efficient handling of multiple presynaptic inputs
        converging on the same postsynaptic receptor:

            model.add_spike(r_exc, weight=0.5)
            model.add_spike(r_exc, weight=0.3)
            model.step()  # Processes total weight=0.8

        **Typical Weight Ranges**

        - AMPA: 0.1 - 5.0 µS (fast, moderate strength)
        - NMDA: 0.1 - 2.0 µS (slow, often weighted lower due to long duration)
        - GABA: 0.5 - 10.0 µS (inhibitory, often stronger to balance excitation)
        - AMPA_NMDA: 0.1 - 3.0 µS (total excitatory drive, NMDA scaled by NMDA_ratio)

        **Example Usage** ::

            # Setup
            model = cm_default()
            soma = model.add_compartment(-1, {...})
            r_exc = model.add_receptor(soma, 'AMPA')
            r_inh = model.add_receptor(soma, 'GABA')
            model.pre_run_hook(dt=0.1)

            # Deliver excitatory spike at t=5.0 ms
            for i in range(50):  # Steps 0-49
                model.step()
            model.add_spike(r_exc, weight=2.0)
            spike = model.step()  # Step 50, processes spike

            # Deliver balanced E/I input at t=10.0 ms
            for i in range(49):  # Steps 51-99
                model.step()
            model.add_spike(r_exc, weight=3.0)
            model.add_spike(r_inh, weight=6.0)
            spike = model.step()  # Step 100, processes both spikes
        """
        rec = self._receptors[receptor_idx]
        key = id(rec)
        self._spike_buffer[key] = self._spike_buffer.get(key, 0.0) + weight

    def add_current(self, comp_idx: int, current: float):
        r"""Inject external current into a specific compartment for the next timestep.

        This method allows direct current injection, useful for probing intrinsic
        excitability, applying step currents, or simulating artificial stimulation
        electrodes.

        Parameters
        ----------
        comp_idx : int
            Index of the compartment where current should be injected.
        current : float
            Current amplitude in nanoamperes (nA). Positive values are depolarizing
            (inward current), negative values are hyperpolarizing (outward current).

        Notes
        -----
        **Current Timing and Buffer**

        - Currents are queued in the compartment's internal buffer (_current_buffer)
        - During the next step(), the current is consumed from the buffer and applied
        - The buffer is a FIFO queue; multiple add_current() calls queue successive currents
        - Unlike spikes, currents are NOT accumulated; each add_current() call adds a
          separate entry to the queue

        **Typical Current Magnitudes**

        For a soma with C_m ≈ 100 nF, to evoke a ~10 mV/ms depolarization rate:

        .. math::

            I = C_m \cdot \frac{dV}{dt} = 100 \text{ nF} \cdot 10 \text{ mV/ms} = 1 \text{ nA}

        Typical ranges:
        - **Subthreshold probing**: 0.01 - 0.5 nA
        - **Spike threshold testing**: 0.5 - 2.0 nA
        - **Strong depolarization**: 2.0 - 10.0 nA

        **Step Current Stimulus** ::

            model = cm_default()
            soma = model.add_compartment(-1, {'C_m': 100.0, 'gbar_Na': 4000.0, 'gbar_K': 800.0})
            model.pre_run_hook(dt=0.1)

            # Apply 1 nA step current for 50 ms (500 timesteps at dt=0.1 ms)
            for i in range(500):
                model.add_current(soma, 1.0)  # Queue current
                spike = model.step()           # Apply and integrate
                if spike:
                    print(f"Spike at t={i*0.1} ms")

        **Ramp Current** ::

            # Linear ramp from 0 to 2 nA over 1000 steps
            for i in range(1000):
                model.add_current(soma, 2.0 * i / 1000.0)
                model.step()

        **Current vs. Synaptic Input**

        - add_current(): Direct current injection, bypasses synaptic dynamics
        - add_spike(): Activates synaptic receptors, produces conductance transients
          with realistic kinetics and reversal potential effects
        """
        comp = self._get_compartment(comp_idx)
        comp._current_buffer.append(current)

    def step(self) -> bool:
        r"""Advance the simulation by one integration timestep.

        This method performs the core computational work: it constructs the Crank-Nicolson
        matrix system from the cable equation, solves for new voltages using the O(n)
        tree algorithm, updates all ion channel and receptor states, and detects spikes
        at the soma.

        Returns
        -------
        bool
            True if a spike was detected at the soma (root compartment) during this
            timestep, False otherwise. A spike is detected when the soma voltage crosses
            V_th from below.

        Notes
        -----
        **Computation Sequence**

        1. **Save previous soma voltage** for spike detection
        2. **Construct matrix elements** for each compartment:
           - Passive membrane terms (leak, coupling)
           - Ion channel contributions (Na, K with exact integration)
           - Synaptic receptor contributions (process buffered spikes)
           - External current injections
        3. **Solve linear system** using two-pass tree algorithm:
           - Down-sweep: Propagate information from leaves to root
           - Up-sweep: Back-substitute voltages from root to leaves
        4. **Detect spikes** via threshold crossing at soma
        5. **Clear spike buffer** for next timestep
        6. **Increment counters** (_step_count, _t)

        **Spike Detection Logic**

        .. math::

            \text{spike} = (V_{\text{soma}}^{\text{old}} < V_{th}) \land
                           (V_{\text{soma}}^{\text{new}} \geq V_{th})

        The strict inequality ensures each threshold crossing generates exactly one spike
        (no double-counting if voltage remains above threshold).

        **Performance Characteristics**

        - Time complexity: O(n) where n = number of compartments
        - Space complexity: O(n) for matrix elements and tree pointers
        - Dominated by: Matrix construction (ion channel/receptor updates), tree solver

        For a typical neuron with 100 compartments, step() completes in ~10-50 µs on
        modern CPUs (pure Python implementation; compiled versions would be 10-100× faster).

        **Simulation Loop Pattern** ::

            model.pre_run_hook(dt=0.1)
            spike_times = []

            for i in range(num_steps):
                # Optional: deliver synaptic input
                if i in input_times:
                    model.add_spike(receptor_idx, weight=...)

                # Integrate one timestep
                spike = model.step()

                # Record spike times
                if spike:
                    spike_times.append(i * dt)

                # Optional: record state variables
                v_soma = model.get_voltage(soma_idx)
                voltages[i] = v_soma

        **Relationship to Real Time**

        Each step() call advances simulation time by dt milliseconds (set in pre_run_hook).
        For dt=0.1 ms and num_steps=10000, the simulation covers 1000 ms (1 second) of
        biological time. The wall-clock time depends on model complexity and hardware.
        """
        dt = self._dt
        v_prev = self._root.v_comp

        # Construct matrix
        for comp in self._compartments:
            comp.construct_matrix_element(dt, self._spike_buffer)

        # Clear spike buffer for next step
        self._spike_buffer.clear()

        # Solve matrix (O(n) tree algorithm)
        self._solve_matrix()

        # Spike detection at root
        spike = (self._root.v_comp >= self.V_th and v_prev < self.V_th)

        self._step_count += 1
        self._t += dt

        return spike

    def get_voltage(self, comp_idx: int) -> float:
        r"""Get the current membrane voltage of a specific compartment.

        Parameters
        ----------
        comp_idx : int
            Index of the compartment to query (must be a valid compartment created
            via add_compartment).

        Returns
        -------
        float
            Current membrane voltage in millivolts. This reflects the voltage state
            immediately after the most recent step() call.

        Notes
        -----
        This method provides read access to compartment voltages for:
        - Monitoring soma voltage during simulation
        - Recording dendritic voltage dynamics
        - Implementing custom spike detection or analysis
        - Debugging model behavior

        **Usage Example** ::

            model.pre_run_hook(dt=0.1)
            v_trace_soma = []
            v_trace_dend = []

            for i in range(1000):
                model.step()
                v_trace_soma.append(model.get_voltage(soma_idx))
                v_trace_dend.append(model.get_voltage(dend_idx))

            # Plot voltage traces
            plt.plot(v_trace_soma, label='Soma')
            plt.plot(v_trace_dend, label='Dendrite')
        """
        return self._get_compartment(comp_idx).v_comp

    def get_voltages(self) -> List[float]:
        r"""Get membrane voltages of all compartments simultaneously.

        Returns
        -------
        list of float
            List of membrane voltages in millivolts, ordered by compartment index.
            The length of the list equals the number of compartments. Element i
            contains the voltage of compartment i.

        Notes
        -----
        This method is more efficient than calling get_voltage() in a loop when
        accessing multiple compartment voltages, as it avoids repeated lookups.

        **Usage Example** ::

            model.pre_run_hook(dt=0.1)
            voltage_matrix = []

            for i in range(1000):
                model.step()
                voltage_matrix.append(model.get_voltages())

            # Convert to numpy array for analysis
            V = np.array(voltage_matrix)  # Shape: (timesteps, num_compartments)

            # Analyze spatial voltage patterns
            plt.imshow(V.T, aspect='auto', cmap='RdBu_r')
            plt.xlabel('Time step')
            plt.ylabel('Compartment index')
            plt.colorbar(label='Voltage (mV)')
        """
        return [comp.v_comp for comp in self._compartments]

    def get_na_state(self, comp_idx: int):
        r"""Get sodium channel gating variable states for a specific compartment.

        Parameters
        ----------
        comp_idx : int
            Index of the compartment to query.

        Returns
        -------
        m_Na : float
            Activation gating variable (0 to 1). Higher values indicate more channels
            in the open/conducting state. Activates rapidly during depolarization.
        h_Na : float
            Inactivation gating variable (0 to 1). Lower values indicate more channels
            in the inactivated state. Inactivates slowly during sustained depolarization,
            preventing continuous Na current.

        Notes
        -----
        The sodium current is proportional to m³h, so even with full activation (m≈1),
        inactivation (h≈0) can shut off the current. This m³h gating produces the
        characteristic transient Na current underlying action potentials.

        **Usage Example** ::

            # Monitor Na channel dynamics during an action potential
            for i in range(200):
                model.step()
                m, h = model.get_na_state(soma_idx)
                print(f"t={i*0.1:.1f} ms: m={m:.3f}, h={h:.3f}, I_Na ~ {m**3 * h:.3f}")
        """
        comp = self._get_compartment(comp_idx)
        return comp.na_chan.m_Na, comp.na_chan.h_Na

    def get_k_state(self, comp_idx: int):
        r"""Get potassium channel activation state for a specific compartment.

        Parameters
        ----------
        comp_idx : int
            Index of the compartment to query.

        Returns
        -------
        n_K : float
            Activation gating variable (0 to 1). The potassium current is directly
            proportional to n (not n⁴ as in classic HH). Higher values indicate more
            channels in the conducting state. Activates with slower kinetics than Na,
            contributing to action potential repolarization and afterhyperpolarization.

        Notes
        -----
        Unlike the classic Hodgkin-Huxley model (I_K ∝ n⁴), this implementation uses
        linear dependence (I_K ∝ n), which is common in simplified compartmental models
        for computational efficiency while preserving qualitative dynamics.
        """
        comp = self._get_compartment(comp_idx)
        return comp.k_chan.n_K

    def get_receptor_state(self, receptor_idx: int):
        r"""Get synaptic receptor conductance state variables.

        Parameters
        ----------
        receptor_idx : int
            Global receptor index (returned by add_receptor).

        Returns
        -------
        dict
            Dictionary mapping state variable names to current values (floats in µS).
            The keys depend on receptor type:

            - AMPA: {'g_r_AMPA': float, 'g_d_AMPA': float}
            - GABA: {'g_r_GABA': float, 'g_d_GABA': float}
            - NMDA: {'g_r_NMDA': float, 'g_d_NMDA': float}
            - AMPA_NMDA: {'g_r_AN_AMPA': float, 'g_d_AN_AMPA': float,
                          'g_r_AN_NMDA': float, 'g_d_AN_NMDA': float}

            where g_r represents the rising phase state and g_d the decaying phase state.
            The total conductance is g_total = g_r + g_d (note: g_r is stored as negative).

        Notes
        -----
        This method is primarily useful for:
        - Debugging synaptic dynamics
        - Verifying correct dual-exponential waveforms
        - Monitoring receptor state evolution during learning simulations
        - Implementing state-dependent plasticity rules

        **Usage Example** ::

            r_idx = model.add_receptor(dend_idx, 'AMPA')
            model.pre_run_hook(dt=0.1)

            # Deliver a spike and track conductance evolution
            model.add_spike(r_idx, weight=2.0)
            g_trace = []
            for i in range(200):
                model.step()
                state = model.get_receptor_state(r_idx)
                g_total = state['g_r_AMPA'] + state['g_d_AMPA']
                g_trace.append(g_total)

            # Plot dual-exponential conductance waveform
            plt.plot(g_trace)
            plt.xlabel('Time (0.1 ms steps)')
            plt.ylabel('Conductance (µS)')
        """
        rec = self._receptors[receptor_idx]
        if isinstance(rec, _AMPAReceptor):
            return {'g_r_AMPA': rec.g_r, 'g_d_AMPA': rec.g_d}
        elif isinstance(rec, _GABAReceptor):
            return {'g_r_GABA': rec.g_r, 'g_d_GABA': rec.g_d}
        elif isinstance(rec, _NMDAReceptor):
            return {'g_r_NMDA': rec.g_r, 'g_d_NMDA': rec.g_d}
        elif isinstance(rec, _AMPA_NMDAReceptor):
            return {
                'g_r_AN_AMPA': rec.g_r_AMPA, 'g_d_AN_AMPA': rec.g_d_AMPA,
                'g_r_AN_NMDA': rec.g_r_NMDA, 'g_d_AN_NMDA': rec.g_d_NMDA,
            }
        return {}

    @property
    def num_compartments(self) -> int:
        return self._size

    @property
    def num_receptors(self) -> int:
        return len(self._receptors)

    # ----- Internal methods -----

    def _get_compartment(self, comp_idx: int) -> Optional[_Compartment]:
        r"""Find compartment by index via search."""
        if comp_idx < 0 or comp_idx >= len(self._compartments):
            return None
        return self._compartments[comp_idx]

    def _update_compartment_list(self):
        r"""Rebuild flat list of compartments in index order."""
        self._compartments = []
        for idx in self._compartment_indices:
            comp = self._find_compartment(self._root, idx)
            if comp is not None:
                self._compartments.append(comp)

    def _find_compartment(self, comp: Optional[_Compartment],
                          target_idx: int) -> Optional[_Compartment]:
        r"""Recursively find compartment by index."""
        if comp is None:
            return None
        if comp.comp_index == target_idx:
            return comp
        for child in comp.children:
            result = self._find_compartment(child, target_idx)
            if result is not None:
                return result
        return None

    def _set_parents(self):
        r"""Set parent pointers for all compartments."""
        for comp in self._compartments:
            if comp.p_index >= 0:
                parent = self._get_compartment(comp.p_index)
                comp.parent = parent
            else:
                comp.parent = None

    def _set_leafs(self):
        r"""Identify leaf compartments (no children)."""
        self._leafs = [
            comp for comp in self._compartments
            if len(comp.children) == 0
        ]

    def _solve_matrix(self):
        r"""Solve the tridiagonal-on-tree system using O(n) algorithm."""
        if len(self._leafs) == 0:
            # Single compartment (root only)
            self._root.v_comp = self._root.ff / self._root.gg
            return

        # Down-sweep: leaf to root
        leaf_iter = iter(self._leafs)
        first_leaf = next(leaf_iter)
        self._solve_downsweep(first_leaf, leaf_iter)

        # Up-sweep: root to leaves
        self._solve_upsweep(self._root, 0.0)

    def _solve_downsweep(self, comp: _Compartment, leaf_iter):
        r"""Recursive down-sweep from leaves to root."""
        g_val, f_val = comp.io()

        if comp.parent is not None:
            parent = comp.parent
            parent.gather_input(g_val, f_val)
            parent.n_passed += 1

            if parent.n_passed == len(parent.children):
                parent.n_passed = 0
                self._solve_downsweep(parent, leaf_iter)
            else:
                try:
                    next_leaf = next(leaf_iter)
                    self._solve_downsweep(next_leaf, leaf_iter)
                except StopIteration:
                    pass

    def _solve_upsweep(self, comp: _Compartment, v_in: float):
        r"""Recursive up-sweep from root to leaves."""
        v_new = comp.calc_v(v_in)
        for child in comp.children:
            self._solve_upsweep(child, v_new)
