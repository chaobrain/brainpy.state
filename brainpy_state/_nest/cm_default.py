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

"""
Multi-compartment neuron model (``cm_default``), compatible with NEST.

This module implements a compartmental neuron model with a user-defined
dendritic tree structure. Each compartment supports optional Na/K ion
channels and AMPA/GABA/NMDA/AMPA_NMDA synaptic receptors.

The numerical integration uses the Crank-Nicolson scheme with an O(n)
tree-based matrix solver, matching NEST's implementation exactly.
"""

import math
from typing import Optional, Dict, List, Any

import numpy as np

__all__ = [
    'cm_default',
]


# ---------------------------------------------------------------------------
# Ion channel classes
# ---------------------------------------------------------------------------

class _NaChannel:
    """Sodium channel, Hodgkin-Huxley style kinetics.

    Based on ModelDB entry 140828 (Branco 2010), from Huguenard et al.
    (1988) and Hamill et al. (1991).

    Parameters
    ----------
    gbar_Na : float
        Maximal conductance [uS]. Default 0.0 (inactive).
    e_Na : float
        Reversal potential [mV]. Default 50.0.
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
        """Advance channel state one timestep.

        Returns (g_val, i_val) for Crank-Nicolson matrix.
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
    """Potassium channel, Hodgkin-Huxley style kinetics.

    Based on ModelDB entry 140828 (Branco 2010), from Sah et al.
    and Hamill et al. (1991).

    Note: conductance is linear in n (not n^4).

    Parameters
    ----------
    gbar_K : float
        Maximal conductance [uS]. Default 0.0 (inactive).
    e_K : float
        Reversal potential [mV]. Default -85.0.
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
        """Advance channel state one timestep.

        Returns (g_val, i_val) for Crank-Nicolson matrix.
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
    """Compute normalization constant for dual-exponential conductance."""
    tp = (tau_r * tau_d) / (tau_d - tau_r) * math.log(tau_d / tau_r)
    return 1.0 / (-math.exp(-tp / tau_r) + math.exp(-tp / tau_d))


class _AMPAReceptor:
    """AMPA receptor with dual-exponential conductance kinetics.

    Parameters
    ----------
    e_AMPA : float
        Reversal potential [mV]. Default 0.0.
    tau_r_AMPA : float
        Rise time constant [ms]. Default 0.2.
    tau_d_AMPA : float
        Decay time constant [ms]. Default 3.0.
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
        self.prop_r = math.exp(-dt / self.tau_r)
        self.prop_d = math.exp(-dt / self.tau_d)

    def f_numstep(self, v_comp: float, spike_weight: float):
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
    """GABA receptor with dual-exponential conductance kinetics.

    Parameters
    ----------
    e_GABA : float
        Reversal potential [mV]. Default -80.0.
    tau_r_GABA : float
        Rise time constant [ms]. Default 0.2.
    tau_d_GABA : float
        Decay time constant [ms]. Default 10.0.
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
        self.prop_r = math.exp(-dt / self.tau_r)
        self.prop_d = math.exp(-dt / self.tau_d)

    def f_numstep(self, v_comp: float, spike_weight: float):
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
    """Mg2+ block function and its voltage derivative."""
    exp_v = math.exp(-0.1 * v_comp)
    denom = 1.0 + 0.3 * exp_v
    B = 1.0 / denom
    dB_dv = 0.03 * exp_v / (denom ** 2)
    return B, dB_dv


class _NMDAReceptor:
    """NMDA receptor with dual-exponential kinetics and Mg2+ block.

    Parameters
    ----------
    e_NMDA : float
        Reversal potential [mV]. Default 0.0.
    tau_r_NMDA : float
        Rise time constant [ms]. Default 0.2.
    tau_d_NMDA : float
        Decay time constant [ms]. Default 43.0.
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
        self.prop_r = math.exp(-dt / self.tau_r)
        self.prop_d = math.exp(-dt / self.tau_d)

    def f_numstep(self, v_comp: float, spike_weight: float):
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
    """Combined AMPA+NMDA receptor with shared reversal potential.

    Parameters
    ----------
    e_AMPA_NMDA : float
        Shared reversal potential [mV]. Default 0.0.
    tau_r_AMPA : float
        AMPA rise time [ms]. Default 0.2.
    tau_d_AMPA : float
        AMPA decay time [ms]. Default 3.0.
    tau_r_NMDA : float
        NMDA rise time [ms]. Default 0.2.
    tau_d_NMDA : float
        NMDA decay time [ms]. Default 43.0.
    NMDA_ratio : float
        Ratio of NMDA vs AMPA conductance. Default 2.0.
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
        self.prop_r_AMPA = math.exp(-dt / self.tau_r_AMPA)
        self.prop_d_AMPA = math.exp(-dt / self.tau_d_AMPA)
        self.prop_r_NMDA = math.exp(-dt / self.tau_r_NMDA)
        self.prop_d_NMDA = math.exp(-dt / self.tau_d_NMDA)

    def f_numstep(self, v_comp: float, spike_weight: float):
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
    """A single compartment in the dendritic tree.

    Parameters
    ----------
    comp_index : int
        Index of this compartment.
    parent_index : int
        Index of parent compartment (-1 for root).
    C_m : float
        Membrane capacitance [nF]. Default 1.0.
    g_C : float
        Coupling conductance to parent [uS]. Default 0.01.
    g_L : float
        Leak conductance [uS]. Default 0.1.
    e_L : float
        Leak reversal potential [mV]. Default -70.0.
    v_comp : float or None
        Initial voltage [mV]. Default: e_L.
    gbar_Na : float
        Na channel maximal conductance [uS]. Default 0.0.
    e_Na : float
        Na reversal potential [mV]. Default 50.0.
    gbar_K : float
        K channel maximal conductance [uS]. Default 0.0.
    e_K : float
        K reversal potential [mV]. Default -85.0.
    """

    def __init__(self, comp_index: int, parent_index: int, params: Optional[Dict] = None):
        self.comp_index = comp_index
        self.p_index = parent_index
        self.parent: Optional['_Compartment'] = None
        self.children: List['_Compartment'] = []

        # Electrical parameters (defaults match NEST)
        self.ca = 1.0       # C_m [nF]
        self.gc = 0.01      # g_C [uS]
        self.gl = 0.1       # g_L [uS]
        self.el = -70.0     # e_L [mV]
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
        """Add a synaptic receptor to this compartment."""
        self.receptors.append(receptor)

    def pre_run_hook(self, dt: float):
        """Compute constants for numerical integration."""
        self.ca__div__dt = self.ca / dt
        self.gl__div__2 = self.gl / 2.0
        self.gg0 = self.ca__div__dt + self.gl__div__2
        self.gc__div__2 = self.gc / 2.0
        self.gl__times__el = self.gl * self.el

        for rec in self.receptors:
            rec.pre_run_hook(dt)

    def construct_matrix_element(self, dt: float, spike_buffers: Dict[int, float]):
        """Build matrix row for this compartment.

        Parameters
        ----------
        dt : float
            Simulation timestep [ms].
        spike_buffers : dict
            Maps receptor global index -> spike weight for this lag.
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
        """Accumulate input from a child compartment during down-sweep."""
        self._xx += g_val
        self._yy += f_val

    def io(self):
        """Compute input-output transformation for down-sweep.

        Returns (g_val, f_val) to pass to parent.
        """
        self.gg -= self._xx
        self.ff -= self._yy

        g_val = self.hh * self.hh / self.gg
        f_val = self.ff * self.hh / self.gg
        return g_val, f_val

    def calc_v(self, v_in: float) -> float:
        """Compute new voltage during up-sweep.

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
# Main cm_default class
# ---------------------------------------------------------------------------

class cm_default:
    r"""Summary
    -------
    
    Multi-compartment neuron model with user-defined dendrite structure.
    
    Parameters
    ----------
    
    V_th : float
        Spike threshold [mV]. Default -55.0.
    
    Compartment parameters (passed in ``add_compartment``):
    
    =========== ======= ====================================================
    ``C_m``     nF      Capacitance of compartment (default 1.0)
    ``g_C``     µS      Coupling conductance with parent (default 0.01)
    ``g_L``     µS      Leak conductance (default 0.1)
    ``e_L``     mV      Leak reversal potential (default -70.0)
    ``v_comp``  mV      Initial voltage (default: e_L)
    ``gbar_Na`` µS      Na channel maximal conductance (default 0.0)
    ``e_Na``    mV      Na reversal potential (default 50.0)
    ``gbar_K``  µS      K channel maximal conductance (default 0.0)
    ``e_K``     mV      K reversal potential (default -85.0)
    =========== ======= ====================================================
    
    Receptor parameters (passed in ``add_receptor``):
    
    AMPA: ``e_AMPA`` (0 mV), ``tau_r_AMPA`` (0.2 ms), ``tau_d_AMPA`` (3.0 ms)
    
    GABA: ``e_GABA`` (-80 mV), ``tau_r_GABA`` (0.2 ms), ``tau_d_GABA`` (10.0 ms)
    
    NMDA: ``e_NMDA`` (0 mV), ``tau_r_NMDA`` (0.2 ms), ``tau_d_NMDA`` (43.0 ms)
    
    AMPA_NMDA: ``e_AMPA_NMDA`` (0 mV), ``tau_r_AMPA`` (0.2 ms),
    ``tau_d_AMPA`` (3.0 ms), ``tau_r_NMDA`` (0.2 ms),
    ``tau_d_NMDA`` (43.0 ms), ``NMDA_ratio`` (2.0)
    
    Raises
    ------
    
    ValueError
        Raised when one of the following conditions is violated:
        - No compartments have been added.
        - Root compartment already exists.
    
    See Also
    --------
    
    hh_psc_alpha : Hodgkin-Huxley neuron (single compartment)
    iaf_cond_alpha : IAF conductance-based neuron
    
    Notes
    -----
    
    Description
    ...........
    
    ``cm_default`` is a compartmental neuron model whose structure -- soma,
    dendrites, axon -- is defined at runtime by adding compartments. Each
    compartment can be assigned synaptic receptors (AMPA, GABA, NMDA, or
    combined AMPA_NMDA).
    
    This is a brainpy.state re-implementation of the NEST simulator model of
    the same name. The implementation faithfully reproduces NEST's:
    
    * Crank-Nicolson integration scheme for the cable equation,
    * O(n) tree-based matrix solver (down-sweep / up-sweep),
    * Exact exponential integration for ion channel gating variables,
    * Dual-exponential conductance kinetics for synaptic receptors,
    * Spike detection via threshold crossing at the root (soma) compartment.
    
    The model is passive by default. Sodium and potassium channels can be
    activated by providing non-zero ``gbar_Na`` and ``gbar_K`` in the
    compartment parameters.
    
    Cable equation
    ..............
    
    For compartment :math:`i`, the membrane potential evolves according to:
    
    .. math::
    
        C_m^{(i)} \frac{dV^{(i)}}{dt} =
            -g_L^{(i)} (V^{(i)} - e_L^{(i)})
            - g_C^{(i)} (V^{(i)} - V^{(\text{parent})})
            - \sum_j g_C^{(j)} (V^{(i)} - V^{(j)})
            + I_{\text{Na}}^{(i)} + I_{\text{K}}^{(i)}
            + I_{\text{syn}}^{(i)} + I_{\text{ext}}^{(i)}
    
    where the sum is over child compartments :math:`j`.
    
    Crank-Nicolson discretization
    .............................
    
    The implicit trapezoidal rule is used:
    
    .. math::
    
        \frac{V^{(i)}_{\text{new}} - V^{(i)}_{\text{old}}}{\Delta t}
        = \frac{F(V_{\text{new}}) + F(V_{\text{old}})}{2}
    
    yielding a tridiagonal-like system on the tree that is solved in
    O(n) via a two-pass algorithm (down-sweep from leaves to root,
    up-sweep from root to leaves).
    
    Spike detection
    ...............
    
    A spike is emitted when the root (soma) compartment voltage crosses
    the threshold :math:`V_{\text{th}}` from below:
    
    .. math::
    
        V_{\text{soma}}^{\text{old}} < V_{\text{th}}
        \quad\text{and}\quad
        V_{\text{soma}}^{\text{new}} \geq V_{\text{th}}
    
    No explicit voltage reset is performed; repolarization relies on
    the Na/K channel dynamics.
    
    Ion channels
    ............
    
    **Sodium (Na)** -- Hodgkin-Huxley style with :math:`m^3 h` gating:
    
    .. math::
    
        I_{\text{Na}} = \bar{g}_{\text{Na}} \, m^3 h \, (e_{\text{Na}} - V)
    
    Gating variables are advanced using exact exponential integration.
    
    **Potassium (K)** -- with linear :math:`n` gating:
    
    .. math::
    
        I_{\text{K}} = \bar{g}_{\text{K}} \, n \, (e_{\text{K}} - V)
    
    Synaptic receptors
    ..................
    
    All receptors use dual-exponential conductance kinetics with rise
    time :math:`\tau_r` and decay time :math:`\tau_d`. NMDA receptors
    include a voltage-dependent Mg²⁺ block:
    
    .. math::
    
        B(V) = \frac{1}{1 + 0.3 \, e^{-0.1 V}}
    
    Unit consistency
    ................
    
    The model does not enforce unit consistency. Voltages are in [mV] and
    times in [ms]. If conductances are in [µS], capacitances should be in
    [nF], currents in [nA], and synaptic weights in [µS].
    
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

    def __init__(self, V_th: float = -55.0):
        self.V_th = V_th

        # Tree structure
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

    def add_compartment(self, parent_idx: int, params: Optional[Dict] = None) -> int:
        """Add a compartment to the neuron tree.

        Parameters
        ----------
        parent_idx : int
            Index of the parent compartment. Use -1 for the root (soma).
        params : dict, optional
            Compartment parameters (C_m, g_C, g_L, e_L, v_comp, gbar_Na,
            e_Na, gbar_K, e_K).

        Returns
        -------
        int
            Index of the newly added compartment.
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
        """Add a synaptic receptor to a compartment.

        Parameters
        ----------
        comp_idx : int
            Index of the target compartment.
        receptor_type : str
            One of 'AMPA', 'GABA', 'NMDA', 'AMPA_NMDA'.
        params : dict, optional
            Receptor parameters.

        Returns
        -------
        int
            Global receptor index (used for spike delivery).
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
        """Initialize the model for simulation.

        Must be called after adding all compartments and receptors,
        before stepping.

        Parameters
        ----------
        dt : float
            Simulation timestep [ms].
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
        """Add a spike to a receptor's buffer for the next timestep.

        Parameters
        ----------
        receptor_idx : int
            Global receptor index (returned by ``add_receptor``).
        weight : float
            Synaptic weight.
        """
        rec = self._receptors[receptor_idx]
        key = id(rec)
        self._spike_buffer[key] = self._spike_buffer.get(key, 0.0) + weight

    def add_current(self, comp_idx: int, current: float):
        """Add an external current to a compartment for the next timestep.

        Parameters
        ----------
        comp_idx : int
            Compartment index.
        current : float
            Current amplitude [nA].
        """
        comp = self._get_compartment(comp_idx)
        comp._current_buffer.append(current)

    def step(self) -> bool:
        """Advance the simulation by one timestep.

        Returns
        -------
        bool
            True if a spike was detected at the soma.
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
        """Get the current voltage of a compartment.

        Parameters
        ----------
        comp_idx : int
            Compartment index.

        Returns
        -------
        float
            Membrane voltage [mV].
        """
        return self._get_compartment(comp_idx).v_comp

    def get_voltages(self) -> List[float]:
        """Get voltages of all compartments.

        Returns
        -------
        list of float
            Voltages in compartment index order.
        """
        return [comp.v_comp for comp in self._compartments]

    def get_na_state(self, comp_idx: int):
        """Get Na channel state variables for a compartment.

        Returns (m_Na, h_Na).
        """
        comp = self._get_compartment(comp_idx)
        return comp.na_chan.m_Na, comp.na_chan.h_Na

    def get_k_state(self, comp_idx: int):
        """Get K channel state variables for a compartment.

        Returns n_K.
        """
        comp = self._get_compartment(comp_idx)
        return comp.k_chan.n_K

    def get_receptor_state(self, receptor_idx: int):
        """Get receptor conductance state variables.

        Returns a dict of state variable names -> values.
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
        """Find compartment by index via search."""
        if comp_idx < 0 or comp_idx >= len(self._compartments):
            return None
        return self._compartments[comp_idx]

    def _update_compartment_list(self):
        """Rebuild flat list of compartments in index order."""
        self._compartments = []
        for idx in self._compartment_indices:
            comp = self._find_compartment(self._root, idx)
            if comp is not None:
                self._compartments.append(comp)

    def _find_compartment(self, comp: Optional[_Compartment],
                          target_idx: int) -> Optional[_Compartment]:
        """Recursively find compartment by index."""
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
        """Set parent pointers for all compartments."""
        for comp in self._compartments:
            if comp.p_index >= 0:
                parent = self._get_compartment(comp.p_index)
                comp.parent = parent
            else:
                comp.parent = None

    def _set_leafs(self):
        """Identify leaf compartments (no children)."""
        self._leafs = [
            comp for comp in self._compartments
            if len(comp.children) == 0
        ]

    def _solve_matrix(self):
        """Solve the tridiagonal-on-tree system using O(n) algorithm."""
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
        """Recursive down-sweep from leaves to root."""
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
        """Recursive up-sweep from root to leaves."""
        v_new = comp.calc_v(v_in)
        for child in comp.children:
            self._solve_upsweep(child, v_new)
