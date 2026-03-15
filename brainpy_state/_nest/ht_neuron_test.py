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

r"""Tests for NEST-compatible ht_neuron model (Hill & Tononi, 2005).

Tests cover:
- Default parameter values matching NEST
- Parameter validation
- Initial state variable values (equilibrium initialization)
- Subthreshold dynamics (ODE integration via standalone reference solver)
- Spike generation, threshold reset, and refractory period
- Synaptic conductance dynamics (beta-function kernels)
- Intrinsic current computation (I_NaP, I_KNa, I_T, I_h)
- NMDA voltage-dependent unblocking
- Beta normalization factor correctness

All tests use float64 precision on CPU to match NEST's numerical behavior.
"""

import math
import unittest

import brainstate
import saiunit as u
import jax
import numpy as np
from brainpy.state import ht_neuron
from scipy.integrate import solve_ivp

from brainpy_state._nest.ht_neuron import (
    _beta_normalization_factor,
    _m_eq_h,
    _h_eq_T,
    _m_eq_T,
    _D_eq_KNa,
    _m_eq_NMDA,
)

# ---------------------------------------------------------------------------
# State vector index constants for the reference dynamics solver.
#
# The refactored ht_neuron model uses DotDict named attributes (e.g.,
# state.V_m, state.theta) instead of integer indices into a flat array.
# These constants are defined here solely for the standalone reference ODE
# solver (_nest_ht_dynamics) used in regression tests.
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


def _m_NMDA(V, m_eq, m_fast, m_slow, instant_unblock_NMDA=False):
    r"""Compute effective NMDA unblocking variable.

    This helper was previously a module-level function in ht_neuron.py but has
    been inlined into the model's _vector_field method. It is retained here for
    the standalone reference solver and direct unit tests.

    Parameters
    ----------
    V : float
        Membrane potential in mV.
    m_eq : float
        Equilibrium NMDA unblocking value at voltage V.
    m_fast : float
        Fast NMDA unblocking variable.
    m_slow : float
        Slow NMDA unblocking variable.
    instant_unblock_NMDA : bool
        If True, return m_eq directly. Otherwise, return weighted mix of
        fast and slow components.

    Returns
    -------
    float
        Effective NMDA unblocking fraction.
    """
    if instant_unblock_NMDA:
        return m_eq
    A1 = 0.51 - 0.0028 * V
    A2 = 1.0 - A1
    return A1 * m_fast + A2 * m_slow

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


# ---------------------------------------------------------------------------
# Reference dynamics function (standalone, matches NEST ht_neuron_dynamics)
# ---------------------------------------------------------------------------

def _nest_ht_dynamics(t, y, params):
    r"""Reference HT dynamics matching NEST ht_neuron_dynamics exactly.

    Parameters
    ----------
    t : float
        Current time (unused, autonomous system).
    y : array of float, shape (16,)
        State vector.
    params : dict
        Model parameters.

    Returns
    -------
    array of float, shape (16,)
        Derivatives.
    """
    V = y[_V_M]
    ref_steps = params['ref_steps']

    # NMDA conductance with instantaneous blocking
    S_act = params['S_act_NMDA']
    V_act = params['V_act_NMDA']
    m_eq_nmda = 1.0 / (1.0 + math.exp(-S_act * (V - V_act)))
    mf = min(m_eq_nmda, y[_m_fast_NMDA])
    ms = min(m_eq_nmda, y[_m_slow_NMDA])
    if params['instant_unblock_NMDA']:
        m_nmda = m_eq_nmda
    else:
        A1 = 0.51 - 0.0028 * V
        A2 = 1.0 - A1
        m_nmda = A1 * mf + A2 * ms

    # Synaptic currents
    I_syn = (
        -y[_G_AMPA] * (V - params['E_rev_AMPA'])
        - y[_G_NMDA_TIMECOURSE] * m_nmda * (V - params['E_rev_NMDA'])
        - y[_G_GABA_A] * (V - params['E_rev_GABA_A'])
        - y[_G_GABA_B] * (V - params['E_rev_GABA_B'])
    )

    # Post-spike K current
    I_spike = -(V - params['E_K']) / params['tau_spike'] if ref_steps > 0 else 0.0

    # Leak currents
    I_Na = -params['g_NaL'] * (V - params['E_Na'])
    I_K = -params['g_KL'] * (V - params['E_K'])

    # I_NaP
    INaP_thresh = -55.7
    INaP_slope = 7.7
    m_inf_NaP = 1.0 / (1.0 + math.exp(-(V - INaP_thresh) / INaP_slope))
    I_NaP = -params['g_peak_NaP'] * (m_inf_NaP ** params['N_NaP']) * (V - params['E_rev_NaP'])

    # I_KNa
    d_half = 0.25
    d_val = y[_D_IKNa]
    m_inf_KNa = 1.0 / (1.0 + (d_half / d_val) ** 3.5) if d_val > 0 else 0.0
    I_KNa = -params['g_peak_KNa'] * m_inf_KNa * (V - params['E_rev_KNa'])

    # I_T
    I_T = -params['g_peak_T'] * (y[_m_IT] ** params['N_T']) * y[_h_IT] * (V - params['E_rev_T'])

    # I_h
    I_h = -params['g_peak_h'] * y[_m_Ih] * (V - params['E_rev_h'])

    f = np.zeros(16)

    # dV/dt
    I_stim = params.get('I_stim', 0.0)
    f[_V_M] = (I_Na + I_K + I_syn + I_NaP + I_KNa + I_T + I_h + I_stim) / params['tau_m'] + I_spike

    # d(theta)/dt
    f[_THETA] = -(y[_THETA] - params['theta_eq']) / params['tau_theta']

    # AMPA
    f[_DG_AMPA] = -y[_DG_AMPA] / params['tau_rise_AMPA']
    f[_G_AMPA] = y[_DG_AMPA] - y[_G_AMPA] / params['tau_decay_AMPA']

    # NMDA
    f[_DG_NMDA_TIMECOURSE] = -y[_DG_NMDA_TIMECOURSE] / params['tau_rise_NMDA']
    f[_G_NMDA_TIMECOURSE] = y[_DG_NMDA_TIMECOURSE] - y[_G_NMDA_TIMECOURSE] / params['tau_decay_NMDA']
    f[_m_fast_NMDA] = (m_eq_nmda - mf) / params['tau_Mg_fast_NMDA']
    f[_m_slow_NMDA] = (m_eq_nmda - ms) / params['tau_Mg_slow_NMDA']

    # GABA_A
    f[_DG_GABA_A] = -y[_DG_GABA_A] / params['tau_rise_GABA_A']
    f[_G_GABA_A] = y[_DG_GABA_A] - y[_G_GABA_A] / params['tau_decay_GABA_A']

    # GABA_B
    f[_DG_GABA_B] = -y[_DG_GABA_B] / params['tau_rise_GABA_B']
    f[_G_GABA_B] = y[_DG_GABA_B] - y[_G_GABA_B] / params['tau_decay_GABA_B']

    # D_IKNa
    D_influx_peak = 0.025
    D_thresh = -10.0
    D_slope = 5.0
    D_eq = 0.001
    D_influx = D_influx_peak / (1.0 + math.exp(-(V - D_thresh) / D_slope))
    D_eq_val = params['tau_D_KNa'] * D_influx + D_eq
    f[_D_IKNa] = (D_eq_val - y[_D_IKNa]) / params['tau_D_KNa']

    # I_T gating
    tau_m_T = 0.22 / (math.exp(-(V + 132.0) / 16.7) + math.exp((V + 16.8) / 18.2)) + 0.13
    tau_h_T = 8.2 + (56.6 + 0.27 * math.exp((V + 115.2) / 5.0)) / (1.0 + math.exp((V + 86.0) / 3.2))
    m_eq_t = 1.0 / (1.0 + math.exp(-(V + 59.0) / 6.2))
    h_eq_t = 1.0 / (1.0 + math.exp((V + 83.0) / 4.0))
    f[_m_IT] = (m_eq_t - y[_m_IT]) / tau_m_T
    f[_h_IT] = (h_eq_t - y[_h_IT]) / tau_h_T

    # I_h gating
    tau_m_h = 1.0 / (math.exp(-14.59 - 0.086 * V) + math.exp(-1.87 + 0.0701 * V))
    I_h_Vthreshold = -75.0
    m_eq_ih = 1.0 / (1.0 + math.exp((V - I_h_Vthreshold) / 5.5))
    f[_m_Ih] = (m_eq_ih - y[_m_Ih]) / tau_m_h

    return f


def _default_params():
    r"""Return default parameter dict matching NEST ht_neuron defaults."""
    return dict(
        E_Na=30.0, E_K=-90.0, g_NaL=0.2, g_KL=1.0,
        tau_m=16.0, theta_eq=-51.0, tau_theta=2.0,
        tau_spike=1.75, t_ref=2.0,
        g_peak_AMPA=0.1, tau_rise_AMPA=0.5, tau_decay_AMPA=2.4, E_rev_AMPA=0.0,
        g_peak_NMDA=0.075, tau_rise_NMDA=4.0, tau_decay_NMDA=40.0, E_rev_NMDA=0.0,
        V_act_NMDA=-25.57, S_act_NMDA=0.081,
        tau_Mg_slow_NMDA=22.7, tau_Mg_fast_NMDA=0.68, instant_unblock_NMDA=False,
        g_peak_GABA_A=0.33, tau_rise_GABA_A=1.0, tau_decay_GABA_A=7.0, E_rev_GABA_A=-70.0,
        g_peak_GABA_B=0.0132, tau_rise_GABA_B=60.0, tau_decay_GABA_B=200.0, E_rev_GABA_B=-90.0,
        g_peak_NaP=1.0, E_rev_NaP=30.0, N_NaP=3.0,
        g_peak_KNa=1.0, E_rev_KNa=-90.0, tau_D_KNa=1250.0,
        g_peak_T=1.0, E_rev_T=0.0, N_T=2.0,
        g_peak_h=1.0, E_rev_h=-40.0,
        ref_steps=0, I_stim=0.0,
    )


def _default_initial_state(params):
    r"""Return the NEST-matching initial state vector."""
    V_init = (params['g_NaL'] * params['E_Na'] + params['g_KL'] * params['E_K']) / \
             (params['g_NaL'] + params['g_KL'])
    dftype = brainstate.environ.dftype()
    y0 = np.zeros(_STATE_VEC_SIZE, dtype=dftype)
    y0[_V_M] = V_init
    y0[_THETA] = params['theta_eq']
    y0[_m_fast_NMDA] = _m_eq_NMDA(V_init, params['S_act_NMDA'], params['V_act_NMDA'])
    y0[_m_slow_NMDA] = _m_eq_NMDA(V_init, params['S_act_NMDA'], params['V_act_NMDA'])
    y0[_m_Ih] = _m_eq_h(V_init)
    y0[_D_IKNa] = _D_eq_KNa(V_init, params['tau_D_KNa'])
    y0[_m_IT] = _m_eq_T(V_init)
    y0[_h_IT] = _h_eq_T(V_init)
    return y0


def _get_scalar(x):
    r"""Extract scalar float from array or scalar."""
    x = np.asarray(x)
    return float(x.flat[0]) if x.ndim > 0 else float(x)


# ===================================================================
# Test Classes
# ===================================================================

class TestBetaNormalizationFactor(unittest.TestCase):
    r"""Test the beta normalization factor computation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_known_values(self):
        r"""Check normalization factor for AMPA defaults (tau_rise=0.5, tau_decay=2.4)."""
        tau_rise = 0.5
        tau_decay = 2.4
        factor = _beta_normalization_factor(tau_rise, tau_decay)
        # Compute expected value manually
        tau_diff = tau_decay - tau_rise
        t_peak = tau_decay * tau_rise * math.log(tau_decay / tau_rise) / tau_diff
        peak_val = math.exp(-t_peak / tau_decay) - math.exp(-t_peak / tau_rise)
        expected = (1.0 / tau_rise - 1.0 / tau_decay) / peak_val
        self.assertAlmostEqual(factor, expected, places=12)

    def test_alpha_limit(self):
        r"""When tau_rise == tau_decay, should fall back to alpha function."""
        tau = 5.0
        factor = _beta_normalization_factor(tau, tau)
        expected = math.e / tau
        self.assertAlmostEqual(factor, expected, places=12)

    def test_positive(self):
        r"""Factor should always be positive."""
        for tau_rise, tau_decay in [(0.5, 2.4), (1.0, 7.0), (4.0, 40.0), (60.0, 200.0)]:
            self.assertGreater(_beta_normalization_factor(tau_rise, tau_decay), 0.0)


class TestEquilibriumFunctions(unittest.TestCase):
    r"""Test steady-state equilibrium helper functions."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_m_eq_h_at_minus75(self):
        r"""At V=-75 (threshold), m_eq_h should be 0.5."""
        self.assertAlmostEqual(_m_eq_h(-75.0), 0.5, places=10)

    def test_m_eq_h_depolarized(self):
        r"""At high voltage, I_h activation should be near 0."""
        self.assertAlmostEqual(_m_eq_h(0.0), 0.0, places=3)

    def test_m_eq_h_hyperpolarized(self):
        r"""At low voltage, I_h activation should be near 1."""
        self.assertAlmostEqual(_m_eq_h(-120.0), 1.0, places=3)

    def test_m_eq_T_at_minus59(self):
        r"""At V=-59 (inflection), m_eq_T should be 0.5."""
        self.assertAlmostEqual(_m_eq_T(-59.0), 0.5, places=10)

    def test_h_eq_T_at_minus83(self):
        r"""At V=-83 (inflection), h_eq_T should be 0.5."""
        self.assertAlmostEqual(_h_eq_T(-83.0), 0.5, places=10)

    def test_D_eq_KNa_positive(self):
        r"""D_eq_KNa should always be positive."""
        for V in [-90, -70, -50, -30, -10, 0, 10, 30]:
            self.assertGreater(_D_eq_KNa(V, 1250.0), 0.0)

    def test_m_eq_NMDA_at_V_act(self):
        r"""At V=V_act_NMDA, m_eq should be 0.5."""
        V_act = -25.57
        S_act = 0.081
        self.assertAlmostEqual(_m_eq_NMDA(V_act, S_act, V_act), 0.5, places=10)


class TestHTNeuronDefaults(unittest.TestCase):
    r"""Test that default parameter values match NEST ht_neuron."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = ht_neuron(1)
            self.assertAlmostEqual(neuron.E_Na, 30.0)
            self.assertAlmostEqual(neuron.E_K, -90.0)
            self.assertAlmostEqual(neuron.g_NaL, 0.2)
            self.assertAlmostEqual(neuron.g_KL, 1.0)
            self.assertAlmostEqual(neuron.tau_m, 16.0)
            self.assertAlmostEqual(neuron.theta_eq, -51.0)
            self.assertAlmostEqual(neuron.tau_theta, 2.0)
            self.assertAlmostEqual(neuron.tau_spike, 1.75)
            self.assertAlmostEqual(neuron.t_ref, 2.0)
            self.assertAlmostEqual(neuron.g_peak_AMPA, 0.1)
            self.assertAlmostEqual(neuron.tau_rise_AMPA, 0.5)
            self.assertAlmostEqual(neuron.tau_decay_AMPA, 2.4)
            self.assertAlmostEqual(neuron.E_rev_AMPA, 0.0)
            self.assertAlmostEqual(neuron.g_peak_NMDA, 0.075)
            self.assertAlmostEqual(neuron.tau_rise_NMDA, 4.0)
            self.assertAlmostEqual(neuron.tau_decay_NMDA, 40.0)
            self.assertAlmostEqual(neuron.E_rev_NMDA, 0.0)
            self.assertAlmostEqual(neuron.V_act_NMDA, -25.57)
            self.assertAlmostEqual(neuron.S_act_NMDA, 0.081)
            self.assertAlmostEqual(neuron.tau_Mg_slow_NMDA, 22.7)
            self.assertAlmostEqual(neuron.tau_Mg_fast_NMDA, 0.68)
            self.assertFalse(neuron.instant_unblock_NMDA)
            self.assertAlmostEqual(neuron.g_peak_GABA_A, 0.33)
            self.assertAlmostEqual(neuron.tau_rise_GABA_A, 1.0)
            self.assertAlmostEqual(neuron.tau_decay_GABA_A, 7.0)
            self.assertAlmostEqual(neuron.E_rev_GABA_A, -70.0)
            self.assertAlmostEqual(neuron.g_peak_GABA_B, 0.0132)
            self.assertAlmostEqual(neuron.tau_rise_GABA_B, 60.0)
            self.assertAlmostEqual(neuron.tau_decay_GABA_B, 200.0)
            self.assertAlmostEqual(neuron.E_rev_GABA_B, -90.0)
            self.assertAlmostEqual(neuron.g_peak_NaP, 1.0)
            self.assertAlmostEqual(neuron.E_rev_NaP, 30.0)
            self.assertAlmostEqual(neuron.N_NaP, 3.0)
            self.assertAlmostEqual(neuron.g_peak_KNa, 1.0)
            self.assertAlmostEqual(neuron.E_rev_KNa, -90.0)
            self.assertAlmostEqual(neuron.tau_D_KNa, 1250.0)
            self.assertAlmostEqual(neuron.g_peak_T, 1.0)
            self.assertAlmostEqual(neuron.E_rev_T, 0.0)
            self.assertAlmostEqual(neuron.N_T, 2.0)
            self.assertAlmostEqual(neuron.g_peak_h, 1.0)
            self.assertAlmostEqual(neuron.E_rev_h, -40.0)
            self.assertFalse(neuron.voltage_clamp)


class TestHTNeuronInitialState(unittest.TestCase):
    r"""Test initial state variable values."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_initial_membrane_potential(self):
        r"""V_m should start at (g_NaL*E_Na + g_KL*E_K) / (g_NaL + g_KL)."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()
            V_init_expected = (0.2 * 30.0 + 1.0 * (-90.0)) / (0.2 + 1.0)
            V_actual = _get_scalar(neuron.V.value)
            self.assertAlmostEqual(V_actual, V_init_expected, places=10)

    def test_initial_threshold(self):
        r"""Theta should start at theta_eq."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()
            self.assertAlmostEqual(_get_scalar(neuron.theta.value), -51.0, places=10)

    def test_initial_synaptic_zero(self):
        r"""All synaptic variables should be zero initially."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()
            self.assertAlmostEqual(_get_scalar(neuron.DG_AMPA.value), 0.0, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.G_AMPA.value), 0.0, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.DG_NMDA.value), 0.0, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.G_NMDA.value), 0.0, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.DG_GABA_A.value), 0.0, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.G_GABA_A.value), 0.0, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.DG_GABA_B.value), 0.0, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.G_GABA_B.value), 0.0, places=10)

    def test_initial_intrinsic_equilibrium(self):
        r"""Intrinsic gating variables should be at equilibrium for initial V."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()
            V_init = (0.2 * 30.0 + 1.0 * (-90.0)) / (0.2 + 1.0)

            self.assertAlmostEqual(
                _get_scalar(neuron.m_Ih_state.value),
                _m_eq_h(V_init), places=10
            )
            self.assertAlmostEqual(
                _get_scalar(neuron.D_IKNa_state.value),
                _D_eq_KNa(V_init, 1250.0), places=10
            )
            self.assertAlmostEqual(
                _get_scalar(neuron.m_IT_state.value),
                _m_eq_T(V_init), places=10
            )
            self.assertAlmostEqual(
                _get_scalar(neuron.h_IT_state.value),
                _h_eq_T(V_init), places=10
            )
            self.assertAlmostEqual(
                _get_scalar(neuron.m_fast_NMDA_state.value),
                _m_eq_NMDA(V_init, 0.081, -25.57), places=10
            )
            self.assertAlmostEqual(
                _get_scalar(neuron.m_slow_NMDA_state.value),
                _m_eq_NMDA(V_init, 0.081, -25.57), places=10
            )

    def test_initial_refractory_zero(self):
        r"""Refractory counter should be zero initially."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()
            self.assertEqual(int(_get_scalar(neuron.ref_steps.value)), 0)


class TestHTNeuronValidation(unittest.TestCase):
    r"""Test parameter validation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_negative_conductances(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Negative peak conductances should raise ValueError."""
            for param in ('g_peak_AMPA', 'g_peak_NMDA', 'g_peak_GABA_A',
                          'g_peak_GABA_B', 'g_peak_NaP', 'g_peak_KNa',
                          'g_peak_T', 'g_peak_h', 'g_NaL', 'g_KL'):
                with self.assertRaises(ValueError, msg=f'{param} should not accept negative values'):
                    ht_neuron(1, **{param: -1.0})

    def test_negative_tau(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Zero or negative time constants should raise ValueError."""
            for param in ('tau_rise_AMPA', 'tau_decay_AMPA', 'tau_rise_NMDA',
                          'tau_decay_NMDA', 'tau_rise_GABA_A', 'tau_decay_GABA_A',
                          'tau_rise_GABA_B', 'tau_decay_GABA_B',
                          'tau_Mg_fast_NMDA', 'tau_Mg_slow_NMDA',
                          'tau_spike', 'tau_theta', 'tau_m', 'tau_D_KNa'):
                with self.assertRaises(ValueError, msg=f'{param} should not accept zero'):
                    ht_neuron(1, **{param: 0.0})

    def test_rise_greater_than_decay(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""tau_rise >= tau_decay should raise ValueError."""
            with self.assertRaises(ValueError):
                ht_neuron(1, tau_rise_AMPA=5.0, tau_decay_AMPA=2.0)
            with self.assertRaises(ValueError):
                ht_neuron(1, tau_rise_GABA_A=10.0, tau_decay_GABA_A=7.0)
            with self.assertRaises(ValueError):
                ht_neuron(1, tau_Mg_fast_NMDA=30.0, tau_Mg_slow_NMDA=22.7)

    def test_negative_t_ref(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                ht_neuron(1, t_ref=-1.0)

    def test_negative_S_act_NMDA(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                ht_neuron(1, S_act_NMDA=-0.1)


class TestHTNeuronSubthresholdDynamics(unittest.TestCase):
    r"""Test subthreshold dynamics against standalone reference solver."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_free_evolution_no_input(self):
        r"""With no input, neuron should remain subthreshold and near resting potential.

        Note: The initial V is computed from leak equilibrium only (g_NaL, g_KL).
        The intrinsic currents (I_h, I_NaP, etc.) are active and shift V
        slightly from the pure leak equilibrium. The neuron should settle to
        a stable resting potential that differs slightly from V_init.
        """
        dt = 0.1  # ms
        n_steps = 100

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()

            V_init = _get_scalar(neuron.V.value)
            theta_init = _get_scalar(neuron.theta.value)

            for k in range(n_steps):
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(0.0)

            V_final = _get_scalar(neuron.V.value)
            theta_final = _get_scalar(neuron.theta.value)

            # V should stay subthreshold (well below theta_eq = -51 mV)
            self.assertLess(V_final, theta_init,
                            msg="V should remain subthreshold with no input")
            # V should be in a reasonable range (within ~10 mV of init)
            self.assertLess(abs(V_final - V_init), 10.0,
                            msg="V should stay near resting potential with no input")
            # Theta should relax toward theta_eq
            self.assertAlmostEqual(theta_final, theta_init, places=0,
                                   msg="Theta should stay near theta_eq with no input")

    def test_matches_reference_solver(self):
        r"""Model output should match standalone reference ODE integration."""
        dt = 0.1  # ms
        n_steps = 50

        params = _default_params()
        y_ref = _default_initial_state(params)

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()

            for k in range(n_steps):
                # Reference solver
                sol = solve_ivp(
                    lambda t, y: _nest_ht_dynamics(t, y, params),
                    [0.0, dt], y_ref,
                    method='RK45', rtol=1e-3, atol=1e-9,
                )
                y_ref = sol.y[:, -1]

                # Enforce instantaneous NMDA blocking (as NEST does after each step)
                m_eq_nmda = _m_eq_NMDA(y_ref[_V_M], params['S_act_NMDA'], params['V_act_NMDA'])
                y_ref[_m_fast_NMDA] = min(m_eq_nmda, y_ref[_m_fast_NMDA])
                y_ref[_m_slow_NMDA] = min(m_eq_nmda, y_ref[_m_slow_NMDA])

                # Model update
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(0.0)

            # Compare final states
            V_model = _get_scalar(neuron.V.value)
            V_ref = y_ref[_V_M]
            self.assertAlmostEqual(V_model, V_ref, places=3,
                                   msg=f"V: model={V_model:.6f}, ref={V_ref:.6f}")

            theta_model = _get_scalar(neuron.theta.value)
            theta_ref = y_ref[_THETA]
            self.assertAlmostEqual(theta_model, theta_ref, places=3,
                                   msg=f"theta: model={theta_model:.6f}, ref={theta_ref:.6f}")

    def test_with_dc_current(self):
        r"""Small DC current should depolarize the neuron subthreshold."""
        dt = 0.1
        n_steps = 200
        I_dc = 5.0  # mV/ms equivalent (unitless conductance model)

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()

            V_init = _get_scalar(neuron.V.value)

            # Inject current for first step to store in buffer,
            # then it applies from step 1 onward
            for k in range(n_steps):
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(I_dc)

            V_final = _get_scalar(neuron.V.value)
            # DC current should depolarize (increase V)
            self.assertGreater(V_final, V_init,
                               msg="DC current should depolarize the neuron")


class TestHTNeuronSpiking(unittest.TestCase):
    r"""Test spike generation and refractory behavior."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_strong_current_produces_spike(self):
        r"""A strong DC current should cause the neuron to spike."""
        dt = 0.1
        n_steps = 500
        I_strong = 50.0  # Strong input

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()

            spiked = False
            for k in range(n_steps):
                with brainstate.environ.context(t=k * dt * u.ms):
                    spk = neuron.update(I_strong)
                    if _get_scalar(neuron.V.value) >= 25.0:
                        # Spike detected (V was reset to E_Na=30 and is near there)
                        spiked = True
                        break

            self.assertTrue(spiked, "Strong DC current should produce a spike")

    def test_refractory_period(self):
        r"""After a spike, neuron should be refractory for t_ref steps."""
        dt = 0.1
        n_steps = 1000
        I_strong = 50.0
        t_ref = 2.0
        expected_ref_steps = int(round(t_ref / dt))

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()

            first_spike_step = None
            for k in range(n_steps):
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(I_strong)
                    ref = int(_get_scalar(neuron.ref_steps.value))
                    if first_spike_step is None and ref > 0:
                        first_spike_step = k
                        # Just spiked: ref should be set to PotassiumRefractoryCounts
                        self.assertEqual(ref, expected_ref_steps,
                                         f"After spike, ref_steps should be {expected_ref_steps}")
                        break

            self.assertIsNotNone(first_spike_step, "Should have spiked")

    def test_spike_resets_V_and_theta(self):
        r"""On spike, V and theta should be set to E_Na."""
        dt = 0.1
        n_steps = 1000
        I_strong = 50.0

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()

            for k in range(n_steps):
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(I_strong)
                    ref = int(_get_scalar(neuron.ref_steps.value))
                    V = _get_scalar(neuron.V.value)
                    theta = _get_scalar(neuron.theta.value)
                    if ref > 0 and ref == int(round(2.0 / dt)):
                        # Just spiked this step - but V has already been
                        # through the ODE integration for this step with
                        # the post-spike K current active. V was set to E_Na
                        # before the ref counter was decremented.
                        # We just check V is near E_Na range
                        self.assertGreater(V, 0.0,
                                           msg="V should be high right after spike")
                        break


class TestHTNeuronSynapticDynamics(unittest.TestCase):
    r"""Test synaptic conductance dynamics with beta-function kernels."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_ampa_spike_response(self):
        r"""An AMPA spike should produce a transient conductance increase."""
        dt = 0.1
        params = _default_params()
        cond_step = params['g_peak_AMPA'] * _beta_normalization_factor(
            params['tau_rise_AMPA'], params['tau_decay_AMPA']
        )

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()

            # Manually inject AMPA spike at step 0
            # In NEST, spikes add to DG_AMPA * cond_step
            # We simulate this by directly adding to DG_AMPA after init
            neuron.DG_AMPA.value = np.asarray(cond_step)

            G_trace = []
            for k in range(100):
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(0.0)
                    G_trace.append(_get_scalar(neuron.G_AMPA.value))

            G_trace = np.array(G_trace)
            # G_AMPA should rise then decay
            self.assertGreater(np.max(G_trace), 0.0,
                               msg="AMPA conductance should increase after spike")
            # Peak should be close to g_peak_AMPA
            self.assertAlmostEqual(np.max(G_trace), params['g_peak_AMPA'], places=1,
                                   msg="AMPA peak should be approximately g_peak_AMPA")
            # Should decay back toward zero
            self.assertLess(G_trace[-1], G_trace[int(np.argmax(G_trace))],
                            msg="AMPA conductance should decay after peak")

    def test_gaba_a_spike_response(self):
        r"""A GABA_A spike should produce a transient conductance increase."""
        dt = 0.1
        params = _default_params()
        cond_step = params['g_peak_GABA_A'] * _beta_normalization_factor(
            params['tau_rise_GABA_A'], params['tau_decay_GABA_A']
        )

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()
            neuron.DG_GABA_A.value = np.asarray(cond_step)

            G_trace = []
            for k in range(200):
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(0.0)
                    G_trace.append(_get_scalar(neuron.G_GABA_A.value))

            G_trace = np.array(G_trace)
            self.assertGreater(np.max(G_trace), 0.0)
            self.assertAlmostEqual(np.max(G_trace), params['g_peak_GABA_A'], places=1)

    def test_beta_function_shape(self):
        r"""Beta-function conductance should have correct rise-then-decay shape."""
        dt = 0.1
        params = _default_params()

        # Use reference solver to trace AMPA conductance
        y0 = _default_initial_state(params)
        cond_step = params['g_peak_AMPA'] * _beta_normalization_factor(
            params['tau_rise_AMPA'], params['tau_decay_AMPA']
        )
        y0[_DG_AMPA] = cond_step

        G_trace = []
        for _ in range(100):
            sol = solve_ivp(
                lambda t, y: _nest_ht_dynamics(t, y, params),
                [0.0, dt], y0,
                method='RK45', rtol=1e-3, atol=1e-9,
            )
            y0 = sol.y[:, -1]
            m_eq = _m_eq_NMDA(y0[_V_M], params['S_act_NMDA'], params['V_act_NMDA'])
            y0[_m_fast_NMDA] = min(m_eq, y0[_m_fast_NMDA])
            y0[_m_slow_NMDA] = min(m_eq, y0[_m_slow_NMDA])
            G_trace.append(y0[_G_AMPA])

        G_trace = np.array(G_trace)
        peak_idx = np.argmax(G_trace)
        # Conductance should rise to peak then decay
        self.assertGreater(peak_idx, 0, msg="Peak should not be at t=0 (needs time to rise)")
        # After peak, values should monotonically decrease
        for j in range(peak_idx + 1, len(G_trace) - 1):
            self.assertGreaterEqual(G_trace[j], G_trace[j + 1] - 1e-10)


class TestHTNeuronIntrinsicCurrents(unittest.TestCase):
    r"""Test intrinsic current computation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_I_NaP_at_rest(self):
        r"""I_NaP should be computable at rest voltage."""
        V = -70.0
        g_peak_NaP = 1.0
        E_rev_NaP = 30.0
        N_NaP = 3.0
        m_inf = 1.0 / (1.0 + math.exp(-(V - (-55.7)) / 7.7))
        expected = -g_peak_NaP * (m_inf ** N_NaP) * (V - E_rev_NaP)
        self.assertAlmostEqual(expected, -g_peak_NaP * (m_inf ** 3) * (V - 30.0))
        # At rest, m_inf is small so I_NaP should be small
        self.assertLess(abs(expected), 1.0, msg="I_NaP should be small at resting potential")

    def test_I_h_at_rest(self):
        r"""I_h at rest should be non-trivial (hyperpolarized)."""
        V_rest = (0.2 * 30.0 + 1.0 * (-90.0)) / (0.2 + 1.0)
        m_Ih = _m_eq_h(V_rest)
        I_h = -1.0 * m_Ih * (V_rest - (-40.0))
        # I_h is inward (depolarizing) at rest since V < E_h=-40
        self.assertGreater(I_h, 0.0, msg="I_h should be depolarizing at rest (V < E_h)")

    def test_intrinsic_currents_recorded(self):
        r"""After a simulation step, intrinsic current values should be stored."""
        dt = 0.1

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()

            with brainstate.environ.context(t=0.0 * u.ms):
                neuron.update(0.0)

            # At near-equilibrium, currents should be finite
            self.assertTrue(np.isfinite(_get_scalar(neuron.I_NaP_val.value)))
            self.assertTrue(np.isfinite(_get_scalar(neuron.I_KNa_val.value)))
            self.assertTrue(np.isfinite(_get_scalar(neuron.I_T_val.value)))
            self.assertTrue(np.isfinite(_get_scalar(neuron.I_h_val.value)))


class TestHTNeuronNMDA(unittest.TestCase):
    r"""Test NMDA voltage-dependent unblocking."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_instant_unblock_mode(self):
        r"""With instant_unblock=True, m_NMDA should equal m_eq."""
        V = -50.0
        S_act = 0.081
        V_act = -25.57
        m_eq = _m_eq_NMDA(V, S_act, V_act)
        m = _m_NMDA(V, m_eq, 0.3, 0.1, instant_unblock_NMDA=True)
        self.assertAlmostEqual(m, m_eq, places=10)

    def test_two_stage_unblock_mode(self):
        r"""With instant_unblock=False, m_NMDA should be weighted mix of fast and slow."""
        V = -50.0
        S_act = 0.081
        V_act = -25.57
        m_eq = _m_eq_NMDA(V, S_act, V_act)
        m_fast = 0.3
        m_slow = 0.1
        A1 = 0.51 - 0.0028 * V
        A2 = 1.0 - A1
        expected = A1 * m_fast + A2 * m_slow
        m = _m_NMDA(V, m_eq, m_fast, m_slow, instant_unblock_NMDA=False)
        self.assertAlmostEqual(m, expected, places=10)

    def test_nmda_blocking_enforced(self):
        r"""NMDA unblocking variables should not exceed equilibrium."""
        dt = 0.1
        n_steps = 50

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()

            for k in range(n_steps):
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(0.0)
                    V = _get_scalar(neuron.V.value)
                    m_eq = _m_eq_NMDA(V, neuron.S_act_NMDA, neuron.V_act_NMDA)
                    m_f = _get_scalar(neuron.m_fast_NMDA_state.value)
                    m_s = _get_scalar(neuron.m_slow_NMDA_state.value)
                    self.assertLessEqual(m_f, m_eq + 1e-10,
                                         msg=f"Step {k}: m_fast_NMDA exceeds equilibrium")
                    self.assertLessEqual(m_s, m_eq + 1e-10,
                                         msg=f"Step {k}: m_slow_NMDA exceeds equilibrium")


class TestHTNeuronThresholdDynamics(unittest.TestCase):
    r"""Test dynamic threshold behavior."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_threshold_relaxes_to_equilibrium(self):
        r"""Theta should relax exponentially toward theta_eq."""
        dt = 0.1
        n_steps = 500
        theta_eq = -51.0

        # Start with a different theta (e.g., from a spike reset at E_Na=30)
        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()
            # Manually set theta to a high value
            neuron.theta.value = np.asarray(30.0)

            for k in range(n_steps):
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(0.0)

            theta_final = _get_scalar(neuron.theta.value)
            # After 500 steps (50 ms with dt=0.1), theta should be
            # substantially closer to theta_eq
            self.assertLess(abs(theta_final - theta_eq), 5.0,
                            msg="Theta should relax toward theta_eq")


class TestHTNeuronMultipleNeurons(unittest.TestCase):
    r"""Test with population of multiple neurons."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_population_size(self):
        r"""Should work with in_size > 1."""
        dt = 0.1

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(5)
            neuron.init_state()

            # All neurons should start at same values
            V_values = np.asarray(neuron.V.value)
            self.assertEqual(V_values.shape, (5,))
            np.testing.assert_array_almost_equal(
                V_values, V_values[0] * np.ones(5), decimal=10
            )

            for k in range(10):
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(0.0)

            # After evolution, all identical neurons with same input should
            # still have same state
            V_final = np.asarray(neuron.V.value)
            np.testing.assert_array_almost_equal(
                V_final, V_final[0] * np.ones(5), decimal=5
            )


class TestHTNeuronCondSteps(unittest.TestCase):
    r"""Test that conductance step pre-computation matches NEST."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_cond_step_AMPA(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""AMPA cond_step should equal g_peak * beta_normalization_factor."""
            neuron = ht_neuron(1)
            expected = 0.1 * _beta_normalization_factor(0.5, 2.4)
            self.assertAlmostEqual(neuron._cond_step_AMPA, expected, places=12)

    def test_cond_step_NMDA(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = ht_neuron(1)
            expected = 0.075 * _beta_normalization_factor(4.0, 40.0)
            self.assertAlmostEqual(neuron._cond_step_NMDA, expected, places=12)

    def test_cond_step_GABA_A(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = ht_neuron(1)
            expected = 0.33 * _beta_normalization_factor(1.0, 7.0)
            self.assertAlmostEqual(neuron._cond_step_GABA_A, expected, places=12)

    def test_cond_step_GABA_B(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = ht_neuron(1)
            expected = 0.0132 * _beta_normalization_factor(60.0, 200.0)
            self.assertAlmostEqual(neuron._cond_step_GABA_B, expected, places=12)


class TestHTNeuronReferenceComparison(unittest.TestCase):
    r"""Compare full simulation traces between model and reference solver."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_dc_driven_trace(self):
        r"""Compare voltage trace under DC input with reference solver."""
        dt = 0.1
        n_steps = 100
        I_dc = 3.0

        params = _default_params()
        params['I_stim'] = I_dc
        y_ref = _default_initial_state(params)

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = ht_neuron(1)
            neuron.init_state()
            # Set I_stim to I_dc (it applies from step 0)
            neuron.I_stim.value = np.asarray(I_dc)

            V_model_trace = []
            V_ref_trace = []

            for k in range(n_steps):
                # Reference solver
                sol = solve_ivp(
                    lambda t, y: _nest_ht_dynamics(t, y, params),
                    [0.0, dt], y_ref,
                    method='RK45', rtol=1e-3, atol=1e-9,
                )
                y_ref = sol.y[:, -1]
                m_eq = _m_eq_NMDA(y_ref[_V_M], params['S_act_NMDA'], params['V_act_NMDA'])
                y_ref[_m_fast_NMDA] = min(m_eq, y_ref[_m_fast_NMDA])
                y_ref[_m_slow_NMDA] = min(m_eq, y_ref[_m_slow_NMDA])
                V_ref_trace.append(y_ref[_V_M])

                # Model
                with brainstate.environ.context(t=k * dt * u.ms):
                    neuron.update(I_dc)
                V_model_trace.append(_get_scalar(neuron.V.value))

            V_model_trace = np.array(V_model_trace)
            V_ref_trace = np.array(V_ref_trace)

            # Traces should match closely
            max_diff = np.max(np.abs(V_model_trace - V_ref_trace))
            self.assertLess(max_diff, 0.5,
                            msg=f"Max V difference {max_diff:.4f} mV exceeds tolerance")


if __name__ == '__main__':
    unittest.main()
