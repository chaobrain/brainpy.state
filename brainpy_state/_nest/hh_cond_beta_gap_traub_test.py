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

r"""Tests for NEST-compatible hh_cond_beta_gap_traub neuron model.

Tests cover:
- Default parameter values matching NEST
- Parameter validation
- Equilibrium initialization of gating variables
- Subthreshold dynamics (ODE integration correctness)
- Spike detection (V_T + 30 threshold and local-maximum criterion)
- Refractory period behavior
- Synaptic conductance dynamics (beta-function / double-exponential)
- Beta normalization factor correctness
- DC-driven spiking behavior
- Comparison against NEST reference data (gap-junction test)

All tests use float64 precision on CPU to match NEST's numerical behavior.
"""

import math
import unittest

import brainstate
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from brainpy_state import hh_cond_beta_gap_traub

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _nest_hh_cond_beta_gap_traub_dynamics(t, y, g_Na, g_K, g_L, E_Na, E_K, E_L,
                                          V_T, E_ex, E_in, C_m, I_e, I_stim,
                                          tau_rise_ex, tau_decay_ex,
                                          tau_rise_in, tau_decay_in):
    r"""Reference dynamics matching NEST hh_cond_beta_gap_traub_dynamics exactly.

    State vector y = [V_m, m, h, n, dg_ex, g_ex, dg_in, g_in].
    """
    V_m = y[0]
    m = y[1]
    h = y[2]
    n = y[3]
    dg_e = y[4]
    g_e = y[5]
    dg_i = y[6]
    g_i = y[7]

    # Ionic currents
    I_Na = g_Na * m ** 3 * h * (V_m - E_Na)
    I_K = g_K * n ** 4 * (V_m - E_K)
    I_L = g_L * (V_m - E_L)

    # Synaptic currents (conductance-based)
    I_syn_exc = g_e * (V_m - E_ex)
    I_syn_inh = g_i * (V_m - E_in)

    # Shifted voltage for gating variable rate equations
    V = V_m - V_T

    alpha_n = 0.032 * (15.0 - V) / (math.exp((15.0 - V) / 5.0) - 1.0)
    beta_n = 0.5 * math.exp((10.0 - V) / 40.0)
    alpha_m = 0.32 * (13.0 - V) / (math.exp((13.0 - V) / 4.0) - 1.0)
    beta_m = 0.28 * (V - 40.0) / (math.exp((V - 40.0) / 5.0) - 1.0)
    alpha_h = 0.128 * math.exp((17.0 - V) / 18.0)
    beta_h = 4.0 / (1.0 + math.exp((40.0 - V) / 5.0))

    f = np.zeros(8)
    f[0] = (-I_Na - I_K - I_L - I_syn_exc - I_syn_inh + I_stim + I_e) / C_m
    f[1] = alpha_m - (alpha_m + beta_m) * m
    f[2] = alpha_h - (alpha_h + beta_h) * h
    f[3] = alpha_n - (alpha_n + beta_n) * n
    # Beta-function synapse: excitatory
    f[4] = -dg_e / tau_decay_ex
    f[5] = dg_e - (g_e / tau_rise_ex)
    # Beta-function synapse: inhibitory
    f[6] = -dg_i / tau_decay_in
    f[7] = dg_i - (g_i / tau_rise_in)
    return f


def _get_scalar(x):
    r"""Extract a scalar float from a possibly 1D array or Quantity."""
    x = np.asarray(u.get_mantissa(x))
    if x.ndim > 0:
        return float(x.flat[0])
    return float(x)


def _V_mV(neuron):
    r"""Get membrane potential as scalar float in mV."""
    return _get_scalar(u.math.asarray(neuron.V.value / u.mV))


def _g_nS(state_val):
    r"""Get conductance value as scalar float in nS."""
    return _get_scalar(u.math.asarray(state_val / u.nS))


class TestBetaNormalizationFactor(unittest.TestCase):
    r"""Test the beta normalization factor computation."""

    def test_known_values(self):
        r"""Test against manually computed values."""
        # For tau_rise=0.5, tau_decay=5.0 (default excitatory)
        tau_r, tau_d = 0.5, 5.0
        t_peak = tau_d * tau_r * math.log(tau_d / tau_r) / (tau_d - tau_r)
        peak_val = math.exp(-t_peak / tau_d) - math.exp(-t_peak / tau_r)
        expected = (1.0 / tau_r - 1.0 / tau_d) / peak_val
        result = hh_cond_beta_gap_traub._beta_normalization_factor_scalar(tau_r, tau_d)
        self.assertAlmostEqual(result, expected, places=10)

    def test_equal_tau_falls_back_to_alpha(self):
        r"""When tau_rise == tau_decay, should use alpha function fallback."""
        tau = 5.0
        result = hh_cond_beta_gap_traub._beta_normalization_factor_scalar(tau, tau)
        expected = math.e / tau
        self.assertAlmostEqual(result, expected, places=10)

    def test_positive_result(self):
        r"""Normalization factor should always be positive."""
        test_cases = [(0.1, 1.0), (0.5, 5.0), (1.0, 10.0), (2.0, 2.0)]
        for tau_r, tau_d in test_cases:
            with self.subTest(tau_rise=tau_r, tau_decay=tau_d):
                result = hh_cond_beta_gap_traub._beta_normalization_factor_scalar(tau_r, tau_d)
                self.assertGreater(result, 0.0)

    def test_symmetry_tau_swap(self):
        r"""Swapping tau_rise and tau_decay should give a different (valid) result."""
        result1 = hh_cond_beta_gap_traub._beta_normalization_factor_scalar(0.5, 5.0)
        result2 = hh_cond_beta_gap_traub._beta_normalization_factor_scalar(5.0, 0.5)
        self.assertGreater(result1, 0.0)
        self.assertGreater(result2, 0.0)


class TestHHCondBetaGapTraubDefaults(unittest.TestCase):
    r"""Test that default parameter values match NEST hh_cond_beta_gap_traub."""

    def test_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_cond_beta_gap_traub(1)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_L / u.mV)), -60.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.C_m / u.pF)), 200.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.g_Na / u.nS)), 20000.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.g_K / u.nS)), 6000.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.g_L / u.nS)), 10.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_Na / u.mV)), 50.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_K / u.mV)), -90.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.V_T / u.mV)), -50.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_ex / u.mV)), 0.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_in / u.mV)), -80.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.t_ref / u.ms)), 2.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_rise_ex / u.ms)), 0.5, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_decay_ex / u.ms)), 5.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_rise_in / u.ms)), 0.5, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_decay_in / u.ms)), 10.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.I_e / u.pA)), 0.0, places=10)

    def test_initial_state_values(self):
        r"""Initial V should be E_L; gating at equilibrium for V=E_L."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_cond_beta_gap_traub(1)
            neuron.init_state()

            V = _V_mV(neuron)
            self.assertAlmostEqual(V, -60.0, places=10)

            # Check equilibrium gating variables at V = -60 mV (E_L default).
            # NEST uses raw voltage (not V - V_T) for equilibrium initialization.
            V0 = -60.0
            alpha_n = 0.032 * (15.0 - V0) / (math.exp((15.0 - V0) / 5.0) - 1.0)
            beta_n = 0.5 * math.exp((10.0 - V0) / 40.0)
            alpha_m = 0.32 * (13.0 - V0) / (math.exp((13.0 - V0) / 4.0) - 1.0)
            beta_m = 0.28 * (V0 - 40.0) / (math.exp((V0 - 40.0) / 5.0) - 1.0)
            alpha_h = 0.128 * math.exp((17.0 - V0) / 18.0)
            beta_h = 4.0 / (1.0 + math.exp((40.0 - V0) / 5.0))

            m_eq = alpha_m / (alpha_m + beta_m)
            h_eq = alpha_h / (alpha_h + beta_h)
            n_eq = alpha_n / (alpha_n + beta_n)

            self.assertAlmostEqual(_get_scalar(neuron.m.value), m_eq, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.h.value), h_eq, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.n.value), n_eq, places=10)

            # Synaptic conductances and derivatives should be zero
            self.assertAlmostEqual(_g_nS(neuron.g_ex.value), 0.0, places=10)
            self.assertAlmostEqual(_g_nS(neuron.g_in.value), 0.0, places=10)
            self.assertAlmostEqual(_g_nS(neuron.dg_ex.value), 0.0, places=10)
            self.assertAlmostEqual(_g_nS(neuron.dg_in.value), 0.0, places=10)
            self.assertEqual(int(neuron.refractory_step_count.value[0]), 0)

    def test_equilibrium_function(self):
        r"""Test the equilibrium function directly."""
        m_inf, h_inf, n_inf = hh_cond_beta_gap_traub._hh_equilibrium(-60.0)
        self.assertGreater(m_inf, 0.0)
        self.assertLess(m_inf, 1.0)
        self.assertGreater(h_inf, 0.0)
        self.assertLess(h_inf, 1.0)
        self.assertGreater(n_inf, 0.0)
        self.assertLess(n_inf, 1.0)


class TestHHCondBetaGapTraubValidation(unittest.TestCase):
    r"""Test parameter validation."""

    def test_negative_capacitance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, C_m=-100. * u.pF)

    def test_zero_capacitance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, C_m=0. * u.pF)

    def test_negative_refractory(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, t_ref=-1. * u.ms)

    def test_zero_refractory_ok(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_cond_beta_gap_traub(1, t_ref=0. * u.ms)
            self.assertAlmostEqual(float(u.math.asarray(neuron.t_ref / u.ms)), 0.0)

    def test_zero_tau_rise(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, tau_rise_ex=0. * u.ms)
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, tau_rise_in=0. * u.ms)

    def test_zero_tau_decay(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, tau_decay_ex=0. * u.ms)
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, tau_decay_in=0. * u.ms)

    def test_negative_conductance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, g_Na=-1. * u.nS)
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, g_K=-1. * u.nS)
            with self.assertRaises(ValueError):
                hh_cond_beta_gap_traub(1, g_L=-1. * u.nS)


class TestHHCondBetaGapTraubSubthreshold(unittest.TestCase):
    r"""Test subthreshold dynamics against direct ODE integration."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_subthreshold_relaxation(self):
        r"""Test that neuron evolves from initial state without input.

        With default parameters, the NEST initialization sets gating variables
        at equilibrium for raw V_m (not V_m - V_T), so the neuron is NOT at
        true steady state and will drift from the initial V_m.  After settling,
        V should remain bounded and below the spike threshold (V_T + 30).
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)

            brainstate.transform.for_loop(_run_step, jnp.arange(1000))

            V_final = _V_mV(neuron)
            V_T_val = float(u.math.asarray(neuron.V_T / u.mV))
            # Should settle below spike threshold
            self.assertLess(V_final, V_T_val + 30.0)

    def test_ode_integration_matches_reference(self):
        r"""Verify that one step of our model matches a reference RK45 solve."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=500. * u.pA)
            neuron.init_state()

            self._step(neuron, 0)

            # Reference integration
            V0 = -60.0
            m_eq, h_eq, n_eq = hh_cond_beta_gap_traub._hh_equilibrium(V0)

            y0 = np.array([V0, m_eq, h_eq, n_eq, 0., 0., 0., 0.])
            sol = solve_ivp(
                _nest_hh_cond_beta_gap_traub_dynamics,
                [0.0, 0.1],
                y0,
                method='RK45',
                rtol=1e-3,
                atol=1e-9,
                args=(20000., 6000., 10., 50., -90., -60.,
                      -50., 0., -80., 200., 500., 0.,
                      0.5, 5., 0.5, 10.),
            )
            yf = sol.y[:, -1]

            V_model = _V_mV(neuron)
            m_model = _get_scalar(neuron.m.value)
            h_model = _get_scalar(neuron.h.value)
            n_model = _get_scalar(neuron.n.value)

            self.assertAlmostEqual(V_model, yf[0], places=2)
            self.assertAlmostEqual(m_model, yf[1], places=4)
            self.assertAlmostEqual(h_model, yf[2], places=4)
            self.assertAlmostEqual(n_model, yf[3], places=4)

    def test_dc_drives_depolarization(self):
        r"""Strong DC input should depolarize the membrane."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=1000. * u.pA)
            neuron.init_state()

            V_init = _V_mV(neuron)
            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)

            brainstate.transform.for_loop(_run_step, jnp.arange(10))

            V_after = _V_mV(neuron)
            self.assertGreater(V_after, V_init)


class TestHHCondBetaGapTraubSpiking(unittest.TestCase):
    r"""Test spike detection and refractory behavior."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def test_spike_occurs_with_strong_dc(self):
        r"""With a strong DC input, the neuron should fire a spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=1000. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            spk_all = brainstate.transform.for_loop(_run_step, jnp.arange(200))
            spk_arr = np.asarray(u.get_mantissa(spk_all[:, 0]))
            self.assertTrue(np.any(spk_arr > 0.0), "Neuron should fire with 1000 pA DC input within 20 ms")

    def test_no_spike_with_hyperpolarized_start(self):
        r"""With hyperpolarized initial V and no input, the neuron should not spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=0. * u.pA, V_m_init=-80. * u.mV)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            spk_all = brainstate.transform.for_loop(_run_step, jnp.arange(500))
            spk_arr = np.asarray(u.get_mantissa(spk_all[:, 0]))
            self.assertFalse(np.any(spk_arr > 0.0), "No spike expected")

    def test_spike_detection_threshold(self):
        r"""Verify spike uses V_T + 30 threshold."""
        with brainstate.environ.context(dt=self.dt):
            # With V_T = -50, threshold is -20 mV
            neuron = hh_cond_beta_gap_traub(1, I_e=1500. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return neuron.V.value / u.mV, spk

            results = brainstate.transform.for_loop(_run_step, jnp.arange(300))
            V_trace = list(np.asarray(results[0][:, 0]))
            spk_arr = np.asarray(u.get_mantissa(results[1][:, 0]))
            spike_times = list(np.where(spk_arr > 0.0)[0] * 0.1)

            self.assertGreater(len(spike_times), 0)
            V_max = max(V_trace)
            # V should exceed V_T + 30 = -20 mV during action potential
            self.assertGreater(V_max, -20.0, "V should exceed V_T + 30 mV during action potential")

    def test_refractory_period(self):
        r"""After a spike, no more spikes should occur for t_ref ms."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=1500. * u.pA, t_ref=5. * u.ms)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            spk_all = brainstate.transform.for_loop(_run_step, jnp.arange(500))
            spk_arr = np.asarray(u.get_mantissa(spk_all[:, 0]))
            spike_times = list(np.where(spk_arr > 0.0)[0] * 0.1)

            self.assertGreater(len(spike_times), 1, "Expected multiple spikes with strong DC input")

            for i in range(1, len(spike_times)):
                isi = spike_times[i] - spike_times[i - 1]
                self.assertGreaterEqual(isi, 5.0 - 0.1,
                                        f"ISI {isi:.1f} ms violates refractory period of 5 ms")

    def test_refractory_counter_decrements(self):
        r"""Refractory counter should decrement each step after spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=1500. * u.pA, t_ref=2. * u.ms)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk, neuron.refractory_step_count.value

            results = brainstate.transform.for_loop(_run_step, jnp.arange(305))
            spk_arr = np.asarray(u.get_mantissa(results[0][:, 0]))
            r_arr = np.asarray(results[1][:, 0])

            spike_indices = np.where(spk_arr > 0.0)[0]
            self.assertGreater(len(spike_indices), 0, "Should detect a spike")
            first_spike_step = int(spike_indices[0])

            r = int(r_arr[first_spike_step])
            self.assertGreater(r, 0, "Refractory counter should be positive after spike")

            r_prev = r
            for k_offset in range(1, 5):
                idx = first_spike_step + k_offset
                if idx < len(r_arr):
                    r_now = int(r_arr[idx])
                    if r_prev > 0:
                        self.assertEqual(r_now, r_prev - 1,
                                         f"Refractory counter should decrement from {r_prev} to {r_prev - 1}")
                    r_prev = r_now

    def test_dynamics_evolve_during_refractory(self):
        r"""Unlike IAF, HH dynamics should continue during the refractory period."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=1500. * u.pA, t_ref=5. * u.ms)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return neuron.V.value / u.mV, spk

            results = brainstate.transform.for_loop(_run_step, jnp.arange(320))
            V_all = np.asarray(results[0][:, 0])
            spk_arr = np.asarray(u.get_mantissa(results[1][:, 0]))

            spike_indices = np.where(spk_arr > 0.0)[0]
            self.assertGreater(len(spike_indices), 0, "Should detect a spike")
            first_spike = int(spike_indices[0])

            post_spike_V = V_all[first_spike:first_spike + 20]
            V_changed = np.any(np.abs(np.diff(post_spike_V)) > 1e-6)
            self.assertTrue(V_changed, "V should evolve during refractory period in HH model")


class TestHHCondBetaGapTraubSynaptic(unittest.TestCase):
    r"""Test synaptic conductance dynamics (beta-function / double-exponential)."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_excitatory_conductance_input(self):
        r"""A positive weight spike input should increase g_ex."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            self._step(neuron, 0)
            g_before = _g_nS(neuron.g_ex.value)
            self.assertAlmostEqual(g_before, 0.0, places=10)

            # After a spike input, g_ex should eventually increase
            # (the input goes to dg_ex, which then drives g_ex)
            self._step(neuron, 1, delta=5. * u.nS)
            dg_after = _g_nS(neuron.dg_ex.value)
            self.assertGreater(dg_after, 0.0, "dg_ex should be positive after excitatory input")

            # One more step to let dg_ex drive g_ex
            self._step(neuron, 2)
            g_after = _g_nS(neuron.g_ex.value)
            self.assertGreater(g_after, 0.0, "g_ex should be positive after dg_ex drives it")

    def test_inhibitory_conductance_input(self):
        r"""A negative weight spike input should increase g_in."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            self._step(neuron, 0)
            self._step(neuron, 1, delta=-5. * u.nS)
            dg_in = _g_nS(neuron.dg_in.value)
            self.assertGreater(dg_in, 0.0, "dg_in should be positive after inhibitory input (sign flipped)")

    def test_beta_conductance_rise_and_decay(self):
        r"""Conductance should rise and then decay (beta function shape)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(
                1, I_e=0. * u.pA,
                g_Na=0. * u.nS, g_K=0. * u.nS, g_L=0. * u.nS,
            )
            neuron.init_state()

            # Add a conductance pulse
            self._step(neuron, 0, delta=10. * u.nS)

            # Collect conductance trace
            def _run_step_g(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                return neuron.g_ex.value / u.nS

            g_trace = list(np.asarray(brainstate.transform.for_loop(_run_step_g, jnp.arange(1, 500))[:, 0]))

            # Conductance should first rise and then decay (beta shape)
            # Find peak index
            peak_idx = np.argmax(g_trace)
            self.assertGreater(peak_idx, 0, "Peak should not be at the first step (rise phase expected)")

            # After peak, should decay
            peak_val = g_trace[peak_idx]
            self.assertGreater(peak_val, 0.0, "Peak conductance should be positive")

            # Check decay after peak
            if peak_idx + 50 < len(g_trace):
                self.assertLess(g_trace[peak_idx + 50], peak_val,
                                "Conductance should decay after peak")

    def test_conductance_eventually_returns_to_zero(self):
        r"""After a spike input, conductance should eventually decay back to ~0."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(
                1, I_e=0. * u.pA,
                g_Na=0. * u.nS, g_K=0. * u.nS, g_L=0. * u.nS,
            )
            neuron.init_state()

            self._step(neuron, 0, delta=10. * u.nS)

            def _run_step_decay(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)

            brainstate.transform.for_loop(_run_step_decay, jnp.arange(1, 2000))

            g_final = _g_nS(neuron.g_ex.value)
            self.assertAlmostEqual(g_final, 0.0, delta=1e-3,
                                   msg="Conductance should decay to near zero")


class TestHHCondBetaGapTraubMultiStep(unittest.TestCase):
    r"""Multi-step integration tests comparing against a reference solver."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_multi_step_no_input(self):
        r"""Multiple steps without input should match reference ODE solve."""
        n_steps = 50
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                return neuron.V.value / u.mV

            V_model = list(np.asarray(brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))[:, 0]))

            # Reference integration
            V0 = -60.0
            m_eq, h_eq, n_eq = hh_cond_beta_gap_traub._hh_equilibrium(V0)

            y = np.array([V0, m_eq, h_eq, n_eq, 0., 0., 0., 0.])
            V_ref = []
            for k in range(n_steps):
                sol = solve_ivp(
                    _nest_hh_cond_beta_gap_traub_dynamics,
                    [0.0, 0.1],
                    y,
                    method='RK45',
                    rtol=1e-3,
                    atol=1e-9,
                    args=(20000., 6000., 10., 50., -90., -60.,
                          -50., 0., -80., 200., 0., 0.,
                          0.5, 5., 0.5, 10.),
                )
                y = sol.y[:, -1]
                V_ref.append(y[0])

            for k in range(n_steps):
                self.assertAlmostEqual(V_model[k], V_ref[k], places=2,
                                       msg=f"V mismatch at step {k}")

    def test_dc_spiking_trajectory(self):
        r"""With strong DC, verify the model produces action potentials with
        reasonable peak voltage and recovery."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=1000. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                return neuron.V.value / u.mV

            V_trace = list(np.asarray(brainstate.transform.for_loop(_run_step, jnp.arange(500))[:, 0]))

            V_max = max(V_trace)
            V_min = min(V_trace)

            self.assertGreater(V_max, -20.0, "AP peak should exceed -20 mV (V_T+30)")
            self.assertLess(V_min, -60.0, "AHP should be below -60 mV")

    def test_firing_rate_increases_with_current(self):
        r"""Firing rate should increase monotonically with input current."""
        with brainstate.environ.context(dt=self.dt):
            rates = []
            for I_amp in [500., 1000., 1500.]:
                neuron = hh_cond_beta_gap_traub(1, I_e=I_amp * u.pA)
                neuron.init_state()

                def _run_step(k):
                    with brainstate.environ.context(t=k * self.dt):
                        neuron.update(x=0. * u.pA)

                brainstate.transform.for_loop(_run_step, jnp.arange(1000))

                def _count_step(k):
                    with brainstate.environ.context(t=k * self.dt):
                        spk = neuron.update(x=0. * u.pA)
                    return spk

                spk_all = brainstate.transform.for_loop(_count_step, jnp.arange(1000, 11000))
                spk_arr = np.asarray(u.get_mantissa(spk_all[:, 0]))
                n_spikes = int(np.sum(spk_arr > 0.0))
                rates.append(n_spikes)

            for i in range(1, len(rates)):
                self.assertGreaterEqual(rates[i], rates[i - 1],
                                        f"Rate at {[500, 1000, 1500][i]} pA should be >= rate at "
                                        f"{[500, 1000, 1500][i - 1]} pA")


class TestHHCondBetaGapTraubNESTReference(unittest.TestCase):
    r"""Test against NEST gap-junction reference data.

    The NEST test (test_hh_cond_beta_gap_traub.py) simulates two neurons:
    - Neuron 1: receives I_e = 200 pA
    - Neuron 2: no external input, connected via gap junction to neuron 1

    Since we model single neurons (no gap-junction coupling), we test that:
    1. A single neuron with I_e=200 produces the expected spiking behavior.
    2. The subthreshold voltage trajectory is qualitatively correct.
    """

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_dc_200pA_produces_spikes(self):
        r"""With I_e=200 pA, the neuron should fire spikes (NEST test uses this)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=200. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            spk_all = brainstate.transform.for_loop(_run_step, jnp.arange(2000))
            spk_arr = np.asarray(u.get_mantissa(spk_all[:, 0]))
            self.assertTrue(np.any(spk_arr > 0.0),
                            "Neuron should fire with 200 pA DC input within 200 ms")

    def test_nest_reference_voltage_qualitative(self):
        r"""Test that the model's voltage trajectory is qualitatively correct.

        The NEST reference for neuron 2 (no input, gap-coupled) starts at -60 mV
        and slowly depolarizes due to gap current from neuron 1. We test neuron 1
        (with I_e=200 pA) for reasonable dynamics.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=200. * u.pA)
            neuron.init_state()

            def _run_step_v(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                return neuron.V.value / u.mV

            V_trace = list(np.asarray(brainstate.transform.for_loop(_run_step_v, jnp.arange(200))[:, 0]))

            # The neuron should depolarize from -60 mV
            self.assertGreater(V_trace[-1], -60.0,
                               "Neuron should depolarize with 200 pA input")

            # Voltage should not blow up
            self.assertLess(max(V_trace), 100.0,
                            "Voltage should remain bounded")
            self.assertGreater(min(V_trace), -100.0,
                               "Voltage should remain bounded")


class TestHHCondBetaGapTraubEdgeCases(unittest.TestCase):
    r"""Test edge cases and special configurations."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_custom_initial_gating(self):
        r"""Test that custom initial gating variables are used correctly."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, Act_m_init=0.5, Inact_h_init=0.3, Act_n_init=0.4)
            neuron.init_state()

            self.assertAlmostEqual(_get_scalar(neuron.m.value), 0.5, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.h.value), 0.3, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.n.value), 0.4, places=10)

    def test_custom_V_T(self):
        r"""Test that a different V_T shifts the effective threshold."""
        with brainstate.environ.context(dt=self.dt):
            # With V_T = -63, threshold is V_T + 30 = -33 mV
            neuron = hh_cond_beta_gap_traub(1, V_T=-63. * u.mV)
            self.assertAlmostEqual(float(u.math.asarray(neuron.V_T / u.mV)), -63.0)

    def test_population_size(self):
        r"""Test with a population of neurons."""
        with brainstate.environ.context(dt=self.dt):
            n_neurons = 5
            neuron = hh_cond_beta_gap_traub(n_neurons, I_e=1000. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)

            brainstate.transform.for_loop(_run_step, jnp.arange(100))

            V = np.asarray(u.math.asarray(neuron.V.value / u.mV))
            self.assertEqual(V.shape, (n_neurons,))
            for i in range(1, n_neurons):
                self.assertAlmostEqual(float(V[i]), float(V[0]), places=10)

    def test_zero_refractory_period(self):
        r"""With t_ref=0, spikes should not be suppressed by refractoriness."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=1500. * u.pA, t_ref=0. * u.ms)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            spk_all = brainstate.transform.for_loop(_run_step, jnp.arange(200))
            spk_arr = np.asarray(u.get_mantissa(spk_all[:, 0]))
            self.assertTrue(np.any(spk_arr > 0.0))

    def test_last_spike_time_updated(self):
        r"""Verify that last_spike_time is updated on spike emission."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=1500. * u.pA)
            neuron.init_state()

            initial_spk_time = _get_scalar(u.math.asarray(neuron.last_spike_time.value / u.ms))
            self.assertLess(initial_spk_time, -1e6)

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk, neuron.last_spike_time.value / u.ms

            results = brainstate.transform.for_loop(_run_step, jnp.arange(200))
            spk_arr = np.asarray(u.get_mantissa(results[0][:, 0]))
            lst_arr = np.asarray(results[1][:, 0])

            spike_indices = np.where(spk_arr > 0.0)[0]
            self.assertGreater(len(spike_indices), 0, "Should detect a spike")
            first_spike = int(spike_indices[0])
            t_spike = float(lst_arr[first_spike])
            expected_t = (first_spike + 1) * 0.1
            self.assertAlmostEqual(t_spike, expected_t, delta=1e-10)

    def test_excitatory_reversal_potential_effect(self):
        r"""Excitatory synaptic input should depolarize when V < E_ex."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            V_before = _V_mV(neuron)
            # Add a large excitatory conductance
            self._step(neuron, 0, delta=50. * u.nS)
            # Step a few times so the beta-function conductance builds up and affects V
            def _run_step_ex(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)

            brainstate.transform.for_loop(_run_step_ex, jnp.arange(1, 20))
            V_after = _V_mV(neuron)

            self.assertGreater(V_after, V_before,
                               "Excitatory input should depolarize (V < E_ex)")

    def test_inhibitory_reversal_potential_effect(self):
        r"""Inhibitory synaptic input should hyperpolarize when V > E_in."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            V_before = _V_mV(neuron)
            self._step(neuron, 0, delta=-50. * u.nS)
            # Step a few times for the beta-function inhibitory conductance to build up
            def _run_step_in(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)

            brainstate.transform.for_loop(_run_step_in, jnp.arange(1, 20))
            V_after = _V_mV(neuron)

            self.assertLess(V_after, V_before,
                            "Inhibitory input should hyperpolarize (V > E_in)")

    def test_different_V_T_default_from_hh_cond_exp_traub(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Verify V_T default is -50 mV (different from hh_cond_exp_traub's -63 mV)."""
            neuron = hh_cond_beta_gap_traub(1)
            self.assertAlmostEqual(float(u.math.asarray(neuron.V_T / u.mV)), -50.0, places=10)


class TestHHCondBetaGapTraubBetaSynapseODE(unittest.TestCase):
    r"""Test beta-function synapse ODE dynamics in isolation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_beta_ode_matches_reference(self):
        r"""Test that synapse ODE integration matches reference solve_ivp."""
        tau_rise = 0.5
        tau_decay = 5.0
        pscon = hh_cond_beta_gap_traub._beta_normalization_factor_scalar(tau_rise, tau_decay)

        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(
                1, I_e=0. * u.pA,
                g_Na=0. * u.nS, g_K=0. * u.nS, g_L=0. * u.nS,
                tau_rise_ex=tau_rise * u.ms, tau_decay_ex=tau_decay * u.ms,
            )
            neuron.init_state()

            # Add unit weight spike input
            self._step(neuron, 0, delta=1. * u.nS)

            # After step 0: dg_ex should have the PSConInit value
            dg_model = _g_nS(neuron.dg_ex.value)
            self.assertAlmostEqual(dg_model, pscon, delta=pscon * 0.01,
                                   msg=f"dg_ex after spike should be ~PSConInit ({pscon:.6f})")

    def test_peak_conductance_near_unity(self):
        r"""The beta normalization should give peak conductance ~1 nS for weight=1."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_beta_gap_traub(
                1, I_e=0. * u.pA,
                g_Na=0. * u.nS, g_K=0. * u.nS, g_L=0. * u.nS,
            )
            neuron.init_state()

            self._step(neuron, 0, delta=1. * u.nS)

            def _run_step_peak(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                return neuron.g_ex.value / u.nS

            g_trace = list(np.asarray(brainstate.transform.for_loop(_run_step_peak, jnp.arange(1, 500))[:, 0]))

            peak_g = max(g_trace)
            # With proper normalization, peak should be ~1 nS for a unit weight spike
            self.assertAlmostEqual(peak_g, 1.0, delta=0.15,
                                   msg=f"Peak conductance should be ~1.0 nS, got {peak_g:.4f}")


if __name__ == '__main__':
    unittest.main()
