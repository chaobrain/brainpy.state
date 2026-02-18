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

r"""Tests for NEST-compatible hh_psc_alpha_gap neuron model.

Tests cover:
- Default parameter values matching NEST
- Parameter validation
- State variable initialization (including gating variable equilibrium)
- Subthreshold dynamics (ODE integration correctness)
- Spike detection (threshold-and-local-maximum search)
- Refractory period behavior
- Synaptic current dynamics (alpha-shaped PSCs)
- DC-driven spiking and firing rate behavior
- Comparison against reference ODE solver

All tests use float64 precision on CPU to match NEST's numerical behavior.
"""

import math
import unittest

import numpy as np
from scipy.integrate import solve_ivp

import brainstate
import braintools
import brainunit as u
import jax

from brainpy.state import hh_psc_alpha_gap

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _nest_hh_gap_dynamics(t, y, g_Na, g_Kv1, g_Kv3, g_L, E_Na, E_K, E_L,
                          C_m, I_e, I_stim, tau_ex, tau_in):
    r"""Reference HH dynamics matching NEST hh_psc_alpha_gap_dynamics exactly.

    State vector order: [V, m, h, n, p, dI_ex, I_ex, dI_in, I_in]
    """
    V = y[0]
    m = y[1]
    h = y[2]
    n = y[3]
    p = y[4]
    dI_ex = y[5]
    I_ex = y[6]
    dI_in = y[7]
    I_in = y[8]

    alpha_m = 40.0 * (V - 75.5) / (1.0 - math.exp(-(V - 75.5) / 13.5))
    beta_m = 1.2262 / math.exp(V / 42.248)
    alpha_h = 0.0035 / math.exp(V / 24.186)
    beta_h = 0.017 * (51.25 + V) / (1.0 - math.exp(-(51.25 + V) / 5.2))
    alpha_n = 0.014 * (V + 44.0) / (1.0 - math.exp(-(V + 44.0) / 2.3))
    beta_n = 0.0043 / math.exp((V + 44.0) / 34.0)
    alpha_p = (V - 95.0) / (1.0 - math.exp(-(V - 95.0) / 11.8))
    beta_p = 0.025 / math.exp(V / 22.222)

    I_Na = g_Na * m ** 3 * h * (V - E_Na)
    I_K = (g_Kv1 * n ** 4 + g_Kv3 * p ** 2) * (V - E_K)
    I_L = g_L * (V - E_L)

    f = np.zeros(9)
    f[0] = (-(I_Na + I_K + I_L) + I_stim + I_e + I_ex + I_in) / C_m
    f[1] = alpha_m * (1.0 - m) - beta_m * m
    f[2] = alpha_h * (1.0 - h) - beta_h * h
    f[3] = alpha_n * (1.0 - n) - beta_n * n
    f[4] = alpha_p * (1.0 - p) - beta_p * p
    f[5] = -dI_ex / tau_ex
    f[6] = dI_ex - (I_ex / tau_ex)
    f[7] = -dI_in / tau_in
    f[8] = dI_in - (I_in / tau_in)
    return f


def _get_scalar(x):
    r"""Extract a scalar float from a possibly 1D array."""
    x = np.asarray(x)
    if x.ndim > 0:
        return float(x.flat[0])
    return float(x)


def _V_mV(neuron):
    r"""Get membrane potential as scalar float in mV."""
    return _get_scalar(u.math.asarray(neuron.V.value / u.mV))


def _I_pA(state_val):
    r"""Get current value as scalar float in pA."""
    return _get_scalar(u.math.asarray(state_val / u.pA))


class TestHHPscAlphaGapDefaults(unittest.TestCase):
    r"""Test that default parameter values match NEST hh_psc_alpha_gap."""

    def test_default_parameters(self):
        neuron = hh_psc_alpha_gap(1)
        self.assertAlmostEqual(float(u.math.asarray(neuron.E_L / u.mV)), -70.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.C_m / u.pF)), 40.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.g_Na / u.nS)), 4500.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.g_Kv1 / u.nS)), 9.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.g_Kv3 / u.nS)), 9000.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.g_L / u.nS)), 10.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.E_Na / u.mV)), 74.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.E_K / u.mV)), -90.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.t_ref / u.ms)), 2.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.tau_syn_ex / u.ms)), 0.2, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.tau_syn_in / u.ms)), 2.0, places=10)
        self.assertAlmostEqual(float(u.math.asarray(neuron.I_e / u.pA)), 0.0, places=10)

    def test_initial_state_values(self):
        r"""Initial V should be -69.60401191631222 mV; gating at equilibrium."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_psc_alpha_gap(1)
            neuron.init_state()

            V = _V_mV(neuron)
            self.assertAlmostEqual(V, -69.60401191631222, places=10)

            # Compute equilibrium gating variables at the initial V
            V0 = -69.60401191631222
            alpha_m = 40.0 * (V0 - 75.5) / (1.0 - math.exp(-(V0 - 75.5) / 13.5))
            beta_m = 1.2262 / math.exp(V0 / 42.248)
            alpha_h = 0.0035 / math.exp(V0 / 24.186)
            beta_h = 0.017 * (51.25 + V0) / (1.0 - math.exp(-(51.25 + V0) / 5.2))
            alpha_n = 0.014 * (V0 + 44.0) / (1.0 - math.exp(-(V0 + 44.0) / 2.3))
            beta_n = 0.0043 / math.exp((V0 + 44.0) / 34.0)
            alpha_p = (V0 - 95.0) / (1.0 - math.exp(-(V0 - 95.0) / 11.8))
            beta_p = 0.025 / math.exp(V0 / 22.222)

            m_eq = alpha_m / (alpha_m + beta_m)
            h_eq = alpha_h / (alpha_h + beta_h)
            n_eq = alpha_n / (alpha_n + beta_n)
            p_eq = alpha_p / (alpha_p + beta_p)

            self.assertAlmostEqual(_get_scalar(neuron.m.value), m_eq, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.h.value), h_eq, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.n.value), n_eq, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.p.value), p_eq, places=10)

            # Synaptic currents should be zero
            self.assertAlmostEqual(_I_pA(neuron.I_syn_ex.value), 0.0, places=10)
            self.assertAlmostEqual(_I_pA(neuron.I_syn_in.value), 0.0, places=10)
            self.assertEqual(int(neuron.refractory_step_count.value[0]), 0)

    def test_nest_gating_variable_values(self):
        r"""Verify gating variable initial values match NEST source comments."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_psc_alpha_gap(1)
            neuron.init_state()

            # NEST source comments: 'Act_n': 0.0005741576228359798,
            # 'Inact_p': 0.00025113182271506364, 'Inact_h': 0.8684620412943986
            self.assertAlmostEqual(_get_scalar(neuron.n.value), 0.0005741576228359798, places=12)
            self.assertAlmostEqual(_get_scalar(neuron.p.value), 0.00025113182271506364, places=12)
            self.assertAlmostEqual(_get_scalar(neuron.h.value), 0.8684620412943986, places=12)


class TestHHPscAlphaGapValidation(unittest.TestCase):
    r"""Test parameter validation."""

    def test_negative_capacitance(self):
        with self.assertRaises(ValueError):
            hh_psc_alpha_gap(1, C_m=-40.0 * u.pF)

    def test_zero_capacitance(self):
        with self.assertRaises(ValueError):
            hh_psc_alpha_gap(1, C_m=0.0 * u.pF)

    def test_negative_refractory(self):
        with self.assertRaises(ValueError):
            hh_psc_alpha_gap(1, t_ref=-1.0 * u.ms)

    def test_zero_refractory_ok(self):
        neuron = hh_psc_alpha_gap(1, t_ref=0.0 * u.ms)
        self.assertAlmostEqual(float(u.math.asarray(neuron.t_ref / u.ms)), 0.0)

    def test_zero_tau_syn(self):
        with self.assertRaises(ValueError):
            hh_psc_alpha_gap(1, tau_syn_ex=0.0 * u.ms)
        with self.assertRaises(ValueError):
            hh_psc_alpha_gap(1, tau_syn_in=0.0 * u.ms)

    def test_negative_conductance(self):
        with self.assertRaises(ValueError):
            hh_psc_alpha_gap(1, g_Na=-1.0 * u.nS)
        with self.assertRaises(ValueError):
            hh_psc_alpha_gap(1, g_Kv1=-1.0 * u.nS)
        with self.assertRaises(ValueError):
            hh_psc_alpha_gap(1, g_Kv3=-1.0 * u.nS)
        with self.assertRaises(ValueError):
            hh_psc_alpha_gap(1, g_L=-1.0 * u.nS)


class TestHHPscAlphaGapSubthreshold(unittest.TestCase):
    r"""Test subthreshold dynamics against direct ODE integration."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_subthreshold_relaxation(self):
        r"""Test that neuron relaxes toward resting potential without input."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=0.0 * u.pA)
            neuron.init_state()

            for k in range(100):
                self._step(neuron, k)

            V_final = _V_mV(neuron)
            # Should be near resting potential (~-70 mV)
            self.assertAlmostEqual(V_final, -70.0, delta=2.0)

    def test_ode_integration_matches_reference(self):
        r"""Verify that one step of our model matches a reference RK45 solve."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=500.0 * u.pA)
            neuron.init_state()

            self._step(neuron, 0)

            # Reference integration
            V0 = -69.60401191631222
            alpha_m = 40.0 * (V0 - 75.5) / (1.0 - math.exp(-(V0 - 75.5) / 13.5))
            beta_m = 1.2262 / math.exp(V0 / 42.248)
            alpha_h = 0.0035 / math.exp(V0 / 24.186)
            beta_h = 0.017 * (51.25 + V0) / (1.0 - math.exp(-(51.25 + V0) / 5.2))
            alpha_n = 0.014 * (V0 + 44.0) / (1.0 - math.exp(-(V0 + 44.0) / 2.3))
            beta_n = 0.0043 / math.exp((V0 + 44.0) / 34.0)
            alpha_p = (V0 - 95.0) / (1.0 - math.exp(-(V0 - 95.0) / 11.8))
            beta_p = 0.025 / math.exp(V0 / 22.222)

            m0 = alpha_m / (alpha_m + beta_m)
            h0 = alpha_h / (alpha_h + beta_h)
            n0 = alpha_n / (alpha_n + beta_n)
            p0 = alpha_p / (alpha_p + beta_p)

            y0 = np.array([V0, m0, h0, n0, p0, 0., 0., 0., 0.])
            sol = solve_ivp(
                _nest_hh_gap_dynamics,
                [0.0, 0.1],
                y0,
                method='RK45',
                rtol=1e-6,
                atol=1e-12,
                args=(4500., 9., 9000., 10., 74., -90., -70., 40., 500., 0., 0.2, 2.0),
            )
            yf = sol.y[:, -1]

            V_model = _V_mV(neuron)
            m_model = _get_scalar(neuron.m.value)
            h_model = _get_scalar(neuron.h.value)
            n_model = _get_scalar(neuron.n.value)
            p_model = _get_scalar(neuron.p.value)

            self.assertAlmostEqual(V_model, yf[0], places=8)
            self.assertAlmostEqual(m_model, yf[1], places=10)
            self.assertAlmostEqual(h_model, yf[2], places=10)
            self.assertAlmostEqual(n_model, yf[3], places=10)
            self.assertAlmostEqual(p_model, yf[4], places=10)

    def test_dc_drives_depolarization(self):
        r"""Strong DC input should depolarize the membrane."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=1000.0 * u.pA)
            neuron.init_state()

            V_init = _V_mV(neuron)
            for k in range(10):
                self._step(neuron, k)

            V_after = _V_mV(neuron)
            self.assertGreater(V_after, V_init)

    def test_multi_step_no_input(self):
        r"""Multiple steps without input should match reference ODE solve."""
        n_steps = 50
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=0.0 * u.pA)
            neuron.init_state()

            V_model = []
            for k in range(n_steps):
                self._step(neuron, k)
                V_model.append(_V_mV(neuron))

            # Reference integration
            V0 = -69.60401191631222
            alpha_m = 40.0 * (V0 - 75.5) / (1.0 - math.exp(-(V0 - 75.5) / 13.5))
            beta_m = 1.2262 / math.exp(V0 / 42.248)
            alpha_h = 0.0035 / math.exp(V0 / 24.186)
            beta_h = 0.017 * (51.25 + V0) / (1.0 - math.exp(-(51.25 + V0) / 5.2))
            alpha_n = 0.014 * (V0 + 44.0) / (1.0 - math.exp(-(V0 + 44.0) / 2.3))
            beta_n = 0.0043 / math.exp((V0 + 44.0) / 34.0)
            alpha_p = (V0 - 95.0) / (1.0 - math.exp(-(V0 - 95.0) / 11.8))
            beta_p = 0.025 / math.exp(V0 / 22.222)

            m0 = alpha_m / (alpha_m + beta_m)
            h0 = alpha_h / (alpha_h + beta_h)
            n0 = alpha_n / (alpha_n + beta_n)
            p0 = alpha_p / (alpha_p + beta_p)

            y = np.array([V0, m0, h0, n0, p0, 0., 0., 0., 0.])
            V_ref = []
            for k in range(n_steps):
                sol = solve_ivp(
                    _nest_hh_gap_dynamics,
                    [0.0, 0.1],
                    y,
                    method='RK45',
                    rtol=1e-6,
                    atol=1e-12,
                    args=(4500., 9., 9000., 10., 74., -90., -70., 40., 0., 0., 0.2, 2.0),
                )
                y = sol.y[:, -1]
                V_ref.append(y[0])

            for k in range(n_steps):
                self.assertAlmostEqual(V_model[k], V_ref[k], places=6,
                                       msg=f"V mismatch at step {k}")

    def test_multi_step_with_dc(self):
        r"""Multiple steps with DC should match reference solver."""
        n_steps = 100
        I_e_val = 200.0
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=I_e_val * u.pA)
            neuron.init_state()

            V_model = []
            for k in range(n_steps):
                self._step(neuron, k)
                V_model.append(_V_mV(neuron))

            # Reference
            V0 = -69.60401191631222
            alpha_m = 40.0 * (V0 - 75.5) / (1.0 - math.exp(-(V0 - 75.5) / 13.5))
            beta_m = 1.2262 / math.exp(V0 / 42.248)
            alpha_h = 0.0035 / math.exp(V0 / 24.186)
            beta_h = 0.017 * (51.25 + V0) / (1.0 - math.exp(-(51.25 + V0) / 5.2))
            alpha_n = 0.014 * (V0 + 44.0) / (1.0 - math.exp(-(V0 + 44.0) / 2.3))
            beta_n = 0.0043 / math.exp((V0 + 44.0) / 34.0)
            alpha_p = (V0 - 95.0) / (1.0 - math.exp(-(V0 - 95.0) / 11.8))
            beta_p = 0.025 / math.exp(V0 / 22.222)

            m0 = alpha_m / (alpha_m + beta_m)
            h0 = alpha_h / (alpha_h + beta_h)
            n0 = alpha_n / (alpha_n + beta_n)
            p0 = alpha_p / (alpha_p + beta_p)

            y = np.array([V0, m0, h0, n0, p0, 0., 0., 0., 0.])
            V_ref = []
            for k in range(n_steps):
                sol = solve_ivp(
                    _nest_hh_gap_dynamics,
                    [0.0, 0.1],
                    y,
                    method='RK45',
                    rtol=1e-6,
                    atol=1e-12,
                    args=(4500., 9., 9000., 10., 74., -90., -70., 40.,
                          I_e_val, 0., 0.2, 2.0),
                )
                y = sol.y[:, -1]
                V_ref.append(y[0])

            for k in range(n_steps):
                self.assertAlmostEqual(V_model[k], V_ref[k], places=5,
                                       msg=f"V mismatch at step {k}")


class TestHHPscAlphaGapSpiking(unittest.TestCase):
    r"""Test spike detection and refractory behavior."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0.0 * u.pA, delta=None):
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
            neuron = hh_psc_alpha_gap(1, I_e=200.0 * u.pA)
            neuron.init_state()

            spike_detected = False
            for k in range(300):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    spike_detected = True
                    break

            self.assertTrue(spike_detected, "Neuron should fire with 200 pA DC input within 30 ms")

    def test_no_spike_without_input(self):
        r"""With no input, the neuron should not spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=0.0 * u.pA)
            neuron.init_state()

            for k in range(500):
                spk = self._step(neuron, k)
                self.assertFalse(self._is_spike(spk), f"No spike expected at step {k}")

    def test_spike_detection_logic(self):
        r"""Verify the threshold + local maximum spike detection logic."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=300.0 * u.pA)
            neuron.init_state()

            V_trace = []
            spike_times = []
            for k in range(500):
                spk = self._step(neuron, k)
                V_trace.append(_V_mV(neuron))
                if self._is_spike(spk):
                    spike_times.append(k * 0.1)

            self.assertGreater(len(spike_times), 0)
            V_max = max(V_trace)
            self.assertGreater(V_max, 0.0, "V should exceed 0 mV during action potential")

    def test_refractory_period(self):
        r"""After a spike, no more spikes should occur for t_ref ms."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=300.0 * u.pA, t_ref=5.0 * u.ms)
            neuron.init_state()

            spike_times = []
            for k in range(1000):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    spike_times.append(k * 0.1)

            self.assertGreater(len(spike_times), 1, "Expected multiple spikes with strong DC input")

            for i in range(1, len(spike_times)):
                isi = spike_times[i] - spike_times[i - 1]
                self.assertGreaterEqual(isi, 5.0 - 0.1,
                                        f"ISI {isi:.1f} ms violates refractory period of 5 ms")

    def test_refractory_counter_decrements(self):
        r"""Refractory counter should decrement each step after spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=300.0 * u.pA, t_ref=2.0 * u.ms)
            neuron.init_state()

            first_spike_step = None
            for k in range(500):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    first_spike_step = k
                    break

            self.assertIsNotNone(first_spike_step, "Should detect a spike")

            r = int(neuron.refractory_step_count.value[0])
            self.assertGreater(r, 0, "Refractory counter should be positive after spike")

            r_prev = r
            for k in range(first_spike_step + 1, first_spike_step + 5):
                self._step(neuron, k)
                r_now = int(neuron.refractory_step_count.value[0])
                if r_prev > 0:
                    self.assertEqual(r_now, r_prev - 1,
                                     f"Refractory counter should decrement from {r_prev} to {r_prev - 1}")
                r_prev = r_now

    def test_dynamics_evolve_during_refractory(self):
        r"""Unlike IAF, HH dynamics should continue during the refractory period."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=300.0 * u.pA, t_ref=5.0 * u.ms)
            neuron.init_state()

            for k in range(500):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    break

            V_prev = _V_mV(neuron)
            V_changed = False
            for k2 in range(k + 1, k + 20):
                self._step(neuron, k2)
                V_now = _V_mV(neuron)
                if abs(V_now - V_prev) > 1e-6:
                    V_changed = True
                V_prev = V_now

            self.assertTrue(V_changed, "V should evolve during refractory period in HH model")

    def test_dc_spiking_trajectory(self):
        r"""With strong DC, verify the model produces action potentials with
        reasonable peak voltage and recovery."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=200.0 * u.pA)
            neuron.init_state()

            V_trace = []
            for k in range(1000):
                self._step(neuron, k)
                V_trace.append(_V_mV(neuron))

            V_max = max(V_trace)
            V_min = min(V_trace)

            self.assertGreater(V_max, 20.0, "AP peak should exceed 20 mV")
            self.assertLess(V_min, -70.0, "AHP should be below -70 mV")


class TestHHPscAlphaGapSynaptic(unittest.TestCase):
    r"""Test synaptic current dynamics (alpha-shaped PSCs)."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_excitatory_spike_input(self):
        r"""A positive weight spike input should increase dI_syn_ex."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=0.0 * u.pA)
            neuron.init_state()

            self._step(neuron, 0)
            dI_before = _get_scalar(neuron.dI_syn_ex.value)
            self.assertAlmostEqual(dI_before, 0.0, places=10)

            self._step(neuron, 1, delta=100.0 * u.pA)
            dI_after = _get_scalar(neuron.dI_syn_ex.value)
            self.assertGreater(dI_after, 0.0, "dI_syn_ex should be positive after excitatory input")

    def test_inhibitory_spike_input(self):
        r"""A negative weight spike input should increase (magnitude) dI_syn_in."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=0.0 * u.pA)
            neuron.init_state()

            self._step(neuron, 0)
            self._step(neuron, 1, delta=-50.0 * u.pA)
            dI_in = _get_scalar(neuron.dI_syn_in.value)
            self.assertLess(dI_in, 0.0, "dI_syn_in should be negative after inhibitory input")

    def test_alpha_psc_waveform(self):
        r"""Test that the synaptic current has an alpha-function shape."""
        tau_ex_ms = 2.0
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=0.0 * u.pA, tau_syn_ex=tau_ex_ms * u.ms)
            neuron.init_state()

            self._step(neuron, 0, delta=100.0 * u.pA)

            I_trace = []
            for k in range(1, 100):
                self._step(neuron, k)
                I_trace.append(_I_pA(neuron.I_syn_ex.value))

            peak_idx = np.argmax(I_trace)
            peak_time = (peak_idx + 1) * 0.1

            self.assertAlmostEqual(peak_time, tau_ex_ms, delta=0.5)
            self.assertGreater(I_trace[peak_idx], I_trace[-1])

    def test_psc_normalization(self):
        r"""A spike with weight 1 should produce peak current ~1 pA."""
        tau_ex_ms = 2.0
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(
                1, I_e=0.0 * u.pA, tau_syn_ex=tau_ex_ms * u.ms,
                g_Na=0.0 * u.nS, g_Kv1=0.0 * u.nS, g_Kv3=0.0 * u.nS, g_L=0.0 * u.nS,
            )
            neuron.init_state()

            self._step(neuron, 0, delta=1.0 * u.pA)

            I_trace = []
            for k in range(1, 200):
                self._step(neuron, k)
                I_trace.append(_I_pA(neuron.I_syn_ex.value))

            peak = max(I_trace)
            self.assertAlmostEqual(peak, 1.0, delta=0.05)

    def test_stim_current_buffering(self):
        r"""Stimulation current should be buffered one step."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=0.0 * u.pA)
            neuron.init_state()

            self._step(neuron, 0, x=500.0 * u.pA)

            I_stim = _I_pA(neuron.I_stim.value)
            self.assertAlmostEqual(I_stim, 500.0, delta=1e-10)


class TestHHPscAlphaGapEdgeCases(unittest.TestCase):
    r"""Test edge cases and special configurations."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_custom_initial_gating(self):
        r"""Test that custom initial gating variables are used correctly."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(
                1, Act_m_init=0.5, Inact_h_init=0.3,
                Act_n_init=0.4, Inact_p_init=0.2
            )
            neuron.init_state()

            self.assertAlmostEqual(_get_scalar(neuron.m.value), 0.5, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.h.value), 0.3, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.n.value), 0.4, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.p.value), 0.2, places=10)

    def test_population_size(self):
        r"""Test with a population of neurons."""
        with brainstate.environ.context(dt=self.dt):
            n_neurons = 5
            neuron = hh_psc_alpha_gap(n_neurons, I_e=200.0 * u.pA)
            neuron.init_state()

            for k in range(100):
                spk = self._step(neuron, k)

            V = np.asarray(u.math.asarray(neuron.V.value / u.mV))
            self.assertEqual(V.shape, (n_neurons,))
            for i in range(1, n_neurons):
                self.assertAlmostEqual(float(V[i]), float(V[0]), places=10)

    def test_zero_refractory_period(self):
        r"""With t_ref=0, spikes should not be suppressed by refractoriness."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=300.0 * u.pA, t_ref=0.0 * u.ms)
            neuron.init_state()

            spike_detected = False
            for k in range(300):
                spk = self._step(neuron, k)
                if bool(u.math.all(spk > 0.0)):
                    spike_detected = True
                    break

            self.assertTrue(spike_detected)

    def test_last_spike_time_updated(self):
        r"""Verify that last_spike_time is updated on spike emission."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=300.0 * u.pA)
            neuron.init_state()

            initial_spk_time = _get_scalar(u.math.asarray(neuron.last_spike_time.value / u.ms))
            self.assertLess(initial_spk_time, -1e6)

            for k in range(500):
                spk = self._step(neuron, k)
                if bool(u.math.all(spk > 0.0)):
                    t_spike = _get_scalar(u.math.asarray(neuron.last_spike_time.value / u.ms))
                    expected_t = (k + 1) * 0.1
                    self.assertAlmostEqual(t_spike, expected_t, delta=1e-10)
                    break

    def test_different_ion_channel_kinetics(self):
        r"""Verify model uses different kinetics than hh_psc_alpha.

        hh_psc_alpha_gap has Kv1+Kv3 channels (n^4 and p^2) rather than a
        single K channel (n^4). The gating rate functions are also different.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1)
            neuron.init_state()

            # The initial m value for hh_psc_alpha_gap should be very different
            # from hh_psc_alpha because the rate functions are completely different
            m_gap = _get_scalar(neuron.m.value)

            # hh_psc_alpha uses alpha_m = 0.1*(V+40)/(1-exp(-(V+40)/10))
            # hh_psc_alpha_gap uses alpha_m = 40*(V-75.5)/(1-exp(-(V-75.5)/13.5))
            # At V ~ -70 mV, these give very different m_inf values
            # hh_psc_alpha at V=-65: m_inf ~ 0.053
            # hh_psc_alpha_gap at V=-69.6: m_inf ~ 0.019
            self.assertLess(m_gap, 0.05, "m_init for gap model should be small")


class TestHHPscAlphaGapFiringProperties(unittest.TestCase):
    r"""Test firing rate and AP waveform properties."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0.0 * u.pA):
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def test_firing_rate_increases_with_current(self):
        r"""Firing rate should increase monotonically with input current."""
        with brainstate.environ.context(dt=self.dt):
            rates = []
            for I_amp in [150., 250., 400.]:
                neuron = hh_psc_alpha_gap(1, I_e=I_amp * u.pA)
                neuron.init_state()

                # Warmup
                for k in range(1000):
                    self._step(neuron, k)

                # Count spikes
                n_spikes = 0
                for k in range(1000, 6000):
                    spk = self._step(neuron, k)
                    if self._is_spike(spk):
                        n_spikes += 1

                rates.append(n_spikes)

            for i in range(1, len(rates)):
                self.assertGreaterEqual(rates[i], rates[i - 1],
                                        f"Rate at {[150, 250, 400][i]} pA should be >= rate at "
                                        f"{[150, 250, 400][i - 1]} pA")

    def test_ap_waveform_shape(self):
        r"""Action potential should have characteristic shape: fast upstroke, overshoot, AHP."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_gap(1, I_e=200.0 * u.pA)
            neuron.init_state()

            V_trace = []
            for k in range(500):
                self._step(neuron, k)
                V_trace.append(_V_mV(neuron))

            V_max = max(V_trace)
            V_min = min(V_trace)

            # AP should overshoot 0 mV significantly
            self.assertGreater(V_max, 30.0, "AP peak should be well above 0 mV")
            # AHP should go below resting
            self.assertLess(V_min, -70.0, "AHP should go below resting potential")
            # AP peak should be physiologically reasonable
            self.assertLess(V_max, 100.0, "AP peak should be < 100 mV")


if __name__ == '__main__':
    unittest.main()
