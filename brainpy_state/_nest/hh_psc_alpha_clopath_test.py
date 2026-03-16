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

r"""Tests for NEST-compatible hh_psc_alpha_clopath neuron model.

Tests cover:
- Default parameter values matching NEST
- Parameter validation
- Subthreshold dynamics (ODE integration correctness)
- Spike detection (threshold-and-local-maximum search)
- Refractory period behavior
- Synaptic current dynamics (alpha-shaped PSCs)
- Clopath low-pass filtered voltage traces (u_bar_plus, u_bar_minus, u_bar_bar)
- DC-driven spiking and firing rate behavior

All tests use float64 precision on CPU to match NEST's numerical behavior.
"""

import math
import unittest

import brainstate
import saiunit as u
import jax
import numpy as np
from brainpy.state import hh_psc_alpha_clopath
from scipy.integrate import solve_ivp

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _nest_hh_clopath_dynamics(t, y, g_Na, g_K, g_L, E_Na, E_K, E_L, C_m, I_e, I_stim,
                              tau_ex, tau_in, tau_ubp, tau_ubm, tau_ubb):
    r"""Reference HH+Clopath dynamics matching NEST hh_psc_alpha_clopath_dynamics exactly.

    State vector y has 11 elements:
        [V_m, m, h, n, dI_ex, I_ex, dI_in, I_in, u_bar_plus, u_bar_minus, u_bar_bar]
    """
    V = y[0]
    m = y[1]
    h = y[2]
    n = y[3]
    dI_ex = y[4]
    I_ex = y[5]
    dI_in = y[6]
    I_in = y[7]
    u_bar_plus = y[8]
    u_bar_minus = y[9]
    u_bar_bar = y[10]

    alpha_n = (0.01 * (V + 55.0)) / (1.0 - math.exp(-(V + 55.0) / 10.0))
    beta_n = 0.125 * math.exp(-(V + 65.0) / 80.0)
    alpha_m = (0.1 * (V + 40.0)) / (1.0 - math.exp(-(V + 40.0) / 10.0))
    beta_m = 4.0 * math.exp(-(V + 65.0) / 18.0)
    alpha_h = 0.07 * math.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))

    I_Na = g_Na * m ** 3 * h * (V - E_Na)
    I_K = g_K * n ** 4 * (V - E_K)
    I_L = g_L * (V - E_L)

    f = np.zeros(11)
    f[0] = (-(I_Na + I_K + I_L) + I_stim + I_e + I_ex + I_in) / C_m
    f[1] = alpha_m * (1.0 - m) - beta_m * m
    f[2] = alpha_h * (1.0 - h) - beta_h * h
    f[3] = alpha_n * (1.0 - n) - beta_n * n
    f[4] = -dI_ex / tau_ex
    f[5] = dI_ex - (I_ex / tau_ex)
    f[6] = -dI_in / tau_in
    f[7] = dI_in - (I_in / tau_in)
    # Clopath filtered voltage traces
    f[8] = (-u_bar_plus + V) / tau_ubp
    f[9] = (-u_bar_minus + V) / tau_ubm
    f[10] = (-u_bar_bar + u_bar_minus) / tau_ubb
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


def _ubp_mV(neuron):
    r"""Get u_bar_plus as scalar float in mV."""
    return _get_scalar(u.math.asarray(neuron.u_bar_plus.value / u.mV))


def _ubm_mV(neuron):
    r"""Get u_bar_minus as scalar float in mV."""
    return _get_scalar(u.math.asarray(neuron.u_bar_minus.value / u.mV))


def _ubb_mV(neuron):
    r"""Get u_bar_bar as scalar float in mV."""
    return _get_scalar(u.math.asarray(neuron.u_bar_bar.value / u.mV))


class TestHHClopathDefaults(unittest.TestCase):
    r"""Test that default parameter values match NEST hh_psc_alpha_clopath."""

    def test_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_psc_alpha_clopath(1)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_L / u.mV)), -54.402, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.C_m / u.pF)), 100.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.g_Na / u.nS)), 12000.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.g_K / u.nS)), 3600.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.g_L / u.nS)), 30.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_Na / u.mV)), 50.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_K / u.mV)), -77.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.t_ref / u.ms)), 2.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_syn_ex / u.ms)), 0.2, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_syn_in / u.ms)), 2.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.I_e / u.pA)), 0.0, places=10)

    def test_default_clopath_time_constants(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Clopath time constants should match NEST defaults."""
            neuron = hh_psc_alpha_clopath(1)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_u_bar_plus / u.ms)), 114.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_u_bar_minus / u.ms)), 10.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_u_bar_bar / u.ms)), 500.0, places=10)

    def test_initial_state_values(self):
        r"""Initial V should be -65 mV; gating at equilibrium for V=-65."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_psc_alpha_clopath(1)
            neuron.init_state()

            V = _V_mV(neuron)
            self.assertAlmostEqual(V, -65.0, places=10)

            # Check equilibrium gating variables at V = -65 mV
            alpha_n = (0.01 * (-65.0 + 55.0)) / (1.0 - math.exp(-(-65.0 + 55.0) / 10.0))
            beta_n = 0.125 * math.exp(-(-65.0 + 65.0) / 80.0)
            alpha_m = (0.1 * (-65.0 + 40.0)) / (1.0 - math.exp(-(-65.0 + 40.0) / 10.0))
            beta_m = 4.0 * math.exp(-(-65.0 + 65.0) / 18.0)
            alpha_h = 0.07 * math.exp(-(-65.0 + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + math.exp(-(-65.0 + 35.0) / 10.0))

            m_eq = alpha_m / (alpha_m + beta_m)
            h_eq = alpha_h / (alpha_h + beta_h)
            n_eq = alpha_n / (alpha_n + beta_n)

            self.assertAlmostEqual(_get_scalar(neuron.m.value), m_eq, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.h.value), h_eq, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.n.value), n_eq, places=10)

            # Synaptic currents should be zero
            self.assertAlmostEqual(_I_pA(neuron.I_syn_ex.value), 0.0, places=10)
            self.assertAlmostEqual(_I_pA(neuron.I_syn_in.value), 0.0, places=10)
            self.assertEqual(int(neuron.refractory_step_count.value[0]), 0)

    def test_initial_clopath_state_values(self):
        r"""Initial Clopath filtered voltages should be 0 mV by default (matching NEST)."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_psc_alpha_clopath(1)
            neuron.init_state()

            self.assertAlmostEqual(_ubp_mV(neuron), 0.0, places=10)
            self.assertAlmostEqual(_ubm_mV(neuron), 0.0, places=10)
            self.assertAlmostEqual(_ubb_mV(neuron), 0.0, places=10)


class TestHHClopathValidation(unittest.TestCase):
    r"""Test parameter validation."""

    def test_negative_capacitance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, C_m=-100. * u.pF)

    def test_zero_capacitance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, C_m=0. * u.pF)

    def test_negative_refractory(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, t_ref=-1. * u.ms)

    def test_zero_refractory_ok(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_psc_alpha_clopath(1, t_ref=0. * u.ms)
            self.assertAlmostEqual(float(u.math.asarray(neuron.t_ref / u.ms)), 0.0)

    def test_zero_tau_syn(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, tau_syn_ex=0. * u.ms)
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, tau_syn_in=0. * u.ms)

    def test_negative_conductance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, g_Na=-1. * u.nS)
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, g_K=-1. * u.nS)
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, g_L=-1. * u.nS)

    def test_zero_clopath_time_constants(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""All Clopath time constants must be strictly positive."""
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, tau_u_bar_plus=0. * u.ms)
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, tau_u_bar_minus=0. * u.ms)
            with self.assertRaises(ValueError):
                hh_psc_alpha_clopath(1, tau_u_bar_bar=0. * u.ms)


class TestHHClopathSubthreshold(unittest.TestCase):
    r"""Test subthreshold dynamics against direct ODE integration."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            if u.get_mantissa(delta) >= 0:
                neuron.add_delta_input(f'delta_{step_idx}', delta, label='w_ex')
            else:
                neuron.add_delta_input(f'delta_{step_idx}', -delta, label='w_in')
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_subthreshold_relaxation(self):
        r"""Test that neuron relaxes toward resting potential without input."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA)
            neuron.init_state()

            for k in range(100):
                self._step(neuron, k)

            V_final = _V_mV(neuron)
            self.assertAlmostEqual(V_final, -65.0, delta=1.0)

    def test_ode_integration_matches_reference(self):
        r"""Verify that one step of our model matches a reference RK45 solve."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=500. * u.pA)
            neuron.init_state()

            self._step(neuron, 0)

            # Reference integration
            V0 = -65.0
            alpha_n = (0.01 * (V0 + 55.0)) / (1.0 - math.exp(-(V0 + 55.0) / 10.0))
            beta_n = 0.125 * math.exp(-(V0 + 65.0) / 80.0)
            alpha_m = (0.1 * (V0 + 40.0)) / (1.0 - math.exp(-(V0 + 40.0) / 10.0))
            beta_m = 4.0 * math.exp(-(V0 + 65.0) / 18.0)
            alpha_h = 0.07 * math.exp(-(V0 + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + math.exp(-(V0 + 35.0) / 10.0))

            m0 = alpha_m / (alpha_m + beta_m)
            h0 = alpha_h / (alpha_h + beta_h)
            n0 = alpha_n / (alpha_n + beta_n)

            # 11-dim state: [V, m, h, n, dI_ex, I_ex, dI_in, I_in, ubp, ubm, ubb]
            y0 = np.array([V0, m0, h0, n0, 0., 0., 0., 0., 0., 0., 0.])
            sol = solve_ivp(
                _nest_hh_clopath_dynamics,
                [0.0, 0.1],
                y0,
                method='RK45',
                rtol=1e-3,
                atol=1e-9,
                args=(12000., 3600., 30., 50., -77., -54.402, 100., 500., 0.,
                      0.2, 2.0, 114.0, 10.0, 500.0),
            )
            yf = sol.y[:, -1]

            V_model = _V_mV(neuron)
            m_model = _get_scalar(neuron.m.value)
            h_model = _get_scalar(neuron.h.value)
            n_model = _get_scalar(neuron.n.value)

            self.assertAlmostEqual(V_model, yf[0], places=8)
            self.assertAlmostEqual(m_model, yf[1], places=10)
            self.assertAlmostEqual(h_model, yf[2], places=10)
            self.assertAlmostEqual(n_model, yf[3], places=10)

            # Clopath variables
            self.assertAlmostEqual(_ubp_mV(neuron), yf[8], places=8)
            self.assertAlmostEqual(_ubm_mV(neuron), yf[9], places=8)
            self.assertAlmostEqual(_ubb_mV(neuron), yf[10], places=8)

    def test_dc_drives_depolarization(self):
        r"""Strong DC input should depolarize the membrane."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=1000. * u.pA)
            neuron.init_state()

            V_init = _V_mV(neuron)
            for k in range(10):
                self._step(neuron, k)

            V_after = _V_mV(neuron)
            self.assertGreater(V_after, V_init)


class TestHHClopathSpiking(unittest.TestCase):
    r"""Test spike detection and refractory behavior."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            if u.get_mantissa(delta) >= 0:
                neuron.add_delta_input(f'delta_{step_idx}', delta, label='w_ex')
            else:
                neuron.add_delta_input(f'delta_{step_idx}', -delta, label='w_in')
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def test_spike_occurs_with_strong_dc(self):
        r"""With a strong DC input, the neuron should fire a spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=1000. * u.pA)
            neuron.init_state()

            spike_detected = False
            for k in range(200):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    spike_detected = True
                    break

            self.assertTrue(spike_detected, "Neuron should fire with 1000 pA DC input within 20 ms")

    def test_no_spike_without_input(self):
        r"""With no input, the neuron should not spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA)
            neuron.init_state()

            for k in range(500):
                spk = self._step(neuron, k)
                self.assertFalse(self._is_spike(spk), f"No spike expected at step {k}")

    def test_spike_detection_logic(self):
        r"""Verify the threshold + local maximum spike detection logic."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=1500. * u.pA)
            neuron.init_state()

            V_trace = []
            spike_times = []
            for k in range(300):
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
            neuron = hh_psc_alpha_clopath(1, I_e=1500. * u.pA, t_ref=5. * u.ms)
            neuron.init_state()

            spike_times = []
            for k in range(500):
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
            neuron = hh_psc_alpha_clopath(1, I_e=1500. * u.pA, t_ref=2. * u.ms)
            neuron.init_state()

            first_spike_step = None
            for k in range(300):
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
            neuron = hh_psc_alpha_clopath(1, I_e=1500. * u.pA, t_ref=5. * u.ms)
            neuron.init_state()

            for k in range(300):
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


class TestHHClopathSynaptic(unittest.TestCase):
    r"""Test synaptic current dynamics (alpha-shaped PSCs)."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            if u.get_mantissa(delta) >= 0:
                neuron.add_delta_input(f'delta_{step_idx}', delta, label='w_ex')
            else:
                neuron.add_delta_input(f'delta_{step_idx}', -delta, label='w_in')
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_excitatory_spike_input(self):
        r"""A positive weight spike input should increase dI_syn_ex."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA)
            neuron.init_state()

            self._step(neuron, 0)
            dI_before = _get_scalar(neuron.dI_syn_ex.value)
            self.assertAlmostEqual(dI_before, 0.0, places=10)

            self._step(neuron, 1, delta=100. * u.pA)
            dI_after = _get_scalar(neuron.dI_syn_ex.value)
            self.assertGreater(dI_after, 0.0, "dI_syn_ex should be positive after excitatory input")

    def test_inhibitory_spike_input(self):
        r"""A negative weight spike input should increase (magnitude) dI_syn_in."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA)
            neuron.init_state()

            self._step(neuron, 0)
            self._step(neuron, 1, delta=-50. * u.pA)
            dI_in = _get_scalar(neuron.dI_syn_in.value)
            self.assertLess(dI_in, 0.0, "dI_syn_in should be negative after inhibitory input")

    def test_alpha_psc_waveform(self):
        r"""Test that the synaptic current has an alpha-function shape."""
        tau_ex_ms = 2.0
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA, tau_syn_ex=tau_ex_ms * u.ms)
            neuron.init_state()

            self._step(neuron, 0, delta=100. * u.pA)

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
            neuron = hh_psc_alpha_clopath(
                1, I_e=0. * u.pA, tau_syn_ex=tau_ex_ms * u.ms,
                g_Na=0. * u.nS, g_K=0. * u.nS, g_L=0. * u.nS,
            )
            neuron.init_state()

            self._step(neuron, 0, delta=1. * u.pA)

            I_trace = []
            for k in range(1, 200):
                self._step(neuron, k)
                I_trace.append(_I_pA(neuron.I_syn_ex.value))

            peak = max(I_trace)
            self.assertAlmostEqual(peak, 1.0, delta=0.05)

    def test_stim_current_buffering(self):
        r"""Stimulation current should be buffered one step."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA)
            neuron.init_state()

            self._step(neuron, 0, x=500. * u.pA)

            I_stim = _I_pA(neuron.I_stim.value)
            self.assertAlmostEqual(I_stim, 500.0, delta=1e-10)


class TestHHClopathVoltageTraces(unittest.TestCase):
    r"""Test Clopath low-pass filtered voltage traces."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            if u.get_mantissa(delta) >= 0:
                neuron.add_delta_input(f'delta_{step_idx}', delta, label='w_ex')
            else:
                neuron.add_delta_input(f'delta_{step_idx}', -delta, label='w_in')
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_clopath_traces_track_voltage(self):
        r"""u_bar_plus, u_bar_minus should track V_m over time.

        Starting from 0 mV initial values, the filtered traces should
        approach V_m = -65 mV (resting) when there is no input.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA)
            neuron.init_state()

            # Run for many steps to let filtered voltages approach V_m
            for k in range(3000):
                self._step(neuron, k)

            V = _V_mV(neuron)
            ubp = _ubp_mV(neuron)
            ubm = _ubm_mV(neuron)
            ubb = _ubb_mV(neuron)

            # All should be close to resting potential.
            # u_bar_bar has tau=500ms so converges very slowly; use wider tolerance.
            self.assertAlmostEqual(ubp, V, delta=1.0)
            self.assertAlmostEqual(ubm, V, delta=1.0)
            self.assertAlmostEqual(ubb, V, delta=10.0)

    def test_u_bar_minus_faster_than_u_bar_plus(self):
        r"""u_bar_minus (tau=10ms) should converge faster than u_bar_plus (tau=114ms)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA)
            neuron.init_state()

            # Run for ~50 ms (500 steps)
            for k in range(500):
                self._step(neuron, k)

            V = _V_mV(neuron)
            ubp = _ubp_mV(neuron)
            ubm = _ubm_mV(neuron)

            # u_bar_minus should be closer to V than u_bar_plus
            err_ubm = abs(ubm - V)
            err_ubp = abs(ubp - V)
            self.assertLess(err_ubm, err_ubp,
                            "u_bar_minus should converge faster than u_bar_plus")

    def test_u_bar_bar_tracks_u_bar_minus(self):
        r"""u_bar_bar low-pass filters u_bar_minus, not V_m directly."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA)
            neuron.init_state()

            # Run for a short time so there's a difference
            for k in range(100):
                self._step(neuron, k)

            ubm = _ubm_mV(neuron)
            ubb = _ubb_mV(neuron)

            # u_bar_bar should be between 0 (initial) and u_bar_minus
            # since tau_u_bar_bar=500ms is very slow
            self.assertLess(abs(ubb), abs(ubm) + 1.0,
                            "u_bar_bar should lag behind u_bar_minus")

    def test_clopath_traces_multi_step_reference(self):
        r"""Multi-step Clopath trace integration should match reference solver."""
        n_steps = 50
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=500. * u.pA)
            neuron.init_state()

            ubp_model = []
            ubm_model = []
            ubb_model = []
            for k in range(n_steps):
                self._step(neuron, k)
                ubp_model.append(_ubp_mV(neuron))
                ubm_model.append(_ubm_mV(neuron))
                ubb_model.append(_ubb_mV(neuron))

            # Reference integration
            V0 = -65.0
            alpha_n = (0.01 * (V0 + 55.0)) / (1.0 - math.exp(-(V0 + 55.0) / 10.0))
            beta_n = 0.125 * math.exp(-(V0 + 65.0) / 80.0)
            alpha_m = (0.1 * (V0 + 40.0)) / (1.0 - math.exp(-(V0 + 40.0) / 10.0))
            beta_m = 4.0 * math.exp(-(V0 + 65.0) / 18.0)
            alpha_h = 0.07 * math.exp(-(V0 + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + math.exp(-(V0 + 35.0) / 10.0))

            m0 = alpha_m / (alpha_m + beta_m)
            h0 = alpha_h / (alpha_h + beta_h)
            n0 = alpha_n / (alpha_n + beta_n)

            y = np.array([V0, m0, h0, n0, 0., 0., 0., 0., 0., 0., 0.])
            ubp_ref = []
            ubm_ref = []
            ubb_ref = []
            for k in range(n_steps):
                sol = solve_ivp(
                    _nest_hh_clopath_dynamics,
                    [0.0, 0.1],
                    y,
                    method='RK45',
                    rtol=1e-3,
                    atol=1e-9,
                    args=(12000., 3600., 30., 50., -77., -54.402, 100., 500., 0.,
                          0.2, 2.0, 114.0, 10.0, 500.0),
                )
                y = sol.y[:, -1]
                ubp_ref.append(y[8])
                ubm_ref.append(y[9])
                ubb_ref.append(y[10])

            for k in range(n_steps):
                self.assertAlmostEqual(ubp_model[k], ubp_ref[k], places=6,
                                       msg=f"u_bar_plus mismatch at step {k}")
                self.assertAlmostEqual(ubm_model[k], ubm_ref[k], places=6,
                                       msg=f"u_bar_minus mismatch at step {k}")
                self.assertAlmostEqual(ubb_model[k], ubb_ref[k], places=6,
                                       msg=f"u_bar_bar mismatch at step {k}")

    def test_custom_clopath_initial_values(self):
        r"""Test that custom initial Clopath trace values are respected."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(
                1, u_bar_plus_init=-50. * u.mV,
                u_bar_minus_init=-60. * u.mV,
                u_bar_bar_init=-55. * u.mV,
            )
            neuron.init_state()

            self.assertAlmostEqual(_ubp_mV(neuron), -50.0, places=10)
            self.assertAlmostEqual(_ubm_mV(neuron), -60.0, places=10)
            self.assertAlmostEqual(_ubb_mV(neuron), -55.0, places=10)

    def test_clopath_traces_respond_to_spike(self):
        r"""After an action potential, the filtered voltage traces should
        show a delayed increase reflecting the spike's depolarization."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(
                1, I_e=1500. * u.pA,
                u_bar_plus_init=-65. * u.mV,
                u_bar_minus_init=-65. * u.mV,
                u_bar_bar_init=-65. * u.mV,
            )
            neuron.init_state()

            ubp_pre_spike = _ubp_mV(neuron)
            ubm_pre_spike = _ubm_mV(neuron)

            # Run until we get past a spike
            spike_step = None
            for k in range(300):
                spk = self._step(neuron, k)
                if bool(u.math.all(spk > 0.0)) and spike_step is None:
                    spike_step = k

            self.assertIsNotNone(spike_step, "Should detect a spike with 1500 pA input")

            # After spiking, at least u_bar_minus (fast) should be significantly
            # shifted from the initial value
            ubm_post = _ubm_mV(neuron)
            # The filtered traces should have been pulled up by the spike
            self.assertNotAlmostEqual(ubm_post, ubm_pre_spike, delta=0.1,
                                      msg="u_bar_minus should change after spike activity")


class TestHHClopathMultiStep(unittest.TestCase):
    r"""Multi-step integration tests comparing against a reference solver."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            if u.get_mantissa(delta) >= 0:
                neuron.add_delta_input(f'delta_{step_idx}', delta, label='w_ex')
            else:
                neuron.add_delta_input(f'delta_{step_idx}', -delta, label='w_in')
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_multi_step_no_input(self):
        r"""Multiple steps without input should match reference ODE solve."""
        n_steps = 50
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=0. * u.pA)
            neuron.init_state()

            V_model = []
            for k in range(n_steps):
                self._step(neuron, k)
                V_model.append(_V_mV(neuron))

            V0 = -65.0
            alpha_n = (0.01 * (V0 + 55.0)) / (1.0 - math.exp(-(V0 + 55.0) / 10.0))
            beta_n = 0.125 * math.exp(-(V0 + 65.0) / 80.0)
            alpha_m = (0.1 * (V0 + 40.0)) / (1.0 - math.exp(-(V0 + 40.0) / 10.0))
            beta_m = 4.0 * math.exp(-(V0 + 65.0) / 18.0)
            alpha_h = 0.07 * math.exp(-(V0 + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + math.exp(-(V0 + 35.0) / 10.0))

            m0 = alpha_m / (alpha_m + beta_m)
            h0 = alpha_h / (alpha_h + beta_h)
            n0 = alpha_n / (alpha_n + beta_n)

            y = np.array([V0, m0, h0, n0, 0., 0., 0., 0., 0., 0., 0.])
            V_ref = []
            for k in range(n_steps):
                sol = solve_ivp(
                    _nest_hh_clopath_dynamics,
                    [0.0, 0.1],
                    y,
                    method='RK45',
                    rtol=1e-3,
                    atol=1e-9,
                    args=(12000., 3600., 30., 50., -77., -54.402, 100., 0., 0.,
                          0.2, 2.0, 114.0, 10.0, 500.0),
                )
                y = sol.y[:, -1]
                V_ref.append(y[0])

            for k in range(n_steps):
                self.assertAlmostEqual(V_model[k], V_ref[k], places=6,
                                       msg=f"V mismatch at step {k}")

    def test_dc_spiking_trajectory(self):
        r"""With strong DC, verify the model produces action potentials with
        reasonable peak voltage and recovery."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=1000. * u.pA)
            neuron.init_state()

            V_trace = []
            for k in range(500):
                self._step(neuron, k)
                V_trace.append(_V_mV(neuron))

            V_max = max(V_trace)
            V_min = min(V_trace)

            self.assertGreater(V_max, 20.0, "AP peak should exceed 20 mV")
            self.assertLess(V_min, -65.0, "AHP should be below -65 mV")

    def test_firing_rate_increases_with_current(self):
        r"""Firing rate should increase monotonically with input current."""
        with brainstate.environ.context(dt=self.dt):
            rates = []
            for I_amp in [500., 1000., 1500.]:
                neuron = hh_psc_alpha_clopath(1, I_e=I_amp * u.pA)
                neuron.init_state()

                for k in range(1000):
                    self._step(neuron, k)

                n_spikes = 0
                for k in range(1000, 11000):
                    spk = self._step(neuron, k)
                    if bool(u.math.all(spk > 0.0)):
                        n_spikes += 1

                rates.append(n_spikes)

            for i in range(1, len(rates)):
                self.assertGreaterEqual(rates[i], rates[i - 1],
                                        f"Rate at {[500, 1000, 1500][i]} pA should be >= rate at "
                                        f"{[500, 1000, 1500][i - 1]} pA")


class TestHHClopathEdgeCases(unittest.TestCase):
    r"""Test edge cases and special configurations."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            if u.get_mantissa(delta) >= 0:
                neuron.add_delta_input(f'delta_{step_idx}', delta, label='w_ex')
            else:
                neuron.add_delta_input(f'delta_{step_idx}', -delta, label='w_in')
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_custom_initial_gating(self):
        r"""Test that custom initial gating variables are used correctly."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, Act_m_init=0.5, Inact_h_init=0.3, Act_n_init=0.4)
            neuron.init_state()

            self.assertAlmostEqual(_get_scalar(neuron.m.value), 0.5, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.h.value), 0.3, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.n.value), 0.4, places=10)

    def test_population_size(self):
        r"""Test with a population of neurons."""
        with brainstate.environ.context(dt=self.dt):
            n_neurons = 5
            neuron = hh_psc_alpha_clopath(n_neurons, I_e=1000. * u.pA)
            neuron.init_state()

            for k in range(100):
                spk = self._step(neuron, k)

            V = np.asarray(u.math.asarray(neuron.V.value / u.mV))
            self.assertEqual(V.shape, (n_neurons,))
            for i in range(1, n_neurons):
                self.assertAlmostEqual(float(V[i]), float(V[0]), places=10)

            # Clopath traces should also be consistent across population
            ubp = np.asarray(u.math.asarray(neuron.u_bar_plus.value / u.mV))
            self.assertEqual(ubp.shape, (n_neurons,))
            for i in range(1, n_neurons):
                self.assertAlmostEqual(float(ubp[i]), float(ubp[0]), places=10)

    def test_zero_refractory_period(self):
        r"""With t_ref=0, spikes should not be suppressed by refractoriness."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=1500. * u.pA, t_ref=0. * u.ms)
            neuron.init_state()

            spike_detected = False
            for k in range(200):
                spk = self._step(neuron, k)
                if bool(u.math.all(spk > 0.0)):
                    spike_detected = True
                    break

            self.assertTrue(spike_detected)

    def test_last_spike_time_updated(self):
        r"""Verify that last_spike_time is updated on spike emission."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha_clopath(1, I_e=1500. * u.pA)
            neuron.init_state()

            initial_spk_time = _get_scalar(u.math.asarray(neuron.last_spike_time.value / u.ms))
            self.assertLess(initial_spk_time, -1e6)

            for k in range(200):
                spk = self._step(neuron, k)
                if bool(u.math.all(spk > 0.0)):
                    t_spike = _get_scalar(u.math.asarray(neuron.last_spike_time.value / u.ms))
                    expected_t = (k + 1) * 0.1
                    self.assertAlmostEqual(t_spike, expected_t, delta=1e-10)
                    break

    def test_matches_hh_psc_alpha_without_clopath(self):
        r"""When Clopath variables are ignored, the model should produce
        very similar V_m, m, h, n dynamics as hh_psc_alpha.

        Note: The 11-dim (clopath) and 8-dim (base) ODE systems produce
        slightly different adaptive step-size choices, so we use a relative
        error tolerance of 1e-3 (0.1%) rather than exact decimal place matching.
        """
        from brainpy.state import hh_psc_alpha

        n_steps = 100
        with brainstate.environ.context(dt=self.dt):
            neuron_clopath = hh_psc_alpha_clopath(1, I_e=500. * u.pA)
            neuron_clopath.init_state()

            neuron_base = hh_psc_alpha(1, I_e=500. * u.pA)
            neuron_base.init_state()

            V_c_trace = []
            V_b_trace = []
            for k in range(n_steps):
                with brainstate.environ.context(t=k * self.dt):
                    neuron_clopath.update(x=0. * u.pA)
                    neuron_base.update(x=0. * u.pA)

                V_c_trace.append(_V_mV(neuron_clopath))
                V_b_trace.append(_get_scalar(u.math.asarray(neuron_base.V.value / u.mV)))

            V_c_arr = np.array(V_c_trace)
            V_b_arr = np.array(V_b_trace)

            # Relative error should be within 0.1% for most steps
            max_abs_diff = np.max(np.abs(V_c_arr - V_b_arr))
            V_range = np.max(V_b_arr) - np.min(V_b_arr)
            rel_error = max_abs_diff / max(V_range, 1.0)
            self.assertLess(rel_error, 1e-3,
                            f"Relative error {rel_error:.2e} exceeds 0.1% of voltage range")


if __name__ == '__main__':
    unittest.main()
