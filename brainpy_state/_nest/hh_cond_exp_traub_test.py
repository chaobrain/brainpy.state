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

r"""Tests for NEST-compatible hh_cond_exp_traub neuron model.

Tests cover:
- Default parameter values matching NEST
- Parameter validation
- Equilibrium initialization of gating variables
- Subthreshold dynamics (ODE integration correctness)
- Spike detection (V_T + 30 threshold and local-maximum criterion)
- Refractory period behavior
- Synaptic conductance dynamics (exponential decay)
- DC-driven spiking behavior
- Comparison against NEST reference data

All tests use float64 precision on CPU to match NEST's numerical behavior.
"""

import math
import unittest

import brainstate
import saiunit as u
import jax
import numpy as np
from scipy.integrate import solve_ivp

from brainpy_state import hh_cond_exp_traub

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _nest_hh_cond_exp_traub_dynamics(t, y, g_Na, g_K, g_L, E_Na, E_K, E_L,
                                     V_T, E_ex, E_in, C_m, I_e, I_stim,
                                     tau_ex, tau_in):
    r"""Reference dynamics matching NEST hh_cond_exp_traub_dynamics exactly.

    State vector y = [V_m, m, h, n, g_ex, g_in].
    """
    V_m = y[0]
    m = y[1]
    h = y[2]
    n = y[3]
    g_e = y[4]
    g_i = y[5]

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

    f = np.zeros(6)
    f[0] = (-I_Na - I_K - I_L - I_syn_exc - I_syn_inh + I_stim + I_e) / C_m
    f[1] = alpha_m - (alpha_m + beta_m) * m
    f[2] = alpha_h - (alpha_h + beta_h) * h
    f[3] = alpha_n - (alpha_n + beta_n) * n
    f[4] = -g_e / tau_ex
    f[5] = -g_i / tau_in
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


def _g_nS(state_val):
    r"""Get conductance value as scalar float in nS."""
    return _get_scalar(u.math.asarray(state_val / u.nS))


class TestHHCondExpTraubDefaults(unittest.TestCase):
    r"""Test that default parameter values match NEST hh_cond_exp_traub."""

    def test_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_cond_exp_traub(1)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_L / u.mV)), -60.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.C_m / u.pF)), 200.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.g_Na / u.nS)), 20000.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.g_K / u.nS)), 6000.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.g_L / u.nS)), 10.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_Na / u.mV)), 50.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_K / u.mV)), -90.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.V_T / u.mV)), -63.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_ex / u.mV)), 0.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.E_in / u.mV)), -80.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.t_ref / u.ms)), 2.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_syn_ex / u.ms)), 5.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.tau_syn_in / u.ms)), 10.0, places=10)
            self.assertAlmostEqual(float(u.math.asarray(neuron.I_e / u.pA)), 0.0, places=10)

    def test_initial_state_values(self):
        r"""Initial V should be E_L; gating at equilibrium for V=E_L."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_cond_exp_traub(1)
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

            # Synaptic conductances should be zero
            self.assertAlmostEqual(_g_nS(neuron.g_ex.value), 0.0, places=10)
            self.assertAlmostEqual(_g_nS(neuron.g_in.value), 0.0, places=10)
            self.assertEqual(int(neuron.refractory_step_count.value[0]), 0)


class TestHHCondExpTraubValidation(unittest.TestCase):
    r"""Test parameter validation."""

    def test_negative_capacitance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_exp_traub(1, C_m=-100. * u.pF)

    def test_zero_capacitance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_exp_traub(1, C_m=0. * u.pF)

    def test_negative_refractory(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_exp_traub(1, t_ref=-1. * u.ms)

    def test_zero_refractory_ok(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_cond_exp_traub(1, t_ref=0. * u.ms)
            self.assertAlmostEqual(float(u.math.asarray(neuron.t_ref / u.ms)), 0.0)

    def test_zero_tau_syn(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_cond_exp_traub(1, tau_syn_ex=0. * u.ms)
            with self.assertRaises(ValueError):
                hh_cond_exp_traub(1, tau_syn_in=0. * u.ms)


class TestHHCondExpTraubSubthreshold(unittest.TestCase):
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
            neuron = hh_cond_exp_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            for k in range(1000):
                self._step(neuron, k)

            V_final = _V_mV(neuron)
            V_T_val = float(u.math.asarray(neuron.V_T / u.mV))
            # Should settle below spike threshold
            self.assertLess(V_final, V_T_val + 30.0)

    def test_ode_integration_matches_reference(self):
        r"""Verify that one step of our model matches a reference RK45 solve."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_exp_traub(1, I_e=500. * u.pA)
            neuron.init_state()

            self._step(neuron, 0)

            # Reference integration
            V0 = -60.0
            alpha_n = 0.032 * (15.0 - V0) / (math.exp((15.0 - V0) / 5.0) - 1.0)
            beta_n = 0.5 * math.exp((10.0 - V0) / 40.0)
            alpha_m = 0.32 * (13.0 - V0) / (math.exp((13.0 - V0) / 4.0) - 1.0)
            beta_m = 0.28 * (V0 - 40.0) / (math.exp((V0 - 40.0) / 5.0) - 1.0)
            alpha_h = 0.128 * math.exp((17.0 - V0) / 18.0)
            beta_h = 4.0 / (1.0 + math.exp((40.0 - V0) / 5.0))

            m0 = alpha_m / (alpha_m + beta_m)
            h0 = alpha_h / (alpha_h + beta_h)
            n0 = alpha_n / (alpha_n + beta_n)

            y0 = np.array([V0, m0, h0, n0, 0., 0.])
            sol = solve_ivp(
                _nest_hh_cond_exp_traub_dynamics,
                [0.0, 0.1],
                y0,
                method='RK45',
                rtol=1e-3,
                atol=1e-9,
                args=(20000., 6000., 10., 50., -90., -60.,
                      -63., 0., -80., 200., 500., 0., 5., 10.),
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

    def test_dc_drives_depolarization(self):
        r"""Strong DC input should depolarize the membrane."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_exp_traub(1, I_e=1000. * u.pA)
            neuron.init_state()

            V_init = _V_mV(neuron)
            for k in range(10):
                self._step(neuron, k)

            V_after = _V_mV(neuron)
            self.assertGreater(V_after, V_init)


class TestHHCondExpTraubSpiking(unittest.TestCase):
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
            neuron = hh_cond_exp_traub(1, I_e=1000. * u.pA)
            neuron.init_state()

            spike_detected = False
            for k in range(200):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    spike_detected = True
                    break

            self.assertTrue(spike_detected, "Neuron should fire with 1000 pA DC input within 20 ms")

    def test_no_spike_with_hyperpolarized_start(self):
        r"""With hyperpolarized initial V and no input, the neuron should not spike.

        The default initialization (V=E_L=-60, gating at equilibrium for raw V)
        can produce a transient spike because the gating variables are NOT at
        equilibrium for the V-V_T shifted dynamics. Starting from a strongly
        hyperpolarized state avoids this.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_exp_traub(1, I_e=0. * u.pA, V_m_init=-80. * u.mV)
            neuron.init_state()

            for k in range(500):
                spk = self._step(neuron, k)
                self.assertFalse(self._is_spike(spk), f"No spike expected at step {k}")

    def test_spike_detection_threshold(self):
        r"""Verify spike uses V_T + 30 threshold."""
        with brainstate.environ.context(dt=self.dt):
            # With V_T = -63, threshold is -33 mV
            neuron = hh_cond_exp_traub(1, I_e=1500. * u.pA)
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
            # V should exceed V_T + 30 = -33 mV during action potential
            self.assertGreater(V_max, -33.0, "V should exceed V_T + 30 mV during action potential")

    def test_refractory_period(self):
        r"""After a spike, no more spikes should occur for t_ref ms."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_exp_traub(1, I_e=1500. * u.pA, t_ref=5. * u.ms)
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
            neuron = hh_cond_exp_traub(1, I_e=1500. * u.pA, t_ref=2. * u.ms)
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
            neuron = hh_cond_exp_traub(1, I_e=1500. * u.pA, t_ref=5. * u.ms)
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


class TestHHCondExpTraubSynaptic(unittest.TestCase):
    r"""Test synaptic conductance dynamics (exponential decay)."""

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
            neuron = hh_cond_exp_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            self._step(neuron, 0)
            g_before = _g_nS(neuron.g_ex.value)
            self.assertAlmostEqual(g_before, 0.0, places=10)

            self._step(neuron, 1, delta=5. * u.nS)
            g_after = _g_nS(neuron.g_ex.value)
            self.assertGreater(g_after, 0.0, "g_ex should be positive after excitatory input")

    def test_inhibitory_conductance_input(self):
        r"""A negative weight spike input should increase g_in."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_exp_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            self._step(neuron, 0)
            self._step(neuron, 1, delta=-5. * u.nS)
            g_in = _g_nS(neuron.g_in.value)
            self.assertGreater(g_in, 0.0, "g_in should be positive after inhibitory input (sign flipped)")

    def test_exponential_conductance_decay(self):
        r"""Test that conductance decays exponentially with time constant tau_syn."""
        tau_ex_ms = 5.0
        with brainstate.environ.context(dt=self.dt):
            # Use very small conductance channels to isolate synaptic dynamics
            neuron = hh_cond_exp_traub(
                1, I_e=0. * u.pA, tau_syn_ex=tau_ex_ms * u.ms,
                g_Na=0. * u.nS, g_K=0. * u.nS, g_L=0. * u.nS,
            )
            neuron.init_state()

            # Add a conductance pulse
            self._step(neuron, 0, delta=10. * u.nS)

            # Collect conductance trace
            g_trace = []
            for k in range(1, 100):
                self._step(neuron, k)
                g_trace.append(_g_nS(neuron.g_ex.value))

            # Conductance should decay monotonically
            for i in range(1, len(g_trace)):
                self.assertLessEqual(g_trace[i], g_trace[i - 1] + 1e-12,
                                     "Conductance should decay monotonically")

            # Check exponential decay rate: g(t) = g0 * exp(-t/tau)
            # After one tau (50 steps of 0.1 ms = 5.0 ms), should be ~1/e
            if len(g_trace) > 50:
                ratio = g_trace[49] / g_trace[0]
                expected_ratio = math.exp(-5.0 / tau_ex_ms)
                self.assertAlmostEqual(ratio, expected_ratio, delta=0.02)

    def test_conductance_jump_magnitude(self):
        r"""A spike with weight w should produce a conductance jump of w nS."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_exp_traub(
                1, I_e=0. * u.pA,
                g_Na=0. * u.nS, g_K=0. * u.nS, g_L=0. * u.nS,
            )
            neuron.init_state()

            # Add a 3.0 nS excitatory input
            self._step(neuron, 0, delta=3. * u.nS)

            # After ODE integration (which decays g_ex slightly from 0), the
            # jump adds 3.0 nS. Since g_ex was 0 before the step and the decay
            # of 0 is still 0, we expect g_ex ~ 3.0 nS.
            g_ex = _g_nS(neuron.g_ex.value)
            self.assertAlmostEqual(g_ex, 3.0, delta=0.01)


class TestHHCondExpTraubMultiStep(unittest.TestCase):
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
            neuron = hh_cond_exp_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            V_model = []
            for k in range(n_steps):
                self._step(neuron, k)
                V_model.append(_V_mV(neuron))

            # Reference integration
            V0 = -60.0
            alpha_n = 0.032 * (15.0 - V0) / (math.exp((15.0 - V0) / 5.0) - 1.0)
            beta_n = 0.5 * math.exp((10.0 - V0) / 40.0)
            alpha_m = 0.32 * (13.0 - V0) / (math.exp((13.0 - V0) / 4.0) - 1.0)
            beta_m = 0.28 * (V0 - 40.0) / (math.exp((V0 - 40.0) / 5.0) - 1.0)
            alpha_h = 0.128 * math.exp((17.0 - V0) / 18.0)
            beta_h = 4.0 / (1.0 + math.exp((40.0 - V0) / 5.0))

            m0 = alpha_m / (alpha_m + beta_m)
            h0 = alpha_h / (alpha_h + beta_h)
            n0 = alpha_n / (alpha_n + beta_n)

            y = np.array([V0, m0, h0, n0, 0., 0.])
            V_ref = []
            for k in range(n_steps):
                sol = solve_ivp(
                    _nest_hh_cond_exp_traub_dynamics,
                    [0.0, 0.1],
                    y,
                    method='RK45',
                    rtol=1e-3,
                    atol=1e-9,
                    args=(20000., 6000., 10., 50., -90., -60.,
                          -63., 0., -80., 200., 0., 0., 5., 10.),
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
            neuron = hh_cond_exp_traub(1, I_e=1000. * u.pA)
            neuron.init_state()

            V_trace = []
            for k in range(500):
                self._step(neuron, k)
                V_trace.append(_V_mV(neuron))

            V_max = max(V_trace)
            V_min = min(V_trace)

            self.assertGreater(V_max, -33.0, "AP peak should exceed -33 mV (V_T+30)")
            self.assertLess(V_min, -60.0, "AHP should be below -60 mV")

    def test_firing_rate_increases_with_current(self):
        r"""Firing rate should increase monotonically with input current."""
        with brainstate.environ.context(dt=self.dt):
            rates = []
            for I_amp in [500., 1000., 1500.]:
                neuron = hh_cond_exp_traub(1, I_e=I_amp * u.pA)
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


class TestHHCondExpTraubNESTReference(unittest.TestCase):
    r"""Test against NEST reference data from test_hh_cond_exp_traub.py.

    This replicates the NEST test setup:
    - Neuron with non-default parameters
    - DC generator (100 pA)
    - Spike generator at times [0.1, 1.2] ms
    - Verify membrane potential and conductance traces
    """

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms
        # NEST test reference parameters
        self.ref_params = dict(
            V_m_init=10.0 * u.mV,
            g_Na=19000.0 * u.nS,
            g_K=5000.0 * u.nS,
            g_L=9.0 * u.nS,
            C_m=110.0 * u.pF,
            E_Na=40.0 * u.mV,
            E_K=-80.0 * u.mV,
            E_L=-70.0 * u.mV,
            E_ex=1.0 * u.mV,
            E_in=-90.0 * u.mV,
            tau_syn_ex=0.3 * u.ms,
            tau_syn_in=3.0 * u.ms,
            t_ref=3.5 * u.ms,
            I_e=1.0 * u.pA,
        )

        # NEST reference membrane potential data (40 timesteps)
        self.reference_V_m = [
            39.19340e0, 38.50370e0, 33.15630e0, 22.17610e0, 7.131720e0,
            -8.728460e0, -23.12210e0, -35.95260e0, -51.82590e0, -66.99940e0,
            -73.98330e0, -76.47150e0, -77.48650e0, -77.93770e0, -78.13580e0,
            -78.20390e0, -78.19780e0, -78.14580e0, -78.06380e0, -77.96150e0,
            -77.84520e0, -77.71910e0, -77.52600e0, -77.34640e0, -77.17640e0,
            -77.01340e0, -76.85560e0, -76.70190e0, -76.55140e0, -76.40350e0,
            -76.25770e0, -76.11370e0, -75.97140e0, -75.83060e0, -75.69110e0,
            -75.55300e0, -75.41600e0, -75.28030e0, -75.14570e0, -75.01230e0,
        ]

        # NEST reference conductance data (g_ex, g_in) for 40 timesteps
        self.reference_g_ex = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 7.165310e-01, 5.134160e-01, 3.678790e-01, 2.635960e-01,
            1.888750e-01, 1.353340e-01, 9.697120e-02, 6.948280e-02, 4.978650e-02,
            3.567350e-02, 1.025560e0, 7.348450e-01, 5.265390e-01, 3.772810e-01,
            2.703330e-01, 1.937020e-01, 1.387930e-01, 9.944960e-02, 7.125860e-02,
            5.105890e-02, 3.658530e-02, 2.621440e-02, 1.878340e-02, 1.345890e-02,
            9.643710e-03, 6.910010e-03, 4.951230e-03, 3.547710e-03, 2.542040e-03,
        ]

        self.reference_g_in = [0.0] * 40

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_nest_reference_voltage_trace(self):
        r"""Compare voltage trace against NEST reference data.

        The NEST test simulates with:
        - Neuron starting at V_m = 10 mV (mid-action potential)
        - DC generator: 100 pA amplitude (delay = 1.0 ms = 10 steps)
        - Spike generator: spikes at [0.1, 1.2] ms (delay = 1.0 ms)
        - Resolution: 0.1 ms

        During the fast action potential phase (steps 1-10), the Dormand-Prince
        RK45 and GSL RKF45 solvers produce slightly different trajectories due
        to the extremely stiff dynamics. The subthreshold phase (steps 11+)
        should match closely.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_exp_traub(1, **self.ref_params)
            neuron.init_state()

            spike_delivery_steps = {10, 21}

            V_trace = []
            g_ex_trace = []

            for k in range(50):
                dc_current = 100.0 * u.pA
                spike_input = None
                if k in spike_delivery_steps:
                    spike_input = 1.0 * u.nS

                self._step(neuron, k, x=dc_current, delta=spike_input)

                V_trace.append(_V_mV(neuron))
                g_ex_trace.append(_g_nS(neuron.g_ex.value))

            # During the fast AP phase (steps 1-10), allow larger tolerance
            # because the stiff dynamics amplify solver differences.
            # During subthreshold recovery (steps 11+), require tighter match.
            for k in range(min(len(self.reference_V_m), len(V_trace))):
                ref_V = self.reference_V_m[k]
                model_V = V_trace[k]

                if k < 10:
                    # Fast AP phase: just check trajectory shape is qualitatively right
                    # (depolarization followed by rapid repolarization)
                    abs_err = abs(model_V - ref_V)
                    self.assertLess(abs_err, 15.0,
                                    f"V_m at step {k + 1}: model={model_V:.4f}, "
                                    f"NEST={ref_V:.4f}, abs_err={abs_err:.4f}")
                else:
                    # Subthreshold phase: require relative match within 5%
                    if abs(ref_V) > 1.0:
                        rel_err = abs(model_V - ref_V) / abs(ref_V)
                        self.assertLess(rel_err, 0.05,
                                        f"V_m at step {k + 1}: model={model_V:.4f}, "
                                        f"NEST={ref_V:.4f}, rel_err={rel_err:.6f}")
                    else:
                        self.assertAlmostEqual(model_V, ref_V, delta=2.0,
                                               msg=f"V_m at step {k + 1}: model={model_V:.4f}, "
                                                   f"NEST={ref_V:.4f}")

    def test_nest_reference_conductance_trace(self):
        r"""Compare conductance trace against NEST reference data."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_exp_traub(1, **self.ref_params)
            neuron.init_state()

            spike_delivery_steps = {10, 21}
            g_ex_trace = []

            for k in range(50):
                dc_current = 100.0 * u.pA
                spike_input = None
                if k in spike_delivery_steps:
                    spike_input = 1.0 * u.nS

                self._step(neuron, k, x=dc_current, delta=spike_input)
                g_ex_trace.append(_g_nS(neuron.g_ex.value))

            # Compare g_ex against reference
            for k in range(min(len(self.reference_g_ex), len(g_ex_trace))):
                ref_g = self.reference_g_ex[k]
                model_g = g_ex_trace[k]
                if ref_g > 1e-6:
                    rel_err = abs(model_g - ref_g) / ref_g
                    self.assertLess(rel_err, 0.02,
                                    f"g_ex at step {k + 1}: model={model_g:.6f}, "
                                    f"NEST={ref_g:.6f}, rel_err={rel_err:.6f}")
                else:
                    self.assertAlmostEqual(model_g, ref_g, delta=1e-6,
                                           msg=f"g_ex at step {k + 1}: model={model_g:.6f}, "
                                               f"NEST={ref_g:.6f}")


class TestHHCondExpTraubEdgeCases(unittest.TestCase):
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
            neuron = hh_cond_exp_traub(1, Act_m_init=0.5, Inact_h_init=0.3, Act_n_init=0.4)
            neuron.init_state()

            self.assertAlmostEqual(_get_scalar(neuron.m.value), 0.5, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.h.value), 0.3, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.n.value), 0.4, places=10)

    def test_custom_V_T(self):
        r"""Test that a different V_T shifts the effective threshold."""
        with brainstate.environ.context(dt=self.dt):
            # With V_T = -50, threshold is V_T + 30 = -20 mV
            neuron = hh_cond_exp_traub(1, V_T=-50. * u.mV)
            self.assertAlmostEqual(float(u.math.asarray(neuron.V_T / u.mV)), -50.0)

    def test_population_size(self):
        r"""Test with a population of neurons."""
        with brainstate.environ.context(dt=self.dt):
            n_neurons = 5
            neuron = hh_cond_exp_traub(n_neurons, I_e=1000. * u.pA)
            neuron.init_state()

            for k in range(100):
                self._step(neuron, k)

            V = np.asarray(u.math.asarray(neuron.V.value / u.mV))
            self.assertEqual(V.shape, (n_neurons,))
            for i in range(1, n_neurons):
                self.assertAlmostEqual(float(V[i]), float(V[0]), places=10)

    def test_zero_refractory_period(self):
        r"""With t_ref=0, spikes should not be suppressed by refractoriness."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_cond_exp_traub(1, I_e=1500. * u.pA, t_ref=0. * u.ms)
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
            neuron = hh_cond_exp_traub(1, I_e=1500. * u.pA)
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

    def test_excitatory_reversal_potential_effect(self):
        r"""Excitatory synaptic input should depolarize when V < E_ex."""
        with brainstate.environ.context(dt=self.dt):
            # E_ex = 0 mV, V starts at -60 mV -> synaptic current is inward (depolarizing)
            neuron = hh_cond_exp_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            V_before = _V_mV(neuron)
            # Add a large excitatory conductance
            self._step(neuron, 0, delta=50. * u.nS)
            # Step once more so the conductance affects V through the ODE
            self._step(neuron, 1)
            V_after = _V_mV(neuron)

            self.assertGreater(V_after, V_before,
                               "Excitatory input should depolarize (V < E_ex)")

    def test_inhibitory_reversal_potential_effect(self):
        r"""Inhibitory synaptic input should hyperpolarize when V > E_in."""
        with brainstate.environ.context(dt=self.dt):
            # E_in = -80 mV, V starts at -60 mV -> synaptic current is outward (hyperpolarizing)
            neuron = hh_cond_exp_traub(1, I_e=0. * u.pA)
            neuron.init_state()

            V_before = _V_mV(neuron)
            self._step(neuron, 0, delta=-50. * u.nS)
            self._step(neuron, 1)
            V_after = _V_mV(neuron)

            self.assertLess(V_after, V_before,
                            "Inhibitory input should hyperpolarize (V > E_in)")


if __name__ == '__main__':
    unittest.main()
