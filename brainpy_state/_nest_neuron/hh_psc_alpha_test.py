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

r"""Tests for NEST-compatible hh_psc_alpha neuron model.

Tests cover:
- Default parameter values matching NEST
- Parameter validation
- Subthreshold dynamics (ODE integration correctness)
- Spike detection (threshold-and-local-maximum search)
- Refractory period behavior
- Synaptic current dynamics (alpha-shaped PSCs)
- DC-driven spiking and firing rate behavior

All tests use float64 precision on CPU to match NEST's numerical behavior.
"""

import math
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainpy.state import hh_psc_alpha
from scipy.integrate import solve_ivp

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _nest_hh_dynamics(t, y, g_Na, g_K, g_L, E_Na, E_K, E_L, C_m, I_e, I_stim, tau_ex, tau_in):
    r"""Reference HH dynamics matching NEST hh_psc_alpha_dynamics exactly."""
    V = y[0]
    m = y[1]
    h = y[2]
    n = y[3]
    dI_ex = y[4]
    I_ex = y[5]
    dI_in = y[6]
    I_in = y[7]

    alpha_n = (0.01 * (V + 55.0)) / (1.0 - math.exp(-(V + 55.0) / 10.0))
    beta_n = 0.125 * math.exp(-(V + 65.0) / 80.0)
    alpha_m = (0.1 * (V + 40.0)) / (1.0 - math.exp(-(V + 40.0) / 10.0))
    beta_m = 4.0 * math.exp(-(V + 65.0) / 18.0)
    alpha_h = 0.07 * math.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))

    I_Na = g_Na * m ** 3 * h * (V - E_Na)
    I_K = g_K * n ** 4 * (V - E_K)
    I_L = g_L * (V - E_L)

    f = np.zeros(8)
    f[0] = (-(I_Na + I_K + I_L) + I_stim + I_e + I_ex + I_in) / C_m
    f[1] = alpha_m * (1.0 - m) - beta_m * m
    f[2] = alpha_h * (1.0 - h) - beta_h * h
    f[3] = alpha_n * (1.0 - n) - beta_n * n
    f[4] = -dI_ex / tau_ex
    f[5] = dI_ex - (I_ex / tau_ex)
    f[6] = -dI_in / tau_in
    f[7] = dI_in - (I_in / tau_in)
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


def _I_pA(state_val):
    r"""Get current value as scalar float in pA."""
    return _get_scalar(u.math.asarray(state_val / u.pA))


class TestHHPscAlphaDefaults(unittest.TestCase):
    r"""Test that default parameter values match NEST hh_psc_alpha."""

    def test_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_psc_alpha(1)
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

    def test_initial_state_values(self):
        r"""Initial V should be -65 mV; gating at equilibrium for V=-65."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_psc_alpha(1)
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


class TestHHPscAlphaValidation(unittest.TestCase):
    r"""Test parameter validation."""

    def test_negative_capacitance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha(1, C_m=-100. * u.pF)

    def test_zero_capacitance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha(1, C_m=0. * u.pF)

    def test_negative_refractory(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha(1, t_ref=-1. * u.ms)

    def test_zero_refractory_ok(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = hh_psc_alpha(1, t_ref=0. * u.ms)
            self.assertAlmostEqual(float(u.math.asarray(neuron.t_ref / u.ms)), 0.0)

    def test_zero_tau_syn(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha(1, tau_syn_ex=0. * u.ms)
            with self.assertRaises(ValueError):
                hh_psc_alpha(1, tau_syn_in=0. * u.ms)

    def test_negative_conductance(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                hh_psc_alpha(1, g_Na=-1. * u.nS)
            with self.assertRaises(ValueError):
                hh_psc_alpha(1, g_K=-1. * u.nS)
            with self.assertRaises(ValueError):
                hh_psc_alpha(1, g_L=-1. * u.nS)


class TestHHPscAlphaSubthreshold(unittest.TestCase):
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
        r"""Test that neuron relaxes toward resting potential without input."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=0. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)

            brainstate.transform.for_loop(_run_step, jnp.arange(100))

            V_final = _V_mV(neuron)
            self.assertAlmostEqual(V_final, -65.0, delta=1.0)

    def test_ode_integration_matches_reference(self):
        r"""Verify that one step of our model matches a reference RK45 solve."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=500. * u.pA)
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

            y0 = np.array([V0, m0, h0, n0, 0., 0., 0., 0.])
            sol = solve_ivp(
                _nest_hh_dynamics,
                [0.0, 0.1],
                y0,
                method='RK45',
                rtol=1e-3,
                atol=1e-9,
                args=(12000., 3600., 30., 50., -77., -54.402, 100., 500., 0., 0.2, 2.0),
            )
            yf = sol.y[:, -1]

            V_model = _V_mV(neuron)
            m_model = _get_scalar(neuron.m.value)
            h_model = _get_scalar(neuron.h.value)
            n_model = _get_scalar(neuron.n.value)

            self.assertAlmostEqual(V_model, yf[0], places=5)
            self.assertAlmostEqual(m_model, yf[1], places=6)
            self.assertAlmostEqual(h_model, yf[2], places=6)
            self.assertAlmostEqual(n_model, yf[3], places=6)

    def test_dc_drives_depolarization(self):
        r"""Strong DC input should depolarize the membrane."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=1000. * u.pA)
            neuron.init_state()

            V_init = _V_mV(neuron)

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)

            brainstate.transform.for_loop(_run_step, jnp.arange(10))

            V_after = _V_mV(neuron)
            self.assertGreater(V_after, V_init)


class TestHHPscAlphaSpiking(unittest.TestCase):
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
            neuron = hh_psc_alpha(1, I_e=1000. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            spk_all = brainstate.transform.for_loop(_run_step, jnp.arange(200))
            spk_arr = np.asarray(u.get_mantissa(spk_all[:, 0]))
            self.assertTrue(np.any(spk_arr > 0.0), "Neuron should fire with 1000 pA DC input within 20 ms")

    def test_no_spike_without_input(self):
        r"""With no input, the neuron should not spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=0. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            spk_all = brainstate.transform.for_loop(_run_step, jnp.arange(500))
            spk_arr = np.asarray(u.get_mantissa(spk_all[:, 0]))
            self.assertFalse(np.any(spk_arr > 0.0), "No spike expected without input")

    def test_spike_detection_logic(self):
        r"""Verify the threshold + local maximum spike detection logic."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=1500. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return neuron.V.value / u.mV, spk

            results = brainstate.transform.for_loop(_run_step, jnp.arange(300))
            V_trace = np.asarray(results[0][:, 0])
            spk_arr = np.asarray(u.get_mantissa(results[1][:, 0]))

            spike_times = np.where(spk_arr > 0.0)[0] * 0.1
            self.assertGreater(len(spike_times), 0)
            V_max = float(V_trace.max())
            self.assertGreater(V_max, 0.0, "V should exceed 0 mV during action potential")

    def test_refractory_period(self):
        r"""After a spike, no more spikes should occur for t_ref ms."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=1500. * u.pA, t_ref=5. * u.ms)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            spk_all = brainstate.transform.for_loop(_run_step, jnp.arange(500))
            spk_arr = np.asarray(u.get_mantissa(spk_all[:, 0]))
            spike_steps = np.where(spk_arr > 0.0)[0]
            spike_times = spike_steps * 0.1

            self.assertGreater(len(spike_times), 1, "Expected multiple spikes with strong DC input")

            for i in range(1, len(spike_times)):
                isi = spike_times[i] - spike_times[i - 1]
                self.assertGreaterEqual(isi, 5.0 - 0.1,
                                        f"ISI {isi:.1f} ms violates refractory period of 5 ms")

    def test_refractory_counter_decrements(self):
        r"""Refractory counter should decrement each step after spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=1500. * u.pA, t_ref=2. * u.ms)
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
            neuron = hh_psc_alpha(1, I_e=1500. * u.pA, t_ref=5. * u.ms)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return neuron.V.value / u.mV, spk

            # Run 320 steps (enough to get spike + 20 post-spike steps)
            results = brainstate.transform.for_loop(_run_step, jnp.arange(320))
            V_all = np.asarray(results[0][:, 0])
            spk_arr = np.asarray(u.get_mantissa(results[1][:, 0]))

            spike_indices = np.where(spk_arr > 0.0)[0]
            self.assertGreater(len(spike_indices), 0, "Should detect a spike")
            first_spike = int(spike_indices[0])

            # Check that V evolves during post-spike (refractory) steps
            post_spike_V = V_all[first_spike:first_spike + 20]
            V_changed = np.any(np.abs(np.diff(post_spike_V)) > 1e-6)
            self.assertTrue(V_changed, "V should evolve during refractory period in HH model")


class TestHHPscAlphaSynaptic(unittest.TestCase):
    r"""Test synaptic current dynamics (alpha-shaped PSCs)."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_excitatory_spike_input(self):
        r"""A positive weight spike input should increase dI_syn_ex."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=0. * u.pA)
            neuron.init_state()

            # Pre-compute per-step excitatory delta: 0 pA at step 0, 100 pA at step 1.
            # Apply manually inside for_loop to avoid two separate JIT compilations.
            pscon_ex = np.e / neuron.tau_syn_ex  # Quantity, units 1/ms
            delta_ex = jnp.array([0., 100.]) * u.pA  # shape (2,), units pA

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                neuron.dI_syn_ex.value = neuron.dI_syn_ex.value + pscon_ex * delta_ex[k]
                return neuron.dI_syn_ex.value / (u.pA / u.ms)

            dI_trace = np.asarray(brainstate.transform.for_loop(_run_step, jnp.arange(2))[:, 0])

            dI_before = float(dI_trace[0])
            self.assertAlmostEqual(dI_before, 0.0, places=10)

            dI_after = float(dI_trace[1])
            self.assertGreater(dI_after, 0.0, "dI_syn_ex should be positive after excitatory input")

    def test_inhibitory_spike_input(self):
        r"""A negative weight spike input should increase (magnitude) dI_syn_in."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=0. * u.pA)
            neuron.init_state()

            # Pre-compute per-step inhibitory delta: 0 pA at step 0, -50 pA at step 1.
            pscon_in = np.e / neuron.tau_syn_in  # Quantity, units 1/ms
            delta_in = jnp.array([0., -50.]) * u.pA  # shape (2,), units pA

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                neuron.dI_syn_in.value = neuron.dI_syn_in.value + pscon_in * delta_in[k]
                return neuron.dI_syn_in.value / (u.pA / u.ms)

            dI_trace = np.asarray(brainstate.transform.for_loop(_run_step, jnp.arange(2))[:, 0])

            dI_in = float(dI_trace[1])
            self.assertLess(dI_in, 0.0, "dI_syn_in should be negative after inhibitory input")

    def test_alpha_psc_waveform(self):
        r"""Test that the synaptic current has an alpha-function shape."""
        tau_ex_ms = 2.0
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=0. * u.pA, tau_syn_ex=tau_ex_ms * u.ms)
            neuron.init_state()

            # Pre-compute per-step excitatory delta: 100 pA at step 0, 0 elsewhere.
            # Merged into a single for_loop to avoid two separate JIT compilations.
            pscon_ex = np.e / neuron.tau_syn_ex  # Quantity, units 1/ms
            n_steps = 100
            delta_ex = jnp.zeros(n_steps).at[0].set(100.) * u.pA  # shape (100,), units pA

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                neuron.dI_syn_ex.value = neuron.dI_syn_ex.value + pscon_ex * delta_ex[k]
                return neuron.I_syn_ex.value / u.pA

            # I_trace[k] = I_syn_ex after step k; step k → time k * dt_ms
            I_trace = np.asarray(brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))[:, 0])

            peak_idx = int(np.argmax(I_trace))
            peak_time = peak_idx * 0.1  # step k → time k * 0.1 ms

            self.assertAlmostEqual(peak_time, tau_ex_ms, delta=0.5)
            self.assertGreater(float(I_trace[peak_idx]), float(I_trace[-1]))

    def test_psc_normalization(self):
        r"""A spike with weight 1 should produce peak current ~1 pA."""
        tau_ex_ms = 2.0
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(
                1, I_e=0. * u.pA, tau_syn_ex=tau_ex_ms * u.ms,
                g_Na=0. * u.nS, g_K=0. * u.nS, g_L=0. * u.nS,
            )
            neuron.init_state()

            # Pre-compute per-step excitatory delta: 1 pA at step 0, 0 elsewhere.
            pscon_ex = np.e / neuron.tau_syn_ex  # Quantity, units 1/ms
            n_steps = 200
            delta_ex = jnp.zeros(n_steps).at[0].set(1.) * u.pA  # shape (200,), units pA

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                neuron.dI_syn_ex.value = neuron.dI_syn_ex.value + pscon_ex * delta_ex[k]
                return neuron.I_syn_ex.value / u.pA

            I_trace = np.asarray(brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))[:, 0])

            peak = float(I_trace.max())
            self.assertAlmostEqual(peak, 1.0, delta=0.05)

    def test_stim_current_buffering(self):
        r"""Stimulation current should be buffered one step."""
        with brainstate.environ.context(dt=self.dt):
            neuron = hh_psc_alpha(1, I_e=0. * u.pA)
            neuron.init_state()

            self._step(neuron, 0, x=500. * u.pA)

            I_stim = _I_pA(neuron.I_stim.value)
            self.assertAlmostEqual(I_stim, 500.0, delta=1e-10)


class TestHHPscAlphaMultiStep(unittest.TestCase):
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
            neuron = hh_psc_alpha(1, I_e=0. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                return neuron.V.value / u.mV

            V_model = list(np.asarray(brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))[:, 0]))

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

            y = np.array([V0, m0, h0, n0, 0., 0., 0., 0.])
            V_ref = []
            for k in range(n_steps):
                sol = solve_ivp(
                    _nest_hh_dynamics,
                    [0.0, 0.1],
                    y,
                    method='RK45',
                    rtol=1e-3,
                    atol=1e-9,
                    args=(12000., 3600., 30., 50., -77., -54.402, 100., 0., 0., 0.2, 2.0),
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
            neuron = hh_psc_alpha(1, I_e=1000. * u.pA)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0. * u.pA)
                return neuron.V.value / u.mV

            V_trace = np.asarray(brainstate.transform.for_loop(_run_step, jnp.arange(500))[:, 0])

            V_max = float(V_trace.max())
            V_min = float(V_trace.min())

            self.assertGreater(V_max, 20.0, "AP peak should exceed 20 mV")
            self.assertLess(V_min, -65.0, "AHP should be below -65 mV")

    def test_firing_rate_increases_with_current(self):
        r"""Firing rate should increase monotonically with input current."""
        I_amp_vals = [500., 1000., 1500.]
        with brainstate.environ.context(dt=self.dt):
            # Batch all 3 amplitudes into a single 3-neuron population so warm-up
            # and measurement run in parallel with a single JIT compilation.
            I_amps = u.math.asarray(jnp.array(I_amp_vals)) * u.pA  # shape (3,)
            neuron = hh_psc_alpha(3, I_e=I_amps)
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            # Warm-up phase (all 3 amplitudes in parallel)
            brainstate.transform.for_loop(_run_step, jnp.arange(1000))
            # Measurement phase
            spk_all = brainstate.transform.for_loop(_run_step, jnp.arange(1000, 11000))
            # spk_all shape: (10000, 3)
            spk_arr = np.asarray(u.get_mantissa(spk_all))
            rates = [int(np.sum(spk_arr[:, i] > 0.0)) for i in range(3)]

        for i in range(1, len(rates)):
            self.assertGreaterEqual(rates[i], rates[i - 1],
                                    f"Rate at {I_amp_vals[i]} pA should be >= rate at "
                                    f"{I_amp_vals[i - 1]} pA")


class TestHHPscAlphaEdgeCases(unittest.TestCase):
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
            neuron = hh_psc_alpha(1, Act_m_init=0.5, Inact_h_init=0.3, Act_n_init=0.4)
            neuron.init_state()

            self.assertAlmostEqual(_get_scalar(neuron.m.value), 0.5, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.h.value), 0.3, places=10)
            self.assertAlmostEqual(_get_scalar(neuron.n.value), 0.4, places=10)

    def test_population_size(self):
        r"""Test with a population of neurons."""
        with brainstate.environ.context(dt=self.dt):
            n_neurons = 5
            neuron = hh_psc_alpha(n_neurons, I_e=1000. * u.pA)
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
            neuron = hh_psc_alpha(1, I_e=1500. * u.pA, t_ref=0. * u.ms)
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
            neuron = hh_psc_alpha(1, I_e=1500. * u.pA)
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


if __name__ == '__main__':
    unittest.main()
