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
Tests for gif_pop_psc_exp population neuron model.

These tests verify that the brainpy.state implementation of gif_pop_psc_exp
matches NEST's update ordering and semantics exactly, including:

- Default parameter values matching NEST C++ source
- Exact exponential integration for membrane and synaptic dynamics
- Population dynamics algorithm (escape rates, survival buffers)
- Adaptation kernel computation and decay
- Stochastic spike generation (binomial / Poisson)
- History buffer rotation and initialization
- Synaptic current handling (excitatory / inhibitory separation)
- Full reference trace comparison against standalone Python reference

All tests use float64 on CPU to match NEST's numerical behavior.
"""

import math
import os
import unittest

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import brainstate
import brainunit as u
import numpy as np

from brainpy_state._nest.gif_pop_psc_exp import gif_pop_psc_exp


# ============================================================================
# Reference implementation matching NEST C++ exactly
# ============================================================================

def _nest_ref_adaptation_kernel(k, h, tau_sfa, q_sfa):
    r"""Compute adaptation kernel value at lag k time steps (NEST adaptation_kernel)."""
    theta_tmp = 0.0
    for j in range(len(tau_sfa)):
        theta_tmp += q_sfa[j] * math.exp(-k * h / tau_sfa[j])
    return theta_tmp


def _nest_ref_get_history_size(h, tau_m, Delta_V, t_ref, tau_sfa, q_sfa):
    r"""Compute automatic history size (NEST get_history_size)."""
    tmax = 20000.0
    k = int(tmax / h)
    kmin = int(5 * tau_m / h)
    while (_nest_ref_adaptation_kernel(k, h, tau_sfa, q_sfa) / Delta_V < 0.1) and k > kmin:
        k -= 1
    if k * h <= t_ref:
        k = int(t_ref / h) + 1
    return k


def _nest_ref_escrate(x, lambda_0, Delta_V):
    r"""Escape rate (NEST escrate)."""
    return lambda_0 * math.exp(x / Delta_V)


def _nest_ref_draw_binomial(n_expect, N, rng):
    r"""Draw binomial (NEST draw_binomial)."""
    p_bino = n_expect / N
    if p_bino >= 1.0:
        return N
    elif p_bino <= 0.0:
        return 0
    else:
        return int(rng.binomial(N, p_bino))


def _nest_ref_draw_poisson(n_expect, N, rng):
    r"""Draw Poisson (NEST draw_poisson)."""
    min_double = np.finfo(np.float64).tiny
    if n_expect > N:
        return N
    elif n_expect > min_double:
        if 1.0 - (n_expect + 1.0) * math.exp(-n_expect) > min_double:
            n_t = int(rng.poisson(n_expect))
        else:
            n_t = int(rng.random() < n_expect)
        return max(0, min(n_t, N))
    else:
        return 0


def run_nest_ref(
    n_steps, h, N, tau_m, C_m, t_ref, lambda_0, Delta_V, E_L, V_reset,
    V_T_star, I_e, tau_syn_ex, tau_syn_in, tau_sfa, q_sfa,
    len_kernel=-1, BinoRand=True, rng_seed=0,
    ex_spikes=None, in_spikes=None, currents=None,
):
    r"""Full reference implementation of NEST gif_pop_psc_exp update loop.

    Returns arrays of V_m, I_syn_ex, I_syn_in, n_spikes, n_expect, theta_hat
    for each step, matching NEST's update order exactly.
    """
    rng = np.random.RandomState(rng_seed)

    # Pre-run hook computations
    P22 = math.exp(-h / tau_m)
    P20 = tau_m / C_m * (1.0 - P22)
    P11_ex = math.exp(-h / tau_syn_ex)
    P11_in = math.exp(-h / tau_syn_in)

    if len_kernel < 1:
        len_kernel = _nest_ref_get_history_size(h, tau_m, Delta_V, t_ref, tau_sfa, q_sfa)

    k_ref = int(round(t_ref / h))

    # Initialize state
    lambda_free = 0.0
    dftype = brainstate.environ.dftype()
    n_buf = np.zeros(len_kernel, dtype=dftype)
    m_buf = np.zeros(len_kernel, dtype=dftype)
    v_buf = np.zeros(len_kernel, dtype=dftype)
    u_buf = np.zeros(len_kernel, dtype=dftype)
    lambda_buf = np.zeros(len_kernel, dtype=dftype)
    theta = np.zeros(len_kernel, dtype=dftype)
    theta_tld = np.zeros(len_kernel, dtype=dftype)

    for k in range(len_kernel):
        theta_tmp = _nest_ref_adaptation_kernel(len_kernel - k, h, tau_sfa, q_sfa)
        theta[k] = theta_tmp
        theta_tld[k] = Delta_V * (1.0 - math.exp(-theta_tmp / Delta_V)) / float(N)

    n_buf[len_kernel - 1] = float(N)
    m_buf[len_kernel - 1] = float(N)
    x_ = 0.0
    z_ = 0.0
    k0 = 0

    Q30 = np.array([math.exp(-h / tau) for tau in tau_sfa], dtype=dftype)
    Q30K = np.array([
        q_sfa[j] * tau_sfa[j] * math.exp(-h * len_kernel / tau_sfa[j])
        for j in range(len(tau_sfa))
    ], dtype=dftype)
    g_ = np.zeros(len(tau_sfa), dtype=dftype)

    V_m = 0.0
    I_syn_ex = 0.0
    I_syn_in = 0.0
    y0 = 0.0
    n_expect = 0.0
    theta_hat = 0.0
    n_spikes = 0

    # Output traces
    V_m_trace = []
    I_syn_ex_trace = []
    I_syn_in_trace = []
    n_spikes_trace = []
    n_expect_trace = []
    theta_hat_trace = []

    for step in range(n_steps):
        # Get inputs for this step
        ex_sp = ex_spikes[step] if ex_spikes is not None and step < len(ex_spikes) else 0.0
        in_sp = in_spikes[step] if in_spikes is not None and step < len(in_spikes) else 0.0
        curr = currents[step] if currents is not None and step < len(currents) else 0.0

        # Main update (Fig 11 of [1])
        h_tot = (I_e + y0) * P20 + E_L  # line 6

        JNA_ex = ex_sp / h
        JNA_in = in_sp / h
        JNA_ex *= tau_syn_ex / C_m
        JNA_in *= tau_syn_in / C_m

        JNy_ex = I_syn_ex / C_m
        JNy_in = I_syn_in / C_m

        h_ex_tmpvar = (
            tau_syn_ex * P11_ex * (JNy_ex - JNA_ex)
            - P22 * (tau_syn_ex * JNy_ex - tau_m * JNA_ex)
        )
        h_in_tmpvar = (
            tau_syn_in * P11_in * (JNy_in - JNA_in)
            - P22 * (tau_syn_in * JNy_in - tau_m * JNA_in)
        )
        h_ex = tau_m * (JNA_ex + h_ex_tmpvar / (tau_syn_ex - tau_m))
        h_in = tau_m * (JNA_in + h_in_tmpvar / (tau_syn_in - tau_m))
        h_tot += h_ex + h_in

        JNy_ex = JNA_ex + (JNy_ex - JNA_ex) * P11_ex
        JNy_in = JNA_in + (JNy_in - JNA_in) * P11_in
        I_syn_ex = JNy_ex * C_m
        I_syn_in = JNy_in * C_m

        y0 = curr

        # UpdatePopulation (Fig 12 of [1])
        W_ = 0.0
        X_ = 0.0
        Y_ = 0.0
        Z_ = 0.0
        theta_hat = V_T_star

        V_m = (V_m - E_L) * P22 + h_tot

        for j in range(len(tau_sfa)):
            g_j_tmp = (1.0 - Q30[j]) * n_buf[k0] / (float(N) * h)
            g_[j] = g_[j] * Q30[j] + g_j_tmp
            theta_hat += Q30K[j] * g_[j]

        lambda_tld = _nest_ref_escrate(V_m - theta_hat, lambda_0, Delta_V)
        P_free = 1.0 - math.exp(-0.0005 * (lambda_free + lambda_tld) * h)
        lambda_free = lambda_tld
        theta_hat -= n_buf[0] * theta_tld[0]

        for km in range(len_kernel):
            X_ += m_buf[km]

        theta_hat_local = theta_hat

        for km in range(len_kernel - k_ref):
            k = (k0 + km) % len_kernel
            th = theta[km] + theta_hat_local
            theta_hat_local += n_buf[k] * theta_tld[km]
            u_buf[k] = (u_buf[k] - E_L) * P22 + h_tot
            lambda_tld_k = _nest_ref_escrate(u_buf[k] - th, lambda_0, Delta_V)

            P_lambda = 0.0005 * (lambda_tld_k + lambda_buf[k]) * h
            if P_lambda > 0.01:
                P_lambda = 1.0 - math.exp(-P_lambda)

            lambda_buf[k] = lambda_tld_k
            Y_ += P_lambda * v_buf[k]
            Z_ += v_buf[k]
            W_ += P_lambda * m_buf[k]

            ompl = 1.0 - P_lambda
            v_buf[k] = ompl * ompl * v_buf[k] + P_lambda * m_buf[k]
            m_buf[k] = ompl * m_buf[k]

        if (Z_ + z_) > 0.0:
            P_Lambda = (Y_ + P_free * z_) / (Z_ + z_)
        else:
            P_Lambda = 0.0

        n_expect = W_ + P_free * x_ + P_Lambda * (N - X_ - x_)

        if BinoRand:
            n_spikes = _nest_ref_draw_binomial(n_expect, N, rng)
        else:
            n_spikes = _nest_ref_draw_poisson(n_expect, N, rng)

        ompf = 1.0 - P_free
        z_ = ompf * ompf * z_ + x_ * P_free + v_buf[k0]
        x_ = x_ * ompf + m_buf[k0]

        n_buf[k0] = float(n_spikes)
        m_buf[k0] = float(n_spikes)
        v_buf[k0] = 0.0
        u_buf[k0] = V_reset
        lambda_buf[k0] = 0.0

        k0 = (k0 + 1) % len_kernel

        V_m_trace.append(V_m)
        I_syn_ex_trace.append(I_syn_ex)
        I_syn_in_trace.append(I_syn_in)
        n_spikes_trace.append(n_spikes)
        n_expect_trace.append(n_expect)
        theta_hat_trace.append(theta_hat)

    return {
        'V_m': np.array(V_m_trace),
        'I_syn_ex': np.array(I_syn_ex_trace),
        'I_syn_in': np.array(I_syn_in_trace),
        'n_spikes': np.array(n_spikes_trace),
        'n_expect': np.array(n_expect_trace),
        'theta_hat': np.array(theta_hat_trace),
    }


# ============================================================================
# Test classes
# ============================================================================

class TestGIFPopPscExpDefaultParams(unittest.TestCase):
    r"""Test that default parameters match NEST C++ source code values."""

    def test_nest_cpp_default_parameters(self):
        r"""Parameters should match NEST gif_pop_psc_exp.cpp defaults."""
        neuron = gif_pop_psc_exp(1)
        self.assertEqual(neuron.N, 100)
        self.assertEqual(neuron.tau_m, 20.0)
        self.assertEqual(neuron.C_m, 250.0)
        self.assertEqual(neuron.t_ref, 4.0)
        self.assertEqual(neuron.lambda_0, 10.0)
        self.assertEqual(neuron.Delta_V, 2.0)
        self.assertEqual(neuron.E_L, 0.0)
        self.assertEqual(neuron.V_reset, 0.0)
        self.assertEqual(neuron.V_T_star, 15.0)
        self.assertEqual(neuron.I_e, 0.0)
        self.assertEqual(neuron.tau_syn_ex, 3.0)
        self.assertEqual(neuron.tau_syn_in, 6.0)
        self.assertEqual(neuron.tau_sfa, (300.0,))
        self.assertEqual(neuron.q_sfa, (0.5,))
        self.assertEqual(neuron.BinoRand, True)
        self.assertEqual(neuron.len_kernel, -1)

    def test_initial_state_matches_nest(self):
        r"""State should be initialized to zero (V_m=0, I_syn=0, etc.)."""
        with brainstate.environ.context(dt=0.5 * u.ms):
            neuron = gif_pop_psc_exp(1)
            neuron.init_state()
            self.assertEqual(neuron.V_m, 0.0)
            self.assertEqual(neuron.I_syn_ex, 0.0)
            self.assertEqual(neuron.I_syn_in, 0.0)
            self.assertEqual(neuron.n_spikes, 0)
            self.assertEqual(neuron.n_expect, 0.0)
            self.assertEqual(neuron.y0, 0.0)


class TestGIFPopPscExpParameterValidation(unittest.TestCase):
    r"""Test that invalid parameters raise appropriate errors."""

    def test_mismatched_tau_sfa_q_sfa_raises(self):
        with self.assertRaises(ValueError):
            gif_pop_psc_exp(1, tau_sfa=[10.0], q_sfa=[1.0, 2.0])

    def test_negative_capacitance_raises(self):
        with self.assertRaises(ValueError):
            gif_pop_psc_exp(1, C_m=-250.0)

    def test_negative_tau_m_raises(self):
        with self.assertRaises(ValueError):
            gif_pop_psc_exp(1, tau_m=-20.0)

    def test_negative_tau_syn_raises(self):
        with self.assertRaises(ValueError):
            gif_pop_psc_exp(1, tau_syn_ex=-3.0)

    def test_negative_tau_sfa_raises(self):
        with self.assertRaises(ValueError):
            gif_pop_psc_exp(1, tau_sfa=[-300.0], q_sfa=[0.5])

    def test_negative_N_raises(self):
        with self.assertRaises(ValueError):
            gif_pop_psc_exp(1, N=-10)

    def test_negative_lambda_0_raises(self):
        with self.assertRaises(ValueError):
            gif_pop_psc_exp(1, lambda_0=-1.0)

    def test_negative_Delta_V_raises(self):
        with self.assertRaises(ValueError):
            gif_pop_psc_exp(1, Delta_V=-2.0)

    def test_negative_t_ref_raises(self):
        with self.assertRaises(ValueError):
            gif_pop_psc_exp(1, t_ref=-1.0)


class TestGIFPopPscExpAdaptationKernel(unittest.TestCase):
    r"""Test adaptation kernel computation."""

    def test_adaptation_kernel_values(self):
        r"""Adaptation kernel at lag k should match sum of exponentials."""
        with brainstate.environ.context(dt=0.5 * u.ms):
            neuron = gif_pop_psc_exp(1, tau_sfa=(300.0, 100.0), q_sfa=(0.5, 1.0))
            neuron.init_state()

            h = 0.5
            for k in [1, 5, 10, 50, 100]:
                expected = 0.5 * math.exp(-k * h / 300.0) + 1.0 * math.exp(-k * h / 100.0)
                actual = neuron._adaptation_kernel(k, h)
                self.assertAlmostEqual(actual, expected, places=12)

    def test_history_size_auto_determination(self):
        r"""Auto-determined history size should be reasonable."""
        with brainstate.environ.context(dt=0.5 * u.ms):
            neuron = gif_pop_psc_exp(1)
            neuron.init_state()
            # With defaults (tau_sfa=300, q_sfa=0.5, Delta_V=2, tau_m=20),
            # history should be > 5*tau_m and accommodate t_ref
            self.assertGreater(neuron._len_kernel, int(5 * 20.0 / 0.5))
            self.assertGreater(neuron._len_kernel * 0.5, neuron.t_ref)

    def test_theta_initialization(self):
        r"""Theta buffer should be correctly initialized."""
        with brainstate.environ.context(dt=0.5 * u.ms):
            neuron = gif_pop_psc_exp(1, tau_sfa=(300.0,), q_sfa=(0.5,))
            neuron.init_state()
            h = 0.5
            L = neuron._len_kernel

            for k in range(min(L, 10)):
                expected = _nest_ref_adaptation_kernel(L - k, h, (300.0,), (0.5,))
                self.assertAlmostEqual(neuron._theta[k], expected, places=12)


class TestGIFPopPscExpIntegrationConstants(unittest.TestCase):
    r"""Test pre-computed integration constants."""

    def test_membrane_constants(self):
        r"""P22 and P20 should match exact integration constants."""
        with brainstate.environ.context(dt=0.5 * u.ms):
            neuron = gif_pop_psc_exp(1, tau_m=20.0, C_m=250.0)
            neuron.init_state()
            h = 0.5

            expected_P22 = math.exp(-h / 20.0)
            expected_P20 = 20.0 / 250.0 * (1.0 - expected_P22)

            self.assertAlmostEqual(neuron._P22, expected_P22, places=15)
            self.assertAlmostEqual(neuron._P20, expected_P20, places=15)

    def test_synaptic_constants(self):
        r"""P11_ex and P11_in should match exponential decay constants."""
        with brainstate.environ.context(dt=0.5 * u.ms):
            neuron = gif_pop_psc_exp(1, tau_syn_ex=3.0, tau_syn_in=6.0)
            neuron.init_state()
            h = 0.5

            expected_P11_ex = math.exp(-h / 3.0)
            expected_P11_in = math.exp(-h / 6.0)

            self.assertAlmostEqual(neuron._P11_ex, expected_P11_ex, places=15)
            self.assertAlmostEqual(neuron._P11_in, expected_P11_in, places=15)


class TestGIFPopPscExpSubthresholdDynamics(unittest.TestCase):
    r"""Test subthreshold membrane dynamics without spiking."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.5  # ms
        self.dt_q = 0.5 * u.ms

    def _make_neuron(self, **kwargs):
        defaults = dict(
            N=100, tau_m=20.0, C_m=250.0, t_ref=4.0,
            lambda_0=0.0,  # disable spiking for subthreshold tests
            Delta_V=2.0, E_L=0.0, V_reset=0.0, V_T_star=15.0,
            I_e=0.0, tau_syn_ex=3.0, tau_syn_in=6.0,
            tau_sfa=(300.0,), q_sfa=(0.5,),
        )
        defaults.update(kwargs)
        neuron = gif_pop_psc_exp(1, **defaults)
        return neuron

    def test_dc_current_drives_membrane(self):
        r"""Constant I_e should drive membrane potential toward steady state."""
        with brainstate.environ.context(dt=self.dt_q):
            neuron = self._make_neuron(I_e=500.0)
            neuron.init_state()

            for _ in range(200):
                neuron.update()

            # Steady state: V_m should approach E_L + R * I_e = 0 + (20/250)*500 = 40 mV
            # But adaptation and population dynamics affect this
            self.assertTrue(neuron.V_m > 0.0, f"V_m should be driven positive by I_e, got {neuron.V_m}")

    def test_membrane_decay_without_input(self):
        r"""Without input, V_m should decay to E_L."""
        with brainstate.environ.context(dt=self.dt_q):
            neuron = self._make_neuron(I_e=0.0)
            neuron.init_state()
            # Manually set V_m to a non-zero value
            neuron._V_m = 10.0

            for _ in range(1000):
                neuron.update()

            self.assertAlmostEqual(neuron.V_m, 0.0, places=3,
                                   msg="V_m should decay to E_L=0 without input")


class TestGIFPopPscExpSynapticDynamics(unittest.TestCase):
    r"""Test synaptic current dynamics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.5  # ms
        self.dt_q = 0.5 * u.ms

    def test_excitatory_spike_input(self):
        r"""Positive delta input should increase I_syn_ex."""
        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, lambda_0=0.0)
            neuron.init_state()

            # Inject excitatory spike
            neuron.add_delta_input('ex_spike_0', 100.0)
            neuron.update()

            self.assertTrue(neuron.I_syn_ex > 0.0,
                            f"I_syn_ex should be positive after exc spike, got {neuron.I_syn_ex}")

    def test_inhibitory_spike_input(self):
        r"""Negative delta input should affect I_syn_in."""
        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, lambda_0=0.0)
            neuron.init_state()

            # Inject inhibitory spike
            neuron.add_delta_input('in_spike_0', -100.0)
            neuron.update()

            self.assertTrue(neuron.I_syn_in < 0.0,
                            f"I_syn_in should be negative after inh spike, got {neuron.I_syn_in}")

    def test_synaptic_current_decay(self):
        r"""Synaptic currents should decay after spike input.

        In the population model, I_syn is integrated via exact propagators
        coupled with the membrane dynamics. The effective decay follows
        the synaptic time constant but is coupled with the membrane update.
        We verify that the synaptic current decays monotonically and that
        consecutive steps decay by P11 = exp(-h/tau_syn).
        """
        h = self.dt
        tau_syn_ex = 3.0
        P11_ex = math.exp(-h / tau_syn_ex)

        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, lambda_0=0.0, tau_syn_ex=tau_syn_ex, tau_syn_in=6.0)
            neuron.init_state()

            # Inject excitatory spike
            neuron.add_delta_input('ex_spike_0', 100.0)
            neuron.update()
            isyn_ex_0 = neuron.I_syn_ex

            # After the initial step, without further input, consecutive
            # JNy_ex updates are: JNy_ex = JNA_ex + (JNy_ex - JNA_ex) * P11
            # With JNA_ex=0, this gives JNy_ex *= P11, so I_syn_ex *= P11
            neuron.update()
            isyn_ex_1 = neuron.I_syn_ex

            # The ratio of consecutive steps (with no input) should be P11_ex
            ratio = isyn_ex_1 / isyn_ex_0 if isyn_ex_0 != 0 else 0
            self.assertAlmostEqual(ratio, P11_ex, places=10)


class TestGIFPopPscExpPopulationDynamics(unittest.TestCase):
    r"""Test population-level dynamics and spike generation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.5  # ms
        self.dt_q = 0.5 * u.ms

    def test_population_generates_spikes_with_drive(self):
        r"""With I_e drive, population should generate spikes."""
        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(
                1, N=100, I_e=500.0, lambda_0=10.0,
                V_T_star=10.0, Delta_V=2.0, rng_seed=42,
            )
            neuron.init_state()

            total_spikes = 0
            for _ in range(200):
                n_spk = neuron.update()
                total_spikes += n_spk

            self.assertGreater(total_spikes, 0,
                               "Population should generate spikes with I_e drive")

    def test_no_spikes_without_drive(self):
        r"""Without drive, population should generate very few or no spikes."""
        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(
                1, N=100, I_e=0.0, lambda_0=10.0,
                V_T_star=100.0,  # very high threshold
                Delta_V=2.0, rng_seed=42,
            )
            neuron.init_state()

            total_spikes = 0
            for _ in range(100):
                n_spk = neuron.update()
                total_spikes += n_spk

            # With V_m near 0 and V_T_star=100, effective threshold is 100 mV
            # escape rate ~ lambda_0 * exp((0 - 100) / 2) ≈ 0
            self.assertEqual(total_spikes, 0,
                             "No spikes expected with very high threshold")

    def test_binomial_vs_poisson_modes(self):
        r"""Both BinoRand modes should produce spike counts in [0, N]."""
        for bino in [True, False]:
            with brainstate.environ.context(dt=self.dt_q):
                neuron = gif_pop_psc_exp(
                    1, N=50, I_e=500.0, lambda_0=10.0,
                    V_T_star=10.0, Delta_V=2.0,
                    BinoRand=bino, rng_seed=42,
                )
                neuron.init_state()

                for _ in range(100):
                    n_spk = neuron.update()
                    self.assertGreaterEqual(n_spk, 0)
                    self.assertLessEqual(n_spk, 50)


class TestGIFPopPscExpHistoryBuffers(unittest.TestCase):
    r"""Test history buffer initialization and rotation."""

    def test_initial_buffers(self):
        r"""Initial buffers: n[L-1]=N, m[L-1]=N, rest zeros."""
        with brainstate.environ.context(dt=0.5 * u.ms):
            neuron = gif_pop_psc_exp(1, N=100)
            neuron.init_state()
            L = neuron._len_kernel

            for k in range(L):
                if k == L - 1:
                    self.assertEqual(neuron._n[k], 100.0)
                    self.assertEqual(neuron._m[k], 100.0)
                else:
                    self.assertEqual(neuron._n[k], 0.0)
                    self.assertEqual(neuron._m[k], 0.0)
                self.assertEqual(neuron._v_buf[k], 0.0)
                self.assertEqual(neuron._u[k], 0.0)
                self.assertEqual(neuron._lambda_buf[k], 0.0)

    def test_rotating_index_wraps(self):
        r"""Rotating index k0 should wrap around correctly."""
        with brainstate.environ.context(dt=0.5 * u.ms):
            neuron = gif_pop_psc_exp(1, N=10, lambda_0=0.0)
            neuron.init_state()
            L = neuron._len_kernel

            # Run for 2*L steps, check k0 wraps
            for _ in range(2 * L):
                neuron.update()

            self.assertEqual(neuron._k0, 0)  # Should wrap back to 0


class TestGIFPopPscExpReferenceTrace(unittest.TestCase):
    r"""Compare full simulation traces against standalone reference implementation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.5  # ms
        self.dt_q = 0.5 * u.ms

    def test_trace_no_input(self):
        r"""Trace with no external input should match reference exactly."""
        params = dict(
            N=100, tau_m=20.0, C_m=250.0, t_ref=4.0,
            lambda_0=10.0, Delta_V=2.0, E_L=0.0, V_reset=0.0,
            V_T_star=15.0, I_e=0.0, tau_syn_ex=3.0, tau_syn_in=6.0,
            tau_sfa=(300.0,), q_sfa=(0.5,), BinoRand=True, rng_seed=42,
        )
        n_steps = 50

        # Reference
        ref = run_nest_ref(n_steps, self.dt, **params)

        # Model
        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, **params)
            neuron.init_state()

            for k in range(n_steps):
                neuron.update()

                self.assertAlmostEqual(
                    neuron.V_m, ref['V_m'][k], places=12,
                    msg=f"V_m mismatch at step {k}"
                )
                self.assertAlmostEqual(
                    neuron.I_syn_ex, ref['I_syn_ex'][k], places=12,
                    msg=f"I_syn_ex mismatch at step {k}"
                )
                self.assertAlmostEqual(
                    neuron.I_syn_in, ref['I_syn_in'][k], places=12,
                    msg=f"I_syn_in mismatch at step {k}"
                )
                self.assertEqual(
                    neuron.n_spikes, ref['n_spikes'][k],
                    msg=f"n_spikes mismatch at step {k}"
                )
                self.assertAlmostEqual(
                    neuron.n_expect, ref['n_expect'][k], places=10,
                    msg=f"n_expect mismatch at step {k}"
                )
                self.assertAlmostEqual(
                    neuron.theta_hat, ref['theta_hat'][k], places=12,
                    msg=f"theta_hat mismatch at step {k}"
                )

    def test_trace_with_dc_current(self):
        r"""Trace with constant I_e should match reference."""
        params = dict(
            N=100, tau_m=20.0, C_m=250.0, t_ref=4.0,
            lambda_0=10.0, Delta_V=2.0, E_L=0.0, V_reset=0.0,
            V_T_star=10.0, I_e=500.0, tau_syn_ex=3.0, tau_syn_in=6.0,
            tau_sfa=(300.0,), q_sfa=(0.5,), BinoRand=True, rng_seed=42,
        )
        n_steps = 100

        ref = run_nest_ref(n_steps, self.dt, **params)

        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, **params)
            neuron.init_state()

            for k in range(n_steps):
                neuron.update()

                self.assertAlmostEqual(
                    neuron.V_m, ref['V_m'][k], places=10,
                    msg=f"V_m mismatch at step {k}"
                )
                self.assertEqual(
                    neuron.n_spikes, ref['n_spikes'][k],
                    msg=f"n_spikes mismatch at step {k}"
                )

    def test_trace_with_excitatory_spikes(self):
        r"""Trace with excitatory spike inputs should match reference."""
        params = dict(
            N=50, tau_m=20.0, C_m=250.0, t_ref=4.0,
            lambda_0=10.0, Delta_V=2.0, E_L=0.0, V_reset=0.0,
            V_T_star=15.0, I_e=0.0, tau_syn_ex=3.0, tau_syn_in=6.0,
            tau_sfa=(300.0,), q_sfa=(0.5,), BinoRand=True, rng_seed=42,
        )
        n_steps = 50

        # Excitatory spike inputs at specific steps
        ex_spikes = np.zeros(n_steps)
        ex_spikes[5] = 100.0
        ex_spikes[10] = 200.0
        ex_spikes[20] = 50.0

        ref = run_nest_ref(
            n_steps, self.dt, **params,
            ex_spikes=ex_spikes,
        )

        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, **params)
            neuron.init_state()

            for k in range(n_steps):
                if ex_spikes[k] != 0.0:
                    neuron.add_delta_input(f'ex_{k}', ex_spikes[k])
                neuron.update()

                self.assertAlmostEqual(
                    neuron.V_m, ref['V_m'][k], places=10,
                    msg=f"V_m mismatch at step {k}"
                )
                self.assertAlmostEqual(
                    neuron.I_syn_ex, ref['I_syn_ex'][k], places=10,
                    msg=f"I_syn_ex mismatch at step {k}"
                )
                self.assertEqual(
                    neuron.n_spikes, ref['n_spikes'][k],
                    msg=f"n_spikes mismatch at step {k}"
                )

    def test_trace_with_inhibitory_spikes(self):
        r"""Trace with inhibitory spike inputs should match reference."""
        params = dict(
            N=50, tau_m=20.0, C_m=250.0, t_ref=4.0,
            lambda_0=10.0, Delta_V=2.0, E_L=0.0, V_reset=0.0,
            V_T_star=15.0, I_e=200.0, tau_syn_ex=3.0, tau_syn_in=6.0,
            tau_sfa=(300.0,), q_sfa=(0.5,), BinoRand=True, rng_seed=42,
        )
        n_steps = 50

        # Inhibitory spike inputs at specific steps
        in_spikes = np.zeros(n_steps)
        in_spikes[3] = -150.0
        in_spikes[8] = -200.0
        in_spikes[15] = -100.0

        ref = run_nest_ref(
            n_steps, self.dt, **params,
            in_spikes=in_spikes,
        )

        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, **params)
            neuron.init_state()

            for k in range(n_steps):
                if in_spikes[k] != 0.0:
                    neuron.add_delta_input(f'in_{k}', in_spikes[k])
                neuron.update()

                self.assertAlmostEqual(
                    neuron.V_m, ref['V_m'][k], places=10,
                    msg=f"V_m mismatch at step {k}"
                )
                self.assertAlmostEqual(
                    neuron.I_syn_in, ref['I_syn_in'][k], places=10,
                    msg=f"I_syn_in mismatch at step {k}"
                )

    def test_trace_with_current_input(self):
        r"""Trace with external current inputs (one-step delay) should match reference."""
        params = dict(
            N=50, tau_m=20.0, C_m=250.0, t_ref=4.0,
            lambda_0=10.0, Delta_V=2.0, E_L=0.0, V_reset=0.0,
            V_T_star=15.0, I_e=0.0, tau_syn_ex=3.0, tau_syn_in=6.0,
            tau_sfa=(300.0,), q_sfa=(0.5,), BinoRand=True, rng_seed=42,
        )
        n_steps = 50

        # Current inputs
        currents = np.zeros(n_steps)
        currents[0] = 300.0
        currents[10] = -100.0
        currents[20] = 500.0

        ref = run_nest_ref(
            n_steps, self.dt, **params,
            currents=currents,
        )

        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, **params)
            neuron.init_state()

            for k in range(n_steps):
                neuron.update(x=currents[k])

                self.assertAlmostEqual(
                    neuron.V_m, ref['V_m'][k], places=10,
                    msg=f"V_m mismatch at step {k}"
                )
                self.assertEqual(
                    neuron.n_spikes, ref['n_spikes'][k],
                    msg=f"n_spikes mismatch at step {k}"
                )

    def test_trace_poisson_mode(self):
        r"""Trace with Poisson spike generation should match reference."""
        params = dict(
            N=100, tau_m=20.0, C_m=250.0, t_ref=4.0,
            lambda_0=10.0, Delta_V=2.0, E_L=0.0, V_reset=0.0,
            V_T_star=10.0, I_e=500.0, tau_syn_ex=3.0, tau_syn_in=6.0,
            tau_sfa=(300.0,), q_sfa=(0.5,), BinoRand=False, rng_seed=42,
        )
        n_steps = 80

        ref = run_nest_ref(n_steps, self.dt, **params)

        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, **params)
            neuron.init_state()

            for k in range(n_steps):
                neuron.update()

                self.assertAlmostEqual(
                    neuron.V_m, ref['V_m'][k], places=10,
                    msg=f"V_m mismatch at step {k}"
                )
                self.assertEqual(
                    neuron.n_spikes, ref['n_spikes'][k],
                    msg=f"n_spikes mismatch at step {k}"
                )

    def test_trace_multi_adaptation(self):
        r"""Trace with multiple adaptation timescales should match reference."""
        params = dict(
            N=100, tau_m=20.0, C_m=250.0, t_ref=4.0,
            lambda_0=10.0, Delta_V=2.0, E_L=0.0, V_reset=0.0,
            V_T_star=10.0, I_e=500.0, tau_syn_ex=3.0, tau_syn_in=6.0,
            tau_sfa=(100.0, 500.0, 1000.0), q_sfa=(1.0, 0.5, 0.2),
            BinoRand=True, rng_seed=42,
        )
        n_steps = 100

        ref = run_nest_ref(n_steps, self.dt, **params)

        with brainstate.environ.context(dt=self.dt_q):
            neuron = gif_pop_psc_exp(1, **params)
            neuron.init_state()

            for k in range(n_steps):
                neuron.update()

                self.assertAlmostEqual(
                    neuron.V_m, ref['V_m'][k], places=10,
                    msg=f"V_m mismatch at step {k}"
                )
                self.assertAlmostEqual(
                    neuron.theta_hat, ref['theta_hat'][k], places=10,
                    msg=f"theta_hat mismatch at step {k}"
                )
                self.assertEqual(
                    neuron.n_spikes, ref['n_spikes'][k],
                    msg=f"n_spikes mismatch at step {k}"
                )


class TestGIFPopPscExpLongSimulation(unittest.TestCase):
    r"""Test long-running simulation for statistical properties."""

    def test_steady_state_rate_with_inhibitory_feedback(self):
        r"""Match the NEST test: inhibitory population with self-connection.

        This test mirrors test_gif_pop_psc_exp.py in NEST:
        - Population size: 500
        - Self-inhibition with weight -6.25
        - Expected steady-state rate ~22 Hz +/- 1 Hz
        - Expected rate variance ~102 Hz^2 +/- 6 Hz^2
        """
        res = 0.5  # ms
        T = 10000.0  # ms
        start_time = 1000.0  # ms
        pop_size = 500

        n_steps = int(T / res)
        start_step = int(start_time / res)

        with brainstate.environ.context(dt=res * u.ms):
            neuron = gif_pop_psc_exp(
                1,
                N=pop_size,
                V_reset=0.0,
                V_T_star=10.0,
                E_L=0.0,
                Delta_V=2.0,
                C_m=250.0,
                tau_m=20.0,
                t_ref=4.0,
                I_e=500.0,
                lambda_0=10.0,
                tau_syn_in=2.0,
                tau_sfa=(500.0,),
                q_sfa=(1.0,),
                rng_seed=42,
            )
            neuron.init_state()

            nspike_trace = []
            for step in range(n_steps):
                # Self-inhibition: feed back the previous spike count with
                # weight=-6.25 and delay=1 step. For simplicity, we feed
                # the spike count from the previous step as inhibitory input.
                if step > 0:
                    inhibitory_input = -6.25 * prev_n_spikes
                    if inhibitory_input != 0.0:
                        neuron.add_delta_input(f'inh_{step}', inhibitory_input)

                n_spk = neuron.update()
                prev_n_spikes = n_spk
                nspike_trace.append(n_spk)

        dftype = brainstate.environ.dftype()
        nspike_array = np.array(nspike_trace[start_step:], dtype=dftype)

        mean_nspike = np.mean(nspike_array)
        mean_rate = mean_nspike / pop_size / res * 1000.0  # Hz

        var_nspike = np.var(nspike_array)
        var_nspike_scaled = var_nspike / pop_size / res * 1000.0
        var_rate = var_nspike_scaled / pop_size / res * 1000.0  # Hz^2

        # These tolerances are wider than NEST's test because different RNG
        # implementations produce different sequences
        self.assertAlmostEqual(mean_rate, 22.0, delta=3.0,
                               msg=f"Mean rate {mean_rate:.1f} Hz not near expected 22 Hz")
        # Variance check with wider tolerance
        self.assertTrue(50.0 < var_rate < 200.0,
                        f"Rate variance {var_rate:.1f} Hz^2 out of expected range")


class TestGIFPopPscExpResetState(unittest.TestCase):
    r"""Test that reset_state properly reinitializes."""

    def test_reset_returns_to_initial_conditions(self):
        r"""After reset, all state should match fresh init."""
        with brainstate.environ.context(dt=0.5 * u.ms):
            neuron = gif_pop_psc_exp(1, N=100, I_e=500.0, rng_seed=0)
            neuron.init_state()

            # Run for a while
            for _ in range(100):
                neuron.update()

            # Reset
            neuron.reset_state()

            # Check initial conditions
            self.assertEqual(neuron.V_m, 0.0)
            self.assertEqual(neuron.I_syn_ex, 0.0)
            self.assertEqual(neuron.I_syn_in, 0.0)
            self.assertEqual(neuron.n_spikes, 0)
            self.assertEqual(neuron._k0, 0)
            self.assertEqual(neuron._x, 0.0)
            self.assertEqual(neuron._z, 0.0)


class TestGIFPopPscExpDeterminism(unittest.TestCase):
    r"""Test that same seed produces identical results."""

    def test_same_seed_same_output(self):
        r"""Two runs with same seed should produce identical spike trains."""
        params = dict(
            N=100, I_e=500.0, V_T_star=10.0,
            lambda_0=10.0, Delta_V=2.0, rng_seed=123,
        )
        n_steps = 100

        spikes1 = []
        spikes2 = []

        for run_spikes in [spikes1, spikes2]:
            with brainstate.environ.context(dt=0.5 * u.ms):
                neuron = gif_pop_psc_exp(1, **params)
                neuron.init_state()
                for _ in range(n_steps):
                    n_spk = neuron.update()
                    run_spikes.append(n_spk)

        self.assertEqual(spikes1, spikes2)

    def test_different_seed_different_output(self):
        r"""Two runs with different seeds should (likely) differ."""
        n_steps = 100

        def run_with_seed(seed):
            spikes = []
            with brainstate.environ.context(dt=0.5 * u.ms):
                neuron = gif_pop_psc_exp(
                    1, N=100, I_e=500.0, V_T_star=10.0,
                    lambda_0=10.0, Delta_V=2.0, rng_seed=seed,
                )
                neuron.init_state()
                for _ in range(n_steps):
                    spikes.append(neuron.update())
            return spikes

        s1 = run_with_seed(0)
        s2 = run_with_seed(999)

        # Very unlikely to be identical with different seeds
        self.assertNotEqual(s1, s2)


if __name__ == '__main__':
    unittest.main()
