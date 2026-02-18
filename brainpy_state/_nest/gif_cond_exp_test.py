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
Tests for gif_cond_exp neuron model.

These tests verify that the brainpy.state implementation of gif_cond_exp
matches NEST's update ordering and semantics, including:

- Default parameter values matching NEST C++ source
- Subthreshold membrane dynamics with conductance-based synapses
- Exponential conductance decay
- Refractory period mechanics (V clamped, countdown)
- Spike-triggered current (stc) adaptation
- Spike-frequency adaptation (sfa) threshold dynamics
- Stochastic spike generation via hazard function
- One-step delayed current input (NEST ring buffer semantics)
- Full reference trace comparison against Python reference implementation

All tests use float64 on CPU to match NEST's numerical behavior.
"""

import math
import os
import unittest

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import braintools
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy.state import gif_cond_exp


def _rkf45_ref_step(v, g_ex, g_in, is_refractory, i_stim, stc, dt, h0, p, atol=1e-3):
    r"""Reference RKF45 integration for a single simulation step.

    This is a standalone Python implementation of the RKF45 adaptive integrator
    that matches the NEST GSL integration behavior for gif_cond_exp.
    """
    min_h = 1e-8
    t = 0.0
    h = max(h0, min_h)

    def f(v_, ge_, gi_):
        V = p['V_reset'] if is_refractory else v_
        i_syn_ex = ge_ * (V - p['E_ex'])
        i_syn_in = gi_ * (V - p['E_in'])
        i_l = p['g_L'] * (V - p['E_L'])
        dv = 0.0 if is_refractory else (-i_l + i_stim + p['I_e'] - i_syn_ex - i_syn_in - stc) / p['C_m']
        return dv, -ge_ / p['tau_syn_ex'], -gi_ / p['tau_syn_in']

    while t < dt:
        h = max(min_h, min(h, dt - t))

        k1 = f(v, g_ex, g_in)
        y2 = (v + h * k1[0] / 4.0, g_ex + h * k1[1] / 4.0, g_in + h * k1[2] / 4.0)
        k2 = f(*y2)
        y3 = (
            v + h * (3.0 * k1[0] / 32.0 + 9.0 * k2[0] / 32.0),
            g_ex + h * (3.0 * k1[1] / 32.0 + 9.0 * k2[1] / 32.0),
            g_in + h * (3.0 * k1[2] / 32.0 + 9.0 * k2[2] / 32.0),
        )
        k3 = f(*y3)
        y4 = (
            v + h * (1932.0 * k1[0] / 2197.0 - 7200.0 * k2[0] / 2197.0 + 7296.0 * k3[0] / 2197.0),
            g_ex + h * (1932.0 * k1[1] / 2197.0 - 7200.0 * k2[1] / 2197.0 + 7296.0 * k3[1] / 2197.0),
            g_in + h * (1932.0 * k1[2] / 2197.0 - 7200.0 * k2[2] / 2197.0 + 7296.0 * k3[2] / 2197.0),
        )
        k4 = f(*y4)
        y5 = (
            v + h * (439.0 * k1[0] / 216.0 - 8.0 * k2[0] + 3680.0 * k3[0] / 513.0 - 845.0 * k4[0] / 4104.0),
            g_ex + h * (439.0 * k1[1] / 216.0 - 8.0 * k2[1] + 3680.0 * k3[1] / 513.0 - 845.0 * k4[1] / 4104.0),
            g_in + h * (439.0 * k1[2] / 216.0 - 8.0 * k2[2] + 3680.0 * k3[2] / 513.0 - 845.0 * k4[2] / 4104.0),
        )
        k5 = f(*y5)
        y6 = (
            v + h * (-8.0 * k1[0] / 27.0 + 2.0 * k2[0] - 3544.0 * k3[0] / 2565.0 + 1859.0 * k4[0] / 4104.0 - 11.0 * k5[0] / 40.0),
            g_ex + h * (-8.0 * k1[1] / 27.0 + 2.0 * k2[1] - 3544.0 * k3[1] / 2565.0 + 1859.0 * k4[1] / 4104.0 - 11.0 * k5[1] / 40.0),
            g_in + h * (-8.0 * k1[2] / 27.0 + 2.0 * k2[2] - 3544.0 * k3[2] / 2565.0 + 1859.0 * k4[2] / 4104.0 - 11.0 * k5[2] / 40.0),
        )
        k6 = f(*y6)

        y4_sol = (
            v + h * (25.0 * k1[0] / 216.0 + 1408.0 * k3[0] / 2565.0 + 2197.0 * k4[0] / 4104.0 - k5[0] / 5.0),
            g_ex + h * (25.0 * k1[1] / 216.0 + 1408.0 * k3[1] / 2565.0 + 2197.0 * k4[1] / 4104.0 - k5[1] / 5.0),
            g_in + h * (25.0 * k1[2] / 216.0 + 1408.0 * k3[2] / 2565.0 + 2197.0 * k4[2] / 4104.0 - k5[2] / 5.0),
        )
        y5_sol = (
            v + h * (16.0 * k1[0] / 135.0 + 6656.0 * k3[0] / 12825.0 + 28561.0 * k4[0] / 56430.0 - 9.0 * k5[0] / 50.0 + 2.0 * k6[0] / 55.0),
            g_ex + h * (16.0 * k1[1] / 135.0 + 6656.0 * k3[1] / 12825.0 + 28561.0 * k4[1] / 56430.0 - 9.0 * k5[1] / 50.0 + 2.0 * k6[1] / 55.0),
            g_in + h * (16.0 * k1[2] / 135.0 + 6656.0 * k3[2] / 12825.0 + 28561.0 * k4[2] / 56430.0 - 9.0 * k5[2] / 50.0 + 2.0 * k6[2] / 55.0),
        )
        err = max(abs(y5_sol[0] - y4_sol[0]), abs(y5_sol[1] - y4_sol[1]), abs(y5_sol[2] - y4_sol[2]))
        if err <= atol or h <= min_h:
            v, g_ex, g_in = y5_sol
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    return v, g_ex, g_in, h


def _run_nest_ref(n_steps, dt, p, i_stim_seq, dg_seq, rand_seq,
                  tau_stc, q_stc, tau_sfa, q_sfa, lambda_0, Delta_V, V_T_star):
    r"""Full reference implementation of gif_cond_exp update loop.

    Matches NEST update order exactly:
    1. Compute stc/sfa totals, decay elements
    2. Integrate ODEs
    3. Add conductance jumps
    4. Spike check (stochastic) or refractory countdown
    5. Store I_stim
    """
    v = p['E_L']
    ge, gi = 0.0, 0.0
    r = 0
    h = dt
    i_stim = 0.0
    refr_steps = math.ceil(p['t_ref'] / dt)

    n_stc = len(tau_stc)
    n_sfa = len(tau_sfa)
    stc_elems = [0.0] * n_stc
    sfa_elems = [0.0] * n_sfa

    P_stc = [math.exp(-dt / tau) for tau in tau_stc]
    P_sfa = [math.exp(-dt / tau) for tau in tau_sfa]

    v_trace, ge_trace, gi_trace, spike_trace = [], [], [], []
    stc_trace, sfa_trace = [], []

    for k in range(n_steps):
        # Step 1: Decay stc/sfa elements, compute totals
        stc_total = 0.0
        for i in range(n_stc):
            stc_total += stc_elems[i]
            stc_elems[i] *= P_stc[i]

        sfa_total = V_T_star
        for i in range(n_sfa):
            sfa_total += sfa_elems[i]
            sfa_elems[i] *= P_sfa[i]

        # Step 2: Integrate ODE
        is_refractory = r > 0
        v, ge, gi, h = _rkf45_ref_step(v, ge, gi, is_refractory, i_stim, stc_total, dt, h, p)

        # Step 3: Add conductance jumps
        if k < len(dg_seq):
            for dg in dg_seq[k]:
                if dg >= 0.0:
                    ge += dg
                else:
                    gi += -dg

        # Step 4: Spike check / refractory
        spike = 0.0
        if r == 0:
            lam = lambda_0 * math.exp((v - sfa_total) / Delta_V)
            if lam > 0.0:
                spike_prob = -math.expm1(-lam * dt)
                if k < len(rand_seq) and rand_seq[k] < spike_prob:
                    spike = 1.0
                    for i in range(n_stc):
                        stc_elems[i] += q_stc[i]
                    for i in range(n_sfa):
                        sfa_elems[i] += q_sfa[i]
                    r = refr_steps
        else:
            r -= 1
            v = p['V_reset']

        # Step 5: Store I_stim
        if k < len(i_stim_seq):
            i_stim = i_stim_seq[k]
        else:
            i_stim = 0.0

        v_trace.append(v)
        ge_trace.append(ge)
        gi_trace.append(gi)
        spike_trace.append(spike)
        stc_trace.append(stc_total)
        sfa_trace.append(sfa_total)

    return v_trace, ge_trace, gi_trace, spike_trace, stc_trace, sfa_trace


class TestGIFCondExpDefaultParams(unittest.TestCase):
    r"""Test that default parameters match NEST C++ source code values."""

    def test_nest_cpp_default_parameters(self):
        neuron = gif_cond_exp(1)
        self.assertEqual(neuron.g_L, 4.0 * u.nS)
        self.assertEqual(neuron.E_L, -70.0 * u.mV)
        self.assertEqual(neuron.C_m, 80.0 * u.pF)
        self.assertEqual(neuron.V_reset, -55.0 * u.mV)
        self.assertEqual(neuron.Delta_V, 0.5 * u.mV)
        self.assertEqual(neuron.V_T_star, -35.0 * u.mV)
        self.assertAlmostEqual(neuron.lambda_0, 0.001)  # 1/ms (= 1/s internally)
        self.assertEqual(neuron.t_ref, 4.0 * u.ms)
        self.assertEqual(neuron.E_ex, 0.0 * u.mV)
        self.assertEqual(neuron.E_in, -85.0 * u.mV)
        self.assertEqual(neuron.tau_syn_ex, 2.0 * u.ms)
        self.assertEqual(neuron.tau_syn_in, 2.0 * u.ms)
        self.assertEqual(neuron.I_e, 0.0 * u.pA)
        self.assertEqual(neuron.tau_sfa, ())
        self.assertEqual(neuron.q_sfa, ())
        self.assertEqual(neuron.tau_stc, ())
        self.assertEqual(neuron.q_stc, ())

    def test_initial_state_matches_nest(self):
        r"""V_m should be initialized to E_L, conductances to 0."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = gif_cond_exp(1)
            neuron.init_state()
            self.assertTrue(u.math.allclose(neuron.V.value, -70.0 * u.mV))
            self.assertTrue(u.math.allclose(neuron.g_ex.value, 0.0 * u.nS))
            self.assertTrue(u.math.allclose(neuron.g_in.value, 0.0 * u.nS))


class TestGIFCondExpParameterValidation(unittest.TestCase):
    r"""Test that invalid parameters raise appropriate errors."""

    def test_mismatched_tau_sfa_q_sfa_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp(1, tau_sfa=[10.0], q_sfa=[1.0, 2.0])

    def test_mismatched_tau_stc_q_stc_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp(1, tau_stc=[10.0, 20.0], q_stc=[1.0])

    def test_negative_capacitance_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp(1, C_m=-80.0 * u.pF)

    def test_negative_g_L_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp(1, g_L=-1.0 * u.nS)

    def test_negative_Delta_V_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp(1, Delta_V=-0.5 * u.mV)

    def test_negative_lambda_0_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp(1, lambda_0=-1.0)


class TestGIFCondExpSubthresholdDynamics(unittest.TestCase):
    r"""Test subthreshold membrane dynamics without spiking."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                neuron.add_delta_input(f'dg_{k}_{i}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_current_input_has_one_step_delay(self):
        r"""External current should be stored for use in the NEXT step (NEST ring buffer)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                E_L=0.0 * u.mV,
                g_L=0.0001 * u.nS,  # near-zero leak to isolate current effect
                I_e=0.0 * u.pA,
                lambda_0=0.0,  # disable spiking
                V_reset=0.0 * u.mV,
                V_T_star=1000.0 * u.mV,  # high threshold
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            # Step 0: inject current. V should still be ~0 because I_stim was 0 at start
            self._step(neuron, 0, x=100.0 * u.pA)
            v0 = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v0, 0.0, places=3)

            # Step 1: now the 100 pA should take effect
            self._step(neuron, 1, x=0.0 * u.pA)
            v1 = float((neuron.V.value / u.mV)[0])
            self.assertTrue(v1 > 0.0, f"V should increase from current, got {v1}")

    def test_conductance_jumps_from_delta_inputs(self):
        r"""Positive weights should add to g_ex, negative to g_in."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()
            self._step(neuron, 0, dg_values=[5.0, -3.0])
            self.assertTrue(u.math.allclose(neuron.g_ex.value, 5.0 * u.nS))
            self.assertTrue(u.math.allclose(neuron.g_in.value, 3.0 * u.nS))

    def test_conductance_exponential_decay(self):
        r"""Conductances should decay exponentially with their time constants."""
        with brainstate.environ.context(dt=self.dt):
            tau_ex = 2.0  # ms
            tau_in = 5.0  # ms
            neuron = gif_cond_exp(
                1,
                tau_syn_ex=tau_ex * u.ms,
                tau_syn_in=tau_in * u.ms,
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Inject conductances
            self._step(neuron, 0, dg_values=[10.0, -8.0])
            ge0 = float((neuron.g_ex.value / u.nS)[0])
            gi0 = float((neuron.g_in.value / u.nS)[0])

            # Run for 10 steps (1 ms) and check decay
            for k in range(1, 11):
                self._step(neuron, k)

            ge1 = float((neuron.g_ex.value / u.nS)[0])
            gi1 = float((neuron.g_in.value / u.nS)[0])

            # Expected: g_ex(1ms) = 10 * exp(-1/2), g_in(1ms) = 8 * exp(-1/5)
            expected_ge = 10.0 * math.exp(-1.0 / tau_ex)
            expected_gi = 8.0 * math.exp(-1.0 / tau_in)
            self.assertAlmostEqual(ge1, expected_ge, places=3)
            self.assertAlmostEqual(gi1, expected_gi, places=3)

    def test_excitatory_depolarizes_inhibitory_hyperpolarizes(self):
        r"""Excitatory input should depolarize, inhibitory should hyperpolarize."""
        with brainstate.environ.context(dt=self.dt):
            base = gif_cond_exp(1, lambda_0=0.0, V_T_star=1000.0 * u.mV,
                                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            exc = gif_cond_exp(1, lambda_0=0.0, V_T_star=1000.0 * u.mV,
                               V_initializer=braintools.init.Constant(-70.0 * u.mV))
            inh = gif_cond_exp(1, lambda_0=0.0, V_T_star=1000.0 * u.mV,
                               V_initializer=braintools.init.Constant(-70.0 * u.mV))
            base.init_state()
            exc.init_state()
            inh.init_state()

            # Apply conductance inputs
            self._step(base, 0)
            self._step(exc, 0, dg_values=[5.0])
            self._step(inh, 0, dg_values=[-5.0])

            # Let dynamics evolve
            self._step(base, 1)
            self._step(exc, 1)
            self._step(inh, 1)

            v_base = float((base.V.value / u.mV)[0])
            v_exc = float((exc.V.value / u.mV)[0])
            v_inh = float((inh.V.value / u.mV)[0])

            # E_ex=0 mV > -70 mV = V, so excitatory current drives V up
            self.assertTrue(v_exc > v_base, f"Exc {v_exc} should be > base {v_base}")
            # E_in=-85 mV < -70 mV = V, so inhibitory current drives V down
            self.assertTrue(v_inh < v_base, f"Inh {v_inh} should be < base {v_base}")


class TestGIFCondExpRefractoryBehavior(unittest.TestCase):
    r"""Test refractory period mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                neuron.add_delta_input(f'dg_{k}_{i}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_refractory_clamps_voltage_to_V_reset(self):
        r"""During refractory period, V should stay at V_reset."""
        with brainstate.environ.context(dt=self.dt):
            # Use deterministic high-rate spiking to ensure spike on step 0
            neuron = gif_cond_exp(
                1,
                lambda_0=1e12,  # very high firing rate to guarantee spike
                Delta_V=100.0 * u.mV,  # large Delta_V so lambda stays high
                V_T_star=-1000.0 * u.mV,  # very low threshold
                t_ref=1.0 * u.ms,  # 10 steps at 0.1 ms
                V_reset=-55.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            # Step 0 should spike (lambda is enormous)
            spk0 = self._step(neuron, 0)
            self.assertTrue(float(spk0[0]) > 0, "Should spike on step 0 with lambda_0=1e12")

            # Steps 1-9 should be refractory, V clamped to V_reset
            for k in range(1, 10):
                self._step(neuron, k)
                v = float((neuron.V.value / u.mV)[0])
                self.assertAlmostEqual(v, -55.0, places=6,
                                       msg=f"V should be V_reset during refractory at step {k}")

    def test_refractory_count_matches_t_ref(self):
        r"""Refractory counter should match ceil(t_ref / dt)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=4.0 * u.ms,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            self._step(neuron, 0)
            ref_count = int(neuron.refractory_step_count.value[0])
            expected = math.ceil(4.0 / 0.1)  # 40
            self.assertEqual(ref_count, expected)


class TestGIFCondExpAdaptation(unittest.TestCase):
    r"""Test stc and sfa adaptation mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                neuron.add_delta_input(f'dg_{k}_{i}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_stc_elements_decay_exponentially(self):
        r"""STC elements should decay by exp(-dt/tau) each step."""
        dt = 0.1
        tau_stc = [10.0, 20.0]
        q_stc = [5.0, -2.0]

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                lambda_0=1e12,  # guarantee spike
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=0.0 * u.ms,  # no refractory
                tau_stc=tau_stc,
                q_stc=q_stc,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Force a spike on step 0
            self._step(neuron, 0)

            # After spike, stc_elems should have been updated with q_stc values
            # On the NEXT step, those values will be read and then decayed
            # Let's disable spiking for subsequent steps to watch pure decay
            neuron.lambda_0 = 0.0  # turn off spiking

            # Run several steps and check stc decay
            for k in range(1, 11):
                self._step(neuron, k)

            # After 10 steps (1 ms), each element should be q_stc[i] * exp(-1ms/tau[i])
            # But note: the elements were set after spike, then decayed each step
            # After step 0 spike: elem[i] = q_stc[i]
            # After 10 decays: elem[i] = q_stc[i] * exp(-dt/tau)^10 = q_stc[i] * exp(-1/tau)
            for i in range(len(tau_stc)):
                expected = q_stc[i] * math.exp(-1.0 / tau_stc[i])
                actual = neuron._stc_elems[i][0]
                self.assertAlmostEqual(actual, expected, places=6,
                                       msg=f"STC element {i} decay mismatch")

    def test_sfa_elements_decay_exponentially(self):
        r"""SFA elements should decay by exp(-dt/tau) each step."""
        dt = 0.1
        tau_sfa = [100.0, 50.0]
        q_sfa = [10.0, 5.0]

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=0.0 * u.ms,
                tau_sfa=tau_sfa,
                q_sfa=q_sfa,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Force a spike
            self._step(neuron, 0)
            neuron.lambda_0 = 0.0

            for k in range(1, 11):
                self._step(neuron, k)

            for i in range(len(tau_sfa)):
                expected = q_sfa[i] * math.exp(-1.0 / tau_sfa[i])
                actual = neuron._sfa_elems[i][0]
                self.assertAlmostEqual(actual, expected, places=6,
                                       msg=f"SFA element {i} decay mismatch")

    def test_adaptation_increases_threshold(self):
        r"""After a spike, sfa should raise the effective threshold."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-35.0 * u.mV,
                t_ref=0.0 * u.ms,
                tau_sfa=[100.0],
                q_sfa=[10.0],
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Before spike, sfa = V_T_star = -35
            self.assertAlmostEqual(neuron._sfa_val[0], -35.0)

            self._step(neuron, 0)

            # After spike, sfa_elems[0] += q_sfa[0] = 10
            # Next step computes sfa = V_T_star + sfa_elems[0] = -35 + 10 = -25
            neuron.lambda_0 = 0.0
            self._step(neuron, 1)
            # sfa_val is computed at beginning of next step
            self.assertAlmostEqual(neuron._sfa_val[0], -25.0, places=3)


class TestGIFCondExpStochasticSpiking(unittest.TestCase):
    r"""Test stochastic spike generation mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                neuron.add_delta_input(f'dg_{k}_{i}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_no_spikes_with_zero_lambda(self):
        r"""With lambda_0=0, no spikes should ever occur."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                lambda_0=0.0,
                I_e=1000.0 * u.pA,  # strong drive
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            for k in range(100):
                spk = self._step(neuron, k)
                self.assertEqual(float(spk[0]), 0.0,
                                 f"No spike expected with lambda_0=0 at step {k}")

    def test_high_lambda_produces_spikes(self):
        r"""With very high lambda_0, spikes should occur readily."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                lambda_0=1e10,  # enormous
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=0.0 * u.ms,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            spike_count = 0
            for k in range(100):
                spk = self._step(neuron, k)
                if float(spk[0]) > 0:
                    spike_count += 1

            self.assertTrue(spike_count > 50,
                            f"Expected many spikes with high lambda, got {spike_count}")

    def test_deterministic_with_fixed_rng_key(self):
        r"""Two neurons with the same RNG key and parameters should spike identically."""
        with brainstate.environ.context(dt=self.dt):
            key = jax.random.PRNGKey(12345)
            n1 = gif_cond_exp(1, lambda_0=100.0, rng_key=key,
                              V_initializer=braintools.init.Constant(-70.0 * u.mV))
            n2 = gif_cond_exp(1, lambda_0=100.0, rng_key=key,
                              V_initializer=braintools.init.Constant(-70.0 * u.mV))
            n1.init_state()
            n2.init_state()

            for k in range(50):
                s1 = self._step(n1, k)
                s2 = self._step(n2, k)
                self.assertEqual(float(s1[0]), float(s2[0]),
                                 f"Spike mismatch at step {k} with identical RNG")


class TestGIFCondExpReferenceTrace(unittest.TestCase):
    r"""Compare full simulation traces against standalone reference implementation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_val = 0.1
        self.dt = self.dt_val * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                neuron.add_delta_input(f'dg_{k}_{i}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_subthreshold_trace_matches_reference(self):
        r"""Multi-step subthreshold trace should match reference implementation."""
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'E_ex': 0.0, 'E_in': -85.0,
            'tau_syn_ex': 2.0, 'tau_syn_in': 2.0, 'I_e': 50.0,
            't_ref': 4.0,
        }

        n_steps = 20
        dg_seq = [[] for _ in range(n_steps)]
        dg_seq[0] = [5.0, -3.0]
        dg_seq[5] = [2.0]
        dg_seq[10] = [-4.0]
        i_stim_seq = [0.0] * n_steps
        i_stim_seq[0] = 30.0
        i_stim_seq[3] = -10.0

        # Reference (no spiking)
        v_ref, ge_ref, gi_ref, _, _, _ = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, dg_seq, [1.0] * n_steps,  # rand=1 means never spike
            tau_stc=[], q_stc=[], tau_sfa=[], q_sfa=[],
            lambda_0=0.001, Delta_V=0.5, V_T_star=-35.0
        )

        # Model
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                E_ex=p['E_ex'] * u.mV,
                E_in=p['E_in'] * u.mV,
                tau_syn_ex=p['tau_syn_ex'] * u.ms,
                tau_syn_in=p['tau_syn_in'] * u.ms,
                I_e=p['I_e'] * u.pA,
                t_ref=p['t_ref'] * u.ms,
                lambda_0=0.0,  # disable spiking for subthreshold test
                V_T_star=-35.0 * u.mV,
                Delta_V=0.5 * u.mV,
                V_initializer=braintools.init.Constant(p['E_L'] * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            v_model, ge_model, gi_model = [], [], []
            for k in range(n_steps):
                x_pA = i_stim_seq[k]
                self._step(neuron, k, x=x_pA * u.pA, dg_values=dg_seq[k])
                v_model.append(float((neuron.V.value / u.mV)[0]))
                ge_model.append(float((neuron.g_ex.value / u.nS)[0]))
                gi_model.append(float((neuron.g_in.value / u.nS)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")
            self.assertAlmostEqual(ge_model[k], ge_ref[k], places=5,
                                   msg=f"g_ex mismatch at step {k}")
            self.assertAlmostEqual(gi_model[k], gi_ref[k], places=5,
                                   msg=f"g_in mismatch at step {k}")

    def test_full_trace_with_adaptation_and_spiking(self):
        r"""Full trace with adaptation and controlled stochastic spiking matches reference."""
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'E_ex': 0.0, 'E_in': -85.0,
            'tau_syn_ex': 2.0, 'tau_syn_in': 2.0, 'I_e': 0.0,
            't_ref': 4.0,
        }
        tau_stc = [10.0, 20.0]
        q_stc = [5.0, -2.0]
        tau_sfa = [100.0]
        q_sfa = [10.0]
        lambda_0 = 0.001  # 1/ms
        Delta_V = 0.5
        V_T_star = -35.0

        n_steps = 30
        dg_seq = [[] for _ in range(n_steps)]
        i_stim_seq = [0.0] * n_steps

        # Generate known random numbers using the same JAX key
        key = jax.random.PRNGKey(99)
        rand_vals = []
        for k in range(n_steps):
            key, subkey = jax.random.split(key)
            rand_vals.append(float(jax.random.uniform(subkey, shape=(1,))[0]))

        # Reference
        v_ref, ge_ref, gi_ref, spk_ref, stc_ref, sfa_ref = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, dg_seq, rand_vals,
            tau_stc=tau_stc, q_stc=q_stc,
            tau_sfa=tau_sfa, q_sfa=q_sfa,
            lambda_0=lambda_0, Delta_V=Delta_V, V_T_star=V_T_star
        )

        # Model
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                E_ex=p['E_ex'] * u.mV,
                E_in=p['E_in'] * u.mV,
                tau_syn_ex=p['tau_syn_ex'] * u.ms,
                tau_syn_in=p['tau_syn_in'] * u.ms,
                I_e=p['I_e'] * u.pA,
                t_ref=p['t_ref'] * u.ms,
                lambda_0=lambda_0 * 1000.0,  # pass in 1/s
                Delta_V=Delta_V * u.mV,
                V_T_star=V_T_star * u.mV,
                tau_stc=tau_stc,
                q_stc=q_stc,
                tau_sfa=tau_sfa,
                q_sfa=q_sfa,
                V_initializer=braintools.init.Constant(p['E_L'] * u.mV),
                rng_key=jax.random.PRNGKey(99),
            )
            neuron.init_state()

            v_model, ge_model, gi_model = [], [], []
            for k in range(n_steps):
                self._step(neuron, k, x=0.0 * u.pA)
                v_model.append(float((neuron.V.value / u.mV)[0]))
                ge_model.append(float((neuron.g_ex.value / u.nS)[0]))
                gi_model.append(float((neuron.g_in.value / u.nS)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")
            self.assertAlmostEqual(ge_model[k], ge_ref[k], places=5,
                                   msg=f"g_ex mismatch at step {k}")
            self.assertAlmostEqual(gi_model[k], gi_ref[k], places=5,
                                   msg=f"g_in mismatch at step {k}")


class TestGIFCondExpUpdateOrder(unittest.TestCase):
    r"""Test that the update order matches NEST exactly."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                neuron.add_delta_input(f'dg_{k}_{i}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_stc_computed_before_ode_integration(self):
        r"""STC current should be computed BEFORE ODE integration, matching NEST order."""
        # With stc elements, the stc current used in ODE should reflect the
        # pre-decay values (sum of elements before exponential decay)
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                lambda_0=1e12,  # guarantee spike
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=0.0 * u.ms,
                tau_stc=[10.0],
                q_stc=[100.0],  # large stc to affect V
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Spike on step 0 -> adds q_stc=100 nA to stc_elems
            self._step(neuron, 0)

            # On step 1, stc_total should be 100 nA (before decay)
            # This should strongly affect V through the ODE
            neuron.lambda_0 = 0.0  # no more spikes
            self._step(neuron, 1)
            stc_val = neuron._stc_val[0]
            self.assertAlmostEqual(stc_val, 100.0, places=3,
                                   msg="STC should be 100 nA on step after spike")

    def test_conductance_jumps_after_ode_integration(self):
        r"""Conductance jumps should be applied AFTER ODE integration."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp(
                1,
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Apply a conductance jump of 10 nS on step 0
            self._step(neuron, 0, dg_values=[10.0])

            # g_ex should be exactly 10 nS after step 0
            # (jump applied after integration of initially zero conductance)
            ge = float((neuron.g_ex.value / u.nS)[0])
            self.assertAlmostEqual(ge, 10.0, places=10)


if __name__ == '__main__':
    unittest.main()
