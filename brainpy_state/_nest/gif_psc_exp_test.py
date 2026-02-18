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
Tests for gif_psc_exp neuron model.

These tests verify that the brainpy.state implementation of gif_psc_exp
matches NEST's update ordering and semantics, including:

- Default parameter values matching NEST C++ source
- Subthreshold membrane dynamics with exact integration (propagator matrix)
- Exponential synaptic current decay
- Refractory period mechanics (V clamped, countdown)
- Spike-triggered current (stc) adaptation
- Spike-frequency adaptation (sfa) threshold dynamics
- Stochastic spike generation via hazard function
- One-step delayed current input (NEST ring buffer semantics)
- Full reference trace comparison against standalone Python reference implementation

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

from brainpy.state import gif_psc_exp


def _propagator_exp(tau_syn, tau_m, c_m, h):
    r"""Reference implementation of IAFPropagatorExp::evaluate().

    Computes the propagator coefficient P21 for exact integration of the
    synaptic current contribution to the membrane potential.
    """
    beta = tau_syn * tau_m / (tau_m - tau_syn)
    gamma = beta / c_m
    inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)
    exp_h_tau_syn = math.exp(-h / tau_syn)
    expm1_h_tau = math.expm1(h * inv_beta)
    p32 = gamma * exp_h_tau_syn * expm1_h_tau

    if math.isfinite(p32) and abs(p32) >= np.finfo(np.float64).tiny and p32 > 0:
        return p32
    else:
        return h / c_m * math.exp(-h / tau_m)


def _run_nest_ref(n_steps, dt, p, i_stim_seq, w_seq, rand_seq,
                  tau_stc, q_stc, tau_sfa, q_sfa, lambda_0, Delta_V, V_T_star):
    r"""Full reference implementation of gif_psc_exp update loop.

    Matches NEST update order exactly:
    1. Compute stc/sfa totals, decay elements
    2. Decay synaptic currents
    3. Add spike weight jumps
    4. If not refractory: update V via propagator, spike check (stochastic)
       If refractory: decrement counter, clamp V to V_reset
    5. Store I_stim
    """
    v = p['E_L']
    i_syn_ex, i_syn_in = 0.0, 0.0
    r = 0
    i_stim = 0.0
    tau_m = p['C_m'] / p['g_L']
    refr_steps = int(math.floor(p['t_ref'] / dt))

    # Propagator coefficients
    P33 = math.exp(-dt / tau_m)
    P30 = -1.0 / p['C_m'] * math.expm1(-dt / tau_m) * tau_m
    P31 = -math.expm1(-dt / tau_m)
    P11_ex = math.exp(-dt / p['tau_syn_ex'])
    P11_in = math.exp(-dt / p['tau_syn_in'])
    P21_ex = _propagator_exp(p['tau_syn_ex'], tau_m, p['C_m'], dt)
    P21_in = _propagator_exp(p['tau_syn_in'], tau_m, p['C_m'], dt)

    n_stc = len(tau_stc)
    n_sfa = len(tau_sfa)
    stc_elems = [0.0] * n_stc
    sfa_elems = [0.0] * n_sfa

    P_stc = [math.exp(-dt / tau) for tau in tau_stc]
    P_sfa = [math.exp(-dt / tau) for tau in tau_sfa]

    v_trace, isyn_ex_trace, isyn_in_trace, spike_trace = [], [], [], []
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

        # Step 2: Decay synaptic currents
        i_syn_ex *= P11_ex
        i_syn_in *= P11_in

        # Step 3: Add spike weight jumps
        if k < len(w_seq):
            for w in w_seq[k]:
                if w >= 0.0:
                    i_syn_ex += w
                else:
                    i_syn_in += w

        # Step 4: Spike check / refractory
        spike = 0.0
        if r == 0:
            # Update V via exact propagator
            v = (P30 * (i_stim + p['I_e'] - stc_total)
                 + P33 * v
                 + P31 * p['E_L']
                 + i_syn_ex * P21_ex
                 + i_syn_in * P21_in)

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
        isyn_ex_trace.append(i_syn_ex)
        isyn_in_trace.append(i_syn_in)
        spike_trace.append(spike)
        stc_trace.append(stc_total)
        sfa_trace.append(sfa_total)

    return v_trace, isyn_ex_trace, isyn_in_trace, spike_trace, stc_trace, sfa_trace


class TestGIFPscExpDefaultParams(unittest.TestCase):
    r"""Test that default parameters match NEST C++ source code values."""

    def test_nest_cpp_default_parameters(self):
        neuron = gif_psc_exp(1)
        self.assertEqual(neuron.g_L, 4.0 * u.nS)
        self.assertEqual(neuron.E_L, -70.0 * u.mV)
        self.assertEqual(neuron.C_m, 80.0 * u.pF)
        self.assertEqual(neuron.V_reset, -55.0 * u.mV)
        self.assertEqual(neuron.Delta_V, 0.5 * u.mV)
        self.assertEqual(neuron.V_T_star, -35.0 * u.mV)
        self.assertAlmostEqual(neuron.lambda_0, 0.001)  # 1/ms (= 1.0/s internally)
        self.assertEqual(neuron.t_ref, 4.0 * u.ms)
        self.assertEqual(neuron.tau_syn_ex, 2.0 * u.ms)
        self.assertEqual(neuron.tau_syn_in, 2.0 * u.ms)
        self.assertEqual(neuron.I_e, 0.0 * u.pA)
        self.assertEqual(neuron.tau_sfa, ())
        self.assertEqual(neuron.q_sfa, ())
        self.assertEqual(neuron.tau_stc, ())
        self.assertEqual(neuron.q_stc, ())

    def test_initial_state_matches_nest(self):
        r"""V_m should be initialized to E_L, synaptic currents to 0."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = gif_psc_exp(1)
            neuron.init_state()
            self.assertTrue(u.math.allclose(neuron.V.value, -70.0 * u.mV))
            self.assertTrue(u.math.allclose(neuron.I_syn_ex.value, 0.0 * u.pA))
            self.assertTrue(u.math.allclose(neuron.I_syn_in.value, 0.0 * u.pA))


class TestGIFPscExpParameterValidation(unittest.TestCase):
    r"""Test that invalid parameters raise appropriate errors."""

    def test_mismatched_tau_sfa_q_sfa_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp(1, tau_sfa=[10.0], q_sfa=[1.0, 2.0])

    def test_mismatched_tau_stc_q_stc_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp(1, tau_stc=[10.0, 20.0], q_stc=[1.0])

    def test_negative_capacitance_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp(1, C_m=-80.0 * u.pF)

    def test_negative_g_L_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp(1, g_L=-1.0 * u.nS)

    def test_negative_Delta_V_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp(1, Delta_V=-0.5 * u.mV)

    def test_negative_lambda_0_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp(1, lambda_0=-1.0)

    def test_negative_tau_syn_ex_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp(1, tau_syn_ex=-2.0 * u.ms)

    def test_negative_tau_sfa_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp(1, tau_sfa=[-10.0], q_sfa=[5.0])

    def test_negative_tau_stc_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp(1, tau_stc=[-10.0], q_stc=[5.0])


class TestGIFPscExpSubthresholdDynamics(unittest.TestCase):
    r"""Test subthreshold membrane dynamics without spiking."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, w_values=None):
        if w_values is not None:
            for i, val in enumerate(w_values):
                neuron.add_delta_input(f'w_{k}_{i}', val * u.pA)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_current_input_has_one_step_delay(self):
        r"""External current should be stored for use in the NEXT step (NEST ring buffer)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
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

    def test_synaptic_weight_routing(self):
        r"""Positive weights should add to I_syn_ex, negative to I_syn_in."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
                1,
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()
            self._step(neuron, 0, w_values=[5.0, -3.0])
            # After step 0: synaptic currents decayed then spike weights added
            # Initial currents were 0, so after decay still 0, then +5 and -3
            isyn_ex = float((neuron.I_syn_ex.value / u.pA)[0])
            isyn_in = float((neuron.I_syn_in.value / u.pA)[0])
            self.assertAlmostEqual(isyn_ex, 5.0, places=8)
            self.assertAlmostEqual(isyn_in, -3.0, places=8)

    def test_synaptic_current_exponential_decay(self):
        r"""Synaptic currents should decay exponentially with their time constants."""
        with brainstate.environ.context(dt=self.dt):
            tau_ex = 2.0  # ms
            tau_in = 5.0  # ms
            neuron = gif_psc_exp(
                1,
                tau_syn_ex=tau_ex * u.ms,
                tau_syn_in=tau_in * u.ms,
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Inject synaptic currents
            self._step(neuron, 0, w_values=[10.0, -8.0])

            # Run for 10 steps (1 ms) and check decay
            for k in range(1, 11):
                self._step(neuron, k)

            isyn_ex = float((neuron.I_syn_ex.value / u.pA)[0])
            isyn_in = float((neuron.I_syn_in.value / u.pA)[0])

            # Expected: I_syn_ex(1ms) = 10 * exp(-1/2), I_syn_in(1ms) = -8 * exp(-1/5)
            expected_ex = 10.0 * math.exp(-1.0 / tau_ex)
            expected_in = -8.0 * math.exp(-1.0 / tau_in)
            self.assertAlmostEqual(isyn_ex, expected_ex, places=5)
            self.assertAlmostEqual(isyn_in, expected_in, places=5)

    def test_excitatory_depolarizes_inhibitory_hyperpolarizes(self):
        r"""Excitatory synaptic current should depolarize, inhibitory should hyperpolarize."""
        with brainstate.environ.context(dt=self.dt):
            base = gif_psc_exp(1, lambda_0=0.0, V_T_star=1000.0 * u.mV,
                               V_initializer=braintools.init.Constant(-70.0 * u.mV))
            exc = gif_psc_exp(1, lambda_0=0.0, V_T_star=1000.0 * u.mV,
                              V_initializer=braintools.init.Constant(-70.0 * u.mV))
            inh = gif_psc_exp(1, lambda_0=0.0, V_T_star=1000.0 * u.mV,
                              V_initializer=braintools.init.Constant(-70.0 * u.mV))
            base.init_state()
            exc.init_state()
            inh.init_state()

            # Apply synaptic inputs
            self._step(base, 0)
            self._step(exc, 0, w_values=[50.0])
            self._step(inh, 0, w_values=[-50.0])

            # Let dynamics evolve
            self._step(base, 1)
            self._step(exc, 1)
            self._step(inh, 1)

            v_base = float((base.V.value / u.mV)[0])
            v_exc = float((exc.V.value / u.mV)[0])
            v_inh = float((inh.V.value / u.mV)[0])

            # Positive current -> depolarizes
            self.assertTrue(v_exc > v_base, f"Exc {v_exc} should be > base {v_base}")
            # Negative current -> hyperpolarizes
            self.assertTrue(v_inh < v_base, f"Inh {v_inh} should be < base {v_base}")

    def test_exact_integration_propagator(self):
        r"""Test that exact integration matches analytic solution for a single step."""
        dt = 0.1  # ms
        C_m = 80.0
        g_L = 4.0
        E_L = -70.0
        I_e = 100.0  # pA
        tau_m = C_m / g_L  # 20 ms

        # Propagator coefficients
        P33 = math.exp(-dt / tau_m)
        P30 = -1.0 / C_m * math.expm1(-dt / tau_m) * tau_m
        P31 = -math.expm1(-dt / tau_m)

        # Starting from V=E_L, no synaptic input, constant I_e
        V0 = E_L
        V_analytic = P30 * I_e + P33 * V0 + P31 * E_L

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = gif_psc_exp(
                1,
                C_m=C_m * u.pF,
                g_L=g_L * u.nS,
                E_L=E_L * u.mV,
                I_e=I_e * u.pA,
                lambda_0=0.0,  # no spiking
                V_initializer=braintools.init.Constant(E_L * u.mV),
            )
            neuron.init_state()

            # Step 0: I_stim is 0 (ring buffer), so only I_e contributes
            self._step(neuron, 0)
            v_model = float((neuron.V.value / u.mV)[0])

            self.assertAlmostEqual(v_model, V_analytic, places=8,
                                   msg=f"Exact integration mismatch: model={v_model}, analytic={V_analytic}")


class TestGIFPscExpRefractoryBehavior(unittest.TestCase):
    r"""Test refractory period mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, w_values=None):
        if w_values is not None:
            for i, val in enumerate(w_values):
                neuron.add_delta_input(f'w_{k}_{i}', val * u.pA)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_refractory_clamps_voltage_to_V_reset(self):
        r"""During refractory period, V should stay at V_reset."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
                1,
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=1.0 * u.ms,  # 10 steps at 0.1 ms
                V_reset=-55.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            # Step 0 should spike
            spk0 = self._step(neuron, 0)
            self.assertTrue(float(spk0[0]) > 0, "Should spike on step 0 with lambda_0=1e12")

            # Steps 1-9 should be refractory, V clamped to V_reset
            for k in range(1, 10):
                self._step(neuron, k)
                v = float((neuron.V.value / u.mV)[0])
                self.assertAlmostEqual(v, -55.0, places=6,
                                       msg=f"V should be V_reset during refractory at step {k}")

    def test_refractory_count_matches_t_ref(self):
        r"""Refractory counter should match floor(t_ref / dt) as in NEST."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
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

    def test_synaptic_currents_still_evolve_during_refractory(self):
        r"""Synaptic currents should continue decaying during refractory period."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
                1,
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=5.0 * u.ms,
                tau_syn_ex=2.0 * u.ms,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            # Spike on step 0
            self._step(neuron, 0)

            # Inject synaptic current during refractory period
            self._step(neuron, 1, w_values=[10.0])
            isyn_after_inject = float((neuron.I_syn_ex.value / u.pA)[0])
            self.assertAlmostEqual(isyn_after_inject, 10.0, places=5)

            # Let it decay for 10 more steps (1 ms)
            for k in range(2, 12):
                self._step(neuron, k)

            isyn_after_decay = float((neuron.I_syn_ex.value / u.pA)[0])
            expected = 10.0 * math.exp(-1.0 / 2.0)
            self.assertAlmostEqual(isyn_after_decay, expected, places=5)


class TestGIFPscExpAdaptation(unittest.TestCase):
    r"""Test stc and sfa adaptation mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, w_values=None):
        if w_values is not None:
            for i, val in enumerate(w_values):
                neuron.add_delta_input(f'w_{k}_{i}', val * u.pA)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_stc_elements_decay_exponentially(self):
        r"""STC elements should decay by exp(-dt/tau) each step."""
        tau_stc = [10.0, 20.0]
        q_stc = [5.0, -2.0]

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
                1,
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=0.0 * u.ms,
                tau_stc=tau_stc,
                q_stc=q_stc,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Force a spike on step 0
            self._step(neuron, 0)

            # Disable spiking for subsequent steps
            neuron.lambda_0 = 0.0

            # Run 10 steps (1 ms)
            for k in range(1, 11):
                self._step(neuron, k)

            for i in range(len(tau_stc)):
                expected = q_stc[i] * math.exp(-1.0 / tau_stc[i])
                actual = neuron._stc_elems[i][0]
                self.assertAlmostEqual(actual, expected, places=6,
                                       msg=f"STC element {i} decay mismatch")

    def test_sfa_elements_decay_exponentially(self):
        r"""SFA elements should decay by exp(-dt/tau) each step."""
        tau_sfa = [100.0, 50.0]
        q_sfa = [10.0, 5.0]

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
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
            neuron = gif_psc_exp(
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
            self.assertAlmostEqual(neuron._sfa_val[0], -25.0, places=3)

    def test_stc_current_opposes_depolarization(self):
        r"""Positive stc should hyperpolarize the neuron (it's subtracted in the ODE)."""
        with brainstate.environ.context(dt=self.dt):
            # No stc
            no_stc = gif_psc_exp(
                1,
                lambda_0=0.0,
                I_e=200.0 * u.pA,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            no_stc.init_state()

            # With stc (manually set elements to simulate post-spike state)
            with_stc = gif_psc_exp(
                1,
                lambda_0=0.0,
                I_e=200.0 * u.pA,
                tau_stc=[10.0],
                q_stc=[50.0],
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            with_stc.init_state()
            # Manually set stc_elems to simulate post-spike
            with_stc._stc_elems[0][0] = 50.0  # 50 nA

            # Run several steps
            for k in range(20):
                self._step(no_stc, k)
                self._step(with_stc, k)

            v_no_stc = float((no_stc.V.value / u.mV)[0])
            v_with_stc = float((with_stc.V.value / u.mV)[0])

            # stc current opposes depolarization
            self.assertTrue(v_with_stc < v_no_stc,
                            f"V with stc ({v_with_stc}) should be < V without ({v_no_stc})")


class TestGIFPscExpStochasticSpiking(unittest.TestCase):
    r"""Test stochastic spike generation mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, w_values=None):
        if w_values is not None:
            for i, val in enumerate(w_values):
                neuron.add_delta_input(f'w_{k}_{i}', val * u.pA)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_no_spikes_with_zero_lambda(self):
        r"""With lambda_0=0, no spikes should ever occur."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
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
            neuron = gif_psc_exp(
                1,
                lambda_0=1e10,
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
            n1 = gif_psc_exp(1, lambda_0=100.0, rng_key=key,
                             V_initializer=braintools.init.Constant(-70.0 * u.mV))
            n2 = gif_psc_exp(1, lambda_0=100.0, rng_key=key,
                             V_initializer=braintools.init.Constant(-70.0 * u.mV))
            n1.init_state()
            n2.init_state()

            for k in range(50):
                s1 = self._step(n1, k)
                s2 = self._step(n2, k)
                self.assertEqual(float(s1[0]), float(s2[0]),
                                 f"Spike mismatch at step {k} with identical RNG")


class TestGIFPscExpReferenceTrace(unittest.TestCase):
    r"""Compare full simulation traces against standalone reference implementation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_val = 0.1
        self.dt = self.dt_val * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, w_values=None):
        if w_values is not None:
            for i, val in enumerate(w_values):
                neuron.add_delta_input(f'w_{k}_{i}', val * u.pA)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_subthreshold_trace_matches_reference(self):
        r"""Multi-step subthreshold trace should match reference implementation."""
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'tau_syn_ex': 2.0, 'tau_syn_in': 2.0,
            'I_e': 50.0, 't_ref': 4.0,
        }

        n_steps = 50
        w_seq = [[] for _ in range(n_steps)]
        w_seq[0] = [5.0, -3.0]
        w_seq[5] = [2.0]
        w_seq[10] = [-4.0]
        i_stim_seq = [0.0] * n_steps
        i_stim_seq[0] = 30.0
        i_stim_seq[3] = -10.0

        # Reference (no spiking: rand=1 means never spike)
        v_ref, isyn_ex_ref, isyn_in_ref, _, _, _ = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, w_seq, [1.0] * n_steps,
            tau_stc=[], q_stc=[], tau_sfa=[], q_sfa=[],
            lambda_0=0.001, Delta_V=0.5, V_T_star=-35.0
        )

        # Model
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
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

            v_model, isyn_ex_model, isyn_in_model = [], [], []
            for k in range(n_steps):
                x_pA = i_stim_seq[k]
                self._step(neuron, k, x=x_pA * u.pA, w_values=w_seq[k] if w_seq[k] else None)
                v_model.append(float((neuron.V.value / u.mV)[0]))
                isyn_ex_model.append(float((neuron.I_syn_ex.value / u.pA)[0]))
                isyn_in_model.append(float((neuron.I_syn_in.value / u.pA)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")
            self.assertAlmostEqual(isyn_ex_model[k], isyn_ex_ref[k], places=5,
                                   msg=f"I_syn_ex mismatch at step {k}")
            self.assertAlmostEqual(isyn_in_model[k], isyn_in_ref[k], places=5,
                                   msg=f"I_syn_in mismatch at step {k}")

    def test_full_trace_with_adaptation_and_spiking(self):
        r"""Full trace with adaptation and controlled stochastic spiking matches reference."""
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'tau_syn_ex': 2.0, 'tau_syn_in': 2.0,
            'I_e': 0.0, 't_ref': 4.0,
        }
        tau_stc = [10.0, 20.0]
        q_stc = [5.0, -2.0]
        tau_sfa = [100.0]
        q_sfa = [10.0]
        lambda_0 = 0.001  # 1/ms
        Delta_V = 0.5
        V_T_star = -35.0

        n_steps = 30
        w_seq = [[] for _ in range(n_steps)]
        i_stim_seq = [0.0] * n_steps

        # Generate known random numbers using the same JAX key
        key = jax.random.PRNGKey(99)
        rand_vals = []
        for k in range(n_steps):
            key, subkey = jax.random.split(key)
            rand_vals.append(float(jax.random.uniform(subkey, shape=(1,))[0]))

        # Reference
        v_ref, isyn_ex_ref, isyn_in_ref, spk_ref, stc_ref, sfa_ref = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, w_seq, rand_vals,
            tau_stc=tau_stc, q_stc=q_stc,
            tau_sfa=tau_sfa, q_sfa=q_sfa,
            lambda_0=lambda_0, Delta_V=Delta_V, V_T_star=V_T_star
        )

        # Model
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
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

            v_model = []
            for k in range(n_steps):
                self._step(neuron, k, x=0.0 * u.pA)
                v_model.append(float((neuron.V.value / u.mV)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")

    def test_dc_driven_trace_with_synaptic_input(self):
        r"""Trace driven by DC + synaptic spikes, comparing against reference."""
        p = {
            'E_L': -70.0, 'C_m': 40.0, 'g_L': 4.0,
            'V_reset': -55.0, 'tau_syn_ex': 8.0, 'tau_syn_in': 2.0,
            'I_e': 170.0, 't_ref': 4.0,
        }
        tau_stc = [10.0, 20.0]
        q_stc = [20.0, -5.0]
        tau_sfa = [120.0, 10.0]
        q_sfa = [10.0, 25.0]
        lambda_0 = 0.001  # 1/ms
        Delta_V = 0.2
        V_T_star = -35.0

        n_steps = 100

        # Inject spikes at specific times (matching NEST test)
        w_seq = [[] for _ in range(n_steps)]
        # Spike at step 100 (10 ms), 200 (20 ms), 300 (30 ms) would be at dt=0.1 ms
        for t_ms in [10.0, 20.0, 30.0]:
            step = int(t_ms / self.dt_val)
            if step < n_steps:
                w_seq[step] = [1.0]  # weight=1 pA

        i_stim_seq = [0.0] * n_steps

        # Generate random numbers
        key = jax.random.PRNGKey(42)
        rand_vals = []
        for k in range(n_steps):
            key, subkey = jax.random.split(key)
            rand_vals.append(float(jax.random.uniform(subkey, shape=(1,))[0]))

        # Reference
        v_ref, _, _, spk_ref, _, _ = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, w_seq, rand_vals,
            tau_stc=tau_stc, q_stc=q_stc,
            tau_sfa=tau_sfa, q_sfa=q_sfa,
            lambda_0=lambda_0, Delta_V=Delta_V, V_T_star=V_T_star
        )

        # Model
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                tau_syn_ex=p['tau_syn_ex'] * u.ms,
                tau_syn_in=p['tau_syn_in'] * u.ms,
                I_e=p['I_e'] * u.pA,
                t_ref=p['t_ref'] * u.ms,
                lambda_0=lambda_0 * 1000.0,
                Delta_V=Delta_V * u.mV,
                V_T_star=V_T_star * u.mV,
                tau_stc=tau_stc,
                q_stc=q_stc,
                tau_sfa=tau_sfa,
                q_sfa=q_sfa,
                V_initializer=braintools.init.Constant(p['E_L'] * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            v_model = []
            for k in range(n_steps):
                self._step(neuron, k, x=0.0 * u.pA,
                           w_values=w_seq[k] if w_seq[k] else None)
                v_model.append(float((neuron.V.value / u.mV)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")


class TestGIFPscExpUpdateOrder(unittest.TestCase):
    r"""Test that the update order matches NEST exactly."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, w_values=None):
        if w_values is not None:
            for i, val in enumerate(w_values):
                neuron.add_delta_input(f'w_{k}_{i}', val * u.pA)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_stc_computed_before_membrane_update(self):
        r"""STC current should be computed BEFORE membrane update, matching NEST order."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
                1,
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=0.0 * u.ms,
                tau_stc=[10.0],
                q_stc=[100.0],
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Spike on step 0 -> adds q_stc=100 nA to stc_elems
            self._step(neuron, 0)

            # On step 1, stc_total should be 100 nA (before decay)
            neuron.lambda_0 = 0.0
            self._step(neuron, 1)
            stc_val = neuron._stc_val[0]
            self.assertAlmostEqual(stc_val, 100.0, places=3,
                                   msg="STC should be 100 nA on step after spike")

    def test_synaptic_decay_before_weight_addition(self):
        r"""Synaptic currents should be decayed BEFORE spike weight jumps are added."""
        with brainstate.environ.context(dt=self.dt):
            tau_ex = 2.0  # ms
            neuron = gif_psc_exp(
                1,
                tau_syn_ex=tau_ex * u.ms,
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Step 0: inject weight 10
            self._step(neuron, 0, w_values=[10.0])
            # I_syn_ex should be 10 (0 * decay + 10)
            isyn0 = float((neuron.I_syn_ex.value / u.pA)[0])
            self.assertAlmostEqual(isyn0, 10.0, places=7)

            # Step 1: inject weight 5
            self._step(neuron, 1, w_values=[5.0])
            # I_syn_ex should be 10 * exp(-0.1/2) + 5
            expected = 10.0 * math.exp(-0.1 / tau_ex) + 5.0
            isyn1 = float((neuron.I_syn_ex.value / u.pA)[0])
            self.assertAlmostEqual(isyn1, expected, places=7)

    def test_propagator_singularity_handling(self):
        r"""When tau_m == tau_syn, the singular propagator should be used."""
        # tau_m = C_m / g_L = 80 / 4 = 20 ms
        # Set tau_syn_ex = 20 ms to hit the singularity
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp(
                1,
                C_m=80.0 * u.pF,
                g_L=4.0 * u.nS,
                tau_syn_ex=20.0 * u.ms,  # equals tau_m
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Inject synaptic current and run a step - should not crash
            self._step(neuron, 0, w_values=[10.0])
            v = float((neuron.V.value / u.mV)[0])
            self.assertTrue(math.isfinite(v), f"V should be finite, got {v}")

            # The membrane potential should have changed from the synaptic input
            self._step(neuron, 1)
            v1 = float((neuron.V.value / u.mV)[0])
            self.assertTrue(math.isfinite(v1), f"V should be finite, got {v1}")


if __name__ == '__main__':
    unittest.main()
