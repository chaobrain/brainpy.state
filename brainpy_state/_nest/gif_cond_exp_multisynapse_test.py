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
Tests for gif_cond_exp_multisynapse neuron model.

These tests verify that the brainpy.state implementation of
gif_cond_exp_multisynapse matches NEST's update ordering and semantics,
including:

- Default parameter values matching NEST C++ source
- Multiple receptor port conductance dynamics
- Exponential conductance decay per receptor
- Subthreshold membrane dynamics with conductance-based synapses
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

from brainpy_state._nest.gif_cond_exp_multisynapse import gif_cond_exp_multisynapse


def _rkf45_ref_step_multi(v, g_vals, is_refractory, i_stim, stc, dt, h0, p,
                          tau_syn, E_rev, atol=1e-3):
    r"""Reference RKF45 integration for a single simulation step (multisynapse).

    This is a standalone Python implementation of the RKF45 adaptive integrator
    that matches the NEST GSL integration behavior for gif_cond_exp_multisynapse.

    Parameters
    ----------
    v : float
        Membrane potential (mV)
    g_vals : list of float
        Per-receptor conductances (nS)
    is_refractory : bool
        Whether neuron is in refractory period
    i_stim : float
        Stimulus current (pA)
    stc : float
        Total spike-triggered current (nA)
    dt : float
        Simulation time step (ms)
    h0 : float
        Initial integration step size (ms)
    p : dict
        Scalar parameters {V_reset, E_L, C_m, g_L, I_e}
    tau_syn : list of float
        Synaptic time constants per receptor (ms)
    E_rev : list of float
        Reversal potentials per receptor (mV)
    atol : float
        Absolute error tolerance

    Returns
    -------
    v, g_vals, h : float, list of float, float
        Updated state and step size
    """
    n_rec = len(tau_syn)
    n = 1 + n_rec  # total state dimension
    min_h = 1e-8
    t = 0.0
    h = max(h0, min_h)

    # Pack state: [v, g_0, g_1, ..., g_{n_rec-1}]
    y = [v] + list(g_vals)

    def f(state):
        V = p['V_reset'] if is_refractory else state[0]
        I_syn = sum(-state[1 + k] * (V - E_rev[k]) for k in range(n_rec))
        I_L = p['g_L'] * (V - p['E_L'])
        dv = 0.0 if is_refractory else (-I_L + i_stim + p['I_e'] + I_syn - stc) / p['C_m']
        dg = [-state[1 + k] / tau_syn[k] for k in range(n_rec)]
        return [dv] + dg

    while t < dt:
        h = max(min_h, min(h, dt - t))

        k1 = f(y)
        y2 = [y[i] + h * k1[i] / 4.0 for i in range(n)]
        k2 = f(y2)
        y3 = [y[i] + h * (3.0 * k1[i] / 32.0 + 9.0 * k2[i] / 32.0) for i in range(n)]
        k3 = f(y3)
        y4 = [y[i] + h * (1932.0 * k1[i] / 2197.0 - 7200.0 * k2[i] / 2197.0 + 7296.0 * k3[i] / 2197.0)
              for i in range(n)]
        k4 = f(y4)
        y5 = [y[i] + h * (439.0 * k1[i] / 216.0 - 8.0 * k2[i] + 3680.0 * k3[i] / 513.0
                           - 845.0 * k4[i] / 4104.0) for i in range(n)]
        k5 = f(y5)
        y6 = [y[i] + h * (-8.0 * k1[i] / 27.0 + 2.0 * k2[i] - 3544.0 * k3[i] / 2565.0
                           + 1859.0 * k4[i] / 4104.0 - 11.0 * k5[i] / 40.0) for i in range(n)]
        k6 = f(y6)

        y4_sol = [y[i] + h * (25.0 * k1[i] / 216.0 + 1408.0 * k3[i] / 2565.0
                               + 2197.0 * k4[i] / 4104.0 - k5[i] / 5.0) for i in range(n)]
        y5_sol = [y[i] + h * (16.0 * k1[i] / 135.0 + 6656.0 * k3[i] / 12825.0
                               + 28561.0 * k4[i] / 56430.0 - 9.0 * k5[i] / 50.0
                               + 2.0 * k6[i] / 55.0) for i in range(n)]

        err = max(abs(y5_sol[i] - y4_sol[i]) for i in range(n))
        if err <= atol or h <= min_h:
            y = y5_sol
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    return y[0], y[1:], h


def _run_nest_ref_multi(n_steps, dt, p, i_stim_seq, dg_seq, rand_seq,
                        tau_stc, q_stc, tau_sfa, q_sfa, lambda_0, Delta_V,
                        V_T_star, tau_syn, E_rev):
    r"""Full reference implementation of gif_cond_exp_multisynapse update loop.

    Matches NEST update order exactly:
    1. Compute stc/sfa totals, decay elements
    2. Integrate ODEs
    3. Add conductance jumps (per receptor)
    4. Spike check (stochastic) or refractory countdown
    5. Store I_stim
    """
    n_rec = len(tau_syn)
    v = p['E_L']
    g_vals = [0.0] * n_rec
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

    v_trace, g_traces, spike_trace = [], [[] for _ in range(n_rec)], []
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
        v, g_vals, h = _rkf45_ref_step_multi(
            v, g_vals, is_refractory, i_stim, stc_total, dt, h, p,
            tau_syn, E_rev
        )

        # Step 3: Add conductance jumps per receptor
        if k < len(dg_seq):
            for port, dg_val in dg_seq[k]:
                if 0 <= port < n_rec:
                    g_vals[port] += dg_val

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
        for port in range(n_rec):
            g_traces[port].append(g_vals[port])
        spike_trace.append(spike)
        stc_trace.append(stc_total)
        sfa_trace.append(sfa_total)

    return v_trace, g_traces, spike_trace, stc_trace, sfa_trace


class TestGIFCondExpMultisynDefaultParams(unittest.TestCase):
    r"""Test that default parameters match NEST C++ source code values."""

    def test_nest_cpp_default_parameters(self):
        neuron = gif_cond_exp_multisynapse(1)
        self.assertEqual(neuron.g_L, 4.0 * u.nS)
        self.assertEqual(neuron.E_L, -70.0 * u.mV)
        self.assertEqual(neuron.C_m, 80.0 * u.pF)
        self.assertEqual(neuron.V_reset, -55.0 * u.mV)
        self.assertEqual(neuron.Delta_V, 0.5 * u.mV)
        self.assertEqual(neuron.V_T_star, -35.0 * u.mV)
        self.assertAlmostEqual(neuron.lambda_0, 0.001)  # 1/ms (= 1/s internally)
        self.assertEqual(neuron.t_ref, 4.0 * u.ms)
        self.assertEqual(neuron.I_e, 0.0 * u.pA)
        # Multisynapse defaults: single receptor with tau_syn=2.0, E_rev=0.0
        self.assertEqual(neuron.tau_syn, (2.0,))
        self.assertEqual(neuron.E_rev, (0.0,))
        self.assertEqual(neuron.n_receptors, 1)
        self.assertEqual(neuron.tau_sfa, ())
        self.assertEqual(neuron.q_sfa, ())
        self.assertEqual(neuron.tau_stc, ())
        self.assertEqual(neuron.q_stc, ())

    def test_initial_state_matches_nest(self):
        r"""V_m should be initialized to E_L, all conductances to 0."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = gif_cond_exp_multisynapse(
                1, tau_syn=(4.0, 8.0), E_rev=(0.0, -85.0)
            )
            neuron.init_state()
            self.assertTrue(u.math.allclose(neuron.V.value, -70.0 * u.mV))
            self.assertEqual(len(neuron.g), 2)
            self.assertTrue(u.math.allclose(neuron.g[0].value, 0.0 * u.nS))
            self.assertTrue(u.math.allclose(neuron.g[1].value, 0.0 * u.nS))

    def test_multiple_receptors(self):
        r"""Verify model with multiple receptor ports."""
        neuron = gif_cond_exp_multisynapse(
            1, tau_syn=(2.0, 4.0, 8.0), E_rev=(0.0, -85.0, -65.0)
        )
        self.assertEqual(neuron.n_receptors, 3)
        self.assertEqual(neuron.tau_syn, (2.0, 4.0, 8.0))
        self.assertEqual(neuron.E_rev, (0.0, -85.0, -65.0))


class TestGIFCondExpMultisynParameterValidation(unittest.TestCase):
    r"""Test that invalid parameters raise appropriate errors."""

    def test_mismatched_tau_syn_E_rev_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, tau_syn=(2.0, 4.0), E_rev=(0.0,))

    def test_empty_tau_syn_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, tau_syn=(), E_rev=())

    def test_mismatched_tau_sfa_q_sfa_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, tau_sfa=[10.0], q_sfa=[1.0, 2.0])

    def test_mismatched_tau_stc_q_stc_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, tau_stc=[10.0, 20.0], q_stc=[1.0])

    def test_negative_capacitance_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, C_m=-80.0 * u.pF)

    def test_negative_g_L_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, g_L=-1.0 * u.nS)

    def test_negative_Delta_V_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, Delta_V=-0.5 * u.mV)

    def test_negative_lambda_0_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, lambda_0=-1.0)

    def test_negative_tau_syn_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, tau_syn=(-2.0,), E_rev=(0.0,))

    def test_zero_tau_syn_raises(self):
        with self.assertRaises(ValueError):
            gif_cond_exp_multisynapse(1, tau_syn=(0.0,), E_rev=(0.0,))


class TestGIFCondExpMultisynSubthresholdDynamics(unittest.TestCase):
    r"""Test subthreshold membrane dynamics without spiking."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        r"""Run one step. dg_values is a list of (port, weight_nS) tuples."""
        if dg_values is not None:
            for port, val in dg_values:
                neuron.add_delta_input(f'receptor_{port}_{k}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_current_input_has_one_step_delay(self):
        r"""External current should be stored for use in the NEXT step (NEST ring buffer)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
                1,
                E_L=0.0 * u.mV,
                g_L=0.0001 * u.nS,
                I_e=0.0 * u.pA,
                lambda_0=0.0,
                V_reset=0.0 * u.mV,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            # Step 0: inject current. V should still be ~0 because I_stim was 0
            self._step(neuron, 0, x=100.0 * u.pA)
            v0 = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v0, 0.0, places=3)

            # Step 1: now the 100 pA should take effect
            self._step(neuron, 1, x=0.0 * u.pA)
            v1 = float((neuron.V.value / u.mV)[0])
            self.assertTrue(v1 > 0.0, f"V should increase from current, got {v1}")

    def test_per_receptor_conductance_jumps(self):
        r"""Conductance jumps should go to the correct receptor port."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
                1,
                tau_syn=(4.0, 8.0),
                E_rev=(0.0, -85.0),
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Add conductance to receptor 0 and receptor 1
            self._step(neuron, 0, dg_values=[(0, 5.0), (1, 3.0)])

            g0 = float((neuron.g[0].value / u.nS)[0])
            g1 = float((neuron.g[1].value / u.nS)[0])
            self.assertAlmostEqual(g0, 5.0, places=10)
            self.assertAlmostEqual(g1, 3.0, places=10)

    def test_conductance_exponential_decay_per_receptor(self):
        r"""Each receptor conductance should decay with its own time constant."""
        with brainstate.environ.context(dt=self.dt):
            tau0, tau1 = 4.0, 8.0
            neuron = gif_cond_exp_multisynapse(
                1,
                tau_syn=(tau0, tau1),
                E_rev=(0.0, -85.0),
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Inject conductances
            self._step(neuron, 0, dg_values=[(0, 10.0), (1, 8.0)])

            # Run for 10 steps (1 ms) and check decay
            for k in range(1, 11):
                self._step(neuron, k)

            g0 = float((neuron.g[0].value / u.nS)[0])
            g1 = float((neuron.g[1].value / u.nS)[0])

            expected_g0 = 10.0 * math.exp(-1.0 / tau0)
            expected_g1 = 8.0 * math.exp(-1.0 / tau1)
            self.assertAlmostEqual(g0, expected_g0, places=3)
            self.assertAlmostEqual(g1, expected_g1, places=3)

    def test_excitatory_depolarizes_inhibitory_hyperpolarizes(self):
        r"""Excitatory reversal (0mV) should depolarize, inhibitory (-85mV) should hyperpolarize."""
        with brainstate.environ.context(dt=self.dt):
            base = gif_cond_exp_multisynapse(
                1, tau_syn=(4.0, 8.0), E_rev=(0.0, -85.0),
                lambda_0=0.0, V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            exc = gif_cond_exp_multisynapse(
                1, tau_syn=(4.0, 8.0), E_rev=(0.0, -85.0),
                lambda_0=0.0, V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            inh = gif_cond_exp_multisynapse(
                1, tau_syn=(4.0, 8.0), E_rev=(0.0, -85.0),
                lambda_0=0.0, V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            base.init_state()
            exc.init_state()
            inh.init_state()

            # Apply conductances
            self._step(base, 0)
            self._step(exc, 0, dg_values=[(0, 5.0)])   # excitatory receptor
            self._step(inh, 0, dg_values=[(1, 5.0)])   # inhibitory receptor

            # Let dynamics evolve
            self._step(base, 1)
            self._step(exc, 1)
            self._step(inh, 1)

            v_base = float((base.V.value / u.mV)[0])
            v_exc = float((exc.V.value / u.mV)[0])
            v_inh = float((inh.V.value / u.mV)[0])

            # E_rev[0]=0 mV > -70 mV = V, so excitatory drives V up
            self.assertTrue(v_exc > v_base, f"Exc {v_exc} should be > base {v_base}")
            # E_rev[1]=-85 mV < -70 mV = V, so inhibitory drives V down
            self.assertTrue(v_inh < v_base, f"Inh {v_inh} should be < base {v_base}")

    def test_three_receptor_ports(self):
        r"""Test with three receptor ports with different reversal potentials."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
                1,
                tau_syn=(2.0, 4.0, 8.0),
                E_rev=(0.0, -85.0, -65.0),
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Add conductance to all three ports
            self._step(neuron, 0, dg_values=[(0, 3.0), (1, 2.0), (2, 1.0)])

            self.assertAlmostEqual(float((neuron.g[0].value / u.nS)[0]), 3.0, places=10)
            self.assertAlmostEqual(float((neuron.g[1].value / u.nS)[0]), 2.0, places=10)
            self.assertAlmostEqual(float((neuron.g[2].value / u.nS)[0]), 1.0, places=10)

            # After 10 steps, each should decay with its own tau
            for k in range(1, 11):
                self._step(neuron, k)

            self.assertAlmostEqual(
                float((neuron.g[0].value / u.nS)[0]),
                3.0 * math.exp(-1.0 / 2.0), places=3
            )
            self.assertAlmostEqual(
                float((neuron.g[1].value / u.nS)[0]),
                2.0 * math.exp(-1.0 / 4.0), places=3
            )
            self.assertAlmostEqual(
                float((neuron.g[2].value / u.nS)[0]),
                1.0 * math.exp(-1.0 / 8.0), places=3
            )


class TestGIFCondExpMultisynRefractoryBehavior(unittest.TestCase):
    r"""Test refractory period mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for port, val in dg_values:
                neuron.add_delta_input(f'receptor_{port}_{k}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_refractory_clamps_voltage_to_V_reset(self):
        r"""During refractory period, V should stay at V_reset."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
                1,
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=1.0 * u.ms,
                V_reset=-55.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            spk0 = self._step(neuron, 0)
            self.assertTrue(float(spk0[0]) > 0, "Should spike on step 0 with lambda_0=1e12")

            # Steps 1-9 should be refractory
            for k in range(1, 10):
                self._step(neuron, k)
                v = float((neuron.V.value / u.mV)[0])
                self.assertAlmostEqual(v, -55.0, places=6,
                                       msg=f"V should be V_reset during refractory at step {k}")

    def test_refractory_count_matches_t_ref(self):
        r"""Refractory counter should match ceil(t_ref / dt)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
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

    def test_conductances_continue_decaying_during_refractory(self):
        r"""Conductances should keep decaying during refractory period."""
        with brainstate.environ.context(dt=self.dt):
            tau0 = 4.0
            neuron = gif_cond_exp_multisynapse(
                1,
                tau_syn=(tau0,), E_rev=(0.0,),
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=2.0 * u.ms,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            # Spike + add conductance
            self._step(neuron, 0, dg_values=[(0, 10.0)])
            g0_after_spike = float((neuron.g[0].value / u.nS)[0])

            # During refractory, conductances still decay
            for k in range(1, 11):
                self._step(neuron, k)

            g0_after = float((neuron.g[0].value / u.nS)[0])
            expected = g0_after_spike * math.exp(-1.0 / tau0)
            self.assertAlmostEqual(g0_after, expected, places=3)


class TestGIFCondExpMultisynAdaptation(unittest.TestCase):
    r"""Test stc and sfa adaptation mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for port, val in dg_values:
                neuron.add_delta_input(f'receptor_{port}_{k}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_stc_elements_decay_exponentially(self):
        r"""STC elements should decay by exp(-dt/tau) each step."""
        tau_stc = [10.0, 20.0]
        q_stc = [5.0, -2.0]

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
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
            neuron.lambda_0 = 0.0  # turn off spiking

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
            neuron = gif_cond_exp_multisynapse(
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
            neuron = gif_cond_exp_multisynapse(
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

            self.assertAlmostEqual(neuron._sfa_val[0], -35.0)

            self._step(neuron, 0)
            neuron.lambda_0 = 0.0
            self._step(neuron, 1)
            # sfa = V_T_star + sfa_elems[0] = -35 + 10 = -25
            self.assertAlmostEqual(neuron._sfa_val[0], -25.0, places=3)


class TestGIFCondExpMultisynStochasticSpiking(unittest.TestCase):
    r"""Test stochastic spike generation mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for port, val in dg_values:
                neuron.add_delta_input(f'receptor_{port}_{k}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_no_spikes_with_zero_lambda(self):
        r"""With lambda_0=0, no spikes should ever occur."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
                1,
                lambda_0=0.0,
                I_e=1000.0 * u.pA,
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
            neuron = gif_cond_exp_multisynapse(
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
            n1 = gif_cond_exp_multisynapse(
                1, lambda_0=100.0, rng_key=key,
                tau_syn=(4.0, 8.0), E_rev=(0.0, -85.0),
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            n2 = gif_cond_exp_multisynapse(
                1, lambda_0=100.0, rng_key=key,
                tau_syn=(4.0, 8.0), E_rev=(0.0, -85.0),
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            n1.init_state()
            n2.init_state()

            for k in range(50):
                s1 = self._step(n1, k)
                s2 = self._step(n2, k)
                self.assertEqual(float(s1[0]), float(s2[0]),
                                 f"Spike mismatch at step {k} with identical RNG")


class TestGIFCondExpMultisynReferenceTrace(unittest.TestCase):
    r"""Compare full simulation traces against standalone reference implementation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_val = 0.1
        self.dt = self.dt_val * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for port, val in dg_values:
                neuron.add_delta_input(f'receptor_{port}_{k}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_subthreshold_trace_matches_reference(self):
        r"""Multi-step subthreshold trace with two receptors should match reference."""
        tau_syn = [4.0, 8.0]
        E_rev = [0.0, -85.0]
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'I_e': 50.0, 't_ref': 4.0,
        }

        n_steps = 20
        # dg_seq: list of (port, dg_nS) tuples per step
        dg_seq = [[] for _ in range(n_steps)]
        dg_seq[0] = [(0, 5.0), (1, 3.0)]
        dg_seq[5] = [(0, 2.0)]
        dg_seq[10] = [(1, 4.0)]
        i_stim_seq = [0.0] * n_steps
        i_stim_seq[0] = 30.0
        i_stim_seq[3] = -10.0

        # Reference (no spiking)
        v_ref, g_ref, _, _, _ = _run_nest_ref_multi(
            n_steps, self.dt_val, p,
            i_stim_seq, dg_seq, [1.0] * n_steps,
            tau_stc=[], q_stc=[], tau_sfa=[], q_sfa=[],
            lambda_0=0.001, Delta_V=0.5, V_T_star=-35.0,
            tau_syn=tau_syn, E_rev=E_rev,
        )

        # Model
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                tau_syn=tau_syn,
                E_rev=E_rev,
                I_e=p['I_e'] * u.pA,
                t_ref=p['t_ref'] * u.ms,
                lambda_0=0.0,
                V_T_star=-35.0 * u.mV,
                Delta_V=0.5 * u.mV,
                V_initializer=braintools.init.Constant(p['E_L'] * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            v_model, g0_model, g1_model = [], [], []
            for k in range(n_steps):
                x_pA = i_stim_seq[k]
                dg_vals = [(port, val) for port, val in dg_seq[k]] if dg_seq[k] else None
                self._step(neuron, k, x=x_pA * u.pA, dg_values=dg_vals)
                v_model.append(float((neuron.V.value / u.mV)[0]))
                g0_model.append(float((neuron.g[0].value / u.nS)[0]))
                g1_model.append(float((neuron.g[1].value / u.nS)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")
            self.assertAlmostEqual(g0_model[k], g_ref[0][k], places=5,
                                   msg=f"g[0] mismatch at step {k}")
            self.assertAlmostEqual(g1_model[k], g_ref[1][k], places=5,
                                   msg=f"g[1] mismatch at step {k}")

    def test_full_trace_with_adaptation_and_spiking(self):
        r"""Full trace with adaptation and controlled stochastic spiking matches reference."""
        tau_syn = [4.0, 8.0]
        E_rev = [0.0, -85.0]
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'I_e': 0.0, 't_ref': 4.0,
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
        v_ref, g_ref, spk_ref, stc_ref, sfa_ref = _run_nest_ref_multi(
            n_steps, self.dt_val, p,
            i_stim_seq, dg_seq, rand_vals,
            tau_stc=tau_stc, q_stc=q_stc,
            tau_sfa=tau_sfa, q_sfa=q_sfa,
            lambda_0=lambda_0, Delta_V=Delta_V, V_T_star=V_T_star,
            tau_syn=tau_syn, E_rev=E_rev,
        )

        # Model
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                tau_syn=tau_syn,
                E_rev=E_rev,
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

            v_model, g0_model, g1_model = [], [], []
            for k in range(n_steps):
                self._step(neuron, k, x=0.0 * u.pA)
                v_model.append(float((neuron.V.value / u.mV)[0]))
                g0_model.append(float((neuron.g[0].value / u.nS)[0]))
                g1_model.append(float((neuron.g[1].value / u.nS)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")
            self.assertAlmostEqual(g0_model[k], g_ref[0][k], places=5,
                                   msg=f"g[0] mismatch at step {k}")
            self.assertAlmostEqual(g1_model[k], g_ref[1][k], places=5,
                                   msg=f"g[1] mismatch at step {k}")

    def test_trace_with_multiple_receptor_inputs_and_adaptation(self):
        r"""Three receptors with staggered inputs, adaptation active."""
        tau_syn = [2.0, 4.0, 8.0]
        E_rev = [0.0, -85.0, -65.0]
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'I_e': 100.0, 't_ref': 4.0,
        }
        tau_stc = [10.0]
        q_stc = [20.0]
        tau_sfa = [50.0, 200.0]
        q_sfa = [5.0, 3.0]
        lambda_0 = 0.001
        Delta_V = 0.5
        V_T_star = -35.0

        n_steps = 50
        dg_seq = [[] for _ in range(n_steps)]
        dg_seq[5] = [(0, 10.0)]
        dg_seq[10] = [(1, 8.0)]
        dg_seq[15] = [(2, 5.0)]
        dg_seq[20] = [(0, 7.0), (1, 3.0)]
        i_stim_seq = [0.0] * n_steps

        key = jax.random.PRNGKey(42)
        rand_vals = []
        for k in range(n_steps):
            key, subkey = jax.random.split(key)
            rand_vals.append(float(jax.random.uniform(subkey, shape=(1,))[0]))

        v_ref, g_ref, spk_ref, _, _ = _run_nest_ref_multi(
            n_steps, self.dt_val, p,
            i_stim_seq, dg_seq, rand_vals,
            tau_stc=tau_stc, q_stc=q_stc,
            tau_sfa=tau_sfa, q_sfa=q_sfa,
            lambda_0=lambda_0, Delta_V=Delta_V, V_T_star=V_T_star,
            tau_syn=tau_syn, E_rev=E_rev,
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                tau_syn=tau_syn,
                E_rev=E_rev,
                I_e=p['I_e'] * u.pA,
                t_ref=p['t_ref'] * u.ms,
                lambda_0=lambda_0 * 1000.0,
                Delta_V=Delta_V * u.mV,
                V_T_star=V_T_star * u.mV,
                tau_stc=tau_stc, q_stc=q_stc,
                tau_sfa=tau_sfa, q_sfa=q_sfa,
                V_initializer=braintools.init.Constant(p['E_L'] * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            v_model = []
            g_model = [[] for _ in range(3)]
            for k in range(n_steps):
                dg_vals = [(port, val) for port, val in dg_seq[k]] if dg_seq[k] else None
                self._step(neuron, k, x=0.0 * u.pA, dg_values=dg_vals)
                v_model.append(float((neuron.V.value / u.mV)[0]))
                for port in range(3):
                    g_model[port].append(float((neuron.g[port].value / u.nS)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")
            for port in range(3):
                self.assertAlmostEqual(g_model[port][k], g_ref[port][k], places=5,
                                       msg=f"g[{port}] mismatch at step {k}")


class TestGIFCondExpMultisynUpdateOrder(unittest.TestCase):
    r"""Test that the update order matches NEST exactly."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for port, val in dg_values:
                neuron.add_delta_input(f'receptor_{port}_{k}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_stc_computed_before_ode_integration(self):
        r"""STC current should be computed BEFORE ODE integration, matching NEST order."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
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

            self._step(neuron, 0)  # spike adds q_stc=100

            neuron.lambda_0 = 0.0
            self._step(neuron, 1)
            stc_val = neuron._stc_val[0]
            self.assertAlmostEqual(stc_val, 100.0, places=3,
                                   msg="STC should be 100 nA on step after spike")

    def test_conductance_jumps_after_ode_integration(self):
        r"""Conductance jumps should be applied AFTER ODE integration."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_cond_exp_multisynapse(
                1,
                tau_syn=(4.0, 8.0),
                E_rev=(0.0, -85.0),
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, dg_values=[(0, 10.0), (1, 5.0)])

            # g should be exactly the jump values (no prior conductance to decay)
            g0 = float((neuron.g[0].value / u.nS)[0])
            g1 = float((neuron.g[1].value / u.nS)[0])
            self.assertAlmostEqual(g0, 10.0, places=10)
            self.assertAlmostEqual(g1, 5.0, places=10)


class TestGIFCondExpMultisynResetState(unittest.TestCase):
    r"""Test that reset_state properly reinitializes all state."""

    def test_reset_restores_initial_state(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = gif_cond_exp_multisynapse(
                1,
                tau_syn=(4.0, 8.0),
                E_rev=(0.0, -85.0),
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                tau_stc=[10.0],
                q_stc=[5.0],
                tau_sfa=[100.0],
                q_sfa=[10.0],
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Run several steps to modify state
            for k in range(20):
                neuron.add_delta_input(f'receptor_0_{k}', 1.0 * u.nS)
                with brainstate.environ.context(t=k * 0.1 * u.ms):
                    neuron.update()

            # Reset state
            neuron.reset_state()

            # Verify state is back to initial values
            self.assertTrue(u.math.allclose(neuron.V.value, -70.0 * u.mV))
            self.assertTrue(u.math.allclose(neuron.g[0].value, 0.0 * u.nS))
            self.assertTrue(u.math.allclose(neuron.g[1].value, 0.0 * u.nS))
            self.assertEqual(int(neuron.refractory_step_count.value[0]), 0)
            np.testing.assert_allclose(neuron._stc_val, 0.0)
            np.testing.assert_allclose(neuron._sfa_val, -1000.0)  # V_T_star


if __name__ == '__main__':
    unittest.main()
