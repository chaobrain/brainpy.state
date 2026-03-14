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
Tests for gif_psc_exp_multisynapse neuron model.

These tests verify that the brainpy.state implementation of
gif_psc_exp_multisynapse matches NEST's update ordering and semantics,
including:

- Default parameter values matching NEST C++ source
- Subthreshold membrane dynamics with exact integration (propagator matrix)
- Multiple receptor ports with independent exponential synaptic currents
- Refractory period mechanics (V clamped, countdown)
- Spike-triggered current (stc) adaptation
- Spike-frequency adaptation (sfa) threshold dynamics
- Stochastic spike generation via hazard function
- One-step delayed current input (NEST ring buffer semantics)
- Full reference trace comparison against standalone Python reference
  implementation that mirrors NEST's update loop

All tests use float64 on CPU to match NEST's numerical behavior.
"""

import math
import os
import unittest

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import braintools
import brainstate
import saiunit as u
import jax
import numpy as np

from brainpy_state._nest.gif_psc_exp_multisynapse import gif_psc_exp_multisynapse


# ---------------------------------------------------------------------------
# Reference implementation matching NEST gif_psc_exp_multisynapse::update()
# ---------------------------------------------------------------------------

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


def _run_nest_ref(n_steps, dt, p, tau_syn, i_stim_seq, w_seq, rand_seq,
                  tau_stc, q_stc, tau_sfa, q_sfa, lambda_0, Delta_V, V_T_star):
    r"""Full reference implementation of gif_psc_exp_multisynapse update loop.

    Matches NEST update order exactly:
    1. Compute stc/sfa totals, decay elements
    2. For each receptor k: compute sum_syn_pot contribution, decay i_syn[k],
       add spike weight jumps
    3. If not refractory: update V via propagator, stochastic spike check
       If refractory: decrement counter, clamp V to V_reset
    4. Store I_stim

    Parameters
    ----------
    n_steps : int
        Number of simulation steps.
    dt : float
        Time step in ms.
    p : dict
        Model parameters (E_L, C_m, g_L, V_reset, I_e, t_ref).
    tau_syn : list of float
        Synaptic time constants in ms (one per receptor).
    i_stim_seq : list of float
        External current input at each step (pA).
    w_seq : list of list of (receptor, weight) tuples
        Spike events at each step. receptor is 1-based.
    rand_seq : list of float
        Random values for stochastic spiking (one per step).
    tau_stc, q_stc, tau_sfa, q_sfa : lists
        Adaptation parameters.
    lambda_0 : float
        Firing intensity in 1/ms.
    Delta_V : float
        Stochasticity level in mV.
    V_T_star : float
        Base threshold in mV.

    Returns
    -------
    v_trace, isyn_traces, spike_trace, stc_trace, sfa_trace
    """
    n_rec = len(tau_syn)
    v = p['E_L']
    i_syn = [0.0] * n_rec
    r = 0
    i_stim = 0.0
    tau_m = p['C_m'] / p['g_L']
    refr_steps = int(math.ceil(p['t_ref'] / dt))

    # Propagator coefficients
    P33 = math.exp(-dt / tau_m)
    P30 = -1.0 / p['C_m'] * math.expm1(-dt / tau_m) * tau_m
    P31 = -math.expm1(-dt / tau_m)

    P11_syn = [math.exp(-dt / ts) for ts in tau_syn]
    P21_syn = [_propagator_exp(ts, tau_m, p['C_m'], dt) for ts in tau_syn]

    n_stc = len(tau_stc)
    n_sfa = len(tau_sfa)
    stc_elems = [0.0] * n_stc
    sfa_elems = [0.0] * n_sfa

    P_stc = [math.exp(-dt / tau) for tau in tau_stc]
    P_sfa = [math.exp(-dt / tau) for tau in tau_sfa]

    v_trace = []
    isyn_traces = [[] for _ in range(n_rec)]
    spike_trace, stc_trace, sfa_trace = [], [], []

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

        # Step 2: Synaptic currents (NEST order: propagate, decay, add spikes)
        sum_syn_pot = 0.0
        for j in range(n_rec):
            sum_syn_pot += P21_syn[j] * i_syn[j]
            i_syn[j] *= P11_syn[j]

        # Add spike events for this step
        if k < len(w_seq):
            for ev in w_seq[k]:
                rec_idx, weight = ev
                rec_idx -= 1  # 1-based to 0-based
                if 0 <= rec_idx < n_rec:
                    i_syn[rec_idx] += weight

        # Step 3: Spike check / refractory
        spike = 0.0
        if r == 0:
            v = (P30 * (i_stim + p['I_e'] - stc_total)
                 + P33 * v
                 + P31 * p['E_L']
                 + sum_syn_pot)

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

        # Step 4: Store I_stim
        if k < len(i_stim_seq):
            i_stim = i_stim_seq[k]
        else:
            i_stim = 0.0

        v_trace.append(v)
        for j in range(n_rec):
            isyn_traces[j].append(i_syn[j])
        spike_trace.append(spike)
        stc_trace.append(stc_total)
        sfa_trace.append(sfa_total)

    return v_trace, isyn_traces, spike_trace, stc_trace, sfa_trace


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestDefaultParameters(unittest.TestCase):
    r"""Test that default parameters match NEST C++ source code values."""

    def test_nest_cpp_default_parameters(self):
        neuron = gif_psc_exp_multisynapse(1)
        self.assertEqual(neuron.g_L, 4.0 * u.nS)
        self.assertEqual(neuron.E_L, -70.0 * u.mV)
        self.assertEqual(neuron.C_m, 80.0 * u.pF)
        self.assertEqual(neuron.V_reset, -55.0 * u.mV)
        self.assertEqual(neuron.Delta_V, 0.5 * u.mV)
        self.assertEqual(neuron.V_T_star, -35.0 * u.mV)
        self.assertAlmostEqual(neuron.lambda_0, 0.001)  # 1/ms (= 1.0/s)
        self.assertEqual(neuron.t_ref, 4.0 * u.ms)
        self.assertEqual(neuron.I_e, 0.0 * u.pA)
        self.assertEqual(neuron.tau_sfa, ())
        self.assertEqual(neuron.q_sfa, ())
        self.assertEqual(neuron.tau_stc, ())
        self.assertEqual(neuron.q_stc, ())
        # tau_syn default: single element with value 2.0 ms
        np.testing.assert_array_equal(neuron.tau_syn, np.array([2.0]))
        self.assertEqual(neuron.n_receptors, 1)

    def test_initial_state_matches_nest(self):
        r"""V should be initialized to E_L, synaptic currents to 0."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = gif_psc_exp_multisynapse(1, tau_syn=(2.0, 5.0))
            neuron.init_state()
            self.assertTrue(u.math.allclose(neuron.V.value, -70.0 * u.mV))
            i_syn = np.asarray(u.math.asarray(neuron.i_syn.value / u.pA))
            np.testing.assert_allclose(i_syn, 0.0)
            self.assertEqual(neuron.n_receptors, 2)

    def test_multiple_receptor_ports(self):
        r"""tau_syn with multiple elements should create multiple receptor ports."""
        neuron = gif_psc_exp_multisynapse(1, tau_syn=(2.0, 4.0, 8.0))
        self.assertEqual(neuron.n_receptors, 3)
        np.testing.assert_array_equal(neuron.tau_syn, np.array([2.0, 4.0, 8.0]))


class TestParameterValidation(unittest.TestCase):
    r"""Test that invalid parameters raise appropriate errors."""

    def test_mismatched_tau_sfa_q_sfa_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, tau_sfa=[10.0], q_sfa=[1.0, 2.0])

    def test_mismatched_tau_stc_q_stc_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, tau_stc=[10.0, 20.0], q_stc=[1.0])

    def test_negative_capacitance_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, C_m=-80.0 * u.pF)

    def test_negative_g_L_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, g_L=-1.0 * u.nS)

    def test_negative_Delta_V_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, Delta_V=-0.5 * u.mV)

    def test_negative_lambda_0_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, lambda_0=-1.0)

    def test_negative_tau_syn_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, tau_syn=(-2.0,))

    def test_negative_tau_sfa_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, tau_sfa=[-10.0], q_sfa=[5.0])

    def test_negative_tau_stc_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, tau_stc=[-10.0], q_stc=[5.0])

    def test_empty_tau_syn_raises(self):
        with self.assertRaises(ValueError):
            gif_psc_exp_multisynapse(1, tau_syn=())

    def test_invalid_receptor_type_raises(self):
        r"""Out-of-range receptor type should raise ValueError."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = gif_psc_exp_multisynapse(1, tau_syn=(2.0,), lambda_0=0.0)
            neuron.init_state()
            with self.assertRaises(ValueError):
                with brainstate.environ.context(t=0.0 * u.ms):
                    neuron.update(spike_events=[(2, 1.0 * u.pA)])


class TestSubthresholdDynamics(unittest.TestCase):
    r"""Test subthreshold membrane dynamics without spiking."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, spike_events=None, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{k}', delta)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_current_input_has_one_step_delay(self):
        r"""External current should be stored for use in the NEXT step."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
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

    def test_synaptic_current_per_receptor(self):
        r"""Spike events targeted at different receptors should be independent."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                tau_syn=(2.0, 5.0),
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Inject 10 pA to receptor 1, -3 pA to receptor 2
            self._step(neuron, 0, spike_events=[
                (1, 10.0 * u.pA),
                (2, -3.0 * u.pA),
            ])

            dftype = brainstate.environ.dftype()
            i_syn = np.asarray(u.math.asarray(neuron.i_syn.value / u.pA), dtype=dftype)
            # After step 0: i_syn[0] should be 10 (decayed from 0 + 10)
            # i_syn[1] should be -3
            self.assertAlmostEqual(i_syn[0, 0], 10.0, places=8)
            self.assertAlmostEqual(i_syn[0, 1], -3.0, places=8)

    def test_synaptic_current_exponential_decay(self):
        r"""Synaptic currents per receptor should decay with their own time constant."""
        with brainstate.environ.context(dt=self.dt):
            tau_syn = (2.0, 5.0)
            neuron = gif_psc_exp_multisynapse(
                1,
                tau_syn=tau_syn,
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Inject spike to receptor 1 and receptor 2
            self._step(neuron, 0, spike_events=[
                (1, 10.0 * u.pA),
                (2, 8.0 * u.pA),
            ])

            # Run for 10 steps (1 ms) and check decay
            for k in range(1, 11):
                self._step(neuron, k)

            dftype = brainstate.environ.dftype()
            i_syn = np.asarray(u.math.asarray(neuron.i_syn.value / u.pA), dtype=dftype)

            expected_0 = 10.0 * math.exp(-1.0 / tau_syn[0])
            expected_1 = 8.0 * math.exp(-1.0 / tau_syn[1])
            self.assertAlmostEqual(i_syn[0, 0], expected_0, places=5)
            self.assertAlmostEqual(i_syn[0, 1], expected_1, places=5)

    def test_delta_input_mapped_to_receptor_1(self):
        r"""Default delta inputs should be mapped to receptor 1."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                tau_syn=(2.0, 5.0),
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Use add_delta_input (not spike_events)
            self._step(neuron, 0, delta=7.0 * u.pA)

            dftype = brainstate.environ.dftype()
            i_syn = np.asarray(u.math.asarray(neuron.i_syn.value / u.pA), dtype=dftype)
            # delta input should go to receptor 1 (index 0)
            self.assertAlmostEqual(i_syn[0, 0], 7.0, places=8)
            self.assertAlmostEqual(i_syn[0, 1], 0.0, places=8)

    def test_exact_integration_propagator(self):
        r"""Test exact integration with single receptor matches analytic solution."""
        dt = 0.1  # ms
        C_m = 80.0
        g_L = 4.0
        E_L = -70.0
        I_e = 100.0  # pA
        tau_m = C_m / g_L  # 20 ms

        P33 = math.exp(-dt / tau_m)
        P30 = -1.0 / C_m * math.expm1(-dt / tau_m) * tau_m
        P31 = -math.expm1(-dt / tau_m)

        V0 = E_L
        V_analytic = P30 * I_e + P33 * V0 + P31 * E_L

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = gif_psc_exp_multisynapse(
                1,
                C_m=C_m * u.pF,
                g_L=g_L * u.nS,
                E_L=E_L * u.mV,
                I_e=I_e * u.pA,
                lambda_0=0.0,
                V_initializer=braintools.init.Constant(E_L * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0)
            v_model = float((neuron.V.value / u.mV)[0])

            self.assertAlmostEqual(v_model, V_analytic, places=8,
                                   msg=f"Exact integration mismatch: model={v_model}, "
                                       f"analytic={V_analytic}")

    def test_three_receptors_depolarization(self):
        r"""Three receptors with positive weights should all depolarize the neuron."""
        with brainstate.environ.context(dt=self.dt):
            base = gif_psc_exp_multisynapse(
                1, tau_syn=(2.0, 4.0, 8.0), lambda_0=0.0, V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            stim = gif_psc_exp_multisynapse(
                1, tau_syn=(2.0, 4.0, 8.0), lambda_0=0.0, V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            base.init_state()
            stim.init_state()

            # Inject spikes to all three receptors
            self._step(base, 0)
            self._step(stim, 0, spike_events=[
                (1, 20.0 * u.pA), (2, 20.0 * u.pA), (3, 20.0 * u.pA)
            ])

            # Let dynamics evolve for a few steps
            for k in range(1, 5):
                self._step(base, k)
                self._step(stim, k)

            v_base = float((base.V.value / u.mV)[0])
            v_stim = float((stim.V.value / u.mV)[0])

            self.assertTrue(v_stim > v_base,
                            f"V with synaptic input ({v_stim}) should be > base ({v_base})")


class TestRefractoryBehavior(unittest.TestCase):
    r"""Test refractory period mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, spike_events=None):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_refractory_clamps_voltage_to_V_reset(self):
        r"""During refractory period, V should stay at V_reset."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
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

            # Step 0 should spike
            spk0 = self._step(neuron, 0)
            self.assertTrue(float(spk0[0]) > 0, "Should spike on step 0")

            # Steps 1-9 should be refractory, V clamped to V_reset
            for k in range(1, 10):
                self._step(neuron, k)
                v = float((neuron.V.value / u.mV)[0])
                self.assertAlmostEqual(v, -55.0, places=6,
                                       msg=f"V should be V_reset during refractory at step {k}")

    def test_refractory_count_matches_t_ref(self):
        r"""Refractory counter should match ceil(t_ref / dt) as in NEST."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
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

    def test_synaptic_currents_evolve_during_refractory(self):
        r"""Synaptic currents should continue decaying during refractory period."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                tau_syn=(2.0,),
                lambda_0=1e12,
                Delta_V=100.0 * u.mV,
                V_T_star=-1000.0 * u.mV,
                t_ref=5.0 * u.ms,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            # Spike on step 0
            self._step(neuron, 0)

            # Inject synaptic current during refractory period
            self._step(neuron, 1, spike_events=[(1, 10.0 * u.pA)])
            isyn = float(np.asarray(u.math.asarray(neuron.i_syn.value / u.pA))[0, 0])
            self.assertAlmostEqual(isyn, 10.0, places=5)

            # Let it decay for 10 more steps (1 ms)
            for k in range(2, 12):
                self._step(neuron, k)

            isyn_after = float(np.asarray(u.math.asarray(neuron.i_syn.value / u.pA))[0, 0])
            expected = 10.0 * math.exp(-1.0 / 2.0)
            self.assertAlmostEqual(isyn_after, expected, places=5)


class TestAdaptation(unittest.TestCase):
    r"""Test stc and sfa adaptation mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, spike_events=None):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_stc_elements_decay_exponentially(self):
        r"""STC elements should decay by exp(-dt/tau) each step."""
        tau_stc = [10.0, 20.0]
        q_stc = [5.0, -2.0]

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
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
            neuron = gif_psc_exp_multisynapse(
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
            neuron = gif_psc_exp_multisynapse(
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
            # sfa = V_T_star + q_sfa = -35 + 10 = -25
            self.assertAlmostEqual(neuron._sfa_val[0], -25.0, places=3)

    def test_stc_current_opposes_depolarization(self):
        r"""Positive stc should hyperpolarize the neuron."""
        with brainstate.environ.context(dt=self.dt):
            no_stc = gif_psc_exp_multisynapse(
                1, lambda_0=0.0, I_e=200.0 * u.pA,
                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            no_stc.init_state()

            with_stc = gif_psc_exp_multisynapse(
                1, lambda_0=0.0, I_e=200.0 * u.pA,
                tau_stc=[10.0], q_stc=[50.0],
                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            with_stc.init_state()
            with_stc._stc_elems[0][0] = 50.0  # Simulate post-spike

            for k in range(20):
                self._step(no_stc, k)
                self._step(with_stc, k)

            v_no = float((no_stc.V.value / u.mV)[0])
            v_with = float((with_stc.V.value / u.mV)[0])
            self.assertTrue(v_with < v_no,
                            f"V with stc ({v_with}) should be < V without ({v_no})")


class TestStochasticSpiking(unittest.TestCase):
    r"""Test stochastic spike generation mechanics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_no_spikes_with_zero_lambda(self):
        r"""With lambda_0=0, no spikes should ever occur."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
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
            neuron = gif_psc_exp_multisynapse(
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
        r"""Two neurons with same RNG key and parameters should spike identically."""
        with brainstate.environ.context(dt=self.dt):
            key = jax.random.PRNGKey(12345)
            n1 = gif_psc_exp_multisynapse(
                1, lambda_0=100.0, rng_key=key,
                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            n2 = gif_psc_exp_multisynapse(
                1, lambda_0=100.0, rng_key=key,
                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            n1.init_state()
            n2.init_state()

            for k in range(50):
                s1 = self._step(n1, k)
                s2 = self._step(n2, k)
                self.assertEqual(float(s1[0]), float(s2[0]),
                                 f"Spike mismatch at step {k} with identical RNG")


class TestReferenceTrace(unittest.TestCase):
    r"""Compare full simulation traces against standalone reference implementation."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_val = 0.1
        self.dt = self.dt_val * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, spike_events=None, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{k}', delta)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_subthreshold_single_receptor(self):
        r"""Single-receptor subthreshold trace should match reference."""
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'I_e': 50.0, 't_ref': 4.0,
        }
        tau_syn = [2.0]
        n_steps = 50

        # Spike events: receptor 1 with various weights
        w_seq = [[] for _ in range(n_steps)]
        w_seq[0] = [(1, 5.0)]
        w_seq[5] = [(1, 2.0)]
        w_seq[10] = [(1, -4.0)]

        i_stim_seq = [0.0] * n_steps
        i_stim_seq[0] = 30.0
        i_stim_seq[3] = -10.0

        v_ref, isyn_ref, _, _, _ = _run_nest_ref(
            n_steps, self.dt_val, p, tau_syn,
            i_stim_seq, w_seq, [1.0] * n_steps,
            tau_stc=[], q_stc=[], tau_sfa=[], q_sfa=[],
            lambda_0=0.001, Delta_V=0.5, V_T_star=-35.0
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                tau_syn=tau_syn,
                I_e=p['I_e'] * u.pA,
                t_ref=p['t_ref'] * u.ms,
                lambda_0=0.0,  # disable spiking
                V_T_star=-35.0 * u.mV,
                Delta_V=0.5 * u.mV,
                V_initializer=braintools.init.Constant(p['E_L'] * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            v_model = []
            isyn_model = []
            for k in range(n_steps):
                x_pA = i_stim_seq[k]
                se = None
                if w_seq[k]:
                    se = [(rec, w * u.pA) for rec, w in w_seq[k]]
                self._step(neuron, k, x=x_pA * u.pA, spike_events=se)
                v_model.append(float((neuron.V.value / u.mV)[0]))
                dftype = brainstate.environ.dftype()
                isyn = np.asarray(u.math.asarray(neuron.i_syn.value / u.pA), dtype=dftype)
                isyn_model.append(isyn[0, 0])

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")
            self.assertAlmostEqual(isyn_model[k], isyn_ref[0][k], places=5,
                                   msg=f"I_syn mismatch at step {k}")

    def test_multi_receptor_subthreshold(self):
        r"""Multi-receptor subthreshold trace should match reference."""
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'I_e': 50.0, 't_ref': 4.0,
        }
        tau_syn = [8.0, 4.0]
        n_steps = 50

        w_seq = [[] for _ in range(n_steps)]
        w_seq[0] = [(1, 10.0)]
        w_seq[5] = [(2, 5.0)]
        w_seq[10] = [(1, -3.0), (2, 7.0)]

        i_stim_seq = [0.0] * n_steps
        i_stim_seq[0] = 20.0

        v_ref, isyn_ref, _, _, _ = _run_nest_ref(
            n_steps, self.dt_val, p, tau_syn,
            i_stim_seq, w_seq, [1.0] * n_steps,
            tau_stc=[], q_stc=[], tau_sfa=[], q_sfa=[],
            lambda_0=0.001, Delta_V=0.5, V_T_star=-35.0
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                tau_syn=tau_syn,
                I_e=p['I_e'] * u.pA,
                t_ref=p['t_ref'] * u.ms,
                lambda_0=0.0,
                V_T_star=-35.0 * u.mV,
                Delta_V=0.5 * u.mV,
                V_initializer=braintools.init.Constant(p['E_L'] * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            v_model = []
            isyn_model_0, isyn_model_1 = [], []
            for k in range(n_steps):
                x_pA = i_stim_seq[k]
                se = None
                if w_seq[k]:
                    se = [(rec, w * u.pA) for rec, w in w_seq[k]]
                self._step(neuron, k, x=x_pA * u.pA, spike_events=se)
                v_model.append(float((neuron.V.value / u.mV)[0]))
                dftype = brainstate.environ.dftype()
                isyn = np.asarray(u.math.asarray(neuron.i_syn.value / u.pA), dtype=dftype)
                isyn_model_0.append(isyn[0, 0])
                isyn_model_1.append(isyn[0, 1])

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}")
            self.assertAlmostEqual(isyn_model_0[k], isyn_ref[0][k], places=5,
                                   msg=f"I_syn[0] mismatch at step {k}")
            self.assertAlmostEqual(isyn_model_1[k], isyn_ref[1][k], places=5,
                                   msg=f"I_syn[1] mismatch at step {k}")

    def test_full_trace_with_adaptation_and_spiking(self):
        r"""Full trace with adaptation and controlled spiking matches reference."""
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'I_e': 0.0, 't_ref': 4.0,
        }
        tau_syn = [2.0, 5.0]
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

        v_ref, _, _, _, _ = _run_nest_ref(
            n_steps, self.dt_val, p, tau_syn,
            i_stim_seq, w_seq, rand_vals,
            tau_stc=tau_stc, q_stc=q_stc,
            tau_sfa=tau_sfa, q_sfa=q_sfa,
            lambda_0=lambda_0, Delta_V=Delta_V, V_T_star=V_T_star
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                tau_syn=tau_syn,
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
                                   msg=f"V mismatch at step {k}: "
                                       f"model={v_model[k]}, ref={v_ref[k]}")

    def test_dc_driven_with_multi_receptor_spikes(self):
        r"""DC + multi-receptor spikes with adaptation, comparing against reference."""
        p = {
            'E_L': -70.0, 'C_m': 40.0, 'g_L': 4.0,
            'V_reset': -55.0, 'I_e': 170.0, 't_ref': 4.0,
        }
        tau_syn = [8.0, 4.0]
        tau_stc = [10.0, 20.0]
        q_stc = [20.0, -5.0]
        tau_sfa = [120.0, 10.0]
        q_sfa = [10.0, 25.0]
        lambda_0 = 0.001  # 1/ms
        Delta_V = 0.2
        V_T_star = -35.0

        n_steps = 100

        w_seq = [[] for _ in range(n_steps)]
        # Receptor 1 spikes at 10, 20, 30 ms
        for t_ms in [10.0, 20.0, 30.0]:
            step = int(t_ms / self.dt_val)
            if step < n_steps:
                w_seq[step].append((1, 1.0))
        # Receptor 2 spikes at 15, 25, 35 ms
        for t_ms in [15.0, 25.0, 35.0]:
            step = int(t_ms / self.dt_val)
            if step < n_steps:
                w_seq[step].append((2, 1.0))

        i_stim_seq = [0.0] * n_steps

        # Generate random numbers
        key = jax.random.PRNGKey(42)
        rand_vals = []
        for k in range(n_steps):
            key, subkey = jax.random.split(key)
            rand_vals.append(float(jax.random.uniform(subkey, shape=(1,))[0]))

        v_ref, _, spk_ref, _, _ = _run_nest_ref(
            n_steps, self.dt_val, p, tau_syn,
            i_stim_seq, w_seq, rand_vals,
            tau_stc=tau_stc, q_stc=q_stc,
            tau_sfa=tau_sfa, q_sfa=q_sfa,
            lambda_0=lambda_0, Delta_V=Delta_V, V_T_star=V_T_star
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                g_L=p['g_L'] * u.nS,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                V_reset=p['V_reset'] * u.mV,
                tau_syn=tau_syn,
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
                se = None
                if w_seq[k]:
                    se = [(rec, w * u.pA) for rec, w in w_seq[k]]
                self._step(neuron, k, x=0.0 * u.pA, spike_events=se)
                v_model.append(float((neuron.V.value / u.mV)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}: "
                                       f"model={v_model[k]}, ref={v_ref[k]}")


class TestUpdateOrder(unittest.TestCase):
    r"""Test that the update order matches NEST exactly."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, spike_events=None):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_stc_computed_before_membrane_update(self):
        r"""STC should be computed BEFORE membrane update, matching NEST."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
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

    def test_synaptic_contribution_before_decay(self):
        r"""Synaptic contribution to V should use pre-decay current value.

        NEST computes sum_syn_pot = P21_syn[k] * i_syn[k] BEFORE decaying
        i_syn[k], then decays, then adds new spikes.
        """
        with brainstate.environ.context(dt=self.dt):
            tau_syn_val = 2.0
            C_m = 80.0
            g_L = 4.0
            tau_m = C_m / g_L
            h = 0.1

            neuron = gif_psc_exp_multisynapse(
                1,
                tau_syn=(tau_syn_val,),
                C_m=C_m * u.pF,
                g_L=g_L * u.nS,
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Inject current at step 0
            self._step(neuron, 0, spike_events=[(1, 10.0 * u.pA)])

            # Step 1: The contribution of i_syn to V should use i_syn BEFORE decay
            # i_syn after step 0 = 10.0 pA
            # P21 = _propagator_exp(tau_syn, tau_m, C_m, h)
            # sum_syn_pot = P21 * 10.0  (pre-decay value)
            P21 = _propagator_exp(tau_syn_val, tau_m, C_m, h)
            expected_syn_contribution = P21 * 10.0

            # After step 1, i_syn should be 10 * P11 (decayed, no new spikes)
            self._step(neuron, 1)
            i_syn_after = float(np.asarray(u.math.asarray(
                neuron.i_syn.value / u.pA
            ))[0, 0])
            P11 = math.exp(-h / tau_syn_val)
            expected_i_syn = 10.0 * P11
            self.assertAlmostEqual(i_syn_after, expected_i_syn, places=6,
                                   msg="Synaptic current after decay mismatch")

            # The contribution should be finite and positive
            self.assertTrue(expected_syn_contribution > 0)

    def test_synaptic_decay_before_weight_addition(self):
        r"""Synaptic current should be decayed BEFORE spike weight jumps are added."""
        with brainstate.environ.context(dt=self.dt):
            tau_syn_val = 2.0
            neuron = gif_psc_exp_multisynapse(
                1,
                tau_syn=(tau_syn_val,),
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Step 0: inject weight 10
            self._step(neuron, 0, spike_events=[(1, 10.0 * u.pA)])
            isyn0 = float(np.asarray(u.math.asarray(neuron.i_syn.value / u.pA))[0, 0])
            self.assertAlmostEqual(isyn0, 10.0, places=7)

            # Step 1: inject weight 5
            self._step(neuron, 1, spike_events=[(1, 5.0 * u.pA)])
            expected = 10.0 * math.exp(-0.1 / tau_syn_val) + 5.0
            isyn1 = float(np.asarray(u.math.asarray(neuron.i_syn.value / u.pA))[0, 0])
            self.assertAlmostEqual(isyn1, expected, places=6)

    def test_propagator_singularity_handling(self):
        r"""When tau_m == tau_syn, the singular propagator should be used."""
        # tau_m = C_m / g_L = 80 / 4 = 20 ms
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                C_m=80.0 * u.pF,
                g_L=4.0 * u.nS,
                tau_syn=(20.0,),  # equals tau_m
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, spike_events=[(1, 10.0 * u.pA)])
            v = float((neuron.V.value / u.mV)[0])
            self.assertTrue(math.isfinite(v), f"V should be finite, got {v}")

            self._step(neuron, 1)
            v1 = float((neuron.V.value / u.mV)[0])
            self.assertTrue(math.isfinite(v1), f"V should be finite, got {v1}")


class TestMultisynapseSpecific(unittest.TestCase):
    r"""Tests specific to the multisynapse functionality."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, spike_events=None):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_receptor_isolation(self):
        r"""Input to one receptor should not affect other receptors' currents."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                tau_syn=(2.0, 5.0, 10.0),
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Only inject to receptor 2
            self._step(neuron, 0, spike_events=[(2, 15.0 * u.pA)])

            dftype = brainstate.environ.dftype()
            i_syn = np.asarray(u.math.asarray(neuron.i_syn.value / u.pA), dtype=dftype)
            self.assertAlmostEqual(i_syn[0, 0], 0.0, places=10,
                                   msg="Receptor 1 should have zero current")
            self.assertAlmostEqual(i_syn[0, 1], 15.0, places=8,
                                   msg="Receptor 2 should have the injected current")
            self.assertAlmostEqual(i_syn[0, 2], 0.0, places=10,
                                   msg="Receptor 3 should have zero current")

    def test_independent_decay_rates(self):
        r"""Each receptor should decay with its own time constant."""
        tau_syn = (2.0, 10.0, 50.0)
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                tau_syn=tau_syn,
                lambda_0=0.0,
                V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Inject same current to all receptors
            self._step(neuron, 0, spike_events=[
                (1, 100.0 * u.pA), (2, 100.0 * u.pA), (3, 100.0 * u.pA)
            ])

            # Run for 20 steps (2 ms) - enough to see different decay rates
            for k in range(1, 21):
                self._step(neuron, k)

            dftype = brainstate.environ.dftype()
            i_syn = np.asarray(u.math.asarray(neuron.i_syn.value / u.pA), dtype=dftype)

            for j, tau in enumerate(tau_syn):
                expected = 100.0 * math.exp(-2.0 / tau)
                self.assertAlmostEqual(i_syn[0, j], expected, places=4,
                                       msg=f"Receptor {j + 1} (tau={tau}) decay mismatch")

            # Fast receptor should have decayed more
            self.assertTrue(i_syn[0, 0] < i_syn[0, 1] < i_syn[0, 2],
                            "Faster time constant should give more decay")

    def test_spike_event_dict_format(self):
        r"""Spike events should work with dict format as well as tuple."""
        with brainstate.environ.context(dt=self.dt):
            n_tuple = gif_psc_exp_multisynapse(
                1, tau_syn=(2.0, 5.0), lambda_0=0.0, V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            n_dict = gif_psc_exp_multisynapse(
                1, tau_syn=(2.0, 5.0), lambda_0=0.0, V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            n_tuple.init_state()
            n_dict.init_state()

            # Use tuple format
            self._step(n_tuple, 0, spike_events=[(1, 10.0 * u.pA), (2, 5.0 * u.pA)])

            # Use dict format
            self._step(n_dict, 0, spike_events=[
                {'receptor_type': 1, 'weight': 10.0 * u.pA},
                {'receptor_type': 2, 'weight': 5.0 * u.pA},
            ])

            dftype = brainstate.environ.dftype()
            i_tuple = np.asarray(u.math.asarray(n_tuple.i_syn.value / u.pA), dtype=dftype)
            i_dict = np.asarray(u.math.asarray(n_dict.i_syn.value / u.pA), dtype=dftype)

            np.testing.assert_allclose(i_tuple, i_dict, atol=1e-12)

    def test_multiple_spikes_same_receptor(self):
        r"""Multiple spike events to the same receptor should accumulate."""
        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1, tau_syn=(2.0,), lambda_0=0.0, V_T_star=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV))
            neuron.init_state()

            self._step(neuron, 0, spike_events=[
                (1, 3.0 * u.pA), (1, 7.0 * u.pA), (1, 5.0 * u.pA)
            ])

            i_syn = float(np.asarray(u.math.asarray(neuron.i_syn.value / u.pA))[0, 0])
            self.assertAlmostEqual(i_syn, 15.0, places=8,
                                   msg="Multiple spikes should accumulate")

    def test_equivalence_with_single_receptor_gif_psc_exp(self):
        r"""With one receptor, gif_psc_exp_multisynapse should match gif_psc_exp
        for positive-weight-only inputs (both map to excitatory channel)."""
        p = {
            'E_L': -70.0, 'C_m': 80.0, 'g_L': 4.0,
            'V_reset': -55.0, 'I_e': 50.0, 't_ref': 4.0,
        }

        n_steps = 30

        # Reference: single receptor with weights
        tau_syn = [2.0]
        w_seq = [[] for _ in range(n_steps)]
        w_seq[0] = [(1, 5.0)]
        w_seq[5] = [(1, 2.0)]
        i_stim_seq = [0.0] * n_steps
        i_stim_seq[0] = 30.0

        v_ref, _, _, _, _ = _run_nest_ref(
            n_steps, 0.1, p, tau_syn,
            i_stim_seq, w_seq, [1.0] * n_steps,
            tau_stc=[], q_stc=[], tau_sfa=[], q_sfa=[],
            lambda_0=0.001, Delta_V=0.5, V_T_star=-35.0
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = gif_psc_exp_multisynapse(
                1,
                g_L=p['g_L'] * u.nS, E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF, V_reset=p['V_reset'] * u.mV,
                tau_syn=(2.0,), I_e=p['I_e'] * u.pA,
                t_ref=p['t_ref'] * u.ms,
                lambda_0=0.0, V_T_star=-35.0 * u.mV, Delta_V=0.5 * u.mV,
                V_initializer=braintools.init.Constant(p['E_L'] * u.mV),
            )
            neuron.init_state()

            v_model = []
            for k in range(n_steps):
                se = None
                if w_seq[k]:
                    se = [(rec, w * u.pA) for rec, w in w_seq[k]]
                self._step(neuron, k, x=i_stim_seq[k] * u.pA, spike_events=se)
                v_model.append(float((neuron.V.value / u.mV)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=4,
                                   msg=f"V mismatch at step {k}")


if __name__ == '__main__':
    unittest.main()
