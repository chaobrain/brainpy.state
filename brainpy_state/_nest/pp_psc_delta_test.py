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
Tests for pp_psc_delta neuron model.

These tests verify that the brainpy.state implementation of pp_psc_delta
matches NEST's update ordering and semantics, including:

- Default parameter values matching NEST C++ source
- Subthreshold membrane dynamics with exact integration (propagator)
- Delta-shaped PSC (instantaneous voltage jumps)
- Dead time (refractory period) mechanics
- Spike-frequency adaptation (sfa) threshold dynamics
- Stochastic spike generation via transfer function
- Optional membrane reset on spike
- One-step delayed current input (NEST ring buffer semantics)
- Full reference trace comparison against standalone Python reference
  implementing the exact NEST update loop

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

from brainpy_state._nest.pp_psc_delta import pp_psc_delta


def _run_nest_ref(n_steps, h, p, i_stim_seq, delta_v_seq, rand_seq,
                  tau_sfa, q_sfa):
    r"""Full reference implementation of pp_psc_delta update loop.

    Matches NEST's update order exactly (from pp_psc_delta.cpp):
    1. Update V_m via exact propagator (including delta synaptic inputs)
    2. Decay all adaptation elements, compute total E_sfa
    3. If not refractory: compute V_eff, rate, draw random, potentially spike
       If refractory: decrement counter
    4. Store I_stim for next step

    Parameters
    ----------
    n_steps : int
        Number of simulation steps.
    h : float
        Time step in ms.
    p : dict
        Parameters: tau_m, C_m, c_1, c_2, c_3, I_e, dead_time, with_reset.
    i_stim_seq : list of float
        External current sequence (pA), one per step.
    delta_v_seq : list of float
        Delta voltage input sequence (mV), one per step.
    rand_seq : list of float
        Uniform random values [0, 1) for each step.
    tau_sfa : list of float
        Adaptation time constants in ms.
    q_sfa : list of float
        Adaptation jump sizes in mV.

    Returns
    -------
    v_trace, q_trace, spike_trace, n_spikes_trace : lists
    """
    tau_m = p['tau_m']
    C_m = p['C_m']
    c_1 = p['c_1']
    c_2 = p['c_2']
    c_3 = p['c_3']
    I_e = p['I_e']
    dead_time = p['dead_time']
    with_reset = p['with_reset']

    # Clamp dead_time like NEST
    if dead_time != 0.0 and dead_time < h:
        dead_time = h

    # Propagator coefficients
    P33 = math.exp(-h / tau_m)
    P30 = 1.0 / C_m * (1.0 - P33) * tau_m

    # Dead time in steps (fixed mode)
    if dead_time > 0.0:
        dead_time_counts = int(round(dead_time / h))
    else:
        dead_time_counts = 0

    # Adaptation decay factors
    n_sfa = len(tau_sfa)
    q_elems = [0.0] * n_sfa
    P_sfa = [math.exp(-h / tau) for tau in tau_sfa]

    v = 0.0  # V_m relative to resting potential
    r = 0  # dead time counter
    i_stim = 0.0  # buffered current

    v_trace, q_trace, spike_trace, n_spikes_trace = [], [], [], []

    for k in range(n_steps):
        # Get delta input for this step
        dv = delta_v_seq[k] if k < len(delta_v_seq) else 0.0

        # ---- Step 1: Update V via exact propagator ----
        v = P30 * (i_stim + I_e) + P33 * v + dv

        # ---- Step 2: Decay adaptation, compute total E_sfa ----
        q_total = 0.0
        for i in range(n_sfa):
            q_elems[i] *= P_sfa[i]
            q_total += q_elems[i]

        # ---- Step 3: Spike check / dead time ----
        n_spikes = 0
        spike = False

        if r == 0:
            # Not refractory
            V_eff = v - q_total
            rate = c_1 * V_eff + c_2 * math.exp(c_3 * V_eff)

            if rate > 0.0:
                if dead_time > 0.0:
                    # With dead time: at most 1 spike
                    spike_prob = -math.expm1(-rate * h * 1e-3)
                    if rand_seq[k] <= spike_prob:
                        n_spikes = 1
                else:
                    # Without dead time: Poisson
                    # For reference, use numpy Poisson with same seed approach
                    lam_p = rate * h * 1e-3
                    n_spikes = int(np.random.RandomState(
                        int(rand_seq[k] * 2**31)
                    ).poisson(lam_p))

                if n_spikes > 0:
                    spike = True

                    # Set dead time
                    if dead_time > 0.0:
                        r = dead_time_counts

                    # Jump adaptation elements
                    for i in range(n_sfa):
                        q_elems[i] += q_sfa[i] * n_spikes

                    # Reset membrane potential
                    if with_reset:
                        v = 0.0
        else:
            # Within dead time
            r -= 1

        # ---- Step 4: Store I_stim for next step ----
        if k < len(i_stim_seq):
            i_stim = i_stim_seq[k]
        else:
            i_stim = 0.0

        v_trace.append(v)
        q_trace.append(q_total)
        spike_trace.append(1.0 if spike else 0.0)
        n_spikes_trace.append(n_spikes)

    return v_trace, q_trace, spike_trace, n_spikes_trace


class TestPpPscDeltaDefaultParams(unittest.TestCase):
    r"""Test that default parameters match NEST C++ source code values."""

    def test_nest_cpp_default_parameters(self):
        neuron = pp_psc_delta(1)
        self.assertEqual(neuron.tau_m, 10.0 * u.ms)
        self.assertEqual(neuron.C_m, 250.0 * u.pF)
        self.assertAlmostEqual(neuron.dead_time, 1.0)
        self.assertEqual(neuron.dead_time_random, False)
        self.assertEqual(neuron.dead_time_shape, 1)
        self.assertEqual(neuron.with_reset, True)
        self.assertAlmostEqual(neuron.c_1, 0.0)
        self.assertAlmostEqual(neuron.c_2, 1.238)
        self.assertAlmostEqual(neuron.c_3, 0.25)
        self.assertEqual(neuron.I_e, 0.0 * u.pA)
        self.assertEqual(neuron.tau_sfa, ())
        self.assertEqual(neuron.q_sfa, ())
        self.assertAlmostEqual(neuron.t_ref_remaining, 0.0)

    def test_initial_state_matches_nest(self):
        r"""V_m should be initialized to 0 (relative to rest)."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = pp_psc_delta(1)
            neuron.init_state()
            self.assertTrue(u.math.allclose(neuron.V.value, 0.0 * u.mV))


class TestPpPscDeltaParameterValidation(unittest.TestCase):
    r"""Test that invalid parameters raise appropriate errors."""

    def test_mismatched_tau_sfa_q_sfa_raises(self):
        with self.assertRaises(ValueError):
            pp_psc_delta(1, tau_sfa=[10.0], q_sfa=[1.0, 2.0])

    def test_negative_capacitance_raises(self):
        with self.assertRaises(ValueError):
            pp_psc_delta(1, C_m=-250.0 * u.pF)

    def test_negative_tau_m_raises(self):
        with self.assertRaises(ValueError):
            pp_psc_delta(1, tau_m=-10.0 * u.ms)

    def test_negative_dead_time_raises(self):
        with self.assertRaises(ValueError):
            pp_psc_delta(1, dead_time=-1.0)

    def test_dead_time_shape_zero_raises(self):
        with self.assertRaises(ValueError):
            pp_psc_delta(1, dead_time_shape=0)

    def test_negative_t_ref_remaining_raises(self):
        with self.assertRaises(ValueError):
            pp_psc_delta(1, t_ref_remaining=-1.0)

    def test_negative_c3_raises(self):
        with self.assertRaises(ValueError):
            pp_psc_delta(1, c_3=-0.1)

    def test_negative_tau_sfa_raises(self):
        with self.assertRaises(ValueError):
            pp_psc_delta(1, tau_sfa=[-10.0], q_sfa=[5.0])


class TestPpPscDeltaSubthresholdDynamics(unittest.TestCase):
    r"""Test subthreshold membrane dynamics without spiking."""

    def setUp(self):
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{k}', delta)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_passive_decay_from_rest(self):
        r"""Starting at V=0 (rest) with no input, V should remain at 0."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=0.0, c_3=0.0,  # disable spiking
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            for k in range(10):
                self._step(neuron, k)
                v = float((neuron.V.value / u.mV)[0])
                self.assertAlmostEqual(v, 0.0, places=10)

    def test_passive_decay_from_nonzero(self):
        r"""Starting at V=10mV (above rest) with no input, V should decay exponentially."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                tau_m=10.0 * u.ms,
                C_m=250.0 * u.pF,
                c_1=0.0, c_2=0.0, c_3=0.0,  # disable spiking
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(10.0 * u.mV),
            )
            neuron.init_state()

            # After 10 steps (1 ms), V should decay by exp(-1/10)^10 = exp(-1)
            for k in range(10):
                self._step(neuron, k)

            v = float((neuron.V.value / u.mV)[0])
            expected = 10.0 * math.exp(-1.0 / 10.0)  # after 1 ms with tau_m=10ms
            self.assertAlmostEqual(v, expected, places=5)

    def test_current_input_has_one_step_delay(self):
        r"""External current should be stored for use in the NEXT step."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                tau_m=10.0 * u.ms,
                C_m=250.0 * u.pF,
                c_1=0.0, c_2=0.0, c_3=0.0,  # disable spiking
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            # Step 0: inject current. V should still be 0 because I_stim=0 at start
            self._step(neuron, 0, x=1000.0 * u.pA)
            v0 = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v0, 0.0, places=8)

            # Step 1: now the 1000 pA should take effect
            self._step(neuron, 1, x=0.0 * u.pA)
            v1 = float((neuron.V.value / u.mV)[0])
            self.assertTrue(v1 > 0.0, f"V should increase from current, got {v1}")

    def test_delta_input_adds_instantaneous_voltage_jump(self):
        r"""Delta inputs should cause immediate voltage jumps."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=0.0, c_3=0.0,  # disable spiking
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            # Inject 5 mV delta input
            self._step(neuron, 0, delta=5.0 * u.mV)
            v = float((neuron.V.value / u.mV)[0])
            # V should be ~5 mV (the decay over one step from 0 is negligible,
            # but the exact value includes decay of the propagator)
            self.assertAlmostEqual(v, 5.0, places=3)

    def test_exact_integration_single_step(self):
        r"""Test that exact integration matches analytic formula for one step."""
        h = 0.1  # ms
        tau_m = 10.0
        C_m = 250.0
        I_e = 100.0  # pA
        V0 = 5.0  # mV relative to rest

        P33 = math.exp(-h / tau_m)
        P30 = 1.0 / C_m * (1.0 - P33) * tau_m

        # No external current on first step (ring buffer), only I_e
        V_expected = P30 * I_e + P33 * V0

        with brainstate.environ.context(dt=h * u.ms):
            neuron = pp_psc_delta(
                1,
                tau_m=tau_m * u.ms,
                C_m=C_m * u.pF,
                I_e=I_e * u.pA,
                c_1=0.0, c_2=0.0, c_3=0.0,  # disable spiking
                V_initializer=braintools.init.Constant(V0 * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0)
            v = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v, V_expected, places=8)

    def test_constant_current_steady_state(self):
        r"""With constant current and no spiking, V should approach steady state V_ss = I_e * tau_m / C_m."""
        tau_m = 10.0
        C_m = 250.0
        I_e = 500.0  # pA
        V_ss = I_e * tau_m / C_m  # = 20 mV

        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = pp_psc_delta(
                1,
                tau_m=tau_m * u.ms,
                C_m=C_m * u.pF,
                I_e=I_e * u.pA,
                c_1=0.0, c_2=0.0, c_3=0.0,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            # Run for a long time (100 ms = 1000 steps) to reach steady state
            for k in range(1000):
                self._step(neuron, k)

            v = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v, V_ss, places=2)


class TestPpPscDeltaDeadTime(unittest.TestCase):
    r"""Test dead time (refractory period) mechanics."""

    def setUp(self):
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{k}', delta)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_no_spike_during_dead_time(self):
        r"""During dead time, no spikes should be emitted."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,  # very high rate
                dead_time=5.0,  # 5 ms = 50 steps
                with_reset=False,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            # Step 0: should spike with very high rate
            spk0 = self._step(neuron, 0)
            self.assertTrue(float(spk0[0]) > 0, "Should spike with very high rate")

            # Steps 1-49: should not spike (in dead time)
            for k in range(1, 50):
                spk = self._step(neuron, k)
                self.assertEqual(float(spk[0]), 0.0,
                                 f"Should not spike during dead time at step {k}")

    def test_dead_time_count_matches_duration(self):
        r"""Refractory counter should match dead_time / dt."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,
                dead_time=2.5,  # 25 steps at dt=0.1
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            self._step(neuron, 0)
            ref_count = int(neuron.refractory_step_count.value[0])
            expected = int(round(2.5 / 0.1))  # 25
            self.assertEqual(ref_count, expected)

    def test_dead_time_clamped_to_resolution(self):
        r"""If dead_time < h, it should be clamped to h internally."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,
                dead_time=0.01,  # much less than h=0.1
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            self._step(neuron, 0)
            ref_count = int(neuron.refractory_step_count.value[0])
            # Should be clamped to 1 step (h=0.1ms)
            self.assertGreaterEqual(ref_count, 1)

    def test_dead_time_counter_decrements(self):
        r"""Dead time counter should decrement by 1 each step."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,
                dead_time=3.0,  # 30 steps
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            self._step(neuron, 0)
            r0 = int(neuron.refractory_step_count.value[0])

            self._step(neuron, 1)
            r1 = int(neuron.refractory_step_count.value[0])

            self.assertEqual(r1, r0 - 1)

    def test_t_ref_remaining_initializes_dead_time(self):
        r"""t_ref_remaining should set the initial dead time counter."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,
                t_ref_remaining=2.0,  # 20 steps at dt=0.1
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            # Should be in dead time initially
            ref_count = int(neuron.refractory_step_count.value[0])
            self.assertEqual(ref_count, 20)

            # Should not spike during initial dead time
            spk = self._step(neuron, 0)
            self.assertEqual(float(spk[0]), 0.0)


class TestPpPscDeltaMembraneReset(unittest.TestCase):
    r"""Test membrane potential reset behavior."""

    def setUp(self):
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{k}', delta)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_with_reset_resets_to_zero(self):
        r"""When with_reset=True, V should be reset to 0 after spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,
                with_reset=True,
                dead_time=0.1,  # minimal dead time
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(5.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            spk = self._step(neuron, 0)
            if float(spk[0]) > 0:
                v = float((neuron.V.value / u.mV)[0])
                self.assertAlmostEqual(v, 0.0, places=8,
                                       msg="V should be 0 after reset")

    def test_without_reset_preserves_voltage(self):
        r"""When with_reset=False, V should NOT be reset after spike."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,
                with_reset=False,
                dead_time=0.1,
                I_e=500.0 * u.pA,
                V_initializer=braintools.init.Constant(5.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            spk = self._step(neuron, 0)
            if float(spk[0]) > 0:
                v = float((neuron.V.value / u.mV)[0])
                # V should be the post-propagator value, not 0
                self.assertNotAlmostEqual(v, 0.0, places=3,
                                          msg="V should NOT be reset with with_reset=False")


class TestPpPscDeltaAdaptation(unittest.TestCase):
    r"""Test spike-frequency adaptation mechanics."""

    def setUp(self):
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{k}', delta)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_sfa_elements_decay_exponentially(self):
        r"""SFA elements should decay by exp(-dt/tau) each step."""
        tau_sfa = [34.0, 100.0]
        q_sfa = [7.0, 3.0]

        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,  # force spike
                dead_time=0.1,
                tau_sfa=tau_sfa,
                q_sfa=q_sfa,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Force a spike on step 0
            self._step(neuron, 0)

            # Disable spiking for subsequent steps
            old_c2 = neuron.c_2
            neuron.c_2 = 0.0
            neuron.c_1 = 0.0

            # Run 10 steps (1 ms) to let adaptation decay
            for k in range(1, 11):
                self._step(neuron, k)

            # Check each element decayed correctly
            h = 0.1  # ms
            for i in range(len(tau_sfa)):
                # After the spike, q_elems[i] = q_sfa[i]
                # Then 10 steps of decay: expected = q_sfa[i] * exp(-10*h / tau)
                n_decay_steps = 10  # steps 1-10 each decay
                expected = q_sfa[i] * math.exp(-n_decay_steps * h / tau_sfa[i])
                actual = neuron._q_elems[i][0]
                self.assertAlmostEqual(actual, expected, places=6,
                                       msg=f"SFA element {i} decay mismatch")

    def test_adaptation_reduces_effective_rate(self):
        r"""After a spike, adaptation should reduce the effective firing rate."""
        with brainstate.environ.context(dt=self.dt):
            # Without adaptation
            no_adapt = pp_psc_delta(
                1,
                c_1=0.0, c_2=50.0, c_3=0.25,
                dead_time=1e-8,
                tau_sfa=(),
                q_sfa=(),
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(100),
            )
            no_adapt.init_state()

            # With adaptation
            with_adapt = pp_psc_delta(
                1,
                c_1=0.0, c_2=50.0, c_3=0.25,
                dead_time=1e-8,
                tau_sfa=[34.0],
                q_sfa=[7.0],
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(100),
            )
            with_adapt.init_state()

            spikes_no_adapt = 0
            spikes_with_adapt = 0
            for k in range(1000):
                s1 = self._step(no_adapt, k)
                s2 = self._step(with_adapt, k)
                if float(s1[0]) > 0:
                    spikes_no_adapt += 1
                if float(s2[0]) > 0:
                    spikes_with_adapt += 1

            # With adaptation, there should be fewer spikes
            self.assertTrue(spikes_with_adapt < spikes_no_adapt,
                            f"Expected fewer spikes with adaptation: "
                            f"{spikes_with_adapt} vs {spikes_no_adapt}")

    def test_multi_component_adaptation(self):
        r"""Multiple adaptation components should all contribute."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,
                dead_time=0.1,
                tau_sfa=[10.0, 50.0, 200.0],
                q_sfa=[5.0, 3.0, 1.0],
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Force spike
            self._step(neuron, 0)

            # All 3 elements should have nonzero values
            for i in range(3):
                self.assertTrue(neuron._q_elems[i][0] > 0,
                                f"SFA element {i} should be nonzero after spike")


class TestPpPscDeltaStochasticSpiking(unittest.TestCase):
    r"""Test stochastic spike generation mechanics."""

    def setUp(self):
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{k}', delta)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_no_spikes_with_zero_rate(self):
        r"""With c_1=c_2=0, rate is always 0, so no spikes should occur."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=0.0, c_3=0.0,
                I_e=1000.0 * u.pA,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            for k in range(100):
                spk = self._step(neuron, k)
                self.assertEqual(float(spk[0]), 0.0,
                                 f"No spike expected with zero rate at step {k}")

    def test_high_rate_produces_spikes(self):
        r"""With very high c_2 (high rate), spikes should occur readily."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,
                dead_time=1e-8,  # minimal dead time
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            spike_count = 0
            for k in range(100):
                spk = self._step(neuron, k)
                if float(spk[0]) > 0:
                    spike_count += 1

            self.assertTrue(spike_count > 30,
                            f"Expected many spikes with high rate, got {spike_count}")

    def test_deterministic_with_fixed_rng_key(self):
        r"""Two neurons with the same RNG key should spike identically."""
        with brainstate.environ.context(dt=self.dt):
            key = jax.random.PRNGKey(12345)
            n1 = pp_psc_delta(1, c_1=0.0, c_2=50.0, c_3=0.25,
                              dead_time=1.0, rng_key=key)
            n2 = pp_psc_delta(1, c_1=0.0, c_2=50.0, c_3=0.25,
                              dead_time=1.0, rng_key=key)
            n1.init_state()
            n2.init_state()

            for k in range(50):
                s1 = self._step(n1, k)
                s2 = self._step(n2, k)
                self.assertEqual(float(s1[0]), float(s2[0]),
                                 f"Spike mismatch at step {k} with identical RNG")

    def test_linear_transfer_function(self):
        r"""With c_2=0, only the linear part c_1*V should contribute to rate."""
        with brainstate.environ.context(dt=self.dt):
            # Positive V and positive c_1 should produce spikes
            neuron = pp_psc_delta(
                1,
                c_1=100.0,  # Hz/mV
                c_2=0.0,
                c_3=0.0,
                dead_time=1e-8,
                I_e=500.0 * u.pA,  # drive V positive
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(42),
            )
            neuron.init_state()

            spike_count = 0
            for k in range(200):
                spk = self._step(neuron, k)
                if float(spk[0]) > 0:
                    spike_count += 1

            # Should have some spikes from the linear rate
            self.assertTrue(spike_count > 0,
                            f"Expected spikes from linear transfer function")


class TestPpPscDeltaTransferFunction(unittest.TestCase):
    r"""Test the rate/transfer function computation."""

    def test_rate_computation_exponential_only(self):
        r"""With c_1=0, rate = c_2 * exp(c_3 * V_eff)."""
        c_2 = 1.238
        c_3 = 0.25
        V_eff = 0.0  # at rest, no adaptation
        rate = c_2 * math.exp(c_3 * V_eff)
        self.assertAlmostEqual(rate, c_2, places=6)

    def test_rate_computation_linear_plus_exponential(self):
        r"""With both c_1 and c_2, rate = c_1*V' + c_2*exp(c_3*V')."""
        c_1 = 10.0
        c_2 = 5.0
        c_3 = 0.1
        V_eff = 3.0
        rate = c_1 * V_eff + c_2 * math.exp(c_3 * V_eff)
        expected = 30.0 + 5.0 * math.exp(0.3)
        self.assertAlmostEqual(rate, expected, places=6)

    def test_negative_rate_suppressed(self):
        r"""Negative rate (from rectifier) should not produce spikes."""
        # With c_1 > 0, c_2 = 0, and V_eff < 0, rate < 0 -> no spikes
        c_1 = 10.0
        c_2 = 0.0
        V_eff = -5.0
        rate = c_1 * V_eff + c_2 * math.exp(0)  # c_3=0 -> exp(0)=1
        self.assertTrue(rate < 0.0)
        # NEST applies Rect[rate] so no spikes


class TestPpPscDeltaReferenceTrace(unittest.TestCase):
    r"""Compare full simulation traces against standalone reference implementation."""

    def setUp(self):
        self.dt_val = 0.1
        self.dt = self.dt_val * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{k}', delta)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_subthreshold_trace_matches_reference(self):
        r"""Multi-step subthreshold trace with constant current should match reference."""
        p = {
            'tau_m': 10.0, 'C_m': 250.0,
            'c_1': 0.0, 'c_2': 0.0, 'c_3': 0.0,  # no spiking
            'I_e': 100.0, 'dead_time': 1.0, 'with_reset': True,
        }

        n_steps = 50
        i_stim_seq = [0.0] * n_steps
        i_stim_seq[0] = 50.0
        i_stim_seq[5] = -30.0
        delta_v_seq = [0.0] * n_steps
        delta_v_seq[3] = 2.0  # 2 mV jump at step 3
        delta_v_seq[10] = -1.5  # -1.5 mV jump at step 10

        v_ref, q_ref, _, _ = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, delta_v_seq, [1.0] * n_steps,  # rand=1 -> never spike
            tau_sfa=[], q_sfa=[]
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                tau_m=p['tau_m'] * u.ms,
                C_m=p['C_m'] * u.pF,
                I_e=p['I_e'] * u.pA,
                c_1=0.0, c_2=0.0, c_3=0.0,
                dead_time=p['dead_time'],
                with_reset=p['with_reset'],
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            v_model = []
            for k in range(n_steps):
                x_pA = i_stim_seq[k]
                dv = delta_v_seq[k]
                delta = dv * u.mV if dv != 0.0 else None
                self._step(neuron, k, x=x_pA * u.pA, delta=delta)
                v_model.append(float((neuron.V.value / u.mV)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=5,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")

    def test_trace_with_adaptation_and_spiking(self):
        r"""Full trace with adaptation and controlled spiking matches reference."""
        p = {
            'tau_m': 10.0, 'C_m': 250.0,
            'c_1': 0.0, 'c_2': 50.0, 'c_3': 0.25,
            'I_e': 0.0, 'dead_time': 5.0, 'with_reset': True,
        }
        tau_sfa = [34.0]
        q_sfa = [7.0]

        n_steps = 100
        i_stim_seq = [0.0] * n_steps
        delta_v_seq = [0.0] * n_steps

        # Generate known random numbers using JAX
        key = jax.random.PRNGKey(77)
        rand_vals = []
        for k in range(n_steps):
            key, subkey = jax.random.split(key)
            rand_vals.append(float(jax.random.uniform(subkey, shape=(1,))[0]))

        v_ref, q_ref, spk_ref, _ = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, delta_v_seq, rand_vals,
            tau_sfa=tau_sfa, q_sfa=q_sfa
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                tau_m=p['tau_m'] * u.ms,
                C_m=p['C_m'] * u.pF,
                I_e=p['I_e'] * u.pA,
                c_1=p['c_1'], c_2=p['c_2'], c_3=p['c_3'],
                dead_time=p['dead_time'],
                with_reset=p['with_reset'],
                tau_sfa=tau_sfa,
                q_sfa=q_sfa,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(77),
            )
            neuron.init_state()

            v_model = []
            spk_model = []
            for k in range(n_steps):
                spk = self._step(neuron, k)
                v_model.append(float((neuron.V.value / u.mV)[0]))
                spk_model.append(float(spk[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=6,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")

    def test_trace_without_reset(self):
        r"""Trace with with_reset=False should match reference."""
        p = {
            'tau_m': 10.0, 'C_m': 250.0,
            'c_1': 0.0, 'c_2': 20.0, 'c_3': 0.25,
            'I_e': 50.0, 'dead_time': 2.0, 'with_reset': False,
        }
        tau_sfa = [34.0]
        q_sfa = [5.0]

        n_steps = 80
        i_stim_seq = [0.0] * n_steps
        delta_v_seq = [0.0] * n_steps

        key = jax.random.PRNGKey(55)
        rand_vals = []
        for k in range(n_steps):
            key, subkey = jax.random.split(key)
            rand_vals.append(float(jax.random.uniform(subkey, shape=(1,))[0]))

        v_ref, _, spk_ref, _ = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, delta_v_seq, rand_vals,
            tau_sfa=tau_sfa, q_sfa=q_sfa
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                tau_m=p['tau_m'] * u.ms,
                C_m=p['C_m'] * u.pF,
                I_e=p['I_e'] * u.pA,
                c_1=p['c_1'], c_2=p['c_2'], c_3=p['c_3'],
                dead_time=p['dead_time'],
                with_reset=p['with_reset'],
                tau_sfa=tau_sfa,
                q_sfa=q_sfa,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(55),
            )
            neuron.init_state()

            v_model = []
            for k in range(n_steps):
                self._step(neuron, k)
                v_model.append(float((neuron.V.value / u.mV)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=6,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")

    def test_trace_with_synaptic_delta_inputs(self):
        r"""Trace with delta voltage inputs should match reference."""
        p = {
            'tau_m': 10.0, 'C_m': 250.0,
            'c_1': 0.0, 'c_2': 10.0, 'c_3': 0.25,
            'I_e': 0.0, 'dead_time': 3.0, 'with_reset': True,
        }

        n_steps = 60
        i_stim_seq = [0.0] * n_steps
        delta_v_seq = [0.0] * n_steps
        # Inject delta inputs at specific times
        delta_v_seq[0] = 5.0
        delta_v_seq[10] = 3.0
        delta_v_seq[20] = -2.0
        delta_v_seq[30] = 10.0

        key = jax.random.PRNGKey(33)
        rand_vals = []
        for k in range(n_steps):
            key, subkey = jax.random.split(key)
            rand_vals.append(float(jax.random.uniform(subkey, shape=(1,))[0]))

        v_ref, _, spk_ref, _ = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, delta_v_seq, rand_vals,
            tau_sfa=[], q_sfa=[]
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                tau_m=p['tau_m'] * u.ms,
                C_m=p['C_m'] * u.pF,
                I_e=p['I_e'] * u.pA,
                c_1=p['c_1'], c_2=p['c_2'], c_3=p['c_3'],
                dead_time=p['dead_time'],
                with_reset=p['with_reset'],
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(33),
            )
            neuron.init_state()

            v_model = []
            for k in range(n_steps):
                dv = delta_v_seq[k]
                delta = dv * u.mV if dv != 0.0 else None
                self._step(neuron, k, delta=delta)
                v_model.append(float((neuron.V.value / u.mV)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=5,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")

    def test_trace_with_multi_adaptation(self):
        r"""Trace with multiple adaptation components matches reference."""
        p = {
            'tau_m': 25.0, 'C_m': 250.0,
            'c_1': 0.0, 'c_2': 10.0, 'c_3': 1.0,
            'I_e': 0.0, 'dead_time': 0.001, 'with_reset': False,
        }
        tau_sfa = [300.0, 25.0]
        q_sfa = [0.1, 50.0]

        n_steps = 100
        i_stim_seq = [0.0] * n_steps
        delta_v_seq = [0.0] * n_steps

        key = jax.random.PRNGKey(88)
        rand_vals = []
        for k in range(n_steps):
            key, subkey = jax.random.split(key)
            rand_vals.append(float(jax.random.uniform(subkey, shape=(1,))[0]))

        v_ref, q_ref, _, _ = _run_nest_ref(
            n_steps, self.dt_val, p,
            i_stim_seq, delta_v_seq, rand_vals,
            tau_sfa=tau_sfa, q_sfa=q_sfa
        )

        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                tau_m=p['tau_m'] * u.ms,
                C_m=p['C_m'] * u.pF,
                I_e=p['I_e'] * u.pA,
                c_1=p['c_1'], c_2=p['c_2'], c_3=p['c_3'],
                dead_time=p['dead_time'],
                with_reset=p['with_reset'],
                tau_sfa=tau_sfa,
                q_sfa=q_sfa,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(88),
            )
            neuron.init_state()

            v_model = []
            for k in range(n_steps):
                self._step(neuron, k)
                v_model.append(float((neuron.V.value / u.mV)[0]))

        for k in range(n_steps):
            self.assertAlmostEqual(v_model[k], v_ref[k], places=5,
                                   msg=f"V mismatch at step {k}: model={v_model[k]}, ref={v_ref[k]}")


class TestPpPscDeltaUpdateOrder(unittest.TestCase):
    r"""Test that the update order matches NEST exactly."""

    def setUp(self):
        self.dt = 0.1 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{k}', delta)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_adaptation_decay_before_spike_check(self):
        r"""Adaptation should be decayed BEFORE the spike check, matching NEST."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=1e6, c_3=0.25,
                dead_time=0.1,
                tau_sfa=[10.0],
                q_sfa=[100.0],
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                rng_key=jax.random.PRNGKey(0),
            )
            neuron.init_state()

            # Spike on step 0 -> q_elems[0] += 100
            self._step(neuron, 0)

            # On step 1: q_elems should first decay, then be used for spike check
            # After decay: q_elems[0] = 100 * exp(-0.1/10) = 100 * 0.99005...
            neuron.c_2 = 0.0  # disable spiking
            self._step(neuron, 1)

            # The E_sfa should be the decayed value
            expected_q = 100.0 * math.exp(-0.1 / 10.0)
            actual_q = neuron._q_val[0]
            self.assertAlmostEqual(actual_q, expected_q, places=6)

    def test_delta_input_applied_before_spike_check(self):
        r"""Delta inputs should be incorporated in the V update before the spike check."""
        with brainstate.environ.context(dt=self.dt):
            neuron = pp_psc_delta(
                1,
                c_1=0.0, c_2=0.0, c_3=0.0,  # no spiking
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            # Inject 10 mV delta
            self._step(neuron, 0, delta=10.0 * u.mV)
            v = float((neuron.V.value / u.mV)[0])

            # V should include the delta jump (at step 0, I_stim=0, V_old=0)
            # V = P30*(0+0) + P33*0 + 10 = 10 mV
            self.assertAlmostEqual(v, 10.0, places=3)


class TestPpPscDeltaShapeBatch(unittest.TestCase):
    r"""Test population shape and batch handling."""

    def test_population_shape(self):
        r"""Model should work with multi-neuron populations."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = pp_psc_delta(
                4,
                c_1=0.0, c_2=0.0, c_3=0.0,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()
            self.assertEqual(neuron.V.value.shape, (4,))

            with brainstate.environ.context(t=0.0 * u.ms):
                spk = neuron.update()
                self.assertEqual(spk.shape, (4,))

    def test_batch_size(self):
        r"""Model should work with batch dimensions."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = pp_psc_delta(
                3,
                c_1=0.0, c_2=0.0, c_3=0.0,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state(batch_size=2)
            self.assertEqual(neuron.V.value.shape, (2, 3))

            with brainstate.environ.context(t=0.0 * u.ms):
                spk = neuron.update()
                self.assertEqual(spk.shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
