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
Tests for mat2_psc_exp neuron model.

These tests verify that the implementation produces the same dynamics as NEST's
mat2_psc_exp model, including:
- Default parameter values
- Parameter validation
- Subthreshold membrane dynamics
- Spike generation and adaptive threshold updates
- Refractory period behavior
- Synaptic current responses
- Exact match with NEST reference data (spike times and V_m/V_th traces)
"""

import math
import unittest

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._nest_neuron.mat2_psc_exp import mat2_psc_exp

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)


class TestMat2PscExp(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def _step(self, neuron, step_idx, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    # ------------------------------------------------------------------
    # Test 1: Default parameters match NEST
    # ------------------------------------------------------------------
    def test_nest_default_parameters(self):
        r"""Verify that all default parameter values match NEST's mat2_psc_exp."""
        neuron = mat2_psc_exp(1)
        self.assertEqual(neuron.E_L, -70. * u.mV)
        self.assertEqual(neuron.C_m, 100. * u.pF)
        self.assertEqual(neuron.tau_m, 5. * u.ms)
        self.assertEqual(neuron.t_ref, 2. * u.ms)
        self.assertEqual(neuron.tau_syn_ex, 1. * u.ms)
        self.assertEqual(neuron.tau_syn_in, 3. * u.ms)
        self.assertEqual(neuron.I_e, 0. * u.pA)
        self.assertEqual(neuron.tau_1, 10. * u.ms)
        self.assertEqual(neuron.tau_2, 200. * u.ms)
        self.assertEqual(neuron.alpha_1, 37. * u.mV)
        self.assertEqual(neuron.alpha_2, 2. * u.mV)
        self.assertEqual(neuron.omega, -51. * u.mV)
        self.assertEqual(neuron.spk_reset, 'hard')

    # ------------------------------------------------------------------
    # Test 2: Parameter validation
    # ------------------------------------------------------------------
    def test_parameter_validation(self):
        r"""Test that invalid parameters raise ValueError."""
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, C_m=0.0 * u.pF)
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, C_m=-1.0 * u.pF)
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, tau_m=0.0 * u.ms)
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, tau_syn_ex=0.0 * u.ms)
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, tau_syn_in=0.0 * u.ms)
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, t_ref=0.0 * u.ms)
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, tau_1=0.0 * u.ms)
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, tau_2=0.0 * u.ms)
        # tau_m must differ from tau_syn_ex and tau_syn_in
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, tau_m=1.0 * u.ms, tau_syn_ex=1.0 * u.ms)
        with self.assertRaises(ValueError):
            mat2_psc_exp(1, tau_m=3.0 * u.ms, tau_syn_in=3.0 * u.ms)

    # ------------------------------------------------------------------
    # Test 3: NEST reference spike times
    # ------------------------------------------------------------------
    def test_nest_reference_spike_times(self):
        r"""
        Reproduce the NEST test_mat2_psc_exp.py simulation exactly.

        A DC current of 2400 pA is injected via a dc_generator connected
        with weight=1 and delay=0.1 ms.  The NEST test expects spikes at
        time steps [11, 32, 54].

        In NEST, dc_generator output arrives one step delayed (min delay).
        We model this by starting the current injection at step 1 (i.e., the
        current is first *used* at step 2 via i_0 buffering), which means
        i_0 is set to the DC amplitude starting from step 1 onwards.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = mat2_psc_exp(
                1,
                omega=-51.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            n_steps = 80
            dt_ms = 0.1

            def step_fn(k):
                dc = jnp.where(k >= 1, 2400.0, 0.0) * u.pA
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    spk = neuron.update(x=dc)
                return spk

            all_spikes = brainstate.transform.for_loop(step_fn, jnp.arange(n_steps))
            spike_mask = np.array(all_spikes).reshape(-1) > 0.5
            spike_times = [int(k) + 1 for k in np.where(spike_mask)[0]]

            np.testing.assert_array_equal(spike_times, [11, 32, 54])

    # ------------------------------------------------------------------
    # Test 4: NEST reference V_m and V_th traces
    # ------------------------------------------------------------------
    def test_nest_reference_potentials(self):
        r"""
        Verify V_m and V_th traces match the NEST reference data from
        test_mat2_psc_exp.py for the first 21 time steps.
        """
        # NEST reference data (times in steps, V_m and V_th in mV)
        expected = [
            # (step, V_m, V_th)
            (1, -70.0, -51.0),
            (2, -70.0, -51.0),
            (3, -67.6238, -51.0),
            (4, -65.2947, -51.0),
            (5, -63.0117, -51.0),
            (6, -60.774, -51.0),
            (7, -58.5805, -51.0),
            (8, -56.4305, -51.0),
            (9, -54.323, -51.0),
            (10, -52.2573, -51.0),
            (11, -50.2324, -12.0),
            (12, -48.2477, -12.3692),
            (13, -46.3023, -12.7346),
            (14, -44.3953, -13.0965),
            (15, -42.5262, -13.4548),
            (16, -40.694, -13.8095),
            (17, -38.8982, -14.1607),
            (18, -37.1379, -14.5084),
            (19, -35.4124, -14.8527),
            (20, -33.7212, -15.1935),
            (21, -32.0634, -15.531),
        ]

        with brainstate.environ.context(dt=self.dt):
            neuron = mat2_psc_exp(
                1,
                omega=-51.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            n_steps = 21
            dt_ms = 0.1
            omega_abs = float(u.math.asarray(neuron.omega / u.mV))

            def step_fn(k):
                dc = jnp.where(k >= 1, 2400.0, 0.0) * u.pA
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    neuron.update(x=dc)
                return neuron.V.value / u.mV, neuron.V_th_1.value / u.mV, neuron.V_th_2.value / u.mV

            all_V, all_vth1, all_vth2 = brainstate.transform.for_loop(
                step_fn, jnp.arange(n_steps)
            )
            all_V_np = np.array(all_V)[:, 0]
            all_vth1_np = np.array(all_vth1)[:, 0]
            all_vth2_np = np.array(all_vth2)[:, 0]

            for i, (step, exp_vm, exp_vth) in enumerate(expected):
                recorded_vm = float(all_V_np[i])
                recorded_vth = omega_abs + float(all_vth1_np[i]) + float(all_vth2_np[i])
                self.assertAlmostEqual(
                    recorded_vm, exp_vm, places=3,
                    msg=f'V_m mismatch at step {step}: got {recorded_vm}, expected {exp_vm}'
                )
                self.assertAlmostEqual(
                    recorded_vth, exp_vth, places=3,
                    msg=f'V_th mismatch at step {step}: got {recorded_vth}, expected {exp_vth}'
                )

    # ------------------------------------------------------------------
    # Test 5: Subthreshold dynamics match exact integration
    # ------------------------------------------------------------------
    def test_subthreshold_dynamics(self):
        r"""
        Verify subthreshold membrane dynamics match the exact integration
        equations when no spikes occur.
        """
        with brainstate.environ.context(dt=self.dt):
            params = dict(
                E_L=-70.0, C_m=100.0, tau_m=5.0,
                tau_syn_ex=1.0, tau_syn_in=3.0,
                I_e=50.0,  # small current, below threshold
                omega=-51.0,
                tau_1=10.0, tau_2=200.0,
                alpha_1=37.0, alpha_2=2.0,
            )
            neuron = mat2_psc_exp(
                1,
                E_L=params['E_L'] * u.mV,
                C_m=params['C_m'] * u.pF,
                tau_m=params['tau_m'] * u.ms,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                omega=params['omega'] * u.mV,
                tau_1=params['tau_1'] * u.ms,
                tau_2=params['tau_2'] * u.ms,
                alpha_1=params['alpha_1'] * u.mV,
                alpha_2=params['alpha_2'] * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            h = 0.1
            tau_m = params['tau_m']
            C_m = params['C_m']
            tau_ex = params['tau_syn_ex']
            tau_in = params['tau_syn_in']
            I_e = params['I_e']

            P11ex = math.exp(-h / tau_ex)
            P11in = math.exp(-h / tau_in)
            P22_expm1 = math.expm1(-h / tau_m)
            P21ex = -tau_m / (C_m * (1.0 - tau_m / tau_ex)) * P11ex * math.expm1(h * (1.0 / tau_ex - 1.0 / tau_m))
            P21in = -tau_m / (C_m * (1.0 - tau_m / tau_in)) * P11in * math.expm1(h * (1.0 / tau_in - 1.0 / tau_m))
            P20 = -tau_m / C_m * P22_expm1

            # Compute reference V_m trajectory
            n_steps = 20
            ref_v = np.zeros(n_steps)
            v = 0.0
            i0 = 0.0
            iex = 0.0
            iin = 0.0
            for k in range(n_steps):
                v = v * P22_expm1 + v + iex * P21ex + iin * P21in + (I_e + i0) * P20
                iex *= P11ex
                iin *= P11in
                ref_v[k] = v + params['E_L']

            dt_ms = 0.1

            def step_fn(k):
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    neuron.update()
                return neuron.V.value / u.mV

            all_V = brainstate.transform.for_loop(step_fn, jnp.arange(n_steps))
            all_V_np = np.array(all_V)[:, 0]

            for k in range(n_steps):
                self.assertAlmostEqual(all_V_np[k], ref_v[k], delta=1e-11,
                                       msg=f'V_m mismatch at step {k}')

    # ------------------------------------------------------------------
    # Test 6: No voltage reset on spike
    # ------------------------------------------------------------------
    def test_no_voltage_reset_on_spike(self):
        r"""
        Verify that the membrane potential is NOT reset after a spike.
        This is the key difference from standard LIF models.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = mat2_psc_exp(
                1,
                omega=-51.0 * u.mV,
                I_e=2400.0 * u.pA,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            n_steps = 80
            dt_ms = 0.1

            def step_fn(k):
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    spk = neuron.update()
                return spk, neuron.V.value / u.mV

            all_spks, all_V = brainstate.transform.for_loop(step_fn, jnp.arange(n_steps))
            all_spks_np = np.array(all_spks).reshape(-1)
            all_V_np = np.array(all_V)[:, 0]

            spike_steps = np.where(all_spks_np > 0.5)[0]
            self.assertGreater(len(spike_steps), 0, 'No spike occurred during test')
            first_spike_step = int(spike_steps[0])
            self.assertGreater(n_steps - 1, first_spike_step, 'Spike at last step, cannot check V_after')

            V_at_spike = float(all_V_np[first_spike_step])
            V_after_spike = float(all_V_np[first_spike_step + 1])
            # After spike, V should continue to evolve, NOT be reset to E_L
            self.assertNotAlmostEqual(V_after_spike, -70.0, places=3,
                                      msg='V_m appears to have been reset to E_L after spike')

    # ------------------------------------------------------------------
    # Test 7: Adaptive threshold jumps on spike
    # ------------------------------------------------------------------
    def test_threshold_jump_on_spike(self):
        r"""
        Verify that V_th_1 and V_th_2 jump by alpha_1 and alpha_2 on spike.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = mat2_psc_exp(
                1,
                omega=-51.0 * u.mV,
                I_e=2400.0 * u.pA,
                alpha_1=37.0 * u.mV,
                alpha_2=2.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            n_steps = 80
            dt_ms = 0.1

            def step_fn(k):
                vth1_before = neuron.V_th_1.value / u.mV
                vth2_before = neuron.V_th_2.value / u.mV
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    spk = neuron.update()
                vth1_after = neuron.V_th_1.value / u.mV
                vth2_after = neuron.V_th_2.value / u.mV
                return spk, vth1_before, vth2_before, vth1_after, vth2_after

            all_spks, all_vth1_before, all_vth2_before, all_vth1_after, all_vth2_after = (
                brainstate.transform.for_loop(step_fn, jnp.arange(n_steps))
            )
            all_spks_np = np.array(all_spks).reshape(-1)
            spike_steps = np.where(all_spks_np > 0.5)[0]
            self.assertGreater(len(spike_steps), 0, 'No spike occurred during test')

            first_spike_step = int(spike_steps[0])
            v_th_1_before = float(np.array(all_vth1_before)[first_spike_step, 0])
            v_th_2_before = float(np.array(all_vth2_before)[first_spike_step, 0])
            v_th_1_after = float(np.array(all_vth1_after)[first_spike_step, 0])
            v_th_2_after = float(np.array(all_vth2_after)[first_spike_step, 0])

            h = 0.1
            P11th = math.exp(-h / 10.0)
            P22th = math.exp(-h / 200.0)
            expected_vth1 = v_th_1_before * P11th + 37.0
            expected_vth2 = v_th_2_before * P22th + 2.0
            self.assertAlmostEqual(v_th_1_after, expected_vth1, delta=1e-10)
            self.assertAlmostEqual(v_th_2_after, expected_vth2, delta=1e-10)

    # ------------------------------------------------------------------
    # Test 8: Refractory period prevents spiking
    # ------------------------------------------------------------------
    def test_refractory_period(self):
        r"""
        Verify that the neuron cannot fire during the refractory period.
        With t_ref=2ms and dt=0.1ms, refractory lasts 20 steps.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = mat2_psc_exp(
                1,
                omega=-51.0 * u.mV,
                I_e=2400.0 * u.pA,
                t_ref=2.0 * u.ms,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            n_steps = 80
            dt_ms = 0.1

            def step_fn(k):
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    spk = neuron.update()
                return spk

            all_spikes = brainstate.transform.for_loop(step_fn, jnp.arange(n_steps))
            spike_mask = np.array(all_spikes).reshape(-1) > 0.5
            spike_steps = list(np.where(spike_mask)[0])

            # Must have at least 2 spikes
            self.assertGreaterEqual(len(spike_steps), 2)
            # Inter-spike interval must be >= 20 steps (2ms / 0.1ms)
            for i in range(1, len(spike_steps)):
                isi = spike_steps[i] - spike_steps[i - 1]
                self.assertGreaterEqual(isi, 20,
                                        f'ISI {isi} steps < refractory period 20 steps')

    # ------------------------------------------------------------------
    # Test 9: Synaptic current response
    # ------------------------------------------------------------------
    def test_synaptic_current_response(self):
        r"""
        Verify that excitatory and inhibitory synaptic currents follow
        exponential decay after spike input.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = mat2_psc_exp(
                1,
                I_e=0.0 * u.pA,
                omega=100.0 * u.mV,  # very high threshold to prevent spiking
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            h = 0.1
            tau_ex = 1.0
            tau_in = 3.0
            P11ex = math.exp(-h / tau_ex)
            P11in = math.exp(-h / tau_in)

            # Step 0: inject excitatory spike (positive weight)
            with brainstate.environ.context(t=0.0 * u.ms):
                neuron.update(spike_delta=100.0 * u.pA)

            # After step 0: i_syn_ex should be 100 (decayed then added)
            iex = 100.0
            actual_iex = float((neuron.i_syn_ex.value / u.pA)[0])
            self.assertAlmostEqual(actual_iex, iex, delta=1e-11)

            # Steps 1-9: follow exponential decay (no delta input)
            dt_ms = 0.1

            def decay_ex_fn(k):
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    neuron.update()
                return neuron.i_syn_ex.value / u.pA

            all_iex = brainstate.transform.for_loop(decay_ex_fn, jnp.arange(1, 10))
            all_iex_np = np.array(all_iex)[:, 0]
            for i in range(9):
                iex *= P11ex
                self.assertAlmostEqual(all_iex_np[i], iex, delta=1e-10,
                                       msg=f'i_syn_ex mismatch at step {i + 1}')

            # Step 10: inject inhibitory spike (negative weight)
            with brainstate.environ.context(t=10.0 * dt_ms * u.ms):
                neuron.update(spike_delta=-50.0 * u.pA)
            iin = -50.0
            actual_iin = float((neuron.i_syn_in.value / u.pA)[0])
            self.assertAlmostEqual(actual_iin, iin, delta=1e-11)

            # Steps 11-19: follow exponential decay
            def decay_in_fn(k):
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    neuron.update()
                return neuron.i_syn_in.value / u.pA

            all_iin = brainstate.transform.for_loop(decay_in_fn, jnp.arange(11, 20))
            all_iin_np = np.array(all_iin)[:, 0]
            for i in range(9):
                iin *= P11in
                self.assertAlmostEqual(all_iin_np[i], iin, delta=1e-10,
                                       msg=f'i_syn_in mismatch at step {i + 1}')

    # ------------------------------------------------------------------
    # Test 10: Full step-by-step equation match
    # ------------------------------------------------------------------
    def test_step_equations_match_reference(self):
        r"""
        Verify internal state variables step-by-step against the exact
        NEST update equations for mat2_psc_exp, including spike input,
        threshold adaptation, and refractory period.
        """
        with brainstate.environ.context(dt=self.dt):
            params = dict(
                E_L=-70.0, C_m=100.0, tau_m=5.0,
                t_ref=0.3,  # 3 steps
                tau_syn_ex=1.0, tau_syn_in=3.0,
                I_e=40.0,
                omega=-51.0,
                tau_1=10.0, tau_2=200.0,
                alpha_1=37.0, alpha_2=2.0,
            )
            neuron = mat2_psc_exp(
                1,
                E_L=params['E_L'] * u.mV,
                C_m=params['C_m'] * u.pF,
                tau_m=params['tau_m'] * u.ms,
                t_ref=params['t_ref'] * u.ms,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                omega=params['omega'] * u.mV,
                tau_1=params['tau_1'] * u.ms,
                tau_2=params['tau_2'] * u.ms,
                alpha_1=params['alpha_1'] * u.mV,
                alpha_2=params['alpha_2'] * u.mV,
                V_initializer=braintools.init.Constant(-67.0 * u.mV),
            )
            neuron.init_state()

            h = 0.1
            tau_m = params['tau_m']
            C_m = params['C_m']
            tau_ex = params['tau_syn_ex']
            tau_in = params['tau_syn_in']
            I_e = params['I_e']
            omega_rel = params['omega'] - params['E_L']  # 19.0
            alpha_1 = params['alpha_1']
            alpha_2 = params['alpha_2']

            # Propagator coefficients
            P11ex = math.exp(-h / tau_ex)
            P11in = math.exp(-h / tau_in)
            P22_expm1 = math.expm1(-h / tau_m)
            P21ex = -tau_m / (C_m * (1.0 - tau_m / tau_ex)) * P11ex * math.expm1(h * (1.0 / tau_ex - 1.0 / tau_m))
            P21in = -tau_m / (C_m * (1.0 - tau_m / tau_in)) * P11in * math.expm1(h * (1.0 / tau_in - 1.0 / tau_m))
            P20 = -tau_m / C_m * P22_expm1
            P11th = math.exp(-h / params['tau_1'])
            P22th = math.exp(-h / params['tau_2'])

            refr = int(math.ceil(params['t_ref'] / h))

            # Input sequences: spike weights and DC currents at specific steps
            w_seq = [0.0, 30.0, -15.0, 0.0, 0.0, 20.0, -10.0, 0.0, 0.0, 0.0]
            x0_seq = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            n_steps = len(w_seq)

            # Compute reference values (Python loop, reference only)
            v = -67.0 - params['E_L']
            i0_ref = 0.0
            iex_ref = 0.0
            iin_ref = 0.0
            vth1_ref = 0.0
            vth2_ref = 0.0
            r_ref = 0
            ref_spike = np.zeros(n_steps, dtype=bool)
            ref_V = np.zeros(n_steps)
            ref_iex = np.zeros(n_steps)
            ref_iin = np.zeros(n_steps)
            ref_vth1 = np.zeros(n_steps)
            ref_vth2 = np.zeros(n_steps)
            ref_r = np.zeros(n_steps, dtype=int)
            for k in range(n_steps):
                w = w_seq[k]
                x0 = x0_seq[k]
                v = v * P22_expm1 + v + iex_ref * P21ex + iin_ref * P21in + (I_e + i0_ref) * P20
                vth1_ref *= P11th
                vth2_ref *= P22th
                iex_ref *= P11ex
                iin_ref *= P11in
                iex_ref += max(w, 0.0)
                iin_ref += min(w, 0.0)
                if r_ref == 0:
                    if v >= omega_rel + vth1_ref + vth2_ref:
                        ref_spike[k] = True
                        r_ref = refr
                        vth1_ref += alpha_1
                        vth2_ref += alpha_2
                else:
                    r_ref -= 1
                i0_ref = x0
                ref_V[k] = v + params['E_L']
                ref_iex[k] = iex_ref
                ref_iin[k] = iin_ref
                ref_vth1[k] = vth1_ref
                ref_vth2[k] = vth2_ref
                ref_r[k] = r_ref

            # Run model with for_loop using pre-computed input arrays
            w_arr = jnp.array(w_seq)
            x0_arr = jnp.array(x0_seq)
            dt_ms = 0.1

            def step_fn(k):
                w = w_arr[k] * u.pA
                x0 = x0_arr[k] * u.pA
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    spk = neuron.update(x=x0, spike_delta=w)
                return (
                    spk,
                    neuron.V.value / u.mV,
                    neuron.i_syn_ex.value / u.pA,
                    neuron.i_syn_in.value / u.pA,
                    neuron.V_th_1.value / u.mV,
                    neuron.V_th_2.value / u.mV,
                    neuron.refractory_step_count.value,
                )

            results = brainstate.transform.for_loop(step_fn, jnp.arange(n_steps))
            all_spks, all_V, all_iex, all_iin, all_vth1, all_vth2, all_r = results

            all_spks_np = np.array(all_spks).reshape(-1)
            all_V_np = np.array(all_V)[:, 0]
            all_iex_np = np.array(all_iex)[:, 0]
            all_iin_np = np.array(all_iin)[:, 0]
            all_vth1_np = np.array(all_vth1)[:, 0]
            all_vth2_np = np.array(all_vth2)[:, 0]
            all_r_np = np.array(all_r)[:, 0]

            for k in range(n_steps):
                self.assertEqual(
                    bool(all_spks_np[k] > 0.5), bool(ref_spike[k]),
                    msg=f'Spike mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float(all_V_np[k]), ref_V[k], delta=1e-11,
                    msg=f'V_m mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float(all_iex_np[k]), ref_iex[k], delta=1e-11,
                    msg=f'i_syn_ex mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float(all_iin_np[k]), ref_iin[k], delta=1e-11,
                    msg=f'i_syn_in mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float(all_vth1_np[k]), ref_vth1[k], delta=1e-11,
                    msg=f'V_th_1 mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float(all_vth2_np[k]), ref_vth2[k], delta=1e-11,
                    msg=f'V_th_2 mismatch at step {k}'
                )
                self.assertEqual(
                    int(all_r_np[k]), ref_r[k],
                    msg=f'Refractory count mismatch at step {k}'
                )

    # ------------------------------------------------------------------
    # Test 11: Threshold decay without spikes
    # ------------------------------------------------------------------
    def test_threshold_decay_without_spikes(self):
        r"""
        Verify that V_th_1 and V_th_2 decay exponentially when no spikes
        occur (e.g., after manually setting them).
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = mat2_psc_exp(
                1,
                I_e=0.0 * u.pA,
                omega=100.0 * u.mV,  # unreachable threshold
                tau_1=10.0 * u.ms,
                tau_2=200.0 * u.ms,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            # Manually set threshold components
            neuron.V_th_1.value = jax.numpy.array([20.0]) * u.mV
            neuron.V_th_2.value = jax.numpy.array([5.0]) * u.mV

            h = 0.1
            P11th = math.exp(-h / 10.0)
            P22th = math.exp(-h / 200.0)

            n_steps = 50
            dt_ms = 0.1
            # Compute reference decay
            vth1_ref = np.zeros(n_steps)
            vth2_ref = np.zeros(n_steps)
            vth1 = 20.0
            vth2 = 5.0
            for k in range(n_steps):
                vth1 *= P11th
                vth2 *= P22th
                vth1_ref[k] = vth1
                vth2_ref[k] = vth2

            def step_fn(k):
                with brainstate.environ.context(t=k.astype(jnp.float64) * dt_ms * u.ms):
                    neuron.update()
                return neuron.V_th_1.value / u.mV, neuron.V_th_2.value / u.mV

            all_vth1, all_vth2 = brainstate.transform.for_loop(step_fn, jnp.arange(n_steps))
            all_vth1_np = np.array(all_vth1)[:, 0]
            all_vth2_np = np.array(all_vth2)[:, 0]

            for k in range(n_steps):
                self.assertAlmostEqual(
                    float(all_vth1_np[k]), vth1_ref[k], delta=1e-11,
                    msg=f'V_th_1 decay mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float(all_vth2_np[k]), vth2_ref[k], delta=1e-11,
                    msg=f'V_th_2 decay mismatch at step {k}'
                )

    # ------------------------------------------------------------------
    # Test 12: State initialization
    # ------------------------------------------------------------------
    def test_state_initialization(self):
        r"""Verify that all state variables are initialized correctly."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mat2_psc_exp(
                1,
                V_initializer=braintools.init.Constant(-65.0 * u.mV),
            )
            neuron.init_state()

            self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), -65.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.V_th_1.value / u.mV)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.V_th_2.value / u.mV)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.i_syn_ex.value / u.pA)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.i_syn_in.value / u.pA)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.i_0.value / u.pA)[0]), 0.0, delta=1e-12)
            self.assertEqual(int(neuron.refractory_step_count.value[0]), 0)


if __name__ == '__main__':
    unittest.main()
