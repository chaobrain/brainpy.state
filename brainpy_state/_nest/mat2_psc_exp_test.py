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

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax

from brainpy_state._nest.mat2_psc_exp import mat2_psc_exp

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class TestMat2PscExp(unittest.TestCase):
    def setUp(self):
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

            # Simulate 80 steps (8.0 ms) with DC current starting at step 1
            # (mimicking dc_generator with delay=0.1ms connected at t=0).
            spike_times = []
            for k in range(80):
                # DC current arrives at step 1 (delay of 1 step)
                dc = 2400.0 * u.pA if k >= 1 else 0.0 * u.pA
                spk = self._step(neuron, k, x=dc)
                if self._is_spike(spk):
                    spike_times.append(k + 1)  # NEST records at lag+1

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

            # In NEST, the multimeter records *before* the update at each step.
            # The data at step k reflects the state after step k-1's update.
            # We record after each update and compare with NEST data.
            # NEST records at steps 1..21 correspond to state after steps 0..20.
            recorded_vm = []
            recorded_vth = []

            for k in range(21):
                dc = 2400.0 * u.pA if k >= 1 else 0.0 * u.pA
                self._step(neuron, k, x=dc)

                V_m = float((neuron.V.value / u.mV)[0])
                omega_abs = float(u.math.asarray(neuron.omega / u.mV))
                V_th_1 = float((neuron.V_th_1.value / u.mV)[0])
                V_th_2 = float((neuron.V_th_2.value / u.mV)[0])
                V_th = omega_abs + V_th_1 + V_th_2

                recorded_vm.append(V_m)
                recorded_vth.append(V_th)

            for i, (step, exp_vm, exp_vth) in enumerate(expected):
                self.assertAlmostEqual(
                    recorded_vm[i], exp_vm, places=3,
                    msg=f'V_m mismatch at step {step}: got {recorded_vm[i]}, expected {exp_vm}'
                )
                self.assertAlmostEqual(
                    recorded_vth[i], exp_vth, places=3,
                    msg=f'V_th mismatch at step {step}: got {recorded_vth[i]}, expected {exp_vth}'
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

            v = 0.0  # V_m relative to E_L
            i0 = 0.0
            iex = 0.0
            iin = 0.0

            for k in range(20):
                self._step(neuron, k)

                # Evolve V_m
                v = v * P22_expm1 + v + iex * P21ex + iin * P21in + (I_e + i0) * P20

                # Decay PSCs (no spikes arriving)
                iex *= P11ex
                iin *= P11in

                actual_v = float((neuron.V.value / u.mV)[0])
                self.assertAlmostEqual(actual_v, v + params['E_L'], delta=1e-11,
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

            V_at_spike = None
            V_after_spike = None

            for k in range(80):
                spk = self._step(neuron, k)
                if self._is_spike(spk) and V_at_spike is None:
                    V_at_spike = float((neuron.V.value / u.mV)[0])
                elif V_at_spike is not None and V_after_spike is None:
                    V_after_spike = float((neuron.V.value / u.mV)[0])
                    break

            self.assertIsNotNone(V_at_spike)
            self.assertIsNotNone(V_after_spike)
            # After spike, V should continue to evolve, NOT be reset to E_L
            # V_after should be greater than V_at since current is still driving it up
            # (but may also decrease due to leak). The key check: it is NOT reset to E_L.
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

            first_spike = False
            for k in range(80):
                v_th_1_before = float((neuron.V_th_1.value / u.mV)[0])
                v_th_2_before = float((neuron.V_th_2.value / u.mV)[0])
                spk = self._step(neuron, k)
                if self._is_spike(spk) and not first_spike:
                    first_spike = True
                    v_th_1_after = float((neuron.V_th_1.value / u.mV)[0])
                    v_th_2_after = float((neuron.V_th_2.value / u.mV)[0])

                    # After spike, V_th_1 should have increased.
                    # The threshold decays during the step and then jumps.
                    # V_th_1 was 0 before, decayed by P11th, then jumped by alpha_1.
                    h = 0.1
                    P11th = math.exp(-h / 10.0)
                    P22th = math.exp(-h / 200.0)
                    expected_vth1 = v_th_1_before * P11th + 37.0
                    expected_vth2 = v_th_2_before * P22th + 2.0
                    self.assertAlmostEqual(v_th_1_after, expected_vth1, delta=1e-10)
                    self.assertAlmostEqual(v_th_2_after, expected_vth2, delta=1e-10)
                    break

            self.assertTrue(first_spike, 'No spike occurred during test')

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

            spike_steps = []
            for k in range(80):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    spike_steps.append(k)

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
            spk = self._step(neuron, 0, delta=100.0 * u.pA)

            # After step 0: i_syn_ex should be 100 (decayed then added)
            iex = 100.0  # added after decay of 0
            actual_iex = float((neuron.i_syn_ex.value / u.pA)[0])
            self.assertAlmostEqual(actual_iex, iex, delta=1e-11)

            # Follow exponential decay for several steps
            for k in range(1, 10):
                self._step(neuron, k)
                iex *= P11ex
                actual_iex = float((neuron.i_syn_ex.value / u.pA)[0])
                self.assertAlmostEqual(actual_iex, iex, delta=1e-10,
                                       msg=f'i_syn_ex mismatch at step {k}')

            # Now inject inhibitory spike (negative weight)
            self._step(neuron, 10, delta=-50.0 * u.pA)
            iin = -50.0
            actual_iin = float((neuron.i_syn_in.value / u.pA)[0])
            self.assertAlmostEqual(actual_iin, iin, delta=1e-11)

            for k in range(11, 20):
                self._step(neuron, k)
                iin *= P11in
                actual_iin = float((neuron.i_syn_in.value / u.pA)[0])
                self.assertAlmostEqual(actual_iin, iin, delta=1e-10,
                                       msg=f'i_syn_in mismatch at step {k}')

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

            v = -67.0 - params['E_L']  # V_m relative to E_L
            i0 = 0.0
            iex = 0.0
            iin = 0.0
            vth1 = 0.0
            vth2 = 0.0
            r = 0

            # Input sequences: spike weights at specific steps
            w_seq = [0.0, 30.0, -15.0, 0.0, 0.0, 20.0, -10.0, 0.0, 0.0, 0.0]
            x0_seq = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            for k in range(len(w_seq)):
                w = w_seq[k]
                x0 = x0_seq[k]

                spk = self._step(neuron, k, x=x0 * u.pA, delta=w * u.pA if w != 0.0 else None)

                # NEST update order:
                # 1. Evolve V_m
                v = v * P22_expm1 + v + iex * P21ex + iin * P21in + (I_e + i0) * P20

                # 2. Evolve adaptive threshold
                vth1 *= P11th
                vth2 *= P22th

                # 3. Decay PSCs and add spikes
                iex *= P11ex
                iin *= P11in
                iex += max(w, 0.0)
                iin += min(w, 0.0)

                # 4-5. Spike detection
                spike_ref = False
                if r == 0:
                    if v >= omega_rel + vth1 + vth2:
                        spike_ref = True
                        r = refr
                        vth1 += alpha_1
                        vth2 += alpha_2
                else:
                    r -= 1

                # 6. Update i_0
                i0 = x0

                self.assertEqual(self._is_spike(spk), spike_ref,
                                 msg=f'Spike mismatch at step {k}')
                self.assertAlmostEqual(
                    float((neuron.V.value / u.mV)[0]), v + params['E_L'], delta=1e-11,
                    msg=f'V_m mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.i_syn_ex.value / u.pA)[0]), iex, delta=1e-11,
                    msg=f'i_syn_ex mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.i_syn_in.value / u.pA)[0]), iin, delta=1e-11,
                    msg=f'i_syn_in mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.V_th_1.value / u.mV)[0]), vth1, delta=1e-11,
                    msg=f'V_th_1 mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.V_th_2.value / u.mV)[0]), vth2, delta=1e-11,
                    msg=f'V_th_2 mismatch at step {k}'
                )
                self.assertEqual(
                    int(neuron.refractory_step_count.value[0]), r,
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

            vth1 = 20.0
            vth2 = 5.0

            for k in range(50):
                self._step(neuron, k)
                vth1 *= P11th
                vth2 *= P22th

                self.assertAlmostEqual(
                    float((neuron.V_th_1.value / u.mV)[0]), vth1, delta=1e-11,
                    msg=f'V_th_1 decay mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.V_th_2.value / u.mV)[0]), vth2, delta=1e-11,
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
