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
Tests for amat2_psc_exp neuron model.

These tests verify that the implementation produces the same dynamics as NEST's
amat2_psc_exp model, including:
- Default parameter values matching NEST C++ source
- Parameter validation (constraint checks)
- Subthreshold membrane dynamics with exact integration
- Voltage-dependent threshold component (beta > 0)
- Spike generation and adaptive threshold updates
- Refractory period behavior
- Synaptic current responses
- Exact match with NEST reference data (spike times, V_m/V_th/V_th_v traces)
- Equivalence with mat2_psc_exp when beta=0
"""

import math
import unittest

import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

from brainpy_state._nest.amat2_psc_exp import amat2_psc_exp

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _compute_propagators(h, taum, tauE, tauI, tauV, c, beta, tau_1, tau_2):
    r"""Compute all propagator coefficients matching NEST pre_run_hook()."""
    eE = math.exp(-h / tauE)
    eI = math.exp(-h / tauI)
    em = math.exp(-h / taum)
    e1 = math.exp(-h / tau_1)
    e2 = math.exp(-h / tau_2)
    eV = math.exp(-h / tauV)

    P11 = eE
    P22 = eI
    P33 = em
    P44 = e1
    P55 = e2
    P66 = eV
    P77 = eV

    P30 = (taum - em * taum) / c
    P31 = ((eE - em) * tauE * taum) / (c * (tauE - taum))
    P32 = ((eI - em) * tauI * taum) / (c * (tauI - taum))

    P60 = (beta * (em - eV) * taum * tauV) / (c * (taum - tauV))
    P61 = (beta * tauE * taum * tauV * (eV * (-tauE + taum) + em * (tauE - tauV) + eE * (-taum + tauV))) \
          / (c * (tauE - taum) * (tauE - tauV) * (taum - tauV))
    P62 = (beta * tauI * taum * tauV * (eV * (-tauI + taum) + em * (tauI - tauV) + eI * (-taum + tauV))) \
          / (c * (tauI - taum) * (tauI - tauV) * (taum - tauV))
    P63 = (beta * (-em + eV) * tauV) / (taum - tauV)

    P70 = (beta * taum * tauV * (em * taum * tauV - eV * (h * (taum - tauV) + taum * tauV))) \
          / (c * (taum - tauV) ** 2)
    P71 = (beta * tauE * taum * tauV
           * ((em * taum * (tauE - tauV) ** 2 - eE * tauE * (taum - tauV) ** 2) * tauV
              - eV * (tauE - taum)
              * (h * (tauE - tauV) * (taum - tauV) + tauE * taum * tauV - tauV ** 3))) \
          / (c * (tauE - taum) * (tauE - tauV) ** 2 * (taum - tauV) ** 2)
    P72 = (beta * tauI * taum * tauV
           * ((em * taum * (tauI - tauV) ** 2 - eI * tauI * (taum - tauV) ** 2) * tauV
              - eV * (tauI - taum)
              * (h * (tauI - tauV) * (taum - tauV) + tauI * taum * tauV - tauV ** 3))) \
          / (c * (tauI - taum) * (tauI - tauV) ** 2 * (taum - tauV) ** 2)
    P73 = (beta * tauV * (-(em * taum * tauV) + eV * (h * (taum - tauV) + taum * tauV))) \
          / (taum - tauV) ** 2
    P76 = eV * h

    return dict(
        P11=P11, P22=P22, P33=P33, P44=P44, P55=P55, P66=P66, P77=P77,
        P30=P30, P31=P31, P32=P32,
        P60=P60, P61=P61, P62=P62, P63=P63,
        P70=P70, P71=P71, P72=P72, P73=P73, P76=P76,
    )


def _nest_reference_step(state, p, P, w_ex=0.0, w_in=0.0, x0_new=0.0):
    r"""Execute one NEST update step for amat2_psc_exp.

    Returns: (state, spiked)
    state is a dict with keys: V_rel, V_th_1, V_th_2, V_th_v, V_th_dv,
                                i_syn_ex, i_syn_in, i_0, r
    """
    V_rel = state['V_rel']
    V_th_1 = state['V_th_1']
    V_th_2 = state['V_th_2']
    V_th_v = state['V_th_v']
    V_th_dv = state['V_th_dv']
    i_syn_ex = state['i_syn_ex']
    i_syn_in = state['i_syn_in']
    i_0 = state['i_0']
    r = state['r']
    I_e = p['I_e']
    omega_rel = p['omega_rel']
    alpha_1 = p['alpha_1']
    alpha_2 = p['alpha_2']
    refr = p['refr']

    # Step 1: Evolve voltage-dependent threshold
    V_th_v_new = (I_e + i_0) * P['P70'] + i_syn_ex * P['P71'] + i_syn_in * P['P72'] \
                 + V_rel * P['P73'] + V_th_dv * P['P76'] + V_th_v * P['P77']
    V_th_dv_new = (I_e + i_0) * P['P60'] + i_syn_ex * P['P61'] + i_syn_in * P['P62'] \
                  + V_rel * P['P63'] + V_th_dv * P['P66']
    V_th_v = V_th_v_new
    V_th_dv = V_th_dv_new

    # Step 2: Evolve membrane potential
    V_rel = (I_e + i_0) * P['P30'] + i_syn_ex * P['P31'] + i_syn_in * P['P32'] + V_rel * P['P33']

    # Step 3: Decay adaptive threshold
    V_th_1 *= P['P44']
    V_th_2 *= P['P55']

    # Step 4: Decay PSCs and add spikes
    i_syn_ex *= P['P11']
    i_syn_in *= P['P22']
    i_syn_ex += w_ex
    i_syn_in += w_in

    # Step 5-6: Spike detection
    spiked = False
    if r == 0:
        if V_rel >= omega_rel + V_th_1 + V_th_2 + V_th_v:
            spiked = True
            r = refr
            V_th_1 += alpha_1
            V_th_2 += alpha_2
    else:
        r -= 1

    # Step 7: Update i_0
    i_0 = x0_new

    return dict(
        V_rel=V_rel, V_th_1=V_th_1, V_th_2=V_th_2, V_th_v=V_th_v,
        V_th_dv=V_th_dv, i_syn_ex=i_syn_ex, i_syn_in=i_syn_in, i_0=i_0, r=r,
    ), spiked


class TestAmat2PscExp(unittest.TestCase):
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
        r"""Verify that all default parameter values match NEST's amat2_psc_exp C++ defaults."""
        neuron = amat2_psc_exp(1)
        self.assertEqual(neuron.E_L, -70. * u.mV)
        self.assertEqual(neuron.C_m, 200. * u.pF)
        self.assertEqual(neuron.tau_m, 10. * u.ms)
        self.assertEqual(neuron.t_ref, 2. * u.ms)
        self.assertEqual(neuron.tau_syn_ex, 1. * u.ms)
        self.assertEqual(neuron.tau_syn_in, 3. * u.ms)
        self.assertEqual(neuron.I_e, 0. * u.pA)
        self.assertEqual(neuron.tau_1, 10. * u.ms)
        self.assertEqual(neuron.tau_2, 200. * u.ms)
        self.assertEqual(neuron.alpha_1, 10. * u.mV)
        self.assertEqual(neuron.alpha_2, 0. * u.mV)
        self.assertEqual(neuron.beta, 0. / u.ms)
        self.assertEqual(neuron.tau_v, 5. * u.ms)
        # omega default: E_L + omega_rel = -70 + 5 = -65 mV
        self.assertEqual(neuron.omega, -65. * u.mV)
        self.assertEqual(neuron.spk_reset, 'hard')

    # ------------------------------------------------------------------
    # Test 2: Parameter validation
    # ------------------------------------------------------------------
    def test_parameter_validation(self):
        r"""Test that invalid parameters raise ValueError."""
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, C_m=0.0 * u.pF)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, C_m=-1.0 * u.pF)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_m=0.0 * u.ms)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_syn_ex=0.0 * u.ms)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_syn_in=0.0 * u.ms)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, t_ref=0.0 * u.ms)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_1=0.0 * u.ms)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_2=0.0 * u.ms)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_v=0.0 * u.ms)
        # tau_m must differ from tau_syn_ex, tau_syn_in, tau_v
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_m=1.0 * u.ms, tau_syn_ex=1.0 * u.ms)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_m=3.0 * u.ms, tau_syn_in=3.0 * u.ms)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_m=5.0 * u.ms, tau_v=5.0 * u.ms)
        # tau_v must differ from tau_syn_ex, tau_syn_in
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_v=1.0 * u.ms, tau_syn_ex=1.0 * u.ms)
        with self.assertRaises(ValueError):
            amat2_psc_exp(1, tau_v=3.0 * u.ms, tau_syn_in=3.0 * u.ms)

    # ------------------------------------------------------------------
    # Test 3: State initialization
    # ------------------------------------------------------------------
    def test_state_initialization(self):
        r"""Verify that all state variables are initialized correctly."""
        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                V_initializer=braintools.init.Constant(-65.0 * u.mV),
            )
            neuron.init_state()

            self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), -65.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.V_th_1.value / u.mV)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.V_th_2.value / u.mV)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.V_th_v.value / u.mV)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.V_th_dv.value / u.mV)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.i_syn_ex.value / u.pA)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.i_syn_in.value / u.pA)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.i_0.value / u.pA)[0]), 0.0, delta=1e-12)
            self.assertEqual(int(neuron.refractory_step_count.value[0]), 0)

    # ------------------------------------------------------------------
    # Test 4: NEST reference spike times (beta=0, matching mat2_psc_exp)
    # ------------------------------------------------------------------
    def test_nest_reference_spike_times_beta0(self):
        r"""
        Reproduce the NEST test_amat2_psc_exp.py simulation.

        With beta=0 and mat2_psc_exp defaults, amat2_psc_exp should produce
        the same spike times as mat2_psc_exp: [11, 32, 54].
        """
        # mat2_psc_exp defaults
        mat2_defaults = dict(
            E_L=-70.0, C_m=100.0, tau_m=5.0, t_ref=2.0,
            tau_syn_ex=1.0, tau_syn_in=3.0, I_e=0.0,
            tau_1=10.0, tau_2=200.0, alpha_1=37.0, alpha_2=2.0,
        )
        # tau_v must differ from tau_m; irrelevant for beta=0
        tau_v = mat2_defaults['tau_m'] + 10.0

        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                E_L=mat2_defaults['E_L'] * u.mV,
                C_m=mat2_defaults['C_m'] * u.pF,
                tau_m=mat2_defaults['tau_m'] * u.ms,
                t_ref=mat2_defaults['t_ref'] * u.ms,
                tau_syn_ex=mat2_defaults['tau_syn_ex'] * u.ms,
                tau_syn_in=mat2_defaults['tau_syn_in'] * u.ms,
                I_e=mat2_defaults['I_e'] * u.pA,
                tau_1=mat2_defaults['tau_1'] * u.ms,
                tau_2=mat2_defaults['tau_2'] * u.ms,
                alpha_1=mat2_defaults['alpha_1'] * u.mV,
                alpha_2=mat2_defaults['alpha_2'] * u.mV,
                beta=0.0 / u.ms,
                tau_v=tau_v * u.ms,
                omega=-51.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            spike_times = []
            for k in range(80):
                dc = 2400.0 * u.pA if k >= 1 else 0.0 * u.pA
                spk = self._step(neuron, k, x=dc)
                if self._is_spike(spk):
                    spike_times.append(k + 1)  # NEST records at lag+1

            np.testing.assert_array_equal(spike_times, [11, 32, 54])

    # ------------------------------------------------------------------
    # Test 5: NEST reference V_m and V_th traces (beta=0)
    # ------------------------------------------------------------------
    def test_nest_reference_potentials_beta0(self):
        r"""
        Verify V_m and V_th traces match the NEST reference data from
        test_amat2_psc_exp.py for the first 21 time steps (beta=0 case).
        """
        expected = [
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

        mat2_defaults = dict(
            E_L=-70.0, C_m=100.0, tau_m=5.0, t_ref=2.0,
            tau_syn_ex=1.0, tau_syn_in=3.0, I_e=0.0,
            tau_1=10.0, tau_2=200.0, alpha_1=37.0, alpha_2=2.0,
        )
        tau_v = mat2_defaults['tau_m'] + 10.0

        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                E_L=mat2_defaults['E_L'] * u.mV,
                C_m=mat2_defaults['C_m'] * u.pF,
                tau_m=mat2_defaults['tau_m'] * u.ms,
                t_ref=mat2_defaults['t_ref'] * u.ms,
                tau_syn_ex=mat2_defaults['tau_syn_ex'] * u.ms,
                tau_syn_in=mat2_defaults['tau_syn_in'] * u.ms,
                I_e=mat2_defaults['I_e'] * u.pA,
                tau_1=mat2_defaults['tau_1'] * u.ms,
                tau_2=mat2_defaults['tau_2'] * u.ms,
                alpha_1=mat2_defaults['alpha_1'] * u.mV,
                alpha_2=mat2_defaults['alpha_2'] * u.mV,
                beta=0.0 / u.ms,
                tau_v=tau_v * u.ms,
                omega=-51.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            recorded_vm = []
            recorded_vth = []

            for k in range(21):
                dc = 2400.0 * u.pA if k >= 1 else 0.0 * u.pA
                self._step(neuron, k, x=dc)

                V_m = float((neuron.V.value / u.mV)[0])
                omega_abs = float(u.math.asarray(neuron.omega / u.mV))
                V_th_1 = float((neuron.V_th_1.value / u.mV)[0])
                V_th_2 = float((neuron.V_th_2.value / u.mV)[0])
                V_th_v = float((neuron.V_th_v.value / u.mV)[0])
                V_th = omega_abs + V_th_1 + V_th_2 + V_th_v

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
    # Test 6: NEST reference voltage-dependent threshold (beta > 0)
    # ------------------------------------------------------------------
    def test_voltage_dependent_threshold_beta_positive(self):
        r"""
        Compare simulation with beta > 0 to NEST reference data obtained
        with Mathematica (from NEST test_amat2_psc_exp.py).

        Subthreshold mode: V_m < V_th for t < 10 ms -> no spikes.
        """
        expected = [
            (10, -1.76209, 2.21168, 0.211679),
            (20, -1.54683, 2.71733, 0.717335),
            (30, -1.35205, 3.36815, 1.36815),
            (40, -1.1758, 4.06297, 2.06297),
            (50, -1.01633, 4.73557, 2.73557),
            (60, -0.872029, 5.34504, 3.34504),
            (70, -0.741463, 5.86852, 3.86852),
            (80, -0.623322, 6.29576, 4.29576),
            (90, -0.516424, 6.62509, 4.62509),
            (100, -0.419699, 6.86044, 4.86044),
        ]

        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                omega=2.0 * u.mV,
                beta=2.0 / u.ms,
                E_L=0.0 * u.mV,
                I_e=10.0 * u.pA,
                alpha_1=10.0 * u.mV,
                alpha_2=10.0 * u.mV,
                V_initializer=braintools.init.Constant(-2.0 * u.mV),
            )
            neuron.init_state()

            recorded = {}
            for k in range(100):
                self._step(neuron, k)

                step = k + 1
                if step in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    V_m = float((neuron.V.value / u.mV)[0])
                    omega_abs = float(u.math.asarray(neuron.omega / u.mV))
                    V_th_1 = float((neuron.V_th_1.value / u.mV)[0])
                    V_th_2 = float((neuron.V_th_2.value / u.mV)[0])
                    V_th_v_val = float((neuron.V_th_v.value / u.mV)[0])
                    V_th = omega_abs + V_th_1 + V_th_2 + V_th_v_val
                    recorded[step] = (V_m, V_th, V_th_v_val)

            for step, exp_vm, exp_vth, exp_vth_v in expected:
                actual_vm, actual_vth, actual_vth_v = recorded[step]
                self.assertAlmostEqual(
                    actual_vm, exp_vm, places=4,
                    msg=f'V_m mismatch at step {step}: got {actual_vm}, expected {exp_vm}'
                )
                self.assertAlmostEqual(
                    actual_vth, exp_vth, places=4,
                    msg=f'V_th mismatch at step {step}: got {actual_vth}, expected {exp_vth}'
                )
                self.assertAlmostEqual(
                    actual_vth_v, exp_vth_v, places=4,
                    msg=f'V_th_v mismatch at step {step}: got {actual_vth_v}, expected {exp_vth_v}'
                )

    # ------------------------------------------------------------------
    # Test 7: No voltage reset on spike
    # ------------------------------------------------------------------
    def test_no_voltage_reset_on_spike(self):
        r"""
        Verify that the membrane potential is NOT reset after a spike.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                omega=-51.0 * u.mV,
                I_e=2400.0 * u.pA,
                C_m=100.0 * u.pF,
                tau_m=5.0 * u.ms,
                tau_v=15.0 * u.ms,
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
            # V_m should NOT be reset to E_L
            self.assertNotAlmostEqual(V_after_spike, -70.0, places=3,
                                      msg='V_m appears to have been reset to E_L after spike')

    # ------------------------------------------------------------------
    # Test 8: Adaptive threshold jumps on spike
    # ------------------------------------------------------------------
    def test_threshold_jump_on_spike(self):
        r"""
        Verify that V_th_1 and V_th_2 jump by alpha_1 and alpha_2 on spike.
        """
        mat2_defaults = dict(
            E_L=-70.0, C_m=100.0, tau_m=5.0,
            tau_syn_ex=1.0, tau_syn_in=3.0,
            tau_1=10.0, tau_2=200.0,
        )
        tau_v = mat2_defaults['tau_m'] + 10.0

        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                E_L=mat2_defaults['E_L'] * u.mV,
                C_m=mat2_defaults['C_m'] * u.pF,
                tau_m=mat2_defaults['tau_m'] * u.ms,
                tau_syn_ex=mat2_defaults['tau_syn_ex'] * u.ms,
                tau_syn_in=mat2_defaults['tau_syn_in'] * u.ms,
                omega=-51.0 * u.mV,
                I_e=2400.0 * u.pA,
                alpha_1=10.0 * u.mV,
                alpha_2=5.0 * u.mV,
                beta=0.0 / u.ms,
                tau_v=tau_v * u.ms,
                tau_1=mat2_defaults['tau_1'] * u.ms,
                tau_2=mat2_defaults['tau_2'] * u.ms,
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

                    h = 0.1
                    P44 = math.exp(-h / mat2_defaults['tau_1'])
                    P55 = math.exp(-h / mat2_defaults['tau_2'])
                    expected_vth1 = v_th_1_before * P44 + 10.0
                    expected_vth2 = v_th_2_before * P55 + 5.0
                    self.assertAlmostEqual(v_th_1_after, expected_vth1, delta=1e-10)
                    self.assertAlmostEqual(v_th_2_after, expected_vth2, delta=1e-10)
                    break

            self.assertTrue(first_spike, 'No spike occurred during test')

    # ------------------------------------------------------------------
    # Test 9: Refractory period prevents spiking
    # ------------------------------------------------------------------
    def test_refractory_period(self):
        r"""
        Verify that the neuron cannot fire during the refractory period.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                omega=-51.0 * u.mV,
                I_e=2400.0 * u.pA,
                t_ref=2.0 * u.ms,
                C_m=100.0 * u.pF,
                tau_m=5.0 * u.ms,
                tau_v=15.0 * u.ms,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            spike_steps = []
            for k in range(80):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    spike_steps.append(k)

            self.assertGreaterEqual(len(spike_steps), 2)
            for i in range(1, len(spike_steps)):
                isi = spike_steps[i] - spike_steps[i - 1]
                self.assertGreaterEqual(isi, 20,
                                        f'ISI {isi} steps < refractory period 20 steps')

    # ------------------------------------------------------------------
    # Test 10: Synaptic current response
    # ------------------------------------------------------------------
    def test_synaptic_current_response(self):
        r"""
        Verify that excitatory and inhibitory synaptic currents follow
        exponential decay after spike input.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                I_e=0.0 * u.pA,
                omega=100.0 * u.mV,  # unreachable threshold
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            h = 0.1
            tau_ex = 1.0
            tau_in = 3.0
            P11 = math.exp(-h / tau_ex)
            P22 = math.exp(-h / tau_in)

            # Step 0: inject excitatory spike
            self._step(neuron, 0, delta=100.0 * u.pA)
            iex = 100.0
            actual_iex = float((neuron.i_syn_ex.value / u.pA)[0])
            self.assertAlmostEqual(actual_iex, iex, delta=1e-11)

            for k in range(1, 10):
                self._step(neuron, k)
                iex *= P11
                actual_iex = float((neuron.i_syn_ex.value / u.pA)[0])
                self.assertAlmostEqual(actual_iex, iex, delta=1e-10,
                                       msg=f'i_syn_ex mismatch at step {k}')

            # Inject inhibitory spike
            self._step(neuron, 10, delta=-50.0 * u.pA)
            iin = -50.0
            actual_iin = float((neuron.i_syn_in.value / u.pA)[0])
            self.assertAlmostEqual(actual_iin, iin, delta=1e-11)

            for k in range(11, 20):
                self._step(neuron, k)
                iin *= P22
                actual_iin = float((neuron.i_syn_in.value / u.pA)[0])
                self.assertAlmostEqual(actual_iin, iin, delta=1e-10,
                                       msg=f'i_syn_in mismatch at step {k}')

    # ------------------------------------------------------------------
    # Test 11: Full step-by-step equation match (beta=0)
    # ------------------------------------------------------------------
    def test_step_equations_match_reference_beta0(self):
        r"""
        Verify internal state variables step-by-step against the exact
        NEST update equations with beta=0.
        """
        with brainstate.environ.context(dt=self.dt):
            params = dict(
                E_L=-70.0, C_m=100.0, tau_m=5.0, t_ref=0.3,
                tau_syn_ex=1.0, tau_syn_in=3.0, I_e=40.0,
                omega_abs=-51.0,
                tau_1=10.0, tau_2=200.0,
                alpha_1=37.0, alpha_2=2.0,
                beta=0.0, tau_v=15.0,
            )
            neuron = amat2_psc_exp(
                1,
                E_L=params['E_L'] * u.mV,
                C_m=params['C_m'] * u.pF,
                tau_m=params['tau_m'] * u.ms,
                t_ref=params['t_ref'] * u.ms,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                omega=params['omega_abs'] * u.mV,
                tau_1=params['tau_1'] * u.ms,
                tau_2=params['tau_2'] * u.ms,
                alpha_1=params['alpha_1'] * u.mV,
                alpha_2=params['alpha_2'] * u.mV,
                beta=params['beta'] / u.ms,
                tau_v=params['tau_v'] * u.ms,
                V_initializer=braintools.init.Constant(-67.0 * u.mV),
            )
            neuron.init_state()

            h = 0.1
            omega_rel = params['omega_abs'] - params['E_L']
            P = _compute_propagators(
                h, params['tau_m'], params['tau_syn_ex'], params['tau_syn_in'],
                params['tau_v'], params['C_m'], params['beta'],
                params['tau_1'], params['tau_2'],
            )
            p = dict(
                I_e=params['I_e'], omega_rel=omega_rel,
                alpha_1=params['alpha_1'], alpha_2=params['alpha_2'],
                refr=int(math.ceil(params['t_ref'] / h)),
            )
            state = dict(
                V_rel=-67.0 - params['E_L'], V_th_1=0.0, V_th_2=0.0,
                V_th_v=0.0, V_th_dv=0.0,
                i_syn_ex=0.0, i_syn_in=0.0, i_0=0.0, r=0,
            )

            w_seq = [0.0, 30.0, -15.0, 0.0, 0.0, 20.0, -10.0, 0.0, 0.0, 0.0]
            x0_seq = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            for k in range(len(w_seq)):
                w = w_seq[k]
                x0 = x0_seq[k]

                spk = self._step(neuron, k, x=x0 * u.pA, delta=w * u.pA if w != 0.0 else None)

                w_ex = max(w, 0.0)
                w_in = min(w, 0.0)
                state, spike_ref = _nest_reference_step(state, p, P, w_ex, w_in, x0)

                self.assertEqual(self._is_spike(spk), spike_ref,
                                 msg=f'Spike mismatch at step {k}')
                self.assertAlmostEqual(
                    float((neuron.V.value / u.mV)[0]), state['V_rel'] + params['E_L'], delta=1e-11,
                    msg=f'V_m mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.i_syn_ex.value / u.pA)[0]), state['i_syn_ex'], delta=1e-11,
                    msg=f'i_syn_ex mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.i_syn_in.value / u.pA)[0]), state['i_syn_in'], delta=1e-11,
                    msg=f'i_syn_in mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.V_th_1.value / u.mV)[0]), state['V_th_1'], delta=1e-11,
                    msg=f'V_th_1 mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.V_th_2.value / u.mV)[0]), state['V_th_2'], delta=1e-11,
                    msg=f'V_th_2 mismatch at step {k}'
                )
                self.assertEqual(
                    int(neuron.refractory_step_count.value[0]), state['r'],
                    msg=f'Refractory count mismatch at step {k}'
                )

    # ------------------------------------------------------------------
    # Test 12: Full step-by-step equation match (beta > 0)
    # ------------------------------------------------------------------
    def test_step_equations_match_reference_beta_positive(self):
        r"""
        Verify internal state variables step-by-step against the exact
        NEST update equations with beta > 0.
        """
        with brainstate.environ.context(dt=self.dt):
            params = dict(
                E_L=0.0, C_m=200.0, tau_m=10.0, t_ref=2.0,
                tau_syn_ex=1.0, tau_syn_in=3.0, I_e=10.0,
                omega_abs=2.0,
                tau_1=10.0, tau_2=200.0,
                alpha_1=10.0, alpha_2=0.0,
                beta=2.0, tau_v=5.0,
            )
            neuron = amat2_psc_exp(
                1,
                E_L=params['E_L'] * u.mV,
                C_m=params['C_m'] * u.pF,
                tau_m=params['tau_m'] * u.ms,
                t_ref=params['t_ref'] * u.ms,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                omega=params['omega_abs'] * u.mV,
                tau_1=params['tau_1'] * u.ms,
                tau_2=params['tau_2'] * u.ms,
                alpha_1=params['alpha_1'] * u.mV,
                alpha_2=params['alpha_2'] * u.mV,
                beta=params['beta'] / u.ms,
                tau_v=params['tau_v'] * u.ms,
                V_initializer=braintools.init.Constant(-2.0 * u.mV),
            )
            neuron.init_state()

            h = 0.1
            omega_rel = params['omega_abs'] - params['E_L']
            P = _compute_propagators(
                h, params['tau_m'], params['tau_syn_ex'], params['tau_syn_in'],
                params['tau_v'], params['C_m'], params['beta'],
                params['tau_1'], params['tau_2'],
            )
            p = dict(
                I_e=params['I_e'], omega_rel=omega_rel,
                alpha_1=params['alpha_1'], alpha_2=params['alpha_2'],
                refr=int(math.ceil(params['t_ref'] / h)),
            )
            state = dict(
                V_rel=-2.0 - params['E_L'], V_th_1=0.0, V_th_2=0.0,
                V_th_v=0.0, V_th_dv=0.0,
                i_syn_ex=0.0, i_syn_in=0.0, i_0=0.0, r=0,
            )

            for k in range(100):
                spk = self._step(neuron, k)
                state, spike_ref = _nest_reference_step(state, p, P)

                self.assertEqual(self._is_spike(spk), spike_ref,
                                 msg=f'Spike mismatch at step {k}')
                self.assertAlmostEqual(
                    float((neuron.V.value / u.mV)[0]), state['V_rel'] + params['E_L'], delta=1e-11,
                    msg=f'V_m mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.V_th_v.value / u.mV)[0]), state['V_th_v'], delta=1e-11,
                    msg=f'V_th_v mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.V_th_dv.value / u.mV)[0]), state['V_th_dv'], delta=1e-11,
                    msg=f'V_th_dv mismatch at step {k}'
                )

    # ------------------------------------------------------------------
    # Test 13: Threshold decay without spikes
    # ------------------------------------------------------------------
    def test_threshold_decay_without_spikes(self):
        r"""
        Verify that V_th_1 and V_th_2 decay exponentially when no spikes
        occur.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                I_e=0.0 * u.pA,
                omega=100.0 * u.mV,  # unreachable threshold
                tau_1=10.0 * u.ms,
                tau_2=200.0 * u.ms,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            neuron.V_th_1.value = jax.numpy.array([20.0]) * u.mV
            neuron.V_th_2.value = jax.numpy.array([5.0]) * u.mV

            h = 0.1
            P44 = math.exp(-h / 10.0)
            P55 = math.exp(-h / 200.0)

            vth1 = 20.0
            vth2 = 5.0

            for k in range(50):
                self._step(neuron, k)
                vth1 *= P44
                vth2 *= P55

                self.assertAlmostEqual(
                    float((neuron.V_th_1.value / u.mV)[0]), vth1, delta=1e-11,
                    msg=f'V_th_1 decay mismatch at step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.V_th_2.value / u.mV)[0]), vth2, delta=1e-11,
                    msg=f'V_th_2 decay mismatch at step {k}'
                )

    # ------------------------------------------------------------------
    # Test 14: V_th_v and V_th_dv remain zero when beta=0
    # ------------------------------------------------------------------
    def test_voltage_dependent_threshold_zero_when_beta0(self):
        r"""
        When beta=0, V_th_v and V_th_dv should remain exactly zero.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                beta=0.0 / u.ms,
                I_e=100.0 * u.pA,
                omega=100.0 * u.mV,  # no spikes
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            for k in range(50):
                self._step(neuron, k)
                self.assertAlmostEqual(
                    float((neuron.V_th_v.value / u.mV)[0]), 0.0, delta=1e-15,
                    msg=f'V_th_v should be 0 when beta=0, step {k}'
                )
                self.assertAlmostEqual(
                    float((neuron.V_th_dv.value / u.mV)[0]), 0.0, delta=1e-15,
                    msg=f'V_th_dv should be 0 when beta=0, step {k}'
                )

    # ------------------------------------------------------------------
    # Test 15: Subthreshold dynamics (exact integration, beta=0)
    # ------------------------------------------------------------------
    def test_subthreshold_dynamics_beta0(self):
        r"""
        Verify subthreshold membrane dynamics match the exact integration
        equations when no spikes occur (beta=0).
        """
        with brainstate.environ.context(dt=self.dt):
            params = dict(
                E_L=-70.0, C_m=200.0, tau_m=10.0,
                tau_syn_ex=1.0, tau_syn_in=3.0,
                I_e=50.0, tau_v=5.0, beta=0.0,
            )
            neuron = amat2_psc_exp(
                1,
                E_L=params['E_L'] * u.mV,
                C_m=params['C_m'] * u.pF,
                tau_m=params['tau_m'] * u.ms,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                beta=params['beta'] / u.ms,
                tau_v=params['tau_v'] * u.ms,
                omega=100.0 * u.mV,  # unreachable
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            h = 0.1
            taum = params['tau_m']
            c = params['C_m']
            tauE = params['tau_syn_ex']
            tauI = params['tau_syn_in']
            I_e = params['I_e']

            em = math.exp(-h / taum)
            eE = math.exp(-h / tauE)
            eI = math.exp(-h / tauI)
            P30 = (taum - em * taum) / c
            P31 = ((eE - em) * tauE * taum) / (c * (tauE - taum))
            P32 = ((eI - em) * tauI * taum) / (c * (tauI - taum))
            P33 = em

            v = 0.0  # V_m relative to E_L
            i0 = 0.0
            iex = 0.0
            iin = 0.0

            for k in range(30):
                self._step(neuron, k)

                v = (I_e + i0) * P30 + iex * P31 + iin * P32 + v * P33

                iex *= eE
                iin *= eI

                actual_v = float((neuron.V.value / u.mV)[0])
                self.assertAlmostEqual(actual_v, v + params['E_L'], delta=1e-11,
                                       msg=f'V_m mismatch at step {k}')

    # ------------------------------------------------------------------
    # Test 16: Current input one-step delay
    # ------------------------------------------------------------------
    def test_current_input_one_step_delay(self):
        r"""
        Verify that external current input has one-step delay (ring buffer).
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                E_L=0.0 * u.mV,
                C_m=200.0 * u.pF,
                tau_m=10.0 * u.ms,
                I_e=0.0 * u.pA,
                omega=1e9 * u.mV,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            # Step 0: inject current; V should stay at 0 (current buffered)
            self._step(neuron, 0, x=100.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.0 * u.mV))

            # Step 1: now current takes effect
            self._step(neuron, 1, x=0.0 * u.pA)
            actual_v = float((neuron.V.value / u.mV)[0])
            # V should have evolved: V = 0 * P33 + 100 * P30 = 100 * P30
            taum = 10.0
            c = 200.0
            em = math.exp(-0.1 / taum)
            P30 = (taum - em * taum) / c
            expected_v = 100.0 * P30
            self.assertAlmostEqual(actual_v, expected_v, delta=1e-11)

    # ------------------------------------------------------------------
    # Test 17: Spike routing (positive/negative weights)
    # ------------------------------------------------------------------
    def test_spike_weight_routing(self):
        r"""
        Positive spike weights go to excitatory; negative to inhibitory.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = amat2_psc_exp(
                1,
                omega=1e9 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, delta=5.0 * u.pA)
            self.assertAlmostEqual(float((neuron.i_syn_ex.value / u.pA)[0]), 5.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.i_syn_in.value / u.pA)[0]), 0.0, delta=1e-12)

        with brainstate.environ.context(dt=self.dt):
            neuron2 = amat2_psc_exp(
                1,
                omega=1e9 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron2.init_state()

            self._step(neuron2, 0, delta=-3.0 * u.pA)
            self.assertAlmostEqual(float((neuron2.i_syn_ex.value / u.pA)[0]), 0.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron2.i_syn_in.value / u.pA)[0]), -3.0, delta=1e-12)


if __name__ == '__main__':
    unittest.main()
