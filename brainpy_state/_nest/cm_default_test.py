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
Tests for cm_default compartmental neuron model.

Test coverage:
- Default parameter values
- Passive single-compartment dynamics
- Passive multi-compartment dynamics (cable equation)
- Ion channel dynamics (Na, K)
- Synaptic receptor dynamics (AMPA, GABA, NMDA, AMPA_NMDA)
- Spike detection via threshold crossing
- Crank-Nicolson matrix solver accuracy
- Comparison against NEST simulation (where available)
- Edge cases (single compartment, deep tree, no channels)
"""

import math
import unittest

from brainpy_state._nest.cm_default import (
    cm_default,
    _NaChannel,
    _KChannel,
    _AMPAReceptor,
    _GABAReceptor,
    _NMDAReceptor,
    _AMPA_NMDAReceptor,
    _compute_g_norm,
    _nmda_sigmoid,
)


class TestCmDefaultConstruction(unittest.TestCase):
    r"""Test model construction, compartment and receptor addition."""

    def test_default_V_th(self):
        model = cm_default()
        self.assertEqual(model.V_th, -55.0)

    def test_custom_V_th(self):
        model = cm_default(V_th=-40.0)
        self.assertEqual(model.V_th, -40.0)

    def test_add_root_compartment(self):
        model = cm_default()
        idx = model.add_compartment(-1)
        self.assertEqual(idx, 0)
        self.assertEqual(model.num_compartments, 1)

    def test_add_child_compartment(self):
        model = cm_default()
        model.add_compartment(-1)
        idx = model.add_compartment(0)
        self.assertEqual(idx, 1)
        self.assertEqual(model.num_compartments, 2)

    def test_add_multiple_children(self):
        model = cm_default()
        model.add_compartment(-1)
        model.add_compartment(0)
        model.add_compartment(0)
        model.add_compartment(1)
        self.assertEqual(model.num_compartments, 4)

    def test_add_root_twice_raises(self):
        model = cm_default()
        model.add_compartment(-1)
        with self.assertRaises(ValueError):
            model.add_compartment(-1)

    def test_add_to_nonexistent_parent_raises(self):
        model = cm_default()
        model.add_compartment(-1)
        with self.assertRaises(ValueError):
            model.add_compartment(5)

    def test_add_receptor(self):
        model = cm_default()
        model.add_compartment(-1)
        r_idx = model.add_receptor(0, 'AMPA')
        self.assertEqual(r_idx, 0)
        self.assertEqual(model.num_receptors, 1)

    def test_add_multiple_receptors(self):
        model = cm_default()
        model.add_compartment(-1)
        model.add_compartment(0)
        r0 = model.add_receptor(0, 'AMPA')
        r1 = model.add_receptor(0, 'GABA')
        r2 = model.add_receptor(1, 'NMDA')
        r3 = model.add_receptor(1, 'AMPA_NMDA')
        self.assertEqual(r0, 0)
        self.assertEqual(r1, 1)
        self.assertEqual(r2, 2)
        self.assertEqual(r3, 3)
        self.assertEqual(model.num_receptors, 4)

    def test_add_receptor_invalid_type_raises(self):
        model = cm_default()
        model.add_compartment(-1)
        with self.assertRaises(ValueError):
            model.add_receptor(0, 'INVALID')

    def test_add_receptor_invalid_compartment_raises(self):
        model = cm_default()
        model.add_compartment(-1)
        with self.assertRaises(ValueError):
            model.add_receptor(5, 'AMPA')

    def test_compartment_default_params(self):
        model = cm_default()
        model.add_compartment(-1)
        comp = model._get_compartment(0)
        self.assertEqual(comp.ca, 1.0)
        self.assertEqual(comp.gc, 0.01)
        self.assertEqual(comp.gl, 0.1)
        self.assertEqual(comp.el, -70.0)
        self.assertEqual(comp.v_comp, -70.0)

    def test_compartment_custom_params(self):
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 89.245,
            'g_L': 8.925,
            'e_L': -75.0,
            'v_comp': -75.0,
        })
        comp = model._get_compartment(0)
        self.assertAlmostEqual(comp.ca, 89.245)
        self.assertAlmostEqual(comp.gl, 8.925)
        self.assertAlmostEqual(comp.el, -75.0)
        self.assertAlmostEqual(comp.v_comp, -75.0)

    def test_compartment_v_comp_defaults_to_e_L(self):
        model = cm_default()
        model.add_compartment(-1, {'e_L': -65.0})
        comp = model._get_compartment(0)
        self.assertAlmostEqual(comp.v_comp, -65.0)


class TestPassiveSingleCompartment(unittest.TestCase):
    r"""Test passive (no ion channels) single compartment dynamics."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1  # ms

    def test_resting_potential_stable(self):
        r"""At rest, voltage should remain at e_L."""
        model = cm_default()
        model.add_compartment(-1, {'e_L': -70.0, 'v_comp': -70.0})
        model.pre_run_hook(self.dt)

        for _ in range(100):
            model.step()

        self.assertAlmostEqual(model.get_voltage(0), -70.0, places=10)

    def test_passive_decay_to_rest(self):
        r"""Voltage above rest should decay exponentially toward e_L."""
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -50.0,
        })
        model.pre_run_hook(self.dt)

        # Analytical Crank-Nicolson step for single passive compartment:
        # The system is: C/dt * V_new + g_L/2 * V_new = (C/dt - g_L/2) * V_old + g_L * e_L
        # => V_new = [(C/dt - g_L/2) * V_old + g_L * e_L] / (C/dt + g_L/2)
        C_m, g_L, e_L = 1.0, 0.1, -70.0
        v_old = -50.0
        for _ in range(100):
            v_new = ((C_m / self.dt - g_L / 2.0) * v_old + g_L * e_L) / (
                C_m / self.dt + g_L / 2.0
            )
            v_old = v_new

        model_v_old = -50.0
        model2 = cm_default()
        model2.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -50.0,
        })
        model2.pre_run_hook(self.dt)
        for _ in range(100):
            model2.step()

        self.assertAlmostEqual(model2.get_voltage(0), v_new, places=10)

    def test_dc_current_injection(self):
        r"""Constant current should drive voltage to steady state."""
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -70.0,
        })
        model.pre_run_hook(self.dt)

        # Inject current for many steps
        I_ext = 1.0  # nA
        for _ in range(10000):
            model.add_current(0, I_ext)
            model.step()

        # Steady state: g_L * (V - e_L) = I_ext => V = e_L + I_ext / g_L
        v_ss = -70.0 + I_ext / 0.1
        self.assertAlmostEqual(model.get_voltage(0), v_ss, places=2)


class TestPassiveMultiCompartment(unittest.TestCase):
    r"""Test passive multi-compartment dynamics and coupling."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1  # ms

    def test_two_compartments_voltage_coupling(self):
        r"""Two coupled compartments should equilibrate."""
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -50.0,
        })
        model.add_compartment(0, {
            'C_m': 1.0, 'g_C': 0.5, 'g_L': 0.1, 'e_L': -70.0,
            'v_comp': -70.0,
        })
        model.pre_run_hook(self.dt)

        for _ in range(100000):
            model.step()

        # Both should approach e_L
        v0 = model.get_voltage(0)
        v1 = model.get_voltage(1)
        self.assertAlmostEqual(v0, -70.0, places=3)
        self.assertAlmostEqual(v1, -70.0, places=3)

    def test_two_compartments_reference_step(self):
        r"""Verify first few steps match manual Crank-Nicolson computation."""
        C_m0 = 1.0
        g_L0 = 0.1
        e_L0 = -70.0
        v0_init = -60.0

        C_m1 = 1.0
        g_C1 = 0.5
        g_L1 = 0.1
        e_L1 = -70.0
        v1_init = -70.0

        dt = self.dt

        # Manual Crank-Nicolson for 2-compartment system
        # Compartment 0 (root): no parent, one child (1)
        # Compartment 1: parent is 0, no children

        v0 = v0_init
        v1 = v1_init

        for step in range(5):
            # Pre-compute constants
            ca0_dt = C_m0 / dt
            gl0_2 = g_L0 / 2.0
            gc1_2 = g_C1 / 2.0

            ca1_dt = C_m1 / dt
            gl1_2 = g_L1 / 2.0

            # Compartment 0: root, has child 1
            gg0 = ca0_dt + gl0_2 + gc1_2
            ff0 = (ca0_dt - gl0_2) * v0 + g_L0 * e_L0
            ff0 -= gc1_2 * (v0 - v1)

            # Compartment 1: child of 0
            gg1 = ca1_dt + gl1_2 + gc1_2
            hh1 = -gc1_2
            ff1 = (ca1_dt - gl1_2) * v1 + g_L1 * e_L1
            ff1 -= gc1_2 * (v1 - v0)

            # Down-sweep: compartment 1 is a leaf
            # io() for compartment 1:
            g_out = hh1 * hh1 / gg1
            f_out = ff1 * hh1 / gg1

            # gather into root (compartment 0)
            gg0 -= g_out
            ff0 -= f_out

            # Up-sweep: root with v_in = 0
            v0_new = ff0 / gg0

            # Compartment 1: v_in = v0_new
            v1_new = (ff1 - v0_new * hh1) / gg1

            v0 = v0_new
            v1 = v1_new

        # Now compare with model
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': C_m0, 'g_L': g_L0, 'e_L': e_L0, 'v_comp': v0_init,
        })
        model.add_compartment(0, {
            'C_m': C_m1, 'g_C': g_C1, 'g_L': g_L1, 'e_L': e_L1,
            'v_comp': v1_init,
        })
        model.pre_run_hook(dt)

        for _ in range(5):
            model.step()

        self.assertAlmostEqual(model.get_voltage(0), v0, places=10)
        self.assertAlmostEqual(model.get_voltage(1), v1, places=10)

    def test_three_compartments_branching(self):
        r"""Root with two children should solve correctly."""
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -50.0,
        })
        model.add_compartment(0, {
            'C_m': 0.5, 'g_C': 0.2, 'g_L': 0.05, 'e_L': -70.0,
            'v_comp': -70.0,
        })
        model.add_compartment(0, {
            'C_m': 0.5, 'g_C': 0.3, 'g_L': 0.05, 'e_L': -70.0,
            'v_comp': -70.0,
        })
        model.pre_run_hook(self.dt)

        # Just verify it runs without errors and converges
        for _ in range(100000):
            model.step()

        voltages = model.get_voltages()
        for v in voltages:
            self.assertAlmostEqual(v, -70.0, places=3)


class TestIonChannels(unittest.TestCase):
    r"""Test Na and K ion channel dynamics."""

    def test_na_channel_default_inactive(self):
        r"""Na channel with gbar=0 should return zero contribution."""
        na = _NaChannel(-70.0, gbar_Na=0.0)
        g, i = na.f_numstep(-70.0, 0.1)
        self.assertEqual(g, 0.0)
        self.assertEqual(i, 0.0)

    def test_k_channel_default_inactive(self):
        r"""K channel with gbar=0 should return zero contribution."""
        k = _KChannel(-70.0, gbar_K=0.0)
        g, i = k.f_numstep(-70.0, 0.1)
        self.assertEqual(g, 0.0)
        self.assertEqual(i, 0.0)

    def test_na_channel_steady_state_init(self):
        r"""Na channel state variables should be at steady state initially."""
        v = -70.0
        na = _NaChannel(v, gbar_Na=10.0)
        m_inf, _ = na._compute_statevar_m(v)
        h_inf, _ = na._compute_statevar_h(v)
        self.assertAlmostEqual(na.m_Na, m_inf, places=12)
        self.assertAlmostEqual(na.h_Na, h_inf, places=12)

    def test_k_channel_steady_state_init(self):
        r"""K channel state variable should be at steady state initially."""
        v = -70.0
        k = _KChannel(v, gbar_K=10.0)
        n_inf, _ = k._compute_statevar_n(v)
        self.assertAlmostEqual(k.n_K, n_inf, places=12)

    def test_na_m_inf_increases_with_depolarization(self):
        r"""Na m_inf should increase as voltage increases."""
        na = _NaChannel(-70.0)
        m_low, _ = na._compute_statevar_m(-70.0)
        m_high, _ = na._compute_statevar_m(-30.0)
        self.assertGreater(m_high, m_low)

    def test_na_h_inf_decreases_with_depolarization(self):
        r"""Na h_inf should decrease as voltage increases."""
        na = _NaChannel(-70.0)
        h_low, _ = na._compute_statevar_h(-70.0)
        h_high, _ = na._compute_statevar_h(-30.0)
        self.assertLess(h_high, h_low)

    def test_na_channel_returns_conductance_and_current(self):
        r"""Active Na channel should produce non-zero g_val, i_val."""
        na = _NaChannel(-40.0, gbar_Na=100.0)
        g, i = na.f_numstep(-40.0, 0.1)
        self.assertGreater(g, 0.0)
        self.assertNotEqual(i, 0.0)

    def test_k_channel_returns_conductance_and_current(self):
        r"""Active K channel should produce non-zero g_val, i_val."""
        k = _KChannel(-40.0, gbar_K=100.0)
        g, i = k.f_numstep(-40.0, 0.1)
        self.assertGreater(g, 0.0)
        self.assertNotEqual(i, 0.0)

    def test_na_singularity_at_minus_35_013(self):
        r"""Na m gate should handle singularity at V = -35.013."""
        na = _NaChannel(-35.013)
        m_inf, tau = na._compute_statevar_m(-35.013)
        self.assertTrue(math.isfinite(m_inf))
        self.assertTrue(math.isfinite(tau))
        self.assertGreater(tau, 0.0)

    def test_na_singularity_at_minus_50_013(self):
        r"""Na h gate should handle singularity near V = -50.013."""
        na = _NaChannel(-50.013)
        h_inf, tau = na._compute_statevar_h(-50.013)
        self.assertTrue(math.isfinite(h_inf))
        self.assertTrue(math.isfinite(tau))
        self.assertGreater(tau, 0.0)

    def test_k_singularity_at_25(self):
        r"""K n gate should handle singularity at V = 25."""
        k = _KChannel(25.0)
        n_inf, tau = k._compute_statevar_n(25.0)
        self.assertTrue(math.isfinite(n_inf))
        self.assertTrue(math.isfinite(tau))
        self.assertGreater(tau, 0.0)

    def test_active_compartment_produces_spike(self):
        r"""Compartment with active Na/K should produce action potential."""
        model = cm_default(V_th=-55.0)
        model.add_compartment(-1, {
            'C_m': 1.0,
            'g_L': 0.1,
            'e_L': -70.0,
            'v_comp': -70.0,
            'gbar_Na': 100.0,
            'e_Na': 50.0,
            'gbar_K': 30.0,
            'e_K': -85.0,
        })
        model.pre_run_hook(0.025)

        # Inject strong current to trigger spike
        spiked = False
        for _ in range(2000):
            model.add_current(0, 5.0)  # nA
            if model.step():
                spiked = True
                break

        self.assertTrue(spiked)


class TestSynapticReceptors(unittest.TestCase):
    r"""Test synaptic receptor dynamics."""

    def test_g_norm_computation(self):
        r"""Normalization constant should give unit peak conductance."""
        tau_r, tau_d = 0.2, 3.0
        g_norm = _compute_g_norm(tau_r, tau_d)
        self.assertGreater(g_norm, 0.0)
        self.assertTrue(math.isfinite(g_norm))

    def test_ampa_spike_response(self):
        r"""AMPA receptor should produce transient conductance after spike."""
        dt = 0.1
        rec = _AMPAReceptor(e_AMPA=0.0, tau_r_AMPA=0.2, tau_d_AMPA=3.0)
        rec.pre_run_hook(dt)

        # No spike: should return zero
        g, i = rec.f_numstep(-70.0, 0.0)
        self.assertAlmostEqual(g, 0.0, places=12)

        # Spike with weight 1.0 -- at the instant of the spike,
        # g_total = g_r + g_d = -s_val + s_val = 0 (by construction).
        # Conductance builds on the next step due to differential decay.
        rec.f_numstep(-70.0, 1.0)

        # After one more step (no spike), conductance should be > 0
        g, i = rec.f_numstep(-70.0, 0.0)
        g_total = rec.g_r + rec.g_d
        self.assertGreater(g_total, 0.0)

    def test_gaba_reversal(self):
        r"""GABA receptor should produce inhibitory current at rest."""
        dt = 0.1
        rec = _GABAReceptor(e_GABA=-80.0)
        rec.pre_run_hook(dt)

        # Spike at rest (-70 mV)
        rec.f_numstep(-70.0, 1.0)

        # After one more step, conductance should be > 0
        g, i = rec.f_numstep(-70.0, 0.0)
        g_total = rec.g_r + rec.g_d
        self.assertGreater(g_total, 0.0)
        # The g_val returned by f_numstep should also be positive
        self.assertGreater(g, 0.0)

    def test_nmda_sigmoid(self):
        r"""NMDA Mg2+ block should be near 1 at depolarized potentials."""
        # At very positive potential, block should be relieved
        B_pos, dB_pos = _nmda_sigmoid(40.0)
        self.assertGreater(B_pos, 0.9)

        # At very negative potential, block should be strong
        B_neg, dB_neg = _nmda_sigmoid(-80.0)
        self.assertLess(B_neg, 0.2)

        # Derivative should be positive
        B_0, dB_0 = _nmda_sigmoid(0.0)
        self.assertGreater(dB_0, 0.0)

    def test_nmda_receptor_voltage_dependence(self):
        r"""NMDA current should be smaller at negative potentials due to Mg block."""
        dt = 0.1
        # Create two NMDA receptors
        rec_dep = _NMDAReceptor()
        rec_dep.pre_run_hook(dt)
        rec_hyp = _NMDAReceptor()
        rec_hyp.pre_run_hook(dt)

        # Same spike (at instant of spike, g_total=0 for both)
        rec_dep.f_numstep(0.0, 1.0)
        rec_hyp.f_numstep(-70.0, 1.0)

        # After one more step, conductance builds up and Mg block matters
        g_dep, i_dep = rec_dep.f_numstep(0.0, 0.0)
        g_hyp, i_hyp = rec_hyp.f_numstep(-70.0, 0.0)

        # At 0 mV, Mg block is partially relieved, so g_val should be larger
        self.assertGreater(g_dep, g_hyp)

    def test_ampa_nmda_combined_receptor(self):
        r"""Combined AMPA_NMDA receptor should produce both components."""
        dt = 0.1
        rec = _AMPA_NMDAReceptor()
        rec.pre_run_hook(dt)

        # Apply spike (at instant, g_total=0 by construction)
        rec.f_numstep(-70.0, 1.0)

        # Check both AMPA and NMDA components are non-zero after spike
        self.assertNotEqual(rec.g_d_AMPA, 0.0)
        self.assertNotEqual(rec.g_d_NMDA, 0.0)

        # After one more step, the overall g_val should be positive
        g, i = rec.f_numstep(-70.0, 0.0)
        self.assertGreater(g, 0.0)

    def test_receptor_dual_exponential_shape(self):
        r"""AMPA conductance should rise then decay (dual exponential shape)."""
        dt = 0.01  # Fine dt for shape test
        rec = _AMPAReceptor(tau_r_AMPA=0.2, tau_d_AMPA=3.0)
        rec.pre_run_hook(dt)

        # Inject spike
        rec.f_numstep(-70.0, 1.0)

        # Record conductance over time
        conductances = []
        for _ in range(500):
            rec.f_numstep(-70.0, 0.0)
            g_total = rec.g_r + rec.g_d
            conductances.append(g_total)

        # Find peak
        peak_idx = conductances.index(max(conductances))
        # Peak should occur after some rise time
        self.assertGreater(peak_idx, 0)
        # Conductance should decay after peak
        self.assertGreater(conductances[peak_idx], conductances[-1])

    def test_receptor_state_access(self):
        r"""Model should expose receptor state variables."""
        model = cm_default()
        model.add_compartment(-1)
        r0 = model.add_receptor(0, 'AMPA')
        r1 = model.add_receptor(0, 'GABA')
        r2 = model.add_receptor(0, 'NMDA')
        r3 = model.add_receptor(0, 'AMPA_NMDA')

        state0 = model.get_receptor_state(r0)
        self.assertIn('g_r_AMPA', state0)
        self.assertIn('g_d_AMPA', state0)

        state1 = model.get_receptor_state(r1)
        self.assertIn('g_r_GABA', state1)

        state2 = model.get_receptor_state(r2)
        self.assertIn('g_r_NMDA', state2)

        state3 = model.get_receptor_state(r3)
        self.assertIn('g_r_AN_AMPA', state3)
        self.assertIn('g_r_AN_NMDA', state3)


class TestSpikeDetection(unittest.TestCase):
    r"""Test spike detection at root compartment."""

    def test_no_spike_below_threshold(self):
        r"""Voltage below threshold should not produce spike."""
        model = cm_default(V_th=-55.0)
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -60.0,
        })
        model.pre_run_hook(0.1)

        spike = model.step()
        self.assertFalse(spike)

    def test_spike_on_threshold_crossing(self):
        r"""Voltage crossing threshold from below should trigger spike."""
        model = cm_default(V_th=-55.0)
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -56.0,
        })
        model.pre_run_hook(0.1)

        # Inject enough current to cross threshold
        # Steady state with I: V = e_L + I/g_L
        # Need V >= -55: I >= (-55 - (-70)) * 0.1 = 1.5 nA
        spiked = False
        for _ in range(1000):
            model.add_current(0, 5.0)
            if model.step():
                spiked = True
                break

        self.assertTrue(spiked)

    def test_no_voltage_reset_after_spike(self):
        r"""cm_default should NOT reset voltage after spike (unlike IF models)."""
        model = cm_default(V_th=-55.0)
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.01, 'e_L': -70.0, 'v_comp': -56.0,
        })
        model.pre_run_hook(0.1)

        # Drive voltage just above threshold
        for _ in range(10000):
            model.add_current(0, 2.0)
            spike = model.step()
            if spike:
                # Voltage should still be >= V_th (no reset)
                v = model.get_voltage(0)
                self.assertGreaterEqual(v, -55.0)
                break

    def test_spike_requires_crossing_from_below(self):
        r"""Spike requires v_prev < V_th AND v_new >= V_th."""
        model = cm_default(V_th=-55.0)
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.01, 'e_L': -50.0, 'v_comp': -50.0,
        })
        model.pre_run_hook(0.1)

        # Voltage starts above threshold; should not spike on first step
        spike = model.step()
        self.assertFalse(spike)


class TestCrankNicolsonAccuracy(unittest.TestCase):
    r"""Test that the Crank-Nicolson scheme is correctly implemented."""

    def test_single_compartment_analytic(self):
        r"""Single passive compartment: V_new from CN should match formula."""
        C_m = 2.0
        g_L = 0.5
        e_L = -65.0
        v_init = -40.0
        dt = 0.1

        model = cm_default()
        model.add_compartment(-1, {
            'C_m': C_m, 'g_L': g_L, 'e_L': e_L, 'v_comp': v_init,
        })
        model.pre_run_hook(dt)

        # One step: analytic CN
        gg = C_m / dt + g_L / 2.0
        ff = (C_m / dt - g_L / 2.0) * v_init + g_L * e_L
        v_analytic = ff / gg

        model.step()
        self.assertAlmostEqual(model.get_voltage(0), v_analytic, places=12)

    def test_cn_stability_large_dt(self):
        r"""Crank-Nicolson should be stable even with large dt."""
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': 100.0,
        })
        dt = 10.0  # Very large dt
        model.pre_run_hook(dt)

        # Should not diverge
        for _ in range(100):
            model.step()
            v = model.get_voltage(0)
            self.assertTrue(math.isfinite(v))

        # Should converge toward e_L
        self.assertAlmostEqual(model.get_voltage(0), -70.0, places=1)


class TestMatrixSolver(unittest.TestCase):
    r"""Test the O(n) tree matrix solver."""

    def test_single_compartment_no_leafs(self):
        r"""Single compartment (root only) should solve directly."""
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -60.0,
        })
        model.pre_run_hook(0.1)
        model.step()

        # Should not crash and voltage should be finite
        self.assertTrue(math.isfinite(model.get_voltage(0)))

    def test_linear_chain(self):
        r"""Linear chain of compartments should solve correctly."""
        model = cm_default()
        n_comps = 5
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -50.0,
        })
        for i in range(1, n_comps):
            model.add_compartment(i - 1, {
                'C_m': 0.5, 'g_C': 0.3, 'g_L': 0.05, 'e_L': -70.0,
                'v_comp': -70.0,
            })
        model.pre_run_hook(0.1)

        for _ in range(50000):
            model.step()

        for i in range(n_comps):
            self.assertAlmostEqual(model.get_voltage(i), -70.0, places=3)

    def test_binary_tree(self):
        r"""Binary tree should solve correctly."""
        model = cm_default()
        #       0
        #      / \
        #     1   2
        #    / \
        #   3   4
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -50.0,
        })
        model.add_compartment(0, {
            'C_m': 0.5, 'g_C': 0.2, 'g_L': 0.05, 'e_L': -70.0,
            'v_comp': -70.0,
        })
        model.add_compartment(0, {
            'C_m': 0.5, 'g_C': 0.2, 'g_L': 0.05, 'e_L': -70.0,
            'v_comp': -70.0,
        })
        model.add_compartment(1, {
            'C_m': 0.3, 'g_C': 0.15, 'g_L': 0.03, 'e_L': -70.0,
            'v_comp': -70.0,
        })
        model.add_compartment(1, {
            'C_m': 0.3, 'g_C': 0.15, 'g_L': 0.03, 'e_L': -70.0,
            'v_comp': -70.0,
        })
        model.pre_run_hook(0.1)

        for _ in range(50000):
            model.step()

        for i in range(5):
            self.assertAlmostEqual(model.get_voltage(i), -70.0, places=3)


class TestNESTComparison(unittest.TestCase):
    r"""Compare against NEST simulation outputs."""

    def _run_reference_single_passive(self, C_m, g_L, e_L, v_init, dt, n_steps, I_ext=0.0):
        r"""Reference Crank-Nicolson for a single passive compartment."""
        v = v_init
        trace = [v]
        for _ in range(n_steps):
            gg = C_m / dt + g_L / 2.0
            ff = (C_m / dt - g_L / 2.0) * v + g_L * e_L + I_ext
            v = ff / gg
            trace.append(v)
        return trace

    def test_single_passive_vs_reference(self):
        r"""Single passive compartment should match reference exactly."""
        dt = 0.1
        C_m, g_L, e_L, v_init = 1.0, 0.1, -70.0, -50.0
        n_steps = 100

        ref = self._run_reference_single_passive(C_m, g_L, e_L, v_init, dt, n_steps)

        model = cm_default()
        model.add_compartment(-1, {
            'C_m': C_m, 'g_L': g_L, 'e_L': e_L, 'v_comp': v_init,
        })
        model.pre_run_hook(dt)

        model_trace = [v_init]
        for _ in range(n_steps):
            model.step()
            model_trace.append(model.get_voltage(0))

        for i in range(n_steps + 1):
            self.assertAlmostEqual(model_trace[i], ref[i], places=10,
                                   msg=f"Mismatch at step {i}")

    def test_single_passive_with_current_vs_reference(self):
        r"""Single passive compartment with DC current should match reference."""
        dt = 0.1
        C_m, g_L, e_L, v_init = 1.0, 0.1, -70.0, -70.0
        I_ext = 0.5  # nA
        n_steps = 50

        ref = self._run_reference_single_passive(
            C_m, g_L, e_L, v_init, dt, n_steps, I_ext
        )

        model = cm_default()
        model.add_compartment(-1, {
            'C_m': C_m, 'g_L': g_L, 'e_L': e_L, 'v_comp': v_init,
        })
        model.pre_run_hook(dt)

        model_trace = [v_init]
        for _ in range(n_steps):
            model.add_current(0, I_ext)
            model.step()
            model_trace.append(model.get_voltage(0))

        for i in range(n_steps + 1):
            self.assertAlmostEqual(model_trace[i], ref[i], places=10,
                                   msg=f"Mismatch at step {i}")

    def test_two_compartment_reference_trace(self):
        r"""Two-compartment passive system should match manual CN computation."""
        dt = 0.1
        C_m0, g_L0, e_L0, v0_init = 1.0, 0.1, -70.0, -50.0
        C_m1, g_C1, g_L1, e_L1, v1_init = 0.5, 0.3, 0.05, -70.0, -70.0

        # Manual reference
        v0, v1 = v0_init, v1_init
        ref_v0 = [v0]
        ref_v1 = [v1]

        for _ in range(50):
            ca0_dt = C_m0 / dt
            gl0_2 = g_L0 / 2.0
            gc1_2 = g_C1 / 2.0
            ca1_dt = C_m1 / dt
            gl1_2 = g_L1 / 2.0

            gg0 = ca0_dt + gl0_2 + gc1_2
            ff0 = (ca0_dt - gl0_2) * v0 + g_L0 * e_L0
            ff0 -= gc1_2 * (v0 - v1)

            gg1 = ca1_dt + gl1_2 + gc1_2
            hh1 = -gc1_2
            ff1 = (ca1_dt - gl1_2) * v1 + g_L1 * e_L1
            ff1 -= gc1_2 * (v1 - v0)

            # Down-sweep
            g_out = hh1 * hh1 / gg1
            f_out = ff1 * hh1 / gg1
            gg0 -= g_out
            ff0 -= f_out

            # Up-sweep
            v0 = ff0 / gg0
            v1 = (ff1 - v0 * hh1) / gg1

            ref_v0.append(v0)
            ref_v1.append(v1)

        # Model
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': C_m0, 'g_L': g_L0, 'e_L': e_L0, 'v_comp': v0_init,
        })
        model.add_compartment(0, {
            'C_m': C_m1, 'g_C': g_C1, 'g_L': g_L1, 'e_L': e_L1,
            'v_comp': v1_init,
        })
        model.pre_run_hook(dt)

        model_v0 = [v0_init]
        model_v1 = [v1_init]
        for _ in range(50):
            model.step()
            model_v0.append(model.get_voltage(0))
            model_v1.append(model.get_voltage(1))

        for i in range(51):
            self.assertAlmostEqual(model_v0[i], ref_v0[i], places=10,
                                   msg=f"V0 mismatch at step {i}")
            self.assertAlmostEqual(model_v1[i], ref_v1[i], places=10,
                                   msg=f"V1 mismatch at step {i}")

    def test_against_nest_passive_two_comp(self):
        r"""Compare against NEST cm_default for passive two-compartment model.

        Pre-computed NEST reference values for a two-compartment passive model
        with specific parameters over 10 steps at dt=0.1 ms.
        """
        try:
            import nest
        except ImportError:
            # Skip if NEST not available
            import pytest
            pytest.skip("NEST simulator not available")
            return

        dt = 0.1
        nest.ResetKernel()
        nest.set(resolution=dt)

        cm = nest.Create('cm_default')
        cm.compartments = [
            {'parent_idx': -1, 'params': {
                'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -50.0}},
            {'parent_idx': 0, 'params': {
                'C_m': 0.5, 'g_C': 0.3, 'g_L': 0.05, 'e_L': -70.0,
                'v_comp': -70.0}},
        ]

        mm = nest.Create('multimeter', params={
            'record_from': ['v_comp0', 'v_comp1'],
            'interval': dt,
        })
        nest.Connect(mm, cm)

        nest.Simulate(10 * dt + dt)

        events = mm.events
        nest_v0 = list(events['v_comp0'])
        nest_v1 = list(events['v_comp1'])

        # brainpy.state model
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -50.0,
        })
        model.add_compartment(0, {
            'C_m': 0.5, 'g_C': 0.3, 'g_L': 0.05, 'e_L': -70.0,
            'v_comp': -70.0,
        })
        model.pre_run_hook(dt)

        bp_v0 = []
        bp_v1 = []
        for _ in range(len(nest_v0)):
            model.step()
            bp_v0.append(model.get_voltage(0))
            bp_v1.append(model.get_voltage(1))

        for i in range(len(nest_v0)):
            self.assertAlmostEqual(
                bp_v0[i], nest_v0[i], places=8,
                msg=f"V0 mismatch at step {i}: bp={bp_v0[i]}, nest={nest_v0[i]}"
            )
            self.assertAlmostEqual(
                bp_v1[i], nest_v1[i], places=8,
                msg=f"V1 mismatch at step {i}: bp={bp_v1[i]}, nest={nest_v1[i]}"
            )

    def test_against_nest_with_ampa(self):
        r"""Compare against NEST with AMPA receptor."""
        try:
            import nest
        except ImportError:
            import pytest
            pytest.skip("NEST simulator not available")
            return

        dt = 0.1
        nest.ResetKernel()
        nest.set(resolution=dt)

        cm = nest.Create('cm_default')
        cm.compartments = [
            {'parent_idx': -1, 'params': {
                'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -70.0}},
        ]
        cm.receptors = [
            {'comp_idx': 0, 'receptor_type': 'AMPA', 'params': {
                'e_AMPA': 0.0, 'tau_r_AMPA': 0.2, 'tau_d_AMPA': 3.0}},
        ]

        sg = nest.Create('spike_generator', params={
            'spike_times': [1.0],
        })
        nest.Connect(sg, cm, syn_spec={
            'weight': 5.0, 'delay': dt, 'receptor_type': 0,
        })

        mm = nest.Create('multimeter', params={
            'record_from': ['v_comp0'],
            'interval': dt,
        })
        nest.Connect(mm, cm)

        nest.Simulate(20.0)

        events = mm.events
        nest_v0 = list(events['v_comp0'])

        # brainpy.state model
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -70.0,
        })
        r_idx = model.add_receptor(0, 'AMPA', {
            'e_AMPA': 0.0, 'tau_r_AMPA': 0.2, 'tau_d_AMPA': 3.0,
        })
        model.pre_run_hook(dt)

        # NEST delivers spike at t=1.0 + delay=0.1 -> arrives at step 11
        # (1-indexed steps in NEST, 0-indexed in our model)
        spike_step = int(round((1.0 + dt) / dt))

        bp_v0 = []
        for step in range(len(nest_v0)):
            if step == spike_step:
                model.add_spike(r_idx, 5.0)
            model.step()
            bp_v0.append(model.get_voltage(0))

        for i in range(len(nest_v0)):
            self.assertAlmostEqual(
                bp_v0[i], nest_v0[i], places=6,
                msg=f"V0 mismatch at step {i}: bp={bp_v0[i]}, nest={nest_v0[i]}"
            )

    def test_against_nest_with_na_k(self):
        r"""Compare against NEST with active Na/K channels."""
        try:
            import nest
        except ImportError:
            import pytest
            pytest.skip("NEST simulator not available")
            return

        dt = 0.025
        nest.ResetKernel()
        nest.set(resolution=dt)

        soma_params = {
            'C_m': 89.245535,
            'g_L': 8.924572508,
            'e_L': -75.0,
            'v_comp': -75.0,
            'gbar_Na': 4608.698576715,
            'e_Na': 60.0,
            'gbar_K': 956.112772900,
            'e_K': -90.0,
        }

        cm = nest.Create('cm_default')
        cm.compartments = [
            {'parent_idx': -1, 'params': soma_params},
        ]

        dc = nest.Create('dc_generator', params={'amplitude': 2.0})
        nest.Connect(dc, cm, syn_spec={
            'weight': 1.0, 'delay': dt, 'receptor_type': 0,
        })

        mm = nest.Create('multimeter', params={
            'record_from': ['v_comp0'],
            'interval': dt,
        })
        nest.Connect(mm, cm)

        nest.Simulate(10.0)

        events = mm.events
        nest_v0 = list(events['v_comp0'])

        # brainpy.state model
        model = cm_default()
        model.add_compartment(-1, soma_params)
        model.pre_run_hook(dt)

        # DC generator delivers current starting at dt with delay dt
        # So current arrives at step 2 (0-indexed) onwards
        dc_start_step = int(round(2 * dt / dt))

        bp_v0 = []
        for step in range(len(nest_v0)):
            if step >= dc_start_step:
                model.add_current(0, 2.0)
            model.step()
            bp_v0.append(model.get_voltage(0))

        for i in range(min(len(nest_v0), len(bp_v0))):
            self.assertAlmostEqual(
                bp_v0[i], nest_v0[i], places=4,
                msg=f"V0 mismatch at step {i}: bp={bp_v0[i]}, nest={nest_v0[i]}"
            )


class TestEdgeCases(unittest.TestCase):
    r"""Test edge cases and error handling."""

    def test_pre_run_hook_without_compartments_raises(self):
        model = cm_default()
        with self.assertRaises(ValueError):
            model.pre_run_hook(0.1)

    def test_get_voltage_out_of_range(self):
        model = cm_default()
        model.add_compartment(-1)
        model.pre_run_hook(0.1)
        # Out of range should return None
        self.assertIsNone(model._get_compartment(5))

    def test_large_tree(self):
        r"""Stress test with a deep linear chain."""
        model = cm_default()
        n_comps = 20
        model.add_compartment(-1, {'e_L': -70.0, 'v_comp': -70.0})
        for i in range(1, n_comps):
            model.add_compartment(i - 1, {
                'g_C': 0.1, 'e_L': -70.0, 'v_comp': -70.0
            })
        model.pre_run_hook(0.1)

        # Should run without error
        for _ in range(100):
            model.step()

        # All should be near rest
        for i in range(n_comps):
            self.assertAlmostEqual(model.get_voltage(i), -70.0, places=5)

    def test_multiple_spikes_to_receptor(self):
        r"""Multiple spikes to same receptor should accumulate."""
        model = cm_default()
        model.add_compartment(-1, {
            'C_m': 1.0, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -70.0,
        })
        r_idx = model.add_receptor(0, 'AMPA')
        model.pre_run_hook(0.1)

        # Add two spikes (they accumulate to weight=5.0)
        model.add_spike(r_idx, 3.0)
        model.add_spike(r_idx, 2.0)
        # At the spike step, g_total=0 (by dual-exponential construction),
        # so voltage doesn't change yet.
        model.step()

        # After one more step (no new spike), the conductance builds up
        # and drives the voltage above rest.
        model.step()

        v = model.get_voltage(0)
        self.assertGreater(v, -70.0)

    def test_receptor_custom_params(self):
        r"""Custom receptor parameters should be respected."""
        model = cm_default()
        model.add_compartment(-1)
        r_idx = model.add_receptor(0, 'AMPA', {
            'e_AMPA': -10.0,
            'tau_r_AMPA': 0.5,
            'tau_d_AMPA': 5.0,
        })
        rec = model._receptors[r_idx]
        self.assertAlmostEqual(rec.e_rev, -10.0)
        self.assertAlmostEqual(rec.tau_r, 0.5)
        self.assertAlmostEqual(rec.tau_d, 5.0)

    def test_ion_channel_state_access(self):
        r"""Should be able to read Na and K channel state variables."""
        model = cm_default()
        model.add_compartment(-1, {
            'gbar_Na': 100.0, 'gbar_K': 30.0, 'v_comp': -70.0,
        })
        model.pre_run_hook(0.1)

        m, h = model.get_na_state(0)
        self.assertTrue(0.0 <= m <= 1.0)
        self.assertTrue(0.0 <= h <= 1.0)

        n = model.get_k_state(0)
        self.assertTrue(0.0 <= n <= 1.0)


if __name__ == '__main__':
    unittest.main()
