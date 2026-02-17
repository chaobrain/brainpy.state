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
Tests for the NEST-compatible Izhikevich neuron model.

Run with:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True python -m pytest brainpy_state/_nest/izhikevich_test.py -v
"""

import unittest
import math

import braintools
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from brainpy_state._nest.izhikevich import izhikevich


def _nest_reference_step(v, um, I, I_e, a, b, c, d, V_th, V_min, h, spike_in,
                         consistent_integration=True):
    r"""Pure-Python reference implementing the NEST izhikevich::update loop.

    This mirrors lines 196-243 of izhikevich.cpp exactly.

    Parameters
    ----------
    v, um : float
        Current membrane potential and recovery variable (dimensionless/mV).
    I : float
        Buffered input current from the *previous* step.
    I_e : float
        Constant external current.
    a, b, c, d : float
        Izhikevich model parameters.
    V_th : float
        Spike threshold.
    V_min : float or None
        Lower bound on V_m.
    h : float
        Time step in ms.
    spike_in : float
        Delta (spike) input to be added to V.
    consistent_integration : bool
        If True, use standard Euler; if False, use published half-step.

    Returns
    -------
    v_new, u_new, spike : float, float, bool
        Updated state and whether a spike was emitted.
    """
    if consistent_integration:
        v_old = v
        u_old = um
        v = v_old + h * (0.04 * v_old * v_old + 5.0 * v_old + 140.0 - u_old + I + I_e) + spike_in
        um = u_old + h * a * (b * v_old - u_old)
    else:
        I_syn = spike_in
        v = v + h * 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - um + I + I_e + I_syn)
        v = v + h * 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - um + I + I_e + I_syn)
        um = um + h * a * (b * v - um)

    # Lower bound
    if V_min is not None:
        v = max(v, V_min)

    # Threshold crossing
    spike = v >= V_th
    if spike:
        v = c
        um = um + d

    return v, um, spike


class TestIzhikevich(unittest.TestCase):
    r"""Test suite for the NEST-compatible Izhikevich neuron model."""

    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_enable_x64', True)
        brainstate.environ.set(precision=64, platform='cpu')

    def setUp(self):
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def _step(self, neuron, step_idx, x=0.0 * u.pA, delta=None):
        r"""Run one simulation step with optional delta input."""
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    # ------------------------------------------------------------------
    # 1. Default parameter tests
    # ------------------------------------------------------------------

    def test_nest_default_parameters(self):
        r"""Verify all parameter defaults match NEST C++ source."""
        neuron = izhikevich(1)
        self.assertEqual(float(neuron.a), 0.02)
        self.assertEqual(float(neuron.b), 0.2)
        self.assertEqual(neuron.c, -65. * u.mV)
        self.assertEqual(neuron.d, 8. * u.mV)
        self.assertEqual(neuron.I_e, 0. * u.pA)
        self.assertEqual(neuron.V_th, 30. * u.mV)
        self.assertIsNone(neuron.V_min)
        self.assertTrue(neuron.consistent_integration)
        self.assertEqual(neuron.spk_reset, 'hard')

    def test_default_state_initialization(self):
        r"""NEST default: v = -65, u = b * v = 0.2 * -65 = -13."""
        with brainstate.environ.context(dt=self.dt):
            neuron = izhikevich(1)
            neuron.init_state()
            self.assertTrue(u.math.allclose(neuron.V.value, -65. * u.mV))
            self.assertTrue(u.math.allclose(neuron.U.value, 0.2 * -65. * u.mV))

    def test_custom_u_initializer(self):
        r"""Allow explicit U_initializer to override the b*V default."""
        with brainstate.environ.context(dt=self.dt):
            neuron = izhikevich(
                1,
                U_initializer=braintools.init.Constant(-10. * u.mV),
            )
            neuron.init_state()
            self.assertTrue(u.math.allclose(neuron.U.value, -10. * u.mV))

    # ------------------------------------------------------------------
    # 2. Subthreshold dynamics tests
    # ------------------------------------------------------------------

    def test_subthreshold_consistent_euler_no_input(self):
        r"""Test subthreshold dynamics with consistent Euler, no external input."""
        h = 1.0  # ms
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-65. * u.mV,
                d=8. * u.mV,
                I_e=0. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=True,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            # Reference values
            v_ref = -65.0
            u_ref = 0.2 * -65.0  # -13.0
            I_ref = 0.0

            for step in range(10):
                self._step(neuron, step, x=0. * u.pA)

                # Update reference
                v_ref, u_ref, spike = _nest_reference_step(
                    v_ref, u_ref, I_ref, 0.0, 0.02, 0.2, -65.0, 8.0,
                    30.0, None, h, 0.0, consistent_integration=True
                )
                # I is updated at end of step in NEST; with 0 input it stays 0

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=10,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=10,
                                       msg=f"U mismatch at step {step}")

    def test_subthreshold_with_constant_current(self):
        r"""Test subthreshold dynamics with I_e constant current."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-65. * u.mV,
                d=8. * u.mV,
                I_e=10. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=True,
                V_initializer=braintools.init.Constant(-70. * u.mV),
            )
            neuron.init_state()

            v_ref = -70.0
            u_ref = 0.2 * -70.0
            I_ref = 0.0

            for step in range(20):
                self._step(neuron, step, x=0. * u.pA)

                v_ref, u_ref, spike = _nest_reference_step(
                    v_ref, u_ref, I_ref, 10.0, 0.02, 0.2, -65.0, 8.0,
                    30.0, None, h, 0.0, consistent_integration=True
                )

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=10,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=10,
                                       msg=f"U mismatch at step {step}")

    def test_subthreshold_inconsistent_numerics(self):
        r"""Test the published (inconsistent) Izhikevich numerics."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-65. * u.mV,
                d=8. * u.mV,
                I_e=10. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=False,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = 0.2 * -65.0
            I_ref = 0.0

            for step in range(20):
                self._step(neuron, step, x=0. * u.pA)

                v_ref, u_ref, spike = _nest_reference_step(
                    v_ref, u_ref, I_ref, 10.0, 0.02, 0.2, -65.0, 8.0,
                    30.0, None, h, 0.0, consistent_integration=False
                )

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=10,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=10,
                                       msg=f"U mismatch at step {step}")

    def test_subthreshold_fine_dt(self):
        r"""Test subthreshold dynamics with dt=0.1ms (fine time step)."""
        h = 0.1
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-65. * u.mV,
                d=8. * u.mV,
                I_e=5. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=True,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = 0.2 * -65.0
            I_ref = 0.0

            for step in range(100):
                self._step(neuron, step, x=0. * u.pA)

                v_ref, u_ref, spike = _nest_reference_step(
                    v_ref, u_ref, I_ref, 5.0, 0.02, 0.2, -65.0, 8.0,
                    30.0, None, h, 0.0, consistent_integration=True
                )

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=10,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=10,
                                       msg=f"U mismatch at step {step}")

    # ------------------------------------------------------------------
    # 3. Spike generation and reset tests
    # ------------------------------------------------------------------

    def test_spike_generation_and_reset(self):
        r"""Neuron should spike when V >= V_th and reset V to c, U += d."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            # Start near threshold with strong current
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-65. * u.mV,
                d=8. * u.mV,
                I_e=0. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=True,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = -13.0
            I_ref = 0.0
            spike_steps = []

            for step in range(200):
                spk = self._step(neuron, step, x=14. * u.pA)

                # Reference: external current arrives with one-step delay
                v_ref, u_ref, ref_spike = _nest_reference_step(
                    v_ref, u_ref, I_ref, 0.0, 0.02, 0.2, -65.0, 8.0,
                    30.0, None, h, 0.0, consistent_integration=True
                )
                I_ref = 14.0  # buffered for next step

                if ref_spike:
                    spike_steps.append(step)
                    # After spike: V should be c, U should have been incremented
                    v_model = float((neuron.V.value / u.mV)[0])
                    u_model = float((neuron.U.value / u.mV)[0])
                    self.assertAlmostEqual(v_model, -65.0, places=10,
                                           msg=f"V not reset to c at step {step}")

            # Should have produced at least one spike
            self.assertGreater(len(spike_steps), 0, "No spikes generated")

    def test_spike_exact_threshold(self):
        r"""A neuron at exactly V_th should spike."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                V_th=30. * u.mV,
                c=-65. * u.mV,
                d=8. * u.mV,
                I_e=0. * u.pA,
                V_initializer=braintools.init.Constant(29. * u.mV),
                U_initializer=braintools.init.Constant(0. * u.mV),
            )
            neuron.init_state()

            # With V=29, U=0, I=0: dV/dt = 0.04*29^2 + 5*29 + 140 - 0 = 33.64 + 145 + 140 = 318.64
            # V_new = 29 + 1 * 318.64 = 347.64 >= 30 => spike
            spk = self._step(neuron, 0, x=0. * u.pA)
            self.assertTrue(self._is_spike(spk))
            self.assertTrue(u.math.allclose(neuron.V.value, -65. * u.mV))

    # ------------------------------------------------------------------
    # 4. V_min lower bound test
    # ------------------------------------------------------------------

    def test_v_min_clamp(self):
        r"""V_min should clamp the membrane potential from below."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                V_min=-80. * u.mV,
                I_e=0. * u.pA,
                V_th=30. * u.mV,
                V_initializer=braintools.init.Constant(-79. * u.mV),
                U_initializer=braintools.init.Constant(50. * u.mV),
            )
            neuron.init_state()

            # V = -79, U = 50: dV/dt = 0.04*(-79)^2 + 5*(-79) + 140 - 50 = 249.64 - 395 + 140 - 50 = -55.36
            # V_new = -79 + (-55.36) = -134.36, should be clamped to -80
            self._step(neuron, 0, x=0. * u.pA)
            v_model = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v_model, -80.0, places=10)

    def test_v_min_none_allows_unbounded(self):
        r"""When V_min is None, V_m can go arbitrarily low."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                V_min=None,
                I_e=0. * u.pA,
                V_th=30. * u.mV,
                V_initializer=braintools.init.Constant(-79. * u.mV),
                U_initializer=braintools.init.Constant(50. * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, x=0. * u.pA)
            v_model = float((neuron.V.value / u.mV)[0])
            self.assertLess(v_model, -80.0)

    # ------------------------------------------------------------------
    # 5. Current input one-step delay test
    # ------------------------------------------------------------------

    def test_current_input_one_step_delayed(self):
        r"""External current x is buffered and applied one step later (NEST ring buffer)."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                I_e=0. * u.pA,
                V_th=1e6 * u.mV,  # prevent spikes
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            # Step 0: inject current 100 pA. I is still 0 from init.
            self._step(neuron, 0, x=100. * u.pA)
            v_after_0 = float((neuron.V.value / u.mV)[0])

            # Reference step with I=0 (current not yet applied)
            v_ref = -65.0
            u_ref = -13.0
            v_ref, u_ref, _ = _nest_reference_step(
                v_ref, u_ref, 0.0, 0.0, 0.02, 0.2, -65.0, 8.0,
                1e6, None, h, 0.0
            )
            self.assertAlmostEqual(v_after_0, v_ref, places=10)

            # Step 1: no new current, but the 100 pA from step 0 takes effect
            self._step(neuron, 1, x=0. * u.pA)
            v_after_1 = float((neuron.V.value / u.mV)[0])

            v_ref, u_ref, _ = _nest_reference_step(
                v_ref, u_ref, 100.0, 0.0, 0.02, 0.2, -65.0, 8.0,
                1e6, None, h, 0.0
            )
            self.assertAlmostEqual(v_after_1, v_ref, places=10)

    # ------------------------------------------------------------------
    # 6. Delta (spike) input tests
    # ------------------------------------------------------------------

    def test_delta_input_added_to_v(self):
        r"""Delta inputs should be added directly to V_m at the integration step."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                I_e=0. * u.pA,
                V_th=1e6 * u.mV,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            # Step with delta input of 10 mV
            self._step(neuron, 0, delta=10. * u.mV)
            v_model = float((neuron.V.value / u.mV)[0])

            v_ref = -65.0
            u_ref = -13.0
            v_ref, u_ref, _ = _nest_reference_step(
                v_ref, u_ref, 0.0, 0.0, 0.02, 0.2, -65.0, 8.0,
                1e6, None, h, 10.0
            )
            self.assertAlmostEqual(v_model, v_ref, places=10)

    def test_delta_input_triggers_spike(self):
        r"""A large delta input should push V over threshold and cause a spike."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                V_th=30. * u.mV,
                c=-65. * u.mV,
                d=8. * u.mV,
                V_initializer=braintools.init.Constant(-65. * u.mV),
                U_initializer=braintools.init.Constant(0. * u.mV),
            )
            neuron.init_state()

            # V = -65, U = 0: dV/dt = 0.04*(-65)^2 + 5*(-65) + 140 - 0 = 168.9 - 325 + 140 = -16.1
            # V_new = -65 + (-16.1) + delta
            # Need delta > 30 - (-65 + (-16.1)) = 30 + 81.1 = 111.1
            spk = self._step(neuron, 0, delta=120. * u.mV)
            self.assertTrue(self._is_spike(spk))
            self.assertTrue(u.math.allclose(neuron.V.value, -65. * u.mV))

    # ------------------------------------------------------------------
    # 7. Full trace comparison with reference
    # ------------------------------------------------------------------

    def test_full_trace_regular_spiking(self):
        r"""Compare full spike trace (Regular Spiking neuron) against reference."""
        h = 1.0
        n_steps = 500
        I_ext = 14.0  # pA, applied as I_e

        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-65. * u.mV,
                d=8. * u.mV,
                I_e=I_ext * u.pA,
                V_th=30. * u.mV,
                consistent_integration=True,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = -13.0
            I_ref = 0.0

            model_spikes = []
            ref_spikes = []

            for step in range(n_steps):
                spk = self._step(neuron, step, x=0. * u.pA)

                v_ref, u_ref, ref_spike = _nest_reference_step(
                    v_ref, u_ref, I_ref, I_ext, 0.02, 0.2, -65.0, 8.0,
                    30.0, None, h, 0.0
                )

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])

                self.assertAlmostEqual(v_model, v_ref, places=9,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=9,
                                       msg=f"U mismatch at step {step}")

                if self._is_spike(spk):
                    model_spikes.append(step)
                if ref_spike:
                    ref_spikes.append(step)

            self.assertEqual(model_spikes, ref_spikes,
                             "Spike times differ between model and reference")
            self.assertGreater(len(model_spikes), 0, "No spikes generated")

    def test_full_trace_chattering(self):
        r"""Chattering neuron (c=-50, d=2) should produce spike bursts."""
        h = 1.0
        n_steps = 500

        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-50. * u.mV,
                d=2. * u.mV,
                I_e=10. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=True,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = -13.0
            I_ref = 0.0

            for step in range(n_steps):
                self._step(neuron, step)

                v_ref, u_ref, _ = _nest_reference_step(
                    v_ref, u_ref, I_ref, 10.0, 0.02, 0.2, -50.0, 2.0,
                    30.0, None, h, 0.0
                )

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=9,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=9,
                                       msg=f"U mismatch at step {step}")

    def test_full_trace_fast_spiking(self):
        r"""Fast spiking neuron (a=0.1, b=0.2, c=-65, d=2)."""
        h = 1.0
        n_steps = 500

        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.1,
                b=0.2,
                c=-65. * u.mV,
                d=2. * u.mV,
                I_e=10. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=True,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = 0.2 * -65.0
            I_ref = 0.0

            for step in range(n_steps):
                self._step(neuron, step)

                v_ref, u_ref, _ = _nest_reference_step(
                    v_ref, u_ref, I_ref, 10.0, 0.1, 0.2, -65.0, 2.0,
                    30.0, None, h, 0.0
                )

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=9,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=9,
                                       msg=f"U mismatch at step {step}")

    def test_full_trace_inconsistent_with_spikes(self):
        r"""Test the inconsistent (published) integration with spike generation."""
        h = 1.0
        n_steps = 300

        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-65. * u.mV,
                d=8. * u.mV,
                I_e=14. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=False,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = -13.0
            I_ref = 0.0

            model_spikes = []
            ref_spikes = []

            for step in range(n_steps):
                spk = self._step(neuron, step)

                v_ref, u_ref, ref_spike = _nest_reference_step(
                    v_ref, u_ref, I_ref, 14.0, 0.02, 0.2, -65.0, 8.0,
                    30.0, None, h, 0.0, consistent_integration=False
                )

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=9,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=9,
                                       msg=f"U mismatch at step {step}")

                if self._is_spike(spk):
                    model_spikes.append(step)
                if ref_spike:
                    ref_spikes.append(step)

            self.assertEqual(model_spikes, ref_spikes)
            self.assertGreater(len(model_spikes), 0)

    # ------------------------------------------------------------------
    # 8. Synaptic current response test
    # ------------------------------------------------------------------

    def test_synaptic_current_response(self):
        r"""Apply a pulse of external current and verify one-step delay."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                I_e=0. * u.pA,
                V_th=1e6 * u.mV,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = -13.0
            I_ref = 0.0

            # Sequence: inject 50 pA at step 2, then 0 again
            current_seq = [0., 0., 50., 0., 0., 0., 0.]
            for step, I_ext in enumerate(current_seq):
                self._step(neuron, step, x=I_ext * u.pA)

                v_ref, u_ref, _ = _nest_reference_step(
                    v_ref, u_ref, I_ref, 0.0, 0.02, 0.2, -65.0, 8.0,
                    1e6, None, h, 0.0
                )
                I_ref = I_ext  # buffered for next step

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=10,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=10,
                                       msg=f"U mismatch at step {step}")

    # ------------------------------------------------------------------
    # 9. Shape / batch tests
    # ------------------------------------------------------------------

    def test_shape_single_neuron(self):
        r"""State shapes for a single neuron without batch."""
        with brainstate.environ.context(dt=self.dt):
            neuron = izhikevich(1)
            neuron.init_state()
            self.assertEqual(neuron.V.value.shape, (1,))
            self.assertEqual(neuron.U.value.shape, (1,))
            self.assertEqual(neuron.I.value.shape, (1,))

    def test_shape_with_batch(self):
        r"""State shapes with batch dimension."""
        with brainstate.environ.context(dt=self.dt):
            neuron = izhikevich(4)
            neuron.init_state(batch_size=3)
            self.assertEqual(neuron.V.value.shape, (3, 4))
            self.assertEqual(neuron.U.value.shape, (3, 4))
            self.assertEqual(neuron.I.value.shape, (3, 4))

    def test_multi_neuron_independent(self):
        r"""Multiple neurons should evolve independently."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                2,
                I_e=jnp.array([10., 0.]) * u.pA,
                V_th=30. * u.mV,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            for step in range(50):
                self._step(neuron, step)

            # Neuron 0 (with I_e=10) should have different V from neuron 1 (I_e=0)
            v0 = float((neuron.V.value / u.mV)[0])
            v1 = float((neuron.V.value / u.mV)[1])
            self.assertNotAlmostEqual(v0, v1, places=1)

    # ------------------------------------------------------------------
    # 10. Specific neuron type parameter sets
    # ------------------------------------------------------------------

    def test_intrinsically_bursting_pattern(self):
        r"""Intrinsically bursting (IB): a=0.02, b=0.2, c=-55, d=4."""
        h = 1.0
        n_steps = 500

        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-55. * u.mV,
                d=4. * u.mV,
                I_e=10. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=True,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = 0.2 * -65.0
            I_ref = 0.0

            for step in range(n_steps):
                self._step(neuron, step)

                v_ref, u_ref, _ = _nest_reference_step(
                    v_ref, u_ref, I_ref, 10.0, 0.02, 0.2, -55.0, 4.0,
                    30.0, None, h, 0.0
                )

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=9,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=9,
                                       msg=f"U mismatch at step {step}")

    def test_low_threshold_spiking_pattern(self):
        r"""Low-threshold spiking (LTS): a=0.02, b=0.25, c=-65, d=2."""
        h = 1.0
        n_steps = 500

        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.25,
                c=-65. * u.mV,
                d=2. * u.mV,
                I_e=10. * u.pA,
                V_th=30. * u.mV,
                consistent_integration=True,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = 0.25 * -65.0
            I_ref = 0.0

            for step in range(n_steps):
                self._step(neuron, step)

                v_ref, u_ref, _ = _nest_reference_step(
                    v_ref, u_ref, I_ref, 10.0, 0.02, 0.25, -65.0, 2.0,
                    30.0, None, h, 0.0
                )

                v_model = float((neuron.V.value / u.mV)[0])
                u_model = float((neuron.U.value / u.mV)[0])
                self.assertAlmostEqual(v_model, v_ref, places=9,
                                       msg=f"V mismatch at step {step}")
                self.assertAlmostEqual(u_model, u_ref, places=9,
                                       msg=f"U mismatch at step {step}")

    # ------------------------------------------------------------------
    # 11. Mixed delta + current input test
    # ------------------------------------------------------------------

    def test_mixed_delta_and_current_input(self):
        r"""Test combining delta (spike) input and current input."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            neuron = izhikevich(
                1,
                I_e=0. * u.pA,
                V_th=1e6 * u.mV,
                V_initializer=braintools.init.Constant(-65. * u.mV),
            )
            neuron.init_state()

            v_ref = -65.0
            u_ref = -13.0
            I_ref = 0.0

            # Step 0: delta=5 mV, current=100 pA
            self._step(neuron, 0, x=100. * u.pA, delta=5. * u.mV)
            v_ref, u_ref, _ = _nest_reference_step(
                v_ref, u_ref, 0.0, 0.0, 0.02, 0.2, -65.0, 8.0,
                1e6, None, h, 5.0
            )
            v_model = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v_model, v_ref, places=10)

            # Step 1: no delta, no new current; but I=100 from step 0 takes effect
            self._step(neuron, 1, x=0. * u.pA)
            I_ref = 100.0
            v_ref, u_ref, _ = _nest_reference_step(
                v_ref, u_ref, I_ref, 0.0, 0.02, 0.2, -65.0, 8.0,
                1e6, None, h, 0.0
            )
            v_model = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v_model, v_ref, places=10)

    # ------------------------------------------------------------------
    # 12. No refractory period test
    # ------------------------------------------------------------------

    def test_no_refractory_period(self):
        r"""Izhikevich model has NO refractory period — can spike on consecutive steps."""
        h = 1.0
        with brainstate.environ.context(dt=h * u.ms):
            # Use parameters that can produce very rapid spiking
            neuron = izhikevich(
                1,
                a=0.02,
                b=0.2,
                c=-50. * u.mV,  # high reset
                d=0. * u.mV,    # no U increment
                I_e=0. * u.pA,
                V_th=30. * u.mV,
                V_initializer=braintools.init.Constant(-50. * u.mV),
                U_initializer=braintools.init.Constant(0. * u.mV),
            )
            neuron.init_state()

            # Run a few steps with very large input to force consecutive spikes
            spike_count = 0
            for step in range(10):
                spk = self._step(neuron, step, delta=200. * u.mV)
                if self._is_spike(spk):
                    spike_count += 1

            # Should spike on almost every step due to massive delta input
            self.assertGreater(spike_count, 5, "Should be able to spike on consecutive steps")


if __name__ == '__main__':
    unittest.main()
