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
Tests for the step_current_generator stimulation device.

Validates the brainpy.state ``step_current_generator`` against:
  1. Expected piecewise constant current output (unit tests).
  2. Validation of parameter constraints.
  3. Driving a neuron and comparing against analytical solutions.
  4. The NEST simulator ``step_current_generator`` reference
     (skipped when NEST is not installed).
"""

import math
import unittest

import brainstate
import saiunit as u
import numpy as np
import numpy.testing as npt

brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import step_current_generator, iaf_psc_delta


class TestStepCurrentGeneratorBasic(unittest.TestCase):
    r"""Unit tests for step_current_generator output values and timing."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_empty_schedule(self):
        r"""With no amplitude schedule, output is always zero."""
        with brainstate.environ.context(dt=self.dt):
            with self.assertRaises(AssertionError):
                scg = step_current_generator()

    def test_single_step(self):
        r"""Single step change: zero before, amplitude after."""
        with brainstate.environ.context(dt=self.dt):
            scg = step_current_generator(
                amplitude_times=[10. * u.ms],
                amplitude_values=[200. * u.pA],
            )
            # Before change
            with brainstate.environ.context(t=5. * u.ms):
                out = scg.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA))

            # At change time
            with brainstate.environ.context(t=10. * u.ms):
                out = scg.update()
            self.assertTrue(u.math.allclose(out, 200. * u.pA))

            # After change
            with brainstate.environ.context(t=50. * u.ms):
                out = scg.update()
            self.assertTrue(u.math.allclose(out, 200. * u.pA))

    def test_multiple_steps(self):
        r"""Multiple step changes produce correct piecewise constant output."""
        with brainstate.environ.context(dt=self.dt):
            scg = step_current_generator(
                amplitude_times=[10. * u.ms, 30. * u.ms, 50. * u.ms],
                amplitude_values=[100. * u.pA, -200. * u.pA, 300. * u.pA],
            )
            # Before first change
            with brainstate.environ.context(t=5. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), 0. * u.pA))

            # First segment
            with brainstate.environ.context(t=20. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), 100. * u.pA))

            # Second segment
            with brainstate.environ.context(t=40. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), -200. * u.pA))

            # Third segment
            with brainstate.environ.context(t=60. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), 300. * u.pA))

    def test_start_stop_gating(self):
        r"""Active window gates the output."""
        with brainstate.environ.context(dt=self.dt):
            scg = step_current_generator(
                amplitude_times=[5. * u.ms],
                amplitude_values=[200. * u.pA],
                start=10. * u.ms,
                stop=30. * u.ms,
            )
            # Before start: even though amp is set, output is 0
            with brainstate.environ.context(t=8. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), 0. * u.pA))

            # During active window
            with brainstate.environ.context(t=15. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), 200. * u.pA))

            # After stop
            with brainstate.environ.context(t=30. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), 0. * u.pA))

    def test_origin_shifts_window(self):
        r"""Origin shifts start and stop."""
        with brainstate.environ.context(dt=self.dt):
            scg = step_current_generator(
                amplitude_times=[5. * u.ms],
                amplitude_values=[200. * u.pA],
                start=10. * u.ms,
                stop=30. * u.ms,
                origin=100. * u.ms,
            )
            # Effective window: 110 to 130 ms
            with brainstate.environ.context(t=105. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), 0. * u.pA))

            with brainstate.environ.context(t=115. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), 200. * u.pA))

            with brainstate.environ.context(t=135. * u.ms):
                self.assertTrue(u.math.allclose(scg.update(), 0. * u.pA))

    def test_output_shape(self):
        r"""Output shape matches in_size."""
        with brainstate.environ.context(dt=self.dt):
            scg = step_current_generator(
                in_size=5,
                amplitude_times=[1. * u.ms],
                amplitude_values=[100. * u.pA],
            )
            with brainstate.environ.context(t=5. * u.ms):
                out = scg.update()
            self.assertEqual(out.shape, ())


class TestStepCurrentGeneratorValidation(unittest.TestCase):
    r"""Test parameter validation."""

    def test_mismatched_lengths_raises(self):
        r"""Different sized amplitude_times and amplitude_values raises error."""
        with self.assertRaises(ValueError):
            step_current_generator(
                amplitude_times=[1. * u.ms, 2. * u.ms],
                amplitude_values=[100. * u.pA],
            )

    def test_non_increasing_times_raises(self):
        r"""Non-strictly increasing times raises error."""
        with self.assertRaises(ValueError):
            step_current_generator(
                amplitude_times=[1. * u.ms, 2. * u.ms, 2. * u.ms],
                amplitude_values=[100. * u.pA, 200. * u.pA, 300. * u.pA],
            )


class TestStepCurrentGeneratorWithNeuron(unittest.TestCase):
    r"""Integration tests with iaf_psc_delta neuron."""

    def test_step_current_vs_analytical(self):
        r"""Step current driving neuron matches analytical integration."""
        dt_ms = 0.1
        simtime = 50.0
        n_steps = int(round(simtime / dt_ms))

        amp_times = [10.0, 30.0]  # ms
        amp_values = [300.0, -100.0]  # pA

        # NEST iaf_psc_delta defaults
        E_L = -70.0
        C_m = 250.0
        tau_m = 10.0
        V_th = -55.0
        V_reset = -70.0
        t_ref = 2.0

        P33 = math.exp(-dt_ms / tau_m)
        P30 = (tau_m / C_m) * (1.0 - P33)

        with brainstate.environ.context(dt=dt_ms * u.ms):
            scg = step_current_generator(
                amplitude_times=[t * u.ms for t in amp_times],
                amplitude_values=[a * u.pA for a in amp_values],
            )
            neuron = iaf_psc_delta(1)
            neuron.init_state()

            dftype = brainstate.environ.dftype()
            vm_bp = np.empty(n_steps, dtype=dftype)
            for step in range(n_steps):
                t = step * dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    current = scg.update()
                    neuron.update(x=current)
                    vm_bp[step] = float(neuron.V.value[0] / u.mV)

        # Analytical trace
        V = E_L
        ref_count = 0
        vm_ref = np.empty(n_steps, dtype=dftype)
        for step in range(n_steps):
            t = step * dt_ms
            # Determine current
            I = 0.0
            for i in range(len(amp_times)):
                if t >= amp_times[i]:
                    I = amp_values[i]

            if ref_count == 0:
                V = E_L + (V - E_L) * P33 + I * P30
            else:
                ref_count -= 1

            if V >= V_th and ref_count == 0:
                ref_count = math.ceil(t_ref / dt_ms)
                V = V_reset

            vm_ref[step] = V

        npt.assert_allclose(vm_bp, vm_ref, atol=1e-10,
                            err_msg="Step current driven V_m doesn't match analytical")


class TestStepCurrentGeneratorVsNEST(unittest.TestCase):
    r"""Compare against NEST simulator."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    @staticmethod
    def _is_nest_available():
        try:
            import nest  # noqa: F401
            return True
        except ImportError:
            return False

    def test_step_current_vs_nest(self):
        r"""Compare step current output against NEST step_current_generator."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        import nest

        amp_times = [5.0, 15.0, 25.0]
        amp_values = [200.0, -100.0, 400.0]
        simtime = 40.0

        # --- NEST ---
        nest.ResetKernel()
        nest.resolution = self.dt_ms

        n_nest = nest.Create("iaf_psc_delta")
        scg_nest = nest.Create("step_current_generator", params={
            "amplitude_times": amp_times,
            "amplitude_values": amp_values,
        })
        vm_rec = nest.Create("voltmeter", params={"interval": self.dt_ms})
        nest.Connect(scg_nest, n_nest)
        nest.Connect(vm_rec, n_nest)
        nest.Simulate(simtime)
        v_nest = np.array(vm_rec.get("events", "V_m"))

        # --- brainpy.state ---
        n_steps = int(round(simtime / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            scg = step_current_generator(
                amplitude_times=[t * u.ms for t in amp_times],
                amplitude_values=[a * u.pA for a in amp_values],
            )
            neuron = iaf_psc_delta(1)
            neuron.init_state()

            dftype = brainstate.environ.dftype()
            v_bp = np.empty(n_steps, dtype=dftype)
            for step in range(n_steps):
                t = step * self.dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    current = scg.update()
                    neuron.update(x=current)
                    v_bp[step] = float(neuron.V.value[0] / u.mV)

        npt.assert_allclose(v_bp[1:len(v_nest) + 1], v_nest, atol=1e-10,
                            err_msg="Step current V_m differs from NEST")


if __name__ == '__main__':
    unittest.main()
