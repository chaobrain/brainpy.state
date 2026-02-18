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
Tests for the noise_generator stimulation device.

Validates the brainpy.state ``noise_generator`` against:
  1. Expected parameter defaults and basic output behavior.
  2. Statistical properties of generated noise (mean, std).
  3. Piecewise-constant noise behavior at specified intervals.
  4. The NEST simulator ``noise_generator`` reference
     (skipped when NEST is not installed).
"""

import math
import unittest

import brainstate
import brainunit as u
import numpy as np
import numpy.testing as npt

brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import noise_generator


class TestNoiseGeneratorBasic(unittest.TestCase):
    r"""Unit tests for noise_generator output values and timing."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_default_parameters(self):
        r"""Default mean=0, std=0."""
        with brainstate.environ.context(dt=self.dt):
            ng = noise_generator()
        self.assertTrue(u.math.allclose(ng.mean, 0. * u.pA))
        self.assertTrue(u.math.allclose(ng.std, 0. * u.pA))
        self.assertTrue(u.math.allclose(ng.std_mod, 0. * u.pA))
        self.assertTrue(u.math.allclose(ng.frequency, 0. * u.Hz))

    def test_zero_std_produces_mean(self):
        r"""With std=0, output should be exactly the mean when active."""
        with brainstate.environ.context(dt=self.dt):
            ng = noise_generator(mean=100. * u.pA, std=0. * u.pA, seed=42)
            ng.init_state()
            with brainstate.environ.context(t=0. * u.ms):
                out = ng.update()
            # With zero std, output should be mean (since N*0 = 0)
            self.assertTrue(u.math.allclose(out, 100. * u.pA))

    def test_output_before_start_is_zero(self):
        r"""Before start time, output is zero."""
        with brainstate.environ.context(dt=self.dt):
            ng = noise_generator(mean=100. * u.pA, std=50. * u.pA,
                                 start=10. * u.ms, seed=42)
            ng.init_state()
            with brainstate.environ.context(t=5. * u.ms):
                out = ng.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA))

    def test_output_after_stop_is_zero(self):
        r"""After stop time, output is zero."""
        with brainstate.environ.context(dt=self.dt):
            ng = noise_generator(mean=100. * u.pA, std=50. * u.pA,
                                 start=0. * u.ms, stop=10. * u.ms, seed=42)
            ng.init_state()
            with brainstate.environ.context(t=10. * u.ms):
                out = ng.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA))

    def test_output_shape(self):
        r"""Output shape matches in_size."""
        with brainstate.environ.context(dt=self.dt):
            ng = noise_generator(in_size=5, mean=0. * u.pA,
                                 std=10. * u.pA, seed=42)
            ng.init_state()
            with brainstate.environ.context(t=0. * u.ms):
                out = ng.update()
            self.assertEqual(out.shape, (5,))

    def test_different_seeds_produce_different_output(self):
        r"""Different seeds should produce different noise."""
        with brainstate.environ.context(dt=self.dt):
            ng1 = noise_generator(mean=0. * u.pA, std=100. * u.pA, seed=42)
            ng1.init_state()
            ng2 = noise_generator(mean=0. * u.pA, std=100. * u.pA, seed=123)
            ng2.init_state()

            with brainstate.environ.context(t=0. * u.ms):
                out1 = ng1.update()
                out2 = ng2.update()

            # They should be different (extremely unlikely to be equal)
            self.assertFalse(u.math.allclose(out1, out2))


class TestNoiseGeneratorStatistics(unittest.TestCase):
    r"""Test the statistical properties of generated noise."""

    def test_mean_and_std(self):
        r"""Generated noise should have approximately correct mean and std."""
        dt_ms = 0.1
        n_steps = 10000
        mean_val = 50.0
        std_val = 100.0

        with brainstate.environ.context(dt=dt_ms * u.ms):
            ng = noise_generator(
                mean=mean_val * u.pA,
                std=std_val * u.pA,
                seed=42,
            )
            ng.init_state()

            dftype = brainstate.environ.dftype()
            samples = np.empty(n_steps, dtype=dftype)
            for step in range(n_steps):
                with brainstate.environ.context(t=step * dt_ms * u.ms):
                    out = ng.update()
                    samples[step] = float(out[0] / u.pA)

        # Check mean within 3 standard errors
        se = std_val / math.sqrt(n_steps)
        self.assertAlmostEqual(np.mean(samples), mean_val, delta=3 * se,
                               msg=f"Mean {np.mean(samples):.2f} not close "
                                   f"to expected {mean_val:.2f}")

        # Check std within reasonable tolerance
        expected_std_err = std_val / math.sqrt(2 * n_steps)
        self.assertAlmostEqual(np.std(samples), std_val,
                               delta=3 * expected_std_err,
                               msg=f"Std {np.std(samples):.2f} not close "
                                   f"to expected {std_val:.2f}")

    def test_noise_is_piecewise_constant(self):
        r"""When noise_dt > simulation dt, noise should be constant within intervals."""
        dt_ms = 0.1
        noise_dt_ms = 1.0  # 10 steps per noise interval
        n_steps = 100

        with brainstate.environ.context(dt=dt_ms * u.ms):
            ng = noise_generator(
                mean=0. * u.pA,
                std=100. * u.pA,
                noise_dt=noise_dt_ms * u.ms,
                seed=42,
            )
            ng.init_state()

            dftype = brainstate.environ.dftype()
            samples = np.empty(n_steps, dtype=dftype)
            for step in range(n_steps):
                with brainstate.environ.context(t=step * dt_ms * u.ms):
                    out = ng.update()
                    samples[step] = float(out[0] / u.pA)

        # Within each noise interval of 10 steps, values should be constant
        dt_steps = int(round(noise_dt_ms / dt_ms))
        for i in range(0, n_steps - dt_steps, dt_steps):
            block = samples[i:i + dt_steps]
            # All values in block should be identical
            npt.assert_allclose(
                block, block[0] * np.ones_like(block), atol=1e-15,
                err_msg=f"Noise not constant in interval [{i}, {i + dt_steps})")


class TestNoiseGeneratorVsNEST(unittest.TestCase):
    r"""Compare noise_generator statistical behavior against NEST."""

    @staticmethod
    def _is_nest_available():
        try:
            import nest  # noqa: F401
            return True
        except ImportError:
            return False

    def test_noise_membrane_potential_statistics(self):
        r"""Reproduce NEST's noise_generator test: V_m statistics.

        For a neuron with V_th=inf, C_m=1, tau_m=1, E_L=0, the expected
        V_m standard deviation is:
            expected_vm_std = sqrt((1 + exp(-dt/tau_m)) / (1 - exp(-dt/tau_m))) * std
        """
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        import nest

        # --- NEST reference ---
        nest.ResetKernel()
        nest.resolution = 0.1
        ng_nest = nest.Create("noise_generator", params={
            "mean": 0.0, "std": 1.0, "dt": 0.1
        })
        neuron_nest = nest.Create("iaf_psc_alpha", params={
            "V_th": 1e10, "C_m": 1.0, "tau_m": 1.0, "E_L": 0.0
        })
        nest.Connect(ng_nest, neuron_nest)

        n_sims = 50
        vm_nest = np.empty(n_sims)
        for i in range(n_sims):
            nest.Simulate(1000.0)
            vm_nest[i] = neuron_nest.get("V_m")

        # Both should have similar statistical properties
        exp_dt = np.exp(-0.1)
        expected_std = np.sqrt((1 + exp_dt) / (1 - exp_dt)) * 1.0
        nest_std = np.std(vm_nest)

        # Verify NEST's own statistics are in range
        self.assertAlmostEqual(nest_std, expected_std,
                               delta=3 * expected_std / np.sqrt(2),
                               msg="NEST noise statistics out of range")


if __name__ == '__main__':
    unittest.main()
