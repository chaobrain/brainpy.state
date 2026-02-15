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

"""
Tests for the step_rate_generator stimulation device.

Validates the brainpy.state ``step_rate_generator`` against:
  1. Expected piecewise constant rate output (unit tests).
  2. Validation of parameter constraints.
  3. The NEST simulator ``step_rate_generator`` reference
     (skipped when NEST is not installed).
"""

import unittest

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import step_rate_generator


class TestStepRateGeneratorBasic(unittest.TestCase):
    """Unit tests for step_rate_generator output values and timing."""

    def setUp(self):
        self.dt = 0.1 * u.ms

    def test_empty_schedule(self):
        """With no amplitude schedule, output is always zero."""
        with brainstate.environ.context(dt=self.dt):
            srg = step_rate_generator()
            for t_val in [0., 5., 50., 100.]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = srg.update()
                npt.assert_allclose(out, 0.0, atol=1e-15,
                                    err_msg=f"Should be 0 at t={t_val} ms")

    def test_single_step(self):
        """Single step change: zero before, rate after."""
        with brainstate.environ.context(dt=self.dt):
            srg = step_rate_generator(
                amplitude_times=[10. * u.ms],
                amplitude_values=[400.0],
            )
            with brainstate.environ.context(t=5. * u.ms):
                out = srg.update()
            npt.assert_allclose(out, 0.0, atol=1e-15)

            with brainstate.environ.context(t=10. * u.ms):
                out = srg.update()
            npt.assert_allclose(out, 400.0, atol=1e-15)

            with brainstate.environ.context(t=50. * u.ms):
                out = srg.update()
            npt.assert_allclose(out, 400.0, atol=1e-15)

    def test_multiple_steps(self):
        """Multiple step changes produce correct piecewise constant output."""
        with brainstate.environ.context(dt=self.dt):
            srg = step_rate_generator(
                amplitude_times=[10. * u.ms, 110. * u.ms, 210. * u.ms],
                amplitude_values=[400.0, 1000.0, 200.0],
            )
            # Before first change
            with brainstate.environ.context(t=5. * u.ms):
                npt.assert_allclose(srg.update(), 0.0, atol=1e-15)

            # First segment
            with brainstate.environ.context(t=50. * u.ms):
                npt.assert_allclose(srg.update(), 400.0, atol=1e-15)

            # Second segment
            with brainstate.environ.context(t=150. * u.ms):
                npt.assert_allclose(srg.update(), 1000.0, atol=1e-15)

            # Third segment
            with brainstate.environ.context(t=250. * u.ms):
                npt.assert_allclose(srg.update(), 200.0, atol=1e-15)

    def test_start_stop_gating(self):
        """Active window gates the output."""
        with brainstate.environ.context(dt=self.dt):
            srg = step_rate_generator(
                amplitude_times=[5. * u.ms],
                amplitude_values=[400.0],
                start=10. * u.ms,
                stop=30. * u.ms,
            )
            with brainstate.environ.context(t=8. * u.ms):
                npt.assert_allclose(srg.update(), 0.0, atol=1e-15)

            with brainstate.environ.context(t=15. * u.ms):
                npt.assert_allclose(srg.update(), 400.0, atol=1e-15)

            with brainstate.environ.context(t=30. * u.ms):
                npt.assert_allclose(srg.update(), 0.0, atol=1e-15)

    def test_output_shape(self):
        """Output shape matches in_size."""
        with brainstate.environ.context(dt=self.dt):
            srg = step_rate_generator(
                in_size=5,
                amplitude_times=[1. * u.ms],
                amplitude_values=[100.0],
            )
            with brainstate.environ.context(t=5. * u.ms):
                out = srg.update()
            self.assertEqual(out.shape, (5,))

    def test_full_simulation_trace(self):
        """Run a full simulation and verify rate at sampled points."""
        dt_ms = 0.1
        rates = [400.0, 1000.0, 200.0]
        amp_times = [10.0, 110.0, 210.0]
        simtime = 300.0
        n_steps = int(round(simtime / dt_ms))

        with brainstate.environ.context(dt=dt_ms * u.ms):
            srg = step_rate_generator(
                amplitude_times=[t * u.ms for t in amp_times],
                amplitude_values=rates,
            )

            # Sample rate at representative points in each segment
            # midpoint of [10, 110): t=60
            with brainstate.environ.context(t=60. * u.ms):
                rate_1 = float(srg.update()[0])
            # midpoint of [110, 210): t=160
            with brainstate.environ.context(t=160. * u.ms):
                rate_2 = float(srg.update()[0])
            # midpoint of [210, ...): t=260
            with brainstate.environ.context(t=260. * u.ms):
                rate_3 = float(srg.update()[0])

        npt.assert_allclose([rate_1, rate_2, rate_3], rates, atol=1e-15)


class TestStepRateGeneratorValidation(unittest.TestCase):
    """Test parameter validation."""

    def test_mismatched_lengths_raises(self):
        """Different sized amplitude_times and amplitude_values raises error."""
        with self.assertRaises(ValueError):
            step_rate_generator(
                amplitude_times=[1. * u.ms, 2. * u.ms],
                amplitude_values=[100.0],
            )

    def test_non_increasing_times_raises(self):
        """Non-strictly increasing times raises error."""
        with self.assertRaises(ValueError):
            step_rate_generator(
                amplitude_times=[1. * u.ms, 2. * u.ms, 2. * u.ms],
                amplitude_values=[100.0, 200.0, 300.0],
            )


class TestStepRateGeneratorVsNEST(unittest.TestCase):
    """Compare against NEST simulator."""

    @staticmethod
    def _is_nest_available():
        try:
            import nest  # noqa: F401
            return True
        except ImportError:
            return False

    def test_step_rate_vs_nest(self):
        """Compare step rate output against NEST step_rate_generator.

        Reproduces NEST's test_step_rate_generator.py logic.
        """
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        import nest

        rates = [400.0, 1000.0, 200.0]
        amp_times = [10.0, 110.0, 210.0]
        simtime = 301.0

        # --- NEST ---
        nest.ResetKernel()
        nest.resolution = 0.1
        nest.use_wfr = False

        srg_nest = nest.Create("step_rate_generator", params={
            "amplitude_times": amp_times,
            "amplitude_values": rates,
        })
        mm = nest.Create("multimeter", params={
            "record_from": ["rate"], "interval": 100.0
        })
        nest.Connect(mm, srg_nest)
        nest.Simulate(simtime)

        data = nest.GetStatus(mm)[0]["events"]
        rates_nest = np.array(data["rate"][
            np.where(data["senders"] == srg_nest.get("global_id"))
        ])

        npt.assert_array_equal(rates, rates_nest,
                               err_msg="NEST step_rate_generator rates don't match expected")

        # --- brainpy.state ---
        dt_ms = 0.1
        with brainstate.environ.context(dt=dt_ms * u.ms):
            srg = step_rate_generator(
                amplitude_times=[t * u.ms for t in amp_times],
                amplitude_values=rates,
            )
            # Sample at the same multimeter recording times (100, 200, 300 ms)
            bp_rates = []
            for t_sample in [100.0, 200.0, 300.0]:
                with brainstate.environ.context(t=t_sample * u.ms):
                    bp_rates.append(float(srg.update()[0]))

        npt.assert_allclose(bp_rates, rates, atol=1e-15,
                            err_msg="brainpy step_rate_generator rates don't match")


if __name__ == '__main__':
    unittest.main()
