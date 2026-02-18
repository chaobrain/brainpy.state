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
Tests for the ac_generator stimulation device.

Validates the brainpy.state ``ac_generator`` against:
  1. Expected sinusoidal current output (unit tests).
  2. Analytical solutions for sine-driven neuron membrane potential.
  3. Comparison against step_current_generator with matching sinusoidal amplitudes
     (reproducing the NEST test_ac_generator logic).
  4. The NEST simulator ``ac_generator`` reference (skipped when NEST is not installed).
"""

import math
import unittest

import brainstate
import brainunit as u
import numpy as np
import numpy.testing as npt
from brainpy.state import ac_generator


# ===================================================================
# Test Suite 1: ac_generator device in isolation
# ===================================================================

class TestACGenerator(unittest.TestCase):
    r"""Unit tests for ac_generator output values and timing."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_default_parameters(self):
        r"""Default amplitude=0, offset=0, frequency=0, phase=0."""
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator()
        self.assertTrue(u.math.allclose(ac.amplitude, 0. * u.pA))
        self.assertTrue(u.math.allclose(ac.offset, 0. * u.pA))
        self.assertTrue(u.math.allclose(ac.frequency, 0. * u.Hz))
        self.assertIsNone(ac.stop)
        self.assertTrue(u.math.allclose(ac.origin, 0. * u.ms))

    def test_output_zero_with_default_params(self):
        r"""With all-zero defaults, output should be zero at any time."""
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator()
            for t_val in [0., 1., 10., 100.]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = ac.update()
                self.assertTrue(u.math.allclose(out, 0. * u.pA),
                                f"Failed at t={t_val} ms")

    def test_dc_offset_only(self):
        r"""With zero amplitude and nonzero offset, output should be constant."""
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator(offset=500. * u.pA)
            for t_val in [0., 5., 50.]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = ac.update()
                self.assertTrue(u.math.allclose(out, 500. * u.pA),
                                f"Failed at t={t_val} ms")

    def test_pure_sine_at_known_times(self):
        r"""Test pure sine output at specific times."""
        amp = 100.  # pA
        freq = 250.  # Hz -> period = 4 ms
        omega = 2 * math.pi * freq / 1000.0  # rad/ms
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator(amplitude=amp * u.pA,
                              frequency=freq * u.Hz)

            # At t=0, sin(0) = 0
            with brainstate.environ.context(t=0. * u.ms):
                out = ac.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA, atol=1e-10 * u.pA))

            # At t = 1 ms (quarter period for 250 Hz),
            # sin(2*pi*250/1000 * 1) = sin(pi/2) = 1.0
            with brainstate.environ.context(t=1. * u.ms):
                out = ac.update()
            self.assertTrue(u.math.allclose(out, amp * u.pA, atol=1e-10 * u.pA),
                            f"Expected {amp} pA at t=1 ms, got {out}")

            # At t = 2 ms (half period), sin(pi) = 0
            with brainstate.environ.context(t=2. * u.ms):
                out = ac.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA, atol=1e-10 * u.pA))

    def test_phase_shift(self):
        r"""Test that phase shifts the sine correctly."""
        amp = 100.  # pA
        freq = 250.  # Hz
        phase = 90.  # degrees -> phi_rad = pi/2
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator(amplitude=amp * u.pA,
                              frequency=freq * u.Hz,
                              phase=phase)
            # At t=0, sin(0 + pi/2) = 1.0
            with brainstate.environ.context(t=0. * u.ms):
                out = ac.update()
            self.assertTrue(u.math.allclose(out, amp * u.pA, atol=1e-10 * u.pA))

    def test_output_before_start_is_zero(self):
        r"""Before start time, output is zero."""
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator(amplitude=100. * u.pA,
                              offset=50. * u.pA,
                              frequency=100. * u.Hz,
                              start=10. * u.ms)
            with brainstate.environ.context(t=5. * u.ms):
                out = ac.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA))

    def test_output_after_stop_is_zero(self):
        r"""After stop time, output is zero."""
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator(amplitude=100. * u.pA,
                              offset=50. * u.pA,
                              frequency=100. * u.Hz,
                              start=0. * u.ms,
                              stop=10. * u.ms)
            with brainstate.environ.context(t=10. * u.ms):
                out = ac.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA))

    def test_output_during_active_window(self):
        r"""During active window, output is nonzero (for nonzero offset+amplitude)."""
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator(amplitude=100. * u.pA,
                              offset=200. * u.pA,
                              frequency=100. * u.Hz,
                              start=5. * u.ms,
                              stop=15. * u.ms)
            with brainstate.environ.context(t=10. * u.ms):
                out = ac.update()
            # Should be offset + amplitude * sin(omega * 10)
            omega = 2 * math.pi * 100. / 1000.
            expected = 200. + 100. * math.sin(omega * 10.)
            self.assertTrue(
                u.math.allclose(out, expected * u.pA, atol=1e-10 * u.pA))

    def test_origin_shifts_window(self):
        r"""Origin shifts start and stop."""
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator(amplitude=100. * u.pA,
                              frequency=100. * u.Hz,
                              start=10. * u.ms,
                              stop=20. * u.ms,
                              origin=100. * u.ms)
            # Effective window: 110 to 120 ms
            with brainstate.environ.context(t=105. * u.ms):
                out = ac.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA))

            with brainstate.environ.context(t=115. * u.ms):
                out = ac.update()
            # Should be nonzero
            omega = 2 * math.pi * 100. / 1000.
            expected = 100. * math.sin(omega * 115.)
            self.assertTrue(
                u.math.allclose(out, expected * u.pA, atol=1e-10 * u.pA))

    def test_output_shape(self):
        r"""Output shape matches in_size."""
        with brainstate.environ.context(dt=self.dt):
            ac = ac_generator(in_size=5, amplitude=100. * u.pA,
                              frequency=100. * u.Hz)
            with brainstate.environ.context(t=1. * u.ms):
                out = ac.update()
            self.assertEqual(out.shape, (5,))


# ===================================================================
# Test Suite 2: ac_generator matches step_current_generator equivalent
# ===================================================================

class TestACGeneratorVsStepCurrent(unittest.TestCase):
    r"""Compare ac_generator against manually computed sine values.

    This mirrors the NEST test_ac_generator.py: an ac_generator should produce
    the same current as a step_current_generator loaded with the equivalent
    sinusoidal values.
    """

    def test_ac_vs_manual_sine_trace(self):
        r"""AC generator output matches manual sin() computation at each step.

        Reproduces the logic of NEST's test_ac_generator.py.
        """
        dc = 1000.0  # pA offset
        ac = 550.0  # pA amplitude
        freq = 100.0  # Hz
        phi = 0.0  # degrees
        start = 5.0  # ms
        stop = 6.0  # ms
        dt_ms = 0.1

        omega = freq / 1000.0 * 2 * math.pi  # rad/ms
        phi_rad = phi * 2 * math.pi / 360.0

        n_steps = int(round(10.0 / dt_ms))  # 10 ms simulation

        with brainstate.environ.context(dt=dt_ms * u.ms):
            ac_gen = ac_generator(
                amplitude=ac * u.pA,
                offset=dc * u.pA,
                frequency=freq * u.Hz,
                phase=phi,
                start=start * u.ms,
                stop=stop * u.ms,
            )

            dftype = brainstate.environ.dftype()
            ac_trace = np.empty(n_steps, dtype=dftype)
            manual_trace = np.empty(n_steps, dtype=dftype)

            for step in range(n_steps):
                t = step * dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    out = ac_gen.update()
                    ac_trace[step] = float(out[0] / u.pA) if hasattr(out, '__len__') else float(out / u.pA)

                # Manual computation
                if start <= t < stop:
                    manual_trace[step] = math.sin(t * omega + phi_rad) * ac + dc
                else:
                    manual_trace[step] = 0.0

        npt.assert_allclose(ac_trace, manual_trace, atol=1e-10,
                            err_msg="AC generator does not match manual sine trace")

    def test_ac_full_range_frequencies(self):
        r"""Test at various frequencies that the output matches sin()."""
        dt_ms = 0.1
        simtime = 20.0
        n_steps = int(round(simtime / dt_ms))

        for freq in [10.0, 50.0, 100.0, 500.0]:
            omega = 2 * math.pi * freq / 1000.0
            amp = 200.0
            offset = 100.0

            with brainstate.environ.context(dt=dt_ms * u.ms):
                ac_gen = ac_generator(
                    amplitude=amp * u.pA,
                    offset=offset * u.pA,
                    frequency=freq * u.Hz,
                )

                for step in range(n_steps):
                    t = step * dt_ms
                    with brainstate.environ.context(t=t * u.ms):
                        out = ac_gen.update()
                        out_val = float(out[0] / u.pA) if hasattr(out, '__len__') else float(out / u.pA)

                    expected = offset + amp * math.sin(omega * t)
                    self.assertAlmostEqual(
                        out_val, expected, places=10,
                        msg=f"Mismatch at freq={freq} Hz, t={t} ms")


# ===================================================================
# Test Suite 3: comparison against NEST simulator
# ===================================================================

class TestACGeneratorVsNEST(unittest.TestCase):
    r"""Compare ac_generator against NEST. Skipped when NEST is not installed."""

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

    def test_ac_current_trace_vs_nest(self):
        r"""Compare AC current output trace against NEST ac_generator."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        import nest

        amp = 550.0
        offset = 1000.0
        freq = 100.0
        phase = 45.0
        start = 5.0
        stop = 50.0
        simtime = 60.0

        # --- NEST reference ---
        nest.ResetKernel()
        nest.resolution = self.dt_ms

        ac_nest = nest.Create("ac_generator", params={
            "amplitude": amp,
            "offset": offset,
            "frequency": freq,
            "phase": phase,
            "start": start,
            "stop": stop,
        })
        mm = nest.Create("multimeter", params={
            "record_from": ["I"],
            "interval": self.dt_ms,
        })
        nest.Connect(mm, ac_nest)
        nest.Simulate(simtime)
        I_nest = np.array(mm.get("events", "I"))

        # --- brainpy.state ---
        n_steps = int(round(simtime / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            ac_gen = ac_generator(
                amplitude=amp * u.pA,
                offset=offset * u.pA,
                frequency=freq * u.Hz,
                phase=phase,
                start=start * u.ms,
                stop=stop * u.ms,
            )

            dftype = brainstate.environ.dftype()
            I_bp = np.empty(n_steps, dtype=dftype)
            for step in range(n_steps):
                t = step * self.dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    out = ac_gen.update()
                    I_bp[step] = float(out[0] / u.pA) if hasattr(out, '__len__') else float(out / u.pA)

        # NEST multimeter records from step 1 (t=0.1 ms), so align
        npt.assert_allclose(I_bp[1:len(I_nest) + 1], I_nest, atol=1e-8,
                            err_msg="AC current trace differs from NEST")

    def test_ac_driven_neuron_vs_nest(self):
        r"""Compare neuron V_m when driven by AC generator vs NEST."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        import nest
        from brainpy.state import iaf_psc_alpha

        amp = 550.0
        offset = 1000.0
        freq = 100.0
        phase = 0.0
        start = 5.0
        stop = 6.0
        simtime = 10.0

        # --- NEST ---
        nest.ResetKernel()
        nest.resolution = self.dt_ms

        n_nest = nest.Create("iaf_psc_alpha")
        ac_nest = nest.Create("ac_generator", params={
            "amplitude": amp, "offset": offset,
            "frequency": freq, "phase": phase,
            "start": start, "stop": stop,
        })
        vm_rec = nest.Create("voltmeter", params={"interval": self.dt_ms})
        nest.Connect(ac_nest, n_nest)
        nest.Connect(vm_rec, n_nest)
        nest.Simulate(simtime)
        v_nest = np.array(vm_rec.get("events", "V_m"))

        # --- brainpy.state ---
        n_steps = int(round(simtime / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            ac_gen = ac_generator(
                amplitude=amp * u.pA, offset=offset * u.pA,
                frequency=freq * u.Hz, phase=phase,
                start=start * u.ms, stop=stop * u.ms,
            )
            neuron = iaf_psc_alpha(1)
            neuron.init_state()

            dftype = brainstate.environ.dftype()
            v_bp = np.empty(n_steps, dtype=dftype)
            for step in range(n_steps):
                t = step * self.dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    current = ac_gen.update()
                    neuron.update(x=current)
                    v_bp[step] = float(neuron.V.value[0] / u.mV)

        npt.assert_allclose(v_bp[1:len(v_nest) + 1], v_nest, atol=1e-8,
                            err_msg="AC-driven V_m differs from NEST")


if __name__ == '__main__':
    unittest.main()
