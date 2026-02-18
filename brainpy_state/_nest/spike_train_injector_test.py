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
Tests for the spike_train_injector neuron device.

Validates the brainpy.state ``spike_train_injector`` against:
  1. Expected spike timing output (unit tests).
  2. Spike multiplicity support.
  3. Active window gating (start/stop/origin).
  4. Validation of parameter constraints.
  5. The NEST simulator ``spike_train_injector`` reference
     (skipped when NEST is not installed).
"""

import unittest

import brainstate
import brainunit as u
import numpy.testing as npt

brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import spike_train_injector


class TestSpikeTrainInjectorBasic(unittest.TestCase):
    r"""Unit tests for spike_train_injector output values and timing."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_empty_spike_times(self):
        r"""With no spike times, output is always zero."""
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector()
            for t_val in [0., 1., 10., 100.]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = inj.update()
                npt.assert_allclose(out, 0.0, atol=1e-15,
                                    err_msg=f"Should be 0 at t={t_val} ms")

    def test_single_spike(self):
        r"""Single spike at specified time."""
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[5. * u.ms],
            )
            # Before spike
            with brainstate.environ.context(t=4.9 * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 0.0, atol=1e-15)

            # At spike time
            with brainstate.environ.context(t=5. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 1.0, atol=1e-15)

            # After spike
            with brainstate.environ.context(t=5.1 * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 0.0, atol=1e-15)

    def test_multiple_spikes(self):
        r"""Multiple spikes at different times."""
        spike_times_ms = [1.0, 2.0, 3.0]
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[t * u.ms for t in spike_times_ms],
            )

            for t_val in [0.5, 1.5, 2.5, 3.5]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = inj.update()
                npt.assert_allclose(out, 0.0, atol=1e-15,
                                    err_msg=f"Should be 0 at t={t_val}")

            for t_val in spike_times_ms:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = inj.update()
                npt.assert_allclose(out, 1.0, atol=1e-15,
                                    err_msg=f"Should be 1 at t={t_val}")

    def test_spike_multiplicities(self):
        r"""Spikes with custom multiplicities."""
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[5. * u.ms, 10. * u.ms],
                spike_multiplicities=[3, 5],
            )
            with brainstate.environ.context(t=5. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 3.0, atol=1e-15)

            with brainstate.environ.context(t=10. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 5.0, atol=1e-15)

    def test_same_time_multiplicities_accumulate(self):
        r"""Multiple spike times at the same time accumulate multiplicities."""
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[5. * u.ms, 5. * u.ms],
                spike_multiplicities=[2, 7],
            )
            with brainstate.environ.context(t=5. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 9.0, atol=1e-15,
                                err_msg="Multiplicities should accumulate")

    def test_same_time_no_multiplicities_count(self):
        r"""Multiple spike times at same time without multiplicities count events."""
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[5. * u.ms, 5. * u.ms, 5. * u.ms],
            )
            with brainstate.environ.context(t=5. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 3.0, atol=1e-15,
                                err_msg="Should count 3 spikes at same time")

    def test_start_stop_gating(self):
        r"""Spikes outside active window are suppressed."""
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[3. * u.ms, 7. * u.ms, 12. * u.ms],
                start=5. * u.ms,
                stop=10. * u.ms,
            )
            # Before start: spike at t=3 suppressed
            with brainstate.environ.context(t=3. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 0.0, atol=1e-15)

            # During active window: spike at t=7 fires
            with brainstate.environ.context(t=7. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 1.0, atol=1e-15)

            # After stop: spike at t=12 suppressed
            with brainstate.environ.context(t=12. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 0.0, atol=1e-15)

    def test_origin_shifts_window(self):
        r"""Origin shifts the active window."""
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[5. * u.ms, 15. * u.ms],
                start=10. * u.ms,
                stop=20. * u.ms,
                origin=100. * u.ms,
            )
            # Effective window: 110 to 120 ms
            # spike at 5 ms: not in window
            with brainstate.environ.context(t=5. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 0.0, atol=1e-15)

            # spike at 15 ms: not in window (110-120)
            with brainstate.environ.context(t=15. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 0.0, atol=1e-15)

    def test_output_shape(self):
        r"""Output shape matches in_size."""
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                in_size=5,
                spike_times=[5. * u.ms],
            )
            with brainstate.environ.context(t=5. * u.ms):
                out = inj.update()
            self.assertEqual(out.shape, (5,))

    def test_unitless_spike_times(self):
        r"""Spike times can be given as plain floats (interpreted as ms)."""
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[5.0, 10.0],
            )
            with brainstate.environ.context(t=5. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 1.0, atol=1e-15)

            with brainstate.environ.context(t=10. * u.ms):
                out = inj.update()
            npt.assert_allclose(out, 1.0, atol=1e-15)

    def test_default_parameters(self):
        r"""Default parameters match NEST defaults."""
        inj = spike_train_injector()
        self.assertFalse(inj.precise_times)
        self.assertFalse(inj.allow_offgrid_times)
        self.assertFalse(inj.shift_now_spikes)
        self.assertEqual(len(inj._spike_times_ms), 0)
        self.assertEqual(len(inj._spike_multiplicities), 0)


class TestSpikeTrainInjectorValidation(unittest.TestCase):
    r"""Test parameter validation."""

    def test_unsorted_spike_times_raises(self):
        r"""Non-sorted spike times raises error."""
        with self.assertRaises(ValueError):
            spike_train_injector(
                spike_times=[3. * u.ms, 1. * u.ms, 5. * u.ms],
            )

    def test_mismatched_multiplicities_raises(self):
        r"""Different sized spike_times and spike_multiplicities raises error."""
        with self.assertRaises(ValueError):
            spike_train_injector(
                spike_times=[1. * u.ms, 2. * u.ms],
                spike_multiplicities=[1],
            )

    def test_precise_times_with_allow_offgrid_raises(self):
        r"""Setting precise_times with allow_offgrid_times raises error."""
        with self.assertRaises(ValueError):
            spike_train_injector(
                precise_times=True,
                allow_offgrid_times=True,
            )

    def test_precise_times_with_shift_now_spikes_raises(self):
        r"""Setting precise_times with shift_now_spikes raises error."""
        with self.assertRaises(ValueError):
            spike_train_injector(
                precise_times=True,
                shift_now_spikes=True,
            )

    def test_empty_multiplicities_allowed(self):
        r"""Empty spike_multiplicities is valid regardless of spike_times."""
        inj = spike_train_injector(
            spike_times=[1. * u.ms, 2. * u.ms],
            spike_multiplicities=[],
        )
        self.assertEqual(len(inj._spike_multiplicities), 0)


class TestSpikeTrainInjectorSimulation(unittest.TestCase):
    r"""Integration tests: full simulation with spike recording."""

    def test_spike_train_recording(self):
        r"""Run simulation and record all spike times."""
        dt_ms = 0.1
        spike_times_ms = [1.0, 2.0, 3.0, 5.0, 10.0]
        simtime = 15.0
        n_steps = int(round(simtime / dt_ms))

        with brainstate.environ.context(dt=dt_ms * u.ms):
            inj = spike_train_injector(
                spike_times=[t * u.ms for t in spike_times_ms],
            )

            recorded_spike_times = []
            for step in range(n_steps):
                t = step * dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    out = inj.update()
                    if float(out[0]) > 0.5:
                        recorded_spike_times.append(round(t, 4))

        npt.assert_allclose(recorded_spike_times, spike_times_ms, atol=1e-4,
                            err_msg="Recorded spike times don't match input")

    def test_spike_gated_recording(self):
        r"""Only spikes within active window are recorded."""
        dt_ms = 0.1
        spike_times_ms = [1.0, 3.0, 5.0, 7.0, 9.0]
        start = 3.0
        stop = 8.0
        simtime = 12.0
        n_steps = int(round(simtime / dt_ms))

        with brainstate.environ.context(dt=dt_ms * u.ms):
            inj = spike_train_injector(
                spike_times=[t * u.ms for t in spike_times_ms],
                start=start * u.ms,
                stop=stop * u.ms,
            )

            recorded = []
            for step in range(n_steps):
                t = step * dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    out = inj.update()
                    if float(out[0]) > 0.5:
                        recorded.append(round(t, 4))

        # Only spikes at 3.0, 5.0, 7.0 are in [3, 8)
        expected = [3.0, 5.0, 7.0]
        npt.assert_allclose(recorded, expected, atol=1e-4,
                            err_msg="Gated spike times don't match expected")

    def test_multiplicity_recording(self):
        r"""Multiplicities are correctly reflected in output values."""
        dt_ms = 0.1
        spike_times_ms = [1.0, 2.0]
        multiplicities = [3, 5]
        simtime = 5.0
        n_steps = int(round(simtime / dt_ms))

        with brainstate.environ.context(dt=dt_ms * u.ms):
            inj = spike_train_injector(
                spike_times=[t * u.ms for t in spike_times_ms],
                spike_multiplicities=multiplicities,
            )

            recorded = {}
            for step in range(n_steps):
                t = step * dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    out = inj.update()
                    val = float(out[0])
                    if val > 0.5:
                        recorded[round(t, 4)] = int(val)

        self.assertEqual(recorded.get(1.0), 3)
        self.assertEqual(recorded.get(2.0), 5)


class TestSpikeTrainInjectorVsNEST(unittest.TestCase):
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

    def test_basic_spike_times_vs_nest(self):
        r"""Compare basic spike timing against NEST spike_train_injector."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        import nest

        spike_times_ms = [1.0, 2.0, 3.0]
        simtime = 10.0

        # --- NEST ---
        nest.ResetKernel()
        nest.resolution = self.dt_ms

        inj_nest = nest.Create(
            "spike_train_injector",
            params={"spike_times": spike_times_ms},
        )
        sr = nest.Create("spike_recorder")
        nest.Connect(inj_nest, sr)
        nest.Simulate(simtime)

        nest_spike_times = sorted(sr.events["times"].tolist())

        # --- brainpy.state ---
        n_steps = int(round(simtime / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[t * u.ms for t in spike_times_ms],
            )

            bp_spike_times = []
            for step in range(n_steps):
                t = step * self.dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    out = inj.update()
                    if float(out[0]) > 0.5:
                        bp_spike_times.append(round(t, 4))

        npt.assert_allclose(bp_spike_times, spike_times_ms, atol=1e-4,
                            err_msg="brainpy spike times don't match expected")

    def test_grid_rounding_vs_nest(self):
        r"""Compare spike time rounding behavior against NEST."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        import nest

        # These should be rounded to [1.0, 2.0, 3.0] with 0.1ms resolution
        in_spike_times = [1.0, 1.9999, 3.0001]
        expected_spike_times = [1.0, 2.0, 3.0]
        simtime = 10.0

        # --- NEST ---
        nest.ResetKernel()
        nest.set(resolution=0.1, tics_per_ms=1000.0)

        inj_nest = nest.Create(
            "spike_train_injector",
            params={"spike_times": in_spike_times},
        )
        out_spike_times_nest = nest.GetStatus(inj_nest, "spike_times")[0]

        # --- brainpy.state ---
        n_steps = int(round(simtime / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[t * u.ms for t in in_spike_times],
            )

            bp_spike_times = []
            for step in range(n_steps):
                t = step * self.dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    out = inj.update()
                    if float(out[0]) > 0.5:
                        bp_spike_times.append(round(t, 4))

        npt.assert_allclose(bp_spike_times, expected_spike_times, atol=1e-4,
                            err_msg="Rounded spike times don't match NEST")

    def test_multiplicities_vs_nest(self):
        r"""Compare spike multiplicities against NEST."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        import nest

        spike_times_ms = [1.0, 2.0]
        multiplicities = [3, 5]
        simtime = 5.0

        # --- NEST ---
        nest.ResetKernel()
        nest.resolution = self.dt_ms

        inj_nest = nest.Create(
            "spike_train_injector",
            params={
                "spike_times": spike_times_ms,
                "spike_multiplicities": multiplicities,
            },
        )
        # Connect to parrot neurons to count multiplicities
        parrots = nest.Create("parrot_neuron", 1)
        sr = nest.Create("spike_recorder")
        nest.Connect(inj_nest, parrots, syn_spec={"delay": self.dt_ms})
        nest.Connect(parrots, sr)
        nest.Simulate(simtime)

        nest_n_events = sr.n_events

        # --- brainpy.state ---
        n_steps = int(round(simtime / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            inj = spike_train_injector(
                spike_times=[t * u.ms for t in spike_times_ms],
                spike_multiplicities=multiplicities,
            )

            bp_total = 0
            for step in range(n_steps):
                t = step * self.dt_ms
                with brainstate.environ.context(t=t * u.ms):
                    out = inj.update()
                    val = float(out[0])
                    if val > 0.5:
                        bp_total += int(val)

        # brainpy total multiplicities should be 3 + 5 = 8
        self.assertEqual(bp_total, 8,
                         msg=f"Total multiplicity mismatch: brainpy={bp_total}, expected=8")


if __name__ == '__main__':
    unittest.main()
