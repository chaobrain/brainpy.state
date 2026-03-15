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
Tests for the ignore_and_fire neuron model.

These tests verify that the brainpy_state implementation matches the NEST
simulator's ignore_and_fire model dynamics exactly. Run with float64 and
CPU backend for numerical fidelity:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True python -m pytest ignore_and_fire_test.py
"""

import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import unittest

import brainstate
import saiunit as u
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from brainpy_state._nest.ignore_and_fire import ignore_and_fire


def _collect_spike_times(neuron, n_steps, dt):
    r"""Run neuron for n_steps and return spike times (in ms) using NEST convention.

    NEST records spike time as the delivery time t + dt.
    """
    spike_times = []
    for step_idx in range(n_steps):
        t = step_idx * dt
        with brainstate.environ.context(t=t):
            spike = neuron.update()
        if float(spike) > 0.0:
            spike_times.append(float(u.get_magnitude((t + dt) / u.ms)))
    return np.array(spike_times)


def _collect_spike_steps(neuron, n_steps, dt):
    r"""Run neuron for n_steps and return step indices where spikes occur."""
    spike_steps = []
    for step_idx in range(n_steps):
        t = step_idx * dt
        with brainstate.environ.context(t=t):
            spike = neuron.update()
        if float(spike) > 0.0:
            spike_steps.append(step_idx)
    return spike_steps


class TestIgnoreAndFireDefaults(unittest.TestCase):
    r"""Verify NEST-compatible default parameter values."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_default_phase(self):
        neuron = ignore_and_fire(1)
        self.assertEqual(float(neuron.phase), 1.0)

    def test_default_rate(self):
        neuron = ignore_and_fire(1)
        self.assertEqual(float(u.get_magnitude(neuron.rate / u.Hz)), 10.0)

    def test_phase_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            ignore_and_fire(1, phase=0.0)
        with self.assertRaises(ValueError):
            ignore_and_fire(1, phase=1.5)
        with self.assertRaises(ValueError):
            ignore_and_fire(1, phase=-0.1)

    def test_rate_non_positive_raises(self):
        with self.assertRaises(ValueError):
            ignore_and_fire(1, rate=0.0 * u.Hz)
        with self.assertRaises(ValueError):
            ignore_and_fire(1, rate=-5.0 * u.Hz)


class TestIgnoreAndFireSpikeTimes(unittest.TestCase):
    r"""Check that spike times match the NEST reference test.

    This reproduces the test from NEST's
    ``testsuite/pytests/test_ignore_and_fire_neuron.py``.
    """

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_spike_times_match_nest_reference(self):
        r"""Spike times must match the analytical prediction from NEST test."""
        rate = 5.0  # Hz
        phase = 0.4
        dt_val = 2 ** -3  # ms (= 0.125 ms)
        T = 1000.0  # ms total simulation time
        dt = dt_val * u.ms

        with brainstate.environ.context(dt=dt):
            neuron = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            neuron.init_state()

            n_steps = int(T / dt_val)
            spike_times = _collect_spike_times(neuron, n_steps, dt)

            # Theoretical spike times from NEST test
            period = 1.0 / rate * 1e3  # ms
            first_spike_time = phase * period + dt_val
            spike_times_target = np.arange(first_spike_time, T, period)

            npt.assert_array_equal(
                spike_times,
                spike_times_target,
                err_msg="Spike times do not match NEST analytical prediction."
            )

    def test_spike_times_default_rate_and_phase(self):
        r"""Default parameters: rate=10 Hz, phase=1.0, dt=0.1 ms."""
        rate = 10.0
        phase = 1.0
        dt_val = 0.1
        T = 500.0
        dt = dt_val * u.ms

        with brainstate.environ.context(dt=dt):
            neuron = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            neuron.init_state()

            n_steps = int(T / dt_val)
            spike_times = _collect_spike_times(neuron, n_steps, dt)

            period = 1.0 / rate * 1e3
            first_spike_time = phase * period + dt_val
            spike_times_target = np.arange(first_spike_time, T, period)

            npt.assert_array_equal(
                spike_times,
                spike_times_target,
            )


class TestIgnoreAndFireInputIgnored(unittest.TestCase):
    r"""Verify that all inputs are truly ignored."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_spike_times_unaffected_by_current_input(self):
        r"""Passing current input should not change spike times."""
        dt_val = 0.1
        dt = dt_val * u.ms
        T = 200.0
        rate = 20.0
        phase = 0.5

        with brainstate.environ.context(dt=dt):
            # Neuron without input
            n1 = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            n1.init_state()

            # Neuron with large current input
            n2 = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            n2.init_state()

            n_steps = int(T / dt_val)
            for step_idx in range(n_steps):
                t = step_idx * dt
                with brainstate.environ.context(t=t):
                    s1 = n1.update()
                    s2 = n2.update(x=1000.0 * u.pA)
                npt.assert_array_equal(
                    np.asarray(s1), np.asarray(s2),
                    err_msg=f"Spike mismatch at step {step_idx} when current input supplied."
                )

    def test_spike_times_unaffected_by_delta_input(self):
        r"""Delta (spike) inputs should not change spike times."""
        dt_val = 0.1
        dt = dt_val * u.ms
        T = 200.0
        rate = 20.0
        phase = 0.5

        with brainstate.environ.context(dt=dt):
            n1 = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            n1.init_state()

            n2 = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            n2.init_state()

            n_steps = int(T / dt_val)
            for step_idx in range(n_steps):
                t = step_idx * dt
                # Add delta inputs to n2
                n2.add_delta_input(f'test_{step_idx}', jnp.array([50.0]))
                with brainstate.environ.context(t=t):
                    s1 = n1.update()
                    s2 = n2.update()
                npt.assert_array_equal(
                    np.asarray(s1), np.asarray(s2),
                    err_msg=f"Spike mismatch at step {step_idx} when delta input supplied."
                )


class TestIgnoreAndFirePhaseSteps(unittest.TestCase):
    r"""Verify internal phase step counter behavior."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_phase_steps_countdown(self):
        r"""phase_steps should decrement by 1 each step until 0, then reset."""
        dt_val = 0.1
        dt = dt_val * u.ms
        rate = 10.0  # Hz => period = 100 ms => 1000 steps at dt=0.1
        phase = 1.0  # initial phase_steps = 1000 steps

        with brainstate.environ.context(dt=dt):
            neuron = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            neuron.init_state()

            initial_phase_steps = int(neuron.phase_steps.value)
            firing_period_steps = int(neuron.firing_period_steps.value)

            # Period should be 100 ms / 0.1 ms = 1000 steps
            self.assertEqual(firing_period_steps, 1000)
            # Initial phase_steps should also be 1000 (phase=1.0 * period)
            self.assertEqual(initial_phase_steps, 1000)

            # Step once: phase_steps was 1000 (not 0), so no spike, decrement
            with brainstate.environ.context(t=0.0 * u.ms):
                spike = neuron.update()
            self.assertEqual(float(spike), 0.0)
            self.assertEqual(int(neuron.phase_steps.value), initial_phase_steps - 1)

    def test_first_spike_timing(self):
        r"""First spike occurs after phase_steps countdown reaches 0."""
        dt_val = 1.0
        dt = dt_val * u.ms
        rate = 10.0  # Hz => period = 100 ms => 100 steps
        phase = 0.25  # first spike after 25 ms => step 25

        with brainstate.environ.context(dt=dt):
            neuron = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            neuron.init_state()

            initial_phase_steps = int(neuron.phase_steps.value)
            # phase_steps = round(0.25/10 * 1000 / 1.0) = 25
            self.assertEqual(initial_phase_steps, 25)

            spike_steps = _collect_spike_steps(neuron, 200, dt)

            # phase_steps starts at 25, counts down to 0
            # Step 0..24: decrement (25 steps)
            # Step 25: phase_steps is 0 -> FIRE, reset to 99
            # Step 25+100=125: phase_steps is 0 -> FIRE
            expected_spike_steps = [25, 125]
            self.assertEqual(spike_steps, expected_spike_steps)


class TestIgnoreAndFireBatch(unittest.TestCase):
    r"""Verify batch dimension handling."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_batch_dimension(self):
        r"""Neurons should maintain independent phase counters in batch mode."""
        dt = 0.1 * u.ms

        with brainstate.environ.context(dt=dt):
            neuron = ignore_and_fire(3, rate=10.0 * u.Hz, phase=1.0)
            neuron.init_state(batch_size=2)

            self.assertEqual(neuron.phase_steps.value.shape, (2, 3))
            self.assertEqual(neuron.firing_period_steps.value.shape, (2, 3))

            with brainstate.environ.context(t=0.0 * u.ms):
                spike = neuron.update()
            self.assertEqual(spike.shape, (2, 3))

    def test_multiple_neurons_different_phases(self):
        r"""Multiple neurons with different phases fire at different times."""
        dt_val = 1.0
        dt = dt_val * u.ms
        rate = 10.0  # 100 ms period

        with brainstate.environ.context(dt=dt):
            # Two neurons with different phases
            phases = np.array([0.1, 0.5])
            neuron = ignore_and_fire(2, rate=rate * u.Hz, phase=phases)
            neuron.init_state()

            spike_steps_0 = []
            spike_steps_1 = []
            for step_idx in range(200):
                t = step_idx * dt
                with brainstate.environ.context(t=t):
                    spike = neuron.update()
                if float(spike[0]) > 0:
                    spike_steps_0.append(step_idx)
                if float(spike[1]) > 0:
                    spike_steps_1.append(step_idx)

            # phase_steps for neuron 0: round(0.1/10*1000/1.0) = 10
            # phase_steps for neuron 1: round(0.5/10*1000/1.0) = 50
            self.assertEqual(spike_steps_0[0], 10)
            self.assertEqual(spike_steps_1[0], 50)

            # Both should have the same period (100 steps)
            if len(spike_steps_0) > 1:
                self.assertEqual(spike_steps_0[1] - spike_steps_0[0], 100)
            if len(spike_steps_1) > 1:
                self.assertEqual(spike_steps_1[1] - spike_steps_1[0], 100)


class TestIgnoreAndFireInterSpikeInterval(unittest.TestCase):
    r"""Verify constant inter-spike interval."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_constant_isi(self):
        r"""Inter-spike interval should be exactly firing_period_steps."""
        dt_val = 0.5
        dt = dt_val * u.ms
        rate = 50.0  # Hz => period = 20 ms => 40 steps at dt=0.5
        phase = 0.3

        with brainstate.environ.context(dt=dt):
            neuron = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            neuron.init_state()

            firing_period_steps = int(neuron.firing_period_steps.value)

            spike_steps = _collect_spike_steps(neuron, 500, dt)

            # All inter-spike intervals should be exactly firing_period_steps
            isis = np.diff(spike_steps)
            npt.assert_array_equal(
                isis,
                np.full_like(isis, firing_period_steps),
                err_msg="Inter-spike intervals are not constant."
            )


class TestIgnoreAndFireVsNEST(unittest.TestCase):
    r"""Compare against NEST simulator outputs when NEST is available."""

    @classmethod
    def setUpClass(cls):
        try:
            import nest
            cls.nest_available = True
        except ImportError:
            cls.nest_available = False

    def test_spike_times_vs_nest(self):
        r"""Compare spike times directly with NEST simulation."""
        if not self.nest_available:
            self.skipTest("NEST not installed")

        import nest

        rate = 5.0
        phase = 0.4
        dt_val = 2 ** -3  # 0.125 ms
        T = 1000.0

        # Run NEST simulation
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": dt_val})
        neuron_nest = nest.Create("ignore_and_fire", 1)
        neuron_nest.set({"rate": rate, "phase": phase})
        sr = nest.Create("spike_recorder")
        nest.Connect(neuron_nest, sr)
        nest.Simulate(T)
        nest_spike_times = nest.GetStatus(sr, "events")[0]["times"]

        # Run brainpy_state simulation
        dt = dt_val * u.ms
        with brainstate.environ.context(dt=dt):
            neuron_bp = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
            neuron_bp.init_state()

            n_steps = int(T / dt_val)
            bp_spike_times = _collect_spike_times(neuron_bp, n_steps, dt)

        npt.assert_array_equal(
            bp_spike_times,
            np.array(nest_spike_times),
            err_msg="Spike times differ between brainpy_state and NEST."
        )

    def test_spike_times_vs_nest_various_params(self):
        r"""Compare spike times for several parameter combinations."""
        if not self.nest_available:
            self.skipTest("NEST not installed")

        import nest

        param_sets = [
            {"rate": 10.0, "phase": 1.0, "dt": 0.1},
            {"rate": 20.0, "phase": 0.5, "dt": 0.1},
            {"rate": 100.0, "phase": 0.1, "dt": 0.01},
            {"rate": 2.0, "phase": 0.8, "dt": 1.0},
        ]

        for params in param_sets:
            rate = params["rate"]
            phase = params["phase"]
            dt_val = params["dt"]
            T = 500.0

            # NEST
            nest.ResetKernel()
            nest.SetKernelStatus({"resolution": dt_val})
            neuron_nest = nest.Create("ignore_and_fire", 1)
            neuron_nest.set({"rate": rate, "phase": phase})
            sr = nest.Create("spike_recorder")
            nest.Connect(neuron_nest, sr)
            nest.Simulate(T)
            nest_spike_times = nest.GetStatus(sr, "events")[0]["times"]

            # brainpy_state
            dt = dt_val * u.ms
            with brainstate.environ.context(dt=dt):
                neuron_bp = ignore_and_fire(1, rate=rate * u.Hz, phase=phase)
                neuron_bp.init_state()

                n_steps = int(T / dt_val)
                bp_spike_times = _collect_spike_times(neuron_bp, n_steps, dt)

            # Same number of spikes
            self.assertEqual(
                len(bp_spike_times), len(nest_spike_times),
                msg=f"Spike count differs for params: {params}"
            )
            # Spike times match (allowing for float64 rounding in time accumulation)
            npt.assert_allclose(
                bp_spike_times,
                np.array(nest_spike_times),
                atol=1e-12,
                err_msg=f"Spike times differ for params: {params}"
            )


if __name__ == '__main__':
    unittest.main()
