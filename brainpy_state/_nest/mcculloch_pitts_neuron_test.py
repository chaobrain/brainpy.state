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

import os

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import unittest

import braintools
import brainstate
import saiunit as u
import jax.numpy as jnp

from brainpy_state._nest.mcculloch_pitts_neuron import mcculloch_pitts_neuron


class TestMcCullochPittsNeuron(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0.0 * u.mV, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    # ------------------------------------------------------------------
    # Parameter defaults
    # ------------------------------------------------------------------

    def test_nest_default_parameters(self):
        r"""Default parameters must match NEST: tau_m=10ms, theta=0mV."""
        neuron = mcculloch_pitts_neuron(1)
        self.assertEqual(neuron.tau_m, 10. * u.ms)
        self.assertEqual(neuron.theta, 0. * u.mV)
        self.assertFalse(neuron.stochastic_update)

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def test_initial_state_defaults(self):
        r"""y should start at 0.0 and h at 0.0 mV by default."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1)
            neuron.init_state()
            self.assertTrue(u.math.allclose(neuron.y.value, jnp.array([0.0])))
            self.assertTrue(u.math.allclose(neuron.h.value, 0.0 * u.mV))

    def test_shape_without_batch(self):
        r"""State shapes should match in_size when no batch."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(4)
            neuron.init_state()
            self.assertEqual(neuron.y.value.shape, (4,))
            self.assertEqual(neuron.h.value.shape, (4,))

    def test_shape_with_batch(self):
        r"""State shapes should be (batch_size, in_size) with batch."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(4)
            neuron.init_state(batch_size=3)
            self.assertEqual(neuron.y.value.shape, (3, 4))
            self.assertEqual(neuron.h.value.shape, (3, 4))

    # ------------------------------------------------------------------
    # Heaviside activation function
    # ------------------------------------------------------------------

    def test_heaviside_above_threshold(self):
        r"""Input above theta -> output 1."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
            neuron.init_state()
            out = self._step(neuron, 0, x=1.0 * u.mV)
            self.assertEqual(float(out[0]), 1.0)

    def test_heaviside_below_threshold(self):
        r"""Input below theta -> output 0."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
            neuron.init_state()
            out = self._step(neuron, 0, x=0.2 * u.mV)
            self.assertEqual(float(out[0]), 0.0)

    def test_heaviside_at_threshold(self):
        r"""Input exactly at theta -> output 0 (strict inequality h > theta in NEST)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
            neuron.init_state()
            out = self._step(neuron, 0, x=0.5 * u.mV)
            self.assertEqual(float(out[0]), 0.0)

    def test_heaviside_negative_theta(self):
        r"""With theta < 0, zero input should produce output 1."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=-10.0 * u.mV)
            neuron.init_state()
            out = self._step(neuron, 0, x=0.0 * u.mV)
            self.assertEqual(float(out[0]), 1.0)

    # ------------------------------------------------------------------
    # Delta input handling (binary spike events)
    # ------------------------------------------------------------------

    def test_delta_input_accumulates_in_h(self):
        r"""Delta inputs should accumulate in h across steps."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=1.5 * u.mV)
            neuron.init_state()

            # First delta: h = 0 + 1.0 = 1.0, below theta=1.5 -> y=0
            out = self._step(neuron, 0, delta=1.0 * u.mV)
            self.assertEqual(float(out[0]), 0.0)
            self.assertTrue(u.math.allclose(neuron.h.value, 1.0 * u.mV))

            # Second delta: h = 1.0 + 1.0 = 2.0, above theta=1.5 -> y=1
            out = self._step(neuron, 1, delta=1.0 * u.mV)
            self.assertEqual(float(out[0]), 1.0)
            self.assertTrue(u.math.allclose(neuron.h.value, 2.0 * u.mV))

    def test_delta_input_negative_brings_below_threshold(self):
        r"""Negative delta input (down-transition) should reduce h."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
            neuron.init_state()

            # Push above threshold
            out = self._step(neuron, 0, delta=1.0 * u.mV)
            self.assertEqual(float(out[0]), 1.0)

            # Negative delta brings below threshold
            out = self._step(neuron, 1, delta=-1.0 * u.mV)
            self.assertEqual(float(out[0]), 0.0)
            self.assertTrue(u.math.allclose(neuron.h.value, 0.0 * u.mV))

    # ------------------------------------------------------------------
    # Current input handling
    # ------------------------------------------------------------------

    def test_current_input_adds_to_h_at_update(self):
        r"""Current input x is added to h at the update point."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
            neuron.init_state()

            # x=1.0 mV, h=0 -> h+c = 1.0 > 0.5 -> y=1
            out = self._step(neuron, 0, x=1.0 * u.mV)
            self.assertEqual(float(out[0]), 1.0)

    def test_current_input_does_not_persist_in_h(self):
        r"""Current input should not accumulate in h across steps (only delta does)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
            neuron.init_state()

            # Step with current input
            self._step(neuron, 0, x=1.0 * u.mV)
            # h should still be 0 (current is not accumulated into h)
            self.assertTrue(u.math.allclose(neuron.h.value, 0.0 * u.mV))

            # Without current input, h=0 <= theta=0.5 -> y=0
            out = self._step(neuron, 1, x=0.0 * u.mV)
            self.assertEqual(float(out[0]), 0.0)

    # ------------------------------------------------------------------
    # State transitions (matching NEST test_binary.py semantics)
    # ------------------------------------------------------------------

    def test_up_transition(self):
        r"""Neuron transitions from 0 to 1 when input exceeds threshold."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.0 * u.mV)
            neuron.init_state()

            # h=0, theta=0, h > theta is False -> y=0
            out = self._step(neuron, 0, x=0.0 * u.mV)
            self.assertEqual(float(out[0]), 0.0)

            # Add positive delta: h=1.0 > 0.0 -> y=1
            out = self._step(neuron, 1, delta=1.0 * u.mV)
            self.assertEqual(float(out[0]), 1.0)

    def test_down_transition(self):
        r"""Neuron transitions from 1 to 0 when input drops below threshold."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
            neuron.init_state()

            # Push above threshold
            self._step(neuron, 0, delta=1.0 * u.mV)
            self.assertEqual(float(neuron.y.value[0]), 1.0)

            # Negative delta: h = 1.0 - 1.0 = 0.0 <= 0.5 -> y=0
            out = self._step(neuron, 1, delta=-1.0 * u.mV)
            self.assertEqual(float(out[0]), 0.0)

    def test_state_persists_without_input_change(self):
        r"""State should remain stable when h doesn't change."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
            neuron.init_state()

            # Push above threshold
            self._step(neuron, 0, delta=1.0 * u.mV)
            self.assertEqual(float(neuron.y.value[0]), 1.0)

            # No input change -> state persists
            for step in range(1, 5):
                out = self._step(neuron, step)
                self.assertEqual(float(out[0]), 1.0)

    # ------------------------------------------------------------------
    # Multi-neuron behavior
    # ------------------------------------------------------------------

    def test_multi_neuron_independent(self):
        r"""Multiple neurons should update independently."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(3, theta=0.5 * u.mV)
            neuron.init_state()

            # Different delta inputs per neuron
            delta = jnp.array([1.0, 0.3, 0.6]) * u.mV
            neuron.add_delta_input('delta_0', delta)
            with brainstate.environ.context(t=0.0 * u.ms):
                out = neuron.update(x=0.0 * u.mV)

            # neuron 0: h=1.0 > 0.5 -> 1
            # neuron 1: h=0.3 <= 0.5 -> 0
            # neuron 2: h=0.6 > 0.5 -> 1
            expected = jnp.array([1.0, 0.0, 1.0])
            self.assertTrue(jnp.allclose(out, expected))

    # ------------------------------------------------------------------
    # Comparison with NEST reference trace
    # ------------------------------------------------------------------

    def test_matches_nest_binary_communication_pattern(self):
        r"""Reproduce the NEST test_binary.py expected h trace.

        In NEST's test, a spike_generator sends spikes at t=10.0 and t=10.0
        (double spike = up-transition, weight +1) and t=15.0 (single spike =
        down-transition, weight -1). The expected h trace recorded by the
        multimeter (at 1ms intervals) is:

            [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1, 0,0,0,0]

        We reproduce this using dt=1.0ms with 19 steps, applying delta inputs
        at the corresponding steps. In NEST's causality scheme, incoming events
        at time t are processed at the beginning of the next step, so the
        up-transition delta at step 9 (t=10ms) makes h=1 visible from step 10
        onward. We model this by applying the delta one step before the
        expected change appears in the trace.
        """
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.0 * u.mV)
            neuron.init_state()

            n_steps = 19
            h_trace = []

            for step in range(n_steps):
                # Apply delta inputs matching NEST's binary spike encoding:
                # Up-transition (double spike) at t=10ms -> delta at step 9
                # Down-transition (single spike) at t=15ms -> delta at step 14
                if step == 9:
                    delta = 1.0 * u.mV
                elif step == 14:
                    delta = -1.0 * u.mV
                else:
                    delta = None

                self._step(neuron, step, delta=delta)
                h_trace.append(float((neuron.h.value / u.mV)[0]))

            # h should be 0 for steps 0-8, 1 for steps 9-13, 0 for steps 14-18
            expected = [0.0] * 9 + [1.0] * 5 + [0.0] * 5
            for i, (got, exp) in enumerate(zip(h_trace, expected)):
                self.assertAlmostEqual(got, exp, places=10,
                                       msg=f"h mismatch at step {i}: got {got}, expected {exp}")

    def test_matches_nest_state_change_with_negative_theta(self):
        r"""Reproduce NEST test_binary_neuron_state_change.

        With theta=-10.0 and tau_m=1.0 (deterministic mode), the neuron
        should transition to state 1 immediately since h=0 > theta=-10.
        """
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(
                1,
                theta=-10.0 * u.mV,
                tau_m=1.0 * u.ms,
            )
            neuron.init_state()

            # First step: h=0 > theta=-10 -> y=1
            out = self._step(neuron, 0)
            self.assertEqual(float(out[0]), 1.0)

            # Change theta to +10: h=0 <= theta=10 -> y=0
            neuron.theta = braintools.init.param(10.0 * u.mV, neuron.varshape)
            out = self._step(neuron, 1)
            self.assertEqual(float(out[0]), 0.0)

    # ------------------------------------------------------------------
    # Custom initializer
    # ------------------------------------------------------------------

    def test_custom_y_initializer(self):
        r"""y_initializer should set the initial binary state."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(
                1,
                y_initializer=braintools.init.Constant(1.0),
            )
            neuron.init_state()
            self.assertEqual(float(neuron.y.value[0]), 1.0)

    # ------------------------------------------------------------------
    # Zero theta edge case
    # ------------------------------------------------------------------

    def test_zero_theta_zero_input(self):
        r"""With theta=0 and h=0, output should be 0 (strict inequality)."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.0 * u.mV)
            neuron.init_state()
            out = self._step(neuron, 0, x=0.0 * u.mV)
            self.assertEqual(float(out[0]), 0.0)

    def test_zero_theta_positive_input(self):
        r"""With theta=0 and h>0, output should be 1."""
        with brainstate.environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.0 * u.mV)
            neuron.init_state()
            out = self._step(neuron, 0, x=0.001 * u.mV)
            self.assertEqual(float(out[0]), 1.0)


class TestMcCullochPittsStochasticForLoop(unittest.TestCase):
    r"""Stochastic (Poisson-timed) updates must work inside ``for_loop``.

    Regression tests for a bug where stochastic mode drew its inter-update
    interval from ``brainstate.environ.get('key')``, which is *not* threaded by
    ``brainstate.transform.for_loop``. With no key, ``t_next`` never advanced
    (it stays at its ``-1e7`` init), so ``should_update`` was always true and the
    neuron updated every step — collapsing the Poisson(tau_m) gate to
    deterministic every-step dynamics and ignoring the seed entirely. The fix
    gives the neuron a self-managed ``rng_key`` state (+ ``rng_seed`` argument),
    mirroring ``ginzburg_neuron``.
    """

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms
        self.tau_m = 10.0 * u.ms

    def _run(self, drive, rng_seed):
        import numpy as np
        from brainstate import transform, environ
        n_steps = drive.shape[0]
        times = u.math.arange(n_steps) * self.dt
        brainstate.random.seed(rng_seed)
        with environ.context(dt=self.dt):
            neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV, tau_m=self.tau_m,
                                            stochastic_update=True, rng_seed=rng_seed)
            brainstate.nn.init_all_states(neuron)

            def step(t, x):
                with environ.context(t=t):
                    return neuron.update(x=x * u.mV)

            return np.asarray(transform.for_loop(step, times, drive)).reshape(-1)

    def test_stochastic_update_is_poisson_gated_in_for_loop(self):
        import numpy as np
        # A drive that flips 0<->1 every 50 steps (5 ms) over 2000 ms = 400 flips.
        # A correctly Poisson(tau_m=10 ms)-gated neuron samples the drive only
        # ~once per 100 steps, so it cannot track every flip; a buggy every-step
        # neuron tracks all ~399 flips.
        n_steps = 20000
        drive = ((np.arange(n_steps) // 50) % 2).astype(float)
        y = self._run(drive, rng_seed=0)
        n_changes = int(np.sum(np.abs(np.diff(y)) > 0.5))
        drive_flips = int(np.sum(np.abs(np.diff(drive)) > 0.5))
        self.assertGreater(drive_flips, 350)               # the drive really does flip a lot
        self.assertLess(n_changes, drive_flips - 100)      # but the neuron does NOT track every flip

    def test_stochastic_update_depends_on_seed(self):
        import numpy as np
        # Constant supra-threshold drive: a Poisson-gated neuron turns on at its
        # first (random) update time, which differs across seeds.
        n_steps = 4000
        drive = np.ones(n_steps)
        y0 = self._run(drive, rng_seed=0)
        y1 = self._run(drive, rng_seed=7)
        self.assertFalse(np.array_equal(y0, y1))           # the seed must matter


if __name__ == '__main__':
    unittest.main()
