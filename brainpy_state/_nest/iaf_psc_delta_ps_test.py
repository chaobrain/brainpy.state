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

import math
import unittest

import brainstate
import braintools
import saiunit as u
import jax.numpy as jnp
from brainpy.state import iaf_psc_delta_ps


class TestIAFPscDeltaPS(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def _step(self, neuron, step_idx, x=0.0 * u.pA, delta=None, spike_events=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        dt = brainstate.environ.get_dt()
        with brainstate.environ.context(t=step_idx * dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_nest_default_parameters(self):
        neuron = iaf_psc_delta_ps(1)
        self.assertEqual(neuron.E_L, -70. * u.mV)
        self.assertEqual(neuron.C_m, 250. * u.pF)
        self.assertEqual(neuron.tau_m, 10. * u.ms)
        self.assertEqual(neuron.t_ref, 2. * u.ms)
        self.assertEqual(neuron.V_th, -55. * u.mV)
        self.assertEqual(neuron.V_reset, -70. * u.mV)
        self.assertEqual(neuron.I_e, 0. * u.pA)
        self.assertIsNone(neuron.V_min)
        self.assertFalse(neuron.refractory_input)
        self.assertEqual(neuron.spk_reset, 'hard')

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            iaf_psc_delta_ps(1, V_reset=-55. * u.mV, V_th=-55. * u.mV)
        with self.assertRaises(ValueError):
            iaf_psc_delta_ps(1, C_m=0.0 * u.pF)
        with self.assertRaises(ValueError):
            iaf_psc_delta_ps(1, tau_m=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_delta_ps(1, V_reset=-80. * u.mV, V_min=-79. * u.mV)

    def test_refractory_time_must_be_at_least_one_step(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta_ps(1, t_ref=0.05 * u.ms)
            neuron.init_state()
            with self.assertRaises(ValueError):
                self._step(neuron, 0)

    def test_current_input_is_one_step_delayed(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta_ps(
                1,
                E_L=0. * u.mV,
                I_e=0. * u.pA,
                C_m=250. * u.pF,
                tau_m=10. * u.ms,
                V_th=1e9 * u.mV,
                V_reset=0. * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, x=100. * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0. * u.mV))

            self._step(neuron, 1, x=0. * u.pA)
            expected = (100. * u.pA) * (10. * u.ms) / (250. * u.pF) * (1. - u.math.exp(-self.dt / (10. * u.ms)))
            self.assertTrue(u.math.allclose(neuron.V.value, expected))

    def test_dc_voltage_accuracy_matches_analytic(self):
        for dt in [1.0, 0.5, 0.1, 0.01]:
            dt_q = dt * u.ms
            for duration, atol in [(5.0, 5e-6), (100.0, 1e-4)]:
                with brainstate.environ.context(dt=dt_q):
                    neuron = iaf_psc_delta_ps(
                        1,
                        E_L=0. * u.mV,
                        V_th=2000. * u.mV,
                        V_reset=0. * u.mV,
                        V_initializer=braintools.init.Constant(0. * u.mV),
                        I_e=1000. * u.pA,
                        tau_m=10. * u.ms,
                        C_m=250. * u.pF,
                    )
                    neuron.init_state()

                    steps = int(round(duration / dt))
                    for k in range(steps):
                        self._step(neuron, k)

                    expected = 1000. * 10. / 250. * (1. - math.exp(-duration / 10.))
                    vm = float((neuron.V.value / u.mV)[0])
                    self.assertAlmostEqual(vm, expected, delta=atol)

    def test_first_spike_time_from_dc_current_matches_analytic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta_ps(
                1,
                E_L=0. * u.mV,
                V_th=15. * u.mV,
                V_reset=0. * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV),
                I_e=1000. * u.pA,
                tau_m=10. * u.ms,
                C_m=250. * u.pF,
            )
            neuron.init_state()

            for k in range(80):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    break

            expected_t = -10.0 * math.log(1.0 - (250.0 * 15.0) / (10.0 * 1000.0))
            t_spike = float((neuron.last_spike_time.value / u.ms)[0])
            self.assertAlmostEqual(t_spike, expected_t, delta=1e-6)

    def test_empirical_trace_matches_nest_compare_delta_case(self):
        dt = 0.01 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = iaf_psc_delta_ps(
                1,
                E_L=-49. * u.mV,
                V_th=-50. * u.mV,
                V_reset=-60. * u.mV,
                C_m=200. * u.pF,
                tau_m=20. * u.ms,
                t_ref=5. * u.ms,
                V_initializer=braintools.init.Constant(-60. * u.mV),
            )
            neuron.init_state()

            # NEST test uses spike_generator with delay 0.1 ms and weight 2.5 mV.
            input_times = [1.1, 2.1, 3.1, 4.1, 5.1, 10.6, 12.1]
            # add_delta_input() is applied at step end (t + dt), so an event at
            # absolute time T must be queued in step index (T - dt) / dt.
            step_ids = {int(round((t - 0.01) / 0.01)) for t in input_times}

            first_spike_time = None
            for k in range(int(round(200.0 / 0.01))):
                if k in step_ids:
                    spk = self._step(neuron, k, delta=2.5 * u.mV)
                else:
                    spk = self._step(neuron, k)
                if first_spike_time is None and self._is_spike(spk):
                    first_spike_time = float((neuron.last_spike_time.value / u.ms)[0])

            self.assertIsNotNone(first_spike_time)
            self.assertAlmostEqual(first_spike_time, 4.1, delta=1e-6)

    def test_refractory_input_discounting_with_precise_event(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta_ps(
                1,
                E_L=-70. * u.mV,
                V_th=-40. * u.mV,
                V_reset=-70. * u.mV,
                tau_m=10. * u.ms,
                t_ref=0.3 * u.ms,  # floor(t_ref / dt) = 3 steps in NEST precise model
                refractory_input=True,
                V_initializer=braintools.init.Constant(-56. * u.mV),
            )
            neuron.init_state()

            # Step 0: trigger instant spike at end of step.
            self.assertTrue(self._is_spike(self._step(neuron, 0, delta=20. * u.mV)))
            # Step 1: refractory spike arrives at offset 0.04 ms.
            self._step(neuron, 1, spike_events=[(0.04 * u.ms, 10. * u.mV)])
            buffered = float((neuron.refractory_spike_buffer.value / u.mV)[0])
            # Advance until refractory ends, then validate buffered jump.
            release_step = None
            for k in range(2, 8):
                self._step(neuron, k)
                if not bool(neuron.is_refractory.value[0]):
                    release_step = k
                    break

            self.assertIsNotNone(release_step)
            expected_v = -70.0 + buffered
            self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), expected_v, delta=1e-6)
            self.assertAlmostEqual(float((neuron.refractory_spike_buffer.value / u.mV)[0]), 0.0, delta=1e-9)

    def test_super_threshold_initialization_spikes_immediately(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta_ps(
                1,
                V_th=-55. * u.mV,
                V_reset=-70. * u.mV,
                V_initializer=braintools.init.Constant(-50. * u.mV),
            )
            neuron.init_state()
            spk = self._step(neuron, 0)
            self.assertTrue(self._is_spike(spk))
            t_spike = float((neuron.last_spike_time.value / u.ms)[0])
            self.assertGreaterEqual(t_spike, 0.0)
            self.assertLess(t_spike, 1e-9)

    def test_shape_and_state_variables(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta_ps(4, ref_var=True)
            neuron.init_state(batch_size=3)
            self.assertEqual(neuron.V.value.shape, (3, 4))
            self.assertEqual(neuron.I_stim.value.shape, (3, 4))
            self.assertEqual(neuron.last_spike_time.value.shape, (3, 4))
            self.assertEqual(neuron.last_spike_step.value.shape, (3, 4))
            self.assertEqual(neuron.last_spike_offset.value.shape, (3, 4))
            self.assertEqual(neuron.is_refractory.value.shape, (3, 4))
            self.assertEqual(neuron.refractory_spike_buffer.value.shape, (3, 4))
            self.assertEqual(neuron.refractory.value.shape, (3, 4))

    def test_refractory_state_updates(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta_ps(
                1,
                t_ref=0.2 * u.ms,
                V_th=-55. * u.mV,
                V_reset=-70. * u.mV,
                V_initializer=braintools.init.Constant(-56. * u.mV),
            )
            neuron.init_state()

            self.assertFalse(bool(neuron.is_refractory.value[0]))
            self._step(neuron, 0, delta=2. * u.mV)
            self.assertTrue(bool(neuron.is_refractory.value[0]))
            self._step(neuron, 1)
            self._step(neuron, 2)
            self.assertFalse(bool(neuron.is_refractory.value[0]))

    def test_spike_event_offset_validation(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta_ps(1)
            neuron.init_state()
            with self.assertRaises(ValueError):
                self._step(neuron, 0, spike_events=[(-0.01 * u.ms, 1.0 * u.mV)])
            with self.assertRaises(ValueError):
                self._step(neuron, 1, spike_events=[(0.11 * u.ms, 1.0 * u.mV)])

    def test_batch_dc_evolution(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta_ps(
                2,
                E_L=0. * u.mV,
                V_th=1e9 * u.mV,
                V_reset=0. * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV),
                I_e=jnp.asarray([1000., 500.]) * u.pA,
                tau_m=10. * u.ms,
                C_m=250. * u.pF,
            )
            neuron.init_state()
            for k in range(20):
                self._step(neuron, k)

            vm = neuron.V.value / u.mV
            self.assertTrue(vm[0] > vm[1])


if __name__ == '__main__':
    unittest.main()
