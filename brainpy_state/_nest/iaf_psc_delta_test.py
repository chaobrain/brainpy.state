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
import jax.numpy as jnp
import numpy as np
import saiunit as u
from brainpy.state import iaf_psc_delta


class TestIAFPscDelta(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def _step(self, neuron, step_idx, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_nest_default_parameters(self):
        neuron = iaf_psc_delta(1)
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

    def test_passive_membrane_evolution(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta(
                1,
                E_L=-70. * u.mV,
                tau_m=10. * u.ms,
                V_th=-40. * u.mV,
                V_initializer=braintools.init.Constant(-60. * u.mV),
                t_ref=0. * u.ms,
            )
            neuron.init_state()
            self._step(neuron, 0, x=0. * u.pA)

            decay = u.math.exp(-self.dt / (10. * u.ms))
            expected = -70. * u.mV + (10. * u.mV) * decay
            self.assertTrue(u.math.allclose(neuron.V.value, expected))

    def test_delta_input_spikes_and_resets_immediately(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta(
                1,
                V_th=-55. * u.mV,
                V_reset=-70. * u.mV,
                V_initializer=braintools.init.Constant(-56. * u.mV),
            )
            neuron.init_state()
            spk = self._step(neuron, 0, delta=2. * u.mV)

            self.assertTrue(self._is_spike(spk))
            self.assertTrue(u.math.allclose(neuron.V.value, -70. * u.mV))

    def test_inputs_during_refractory_are_discarded_by_default(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta(
                1,
                E_L=-70. * u.mV,
                V_th=-55. * u.mV,
                V_reset=-70. * u.mV,
                tau_m=10. * u.ms,
                t_ref=0.25 * u.ms,  # ceil(t_ref / dt) = 3 steps in NEST
                V_initializer=braintools.init.Constant(-56. * u.mV),
            )
            neuron.init_state()

            # Trigger a spike first.
            self.assertTrue(self._is_spike(self._step(neuron, 0, delta=2. * u.mV)))

            # Three subsequent steps are refractory and should ignore spike jumps.
            # Pre-compute per-step delta inputs and run via for_loop.
            delta_arr = jnp.full((3,), 20.)

            def _ref_step(k):
                neuron.add_delta_input('delta', delta_arr[k] * u.mV)
                with brainstate.environ.context(t=(1 + k) * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return spk

            spk_all = brainstate.transform.for_loop(_ref_step, jnp.arange(3))
            for i in range(3):
                self.assertFalse(self._is_spike(spk_all[i]))

            # At the next step, no refractory residual should remain.
            self._step(neuron, 4)
            self.assertTrue(u.math.allclose(neuron.V.value, -70. * u.mV))

    def test_refractory_input_matches_nest_discounting(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta(
                1,
                E_L=-70. * u.mV,
                V_th=-40. * u.mV,
                V_reset=-70. * u.mV,
                tau_m=10. * u.ms,
                t_ref=0.2 * u.ms,  # 2 steps
                refractory_input=True,
                V_initializer=braintools.init.Constant(-56. * u.mV),
            )
            neuron.init_state()

            # Spike at step 0.
            self.assertTrue(self._is_spike(self._step(neuron, 0, delta=20. * u.mV)))
            # Spike during refractory at step 1 should be accumulated with exp(-r*dt/tau), r=2.
            self._step(neuron, 1, delta=10. * u.mV)
            # Final refractory step.
            self._step(neuron, 2)
            # Re-enter non-refractory mode; buffered jump is applied now.
            self._step(neuron, 3)

            expected_jump = 10. * u.mV * u.math.exp(-2. * self.dt / (10. * u.ms))
            expected_v = -70. * u.mV + expected_jump
            self.assertTrue(u.math.allclose(neuron.V.value, expected_v))

    def test_last_spike_time_uses_t_plus_dt(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta(
                1,
                V_th=-55. * u.mV,
                V_reset=-70. * u.mV,
                V_initializer=braintools.init.Constant(-56. * u.mV),
            )
            neuron.init_state()
            self._step(neuron, 0, delta=2. * u.mV)
            self.assertTrue(u.math.allclose(neuron.last_spike_time.value, self.dt))

    def test_matches_nest_reference_trace(self):
        r"""Regression: validate a deterministic trace against NEST step equations."""
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta(
                1,
                E_L=-70. * u.mV,
                C_m=250. * u.pF,
                tau_m=10. * u.ms,
                t_ref=0.25 * u.ms,
                V_th=-55. * u.mV,
                V_reset=-70. * u.mV,
                I_e=0. * u.pA,
                refractory_input=True,
                V_initializer=braintools.init.Constant(-56. * u.mV),
            )
            neuron.init_state()

            delta_seq = [2., 20., 0., 0., 0.]

            # Pre-compute per-step delta inputs as a JAX array and run via for_loop.
            delta_arr = jnp.array(delta_seq)

            def _run_step(k):
                neuron.add_delta_input('delta', delta_arr[k] * u.mV)
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0. * u.pA)
                return (neuron.V.value / u.mV)[0], spk[0]

            results = brainstate.transform.for_loop(_run_step, jnp.arange(5))
            v_all = np.asarray(results[0])
            s_all = np.asarray(results[1])
            model_v = [float(v_all[i]) for i in range(5)]
            model_s = [float(s_all[i]) for i in range(5)]

            E_L = -70.0
            C_m = 250.0
            tau = 10.0
            V_th = -55.0 - E_L
            V_reset = -70.0 - E_L
            P33 = math.exp(-0.1 / tau)
            P30 = (1.0 - P33) * tau / C_m
            refr_steps = math.ceil(0.25 / 0.1)
            y3 = -56.0 - E_L
            y0 = 0.0
            r = 0
            refr_buf = 0.0
            ref_v = []
            ref_s = []
            for dv in delta_seq:
                if r == 0:
                    y3 = P30 * y0 + P33 * y3 + dv + refr_buf
                    refr_buf = 0.0
                else:
                    refr_buf += dv * math.exp(-r * 0.1 / tau)
                    r -= 1
                spike = 1.0 if y3 >= V_th else 0.0
                if spike:
                    r = refr_steps
                    y3 = V_reset
                ref_v.append(y3 + E_L)
                ref_s.append(spike)

            self.assertEqual(model_s, ref_s)
            self.assertTrue(all(abs(v1 - v2) < 1e-9 for v1, v2 in zip(model_v, ref_v)))


if __name__ == '__main__':
    unittest.main()
