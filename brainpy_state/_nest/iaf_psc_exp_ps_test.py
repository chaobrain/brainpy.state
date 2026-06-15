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
import brainunit as u
import jax
from brainpy.state import iaf_psc_exp_ps

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class TestIAFPscExpPS(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def _step(self, neuron, step_idx, x=0. * u.pA, spike_events=None):
        with brainstate.environ.context(t=step_idx * brainstate.environ.get_dt()):
            return neuron.update(x=x, spike_events=spike_events)

    def test_default_parameters(self):
        n = iaf_psc_exp_ps(1)
        self.assertEqual(n.E_L, -70. * u.mV)
        self.assertEqual(n.C_m, 250. * u.pF)
        self.assertEqual(n.tau_m, 10. * u.ms)
        self.assertEqual(n.tau_syn_ex, 2. * u.ms)
        self.assertEqual(n.tau_syn_in, 2. * u.ms)
        self.assertEqual(n.t_ref, 2. * u.ms)
        self.assertEqual(n.V_th, -55. * u.mV)
        self.assertEqual(n.V_reset, -70. * u.mV)

    def test_dc_spike_time_matches_analytic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_exp_ps(
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

            first_spike = None
            for k in range(80):
                spk = self._step(neuron, k)
                if first_spike is None and self._is_spike(spk):
                    first_spike = float((neuron.last_spike_time.value / u.ms)[0])
                    break

            expected_t = -10.0 * math.log(1.0 - (250.0 * 15.0) / (10.0 * 1000.0))
            self.assertIsNotNone(first_spike)
            self.assertAlmostEqual(first_spike, expected_t, delta=1e-6)

    def test_off_grid_psp_matches_analytic(self):
        # NEST-style PSP accuracy check for a precise off-grid input spike.
        with brainstate.environ.context(dt=self.dt):
            tau_syn = 0.3
            tau_m = 10.0
            c_m = 250.0
            w = 500.0
            t_event = 2.132123512  # ms

            n = iaf_psc_exp_ps(
                1,
                E_L=0. * u.mV,
                V_th=15. * u.mV,
                V_reset=0. * u.mV,
                I_e=0. * u.pA,
                tau_m=tau_m * u.ms,
                tau_syn_ex=tau_syn * u.ms,
                tau_syn_in=tau_syn * u.ms,
                C_m=c_m * u.pF,
                V_initializer=braintools.init.Constant(0. * u.mV),
            )
            n.init_state()

            T1 = 3.0
            T2 = 6.0
            i1 = int(round(T1 / 0.1))
            i2 = int(round(T2 / 0.1))
            v_t1 = None
            v_t2 = None
            for k in range(i2):
                t0 = k * 0.1
                t1 = (k + 1) * 0.1
                events = None
                if t0 <= t_event <= t1:
                    offs = t1 - t_event
                    events = [(offs * u.ms, w * u.pA)]
                self._step(n, k, spike_events=events)
                if k + 1 == i1:
                    v_t1 = float((n.V.value / u.mV)[0])
                if k + 1 == i2:
                    v_t2 = float((n.V.value / u.mV)[0])

            def vm_ref(t_abs):
                dt = t_abs - t_event
                if dt < 0.0:
                    return 0.0
                return w / (c_m * (1.0 / tau_syn - 1.0 / tau_m)) * (
                    math.exp(-dt / tau_m) - math.exp(-dt / tau_syn)
                )

            self.assertIsNotNone(v_t1)
            self.assertIsNotNone(v_t2)
            self.assertAlmostEqual(v_t1, vm_ref(T1), delta=5e-6)
            self.assertAlmostEqual(v_t2, vm_ref(T2), delta=5e-6)


if __name__ == '__main__':
    unittest.main()
