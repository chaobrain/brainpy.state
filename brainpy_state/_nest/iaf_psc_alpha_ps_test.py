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

from brainpy.state import iaf_psc_alpha_ps

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class TestIAFPscAlphaPS(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    @staticmethod
    def _step(neuron, step_idx, spike_events=None):
        with brainstate.environ.context(t=step_idx * brainstate.environ.get_dt()):
            return neuron.update(spike_events=spike_events)

    def test_default_parameters(self):
        n = iaf_psc_alpha_ps(1)
        self.assertEqual(n.E_L, -70. * u.mV)
        self.assertEqual(n.C_m, 250. * u.pF)
        self.assertEqual(n.tau_m, 10. * u.ms)
        self.assertEqual(n.tau_syn_ex, 2. * u.ms)
        self.assertEqual(n.tau_syn_in, 2. * u.ms)
        self.assertEqual(n.t_ref, 2. * u.ms)
        self.assertEqual(n.V_th, -55. * u.mV)
        self.assertEqual(n.V_reset, -70. * u.mV)

    def test_off_grid_psp_accuracy(self):
        # Mirrors NEST test_iaf_ps_psp_accuracy style check.
        tau_syn = 0.3
        tau_m = 10.0
        c_m = 250.0
        delay = 1.0
        weight = 500.0
        spike_emission = 1.132123512
        t_impact = spike_emission + delay

        for dt in [1.0, 0.1, 0.01]:
            with brainstate.environ.context(dt=dt * u.ms):
                n = iaf_psc_alpha_ps(
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
                i2 = int(round(T2 / dt))
                v_t1 = None
                v_t2 = None
                for k in range(i2):
                    t0 = k * dt
                    t1 = (k + 1) * dt
                    events = None
                    if t0 <= t_impact <= t1:
                        offs = t1 - t_impact
                        events = [(offs * u.ms, weight * u.pA)]
                    self._step(n, k, spike_events=events)
                    if abs((k + 1) * dt - T1) < 1e-12:
                        v_t1 = float((n.V.value / u.mV)[0])
                    if abs((k + 1) * dt - T2) < 1e-12:
                        v_t2 = float((n.V.value / u.mV)[0])

                def v_ref(t_abs):
                    t = t_abs - t_impact
                    if t < 0.0:
                        return 0.0
                    pref = weight * math.e / (tau_syn * c_m)
                    term1 = (math.exp(-t / tau_m) - math.exp(-t / tau_syn)) / (1.0 / tau_syn - 1.0 / tau_m) ** 2
                    term2 = t * math.exp(-t / tau_syn) / (1.0 / tau_syn - 1.0 / tau_m)
                    return pref * (term1 - term2)

                self.assertIsNotNone(v_t1)
                self.assertIsNotNone(v_t2)
                self.assertAlmostEqual(v_t1, v_ref(T1), delta=2e-5)
                self.assertAlmostEqual(v_t2, v_ref(T2), delta=2e-5)


if __name__ == '__main__':
    unittest.main()
