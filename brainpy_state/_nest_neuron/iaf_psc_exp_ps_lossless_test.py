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
from brainpy.state import iaf_psc_exp_ps, iaf_psc_exp_ps_lossless

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)


class TestIAFPscExpPSLossless(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def _step(self, neuron, step_idx, x=0. * u.pA, spike_events=None):
        with brainstate.environ.context(t=step_idx * brainstate.environ.get_dt()):
            return neuron.update(x=x, spike_events=spike_events)

    def test_dc_spike_time_matches_analytic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_exp_ps_lossless(
                1,
                E_L=0. * u.mV,
                V_th=15. * u.mV,
                V_reset=0. * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV),
                I_e=1000. * u.pA,
                tau_m=10. * u.ms,
                C_m=250. * u.pF,
                tau_syn_ex=2. * u.ms,
                tau_syn_in=2. * u.ms,
            )
            neuron.init_state()
            first_spike = None
            for k in range(80):
                spk = self._step(neuron, k)
                if self._is_spike(spk):
                    first_spike = float((neuron.last_spike_time.value / u.ms)[0])
                    break
            expected_t = -10.0 * math.log(1.0 - (250.0 * 15.0) / (10.0 * 1000.0))
            self.assertIsNotNone(first_spike)
            self.assertAlmostEqual(first_spike, expected_t, delta=1e-6)

    def test_lossless_can_detect_spike_missed_by_lossy(self):
        # Search a small parameter grid for a state where lossless emits a spike
        # while lossy exp_ps does not (S2 region in NEST terminology).
        with brainstate.environ.context(dt=1.0 * u.ms):
            params = dict(
                tau_m=100.0 * u.ms,
                tau_syn_ex=1.0 * u.ms,
                tau_syn_in=1.0 * u.ms,
                C_m=250.0 * u.pF,
                V_th=-49.0 * u.mV,
                V_reset=-70.0 * u.mV,
                E_L=-70.0 * u.mV,
                I_e=0.0 * u.pA,
                t_ref=2.0 * u.ms,
                V_initializer=braintools.init.Constant(-49.001 * u.mV),
            )
            found = False
            for w in [55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]:
                for offs in [0.9, 0.7, 0.5, 0.3, 0.1]:
                    lossy = iaf_psc_exp_ps(1, **params)
                    lossless = iaf_psc_exp_ps_lossless(1, **params)
                    lossy.init_state()
                    lossless.init_state()

                    spk_l = self._step(lossy, 0, spike_events=[(offs * u.ms, w * u.pA)])
                    spk_ll = self._step(lossless, 0, spike_events=[(offs * u.ms, w * u.pA)])

                    if (not self._is_spike(spk_l)) and self._is_spike(spk_ll):
                        found = True
                        break
                if found:
                    break

            self.assertTrue(found, 'Did not find S2-like case where lossless detects a missed spike.')


if __name__ == '__main__':
    unittest.main()
