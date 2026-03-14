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
import jax
import numpy as np
from brainpy.state import iaf_psc_exp_htum

from brainpy_state._nest.iaf_psc_exp import iaf_psc_exp

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class TestIAFPscExpHtum(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def _step(self, neuron, step_idx, x=0. * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_defaults_and_validation(self):
        neuron = iaf_psc_exp_htum(1)
        self.assertEqual(neuron.t_ref_abs, 2. * u.ms)
        self.assertEqual(neuron.t_ref_tot, 2. * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_exp_htum(1, t_ref_abs=3. * u.ms, t_ref_tot=2. * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_exp_htum(1, t_ref_abs=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_exp_htum(1, t_ref_tot=0.0 * u.ms)

    def test_absolute_and_total_refractory_semantics(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_exp_htum(
                1,
                E_L=0. * u.mV,
                V_reset=0. * u.mV,
                V_th=1. * u.mV,
                tau_m=10. * u.ms,
                C_m=250. * u.pF,
                tau_syn_ex=2. * u.ms,
                tau_syn_in=2. * u.ms,
                I_e=10000. * u.pA,
                t_ref_abs=0.3 * u.ms,
                t_ref_tot=0.5 * u.ms,
                V_initializer=braintools.init.Constant(0. * u.mV),
            )
            neuron.init_state()

            spikes = []
            v_vals = []
            for k in range(12):
                spk = self._step(neuron, k)
                spikes.append(self._is_spike(spk))
                v_vals.append(float((neuron.V.value / u.mV)[0]))

            self.assertTrue(spikes[0])
            # total refractory blocks spikes for ceil(0.5/0.1)=5 steps after spike.
            for k in range(1, 6):
                self.assertFalse(spikes[k])
            # absolute refractory clamps V for ceil(0.3/0.1)=3 steps after spike.
            for k in range(1, 4):
                self.assertAlmostEqual(v_vals[k], 0.0, delta=1e-12)

    def test_step_equations_match_reference(self):
        with brainstate.environ.context(dt=self.dt):
            p = dict(
                E_L=-70.0,
                C_m=250.0,
                tau_m=10.0,
                tau_ex=2.0,
                tau_in=3.0,
                I_e=50.0,
                V_th=-55.0,
                V_reset=-70.0,
                t_ref_abs=0.2,
                t_ref_tot=0.4,
            )
            neuron = iaf_psc_exp_htum(
                1,
                E_L=p['E_L'] * u.mV,
                C_m=p['C_m'] * u.pF,
                tau_m=p['tau_m'] * u.ms,
                tau_syn_ex=p['tau_ex'] * u.ms,
                tau_syn_in=p['tau_in'] * u.ms,
                I_e=p['I_e'] * u.pA,
                V_th=p['V_th'] * u.mV,
                V_reset=p['V_reset'] * u.mV,
                t_ref_abs=p['t_ref_abs'] * u.ms,
                t_ref_tot=p['t_ref_tot'] * u.ms,
                V_initializer=braintools.init.Constant(-67.0 * u.mV),
            )
            neuron.init_state()

            x_seq = [10.0, 20.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0]
            w_seq = [0.0, 30.0, -20.0, 0.0, 0.0, 20.0, -10.0, 0.0]

            v = -67.0 - p['E_L']
            i0 = 0.0
            iex = 0.0
            iin = 0.0
            r_abs = 0
            r_tot = 0
            th = p['V_th'] - p['E_L']
            reset = p['V_reset'] - p['E_L']
            p11ex = math.exp(-0.1 / p['tau_ex'])
            p11in = math.exp(-0.1 / p['tau_in'])
            p22 = math.exp(-0.1 / p['tau_m'])
            p20 = p['tau_m'] / p['C_m'] * (1.0 - p22)
            p21ex = iaf_psc_exp._propagator_exp(np.asarray(p['tau_ex']), np.asarray(p['tau_m']), np.asarray(p['C_m']),
                                                0.1)
            p21in = iaf_psc_exp._propagator_exp(np.asarray(p['tau_in']), np.asarray(p['tau_m']), np.asarray(p['C_m']),
                                                0.1)
            ref_abs = int(math.ceil(p['t_ref_abs'] / 0.1))
            ref_tot = int(math.ceil(p['t_ref_tot'] / 0.1))

            for k, (x, w) in enumerate(zip(x_seq, w_seq)):
                spk = self._step(neuron, k, x=x * u.pA, delta=w * u.pA)

                if r_abs == 0:
                    v = v * p22 + iex * p21ex + iin * p21in + (p['I_e'] + i0) * p20
                else:
                    r_abs -= 1

                iex = iex * p11ex + max(w, 0.0)
                iin = iin * p11in + min(w, 0.0)

                spike_ref = False
                if r_tot == 0:
                    if v >= th:
                        spike_ref = True
                        r_abs = ref_abs
                        r_tot = ref_tot
                        v = reset
                else:
                    r_tot -= 1

                i0 = x
                self.assertEqual(self._is_spike(spk), spike_ref)
                self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), v + p['E_L'], delta=1e-11)
                self.assertEqual(int(neuron.refractory_abs_step_count.value[0]), r_abs)
                self.assertEqual(int(neuron.refractory_tot_step_count.value[0]), r_tot)


if __name__ == '__main__':
    unittest.main()
