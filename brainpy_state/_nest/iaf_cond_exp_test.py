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
from brainpy.state import iaf_cond_exp

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _rkf45_ref_step(v, g_ex, g_in, is_refractory, i_stim, dt, h0, p, atol=1e-3):
    min_h = 1e-8
    t = 0.0
    h = max(h0, min_h)

    def f(v_, ge_, gi_):
        if is_refractory:
            return 0.0, -ge_ / p['tau_syn_ex'], -gi_ / p['tau_syn_in']
        v_eff = min(v_, p['V_th'])
        i_syn_ex = ge_ * (v_eff - p['E_ex'])
        i_syn_in = gi_ * (v_eff - p['E_in'])
        i_l = p['g_L'] * (v_eff - p['E_L'])
        dv = (-i_l + i_stim + p['I_e'] - i_syn_ex - i_syn_in) / p['C_m']
        return dv, -ge_ / p['tau_syn_ex'], -gi_ / p['tau_syn_in']

    while t < dt:
        h = max(min_h, min(h, dt - t))

        k1 = f(v, g_ex, g_in)
        y2 = (v + h * k1[0] / 4.0, g_ex + h * k1[1] / 4.0, g_in + h * k1[2] / 4.0)
        k2 = f(*y2)
        y3 = (
            v + h * (3.0 * k1[0] / 32.0 + 9.0 * k2[0] / 32.0),
            g_ex + h * (3.0 * k1[1] / 32.0 + 9.0 * k2[1] / 32.0),
            g_in + h * (3.0 * k1[2] / 32.0 + 9.0 * k2[2] / 32.0),
        )
        k3 = f(*y3)
        y4 = (
            v + h * (1932.0 * k1[0] / 2197.0 - 7200.0 * k2[0] / 2197.0 + 7296.0 * k3[0] / 2197.0),
            g_ex + h * (1932.0 * k1[1] / 2197.0 - 7200.0 * k2[1] / 2197.0 + 7296.0 * k3[1] / 2197.0),
            g_in + h * (1932.0 * k1[2] / 2197.0 - 7200.0 * k2[2] / 2197.0 + 7296.0 * k3[2] / 2197.0),
        )
        k4 = f(*y4)
        y5 = (
            v + h * (439.0 * k1[0] / 216.0 - 8.0 * k2[0] + 3680.0 * k3[0] / 513.0 - 845.0 * k4[0] / 4104.0),
            g_ex + h * (439.0 * k1[1] / 216.0 - 8.0 * k2[1] + 3680.0 * k3[1] / 513.0 - 845.0 * k4[1] / 4104.0),
            g_in + h * (439.0 * k1[2] / 216.0 - 8.0 * k2[2] + 3680.0 * k3[2] / 513.0 - 845.0 * k4[2] / 4104.0),
        )
        k5 = f(*y5)
        y6 = (
            v + h * (-8.0 * k1[0] / 27.0 + 2.0 * k2[0] - 3544.0 * k3[0] / 2565.0 + 1859.0 * k4[0] / 4104.0 - 11.0 * k5[
                0] / 40.0),
            g_ex + h * (
                    -8.0 * k1[1] / 27.0 + 2.0 * k2[1] - 3544.0 * k3[1] / 2565.0 + 1859.0 * k4[1] / 4104.0 - 11.0 * k5[
                    1] / 40.0),
            g_in + h * (
                    -8.0 * k1[2] / 27.0 + 2.0 * k2[2] - 3544.0 * k3[2] / 2565.0 + 1859.0 * k4[2] / 4104.0 - 11.0 * k5[
                    2] / 40.0),
        )
        k6 = f(*y6)

        y4_sol = (
            v + h * (25.0 * k1[0] / 216.0 + 1408.0 * k3[0] / 2565.0 + 2197.0 * k4[0] / 4104.0 - k5[0] / 5.0),
            g_ex + h * (25.0 * k1[1] / 216.0 + 1408.0 * k3[1] / 2565.0 + 2197.0 * k4[1] / 4104.0 - k5[1] / 5.0),
            g_in + h * (25.0 * k1[2] / 216.0 + 1408.0 * k3[2] / 2565.0 + 2197.0 * k4[2] / 4104.0 - k5[2] / 5.0),
        )
        y5_sol = (
            v + h * (16.0 * k1[0] / 135.0 + 6656.0 * k3[0] / 12825.0 + 28561.0 * k4[0] / 56430.0 - 9.0 * k5[
                0] / 50.0 + 2.0 * k6[0] / 55.0),
            g_ex + h * (16.0 * k1[1] / 135.0 + 6656.0 * k3[1] / 12825.0 + 28561.0 * k4[1] / 56430.0 - 9.0 * k5[
                1] / 50.0 + 2.0 * k6[1] / 55.0),
            g_in + h * (16.0 * k1[2] / 135.0 + 6656.0 * k3[2] / 12825.0 + 28561.0 * k4[2] / 56430.0 - 9.0 * k5[
                2] / 50.0 + 2.0 * k6[2] / 55.0),
        )
        err = max(abs(y5_sol[0] - y4_sol[0]), abs(y5_sol[1] - y4_sol[1]), abs(y5_sol[2] - y4_sol[2]))
        if err <= atol or h <= min_h:
            v, g_ex, g_in = y5_sol
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    return v, g_ex, g_in, h


class TestIAFCondExp(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        dftype = brainstate.environ.dftype()
        return bool(np.asarray(u.math.asarray(spk), dtype=dftype)[0] > 0.0)

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                neuron.add_delta_input(f'dg_{k}_{i}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_nest_cpp_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = iaf_cond_exp(1)
            self.assertEqual(neuron.E_L, -70. * u.mV)
            self.assertEqual(neuron.C_m, 250. * u.pF)
            self.assertEqual(neuron.t_ref, 2. * u.ms)
            self.assertEqual(neuron.V_th, -55. * u.mV)
            self.assertEqual(neuron.V_reset, -60. * u.mV)
            self.assertEqual(neuron.E_ex, 0. * u.mV)
            self.assertEqual(neuron.E_in, -85. * u.mV)
            self.assertEqual(neuron.g_L, 16.6667 * u.nS)
            self.assertEqual(neuron.tau_syn_ex, 0.2 * u.ms)
            self.assertEqual(neuron.tau_syn_in, 2.0 * u.ms)
            self.assertEqual(neuron.I_e, 0. * u.pA)

    def test_parameter_validation(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                iaf_cond_exp(1, C_m=0.0 * u.pF)
            with self.assertRaises(ValueError):
                iaf_cond_exp(1, tau_syn_ex=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_cond_exp(1, tau_syn_in=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_cond_exp(1, t_ref=-0.1 * u.ms)
            with self.assertRaises(ValueError):
                iaf_cond_exp(1, V_reset=-55. * u.mV, V_th=-55. * u.mV)

    def test_signed_spike_weights_split_into_ex_and_in(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_exp(
                1,
                V_th=100. * u.mV,
                V_initializer=braintools.init.Constant(-70. * u.mV),
            )
            neuron.init_state()
            self._step(neuron, 0, dg_values=[5.0, -3.0])
            self.assertTrue(u.math.allclose(neuron.g_ex.value, 5.0 * u.nS))
            self.assertTrue(u.math.allclose(neuron.g_in.value, 3.0 * u.nS))

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_exp(
                1,
                E_L=0. * u.mV,
                g_L=0. * u.nS,
                I_e=0. * u.pA,
                V_th=1e9 * u.mV,
                V_reset=0. * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV),
            )
            neuron.init_state()
            self._step(neuron, 0, x=100. * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0. * u.mV))
            self._step(neuron, 1, x=0. * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.04 * u.mV))

    def test_excitatory_and_inhibitory_directions_match_nest_tests(self):
        with brainstate.environ.context(dt=self.dt):
            base = iaf_cond_exp(1, V_th=100. * u.mV, V_initializer=braintools.init.Constant(-70. * u.mV))
            exc = iaf_cond_exp(1, V_th=100. * u.mV, V_initializer=braintools.init.Constant(-70. * u.mV))
            inh = iaf_cond_exp(1, V_th=100. * u.mV, V_initializer=braintools.init.Constant(-70. * u.mV))
            base.init_state()
            exc.init_state()
            inh.init_state()

            self._step(base, 0)
            self._step(exc, 0, dg_values=[5.0])
            self._step(inh, 0, dg_values=[-5.0])

            self._step(base, 1)
            self._step(exc, 1)
            self._step(inh, 1)

            self.assertTrue(bool((exc.V.value > base.V.value)[0]))
            self.assertTrue(bool((inh.V.value < base.V.value)[0]))

    def test_refractory_countdown_and_spike_timing(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_exp(
                1,
                E_L=0. * u.mV,
                g_L=0. * u.nS,
                C_m=250. * u.pF,
                I_e=10000. * u.pA,
                V_th=1. * u.mV,
                V_reset=0. * u.mV,
                t_ref=0.25 * u.ms,
                V_initializer=braintools.init.Constant(0. * u.mV),
            )
            neuron.init_state()

            spikes = []
            for k in range(5):
                spikes.append(self._is_spike(self._step(neuron, k)))

            self.assertTrue(spikes[0])
            self.assertTrue((not spikes[1]) and (not spikes[2]) and (not spikes[3]))
            self.assertTrue(spikes[4])
            self.assertTrue(u.math.allclose(neuron.last_spike_time.value, 5.0 * self.dt))

    def test_reference_trace_matches_nest_step_logic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_exp(
                1,
                E_L=-70. * u.mV,
                C_m=250. * u.pF,
                t_ref=0.25 * u.ms,
                V_th=-55. * u.mV,
                V_reset=-60. * u.mV,
                E_ex=0. * u.mV,
                E_in=-85. * u.mV,
                g_L=16.6667 * u.nS,
                tau_syn_ex=0.2 * u.ms,
                tau_syn_in=2.0 * u.ms,
                I_e=20. * u.pA,
                V_initializer=braintools.init.Constant(-69.5 * u.mV),
            )
            neuron.init_state()

            x_seq = [0.0, 30.0, 0.0, 0.0, -20.0, 0.0, 0.0]
            dg_seq = [[4.0], [], [-2.0], [3.0, -1.0], [], [], [1.5]]

            v_model, ge_model, gi_model, s_model = [], [], [], []
            for k in range(len(x_seq)):
                spk = self._step(neuron, k, x=x_seq[k] * u.pA, dg_values=dg_seq[k])
                v_model.append(float((neuron.V.value / u.mV)[0]))
                ge_model.append(float((neuron.g_ex.value / u.nS)[0]))
                gi_model.append(float((neuron.g_in.value / u.nS)[0]))
                s_model.append(self._is_spike(spk))

            dt = 0.1
            p = {
                'E_L': -70.0,
                'C_m': 250.0,
                'V_th': -55.0,
                'V_reset': -60.0,
                'E_ex': 0.0,
                'E_in': -85.0,
                'g_L': 16.6667,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 2.0,
                'I_e': 20.0,
            }
            v, ge, gi = -69.5, 0.0, 0.0
            r = 0
            h = dt
            i_stim = 0.0
            refr_steps = math.ceil(0.25 / dt)
            v_ref, ge_ref, gi_ref, s_ref = [], [], [], []

            for x, dgs in zip(x_seq, dg_seq):
                v, ge, gi, h = _rkf45_ref_step(v, ge, gi, r > 0, i_stim, dt, h, p)
                for dg in dgs:
                    if dg >= 0.0:
                        ge += dg
                    else:
                        gi += -dg

                if r > 0:
                    r -= 1
                    v = p['V_reset']
                    spike = False
                else:
                    if v >= p['V_th']:
                        spike = True
                        r = refr_steps
                        v = p['V_reset']
                    else:
                        spike = False
                i_stim = x
                v_ref.append(v)
                ge_ref.append(ge)
                gi_ref.append(gi)
                s_ref.append(spike)

            self.assertEqual(s_model, s_ref)
            self.assertTrue(all(abs(a - b) < 5e-6 for a, b in zip(v_model, v_ref)))
            self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(ge_model, ge_ref)))
            self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(gi_model, gi_ref)))


if __name__ == '__main__':
    unittest.main()
