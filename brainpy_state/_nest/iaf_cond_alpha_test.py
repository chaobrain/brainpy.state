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

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import iaf_cond_alpha


def _rkf45_ref_step(v, dg_ex, g_ex, dg_in, g_in, is_refractory, i_stim, dt, h0, p, atol=1e-3):
    min_h = 1e-8
    t = 0.0
    h = max(h0, min_h)
    y = np.asarray([v, dg_ex, g_ex, dg_in, g_in], dtype=np.float64)

    def f(y_):
        v_ = y_[0]
        dg_ex_ = y_[1]
        g_ex_ = y_[2]
        dg_in_ = y_[3]
        g_in_ = y_[4]

        v_eff = p['V_reset'] if is_refractory else min(v_, p['V_th'])
        i_syn_ex = g_ex_ * (v_eff - p['E_ex'])
        i_syn_in = g_in_ * (v_eff - p['E_in'])
        i_leak = p['g_L'] * (v_eff - p['E_L'])
        dv = 0.0 if is_refractory else (
            -i_leak - i_syn_ex - i_syn_in + i_stim + p['I_e']
        ) / p['C_m']

        ddg_ex = -dg_ex_ / p['tau_syn_ex']
        dg_ex_dt = dg_ex_ - g_ex_ / p['tau_syn_ex']
        ddg_in = -dg_in_ / p['tau_syn_in']
        dg_in_dt = dg_in_ - g_in_ / p['tau_syn_in']
        return np.asarray([dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt], dtype=np.float64)

    while t < dt:
        h = max(min_h, min(h, dt - t))

        k1 = f(y)
        k2 = f(y + h * (1.0 / 4.0) * k1)
        k3 = f(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
        k4 = f(y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0))
        k5 = f(y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0))
        k6 = f(y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0))

        y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
        y5 = y + h * (16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0)
        err = float(np.max(np.abs(y5 - y4)))

        if err <= atol or h <= min_h:
            y = y5
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    return y[0], y[1], y[2], y[3], y[4], h


def _reference_step(state, params, x_next, w_step, dt):
    v, dg_ex, g_ex, dg_in, g_in, h = _rkf45_ref_step(
        state['v'],
        state['dg_ex'],
        state['g_ex'],
        state['dg_in'],
        state['g_in'],
        state['r'] > 0,
        state['i_stim'],
        dt,
        state['h'],
        params,
    )

    if state['r'] > 0:
        r = state['r'] - 1
        v = params['V_reset']
        spike = False
    else:
        if v >= params['V_th']:
            spike = True
            r = int(math.ceil(params['t_ref'] / dt))
            v = params['V_reset']
        else:
            spike = False
            r = 0

    if w_step >= 0.0:
        dg_ex += math.e / params['tau_syn_ex'] * w_step
    else:
        dg_in += math.e / params['tau_syn_in'] * (-w_step)

    state['v'] = v
    state['dg_ex'] = dg_ex
    state['g_ex'] = g_ex
    state['dg_in'] = dg_in
    state['g_in'] = g_in
    state['r'] = r
    state['h'] = h
    state['i_stim'] = x_next

    return spike


class TestIAFCondAlpha(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(np.asarray(u.math.asarray(spk), dtype=np.float64)[0] > 0.0)

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                neuron.add_delta_input(f'delta_{k}_{i}', val * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_nest_cpp_default_parameters(self):
        neuron = iaf_cond_alpha(1)
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
        with self.assertRaises(ValueError):
            iaf_cond_alpha(1, C_m=0.0 * u.pF)
        with self.assertRaises(ValueError):
            iaf_cond_alpha(1, tau_syn_ex=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_cond_alpha(1, tau_syn_in=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_cond_alpha(1, t_ref=-0.1 * u.ms)
        with self.assertRaises(ValueError):
            iaf_cond_alpha(1, V_reset=-55. * u.mV, V_th=-55. * u.mV)

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_alpha(
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

    def test_spike_weights_split_into_excitatory_and_inhibitory_channels(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_alpha(
                1,
                V_th=1000. * u.mV,
                V_initializer=braintools.init.Constant(-70. * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, dg_values=[5.0, -3.0])

            expected_dg_ex = math.e / 0.2 * 5.0
            expected_dg_in = math.e / 2.0 * 3.0
            self.assertAlmostEqual(float(neuron.dg_ex.value[0]), expected_dg_ex, delta=1e-12)
            self.assertAlmostEqual(float(neuron.dg_in.value[0]), expected_dg_in, delta=1e-12)
            self.assertTrue(u.math.allclose(neuron.g_ex.value, 0.0 * u.nS))
            self.assertTrue(u.math.allclose(neuron.g_in.value, 0.0 * u.nS))

            self._step(neuron, 1)
            self.assertTrue(bool((neuron.g_ex.value > 0.0 * u.nS)[0]))
            self.assertTrue(bool((neuron.g_in.value > 0.0 * u.nS)[0]))

    def test_reference_trace_matches_nest_step_logic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_alpha(
                1,
                E_L=-70. * u.mV,
                C_m=250. * u.pF,
                t_ref=0.25 * u.ms,
                V_th=-58. * u.mV,
                V_reset=-60. * u.mV,
                E_ex=0. * u.mV,
                E_in=-85. * u.mV,
                g_L=16.6667 * u.nS,
                tau_syn_ex=0.2 * u.ms,
                tau_syn_in=2.0 * u.ms,
                I_e=900. * u.pA,
                V_initializer=braintools.init.Constant(-69.5 * u.mV),
            )
            neuron.init_state()

            x_seq = [0.0, 20.0, 0.0, -30.0, 0.0, 40.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0] + [0.0] * 36
            w_seq = [0.0, 5.0, -2.0, 0.0, 4.0, -3.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2.5] + [0.0] * 36

            params = {
                'E_L': -70.0,
                'C_m': 250.0,
                't_ref': 0.25,
                'V_th': -58.0,
                'V_reset': -60.0,
                'E_ex': 0.0,
                'E_in': -85.0,
                'g_L': 16.6667,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 2.0,
                'I_e': 900.0,
            }
            ref_state = {
                'v': -69.5,
                'dg_ex': 0.0,
                'g_ex': 0.0,
                'dg_in': 0.0,
                'g_in': 0.0,
                'r': 0,
                'h': 0.1,
                'i_stim': 0.0,
            }

            spikes_model = []
            spikes_ref = []
            for k, (x_i, w_i) in enumerate(zip(x_seq, w_seq)):
                spk = self._step(neuron, k, x=x_i * u.pA, dg_values=[w_i] if w_i != 0.0 else None)
                spikes_model.append(self._is_spike(spk))
                spikes_ref.append(_reference_step(ref_state, params, x_i, w_i, 0.1))

                self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), ref_state['v'], delta=2e-6)
                self.assertAlmostEqual(float(neuron.dg_ex.value[0]), ref_state['dg_ex'], delta=2e-6)
                self.assertAlmostEqual(float((neuron.g_ex.value / u.nS)[0]), ref_state['g_ex'], delta=2e-6)
                self.assertAlmostEqual(float(neuron.dg_in.value[0]), ref_state['dg_in'], delta=2e-6)
                self.assertAlmostEqual(float((neuron.g_in.value / u.nS)[0]), ref_state['g_in'], delta=2e-6)
                self.assertEqual(int(neuron.refractory_step_count.value[0]), ref_state['r'])
                self.assertAlmostEqual(float((neuron.integration_step.value / u.ms)[0]), ref_state['h'], delta=2e-6)

            self.assertEqual(spikes_model, spikes_ref)
            self.assertTrue(any(spikes_model))


if __name__ == '__main__':
    unittest.main()
