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
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import iaf_chxk_2008


def _rkf45_ref_step(v, dg_ex, g_ex, dg_in, g_in, dg_ahp, g_ahp, i_stim, dt, h0, p, atol=1e-3):
    min_h = 1e-8
    t = 0.0
    h = max(h0, min_h)
    y = np.asarray([v, dg_ex, g_ex, dg_in, g_in, dg_ahp, g_ahp], dtype=np.float64)

    def f(y_):
        v_ = y_[0]
        dg_ex_ = y_[1]
        g_ex_ = y_[2]
        dg_in_ = y_[3]
        g_in_ = y_[4]
        dg_ahp_ = y_[5]
        g_ahp_ = y_[6]

        i_syn_ex = g_ex_ * (v_ - p['E_ex'])
        i_syn_in = g_in_ * (v_ - p['E_in'])
        i_ahp = g_ahp_ * (v_ - p['E_ahp'])
        i_leak = p['g_L'] * (v_ - p['E_L'])

        dv = (-i_leak - i_syn_ex - i_syn_in - i_ahp + i_stim + p['I_e']) / p['C_m']
        ddg_ex = -dg_ex_ / p['tau_syn_ex']
        dg_ex_dt = dg_ex_ - g_ex_ / p['tau_syn_ex']
        ddg_in = -dg_in_ / p['tau_syn_in']
        dg_in_dt = dg_in_ - g_in_ / p['tau_syn_in']
        ddg_ahp = -dg_ahp_ / p['tau_ahp']
        dg_ahp_dt = dg_ahp_ - g_ahp_ / p['tau_ahp']
        return np.asarray([dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt, ddg_ahp, dg_ahp_dt], dtype=np.float64)

    while t < dt:
        h = max(min_h, min(h, dt - t))

        k1 = f(y)
        k2 = f(y + h * (1.0 / 4.0) * k1)
        k3 = f(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
        k4 = f(y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0))
        k5 = f(y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0))
        k6 = f(y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0))

        y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
        y5 = y + h * (
                16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0)
        err = float(np.max(np.abs(y5 - y4)))

        if err <= atol or h <= min_h:
            y = y5
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    return y[0], y[1], y[2], y[3], y[4], y[5], y[6], h


def _reference_step(state, p, x_next, signed_weights, dt, t_step):
    vm_prev = state['v']
    v, dg_ex, g_ex, dg_in, g_in, dg_ahp, g_ahp, h = _rkf45_ref_step(
        state['v'],
        state['dg_ex'],
        state['g_ex'],
        state['dg_in'],
        state['g_in'],
        state['dg_ahp'],
        state['g_ahp'],
        state['i_stim'],
        dt,
        state['h'],
        p,
    )

    crossed = (vm_prev < p['V_th']) and (v >= p['V_th'])
    if crossed:
        denom = v - vm_prev
        dt_from_spike_to_end = 0.0 if abs(denom) < np.finfo(np.float64).tiny else dt * (v - p['V_th']) / denom
        dt_from_spike_to_end = min(dt, max(0.0, float(dt_from_spike_to_end)))

        delta_dg = p['PSConInit_AHP'] * math.exp(-dt_from_spike_to_end / p['tau_ahp'])
        delta_g = delta_dg * dt_from_spike_to_end
        if p['ahp_bug']:
            g_ahp = delta_g
            dg_ahp = delta_dg
        else:
            g_ahp = g_ahp + delta_g
            dg_ahp = dg_ahp + delta_dg
        spike_time = t_step + dt - dt_from_spike_to_end
        spike_offset = dt_from_spike_to_end
        spike = True
    else:
        spike_time = state['last_spike_time']
        spike_offset = state['last_spike_offset']
        spike = False

    for w in signed_weights:
        if w >= 0.0:
            dg_ex = dg_ex + p['PSConInit_E'] * w
        else:
            dg_in = dg_in + p['PSConInit_I'] * (-w)

    i_syn_ex = g_ex * (v - p['E_ex'])
    i_syn_in = g_in * (v - p['E_in'])
    i_ahp = g_ahp * (v - p['E_ahp'])

    state['v'] = v
    state['dg_ex'] = dg_ex
    state['g_ex'] = g_ex
    state['dg_in'] = dg_in
    state['g_in'] = g_in
    state['dg_ahp'] = dg_ahp
    state['g_ahp'] = g_ahp
    state['i_syn_ex'] = i_syn_ex
    state['i_syn_in'] = i_syn_in
    state['i_ahp'] = i_ahp
    state['h'] = h
    state['i_stim'] = x_next
    state['last_spike_time'] = spike_time
    state['last_spike_offset'] = spike_offset
    return spike


class TestIAFChxk2008(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _scalar(x, unit=None):
        if unit is None:
            return float(np.asarray(u.math.asarray(x), dtype=np.float64).reshape(-1)[0])
        return float(np.asarray(u.math.asarray(x / unit), dtype=np.float64).reshape(-1)[0])

    @staticmethod
    def _is_spike(spk):
        return bool(np.asarray(u.math.asarray(spk), dtype=np.float64).reshape(-1)[0] > 0.0)

    def _step(self, neuron, k, x=0.0 * u.pA, weights=None):
        if weights is not None:
            for i, w in enumerate(weights):
                neuron.add_delta_input(f'w_{k}_{i}', float(w) * u.nS)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_nest_cpp_default_parameters_and_metadata(self):
        neuron = iaf_chxk_2008(1)
        self.assertEqual(neuron.V_th, -45.0 * u.mV)
        self.assertEqual(neuron.g_L, 100.0 * u.nS)
        self.assertEqual(neuron.C_m, 1000.0 * u.pF)
        self.assertEqual(neuron.E_ex, 20.0 * u.mV)
        self.assertEqual(neuron.E_in, -90.0 * u.mV)
        self.assertEqual(neuron.E_L, -60.0 * u.mV)
        self.assertEqual(neuron.tau_syn_ex, 1.0 * u.ms)
        self.assertEqual(neuron.tau_syn_in, 1.0 * u.ms)
        self.assertEqual(neuron.I_e, 0.0 * u.pA)
        self.assertEqual(neuron.tau_ahp, 0.5 * u.ms)
        self.assertEqual(neuron.E_ahp, -95.0 * u.mV)
        self.assertEqual(neuron.g_ahp, 443.8 * u.nS)
        self.assertFalse(bool(np.asarray(u.math.asarray(neuron.ahp_bug)).reshape(-1)[0]))
        self.assertEqual(
            neuron.recordables,
            ['V_m', 'g_ex', 'g_in', 'g_ahp', 'I_syn_ex', 'I_syn_in', 'I_ahp'],
        )

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            iaf_chxk_2008(1, C_m=0.0 * u.pF)
        with self.assertRaises(ValueError):
            iaf_chxk_2008(1, tau_syn_ex=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_chxk_2008(1, tau_syn_in=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_chxk_2008(1, tau_ahp=0.0 * u.ms)

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_chxk_2008(
                1,
                E_L=0.0 * u.mV,
                g_L=0.0 * u.nS,
                I_e=0.0 * u.pA,
                V_th=1e9 * u.mV,
                E_ex=0.0 * u.mV,
                E_in=0.0 * u.mV,
                E_ahp=0.0 * u.mV,
                g_ahp=0.0 * u.nS,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, x=100.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.0 * u.mV))

            self._step(neuron, 1, x=0.0 * u.pA)
            self.assertAlmostEqual(self._scalar(neuron.V.value, u.mV), 0.01, delta=1e-12)

    def test_spike_weights_split_and_use_alpha_normalization(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_chxk_2008(
                1,
                V_th=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, weights=[5.0, -3.0])
            self.assertAlmostEqual(float(neuron.dg_ex.value[0]), math.e * 5.0, delta=1e-12)
            self.assertAlmostEqual(float(neuron.dg_in.value[0]), math.e * 3.0, delta=1e-12)
            self.assertTrue(u.math.allclose(neuron.g_ex.value, 0.0 * u.nS))
            self.assertTrue(u.math.allclose(neuron.g_in.value, 0.0 * u.nS))

            self._step(neuron, 1)
            self.assertGreater(self._scalar(neuron.g_ex.value, u.nS), 0.0)
            self.assertGreater(self._scalar(neuron.g_in.value, u.nS), 0.0)

    def test_spike_requires_crossing_from_below(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_chxk_2008(
                1,
                V_th=-45.0 * u.mV,
                g_L=0.0 * u.nS,
                I_e=0.0 * u.pA,
                g_ahp=0.0 * u.nS,
                V_initializer=braintools.init.Constant(-40.0 * u.mV),
            )
            neuron.init_state()

            spikes = []
            for k in range(8):
                spk = self._step(neuron, k)
                spikes.append(self._is_spike(spk))

            self.assertEqual(spikes, [False] * 8)
            self.assertTrue(u.math.allclose(neuron.last_spike_time.value, -1e7 * u.ms))

    def test_reference_trace_matches_nest_step_logic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_chxk_2008(
                1,
                V_th=-45.0 * u.mV,
                g_L=100.0 * u.nS,
                C_m=1000.0 * u.pF,
                E_ex=20.0 * u.mV,
                E_in=-90.0 * u.mV,
                E_L=-60.0 * u.mV,
                tau_syn_ex=1.0 * u.ms,
                tau_syn_in=1.0 * u.ms,
                I_e=3000.0 * u.pA,
                tau_ahp=0.5 * u.ms,
                E_ahp=-95.0 * u.mV,
                g_ahp=443.8 * u.nS,
                ahp_bug=False,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
            )
            neuron.init_state()

            x_seq = [0.0, 100.0, 0.0, -80.0, 0.0, 140.0, 0.0, -60.0, 0.0, 0.0] + [0.0] * 70
            w_seq = [[0.0], [5.0], [-3.0], [2.0, -1.0], [], [4.0], [], [], [-2.0], [3.0]] + [[]] * 70

            p = {
                'V_th': -45.0,
                'g_L': 100.0,
                'C_m': 1000.0,
                'E_ex': 20.0,
                'E_in': -90.0,
                'E_L': -60.0,
                'tau_syn_ex': 1.0,
                'tau_syn_in': 1.0,
                'I_e': 3000.0,
                'tau_ahp': 0.5,
                'E_ahp': -95.0,
                'PSConInit_E': math.e / 1.0,
                'PSConInit_I': math.e / 1.0,
                'PSConInit_AHP': 443.8 * math.e / 0.5,
                'ahp_bug': False,
            }
            ref = {
                'v': -60.0,
                'dg_ex': 0.0,
                'g_ex': 0.0,
                'dg_in': 0.0,
                'g_in': 0.0,
                'dg_ahp': 0.0,
                'g_ahp': 0.0,
                'i_syn_ex': 0.0,
                'i_syn_in': 0.0,
                'i_ahp': 0.0,
                'i_stim': 0.0,
                'h': 0.1,
                'last_spike_time': -1e7,
                'last_spike_offset': 0.0,
            }

            spikes_model = []
            spikes_ref = []
            for k, (x_i, ws) in enumerate(zip(x_seq, w_seq)):
                spk = self._step(neuron, k, x=x_i * u.pA, weights=ws if ws else None)
                spikes_model.append(self._is_spike(spk))
                spikes_ref.append(_reference_step(ref, p, x_i, ws, 0.1, 0.1 * k))

                self.assertAlmostEqual(self._scalar(neuron.V.value, u.mV), ref['v'], delta=3e-6)
                self.assertAlmostEqual(float(neuron.dg_ex.value[0]), ref['dg_ex'], delta=3e-6)
                self.assertAlmostEqual(self._scalar(neuron.g_ex.value, u.nS), ref['g_ex'], delta=3e-6)
                self.assertAlmostEqual(float(neuron.dg_in.value[0]), ref['dg_in'], delta=3e-6)
                self.assertAlmostEqual(self._scalar(neuron.g_in.value, u.nS), ref['g_in'], delta=3e-6)
                self.assertAlmostEqual(float(neuron.dg_ahp.value[0]), ref['dg_ahp'], delta=4e-6)
                self.assertAlmostEqual(self._scalar(neuron.g_ahp_state.value, u.nS), ref['g_ahp'], delta=4e-6)
                self.assertAlmostEqual(self._scalar(neuron.I_syn_ex.value, u.pA), ref['i_syn_ex'], delta=4e-5)
                self.assertAlmostEqual(self._scalar(neuron.I_syn_in.value, u.pA), ref['i_syn_in'], delta=4e-5)
                self.assertAlmostEqual(self._scalar(neuron.I_ahp.value, u.pA), ref['i_ahp'], delta=4e-5)
                self.assertAlmostEqual(self._scalar(neuron.I_stim.value, u.pA), ref['i_stim'], delta=1e-12)
                self.assertAlmostEqual(self._scalar(neuron.integration_step.value, u.ms), ref['h'], delta=4e-6)
                self.assertAlmostEqual(self._scalar(neuron.last_spike_time.value, u.ms), ref['last_spike_time'],
                                       delta=3e-6)
                self.assertAlmostEqual(self._scalar(neuron.last_spike_offset.value, u.ms), ref['last_spike_offset'],
                                       delta=3e-6)

            self.assertEqual(spikes_model, spikes_ref)
            self.assertTrue(any(spikes_model))

    def test_reference_trace_matches_nest_step_logic_with_ahp_bug(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_chxk_2008(
                1,
                V_th=-45.0 * u.mV,
                g_L=100.0 * u.nS,
                C_m=1000.0 * u.pF,
                E_ex=20.0 * u.mV,
                E_in=-90.0 * u.mV,
                E_L=-60.0 * u.mV,
                tau_syn_ex=1.0 * u.ms,
                tau_syn_in=1.0 * u.ms,
                I_e=3000.0 * u.pA,
                tau_ahp=0.5 * u.ms,
                E_ahp=-95.0 * u.mV,
                g_ahp=443.8 * u.nS,
                ahp_bug=True,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
            )
            neuron.init_state()

            x_seq = [0.0, 0.0, 80.0, 0.0, -50.0, 0.0, 0.0] + [0.0] * 53
            w_seq = [[3.0], [], [-2.0], [4.0], [], [-1.0], []] + [[]] * 53

            p = {
                'V_th': -45.0,
                'g_L': 100.0,
                'C_m': 1000.0,
                'E_ex': 20.0,
                'E_in': -90.0,
                'E_L': -60.0,
                'tau_syn_ex': 1.0,
                'tau_syn_in': 1.0,
                'I_e': 3000.0,
                'tau_ahp': 0.5,
                'E_ahp': -95.0,
                'PSConInit_E': math.e / 1.0,
                'PSConInit_I': math.e / 1.0,
                'PSConInit_AHP': 443.8 * math.e / 0.5,
                'ahp_bug': True,
            }
            ref = {
                'v': -60.0,
                'dg_ex': 0.0,
                'g_ex': 0.0,
                'dg_in': 0.0,
                'g_in': 0.0,
                'dg_ahp': 0.0,
                'g_ahp': 0.0,
                'i_syn_ex': 0.0,
                'i_syn_in': 0.0,
                'i_ahp': 0.0,
                'i_stim': 0.0,
                'h': 0.1,
                'last_spike_time': -1e7,
                'last_spike_offset': 0.0,
            }

            for k, (x_i, ws) in enumerate(zip(x_seq, w_seq)):
                self._step(neuron, k, x=x_i * u.pA, weights=ws if ws else None)
                _reference_step(ref, p, x_i, ws, 0.1, 0.1 * k)

                self.assertAlmostEqual(self._scalar(neuron.V.value, u.mV), ref['v'], delta=3e-6)
                self.assertAlmostEqual(float(neuron.dg_ex.value[0]), ref['dg_ex'], delta=3e-6)
                self.assertAlmostEqual(self._scalar(neuron.g_ex.value, u.nS), ref['g_ex'], delta=3e-6)
                self.assertAlmostEqual(float(neuron.dg_in.value[0]), ref['dg_in'], delta=3e-6)
                self.assertAlmostEqual(self._scalar(neuron.g_in.value, u.nS), ref['g_in'], delta=3e-6)
                self.assertAlmostEqual(float(neuron.dg_ahp.value[0]), ref['dg_ahp'], delta=4e-6)
                self.assertAlmostEqual(self._scalar(neuron.g_ahp_state.value, u.nS), ref['g_ahp'], delta=4e-6)
                self.assertAlmostEqual(self._scalar(neuron.last_spike_time.value, u.ms), ref['last_spike_time'],
                                       delta=3e-6)
                self.assertAlmostEqual(self._scalar(neuron.last_spike_offset.value, u.ms), ref['last_spike_offset'],
                                       delta=3e-6)


if __name__ == '__main__':
    unittest.main()
