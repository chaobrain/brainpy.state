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
import jax.numpy as jnp
import numpy as np
from brainpy.state import iaf_cond_exp, iaf_cond_exp_sfa_rr

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _rkf45_ref_step(v, g_ex, g_in, g_sfa, g_rr, is_refractory, i_stim, dt, h0, p, atol=1e-3):
    min_h = 1e-8
    t = 0.0
    h = max(h0, min_h)
    dftype = brainstate.environ.dftype()
    y = np.asarray([v, g_ex, g_in, g_sfa, g_rr], dtype=dftype)

    def f(y_):
        v_ = y_[0]
        g_ex_ = y_[1]
        g_in_ = y_[2]
        g_sfa_ = y_[3]
        g_rr_ = y_[4]

        v_eff = p['V_reset'] if is_refractory else min(v_, p['V_th'])

        i_syn_ex = g_ex_ * (v_eff - p['E_ex'])
        i_syn_in = g_in_ * (v_eff - p['E_in'])
        i_l = p['g_L'] * (v_eff - p['E_L'])
        i_sfa = g_sfa_ * (v_eff - p['E_sfa'])
        i_rr = g_rr_ * (v_eff - p['E_rr'])

        dv = 0.0 if is_refractory else (
                                           -i_l + i_stim + p['I_e'] - i_syn_ex - i_syn_in - i_sfa - i_rr
                                       ) / p['C_m']

        dg_ex = -g_ex_ / p['tau_syn_ex']
        dg_in = -g_in_ / p['tau_syn_in']
        dg_sfa = -g_sfa_ / p['tau_sfa']
        dg_rr = -g_rr_ / p['tau_rr']
        dftype = brainstate.environ.dftype()
        return np.asarray([dv, dg_ex, dg_in, dg_sfa, dg_rr], dtype=dftype)

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

    return y[0], y[1], y[2], y[3], y[4], h


def _reference_step(state, params, x_next, dg_values, dt):
    v, g_ex, g_in, g_sfa, g_rr, h = _rkf45_ref_step(
        state['v'],
        state['g_ex'],
        state['g_in'],
        state['g_sfa'],
        state['g_rr'],
        state['r'] > 0,
        state['i_stim'],
        dt,
        state['h'],
        params,
    )

    for dg in dg_values:
        if dg >= 0.0:
            g_ex += dg
        else:
            g_in += -dg

    if state['r'] > 0:
        r = state['r'] - 1
        v = params['V_reset']
        spike = False
    else:
        if v >= params['V_th']:
            spike = True
            r = int(math.ceil(params['t_ref'] / dt))
            v = params['V_reset']
            g_sfa += params['q_sfa']
            g_rr += params['q_rr']
        else:
            spike = False
            r = 0

    state['v'] = v
    state['g_ex'] = g_ex
    state['g_in'] = g_in
    state['g_sfa'] = g_sfa
    state['g_rr'] = g_rr
    state['r'] = r
    state['h'] = h
    state['i_stim'] = x_next

    return spike


class TestIAFCondExpSfaRr(unittest.TestCase):
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
                if val >= 0:
                    neuron.add_delta_input(f'dg_{k}_{i}', val * u.nS, label='w_ex')
                else:
                    neuron.add_delta_input(f'dg_{k}_{i}', (-val) * u.nS, label='w_in')
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_nest_cpp_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = iaf_cond_exp_sfa_rr(1)
            self.assertEqual(neuron.V_th, -57.0 * u.mV)
            self.assertEqual(neuron.V_reset, -70.0 * u.mV)
            self.assertEqual(neuron.t_ref, 0.5 * u.ms)
            self.assertEqual(neuron.g_L, 28.95 * u.nS)
            self.assertEqual(neuron.C_m, 289.5 * u.pF)
            self.assertEqual(neuron.E_ex, 0.0 * u.mV)
            self.assertEqual(neuron.E_in, -75.0 * u.mV)
            self.assertEqual(neuron.E_L, -70.0 * u.mV)
            self.assertEqual(neuron.tau_syn_ex, 1.5 * u.ms)
            self.assertEqual(neuron.tau_syn_in, 10.0 * u.ms)
            self.assertEqual(neuron.I_e, 0.0 * u.pA)
            self.assertEqual(neuron.tau_sfa, 110.0 * u.ms)
            self.assertEqual(neuron.tau_rr, 1.97 * u.ms)
            self.assertEqual(neuron.E_sfa, -70.0 * u.mV)
            self.assertEqual(neuron.E_rr, -70.0 * u.mV)
            self.assertEqual(neuron.q_sfa, 14.48 * u.nS)
            self.assertEqual(neuron.q_rr, 3214.0 * u.nS)

    def test_parameter_validation(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                iaf_cond_exp_sfa_rr(1, C_m=0.0 * u.pF)
            with self.assertRaises(ValueError):
                iaf_cond_exp_sfa_rr(1, tau_syn_ex=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_cond_exp_sfa_rr(1, tau_syn_in=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_cond_exp_sfa_rr(1, tau_sfa=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_cond_exp_sfa_rr(1, tau_rr=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_cond_exp_sfa_rr(1, t_ref=-0.1 * u.ms)
            with self.assertRaises(ValueError):
                iaf_cond_exp_sfa_rr(1, V_reset=-57.0 * u.mV, V_th=-57.0 * u.mV)

    def test_signed_spike_weights_split_into_ex_and_in(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_exp_sfa_rr(
                1,
                V_th=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()
            self._step(neuron, 0, dg_values=[5.0, -3.0])
            self.assertTrue(u.math.allclose(neuron.g_ex.value, 5.0 * u.nS))
            self.assertTrue(u.math.allclose(neuron.g_in.value, 3.0 * u.nS))

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_exp_sfa_rr(
                1,
                E_L=0.0 * u.mV,
                g_L=0.0 * u.nS,
                I_e=0.0 * u.pA,
                q_sfa=0.0 * u.nS,
                q_rr=0.0 * u.nS,
                V_th=1e9 * u.mV,
                V_reset=0.0 * u.mV,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, x=100.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.0 * u.mV))

            self._step(neuron, 1, x=0.0 * u.pA)
            expected = (100.0 / 289.5) * 0.1
            self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), expected, delta=1e-12)

    def test_spike_adds_sfa_and_rr_conductances(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_exp_sfa_rr(
                1,
                E_L=0.0 * u.mV,
                g_L=0.0 * u.nS,
                C_m=200.0 * u.pF,
                I_e=8000.0 * u.pA,
                V_th=1.0 * u.mV,
                V_reset=0.0 * u.mV,
                t_ref=0.2 * u.ms,
                tau_sfa=10.0 * u.ms,
                tau_rr=2.0 * u.ms,
                q_sfa=3.0 * u.nS,
                q_rr=7.0 * u.nS,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            spk0 = self._step(neuron, 0)
            self.assertTrue(self._is_spike(spk0))
            self.assertAlmostEqual(float((neuron.g_sfa.value / u.nS)[0]), 3.0, delta=1e-10)
            self.assertAlmostEqual(float((neuron.g_rr.value / u.nS)[0]), 7.0, delta=1e-10)

            self._step(neuron, 1)
            self.assertLess(float((neuron.g_sfa.value / u.nS)[0]), 3.0)
            self.assertLess(float((neuron.g_rr.value / u.nS)[0]), 7.0)

    def test_matches_iaf_cond_exp_when_adaptation_disabled(self):
        with brainstate.environ.context(dt=self.dt):
            common = dict(
                E_L=-70.0 * u.mV,
                C_m=250.0 * u.pF,
                t_ref=0.5 * u.ms,
                V_th=-55.0 * u.mV,
                V_reset=-60.0 * u.mV,
                E_ex=0.0 * u.mV,
                E_in=-85.0 * u.mV,
                g_L=16.6667 * u.nS,
                tau_syn_ex=1.5 * u.ms,
                tau_syn_in=10.0 * u.ms,
                I_e=150.0 * u.pA,
                V_initializer=braintools.init.Constant(-69.0 * u.mV),
            )

            neuron_ref = iaf_cond_exp(1, **common)
            neuron_sfa = iaf_cond_exp_sfa_rr(
                1,
                **common,
                tau_sfa=110.0 * u.ms,
                tau_rr=1.97 * u.ms,
                q_sfa=0.0 * u.nS,
                q_rr=0.0 * u.nS,
                g_sfa_initializer=braintools.init.Constant(0.0 * u.nS),
                g_rr_initializer=braintools.init.Constant(0.0 * u.nS),
                E_sfa=-70.0 * u.mV,
                E_rr=-70.0 * u.mV,
            )
            neuron_ref.init_state()
            neuron_sfa.init_state()

            x_seq = [0.0, 80.0, 0.0, -30.0, 0.0, 60.0, 0.0, 0.0, -10.0, 0.0]
            dg_seq = [[4.0], [], [-2.0], [3.0, -1.0], [], [], [1.0], [], [], [2.0]]
            n_steps = len(x_seq)

            # Pre-compute per-step inputs as JAX arrays.
            x_arr = jnp.array(x_seq)
            w_ex_arr = jnp.array([sum(max(0.0, dg) for dg in dgs) for dgs in dg_seq])
            w_in_arr = jnp.array([sum(max(0.0, -dg) for dg in dgs) for dgs in dg_seq])

            # JIT-compiled simulation via for_loop with fixed-key delta inputs.
            def _run_step(k):
                neuron_ref.add_delta_input('w_ex', w_ex_arr[k] * u.nS, label='w_ex')
                neuron_ref.add_delta_input('w_in', w_in_arr[k] * u.nS, label='w_in')
                neuron_sfa.add_delta_input('w_ex', w_ex_arr[k] * u.nS, label='w_ex')
                neuron_sfa.add_delta_input('w_in', w_in_arr[k] * u.nS, label='w_in')
                with brainstate.environ.context(t=k * self.dt):
                    spk_ref = neuron_ref.update(x=x_arr[k] * u.pA)
                    spk_sfa = neuron_sfa.update(x=x_arr[k] * u.pA)
                return (
                    spk_ref,
                    neuron_ref.V.value / u.mV,
                    neuron_ref.g_ex.value / u.nS,
                    neuron_ref.g_in.value / u.nS,
                    spk_sfa,
                    neuron_sfa.V.value / u.mV,
                    neuron_sfa.g_ex.value / u.nS,
                    neuron_sfa.g_in.value / u.nS,
                )

            results = brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))
            spk_ref_all = np.asarray(results[0][:, 0])
            v_ref_all = np.asarray(results[1][:, 0])
            g_ex_ref_all = np.asarray(results[2][:, 0])
            g_in_ref_all = np.asarray(results[3][:, 0])
            spk_sfa_all = np.asarray(results[4][:, 0])
            v_sfa_all = np.asarray(results[5][:, 0])
            g_ex_sfa_all = np.asarray(results[6][:, 0])
            g_in_sfa_all = np.asarray(results[7][:, 0])

            for k in range(n_steps):
                self.assertEqual(bool(spk_ref_all[k] > 0.0), bool(spk_sfa_all[k] > 0.0))
                self.assertAlmostEqual(float(v_ref_all[k]), float(v_sfa_all[k]), delta=3e-6)
                self.assertAlmostEqual(float(g_ex_ref_all[k]), float(g_ex_sfa_all[k]), delta=2e-6)
                self.assertAlmostEqual(float(g_in_ref_all[k]), float(g_in_sfa_all[k]), delta=2e-6)

    def test_reference_trace_matches_nest_step_logic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_exp_sfa_rr(
                1,
                E_L=-70.0 * u.mV,
                C_m=289.5 * u.pF,
                t_ref=0.3 * u.ms,
                V_th=-57.0 * u.mV,
                V_reset=-70.0 * u.mV,
                E_ex=0.0 * u.mV,
                E_in=-75.0 * u.mV,
                g_L=28.95 * u.nS,
                tau_syn_ex=1.5 * u.ms,
                tau_syn_in=10.0 * u.ms,
                tau_sfa=30.0 * u.ms,
                tau_rr=2.0 * u.ms,
                E_sfa=-70.0 * u.mV,
                E_rr=-70.0 * u.mV,
                q_sfa=1.5 * u.nS,
                q_rr=200.0 * u.nS,
                I_e=500.0 * u.pA,
                gsl_error_tol=1e-3,
                V_initializer=braintools.init.Constant(-69.5 * u.mV),
            )
            neuron.init_state()

            x_seq = [0.0, 20.0, 0.0, -25.0, 0.0, 40.0, 0.0, 0.0, -10.0, 0.0, 0.0, 15.0] + [0.0] * 36
            dg_seq = [[5.0], [], [-1.5], [2.0, -0.5], [], [], [1.5], [], [0.0], [-2.0], [1.0], []] + [[]] * 36
            n_steps = len(x_seq)

            params = {
                'E_L': -70.0,
                'C_m': 289.5,
                't_ref': 0.3,
                'V_th': -57.0,
                'V_reset': -70.0,
                'E_ex': 0.0,
                'E_in': -75.0,
                'g_L': 28.95,
                'tau_syn_ex': 1.5,
                'tau_syn_in': 10.0,
                'tau_sfa': 30.0,
                'tau_rr': 2.0,
                'E_sfa': -70.0,
                'E_rr': -70.0,
                'q_sfa': 1.5,
                'q_rr': 200.0,
                'I_e': 500.0,
            }
            ref_state = {
                'v': -69.5,
                'g_ex': 0.0,
                'g_in': 0.0,
                'g_sfa': 0.0,
                'g_rr': 0.0,
                'r': 0,
                'h': 0.1,
                'i_stim': 0.0,
            }

            # Run reference in pure Python/NumPy first.
            ref_v, ref_g_ex, ref_g_in, ref_g_sfa, ref_g_rr = [], [], [], [], []
            ref_r, ref_h = [], []
            spikes_ref = []
            for k, (x_i, dgs) in enumerate(zip(x_seq, dg_seq)):
                spikes_ref.append(_reference_step(ref_state, params, x_i, dgs, 0.1))
                ref_v.append(ref_state['v'])
                ref_g_ex.append(ref_state['g_ex'])
                ref_g_in.append(ref_state['g_in'])
                ref_g_sfa.append(ref_state['g_sfa'])
                ref_g_rr.append(ref_state['g_rr'])
                ref_r.append(ref_state['r'])
                ref_h.append(ref_state['h'])

            # Pre-compute per-step inputs as JAX arrays.
            x_arr = jnp.array(x_seq)
            w_ex_arr = jnp.array([sum(max(0.0, dg) for dg in dgs) for dgs in dg_seq])
            w_in_arr = jnp.array([sum(max(0.0, -dg) for dg in dgs) for dgs in dg_seq])

            # JIT-compiled simulation via for_loop with fixed-key delta inputs.
            def _run_step(k):
                neuron.add_delta_input('w_ex', w_ex_arr[k] * u.nS, label='w_ex')
                neuron.add_delta_input('w_in', w_in_arr[k] * u.nS, label='w_in')
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=x_arr[k] * u.pA)
                return (
                    neuron.V.value / u.mV,
                    spk,
                    neuron.g_ex.value / u.nS,
                    neuron.g_in.value / u.nS,
                    neuron.g_sfa.value / u.nS,
                    neuron.g_rr.value / u.nS,
                    neuron.refractory_step_count.value,
                    neuron.integration_step.value / u.ms,
                )

            results = brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))
            v_all = np.asarray(results[0][:, 0])
            spk_all = np.asarray(results[1][:, 0])
            g_ex_all = np.asarray(results[2][:, 0])
            g_in_all = np.asarray(results[3][:, 0])
            g_sfa_all = np.asarray(results[4][:, 0])
            g_rr_all = np.asarray(results[5][:, 0])
            r_all = np.asarray(results[6][:, 0])
            h_all = np.asarray(results[7][:, 0])

            for k in range(n_steps):
                self.assertAlmostEqual(float(v_all[k]), ref_v[k], delta=4e-6)
                self.assertAlmostEqual(float(g_ex_all[k]), ref_g_ex[k], delta=3e-6)
                self.assertAlmostEqual(float(g_in_all[k]), ref_g_in[k], delta=3e-6)
                self.assertAlmostEqual(float(g_sfa_all[k]), ref_g_sfa[k], delta=3e-6)
                self.assertAlmostEqual(float(g_rr_all[k]), ref_g_rr[k], delta=3e-6)
                self.assertEqual(int(r_all[k]), ref_r[k])
                self.assertAlmostEqual(float(h_all[k]), ref_h[k], delta=3e-6)

            spikes_model = [bool(v > 0.0) for v in spk_all]
            self.assertEqual(spikes_model, spikes_ref)


if __name__ == '__main__':
    unittest.main()
