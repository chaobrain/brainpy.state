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

import importlib.util
import math
import unittest

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import aeif_psc_alpha


def _rhs(y, is_refractory, i_stim, p):
    v, dI_ex, I_ex, dI_in, I_in, w = y
    v_eff = p['V_reset'] if is_refractory else min(v, p['V_peak_rhs'])

    i_spike = 0.0 if p['Delta_T'] == 0.0 else p['g_L'] * p['Delta_T'] * math.exp((v_eff - p['V_th']) / p['Delta_T'])
    dv = 0.0 if is_refractory else (
                                       -p['g_L'] * (v_eff - p['E_L']) + i_spike + I_ex - I_in - w + p['I_e'] + i_stim
                                   ) / p['C_m']

    ddI_ex = -dI_ex / p['tau_syn_ex']
    dI_ex_dt = dI_ex - I_ex / p['tau_syn_ex']
    ddI_in = -dI_in / p['tau_syn_in']
    dI_in_dt = dI_in - I_in / p['tau_syn_in']
    dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
    dftype = brainstate.environ.dftype()
    return np.asarray([dv, ddI_ex, dI_ex_dt, ddI_in, dI_in_dt, dw], dtype=dftype)


def _reference_step(state, p, x_next, w_step, dt_ms):
    min_h = 1e-8
    t = 0.0
    h = max(state['h'], min_h)
    dftype = brainstate.environ.dftype()
    y = np.asarray([state['v'], state['dI_ex'], state['I_ex'], state['dI_in'], state['I_in'], state['w']],
                   dtype=dftype)
    r = int(state['r'])
    spike_count = 0
    iters = 0

    while t < dt_ms and iters < 100000:
        iters += 1
        h = max(min_h, min(h, dt_ms - t))
        is_refractory = r > 0

        k1 = _rhs(y, is_refractory, state['i_stim'], p)
        k2 = _rhs(y + h * (1.0 / 4.0) * k1, is_refractory, state['i_stim'], p)
        k3 = _rhs(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0), is_refractory, state['i_stim'], p)
        k4 = _rhs(
            y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
            is_refractory,
            state['i_stim'],
            p,
        )
        k5 = _rhs(
            y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
            is_refractory,
            state['i_stim'],
            p,
        )
        k6 = _rhs(
            y
            + h
            * (
                -8.0 * k1 / 27.0
                + 2.0 * k2
                - 3544.0 * k3 / 2565.0
                + 1859.0 * k4 / 4104.0
                - 11.0 * k5 / 40.0
            ),
            is_refractory,
            state['i_stim'],
            p,
        )

        y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
        y5 = y + h * (
                16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0)
        err = float(np.max(np.abs(y5 - y4)))
        atol = p['atol']

        if err <= atol or h <= min_h:
            y = y5
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)

            if y[0] < -1e3 or y[5] < -1e6 or y[5] > 1e6:
                raise ValueError('Numerical instability in reference aeif_psc_alpha.')

            if r > 0:
                y[0] = p['V_reset']
            elif y[0] >= p['V_peak_detect']:
                spike_count += 1
                y[0] = p['V_reset']
                y[5] += p['b']
                r = (p['refr_counts'] + 1) if p['refr_counts'] > 0 else 0
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    if r > 0:
        r -= 1

    if w_step >= 0.0:
        y[1] += math.e / p['tau_syn_ex'] * w_step
    else:
        y[3] += math.e / p['tau_syn_in'] * (-w_step)

    state['v'] = y[0]
    state['dI_ex'] = y[1]
    state['I_ex'] = y[2]
    state['dI_in'] = y[3]
    state['I_in'] = y[4]
    state['w'] = y[5]
    state['r'] = r
    state['h'] = h
    state['i_stim'] = x_next
    return spike_count


class TestAEIFPscAlpha(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        dftype = brainstate.environ.dftype()
        return bool(np.asarray(u.math.asarray(spk), dtype=dftype)[0] > 0.0)

    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def _step(self, neuron, k, x=0.0 * u.pA, dI_values=None):
        if dI_values is not None:
            for i, val in enumerate(dI_values):
                if val >= 0:
                    neuron.add_delta_input(f'delta_{k}_{i}', val * u.pA, label='w_ex')
                else:
                    neuron.add_delta_input(f'delta_{k}_{i}', (-val) * u.pA, label='w_in')
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_nest_cpp_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = aeif_psc_alpha(1)
            self.assertEqual(neuron.V_peak, 0.0 * u.mV)
            self.assertEqual(neuron.V_reset, -60.0 * u.mV)
            self.assertEqual(neuron.t_ref, 0.0 * u.ms)
            self.assertEqual(neuron.g_L, 30.0 * u.nS)
            self.assertEqual(neuron.C_m, 281.0 * u.pF)
            self.assertEqual(neuron.E_L, -70.6 * u.mV)
            self.assertEqual(neuron.Delta_T, 2.0 * u.mV)
            self.assertEqual(neuron.tau_w, 144.0 * u.ms)
            self.assertEqual(neuron.a, 4.0 * u.nS)
            self.assertEqual(neuron.b, 80.5 * u.pA)
            self.assertEqual(neuron.V_th, -50.4 * u.mV)
            self.assertEqual(neuron.tau_syn_ex, 0.2 * u.ms)
            self.assertEqual(neuron.tau_syn_in, 2.0 * u.ms)
            self.assertEqual(neuron.I_e, 0.0 * u.pA)

    def test_parameter_validation(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, V_reset=0.0 * u.mV, V_peak=0.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, Delta_T=-1.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, V_peak=-55.0 * u.mV, V_th=-50.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, C_m=0.0 * u.pF)
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, t_ref=-0.1 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, tau_syn_ex=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, tau_syn_in=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, tau_w=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, gsl_error_tol=0.0)
            with self.assertRaises(ValueError):
                aeif_psc_alpha(1, V_peak=1500.0 * u.mV, Delta_T=1e-12 * u.mV)

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_psc_alpha(
                1,
                V_th=1e6 * u.mV,
                V_peak=1e6 * u.mV,
                Delta_T=0.0 * u.mV,
                g_L=0.0 * u.nS,
                a=0.0 * u.nS,
                b=0.0 * u.pA,
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                V_reset=0.0 * u.mV,
            )
            neuron.init_state()

            self._step(neuron, 0, x=100.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.0 * u.mV))

            self._step(neuron, 1, x=0.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.03558718861209964 * u.mV, atol=1e-12 * u.mV))

    def test_reference_trace_matches_nest_step_logic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_psc_alpha(
                1,
                V_peak=0.0 * u.mV,
                V_reset=-58.0 * u.mV,
                t_ref=0.25 * u.ms,
                g_L=11.0 * u.nS,
                C_m=200.0 * u.pF,
                E_L=-70.0 * u.mV,
                Delta_T=2.0 * u.mV,
                tau_w=300.0 * u.ms,
                a=3.0 * u.nS,
                b=40.0 * u.pA,
                V_th=-50.0 * u.mV,
                tau_syn_ex=0.2 * u.ms,
                tau_syn_in=2.0 * u.ms,
                I_e=1200.0 * u.pA,
                gsl_error_tol=1e-6,
                V_initializer=braintools.init.Constant(-68.0 * u.mV),
                I_ex_initializer=braintools.init.Constant(0.8 * u.pA),
                I_in_initializer=braintools.init.Constant(0.3 * u.pA),
                w_initializer=braintools.init.Constant(5.0 * u.pA),
            )
            neuron.init_state()

            x_seq = [0.0, 20.0, 0.0, -30.0, 0.0, 40.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0] + [0.0] * 48
            w_seq = [0.0, 5.0, -2.0, 0.0, 4.0, -3.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2.5] + [0.0] * 48
            n_steps = len(x_seq)

            # Pre-compute per-step inputs as JAX arrays for JIT-compatible loop.
            x_arr = jnp.array(x_seq, dtype=jnp.float64)
            w_arr = jnp.array(w_seq, dtype=jnp.float64)

            def _model_step(k):
                x_i = x_arr[k]
                w_i = w_arr[k]
                # Route signed weights to excitatory / inhibitory labels via
                # jnp.where so the body is fully JIT-compatible.
                w_ex_k = u.math.where(w_i > 0., w_i, 0.) * u.pA
                w_in_k = u.math.where(w_i < 0., -w_i, 0.) * u.pA
                neuron.add_delta_input('_fl_ex', w_ex_k, label='w_ex')
                neuron.add_delta_input('_fl_in', w_in_k, label='w_in')
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=x_i * u.pA)
                return (
                    neuron.V.value / u.mV,
                    u.get_mantissa(neuron.dI_ex.value),
                    neuron.I_ex.value / u.pA,
                    u.get_mantissa(neuron.dI_in.value),
                    neuron.I_in.value / u.pA,
                    neuron.w.value / u.pA,
                    neuron.refractory_step_count.value,
                    neuron.integration_step.value / u.ms,
                    spk,
                )

            results = brainstate.transform.for_loop(_model_step, jnp.arange(n_steps))

            # Extract model results (shape: (n_steps, 1) → reshape to (n_steps,)).
            model_V = np.asarray(results[0]).reshape(n_steps)
            model_dI_ex = np.asarray(results[1]).reshape(n_steps)
            model_I_ex = np.asarray(results[2]).reshape(n_steps)
            model_dI_in = np.asarray(results[3]).reshape(n_steps)
            model_I_in = np.asarray(results[4]).reshape(n_steps)
            model_w = np.asarray(results[5]).reshape(n_steps)
            model_r = np.asarray(results[6]).reshape(n_steps)
            model_h = np.asarray(results[7]).reshape(n_steps)
            model_spk = np.asarray(results[8]).reshape(n_steps)

            # Run Python reference loop to compute expected values.
            p = {
                'V_peak_rhs': 0.0,
                'V_peak_detect': 0.0,
                'V_reset': -58.0,
                'g_L': 11.0,
                'C_m': 200.0,
                'E_L': -70.0,
                'Delta_T': 2.0,
                'tau_w': 300.0,
                'a': 3.0,
                'b': 40.0,
                'V_th': -50.0,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 2.0,
                'I_e': 1200.0,
                'atol': 1e-6,
                'refr_counts': int(math.ceil(0.25 / 0.1)),
            }
            ref_state = {
                'v': -68.0,
                'dI_ex': 0.0,
                'I_ex': 0.8,
                'dI_in': 0.0,
                'I_in': 0.3,
                'w': 5.0,
                'r': 0,
                'h': 0.1,
                'i_stim': 0.0,
            }
            ref_V, ref_dI_ex, ref_I_ex = [], [], []
            ref_dI_in, ref_I_in, ref_w, ref_r, ref_h, ref_spk = [], [], [], [], [], []
            for x_i, w_i in zip(x_seq, w_seq):
                n_spk = _reference_step(ref_state, p, x_i, w_i, 0.1)
                ref_V.append(ref_state['v'])
                ref_dI_ex.append(ref_state['dI_ex'])
                ref_I_ex.append(ref_state['I_ex'])
                ref_dI_in.append(ref_state['dI_in'])
                ref_I_in.append(ref_state['I_in'])
                ref_w.append(ref_state['w'])
                ref_r.append(ref_state['r'])
                ref_h.append(ref_state['h'])
                ref_spk.append(n_spk > 0)

            # Bulk post-loop comparisons.
            npt.assert_allclose(model_V, ref_V, atol=2e-6, rtol=0, err_msg='V mismatch')
            npt.assert_allclose(model_dI_ex, ref_dI_ex, atol=2e-6, rtol=0, err_msg='dI_ex mismatch')
            npt.assert_allclose(model_I_ex, ref_I_ex, atol=2e-6, rtol=0, err_msg='I_ex mismatch')
            npt.assert_allclose(model_dI_in, ref_dI_in, atol=2e-6, rtol=0, err_msg='dI_in mismatch')
            npt.assert_allclose(model_I_in, ref_I_in, atol=2e-6, rtol=0, err_msg='I_in mismatch')
            npt.assert_allclose(model_w, ref_w, atol=2e-6, rtol=0, err_msg='w mismatch')
            npt.assert_array_equal(model_r, ref_r)
            npt.assert_allclose(model_h, ref_h, atol=2e-6, rtol=0, err_msg='h mismatch')
            npt.assert_array_equal(model_spk > 0, ref_spk)
            self.assertTrue(np.any(model_spk > 0))

    @unittest.skip('Multiple internal spikes per integration step not yet supported by RK45 integrator')
    def test_zero_refractory_allows_multiple_internal_spikes_and_updates_w(self):
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = aeif_psc_alpha(
                1,
                V_peak=0.0 * u.mV,
                V_reset=-60.0 * u.mV,
                t_ref=0.0 * u.ms,
                g_L=0.0 * u.nS,
                C_m=10.0 * u.pF,
                E_L=-70.0 * u.mV,
                Delta_T=0.0 * u.mV,
                tau_w=1000.0 * u.ms,
                a=0.0 * u.nS,
                b=1.0 * u.pA,
                V_th=-55.0 * u.mV,
                tau_syn_ex=0.2 * u.ms,
                tau_syn_in=2.0 * u.ms,
                I_e=10000.0 * u.pA,
                gsl_error_tol=1e-15,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
                w_initializer=braintools.init.Constant(0.0 * u.pA),
            )
            neuron.init_state()

            p = {
                'V_peak_rhs': 0.0,
                'V_peak_detect': -55.0,  # Delta_T == 0 uses V_th for detection.
                'V_reset': -60.0,
                'g_L': 0.0,
                'C_m': 10.0,
                'E_L': -70.0,
                'Delta_T': 0.0,
                'tau_w': 1000.0,
                'a': 0.0,
                'b': 1.0,
                'V_th': -55.0,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 2.0,
                'I_e': 10000.0,
                'atol': 1e-15,
                'refr_counts': 0,
            }
            ref_state = {
                'v': -60.0,
                'dI_ex': 0.0,
                'I_ex': 0.0,
                'dI_in': 0.0,
                'I_in': 0.0,
                'w': 0.0,
                'r': 0,
                'h': 1.0,
                'i_stim': 0.0,
            }

            n_spikes = _reference_step(ref_state, p, 0.0, 0.0, 1.0)
            self.assertGreater(n_spikes, 1)

            with brainstate.environ.context(t=0.0 * u.ms):
                spk = neuron.update(x=0.0 * u.pA)

            self.assertTrue(self._is_spike(spk))
            self.assertAlmostEqual(float((neuron.w.value / u.pA)[0]), ref_state['w'], delta=2e-6)
            self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), ref_state['v'], delta=2e-6)
            self.assertEqual(int(neuron.refractory_step_count.value[0]), ref_state['r'])
            self.assertAlmostEqual(float((neuron.last_spike_time.value / u.ms)[0]), 1.0, delta=1e-12)

    def test_direct_trace_matches_nest_if_available(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        if 'aeif_psc_alpha' not in nest.Models():
            self.skipTest('NEST model aeif_psc_alpha not available')

        dt_ms = 0.1
        n_steps = 200

        params = {
            'V_peak': 0.0,
            'V_reset': -58.0,
            't_ref': 0.2,
            'g_L': 11.0,
            'C_m': 200.0,
            'E_L': -70.0,
            'Delta_T': 2.0,
            'tau_w': 300.0,
            'a': 3.0,
            'b': 40.0,
            'V_th': -50.0,
            'tau_syn_ex': 0.2,
            'tau_syn_in': 2.0,
            'I_e': 420.0,
            'gsl_error_tol': 1e-6,
            'V_m': -67.2,
            'w': 5.0,
            'I_syn_ex': 0.8,
            'dI_syn_ex': 0.4,
            'I_syn_in': 0.3,
            'dI_syn_in': 0.1,
        }

        nest.ResetKernel()
        nest.resolution = dt_ms

        nrn = nest.Create('aeif_psc_alpha', params=params)
        mm = nest.Create('multimeter', params={
            'record_from': ['V_m', 'w', 'I_syn_ex', 'I_syn_in'],
            'interval': dt_ms,
        })
        nest.Connect(mm, nrn)
        nest.Simulate(n_steps * dt_ms)

        events = mm.get('events')
        dftype = brainstate.environ.dftype()
        nest_v = np.asarray(events['V_m'], dtype=dftype)
        nest_w = np.asarray(events['w'], dtype=dftype)
        nest_i_ex = np.asarray(events['I_syn_ex'], dtype=dftype)
        nest_i_in = np.asarray(events['I_syn_in'], dtype=dftype)
        nest_times = np.asarray(events['times'], dtype=dftype)

        with brainstate.environ.context(dt=dt_ms * u.ms):
            neuron = aeif_psc_alpha(
                1,
                V_peak=params['V_peak'] * u.mV,
                V_reset=params['V_reset'] * u.mV,
                t_ref=params['t_ref'] * u.ms,
                g_L=params['g_L'] * u.nS,
                C_m=params['C_m'] * u.pF,
                E_L=params['E_L'] * u.mV,
                Delta_T=params['Delta_T'] * u.mV,
                tau_w=params['tau_w'] * u.ms,
                a=params['a'] * u.nS,
                b=params['b'] * u.pA,
                V_th=params['V_th'] * u.mV,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                gsl_error_tol=params['gsl_error_tol'],
                V_initializer=braintools.init.Constant(params['V_m'] * u.mV),
                I_ex_initializer=braintools.init.Constant(params['I_syn_ex'] * u.pA),
                I_in_initializer=braintools.init.Constant(params['I_syn_in'] * u.pA),
                w_initializer=braintools.init.Constant(params['w'] * u.pA),
            )
            neuron.init_state()
            neuron.dI_ex.value = np.asarray([params['dI_syn_ex']], dtype=dftype) * (u.pA / u.ms)
            neuron.dI_in.value = np.asarray([params['dI_syn_in']], dtype=dftype) * (u.pA / u.ms)

            def _run_step(k):
                with brainstate.environ.context(t=(k * dt_ms) * u.ms):
                    neuron.update(x=0.0 * u.pA)
                return (
                    neuron.V.value / u.mV,
                    neuron.w.value / u.pA,
                    neuron.I_ex.value / u.pA,
                    neuron.I_in.value / u.pA,
                )

            results = brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))
            bp_v = np.asarray(results[0].flatten(), dtype=dftype)
            bp_w = np.asarray(results[1].flatten(), dtype=dftype)
            bp_i_ex = np.asarray(results[2].flatten(), dtype=dftype)
            bp_i_in = np.asarray(results[3].flatten(), dtype=dftype)

        bp_indices = np.rint(nest_times / dt_ms).astype(np.int64) - 1
        self.assertTrue(np.all(bp_indices >= 0))
        self.assertTrue(np.all(bp_indices < n_steps))

        npt.assert_allclose(bp_v[bp_indices], nest_v, atol=3e-5, rtol=0.0, err_msg='V_m trace mismatch vs NEST')
        npt.assert_allclose(bp_w[bp_indices], nest_w, atol=2e-5, rtol=0.0, err_msg='w trace mismatch vs NEST')
        npt.assert_allclose(bp_i_ex[bp_indices], nest_i_ex, atol=2e-5, rtol=0.0,
                            err_msg='I_syn_ex trace mismatch vs NEST')
        npt.assert_allclose(bp_i_in[bp_indices], nest_i_in, atol=2e-5, rtol=0.0,
                            err_msg='I_syn_in trace mismatch vs NEST')


if __name__ == '__main__':
    unittest.main()
