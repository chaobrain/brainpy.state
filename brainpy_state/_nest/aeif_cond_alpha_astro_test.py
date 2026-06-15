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
import inspect
import math
import unittest

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from brainpy_state import aeif_cond_alpha_astro

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _rhs(y, is_refractory, i_stim, i_sic, p):
    v, dg_ex, g_ex, dg_in, g_in, w = y
    v_eff = p['V_reset'] if is_refractory else min(v, p['V_peak_rhs'])

    i_syn_ex = g_ex * (v_eff - p['E_ex'])
    i_syn_in = g_in * (v_eff - p['E_in'])
    i_spike = 0.0 if p['Delta_T'] == 0.0 else p['g_L'] * p['Delta_T'] * math.exp((v_eff - p['V_th']) / p['Delta_T'])
    dv = 0.0 if is_refractory else (
                                       -p['g_L'] * (v_eff - p['E_L']) + i_spike - i_syn_ex - i_syn_in - w + p[
                                       'I_e'] + i_stim + i_sic
                                   ) / p['C_m']

    ddg_ex = -dg_ex / p['tau_syn_ex']
    dg_ex_dt = dg_ex - g_ex / p['tau_syn_ex']
    ddg_in = -dg_in / p['tau_syn_in']
    dg_in_dt = dg_in - g_in / p['tau_syn_in']
    dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
    dftype = brainstate.environ.dftype()
    return np.asarray([dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt, dw], dtype=dftype)


def _reference_step(state, p, x_next, w_step, dt_ms):
    min_h = 1e-8
    t = 0.0
    h = max(state['h'], min_h)
    dftype = brainstate.environ.dftype()
    y = np.asarray([state['v'], state['dg_ex'], state['g_ex'], state['dg_in'], state['g_in'], state['w']],
                   dtype=dftype)
    r = int(state['r'])
    spike_count = 0
    iters = 0

    while t < dt_ms and iters < 100000:
        iters += 1
        h = max(min_h, min(h, dt_ms - t))
        is_refractory = r > 0

        k1 = _rhs(y, is_refractory, state['i_stim'], state['i_sic'], p)
        k2 = _rhs(y + h * (1.0 / 4.0) * k1, is_refractory, state['i_stim'], state['i_sic'], p)
        k3 = _rhs(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0), is_refractory, state['i_stim'], state['i_sic'], p)
        k4 = _rhs(
            y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
            is_refractory,
            state['i_stim'],
            state['i_sic'],
            p,
        )
        k5 = _rhs(
            y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
            is_refractory,
            state['i_stim'],
            state['i_sic'],
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
            state['i_sic'],
            p,
        )

        y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
        y5 = y + h * (
            16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0
        )
        err = float(np.max(np.abs(y5 - y4)))
        atol = p['atol']

        if err <= atol or h <= min_h:
            y = y5
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)

            if y[0] < -1e3 or y[5] < -1e6 or y[5] > 1e6:
                raise ValueError('Numerical instability in reference aeif_cond_alpha_astro.')

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
    state['dg_ex'] = y[1]
    state['g_ex'] = y[2]
    state['dg_in'] = y[3]
    state['g_in'] = y[4]
    state['w'] = y[5]
    state['r'] = r
    state['h'] = h
    state['i_stim'] = x_next
    return spike_count


def _enqueue_reference_sic_event(queue, current_step, event):
    if event is None:
        return

    weight = float(event.get('weight', 1.0))
    dftype = brainstate.environ.dftype()
    coeffs = np.asarray(event.get('coeffs', event.get('coefficients', 0.0)), dtype=dftype).reshape(-1)
    delay_steps = int(event.get('delay_steps', 1))
    if delay_steps <= 0:
        raise ValueError('delay_steps must be positive')

    base_offset = delay_steps - 1
    for i, coeff in enumerate(coeffs):
        idx = current_step + base_offset + i
        queue[idx] = queue.get(idx, 0.0) + weight * float(coeff)


class TestAEIFCondAlphaAstro(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        dftype = brainstate.environ.dftype()
        return bool(np.asarray(u.math.asarray(spk), dtype=dftype).reshape(-1)[0] > 0.0)

    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                if val >= 0:
                    neuron.add_delta_input(f'delta_{k}_{i}', val * u.nS, label='w_ex')
                else:
                    neuron.add_delta_input(f'delta_{k}_{i}', (-val) * u.nS, label='w_in')
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_nest_cpp_default_parameters_and_recordables(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = aeif_cond_alpha_astro(1)
            self.assertEqual(neuron.V_peak, 0.0 * u.mV)
            self.assertEqual(neuron.V_reset, -60.0 * u.mV)
            self.assertEqual(neuron.t_ref, 0.0 * u.ms)
            self.assertEqual(neuron.g_L, 30.0 * u.nS)
            self.assertEqual(neuron.C_m, 281.0 * u.pF)
            self.assertEqual(neuron.E_ex, 0.0 * u.mV)
            self.assertEqual(neuron.E_in, -85.0 * u.mV)
            self.assertEqual(neuron.E_L, -70.6 * u.mV)
            self.assertEqual(neuron.Delta_T, 2.0 * u.mV)
            self.assertEqual(neuron.tau_w, 144.0 * u.ms)
            self.assertEqual(neuron.a, 4.0 * u.nS)
            self.assertEqual(neuron.b, 80.5 * u.pA)
            self.assertEqual(neuron.V_th, -50.4 * u.mV)
            self.assertEqual(neuron.tau_syn_ex, 0.2 * u.ms)
            self.assertEqual(neuron.tau_syn_in, 2.0 * u.ms)
            self.assertEqual(neuron.I_e, 0.0 * u.pA)
            self.assertEqual(neuron.recordables, ['V_m', 'g_ex', 'g_in', 'w', 'I_SIC'])

    def test_parameter_validation(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, V_reset=0.0 * u.mV, V_peak=0.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, Delta_T=-1.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, V_peak=-55.0 * u.mV, V_th=-50.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, C_m=0.0 * u.pF)
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, t_ref=-0.1 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, tau_syn_ex=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, tau_syn_in=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, tau_w=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, gsl_error_tol=0.0)
            with self.assertRaises(ValueError):
                aeif_cond_alpha_astro(1, V_peak=1500.0 * u.mV, Delta_T=1e-12 * u.mV)

    def test_reference_trace_matches_nest_step_logic_with_sic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_cond_alpha_astro(
                1,
                V_peak=0.0 * u.mV,
                V_reset=-58.0 * u.mV,
                t_ref=0.25 * u.ms,
                g_L=11.0 * u.nS,
                C_m=200.0 * u.pF,
                E_ex=0.0 * u.mV,
                E_in=-85.0 * u.mV,
                E_L=-70.0 * u.mV,
                Delta_T=2.0 * u.mV,
                tau_w=300.0 * u.ms,
                a=3.0 * u.nS,
                b=40.0 * u.pA,
                V_th=-50.0 * u.mV,
                tau_syn_ex=0.2 * u.ms,
                tau_syn_in=2.0 * u.ms,
                I_e=900.0 * u.pA,
                gsl_error_tol=1e-6,
                V_initializer=braintools.init.Constant(-68.0 * u.mV),
                w_initializer=braintools.init.Constant(5.0 * u.pA),
            )
            neuron.init_state()

            n_steps = 80
            dftype = brainstate.environ.dftype()
            x_seq = np.zeros(n_steps, dtype=dftype)
            x_seq[[1, 5, 9, 13, 21, 34]] = np.asarray([20.0, -30.0, 40.0, -10.0, 50.0, -20.0], dtype=dftype)
            w_seq = np.zeros(n_steps, dtype=dftype)
            w_seq[[0, 2, 4, 7, 11, 30]] = np.asarray([4.0, -2.5, 3.0, -1.0, 2.0, -1.5], dtype=dftype)

            sic_seq = [None] * n_steps
            sic_seq[0] = {'weight': 1.5, 'coeffs': [10.0, -4.0], 'delay_steps': 2}
            sic_seq[6] = {'weight': -0.8, 'coeffs': [6.0, 0.0, 2.0], 'delay_steps': 3}
            sic_seq[18] = {'weight': 1.2, 'coeffs': 5.0, 'delay_steps': 1}

            p = {
                'V_peak_rhs': 0.0,
                'V_peak_detect': 0.0,
                'V_reset': -58.0,
                'g_L': 11.0,
                'C_m': 200.0,
                'E_ex': 0.0,
                'E_in': -85.0,
                'E_L': -70.0,
                'Delta_T': 2.0,
                'tau_w': 300.0,
                'a': 3.0,
                'b': 40.0,
                'V_th': -50.0,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 2.0,
                'I_e': 900.0,
                'atol': 1e-6,
                'refr_counts': int(math.ceil(float((0.25 * u.ms) / self.dt))),
            }
            ref_state = {
                'v': -68.0,
                'dg_ex': 0.0,
                'g_ex': 0.0,
                'dg_in': 0.0,
                'g_in': 0.0,
                'w': 5.0,
                'r': 0,
                'h': float((self.dt / u.ms)),
                'i_stim': 0.0,
                'i_sic': 0.0,
            }

            # --- Phase 1: Reference simulation (pure Python) ---
            sic_queue = {}
            ditype = brainstate.environ.ditype()
            ref_v = np.zeros(n_steps, dtype=dftype)
            ref_w = np.zeros(n_steps, dtype=dftype)
            ref_g_ex = np.zeros(n_steps, dtype=dftype)
            ref_g_in = np.zeros(n_steps, dtype=dftype)
            ref_i_sic = np.zeros(n_steps, dtype=dftype)
            ref_r = np.zeros(n_steps, dtype=ditype)
            ref_spk = np.zeros(n_steps, dtype=np.bool_)

            for k in range(n_steps):
                n_spikes = _reference_step(ref_state, p, x_seq[k], w_seq[k], float((self.dt / u.ms)))
                _enqueue_reference_sic_event(sic_queue, k, sic_seq[k])
                ref_state['i_sic'] = sic_queue.pop(k, 0.0)

                ref_spk[k] = n_spikes > 0
                ref_v[k] = ref_state['v']
                ref_w[k] = ref_state['w']
                ref_g_ex[k] = ref_state['g_ex']
                ref_g_in[k] = ref_state['g_in']
                ref_i_sic[k] = ref_state['i_sic']
                ref_r[k] = ref_state['r']

            # --- Phase 2: Pre-compute SIC output and delta inputs ---
            sic_q_pre = {}
            sic_output = np.zeros(n_steps, dtype=dftype)
            for k in range(n_steps):
                _enqueue_reference_sic_event(sic_q_pre, k, sic_seq[k])
                sic_output[k] = sic_q_pre.pop(k, 0.0)

            w_ex_arr = np.where(w_seq >= 0, w_seq, 0.0).astype(dftype)
            w_in_arr = np.where(w_seq < 0, -w_seq, 0.0).astype(dftype)

            # --- Phase 3: brainpy simulation with for_loop ---
            x_jnp = jnp.asarray(x_seq)
            w_ex_jnp = jnp.asarray(w_ex_arr)
            w_in_jnp = jnp.asarray(w_in_arr)
            sic_out_jnp = jnp.asarray(sic_output).reshape(n_steps, 1)
            pscon_ex = np.e / neuron.tau_syn_ex
            pscon_in = np.e / neuron.tau_syn_in

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=x_jnp[k] * u.pA)
                neuron.dg_ex.value = neuron.dg_ex.value + pscon_ex * w_ex_jnp[k] * u.nS
                neuron.dg_in.value = neuron.dg_in.value + pscon_in * w_in_jnp[k] * u.nS
                neuron.I_sic.value = sic_out_jnp[k] * u.pA
                return (
                    neuron.V.value / u.mV,
                    neuron.w.value / u.pA,
                    neuron.g_ex.value / u.nS,
                    neuron.g_in.value / u.nS,
                    neuron.I_sic.value / u.pA,
                    u.math.asarray(neuron.refractory_step_count.value, dtype=dftype),
                    spk,
                )

            results = brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))
            bp_v = np.asarray(results[0].flatten(), dtype=dftype)
            bp_w = np.asarray(results[1].flatten(), dtype=dftype)
            bp_g_ex = np.asarray(results[2].flatten(), dtype=dftype)
            bp_g_in = np.asarray(results[3].flatten(), dtype=dftype)
            bp_i_sic = np.asarray(results[4].flatten(), dtype=dftype)
            bp_r = np.asarray(results[5].flatten(), dtype=ditype)
            bp_spk = np.asarray(results[6].flatten() > 0, dtype=np.bool_)

            npt.assert_allclose(bp_v, ref_v, atol=2e-6, rtol=0.0)
            npt.assert_allclose(bp_w, ref_w, atol=2e-6, rtol=0.0)
            npt.assert_allclose(bp_g_ex, ref_g_ex, atol=2e-6, rtol=0.0)
            npt.assert_allclose(bp_g_in, ref_g_in, atol=2e-6, rtol=0.0)
            npt.assert_allclose(bp_i_sic, ref_i_sic, atol=1e-12, rtol=0.0)
            npt.assert_array_equal(bp_r, ref_r)
            npt.assert_array_equal(bp_spk, ref_spk)

    def test_direct_trace_matches_nest_if_available(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        if 'aeif_cond_alpha_astro' not in nest.Models():
            self.skipTest('NEST model aeif_cond_alpha_astro not available')

        dt_ms = 0.1
        n_steps = 200

        params = {
            'V_peak': 0.0,
            'V_reset': -58.0,
            't_ref': 0.2,
            'g_L': 11.0,
            'C_m': 200.0,
            'E_ex': 0.0,
            'E_in': -85.0,
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
            'g_ex': 0.8,
            'dg_ex': 0.4,
            'g_in': 0.3,
            'dg_in': 0.1,
        }

        nest.ResetKernel()
        nest.resolution = dt_ms

        nrn = nest.Create('aeif_cond_alpha_astro', params=params)
        mm = nest.Create('multimeter', params={
            'record_from': ['V_m', 'w', 'g_ex', 'g_in', 'I_SIC'],
            'interval': dt_ms,
        })
        nest.Connect(mm, nrn)
        nest.Simulate(n_steps * dt_ms)

        events = mm.get('events')
        dftype = brainstate.environ.dftype()
        nest_v = np.asarray(events['V_m'], dtype=dftype)
        nest_w = np.asarray(events['w'], dtype=dftype)
        nest_g_ex = np.asarray(events['g_ex'], dtype=dftype)
        nest_g_in = np.asarray(events['g_in'], dtype=dftype)
        nest_i_sic = np.asarray(events['I_SIC'], dtype=dftype)
        nest_times = np.asarray(events['times'], dtype=dftype)

        with brainstate.environ.context(dt=dt_ms * u.ms):
            neuron = aeif_cond_alpha_astro(
                1,
                V_peak=params['V_peak'] * u.mV,
                V_reset=params['V_reset'] * u.mV,
                t_ref=params['t_ref'] * u.ms,
                g_L=params['g_L'] * u.nS,
                C_m=params['C_m'] * u.pF,
                E_ex=params['E_ex'] * u.mV,
                E_in=params['E_in'] * u.mV,
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
                g_ex_initializer=braintools.init.Constant(params['g_ex'] * u.nS),
                g_in_initializer=braintools.init.Constant(params['g_in'] * u.nS),
                w_initializer=braintools.init.Constant(params['w'] * u.pA),
            )
            neuron.init_state()
            neuron.dg_ex.value = np.asarray([params['dg_ex']], dtype=dftype) * (u.nS / u.ms)
            neuron.dg_in.value = np.asarray([params['dg_in']], dtype=dftype) * (u.nS / u.ms)

            def _run_step(k):
                with brainstate.environ.context(t=(k * dt_ms) * u.ms):
                    neuron.update(x=0.0 * u.pA)
                return (
                    neuron.V.value / u.mV,
                    neuron.w.value / u.pA,
                    neuron.g_ex.value / u.nS,
                    neuron.g_in.value / u.nS,
                    neuron.I_sic.value / u.pA,
                )

            results = brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))
            bp_v = np.asarray(results[0].flatten(), dtype=dftype)
            bp_w = np.asarray(results[1].flatten(), dtype=dftype)
            bp_g_ex = np.asarray(results[2].flatten(), dtype=dftype)
            bp_g_in = np.asarray(results[3].flatten(), dtype=dftype)
            bp_i_sic = np.asarray(results[4].flatten(), dtype=dftype)

        bp_indices = np.rint(nest_times / dt_ms).astype(np.int64) - 1
        valid = (bp_indices >= 0) & (bp_indices < n_steps)
        bp_indices = bp_indices[valid]

        npt.assert_allclose(bp_v[bp_indices], nest_v[valid], atol=2e-5, rtol=0.0, err_msg='V_m trace mismatch vs NEST')
        npt.assert_allclose(bp_w[bp_indices], nest_w[valid], atol=2e-5, rtol=0.0, err_msg='w trace mismatch vs NEST')
        npt.assert_allclose(
            bp_g_ex[bp_indices], nest_g_ex[valid], atol=2e-5, rtol=0.0, err_msg='g_ex trace mismatch vs NEST'
        )
        npt.assert_allclose(
            bp_g_in[bp_indices], nest_g_in[valid], atol=2e-5, rtol=0.0, err_msg='g_in trace mismatch vs NEST'
        )
        npt.assert_allclose(
            bp_i_sic[bp_indices], nest_i_sic[valid], atol=2e-12, rtol=0.0, err_msg='I_SIC trace mismatch vs NEST'
        )

    def test_direct_sic_trace_matches_nest_astrocyte_pipeline_if_available(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        required_models = {'astrocyte_lr_1994', 'aeif_cond_alpha_astro'}
        if not required_models.issubset(set(nest.Models())):
            self.skipTest('NEST astrocyte/SIC models not available')

        try:
            sic_delay_ms = float(nest.GetDefaults('sic_connection')['delay'])
        except Exception:
            self.skipTest('NEST model sic_connection not available')

        dt_ms = 0.1
        sim_ms = 30.0

        nest.ResetKernel()
        nest.resolution = dt_ms

        astro = nest.Create('astrocyte_lr_1994', params={'Ca_astro': 0.2})
        nrn = nest.Create('aeif_cond_alpha_astro')
        nest.Connect(astro, nrn, syn_spec={'synapse_model': 'sic_connection'})

        mm_nrn = nest.Create('multimeter',
                             params={'record_from': ['V_m', 'w', 'g_ex', 'g_in', 'I_SIC'], 'interval': dt_ms})
        mm_astro = nest.Create('multimeter', params={'record_from': ['Ca_astro'], 'interval': dt_ms})
        nest.Connect(mm_nrn, nrn)
        nest.Connect(mm_astro, astro)

        nest.Simulate(sim_ms)

        nrn_events = mm_nrn.get('events')
        astro_events = mm_astro.get('events')
        dftype = brainstate.environ.dftype()
        nest_times = np.asarray(nrn_events['times'], dtype=dftype)
        nest_v = np.asarray(nrn_events['V_m'], dtype=dftype)
        nest_w = np.asarray(nrn_events['w'], dtype=dftype)
        nest_g_ex = np.asarray(nrn_events['g_ex'], dtype=dftype)
        nest_g_in = np.asarray(nrn_events['g_in'], dtype=dftype)
        nest_i_sic = np.asarray(nrn_events['I_SIC'], dtype=dftype)
        ca = np.asarray(astro_events['Ca_astro'], dtype=dftype)
        self.assertGreater(len(ca), 0)

        # Same coefficient function as NEST test_sic_connection.py.
        arg = ca * 1000.0 - 196.69
        coeff = np.zeros_like(ca)
        valid_arg = arg > 1.0
        coeff[valid_arg] = np.log(arg[valid_arg])

        delay_steps = int(round(sic_delay_ms / dt_ms))
        # Coefficients are derived from multimeter samples at t+dt, so we add one step
        # to align emitted SIC coefficients with the event delivery timeline.
        injected_delay_steps = delay_steps + 1

        with brainstate.environ.context(dt=dt_ms * u.ms):
            neuron = aeif_cond_alpha_astro(1)
            neuron.init_state()

            n_steps = len(ca)

            # Pre-compute SIC output values from the queue logic
            sic_q = {}
            sic_output = np.zeros(n_steps, dtype=dftype)
            for k in range(n_steps):
                _enqueue_reference_sic_event(
                    sic_q, k, {'weight': 1.0, 'coeffs': coeff[k], 'delay_steps': injected_delay_steps}
                )
                sic_output[k] = sic_q.pop(k, 0.0)

            sic_out_jnp = jnp.asarray(sic_output).reshape(n_steps, 1)

            def _run_step(k):
                with brainstate.environ.context(t=(k * dt_ms) * u.ms):
                    neuron.update(x=0.0 * u.pA)
                neuron.I_sic.value = sic_out_jnp[k] * u.pA
                return (
                    neuron.V.value / u.mV,
                    neuron.w.value / u.pA,
                    neuron.g_ex.value / u.nS,
                    neuron.g_in.value / u.nS,
                    neuron.I_sic.value / u.pA,
                )

            results = brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))
            bp_v = np.asarray(results[0].flatten(), dtype=dftype)
            bp_w = np.asarray(results[1].flatten(), dtype=dftype)
            bp_g_ex = np.asarray(results[2].flatten(), dtype=dftype)
            bp_g_in = np.asarray(results[3].flatten(), dtype=dftype)
            bp_i_sic = np.asarray(results[4].flatten(), dtype=dftype)

        bp_indices = np.rint(nest_times / dt_ms).astype(np.int64) - 1
        valid = (bp_indices >= 0) & (bp_indices < len(bp_v))
        bp_indices = bp_indices[valid]

        npt.assert_allclose(bp_v[bp_indices], nest_v[valid], atol=2e-5, rtol=0.0,
                            err_msg='V_m mismatch vs NEST SIC run')
        npt.assert_allclose(bp_w[bp_indices], nest_w[valid], atol=2e-5, rtol=0.0, err_msg='w mismatch vs NEST SIC run')
        npt.assert_allclose(
            bp_g_ex[bp_indices], nest_g_ex[valid], atol=2e-12, rtol=0.0, err_msg='g_ex mismatch vs NEST SIC run'
        )
        npt.assert_allclose(
            bp_g_in[bp_indices], nest_g_in[valid], atol=2e-12, rtol=0.0, err_msg='g_in mismatch vs NEST SIC run'
        )
        npt.assert_allclose(
            bp_i_sic[bp_indices], nest_i_sic[valid], atol=2e-10, rtol=0.0, err_msg='I_SIC mismatch vs NEST SIC run'
        )


class TestAEIFCondAlphaAstroSICChannel(unittest.TestCase):
    r"""The de-queued SIC path: the astrocyte slow-inward current arrives via a
    labelled ``'I_SIC'`` *current* channel (``add_current_input``), is read one step
    before it enters ``dV/dt`` (one-step lag, like ``I_stim``), and never collides
    with the device ``I_stim`` channel. No host-side ``sic_events`` queue remains.

    These pin the bucket-3 de-queue: the host Python dict event-emulator
    (``_sic_queue``/``_sic_step``) and the ``sic_events=`` kwarg are gone; the SIC
    current is a State-backed channel read instead, so the whole loop lowers under
    ``brainstate.transform.for_loop``.
    """

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _passive(self, **over):
        # Passive integrator (no leak/adaptation/spike): C_m = 100 pF so a constant
        # current I drives V linearly (dV = I/C_m * dt). Lets us read SIC delivery
        # straight off V / I_sic without ODE confounds.
        params = dict(
            V_th=1e6 * u.mV, V_peak=1e6 * u.mV, Delta_T=0.0 * u.mV, g_L=0.0 * u.nS,
            C_m=100.0 * u.pF, a=0.0 * u.nS, b=0.0 * u.pA, I_e=0.0 * u.pA,
            V_initializer=braintools.init.Constant(0.0 * u.mV), V_reset=0.0 * u.mV,
        )
        params.update(over)
        n = aeif_cond_alpha_astro(1, **params)
        n.init_state()
        return n

    def test_update_has_no_sic_events_kwarg(self):
        # The host-queue kwarg is removed; SIC arrives via the current channel.
        n = self._passive()
        self.assertNotIn('sic_events', inspect.signature(n.update).parameters)

    def test_no_host_sic_queue_attributes(self):
        # The host-side dict event-emulator is gone (not JAX-traceable).
        n = self._passive()
        self.assertFalse(hasattr(n, '_sic_queue'))
        self.assertFalse(hasattr(n, '_sic_step'))

    def test_sic_current_channel_one_step_lag(self):
        # A SIC current deposited on the labelled channel lands in I_sic this step
        # (stored for the next), and modulates V one step later — never V now.
        n = self._passive()
        with brainstate.environ.context(t=0.0 * u.ms):
            n.add_current_input('sic', 50.0, label='I_SIC')  # unitless -> pA on write-back
            n.update(x=0.0 * u.pA)
        self.assertTrue(u.math.allclose(n.V.value, 0.0 * u.mV, atol=1e-12 * u.mV))
        self.assertTrue(u.math.allclose(n.I_sic.value, 50.0 * u.pA, atol=1e-12 * u.pA))

        with brainstate.environ.context(t=self.dt):
            n.update(x=0.0 * u.pA)  # I_sic = 50 pA drives V: 50pA/100pF*0.1ms = 0.05 mV
        self.assertTrue(u.math.allclose(n.V.value, 0.05 * u.mV, atol=1e-10 * u.mV))
        self.assertTrue(u.math.allclose(n.I_sic.value, 0.0 * u.pA, atol=1e-12 * u.pA))

    def test_i_stim_and_i_sic_do_not_collide(self):
        # A device current (I_stim via x=) and a SIC current (labelled channel)
        # coexist: each lands in its OWN term (read order pops I_SIC first), and the
        # membrane sees their SUM one step later — not one channel swallowing both.
        n = self._passive()
        with brainstate.environ.context(t=0.0 * u.ms):
            n.add_current_input('sic', 50.0, label='I_SIC')
            n.update(x=100.0 * u.pA)
        self.assertTrue(u.math.allclose(n.I_stim.value, 100.0 * u.pA, atol=1e-12 * u.pA))
        self.assertTrue(u.math.allclose(n.I_sic.value, 50.0 * u.pA, atol=1e-12 * u.pA))

        with brainstate.environ.context(t=self.dt):
            n.update(x=0.0 * u.pA)  # (100 + 50) pA -> 0.15 mV
        self.assertTrue(u.math.allclose(n.V.value, 0.15 * u.mV, atol=1e-10 * u.mV))

    def test_sic_current_channel_lowers_under_for_loop(self):
        # A SIC current re-deposited each step lowers under for_loop with a stable
        # (1,) carry (the cluster-12 (N,)->() collapse guard).
        n = self._passive()

        def _run_step(k):
            n.add_current_input('sic', jnp.full(n.varshape, 50.0), label='I_SIC')
            with brainstate.environ.context(t=k * self.dt):
                n.update(x=0.0 * u.pA)
            return n.I_sic.value / u.pA

        out = brainstate.transform.for_loop(_run_step, jnp.arange(4))
        self.assertEqual(tuple(out.shape), (4, 1))
        npt.assert_allclose(np.asarray(out).reshape(-1), [50., 50., 50., 50.], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
