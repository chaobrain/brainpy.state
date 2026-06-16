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
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy.state import aeif_psc_delta_clopath


def _rhs(y, is_refractory, is_clamped, i_stim, p):
    v, w, z, v_th, u_plus, u_minus, u_bar = y

    if is_refractory or is_clamped:
        v_eff = p['V_clamp'] if is_clamped else p['V_reset']
    else:
        v_eff = min(v, p['V_peak_rhs'])

    i_spike = 0.0 if p['Delta_T'] == 0.0 else p['g_L'] * p['Delta_T'] * math.exp((v_eff - v_th) / p['Delta_T'])
    dv = 0.0 if (is_refractory or is_clamped) else (
                                                       -p['g_L'] * (v_eff - p['E_L']) + i_spike - w + z + p[
                                                       'I_e'] + i_stim
                                                   ) / p['C_m']

    dw = 0.0 if is_clamped else (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
    dz = -z / p['tau_z']
    dv_th = -(v_th - p['V_th_rest']) / p['tau_V_th']

    du_plus = (-u_plus + v_eff) / p['tau_u_bar_plus']
    du_minus = (-u_minus + v_eff) / p['tau_u_bar_minus']
    du_bar = (-u_bar + u_minus) / p['tau_u_bar_bar']

    dftype = brainstate.environ.dftype()
    return np.asarray([dv, dw, dz, dv_th, du_plus, du_minus, du_bar], dtype=dftype)


def _reference_step(state, p, x_next, delta_step, dt_ms):
    min_h = 1e-8
    t = 0.0
    h = max(state['h'], min_h)
    dftype = brainstate.environ.dftype()
    y = np.asarray([
        state['v'],
        state['w'],
        state['z'],
        state['v_th'],
        state['u_plus'],
        state['u_minus'],
        state['u_bar'],
    ], dtype=dftype)

    r = int(state['r'])
    clamp = int(state['clamp'])
    pending_delta = float(delta_step)

    spike_count = 0
    iters = 0

    while t < dt_ms and iters < 100000:
        iters += 1
        h = max(min_h, min(h, dt_ms - t))

        is_refractory = r > 0
        is_clamped = clamp > 0

        k1 = _rhs(y, is_refractory, is_clamped, state['i_stim'], p)
        k2 = _rhs(y + h * (1.0 / 4.0) * k1, is_refractory, is_clamped, state['i_stim'], p)
        k3 = _rhs(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0), is_refractory, is_clamped, state['i_stim'], p)
        k4 = _rhs(
            y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
            is_refractory,
            is_clamped,
            state['i_stim'],
            p,
        )
        k5 = _rhs(
            y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
            is_refractory,
            is_clamped,
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
            is_clamped,
            state['i_stim'],
            p,
        )

        y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
        y5 = y + h * (
            16.0 * k1 / 135.0
            + 6656.0 * k3 / 12825.0
            + 28561.0 * k4 / 56430.0
            - 9.0 * k5 / 50.0
            + 2.0 * k6 / 55.0
        )

        err = float(np.max(np.abs(y5 - y4)))
        atol = p['atol']

        if err <= atol or h <= min_h:
            y = y5
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)

            if y[0] < -1e3 or y[1] < -1e6 or y[1] > 1e6:
                raise ValueError('Numerical instability in reference aeif_psc_delta_clopath.')

            if r == 0 and clamp == 0:
                y[0] += pending_delta
            pending_delta = 0.0

            v_peak_detect = p['V_peak_detect_fixed'] if p['Delta_T'] > 0.0 else y[3]

            if y[0] >= v_peak_detect and clamp == 0:
                spike_count += 1

                y[0] = p['V_clamp']
                y[1] += p['b']
                y[2] = p['I_sp']
                y[3] = p['V_th_max']

                clamp = (p['clamp_counts'] + 1) if p['clamp_counts'] > 0 else 0
            elif clamp == 1:
                y[0] = p['V_reset']
                clamp = 0
                r = (p['refr_counts'] + 1) if p['refr_counts'] > 0 else 0

            if r > 0:
                y[0] = p['V_reset']
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    # write_clopath_history logic
    state['delay_plus'][state['delay_idx']] = y[4]
    state['delay_minus'][state['delay_idx']] = y[5]

    state['delay_idx'] = (state['delay_idx'] + 1) % state['delay_steps']

    delayed_plus = state['delay_plus'][state['delay_idx']]
    delayed_minus = state['delay_minus'][state['delay_idx']]

    if p['A_LTD_const']:
        state['last_ltd_dw'] = p['A_LTD'] * (delayed_minus - p['theta_minus']) if delayed_minus > p[
            'theta_minus'] else 0.0
    else:
        state['last_ltd_dw'] = (
            p['A_LTD'] * y[6] * y[6] * (delayed_minus - p['theta_minus']) / p['u_ref_squared']
            if delayed_minus > p['theta_minus']
            else 0.0
        )

    state['last_ltp_dw'] = (
        p['A_LTP'] * (y[0] - p['theta_plus']) * (delayed_plus - p['theta_minus']) * dt_ms
        if (y[0] > p['theta_plus'] and delayed_plus > p['theta_minus'])
        else 0.0
    )

    if clamp > 0:
        clamp -= 1
    if r > 0:
        r -= 1

    state['v'] = y[0]
    state['w'] = y[1]
    state['z'] = y[2]
    state['v_th'] = y[3]
    state['u_plus'] = y[4]
    state['u_minus'] = y[5]
    state['u_bar'] = y[6]

    state['r'] = r
    state['clamp'] = clamp
    state['h'] = h
    state['i_stim'] = x_next

    return spike_count


class TestAEIFPscDeltaClopath(unittest.TestCase):
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

    def _step(self, neuron, k, x=0.0 * u.pA, dV_values=None):
        if dV_values is not None:
            for i, val in enumerate(dV_values):
                neuron.add_delta_input(f'delta_{k}_{i}', val * u.mV)
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_nest_cpp_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = aeif_psc_delta_clopath(1)

            self.assertEqual(neuron.V_peak, 33.0 * u.mV)
            self.assertEqual(neuron.V_reset, -60.0 * u.mV)
            self.assertEqual(neuron.t_ref, 0.0 * u.ms)
            self.assertEqual(neuron.g_L, 30.0 * u.nS)
            self.assertEqual(neuron.C_m, 281.0 * u.pF)
            self.assertEqual(neuron.E_L, -70.6 * u.mV)
            self.assertEqual(neuron.Delta_T, 2.0 * u.mV)

            self.assertEqual(neuron.tau_w, 144.0 * u.ms)
            self.assertEqual(neuron.tau_z, 40.0 * u.ms)
            self.assertEqual(neuron.tau_V_th, 50.0 * u.ms)
            self.assertEqual(neuron.V_th_max, 30.4 * u.mV)
            self.assertEqual(neuron.V_th_rest, -50.4 * u.mV)

            self.assertEqual(neuron.tau_u_bar_plus, 7.0 * u.ms)
            self.assertEqual(neuron.tau_u_bar_minus, 10.0 * u.ms)
            self.assertEqual(neuron.tau_u_bar_bar, 500.0 * u.ms)

            self.assertEqual(neuron.a, 4.0 * u.nS)
            self.assertEqual(neuron.b, 80.5 * u.pA)
            self.assertEqual(neuron.I_sp, 400.0 * u.pA)
            self.assertEqual(neuron.I_e, 0.0 * u.pA)

            self.assertEqual(float(neuron.A_LTD), 14.0e-5)
            self.assertEqual(float(neuron.A_LTP), 8.0e-5)
            self.assertEqual(neuron.theta_plus, -45.3 * u.mV)
            self.assertEqual(neuron.theta_minus, -70.6 * u.mV)
            self.assertTrue(neuron.A_LTD_const)

            self.assertEqual(neuron.delay_u_bars, 5.0 * u.ms)
            self.assertEqual(float(neuron.u_ref_squared), 60.0)

            self.assertEqual(neuron.t_clamp, 2.0 * u.ms)
            self.assertEqual(neuron.V_clamp, 33.0 * u.mV)

    def test_parameter_validation(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, V_reset=10.0 * u.mV, V_peak=10.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, Delta_T=-1.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, V_th_max=-60.0 * u.mV, V_th_rest=-50.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, V_peak=-60.0 * u.mV, V_th_rest=-50.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, C_m=0.0 * u.pF)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, t_ref=-0.1 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, t_clamp=-0.1 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, tau_w=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, tau_z=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, tau_V_th=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, tau_u_bar_plus=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, tau_u_bar_minus=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, tau_u_bar_bar=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, u_ref_squared=0.0)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, gsl_error_tol=0.0)
            with self.assertRaises(ValueError):
                aeif_psc_delta_clopath(1, V_peak=1500.0 * u.mV, Delta_T=1e-12 * u.mV)

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_psc_delta_clopath(
                1,
                V_peak=1e6 * u.mV,
                V_reset=0.0 * u.mV,
                Delta_T=0.0 * u.mV,
                g_L=0.0 * u.nS,
                a=0.0 * u.nS,
                b=0.0 * u.pA,
                I_sp=0.0 * u.pA,
                I_e=0.0 * u.pA,
                V_th_rest=1e6 * u.mV,
                V_th_max=1e6 * u.mV,
                t_clamp=0.0 * u.ms,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                w_initializer=braintools.init.Constant(0.0 * u.pA),
                z_initializer=braintools.init.Constant(0.0 * u.pA),
                V_th_initializer=braintools.init.Constant(1e6 * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, x=100.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.0 * u.mV))

            self._step(neuron, 1, x=0.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.03558718861209964 * u.mV, atol=1e-12 * u.mV))

    def test_reference_trace_matches_nest_step_logic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_psc_delta_clopath(
                1,
                V_peak=5.0 * u.mV,
                V_reset=-58.0 * u.mV,
                t_ref=0.3 * u.ms,
                g_L=12.0 * u.nS,
                C_m=200.0 * u.pF,
                E_L=-70.0 * u.mV,
                Delta_T=2.0 * u.mV,
                tau_w=300.0 * u.ms,
                tau_z=40.0 * u.ms,
                tau_V_th=50.0 * u.ms,
                V_th_max=18.0 * u.mV,
                V_th_rest=-50.0 * u.mV,
                tau_u_bar_plus=7.0 * u.ms,
                tau_u_bar_minus=10.0 * u.ms,
                tau_u_bar_bar=500.0 * u.ms,
                a=3.0 * u.nS,
                b=40.0 * u.pA,
                I_sp=250.0 * u.pA,
                I_e=1500.0 * u.pA,
                A_LTD=14.0e-5,
                A_LTP=8.0e-5,
                theta_plus=-45.3 * u.mV,
                theta_minus=-70.6 * u.mV,
                A_LTD_const=True,
                delay_u_bars=0.2 * u.ms,
                u_ref_squared=60.0,
                gsl_error_tol=1e-6,
                t_clamp=0.2 * u.ms,
                V_clamp=25.0 * u.mV,
                V_initializer=braintools.init.Constant(-68.0 * u.mV),
                w_initializer=braintools.init.Constant(5.0 * u.pA),
                z_initializer=braintools.init.Constant(0.0 * u.pA),
                V_th_initializer=braintools.init.Constant(-50.0 * u.mV),
                u_bar_plus_initializer=braintools.init.Constant(-70.0 * u.mV),
                u_bar_minus_initializer=braintools.init.Constant(-70.0 * u.mV),
                u_bar_bar_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            x_seq = [0.0, 20.0, 0.0, -30.0, 0.0, 40.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0] + [0.0] * 48
            dV_seq = [0.0, 5.0, -2.0, 0.0, 4.0, -3.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2.5] + [0.0] * 48

            dt_ms = 0.1
            delay_steps = int(np.rint(0.2 / dt_ms)) + 1

            p = {
                'V_peak_rhs': 5.0,
                'V_peak_detect_fixed': 5.0,
                'V_reset': -58.0,
                'g_L': 12.0,
                'C_m': 200.0,
                'E_L': -70.0,
                'Delta_T': 2.0,
                'tau_w': 300.0,
                'tau_z': 40.0,
                'tau_V_th': 50.0,
                'V_th_max': 18.0,
                'V_th_rest': -50.0,
                'tau_u_bar_plus': 7.0,
                'tau_u_bar_minus': 10.0,
                'tau_u_bar_bar': 500.0,
                'a': 3.0,
                'b': 40.0,
                'I_sp': 250.0,
                'I_e': 1500.0,
                'A_LTD': 14.0e-5,
                'A_LTP': 8.0e-5,
                'theta_plus': -45.3,
                'theta_minus': -70.6,
                'A_LTD_const': True,
                'u_ref_squared': 60.0,
                'V_clamp': 25.0,
                'atol': 1e-6,
                'refr_counts': int(math.ceil(0.3 / dt_ms)),
                'clamp_counts': int(math.ceil(0.2 / dt_ms)),
            }

            dftype = brainstate.environ.dftype()
            ditype = brainstate.environ.ditype()
            ref_state = {
                'v': -68.0,
                'w': 5.0,
                'z': 0.0,
                'v_th': -50.0,
                'u_plus': -70.0,
                'u_minus': -70.0,
                'u_bar': -70.0,
                'r': 0,
                'clamp': 0,
                'h': 0.1,
                'i_stim': 0.0,
                'delay_steps': delay_steps,
                'delay_idx': 0,
                'delay_plus': np.zeros(delay_steps, dtype=dftype),
                'delay_minus': np.zeros(delay_steps, dtype=dftype),
                'last_ltd_dw': 0.0,
                'last_ltp_dw': 0.0,
            }

            n_steps = len(x_seq)
            x_arr = jnp.array(x_seq, dtype=dftype)
            dV_arr = jnp.array(dV_seq, dtype=dftype)

            # --- Model loop via for_loop (JIT-compiled) ---
            def _model_step(k):
                neuron.add_delta_input('_dV', dV_arr[k] * u.mV)
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=x_arr[k] * u.pA)
                return (
                    spk,
                    neuron.V.value / u.mV,
                    neuron.w.value / u.pA,
                    neuron.z.value / u.pA,
                    neuron.V_th.value / u.mV,
                    neuron.u_bar_plus.value / u.mV,
                    neuron.u_bar_minus.value / u.mV,
                    neuron.u_bar_bar.value / u.mV,
                    neuron.refractory_step_count.value,
                    neuron.clamp_step_count.value,
                    neuron.integration_step.value / u.ms,
                    neuron.delayed_u_bars_idx.value,
                    neuron.delayed_u_bar_plus_buffer.value[:, 0],
                    neuron.delayed_u_bar_minus_buffer.value[:, 0],
                )

            results = brainstate.transform.for_loop(_model_step, jnp.arange(n_steps))
            spk_arr = np.asarray(results[0].flatten(), dtype=dftype)
            v_arr = np.asarray(results[1].flatten(), dtype=dftype)
            w_arr_m = np.asarray(results[2].flatten(), dtype=dftype)
            z_arr_m = np.asarray(results[3].flatten(), dtype=dftype)
            vth_arr = np.asarray(results[4].flatten(), dtype=dftype)
            uplus_arr = np.asarray(results[5].flatten(), dtype=dftype)
            uminus_arr = np.asarray(results[6].flatten(), dtype=dftype)
            ubar_arr = np.asarray(results[7].flatten(), dtype=dftype)
            refr_arr = np.asarray(results[8].flatten(), dtype=ditype)
            clamp_arr = np.asarray(results[9].flatten(), dtype=ditype)
            h_arr = np.asarray(results[10].flatten(), dtype=dftype)
            delay_idx_arr = np.asarray(results[11], dtype=ditype)           # (n_steps,)
            delay_plus_arr = np.asarray(results[12], dtype=dftype)          # (n_steps, delay_steps)
            delay_minus_arr = np.asarray(results[13], dtype=dftype)         # (n_steps, delay_steps)

            # --- Reference loop (pure Python) ---
            spikes_ref = []
            ref_v = np.empty(n_steps, dtype=dftype)
            ref_w = np.empty(n_steps, dtype=dftype)
            ref_z = np.empty(n_steps, dtype=dftype)
            ref_vth = np.empty(n_steps, dtype=dftype)
            ref_uplus = np.empty(n_steps, dtype=dftype)
            ref_uminus = np.empty(n_steps, dtype=dftype)
            ref_ubar = np.empty(n_steps, dtype=dftype)
            ref_r = np.empty(n_steps, dtype=ditype)
            ref_clamp = np.empty(n_steps, dtype=ditype)
            ref_h = np.empty(n_steps, dtype=dftype)
            ref_delay_idx = np.empty(n_steps, dtype=ditype)
            ref_delay_plus = np.empty((n_steps, delay_steps), dtype=dftype)
            ref_delay_minus = np.empty((n_steps, delay_steps), dtype=dftype)

            for k, (x_i, dV_i) in enumerate(zip(x_seq, dV_seq)):
                n_spk_ref = _reference_step(ref_state, p, x_i, dV_i, dt_ms)
                spikes_ref.append(n_spk_ref > 0)
                ref_v[k] = ref_state['v']
                ref_w[k] = ref_state['w']
                ref_z[k] = ref_state['z']
                ref_vth[k] = ref_state['v_th']
                ref_uplus[k] = ref_state['u_plus']
                ref_uminus[k] = ref_state['u_minus']
                ref_ubar[k] = ref_state['u_bar']
                ref_r[k] = ref_state['r']
                ref_clamp[k] = ref_state['clamp']
                ref_h[k] = ref_state['h']
                ref_delay_idx[k] = ref_state['delay_idx']
                ref_delay_plus[k] = ref_state['delay_plus'].copy()
                ref_delay_minus[k] = ref_state['delay_minus'].copy()

            # --- Batch assertions ---
            spikes_model = list(spk_arr > 0.5)
            saw_clamp = bool(np.any(clamp_arr > 0))
            saw_refractory = bool(np.any(refr_arr > 0))

            npt.assert_allclose(v_arr, ref_v, atol=3e-6, rtol=0.0, err_msg='V mismatch')
            npt.assert_allclose(w_arr_m, ref_w, atol=3e-6, rtol=0.0, err_msg='w mismatch')
            npt.assert_allclose(z_arr_m, ref_z, atol=3e-6, rtol=0.0, err_msg='z mismatch')
            npt.assert_allclose(vth_arr, ref_vth, atol=3e-6, rtol=0.0, err_msg='V_th mismatch')
            npt.assert_allclose(uplus_arr, ref_uplus, atol=3e-6, rtol=0.0, err_msg='u_bar_plus mismatch')
            npt.assert_allclose(uminus_arr, ref_uminus, atol=3e-6, rtol=0.0, err_msg='u_bar_minus mismatch')
            npt.assert_allclose(ubar_arr, ref_ubar, atol=3e-6, rtol=0.0, err_msg='u_bar_bar mismatch')
            np.testing.assert_array_equal(refr_arr, ref_r)
            np.testing.assert_array_equal(clamp_arr, ref_clamp)
            npt.assert_allclose(h_arr, ref_h, atol=3e-6, rtol=0.0, err_msg='integration_step mismatch')
            np.testing.assert_array_equal(delay_idx_arr, ref_delay_idx)
            npt.assert_allclose(delay_plus_arr, ref_delay_plus, atol=3e-6, rtol=0.0,
                                err_msg='delay_plus buffer mismatch')
            npt.assert_allclose(delay_minus_arr, ref_delay_minus, atol=3e-6, rtol=0.0,
                                err_msg='delay_minus buffer mismatch')

            self.assertEqual(spikes_model, spikes_ref)
            self.assertTrue(any(spikes_model))
            self.assertTrue(saw_clamp)
            self.assertTrue(saw_refractory)

    def test_direct_trace_matches_nest_if_available(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        if 'aeif_psc_delta_clopath' not in nest.Models():
            self.skipTest('NEST model aeif_psc_delta_clopath not available')

        dt_ms = 0.1
        n_steps = 250

        params = {
            'V_peak': 33.0,
            'V_reset': -60.0,
            't_ref': 0.2,
            'g_L': 30.0,
            'C_m': 281.0,
            'E_L': -70.6,
            'Delta_T': 2.0,
            'tau_w': 144.0,
            'tau_z': 40.0,
            'tau_V_th': 50.0,
            'V_th_max': 30.4,
            'V_th_rest': -49.0,
            'tau_u_bar_plus': 7.0,
            'tau_u_bar_minus': 10.0,
            'tau_u_bar_bar': 500.0,
            'a': 4.0,
            'b': 80.5,
            'I_sp': 400.0,
            'I_e': 100.0,
            'A_LTD': 14.0e-5,
            'A_LTP': 8.0e-5,
            'theta_plus': -45.3,
            'theta_minus': -70.6,
            'A_LTD_const': True,
            'delay_u_bars': 5.0,
            'u_ref_squared': 60.0,
            'gsl_error_tol': 1e-6,
            't_clamp': 2.0,
            'V_clamp': 33.0,
            'V_m': -67.2,
            'w': 5.0,
            'u_bar_plus': -70.6,
            'u_bar_minus': -70.6,
            'u_bar_bar': -70.6,
        }

        nest.ResetKernel()
        nest.resolution = dt_ms

        nrn = nest.Create('aeif_psc_delta_clopath', params=params)
        mm = nest.Create('multimeter', params={
            'record_from': ['V_m', 'w', 'z', 'V_th', 'u_bar_plus', 'u_bar_minus', 'u_bar_bar'],
            'interval': dt_ms,
        })
        nest.Connect(mm, nrn)
        nest.Simulate(n_steps * dt_ms)

        events = mm.get('events')
        dftype = brainstate.environ.dftype()
        nest_v = np.asarray(events['V_m'], dtype=dftype)
        nest_w = np.asarray(events['w'], dtype=dftype)
        nest_z = np.asarray(events['z'], dtype=dftype)
        nest_v_th = np.asarray(events['V_th'], dtype=dftype)
        nest_u_plus = np.asarray(events['u_bar_plus'], dtype=dftype)
        nest_u_minus = np.asarray(events['u_bar_minus'], dtype=dftype)
        nest_u_bar = np.asarray(events['u_bar_bar'], dtype=dftype)
        nest_times = np.asarray(events['times'], dtype=dftype)

        with brainstate.environ.context(dt=dt_ms * u.ms):
            neuron = aeif_psc_delta_clopath(
                1,
                V_peak=params['V_peak'] * u.mV,
                V_reset=params['V_reset'] * u.mV,
                t_ref=params['t_ref'] * u.ms,
                g_L=params['g_L'] * u.nS,
                C_m=params['C_m'] * u.pF,
                E_L=params['E_L'] * u.mV,
                Delta_T=params['Delta_T'] * u.mV,
                tau_w=params['tau_w'] * u.ms,
                tau_z=params['tau_z'] * u.ms,
                tau_V_th=params['tau_V_th'] * u.ms,
                V_th_max=params['V_th_max'] * u.mV,
                V_th_rest=params['V_th_rest'] * u.mV,
                tau_u_bar_plus=params['tau_u_bar_plus'] * u.ms,
                tau_u_bar_minus=params['tau_u_bar_minus'] * u.ms,
                tau_u_bar_bar=params['tau_u_bar_bar'] * u.ms,
                a=params['a'] * u.nS,
                b=params['b'] * u.pA,
                I_sp=params['I_sp'] * u.pA,
                I_e=params['I_e'] * u.pA,
                A_LTD=params['A_LTD'],
                A_LTP=params['A_LTP'],
                theta_plus=params['theta_plus'] * u.mV,
                theta_minus=params['theta_minus'] * u.mV,
                A_LTD_const=params['A_LTD_const'],
                delay_u_bars=params['delay_u_bars'] * u.ms,
                u_ref_squared=params['u_ref_squared'],
                gsl_error_tol=params['gsl_error_tol'],
                t_clamp=params['t_clamp'] * u.ms,
                V_clamp=params['V_clamp'] * u.mV,
                V_initializer=braintools.init.Constant(params['V_m'] * u.mV),
                w_initializer=braintools.init.Constant(params['w'] * u.pA),
            )
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * (dt_ms * u.ms)):
                    neuron.update(x=0.0 * u.pA)
                return (
                    neuron.V.value / u.mV,
                    neuron.w.value / u.pA,
                    neuron.z.value / u.pA,
                    neuron.V_th.value / u.mV,
                    neuron.u_bar_plus.value / u.mV,
                    neuron.u_bar_minus.value / u.mV,
                    neuron.u_bar_bar.value / u.mV,
                )

            results = brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))
            bp_v = np.asarray(results[0].flatten(), dtype=dftype)
            bp_w = np.asarray(results[1].flatten(), dtype=dftype)
            bp_z = np.asarray(results[2].flatten(), dtype=dftype)
            bp_v_th = np.asarray(results[3].flatten(), dtype=dftype)
            bp_u_plus = np.asarray(results[4].flatten(), dtype=dftype)
            bp_u_minus = np.asarray(results[5].flatten(), dtype=dftype)
            bp_u_bar = np.asarray(results[6].flatten(), dtype=dftype)

        bp_indices = np.rint(nest_times / dt_ms).astype(np.int64) - 1
        self.assertTrue(np.all(bp_indices >= 0))
        self.assertTrue(np.all(bp_indices < n_steps))

        npt.assert_allclose(bp_v[bp_indices], nest_v, atol=2e-4, rtol=0.0, err_msg='V_m trace mismatch vs NEST')
        npt.assert_allclose(bp_w[bp_indices], nest_w, atol=2e-4, rtol=0.0, err_msg='w trace mismatch vs NEST')
        npt.assert_allclose(bp_z[bp_indices], nest_z, atol=2e-4, rtol=0.0, err_msg='z trace mismatch vs NEST')
        npt.assert_allclose(bp_v_th[bp_indices], nest_v_th, atol=2e-4, rtol=0.0, err_msg='V_th trace mismatch vs NEST')
        npt.assert_allclose(bp_u_plus[bp_indices], nest_u_plus, atol=2e-4, rtol=0.0,
                            err_msg='u_bar_plus trace mismatch vs NEST')
        npt.assert_allclose(bp_u_minus[bp_indices], nest_u_minus, atol=2e-4, rtol=0.0,
                            err_msg='u_bar_minus trace mismatch vs NEST')
        npt.assert_allclose(bp_u_bar[bp_indices], nest_u_bar, atol=2e-4, rtol=0.0,
                            err_msg='u_bar_bar trace mismatch vs NEST')


if __name__ == '__main__':
    unittest.main()
