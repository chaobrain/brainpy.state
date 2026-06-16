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
import jax.scipy as jsp
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import iaf_bw_2001, iaf_cond_exp


def _nmda_jump_constants_ref(alpha, tau_rise, tau_decay):
    alpha_tau = alpha * tau_rise
    tau_ratio = tau_rise / tau_decay
    k1 = math.expm1(-alpha_tau)

    a = 1.0 - tau_ratio
    x = alpha_tau
    dftype = brainstate.environ.dftype()
    lower_gamma = float(
        jsp.special.gammainc(jnp.asarray(a, dtype=dftype), jnp.asarray(x, dtype=dftype))
        * jnp.exp(jsp.special.gammaln(jnp.asarray(a, dtype=dftype)))
    )
    k0 = (alpha_tau ** tau_ratio) * lower_gamma
    return k0, k1


def _nmda_currents(v, s_ampa, s_gaba, s_nmda, p):
    i_ampa = (v - p['E_ex']) * s_ampa
    i_gaba = (v - p['E_in']) * s_gaba
    denom = 1.0 + p['conc_Mg2'] * math.exp(-0.062 * v) / 3.57
    i_nmda = (v - p['E_ex']) / denom * s_nmda
    return i_ampa, i_gaba, i_nmda


def _dynamics_ref(y, i_stim, p):
    v, s_ampa, s_gaba, s_nmda = y
    i_ampa, i_gaba, i_nmda = _nmda_currents(v, s_ampa, s_gaba, s_nmda, p)
    i_syn = i_ampa + i_gaba + i_nmda

    dv = (-p['g_L'] * (v - p['E_L']) - i_syn + i_stim) / p['C_m']
    ds_ampa = -s_ampa / p['tau_AMPA']
    ds_gaba = -s_gaba / p['tau_GABA']
    ds_nmda = -s_nmda / p['tau_decay_NMDA']
    dftype = brainstate.environ.dftype()
    return np.asarray([dv, ds_ampa, ds_gaba, ds_nmda], dtype=dftype)


def _rkf45_ref_step(y0, i_stim, dt, h0, p, atol):
    min_h = 1e-8
    t = 0.0
    h = max(h0, min_h)
    dftype = brainstate.environ.dftype()
    y = np.asarray(y0, dtype=dftype)

    while t < dt:
        h = max(min_h, min(h, dt - t))

        k1 = _dynamics_ref(y, i_stim, p)
        k2 = _dynamics_ref(y + h * (1.0 / 4.0) * k1, i_stim, p)
        k3 = _dynamics_ref(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0), i_stim, p)
        k4 = _dynamics_ref(y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0), i_stim, p)
        k5 = _dynamics_ref(y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0), i_stim,
                           p)
        k6 = _dynamics_ref(
            y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0),
            i_stim, p)

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

    i_ampa, i_gaba, i_nmda = _nmda_currents(y[0], y[1], y[2], y[3], p)
    return y, h, i_ampa, i_gaba, i_nmda


def _reference_step(state, p, x_next, step_events, dt, t_step):
    dftype = brainstate.environ.dftype()
    y, h, i_ampa, i_gaba, i_nmda = _rkf45_ref_step(
        np.asarray([state['v'], state['s_ampa'], state['s_gaba'], state['s_nmda']], dtype=dftype),
        state['i_stim'],
        dt,
        state['h'],
        p,
        p['gsl_error_tol'],
    )

    s_ampa, s_gaba, s_nmda = y[1], y[2], y[3]
    for receptor, weight, offset in step_events:
        if receptor == 'AMPA':
            s_ampa += weight
        elif receptor == 'GABA':
            s_gaba += weight
        else:
            s_nmda += weight * offset

    v = y[0]
    if state['r'] > 0:
        v = p['V_reset']
        r = state['r'] - 1
        spike = False
        spike_offset = 0.0
    else:
        if v >= p['V_th']:
            spike = True
            v = p['V_reset']
            r = int(math.ceil(p['t_ref'] / dt))

            t_spike = t_step + dt
            s_pre = state['s_nmda_pre'] * math.exp(-(t_spike - state['last_spike_time']) / p['tau_decay_NMDA'])
            spike_offset = p['k0'] + p['k1'] * s_pre
            s_pre = s_pre + spike_offset
            last_spike_time = t_spike
        else:
            spike = False
            r = 0
            spike_offset = 0.0
            s_pre = state['s_nmda_pre']
            last_spike_time = state['last_spike_time']

    if state['r'] > 0:
        s_pre = state['s_nmda_pre']
        last_spike_time = state['last_spike_time']

    state['v'] = v
    state['s_ampa'] = s_ampa
    state['s_gaba'] = s_gaba
    state['s_nmda'] = s_nmda
    state['i_ampa'] = i_ampa
    state['i_gaba'] = i_gaba
    state['i_nmda'] = i_nmda
    state['r'] = r
    state['h'] = h
    state['i_stim'] = x_next
    state['s_nmda_pre'] = s_pre
    state['last_spike_time'] = last_spike_time
    state['spike_offset'] = spike_offset

    return spike


class TestIAFBW2001(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        dftype = brainstate.environ.dftype()
        return bool(np.asarray(u.math.asarray(spk), dtype=dftype).reshape(-1)[0] > 0.0)

    def _step(self, neuron, k, x=0.0 * u.pA, spike_events=None):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_nest_cpp_default_parameters_and_metadata(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = iaf_bw_2001(1)

            self.assertEqual(neuron.E_L, -70.0 * u.mV)
            self.assertEqual(neuron.E_ex, 0.0 * u.mV)
            self.assertEqual(neuron.E_in, -70.0 * u.mV)
            self.assertEqual(neuron.V_th, -55.0 * u.mV)
            self.assertEqual(neuron.V_reset, -60.0 * u.mV)
            self.assertEqual(neuron.C_m, 500.0 * u.pF)
            self.assertEqual(neuron.g_L, 25.0 * u.nS)
            self.assertEqual(neuron.t_ref, 2.0 * u.ms)
            self.assertEqual(neuron.tau_AMPA, 2.0 * u.ms)
            self.assertEqual(neuron.tau_GABA, 5.0 * u.ms)
            self.assertEqual(neuron.tau_decay_NMDA, 100.0 * u.ms)
            self.assertEqual(neuron.tau_rise_NMDA, 2.0 * u.ms)
            self.assertEqual(neuron.alpha, 0.5 / u.ms)
            self.assertEqual(neuron.conc_Mg2, 1.0 * u.mM)
            self.assertEqual(neuron.gsl_error_tol, 1e-3)

            self.assertEqual(neuron.receptor_types, {'AMPA': 1, 'GABA': 2, 'NMDA': 3})
            self.assertEqual(
                neuron.recordables,
                ['V_m', 's_AMPA', 's_GABA', 's_NMDA', 'I_NMDA', 'I_AMPA', 'I_GABA'],
            )

    def test_parameter_validation(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, C_m=0.0 * u.pF)
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, t_ref=-0.1 * u.ms)
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, tau_AMPA=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, tau_GABA=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, tau_decay_NMDA=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, tau_rise_NMDA=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, alpha=0.0 / u.ms)
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, conc_Mg2=0.0 * u.mM)
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, gsl_error_tol=0.0)
            with self.assertRaises(ValueError):
                iaf_bw_2001(1, V_reset=-55.0 * u.mV, V_th=-55.0 * u.mV)

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_bw_2001(
                1,
                E_L=0.0 * u.mV,
                E_ex=0.0 * u.mV,
                E_in=0.0 * u.mV,
                g_L=0.0 * u.nS,
                V_th=1e9 * u.mV,
                V_reset=0.0 * u.mV,
                C_m=500.0 * u.pF,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, x=100.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.0 * u.mV))

            self._step(neuron, 1, x=0.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.02 * u.mV))

    def test_receptor_routing_and_illegal_nmda_sender(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_bw_2001(
                1,
                E_L=0.0 * u.mV,
                E_ex=0.0 * u.mV,
                E_in=0.0 * u.mV,
                g_L=0.0 * u.nS,
                V_th=1e9 * u.mV,
                V_reset=0.0 * u.mV,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            self._step(
                neuron,
                0,
                spike_events=[
                    ('AMPA', 5.0 * u.nS),
                    ('GABA', 3.0 * u.nS),
                    ('NMDA', 8.0 * u.nS, 0.25),
                ],
            )
            self.assertAlmostEqual(float((neuron.s_AMPA.value / u.nS)[0]), 5.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.s_GABA.value / u.nS)[0]), 3.0, delta=1e-12)
            self.assertAlmostEqual(float((neuron.s_NMDA.value / u.nS)[0]), 2.0, delta=1e-12)

            neuron.add_delta_input('a0', 1.5 * u.nS, label='AMPA')
            neuron.add_delta_input('g0', 2.5 * u.nS, label='GABA')
            neuron.add_delta_input('n0', 0.75 * u.nS, label='NMDA')
            self._step(neuron, 1)
            expected_ampa = 5.0 * math.exp(-0.1 / 2.0) + 1.5
            expected_gaba = 3.0 * math.exp(-0.1 / 5.0) + 2.5
            expected_nmda = 2.0 * math.exp(-0.1 / 100.0) + 0.75
            self.assertAlmostEqual(float((neuron.s_AMPA.value / u.nS)[0]), expected_ampa, delta=1e-9)
            self.assertAlmostEqual(float((neuron.s_GABA.value / u.nS)[0]), expected_gaba, delta=1e-9)
            self.assertAlmostEqual(float((neuron.s_NMDA.value / u.nS)[0]), expected_nmda, delta=1e-9)

            with self.assertRaises(ValueError):
                self._step(
                    neuron,
                    2,
                    spike_events=[
                        {
                            'receptor_type': 'NMDA',
                            'weight': 1.0 * u.nS,
                            'sender_model': 'iaf_cond_exp',
                        }
                    ],
                )

    def test_matches_iaf_cond_exp_without_nmda(self):
        with brainstate.environ.context(dt=self.dt):
            bw = iaf_bw_2001(
                1,
                E_L=-70.0 * u.mV,
                E_ex=0.0 * u.mV,
                E_in=-70.0 * u.mV,
                V_th=1000.0 * u.mV,
                V_reset=-55.0 * u.mV,
                C_m=500.0 * u.pF,
                g_L=25.0 * u.nS,
                t_ref=0.0 * u.ms,
                tau_AMPA=2.0 * u.ms,
                tau_GABA=5.0 * u.ms,
                tau_decay_NMDA=100.0 * u.ms,
                tau_rise_NMDA=2.0 * u.ms,
                alpha=0.5 / u.ms,
                conc_Mg2=1.0 * u.mM,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            ce = iaf_cond_exp(
                1,
                E_L=-70.0 * u.mV,
                E_ex=0.0 * u.mV,
                E_in=-70.0 * u.mV,
                V_th=1000.0 * u.mV,
                V_reset=-55.0 * u.mV,
                C_m=500.0 * u.pF,
                g_L=25.0 * u.nS,
                t_ref=0.0 * u.ms,
                tau_syn_ex=2.0 * u.ms,
                tau_syn_in=5.0 * u.ms,
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            bw.init_state()
            ce.init_state()

            # Pre-compute all per-step inputs so the loop body has a fixed JAX graph.
            rng = np.random.default_rng(123)
            r = rng.random((240, 2))  # r[:,0]=AMPA draws, r[:,1]=GABA draws
            ampa_w = jnp.where(jnp.array(r[:, 0] < 0.1), 40.0, 0.0)   # nS per step
            gaba_w = jnp.where(jnp.array(r[:, 1] < 0.08), 15.0, 0.0)  # nS per step
            x_vals = jnp.array([20.0 * math.sin(0.07 * k) for k in range(240)])  # pA per step

            def _body(k):
                # Use fixed-key delta inputs so the loop body traces identically each step.
                bw.add_delta_input('ampa', ampa_w[k] * u.nS, label='AMPA')
                bw.add_delta_input('gaba', gaba_w[k] * u.nS, label='GABA')
                ce.add_delta_input('ampa_ce', ampa_w[k] * u.nS, label='w_ex')
                ce.add_delta_input('gaba_ce', gaba_w[k] * u.nS, label='w_in')
                with brainstate.environ.context(t=k * self.dt):
                    bw.update(x=x_vals[k] * u.pA)
                    ce.update(x=x_vals[k] * u.pA)
                return (
                    bw.V.value / u.mV,
                    ce.V.value / u.mV,
                    bw.s_AMPA.value / u.nS,
                    ce.g_ex.value / u.nS,
                    bw.s_GABA.value / u.nS,
                    ce.g_in.value / u.nS,
                )

            results = brainstate.transform.for_loop(_body, jnp.arange(240))
            v_bw = np.asarray(results[0][:, 0])
            v_ce = np.asarray(results[1][:, 0])
            s_ampa_trace = np.asarray(results[2][:, 0])
            g_ex_trace = np.asarray(results[3][:, 0])
            s_gaba_trace = np.asarray(results[4][:, 0])
            g_in_trace = np.asarray(results[5][:, 0])

            for k in range(240):
                self.assertAlmostEqual(s_ampa_trace[k], g_ex_trace[k], delta=5e-6)
                self.assertAlmostEqual(s_gaba_trace[k], g_in_trace[k], delta=5e-6)

            self.assertTrue(np.max(np.abs(v_bw - v_ce)) < 6e-6)

    def test_reference_trace_matches_nest_update_logic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_bw_2001(
                1,
                E_L=-70.0 * u.mV,
                E_ex=0.0 * u.mV,
                E_in=-70.0 * u.mV,
                V_th=-63.0 * u.mV,
                V_reset=-68.0 * u.mV,
                C_m=500.0 * u.pF,
                g_L=25.0 * u.nS,
                t_ref=0.3 * u.ms,
                tau_AMPA=2.0 * u.ms,
                tau_GABA=5.0 * u.ms,
                tau_decay_NMDA=100.0 * u.ms,
                tau_rise_NMDA=2.0 * u.ms,
                alpha=0.5 / u.ms,
                conc_Mg2=1.0 * u.mM,
                gsl_error_tol=1e-3,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            dt = 0.1
            N = 80
            k0, k1 = _nmda_jump_constants_ref(0.5, 2.0, 100.0)
            params = {
                'E_L': -70.0, 'E_ex': 0.0, 'E_in': -70.0,
                'V_th': -63.0, 'V_reset': -68.0, 'C_m': 500.0, 'g_L': 25.0,
                't_ref': 0.3, 'tau_AMPA': 2.0, 'tau_GABA': 5.0,
                'tau_decay_NMDA': 100.0, 'tau_rise_NMDA': 2.0,
                'alpha': 0.5, 'conc_Mg2': 1.0, 'gsl_error_tol': 1e-3,
                'k0': k0, 'k1': k1,
            }

            # Pre-compute all per-step inputs as JAX arrays.
            nmda_steps = {10, 20, 30, 40, 50}
            x_list, ampa_list, gaba_list, nmda_list = [], [], [], []
            spike_events_seq = []
            for k in range(N):
                if 5 <= k < 55:
                    x_list.append(2200.0)
                elif 55 <= k < 65:
                    x_list.append(-300.0)
                else:
                    x_list.append(0.0)
                ampa_list.append(40.0 if k % 7 == 1 else 0.0)
                gaba_list.append(15.0 if k % 11 == 3 else 0.0)
                nmda_list.append(25.0 * 0.6 if k in nmda_steps else 0.0)

                ev = []
                if k % 7 == 1:
                    ev.append(('AMPA', 40.0, 1.0))
                if k % 11 == 3:
                    ev.append(('GABA', 15.0, 1.0))
                if k in nmda_steps:
                    ev.append(('NMDA', 25.0, 0.6))
                spike_events_seq.append(ev)

            x_arr = jnp.array(x_list)
            ampa_arr = jnp.array(ampa_list)
            gaba_arr = jnp.array(gaba_list)
            nmda_arr = jnp.array(nmda_list)

            # Run model with for_loop (single JAX compilation).
            def _body(k):
                neuron.add_delta_input('ampa', ampa_arr[k] * u.nS, label='AMPA')
                neuron.add_delta_input('gaba', gaba_arr[k] * u.nS, label='GABA')
                neuron.add_delta_input('nmda', nmda_arr[k] * u.nS, label='NMDA')
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=x_arr[k] * u.pA)
                return (
                    neuron.V.value / u.mV,
                    neuron.s_AMPA.value / u.nS,
                    neuron.s_GABA.value / u.nS,
                    neuron.s_NMDA.value / u.nS,
                    neuron.I_AMPA.value / u.pA,
                    neuron.I_GABA.value / u.pA,
                    neuron.I_NMDA.value / u.pA,
                    neuron.refractory_step_count.value,
                    neuron.integration_step.value / u.ms,
                    neuron.s_NMDA_pre.value,
                    neuron.spike_offset.value,
                    neuron.last_spike_time.value / u.ms,
                    spk,
                )

            results = brainstate.transform.for_loop(_body, jnp.arange(N))

            # Extract model traces (shape: (N, 1) -> (N,)).
            v_m = np.asarray(results[0][:, 0])
            s_ampa_m = np.asarray(results[1][:, 0])
            s_gaba_m = np.asarray(results[2][:, 0])
            s_nmda_m = np.asarray(results[3][:, 0])
            i_ampa_m = np.asarray(results[4][:, 0])
            i_gaba_m = np.asarray(results[5][:, 0])
            i_nmda_m = np.asarray(results[6][:, 0])
            r_m = np.asarray(results[7][:, 0], dtype=int)
            h_m = np.asarray(results[8][:, 0])
            s_nmda_pre_m = np.asarray(results[9][:, 0])
            spike_offset_m = np.asarray(results[10][:, 0])
            last_spike_t_m = np.asarray(results[11][:, 0])
            spk_m = np.asarray(results[12][:, 0])

            # Run reference implementation (pure Python, fast).
            ref = {
                'v': -70.0, 's_ampa': 0.0, 's_gaba': 0.0, 's_nmda': 0.0,
                'i_ampa': 0.0, 'i_gaba': 0.0, 'i_nmda': 0.0,
                'r': 0, 'h': dt, 'i_stim': 0.0,
                's_nmda_pre': 0.0, 'last_spike_time': -1e7, 'spike_offset': 0.0,
            }
            ref_v = np.empty(N)
            ref_s_ampa = np.empty(N)
            ref_s_gaba = np.empty(N)
            ref_s_nmda = np.empty(N)
            ref_i_ampa = np.empty(N)
            ref_i_gaba = np.empty(N)
            ref_i_nmda = np.empty(N)
            ref_r = np.empty(N, dtype=int)
            ref_h = np.empty(N)
            ref_s_nmda_pre = np.empty(N)
            ref_spike_offset = np.empty(N)
            ref_last_spike_t = np.empty(N)
            ref_spk = np.empty(N, dtype=bool)

            for k in range(N):
                ref_spk[k] = _reference_step(
                    ref, params, float(x_arr[k]), spike_events_seq[k], dt, k * dt
                )
                ref_v[k] = ref['v']
                ref_s_ampa[k] = ref['s_ampa']
                ref_s_gaba[k] = ref['s_gaba']
                ref_s_nmda[k] = ref['s_nmda']
                ref_i_ampa[k] = ref['i_ampa']
                ref_i_gaba[k] = ref['i_gaba']
                ref_i_nmda[k] = ref['i_nmda']
                ref_r[k] = ref['r']
                ref_h[k] = ref['h']
                ref_s_nmda_pre[k] = ref['s_nmda_pre']
                ref_spike_offset[k] = ref['spike_offset']
                ref_last_spike_t[k] = ref['last_spike_time']

            # Compare all traces at once.
            tol = 7e-6
            np.testing.assert_allclose(v_m, ref_v, atol=tol, err_msg='V mismatch')
            np.testing.assert_allclose(s_ampa_m, ref_s_ampa, atol=tol, err_msg='s_AMPA mismatch')
            np.testing.assert_allclose(s_gaba_m, ref_s_gaba, atol=tol, err_msg='s_GABA mismatch')
            np.testing.assert_allclose(s_nmda_m, ref_s_nmda, atol=tol, err_msg='s_NMDA mismatch')
            np.testing.assert_allclose(i_ampa_m, ref_i_ampa, atol=tol, err_msg='I_AMPA mismatch')
            np.testing.assert_allclose(i_gaba_m, ref_i_gaba, atol=tol, err_msg='I_GABA mismatch')
            np.testing.assert_allclose(i_nmda_m, ref_i_nmda, atol=tol, err_msg='I_NMDA mismatch')
            np.testing.assert_array_equal(r_m, ref_r, err_msg='refractory count mismatch')
            np.testing.assert_allclose(h_m, ref_h, atol=tol, err_msg='integration step mismatch')
            np.testing.assert_allclose(s_nmda_pre_m, ref_s_nmda_pre, atol=tol, err_msg='s_NMDA_pre mismatch')
            np.testing.assert_allclose(spike_offset_m, ref_spike_offset, atol=tol, err_msg='spike_offset mismatch')
            np.testing.assert_allclose(last_spike_t_m, ref_last_spike_t, atol=tol, err_msg='last_spike_time mismatch')

            spk_model_bool = [bool(s > 0.0) for s in spk_m]
            spk_ref_bool = [bool(s) for s in ref_spk]
            self.assertEqual(spk_model_bool, spk_ref_bool)
            self.assertTrue(any(spk_model_bool))

    def test_nmda_increases_voltage_vs_no_nmda(self):
        with brainstate.environ.context(dt=self.dt):
            base = iaf_bw_2001(
                1,
                V_th=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                t_ref=0.0 * u.ms,
            )
            nmda = iaf_bw_2001(
                1,
                V_th=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                t_ref=0.0 * u.ms,
            )
            base.init_state()
            nmda.init_state()

            # Pre-compute all per-step inputs (same RNG sequence as the original loop).
            rng = np.random.default_rng(4321)
            r = rng.random((280, 3))  # r[:,0]=AMPA, r[:,1]=GABA, r[:,2]=NMDA draws
            ampa_w = jnp.where(jnp.array(r[:, 0] < 0.12), 40.0, 0.0)       # nS
            gaba_w = jnp.where(jnp.array(r[:, 1] < 0.09), 15.0, 0.0)       # nS
            # NMDA: weight * offset = 40 * 0.55 = 22 nS (already scaled)
            nmda_w = jnp.where(jnp.array(r[:, 2] < 0.10), 40.0 * 0.55, 0.0)  # nS

            def _body(k):
                base.add_delta_input('ampa', ampa_w[k] * u.nS, label='AMPA')
                base.add_delta_input('gaba', gaba_w[k] * u.nS, label='GABA')
                nmda.add_delta_input('ampa', ampa_w[k] * u.nS, label='AMPA')
                nmda.add_delta_input('gaba', gaba_w[k] * u.nS, label='GABA')
                nmda.add_delta_input('nmda', nmda_w[k] * u.nS, label='NMDA')
                with brainstate.environ.context(t=k * self.dt):
                    base.update()
                    nmda.update()
                return (base.V.value / u.mV, nmda.V.value / u.mV)

            results = brainstate.transform.for_loop(_body, jnp.arange(280))
            v_base = np.asarray(results[0][:, 0])
            v_nmda_arr = np.asarray(results[1][:, 0])

            diff = v_nmda_arr - v_base
            self.assertGreater(np.mean(diff[120:]), 0.0)
            self.assertGreater(np.max(diff), 0.05)


if __name__ == '__main__':
    unittest.main()
