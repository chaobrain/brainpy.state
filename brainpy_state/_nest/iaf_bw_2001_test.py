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

            rng = np.random.default_rng(123)
            v_bw = []
            v_ce = []

            for k in range(240):
                events = []
                if rng.random() < 0.1:
                    events.append(('AMPA', 40.0 * u.nS))
                if rng.random() < 0.08:
                    events.append(('GABA', 15.0 * u.nS))

                x_k = (20.0 * math.sin(0.07 * k)) * u.pA

                for i, ev in enumerate(events):
                    receptor, w = ev
                    if receptor == 'AMPA':
                        ce.add_delta_input(f'e_{k}_{i}', w)
                    else:
                        ce.add_delta_input(f'i_{k}_{i}', -w)

                with brainstate.environ.context(t=k * self.dt):
                    bw.update(x=x_k, spike_events=events)
                    ce.update(x=x_k)

                v_bw.append(float((bw.V.value / u.mV)[0]))
                v_ce.append(float((ce.V.value / u.mV)[0]))

                self.assertAlmostEqual(float((bw.s_AMPA.value / u.nS)[0]), float((ce.g_ex.value / u.nS)[0]), delta=5e-6)
                self.assertAlmostEqual(float((bw.s_GABA.value / u.nS)[0]), float((ce.g_in.value / u.nS)[0]), delta=5e-6)

            self.assertTrue(np.max(np.abs(np.asarray(v_bw) - np.asarray(v_ce))) < 6e-6)

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
            k0, k1 = _nmda_jump_constants_ref(0.5, 2.0, 100.0)
            params = {
                'E_L': -70.0,
                'E_ex': 0.0,
                'E_in': -70.0,
                'V_th': -63.0,
                'V_reset': -68.0,
                'C_m': 500.0,
                'g_L': 25.0,
                't_ref': 0.3,
                'tau_AMPA': 2.0,
                'tau_GABA': 5.0,
                'tau_decay_NMDA': 100.0,
                'tau_rise_NMDA': 2.0,
                'alpha': 0.5,
                'conc_Mg2': 1.0,
                'gsl_error_tol': 1e-3,
                'k0': k0,
                'k1': k1,
            }
            ref = {
                'v': -70.0,
                's_ampa': 0.0,
                's_gaba': 0.0,
                's_nmda': 0.0,
                'i_ampa': 0.0,
                'i_gaba': 0.0,
                'i_nmda': 0.0,
                'r': 0,
                'h': dt,
                'i_stim': 0.0,
                's_nmda_pre': 0.0,
                'last_spike_time': -1e7,
                'spike_offset': 0.0,
            }

            x_seq = []
            spike_events_seq = []
            for k in range(80):
                if 5 <= k < 55:
                    x_seq.append(2200.0)
                elif 55 <= k < 65:
                    x_seq.append(-300.0)
                else:
                    x_seq.append(0.0)

                ev = []
                if k % 7 == 1:
                    ev.append(('AMPA', 40.0, 1.0))
                if k % 11 == 3:
                    ev.append(('GABA', 15.0, 1.0))
                if k in (10, 20, 30, 40, 50):
                    ev.append(('NMDA', 25.0, 0.6))
                spike_events_seq.append(ev)

            spk_model = []
            spk_ref = []

            for k, (x_k, ev_k) in enumerate(zip(x_seq, spike_events_seq)):
                ev_model = []
                for rec, w, off in ev_k:
                    if rec == 'NMDA':
                        ev_model.append((rec, w * u.nS, off))
                    else:
                        ev_model.append((rec, w * u.nS))

                spk = self._step(neuron, k, x=x_k * u.pA, spike_events=ev_model)
                spk_model.append(self._is_spike(spk))

                spk_ref.append(_reference_step(ref, params, x_k, ev_k, dt, k * dt))

                self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), ref['v'], delta=7e-6)
                self.assertAlmostEqual(float((neuron.s_AMPA.value / u.nS)[0]), ref['s_ampa'], delta=7e-6)
                self.assertAlmostEqual(float((neuron.s_GABA.value / u.nS)[0]), ref['s_gaba'], delta=7e-6)
                self.assertAlmostEqual(float((neuron.s_NMDA.value / u.nS)[0]), ref['s_nmda'], delta=7e-6)
                self.assertAlmostEqual(float((neuron.I_AMPA.value / u.pA)[0]), ref['i_ampa'], delta=7e-6)
                self.assertAlmostEqual(float((neuron.I_GABA.value / u.pA)[0]), ref['i_gaba'], delta=7e-6)
                self.assertAlmostEqual(float((neuron.I_NMDA.value / u.pA)[0]), ref['i_nmda'], delta=7e-6)
                self.assertEqual(int(neuron.refractory_step_count.value[0]), ref['r'])
                self.assertAlmostEqual(float((neuron.integration_step.value / u.ms)[0]), ref['h'], delta=7e-6)
                self.assertAlmostEqual(float(neuron.s_NMDA_pre.value[0]), ref['s_nmda_pre'], delta=7e-6)
                self.assertAlmostEqual(float(neuron.spike_offset.value[0]), ref['spike_offset'], delta=7e-6)
                self.assertAlmostEqual(float((neuron.last_spike_time.value / u.ms)[0]), ref['last_spike_time'],
                                       delta=7e-6)

            self.assertEqual(spk_model, spk_ref)
            self.assertTrue(any(spk_model))

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

            rng = np.random.default_rng(4321)
            v_base = []
            v_nmda = []

            for k in range(280):
                events = []
                if rng.random() < 0.12:
                    events.append(('AMPA', 40.0 * u.nS))
                if rng.random() < 0.09:
                    events.append(('GABA', 15.0 * u.nS))

                events_nmda = list(events)
                if rng.random() < 0.10:
                    events_nmda.append(('NMDA', 40.0 * u.nS, 0.55))

                with brainstate.environ.context(t=k * self.dt):
                    base.update(spike_events=events)
                    nmda.update(spike_events=events_nmda)

                v_base.append(float((base.V.value / u.mV)[0]))
                v_nmda.append(float((nmda.V.value / u.mV)[0]))

            diff = np.asarray(v_nmda) - np.asarray(v_base)
            self.assertGreater(np.mean(diff[120:]), 0.0)
            self.assertGreater(np.max(diff), 0.05)


if __name__ == '__main__':
    unittest.main()
