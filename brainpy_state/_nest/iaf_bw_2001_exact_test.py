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

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import iaf_bw_2001_exact, iaf_cond_exp


def _nmda_currents_ref(v, s_ampa, s_gaba, s_nmda_sum, p):
    i_ampa = (v - p['E_ex']) * s_ampa
    i_gaba = (v - p['E_in']) * s_gaba
    denom = 1.0 + p['conc_Mg2'] * math.exp(-0.062 * v) / 3.57
    i_nmda = (v - p['E_ex']) / denom * s_nmda_sum
    return i_ampa, i_gaba, i_nmda


def _dynamics_ref(y, i_stim, p, nmda_weights):
    n_nmda = nmda_weights.shape[0]
    v = y[0]
    s_ampa = y[1]
    s_gaba = y[2]
    x_nmda = y[3:3 + n_nmda]
    s_nmda = y[3 + n_nmda:]
    s_nmda_sum = float(np.dot(s_nmda, nmda_weights)) if n_nmda > 0 else 0.0

    i_ampa, i_gaba, i_nmda = _nmda_currents_ref(v, s_ampa, s_gaba, s_nmda_sum, p)
    i_syn = i_ampa + i_gaba + i_nmda

    dftype = brainstate.environ.dftype()
    dy = np.zeros_like(y, dtype=dftype)
    dy[0] = (-p['g_L'] * (v - p['E_L']) - i_syn + i_stim) / p['C_m']
    dy[1] = -s_ampa / p['tau_AMPA']
    dy[2] = -s_gaba / p['tau_GABA']
    if n_nmda > 0:
        dy[3:3 + n_nmda] = -x_nmda / p['tau_rise_NMDA']
        dy[3 + n_nmda:] = -s_nmda / p['tau_decay_NMDA'] + p['alpha'] * x_nmda * (1.0 - s_nmda)
    return dy


def _rkf45_ref_step(y0, i_stim, dt, h0, p, nmda_weights, atol):
    min_h = 1e-8
    t = 0.0
    h = max(h0, min_h)
    dftype = brainstate.environ.dftype()
    y = np.asarray(y0, dtype=dftype)

    while t < dt:
        h = max(min_h, min(h, dt - t))

        k1 = _dynamics_ref(y, i_stim, p, nmda_weights)
        k2 = _dynamics_ref(y + h * (1.0 / 4.0) * k1, i_stim, p, nmda_weights)
        k3 = _dynamics_ref(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0), i_stim, p, nmda_weights)
        k4 = _dynamics_ref(
            y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
            i_stim,
            p,
            nmda_weights,
        )
        k5 = _dynamics_ref(
            y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
            i_stim,
            p,
            nmda_weights,
        )
        k6 = _dynamics_ref(
            y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0),
            i_stim,
            p,
            nmda_weights,
        )

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

    n_nmda = nmda_weights.shape[0]
    s_nmda_sum = float(np.dot(y[3 + n_nmda:], nmda_weights)) if n_nmda > 0 else 0.0
    i_ampa, i_gaba, i_nmda = _nmda_currents_ref(y[0], y[1], y[2], s_nmda_sum, p)
    return y, h, i_ampa, i_gaba, i_nmda, s_nmda_sum


def _reference_step(state, p, x_next, events, dt, t_step):
    dftype = brainstate.environ.dftype()
    y0 = np.concatenate([
        np.asarray([state['v'], state['s_ampa'], state['s_gaba']], dtype=dftype),
        state['x_nmda'].astype(np.float64),
        state['s_nmda'].astype(np.float64),
    ])
    y, h, i_ampa, i_gaba, i_nmda, s_nmda_sum = _rkf45_ref_step(
        y0=y0,
        i_stim=state['i_stim'],
        dt=dt,
        h0=state['h'],
        p=p,
        nmda_weights=state['nmda_weights'],
        atol=p['gsl_error_tol'],
    )

    ds_ampa = 0.0
    ds_gaba = 0.0
    dx_nmda = np.zeros_like(state['x_nmda'])
    for ev in events:
        receptor = ev['receptor']
        weight = ev['weight']
        mult = ev.get('multiplicity', 1.0)
        if receptor == 'AMPA':
            ds_ampa += weight * mult
        elif receptor == 'GABA':
            ds_gaba += weight * mult
        else:
            port = ev.get('port', 0)
            idx = state['port_to_idx'][port]
            if weight != state['nmda_weights'][idx]:
                raise ValueError('Reference NMDA weight mismatch for port.')
            dx_nmda[idx] += mult

    n_nmda = state['nmda_weights'].shape[0]
    v = y[0]
    s_ampa = y[1] + ds_ampa
    s_gaba = y[2] + ds_gaba
    x_nmda = y[3:3 + n_nmda] + dx_nmda
    s_nmda = y[3 + n_nmda:]

    if state['r'] > 0:
        v_for_spike = p['V_reset']
        v = p['V_reset']
        r = state['r'] - 1
        spike = False
        last_spike = state['last_spike_time']
    else:
        v_for_spike = v
        if v >= p['V_th']:
            spike = True
            v = p['V_reset']
            r = int(math.ceil(p['t_ref'] / dt))
            last_spike = t_step + dt
        else:
            spike = False
            r = 0
            last_spike = state['last_spike_time']

    state['v_for_spike'] = v_for_spike
    state['v'] = v
    state['s_ampa'] = s_ampa
    state['s_gaba'] = s_gaba
    state['x_nmda'] = x_nmda
    state['s_nmda'] = s_nmda
    state['s_nmda_sum'] = s_nmda_sum
    state['i_ampa'] = i_ampa
    state['i_gaba'] = i_gaba
    state['i_nmda'] = i_nmda
    state['r'] = r
    state['h'] = h
    state['i_stim'] = x_next
    state['last_spike_time'] = last_spike
    return spike


class TestIAFBW2001Exact(unittest.TestCase):
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
        neuron = iaf_bw_2001_exact(1)
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
        self.assertEqual(neuron.tau_rise_NMDA, 2.0 * u.ms)
        self.assertEqual(neuron.tau_decay_NMDA, 100.0 * u.ms)
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
            iaf_bw_2001_exact(1, C_m=0.0 * u.pF)
        with self.assertRaises(ValueError):
            iaf_bw_2001_exact(1, t_ref=-0.1 * u.ms)
        with self.assertRaises(ValueError):
            iaf_bw_2001_exact(1, tau_AMPA=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_bw_2001_exact(1, tau_GABA=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_bw_2001_exact(1, tau_rise_NMDA=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_bw_2001_exact(1, tau_decay_NMDA=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_bw_2001_exact(1, alpha=0.0 / u.ms)
        with self.assertRaises(ValueError):
            iaf_bw_2001_exact(1, conc_Mg2=0.0 * u.mM)
        with self.assertRaises(ValueError):
            iaf_bw_2001_exact(1, gsl_error_tol=0.0)
        with self.assertRaises(ValueError):
            iaf_bw_2001_exact(1, V_reset=-55.0 * u.mV, V_th=-55.0 * u.mV)

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_bw_2001_exact(
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

    def test_matches_iaf_cond_exp_without_nmda(self):
        with brainstate.environ.context(dt=self.dt):
            bw = iaf_bw_2001_exact(
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
                tau_rise_NMDA=2.0 * u.ms,
                tau_decay_NMDA=100.0 * u.ms,
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
            for k in range(260):
                events = []
                if rng.random() < 0.1:
                    events.append(('AMPA', 40.0 * u.nS))
                if rng.random() < 0.08:
                    events.append(('GABA', 15.0 * u.nS))
                x_k = (20.0 * math.sin(0.07 * k)) * u.pA

                for i, (rec, w) in enumerate(events):
                    if rec == 'AMPA':
                        ce.add_delta_input(f'e_{k}_{i}', w)
                    else:
                        ce.add_delta_input(f'i_{k}_{i}', -w)

                with brainstate.environ.context(t=k * self.dt):
                    bw.update(x=x_k, spike_events=events)
                    ce.update(x=x_k)

                v_bw.append(float((bw.V.value / u.mV)[0]))
                v_ce.append(float((ce.V.value / u.mV)[0]))
                self.assertAlmostEqual(float((bw.s_AMPA.value / u.nS)[0]), float((ce.g_ex.value / u.nS)[0]), delta=6e-6)
                self.assertAlmostEqual(float((bw.s_GABA.value / u.nS)[0]), float((ce.g_in.value / u.nS)[0]), delta=6e-6)

            self.assertTrue(np.max(np.abs(np.asarray(v_bw) - np.asarray(v_ce))) < 7e-6)

    def test_reference_trace_matches_nest_update_logic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_bw_2001_exact(
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
                tau_rise_NMDA=2.0 * u.ms,
                tau_decay_NMDA=100.0 * u.ms,
                alpha=0.5 / u.ms,
                conc_Mg2=1.0 * u.mM,
                gsl_error_tol=1e-3,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            dt = 0.1
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
                'tau_rise_NMDA': 2.0,
                'tau_decay_NMDA': 100.0,
                'alpha': 0.5,
                'conc_Mg2': 1.0,
                'gsl_error_tol': 1e-3,
            }
            dftype = brainstate.environ.dftype()
            ref = {
                'v': -70.0,
                's_ampa': 0.0,
                's_gaba': 0.0,
                'x_nmda': np.zeros((2,), dtype=dftype),
                's_nmda': np.zeros((2,), dtype=dftype),
                's_nmda_sum': 0.0,
                'nmda_weights': np.asarray([25.0, 12.0], dtype=dftype),
                'port_to_idx': {'p0': 0, 'p1': 1},
                'i_ampa': 0.0,
                'i_gaba': 0.0,
                'i_nmda': 0.0,
                'r': 0,
                'h': dt,
                'i_stim': 0.0,
                'last_spike_time': -1e7,
            }

            x_seq = []
            spike_events_float = []
            spike_events_model = []
            for k in range(90):
                if 5 <= k < 55:
                    x_seq.append(2200.0)
                elif 55 <= k < 65:
                    x_seq.append(-300.0)
                else:
                    x_seq.append(0.0)

                ev_f = []
                ev_m = []
                if k == 0:
                    ev_f.extend([
                        {'receptor': 'NMDA', 'weight': 25.0, 'port': 'p0', 'multiplicity': 0.0},
                        {'receptor': 'NMDA', 'weight': 12.0, 'port': 'p1', 'multiplicity': 0.0},
                    ])
                    ev_m.extend([
                        {'receptor_type': 'NMDA', 'weight': 25.0 * u.nS, 'port': 'p0', 'multiplicity': 0.0},
                        {'receptor_type': 'NMDA', 'weight': 12.0 * u.nS, 'port': 'p1', 'multiplicity': 0.0},
                    ])
                if k % 7 == 1:
                    ev_f.append({'receptor': 'AMPA', 'weight': 40.0, 'multiplicity': 1.0})
                    ev_m.append(('AMPA', 40.0 * u.nS))
                if k % 11 == 3:
                    ev_f.append({'receptor': 'GABA', 'weight': 15.0, 'multiplicity': 1.0})
                    ev_m.append(('GABA', 15.0 * u.nS))
                if k in (10, 20, 30, 40, 50):
                    ev_f.append({'receptor': 'NMDA', 'weight': 25.0, 'port': 'p0', 'multiplicity': 1.0})
                    ev_m.append({'receptor_type': 'NMDA', 'weight': 25.0 * u.nS, 'port': 'p0', 'multiplicity': 1.0})
                if k in (15, 35, 55, 75):
                    ev_f.append({'receptor': 'NMDA', 'weight': 12.0, 'port': 'p1', 'multiplicity': 2.0})
                    ev_m.append({'receptor_type': 'NMDA', 'weight': 12.0 * u.nS, 'port': 'p1', 'multiplicity': 2.0})

                spike_events_float.append(ev_f)
                spike_events_model.append(ev_m)

            spk_model = []
            spk_ref = []
            for k, (x_k, ev_f, ev_m) in enumerate(zip(x_seq, spike_events_float, spike_events_model)):
                spk = self._step(neuron, k, x=x_k * u.pA, spike_events=ev_m)
                spk_model.append(self._is_spike(spk))
                spk_ref.append(_reference_step(ref, params, x_k, ev_f, dt, k * dt))

                self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), ref['v'], delta=1e-5)
                self.assertAlmostEqual(float((neuron.s_AMPA.value / u.nS)[0]), ref['s_ampa'], delta=1e-5)
                self.assertAlmostEqual(float((neuron.s_GABA.value / u.nS)[0]), ref['s_gaba'], delta=1e-5)
                self.assertAlmostEqual(float((neuron.s_NMDA.value / u.nS)[0]), ref['s_nmda_sum'], delta=1e-5)
                self.assertAlmostEqual(float((neuron.I_AMPA.value / u.pA)[0]), ref['i_ampa'], delta=1e-5)
                self.assertAlmostEqual(float((neuron.I_GABA.value / u.pA)[0]), ref['i_gaba'], delta=1e-5)
                self.assertAlmostEqual(float((neuron.I_NMDA.value / u.pA)[0]), ref['i_nmda'], delta=1e-5)
                self.assertEqual(int(neuron.refractory_step_count.value[0]), ref['r'])
                self.assertAlmostEqual(float((neuron.integration_step.value / u.ms)[0]), ref['h'], delta=1e-5)
                self.assertAlmostEqual(float((neuron.last_spike_time.value / u.ms)[0]), ref['last_spike_time'],
                                       delta=1e-5)
                self.assertTrue(
                    np.allclose(np.asarray(neuron.x_NMDA.value[0], dtype=dftype), ref['x_nmda'], atol=1e-5))
                self.assertTrue(
                    np.allclose(np.asarray(neuron.s_NMDA_components.value[0], dtype=dftype), ref['s_nmda'],
                                atol=1e-5))

            self.assertEqual(spk_model, spk_ref)
            self.assertTrue(any(spk_model))

    def test_nmda_increases_voltage_vs_no_nmda(self):
        with brainstate.environ.context(dt=self.dt):
            base = iaf_bw_2001_exact(
                1,
                V_th=1000.0 * u.mV,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
                t_ref=0.0 * u.ms,
            )
            nmda = iaf_bw_2001_exact(
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
            for k in range(320):
                events = []
                if rng.random() < 0.12:
                    events.append(('AMPA', 40.0 * u.nS))
                if rng.random() < 0.09:
                    events.append(('GABA', 15.0 * u.nS))

                events_nmda = list(events)
                if k == 0:
                    events_nmda.append(
                        {'receptor_type': 'NMDA', 'weight': 40.0 * u.nS, 'port': 'p0', 'multiplicity': 0.0})
                if rng.random() < 0.10:
                    events_nmda.append(
                        {'receptor_type': 'NMDA', 'weight': 40.0 * u.nS, 'port': 'p0', 'multiplicity': 1.0})

                with brainstate.environ.context(t=k * self.dt):
                    base.update(spike_events=events)
                    nmda.update(spike_events=events_nmda)

                v_base.append(float((base.V.value / u.mV)[0]))
                v_nmda.append(float((nmda.V.value / u.mV)[0]))

            diff = np.asarray(v_nmda) - np.asarray(v_base)
            self.assertGreater(np.mean(diff[120:]), 0.0)
            self.assertGreater(np.max(diff), 0.05)

    def test_nmda_port_constraints_match_nest_semantics(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_bw_2001_exact(1, V_th=1000.0 * u.mV, V_initializer=braintools.init.Constant(-70.0 * u.mV))
            neuron.init_state()

            self._step(
                neuron,
                0,
                spike_events=[{'receptor_type': 'NMDA', 'weight': 5.0 * u.nS, 'port': 'nmda0', 'multiplicity': 0.0}],
            )
            self._step(
                neuron,
                1,
                spike_events=[{'receptor_type': 'NMDA', 'weight': 5.0 * u.nS, 'port': 'nmda0', 'multiplicity': 1.0}],
            )
            with self.assertRaises(ValueError):
                self._step(
                    neuron,
                    2,
                    spike_events=[
                        {'receptor_type': 'NMDA', 'weight': 6.0 * u.nS, 'port': 'nmda0', 'multiplicity': 1.0}],
                )

            neuron2 = iaf_bw_2001_exact(1, V_th=1000.0 * u.mV, V_initializer=braintools.init.Constant(-70.0 * u.mV))
            neuron2.init_state()
            self._step(neuron2, 0, spike_events=[])
            with self.assertRaises(ValueError):
                self._step(
                    neuron2,
                    1,
                    spike_events=[
                        {'receptor_type': 'NMDA', 'weight': 1.0 * u.nS, 'port': 'late_port', 'multiplicity': 1.0}],
                )


if __name__ == '__main__':
    unittest.main()
