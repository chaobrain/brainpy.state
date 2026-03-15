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
import numpy as np
import numpy.testing as npt

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import aeif_cond_beta_multisynapse


def _beta_norm_factor(tau_rise, tau_decay):
    eps = np.finfo(np.float64).eps
    tau_difference = tau_decay - tau_rise
    peak_value = 0.0
    if abs(tau_difference) > eps:
        t_peak = tau_decay * tau_rise * math.log(tau_decay / tau_rise) / tau_difference
        peak_value = math.exp(-t_peak / tau_decay) - math.exp(-t_peak / tau_rise)
    if abs(peak_value) < eps:
        return math.e / tau_decay
    return (1.0 / tau_rise - 1.0 / tau_decay) / peak_value


def _rhs(y, is_refractory, i_stim, p):
    v = y[0]
    w = y[1]
    dg = y[2::2]
    g = y[3::2]

    v_eff = p['V_reset'] if is_refractory else min(v, p['V_peak_rhs'])
    i_syn = float(np.sum(g * (p['E_rev'] - v_eff)))
    i_spike = 0.0 if p['Delta_T'] == 0.0 else p['g_L'] * p['Delta_T'] * math.exp((v_eff - p['V_th']) / p['Delta_T'])
    dv = 0.0 if is_refractory else (
                                       -p['g_L'] * (v_eff - p['E_L']) + i_spike + i_syn - w + p['I_e'] + i_stim
                                   ) / p['C_m']
    dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']

    dftype = brainstate.environ.dftype()
    dy = np.empty_like(y, dtype=dftype)
    dy[0] = dv
    dy[1] = dw
    dy[2::2] = -dg / p['tau_rise']
    dy[3::2] = dg - g / p['tau_decay']
    return dy


def _reference_step(state, p, x_next, w_step, dt_ms):
    min_h = 1e-8
    t = 0.0
    h = max(state['h'], min_h)
    dftype = brainstate.environ.dftype()
    y = np.asarray(state['y'], dtype=dftype)
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
                raise ValueError('Numerical instability in reference aeif_cond_beta_multisynapse.')

            if r > 0:
                y[0] = p['V_reset']
            elif y[0] >= p['V_peak_detect']:
                spike_count += 1
                y[0] = p['V_reset']
                y[1] += p['b']
                r = (p['refr_counts'] + 1) if p['refr_counts'] > 0 else 0
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    if r > 0:
        r -= 1

    y[2::2] += p['g0'] * w_step

    state['y'] = y
    state['r'] = r
    state['h'] = h
    state['i_stim'] = x_next
    return spike_count


def _parse_event_weights(events, n_receptors):
    dftype = brainstate.environ.dftype()
    out = np.zeros(n_receptors, dtype=dftype)
    if events is None:
        return out

    if isinstance(events, dict):
        events = [events]

    for ev in events:
        if isinstance(ev, dict):
            receptor = int(ev.get('receptor_type', ev.get('receptor', 1)))
            weight = ev.get('weight', 0.0)
        else:
            receptor, weight = ev
            receptor = int(receptor)
        out[receptor - 1] += float(np.asarray(u.math.asarray(weight / u.nS), dtype=dftype))
    return out


class TestAEIFCondBetaMultisynapse(unittest.TestCase):
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

    def _step(self, neuron, k, x=0.0 * u.pA, spike_events=None):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_nest_cpp_default_parameters_and_recordables(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = aeif_cond_beta_multisynapse(1)
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
            self.assertEqual(neuron.I_e, 0.0 * u.pA)
            self.assertTrue(np.allclose(neuron.tau_rise, [2.0]))
            self.assertTrue(np.allclose(neuron.tau_decay, [20.0]))
            self.assertTrue(np.allclose(neuron.E_rev, [0.0]))
            self.assertEqual(neuron.n_receptors, 1)
            self.assertEqual(neuron.recordables, ['V_m', 'w', 'g_1'])

            neuron3 = aeif_cond_beta_multisynapse(
                1,
                tau_rise=[0.2, 0.5, 1.0] * u.ms,
                tau_decay=[0.8, 2.0, 4.0] * u.ms,
                E_rev=[0.0, -85.0, 20.0] * u.mV,
            )
            self.assertEqual(neuron3.recordables, ['V_m', 'w', 'g_1', 'g_2', 'g_3'])

    def test_parameter_validation(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(
                    1,
                    E_rev=[0.0, -85.0] * u.mV,
                    tau_rise=[2.0] * u.ms,
                    tau_decay=[20.0] * u.ms,
                )
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(
                    1,
                    E_rev=[0.0] * u.mV,
                    tau_rise=[2.0] * u.ms,
                    tau_decay=[0.0] * u.ms,
                )
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(
                    1,
                    E_rev=[0.0] * u.mV,
                    tau_rise=[2.0] * u.ms,
                    tau_decay=[1.0] * u.ms,
                )
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(1, V_peak=-55.0 * u.mV, V_th=-50.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(1, V_reset=0.0 * u.mV, V_peak=0.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(1, Delta_T=-1.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(1, C_m=0.0 * u.pF)
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(1, t_ref=-0.1 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(1, tau_w=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(1, gsl_error_tol=0.0)
            with self.assertRaises(ValueError):
                aeif_cond_beta_multisynapse(1, V_peak=1500.0 * u.mV, Delta_T=1e-12 * u.mV)

    def test_spike_receptor_routing_and_nonnegative_weight_constraint(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_cond_beta_multisynapse(
                1,
                V_peak=1e6 * u.mV,
                V_th=1e6 * u.mV,
                V_reset=0.0 * u.mV,
                Delta_T=0.0 * u.mV,
                g_L=0.0 * u.nS,
                a=0.0 * u.nS,
                b=0.0 * u.pA,
                I_e=0.0 * u.pA,
                tau_rise=[2.0, 5.0, 10.0] * u.ms,
                tau_decay=[6.0, 20.0, 40.0] * u.ms,
                E_rev=[0.0, 0.0, -85.0] * u.mV,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                w_initializer=braintools.init.Constant(0.0 * u.pA),
            )
            neuron.init_state()

            self._step(
                neuron,
                0,
                spike_events=[
                    (1, 2.0 * u.nS),
                    {'receptor_type': 2, 'weight': 0.5 * u.nS},
                    (3, 1.5 * u.nS),
                ],
            )
            dftype = brainstate.environ.dftype()
            dg = np.asarray(neuron.dg.value, dtype=dftype)[0]
            self.assertAlmostEqual(dg[0], _beta_norm_factor(2.0, 6.0) * 2.0, delta=1e-12)
            self.assertAlmostEqual(dg[1], _beta_norm_factor(5.0, 20.0) * 0.5, delta=1e-12)
            self.assertAlmostEqual(dg[2], _beta_norm_factor(10.0, 40.0) * 1.5, delta=1e-12)

            with self.assertRaises(ValueError):
                self._step(neuron, 1, spike_events=[(1, -1.0 * u.nS)])
            with self.assertRaises(ValueError):
                self._step(neuron, 1, spike_events=[(4, 1.0 * u.nS)])

            neuron.add_delta_input('neg_default', -0.1 * u.nS)
            with self.assertRaises(ValueError):
                self._step(neuron, 2)

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_cond_beta_multisynapse(
                1,
                V_peak=1e6 * u.mV,
                V_th=1e6 * u.mV,
                V_reset=0.0 * u.mV,
                Delta_T=0.0 * u.mV,
                g_L=0.0 * u.nS,
                C_m=100.0 * u.pF,
                a=0.0 * u.nS,
                b=0.0 * u.pA,
                I_e=0.0 * u.pA,
                tau_rise=[2.0] * u.ms,
                tau_decay=[20.0] * u.ms,
                E_rev=[0.0] * u.mV,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                w_initializer=braintools.init.Constant(0.0 * u.pA),
            )
            neuron.init_state()

            self._step(neuron, 0, x=100.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.0 * u.mV, atol=1e-12 * u.mV))

            self._step(neuron, 1, x=0.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.1 * u.mV, atol=1e-11 * u.mV))

    def test_refractoriness_clamping_matches_nest(self):
        for t_ref in (0.0, 0.1):
            with brainstate.environ.context(dt=0.1 * u.ms):
                neuron = aeif_cond_beta_multisynapse(
                    1,
                    t_ref=t_ref * u.ms,
                    V_reset=-111.0 * u.mV,
                    Delta_T=0.0 * u.mV,
                    a=0.0 * u.nS,
                    b=0.0 * u.pA,
                    I_e=1000.0 * u.pA,
                )
                neuron.init_state()

                v_trace = []
                spikes = []
                for k in range(120):
                    with brainstate.environ.context(t=k * 0.1 * u.ms):
                        spk = neuron.update(x=0.0 * u.pA)
                    v_trace.append(float((neuron.V.value / u.mV)[0]))
                    spikes.append(self._is_spike(spk))

                spike_idx = int(np.argmax(spikes))
                self.assertTrue(spikes[spike_idx])
                self.assertAlmostEqual(v_trace[spike_idx], -111.0, delta=1e-12)

                if t_ref == 0.0:
                    self.assertGreater(v_trace[spike_idx + 1], -111.0)
                else:
                    self.assertAlmostEqual(v_trace[spike_idx + 1], -111.0, delta=1e-12)
                    self.assertGreater(v_trace[spike_idx + 2], -111.0)

    def test_w_dynamics_during_refractoriness(self):
        with brainstate.environ.context(dt=1.0 * u.ms):
            V_reset = -111.0
            E_L = -70.0
            t_ref = 100.0
            a = 10.0
            b = 100.0
            tau_w = 1.0

            neuron = aeif_cond_beta_multisynapse(
                1,
                Delta_T=0.0 * u.mV,
                t_ref=t_ref * u.ms,
                I_e=1000.0 * u.pA,
                E_L=E_L * u.mV,
                V_reset=V_reset * u.mV,
                a=a * u.nS,
                b=b * u.pA,
                tau_w=tau_w * u.ms,
                w_initializer=braintools.init.Constant(0.0 * u.pA),
            )
            neuron.init_state()

            v_trace = []
            w_trace = []
            spikes = []
            for k in range(50):
                with brainstate.environ.context(t=k * 1.0 * u.ms):
                    spk = neuron.update(x=0.0 * u.pA)
                v_trace.append(float((neuron.V.value / u.mV)[0]))
                w_trace.append(float((neuron.w.value / u.pA)[0]))
                spikes.append(self._is_spike(spk))

            spike_idx = int(np.argmax(spikes))
            self.assertTrue(spikes[spike_idx])
            self.assertLess(20.0, t_ref)

            w0 = w_trace[spike_idx]
            w1 = w_trace[spike_idx + 20]
            v0 = v_trace[spike_idx]
            v1 = v_trace[spike_idx + 20]
            self.assertAlmostEqual(v0, V_reset, delta=1e-12)
            self.assertAlmostEqual(v1, V_reset, delta=1e-12)

            w_theory = w0 * math.exp(-20.0 / tau_w) + a * (V_reset - E_L) * (1.0 - math.exp(-20.0 / tau_w))
            self.assertAlmostEqual(w1, w_theory, delta=2e-8)

    def test_reference_trace_matches_nest_update_logic(self):
        dftype = brainstate.environ.dftype()
        tau_rise = np.asarray([0.2, 0.5, 2.0, 10.0], dtype=dftype)
        tau_decay = np.asarray([0.5, 1.5, 5.0, 20.0], dtype=dftype)
        E_rev = np.asarray([0.0, 0.0, -85.0, 20.0], dtype=dftype)
        g0 = np.asarray([_beta_norm_factor(tr, td) for tr, td in zip(tau_rise, tau_decay)], dtype=dftype)

        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_cond_beta_multisynapse(
                1,
                V_peak=0.0 * u.mV,
                V_reset=-58.0 * u.mV,
                t_ref=0.3 * u.ms,
                g_L=11.0 * u.nS,
                C_m=200.0 * u.pF,
                E_L=-70.0 * u.mV,
                Delta_T=2.0 * u.mV,
                tau_w=300.0 * u.ms,
                a=3.0 * u.nS,
                b=40.0 * u.pA,
                V_th=-50.0 * u.mV,
                tau_rise=tau_rise * u.ms,
                tau_decay=tau_decay * u.ms,
                E_rev=E_rev * u.mV,
                I_e=1200.0 * u.pA,
                gsl_error_tol=1e-6,
                V_initializer=braintools.init.Constant(-68.0 * u.mV),
                w_initializer=braintools.init.Constant(5.0 * u.pA),
            )
            neuron.init_state()
            neuron.dg.value = np.asarray([[0.20, 0.01, 0.05, 0.03]], dtype=dftype)
            neuron.g.value = np.asarray([[0.10, 0.20, 0.30, 0.00]], dtype=dftype) * u.nS

            n_steps = 80
            x_seq = np.zeros(n_steps, dtype=dftype)
            x_seq[[1, 5, 9, 16, 30, 45]] = np.asarray([25.0, -30.0, 40.0, -10.0, 50.0, -20.0], dtype=dftype)

            step_events = {
                0: [(1, 1.5 * u.nS), (3, 0.8 * u.nS)],
                2: [(2, 1.2 * u.nS)],
                4: [(4, 0.6 * u.nS)],
                7: [(1, 0.4 * u.nS), (2, 0.3 * u.nS), (3, 0.2 * u.nS), (4, 0.5 * u.nS)],
                15: [(3, 1.0 * u.nS)],
                28: [(2, 0.7 * u.nS), (4, 0.9 * u.nS)],
                50: [(1, 0.5 * u.nS)],
            }

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
                'tau_rise': tau_rise,
                'tau_decay': tau_decay,
                'E_rev': E_rev,
                'I_e': 1200.0,
                'atol': 1e-6,
                'refr_counts': int(math.ceil(float((0.3 * u.ms) / self.dt))),
                'g0': g0,
            }
            ref_state = {
                'y': np.asarray([-68.0, 5.0, 0.20, 0.10, 0.01, 0.20, 0.05, 0.30, 0.03, 0.00], dtype=dftype),
                'r': 0,
                'h': float(self.dt / u.ms),
                'i_stim': 0.0,
            }

            spikes_model = []
            spikes_ref = []
            for k in range(n_steps):
                ev = step_events.get(k, None)
                spk = self._step(neuron, k, x=x_seq[k] * u.pA, spike_events=ev)
                spikes_model.append(self._is_spike(spk))

                w_step = _parse_event_weights(ev, neuron.n_receptors)
                n_spk_ref = _reference_step(ref_state, p, x_seq[k], w_step, 0.1)
                spikes_ref.append(n_spk_ref > 0)

                self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), ref_state['y'][0], delta=3e-6)
                self.assertAlmostEqual(float((neuron.w.value / u.pA)[0]), ref_state['y'][1], delta=3e-6)
                npt.assert_allclose(
                    np.asarray(neuron.dg.value, dtype=dftype)[0],
                    ref_state['y'][2::2],
                    atol=3e-6,
                    rtol=0.0,
                )
                npt.assert_allclose(
                    np.asarray(u.math.asarray(neuron.g.value / u.nS), dtype=dftype)[0],
                    ref_state['y'][3::2],
                    atol=3e-6,
                    rtol=0.0,
                )
                self.assertEqual(int(neuron.refractory_step_count.value[0]), ref_state['r'])
                self.assertAlmostEqual(float((neuron.integration_step.value / u.ms)[0]), ref_state['h'], delta=3e-6)

            self.assertEqual(spikes_model, spikes_ref)
            self.assertTrue(any(spikes_model))

    def test_direct_trace_matches_nest_if_available(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        if 'aeif_cond_beta_multisynapse' not in nest.Models():
            self.skipTest('NEST model aeif_cond_beta_multisynapse not available')

        dt_ms = 0.1
        n_steps = 300
        dftype = brainstate.environ.dftype()
        tau_rise = np.asarray([0.2, 2.0, 10.0], dtype=dftype)
        tau_decay = np.asarray([0.5, 5.0, 20.0], dtype=dftype)
        e_rev = np.asarray([0.0, -85.0, 20.0], dtype=dftype)
        delays = np.asarray([2.0, 5.0, 12.0], dtype=dftype)
        weights = np.asarray([1.0, 0.7, 1.2], dtype=dftype)

        params = {
            'V_peak': 1000.0,
            'V_reset': -60.0,
            't_ref': 0.0,
            'g_L': 11.0,
            'C_m': 200.0,
            'E_L': -70.0,
            'Delta_T': 0.0,
            'tau_w': 300.0,
            'a': 0.0,
            'b': 0.0,
            'V_th': 1000.0,
            'tau_rise': list(tau_rise),
            'tau_decay': list(tau_decay),
            'E_rev': list(e_rev),
            'I_e': 0.0,
            'gsl_error_tol': 1e-6,
            'V_m': -70.0,
            'w': 0.0,
        }

        nest.ResetKernel()
        nest.resolution = dt_ms

        sg = nest.Create('spike_generator', params={'spike_times': [1.0]})
        nrn = nest.Create('aeif_cond_beta_multisynapse', params=params)
        for i, (dly, w) in enumerate(zip(delays, weights), start=1):
            nest.Connect(sg, nrn, syn_spec={'delay': float(dly), 'weight': float(w), 'receptor_type': i})

        mm = nest.Create('multimeter', params={
            'record_from': ['V_m', 'w', 'g_1', 'g_2', 'g_3'],
            'interval': dt_ms,
        })
        nest.Connect(mm, nrn)
        nest.Simulate(n_steps * dt_ms)

        events = mm.get('events')
        nest_v = np.asarray(events['V_m'], dtype=dftype)
        nest_w = np.asarray(events['w'], dtype=dftype)
        nest_g1 = np.asarray(events['g_1'], dtype=dftype)
        nest_g2 = np.asarray(events['g_2'], dtype=dftype)
        nest_g3 = np.asarray(events['g_3'], dtype=dftype)
        nest_times = np.asarray(events['times'], dtype=dftype)

        step_events = {}
        for ridx, (dly, w) in enumerate(zip(delays, weights), start=1):
            t_event = 1.0 + dly
            k = int(round((t_event - dt_ms) / dt_ms))
            step_events.setdefault(k, []).append((ridx, w * u.nS))

        with brainstate.environ.context(dt=dt_ms * u.ms):
            neuron = aeif_cond_beta_multisynapse(
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
                tau_rise=tau_rise * u.ms,
                tau_decay=tau_decay * u.ms,
                E_rev=e_rev * u.mV,
                I_e=params['I_e'] * u.pA,
                gsl_error_tol=params['gsl_error_tol'],
                V_initializer=braintools.init.Constant(params['V_m'] * u.mV),
                w_initializer=braintools.init.Constant(params['w'] * u.pA),
            )
            neuron.init_state()

            bp_v = np.empty(n_steps, dtype=dftype)
            bp_w = np.empty(n_steps, dtype=dftype)
            bp_g = np.empty((n_steps, 3), dtype=dftype)

            for k in range(n_steps):
                with brainstate.environ.context(t=(k * dt_ms) * u.ms):
                    neuron.update(x=0.0 * u.pA, spike_events=step_events.get(k, None))
                bp_v[k] = float((neuron.V.value / u.mV)[0])
                bp_w[k] = float((neuron.w.value / u.pA)[0])
                bp_g[k, :] = np.asarray(u.math.asarray(neuron.g.value / u.nS), dtype=dftype)[0]

        bp_indices = np.rint(nest_times / dt_ms).astype(np.int64) - 1
        self.assertTrue(np.all(bp_indices >= 0))
        self.assertTrue(np.all(bp_indices < n_steps))

        npt.assert_allclose(bp_v[bp_indices], nest_v, atol=3e-5, rtol=0.0, err_msg='V_m trace mismatch vs NEST')
        npt.assert_allclose(bp_w[bp_indices], nest_w, atol=3e-5, rtol=0.0, err_msg='w trace mismatch vs NEST')
        npt.assert_allclose(bp_g[bp_indices, 0], nest_g1, atol=3e-5, rtol=0.0, err_msg='g_1 trace mismatch vs NEST')
        npt.assert_allclose(bp_g[bp_indices, 1], nest_g2, atol=3e-5, rtol=0.0, err_msg='g_2 trace mismatch vs NEST')
        npt.assert_allclose(bp_g[bp_indices, 2], nest_g3, atol=3e-5, rtol=0.0, err_msg='g_3 trace mismatch vs NEST')


if __name__ == '__main__':
    unittest.main()
