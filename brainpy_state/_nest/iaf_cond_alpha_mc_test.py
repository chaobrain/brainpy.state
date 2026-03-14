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
import saiunit as u
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import iaf_cond_alpha_mc

SOMA = 0
PROX = 1
DIST = 2

V_M = 0
DG_EX = 1
G_EX = 2
DG_IN = 3
G_IN = 4

NCOMP = 3
NSTATE_COMP = 5


def _idx(comp, elem):
    return comp * NSTATE_COMP + elem


def _dynamics_ref(y, is_refractory, i_stim, p):
    dftype = brainstate.environ.dftype()
    f = np.zeros_like(y, dtype=dftype)

    for n in range(NCOMP):
        if n == SOMA:
            v_eff = p['V_reset'] if is_refractory else min(y[_idx(SOMA, V_M)], p['V_th'])
            i_conn = p['g_sp'] * (v_eff - y[_idx(PROX, V_M)])
        elif n == PROX:
            v_eff = y[_idx(PROX, V_M)]
            i_conn = p['g_sp'] * (v_eff - y[_idx(SOMA, V_M)]) + p['g_pd'] * (v_eff - y[_idx(DIST, V_M)])
        else:
            v_eff = y[_idx(DIST, V_M)]
            i_conn = p['g_pd'] * (v_eff - y[_idx(PROX, V_M)])

        i_syn_ex = y[_idx(n, G_EX)] * (v_eff - p['E_ex'][n])
        i_syn_in = y[_idx(n, G_IN)] * (v_eff - p['E_in'][n])
        i_leak = p['g_L'][n] * (v_eff - p['E_L'][n])

        f[_idx(n, V_M)] = 0.0 if is_refractory else (
                                                        -i_leak - i_syn_ex - i_syn_in - i_conn + i_stim[n] + p['I_e'][n]
                                                    ) / p['C_m'][n]

        f[_idx(n, DG_EX)] = -y[_idx(n, DG_EX)] / p['tau_syn_ex'][n]
        f[_idx(n, G_EX)] = y[_idx(n, DG_EX)] - y[_idx(n, G_EX)] / p['tau_syn_ex'][n]

        f[_idx(n, DG_IN)] = -y[_idx(n, DG_IN)] / p['tau_syn_in'][n]
        f[_idx(n, G_IN)] = y[_idx(n, DG_IN)] - y[_idx(n, G_IN)] / p['tau_syn_in'][n]

    return f


def _rkf45_ref_step(y0, is_refractory, i_stim, dt, h0, p, atol=1e-3):
    min_h = 1e-8
    t = 0.0
    h = max(h0, min_h)
    dftype = brainstate.environ.dftype()
    y = np.asarray(y0, dtype=dftype)

    while t < dt:
        h = max(min_h, min(h, dt - t))

        k1 = _dynamics_ref(y, is_refractory, i_stim, p)
        k2 = _dynamics_ref(y + h * (1.0 / 4.0) * k1, is_refractory, i_stim, p)
        k3 = _dynamics_ref(
            y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0),
            is_refractory,
            i_stim,
            p,
        )
        k4 = _dynamics_ref(
            y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
            is_refractory,
            i_stim,
            p,
        )
        k5 = _dynamics_ref(
            y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
            is_refractory,
            i_stim,
            p,
        )
        k6 = _dynamics_ref(
            y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0),
            is_refractory,
            i_stim,
            p,
        )

        y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
        y5 = y + h * (
            16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0
        )
        err = float(np.max(np.abs(y5 - y4)))

        if err <= atol or h <= min_h:
            y = y5
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    return y, h


def _reference_step(state, p, x_next, w_step, dt):
    y, h = _rkf45_ref_step(state['y'], state['r'] > 0, state['i_stim'], dt, state['h'], p)

    for c in range(NCOMP):
        y[_idx(c, DG_EX)] += math.e / p['tau_syn_ex'][c] * w_step[2 * c]
        y[_idx(c, DG_IN)] += math.e / p['tau_syn_in'][c] * w_step[2 * c + 1]

    v_soma = y[_idx(SOMA, V_M)]
    if state['r'] > 0:
        y[_idx(SOMA, V_M)] = p['V_reset']
        r = state['r'] - 1
        spk = False
    else:
        if v_soma >= p['V_th']:
            y[_idx(SOMA, V_M)] = p['V_reset']
            r = int(math.ceil(p['t_ref'] / dt))
            spk = True
        else:
            r = 0
            spk = False

    state['y'] = y
    state['r'] = r
    state['h'] = h
    dftype = brainstate.environ.dftype()
    state['i_stim'] = np.asarray(x_next, dtype=dftype)
    return spk


class TestIAFCondAlphaMC(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        dftype = brainstate.environ.dftype()
        return bool(np.asarray(u.math.asarray(spk), dtype=dftype).reshape(-1)[0] > 0.0)

    def _step(self, neuron, step_idx, x=0.0 * u.pA, spike_events=None, current_events=None):
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x, spike_events=spike_events, current_events=current_events)

    def test_nest_cpp_default_parameters(self):
        neuron = iaf_cond_alpha_mc(1)

        self.assertEqual(neuron.V_th, -55.0 * u.mV)
        self.assertEqual(neuron.V_reset, -60.0 * u.mV)
        self.assertEqual(neuron.t_ref, 2.0 * u.ms)
        self.assertEqual(neuron.g_sp, 2.5 * u.nS)
        self.assertEqual(neuron.g_pd, 1.0 * u.nS)

        self.assertEqual(neuron.soma['g_L'], 10.0 * u.nS)
        self.assertEqual(neuron.soma['C_m'], 150.0 * u.pF)
        self.assertEqual(neuron.soma['tau_syn_ex'], 0.5 * u.ms)
        self.assertEqual(neuron.soma['tau_syn_in'], 2.0 * u.ms)

        self.assertEqual(neuron.proximal['g_L'], 5.0 * u.nS)
        self.assertEqual(neuron.proximal['C_m'], 75.0 * u.pF)

        self.assertEqual(neuron.distal['g_L'], 10.0 * u.nS)
        self.assertEqual(neuron.distal['C_m'], 150.0 * u.pF)

    def test_receptor_types_and_recordables_match_nest(self):
        neuron = iaf_cond_alpha_mc(1)

        self.assertEqual(
            neuron.receptor_types,
            {
                'soma_exc': 1,
                'soma_inh': 2,
                'proximal_exc': 3,
                'proximal_inh': 4,
                'distal_exc': 5,
                'distal_inh': 6,
                'soma_curr': 7,
                'proximal_curr': 8,
                'distal_curr': 9,
            },
        )
        self.assertEqual(
            neuron.recordables,
            [
                'V_m.s', 'g_ex.s', 'g_in.s',
                'V_m.p', 'g_ex.p', 'g_in.p',
                'V_m.d', 'g_ex.d', 'g_in.d',
                't_ref_remaining',
            ],
        )

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            iaf_cond_alpha_mc(1, V_reset=-55.0 * u.mV, V_th=-55.0 * u.mV)
        with self.assertRaises(ValueError):
            iaf_cond_alpha_mc(1, t_ref=-0.1 * u.ms)
        with self.assertRaises(ValueError):
            iaf_cond_alpha_mc(1, soma={'C_m': 0.0 * u.pF})
        with self.assertRaises(ValueError):
            iaf_cond_alpha_mc(1, proximal={'tau_syn_ex': 0.0 * u.ms})
        with self.assertRaises(ValueError):
            iaf_cond_alpha_mc(1, distal={'tau_syn_in': 0.0 * u.ms})

    def test_current_input_is_compartment_specific_and_delayed_by_one_step(self):
        zero_comp = {
            'g_L': 0.0 * u.nS,
            'C_m': 100.0 * u.pF,
            'E_ex': 0.0 * u.mV,
            'E_in': -85.0 * u.mV,
            'E_L': 0.0 * u.mV,
            'tau_syn_ex': 0.5 * u.ms,
            'tau_syn_in': 2.0 * u.ms,
            'I_e': 0.0 * u.pA,
        }

        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_alpha_mc(
                1,
                V_th=1e9 * u.mV,
                V_reset=0.0 * u.mV,
                g_sp=0.0 * u.nS,
                g_pd=0.0 * u.nS,
                soma=zero_comp,
                proximal=zero_comp,
                distal=zero_comp,
                V_initializer={'soma': 0.0 * u.mV, 'proximal': 0.0 * u.mV, 'distal': 0.0 * u.mV},
            )
            neuron.init_state()

            self._step(
                neuron,
                0,
                x={'soma': 100.0 * u.pA, 'proximal': 50.0 * u.pA, 'distal': -20.0 * u.pA},
            )
            dftype = brainstate.environ.dftype()
            v0 = np.asarray(u.math.asarray(neuron.V.value / u.mV), dtype=dftype).reshape(-1, 3)[0]
            self.assertTrue(np.allclose(v0, [0.0, 0.0, 0.0]))

            self._step(neuron, 1, x=0.0 * u.pA)
            v = np.asarray(u.math.asarray(neuron.V.value / u.mV), dtype=dftype).reshape(-1, 3)[0]
            self.assertAlmostEqual(v[0], 0.1, delta=1e-10)
            self.assertAlmostEqual(v[1], 0.05, delta=1e-10)
            self.assertAlmostEqual(v[2], -0.02, delta=1e-10)

    def test_spike_receptor_routing_and_nonnegative_weight_constraint(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_alpha_mc(1, V_th=1e9 * u.mV)
            neuron.init_state()

            self._step(
                neuron,
                0,
                spike_events=[
                    ('soma_exc', 2.0 * u.nS),
                    ('proximal_inh', 3.0 * u.nS),
                    (6, 5.0 * u.nS),
                ],
            )

            dftype = brainstate.environ.dftype()
            dg_ex = np.asarray(neuron.dg_ex.value, dtype=dftype).reshape(-1, 3)[0]
            dg_in = np.asarray(neuron.dg_in.value, dtype=dftype).reshape(-1, 3)[0]

            self.assertAlmostEqual(dg_ex[SOMA], math.e / 0.5 * 2.0, delta=1e-12)
            self.assertAlmostEqual(dg_in[PROX], math.e / 2.0 * 3.0, delta=1e-12)
            self.assertAlmostEqual(dg_in[DIST], math.e / 2.0 * 5.0, delta=1e-12)

            with self.assertRaises(ValueError):
                self._step(neuron, 1, spike_events=[(1, -1.0 * u.nS)])

    def test_reference_trace_matches_nest_update_logic(self):
        soma = {
            'g_L': 12.0 * u.nS,
            'C_m': 150.0 * u.pF,
            'E_ex': 0.0 * u.mV,
            'E_in': -85.0 * u.mV,
            'E_L': -70.0 * u.mV,
            'tau_syn_ex': 0.5 * u.ms,
            'tau_syn_in': 2.0 * u.ms,
            'I_e': 1500.0 * u.pA,
        }
        proximal = {
            'g_L': 5.0 * u.nS,
            'C_m': 75.0 * u.pF,
            'E_ex': 0.0 * u.mV,
            'E_in': -85.0 * u.mV,
            'E_L': -70.0 * u.mV,
            'tau_syn_ex': 0.5 * u.ms,
            'tau_syn_in': 2.0 * u.ms,
            'I_e': 0.0 * u.pA,
        }
        distal = {
            'g_L': 10.0 * u.nS,
            'C_m': 150.0 * u.pF,
            'E_ex': 0.0 * u.mV,
            'E_in': -85.0 * u.mV,
            'E_L': -70.0 * u.mV,
            'tau_syn_ex': 0.5 * u.ms,
            'tau_syn_in': 2.0 * u.ms,
            'I_e': 0.0 * u.pA,
        }

        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_cond_alpha_mc(
                1,
                V_th=-60.0 * u.mV,
                V_reset=-65.0 * u.mV,
                t_ref=0.3 * u.ms,
                g_sp=4.0 * u.nS,
                g_pd=1.5 * u.nS,
                soma=soma,
                proximal=proximal,
                distal=distal,
                V_initializer={'soma': -69.0 * u.mV, 'proximal': -70.0 * u.mV, 'distal': -71.0 * u.mV},
            )
            neuron.init_state()

            x_seq = []
            w_seq = []
            for k in range(80):
                dftype = brainstate.environ.dftype()
                x_k = np.zeros(3, dtype=dftype)
                if 10 <= k < 20:
                    x_k[DIST] += 80.0
                if 25 <= k < 35:
                    x_k[PROX] -= 40.0
                if 40 <= k < 55:
                    x_k[SOMA] += 50.0
                if 60 <= k < 65:
                    x_k[SOMA] -= 30.0
                x_seq.append(x_k)

                w_k = np.zeros(6, dtype=dftype)
                if k in (12, 28, 45):
                    w_k[4] = 1.2  # distal_exc
                if k in (30, 46):
                    w_k[3] = 0.8  # proximal_inh
                if k in (50, 70):
                    w_k[0] = 1.5  # soma_exc
                if k in (52, 71):
                    w_k[1] = 0.9  # soma_inh
                w_seq.append(w_k)

            p = {
                'V_th': -60.0,
                'V_reset': -65.0,
                't_ref': 0.3,
                'g_sp': 4.0,
                'g_pd': 1.5,
                'g_L': np.asarray([12.0, 5.0, 10.0], dtype=dftype),
                'C_m': np.asarray([150.0, 75.0, 150.0], dtype=dftype),
                'E_ex': np.asarray([0.0, 0.0, 0.0], dtype=dftype),
                'E_in': np.asarray([-85.0, -85.0, -85.0], dtype=dftype),
                'E_L': np.asarray([-70.0, -70.0, -70.0], dtype=dftype),
                'tau_syn_ex': np.asarray([0.5, 0.5, 0.5], dtype=dftype),
                'tau_syn_in': np.asarray([2.0, 2.0, 2.0], dtype=dftype),
                'I_e': np.asarray([1500.0, 0.0, 0.0], dtype=dftype),
            }

            y0 = np.zeros(NCOMP * NSTATE_COMP, dtype=dftype)
            y0[_idx(SOMA, V_M)] = -69.0
            y0[_idx(PROX, V_M)] = -70.0
            y0[_idx(DIST, V_M)] = -71.0
            ref_state = {
                'y': y0,
                'r': 0,
                'h': 0.1,
                'i_stim': np.zeros(3, dtype=dftype),
            }

            spikes_model = []
            spikes_ref = []
            for k in range(len(x_seq)):
                events = []
                for rid, w in enumerate(w_seq[k], start=1):
                    if w != 0.0:
                        events.append((rid, w * u.nS))
                events = events if events else None

                spk = self._step(
                    neuron,
                    k,
                    x={'soma': x_seq[k][0] * u.pA, 'proximal': x_seq[k][1] * u.pA, 'distal': x_seq[k][2] * u.pA},
                    spike_events=events,
                )
                spikes_model.append(self._is_spike(spk))
                spikes_ref.append(_reference_step(ref_state, p, x_seq[k], w_seq[k], 0.1))

                v_m = np.asarray(u.math.asarray(neuron.V.value / u.mV), dtype=dftype).reshape(-1, 3)[0]
                dg_ex = np.asarray(neuron.dg_ex.value, dtype=dftype).reshape(-1, 3)[0]
                g_ex = np.asarray(u.math.asarray(neuron.g_ex.value / u.nS), dtype=dftype).reshape(-1, 3)[0]
                dg_in = np.asarray(neuron.dg_in.value, dtype=dftype).reshape(-1, 3)[0]
                g_in = np.asarray(u.math.asarray(neuron.g_in.value / u.nS), dtype=dftype).reshape(-1, 3)[0]

                for c in range(NCOMP):
                    self.assertAlmostEqual(v_m[c], ref_state['y'][_idx(c, V_M)], delta=3e-6)
                    self.assertAlmostEqual(dg_ex[c], ref_state['y'][_idx(c, DG_EX)], delta=3e-6)
                    self.assertAlmostEqual(g_ex[c], ref_state['y'][_idx(c, G_EX)], delta=3e-6)
                    self.assertAlmostEqual(dg_in[c], ref_state['y'][_idx(c, DG_IN)], delta=3e-6)
                    self.assertAlmostEqual(g_in[c], ref_state['y'][_idx(c, G_IN)], delta=3e-6)

                ditype = brainstate.environ.ditype()
                self.assertEqual(int(np.asarray(neuron.refractory_step_count.value, dtype=ditype).reshape(-1)[0]),
                                 ref_state['r'])
                self.assertAlmostEqual(
                    float(
                        np.asarray(u.math.asarray(neuron.integration_step.value / u.ms), dtype=dftype).reshape(-1)[
                            0]),
                    ref_state['h'],
                    delta=3e-6,
                )

            self.assertEqual(spikes_model, spikes_ref)
            self.assertTrue(any(spikes_model))


if __name__ == '__main__':
    unittest.main()
