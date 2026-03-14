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
from brainpy.state import iaf_psc_alpha_multisynapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def alpha_fn(t, tau_syn):
    vals = np.zeros_like(t)
    mask = t > 0.0
    vals[mask] = math.e / tau_syn * t[mask] * np.exp(-t[mask] / tau_syn)
    return vals


def alpha_psc_voltage_response(t, tau_syn, tau_m, c_m, w):
    vals = np.zeros_like(t)
    mask = t > 0.0
    pref = w * math.e / (tau_syn * c_m)
    d = (1.0 / tau_syn - 1.0 / tau_m)
    term1 = (np.exp(-t[mask] / tau_m) - np.exp(-t[mask] / tau_syn)) / (d * d)
    term2 = t[mask] * np.exp(-t[mask] / tau_syn) / d
    vals[mask] = pref * (term1 - term2)
    return vals


class TestIAFPscAlphaMultisynapse(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, spike_events=None):
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_defaults_and_validation(self):
        neuron = iaf_psc_alpha_multisynapse(1)
        self.assertEqual(neuron.n_receptors, 1)
        self.assertTrue(np.allclose(neuron.tau_syn, [2.0]))
        with self.assertRaises(ValueError):
            iaf_psc_alpha_multisynapse(1, tau_syn=[-1.0] * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_alpha_multisynapse(1, V_reset=-55. * u.mV, V_th=-55. * u.mV)

    def test_simulation_against_analytical_solution(self):
        dftype = brainstate.environ.dftype()
        tau_syns = np.asarray([2.0, 20.0, 60.0, 100.0], dtype=dftype)
        delays = np.asarray([7.0, 5.0, 2.0, 1.0], dtype=dftype)
        weights = np.asarray([30.0, 50.0, 20.0, 10.0], dtype=dftype)
        c_m = 250.0
        tau_m = 15.0
        spike_time = 1.0
        simtime = 100.0
        dt = 1.0

        with brainstate.environ.context(dt=dt * u.ms):
            neuron = iaf_psc_alpha_multisynapse(
                1,
                C_m=c_m * u.pF,
                E_L=0.0 * u.mV,
                V_th=1500.0 * u.mV,
                I_e=0.0 * u.pA,
                tau_m=tau_m * u.ms,
                tau_syn=tau_syns * u.ms,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
            )
            neuron.init_state()

            steps = int(round(simtime / dt))
            step_events = {}
            for ridx, (dly, w) in enumerate(zip(delays, weights), start=1):
                t_event = spike_time + dly
                k = int(round((t_event - dt) / dt))
                step_events.setdefault(k, []).append((ridx, w * u.pA))

            times = []
            i_syn_trace = []
            v_trace = []
            for k in range(steps):
                self._step(neuron, k, spike_events=step_events.get(k, None))
                times.append((k + 1) * dt)
                i_syn_trace.append(np.asarray(u.math.asarray(neuron.y2_syn.value[0] / u.pA), dtype=dftype))
                v_trace.append(float((neuron.V.value / u.mV)[0]))

            times = np.asarray(times, dtype=dftype)
            i_syn_trace = np.asarray(i_syn_trace, dtype=dftype)
            v_trace = np.asarray(v_trace, dtype=dftype)

            i_syn_ref = []
            v_ref = np.zeros_like(times)
            for w, dly, tau_s in zip(weights, delays, tau_syns):
                i_syn_ref.append(alpha_fn(times - dly - spike_time, tau_s) * w)
                v_ref += alpha_psc_voltage_response(times - dly - spike_time, tau_s, tau_m, c_m, w)
            i_syn_ref = np.asarray(i_syn_ref, dtype=dftype).T

            self.assertTrue(np.allclose(i_syn_trace, i_syn_ref, atol=5e-9, rtol=1e-8))
            self.assertTrue(np.allclose(v_trace, v_ref, atol=5e-8, rtol=1e-8))


if __name__ == '__main__':
    unittest.main()
