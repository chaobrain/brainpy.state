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

import unittest

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainpy.state import iaf_psc_exp_multisynapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def exp_psc_fn(t, tau_syn):
    vals = np.zeros_like(t)
    mask = t > 0.0
    vals[mask] = np.exp(-t[mask] / tau_syn)
    return vals


def exp_psc_voltage_response(t, tau_syn, tau_m, c_m, w):
    vals = np.zeros_like(t)
    mask = t > 0.0
    delta_e = np.exp(-t[mask] / tau_m) - np.exp(-t[mask] / tau_syn)
    vals[mask] = w / (c_m * (1.0 / tau_syn - 1.0 / tau_m)) * delta_e
    return vals


class TestIAFPscExpMultisynapse(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0. * u.pA, spike_events=None):
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x, spike_events=spike_events)

    def test_defaults_and_validation(self):
        neuron = iaf_psc_exp_multisynapse(1)
        self.assertEqual(neuron.n_receptors, 1)
        self.assertTrue(np.allclose(neuron.tau_syn, [2.0]))
        with self.assertRaises(ValueError):
            iaf_psc_exp_multisynapse(1, tau_syn=[2.0, 10.0] * u.ms, tau_m=10.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_exp_multisynapse(1, tau_syn=[-1.0] * u.ms)

    def test_simulation_against_analytical_solution(self):
        dftype = brainstate.environ.dftype()
        tau_syns = np.asarray([2.0, 20.0, 60.0, 100.0], dtype=dftype)
        delays = np.asarray([7.0, 5.0, 2.0, 1.0], dtype=dftype)
        weights = np.asarray([30.0, 50.0, 20.0, 10.0], dtype=dftype)
        c_m = 250.0
        tau_m = 15.0
        spike_time = 0.1
        simtime = 8.0

        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_exp_multisynapse(
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

            steps = int(round(simtime / 0.1))
            # In NEST, on-grid spike input affects the state in the current step.
            # For an event at absolute time T, queue it at step (T - dt)/dt.
            step_events = {}
            for ridx, (dly, w) in enumerate(zip(delays, weights), start=1):
                t_event = spike_time + dly
                k = int(round((t_event - 0.1) / 0.1))
                step_events.setdefault(k, []).append((ridx, w * u.pA))

            # Pre-compute per-step spike weight array for JIT-compatible for_loop.
            n_rec = len(tau_syns)
            w_per_step = np.zeros((steps, n_rec), dtype=dftype)
            for k_idx, events in step_events.items():
                for ridx, w in events:
                    w_per_step[k_idx, ridx - 1] += float(w / u.pA)
            w_per_step_jax = jnp.array(w_per_step)

            def _body(k):
                with brainstate.environ.context(t=k * 0.1 * u.ms):
                    neuron.update(w_by_rec=w_per_step_jax[k])
                return (
                    neuron.i_syn.value[0] / u.pA,
                    (neuron.V.value / u.mV)[0],
                )

            results = brainstate.transform.for_loop(_body, jnp.arange(steps))
            times = np.asarray([(k + 1) * 0.1 for k in range(steps)], dtype=dftype)
            i_syn_trace = np.asarray(u.get_mantissa(results[0]), dtype=dftype)
            v_trace = np.asarray(u.get_mantissa(results[1]), dtype=dftype)

            i_syn_ref = []
            v_ref = np.zeros_like(times)
            for w, dly, tau_s in zip(weights, delays, tau_syns):
                i_syn_ref.append(exp_psc_fn(times - dly - spike_time, tau_s) * w)
                v_ref += exp_psc_voltage_response(times - dly - spike_time, tau_s, tau_m, c_m, w)
            i_syn_ref = np.asarray(i_syn_ref, dtype=dftype).T

            self.assertTrue(np.allclose(i_syn_trace, i_syn_ref, atol=2e-10, rtol=1e-10))
            self.assertTrue(np.allclose(v_trace, v_ref, atol=5e-10, rtol=1e-10))


if __name__ == '__main__':
    unittest.main()
