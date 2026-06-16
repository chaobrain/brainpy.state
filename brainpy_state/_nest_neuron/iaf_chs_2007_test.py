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
import numpy as np
from brainpy.state import iaf_chs_2007

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)


def _reference_step(state, params, w_step):
    r"""Single-step scalar reference that mirrors NEST iaf_chs_2007 C++ update."""
    h = params['dt']

    p11 = math.exp(-h / params['tau_epsp'])
    p22 = p11
    p30 = math.exp(-h / params['tau_reset'])
    p21 = params['V_epsp'] * math.e * p11 * h / params['tau_epsp']

    v_syn = state['V_syn'] * p22 + state['i_syn_ex'] * p21
    i_syn_ex = state['i_syn_ex'] * p11
    i_syn_ex += max(w_step, 0.0)
    v_spike = state['V_spike'] * p30

    noise_term = 0.0
    if params['V_noise'] > 0.0 and params['noise'].size > 0:
        pos = state['position']
        if pos >= params['noise'].size:
            raise IndexError('Noise signal exhausted.')
        noise_term = params['V_noise'] * float(params['noise'][pos])
        state['position'] = pos + 1

    v_m = v_syn + v_spike + noise_term
    spike = v_m >= 1.0
    if spike:
        v_spike -= params['V_reset']
        v_m -= params['V_reset']

    state['i_syn_ex'] = i_syn_ex
    state['V_syn'] = v_syn
    state['V_spike'] = v_spike
    state['V_m'] = v_m

    return spike


class TestIAFChs2007(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _scalar(x):
        dftype = brainstate.environ.dftype()
        return float(np.asarray(u.math.asarray(x), dtype=dftype).reshape(-1)[0])

    @staticmethod
    def _is_spike(spk):
        dftype = brainstate.environ.dftype()
        return bool(np.asarray(u.math.asarray(spk), dtype=dftype).reshape(-1)[0] > 0.0)

    def _step(self, neuron, k, x=0.0, delta_weights=None):
        if delta_weights is not None:
            for i, w in enumerate(delta_weights):
                neuron.add_delta_input(f'w_{k}_{i}', float(w), label='w_ex')
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_nest_cpp_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = iaf_chs_2007(1)
            self.assertAlmostEqual(self._scalar(neuron.tau_epsp / u.ms), 8.5, delta=0.0)
            self.assertAlmostEqual(self._scalar(neuron.tau_reset / u.ms), 15.4, delta=0.0)
            self.assertAlmostEqual(self._scalar(neuron.V_epsp), 0.77, delta=0.0)
            self.assertAlmostEqual(self._scalar(neuron.V_reset), 2.31, delta=0.0)
            self.assertAlmostEqual(self._scalar(neuron.V_noise), 0.0, delta=0.0)
            self.assertEqual(neuron.noise.size, 0)
            self.assertEqual(neuron.spk_reset, 'hard')

    def test_parameter_validation(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                iaf_chs_2007(1, V_epsp=-1e-3)
            with self.assertRaises(ValueError):
                iaf_chs_2007(1, V_reset=-1e-3)
            with self.assertRaises(ValueError):
                iaf_chs_2007(1, tau_epsp=0.0 * u.ms)
            with self.assertRaises(ValueError):
                iaf_chs_2007(1, tau_reset=0.0 * u.ms)

    def test_positive_spike_weights_only_and_one_step_delay_to_vm(self):
        with brainstate.environ.context(dt=self.dt):
            n_pos = iaf_chs_2007(1, V_initializer=braintools.init.Constant(0.0))
            n_mix = iaf_chs_2007(1, V_initializer=braintools.init.Constant(0.0))
            n_pos.init_state()
            n_mix.init_state()

            s_pos_0 = self._step(n_pos, 0, delta_weights=[5.0])
            s_mix_0 = self._step(n_mix, 0, delta_weights=[5.0, -3.0])
            self.assertFalse(self._is_spike(s_pos_0))
            self.assertFalse(self._is_spike(s_mix_0))

            self.assertAlmostEqual(self._scalar(n_pos.V.value), 0.0, delta=1e-15)
            self.assertAlmostEqual(self._scalar(n_mix.V.value), 0.0, delta=1e-15)
            self.assertAlmostEqual(self._scalar(n_pos.i_syn_ex.value), 5.0, delta=1e-15)
            self.assertAlmostEqual(self._scalar(n_mix.i_syn_ex.value), 5.0, delta=1e-15)

            self._step(n_pos, 1)
            self._step(n_mix, 1)

            self.assertAlmostEqual(self._scalar(n_pos.V.value), self._scalar(n_mix.V.value), delta=1e-15)
            self.assertAlmostEqual(self._scalar(n_pos.V_syn.value), self._scalar(n_mix.V_syn.value), delta=1e-15)
            self.assertGreater(self._scalar(n_pos.V.value), 0.0)

    def test_current_input_argument_is_ignored_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            n_a = iaf_chs_2007(1, V_initializer=braintools.init.Constant(0.0))
            n_b = iaf_chs_2007(1, V_initializer=braintools.init.Constant(0.0))
            n_a.init_state()
            n_b.init_state()

            for k, x_a in enumerate([1000.0 * u.pA, -800.0 * u.pA, 250.0 * u.pA]):
                d = [7.0] if k == 0 else None
                self._step(n_a, k, x=x_a, delta_weights=d)
                self._step(n_b, k, x=0.0 * u.pA, delta_weights=d)

                self.assertAlmostEqual(self._scalar(n_a.V.value), self._scalar(n_b.V.value), delta=1e-15)
                self.assertAlmostEqual(self._scalar(n_a.i_syn_ex.value), self._scalar(n_b.i_syn_ex.value), delta=1e-15)
                self.assertAlmostEqual(self._scalar(n_a.V_syn.value), self._scalar(n_b.V_syn.value), delta=1e-15)
                self.assertAlmostEqual(self._scalar(n_a.V_spike.value), self._scalar(n_b.V_spike.value), delta=1e-15)

    def test_noise_sequence_consumption(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_chs_2007(
                1,
                V_epsp=0.0,
                V_reset=0.0,
                V_noise=0.2,
                noise=[1.0, -1.0, 0.5],
                V_initializer=braintools.init.Constant(0.0),
            )
            neuron.init_state()

            expected_vm = [0.2, -0.2, 0.1]
            for k, vm in enumerate(expected_vm):
                spk = self._step(neuron, k)
                self.assertFalse(self._is_spike(spk))
                self.assertAlmostEqual(self._scalar(neuron.V.value), vm, delta=1e-15)
                self.assertEqual(int(np.asarray(neuron.position.value).reshape(-1)[0]), k + 1)

            with self.assertRaises(IndexError):
                self._step(neuron, len(expected_vm))

    def test_reference_trace_matches_nest_step_logic(self):
        with brainstate.environ.context(dt=self.dt):
            import jax.numpy as jnp
            dftype = brainstate.environ.dftype()
            noise = np.linspace(-1.0, 1.0, 256, dtype=dftype)
            neuron = iaf_chs_2007(
                1,
                tau_epsp=8.5 * u.ms,
                tau_reset=15.4 * u.ms,
                V_epsp=0.77,
                V_reset=2.31,
                V_noise=0.05,
                noise=noise,
                V_initializer=braintools.init.Constant(0.0),
            )
            neuron.init_state()

            w_seq = [0.0, 50.0, 0.0, -20.0, 0.0, 0.0, 30.0, 0.0, 0.0, 10.0, 0.0, -7.0, 0.0, 0.0, 0.0] + [0.0] * 30
            n_steps = len(w_seq)

            params = {
                'dt': 0.1,
                'tau_epsp': 8.5,
                'tau_reset': 15.4,
                'V_epsp': 0.77,
                'V_reset': 2.31,
                'V_noise': 0.05,
                'noise': noise,
            }
            ref_state = {
                'i_syn_ex': 0.0,
                'V_syn': 0.0,
                'V_spike': 0.0,
                'V_m': 0.0,
                'position': 0,
            }

            # Build reference traces with the Python scalar reference loop.
            ref_i_syn_ex = []
            ref_V_syn = []
            ref_V_spike = []
            ref_V_m = []
            ref_position = []
            spikes_ref = []
            for w in w_seq:
                spikes_ref.append(_reference_step(ref_state, params, w))
                ref_i_syn_ex.append(ref_state['i_syn_ex'])
                ref_V_syn.append(ref_state['V_syn'])
                ref_V_spike.append(ref_state['V_spike'])
                ref_V_m.append(ref_state['V_m'])
                ref_position.append(ref_state['position'])

            # Pre-compute per-step weights as a JAX array so the loop body has
            # a fixed, JIT-traceable graph.
            w_array = jnp.array(w_seq, dtype=dftype)

            def _body(k):
                # Always add the weight; clamping to 0 inside update() matches
                # the original "delta_weights=[w] if w != 0.0 else None" semantics.
                neuron.add_delta_input('w_ex', w_array[k], label='w_ex')
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update()
                return (
                    spk,
                    neuron.i_syn_ex.value,
                    neuron.V_syn.value,
                    neuron.V_spike.value,
                    neuron.V.value,
                    neuron.position.value,
                )

            results = brainstate.transform.for_loop(_body, jnp.arange(n_steps))

            # results[i] has shape (n_steps, 1); [k, 0] extracts step k, neuron 0.
            spikes_model = [bool(float(results[0][k, 0]) > 0.5) for k in range(n_steps)]
            i_syn_trace = np.asarray(results[1][:, 0])
            V_syn_trace = np.asarray(results[2][:, 0])
            V_spike_trace = np.asarray(results[3][:, 0])
            V_m_trace = np.asarray(results[4][:, 0])
            pos_trace = np.asarray(results[5][:, 0], dtype=int)

            for k in range(n_steps):
                self.assertAlmostEqual(float(i_syn_trace[k]), ref_i_syn_ex[k], delta=1e-10)
                self.assertAlmostEqual(float(V_syn_trace[k]), ref_V_syn[k], delta=1e-10)
                self.assertAlmostEqual(float(V_spike_trace[k]), ref_V_spike[k], delta=1e-10)
                self.assertAlmostEqual(float(V_m_trace[k]), ref_V_m[k], delta=1e-10)
                self.assertEqual(pos_trace[k], ref_position[k])

            self.assertEqual(spikes_model, spikes_ref)
            self.assertTrue(any(spikes_model))


if __name__ == '__main__':
    unittest.main()
