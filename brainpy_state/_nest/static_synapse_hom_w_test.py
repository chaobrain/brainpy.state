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

import os

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import unittest

import brainstate
import brainunit as u
import jax
import numpy as np

from brainpy.state import iaf_psc_alpha, static_synapse_hom_w

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

# Reference from NEST testsuite/pytests/sli2py_synapses/test_syn_hom_w.py
# Columns: [time_step, V_m], where time_step must be multiplied by dt (0.1 ms).
dftype = brainstate.environ.dftype()
REFERENCE_DATA = np.array(
    [
        [1, -70.0],
        [2, -70.0],
        [3, -70.0],
        [4, -70.0],
        [27, -70.0],
        [28, -70.0],
        [29, -70.0],
        [30, -70.0],
        [31, -6.999740e01],
        [32, -6.998990e01],
        [33, -6.997810e01],
        [34, -6.996240e01],
        [35, -6.994340e01],
        [45, -6.964350e01],
        [46, -6.960840e01],
        [47, -6.957320e01],
        [48, -6.953800e01],
        [49, -6.950290e01],
        [50, -6.946810e01],
        [51, -6.943360e01],
        [52, -6.939950e01],
        [53, -6.936600e01],
        [54, -6.933300e01],
        [55, -6.930080e01],
        [60, -6.915080e01],
    ],
    dtype=dftype,
)


class _MockReceiver:
    def __init__(self):
        self.delta_events = []
        self.current_events = []

    def add_delta_input(self, key, inp, label=None):
        self.delta_events.append((key, inp, label))

    def add_current_input(self, key, inp, label=None):
        self.current_events.append((key, inp, label))


def _spike_steps_from_times(spike_times_ms, dt_ms):
    return {int(round((float(t_ms) - dt_ms) / dt_ms)) for t_ms in spike_times_ms}


def _run_bp_vm_trace(*, dt_ms, delay_ms, weight_pA, spike_times_ms, sim_steps):
    dt = dt_ms * u.ms
    spike_steps = _spike_steps_from_times(spike_times_ms, dt_ms)

    with brainstate.environ.context(dt=dt):
        neuron = iaf_psc_alpha(1)
        neuron.init_state()

        syn = static_synapse_hom_w(
            weight=weight_pA * u.pA,
            delay=delay_ms * u.ms,
            post=neuron,
            event_type='spike',
        )
        syn.init_state()

        vm_trace = []
        for step in range(sim_steps):
            pre_spike = 1.0 if step in spike_steps else 0.0
            with brainstate.environ.context(t=step * dt):
                syn.update(pre_spike=pre_spike)
                neuron.update(x=0.0 * u.pA)
            vm_trace.append(float((neuron.V.value / u.mV)[0]))

    dftype = brainstate.environ.dftype()
    return np.asarray(vm_trace, dtype=dftype)


class TestStaticSynapseHomWParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = static_synapse_hom_w()
            syn.init_state()
            syn.update(pre_spike=0.0)
            params = syn.get()

        self.assertEqual(params['weight'], 1.0)
        self.assertEqual(params['delay'], 1.0)
        self.assertEqual(params['delay_steps'], 1)
        self.assertEqual(params['receptor_type'], 0)
        self.assertEqual(params['event_type'], 'spike')
        self.assertEqual(params['synapse_model'], 'static_synapse_hom_w')

    def test_set_weight_is_forbidden(self):
        syn = static_synapse_hom_w()
        with self.assertRaisesRegex(ValueError, 'Setting of individual weights is not possible'):
            syn.set_weight(2.0)

    def test_check_synapse_params_rejects_weight(self):
        syn = static_synapse_hom_w()
        with self.assertRaisesRegex(ValueError, 'Weight cannot be specified'):
            syn.check_synapse_params({'synapse_model': 'static_synapse_hom_w', 'weight': 2.0})

        syn.check_synapse_params({'synapse_model': 'static_synapse_hom_w', 'delay': 1.5})

    def test_common_weight_can_be_updated_with_set(self):
        recv = _MockReceiver()
        with brainstate.environ.context(dt=1.0 * u.ms):
            syn = static_synapse_hom_w(weight=1.0, delay=1.0 * u.ms, post=recv)
            syn.init_state()
            syn.set(weight=2.5)

            with brainstate.environ.context(t=0.0 * u.ms):
                syn.update(pre_spike=1.0)
            with brainstate.environ.context(t=1.0 * u.ms):
                syn.update(pre_spike=0.0)

        self.assertEqual(len(recv.delta_events), 1)
        _, value, _ = recv.delta_events[0]
        dftype = brainstate.environ.dftype()
        value_f = float(np.asarray(u.math.asarray(value), dtype=dftype).reshape(()))
        self.assertAlmostEqual(value_f, 2.5, delta=1e-12)


class TestStaticSynapseHomWDynamics(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_delivery_occurs_after_delay_steps(self):
        recv = _MockReceiver()
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            syn = static_synapse_hom_w(
                weight=2.5,
                delay=0.3 * u.ms,
                receptor_type=7,
                post=recv,
                event_type='spike',
            )
            syn.init_state()

            delivered_per_step = []
            for step in range(6):
                with brainstate.environ.context(t=step * dt):
                    delivered_per_step.append(syn.update(pre_spike=1.0 if step == 2 else 0.0))

        self.assertEqual(delivered_per_step, [0, 0, 0, 0, 0, 1])
        self.assertEqual(len(recv.delta_events), 1)
        _, value, label = recv.delta_events[0]
        dftype = brainstate.environ.dftype()
        value_f = float(np.asarray(u.math.asarray(value), dtype=dftype).reshape(()))
        self.assertAlmostEqual(value_f, 2.5, delta=1e-12)
        self.assertEqual(label, 'receptor_7')

    def test_matches_nest_reference_trace(self):
        dt_ms = 0.1
        sim_steps = int(round(7.0 / dt_ms))
        vm_trace = _run_bp_vm_trace(
            dt_ms=dt_ms,
            delay_ms=1.0,
            weight_pA=100.0,
            spike_times_ms=[2.0],
            sim_steps=sim_steps,
        )

        ref_steps = REFERENCE_DATA[:, 0].astype(np.int64)
        actual = np.column_stack((ref_steps * dt_ms, vm_trace[ref_steps - 1]))
        expected = REFERENCE_DATA.copy()
        expected[:, 0] = expected[:, 0] * dt_ms

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)


class TestStaticSynapseHomWVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest
            if hasattr(nest, 'synapse_models'):
                return 'static_synapse_hom_w' in nest.synapse_models
            return 'static_synapse_hom_w' in nest.Models()
        except Exception:
            return False

    @staticmethod
    def _run_nest_vm_trace(*, dt_ms, delay_ms, weight, sim_ms):
        import nest

        nest.ResetKernel()
        nest.set(resolution=float(dt_ms), local_num_threads=1)

        sg = nest.Create(
            'spike_generator',
            params={
                'precise_times': False,
                'origin': 0.0,
                'spike_times': [2.0],
                'start': 1.0,
                'stop': 3.0,
            },
        )
        neuron = nest.Create('iaf_psc_alpha')
        vm = nest.Create('voltmeter', params={'time_in_steps': False, 'interval': float(dt_ms)})

        nest.SetDefaults('static_synapse_hom_w', {'weight': float(weight), 'delay': float(delay_ms)})
        nest.Connect(sg, neuron, syn_spec={'synapse_model': 'static_synapse_hom_w'})
        nest.Connect(vm, neuron, syn_spec={'synapse_model': 'static_synapse_hom_w'})
        nest.Simulate(float(sim_ms))

        events = vm.get('events')
        dftype = brainstate.environ.dftype()
        times = np.asarray(events['times'], dtype=dftype)
        voltages = np.asarray(events['V_m'], dtype=dftype)
        return times, voltages

    def test_vm_trace_matches_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.1
        sim_ms = 7.0
        sim_steps = int(round(sim_ms / dt_ms))

        nest_times, nest_vm = self._run_nest_vm_trace(
            dt_ms=dt_ms,
            delay_ms=1.0,
            weight=100.0,
            sim_ms=sim_ms,
        )
        bp_vm = _run_bp_vm_trace(
            dt_ms=dt_ms,
            delay_ms=1.0,
            weight_pA=100.0,
            spike_times_ms=[2.0],
            sim_steps=sim_steps,
        )

        sample_steps = np.rint(nest_times / dt_ms).astype(np.int64)
        bp_samples = bp_vm[sample_steps - 1]
        np.testing.assert_allclose(bp_samples, nest_vm, atol=1e-10, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
