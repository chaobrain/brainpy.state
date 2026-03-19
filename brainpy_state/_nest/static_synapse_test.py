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

import math
import unittest

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy.state import iaf_psc_delta, static_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


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


def _run_bp_trace(spike_times_ms, dt_ms, sim_steps):
    dt = dt_ms * u.ms
    spike_steps = _spike_steps_from_times(spike_times_ms, dt_ms)

    params = dict(
        E_L=-49.0 * u.mV,
        V_th=-50.0 * u.mV,
        V_reset=-60.0 * u.mV,
        C_m=200.0 * u.pF,
        tau_m=20.0 * u.ms,
        t_ref=5.0 * u.ms,
        I_e=0.0 * u.pA,
        V_initializer=braintools.init.Constant(-60.0 * u.mV),
    )

    with brainstate.environ.context(dt=dt):
        neuron = iaf_psc_delta(1, **params)
        neuron.init_state()

        # Pre-compute synapse delivery schedule using NEST ld_round:
        # delay=0.1 ms → delay_steps = floor(0.1/dt_ms + 0.5).
        # Bypass the Python-queue static_synapse so the loop is JIT-compilable.
        delay_steps = int(math.floor(0.1 / dt_ms + 0.5))
        dftype = brainstate.environ.dftype()
        delta_v_np = np.zeros(sim_steps, dtype=dftype)
        for s in spike_steps:
            deliver_step = s + delay_steps
            if 0 <= deliver_step < sim_steps:
                delta_v_np[deliver_step] += 2.5  # weight in mV
        delta_v_arr = jnp.array(delta_v_np, dtype=dftype)

        def _step(k):
            neuron.add_delta_input('_syn_spike', delta_v_arr[k] * u.mV)
            with brainstate.environ.context(t=k * dt):
                spk = neuron.update(x=0.0 * u.pA)
            return (neuron.V.value / u.mV)[0], spk[0]

        v_all, spk_all = brainstate.transform.for_loop(_step, jnp.arange(sim_steps))

    dftype = brainstate.environ.dftype()
    v_trace = np.asarray(v_all, dtype=dftype)
    spk_trace = np.asarray(spk_all, dtype=dftype)
    spike_times_out = [(step + 1) * dt_ms for step in range(sim_steps) if spk_trace[step] > 0.0]
    return np.asarray(spike_times_out, dtype=dftype), v_trace


class TestStaticSynapseParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = static_synapse()
            syn.init_state()
            syn.update(pre_spike=0.0)
            params = syn.get()

        self.assertEqual(params['weight'], 1.0)
        self.assertEqual(params['delay'], 1.0)
        self.assertEqual(params['delay_steps'], 1)
        self.assertEqual(params['receptor_type'], 0)
        self.assertEqual(params['event_type'], 'spike')
        self.assertEqual(params['synapse_model'], 'static_synapse')

    def test_delay_rounding_matches_nest_ld_round(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = static_synapse(delay=3.5 * u.ms)
            syn.init_state()
            syn.update(pre_spike=0.0)
            self.assertEqual(syn.get()['delay_steps'], 4)
            self.assertAlmostEqual(syn.get()['delay'], 4.0, delta=1e-12)

            syn.set(delay=2.49 * u.ms)
            self.assertEqual(syn.get()['delay_steps'], 2)
            self.assertAlmostEqual(syn.get()['delay'], 2.0, delta=1e-12)

            syn.set(delay=2.5 * u.ms)
            self.assertEqual(syn.get()['delay_steps'], 3)
            self.assertAlmostEqual(syn.get()['delay'], 3.0, delta=1e-12)

    def test_delay_must_be_at_least_resolution(self):
        with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
            with self.assertRaisesRegex(ValueError, 'Delay must be greater than or equal to resolution'):
                static_synapse(delay=0.01 * u.ms)

    def test_parameter_validation(self):
        with self.assertRaisesRegex(ValueError, 'weight must be scalar'):
            dftype = brainstate.environ.dftype()
            static_synapse(weight=np.asarray([1.0, 2.0], dtype=dftype))

        with self.assertRaisesRegex(ValueError, 'delay must be scalar'):
            static_synapse(delay=np.asarray([1.0, 2.0]) * u.ms)

        with self.assertRaisesRegex(ValueError, 'receptor_type must be an integer'):
            static_synapse(receptor_type=1.25)

        with self.assertRaisesRegex(ValueError, 'Unsupported event_type'):
            static_synapse(event_type='invalid')


class TestStaticSynapseOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_delivery_occurs_after_delay_steps(self):
        recv = _MockReceiver()
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            syn = static_synapse(
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
        self.assertAlmostEqual(float(np.asarray(u.math.asarray(value), dtype=dftype).reshape(())), 2.5, delta=1e-12)
        self.assertEqual(label, 'receptor_7')

    def test_multiplicity_scales_weight(self):
        recv = _MockReceiver()
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            syn = static_synapse(weight=-1.2, delay=1.0 * u.ms, post=recv)
            syn.init_state()
            with brainstate.environ.context(t=0.0 * u.ms):
                syn.update(pre_spike=3.0)
            with brainstate.environ.context(t=1.0 * u.ms):
                syn.update(pre_spike=0.0)

        self.assertEqual(len(recv.delta_events), 1)
        _, value, _ = recv.delta_events[0]
        dftype = brainstate.environ.dftype()
        self.assertAlmostEqual(float(np.asarray(u.math.asarray(value), dtype=dftype).reshape(())), -3.6,
                               delta=1e-12)

    def test_current_and_delta_inputs_are_summed_before_send(self):
        recv = _MockReceiver()
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            syn = static_synapse(weight=2.0, delay=1.0 * u.ms, post=recv)
            syn.init_state()

            syn.add_current_input('pre_c', 1.0)
            syn.add_delta_input('pre_d', 2.0)

            with brainstate.environ.context(t=0.0 * u.ms):
                syn.update(pre_spike=1.0)  # total multiplicity: 1 + 1 + 2 = 4
            with brainstate.environ.context(t=1.0 * u.ms):
                syn.update(pre_spike=0.0)

        self.assertEqual(len(recv.delta_events), 1)
        _, value, _ = recv.delta_events[0]
        dftype = brainstate.environ.dftype()
        self.assertAlmostEqual(float(np.asarray(u.math.asarray(value), dtype=dftype).reshape(())), 8.0, delta=1e-12)

    def test_current_event_uses_current_input_channel(self):
        recv = _MockReceiver()
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            syn = static_synapse(weight=1.5, delay=1.0 * u.ms, post=recv, event_type='current')
            syn.init_state()
            with brainstate.environ.context(t=0.0 * u.ms):
                syn.update(pre_spike=2.0)
            with brainstate.environ.context(t=1.0 * u.ms):
                syn.update(pre_spike=0.0)

        self.assertEqual(len(recv.current_events), 1)
        _, value, _ = recv.current_events[0]
        dftype = brainstate.environ.dftype()
        self.assertAlmostEqual(float(np.asarray(u.math.asarray(value), dtype=dftype).reshape(())), 3.0, delta=1e-12)
        self.assertEqual(len(recv.delta_events), 0)


class TestStaticSynapseDynamics(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_iaf_psc_delta_trace_matches_nest_style_reference(self):
        dt_ms = 0.01
        sim_steps = 2000  # 20 ms
        spike_times_ms = [1.0, 2.0, 3.0, 4.0, 5.0, 10.5, 12.0]

        model_spikes, model_v = _run_bp_trace(spike_times_ms=spike_times_ms, dt_ms=dt_ms, sim_steps=sim_steps)

        # Independent reference simulation using NEST ordering:
        # pre step -> static_synapse delay queue -> iaf_psc_delta update.
        E_L = -49.0
        V = -60.0
        V_th = -50.0
        V_reset = -60.0
        tau_m = 20.0
        t_ref = 5.0
        delay_steps = int(round(0.1 / dt_ms))
        refr_steps = int(math.ceil(t_ref / dt_ms))
        decay = math.exp(-dt_ms / tau_m)

        source_steps = _spike_steps_from_times(spike_times_ms, dt_ms)
        arrival_steps = {s + delay_steps for s in source_steps}

        r = 0
        ref_spikes = []
        ref_v = []
        for step in range(sim_steps):
            delta_v = 2.5 if step in arrival_steps else 0.0

            if r == 0:
                V = E_L + (V - E_L) * decay + delta_v
            else:
                r -= 1

            spike = V >= V_th
            if spike:
                r = refr_steps
                V = V_reset
                ref_spikes.append((step + 1) * dt_ms)
            ref_v.append(V)

        dftype = brainstate.environ.dftype()
        ref_spikes = np.asarray(ref_spikes, dtype=dftype)
        ref_v = np.asarray(ref_v, dtype=dftype)

        np.testing.assert_allclose(model_spikes, ref_spikes, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(model_v, ref_v, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(model_spikes, np.asarray([4.1], dtype=dftype), atol=1e-12, rtol=0.0)


class TestStaticSynapseVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest
            if hasattr(nest, 'synapse_models'):
                return 'static_synapse' in nest.synapse_models
            return 'static_synapse' in nest.Models()
        except Exception:
            return False

    @staticmethod
    def _run_nest_spike_times(spike_times_ms, dt_ms, sim_ms):
        import nest

        nest.ResetKernel()
        nest.resolution = float(dt_ms)
        nest.local_num_threads = 1

        post = nest.Create(
            'iaf_psc_delta',
            params={
                'E_L': -49.0,
                'V_m': -60.0,
                'V_th': -50.0,
                'V_reset': -60.0,
                'C_m': 200.0,
                'tau_m': 20.0,
                't_ref': 5.0,
            },
        )
        dftype = brainstate.environ.dftype()
        sg = nest.Create(
            'spike_generator',
            params={
                'spike_times': list(np.asarray(spike_times_ms, dtype=dftype)),
                'precise_times': False,
            },
        )
        sr = nest.Create('spike_recorder')

        nest.Connect(
            sg,
            post,
            syn_spec={
                'synapse_model': 'static_synapse',
                'weight': 2.5,
                'delay': 0.1,
            },
        )
        nest.Connect(post, sr)
        nest.Simulate(float(sim_ms))

        events = sr.get('events')
        times = np.asarray(events['times'], dtype=dftype)
        return np.sort(times)

    def test_spike_times_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.01
        sim_ms = 20.0
        sim_steps = int(round(sim_ms / dt_ms))
        spike_times_ms = [1.0, 2.0, 3.0, 4.0, 5.0, 10.5, 12.0]

        nest_spikes = self._run_nest_spike_times(
            spike_times_ms=spike_times_ms,
            dt_ms=dt_ms,
            sim_ms=sim_ms,
        )
        bp_spikes, _ = _run_bp_trace(
            spike_times_ms=spike_times_ms,
            dt_ms=dt_ms,
            sim_steps=sim_steps,
        )

        np.testing.assert_allclose(bp_spikes, nest_spikes, atol=1e-12, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
