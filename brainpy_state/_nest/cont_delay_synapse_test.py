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
import brainunit as u
import jax
import numpy as np

from brainpy.state import cont_delay_synapse, iaf_psc_delta_ps

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class _PreciseReceiver:
    def __init__(self):
        self._events = []

    def handle_cont_delay_synapse_event(self, value, receptor_type, event_type, offset_ms):
        self._events.append((value, int(receptor_type), str(event_type), float(offset_ms)))

    def pop_events(self):
        events = self._events
        self._events = []
        return events

    def pop_spike_events(self):
        spike_events = []
        for value, _receptor_type, event_type, offset_ms in self.pop_events():
            if event_type != 'spike':
                raise ValueError(f'Only spike events are expected, got: {event_type}')
            spike_events.append((offset_ms * u.ms, value))
        return spike_events


def _time_to_step_offset(time_ms: float, dt_ms: float) -> tuple[int, float]:
    t = float(time_ms)
    dt = float(dt_ms)
    k = t / dt
    nearest = round(k)
    if math.isclose(k, nearest, rel_tol=0.0, abs_tol=1e-12):
        step = int(nearest) - 1
        offset = 0.0
    else:
        step = int(math.floor(k))
        offset = (step + 1) * dt - t
    if step < 0:
        raise ValueError('Spike times must be strictly positive.')
    return step, offset


def _source_events_from_spike_times(spike_times_ms, dt_ms: float):
    events = {}
    for t_ms in spike_times_ms:
        step, offset = _time_to_step_offset(float(t_ms), dt_ms)
        events.setdefault(step, []).append((offset * u.ms, 1.0))
    return events


def _run_bp_simulation(resolution_ms: float, delay_ms: float, explicit: bool = False):
    dt = float(resolution_ms) * u.ms
    source_spike_times_ms = [2.0, 5.5]
    source_events = _source_events_from_spike_times(source_spike_times_ms, float(resolution_ms))

    with brainstate.environ.context(dt=dt):
        neuron = iaf_psc_delta_ps(1)
        neuron.init_state()

        receiver = _PreciseReceiver()
        if explicit:
            syn = cont_delay_synapse(weight=100.0 * u.mV, delay=1.0 * u.ms, post=receiver)
            syn.set(delay=float(delay_ms) * u.ms)
        else:
            syn = cont_delay_synapse(weight=100.0 * u.mV, delay=float(delay_ms) * u.ms, post=receiver)
        syn.init_state()

        spike_times_out = []
        n_steps = int(round(10.0 / float(resolution_ms)))
        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt):
                syn.update(pre_spike=0.0, spike_events=source_events.get(step, None))
                spk = neuron.update(spike_events=receiver.pop_spike_events())

            if bool(u.math.all(spk > 0.0)):
                t_spk = float(np.asarray(neuron.last_spike_time.value / u.ms, dtype=np.float64).reshape(()))
                spike_times_out.append(t_spk)

    return np.asarray(spike_times_out, dtype=np.float64)


class TestContDelaySynapseParameters(unittest.TestCase):
    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = cont_delay_synapse()
            syn.init_state()
            syn.update(pre_spike=0.0)
            params = syn.get()

        self.assertEqual(params['weight'], 1.0)
        self.assertEqual(params['delay'], 1.0)
        self.assertEqual(params['delay_steps'], 1)
        self.assertAlmostEqual(params['delay_offset'], 0.0, delta=1e-15)
        self.assertEqual(params['receptor_type'], 0)
        self.assertEqual(params['event_type'], 'spike')
        self.assertEqual(params['synapse_model'], 'cont_delay_synapse')

    def test_delay_decomposition_matches_nest(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = cont_delay_synapse(delay=1.7 * u.ms)
            syn.init_state()
            syn.update(pre_spike=0.0)

            params = syn.get()
            self.assertEqual(params['delay_steps'], 2)
            self.assertAlmostEqual(params['delay_offset'], 0.3, delta=1e-12)
            self.assertAlmostEqual(params['delay'], 1.7, delta=1e-12)

            syn.set(delay=3.0 * u.ms)
            params = syn.get()
            self.assertEqual(params['delay_steps'], 3)
            self.assertAlmostEqual(params['delay_offset'], 0.0, delta=1e-12)
            self.assertAlmostEqual(params['delay'], 3.0, delta=1e-12)

    def test_check_synapse_params_warns_about_connect_delay_rounding(self):
        syn = cont_delay_synapse()
        with self.assertWarnsRegex(UserWarning, 'The delay will be rounded to the next multiple'):
            syn.check_synapse_params({'synapse_model': 'cont_delay_synapse', 'delay': 1.7})

    def test_delay_must_be_at_least_resolution(self):
        with brainstate.environ.context(dt=1.0 * u.ms):
            with self.assertRaisesRegex(
                ValueError,
                'Continuous delay must be greater than or equal to resolution',
            ):
                cont_delay_synapse(delay=0.7 * u.ms)


class TestContDelaySynapseOrdering(unittest.TestCase):
    def test_offset_carry_matches_nest_send_semantics(self):
        recv = _PreciseReceiver()
        dt = 1.0 * u.ms

        with brainstate.environ.context(dt=dt):
            syn = cont_delay_synapse(weight=2.0, delay=1.7 * u.ms, receptor_type=3, post=recv)
            syn.init_state()

            with brainstate.environ.context(t=0.0 * u.ms):
                syn.update(
                    pre_spike=0.0,
                    spike_events=[
                        (0.8 * u.ms, 1.0),  # 0.8 + 0.3 => carry -> step+1 with offset 0.1
                        (0.5 * u.ms, 1.0),  # 0.5 + 0.3 => step+2 with offset 0.8
                    ],
                )
            self.assertEqual(recv.pop_events(), [])

            with brainstate.environ.context(t=1.0 * u.ms):
                delivered = syn.update(pre_spike=0.0)
                events_step_1 = recv.pop_events()

            with brainstate.environ.context(t=2.0 * u.ms):
                delivered_2 = syn.update(pre_spike=0.0)
                events_step_2 = recv.pop_events()

        self.assertEqual(delivered, 1)
        self.assertEqual(delivered_2, 1)

        self.assertEqual(len(events_step_1), 1)
        self.assertEqual(len(events_step_2), 1)

        value_1, receptor_1, event_type_1, offset_1 = events_step_1[0]
        value_2, receptor_2, event_type_2, offset_2 = events_step_2[0]

        v1 = float(np.asarray(u.math.asarray(value_1), dtype=np.float64).reshape(()))
        v2 = float(np.asarray(u.math.asarray(value_2), dtype=np.float64).reshape(()))

        self.assertAlmostEqual(v1, 2.0, delta=1e-12)
        self.assertAlmostEqual(v2, 2.0, delta=1e-12)
        self.assertEqual(receptor_1, 3)
        self.assertEqual(receptor_2, 3)
        self.assertEqual(event_type_1, 'spike')
        self.assertEqual(event_type_2, 'spike')
        self.assertAlmostEqual(offset_1, 0.1, delta=1e-12)
        self.assertAlmostEqual(offset_2, 0.8, delta=1e-12)

    def test_source_offset_validation(self):
        recv = _PreciseReceiver()
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            syn = cont_delay_synapse(post=recv)
            syn.init_state()

            with brainstate.environ.context(t=0.0 * u.ms):
                with self.assertRaisesRegex(ValueError, 'source_offset must satisfy 0 <= source_offset <= dt'):
                    syn.send(1.0, source_offset=1.1 * u.ms)


class TestContDelaySynapseDynamics(unittest.TestCase):
    def test_delay_compatible_with_resolution(self):
        # Mirrors NEST testsuite/pytests/sli2py_synapses/test_cont_delay_synapse.py.
        cases = [
            ([3.0, 6.5], 1.0, 1.0, False),
            ([3.7, 7.2], 1.0, 1.7, False),
            ([3.7, 7.2], 1.0, 1.7, True),
        ]

        for expected, resolution, delay, explicit in cases:
            with self.subTest(
                expected=expected,
                resolution=resolution,
                delay=delay,
                explicit=explicit,
            ):
                actual = _run_bp_simulation(
                    resolution_ms=resolution,
                    delay_ms=delay,
                    explicit=explicit,
                )
                np.testing.assert_allclose(
                    actual,
                    np.asarray(expected, dtype=np.float64),
                    atol=1e-12,
                    rtol=0.0,
                )

    def test_delay_shorter_than_resolution(self):
        with brainstate.environ.context(dt=1.0 * u.ms):
            with self.assertRaisesRegex(
                ValueError,
                'Continuous delay must be greater than or equal to resolution',
            ):
                cont_delay_synapse(delay=0.7 * u.ms)


class TestContDelaySynapseVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest

            if hasattr(nest, 'synapse_models'):
                return 'cont_delay_synapse' in nest.synapse_models
            return 'cont_delay_synapse' in nest.Models()
        except Exception:
            return False

    @staticmethod
    def _run_nest_simulation(resolution_ms: float, delay_ms: float, explicit: bool = False):
        import nest

        nest.ResetKernel()
        try:
            nest.set(resolution=float(resolution_ms), local_num_threads=1)
        except Exception:
            nest.resolution = float(resolution_ms)
            nest.local_num_threads = 1

        n = nest.Create('iaf_psc_delta_ps')
        sg = nest.Create(
            'spike_generator',
            params={
                'precise_times': True,
                'spike_times': [2.0, 5.5],
            },
        )
        sr = nest.Create('spike_recorder')

        if explicit:
            nest.Connect(
                sg,
                n,
                syn_spec={
                    'synapse_model': 'cont_delay_synapse',
                    'weight': 100.0,
                    'delay': float(delay_ms),
                },
            )
            for conn in nest.GetConnections(source=sg):
                nest.SetStatus(conn, params={'delay': float(delay_ms)})
        else:
            nest.SetDefaults(
                'cont_delay_synapse',
                {'weight': 100.0, 'delay': float(delay_ms)},
            )
            nest.Connect(sg, n, syn_spec={'synapse_model': 'cont_delay_synapse'})

        nest.Connect(n, sr)
        nest.Simulate(10.0)
        return np.asarray(sr.get('events')['times'], dtype=np.float64)

    def test_dynamics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                nest_times = self._run_nest_simulation(1.0, 1.7, explicit=explicit)
                bp_times = _run_bp_simulation(1.0, 1.7, explicit=explicit)
                np.testing.assert_allclose(bp_times, nest_times, atol=1e-12, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
