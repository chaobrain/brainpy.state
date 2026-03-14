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
import unittest

import brainstate
import saiunit as u
import jax
import numpy as np
import numpy.testing as npt

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import weight_recorder


def _events_signature(events):
    ditype = brainstate.environ.ditype()
    senders = np.asarray(events['senders'], dtype=ditype)
    targets = np.asarray(events['targets'], dtype=ditype)
    receptors = np.asarray(events['receptors'], dtype=ditype)
    ports = np.asarray(events['ports'], dtype=ditype)
    dftype = brainstate.environ.dftype()
    weights = np.asarray(events['weights'], dtype=dftype)

    if 'offsets' in events:
        times = np.asarray(events['times'], dtype=ditype)
        offsets = np.asarray(events['offsets'], dtype=dftype)
        return sorted(
            (
                int(times[i]),
                round(float(offsets[i]), 12),
                int(senders[i]),
                int(targets[i]),
                int(receptors[i]),
                int(ports[i]),
                round(float(weights[i]), 12),
            )
            for i in range(times.size)
        )

    times = np.asarray(events['times'], dtype=dftype)
    return sorted(
        (
            round(float(times[i]), 12),
            int(senders[i]),
            int(targets[i]),
            int(receptors[i]),
            int(ports[i]),
            round(float(weights[i]), 12),
        )
        for i in range(times.size)
    )


def _run_bp_from_schedule(simtime_ms, dt_ms, wr_params, conn_dict, source_spike_times):
    n_steps = int(round(simtime_ms / dt_ms))
    dt = dt_ms * u.ms

    ditype = brainstate.environ.ditype()
    conn_source = np.asarray(conn_dict['source'], dtype=ditype)
    conn_target = np.asarray(conn_dict['target'], dtype=ditype)
    conn_port = np.asarray(conn_dict['port'], dtype=ditype)
    conn_receptor = np.asarray(conn_dict['receptor'], dtype=ditype)
    dftype = brainstate.environ.dftype()
    conn_weight = np.asarray(conn_dict['weight'], dtype=dftype)

    events_by_step = {}
    for src, spikes_ms in source_spike_times.items():
        src = int(src)
        src_mask = conn_source == src
        idxs = np.where(src_mask)[0]

        for spike_time in np.asarray(spikes_ms, dtype=dftype):
            stamp_step = int(np.rint(spike_time / dt_ms))
            step_events = events_by_step.setdefault(stamp_step, [])
            for idx in idxs:
                step_events.append(
                    (
                        conn_weight[idx],
                        conn_source[idx],
                        conn_target[idx],
                        conn_receptor[idx],
                        conn_port[idx],
                    )
                )

    with brainstate.environ.context(dt=dt):
        wr = weight_recorder(**wr_params)

        for step in range(n_steps):
            stamp_step = step + 1
            with brainstate.environ.context(t=step * dt):
                if stamp_step in events_by_step:
                    ev = events_by_step[stamp_step]
                    wr.update(
                        weights=np.asarray([x[0] for x in ev], dtype=dftype),
                        senders=np.asarray([x[1] for x in ev], dtype=ditype),
                        targets=np.asarray([x[2] for x in ev], dtype=ditype),
                        receptors=np.asarray([x[3] for x in ev], dtype=ditype),
                        ports=np.asarray([x[4] for x in ev], dtype=ditype),
                    )
                else:
                    wr.update()

    return wr.events


class TestWeightRecorder(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def test_default_parameters(self):
        wr = weight_recorder()
        self.assertEqual(wr.senders, ())
        self.assertEqual(wr.targets, ())
        self.assertTrue(u.math.allclose(wr.start, 0.0 * u.ms))
        self.assertIsNone(wr.stop)
        self.assertTrue(u.math.allclose(wr.origin, 0.0 * u.ms))
        self.assertFalse(wr.time_in_steps)
        self.assertEqual(wr.n_events, 0)

        ev = wr.events
        self.assertEqual(ev['times'].size, 0)
        self.assertEqual(ev['senders'].size, 0)
        self.assertEqual(ev['targets'].size, 0)
        self.assertEqual(ev['weights'].size, 0)
        self.assertEqual(ev['receptors'].size, 0)
        self.assertEqual(ev['ports'].size, 0)
        self.assertTrue('offsets' not in ev)

    def test_window_is_start_exclusive_and_stop_inclusive(self):
        with brainstate.environ.context(dt=self.dt):
            wr = weight_recorder(start=0.5 * u.ms, stop=1.0 * u.ms)
            for step in range(12):
                with brainstate.environ.context(t=step * self.dt):
                    dftype = brainstate.environ.dftype()
                    ditype = brainstate.environ.ditype()
                    wr.update(
                        weights=np.array([2.0], dtype=dftype),
                        senders=np.array([9], dtype=ditype),
                        targets=np.array([11], dtype=ditype),
                    )

        ev = wr.events
        expected_times = np.array([0.6, 0.7, 0.8, 0.9, 1.0], dtype=dftype)
        npt.assert_allclose(ev['times'], expected_times, atol=1e-12)
        npt.assert_array_equal(ev['senders'], np.full(expected_times.shape, 9, dtype=ditype))
        npt.assert_array_equal(ev['targets'], np.full(expected_times.shape, 11, dtype=ditype))
        npt.assert_allclose(ev['weights'], np.full(expected_times.shape, 2.0, dtype=dftype), atol=1e-12)
        npt.assert_array_equal(ev['receptors'], np.zeros(expected_times.shape, dtype=ditype))
        npt.assert_array_equal(ev['ports'], -np.ones(expected_times.shape, dtype=ditype))

    def test_sender_target_filters(self):
        with brainstate.environ.context(dt=self.dt):
            wr = weight_recorder(senders=[2, 4], targets=[8])

            with brainstate.environ.context(t=0.0 * u.ms):
                dftype = brainstate.environ.dftype()
                ditype = brainstate.environ.ditype()
                wr.update(
                    weights=np.array([1.0, 2.0, 3.0, 4.0], dtype=dftype),
                    senders=np.array([1, 2, 4, 4], dtype=ditype),
                    targets=np.array([8, 8, 7, 8], dtype=ditype),
                    receptors=np.array([0, 1, 2, 3], dtype=ditype),
                    ports=np.array([10, 11, 12, 13], dtype=ditype),
                )

        ev = wr.events
        self.assertEqual(ev['times'].size, 2)
        npt.assert_array_equal(ev['senders'], np.array([2, 4], dtype=ditype))
        npt.assert_array_equal(ev['targets'], np.array([8, 8], dtype=ditype))
        npt.assert_allclose(ev['weights'], np.array([2.0, 4.0], dtype=dftype), atol=1e-12)
        npt.assert_array_equal(ev['receptors'], np.array([1, 3], dtype=ditype))
        npt.assert_array_equal(ev['ports'], np.array([11, 13], dtype=ditype))

    def test_n_events_can_only_be_set_to_zero(self):
        with brainstate.environ.context(dt=self.dt):
            wr = weight_recorder()
            for step in range(3):
                with brainstate.environ.context(t=step * self.dt):
                    dftype = brainstate.environ.dftype()
                    wr.update(weights=np.array([1.0], dtype=dftype))

        self.assertEqual(wr.n_events, 3)
        wr.n_events = 0
        self.assertEqual(wr.n_events, 0)
        self.assertEqual(wr.events['times'].size, 0)

        with self.assertRaises(ValueError):
            wr.n_events = 2

    def test_time_in_steps_records_step_and_offset_and_locks(self):
        with brainstate.environ.context(dt=self.dt):
            wr = weight_recorder(time_in_steps=True)
            with brainstate.environ.context(t=0.0 * u.ms):
                dftype = brainstate.environ.dftype()
                ditype = brainstate.environ.ditype()
                wr.update(
                    weights=np.array([1.5], dtype=dftype),
                    senders=np.array([5], dtype=ditype),
                    targets=np.array([6], dtype=ditype),
                    offsets=np.array([0.03], dtype=dftype) * u.ms,
                )

        ev = wr.events
        npt.assert_array_equal(ev['senders'], np.array([5], dtype=ditype))
        npt.assert_array_equal(ev['targets'], np.array([6], dtype=ditype))
        npt.assert_array_equal(ev['times'], np.array([1], dtype=ditype))
        npt.assert_allclose(ev['offsets'], np.array([0.03], dtype=dftype), atol=1e-12)
        npt.assert_allclose(ev['weights'], np.array([1.5], dtype=dftype), atol=1e-12)

        with self.assertRaises(ValueError):
            wr.time_in_steps = False

    def test_matches_nest_static_synapse_trace(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        simtime_ms = 5.0
        dt_ms = self.dt_ms

        nest.ResetKernel()
        nest.resolution = dt_ms

        wr = nest.Create('weight_recorder')
        nest.CopyModel('static_synapse', 'static_synapse_rec', {'weight_recorder': wr, 'weight': 2.5})

        dftype = brainstate.environ.dftype()
        sg_times_0 = np.array([1.0, 2.0], dtype=dftype)
        sg_times_1 = np.array([1.5], dtype=dftype)

        sg = nest.Create('spike_generator', 2, params=[
            {'spike_times': list(sg_times_0)},
            {'spike_times': list(sg_times_1)},
        ])
        pre = nest.Create('parrot_neuron', 2)
        post = nest.Create('parrot_neuron', 2)

        nest.Connect(sg, pre, 'one_to_one')
        nest.Connect(pre, post, 'all_to_all', syn_spec='static_synapse_rec')

        conn_dict = nest.GetConnections(pre, post).get(['source', 'target', 'weight', 'port', 'receptor'])

        ditype = brainstate.environ.ditype()
        pre_ids = np.asarray(pre.tolist(), dtype=ditype)
        source_spike_times = {
            int(pre_ids[0]): sg_times_0 + 1.0,
            int(pre_ids[1]): sg_times_1 + 1.0,
        }

        nest.Simulate(simtime_ms)
        nest_events = wr.get('events')
        nest_events = {
            'times': np.asarray(nest_events['times'], dtype=dftype),
            'senders': np.asarray(nest_events['senders'], dtype=ditype),
            'targets': np.asarray(nest_events['targets'], dtype=ditype),
            'weights': np.asarray(nest_events['weights'], dtype=dftype),
            'receptors': np.asarray(nest_events['receptors'], dtype=ditype),
            'ports': np.asarray(nest_events['ports'], dtype=ditype),
        }

        bp_events = _run_bp_from_schedule(
            simtime_ms=simtime_ms,
            dt_ms=dt_ms,
            wr_params={},
            conn_dict=conn_dict,
            source_spike_times=source_spike_times,
        )

        self.assertEqual(_events_signature(bp_events), _events_signature(nest_events))

    def test_matches_nest_with_filters_window_and_time_in_steps(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        simtime_ms = 5.0
        dt_ms = self.dt_ms

        nest.ResetKernel()
        nest.resolution = dt_ms

        nest_wr_params = {
            'start': 1.8,
            'stop': 3.0,
            'origin': 0.2,
            'time_in_steps': True,
        }

        wr = nest.Create('weight_recorder', params=nest_wr_params)
        nest.CopyModel('static_synapse', 'static_synapse_rec', {'weight_recorder': wr, 'weight': 2.5})

        dftype = brainstate.environ.dftype()
        sg_times_0 = np.array([1.0, 2.0], dtype=dftype)
        sg_times_1 = np.array([1.5], dtype=dftype)

        sg = nest.Create('spike_generator', 2, params=[
            {'spike_times': list(sg_times_0)},
            {'spike_times': list(sg_times_1)},
        ])
        pre = nest.Create('parrot_neuron', 2)
        post = nest.Create('parrot_neuron', 2)

        nest.Connect(sg, pre, 'one_to_one')
        nest.Connect(pre, post, 'all_to_all', syn_spec='static_synapse_rec')

        conn_dict = nest.GetConnections(pre, post).get(['source', 'target', 'weight', 'port', 'receptor'])

        ditype = brainstate.environ.ditype()
        pre_ids = np.asarray(pre.tolist(), dtype=ditype)
        post_ids = np.asarray(post.tolist(), dtype=ditype)

        nest.SetStatus(wr, {
            'senders': [int(pre_ids[1])],
            'targets': [int(post_ids[1])],
        })

        source_spike_times = {
            int(pre_ids[0]): sg_times_0 + 1.0,
            int(pre_ids[1]): sg_times_1 + 1.0,
        }

        nest.Simulate(simtime_ms)
        nest_events = wr.get('events')
        nest_events = {
            'times': np.asarray(nest_events['times'], dtype=ditype),
            'offsets': np.asarray(nest_events['offsets'], dtype=dftype),
            'senders': np.asarray(nest_events['senders'], dtype=ditype),
            'targets': np.asarray(nest_events['targets'], dtype=ditype),
            'weights': np.asarray(nest_events['weights'], dtype=dftype),
            'receptors': np.asarray(nest_events['receptors'], dtype=ditype),
            'ports': np.asarray(nest_events['ports'], dtype=ditype),
        }

        bp_events = _run_bp_from_schedule(
            simtime_ms=simtime_ms,
            dt_ms=dt_ms,
            wr_params={
                'senders': [int(pre_ids[1])],
                'targets': [int(post_ids[1])],
                'start': 1.8 * u.ms,
                'stop': 3.0 * u.ms,
                'origin': 0.2 * u.ms,
                'time_in_steps': True,
            },
            conn_dict=conn_dict,
            source_spike_times=source_spike_times,
        )

        self.assertEqual(_events_signature(bp_events), _events_signature(nest_events))

    def test_filter_getters_match_constructor_input(self):
        ditype = brainstate.environ.ditype()
        wr = weight_recorder(senders=[2, 5], targets=np.array([8, 13], dtype=ditype))
        npt.assert_array_equal(wr.get('senders'), np.array([2, 5], dtype=ditype))
        npt.assert_array_equal(wr.get('targets'), np.array([8, 13], dtype=ditype))

    def test_validation_rules(self):
        with brainstate.environ.context(dt=self.dt):
            with brainstate.environ.context(t=0.0 * u.ms):
                with self.assertRaises(ValueError):
                    weight_recorder(senders=[0])
                with self.assertRaises(ValueError):
                    weight_recorder(targets=[-1])
                with self.assertRaises(ValueError):
                    dftype = brainstate.environ.dftype()
                    weight_recorder(start=1.0 * u.ms, stop=0.5 * u.ms).update(weights=np.array([1.0], dtype=dftype))
                with self.assertRaises(ValueError):
                    ditype = brainstate.environ.ditype()
                    weight_recorder().update(weights=np.array([1.0, 2.0], dtype=dftype),
                                             senders=np.array([1, 2, 3], dtype=ditype))
                with self.assertRaises(ValueError):
                    weight_recorder().update(weights=np.array([1.0], dtype=dftype), offsets=np.array([np.nan]))


if __name__ == '__main__':
    unittest.main()
