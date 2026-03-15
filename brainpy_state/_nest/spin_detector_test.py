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

from brainpy.state import spin_detector


def _to_ms_scalar(value):
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        value = value / u.ms
    dftype = brainstate.environ.dftype()
    arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
    if arr.size != 1:
        raise ValueError('Expected scalar time value.')
    return float(arr[0])


def _prepare_bp_schedule(sender_to_times, dt_ms):
    schedule = {}
    eps = 1e-12

    for sender in sorted(int(s) for s in sender_to_times.keys()):
        dftype = brainstate.environ.dftype()
        times = np.asarray(sender_to_times[sender], dtype=dftype)
        if times.size == 0:
            continue

        uniq_times, counts = np.unique(times, return_counts=True)
        for t_spike, multiplicity in zip(uniq_times, counts):
            step_f = float(t_spike) / float(dt_ms)
            stamp_step = int(np.ceil(step_f - eps))
            if stamp_step <= 0:
                raise ValueError('Spike times must be strictly positive.')
            offset_ms = stamp_step * dt_ms - float(t_spike)
            step_events = schedule.setdefault(stamp_step, [])
            step_events.append((sender, int(multiplicity), float(offset_ms)))

    for stamp_step in schedule:
        schedule[stamp_step].sort(key=lambda x: (x[0], x[2]))
    return schedule


def _run_bp_spin_trace(simtime_ms, dt_ms, sd_params, sender_to_times):
    n_steps = int(round(simtime_ms / dt_ms))
    dt = dt_ms * u.ms
    schedule = _prepare_bp_schedule(sender_to_times, dt_ms=dt_ms)

    with brainstate.environ.context(dt=dt):
        sd = spin_detector(**sd_params)

        for step in range(n_steps):
            stamp_step = step + 1
            with brainstate.environ.context(t=step * dt):
                if stamp_step in schedule:
                    ev = schedule[stamp_step]
                    dftype = brainstate.environ.dftype()
                    ditype = brainstate.environ.ditype()
                    sd.update(
                        spikes=np.ones((len(ev),), dtype=dftype),
                        senders=np.asarray([x[0] for x in ev], dtype=ditype),
                        offsets=np.asarray([x[2] for x in ev], dtype=dftype) * u.ms,
                        multiplicities=np.asarray([x[1] for x in ev], dtype=ditype),
                    )
                else:
                    sd.update()

    return sd.events


def _run_nest_spin_trace(simtime_ms, dt_ms, sd_params, source_spike_times, precise_times=False):
    import nest

    nest.ResetKernel()
    nest.resolution = dt_ms

    gen_params = []
    for times in source_spike_times:
        dftype = brainstate.environ.dftype()
        p = {'spike_times': list(np.asarray(times, dtype=dftype))}
        if precise_times:
            p['precise_times'] = True
        gen_params.append(p)

    sgen = nest.Create('spike_generator', len(source_spike_times), params=gen_params)

    nest_sd_params = {
        'start': _to_ms_scalar(sd_params.get('start', 0.0 * u.ms)),
        'origin': _to_ms_scalar(sd_params.get('origin', 0.0 * u.ms)),
        'time_in_steps': bool(sd_params.get('time_in_steps', False)),
    }
    stop = sd_params.get('stop', None)
    if stop is not None:
        nest_sd_params['stop'] = _to_ms_scalar(stop)

    sd = nest.Create('spin_detector', params=nest_sd_params)
    nest.Connect(sgen, sd)
    nest.Simulate(simtime_ms)

    ev = sd.events
    ditype = brainstate.environ.ditype()
    out = {
        'senders': np.asarray(ev['senders'], dtype=ditype),
        'state': np.asarray(ev['state'], dtype=ditype),
    }
    if nest_sd_params['time_in_steps']:
        out['times'] = np.asarray(ev['times'], dtype=ditype)
        out['offsets'] = np.asarray(ev['offsets'], dtype=dftype)
    else:
        out['times'] = np.asarray(ev['times'], dtype=dftype)

    source_ids = np.asarray(sgen.tolist(), dtype=ditype)
    sender_to_times = {
        int(source_ids[i]): np.asarray(source_spike_times[i], dtype=dftype)
        for i in range(len(source_spike_times))
    }

    return out, sender_to_times


class TestSpinDetector(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def test_default_parameters(self):
        sd = spin_detector()
        self.assertTrue(u.math.allclose(sd.start, 0.0 * u.ms))
        self.assertIsNone(sd.stop)
        self.assertTrue(u.math.allclose(sd.origin, 0.0 * u.ms))
        self.assertFalse(sd.time_in_steps)
        self.assertEqual(sd.n_events, 0)

        ev = sd.events
        self.assertEqual(ev['times'].size, 0)
        self.assertEqual(ev['senders'].size, 0)
        self.assertEqual(ev['state'].size, 0)
        self.assertTrue('offsets' not in ev)

    def test_n_events_can_only_be_set_to_zero(self):
        with brainstate.environ.context(dt=self.dt):
            sd = spin_detector()
            for step in range(3):
                with brainstate.environ.context(t=step * self.dt):
                    dftype = brainstate.environ.dftype()
                    ditype = brainstate.environ.ditype()
                    sd.update(spikes=np.array([1.0], dtype=dftype), senders=np.array([1], dtype=ditype))

        self.assertEqual(sd.n_events, 3)
        sd.n_events = 0
        self.assertEqual(sd.n_events, 0)
        self.assertEqual(sd.events['times'].size, 0)

        with self.assertRaises(ValueError):
            sd.n_events = 2

    def test_time_in_steps_records_step_and_offset_and_locks(self):
        with brainstate.environ.context(dt=self.dt):
            sd = spin_detector(time_in_steps=True)
            with brainstate.environ.context(t=0.0 * u.ms):
                dftype = brainstate.environ.dftype()
                ditype = brainstate.environ.ditype()
                sd.update(
                    spikes=np.array([1.0], dtype=dftype),
                    senders=np.array([5], dtype=ditype),
                    offsets=np.array([0.03], dtype=dftype) * u.ms,
                    multiplicities=np.array([1], dtype=ditype),
                )

        ev = sd.events
        npt.assert_array_equal(ev['senders'], np.array([5], dtype=ditype))
        npt.assert_array_equal(ev['state'], np.array([0], dtype=ditype))
        npt.assert_array_equal(ev['times'], np.array([1], dtype=ditype))
        npt.assert_allclose(ev['offsets'], np.array([0.03], dtype=dftype), atol=1e-12)

        with self.assertRaises(ValueError):
            sd.time_in_steps = False

    def test_direct_multiplicity_two_decodes_to_state_one(self):
        with brainstate.environ.context(dt=self.dt):
            sd = spin_detector()
            with brainstate.environ.context(t=0.0 * u.ms):
                dftype = brainstate.environ.dftype()
                ditype = brainstate.environ.ditype()
                sd.update(
                    spikes=np.array([1.0], dtype=dftype),
                    senders=np.array([7], dtype=ditype),
                    multiplicities=np.array([2], dtype=ditype),
                )

        ev = sd.events
        npt.assert_array_equal(ev['senders'], np.array([7], dtype=ditype))
        npt.assert_array_equal(ev['state'], np.array([1], dtype=ditype))
        npt.assert_allclose(ev['times'], np.array([0.1], dtype=dftype), atol=1e-12)

    def test_matches_nest_reference_decoding(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        sd_params = {'time_in_steps': False}
        source_spike_times = [[10.0, 10.0, 15.0]]

        nest_events, sender_to_times = _run_nest_spin_trace(
            simtime_ms=20.0,
            dt_ms=self.dt_ms,
            sd_params=sd_params,
            source_spike_times=source_spike_times,
            precise_times=False,
        )
        bp_events = _run_bp_spin_trace(
            simtime_ms=20.0,
            dt_ms=self.dt_ms,
            sd_params=sd_params,
            sender_to_times=sender_to_times,
        )

        npt.assert_array_equal(bp_events['senders'], nest_events['senders'])
        npt.assert_array_equal(bp_events['state'], nest_events['state'])
        npt.assert_allclose(bp_events['times'], nest_events['times'], atol=1e-12)

    def test_matches_nest_precise_time_in_steps(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        sd_params = {'time_in_steps': True}
        source_spike_times = [[10.03, 10.03, 15.02]]

        nest_events, sender_to_times = _run_nest_spin_trace(
            simtime_ms=20.0,
            dt_ms=self.dt_ms,
            sd_params=sd_params,
            source_spike_times=source_spike_times,
            precise_times=True,
        )
        bp_events = _run_bp_spin_trace(
            simtime_ms=20.0,
            dt_ms=self.dt_ms,
            sd_params=sd_params,
            sender_to_times=sender_to_times,
        )

        npt.assert_array_equal(bp_events['senders'], nest_events['senders'])
        npt.assert_array_equal(bp_events['state'], nest_events['state'])
        npt.assert_array_equal(bp_events['times'], nest_events['times'])
        npt.assert_allclose(bp_events['offsets'], nest_events['offsets'], atol=1e-12)

    def test_matches_nest_concurrent_senders_ordering(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        sd_params = {'time_in_steps': False}
        source_spike_times = [[5.0], [5.0], [5.0]]

        nest_events, sender_to_times = _run_nest_spin_trace(
            simtime_ms=6.0,
            dt_ms=1.0,
            sd_params=sd_params,
            source_spike_times=source_spike_times,
            precise_times=False,
        )
        bp_events = _run_bp_spin_trace(
            simtime_ms=6.0,
            dt_ms=1.0,
            sd_params=sd_params,
            sender_to_times=sender_to_times,
        )

        npt.assert_array_equal(bp_events['senders'], nest_events['senders'])
        npt.assert_array_equal(bp_events['state'], nest_events['state'])
        npt.assert_allclose(bp_events['times'], nest_events['times'], atol=1e-12)

    def test_matches_nest_window_and_origin(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        sd_params = {
            'start': 0.1 * u.ms,
            'stop': 0.3 * u.ms,
            'origin': 0.1 * u.ms,
            'time_in_steps': False,
        }
        source_spike_times = [[0.2, 0.3, 0.3, 0.4, 0.4, 0.5]]

        nest_events, sender_to_times = _run_nest_spin_trace(
            simtime_ms=1.0,
            dt_ms=self.dt_ms,
            sd_params=sd_params,
            source_spike_times=source_spike_times,
            precise_times=False,
        )
        bp_events = _run_bp_spin_trace(
            simtime_ms=1.0,
            dt_ms=self.dt_ms,
            sd_params=sd_params,
            sender_to_times=sender_to_times,
        )

        npt.assert_array_equal(bp_events['senders'], nest_events['senders'])
        npt.assert_array_equal(bp_events['state'], nest_events['state'])
        npt.assert_allclose(bp_events['times'], nest_events['times'], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
