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
import brainunit as u
import jax
import numpy as np
import numpy.testing as npt

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import iaf_psc_delta, spike_recorder


def _to_ms_scalar(value):
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        value = value / u.ms
    arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
    if arr.size != 1:
        raise ValueError('Expected scalar time value.')
    return float(arr[0])


def _run_bp_iaf_trace(simtime_ms, dt_ms, sr_params, i_e_pA):
    n_steps = int(round(simtime_ms / dt_ms))
    dt = dt_ms * u.ms

    with brainstate.environ.context(dt=dt):
        neuron = iaf_psc_delta(1, I_e=i_e_pA * u.pA)
        neuron.init_state()
        sr = spike_recorder(**sr_params)

        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt):
                spk = neuron.update()
                spk_np = np.asarray(u.math.asarray(spk), dtype=np.float64).reshape(-1)
                sr.update(spikes=spk_np, senders=np.array([1], dtype=np.int64))

    return sr.events


def _run_nest_iaf_trace(simtime_ms, dt_ms, sr_params, i_e_pA):
    import nest

    nest.ResetKernel()
    nest.resolution = dt_ms

    neuron = nest.Create('iaf_psc_delta', params={'I_e': float(i_e_pA)})

    nest_sr_params = {
        'start': _to_ms_scalar(sr_params.get('start', 0.0 * u.ms)),
        'origin': _to_ms_scalar(sr_params.get('origin', 0.0 * u.ms)),
        'time_in_steps': bool(sr_params.get('time_in_steps', False)),
    }
    stop = sr_params.get('stop', None)
    if stop is not None:
        nest_sr_params['stop'] = _to_ms_scalar(stop)

    sr = nest.Create('spike_recorder', params=nest_sr_params)

    # delay and weight are ignored by NEST spike_recorder; keep non-defaults
    # to verify matching behavior.
    nest.Connect(neuron, sr, syn_spec={'weight': 5.0, 'delay': 0.3})
    nest.Simulate(simtime_ms)

    ev = sr.events
    out = {
        'senders': np.asarray(ev['senders'], dtype=np.int64),
    }
    if nest_sr_params['time_in_steps']:
        out['times'] = np.asarray(ev['times'], dtype=np.int64)
        out['offsets'] = np.asarray(ev['offsets'], dtype=np.float64)
    else:
        out['times'] = np.asarray(ev['times'], dtype=np.float64)
    return out


def _run_bp_precise_source(spike_times_ms, simtime_ms, dt_ms, sr_params):
    spike_times_ms = np.asarray(spike_times_ms, dtype=np.float64)
    n_steps = int(round(simtime_ms / dt_ms))
    dt = dt_ms * u.ms
    eps = 1e-12

    with brainstate.environ.context(dt=dt):
        sr = spike_recorder(**sr_params)
        idx = 0

        for step in range(n_steps):
            t_start = step * dt_ms
            t_end = (step + 1) * dt_ms

            step_offsets = []
            while idx < spike_times_ms.size and spike_times_ms[idx] <= t_end + eps:
                st = float(spike_times_ms[idx])
                if st > t_start + eps:
                    step_offsets.append(t_end - st)
                idx += 1

            with brainstate.environ.context(t=step * dt):
                if step_offsets:
                    n_ev = len(step_offsets)
                    sr.update(
                        spikes=np.ones((n_ev,), dtype=np.float64),
                        senders=np.ones((n_ev,), dtype=np.int64),
                        offsets=np.asarray(step_offsets, dtype=np.float64) * u.ms,
                    )
                else:
                    sr.update()

    return sr.events


def _run_nest_precise_source(spike_times_ms, simtime_ms, dt_ms, sr_params):
    import nest

    nest.ResetKernel()
    nest.resolution = dt_ms

    sgen = nest.Create('spike_generator', params={
        'spike_times': list(np.asarray(spike_times_ms, dtype=np.float64)),
        'precise_times': True,
    })

    nest_sr_params = {
        'start': _to_ms_scalar(sr_params.get('start', 0.0 * u.ms)),
        'origin': _to_ms_scalar(sr_params.get('origin', 0.0 * u.ms)),
        'time_in_steps': bool(sr_params.get('time_in_steps', False)),
    }
    stop = sr_params.get('stop', None)
    if stop is not None:
        nest_sr_params['stop'] = _to_ms_scalar(stop)

    sr = nest.Create('spike_recorder', params=nest_sr_params)
    nest.Connect(sgen, sr)
    nest.Simulate(simtime_ms)

    ev = sr.events
    out = {
        'senders': np.asarray(ev['senders'], dtype=np.int64),
    }
    if nest_sr_params['time_in_steps']:
        out['times'] = np.asarray(ev['times'], dtype=np.int64)
        out['offsets'] = np.asarray(ev['offsets'], dtype=np.float64)
    else:
        out['times'] = np.asarray(ev['times'], dtype=np.float64)
    return out


class TestSpikeRecorder(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def setUp(self):
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def test_default_parameters(self):
        sr = spike_recorder()
        self.assertTrue(u.math.allclose(sr.start, 0.0 * u.ms))
        self.assertIsNone(sr.stop)
        self.assertTrue(u.math.allclose(sr.origin, 0.0 * u.ms))
        self.assertFalse(sr.time_in_steps)
        self.assertEqual(sr.n_events, 0)

        ev = sr.events
        self.assertEqual(ev['times'].size, 0)
        self.assertEqual(ev['senders'].size, 0)
        self.assertTrue('offsets' not in ev)

    def test_window_is_start_exclusive_and_stop_inclusive(self):
        with brainstate.environ.context(dt=self.dt):
            sr = spike_recorder(start=0.5 * u.ms, stop=1.0 * u.ms)
            for step in range(12):
                with brainstate.environ.context(t=step * self.dt):
                    sr.update(spikes=np.array([1.0], dtype=np.float64), senders=np.array([9], dtype=np.int64))

        ev = sr.events
        expected_times = np.array([0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
        npt.assert_allclose(ev['times'], expected_times, atol=1e-12)
        npt.assert_array_equal(ev['senders'], np.full(expected_times.shape, 9, dtype=np.int64))

    def test_n_events_can_only_be_set_to_zero(self):
        with brainstate.environ.context(dt=self.dt):
            sr = spike_recorder()
            for step in range(3):
                with brainstate.environ.context(t=step * self.dt):
                    sr.update(spikes=np.array([1.0], dtype=np.float64), senders=np.array([1], dtype=np.int64))

        self.assertEqual(sr.n_events, 3)
        sr.n_events = 0
        self.assertEqual(sr.n_events, 0)
        self.assertEqual(sr.events['times'].size, 0)

        with self.assertRaises(ValueError):
            sr.n_events = 2

    def test_time_in_steps_records_step_and_offset_and_locks(self):
        with brainstate.environ.context(dt=self.dt):
            sr = spike_recorder(time_in_steps=True)
            with brainstate.environ.context(t=0.0 * u.ms):
                sr.update(
                    spikes=np.array([1.0], dtype=np.float64),
                    senders=np.array([5], dtype=np.int64),
                    offsets=np.array([0.03], dtype=np.float64) * u.ms,
                )

        ev = sr.events
        npt.assert_array_equal(ev['senders'], np.array([5], dtype=np.int64))
        npt.assert_array_equal(ev['times'], np.array([1], dtype=np.int64))
        npt.assert_allclose(ev['offsets'], np.array([0.03], dtype=np.float64), atol=1e-12)

        with self.assertRaises(ValueError):
            sr.time_in_steps = False

    def test_multiplicity_repeats_event_writes(self):
        with brainstate.environ.context(dt=self.dt):
            sr = spike_recorder()
            with brainstate.environ.context(t=0.0 * u.ms):
                sr.update(
                    spikes=np.array([1.0], dtype=np.float64),
                    senders=np.array([7], dtype=np.int64),
                    offsets=np.array([0.01], dtype=np.float64) * u.ms,
                    multiplicities=np.array([3], dtype=np.int64),
                )

        ev = sr.events
        npt.assert_array_equal(ev['senders'], np.array([7, 7, 7], dtype=np.int64))
        npt.assert_allclose(ev['times'], np.array([0.09, 0.09, 0.09], dtype=np.float64), atol=1e-12)

    def test_matches_nest_iaf_trace(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        sr_params = {
            'start': 0.0 * u.ms,
            'stop': None,
            'origin': 0.0 * u.ms,
            'time_in_steps': False,
        }

        bp_events = _run_bp_iaf_trace(
            simtime_ms=120.0,
            dt_ms=self.dt_ms,
            sr_params=sr_params,
            i_e_pA=500.0,
        )
        nest_events = _run_nest_iaf_trace(
            simtime_ms=120.0,
            dt_ms=self.dt_ms,
            sr_params=sr_params,
            i_e_pA=500.0,
        )

        npt.assert_allclose(bp_events['times'], nest_events['times'], atol=1e-12)
        npt.assert_array_equal(bp_events['senders'], nest_events['senders'])

    def test_matches_nest_time_window_and_time_in_steps(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        sr_params = {
            'start': 10.0 * u.ms,
            'stop': 60.0 * u.ms,
            'origin': 5.0 * u.ms,
            'time_in_steps': True,
        }

        bp_events = _run_bp_iaf_trace(
            simtime_ms=120.0,
            dt_ms=self.dt_ms,
            sr_params=sr_params,
            i_e_pA=500.0,
        )
        nest_events = _run_nest_iaf_trace(
            simtime_ms=120.0,
            dt_ms=self.dt_ms,
            sr_params=sr_params,
            i_e_pA=500.0,
        )

        npt.assert_array_equal(bp_events['times'], nest_events['times'])
        npt.assert_array_equal(bp_events['senders'], nest_events['senders'])
        npt.assert_allclose(bp_events['offsets'], nest_events['offsets'], atol=1e-12)

    def test_precise_source_matches_nest_across_resolutions(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        spike_times = np.array([0.1, 5.0, 5.3, 5.33, 5.4, 5.9, 6.0], dtype=np.float64)
        for resolution in (1.0, 0.1, 0.02, 0.01, 0.001):
            sr_params = {'time_in_steps': False}
            bp_events = _run_bp_precise_source(
                spike_times_ms=spike_times,
                simtime_ms=8.0,
                dt_ms=resolution,
                sr_params=sr_params,
            )
            nest_events = _run_nest_precise_source(
                spike_times_ms=spike_times,
                simtime_ms=8.0,
                dt_ms=resolution,
                sr_params=sr_params,
            )
            npt.assert_allclose(bp_events['times'], spike_times, atol=1e-12)
            npt.assert_allclose(nest_events['times'], spike_times, atol=1e-12)
            npt.assert_allclose(bp_events['times'], nest_events['times'], atol=1e-12)

    def test_precise_source_time_in_steps_matches_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.1
        spike_times = np.array([0.1, 5.0, 5.3, 5.33, 5.4, 5.9, 6.0], dtype=np.float64)
        sr_params = {'time_in_steps': True}

        bp_events = _run_bp_precise_source(
            spike_times_ms=spike_times,
            simtime_ms=8.0,
            dt_ms=dt_ms,
            sr_params=sr_params,
        )
        nest_events = _run_nest_precise_source(
            spike_times_ms=spike_times,
            simtime_ms=8.0,
            dt_ms=dt_ms,
            sr_params=sr_params,
        )

        npt.assert_array_equal(bp_events['times'], nest_events['times'])
        npt.assert_allclose(bp_events['offsets'], nest_events['offsets'], atol=1e-12)
        npt.assert_array_equal(bp_events['senders'], nest_events['senders'])


if __name__ == '__main__':
    unittest.main()
