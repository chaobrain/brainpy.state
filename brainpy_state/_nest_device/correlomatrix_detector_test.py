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

from brainpy.state import correlomatrix_detector


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


def _prepare_bp_schedule(spike_times_by_channel, dt_ms):
    schedule = {}
    eps = 1e-12

    for channel, times in enumerate(spike_times_by_channel):
        dftype = brainstate.environ.dftype()
        times = np.asarray(times, dtype=dftype)
        if times.size == 0:
            continue

        uniq_times, counts = np.unique(times, return_counts=True)
        for t_spike, multiplicity in zip(uniq_times, counts):
            step_f = float(t_spike) / float(dt_ms)
            stamp_step = int(np.ceil(step_f - eps))
            if stamp_step <= 0:
                raise ValueError('Spike times must be strictly positive.')
            per_step = schedule.setdefault(stamp_step, [])
            per_step.append((int(channel), int(multiplicity)))

    for stamp_step in schedule:
        schedule[stamp_step].sort(key=lambda x: x[0])
    return schedule


def _run_bp_correlomatrix(
    spike_times_by_channel,
    simtime_ms,
    dt_ms,
    cmd_params=None,
    channel_weights=None,
):
    cmd_params = {} if cmd_params is None else dict(cmd_params)
    n_channels = len(spike_times_by_channel)

    dftype = brainstate.environ.dftype()
    ditype = brainstate.environ.ditype()

    if channel_weights is None:
        channel_weights = np.ones((n_channels,), dtype=dftype)
    else:
        channel_weights = np.asarray(channel_weights, dtype=dftype)
        if channel_weights.size != n_channels:
            raise ValueError('channel_weights length mismatch.')

    n_steps = int(round(simtime_ms / dt_ms))
    dt = dt_ms * u.ms
    schedule = _prepare_bp_schedule(spike_times_by_channel, dt_ms=dt_ms)

    with brainstate.environ.context(dt=dt):
        cmd = correlomatrix_detector(**cmd_params)

        for step in range(n_steps):
            stamp_step = step + 1
            with brainstate.environ.context(t=step * dt):
                if stamp_step in schedule:
                    events = schedule[stamp_step]
                    cmd.update(
                        spikes=np.ones((len(events),), dtype=dftype),
                        receptor_ports=np.asarray([e[0] for e in events], dtype=ditype),
                        weights=np.asarray([channel_weights[e[0]] for e in events], dtype=dftype),
                        multiplicities=np.asarray([e[1] for e in events], dtype=ditype),
                        stamp_steps=np.full((len(events),), stamp_step, dtype=ditype),
                    )
                else:
                    cmd.update()

    return {
        'covariance': np.asarray(cmd.get('covariance'), dtype=dftype),
        'count_covariance': np.asarray(cmd.get('count_covariance'), dtype=ditype),
        'n_events': np.asarray(cmd.get('n_events'), dtype=ditype),
    }


def _to_nest_cmd_params(cmd_params):
    out = {}
    for key, value in cmd_params.items():
        if key in ('delta_tau', 'tau_max', 'Tstart', 'Tstop', 'start', 'stop', 'origin'):
            if value is None and key in ('stop', 'Tstop'):
                continue
            out[key] = _to_ms_scalar(value)
        else:
            out[key] = value
    return out


def _run_nest_correlomatrix(
    spike_times_by_channel,
    simtime_ms,
    dt_ms,
    cmd_params=None,
    channel_weights=None,
    channel_delays=None,
):
    import nest

    cmd_params = {} if cmd_params is None else dict(cmd_params)
    n_channels = len(spike_times_by_channel)

    dftype = brainstate.environ.dftype()

    if channel_weights is None:
        channel_weights = np.ones((n_channels,), dtype=dftype)
    else:
        channel_weights = np.asarray(channel_weights, dtype=dftype)

    if channel_delays is None:
        channel_delays = np.ones((n_channels,), dtype=dftype)
    else:
        channel_delays = np.asarray(channel_delays, dtype=dftype)

    nest.ResetKernel()
    nest.resolution = dt_ms

    cmd = nest.Create('correlomatrix_detector', params=_to_nest_cmd_params(cmd_params))

    for channel, times in enumerate(spike_times_by_channel):
        sg = nest.Create(
            'spike_generator',
            params={
                'spike_times': list(np.asarray(times, dtype=dftype)),
                'precise_times': False,
            },
        )
        nest.Connect(
            sg,
            cmd,
            syn_spec={
                'receptor_type': int(channel),
                'weight': float(channel_weights[channel]),
                'delay': float(channel_delays[channel]),
            },
        )

    nest.Simulate(float(simtime_ms))

    ditype = brainstate.environ.ditype()
    return {
        'covariance': np.asarray(cmd.get('covariance'), dtype=dftype),
        'count_covariance': np.asarray(cmd.get('count_covariance'), dtype=ditype),
        'n_events': np.asarray(cmd.get('n_events'), dtype=ditype),
    }


class TestCorrelomatrixDetector(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def test_default_parameters_and_matrix_shape(self):
        with brainstate.environ.context(dt=self.dt):
            cmd = correlomatrix_detector()
            with brainstate.environ.context(t=0.0 * u.ms):
                cmd.update()

        self.assertAlmostEqual(cmd.get('delta_tau'), 0.5, places=12)
        self.assertAlmostEqual(cmd.get('tau_max'), 5.0, places=12)
        self.assertEqual(cmd.get('N_channels'), 1)
        self.assertEqual(cmd.get('covariance').shape, (1, 1, 11))
        self.assertEqual(cmd.get('count_covariance').shape, (1, 1, 11))
        ditype = brainstate.environ.ditype()
        npt.assert_array_equal(cmd.get('n_events'), np.array([0], dtype=ditype))
        dftype = brainstate.environ.dftype()
        npt.assert_allclose(cmd.get('covariance'), np.zeros((1, 1, 11), dtype=dftype), atol=1e-12)
        npt.assert_array_equal(cmd.get('count_covariance'), np.zeros((1, 1, 11), dtype=ditype))

    def test_validation_rules(self):
        with brainstate.environ.context(dt=self.dt):
            with brainstate.environ.context(t=0.0 * u.ms):
                with self.assertRaises(ValueError):
                    correlomatrix_detector(N_channels=0).update()

                with self.assertRaises(ValueError):
                    correlomatrix_detector(delta_tau=0.25 * u.ms).update()

                with self.assertRaises(ValueError):
                    correlomatrix_detector(delta_tau=1.0 * u.ms).update()

                with self.assertRaises(ValueError):
                    correlomatrix_detector(delta_tau=0.3 * u.ms, tau_max=1.0 * u.ms).update()

                with self.assertRaises(ValueError):
                    dftype = brainstate.environ.dftype()
                    ditype = brainstate.environ.ditype()
                    correlomatrix_detector(delta_tau=0.3 * u.ms, tau_max=0.9 * u.ms, N_channels=2).update(
                        spikes=np.array([1.0], dtype=dftype),
                        receptor_ports=np.array([2], dtype=ditype),
                    )

    def test_known_covariance_from_nest_testsuite(self):
        # Reference from NEST testsuite/pytests/sli2py_other/test_corr_matrix_det.py
        spike_times = [[1.5, 2.5, 4.5], [0.5, 2.5]]

        out = _run_bp_correlomatrix(
            spike_times_by_channel=spike_times,
            simtime_ms=8.0,
            dt_ms=self.dt_ms,
            cmd_params={'delta_tau': 0.5 * u.ms, 'tau_max': 2.0 * u.ms, 'N_channels': 2},
        )

        ditype = brainstate.environ.ditype()
        npt.assert_array_equal(out['n_events'], np.array([3, 2], dtype=ditype))

        npt.assert_array_equal(out['count_covariance'][0, 1], np.array([1, 0, 1, 0, 2], dtype=ditype))
        npt.assert_array_equal(out['count_covariance'][1, 0], np.array([1, 0, 1, 0, 0], dtype=ditype))
        npt.assert_array_equal(out['count_covariance'][0, 0], np.array([3, 0, 1, 0, 1], dtype=ditype))
        npt.assert_array_equal(out['count_covariance'][1, 1], np.array([2, 0, 0, 0, 1], dtype=ditype))

        dftype = brainstate.environ.dftype()
        npt.assert_allclose(out['covariance'][0, 1], np.array([1, 0, 1, 0, 2], dtype=dftype), atol=1e-12)
        npt.assert_allclose(out['covariance'][1, 0], np.array([1, 0, 1, 0, 0], dtype=dftype), atol=1e-12)

    def test_reset_on_n_channels_change(self):
        with brainstate.environ.context(dt=self.dt):
            cmd = correlomatrix_detector(delta_tau=0.5 * u.ms, tau_max=2.0 * u.ms, N_channels=2)
            with brainstate.environ.context(t=0.0 * u.ms):
                dftype = brainstate.environ.dftype()
                ditype = brainstate.environ.ditype()
                cmd.update(
                    spikes=np.array([1.0, 1.0], dtype=dftype),
                    receptor_ports=np.array([0, 1], dtype=ditype),
                )
            self.assertTrue(np.any(cmd.get('n_events') > 0))

            cmd.N_channels = 3
            with brainstate.environ.context(t=0.1 * u.ms):
                cmd.update()

            self.assertEqual(cmd.get('N_channels'), 3)
            self.assertEqual(cmd.get('covariance').shape, (3, 3, 5))
            npt.assert_array_equal(cmd.get('n_events'), np.zeros((3,), dtype=ditype))
            npt.assert_allclose(cmd.get('covariance'), np.zeros((3, 3, 5), dtype=dftype), atol=1e-12)
            npt.assert_array_equal(cmd.get('count_covariance'), np.zeros((3, 3, 5), dtype=ditype))

    def test_matches_nest_weighted_covariance_and_windows(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        spike_times = [
            [0.2, 0.9, 1.8, 2.7],
            [0.4, 1.3, 2.2, 3.1],
            [0.6, 1.6, 2.5, 3.0],
        ]
        cmd_params = {
            'delta_tau': 0.3 * u.ms,
            'tau_max': 0.9 * u.ms,
            'N_channels': 3,
            'start': 0.1 * u.ms,
            'stop': 3.2 * u.ms,
            'origin': 0.0 * u.ms,
            'Tstart': 0.4 * u.ms,
            'Tstop': 2.8 * u.ms,
        }
        dftype = brainstate.environ.dftype()
        channel_weights = np.array([1.7, -0.8, 0.5], dtype=dftype)
        channel_delays = np.array([1.0, 1.0, 1.0], dtype=dftype)

        bp_out = _run_bp_correlomatrix(
            spike_times_by_channel=spike_times,
            simtime_ms=5.0,
            dt_ms=self.dt_ms,
            cmd_params=cmd_params,
            channel_weights=channel_weights,
        )
        nest_out = _run_nest_correlomatrix(
            spike_times_by_channel=spike_times,
            simtime_ms=5.0,
            dt_ms=self.dt_ms,
            cmd_params=cmd_params,
            channel_weights=channel_weights,
            channel_delays=channel_delays,
        )

        npt.assert_array_equal(bp_out['n_events'], nest_out['n_events'])
        npt.assert_array_equal(bp_out['count_covariance'], nest_out['count_covariance'])
        npt.assert_allclose(bp_out['covariance'], nest_out['covariance'], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
