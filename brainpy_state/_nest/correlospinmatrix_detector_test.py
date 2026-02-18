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

from brainpy.state import correlospinmatrix_detector


def _to_ms_scalar(value):
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        value = value / u.ms
    arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
    if arr.size != 1:
        raise ValueError('Expected scalar time value.')
    return float(arr[0])


def _prepare_bp_schedule(spike_times_by_channel, dt_ms):
    schedule = {}
    eps = 1e-12

    for channel, times in enumerate(spike_times_by_channel):
        times = np.asarray(times, dtype=np.float64)
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


def _run_bp_correlospinmatrix(
    spike_times_by_channel,
    simtime_ms,
    dt_ms,
    cmd_params=None,
):
    cmd_params = {} if cmd_params is None else dict(cmd_params)

    n_steps = int(round(simtime_ms / dt_ms))
    dt = dt_ms * u.ms
    schedule = _prepare_bp_schedule(spike_times_by_channel, dt_ms=dt_ms)

    with brainstate.environ.context(dt=dt):
        cmd = correlospinmatrix_detector(**cmd_params)

        for step in range(n_steps):
            stamp_step = step + 1
            with brainstate.environ.context(t=step * dt):
                if stamp_step in schedule:
                    events = schedule[stamp_step]
                    cmd.update(
                        spikes=np.ones((len(events),), dtype=np.float64),
                        receptor_ports=np.asarray([e[0] for e in events], dtype=np.int64),
                        multiplicities=np.asarray([e[1] for e in events], dtype=np.int64),
                        stamp_steps=np.full((len(events),), stamp_step, dtype=np.int64),
                    )
                else:
                    cmd.update()

    return {
        'count_covariance': np.asarray(cmd.get('count_covariance'), dtype=np.int64),
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


def _run_nest_correlospinmatrix(
    spike_times_by_channel,
    simtime_ms,
    dt_ms,
    cmd_params=None,
):
    import nest

    cmd_params = {} if cmd_params is None else dict(cmd_params)

    nest.ResetKernel()
    nest.resolution = dt_ms

    cmd = nest.Create('correlospinmatrix_detector', params=_to_nest_cmd_params(cmd_params))

    for channel, times in enumerate(spike_times_by_channel):
        sg = nest.Create(
            'spike_generator',
            params={
                'spike_times': list(np.asarray(times, dtype=np.float64)),
                'precise_times': False,
            },
        )
        nest.Connect(
            sg,
            cmd,
            syn_spec={
                'receptor_type': int(channel),
            },
        )

    nest.Simulate(float(simtime_ms))

    return {
        'count_covariance': np.asarray(cmd.get('count_covariance'), dtype=np.int64),
    }


class TestCorrelospinmatrixDetector(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def test_default_parameters_and_matrix_shape(self):
        with brainstate.environ.context(dt=self.dt):
            cmd = correlospinmatrix_detector()
            with brainstate.environ.context(t=0.0 * u.ms):
                cmd.update()

        self.assertAlmostEqual(cmd.get('delta_tau'), 0.1, places=12)
        self.assertAlmostEqual(cmd.get('tau_max'), 1.0, places=12)
        self.assertEqual(cmd.get('N_channels'), 1)
        self.assertEqual(cmd.get('count_covariance').shape, (1, 1, 21))
        npt.assert_array_equal(cmd.get('count_covariance'), np.zeros((1, 1, 21), dtype=np.int64))

    def test_validation_rules(self):
        with brainstate.environ.context(dt=self.dt):
            with brainstate.environ.context(t=0.0 * u.ms):
                with self.assertRaises(ValueError):
                    correlospinmatrix_detector(N_channels=0).update()

                with self.assertRaises(ValueError):
                    correlospinmatrix_detector(delta_tau=-0.1 * u.ms).update()

                with self.assertRaises(ValueError):
                    correlospinmatrix_detector(delta_tau=0.15 * u.ms).update()

                with self.assertRaises(ValueError):
                    correlospinmatrix_detector(delta_tau=0.3 * u.ms, tau_max=1.0 * u.ms).update()

                with self.assertRaises(ValueError):
                    correlospinmatrix_detector(Tstart=-1.0 * u.ms).update()

                with self.assertRaises(ValueError):
                    correlospinmatrix_detector(Tstop=-1.0 * u.ms).update()

                with self.assertRaises(ValueError):
                    correlospinmatrix_detector(N_channels=2).update(
                        spikes=np.array([1.0], dtype=np.float64),
                        receptor_ports=np.array([2], dtype=np.int64),
                    )

    def test_reset_on_parameter_change(self):
        cmd_params = {
            'N_channels': 3,
            'tau_max': 10.0 * u.ms,
            'delta_tau': 1.0 * u.ms,
        }
        spike_times = [
            [10.0, 10.0, 16.0],
            [15.0, 15.0, 20.0],
            [25.0],
        ]

        with brainstate.environ.context(dt=self.dt):
            cmd = correlospinmatrix_detector(**cmd_params)
            schedule = _prepare_bp_schedule(spike_times, dt_ms=self.dt_ms)
            for step in range(int(round(100.0 / self.dt_ms))):
                stamp_step = step + 1
                with brainstate.environ.context(t=step * self.dt):
                    if stamp_step in schedule:
                        events = schedule[stamp_step]
                        cmd.update(
                            spikes=np.ones((len(events),), dtype=np.float64),
                            receptor_ports=np.asarray([e[0] for e in events], dtype=np.int64),
                            multiplicities=np.asarray([e[1] for e in events], dtype=np.int64),
                            stamp_steps=np.full((len(events),), stamp_step, dtype=np.int64),
                        )
                    else:
                        cmd.update()

            self.assertTrue(np.any(cmd.get('count_covariance') > 0))

            cmd.tau_max = 5.0 * u.ms
            with brainstate.environ.context(t=100.0 * u.ms):
                cmd.update()

            self.assertEqual(cmd.get('count_covariance').shape, (3, 3, 11))
            npt.assert_array_equal(cmd.get('count_covariance'), np.zeros((3, 3, 11), dtype=np.int64))

    def test_known_count_covariance_from_nest_testsuite(self):
        expected_corr = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0, 0, 0, 0, 0],
                    [0, 10, 20, 30, 40, 50, 50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30, 40, 50, 50, 40, 30, 20, 10, 0],
                    [0, 0, 0, 0, 0, 0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=np.int64,
        )

        spike_times = [
            [10.0, 10.0, 16.0],
            [15.0, 15.0, 20.0],
            [25.0],
        ]
        out = _run_bp_correlospinmatrix(
            spike_times_by_channel=spike_times,
            simtime_ms=100.0,
            dt_ms=self.dt_ms,
            cmd_params={'N_channels': 3, 'tau_max': 10.0 * u.ms, 'delta_tau': 1.0 * u.ms},
        )

        npt.assert_array_equal(out['count_covariance'], expected_corr)

    def test_matches_nest_reference_with_tstart_tstop(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        spike_times = [
            [10.0, 10.0, 16.0],
            [15.0, 15.0, 20.0],
            [25.0],
        ]
        cmd_params = {
            'N_channels': 3,
            'tau_max': 10.0 * u.ms,
            'delta_tau': 1.0 * u.ms,
            'Tstart': 50.0 * u.ms,
            'Tstop': 60.0 * u.ms,
        }

        bp_out = _run_bp_correlospinmatrix(
            spike_times_by_channel=spike_times,
            simtime_ms=100.0,
            dt_ms=self.dt_ms,
            cmd_params=cmd_params,
        )
        nest_out = _run_nest_correlospinmatrix(
            spike_times_by_channel=spike_times,
            simtime_ms=100.0,
            dt_ms=self.dt_ms,
            cmd_params=cmd_params,
        )

        npt.assert_array_equal(bp_out['count_covariance'], nest_out['count_covariance'])


if __name__ == '__main__':
    unittest.main()
