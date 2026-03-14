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

from brainpy.state import correlation_detector


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


def _prepare_bp_schedule(spike_times_by_port, dt_ms):
    schedule = {}
    eps = 1e-12
    for port, times in enumerate(spike_times_by_port):
        dftype = brainstate.environ.dftype()
        times = np.asarray(times, dtype=dftype)
        for t_spike in times:
            step_f = float(t_spike) / float(dt_ms)
            stamp_step = int(np.ceil(step_f - eps))
            if stamp_step <= 0:
                raise ValueError('Spike times must be strictly positive.')
            per_step = schedule.setdefault(stamp_step, [])
            per_step.append((int(port), 1))

    for stamp_step in schedule:
        schedule[stamp_step].sort(key=lambda x: x[0])
    return schedule


def _run_bp_corrdet(
    spike_times_by_port,
    simtime_ms,
    dt_ms,
    cd_params=None,
    port_weights=(1.0, 1.0),
):
    cd_params = {} if cd_params is None else dict(cd_params)
    n_steps = int(round(simtime_ms / dt_ms))
    dt = dt_ms * u.ms
    schedule = _prepare_bp_schedule(spike_times_by_port, dt_ms=dt_ms)

    with brainstate.environ.context(dt=dt):
        cd = correlation_detector(**cd_params)
        for step in range(n_steps):
            stamp_step = step + 1
            with brainstate.environ.context(t=step * dt):
                if stamp_step in schedule:
                    events = schedule[stamp_step]
                    ditype = brainstate.environ.ditype()
                    ports = np.asarray([e[0] for e in events], dtype=ditype)
                    multiplicities = np.asarray([e[1] for e in events], dtype=ditype)
                    dftype = brainstate.environ.dftype()
                    weights = np.asarray([port_weights[e[0]] for e in events], dtype=dftype)
                    cd.update(
                        spikes=np.ones((len(events),), dtype=dftype),
                        receptor_ports=ports,
                        weights=weights,
                        multiplicities=multiplicities,
                        stamp_steps=np.full((len(events),), stamp_step, dtype=ditype),
                    )
                else:
                    cd.update()

    return {
        'histogram': np.asarray(cd.get('histogram'), dtype=dftype),
        'count_histogram': np.asarray(cd.get('count_histogram'), dtype=ditype),
        'n_events': np.asarray(cd.get('n_events'), dtype=ditype),
    }


def _to_nest_cd_params(cd_params):
    out = {}
    for key, value in cd_params.items():
        if key in ('delta_tau', 'tau_max', 'Tstart', 'Tstop', 'start', 'stop', 'origin'):
            if value is None and key in ('stop', 'Tstop'):
                continue
            out[key] = _to_ms_scalar(value)
        else:
            out[key] = value
    return out


def _run_nest_corrdet(
    spike_times_by_port,
    simtime_ms,
    dt_ms,
    cd_params=None,
    port_weights=(1.0, 1.0),
    port_delays=(1.0, 1.0),
):
    import nest

    cd_params = {} if cd_params is None else dict(cd_params)

    nest.ResetKernel()
    nest.resolution = dt_ms

    cd = nest.Create('correlation_detector', params=_to_nest_cd_params(cd_params))

    dftype = brainstate.environ.dftype()
    sg1 = nest.Create(
        'spike_generator',
        params={
            'spike_times': list(np.asarray(spike_times_by_port[0], dtype=dftype)),
            'precise_times': False,
        },
    )
    sg2 = nest.Create(
        'spike_generator',
        params={
            'spike_times': list(np.asarray(spike_times_by_port[1], dtype=dftype)),
            'precise_times': False,
        },
    )

    nest.Connect(
        sg1,
        cd,
        syn_spec={
            'receptor_type': 0,
            'weight': float(port_weights[0]),
            'delay': float(port_delays[0]),
        },
    )
    nest.Connect(
        sg2,
        cd,
        syn_spec={
            'receptor_type': 1,
            'weight': float(port_weights[1]),
            'delay': float(port_delays[1]),
        },
    )

    nest.Simulate(float(simtime_ms))

    ditype = brainstate.environ.ditype()
    return {
        'histogram': np.asarray(cd.get('histogram'), dtype=dftype),
        'count_histogram': np.asarray(cd.get('count_histogram'), dtype=ditype),
        'n_events': np.asarray(cd.get('n_events'), dtype=ditype),
    }


class TestCorrelationDetector(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def test_default_parameters_and_histogram_size(self):
        with brainstate.environ.context(dt=self.dt):
            cd = correlation_detector()
            with brainstate.environ.context(t=0.0 * u.ms):
                cd.update()

        self.assertAlmostEqual(cd.get('delta_tau'), 0.5, places=12)
        self.assertAlmostEqual(cd.get('tau_max'), 5.0, places=12)
        self.assertEqual(cd.get('histogram').size, 21)
        ditype = brainstate.environ.ditype()
        npt.assert_array_equal(cd.get('n_events'), np.array([0, 0], dtype=ditype))
        npt.assert_array_equal(cd.get('count_histogram'), np.zeros((21,), dtype=ditype))
        dftype = brainstate.environ.dftype()
        npt.assert_allclose(cd.get('histogram'), np.zeros((21,), dtype=dftype), atol=1e-12)

    def test_validation_rules(self):
        with brainstate.environ.context(dt=self.dt):
            with brainstate.environ.context(t=0.0 * u.ms):
                with self.assertRaises(ValueError):
                    correlation_detector(delta_tau=0.25 * u.ms).update()

                with self.assertRaises(ValueError):
                    correlation_detector(delta_tau=1.0 * u.ms, tau_max=2.5 * u.ms).update()

                with self.assertRaises(ValueError):
                    dftype = brainstate.environ.dftype()
                    ditype = brainstate.environ.ditype()
                    correlation_detector(delta_tau=0.1 * u.ms, tau_max=1.0 * u.ms).update(
                        spikes=np.array([1.0], dtype=dftype),
                        receptor_ports=np.array([2], dtype=ditype),
                    )

    def test_known_histograms_from_nest_testsuite(self):
        # Reference from NEST testsuite/pytests/sli2py_recording/test_corr_det.py
        cases = [
            (
                [[1.0, 2.0, 6.0], [2.0, 4.0]],
                [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
            ),
            (
                [[6.0], [0.5, 5.4, 5.5, 5.6, 6.4, 6.5, 6.6, 11.5]],
                [1, 0, 0, 0, 1, 3, 2, 0, 0, 0, 0],
            ),
        ]

        for spike_times, expected_hist in cases:
            simtime = max(max(spike_times[0]), max(spike_times[1])) + 2.0
            out = _run_bp_corrdet(
                spike_times_by_port=spike_times,
                simtime_ms=simtime,
                dt_ms=self.dt_ms,
                cd_params={'delta_tau': 1.0 * u.ms, 'tau_max': 5.0 * u.ms},
            )

            ditype = brainstate.environ.ditype()
            expected_hist_arr = np.asarray(expected_hist, dtype=ditype)
            npt.assert_array_equal(out['count_histogram'], expected_hist_arr)
            npt.assert_allclose(out['histogram'], expected_hist_arr.astype(np.float64), atol=1e-12)
            npt.assert_array_equal(
                out['n_events'],
                np.array([len(spike_times[0]), len(spike_times[1])], dtype=ditype),
            )

    def test_n_events_reset_semantics(self):
        with brainstate.environ.context(dt=self.dt):
            cd = correlation_detector(delta_tau=1.0 * u.ms, tau_max=5.0 * u.ms)
            with brainstate.environ.context(t=0.0 * u.ms):
                dftype = brainstate.environ.dftype()
                ditype = brainstate.environ.ditype()
                cd.update(
                    spikes=np.array([1.0, 1.0], dtype=dftype),
                    receptor_ports=np.array([0, 1], dtype=ditype),
                )

            self.assertTrue(np.any(cd.get('n_events') > 0))
            cd.n_events = np.array([0, 0], dtype=ditype)
            npt.assert_array_equal(cd.get('n_events'), np.array([0, 0], dtype=ditype))
            npt.assert_allclose(cd.get('histogram'), np.zeros((11,), dtype=dftype), atol=1e-12)
            npt.assert_array_equal(cd.get('count_histogram'), np.zeros((11,), dtype=ditype))

            with self.assertRaises(ValueError):
                cd.n_events = np.array([1, 1], dtype=ditype)

    def test_matches_nest_weighted_histogram_and_windows(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        spike_times = [
            [0.2, 0.3, 1.0, 2.4, 2.9, 3.2],
            [0.2, 1.0, 1.5, 2.4, 3.2],
        ]
        cd_params = {
            'delta_tau': 0.2 * u.ms,
            'tau_max': 0.6 * u.ms,
            'start': 0.1 * u.ms,
            'stop': 3.2 * u.ms,
            'origin': 0.0 * u.ms,
            'Tstart': 0.5 * u.ms,
            'Tstop': 2.8 * u.ms,
        }
        port_weights = (1.7, -0.8)
        port_delays = (1.9, 0.4)

        bp_out = _run_bp_corrdet(
            spike_times_by_port=spike_times,
            simtime_ms=5.0,
            dt_ms=self.dt_ms,
            cd_params=cd_params,
            port_weights=port_weights,
        )
        nest_out = _run_nest_corrdet(
            spike_times_by_port=spike_times,
            simtime_ms=5.0,
            dt_ms=self.dt_ms,
            cd_params=cd_params,
            port_weights=port_weights,
            port_delays=port_delays,
        )

        npt.assert_array_equal(bp_out['n_events'], nest_out['n_events'])
        npt.assert_array_equal(bp_out['count_histogram'], nest_out['count_histogram'])
        npt.assert_allclose(bp_out['histogram'], nest_out['histogram'], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
