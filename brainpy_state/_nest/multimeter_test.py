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

from brainpy.state import iaf_psc_delta, multimeter


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


def _run_bp_vm_trace(simtime_ms, dt_ms, mm_params, i_e_pA):
    n_steps = int(round(simtime_ms / dt_ms))
    dt = dt_ms * u.ms

    with brainstate.environ.context(dt=dt):
        neuron = iaf_psc_delta(1, I_e=i_e_pA * u.pA)
        neuron.init_state()
        mm = multimeter(**mm_params)

        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt):
                neuron.update()
                vm = float(neuron.V.value[0] / u.mV)
                dftype = brainstate.environ.dftype()
                ditype = brainstate.environ.ditype()
                mm.update({'V_m': np.array([vm], dtype=dftype)}, senders=np.array([1], dtype=ditype))

    return mm.events


def _run_nest_vm_trace(simtime_ms, dt_ms, mm_params, i_e_pA):
    import nest

    nest.ResetKernel()
    nest.resolution = dt_ms

    neuron = nest.Create('iaf_psc_delta', params={'I_e': float(i_e_pA)})
    nest_mm_params = {
        'record_from': list(mm_params['record_from']),
        'interval': _to_ms_scalar(mm_params['interval']),
        'offset': _to_ms_scalar(mm_params.get('offset', 0.0 * u.ms)),
        'start': _to_ms_scalar(mm_params.get('start', 0.0 * u.ms)),
        'origin': _to_ms_scalar(mm_params.get('origin', 0.0 * u.ms)),
    }
    stop = mm_params.get('stop', None)
    if stop is not None:
        nest_mm_params['stop'] = _to_ms_scalar(stop)

    mm = nest.Create('multimeter', params=nest_mm_params)
    # delay=dt reproduces one-step multimeter lag without extra min-delay batching
    nest.Connect(mm, neuron, syn_spec={'delay': dt_ms})
    nest.Simulate(simtime_ms)

    ev = mm.events
    dftype = brainstate.environ.dftype()
    ditype = brainstate.environ.ditype()
    return {
        'times': np.asarray(ev['times'], dtype=dftype),
        'senders': np.asarray(ev['senders'], dtype=ditype),
        'V_m': np.asarray(ev['V_m'], dtype=dftype),
    }


class TestMultimeter(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _run_signal(self, mm, simtime_ms, sender=7):
        n_steps = int(round(simtime_ms / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            for step in range(n_steps):
                stamp = (step + 1) * self.dt_ms
                with brainstate.environ.context(t=step * self.dt):
                    dftype = brainstate.environ.dftype()
                    ditype = brainstate.environ.ditype()
                    mm.update({'x': np.array([stamp], dtype=dftype)}, senders=np.array([sender], dtype=ditype))

    def test_default_parameters(self):
        mm = multimeter()
        self.assertEqual(mm.record_from, ())
        self.assertTrue(u.math.allclose(mm.interval, 1.0 * u.ms))
        self.assertTrue(u.math.allclose(mm.offset, 0.0 * u.ms))
        self.assertTrue(u.math.allclose(mm.start, 0.0 * u.ms))
        self.assertIsNone(mm.stop)
        self.assertTrue(u.math.allclose(mm.origin, 0.0 * u.ms))
        self.assertEqual(mm.n_events, 0)
        self.assertEqual(mm.events['times'].size, 0)

    def test_sampling_window_with_offset(self):
        mm = multimeter(
            record_from=['x'],
            interval=0.3 * u.ms,
            offset=0.2 * u.ms,
            start=0.5 * u.ms,
            stop=2.8 * u.ms,
        )
        self._run_signal(mm, simtime_ms=3.1, sender=7)
        ev = mm.events

        dftype = brainstate.environ.dftype()
        expected_times = np.array([0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6], dtype=dftype)
        npt.assert_allclose(ev['times'], expected_times, atol=1e-12)
        npt.assert_allclose(ev['x'], expected_times, atol=1e-12)
        ditype = brainstate.environ.ditype()
        npt.assert_array_equal(ev['senders'], np.full(expected_times.shape, 7, dtype=ditype))

    def test_origin_shifts_window_not_sampling_grid(self):
        mm = multimeter(
            record_from=['x'],
            interval=0.3 * u.ms,
            offset=0.2 * u.ms,
            start=0.5 * u.ms,
            stop=2.8 * u.ms,
            origin=1.0 * u.ms,
        )
        self._run_signal(mm, simtime_ms=3.1, sender=9)
        ev = mm.events

        dftype = brainstate.environ.dftype()
        expected_times = np.array([1.7, 2.0, 2.3, 2.6, 2.9], dtype=dftype)
        npt.assert_allclose(ev['times'], expected_times, atol=1e-12)
        npt.assert_allclose(ev['x'], expected_times, atol=1e-12)
        ditype = brainstate.environ.ditype()
        npt.assert_array_equal(ev['senders'], np.full(expected_times.shape, 9, dtype=ditype))

    def test_one_step_delivery_lag_and_flush(self):
        mm = multimeter(record_from=['x'], interval=0.1 * u.ms)
        self._run_signal(mm, simtime_ms=1.0)

        no_flush = mm.events
        dftype = brainstate.environ.dftype()
        expected_no_flush = np.arange(0.1, 1.0, 0.1, dtype=dftype)
        npt.assert_allclose(no_flush['times'], expected_no_flush, atol=1e-12)
        npt.assert_allclose(no_flush['x'], expected_no_flush, atol=1e-12)

        with brainstate.environ.context(dt=self.dt):
            flushed = mm.flush()
        expected_flush = np.arange(0.1, 1.1, 0.1, dtype=dftype)
        npt.assert_allclose(flushed['times'], expected_flush, atol=1e-12)
        npt.assert_allclose(flushed['x'], expected_flush, atol=1e-12)

    def test_interval_offset_and_record_from_locked_after_connect(self):
        mm = multimeter(record_from=['x'], interval=0.1 * u.ms, offset=0.0 * u.ms)
        with brainstate.environ.context(dt=self.dt):
            with brainstate.environ.context(t=0.0 * u.ms):
                dftype = brainstate.environ.dftype()
                mm.update({'x': np.array([0.1], dtype=dftype)})

        with self.assertRaises(ValueError):
            mm.interval = 0.2 * u.ms
        with self.assertRaises(ValueError):
            mm.offset = 0.2 * u.ms
        with self.assertRaises(ValueError):
            mm.record_from = ['x']

    def test_validation_rules(self):
        with brainstate.environ.context(dt=self.dt):
            with brainstate.environ.context(t=0.0 * u.ms):
                with self.assertRaises(ValueError):
                    multimeter(record_from=['x'], interval=0.05 * u.ms).update({'x': np.array([0.0])})
                with self.assertRaises(ValueError):
                    multimeter(record_from=['x'], interval=0.15 * u.ms).update({'x': np.array([0.0])})
                with self.assertRaises(ValueError):
                    multimeter(record_from=['x'], offset=0.05 * u.ms).update({'x': np.array([0.0])})
                with self.assertRaises(ValueError):
                    multimeter(record_from=['x'], offset=0.15 * u.ms).update({'x': np.array([0.0])})
                with self.assertRaises(ValueError):
                    multimeter(record_from=['x'], start=1.0 * u.ms, stop=0.5 * u.ms).update({'x': np.array([0.0])})

    def test_matches_nest_vm_trace(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        mm_params = {
            'record_from': ['V_m'],
            'interval': 0.1 * u.ms,
            'offset': 0.0 * u.ms,
            'start': 0.0 * u.ms,
            'stop': None,
            'origin': 0.0 * u.ms,
        }

        bp_events = _run_bp_vm_trace(
            simtime_ms=10.0,
            dt_ms=self.dt_ms,
            mm_params=mm_params,
            i_e_pA=300.0,
        )
        nest_events = _run_nest_vm_trace(
            simtime_ms=10.0,
            dt_ms=self.dt_ms,
            mm_params=mm_params,
            i_e_pA=300.0,
        )

        npt.assert_allclose(bp_events['times'], nest_events['times'], atol=1e-12)
        npt.assert_array_equal(bp_events['senders'], nest_events['senders'])
        npt.assert_allclose(bp_events['V_m'], nest_events['V_m'], atol=1e-10)

    def test_matches_nest_with_offset_start_stop_origin(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        mm_params = {
            'record_from': ['V_m'],
            'interval': 0.3 * u.ms,
            'offset': 0.2 * u.ms,
            'start': 0.5 * u.ms,
            'stop': 2.8 * u.ms,
            'origin': 1.0 * u.ms,
        }

        bp_events = _run_bp_vm_trace(
            simtime_ms=3.1,
            dt_ms=self.dt_ms,
            mm_params=mm_params,
            i_e_pA=500.0,
        )
        nest_events = _run_nest_vm_trace(
            simtime_ms=3.1,
            dt_ms=self.dt_ms,
            mm_params=mm_params,
            i_e_pA=500.0,
        )

        npt.assert_allclose(bp_events['times'], nest_events['times'], atol=1e-12)
        npt.assert_array_equal(bp_events['senders'], nest_events['senders'])
        npt.assert_allclose(bp_events['V_m'], nest_events['V_m'], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
