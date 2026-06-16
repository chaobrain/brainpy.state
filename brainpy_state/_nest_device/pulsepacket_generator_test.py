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

import jax

jax.config.update('jax_enable_x64', True)
import brainstate
import brainunit as u
import numpy as np

from brainpy_state._nest_device.pulsepacket_generator import pulsepacket_generator

brainstate.environ.set(precision=64)


def _run_bp_counts(
    dt_ms,
    simtime_ms,
    n_trains,
    *,
    pulse_times_ms,
    activity,
    sdev_ms,
    start_ms=0.0,
    stop_ms=None,
    origin_ms=0.0,
    rng_seed=0,
):
    dt = dt_ms * u.ms
    n_steps = int(round(simtime_ms / dt_ms))
    dftype = brainstate.environ.dftype()
    totals = np.zeros(n_steps, dtype=dftype)

    with brainstate.environ.context(dt=dt):
        gen = pulsepacket_generator(
            in_size=n_trains,
            pulse_times=np.asarray(pulse_times_ms, dtype=dftype) * u.ms,
            activity=activity,
            sdev=sdev_ms * u.ms,
            start=start_ms * u.ms,
            stop=(stop_ms * u.ms) if stop_ms is not None else None,
            origin=origin_ms * u.ms,
            rng_seed=rng_seed,
        )
        gen.init_state()

        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt):
                ditype = brainstate.environ.ditype()
                out = np.asarray(gen.update(), dtype=ditype).reshape(-1)
                totals[step] = float(out.sum())

    return totals


class TestPulsepacketGeneratorParameters(unittest.TestCase):
    def test_nest_default_parameters(self):
        gen = pulsepacket_generator()
        params = gen.get()

        self.assertEqual(params['pulse_times'], [])
        self.assertEqual(params['activity'], 0)
        self.assertEqual(params['sdev'], 0.0)
        self.assertEqual(params['start'], 0.0)
        self.assertTrue(np.isinf(params['stop']))
        self.assertEqual(params['origin'], 0.0)

    def test_set_and_get_parameters(self):
        gen = pulsepacket_generator()
        gen.set(
            pulse_times=[2.5, 1.0, 4.6],
            activity=150,
            sdev=0.1234 * u.ms,
            start=10.0 * u.ms,
            stop=200.0 * u.ms,
            origin=3.0 * u.ms,
        )
        params = gen.get()

        self.assertEqual(params['pulse_times'], [1.0, 2.5, 4.6])
        self.assertEqual(params['activity'], 150)
        self.assertAlmostEqual(params['sdev'], 0.1234)
        self.assertEqual(params['start'], 10.0)
        self.assertEqual(params['stop'], 200.0)
        self.assertEqual(params['origin'], 3.0)

    def test_set_illegal_values(self):
        with self.assertRaisesRegex(ValueError, 'activity cannot be negative'):
            pulsepacket_generator(activity=-5)
        with self.assertRaisesRegex(ValueError, 'standard deviation cannot be negative'):
            pulsepacket_generator(sdev=-5.0 * u.ms)
        with self.assertRaisesRegex(ValueError, 'stop >= start required'):
            pulsepacket_generator(start=2.0 * u.ms, stop=1.0 * u.ms)

    def test_valid_to_pass_empty_pulse_times(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            gen = pulsepacket_generator(pulse_times=[], activity=0, sdev=0.0 * u.ms)
            gen.init_state()
            for step in range(20):
                with brainstate.environ.context(t=step * dt):
                    ditype = brainstate.environ.ditype()
                    out = np.asarray(gen.update(), dtype=ditype)
                    self.assertEqual(int(out.sum()), 0)


class TestPulsepacketGeneratorOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 1.0 * u.ms

    def _run_trace(self, gen, n_steps):
        trace = []
        for step in range(n_steps):
            with brainstate.environ.context(t=step * self.dt):
                trace.append(int(np.asarray(gen.update())[0]))
        return trace

    def test_deterministic_emission_at_pulse_step(self):
        with brainstate.environ.context(dt=self.dt):
            gen = pulsepacket_generator(
                in_size=1,
                pulse_times=[5.0] * u.ms,
                activity=3,
                sdev=0.0 * u.ms,
                rng_seed=1,
            )
            gen.init_state()
            trace = self._run_trace(gen, n_steps=8)

        self.assertEqual(trace, [0, 0, 0, 0, 0, 3, 0, 0])

    def test_current_generator_activity_shift(self):
        with brainstate.environ.context(dt=self.dt):
            gen = pulsepacket_generator(
                in_size=1,
                pulse_times=[3.0] * u.ms,
                activity=1,
                sdev=0.0 * u.ms,
                start=2.0 * u.ms,
                stop=5.0 * u.ms,
                rng_seed=1,
            )
            gen.init_state()
            trace = self._run_trace(gen, n_steps=6)

        self.assertEqual(trace, [0, 0, 0, 1, 0, 0])

    def test_number_of_spikes_for_well_separated_window(self):
        counts = _run_bp_counts(
            dt_ms=0.1,
            simtime_ms=300.0,
            n_trains=1,
            pulse_times_ms=[10.0, 125.0, 175.0, 275.0],
            activity=10,
            sdev_ms=5.0,
            start_ms=75.0,
            stop_ms=225.0,
            rng_seed=42,
        )

        self.assertEqual(int(np.sum(counts)), 20)


class TestPulsepacketGeneratorVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest
            if hasattr(nest, 'node_models'):
                return 'pulsepacket_generator' in nest.node_models
            return 'pulsepacket_generator' in nest.Models()
        except Exception:
            return False

    def _run_nest_counts(
        self,
        dt_ms,
        simtime_ms,
        n_trains,
        *,
        pulse_times_ms,
        activity,
        sdev_ms,
        start_ms=0.0,
        stop_ms=None,
        origin_ms=0.0,
    ):
        import nest

        n_steps = int(round(simtime_ms / dt_ms))
        nest.ResetKernel()
        nest.resolution = dt_ms
        nest.local_num_threads = 1
        nest.rng_seed = 12345

        dftype = brainstate.environ.dftype()
        params = {
            'pulse_times': list(np.asarray(pulse_times_ms, dtype=dftype)),
            'activity': int(activity),
            'sdev': float(sdev_ms),
            'start': float(start_ms),
            'origin': float(origin_ms),
        }
        if stop_ms is not None:
            params['stop'] = float(stop_ms)

        gens = nest.Create('pulsepacket_generator', n_trains, params=params)
        sr = nest.Create('spike_recorder')
        nest.Connect(gens, sr)
        nest.Simulate(simtime_ms)

        events = sr.get('events')
        if len(events['times']) == 0:
            return np.zeros(n_steps, dtype=dftype)

        steps = np.rint(np.asarray(events['times'], dtype=dftype) / dt_ms).astype(np.int64)
        counts = np.bincount(steps, minlength=n_steps + 2).astype(np.float64)
        return counts[1:n_steps + 1]

    def test_mean_dynamics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.1
        simtime_ms = 300.0
        n_trains = 256
        pulse_times_ms = [125.0, 175.0]
        activity = 20
        sdev_ms = 5.0
        start_ms = 75.0
        stop_ms = 225.0

        nest_counts = self._run_nest_counts(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            pulse_times_ms=pulse_times_ms,
            activity=activity,
            sdev_ms=sdev_ms,
            start_ms=start_ms,
            stop_ms=stop_ms,
            origin_ms=0.0,
        )
        bp_counts = _run_bp_counts(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            pulse_times_ms=pulse_times_ms,
            activity=activity,
            sdev_ms=sdev_ms,
            start_ms=start_ms,
            stop_ms=stop_ms,
            origin_ms=0.0,
            rng_seed=12345,
        )

        # Align local send-time counts with NEST recorder timestamps (+1 step).
        bp_counts_aligned = np.zeros_like(bp_counts)
        bp_counts_aligned[1:] = bp_counts[:-1]

        expected_total = float(n_trains * activity * len(pulse_times_ms))
        self.assertEqual(float(np.sum(nest_counts)), expected_total)
        self.assertEqual(float(np.sum(bp_counts_aligned)), expected_total)

        dftype = brainstate.environ.dftype()
        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=dftype)
        kernel = kernel / np.sum(kernel)
        nest_smooth = np.convolve(nest_counts, kernel, mode='same')
        bp_smooth = np.convolve(bp_counts_aligned, kernel, mode='same')

        compare_window = slice(int(80.0 / dt_ms), int(220.0 / dt_ms))
        corr = np.corrcoef(nest_smooth[compare_window], bp_smooth[compare_window])[0, 1]
        self.assertGreater(corr, 0.93)

        for center in pulse_times_ms:
            i0 = int(round((center - 20.0) / dt_ms))
            i1 = int(round((center + 20.0) / dt_ms))

            nest_mass = float(np.sum(nest_counts[i0:i1]))
            bp_mass = float(np.sum(bp_counts_aligned[i0:i1]))
            self.assertAlmostEqual(bp_mass, nest_mass, delta=0.08 * max(nest_mass, 1.0))

            nest_peak = i0 + int(np.argmax(nest_smooth[i0:i1]))
            bp_peak = i0 + int(np.argmax(bp_smooth[i0:i1]))
            self.assertLessEqual(abs(bp_peak - nest_peak) * dt_ms, 0.8)


if __name__ == '__main__':
    unittest.main()
