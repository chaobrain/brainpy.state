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

from brainpy_state._nest_device.poisson_generator_ps import poisson_generator_ps

brainstate.environ.set(precision=64, platform='cpu')


def _run_bp_counts_and_times(
    dt_ms,
    simtime_ms,
    n_trains,
    *,
    rate_hz,
    dead_time_ms=0.0,
    start_ms=0.0,
    stop_ms=None,
    origin_ms=0.0,
    rng_seed=0,
):
    dt = dt_ms * u.ms
    n_steps = int(round(simtime_ms / dt_ms))
    dftype = brainstate.environ.dftype()
    totals = np.zeros(n_steps, dtype=dftype)
    spikes = [[] for _ in range(n_trains)]

    with brainstate.environ.context(dt=dt):
        gen = poisson_generator_ps(
            in_size=n_trains,
            rate=rate_hz * u.Hz,
            dead_time=dead_time_ms * u.ms,
            start=start_ms * u.ms,
            stop=(stop_ms * u.ms) if stop_ms is not None else None,
            origin=origin_ms * u.ms,
            rng_seed=rng_seed,
        )
        gen.init_state()

        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt):
                counts, step_times = gen.update(return_precise_times=True)
                totals[step] = float(np.asarray(counts, dtype=dftype).sum())
                for i, ev in enumerate(step_times):
                    if ev.size:
                        spikes[i].append(np.asarray(ev, dtype=dftype))

    spike_times = []
    for chunks in spikes:
        if chunks:
            spike_times.append(np.concatenate(chunks))
        else:
            spike_times.append(np.asarray([], dtype=dftype))

    return totals, spike_times


class TestPoissonGeneratorPSParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_default_parameters(self):
        gen = poisson_generator_ps()
        params = gen.get()
        self.assertEqual(params['rate'], 0.0)
        self.assertEqual(params['dead_time'], 0.0)
        self.assertEqual(params['start'], 0.0)
        self.assertTrue(np.isinf(params['stop']))
        self.assertEqual(params['origin'], 0.0)

    def test_negative_rate_raises(self):
        with self.assertRaisesRegex(ValueError, 'rate cannot be negative'):
            poisson_generator_ps(rate=-1.0 * u.Hz)

    def test_negative_dead_time_raises(self):
        with self.assertRaisesRegex(ValueError, 'dead time cannot be negative'):
            poisson_generator_ps(dead_time=-0.1 * u.ms)

    def test_inverse_rate_vs_dead_time_constraint(self):
        with self.assertRaisesRegex(ValueError, 'inverse rate cannot be smaller than the dead time'):
            poisson_generator_ps(rate=1000.0 * u.Hz, dead_time=1.2 * u.ms)

    def test_set_rate_resets_next_spike_state(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            gen = poisson_generator_ps(in_size=3, rate=500.0 * u.Hz, rng_seed=1)
            gen.init_state()
            dftype = brainstate.environ.dftype()
            gen.next_spike_time.value = np.asarray([3.0, 4.0, 5.0], dtype=dftype)
            gen.set(rate=500.0 * u.Hz)
            self.assertTrue(np.all(np.isneginf(np.asarray(gen.next_spike_time.value))))


class TestPoissonGeneratorPSOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_update_returns_precise_times(self):
        with brainstate.environ.context(dt=1.0 * u.ms):
            gen = poisson_generator_ps(in_size=4, rate=1000.0 * u.Hz, rng_seed=2)
            gen.init_state()
            with brainstate.environ.context(t=0.0 * u.ms):
                counts, step_times = gen.update(return_precise_times=True)
            self.assertEqual(np.asarray(counts).shape, (4,))
            self.assertEqual(len(step_times), 4)

    def test_dead_time_enforced(self):
        _, spike_times = _run_bp_counts_and_times(
            dt_ms=0.1,
            simtime_ms=300.0,
            n_trains=1,
            rate_hz=800.0,
            dead_time_ms=0.7,
            rng_seed=3,
        )
        sp = spike_times[0]
        self.assertGreater(sp.size, 20)
        isi = np.diff(sp)
        self.assertTrue(np.all(isi >= 0.7 - 1e-12))

    def test_resolution_invariance_and_time_window(self):
        params = {
            'rate_hz': 12345.0,
            'dead_time_ms': 0.03,
            'start_ms': 1.0,
            'stop_ms': 2.0,
            'origin_ms': 0.0,
            'rng_seed': 11,
        }
        resolutions = [0.01, 0.1, 0.2, 0.5, 1.0]
        simtime_ms = params['stop_ms'] + 2.0

        all_traces = []
        for h in resolutions:
            _, spikes = _run_bp_counts_and_times(
                dt_ms=h,
                simtime_ms=simtime_ms,
                n_trains=2,
                **params,
            )

            # Distinct trains across targets (matching NEST expectation).
            self.assertFalse(np.array_equal(spikes[0], spikes[1]))

            merged = np.concatenate(spikes)
            self.assertTrue(np.all(merged > params['origin_ms'] + params['start_ms']))
            self.assertTrue(np.all(merged <= params['origin_ms'] + params['stop_ms']))
            all_traces.append(merged)

        for trace in all_traces[1:]:
            self.assertTrue(np.allclose(trace, all_traces[0], atol=1e-15, rtol=0.0))


class TestPoissonGeneratorPSVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest  # noqa: F401
            return True
        except ImportError:
            return False

    def _run_nest_counts(
        self,
        dt_ms,
        simtime_ms,
        n_trains,
        *,
        rate_hz,
        dead_time_ms,
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

        params = {
            'rate': rate_hz,
            'dead_time': dead_time_ms,
            'start': start_ms,
            'origin': origin_ms,
        }
        if stop_ms is not None:
            params['stop'] = stop_ms

        gens = nest.Create('poisson_generator_ps', n_trains, params=params)
        sr = nest.Create('spike_recorder')
        nest.Connect(gens, sr)
        nest.Simulate(simtime_ms)

        dftype = brainstate.environ.dftype()
        events = sr.get('events')
        if len(events['times']) == 0:
            return np.zeros(n_steps, dtype=dftype)

        steps = np.rint(np.asarray(events['times'], dtype=dftype) / dt_ms).astype(np.int64)
        counts = np.bincount(steps, minlength=n_steps + 2).astype(np.float64)

        # Recorder timestamps include one-step transmission delay.
        return counts[1:n_steps + 1]

    def test_mean_dynamics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.1
        simtime_ms = 320.0
        n_trains = 256
        rate_hz = 1200.0
        dead_time_ms = 0.2
        start_ms = 20.0
        stop_ms = 220.0
        origin_ms = 10.0

        nest_counts = self._run_nest_counts(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            dead_time_ms=dead_time_ms,
            start_ms=start_ms,
            stop_ms=stop_ms,
            origin_ms=origin_ms,
        )

        bp_counts, _ = _run_bp_counts_and_times(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            dead_time_ms=dead_time_ms,
            start_ms=start_ms,
            stop_ms=stop_ms,
            origin_ms=origin_ms,
            rng_seed=12345,
        )

        # Align local send-time counts with NEST recorder timestamps (+1 step).
        bp_counts_aligned = np.zeros_like(bp_counts)
        bp_counts_aligned[1:] = bp_counts[:-1]

        active = slice(600, 1900)  # [60, 190) ms, safely inside active interval
        off_early = slice(50, 200)  # [5, 20) ms, before activation
        off_late = slice(2600, 3050)  # [260, 305) ms, after deactivation

        nest_mean_active = float(np.mean(nest_counts[active]))
        bp_mean_active = float(np.mean(bp_counts_aligned[active]))
        expected_active = n_trains * rate_hz * dt_ms / 1000.0

        self.assertAlmostEqual(nest_mean_active, expected_active, delta=0.18 * expected_active)
        self.assertAlmostEqual(bp_mean_active, expected_active, delta=0.18 * expected_active)
        self.assertAlmostEqual(bp_mean_active, nest_mean_active, delta=0.14 * max(nest_mean_active, 1.0))

        self.assertLess(float(np.mean(nest_counts[off_early])), 1e-12)
        self.assertLess(float(np.mean(bp_counts_aligned[off_early])), 1e-12)
        self.assertLess(float(np.mean(nest_counts[off_late])), 1e-12)
        self.assertLess(float(np.mean(bp_counts_aligned[off_late])), 1e-12)


if __name__ == '__main__':
    unittest.main()
