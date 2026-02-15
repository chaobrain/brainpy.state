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

from brainpy_state._nest.ppd_sup_generator import ppd_sup_generator

brainstate.environ.set(precision=64, platform='cpu')


def _run_bp_counts_and_spikes(
    dt_ms,
    simtime_ms,
    n_trains,
    *,
    rate_hz,
    dead_time_ms,
    n_proc,
    frequency_hz=0.0,
    relative_amplitude=0.0,
    start_ms=0.0,
    stop_ms=None,
    origin_ms=0.0,
    rng_seed=0,
    collect_spikes=True,
):
    dt = dt_ms * u.ms
    n_steps = int(round(simtime_ms / dt_ms))
    totals = np.zeros(n_steps, dtype=np.float64)
    spike_chunks = [[] for _ in range(n_trains)] if collect_spikes else None

    with brainstate.environ.context(dt=dt):
        gen = ppd_sup_generator(
            in_size=n_trains,
            rate=rate_hz * u.Hz,
            dead_time=dead_time_ms * u.ms,
            n_proc=n_proc,
            frequency=frequency_hz * u.Hz,
            relative_amplitude=relative_amplitude,
            start=start_ms * u.ms,
            stop=(stop_ms * u.ms) if stop_ms is not None else None,
            origin=origin_ms * u.ms,
            rng_seed=rng_seed,
        )
        gen.init_state()

        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt):
                out = np.asarray(gen.update(), dtype=np.int64).reshape(-1)
                totals[step] = float(out.sum())
                if spike_chunks is not None:
                    event_time = (step + 1.0) * dt_ms
                    for i, k in enumerate(out):
                        k = int(k)
                        if k > 0:
                            spike_chunks[i].append(np.full(k, event_time, dtype=np.float64))

    if spike_chunks is None:
        return totals, None

    spikes = []
    for chunks in spike_chunks:
        if chunks:
            spikes.append(np.concatenate(chunks))
        else:
            spikes.append(np.asarray([], dtype=np.float64))
    return totals, spikes


def _cvsq(spike_times_ms: np.ndarray) -> float:
    isi = np.diff(np.asarray(spike_times_ms, dtype=np.float64))
    isi_m1 = np.sum(isi)
    isi_m2 = np.sum(isi ** 2)
    isi_mean = isi_m1 / len(isi)
    isi_var = isi_m2 / len(isi) - isi_mean ** 2
    return float(isi_var / (isi_mean ** 2))


class TestPPDSupGeneratorParameters(unittest.TestCase):
    def test_nest_default_parameters(self):
        gen = ppd_sup_generator()
        params = gen.get()
        self.assertEqual(params['rate'], 0.0)
        self.assertEqual(params['dead_time'], 0.0)
        self.assertEqual(params['n_proc'], 1)
        self.assertEqual(params['frequency'], 0.0)
        self.assertEqual(params['relative_amplitude'], 0.0)
        self.assertEqual(params['start'], 0.0)
        self.assertTrue(np.isinf(params['stop']))
        self.assertEqual(params['origin'], 0.0)

    def test_parameter_validation(self):
        with self.assertRaisesRegex(ValueError, 'dead time cannot be negative'):
            ppd_sup_generator(dead_time=-0.1 * u.ms)
        with self.assertRaisesRegex(ValueError, 'inverse rate has to be larger than the dead time'):
            ppd_sup_generator(rate=1000.0 * u.Hz, dead_time=1.0 * u.ms)
        with self.assertRaisesRegex(ValueError, 'component processes cannot be smaller than one'):
            ppd_sup_generator(n_proc=0)
        with self.assertRaisesRegex(ValueError, 'relative amplitude of the rate modulation must be in \\[0,1\\]'):
            ppd_sup_generator(relative_amplitude=1.2)
        with self.assertRaisesRegex(ValueError, 'stop >= start required'):
            ppd_sup_generator(start=2.0 * u.ms, stop=1.0 * u.ms)

    def test_grid_time_validation_matches_nest(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            with self.assertRaisesRegex(ValueError, 'must be a multiple of the simulation resolution'):
                ppd_sup_generator(start=0.15 * u.ms)

    def test_set_validation(self):
        gen = ppd_sup_generator()
        with self.assertRaisesRegex(ValueError, 'n_proc must be an integer'):
            gen.set(n_proc=1.2)
        with self.assertRaisesRegex(ValueError, 'relative amplitude of the rate modulation must be in \\[0,1\\]'):
            gen.set(relative_amplitude=-0.1)


class TestPPDSupGeneratorOrdering(unittest.TestCase):
    def setUp(self):
        self.dt = 1.0 * u.ms

    def _run_trace(self, gen, n_steps):
        trace = []
        for step in range(n_steps):
            with brainstate.environ.context(t=step * self.dt):
                trace.append(int(np.asarray(gen.update())[0]))
        return trace

    def test_start_exclusive_stop_inclusive(self):
        with brainstate.environ.context(dt=self.dt):
            gen = ppd_sup_generator(
                in_size=1,
                rate=500.0 * u.Hz,
                dead_time=0.0 * u.ms,
                n_proc=4,
                start=2.0 * u.ms,
                stop=5.0 * u.ms,
                rng_seed=1,
            )
            gen.init_state()
            gen._sample_binomial = lambda n, p: int(round(n * p))

            trace = self._run_trace(gen, n_steps=7)
            self.assertEqual(trace, [0, 0, 0, 2, 2, 2, 0])

    def test_sinusoidal_hazard_uses_left_step_edge_time(self):
        with brainstate.environ.context(dt=self.dt):
            gen = ppd_sup_generator(
                in_size=1,
                rate=100.0 * u.Hz,
                dead_time=0.0 * u.ms,
                n_proc=10,
                frequency=250.0 * u.Hz,
                relative_amplitude=1.0,
                start=-1.0 * u.ms,
                stop=10.0 * u.ms,
                rng_seed=2,
            )
            gen.init_state()
            gen._sample_binomial = lambda n, p: int(round(n * p))

            trace = self._run_trace(gen, n_steps=5)
            self.assertEqual(trace, [1, 2, 1, 0, 1])

    def test_multiplicity_can_exceed_one(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            gen = ppd_sup_generator(
                in_size=1,
                rate=200.0 * u.Hz,
                dead_time=0.0 * u.ms,
                n_proc=500,
                rng_seed=3,
            )
            gen.init_state()

            max_count = 0
            for step in range(300):
                with brainstate.environ.context(t=step * 0.1 * u.ms):
                    max_count = max(max_count, int(np.asarray(gen.update())[0]))
            self.assertGreater(max_count, 1)

    def test_different_outputs(self):
        simtime_ms = 10.0
        _, spikes = _run_bp_counts_and_spikes(
            dt_ms=0.01,
            simtime_ms=simtime_ms,
            n_trains=2,
            rate_hz=25.0,
            dead_time_ms=5.0,
            n_proc=1000,
            rng_seed=4,
            collect_spikes=True,
        )

        self.assertFalse(np.array_equal(spikes[0], spikes[1]))

        rate_ana = 25.0 * 1000.0
        rate_1 = len(spikes[0]) / (simtime_ms * 1e-3)
        rate_2 = len(spikes[1]) / (simtime_ms * 1e-3)
        self.assertTrue(0.8 < rate_1 / rate_ana < 1.2)
        self.assertTrue(0.8 < rate_2 / rate_ana < 1.2)


class TestPPDSupGeneratorStatistics(unittest.TestCase):
    def test_superposition_rate_and_cvsq_match_theory(self):
        err = 0.2
        dead_time = 25.0
        rate = 10.0
        simtime_ms = 10000.0
        n_proc = 5

        _, spikes = _run_bp_counts_and_spikes(
            dt_ms=1.0,
            simtime_ms=simtime_ms,
            n_trains=1,
            rate_hz=rate,
            dead_time_ms=dead_time,
            n_proc=n_proc,
            rng_seed=7,
            collect_spikes=True,
        )
        spikes = spikes[0]

        rate_sim = len(spikes) / (simtime_ms * 1e-3)
        ratio_rate = rate_sim / (rate * n_proc)
        self.assertTrue((1.0 - err) < ratio_rate < (1.0 + err))

        cvsq_sim = _cvsq(spikes)
        mu = (1.0 / rate) * 1e3
        cvfact1 = 1.0 / (1.0 + n_proc)
        cvfact2 = n_proc - 1.0 + 2.0 * ((1.0 - dead_time / mu) ** (n_proc + 1))
        cvsq_theo = cvfact1 * cvfact2
        ratio_cvsq = cvsq_sim / cvsq_theo
        self.assertTrue((1.0 - err) <= ratio_cvsq <= (1.0 + err))

    def test_single_process_rate_dead_time_and_cvsq_match_theory(self):
        err = 0.2
        dead_time = 45.0
        rate = 15.0
        simtime_ms = 100000.0
        n_proc = 1

        _, spikes = _run_bp_counts_and_spikes(
            dt_ms=1.0,
            simtime_ms=simtime_ms,
            n_trains=1,
            rate_hz=rate,
            dead_time_ms=dead_time,
            n_proc=n_proc,
            rng_seed=9,
            collect_spikes=True,
        )
        spikes = spikes[0]

        rate_sim = len(spikes) / (simtime_ms * 1e-3)
        ratio_rate = rate_sim / (rate * n_proc)
        self.assertTrue((1.0 - err) < ratio_rate < (1.0 + err))

        isi = np.diff(spikes)
        self.assertGreater(isi.size, 10)
        self.assertTrue(np.all(isi >= dead_time - 1e-12))

        cvsq_sim = _cvsq(spikes)
        mu = 1.0 / rate
        lam = mu - dead_time / 1000.0
        cvsq_theo = (lam / mu) ** 2
        ratio_cvsq = cvsq_sim / cvsq_theo
        self.assertTrue((1.0 - err) <= ratio_cvsq <= (1.0 + err))


class TestPPDSupGeneratorVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest
            if hasattr(nest, 'node_models'):
                return 'ppd_sup_generator' in nest.node_models
            return 'ppd_sup_generator' in nest.Models()
        except Exception:
            return False

    def _run_nest_counts(
        self,
        dt_ms,
        simtime_ms,
        n_trains,
        *,
        rate_hz,
        dead_time_ms,
        n_proc,
        frequency_hz=0.0,
        relative_amplitude=0.0,
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
            'n_proc': n_proc,
            'frequency': frequency_hz,
            'relative_amplitude': relative_amplitude,
            'start': start_ms,
            'origin': origin_ms,
        }
        if stop_ms is not None:
            params['stop'] = stop_ms

        gens = nest.Create('ppd_sup_generator', n_trains, params=params)
        sr = nest.Create('spike_recorder')
        nest.Connect(gens, sr)
        nest.Simulate(simtime_ms)

        events = sr.get('events')
        if len(events['times']) == 0:
            return np.zeros(n_steps, dtype=np.float64)

        steps = np.rint(np.asarray(events['times'], dtype=np.float64) / dt_ms).astype(np.int64)
        counts = np.bincount(steps, minlength=n_steps + 2).astype(np.float64)
        return counts[1:n_steps + 1]

    def test_mean_dynamics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.1
        simtime_ms = 320.0
        n_trains = 512
        rate_hz = 40.0
        dead_time_ms = 2.0
        n_proc = 8
        start_ms = 20.0
        stop_ms = 260.0
        origin_ms = 10.0

        nest_counts = self._run_nest_counts(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            dead_time_ms=dead_time_ms,
            n_proc=n_proc,
            frequency_hz=0.0,
            relative_amplitude=0.0,
            start_ms=start_ms,
            stop_ms=stop_ms,
            origin_ms=origin_ms,
        )
        bp_counts, _ = _run_bp_counts_and_spikes(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            dead_time_ms=dead_time_ms,
            n_proc=n_proc,
            frequency_hz=0.0,
            relative_amplitude=0.0,
            start_ms=start_ms,
            stop_ms=stop_ms,
            origin_ms=origin_ms,
            rng_seed=12345,
            collect_spikes=False,
        )

        # Align local send-time counts with NEST recorder timestamps (+1 step).
        bp_counts_aligned = np.zeros_like(bp_counts)
        bp_counts_aligned[1:] = bp_counts[:-1]

        active = slice(700, 2400)      # [70, 240) ms, safely inside active window
        off_early = slice(50, 250)     # [5, 25) ms, before activation
        off_late = slice(2850, 3150)   # [285, 315) ms, after deactivation

        nest_mean_active = float(np.mean(nest_counts[active]))
        bp_mean_active = float(np.mean(bp_counts_aligned[active]))
        expected = n_trains * n_proc * rate_hz * dt_ms / 1000.0

        self.assertAlmostEqual(nest_mean_active, expected, delta=0.18 * expected)
        self.assertAlmostEqual(bp_mean_active, expected, delta=0.18 * expected)
        self.assertAlmostEqual(bp_mean_active, nest_mean_active, delta=0.14 * max(nest_mean_active, 1.0))

        nest_std_active = float(np.std(nest_counts[active]))
        bp_std_active = float(np.std(bp_counts_aligned[active]))
        self.assertAlmostEqual(bp_std_active, nest_std_active, delta=0.22 * max(nest_std_active, 1.0))

        self.assertLess(float(np.mean(nest_counts[off_early])), 1e-12)
        self.assertLess(float(np.mean(bp_counts_aligned[off_early])), 1e-12)
        self.assertLess(float(np.mean(nest_counts[off_late])), 1e-12)
        self.assertLess(float(np.mean(bp_counts_aligned[off_late])), 1e-12)


if __name__ == '__main__':
    unittest.main()
