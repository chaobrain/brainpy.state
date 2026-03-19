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
import saiunit as u
import jax.numpy as jnp
import numpy as np

from brainpy_state._nest.poisson_generator import poisson_generator

brainstate.environ.set(precision=64, platform='cpu')


def _run_bp_counts(
    dt_ms,
    simtime_ms,
    n_trains,
    *,
    rate_hz,
    start_ms=0.0,
    stop_ms=None,
    origin_ms=0.0,
    rng_seed=0,
):
    dt = dt_ms * u.ms
    n_steps = int(round(simtime_ms / dt_ms))
    dftype = brainstate.environ.dftype()

    with brainstate.environ.context(dt=dt):
        gen = poisson_generator(
            in_size=n_trains,
            rate=rate_hz * u.Hz,
            start=start_ms * u.ms,
            stop=(stop_ms * u.ms) if stop_ms is not None else None,
            origin=origin_ms * u.ms,
            rng_seed=rng_seed,
        )
        gen.init_state()

        t_array = jnp.arange(n_steps, dtype=dftype) * dt_ms

        def step_fn(t_ms):
            with brainstate.environ.context(t=t_ms * u.ms):
                out = gen.update()
            return jnp.asarray(out, dtype=dftype).sum()

        totals = np.array(brainstate.transform.for_loop(step_fn, t_array))

    return totals


class TestPoissonGeneratorParameters(unittest.TestCase):
    def test_nest_default_parameters(self):
        gen = poisson_generator()
        self.assertEqual(gen.get()['rate'], 0.0)
        self.assertEqual(gen.get()['start'], 0.0)
        self.assertTrue(np.isinf(gen.get()['stop']))
        self.assertEqual(gen.get()['origin'], 0.0)

    def test_negative_rate_raises(self):
        with self.assertRaisesRegex(ValueError, 'rate cannot be negative'):
            poisson_generator(rate=-1.0 * u.Hz)

    def test_stop_before_start_raises(self):
        with self.assertRaisesRegex(ValueError, 'stop >= start required'):
            poisson_generator(start=2.0 * u.ms, stop=1.0 * u.ms)

    def test_grid_time_validation_matches_nest(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            with self.assertRaisesRegex(ValueError, 'must be a multiple of the simulation resolution'):
                poisson_generator(start=0.15 * u.ms)

    def test_set_parameter_validation(self):
        gen = poisson_generator()
        with self.assertRaisesRegex(ValueError, 'rate cannot be negative'):
            gen.set(rate=-3.0)
        with self.assertRaisesRegex(ValueError, 'stop >= start required'):
            gen.set(start=2.0 * u.ms, stop=1.0 * u.ms)


class TestPoissonGeneratorOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 1.0 * u.ms

    def _run_trace(self, gen, n_steps):
        trace = []
        for step in range(n_steps):
            with brainstate.environ.context(t=step * self.dt):
                trace.append(int(np.asarray(gen.update())[0]))
        return trace

    def test_start_exclusive_stop_inclusive(self):
        with brainstate.environ.context(dt=self.dt):
            gen = poisson_generator(
                in_size=1,
                rate=1000.0 * u.Hz,  # lam = 1 at dt=1 ms
                start=2.0 * u.ms,
                stop=5.0 * u.ms,
                rng_seed=1,
            )
            gen.init_state()
            ditype = brainstate.environ.ditype()
            gen._sample_poisson = lambda lam: jnp.asarray([int(round(float(lam)))], dtype=ditype)

            trace = self._run_trace(gen, 7)
            # Active timestamps are t in (2, 5] ms -> 3,4,5 ms.
            expected = [0, 0, 0, 1, 1, 1, 0]
            self.assertEqual(trace, expected)

    def test_multiplicity_can_exceed_one(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            gen = poisson_generator(
                in_size=1,
                rate=50000.0 * u.Hz,  # lam = 5
                rng_seed=3,
            )
            gen.init_state()

            maxima = 0
            for step in range(300):
                with brainstate.environ.context(t=step * dt):
                    maxima = max(maxima, int(np.asarray(gen.update())[0]))
            self.assertGreater(maxima, 1)


class TestPoissonGeneratorVsNEST(unittest.TestCase):
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
            'start': start_ms,
            'origin': origin_ms,
        }
        if stop_ms is not None:
            params['stop'] = stop_ms

        gens = nest.Create('poisson_generator', n_trains, params=params)
        sr = nest.Create('spike_recorder')
        nest.Connect(gens, sr)
        nest.Simulate(simtime_ms)

        dftype = brainstate.environ.dftype()
        events = sr.get('events')
        if len(events['times']) == 0:
            return np.zeros(n_steps, dtype=dftype)

        steps = np.rint(np.asarray(events['times'], dtype=dftype) / dt_ms).astype(np.int64)
        counts = np.bincount(steps, minlength=n_steps + 2).astype(np.float64)

        # Spike recorder timestamps include one-step transmission delay.
        return counts[1:n_steps + 1]

    def test_mean_dynamics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.1
        simtime_ms = 240.0
        n_trains = 1024
        rate_hz = 800.0
        start_ms = 20.0
        stop_ms = 180.0
        origin_ms = 10.0

        nest_counts = self._run_nest_counts(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            start_ms=start_ms,
            stop_ms=stop_ms,
            origin_ms=origin_ms,
        )
        bp_counts = _run_bp_counts(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            start_ms=start_ms,
            stop_ms=stop_ms,
            origin_ms=origin_ms,
            rng_seed=12345,
        )

        # Align local send-time counts with NEST recorder timestamps (+1 step).
        bp_counts_aligned = np.zeros_like(bp_counts)
        bp_counts_aligned[1:] = bp_counts[:-1]

        off_early = slice(50, 250)  # [5, 25) ms, before active interval
        active = slice(700, 1600)  # [70, 160) ms, well inside active interval
        off_late = slice(2050, 2350)  # [205, 235) ms, after active interval

        nest_mean_active = float(np.mean(nest_counts[active]))
        bp_mean_active = float(np.mean(bp_counts_aligned[active]))

        nest_mean_off_early = float(np.mean(nest_counts[off_early]))
        bp_mean_off_early = float(np.mean(bp_counts_aligned[off_early]))
        nest_mean_off_late = float(np.mean(nest_counts[off_late]))
        bp_mean_off_late = float(np.mean(bp_counts_aligned[off_late]))

        expected_active = n_trains * rate_hz * dt_ms / 1000.0

        self.assertAlmostEqual(nest_mean_active, expected_active, delta=0.15 * expected_active)
        self.assertAlmostEqual(bp_mean_active, expected_active, delta=0.15 * expected_active)
        self.assertAlmostEqual(bp_mean_active, nest_mean_active, delta=0.12 * max(nest_mean_active, 1.0))

        self.assertLess(nest_mean_off_early, 1e-12)
        self.assertLess(bp_mean_off_early, 1e-12)
        self.assertLess(nest_mean_off_late, 1e-12)
        self.assertLess(bp_mean_off_late, 1e-12)


if __name__ == '__main__':
    unittest.main()
