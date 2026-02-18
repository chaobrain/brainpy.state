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
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from brainpy_state._nest.inhomogeneous_poisson_generator import (
    inhomogeneous_poisson_generator,
)

brainstate.environ.set(precision=64, platform='cpu')


def _run_bp_counts(
    dt_ms,
    simtime_ms,
    n_trains,
    *,
    rate_times,
    rate_values,
    allow_offgrid_times=False,
    start_ms=0.0,
    stop_ms=None,
    origin_ms=0.0,
    rng_seed=0,
):
    dt = dt_ms * u.ms
    n_steps = int(round(simtime_ms / dt_ms))
    totals = np.zeros(n_steps, dtype=np.float64)

    with brainstate.environ.context(dt=dt):
        gen = inhomogeneous_poisson_generator(
            in_size=n_trains,
            rate_times=rate_times,
            rate_values=rate_values,
            allow_offgrid_times=allow_offgrid_times,
            start=start_ms * u.ms,
            stop=(stop_ms * u.ms) if stop_ms is not None else None,
            origin=origin_ms * u.ms,
            rng_seed=rng_seed,
        )
        gen.init_state()

        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt):
                totals[step] = float(np.asarray(gen.update(), dtype=np.float64).sum())

    return totals


class TestInhomogeneousPoissonGeneratorParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_rate_times_and_values_match(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator()
            with self.assertRaisesRegex(ValueError, 'Rate times and values must be reset together'):
                gen.set(rate_values=[10.0])

    def test_rate_times_len_and_values_match(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator()
            with self.assertRaisesRegex(ValueError, 'Rate times and values have to be the same size'):
                gen.set(rate_times=[1.0, 2.0], rate_values=[10.0])

    def test_rate_times_strictly_increasing(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator()
            with self.assertRaisesRegex(ValueError, 'Rate times must be strictly increasing'):
                gen.set(rate_times=[1.0, 5.0, 3.0], rate_values=[10.0, 20.0, 5.0])

    def test_offgrid_time_point(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator()
            with self.assertRaisesRegex(ValueError, 'is not representable in current resolution'):
                gen.set(rate_times=[1.23], rate_values=[10.0])

    def test_allow_offgrid_time_point(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator(allow_offgrid_times=True)
            gen.set(rate_times=[1.23], rate_values=[10.0])
            params = gen.get()
            self.assertAlmostEqual(params['rate_times'], 1.3, places=12)

    def test_no_allow_offgrid_times_after_rate_set(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator()
            gen.set(rate_times=[1.2], rate_values=[10.0])
            with self.assertRaisesRegex(
                ValueError,
                'Option can only be set together with rate times or if no rate times have been set',
            ):
                gen.set(allow_offgrid_times=True)

    def test_allow_offgrid_times_modified_after_rate_set(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator(allow_offgrid_times=True)
            gen.set(rate_times=[1.23], rate_values=[10.0])
            with self.assertRaisesRegex(ValueError, 'is not representable in current resolution'):
                gen.set(allow_offgrid_times=False, rate_times=[1.25], rate_values=[10.0])

    def test_time_points_in_future(self):
        with brainstate.environ.context(dt=self.dt, t=0.0 * u.ms):
            gen = inhomogeneous_poisson_generator()
            with self.assertRaisesRegex(ValueError, 'Time points must lie strictly in the future'):
                gen.set(rate_times=[0.0], rate_values=[30.0])

    def test_set_empty_params(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator(
                rate_times=[10.0, 110.0],
                rate_values=[400.0, 1000.0],
            )
            gen.set(rate_times=[], rate_values=[])
            params = gen.get()
            self.assertEqual(params['rate_times'], [])
            self.assertEqual(params['rate_values'], [])

    def test_params_set_implicitly_on_creation(self):
        with brainstate.environ.context(dt=self.dt):
            params = {'rate_times': [10.0, 110.0, 210.0], 'rate_values': [400.0, 1000.0, 200.0]}
            gen = inhomogeneous_poisson_generator(**params)
            got = gen.get()
            npt.assert_array_equal(got['rate_times'], params['rate_times'])
            npt.assert_array_equal(got['rate_values'], params['rate_values'])


class TestInhomogeneousPoissonGeneratorOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 1.0 * u.ms

    def _run_trace(self, gen, n_steps):
        counts = []
        for step in range(n_steps):
            with brainstate.environ.context(t=step * self.dt):
                out = gen.update()
            counts.append(int(np.asarray(out)[0]))
        return counts

    def test_rate_changes_applied_one_step_ahead(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator(
                in_size=1,
                rate_times=[5.0, 8.0],
                rate_values=[1000.0, 2000.0],
                rng_seed=1,
            )
            gen.init_state()
            gen._sample_poisson = lambda lam: jnp.asarray([int(round(float(lam)))], dtype=jnp.int64)

            trace = self._run_trace(gen, n_steps=11)
            expected = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
            self.assertEqual(trace, expected)

    def test_start_exclusive_stop_inclusive(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator(
                in_size=1,
                rate_times=[1.0],
                rate_values=[1000.0],
                start=2.0 * u.ms,
                stop=5.0 * u.ms,
                rng_seed=2,
            )
            gen.init_state()
            gen._sample_poisson = lambda lam: jnp.asarray([int(round(float(lam)))], dtype=jnp.int64)

            trace = self._run_trace(gen, n_steps=7)
            expected = [0, 0, 0, 1, 1, 1, 0]
            self.assertEqual(trace, expected)

    def test_past_rate_times_are_skipped(self):
        with brainstate.environ.context(dt=self.dt):
            gen = inhomogeneous_poisson_generator(
                in_size=1,
                rate_times=[2.0, 4.0],
                rate_values=[1000.0, 2000.0],
                rng_seed=3,
            )
            gen.init_state()
            with brainstate.environ.context(t=10.0 * u.ms):
                out = gen.update()
            self.assertEqual(int(np.asarray(out)[0]), 0)
            self.assertEqual(int(gen._rate_idx.value), 2)


class TestInhomogeneousPoissonGeneratorVsNEST(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    @staticmethod
    def _is_nest_available():
        try:
            import nest  # noqa: F401
            return True
        except ImportError:
            return False

    def _run_nest_counts(
        self,
        simtime_ms,
        n_trains,
        *,
        rate_times,
        rate_values,
        allow_offgrid_times=False,
        start_ms=0.0,
        stop_ms=None,
        origin_ms=0.0,
    ):
        import nest

        n_steps = int(round(simtime_ms / self.dt_ms))
        nest.ResetKernel()
        nest.resolution = self.dt_ms
        nest.local_num_threads = 1

        params = {
            'rate_times': rate_times,
            'rate_values': rate_values,
            'allow_offgrid_times': allow_offgrid_times,
            'start': start_ms,
            'origin': origin_ms,
        }
        if stop_ms is not None:
            params['stop'] = stop_ms

        gens = nest.Create('inhomogeneous_poisson_generator', n_trains, params=params)
        sr = nest.Create('spike_recorder')
        nest.Connect(gens, sr)
        nest.Simulate(simtime_ms)

        events = sr.get('events')
        steps = np.rint(np.asarray(events['times'], dtype=np.float64) / self.dt_ms).astype(np.int64)
        counts = np.bincount(steps, minlength=n_steps + 2).astype(np.float64)

        # Recorder timestamps include one-step transmission delay.
        return counts[1:n_steps + 1]

    def test_mean_dynamics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        simtime_ms = 120.0
        n_trains = 512
        rate_times = [20.0, 60.0, 90.0]
        rate_values = [500.0, 1500.0, 0.0]

        nest_counts = self._run_nest_counts(
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_times=rate_times,
            rate_values=rate_values,
        )
        bp_counts = _run_bp_counts(
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_times=rate_times,
            rate_values=rate_values,
            rng_seed=0,
        )

        # Align local send-time counts with NEST recorder timestamps (+1 step).
        bp_counts_aligned = np.zeros_like(bp_counts)
        bp_counts_aligned[1:] = bp_counts[:-1]

        phase2 = slice(250, 550)   # inside [20, 60) ms
        phase3 = slice(650, 850)   # inside [60, 90) ms

        nest_mean_2 = float(np.mean(nest_counts[phase2]))
        nest_mean_3 = float(np.mean(nest_counts[phase3]))
        bp_mean_2 = float(np.mean(bp_counts_aligned[phase2]))
        bp_mean_3 = float(np.mean(bp_counts_aligned[phase3]))

        self.assertAlmostEqual(bp_mean_2, nest_mean_2, delta=0.12 * max(nest_mean_2, 1.0))
        self.assertAlmostEqual(bp_mean_3, nest_mean_3, delta=0.12 * max(nest_mean_3, 1.0))


if __name__ == '__main__':
    unittest.main()
