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

from brainpy_state._nest.sinusoidal_poisson_generator import (
    sinusoidal_poisson_generator,
)

brainstate.environ.set(precision=64, platform='cpu')


def _run_bp_counts(
    dt_ms,
    simtime_ms,
    n_trains,
    *,
    rate_hz,
    amplitude_hz,
    frequency_hz,
    phase_deg,
    start_ms=0.0,
    stop_ms=None,
    origin_ms=0.0,
    individual_spike_trains=True,
    rng_seed=0,
):
    dt = dt_ms * u.ms
    n_steps = int(round(simtime_ms / dt_ms))
    totals = np.zeros(n_steps, dtype=np.float64)

    with brainstate.environ.context(dt=dt):
        gen = sinusoidal_poisson_generator(
            in_size=n_trains,
            rate=rate_hz * u.Hz,
            amplitude=amplitude_hz * u.Hz,
            frequency=frequency_hz * u.Hz,
            phase=phase_deg,
            individual_spike_trains=individual_spike_trains,
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


class TestSinusoidalPoissonGeneratorParameters(unittest.TestCase):
    def test_nest_default_parameters(self):
        gen = sinusoidal_poisson_generator()
        params = gen.get()
        self.assertEqual(params['rate'], 0.0)
        self.assertEqual(params['amplitude'], 0.0)
        self.assertEqual(params['frequency'], 0.0)
        self.assertEqual(params['phase'], 0.0)
        self.assertTrue(params['individual_spike_trains'])
        self.assertEqual(params['start'], 0.0)
        self.assertTrue(np.isinf(params['stop']))
        self.assertEqual(params['origin'], 0.0)

    def test_stop_before_start_raises(self):
        with self.assertRaisesRegex(ValueError, 'stop >= start required'):
            sinusoidal_poisson_generator(start=2.0 * u.ms, stop=1.0 * u.ms)

    def test_grid_time_validation_matches_nest_behavior(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            with self.assertRaisesRegex(
                ValueError,
                'must be a multiple of the simulation resolution',
            ):
                sinusoidal_poisson_generator(start=0.15 * u.ms)


class TestSinusoidalPoissonGeneratorOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 1.0 * u.ms

    def _run_trace(self, gen, n_steps):
        trace = []
        for step in range(n_steps):
            with brainstate.environ.context(t=step * self.dt):
                trace.append(int(np.asarray(gen.update())[0]))
        return trace

    def test_start_stop_uses_current_generator_activity_shift(self):
        with brainstate.environ.context(dt=self.dt):
            gen = sinusoidal_poisson_generator(
                in_size=1,
                rate=1000.0 * u.Hz,   # lam = 1 at dt = 1 ms
                amplitude=0.0 * u.Hz,
                frequency=0.0 * u.Hz,
                start=2.0 * u.ms,
                stop=5.0 * u.ms,
                rng_seed=1,
            )
            gen.init_state()
            gen._sample_poisson_individual = (
                lambda lam: jnp.asarray([int(round(float(lam)))], dtype=jnp.int64)
            )

            trace = self._run_trace(gen, n_steps=7)
            # NEST uses CURRENT_GENERATOR activity handling for this model:
            # active when start < step+2 <= stop. For start=2, stop=5, dt=1:
            # active send steps are 1,2,3.
            self.assertEqual(trace, [0, 1, 1, 1, 0, 0, 0])

    def test_recorded_rate_matches_sinusoidal_profile(self):
        dt_ms = 0.1
        dt = dt_ms * u.ms
        n_steps = 1000

        dc_hz = 1.0
        ac_hz = 0.5
        freq_hz = 10.0
        phi_rad = 2.0
        phase_deg = phi_rad / np.pi * 180.0

        recorded = np.zeros(n_steps, dtype=np.float64)
        with brainstate.environ.context(dt=dt):
            gen = sinusoidal_poisson_generator(
                in_size=1,
                rate=dc_hz * u.Hz,
                amplitude=ac_hz * u.Hz,
                frequency=freq_hz * u.Hz,
                phase=phase_deg,
                rng_seed=3,
            )
            gen.init_state()
            for step in range(n_steps):
                with brainstate.environ.context(t=step * dt):
                    gen.update()
                recorded[step] = gen.get_recorded_rate()

        times_ms = (np.arange(n_steps, dtype=np.float64) + 1.0) * dt_ms
        expected = np.maximum(
            0.0,
            dc_hz + ac_hz * np.sin(2.0 * np.pi * freq_hz * times_ms / 1000.0 + phi_rad),
        )
        npt.assert_allclose(recorded, expected, rtol=0.0, atol=1e-12)

    def test_individual_spike_trains_false_broadcasts_multiplicity(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            gen = sinusoidal_poisson_generator(
                in_size=16,
                rate=50000.0 * u.Hz,
                amplitude=0.0 * u.Hz,
                frequency=0.0 * u.Hz,
                individual_spike_trains=False,
                rng_seed=7,
            )
            gen.init_state()

            for step in range(50):
                with brainstate.environ.context(t=step * dt):
                    out = np.asarray(gen.update(), dtype=np.int64)
                self.assertTrue(np.all(out == out.reshape(-1)[0]))

    def test_individual_spike_trains_true_are_not_forced_equal(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            gen = sinusoidal_poisson_generator(
                in_size=16,
                rate=50000.0 * u.Hz,
                amplitude=0.0 * u.Hz,
                frequency=0.0 * u.Hz,
                individual_spike_trains=True,
                rng_seed=11,
            )
            gen.init_state()

            has_non_uniform_step = False
            for step in range(80):
                with brainstate.environ.context(t=step * dt):
                    out = np.asarray(gen.update(), dtype=np.int64)
                if np.any(out != out.reshape(-1)[0]):
                    has_non_uniform_step = True
                    break

            self.assertTrue(has_non_uniform_step)


class TestSinusoidalPoissonGeneratorVsNEST(unittest.TestCase):
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
        amplitude_hz,
        frequency_hz,
        phase_deg,
        start_ms=0.0,
        stop_ms=None,
        origin_ms=0.0,
        individual_spike_trains=True,
    ):
        import nest

        n_steps = int(round(simtime_ms / dt_ms))
        nest.ResetKernel()
        nest.resolution = dt_ms
        nest.local_num_threads = 1
        nest.rng_seed = 12345

        params = {
            'rate': rate_hz,
            'amplitude': amplitude_hz,
            'frequency': frequency_hz,
            'phase': phase_deg,
            'individual_spike_trains': individual_spike_trains,
            'start': start_ms,
            'origin': origin_ms,
        }
        if stop_ms is not None:
            params['stop'] = stop_ms

        gens = nest.Create('sinusoidal_poisson_generator', n_trains, params=params)
        sr = nest.Create('spike_recorder')
        nest.Connect(gens, sr)
        nest.Simulate(simtime_ms)

        events = sr.get('events')
        if len(events['times']) == 0:
            return np.zeros(n_steps, dtype=np.float64)

        steps = np.rint(np.asarray(events['times'], dtype=np.float64) / dt_ms).astype(np.int64)
        counts = np.bincount(steps, minlength=n_steps + 2).astype(np.float64)

        # Spike recorder timestamps include one-step transmission delay.
        return counts[1:n_steps + 1]

    def test_mean_dynamics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.1
        simtime_ms = 320.0
        n_trains = 4096
        rate_hz = 900.0
        amplitude_hz = 450.0
        frequency_hz = 10.0
        phase_deg = 40.0

        nest_counts = self._run_nest_counts(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            amplitude_hz=amplitude_hz,
            frequency_hz=frequency_hz,
            phase_deg=phase_deg,
            individual_spike_trains=True,
        )
        bp_counts = _run_bp_counts(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            amplitude_hz=amplitude_hz,
            frequency_hz=frequency_hz,
            phase_deg=phase_deg,
            individual_spike_trains=True,
            rng_seed=12345,
        )

        # Align local send-time counts with NEST recorder timestamps (+1 step).
        bp_counts_aligned = np.zeros_like(bp_counts)
        bp_counts_aligned[1:] = bp_counts[:-1]

        n_steps = bp_counts_aligned.size
        times_ms = np.arange(1, n_steps + 1, dtype=np.float64) * dt_ms
        phi_rad = np.deg2rad(phase_deg)
        expected_rate = np.maximum(
            0.0,
            rate_hz + amplitude_hz * np.sin(2.0 * np.pi * frequency_hz * times_ms / 1000.0 + phi_rad),
        )
        expected_counts = n_trains * expected_rate * dt_ms / 1000.0

        peak_mask = expected_counts >= np.quantile(expected_counts, 0.9)
        trough_mask = expected_counts <= np.quantile(expected_counts, 0.1)

        nest_peak = float(np.mean(nest_counts[peak_mask]))
        bp_peak = float(np.mean(bp_counts_aligned[peak_mask]))
        nest_trough = float(np.mean(nest_counts[trough_mask]))
        bp_trough = float(np.mean(bp_counts_aligned[trough_mask]))

        self.assertAlmostEqual(bp_peak, nest_peak, delta=0.10 * max(nest_peak, 1.0))
        self.assertAlmostEqual(bp_trough, nest_trough, delta=0.16 * max(nest_trough, 1.0))

        mae_nest = float(np.mean(np.abs(nest_counts - expected_counts)))
        mae_bp = float(np.mean(np.abs(bp_counts_aligned - expected_counts)))
        self.assertAlmostEqual(mae_bp, mae_nest, delta=0.20 * max(mae_nest, 1.0))


if __name__ == '__main__':
    unittest.main()
