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
import numpy.testing as npt

from brainpy_state._nest.sinusoidal_gamma_generator import (
    sinusoidal_gamma_generator,
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
    order,
    start_ms=0.0,
    stop_ms=None,
    origin_ms=0.0,
    individual_spike_trains=True,
    rng_seed=0,
):
    dt = dt_ms * u.ms
    n_steps = int(round(simtime_ms / dt_ms))
    dftype = brainstate.environ.dftype()
    totals = np.zeros(n_steps, dtype=dftype)

    with brainstate.environ.context(dt=dt):
        gen = sinusoidal_gamma_generator(
            in_size=n_trains,
            rate=rate_hz * u.Hz,
            amplitude=amplitude_hz * u.Hz,
            frequency=frequency_hz * u.Hz,
            phase=phase_deg,
            order=order,
            individual_spike_trains=individual_spike_trains,
            start=start_ms * u.ms,
            stop=(stop_ms * u.ms) if stop_ms is not None else None,
            origin=origin_ms * u.ms,
            rng_seed=rng_seed,
        )
        gen.init_state()

        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt):
                totals[step] = float(np.asarray(gen.update(), dtype=dftype).sum())

    return totals


class TestSinusoidalGammaGeneratorParameters(unittest.TestCase):
    def test_nest_default_parameters(self):
        gen = sinusoidal_gamma_generator()
        params = gen.get()
        self.assertEqual(params['rate'], 0.0)
        self.assertEqual(params['amplitude'], 0.0)
        self.assertEqual(params['frequency'], 0.0)
        self.assertEqual(params['phase'], 0.0)
        self.assertEqual(params['order'], 1.0)
        self.assertTrue(params['individual_spike_trains'])
        self.assertEqual(params['start'], 0.0)
        self.assertTrue(np.isinf(params['stop']))
        self.assertEqual(params['origin'], 0.0)

    def test_order_must_be_at_least_one(self):
        with self.assertRaisesRegex(ValueError, 'gamma order must be at least 1'):
            sinusoidal_gamma_generator(order=0.99)

    def test_amplitude_constraints(self):
        with self.assertRaisesRegex(ValueError, '0 <= amplitude <= rate'):
            sinusoidal_gamma_generator(rate=10.0 * u.Hz, amplitude=11.0 * u.Hz)
        with self.assertRaisesRegex(ValueError, '0 <= amplitude <= rate'):
            sinusoidal_gamma_generator(rate=10.0 * u.Hz, amplitude=-1.0 * u.Hz)

    def test_stop_before_start_raises(self):
        with self.assertRaisesRegex(ValueError, 'stop >= start required'):
            sinusoidal_gamma_generator(start=2.0 * u.ms, stop=1.0 * u.ms)

    def test_grid_time_validation_matches_nest_behavior(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            with self.assertRaisesRegex(
                ValueError,
                'must be a multiple of the simulation resolution',
            ):
                sinusoidal_gamma_generator(start=0.15 * u.ms)

    def test_set_parameter_validation(self):
        gen = sinusoidal_gamma_generator()
        with self.assertRaisesRegex(ValueError, 'gamma order must be at least 1'):
            gen.set(order=0.5)
        with self.assertRaisesRegex(ValueError, '0 <= amplitude <= rate'):
            gen.set(rate=1.0, amplitude=2.0)


class TestSinusoidalGammaGeneratorOrdering(unittest.TestCase):
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
            gen = sinusoidal_gamma_generator(
                in_size=1,
                rate=1000.0 * u.Hz,  # order=1 -> hazard = 1 at dt=1 ms
                amplitude=0.0 * u.Hz,
                frequency=0.0 * u.Hz,
                order=1.0,
                start=2.0 * u.ms,
                stop=5.0 * u.ms,
                rng_seed=1,
            )
            gen.init_state()
            dftype = brainstate.environ.dftype()
            gen._sample_uniform = lambda shape=(): np.full(shape, 0.5, dtype=dftype)

            trace = self._run_trace(gen, n_steps=7)
            # Spike-device activity interval in NEST: start < step <= stop.
            self.assertEqual(trace, [0, 0, 0, 1, 1, 1, 0])

    def test_recorded_rate_matches_sinusoidal_profile(self):
        dt_ms = 0.1
        dt = dt_ms * u.ms
        n_steps = 1000

        dc_hz = 1.0
        ac_hz = 0.5
        freq_hz = 10.0
        phi_rad = 2.0
        phase_deg = phi_rad / np.pi * 180.0

        dftype = brainstate.environ.dftype()
        recorded = np.zeros(n_steps, dtype=dftype)
        with brainstate.environ.context(dt=dt):
            gen = sinusoidal_gamma_generator(
                in_size=1,
                rate=dc_hz * u.Hz,
                amplitude=ac_hz * u.Hz,
                frequency=freq_hz * u.Hz,
                phase=phase_deg,
                order=2.0,
                rng_seed=3,
            )
            gen.init_state()
            for step in range(n_steps):
                with brainstate.environ.context(t=step * dt):
                    gen.update()
                recorded[step] = gen.get_recorded_rate()

        times_ms = (np.arange(n_steps, dtype=dftype) + 1.0) * dt_ms
        expected = dc_hz + ac_hz * np.sin(
            2.0 * np.pi * freq_hz * times_ms / 1000.0 + phi_rad
        )
        npt.assert_allclose(recorded, expected, rtol=0.0, atol=1e-12)

    def test_set_accumulates_lambda_piecewise(self):
        with brainstate.environ.context(dt=self.dt):
            gen = sinusoidal_gamma_generator(
                in_size=1,
                rate=500.0 * u.Hz,
                amplitude=0.0 * u.Hz,
                frequency=0.0 * u.Hz,
                order=2.0,
                rng_seed=5,
            )
            gen.init_state()
            dftype = brainstate.environ.dftype()
            gen._sample_uniform = lambda shape=(): np.full(shape, 2.0, dtype=dftype)

            for step in range(3):
                with brainstate.environ.context(t=step * self.dt):
                    gen.update()

            with brainstate.environ.context(t=3.0 * self.dt):
                gen.set(rate=1000.0 * u.Hz)

            lambda_t0 = float(np.asarray(gen.Lambda_t0.value, dtype=dftype)[0])
            t0_ms = float(np.asarray(gen.t0_ms.value, dtype=dftype)[0])

            # For order=2, rate=500 Hz (=0.5/ms), over [0,3] ms:
            # Lambda = order * rate * dt = 2 * 0.5 * 3 = 3.
            self.assertAlmostEqual(lambda_t0, 3.0, places=12)
            self.assertAlmostEqual(t0_ms, 3.0, places=12)

    def test_individual_spike_trains_false_broadcasts_multiplicity(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            gen = sinusoidal_gamma_generator(
                in_size=16,
                rate=5000.0 * u.Hz,
                amplitude=0.0 * u.Hz,
                frequency=0.0 * u.Hz,
                order=1.0,
                individual_spike_trains=False,
                rng_seed=7,
            )
            gen.init_state()

            for step in range(50):
                with brainstate.environ.context(t=step * dt):
                    ditype = brainstate.environ.ditype()
                    out = np.asarray(gen.update(), dtype=ditype)
                self.assertTrue(np.all(out == out.reshape(-1)[0]))

    def test_individual_spike_trains_true_are_not_forced_equal(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            gen = sinusoidal_gamma_generator(
                in_size=16,
                rate=5000.0 * u.Hz,
                amplitude=0.0 * u.Hz,
                frequency=0.0 * u.Hz,
                order=1.0,
                individual_spike_trains=True,
                rng_seed=11,
            )
            gen.init_state()

            has_non_uniform_step = False
            for step in range(80):
                with brainstate.environ.context(t=step * dt):
                    ditype = brainstate.environ.ditype()
                    out = np.asarray(gen.update(), dtype=ditype)
                if np.any(out != out.reshape(-1)[0]):
                    has_non_uniform_step = True
                    break

            self.assertTrue(has_non_uniform_step)


class TestSinusoidalGammaGeneratorVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest

            if hasattr(nest, 'node_models'):
                return 'sinusoidal_gamma_generator' in nest.node_models
            return 'sinusoidal_gamma_generator' in nest.Models()
        except Exception:
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
        order,
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
            'order': order,
            'individual_spike_trains': individual_spike_trains,
            'start': start_ms,
            'origin': origin_ms,
        }
        if stop_ms is not None:
            params['stop'] = stop_ms

        gens = nest.Create('sinusoidal_gamma_generator', n_trains, params=params)
        sr = nest.Create('spike_recorder')
        nest.Connect(gens, sr)
        nest.Simulate(simtime_ms)

        events = sr.get('events')
        if len(events['times']) == 0:
            dftype = brainstate.environ.dftype()
            return np.zeros(n_steps, dtype=dftype)

        steps = np.rint(np.asarray(events['times'], dtype=dftype) / dt_ms).astype(np.int64)
        counts = np.bincount(steps, minlength=n_steps + 2).astype(np.float64)

        # Spike recorder timestamps include one-step transmission delay.
        return counts[1:n_steps + 1]

    def test_mean_dynamics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.1
        simtime_ms = 320.0
        n_trains = 1024
        rate_hz = 900.0
        amplitude_hz = 450.0
        frequency_hz = 10.0
        phase_deg = 40.0
        order = 3.0

        nest_counts = self._run_nest_counts(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            amplitude_hz=amplitude_hz,
            frequency_hz=frequency_hz,
            phase_deg=phase_deg,
            order=order,
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
            order=order,
            individual_spike_trains=True,
            rng_seed=12345,
        )

        # Align local send-time counts with NEST recorder timestamps (+1 step).
        bp_counts_aligned = np.zeros_like(bp_counts)
        bp_counts_aligned[1:] = bp_counts[:-1]

        n_steps = bp_counts_aligned.size
        dftype = brainstate.environ.dftype()
        times_ms = np.arange(1, n_steps + 1, dtype=dftype) * dt_ms
        phi_rad = np.deg2rad(phase_deg)
        expected_rate = rate_hz + amplitude_hz * np.sin(
            2.0 * np.pi * frequency_hz * times_ms / 1000.0 + phi_rad
        )
        expected_counts = n_trains * expected_rate * dt_ms / 1000.0

        peak_mask = expected_counts >= np.quantile(expected_counts, 0.9)
        trough_mask = expected_counts <= np.quantile(expected_counts, 0.1)

        nest_peak = float(np.mean(nest_counts[peak_mask]))
        bp_peak = float(np.mean(bp_counts_aligned[peak_mask]))
        nest_trough = float(np.mean(nest_counts[trough_mask]))
        bp_trough = float(np.mean(bp_counts_aligned[trough_mask]))

        self.assertAlmostEqual(bp_peak, nest_peak, delta=0.14 * max(nest_peak, 1.0))
        self.assertAlmostEqual(bp_trough, nest_trough, delta=0.18 * max(nest_trough, 1.0))

        mae_nest = float(np.mean(np.abs(nest_counts - expected_counts)))
        mae_bp = float(np.mean(np.abs(bp_counts_aligned - expected_counts)))
        self.assertAlmostEqual(mae_bp, mae_nest, delta=0.22 * max(mae_nest, 1.0))


if __name__ == '__main__':
    unittest.main()
