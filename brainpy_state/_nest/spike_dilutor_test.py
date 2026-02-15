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

from brainpy_state._nest.spike_dilutor import spike_dilutor

brainstate.environ.set(precision=64, platform='cpu')


class TestSpikeDilutorParameters(unittest.TestCase):
    def test_nest_default_parameters(self):
        dil = spike_dilutor()
        params = dil.get()
        self.assertEqual(params['p_copy'], 1.0)
        self.assertEqual(params['start'], 0.0)
        self.assertTrue(np.isinf(params['stop']))
        self.assertEqual(params['origin'], 0.0)

    def test_parameter_validation(self):
        with self.assertRaisesRegex(ValueError, 'Copy probability must be in \\[0, 1\\]'):
            spike_dilutor(p_copy=-0.1)
        with self.assertRaisesRegex(ValueError, 'Copy probability must be in \\[0, 1\\]'):
            spike_dilutor(p_copy=1.1)
        with self.assertRaisesRegex(ValueError, 'stop >= start required'):
            spike_dilutor(start=2.0 * u.ms, stop=1.0 * u.ms)

    def test_grid_time_validation_matches_nest(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            with self.assertRaisesRegex(ValueError, 'must be a multiple of the simulation resolution'):
                spike_dilutor(start=0.15 * u.ms)

    def test_set_validation(self):
        dil = spike_dilutor()
        with self.assertRaisesRegex(ValueError, 'Copy probability must be in \\[0, 1\\]'):
            dil.set(p_copy=2.0)
        with self.assertRaisesRegex(ValueError, 'stop >= start required'):
            dil.set(start=2.0 * u.ms, stop=1.0 * u.ms)

    def test_negative_mother_spike_input_raises(self):
        with brainstate.environ.context(dt=0.1 * u.ms, t=0.1 * u.ms):
            dil = spike_dilutor()
            dil.init_state()
            with self.assertRaisesRegex(ValueError, 'mother_spikes must be non-negative'):
                dil.update(-1.0)


class TestSpikeDilutorOrdering(unittest.TestCase):
    def setUp(self):
        self.dt = 1.0 * u.ms

    def _run_trace(self, dil, n_steps, mother_spikes):
        trace = []
        for step in range(n_steps):
            with brainstate.environ.context(t=step * self.dt):
                trace.append(int(np.asarray(dil.update(mother_spikes=mother_spikes))[0]))
        return trace

    def test_start_exclusive_stop_inclusive(self):
        with brainstate.environ.context(dt=self.dt):
            dil = spike_dilutor(
                in_size=1,
                p_copy=1.0,
                start=2.0 * u.ms,
                stop=5.0 * u.ms,
                rng_seed=1,
            )
            dil.init_state()
            trace = self._run_trace(dil, n_steps=7, mother_spikes=1.0)
            self.assertEqual(trace, [0, 0, 0, 1, 1, 1, 0])

    def test_input_is_sum_of_argument_and_delta_inputs(self):
        with brainstate.environ.context(dt=self.dt):
            dil = spike_dilutor(in_size=1, p_copy=1.0, start=-1.0 * u.ms, rng_seed=1)
            dil.init_state()
            dil.add_delta_input('mother0', 3.0)

            with brainstate.environ.context(t=0.0 * u.ms):
                out = dil.update(mother_spikes=2.0)
            self.assertEqual(int(np.asarray(out)[0]), 5)

    def test_explicit_bernoulli_loop_semantics(self):
        p_copy = 0.35
        n_targets = 4
        n_mother = 7
        seed = 17

        with brainstate.environ.context(dt=self.dt):
            dil = spike_dilutor(
                in_size=n_targets,
                p_copy=p_copy,
                start=-1.0 * u.ms,
                rng_seed=seed,
            )
            dil.init_state()
            with brainstate.environ.context(t=0.0 * u.ms):
                got = np.asarray(dil.update(mother_spikes=n_mother), dtype=np.int64).reshape(-1)

        rng = np.random.default_rng(seed)
        expected = np.asarray(
            [np.count_nonzero(rng.random(n_mother) < p_copy) for _ in range(n_targets)],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(got, expected)

    def test_copy_probability_extremes(self):
        with brainstate.environ.context(dt=self.dt):
            d0 = spike_dilutor(in_size=3, p_copy=0.0, start=-1.0 * u.ms, rng_seed=1)
            d1 = spike_dilutor(in_size=3, p_copy=1.0, start=-1.0 * u.ms, rng_seed=1)
            d0.init_state()
            d1.init_state()

            with brainstate.environ.context(t=0.0 * u.ms):
                out0 = np.asarray(d0.update(mother_spikes=9), dtype=np.int64).reshape(-1)
                out1 = np.asarray(d1.update(mother_spikes=9), dtype=np.int64).reshape(-1)

            np.testing.assert_array_equal(out0, np.zeros(3, dtype=np.int64))
            np.testing.assert_array_equal(out1, np.full(3, 9, dtype=np.int64))


class TestSpikeDilutorStatistics(unittest.TestCase):
    def test_mean_matches_nest_unittest_spec(self):
        # Equivalent to NEST's test_spike_dilutor.sli:
        # 5000 mother spikes, p_copy=0.2, 10 targets, mean should be within 5%.
        dt = 1.0 * u.ms
        n_spikes = 5000
        n_targets = 10
        p_copy = 0.2
        expected = n_spikes * p_copy

        with brainstate.environ.context(dt=dt):
            dil = spike_dilutor(in_size=n_targets, p_copy=p_copy, rng_seed=12345)
            dil.init_state()

            counts = np.zeros(n_targets, dtype=np.float64)
            for step in range(1, n_spikes + 1):
                with brainstate.environ.context(t=step * dt):
                    counts += np.asarray(dil.update(mother_spikes=1.0), dtype=np.float64).reshape(-1)

        avg_count = float(np.mean(counts))
        rel_err = abs(avg_count - expected) / expected
        self.assertLess(rel_err, 0.05)


class TestSpikeDilutorVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest
            if hasattr(nest, 'node_models'):
                return 'spike_dilutor' in nest.node_models
            return 'spike_dilutor' in nest.Models()
        except Exception:
            return False

    @staticmethod
    def _run_nest_average_count(n_spikes, n_targets, p_copy):
        import nest

        nest.ResetKernel()
        nest.resolution = 1.0
        nest.local_num_threads = 1
        nest.rng_seed = 12345

        spike_times = np.arange(1.0, float(n_spikes) + 1.0, 1.0, dtype=np.float64)
        sg = nest.Create('spike_generator', params={'spike_times': spike_times})
        dil = nest.Create('spike_dilutor', params={'p_copy': float(p_copy)})
        sr = nest.Create('spike_recorder', n_targets)

        nest.Connect(sg, dil)
        nest.Connect(dil, sr)
        nest.Simulate(float(n_spikes + 2))

        try:
            events = np.asarray(sr.get('n_events'), dtype=np.float64)
        except Exception:
            events = np.asarray(nest.GetStatus(sr, 'n_events'), dtype=np.float64)
        return float(np.mean(events))

    @staticmethod
    def _run_bp_average_count(n_spikes, n_targets, p_copy):
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            dil = spike_dilutor(in_size=n_targets, p_copy=p_copy, rng_seed=12345)
            dil.init_state()

            counts = np.zeros(n_targets, dtype=np.float64)
            for step in range(1, n_spikes + 1):
                with brainstate.environ.context(t=step * dt):
                    counts += np.asarray(dil.update(mother_spikes=1.0), dtype=np.float64).reshape(-1)
        return float(np.mean(counts))

    def test_mean_dynamics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        n_spikes = 5000
        n_targets = 10
        p_copy = 0.2

        nest_mean = self._run_nest_average_count(n_spikes=n_spikes, n_targets=n_targets, p_copy=p_copy)
        bp_mean = self._run_bp_average_count(n_spikes=n_spikes, n_targets=n_targets, p_copy=p_copy)

        rel_diff = abs(bp_mean - nest_mean) / max(nest_mean, 1.0)
        self.assertLess(rel_diff, 0.05)


if __name__ == '__main__':
    unittest.main()
