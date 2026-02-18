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

from brainpy_state._nest.mip_generator import mip_generator

brainstate.environ.set(precision=64, platform='cpu')


def _run_bp_matrix(
    dt_ms,
    simtime_ms,
    n_trains,
    *,
    rate_hz,
    p_copy,
    start_ms=0.0,
    stop_ms=None,
    origin_ms=0.0,
    rng_seed=0,
):
    dt = dt_ms * u.ms
    n_steps = int(round(simtime_ms / dt_ms))
    ditype = brainstate.environ.ditype()
    mat = np.zeros((n_steps, n_trains), dtype=ditype)

    with brainstate.environ.context(dt=dt):
        gen = mip_generator(
            in_size=n_trains,
            rate=rate_hz * u.Hz,
            p_copy=p_copy,
            start=start_ms * u.ms,
            stop=(stop_ms * u.ms) if stop_ms is not None else None,
            origin=origin_ms * u.ms,
            rng_seed=rng_seed,
        )
        gen.init_state()

        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt):
                mat[step] = np.asarray(gen.update(), dtype=ditype).reshape(-1)

    return mat


def _mean_pairwise_corr(mat: np.ndarray) -> float:
    dftype = brainstate.environ.dftype()
    mat = np.asarray(mat, dtype=dftype)
    if mat.ndim != 2 or mat.shape[1] < 2:
        return float('nan')
    corr = np.corrcoef(mat, rowvar=False)
    iu = np.triu_indices(corr.shape[0], k=1)
    vals = corr[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float('nan')
    return float(np.mean(vals))


class TestMIPGeneratorParameters(unittest.TestCase):
    def test_nest_default_parameters(self):
        gen = mip_generator()
        params = gen.get()
        self.assertEqual(params['rate'], 0.0)
        self.assertEqual(params['p_copy'], 1.0)
        self.assertEqual(params['start'], 0.0)
        self.assertTrue(np.isinf(params['stop']))
        self.assertEqual(params['origin'], 0.0)

    def test_parameter_validation(self):
        with self.assertRaisesRegex(ValueError, 'Rate must be non-negative'):
            mip_generator(rate=-1.0 * u.Hz)
        with self.assertRaisesRegex(ValueError, 'Copy probability must be in \\[0, 1\\]'):
            mip_generator(p_copy=-0.1)
        with self.assertRaisesRegex(ValueError, 'Copy probability must be in \\[0, 1\\]'):
            mip_generator(p_copy=1.1)
        with self.assertRaisesRegex(ValueError, 'stop >= start required'):
            mip_generator(start=2.0 * u.ms, stop=1.0 * u.ms)

    def test_grid_time_validation_matches_nest(self):
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            with self.assertRaisesRegex(ValueError, 'must be a multiple of the simulation resolution'):
                mip_generator(start=0.15 * u.ms)

    def test_set_validation(self):
        gen = mip_generator()
        with self.assertRaisesRegex(ValueError, 'Rate must be non-negative'):
            gen.set(rate=-2.0)
        with self.assertRaisesRegex(ValueError, 'Copy probability must be in \\[0, 1\\]'):
            gen.set(p_copy=2.0)


class TestMIPGeneratorOrdering(unittest.TestCase):
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
            gen = mip_generator(
                in_size=1,
                rate=1000.0 * u.Hz,
                p_copy=1.0,
                start=2.0 * u.ms,
                stop=5.0 * u.ms,
                rng_seed=1,
            )
            gen.init_state()
            gen._sample_parent_spikes = lambda lam: 1
            ditype = brainstate.environ.ditype()
            gen._sample_child_spikes = lambda n_parent_spikes: np.asarray([n_parent_spikes], dtype=ditype)

            trace = self._run_trace(gen, n_steps=7)
            self.assertEqual(trace, [0, 0, 0, 1, 1, 1, 0])

    def test_copy_probability_zero_blocks_all_children(self):
        with brainstate.environ.context(dt=self.dt):
            gen = mip_generator(
                in_size=1,
                rate=1000.0 * u.Hz,
                p_copy=0.0,
                start=-1.0 * u.ms,
                stop=4.0 * u.ms,
                rng_seed=2,
            )
            gen.init_state()
            gen._sample_parent_spikes = lambda lam: 5

            trace = self._run_trace(gen, n_steps=4)
            self.assertEqual(trace, [0, 0, 0, 0])

    def test_multiplicity_can_exceed_one(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            gen = mip_generator(
                in_size=1,
                rate=80000.0 * u.Hz,
                p_copy=1.0,
                rng_seed=3,
            )
            gen.init_state()

            max_count = 0
            for step in range(300):
                with brainstate.environ.context(t=step * 0.1 * u.ms):
                    max_count = max(max_count, int(np.asarray(gen.update())[0]))
            self.assertGreater(max_count, 1)


class TestMIPGeneratorStatistics(unittest.TestCase):
    def test_rate_and_pairwise_correlation_match_theory(self):
        dt_ms = 0.1
        simtime_ms = 8000.0
        n_trains = 24
        rate_hz = 120.0
        p_copy = 0.35

        mat = _run_bp_matrix(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            p_copy=p_copy,
            rng_seed=11,
        )

        simtime_s = simtime_ms * 1e-3
        rate_sim = float(mat.sum()) / (n_trains * simtime_s)
        corr_sim = _mean_pairwise_corr(mat)

        expected_rate = rate_hz * p_copy
        self.assertAlmostEqual(rate_sim, expected_rate, delta=0.12 * expected_rate)
        self.assertAlmostEqual(corr_sim, p_copy, delta=0.12)


class TestMIPGeneratorVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest
            if hasattr(nest, 'node_models'):
                return 'mip_generator' in nest.node_models
            return 'mip_generator' in nest.Models()
        except Exception:
            return False

    def _run_nest_matrix(
        self,
        dt_ms,
        simtime_ms,
        n_trains,
        *,
        rate_hz,
        p_copy,
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
            'rate': float(rate_hz),
            'p_copy': float(p_copy),
            'start': float(start_ms),
            'origin': float(origin_ms),
        }
        if stop_ms is not None:
            params['stop'] = float(stop_ms)

        mg = nest.Create('mip_generator', params=params)
        pn = nest.Create('parrot_neuron', n_trains)
        sr = nest.Create('spike_recorder')

        nest.Connect(mg, pn)
        nest.Connect(pn, sr)
        nest.Simulate(simtime_ms)

        ditype = brainstate.environ.ditype()
        mat = np.zeros((n_steps, n_trains), dtype=ditype)
        events = sr.get('events')
        if len(events['times']) == 0:
            return mat

        try:
            gids = np.asarray(pn.get('global_id'), dtype=ditype)
        except Exception:
            gids = np.asarray(pn.tolist(), dtype=ditype)
        id_to_idx = {int(g): i for i, g in enumerate(gids)}

        dftype = brainstate.environ.dftype()
        steps = np.rint(np.asarray(events['times'], dtype=dftype) / dt_ms).astype(np.int64)
        senders = np.asarray(events['senders'], dtype=ditype)

        for step, sender in zip(steps, senders):
            i = id_to_idx.get(int(sender), None)
            if i is None:
                continue
            if 0 <= step < n_steps:
                mat[step, i] += 1

        return mat

    def test_statistics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        dt_ms = 0.1
        simtime_ms = 8000.0
        n_trains = 24
        rate_hz = 120.0
        p_copy = 0.35

        nest_mat = self._run_nest_matrix(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            p_copy=p_copy,
            start_ms=0.0,
            stop_ms=None,
            origin_ms=0.0,
        )
        bp_mat = _run_bp_matrix(
            dt_ms=dt_ms,
            simtime_ms=simtime_ms,
            n_trains=n_trains,
            rate_hz=rate_hz,
            p_copy=p_copy,
            start_ms=0.0,
            stop_ms=None,
            origin_ms=0.0,
            rng_seed=12345,
        )

        simtime_s = simtime_ms * 1e-3
        expected_rate = rate_hz * p_copy

        nest_rate = float(nest_mat.sum()) / (n_trains * simtime_s)
        bp_rate = float(bp_mat.sum()) / (n_trains * simtime_s)

        nest_corr = _mean_pairwise_corr(nest_mat)
        bp_corr = _mean_pairwise_corr(bp_mat)

        self.assertAlmostEqual(nest_rate, expected_rate, delta=0.18 * expected_rate)
        self.assertAlmostEqual(bp_rate, expected_rate, delta=0.18 * expected_rate)
        self.assertAlmostEqual(bp_rate, nest_rate, delta=0.12 * max(1.0, nest_rate))

        self.assertAlmostEqual(nest_corr, p_copy, delta=0.12)
        self.assertAlmostEqual(bp_corr, p_copy, delta=0.12)
        self.assertAlmostEqual(bp_corr, nest_corr, delta=0.08)


if __name__ == '__main__':
    unittest.main()
