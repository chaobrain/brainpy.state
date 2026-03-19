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

import math
import unittest

import brainstate
import saiunit as u
import jax
import numpy as np

from brainpy.state import bernoulli_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class _MockReceiver:
    def __init__(self):
        self.delta_events = []

    def add_delta_input(self, key, inp, label=None):
        self.delta_events.append((key, inp, label))


def _run_bp_transmitted_count(*, n_spikes, p_transmit, seed):
    np.random.seed(seed)
    recv = _MockReceiver()
    dt = 1.0 * u.ms

    with brainstate.environ.context(dt=dt):
        syn = bernoulli_synapse(
            weight=1.0,
            delay=1.0 * u.ms,
            p_transmit=p_transmit,
            post=recv,
            event_type='spike',
        )
        syn.init_state()

        for step in range(n_spikes + 2):
            pre_spike = 1.0 if 1 <= step <= n_spikes else 0.0
            with brainstate.environ.context(t=step * dt):
                syn.update(pre_spike=pre_spike)

    return len(recv.delta_events)


class TestBernoulliSynapseParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = bernoulli_synapse()
            syn.init_state()
            syn.update(pre_spike=0.0)
            params = syn.get()

        self.assertEqual(params['weight'], 1.0)
        self.assertEqual(params['delay'], 1.0)
        self.assertEqual(params['delay_steps'], 1)
        self.assertEqual(params['receptor_type'], 0)
        self.assertEqual(params['event_type'], 'spike')
        self.assertEqual(params['p_transmit'], 1.0)
        self.assertEqual(params['synapse_model'], 'bernoulli_synapse')

    def test_p_transmit_must_be_in_unit_interval(self):
        with self.assertRaisesRegex(ValueError, 'Spike transmission probability must be in \\[0, 1\\]'):
            bernoulli_synapse(p_transmit=-0.1)

        with self.assertRaisesRegex(ValueError, 'Spike transmission probability must be in \\[0, 1\\]'):
            bernoulli_synapse(p_transmit=1.1)

        syn = bernoulli_synapse()
        with self.assertRaisesRegex(ValueError, 'Spike transmission probability must be in \\[0, 1\\]'):
            syn.set(p_transmit=1.2)

    def test_p_transmit_must_be_scalar(self):
        with self.assertRaisesRegex(ValueError, 'p_transmit must be scalar'):
            dftype = brainstate.environ.dftype()
            bernoulli_synapse(p_transmit=np.asarray([0.1, 0.2], dtype=dftype))


class TestBernoulliSynapseOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_bernoulli_trial_applied_before_event_scheduling(self):
        recv = _MockReceiver()
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            syn = bernoulli_synapse(
                weight=2.5,
                delay=2.0 * u.ms,
                receptor_type=7,
                p_transmit=0.5,
                post=recv,
            )
            syn.init_state()

            draws = iter([0.2, 0.8, 0.1])  # send, drop, send
            syn._sample_uniform = lambda: float(next(draws))

            delivered = []
            for step in range(6):
                pre = 1.0 if step in (0, 1, 2) else 0.0
                with brainstate.environ.context(t=step * dt):
                    delivered.append(syn.update(pre_spike=pre))

        self.assertEqual(delivered, [0, 0, 1, 0, 1, 0])
        self.assertEqual(len(recv.delta_events), 2)

        _, val0, lbl0 = recv.delta_events[0]
        _, val1, lbl1 = recv.delta_events[1]
        dftype = brainstate.environ.dftype()
        val0_f = float(np.asarray(u.math.asarray(val0), dtype=dftype).reshape(()))
        val1_f = float(np.asarray(u.math.asarray(val1), dtype=dftype).reshape(()))
        self.assertAlmostEqual(val0_f, 2.5, delta=1e-12)
        self.assertAlmostEqual(val1_f, 2.5, delta=1e-12)
        self.assertEqual(lbl0, 'receptor_7')
        self.assertEqual(lbl1, 'receptor_7')

    def test_multiplicity_scales_weight_on_success(self):
        recv = _MockReceiver()
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            syn = bernoulli_synapse(weight=1.5, delay=1.0 * u.ms, p_transmit=1.0, post=recv)
            syn.init_state()
            with brainstate.environ.context(t=0.0 * u.ms):
                syn.update(pre_spike=3.0)
            with brainstate.environ.context(t=1.0 * u.ms):
                syn.update(pre_spike=0.0)

        self.assertEqual(len(recv.delta_events), 1)
        _, value, _ = recv.delta_events[0]
        dftype = brainstate.environ.dftype()
        value_f = float(np.asarray(u.math.asarray(value), dtype=dftype).reshape(()))
        self.assertAlmostEqual(value_f, 4.5, delta=1e-12)


class TestBernoulliSynapseStatistics(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_statistics_match_binomial_expectation(self):
        # Mirrors NEST testsuite/pytests/test_bernoulli_synapse.py.
        n_spikes = 1000
        p_transmit = 0.25
        expected = n_spikes * p_transmit
        margin = 3.0 * math.sqrt(n_spikes * p_transmit * (1.0 - p_transmit))

        for seed in range(123, 133):
            with self.subTest(seed=seed):
                count = _run_bp_transmitted_count(
                    n_spikes=n_spikes,
                    p_transmit=p_transmit,
                    seed=seed,
                )
                self.assertLessEqual(abs(count - expected), margin)


class TestBernoulliSynapseVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest
            if hasattr(nest, 'synapse_models'):
                return 'bernoulli_synapse' in nest.synapse_models
            return 'bernoulli_synapse' in nest.Models()
        except Exception:
            return False

    @staticmethod
    def _run_nest_transmitted_count(*, n_spikes, p_transmit, seed):
        import nest

        nest.ResetKernel()
        try:
            nest.set(resolution=1.0, local_num_threads=1)
            nest.set(rng_seed=int(seed))
        except Exception:
            nest.resolution = 1.0
            nest.local_num_threads = 1
            nest.rng_seed = int(seed)

        dftype = brainstate.environ.dftype()
        sg = nest.Create(
            'spike_generator',
            {'spike_times': np.arange(1.0, float(n_spikes) + 1.0, 1.0, dtype=dftype)},
        )
        pre = nest.Create('parrot_neuron')
        post = nest.Create('parrot_neuron')
        sr = nest.Create('spike_recorder')

        nest.Connect(sg, pre)
        nest.Connect(pre, post, syn_spec={'synapse_model': 'bernoulli_synapse', 'p_transmit': float(p_transmit)})
        nest.Connect(post, sr)
        nest.Simulate(float(n_spikes + 2))

        events = sr.get('events')
        return int(np.asarray(events['times'], dtype=dftype).size)

    def test_population_statistics_match_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        n_spikes = 1000
        p_transmit = 0.25
        seeds = range(123, 133)

        dftype = brainstate.environ.dftype()
        nest_counts = np.asarray(
            [
                self._run_nest_transmitted_count(
                    n_spikes=n_spikes,
                    p_transmit=p_transmit,
                    seed=seed,
                )
                for seed in seeds
            ],
            dtype=dftype,
        )
        bp_counts = np.asarray(
            [
                _run_bp_transmitted_count(
                    n_spikes=n_spikes,
                    p_transmit=p_transmit,
                    seed=seed,
                )
                for seed in seeds
            ],
            dtype=dftype,
        )

        expected = n_spikes * p_transmit
        margin = 3.0 * math.sqrt(n_spikes * p_transmit * (1.0 - p_transmit))
        self.assertLessEqual(abs(float(np.mean(nest_counts)) - expected), margin)
        self.assertLessEqual(abs(float(np.mean(bp_counts)) - expected), margin)
        self.assertLessEqual(abs(float(np.mean(bp_counts)) - float(np.mean(nest_counts))), 25.0)


if __name__ == '__main__':
    unittest.main()
