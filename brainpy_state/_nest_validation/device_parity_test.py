# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Parity for stimulation/recording devices: poisson_generator, spike_recorder."""
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainpy_state

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest_validation.tolerance_conventions import CAT_D


class TestPoissonRate(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_empirical_rate_close_to_configured(self):
        import brainstate.transform as transform
        n_trains, rate_hz, T_ms = 2000, 1000.0, 1000.0
        gen = brainstate.nn.init_all_states(
            brainpy_state.poisson_generator(n_trains, rate=rate_hz * u.Hz, rng_seed=0))
        times = u.math.arange(0.0 * u.ms, T_ms * u.ms, 0.1 * u.ms)
        indices = u.math.arange(times.size)

        def step(t, i):
            with brainstate.environ.context(t=t, i=i):
                return jnp.sum(gen.update())

        per_step = transform.for_loop(step, times, indices)
        total = float(jnp.sum(per_step))
        emp_rate = total / n_trains / (T_ms / 1000.0)
        # empirical Poisson rate vs configured ground-truth -> distributional (category D).
        compare_distributional([rate_hz], [emp_rate], tol=CAT_D, metric="poisson rate").assert_()


@requires_nest
class TestPoissonRateVsNest(unittest.TestCase):
    def test_mean_count_matches_nest_within_tolerance(self):
        nest.ResetKernel()
        nest.resolution = 0.1
        n = nest.Create("parrot_neuron", 2000)
        g = nest.Create("poisson_generator", params={"rate": 1000.0})
        sr = nest.Create("spike_recorder")
        nest.Connect(g, n, syn_spec={"delay": 0.1})
        nest.Connect(n, sr)
        nest.Simulate(1000.0)
        nest_rate = sr.n_events / 2000 / 1.0
        # live-NEST Poisson rate vs configured ground-truth -> distributional (category D).
        compare_distributional([1000.0], [nest_rate], tol=CAT_D, metric="nest poisson rate").assert_()


class TestSpikeRecorderStamp(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_counts_and_stamp_step(self):
        sr = brainstate.nn.init_all_states(brainpy_state.spike_recorder())
        with brainstate.environ.context(t=1.0 * u.ms, i=10):
            sr.update(spikes=np.array([1., 0., 2.]), senders=np.array([3, 4, 5]))
        ev = sr.events
        self.assertEqual(sr.n_events, 3)                  # 1 + 0 + 2
        # stamp step = round(1.0/0.1) + 1 = 11 -> time 1.1 ms
        self.assertTrue(np.allclose(np.unique(ev["times"]), [1.1]))
