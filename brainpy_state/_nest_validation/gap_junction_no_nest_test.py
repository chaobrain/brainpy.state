# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""CI-safe companion for the gap-junction examples (no NEST required).

Exercises the ported example helpers headless: the 2-neuron pair synchronizes under gap
coupling (and does *not* without it), a tiny inhibitory network shows the gap-driven
synchronization law, and the synchrony helpers behave. The live-NEST parity lives in
``gap_junction_parity_test.py`` and ``gap_junction_inhibitory_network_parity_test.py``
(both ``@requires_nest``).
"""
import unittest

import jax
import brainstate
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from examples.nest_like.gap_junctions_two_neurons import run_two_neuron, synchrony_gap
from examples.nest_like.gap_junctions_inhibitory_network import run_network, golomb_chi


class TestGapTwoNeuronNoNest(unittest.TestCase):
    """The 2-neuron example synchronizes under coupling and stays finite."""

    def test_coupled_pair_converges_to_synchrony(self):
        """g = 0.5 nS: the membrane gap collapses from ~30 mV to sub-mV (synchrony)."""
        out = run_two_neuron(gap_weight=0.5, T=351.0)
        self.assertTrue(np.all(np.isfinite(out['V'])))
        early, late = synchrony_gap(out['V'])
        self.assertGreater(early, 10.0)             # genuinely desynchronized start
        self.assertLess(late, 2.0)                  # converged to near-synchrony
        self.assertLess(late, 0.2 * early)

    def test_uncoupled_pair_does_not_synchronize(self):
        """g = 0: without the gap the two cells keep a large phase gap (coupling is causal)."""
        coupled = synchrony_gap(run_two_neuron(gap_weight=0.5, T=351.0)['V'])[1]
        uncoupled = synchrony_gap(run_two_neuron(gap_weight=0.0, T=351.0)['V'])[1]
        self.assertGreater(uncoupled, 3.0)          # stays desynchronized
        self.assertGreater(uncoupled, 3.0 * coupled)  # coupling is what synchronizes


class TestGapNetworkNoNest(unittest.TestCase):
    """A tiny inhibitory network reproduces the gap-driven synchronization law."""

    def test_gap_increases_population_coherence(self):
        """chi rises when the gap graph is added (the network synchronization law)."""
        chi_async = golomb_chi(run_network(0.0, seed=0, n_neuron=60, inh=15, gap_k=10,
                                           T=300.0, record_spikes=False)['V'])
        chi_sync = golomb_chi(run_network(2.0, seed=0, n_neuron=60, inh=15, gap_k=10,
                                          T=300.0, record_spikes=False)['V'])
        for c in (chi_async, chi_sync):
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)
        self.assertGreater(chi_sync, chi_async)     # gap junctions raise synchrony

    def test_network_runs_and_records_spikes(self):
        """The example wires a Poisson-driven net + spike recorder and runs headless."""
        out = run_network(0.5, seed=2, n_neuron=40, inh=10, gap_k=8, T=120.0)
        self.assertTrue(np.all(np.isfinite(out['V'])))
        self.assertEqual(out['V'].shape[1], 40)
        self.assertIsNotNone(out['spikes'])
        self.assertGreaterEqual(float(np.sum(out['spikes'] > 0)), 0.0)


class TestGapSynchronyHelpers(unittest.TestCase):
    """The synchrony helpers behave on hand-built signals."""

    def test_golomb_chi_bounds(self):
        # identical neurons -> fully synchronous (chi == 1); independent noise -> small chi.
        t = np.linspace(0, 1, 500)
        sync = np.stack([np.sin(2 * np.pi * 5 * t)] * 8, axis=1)
        self.assertAlmostEqual(golomb_chi(sync, skip_ms=0.0, dt=1.0), 1.0, places=6)
        rng = np.random.default_rng(0)
        asyn = rng.standard_normal((500, 50))
        self.assertLess(golomb_chi(asyn, skip_ms=0.0, dt=1.0), 0.3)

    def test_synchrony_gap_windows(self):
        # a trace that starts split and ends merged -> early >> late.
        V = np.zeros((1000, 2))
        V[:, 0] = np.r_[np.full(500, 20.0), np.zeros(500)]
        early, late = synchrony_gap(V, window_ms=5.0, dt=0.05)
        self.assertGreater(early, 15.0)
        self.assertEqual(late, 0.0)


if __name__ == '__main__':
    unittest.main()
