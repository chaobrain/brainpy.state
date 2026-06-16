# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/one_neuron_with_noise.py``.

NEST's ``one_neuron_with_noise`` drives an ``iaf_psc_alpha`` from a 2-channel
``poisson_generator`` (rates 80 kHz / 15 kHz) with signed per-channel weights
``[1.2, -1.0] pA`` and records ``V_m``. The Poisson drive is PRNG-divergent, so
parity is distributional (category D): the seed-mean firing rate of the target
neuron must match live NEST within 5 %.
"""
import unittest

import brainstate
import jax
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest_validation.tolerance_conventions import CAT_D

SIMTIME = 1000.0
SEEDS = (0, 1, 2, 3)


def _nest_rate(seed, simtime):
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.rng_seed = seed + 1                    # offset to decorrelate from JAX
    neuron = nest.Create("iaf_psc_alpha")
    noise = nest.Create("poisson_generator", 2)
    noise[0].rate = 80000.0
    noise[1].rate = 15000.0
    sr = nest.Create("spike_recorder")
    nest.Connect(noise, neuron, syn_spec={"weight": [[1.2, -1.0]], "delay": 1.0})
    nest.Connect(neuron, sr)
    nest.Simulate(simtime)
    return sr.n_events * 1000.0 / simtime


@requires_nest
class TestOneNeuronWithNoiseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_rate_matches_nest_distributional(self):
        from examples.nest_like.one_neuron_with_noise import build

        def bp_rate(seed):
            sim, _vm, sr, _neuron, _t = build(seed=seed, simtime=SIMTIME)
            return sim.simulate(SIMTIME * u.ms).rate(sr)

        bp = [bp_rate(s) for s in SEEDS]
        ns = [_nest_rate(s, SIMTIME) for s in SEEDS]
        self.assertGreater(sum(ns) / len(ns), 0.0)   # the drive must make it fire
        compare_distributional(ns, bp, tol=CAT_D, metric="one_neuron_with_noise rate").assert_()


if __name__ == "__main__":
    unittest.main()
