# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/balancedneuron.py``.

NEST's ``balancedneuron`` bisects the inhibitory Poisson rate so an
``iaf_psc_alpha`` driven by signed-weight excitatory/inhibitory Poisson channels
fires at the excitatory rate ``r_ex = 5 Hz``. This exercises the rebuild-per-trial
sweep (re-``simulate`` per bisection step) and per-generator weight vectors
together. The objective is steep near the root, so the bisected inhibitory rate is
well-determined; we compare brainpy.state's root against live NEST's within a
documented bound. To bound cost the parity test uses a shorter horizon and a
modest tolerance than the demo's NEST-faithful 25 s.
"""
import unittest

import brainstate
import jax
from scipy.optimize import bisect

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainunit as u

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import requires_nest

SIMTIME = 4000.0       # 4 s: stable enough given the steep objective near the root
LOWER, UPPER = 18.0, 24.0
XTOL = 0.5
ROOT_BOUND_HZ = 1.5    # documented parity bound on the bisected inhibitory rate


def _nest_output_rate(r_in, simtime, seed):
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.resolution = 0.1
    nest.rng_seed = seed + 101
    neuron = nest.Create("iaf_psc_alpha")
    noise = nest.Create("poisson_generator", 2)
    noise[0].rate = 16000 * 5.0
    noise[1].rate = 4000 * abs(r_in)
    sr = nest.Create("spike_recorder")
    nest.Connect(noise, neuron, syn_spec={"weight": [[45.0, -45.0]], "delay": 1.0})
    nest.Connect(neuron, sr)
    nest.Simulate(simtime)
    return sr.n_events * 1000.0 / simtime


@requires_nest
class TestBalancedNeuronParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_bisected_inhibitory_rate_matches_nest(self):
        from examples.nest_like.balancedneuron import find_inhibitory_rate

        root_bp = find_inhibitory_rate(simtime=SIMTIME, lower=LOWER, upper=UPPER,
                                       prec=XTOL, seed=0)
        root_ns = bisect(lambda x: _nest_output_rate(x, SIMTIME, 0) - 5.0,
                         LOWER, UPPER, xtol=XTOL)
        self.assertTrue(LOWER <= root_bp <= UPPER, f"brainpy root {root_bp} out of bracket")
        self.assertLessEqual(
            abs(root_bp - root_ns), ROOT_BOUND_HZ,
            f"bisected inhibitory rate brainpy={root_bp:.3f} NEST={root_ns:.3f} "
            f"(bound {ROOT_BOUND_HZ} Hz)")


if __name__ == "__main__":
    unittest.main()
