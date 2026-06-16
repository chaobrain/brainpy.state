# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/if_curve.py``.

NEST's ``if_curve`` measures the transfer function of an ``aeif_cond_exp``
population driven by a white-noise current ``I_mean + I_std * W(t)`` from a
``noise_generator`` (a current-injecting device). This validates Extension B
(current injection via the neuron's ring buffer) and the rebuild-per-trial sweep.

* The ``std = 0`` point is a deterministic constant current: the population rate
  must match live NEST within 5 % (category C, mean-field rate).
* The noisy points are PRNG-divergent: the seed-mean population rate must match
  live NEST within 5 % (category D). A coarse grid bounds compile cost.
"""
import unittest

import brainstate
import jax

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import (
    requires_nest, compare_trace, compare_distributional)
from brainpy_state._nest_validation.tolerance_conventions import CAT_C_RATE, CAT_D

N_NEURONS = 100
SIMTIME = 1000.0
SEEDS = (0, 1, 2, 3)
NOISE_POINTS = ((800.0, 100.0), (900.0, 200.0))
DET_POINT = (700.0, 0.0)

_NEST_PARAMS = {"a": 4.0, "b": 80.8, "V_th": -50.4, "Delta_T": 2.0, "I_e": 0.0,
                "C_m": 281.0, "g_L": 30.0, "V_reset": -70.6, "tau_w": 144.0,
                "t_ref": 5.0, "V_peak": -40.0, "E_L": -70.6, "E_ex": 0.0, "E_in": -70.0}


def _nest_rate(mean, std, seed, n=N_NEURONS, simtime=SIMTIME):
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.resolution = 0.1
    nest.rng_seed = seed + 101
    neu = nest.Create("aeif_cond_exp", n, params=_NEST_PARAMS)
    ng = nest.Create("noise_generator",
                     params={"mean": mean, "std": std, "start": 0.0,
                             "stop": simtime, "origin": 0.0})
    sr = nest.Create("spike_recorder")
    nest.Connect(ng, neu, "all_to_all")
    nest.Connect(neu, sr, "all_to_all")
    nest.Simulate(simtime)
    return sr.n_events * 1000.0 / (n * simtime)


@requires_nest
class TestIfCurveParity(unittest.TestCase):
    def test_deterministic_point_matches_nest(self):
        from examples.nest_like.if_curve import output_rate
        mean, std = DET_POINT
        bp = output_rate(mean, std, n_neurons=N_NEURONS, seed=0, simtime=SIMTIME)
        ns = _nest_rate(mean, std, seed=0)
        self.assertGreater(ns, 0.0)
        compare_trace(ns, bp, tol=CAT_C_RATE,
                      metric=f"if_curve rate mean={mean} std={std}").assert_()

    def test_noisy_points_match_nest_distributional(self):
        from examples.nest_like.if_curve import output_rate
        for mean, std in NOISE_POINTS:
            with self.subTest(mean=mean, std=std):
                bp = [output_rate(mean, std, n_neurons=N_NEURONS, seed=s,
                                  simtime=SIMTIME) for s in SEEDS]
                ns = [_nest_rate(mean, std, seed=s) for s in SEEDS]
                self.assertGreater(sum(ns) / len(ns), 0.0)
                compare_distributional(
                    ns, bp, tol=CAT_D,
                    metric=f"if_curve rate mean={mean} std={std}").assert_()


if __name__ == "__main__":
    unittest.main()
