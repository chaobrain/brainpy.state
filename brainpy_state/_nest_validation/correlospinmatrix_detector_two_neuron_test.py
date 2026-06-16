# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/correlospinmatrix_detector_two_neuron.py``.

NEST's ``correlospinmatrix_detector_two_neuron`` (Ginzburg & Sompolinsky 1994,
Fig. 1) wires a stochastic ``ginzburg_neuron`` (n1) into a deterministic
``mcculloch_pitts_neuron`` (n2) with weight 1, taps both binary trains into a
two-channel ``correlospinmatrix_detector``, and reads the auto-/cross-covariance
functions plus the per-channel mean activities.

The brainpy port runs the two binary neurons in a JAX ``for_loop`` (n2 driven by
n1's one-step-delayed state, the telescoped analog of NEST's per-transition
delta delivery) and feeds the recorded transitions to the detector **eagerly**
(post-hoc) — the detector is an imperative host device. Because the two PRNG
streams diverge, parity is distributional (category D):

* per-channel **mean activities** match the live-NEST seed mean within
  ``rate_rtol`` (``compare_distributional(statistic="mean")``);
* the four **covariance functions** :math:`c_{11}, c_{12}, c_{21}, c_{22}`
  (1-D over lags) match the seed-averaged NEST functions element-wise within
  ``autocorr_max_diff`` (``compare_distributional(statistic="autocorr")``).

The NEST-free structural tests assert the eager driver's invariants directly:
a silent pair yields a zero covariance tensor, and a single square-wave channel
reconstructs its duty cycle and the analytic auto-covariance peak
:math:`m(1-m)`.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest_validation.tolerance_conventions import CAT_D

#: Reduced horizon (ms) — the upstream's 1e6 ms is unnecessary for seed-mean parity.
SIMTIME = 50000.0
SEEDS = (0, 1, 2, 3, 4)


def _nest_run(seed, simtime):
    """Build the live-NEST two-neuron network and return its ``count_covariance``."""
    from examples.nest_like.correlospinmatrix_detector_two_neuron import (
        M_X, TAU_M, RESOLUTION, TAU_MAX, DELTA_TAU, WEIGHT)
    nest.ResetKernel()
    nest.local_num_threads = 1
    nest.resolution = RESOLUTION
    nest.rng_seed = seed + 1                       # offset to decorrelate from JAX
    csd = nest.Create("correlospinmatrix_detector")
    csd.set(N_channels=2, tau_max=TAU_MAX, Tstart=TAU_MAX, delta_tau=DELTA_TAU)
    n1 = nest.Create("ginzburg_neuron")
    n1.set(theta=0.0, tau_m=TAU_M, c_1=0.0, c_2=2.0 * M_X, c_3=1.0)
    n2 = nest.Create("mcculloch_pitts_neuron")
    n2.set(theta=0.5, tau_m=TAU_M)
    nest.Connect(n1, n2, syn_spec={"weight": WEIGHT})
    nest.Connect(n1, csd, syn_spec={"receptor_type": 0})
    nest.Connect(n2, csd, syn_spec={"receptor_type": 1})
    nest.Simulate(simtime)
    return np.asarray(csd.get("count_covariance"), dtype=float)


class TestCorrelospinmatrixStructural(unittest.TestCase):
    """Eager-driver invariants — the demo's reconstruction payload, NEST-free."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_silent_pair_gives_zero_covariance(self):
        from examples.nest_like.correlospinmatrix_detector_two_neuron import (
            run_correlospinmatrix, mean_activities, covariance_matrix, RESOLUTION)
        n = 2000
        y = np.zeros(n)
        cc = run_correlospinmatrix(y, y, dt=RESOLUTION)
        self.assertEqual(cc.shape, (2, 2, 2001))           # 1 + 2*(100/0.1)
        np.testing.assert_array_equal(cc, 0)
        ma = mean_activities(cc, dt=RESOLUTION, simtime=n * RESOLUTION)
        np.testing.assert_array_equal(ma, np.zeros(2))
        cov = covariance_matrix(cc, ma, dt=RESOLUTION, simtime=n * RESOLUTION)
        np.testing.assert_array_equal(cov, 0)

    def test_square_wave_reconstructs_duty_and_autocovariance(self):
        from examples.nest_like.correlospinmatrix_detector_two_neuron import (
            run_correlospinmatrix, mean_activities, covariance_matrix, RESOLUTION)
        # ch0: 100 periods of [60 up, 40 down] -> duty 0.6; ch1 silent.
        up, down, k = 60, 40, 100
        period = np.concatenate([np.ones(up), np.zeros(down)])
        y1 = np.tile(period, k)
        y2 = np.zeros_like(y1)
        simtime = y1.size * RESOLUTION
        cc = run_correlospinmatrix(y1, y2, dt=RESOLUTION)
        ma = mean_activities(cc, dt=RESOLUTION, simtime=simtime)
        # The detector finalizes an up-pulse only when its down is confirmed by a
        # following event, so the last (unconfirmed) period is dropped: ~ (k-1)/k.
        self.assertAlmostEqual(ma[0], 0.6, delta=0.02)
        self.assertEqual(ma[1], 0.0)
        cov = covariance_matrix(cc, ma, dt=RESOLUTION, simtime=simtime)
        center = cc.shape[-1] // 2
        # Binary auto-covariance peaks at zero lag with height m*(1-m).
        self.assertAlmostEqual(cov[0, 0][center], 0.6 * 0.4, delta=0.02)
        self.assertEqual(cov[0, 0][center], cov[0, 0].max())
        # Symmetric about zero lag; silent channel + cross terms vanish.
        np.testing.assert_allclose(cov[0, 0], cov[0, 0][::-1], atol=1e-9)
        np.testing.assert_array_equal(cov[1, 1], 0.0)
        np.testing.assert_array_equal(cov[0, 1], 0.0)

    def test_simulate_pair_shapes_and_binary(self):
        from examples.nest_like.correlospinmatrix_detector_two_neuron import simulate_pair, RESOLUTION
        y1, y2 = simulate_pair(seed=0, simtime=2000.0)
        n = int(round(2000.0 / RESOLUTION))
        self.assertEqual(y1.shape, (n,))
        self.assertEqual(y2.shape, (n,))
        self.assertTrue(set(np.unique(y1)).issubset({0.0, 1.0}))
        self.assertTrue(set(np.unique(y2)).issubset({0.0, 1.0}))
        # n1 is autonomous with gain 0.5 -> roughly balanced occupancy.
        self.assertGreater(y1.mean(), 0.3)
        self.assertLess(y1.mean(), 0.7)

    def test_normalization_requires_simtime(self):
        from examples.nest_like.correlospinmatrix_detector_two_neuron import (
            mean_activities, covariance_matrix)
        cc = np.zeros((2, 2, 2001))
        with self.assertRaises(ValueError):
            mean_activities(cc, dt=0.1)
        with self.assertRaises(ValueError):
            covariance_matrix(cc, np.zeros(2), dt=0.1)

    def test_main_smoke(self):
        import io
        import contextlib
        from examples.nest_like.correlospinmatrix_detector_two_neuron import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(seed=1, simtime=2000.0)
        out = buf.getvalue()
        self.assertIn("mean activities", out)
        self.assertIn("c12", out)


@requires_nest
class TestCorrelospinmatrixParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _bp_count_covariances(self):
        from examples.nest_like.correlospinmatrix_detector_two_neuron import (
            simulate_pair, run_correlospinmatrix, RESOLUTION)
        out = []
        for s in SEEDS:
            y1, y2 = simulate_pair(seed=s, simtime=SIMTIME)
            out.append(run_correlospinmatrix(y1, y2, dt=RESOLUTION))
        return out

    def test_mean_activities_match_nest(self):
        from examples.nest_like.correlospinmatrix_detector_two_neuron import (
            mean_activities, RESOLUTION)
        bp_cc = self._bp_count_covariances()
        ne_cc = [_nest_run(s, SIMTIME) for s in SEEDS]
        bp_ma = np.array([mean_activities(cc, dt=RESOLUTION, simtime=SIMTIME) for cc in bp_cc])
        ne_ma = np.array([mean_activities(cc, dt=RESOLUTION, simtime=SIMTIME) for cc in ne_cc])
        # Both channels live near 0.5; compare each channel's seed mean.
        self.assertGreater(float(ne_ma.mean(axis=0)[1]), 0.1)   # n2 must actually fire
        for ch, label in ((0, "n1 mean activity"), (1, "n2 mean activity")):
            compare_distributional(ne_ma[:, ch], bp_ma[:, ch], tol=CAT_D,
                                   statistic="mean", metric=label).assert_()

    def test_covariance_functions_match_nest(self):
        from examples.nest_like.correlospinmatrix_detector_two_neuron import (
            mean_activities, covariance_matrix, RESOLUTION)
        bp_cc = self._bp_count_covariances()
        ne_cc = [_nest_run(s, SIMTIME) for s in SEEDS]

        def funcs(cc_list, i, j):
            out = []
            for cc in cc_list:
                ma = mean_activities(cc, dt=RESOLUTION, simtime=SIMTIME)
                out.append(covariance_matrix(cc, ma, dt=RESOLUTION, simtime=SIMTIME)[i, j])
            return out

        for (i, j, label) in ((0, 0, "c11"), (0, 1, "c12"), (1, 0, "c21"), (1, 1, "c22")):
            compare_distributional(funcs(ne_cc, i, j), funcs(bp_cc, i, j), tol=CAT_D,
                                   statistic="autocorr",
                                   metric=f"covariance {label}").assert_()


if __name__ == "__main__":
    unittest.main()
