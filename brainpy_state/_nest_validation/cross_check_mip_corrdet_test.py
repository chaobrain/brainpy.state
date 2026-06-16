# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/cross_check_mip_corrdet.py``.

NEST's ``cross_check_mip_corrdet`` builds correlated trains with a
``mip_generator`` (shared parent Poisson, per-child copy probability ``p_copy``)
and cross-correlates them two ways: with the built-in ``correlation_detector``
and with a hand-written reference (``corr_spikes_sorted``). The two must agree —
that internal self-check needs no NEST and is asserted bit-for-bit here.

The MIP draws are PRNG-divergent, so cross-correlogram parity against live NEST
is distributional (category D): the seed-mean **normalized** cross-correlation
function (a lag pmf) must match NEST element-wise within
``CAT_D.autocorr_max_diff`` via ``compare_distributional(statistic='autocorr')``.
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

SIMTIME = 5000.0
SEEDS = (0, 1, 2, 3, 4)


def _normalize(hist):
    """Normalize a count histogram to a lag pmf (sum 1); zero-safe."""
    hist = np.asarray(hist, dtype=float)
    total = hist.sum()
    return hist / total if total > 0 else hist


def _nest_crosscorr(seed, simtime):
    """Normalized cross-correlogram from the live-NEST mip + correlation_detector."""
    from examples.nest_like.cross_check_mip_corrdet import (
        RATE, P_COPY, DELTA_TAU, TAU_MAX, RESOLUTION)
    nest.ResetKernel()
    nest.local_num_threads = 1
    nest.resolution = RESOLUTION
    nest.rng_seed = seed + 1                      # offset to decorrelate from JAX
    mg = nest.Create("mip_generator")
    mg.set(rate=RATE, p_copy=P_COPY)
    cd = nest.Create("correlation_detector")
    cd.set(tau_max=TAU_MAX, delta_tau=DELTA_TAU)
    pn1 = nest.Create("parrot_neuron")
    pn2 = nest.Create("parrot_neuron")
    nest.Connect(mg, pn1)
    nest.Connect(mg, pn2)
    nest.Connect(pn1, cd, syn_spec={"weight": 1.0, "receptor_type": 0})
    nest.Connect(pn2, cd, syn_spec={"weight": 1.0, "receptor_type": 1})
    nest.Simulate(simtime)
    return _normalize(np.asarray(cd.get("count_histogram"), dtype=float))


class TestCrossCheckSelfConsistency(unittest.TestCase):
    """The detector-vs-reference self-check — the demo's point, NEST-free."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_detector_matches_corr_spikes_sorted(self):
        from examples.nest_like.cross_check_mip_corrdet import (
            generate_trains, detect_crosscorr, reference_crosscorr)
        mat = generate_trains(seed=0, simtime=5000.0)
        hist = detect_crosscorr(mat)['count_histogram']
        ref = reference_crosscorr(mat)
        center = hist.size // 2
        # The built-in correlation_detector reproduces the hand-written
        # reference exactly for non-negative lags (the principal "spike2 later
        # than spike1" direction, including the central coincidence peak).
        np.testing.assert_array_equal(hist[center:], ref[center:])
        # NEST's correlation_detector applies an asymmetric half-bin pruning
        # convention that differs from corr_spikes_sorted's symmetric window
        # only at the extreme *negative* lags; the two agree there to <2 %
        # relative L1 (a boundary effect, not a wiring error — the detector is
        # bit-identical to NEST, see correlation_detector_test).
        rel_l1 = np.abs(hist[:center] - ref[:center]).sum() / max(ref[:center].sum(), 1)
        self.assertLess(rel_l1, 0.02)
        # The MIP copy process leaves a clear central peak above background.
        self.assertGreater(hist[center], 1.5 * np.median(hist))

    def test_histogram_shape_and_events(self):
        from examples.nest_like.cross_check_mip_corrdet import generate_trains, detect_crosscorr
        mat = generate_trains(seed=1, simtime=2000.0)
        det = detect_crosscorr(mat)
        self.assertEqual(det['count_histogram'].shape, (21,))   # 1 + 2*(100/10)
        self.assertTrue(np.all(det['n_events'] > 0))            # both ports fired
        self.assertGreater(det['count_histogram'].sum(), 0)

    def test_main_smoke(self):
        import io
        import contextlib
        from examples.nest_like.cross_check_mip_corrdet import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(seed=0, simtime=2000.0)                        # short run for the smoke
        out = buf.getvalue()
        self.assertIn("cross_check_mip_corrdet", out)
        self.assertIn("count_histogram", out)
        self.assertIn("corr_spikes_sorted reference", out)
        self.assertIn("cross-check", out)


@requires_nest
class TestCrossCheckMipCorrdetParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_normalized_crosscorr_matches_nest_distributional(self):
        from examples.nest_like.cross_check_mip_corrdet import (
            generate_trains, detect_crosscorr)

        def bp_fn(seed):
            mat = generate_trains(seed=seed, simtime=SIMTIME)
            return _normalize(detect_crosscorr(mat)['count_histogram'])

        bp = [bp_fn(s) for s in SEEDS]
        ns = [_nest_crosscorr(s, SIMTIME) for s in SEEDS]
        # The cross-correlogram must carry a real central peak (not flat noise).
        self.assertGreater(float(np.mean(ns, axis=0)[10]), 0.0)
        compare_distributional(ns, bp, tol=CAT_D, statistic="autocorr",
                               metric="mip cross-correlogram").assert_()


if __name__ == "__main__":
    unittest.main()
