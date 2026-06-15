# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest/recording_demo.py``.

NEST's ``recording_demo`` is a recording-API tour: a ``poisson_generator`` drives
an ``iaf_psc_exp`` whose spikes are captured by a ``spike_recorder``. The drive
is PRNG-divergent, so firing-rate parity is distributional (category D) — the
seed-mean rate must match live NEST within 5 %. The upstream's 1 MHz rate pins
the neuron to its refractory-saturated rate, so the match is in practice exact
(identical across seeds). Structural recording invariants — analog ``V_m`` trace
shape, a monotone time axis, ``n_events > 0`` — are checked NEST-free, and the
``read_spikes`` tour helper's two output forms (step indices vs ms) are verified
mutually consistent.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest._validation.tolerance_conventions import CAT_D

SIMTIME = 1000.0
SEEDS = (0, 1, 2, 3)


def _nest_rate(seed, simtime):
    """Firing rate (spks/s) of the NEST iaf_psc_exp under the same Poisson drive."""
    from examples.nest.recording_demo import RATE, WEIGHT, DELAY
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.rng_seed = seed + 1                     # offset to decorrelate from JAX
    n = nest.Create("iaf_psc_exp")
    pg = nest.Create("poisson_generator", 1, {"rate": RATE})
    sr = nest.Create("spike_recorder")
    nest.Connect(pg, n, syn_spec={"weight": WEIGHT, "delay": DELAY})
    nest.Connect(n, sr)
    nest.Simulate(simtime)
    return sr.n_events * 1000.0 / simtime


class TestRecordingDemoStructural(unittest.TestCase):
    """Recording-API invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_structural_recording_invariants(self):
        from examples.nest.recording_demo import build
        sim, sr, mm, _n, _t = build(seed=0, simtime=50.0)
        res = sim.simulate(50.0 * u.ms)
        nsteps = 500
        # Analog trace: (n_steps, 1), the V_m recordable is readable.
        self.assertEqual(res.trace(mm, 'V_m').shape, (nsteps, 1))
        # Time axis: shape (n_steps,), monotone increasing, dt spacing.
        t = np.asarray(u.get_mantissa(res.times / u.ms))
        self.assertEqual(t.shape, (nsteps,))
        self.assertTrue(np.all(np.diff(t) > 0))
        self.assertAlmostEqual(float(t[1] - t[0]), 0.1, places=9)
        # The drive must make the neuron fire -> spikes get recorded.
        self.assertGreater(res.n_events(sr), 0)

    def test_read_spikes_steps_vs_ms_consistent(self):
        from examples.nest.recording_demo import build, read_spikes
        sim, sr, _mm, _n, _t = build(seed=0, simtime=50.0)
        res = sim.simulate(50.0 * u.ms)
        steps = read_spikes(res, sr, time_in_steps=True)
        ms = read_spikes(res, sr, time_in_steps=False)
        self.assertEqual(len(steps), len(ms))
        self.assertGreater(len(steps), 0)
        # The ms form is exactly the step index times the resolution.
        np.testing.assert_allclose(ms, steps * 0.1, rtol=0, atol=1e-9)
        # Step indices are integer-valued and strictly increasing.
        self.assertTrue(np.all(steps == np.round(steps)))
        self.assertTrue(np.all(np.diff(steps) > 0))

    def test_main_smoke(self):
        import io
        import contextlib
        from examples.nest.recording_demo import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main()
        out = buf.getvalue()
        self.assertIn("recording_demo", out)
        self.assertIn("firing rate", out)
        self.assertIn("time_in_steps", out)        # the backend tour's two formats
        self.assertIn("V_m trace", out)            # analog recording alongside


@requires_nest
class TestRecordingDemoParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_rate_matches_nest_distributional(self):
        from examples.nest.recording_demo import build

        def bp_rate(seed):
            sim, sr, _mm, _n, _t = build(seed=seed, simtime=SIMTIME)
            return sim.simulate(SIMTIME * u.ms).rate(sr)

        bp = [bp_rate(s) for s in SEEDS]
        ns = [_nest_rate(s, SIMTIME) for s in SEEDS]
        self.assertGreater(sum(ns) / len(ns), 0.0)   # the drive must make it fire
        compare_distributional(ns, bp, tol=CAT_D, metric="recording_demo rate").assert_()


if __name__ == "__main__":
    unittest.main()
