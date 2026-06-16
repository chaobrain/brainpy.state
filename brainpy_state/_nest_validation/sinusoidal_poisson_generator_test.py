# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest/sinusoidal_poisson_generator.py``.

NEST's §3.7 ``sinusoidal_poisson_generator`` demo drives parrot relays with an
inhomogeneous Poisson train of rate ``λ(t) = max(0, dc + ac·sin(2πf t + φ))``. The
drive is PRNG-divergent, so parity is **distributional** (category D): the
seed-averaged per-bin population spike-count **autocorrelation** — which carries
the modulation period — must match live NEST element-wise within
``CAT_D.autocorr_max_diff``. Structural facts (PSTH tracks ``λ(t)``, the
individual/shared noise modes, rate clamping when ``ac > dc``, ``dt`` invariance)
are checked NEST-free.
"""
import gc
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

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest_validation.tolerance_conventions import CAT_D

SEEDS = (0, 1, 2, 3, 4)


def _nest_pop_autocorr(seed, simtime):
    """Per-bin population spike-count autocorrelation from NEST (same recipe as bp)."""
    from examples.nest.sinusoidal_poisson_generator import (
        RATE, AMPLITUDE, FREQUENCY, PHASE, N_TARGETS, PST_BIN, DT, MAX_LAG,
    )
    nest.ResetKernel()
    nest.resolution = DT
    nest.rng_seed = seed + 1                       # offset to decorrelate from JAX
    g = nest.Create("sinusoidal_poisson_generator", 1, {
        "rate": RATE, "amplitude": AMPLITUDE, "frequency": FREQUENCY,
        "phase": PHASE, "individual_spike_trains": True})
    p = nest.Create("parrot_neuron", N_TARGETS)
    sr = nest.Create("spike_recorder")
    nest.Connect(g, p, "all_to_all")
    nest.Connect(p, sr)
    nest.Simulate(simtime)
    times = np.asarray(sr.get("events")["times"], dtype=float)
    counts = np.histogram(times, bins=int(round(simtime / PST_BIN)),
                          range=(0.0, simtime))[0].astype(float)
    c = counts - counts.mean()
    var = float(np.dot(c, c))
    n = c.shape[0]
    if var <= 0.0:
        return np.zeros(MAX_LAG + 1)
    return np.array([float(np.dot(c[:n - lag], c[lag:]))
                     for lag in range(MAX_LAG + 1)]) / var


class TestSinusoidalPoissonStructural(unittest.TestCase):
    """Sinusoidal-Poisson invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def tearDown(self):
        # Each run_spikes builds a fresh generator inside a new environ.context, so
        # every call is a fresh JAX trace+compile whose artifacts otherwise accumulate
        # across tests in one process and slow JAX cache lookups. Bound it per test.
        jax.clear_caches()
        gc.collect()

    def test_psth_tracks_lambda(self):
        from examples.nest.sinusoidal_poisson_generator import (
            run_spikes, population_psth, lam_of_t)
        spk = run_spikes(seed=0, individual=True)
        centers, psth = population_psth(spk)
        lam = lam_of_t(centers)
        corr = np.corrcoef(psth, lam)[0, 1]
        self.assertGreater(corr, 0.85)                 # PSTH follows λ(t)
        self.assertGreater(psth.max() / max(psth.min(), 1e-9), 1.5)   # modulation visible

    def test_individual_vs_shared_modes(self):
        from examples.nest.sinusoidal_poisson_generator import run_spikes
        indiv = run_spikes(seed=0, simtime=200.0, individual=True)
        shared = run_spikes(seed=0, simtime=200.0, individual=False)
        # individual: the N target columns are not all identical (independent trains)
        self.assertFalse(bool(np.all(indiv == indiv[:, :1])))
        # shared: every target sees the same relayed train (perfect synchrony)
        self.assertTrue(bool(np.all(shared == shared[:, :1])))
        self.assertGreater(int(shared.sum()), 0)

    def test_ac_zero_is_stationary(self):
        from examples.nest.sinusoidal_poisson_generator import (
            run_spikes, population_psth, RATE, FREQUENCY)
        spk = run_spikes(seed=0, simtime=500.0, individual=True, amplitude=0.0)
        centers, psth = population_psth(spk)
        # No modulation: mean ≈ dc rate, and (unlike the ac>0 case) the PSTH is
        # not phase-locked to f — its correlation with the drive sinusoid is ~0.
        self.assertAlmostEqual(psth.mean(), RATE, delta=0.1 * RATE)
        sin_wave = np.sin(2.0 * np.pi * FREQUENCY * centers / 1000.0)
        self.assertLess(abs(np.corrcoef(psth, sin_wave)[0, 1]), 0.4)

    def test_ac_greater_than_dc_clamps_rate(self):
        from examples.nest.sinusoidal_poisson_generator import (
            run_spikes, population_psth, RATE)
        # ac > dc: λ would go negative -> clamped at 0, so troughs are near-empty.
        spk = run_spikes(seed=0, simtime=500.0, individual=True, amplitude=2.0 * RATE)
        _, psth = population_psth(spk)
        self.assertTrue(np.all(psth >= 0.0))           # counts never negative
        self.assertLess(psth.min(), 0.15 * RATE)       # clamped trough ≈ 0
        self.assertGreater(psth.max(), 1.5 * RATE)     # peak well above dc

    def test_autocorr_carries_modulation_period(self):
        from examples.nest.sinusoidal_poisson_generator import (
            run_spikes, spike_count_autocorr, FREQUENCY, PST_BIN)
        acf = spike_count_autocorr(run_spikes(seed=0, individual=True))
        self.assertAlmostEqual(acf[0], 1.0, places=9)  # lag-0 normalized
        period_bins = int(round((1000.0 / FREQUENCY) / PST_BIN))   # 10 bins
        self.assertLess(acf[period_bins // 2], -0.3)   # anti-phase trough
        self.assertGreater(acf[period_bins], 0.3)      # one-period echo peak

    def test_dt_invariance(self):
        from examples.nest.sinusoidal_poisson_generator import run_spikes, population_psth
        m1 = population_psth(run_spikes(seed=0, dt=0.1, simtime=500.0))[1]
        m2 = population_psth(run_spikes(seed=0, dt=0.05, simtime=500.0), dt=0.05)[1]
        self.assertAlmostEqual(m1.mean(), m2.mean(), delta=0.08 * m1.mean())

    def test_update_lowers_under_for_loop(self):
        # cluster-12 discipline: the generator's update() must lower into a single
        # compiled for_loop (traced once), not run op-by-op. Drive a multi-channel
        # instance directly — this also exercises the in_size=N individual path the
        # Simulator fan-out demo does not use.
        import brainstate.transform as transform
        from brainpy_state import sinusoidal_poisson_generator
        from examples.nest.sinusoidal_poisson_generator import RATE, AMPLITUDE, FREQUENCY
        n_targets, n_steps, dt = 8, 200, 0.1
        gen = sinusoidal_poisson_generator(
            in_size=n_targets, rate=RATE * u.Hz, amplitude=AMPLITUDE * u.Hz,
            frequency=FREQUENCY * u.Hz, individual_spike_trains=True, rng_seed=0)
        brainstate.nn.init_all_states(gen)
        with brainstate.environ.context(dt=dt * u.ms):
            times = u.math.arange(0.0 * u.ms, n_steps * dt * u.ms, dt * u.ms)
            idx = u.math.arange(times.size)

            def step(t, i):
                with brainstate.environ.context(t=t, i=i):
                    return gen.update()

            out = np.asarray(transform.for_loop(step, times, idx))
        self.assertEqual(out.shape, (n_steps, n_targets))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(int(out.sum()), 0)          # the drive actually fires
        self.assertFalse(bool(np.all(out == out[:, :1])))  # independent columns

    def test_main_smoke(self):
        import io
        import contextlib
        from examples.nest.sinusoidal_poisson_generator import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main()
        out = buf.getvalue()
        self.assertIn("sinusoidal_poisson_generator", out)
        self.assertIn("PSTH-vs-λ corr", out)
        self.assertIn("autocorr", out)
        self.assertIn("columns identical", out)        # the shared-mode line


@requires_nest
class TestSinusoidalPoissonParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def tearDown(self):
        # Each run_spikes builds a fresh generator inside a new environ.context, so
        # every call is a fresh JAX trace+compile whose artifacts otherwise accumulate
        # across tests in one process and slow JAX cache lookups. Bound it per test.
        jax.clear_caches()
        gc.collect()

    def test_psth_autocorr_matches_nest_distributional(self):
        from examples.nest.sinusoidal_poisson_generator import (
            run_spikes, spike_count_autocorr, SIMTIME)

        bp = [spike_count_autocorr(run_spikes(seed=s, individual=True)) for s in SEEDS]
        ns = [_nest_pop_autocorr(s, SIMTIME) for s in SEEDS]
        # Both must show real modulation structure (not a flat autocorr).
        self.assertLess(float(np.mean(np.stack(ns), axis=0).min()), -0.2)
        compare_distributional(
            ns, bp, tol=CAT_D, metric="sinusoidal_poisson pop-autocorr",
            statistic="autocorr").assert_()


if __name__ == "__main__":
    unittest.main()
