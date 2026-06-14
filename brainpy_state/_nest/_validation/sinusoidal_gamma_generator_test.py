# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest/sinusoidal_gamma_generator.py``.

NEST's §3.7 ``sinusoidal_gamma_generator`` demo draws spikes from a gamma renewal
process of order ``m`` whose rate follows ``λ(t) = max(0, dc + ac·sin(2πf t + φ))``.
The headline is the gamma-regularization law: the ISI coefficient of variation
``CV → 1/√m``. Parity is **distributional** (category D): the seed-averaged ISI
CV must match live NEST within the gamma CV band, and (modulated case) the
population spike-count autocorrelation must match element-wise within
``CAT_D.autocorr_max_diff``. The qualitative ``CV ≈ 1/√m`` law, the rate-profile
preservation under modulation, binary emission, and the noise modes are checked
NEST-free.
"""
import gc
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest._validation.tolerance_conventions import CAT_D, DistributionalTolerance
from examples.nest.sinusoidal_gamma_generator import AMPLITUDE as AMPLITUDE_FIXTURE

SEEDS = (0, 1, 2, 3, 4)
#: Gamma order for the CV parity comparison.
CV_PARITY_ORDER = 6
#: Documented gamma CV band: the seed-mean CV is a stochastic estimate, so the
#: ``mean_diff_pct`` is widened from CAT_D's 2% to 6% (the qualitative 1/√m law is
#: asserted separately within ~12%). autocorr/rate fields inherit CAT_D.
_CV_BAND = DistributionalTolerance(rate_rtol=CAT_D.rate_rtol, mean_diff_pct=6e-2,
                                   autocorr_max_diff=CAT_D.autocorr_max_diff,
                                   n_seeds=CAT_D.n_seeds)
#: Maximum autocorrelation lag (bins) for the modulated-rate parity.
MAX_LAG = 30


def _nest_events(seed, order, amplitude, simtime):
    """Spike (senders, times) from a NEST sinusoidal_gamma_generator → parrots."""
    from examples.nest.sinusoidal_gamma_generator import (
        RATE, FREQUENCY, PHASE, N_TARGETS, DT)
    nest.ResetKernel()
    nest.resolution = DT
    nest.rng_seed = seed + 1
    g = nest.Create("sinusoidal_gamma_generator", 1, {
        "rate": RATE, "amplitude": amplitude, "frequency": FREQUENCY,
        "phase": PHASE, "order": float(order), "individual_spike_trains": True})
    p = nest.Create("parrot_neuron", N_TARGETS)
    sr = nest.Create("spike_recorder")
    nest.Connect(g, p, "all_to_all")
    nest.Connect(p, sr)
    nest.Simulate(simtime)
    ev = sr.get("events")
    return np.asarray(ev["senders"]), np.asarray(ev["times"], dtype=float)


def _nest_cv(seed, order, simtime):
    """Pooled per-sender ISI CV of the stationary NEST gamma train."""
    senders, times = _nest_events(seed, order, 0.0, simtime)
    isis = []
    for s in np.unique(senders):
        st = np.sort(times[senders == s])
        if st.size > 1:
            isis.append(np.diff(st))
    isis = np.concatenate(isis) if isis else np.array([])
    return float(np.std(isis) / np.mean(isis)) if isis.size else float("nan")


def _nest_pop_autocorr(seed, amplitude, order, simtime):
    """Per-bin population spike-count autocorrelation from NEST (modulated case)."""
    from examples.nest.sinusoidal_gamma_generator import PST_BIN
    _senders, times = _nest_events(seed, order, amplitude, simtime)
    counts = np.histogram(times, bins=int(round(simtime / PST_BIN)),
                          range=(0.0, simtime))[0].astype(float)
    c = counts - counts.mean()
    var = float(np.dot(c, c))
    n = c.shape[0]
    if var <= 0.0:
        return np.zeros(MAX_LAG + 1)
    return np.array([float(np.dot(c[:n - lag], c[lag:]))
                     for lag in range(MAX_LAG + 1)]) / var


class TestSinusoidalGammaStructural(unittest.TestCase):
    """Gamma-regularization invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def tearDown(self):
        # Each run_spikes builds a fresh generator inside a new environ.context, so
        # every call is a fresh JAX trace+compile. The gamma update() lowers a
        # while_loop-in-scan (rejection sampling) that is costly to compile; left
        # uncleared, these artifacts accumulate across tests in one process until
        # JAX cache lookups degrade into an apparent hang. Bound it per test.
        jax.clear_caches()
        gc.collect()

    def test_cv_approaches_one_over_sqrt_m(self):
        from examples.nest.sinusoidal_gamma_generator import cv_by_order, ORDERS
        cvs = cv_by_order(seed=0)
        for m, cv in zip(ORDERS, cvs):
            self.assertAlmostEqual(cv, 1.0 / np.sqrt(m), delta=0.12)   # CV ≈ 1/√m
        # Higher order ⇒ more regular ⇒ strictly smaller CV.
        self.assertTrue(np.all(np.diff(cvs) < 0.0))

    def test_order_one_is_poisson(self):
        from examples.nest.sinusoidal_gamma_generator import run_spikes, pooled_isis, isi_cv
        cv = isi_cv(pooled_isis(run_spikes(seed=0, order=1, amplitude=0.0)))
        self.assertAlmostEqual(cv, 1.0, delta=0.1)        # m=1 ≡ Poisson, CV → 1

    def test_binary_emission(self):
        from examples.nest.sinusoidal_gamma_generator import run_spikes
        spk = run_spikes(seed=0, simtime=300.0, order=2, amplitude=AMPLITUDE_FIXTURE)
        self.assertLessEqual(int(spk.max()), 1)           # renewal: ≤ 1 spike/step

    def test_modulated_rate_tracks_lambda(self):
        from examples.nest.sinusoidal_gamma_generator import (
            run_spikes, population_psth, lam_of_t, MODULATED_ORDER)
        spk = run_spikes(seed=0, order=MODULATED_ORDER, amplitude=AMPLITUDE_FIXTURE)
        centers, psth = population_psth(spk)
        corr = np.corrcoef(psth, lam_of_t(centers))[0, 1]
        self.assertGreater(corr, 0.85)                    # rate profile preserved

    def test_individual_vs_shared_modes(self):
        from examples.nest.sinusoidal_gamma_generator import run_spikes
        indiv = run_spikes(seed=0, simtime=300.0, order=2, individual=True)
        shared = run_spikes(seed=0, simtime=300.0, order=2, individual=False)
        self.assertFalse(bool(np.all(indiv == indiv[:, :1])))
        self.assertTrue(bool(np.all(shared == shared[:, :1])))
        self.assertGreater(int(shared.sum()), 0)

    def test_update_lowers_under_for_loop(self):
        # cluster-12 discipline: the gamma update() must lower into a single
        # compiled for_loop (traced once). Drive a multi-channel instance directly.
        import brainstate.transform as transform
        from brainpy_state import sinusoidal_gamma_generator
        from examples.nest.sinusoidal_gamma_generator import RATE, FREQUENCY
        n_targets, n_steps, dt = 8, 300, 0.1
        gen = sinusoidal_gamma_generator(
            in_size=n_targets, rate=RATE * u.Hz, amplitude=0.0 * u.Hz,
            frequency=FREQUENCY * u.Hz, order=4.0, individual_spike_trains=True,
            rng_seed=0)
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
        self.assertLessEqual(int(out.max()), 1)           # binary
        self.assertGreater(int(out.sum()), 0)

    def test_main_smoke(self):
        import io
        import contextlib
        from examples.nest.sinusoidal_gamma_generator import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main()
        out = buf.getvalue()
        self.assertIn("sinusoidal_gamma_generator", out)
        self.assertIn("CV", out)
        self.assertIn("1/sqrt(m)", out)
        self.assertIn("modulated", out)


@requires_nest
class TestSinusoidalGammaParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def tearDown(self):
        # Each run_spikes builds a fresh generator inside a new environ.context, so
        # every call is a fresh JAX trace+compile. The gamma update() lowers a
        # while_loop-in-scan (rejection sampling) that is costly to compile; left
        # uncleared, these artifacts accumulate across tests in one process until
        # JAX cache lookups degrade into an apparent hang. Bound it per test.
        jax.clear_caches()
        gc.collect()

    def test_cv_matches_nest_distributional(self):
        from examples.nest.sinusoidal_gamma_generator import (
            run_spikes, pooled_isis, isi_cv, SIMTIME)

        def bp_cv(seed):
            return isi_cv(pooled_isis(
                run_spikes(seed=seed, order=CV_PARITY_ORDER, amplitude=0.0)))

        bp = [bp_cv(s) for s in SEEDS]
        ns = [_nest_cv(s, CV_PARITY_ORDER, SIMTIME) for s in SEEDS]
        # Both must land near the 1/√m law before we compare them to each other.
        target = 1.0 / np.sqrt(CV_PARITY_ORDER)
        self.assertAlmostEqual(float(np.mean(bp)), target, delta=0.12)
        self.assertAlmostEqual(float(np.mean(ns)), target, delta=0.12)
        compare_distributional(ns, bp, tol=_CV_BAND,
                               metric="sinusoidal_gamma ISI-CV", statistic="cv").assert_()

    def test_modulated_autocorr_matches_nest_distributional(self):
        from examples.nest.sinusoidal_gamma_generator import (
            run_spikes, population_psth, MODULATED_ORDER, SIMTIME, PST_BIN, DT)

        def bp_autocorr(seed):
            spk = run_spikes(seed=seed, order=MODULATED_ORDER, amplitude=AMPLITUDE_FIXTURE)
            counts = population_psth(spk)[1]  # Hz per bin; autocorr is scale-free
            c = counts - counts.mean()
            var = float(np.dot(c, c))
            n = c.shape[0]
            if var <= 0.0:
                return np.zeros(MAX_LAG + 1)
            return np.array([float(np.dot(c[:n - lag], c[lag:]))
                             for lag in range(MAX_LAG + 1)]) / var

        bp = [bp_autocorr(s) for s in SEEDS]
        ns = [_nest_pop_autocorr(s, AMPLITUDE_FIXTURE, MODULATED_ORDER, SIMTIME)
              for s in SEEDS]
        self.assertLess(float(np.mean(np.stack(ns), axis=0).min()), -0.2)
        compare_distributional(ns, bp, tol=CAT_D,
                               metric="sinusoidal_gamma modulated autocorr",
                               statistic="autocorr").assert_()


if __name__ == "__main__":
    unittest.main()
