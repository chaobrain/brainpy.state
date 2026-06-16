# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Analytic cross-check for the Campbell & Siegert demo (§3.5) — a documented carve-out.

The demo ``examples/nest_like/CampbellSiegert.py`` drives an ``iaf_psc_alpha`` population
with Poisson input and compares the empirical free-membrane statistics and firing
rate against two analytic predictions: Campbell's theorem (mean + variance of the
free ``V_m``) and Siegert's stationary-rate approximation.

**Why this is a carve-out, not a live-NEST trace parity.** The drive is a Poisson
PRNG stream that diverges between NEST and JAX, so a per-sample ``V_m`` comparison
against NEST is meaningless — both simulators only agree *distributionally*. The
ground truth here is the *theory itself*: this test asserts that the Simulator's
empirical statistics reproduce the Campbell/Siegert formulae. (Live NEST is known to
match the same formulae — that is precisely what NEST's published example
demonstrates — so re-running NEST would only re-confirm the theory at the cost of a
PRNG-divergent, equally-approximate estimate.)

Tolerances reflect each estimator's nature:

* **Campbell mean** — converges to the analytic ``mu`` to well under 0.5 mV; this is
  the strong cross-check that the Poisson drive and the PSP/PSC conversion are right.
* **Campbell variance** — a noisier second moment, matched within ~30 %.
* **Siegert rate** — Siegert is an *approximation* (a systematic few-×10 % bias) and
  the spike count is low, so the seed-averaged rate is matched within ~35 %.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import examples.nest_like.CampbellSiegert as demo

# Shorter than the demo's 20 s (enough for the mean to converge well) and averaged
# over several seeds so the low-count rate estimate is not flaky.
_SIMTIME = 8000.0
_SEEDS = (0, 1, 2, 3)


class TestCampbellSiegertCrossCheck(unittest.TestCase):
    """The Simulator reproduces the Campbell/Siegert analytic predictions."""

    @classmethod
    def setUpClass(cls):
        brainstate.environ.set(dt=demo.DT * u.ms)
        cls.results = [demo.run_analysis(simtime=_SIMTIME, seed=s) for s in _SEEDS]

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_campbell_mean_matches_theory(self):
        # Campbell's theorem for the mean free V_m: the strongest cross-check.
        r0 = self.results[0]
        self.assertAlmostEqual(r0['mean_act'], r0['mu_mV'], delta=0.5,
                               msg=f"mean V_m actual={r0['mean_act']:.4f} "
                                   f"calc={r0['mu_mV']:.4f} mV")

    def test_campbell_variance_matches_theory(self):
        # Variance is a noisier second moment; allow ~30%.
        r0 = self.results[0]
        rel = abs(r0['var_act'] - r0['var_mV2']) / r0['var_mV2']
        self.assertLess(rel, 0.30,
                        f"var actual={r0['var_act']:.4f} calc={r0['var_mV2']:.4f} "
                        f"mV^2 (rel={rel:.3f})")

    def test_siegert_rate_matches_theory(self):
        # Siegert is approximate + the spike count is low; compare the seed-mean
        # rate within ~35% of the analytic value.
        rate_calc = self.results[0]['rate_hz']
        rate_act = float(np.mean([r['rate_act'] for r in self.results]))
        rel = abs(rate_act - rate_calc) / rate_calc
        self.assertLess(rel, 0.35,
                        f"seed-mean rate actual={rate_act:.4f} calc={rate_calc:.4f} "
                        f"Hz (rel={rel:.3f}, seeds={_SEEDS})")
        self.assertGreater(rate_act, 0.0, "neurons should fire under the Poisson drive")

    def test_two_source_variant_matches_single_source(self):
        # The demo's documented invariant: splitting one source into two at half the
        # rate each leaves Campbell's mean/variance unchanged (sources add).
        J1, mu1, s2_1, _r1 = demo.analytic(weights=[0.1], rates=[10000.0])
        J2, mu2, s2_2, _r2 = demo.analytic(weights=[0.1, 0.1], rates=[5000.0, 5000.0])
        self.assertAlmostEqual(mu1, mu2, places=9)
        self.assertAlmostEqual(s2_1, s2_2, places=12)


if __name__ == "__main__":
    unittest.main()
