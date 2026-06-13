# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Distributional parity for the gif_population demo (§3.5).

The demo ``examples/nest/gif_population.py`` runs a recurrently-connected
population of ``gif_psc_exp`` neurons driven by Poisson noise. Spike-frequency
adaptation makes the population oscillate on the adaptation time scale (Schwalger
et al. 2017).

**Why this is a carve-out, not a trace parity.** The GIF escape-rate spiking and
the Poisson drive are PRNG streams that diverge between NEST and JAX, so a
per-sample comparison is meaningless; both simulators agree only *distributionally*.
This is a category **D** comparison (seed-aggregated, never per-sample).

The recurrence uses ``fixed_indegree(K = round(p*N) = 30)`` — the Simulator's
fixed-mean-in-degree equivalent of NEST's ``pairwise_bernoulli(p=0.3)`` (same
expected in-degree). The ``@requires_nest`` reference wires NEST with the same
``fixed_indegree`` rule so the connectivity is identical on both sides.

The NEST-free tests pin the science (the adaptation oscillation appears as a
sub-zero dip in the binned-rate autocorrelation, and recurrence sharpens the
population fluctuations). The ``@requires_nest`` tests confirm the seed-averaged
population rate (rel ~1.4 %) and the binned-rate autocorrelation shape
(max|Δ| ~0.024) match live NEST.

This is also the regression guard for the ``gif_psc_exp`` ``I_stim`` for_loop fix:
running the population through the Simulator (one compiled ``for_loop``) exercises
the lowered path the model's own eager step-by-step tests never did.
"""
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest._validation.nest_compare import compare_distributional, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc

import examples.nest.gif_population as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

_N = demo.N_EX
_T = 2000.0          # parity horizon (matches the demo + NEST calibration)
_T_FAST = 1000.0     # shorter horizon for the NEST-free behavioural checks
_SEEDS = (0, 1, 2, 3)
_BIN_MS = 5.0
_MAX_LAG = 80        # 400 ms of autocorrelation lags

# Distributional tolerances (category D), calibrated against a live NEST run:
#   seed-mean rate : measured rel ~0.014 -> CAT_D (5 %)
#   autocorr shape : measured max|Δ| ~0.024 over 80 lags -> CAT_D (autocorr 0.05)


def _nest_run(seed):
    """Live-NEST gif_population run -> (rate_hz, binned-rate autocorrelation)."""
    nest.ResetKernel()
    nest.resolution = demo.DT
    nest.rng_seed = seed + 1
    nest.set_verbosity("M_ERROR")
    pop = nest.Create("gif_psc_exp", _N, params={
        "C_m": 83.1, "g_L": 3.7, "E_L": -67.0, "Delta_V": 1.4, "V_T_star": -39.6,
        "t_ref": 4.0, "V_reset": -36.7, "lambda_0": 1.0,
        "q_stc": [56.7, -6.9], "tau_stc": [57.8, 218.2],
        "q_sfa": [11.7, 1.8], "tau_sfa": [53.8, 640.0], "tau_syn_ex": 10.0})
    noise = nest.Create("poisson_generator", demo.N_NOISE, params={"rate": demo.RATE_NOISE})
    sr = nest.Create("spike_recorder")
    nest.Connect(pop, pop, {"rule": "fixed_indegree", "indegree": demo.K_EX},
                 syn_spec={"weight": demo.W_EX, "delay": demo.DELAY})
    nest.Connect(noise, pop, "all_to_all",
                 syn_spec={"weight": demo.W_NOISE, "delay": demo.DELAY})
    nest.Connect(pop, sr)
    nest.Simulate(_T)
    times = np.asarray(sr.events["times"], dtype=float)
    rate = times.size / (_N * _T / 1000.0)
    act = demo.activity_from_times(times, _N, _T, _BIN_MS)
    return rate, demo.autocorr(act, _MAX_LAG)


def _bp_run(seed):
    """brainpy gif_population run -> (rate_hz, binned-rate autocorrelation)."""
    r = demo.run_population(seed=seed, t_sim=_T, bin_ms=_BIN_MS)
    return r['rate_hz'], demo.autocorr(r['binned_rate'], _MAX_LAG)


def _cv(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    return float(x.std() / m) if m > 0 else 0.0


class TestGifPopulationBehaviour(unittest.TestCase):
    """NEST-free: the adaptation oscillation and the effect of recurrence."""

    @classmethod
    def setUpClass(cls):
        brainstate.environ.set(dt=demo.DT * u.ms)
        cls.rec = demo.run_population(seed=0, recurrent=True, t_sim=_T_FAST, bin_ms=_BIN_MS)
        cls.ctl = demo.run_population(seed=0, recurrent=False, t_sim=_T_FAST, bin_ms=_BIN_MS)

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_population_rate_is_in_a_sane_band(self):
        self.assertGreater(self.rec['rate_hz'], 3.0,
                           f"population should fire (rate={self.rec['rate_hz']:.2f})")
        self.assertLess(self.rec['rate_hz'], 40.0,
                        f"rate implausibly high ({self.rec['rate_hz']:.2f})")

    def test_adaptation_oscillation_present(self):
        # SFA makes a burst suppress the next -> the binned-rate autocorrelation dips
        # below zero at an intermediate lag (tens to a few hundred ms).
        ac = demo.autocorr(self.rec['binned_rate'], _MAX_LAG)
        dip_lag = int(np.argmin(ac[1:])) + 1
        self.assertLess(ac[1:].min(), -0.05,
                        f"adaptation should anti-correlate the rate (min={ac[1:].min():.3f})")
        self.assertTrue(4 <= dip_lag <= 70,
                        f"dip at {dip_lag * _BIN_MS:.0f} ms is not an adaptation-scale lag")

    def test_recurrence_sharpens_population_fluctuations(self):
        # Recurrent excitation synchronizes the population -> larger relative
        # fluctuation of the binned rate than the unconnected (Poisson-only) control.
        cv_rec = _cv(self.rec['binned_rate'])
        cv_ctl = _cv(self.ctl['binned_rate'])
        self.assertGreater(cv_rec, 1.3 * cv_ctl,
                           f"recurrence should raise synchrony (CV {cv_rec:.3f} vs "
                           f"control {cv_ctl:.3f})")

    def test_reproducible_given_seed(self):
        a = demo.run_population(seed=2, t_sim=_T_FAST, bin_ms=_BIN_MS)
        b = demo.run_population(seed=2, t_sim=_T_FAST, bin_ms=_BIN_MS)
        self.assertEqual(int(a['spk'].sum()), int(b['spk'].sum()),
                         "same seed must reproduce the spike count")


@requires_nest
class TestGifPopulationDistributionalParity(unittest.TestCase):
    """The seed-averaged rate and binned-rate autocorrelation match live NEST."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_NEST:
            return
        brainstate.environ.set(dt=demo.DT * u.ms)
        cls._nest = [_nest_run(s) for s in _SEEDS]
        cls._bp = [_bp_run(s) for s in _SEEDS]

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_seed_mean_rate_matches_nest(self):
        nrate = [r[0] for r in self._nest]
        brate = [r[0] for r in self._bp]
        compare_distributional(nrate, brate, tol=tc.CAT_D, metric="gif_population rate",
                               statistic="mean").assert_()

    def test_binned_rate_autocorrelation_matches_nest(self):
        # Seed-averaged autocorrelation functions, compared elementwise over lags.
        nac = [r[1] for r in self._nest]
        bac = [r[1] for r in self._bp]
        compare_distributional(nac, bac, tol=tc.CAT_D,
                               metric="gif_population binned-rate autocorr",
                               statistic="autocorr").assert_()


if __name__ == "__main__":
    unittest.main()
