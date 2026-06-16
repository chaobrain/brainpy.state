# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Mesoscopic-vs-microscopic parity for the gif_pop_psc_exp demo (§3.5).

The demo ``examples/nest_like/gif_pop_psc_exp.py`` simulates the same finite
two-population GIF network two ways: a **mesoscopic** population-rate model
(:class:`gif_pop_psc_exp`, host-side NumPy) and the **microscopic** network of
individual ``gif_psc_exp`` neurons (Simulator, one compiled ``for_loop``). The
demo's scientific claim (Schwalger et al. 2017; NEST figures 1 vs 2) is that the
two population activities ``A_N(t)`` agree *distributionally*.

**Why this is a carve-out, not a trace parity.** The mesoscopic finite-N binomial
spike count and the microscopic escape-rate spiking are independent PRNG streams;
they agree only in distribution (mean rate per population, step-evoked jump,
fluctuation autocorrelation), never sample-by-sample. This is a category **D**
comparison.

**Test economics.** The microscopic half is the only expensive piece (it lowers a
1000-neuron GIF network into one compiled ``for_loop``). The populations are large
(800 + 200), so a *single* run is already well-averaged; the NEST-free class
therefore does **one** micro run in ``setUpClass`` and compares it to the (cheap,
host-side) mesoscopic run. The ``@requires_nest`` guard lives entirely on the fast
host-side meso-vs-NEST path (no Simulator compile), so it can afford several seeds.

**What the @requires_nest guard covers.** :class:`gif_pop_psc_exp` is already
verified cell-by-cell against NEST in ``gif_pop_psc_exp_test.py``. The guard here
instead pins the *demo driver* (parameter mapping, ``A_N`` binning, step-current
injection via ``update(x=...)``) for a single **uncoupled** population, so a driver
regression cannot hide behind the coupling logic.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_validation.nest_compare import compare_distributional, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

import examples.nest_like.gif_pop_psc_exp as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

_SEED = 1
_MAX_LAG = 60        # 60 ms of autocorrelation lags (bins are 1 ms)

# Meso-vs-micro are two *different* solution methods of the same finite network,
# so the bar is looser than a NEST-vs-brainpy port. Calibrated against a single
# full-horizon run of each (meso vs micro): window-mean rates agree to ~11 %
# (ex_pre 5.43 vs 6.07, ih_pre 5.76 vs 6.15), both show the step jump
# (x3.36 vs x2.73), and both show slow positive activity autocorrelation
# (ac[60 ms] = 0.18 vs 0.30). NB the *dip* is method-specific: the cleaner meso
# rate model anti-correlates (min -0.06) but the noisier micro network does not
# (min +0.10), so the shared feature asserted is the slow correlation, not the dip.
_RATE_RTOL = 0.20
_MIN_STEP_JUMP = 2.0


def _exc_inh_window_rates(r):
    """(exc_pre, exc_post, inh_pre, inh_post) mean rates around the step."""
    t = r['t']
    a_ex, a_in = r['A_N'][:, 0], r['A_N'][:, 1]
    return (demo.window_rate(a_ex, t, 1000.0, demo.T_STEP),
            demo.window_rate(a_ex, t, demo.T_STEP, demo.T_END),
            demo.window_rate(a_in, t, 1000.0, demo.T_STEP),
            demo.window_rate(a_in, t, demo.T_STEP, demo.T_END))


class TestGifPopMesoVsMicro(unittest.TestCase):
    """NEST-free: the mesoscopic and microscopic population activities agree."""

    @classmethod
    def setUpClass(cls):
        brainstate.environ.set(dt=demo.DT * u.ms)
        # One cheap host-side meso run and one (expensive) micro for_loop run.
        cls.meso = demo.run_meso(seed=_SEED, coupled=True)
        cls.micro = demo.run_micro(seed=_SEED)
        cls.meso_r = _exc_inh_window_rates(cls.meso)
        cls.micro_r = _exc_inh_window_rates(cls.micro)

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_exc_rate_agrees(self):
        # Pre-step excitatory rate: meso vs micro within _RATE_RTOL.
        m, mu = self.meso_r[0], self.micro_r[0]
        self.assertGreater(mu, 0.0)
        self.assertLess(abs(m - mu) / mu, _RATE_RTOL,
                        f"exc pre-step rate meso={m:.2f} vs micro={mu:.2f} spk/s")

    def test_inh_rate_agrees(self):
        # Pre-step inhibitory rate: meso vs micro within _RATE_RTOL.
        m, mu = self.meso_r[2], self.micro_r[2]
        self.assertGreater(mu, 0.0)
        self.assertLess(abs(m - mu) / mu, _RATE_RTOL,
                        f"inh pre-step rate meso={m:.2f} vs micro={mu:.2f} spk/s")

    def test_step_response_present_in_both(self):
        # The +20 mV step in mu at t=1500 raises the excitatory rate in both sims.
        for label, (pre, post, _, _) in (("meso", self.meso_r), ("micro", self.micro_r)):
            self.assertGreater(post / pre, _MIN_STEP_JUMP,
                               f"{label}: exc step jump {post / pre:.2f}x "
                               f"(pre={pre:.2f}, post={post:.2f})")

    def test_both_show_slow_fluctuations(self):
        # Both methods produce temporally-correlated population activity (slow
        # finite-size + coupling-driven fluctuations), not independent firing:
        # the binned-activity autocorrelation stays well above zero at a 60 ms
        # lag. We assert this *shared* feature rather than the method-specific
        # adaptation dip (the cleaner meso rate model dips into anti-correlation
        # at ~26 ms; the noisier micro network's autocorrelation stays positive).
        for label, r in (("meso", self.meso), ("micro", self.micro)):
            ac = demo.autocorr(r['A_N'][:, 0], _MAX_LAG)
            self.assertGreater(ac[_MAX_LAG], 0.10,
                               f"{label}: activity not slowly correlated "
                               f"(ac[{_MAX_LAG} ms]={ac[_MAX_LAG]:.3f})")

    def test_meso_reproducible_given_seed(self):
        a = demo.run_meso(seed=3, coupled=True, t_end=400.0)['A_N']
        b = demo.run_meso(seed=3, coupled=True, t_end=400.0)['A_N']
        self.assertTrue(np.array_equal(a, b), "same seed must reproduce meso A_N exactly")


@requires_nest
class TestGifPopMesoMatchesNest(unittest.TestCase):
    """The host-side meso *driver* matches NEST gif_pop_psc_exp (uncoupled pop)."""

    _SEEDS = (1, 2, 3, 4)
    _T = 1000.0          # short window: this path has no Simulator compile

    @classmethod
    def setUpClass(cls):
        if not _HAS_NEST:
            return
        brainstate.environ.set(dt=demo.DT * u.ms)
        cls._bp = [cls._bp_run(s) for s in cls._SEEDS]
        cls._nest = [cls._nest_run(s) for s in cls._SEEDS]

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    @classmethod
    def _bp_run(cls, seed):
        """brainpy meso, single uncoupled excitatory population -> mean rate."""
        r = demo.run_meso(seed=seed, coupled=False, t_end=cls._T)
        return float(r['A_N'][:, 0].mean())

    @classmethod
    def _nest_run(cls, seed):
        """NEST gif_pop_psc_exp, single excitatory population -> mean rate."""
        nest.ResetKernel()
        nest.resolution = demo.DT
        nest.rng_seed = seed
        nest.set_verbosity("M_ERROR")
        pop = nest.Create("gif_pop_psc_exp", 1, params={
            "N": demo.N[0], "tau_m": demo.TAU_M, "C_m": demo.C_M, "t_ref": demo.T_REF,
            "lambda_0": demo.LAMBDA_0, "Delta_V": demo.DELTA_U, "E_L": demo.E_L,
            "V_reset": demo.V_RESET, "V_T_star": demo.V_TH, "I_e": demo.I_E,
            "tau_syn_ex": demo.TAU_EX, "tau_syn_in": demo.TAU_IN,
            "tau_sfa": list(demo.TAU_SFA), "q_sfa": list(demo.Q_SFA), "len_kernel": -1})
        sr = nest.Create("spike_recorder")
        sr.time_in_steps = True
        nest.Connect(pop, sr, syn_spec={"weight": 1.0, "delay": demo.DT})
        nest.Simulate(cls._T + demo.DT)
        times = np.asarray(sr.get("events", "times"), dtype=float) * demo.DT
        # spikes per (neuron * s): n_spikes / N / T
        return float(times.size) / (demo.N[0] * cls._T / 1000.0)

    def test_uncoupled_mean_rate_matches_nest(self):
        compare_distributional(self._nest, self._bp, tol=tc.CAT_D,
                               metric="gif_pop_psc_exp uncoupled rate",
                               statistic="mean").assert_()


if __name__ == "__main__":
    unittest.main()
