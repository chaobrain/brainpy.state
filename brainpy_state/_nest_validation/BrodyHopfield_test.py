# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Distributional / phase-locking carve-out for the BrodyHopfield demo (§3.5).

The demo ``examples/nest_like/BrodyHopfield.py`` drives a population of ``iaf_psc_alpha``
neurons with a *shared* 35 Hz subthreshold oscillation, *independent* Gaussian
noise, and a per-neuron DC bias ramp, and shows that the spikes synchronize to the
oscillation (Brody & Hopfield 2003, Fig. 1).

**Why this is a carve-out, not a live-NEST trace parity.** The noise is a PRNG
stream that diverges between NEST and JAX, so a per-sample ``V_m`` / spike
comparison is meaningless — both simulators agree only *distributionally*. The
phase-locking is imposed by the *deterministic* shared oscillation, so the phase
statistics (vector strength, firing rate, phase histogram) are the robust ground
truth; this is a category **D** distributional comparison (seed-aggregated, never
per-sample).

Two things had to match NEST for the statistics to agree:

* the ``noise_generator`` refresh interval is **1.0 ms** (NEST's default ``dt``,
  *not* the simulation step) — the membrane noise variance scales with this
  interval, so a 0.1 ms refresh would make the neurons ~10x quieter and lock far
  too tightly (vector strength ~0.44 instead of NEST's ~0.27);
* the AC drive fans out *identically* to every neuron (one shared oscillation)
  while the noise fans out *independently* — exactly the Simulator's
  realize-at-post-size current-device semantics.

The NEST-free tests pin the science (the oscillation induces locking; removing it
abolishes it) and the device fan-out (independent noise vs identical AC). The
``@requires_nest`` tests confirm the seed-averaged vector strength, firing rate and
phase-histogram shape match live NEST. The preferred phase carries a small constant
offset (~0.27 rad) between NEST and the Simulator — the device connection-delay
phase shift — which is bounded (not asserted equal) and to which the vector strength
is invariant.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state._nest_validation.nest_compare import compare_distributional, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

import examples.nest_like.BrodyHopfield as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

# Reduced from the demo's N=1000 / T=600 ms: still ~10 oscillation periods and
# thousands of spikes after warmup, so the seed-mean statistics are stable while
# the per-seed NEST+brainpy runs stay fast. >= CAT_D.n_seeds seeds.
_N = 400
_T = 400.0
_SEEDS = (0, 1, 2, 3, 4)

# Distributional tolerances (category D), calibrated against a live NEST run:
#   seed-mean R    : measured rel ~0.02  -> 0.10 bound (R moves with the noise
#                    realization; also catches the 10x noise-dt regression at ~0.6)
#   seed-mean rate : measured rel ~0.003 -> CAT_D (5 %)
#   phase-hist     : measured max|Δ| ~0.006 over 18 bins -> CAT_D (autocorr 0.05)
_R_TOL = tc.DistributionalTolerance(
    rate_rtol=0.10, mean_diff_pct=0.02, autocorr_max_diff=0.05,
    n_seeds=tc.CAT_D.n_seeds,
    note="vector strength; seed-aggregated, moves with the noise realization")
# Preferred phase is robust up to the device connection-delay phase shift.
_PHASE_OFFSET_BOUND = 0.6   # rad (measured ~0.27)
_N_BINS = 18


def _rate(times, n, t_sim, warmup):
    """Post-warmup population rate (spk/s) from pooled spike times, both sides alike."""
    return float(np.asarray(times).size) / (n * (t_sim - warmup) / 1000.0)


def _bp_run(seed):
    """brainpy phase-locking run -> (R, preferred_phase, rate, pooled_times)."""
    r = demo.run_phase_locking(n=_N, t_sim=_T, seed=seed, warmup=demo.WARMUP)
    return r['R'], r['preferred_phase'], _rate(r['spike_times'], _N, _T, demo.WARMUP), \
        r['spike_times']


def _nest_run(seed):
    """Live-NEST run mirroring the demo's exact wiring -> (R, psi, rate, times)."""
    nest.ResetKernel()
    nest.resolution = demo.DT
    nest.rng_seed = seed + 1
    nest.set_verbosity("M_ERROR")
    neurons = nest.Create("iaf_psc_alpha", _N, params={
        "tau_m": demo.TAU_M, "V_th": demo.V_TH, "E_L": demo.E_L, "t_ref": demo.T_REF,
        "V_reset": demo.V_RESET, "C_m": demo.C_M, "V_m": demo.V_INIT})
    neurons.I_e = list(demo.bias_ramp(_N))
    # NEST noise_generator default dt is 1.0 ms (== demo.NOISE_DT); AC drive defaults
    # to phase 0. One generator each -> shared AC, independent noise per target.
    drive = nest.Create("ac_generator",
                         params={"amplitude": demo.DRIVE_AMP, "frequency": demo.DRIVE_FREQ})
    noise = nest.Create("noise_generator",
                        params={"mean": 0.0, "std": demo.NOISE_STD, "dt": demo.NOISE_DT})
    sr = nest.Create("spike_recorder")
    nest.Connect(drive, neurons)
    nest.Connect(noise, neurons)
    nest.Connect(neurons, sr)
    nest.Simulate(_T)
    times = np.asarray(sr.events["times"], dtype=float)
    times = times[times >= demo.WARMUP]
    R, psi = demo.vector_strength(times)
    return R, psi, _rate(times, _N, _T, demo.WARMUP), times


class TestBrodyHopfieldPhaseLocking(unittest.TestCase):
    """NEST-free: the science (oscillation -> locking) and the device fan-out."""

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_oscillation_induces_phase_locking(self):
        # The core Brody-Hopfield phenomenon: the 35 Hz drive synchronizes spikes;
        # removing it (amplitude=0) leaves phases ~uniform (vector strength ~0).
        osc = demo.run_phase_locking(n=_N, t_sim=_T, seed=0)
        flat = demo.run_phase_locking(n=_N, t_sim=_T, seed=0, drive_amp=0.0)
        self.assertGreater(osc['R'], 0.15,
                           f"35 Hz drive should phase-lock (R={osc['R']:.3f})")
        self.assertLess(flat['R'], 0.08,
                        f"no-oscillation control should be near chance (R={flat['R']:.3f})")
        self.assertGreater(osc['R'], 3.0 * flat['R'],
                           f"locking must be driven by the oscillation "
                           f"(R_osc={osc['R']:.3f} vs R_flat={flat['R']:.3f})")

    def test_noise_is_independent_per_neuron(self):
        # Identical bias, noise on, AC off: a single noise_generator must deliver an
        # independent stream per neuron, so the spike trains differ.
        sim, sr, _t = demo.build(n=6, t_sim=300.0, drive_amp=0.0, noise_std=demo.NOISE_STD,
                                 seed=0, i_e=180.0)
        spk = np.asarray(sim.simulate(300.0 * u.ms).spikes(sr))
        all_equal = all(np.array_equal(spk[:, 0], spk[:, j]) for j in range(1, spk.shape[1]))
        self.assertFalse(all_equal, "noise_generator must fan out independent streams")
        self.assertGreater(int(spk.sum()), 0, "neurons should fire under the noise drive")

    def test_ac_drive_is_identical_per_neuron(self):
        # Identical bias, noise off, AC on: a single ac_generator must drive every
        # neuron with the *same* deterministic current, so the spike trains coincide.
        sim, sr, _t = demo.build(n=6, t_sim=300.0, drive_amp=demo.DRIVE_AMP, noise_std=0.0,
                                 seed=0, i_e=180.0)
        spk = np.asarray(sim.simulate(300.0 * u.ms).spikes(sr))
        all_equal = all(np.array_equal(spk[:, 0], spk[:, j]) for j in range(1, spk.shape[1]))
        self.assertTrue(all_equal, "ac_generator must fan out an identical drive")
        self.assertGreater(int(spk.sum()), 0, "biased neurons should fire")

    def test_reproducible_given_seed(self):
        # Same seed -> bit-identical phase-locking statistic.
        a = demo.run_phase_locking(n=_N, t_sim=_T, seed=3)
        b = demo.run_phase_locking(n=_N, t_sim=_T, seed=3)
        self.assertEqual(a['R'], b['R'], "same seed must reproduce the vector strength")
        self.assertEqual(a['n_spikes'], b['n_spikes'])


@requires_nest
class TestBrodyHopfieldDistributionalParity(unittest.TestCase):
    """The seed-averaged phase statistics match live NEST (category D)."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_NEST:
            return
        brainstate.environ.set(dt=demo.DT * u.ms)
        cls._nest = [_nest_run(s) for s in _SEEDS]
        cls._bp = [_bp_run(s) for s in _SEEDS]
        # Pool spikes across seeds per side; each side's own preferred phase aligns
        # the histograms so the constant connection-delay offset is removed.
        cls._nest_times = np.concatenate([r[3] for r in cls._nest])
        cls._bp_times = np.concatenate([r[3] for r in cls._bp])
        cls._nest_psi = demo.vector_strength(cls._nest_times)[1]
        cls._bp_psi = demo.vector_strength(cls._bp_times)[1]

    def setUp(self):
        brainstate.environ.set(dt=demo.DT * u.ms)

    def test_seed_mean_vector_strength_matches_nest(self):
        nR = [r[0] for r in self._nest]
        bR = [r[0] for r in self._bp]
        compare_distributional(nR, bR, tol=_R_TOL, metric="BrodyHopfield vector strength",
                               statistic="mean").assert_()

    def test_seed_mean_rate_matches_nest(self):
        nrate = [r[2] for r in self._nest]
        brate = [r[2] for r in self._bp]
        compare_distributional(nrate, brate, tol=tc.CAT_D, metric="BrodyHopfield rate",
                               statistic="mean").assert_()

    def test_phase_histogram_shape_matches_nest(self):
        # Seed-pooled, each aligned to its own preferred phase, compared elementwise.
        nh = demo.phase_histogram(self._nest_times, n_bins=_N_BINS, align_phase=self._nest_psi)
        bh = demo.phase_histogram(self._bp_times, n_bins=_N_BINS, align_phase=self._bp_psi)
        compare_distributional(nh, bh, tol=tc.CAT_D, metric="BrodyHopfield phase histogram",
                               statistic="autocorr").assert_()

    def test_preferred_phase_offset_is_bounded(self):
        # The locking is to the same phase region up to a constant connection-delay
        # shift; R (rotation-invariant) is unaffected. Bound, don't assert equal.
        dpsi = float(np.angle(np.exp(1j * (self._bp_psi - self._nest_psi))))
        self.assertLess(abs(dpsi), _PHASE_OFFSET_BOUND,
                        f"preferred-phase circular offset {np.degrees(dpsi):.1f} deg "
                        f"exceeds {np.degrees(_PHASE_OFFSET_BOUND):.0f} deg")


if __name__ == "__main__":
    unittest.main()
