# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: dopamine-modulated ``stdp_dopamine_synapse`` weight (cluster-08).

Layered on the ``volume_transmitter`` ``n(t)`` precondition
(``volume_transmitter_parity_test.py``, validated near-exact upstream), this locks
the **weight trajectory** the broadcast ``n`` drives. A single dopamine edge is
driven with deterministic pre/post/dopa trains; NEST's ``weight_recorder`` logs the
weight at every pre ``send`` and we sample our per-step online integral at the same
steps (the shared drive lives in
:mod:`brainpy_state._nest_validation._stdp_dopamine_drive`).

Three regimes are covered:

1. **Eligibility window.** A single pre/post pair at varying lag sets the sign and
   magnitude of the eligibility trace ``c``; a common dopa read-out pulse converts
   ``c`` into a weight change. Pre-before-post potentiates, post-before-pre depresses,
   ``|Δw|`` grows as the pairing tightens — direction, ordering, and magnitude all
   match NEST.
2. **Sustained trajectory.** Periodic pairing under a steady dopa train drives a long
   (multi-second) potentiation / depression trajectory that tracks NEST send-for-send.
3. **Clamp.** The ``[Wmin, Wmax]`` saturation matches NEST.

**Why the band is tight (unlike Clopath).** Clopath's online instantaneous voltage
read diverges a few percent from NEST's ring-buffered history. Here the broadcast
``n`` is a clean scalar recursion with no analog-history dependence, so the only
residual is the one-step ``n`` lag and the per-step-vs-deferred weight integration —
observed ``max|Δw| < 0.01`` pA over trajectories spanning tens to hundreds of pA
(~0.2 % of ``Δw``). Direction and ordering are exact.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc
from brainpy_state._nest_validation import _stdp_dopamine_drive as drv

# Documented dopamine weight band: the online one-step-n-lag + per-step-vs-deferred
# integration residual. Observed max|Δw| ~ 7e-3 pA across LTP/LTD/clamp/sweep and
# tau_n in [50, 200] ms; the band is set ~4x above that (atol) with a gentle rtol
# for the large-amplitude sustained trajectories. Weights are bare pA mantissas.
_WEIGHT_BAND = tc.TraceTolerance(3e-2, 2e-3, label="dopamine",
                                 note="online one-step-n-lag vs NEST deferred integral")
_INIT_W = 50.0
_TAU_MINUS = 20.0
# common (CopyModel) dopamine properties shared by the directional scenarios
_COMMON = dict(A_plus=1.0, A_minus=1.5, tau_plus=20.0, tau_c=1000.0, tau_n=200.0,
               b=0.0, Wmin=-1.0e9, Wmax=1.0e9)
#: STDP-window lags (ms): negative = post-before-pre (LTD), positive = pre-before-post (LTP).
_SWEEP_LAGS = (-40.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 40.0)


def _sweep_dw(lag):
    """Δw for a single pre/post pair at ``lag`` ms, read out by a common dopa pulse.

    The dopa read-out fires *after* the single pre send, so the decisive weight change
    is captured by the final-step weight (``our_final`` / NEST ``GetConnections``), not
    the send-time sample (which is still at the initial weight).
    """
    P = 100.0
    dopa = list(np.arange(160.0, 260.0, 5.0))
    wr_t, nest_w, our_w, final, our_final = drv.dopamine_trajectory(
        _COMMON, _TAU_MINUS, pre_want=[P], post_want=[P + lag], dopa_arrival=dopa,
        T=400.0, init_w=_INIT_W)
    return final['weight'] - _INIT_W, our_final - _INIT_W, final['c']


@requires_nest
class TestStdpDopamineSynapseParity(unittest.TestCase):
    """Live-NEST parity for the rebuilt dopamine-modulated ``stdp_dopamine_synapse``."""

    @classmethod
    def setUpClass(cls):
        brainstate.environ.set(dt=drv.DT * u.ms)

        # 1. eligibility-window sweep (single pair at varying lag + dopa read-out)
        cls.lags = np.asarray(_SWEEP_LAGS)
        nd, od = [], []
        for lag in _SWEEP_LAGS:
            n_dw, o_dw, _c = _sweep_dw(lag)
            nd.append(n_dw)
            od.append(o_dw)
        cls.nest_sweep = np.asarray(nd)
        cls.our_sweep = np.asarray(od)

        # 2a. sustained potentiation (~5 s): periodic post-after-pre under steady dopa
        T = 5000.0
        pre = list(np.arange(60.0, T - 20, 60.0))
        post = [p + 10.0 for p in pre]
        dopa = list(np.arange(30.0, T - 20, 50.0))
        (cls.ltp_t, cls.ltp_nest, cls.ltp_our, cls.ltp_final,
         cls.ltp_our_final) = drv.dopamine_trajectory(
            _COMMON, _TAU_MINUS, pre, post, dopa, T, init_w=_INIT_W)

        # 2b. sustained depression: periodic post-before-pre under steady dopa
        T = 1500.0
        pre = list(np.arange(70.0, T - 20, 50.0))
        post = [p - 12.0 for p in pre]
        dopa = list(np.arange(30.0, T - 20, 40.0))
        (cls.ltd_t, cls.ltd_nest, cls.ltd_our, cls.ltd_final,
         cls.ltd_our_final) = drv.dopamine_trajectory(
            _COMMON, _TAU_MINUS, pre, post, dopa, T, init_w=_INIT_W)

        # 3. clamp: potentiation into a finite Wmax (and Wmin=init so it can only rise)
        T = 2500.0
        clamp_common = dict(_COMMON, Wmin=_INIT_W, Wmax=_INIT_W + 40.0)
        pre = list(np.arange(60.0, T - 20, 50.0))
        post = [p + 10.0 for p in pre]
        dopa = list(np.arange(30.0, T - 20, 40.0))
        (cls.clamp_t, cls.clamp_nest, cls.clamp_our, cls.clamp_final,
         cls.clamp_our_final) = drv.dopamine_trajectory(
            clamp_common, _TAU_MINUS, pre, post, dopa, T, init_w=_INIT_W)

    # -- 1a. eligibility window: direction matches NEST at every lag --------
    def test_eligibility_window_direction_matches_nest(self):
        for lag, nd, od in zip(self.lags, self.nest_sweep, self.our_sweep):
            with self.subTest(lag=lag):
                # pre-before-post (lag>0) potentiates; post-before-pre (lag<0) depresses
                want = 1 if lag > 0 else -1
                self.assertEqual(int(np.sign(round(nd, 6))), want, "NEST window sign")
                self.assertEqual(int(np.sign(round(od, 6))), int(np.sign(round(nd, 6))),
                                 f"our window sign must match NEST at lag {lag}")

    # -- 1b. eligibility window: magnitude matches NEST in-band -------------
    def test_eligibility_window_magnitude_within_band(self):
        compare_trace(self.nest_sweep, self.our_sweep,
                      tol=_WEIGHT_BAND, metric="dopamine STDP window").assert_()

    # -- 1c. eligibility window: |Δw| grows as the pairing tightens ---------
    def test_eligibility_window_monotonic_ordering(self):
        # potentiation side (lag>0): |Δw| decreases with lag; depression side mirrors.
        pos = self.lags > 0
        neg = self.lags < 0
        for side, mask in (("LTP", pos), ("LTD", neg)):
            order = np.argsort(np.abs(self.lags[mask]))    # nearest pairing first
            nest_mag = np.abs(self.nest_sweep[mask])[order]
            our_mag = np.abs(self.our_sweep[mask])[order]
            with self.subTest(side=side):
                self.assertTrue(np.all(np.diff(nest_mag) < 0), f"NEST {side} not monotone: {nest_mag}")
                self.assertTrue(np.all(np.diff(our_mag) < 0), f"our {side} not monotone: {our_mag}")

    # -- 2a. sustained potentiation tracks NEST send-for-send --------------
    def test_potentiation_trajectory_within_band(self):
        self.assertGreater(self.ltp_nest[-1], _INIT_W, "NEST sanity: trajectory potentiates")
        self.assertGreater(self.ltp_our[-1], _INIT_W, "ours must potentiate too")
        compare_trace(self.ltp_nest, self.ltp_our,
                      tol=_WEIGHT_BAND, metric="dopamine LTP trajectory").assert_()

    # -- 2b. potentiation is monotone-up on both sides ---------------------
    def test_potentiation_trajectory_monotone(self):
        self.assertTrue(np.all(np.diff(self.ltp_nest) >= -1e-9), "NEST LTP not monotone")
        self.assertTrue(np.all(np.diff(self.ltp_our) >= -1e-9), "our LTP not monotone")

    # -- 2c. sustained depression tracks NEST send-for-send ----------------
    def test_depression_trajectory_within_band(self):
        self.assertLess(self.ltd_nest[-1], _INIT_W, "NEST sanity: trajectory depresses")
        self.assertLess(self.ltd_our[-1], _INIT_W, "ours must depress too")
        compare_trace(self.ltd_nest, self.ltd_our,
                      tol=_WEIGHT_BAND, metric="dopamine LTD trajectory").assert_()

    # -- 2d. direction matches NEST at every send (sustained LTP) ----------
    def test_sustained_direction_matches_nest_each_send(self):
        nest_dir = np.sign(np.round(self.ltp_nest - _INIT_W, 6))
        our_dir = np.sign(np.round(self.ltp_our - _INIT_W, 6))
        np.testing.assert_array_equal(our_dir, nest_dir,
                                      "per-send weight-change sign must match NEST")

    # -- 3. clamp saturation matches NEST ----------------------------------
    def test_weight_clamp_matches_nest(self):
        wmax = _INIT_W + 40.0
        # both saturate at Wmax and the whole trajectory stays in-band
        self.assertAlmostEqual(float(self.clamp_nest[-1]), wmax, places=6)
        self.assertLessEqual(float(np.max(self.clamp_our)), wmax + 1e-6, "ours must not exceed Wmax")
        self.assertAlmostEqual(float(self.clamp_our[-1]), wmax, places=3)
        compare_trace(self.clamp_nest, self.clamp_our,
                      tol=_WEIGHT_BAND, metric="dopamine clamp trajectory").assert_()

    # -- final-state cross-check: our online weight matches NEST's stored w -
    def test_final_weight_matches_nest_connection(self):
        # NEST's GetConnections weight at T (its own deferred integral) vs our last send sample
        for lab, nest_final, our_final in (
            ("LTP", self.ltp_final['weight'], self.ltp_our_final),
            ("LTD", self.ltd_final['weight'], self.ltd_our_final),
        ):
            with self.subTest(scenario=lab):
                compare_trace(nest_final, our_final,
                              tol=_WEIGHT_BAND, metric=f"dopamine {lab} final w").assert_()


if __name__ == "__main__":
    unittest.main()
