# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: symmetric nearest-neighbour ``stdp_nn_symm_synapse``.

A single plastic edge is driven by deterministic pre/post trains and its
per-send weight trajectory must match NEST's ``stdp_nn_symm_synapse``
``weight_recorder`` step-for-step. Unlike :mod:`stdp_synapse_parity_test`, the
oracle here is the *nearest-neighbour* NEST model: both ``K+`` and ``K-`` pair
only with the single nearest partner (NEST resets each trace to 1 on its own
spike). The scenarios exercise the three places this matters:

* an isolated-pair ``dt`` sweep — where nearest and all-to-all coincide, so this
  pins the basic LTP/LTD window against the symm model (category B);
* a two-pre coincident pair — NEST discards the exactly-coinciding partner and
  pairs the **second latest** (``stdp_nn_symm_synapse.h:60-64``); the substrate's
  ``K -= spike`` exclusion must reproduce it (category B);
* a dense **pre-burst** train — three pres per burst before each post, where
  nearest facilitation (last pre only) diverges sharply from the all-to-all sum;
  matching NEST symm here is the scheme-distinguishing check (category A).

The shared drive (decoupled ``iaf_psc_delta`` post, recorded fire times, NEST
dendritic-delay shift, online-vs-deferred sampling at send steps) lives in
:mod:`brainpy_state._nest_validation._stdp_drive`.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state import stdp_nn_symm_synapse
from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc
from brainpy_state._nest_validation import _stdp_drive as drv

# Mid-range params: plain LTP/LTD pairs move the weight visibly without hitting
# the [0, Wmax] clamp (so a mismatch shows up rather than being masked).
_SYN = dict(weight=5.0, Wmax=100.0, lambda_=0.1, alpha=1.0,
            mu_plus=1.0, mu_minus=1.0, tau_plus=20.0, tau_minus=20.0)


def _rule(syn):
    return stdp_nn_symm_synapse(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau_plus=syn['tau_plus'] * u.ms, tau_minus=syn['tau_minus'] * u.ms,
        lambda_=syn['lambda_'], alpha=syn['alpha'],
        mu_plus=syn['mu_plus'], mu_minus=syn['mu_minus'], Wmax=syn['Wmax'])


def _per_conn(syn):
    # NEST stdp_nn_symm_synapse exposes the stdp_synapse params minus Kplus;
    # tau_minus lives on the postsynaptic archiving node, not the synapse.
    return {"weight": syn['weight'], "Wmax": syn['Wmax'], "lambda": syn['lambda_'],
            "alpha": syn['alpha'], "mu_plus": syn['mu_plus'],
            "mu_minus": syn['mu_minus'], "tau_plus": syn['tau_plus']}


@requires_nest
class TestStdpNnSymmSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, post_want, T, tol, label):
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_nn_symm_synapse", _per_conn(syn), syn['tau_minus'],
            pre_want, post_want, T)
        bp_all = drv.bp_weight_trace(_rule(syn), pre_fire, post_fire, int(round(T / drv.DT)))
        bp_w = bp_all[drv.steps(wr_t)]                    # sample at send (pre-fire) steps
        m = min(len(nest_w), len(bp_w))
        self.assertGreater(m, 0, f"{label}: no weight samples")
        self.assertEqual(len(nest_w), len(pre_fire), f"{label}: one send per pre fire")
        compare_trace(nest_w[:m], bp_w[:m], tol=tol, metric=label).assert_()

    # -- isolated-pair window: nearest == all-to-all here, pins LTP/LTD shape --
    def test_single_pair_window_sweep(self):
        # NEST quirk: symm has no Kplus trace, so its first send (t_lastspike_=0)
        # facilitates any *preceding* post with exp(-(t_post+d)/tau_plus) — pairing
        # it with a phantom pre at t=0. The substrate seeds pre_trace=0 ("no virtual
        # pre", the physically correct nearest semantics) and does not reproduce it.
        # Placing the pair at P0=500 makes that phantom term < CAT_B atol for every
        # dt<0 (exp(-461/20)~1e-10), so depression parity is asserted cleanly. dt>0
        # (post after pre) never triggers the phantom. See develop/NEST_PARITY_LEDGER.md Lessons (05).
        P0, P_flush = 500.0, 720.0           # flush pre records the deferred LTP
        for dt in (-40., -20., -10., -5., 5., 10., 20., 40.):
            with self.subTest(dt=dt):
                self._run(_SYN, [P0, P_flush], [P0 + dt], T=P_flush + 100.,
                          tol=tc.CAT_B, label=f"symm pair dt={dt:+.0f}")

    # -- coincident post effect lands on the pre step (same-step exclusion) ---
    def test_simultaneous_pair_matches_nest(self):
        # A real lead pre@100 precedes the coincident post (effect q+d lands on
        # pre@300), so no phantom: facilitation pairs the post with pre@100, while
        # depression at pre@300 must EXCLUDE the exactly-coinciding post (K- = 0).
        self._run(_SYN, [100.0, 300.0], [300.0 - drv.DEND_D], T=400.0,
                  tol=tc.CAT_B, label="symm coincident")

    # -- second-latest pairing on coincidence: two pres, post coincides with the
    #    later one -> NEST discards it and facilitates with the earlier pre -----
    def test_second_latest_on_coincidence_matches_nest(self):
        # pres at 100 and 140; the post's effect (q + d) coincides with pre@140,
        # so the facilitating partner is the *second latest* pre (100), per
        # stdp_nn_symm_synapse.h:60-64. A flush pre at 360 records the last LTP.
        self._run(_SYN, [100.0, 140.0, 360.0], [140.0 - drv.DEND_D], T=460.0,
                  tol=tc.CAT_B, label="symm second-latest")

    # -- THE scheme-distinguishing train: 3-pre bursts before each post, where
    #    nearest facilitation (last pre only) diverges from the all-to-all sum --
    def test_divergent_burst_train_matches_nest(self):
        pre_want, post_want = [], []
        for base in np.arange(100.0, 1000.0, 100.0):       # 9 bursts
            pre_want += [base, base + 10.0, base + 20.0]    # dense burst (10 ms ISI)
            post_want.append(base + 25.0)                   # post just after the burst
        pre_want.append(1050.0)                             # flush the final LTP
        self._run(_SYN, pre_want, post_want, T=1150.0,
                  tol=tc.CAT_A, label="symm burst train")

    # -- 5 s fixed train: many interacting nearest pairs, soft multiplicative bound
    def test_fixed_train_trajectory(self):
        pre_want = list(np.arange(50.0, 5000.0, 50.0))      # 50 ms ISI, ~99 pairs
        post_want = [p + 10.0 for p in pre_want]            # post 10 ms after pre (LTP)
        self._run(_SYN, pre_want, post_want, T=5100.0,
                  tol=tc.CAT_A, label="symm 5s train")


if __name__ == "__main__":
    unittest.main()
