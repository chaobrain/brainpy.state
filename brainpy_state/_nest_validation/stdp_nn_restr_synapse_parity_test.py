# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: restricted symmetric nearest-neighbour ``stdp_nn_restr_synapse``.

A single plastic edge is driven by deterministic pre/post trains and its per-send
weight trajectory must match NEST's ``stdp_nn_restr_synapse`` ``weight_recorder``
step-for-step. The *restricted* scheme gates both updates on "a partner occurred
since the previous same-side spike" (NEST ``start != finish``), so each spike pairs
at most once. The scenarios cover:

* an isolated-pair ``dt`` sweep — restr == symm == stdp here, pinning the window
  (category B; placed at P0=500 so the symm/restr phantom-pre-at-0 term, which the
  substrate does not model, stays below atol for dt<0);
* a one-pre/two-post burst — only the **first** post facilitates, the second is
  gated off; depression then pairs the last post at the flush pre (category B);
* a one-post/two-pre burst — only the **first** pre depresses (category B);
* a 1-pre/3-post-per-cycle train — the restriction fires every cycle, diverging
  hard from the all-pairs symm scheme (category A).

Shared drive in :mod:`brainpy_state._nest_validation._stdp_drive`.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import stdp_nn_restr_synapse
from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc
from brainpy_state._nest_validation import _stdp_drive as drv

_SYN = dict(weight=5.0, Wmax=100.0, lambda_=0.1, alpha=1.0,
            mu_plus=1.0, mu_minus=1.0, tau_plus=20.0, tau_minus=20.0)


def _rule(syn):
    return stdp_nn_restr_synapse(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau_plus=syn['tau_plus'] * u.ms, tau_minus=syn['tau_minus'] * u.ms,
        lambda_=syn['lambda_'], alpha=syn['alpha'],
        mu_plus=syn['mu_plus'], mu_minus=syn['mu_minus'], Wmax=syn['Wmax'])


def _per_conn(syn):
    # NEST stdp_nn_restr_synapse exposes the stdp_synapse params minus Kplus.
    return {"weight": syn['weight'], "Wmax": syn['Wmax'], "lambda": syn['lambda_'],
            "alpha": syn['alpha'], "mu_plus": syn['mu_plus'],
            "mu_minus": syn['mu_minus'], "tau_plus": syn['tau_plus']}


@requires_nest
class TestStdpNnRestrSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, post_want, T, tol, label):
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_nn_restr_synapse", _per_conn(syn), syn['tau_minus'],
            pre_want, post_want, T)
        bp_all = drv.bp_weight_trace(_rule(syn), pre_fire, post_fire, int(round(T / drv.DT)))
        bp_w = bp_all[drv.steps(wr_t)]
        m = min(len(nest_w), len(bp_w))
        self.assertGreater(m, 0, f"{label}: no weight samples")
        self.assertEqual(len(nest_w), len(pre_fire), f"{label}: one send per pre fire")
        compare_trace(nest_w[:m], bp_w[:m], tol=tol, metric=label).assert_()

    # -- isolated-pair window: restr == symm == stdp here -------------------
    def test_single_pair_window_sweep(self):
        # P0=500 keeps the phantom-pre-at-0 facilitation (symm/restr have no Kplus
        # trace; the substrate seeds the gate/trace empty and does not model it)
        # below CAT_B atol for every dt<0. See CONTEXT.md Lessons (05).
        P0, P_flush = 500.0, 720.0
        for dt in (-40., -20., -10., -5., 5., 10., 20., 40.):
            with self.subTest(dt=dt):
                self._run(_SYN, [P0, P_flush], [P0 + dt], T=P_flush + 100.,
                          tol=tc.CAT_B, label=f"restr pair dt={dt:+.0f}")

    # -- one pre, two posts: only the FIRST post facilitates (restriction) ---
    def test_restricted_two_post_burst_matches_nest(self):
        # pre@100 then post@120, post@140; flush pre@320 records the deferred
        # facilitation (post@120 only) and depression (nearest post@140).
        self._run(_SYN, [100.0, 320.0], [120.0, 140.0], T=420.0,
                  tol=tc.CAT_B, label="restr 2-post burst")

    # -- one post, two pres: only the FIRST pre depresses (restriction) ------
    def test_restricted_two_pre_burst_matches_nest(self):
        # post@500 precedes pre@520, pre@540: pre@520 depresses with post@500,
        # pre@540 is gated off (post already consumed). Late so the lone post's
        # phantom facilitation at pre@520 is < atol. Flush pre@760.
        self._run(_SYN, [520.0, 540.0, 760.0], [500.0], T=860.0,
                  tol=tc.CAT_B, label="restr 2-pre burst")

    # -- divergent train: 1 pre + 3 posts per cycle; restriction fires each --
    def test_divergent_restricted_train_matches_nest(self):
        pre_want, post_want = [], []
        for base in np.arange(100.0, 1000.0, 100.0):       # 9 cycles
            pre_want.append(base)
            post_want += [base + 10.0, base + 20.0, base + 30.0]   # 3 posts/cycle
        pre_want.append(1050.0)                            # flush
        self._run(_SYN, pre_want, post_want, T=1150.0,
                  tol=tc.CAT_A, label="restr divergent train")

    # -- alternating 5 s train: one-to-one pairs (restr == symm baseline) ----
    def test_fixed_train_trajectory(self):
        pre_want = list(np.arange(50.0, 5000.0, 50.0))
        post_want = [p + 10.0 for p in pre_want]
        self._run(_SYN, pre_want, post_want, T=5100.0,
                  tol=tc.CAT_A, label="restr 5s train")


if __name__ == "__main__":
    unittest.main()
