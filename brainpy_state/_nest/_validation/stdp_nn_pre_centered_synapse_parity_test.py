# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: presynaptic-centered ``stdp_nn_pre_centered_synapse``.

A single plastic edge is driven by deterministic pre/post trains and its per-send
weight trajectory must match NEST's ``stdp_nn_pre_centered_synapse``
``weight_recorder`` step-for-step. Here the presynaptic ``Kplus`` is per-edge:
it accumulates ``+1`` per pre, decays at ``tau_plus``, and resets to 0 on each post,
so a post facilitates with the *sum* of pres since the previous post. The scenarios:

* an isolated-pair ``dt`` sweep — pinning the basic LTP/LTD window. Unlike symm/restr
  this model is immune to the phantom-pre-at-0 (facilitation is ``Kplus·exp`` and
  ``Kplus`` starts at 0), so the pair sits at P0=100 (category B);
* a two-pre/one-post burst — the post facilitates with the **accumulated** ``Kplus``
  (sum of both pres), the presynaptic-centered signature (category B);
* a 2-pre + 1-post-per-cycle train — accumulation and post-triggered reset fire every
  cycle (category A);
* an alternating 5 s train (category A baseline).

Shared drive in :mod:`brainpy_state._nest._validation._stdp_drive`.
"""
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import stdp_nn_pre_centered_synapse
from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _stdp_drive as drv

_SYN = dict(weight=5.0, Wmax=100.0, lambda_=0.1, alpha=1.0,
            mu_plus=1.0, mu_minus=1.0, tau_plus=20.0, tau_minus=20.0, Kplus=0.0)


def _rule(syn):
    return stdp_nn_pre_centered_synapse(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau_plus=syn['tau_plus'] * u.ms, tau_minus=syn['tau_minus'] * u.ms,
        lambda_=syn['lambda_'], alpha=syn['alpha'],
        mu_plus=syn['mu_plus'], mu_minus=syn['mu_minus'], Wmax=syn['Wmax'],
        Kplus=syn['Kplus'])


def _per_conn(syn):
    return {"weight": syn['weight'], "Wmax": syn['Wmax'], "lambda": syn['lambda_'],
            "alpha": syn['alpha'], "mu_plus": syn['mu_plus'],
            "mu_minus": syn['mu_minus'], "tau_plus": syn['tau_plus'], "Kplus": syn['Kplus']}


@requires_nest
class TestStdpNnPreCenteredSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, post_want, T, tol, label):
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_nn_pre_centered_synapse", _per_conn(syn), syn['tau_minus'],
            pre_want, post_want, T)
        bp_all = drv.bp_weight_trace(_rule(syn), pre_fire, post_fire, int(round(T / drv.DT)))
        bp_w = bp_all[drv.steps(wr_t)]
        m = min(len(nest_w), len(bp_w))
        self.assertGreater(m, 0, f"{label}: no weight samples")
        self.assertEqual(len(nest_w), len(pre_fire), f"{label}: one send per pre fire")
        compare_trace(nest_w[:m], bp_w[:m], tol=tol, metric=label).assert_()

    # -- isolated-pair window (immune to the phantom-pre-at-0; P0=100) -------
    def test_single_pair_window_sweep(self):
        P0, P_flush = 100.0, 320.0
        for dt in (-40., -20., -10., -5., 5., 10., 20., 40.):
            with self.subTest(dt=dt):
                self._run(_SYN, [P0, P_flush], [P0 + dt], T=P_flush + 100.,
                          tol=tc.CAT_B, label=f"prec pair dt={dt:+.0f}")

    # -- two pres then a post: facilitation uses the ACCUMULATED Kplus -------
    def test_accumulation_two_pre_burst_matches_nest(self):
        # pre@100, pre@120 accumulate; post@140 facilitates with their sum; flush@320.
        self._run(_SYN, [100.0, 120.0, 320.0], [140.0], T=420.0,
                  tol=tc.CAT_B, label="prec accumulation burst")

    # -- divergent train: 2 pres + 1 post per cycle; accumulate then reset ---
    def test_divergent_pre_centered_train_matches_nest(self):
        pre_want, post_want = [], []
        for base in np.arange(100.0, 1000.0, 100.0):       # 9 cycles
            pre_want += [base, base + 10.0]                 # two pres accumulate
            post_want.append(base + 30.0)                   # post sums them, then resets
        pre_want.append(1050.0)                            # flush
        self._run(_SYN, pre_want, post_want, T=1150.0,
                  tol=tc.CAT_A, label="prec divergent train")

    # -- alternating 5 s train (one pre per post -> Kplus==1 each pairing) ----
    def test_fixed_train_trajectory(self):
        pre_want = list(np.arange(50.0, 5000.0, 50.0))
        post_want = [p + 10.0 for p in pre_want]
        self._run(_SYN, pre_want, post_want, T=5100.0,
                  tol=tc.CAT_A, label="prec 5s train")


if __name__ == "__main__":
    unittest.main()
