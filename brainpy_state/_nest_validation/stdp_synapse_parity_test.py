# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: pair-based ``stdp_synapse`` weight trajectories.

A single plastic STDP edge is driven by deterministic pre/post spike trains and
its per-send weight trajectory must match NEST's ``weight_recorder``
step-for-step (category B for single pairs, category A over a 5 s train). The
shared drive (decoupled ``iaf_psc_delta`` post, recorded fire times, NEST
dendritic-delay shift, online-vs-deferred sampling at send steps) lives in
:mod:`brainpy_state._nest_validation._stdp_drive`.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import stdp_synapse
from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc
from brainpy_state._nest_validation import _stdp_drive as drv

# Default pair-based stdp_synapse parameters (mid-range so plain LTP/LTD pairs
# move the weight visibly without hitting the [0, Wmax] clamp).
_SYN = dict(weight=5.0, Wmax=100.0, lambda_=0.1, alpha=1.0,
            mu_plus=1.0, mu_minus=1.0, tau_plus=20.0, tau_minus=20.0)


def _rule(syn):
    return stdp_synapse(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau_plus=syn['tau_plus'] * u.ms, tau_minus=syn['tau_minus'] * u.ms,
        lambda_=syn['lambda_'], alpha=syn['alpha'],
        mu_plus=syn['mu_plus'], mu_minus=syn['mu_minus'], Wmax=syn['Wmax'])


def _per_conn(syn):
    return {"weight": syn['weight'], "Wmax": syn['Wmax'], "lambda": syn['lambda_'],
            "alpha": syn['alpha'], "mu_plus": syn['mu_plus'],
            "mu_minus": syn['mu_minus'], "tau_plus": syn['tau_plus']}


@requires_nest
class TestStdpSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, post_want, T, tol, label):
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_synapse", _per_conn(syn), syn['tau_minus'], pre_want, post_want, T)
        bp_all = drv.bp_weight_trace(_rule(syn), pre_fire, post_fire, int(round(T / drv.DT)))
        bp_w = bp_all[drv.steps(wr_t)]                    # sample at send (pre-fire) steps
        m = min(len(nest_w), len(bp_w))
        self.assertGreater(m, 0, f"{label}: no weight samples")
        self.assertEqual(len(nest_w), len(pre_fire), f"{label}: one send per pre fire")
        compare_trace(nest_w[:m], bp_w[:m], tol=tol, metric=label).assert_()

    # -- single-pair STDP window (LTP for dt>0, LTD for dt<0) --------------
    def test_single_pair_window_sweep(self):
        P0, P_flush = 100.0, 320.0           # flush pre records the deferred LTP
        for dt in (-40., -20., -10., -5., 5., 10., 20., 40.):
            with self.subTest(dt=dt):
                self._run(_SYN, [P0, P_flush], [P0 + dt], T=P_flush + 100.,
                          tol=tc.CAT_B, label=f"stdp pair dt={dt:+.0f}")

    # -- a coincident pre&post step (post effect q+d lands on pre p) -------
    def test_simultaneous_pair_matches_nest(self):
        # post fires d earlier so its effect (q + d) coincides with the pre step:
        # NEST facilitates (window) then depresses at the same send; the kernel's
        # `K+ - pre_spike` / `K- - post_spike` exclusions must reproduce it.
        P0, P_flush = 100.0, 320.0
        self._run(_SYN, [P0, P_flush], [P0 - drv.DEND_D], T=P_flush + 100.,
                  tol=tc.CAT_B, label="stdp coincident")

    # -- 5 s fixed train: many interacting pairs, soft multiplicative bound -
    def test_fixed_train_trajectory(self):
        pre_want = list(np.arange(50.0, 5000.0, 50.0))    # 50 ms ISI, ~99 pairs
        post_want = [p + 10.0 for p in pre_want]          # post 10 ms after pre (LTP)
        self._run(_SYN, pre_want, post_want, T=5100.0,
                  tol=tc.CAT_A, label="stdp 5s train")

    # -- weight saturates toward Wmax under strong repeated LTP (clamp parity)
    def test_wmax_saturation_matches_nest(self):
        syn = dict(_SYN, weight=80.0, lambda_=0.5)        # strong, near the bound
        pre_want = list(np.arange(50.0, 1500.0, 30.0))
        post_want = [p + 5.0 for p in pre_want]
        self._run(syn, pre_want, post_want, T=1600.0,
                  tol=tc.CAT_A, label="stdp Wmax sat")


if __name__ == "__main__":
    unittest.main()
