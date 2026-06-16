# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``stdp_pl_synapse_hom`` (power-law STDP) weight trajectories.

Power-law potentiation (``w + lambda*w^mu*K+``) and linear depression
(``max(w - alpha*lambda*w*K-, 0)``) with no upper bound. The parameters are NEST
*common* (homogeneous) properties (set via ``CopyModel`` defaults); only the
weight is per-connection. The rebuilt spec must reproduce NEST's weight_recorder
trajectory step-for-step. Drive shared with the canonical test
(:mod:`brainpy_state._nest_validation._stdp_drive`).
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state import stdp_pl_synapse_hom
from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc
from brainpy_state._nest_validation import _stdp_drive as drv

_SYN = dict(weight=5.0, lambda_=0.1, alpha=1.0, mu=0.4, tau_plus=20.0, tau_minus=20.0)


def _rule(syn):
    return stdp_pl_synapse_hom(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau_plus=syn['tau_plus'] * u.ms, tau_minus=syn['tau_minus'] * u.ms,
        lambda_=syn['lambda_'], alpha=syn['alpha'], mu=syn['mu'])


@requires_nest
class TestStdpPlSynapseHomParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, post_want, T, tol, label):
        # pl_hom: tau_plus/lambda/alpha/mu are common (CopyModel defaults).
        common = {"lambda": syn['lambda_'], "alpha": syn['alpha'],
                  "mu": syn['mu'], "tau_plus": syn['tau_plus']}
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_pl_synapse_hom", {"weight": syn['weight']}, syn['tau_minus'],
            pre_want, post_want, T, common=common)
        bp_all = drv.bp_weight_trace(_rule(syn), pre_fire, post_fire, int(round(T / drv.DT)))
        bp_w = bp_all[drv.steps(wr_t)]
        m = min(len(nest_w), len(bp_w))
        self.assertGreater(m, 0, f"{label}: no weight samples")
        self.assertEqual(len(nest_w), len(pre_fire), f"{label}: one send per pre fire")
        compare_trace(nest_w[:m], bp_w[:m], tol=tol, metric=label).assert_()

    def test_single_pair_window_sweep(self):
        P0, P_flush = 100.0, 320.0
        for dt in (-30., -10., -5., 5., 10., 30.):
            with self.subTest(dt=dt):
                self._run(_SYN, [P0, P_flush], [P0 + dt], T=P_flush + 100.,
                          tol=tc.CAT_B, label=f"pl pair dt={dt:+.0f}")

    def test_fixed_train_trajectory(self):
        pre_want = list(np.arange(50.0, 5000.0, 50.0))
        post_want = [p + 10.0 for p in pre_want]
        self._run(_SYN, pre_want, post_want, T=5100.0,
                  tol=tc.CAT_A, label="pl 5s train")

    def test_depression_dominated_train(self):
        # post before pre (LTD-dominated) drives the weight down toward the 0 clip.
        pre_want = list(np.arange(50.0, 2000.0, 40.0))
        post_want = [p - 10.0 for p in pre_want]
        self._run(dict(_SYN, weight=20.0), pre_want, post_want, T=2100.0,
                  tol=tc.CAT_A, label="pl LTD train")


if __name__ == "__main__":
    unittest.main()
