# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``jonke_synapse`` (exponential-weight STDP) trajectories.

Exponential weight factors ``exp(mu_plus*w)`` / ``exp(mu_minus*w)`` plus the
``beta`` offset, with one-sided clips (facilitation upper-bound ``Wmax``,
depression lower-bound ``0``). Parameters are per-connection (``jonke_synapse``
is not a ``*_hom`` model). The rebuilt spec must reproduce NEST's weight_recorder
trajectory step-for-step. Drive shared with the canonical test
(:mod:`brainpy_state._nest._validation._stdp_drive`).
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import jonke_synapse
from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _stdp_drive as drv

# mu>0 to exercise the exponential weight dependence (the interesting path).
_SYN = dict(weight=10.0, Wmax=100.0, lambda_=0.02, alpha=1.0,
            mu_plus=0.05, mu_minus=0.02, beta=0.0, tau_plus=20.0, tau_minus=20.0)


def _rule(syn):
    return jonke_synapse(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau_plus=syn['tau_plus'] * u.ms, tau_minus=syn['tau_minus'] * u.ms,
        lambda_=syn['lambda_'], alpha=syn['alpha'], mu_plus=syn['mu_plus'],
        mu_minus=syn['mu_minus'], beta=syn['beta'], Wmax=syn['Wmax'])


def _common(syn):
    # jonke_synapse stores plasticity params as common (CopyModel) properties.
    return {"Wmax": syn['Wmax'], "lambda": syn['lambda_'], "alpha": syn['alpha'],
            "mu_plus": syn['mu_plus'], "mu_minus": syn['mu_minus'],
            "beta": syn['beta'], "tau_plus": syn['tau_plus']}


@requires_nest
class TestJonkeSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, post_want, T, tol, label):
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "jonke_synapse", {"weight": syn['weight']}, syn['tau_minus'],
            pre_want, post_want, T, common=_common(syn))
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
                          tol=tc.CAT_B, label=f"jonke pair dt={dt:+.0f}")

    def test_fixed_train_trajectory(self):
        pre_want = list(np.arange(50.0, 5000.0, 50.0))
        post_want = [p + 10.0 for p in pre_want]
        self._run(_SYN, pre_want, post_want, T=5100.0,
                  tol=tc.CAT_A, label="jonke 5s train")

    def test_beta_offset_train(self):
        # nonzero beta: an activity-independent bias on every update.
        syn = dict(_SYN, beta=0.05, mu_plus=0.0, mu_minus=0.0)
        pre_want = list(np.arange(50.0, 2000.0, 40.0))
        post_want = [p + 8.0 for p in pre_want]
        self._run(syn, pre_want, post_want, T=2100.0,
                  tol=tc.CAT_A, label="jonke beta train")


if __name__ == "__main__":
    unittest.main()
