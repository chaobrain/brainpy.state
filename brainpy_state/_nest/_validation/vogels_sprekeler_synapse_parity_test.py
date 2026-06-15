# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``vogels_sprekeler_synapse`` (symmetric inhibitory) trajectories.

Symmetric facilitation on both sides plus a constant per-pre-spike depression,
with magnitude saturation at ``±|Wmax|``. The symmetric trace requires the post
node's ``tau_minus`` to equal the synapse ``tau``. The rebuilt spec must
reproduce NEST's weight_recorder trajectory step-for-step. Drive shared with the
canonical test (:mod:`brainpy_state._nest._validation._stdp_drive`).
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import vogels_sprekeler_synapse
from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _stdp_drive as drv

# eta bumped from the 0.001 default so single-pair changes are well above 1e-6.
_SYN = dict(weight=0.5, Wmax=1.0, eta=0.01, alpha=0.12, tau=20.0)


def _rule(syn):
    return vogels_sprekeler_synapse(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau=syn['tau'] * u.ms, eta=syn['eta'], alpha=syn['alpha'], Wmax=syn['Wmax'])


@requires_nest
class TestVogelsSprekelerSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, post_want, T, tol, label):
        # symmetric trace: post node tau_minus must equal the synapse tau.
        common = {"Wmax": syn['Wmax'], "eta": syn['eta'], "alpha": syn['alpha'],
                  "tau": syn['tau']}
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "vogels_sprekeler_synapse", {"weight": syn['weight']}, syn['tau'],
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
                          tol=tc.CAT_B, label=f"vogels pair dt={dt:+.0f}")

    def test_fixed_train_trajectory(self):
        # paired pre/post drives facilitation against the constant depression.
        pre_want = list(np.arange(50.0, 5000.0, 50.0))
        post_want = [p + 5.0 for p in pre_want]
        self._run(_SYN, pre_want, post_want, T=5100.0,
                  tol=tc.CAT_A, label="vogels 5s train")

    def test_pre_only_constant_depression_train(self):
        # pre-only activity: pure constant depression toward the 0 floor.
        pre_want = list(np.arange(50.0, 1500.0, 25.0))
        self._run(dict(_SYN, weight=0.9), pre_want, [], T=1600.0,
                  tol=tc.CAT_A, label="vogels pre-only")


if __name__ == "__main__":
    unittest.main()
