# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``ht_synapse`` (Hill-Tononi vesicle-pool depression) trajectories.

``ht_synapse`` is trace-free and **presynaptic only**: its stored weight is static
and the observable is the *delivered* amplitude ``w * P_send`` (what NEST's
``weight_recorder`` logs), so the brainpy side samples the delivered ``w_eff``
(``bp_weight_trace(..., delivered=True)``) rather than the stored weight. There is
no postsynaptic drive (``post_want=[]``). The recover->emit->deplete->update order
makes the trajectory exact, so single trains and the long trajectory are both held
to the tight CAT_B / CAT_A bands. Drive shared with the canonical pair test
(:mod:`brainpy_state._nest._validation._stdp_drive`).
"""
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import ht_synapse
from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _stdp_drive as drv

_SYN = dict(weight=100.0, tau_P=300.0, delta_P=0.2, P=1.0)


def _rule(syn):
    return ht_synapse(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau_P=syn['tau_P'] * u.ms, delta_P=syn['delta_P'], P=syn['P'])


@requires_nest
class TestHtSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, T, tol, label):
        # ht params are all per-connection; no post drive (depression is pre-only).
        pre_fire, _post, nest_w, wr_t = drv.nest_pair_run(
            "ht_synapse", dict(syn), 20.0, pre_want, [], T)
        bp_all = drv.bp_weight_trace(_rule(syn), pre_fire, [], int(round(T / drv.DT)),
                                     delivered=True)
        bp_w = bp_all[drv.steps(wr_t)]
        m = min(len(nest_w), len(bp_w))
        self.assertGreater(m, 0, f"{label}: no weight samples")
        self.assertEqual(len(nest_w), len(pre_fire), f"{label}: one send per pre fire")
        compare_trace(nest_w[:m], bp_w[:m], tol=tol, metric=label).assert_()

    def test_single_train_matches_nest(self):
        # isolated -> paired -> burst: spans full recovery to deep depression.
        pre_want = [50.0, 70.0, 95.0, 130.0, 200.0, 205.0, 210.0, 215.0]
        self._run(_SYN, pre_want, T=300.0, tol=tc.CAT_B, label="ht single train")

    def test_burst_then_recover(self):
        # tight burst depletes the pool; a long gap recovers it before the next burst.
        burst1 = list(np.arange(50.0, 80.0, 5.0))
        burst2 = list(np.arange(900.0, 930.0, 5.0))     # ~2.7 tau_P gap -> near-full recover
        self._run(_SYN, burst1 + burst2, T=1000.0, tol=tc.CAT_B,
                  label="ht burst+recover")

    def test_strong_depletion_train(self):
        # delta_P close to 1: each spike nearly empties the pool.
        syn = dict(_SYN, delta_P=0.9, tau_P=500.0)
        pre_want = list(np.arange(40.0, 400.0, 20.0))
        self._run(syn, pre_want, T=500.0, tol=tc.CAT_B, label="ht strong depletion")

    def test_long_train_trajectory(self):
        pre_want = list(np.arange(50.0, 5000.0, 50.0))
        self._run(_SYN, pre_want, T=5100.0, tol=tc.CAT_A, label="ht 5s train")


if __name__ == "__main__":
    unittest.main()
