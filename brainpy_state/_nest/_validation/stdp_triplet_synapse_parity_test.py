# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``stdp_triplet_synapse`` (Pfister-Gerstner triplet) trajectories.

The first model exercised on the substrate's **multi-trace seam**: a fast/slow
trace pair on each side (``pre_trace_tau = (tau_plus, tau_plus_triplet)``,
``post_trace_tau = (tau_minus, tau_minus_triplet)``). Potentiation reads the fast
pre trace weighted by the slow post trace; depression reads the fast post trace
weighted by the slow pre trace. The post node carries **both** ``tau_minus`` and
``tau_minus_triplet`` (set via ``post_params``). High-frequency trains let the
slow traces accumulate, so the train tests are the real check that the triplet
terms (not just the pair terms) match NEST. Drive shared with the canonical pair
test (:mod:`brainpy_state._nest._validation._stdp_drive`).
"""
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import stdp_triplet_synapse
from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _stdp_drive as drv

# A2/A3 bumped well above the 5e-10 defaults so single pairs move past 1e-6.
_SYN = dict(weight=5.0, Wmax=100.0,
            tau_plus=16.8, tau_plus_triplet=101.0, tau_minus=20.0, tau_minus_triplet=110.0,
            Aplus=0.005, Aplus_triplet=0.005, Aminus=0.005, Aminus_triplet=0.005)


def _rule(syn):
    return stdp_triplet_synapse(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau_plus=syn['tau_plus'] * u.ms, tau_plus_triplet=syn['tau_plus_triplet'] * u.ms,
        tau_minus=syn['tau_minus'] * u.ms, tau_minus_triplet=syn['tau_minus_triplet'] * u.ms,
        Aplus=syn['Aplus'], Aplus_triplet=syn['Aplus_triplet'],
        Aminus=syn['Aminus'], Aminus_triplet=syn['Aminus_triplet'], Wmax=syn['Wmax'])


def _per_conn(syn):
    # all triplet plasticity params are per-connection (not *_hom / common-only).
    return {"weight": syn['weight'], "tau_plus": syn['tau_plus'],
            "tau_plus_triplet": syn['tau_plus_triplet'], "Aplus": syn['Aplus'],
            "Aplus_triplet": syn['Aplus_triplet'], "Aminus": syn['Aminus'],
            "Aminus_triplet": syn['Aminus_triplet'], "Wmax": syn['Wmax']}


@requires_nest
class TestStdpTripletSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, post_want, T, tol, label):
        # post node holds both K- constants (tau_minus + tau_minus_triplet).
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_triplet_synapse", _per_conn(syn), syn['tau_minus'],
            pre_want, post_want, T,
            post_params={"tau_minus_triplet": syn['tau_minus_triplet']})
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
                          tol=tc.CAT_B, label=f"triplet pair dt={dt:+.0f}")

    def test_simultaneous_pair_matches_nest(self):
        # coincident pre&post: exclusion makes the pair term vanish; triplet exact.
        self._run(_SYN, [100.0, 320.0], [100.0], T=420.0,
                  tol=tc.CAT_B, label="triplet coincident")

    def test_high_frequency_train_exercises_triplet(self):
        # 50 Hz pre/post pairing: slow traces r2/o2 accumulate, so the triplet
        # terms dominate -> a strong test that they match NEST, not just the pair.
        pre_want = list(np.arange(100.0, 1100.0, 20.0))
        post_want = [p + 5.0 for p in pre_want]
        self._run(_SYN, pre_want, post_want, T=1200.0,
                  tol=tc.CAT_A, label="triplet 50Hz train")

    def test_fixed_train_trajectory(self):
        pre_want = list(np.arange(50.0, 5000.0, 50.0))
        post_want = [p + 10.0 for p in pre_want]
        self._run(_SYN, pre_want, post_want, T=5100.0,
                  tol=tc.CAT_A, label="triplet 5s train")

    def test_wmax_saturation_matches_nest(self):
        # strong potentiation drives the weight into the Wmax ceiling.
        syn = dict(_SYN, weight=95.0, Aplus=0.5, Aplus_triplet=0.5)
        pre_want = list(np.arange(50.0, 1500.0, 30.0))
        post_want = [p + 5.0 for p in pre_want]
        self._run(syn, pre_want, post_want, T=1600.0,
                  tol=tc.CAT_A, label="triplet Wmax saturation")


if __name__ == "__main__":
    unittest.main()
