# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for the ``clopath_synapse_spike_pairing`` example (§3.3 demo).

The example presents the canonical Clopath spike-pairing protocol on the
``Simulator`` API (10 trains: 5 post-before-pre + 5 pre-before-post, 10-50 Hz). It
reuses the cluster-07 drive :mod:`brainpy_state._nest._validation._clopath_drive`,
so this test asserts the same frozen guarantees the synapse-level
``clopath_synapse_parity_test.py`` proved, now driving the **example** surface:

* every train's final stored weight matches NEST within the documented clopath
  band (``atol 2e-3`` mV, ``rtol 5 %``) -- the online instantaneous-read vs NEST
  deferred-history divergence (LTD near-exact, LTP within 5 %, growing with
  pairing frequency);
* each clearly-directional train has NEST's sign (neutral at the LTD/LTP
  crossover);
* potentiation grows monotonically with pairing frequency.

The weight is a bare mV mantissa (``aeif_psc_delta_clopath`` is a delta neuron),
init ``0.5``.
"""
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _clopath_drive as drv

# Documented Clopath stored-weight band (see _clopath_drive module docstring):
# online instantaneous-read vs NEST deferred-history + one-step event-delivery lag.
_WEIGHT_BAND = tc.TraceTolerance(2e-3, 5e-2, label="clopath",
                                 note="online instantaneous-read vs NEST deferred history")
_PURE_LTD_TOL = tc.TraceTolerance(1e-3, 5e-3, label="clopath-LTD",
                                  note="pure voltage-gated depression path (near-exact)")
# Net |Δw| below this is the LTD/LTP crossover (its strict sign is meaningless).
_CROSSOVER_EPS = 3e-3


@requires_nest
class TestClopathSpikePairingExample(unittest.TestCase):
    """Live-NEST parity for the spike-pairing example's weight curves."""

    @classmethod
    def setUpClass(cls):
        from examples.nest.clopath_synapse_spike_pairing import run, normalized_weight_change
        brainstate.environ.set(dt=drv.DT * u.ms)
        cls.normalized_weight_change = staticmethod(normalized_weight_change)
        cls.rho, cls.post_pre, cls.pre_post, cls.our_w = run()
        cls.nest_w = np.array([drv.nest_pairing_weight(sp, sq)
                               for sp, sq in zip(drv.SPIKE_TIMES_PRE, drv.SPIKE_TIMES_POST)])

    # -- the example's raw weights track NEST within the frozen band -------
    def test_all_trains_within_band(self):
        for i, lab in enumerate(drv.TRAIN_LABELS):
            with self.subTest(train=lab):
                compare_trace(self.nest_w[i], self.our_w[i],
                              tol=_WEIGHT_BAND, metric=f"clopath {lab}").assert_()

    # -- every clearly-directional train has NEST's sign -------------------
    def test_direction_matches_nest(self):
        nest_dw = self.nest_w - drv.INIT_W
        our_dw = self.our_w - drv.INIT_W
        for i, lab in enumerate(drv.TRAIN_LABELS):
            with self.subTest(train=lab):
                if abs(nest_dw[i]) <= _CROSSOVER_EPS:
                    self.assertLessEqual(
                        abs(our_dw[i]), _CROSSOVER_EPS,
                        f"{lab}: NEST neutral (Δw={nest_dw[i]:.2e}) but ours Δw={our_dw[i]:.2e}")
                else:
                    self.assertEqual(
                        int(np.sign(our_dw[i])), int(np.sign(nest_dw[i])),
                        f"{lab}: net Δw sign must match NEST")

    # -- potentiation grows monotonically with pairing frequency -----------
    def test_potentiation_monotonic_in_frequency(self):
        our_dw = self.our_w[list(drv.LTP_TRAINS)] - drv.INIT_W
        self.assertTrue(np.all(np.diff(our_dw) > 0), f"LTP not monotone: {our_dw}")

    # -- the lowest-frequency depression train is near-exact ---------------
    def test_pure_ltd_near_exact(self):
        i = drv.LTD_TRAINS[0]
        self.assertLess(self.our_w[i], drv.INIT_W, "pure-LTD train must depress")
        compare_trace(self.nest_w[i], self.our_w[i],
                      tol=_PURE_LTD_TOL, metric="clopath pure-LTD").assert_()

    # -- the example's normalisation maps the no-change baseline to 100 % --
    def test_normalization_baseline(self):
        self.assertAlmostEqual(float(self.normalized_weight_change(drv.INIT_W)), 100.0, places=6)


if __name__ == "__main__":
    unittest.main()
