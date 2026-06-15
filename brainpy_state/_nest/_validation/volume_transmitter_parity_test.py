# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``volume_transmitter`` dopamine concentration ``n(t)`` (cluster-08).

The ``volume_transmitter`` is the **broadcast source** of the dopamine-modulated
STDP seam, so its concentration ``n(t)`` is validated *before* the
``stdp_dopamine_synapse`` weight trajectory that consumes it (the upstream-first
discipline of the Clopath neuron-voltage precondition).

``n`` is the pure ``update_dopamine_`` recursion ``n <- n e^{-dt/tau_n} +
count/tau_n`` (``stdp_dopamine_synapse.h:419-425``): on a dopa-arrival step it
jumps by ``1/tau_n`` (no decay applied that step), and decays by ``e^{-dt/tau_n}``
every step otherwise. NEST integrates it on each synapse from the transmitter's
relayed train; we moved it onto the broadcast node — but it is the *same* scalar
recursion, so the two agree to **machine precision** (and cross-check against a
closed-form reference). The only modelling choice is timing: a dopa
``spike_generator`` spike relayed ``sg -> parrot -> vt`` makes ``n`` jump
``DOPA_RELAY`` later; the drive offsets each side so both jump at the intended
VT-arrival step (see :mod:`brainpy_state._nest._validation._stdp_dopamine_drive`).
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _stdp_dopamine_drive as drv

# n(t) is the pure update_dopamine_ recursion on both sides — a near-exact anchor
# (observed max|Δn| ~ 5e-16 vs NEST, exactly 0 vs the closed form). The band is set
# well above float64 noise but far below any modelling error.
_N_TOL = tc.TraceTolerance(1e-9, 1e-9, label="vt-n",
                           note="pure update_dopamine_ recursion (machine-precision)")
_TAU_N = 200.0
_T = 320.0
# Irregular dopa VT-arrival times: exercise the jump, multi-spike accumulation,
# and long inter-spike decay (the 140->260 gap lets n relax well toward 0).
_DOPA = [10.0, 35.0, 60.0, 95.0, 140.0, 260.0]


@requires_nest
class TestVolumeTransmitterParity(unittest.TestCase):
    """Live-NEST parity for the rebuilt broadcast ``volume_transmitter`` ``n(t)``."""

    @classmethod
    def setUpClass(cls):
        brainstate.environ.set(dt=drv.DT * u.ms)
        cls.t_a, cls.n_a = drv.analytic_vt_n_trace(_DOPA, _T, _TAU_N, sample_dt=1.0)
        cls.t_o, cls.n_o = drv.our_vt_n_trace(_DOPA, _T, _TAU_N, sample_dt=1.0)
        cls.t_n, cls.n_n = drv.nest_vt_n_trace(_DOPA, _T, _TAU_N, sample_dt=1.0)
        cls.m = min(len(cls.n_a), len(cls.n_o), len(cls.n_n))

    # -- sampling alignment ------------------------------------------------
    def test_sample_times_aligned(self):
        # all three traces share the same sample grid (so per-sample compare is valid)
        m = self.m
        np.testing.assert_allclose(self.t_o[:m], self.t_n[:m])
        np.testing.assert_allclose(self.t_o[:m], self.t_a[:m])

    # -- 1. broadcast n(t) matches NEST sample-for-sample (near-exact) ------
    def test_concentration_matches_nest(self):
        m = self.m
        compare_trace(self.n_n[:m], self.n_o[:m], tol=_N_TOL, metric="vt n(t)").assert_()

    # -- 2. broadcast n(t) reproduces the closed-form recursion exactly -----
    def test_concentration_matches_analytic(self):
        m = self.m
        # our recursion *is* the closed form -> bit-for-bit (no float reordering)
        self.assertEqual(float(np.max(np.abs(self.n_o[:m] - self.n_a[:m]))), 0.0)

    # -- 3. NEST itself reproduces the closed form (sanity on the reference) -
    def test_nest_matches_analytic(self):
        m = self.m
        compare_trace(self.n_a[:m], self.n_n[:m], tol=_N_TOL, metric="nest n(t)").assert_()

    # -- 4. the jump on the first dopa arrival is exactly 1/tau_n -----------
    def test_jump_value_is_inverse_tau_n(self):
        # first arrival D=10 ms -> n(11 ms) sample is one decay step past the jump;
        # the jump itself (n at 10.0 ms) equals 1/tau_n on both sides.
        ta, na = drv.analytic_vt_n_trace([10.0], 20.0, _TAU_N, sample_dt=0.1)
        to, no = drv.our_vt_n_trace([10.0], 20.0, _TAU_N, sample_dt=0.1)
        k = int(np.argmin(np.abs(ta - 10.0)))
        self.assertAlmostEqual(float(na[k]), 1.0 / _TAU_N, places=12)
        self.assertAlmostEqual(float(no[k]), 1.0 / _TAU_N, places=12)

    # -- 5. n decays monotonically between arrivals (and never goes negative)
    def test_decays_between_arrivals(self):
        m = self.m
        # after the last arrival (260 ms) n only decays -> strictly non-increasing tail
        tail = self.n_o[self.t_o[:m] > 270.0]
        self.assertTrue(np.all(np.diff(tail) <= 1e-15), "n must decay after the last dopa spike")
        self.assertTrue(np.all(self.n_o[:m] >= 0.0), "n must stay non-negative")
        self.assertGreater(float(np.max(self.n_o[:m])), 0.0, "n must rise on dopa arrivals")


if __name__ == "__main__":
    unittest.main()
