# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Validation for ``examples/nest/hh_phaseplane.py``.

This demo is a phase-plane **analysis** carve-out, not a spike-train parity demo,
so it is validated in two complementary layers:

* **NEST-free self-consistency + analytic oracle.** The reduced ``(V, n)`` vector
  field must be finite everywhere; the extracted ``n``-nullcline (``dn/dt = 0``)
  must coincide with the closed-form ``n_inf(V) = α_n/(α_n+β_n)`` to within one
  grid step; every extracted ``V``-nullcline point must bracket a genuine
  ``dV/dt`` sign change; and the reduced-plane trajectory must relax to the
  resting fixed point (with ``m`` frozen the Na upstroke is disabled, so it cannot
  fire) — the fixed point landing on the ``n``-nullcline.
* **Live-NEST parity** (``@requires_nest``). The reduced-plane trajectory must
  reproduce NEST's one-step-``Simulate``-with-``m``/``h``-clamp orbit
  ``(V(t), n(t))`` step-for-step over 1000 steps (category A; ~4e-3 mV of
  accumulated RKF45-stream drift, well within tolerance).

The ``n_inf`` oracle is recomputed from the textbook HH rate functions here,
independently of the model's RHS code path, so the nullcline check is not
circular.
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

import brainunit as u

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import TraceTolerance

from examples.nest.hh_phaseplane import (
    vector_field, nullclines, ap_trajectory, DELTA_N)

# Category A, no recorder-offset alignment (states are read directly; both the
# brainpy and NEST orbits start exactly at the seed). The V trajectory accrues
# ~4e-3 mV of drift over 1000 independent RKF45 steps, so atol is set to 5e-3 mV.
CAT_A_TRAJ_V = TraceTolerance(5e-3 * u.mV, 1e-3, align_steps=0, label="A",
                              note="HH reduced-plane V(t), 1000-step RKF45 drift")
CAT_A_TRAJ_N = TraceTolerance(1e-3, 1e-3, align_steps=0, label="A",
                              note="HH reduced-plane n(t)")


def _n_inf(V):
    """Closed-form K-activation steady state ``n_inf(V)`` (textbook HH rates)."""
    a = (0.01 * (V + 55.0)) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    b = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return a / (a + b)


def _nest_trajectory(amplitude=100.0, dt=0.1, n_steps=1000, V0=-34.0, n0=0.2):
    """NEST reduced-plane orbit (``hh_phaseplane.py`` ``ap`` loop, verbatim)."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    neu = nest.Create('hh_psc_alpha')
    nest.Simulate(1000.0)                       # relax for equilibrium m, h
    m_eq, h_eq = neu.Act_m, neu.Inact_h
    neu.I_e = amplitude
    neu.set(V_m=V0, Act_n=n0, Act_m=m_eq, Inact_h=h_eq)
    ap = np.zeros((n_steps, 2))
    for i in range(n_steps):
        ap[i] = [neu.V_m, neu.Act_n]
        neu.set(Act_m=m_eq, Inact_h=h_eq)
        nest.Simulate(dt)
    return ap


class TestPhasePlaneSelfConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.V_vec, cls.n_vec, cls.dVdt, cls.dndt = vector_field()
        cls.nc_V, cls.nc_n = nullclines(cls.V_vec, cls.n_vec, cls.dVdt, cls.dndt)

    def test_vector_field_finite(self):
        # Regression guard: a one-step-integration probe overflows to NaN at the
        # extreme grid corners; the analytic RHS must stay finite everywhere.
        self.assertTrue(np.isfinite(self.dVdt).all())
        self.assertTrue(np.isfinite(self.dndt).all())

    def test_n_nullcline_matches_analytic_n_inf(self):
        # dn/dt = 0  <=>  n = n_inf(V), independent of m, h, I_e. The grid-extracted
        # nullcline must match the closed form to within one grid step.
        self.assertGreater(len(self.nc_n), 10)
        err = max(abs(n - _n_inf(V)) for V, n in self.nc_n)
        self.assertLessEqual(err, DELTA_N)

    def test_V_nullcline_brackets_sign_change(self):
        # dV/dt is monotonic in n, so an interior |dV/dt| minimum must bracket a
        # true zero crossing: dV/dt at the n-neighbours has opposite signs.
        self.assertGreater(len(self.nc_V), 5)
        for V, n in self.nc_V:
            i = int(np.where(self.V_vec == V)[0][0])
            j = int(np.where(np.isclose(self.n_vec, n))[0][0])
            self.assertNotEqual(
                np.sign(self.dVdt[j - 1, i]), np.sign(self.dVdt[j + 1, i]),
                msg=f"no dV/dt sign change across n at V={V}, n={n}")

    def test_trajectory_relaxes_to_fixed_point(self):
        ap = ap_trajectory(n_steps=1000)
        self.assertTrue(np.isfinite(ap).all())
        self.assertLess(ap[:, 0].max(), 0.0)               # never reaches 0 mV: no AP
        self.assertTrue(-70.0 < ap[-1, 0] < -55.0)          # settles sub-threshold
        self.assertLess(abs(ap[-1, 0] - ap[-50, 0]), 1e-2)  # converged
        # the fixed point sits on the n-nullcline (dn/dt = 0  =>  n = n_inf(V))
        self.assertLess(abs(ap[-1, 1] - _n_inf(ap[-1, 0])), 1e-3)


@requires_nest
class TestPhasePlaneNestParity(unittest.TestCase):
    def test_trajectory_matches_nest(self):
        ap_bp = ap_trajectory(n_steps=1000)
        ap_ns = _nest_trajectory(n_steps=1000)
        compare_trace(ap_ns[:, 0], ap_bp[:, 0], tol=CAT_A_TRAJ_V,
                      metric='hh_phaseplane V(t)').assert_()
        compare_trace(ap_ns[:, 1], ap_bp[:, 1], tol=CAT_A_TRAJ_N,
                      metric='hh_phaseplane n(t)').assert_()


if __name__ == '__main__':
    unittest.main()
