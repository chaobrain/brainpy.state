# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""``siegert_neuron`` 15a-owned paths: eager ``update`` + the Siegert Φ integral.

``siegert_neuron`` is the one rate family whose step does **not** lower under
``brainstate.transform.for_loop`` — its transfer function is a host-side
special-function integral (SciPy / Gauss-Legendre on concrete values), so the model
documents an eager host loop as the supported driver (CLAUDE.md #10 exception). Its
network *diffusion* routing is deferred to goal 15c; in 15a the drift/variance arrive
through ``update``'s direct ``drift_input`` / ``diffusion_input`` arguments.

This pins the 15a-reachable code the existing ``siegert_neuron_test`` (which only calls
``siegert_rate`` in the supra-threshold / noise-free regime) never touches:

* ``update(drift_input=μ, diffusion_input=σ²)`` driven eagerly relaxes the rate to
  ``mean + siegert_rate(μ, σ²)`` (NEST non-WFR exponential-Euler step);
* the fluctuation-driven regime of ``siegert_rate`` (finite σ²), which routes through the
  ``erfcx`` / ``dawson`` integral that the noise-free limit skips;
* the no-SciPy Gauss-Legendre / asymptotic fallback — forced by patching ``_HAVE_SCIPY``
  — must agree with the SciPy quadrature it stands in for.

Not covered here (by design): network diffusion delivery (goal 15c).
"""
import unittest
from unittest import mock

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import brainunit as u

import sys
import brainpy_state._nest_neuron.siegert_neuron  # ensure the submodule is imported
# The package re-exports the class under the same name, shadowing the submodule
# attribute; reach the real module (which owns the ``_HAVE_SCIPY`` global) via sys.modules.
siegert_mod = sys.modules['brainpy_state._nest_neuron.siegert_neuron']
from brainpy_state import siegert_neuron

# NEST reference operating point (cf. test_siegert_neuron.py): mu at threshold.
MU_REF = 15.0
SIGMA2_REF = 0.1 * MU_REF  # = 1.5
RATE_REF = 27.1095934379


def _nrn():
    return siegert_neuron(1, tau=1.0 * u.ms, tau_m=10.0 * u.ms, t_ref=2.0 * u.ms,
                          theta=15.0, V_reset=0.0)


class TestSiegertUpdate(unittest.TestCase):
    """The eager ``update`` step relaxes the rate to ``mean + siegert_rate``."""

    def test_update_converges_to_siegert_rate(self):
        nrn = _nrn()
        brainstate.nn.init_all_states(nrn)
        target = float(np.asarray(
            nrn.siegert_rate(np.asarray([MU_REF]), np.asarray([SIGMA2_REF]))).reshape(-1)[0])
        # Eager host loop (siegert does not lower under for_loop -- documented exception).
        with brainstate.environ.context(dt=0.1 * u.ms):
            for _ in range(400):  # ~40 tau at tau=1 ms -> fully relaxed
                r = nrn.update(drift_input=MU_REF, diffusion_input=SIGMA2_REF)
        self.assertAlmostEqual(float(np.asarray(r).reshape(-1)[0]), target, places=4)
        self.assertAlmostEqual(target, RATE_REF, delta=1e-5)

    def test_zero_drive_stays_at_zero(self):
        """No drift, no diffusion -> the Siegert rate is zero and the state stays put."""
        nrn = _nrn()
        brainstate.nn.init_all_states(nrn)
        with brainstate.environ.context(dt=0.1 * u.ms):
            for _ in range(50):
                r = nrn.update(drift_input=0.0, diffusion_input=0.0)
        self.assertAlmostEqual(float(np.asarray(r).reshape(-1)[0]), 0.0, places=8)


class TestSiegertFluctuationRegime(unittest.TestCase):
    """``siegert_rate`` over the noise-driven regime that uses the erfcx integral."""

    def test_finite_nonnegative_and_monotone_in_mu(self):
        nrn = _nrn()
        sigma2 = 9.0  # finite noise -> fluctuation-driven regime
        mus = np.array([5.0, 10.0, 15.0, 20.0])
        rates = np.asarray(
            nrn.siegert_rate(mus, np.full_like(mus, sigma2))).reshape(-1)
        self.assertTrue(np.all(np.isfinite(rates)))
        self.assertTrue(np.all(rates >= 0.0))
        # More mean drive -> higher firing rate.
        self.assertTrue(np.all(np.diff(rates) > 0.0), msg=f'not monotone: {rates}')


class TestSiegertScipyFallbackParity(unittest.TestCase):
    """The no-SciPy Gauss-Legendre / asymptotic fallback matches SciPy quadrature."""

    def test_fallback_matches_scipy(self):
        if not siegert_mod._HAVE_SCIPY:
            self.skipTest('SciPy not installed; fallback is the only path.')
        nrn = _nrn()
        mus = np.array([8.0, 12.0, 15.0, 18.0])
        sigma2 = np.full_like(mus, 4.0)

        ref = np.asarray(nrn.siegert_rate(mus, sigma2)).reshape(-1)
        # Force the SciPy-free quadrature fallback and re-evaluate.
        with mock.patch.object(siegert_mod, '_HAVE_SCIPY', False):
            alt = np.asarray(nrn.siegert_rate(mus, sigma2)).reshape(-1)

        self.assertTrue(np.all(np.isfinite(alt)))
        # The fallback is Gauss-Legendre + an asymptotic expansion; it tracks the SciPy
        # quadrature to ~1e-3 relative (the asymptotic split dominates supra-threshold).
        np.testing.assert_allclose(alt, ref, rtol=2e-3, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
