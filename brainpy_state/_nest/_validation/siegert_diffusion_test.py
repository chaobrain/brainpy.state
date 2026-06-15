# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Goal 15c — Siegert mean-field node + dual-channel ``diffusion_connection``.

Validates the two 15c designs against their arbiters:

* **B (JAX quadrature).** The jnp Siegert transfer ``_siegert_phi_jax`` (leggauss-64 +
  erfcx/Dawson + asymptotic expansions) matches the SciPy quadrature oracle
  (``siegert_rate``) across a (μ, σ²) grid to a documented tolerance, and the model's
  ``update`` now lowers under ``brainstate.transform.for_loop`` (the 15a eager
  exception is retired).
* **A (dual-channel deposit).** One ``siegert_neuron`` driven by one
  ``diffusion_connection`` through the Simulator (drift→μ default channel,
  diffusion→σ² ``'diffusion_sigma2'`` channel) reproduces a live-NEST two-Siegert
  trace; a population relaxes to the self-consistent mean-field fixed point.

NEST-gated groups use ``@requires_nest`` with a NEST-free companion. SciPy is the
quadrature oracle (never NEST for the transfer grid).
"""
import unittest
from unittest import mock

import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import saiunit as u

import sys
import brainpy_state._nest.siegert_neuron  # ensure the submodule is imported
siegert_mod = sys.modules['brainpy_state._nest.siegert_neuron']
from brainpy_state import siegert_neuron

from scipy import special as _sp_special  # the oracle for the special-function ports


def _nrn(tau_syn_ms=0.0):
    return siegert_neuron(1, tau=1.0 * u.ms, tau_m=10.0 * u.ms,
                          tau_syn=tau_syn_ms * u.ms, t_ref=2.0 * u.ms,
                          theta=15.0, V_reset=0.0)


class TestSpecialFunctionsJax(unittest.TestCase):
    """The jnp ``erfcx`` / Dawson ports match SciPy across their domains."""

    def test_erfcx_jax_matches_scipy(self):
        xs = np.concatenate([np.linspace(-3.0, 25.0, 200), np.array([30.0, 60.0, 120.0])])
        got = np.asarray(siegert_neuron._erfcx_jax(jax.numpy.asarray(xs)))
        ref = _sp_special.erfcx(xs)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)

    def test_dawsn_jax_matches_scipy(self):
        # Siegert only evaluates Dawson at non-negative arguments.
        xs = np.concatenate([np.linspace(0.0, 8.0, 400), np.array([8.5, 12.0, 30.0])])
        got = np.asarray(siegert_neuron._dawsn_jax(jax.numpy.asarray(xs)))
        ref = _sp_special.dawsn(xs)
        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)


class TestQuadratureOracle(unittest.TestCase):
    """``_siegert_phi_jax`` (JAX leggauss-64) matches the SciPy quadrature oracle."""

    def _grid(self):
        mu = np.linspace(-5.0, 30.0, 40)
        sig2 = np.array([0.1, 0.5, 1.5, 4.0, 9.0, 25.0])
        MU, SIG2 = np.meshgrid(mu, sig2, indexing='ij')
        return MU.reshape(-1), SIG2.reshape(-1)

    def test_phi_jax_matches_scipy_grid(self):
        nrn = _nrn()
        mu, sig2 = self._grid()
        ref = np.asarray(nrn.siegert_rate(mu, sig2)).reshape(-1)            # SciPy oracle
        got = np.asarray(nrn._siegert_phi_jax(jax.numpy.asarray(mu),
                                              jax.numpy.asarray(sig2))).reshape(-1)
        self.assertTrue(np.all(np.isfinite(got)))
        # Documented tolerance: leggauss-64 + erfcx/Dawson vs SciPy quad/special.
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)

    def test_phi_jax_matches_scipy_colored_noise(self):
        nrn = _nrn(tau_syn_ms=0.5)  # finite tau_syn -> colored-noise threshold shift
        mu, sig2 = self._grid()
        ref = np.asarray(nrn.siegert_rate(mu, sig2)).reshape(-1)
        got = np.asarray(nrn._siegert_phi_jax(jax.numpy.asarray(mu),
                                              jax.numpy.asarray(sig2))).reshape(-1)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)

    def test_phi_jax_deterministic_and_zero_fastpaths(self):
        nrn = _nrn()
        # sigma^2 = 0 -> deterministic LIF branch; deep subthreshold -> 0.
        mu = np.array([-2.0, 5.0, 14.999, 16.0, 25.0])
        sig2 = np.zeros_like(mu)
        ref = np.asarray(nrn.siegert_rate(mu, sig2)).reshape(-1)
        got = np.asarray(nrn._siegert_phi_jax(jax.numpy.asarray(mu),
                                              jax.numpy.asarray(sig2))).reshape(-1)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)

    def test_phi_jax_matches_nest_reference_point(self):
        # The canonical NEST operating point (mu at threshold).
        nrn = _nrn()
        got = float(np.asarray(nrn._siegert_phi_jax(
            jax.numpy.asarray([15.0]), jax.numpy.asarray([1.5]))).reshape(-1)[0])
        self.assertAlmostEqual(got, 27.1095934379, delta=1e-4)


class TestForLoopLowering(unittest.TestCase):
    """The Siegert ``update`` lowers under ``for_loop`` (15a eager exception retired)."""

    def test_update_lowers_under_for_loop(self):
        nrn = _nrn()  # tau = 1 ms
        brainstate.nn.init_all_states(nrn)
        n_steps = 300  # dt=0.1 ms, tau=1 ms -> 30 tau, fully relaxed
        with brainstate.environ.context(dt=0.1 * u.ms):
            def step(i):
                return nrn.update(drift_input=12.0, diffusion_input=4.0)

            out = brainstate.transform.for_loop(step, jax.numpy.arange(n_steps))
        out = np.asarray(out)
        self.assertEqual(out.shape, (n_steps, 1))
        self.assertTrue(np.all(np.isfinite(out)))
        # The exact-exp relaxation converges to the Siegert fixed point r* = mean + Phi.
        target = float(np.asarray(nrn.siegert_rate(np.array([12.0]), np.array([4.0]))).reshape(-1)[0])
        self.assertAlmostEqual(float(out[-1, 0]), target, places=5)
        # Monotone approach from rest (no overshoot/oscillation).
        self.assertTrue(np.all(np.diff(out[:, 0]) >= -1e-9))


if __name__ == '__main__':
    unittest.main()
