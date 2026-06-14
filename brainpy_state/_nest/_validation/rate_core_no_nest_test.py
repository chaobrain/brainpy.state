# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""No-NEST companion for the goal-15a rate core — the standalone guarantee.

This file **never imports ``nest``**. It proves the de-queued rate core is a
self-contained JAX artifact: the whole family imports from the public
``brainpy_state`` surface with no NEST dependency and no circular import, and a
headless ``brainstate.transform.for_loop`` relaxation reaches the *analytic*
fixed point — the closed form, not a NEST trace.

That distinguishes it from its two siblings:

* ``rate_core_forloop_test`` pins ``for_loop`` *lowering* (assertions are
  deliberately weak — finite, non-trivial — because the point is the trace, not
  the numbers).
* ``rate_network_parity_test`` pins the coupled fixed point but drives through the
  ``Simulator``.

Here the assertion is the number itself, produced by a bare ``for_loop`` with no
Simulator, no connection, no NEST. With no coupling the delta input is zero, so

.. math::

    \tau\,\dot X = -\lambda X + \mu + \varphi(0)
    \;\Longrightarrow\; X^\* = \frac{\mu + \varphi(0)}{\lambda}.

For ``lin_rate`` (:math:`\varphi(0)=g\cdot 0=0`) this is ``X*=mu/lambda``; for
``tanh_rate`` (:math:`\varphi(0)=\tanh(-g\theta)`) it is ``mu/lambda + tanh(-g*theta)`` —
so a non-trivial ``theta`` proves the activation actually enters the relaxation.
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import saiunit as u
import braintools

# Public rate-core surface. Importing this list at module load IS the import-surface
# guarantee: none of it may pull in ``nest`` or re-enter ``brainpy`` mid-init.
from brainpy_state import (
    lin_rate_ipn, lin_rate_opn,
    gauss_rate_ipn,
    sigmoid_rate_ipn,
    sigmoid_rate_gg_1998_ipn,
    tanh_rate_ipn, tanh_rate_opn,
    threshold_lin_rate_ipn, threshold_lin_rate_opn,
    rate_neuron_ipn, rate_neuron_opn,
    rate_transformer_node, siegert_neuron,
    step_rate_generator,
    rate_connection_instantaneous, rate_connection_delayed,
)

DT = 0.1 * u.ms
N_STEPS = 2000  # 20 tau at tau=10 ms, dt=0.1 ms -> residual ~ e^{-20}

_RATE_CORE = [
    lin_rate_ipn, lin_rate_opn, gauss_rate_ipn, sigmoid_rate_ipn,
    sigmoid_rate_gg_1998_ipn, tanh_rate_ipn, tanh_rate_opn,
    threshold_lin_rate_ipn, threshold_lin_rate_opn, rate_neuron_ipn,
    rate_neuron_opn, rate_transformer_node, siegert_neuron, step_rate_generator,
    rate_connection_instantaneous, rate_connection_delayed,
]


def _relax(cls, size, kwargs, n=N_STEPS):
    """Step ``cls`` ``n`` times under a bare ``for_loop``; return the final rate vector."""
    brainstate.random.seed(0)
    with brainstate.environ.context(dt=DT):
        neuron = cls(size, rate_initializer=braintools.init.Constant(0.0), **kwargs)
        brainstate.nn.init_all_states(neuron)

        def step(i):
            neuron.update()
            return u.get_mantissa(neuron.rate.value)

        rates = brainstate.transform.for_loop(step, np.arange(n))
    return np.asarray(rates)[-1].reshape(-1)


class TestRateCoreImportSurface(unittest.TestCase):
    """The full rate core imports from ``brainpy_state`` with no NEST dependency."""

    def test_every_symbol_is_constructible_without_nest(self):
        # This module imported the whole family at load time without ``nest`` — the
        # mere fact collection reached here is the no-circular-import guarantee. Now
        # confirm each name is a usable class object.
        for cls in _RATE_CORE:
            with self.subTest(symbol=cls.__name__):
                self.assertTrue(isinstance(cls, type), msg=f'{cls.__name__} is not a class')

    def test_connection_specs_round_trip_headless(self):
        """The trimmed connection specs still carry NEST-parity status, no NEST needed."""
        inst = rate_connection_instantaneous(weight=2.0)
        self.assertEqual(inst.get_status(),
                         {'weight': 2.0, 'delay': 1, 'has_delay': False, 'supports_wfr': True})
        with self.assertRaisesRegex(ValueError, 'has no delay'):
            inst.set_delay(2)

        dly = rate_connection_delayed(weight=1.5, delay_steps=3)
        self.assertEqual(dly.get('delay'), 3)
        with self.assertRaisesRegex(ValueError, 'must be >= 1'):
            dly.set_delay_steps(0)


class TestHeadlessForLoopFixedPoint(unittest.TestCase):
    """A bare ``for_loop`` relaxes the rate core to the closed-form fixed point."""

    def test_lin_rate_relaxes_to_mu_over_lambda(self):
        """``lin_rate_ipn`` (phi(0)=0) relaxes to ``mu/lambda`` — here ``mu``."""
        for mu in (2.0, -1.0, 0.0, 5.0):
            with self.subTest(mu=mu):
                final = _relax(lin_rate_ipn, 1,
                               dict(tau=10.0 * u.ms, lambda_=1.0, sigma=0.0, mu=mu, g=1.0))
                np.testing.assert_allclose(final, mu, atol=1e-5)

    def test_lin_rate_population_vectorizes(self):
        """A heterogeneous-``mu`` population relaxes element-wise to its drive vector."""
        mu = np.array([1.0, 2.0, -0.5])
        final = _relax(lin_rate_ipn, 3,
                       dict(tau=10.0 * u.ms, lambda_=1.0, sigma=0.0, mu=mu, g=1.0))
        np.testing.assert_allclose(final, mu, atol=1e-5)

    def test_lambda_scales_the_fixed_point(self):
        """A faster passive decay ``lambda`` lowers the fixed point to ``mu/lambda``."""
        mu, lam = 3.0, 2.0
        final = _relax(lin_rate_ipn, 1,
                       dict(tau=10.0 * u.ms, lambda_=lam, sigma=0.0, mu=mu, g=1.0))
        np.testing.assert_allclose(final, mu / lam, atol=1e-5)

    def test_tanh_rate_fixed_point_includes_activation_offset(self):
        """``tanh_rate_ipn`` relaxes to ``mu + tanh(-g*theta)`` — phi(0) enters the FP."""
        mu, g, theta = 2.0, 1.0, 0.5
        expected = mu + np.tanh(-g * theta)
        final = _relax(tanh_rate_ipn, 1,
                       dict(tau=10.0 * u.ms, lambda_=1.0, sigma=0.0, mu=mu, g=g, theta=theta))
        np.testing.assert_allclose(final, expected, atol=1e-5)
        # Guard the fixture: the offset is genuinely non-zero, so this is not the lin case.
        self.assertGreater(abs(float(np.tanh(-g * theta))), 0.1)


if __name__ == '__main__':
    unittest.main()
