# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Edge-case coverage for the goal-15a rate core (NEST-free).

The parity / fixed-point suites exercise the *critical path* (relaxation to the
analytic / NEST steady state). This file pins the **branches around it** that
those happy-path runs never reach:

* connection spec accessors — ``get(key)`` dispatch, ``KeyError`` on a bad key, and
  the scalar/``Quantity`` validation in ``_to_float_scalar``;
* per-φ ``tau <= 0`` parameter validation (raised at construction);
* the exponential-Euler propagator's ``lambda == 0`` special cases (the all-zero
  ``else`` branch and the mixed-population ``where`` branch) — these guard against a
  divide-by-zero that the ``lambda > 0`` happy path never sees;
* the multiplicative-coupling factors ``(H_ex, H_in)`` — real for ``lin_rate`` /
  ``rate_neuron``, identically one for the fixed-nonlinearity models;
* output rectification (``rectify_output``);
* the ``rate_neuron`` template's user ``input_nonlinearity`` dispatch (``fn(self, x)``
  then ``fn(x)``) and its ``_phi_signature``.
"""
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import saiunit as u
import braintools

from brainpy_state import (
    lin_rate_ipn, gauss_rate_ipn, sigmoid_rate_ipn, sigmoid_rate_gg_1998_ipn,
    tanh_rate_ipn, threshold_lin_rate_ipn, rate_neuron_ipn, rate_neuron_opn,
    rate_connection_instantaneous, rate_connection_delayed,
)

DT = 0.1 * u.ms

#: (label, class, base kwargs that give a finite step). gauss needs sigma > 0
#: (its gain width doubles as the noise scale); the others stay deterministic.
_FAMILIES = [
    ('lin_rate_ipn', lin_rate_ipn, dict(sigma=0.0, mu=1.0, g=1.0)),
    ('gauss_rate_ipn', gauss_rate_ipn, dict(sigma=1.0, mu=1.0, g=1.0)),
    ('sigmoid_rate_ipn', sigmoid_rate_ipn, dict(sigma=0.0, mu=1.0, g=1.0)),
    ('sigmoid_rate_gg_1998_ipn', sigmoid_rate_gg_1998_ipn, dict(sigma=0.0, mu=1.0, g=1.0)),
    ('tanh_rate_ipn', tanh_rate_ipn, dict(sigma=0.0, mu=1.0, g=1.0)),
    ('threshold_lin_rate_ipn', threshold_lin_rate_ipn, dict(sigma=0.0, mu=1.0, g=1.0)),
    ('rate_neuron_ipn', rate_neuron_ipn, dict(sigma=0.0, mu=1.0, g=1.0)),
    ('rate_neuron_opn', rate_neuron_opn, dict(sigma=0.0, mu=1.0, g=1.0)),
]

#: The ``*_opn`` output-noise template has no passive-decay ``lambda_`` and no
#: ``rectify_output`` (it integrates ``tau X' = -X + mu + phi(h) + sigma*xi``), so it is
#: excluded from the propagator branches that those parameters gate.
_LAMBDA_FAMILIES = [f for f in _FAMILIES if f[0] != 'rate_neuron_opn']


def _step(cls, size, kw, n=3):
    """Step ``cls`` ``n`` times under ``for_loop``; return the (n, size) rate trace."""
    brainstate.random.seed(0)
    with brainstate.environ.context(dt=DT):
        node = cls(size, rate_initializer=braintools.init.Constant(0.0), **kw)
        brainstate.nn.init_all_states(node)

        def step(i):
            node.update()
            return u.get_mantissa(node.rate.value)

        out = brainstate.transform.for_loop(step, np.arange(n))
    return np.asarray(out)


class TestConnectionSpecAccessors(unittest.TestCase):
    """The trimmed connection specs round-trip their NEST-parity status surface."""

    def test_instantaneous_get_dispatch_and_bad_key(self):
        conn = rate_connection_instantaneous(weight=2.0)
        self.assertEqual(conn.get('weight'), 2.0)
        self.assertEqual(conn.get('has_delay'), False)
        self.assertEqual(conn.get('status'),
                         {'weight': 2.0, 'delay': 1, 'has_delay': False, 'supports_wfr': True})
        with self.assertRaises(KeyError):
            conn.get('not_a_key')

    def test_delayed_get_dispatch_and_bad_key(self):
        conn = rate_connection_delayed(weight=1.5, delay_steps=3)
        self.assertEqual(conn.get('weight'), 1.5)
        self.assertEqual(conn.get('delay_steps'), 3)
        self.assertEqual(conn.get('delay'), 3)
        with self.assertRaises(KeyError):
            conn.get('not_a_key')

    def test_weight_scalar_validation_and_quantity(self):
        # Non-scalar weight is rejected by _to_float_scalar.
        with self.assertRaisesRegex(ValueError, 'scalar'):
            rate_connection_instantaneous(weight=np.array([1.0, 2.0]))
        # A Quantity weight has its mantissa extracted (the Quantity branch).
        conn = rate_connection_instantaneous(weight=u.Quantity(2.5))
        self.assertEqual(conn.weight, 2.5)


class TestParameterValidation(unittest.TestCase):
    """``tau <= 0`` is rejected at construction for every rate family."""

    def test_tau_must_be_positive(self):
        for label, cls, kw in _FAMILIES:
            with self.subTest(model=label):
                with self.assertRaisesRegex(ValueError, 'tau'):
                    cls(1, tau=0.0 * u.ms, **kw)

    def test_receptor_types_surface(self):
        for label, cls, kw in _FAMILIES:
            with self.subTest(model=label):
                node = cls(1, tau=10.0 * u.ms, **kw)
                self.assertEqual(node.receptor_types, {'RATE': 0})


class TestZeroLambdaPropagator(unittest.TestCase):
    """The ``lambda == 0`` branches of the exponential-Euler propagator stay finite."""

    def test_all_zero_lambda_branch(self):
        """A homogeneous ``lambda = 0`` takes the all-zero ``else`` branch."""
        for label, cls, kw in _LAMBDA_FAMILIES:
            with self.subTest(model=label):
                trace = _step(cls, 1, dict(lambda_=0.0, **kw))
                self.assertEqual(trace.shape, (3, 1))
                self.assertTrue(np.all(np.isfinite(trace)), msg=label)

    def test_mixed_zero_lambda_branch(self):
        """A population with mixed ``lambda`` exercises the ``where(zero_lambda, ...)`` patch."""
        for label, cls, kw in _LAMBDA_FAMILIES:
            with self.subTest(model=label):
                trace = _step(cls, 2, dict(lambda_=np.array([0.0, 1.0]), **kw))
                self.assertEqual(trace.shape, (3, 2))
                self.assertTrue(np.all(np.isfinite(trace)), msg=label)


class TestMultFactors(unittest.TestCase):
    """Multiplicative-coupling factors: real ``H`` vs the unity no-op."""

    def test_fixed_nonlinearity_models_return_unity(self):
        for label, cls, kw in [
            ('gauss_rate_ipn', gauss_rate_ipn, dict(sigma=1.0)),
            ('sigmoid_rate_ipn', sigmoid_rate_ipn, dict(sigma=0.0)),
            ('sigmoid_rate_gg_1998_ipn', sigmoid_rate_gg_1998_ipn, dict(sigma=0.0)),
            ('tanh_rate_ipn', tanh_rate_ipn, dict(sigma=0.0)),
            ('threshold_lin_rate_ipn', threshold_lin_rate_ipn, dict(sigma=0.0)),
        ]:
            with self.subTest(model=label):
                node = cls(1, tau=10.0 * u.ms, **kw)
                H_ex, H_in = node._mult_factors(jnp.asarray([0.7]))
                np.testing.assert_allclose(np.asarray(H_ex), 1.0)
                np.testing.assert_allclose(np.asarray(H_in), 1.0)

    def test_lin_and_template_return_linear_rate_form(self):
        """``H_ex = g_ex(theta_ex - r)``, ``H_in = g_in(theta_in + r)``."""
        for label, cls in [('lin_rate_ipn', lin_rate_ipn),
                           ('rate_neuron_ipn', rate_neuron_ipn),
                           ('rate_neuron_opn', rate_neuron_opn)]:
            with self.subTest(model=label):
                node = cls(1, tau=10.0 * u.ms, sigma=0.0,
                           g_ex=0.3, g_in=0.2, theta_ex=3.0, theta_in=0.5)
                r = jnp.asarray([1.0])
                H_ex, H_in = node._mult_factors(r)
                np.testing.assert_allclose(np.asarray(H_ex), 0.3 * (3.0 - 1.0))
                np.testing.assert_allclose(np.asarray(H_in), 0.2 * (0.5 + 1.0))


class TestRectifyOutput(unittest.TestCase):
    """``rectify_output`` clamps the rate at ``rectify_rate`` every step."""

    def test_negative_drive_is_rectified(self):
        for label, cls, kw in _LAMBDA_FAMILIES:
            with self.subTest(model=label):
                kw2 = dict(kw)
                kw2['mu'] = -5.0  # drive well below the rectification floor
                trace = _step(cls, 1, dict(lambda_=1.0, rectify_output=True,
                                           rectify_rate=0.0, **kw2))
                self.assertTrue(np.all(trace >= -1e-12), msg=f'{label}: not rectified')


class TestTemplateActivationDispatch(unittest.TestCase):
    """The ``rate_neuron`` template's user-nonlinearity dispatch and phi-signature."""

    _TEMPLATES = [('rate_neuron_ipn', rate_neuron_ipn), ('rate_neuron_opn', rate_neuron_opn)]

    def test_call_nl_tries_fn_self_x_then_fn_x(self):
        for label, cls in self._TEMPLATES:
            with self.subTest(model=label):
                node = cls(1, tau=10.0 * u.ms, sigma=0.0)
                # two-arg callable: fn(self, x) succeeds
                out2 = node._call_nl(lambda self, x: 3.0 * x, jnp.asarray([2.0]))
                np.testing.assert_allclose(np.asarray(out2), 6.0)
                # one-arg callable: fn(self, x) raises TypeError -> falls back to fn(x)
                out1 = node._call_nl(lambda x: 4.0 * x, jnp.asarray([2.0]))
                np.testing.assert_allclose(np.asarray(out1), 8.0)

    def test_user_input_nonlinearity_runs_in_update(self):
        for label, cls in self._TEMPLATES:
            with self.subTest(model=label):
                node_kw = dict(tau=10.0 * u.ms, sigma=0.0, mu=1.0,
                               input_nonlinearity=lambda self, x: 2.0 * x)
                trace = _step(cls, 1, node_kw)
                self.assertTrue(np.all(np.isfinite(trace)))

    def test_phi_signature_includes_input_nonlinearity(self):
        fn = lambda self, x: x
        for label, cls in self._TEMPLATES:
            with self.subTest(model=label):
                node = cls(1, tau=10.0 * u.ms, sigma=0.0, input_nonlinearity=fn)
                sig = node._phi_signature
                # last element pairs the marker with the callable identity
                self.assertEqual(sig[-1][0], 'input_nonlinearity')
                self.assertIs(sig[-1][1], fn)


if __name__ == '__main__':
    unittest.main()
