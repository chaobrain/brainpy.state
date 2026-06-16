# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""``rate_transformer_node`` update path on the seam-(H) substrate — goal-15a.

A ``rate_transformer_node`` is a *static* nonlinearity node: it has no leak dynamics,
it just maps its aggregated coupling input ``h = sum_delta_inputs`` to an output rate
each step. With ``linear_summation=True`` it applies its own gain, ``X = φ(h) = g·h``
(or a user ``input_nonlinearity``); with ``linear_summation=False`` the presynaptic gain
was already applied per connection and the node forwards the sum, ``X = h``.

Driven through the substrate by a ``step_rate_generator`` (rate ``R``, weight ``w`` →
``h = w·R`` after the one-step pipeline lag), the node settles in a single step (no
relaxation) to ``φ(w·R)`` or ``w·R``. The run is ``for_loop``-lowered by the Simulator.

This restores the update-path coverage the deleted-API prune removed (the old tests
drove the node through the host event queue that goal-15a deleted).
"""
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import brainunit as u
import braintools

from brainpy_state import (Simulator, step_rate_generator, rate_transformer_node,
                           multimeter, one_to_one)

DT = 0.1
W = 0.5
R = 5.0          # generator plateau rate
T = 20.0         # static node settles in one step; a short run suffices
H = W * R        # aggregated coupling input the node sees


def _bp_transformer(*, linear_summation=True, g=1.0, input_nonlinearity=None, w=W):
    sim = Simulator(dt=DT * u.ms)
    gen = sim.create(step_rate_generator, 1, params=dict(
        amplitude_times=[1.0 * u.ms], amplitude_values=[R]))
    node = sim.create(rate_transformer_node, 1, params=dict(
        linear_summation=linear_summation, g=g, input_nonlinearity=input_nonlinearity,
        rate_initializer=braintools.init.Constant(0.0)))
    sim.connect(gen, node, weight=w, rule=one_to_one, comm='dense')
    mm = sim.create(multimeter, record_from=('rate',))
    sim.connect(mm, node)
    res = sim.simulate(T * u.ms)
    return np.asarray(u.get_mantissa(res.trace(mm, 'rate'))).reshape(-1)


class TestRateTransformerNodeSubstrate(unittest.TestCase):
    """The static transformer maps its seam-(H) coupling input to ``φ(h)`` / ``h``."""

    def test_linear_summation_true_applies_gain(self):
        """``linear_summation=True`` applies the node's own gain: ``X = g·h``."""
        r = _bp_transformer(linear_summation=True, g=2.0)
        self.assertAlmostEqual(float(r[-1]), 2.0 * H, places=6)

    def test_linear_summation_false_forwards_sum(self):
        """``linear_summation=False`` forwards the already-weighted sum: ``X = h``."""
        r = _bp_transformer(linear_summation=False)
        self.assertAlmostEqual(float(r[-1]), H, places=6)

    def test_user_input_nonlinearity_is_applied(self):
        """A user ``input_nonlinearity`` replaces the default linear gain."""
        r = _bp_transformer(linear_summation=True,
                            input_nonlinearity=lambda self, x: jnp.tanh(x))
        self.assertAlmostEqual(float(r[-1]), float(np.tanh(H)), places=6)

    def test_zero_weight_leaves_node_at_phi_of_zero(self):
        """With ``weight=0`` the node sees ``h=0`` and outputs ``φ(0)=0``."""
        r = _bp_transformer(linear_summation=True, g=2.0, w=0.0)
        self.assertAlmostEqual(float(np.max(np.abs(r))), 0.0, places=10)


class TestRateTransformerNodeForLoop(unittest.TestCase):
    """The node's ``update`` lowers under a bare ``for_loop`` (no Simulator)."""

    def test_headless_for_loop_lowers_and_is_finite(self):
        brainstate.random.seed(0)
        with brainstate.environ.context(dt=DT * u.ms):
            node = rate_transformer_node(2, linear_summation=True, g=1.5,
                                         rate_initializer=braintools.init.Constant(0.0))
            brainstate.nn.init_all_states(node)

            def step(i):
                node.update()
                return u.get_mantissa(node.rate.value)

            trace = brainstate.transform.for_loop(step, np.arange(50))
        trace = np.asarray(trace)
        self.assertEqual(trace.shape, (50, 2))
        self.assertTrue(np.all(np.isfinite(trace)))
        # No coupling deposited -> h = 0 -> phi(0) = g*0 = 0 every step.
        np.testing.assert_allclose(trace, 0.0, atol=1e-12)


class TestRateTransformerNodeDefaults(unittest.TestCase):
    """NEST default parameters and recordable/receptor surface."""

    def test_defaults_and_surface(self):
        node = rate_transformer_node(1)
        self.assertEqual(node.linear_summation, True)
        self.assertEqual(float(u.get_mantissa(node.g)), 1.0)
        self.assertEqual(node.recordables, ['rate'])
        self.assertEqual(node.receptor_types, {'RATE': 0})


if __name__ == '__main__':
    unittest.main()
