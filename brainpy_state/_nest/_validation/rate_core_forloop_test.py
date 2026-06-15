# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""``for_loop``-lowering regression for the de-queued rate-neuron family.

Every rate neuron must drive under ``brainstate.transform.for_loop`` (the
substrate's compiled multi-step primitive): the JAX dynamics trace once and the
whole rollout lowers into one XLA program. The pre-de-queue host dict-queue
(``_common_inputs`` reading ``sum_delta_inputs`` through ``np.asarray``) cannot
-- ``sum_delta_inputs`` is a tracer under ``for_loop`` and ``np.asarray`` on it
raises ``TracerArrayConversionError``. So this is the RED that drives the
de-queue of each φ-family and the GREEN that pins it afterwards.

Headless (no Simulator, no connection): a single neuron stepped ``N`` times with
``mu`` drive and ``sum_delta_inputs(0.0) == 0``. The assertion is deliberately
weak -- the trajectory is finite and non-trivial -- because the point is the
*lowering*, not the numbers (those are the NEST parity suite's job).
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainunit as u
import braintools

from brainpy_state import (
    lin_rate_ipn, lin_rate_opn,
    gauss_rate_ipn,
    sigmoid_rate_ipn,
    sigmoid_rate_gg_1998_ipn,
    tanh_rate_ipn, tanh_rate_opn,
    threshold_lin_rate_ipn, threshold_lin_rate_opn,
    rate_neuron_ipn, rate_neuron_opn,
)

DT = 0.1 * u.ms
N_STEPS = 200

#: (label, class, kwargs). ``mu`` is the drive; ``sigma=0`` keeps the step
#: deterministic (gauss is the exception -- its gain width must be > 0, so it
#: stays stochastic but finite).
_CASES = [
    ('lin_rate_ipn', lin_rate_ipn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
    ('lin_rate_opn', lin_rate_opn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
    ('gauss_rate_ipn', gauss_rate_ipn, dict(tau=10.0 * u.ms, mu=1.0, sigma=1.0, g=1.0)),
    ('sigmoid_rate_ipn', sigmoid_rate_ipn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
    ('sigmoid_rate_gg_1998_ipn', sigmoid_rate_gg_1998_ipn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
    ('tanh_rate_ipn', tanh_rate_ipn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
    ('tanh_rate_opn', tanh_rate_opn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
    ('threshold_lin_rate_ipn', threshold_lin_rate_ipn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
    ('threshold_lin_rate_opn', threshold_lin_rate_opn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
    ('rate_neuron_ipn', rate_neuron_ipn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
    ('rate_neuron_opn', rate_neuron_opn, dict(tau=10.0 * u.ms, mu=2.0, sigma=0.0)),
]


def _run_for_loop(cls, kwargs, n=N_STEPS):
    """Step one ``cls`` neuron ``n`` times under ``for_loop``; return the rate trace."""
    brainstate.random.seed(0)
    with brainstate.environ.context(dt=DT):
        neuron = cls(1, rate_initializer=braintools.init.Constant(0.0), **kwargs)
        brainstate.nn.init_all_states(neuron)

        def step(i):
            neuron.update()
            return u.get_mantissa(neuron.rate.value)

        rates = brainstate.transform.for_loop(step, np.arange(n))
    return np.asarray(rates).reshape(n, -1)


class TestRateForLoopLowering(unittest.TestCase):
    """Each rate neuron lowers into ``for_loop`` and produces a finite trajectory."""

    def test_each_family_lowers_and_is_finite(self):
        for label, cls, kwargs in _CASES:
            with self.subTest(model=label):
                trace = _run_for_loop(cls, kwargs)
                self.assertEqual(trace.shape, (N_STEPS, 1), msg=label)
                self.assertTrue(np.all(np.isfinite(trace)),
                                msg=f'{label}: non-finite rate under for_loop')
                # mu drive relaxes the rate away from the zero initial condition.
                self.assertGreater(float(np.abs(trace[-1, 0])), 1e-3, msg=label)


if __name__ == '__main__':
    unittest.main()
