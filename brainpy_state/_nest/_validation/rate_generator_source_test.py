# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""``step_rate_generator`` as a seam-(H) rate source — goal-15a Phase 1b.8.

A ``step_rate_generator`` emits a deterministic piecewise-constant rate. Wired as a
Simulator source, ``connect(step_rate_generator, rate_neuron, comm='dense')`` routes
``weight·rate`` into the post's default delta channel each step (the same seam-(H) path
a rate neuron's emission uses) — the generator needs no host event queue. A driven
``lin_rate_ipn`` (``λ=1, μ=0``) therefore relaxes, plateau by plateau, to ``weight·rate``.

The whole run is ``for_loop``-lowered (``Simulator.simulate`` compiles the step), so this
also pins the generator-source carry-shape contract.

* ``TestStepRateGeneratorSource`` (NEST-free) — each plateau relaxes to ``weight·rate``;
  the step change moves the post monotonically.
* ``TestStepRateGeneratorNestParity`` (``@requires_nest``) — the driven trajectory matches
  live NEST ``step_rate_generator`` + ``rate_connection_instantaneous``.
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import saiunit as u
import braintools

try:
    import nest
except Exception:
    nest = None

from brainpy_state import (Simulator, step_rate_generator, lin_rate_ipn,
                           multimeter, one_to_one)
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import TraceTolerance

DT = 0.1
TAU = 10.0
W = 0.5
TIMES = [5.0, 80.0]       # plateau change times (ms, on the dt grid)
VALUES = [10.0, 30.0]     # plateau rates (Hz)
T = 180.0                 # plateau 1 spans ~7.5 tau, plateau 2 ~10 tau -> both settle
I_P1 = int(79.0 / DT)     # deep in plateau 1, just before the 80 ms change
I_P2 = int((T - DT) / DT)  # deep in plateau 2 (end of run)

GEN_TOL = TraceTolerance(1e-4, 1e-4, align_steps=6, label='C',
                         note='step_rate_generator rate source vs live NEST')


def _bp_driven(*, w=W, T=T, dt=DT):
    sim = Simulator(dt=dt * u.ms)
    gen = sim.create(step_rate_generator, 1, params=dict(
        amplitude_times=[t * u.ms for t in TIMES], amplitude_values=list(VALUES)))
    post = sim.create(lin_rate_ipn, 1, params=dict(
        tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=0.0, g=1.0, linear_summation=True,
        rate_initializer=braintools.init.Constant(0.0),
        noise_initializer=braintools.init.Constant(0.0)))
    sim.connect(gen, post, weight=w, rule=one_to_one, comm='dense')
    mm = sim.create(multimeter, record_from=('rate',))
    sim.connect(mm, post)
    res = sim.simulate(T * u.ms)
    return np.asarray(u.get_mantissa(res.trace(mm, 'rate'))).reshape(-1)


class TestStepRateGeneratorSource(unittest.TestCase):
    """A ``step_rate_generator`` drives a rate neuron through the seam-(H) deposit."""

    def test_each_plateau_relaxes_to_weight_times_rate(self):
        r = _bp_driven()
        # Each settled plateau sits at weight * generator-rate (residual ~ e^{-7.5 tau}).
        self.assertAlmostEqual(float(r[I_P1]), W * VALUES[0], delta=5e-3)
        self.assertAlmostEqual(float(r[I_P2]), W * VALUES[1], delta=5e-3)

    def test_step_change_drives_post_upward(self):
        """Raising the generator rate (10 -> 30) raises the driven rate by weight*delta."""
        r = _bp_driven()
        self.assertGreater(float(r[I_P2]), float(r[I_P1]) + 1.0)
        self.assertAlmostEqual(float(r[I_P2] - r[I_P1]),
                               W * (VALUES[1] - VALUES[0]), delta=1e-2)

    def test_zero_weight_leaves_post_at_rest(self):
        """With ``weight=0`` the generator delivers nothing; the post stays at ``μ=0``."""
        r = _bp_driven(w=0.0)
        self.assertAlmostEqual(float(np.max(np.abs(r))), 0.0, places=10)


def _nest_driven(*, w=W, T=T, dt=DT):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt, 'use_wfr': False})
    gen = nest.Create('step_rate_generator', params={
        'amplitude_times': list(TIMES), 'amplitude_values': list(VALUES)})
    post = nest.Create('lin_rate_ipn', 1, params={
        'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'mu': 0.0, 'g': 1.0,
        'linear_summation': True, 'rate': 0.0})
    # step_rate_generator only emits DelayedRateConnectionEvent (no instantaneous
    # output), so it couples through rate_connection_delayed at the minimum delay --
    # the one-step lag matches the brainpy generator path's pipeline lag.
    nest.Connect(gen, post,
                 syn_spec={'synapse_model': 'rate_connection_delayed',
                           'weight': w, 'delay': dt})
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    nest.Connect(mm, post)
    nest.Simulate(T)
    ev = mm.events
    order = np.argsort(np.asarray(ev['times']), kind='stable')
    return np.asarray(ev['rate'])[order]


@requires_nest
class TestStepRateGeneratorNestParity(unittest.TestCase):
    """The generator-driven trajectory matches live NEST."""

    def test_driven_trajectory_matches_nest(self):
        bp = _bp_driven()
        ns = _nest_driven()
        k = min(len(bp), len(ns))
        # Coupling actually drove the post above rest.
        self.assertGreater(ns[:k].max(), W * VALUES[0] * 0.5)
        compare_trace(ns[:k], bp[:k], tol=GEN_TOL, metric='step_rate_source').assert_()


if __name__ == '__main__':
    unittest.main()
