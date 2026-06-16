# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Rate-coupling micro-parity — the **design-A arbiter** for goal 15a.

Does routing a rate connection ``r_post = Σ w·r_pre`` through the ordinary
``Simulator`` static-connection path —

    ``connect(driver, driven, comm='dense')``

— with the presynaptic ``rate`` emitted via the **receptorless** seam-(H)
continuous-emission branch (deposit ``weight·rate`` into the driven neuron's
**default** delta channel) reproduce a linear rate network's behaviour? If yes,
**option (a)** is sufficient: no bespoke rate-deposit primitive is needed for the
base (``mult_coupling=False``) path.

Two arbiters:

* ``TestRateCouplingFixedPoint`` (NEST-free, always runs) — two ``lin_rate_ipn``
  (``sigma=0``, ``lambda_=1``, ``g=1``) coupled ``driver -> driven`` by one
  instantaneous connection relax to the **analytic** coupled fixed point
  ``r0* = mu0``, ``r1* = mu1 + w·mu0``. This is the RED test that drives the
  lin_rate JAX de-queue + the substrate continuous-emitter branch.
* ``TestRateCouplingNestParity`` (``@requires_nest``) — the same two-neuron net vs
  live NEST ``lin_rate_ipn`` + ``rate_connection_instantaneous`` run with
  ``use_wfr=False`` (so NEST seeds the instantaneous coupling from the previous
  step, matching our intrinsic one-step pipeline lag). Tight band; deterministic.
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import brainunit as u
import braintools

try:
    import nest
except Exception:
    nest = None

from brainpy_state import Simulator, lin_rate_ipn, multimeter, one_to_one
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

DT = 0.1
TAU = 10.0
MU0 = 2.0
MU1 = 0.0
W = 0.5
T = 400.0

#: Deterministic instantaneous-rate trajectory vs NEST (use_wfr=False seeds the
#: coupling from the previous step, matching our one-step pipeline lag). The
#: dynamics are an exact exponential-Euler propagator, so a tight band holds once
#: ``align_steps`` absorbs the fixed integer pipeline-latency offset.
RATE_TOL = TraceTolerance(1e-4, 1e-4, align_steps=4, label='C',
                          note='instantaneous rate coupling vs live NEST (use_wfr=False)')


def _lin_rate_units(mu):
    """brainpy ``lin_rate_ipn`` params (units attached) for a deterministic cell."""
    return dict(tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=mu, g=1.0,
                linear_summation=True,
                rate_initializer=braintools.init.Constant(0.0),
                noise_initializer=braintools.init.Constant(0.0))


def _bp_two_neuron(mu0, mu1, w, *, T, dt=DT):
    """Two coupled ``lin_rate_ipn`` via the Simulator; (samples, 2) rate trace."""
    sim = Simulator(dt=dt * u.ms)
    driver = sim.create(lin_rate_ipn, 1, params=_lin_rate_units(mu0))
    driven = sim.create(lin_rate_ipn, 1, params=_lin_rate_units(mu1))
    sim.connect(driver, driven, weight=w, rule=one_to_one, comm='dense')
    mm0 = sim.create(multimeter, record_from=('rate',))
    mm1 = sim.create(multimeter, record_from=('rate',))
    sim.connect(mm0, driver)
    sim.connect(mm1, driven)
    res = sim.simulate(T * u.ms)
    r0 = np.asarray(u.get_mantissa(res.trace(mm0, 'rate'))).reshape(-1)
    r1 = np.asarray(u.get_mantissa(res.trace(mm1, 'rate'))).reshape(-1)
    return np.stack([r0, r1], axis=1)


class TestRateCouplingFixedPoint(unittest.TestCase):
    """Two coupled lin_rate neurons relax to the analytic coupled fixed point."""

    def test_instantaneous_coupling_reaches_analytic_fixed_point(self):
        """``r0* = mu0`` and ``r1* = mu1 + w·mu0`` (lambda=1, g=1, sigma=0).

        The driven neuron's steady-state rate is its own drive ``mu1`` plus the
        weighted driver rate ``w·r0*`` delivered as the coupling input ``h`` and
        passed through the linear gain ``phi(h)=g·h``.
        """
        traj = _bp_two_neuron(MU0, MU1, W, T=T)
        r0_final = traj[-1, 0]
        r1_final = traj[-1, 1]
        self.assertAlmostEqual(r0_final, MU0, places=4,
                               msg='driver should relax to mu0/lambda')
        self.assertAlmostEqual(r1_final, MU1 + W * MU0, places=4,
                               msg='driven should relax to mu1 + w*mu0')


# --- NEST side --------------------------------------------------------------------

def _nest_two_neuron(mu0, mu1, w, *, T, dt=DT):
    """Two coupled NEST ``lin_rate_ipn`` (instantaneous, use_wfr=False); (samples, 2)."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt, 'use_wfr': False})
    # NEST spells the passive-decay rate ``lambda`` (a Python keyword), so it must
    # go through a string key rather than ``lambda_=...``.
    params = {'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'g': 1.0,
              'linear_summation': True, 'rate': 0.0}
    driver = nest.Create('lin_rate_ipn', 1, params={**params, 'mu': mu0})
    driven = nest.Create('lin_rate_ipn', 1, params={**params, 'mu': mu1})
    nest.Connect(driver, driven,
                 syn_spec={'synapse_model': 'rate_connection_instantaneous',
                           'weight': w})
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    nest.Connect(mm, driver)
    nest.Connect(mm, driven)
    nest.Simulate(T)
    ev = mm.events
    senders = np.asarray(ev['senders'])
    times = np.asarray(ev['times'])
    rate = np.asarray(ev['rate'])
    cols = []
    for nid in driver.tolist() + driven.tolist():
        m = senders == nid
        order = np.argsort(times[m], kind='stable')
        cols.append(rate[m][order])
    return np.stack(cols, axis=1)


@requires_nest
class TestRateCouplingNestParity(unittest.TestCase):
    """Instantaneous rate coupling via the Simulator matches live NEST (design A)."""

    def test_driven_rate_trajectory_matches_nest(self):
        """The driven neuron's full ``rate`` trajectory matches NEST (use_wfr=False)."""
        bp = _bp_two_neuron(MU0, MU1, W, T=T)
        ns = _nest_two_neuron(MU0, MU1, W, T=T)
        n = min(bp.shape[0], ns.shape[0])
        # Sanity: the coupling actually drove the driven neuron above its own mu.
        self.assertGreater(ns[:n, 1].max(), MU1 + 0.5 * W * MU0)
        for col, name in ((0, 'driver'), (1, 'driven')):
            compare_trace(ns[:n, col], bp[:n, col],
                          tol=RATE_TOL, metric=f'rate[{name}]').assert_()


if __name__ == '__main__':
    unittest.main()
