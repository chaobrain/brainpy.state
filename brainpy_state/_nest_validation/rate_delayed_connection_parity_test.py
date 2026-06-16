# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Delayed rate connection parity — ``rate_connection_delayed`` on the substrate.

A delayed rate connection is the instantaneous seam-(H) ``EventProjection`` plus
an axonal ``delay_seam = InputDelay((n_pre,), delay)`` (dt-rounded). The total
latency is ``1 (intrinsic pipeline) + round(delay/dt)``; the **difference** from
the instantaneous variant is therefore exactly ``delay_steps = round(delay/dt)``
(spec §3.4). The delay shifts only the *transient* — the coupled fixed point
``r1^\* = mu1 + w·mu0`` is unchanged.

Two arbiters:

* ``TestDelayedRateConnectionShift`` (NEST-free, always runs) — a feed-forward
  ``driver -> driven`` pair: the delayed ``rate`` trajectory equals the
  instantaneous one shifted by exactly ``delay_steps`` (machine precision), at
  ``d_min = 1`` and ``> d_min``; both reach the same fixed point.
* ``TestDelayedRateConnectionNestParity`` (``@requires_nest``) — the delayed
  Simulator net matches live NEST ``lin_rate_ipn`` + ``rate_connection_delayed``
  (matched ``delay = delay_steps·dt``); ``align_steps`` absorbs the uniform
  pipeline offset.
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
G = 1.0
MU0 = 2.0
MU1 = 0.0
W = 0.5
T = 300.0

#: The delayed net is the instantaneous net shifted by ``delay_steps``; the band
#: is tight once ``align_steps`` (>= 1 pipeline + max tested delay) absorbs the
#: uniform integer offset.
DELAY_TOL = TraceTolerance(1e-4, 1e-4, align_steps=12, label='C',
                           note='delayed rate coupling vs live NEST rate_connection_delayed')


def _lin_rate_units(mu):
    return dict(tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=float(mu), g=G,
                linear_summation=True,
                rate_initializer=braintools.init.Constant(0.0),
                noise_initializer=braintools.init.Constant(0.0))


def _bp_pair(delay_steps, *, T=T, dt=DT):
    """Feed-forward ``driver -> driven`` rate pair; returns the driven ``rate`` trace.

    ``delay_steps == 0`` builds the instantaneous connection (``delay=None``);
    otherwise an axonal ``InputDelay`` of ``delay_steps·dt`` is wired in.
    """
    sim = Simulator(dt=dt * u.ms)
    driver = sim.create(lin_rate_ipn, 1, params=_lin_rate_units(MU0))
    driven = sim.create(lin_rate_ipn, 1, params=_lin_rate_units(MU1))
    delay = None if delay_steps == 0 else delay_steps * dt * u.ms
    sim.connect(driver, driven, weight=W, rule=one_to_one, comm='dense', delay=delay)
    mm = sim.create(multimeter, record_from=('rate',))
    sim.connect(mm, driven)
    res = sim.simulate(T * u.ms)
    return np.asarray(u.get_mantissa(res.trace(mm, 'rate'))).reshape(-1)


class TestDelayedRateConnectionShift(unittest.TestCase):
    """A delayed rate connection is the instantaneous one shifted by ``delay_steps``."""

    def test_delay_shifts_trajectory_by_exactly_delay_steps(self):
        inst = _bp_pair(0)
        for d in (1, 3, 10):
            with self.subTest(delay_steps=d):
                dly = _bp_pair(d)
                k = min(len(inst), len(dly))
                # delayed[t] == instantaneous[t - d] for t >= d.
                np.testing.assert_allclose(dly[d:k], inst[:k - d], atol=1e-9,
                                           err_msg=f'delay_steps={d}: not a pure {d}-step shift')

    def test_delay_preserves_fixed_point(self):
        """The delay changes only the transient; the steady state is ``mu1 + w·mu0``."""
        fixed = MU1 + W * MU0
        for d in (0, 1, 10):
            with self.subTest(delay_steps=d):
                self.assertAlmostEqual(float(_bp_pair(d)[-1]), fixed, places=4)

    def test_d_min_is_one_step(self):
        """The smallest delayed connection (``delay = dt``) lags instantaneous by 1 step."""
        inst = _bp_pair(0)
        dly = _bp_pair(1)
        k = min(len(inst), len(dly))
        self.assertGreater(float(np.max(np.abs(dly[:k] - inst[:k]))), 1e-3,
                           msg='delay=dt must differ from instantaneous on the transient')
        np.testing.assert_allclose(dly[1:k], inst[:k - 1], atol=1e-9)


# --- NEST side --------------------------------------------------------------------

def _nest_pair(delay_steps, *, T=T, dt=DT):
    """Feed-forward NEST ``driver -> driven`` via ``rate_connection_delayed``."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt})
    base = {'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'g': G,
            'linear_summation': True, 'rate': 0.0}
    driver = nest.Create('lin_rate_ipn', 1, params={**base, 'mu': MU0})
    driven = nest.Create('lin_rate_ipn', 1, params={**base, 'mu': MU1})
    nest.Connect(driver, driven,
                 syn_spec={'synapse_model': 'rate_connection_delayed',
                           'weight': W, 'delay': delay_steps * dt})
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    nest.Connect(mm, driven)
    nest.Simulate(T)
    ev = mm.events
    order = np.argsort(np.asarray(ev['times']), kind='stable')
    return np.asarray(ev['rate'])[order]


@requires_nest
class TestDelayedRateConnectionNestParity(unittest.TestCase):
    """Delayed rate coupling via the Simulator matches live NEST ``rate_connection_delayed``."""

    def test_delayed_trajectory_matches_nest(self):
        for d in (1, 5):
            with self.subTest(delay_steps=d):
                bp = _bp_pair(d)
                ns = _nest_pair(d)
                k = min(len(bp), len(ns))
                # Coupling actually drove the driven cell above its own mu.
                self.assertGreater(ns[:k].max(), MU1 + 0.5 * W * MU0)
                compare_trace(ns[:k], bp[:k], tol=DELAY_TOL,
                              metric=f'delayed_rate[d={d}]').assert_()


if __name__ == '__main__':
    unittest.main()
