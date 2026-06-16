# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Multiplicative rate coupling (``mult_coupling``) dual-channel parity — spec §3.2.

NEST's linear rate neuron can couple **multiplicatively**: the presynaptic drive is
split by weight sign into an excitatory partial sum (``w>0`` edges) and an inhibitory
one (``w<0`` edges), and each is scaled by a receiver-state factor before it enters
the dynamics (``rate_neuron_ipn_impl.h``)

.. math::

    \tau\,\dot r = -\lambda r + \mu
        + H_\mathrm{ex}(r)\,\phi\!\Big(\sum_{w>0} w\,r_\mathrm{pre}\Big)
        + H_\mathrm{in}(r)\,\phi\!\Big(\sum_{w<0} w\,r_\mathrm{pre}\Big),

with ``H_ex(r) = g_ex(θ_ex − r)`` and ``H_in(r) = g_in(θ_in + r)``. On the substrate
the split is realised by two labelled rate projections (``max(W,0)`` into the post's
``'rate_ex'`` delta channel, ``min(W,0)`` into ``'rate_in'``); the receiver reads them
back with ``sum_delta_inputs(label=...)`` and applies ``H_ex``/``H_in``.

Because ``φ`` here is linear (``φ(h)=g·h``) and ``H`` is affine in ``r``, the coupled
fixed point stays closed-form, which lets a NEST-free arbiter pin the steady state
exactly; a second arbiter matches the full trajectory against live NEST.

Arbiters:

* ``TestMultCouplingDualChannelFixedPoint`` (NEST-free) — an ex/in driver pair into a
  ``mult_coupling`` post relaxes to the closed-form fixed point derived from NEST's
  published ``H_ex``/``H_in`` equations; flipping a driver's sign moves it between the
  channels.
* ``TestMultCouplingNestParity`` (``@requires_nest``) — the same net matches live NEST
  ``lin_rate_ipn(mult_coupling=True)`` + ``rate_connection_instantaneous``.
* ``TestMultCouplingNoOpForFixedPhi`` (NEST-free) — for a fixed-φ model
  (``gauss_rate``) ``mult_coupling=True`` is a *no-op* (its ``H≡1``), so it is bit-for-bit
  identical to ``mult_coupling=False``.
* ``TestRatePhiHomogeneityGuard`` (NEST-free) — ``connect()`` refuses a
  ``linear_summation=False`` rate connection between heterogeneous φ (and a summation-mode
  mismatch), and admits the homogeneous / ``linear_summation=True`` cases.
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainunit as u
import braintools

try:
    import nest
except Exception:
    nest = None

from brainpy_state import Simulator, lin_rate_ipn, gauss_rate_ipn, multimeter, one_to_one
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

DT = 0.1
TAU = 10.0
G = 1.0
T = 2000.0

# Post mult_coupling factors H_ex = g_ex(θ_ex − r), H_in = g_in(θ_in + r).
# θ_ex is kept clear of the fixed point so H_ex does not vanish at steady state
# (a degenerate θ_ex = r* would make the excitatory channel invisible).
G_EX, G_IN, THETA_EX, THETA_IN = 0.3, 0.2, 3.0, 0.5
MU_P = 1.0          # post intrinsic drive
MU_EX, MU_IN = 2.0, 1.5   # excitatory / inhibitory driver rates (= their mu)
W_EX, W_IN = 0.5, -0.4    # connection weights (sign selects the channel)

MC_TOL = TraceTolerance(1e-4, 1e-4, align_steps=6, label='C',
                        note='multiplicative rate coupling vs live NEST (use_wfr=False)')


def _mult_coupling_fixed_point(mu_p, g, g_ex, g_in, theta_ex, theta_in, h_ex, h_in):
    r"""Closed-form steady state of a linear-φ ``mult_coupling`` post.

    With ``φ(h)=g·h`` the increment ``H_ex·g·h_ex + H_in·g·h_in`` is affine in ``r``
    (through ``H``), so the fixed point ``r = μ + H_ex(r)·g·h_ex + H_in(r)·g·h_in``
    solves linearly:

    .. math::

        r^\* = \frac{\mu + g_\mathrm{ex}θ_\mathrm{ex}A + g_\mathrm{in}θ_\mathrm{in}B}
                    {1 + g_\mathrm{ex}A - g_\mathrm{in}B},\quad A=g\,h_\mathrm{ex},\ B=g\,h_\mathrm{in}.
    """
    A, B = g * h_ex, g * h_in
    return (mu_p + g_ex * theta_ex * A + g_in * theta_in * B) / (1.0 + g_ex * A - g_in * B)


def _post_params(mult_coupling):
    return dict(tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=MU_P, g=G,
                mult_coupling=mult_coupling, g_ex=G_EX, g_in=G_IN,
                theta_ex=THETA_EX, theta_in=THETA_IN, linear_summation=True,
                rate_initializer=braintools.init.Constant(0.0),
                noise_initializer=braintools.init.Constant(0.0))


def _driver_params(mu):
    return dict(tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=float(mu), g=G,
                linear_summation=True,
                rate_initializer=braintools.init.Constant(0.0),
                noise_initializer=braintools.init.Constant(0.0))


def _bp_mult_coupling(w_ex, w_in, *, mu_ex=MU_EX, mu_in=MU_IN, T=T, dt=DT):
    """Two constant drivers (one per sign) into one ``mult_coupling`` post; post rate."""
    sim = Simulator(dt=dt * u.ms)
    drv_ex = sim.create(lin_rate_ipn, 1, params=_driver_params(mu_ex))
    drv_in = sim.create(lin_rate_ipn, 1, params=_driver_params(mu_in))
    post = sim.create(lin_rate_ipn, 1, params=_post_params(True))
    sim.connect(drv_ex, post, weight=w_ex, rule=one_to_one, comm='dense')
    sim.connect(drv_in, post, weight=w_in, rule=one_to_one, comm='dense')
    mm = sim.create(multimeter, record_from=('rate',))
    sim.connect(mm, post)
    res = sim.simulate(T * u.ms)
    return np.asarray(u.get_mantissa(res.trace(mm, 'rate'))).reshape(-1)


class TestMultCouplingDualChannelFixedPoint(unittest.TestCase):
    """A ``mult_coupling`` post relaxes to the closed-form ex/in fixed point (NEST-free)."""

    def test_reaches_closed_form_fixed_point(self):
        rstar = _mult_coupling_fixed_point(MU_P, G, G_EX, G_IN, THETA_EX, THETA_IN,
                                           h_ex=W_EX * MU_EX, h_in=W_IN * MU_IN)
        final = float(_bp_mult_coupling(W_EX, W_IN)[-1])
        self.assertAlmostEqual(final, rstar, places=4,
                               msg=f'mult_coupling steady state {final} != closed form {rstar}')

    def test_excitatory_and_inhibitory_channels_move_the_post_oppositely(self):
        """The ex channel (``w>0``) raises the post; the in channel (``w<0``) lowers it.

        ``H_ex=g_ex(θ_ex−r)>0`` with ``h_ex>0`` pushes ``r`` up; ``H_in=g_in(θ_in+r)>0``
        with ``h_in<0`` pushes it down. Driving only one channel must bracket the
        intrinsic ``μ`` from the expected side.
        """
        base = MU_P
        ex_only = float(_bp_mult_coupling(W_EX, 0.0)[-1])
        in_only = float(_bp_mult_coupling(0.0, W_IN)[-1])
        self.assertGreater(ex_only, base + 1e-3)
        self.assertLess(in_only, base - 1e-3)

    def test_sign_routes_the_channel(self):
        """A driver routed through ``rate_ex`` (``w>0``) vs ``rate_in`` (``w<0``).

        Feeding the *same* driver once with ``+w`` and once with ``−w`` exercises the two
        channels with their different ``H`` factors, so the steady states must differ and
        each must match the closed form for its channel.
        """
        w = 0.5
        pos = float(_bp_mult_coupling(w, 0.0, mu_ex=2.0)[-1])
        neg = float(_bp_mult_coupling(0.0, -w, mu_in=2.0)[-1])
        rstar_pos = _mult_coupling_fixed_point(MU_P, G, G_EX, G_IN, THETA_EX, THETA_IN,
                                               h_ex=w * 2.0, h_in=0.0)
        rstar_neg = _mult_coupling_fixed_point(MU_P, G, G_EX, G_IN, THETA_EX, THETA_IN,
                                               h_ex=0.0, h_in=-w * 2.0)
        self.assertAlmostEqual(pos, rstar_pos, places=4)
        self.assertAlmostEqual(neg, rstar_neg, places=4)
        self.assertGreater(abs(pos - neg), 1e-3)


# --- NEST side --------------------------------------------------------------------

def _nest_mult_coupling(w_ex, w_in, *, mu_ex=MU_EX, mu_in=MU_IN, T=T, dt=DT):
    """Live NEST: ex/in drivers into a ``mult_coupling`` ``lin_rate_ipn`` post."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt, 'use_wfr': False})
    drv = {'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'g': G, 'linear_summation': True, 'rate': 0.0}
    drv_ex = nest.Create('lin_rate_ipn', 1, params={**drv, 'mu': float(mu_ex)})
    drv_in = nest.Create('lin_rate_ipn', 1, params={**drv, 'mu': float(mu_in)})
    post = nest.Create('lin_rate_ipn', 1, params={
        'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'g': G, 'mu': MU_P, 'rate': 0.0,
        'linear_summation': True, 'mult_coupling': True,
        'g_ex': G_EX, 'g_in': G_IN, 'theta_ex': THETA_EX, 'theta_in': THETA_IN})
    nest.Connect(drv_ex, post, syn_spec={'synapse_model': 'rate_connection_instantaneous', 'weight': w_ex})
    nest.Connect(drv_in, post, syn_spec={'synapse_model': 'rate_connection_instantaneous', 'weight': w_in})
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    nest.Connect(mm, post)
    nest.Simulate(T)
    ev = mm.events
    order = np.argsort(np.asarray(ev['times']), kind='stable')
    return np.asarray(ev['rate'])[order]


@requires_nest
class TestMultCouplingNestParity(unittest.TestCase):
    """Dual-channel multiplicative coupling matches live NEST."""

    def test_trajectory_matches_nest(self):
        bp = _bp_mult_coupling(W_EX, W_IN)
        ns = _nest_mult_coupling(W_EX, W_IN)
        k = min(len(bp), len(ns))
        compare_trace(ns[:k], bp[:k], tol=MC_TOL, metric='mult_coupling_rate').assert_()

    def test_nest_steady_state_matches_closed_form(self):
        """Sanity on NEST itself: its relaxed rate is the same closed form."""
        ns = _nest_mult_coupling(W_EX, W_IN)
        rstar = _mult_coupling_fixed_point(MU_P, G, G_EX, G_IN, THETA_EX, THETA_IN,
                                           h_ex=W_EX * MU_EX, h_in=W_IN * MU_IN)
        self.assertAlmostEqual(float(ns[-1]), rstar, places=3)


# --- no-op gate for fixed-φ models ------------------------------------------------

class TestMultCouplingNoOpForFixedPhi(unittest.TestCase):
    """For a fixed-φ model (``H≡1``) ``mult_coupling`` is a true no-op (the gate)."""

    @staticmethod
    def _gauss_post_trace(mult_coupling):
        sim = Simulator(dt=DT * u.ms)
        drv = sim.create(lin_rate_ipn, 1, params=_driver_params(2.0))
        # gauss_rate exposes ``mult_coupling`` but no gain params (H≡1 for it);
        # the gate must make True identical to False regardless. sigma=0 keeps the
        # run deterministic so the comparison isolates the gate, not the noise draw.
        post = sim.create(gauss_rate_ipn, 1, params=dict(
            tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=0.5, g=1.0,
            mult_coupling=mult_coupling, linear_summation=True,
            rate_initializer=braintools.init.Constant(0.0),
            noise_initializer=braintools.init.Constant(0.0)))
        sim.connect(drv, post, weight=0.5, rule=one_to_one, comm='dense')
        mm = sim.create(multimeter, record_from=('rate',))
        sim.connect(mm, post)
        res = sim.simulate(200.0 * u.ms)
        return np.asarray(u.get_mantissa(res.trace(mm, 'rate'))).reshape(-1)

    def test_gauss_mult_coupling_true_equals_false(self):
        on = self._gauss_post_trace(True)
        off = self._gauss_post_trace(False)
        np.testing.assert_allclose(
            on, off, atol=1e-12,
            err_msg='gauss_rate has H≡1, so mult_coupling=True must be identical to False')


# --- homogeneity guard ------------------------------------------------------------

class TestRatePhiHomogeneityGuard(unittest.TestCase):
    """``connect()`` enforces the φ / summation-mode contract (spec §3.3)."""

    @staticmethod
    def _lin(sim, *, g=1.0, linear_summation):
        return sim.create(lin_rate_ipn, 1, params=dict(
            tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=0.0, g=g,
            linear_summation=linear_summation,
            rate_initializer=braintools.init.Constant(0.0),
            noise_initializer=braintools.init.Constant(0.0)))

    @staticmethod
    def _gauss(sim, *, linear_summation):
        return sim.create(gauss_rate_ipn, 1, params=dict(
            tau=TAU * u.ms, lambda_=1.0, sigma=0.5, mu=0.0, g=1.0,
            linear_summation=linear_summation,
            rate_initializer=braintools.init.Constant(0.0),
            noise_initializer=braintools.init.Constant(0.0)))

    def test_homogeneous_phi_false_summation_connects(self):
        sim = Simulator(dt=DT * u.ms)
        a = self._lin(sim, g=1.5, linear_summation=False)
        b = self._lin(sim, g=1.5, linear_summation=False)
        sim.connect(a, b, weight=0.5, rule=one_to_one, comm='dense')  # no raise

    def test_heterogeneous_phi_false_summation_raises(self):
        sim = Simulator(dt=DT * u.ms)
        a = self._lin(sim, linear_summation=False)
        b = self._gauss(sim, linear_summation=False)
        with self.assertRaisesRegex(ValueError, 'homogeneous input nonlinearity'):
            sim.connect(a, b, weight=0.5, rule=one_to_one, comm='dense')

    def test_different_gain_false_summation_raises(self):
        sim = Simulator(dt=DT * u.ms)
        a = self._lin(sim, g=1.0, linear_summation=False)
        b = self._lin(sim, g=2.0, linear_summation=False)
        with self.assertRaisesRegex(ValueError, 'homogeneous input nonlinearity'):
            sim.connect(a, b, weight=0.5, rule=one_to_one, comm='dense')

    def test_heterogeneous_phi_true_summation_connects(self):
        """With ``linear_summation=True`` the receiver applies its own φ to the raw rate,
        so heterogeneous φ is admissible."""
        sim = Simulator(dt=DT * u.ms)
        a = self._lin(sim, linear_summation=True)
        b = self._gauss(sim, linear_summation=True)
        sim.connect(a, b, weight=0.5, rule=one_to_one, comm='dense')  # no raise

    def test_summation_mode_mismatch_raises(self):
        sim = Simulator(dt=DT * u.ms)
        a = self._lin(sim, linear_summation=True)
        b = self._lin(sim, linear_summation=False)
        with self.assertRaisesRegex(ValueError, 'linear_summation must match'):
            sim.connect(a, b, weight=0.5, rule=one_to_one, comm='dense')


if __name__ == '__main__':
    unittest.main()
