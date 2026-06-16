# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Linear-rate **network** fixed-point parity — the goal-15a substrate crux.

The micro-parity arbiter (``rate_coupling_micro_parity_test``) proved a single
feed-forward instantaneous rate connection. This test closes the loop: a
**recurrent** linear-rate network coupled through the seam-(H) continuous
emission must relax to the analytic coupled fixed point

.. math::

    \tau\,\dot r_i = -\lambda r_i + \mu_i + g\,h_i,\qquad
    h_i = \sum_j C_{ij}\,r_j

so at steady state (``lambda=1``)

.. math::

    (I - g\,C)\,r^\* = \mu \;\Longrightarrow\; r^\* = (I - g\,C)^{-1}\mu .

Each off-diagonal ``C_ij`` is realised as one instantaneous rate connection
``pre=j -> post=i`` with weight ``C_ij`` (``comm='dense'``); the post reads the
weighted presynaptic-rate sum back through its **default** delta channel. The
intrinsic one-step pipeline lag is NEST's ``use_wfr=False`` seed: it diverges on
the transient but **preserves the fixed point** (spec §3.4), which is exactly
what the recurrent feedback loop here stresses — the explicit lag breaks the
otherwise-algebraic loop ``r0 <- r1 <- r0``.

Two arbiters:

* ``TestRateNetworkFixedPoint`` (NEST-free, always runs) — the relaxed rates
  match ``(I - gC)^{-1}\mu`` to ``1e-4`` for a 2-neuron mutual loop, a 3-neuron
  mixed-sign feedback net, and a zero-coupling control (``r^\* = mu``).
* ``TestRateNetworkNestParity`` (``@requires_nest``) — the same nets vs live NEST
  (``lin_rate_ipn`` + ``rate_connection_instantaneous``, ``use_wfr=False``);
  full ``rate`` trajectories match within a tight band once ``align_steps``
  absorbs the uniform pipeline offset.
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

from brainpy_state import Simulator, lin_rate_ipn, multimeter, one_to_one
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

DT = 0.1
TAU = 10.0
G = 1.0
T = 1500.0

#: Deterministic exponential-Euler propagator; once ``align_steps`` absorbs the
#: fixed integer pipeline latency the band is tight.
NET_TOL = TraceTolerance(1e-4, 1e-4, align_steps=6, label='C',
                         note='recurrent instantaneous rate network vs live NEST (use_wfr=False)')

# --- the test networks: (label, coupling matrix C, drive mu) --------------------
# C[i, j] is the gain from presynaptic rate r_j into post i's coupling input h_i.
# Zero diagonal (no autapse, spec §5). rho(g*C) < 1 keeps (I - gC) invertible/stable.
_NETS = {
    'mutual_2': (
        np.array([[0.0, 0.5],
                  [0.4, 0.0]]),
        np.array([2.0, -1.0]),
    ),
    'feedback_3_mixed_sign': (
        np.array([[0.0, 0.3, -0.2],
                  [0.1, 0.0, 0.25],
                  [-0.15, 0.2, 0.0]]),
        np.array([2.0, -1.0, 0.5]),
    ),
    'zero_coupling_control': (
        np.zeros((2, 2)),
        np.array([1.5, -0.5]),
    ),
}


def _fixed_point(C, mu, g=G):
    """Analytic coupled fixed point ``r^* = (I - gC)^{-1} mu``."""
    n = C.shape[0]
    return np.linalg.solve(np.eye(n) - g * C, mu)


def _lin_rate_units(mu):
    """brainpy ``lin_rate_ipn`` params (units attached) for a deterministic cell."""
    return dict(tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=float(mu), g=G,
                linear_summation=True,
                rate_initializer=braintools.init.Constant(0.0),
                noise_initializer=braintools.init.Constant(0.0))


def _bp_network(C, mu, *, T=T, dt=DT):
    """N coupled ``lin_rate_ipn`` via the Simulator; returns (samples, N) rate trace."""
    n = C.shape[0]
    sim = Simulator(dt=dt * u.ms)
    pops = [sim.create(lin_rate_ipn, 1, params=_lin_rate_units(mu[i])) for i in range(n)]
    for i in range(n):        # post
        for j in range(n):    # pre
            if i != j and C[i, j] != 0.0:
                sim.connect(pops[j], pops[i], weight=float(C[i, j]),
                            rule=one_to_one, comm='dense')
    mms = [sim.create(multimeter, record_from=('rate',)) for _ in range(n)]
    for i in range(n):
        sim.connect(mms[i], pops[i])
    res = sim.simulate(T * u.ms)
    cols = [np.asarray(u.get_mantissa(res.trace(mms[i], 'rate'))).reshape(-1) for i in range(n)]
    return np.stack(cols, axis=1)


class TestRateNetworkFixedPoint(unittest.TestCase):
    """Recurrent linear-rate networks relax to ``(I - gC)^{-1} mu`` (NEST-free)."""

    def test_networks_reach_analytic_fixed_point(self):
        for label, (C, mu) in _NETS.items():
            with self.subTest(net=label):
                rstar = _fixed_point(C, mu)
                # Guard the test fixtures themselves: must be a stable contraction.
                rho = float(np.max(np.abs(np.linalg.eigvals(G * C))))
                self.assertLess(rho, 1.0, msg=f'{label}: unstable fixture rho(gC)={rho}')
                traj = _bp_network(C, mu)
                final = traj[-1]
                np.testing.assert_allclose(
                    final, rstar, atol=1e-4,
                    err_msg=f'{label}: relaxed rate {final} != (I-gC)^-1 mu {rstar}')

    def test_zero_coupling_reduces_to_drive(self):
        """With ``C = 0`` the network decouples and each rate relaxes to its own ``mu``."""
        C, mu = _NETS['zero_coupling_control']
        traj = _bp_network(C, mu)
        np.testing.assert_allclose(traj[-1], mu, atol=1e-4)

    def test_recurrent_loop_fixed_point_exceeds_feedforward_estimate(self):
        """The 2-neuron mutual loop's fixed point reflects the *closed* feedback.

        A naive one-pass estimate ``r1 ~ mu1 + C10*mu0`` ignores the back-action
        ``r0 <- C01*r1``; the true loop solution differs, and the substrate must
        find the loop solution (not the feed-forward one).
        """
        C, mu = _NETS['mutual_2']
        rstar = _fixed_point(C, mu)
        one_pass = np.array([mu[0] + C[0, 1] * mu[1], mu[1] + C[1, 0] * mu[0]])
        # The loop solution is genuinely different from the feed-forward estimate.
        self.assertGreater(float(np.max(np.abs(rstar - one_pass))), 1e-3)
        traj = _bp_network(C, mu)
        np.testing.assert_allclose(traj[-1], rstar, atol=1e-4)


# --- NEST side --------------------------------------------------------------------

def _nest_network(C, mu, *, T=T, dt=DT):
    """N coupled NEST ``lin_rate_ipn`` (instantaneous, use_wfr=False); (samples, N)."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt, 'use_wfr': False})
    n = C.shape[0]
    # NEST spells the passive-decay rate ``lambda`` (a Python keyword) via string key.
    base = {'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'g': G,
            'linear_summation': True, 'rate': 0.0}
    cells = [nest.Create('lin_rate_ipn', 1, params={**base, 'mu': float(mu[i])}) for i in range(n)]
    for i in range(n):        # post
        for j in range(n):    # pre
            if i != j and C[i, j] != 0.0:
                nest.Connect(cells[j], cells[i],
                             syn_spec={'synapse_model': 'rate_connection_instantaneous',
                                       'weight': float(C[i, j])})
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    for c in cells:
        nest.Connect(mm, c)
    nest.Simulate(T)
    ev = mm.events
    senders = np.asarray(ev['senders'])
    times = np.asarray(ev['times'])
    rate = np.asarray(ev['rate'])
    cols = []
    for c in cells:
        nid = c.tolist()[0]
        m = senders == nid
        order = np.argsort(times[m], kind='stable')
        cols.append(rate[m][order])
    return np.stack(cols, axis=1)


@requires_nest
class TestRateNetworkNestParity(unittest.TestCase):
    """Recurrent instantaneous rate networks match live NEST (use_wfr=False)."""

    def test_network_trajectories_match_nest(self):
        for label, (C, mu) in _NETS.items():
            if not np.any(C):
                continue  # zero-coupling control has no NEST connections to compare
            with self.subTest(net=label):
                bp = _bp_network(C, mu)
                ns = _nest_network(C, mu)
                k = min(bp.shape[0], ns.shape[0])
                for col in range(C.shape[0]):
                    compare_trace(ns[:k, col], bp[:k, col],
                                  tol=NET_TOL, metric=f'{label}:rate[{col}]').assert_()

    def test_nest_steady_state_matches_closed_form(self):
        """Sanity on the NEST side itself: its relaxed rate is ``(I - gC)^{-1} mu``."""
        C, mu = _NETS['feedback_3_mixed_sign']
        ns = _nest_network(C, mu)
        np.testing.assert_allclose(ns[-1], _fixed_point(C, mu), atol=1e-3)


if __name__ == '__main__':
    unittest.main()
