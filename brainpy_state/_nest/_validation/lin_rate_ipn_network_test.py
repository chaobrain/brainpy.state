# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Fixed-point + trajectory parity for the ``lin_rate_ipn_network`` demo (goal 17).

The NEST demo wires an excitatory and an inhibitory ``lin_rate_ipn`` population with
**delayed excitatory** and **instantaneous inhibitory** connections. Random
``fixed_outdegree`` connectivity diverges sample-by-sample between simulators, so the
parity arbiter here is a *small deterministic* (``sigma=0``) E/I net that carries the same
structure -- E-origin **delayed**, I-origin **instantaneous** -- over ``all_to_all``
(no autapses). Both simulators then realise an identical weight matrix ``W`` and relax to
the exact closed form

.. math::

    \tau\,\dot r = -\lambda r + \mu + W r \;\Longrightarrow\;
    r^\* = (\lambda I - W)^{-1}\mu .

The substrate's one-step pipeline lag is NEST's ``use_wfr=False`` seed: it diverges on the
transient but **preserves the fixed point** (the FP needs no alignment; the trajectory needs
``align_steps``). The delay shifts only the transient, so the FP is delay-invariant.

Two classes (cluster-16 house style):

* ``TestLinRateIpnNetworkStructure`` -- NEST-free, always runs (the no-NEST companion):
  analytic FP, zero-coupling control, delay-/dt-invariance of the FP, ``for_loop`` lowering,
  and a standalone smoke run of the example.
* ``TestLinRateIpnNetworkNestParity`` (``@requires_nest``) -- the same deterministic net vs
  live NEST (``lin_rate_ipn`` + ``rate_connection_{delayed,instantaneous}``, ``use_wfr=False``):
  the FP matches the closed form and NEST tightly; the per-neuron trajectories match NEST
  once ``align_steps`` absorbs the uniform pipeline+delay offset.
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainunit as u

try:
    import nest
except Exception:
    nest = None

from brainpy_state import Simulator, lin_rate_ipn, multimeter, all_to_all
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import TraceTolerance

DT = 0.1          # ms resolution
TAU = 10.0        # ms rate time constant
LAM = 1.0         # passive decay lambda
T = 1500.0        # ms relaxation horizon (150 tau -> fully converged)
NE, NI = 2, 2     # tiny deterministic E / I populations
W_E = 0.05        # excitatory connection weight
G = 5.0           # inhibitory / excitatory weight ratio (w_i = -G*W_E = -0.25)
MU = 2.0          # mean drive of every neuron
D_TEST = 0.5      # ms delay of the (delayed) excitatory connections -> 5 steps

#: Trajectory band: tight once ``align_steps`` (>= 1 pipeline + 5 delay steps) absorbs the
#: uniform integer offset between the NEST multimeter phase and the substrate capture.
TRAJ_TOL = TraceTolerance(1e-4, 1e-4, align_steps=12, label='C',
                          note='lin_rate_ipn_network E/I trajectory vs live NEST (use_wfr=False)')


def _weight_matrix():
    r"""``W[i, j]`` = weight of the edge pre ``j`` -> post ``i`` (0 diagonal).

    Neuron order is ``[E0, E1, I0, I1]``. Excitatory presynaptic columns carry ``W_E``;
    inhibitory columns carry ``-G*W_E``. This is the realised matrix of the ``all_to_all``
    (no-autapse) net both simulators build, so ``(lambda I - W)^{-1} mu`` is the shared
    closed-form fixed point.
    """
    n = NE + NI
    is_e = np.array([True] * NE + [False] * NI)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i, j] = W_E if is_e[j] else -G * W_E
    return W


def _fixed_point(W):
    """Analytic linear-rate fixed point ``r^* = (lambda I - W)^{-1} mu``."""
    n = W.shape[0]
    return np.linalg.solve(LAM * np.eye(n) - W, np.full(n, MU))


def _bp_net(*, delayed=True, sigma=0.0, T=T, dt=DT):
    """Build the small deterministic E/I net; return the ``(samples, NE+NI)`` rate trace.

    E-origin connections are delayed (``delay = D_TEST`` when ``delayed``); I-origin
    connections are always instantaneous. Neurons are ordered ``[E..., I...]`` to match
    :func:`_weight_matrix`.
    """
    brainstate.random.seed(0)
    npar = dict(tau=TAU * u.ms, mu=MU, sigma=sigma, lambda_=LAM, g=1.0,
                linear_summation=True)
    sim = Simulator(dt=dt * u.ms)
    e = sim.create(lin_rate_ipn, NE, params=npar)
    i = sim.create(lin_rate_ipn, NI, params=npar)
    de = (D_TEST * u.ms) if delayed else None
    sim.connect(e, e, weight=W_E, delay=de, rule=all_to_all, comm='dense',
                allow_autapses=False)
    sim.connect(e, i, weight=W_E, delay=de, rule=all_to_all, comm='dense')
    sim.connect(i, e, weight=-G * W_E, rule=all_to_all, comm='dense')        # instantaneous
    sim.connect(i, i, weight=-G * W_E, rule=all_to_all, comm='dense',
                allow_autapses=False)
    mm_e = sim.create(multimeter, record_from=('rate',))
    mm_i = sim.create(multimeter, record_from=('rate',))
    sim.connect(mm_e, e)
    sim.connect(mm_i, i)
    res = sim.simulate(T * u.ms)
    re = np.asarray(u.get_mantissa(res.trace(mm_e, 'rate'))).reshape(-1, NE)
    ri = np.asarray(u.get_mantissa(res.trace(mm_i, 'rate'))).reshape(-1, NI)
    return np.concatenate([re, ri], axis=1)


class TestLinRateIpnNetworkStructure(unittest.TestCase):
    """The E/I linear-rate net relaxes to ``(lambda I - W)^{-1} mu`` (NEST-free)."""

    def test_relaxes_to_analytic_fixed_point(self):
        W = _weight_matrix()
        rho = float(np.max(np.abs(np.linalg.eigvals(W / LAM))))
        self.assertLess(rho, 1.0, msg=f'unstable fixture rho(W/lambda)={rho}')
        final = _bp_net(delayed=True)[-1]
        np.testing.assert_allclose(final, _fixed_point(W), atol=1e-4,
                                   err_msg=f'relaxed {final} != (lambda I - W)^-1 mu')

    def test_zero_coupling_control(self):
        """No connections: every rate relaxes to its own ``mu / lambda``."""
        brainstate.random.seed(0)
        sim = Simulator(dt=DT * u.ms)
        p = sim.create(lin_rate_ipn, 3,
                       params=dict(tau=TAU * u.ms, mu=MU, sigma=0.0, lambda_=LAM, g=1.0))
        mm = sim.create(multimeter, record_from=('rate',))
        sim.connect(mm, p)
        res = sim.simulate(T * u.ms)
        final = np.asarray(u.get_mantissa(res.trace(mm, 'rate'))).reshape(-1, 3)[-1]
        np.testing.assert_allclose(final, MU / LAM, atol=1e-4)

    def test_fixed_point_is_delay_invariant(self):
        """The excitatory delay shifts only the transient; the fixed point is unchanged."""
        delayed = _bp_net(delayed=True)[-1]
        instant = _bp_net(delayed=False)[-1]
        np.testing.assert_allclose(delayed, instant, atol=1e-4)

    def test_fixed_point_dt_invariant(self):
        """The fixed point does not depend on the integration step."""
        a = _bp_net(delayed=True, dt=0.1)[-1]
        b = _bp_net(delayed=True, dt=0.05)[-1]
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_network_lowers_under_for_loop(self):
        """The whole net runs through ``simulate`` (one ``for_loop``) with finite output."""
        traj = _bp_net(delayed=True, T=50.0)
        self.assertEqual(traj.shape[1], NE + NI)
        self.assertTrue(np.all(np.isfinite(traj)))

    def test_example_run_smoke(self):
        """The example script runs standalone (no NEST) and returns finite rate traces."""
        from examples.nest.lin_rate_ipn_network import run
        rate_e0, rate_i0, times, rate_e, rate_i = run(order=5, T=30.0)
        self.assertTrue(np.all(np.isfinite(rate_e0)) and np.all(np.isfinite(rate_i0)))
        self.assertEqual(rate_e.shape[1], 20)   # NE = 4 * order = 20
        self.assertEqual(rate_i.shape[1], 5)    # NI = order = 5


# --- NEST side --------------------------------------------------------------------

def _nest_net(*, delayed=True, T=T, dt=DT):
    """The same deterministic E/I net in live NEST (``use_wfr=False``); ``(samples, 4)``."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt, 'use_wfr': False})
    base = {'tau': TAU, 'lambda': LAM, 'sigma': 0.0, 'g': 1.0,
            'linear_summation': True, 'rate': 0.0, 'mu': MU}
    e = nest.Create('lin_rate_ipn', NE, params=base)
    i = nest.Create('lin_rate_ipn', NI, params=base)
    if delayed:
        syn_e = {'synapse_model': 'rate_connection_delayed', 'weight': W_E, 'delay': D_TEST}
    else:
        syn_e = {'synapse_model': 'rate_connection_instantaneous', 'weight': W_E}
    syn_i = {'synapse_model': 'rate_connection_instantaneous', 'weight': -G * W_E}
    aa_no_auto = {'rule': 'all_to_all', 'allow_autapses': False}
    nest.Connect(e, e, aa_no_auto, syn_e)
    nest.Connect(e, i, 'all_to_all', syn_e)
    nest.Connect(i, e, 'all_to_all', syn_i)
    nest.Connect(i, i, aa_no_auto, syn_i)
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    nest.Connect(mm, e + i)
    nest.Simulate(T)
    ev = mm.events
    senders = np.asarray(ev['senders'])
    times = np.asarray(ev['times'])
    rate = np.asarray(ev['rate'])
    cols = []
    for nid in (e + i).tolist():
        m = senders == nid
        order = np.argsort(times[m], kind='stable')
        cols.append(rate[m][order])
    return np.stack(cols, axis=1)


@requires_nest
class TestLinRateIpnNetworkNestParity(unittest.TestCase):
    """The deterministic E/I net matches live NEST (``use_wfr=False``)."""

    def test_fixed_point_matches_nest(self):
        W = _weight_matrix()
        cf = _fixed_point(W)
        bp = _bp_net(delayed=True)[-1]
        ns = _nest_net(delayed=True)[-1]
        # NEST itself relaxes to the closed form...
        np.testing.assert_allclose(ns, cf, atol=1e-3,
                                   err_msg=f'NEST {ns} != closed form {cf}')
        # ...and brainpy matches NEST (and hence the closed form).
        np.testing.assert_allclose(bp, ns, atol=1e-3,
                                   err_msg=f'brainpy {bp} != NEST {ns}')

    def test_trajectory_matches_nest(self):
        bp = _bp_net(delayed=True)
        ns = _nest_net(delayed=True)
        k = min(bp.shape[0], ns.shape[0])
        for col in range(NE + NI):
            compare_trace(ns[:k, col], bp[:k, col], tol=TRAJ_TOL,
                          metric=f'rate[{col}]').assert_()


if __name__ == '__main__':
    unittest.main()
