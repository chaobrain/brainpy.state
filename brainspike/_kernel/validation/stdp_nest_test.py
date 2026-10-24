# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate G8: ``stdp_synapse`` weight trajectories vs live NEST (``requires_nest``).

Both sides are real networks with the **same** delay -- no dendritic shift is faked
(the old substrate's ``_nest_validation/_stdp_drive.py`` injected each post spike
``d`` later and disabled the axonal delay; the kernel implements NEST's dendritic
seam itself):

* NEST: ``spike_generator -> parrot_neuron`` (pre), ``spike_generator ->
  iaf_psc_delta`` with a suprathreshold weight (post, an archiving node firing only
  at the chosen times), ``parrot -> post`` through ``stdp_synapse`` with delay
  ``d``, a ``weight_recorder`` on the plastic synapse;
* kernel: ``spike_generator`` (pre, at NEST's recorded pre times), ``LIF`` post
  forced by a ``spike_generator`` driver through its ``delta`` port (fires at
  NEST's recorded post times, asserted), ``pre -> post`` with
  ``plasticity=stdp_synapse(...)`` and the same delay ``d``.

The kernel's observable (lazy) weight after the synapse step of each pre spike is
compared with NEST's ``weight_recorder`` entry of that send, ``CAT_B``. Many pairs
run as independent units of one kernel network (one compile per protocol set).
"""
from __future__ import annotations

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.connectivity import one_to_one
from brainspike._kernel.models import LIF, spike_generator
from brainspike._kernel.network import Network
from brainspike._kernel.plasticity import stdp_synapse
from brainspike._nest_validation import tolerance_conventions as tc
from brainspike._nest_validation.nest_compare import compare_trace, requires_nest

DT = 0.1
PRE_D = 1.0    # NEST: spike_generator -> parrot
POST_D = 1.0   # NEST: driver -> post
DRIVE_W = 1e6
POST_PARAMS = dict(C_m=1.0, tau_m=10.0, t_ref=0.1, E_L=0.0, V_reset=0.0, V_m=0.0, V_th=1e4)

MULT = dict(weight=5.0, Wmax=100.0, lambda_=0.1, alpha=1.0, mu_plus=1.0, mu_minus=1.0,
            tau_plus=20.0, tau_minus=20.0)
ADD = dict(weight=40.0, Wmax=100.0, lambda_=0.05, alpha=1.2, mu_plus=0.0, mu_minus=0.0,
           tau_plus=16.8, tau_minus=33.7)


def nest_pair(syn, pre_want, post_want, T, delay):
    """One NEST run; returns recorded pre/post fire times and the send weights."""
    import nest
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = DT
    parrot = nest.Create('parrot_neuron')
    post = nest.Create('iaf_psc_delta', params={**POST_PARAMS, 'tau_minus': syn['tau_minus']})
    sg_pre = nest.Create('spike_generator',
                         params={'spike_times': sorted(round(p - PRE_D, 4) for p in pre_want)})
    nest.Connect(sg_pre, parrot, syn_spec={'delay': PRE_D})
    if len(post_want):
        sg_post = nest.Create('spike_generator',
                              params={'spike_times': sorted(round(q - POST_D, 4) for q in post_want)})
        nest.Connect(sg_post, post, syn_spec={'weight': DRIVE_W, 'delay': POST_D})
    rec_pre = nest.Create('spike_recorder')
    rec_post = nest.Create('spike_recorder')
    nest.Connect(parrot, rec_pre)
    nest.Connect(post, rec_post)
    wr = nest.Create('weight_recorder')
    nest.CopyModel('stdp_synapse', 'stdp_rec', {'weight_recorder': wr})
    nest.Connect(parrot, post, syn_spec={
        'synapse_model': 'stdp_rec', 'delay': delay, 'weight': syn['weight'], 'Wmax': syn['Wmax'],
        'lambda': syn['lambda_'], 'alpha': syn['alpha'], 'mu_plus': syn['mu_plus'],
        'mu_minus': syn['mu_minus'], 'tau_plus': syn['tau_plus']})
    nest.Simulate(T)
    ev = wr.get('events')
    order = np.argsort(np.asarray(ev['times']))
    return (np.sort(np.asarray(rec_pre.get('events')['times'])),
            np.sort(np.asarray(rec_post.get('events')['times'])),
            np.asarray(ev['weights'])[order], np.asarray(ev['times'])[order])


def kernel_pairs(syn, trains, T, delay, path='auto', timing='lazy'):
    """One kernel network with a unit per ``(pre_fire, post_fire)`` pair.

    Returns
    -------
    W : ndarray, (n_steps, K)
        Observable weight of each unit's plastic edge after every step.
    S : ndarray, (n_steps, K)
        Post spikes per step.
    plan : Plan
    """
    K = len(trains)
    net = Network(dt=DT * u.ms)
    pre = net.add(spike_generator, K, name='pre',
                  params=dict(spike_times=[np.asarray(p, float) * u.ms for p, _ in trains]))
    # the driver's spike reaches the LIF one step later, which fires in that step:
    # stamp of a driver time T is T, emitted at step T/dt - 1, LIF fires at step T/dt,
    # i.e. NEST stamp T + dt -> drive at q - dt for a post stamp q.
    drv = net.add(spike_generator, K, name='drv',
                  params=dict(spike_times=[np.round(np.asarray(q, float) - DT, 6) * u.ms
                                           for _, q in trains]))
    post = net.add(LIF, K, name='post', params=dict(V_th=1e4 * u.mV), spk_reset='hard')
    net.connect(drv, post, one_to_one(), port='delta', weight=DRIVE_W * u.mV, delay=DT * u.ms)
    rule = stdp_synapse(tau_plus=syn['tau_plus'] * u.ms, tau_minus=syn['tau_minus'] * u.ms,
                        lambda_=syn['lambda_'], alpha=syn['alpha'], mu_plus=syn['mu_plus'],
                        mu_minus=syn['mu_minus'], Wmax=syn['Wmax'], path=path, timing=timing)
    proj = net.connect(pre, post, one_to_one(), port='delta', weight=syn['weight'] * u.mV,
                       delay=delay * u.ms, plasticity=rule, name='stdp')
    plan = net.compile('float64')
    prog = plan.runner().program
    pp = plan.projection(proj)
    params = prog.read_params()
    n_steps = int(round(T / DT))
    i_post = plan.population('post').pop.index

    def body(st, n):
        st, cur = prog.step(params, st, n, jnp.zeros((), jnp.uint32))
        return st, (pp.plastic.weight(st['plastic']['stdp']), cur[i_post]['spike'])

    run = brainstate.transform.jit(
        lambda st: brainstate.transform.scan(body, st, jnp.arange(n_steps, dtype=jnp.int32)))
    _, (W, S) = run(prog.init_state())
    # one_to_one: CSR (pre-major) order == unit order
    return np.asarray(W), np.asarray(S), plan


@requires_nest
class TestStdpVsNest(unittest.TestCase):
    """G8: lazy weight at every send equals NEST's weight_recorder (CAT_B)."""

    def _check(self, syn, protocols, T, delay, path='auto'):
        ref = [nest_pair(syn, pre, post, T, delay) for pre, post in protocols]
        trains = [(r[0], r[1]) for r in ref]
        with jax.enable_x64(True):
            W, S, plan = kernel_pairs(syn, trains, T, delay, path=path)
        for k, (pre_f, post_f, w_ref, t_ref) in enumerate(ref):
            label = f'unit {k} delay={delay} protocol={protocols[k]}'
            # the post fired exactly at NEST's times (stamp (n + 1) dt)
            got_post = (np.nonzero(S[:, k])[0] + 1) * DT
            np.testing.assert_allclose(got_post, post_f, atol=1e-9, err_msg=label)
            self.assertEqual(len(w_ref), len(pre_f), label)
            steps = np.rint(t_ref / DT).astype(int) - 1
            compare_trace(w_ref, W[steps, k], tol=tc.CAT_B, metric=label).assert_()
        return W, plan

    def test_pair_window_sweep(self):
        P0, P1 = 100.0, 320.0  # the second pre send records the deferred LTP
        offsets = (-40.0, -20.0, -10.0, -5.0, -1.0, 1.0, 5.0, 10.0, 20.0, 40.0)
        protocols = [([P0, P1], [P0 + o]) for o in offsets]
        for delay in (1.0, 2.5):
            with self.subTest(delay=delay):
                W, plan = self._check(MULT, protocols, P1 + 50.0, delay)
                pp = plan.projection('stdp')
                self.assertEqual((pp.plastic.d_ax, pp.plastic.d_den), (0, int(round(delay / DT))))
                self.assertEqual(pp.plastic.path, 'reference')
                # LTP for positive offsets beyond the dendritic delay, LTD otherwise
                final = W[-1]
                self.assertTrue(np.all(final[np.asarray(offsets) > delay] > MULT['weight']))
                self.assertTrue(np.all(final[np.asarray(offsets) < -1.0] < MULT['weight']))

    def test_post_arrival_coincides_with_pre(self):
        # post fires d before the pre: its arrival at the synapse (q + d) is the pre's
        # step -- NEST facilitates (window (t_last, t_pre]) then depresses with K-
        # strictly before t_pre.
        P0, P1 = 100.0, 320.0
        for delay in (1.0, 2.5):
            with self.subTest(delay=delay):
                protocols = [([P0, P1], [P0 - delay]), ([P0, P1], [P0 - delay, P0 - delay + 7.0]),
                             ([P0, P0 + 30.0, P1], [P0 + 30.0 - delay])]
                self._check(MULT, protocols, P1 + 50.0, delay)

    def test_train_trajectory(self):
        rng = np.random.default_rng(7)
        pre = list(np.arange(20.0, 2000.0, 37.0))
        protocols = []
        for _ in range(3):
            post = np.sort(rng.choice(np.arange(15.0, 2000.0, 0.1), 60, replace=False))
            protocols.append((pre, list(np.round(post, 1))))
        self._check(MULT, protocols, 2050.0, 2.5)

    def test_additive_fast_and_reference_paths(self):
        rng = np.random.default_rng(3)
        pre = list(np.arange(30.0, 1500.0, 23.0))
        post = list(np.round(np.sort(rng.choice(np.arange(20.0, 1500.0, 0.1), 80, replace=False)), 1))
        protocols = [(pre, post), ([100.0, 120.0, 400.0], [105.0, 300.0])]
        Wf, plan_f = self._check(ADD, protocols, 1550.0, 1.5, path='fast')
        Wr, plan_r = self._check(ADD, protocols, 1550.0, 1.5, path='reference')
        self.assertEqual(plan_f.projection('stdp').plastic.path, 'fast')
        self.assertEqual(plan_r.projection('stdp').plastic.path, 'reference')
        np.testing.assert_allclose(Wf, Wr, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
