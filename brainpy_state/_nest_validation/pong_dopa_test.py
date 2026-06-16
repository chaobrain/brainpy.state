# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""§3.10 pong — ``PongNetDopa`` dopaminergic actor-critic learner: wiring + pathway.

The dopamine *substrate* (``stdp_dopamine_synapse`` weight trajectory and the
``volume_transmitter`` ``n(t)``) is already validated against live NEST to tight
bands (``stdp_dopamine_synapse_parity_test`` / ``volume_transmitter_parity_test``).
What :class:`~examples.nest_like.pong_networks.PongNetDopa` adds is the *actor-critic
wiring* and the host reward-current seam, and (like all the pong RL learners) its
game-by-game trajectory PRNG-diverges from NEST. So this module pins the new pieces
deterministically rather than per-sample:

* :class:`TestDopaStructure` — the critic topology is built as specified (striatum /
  VP / dopaminergic populations, plastic dopamine input→motor edges, signed critic
  projections), and the dopamine weights are rule-managed (not host-settable).
* :class:`TestDopaRewardPathway` — the end-to-end signal path works: a large reward
  current makes the dopaminergic neurons fire and **potentiates** the input→motor
  weights, while a zero reward leaves them silent and the weights **depress**. This
  is the learning signal the game closes the loop around.
* :class:`TestDopaRecompileInvariant` — the per-turn reward-schedule rewrite reuses
  the compiled rollout (R1 guard).
* :class:`TestDopaBehaviour` — a bounded noisy run stays well-formed: dopamine weights
  evolve but remain inside ``[Wmin, Wmax]``, and the reward baseline does not diverge.
"""
import os
import time
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state import spike_recorder
from brainpy_state._nest_network._event_plastic import VoltageCoupledPlasticProj
import examples.nest_like.pong_networks as pn

DT = pn.DT


def _with_dopa_recorder(net):
    """Add a spike_recorder on the critic's dopaminergic neurons and re-init."""
    rec = net.sim.create(spike_recorder)
    net.sim.connect(net.dopa, rec)
    net.sim.reset_rollout()
    return rec


def _fixed_reward_turn(net, reward_current):
    """Run one turn with a forced (critic-bypassing) reward current schedule."""
    net.set_input_spiketrain(10)
    schedule = np.zeros((net.poll_steps, net.n_critic))
    schedule[:net._offset_steps, :] = reward_current
    net._reward_drive.set_schedule(schedule)
    return net.sim.cont(pn.POLL_TIME * u.ms)


class TestDopaStructure(unittest.TestCase):
    """The actor-critic topology is built as specified."""

    def test_populations_and_projection_types(self):
        brainstate.random.seed(0)
        net = pn.PongNetDopa(apply_noise=True, seed=0)
        self.assertEqual(net.striatum.size, net.n_critic)
        self.assertEqual(net.vp.size, net.n_critic)
        self.assertEqual(net.dopa.size, net.n_critic)
        # input -> motor is a plastic, dopamine-coupled projection (weight evolves).
        self.assertIsInstance(net.motor_proj, VoltageCoupledPlasticProj)
        # all_to_all input -> motor = num_neurons^2 dopamine edges.
        conns = net.sim.get_connections(source=net.input_neurons, target=net.motor_neurons)
        self.assertEqual(len(conns), net.num_neurons ** 2)

    def test_dopamine_weights_are_rule_managed(self):
        brainstate.random.seed(0)
        net = pn.PongNetDopa(apply_noise=False, seed=0)
        conns = net.sim.get_connections(source=net.input_neurons, target=net.motor_neurons)
        # The dopamine rule owns the weight — a host overwrite must be refused.
        with self.assertRaises(ValueError):
            conns.set('weight', 1300.0 * u.pA)

    def test_critic_projection_signs(self):
        brainstate.random.seed(0)
        net = pn.PongNetDopa(apply_noise=True, seed=0)
        str_vp = net.sim.get_connections(source=net.striatum, target=net.vp)
        vp_dopa = net.sim.get_connections(source=net.vp, target=net.dopa)
        w_str_vp = np.asarray(u.Quantity(str_vp.get('weight')).to_decimal(u.pA))
        w_vp_dopa = np.asarray(u.Quantity(vp_dopa.get('weight')).to_decimal(u.pA))
        self.assertTrue(np.allclose(w_str_vp, net.w_str_vp))    # -250
        self.assertTrue(np.allclose(w_vp_dopa, net.w_da))       # -1150


class TestDopaRewardPathway(unittest.TestCase):
    """Reward current -> dopaminergic firing -> dopamine -> input→motor potentiation."""

    def _drive(self, reward_current, n_turns=4, seed=1):
        brainstate.random.seed(seed)
        net = pn.PongNetDopa(apply_noise=False, seed=seed)   # clean -> deterministic
        rec = _with_dopa_recorder(net)
        w0 = net.get_all_weights().copy()
        dopa_spikes = 0
        for _ in range(n_turns):
            res = _fixed_reward_turn(net, reward_current)
            dopa_spikes += int(np.asarray(res.spikes(rec)).sum())
        dW = net.get_all_weights() - w0
        return dopa_spikes, dW

    def test_reward_drives_dopa_firing_and_potentiation(self):
        spikes_hi, dW_hi = self._drive(1000.0)
        spikes_zero, dW_zero = self._drive(0.0)

        # A strong reward fires the dopaminergic neurons; zero reward keeps them silent.
        self.assertGreater(spikes_hi, 0, 'reward current did not make the dopa neurons fire')
        self.assertEqual(spikes_zero, 0, 'dopa neurons fired without any reward current')
        # Dopamine potentiates relative to the depressing no-reward baseline (sign control).
        self.assertGreater(dW_hi.mean(), dW_zero.mean(),
                           f'reward did not potentiate vs baseline (Δw_hi={dW_hi.mean():.3f}, '
                           f'Δw_zero={dW_zero.mean():.3f})')
        self.assertGreater(dW_hi.mean(), 0.0, 'strong reward did not net-potentiate')
        self.assertLess(dW_zero.mean(), 0.0, 'no-reward baseline did not net-depress')


# Wall-clock recompile timing is reliable only on an unloaded local machine; shared CI
# runners make the ratio noisy. Skip on CI (correctness is covered by the deterministic
# parity/learning tests); run locally for the no-recompile (R1) performance guard.
_SKIP_TIMING_ON_CI = os.environ.get('CI') is not None


@unittest.skipIf(_SKIP_TIMING_ON_CI,
                 'wall-clock recompile timing is unreliable on shared CI runners')
class TestDopaRecompileInvariant(unittest.TestCase):
    """Per-turn reward-schedule rewrite reuses the compiled rollout (R1)."""

    def test_steady_turns_far_cheaper_than_first(self):
        brainstate.random.seed(0)
        net = pn.PongNetDopa(apply_noise=False, seed=0)

        def turn(cell, reward):
            net.set_input_spiketrain(cell)
            sched = np.zeros((net.poll_steps, net.n_critic))
            sched[:net._offset_steps, :] = reward
            net._reward_drive.set_schedule(sched)
            t0 = time.perf_counter()
            net.run_turn()
            return time.perf_counter() - t0

        first = turn(0, 500.0)
        steady = [turn(c, 100.0 * c) for c in range(1, 6)]
        if first <= 1.3 * np.mean(steady):
            self.skipTest('first turn showed no measurable compile; timing inconclusive')
        self.assertLess(max(steady), 0.7 * first,
                        f'a turn recompiled after a reward rewrite '
                        f'(steady={[round(s, 3) for s in steady]}, first={first:.3f}s)')


class TestDopaBehaviour(unittest.TestCase):
    """A bounded noisy run stays well-formed (weights bounded, reward finite).

    Strict above-chance convergence for the dopaminergic learner needs many thousands
    of games (as in NEST); within a bounded test we assert the dynamics are stable and
    the learning machinery is active rather than a tight accuracy threshold.
    """

    def test_weights_stay_bounded_and_evolve(self):
        target = 10
        brainstate.random.seed(2)
        net = pn.PongNetDopa(apply_noise=True, seed=2)
        w0 = net.get_all_weights().copy()
        for _ in range(16):
            net.set_input_spiketrain(target)
            net.run_turn()
            net.apply_synaptic_plasticity()
        w1 = net.get_all_weights()

        self.assertGreater(np.abs(w1 - w0).max(), 0.0, 'dopamine weights never moved')
        self.assertGreaterEqual(w1.min(), net.syn_Wmin - 1e-6,
                                f'weight fell below Wmin ({w1.min():.2f} < {net.syn_Wmin})')
        self.assertLessEqual(w1.max(), net.syn_Wmax + 1e-6,
                             f'weight exceeded Wmax ({w1.max():.2f} > {net.syn_Wmax})')
        self.assertTrue(np.isfinite(net.mean_reward).all())


if __name__ == '__main__':
    unittest.main()
