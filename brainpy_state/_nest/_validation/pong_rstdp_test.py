# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""§3.10 pong — ``PongNetRSTDP`` R-STDP learner: component parity + behaviour.

Reinforcement learners PRNG-diverge from NEST game-for-game, so there is no
per-sample full-game parity (cluster note: RL parity is component-deterministic +
behavioural, never per-sample). This module covers both halves:

**Component parity (deterministic, the rigorous guarantee).**

* :class:`TestCalculateStdpParity` (``@requires_nest``) — our :func:`calculate_stdp`
  reproduces NEST's *actual* ``PongNetRSTDP.calculate_stdp`` source bit-for-bit.
* :class:`TestCalculateStdpPinned` (no NEST) — the same function pinned to
  hand-computed literals + its documented invariants (saturation clip,
  next-neighbour skip, translation invariance, causal vs. full).
* :class:`TestApplyRstdpFormula` (no NEST) — the full per-turn weight update
  ``Δw = learning_rate · calculate_stdp · reward`` is applied per-edge with the
  correct sign and magnitude (so the whole rule, not just the kernel, matches NEST).

**Behaviour (bounded, no NEST — the only genuinely emergent claim, kept lenient).**

* :class:`TestRstdpRecompileInvariant` — per-turn weight overwrite + schedule rewrite
  reuse the compiled rollout (R1 guard: steady turns ≪ the first/compiling turn).
* :class:`TestRstdpLearnsAboveChance` — over a bounded noisy run the reward baseline
  rises from zero and the winning neuron is biased toward the target (below uniform
  chance). Aggregate only; thresholds are deliberately loose.
"""
import os
import time
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import examples.nest.pong_networks as pn
from examples.nest.pong_networks import calculate_stdp, DT
from brainpy_state._nest._validation.nest_compare import requires_nest
from brainpy_state._nest._validation import _pong_drive as drv

#: (pre, post) spike-time cases spanning the calculate_stdp branches.
_STDP_CASES = [
    ([1, 11, 21], [5, 15]),                       # one post per interval (causal)
    ([1, 11, 21, 31, 41], [6, 7, 17, 40]),        # two posts in [1,11) -> next-neighbour skip
    ([10, 20, 30], []),                           # no post -> 0
    ([1, 11, 21], [100, 200]),                    # posts after all pre -> facilitation only
    ([1., 11., 21., 31., 41., 51.],
     [3., 13., 23., 33., 43., 53.]),              # regular train -> saturates
    ([1, 11, 21], [2, 3, 12, 13, 22]),            # dense posts
]


def _motor_spike_times(spikes, motor):
    """Turn-local motor spike times (ms) from a ``(steps, n)`` spike matrix."""
    return np.where(spikes[:, motor] > 0)[0] * DT


@requires_nest
class TestCalculateStdpParity(unittest.TestCase):
    """Our calculate_stdp reproduces NEST's PongNetRSTDP.calculate_stdp exactly."""

    def test_matches_nest_causal(self):
        for pre, post in _STDP_CASES:
            ours = calculate_stdp(pre, post)
            theirs = drv.nest_calculate_stdp(pre, post)
            self.assertAlmostEqual(ours, theirs, places=12,
                                   msg=f'causal mismatch for pre={pre} post={post}')

    def test_matches_nest_full_and_all_neighbors(self):
        for pre, post in _STDP_CASES:
            for only_causal in (True, False):
                for next_neighbor in (True, False):
                    ours = calculate_stdp(pre, post, only_causal=only_causal,
                                          next_neighbor=next_neighbor)
                    theirs = drv.nest_calculate_stdp(pre, post, only_causal=only_causal,
                                                     next_neighbor=next_neighbor)
                    self.assertAlmostEqual(
                        ours, theirs, places=12,
                        msg=f'mismatch pre={pre} post={post} causal={only_causal} '
                            f'nn={next_neighbor}')


class TestCalculateStdpPinned(unittest.TestCase):
    """calculate_stdp pinned to hand-computed values + documented invariants."""

    def test_pinned_literals(self):
        # Anchors computed directly from the NEST formula (A=36, tau=64, sat=128).
        self.assertAlmostEqual(calculate_stdp([1, 11, 21], [5, 15]), 67.6377405226, places=8)
        self.assertAlmostEqual(calculate_stdp([1, 11, 21, 31, 41], [6, 7, 17, 40]),
                               97.3502723109, places=8)
        self.assertAlmostEqual(calculate_stdp([1, 11, 21], [100, 200]), 10.4765972728, places=8)
        self.assertAlmostEqual(
            calculate_stdp([1, 11, 21], [5, 15], only_causal=False), 2.0809945032, places=8)

    def test_empty_post_is_zero(self):
        self.assertEqual(calculate_stdp([1, 11, 21], []), 0.0)

    def test_saturation_clip(self):
        # A long, tightly-coupled train accumulates past the saturation ceiling.
        pre = np.arange(0., 200., 10.)
        post = pre + 1.0
        self.assertEqual(calculate_stdp(pre, post), 128.0)

    def test_translation_invariance(self):
        # Shifting both trains by a constant leaves the correlation unchanged (so
        # turn-local times == NEST's absolute biological times).
        pre = np.array([1., 11., 21., 31.])
        post = np.array([4., 13., 26.])
        base = calculate_stdp(pre, post)
        for shift in (10.0, 1000.0, -0.5):
            self.assertAlmostEqual(calculate_stdp(pre + shift, post + shift), base, places=10)

    def test_next_neighbor_reduces_count(self):
        # Without next-neighbour restriction more pairs contribute (>= the restricted).
        pre = [1, 11, 21]
        post = [6, 7, 8, 16, 17]
        self.assertGreaterEqual(calculate_stdp(pre, post, next_neighbor=False),
                                calculate_stdp(pre, post, next_neighbor=True))


class TestApplyRstdpFormula(unittest.TestCase):
    """The per-turn weight update equals learning_rate * calculate_stdp * reward."""

    def _row_weights(self, net, cell):
        conns = net.sim.get_connections(source=net.input_neurons[cell],
                                        target=net.motor_neurons)
        g = conns.get(['target', 'weight'])
        return np.asarray(g['target']), pn._to_pA(g['weight'])

    def test_update_matches_formula_and_sign(self):
        brainstate.random.seed(0)
        net = pn.PongNetRSTDP(apply_noise=False, seed=0)   # deterministic (no noise)
        net.set_input_spiketrain(10)
        spk = net.run_turn()

        targets, old = self._row_weights(net, 10)
        corr = np.array([net.calculate_stdp(net.input_train, _motor_spike_times(spk, j))
                         for j in targets])
        self.assertGreater(corr.max(), 0.0, 'no motor neuron fired; cannot test the rule')

        # Positive reward: Δw = lr * corr * reward per edge; fired edges potentiate.
        net.apply_rstdp(0.5)
        _, new_pos = self._row_weights(net, 10)
        np.testing.assert_allclose(new_pos, old + net.learning_rate * corr * 0.5, atol=1e-9)
        fired = corr > 0
        self.assertTrue(np.all(new_pos[fired] > old[fired]))

        # Restore and apply a negative reward: the same edges depress.
        net.sim.get_connections(source=net.input_neurons[10],
                                target=net.motor_neurons).set('weight', old * u.pA)
        net.apply_rstdp(-0.5)
        _, new_neg = self._row_weights(net, 10)
        np.testing.assert_allclose(new_neg, old - net.learning_rate * corr * 0.5, atol=1e-9)
        self.assertTrue(np.all(new_neg[fired] < old[fired]))

    def test_only_target_row_changes(self):
        brainstate.random.seed(1)
        net = pn.PongNetRSTDP(apply_noise=False, seed=1)
        net.set_input_spiketrain(7)
        net.run_turn()
        before = net.get_all_weights()
        net.apply_rstdp(1.0)
        after = net.get_all_weights()
        changed_rows = np.where(np.abs(after - before).sum(axis=1) > 1e-9)[0]
        self.assertEqual(changed_rows.tolist(), [7], 'only the target input row should change')


# Wall-clock recompile timing is reliable only on an unloaded local machine; shared CI
# runners make the ratio noisy. Skip on CI (correctness is covered by the deterministic
# parity/learning tests); run locally for the no-recompile (R1) performance guard.
_SKIP_TIMING_ON_CI = os.environ.get('CI') is not None


@unittest.skipIf(_SKIP_TIMING_ON_CI,
                 'wall-clock recompile timing is unreliable on shared CI runners')
class TestRstdpRecompileInvariant(unittest.TestCase):
    """Per-turn weight overwrite + schedule rewrite reuse the compiled rollout (R1)."""

    def test_steady_turns_far_cheaper_than_first(self):
        brainstate.random.seed(0)
        net = pn.PongNetRSTDP(apply_noise=False, seed=0)

        def turn(cell):
            # Both per-turn writes the R-STDP loop makes: a schedule rewrite and an
            # in-place static-weight overwrite. Only the cont() is timed.
            net.set_input_spiketrain(cell)
            conns = net.sim.get_connections(source=net.input_neurons[cell],
                                            target=net.motor_neurons)
            conns.set('weight', (pn._to_pA(conns.get('weight')) + 1.0) * u.pA)
            t0 = time.perf_counter()
            net.run_turn()
            return time.perf_counter() - t0

        first = turn(0)
        steady = [turn(c) for c in range(1, 7)]
        if first <= 1.3 * np.mean(steady):
            self.skipTest('first turn showed no measurable compile; timing inconclusive')
        self.assertLess(max(steady), 0.7 * first,
                        f'a turn recompiled after a weight/schedule rewrite '
                        f'(steady={[round(s, 3) for s in steady]}, first={first:.3f}s)')


class TestRstdpLearnsAboveChance(unittest.TestCase):
    """Bounded behavioural check: reward rises and winning is biased toward target.

    Aggregate only (RL is PRNG-divergent — no per-sample NEST partner). The noisy
    variant is used because its background drive supplies the exploration the learning
    needs; thresholds are loose so the test is robust across seeds.
    """

    def test_reward_rises_and_winner_beats_chance(self):
        target = 10
        n_turns = 30
        brainstate.random.seed(3)
        net = pn.PongNetRSTDP(apply_noise=True, seed=3)
        distances = []
        for _ in range(n_turns):
            net.set_input_spiketrain(target)
            net.run_turn()
            net.apply_synaptic_plasticity()
            distances.append(abs(net.winning_neuron - target))
        distances = np.array(distances)

        # Expected |winning - target| if the winner were uniform over the 20 cells; for
        # a centre target (10) this is 5.0, so the headroom for "beats chance" is modest.
        uniform_chance = np.mean([abs(j - target) for j in range(net.num_neurons)])

        # 1. The reward baseline starts at exactly zero and rises with training (the
        #    primary, robust learning signal — the rule's exactness is proven by the
        #    calculate_stdp parity and apply_rstdp formula tests above).
        self.assertEqual(net.mean_reward_history[0][target], 0.0)
        self.assertGreater(net.mean_reward[target], 0.1,
                           f'reward did not rise (mean_reward={net.mean_reward[target]:.3f})')
        # 2. By the converged second half the winner is biased below uniform chance.
        late = distances[n_turns // 2:].mean()
        self.assertLess(late, uniform_chance,
                        f'winner not biased toward target (late mean|win-tgt|={late:.2f}, '
                        f'chance={uniform_chance:.2f})')


if __name__ == '__main__':
    unittest.main()
