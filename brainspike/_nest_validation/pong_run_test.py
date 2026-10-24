# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""§3.10 pong — ``AIPong`` head-to-head harness: the closed game/learn loop is well-formed.

:mod:`examples.nest_like.pong_run` orchestrates already-validated pieces — the persistent
:meth:`Simulator.cont` rollout (``_simulator_cont_test``), the ``host_drive`` clamped input
(``host_drive_test``), and the two learners (``pong_rstdp_test`` / ``pong_dopa_test``). Their
game-by-game trajectories PRNG-diverge from NEST (RL is non-deterministic across simulators),
so — as with the other pong RL tests — this harness test pins the *loop* rather than a
convergence threshold:

* :class:`TestHarnessSmoke` — a bounded R-STDP-vs-dopaminergic run returns well-formed
  per-network performance histories, advances the game (paddles move within bounds, driven by
  the networks), and produces a finite, non-trivial reward signal; and two dopaminergic
  players can train simultaneously — each owns its own Simulator and ``volume_transmitter``,
  lifting NEST's single-global-vt restriction (documented in ``pong_run.py``).
* :class:`TestMakePlayer` — the NEST player codes map to the right learner + noise flag.
* :class:`TestPlotComparison` — the learning-curve artifact is written (when matplotlib is
  available).
"""
import os
import tempfile
import unittest
from unittest import mock

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import examples.nest_like.pong as pong
import examples.nest_like.pong_run as pr


class TestHarnessSmoke(unittest.TestCase):
    """A bounded run drives a well-formed game/learn loop for both learner families."""

    def test_rstdp_vs_dopa_well_formed(self):
        runs = 10
        result = pr.run_comparison(('r', 'd'), runs=runs, seed=0)

        # Two distinct learners controlled the two paddles.
        self.assertIn('R-STDP', result['players'][0])
        self.assertIn('TD', result['players'][1])

        # Per-network performance histories are well-formed (one entry per turn, finite).
        for curve in result['rewards']:
            self.assertEqual(len(curve), runs)
            self.assertTrue(np.isfinite(curve).all())

        # The game advanced one record per turn; paddles stayed on the field (clamped).
        for key in ('ball', 'l_paddle', 'r_paddle'):
            self.assertEqual(len(result[key]), runs)
        for key in ('l_paddle', 'r_paddle'):
            ys = result[key][:, 1]
            self.assertTrue((ys >= 0.0).all() and (ys <= pong.GameOfPong.y_length).all(),
                            f'{key} left the field: {ys}')

        # Score is well-formed (non-negative, no more goals than turns).
        l_score, r_score = result['score']
        self.assertGreaterEqual(l_score, 0)
        self.assertGreaterEqual(r_score, 0)
        self.assertLessEqual(l_score + r_score, runs)

        # The closed loop is live: the networks actually moved a paddle, and the reward
        # machinery produced a non-trivial (nonzero) baseline at least once.
        moved = any(np.ptp(result[k][:, 1]) > 0.0 for k in ('l_paddle', 'r_paddle'))
        self.assertTrue(moved, 'neither paddle moved — the network→paddle loop is dead')
        signal = any(np.abs(c).max() > 0.0 for c in result['rewards'])
        self.assertTrue(signal, 'reward signal never became nonzero — learning loop inactive')

    def test_two_dopaminergic_players_cotrain(self):
        # NEST forbids this (single global volume_transmitter); here each PongNetDopa owns
        # its own Simulator + vt, so two dopaminergic learners train head-to-head.
        runs = 2
        result = pr.run_comparison(('d', 'd'), runs=runs, seed=5)
        self.assertEqual(result['players'], ('clean TD', 'clean TD'))
        for curve in result['rewards']:
            self.assertEqual(len(curve), runs)
            self.assertTrue(np.isfinite(curve).all())


class TestMakePlayer(unittest.TestCase):
    """``make_player`` maps a NEST player code to the right learner class + noise flag."""

    def test_codes_map_to_learners(self):
        brainstate.random.seed(0)
        clean_rstdp = pr.make_player('r')
        self.assertIsInstance(clean_rstdp, pr.PongNetRSTDP)
        self.assertFalse(clean_rstdp.apply_noise)

        noisy_dopa = pr.make_player('dn')
        self.assertIsInstance(noisy_dopa, pr.PongNetDopa)
        self.assertTrue(noisy_dopa.apply_noise)


class TestPlotComparison(unittest.TestCase):
    """The learning-curve comparison figure is written to disk."""

    def test_writes_figure(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest('matplotlib not installed')

        result = {
            'players': ('clean R-STDP', 'clean TD'),
            'rewards': (np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.05, 0.15])),
            'score': (1, 2),
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'cmp.png')
            out = pr.plot_comparison(result, path)
            self.assertEqual(out, path)
            self.assertTrue(os.path.exists(path) and os.path.getsize(path) > 0)


class _FakePlayer:
    """A minimal duck-typed pong learner — exercises AIPong's host loop without spiking."""

    def __init__(self, winning_neuron=0):
        self.winning_neuron = winning_neuron
        self._reward_hist = []

    def set_input_spiketrain(self, cell):
        pass

    def run_turn(self):
        return None

    def apply_synaptic_plasticity(self):
        self._reward_hist.append(np.zeros(20))

    def reset(self):
        pass

    def get_performance_data(self):
        return self._reward_hist, []

    def __repr__(self):
        return 'fake'


class TestAIPongScoring(unittest.TestCase):
    """AIPong books a goal and resets the ball when a paddle misses (harness bookkeeping)."""

    def _score_after_miss(self, *, ball_x, paddle_attr, paddle_y):
        game = pong.GameOfPong()
        game.ball.x_pos = ball_x
        game.ball.y_pos = 0.5
        getattr(game, paddle_attr).y_pos = paddle_y   # parked far from the ball -> a miss
        result = pr.AIPong(_FakePlayer(), _FakePlayer(), game).run_games(max_runs=1)
        return result['score']

    def test_right_scores_when_left_paddle_misses(self):
        # Ball at the left edge, left paddle parked away -> the right player scores.
        self.assertEqual(self._score_after_miss(
            ball_x=0.01, paddle_attr='l_paddle', paddle_y=0.95), (0, 1))

    def test_left_scores_when_right_paddle_misses(self):
        # Ball at the right edge, right paddle parked away -> the left player scores.
        self.assertEqual(self._score_after_miss(
            ball_x=pong.GameOfPong.x_length - 0.01, paddle_attr='r_paddle',
            paddle_y=0.95), (1, 0))


class TestMainCli(unittest.TestCase):
    """``main`` parses args, prints a summary, and writes the figure (heavy run stubbed)."""

    def test_quick_run_glue(self):
        fake = {
            'players': ('clean R-STDP', 'noisy R-STDP'),
            'rewards': (np.array([0.0, 0.1]), np.array([0.0, 0.2])),
            'score': (1, 0),
        }
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'cmp.png')
            argv = ['pong_run.py', '--quick', '--seed', '0', '--out', out,
                    '--players', 'r', 'rn']
            with mock.patch.object(pr, 'run_comparison', return_value=fake), \
                    mock.patch('sys.argv', argv):
                pr.main()
            # When matplotlib is available main writes the figure; otherwise it skips it.
            try:
                import matplotlib  # noqa: F401
                self.assertTrue(os.path.exists(out))
            except ImportError:
                pass


if __name__ == '__main__':
    unittest.main()
