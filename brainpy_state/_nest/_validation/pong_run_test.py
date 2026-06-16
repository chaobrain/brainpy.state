# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""§3.10 pong — ``AIPong`` head-to-head harness: the closed game/learn loop is well-formed.

:mod:`examples.nest.pong_run` orchestrates already-validated pieces — the persistent
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

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import examples.nest.pong as pong
import examples.nest.pong_run as pr


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


if __name__ == '__main__':
    unittest.main()
