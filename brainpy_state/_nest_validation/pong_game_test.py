# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Pure-Python game-logic checks for the §3.10 pong port (``examples/nest/pong.py``).

No spiking machinery and no live NEST: the game is a deterministic 1:1 translation
of NEST's ``pong.py``, so these tests pin the reflection, scoring, grid-cell, and
propagation rules directly against the documented mechanics (seeding ``np.random``
for the stochastic ball reset).
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import examples.nest.pong as pong


class TestPongGame(unittest.TestCase):

    def _game(self):
        np.random.seed(1234)            # make the ball reset deterministic
        return pong.GameOfPong()

    # -- wall reflection ---------------------------------------------------
    def test_upper_wall_reflects_downward(self):
        g = self._game()
        g.ball.x_pos = 0.8              # mid-field (no paddle interaction)
        g.ball.y_pos = 0.99             # within radius of the top edge (>= 1.0 - r)
        g.ball.direction = [0.5, 0.5]   # moving up
        status = g.update_ball_direction()
        self.assertEqual(status, pong.GAME_CONTINUES)
        self.assertLess(g.ball.direction[1], 0.0, 'top wall should send the ball down')

    def test_lower_wall_reflects_upward(self):
        g = self._game()
        g.ball.x_pos = 0.8
        g.ball.y_pos = 0.01             # within radius of the bottom edge
        g.ball.direction = [0.5, -0.5]  # moving down
        g.update_ball_direction()
        self.assertGreater(g.ball.direction[1], 0.0, 'bottom wall should send the ball up')

    # -- paddle collision + scoring ---------------------------------------
    def test_left_paddle_hit_reflects_right(self):
        g = self._game()
        g.ball.x_pos = 0.0              # at the left edge
        g.ball.y_pos = 0.5
        g.ball.direction = [-0.5, 0.2]
        g.l_paddle.y_pos = 0.5          # aligned (|dy| = 0 <= length/2 = 0.1)
        status = g.update_ball_direction()
        self.assertEqual(status, pong.GAME_CONTINUES)
        self.assertGreater(g.ball.direction[0], 0.0, 'paddle hit should reverse x to +')

    def test_left_paddle_miss_scores_right(self):
        g = self._game()
        g.ball.x_pos = 0.0
        g.ball.y_pos = 0.5
        g.ball.direction = [-0.5, 0.2]
        g.l_paddle.y_pos = 0.9          # |0.9 - 0.5| = 0.4 > 0.1 -> miss
        self.assertEqual(g.update_ball_direction(), pong.RIGHT_SCORE)

    def test_right_paddle_hit_and_miss(self):
        g = self._game()
        g.ball.x_pos = g.x_length       # at the right edge
        g.ball.y_pos = 0.3
        g.ball.direction = [0.5, 0.0]
        g.r_paddle.y_pos = 0.3          # aligned -> hit
        self.assertEqual(g.update_ball_direction(), pong.GAME_CONTINUES)
        self.assertLess(g.ball.direction[0], 0.0, 'paddle hit should reverse x to -')

        g.ball.x_pos = g.x_length
        g.ball.direction = [0.5, 0.0]
        g.r_paddle.y_pos = 0.9          # far from y=0.3 -> miss
        self.assertEqual(g.update_ball_direction(), pong.LEFT_SCORE)

    def test_step_scores_when_ball_passes_missing_paddle(self):
        g = self._game()
        g.ball.x_pos = 0.0
        g.ball.y_pos = 0.5
        g.ball.direction = [-0.5, 0.0]
        g.l_paddle.y_pos = 0.95         # miss
        result = g.step()
        self.assertEqual(result, pong.RIGHT_SCORE)
        self.assertEqual(g.result, pong.RIGHT_SCORE)

    # -- grid cell mapping -------------------------------------------------
    def test_get_cell_maps_position_to_grid(self):
        g = self._game()
        g.ball.x_pos, g.ball.y_pos = 0.8, 0.5
        g.ball.update_cell()
        self.assertEqual(g.get_ball_cell(), [16, 10])   # floor(0.8/1.6*32), floor(0.5*20)
        # y-cell stays within the network's 0..19 input range across the field.
        for y in np.linspace(0.0, 0.999, 50):
            g.ball.y_pos = float(y)
            g.ball.update_cell()
            self.assertTrue(0 <= g.get_ball_cell()[1] <= g.y_grid - 1)

    # -- propagation + clamping -------------------------------------------
    def test_propagate_moves_ball_by_velocity_direction(self):
        g = self._game()
        g.ball.x_pos, g.ball.y_pos = 0.5, 0.5
        g.ball.velocity = 0.025
        g.ball.direction = [1.0, 0.0]
        g.l_paddle.dont_move()
        g.r_paddle.dont_move()
        g.propagate_ball_and_paddles()
        self.assertAlmostEqual(g.ball.x_pos, 0.525, places=12)
        self.assertAlmostEqual(g.ball.y_pos, 0.5, places=12)

    def test_paddle_clamped_to_field(self):
        g = self._game()
        g.l_paddle.y_pos = 0.99
        g.l_paddle.move_up()            # direction = +1
        g.propagate_ball_and_paddles()
        self.assertLessEqual(g.l_paddle.y_pos, g.y_length)
        self.assertAlmostEqual(g.l_paddle.y_pos, g.y_length, places=12)

    # -- deterministic reset ----------------------------------------------
    def test_reset_ball_in_field_and_seed_reproducible(self):
        np.random.seed(7)
        g1 = pong.GameOfPong()
        p1 = g1.ball.get_pos()
        np.random.seed(7)
        g2 = pong.GameOfPong()
        p2 = g2.ball.get_pos()
        self.assertEqual(p1, p2)                       # same seed -> same ball
        self.assertTrue(0.0 <= g1.ball.y_pos <= g1.y_length)


if __name__ == '__main__':
    unittest.main()
