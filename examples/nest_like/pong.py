# examples/nest_like/pong.py
r"""Classic game of Pong — pure-Python game logic (NEST §3.10 ``pong/`` port).

A faithful translation of NEST's ``pynest/examples/pong/pong.py``: a 1.6 x 1.0
playing field discretised into a 32 x 20 grid, a ball, and two paddles. The
spiking learners in ``pong_networks.py`` observe the ball's y-cell
(``get_ball_cell()[1]`` in ``0..19``) and move their paddle one cell toward the
predicted target each turn.

No spiking machinery here — this module is plain ``numpy`` so the game can be
stepped from the host loop between ``Simulator.cont()`` chunks. The env preamble is
kept for convention (all §3.10 examples set float64/CPU at import).

Reference
---------
Wunderlich T. et al. (2019), *Demonstrating advantages of neuromorphic
computation: a pilot study.* Front. Neurosci. 13:260.
Original implementation: https://github.com/electronicvisions/model-sw-pong
"""
import jax
import brainstate
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

LEFT_SCORE = -1
RIGHT_SCORE = +1
GAME_CONTINUES = 0

MOVE_DOWN = -1
MOVE_UP = +1
DONT_MOVE = 0


class GameObject:
    """Base class for :class:`Ball` and :class:`Paddle` — a positioned object."""

    def __init__(self, game, x_pos=0.5, y_pos=0.5, velocity=0.2, direction=(0, 0)):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.velocity = velocity
        self.direction = direction
        self.game = game
        self.update_cell()

    def get_cell(self):
        return self.cell

    def get_pos(self):
        return self.x_pos, self.y_pos

    def update_cell(self):
        """Update the (x, y) grid cell from the continuous position."""
        x_cell = int(np.floor((self.x_pos / self.game.x_length) * self.game.x_grid))
        y_cell = int(np.floor((self.y_pos / self.game.y_length) * self.game.y_grid))
        self.cell = [x_cell, y_cell]


class Ball(GameObject):
    """The ball. ``radius`` is in unit length; see :class:`GameObject` for the rest."""

    def __init__(self, game, x_pos=0.8, y_pos=0.5, velocity=0.025,
                 direction=(-1 / 2.0, 1 / 2.0), radius=0.025):
        super().__init__(game, x_pos, y_pos, velocity, direction)
        self.ball_radius = radius
        self.update_cell()


class Paddle(GameObject):
    """A paddle on the left (``left=True``) or right end of the field."""

    length = 0.2  # paddle length in units of GameOfPong.y_length

    def __init__(self, game, left, y_pos=0.5, velocity=0.05, direction=0):
        x_pos = 0.0 if left else game.x_length
        super().__init__(game, x_pos, y_pos, velocity, direction)
        self.update_cell()

    def move_up(self):
        self.direction = MOVE_UP

    def move_down(self):
        self.direction = MOVE_DOWN

    def dont_move(self):
        self.direction = DONT_MOVE


class GameOfPong(object):
    """A game of Pong on a 1.6 x 1.0 field discretised into ``x_grid`` x ``y_grid``."""

    x_grid = 32
    y_grid = 20
    x_length = 1.6
    y_length = 1.0

    def __init__(self):
        self.r_paddle = Paddle(self, False)
        self.l_paddle = Paddle(self, True)
        self.reset_ball()
        self.result = 0

    def reset_ball(self, towards_left=False):
        """Re-centre the ball after a goal with a random direction and y position."""
        initial_vx = 0.5 + 0.5 * np.random.random()
        initial_vy = 1.0 - initial_vx
        if towards_left:
            initial_vx *= -1
        initial_vy *= np.random.choice([-1.0, 1.0])

        self.ball = Ball(self, direction=[initial_vx, initial_vy])
        self.ball.y_pos = np.random.random() * self.y_length

    def update_ball_direction(self):
        """Reflect off walls/paddles; return the score (or GAME_CONTINUES)."""
        if self.ball.y_pos + self.ball.ball_radius >= self.y_length:
            # Ball on upper edge
            self.ball.direction[1] = -1 * abs(self.ball.direction[1])
        elif self.ball.y_pos - self.ball.ball_radius <= 0:
            # Ball on lower edge
            self.ball.direction[1] = abs(self.ball.direction[1])

        if self.ball.x_pos - self.ball.ball_radius <= 0:
            # Ball on left edge
            if abs(self.l_paddle.y_pos - self.ball.y_pos) <= Paddle.length / 2:
                self.ball.direction[0] = abs(self.ball.direction[0])
            else:
                return RIGHT_SCORE
        elif self.ball.x_pos + self.ball.ball_radius >= self.x_length:
            # Ball on right edge
            if abs(self.r_paddle.y_pos - self.ball.y_pos) <= Paddle.length / 2:
                self.ball.direction[0] = -1 * abs(self.ball.direction[0])
            else:
                return LEFT_SCORE
        return GAME_CONTINUES

    def propagate_ball_and_paddles(self):
        """Advance ball and paddles by ``velocity * direction`` (paddles clamped)."""
        for paddle in [self.r_paddle, self.l_paddle]:
            paddle.y_pos += paddle.direction * paddle.velocity
            paddle.y_pos = min(max(0, paddle.y_pos), self.y_length)
            paddle.update_cell()
        self.ball.y_pos += self.ball.velocity * self.ball.direction[1]
        self.ball.x_pos += self.ball.velocity * self.ball.direction[0]
        self.ball.update_cell()

    def get_ball_cell(self):
        return self.ball.get_cell()

    def step(self):
        """One game step: collide, propagate, return the new game state."""
        ball_status = self.update_ball_direction()
        self.propagate_ball_and_paddles()
        self.result = ball_status
        return ball_status
