# examples/nest_like/pong_run.py
r"""Train two spiking networks to play Pong against each other — NEST §3.10 harness.

A faithful port of NEST's ``pong/run_simulations.py``: two :mod:`pong_networks`
learners (an R-STDP and/or a dopaminergic player) compete on a shared
:class:`~examples.nest_like.pong.GameOfPong`. Each turn the ball's y-cell is presented to
both networks as input; after a 200 ms :meth:`Simulator.cont` turn each network's
most-active motor neuron moves its paddle one cell toward the predicted target, the
game advances, and the score / ball reset follow. Over many turns the reward-driven
plasticity pulls each network's winning neuron toward the ball cell.

The ``__main__`` runs the head-to-head experiment and writes a learning-curve
comparison (mean reward vs game index per network).

**Difference from NEST.** Each network owns its own :class:`Simulator` (and, for the
dopaminergic learner, its own ``volume_transmitter``), so — unlike NEST, where a
single global volume transmitter forbids training two dopaminergic networks at once —
any pairing is allowed here, including ``d`` vs ``dn``.

Run::

    PYTHONPATH=. python examples/nest_like/pong_run.py --quick
    PYTHONPATH=. python examples/nest_like/pong_run.py --players rn dn --runs 200

Reference
---------
Wunderlich T. et al. (2019), Front. Neurosci. 13:260; Potjans, Diesmann & Morrison
(2011), PLoS Comput. Biol. 7(5):e1001133. Original game/R-STDP implementation:
https://github.com/electronicvisions/model-sw-pong
"""
import argparse

import jax
import brainstate
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import examples.nest_like.pong as pong
from examples.nest_like.pong_networks import PongNetRSTDP, PongNetDopa

#: Player codes (as in NEST): r/rn = R-STDP without/with noise, d/dn = dopaminergic.
PLAYER_CHOICES = ('r', 'rn', 'd', 'dn')


def make_player(code, *, seed=0):
    """Build a pong learner from a NEST-style player code (``r``/``rn``/``d``/``dn``)."""
    apply_noise = code.endswith('n')
    if code[0] == 'r':
        return PongNetRSTDP(apply_noise, seed=seed)
    return PongNetDopa(apply_noise, seed=seed)


class AIPong:
    """Runs and records a Pong game between two competing spiking networks.

    Parameters
    ----------
    player1, player2 : PongNetBase
        Networks controlling the left and right paddle respectively.
    game : GameOfPong, optional
        The game instance; a fresh one is created if omitted.
    """

    def __init__(self, player1, player2, game=None):
        self.game = game if game is not None else pong.GameOfPong()
        self.player1 = player1
        self.player2 = player2

    def run_games(self, max_runs=100):
        """Play ``max_runs`` turns; return per-network performance + game history.

        Returns
        -------
        dict
            ``{'players': (repr1, repr2), 'rewards': (hist1, hist2), 'score': (l, r),
            'ball', 'l_paddle', 'r_paddle'}`` — ``rewards`` are per-turn mean-reward
            curves (mean over neurons), the paddle/ball arrays are per-turn positions.
        """
        l_score, r_score = 0, 0
        ball_pos, l_pos, r_pos = [], [], []

        for _ in range(max_runs):
            input_index = self.game.ball.get_cell()[1]
            self.player1.set_input_spiketrain(input_index)
            self.player2.set_input_spiketrain(input_index)

            # Each network advances its own persistent rollout one 200 ms turn.
            self.player1.run_turn()
            self.player2.run_turn()

            for network, paddle in ((self.player1, self.game.l_paddle),
                                    (self.player2, self.game.r_paddle)):
                network.apply_synaptic_plasticity()
                network.reset()
                position_diff = network.winning_neuron - paddle.get_cell()[1]
                if position_diff > 0:
                    paddle.move_up()
                elif position_diff == 0:
                    paddle.dont_move()
                else:
                    paddle.move_down()

            self.game.step()
            ball_pos.append(self.game.ball.get_pos())
            l_pos.append(self.game.l_paddle.get_pos())
            r_pos.append(self.game.r_paddle.get_pos())

            if self.game.result == pong.RIGHT_SCORE:
                self.game.reset_ball(False)
                r_score += 1
            elif self.game.result == pong.LEFT_SCORE:
                self.game.reset_ball(True)
                l_score += 1

        return {
            'players': (repr(self.player1), repr(self.player2)),
            'rewards': (_reward_curve(self.player1), _reward_curve(self.player2)),
            'score': (l_score, r_score),
            'ball': np.array(ball_pos),
            'l_paddle': np.array(l_pos),
            'r_paddle': np.array(r_pos),
        }


def _reward_curve(network):
    """Per-turn performance: the mean (over neurons) reward baseline at each turn."""
    history, _ = network.get_performance_data()
    return np.array([float(np.mean(r)) for r in history])


def run_comparison(player_codes=('r', 'rn'), *, runs=100, seed=0):
    """Train two players head-to-head and return the :meth:`AIPong.run_games` result.

    Seeds the game (global ``numpy``), the network background noise
    (``brainstate.random``), and each network's private RNG, so a given
    ``(player_codes, runs, seed)`` reproduces exactly.
    """
    np.random.seed(seed)
    brainstate.random.seed(seed)
    p1 = make_player(player_codes[0], seed=seed)
    p2 = make_player(player_codes[1], seed=seed + 1)
    return AIPong(p1, p2).run_games(max_runs=runs)


def plot_comparison(result, path):
    """Write a learning-curve comparison (mean reward vs game index) to ``path``."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, curve in zip(result['players'], result['rewards']):
        ax.plot(curve, label=label)
    ax.set_xlabel('game')
    ax.set_ylabel('mean reward')
    ax.set_title(f"pong: {result['players'][0]} vs {result['players'][1]} "
                 f"(score {result['score']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--players', nargs=2, choices=PLAYER_CHOICES, default=['r', 'rn'],
                        help='Two player codes: r/rn = R-STDP -/+ noise, d/dn = dopaminergic.')
    parser.add_argument('--runs', type=int, default=100, help='Number of game turns.')
    parser.add_argument('--quick', action='store_true',
                        help='Bounded CI-sized run (15 turns) for a fast smoke check.')
    parser.add_argument('--seed', type=int, default=0, help='Reproducibility seed.')
    parser.add_argument('--out', type=str, default='examples/nest_like/pong_run.png',
                        help='Output path for the learning-curve comparison figure.')
    args = parser.parse_args()

    runs = 15 if args.quick else args.runs
    print(f"pong: training {args.players[0]} vs {args.players[1]} for {runs} turns "
          f"(seed {args.seed})...")
    result = run_comparison(tuple(args.players), runs=runs, seed=args.seed)
    print(f"  final score (left, right): {result['score']}")
    for label, curve in zip(result['players'], result['rewards']):
        print(f"  {label:>14}: mean reward {curve[0]:.3f} -> {curve[-1]:.3f}")
    try:
        out = plot_comparison(result, args.out)
        print(f"  wrote {out}")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == '__main__':
    main()
