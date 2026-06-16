# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-free dynamics check for the §3.10 ``sudoku/`` clue clamp.

Confirms the spiking WTA actually *works*: a clamped clue makes its digit win that
cell within a few relaxation chunks, while the rest of the grid stays non-degenerate
(many distinct digits settle -- the clamp does not spuriously pin every cell). This is
the test that would catch the unit-scaling collapse (synaptic PSPs starved 1000x below
the bias current), so it pins the corrected ``pF``/``pA`` parameterization.

No live NEST: the check is on the brainpy network's own emergent behaviour.
"""
import unittest

import brainstate
import brainunit as u
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from examples.nest.sudoku_net import SudokuNet


def _relax(net, puzzle, n_chunks=3, chunk_ms=100.0, seed=0):
    """Clamp ``puzzle`` and relax ``net`` for ``n_chunks``; return the last solution + counts."""
    np.random.seed(seed)
    brainstate.random.seed(seed)
    net.set_input_config(puzzle)
    net.sim.reset_rollout()
    sol, counts = None, None
    for _ in range(n_chunks):
        res = net.sim.cont(chunk_ms * u.ms)
        counts = net.read_counts(res)
        sol = net.read_solution(res)
    return sol, counts


class TestClueClamp(unittest.TestCase):

    def test_single_clue_locks_its_cell(self):
        # A lone clue at (0, 0) = 5 must win its own cell, and its digit population must
        # dominate (the stim drives it and -- via inhibition -- silences the other digits).
        net = SudokuNet(seed=0)
        puzzle = np.zeros((9, 9), dtype=int)
        puzzle[0, 0] = 5
        sol, counts = _relax(net, puzzle, n_chunks=3)
        self.assertEqual(int(sol[0, 0]), 5)
        dc = counts[0, 0]
        self.assertEqual(int(np.argmax(dc)) + 1, 5)
        self.assertGreater(int(dc[4]), 3 * int(np.sort(dc)[-2]))   # clue digit dominates

    def test_clamp_does_not_collapse_the_grid(self):
        # The clue must not spuriously pin every cell to one digit: the un-clued cells
        # relax to a healthy variety (the unit-collapse regression would fail this).
        net = SudokuNet(seed=0)
        puzzle = np.zeros((9, 9), dtype=int)
        puzzle[0, 0] = 5
        sol, _ = _relax(net, puzzle, n_chunks=3)
        self.assertGreaterEqual(len(np.unique(sol)), 4)


if __name__ == '__main__':
    unittest.main()
