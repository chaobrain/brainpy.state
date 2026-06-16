# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Live-NEST solve-rate parity for the §3.10 ``sudoku/`` port (``@requires_nest``).

Distributional, never per-sample (the §3.14 posture): the spiking WTA is a *stochastic*
constraint solver, so the bar is matching NEST's own solve behaviour on the same boards,
not a fixed success.

Two regimes, both measured against live NEST (its bundled ``sudoku_net.py``, driven here
with an identical host relaxation loop):

* **Easy / near-complete board** -- both solvers complete it within a couple of chunks;
  brainpy's solve rate must sit within a band of NEST's (both ~1.0).
* **Hard board (puzzle 4, NEST's default)** -- *neither* solver cracks it inside a
  practical chunk budget (NEST's own example plateaus around ratio 0.8-0.93 and does not
  solve it in 100 chunks across seeds). This is the documented partial: we assert brainpy
  relaxes well above chance, like NEST, and pin that both fail to fully solve in-budget.

NEST's bundled example lives outside the package; if its directory is absent the live
checks skip (``@requires_nest`` already gates on importing ``nest`` at all).
"""
import gc
import os
import sys
import unittest

import brainstate
import brainunit as u
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_validation.nest_compare import requires_nest
from examples.nest_like.sudoku_net import SudokuNet, SudokuSolver
from examples.nest_like.sudoku_puzzles import get_puzzle, make_easy_puzzle, validate_solution

NEST_SUDOKU_DIR = ('/mnt/d/codes/githubs/computational_neuroscience/'
                   'nest-simulator/pynest/examples/sudoku')
_HAS_NEST_SUDOKU = os.path.isdir(NEST_SUDOKU_DIR)
_NO_SUDOKU = f'NEST sudoku example not found at {NEST_SUDOKU_DIR}'

EASY_BLANKS = 12
N_SEEDS = 3
EASY_MAX_ITERS = 12
HARD_MAX_ITERS = 10


def _ratio_correct(puzzle, solution):
    _valid, boxes, rows, cols = validate_solution(puzzle, solution)
    return float(boxes.sum() + rows.sum() + cols.sum()) / 27.0


def _brainpy_relax(puzzle, seed, max_iterations):
    """Run brainpy's host loop; return (solved, chunks, best_ratio_correct)."""
    net = SudokuNet(seed=seed)
    np.random.seed(seed)
    brainstate.random.seed(seed)
    net.set_input_config(puzzle)
    net.sim.reset_rollout()
    best = 0.0
    for run in range(max_iterations):
        res = net.sim.cont(100.0 * u.ms)
        sol = net.read_solution(res)
        valid, *_ = validate_solution(puzzle, sol)
        best = max(best, _ratio_correct(puzzle, sol))
        if valid:
            return True, run + 1, best
    return False, max_iterations, best


def _nest_relax(puzzle, seed, max_iterations):
    """Run NEST's bundled sudoku network with the same host loop; same return tuple."""
    import nest
    nest.set_verbosity('M_ERROR')
    if NEST_SUDOKU_DIR not in sys.path:
        sys.path.insert(0, NEST_SUDOKU_DIR)
    import sudoku_net as nest_sudoku

    nest.ResetKernel()
    nest.rng_seed = seed + 1
    np.random.seed(seed)
    net = nest_sudoku.SudokuNet(pop_size=5, input=np.asarray(puzzle), noise_rate=350)
    best = 0.0
    for run in range(max_iterations):
        net.reset_spike_recorders()
        nest.Simulate(100.0)
        trains = net.get_spike_trains()
        sol = np.zeros((9, 9), dtype=np.uint8)
        for r in range(9):
            for c in range(9):
                cs = trains[net.io_indices[r, c]]
                counts = np.array([len(s['times']) for s in cs])
                sol[r, c] = int(np.random.choice(np.flatnonzero(counts == counts.max()))) + 1
        valid, *_ = validate_solution(puzzle, sol)
        best = max(best, _ratio_correct(puzzle, sol))
        if valid:
            return True, run + 1, best
    return False, max_iterations, best


@requires_nest
class TestSolveRateParity(unittest.TestCase):
    """brainpy's solve behaviour stays within a band of live NEST on the same boards."""

    def tearDown(self):
        jax.clear_caches()          # bound the XLA compile cache across repeated builds
        gc.collect()

    @unittest.skipUnless(_HAS_NEST_SUDOKU, _NO_SUDOKU)
    def test_easy_board_solve_rate_within_band_of_nest(self):
        puzzle = make_easy_puzzle(EASY_BLANKS, seed=0)
        bp = [_brainpy_relax(puzzle, s, EASY_MAX_ITERS)[0] for s in range(N_SEEDS)]
        nest = [_nest_relax(puzzle, s, EASY_MAX_ITERS)[0] for s in range(N_SEEDS)]
        bp_rate, nest_rate = np.mean(bp), np.mean(nest)
        # NEST completes a near-complete board essentially every time...
        self.assertGreaterEqual(nest_rate, 0.99, f'NEST easy solve rate {nest_rate}')
        # ...and brainpy must match it within a one-seed band (it solves all seeds here).
        self.assertGreaterEqual(bp_rate, 2 / 3, f'brainpy easy solve rate {bp_rate}')
        self.assertGreaterEqual(bp_rate, nest_rate - 1 / 3,
                                f'brainpy {bp_rate} below NEST band {nest_rate}')

    @unittest.skipUnless(_HAS_NEST_SUDOKU, _NO_SUDOKU)
    def test_hard_puzzle4_is_a_documented_partial_for_both(self):
        puzzle = get_puzzle(4)
        bp_solved, _bp_chunks, bp_best = _brainpy_relax(puzzle, 0, HARD_MAX_ITERS)
        nest_solved, _n_chunks, nest_best = _nest_relax(puzzle, 0, HARD_MAX_ITERS)
        # Neither cracks the hard board in a small budget (NEST's own example does not
        # solve puzzle 4 in 100 chunks): this is the documented partial.
        self.assertFalse(bp_solved, 'brainpy unexpectedly solved puzzle 4 in budget')
        self.assertFalse(nest_solved, 'NEST unexpectedly solved puzzle 4 in budget')
        # But both relax well above chance, and brainpy tracks NEST within a band.
        self.assertGreaterEqual(bp_best, 0.5, f'brainpy puzzle-4 best ratio {bp_best}')
        self.assertGreaterEqual(bp_best, nest_best - 0.3,
                                f'brainpy {bp_best} far below NEST {nest_best}')


if __name__ == '__main__':
    unittest.main()
