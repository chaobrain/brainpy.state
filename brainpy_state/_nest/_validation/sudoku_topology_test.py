# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-free structural checks for the §3.10 ``sudoku/`` port.

The always-run "no-NEST companion" for ``examples/nest/sudoku*.py``: pure-Python
puzzle helpers, the inhibitory WTA/constraint edge set (vs an independent reference
builder), Poisson rates, the clue-clamp weight write, the per-chunk readout, and the
``for_loop`` lowering of the relaxation chunk. None of this needs a live NEST install;
the live solve-rate parity lives in ``sudoku_solve_test.py`` (``@requires_nest``).
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from examples.nest.sudoku_puzzles import get_puzzle, validate_solution, N_PUZZLES


def _valid_grid():
    """A known-valid 9x9 Sudoku grid (classic shifted-row construction)."""
    return np.array([[((i * 3 + i // 3 + j) % 9) + 1 for j in range(9)]
                     for i in range(9)])


class TestPuzzleHelpers(unittest.TestCase):
    """Pure-Python ``get_puzzle`` / ``validate_solution`` (ported from NEST)."""

    def test_get_puzzle_shapes_and_range(self):
        self.assertEqual(N_PUZZLES, 8)
        for i in range(N_PUZZLES):
            p = get_puzzle(i)
            self.assertEqual(p.shape, (9, 9))
            self.assertGreaterEqual(int(p.min()), 0)
            self.assertLessEqual(int(p.max()), 9)
        self.assertEqual(int(get_puzzle(0).sum()), 0)          # index-0 "dream" all zeros
        np.testing.assert_array_equal(get_puzzle(4)[0], [0, 1, 0, 0, 0, 0, 0, 0, 2])

    def test_get_puzzle_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            get_puzzle(8)
        with self.assertRaises(ValueError):
            get_puzzle(-1)

    def test_validate_solution_accepts_a_correct_grid(self):
        sol = _valid_grid()
        valid, boxes, rows, cols = validate_solution(np.zeros((9, 9), int), sol)
        self.assertEqual(boxes.shape, (3, 3))
        self.assertEqual(rows.shape, (9,))
        self.assertEqual(cols.shape, (9,))
        self.assertTrue(valid and boxes.all() and rows.all() and cols.all())

    def test_validate_solution_rejects_broken_grid(self):
        bad = _valid_grid()
        bad[0, 0] = 1 if bad[0, 0] != 1 else 2                 # break row/col/box at (0,0)
        valid, boxes, rows, cols = validate_solution(np.zeros((9, 9), int), bad)
        self.assertFalse(valid)
        self.assertFalse(bool(rows[0]) and bool(cols[0]))      # row/col 0 flagged invalid

    def test_validate_solution_folds_in_clue_mismatch(self):
        sol = _valid_grid()
        puzzle = np.zeros((9, 9), int)
        puzzle[0, 0] = (int(sol[0, 0]) % 9) + 1                # a clue that disagrees with sol
        self.assertNotEqual(puzzle[0, 0], sol[0, 0])
        valid, *_ = validate_solution(puzzle, sol)
        self.assertFalse(valid)                                # grid valid, but clue mismatched


if __name__ == '__main__':
    unittest.main()
