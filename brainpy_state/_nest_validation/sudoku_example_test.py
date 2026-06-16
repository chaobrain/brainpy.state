# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-free tests for the §3.10 ``sudoku`` runnable harness (:mod:`examples.nest_like.sudoku`).

The whole harness imports only ``brainpy.state`` + the pure-Python puzzle bank, never
``nest`` -- so these tests *running green without* ``@requires_nest`` is itself the
"standalone, NEST not installed" guarantee: the example solves a board, instruments the
relaxation, renders the figure, and the CLI runs, none of it touching live NEST.
"""
import gc
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from examples.nest_like import sudoku as harness
from examples.nest_like.sudoku_puzzles import get_puzzle, make_easy_puzzle


class TestPureHelpers(unittest.TestCase):
    """The NumPy-only helpers need no network and are exercised directly."""

    def test_ratio_correct_counts_27_constraints(self):
        boxes = np.ones((3, 3), bool)
        rows = np.ones(9, bool)
        cols = np.ones(9, bool)
        self.assertEqual(harness._ratio_correct(boxes, rows, cols), 1.0)
        rows[0] = False                                  # 26/27 satisfied
        self.assertAlmostEqual(harness._ratio_correct(boxes, rows, cols), 26 / 27)

    def test_resolve_puzzle_easy_and_indexed(self):
        easy, easy_label = harness._resolve_puzzle('easy', 12)
        self.assertEqual(int((easy == 0).sum()), 12)
        self.assertIn('easy', easy_label)
        indexed, indexed_label = harness._resolve_puzzle('4', 12)
        np.testing.assert_array_equal(indexed, get_puzzle(4))
        self.assertEqual(indexed_label, 'puzzle 4')

    def test_cell_correct_mask_full_and_broken(self):
        grid = np.array([[((i * 3 + i // 3 + j) % 9) + 1 for j in range(9)]
                         for i in range(9)])
        empty = np.zeros((9, 9), dtype=int)
        self.assertTrue(harness._cell_correct_mask(empty, grid).all())
        broken = grid.copy()
        broken[0, 0] = grid[0, 1]                        # duplicate in row 0 / col 0 / box 0
        mask = harness._cell_correct_mask(empty, broken)
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[8, 8])                      # far corner untouched


class TestSolve(unittest.TestCase):
    """Solve paths over a build-once network."""

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_solve_puzzle_easy_solves_and_records_trajectory(self):
        puzzle = make_easy_puzzle(12, seed=0)
        res = harness.solve_puzzle(puzzle, seed=0, max_iterations=6)
        self.assertTrue(res['valid'])
        self.assertEqual(res['ratio'], 1.0)
        self.assertEqual(len(res['trajectory']), res['chunks'])   # one ratio per chunk
        self.assertEqual(res['solution'].shape, (9, 9))

    def test_solve_puzzle_hard_unsolved_in_one_chunk(self):
        # Puzzle 4 will not solve in a single chunk: exercises the unsolved branch.
        res = harness.solve_puzzle(get_puzzle(4), seed=0, max_iterations=1)
        self.assertFalse(res['valid'])
        self.assertEqual(res['chunks'], 1)
        self.assertLess(res['ratio'], 1.0)

    def test_solve_seeds_reuses_one_net_across_seeds(self):
        # solve_seeds builds ONE net and relaxes every seed on it (the build-once point).
        with mock.patch.object(harness, 'SudokuNet', wraps=harness.SudokuNet) as spy:
            result = harness.solve_seeds(make_easy_puzzle(12, seed=0),
                                         label='easy', seeds=2, max_iterations=6)
        self.assertEqual(spy.call_count, 1)                       # exactly one build
        self.assertEqual(len(result['per_seed']), 2)
        self.assertGreaterEqual(result['solve_rate'], 0.5)
        self.assertIs(result['best'],
                      max(result['per_seed'], key=lambda r: r['ratio']))


class TestFigureAndCLI(unittest.TestCase):
    """The plot and the ``main`` CLI, end to end, with a temp output path."""

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_plot_result_writes_a_png(self):
        result = harness.solve_seeds(make_easy_puzzle(12, seed=0),
                                     label='easy', seeds=1, max_iterations=6)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'grid.png'
            self.assertEqual(harness.plot_result(result, str(out)), str(out))
            self.assertTrue(out.exists() and out.stat().st_size > 0)

    def test_main_quick_writes_figure(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'sudoku.png'
            with mock.patch('sys.argv', ['sudoku.py', '--quick', '--out', str(out)]):
                harness.main()
            self.assertTrue(out.exists())

    def test_main_explicit_args_non_quick_branch(self):
        # Exercises the non-quick CLI path (explicit --puzzle/--seeds/--max-iterations).
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'sudoku.png'
            argv = ['sudoku.py', '--puzzle', 'easy', '--seeds', '1',
                    '--max-iterations', '6', '--out', str(out)]
            with mock.patch('sys.argv', argv):
                harness.main()
            self.assertTrue(out.exists())

    def test_main_skips_plot_when_matplotlib_missing(self):
        # Force the matplotlib-absent branch and confirm main degrades gracefully.
        import io
        from contextlib import redirect_stdout
        with mock.patch.object(harness, 'plot_result', side_effect=ImportError):
            with mock.patch('sys.argv', ['sudoku.py', '--quick', '--out', 'unused.png']):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    harness.main()
        self.assertIn('matplotlib not installed', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
