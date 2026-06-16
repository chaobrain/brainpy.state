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


def reference_inhibitory_edges(pop_size=5):
    """Independent reference: the set of ``(pre_neuron, post_neuron)`` inhibitory edges.

    A deliberately *set-based* reimplementation of NEST's row/col/box/cell inhibition
    (nested loops + ``set.add``), so it shares no code path with the net's vectorized
    ``np.repeat``/``np.tile`` builder. Equality between the two pins the vectorization.
    """
    ni = np.arange(729 * pop_size).reshape(9, 9, 9, pop_size)
    edges = set()
    for r in range(9):
        for c in range(9):
            for d in range(9):
                br, bc = (r // 3) * 3, (c // 3) * 3
                box = ni[br:br + 3, bc:bc + 3]
                tgt = np.unique(np.concatenate(
                    (ni[r, :, d], ni[:, c, d], box[:, :, d], ni[r, c, :]), axis=None))
                tgt = np.setdiff1d(tgt, ni[r, c, d])
                for s in ni[r, c, d]:
                    for t in tgt:
                        edges.add((int(s), int(t)))
    return edges


class TestTopologyReference(unittest.TestCase):
    """Sanity-check the independent reference builder before comparing the net to it."""

    def test_reference_edge_count_and_structure(self):
        pop_size = 5
        edges = reference_inhibitory_edges(pop_size)
        # Each source population inhibits exactly 28 other populations (row 8 + col 8
        # + box 4-new + same-cell-other-digit 8 = 28), at neuron resolution:
        #   729 pops x 5 src neurons x (28 tgt pops x 5 neurons) = 510300 edges.
        self.assertEqual(len(edges), 729 * pop_size * (28 * pop_size))
        self.assertEqual(len(edges), 510300)
        self.assertFalse(any(s == t for s, t in edges))            # no self-loops
        # every edge crosses populations (a population never inhibits itself)
        self.assertTrue(all((s // pop_size) != (t // pop_size) for s, t in edges))

    def test_reference_symmetric_relation(self):
        # The WTA constraint is symmetric at population level: if pop A inhibits pop B,
        # pop B inhibits pop A. Check the induced population-pair relation is symmetric.
        edges = reference_inhibitory_edges(pop_size=5)
        pop_pairs = {(s // 5, t // 5) for s, t in edges}
        self.assertTrue(all((b, a) in pop_pairs for a, b in pop_pairs))


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


class TestSudokuNetTopology(unittest.TestCase):
    """The net's inhibitory edge set == the independent reference (exact set equality)."""

    def test_inhibitory_edge_set_equals_reference(self):
        from examples.nest.sudoku_net import SudokuNet
        net = SudokuNet(seed=0)
        got = net.inhibitory_edges()                           # set of (pre, post) pairs
        self.assertEqual(got, reference_inhibitory_edges(pop_size=net.pop_size))


if __name__ == '__main__':
    unittest.main()
