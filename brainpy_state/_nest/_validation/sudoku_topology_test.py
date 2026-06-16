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
import brainunit as u
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import Simulator, poisson_generator, parrot_neuron, spike_recorder

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

    def test_make_easy_puzzle_clues_match_a_valid_grid(self):
        from examples.nest.sudoku_puzzles import make_easy_puzzle
        puzzle = make_easy_puzzle(12, seed=0)
        self.assertEqual(int((puzzle == 0).sum()), 12)
        valid, *_ = validate_solution(puzzle, _valid_grid())   # source grid solves it
        self.assertTrue(valid)

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


class TestStimClueWeights(unittest.TestCase):
    """``set_input_config`` clamps clue populations at ``weight_stim`` and zeroes the rest."""

    def test_set_input_config_sets_clue_weights(self):
        from examples.nest.sudoku_net import SudokuNet, WEIGHT_STIM
        net = SudokuNet(seed=0)
        puzzle = get_puzzle(4)
        net.set_input_config(puzzle)
        w = net.stim_weights_pA()                              # (n_total,) cell-indexed, pA
        ps = net.pop_size
        for r in range(9):
            for c in range(9):
                v = int(puzzle[r, c])
                for d in range(9):
                    p = (r * 9 + c) * 9 + d
                    expect = WEIGHT_STIM if (v != 0 and d == v - 1) else 0.0
                    np.testing.assert_allclose(w[ps * p: ps * p + ps], expect)

    def test_reset_input_zeroes_all_clue_weights(self):
        from examples.nest.sudoku_net import SudokuNet
        net = SudokuNet(seed=0)
        net.set_input_config(get_puzzle(4))
        self.assertGreater(float(net.stim_weights_pA().sum()), 0.0)   # some clues set
        net.reset_input()
        np.testing.assert_allclose(net.stim_weights_pA(), 0.0)

    def test_set_input_config_dream_board_clamps_nothing(self):
        from examples.nest.sudoku_net import SudokuNet
        net = SudokuNet(seed=0)
        net.set_input_config(get_puzzle(0))                    # all-zero "dream" board
        np.testing.assert_allclose(net.stim_weights_pA(), 0.0)


class TestReadout(unittest.TestCase):
    """Per-chunk readout: population spike counts -> argmax-with-tiebreak solution."""

    def test_read_counts_and_solution_shapes(self):
        from examples.nest.sudoku_net import SudokuNet
        net = SudokuNet(seed=0)
        net.sim.reset_rollout()
        res = net.sim.cont(20.0 * u.ms)
        counts = net.read_counts(res)
        self.assertEqual(counts.shape, (9, 9, 9))
        self.assertTrue(np.issubdtype(counts.dtype, np.integer))
        sol = net.read_solution(res)
        self.assertEqual(sol.shape, (9, 9))
        self.assertGreaterEqual(int(sol.min()), 1)
        self.assertLessEqual(int(sol.max()), 9)

    def test_read_solution_tiebreak_deterministic_under_seed(self):
        from examples.nest.sudoku_net import SudokuNet
        net = SudokuNet(seed=0)
        net.sim.reset_rollout()
        res = net.sim.cont(20.0 * u.ms)
        np.random.seed(123)
        sol1 = net.read_solution(res)
        np.random.seed(123)
        sol2 = net.read_solution(res)
        np.testing.assert_array_equal(sol1, sol2)


class TestForLoopLowering(unittest.TestCase):
    """The 100 ms chunk rollout lowers via ``cont()``'s ``for_loop`` -- one trace, not a
    Python step loop (cluster-12 discipline; the property that makes the solver tractable)."""

    def test_chunk_rollout_lowers_under_for_loop(self):
        from examples.nest.sudoku_net import SudokuNet
        net = SudokuNet(seed=0)
        sim = net.sim

        # Spy on the per-step update: under for_loop the body is *traced once*, so the
        # Python-level update is called O(1) times regardless of step count. A bare
        # Python step loop would call it once per step (== n_steps).
        calls = [0]
        orig_update = sim.update

        def counting_update(*args, **kwargs):
            calls[0] += 1
            return orig_update(*args, **kwargs)

        sim.update = counting_update
        sim.reset_rollout()
        n_steps = 50                                           # 5 ms at dt = 0.1 ms
        res = sim.cont(n_steps * 0.1 * u.ms)
        self.assertLess(calls[0], 10, 'rollout did not lower under for_loop (Python step loop?)')
        spikes = np.asarray(res.spikes(net.recorder))
        self.assertEqual(spikes.shape, (n_steps, net.n_total))
        self.assertTrue(np.all(np.isfinite(spikes)))


class TestSolver(unittest.TestCase):
    """``SudokuSolver`` runs the host relaxation loop and exits gracefully."""

    def test_solve_returns_shaped_result_and_respects_max_iterations(self):
        from examples.nest.sudoku_net import SudokuNet, SudokuSolver
        net = SudokuNet(seed=0)
        solver = SudokuSolver(net)
        solution, valid, chunks = solver.solve(get_puzzle(4), max_iterations=3, seed=0)
        self.assertEqual(solution.shape, (9, 9))
        self.assertIn(bool(valid), (True, False))
        self.assertLessEqual(chunks, 3)
        self.assertGreaterEqual(chunks, 1)
        self.assertGreaterEqual(int(solution.min()), 1)
        self.assertLessEqual(int(solution.max()), 9)


class TestDriveRates(unittest.TestCase):
    """The Poisson drives fire at SudokuNet's configured rates (sampling tolerance).

    Probes the exact drive path SudokuNet uses -- a ``poisson_generator`` at rate ``R``
    relayed 1:1 through a ``parrot_neuron`` -- at the example's own ``NOISE_RATE`` /
    ``STIM_RATE`` constants, on a tiny net (fast; no need to run the 510k-edge WTA).
    """

    @staticmethod
    def _measure_relayed_rate(rate_hz, sim_ms=3000.0, seed=0):
        sim = Simulator(dt=0.1 * u.ms)
        gen = sim.create(poisson_generator, rate=rate_hz * u.Hz)
        relay = sim.create(parrot_neuron, 1)
        rec = sim.create(spike_recorder)
        sim.connect(gen, relay, weight=1.0, delay=1.0 * u.ms)
        sim.connect(relay, rec)
        brainstate.random.seed(seed)
        sim.reset_rollout()
        res = sim.cont(sim_ms * u.ms)
        n = float(np.asarray(res.spikes(rec)).sum())
        return n / (sim_ms / 1000.0)

    def test_noise_rate_is_350hz(self):
        from examples.nest.sudoku_net import NOISE_RATE
        self.assertAlmostEqual(NOISE_RATE, 350.0)
        self.assertLess(abs(self._measure_relayed_rate(NOISE_RATE) - 350.0), 45.0)

    def test_stim_rate_is_200hz(self):
        from examples.nest.sudoku_net import STIM_RATE
        self.assertAlmostEqual(STIM_RATE, 200.0)
        self.assertLess(abs(self._measure_relayed_rate(STIM_RATE) - 200.0), 35.0)


if __name__ == '__main__':
    unittest.main()
