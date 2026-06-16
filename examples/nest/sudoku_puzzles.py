# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Pure-Python puzzle bank and solution validator for the Sudoku WTA solver.

A faithful, dependency-light port of NEST's ``pynest/examples/sudoku/helpers_sudoku.py``
``get_puzzle`` / ``validate_solution`` (NumPy only -- no spiking machinery, no
matplotlib). The eight boards and the validation logic match NEST exactly so the
brainpy spiking solver in :mod:`examples.nest.sudoku_net` can be measured against the
same configurations and the same notion of "valid".

See Also
--------
examples.nest.sudoku_net : the spiking WTA constraint network that relaxes a puzzle.
examples.nest.sudoku : the runnable host-loop solver harness.
"""
import numpy as np

__all__ = ['N_PUZZLES', 'get_puzzle', 'validate_solution', 'make_easy_puzzle']

#: Number of bundled puzzle configurations (indices ``0`` .. ``N_PUZZLES - 1``).
N_PUZZLES = 8

# The eight boards, verbatim from NEST's ``get_puzzle``. ``0`` marks an empty cell;
# a non-zero digit is a clue. Index 0 is the all-zero "dream" board (no clues -> the
# network must invent any valid Sudoku); index 6 is the "world's hardest" board.
_PUZZLES = {
    0: np.zeros((9, 9), dtype=np.uint8),
    1: [
        [0, 0, 1, 0, 0, 8, 0, 7, 3],
        [0, 0, 5, 6, 0, 0, 0, 0, 1],
        [7, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 9, 0, 8, 1, 0, 0, 0, 0],
        [5, 3, 0, 0, 0, 0, 0, 4, 6],
        [0, 0, 0, 0, 6, 5, 0, 3, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 4],
        [8, 0, 0, 0, 0, 9, 3, 0, 0],
        [9, 4, 0, 5, 0, 0, 7, 0, 0],
    ],
    2: [
        [2, 0, 0, 0, 0, 6, 0, 3, 0],
        [4, 8, 0, 0, 1, 9, 0, 0, 0],
        [0, 0, 7, 0, 2, 0, 9, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 9, 0],
        [7, 0, 8, 0, 0, 0, 1, 0, 5],
        [0, 4, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 4, 0, 9, 0, 6, 0, 0],
        [0, 0, 0, 6, 4, 0, 0, 1, 9],
        [0, 5, 0, 1, 0, 0, 0, 0, 8],
    ],
    3: [
        [0, 0, 3, 2, 0, 0, 0, 7, 0],
        [0, 0, 5, 0, 0, 0, 3, 0, 0],
        [0, 0, 8, 9, 7, 0, 0, 5, 0],
        [0, 0, 0, 8, 9, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 6, 1, 0, 0, 0],
        [0, 1, 0, 0, 2, 5, 6, 0, 0],
        [0, 0, 4, 0, 0, 0, 8, 0, 0],
        [0, 9, 0, 0, 0, 7, 5, 0, 0],
    ],
    4: [
        [0, 1, 0, 0, 0, 0, 0, 0, 2],
        [8, 7, 0, 0, 0, 0, 5, 0, 4],
        [5, 0, 2, 0, 0, 0, 0, 9, 0],
        [0, 5, 0, 4, 0, 9, 0, 0, 1],
        [0, 0, 0, 7, 3, 2, 0, 0, 0],
        [9, 0, 0, 5, 0, 1, 0, 4, 0],
        [0, 2, 0, 0, 0, 0, 4, 0, 8],
        [4, 0, 6, 0, 0, 0, 0, 1, 3],
        [1, 0, 0, 0, 0, 0, 0, 2, 0],
    ],
    5: [
        [8, 9, 0, 2, 0, 0, 0, 7, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 4, 1, 0, 3, 0, 5, 0, 0],
        [2, 5, 8, 0, 0, 0, 0, 0, 6],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 1, 4, 7],
        [0, 0, 7, 0, 1, 0, 4, 3, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 7, 0, 5, 1],
    ],
    6: [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0],
    ],
    7: [
        [1, 0, 0, 4, 0, 0, 0, 0, 0],
        [7, 0, 0, 5, 0, 0, 6, 0, 3],
        [0, 0, 0, 0, 3, 0, 4, 2, 0],
        [0, 0, 9, 0, 0, 0, 0, 3, 5],
        [0, 0, 0, 3, 0, 5, 0, 0, 0],
        [6, 3, 0, 0, 0, 0, 1, 0, 0],
        [0, 2, 6, 0, 5, 0, 0, 0, 0],
        [9, 0, 4, 0, 0, 6, 0, 0, 7],
        [0, 0, 0, 0, 0, 8, 0, 0, 2],
    ],
}


def get_puzzle(puzzle_index):
    """Return one of the eight bundled Sudoku configurations.

    Parameters
    ----------
    puzzle_index : int
        Index in ``range(N_PUZZLES)`` (0..7) selecting the board.

    Returns
    -------
    numpy.ndarray
        A ``(9, 9)`` ``uint8`` array: ``0`` where no clue is given, otherwise the
        clue digit (1..9). A fresh copy is returned on each call, so the caller may
        mutate it freely.

    Raises
    ------
    ValueError
        If ``puzzle_index`` is not in ``range(N_PUZZLES)``.

    See Also
    --------
    validate_solution : check a proposed solution against a puzzle.

    Examples
    --------
    .. code-block:: python

        >>> from examples.nest.sudoku_puzzles import get_puzzle
        >>> puzzle = get_puzzle(4)
        >>> puzzle.shape
        (9, 9)
        >>> puzzle[0].tolist()
        [0, 1, 0, 0, 0, 0, 0, 0, 2]

        >>> int(get_puzzle(0).sum())          # index 0 is the all-zero "dream" board
        0
    """
    if puzzle_index not in _PUZZLES:
        raise ValueError(
            f'No puzzle for index {puzzle_index}; expected 0..{N_PUZZLES - 1}.')
    return np.array(_PUZZLES[puzzle_index], dtype=np.uint8)


def make_easy_puzzle(n_blank=12, seed=0):
    """Return a near-complete, solvable board: a valid grid with cells cleared.

    Useful as a fast-converging board for demos and solve-rate parity: the spiking WTA
    (in NEST and brainpy alike) reliably completes a board with only a handful of blanks
    within a couple of relaxation chunks, where it does not reliably crack a hard board.

    Parameters
    ----------
    n_blank : int, optional
        Number of cells to clear from a complete valid grid. Default ``12``.
    seed : int, optional
        Seed selecting which cells are cleared (independent of any solver seed), so the
        same ``(n_blank, seed)`` always yields the same board. Default ``0``.

    Returns
    -------
    numpy.ndarray
        ``(9, 9)`` ``uint8`` puzzle (clues are the un-cleared cells of a valid grid).

    Examples
    --------
    .. code-block:: python

        >>> from examples.nest.sudoku_puzzles import make_easy_puzzle, validate_solution
        >>> puzzle = make_easy_puzzle(12, seed=0)
        >>> puzzle.shape
        (9, 9)
        >>> int((puzzle == 0).sum())          # exactly n_blank cells cleared
        12
    """
    grid = np.array([[((i * 3 + i // 3 + j) % 9) + 1 for j in range(9)] for i in range(9)],
                    dtype=np.uint8)
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(81, int(n_blank), replace=False)
    flat = grid.flatten()
    flat[idx] = 0
    return flat.reshape(9, 9)


def validate_solution(puzzle, solution):
    """Validate a proposed solution for a Sudoku puzzle (NEST-faithful).

    Checks the three Sudoku constraints (every 3x3 box, every row, every column is a
    permutation of 1..9) and that the solution honours the puzzle's clues. A solution
    is overall valid only if all components hold -- including the rare case where the
    network settles on a valid grid that nonetheless overrides a clued cell.

    Parameters
    ----------
    puzzle : numpy.ndarray
        ``(9, 9)`` clue array (see :func:`get_puzzle`); ``0`` marks an empty cell.
    solution : numpy.ndarray
        ``(9, 9)`` proposed solution with entries in 1..9.

    Returns
    -------
    valid : bool
        ``True`` iff every box, row, and column is valid *and* all clues match.
    boxes : numpy.ndarray
        ``(3, 3)`` boolean array, ``True`` where the corresponding 3x3 box is valid.
    rows : numpy.ndarray
        ``(9,)`` boolean array, ``True`` where the corresponding row is valid.
    cols : numpy.ndarray
        ``(9,)`` boolean array, ``True`` where the corresponding column is valid.

    See Also
    --------
    get_puzzle : the bundled puzzle configurations.

    Notes
    -----
    The component arrays (``boxes`` / ``rows`` / ``cols``) are returned so a caller can
    report a *fraction-correct* progress signal during relaxation, e.g.
    ``(boxes.sum() + rows.sum() + cols.sum()) / 27``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from examples.nest.sudoku_puzzles import validate_solution
        >>> grid = np.array([[((i * 3 + i // 3 + j) % 9) + 1 for j in range(9)]
        ...                  for i in range(9)])
        >>> valid, boxes, rows, cols = validate_solution(np.zeros((9, 9), int), grid)
        >>> bool(valid), boxes.shape, rows.shape
        (True, (3, 3), (9,))

        >>> broken = grid.copy()
        >>> broken[0, 0] = grid[0, 1]          # duplicate within row 0
        >>> bool(validate_solution(np.zeros((9, 9), int), broken)[0])
        False
    """
    puzzle = np.asarray(puzzle)
    solution = np.asarray(solution)

    boxes = np.ones((3, 3), dtype=bool)
    rows = np.ones(9, dtype=bool)
    cols = np.ones(9, dtype=bool)

    expected_numbers = set(range(1, 10))

    for i in range(3):
        for j in range(3):
            box = solution[3 * i: 3 * i + 3, 3 * j: 3 * j + 3]
            if expected_numbers != set(box.flatten().tolist()):
                boxes[i, j] = False

    for i in range(9):
        if expected_numbers != set(solution[i, :].tolist()):
            rows[i] = False
        if expected_numbers != set(solution[:, i].tolist()):
            cols[i] = False

    # A valid grid may still override a clue; fold that into overall validity.
    input_cells = np.where(puzzle != 0)
    puzzle_matched = puzzle[input_cells] == solution[input_cells]

    valid = bool(boxes.all() and rows.all() and cols.all() and puzzle_matched.all())
    return valid, boxes, rows, cols
