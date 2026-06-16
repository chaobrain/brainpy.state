# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Solve Sudoku with a spiking winner-take-all network — NEST §3.10 harness.

A faithful port of NEST's ``pynest/examples/sudoku/sudoku_solver.py``. One
:class:`~examples.nest_like.sudoku_net.SudokuNet` (3645 ``iaf_psc_exp`` neurons whose
inhibitory connectivity encodes the row/column/box/cell Sudoku constraints) is built
**once** and relaxed by a host-side loop: each 100 ms :meth:`Simulator.cont` chunk the
clued cells are clamped by Poisson stimulation, background noise drives exploration, and
the per-cell winning digit is read out from the chunk's spike tally. The loop stops as
soon as the read-out grid validates, or when the chunk budget is exhausted.

The network is a *stochastic* constraint solver, so success is distributional, not
guaranteed: a near-complete board (``--puzzle easy``) reliably completes within a couple
of chunks, while a hard board (the default ``--puzzle 4``, NEST's own default) typically
plateaus below a full solution within a practical budget — exactly as NEST's bundled
example behaves. Run several ``--seeds`` to see the solve rate.

Run::

    PYTHONPATH=. python examples/nest_like/sudoku.py --quick
    PYTHONPATH=. python examples/nest_like/sudoku.py --puzzle easy --seeds 5
    PYTHONPATH=. python examples/nest_like/sudoku.py --puzzle 4 --max-iterations 100

Reference
---------
Original NEST implementation by S. B. Furber et al.'s spiking-Sudoku approach, ported in
NEST's ``pynest/examples/sudoku/``. See also Binas, Neil, Liu & Delbruck (2016),
"Precise deep neural network computation on imprecise low-power analog hardware".
"""
import argparse

import jax
import brainstate
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainunit as u

from examples.nest_like.sudoku_net import SudokuNet, DEFAULT_MAX_ITERATIONS, DEFAULT_SIM_TIME_MS
from examples.nest_like.sudoku_puzzles import get_puzzle, make_easy_puzzle, validate_solution


def _ratio_correct(boxes, rows, cols):
    """Fraction of the 27 Sudoku constraints (9 boxes + 9 rows + 9 cols) satisfied."""
    return float(boxes.sum() + rows.sum() + cols.sum()) / 27.0


def solve_puzzle(puzzle, *, net=None, seed=0, pop_size=5,
                 max_iterations=DEFAULT_MAX_ITERATIONS, sim_time_ms=DEFAULT_SIM_TIME_MS):
    """Relax one puzzle on a spiking WTA network, recording per-chunk progress.

    The instrumented twin of :meth:`~examples.nest_like.sudoku_net.SudokuSolver.solve`: it
    drives the same build-once ``cont()`` host loop but also records the fraction of
    constraints satisfied after every chunk, so the relaxation can be plotted.

    Parameters
    ----------
    puzzle : numpy.ndarray
        ``(9, 9)`` clue array (see :func:`~examples.nest_like.sudoku_puzzles.get_puzzle`).
    net : SudokuNet, optional
        A pre-built network to relax (reused as-is). If ``None``, one is built with
        ``seed`` and ``pop_size``. Pass a shared net to relax many seeds without
        rebuilding (the membrane init and Poisson trains are re-drawn each call).
    seed : int, optional
        Seed for ``numpy.random`` (readout tiebreak) and ``brainstate.random``
        (membrane init + Poisson trains). Default ``0``.
    pop_size : int, optional
        Neurons per ``(row, col, digit)`` population, used only when building a net.
        Default ``5``.
    max_iterations : int, optional
        Maximum number of relaxation chunks. Default ``100``.
    sim_time_ms : float, optional
        Biological duration of each chunk in ms. Default ``100.0``.

    Returns
    -------
    dict
        ``{'puzzle', 'solution', 'valid', 'chunks', 'ratio', 'trajectory', 'seed'}``:
        the decoded ``(9, 9)`` grid, whether it validates, the number of chunks run, the
        final fraction-correct, the per-chunk fraction-correct list, and the seed used.
    """
    if net is None:
        net = SudokuNet(seed=seed, pop_size=pop_size)
    np.random.seed(seed)
    brainstate.random.seed(seed)
    net.set_input_config(puzzle)
    net.sim.reset_rollout()

    trajectory = []
    solution, valid, boxes, rows, cols = None, False, None, None, None
    run = 0
    while not valid and run < max_iterations:
        res = net.sim.cont(sim_time_ms * u.ms)
        solution = net.read_solution(res)
        valid, boxes, rows, cols = validate_solution(puzzle, solution)
        trajectory.append(_ratio_correct(boxes, rows, cols))
        run += 1
    return {
        'puzzle': np.asarray(puzzle),
        'solution': solution,
        'valid': bool(valid),
        'chunks': run,
        'ratio': trajectory[-1] if trajectory else 0.0,
        'trajectory': trajectory,
        'seed': seed,
    }


def solve_seeds(puzzle, *, label='puzzle', seeds=1, pop_size=5,
                max_iterations=DEFAULT_MAX_ITERATIONS, sim_time_ms=DEFAULT_SIM_TIME_MS):
    """Relax ``puzzle`` over ``seeds`` seeds on a single build-once network.

    One :class:`~examples.nest_like.sudoku_net.SudokuNet` is built and reused across every
    seed (the whole point of the build-once design); each seed re-draws the membrane
    init and Poisson trains via :func:`solve_puzzle`.

    Parameters
    ----------
    puzzle : numpy.ndarray
        ``(9, 9)`` clue array.
    label : str, optional
        Human-readable puzzle label for reporting/plot titles. Default ``'puzzle'``.
    seeds : int, optional
        Number of seeds to try (``range(seeds)``). Default ``1``.
    pop_size : int, optional
        Neurons per population. Default ``5``.
    max_iterations, sim_time_ms
        Passed through to :func:`solve_puzzle`.

    Returns
    -------
    dict
        ``{'label', 'puzzle', 'per_seed', 'solve_rate', 'best'}`` — ``per_seed`` is the
        list of :func:`solve_puzzle` results, ``solve_rate`` the fraction that validated,
        and ``best`` the result with the highest final fraction-correct (for display).
    """
    net = SudokuNet(seed=0, pop_size=pop_size)
    per_seed = [solve_puzzle(puzzle, net=net, seed=s, max_iterations=max_iterations,
                             sim_time_ms=sim_time_ms) for s in range(seeds)]
    solve_rate = float(np.mean([r['valid'] for r in per_seed]))
    best = max(per_seed, key=lambda r: r['ratio'])
    return {'label': label, 'puzzle': np.asarray(puzzle),
            'per_seed': per_seed, 'solve_rate': solve_rate, 'best': best}


def _cell_correct_mask(puzzle, solution):
    """``(9, 9)`` bool mask: a cell is correct iff its row, col, and box all validate."""
    _valid, boxes, rows, cols = validate_solution(puzzle, solution)
    mask = np.zeros((9, 9), dtype=bool)
    for r in range(9):
        for c in range(9):
            mask[r, c] = bool(rows[r] and cols[c] and boxes[r // 3, c // 3])
    return mask


def plot_result(result, path):
    """Write a two-panel figure (decoded grid + relaxation curves) to ``path``.

    Left: the best seed's decoded grid, cells tinted green where their row/col/box all
    validate and red otherwise, with clue digits underlined. Right: the fraction of
    constraints satisfied versus chunk index, one curve per seed.

    Parameters
    ----------
    result : dict
        A :func:`solve_seeds` result.
    path : str
        Output image path.

    Returns
    -------
    str
        ``path`` (for convenient chaining/printing).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    best = result['best']
    puzzle, solution = result['puzzle'], best['solution']
    mask = _cell_correct_mask(puzzle, solution)

    fig, (ax_grid, ax_curve) = plt.subplots(1, 2, figsize=(12, 5.5))

    # -- left: decoded grid, tinted by per-cell correctness, clues underlined ----------
    ax_grid.imshow(np.where(mask, 0.85, 0.55), cmap='RdYlGn', vmin=0, vmax=1,
                   extent=(0, 9, 9, 0))
    for r in range(9):
        for c in range(9):
            is_clue = puzzle[r, c] != 0
            ax_grid.text(c + 0.5, r + 0.5, str(int(solution[r, c])),
                         ha='center', va='center',
                         fontsize=14, fontweight='bold' if is_clue else 'normal',
                         color='black')
            if is_clue:                    # underline clue digits (mpl has no text underline)
                ax_grid.plot([c + 0.3, c + 0.7], [r + 0.78, r + 0.78], color='navy', lw=1.5)
    for k in range(10):                    # thin cell lines, thick 3x3 block lines
        lw = 2.5 if k % 3 == 0 else 0.5
        ax_grid.plot([k, k], [0, 9], color='black', lw=lw)
        ax_grid.plot([0, 9], [k, k], color='black', lw=lw)
    ax_grid.set_xlim(0, 9)
    ax_grid.set_ylim(9, 0)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_title(f"{result['label']}: seed {best['seed']} — "
                      f"{'SOLVED' if best['valid'] else f'ratio {best['ratio']:.2f}'} "
                      f"in {best['chunks']} chunks")

    # -- right: relaxation curves (fraction of constraints satisfied per chunk) ---------
    for r in result['per_seed']:
        ax_curve.plot(np.arange(1, len(r['trajectory']) + 1), r['trajectory'],
                      marker='.', label=f"seed {r['seed']}"
                                        f"{' ✓' if r['valid'] else ''}")
    ax_curve.axhline(1.0, color='grey', ls='--', lw=1)
    ax_curve.set_xlabel('relaxation chunk (100 ms each)')
    ax_curve.set_ylabel('fraction of constraints satisfied')
    ax_curve.set_ylim(0, 1.05)
    ax_curve.set_title(f"solve rate {result['solve_rate']:.0%} "
                       f"over {len(result['per_seed'])} seed(s)")
    if len(result['per_seed']) <= 10:
        ax_curve.legend(loc='lower right', fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _resolve_puzzle(spec, blanks):
    """Map a CLI ``--puzzle`` spec to ``(puzzle_array, label)``.

    ``spec`` is either ``'easy'`` (a near-complete board with ``blanks`` blanks) or a
    digit string selecting one of the eight bundled boards.
    """
    if spec == 'easy':
        return make_easy_puzzle(blanks, seed=0), f'easy ({blanks} blanks)'
    return get_puzzle(int(spec)), f'puzzle {int(spec)}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--puzzle', type=str, default='4',
                        help="Board: 'easy' (near-complete) or an index 0..7 (default 4, "
                             "NEST's default; index 6 is the 'world's hardest').")
    parser.add_argument('--blanks', type=int, default=12,
                        help="Number of blanks for --puzzle easy (default 12).")
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds to relax (reports the solve rate).')
    parser.add_argument('--pop-size', type=int, default=5,
                        help='Neurons per (row, col, digit) population (default 5).')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS,
                        help='Maximum relaxation chunks per seed (default 100).')
    parser.add_argument('--sim-time-ms', type=float, default=DEFAULT_SIM_TIME_MS,
                        help='Biological ms per relaxation chunk (default 100).')
    parser.add_argument('--quick', action='store_true',
                        help='Bounded smoke run: easy board, 1 seed, <=12 chunks.')
    parser.add_argument('--out', type=str, default='examples/nest_like/sudoku.png',
                        help='Output path for the grid + relaxation figure.')
    args = parser.parse_args()

    if args.quick:
        puzzle, label = _resolve_puzzle('easy', args.blanks)
        seeds, max_iterations = 1, min(args.max_iterations, 12)
    else:
        puzzle, label = _resolve_puzzle(args.puzzle, args.blanks)
        seeds, max_iterations = args.seeds, args.max_iterations

    print(f"sudoku: relaxing {label} (pop_size {args.pop_size}) over {seeds} seed(s), "
          f"up to {max_iterations} chunks each...")
    result = solve_seeds(puzzle, label=label, seeds=seeds, pop_size=args.pop_size,
                         max_iterations=max_iterations, sim_time_ms=args.sim_time_ms)
    for r in result['per_seed']:
        status = 'SOLVED' if r['valid'] else 'unsolved'
        print(f"  seed {r['seed']}: {status} after {r['chunks']} chunks, "
              f"best ratio {max(r['trajectory']):.2f}")
    n_solved = sum(r['valid'] for r in result['per_seed'])
    print(f"  solve rate: {n_solved}/{seeds} ({result['solve_rate']:.0%}); "
          f"best final ratio {result['best']['ratio']:.2f}")
    try:
        out = plot_result(result, args.out)
        print(f"  wrote {out}")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == '__main__':
    main()
