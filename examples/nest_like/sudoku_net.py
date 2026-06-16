# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Spiking WTA constraint network that relaxes a Sudoku puzzle (NEST §3.10 port).

A faithful brainpy ``Simulator`` port of NEST's ``pynest/examples/sudoku/sudoku_net.py``.
The network is **one** population of ``9*9*9 * pop_size`` ``iaf_psc_exp`` neurons whose
inhibitory connectivity encodes the Sudoku constraints: a population coding digit ``d``
in cell ``(r, c)`` inhibits every population coding the same digit in the same row,
column, or 3x3 box, and every population coding a *different* digit in the same cell.
Background Poisson noise drives exploration; per-population Poisson stimulation clamps
the clues. Driven by the noise the network settles on states that read out as valid
Sudoku grids.

The port differs from NEST only in *construction*, not dynamics: the 510 300 inhibitory
edges are realized as **one** sparse :func:`brainpy.state.explicit_edges` projection
(not 729 separate ``Connect`` calls), so ``Simulator.cont()`` compiles the relaxation
chunk once and reuses it across the host-side relaxation loop instead of retracing.

See Also
--------
examples.nest_like.sudoku_puzzles : the puzzle bank + solution validator.
examples.nest_like.sudoku : the runnable host-loop solver harness.

Notes
-----
Units follow NEST's ``iaf_psc_exp`` convention exactly: capacitance in **pF**, injected
current in **pA**, synaptic weights in **pA** (negative ⇒ inhibitory), times in ms,
voltages in mV. In particular ``C_m = 0.25 pF`` and ``I_e = 0.5 pA`` are NEST's bundled
values taken in NEST's native units -- *not* nF/nA (which would scale every synaptic
PSP 1000x below the bias current and collapse the competition into tonic firing).
"""
import jax

jax.config.update('jax_enable_x64', True)

import numpy as np
import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

import braintools
from brainpy import state as bps

from examples.nest_like.sudoku_puzzles import validate_solution

# -- NEST constants (verbatim from sudoku_net.py, in NEST's native units) ----------
N_POPULATIONS = 9 ** 3              # 729 = rows x cols x digits
DT = 0.1                            # ms, integration step
DELAY = 1.0                         # ms, every synapse
NOISE_RATE = 350.0                  # Hz, background poisson_generator
STIM_RATE = 200.0                   # Hz, per-population clue stimulation
INTER_NEURON_WEIGHT = -0.2          # pA, inhibitory WTA/constraint synapse
WEIGHT_NOISE = 1.6                  # pA, background noise -> neuron
WEIGHT_STIM = 1.3                   # pA, clue stimulation -> neuron

#: ``iaf_psc_exp`` parameters, verbatim from NEST (pF / pA / ms / mV).
NEURON_PARAMS = dict(
    C_m=0.25 * u.pF, I_e=0.5 * u.pA, tau_m=20.0 * u.ms, t_ref=2.0 * u.ms,
    tau_syn_ex=5.0 * u.ms, tau_syn_in=5.0 * u.ms,
    V_reset=-70.0 * u.mV, E_L=-65.0 * u.mV, V_th=-50.0 * u.mV,
)


def _inhibitory_edge_arrays(pop_size):
    """Build the inhibitory WTA/constraint edge list as ``(pre_idx, post_idx)`` arrays.

    Vectorized (``np.repeat`` / ``np.tile``) port of NEST's row/col/box/cell inhibition.
    For every source population ``(r, c, d)`` the unique target *neuron* set is the union
    of same-digit row, same-digit column, same-digit box, and same-cell other-digit
    populations, minus the source population itself.

    Parameters
    ----------
    pop_size : int
        Neurons per population.

    Returns
    -------
    pre_idx, post_idx : numpy.ndarray
        Equal-length 1-D int arrays of segment-local source/target neuron indices.
    """
    ni = np.arange(N_POPULATIONS * pop_size).reshape(9, 9, 9, pop_size)
    pre_list, post_list = [], []
    for r in range(9):
        for c in range(9):
            br, bc = (r // 3) * 3, (c // 3) * 3
            box = ni[br:br + 3, bc:bc + 3]
            for d in range(9):
                src = ni[r, c, d]
                tgt = np.unique(np.concatenate(
                    (ni[r, :, d], ni[:, c, d], box[:, :, d], ni[r, c, :]), axis=None))
                tgt = np.setdiff1d(tgt, src)
                pre_list.append(np.repeat(src, tgt.size))
                post_list.append(np.tile(tgt, src.size))
    return np.concatenate(pre_list), np.concatenate(post_list)


class SudokuNet:
    r"""Build-once spiking WTA network for Sudoku, driven by a host relaxation loop.

    Parameters
    ----------
    seed : int, optional
        Seed for ``brainstate.random`` (membrane-potential init + Poisson trains).
        Default ``0``.
    pop_size : int, optional
        Neurons per ``(row, col, digit)`` population. Default ``5`` (NEST's value).
    noise_rate : float, optional
        Background Poisson rate in Hz. Default ``350.0``.
    stim_rate : float, optional
        Per-population clue-stimulation Poisson rate in Hz. Default ``200.0``.

    Attributes
    ----------
    sim : brainpy.state.Simulator
        The compiled simulator (build once; never rebuilt per chunk/puzzle).
    cells : brainpy.state.NodeView
        The single ``iaf_psc_exp`` population of ``N_POPULATIONS * pop_size`` neurons,
        indexed ``((row*9 + col)*9 + digit)*pop_size + k``.

    Notes
    -----
    Neuron ``((r*9 + c)*9 + d)*pop_size + k`` belongs to population
    ``p = (r*9 + c)*9 + d``, which owns neurons ``[pop_size*p, pop_size*p + pop_size)``.
    """

    def __init__(self, seed=0, pop_size=5, noise_rate=NOISE_RATE, stim_rate=STIM_RATE):
        self.seed = int(seed)
        self.pop_size = int(pop_size)
        self.n_total = N_POPULATIONS * self.pop_size
        self.noise_rate = float(noise_rate)
        self.stim_rate = float(stim_rate)

        brainstate.random.seed(self.seed)
        self.sim = bps.Simulator(dt=DT * u.ms)

        params = dict(NEURON_PARAMS)
        params['V_initializer'] = braintools.init.Uniform(-65.0 * u.mV, -55.0 * u.mV)
        self.cells = self.sim.create(bps.iaf_psc_exp, self.n_total, params=params)

        # Inhibitory WTA/constraint topology: ONE sparse explicit-edge projection.
        pre, post = _inhibitory_edge_arrays(self.pop_size)
        self.sim.connect(
            self.cells, self.cells,
            rule=bps.explicit_edges(pre, post),
            weight=INTER_NEURON_WEIGHT * u.pA, delay=DELAY * u.ms, comm='sparse')

        # Background noise: one generator, all_to_all -> N independent Poisson trains.
        self.noise = self.sim.create(bps.poisson_generator, rate=self.noise_rate * u.Hz)
        self.sim.connect(self.noise, self.cells, rule=bps.all_to_all,
                         weight=WEIGHT_NOISE * u.pA, delay=DELAY * u.ms)

        # Clue stimulation: one 200 Hz generator -> 729 parrots (one independent train
        # each), then each parrot -> its population's pop_size cells via a single sparse
        # explicit-edge projection whose weights are toggled per puzzle (0 <-> 1.3 pA).
        self.stim_gen = self.sim.create(bps.poisson_generator, rate=self.stim_rate * u.Hz)
        self.parrots = self.sim.create(bps.parrot_neuron, N_POPULATIONS)
        self.sim.connect(self.stim_gen, self.parrots, rule=bps.all_to_all,
                         weight=1.0, delay=DELAY * u.ms)            # unit gate weight
        stim_pre = np.repeat(np.arange(N_POPULATIONS), self.pop_size)   # parrot p
        stim_post = np.arange(self.n_total)                             # cell pop_size*p+k
        self.sim.connect(self.parrots, self.cells,
                         rule=bps.explicit_edges(stim_pre, stim_post),
                         weight=0.0 * u.pA, delay=DELAY * u.ms, comm='sparse')

        # Wide readout: one spike_recorder taps every cell (columns in neuron order).
        self.recorder = self.sim.create(bps.spike_recorder)
        self.sim.connect(self.cells, self.recorder)

    def read_counts(self, res):
        """Per-population spike counts for the chunk in ``res``.

        Parameters
        ----------
        res : brainpy.state.SimulationResult
            The result of the most recent :meth:`~brainpy.state.Simulator.cont` chunk.

        Returns
        -------
        numpy.ndarray
            ``(9, 9, 9)`` integer array: spikes summed over the chunk and over each
            population's ``pop_size`` neurons, indexed ``[row, col, digit]``.
        """
        spikes = np.asarray(res.spikes(self.recorder))             # (n_steps, n_total)
        per_neuron = spikes.sum(axis=0)                            # (n_total,)
        return per_neuron.reshape(9, 9, 9, self.pop_size).sum(axis=-1).astype(int)

    def read_solution(self, res):
        """Decode a ``(9, 9)`` solution from a chunk by argmax-per-cell with tiebreak.

        For each cell the digit whose population fired most wins; ties (including the
        all-silent case) are broken uniformly at random via ``numpy.random`` -- seed it
        for reproducibility. Digits are returned in 1..9 (NEST's convention).

        Parameters
        ----------
        res : brainpy.state.SimulationResult
            The result of the most recent :meth:`~brainpy.state.Simulator.cont` chunk.

        Returns
        -------
        numpy.ndarray
            ``(9, 9)`` integer array of decoded digits (1..9).
        """
        counts = self.read_counts(res)
        sol = np.zeros((9, 9), dtype=int)
        for r in range(9):
            for c in range(9):
                dc = counts[r, c]
                winners = np.flatnonzero(dc == dc.max())
                sol[r, c] = int(np.random.choice(winners)) + 1
        return sol

    def _clue_cell_weights(self, puzzle):
        """Return the ``(n_total,)`` per-cell stim weight (pA) for a puzzle's clues."""
        puzzle = np.asarray(puzzle)
        w = np.zeros(self.n_total)
        for r, c in np.argwhere(puzzle != 0):
            d = int(puzzle[r, c]) - 1
            p = (int(r) * 9 + int(c)) * 9 + d
            w[self.pop_size * p: self.pop_size * p + self.pop_size] = WEIGHT_STIM
        return w

    def set_input_config(self, puzzle):
        """Clamp the clues of ``puzzle`` by weighting their stimulation edges.

        For every clued cell ``(r, c)`` with value ``v``, the parrot->cell edges feeding
        population ``p = (r*9 + c)*9 + (v - 1)`` are set to ``weight_stim`` (1.3 pA); all
        other stimulation edges are set to 0. This is a live ``State`` write -- no
        recompile -- so it may be called between relaxation chunks.

        Parameters
        ----------
        puzzle : numpy.ndarray
            ``(9, 9)`` clue array (see :func:`~examples.nest_like.sudoku_puzzles.get_puzzle`);
            zero-valued cells are left unstimulated.
        """
        per_cell = self._clue_cell_weights(puzzle)
        conns = self.sim.get_connections(source=self.parrots, target=self.cells)
        tgt = np.asarray(conns.get('target'))                      # cell index per edge
        conns.set('weight', per_cell[tgt] * u.pA)                  # keyed by target cell

    def reset_input(self):
        """Set all stimulation (parrot->cell) weights to 0 -- unclamp every cell."""
        conns = self.sim.get_connections(source=self.parrots, target=self.cells)
        conns.set('weight', np.zeros(int(len(conns))) * u.pA)

    def stim_weights_pA(self):
        """Return the live per-cell stimulation weight as a ``(n_total,)`` array in pA.

        Indexed by *cell* (``stim_weights_pA()[pop_size*p + k]`` is the stim weight into
        neuron ``k`` of population ``p``), so the result is independent of the internal
        edge ordering.
        """
        conns = self.sim.get_connections(source=self.parrots, target=self.cells)
        tgt = np.asarray(conns.get('target'))
        w = np.asarray(u.Quantity(conns.get('weight')).to_decimal(u.pA))
        out = np.zeros(self.n_total)
        out[tgt] = w
        return out

    def inhibitory_edges(self):
        """Return the realized inhibitory edge set as ``{(pre_neuron, post_neuron), ...}``.

        Reads the live connectivity via
        :meth:`~brainpy.state.Simulator.get_connections`, filtered to ``cells -> cells``.

        Returns
        -------
        set of tuple of int
            Segment-local ``(source, target)`` neuron-index pairs.
        """
        conns = self.sim.get_connections(source=self.cells, target=self.cells)
        src = np.asarray(conns.get('source')).tolist()
        tgt = np.asarray(conns.get('target')).tolist()
        return set(zip(src, tgt))


# default relaxation budget: NEST runs up to 10 s of biological time in 100 ms chunks
DEFAULT_SIM_TIME_MS = 100.0
DEFAULT_MAX_ITERATIONS = 100


class SudokuSolver:
    r"""Drive a :class:`SudokuNet` with NEST's host-side relaxation loop.

    The network is built once; :meth:`solve` clamps the clues, resets the rollout, then
    repeatedly advances the *same* compiled ``cont()`` chunk, reading out a candidate
    solution after each chunk and stopping as soon as it validates (or the iteration
    budget is exhausted). State (membrane potentials, synaptic currents) evolves
    continuously across chunks -- only the per-chunk spike tally drives the readout --
    exactly as NEST interleaves ``reset_spike_recorders()`` with a continuous
    ``Simulate``.

    Parameters
    ----------
    net : SudokuNet
        The build-once network to relax.

    See Also
    --------
    SudokuNet : the spiking network this loop drives.
    """

    def __init__(self, net):
        self.net = net

    def solve(self, puzzle, max_iterations=DEFAULT_MAX_ITERATIONS,
              sim_time_ms=DEFAULT_SIM_TIME_MS, seed=0):
        """Relax ``puzzle`` until a valid solution is read out or the budget runs out.

        Parameters
        ----------
        puzzle : numpy.ndarray
            ``(9, 9)`` clue array (see :func:`~examples.nest_like.sudoku_puzzles.get_puzzle`).
        max_iterations : int, optional
            Maximum number of relaxation chunks. Default ``100`` (NEST's ``10 s / 100 ms``).
        sim_time_ms : float, optional
            Biological duration of each chunk in ms. Default ``100.0``.
        seed : int, optional
            Seed for ``numpy.random`` (readout tiebreak) and ``brainstate.random``
            (membrane init + Poisson trains). Default ``0``.

        Returns
        -------
        solution : numpy.ndarray
            The last decoded ``(9, 9)`` grid (digits 1..9).
        valid : bool
            Whether ``solution`` is a valid solution of ``puzzle``.
        chunks : int
            Number of relaxation chunks run (``<= max_iterations``); the
            chunks-to-solution when ``valid`` is ``True``.
        """
        np.random.seed(seed)
        brainstate.random.seed(seed)
        self.net.set_input_config(puzzle)
        self.net.sim.reset_rollout()

        run, valid, solution = 0, False, None
        while not valid and run < max_iterations:
            res = self.net.sim.cont(sim_time_ms * u.ms)
            solution = self.net.read_solution(res)
            valid, _boxes, _rows, _cols = validate_solution(puzzle, solution)
            run += 1
        return solution, bool(valid), run
