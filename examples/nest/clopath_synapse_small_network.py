# examples/nest/clopath_synapse_small_network.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Clopath voltage-based STDP in a small recurrent network -- NEST-style port.

Adapted from NEST's ``clopath_synapse_small_network.py`` (Clopath et al. 2010,
fig. 5: the rule establishes *directional* structure). The upstream drives ten
excitatory + three inhibitory ``aeif_psc_delta_clopath`` neurons with 500 Poisson
generators whose Gaussian rate profile jumps randomly every 100 ms -- not
reproducible sample-for-sample. This port keeps the demonstrated effect but makes
it **deterministic**: a small all-to-all recurrent Clopath population (no
autapses) whose neurons are forced to spike by per-neuron ``spike_generator``
clamps (80 mV, the cluster-07 ``DRIVE_W``) at *staggered* times.

Each cycle the neurons fire in the order ``0 -> 1 -> 2`` (10 ms apart), so for
every ordered pair the incoming recurrent edge sees the canonical spike-pairing
protocol: forward edges (``i -> j`` with ``i`` firing before ``j``) get
pre-before-post pairing and **potentiate**; backward edges get post-before-pre
pairing and **depress**. The recurrent weight matrix is recorded with
``res.weight_trace`` and its evolution shows the feedforward chain emerging.

``aeif_psc_delta_clopath`` is a **delta** neuron, so the bare ``clopath_synapse``
weight is in **mV**. The neuron parameters, delays, init weight and 5 % parity
band are the cluster-07 ones (shared via
:mod:`brainpy_state._nest._validation._clopath_drive`); per edge this is exactly
the validated pairing protocol, so the matrix matches NEST within the documented
Clopath band (LTD near-exact, LTP within 5 %).

Run:  python examples/nest/clopath_synapse_small_network.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import (Simulator, spike_generator, clopath_synapse,
                           static_synapse, all_to_all)
from brainpy_state._nest._validation import _clopath_drive as drv

#: Number of recurrent excitatory Clopath neurons.
N = 3
#: Inter-neuron firing stagger (ms): neuron ``i`` fires ``i * LAG`` into each cycle.
LAG = 10.0
#: Cycle period (ms) and first-spike offset (ms).
PERIOD = 50.0
BASE = 20.0
#: Number of stimulation cycles.
N_CYCLES = 5
#: Per-neuron clamp trains (ms): neuron ``i`` fires at ``BASE + i*LAG + c*PERIOD``.
TRAINS = [[BASE + i * LAG + c * PERIOD for c in range(N_CYCLES)] for i in range(N)]
#: Simulation horizon (ms): last clamp plus a short settle.
T_SIM = TRAINS[-1][-1] + 20.0


def run():
    """Drive the recurrent Clopath network and record its weight-matrix evolution.

    Builds ``N`` ``aeif_psc_delta_clopath`` neurons, connects them all-to-all
    (no autapses) with a single recorded ``clopath_synapse`` projection, and
    clamps each neuron to spike at its staggered train via an 80 mV
    ``static_synapse`` driver.

    Returns
    -------
    times : numpy.ndarray
        Recorder time axis (ms).
    weights : numpy.ndarray
        Recurrent weight trajectory, shape ``(n_steps, N*(N-1))`` (mV); column
        ``c`` is the edge ``edges[c]``.
    edges : list of tuple of int
        ``(pre, post)`` local indices for each weight column, in the projection's
        CSR (sorted-by-pre) order.
    """
    sim = Simulator(dt=drv.DT * u.ms)
    pop = drv._our_clopath_neuron(sim, n=N)
    rec = sim.connect(pop, pop, synapse=clopath_synapse(weight=drv.INIT_W * u.mV),
                      rule=all_to_all, allow_autapses=False, delay=drv.RELAY_D * u.ms)
    sim.record_weight(rec)
    for i in range(N):
        sg = sim.create(spike_generator, spike_times=np.asarray(TRAINS[i]) * u.ms)
        sim.connect(sg, pop[i], synapse=static_synapse(weight=drv.DRIVE_W * u.mV),
                    delay=drv.RELAY_D * u.ms)
    res = sim.simulate(T_SIM * u.ms)
    times = np.asarray(u.get_mantissa(res.times / u.ms))
    weights = np.asarray(u.get_mantissa(res.weight_trace(rec) / u.mV))
    edges = list(zip(np.asarray(rec._pre_idx).tolist(),
                     np.asarray(rec._post_idx).tolist()))
    return times, weights, edges


def weight_matrix(weight_row, edges, n=N, fill=np.nan):
    """Scatter a per-edge weight vector into an ``n x n`` matrix.

    Parameters
    ----------
    weight_row : array_like
        One weight per edge (e.g. ``weights[-1]`` for the final matrix).
    edges : sequence of tuple of int
        ``(pre, post)`` for each entry of ``weight_row``.
    n : int, optional
        Matrix size. Default :data:`N`.
    fill : float, optional
        Value for the (absent) diagonal. Default ``numpy.nan``.

    Returns
    -------
    numpy.ndarray
        ``n x n`` matrix ``M`` with ``M[pre, post] = weight``.
    """
    M = np.full((n, n), fill, dtype=float)
    for (i, j), w in zip(edges, np.asarray(weight_row)):
        M[i, j] = w
    return M


def main():
    times, weights, edges = run()
    final = weights[-1]
    print("Clopath small recurrent network (brainpy.state, w in mV, init "
          f"{drv.INIT_W})")
    print("  final recurrent weight matrix (rows = pre, cols = post):")
    M = weight_matrix(final, edges)
    for i in range(N):
        row = "  ".join("   .  " if np.isnan(M[i, j]) else f"{M[i, j]:6.3f}"
                        for j in range(N))
        print(f"    {row}")
    print(f"  forward edges mean {np.mean([w for (i, j), w in zip(edges, final) if j > i]):.4f} "
          f"(> {drv.INIT_W} = potentiated), "
          f"backward mean {np.mean([w for (i, j), w in zip(edges, final) if j < i]):.4f} "
          f"(< {drv.INIT_W} = depressed)")

    try:
        import matplotlib.pyplot as plt
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
        for c, (i, j) in enumerate(edges):
            fwd = j > i
            ax0.plot(times, weights[:, c], color="C0" if fwd else "C3",
                     lw=1.2, label="forward (i<j)" if (fwd and c == 0) else
                     ("backward (i>j)" if (not fwd and c == 2) else None))
        ax0.axhline(drv.INIT_W, color="0.6", ls=":", lw=1)
        ax0.set_xlabel("time (ms)"); ax0.set_ylabel("recurrent weight (mV)")
        ax0.set_title("Clopath weight evolution"); ax0.legend(fontsize=8)
        im = ax1.imshow(weight_matrix(final, edges, fill=drv.INIT_W),
                        cmap="RdBu_r", vmin=drv.INIT_W - 0.04, vmax=drv.INIT_W + 0.04)
        ax1.set_title("final weight matrix"); ax1.set_xlabel("post"); ax1.set_ylabel("pre")
        ax1.set_xticks(range(N)); ax1.set_yticks(range(N))
        fig.colorbar(im, ax=ax1, label="w (mV)")
        fig.tight_layout()
        fig.savefig("examples/nest/clopath_synapse_small_network.png", dpi=100)
        print("  wrote examples/nest/clopath_synapse_small_network.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
