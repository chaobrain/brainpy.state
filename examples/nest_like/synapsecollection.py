# examples/nest_like/synapsecollection.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""SynapseCollection introspection tour — NEST-style port.

Ports NEST's ``pynest/examples/synapsecollection.py`` to the Simulator API. The
upstream connects neurons under several rules / synapse models / weight
distributions, retrieves the connections with ``nest.GetConnections()`` (a
``SynapseCollection``), and reads / sets ``.source``, ``.target``, ``.weight`` to
build weight matrices. This port reproduces every idiom with
:meth:`Simulator.get_connections` and
:class:`~brainpy.state.SynapseCollection`:

* ``get(['source', 'target', 'weight'])`` — read several attributes at once;
* ``set('weight', values)`` — per-edge write-back, round-tripped through ``get``;
* ``get_connections()`` — every edge; ``get_connections(src, tgt)`` — a
  source/target slice; ``get_connections(synapse=model)`` — one synapse model.

Three faithful adaptations to ``brainpy.state``:

* **one_to_one is per-edge here.** NEST's ``one_to_one`` uses the per-edge-settable
  default ``static_synapse``; the homogeneous ``one_to_one`` *EventProjection*
  shares a single scalar, so to keep the upstream's *per-edge* weight set we route
  the ``one_to_one`` block through the per-edge ``static_synapse`` plastic path
  (same realized connectivity, per-edge weights).
* **Distributional weights are sampled at connect on the static path.** The
  ``all_to_all`` block draws ``Uniform(0.5, 4.5)`` pA with the ``dist`` API (the
  brainpy.state ``Parameter`` analogue, sampled once at ``connect``). The plastic
  ``stdp_synapse`` connects take a concrete initial weight (the rule evolves it
  during ``simulate``), so the complex block uses fixed initial weights there.
* **Population-local indices.** ``source`` / ``target`` index within their
  population (no global node-id space), so the matrices are indexed directly.

The demo is pure introspection — it never calls ``simulate()``.

Run:  PYTHONPATH=. python examples/nest_like/synapsecollection.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy.state import (Simulator, iaf_psc_alpha, one_to_one, all_to_all,
                           pairwise_bernoulli, fixed_total_number, fixed_indegree,
                           static_synapse, stdp_synapse)
from brainpy.state import dist

DT = 0.1            # resolution [ms]


def ramp_weights(n):
    """NEST's ``[{'weight': x} for x in range(1, n+1)]`` as a pA array ``[1..n]``."""
    return np.arange(1, n + 1, dtype=float) * u.pA


def collection_arrays(conns):
    """Read a SynapseCollection into ``(sources, targets, weights_pA, matrix)``.

    Parameters
    ----------
    conns : SynapseCollection
        The connections to read.

    Returns
    -------
    sources, targets : numpy.ndarray
        Population-local source / target index per edge.
    weights : numpy.ndarray
        Per-edge weight in pA.
    matrix : numpy.ndarray
        Dense ``(max_src+1, max_tgt+1)`` weight matrix (``W[src, tgt] += w``); a
        ``(0, 0)`` array when the collection is empty.
    """
    g = conns.get(['source', 'target', 'weight'])
    srcs = np.asarray(g['source'])
    tgts = np.asarray(g['target'])
    weights = np.asarray(u.Quantity(g['weight']).to_decimal(u.pA))
    if srcs.size == 0:
        return srcs, tgts, weights, np.zeros((0, 0))
    M = np.zeros((int(srcs.max()) + 1, int(tgts.max()) + 1))
    np.add.at(M, (srcs, tgts), weights)
    return srcs, tgts, weights, M


def build_one_to_one(n=10):
    """``n`` neurons wired one-to-one with a per-edge ``static_synapse``.

    Returns ``(sim, nrns)``. The connection is the per-edge ``static_synapse``
    plastic path (not the homogeneous ``one_to_one`` EventProjection), so the
    upstream's per-edge weight ``set`` round-trips.
    """
    sim = Simulator(dt=DT * u.ms)
    nrns = sim.create(iaf_psc_alpha, n)
    sim.connect(nrns, nrns, rule=one_to_one,
                synapse=static_synapse(weight=1. * u.pA))
    return sim, nrns


def build_all_to_all(n_pre=10, n_post=5, seed=0):
    """``n_pre -> n_post`` all-to-all with ``Uniform(0.5, 4.5)`` pA weights.

    Returns ``(sim, pre, post)``. Weights are drawn once at ``connect`` from the
    ``dist`` Parameter API (the static path samples distributions eagerly).
    """
    sim = Simulator(dt=DT * u.ms)
    pre = sim.create(iaf_psc_alpha, n_pre)
    post = sim.create(iaf_psc_alpha, n_post)
    sim.connect(pre, post, rule=all_to_all,
                weight=dist.Uniform(0.5 * u.pA, 4.5 * u.pA), comm='sparse', seed=seed)
    return sim, pre, post


def build_complex(n=15, seed=0):
    """``n`` neurons wired with five rules / models (the upstream's complex case).

    Returns ``(sim, nrns)``. Mirrors NEST's five ``Connect`` calls: a
    ``one_to_one`` ``stdp_synapse``, a ``pairwise_bernoulli`` static block, a
    ``fixed_total_number`` static block, an ``all_to_all`` ``stdp_synapse``, and a
    ``fixed_indegree`` static block.
    """
    sim = Simulator(dt=DT * u.ms)
    nrns = sim.create(iaf_psc_alpha, n)
    sim.connect(nrns[:5], nrns[:5], rule=one_to_one,
                synapse=stdp_synapse(weight=5. * u.pA))
    sim.connect(nrns[:10], nrns[5:12], rule=pairwise_bernoulli(0.4),
                weight=4. * u.pA, comm='sparse', seed=seed)
    sim.connect(nrns[5:10], nrns[:5], rule=fixed_total_number(5),
                weight=3. * u.pA, comm='sparse', seed=seed)
    sim.connect(nrns[10:], nrns[:12], rule=all_to_all,
                synapse=stdp_synapse(weight=4. * u.pA))
    sim.connect(nrns, nrns[12:], rule=fixed_indegree(3),
                weight=1. * u.pA, comm='sparse', seed=seed)
    return sim, nrns


def main():
    print("synapsecollection: SynapseCollection introspection tour (brainpy.state)")

    # 1. one_to_one: identity, then a per-edge weight set.
    sim, nrns = build_one_to_one(10)
    conns = sim.get_connections()
    srcs, tgts, w0, _ = collection_arrays(conns)
    print(f"  one_to_one : {len(conns)} edges, identity={np.array_equal(srcs, tgts)}, "
          f"uniform weight={np.unique(w0).tolist()} pA")
    conns.set('weight', ramp_weights(10))
    w1 = np.asarray(u.Quantity(conns.get('weight')).to_decimal(u.pA))
    print(f"  one_to_one : after set, diagonal weights={w1.tolist()} pA")

    # 2. all_to_all with uniformly distributed weights, asymmetric sizes.
    sim, pre, post = build_all_to_all(10, 5)
    conns = sim.get_connections()
    _, _, w, _ = collection_arrays(conns)
    print(f"  all_to_all : {len(conns)} edges, weight in "
          f"[{w.min():.2f}, {w.max():.2f}] pA (Uniform 0.5..4.5)")

    # 3. complex: five rules/models; query all / a subset / one model / set a subset.
    sim, nrns = build_complex(15)
    all_conns = sim.get_connections()
    subset = sim.get_connections(source=nrns[:10], target=nrns[:10])
    stdp = sim.get_connections(synapse='stdp_synapse')
    ftn = sim.get_connections(source=nrns[5:10], target=nrns[:5])
    print(f"  complex    : all={len(all_conns)} edges, "
          f"first-ten subset={len(subset)}, stdp_synapse={len(stdp)}, "
          f"fixed_total_number={len(ftn)}")
    ftn.set('weight', ramp_weights(len(ftn)))
    fw = np.asarray(u.Quantity(ftn.get('weight')).to_decimal(u.pA))
    print(f"  complex    : fixed_total_number after set, weights={fw.tolist()} pA")

    try:
        import matplotlib.pyplot as plt

        def _plot(ax, conns, title):
            _srcs, _tgts, _weights, M = collection_arrays(conns)
            if M.size:
                im = ax.imshow(M, aspect="auto")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("target")
            ax.set_ylabel("source")
            ax.set_title(title)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        _plot(axes[0, 0], sim.get_connections(), "All connections")
        _plot(axes[0, 1], sim.get_connections(source=nrns[:10], target=nrns[:10]),
              "First ten neurons")
        _plot(axes[1, 0], sim.get_connections(synapse='stdp_synapse'),
              "stdp_synapse only")
        _plot(axes[1, 1], ftn, "fixed_total_number, set weight")
        fig.tight_layout()
        fig.savefig("examples/nest_like/synapsecollection.png", dpi=100)
        print("  wrote examples/nest_like/synapsecollection.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
