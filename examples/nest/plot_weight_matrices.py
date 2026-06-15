# examples/nest/plot_weight_matrices.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Weight-matrix extraction by connection introspection — NEST-style port.

Ports NEST's ``pynest/examples/plot_weight_matrices.py`` to the Simulator API.
The upstream builds an excitatory population ``E`` and an inhibitory population
``I``, connects them with ``fixed_indegree`` synapses (excitatory weights drawn
``Normal(20, 0.5)`` pA, inhibitory weights ``-g`` times as large), then extracts
the weight of *every* realized connection with ``nest.GetConnections(pre, post)``
and assembles the four weight matrices ``EE / EI / IE / II`` for visualization.

This port reproduces that with :meth:`Simulator.get_connections`: for each of the
four population pairings it enumerates the realized edges and scatters
``W[source, target] += weight`` into a dense matrix — no held projection handle,
exactly NEST's ``SynapseCollection`` idiom.

Two faithful adaptations to ``brainpy.state``:

* **Population-local indices.** ``source`` / ``target`` are indices *within* their
  population, so the matrices are indexed directly — there is no global node-id
  space and hence no ``- min(node_id)`` subtraction (NEST's offset).
* **"post-pre" naming.** As in the upstream, the matrix from inhibitory to
  excitatory neurons (I->E) is named ``W_EI`` and excitatory to inhibitory (E->I)
  is ``W_IE`` (post-pre convention). Multapses, if any, sum into a cell exactly as
  NEST's ``W[i, j] += w`` loop sums them.

The demo is pure connectivity introspection — it never calls ``simulate()`` — so
the weights read are the values realized at ``connect`` time.

Run:  PYTHONPATH=. python examples/nest/plot_weight_matrices.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import Simulator, iaf_psc_alpha, fixed_indegree
from brainpy_state import dist

# --- NEST plot_weight_matrices parameters (verbatim) -------------------------
NE = 100            # number of excitatory neurons
NI = 25             # number of inhibitory neurons
CE = int(0.1 * NE)  # excitatory synapses per neuron (fixed_indegree = 10)
CI = int(0.1 * NI)  # inhibitory synapses per neuron (fixed_indegree = 2)

DELAY = 1.5         # synaptic delay [ms]
G = 5.0             # ratio |inhibitory weight| / excitatory weight
W_EX_MEAN = 20.0    # excitatory weight mean [pA]
W_EX_STD = 0.5      # excitatory weight std [pA]
DT = 0.1            # resolution [ms]


def build(*, ne=NE, ni=NI, ce=CE, ci=CI, seed=0):
    """Build the E/I network and return ``(sim, E, I)``.

    Excitatory edges (``E -> E+I``) draw ``Normal(W_EX_MEAN, W_EX_STD)`` pA;
    inhibitory edges (``I -> E+I``) draw the ``-G``-scaled normal
    ``Normal(-G*W_EX_MEAN, G*W_EX_STD)`` pA, reproducing NEST's ``w_in = -g*w_ex``
    (scaling a normal by ``-g`` scales its mean and std). Connectivity is
    ``fixed_indegree`` (each post neuron gets ``ce`` excitatory and ``ci``
    inhibitory inputs); edges are stored sparsely (memory-light fan-out).

    Parameters
    ----------
    ne, ni : int, optional
        Excitatory / inhibitory population sizes. Defaults :data:`NE` / :data:`NI`.
    ce, ci : int, optional
        Excitatory / inhibitory in-degree. Defaults :data:`CE` / :data:`CI`.
    seed : int, optional
        Base connectivity / weight seed. Default ``0``.

    Returns
    -------
    sim : Simulator
    E, I : NodeView
        The excitatory and inhibitory population views.
    """
    sim = Simulator(dt=DT * u.ms)
    E = sim.create(iaf_psc_alpha, ne)
    I = sim.create(iaf_psc_alpha, ni)
    w_ex = dist.Normal(W_EX_MEAN * u.pA, W_EX_STD * u.pA)
    w_in = dist.Normal(-G * W_EX_MEAN * u.pA, G * W_EX_STD * u.pA)
    sim.connect(E, E + I, rule=fixed_indegree(ce), weight=w_ex, delay=DELAY * u.ms,
                comm='sparse', seed=seed)
    sim.connect(I, E + I, rule=fixed_indegree(ci), weight=w_in, delay=DELAY * u.ms,
                comm='sparse', seed=seed + 1)
    return sim, E, I


def weight_matrix(sim, source, target, n_source, n_target):
    """Dense ``(n_source, n_target)`` weight matrix for one population pairing.

    Enumerates the realized ``source -> target`` synapses via
    :meth:`Simulator.get_connections` and scatters each edge's weight (pA mantissa)
    into ``W[source_local, target_local]``, summing any multapses (NEST's
    ``W[i, j] += w``).

    Parameters
    ----------
    sim : Simulator
        The built network.
    source, target : NodeView
        Pre- and post-synaptic population views.
    n_source, n_target : int
        Matrix dimensions (population sizes).

    Returns
    -------
    numpy.ndarray
        ``(n_source, n_target)`` weight matrix in pA.
    """
    conns = sim.get_connections(source=source, target=target)
    W = np.zeros((n_source, n_target))
    if len(conns) == 0:
        return W
    src = np.asarray(conns.source)
    trg = np.asarray(conns.target)
    w = np.asarray(u.Quantity(conns.get('weight')).to_decimal(u.pA))
    np.add.at(W, (src, trg), w)            # += per edge, multapses summed
    return W


def weight_matrices(sim, E, I):
    """The four E/I weight matrices ``(W_EE, W_EI, W_IE, W_II)`` (post-pre naming).

    Returns
    -------
    dict
        ``{'EE': W_EE, 'EI': W_EI, 'IE': W_IE, 'II': W_II}`` — ``EE`` is E->E,
        ``EI`` is I->E, ``IE`` is E->I, ``II`` is I->I (all pA).
    """
    ne = E.size
    ni = I.size
    return {
        'EE': weight_matrix(sim, E, E, ne, ne),     # E -> E
        'EI': weight_matrix(sim, I, E, ni, ne),     # I -> E  (post-pre: W_EI)
        'IE': weight_matrix(sim, E, I, ne, ni),     # E -> I  (post-pre: W_IE)
        'II': weight_matrix(sim, I, I, ni, ni),     # I -> I
    }


def main():
    print("plot_weight_matrices: E/I weight-matrix extraction by connection "
          "introspection (brainpy.state)")
    sim, E, I = build()
    W = weight_matrices(sim, E, I)
    for name in ('EE', 'EI', 'IE', 'II'):
        m = W[name]
        nz = m[m != 0.0]
        mean = float(nz.mean()) if nz.size else 0.0
        print(f"  W_{name}: shape {m.shape}, {nz.size} edges, mean weight "
              f"{mean:8.2f} pA")

    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig = plt.figure()
        fig.suptitle("Weight matrices", fontsize=14)
        gs = gridspec.GridSpec(4, 4)
        axes = [plt.subplot(gs[:-1, :-1]), plt.subplot(gs[:-1, -1]),
                plt.subplot(gs[-1, :-1]), plt.subplot(gs[-1, -1])]
        for ax, name in zip(axes, ('EE', 'IE', 'EI', 'II')):
            im = ax.imshow(W[name], cmap="jet")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(im, cax=cax)
            ax.set_title(f"$W_{{{name}}}$")
        fig.tight_layout()
        fig.savefig("examples/nest/plot_weight_matrices.png", dpi=100)
        print("  wrote examples/nest/plot_weight_matrices.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
