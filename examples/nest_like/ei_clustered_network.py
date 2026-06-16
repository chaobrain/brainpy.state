# examples/nest_like/ei_clustered_network.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""EI-clustered random balanced network — NEST-style port.

Port of NEST's ``EI_clustered_network/`` (``clustering="weight"``). An
``iaf_psc_exp`` random balanced network (RBN) whose excitatory and inhibitory
populations are each split into ``Q`` clusters. In-cluster synapses are
potentiated by ``J+`` and out-cluster synapses depressed by ``J-`` so that each
row's mean weight is preserved (Rostami et al. 2020, Eqs 7-10). The network is
driven by a constant per-neuron rheobase current — there is **no external
Poisson** drive; the spontaneous activity comes entirely from the recurrent
balanced dynamics.

The headline phenomenon is **metastability**: with clustering (``rep > 1``) the
network no longer settles into a homogeneous asynchronous-irregular state but
hops between *winner-take-all* configurations in which a few clusters fire fast
while the rest are suppressed. This shows up as large across-cluster rate
heterogeneity and more irregular (bursty) firing than the homogeneous
``rep = 1`` control, which is an ordinary balanced random network.

Weight construction (verified identical to NEST ``helper.py`` / ``network.py``)
-------------------------------------------------------------------------------
* :func:`rbn_weights` builds the ``2x2`` base weights ``js`` so that
  ``sqrt(K)`` input spikes reach threshold and the E/I rows are balanced.
* :func:`cluster_weights` builds ``J+`` and ``J-`` from the clustering factors:
  ``jep = rep``, ``jip = 1 + (rep - 1)·rj``, ``jplus = [[jep, jip], [jip, jip]]``
  and ``jminus = (Q - jplus) / (Q - 1)``. Choosing ``rep < Q`` keeps ``J-``
  positive (the canonical NEST config is ``Q = 20, rep = 6``).
* Each ordered ``(pre-cluster, post-cluster)`` block is wired with
  ``pairwise_bernoulli`` at ``baseline_conn_prob`` (``allow_autapses=False`` on
  the E→E and I→I blocks, ``allow_multapses=False``); same-cluster blocks use the
  ``J+`` weight, the rest ``J-``. Here this is realized as a **masked-dense**
  projection — one ``all_to_all`` connection per block whose flat weight vector
  carries the Bernoulli mask times the per-entry cluster weight — which is much
  cheaper than ``Q²`` sparse projections and statistically identical.

Implementation notes
--------------------
* **Separate** ``spike_recorder``\\ s for E and I; tapping both populations into one
  recorder mis-orders the returned columns.
* ``all_to_all`` resolves the weight to a per-edge vector ordered **row-major
  ``(pre, post)``** (edge ``k`` = ``pre[k // n_post], post[k % n_post]``), so a
  flattened ``(n_pre, n_post)`` matrix maps correctly.

Run:  PYTHONPATH=. python examples/nest_like/ei_clustered_network.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import Simulator, iaf_psc_exp, spike_recorder, all_to_all

#: Excitatory / inhibitory population sizes (NEST canonical 4000/1000 scaled 10x).
N_E = 400
N_I = 100
#: Number of clusters (NEST canonical). ``N_E``/``N_I`` must be divisible by ``Q``.
Q = 20
#: Clustering factors: excitatory potentiation and relative E/I clustering ratio.
REP = 6.0
RJ = 0.82
#: ``iaf_psc_exp`` parameters (NEST EI-clustered defaults).
E_L = 0.0
C_M = 1.0
TAU_E = 20.0
TAU_I = 10.0
T_REF = 5.0
V_TH = 20.0
V_R = 0.0
TAU_SYN = 5.0
DELAY = 0.1
#: Feed-forward drive in units of the rheobase current (E and I).
I_TH_E = 1.25
I_TH_I = 0.78
#: Inhibitory weight ratios (balanced-network scaling).
GEI = 1.2
GIE = 1.0
GII = 1.0
#: Baseline connection probabilities: [[E→E, I→E], [E→I, I→I]].
BCP = np.array([[0.2, 0.5], [0.5, 0.5]])
#: Resolution (ms), warm-up discarded before statistics (ms), measured window (ms).
DT = 0.1
WARMUP = 500.0
SIMTIME = 2000.0


def psc_to_psp(tau_m, tau_syn, c_m=1.0, e_l=0.0):
    """Peak post-synaptic potential (mV) for a 1 pA exponential current spike.

    Parameters
    ----------
    tau_m : float
        Membrane time constant (ms).
    tau_syn : float
        Synaptic time constant (ms).
    c_m : float, optional
        Membrane capacitance (pF). Default ``1.0``.
    e_l : float, optional
        Resting potential (mV). Default ``0.0``.

    Returns
    -------
    float
        Maximum PSP amplitude (mV) for a unit (1 pA) current spike.

    Examples
    --------
    .. code-block:: python

        >>> from examples.nest_like.ei_clustered_network import psc_to_psp
        >>> round(float(psc_to_psp(20.0, 5.0)), 4)
        3.1498
    """
    tmax = np.log(tau_syn / tau_m) / (1 / tau_m - 1 / tau_syn)
    pre = tau_m * tau_syn / c_m / (tau_syn - tau_m)
    return (e_l - pre) * np.exp(-tmax / tau_m) + pre * np.exp(-tmax / tau_syn)


def rheobase(tau_m, e_l, v_th, c_m):
    """Rheobase current (pA): the constant current that just reaches threshold.

    Parameters
    ----------
    tau_m : float
        Membrane time constant (ms).
    e_l : float
        Resting potential (mV).
    v_th : float
        Threshold potential (mV).
    c_m : float
        Membrane capacitance (pF).

    Returns
    -------
    float
        Rheobase current (pA).
    """
    return (v_th - e_l) * c_m / tau_m


def rbn_weights(ne=N_E, ni=N_I):
    """Base ``2x2`` synaptic weights of the random balanced network (Eqs 7-10).

    The weights are scaled so that ``sqrt(K)`` simultaneous input spikes drive the
    membrane from rest to threshold and the excitatory / inhibitory contributions
    to each row balance.

    Parameters
    ----------
    ne, ni : int, optional
        Excitatory / inhibitory population sizes. Default the module constants.

    Returns
    -------
    numpy.ndarray
        ``2x2`` matrix ``[[EE, EI], [IE, II]]``; the E rows are positive
        (excitatory) and the I rows negative (inhibitory).
    """
    n = ne + ni
    amp_ee = amp_ei = psc_to_psp(TAU_E, TAU_SYN)
    amp_ie = amp_ii = psc_to_psp(TAU_I, TAU_SYN)
    js = np.zeros((2, 2))
    k_ee = ne * BCP[0, 0]
    js[0, 0] = (V_TH - E_L) * (k_ee ** -0.5) * n ** 0.5 / amp_ee
    js[0, 1] = -GEI * js[0, 0] * BCP[0, 0] * ne * amp_ee / (BCP[0, 1] * ni * amp_ei)
    k_ie = ne * BCP[1, 0]
    js[1, 0] = GIE * (V_TH - E_L) * (k_ie ** -0.5) * n ** 0.5 / amp_ie
    js[1, 1] = -GII * js[1, 0] * BCP[1, 0] * ne * amp_ie / (BCP[1, 1] * ni * amp_ii)
    return js


def cluster_weights(rep, rj, q):
    """In-cluster ``J+`` and out-cluster ``J-`` weight-factor matrices.

    Parameters
    ----------
    rep : float
        Excitatory clustering factor (in-cluster E→E potentiation). ``rep = 1`` is
        the homogeneous (no-clustering) balanced network.
    rj : float
        Relative E/I clustering ratio; sets ``jip = 1 + (rep - 1)·rj``.
    q : int
        Number of clusters. ``J-`` stays positive when ``rep < q``.

    Returns
    -------
    jplus, jminus : numpy.ndarray
        ``2x2`` potentiation / depression factors. ``jminus`` is chosen so each
        row's mean weight (over one in-cluster and ``q - 1`` out-cluster blocks)
        equals the base weight.

    Examples
    --------
    .. code-block:: python

        >>> from examples.nest_like.ei_clustered_network import cluster_weights
        >>> jplus, jminus = cluster_weights(6.0, 0.82, 20)
        >>> (round(float(jplus[0, 0]), 1), round(float(jminus[0, 0]), 4))
        (6.0, 0.7368)
    """
    jep = rep
    jip = 1.0 + (rep - 1.0) * rj
    jplus = np.array([[jep, jip], [jip, jip]])
    if q > 1:
        jminus = (q - jplus) / (q - 1.0)
    else:
        jplus = np.ones((2, 2))
        jminus = np.ones((2, 2))
    return jplus, jminus


def cluster_weight_matrix(npre, spre, npost, spost, base_j, plus, minus, p,
                          no_auto, seed):
    """Masked-dense weight matrix for one ``(pre, post)`` block.

    Samples a Bernoulli connection mask at probability ``p`` and assigns each
    realized edge the in-cluster weight ``plus·base_j`` (when pre- and
    postsynaptic neurons share a cluster) or the out-cluster weight
    ``minus·base_j`` (otherwise). Absent edges are ``0`` (a zero weight on an
    ``all_to_all`` projection is an absent connection).

    Parameters
    ----------
    npre, npost : int
        Pre- / postsynaptic population sizes.
    spre, spost : int
        Cluster sizes (neurons per cluster) for the pre / post populations; the
        cluster of neuron ``k`` is ``k // s``.
    base_j : float
        Base synaptic weight for this block (pA).
    plus, minus : float
        In-cluster / out-cluster weight factors.
    p : float
        Connection probability.
    no_auto : bool
        If ``True`` and the block is same-population (``npre == npost``), remove
        the diagonal (no autapses).
    seed : int
        PRNG seed for the Bernoulli mask.

    Returns
    -------
    numpy.ndarray
        ``(npre, npost)`` weight matrix (pA), ``0`` where there is no edge.
    """
    rng = np.random.RandomState(seed)
    cpre = np.arange(npre) // spre
    cpost = np.arange(npost) // spost
    same = cpre[:, None] == cpost[None, :]
    mask = rng.random((npre, npost)) < p
    if no_auto and npre == npost:
        np.fill_diagonal(mask, False)
    return np.where(same, plus, minus) * base_j * mask


def build(seed, *, rep=REP, rj=RJ, ne=N_E, ni=N_I, q=Q, dt=DT, comm='dense'):
    """Construct the clustered RBN simulator (E and I populations, 4 blocks).

    Parameters
    ----------
    seed : int
        Seeds the initial potentials and the per-block connectivity masks.
    rep : float, optional
        Excitatory clustering factor (``1`` = homogeneous). Default :data:`REP`.
    rj : float, optional
        Relative E/I clustering ratio. Default :data:`RJ`.
    ne, ni : int, optional
        Population sizes (must be divisible by ``q``). Default the module sizes.
    q : int, optional
        Number of clusters. Default :data:`Q`.
    dt : float, optional
        Resolution (ms). Default :data:`DT`.
    comm : str, optional
        Connectivity backend (``'dense'`` is fastest at these sizes).

    Returns
    -------
    sim : Simulator
        The configured simulator.
    esr, isr : NodeView
        Separate excitatory / inhibitory spike recorders.
    """
    n = ne + ni
    se, si = ne // q, ni // q
    js = rbn_weights(ne, ni)
    jee, jei = js[0, 0] / np.sqrt(n), js[0, 1] / np.sqrt(n)
    jie, jii = js[1, 0] / np.sqrt(n), js[1, 1] / np.sqrt(n)
    jplus, jminus = cluster_weights(rep, rj, q)
    ix_e = I_TH_E * rheobase(TAU_E, E_L, V_TH, C_M)
    ix_i = I_TH_I * rheobase(TAU_I, E_L, V_TH, C_M)
    rng = np.random.RandomState(seed)
    v_e = V_TH - 20.0 * rng.lognormal(0, 1, ne)
    v_i = V_TH - 20.0 * rng.lognormal(0, 1, ni)

    sim = Simulator(dt=dt * u.ms)
    exc = sim.create(iaf_psc_exp, ne, params=dict(
        E_L=E_L * u.mV, C_m=C_M * u.pF, tau_m=TAU_E * u.ms, t_ref=T_REF * u.ms,
        V_th=V_TH * u.mV, V_reset=V_R * u.mV, tau_syn_ex=TAU_SYN * u.ms,
        tau_syn_in=TAU_SYN * u.ms, I_e=ix_e * u.pA, V_initializer=v_e * u.mV))
    inh = sim.create(iaf_psc_exp, ni, params=dict(
        E_L=E_L * u.mV, C_m=C_M * u.pF, tau_m=TAU_I * u.ms, t_ref=T_REF * u.ms,
        V_th=V_TH * u.mV, V_reset=V_R * u.mV, tau_syn_ex=TAU_SYN * u.ms,
        tau_syn_in=TAU_SYN * u.ms, I_e=ix_i * u.pA, V_initializer=v_i * u.mV))
    esr = sim.create(spike_recorder)
    isr = sim.create(spike_recorder)

    def conn(pre, post, w):
        sim.connect(pre, post, rule=all_to_all, weight=w.flatten() * u.pA,
                    delay=DELAY * u.ms, comm=comm)
    conn(exc, exc, cluster_weight_matrix(ne, se, ne, se, jee, jplus[0, 0],
                                         jminus[0, 0], BCP[0, 0], True, 1000 * seed + 11))
    conn(inh, exc, cluster_weight_matrix(ni, si, ne, se, jei, jplus[0, 1],
                                         jminus[0, 1], BCP[0, 1], False, 1000 * seed + 22))
    conn(exc, inh, cluster_weight_matrix(ne, se, ni, si, jie, jplus[1, 0],
                                         jminus[1, 0], BCP[1, 0], False, 1000 * seed + 33))
    conn(inh, inh, cluster_weight_matrix(ni, si, ni, si, jii, jplus[1, 1],
                                         jminus[1, 1], BCP[1, 1], True, 1000 * seed + 44))
    sim.connect(exc, esr)
    sim.connect(inh, isr)
    return sim, esr, isr


def population_rate(raster, T):
    """Mean per-neuron firing rate (Hz) of a raster over a ``T`` ms window."""
    raster = np.asarray(raster)
    return float(raster.sum() / raster.shape[1] / (T / 1000.0))


def cluster_rate_std(e_raster, q, T):
    """Standard deviation (Hz) of the per-cluster mean firing rates.

    The metastability signature: near zero for a homogeneous network (all clusters
    fire alike), large when clustering drives winner-take-all dynamics.

    Parameters
    ----------
    e_raster : numpy.ndarray
        ``(n_steps, n_E)`` excitatory raster.
    q : int
        Number of clusters (``n_E`` must be divisible by ``q``).
    T : float
        Measurement window (ms).

    Returns
    -------
    float
        Across-cluster standard deviation of the per-cluster rates (Hz).
    """
    e_raster = np.asarray(e_raster)
    s = e_raster.shape[1] // q
    rates = [e_raster[:, c * s:(c + 1) * s].sum() / s / (T / 1000.0) for c in range(q)]
    return float(np.std(rates))


def cv_isi(raster, dt):
    """Mean coefficient of variation of the inter-spike intervals.

    Averaged over neurons with at least three spikes (two ISIs). ``CV ≈ 0`` for a
    perfectly regular train, ``≈ 1`` for a Poisson (memoryless) train.

    Parameters
    ----------
    raster : numpy.ndarray
        ``(n_steps, n_neurons)`` boolean raster.
    dt : float
        Resolution (ms).

    Returns
    -------
    float
        Mean ISI CV, or ``nan`` if no neuron has enough spikes.
    """
    raster = np.asarray(raster)
    cvs = []
    for j in range(raster.shape[1]):
        idx = np.flatnonzero(raster[:, j])
        if idx.size >= 3:
            isi = np.diff(idx) * dt
            cvs.append(np.std(isi) / np.mean(isi))
    return float(np.mean(cvs)) if cvs else float('nan')


def simulate(seed, *, rep=REP, rj=RJ, ne=N_E, ni=N_I, q=Q, warmup=WARMUP,
             simtime=SIMTIME, dt=DT, comm='dense'):
    """Run one realization and return its summary statistics.

    Parameters
    ----------
    seed : int
        Realization seed (initial potentials + connectivity masks).
    rep, rj : float, optional
        Clustering factors (see :func:`build`).
    ne, ni, q : int, optional
        Population sizes and cluster count.
    warmup : float, optional
        Transient discarded before statistics (ms). Default :data:`WARMUP`.
    simtime : float, optional
        Measured window after warm-up (ms). Default :data:`SIMTIME`.
    dt : float, optional
        Resolution (ms).
    comm : str, optional
        Connectivity backend.

    Returns
    -------
    dict
        ``e_rate`` / ``i_rate`` : float
            Excitatory / inhibitory population firing rates (Hz).
        ``cluster_std`` : float
            Across-cluster rate standard deviation (Hz) — the clustering signature.
        ``cv_e`` : float
            Mean excitatory ISI CV.
    """
    sim, esr, isr = build(seed, rep=rep, rj=rj, ne=ne, ni=ni, q=q, dt=dt, comm=comm)
    res = sim.simulate((warmup + simtime) * u.ms)
    w = int(round(warmup / dt))
    e_spk = np.asarray(res.spikes(esr))[w:] > 0
    i_spk = np.asarray(res.spikes(isr))[w:] > 0
    return dict(
        e_rate=population_rate(e_spk, simtime),
        i_rate=population_rate(i_spk, simtime),
        cluster_std=cluster_rate_std(e_spk, q, simtime),
        cv_e=cv_isi(e_spk, dt),
    )


#: Seed for the standalone demo. Avoids the occasional globally-synchronized
#: realization (e.g. seed 1) so the homogeneous control is a clean AI state.
DEMO_SEED = 2


def main():
    print("ei_clustered_network (brainpy.state, iaf_psc_exp RBN)")
    print(f"  N_E={N_E}, N_I={N_I}, Q={Q}, rep={REP}, rj={RJ}, "
          f"warmup={WARMUP} ms, simtime={SIMTIME} ms")
    for rep in (REP, 1.0):
        d = simulate(DEMO_SEED, rep=rep)
        tag = "clustered " if rep > 1 else "homogeneous"
        print(f"  rep={rep:>4}  [{tag}]  e_rate={d['e_rate']:5.2f}  "
              f"i_rate={d['i_rate']:5.2f} Hz  cluster-std={d['cluster_std']:5.2f}  "
              f"CV_E={d['cv_e']:.3f}")

    # Cluster-colored raster of the clustered network.
    sim, esr, isr = build(DEMO_SEED, rep=REP)
    res = sim.simulate((WARMUP + SIMTIME) * u.ms)
    w = int(round(WARMUP / DT))
    e_spk = np.asarray(res.spikes(esr))[w:] > 0
    try:
        import matplotlib.pyplot as plt
        se = N_E // Q
        plt.figure(figsize=(9, 5))
        t, nidx = np.nonzero(e_spk)
        colors = plt.cm.tab20(np.linspace(0, 1, Q))
        for c in range(Q):
            m = (nidx >= c * se) & (nidx < (c + 1) * se)
            plt.plot((t[m] + w) * DT, nidx[m], '.', ms=1.5, color=colors[c])
        plt.xlabel("time (ms)"); plt.ylabel("excitatory neuron id")
        plt.title(f"EI-clustered network (rep={REP}): metastable cluster activity")
        plt.tight_layout()
        plt.savefig("examples/nest_like/ei_clustered_network.png", dpi=100)
        print("  wrote examples/nest_like/ei_clustered_network.png")
    except ImportError:
        print("  (matplotlib not installed; skipping raster)")


if __name__ == "__main__":
    main()
