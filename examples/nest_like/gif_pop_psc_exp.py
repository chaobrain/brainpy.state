# examples/nest_like/gif_pop_psc_exp.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Mesoscopic GIF population-rate network vs. its microscopic realization.

Ports NEST's ``pynest/examples/gif_pop_psc_exp.py`` to brainpy.state. A finite
Brunel-style network of two coupled populations (one excitatory, one inhibitory)
of generalized integrate-and-fire neurons is simulated **two ways**:

* **Mesoscopic** — one :class:`gif_pop_psc_exp` unit per population. This is the
  effective stochastic *population-rate* model of Schwalger et al. (2017): it
  emits a population spike count ``n_spikes`` per step **without** simulating the
  individual neurons.
* **Microscopic** — the corresponding network of individual ``gif_psc_exp``
  neurons (800 excitatory + 200 inhibitory), run on the Simulator.

The scientific claim the demo reproduces (NEST figures 1 vs 2): the mesoscopic
population activity ``A_N(t)`` *looks like* the microscopic one — they agree
**distributionally** (mean rate per population, fluctuation autocorrelation, and
the step-evoked rate jump at ``t = 1500 ms``), even though neither matches the
other sample-by-sample.

**Why the mesoscopic half uses a host-side Python loop.**
:class:`gif_pop_psc_exp` is a *host-side* model: its state lives in NumPy and its
stochastic spike count is drawn with ``numpy.random`` (a finite-N binomial), so it
is **not** a JAX-traceable ``brainstate`` module and cannot be lowered into a
``for_loop``. It is therefore driven exactly as its own unit tests drive it — a
plain Python loop calling :meth:`gif_pop_psc_exp.update` once per step (the
explicit carve-out in working-agreement rule #10 for untraceable models). The
**microscopic** half *is* fully traceable and runs in one compiled ``for_loop``.

**Coupling conventions (meso vs micro).** Both sides use the same effective
synaptic weights ``J_syn * g_syn`` (``J_syn`` already folds the finite-size
``1/C`` scaling of Schwalger et al.). They differ only in how the connection
probability ``pconn`` enters:

* meso multiplies the weight by ``pconn`` (mean field: a population of rate ``A``
  delivers expected weighted input ``pconn * N * w * A``), realized as one delayed
  delta input per source population: ``w_meso[i,j] = J_syn[i,j]*g_syn[i,j]*pconn``;
* micro realizes ``pconn`` as the *connection density* ``fixed_indegree(pconn*N)``,
  so its per-synapse weight omits the ``pconn`` factor:
  ``w_micro[i,j] = J_syn[i,j]*g_syn[i,j]``.

Net mean input matches. The sign of ``J_syn`` (negative for the inhibitory source
column) routes the input to the inhibitory synaptic channel on both sides:
mesoscopically via :meth:`gif_pop_psc_exp.update`'s sign split, microscopically via
``gif_psc_exp``'s ``_sum_signed_delta_inputs`` (negative ``connect`` weight ->
inhibitory channel).

Run:  PYTHONPATH=. python examples/nest_like/gif_pop_psc_exp.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy_state import (Simulator, gif_psc_exp, gif_pop_psc_exp,
                           step_current_generator, spike_recorder)
from brainpy_state import fixed_indegree

# ---------------------------------------------------------------------------
# Parameters (NEST gif_pop_psc_exp.py verbatim).
# ---------------------------------------------------------------------------
DT = 0.5            # simulation resolution [ms]
DT_REC = 1.0        # activity recording bin [ms]
T_END = 2000.0      # simulation time [ms]

M = 2                       # number of populations (exc, inh)
N = (800, 200)              # population sizes  ((4, 1) * 200)

# Neuronal parameters (shared by both populations).
T_REF = 4.0         # absolute refractory period [ms]
TAU_M = 20.0        # membrane time constant [ms]
MU = 24.0           # base current  mu = R*(I0+Vrest)  [mV]
LAMBDA_0 = 10.0     # base rate of the exponential link function [Hz]
DELTA_U = 2.5       # softness of the exponential link function [mV]
V_RESET = 0.0       # reset potential [mV]
V_TH = 15.0         # baseline threshold [mV]
TAU_SFA = (100.0, 1000.0)   # adaptation time constants [ms]
J_SFA = (1000.0, 1000.0)    # feedback-kernel areas theta [mV*ms]
C_M = 250.0         # membrane capacity [pF] (cancels out; sets current scale)
E_L = 0.0           # leak reversal [mV]
TAU_EX = 3.0        # excitatory PSC time constant [ms]
TAU_IN = 6.0        # inhibitory PSC time constant [ms]

# Connectivity.
J = 0.3             # excitatory synaptic weight [mV] at reference in-degree C0
G = 5.0             # inhibition-to-excitation ratio
PCONN = 0.2         # connection probability (all population pairs)
DELAY = 1.0         # synaptic delay [ms]

# Step current input: a +20 mV jump in mu at t = 1500 ms.
STEP_MV = 20.0
T_STEP = 1500.0

# ---- derived quantities ----
G_L = C_M / TAU_M                                  # 12.5  [pA/mV]
I_E = MU * G_L                                     # 300.0 [pA] constant base current
Q_SFA = tuple(J_SFA[k] / TAU_SFA[k] for k in range(2))   # (10.0, 1.0) [mV]
STEP_AMP = STEP_MV * G_L                           # 250.0 [pA]
DELAY_STEPS = int(round(DELAY / DT))               # 2 steps

# Synaptic weight matrices.  C0 == C here, so J_syn = [[J, -gJ], [J, -gJ]].
_C0 = np.array([[800, 200], [800, 200]], dtype=float) * 0.2
_C = np.vstack((N, N)).astype(float) * PCONN
J_SYN = np.array([[J, -G * J], [J, -G * J]], dtype=float) * _C0 / _C
G_SYN = np.ones((M, M)); G_SYN[:, 0] = C_M / TAU_EX; G_SYN[:, 1] = C_M / TAU_IN
W_MESO = J_SYN * G_SYN * PCONN          # meso: weight folds pconn
W_MICRO = J_SYN * G_SYN                 # micro: pconn realized as connection density


# ===========================================================================
# Mesoscopic simulation (host-side population-rate model).
# ===========================================================================
def _make_meso_pop(i, seed):
    """One :class:`gif_pop_psc_exp` population unit (host-side model)."""
    return gif_pop_psc_exp(
        1, N=N[i], tau_m=TAU_M, C_m=C_M, t_ref=T_REF, lambda_0=LAMBDA_0,
        Delta_V=DELTA_U, E_L=E_L, V_reset=V_RESET, V_T_star=V_TH, I_e=I_E,
        tau_syn_ex=TAU_EX, tau_syn_in=TAU_IN, tau_sfa=TAU_SFA, q_sfa=Q_SFA,
        rng_seed=seed + i)


def run_meso(seed=1, *, coupled=True, t_end=T_END):
    r"""Run the mesoscopic two-population network with a host-side Python loop.

    :class:`gif_pop_psc_exp` is host-side NumPy (not JAX-traceable), so it is
    stepped with a plain Python loop -- the documented rule-#10 carve-out for
    untraceable models. Recurrent coupling is delivered as one delayed delta
    input per source population; the step current is delivered through
    :meth:`gif_pop_psc_exp.update`'s ``x`` argument.

    Parameters
    ----------
    seed : int, optional
        Base seed for the per-population NumPy RNGs. Default ``1``.
    coupled : bool, optional
        If ``True`` (default) wire the recurrent ``pconn``-weighted coupling; if
        ``False`` run the populations uncoupled (base dynamics only).
    t_end : float, optional
        Simulation horizon in ms. Default :data:`T_END`.

    Returns
    -------
    dict
        ``A_N`` (``(n_bins, M)`` population activity in spk/s), ``t`` (bin centres
        in ms).
    """
    nsteps = int(round(t_end / DT))
    steps_per_bin = int(round(DT_REC / DT))
    nbins = nsteps // steps_per_bin
    n_arr = np.array(N, dtype=float)

    with brainstate.environ.context(dt=DT * u.ms):
        pops = [_make_meso_pop(i, seed) for i in range(M)]
        for p in pops:
            p.init_state()

        A_N = np.zeros((nbins, M))
        delay_buf = np.zeros((DELAY_STEPS, M))   # ring buffer of past n_spikes
        nspk = np.zeros(M)
        bin_acc = np.zeros(M)

        # Host-side stepping loop (rule #10 carve-out: host-side model).
        for s in range(nsteps):
            t = s * DT
            slot = s % DELAY_STEPS
            delayed = delay_buf[slot].copy()       # n_spikes emitted DELAY_STEPS ago
            for i in range(M):
                if coupled:
                    for j in range(M):
                        pops[i].add_delta_input(f'rec{j}', float(W_MESO[i, j] * delayed[j]))
                x = STEP_AMP if t >= T_STEP else 0.0
                nspk[i] = pops[i].update(x=x)
            delay_buf[slot] = nspk                 # store current emission for the future
            bin_acc += nspk
            if (s + 1) % steps_per_bin == 0:
                b = (s + 1) // steps_per_bin - 1
                A_N[b] = bin_acc * 1000.0 / (n_arr * DT_REC)
                bin_acc = np.zeros(M)

    return dict(A_N=A_N, t=(np.arange(nbins) + 1) * DT_REC)


# ===========================================================================
# Microscopic simulation (Simulator network of individual gif_psc_exp neurons).
# ===========================================================================
def build_micro(seed=1, *, t_end=T_END):
    """Build the microscopic ``gif_psc_exp`` network on the Simulator.

    Returns
    -------
    sim : Simulator
    srs : list of NodeView
        Per-population spike recorders.
    t_end : float
    """
    sim = Simulator(dt=DT * u.ms)
    pops = [
        sim.create(gif_psc_exp, N[i], params=dict(
            C_m=C_M * u.pF, g_L=G_L * u.nS, E_L=E_L * u.mV, Delta_V=DELTA_U * u.mV,
            V_T_star=V_TH * u.mV, V_reset=V_RESET * u.mV, t_ref=T_REF * u.ms,
            lambda_0=LAMBDA_0, tau_sfa=TAU_SFA, q_sfa=Q_SFA,
            tau_syn_ex=TAU_EX * u.ms, tau_syn_in=TAU_IN * u.ms, I_e=I_E * u.pA,
            rng_key=jax.random.PRNGKey(seed + i)))
        for i in range(M)
    ]
    srs = [sim.create(spike_recorder) for _ in range(M)]
    for i in range(M):
        for j in range(M):
            # pconn realized as connection density; sign of the weight routes
            # the input to the exc (+) or inh (-) channel.
            sim.connect(pops[j], pops[i], rule=fixed_indegree(int(round(PCONN * N[j]))),
                        weight=float(W_MICRO[i, j]) * u.pA, delay=DELAY * u.ms)
        step = sim.create(step_current_generator,
                          amplitude_times=np.array([DT, T_STEP]) * u.ms,
                          amplitude_values=np.array([0.0, STEP_AMP]) * u.pA)
        sim.connect(step, pops[i])
        sim.connect(pops[i], srs[i])
    return sim, srs, t_end


def run_micro(seed=1, *, t_end=T_END):
    r"""Run the microscopic network (one compiled ``for_loop``) -> ``A_N``.

    Returns
    -------
    dict
        ``A_N`` (``(n_bins, M)`` population activity in spk/s), ``t`` (bin centres
        in ms).
    """
    sim, srs, t_end = build_micro(seed, t_end=t_end)
    res = sim.simulate(t_end * u.ms)
    steps_per_bin = int(round(DT_REC / DT))
    A_N = None
    for i in range(M):
        counts = np.asarray(res.spikes(srs[i])).sum(axis=1)     # per-step spike count
        binned = _bin_counts(counts, steps_per_bin)
        if A_N is None:
            A_N = np.zeros((binned.size, M))
        A_N[:, i] = binned * 1000.0 / (N[i] * DT_REC)
    return dict(A_N=A_N, t=(np.arange(A_N.shape[0]) + 1) * DT_REC)


# ===========================================================================
# Shared analysis helpers.
# ===========================================================================
def _bin_counts(per_step_counts, steps_per_bin):
    """Sum a per-step count vector into ``steps_per_bin``-wide bins."""
    c = np.asarray(per_step_counts, dtype=float)
    nb = c.size // steps_per_bin
    return c[:nb * steps_per_bin].reshape(nb, steps_per_bin).sum(axis=1)


def autocorr(x, max_lag):
    """Normalized autocorrelation of ``x`` for lags ``0..max_lag`` (``ac[0]=1``)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0.0:
        return np.zeros(max_lag + 1)
    return np.array([float(np.sum(x[:x.size - k] * x[k:])) / denom
                     for k in range(max_lag + 1)])


def window_rate(act, t, t0, t1):
    """Mean activity (spk/s) over the time window ``[t0, t1)`` ms."""
    mask = (t >= t0) & (t < t1)
    return float(np.asarray(act)[mask].mean())


def main():
    print("Mesoscopic GIF population network vs microscopic realization "
          "(brainpy.state)")
    meso = run_meso(seed=1)
    micro = run_micro(seed=1)

    for label, r in (("mesoscopic", meso), ("microscopic", micro)):
        t = r['t']
        a_ex = r['A_N'][:, 0]
        pre = window_rate(a_ex, t, 1000.0, T_STEP)
        post = window_rate(a_ex, t, T_STEP, T_END)
        print(f"  {label:11s}: exc rate pre-step={pre:6.2f}  post-step={post:6.2f} spk/s "
              f"(jump x{post / pre:4.2f})" if pre > 0 else f"  {label}: exc rate pre={pre}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for k, (label, r) in enumerate((("mesoscopic", meso), ("microscopic", micro))):
            ax[k].plot(r['t'], r['A_N'][:, 0], color="C0", lw=0.8, label="exc")
            ax[k].plot(r['t'], r['A_N'][:, 1], color="C3", lw=0.8, alpha=0.7, label="inh")
            ax[k].axvline(T_STEP, color="k", ls=":", lw=0.8)
            ax[k].set_ylabel(r"$A_N$ [spk/s]")
            ax[k].set_title(f"Population activity ({label} sim.)")
            ax[k].legend(loc="upper left", fontsize=8)
        ax[1].set_xlabel("time [ms]")
        fig.tight_layout()
        fig.savefig("examples/nest_like/gif_pop_psc_exp.png", dpi=100)
        print("  wrote examples/nest_like/gif_pop_psc_exp.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
