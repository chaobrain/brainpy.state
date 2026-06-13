# examples/nest/gif_population.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Population of GIF neurons with adaptation-driven oscillation — NEST-style port.

Ports NEST's ``pynest/examples/gif_population.py`` to the Simulator API. A
recurrently-connected population of generalized integrate-and-fire
(``gif_psc_exp``) neurons is driven by a group of Poisson generators. Because the
GIF model has **spike-frequency adaptation** (slow ``stc`` spike-triggered current
and ``sfa`` threshold-movement elements), the population tends to oscillate on the
time scale of the adaptation constants: a burst of population activity drives the
adaptation up, which suppresses firing, which lets the adaptation decay, which
permits the next burst (Schwalger et al. 2017; Mensi et al. 2012).

``gif_psc_exp`` is a fully Simulator-compatible stochastic spiking model (its
escape-rate spiking draws a JAX PRNG carried as Simulator state), so the whole
``T = 2000 ms`` run is lowered into one compiled ``for_loop`` — no host-side
stepping.

**Connectivity note.** NEST wires the recurrent population with
``pairwise_bernoulli(p=0.3)``; the Simulator's equivalent fixed-mean-indegree rule
is ``fixed_indegree(K = round(p*N) = 30)`` — the same expected in-degree (30), so
the population-rate statistics agree distributionally. The Poisson group fans in
all-to-all, exactly as in NEST.

**Validation is a distributional carve-out, not a trace parity.** The GIF escape
spiking and the Poisson drive are PRNG streams that diverge between NEST and JAX,
so a per-sample comparison is meaningless; both simulators agree only
*distributionally*. ``brainpy_state/_nest/_validation/gif_population_test.py``
asserts (NEST-free) that the adaptation oscillation is present (the binned
population-rate autocorrelation dips below zero at an intermediate lag) and that
recurrence sharpens it, and — when NEST is present — that the seed-averaged
population rate and binned-rate autocorrelation match live NEST.

Run:  PYTHONPATH=. python examples/nest/gif_population.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import saiunit as u

from brainpy_state import Simulator, gif_psc_exp, poisson_generator, spike_recorder
from brainpy_state import fixed_indegree, all_to_all

DT = 0.1            # resolution [ms]
T_SIM = 2000.0      # simulation time [ms]

# GIF population (NEST gif_population.py verbatim).
N_EX = 100          # population size
P_EX = 0.3          # recurrent connection probability
W_EX = 30.0         # recurrent synaptic weight [pA]
K_EX = round(P_EX * N_EX)   # fixed in-degree matching mean of pairwise_bernoulli(p)

# Poisson drive.
N_NOISE = 50        # number of Poisson generators
RATE_NOISE = 10.0   # Poisson rate [Hz]
W_NOISE = 20.0      # Poisson -> population weight [pA]

DELAY = 1.0         # connection delay [ms] (NEST default)

# gif_psc_exp parameters (NEST gif_population.py; membrane unit-wrapped, the
# adaptation/threshold elements are plain sequences in ms / mV / pA, lambda_0 in 1/s).
NEURON_PARAMS = dict(
    C_m=83.1 * u.pF,
    g_L=3.7 * u.nS,
    E_L=-67.0 * u.mV,
    Delta_V=1.4 * u.mV,
    V_T_star=-39.6 * u.mV,
    t_ref=4.0 * u.ms,
    V_reset=-36.7 * u.mV,
    lambda_0=1.0,                       # [1/s]
    q_stc=[56.7, -6.9],                 # [pA] spike-triggered current amplitudes
    tau_stc=[57.8, 218.2],              # [ms] stc time constants
    q_sfa=[11.7, 1.8],                  # [mV] threshold-movement amplitudes
    tau_sfa=[53.8, 640.0],              # [ms] sfa time constants
    tau_syn_ex=10.0 * u.ms,
)


def build(seed=0, *, recurrent=True, n_ex=N_EX, t_sim=T_SIM):
    """Build the GIF population + Poisson drive (+ optional recurrence).

    Parameters
    ----------
    seed : int, optional
        Base PRNG seed (threads both the GIF escape spiking and the Poisson
        streams). Default ``0``.
    recurrent : bool, optional
        If ``True`` (default) wire the ``fixed_indegree(K_EX)`` recurrence; if
        ``False`` build the unconnected control (Poisson drive only).
    n_ex : int, optional
        Population size. Default :data:`N_EX`.
    t_sim : float, optional
        Simulation horizon in ms. Default :data:`T_SIM`.

    Returns
    -------
    sim : Simulator
    sr : NodeView
        Spike recorder on the population (``res.spikes(sr)`` / ``res.rate(sr)``).
    t_sim : float
    """
    sim = Simulator(dt=DT * u.ms)
    pop = sim.create(gif_psc_exp, n_ex,
                     params=dict(NEURON_PARAMS, rng_key=jax.random.PRNGKey(seed)))
    noise = sim.create(poisson_generator, N_NOISE, rate=RATE_NOISE * u.Hz, rng_seed=seed)
    sr = sim.create(spike_recorder)
    if recurrent:
        sim.connect(pop, pop, rule=fixed_indegree(K_EX), weight=W_EX * u.pA,
                    delay=DELAY * u.ms)
    sim.connect(noise, pop, rule=all_to_all, weight=W_NOISE * u.pA, delay=DELAY * u.ms)
    sim.connect(pop, sr)
    return sim, sr, t_sim


def population_activity(spk, dt_ms=DT, bin_ms=5.0, n_neurons=None):
    """Binned mean population firing rate (spk/s per neuron).

    Parameters
    ----------
    spk : array_like
        ``(n_steps, n_neurons)`` spike matrix.
    dt_ms : float, optional
        Simulation step in ms. Default :data:`DT`.
    bin_ms : float, optional
        Bin width in ms. Default ``5.0``.
    n_neurons : int or None, optional
        Population size for the per-neuron normalization (defaults to
        ``spk.shape[1]``).

    Returns
    -------
    numpy.ndarray
        Per-bin mean rate (spk/s), one entry per ``bin_ms`` bin.
    """
    spk = np.asarray(spk)
    counts = spk.sum(axis=1)
    steps_per_bin = int(round(bin_ms / dt_ms))
    nbins = counts.size // steps_per_bin
    counts = counts[:nbins * steps_per_bin].reshape(nbins, steps_per_bin).sum(axis=1)
    n = n_neurons if n_neurons is not None else spk.shape[1]
    return counts / (n * bin_ms / 1000.0)


def activity_from_times(times_ms, n_neurons, t_sim=T_SIM, bin_ms=5.0):
    """Binned mean population rate (spk/s/neuron) from a flat spike-time array.

    The NEST-side analogue of :func:`population_activity` (NEST returns spike times,
    not a per-step matrix).
    """
    edges = np.arange(0.0, t_sim + bin_ms, bin_ms)
    counts, _ = np.histogram(np.asarray(times_ms, dtype=float), bins=edges)
    return counts / (n_neurons * bin_ms / 1000.0)


def autocorr(x, max_lag):
    """Normalized autocorrelation of ``x`` for lags ``0..max_lag`` (``ac[0]=1``).

    Parameters
    ----------
    x : array_like
        1-D signal (here the binned population rate).
    max_lag : int
        Maximum lag in bins.

    Returns
    -------
    numpy.ndarray
        Length-``max_lag+1`` autocorrelation; all-zero for a constant signal.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0.0:
        return np.zeros(max_lag + 1)
    return np.array([float(np.sum(x[:x.size - k] * x[k:])) / denom
                     for k in range(max_lag + 1)])


def run_population(seed=0, *, recurrent=True, n_ex=N_EX, t_sim=T_SIM, bin_ms=5.0):
    """Run the network and return the population-activity summary.

    Returns
    -------
    dict
        ``rate_hz`` (mean population rate), ``binned_rate`` (spk/s per ``bin_ms``
        bin), ``spk`` (``(T, n)`` matrix), ``t`` (ms axis), ``n_ex``, ``bin_ms``.
    """
    sim, sr, t_sim = build(seed, recurrent=recurrent, n_ex=n_ex, t_sim=t_sim)
    res = sim.simulate(t_sim * u.ms)
    spk = np.asarray(res.spikes(sr))
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    return dict(rate_hz=float(res.rate(sr)),
                binned_rate=population_activity(spk, DT, bin_ms, n_ex),
                spk=spk, t=t, n_ex=n_ex, bin_ms=bin_ms)


def main():
    print("Population of GIF neurons with adaptation-driven oscillation "
          "(brainpy.state)")
    rec = run_population(seed=0, recurrent=True)
    ctl = run_population(seed=0, recurrent=False)
    max_lag = int(round(500.0 / rec['bin_ms']))         # 500 ms of lags
    ac_rec = autocorr(rec['binned_rate'], max_lag)
    print(f"  mean population rate : {rec['rate_hz']:.2f} spk/s "
          f"({rec['spk'].sum():.0f} spikes, N={rec['n_ex']})")
    print(f"  binned-rate fluctuation (std/mean): recurrent={_cv(rec['binned_rate']):.3f}"
          f"  control={_cv(ctl['binned_rate']):.3f}")
    dip = float(ac_rec[1:].min())
    print(f"  adaptation oscillation: autocorr min over lags = {dip:.3f} "
          f"at lag {int(np.argmin(ac_rec[1:])) + 1} bins "
          f"({(int(np.argmin(ac_rec[1:])) + 1) * rec['bin_ms']:.0f} ms)")

    try:
        import matplotlib.pyplot as plt
        spk, t = rec['spk'], rec['t']
        steps, ids = np.nonzero(spk > 0)
        tb = (np.arange(rec['binned_rate'].size) + 0.5) * rec['bin_ms']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax1.scatter(t[steps], ids, s=1.0, color="k")
        ax1.set_ylabel("GIF neuron")
        ax1.set_title("Population dynamics (gif_psc_exp)")
        ax2.plot(tb, rec['binned_rate'], color="C0", label="recurrent")
        ax2.plot(tb, ctl['binned_rate'], color="C3", alpha=0.6, label="control (no recurrence)")
        ax2.set_xlabel("time [ms]")
        ax2.set_ylabel("population rate [spk/s]")
        ax2.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig("examples/nest/gif_population.png", dpi=100)
        print("  wrote examples/nest/gif_population.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


def _cv(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    return float(x.std() / m) if m > 0 else 0.0


if __name__ == "__main__":
    main()
