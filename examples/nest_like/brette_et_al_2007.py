# examples/nest_like/brette_et_al_2007.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brette et al. 2007 (Vogels-Abbott) COBA + CUBA benchmarks — NEST-style port.

Port of NEST's ``brette_et_al_2007/{coba,cuba}.py`` onto brainpy.state's explicit
``Simulator`` API. Both are the Vogels & Abbott (2005) [1]_ sparse self-sustained
excitatory/inhibitory network — the FACETS simulator-review benchmarks [2]_:

* **COBA** (benchmark 1): conductance-based ``iaf_cond_exp``. Excitatory input is
  routed to ``g_ex`` (``receptor_type=1``) and inhibitory input to ``g_in``
  (``receptor_type=2``); inhibition arises from the reversal potential ``E_in``,
  so the inhibitory weight is a **positive** conductance magnitude.
* **CUBA** (benchmark 2): current-based ``iaf_psc_exp``. The neuron splits
  excitatory/inhibitory postsynaptic currents by weight **sign** internally, so
  the inhibitory weight is **negative** and no receptor routing is needed.

Both share one architecture: ``NE=3200`` excitatory + ``NI=800`` inhibitory neurons,
connection probability ``epsilon=0.02`` (``fixed_indegree`` of ``CE=64`` excitatory
and ``CI=16`` inhibitory inputs onto every neuron), and a brief 50 ms Poisson **kick**
(300 Hz to the first 50 excitatory neurons) that ignites activity which then
self-sustains in an asynchronous-irregular state. The spike output of ``Nrec=500``
neurons per population is recorded; the reported observable is the mean per-neuron
population firing rate.

The Hodgkin-Huxley variant of this benchmark (NEST ``hh_coba.py``) already ships as
``examples/brainpy_like/106_COBA_HH_2007.py``.

Run:  python examples/nest_like/brette_et_al_2007.py

References
----------
.. [1] Vogels TP, Abbott LF. 2005. Signal propagation and logic gating in networks
       of integrate-and-fire neurons. J Neurosci. 25(46):10786-10795.
.. [2] Brette R, Rudolph M, Carnevale T, et al. 2007. Simulation of networks of
       spiking neurons: a review of tools and strategies. J Comput Neurosci.
       23(3):349-398.
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u
import braintools

from brainpy.state import (
    Simulator, fixed_indegree, one_to_one,
    iaf_cond_exp, iaf_psc_exp, poisson_generator, spike_recorder,
)

# --- Shared Vogels-Abbott architecture -------------------------------------------
NE, NI = 3200, 800          # excitatory / inhibitory population sizes
EPSILON = 0.02              # connection probability -> fixed in-degrees
NSTIM = 50                  # # excitatory neurons receiving the ignition kick
NREC = 500                  # # neurons per population recorded for the rate
DELAY = 0.1                 # synaptic delay [ms]
DT = 0.1                    # integration step [ms]
KICK_RATE = 300.0           # ignition Poisson rate [spikes/s]
KICK_START, KICK_STOP = 1.0, 51.0   # ignition window [ms]

# --- COBA (benchmark 1: iaf_cond_exp, conductance-based) -------------------------
COBA_PARAMS = dict(E_L=-60.0, V_th=-50.0, V_reset=-60.0, t_ref=5.0, E_ex=0.0,
                   E_in=-80.0, C_m=200.0, g_L=10.0, tau_syn_ex=5.0, tau_syn_in=10.0)
COBA_W_E = 6.0              # excitatory weight [nS]  -> receptor 1 (g_ex)
COBA_W_I = 67.0            # inhibitory weight [nS]  -> receptor 2 (g_in), positive magnitude
COBA_SIMTIME = 1000.0      # [ms]

# --- CUBA (benchmark 2: iaf_psc_exp, current-based) ------------------------------
CUBA_PARAMS = dict(E_L=-49.0, V_th=-50.0, V_reset=-60.0, C_m=200.0, tau_m=20.0,
                   tau_syn_ex=5.0, tau_syn_in=10.0, t_ref=5.0)
CUBA_VINIT = -49.0         # initial membrane potential [mV]
CUBA_W_E = 16.2            # excitatory PSC amplitude [pA]  (positive, sign-routed)
CUBA_W_I = -139.5          # inhibitory PSC amplitude [pA]  (negative, sign-routed)
CUBA_SIMTIME = 10000.0     # [ms] (10x longer than COBA: low load needs a longer window)


def connection_counts(ne=NE, ni=NI, epsilon=EPSILON):
    """Fixed in-degrees ``(CE, CI)`` for the Vogels-Abbott network.

    Parameters
    ----------
    ne, ni : int, optional
        Excitatory / inhibitory population sizes.
    epsilon : float, optional
        Connection probability; the in-degree is ``int(N * epsilon)``.

    Returns
    -------
    tuple of int
        ``(CE, CI)`` — excitatory and inhibitory inputs onto every neuron.

    Examples
    --------
    .. code-block:: python

        >>> from examples.nest_like.brette_et_al_2007 import connection_counts
        >>> connection_counts(3200, 800, 0.02)
        (64, 16)
    """
    return int(ne * epsilon), int(ni * epsilon)


def population_rate(raster, n_rec, simtime_ms):
    """Mean per-neuron firing rate [spikes/s] from a spike-count raster.

    Matches NEST's benchmark formula ``n_spikes / (n_rec * simtime) * 1000``.

    Parameters
    ----------
    raster : ArrayLike
        ``(T, N)`` per-step spike-count matrix (as returned by ``result.spikes``).
    n_rec : int
        Number of neurons the rate is averaged over.
    simtime_ms : float
        Duration of the spike count window [ms].

    Returns
    -------
    float
        Mean firing rate [spikes/s].

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from examples.nest_like.brette_et_al_2007 import population_rate
        >>> raster = np.zeros((1000, 2)); raster[::100] = 1   # 10 spikes/neuron in 100 ms
        >>> population_rate(raster, 2, 100.0)
        100.0
    """
    return float(np.asarray(raster).sum()) / (n_rec * simtime_ms) * 1000.0


def build_coba(seed, *, ne=NE, ni=NI, dt=DT, comm='sparse'):
    """Build the conductance-based (COBA) Vogels-Abbott network.

    Parameters
    ----------
    seed : int
        Seed for the kick generator and the four connectivity draws (offset per block).
    ne, ni : int, optional
        Excitatory / inhibitory population sizes.
    dt : float, optional
        Integration step [ms].
    comm : str, optional
        Connectivity backend, ``'sparse'`` (CSR) or ``'dense'``. Default ``'sparse'``.

    Returns
    -------
    tuple
        ``(sim, esr, isr)`` — the configured ``Simulator`` and the excitatory /
        inhibitory ``spike_recorder`` handles.
    """
    ce, ci = connection_counts(ne, ni)
    params = dict(
        E_L=COBA_PARAMS['E_L'] * u.mV, V_th=COBA_PARAMS['V_th'] * u.mV,
        V_reset=COBA_PARAMS['V_reset'] * u.mV, t_ref=COBA_PARAMS['t_ref'] * u.ms,
        E_ex=COBA_PARAMS['E_ex'] * u.mV, E_in=COBA_PARAMS['E_in'] * u.mV,
        C_m=COBA_PARAMS['C_m'] * u.pF, g_L=COBA_PARAMS['g_L'] * u.nS,
        tau_syn_ex=COBA_PARAMS['tau_syn_ex'] * u.ms, tau_syn_in=COBA_PARAMS['tau_syn_in'] * u.ms,
        V_initializer=braintools.init.Constant(COBA_PARAMS['E_L'] * u.mV))

    sim = Simulator(dt=dt * u.ms)
    e = sim.create(iaf_cond_exp, ne, params=params)
    i = sim.create(iaf_cond_exp, ni, params=params)
    kick = sim.create(poisson_generator, NSTIM, rate=KICK_RATE * u.Hz,
                      start=KICK_START * u.ms, stop=KICK_STOP * u.ms, rng_seed=seed)
    esr = sim.create(spike_recorder)
    isr = sim.create(spike_recorder)

    # Recurrent: every neuron gets CE excitatory (receptor 1 -> g_ex) and CI
    # inhibitory (receptor 2 -> g_in) inputs; inhibitory weight is a positive nS.
    sim.connect(e, e + i, weight=COBA_W_E * u.nS, delay=DELAY * u.ms,
                rule=fixed_indegree(ce), comm=comm, receptor_type=1,
                allow_multapses=True, seed=10 * seed + 1)
    sim.connect(i, e + i, weight=COBA_W_I * u.nS, delay=DELAY * u.ms,
                rule=fixed_indegree(ci), comm=comm, receptor_type=2,
                allow_multapses=True, seed=10 * seed + 2)
    # Ignition kick: independent 300 Hz trains -> first NSTIM excitatory neurons.
    sim.connect(kick, e[:NSTIM], weight=COBA_W_E * u.nS, delay=DELAY * u.ms,
                rule=one_to_one, receptor_type=1)
    sim.connect(e[:NREC], esr)
    sim.connect(i[:NREC], isr)
    return sim, esr, isr


def build_cuba(seed, *, ne=NE, ni=NI, dt=DT, comm='sparse'):
    """Build the current-based (CUBA) Vogels-Abbott network.

    Same architecture as :func:`build_coba` but with ``iaf_psc_exp`` neurons; the
    neuron routes excitatory/inhibitory PSCs by weight **sign**, so the inhibitory
    weight is negative and no receptor routing is used.

    Parameters
    ----------
    seed : int
        Seed for the kick generator and connectivity draws.
    ne, ni : int, optional
        Population sizes.
    dt : float, optional
        Integration step [ms].
    comm : str, optional
        Connectivity backend. Default ``'sparse'``.

    Returns
    -------
    tuple
        ``(sim, esr, isr)``.
    """
    ce, ci = connection_counts(ne, ni)
    params = dict(
        E_L=CUBA_PARAMS['E_L'] * u.mV, V_th=CUBA_PARAMS['V_th'] * u.mV,
        V_reset=CUBA_PARAMS['V_reset'] * u.mV, C_m=CUBA_PARAMS['C_m'] * u.pF,
        tau_m=CUBA_PARAMS['tau_m'] * u.ms, tau_syn_ex=CUBA_PARAMS['tau_syn_ex'] * u.ms,
        tau_syn_in=CUBA_PARAMS['tau_syn_in'] * u.ms, t_ref=CUBA_PARAMS['t_ref'] * u.ms,
        V_initializer=braintools.init.Constant(CUBA_VINIT * u.mV))

    sim = Simulator(dt=dt * u.ms)
    e = sim.create(iaf_psc_exp, ne, params=params)
    i = sim.create(iaf_psc_exp, ni, params=params)
    kick = sim.create(poisson_generator, NSTIM, rate=KICK_RATE * u.Hz,
                      start=KICK_START * u.ms, stop=KICK_STOP * u.ms, rng_seed=seed)
    esr = sim.create(spike_recorder)
    isr = sim.create(spike_recorder)

    # Recurrent: excitatory weight positive, inhibitory weight negative (sign-routed).
    sim.connect(e, e + i, weight=CUBA_W_E * u.pA, delay=DELAY * u.ms,
                rule=fixed_indegree(ce), comm=comm, allow_multapses=True, seed=10 * seed + 1)
    sim.connect(i, e + i, weight=CUBA_W_I * u.pA, delay=DELAY * u.ms,
                rule=fixed_indegree(ci), comm=comm, allow_multapses=True, seed=10 * seed + 2)
    sim.connect(kick, e[:NSTIM], weight=CUBA_W_E * u.pA, delay=DELAY * u.ms,
                rule=one_to_one)
    sim.connect(e[:NREC], esr)
    sim.connect(i[:NREC], isr)
    return sim, esr, isr


def _rates(res, esr, isr, simtime, n_rec=NREC):
    """Full-window and second-half (self-sustained) E/I rates from the recorders."""
    e_raster = np.asarray(res.spikes(esr.segments[0].population))   # (T, n_rec)
    i_raster = np.asarray(res.spikes(isr.segments[0].population))
    half = e_raster.shape[0] // 2
    return dict(
        e_rate=population_rate(e_raster, n_rec, simtime),
        i_rate=population_rate(i_raster, n_rec, simtime),
        e_rate_late=population_rate(e_raster[half:], n_rec, simtime / 2.0),
        i_rate_late=population_rate(i_raster[half:], n_rec, simtime / 2.0),
    )


def simulate_coba(seed, *, simtime=COBA_SIMTIME, ne=NE, ni=NI, dt=DT, comm='sparse'):
    """Build, run, and measure the COBA network for one seed.

    Parameters
    ----------
    seed : int
        Network seed.
    simtime : float, optional
        Simulated duration [ms].
    ne, ni : int, optional
        Population sizes.
    dt : float, optional
        Integration step [ms].
    comm : str, optional
        Connectivity backend.

    Returns
    -------
    dict
        ``e_rate``/``i_rate`` (full-window, NEST formula) and ``e_rate_late``/
        ``i_rate_late`` (second-half, evidence of self-sustained activity) [spikes/s].
    """
    sim, esr, isr = build_coba(seed, ne=ne, ni=ni, dt=dt, comm=comm)
    res = sim.simulate(simtime * u.ms)
    return _rates(res, esr, isr, simtime)


def simulate_cuba(seed, *, simtime=CUBA_SIMTIME, ne=NE, ni=NI, dt=DT, comm='sparse'):
    """Build, run, and measure the CUBA network for one seed.

    Parameters
    ----------
    seed : int
        Network seed.
    simtime : float, optional
        Simulated duration [ms].
    ne, ni : int, optional
        Population sizes.
    dt : float, optional
        Integration step [ms].
    comm : str, optional
        Connectivity backend.

    Returns
    -------
    dict
        ``e_rate``/``i_rate`` and ``e_rate_late``/``i_rate_late`` [spikes/s].
    """
    sim, esr, isr = build_cuba(seed, ne=ne, ni=ni, dt=dt, comm=comm)
    res = sim.simulate(simtime * u.ms)
    return _rates(res, esr, isr, simtime)


def main():
    print("Brette et al. 2007 (Vogels-Abbott) self-sustained E/I network — "
          f"NE={NE}, NI={NI}, CE/CI={connection_counts()}.")
    print(f"  COBA (iaf_cond_exp, conductance) simtime={COBA_SIMTIME} ms ...")
    coba = simulate_coba(seed=1)
    print(f"    E rate {coba['e_rate']:.2f} Hz | I rate {coba['i_rate']:.2f} Hz "
          f"(2nd-half E {coba['e_rate_late']:.2f} Hz -> self-sustained)")
    print(f"  CUBA (iaf_psc_exp, current) simtime={CUBA_SIMTIME} ms ...")
    cuba = simulate_cuba(seed=1)
    print(f"    E rate {cuba['e_rate']:.2f} Hz | I rate {cuba['i_rate']:.2f} Hz "
          f"(2nd-half E {cuba['e_rate_late']:.2f} Hz -> self-sustained)")


if __name__ == "__main__":
    main()
