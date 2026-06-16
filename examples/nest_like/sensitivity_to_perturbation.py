# examples/nest_like/sensitivity_to_perturbation.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Sensitivity to perturbation in a balanced E/I network — NEST-style port.

Port of NEST's ``sensitivity_to_perturbation.py``. A Brunel-style sparse balanced
network of ``iaf_psc_delta`` neurons (excitatory + inhibitory, random
``fixed_indegree`` connectivity, independent external Poisson drive) is simulated
in **two trials that are identical except for one extra input spike** at
``t_stim``, injected into the first neuron to fire after ``t_stim``. In the
balanced regime the network is chaotic, so that single spike can decorrelate the
whole population — a discrete-network analogue of London et al. (2010).

The transition to chaos is *probabilistic*: each perturbation either dies out
(the trajectories re-converge) or triggers a near-complete decorrelation, the
outcome depending on the realization. The headline statistic is the **divergence**
between the two trials — zero before ``t_stim`` (the trials share network, initial
state and external drive), then growing after the perturbation in the chaotic case.

Implementation notes
--------------------
* The external drive is a **single** ``poisson_generator`` broadcast to all
  neurons; NEST (and brainpy.state) deliver an *independent* realization to each
  target, so one generator suffices and is far cheaper than one-per-neuron.
* Determinism is engineered so the two trials are bit-identical before ``t_stim``:
  the recurrent connections are sampled under fixed ``seed``\\ s, the initial
  potentials come from a seeded array, and the Poisson drive uses a fixed
  ``rng_seed``. The perturbation ``spike_generator`` is added last so it cannot
  perturb the shared network's construction.

Run:  PYTHONPATH=. python examples/nest_like/sensitivity_to_perturbation.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy_state import (
    Simulator, fixed_indegree, all_to_all,
    iaf_psc_delta, poisson_generator, spike_generator, spike_recorder,
)

#: Number of excitatory / inhibitory neurons (NEST defaults).
NE = 1000
NI = 250
#: Excitatory / inhibitory in-degree (synapses per postsynaptic neuron).
KE = 100
KI = 25
#: Excitatory synaptic weight (mV; delta-PSP jump) and relative inhibitory weight.
J = 0.5
G = 6.0
#: External Poisson drive: per-spike PSP amplitude (mV) and rate (Hz).
JEXT = 0.2
RATE_EXT = 6500.0
#: Perturbation: time of the extra spike (ms) and its amplitude (mV).
T_STIM = 400.0
JSTIM = JEXT
#: Simulation horizon (ms) and resolution (ms). NEST's original uses dt = 0.01;
#: this port uses 0.1 (10x coarser) for tractability — the sensitivity is dt-robust.
T_SIM = 1000.0
DT = 0.1
#: ``iaf_psc_delta`` parameters (NEST defaults).
C_M = 250.0
TAU_M = 10.0
T_REF = 2.0
E_L = -70.0
V_RESET = -70.0
V_TH = -55.0
#: Initial-potential band: uniform on ``[E_L, V_th]`` (the NEST convention).
VMIN = E_L
VMAX = V_TH


def initial_voltages(n, seed):
    """Seeded uniform initial membrane potentials on ``[VMIN, VMAX]``.

    Parameters
    ----------
    n : int
        Number of neurons.
    seed : int
        NumPy PRNG seed; the same seed reproduces the same potentials, which is
        what makes the two trials share an initial state.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` initial potentials in mV.
    """
    rng = np.random.RandomState(seed)
    return VMIN + (VMAX - VMIN) * rng.rand(n)


def build(seed, perturb_id=None, *, ne=NE, ni=NI, ke=KE, ki=KI, j=J, g=G,
          jext=JEXT, rate_ext=RATE_EXT, t_stim=T_STIM, dt=DT, T=T_SIM, comm='dense'):
    """Build one trial of the balanced network (optionally perturbed).

    Parameters
    ----------
    seed : int
        Seed shared by the connectivity sampling, the initial potentials and the
        external Poisson drive, so two builds with the same ``seed`` differ only in
        ``perturb_id``.
    perturb_id : int, optional
        If given, a ``spike_generator`` emitting one spike at ``t_stim`` is wired
        into neuron ``perturb_id`` (weight ``jext``). ``None`` (default) is the
        unperturbed trial.
    ne, ni : int, optional
        Excitatory / inhibitory population sizes.
    ke, ki : int, optional
        Excitatory / inhibitory in-degrees.
    j, g : float, optional
        Excitatory weight (mV) and relative inhibitory weight (``-g*j``).
    jext, rate_ext : float, optional
        External-drive PSP amplitude (mV) and rate (Hz).
    t_stim : float, optional
        Perturbation time (ms).
    dt : float, optional
        Resolution (ms); also the synaptic delay.
    T : float, optional
        Simulation horizon (ms); the external drive stops at ``T``.
    comm : str, optional
        Connectivity backend (``'dense'`` is fastest for ``N <= ~1250``).

    Returns
    -------
    sim : Simulator
        The configured simulator.
    sr : NodeView
        Spike recorder tapping the whole population.
    """
    n = ne + ni
    sim = Simulator(dt=dt * u.ms)
    v0 = initial_voltages(n, seed) * u.mV
    npar = dict(C_m=C_M * u.pF, tau_m=TAU_M * u.ms, t_ref=T_REF * u.ms,
                E_L=E_L * u.mV, V_reset=V_RESET * u.mV, V_th=V_TH * u.mV,
                V_initializer=v0)
    pop = sim.create(iaf_psc_delta, n, params=npar)
    ne_v, ni_v = pop[:ne], pop[ne:]
    ext = sim.create(poisson_generator, 1, rate=rate_ext * u.Hz, stop=T * u.ms,
                     rng_seed=seed)
    sr = sim.create(spike_recorder)
    # One generator broadcast → independent per-target Poisson streams (NEST semantics).
    sim.connect(ext, pop, weight=jext * u.mV, delay=dt * u.ms, rule=all_to_all)
    sim.connect(ne_v, pop, weight=j * u.mV, delay=dt * u.ms,
                rule=fixed_indegree(ke), comm=comm, allow_multapses=True, seed=1001)
    sim.connect(ni_v, pop, weight=-g * j * u.mV, delay=dt * u.ms,
                rule=fixed_indegree(ki), comm=comm, allow_multapses=True, seed=2002)
    if perturb_id is not None:
        stim = sim.create(spike_generator, 1, spike_times=[t_stim] * u.ms)
        sim.connect(stim, pop[int(perturb_id):int(perturb_id) + 1],
                    weight=jext * u.mV, delay=dt * u.ms)
    sim.connect(pop, sr)
    return sim, sr


def run_raster(seed, perturb_id=None, *, T=T_SIM, **kw):
    """Simulate one trial and return its per-step spike raster.

    Parameters
    ----------
    seed : int
        Trial seed (see :func:`build`).
    perturb_id : int, optional
        Perturbed neuron, or ``None`` for the unperturbed trial.
    T : float, optional
        Simulation horizon (ms).
    **kw
        Forwarded to :func:`build` (``ne``, ``ni``, ``ke``, ``ki``, ``dt``, ...).

    Returns
    -------
    numpy.ndarray
        ``(n_steps, n_neurons)`` boolean raster (``True`` where a neuron spiked).
    """
    sim, sr = build(seed, perturb_id=perturb_id, T=T, **kw)
    res = sim.simulate(T * u.ms)
    return np.asarray(res.spikes(sr)) > 0


def first_spike_after(raster, t_split, dt=DT):
    """Index of the first neuron to fire after ``t_split``.

    Returns the lowest-id neuron firing at the earliest step on/after ``t_split``
    — the perturbation target, matching NEST's "first neuron to fire after
    ``t_stim``".

    Parameters
    ----------
    raster : numpy.ndarray
        ``(n_steps, n_neurons)`` boolean raster.
    t_split : float
        Split time (ms).
    dt : float, optional
        Resolution (ms).

    Returns
    -------
    int or None
        Neuron index, or ``None`` if no neuron fires after ``t_split``.
    """
    raster = np.asarray(raster)
    step0 = int(round(t_split / dt))
    post = raster[step0:]
    rows = np.nonzero(post.any(axis=1))[0]
    if rows.size == 0:
        return None
    return int(np.nonzero(post[rows[0]])[0][0])


def divergence(r0, r1, t_stim, dt=DT):
    """Quantify how two trials' spike rasters diverge around ``t_stim``.

    Parameters
    ----------
    r0, r1 : numpy.ndarray
        ``(n_steps, n_neurons)`` boolean rasters of the unperturbed and perturbed
        trials.
    t_stim : float
        Perturbation time (ms); splits the "before" (must be identical) and
        "after" windows.
    dt : float, optional
        Resolution (ms).

    Returns
    -------
    dict
        ``d_before`` : int
            Number of differing ``(step, neuron)`` spikes before ``t_stim``
            (``0`` when the trials are correctly identical up to the perturbation).
        ``d_after`` : int
            Differing spikes after ``t_stim`` (grows in the chaotic case).
        ``frac_decorr`` : float
            Fraction of neurons whose spike train differs at some point after
            ``t_stim`` (``~0`` non-chaotic, ``~1`` fully decorrelated).
        ``onset_ms`` : float or None
            Time of the first differing step (``None`` if the trials never differ).
    """
    r0 = np.asarray(r0)
    r1 = np.asarray(r1)
    step_stim = int(round(t_stim / dt))
    diff = (r0 != r1)
    after = diff[step_stim:]
    onset = np.nonzero(diff.any(axis=1))[0]
    return dict(
        d_before=int(diff[:step_stim].sum()),
        d_after=int(after.sum()),
        frac_decorr=float(after.any(axis=0).mean()) if after.size else 0.0,
        onset_ms=float(onset[0] * dt) if onset.size else None,
    )


def population_rate(raster, T):
    """Mean per-neuron firing rate (Hz) of a raster over a window of length ``T`` ms."""
    raster = np.asarray(raster)
    return float(raster.sum() / raster.shape[1] / (T / 1000.0))


def network_rate(seed, *, T=T_SIM, **kw):
    """Population firing rate (Hz) of one unperturbed trial.

    Parameters
    ----------
    seed : int
        Trial seed.
    T : float, optional
        Simulation horizon (ms).
    **kw
        Forwarded to :func:`build`.

    Returns
    -------
    float
        Mean per-neuron rate in Hz.
    """
    return population_rate(run_raster(seed, T=T, **kw), T)


def run_two_trials(seed, *, t_stim=T_STIM, dt=DT, T=T_SIM, **kw):
    """Run the unperturbed and perturbed trials and return both rasters.

    Parameters
    ----------
    seed : int
        Shared trial seed.
    t_stim : float, optional
        Perturbation time (ms).
    dt : float, optional
        Resolution (ms).
    T : float, optional
        Simulation horizon (ms).
    **kw
        Forwarded to :func:`build`.

    Returns
    -------
    r0, r1 : numpy.ndarray
        Unperturbed and perturbed ``(n_steps, n_neurons)`` rasters.
    perturb_id : int or None
        The perturbed neuron (first to fire after ``t_stim`` in the unperturbed
        trial).
    """
    r0 = run_raster(seed, t_stim=t_stim, dt=dt, T=T, **kw)
    perturb_id = first_spike_after(r0, t_stim, dt)
    r1 = run_raster(seed, perturb_id=perturb_id, t_stim=t_stim, dt=dt, T=T, **kw)
    return r0, r1, perturb_id


def run_sensitivity(seed, *, t_stim=T_STIM, dt=DT, T=T_SIM, **kw):
    """Run both trials and return the divergence metrics plus the rate.

    Parameters
    ----------
    seed : int
        Shared trial seed.
    t_stim, dt, T : float, optional
        Perturbation time, resolution, horizon (ms).
    **kw
        Forwarded to :func:`build`.

    Returns
    -------
    dict
        The :func:`divergence` dictionary, plus ``perturb_id`` (the perturbed
        neuron) and ``rate`` (the unperturbed-trial population rate, Hz).
    """
    r0, r1, perturb_id = run_two_trials(seed, t_stim=t_stim, dt=dt, T=T, **kw)
    d = divergence(r0, r1, t_stim, dt)
    d['perturb_id'] = perturb_id
    d['rate'] = population_rate(r0, T)
    return d


def main():
    # The scaled config (full in-degree K, smaller N) keeps the balanced AI regime
    # but makes the chaotic transition frequent enough to display reliably; the
    # transition is probabilistic, so the seed scan shows a mix of stable and
    # chaotic realizations.
    cfg = dict(ne=300, ni=75, ke=KE, ki=KI, t_stim=120.0, dt=DT, T=250.0)
    print("sensitivity_to_perturbation (brainpy.state, iaf_psc_delta balanced E/I)")
    print(f"  N={cfg['ne'] + cfg['ni']}, K={cfg['ke']}/{cfg['ki']}, J={J} mV, g={G}, "
          f"t_stim={cfg['t_stim']} ms, dt={cfg['dt']} ms")
    chaotic = None
    for seed in (7, 8, 9, 11, 12, 13):
        d = run_sensitivity(seed, **cfg)
        flag = "CHAOTIC" if d['frac_decorr'] > 0.5 else "stable "
        print(f"  seed {seed:2d}: rate={d['rate']:5.1f} Hz  d_before={d['d_before']}  "
              f"frac_decorr={d['frac_decorr']:.3f}  [{flag}]  (perturbed id {d['perturb_id']})")
        if chaotic is None and d['frac_decorr'] > 0.5:
            chaotic = seed

    seed = chaotic if chaotic is not None else 7
    r0, r1, pid = run_two_trials(seed, **cfg)
    print(f"  plotting seed {seed} (perturbed neuron {pid})")
    try:
        import matplotlib.pyplot as plt
        t0, n0 = np.nonzero(r0)
        t1, n1 = np.nonzero(r1)
        plt.figure(figsize=(9, 4))
        plt.plot(t0 * cfg['dt'], n0, "r.", ms=2.0, label="unperturbed")
        plt.plot(t1 * cfg['dt'], n1, "k.", ms=1.0, label="perturbed")
        plt.axvline(cfg['t_stim'], color="b", lw=0.8, ls="--", label="t_stim")
        plt.xlabel("time (ms)"); plt.ylabel("neuron id")
        plt.xlim(0, cfg['T']); plt.legend(loc="upper right", markerscale=4)
        plt.title(f"Sensitivity to perturbation (seed {seed})")
        plt.tight_layout()
        plt.savefig("examples/nest_like/sensitivity_to_perturbation.png", dpi=100)
        print("  wrote examples/nest_like/sensitivity_to_perturbation.png")
    except ImportError:
        print("  (matplotlib not installed; skipping raster)")


if __name__ == "__main__":
    main()
