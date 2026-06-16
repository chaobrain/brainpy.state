# examples/nest_like/artificial_synchrony.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Artificial synchrony in a grid-constrained IAF population — NEST-style port.

Port of NEST's ``artificial_synchrony.py``. A population of ``iaf_psc_alpha``
neurons under constant suprathreshold current ``I_e``, all-to-all coupled, with a
graded initial-V fan that seeds a controlled phase spread (``gamma``). As the
coupling ``strength`` grows the population synchronizes; on a discrete time grid
this synchrony is *artificially* amplified (Hansel et al. 1998; Morrison et al.
2007). The headline statistic is the Golomb–Rinzel synchrony measure

    Σ = var_t(mean_n V) / mean_n(var_t V)

over a late analysis window — Σ → 1 is full synchrony, Σ → 0 is asynchronous.

This port is **fixed-dt**, so it reproduces NEST's **grid** (``iaf_psc_alpha``)
branch. NEST's precise/off-grid (``iaf_psc_alpha_ps``) branch — whose whole point
is that off-grid spike times *reduce* the artificial synchrony — has no fixed-dt
analog and is not ported; see the parity test for the grid-vs-grid comparison.

Run:  PYTHONPATH=. python examples/nest_like/artificial_synchrony.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy_state import Simulator, iaf_psc_alpha, voltmeter, all_to_all

#: Population size.
N = 64
#: Simulation resolution (ms). Chosen so ``DELAY`` / ``T_REF`` land on the grid.
DT = 0.05
#: Total simulated time (ms).
SIMTIME = 1500.0
#: Start of the late synchrony-analysis window (ms); the first half is transient.
T_START = 750.0
#: Constant suprathreshold drive (pA).
I_E = 575.0
#: Membrane capacitance (pF).
C_M = 250.0
#: Membrane time constant (ms).
TAU_M = 10.0
#: Synaptic time constant (ms), ``3/2·ln 3`` as in the upstream.
TAU_SYN = 1.648
#: Reset / resting / threshold potentials (mV); ``E_L = V_reset = 0``.
V_RESET = 0.0
E_L = 0.0
V_TH = 20.0
#: Refractory period and synaptic delay (ms).
T_REF = 0.25
DELAY = 0.25
#: Initial-fan synchrony seed (0 → all at rest, 1 → full sub-threshold spread).
GRADED_GAMMA = 0.5
#: Default coupling strengths to sweep (pA).
STRENGTHS = (0.0, 1.0, 2.0, 3.0, 4.0)


def graded_v0(n, i_e=I_E, tau_m=TAU_M, c_m=C_M, e_l=E_L, v_reset=V_RESET,
              v_th=V_TH, gamma=GRADED_GAMMA):
    """Graded initial-V fan imposing a controlled phase spread.

    Reproduces the upstream seeding ``V0_i = R·I_e·(1 − exp(−γ·i/n·T/τ_m))`` with
    ``R = τ_m/C_m`` and ``T`` the analytic inter-spike interval. Neuron ``0``
    starts at rest, later neurons progressively closer to threshold.

    Parameters
    ----------
    n : int
        Population size.
    i_e, tau_m, c_m, e_l, v_reset, v_th, gamma : float, optional
        Neuron/seed parameters (defaults are the module constants).

    Returns
    -------
    numpy.ndarray
        ``(n,)`` initial membrane potentials in mV, within ``[v_reset, v_th]``.
    """
    R = tau_m / c_m
    T = tau_m * np.log((R * i_e + e_l - v_reset) / (R * i_e + e_l - v_th))
    i = np.arange(n)
    return R * i_e * (1.0 - np.exp(-gamma * i / n * T / tau_m))


def calc_synchrony(V, times_ms, t_start):
    """Golomb–Rinzel synchrony ``Σ = var_t(mean_n V) / mean_n(var_t V)``.

    Parameters
    ----------
    V : numpy.ndarray
        ``(n_steps, n_neurons)`` membrane-potential matrix (mV). NaNs (e.g. a
        NEST recorder's ragged steps) are ignored.
    times_ms : numpy.ndarray
        ``(n_steps,)`` sample times in ms.
    t_start : float
        Start of the analysis window in ms; samples before it are discarded.

    Returns
    -------
    float
        The synchrony measure Σ over the window (``nan`` if Δ is zero).
    """
    V = np.asarray(V)
    mask = np.asarray(times_ms) >= t_start
    Vw = V[mask]
    mean_V_t = np.nanmean(Vw, axis=1)          # population mean per time step
    Delta_N = np.nanvar(mean_V_t)              # variance of the population mean
    Delta = np.nanmean(np.nanvar(Vw, axis=0))  # mean over neurons of temporal var
    return float(Delta_N / Delta) if Delta != 0 else float('nan')


def build(strength, n=N, dt=DT, simtime=SIMTIME):
    """Build the coupled population with the graded initial-V fan.

    Parameters
    ----------
    strength : float
        All-to-all coupling weight (pA). ``0`` leaves the population uncoupled.
    n : int, optional
        Population size. Default :data:`N`.
    dt : float, optional
        Resolution in ms. Default :data:`DT`.
    simtime : float, optional
        Simulated horizon in ms. Default :data:`SIMTIME`.

    Returns
    -------
    sim : Simulator
        The configured simulator.
    vm : NodeView
        Voltmeter handle (``res.trace(vm, 'V_m')``).
    n : int
        Population size (echoed).
    simtime : float
        Simulated horizon in ms (echoed).
    """
    sim = Simulator(dt=dt * u.ms)
    npar = dict(C_m=C_M * u.pF, E_L=E_L * u.mV, I_e=I_E * u.pA, tau_m=TAU_M * u.ms,
                tau_syn_ex=TAU_SYN * u.ms, tau_syn_in=TAU_SYN * u.ms,
                V_reset=V_RESET * u.mV, V_th=V_TH * u.mV, t_ref=T_REF * u.ms,
                V_initializer=graded_v0(n) * u.mV)
    pop = sim.create(iaf_psc_alpha, n, params=npar)
    if strength != 0.0:
        sim.connect(pop, pop, rule=all_to_all, weight=strength * u.pA,
                    delay=DELAY * u.ms, allow_autapses=True)
    vm = sim.create(voltmeter, interval=dt * u.ms)
    sim.connect(vm, pop)
    return sim, vm, n, simtime


def run_synchrony(strength, n=N, dt=DT, simtime=SIMTIME, t_start=T_START):
    """Build, simulate, and return the synchrony measure Σ for one coupling.

    Parameters
    ----------
    strength : float
        Coupling weight (pA).
    n, dt, simtime, t_start : optional
        Network/analysis parameters (module-constant defaults).

    Returns
    -------
    float
        Σ over the late window.
    """
    sim, vm, n, simtime = build(strength, n=n, dt=dt, simtime=simtime)
    res = sim.simulate(simtime * u.ms)
    V = np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV))
    times = np.asarray(u.get_mantissa(res.times / u.ms))
    return calc_synchrony(V, times, t_start)


def sweep(strengths=STRENGTHS, **kw):
    """Return ``[Σ(strength) for strength in strengths]``."""
    return [run_synchrony(s, **kw) for s in strengths]


def main():
    print("artificial_synchrony (brainpy.state, iaf_psc_alpha, grid branch)")
    print(f"  N={N}, dt={DT} ms, simtime={SIMTIME} ms, window=[{T_START}, {SIMTIME}] ms")
    sigmas = sweep()
    for s, sig in zip(STRENGTHS, sigmas):
        print(f"  strength {s:.1f} pA  ->  Σ = {sig:.4f}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(STRENGTHS, sigmas, "ko-")
        plt.xlabel("coupling strength (pA)"); plt.ylabel("synchrony Σ")
        plt.title("Artificial synchrony vs coupling (grid)")
        plt.ylim(0, 1); plt.tight_layout()
        plt.savefig("examples/nest_like/artificial_synchrony.png", dpi=100)
        print("  wrote examples/nest_like/artificial_synchrony.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
