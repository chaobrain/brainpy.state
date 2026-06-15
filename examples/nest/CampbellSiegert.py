# examples/nest/CampbellSiegert.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Campbell & Siegert approximation — NEST-style port (analytic cross-check).

Ports NEST's ``pynest/examples/CampbellSiegert.py``. For an ``iaf_psc_alpha``
neuron driven by one or more Poisson sources (each with a rate and a PSP
amplitude), two classical results predict its response analytically:

* **Campbell's theorem** gives the mean and variance of the *free* (sub-threshold,
  non-resetting) membrane potential as a sum of per-source contributions.
* **Siegert's approximation** gives the stationary firing rate of the
  integrate-and-fire neuron from that mean ``mu`` and standard deviation ``sigma``.

The demo computes ``mu``, ``sigma^2`` and the Siegert rate ``r`` in closed form
(the PSP→PSC ``fudge`` factor is found with ``scipy.optimize.fmin``; the rate
integral uses ``scipy.special.erf``), then simulates a small population of
``iaf_psc_alpha`` neurons receiving the same Poisson drive and compares the
empirical mean / variance of a free neuron's ``V_m`` and the population firing
rate against the analytic predictions.

**Validation is an analytic cross-check, not a live-NEST trace parity.** The
Poisson drive is a PRNG stream that diverges between NEST and JAX, so a per-sample
``V_m`` comparison against NEST is meaningless. Instead the *theory itself* is the
ground truth: ``brainpy_state/_nest/_validation/CampbellSiegert_test.py`` asserts
that the Simulator's empirical mean/variance/rate match the Campbell/Siegert
formulae within statistical tolerance (and, when NEST is present, that live NEST
matches the same formulae — both agree with theory without agreeing sample by
sample).

Run:  PYTHONPATH=. python examples/nest/CampbellSiegert.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import braintools
import brainunit as u
from scipy.optimize import fmin
from scipy.special import erf

from brainpy_state import (Simulator, iaf_psc_alpha, poisson_generator,
                           voltmeter, spike_recorder)

# --- NEST CampbellSiegert parameters (single Poisson source variant) ---------
WEIGHTS = [0.1]        # (mV) PSP amplitudes
RATES = [10000.0]      # (1/s) rates of the Poisson sources
# A two-source variant (half the rate each) gives the same result:
#   WEIGHTS = [0.1, 0.1]; RATES = [5000.0, 5000.0]

C_M = 250.0            # (pF) capacitance
E_L = -70.0            # (mV) resting / reset potential
V_TH = -55.0           # (mV) firing threshold
V_RESET = -70.0        # (mV) reset potential (= E_L)
T_REF = 2.0            # (ms) refractory period
TAU_M = 10.0           # (ms) membrane time constant
TAU_SYN_EX = 0.5       # (ms) excitatory synaptic time constant
TAU_SYN_IN = 2.0       # (ms) inhibitory synaptic time constant

SIMTIME = 20000.0      # (ms) duration
N_NEURONS = 10         # number of threshold neurons (for the rate estimate)
N_SKIP = 500           # initial V_m steps discarded (transient)
DT = 0.1               # (ms) resolution

# SI scale factors (the analytic block works in SI, like the NEST script).
_pF, _ms, _pA, _mV = 1e-12, 1e-3, 1e-12, 1e-3


def analytic(weights=WEIGHTS, rates=RATES):
    r"""Campbell mean/variance and Siegert rate for the configured drive.

    Returns
    -------
    J : list of float
        Per-source PSC amplitudes (pA) — the synaptic weights to set in the model.
    mu : float
        Mean free membrane potential (V, SI).
    sigma2 : float
        Variance of the free membrane potential (V^2, SI).
    r : float
        Siegert stationary firing rate (Hz).
    """
    assert len(weights) == len(rates)
    mu, sigma2, J = 0.0, 0.0, []
    for rate, weight in zip(rates, weights):
        tau_syn = TAU_SYN_EX if weight > 0 else TAU_SYN_IN

        def psp(x):
            # Single alpha PSP shape; its extremum sets the PSP/PSC fudge factor.
            return -(
                (C_M * _pF) / (tau_syn * _ms) * (1 / (C_M * _pF))
                * (np.exp(1) / (tau_syn * _ms))
                * (((-x * np.exp(-x / (tau_syn * _ms))) / (1 / (tau_syn * _ms) - 1 / (TAU_M * _ms)))
                   + (np.exp(-x / (TAU_M * _ms)) - np.exp(-x / (tau_syn * _ms)))
                   / ((1 / (tau_syn * _ms) - 1 / (TAU_M * _ms)) ** 2))
            )

        min_result = fmin(psp, [0], full_output=1, disp=0)
        fudge = -1.0 / min_result[1]
        J.append(C_M * weight / tau_syn * fudge)

        # Campbell's theorem: mean and variance add over independent sources.
        mu += rate * (J[-1] * _pA) * (tau_syn * _ms) * np.exp(1) * (TAU_M * _ms) / (C_M * _pF)
        sigma2 += (
            rate * (2 * TAU_M * _ms + tau_syn * _ms)
            * (J[-1] * _pA * tau_syn * _ms * np.exp(1) * TAU_M * _ms
               / (2 * (C_M * _pF) * (TAU_M * _ms + tau_syn * _ms))) ** 2
        )
    mu += E_L * _mV
    sigma = np.sqrt(sigma2)

    # Siegert's stationary-rate approximation (trapezoidal integral).
    num_iterations = 100
    upper = (V_TH * _mV - mu) / (sigma * np.sqrt(2))
    lower = (E_L * _mV - mu) / (sigma * np.sqrt(2))
    interval = (upper - lower) / num_iterations
    tmpsum = 0.0
    for cu in range(num_iterations + 1):
        uu = lower + cu * interval
        tmpsum += interval * np.sqrt(np.pi) * np.exp(uu ** 2) * (1 + erf(uu))
    r = 1.0 / (T_REF * _ms + TAU_M * _ms * tmpsum)
    return J, mu, sigma2, r


def _neuron_params(V_th=V_TH):
    return dict(
        C_m=C_M * u.pF, tau_m=TAU_M * u.ms,
        tau_syn_ex=TAU_SYN_EX * u.ms, tau_syn_in=TAU_SYN_IN * u.ms,
        E_L=E_L * u.mV, V_reset=V_RESET * u.mV, t_ref=T_REF * u.ms,
        V_th=V_th * u.mV, V_initializer=braintools.init.Constant(E_L * u.mV),
    )


def build(J, simtime=SIMTIME, seed=0):
    """Build the population + free neuron + Poisson drive.

    Parameters
    ----------
    J : list of float
        Per-source PSC weights (pA) from :func:`analytic`.
    simtime : float, optional
        Simulation horizon in ms. Default :data:`SIMTIME`.
    seed : int, optional
        Base PRNG seed for the Poisson sources. Default ``0``.

    Returns
    -------
    sim : Simulator
    vm : NodeView
        Voltmeter on the free neuron (``res.trace(vm, 'V_m')``).
    sr : NodeView
        Spike recorder on the threshold population (``res.n_events(sr)``).
    simtime : float
    """
    sim = Simulator(dt=DT * u.ms)
    pop = sim.create(iaf_psc_alpha, N_NEURONS, params=_neuron_params())
    free = sim.create(iaf_psc_alpha, 1, params=_neuron_params(V_th=1e12))
    pg = sim.create(poisson_generator, len(RATES),
                    rate=np.asarray(RATES) * u.Hz, rng_seed=seed)
    vm = sim.create(voltmeter, interval=DT * u.ms)
    sr = sim.create(spike_recorder)
    w = np.asarray(J) * u.pA
    sim.connect(pg, pop, weight=w, delay=DT * u.ms)
    sim.connect(pg, free, weight=w, delay=DT * u.ms)
    sim.connect(vm, free)
    sim.connect(pop, sr)
    return sim, vm, sr, simtime


def run_analysis(simtime=SIMTIME, seed=0):
    """Run the simulation and return analytic vs empirical statistics.

    Returns
    -------
    dict
        Keys ``mu_mV``, ``var_mV2``, ``rate_hz`` (analytic) and ``mean_act``,
        ``var_act``, ``rate_act`` (empirical, from the Simulator).
    """
    J, mu, sigma2, r = analytic()
    sim, vm, sr, _t = build(J, simtime, seed)
    res = sim.simulate(simtime * u.ms)
    # The voltmeter observes the single free neuron; drop the initial transient.
    v = np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV)).reshape(-1)[N_SKIP:]
    rate_act = int(res.n_events(sr)) / (N_NEURONS * simtime * _ms)
    return dict(
        mu_mV=mu * 1e3, var_mV2=sigma2 * 1e6, rate_hz=r,
        mean_act=float(np.mean(v)), var_act=float(np.var(v)), rate_act=rate_act,
    )


def main():
    print("Campbell & Siegert approximation (brainpy.state)")
    res = run_analysis()
    print(f"  mean membrane potential (actual / calculated): "
          f"{res['mean_act']:.4f} / {res['mu_mV']:.4f} mV")
    print(f"  variance               (actual / calculated): "
          f"{res['var_act']:.4f} / {res['var_mV2']:.4f} mV^2")
    print(f"  firing rate            (actual / calculated): "
          f"{res['rate_act']:.4f} / {res['rate_hz']:.4f} Hz")


if __name__ == "__main__":
    main()
