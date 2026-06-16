# examples/nest_like/balancedneuron.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Balanced neuron — NEST-style port.

Port of NEST's ``balancedneuron.py``. A single ``iaf_psc_alpha`` is driven by an
excitatory and an inhibitory Poisson population (a 2-channel ``poisson_generator``
with signed per-channel weights ``[epsc, ipsc]``). The goal is to find the
inhibitory firing rate that makes the target neuron fire at the same rate as the
excitatory population (``r_ex``). The root is found with SciPy's ``bisect``,
rebuilding and re-simulating the network at each trial (the rebuild-per-trial
sweep pattern + per-generator weight vectors).

Run:  python examples/nest_like/balancedneuron.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u
from scipy.optimize import bisect

from brainpy_state import (Simulator, iaf_psc_alpha, poisson_generator,
                           voltmeter, spike_recorder)

# NEST balancedneuron parameters.
N_EX = 16000      # size of the excitatory population
N_IN = 4000       # size of the inhibitory population
R_EX = 5.0        # mean rate of the excitatory population (Hz)
EPSC = 45.0       # peak amplitude of excitatory synaptic currents (pA)
IPSC = -45.0      # peak amplitude of inhibitory synaptic currents (pA)
DELAY = 1.0       # synaptic delay (ms)


def build(r_in, seed=0, n_ex=N_EX, n_in=N_IN, r_ex=R_EX,
          epsc=EPSC, ipsc=IPSC, delay=DELAY):
    """Build the balanced-neuron network for an inhibitory rate ``r_in``.

    Parameters
    ----------
    r_in : float
        Inhibitory population rate in Hz (the bisection variable). The inhibitory
        Poisson channel is driven at ``n_in * r_in`` Hz.
    seed : int, optional
        Base PRNG seed for the Poisson channels. Default ``0``.
    n_ex, n_in : int, optional
        Excitatory / inhibitory population sizes. Defaults ``16000`` / ``4000``.
    r_ex : float, optional
        Excitatory population rate in Hz. Default ``5.0``.
    epsc, ipsc : float, optional
        Per-channel synaptic weights in pA. Defaults ``45`` / ``-45``.
    delay : float, optional
        Synaptic delay in ms. Default ``1.0``.

    Returns
    -------
    sim : Simulator
    vm : NodeView
        Voltmeter handle (``res.trace(vm, 'V_m')``).
    sr : NodeView
        Spike-recorder handle (``res.rate(sr)``).
    neuron : NodeView
    """
    sim = Simulator(dt=0.1 * u.ms)
    neuron = sim.create(iaf_psc_alpha, 1)
    noise = sim.create(poisson_generator, 2,
                       rate=[n_ex * r_ex, n_in * abs(r_in)] * u.Hz, rng_seed=seed)
    vm = sim.create(voltmeter)
    sr = sim.create(spike_recorder)
    sim.connect(noise, neuron, weight=[epsc, ipsc] * u.pA, delay=delay * u.ms)
    sim.connect(vm, neuron)
    sim.connect(neuron, sr)
    return sim, vm, sr, neuron


def output_rate(r_in, simtime=25000.0, seed=0, **kw):
    """Firing rate (spks/s) of the target neuron for inhibitory rate ``r_in``."""
    sim, _vm, sr, _neuron = build(r_in, seed=seed, **kw)
    return sim.simulate(simtime * u.ms).rate(sr)


def find_inhibitory_rate(simtime=25000.0, lower=15.0, upper=25.0, prec=0.01,
                         seed=0, r_ex=R_EX):
    """Bisect the inhibitory rate so the target neuron fires at ``r_ex``."""
    def objective(x):
        out = output_rate(x, simtime=simtime, seed=seed, r_ex=r_ex)
        print(f"  inhibitory rate {x:5.2f} -> neuron rate {out:6.2f} spks/s "
              f"(goal {r_ex:.2f})")
        return out - r_ex
    return bisect(objective, lower, upper, xtol=prec)


def main():
    # 25 s as in NEST gives a stable estimate but is slow under JIT; trim if needed.
    simtime = 25000.0
    print("Balanced neuron (brainpy.state): bisecting the inhibitory rate ...")
    in_rate = find_inhibitory_rate(simtime=simtime)
    print(f"Optimal inhibitory population rate: {in_rate:.2f} spks/s")


if __name__ == "__main__":
    main()
