# examples/nest_like/if_curve.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""I-F curve of an adaptive exponential neuron — NEST-style port.

Port of NEST's ``if_curve.py``. A population of ``aeif_cond_exp`` neurons is
driven by a noisy current ``I(t) = I_mean + I_std * W(t)`` from a
``noise_generator`` (a *current*-injecting device, wired through the neuron's
current ring buffer), and the population firing rate is measured across a grid of
``(I_mean, I_std)`` values --- the neuron's transfer function. The network is
rebuilt per grid point (the rebuild-per-trial sweep pattern).

Run:  python examples/nest_like/if_curve.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import braintools
import brainunit as u

from brainpy.state import (Simulator, aeif_cond_exp, noise_generator,
                           spike_recorder)

# NEST if_curve neuron parameters (aeif_cond_exp).
MODEL_PARAMS = dict(
    a=4.0 * u.nS, b=80.8 * u.pA, V_th=-50.4 * u.mV, Delta_T=2.0 * u.mV,
    I_e=0.0 * u.pA, C_m=281.0 * u.pF, g_L=30.0 * u.nS, V_reset=-70.6 * u.mV,
    tau_w=144.0 * u.ms, t_ref=5.0 * u.ms, V_peak=-40.0 * u.mV, E_L=-70.6 * u.mV,
    E_ex=0.0 * u.mV, E_in=-70.0 * u.mV,
    V_initializer=braintools.init.Constant(-70.6 * u.mV),
)


def build(mean, std, n_neurons=100, seed=0, simtime=1000.0):
    """Build a population of ``aeif_cond_exp`` driven by a noisy current.

    Parameters
    ----------
    mean, std : float
        Mean and standard deviation of the injected white-noise current, in pA.
    n_neurons : int, optional
        Number of neurons (each receives an independent noise stream). Default ``100``.
    seed : int, optional
        Base PRNG seed for the noise generator. Default ``0``.
    simtime : float, optional
        Measurement-trial duration in ms (the noise window). Default ``1000.0``.

    Returns
    -------
    sim : Simulator
    sr : NodeView
        Spike-recorder handle (``res.rate(sr)``).
    neuron : NodeView
    simtime : float
    """
    sim = Simulator(dt=0.1 * u.ms)
    neuron = sim.create(aeif_cond_exp, n_neurons, params=MODEL_PARAMS)
    noise = sim.create(noise_generator, mean=mean * u.pA, std=std * u.pA,
                       start=0.0 * u.ms, stop=simtime * u.ms, seed=seed)
    sr = sim.create(spike_recorder)
    sim.connect(noise, neuron)           # current injection (independent per neuron)
    sim.connect(neuron, sr)
    return sim, sr, neuron, simtime


def output_rate(mean, std, n_neurons=100, seed=0, simtime=1000.0):
    """Population firing rate (spks/s) for noise current ``(mean, std)``."""
    sim, sr, _neuron, _t = build(mean, std, n_neurons=n_neurons, seed=seed,
                                 simtime=simtime)
    return sim.simulate(simtime * u.ms).rate(sr)


def compute_transfer(i_mean=(400.0, 900.0, 100.0), i_std=(0.0, 600.0, 150.0),
                     n_neurons=100, simtime=1000.0):
    """Measure the I-F surface over the ``(I_mean, I_std)`` grid."""
    i_range = np.arange(*i_mean)
    std_range = np.arange(*i_std)
    rate = np.zeros((i_range.size, std_range.size))
    for n, i in enumerate(i_range):
        for m, s in enumerate(std_range):
            rate[n, m] = output_rate(i, s, n_neurons=n_neurons, simtime=simtime)
    return i_range, std_range, rate


def main():
    print("I-F curve (brainpy.state, aeif_cond_exp + noise_generator)")
    i_range, std_range, rate = compute_transfer()
    for n, i in enumerate(i_range):
        row = "  ".join(f"{rate[n, m]:6.1f}" for m in range(std_range.size))
        print(f"  I_mean={i:6.1f} pA | rates: {row}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        extent = [std_range[0], std_range[-1], i_range[0], i_range[-1]]
        plt.imshow(rate, origin="lower", aspect="auto", extent=extent)
        plt.colorbar(label="rate (spks/s)")
        plt.xlabel("I_std (pA)"); plt.ylabel("I_mean (pA)")
        plt.title("aeif_cond_exp I-F surface")
        plt.tight_layout()
        plt.savefig("examples/nest_like/if_curve.png", dpi=100)
        print("  wrote examples/nest_like/if_curve.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
