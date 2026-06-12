# examples/nest/one_neuron_with_noise.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""One neuron driven by Poisson noise — NEST-style port.

Port of NEST's ``one_neuron_with_noise.py``. A single ``iaf_psc_alpha`` is driven
by a 2-channel ``poisson_generator`` (an excitatory channel at 80 kHz and an
inhibitory channel at 15 kHz) with signed per-channel synaptic weights
``[1.2, -1.0] pA``, and its membrane potential is observed by a ``voltmeter``.

The 2-channel generator maps to a 2-segment NodeView: ``connect(noise, neuron,
weight=[1.2, -1.0] * u.pA)`` applies one weight per channel, so the neuron
integrates ``1.2 * train_ex - 1.0 * train_in`` (positive = excitatory, negative
= inhibitory). A passive ``spike_recorder`` is also attached so the firing rate
can be read back (it does not affect the dynamics).

Run:  python examples/nest/one_neuron_with_noise.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import saiunit as u

from brainpy_state import (Simulator, iaf_psc_alpha, poisson_generator,
                           voltmeter, spike_recorder)


def build(seed=0, simtime=1000.0, rate_ex=80000.0, rate_in=15000.0,
          w_ex=1.2, w_in=-1.0, delay=1.0):
    """Build the noise-driven one-neuron network.

    Parameters
    ----------
    seed : int, optional
        Base PRNG seed for the Poisson channels. Default ``0``.
    simtime : float, optional
        Intended simulation horizon in ms (returned for the caller).
    rate_ex, rate_in : float, optional
        Excitatory / inhibitory Poisson rates in Hz. Defaults ``80000`` / ``15000``.
    w_ex, w_in : float, optional
        Per-channel synaptic weights in pA (signed). Defaults ``1.2`` / ``-1.0``.
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
    simtime : float
    """
    sim = Simulator(dt=0.1 * u.ms)
    neuron = sim.create(iaf_psc_alpha, 1)
    noise = sim.create(poisson_generator, 2, rate=[rate_ex, rate_in] * u.Hz,
                       rng_seed=seed)
    vm = sim.create(voltmeter)
    sr = sim.create(spike_recorder)
    sim.connect(noise, neuron, weight=[w_ex, w_in] * u.pA, delay=delay * u.ms)
    sim.connect(vm, neuron)              # reversed: the voltmeter observes the neuron
    sim.connect(neuron, sr)
    return sim, vm, sr, neuron, simtime


def main():
    sim, vm, sr, _neuron, simtime = build()
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    v = np.asarray(u.get_mantissa(res.trace(vm, "V_m") / u.mV)).reshape(-1)
    print("One neuron with noise (brainpy.state)")
    print(f"  Poisson drive 80 kHz (w=+1.2) / 15 kHz (w=-1.0)")
    print(f"  V_m: mean {v.mean():.2f} mV, std {v.std():.2f} mV")
    print(f"  Firing rate: {res.rate(sr):.2f} spks/s")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(t, v, color="k", lw=0.6)
        plt.xlabel("time (ms)"); plt.ylabel("V_m (mV)")
        plt.title("one neuron with noise — membrane potential")
        plt.tight_layout()
        plt.savefig("examples/nest/one_neuron_with_noise.png", dpi=100)
        print("  wrote examples/nest/one_neuron_with_noise.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
