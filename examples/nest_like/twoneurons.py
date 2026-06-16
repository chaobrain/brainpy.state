# examples/nest_like/twoneurons.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Two connected neurons — NEST-style port.

Port of NEST's ``twoneurons.py``. A presynaptic ``iaf_psc_alpha`` (``neuron_1``)
is driven by a constant current ``I_e = 376 pA`` and connected to a postsynaptic
``iaf_psc_alpha`` (``neuron_2``) through a static synapse (``weight = 20 pA``,
``delay = 1 ms``). Each neuron's membrane potential is observed by its own
``voltmeter`` --- a brainpy.state analog tap records a single population, so the
NEST demo's single voltmeter-to-both-neurons connection becomes one voltmeter per
neuron (the reversed ``connect(voltmeter, neuron)`` direction is the same).

Run:  python examples/nest_like/twoneurons.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import Simulator, iaf_psc_alpha, voltmeter


def build(I_e=376.0, weight=20.0, delay=1.0, simtime=1000.0):
    """Build the two-neuron network.

    Parameters
    ----------
    I_e : float, optional
        Constant external current on ``neuron_1`` in pA. Default ``376.0``.
    weight : float, optional
        Static synaptic weight ``neuron_1 -> neuron_2`` in pA. Default ``20.0``.
    delay : float, optional
        Synaptic delay in ms. Default ``1.0``.
    simtime : float, optional
        Intended simulation horizon in ms (returned for the caller).

    Returns
    -------
    sim : Simulator
    vm1, vm2 : NodeView
        Voltmeter handles for ``neuron_1`` / ``neuron_2``.
    neuron_1, neuron_2 : NodeView
    simtime : float
    """
    sim = Simulator(dt=0.1 * u.ms)
    neuron_1 = sim.create(iaf_psc_alpha, 1, I_e=I_e * u.pA)
    neuron_2 = sim.create(iaf_psc_alpha, 1)
    vm1 = sim.create(voltmeter)
    vm2 = sim.create(voltmeter)
    sim.connect(neuron_1, neuron_2, weight=weight * u.pA, delay=delay * u.ms)
    sim.connect(vm1, neuron_1)           # reversed: each voltmeter observes its neuron
    sim.connect(vm2, neuron_2)
    return sim, vm1, vm2, neuron_1, neuron_2, simtime


def main():
    sim, vm1, vm2, _n1, _n2, simtime = build()
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    v1 = np.asarray(u.get_mantissa(res.trace(vm1, "V_m") / u.mV)).reshape(-1)
    v2 = np.asarray(u.get_mantissa(res.trace(vm2, "V_m") / u.mV)).reshape(-1)
    print("Two neurons (brainpy.state)")
    print(f"  neuron_1 (I_e=376 pA): {int((np.diff(v1) < -10.0).sum())} spikes")
    print(f"  neuron_2 (w=20 pA PSPs): V_m in [{v2.min():.2f}, {v2.max():.2f}] mV")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(t, v1, label="neuron_1", color="k")
        plt.plot(t, v2, label="neuron_2", color="tab:red")
        plt.xlabel("time (ms)"); plt.ylabel("V_m (mV)")
        plt.title("two neurons — membrane potentials")
        plt.legend(loc=3)
        plt.tight_layout()
        plt.savefig("examples/nest_like/twoneurons.png", dpi=100)
        print("  wrote examples/nest_like/twoneurons.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
