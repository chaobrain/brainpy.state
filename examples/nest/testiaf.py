# examples/nest/testiaf.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""IAF neuron with a constant current, swept over resolution — NEST-style port.

Port of NEST's ``testiaf.py``. A constant current ``I_e = 376 pA`` is injected
into an ``iaf_psc_alpha``; the membrane charges, a spike is emitted, the neuron
becomes refractory, and it recovers. The network is rebuilt and re-simulated at
three resolutions ``dt in {0.1, 0.5, 1.0}`` ms (the rebuild-per-trial sweep
pattern), and ``V_m`` and the spike count are recorded at each.

Run:  python examples/nest/testiaf.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import saiunit as u

from brainpy_state import Simulator, iaf_psc_alpha, voltmeter, spike_recorder


def build(dt=0.1, I_e=376.0, simtime=1000.0):
    """Build the single-neuron network at resolution ``dt``.

    Parameters
    ----------
    dt : float, optional
        Simulation resolution in ms. Default ``0.1``.
    I_e : float, optional
        Constant external current in pA. Default ``376.0``.
    simtime : float, optional
        Intended simulation horizon in ms (returned for the caller).

    Returns
    -------
    sim : Simulator
    vm : NodeView
        Voltmeter handle (``res.trace(vm, 'V_m')``).
    sr : NodeView
        Spike-recorder handle (``res.n_events(sr)``).
    neuron : NodeView
    simtime : float
    """
    sim = Simulator(dt=dt * u.ms)
    neuron = sim.create(iaf_psc_alpha, 1, I_e=I_e * u.pA)
    vm = sim.create(voltmeter)
    sr = sim.create(spike_recorder)
    sim.connect(vm, neuron)              # reversed: the voltmeter observes the neuron
    sim.connect(neuron, sr)
    return sim, vm, sr, neuron, simtime


def main():
    simtime = 1000.0
    try:
        import matplotlib.pyplot as plt
        have_plt = True
        plt.figure(figsize=(8, 4))
    except ImportError:
        have_plt = False

    for dt in (0.1, 0.5, 1.0):
        print(f"Running simulation with dt={dt:.2f}")
        sim, vm, sr, _neuron, _t = build(dt=dt, simtime=simtime)
        res = sim.simulate(simtime * u.ms)
        t = np.asarray(u.get_mantissa(res.times / u.ms))
        v = np.asarray(u.get_mantissa(res.trace(vm, "V_m") / u.mV)).reshape(-1)
        print(f"  Number of spikes: {res.n_events(sr)}")
        if have_plt:
            plt.plot(t, v, label=f"dt={dt:.2f}")

    if have_plt:
        plt.legend(loc=3)
        plt.xlabel("time (ms)"); plt.ylabel("V_m (mV)")
        plt.title("testiaf — membrane potential vs resolution")
        plt.tight_layout()
        plt.savefig("examples/nest/testiaf.png", dpi=100)
        print("  wrote examples/nest/testiaf.png")
    else:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
