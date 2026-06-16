# examples/nest_like/one_neuron.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""One neuron driven by a constant current — NEST-style port.

Port of NEST's ``one_neuron.py`` onto brainpy.state's explicit Simulator API.
A single ``iaf_psc_alpha`` is driven by a constant external current ``I_e`` and
its membrane potential is observed by a ``voltmeter``. As in NEST, the voltmeter
is connected in the *reversed* direction --- ``connect(voltmeter, neuron)`` ---
because it observes the neuron rather than receiving events from it.

With ``I_e = 376 pA`` the steady state (V_inf = E_L + I_e * tau_m / C_m ~
-54.96 mV) sits just above threshold (V_th = -55 mV), so the membrane charges,
fires, and repeats.

Run:  python examples/nest_like/one_neuron.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import Simulator, iaf_psc_alpha, voltmeter


def build(I_e=376.0, simtime=1000.0):
    """Build the one-neuron network.

    Parameters
    ----------
    I_e : float, optional
        Constant external current in pA. Default ``376.0`` (NEST's value).
    simtime : float, optional
        Intended simulation horizon in ms (returned for the caller).

    Returns
    -------
    sim : Simulator
        The configured simulator.
    vm : NodeView
        The voltmeter handle (read via ``res.trace(vm, 'V_m')``).
    neuron : NodeView
        The neuron population handle.
    simtime : float
        The simulation horizon in ms.
    """
    sim = Simulator(dt=0.1 * u.ms)
    neuron = sim.create(iaf_psc_alpha, 1, I_e=I_e * u.pA)
    vm = sim.create(voltmeter)
    sim.connect(vm, neuron)              # reversed: the voltmeter observes the neuron
    return sim, vm, neuron, simtime


def main():
    sim, vm, _neuron, simtime = build()
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    v = np.asarray(u.get_mantissa(res.trace(vm, "V_m") / u.mV)).reshape(-1)
    print("One neuron (brainpy.state, I_e = 376 pA)")
    print(f"  V_m: start {v[0]:.2f} mV, max {v.max():.2f} mV, "
          f"{int((np.diff(v) < -10.0).sum())} spikes in {simtime:.0f} ms")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(t, v, color="k")
        plt.xlabel("time (ms)"); plt.ylabel("V_m (mV)")
        plt.title("one neuron — membrane potential")
        plt.tight_layout()
        plt.savefig("examples/nest_like/one_neuron.png", dpi=100)
        print("  wrote examples/nest_like/one_neuron.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
