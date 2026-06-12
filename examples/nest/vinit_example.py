# examples/nest/vinit_example.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Initial membrane voltage sweep — NEST-style port.

Port of NEST's ``vinit_example.py``. The ``iaf_cond_exp_sfa_rr`` neuron is run
with no input from several initial membrane voltages; each run relaxes passively
toward the resting potential E_L. The network is rebuilt per initial voltage (the
rebuild-per-trial sweep pattern) and ``V_m(t)`` is observed by a ``voltmeter``.

Run:  python examples/nest/vinit_example.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import braintools
import saiunit as u

from brainpy_state import Simulator, iaf_cond_exp_sfa_rr, voltmeter


def build(vinit=-70.0, simtime=75.0):
    """Build a single ``iaf_cond_exp_sfa_rr`` initialized at ``vinit``.

    Parameters
    ----------
    vinit : float, optional
        Initial membrane potential in mV. Default ``-70.0``.
    simtime : float, optional
        Intended simulation horizon in ms (returned for the caller).

    Returns
    -------
    sim : Simulator
    vm : NodeView
        Voltmeter handle (``res.trace(vm, 'V_m')``).
    neuron : NodeView
    simtime : float
    """
    sim = Simulator(dt=0.1 * u.ms)
    neuron = sim.create(iaf_cond_exp_sfa_rr, 1,
                        V_initializer=braintools.init.Constant(vinit * u.mV))
    vm = sim.create(voltmeter)
    sim.connect(vm, neuron)              # reversed: the voltmeter observes the neuron
    return sim, vm, neuron, simtime


def main():
    simtime = 75.0
    try:
        import matplotlib.pyplot as plt
        have_plt = True
        plt.figure(figsize=(8, 4))
    except ImportError:
        have_plt = False

    print("Initial membrane voltage sweep (brainpy.state, iaf_cond_exp_sfa_rr)")
    for vinit in np.arange(-100, -50, 10, float):
        sim, vm, _neuron, _t = build(vinit=vinit, simtime=simtime)
        res = sim.simulate(simtime * u.ms)
        t = np.asarray(u.get_mantissa(res.times / u.ms))
        v = np.asarray(u.get_mantissa(res.trace(vm, "V_m") / u.mV)).reshape(-1)
        print(f"  V_m(0)={vinit:7.2f} mV -> V_m({simtime:.0f} ms)={v[-1]:7.2f} mV")
        if have_plt:
            plt.plot(t, v, label=f"initial V_m = {vinit:.2f} mV")

    if have_plt:
        plt.legend(loc=4)
        plt.xlabel("time (ms)"); plt.ylabel("V_m (mV)")
        plt.title("vinit_example — relaxation from initial V_m")
        plt.tight_layout()
        plt.savefig("examples/nest/vinit_example.png", dpi=100)
        print("  wrote examples/nest/vinit_example.png")
    else:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
