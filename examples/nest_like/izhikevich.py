# examples/nest_like/izhikevich.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Izhikevich neuron firing regimes — NEST-style port.

Reproduces the canonical firing patterns of the Izhikevich (2003) two-variable
model on the Simulator API. The model

.. math::

    \frac{dV}{dt} = 0.04 V^2 + 5 V + 140 - U + I, \qquad
    \frac{dU}{dt} = a (b V - U),

with the after-spike reset ``V \leftarrow c``, ``U \leftarrow U + d`` when
``V \geq V_th``, produces qualitatively different regimes for different
``(a, b, c, d)``. Four classic ones are shown here, each driven by a constant
``I_e = 10`` pA so the *intrinsic* dynamics set the pattern:

* **RS** — regular spiking (cortical excitatory), spike-frequency adaptation.
* **IB** — intrinsically bursting: an initial burst, then tonic firing.
* **CH** — chattering: repeated high-frequency bursts.
* **FS** — fast spiking (cortical inhibitory): fast, non-adapting.

Each regime records both the membrane potential ``V_m`` and the recovery
variable ``U_m`` through a multimeter, so the demo also exercises the ``U_m``
recordable. The integration uses NEST's default ``consistent_integration=True``
(forward Euler), so the traces match a live ``nest.izhikevich`` to machine
precision (see ``brainpy_state/_nest/_validation/izhikevich_test.py``).

Run:  PYTHONPATH=. python examples/nest_like/izhikevich.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import braintools
import brainunit as u

from brainpy_state import Simulator, izhikevich, multimeter, spike_recorder

# Canonical Izhikevich (2003) regime parameters: (a, b, c [mV], d [mV]).
REGIMES = {
    "RS": dict(a=0.02, b=0.2, c=-65.0, d=8.0),   # regular spiking
    "IB": dict(a=0.02, b=0.2, c=-55.0, d=4.0),   # intrinsically bursting
    "CH": dict(a=0.02, b=0.2, c=-50.0, d=2.0),   # chattering
    "FS": dict(a=0.10, b=0.2, c=-65.0, d=2.0),   # fast spiking
}

V_TH = 30.0        # spike cutoff [mV]
V0 = -70.0         # initial membrane potential [mV]
I_DRIVE = 10.0     # constant drive current [pA]
T_SIM = 300.0      # simulation time [ms]
DT = 0.1           # resolution [ms]


def _neuron_params(regime):
    p = REGIMES[regime]
    return dict(
        a=p["a"], b=p["b"], c=p["c"] * u.mV, d=p["d"] * u.mV,
        V_th=V_TH * u.mV, I_e=I_DRIVE * u.pA, consistent_integration=True,
        V_initializer=braintools.init.Constant(V0 * u.mV),
    )


def build(regime, simtime=T_SIM):
    """Build a single-neuron Simulator for one Izhikevich regime.

    Parameters
    ----------
    regime : {'RS', 'IB', 'CH', 'FS'}
        Which firing regime to instantiate.
    simtime : float, optional
        Simulation horizon in ms. Default :data:`T_SIM`.

    Returns
    -------
    sim : Simulator
    mm : NodeView
        Multimeter handle recording ``V_m`` and ``U_m``
        (``res.trace(mm, 'V_m')`` / ``res.trace(mm, 'U_m')``).
    sr : NodeView
        Spike recorder handle (``res.n_events(sr)``).
    simtime : float
    """
    sim = Simulator(dt=DT * u.ms)
    neuron = sim.create(izhikevich, 1, params=_neuron_params(regime))
    mm = sim.create(multimeter, record_from=["V_m", "U_m"], interval=DT * u.ms)
    sr = sim.create(spike_recorder)
    sim.connect(mm, neuron)              # the multimeter observes the neuron
    sim.connect(neuron, sr)
    return sim, mm, sr, simtime


def run_traces(regime, simtime=T_SIM):
    """Return ``(t_ms, v_mV, u_mV, n_spikes)`` for one regime."""
    sim, mm, sr, _t = build(regime, simtime)
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    v = np.asarray(u.get_mantissa(res.trace(mm, "V_m") / u.mV)).reshape(-1)
    um = np.asarray(u.get_mantissa(res.trace(mm, "U_m") / u.mV)).reshape(-1)
    return t, v, um, int(res.n_events(sr))


def main():
    print("Izhikevich firing regimes (brainpy.state, constant 10 pA drive)")
    results = {}
    for regime in REGIMES:
        t, v, um, n = run_traces(regime)
        results[regime] = (t, v, um)
        print(f"  {regime}: {n:3d} spikes, V_m peak {v.max():6.2f} mV, "
              f"U_m range [{um.min():.1f}, {um.max():.1f}] mV")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(REGIMES), 1, figsize=(8, 9), sharex=True)
        for ax, regime in zip(axes, REGIMES):
            t, v, _um = results[regime]
            ax.plot(t, v, color="k", lw=0.7)
            ax.set_ylabel("V_m (mV)")
            ax.set_title(f"{regime}", loc="left", fontsize=9)
        axes[-1].set_xlabel("time (ms)")
        fig.suptitle("Izhikevich regimes under constant drive")
        fig.tight_layout()
        fig.savefig("examples/nest_like/izhikevich.png", dpi=100)
        print("  wrote examples/nest_like/izhikevich.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
