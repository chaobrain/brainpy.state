# examples/nest_like/multimeter_file.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Multimeter recording of several analog variables — NEST-style port.

Port of NEST's ``multimeter_file.py``, the multi-recordable recording demo. The
upstream writes the data to an ``ascii`` file backend; brainpy.state has no
file-backed recorders (``devices-gap.md`` P2), so this is the **in-memory
equivalent** — the multimeter taps the neuron's State each step and the trace is
read back with ``res.trace(mm, name)``.

The upstream records ``V_m``/``g_ex``/``g_in`` from a conductance-based
``iaf_cond_alpha``. Driving conductance synapses from a spike source needs a
``w_ex``/``w_in`` labelled-delta routing seam that the explicit ``Simulator``
does not yet provide (a documented follow-up; see ``develop/NEST_PARITY_LEDGER.md`` Lessons). This
port keeps the same demo *structure* — a multimeter recording several analog
variables from a spike-driven neuron — on the current-based ``iaf_psc_exp``,
whose recordables are ``V_m``, ``I_syn_ex`` and ``I_syn_in``. Two
``spike_generator``s deliver excitatory (positive weight → ``I_syn_ex``) and
inhibitory (negative weight → ``I_syn_in``) input; ``iaf_psc_exp`` splits a
delta event by sign, exactly as NEST routes a signed weight to its ex/in port.

Weights are small enough that the neuron stays sub-threshold over the 100 ms
window, so the three traces are smooth analytic-propagator curves that match
NEST to recorder precision (``CAT_B_ALIGNED``).

Run:  python examples/nest_like/multimeter_file.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import Simulator, iaf_psc_exp, multimeter, spike_generator

#: Recordables tapped from ``iaf_psc_exp`` (NEST vocabulary).
RECORD_FROM = ('V_m', 'I_syn_ex', 'I_syn_in')
#: Excitatory / inhibitory spike times (ms) — the upstream's two trains.
SPIKE_TIMES_EX = (10.0, 20.0, 50.0)
SPIKE_TIMES_IN = (15.0, 25.0, 55.0)
#: Signed synaptic weights (pA) and the homogeneous delivery delay (ms).
W_EX = 80.0
W_IN = -40.0
DELAY = 1.0


def build(dt=0.1, simtime=100.0):
    """Build the multi-recordable recording network.

    Parameters
    ----------
    dt : float, optional
        Simulation resolution in ms. Default ``0.1``. The multimeter's
        ``interval`` is set to ``dt`` so its sample grid matches the run.
    simtime : float, optional
        Intended simulation horizon in ms (returned for the caller). Default
        ``100.0``.

    Returns
    -------
    sim : Simulator
        The configured simulator.
    mm : NodeView
        The multimeter handle (read via ``res.trace(mm, name)`` for each
        ``name`` in :data:`RECORD_FROM`).
    neuron : NodeView
        The neuron population handle.
    simtime : float
        The simulation horizon in ms.
    """
    sim = Simulator(dt=dt * u.ms)
    neuron = sim.create(iaf_psc_exp, 1)
    mm = sim.create(multimeter, record_from=list(RECORD_FROM), interval=dt * u.ms)
    s_ex = sim.create(spike_generator, spike_times=np.asarray(SPIKE_TIMES_EX) * u.ms)
    s_in = sim.create(spike_generator, spike_times=np.asarray(SPIKE_TIMES_IN) * u.ms)
    sim.connect(s_ex, neuron, weight=W_EX * u.pA, delay=DELAY * u.ms)
    sim.connect(s_in, neuron, weight=W_IN * u.pA, delay=DELAY * u.ms)
    sim.connect(mm, neuron)              # reversed: the multimeter observes the neuron
    return sim, mm, neuron, simtime


def main():
    sim, mm, _neuron, simtime = build()
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    v = np.asarray(u.get_mantissa(res.trace(mm, "V_m") / u.mV)).reshape(-1)
    iex = np.asarray(u.get_mantissa(res.trace(mm, "I_syn_ex") / u.pA)).reshape(-1)
    iin = np.asarray(u.get_mantissa(res.trace(mm, "I_syn_in") / u.pA)).reshape(-1)
    print("multimeter_file (brainpy.state, iaf_psc_exp, in-memory)")
    print(f"  V_m: start {v[0]:.2f} mV, min {v.min():.2f} mV, max {v.max():.2f} mV")
    print(f"  I_syn_ex: max {iex.max():.2f} pA ; I_syn_in: min {iin.min():.2f} pA")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.plot(t, v, color="k")
        plt.ylabel("membrane potential (mV)")
        plt.subplot(212)
        plt.plot(t, iex, label="I_syn_ex")
        plt.plot(t, iin, label="I_syn_in")
        plt.xlabel("time (ms)"); plt.ylabel("synaptic current (pA)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("examples/nest_like/multimeter_file.png", dpi=100)
        print("  wrote examples/nest_like/multimeter_file.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
