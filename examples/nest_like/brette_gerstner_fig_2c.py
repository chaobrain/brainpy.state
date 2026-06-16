# examples/nest_like/brette_gerstner_fig_2c.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Brette & Gerstner (2005) Fig 2C — AdEx spike-frequency adaptation (NEST port).

Port of NEST's ``brette_gerstner_fig_2c.py``. A single adaptive exponential
integrate-and-fire neuron (:class:`~brainpy.state.aeif_cond_alpha`) is driven by
two rectangular current pulses and reproduces Figure 2C of Brette & Gerstner
[1]_: a 500 pA pulse over ``[0, 200) ms`` charges the membrane sub-threshold,
then an 800 pA pulse over ``[500, 1000) ms`` drives a spike train whose
inter-spike intervals *lengthen* as the adaptation current ``w`` builds up —
the hallmark of spike-frequency adaptation.

The adaptation parameters come from the paper: ``a = 4 nS`` (sub-threshold
coupling) and ``b = 80.5 pA`` (spike-triggered increment). Brette & Gerstner
quote ``b`` in nA; to be consistent with the other quantities it must be
expressed in pA (the silent factor-1000 trap NEST's example warns about).

Every other AdEx parameter is left at its default, which equals NEST's
``aeif_cond_alpha`` default (``C_m = 281 pF``, ``g_L = 30 nS``,
``E_L = -70.6 mV``, ``V_th = -50.4 mV`` the exponential-term threshold,
``V_peak = 0 mV`` the spike cutoff, ``V_reset = -60 mV``, ``Delta_T = 2 mV``,
``tau_w = 144 ms``); the membrane starts at ``V_m = -70.6 mV = E_L``.

The whole model runs through the :class:`~brainpy.state.Simulator` (one compiled
``for_loop``); there is no Python step loop.

Run:  python examples/nest_like/brette_gerstner_fig_2c.py

References
----------
.. [1] Brette R and Gerstner W (2005). Adaptive exponential integrate-and-fire
   model as an effective description of neuronal activity. J. Neurophysiology 94.
   https://doi.org/10.1152/jn.00686.2005
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp

#: Integration step (ms) — fine, because the AdEx spike upstroke is very steep.
DT = 0.1
#: Simulation horizon (ms).
T_SIM = 1000.0
#: Paper adaptation parameters: ``a`` in nS, ``b`` in pA (converted from nA).
A_NS = 4.0
B_PA = 80.5
#: The two DC pulses ``(amplitude pA, start ms, stop ms)``.
DC_PULSES = ((500.0, 0.0, 200.0), (800.0, 500.0, 1000.0))


def run(simtime=T_SIM, dt=DT, a=A_NS, b=B_PA):
    """Drive one AdEx neuron with the Fig 2C protocol and return its V trace.

    Parameters
    ----------
    simtime : float, default 1000.0
        Simulation duration in ms.
    dt : float, default 0.1
        Integration time step in ms.
    a : float, default 4.0
        Sub-threshold adaptation coupling (nS). ``a = 0`` removes sub-threshold
        adaptation.
    b : float, default 80.5
        Spike-triggered adaptation increment (pA). ``b = 0`` removes
        spike-triggered adaptation; ``a = b = 0`` reduces AdEx to a plain
        exponential integrate-and-fire neuron (regular firing).

    Returns
    -------
    times : numpy.ndarray
        Time axis (ms), shape ``(n_steps,)``.
    v : numpy.ndarray
        Membrane potential ``V_m`` (mV), shape ``(n_steps,)``.
    spike_steps : numpy.ndarray
        Integer step indices at which the neuron spiked.
    """
    brainstate.environ.set(dt=dt * u.ms)
    sim = bp.Simulator(dt=dt * u.ms)

    # a, b set explicitly (they equal the defaults, but the paper pins them);
    # every other parameter, including the V_m = -70.6 mV start, is the default.
    neuron = sim.create(bp.aeif_cond_alpha, 1, params={'a': a * u.nS, 'b': b * u.pA})

    for amplitude, start, stop in DC_PULSES:
        gen = sim.create(bp.dc_generator, amplitude=amplitude * u.pA,
                         start=start * u.ms, stop=stop * u.ms)
        sim.connect(gen, neuron)

    mm = sim.create(bp.multimeter, record_from=['V_m'], interval=dt * u.ms)
    sr = sim.create(bp.spike_recorder)
    sim.connect(mm, neuron)
    sim.connect(neuron, sr)

    res = sim.simulate(simtime * u.ms)
    times = np.asarray(u.get_mantissa(res.times / u.ms)).reshape(-1)
    v = np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV)).reshape(-1)
    spk = np.asarray(res.spikes(sr)).reshape(len(v), -1)[:, 0]
    spike_steps = np.where(spk > 0)[0]
    return times, v, spike_steps


def main():                                            # pragma: no cover - demo driver
    """Run the Fig 2C protocol and plot the membrane potential."""
    times, v, spikes = run()
    isis = np.diff(spikes) * DT
    print(f"brette_gerstner_fig_2c: {len(spikes)} spikes; "
          f"ISIs (ms) = {[round(float(x), 1) for x in isis]} "
          f"(lengthening = spike-frequency adaptation)")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4, 4.0), dpi=100)
        plt.plot(times, v, lw=0.8)
        plt.axis([0, 1000, -80, -20])
        plt.xlabel("Time (ms)")
        plt.ylabel(r"$V_m$ (mV)")
        plt.title("Brette & Gerstner 2005, Fig 2C — AdEx adaptation")
        plt.tight_layout()
        plt.savefig("examples/nest_like/brette_gerstner_fig_2c.png", dpi=100)
        print("  wrote examples/nest_like/brette_gerstner_fig_2c.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
