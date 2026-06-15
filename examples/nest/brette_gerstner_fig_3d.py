# examples/nest/brette_gerstner_fig_3d.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Brette & Gerstner (2005) Fig 3D — AdEx post-inhibitory rebound (NEST port).

Port of NEST's ``brette_gerstner_fig_3d.py``. A single adaptive exponential
integrate-and-fire neuron (:class:`~brainpy.state.aeif_cond_exp`) is hyperpolarised
by an 800 pA *inhibitory* (negative) step current over ``[0, 400) ms``; when the
step is released the membrane *rebounds* through threshold and fires a short
burst — Figure 3D of Brette & Gerstner [1]_, the post-inhibitory-rebound regime.

The parameters come from the paper: ``V_peak = 20 mV``, ``E_L = -60 mV``,
``a = 80 nS`` (strong sub-threshold adaptation), ``b = 80.5 pA`` (``b`` given in
nA in the paper — convert to pA), ``tau_w = 720 ms`` (slow adaptation). All other
AdEx parameters are NEST's ``aeif_cond_exp`` defaults.

.. note::

   The upstream demo sets ``E_L = -60 mV`` but leaves ``V_m`` at its default, so
   the membrane starts at ``V_m = -70.6 mV`` — *below* the leak reversal — and the
   recorded trace is the hyperpolarisation from there. NEST does not move ``V_m``
   to a freshly-set ``E_L``; this port keeps the matching ``V_m = -70.6 mV``
   default (it does **not** initialise at ``E_L``). Pinning ``V_m`` to ``E_L``
   would start the run 10.6 mV too high and break the parity.

The whole model runs through the :class:`~brainpy.state.Simulator` (one compiled
``for_loop``); there is no Python step loop.

Run:  python examples/nest/brette_gerstner_fig_3d.py

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

#: Integration step (ms).
DT = 0.1
#: Simulation horizon (ms).
T_SIM = 1000.0
#: Paper parameters.
V_PEAK_MV = 20.0
E_L_MV = -60.0
A_NS = 80.0
B_PA = 80.5
TAU_W_MS = 720.0
#: The single inhibitory DC step ``(amplitude pA, start ms, stop ms)``.
DC_STEP = (-800.0, 0.0, 400.0)


def run(simtime=T_SIM, dt=DT, a=A_NS, b=B_PA, tau_w=TAU_W_MS):
    """Drive one AdEx neuron with the Fig 3D protocol and return its V trace.

    Parameters
    ----------
    simtime : float, default 1000.0
        Simulation duration in ms.
    dt : float, default 0.1
        Integration time step in ms.
    a : float, default 80.0
        Sub-threshold adaptation coupling (nS).
    b : float, default 80.5
        Spike-triggered adaptation increment (pA).
    tau_w : float, default 720.0
        Adaptation time constant (ms).

    Returns
    -------
    times : numpy.ndarray
        Time axis (ms), shape ``(n_steps,)``.
    v : numpy.ndarray
        Membrane potential ``V_m`` (mV), shape ``(n_steps,)``.
    spike_steps : numpy.ndarray
        Integer step indices at which the neuron spiked (the rebound burst).
    """
    brainstate.environ.set(dt=dt * u.ms)
    sim = bp.Simulator(dt=dt * u.ms)

    # V_m is intentionally NOT initialised to E_L: the default -70.6 mV matches
    # NEST, which leaves V_m at its default when E_L is set to -60 mV (see note).
    neuron = sim.create(bp.aeif_cond_exp, 1, params={
        'V_peak': V_PEAK_MV * u.mV, 'E_L': E_L_MV * u.mV, 'a': a * u.nS,
        'b': b * u.pA, 'tau_w': tau_w * u.ms})

    amplitude, start, stop = DC_STEP
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
    """Run the Fig 3D protocol and plot the membrane potential."""
    times, v, spikes = run()
    rebound = spikes[spikes * DT >= DC_STEP[2]]
    print(f"brette_gerstner_fig_3d: {len(spikes)} spike(s); "
          f"first rebound at t = {rebound[0] * DT:.1f} ms (current released at "
          f"{DC_STEP[2]:.0f} ms) — post-inhibitory rebound")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4, 4.0), dpi=100)
        plt.plot(times, v, lw=0.8)
        plt.axis([0, 1000, -85, 0])
        plt.xlabel("Time (ms)")
        plt.ylabel(r"$V_m$ (mV)")
        plt.title("Brette & Gerstner 2005, Fig 3D — post-inhibitory rebound")
        plt.tight_layout()
        plt.savefig("examples/nest/brette_gerstner_fig_3d.png", dpi=100)
        print("  wrote examples/nest/brette_gerstner_fig_3d.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
