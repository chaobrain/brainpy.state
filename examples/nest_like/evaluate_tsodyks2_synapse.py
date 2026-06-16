# examples/nest_like/evaluate_tsodyks2_synapse.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tsodyks-2 short-term plasticity -- NEST-style port.

Port of NEST's ``evaluate_tsodyks2_synapse.py``. A regular presynaptic spike burst
(plus a recovery pair) drives a single ``tsodyks2_synapse`` edge onto a linear,
never-spiking ``iaf_psc_exp`` post neuron (``V_th = 1e4`` mV, ``tau_syn_ex = 3 ms``);
the post membrane potential **is** the PSC-amplitude train, carrying the depression
or facilitation envelope. Two regimes are shown:

* depression   ``U=0.67, tau_rec=450 ms, tau_fac=0``    -- EPSP peaks shrink
* facilitation ``U=0.1,  tau_rec=100 ms, tau_fac=1000`` -- EPSP peaks grow

Plastic synapses cannot be driven by a device in NEST (a ``parrot_neuron`` relays
the train); on the ``Simulator`` API a ``spike_generator`` drives the plastic edge
directly. Parameters follow ``pynest/examples/evaluate_tsodyks2_synapse.py`` and the
cluster-01 STP parity drive. ``tsodyks2_synapse`` delivers ``w_eff = x*u*weight``.

Run:  python examples/nest_like/evaluate_tsodyks2_synapse.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import braintools
import numpy as np
import brainunit as u

from brainpy.state import (Simulator, iaf_psc_exp, spike_generator, multimeter,
                           tsodyks2_synapse)

#: Resolution (ms).
DT = 0.1
#: 50 ms-ISI burst then a recovery pair (cluster-01 STP protocol), in ms.
TRAIN = [50., 100., 150., 200., 250., 300., 350., 400., 650., 700.]
#: Simulation horizon (ms).
T_SIM = 800.0
#: Per-edge weight (pA).
WEIGHT = 250.0
#: The two STP regimes (Tsodyks et al. parameters, upstream values).
REGIMES = {
    "depression": dict(U=0.67, u=0.67, x=1.0, tau_rec=450.0, tau_fac=0.0),
    "facilitation": dict(U=0.1, u=0.1, x=1.0, tau_rec=100.0, tau_fac=1000.0),
}


def _post(sim):
    """Build the linear, never-spiking ``iaf_psc_exp`` post (V_m == PSC train)."""
    return sim.create(
        iaf_psc_exp, 1, C_m=250. * u.pF, tau_m=20. * u.ms,
        tau_syn_ex=3. * u.ms, tau_syn_in=3. * u.ms, t_ref=2. * u.ms,
        E_L=0. * u.mV, V_reset=0. * u.mV, V_th=1e4 * u.mV,
        V_initializer=braintools.init.Constant(0. * u.mV))


def run(regime="depression", weight=WEIGHT, train=TRAIN, t_sim=T_SIM):
    """Drive one ``tsodyks2_synapse`` edge with the burst-and-recovery protocol.

    Parameters
    ----------
    regime : {"depression", "facilitation"}, optional
        Which STP regime's parameters to use. Default ``"depression"``.
    weight : float, optional
        Per-edge weight (pA). Default :data:`WEIGHT`.
    train : sequence of float, optional
        Presynaptic spike times (ms). Default :data:`TRAIN`.
    t_sim : float, optional
        Simulation horizon (ms). Default :data:`T_SIM`.

    Returns
    -------
    times : numpy.ndarray
        Recorder time axis (ms).
    vm : numpy.ndarray
        Post membrane potential (mV) -- the PSC-amplitude train.
    """
    p = REGIMES[regime]
    sim = Simulator(dt=DT * u.ms)
    post = _post(sim)
    sg = sim.create(spike_generator, spike_times=np.asarray(train) * u.ms)
    sim.connect(sg, post, synapse=tsodyks2_synapse(
        weight=weight * u.pA, U=p["U"], u=p["u"], x=p["x"],
        tau_rec=p["tau_rec"] * u.ms, tau_fac=p["tau_fac"] * u.ms))
    mm = sim.create(multimeter, record_from=["V_m"], interval=DT * u.ms)
    sim.connect(mm, post)              # reversed: the multimeter observes the neuron
    res = sim.simulate(t_sim * u.ms)
    times = np.asarray(u.get_mantissa(res.times / u.ms))
    vm = np.asarray(u.get_mantissa(res.trace(mm, "V_m") / u.mV)).reshape(-1)
    return times, vm


def burst_peak_ratio(vm, train=TRAIN, dt=DT):
    """Ratio of the last to first EPSP peak across the initial burst.

    A crude depression/facilitation indicator: ``< 1`` for depression (peaks
    shrink), ``> 1`` for facilitation (peaks grow). The peak after spike ``k`` is
    sampled a few ms after the spike (EPSP rise), within the 50 ms-ISI burst.

    Parameters
    ----------
    vm : numpy.ndarray
        Post V_m trace (mV).
    train : sequence of float, optional
        Presynaptic spike times (ms).
    dt : float, optional
        Resolution (ms).

    Returns
    -------
    float
        ``peak_last / peak_first`` over the eight-spike burst.
    """
    burst = [t for t in train if t <= 400.0]
    peaks = []
    for t in burst:
        lo, hi = int(round(t / dt)), int(round((t + 40.0) / dt))
        peaks.append(float(np.max(vm[lo:hi])))
    return peaks[-1] / peaks[0]


def main():
    print("Tsodyks-2 STP (brainpy.state, iaf_psc_exp post, V_m = PSC train)")
    traces = {}
    for regime in REGIMES:
        t, v = run(regime)
        traces[regime] = (t, v)
        ratio = burst_peak_ratio(v)
        print(f"  {regime:12s}: V_m max {v.max():.3f} mV, last/first burst peak {ratio:.3f}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        for regime, c in (("depression", "C0"), ("facilitation", "C1")):
            t, v = traces[regime]
            plt.plot(t, v, c, label=regime)
        plt.xlabel("time (ms)"); plt.ylabel("V_m (mV)")
        plt.title("tsodyks2_synapse -- depression vs facilitation")
        plt.legend(); plt.tight_layout()
        plt.savefig("examples/nest_like/evaluate_tsodyks2_synapse.png", dpi=100)
        print("  wrote examples/nest_like/evaluate_tsodyks2_synapse.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
