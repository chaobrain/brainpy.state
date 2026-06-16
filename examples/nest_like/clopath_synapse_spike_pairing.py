# examples/nest_like/clopath_synapse_spike_pairing.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Clopath voltage-based STDP spike-pairing -- NEST-style port.

Port of NEST's ``clopath_synapse_spike_pairing.py``. A presynaptic spike train is
paired with a postsynaptic train (driven by an 80 mV ``spike_generator`` clamp)
across pairing frequencies 10-50 Hz, in both post-before-pre (depression-leaning)
and pre-before-post (potentiation) orderings, onto an ``aeif_psc_delta_clopath``
neuron through a single ``clopath_synapse`` edge. The stored weight after the
protocol is read with ``res.weight_trace`` and the *normalised weight change* is
plotted against pairing frequency (the upstream figure).

``aeif_psc_delta_clopath`` is a **delta** neuron, so the bare ``clopath_synapse``
weight is in **mV** (not pA). The protocol, parameters, and frozen 5 % parity band
are the cluster-07 ones (LTD near-exact, LTP within 5 %). The presynaptic train
relays through a ``spike_generator`` directly into the plastic edge (in NEST a
``parrot_neuron`` relays it, since a device cannot drive a plastic synapse).

Run:  python examples/nest_like/clopath_synapse_spike_pairing.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u
import braintools

from brainpy.state import (Simulator, spike_generator, clopath_synapse,
                           static_synapse, aeif_psc_delta_clopath)

# -- canonical cluster-07 spike-pairing protocol (NEST pynest example) ------------
DT = 0.1            # ms, resolution
RELAY_D = 0.1       # spike_generator -> driver delay (ms) == one step
INIT_W = 0.5        # initial Clopath weight (mV; aeif_psc_delta_clopath is a delta model)
DRIVE_W = 80.0      # post driver weight (mV voltage jump -> forces the V_clamp spike)
DELAY_UBARS = 0.1   # ms; aligned to the substrate's one-step read lag

#: Pairing frequencies (Hz) of the five trains in each ordering (upstream).
RHO = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

# Canonical pre/post spike trains: first five = post-before-pre (depression-leaning),
# last five = pre-before-post (potentiation), at 10/20/30/40/50 Hz pairing rates.
SPIKE_TIMES_PRE = [
    [20.0, 120.0, 220.0, 320.0, 420.0],
    [20.0, 70.0, 120.0, 170.0, 220.0],
    [20.0, 53.3, 86.7, 120.0, 153.3],
    [20.0, 45.0, 70.0, 95.0, 120.0],
    [20.0, 40.0, 60.0, 80.0, 100.0],
    [120.0, 220.0, 320.0, 420.0, 520.0, 620.0],
    [70.0, 120.0, 170.0, 220.0, 270.0, 320.0],
    [53.3, 86.6, 120.0, 153.3, 186.6, 220.0],
    [45.0, 70.0, 95.0, 120.0, 145.0, 170.0],
    [40.0, 60.0, 80.0, 100.0, 120.0, 140.0],
]
SPIKE_TIMES_POST = [
    [10.0, 110.0, 210.0, 310.0, 410.0],
    [10.0, 60.0, 110.0, 160.0, 210.0],
    [10.0, 43.3, 76.7, 110.0, 143.3],
    [10.0, 35.0, 60.0, 85.0, 110.0],
    [10.0, 30.0, 50.0, 70.0, 90.0],
    [130.0, 230.0, 330.0, 430.0, 530.0, 630.0],
    [80.0, 130.0, 180.0, 230.0, 280.0, 330.0],
    [63.3, 96.6, 130.0, 163.3, 196.6, 230.0],
    [55.0, 80.0, 105.0, 130.0, 155.0, 180.0],
    [50.0, 70.0, 90.0, 110.0, 130.0, 150.0],
]
#: Index range of the pure depression-leaning (post-before-pre) trains.
LTD_TRAINS = range(0, 5)
#: Index range of the potentiation (pre-before-post) trains.
LTP_TRAINS = range(5, 10)


def clopath_neuron(sim, n=1):
    """Create ``n`` ``aeif_psc_delta_clopath`` neurons with the canonical parameters."""
    return sim.create(
        aeif_psc_delta_clopath, n,
        E_L=-70.6 * u.mV, V_peak=33.0 * u.mV, C_m=281.0 * u.pF, theta_minus=-70.6 * u.mV,
        theta_plus=-45.3 * u.mV, A_LTD=14.0e-5, A_LTP=8.0e-5, tau_u_bar_minus=10.0 * u.ms,
        tau_u_bar_plus=7.0 * u.ms, delay_u_bars=DELAY_UBARS * u.ms, a=4.0 * u.nS,
        b=0.0805 * u.pA, V_reset=-49.6 * u.mV, V_clamp=33.0 * u.mV, t_clamp=2.0 * u.ms,
        t_ref=0.0 * u.ms, I_e=0.0 * u.pA,
        V_initializer=braintools.init.Constant(-70.6 * u.mV))


def pairing_weight(s_pre, s_post):
    """Final ``clopath_synapse`` weight (mV) after the canonical pairing protocol.

    Builds a single ``clopath_synapse`` edge onto an ``aeif_psc_delta_clopath`` post,
    clamps the post with an 80 mV ``spike_generator`` at the postsynaptic times, and
    reads the final stored weight with ``res.weight_trace``.
    """
    sim = Simulator(dt=DT * u.ms)
    post = clopath_neuron(sim)
    sg_pre = sim.create(spike_generator, spike_times=np.asarray(s_pre) * u.ms)
    sg_post = sim.create(spike_generator, spike_times=np.asarray(s_post) * u.ms)
    sim.connect(sg_post, post, synapse=static_synapse(weight=DRIVE_W * u.mV),
                delay=RELAY_D * u.ms)
    proj = sim.connect(sg_pre, post, synapse=clopath_synapse(weight=INIT_W * u.mV),
                       delay=RELAY_D * u.ms)
    sim.record_weight(proj)
    res = sim.simulate((10.0 + max(s_pre[-1], s_post[-1])) * u.ms)
    return float(u.get_mantissa(res.weight_trace(proj))[-1, 0])


def normalized_weight_change(weights, init_w=INIT_W):
    """Normalise a final weight to the upstream's percent-of-baseline scale.

    The NEST example reports ``100 * 15 * (w - init) / init + 100`` so the
    no-change baseline maps to ``100 %``.

    Parameters
    ----------
    weights : array_like
        Final Clopath weights (mV).
    init_w : float, optional
        Initial weight (mV). Default :data:`INIT_W` (``0.5``).

    Returns
    -------
    numpy.ndarray
        Normalised weight change in percent.
    """
    return 100.0 * 15.0 * (np.asarray(weights) - init_w) / init_w + 100.0


def run():
    """Run the 10 canonical pairing trains on the ``Simulator`` API.

    Returns
    -------
    rho : numpy.ndarray
        Pairing frequencies (Hz), :data:`RHO`.
    post_pre : numpy.ndarray
        Normalised weight change for the post-before-pre trains (:data:`LTD_TRAINS`).
    pre_post : numpy.ndarray
        Normalised weight change for the pre-before-post trains (:data:`LTP_TRAINS`).
    weights : numpy.ndarray
        Raw final Clopath weights (mV) for all 10 trains, in train-index order.
    """
    brainstate.environ.set(dt=DT * u.ms)
    weights = np.array([pairing_weight(sp, sq)
                        for sp, sq in zip(SPIKE_TIMES_PRE, SPIKE_TIMES_POST)])
    post_pre = normalized_weight_change(weights[list(LTD_TRAINS)])
    pre_post = normalized_weight_change(weights[list(LTP_TRAINS)])
    return RHO, post_pre, pre_post, weights


def main():
    rho, post_pre, pre_post, weights = run()
    print("Clopath spike-pairing (brainpy.state, aeif_psc_delta_clopath, w in mV)")
    for r, a, b in zip(rho, post_pre, pre_post):
        print(f"  {r:4.0f} Hz: post-pre {a:7.2f} %   pre-post {b:7.2f} %")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(rho, post_pre, "g.-", label="post-before-pre")
        plt.plot(rho, pre_post, "b.-", label="pre-before-post")
        plt.xlabel("rho (Hz)"); plt.ylabel("normalized weight change (%)")
        plt.title("Clopath synapse -- spike pairing")
        plt.legend(); plt.tight_layout()
        plt.savefig("examples/nest_like/clopath_synapse_spike_pairing.png", dpi=100)
        print("  wrote examples/nest_like/clopath_synapse_spike_pairing.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
