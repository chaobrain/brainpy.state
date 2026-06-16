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
are the ones validated in cluster 07 and shared via
:mod:`brainpy_state._nest_validation._clopath_drive`; this script is the
user-facing presentation of that proof. The presynaptic train relays through a
``spike_generator`` directly into the plastic edge (in NEST a ``parrot_neuron``
relays it, since a device cannot drive a plastic synapse).

Run:  python examples/nest_like/clopath_synapse_spike_pairing.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state._nest_validation import _clopath_drive as drv

#: Pairing frequencies (Hz) of the five trains in each ordering (upstream).
RHO = np.array([10.0, 20.0, 30.0, 40.0, 50.0])


def normalized_weight_change(weights, init_w=drv.INIT_W):
    """Normalise a final weight to the upstream's percent-of-baseline scale.

    The NEST example reports ``100 * 15 * (w - init) / init + 100`` so the
    no-change baseline maps to ``100 %``.

    Parameters
    ----------
    weights : array_like
        Final Clopath weights (mV).
    init_w : float, optional
        Initial weight (mV). Default :data:`_clopath_drive.INIT_W` (``0.5``).

    Returns
    -------
    numpy.ndarray
        Normalised weight change in percent.
    """
    return 100.0 * 15.0 * (np.asarray(weights) - init_w) / init_w + 100.0


def run():
    """Run the 10 canonical pairing trains on the ``Simulator`` API.

    Each train is driven through :func:`_clopath_drive.our_pairing_weight`, which
    builds a single ``clopath_synapse`` edge onto an ``aeif_psc_delta_clopath``
    post, clamps the post with an 80 mV ``spike_generator`` at the postsynaptic
    times, and reads the final stored weight with ``res.weight_trace``.

    Returns
    -------
    rho : numpy.ndarray
        Pairing frequencies (Hz), :data:`RHO`.
    post_pre : numpy.ndarray
        Normalised weight change for the post-before-pre trains
        (:data:`_clopath_drive.LTD_TRAINS`).
    pre_post : numpy.ndarray
        Normalised weight change for the pre-before-post trains
        (:data:`_clopath_drive.LTP_TRAINS`).
    weights : numpy.ndarray
        Raw final Clopath weights (mV) for all 10 trains, in train-index order.
    """
    brainstate.environ.set(dt=drv.DT * u.ms)
    weights = np.array([drv.our_pairing_weight(sp, sq)
                        for sp, sq in zip(drv.SPIKE_TIMES_PRE, drv.SPIKE_TIMES_POST)])
    post_pre = normalized_weight_change(weights[list(drv.LTD_TRAINS)])
    pre_post = normalized_weight_change(weights[list(drv.LTP_TRAINS)])
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
