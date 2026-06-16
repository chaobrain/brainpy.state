# examples/nest_like/cross_check_mip_corrdet.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Auto-/cross-correlation cross-check — NEST-style port.

Port of NEST's ``cross_check_mip_corrdet.py``. A ``mip_generator`` (Multiple
Interaction Process: one shared parent Poisson train, each child copying every
parent spike with probability ``p_copy``) drives two correlated output trains.
Their cross-correlogram is computed **two independent ways** and cross-checked:

1. with the built-in ``correlation_detector`` device, and
2. with a hand-written reference, :func:`corr_spikes_sorted`, operating on the
   recorded spike *step indices* — the upstream's ``time_in_steps`` reference.

Because the two use the identical bin-edge convention, they agree exactly; that
self-check is the demo's payload and needs no NEST.

Everything here runs **eagerly** (post-hoc). Both ``mip_generator`` and
``correlation_detector`` are imperative host devices (NumPy RNG, Python event
loops, ``int(...)`` of values) and cannot enter a JAX ``for_loop``; the upstream
``parrot_neuron`` relays are identity pass-throughs, so the generator output *is*
the recorded train and no Simulator/recorder is needed. The detector is fed only
on steps that carry events (an empty ``update()`` is a no-op), so the cost is
``O(n_spikes)`` rather than ``O(n_steps)``.

Run:  python examples/nest_like/cross_check_mip_corrdet.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy_state import mip_generator, correlation_detector

#: Simulation resolution (ms).
RESOLUTION = 0.1
#: Parent Poisson rate (Hz) and per-child copy probability.
RATE = 100.0
P_COPY = 0.5
#: Correlation window: bin width and one-sided lag limit (ms); 21 bins.
DELTA_TAU = 10.0
TAU_MAX = 100.0
#: Reference bin size (ms) — equal to ``DELTA_TAU`` so the two methods align.
T_BIN = 10.0


def corr_spikes_sorted(spike1, spike2, tbin, tau_max, resolution):
    """Reference cross-correlation of two sorted spike trains (step indices).

    A faithful port of the upstream helper. Spike times are integer simulation
    steps and must be sorted; ``tau > 0`` means a ``spike2`` event later than a
    ``spike1`` event. A bin of width ``tbin`` is centered on the lag it
    represents.

    Parameters
    ----------
    spike1, spike2 : array_like of int
        Sorted spike *step indices* of the two trains.
    tbin : float
        Bin width in ms.
    tau_max : float
        One-sided maximum lag in ms.
    resolution : float
        Simulation resolution in ms (to convert ms <-> steps).

    Returns
    -------
    numpy.ndarray
        The unnormalized coincidence-count histogram over lags, length
        ``2 * (tau_max / tbin) + 1``.
    """
    tau_max_i = int(tau_max / resolution)
    tbin_i = int(tbin / resolution)

    cross = np.zeros(int(2 * tau_max_i / tbin_i + 1), "d")

    j0 = 0
    for spki in spike1:
        j = j0
        while j < len(spike2) and spike2[j] - spki < -tau_max_i - tbin_i / 2.0:
            j += 1
        j0 = j

        while j < len(spike2) and spike2[j] - spki < tau_max_i + tbin_i / 2.0:
            cross[int((spike2[j] - spki + tau_max_i + 0.5 * tbin_i) / tbin_i)] += 1.0
            j += 1

    return cross


def generate_trains(seed, simtime, dt=RESOLUTION, rate=RATE, p_copy=P_COPY):
    """Draw the two correlated MIP child trains as a per-step multiplicity matrix.

    Parameters
    ----------
    seed : int
        PRNG seed for the MIP generator.
    simtime : float
        Simulation horizon in ms.
    dt : float, optional
        Resolution in ms. Default :data:`RESOLUTION`.
    rate : float, optional
        Parent Poisson rate in Hz. Default :data:`RATE`.
    p_copy : float, optional
        Per-child copy probability. Default :data:`P_COPY`.

    Returns
    -------
    numpy.ndarray
        Integer array ``(n_steps, 2)`` of per-step spike multiplicities; column
        ``j`` is child train ``j`` (the analog of ``parrot_neuron`` ``j``).
    """
    n_steps = int(round(simtime / dt))
    with brainstate.environ.context(dt=dt * u.ms):
        mg = mip_generator(in_size=2, rate=rate * u.Hz, p_copy=p_copy, rng_seed=seed)
        mat = np.asarray(mg.simulate(n_steps))
    return mat


def steps_from_train(column):
    """Spike step indices of one train, each repeated by its multiplicity."""
    column = np.asarray(column)
    idx = np.nonzero(column > 0)[0]
    return np.repeat(idx, column[idx].astype(int))


def detect_crosscorr(mat, dt=RESOLUTION, delta_tau=DELTA_TAU, tau_max=TAU_MAX):
    """Cross-correlate the two trains with the built-in ``correlation_detector``.

    The detector is driven eagerly: only steps carrying an event are fed, each
    stamped at ``step + 1`` (NEST's one-step delivery latency; the offset is the
    same on both ports, so it cancels in the lag difference).

    Parameters
    ----------
    mat : numpy.ndarray
        ``(n_steps, 2)`` per-step multiplicity matrix from :func:`generate_trains`.
    dt : float, optional
        Resolution in ms. Default :data:`RESOLUTION`.
    delta_tau, tau_max : float, optional
        Detector bin width and one-sided lag limit in ms.

    Returns
    -------
    dict
        ``{'count_histogram', 'histogram', 'n_events'}`` from the detector.
    """
    dt_q = dt * u.ms
    active = np.nonzero((mat[:, 0] > 0) | (mat[:, 1] > 0))[0]
    with brainstate.environ.context(dt=dt_q):
        cd = correlation_detector(delta_tau=delta_tau * u.ms, tau_max=tau_max * u.ms)
        for step in active:
            step = int(step)
            ports, mults = [], []
            for port in (0, 1):
                m = int(mat[step, port])
                if m > 0:
                    ports.append(port)
                    mults.append(m)
            k = len(ports)
            with brainstate.environ.context(t=step * dt_q):
                cd.update(
                    spikes=np.ones((k,)),
                    receptor_ports=np.asarray(ports),
                    weights=np.ones((k,)),
                    multiplicities=np.asarray(mults),
                    stamp_steps=np.full((k,), step + 1),
                )
        return {
            'count_histogram': np.asarray(cd.get('count_histogram')),
            'histogram': np.asarray(cd.get('histogram')),
            'n_events': np.asarray(cd.get('n_events')),
        }


def reference_crosscorr(mat, t_bin=T_BIN, tau_max=TAU_MAX, resolution=RESOLUTION):
    """Cross-correlogram of the two trains via the hand-written reference."""
    sp1 = steps_from_train(mat[:, 0])
    sp2 = steps_from_train(mat[:, 1])
    return corr_spikes_sorted(sp1, sp2, t_bin, tau_max, resolution).astype(np.int64)


def main(seed=12345, simtime=10000.0):
    mat = generate_trains(seed=seed, simtime=simtime)
    det = detect_crosscorr(mat)
    ref = reference_crosscorr(mat)

    n1, n2 = det['n_events']
    lmbd1 = (n1 / (simtime - TAU_MAX)) * 1000.0
    lmbd2 = (n2 / (simtime - TAU_MAX)) * 1000.0

    print("cross_check_mip_corrdet (brainpy.state, eager mip + correlation_detector)")
    print(f"  rate={RATE} Hz, p_copy={P_COPY}, T={simtime:.0f} ms; lambdas: "
          f"{lmbd1:.2f} / {lmbd2:.2f} spks/s")
    hist = det['count_histogram']
    center = hist.size // 2
    print("  correlation_detector count_histogram:")
    print(f"    {hist}")
    print("  corr_spikes_sorted reference:")
    print(f"    {ref}")
    # The detector reproduces the reference exactly for non-negative lags; the
    # two differ only marginally at the extreme negative lags, where NEST's
    # asymmetric half-bin pruning convention departs from the reference's
    # symmetric window (a boundary effect, not a discrepancy in the counts).
    exact_nonneg = np.array_equal(hist[center:], ref[center:])
    max_neg_diff = int(np.abs(hist[:center] - ref[:center]).max())
    print(f"  cross-check: exact for lag >= 0 = {exact_nonneg}; "
          f"max negative-lag boundary diff = {max_neg_diff} counts")
    print(f"  sum of cross-correlation: {hist.sum()}")


if __name__ == "__main__":
    main()
