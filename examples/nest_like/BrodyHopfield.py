# examples/nest_like/BrodyHopfield.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Spike synchronization through a subthreshold oscillation — NEST-style port.

Ports NEST's ``pynest/examples/BrodyHopfield.py`` (Brody & Hopfield, 2003,
*Neuron* 37:843, Fig. 1) to the Simulator API. A population of
``iaf_psc_alpha`` neurons is driven by a *weak, shared* 35 Hz subthreshold
oscillation, an *independent* Gaussian noise current, and a per-neuron internal
DC bias that ramps across the population. Although the oscillation alone is too
weak to make any neuron fire, it biases *when* the noise-plus-DC drive pushes a
neuron over threshold, so the population spikes preferentially near a fixed phase
of the oscillation — the spikes synchronize to a current that never by itself
elicits a spike.

Each of the ``N`` neurons receives:

1. a **shared** 35 Hz drive — one ``ac_generator`` (amplitude 50 pA), identical
   to every neuron, so the subthreshold oscillation has a common phase;
2. **independent** white noise — one ``noise_generator`` (mean 0, std 200 pA),
   delivering an independent stream to each neuron;
3. a per-neuron internal DC bias ``I_e`` ramping ``140 → 200 pA`` across the
   population (neuron ``n`` of ``N`` gets ``n*(200-140)/N + 140`` pA).

**The fan-out is NEST-faithful with no manual sizing.** The Simulator realizes a
current device at the post size and scatters it one channel per target: the
deterministic ``ac_generator`` therefore drives every neuron with an *identical*
sinusoid (one shared oscillation), while the ``noise_generator`` draws an
*independent* sample per neuron each step (independent noise) — exactly NEST's
single-generator fan-out semantics.

**Validation is a distributional / phase carve-out, not a trace parity.** The
noise is a PRNG stream that diverges between NEST and JAX, so a per-sample
comparison is meaningless; both simulators agree only *distributionally*. The
phase-locking is imposed by the *deterministic* shared oscillation, so the phase
statistics are the robust ground truth:
``brainpy_state/_nest/_validation/BrodyHopfield_test.py`` asserts that the
oscillation induces phase-locking (and that removing it abolishes it), and — when
NEST is present — that the seed-averaged vector strength, firing rate and phase
histogram match live NEST.

The locking is quantified by the **vector strength** (Rayleigh resultant length)
``R = |mean(exp(i*theta_k))|`` of the spike phases ``theta_k`` relative to the
35 Hz drive: ``R = 0`` is no locking (uniform phases), ``R -> 1`` is perfect
locking.

Run:  PYTHONPATH=. python examples/nest_like/BrodyHopfield.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import braintools
import brainunit as u

from brainpy.state import (Simulator, iaf_psc_alpha, ac_generator,
                           noise_generator, spike_recorder)

# --- NEST BrodyHopfield parameters (verbatim) --------------------------------
N = 1000            # number of neurons
BIAS_BEGIN = 140.0  # minimal per-neuron DC bias [pA]
BIAS_END = 200.0    # maximal per-neuron DC bias [pA]
T = 600.0           # simulation time [ms]
DT = 0.1            # resolution [ms]

DRIVE_AMP = 50.0    # AC drive amplitude [pA]
DRIVE_FREQ = 35.0   # AC drive frequency [Hz]
NOISE_STD = 200.0   # noise current standard deviation [pA]
NOISE_DT = 1.0      # noise refresh interval [ms] (NEST noise_generator default)

# iaf_psc_alpha params (NEST verbatim; potentials are absolute mV).
TAU_M = 20.0        # membrane time constant [ms]
V_TH = 20.0         # threshold potential [mV]
E_L = 10.0          # resting potential [mV]
T_REF = 2.0         # refractory period [ms]
V_RESET = 0.0       # reset potential [mV]
C_M = 200.0         # membrane capacitance [pF]
V_INIT = 0.0        # initial membrane potential [mV]

WARMUP = 100.0      # spikes before this (ms) are dropped (V_m=0 onset transient)


def bias_ramp(n):
    """Per-neuron DC bias ``I_e`` (pA), NEST's ``k*(end-begin)/n + begin`` ramp.

    Parameters
    ----------
    n : int
        Number of neurons. The divisor is ``n`` (as in NEST, where it is the
        population size), so the ramp spans ``(begin, end]`` for any ``n``.

    Returns
    -------
    numpy.ndarray
        Length-``n`` bias currents in pA, ascending from just above
        :data:`BIAS_BEGIN` to :data:`BIAS_END`.
    """
    return np.arange(1, n + 1) * (BIAS_END - BIAS_BEGIN) / n + BIAS_BEGIN


def vector_strength(spike_times_ms, freq_hz=DRIVE_FREQ):
    r"""Vector strength and preferred phase of spikes wrt a reference oscillation.

    Parameters
    ----------
    spike_times_ms : array_like
        Pooled spike times in ms.
    freq_hz : float, optional
        Reference frequency in Hz. Default :data:`DRIVE_FREQ`.

    Returns
    -------
    R : float
        Vector strength ``|mean(exp(i*theta))|`` in ``[0, 1]`` (Rayleigh
        resultant length); ``0`` for an empty spike set.
    preferred_phase : float
        Mean phase ``angle(mean(exp(i*theta)))`` in radians, ``[-pi, pi]``;
        ``0`` for an empty spike set.
    """
    t = np.asarray(spike_times_ms, dtype=float)
    if t.size == 0:
        return 0.0, 0.0
    omega = 2.0 * np.pi * freq_hz / 1000.0          # rad/ms
    z = np.mean(np.exp(1j * omega * t))
    return float(np.abs(z)), float(np.angle(z))


def phase_histogram(spike_times_ms, freq_hz=DRIVE_FREQ, n_bins=18, align_phase=None):
    r"""Normalized spike-phase histogram wrt a reference oscillation.

    Parameters
    ----------
    spike_times_ms : array_like
        Pooled spike times in ms.
    freq_hz : float, optional
        Reference frequency in Hz. Default :data:`DRIVE_FREQ`.
    n_bins : int, optional
        Number of phase bins over ``[0, 2*pi)``. Default ``18``.
    align_phase : float or None, optional
        If given, phases are rotated by ``-align_phase`` before binning (used to
        remove a constant phase offset, e.g. a connection-delay shift, so two
        histograms can be compared by shape). Default ``None``.

    Returns
    -------
    numpy.ndarray
        Length-``n_bins`` density that sums to 1 (all-zero for an empty spike set).
    """
    t = np.asarray(spike_times_ms, dtype=float)
    if t.size == 0:
        return np.zeros(n_bins, dtype=float)
    omega = 2.0 * np.pi * freq_hz / 1000.0
    theta = (omega * t) % (2.0 * np.pi)
    if align_phase is not None:
        theta = (theta - align_phase) % (2.0 * np.pi)
    counts, _ = np.histogram(theta, bins=n_bins, range=(0.0, 2.0 * np.pi))
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def pooled_spike_times(spk, t_ms, warmup=WARMUP):
    """Pool a ``(T, N)`` spike matrix into a 1-D array of spike times (ms).

    Parameters
    ----------
    spk : array_like
        ``(n_steps, n_neurons)`` spike matrix (``> 0`` marks a spike).
    t_ms : array_like
        ``(n_steps,)`` time axis in ms.
    warmup : float, optional
        Spikes at ``t < warmup`` are dropped (onset transient). Default
        :data:`WARMUP`.

    Returns
    -------
    numpy.ndarray
        Pooled spike times (ms), ``t >= warmup``.
    """
    spk = np.asarray(spk)
    t_ms = np.asarray(t_ms, dtype=float)
    steps, _neurons = np.nonzero(spk > 0)
    times = t_ms[steps]
    return times[times >= warmup]


def build(n=N, t_sim=T, *, drive_amp=DRIVE_AMP, noise_std=NOISE_STD, seed=0, i_e=None):
    """Build the BrodyHopfield Simulator: population + shared AC + independent noise.

    Parameters
    ----------
    n : int, optional
        Number of neurons. Default :data:`N`.
    t_sim : float, optional
        Simulation horizon in ms. Default :data:`T`.
    drive_amp : float, optional
        AC drive amplitude in pA (``0`` disables the oscillation — the control).
        Default :data:`DRIVE_AMP`.
    noise_std : float, optional
        Noise current std in pA (``0`` disables the noise). Default
        :data:`NOISE_STD`.
    seed : int, optional
        Base PRNG seed for the noise generator. Default ``0``.
    i_e : float or None, optional
        If given, every neuron gets this constant DC bias (pA) instead of the
        per-neuron :func:`bias_ramp`. Used to isolate the device fan-out (identical
        bias) from the bias gradient. Default ``None`` (use the ramp).

    Returns
    -------
    sim : Simulator
    sr : NodeView
        Spike recorder on the population (``res.spikes(sr)`` / ``res.rate(sr)``).
    t_sim : float
    """
    bias = np.full(n, float(i_e)) if i_e is not None else bias_ramp(n)
    sim = Simulator(dt=DT * u.ms)
    neurons = sim.create(iaf_psc_alpha, n, params=dict(
        tau_m=TAU_M * u.ms, V_th=V_TH * u.mV, E_L=E_L * u.mV, t_ref=T_REF * u.ms,
        V_reset=V_RESET * u.mV, C_m=C_M * u.pF, I_e=bias * u.pA,
        V_initializer=braintools.init.Constant(V_INIT * u.mV),
    ))
    # Shared deterministic oscillation: realized at n, identical on every channel.
    drive = sim.create(ac_generator, amplitude=drive_amp * u.pA,
                       frequency=DRIVE_FREQ * u.Hz)
    # Independent noise: realized at n, draws an independent sample per neuron,
    # refreshed every NOISE_DT ms (NEST's noise_generator default; the membrane
    # noise variance scales with this interval, so it must match NEST's 1.0 ms).
    noise = sim.create(noise_generator, mean=0.0 * u.pA, std=noise_std * u.pA,
                       noise_dt=NOISE_DT * u.ms, seed=seed)
    sr = sim.create(spike_recorder)
    sim.connect(drive, neurons)     # ordinal fixed before noise -> stable noise seed
    sim.connect(noise, neurons)
    sim.connect(neurons, sr)
    return sim, sr, t_sim


def run_phase_locking(n=N, t_sim=T, *, drive_amp=DRIVE_AMP, noise_std=NOISE_STD,
                      seed=0, warmup=WARMUP):
    """Run the demo and return the phase-locking summary plus raw spike data.

    Returns
    -------
    dict
        ``R`` (vector strength), ``preferred_phase`` (rad), ``rate_hz`` (mean
        population rate), ``n_spikes`` (post-warmup pooled count), ``spike_times``
        (pooled ms array), ``spk`` (``(T, n)`` matrix) and ``t`` (ms axis).
    """
    sim, sr, t_sim = build(n, t_sim, drive_amp=drive_amp, noise_std=noise_std,
                           seed=seed)
    res = sim.simulate(t_sim * u.ms)
    spk = np.asarray(res.spikes(sr))
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    times = pooled_spike_times(spk, t, warmup)
    R, psi = vector_strength(times)
    return dict(R=R, preferred_phase=psi, rate_hz=float(res.rate(sr)),
                n_spikes=int(times.size), spike_times=times, spk=spk, t=t)


def main():
    print("BrodyHopfield: spike synchronization through subthreshold oscillation "
          "(brainpy.state)")
    osc = run_phase_locking()
    flat = run_phase_locking(drive_amp=0.0)     # control: no oscillation
    print(f"  with 35 Hz drive : vector strength R = {osc['R']:.3f}, "
          f"preferred phase = {np.degrees(osc['preferred_phase']):6.1f} deg, "
          f"rate = {osc['rate_hz']:.1f} spk/s, spikes = {osc['n_spikes']}")
    print(f"  no oscillation   : vector strength R = {flat['R']:.3f}  "
          f"(control; should be near zero)")
    print(f"  locking gain     : R_osc / R_flat = {osc['R'] / max(flat['R'], 1e-9):.1f}x")

    try:
        import matplotlib.pyplot as plt
        spk, t = osc['spk'], osc['t']
        steps, ids = np.nonzero(spk > 0)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
        ax1.scatter(t[steps], ids, s=0.5, color="k")
        ax1.set_xlabel("time [ms]")
        ax1.set_ylabel("neuron (sorted by bias current)")
        ax1.set_title(f"Phase-locking raster (R = {osc['R']:.3f})")
        hist = phase_histogram(osc['spike_times'])
        centers = np.degrees((np.arange(hist.size) + 0.5) * 2 * np.pi / hist.size)
        ax2.bar(centers, hist, width=360.0 / hist.size * 0.9, color="C0",
                label="35 Hz drive")
        ax2.bar(centers, phase_histogram(flat['spike_times']),
                width=360.0 / hist.size * 0.45, color="C3", alpha=0.7,
                label="no oscillation")
        ax2.axhline(1.0 / hist.size, color="gray", ls=":", label="uniform")
        ax2.set_xlabel("oscillation phase [deg]")
        ax2.set_ylabel("spike probability")
        ax2.set_title("Spike-phase distribution")
        ax2.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig("examples/nest_like/BrodyHopfield.png", dpi=100)
        print("  wrote examples/nest_like/BrodyHopfield.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
