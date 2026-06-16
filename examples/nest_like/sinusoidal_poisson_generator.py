# examples/nest_like/sinusoidal_poisson_generator.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Sinusoidally modulated Poisson drive — NEST-style port.

Port of NEST's §3.7 ``sinusoidal_poisson_generator`` demo. The generator emits an
inhomogeneous Poisson spike train whose instantaneous rate oscillates,

.. math::

    \\lambda(t) = \\max\\!\\big(0,\\; \\mathrm{dc} + \\mathrm{ac}\\cdot
                 \\sin(2\\pi f t + \\varphi)\\big),

across a bank of ``N`` output channels. The headline is that the **population
PSTH tracks** ``λ(t)``, and that the per-bin spike-count **autocorrelation**
carries the modulation period — both reproduced against live NEST distributionally
(see the parity test).

The generator is for_loop-traceable, so it is driven **directly** by
:func:`brainstate.transform.for_loop` over a single ``in_size=N`` instance (the
same loop primitive :meth:`Simulator.simulate` uses internally, and the eager
device idiom of ``cross_check_mip_corrdet``). Driving it this way exercises the
``individual_spike_trains`` flag, which selects the noise mode:

* **individual** (``individual_spike_trains=True``): every channel draws an
  *independent* Poisson sample each step, so the N trains are independent and the
  population PSTH is a smooth estimate of ``λ(t)``.
* **shared** (``individual_spike_trains=False``): a single sample is broadcast to
  all N channels each step, so every column is identical (perfectly synchronous);
  the per-channel rate still follows ``λ(t)`` but the population is one effective
  train.

``λ(t)`` is computed **analytically** (an exact, deterministic ground truth); the
PSTH is built by **summing the per-step spike counts** per bin (Poisson draws give
multiplicities > 1, so the matrix carries counts, not a binary mask).

Run:  PYTHONPATH=. python examples/nest_like/sinusoidal_poisson_generator.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u
import brainstate.transform as transform

from brainpy.state import sinusoidal_poisson_generator

#: DC (mean) rate of the sinusoidal drive (Hz).
RATE = 100.0
#: AC (modulation) amplitude (Hz); ``λ`` swings in ``[dc-ac, dc+ac]`` before clamping.
AMPLITUDE = 50.0
#: Modulation frequency (Hz).
FREQUENCY = 10.0
#: Phase offset (degrees) — NEST's ``phase`` is in degrees.
PHASE = 0.0
#: Number of output channels (independent or shared trains).
N_TARGETS = 50
#: PSTH bin width (ms).
PST_BIN = 10.0
#: Default per-trial horizon (ms) — 10 modulation cycles at 10 Hz.
SIMTIME = 1000.0
#: Simulation resolution (ms).
DT = 0.1
#: One-sided maximum lag (bins) for the spike-count autocorrelation.
MAX_LAG = 30


def lam_of_t(t_ms, rate=RATE, amplitude=AMPLITUDE, frequency=FREQUENCY, phase=PHASE):
    """Analytical instantaneous rate ``λ(t)`` of the sinusoidal Poisson process.

    Parameters
    ----------
    t_ms : array_like
        Time(s) in ms.
    rate : float, optional
        DC rate ``dc`` in Hz. Default :data:`RATE`.
    amplitude : float, optional
        AC amplitude ``ac`` in Hz. Default :data:`AMPLITUDE`.
    frequency : float, optional
        Modulation frequency in Hz. Default :data:`FREQUENCY`.
    phase : float, optional
        Phase offset in degrees. Default :data:`PHASE`.

    Returns
    -------
    numpy.ndarray
        ``λ(t) = max(0, dc + ac·sin(2π·f·t/1000 + phase_rad))`` in Hz, clamped at
        zero (NEST forbids negative rates).
    """
    t_ms = np.asarray(t_ms, dtype=float)
    return np.maximum(
        0.0,
        rate + amplitude * np.sin(2.0 * np.pi * frequency * t_ms / 1000.0
                                  + np.deg2rad(phase)),
    )


def run_spikes(seed=0, dt=DT, simtime=SIMTIME, individual=True, rate=RATE,
               amplitude=AMPLITUDE, frequency=FREQUENCY, phase=PHASE,
               n_targets=N_TARGETS):
    """Drive an ``in_size=n_targets`` generator and return its spike-count matrix.

    The generator is constructed **inside** an ``environ.context(dt=...)`` so its
    timing cache is populated at build time — otherwise the dt-cache-refresh branch
    of ``update()`` fires on the first traced step and raises a tracer-conversion
    error. The whole rollout is one :func:`brainstate.transform.for_loop` (traced
    once, lowered to a single compiled program).

    Parameters
    ----------
    seed : int, optional
        PRNG seed for the generator. Default ``0``.
    dt : float, optional
        Simulation resolution in ms. Default :data:`DT`.
    simtime : float, optional
        Horizon in ms. Default :data:`SIMTIME`.
    individual : bool, optional
        ``True`` (default) → each channel is an independent train; ``False`` → one
        train broadcast to all channels (the shared mode). Sets the generator's
        ``individual_spike_trains`` flag.
    rate, amplitude, frequency, phase : float, optional
        Drive parameters; see :func:`lam_of_t`. ``amplitude`` may exceed ``rate``
        (the rate is clamped at zero), and ``amplitude=0`` gives a stationary
        Poisson drive at ``rate``.
    n_targets : int, optional
        Number of output channels. Default :data:`N_TARGETS`.

    Returns
    -------
    numpy.ndarray
        ``(n_steps, n_targets)`` per-step spike counts (multiplicity preserved).
    """
    n_steps = int(round(simtime / dt))
    with brainstate.environ.context(dt=dt * u.ms):
        gen = sinusoidal_poisson_generator(
            in_size=n_targets, rate=rate * u.Hz, amplitude=amplitude * u.Hz,
            frequency=frequency * u.Hz, phase=phase,
            individual_spike_trains=individual, rng_seed=seed)
        brainstate.nn.init_all_states(gen)
        times = u.math.arange(0.0 * u.ms, n_steps * dt * u.ms, dt * u.ms)
        idx = u.math.arange(times.size)

        def step(t, i):
            with brainstate.environ.context(t=t, i=i):
                return gen.update()

        return np.asarray(transform.for_loop(step, times, idx))


def _pop_counts(spk, dt=DT, bin_ms=PST_BIN):
    """Per-bin population spike count (sum over channels, then bin)."""
    spk = np.asarray(spk)
    n_steps, _n = spk.shape
    per_bin = int(round(bin_ms / dt))
    n_bins = n_steps // per_bin
    return spk[:n_bins * per_bin].sum(axis=1).reshape(n_bins, per_bin).sum(axis=1)


def population_psth(spk, dt=DT, bin_ms=PST_BIN):
    """Population PSTH in Hz from the recorded spike-count matrix.

    Parameters
    ----------
    spk : numpy.ndarray
        ``(n_steps, n_targets)`` per-step spike counts from :func:`run_spikes`.
    dt : float, optional
        Resolution in ms. Default :data:`DT`.
    bin_ms : float, optional
        PSTH bin width in ms. Default :data:`PST_BIN`.

    Returns
    -------
    centers_ms : numpy.ndarray
        Bin-center times in ms.
    psth_hz : numpy.ndarray
        Population firing rate per bin in Hz: ``counts / (n_targets · bin_s)``.
    """
    spk = np.asarray(spk)
    n = spk.shape[1]
    counts = _pop_counts(spk, dt, bin_ms)
    n_bins = counts.shape[0]
    centers = (np.arange(n_bins) + 0.5) * bin_ms
    return centers, counts / (n * bin_ms / 1000.0)


def spike_count_autocorr(spk, dt=DT, bin_ms=PST_BIN, max_lag=MAX_LAG):
    """Normalized autocorrelation of the per-bin population spike-count series.

    The modulation imprints a periodic structure (period ``1/frequency``) on the
    population count series, so its autocorrelation oscillates at the same period.
    This is the quantity compared against NEST distributionally.

    Parameters
    ----------
    spk : numpy.ndarray
        ``(n_steps, n_targets)`` spike-count matrix.
    dt : float, optional
        Resolution in ms. Default :data:`DT`.
    bin_ms : float, optional
        Bin width in ms. Default :data:`PST_BIN`.
    max_lag : int, optional
        One-sided maximum lag in bins. Default :data:`MAX_LAG`.

    Returns
    -------
    numpy.ndarray
        Autocorrelation at lags ``0 … max_lag`` (length ``max_lag + 1``),
        normalized so lag 0 is 1.0; all zeros if the series has no variance.
    """
    c = _pop_counts(spk, dt, bin_ms).astype(float)
    c = c - c.mean()
    var = float(np.dot(c, c))
    if var <= 0.0:
        return np.zeros(max_lag + 1)
    n = c.shape[0]
    acf = np.array([float(np.dot(c[:n - lag], c[lag:])) for lag in range(max_lag + 1)])
    return acf / var


def main():
    print("sinusoidal_poisson_generator (brainpy.state, eager for_loop drive)")
    print(f"  λ(t) = max(0, {RATE:.0f} + {AMPLITUDE:.0f}·sin(2π·{FREQUENCY:.0f}·t)) Hz, "
          f"{N_TARGETS} channels, T={SIMTIME:.0f} ms")

    spk = run_spikes(seed=0, individual=True)
    centers, psth = population_psth(spk)
    lam = lam_of_t(centers)
    corr = np.corrcoef(psth, lam)[0, 1]
    depth = psth.max() / max(psth.min(), 1e-9)
    print(f"  [individual] PSTH-vs-λ corr = {corr:.3f}, modulation depth = {depth:.2f}, "
          f"mean = {psth.mean():.1f} Hz")
    acf = spike_count_autocorr(spk)
    print(f"  [individual] autocorr (lags 0..5): "
          f"{np.array2string(acf[:6], precision=2, separator=', ')}")

    shared = run_spikes(seed=0, individual=False)
    identical = bool(np.all(shared == shared[:, :1]))
    print(f"  [shared] all {N_TARGETS} columns identical: {identical} "
          f"(synchronous single train)")


if __name__ == "__main__":
    main()
