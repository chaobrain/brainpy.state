# examples/nest_like/sinusoidal_gamma_generator.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Sinusoidally modulated gamma drive — NEST-style port.

Port of NEST's §3.7 ``sinusoidal_gamma_generator`` demo. Like the
:mod:`sinusoidal_poisson_generator` demo the instantaneous rate is

.. math::

    \\lambda(t) = \\max\\!\\big(0,\\; \\mathrm{dc} + \\mathrm{ac}\\cdot
                 \\sin(2\\pi f t + \\varphi)\\big),

but spikes are drawn from a **gamma renewal process of order** ``m`` rather than a
Poisson process. The headline is the **gamma regularization law**: for a renewal
process of order ``m`` the inter-spike-interval coefficient of variation is

.. math::

    \\mathrm{CV} = \\frac{\\sigma_{\\mathrm{ISI}}}{\\mu_{\\mathrm{ISI}}}
                 \\;\\longrightarrow\\; \\frac{1}{\\sqrt{m}},

so ``m = 1`` recovers a Poisson train (CV → 1) and large ``m`` gives an
increasingly clock-like train (CV → 0). Two scenarios are shown:

* **stationary CV → 1/√m** (the headline): ``amplitude = 0`` (pure DC rate),
  swept over ``m ∈`` :data:`ORDERS`. ISIs are pooled across the output channels
  and the CV is compared to ``1/√m`` and, distributionally, to live NEST.
* **modulated rate**: ``amplitude > 0`` — the population PSTH still tracks
  ``λ(t)`` (the gamma process only *regularizes* the train, it does not change the
  mean rate profile).

The generator is driven **directly** by :func:`brainstate.transform.for_loop` over
a single ``in_size=N`` instance (the same loop primitive :meth:`Simulator.simulate`
uses internally). It emits **binary** spikes (a renewal process fires at most once
per step), and the ``individual_spike_trains`` flag selects independent vs shared
channels exactly as in the Poisson demo.

Run:  PYTHONPATH=. python examples/nest_like/sinusoidal_gamma_generator.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u
import brainstate.transform as transform

from brainpy.state import sinusoidal_gamma_generator

#: DC (mean) rate of the drive (Hz).
RATE = 100.0
#: AC (modulation) amplitude (Hz) for the modulated-rate scenario; ``0`` for CV.
AMPLITUDE = 50.0
#: Modulation frequency (Hz).
FREQUENCY = 10.0
#: Phase offset (degrees).
PHASE = 0.0
#: Gamma orders swept by the stationary CV scenario (m=1 ≡ Poisson).
ORDERS = (1, 2, 6, 10)
#: Gamma order used by the modulated-rate scenario.
MODULATED_ORDER = 3
#: Number of output channels.
N_TARGETS = 50
#: PSTH bin width (ms).
PST_BIN = 10.0
#: Default horizon (ms) — long enough to pool many ISIs per channel.
SIMTIME = 2000.0
#: Simulation resolution (ms).
DT = 0.1


def lam_of_t(t_ms, rate=RATE, amplitude=AMPLITUDE, frequency=FREQUENCY, phase=PHASE):
    """Analytical instantaneous rate ``λ(t)`` of the drive (Hz, clamped at zero).

    Parameters
    ----------
    t_ms : array_like
        Time(s) in ms.
    rate, amplitude, frequency : float, optional
        DC rate, AC amplitude, and modulation frequency in Hz.
    phase : float, optional
        Phase offset in degrees.

    Returns
    -------
    numpy.ndarray
        ``max(0, dc + ac·sin(2π·f·t/1000 + phase_rad))`` in Hz.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    return np.maximum(
        0.0,
        rate + amplitude * np.sin(2.0 * np.pi * frequency * t_ms / 1000.0
                                  + np.deg2rad(phase)),
    )


def run_spikes(seed=0, dt=DT, simtime=SIMTIME, order=MODULATED_ORDER,
               amplitude=AMPLITUDE, individual=True, rate=RATE, frequency=FREQUENCY,
               phase=PHASE, n_targets=N_TARGETS):
    """Drive an ``in_size=n_targets`` gamma generator → binary spike-count matrix.

    The generator is constructed inside an ``environ.context(dt=...)`` so its timing
    cache is set before tracing (the dt-refresh branch of ``update()`` is not
    tracer-safe), and the rollout is a single :func:`brainstate.transform.for_loop`.

    Parameters
    ----------
    seed : int, optional
        PRNG seed for the generator. Default ``0``.
    dt : float, optional
        Resolution in ms. Default :data:`DT`.
    simtime : float, optional
        Horizon in ms. Default :data:`SIMTIME`.
    order : float, optional
        Gamma renewal order ``m`` (``≥ 1``). Default :data:`MODULATED_ORDER`.
    amplitude : float, optional
        AC amplitude in Hz (``0 ≤ amplitude ≤ rate``); ``0`` is the stationary CV
        scenario. Default :data:`AMPLITUDE`.
    individual : bool, optional
        ``True`` (default) → independent channels; ``False`` → one train broadcast
        to all channels. Sets ``individual_spike_trains``.
    rate, frequency, phase, n_targets : optional
        Drive parameters; see :func:`lam_of_t` and :data:`N_TARGETS`.

    Returns
    -------
    numpy.ndarray
        ``(n_steps, n_targets)`` binary spike matrix (renewal fires ≤ 1/step).
    """
    n_steps = int(round(simtime / dt))
    with brainstate.environ.context(dt=dt * u.ms):
        gen = sinusoidal_gamma_generator(
            in_size=n_targets, rate=rate * u.Hz, amplitude=amplitude * u.Hz,
            frequency=frequency * u.Hz, phase=phase, order=order,
            individual_spike_trains=individual, rng_seed=seed)
        brainstate.nn.init_all_states(gen)
        times = u.math.arange(0.0 * u.ms, n_steps * dt * u.ms, dt * u.ms)
        idx = u.math.arange(times.size)

        def step(t, i):
            with brainstate.environ.context(t=t, i=i):
                return gen.update()

        return np.asarray(transform.for_loop(step, times, idx))


def pooled_isis(spk, dt=DT):
    """Inter-spike intervals (ms) pooled across all output channels.

    Parameters
    ----------
    spk : numpy.ndarray
        ``(n_steps, n_targets)`` binary spike matrix.
    dt : float, optional
        Resolution in ms. Default :data:`DT`.

    Returns
    -------
    numpy.ndarray
        1-D array of ISIs in ms (empty if no channel has ≥ 2 spikes).
    """
    spk = np.asarray(spk)
    isis = []
    for col in spk.T:
        idx = np.nonzero(col > 0)[0]
        if idx.size > 1:
            isis.append(np.diff(idx) * dt)
    return np.concatenate(isis) if isis else np.array([])


def isi_cv(isis):
    """Coefficient of variation ``std/mean`` of an ISI array (NaN if empty)."""
    isis = np.asarray(isis)
    if isis.size == 0:
        return float("nan")
    return float(np.std(isis) / np.mean(isis))


def population_psth(spk, dt=DT, bin_ms=PST_BIN):
    """Population PSTH in Hz (sum over channels, bin, normalize) — see Poisson demo.

    Parameters
    ----------
    spk : numpy.ndarray
        ``(n_steps, n_targets)`` binary spike matrix.
    dt, bin_ms : float, optional
        Resolution and bin width in ms.

    Returns
    -------
    centers_ms : numpy.ndarray
        Bin-center times in ms.
    psth_hz : numpy.ndarray
        Population rate per bin in Hz.
    """
    spk = np.asarray(spk)
    n_steps, n = spk.shape
    per_bin = int(round(bin_ms / dt))
    n_bins = n_steps // per_bin
    counts = spk[:n_bins * per_bin].sum(axis=1).reshape(n_bins, per_bin).sum(axis=1)
    centers = (np.arange(n_bins) + 0.5) * bin_ms
    return centers, counts / (n * bin_ms / 1000.0)


def cv_by_order(seed=0, orders=ORDERS, simtime=SIMTIME, dt=DT, n_targets=N_TARGETS):
    """Stationary (amplitude=0) ISI CV for each gamma order.

    Parameters
    ----------
    seed : int, optional
        PRNG seed. Default ``0``.
    orders : sequence of int, optional
        Gamma orders to sweep. Default :data:`ORDERS`.
    simtime, dt, n_targets : optional
        See :func:`run_spikes`.

    Returns
    -------
    list of float
        Measured ISI CV per order (aligned with ``orders``).
    """
    out = []
    for m in orders:
        spk = run_spikes(seed=seed, dt=dt, simtime=simtime, order=m, amplitude=0.0,
                         individual=True, n_targets=n_targets)
        out.append(isi_cv(pooled_isis(spk, dt)))
    return out


def main():
    print("sinusoidal_gamma_generator (brainpy.state, eager for_loop drive)")
    print(f"  gamma regularization: CV -> 1/sqrt(m), dc={RATE:.0f} Hz, "
          f"{N_TARGETS} channels, T={SIMTIME:.0f} ms")
    cvs = cv_by_order()
    for m, cv in zip(ORDERS, cvs):
        print(f"    order m={m:2d}: CV = {cv:.3f}   1/sqrt(m) = {1.0 / np.sqrt(m):.3f}")

    spk = run_spikes(seed=0, order=MODULATED_ORDER, amplitude=AMPLITUDE, individual=True)
    centers, psth = population_psth(spk)
    lam = lam_of_t(centers)
    corr = np.corrcoef(psth, lam)[0, 1]
    print(f"  [modulated m={MODULATED_ORDER}] PSTH-vs-λ corr = {corr:.3f}, "
          f"mean = {psth.mean():.1f} Hz (rate profile preserved)")


if __name__ == "__main__":
    main()
