# examples/nest_like/repeated_stimulation.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Repeated stimulation by a Poisson generator — NEST-style port.

Port of NEST's ``repeated_stimulation.py``. A ``poisson_generator`` is gated to a
fixed ``[start, stop]`` window and the identical stimulation is repeated across
trials. NEST repeats by advancing the generator's ``origin`` each trial; this
port reproduces the same windowed drive by running one trial per
``simulate`` call (the cluster-02 rebuild-per-trial idiom) — every trial starts
its kernel state fresh, so the ``[start, stop]`` window recurs identically.

The generator emits no spikes of its own into a recorder (a NEST recorder taps a
*node*), so its train is relayed 1:1 through a :func:`parrot_neuron` and captured
by a ``spike_recorder`` — the canonical ``poisson_generator → parrot_neuron →
spike_recorder`` chain. The headline quantity is the spike count inside the
active window (≈ ``rate · (stop − start)``); outside the window the train is
silent.

Run:  PYTHONPATH=. python examples/nest_like/repeated_stimulation.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import (
    Simulator, poisson_generator, parrot_neuron, spike_recorder,
)

#: Poisson drive rate (Hz).
RATE = 1000.0
#: Active-window start (ms, exclusive) — stimulation begins after this time.
T_START = 100.0
#: Active-window stop (ms, inclusive) — stimulation ends at this time.
T_STOP = 500.0
#: Duration of a single trial (ms).
TRIAL_DURATION = 1000.0
#: Number of repeated trials.
NUM_TRIALS = 5
#: Simulation resolution (ms).
DT = 0.1


def build(seed=0, rate=RATE, dt=DT, t_start=T_START, t_stop=T_STOP,
          trial_duration=TRIAL_DURATION):
    """Build one trial: ``poisson_generator → parrot_neuron → spike_recorder``.

    Parameters
    ----------
    seed : int, optional
        PRNG seed for the Poisson generator. Default ``0``.
    rate : float, optional
        Poisson drive rate in Hz. Default :data:`RATE`. ``0`` makes the trial
        silent (empty-drive edge case).
    dt : float, optional
        Simulation resolution in ms. Default :data:`DT`.
    t_start, t_stop : float, optional
        Active-window bounds in ms (``start`` exclusive, ``stop`` inclusive),
        defaults :data:`T_START` / :data:`T_STOP`.
    trial_duration : float, optional
        Trial length in ms. Default :data:`TRIAL_DURATION`.

    Returns
    -------
    sim : Simulator
        The configured single-trial simulator.
    sr : NodeView
        Spike-recorder handle tapping the parrot relay.
    trial_duration : float
        The trial length in ms (echoed for the caller).
    """
    sim = Simulator(dt=dt * u.ms)
    parrot = sim.create(parrot_neuron, 1)
    pg = sim.create(poisson_generator, 1, rate=rate * u.Hz,
                    start=t_start * u.ms, stop=t_stop * u.ms, rng_seed=seed)
    sr = sim.create(spike_recorder)
    sim.connect(pg, parrot, weight=1.0, delay=dt * u.ms)   # weight unit-gated to 1.0
    sim.connect(parrot, sr)
    return sim, sr, trial_duration


def run_trials(num_trials=NUM_TRIALS, seed=0, **kw):
    """Run ``num_trials`` repeats and return each trial's spike-step indices.

    Parameters
    ----------
    num_trials : int, optional
        Number of repeated trials. Default :data:`NUM_TRIALS`.
    seed : int, optional
        Base PRNG seed; trial ``k`` uses ``seed + k``. Default ``0``.
    **kw
        Forwarded to :func:`build` (e.g. ``rate``, ``dt``).

    Returns
    -------
    list of numpy.ndarray
        One entry per trial: the per-step relayed spike *count* (shape
        ``(n_steps,)``). The parrot relays the Poisson **multiplicity** (a step
        may carry >1 spike at high rate), so these are counts, not a binary mask
        — :func:`window_count` sums them to match NEST's ``n_events``.
    """
    trains = []
    for k in range(num_trials):
        sim, sr, _dur = build(seed=seed + k, **kw)
        res = sim.simulate(_dur * u.ms)
        trains.append(np.asarray(res.spikes(sr)).reshape(-1))
    return trains


def window_count(counts, lo_ms, hi_ms, dt=DT):
    """Sum relayed spike counts whose step time falls in ``[lo_ms, hi_ms)``.

    Sums the per-step **multiplicity** (not steps-with-a-spike): the
    ``poisson_generator`` emits Poisson counts (>1/step at high rate) and the
    parrot relays them verbatim, so the faithful active-window count matches
    NEST's multiplicity-counting ``spike_recorder.n_events``.

    Parameters
    ----------
    counts : numpy.ndarray
        Per-step spike-count array (a :func:`run_trials` entry).
    lo_ms, hi_ms : float
        Window bounds in ms (lower inclusive, upper exclusive).
    dt : float, optional
        Resolution in ms used to convert steps to time. Default :data:`DT`.

    Returns
    -------
    int
        Total number of spikes (with multiplicity) in the window.
    """
    counts = np.asarray(counts)
    t_ms = np.arange(counts.shape[0]) * dt
    return int(counts[(t_ms >= lo_ms) & (t_ms < hi_ms)].sum())


def main():
    trains = run_trials()
    print("repeated_stimulation (brainpy.state, poisson -> parrot -> recorder)")
    print(f"  rate {RATE:.0f} Hz, window ({T_START:.0f}, {T_STOP:.0f}] ms, "
          f"{NUM_TRIALS} trials")
    expected = RATE * (T_STOP - T_START) / 1000.0
    for k, tr in enumerate(trains):
        n_in = window_count(tr, T_START, T_STOP)
        n_out = window_count(tr, 0.0, T_START) + window_count(tr, T_STOP, TRIAL_DURATION)
        print(f"  trial {k}: {n_in} spikes in window (expected ~{expected:.0f}), "
              f"{n_out} outside")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        for k, tr in enumerate(trains):
            steps = np.nonzero(tr > 0)[0]
            plt.plot(steps * DT, np.full(steps.shape, k, dtype=float), "k|", ms=6)
        plt.xlabel("time (ms)"); plt.ylabel("trial"); plt.xlim(0, TRIAL_DURATION)
        plt.title("Repeated stimulation by Poisson generator")
        plt.tight_layout()
        plt.savefig("examples/nest_like/repeated_stimulation_raster.png", dpi=100)
        print("  wrote examples/nest_like/repeated_stimulation_raster.png")
    except ImportError:
        print("  (matplotlib not installed; skipping raster)")


if __name__ == "__main__":
    main()
