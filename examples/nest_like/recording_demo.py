# examples/nest_like/recording_demo.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Recording-API tour — NEST-style port.

Port of NEST's ``recording_demo.py``. The upstream loops over recording
*backends* (``ascii`` file vs ``memory``) and time *formats* (``time_in_steps``
True/False) to show how to select a backend and read the data back. The network
is deliberately trivial: a single ``poisson_generator`` stimulates one
``iaf_psc_exp`` so there is some spike data to record.

brainpy.state records **in memory** only — there is no file-backed backend yet
(``devices-gap.md`` P2) — so the ``record_to`` axis of the tour collapses to the
in-memory equivalent: a ``spike_recorder`` taps the neuron's per-step spikes and
the trace is read back with ``res.spikes`` / ``res.rate`` / ``res.n_events``. The
``time_in_steps`` axis *is* reproduced post-hoc by :func:`read_spikes`, which
returns the recorded spike times either as integer step indices
(``time_in_steps=True``) or in ms (``time_in_steps=False``). A ``multimeter``
additionally records the membrane potential, demonstrating analog recording
alongside the spike recorder.

The upstream's 1 MHz drive saturates the neuron (it fires every refractory
period), so it produces a steady stream of spikes; this is kept as the default
``RATE``. Because the rate is refractory-limited it is effectively deterministic
and matches live NEST exactly, while the Poisson drive itself remains
PRNG-divergent (parity is therefore distributional; see the parity test).

Run:  python examples/nest_like/recording_demo.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy_state import (Simulator, iaf_psc_exp, poisson_generator,
                           spike_recorder, multimeter)

#: Poisson drive rate (Hz) — the upstream's saturating 1 MHz.
RATE = 1_000_000.0
#: Synaptic weight (pA) of the generator -> neuron projection.
WEIGHT = 10.0
#: Homogeneous synaptic delay (ms).
DELAY = 1.0


def build(seed=0, dt=0.1, simtime=30.0, rate=RATE, weight=WEIGHT, delay=DELAY):
    """Build the recording-tour network (Poisson -> neuron -> recorders).

    Parameters
    ----------
    seed : int, optional
        PRNG seed for the Poisson generator. Default ``0``.
    dt : float, optional
        Simulation resolution in ms. Default ``0.1``. The multimeter's
        ``interval`` is set to ``dt`` so its sample grid matches the run.
    simtime : float, optional
        Intended simulation horizon in ms (returned for the caller). Default
        ``30.0`` (the upstream tour value).
    rate : float, optional
        Poisson drive rate in Hz. Default :data:`RATE`.
    weight : float, optional
        Synaptic weight in pA. Default :data:`WEIGHT`.
    delay : float, optional
        Synaptic delay in ms. Default :data:`DELAY`.

    Returns
    -------
    sim : Simulator
        The configured simulator.
    sr : NodeView
        Spike-recorder handle (``res.spikes(sr)`` / ``res.rate(sr)`` /
        ``res.n_events(sr)``).
    mm : NodeView
        Multimeter handle (``res.trace(mm, 'V_m')``).
    neuron : NodeView
        The neuron population handle.
    simtime : float
        The simulation horizon in ms.
    """
    sim = Simulator(dt=dt * u.ms)
    neuron = sim.create(iaf_psc_exp, 1)
    pg = sim.create(poisson_generator, 1, rate=rate * u.Hz, rng_seed=seed)
    sr = sim.create(spike_recorder)
    mm = sim.create(multimeter, record_from=['V_m'], interval=dt * u.ms)
    sim.connect(pg, neuron, weight=weight * u.pA, delay=delay * u.ms)
    sim.connect(neuron, sr)
    sim.connect(mm, neuron)              # reversed: the multimeter observes the neuron
    return sim, sr, mm, neuron, simtime


def read_spikes(res, sr, time_in_steps=False):
    """Read recorded spike times back — the ``time_in_steps`` half of the tour.

    Mirrors NEST's ``get_data(sr)``: the recorded events are returned either as
    integer simulation-step indices or as times in ms.

    Parameters
    ----------
    res : SimulationResult
        The result returned by :meth:`Simulator.simulate`.
    sr : NodeView
        The spike-recorder handle from :func:`build`.
    time_in_steps : bool, optional
        If ``True``, return integer step indices (NEST's ``time_in_steps``
        format); if ``False`` (default), return spike times in ms.

    Returns
    -------
    numpy.ndarray
        1-D array of spike step indices (``int``) or spike times in ms
        (``float``), one entry per recorded spike, in increasing order.
    """
    spk = np.asarray(res.spikes(sr)).reshape(-1)   # (n_steps,) for the single neuron
    steps = np.nonzero(spk > 0)[0]
    if time_in_steps:
        return steps
    t_ms = np.asarray(u.get_mantissa(res.times / u.ms))
    return t_ms[steps]


def main():
    sim, sr, mm, _neuron, simtime = build()
    res = sim.simulate(simtime * u.ms)

    print("recording_demo (brainpy.state, iaf_psc_exp, in-memory backend)")
    print(f"  Poisson drive {RATE:.0f} Hz (w={WEIGHT} pA) -> single neuron")
    print(f"  resolution: {float(u.get_mantissa(res.times[1] - res.times[0])):.1f} ms")
    print(f"  firing rate: {res.rate(sr):.2f} spks/s ; n_events: {res.n_events(sr)}")

    # The time_in_steps half of the backend tour: same data, two formats.
    for time_in_steps in (True, False):
        data = read_spikes(res, sr, time_in_steps=time_in_steps)
        unit = "steps" if time_in_steps else "ms"
        head = np.array2string(data[:8], precision=1, separator=", ")
        print(f"  spikes (time_in_steps={time_in_steps}) [{unit}]: {head} ...")

    # Analog recording alongside the spike recorder.
    v = np.asarray(u.get_mantissa(res.trace(mm, "V_m") / u.mV)).reshape(-1)
    print(f"  V_m trace: {v.shape[0]} samples, min {v.min():.2f} mV, max {v.max():.2f} mV")
    print("  (file/ascii backend unavailable: devices-gap.md P2 — memory only)")


if __name__ == "__main__":
    main()
