# examples/nest_like/glif_psc_double_alpha_neuron.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Double-alpha current-based GLIF neuron — NEST-style port.

Port of NEST's ``glif_psc_double_alpha_neuron.py`` onto brainpy.state's explicit
Simulator API. ``glif_psc_double_alpha`` behaves exactly like ``glif_psc`` except
that each synaptic port's post-synaptic current is a **double** alpha function — a
fast alpha plus ``amp_slow`` times a slow alpha — giving far more control over the
*tail* of the synaptic current. Following the NEST example, all spike-generating
mechanisms are **off** (plain LIF), so the comparison is purely about synaptic
current shape; three identical excitatory inputs probe three receptor ports.

Three neurons are compared:

==========================  ===================================================
neuron                      synaptic kernel
==========================  ===================================================
``glif_psc``                single alpha, ``tau_syn = [2, 2, 2]`` ms
``glif_psc_double_alpha``   fast ``[2, 2, 2]`` + slow ``[4, 6, 8]`` ms,
  (timing variation)        ``amp_slow = [0.5, 0.5, 0.5]`` — vary the slow *τ*
``glif_psc_double_alpha``   fast ``[2, 2, 2]`` + slow ``[6, 6, 6]`` ms,
  (amplitude variation)     ``amp_slow = [0.2, 0.5, 0.8]`` — vary the slow *amp*
==========================  ===================================================

A single 20 pA excitatory spike is delivered to **receptor 1 at 10 ms**,
**receptor 2 at 110 ms**, and **receptor 3 at 210 ms** (each port has its own
fast/slow time constants), and the membrane potential and the summed synaptic
current ``I_syn`` are tapped with a ``multimeter`` over 300 ms. The inputs are
weak, so every neuron stays sub-threshold (no spikes) — exactly the regime in
which the synaptic-current shapes are meaningfully comparable.

Run:  python examples/nest_like/glif_psc_double_alpha_neuron.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy_state import (Simulator, glif_psc, glif_psc_double_alpha,
                           spike_generator, multimeter, spike_recorder)

RESOLUTION = 0.05               # ms; NEST example resolution
SIMTIME = 300.0                 # ms

#: Mechanism flags: all off (plain LIF) — the demo isolates synaptic shape.
MECH_OFF = dict(spike_dependent_threshold=False, after_spike_currents=False,
                adapting_threshold=False)

#: Single-alpha reference (glif_psc) time constants, one per receptor port.
TAU_PSC = (2.0, 2.0, 2.0)
#: Double-alpha "timing" variation: fixed slow amplitude, varied slow tau.
DA_TIMING = dict(tau_syn_fast=(2.0, 2.0, 2.0), tau_syn_slow=(4.0, 6.0, 8.0),
                 amp_slow=(0.5, 0.5, 0.5))
#: Double-alpha "amplitude" variation: fixed slow tau, varied slow amplitude.
DA_AMP = dict(tau_syn_fast=(2.0, 2.0, 2.0), tau_syn_slow=(6.0, 6.0, 6.0),
              amp_slow=(0.2, 0.5, 0.8))

#: The three compared neurons: (label, double-alpha kwargs or ``None`` for glif_psc).
CONFIGS = (
    ('glif_psc', None),
    ('double_alpha_timing', DA_TIMING),
    ('double_alpha_amp', DA_AMP),
)

#: (spike time ms, receptor port) — one 20 pA excitatory spike per port.
ESPIKES = ((10.0, 1), (110.0, 2), (210.0, 3))
ESPIKE_W = 20.0                 # pA

RECORD_FROM = ('V_m', 'I_syn')


def build(cfg, dt=RESOLUTION, simtime=SIMTIME):
    """Build one neuron (single- or double-alpha) driven on three receptor ports.

    Parameters
    ----------
    cfg : dict or None
        ``None`` selects the single-alpha ``glif_psc`` (with :data:`TAU_PSC`);
        otherwise a ``glif_psc_double_alpha`` keyword dict (``tau_syn_fast``,
        ``tau_syn_slow``, ``amp_slow``).
    dt : float, optional
        Resolution in ms; the multimeter ``interval`` matches it. Default ``0.05``.
    simtime : float, optional
        Horizon in ms (returned for the caller). Default ``300.0``.

    Returns
    -------
    sim : Simulator
    mm : NodeView
        Multimeter handle (``res.trace(mm, name)`` for each recordable).
    sr : NodeView
        Spike-recorder handle.
    simtime : float
    """
    sim = Simulator(dt=dt * u.ms)
    if cfg is None:
        neuron = sim.create(glif_psc, 1, params=dict(MECH_OFF, tau_syn=TAU_PSC))
    else:
        neuron = sim.create(glif_psc_double_alpha, 1, params=dict(MECH_OFF, **cfg))
    mm = sim.create(multimeter, record_from=list(RECORD_FROM), interval=dt * u.ms)
    sr = sim.create(spike_recorder)
    for t, rec in ESPIKES:
        g = sim.create(spike_generator, spike_times=np.asarray([t]) * u.ms)
        sim.connect(g, neuron, receptor_type=rec, weight=ESPIKE_W * u.pA, delay=dt * u.ms)
    sim.connect(mm, neuron)                        # reversed: multimeter observes neuron
    sim.connect(neuron, sr)
    return sim, mm, sr, simtime


def run_traces(cfg, dt=RESOLUTION, simtime=SIMTIME):
    """Simulate one configuration and return its V_m / I_syn traces and spike count.

    Parameters
    ----------
    cfg : dict or None
        Configuration (see :func:`build`).
    dt : float, optional
        Resolution in ms. Default ``0.05``.
    simtime : float, optional
        Horizon in ms. Default ``300.0``.

    Returns
    -------
    dict
        ``times`` (ms), ``V_m`` (mV), ``I_syn`` (pA), and ``n_spikes`` (int).
    """
    sim, mm, sr, _t = build(cfg, dt=dt, simtime=simtime)
    res = sim.simulate(simtime * u.ms)
    t_ms = np.asarray(u.get_mantissa(res.times / u.ms)).reshape(-1)
    out = {'times': t_ms}
    out['V_m'] = np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV)).reshape(-1)
    out['I_syn'] = np.asarray(u.get_mantissa(res.trace(mm, 'I_syn') / u.pA)).reshape(-1)
    spk = np.asarray(res.spikes(sr)).reshape(t_ms.size, -1)
    out['n_spikes'] = int(np.sum(spk[:, 0] > 0))
    return out


def main():
    print("glif_psc_double_alpha_neuron (brainpy.state) — single vs double alpha")
    traces = {}
    for label, cfg in CONFIGS:
        tr = run_traces(cfg)
        traces[label] = tr
        print(f"  {label:<20s}: spikes={tr['n_spikes']:2d}  "
              f"I_syn|max|={np.abs(tr['I_syn']).max():6.2f} pA  "
              f"V_m[{tr['V_m'].min():.2f}, {tr['V_m'].max():.2f}] mV")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        colors = {'glif_psc': 'b', 'double_alpha_timing': 'r', 'double_alpha_amp': 'g'}
        for label, _cfg in CONFIGS:
            tr = traces[label]
            axes[0].plot(tr['times'], tr['I_syn'], colors[label], lw=0.8, label=label)
            axes[1].plot(tr['times'], tr['V_m'], colors[label], lw=0.8, label=label)
        axes[0].set_ylabel("I_syn (pA)")
        axes[1].set_ylabel("V_m (mV)")
        axes[1].set_xlabel("time (ms)")
        axes[0].set_title("glif_psc_double_alpha — single vs double-alpha synaptic current")
        axes[0].legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        fig.savefig("examples/nest_like/glif_psc_double_alpha_neuron.png", dpi=100)
        print("  wrote examples/nest_like/glif_psc_double_alpha_neuron.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
