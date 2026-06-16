# examples/nest_like/glif_psc_neuron.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Current-based GLIF neuron, five mechanism levels — NEST-style port.

Port of NEST's ``glif_psc_neuron.py`` onto brainpy.state's explicit Simulator
API, driving the real ``glif_psc`` neuron. Unlike its conductance-based sibling
``glif_cond`` (whose synapses are alpha *conductances* integrated with RKF45),
``glif_psc`` injects alpha-shaped *post-synaptic currents* (pA) and advances the
whole system with **exact propagator matrices** — so its traces match NEST to
near machine precision. The Allen-Institute generalized-LIF model is exercised
at its **five mechanism levels**, selected by three boolean flags
``(spike_dependent_threshold, after_spike_currents, adapting_threshold)``:

========  =========================  ===========================================
level     flags (r, asc, a)          active mechanisms
========  =========================  ===========================================
lif        (F, F, F)                 plain LIF
lif_r      (T, F, F)                 + spike-dependent threshold
lif_asc    (F, T, F)                 + after-spike currents
lif_r_asc  (T, T, F)                 + both of the above
lif_r_asc_a(T, T, T)                 + voltage-adapting threshold
========  =========================  ===========================================

**Four stimulation paradigms** drive every level (each on its own seam):

1. a 400 pA **step current** over 200–500 ms — the current-input seam,
   ``connect(cg, neuron)`` (no receptor);
2. **excitatory** spikes at 10/100/150 ms → ``receptor_type=1`` (+20 pA);
3. **inhibitory** spikes at 15/99/150 ms → ``receptor_type=1`` (−20 pA) — both
   spike trains land on the *same* port, exactly as the NEST example wires them;
4. a 150 kHz **Poisson window** over 600–900 ms relayed 1:1 through a
   ``parrot_neuron`` into ``receptor_type=2`` — the canonical
   ``poisson_generator → parrot_neuron → neuron`` chain.

The two synaptic ports use ``tau_syn = [2.0, 1.0]`` ms. The membrane potential,
the injected current, the summed post-synaptic current, the (possibly adapting)
threshold and its components, and the summed after-spike current are tapped with
a ``multimeter``; spikes are captured by a ``spike_recorder``.

Run:  python examples/nest_like/glif_psc_neuron.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy_state import (Simulator, glif_psc, spike_generator,
                           step_current_generator, poisson_generator,
                           parrot_neuron, multimeter, spike_recorder)

RESOLUTION = 0.05               # ms; NEST example resolution
SIMTIME = 1000.0                # ms

#: Two current-based synaptic ports (alpha-PSC time constants, ms).
TAU_SYN = (2.0, 1.0)

#: The five GLIF mechanism levels: (label, mechanism-flag dict).
GLIF_LEVELS = (
    ('lif',         dict(spike_dependent_threshold=False, after_spike_currents=False, adapting_threshold=False)),
    ('lif_r',       dict(spike_dependent_threshold=True,  after_spike_currents=False, adapting_threshold=False)),
    ('lif_asc',     dict(spike_dependent_threshold=False, after_spike_currents=True,  adapting_threshold=False)),
    ('lif_r_asc',   dict(spike_dependent_threshold=True,  after_spike_currents=True,  adapting_threshold=False)),
    ('lif_r_asc_a', dict(spike_dependent_threshold=True,  after_spike_currents=True,  adapting_threshold=True)),
)

ESPIKE_TIMES = (10.0, 100.0, 150.0)   # ms; excitatory presynaptic spikes
ISPIKE_TIMES = (15.0, 99.0, 150.0)    # ms; inhibitory presynaptic spikes
ESPIKE_W = 20.0                       # pA into receptor 1
ISPIKE_W = -20.0                      # pA into receptor 1 (same port, negative)
STEP_AMP = 400.0                      # pA
STEP_START, STEP_STOP = 200.0, 500.0  # ms
POISSON_RATE = 150000.0               # Hz
POISSON_START, POISSON_STOP = 600.0, 900.0  # ms
PARROT_W = 1.0                        # pA into receptor 2 (NEST default weight)

RECORD_FROM = ('V_m', 'I', 'I_syn', 'threshold', 'threshold_spike',
               'threshold_voltage', 'ASCurrents_sum')


def build(mech, with_poisson=True, dt=RESOLUTION, simtime=SIMTIME):
    """Build one ``glif_psc`` neuron at mechanism level ``mech``, fully driven.

    Parameters
    ----------
    mech : dict
        The three mechanism-flag keyword arguments
        (``spike_dependent_threshold``, ``after_spike_currents``,
        ``adapting_threshold``) selecting the GLIF level. ``tau_syn`` is merged
        in from :data:`TAU_SYN`.
    with_poisson : bool, optional
        Include the 150 kHz Poisson window (600–900 ms) relayed through a
        ``parrot_neuron`` into receptor 2. Default ``True`` (the faithful demo).
        Pass ``False`` for a fully deterministic drive (used by the parity test,
        whose stochastic Poisson window cannot be compared sample-by-sample to
        NEST's independent PRNG stream).
    dt : float, optional
        Resolution in ms; the multimeter ``interval`` matches it. Default
        ``0.05``.
    simtime : float, optional
        Horizon in ms (returned for the caller). Default ``1000.0``.

    Returns
    -------
    sim : Simulator
    mm : NodeView
        Multimeter handle (``res.trace(mm, name)`` for each recordable).
    sr : NodeView
        Spike-recorder handle (``res.n_events(sr)`` / ``res.spikes(sr)``).
    simtime : float
    """
    sim = Simulator(dt=dt * u.ms)
    neuron = sim.create(glif_psc, 1, params=dict(mech, tau_syn=TAU_SYN))
    espk = sim.create(spike_generator, spike_times=np.asarray(ESPIKE_TIMES) * u.ms)
    ispk = sim.create(spike_generator, spike_times=np.asarray(ISPIKE_TIMES) * u.ms)
    cg = sim.create(step_current_generator,
                    amplitude_times=np.asarray([STEP_START]) * u.ms,
                    amplitude_values=np.asarray([STEP_AMP]) * u.pA,
                    start=STEP_START * u.ms, stop=STEP_STOP * u.ms)
    mm = sim.create(multimeter, record_from=list(RECORD_FROM), interval=dt * u.ms)
    sr = sim.create(spike_recorder)

    sim.connect(cg, neuron)                       # current-input seam (no receptor)
    sim.connect(espk, neuron, receptor_type=1, weight=ESPIKE_W * u.pA, delay=dt * u.ms)
    sim.connect(ispk, neuron, receptor_type=1, weight=ISPIKE_W * u.pA, delay=dt * u.ms)
    if with_poisson:
        pg = sim.create(poisson_generator, rate=POISSON_RATE * u.Hz,
                        start=POISSON_START * u.ms, stop=POISSON_STOP * u.ms)
        parrot = sim.create(parrot_neuron, 1)
        sim.connect(pg, parrot, weight=1.0, delay=dt * u.ms)        # weight ignored
        sim.connect(parrot, neuron, receptor_type=2, weight=PARROT_W * u.pA, delay=dt * u.ms)
    sim.connect(mm, neuron)                        # reversed: multimeter observes neuron
    sim.connect(neuron, sr)
    return sim, mm, sr, simtime


def run_traces(mech, with_poisson=True, dt=RESOLUTION, simtime=SIMTIME):
    """Simulate one mechanism level and return its traces, spikes, and metadata.

    Parameters
    ----------
    mech : dict
        Mechanism-flag keyword arguments (see :func:`build`).
    with_poisson : bool, optional
        Include the Poisson window. Default ``True``.
    dt : float, optional
        Resolution in ms. Default ``0.05``.
    simtime : float, optional
        Horizon in ms. Default ``1000.0``.

    Returns
    -------
    dict
        ``times`` (ms), ``V_m`` (mV), ``I``/``I_syn`` (pA), ``threshold`` and
        ``threshold_spike``/``threshold_voltage`` (mV, GLIF frame relative to
        ``E_L``), ``ASCurrents_sum`` (pA), ``n_spikes`` (int), and
        ``spike_times`` (ms array).
    """
    sim, mm, sr, _t = build(mech, with_poisson=with_poisson, dt=dt, simtime=simtime)
    res = sim.simulate(simtime * u.ms)
    t_ms = np.asarray(u.get_mantissa(res.times / u.ms)).reshape(-1)
    out = {'times': t_ms}
    out['V_m'] = np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV)).reshape(-1)
    # I (injected) and I_syn (summed PSC) are pA Quantities.
    for name in ('I', 'I_syn'):
        out[name] = np.asarray(u.get_mantissa(res.trace(mm, name) / u.pA)).reshape(-1)
    # threshold components are stored as bare floats (mV, relative to E_L);
    # ASCurrents_sum is a bare float (pA).
    for name in ('threshold', 'threshold_spike', 'threshold_voltage', 'ASCurrents_sum'):
        out[name] = np.asarray(u.get_mantissa(res.trace(mm, name))).reshape(-1)
    spk = np.asarray(res.spikes(sr)).reshape(t_ms.size, -1)
    out['n_spikes'] = int(np.sum(spk[:, 0] > 0))
    out['spike_times'] = t_ms[spk[:, 0] > 0]
    return out


def main():
    print("glif_psc_neuron (brainpy.state) — five GLIF mechanism levels")
    traces = {}
    for label, mech in GLIF_LEVELS:
        tr = run_traces(mech, with_poisson=True)
        traces[label] = tr
        print(f"  {label:<12s}: spikes={tr['n_spikes']:3d}  "
              f"V_m[{tr['V_m'].min():.1f}, {tr['V_m'].max():.1f}] mV  "
              f"I_syn|max|={np.abs(tr['I_syn']).max():.1f} pA  "
              f"ASC|max|={np.abs(tr['ASCurrents_sum']).max():.2f} pA")

    try:
        import matplotlib.pyplot as plt
        E_L = -78.85
        fig, axes = plt.subplots(len(GLIF_LEVELS), 1, figsize=(10, 12), sharex=True)
        for ax, (label, _mech) in zip(axes, GLIF_LEVELS):
            tr = traces[label]
            ax.plot(tr['times'], tr['V_m'], 'b', lw=0.7, label='V_m')
            ax.plot(tr['times'], tr['threshold'] + E_L, 'g--', lw=0.8, label='threshold')
            if tr['spike_times'].size:
                ax.plot(tr['spike_times'],
                        [tr['V_m'].max()] * tr['spike_times'].size, 'r.', ms=3)
            ax.set_ylabel(f"{label}\nV (mV)", fontsize=8)
            ax.legend(loc='upper right', fontsize=7)
        axes[0].set_title("glif_psc — 5 GLIF levels, 4 stimulation paradigms")
        axes[-1].set_xlabel("time (ms)")
        fig.tight_layout()
        fig.savefig("examples/nest_like/glif_psc_neuron.png", dpi=100)
        print("  wrote examples/nest_like/glif_psc_neuron.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
