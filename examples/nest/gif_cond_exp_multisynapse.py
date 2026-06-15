# examples/nest/gif_cond_exp_multisynapse.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Generalized-IAF neuron with conductance multisynapse — NEST-style port.

Port of NEST's ``gif_cond_exp_multisynapse.py`` onto brainpy.state's explicit
Simulator API, driving the real ``gif_cond_exp_multisynapse`` neuron. A single
``spike_generator`` (one spike at ``t = 10 ms``) fans out to **two distinct
exponential-conductance receptors** of the same neuron via
``connect(receptor_type=k)``:

======  ========  ========  =========  ===============================
port k  delay ms  weight    tau_syn    E_rev (mV)
======  ========  ========  =========  ===============================
1       1         1.0 nS    4.0 ms     0    (excitatory, fast)
2       30        5.0 nS    8.0 ms     -85  (inhibitory, slow, stronger)
======  ========  ========  =========  ===============================

Like ``aeif_cond_beta_multisynapse`` this rides the *blob* receptor bridge (the
Simulator gathers one ``(N, n_receptors)`` deposit and passes it as ``w_by_rec``);
``gif`` additionally keeps its legacy key-split deposit path for back-compat. The
single presynaptic spike stays sub-threshold, so the generalized-IAF neuron's
stochastic escape-rate spiking never triggers and ``V_m`` is deterministic.

The per-port conductances ``g_1``/``g_2`` and ``V_m`` are tapped with a
``multimeter``. (NEST's ``E_rev``/``tau_syn`` are passed as **bare float**
sequences, not unit quantities, matching the model's constructor.)

Run:  python examples/nest/gif_cond_exp_multisynapse.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import (Simulator, gif_cond_exp_multisynapse,
                           spike_generator, multimeter)

#: NEST ``gif_cond_exp_multisynapse`` params set by the example (2 receptors;
#: every other parameter is left at the shared NEST/brainpy default, and there is
#: no spike-frequency / spike-triggered adaptation). ``E_rev``/``tau_syn`` are
#: bare floats (mV / ms), as the constructor expects.
MODEL_PARAMS = dict(E_rev=(0.0, -85.0), tau_syn=(4.0, 8.0))

SPIKE_TIME = 10.0               # ms; single presynaptic spike
DELAYS = (1.0, 30.0)            # ms; per-receptor transmission delay
WEIGHTS = (1.0, 5.0)            # nS; per-receptor weight
RECORD_FROM = ('V_m', 'g_1', 'g_2')


def build(ports=(1, 2), dt=0.1, simtime=100.0):
    """Build a 2-receptor ``gif`` driven by one spike fanned to ``ports``.

    Parameters
    ----------
    ports : sequence of int, optional
        Receptor ports (1-indexed) to drive. Default both. Pass a single port
        (e.g. ``(2,)``) to isolate one receptor's response.
    dt : float, optional
        Resolution in ms; the multimeter ``interval`` matches it. Default ``0.1``.
    simtime : float, optional
        Horizon in ms (returned for the caller). Default ``100.0``.

    Returns
    -------
    sim : Simulator
    mm : NodeView
        Multimeter handle (``res.trace(mm, 'V_m')`` / ``'g_1'`` / ``'g_2'``).
    neuron : NodeView
    simtime : float
    """
    sim = Simulator(dt=dt * u.ms)
    neuron = sim.create(gif_cond_exp_multisynapse, 1, params=MODEL_PARAMS)
    spike = sim.create(spike_generator, spike_times=np.asarray([SPIKE_TIME]) * u.ms)
    mm = sim.create(multimeter, record_from=list(RECORD_FROM), interval=dt * u.ms)
    for k in ports:
        sim.connect(spike, neuron, receptor_type=int(k),
                    weight=WEIGHTS[k - 1] * u.nS, delay=DELAYS[k - 1] * u.ms)
    sim.connect(mm, neuron)              # reversed: the multimeter observes the neuron
    return sim, mm, neuron, simtime


def run_traces(ports=(1, 2), dt=0.1, simtime=100.0):
    """Simulate once and return ``V_m`` plus the two per-port conductances.

    Parameters
    ----------
    ports : sequence of int, optional
        Receptor ports to drive. Default both.
    dt : float, optional
        Resolution in ms. Default ``0.1``.
    simtime : float, optional
        Horizon in ms. Default ``100.0``.

    Returns
    -------
    dict
        ``{'times': (T,) ms, 'V_m': (T,) mV, 'g_1'/'g_2': (T,) nS}``.
    """
    sim, mm, _neuron, _t = build(ports=ports, dt=dt, simtime=simtime)
    res = sim.simulate(simtime * u.ms)
    out = {'times': np.asarray(u.get_mantissa(res.times / u.ms)).reshape(-1)}
    out['V_m'] = np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV)).reshape(-1)
    for k in (1, 2):
        out[f'g_{k}'] = np.asarray(
            u.get_mantissa(res.trace(mm, f'g_{k}') / u.nS)).reshape(-1)
    return out


def main():
    print("gif_cond_exp_multisynapse (brainpy.state) — 2-receptor PSPs")
    tr = run_traces(dt=0.1, simtime=100.0)
    for k, d, e in zip((1, 2), DELAYS, MODEL_PARAMS['E_rev']):
        gk = tr[f'g_{k}']
        print(f"  port {k} (delay {d:4.0f} ms, E_rev {e:+.0f}): peak g = {gk.max():.4f} nS")
    print(f"  V_m range: {tr['V_m'].min():.3f} .. {tr['V_m'].max():.3f} mV")

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax1.plot(tr['times'], tr['V_m'], 'k')
        ax1.set_ylabel("V_m (mV)")
        ax1.set_title("gif_cond_exp_multisynapse — excitatory + inhibitory PSPs")
        for k in (1, 2):
            ax2.plot(tr['times'], tr[f'g_{k}'], label=f"g_{k}")
        ax2.set_xlabel("time (ms)"); ax2.set_ylabel("g (nS)"); ax2.legend()
        fig.tight_layout()
        fig.savefig("examples/nest/gif_cond_exp_multisynapse.png", dpi=100)
        print("  wrote examples/nest/gif_cond_exp_multisynapse.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
