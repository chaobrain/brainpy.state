# examples/nest/aeif_cond_beta_multisynapse.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""AdEx neuron with multiple beta-function receptors — NEST-style port.

Port of NEST's ``aeif_cond_beta_multisynapse.py`` onto brainpy.state's explicit
Simulator API, driving the real ``aeif_cond_beta_multisynapse`` neuron. A single
``spike_generator`` (one spike at ``t = 10 ms``) fans out to **four distinct
conductance receptors** of the same neuron, each reached through
``connect(receptor_type=k)`` with its own transmission delay so the four
post-synaptic responses are staggered in time:

======  ========  =========  =========  ============================
port k  delay ms  tau_rise   tau_decay  E_rev (mV)
======  ========  =========  =========  ============================
1       1         10.0       50.0       0    (excitatory, slow)
2       300       10.0       20.0       0    (excitatory)
3       500        1.0       20.0       0    (excitatory, fast rise)
4       700        1.0       20.0       -85  (inhibitory)
======  ========  =========  =========  ============================

This is the first demo to exercise the multi-receptor routing seam end to end:
``receptor_type=k`` deposits each input into exactly port ``k`` (the ``aeif`` model
rides the *blob* bridge — the Simulator gathers one ``(N, n_receptors)`` deposit
and passes it as ``w_by_rec``). The per-port conductances ``g_1..g_4`` and the
membrane potential ``V_m`` are tapped with a ``multimeter``.

Run:  python examples/nest/aeif_cond_beta_multisynapse.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import (Simulator, aeif_cond_beta_multisynapse,
                           spike_generator, multimeter)

#: NEST ``aeif_cond_beta_multisynapse`` params set by the example (4 receptors;
#: every other parameter is left at the shared NEST/brainpy default).
MODEL_PARAMS = dict(
    V_peak=0.0 * u.mV, a=4.0 * u.nS, b=80.5 * u.pA,
    E_rev=(0.0, 0.0, 0.0, -85.0) * u.mV,
    tau_decay=(50.0, 20.0, 20.0, 20.0) * u.ms,
    tau_rise=(10.0, 10.0, 1.0, 1.0) * u.ms,
)

SPIKE_TIME = 10.0                       # ms; single presynaptic spike
DELAYS = (1.0, 300.0, 500.0, 700.0)     # ms; per-receptor transmission delay
WEIGHTS = (1.0, 1.0, 1.0, 1.0)          # nS; per-receptor weight
RECORD_FROM = ('V_m', 'g_1', 'g_2', 'g_3', 'g_4')


def build(ports=(1, 2, 3, 4), dt=0.1, simtime=1000.0):
    """Build a 4-receptor ``aeif`` driven by one spike fanned to ``ports``.

    Parameters
    ----------
    ports : sequence of int, optional
        Receptor ports (1-indexed) to drive. Default all four. Pass a single
        port (e.g. ``(3,)``) to isolate one receptor's response.
    dt : float, optional
        Resolution in ms; the multimeter ``interval`` matches it. Default ``0.1``.
    simtime : float, optional
        Horizon in ms (returned for the caller). Default ``1000.0``.

    Returns
    -------
    sim : Simulator
    mm : NodeView
        Multimeter handle (``res.trace(mm, 'V_m')`` / ``'g_1'`` …).
    neuron : NodeView
    simtime : float
    """
    sim = Simulator(dt=dt * u.ms)
    neuron = sim.create(aeif_cond_beta_multisynapse, 1, params=MODEL_PARAMS)
    spike = sim.create(spike_generator, spike_times=np.asarray([SPIKE_TIME]) * u.ms)
    mm = sim.create(multimeter, record_from=list(RECORD_FROM), interval=dt * u.ms)
    for k in ports:
        sim.connect(spike, neuron, receptor_type=int(k),
                    weight=WEIGHTS[k - 1] * u.nS, delay=DELAYS[k - 1] * u.ms)
    sim.connect(mm, neuron)              # reversed: the multimeter observes the neuron
    return sim, mm, neuron, simtime


def run_traces(ports=(1, 2, 3, 4), dt=0.1, simtime=1000.0):
    """Simulate once and return ``V_m`` plus the four per-port conductances.

    Parameters
    ----------
    ports : sequence of int, optional
        Receptor ports to drive. Default all four.
    dt : float, optional
        Resolution in ms. Default ``0.1``.
    simtime : float, optional
        Horizon in ms. Default ``1000.0``.

    Returns
    -------
    dict
        ``{'times': (T,) ms, 'V_m': (T,) mV, 'g_1'..'g_4': (T,) nS}``.
    """
    sim, mm, _neuron, _t = build(ports=ports, dt=dt, simtime=simtime)
    res = sim.simulate(simtime * u.ms)
    out = {'times': np.asarray(u.get_mantissa(res.times / u.ms)).reshape(-1)}
    out['V_m'] = np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV)).reshape(-1)
    for k in (1, 2, 3, 4):
        out[f'g_{k}'] = np.asarray(
            u.get_mantissa(res.trace(mm, f'g_{k}') / u.nS)).reshape(-1)
    return out


def main():
    print("aeif_cond_beta_multisynapse (brainpy.state) — 4-receptor PSPs")
    tr = run_traces(dt=0.1, simtime=1000.0)
    for k, d in zip((1, 2, 3, 4), DELAYS):
        gk = tr[f'g_{k}']
        print(f"  port {k} (delay {d:5.0f} ms): peak g = {gk.max():.4f} nS")
    print(f"  V_m range: {tr['V_m'].min():.3f} .. {tr['V_m'].max():.3f} mV")

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax1.plot(tr['times'], tr['V_m'], 'k')
        ax1.set_ylabel("V_m (mV)")
        ax1.set_title("aeif_cond_beta_multisynapse — staggered multi-receptor PSPs")
        for k in (1, 2, 3, 4):
            ax2.plot(tr['times'], tr[f'g_{k}'], label=f"g_{k}")
        ax2.set_xlabel("time (ms)"); ax2.set_ylabel("g (nS)"); ax2.legend()
        fig.tight_layout()
        fig.savefig("examples/nest/aeif_cond_beta_multisynapse.png", dpi=100)
        print("  wrote examples/nest/aeif_cond_beta_multisynapse.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
