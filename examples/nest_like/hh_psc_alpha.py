# examples/nest_like/hh_psc_alpha.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Hodgkin–Huxley neuron (``hh_psc_alpha``) F–I curve — NEST-style port.

Port of NEST's ``hh_psc_alpha.py`` onto brainpy.state's explicit Simulator API.
A single ``hh_psc_alpha`` neuron is driven by a constant bias current ``I_e`` (the
NEST-faithful drive for this demo); the membrane potential and the three gating
variables are recorded with a ``multimeter``, and the spike count is recorded with
a ``spike_recorder``. Sweeping ``I_e`` over a grid (rebuild-per-amplitude) and
counting spikes per second after a warm-up gives the neuron's F–I (frequency vs
input-current) curve.

Recordables follow NEST names — ``V_m`` plus the gating variables ``Act_m`` /
``Inact_h`` / ``Act_n`` (brainpy's ``m`` / ``h`` / ``n``, resolved by the
Simulator's recordable-alias table).

Run:  python examples/nest_like/hh_psc_alpha.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import braintools
import brainunit as u

from brainpy.state import Simulator, hh_psc_alpha, multimeter, spike_recorder

#: NEST ``hh_psc_alpha`` default parameters (excluding the swept ``I_e``).
MODEL_PARAMS = dict(
    E_L=-54.402 * u.mV, C_m=100.0 * u.pF,
    g_Na=12000.0 * u.nS, g_K=3600.0 * u.nS, g_L=30.0 * u.nS,
    E_Na=50.0 * u.mV, E_K=-77.0 * u.mV, t_ref=2.0 * u.ms,
    tau_syn_ex=0.2 * u.ms, tau_syn_in=2.0 * u.ms,
    V_m_init=-65.0 * u.mV,
)

#: Recordables tapped by the multimeter (NEST names; gating via alias table).
RECORD_FROM = ('V_m', 'Act_m', 'Inact_h', 'Act_n')


def build(I_e=0.0, dt=0.1, simtime=100.0):
    """Build a single ``hh_psc_alpha`` driven by a constant bias current.

    Parameters
    ----------
    I_e : float, optional
        Constant bias current in pA. Default ``0.0`` (subthreshold relaxation
        from ``V_m_init`` toward rest).
    dt : float, optional
        Simulation resolution in ms; the multimeter ``interval`` matches it.
        Default ``0.1``.
    simtime : float, optional
        Simulation horizon in ms (returned for the caller). Default ``100.0``.

    Returns
    -------
    sim : Simulator
    mm : NodeView
        Multimeter handle (``res.trace(mm, 'V_m')`` etc.).
    sr : NodeView
        Spike-recorder handle (``res.n_events(sr)`` / ``res.rate(sr)``).
    neuron : NodeView
    simtime : float
    """
    sim = Simulator(dt=dt * u.ms)
    neuron = sim.create(hh_psc_alpha, 1, params=dict(I_e=I_e * u.pA, **MODEL_PARAMS))
    mm = sim.create(multimeter, record_from=list(RECORD_FROM), interval=dt * u.ms)
    sr = sim.create(spike_recorder)
    sim.connect(mm, neuron)              # reversed: the multimeter observes the neuron
    sim.connect(neuron, sr)
    return sim, mm, sr, neuron, simtime


def run_traces(I_e=0.0, dt=0.1, simtime=100.0):
    """Simulate once and return the recorded traces + spike count.

    Parameters
    ----------
    I_e : float, optional
        Constant bias current in pA. Default ``0.0``.
    dt : float, optional
        Resolution in ms. Default ``0.1``.
    simtime : float, optional
        Horizon in ms. Default ``100.0``.

    Returns
    -------
    dict
        ``{'times': (T,) ms, 'V_m': (T,) mV, 'Act_m'/'Inact_h'/'Act_n': (T,),
        'n_spikes': int}``.
    """
    sim, mm, sr, _neuron, _t = build(I_e=I_e, dt=dt, simtime=simtime)
    res = sim.simulate(simtime * u.ms)
    out = {'times': np.asarray(u.get_mantissa(res.times / u.ms)).reshape(-1)}
    out['V_m'] = np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV)).reshape(-1)
    for name in ('Act_m', 'Inact_h', 'Act_n'):
        out[name] = np.asarray(u.get_mantissa(res.trace(mm, name))).reshape(-1)
    out['n_spikes'] = int(res.n_events(sr))
    return out


def fi_curve(amps, dt=0.1, simtime=1000.0, warmup=200.0):
    """F–I curve: spikes/s after ``warmup`` for each bias current in ``amps``.

    Parameters
    ----------
    amps : sequence of float
        Bias currents ``I_e`` in pA.
    dt : float, optional
        Resolution in ms. Default ``0.1``.
    simtime : float, optional
        Per-amplitude horizon in ms. Default ``1000.0``.
    warmup : float, optional
        Initial transient discarded before counting, in ms. Default ``200.0``.

    Returns
    -------
    numpy.ndarray
        Firing rate in spks/s for each amplitude (counted over
        ``[warmup, simtime]``).
    """
    rates = np.zeros(len(amps))
    for i, amp in enumerate(amps):
        sim, _mm, sr, _neuron, _t = build(I_e=amp, dt=dt, simtime=simtime)
        res = sim.simulate(simtime * u.ms)
        spk = np.asarray(res.spikes(sr)).reshape(-1)
        t_ms = np.asarray(u.get_mantissa(res.times / u.ms)).reshape(-1)
        count = int(np.sum(spk[t_ms >= warmup] > 0))
        rates[i] = count * 1000.0 / (simtime - warmup)
    return rates


def main():
    print("hh_psc_alpha (brainpy.state) — subthreshold trace + F–I curve")

    # Subthreshold relaxation from V_m_init = -65 mV toward rest.
    tr = run_traces(I_e=0.0, dt=0.1, simtime=50.0)
    print(f"  I_e=0: V_m {tr['V_m'][0]:.2f} -> {tr['V_m'][-1]:.2f} mV "
          f"({tr['n_spikes']} spikes)")

    amps = np.arange(0.0, 2001.0, 200.0)
    rates = fi_curve(amps, simtime=1000.0, warmup=200.0)
    for a, r in zip(amps, rates):
        print(f"  I_e={a:7.1f} pA -> {r:6.1f} spks/s")

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(tr['times'], tr['V_m'], 'k')
        ax1.set_xlabel("time (ms)"); ax1.set_ylabel("V_m (mV)")
        ax1.set_title("hh_psc_alpha subthreshold relaxation (I_e=0)")
        ax2.plot(amps, rates, 'o-k')
        ax2.set_xlabel("I_e (pA)"); ax2.set_ylabel("rate (spks/s)")
        ax2.set_title("hh_psc_alpha F–I curve")
        fig.tight_layout()
        fig.savefig("examples/nest_like/hh_psc_alpha.png", dpi=100)
        print("  wrote examples/nest_like/hh_psc_alpha.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
