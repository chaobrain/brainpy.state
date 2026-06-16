# examples/nest_like/mat_psc_exp.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Multi-timescale adaptive-threshold neurons (MAT / AMAT) — NEST-style port.

The ``mat2_psc_exp`` and ``amat2_psc_exp`` models (Kobayashi, Tsubo & Shinomoto
2009) capture spike-frequency adaptation through a *moving threshold* instead of
an adaptation current — and, crucially, **the membrane potential is never reset**.
After every spike the threshold jumps and then relaxes on two timescales, so the
neuron must climb a higher bar to fire again:

.. math::

    V_{th}(t) = \omega + V_{th,1}(t) + V_{th,2}(t) \;[+\; V_{th,v}(t)]

* ``mat2_psc_exp`` — two threshold components ``V_th_1`` (fast, ``tau_1``) and
  ``V_th_2`` (slow, ``tau_2``), each jumping by ``alpha_1`` / ``alpha_2`` per spike.
* ``amat2_psc_exp`` — adds a **voltage-dependent** component ``V_th_v`` that tracks
  a filtered ``dV_m/dt`` scaled by ``beta`` (with ``beta = 0`` it reduces to
  ``mat2_psc_exp``). This demo sets ``beta > 0`` so ``V_th_v`` is genuinely active.

Each neuron is driven by a constant current; the demo records both the membrane
potential ``V_m`` and the composite adaptive threshold ``V_th`` through a
multimeter, so you can watch ``V_m`` chase the moving threshold. The traces match
a live NEST ``mat2_psc_exp`` / ``amat2_psc_exp`` to machine precision (see
``brainpy_state/_nest/_validation/mat_psc_exp_test.py``).

Run:  PYTHONPATH=. python examples/nest_like/mat_psc_exp.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import braintools
import brainunit as u

from brainpy.state import (Simulator, mat2_psc_exp, amat2_psc_exp,
                           multimeter, spike_recorder)

V0 = -70.0         # initial membrane potential [mV]
T_SIM = 200.0      # simulation time [ms]
DT = 0.1           # resolution [ms]

# Two configurations: the classic MAT and the voltage-coupled AMAT. Each entry is
# (model class, model parameters incl. the constant I_e drive). The currents are
# chosen for a clear, modestly-adapting train; omega=-51 mV is NEST's canonical
# mat2 test value, and beta=0.2/ms switches on amat2's V_th_v component.
CONFIGS = {
    "mat2": (mat2_psc_exp, dict(omega=-51.0 * u.mV, I_e=600.0 * u.pA)),
    "amat2": (amat2_psc_exp, dict(beta=0.2 / u.ms, I_e=200.0 * u.pA)),
}


def build(config, simtime=T_SIM):
    """Build a single-neuron Simulator for one MAT/AMAT configuration.

    Parameters
    ----------
    config : {'mat2', 'amat2'}
        Which model/parameter set to instantiate.
    simtime : float, optional
        Simulation horizon in ms. Default :data:`T_SIM`.

    Returns
    -------
    sim : Simulator
    mm : NodeView
        Multimeter recording ``V_m`` and the composite threshold ``V_th``.
    sr : NodeView
        Spike recorder handle.
    simtime : float
    """
    cls, params = CONFIGS[config]
    sim = Simulator(dt=DT * u.ms)
    neuron = sim.create(cls, 1, params=dict(
        V_initializer=braintools.init.Constant(V0 * u.mV), **params))
    mm = sim.create(multimeter, record_from=["V_m", "V_th"], interval=DT * u.ms)
    sr = sim.create(spike_recorder)
    sim.connect(mm, neuron)
    sim.connect(neuron, sr)
    return sim, mm, sr, simtime


def run_traces(config, simtime=T_SIM):
    """Return ``(t_ms, v_mV, vth_mV, n_spikes)`` for one configuration."""
    sim, mm, sr, _t = build(config, simtime)
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    v = np.asarray(u.get_mantissa(res.trace(mm, "V_m") / u.mV)).reshape(-1)
    vth = np.asarray(u.get_mantissa(res.trace(mm, "V_th") / u.mV)).reshape(-1)
    return t, v, vth, int(res.n_events(sr))


def main():
    print("MAT / AMAT adaptive-threshold neurons (brainpy.state, constant drive)")
    results = {}
    for config in CONFIGS:
        t, v, vth, n = run_traces(config)
        results[config] = (t, v, vth)
        print(f"  {config:5s}: {n:3d} spikes, V_m peak {v.max():6.2f} mV, "
              f"V_th range [{vth.min():.1f}, {vth.max():.1f}] mV")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(CONFIGS), 1, figsize=(8, 6), sharex=True)
        for ax, config in zip(axes, CONFIGS):
            t, v, vth = results[config]
            ax.plot(t, v, color="k", lw=0.7, label="V_m")
            ax.plot(t, vth, color="C3", lw=0.9, label="V_th (adaptive)")
            ax.set_ylabel("mV")
            ax.set_title(config, loc="left", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (ms)")
        fig.suptitle("Multi-timescale adaptive threshold: V_m chases V_th")
        fig.tight_layout()
        fig.savefig("examples/nest_like/mat_psc_exp.png", dpi=100)
        print("  wrote examples/nest_like/mat_psc_exp.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
