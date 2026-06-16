# examples/nest_like/brunel_siegert.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel (2000) balanced random network -- mean-field (Siegert) port.

Port of NEST's ``brunel_siegert_nest.py``. Rather than simulating the spiking
network (see ``brunel_delta.py``), this solves the self-consistent equation for
the population-averaged excitatory/inhibitory firing rates (Hahne et al. 2017,
eqs. 27-30) by relaxing the pseudo-time rate dynamics of three real
``siegert_neuron`` nodes: one per population (excitatory, inhibitory) plus a
constant driving node that replaces the Poisson background. The asymptotic rates
are the mean-field prediction for the spiking ``brunel_delta`` network.

Each population is one rate node. A ``diffusion_connection A -> B`` carries
``drift_factor = tau_m * 1e-3 * K_BA * J_BA`` and
``diffusion_factor = tau_m * 1e-3 * K_BA * J_BA**2`` (eqs. 28-29): the source's
rate is deposited as ``rate * drift_factor`` into B's drift (mu) channel and
``rate * diffusion_factor`` into B's diffusion (sigma^2) channel, summed over all
incoming connections -- exactly NEST's ``diffusion_connection``. The whole
relaxation runs end-to-end through the ``Simulator`` (one compiled ``for_loop``
over the rate dynamics), so there is no Python step loop.

Run:  python examples/nest_like/brunel_siegert.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(order=2500, simtime=50.0, dt=0.1):
    """Relax the 3-node Siegert mean-field network; return asymptotic rates.

    Builds three ``siegert_neuron`` nodes (excitatory, inhibitory, constant
    drive) wired by ``diffusion_connection`` and relaxes them through the
    :class:`~brainpy.state.Simulator` for ``simtime`` ms.

    Returns ``(erate, irate, times, ex_hist, in_hist)`` where ``erate``/``irate``
    are the final (asymptotic) excitatory/inhibitory rates in spks/s and the
    ``*_hist`` arrays trace the pseudo-time relaxation toward the fixed point. By
    the symmetry of the Brunel mean-field reduction -- both populations receive
    the same drift and diffusion -- ``erate`` and ``irate`` coincide.
    """
    g, eta, epsilon = 5.0, 2.0, 0.1
    NE, NI = 4 * order, 1 * order
    CE, CI = int(epsilon * NE), int(epsilon * NI)
    tauMem, theta = 20.0, 20.0

    J = 0.1                       # mV postsynaptic amplitude (as in brunel_delta)
    J_ex = J
    J_in = -g * J_ex
    pref = tauMem * 1e-3          # eqs. 28-29 prefactor (tau_m in ms -> s)
    drift_factor_ext = pref * J_ex
    drift_factor_ex = pref * CE * J_ex
    drift_factor_in = pref * CI * J_in
    diffusion_factor_ext = pref * J_ex ** 2
    diffusion_factor_ex = pref * CE * J_ex ** 2
    diffusion_factor_in = pref * CI * J_in ** 2

    nu_th = theta / (J * CE * tauMem)
    nu_ex = eta * nu_th
    p_rate = 1000.0 * nu_ex * CE

    npar = dict(tau_m=tauMem * u.ms, t_ref=2.0 * u.ms, theta=theta, V_reset=0.0)

    net = bp.Simulator(dt=dt * u.ms)
    ex = net.create(bp.siegert_neuron, 1, params=npar)
    inh = net.create(bp.siegert_neuron, 1, params=npar)
    drive = net.create(bp.siegert_neuron, 1, params=dict(mean=p_rate))

    # Six convergent diffusion_connections: each source A deposits rate*drift into
    # the {ex, inh} drift (mu) channel and rate*diffusion into their diffusion
    # (sigma^2) channel; the ex->ex and inh->inh edges are the population
    # self-coupling. Deposits sharing a target accumulate, so each node sees the
    # summed mu / sigma^2 -- exactly the NEST mean-field input.
    for src, drift, diffusion in (
            (drive, drift_factor_ext, diffusion_factor_ext),
            (ex, drift_factor_ex, diffusion_factor_ex),
            (inh, drift_factor_in, diffusion_factor_in)):
        for tgt in (ex, inh):
            net.connect(src, tgt, synapse=bp.diffusion_connection(
                drift_factor=drift, diffusion_factor=diffusion))

    mm_ex = net.create(bp.multimeter, record_from=['rate'])
    mm_in = net.create(bp.multimeter, record_from=['rate'])
    net.connect(mm_ex, ex)
    net.connect(mm_in, inh)

    res = net.simulate(simtime * u.ms)
    ex_hist = np.asarray(res.trace(mm_ex, 'rate')).reshape(-1)
    in_hist = np.asarray(res.trace(mm_in, 'rate')).reshape(-1)
    times = (np.arange(ex_hist.shape[0]) + 1) * dt
    return float(ex_hist[-1]), float(in_hist[-1]), times, ex_hist, in_hist


def main():
    order = 2500
    print(f"Brunel mean-field (Siegert): order={order} -> "
          f"{4 * order} exc + {order} inh neurons (one rate node each).")
    erate, irate, times, ex_hist, in_hist = run(order=order)
    print("Brunel network (brainpy.state, Siegert mean-field)")
    print(f"  Excitatory rate : {erate:.2f} spks/s")
    print(f"  Inhibitory rate : {irate:.2f} spks/s")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(times, ex_hist, label="excitatory")
        plt.plot(times, in_hist, "--", label="inhibitory")
        plt.xlabel("pseudo-time (ms)"); plt.ylabel("rate (spks/s)")
        plt.title("Brunel network — Siegert mean-field relaxation")
        plt.legend(); plt.tight_layout()
        plt.savefig("examples/nest_like/brunel_siegert_relaxation.png", dpi=100)
        print("  wrote examples/nest_like/brunel_siegert_relaxation.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
