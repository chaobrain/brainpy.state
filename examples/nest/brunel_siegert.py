# examples/nest/brunel_siegert.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel (2000) balanced random network — mean-field (Siegert) port.

Port of NEST's ``brunel_siegert_nest.py``. Rather than simulating the spiking
network (see ``brunel_delta.py``), this solves the self-consistent equation for
the population-averaged excitatory/inhibitory firing rates (Hahne et al. 2017,
eqs. 27-30) by integrating the pseudo-time rate dynamics of three real
``siegert_neuron`` nodes: one per population (excitatory, inhibitory) plus a
constant driving node that replaces the Poisson background. The asymptotic rates
are the mean-field prediction for the spiking ``brunel_delta`` network.

Each population is one rate node. The diffusion coupling ``A -> B`` carries
``drift_factor = tauMem * 1e-3 * K_BA * J_BA`` and
``diffusion_factor = tauMem * 1e-3 * K_BA * J_BA**2`` (eqs. 28-29), so the drift
into a node is ``sum_A rate_A * drift_factor_A`` and the diffusion (variance) is
``sum_A rate_A * diffusion_factor_A`` — exactly NEST's ``diffusion_connection``.
Here the spiking ``Simulator`` (delta/spike events) does not apply; the three
nodes are wired by hand as a faithful pseudo-time iteration.

Run:  python examples/nest/brunel_siegert.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import saiunit as u

from brainpy_state import siegert_neuron


def run(order=2500, simtime=50.0, dt=0.1):
    """Integrate the 3-node Siegert mean-field network; return asymptotic rates.

    Returns ``(erate, irate, times, ex_hist, in_hist)`` where ``erate``/``irate``
    are the final (asymptotic) excitatory/inhibitory rates in spks/s and the
    ``*_hist`` arrays trace the pseudo-time relaxation toward the fixed point.
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

    with brainstate.environ.context(dt=dt * u.ms):
        ex = siegert_neuron(1, **npar)
        inh = siegert_neuron(1, **npar)
        drive = siegert_neuron(1, mean=p_rate)   # rate node, relaxes to p_rate
        for nrn in (ex, inh, drive):
            nrn.init_state()

        n_steps = int(round(simtime / dt))
        ex_hist = np.empty(n_steps)
        in_hist = np.empty(n_steps)
        for k in range(n_steps):
            # Read the previous step's rates (Jacobi coupling, NEST min-delay=1).
            r_ex = float(np.asarray(ex.rate.value).reshape(-1)[0])
            r_in = float(np.asarray(inh.rate.value).reshape(-1)[0])
            r_dr = float(np.asarray(drive.rate.value).reshape(-1)[0])
            mu = (r_dr * drift_factor_ext + r_ex * drift_factor_ex
                  + r_in * drift_factor_in)
            sig2 = (r_dr * diffusion_factor_ext + r_ex * diffusion_factor_ex
                    + r_in * diffusion_factor_in)
            drive.update()
            ex.update(drift_input=mu, diffusion_input=sig2)
            inh.update(drift_input=mu, diffusion_input=sig2)
            ex_hist[k] = float(np.asarray(ex.rate.value).reshape(-1)[0])
            in_hist[k] = float(np.asarray(inh.rate.value).reshape(-1)[0])

    times = (np.arange(n_steps) + 1) * dt
    return ex_hist[-1], in_hist[-1], times, ex_hist, in_hist


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
        plt.savefig("examples/nest/brunel_siegert_relaxation.png", dpi=100)
        print("  wrote examples/nest/brunel_siegert_relaxation.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
