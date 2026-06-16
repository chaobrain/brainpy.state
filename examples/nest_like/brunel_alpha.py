# examples/nest_like/brunel_alpha.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel (2000) random balanced network with alpha synapses — NEST-style port.

Port of NEST's ``brunel_alpha_nest.py`` onto brainpy.state's explicit Simulator
API, driving the real ``iaf_psc_alpha`` / ``poisson_generator`` / ``spike_recorder``
models. Default ``order=400`` keeps the dense event projection memory-light; pass
a larger ``order`` once the sparse comm (EventFixedNumConn) lands.

Run:  python examples/nest_like/brunel_alpha.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import scipy.special as sp
import brainunit as u
import braintools

from brainpy.state import (
    Simulator, fixed_indegree, all_to_all,
    iaf_psc_alpha, poisson_generator, spike_recorder,
)


def LambertWm1(x):
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = tauMem / tauSyn
    b = 1.0 / tauSyn - 1.0 / tauMem
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)
    return (np.exp(1.0) / (tauSyn * CMem * b)
            * ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b
               - t_max * np.exp(-t_max / tauSyn)))


def build(order=2500, simtime=1000.0, comm='sparse'):
    dt, delay = 0.1, 1.5
    g, eta, epsilon = 5.0, 2.0, 0.1
    NE, NI = 4 * order, 1 * order
    CE, CI = int(epsilon * NE), int(epsilon * NI)
    N_rec = 50
    tauSyn, tauMem, CMem, theta, tref = 0.5, 20.0, 250.0, 20.0, 2.0

    J = 0.1
    J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
    J_ex = J / J_unit
    J_in = -g * J_ex
    nu_th = (theta * CMem) / (J_ex * CE * np.exp(1) * tauMem * tauSyn)
    p_rate = 1000.0 * (eta * nu_th) * CE

    npar = dict(C_m=CMem * u.pF, tau_m=tauMem * u.ms, tau_syn_ex=tauSyn * u.ms,
                tau_syn_in=tauSyn * u.ms, t_ref=tref * u.ms, E_L=0. * u.mV,
                V_reset=0. * u.mV, V_th=theta * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV))

    sim = Simulator(dt=dt * u.ms)
    ne = sim.create(iaf_psc_alpha, NE, params=npar)
    ni = sim.create(iaf_psc_alpha, NI, params=npar)
    noise = sim.create(poisson_generator, rate=p_rate * u.Hz)
    esr = sim.create(spike_recorder)
    isr = sim.create(spike_recorder)

    sim.connect(noise, ne, weight=J_ex * u.pA, delay=delay * u.ms, rule=all_to_all)
    sim.connect(noise, ni, weight=J_ex * u.pA, delay=delay * u.ms, rule=all_to_all)
    sim.connect(ne, ne + ni, weight=J_ex * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(CE), comm=comm, allow_multapses=True, seed=1)
    sim.connect(ni, ne + ni, weight=J_in * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(CI), comm=comm, allow_multapses=True, seed=2)
    sim.connect(ne[:N_rec], esr)
    sim.connect(ni[:N_rec], isr)
    return sim, esr, isr, N_rec, simtime


def main():
    order = 2500
    print(f"Brunel network: order={order} -> {4 * order} exc + {order} inh neurons, "
          f"sparse comm.")
    print("  Building (fixed-indegree sampling is O(N); ~1-2 min at this size)...")
    sim, esr, isr, N_rec, simtime = build(order=order)
    res = sim.simulate(simtime * u.ms)
    erate = res.rate(esr.segments[0].population)
    irate = res.rate(isr.segments[0].population)
    print("Brunel network (brainpy.state, alpha synapses)")
    print(f"  Excitatory rate : {erate:.2f} spks/s")
    print(f"  Inhibitory rate : {irate:.2f} spks/s")

    try:
        import matplotlib.pyplot as plt
        spk = np.asarray(res.spikes(esr.segments[0].population))   # (T, N_rec)
        ts, ids = np.nonzero(spk > 0)
        plt.figure(figsize=(8, 4))
        plt.scatter(ts * 0.1, ids, s=1.0, color="k")
        plt.xlabel("time (ms)"); plt.ylabel("exc neuron")
        plt.title("Brunel network — excitatory raster")
        plt.tight_layout()
        plt.savefig("examples/nest_like/brunel_alpha_raster.png", dpi=100)
        print("  wrote examples/nest_like/brunel_alpha_raster.png")
    except ImportError:
        print("  (matplotlib not installed; skipping raster)")


if __name__ == "__main__":
    main()
