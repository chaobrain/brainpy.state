# examples/nest_like/brunel_exp_multisynapse.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel (2000) network with exponential multi-receptor synapses — NEST-style port.

Port of NEST's ``brunel_exp_multisynapse_nest.py`` onto brainpy.state's explicit
Simulator API, driving the real ``iaf_psc_exp_multisynapse`` neuron. Each neuron
exposes ``nr_ports`` receptor ports whose synaptic time constants span 0.1–1.09 ms;
every connection (noise and recurrent) is routed to a uniformly-drawn port via
``receptor_type='uniform'``, so the PSP time constants are uniformly distributed.

Run:  python examples/nest_like/brunel_exp_multisynapse.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u
import braintools

from brainpy.state import (
    Simulator, fixed_indegree, all_to_all,
    iaf_psc_exp_multisynapse, poisson_generator, spike_recorder,
)


def build(order=2500, simtime=1000.0, comm='sparse', n_rec=50, seed=0):
    dt, delay = 0.1, 1.5
    g, eta, epsilon = 5.0, 2.0, 0.1
    NE, NI = 4 * order, 1 * order
    CE, CI = int(epsilon * NE), int(epsilon * NI)
    N_rec = n_rec
    tauMem, CMem, theta, tref = 20.0, 1.0, 20.0, 2.0
    nr_ports = 100
    tau_syn = np.array([0.1 + 0.01 * i for i in range(nr_ports)])  # 0.1 .. 1.09 ms
    s0 = 1 + 4 * int(seed)        # distinct connection seeds per realization

    J = 0.1                       # pA, postsynaptic amplitude (no ComputePSPnorm)
    J_ex = J
    J_in = -g * J_ex
    nu_th = theta / (J * CE * tauMem)
    p_rate = 1000.0 * (eta * nu_th) * CE

    npar = dict(C_m=CMem * u.pF, tau_m=tauMem * u.ms, t_ref=tref * u.ms,
                E_L=0. * u.mV, V_reset=0. * u.mV, V_th=theta * u.mV,
                tau_syn=tau_syn * u.ms,
                V_initializer=braintools.init.Constant(0. * u.mV))

    sim = Simulator(dt=dt * u.ms)
    ne = sim.create(iaf_psc_exp_multisynapse, NE, params=npar)
    ni = sim.create(iaf_psc_exp_multisynapse, NI, params=npar)
    noise = sim.create(poisson_generator, rate=p_rate * u.Hz)
    esr = sim.create(spike_recorder)
    isr = sim.create(spike_recorder)

    sim.connect(noise, ne, weight=J_ex * u.pA, delay=delay * u.ms,
                rule=all_to_all, receptor_type='uniform', seed=s0 + 100)
    sim.connect(noise, ni, weight=J_ex * u.pA, delay=delay * u.ms,
                rule=all_to_all, receptor_type='uniform', seed=s0 + 200)
    sim.connect(ne, ne + ni, weight=J_ex * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(CE), comm=comm, receptor_type='uniform',
                allow_multapses=True, seed=s0)
    sim.connect(ni, ne + ni, weight=J_in * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(CI), comm=comm, receptor_type='uniform',
                allow_multapses=True, seed=s0 + 1)
    sim.connect(ne[:N_rec], esr)
    sim.connect(ni[:N_rec], isr)
    return sim, esr, isr, N_rec, simtime


def main():
    order = 2500
    print(f"Brunel network: order={order} -> {4 * order} exc + {order} inh neurons, "
          f"100 receptor ports, sparse comm.")
    print("  Building (fixed-indegree sampling is O(N); ~1-2 min at this size)...")
    sim, esr, isr, N_rec, simtime = build(order=order)
    res = sim.simulate(simtime * u.ms)
    erate = res.rate(esr.segments[0].population)
    irate = res.rate(isr.segments[0].population)
    print("Brunel network (brainpy.state, exp multi-receptor synapses)")
    print(f"  Excitatory rate : {erate:.2f} spks/s")
    print(f"  Inhibitory rate : {irate:.2f} spks/s")

    try:
        import matplotlib.pyplot as plt
        spk = np.asarray(res.spikes(esr.segments[0].population))   # (T, N_rec)
        ts, ids = np.nonzero(spk > 0)
        plt.figure(figsize=(8, 4))
        plt.scatter(ts * 0.1, ids, s=1.0, color="k")
        plt.xlabel("time (ms)"); plt.ylabel("exc neuron")
        plt.title("Brunel network — excitatory raster (exp multisynapse)")
        plt.tight_layout()
        plt.savefig("examples/nest_like/brunel_exp_multisynapse_raster.png", dpi=100)
        print("  wrote examples/nest_like/brunel_exp_multisynapse_raster.png")
    except ImportError:
        print("  (matplotlib not installed; skipping raster)")


if __name__ == "__main__":
    main()
