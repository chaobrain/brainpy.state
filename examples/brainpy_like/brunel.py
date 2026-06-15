# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel random network — flagship example for the brainpy.state Network API.

Two populations (excitatory + inhibitory), random fixed-indegree connectivity,
conductance-based synapses. Built with the imperative ``Builder`` style.

Run:
    python examples/brunel.py
"""
import brainstate
import jax.numpy as jnp
import brainunit as u

from brainpy_state import (
    Builder, LIF, Expon, COBA, FixedIndegreeProj,
)


def main():
    brainstate.environ.set(dt=0.1 * u.ms)

    N_E, N_I = 800, 200
    eps = 0.1
    K_E = int(eps * N_E)
    K_I = int(eps * N_I)

    b = Builder()
    exc = b.add('exc', LIF(N_E, tau=20*u.ms,
                           V_th=-50*u.mV, V_reset=-60*u.mV,
                           V_rest=-65*u.mV))
    inh = b.add('inh', LIF(N_I, tau=20*u.ms,
                           V_th=-50*u.mV, V_reset=-60*u.mV,
                           V_rest=-65*u.mV))

    for src, tgt, w, K in [
        (exc, exc, 0.1*u.nS, K_E),
        (exc, inh, 0.1*u.nS, K_E),
        (inh, exc, -0.5*u.nS, K_I),
        (inh, inh, -0.5*u.nS, K_I),
    ]:
        b.connect(src, tgt, rule=FixedIndegreeProj,
                  K=K, weight=w,
                  syn=Expon.desc(tgt.in_size, tau=5*u.ms),
                  out=COBA.desc(E=0*u.mV),
                  seed=42, allow_multapses=False)

    brainstate.nn.init_all_states(b)
    out = b.simulate(
        500 * u.ms,
        monitor={
            'exc_spike': lambda n: n.exc.get_spike(n.exc.V.value),
        },
    )

    spikes = out['exc_spike']  # (T, N_E)
    n_spikes = int(jnp.sum(spikes > 0))
    print(f'Brunel-style network: {N_E + N_I} neurons, 500 ms simulation')
    print(f'  excitatory spike count: {n_spikes}')

    try:
        import matplotlib.pyplot as plt
        times, neurons = (spikes > 0).nonzero()
        plt.figure(figsize=(8, 4))
        plt.scatter(times * 0.1, neurons, s=0.5, color='k')
        plt.xlabel('time (ms)')
        plt.ylabel('exc neuron index')
        plt.title('Brunel-style network — excitatory raster')
        plt.tight_layout()
        plt.savefig('examples/brunel_raster.png', dpi=100)
        print('  wrote examples/brunel_raster.png')
    except ImportError:
        print('  (matplotlib not installed; skipping raster plot)')


if __name__ == '__main__':
    main()
