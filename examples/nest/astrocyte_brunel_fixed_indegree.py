# examples/nest/astrocyte_brunel_fixed_indegree.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""A Brunel network with astrocytes, fixed-indegree wiring (NEST port).

Port of NEST's ``astrocyte_brunel_fixed_indegree``. Identical to the sibling
:mod:`astrocyte_brunel_bernoulli` except the **primary** neuron->neuron rule is
``fixed_indegree`` (every neuron receives exactly ``CE`` excitatory and ``CI``
inhibitory inputs) instead of ``pairwise_bernoulli``. The excitatory population
projects to both neurons and astrocytes through
:meth:`~brainpy.state.Simulator.tripartite_connect` (one shared primary sample feeds
the primary, ``third_in`` and ``third_out`` arms); the astrocyte third-factor rule is
``third_factor_bernoulli_with_pool`` (``pool_size=10``, ``pool_type='random'``).

All neurons live in **one** population sliced into ``ex = neurons[:N_ex]`` /
``inh = neurons[N_ex:]`` so the tripartite target ``post = neurons`` is a single
population view. The shared assembly lives in
:func:`astrocyte_brunel_bernoulli.build`; this module supplies the ``fixed_indegree``
primary rule.

Synapse / scale divergence from NEST: as in the Bernoulli sibling, the
primary/third_in arms use **static** (not ``tsodyks``) synapses, and the default
sizes are a tractable dense-friendly scale of the full demo (the rule, in-degrees,
pool and weights are the demo's) -- pass larger sizes to scale up.

Run:  python examples/nest/astrocyte_brunel_fixed_indegree.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp

from examples.nest.astrocyte_brunel_bernoulli import build


def run(sim_time=1000.0, *, dt=0.1, n_ex=400, n_in=100, n_astro=500, ce=None, ci=None,
        p_third=0.5, pool_size=10, pool_type='random', poisson_rate=2000.0, w_e=1.0,
        w_i=4.0, w_a2n=0.05, d_e=2.0, d_i=1.0, tau_syn_ex=2.0, tau_syn_in=4.0,
        IP3_0=0.4, sic_delay_steps=10, seed=0, n_rec_astro=50):
    """Build + run the fixed-indegree Brunel-with-astrocytes network.

    Parameters
    ----------
    sim_time : float, default 1000.0
        Simulation duration (ms).
    dt : float, default 0.1
        Integration step (ms).
    n_ex, n_in, n_astro : int
        Population sizes (default: a tractable dense-friendly scale of the demo).
    ce, ci : int, optional
        Excitatory / inhibitory in-degrees. Default to ``round(0.1 * n_ex)`` /
        ``round(0.1 * n_in)`` (the demo's ``CE/N_ex = CI/N_in = 0.1`` ratio).
    p_third, pool_size, pool_type
        Astrocyte third-factor rule (demo values ``0.5``, ``10``, ``'random'``).
    poisson_rate, w_e, w_i, w_a2n, d_e, d_i, tau_syn_ex, tau_syn_in, IP3_0, sic_delay_steps
        Background / synapse / neuron / astrocyte parameters (demo values).
    seed : int, default 0
        Base PRNG seed (Poisson drive + connectivity draws).
    n_rec_astro : int, default 50
        Number of astrocytes recorded for IP3 / Ca.

    Returns
    -------
    dict
        ``{'times', 'rate', 'e_rate', 'i_rate', 'IP3', 'Ca', 'I_SIC'}`` -- as in
        :func:`astrocyte_brunel_bernoulli.run`.
    """
    brainstate.random.seed(seed)
    brainstate.environ.set(dt=dt * u.ms)
    sim = bp.Simulator(dt=dt * u.ms)

    ce = int(round(0.1 * n_ex)) if ce is None else int(ce)
    ci = int(round(0.1 * n_in)) if ci is None else int(ci)

    def primary_rule(role):
        return bp.fixed_indegree(ce if role == 'ex' else ci)

    net = build(
        sim, primary_rule=primary_rule, n_ex=n_ex, n_in=n_in, n_astro=n_astro,
        p_third=p_third, pool_size=pool_size, pool_type=pool_type,
        poisson_rate=poisson_rate, w_e=w_e, w_i=w_i, w_a2n=w_a2n, d_e=d_e, d_i=d_i,
        tau_syn_ex=tau_syn_ex, tau_syn_in=tau_syn_in, IP3_0=IP3_0,
        sic_delay_steps=sic_delay_steps, seed=seed)
    neurons, ex, inh, astro = net['neurons'], net['ex'], net['inh'], net['astro']

    esr = sim.create(bp.spike_recorder); sim.connect(ex, esr)
    isr = sim.create(bp.spike_recorder); sim.connect(inh, isr)
    mm_a = sim.create(bp.multimeter, record_from=['IP3', 'Ca'])
    sim.connect(mm_a, astro[:n_rec_astro])
    mm_n = sim.create(bp.multimeter, record_from=['I_SIC'])
    sim.connect(mm_n, neurons[:n_rec_astro])

    res = sim.simulate(sim_time * u.ms)
    e_rate = float(res.rate(esr.segments[0].population))
    i_rate = float(res.rate(isr.segments[0].population))
    rate = (e_rate * n_ex + i_rate * n_in) / (n_ex + n_in)
    return {'times': res.times, 'rate': rate, 'e_rate': e_rate, 'i_rate': i_rate,
            'IP3': res.trace(mm_a, 'IP3'), 'Ca': res.trace(mm_a, 'Ca'),
            'I_SIC': res.trace(mm_n, 'I_SIC')}


def main():                                            # pragma: no cover - demo driver
    """Run the fixed-indegree Brunel-with-astrocytes demo and report rates / dynamics."""
    out = run()
    ip3 = np.asarray(u.get_mantissa(out['IP3']))
    isic = np.asarray(u.get_mantissa(out['I_SIC']))
    print(f"astrocyte_brunel_fixed_indegree: e_rate={out['e_rate']:.2f} Hz, "
          f"i_rate={out['i_rate']:.2f} Hz, peak IP3={ip3.max():.3f} µM, "
          f"peak I_SIC={isic.max():.3f} pA")
    try:
        import matplotlib.pyplot as plt
        t = np.asarray(u.get_mantissa(out['times'])).reshape(-1)
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6.4, 4.8), dpi=100)
        ax[0].plot(t, ip3.mean(axis=1)); ax[0].set_ylabel(r"Mean [IP$_3$] ($\mu$M)")
        ax[0].set_title("Astrocytes")
        ax[1].plot(t, isic.mean(axis=1)); ax[1].set_ylabel("Mean SIC (pA)")
        ax[1].set_xlabel("Time (ms)"); ax[1].set_title("Neurons")
        plt.tight_layout()
        plt.savefig("examples/nest/astrocyte_brunel_fixed_indegree.png", dpi=100)
        print("  wrote examples/nest/astrocyte_brunel_fixed_indegree.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
