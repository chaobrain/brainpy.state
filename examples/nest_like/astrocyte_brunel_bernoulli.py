# examples/nest_like/astrocyte_brunel_bernoulli.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""A Brunel network with astrocytes, Bernoulli wiring (NEST port).

Port of NEST's ``astrocyte_brunel_bernoulli``. A balanced random (Brunel) network
of :class:`~brainpy.state.aeif_cond_alpha_astro` neurons (excitatory + inhibitory)
plus a population of :class:`~brainpy.state.astrocyte_lr_1994` astrocytes. The
**excitatory** population projects to both neurons and astrocytes through
:meth:`~brainpy.state.Simulator.tripartite_connect` (one shared primary sample feeds
all three arms); the inhibitory population projects to neurons only.

* primary ``ex -> all`` -- ``pairwise_bernoulli(p_primary)`` conductance EPSP
  (receptor 1 -> ``g_ex``, ``w_e`` nS).
* ``third_in`` ``ex -> astro`` -- a delta drive raising astrocytic IP3.
* ``third_out`` ``astro -> all`` -- a :class:`~brainpy.state.sic_connection` feeding
  the slow inward current ``I_SIC`` back into the neurons; the astrocyte pool of each
  target neuron is ``pool_size`` astrocytes drawn at random (``pool_type='random'``).
* inhibitory ``inh -> all`` -- ``pairwise_bernoulli(p_primary)`` (receptor 2 ->
  ``g_in``, positive ``w_i`` nS; inhibition arises from the reversal potential), no
  astrocyte pairing.
* background -- an independent ``poisson_generator`` per neuron (receptor 1).

All neurons live in **one** population sliced into ``ex = neurons[:N_ex]`` /
``inh = neurons[N_ex:]`` so the tripartite target ``post = neurons`` is a single
population view (``tripartite_connect`` operates on single-population views). The
sibling :mod:`astrocyte_brunel_fixed_indegree` is identical except its primary rule
is ``fixed_indegree``.

Synapse divergence from NEST: NEST's demo uses ``tsodyks_synapse`` for the
primary/third_in arms; this port uses **static** synapses (the 15d-validated SIC
loop). Scale divergence: the full demo (8000+2000 neurons, 10000 astrocytes) needs a
sparse backend; this port defaults to a smaller, dense-friendly scale (the rule,
probabilities, pool and weights are the demo's) -- pass larger sizes to scale up.

Run:  python examples/nest_like/astrocyte_brunel_bernoulli.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp


def build(sim, *, primary_rule, n_ex, n_in, n_astro, p_third, pool_size,
          pool_type, poisson_rate, w_e, w_i, w_a2n, d_e, d_i, tau_syn_ex, tau_syn_in,
          IP3_0, sic_delay_steps, seed):
    """Assemble the Brunel-with-astrocytes network into ``sim``.

    The single shared parameterized builder behind both the ``bernoulli`` and the
    ``fixed_indegree`` ports; they differ only in ``primary_rule`` (the excitatory
    and inhibitory primary connection rule).

    Parameters
    ----------
    sim : Simulator
        The simulator to populate.
    primary_rule : callable
        ``primary_rule('ex')`` / ``primary_rule('in')`` must return the excitatory /
        inhibitory primary :class:`~brainpy.state.ConnRule` (e.g.
        ``pairwise_bernoulli(p)`` or ``fixed_indegree(C)``).
    n_ex, n_in, n_astro : int
        Excitatory / inhibitory neuron and astrocyte population sizes.
    p_third : float
        Probability of pairing each realized excitatory primary edge with an astrocyte.
    pool_size : int
        Astrocytes per target neuron pool.
    pool_type : {'random', 'block'}
        Pool-assignment scheme.
    poisson_rate : float
        Per-neuron background Poisson rate (Hz).
    w_e, w_i, w_a2n : float
        Excitatory (nS, ``g_ex``), inhibitory (nS magnitude, ``g_in``) and SIC weights.
    d_e, d_i : float
        Excitatory / inhibitory delays (ms).
    tau_syn_ex, tau_syn_in : float
        Synaptic time constants (ms).
    IP3_0 : float
        Initial astrocytic IP3 (µM).
    sic_delay_steps : int
        SIC delivery delay in steps.
    seed : int
        Base seed for the tripartite + inhibitory connectivity draws.

    Returns
    -------
    dict
        ``{'neurons', 'ex', 'inh', 'astro', 'primary', 'third_in', 'third_out',
        'inhibitory'}`` -- the neuron population, its excitatory / inhibitory slices,
        the astrocyte population, and the four realized projections (the tripartite
        ``primary`` / ``third_in`` / ``third_out`` arms and the ``inhibitory`` arm).
    """
    npar = {'tau_syn_ex': tau_syn_ex * u.ms, 'tau_syn_in': tau_syn_in * u.ms}
    neurons = sim.create(bp.aeif_cond_alpha_astro, n_ex + n_in, params=npar)
    astro = sim.create(bp.astrocyte_lr_1994, n_astro, params={'IP3_initializer': IP3_0})
    ex, inh = neurons[:n_ex], neurons[n_ex:]

    # Independent background Poisson drive (one train per neuron, receptor 1 -> g_ex).
    pg = sim.create(bp.poisson_generator, n_ex + n_in, rate=poisson_rate * u.Hz)
    sim.connect(pg, neurons, rule=bp.one_to_one, weight=w_e * u.nS,
                delay=d_e * u.ms, receptor_type=1)

    # Excitatory primary + astrocyte pairing (one shared sample across the 3 arms).
    primary, third_in, third_out = sim.tripartite_connect(
        ex, neurons, astro,
        conn_spec=primary_rule('ex'),
        third_factor_conn_spec=bp.third_factor_bernoulli_with_pool(
            p=p_third, pool_size=pool_size, pool_type=pool_type),
        syn_specs={
            'primary': {'weight': w_e * u.nS, 'delay': d_e * u.ms, 'receptor_type': 1},
            'third_in': {'weight': w_e, 'delay': d_e * u.ms},
            'third_out': {'synapse': bp.sic_connection(weight=w_a2n,
                                                       delay_steps=sic_delay_steps)},
        },
        seed=seed)

    # Inhibitory primary (receptor 2 -> g_in, positive magnitude; no astrocytes).
    inhibitory = sim.connect(inh, neurons, rule=primary_rule('in'), weight=w_i * u.nS,
                             delay=d_i * u.ms, receptor_type=2, seed=seed + 1)
    return {'neurons': neurons, 'ex': ex, 'inh': inh, 'astro': astro,
            'primary': primary, 'third_in': third_in, 'third_out': third_out,
            'inhibitory': inhibitory}


def run(sim_time=1000.0, *, dt=0.1, n_ex=400, n_in=100, n_astro=500, p_primary=0.1,
        p_third=0.5, pool_size=10, pool_type='random', poisson_rate=2000.0, w_e=1.0,
        w_i=4.0, w_a2n=0.05, d_e=2.0, d_i=1.0, tau_syn_ex=2.0, tau_syn_in=4.0,
        IP3_0=0.4, sic_delay_steps=10, seed=0, n_rec_astro=50):
    """Build + run the Bernoulli Brunel-with-astrocytes network.

    Parameters
    ----------
    sim_time : float, default 1000.0
        Simulation duration (ms).
    dt : float, default 0.1
        Integration step (ms).
    n_ex, n_in, n_astro : int
        Population sizes (default: a tractable dense-friendly scale of the demo).
    p_primary : float, default 0.1
        Bernoulli connection probability for the excitatory and inhibitory primaries.
    p_third, pool_size, pool_type
        Astrocyte third-factor rule (defaults: ``0.5``, ``10``, ``'random'`` -- the
        demo values).
    poisson_rate : float, default 2000.0
        Per-neuron background Poisson rate (Hz).
    w_e, w_i, w_a2n, d_e, d_i, tau_syn_ex, tau_syn_in, IP3_0, sic_delay_steps
        Synapse / neuron / astrocyte parameters (demo values).
    seed : int, default 0
        Base PRNG seed (Poisson drive + connectivity draws).
    n_rec_astro : int, default 50
        Number of astrocytes recorded for IP3 / Ca.

    Returns
    -------
    dict
        ``{'times', 'rate', 'e_rate', 'i_rate', 'IP3', 'Ca', 'I_SIC'}`` -- the time
        axis, mean per-neuron firing rates (overall / excitatory / inhibitory,
        spikes/s) and recorded astrocyte IP3/Ca + neuron I_SIC traces.
    """
    brainstate.random.seed(seed)
    brainstate.environ.set(dt=dt * u.ms)
    sim = bp.Simulator(dt=dt * u.ms)

    def primary_rule(_role):
        return bp.pairwise_bernoulli(p_primary)

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
    """Run the Bernoulli Brunel-with-astrocytes demo and report rates / dynamics."""
    out = run()
    ip3 = np.asarray(u.get_mantissa(out['IP3']))
    isic = np.asarray(u.get_mantissa(out['I_SIC']))
    print(f"astrocyte_brunel_bernoulli: e_rate={out['e_rate']:.2f} Hz, "
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
        plt.savefig("examples/nest_like/astrocyte_brunel_bernoulli.png", dpi=100)
        print("  wrote examples/nest_like/astrocyte_brunel_bernoulli.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
