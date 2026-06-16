# examples/nest_like/astrocyte_small_network.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""A small neuron-astrocyte network (NEST ``astrocyte_small_network`` port).

Port of NEST's ``astrocyte_small_network``. Two populations of
:class:`~brainpy.state.aeif_cond_alpha_astro` neurons (presynaptic + postsynaptic)
and a population of :class:`~brainpy.state.astrocyte_lr_1994` astrocytes are wired
with :meth:`~brainpy.state.Simulator.tripartite_connect`: the primary neuron->neuron
edges use ``pairwise_bernoulli`` and the astrocyte third-factor edges use
``third_factor_bernoulli_with_pool`` (``pool_size=1``, ``pool_type='block'`` -- each
block of postsynaptic neurons draws its astrocyte from a non-overlapping pool). One
realized primary sample is shared across all three arms (NEST tripartite semantics):

* **primary** ``pre -> post`` -- a conductance EPSP (receptor 1, nS).
* **third_in** ``pre -> astro`` -- a delta drive raising astrocytic IP3.
* **third_out** ``astro -> post`` -- a :class:`~brainpy.state.sic_connection`
  feeding the slow inward current ``I_SIC`` back to the postsynaptic neurons.

Presynaptic spikes (driven by a constant ``I_e``) raise IP3, whose calcium crosses
the SIC threshold and feeds ``I_SIC`` into the postsynaptic neurons -- the tripartite
loop (Li & Rinzel 1994; Nadkarni & Jung 2003). The whole network runs end-to-end
through the :class:`~brainpy.state.Simulator` (one compiled ``for_loop``).

Synapse divergence from NEST: NEST's demo uses ``tsodyks_synapse`` for the
primary/third_in arms; this port (and its parity harness) uses **static** synapses,
matching the 15d-validated SIC loop. The connectivity is identical; only the
short-term-plasticity dynamics on those arms differ.

Run:  python examples/nest_like/astrocyte_small_network.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(sim_time=1000.0, *, dt=0.1, n_neurons=10, n_astro=5, p_primary=1.0,
        p_third=1.0, pool_size=1, pool_type='block', I_e=1000.0, tau_syn_ex=2.0,
        delta_IP3=0.2, IP3_0=0.4, w_primary=1.0, w_third_in=1.0, w_a2n=1.0,
        conn_delay=0.1, sic_delay_steps=10, seed=0):
    """Build + run the small neuron-astrocyte tripartite network.

    Parameters
    ----------
    sim_time : float, default 1000.0
        Simulation duration in ms.
    dt : float, default 0.1
        Integration time step in ms.
    n_neurons : int, default 10
        Size of each neuron population (presynaptic and postsynaptic).
    n_astro : int, default 5
        Number of astrocytes. For ``pool_type='block'`` with ``pool_size=1``,
        ``n_neurons`` must be a multiple of ``n_astro`` (each astrocyte serves a
        non-overlapping block of ``n_neurons // n_astro`` postsynaptic neurons).
    p_primary : float, default 1.0
        Bernoulli probability of each pre->post primary edge.
    p_third : float, default 1.0
        Probability of pairing each realized primary edge with an astrocyte.
    pool_size : int, default 1
        Astrocytes per postsynaptic pool.
    pool_type : {'block', 'random'}, default 'block'
        Astrocyte-pool assignment scheme.
    I_e : float, default 1000.0
        Constant drive current (pA) on every presynaptic neuron.
    tau_syn_ex : float, default 2.0
        Excitatory synaptic time constant (ms) of both neuron populations.
    delta_IP3 : float, default 0.2
        IP3 increment per presynaptic spike at the astrocyte.
    IP3_0 : float, default 0.4
        Initial astrocytic IP3 (µM).
    w_primary, w_third_in, w_a2n : float
        Weights of the primary (nS conductance EPSP), third_in (delta -> IP3), and
        third_out (SIC) arms.
    conn_delay : float, default 0.1
        Delay (ms) of the primary / third_in connections.
    sic_delay_steps : int, default 10
        SIC delivery delay in steps (NEST's 1.0 ms default = 10 steps at dt=0.1).
    seed : int, default 0
        PRNG seed for the connectivity sample (irrelevant for the deterministic
        ``p_primary=p_third=1`` block default; matters for random pools / ``p<1``).

    Returns
    -------
    dict
        ``{'times', 'V_pre', 'V_post', 'I_SIC', 'IP3', 'Ca'}`` -- the time axis and
        per-neuron / per-astrocyte recorded traces.
    """
    brainstate.random.seed(seed)
    brainstate.environ.set(dt=dt * u.ms)
    sim = bp.Simulator(dt=dt * u.ms)

    npar = {'tau_syn_ex': tau_syn_ex * u.ms, 'I_e': I_e * u.pA}
    pre = sim.create(bp.aeif_cond_alpha_astro, n_neurons, params=npar)
    post = sim.create(bp.aeif_cond_alpha_astro, n_neurons, params=npar)
    astro = sim.create(bp.astrocyte_lr_1994, n_astro,
                       params={'delta_IP3': delta_IP3, 'IP3_initializer': IP3_0})

    sim.tripartite_connect(
        pre, post, astro,
        conn_spec=bp.pairwise_bernoulli(p_primary),
        third_factor_conn_spec=bp.third_factor_bernoulli_with_pool(
            p=p_third, pool_size=pool_size, pool_type=pool_type),
        syn_specs={
            'primary': {'weight': w_primary * u.nS, 'delay': conn_delay * u.ms,
                        'receptor_type': 1},
            'third_in': {'weight': w_third_in, 'delay': conn_delay * u.ms},
            'third_out': {'synapse': bp.sic_connection(weight=w_a2n,
                                                       delay_steps=sic_delay_steps)},
        },
        seed=seed)

    mm_pre = sim.create(bp.multimeter, record_from=['V_m'])
    sim.connect(mm_pre, pre)
    mm_post = sim.create(bp.multimeter, record_from=['V_m', 'I_SIC'])
    sim.connect(mm_post, post)
    mm_a = sim.create(bp.multimeter, record_from=['IP3', 'Ca'])
    sim.connect(mm_a, astro)

    res = sim.simulate(sim_time * u.ms)
    return {'times': res.times,
            'V_pre': res.trace(mm_pre, 'V_m'),
            'V_post': res.trace(mm_post, 'V_m'),
            'I_SIC': res.trace(mm_post, 'I_SIC'),
            'IP3': res.trace(mm_a, 'IP3'),
            'Ca': res.trace(mm_a, 'Ca')}


def main():                                            # pragma: no cover - demo driver
    """Run the small tripartite network and plot mean V / IP3 / Ca / I_SIC."""
    out = run()
    ip3 = np.asarray(u.get_mantissa(out['IP3']))
    isic = np.asarray(u.get_mantissa(out['I_SIC']))
    print(f"astrocyte_small_network: peak IP3={ip3.max():.3f} µM, "
          f"peak I_SIC={isic.max():.3f} pA")
    try:
        import matplotlib.pyplot as plt
        t = np.asarray(u.get_mantissa(out['times'])).reshape(-1)

        def _mean(x):
            a = np.asarray(u.get_mantissa(x))
            return a.mean(axis=1) if a.ndim == 2 else a

        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(6.4, 4.8), dpi=100)
        a = ax.flat
        a[0].plot(t, _mean(out['V_pre'])); a[0].set_title("Presynaptic neurons")
        a[0].set_ylabel("Mean V (mV)")
        a[1].plot(t, _mean(out['IP3'])); a[1].set_title("Astrocytes")
        a[1].set_ylabel(r"Mean [IP$_{3}$] ($\mu$M)")
        a[2].plot(t, _mean(out['I_SIC'])); a[2].set_title("Postsynaptic neurons")
        a[2].set_ylabel("Mean SIC (pA)"); a[2].set_xlabel("Time (ms)")
        a[3].plot(t, _mean(out['Ca'])); a[3].set_ylabel(r"Mean [Ca$^{2+}$] ($\mu$M)")
        a[3].set_xlabel("Time (ms)")
        plt.tight_layout()
        plt.savefig("examples/nest_like/astrocyte_small_network.png", dpi=100)
        print("  wrote examples/nest_like/astrocyte_small_network.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
