# examples/nest/astrocyte_interaction.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""A tripartite interaction: two neurons and one astrocyte (NEST port).

Port of NEST's ``astrocyte_interaction.py``. A presynaptic
:class:`~brainpy.state.aeif_cond_alpha_astro` projects both to a postsynaptic
neuron (a direct EPSP) and to an :class:`~brainpy.state.astrocyte_lr_1994` (raising
astrocytic IP3). The astrocyte's cytosolic calcium crosses the SIC threshold and
feeds a slow inward current back into the postsynaptic neuron through a
:class:`~brainpy.state.sic_connection` -- the tripartite loop (Bazargani & Attwell
2016; Li & Rinzel 1994).

The presynaptic neuron is Poisson-driven in the demo (``drive='poisson'``); the
parity test drives it deterministically with a constant current
(``drive='current'``) so the loop matches live NEST sample-for-sample. Records:
presynaptic ``V_m``, astrocytic IP3/Ca, postsynaptic ``I_SIC``. Runs end-to-end
through the :class:`~brainpy.state.Simulator` (one compiled ``for_loop``).

Run:  python examples/nest/astrocyte_interaction.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(sim_time=60000.0, *, dt=0.1, drive='poisson', poisson_rate=1500.0,
        poisson_weight=1.0, I_e=1000.0, delta_IP3=0.2, tau_syn_ex=2.0,
        w_pre2post=1.0, w_pre2astro=1.0, w_astro2post=1.0, conn_delay=1.0,
        sic_delay_steps=10, seed=0, record_v_post=False):
    """Build + run the tripartite loop; return ``(times, V_pre, IP3, Ca, I_SIC)``.

    Parameters
    ----------
    sim_time : float, default 60000.0
        Simulation duration in ms.
    dt : float, default 0.1
        Integration time step in ms.
    drive : {'poisson', 'current'}, default 'poisson'
        ``'poisson'`` injects a ``poisson_generator`` (``poisson_rate`` Hz) into the
        presynaptic neuron (the demo); ``'current'`` sets the presynaptic neuron's
        ``I_e`` to ``I_e`` pA (the deterministic parity path).
    poisson_rate, poisson_weight : float
        Rate (Hz) / weight of the Poisson drive (``drive='poisson'``).
    I_e : float, default 1000.0
        Constant current (pA) on the presynaptic neuron (``drive='current'``).
    delta_IP3 : float, default 0.2
        IP3 increment per presynaptic spike at the astrocyte.
    tau_syn_ex : float, default 2.0
        Excitatory synaptic time constant (ms) of both neurons (NEST demo value).
    w_pre2post, w_pre2astro, w_astro2post : float
        Connection weights; ``w_pre2post=0`` drops the direct EPSP arm and
        ``w_astro2post=0`` drops the SIC arm (each decoupling test).
    conn_delay : float, default 1.0
        Delay (ms) of the pre->post and pre->astro connections (NEST default 1.0).
    sic_delay_steps : int, default 10
        SIC delivery delay in steps (NEST's 1.0 ms default = 10 steps at dt=0.1).
    seed : int, default 0
        PRNG seed for the Poisson drive.
    record_v_post : bool, default False
        Additionally record + return the postsynaptic ``V_m`` (the SIC-modulation
        law).

    Returns
    -------
    times : Quantity
        Time axis (ms).
    v_pre : Quantity
        Presynaptic membrane voltage (mV).
    ip3, ca : Quantity
        Astrocytic IP3 and cytosolic Ca (µM).
    isic : Quantity
        Postsynaptic ``I_SIC`` (pA).
    v_post : Quantity
        Only when ``record_v_post`` -- the postsynaptic ``V_m`` (mV).
    """
    brainstate.random.seed(seed)
    brainstate.environ.set(dt=dt * u.ms)
    sim = bp.Simulator(dt=dt * u.ms)

    npar = {'tau_syn_ex': tau_syn_ex * u.ms}
    pre_par = {**npar, 'I_e': I_e * u.pA} if drive == 'current' else dict(npar)
    pre = sim.create(bp.aeif_cond_alpha_astro, 1, params=pre_par)
    post = sim.create(bp.aeif_cond_alpha_astro, 1, params=npar)
    astro = sim.create(bp.astrocyte_lr_1994, 1, params={'delta_IP3': delta_IP3})

    if drive != 'current':
        pg = sim.create(bp.poisson_generator, 1, rate=poisson_rate * u.Hz)
        sim.connect(pg, pre, weight=poisson_weight, delay=conn_delay * u.ms)
    if w_pre2post > 0:
        sim.connect(pre, post, weight=w_pre2post, delay=conn_delay * u.ms)
    sim.connect(pre, astro, weight=w_pre2astro, delay=conn_delay * u.ms)
    if w_astro2post > 0:
        sim.connect(astro, post,
                    synapse=bp.sic_connection(weight=w_astro2post, delay_steps=sic_delay_steps))

    mm_pre = sim.create(bp.multimeter, record_from=['V_m'])
    sim.connect(mm_pre, pre)
    mm_a = sim.create(bp.multimeter, record_from=['IP3', 'Ca'])
    sim.connect(mm_a, astro)
    post_records = ['I_SIC'] + (['V_m'] if record_v_post else [])
    mm_p = sim.create(bp.multimeter, record_from=post_records)
    sim.connect(mm_p, post)

    res = sim.simulate(sim_time * u.ms)
    out = (res.times, res.trace(mm_pre, 'V_m'), res.trace(mm_a, 'IP3'),
           res.trace(mm_a, 'Ca'), res.trace(mm_p, 'I_SIC'))
    if record_v_post:
        return out + (res.trace(mm_p, 'V_m'),)
    return out


def main():                                            # pragma: no cover - demo driver
    """Run the Poisson-driven tripartite demo and plot V_pre / IP3 / I_SIC / Ca."""
    times, v_pre, ip3, ca, isic = run()
    print(f"astrocyte_interaction: peak IP3={float(u.get_mantissa(ip3).max()):.3f} µM, "
          f"peak Ca={float(u.get_mantissa(ca).max()):.3f} µM, "
          f"peak I_SIC={float(u.get_mantissa(isic).max()):.3f} pA")
    try:
        import matplotlib.pyplot as plt
        t = np.asarray(u.get_mantissa(times)).reshape(-1)
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(6.4, 4.8), dpi=100)
        a = ax.flat
        a[0].plot(t, np.asarray(u.get_mantissa(v_pre)).reshape(-1))
        a[1].plot(t, np.asarray(u.get_mantissa(ip3)).reshape(-1))
        a[2].plot(t, np.asarray(u.get_mantissa(isic)).reshape(-1))
        a[3].plot(t, np.asarray(u.get_mantissa(ca)).reshape(-1))
        a[0].set_title("Presynaptic neuron"); a[0].set_ylabel("Membrane potential (mV)")
        a[1].set_title("Astrocyte"); a[1].set_ylabel(r"[IP$_{3}$] ($\mu$M)")
        a[2].set_title("Postsynaptic neuron"); a[2].set_ylabel("Slow inward current (pA)")
        a[2].set_xlabel("Time (ms)")
        a[3].set_ylabel(r"[Ca$^{2+}$] ($\mu$M)"); a[3].set_xlabel("Time (ms)")
        plt.tight_layout()
        plt.savefig("examples/nest/astrocyte_interaction.png", dpi=100)
        print("  wrote examples/nest/astrocyte_interaction.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
