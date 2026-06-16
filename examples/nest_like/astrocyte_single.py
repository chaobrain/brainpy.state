# examples/nest_like/astrocyte_single.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""A single astrocyte with calcium dynamics (NEST ``astrocyte_single`` port).

Port of NEST's ``astrocyte_single.py``. One :class:`~brainpy.state.astrocyte_lr_1994`
is driven by a Poissonian spike train: each presynaptic spike raises astrocytic
inositol 1,4,5-trisphosphate (IP3), which in turn drives the cytosolic calcium
dynamics (Li & Rinzel 1994; De Young & Keizer 1992; Nadkarni & Jung 2003).

NEST's ``astrocyte_lr_1994`` exposes no ``SIC`` recordable (only ``IP3`` /
``Ca_astro`` / ``h_IP3R``), so to surface the resulting slow inward current this
port adds one downstream :class:`~brainpy.state.aeif_cond_alpha_astro` connected by
a :class:`~brainpy.state.sic_connection`; its ``I_SIC`` is the astrocyte's emitted
SIC delivered ``sic_delay_steps`` later. The IP3/Ca dynamics are identical to NEST's
single-astrocyte demo.

The whole model runs through the :class:`~brainpy.state.Simulator` (one compiled
``for_loop``); there is no Python step loop.

Run:  python examples/nest_like/astrocyte_single.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(sim_time=60000.0, *, dt=0.1, poisson_rate=1.0, poisson_weight=1.0,
        spike_times=None, spike_weight=2.0, delta_IP3=0.2,
        ip3_init=None, ca_init=0.073, h_init=0.793,
        sic_weight=1.0, sic_delay_steps=10, with_sic=True,
        seed=0, return_astro_sic=False):
    """Drive one astrocyte and return its IP3/Ca (and downstream ``I_SIC``) traces.

    Parameters
    ----------
    sim_time : float, default 60000.0
        Simulation duration in ms.
    dt : float, default 0.1
        Integration time step in ms.
    poisson_rate : float, default 1.0
        Rate (Hz) of the Poisson drive into the astrocyte (used when
        ``spike_times`` is ``None``).
    poisson_weight : float, default 1.0
        Weight of the Poisson -> astrocyte connection.
    spike_times : sequence of float or None, default None
        If given (ms), a deterministic ``spike_generator`` replaces the Poisson
        drive (the parity-test path). An empty sequence drives the astrocyte with
        no input.
    spike_weight : float, default 2.0
        Weight of each deterministic spike into the astrocyte.
    delta_IP3 : float, default 0.2
        IP3 increment per presynaptic spike (the NEST ``astrocyte_single`` value).
    ip3_init, ca_init, h_init : float
        Initial IP3 / cytosolic Ca / non-inactivated IP3R fraction. ``ip3_init=None``
        keeps the model default (``IP3_0 = 0.16`` µM).
    sic_weight : float, default 1.0
        Weight of the astrocyte -> neuron ``sic_connection`` (0 decouples the arm).
    sic_delay_steps : int, default 10
        SIC delivery delay in steps (NEST's 1.0 ms default = 10 steps at dt=0.1).
    with_sic : bool, default True
        Build the downstream neuron + ``sic_connection`` to expose ``I_SIC``.
    seed : int, default 0
        PRNG seed for the Poisson drive.
    return_astro_sic : bool, default False
        Additionally return the astrocyte's own ``SIC`` recordable.

    Returns
    -------
    times : Quantity
        Time axis (ms).
    ip3, ca : Quantity
        Astrocytic IP3 and cytosolic Ca traces (µM), shape ``(n_steps, 1)``.
    isic : Quantity or None
        Downstream ``I_SIC`` (pA) if ``with_sic`` else ``None``.
    astro_sic : Quantity
        Only when ``return_astro_sic`` -- the astrocyte's emitted ``SIC`` (pA).
    """
    brainstate.random.seed(seed)
    brainstate.environ.set(dt=dt * u.ms)
    sim = bp.Simulator(dt=dt * u.ms)

    apar = {'delta_IP3': delta_IP3, 'Ca_initializer': ca_init, 'h_IP3R_initializer': h_init}
    if ip3_init is not None:
        apar['IP3_initializer'] = ip3_init
    astro = sim.create(bp.astrocyte_lr_1994, 1, params=apar)

    if spike_times is not None:
        if len(spike_times):
            src = sim.create(bp.spike_generator, 1, spike_times=np.asarray(spike_times) * u.ms)
            sim.connect(src, astro, weight=spike_weight, delay=dt * u.ms)
    else:
        src = sim.create(bp.poisson_generator, 1, rate=poisson_rate * u.Hz)
        sim.connect(src, astro, weight=poisson_weight, delay=dt * u.ms)

    astro_records = ['IP3', 'Ca'] + (['SIC'] if return_astro_sic else [])
    mm_a = sim.create(bp.multimeter, record_from=astro_records)
    sim.connect(mm_a, astro)

    mm_p = None
    if with_sic:
        post = sim.create(bp.aeif_cond_alpha_astro, 1)
        sim.connect(astro, post,
                    synapse=bp.sic_connection(weight=sic_weight, delay_steps=sic_delay_steps))
        mm_p = sim.create(bp.multimeter, record_from=['I_SIC'])
        sim.connect(mm_p, post)

    res = sim.simulate(sim_time * u.ms)
    ip3, ca = res.trace(mm_a, 'IP3'), res.trace(mm_a, 'Ca')
    isic = res.trace(mm_p, 'I_SIC') if mm_p is not None else None
    if return_astro_sic:
        return res.times, ip3, ca, isic, res.trace(mm_a, 'SIC')
    return res.times, ip3, ca, isic


def main():                                            # pragma: no cover - demo driver
    """Run the Poisson-driven single-astrocyte demo and plot IP3 / Ca / I_SIC."""
    times, ip3, ca, isic = run()
    print(f"astrocyte_single: peak IP3={float(u.get_mantissa(ip3).max()):.3f} µM, "
          f"peak Ca={float(u.get_mantissa(ca).max()):.3f} µM, "
          f"peak I_SIC={float(u.get_mantissa(isic).max()):.3f} pA")
    try:
        import matplotlib.pyplot as plt
        t = np.asarray(u.get_mantissa(times)).reshape(-1)
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6.4, 5.4), dpi=100)
        ax[0].plot(t, np.asarray(u.get_mantissa(ip3)).reshape(-1))
        ax[1].plot(t, np.asarray(u.get_mantissa(ca)).reshape(-1))
        ax[2].plot(t, np.asarray(u.get_mantissa(isic)).reshape(-1))
        ax[0].set_ylabel(r"[IP$_{3}$] ($\mu$M)")
        ax[1].set_ylabel(r"[Ca$^{2+}$] ($\mu$M)")
        ax[2].set_ylabel("Slow inward current (pA)")
        ax[2].set_xlabel("Time (ms)")
        plt.tight_layout()
        plt.savefig("examples/nest_like/astrocyte_single.png", dpi=100)
        print("  wrote examples/nest_like/astrocyte_single.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
