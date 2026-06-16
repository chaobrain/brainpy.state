# examples/nest_like/evaluate_quantal_stp_synapse.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Quantal (stochastic) short-term plasticity -- NEST-style port.

Port of NEST's ``evaluate_quantal_stp_synapse.py``. ``quantal_stp_synapse`` is the
*stochastic* variant of the Tsodyks-Markram model: each edge has ``n`` release
sites, and every presynaptic spike stochastically recovers depleted sites then
releases ``n_rel ~ Binomial(available, u)`` of them, delivering ``n_rel * weight``
(so the maximal amplitude is ``n * weight``). Averaged over trials the released
fraction converges to the deterministic ``tsodyks2_synapse`` envelope -- this
script shows that convergence for a depressing and a facilitating regime.

A regular presynaptic burst drives a single ``quantal_stp_synapse`` edge onto a
linear, never-spiking ``iaf_psc_exp`` post (``V_th = 1e4`` mV, ``tau_syn_ex = 3``),
so the post V_m **is** the (stochastic) PSC-amplitude train. The per-run ``seed``
is forwarded to ``connect`` and controls the release PRNG, so realizations differ
across seeds and the seed-mean tracks the deterministic limit (plotted as the
``tsodyks2`` line with ``weight = n * w``).

In NEST a plastic synapse cannot be driven by a device, so a ``parrot_neuron``
relays the train; on the ``Simulator`` API a ``spike_generator`` drives the edge
directly. The two STP regimes follow the cluster-01 quantal-STP parity drive; the
example uses more release sites (``N_SITES = 100``, ``WEIGHT`` scaled so ``n*w``
is unchanged) to lower the per-trial variance. Because the PRNG stream differs
from NEST, parity is **distributional** (seed-mean V_m, category D).

Run:  python examples/nest_like/evaluate_quantal_stp_synapse.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import braintools
import numpy as np
import brainunit as u

from brainpy_state import (Simulator, iaf_psc_exp, spike_generator, multimeter,
                           quantal_stp_synapse, tsodyks2_synapse)

#: Resolution (ms).
DT = 0.1
#: 30-spike regular burst at 15 ms ISI (cluster-01 quantal protocol), in ms.
TRAIN = list(np.arange(50.0, 50.0 + 30 * 15.0, 15.0))
#: Simulation horizon (ms).
T_SIM = 700.0
#: Release sites per edge. Larger than the cluster-01 drive's 30 to lower the
#: per-trial variance (so the seed-mean is a tight estimate at a modest seed
#: count); ``WEIGHT`` is scaled to keep the deterministic envelope ``n*w`` fixed.
N_SITES = 100
#: Per-site weight (pA); the maximal per-spike amplitude is ``N_SITES * WEIGHT``
#: (== 1800 pA, the cluster-01 ``30 * 60``).
WEIGHT = 18.0
#: Seeds whose mean is compared to NEST distributionally (category D).
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
#: The two STP regimes (release-probability / time-constant pairs).
REGIMES = {
    "depression": dict(U=0.5, tau_rec=150.0, tau_fac=0.0),
    "facilitation": dict(U=0.15, tau_rec=120.0, tau_fac=500.0),
}


def _post(sim):
    """Build the linear, never-spiking ``iaf_psc_exp`` post (V_m == PSC train)."""
    return sim.create(
        iaf_psc_exp, 1, C_m=250. * u.pF, tau_m=20. * u.ms,
        tau_syn_ex=3. * u.ms, tau_syn_in=3. * u.ms, t_ref=2. * u.ms,
        E_L=0. * u.mV, V_reset=0. * u.mV, V_th=1e4 * u.mV,
        V_initializer=braintools.init.Constant(0. * u.mV))


def run(regime="depression", seed=1, weight=WEIGHT, n_sites=N_SITES,
        train=TRAIN, t_sim=T_SIM):
    """Drive one ``quantal_stp_synapse`` edge for a single stochastic realization.

    Parameters
    ----------
    regime : {"depression", "facilitation"}, optional
        Which STP regime's parameters to use. Default ``"depression"``.
    seed : int, optional
        Per-run release-PRNG seed (forwarded to ``connect``). Default ``1``.
    weight : float, optional
        Per-site weight (pA). Default :data:`WEIGHT`.
    n_sites : int, optional
        Number of release sites. Default :data:`N_SITES`.
    train : sequence of float, optional
        Presynaptic spike times (ms). Default :data:`TRAIN`.
    t_sim : float, optional
        Simulation horizon (ms). Default :data:`T_SIM`.

    Returns
    -------
    times : numpy.ndarray
        Recorder time axis (ms).
    vm : numpy.ndarray
        Post membrane potential (mV) -- one stochastic PSC-amplitude train.
    """
    p = REGIMES[regime]
    sim = Simulator(dt=DT * u.ms)
    post = _post(sim)
    sg = sim.create(spike_generator, spike_times=np.asarray(train) * u.ms)
    sim.connect(sg, post, synapse=quantal_stp_synapse(
        weight=weight * u.pA, n=n_sites, U=p["U"],
        tau_rec=p["tau_rec"] * u.ms, tau_fac=p["tau_fac"] * u.ms), seed=seed)
    mm = sim.create(multimeter, record_from=["V_m"], interval=DT * u.ms)
    sim.connect(mm, post)              # reversed: the multimeter observes the neuron
    res = sim.simulate(t_sim * u.ms)
    times = np.asarray(u.get_mantissa(res.times / u.ms))
    vm = np.asarray(u.get_mantissa(res.trace(mm, "V_m") / u.mV)).reshape(-1)
    return times, vm


def mean_vm(regime="depression", seed=1, **kw):
    """Time-mean post V_m (mV) of one realization -- the per-seed parity sample.

    Parameters
    ----------
    regime : {"depression", "facilitation"}, optional
        STP regime. Default ``"depression"``.
    seed : int, optional
        Release-PRNG seed. Default ``1``.
    **kw
        Forwarded to :func:`run`.

    Returns
    -------
    float
        Mean of the post V_m trace (mV) over the run.
    """
    return float(np.mean(run(regime, seed, **kw)[1]))


def seed_mean_trace(regime="depression", seeds=SEEDS, **kw):
    """Seed-averaged V_m trace -- the stochastic envelope shown in the figure.

    Parameters
    ----------
    regime : {"depression", "facilitation"}, optional
        STP regime. Default ``"depression"``.
    seeds : sequence of int, optional
        Seeds to average over. Default :data:`SEEDS`.
    **kw
        Forwarded to :func:`run`.

    Returns
    -------
    times : numpy.ndarray
        Recorder time axis (ms).
    mean_trace : numpy.ndarray
        Mean post V_m (mV) across ``seeds``.
    """
    traces = [run(regime, s, **kw)[1] for s in seeds]
    times, _ = run(regime, seeds[0], **kw)
    return times, np.mean(np.stack(traces), axis=0)


def deterministic_reference(regime="depression", weight=WEIGHT, n_sites=N_SITES,
                            train=TRAIN, t_sim=T_SIM):
    """Deterministic ``tsodyks2`` limit the quantal seed-mean converges to.

    The quantal mean amplitude is ``E[n_rel] * weight`` with ``n_rel`` binomial
    over ``n_sites`` sites, whose limit is the Tsodyks-Markram ``x*u*weight_t``
    with ``weight_t = n_sites * weight``. This builds that reference for overlay.

    Parameters
    ----------
    regime : {"depression", "facilitation"}, optional
        STP regime. Default ``"depression"``.
    weight : float, optional
        Per-site quantal weight (pA). Default :data:`WEIGHT`.
    n_sites : int, optional
        Number of release sites. Default :data:`N_SITES`.
    train : sequence of float, optional
        Presynaptic spike times (ms). Default :data:`TRAIN`.
    t_sim : float, optional
        Simulation horizon (ms). Default :data:`T_SIM`.

    Returns
    -------
    times : numpy.ndarray
        Recorder time axis (ms).
    vm : numpy.ndarray
        Deterministic post V_m (mV).
    """
    p = REGIMES[regime]
    sim = Simulator(dt=DT * u.ms)
    post = _post(sim)
    sg = sim.create(spike_generator, spike_times=np.asarray(train) * u.ms)
    sim.connect(sg, post, synapse=tsodyks2_synapse(
        weight=n_sites * weight * u.pA, U=p["U"], u=p["U"], x=1.0,
        tau_rec=p["tau_rec"] * u.ms, tau_fac=p["tau_fac"] * u.ms))
    mm = sim.create(multimeter, record_from=["V_m"], interval=DT * u.ms)
    sim.connect(mm, post)
    res = sim.simulate(t_sim * u.ms)
    times = np.asarray(u.get_mantissa(res.times / u.ms))
    vm = np.asarray(u.get_mantissa(res.trace(mm, "V_m") / u.mV)).reshape(-1)
    return times, vm


def main():
    print("Quantal STP (brainpy.state, iaf_psc_exp post, V_m = stochastic PSC train)")
    fig_data = {}
    for regime in REGIMES:
        t, mean_tr = seed_mean_trace(regime)
        _, det = deterministic_reference(regime)
        fig_data[regime] = (t, mean_tr, det)
        print(f"  {regime:12s}: seed-mean <V_m> {mean_tr.mean():8.3f} mV   "
              f"deterministic <V_m> {det.mean():8.3f} mV")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        for ax, regime in zip(axes, REGIMES):
            t, mean_tr, det = fig_data[regime]
            for s in SEEDS:                       # thin individual realizations
                _, v = run(regime, s)
                ax.plot(t, v, color="0.8", lw=0.6, zorder=1)
            ax.plot(t, mean_tr, "C0", lw=1.6, label="quantal seed-mean", zorder=3)
            ax.plot(t, det, "C3--", lw=1.4, label="tsodyks2 limit (w=n*w)", zorder=2)
            ax.set_title(regime); ax.set_xlabel("time (ms)"); ax.set_ylabel("V_m (mV)")
            ax.legend(fontsize=8)
        fig.suptitle("quantal_stp_synapse -- stochastic release vs deterministic limit")
        fig.tight_layout()
        fig.savefig("examples/nest_like/evaluate_quantal_stp_synapse.png", dpi=100)
        print("  wrote examples/nest_like/evaluate_quantal_stp_synapse.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
