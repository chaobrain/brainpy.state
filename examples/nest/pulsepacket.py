# examples/nest/pulsepacket.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Pulse-packet drive — NEST-style port.

Port of NEST's §3.7 ``pulsepacket`` demo. A **pulse packet** is a transient,
synchronous spike volley with a Gaussian temporal profile: for each pulse center
:math:`t_c` the :class:`pulsepacket_generator` emits ``activity`` spikes whose
times are drawn from :math:`\mathcal{N}(t_c, \mathrm{sdev}^2)`. Two things are
shown, both reproduced against the NEST reference:

* **packet shape** (the headline): the population spike-time histogram is a
  Gaussian centered at :math:`t_c` whose **width is the jitter** ``sdev`` — the
  pooled spike-time standard deviation tracks ``sdev`` and the per-step count
  profile matches live NEST distributionally.
* **membrane excursion**: when each packet drives one ``iaf_psc_alpha`` neuron
  (firing threshold raised so the cell never spikes), the **neuron-averaged**
  membrane response is the Gaussian packet profile convolved with the single
  spike post-synaptic potential (PSP) — the analytical excursion of Diesmann
  [1]_. This part is checked **NEST-free** against that analytical solution.

Unlike the sinusoidal generators, ``pulsepacket_generator`` is **host-side**
(NumPy ``default_rng`` draws, a per-train ``deque`` queue, Python control flow on
the integer step), so its ``update()`` is *not* JAX-traceable. It is therefore
driven by an explicit host loop — the eager ``pulsepacket_generator_test`` recipe
— rather than :func:`brainstate.transform.for_loop`. The membrane drive *is* a
compiled rollout: the precomputed packet is replayed through the ``Simulator`` as
a :class:`SpikeTime` population (one spike source per neuron, ``one_to_one``).

Run:  PYTHONPATH=. python examples/nest/pulsepacket.py

References
----------
.. [1] Diesmann M. 2002. Conditions for stable propagation of synchronous
       spiking in cortical neural networks. Dissertation. http://d-nb.info/968772781/34.
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u
import jax.numpy as jnp

from brainpy_state import (Simulator, SpikeTime, iaf_psc_alpha, voltmeter,
                           one_to_one, pulsepacket_generator)

#: Number of spikes per packet (NEST ``activity``).
ACTIVITY = 100
#: Gaussian jitter / packet width (ms) — the headline quantity.
SDEV = 10.0
#: Packet center time (ms). Placed late so the membrane warm-up transient (the
#: V_m relaxation from its init level to E_L) has fully decayed before the pulse.
PULSE_T = 500.0
#: Number of neurons / independent packets (one per neuron).
N_NEURONS = 100
#: Synaptic weight (pA) — shared by the simulated drive and the analytical PSP,
#: so the two excursions are on the same scale regardless of its value.
WEIGHT = 20.0
#: Neuron membrane capacitance (pF), membrane and synaptic time constants (ms).
C_M = 200.0
TAU_M = 20.0
TAU_SYN = 0.5
#: Resting / reset potential (mV); threshold raised so the cell never fires.
E_L = 0.0
V_TH_LARGE = 1.0e9
#: Horizon (ms) and resolution (ms).
SIMTIME = 1000.0
DT = 0.1
#: Histogram bin width (ms) for the packet PSTH display/correlation.
PST_BIN = 2.0


def generate_packet(seed=0, simtime=SIMTIME, dt=DT, n_neurons=N_NEURONS,
                    activity=ACTIVITY, sdev=SDEV, pulse_times=(PULSE_T,)):
    """Drive a host-side pulse-packet generator → per-step spike-count matrix.

    ``pulsepacket_generator`` is imperative (NumPy RNG + ``deque`` queues), so the
    rollout is an explicit Python loop over steps — *not*
    :func:`brainstate.transform.for_loop` — calling ``update()`` once per step
    inside ``environ.context(t=...)``. The generator is built inside
    ``environ.context(dt=...)`` so its timing cache is set before the loop.

    Parameters
    ----------
    seed : int, optional
        PRNG seed for the Gaussian jitter draws. Default ``0``.
    simtime : float, optional
        Horizon in ms. Default :data:`SIMTIME`.
    dt : float, optional
        Resolution in ms. Default :data:`DT`.
    n_neurons : int, optional
        Number of independent packets / output trains. Default :data:`N_NEURONS`.
    activity : int, optional
        Spikes per packet center. Default :data:`ACTIVITY`.
    sdev : float, optional
        Gaussian jitter standard deviation in ms (``0`` ⇒ perfectly synchronous).
        Default :data:`SDEV`.
    pulse_times : sequence of float, optional
        Packet center times in ms. Default ``(PULSE_T,)``.

    Returns
    -------
    numpy.ndarray
        ``(n_steps, n_neurons)`` integer matrix of per-step spike multiplicities.
    """
    n_steps = int(round(simtime / dt))
    pulse_times = np.asarray(pulse_times, dtype=float)
    mat = np.zeros((n_steps, n_neurons), dtype=np.int64)
    with brainstate.environ.context(dt=dt * u.ms):
        gen = pulsepacket_generator(
            in_size=n_neurons, pulse_times=pulse_times * u.ms,
            activity=activity, sdev=sdev * u.ms, rng_seed=seed)
        gen.init_state()
        for step in range(n_steps):
            with brainstate.environ.context(t=step * dt * u.ms):
                mat[step] = np.asarray(gen.update()).reshape(-1)
    return mat


def packet_stats(mat, dt=DT):
    """Total count and pooled spike-time mean/standard deviation of a packet.

    The standard deviation is the headline: for a single packet it estimates the
    jitter ``sdev``; the mean estimates the packet center.

    Parameters
    ----------
    mat : numpy.ndarray
        ``(n_steps, n_neurons)`` spike-count matrix from :func:`generate_packet`.
    dt : float, optional
        Resolution in ms. Default :data:`DT`.

    Returns
    -------
    dict
        ``{'total': int, 'mean_ms': float, 'std_ms': float}``. ``mean_ms`` and
        ``std_ms`` are ``nan`` when the matrix is empty.
    """
    mat = np.asarray(mat)
    per_step = mat.sum(axis=1).astype(float)
    total = int(per_step.sum())
    if total == 0:
        return {'total': 0, 'mean_ms': float('nan'), 'std_ms': float('nan')}
    t_ms = np.arange(mat.shape[0]) * dt
    mean = float(np.sum(t_ms * per_step) / total)
    var = float(np.sum(per_step * (t_ms - mean) ** 2) / total)
    return {'total': total, 'mean_ms': mean, 'std_ms': float(np.sqrt(max(var, 0.0)))}


def packet_psth(mat, dt=DT, bin_ms=PST_BIN):
    """Population spike-count histogram (summed over neurons, then binned).

    Parameters
    ----------
    mat : numpy.ndarray
        ``(n_steps, n_neurons)`` spike-count matrix.
    dt : float, optional
        Resolution in ms. Default :data:`DT`.
    bin_ms : float, optional
        Histogram bin width in ms. Default :data:`PST_BIN`.

    Returns
    -------
    centers_ms : numpy.ndarray
        Bin-center times in ms.
    counts : numpy.ndarray
        Total spike count per bin (summed over all neurons).
    """
    mat = np.asarray(mat)
    n_steps = mat.shape[0]
    per_bin = int(round(bin_ms / dt))
    n_bins = n_steps // per_bin
    pop = mat[:n_bins * per_bin].sum(axis=1).reshape(n_bins, per_bin).sum(axis=1)
    centers = (np.arange(n_bins) + 0.5) * bin_ms
    return centers, pop.astype(float)


def make_psp(t_ms, tau_s=TAU_SYN, tau_m=TAU_M, c_m=C_M, weight=WEIGHT):
    r"""Single-spike post-synaptic potential of ``iaf_psc_alpha`` (mV).

    Diesmann [1]_ eq. 2.3: the membrane response to one alpha-shaped current
    pulse of weight ``weight`` (pA) in a neuron of capacitance ``c_m`` (pF) with
    membrane / synaptic time constants ``tau_m`` / ``tau_s`` (ms). Values for
    ``t_ms <= 0`` are zero (causality).

    Parameters
    ----------
    t_ms : array_like
        Time(s) since the input spike, in ms.
    tau_s, tau_m : float, optional
        Synaptic and membrane time constants in ms.
    c_m : float, optional
        Membrane capacitance in pF.
    weight : float, optional
        Synaptic weight in pA.

    Returns
    -------
    numpy.ndarray
        PSP in mV, same shape as ``t_ms``.
    """
    # Work in SI (s, F, A) as in the NEST demo, then return mV.
    t = np.asarray(t_ms, dtype=float) * 1e-3
    tau_s_s, tau_m_s, c_m_f, w_a = tau_s * 1e-3, tau_m * 1e-3, c_m * 1e-12, weight * 1e-12
    term1 = 1.0 / tau_s_s - 1.0 / tau_m_s
    term2 = np.exp(-t / tau_s_s)
    term3 = np.exp(-t / tau_m_s)
    psp = w_a / c_m_f * np.exp(1.0) / tau_s_s * (
        (-t * term2) / term1 + (term3 - term2) / term1 ** 2)
    psp = psp * 1e3  # V -> mV
    return np.where(np.asarray(t_ms, dtype=float) > 0.0, psp, 0.0)


def analytical_excursion(simtime=SIMTIME, dt=DT, pulse_time=PULSE_T,
                         activity=ACTIVITY, sdev=SDEV, **psp_kw):
    r"""Analytical neuron-averaged membrane excursion (mV) on the sim time grid.

    The averaged response is the Gaussian packet profile convolved with the
    single-spike PSP (Diesmann [1]_ eq. 6.9):
    :math:`U(t) = \mathrm{activity}\cdot(\mathcal{N}(\cdot;t_c,\mathrm{sdev})
    \ast \mathrm{PSP})(t)`.

    Parameters
    ----------
    simtime, dt, pulse_time, activity, sdev : float, optional
        Horizon, resolution, packet center, spikes/packet, jitter (ms / counts).
    **psp_kw
        Forwarded to :func:`make_psp` (``tau_s``, ``tau_m``, ``c_m``, ``weight``).

    Returns
    -------
    t_ms : numpy.ndarray
        Time grid in ms (``0 … simtime``).
    u_mv : numpy.ndarray
        Analytical averaged membrane excursion in mV (baseline 0).
    """
    n_steps = int(round(simtime / dt))
    t_ms = np.arange(n_steps) * dt
    # Causal PSP kernel out to ~10 membrane+synaptic time constants.
    kernel_ms = np.arange(0.0, 10.0 * (TAU_M + TAU_SYN), dt)
    psp = make_psp(kernel_ms, **psp_kw)
    # Gaussian packet density centered at pulse_time, integrating to one spike.
    gauss = np.exp(-0.5 * ((t_ms - pulse_time) / sdev) ** 2) / (sdev * np.sqrt(2.0 * np.pi)) * dt
    u_mv = activity * np.convolve(gauss, psp)[:n_steps]
    return t_ms, u_mv


def excursion_window(t_ms, pulse_time=PULSE_T):
    """Boolean mask of the comparison window around a packet.

    Matches the NEST demo's plot range
    ``[pulse - 5(tau_m+tau_s), pulse + 10(tau_m+tau_s)]`` — wide enough to hold the
    full excursion, tight enough to exclude the pre-pulse membrane warm-up.

    Parameters
    ----------
    t_ms : array_like
        Time grid in ms.
    pulse_time : float, optional
        Packet center in ms. Default :data:`PULSE_T`.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` inside the window.
    """
    t_ms = np.asarray(t_ms)
    lo = pulse_time - 5.0 * (TAU_M + TAU_SYN)
    hi = pulse_time + 10.0 * (TAU_M + TAU_SYN)
    return (t_ms >= lo) & (t_ms < hi)


def averaged_membrane(mat, dt=DT, simtime=SIMTIME, weight=WEIGHT):
    """Replay a packet through ``iaf_psc_alpha`` neurons and average V_m.

    Each neuron receives its own packet column via a :class:`SpikeTime` source
    (``one_to_one``); the threshold is raised so no neuron fires. The recorded
    membrane potentials are averaged over neurons and the pre-pulse baseline is
    subtracted, giving the excursion to compare with :func:`analytical_excursion`.

    Parameters
    ----------
    mat : numpy.ndarray
        ``(n_steps, n_neurons)`` spike-count matrix from :func:`generate_packet`.
    dt : float, optional
        Resolution in ms. Default :data:`DT`.
    simtime : float, optional
        Horizon in ms. Default :data:`SIMTIME`.
    weight : float, optional
        Synaptic weight in pA. Default :data:`WEIGHT`.

    Returns
    -------
    t_ms : numpy.ndarray
        Sample times in ms.
    excursion_mv : numpy.ndarray
        Neuron-averaged membrane excursion in mV (pre-pulse baseline removed).
    """
    mat = np.asarray(mat)
    n_steps, n_neurons = mat.shape
    steps, neurons = np.nonzero(mat)
    counts = mat[steps, neurons].astype(float)
    times_ms = steps * dt

    sim = Simulator(dt=dt * u.ms)
    # SpikeTime requires JAX-backed inputs: u.lax.sort in its constructor rejects
    # NumPy-backed Quantities (brainunit BackendError). Multiplicity rides in weights.
    src = sim.create(SpikeTime, n_neurons,
                     indices=jnp.asarray(neurons),
                     times=jnp.asarray(times_ms) * u.ms,
                     weights=jnp.asarray(counts))
    neu = sim.create(iaf_psc_alpha, n_neurons, E_L=E_L * u.mV, C_m=C_M * u.pF,
                     tau_m=TAU_M * u.ms, V_th=V_TH_LARGE * u.mV, V_reset=E_L * u.mV,
                     tau_syn_ex=TAU_SYN * u.ms)
    vm = sim.create(voltmeter, interval=dt * u.ms)
    sim.connect(src, neu, rule=one_to_one, weight=weight * u.pA, delay=dt * u.ms)
    sim.connect(vm, neu)
    res = sim.simulate(simtime * u.ms)

    v = np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV))
    avg = v.mean(axis=1) if v.ndim == 2 else v
    t_ms = np.arange(avg.shape[0]) * dt
    pre = (t_ms >= PULSE_T - 100.0) & (t_ms < PULSE_T - 5.0)
    baseline = float(avg[pre].mean()) if np.any(pre) else 0.0
    return t_ms, avg - baseline


def main():
    print("pulsepacket (brainpy.state, host-loop generator + Simulator membrane drive)")
    mat = generate_packet(seed=0)
    stats = packet_stats(mat)
    print(f"  packet: {N_NEURONS} neurons x activity={ACTIVITY} -> total={stats['total']} "
          f"(expected {N_NEURONS * ACTIVITY}); center={stats['mean_ms']:.1f} ms "
          f"(pulse {PULSE_T:.0f}); width={stats['std_ms']:.2f} ms (sdev {SDEV:.0f})")

    t_sim, exc = averaged_membrane(mat)
    t_an, u_an = analytical_excursion()
    win = excursion_window(t_sim)
    sim_peak_i = int(np.argmax(exc))
    an_peak_i = int(np.argmax(u_an))
    corr = np.corrcoef(exc[win], u_an[win])[0, 1]
    print(f"  averaged V_m excursion: peak {exc[sim_peak_i]:.3f} mV at {t_sim[sim_peak_i]:.1f} ms; "
          f"analytical peak {u_an[an_peak_i]:.3f} mV at {t_an[an_peak_i]:.1f} ms")
    print(f"  excursion-vs-analytical shape corr = {corr:.4f} "
          f"(latency ~{t_sim[sim_peak_i] - PULSE_T:.1f} ms after the packet)")


if __name__ == "__main__":
    main()
