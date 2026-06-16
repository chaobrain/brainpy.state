# examples/nest_like/wang_decision_making.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Wang (2002) spiking decision-making network on ``iaf_bw_2001``.

Port of NEST's ``wang_decision_making.py`` onto brainpy.state's explicit
``Simulator`` API, driving the real ``iaf_bw_2001`` conductance neuron with its
AMPA / GABA / **recurrent NMDA** receptors.

The model
---------
Wang's cortical attractor network [1]_ implements perceptual decision making as
evidence accumulation. Two selective excitatory populations (A, B), each 15 % of
the excitatory pool, compete: each **excites itself** through strong, slow NMDA
synapses (potentiated weight ``w_plus``) and **depresses the other** (``w_minus``),
while a shared inhibitory pool mediates a winner-take-all competition. A small
input bias toward A or B (``coherence``) is integrated by the slow NMDA
reverberation until one population ramps to a high rate and the other is
suppressed. The remaining nonselective excitatory neurons and the inhibitory pool
close the loop. External drive is independent per-neuron Poisson AMPA: a 2400 Hz
background to every cell, plus a time-inhomogeneous signal (rates resampled every
50 ms from ``Normal(mu_0 +/- rho * coherence, sigma)``) onto A and B during
``[1000, 2000]`` ms.

The recurrent-NMDA seam
-----------------------
An NMDA event does not deposit ``weight``; it deposits ``weight * spike_offset``,
where ``spike_offset = k0 + k1 * s_NMDA_pre`` is a **presynaptic** quantity (the
sender's per-spike NMDA gate increment; NEST requires the sender be an
``iaf_bw_2001``). This network is therefore built on the *graded-emission* seam:
``connect(pre, post, receptor_type=NMDA, comm='dense')`` delivers ``weight *
spike_offset`` over the NMDA channel (dense matmul — the sparse path would binarize
the presynaptic value). That seam is validated against live NEST to machine
precision, both feed-forward
(``brainpy_state/_nest/_validation/iaf_bw_2001_nest_parity_test.py``) and
**recurrently** — the design arbiter
(``brainpy_state/_nest/_validation/iaf_bw_2001_recurrent_nmda_parity_test.py``).
Distributional winner-take-all parity for *this* network against live NEST is in
``brainpy_state/_nest/_validation/wang_decision_making_test.py``.

Scaling
-------
``build`` accepts a reduced ``(ne, ni)`` and rescales recurrent weights by
``N_full / N`` (mean-field preserving: an all-to-all sum over fewer presynaptic
cells keeps the same total recurrent conductance). External (per-neuron Poisson)
weights are size-invariant and are not rescaled.

Run:  python examples/nest_like/wang_decision_making.py

References
----------
.. [1] Wang X-J. 2002. Probabilistic decision making by slow reverberation in
       cortical circuits. Neuron 36(5):955-968.
       https://doi.org/10.1016/S0896-6273(02)01092-9
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u
import braintools

from brainpy.state import (Simulator, iaf_bw_2001, poisson_generator,
                           inhomogeneous_poisson_generator, spike_recorder,
                           all_to_all)

# -- network constants (Wang 2002; values from upstream wang_decision_making.py) --
NE_FULL, NI_FULL = 1600, 400      # full-scale excitatory / inhibitory pool sizes
F = 0.15                          # fraction of E in each selective population

#: Excitatory-population neuron parameters (plain floats; units attached in build).
EPOP = dict(tau_GABA=5.0, tau_AMPA=2.0, tau_decay_NMDA=100.0, tau_rise_NMDA=2.0,
            alpha=0.5, conc_Mg2=1.0, g_L=25.0, E_L=-70.0, E_ex=0.0, E_in=-70.0,
            V_reset=-55.0, V_th=-50.0, C_m=500.0, t_ref=2.0)
#: Inhibitory population: faster, smaller cells.
IPOP = {**EPOP, 'g_L': 20.0, 'C_m': 200.0, 't_ref': 1.0}

#: Conductances (nS). ``_ex`` / ``_in`` is the *target* type; ``ext`` is external.
G = dict(AMPA_ex=0.05, AMPA_ext_ex=2.1, NMDA_ex=0.165, GABA_ex=1.3,
         AMPA_in=0.04, AMPA_ext_in=1.62, NMDA_in=0.13, GABA_in=1.0)

W_PLUS = 1.7                                       # within-selective potentiation
W_MINUS = 1.0 - F * (W_PLUS - 1.0) / (1.0 - F)     # cross / nonsel->sel depression
DELAY, DELAY_EXT = 0.5, 0.1                        # recurrent / external delay (ms)

MU0, SIGMA = 40.0, 4.0                             # signal base rate / std (Hz)
RHO = MU0 / 100.0                                  # coherence -> rate scaling
SIGNAL_START, SIGNAL_DUR, SIGNAL_DT = 1000.0, 1000.0, 50.0
RATE_BG = 2400.0                                   # background Poisson rate (Hz)


def _bw_params(kind):
    """Return an ``iaf_bw_2001`` parameter dict (units attached) for ``'E'``/``'I'``."""
    p = EPOP if kind == 'E' else IPOP
    return dict(
        E_L=p['E_L'] * u.mV, E_ex=p['E_ex'] * u.mV, E_in=p['E_in'] * u.mV,
        V_th=p['V_th'] * u.mV, V_reset=p['V_reset'] * u.mV, C_m=p['C_m'] * u.pF,
        g_L=p['g_L'] * u.nS, t_ref=p['t_ref'] * u.ms, tau_AMPA=p['tau_AMPA'] * u.ms,
        tau_GABA=p['tau_GABA'] * u.ms, tau_decay_NMDA=p['tau_decay_NMDA'] * u.ms,
        tau_rise_NMDA=p['tau_rise_NMDA'] * u.ms, alpha=p['alpha'] / u.ms,
        conc_Mg2=p['conc_Mg2'] * u.mM,
        V_initializer=braintools.init.Constant(p['E_L'] * u.mV))


def _signal_rates(coherence, seed):
    """Per-interval signal rates for A and B (resampled every ``SIGNAL_DT`` ms).

    Returns ``(times, rates_a, rates_b)`` with absolute ``times`` (ms) and rates in
    Hz, clipped at 0. ``mu_a = MU0 + RHO * coherence``, ``mu_b = MU0 - RHO *
    coherence``; positive coherence biases A.
    """
    rng = np.random.default_rng(seed)
    n_upd = int(SIGNAL_DUR / SIGNAL_DT)
    mu_a, mu_b = MU0 + RHO * coherence, MU0 - RHO * coherence
    ra = np.clip(rng.normal(mu_a, SIGMA, n_upd), 0.0, None)
    rb = np.clip(rng.normal(mu_b, SIGMA, n_upd), 0.0, None)
    times = SIGNAL_START + SIGNAL_DT * np.arange(n_upd)
    return times, ra, rb


def build(coherence, seed, *, ne=NE_FULL, ni=NI_FULL, T=4000.0, dt=0.1):
    """Assemble the Wang decision network on the ``Simulator``.

    Parameters
    ----------
    coherence : float
        Input bias toward A (positive) or B (negative); ``0`` is unbiased.
    seed : int
        Seeds the signal-rate resampling and every Poisson generator (so distinct
        seeds are distinct trials; the same seed reproduces a trial).
    ne, ni : int, optional
        Excitatory / inhibitory pool sizes. Reduced sizes rescale recurrent weights
        by ``N_full / N`` (mean-field preserving). Defaults are full scale.
    T : float, optional
        Simulation duration (ms), stored in the returned record. Default 4000.
    dt : float, optional
        Time step (ms). Default 0.1.

    Returns
    -------
    tuple
        ``(sim, rec)`` where ``rec`` holds the ``selA``/``selB``/``NS`` views, the
        ``srA``/``srB`` spike recorders, ``nA`` (selective size), ``T`` and ``dt``.
    """
    scale_E, scale_I = NE_FULL / ne, NI_FULL / ni
    nA = int(F * ne)
    AMPA, GABA, NMDA = iaf_bw_2001.AMPA, iaf_bw_2001.GABA, iaf_bw_2001.NMDA

    sim = Simulator(dt=dt * u.ms)
    E = sim.create(iaf_bw_2001, ne, params=_bw_params('E'))
    I = sim.create(iaf_bw_2001, ni, params=_bw_params('I'))
    selA, selB, NS = E[:nA], E[nA:2 * nA], E[2 * nA:]

    def conn(src, tgt, factor, g, rt, scale):
        sim.connect(src, tgt, weight=factor * g * scale * u.nS, delay=DELAY * u.ms,
                    rule=all_to_all, receptor_type=rt, comm='dense')

    def conn_exc(src, tgt, factor):
        """Excitatory projection: AMPA + recurrent NMDA, E-target conductances."""
        conn(src, tgt, factor, G['AMPA_ex'], AMPA, scale_E)
        conn(src, tgt, factor, G['NMDA_ex'], NMDA, scale_E)

    # Recurrent E->E with the block-structured WTA weight matrix:
    #            selA      selB      NS
    #   selA   w_plus    w_minus    1.0
    #   selB   w_minus   w_plus     1.0
    #   NS     w_minus   w_minus    1.0
    # The whole-E -> NS call covers the uniform "1.0 into NS" column (incl. NS->NS
    # autapses, matching NEST's default all_to_all).
    conn_exc(E, NS, 1.0)
    conn_exc(selA, selA, W_PLUS)
    conn_exc(selB, selB, W_PLUS)
    conn_exc(selA, selB, W_MINUS)
    conn_exc(selB, selA, W_MINUS)
    conn_exc(NS, selA, W_MINUS)
    conn_exc(NS, selB, W_MINUS)

    # E -> I (AMPA + NMDA, I-target conductances) and I -> E / I -> I (GABA).
    conn(E, I, 1.0, G['AMPA_in'], AMPA, scale_E)
    conn(E, I, 1.0, G['NMDA_in'], NMDA, scale_E)
    conn(I, E, 1.0, G['GABA_ex'], GABA, scale_I)
    conn(I, I, 1.0, G['GABA_in'], GABA, scale_I)

    # External background: independent per-neuron Poisson AMPA (size-invariant).
    bg = sim.create(poisson_generator, rate=RATE_BG * u.Hz, rng_seed=seed)
    sim.connect(bg, E, weight=G['AMPA_ext_ex'] * u.nS, delay=DELAY_EXT * u.ms,
                rule=all_to_all, receptor_type=AMPA)
    sim.connect(bg, I, weight=G['AMPA_ext_in'] * u.nS, delay=DELAY_EXT * u.ms,
                rule=all_to_all, receptor_type=AMPA)

    # Selective time-inhomogeneous signal (0 outside [SIGNAL_START, +SIGNAL_DUR]).
    times, ra, rb = _signal_rates(coherence, seed)
    rt_times = np.concatenate(([DELAY_EXT], times, [SIGNAL_START + SIGNAL_DUR]))
    sig_a = sim.create(inhomogeneous_poisson_generator, rng_seed=seed + 101,
                       rate_times=rt_times * u.ms,
                       rate_values=np.concatenate(([0.0], ra, [0.0])) * u.Hz)
    sig_b = sim.create(inhomogeneous_poisson_generator, rng_seed=seed + 202,
                       rate_times=rt_times * u.ms,
                       rate_values=np.concatenate(([0.0], rb, [0.0])) * u.Hz)
    sim.connect(sig_a, selA, weight=G['AMPA_ext_ex'] * u.nS, delay=DELAY_EXT * u.ms,
                rule=all_to_all, receptor_type=AMPA)
    sim.connect(sig_b, selB, weight=G['AMPA_ext_ex'] * u.nS, delay=DELAY_EXT * u.ms,
                rule=all_to_all, receptor_type=AMPA)

    srA = sim.create(spike_recorder)
    srB = sim.create(spike_recorder)
    sim.connect(selA, srA)
    sim.connect(selB, srB)
    return sim, dict(selA=selA, selB=selB, NS=NS, E=E, I=I, nA=nA, T=T, dt=dt,
                     srA=srA, srB=srB)


def _pop_rate(spk, dt, *, window_ms=50.0):
    """Population firing rate (Hz) over time from a ``(T, n)`` spike matrix.

    A boxcar moving average of the per-step population spike count, converted to
    spikes/second/neuron.
    """
    spk = np.asarray(spk)
    n = max(1, spk.shape[1])
    count = spk.sum(axis=1).astype(float)
    w = max(1, int(round(window_ms / dt)))
    smooth = np.convolve(count, np.ones(w) / w, mode='same')
    return smooth / n / (dt / 1000.0)


def decision_from_rates(rate_a, rate_b, dt, *, start_ms=SIGNAL_START, thr_hz=15.0):
    """Read out the winner and decision time from two rate traces.

    The winner is the first population whose rate crosses ``thr_hz`` (and leads the
    other) at or after ``start_ms``.

    Parameters
    ----------
    rate_a, rate_b : array_like
        Per-step firing-rate traces (Hz) for populations A and B.
    dt : float
        Time step (ms) of the traces.
    start_ms : float, optional
        Only crossings at/after this time count (the signal onset). Default
        ``SIGNAL_START``.
    thr_hz : float, optional
        Decision threshold (Hz). Default 15.

    Returns
    -------
    dict
        ``{'winner': 'A' | 'B' | None, 't_decision': float | None}`` (time in ms).
    """
    a, b = np.asarray(rate_a, float), np.asarray(rate_b, float)
    k0 = int(round(start_ms / dt))
    cross_a = (a >= thr_hz) & (a > b)
    cross_b = (b >= thr_hz) & (b > a)
    cross_a[:k0] = False
    cross_b[:k0] = False
    ia = int(np.argmax(cross_a)) if cross_a.any() else None
    ib = int(np.argmax(cross_b)) if cross_b.any() else None
    if ia is None and ib is None:
        return dict(winner=None, t_decision=None)
    if ib is None or (ia is not None and ia <= ib):
        return dict(winner='A', t_decision=ia * dt)
    return dict(winner='B', t_decision=ib * dt)


def run_decision(coherence, seed, *, ne=NE_FULL, ni=NI_FULL, T=4000.0, dt=0.1,
                 thr_hz=15.0):
    """Build, simulate, and read out a single decision trial.

    Returns
    -------
    dict
        ``winner`` ('A'/'B'/None), ``t_decision`` (ms or None), and the smoothed
        ``rate_a`` / ``rate_b`` traces (Hz, ``(T,)`` arrays).
    """
    sim, rec = build(coherence, seed, ne=ne, ni=ni, T=T, dt=dt)
    res = sim.simulate(T * u.ms)
    rate_a = _pop_rate(res.spikes(rec['srA'].segments[0].population), dt)
    rate_b = _pop_rate(res.spikes(rec['srB'].segments[0].population), dt)
    dec = decision_from_rates(rate_a, rate_b, dt, thr_hz=thr_hz)
    return dict(winner=dec['winner'], t_decision=dec['t_decision'],
                rate_a=rate_a, rate_b=rate_b)


def main():   # pragma: no cover - manual full-scale demo driver (I/O + matplotlib)
    coherence, seed = 25.6, 1
    print(f"Wang (2002) decision network on iaf_bw_2001 — full scale "
          f"({NE_FULL} exc + {NI_FULL} inh), coherence=+{coherence} (biased to A).")
    print("  Building + simulating 4000 ms (dense recurrent NMDA; ~2-4 min on CPU)...")
    out = run_decision(coherence, seed)
    a, b = out['rate_a'], out['rate_b']
    late = slice(int(2000 / 0.1), int(3000 / 0.1))   # post-signal attractor window
    print(f"  Winner: {out['winner']}  decision time: {out['t_decision']} ms")
    print(f"  Late-window mean rate  A: {a[late].mean():6.2f} Hz   "
          f"B: {b[late].mean():6.2f} Hz")
    # Tiny text rate trace (downsampled) so the demo is informative headless.
    step = int(len(a) / 40) or 1
    sa = a[::step]
    peak = max(sa.max(), 1.0)
    print("  A-rate over time (each row ~100 ms):")
    for k in range(0, len(sa), 4):
        bar = '#' * int(40 * sa[k] / peak)
        print(f"    t={k * step * 0.1:6.0f} ms |{bar}")

    try:
        import matplotlib.pyplot as plt
        t = np.arange(len(a)) * 0.1
        plt.figure(figsize=(8, 4))
        plt.plot(t, a, label='pop A', color='C0')
        plt.plot(t, b, label='pop B', color='C1')
        plt.axvspan(SIGNAL_START, SIGNAL_START + SIGNAL_DUR, color='k', alpha=0.07,
                    label='signal')
        plt.xlabel('time (ms)'); plt.ylabel('rate (Hz)')
        plt.title(f'Wang decision network — coherence +{coherence}')
        plt.legend(); plt.tight_layout()
        plt.savefig('examples/nest_like/wang_decision_making_rates.png', dpi=100)
        print("  wrote examples/nest_like/wang_decision_making_rates.png")
    except ImportError:
        print("  (matplotlib not installed; skipping rate plot)")


if __name__ == '__main__':   # pragma: no cover
    main()
