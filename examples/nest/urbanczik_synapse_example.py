# examples/nest/urbanczik_synapse_example.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Urbanczik-Senn dendritic prediction-error plasticity -- NEST-style port.

Port of NEST's ``urbanczik_synapse_example.py`` (Urbanczik & Senn, 2014, Fig. 1B).
A two-compartment ``pp_cond_exp_mc_urbanczik`` neuron has its **soma** driven by a
time-varying excitatory/inhibitory conductance *teacher* (so the somatic potential
follows a matching potential ``U_M`` and the soma fires at the imposed rate), while
a fixed spike pattern of ``n_pg`` presynaptic trains drives the **dendrite** through
plastic ``urbanczik_synapse`` edges. The pattern repeats every ``PATTERN`` ms; the
dendritic weights adapt so the dendritic prediction ``V_W*`` reproduces the
somatically-imposed signal -- i.e. the rate prediction error ``phi(U) - phi(V_W*)``
shrinks over training.

This is the user-facing presentation of cluster-21's rebuilt rule: a frozen
``urbanczik_synapse`` spec + pure ``update`` kernel on the
``VoltageCoupledPlasticProj`` substrate (primitive #2, the dendritic post-state
reader), validated sample-for-sample against live NEST in
``brainpy_state/_nest/_validation/urbanczik_synapse_parity_test.py``.

Implementation notes (vs the upstream NEST script)

* **Dendritic pattern source.** Upstream records ``n_pg`` Poisson trains once and
  replays them through ``parrot_neuron`` relays into the plastic edges. Here the
  same replayed pattern feeds a single ``SpikeTime`` population (one spike source
  with per-neuron trains) and a single ``n_pg -> 1`` plastic projection -- the
  substrate, unlike NEST, lets a device drive a plastic edge directly.
* **Somatic conductance teacher.** Upstream drives ``soma_exc``/``soma_inh`` with a
  ``spike_generator`` that fires every step carrying time-varying ``spike_weights``
  (the conductance profile); the port does the same through the routing seam
  (``receptor_type = 1`` / ``2``). The exc profile is a sine; the inh profile is a
  constant during the driven window.
* **Weights.** Dendritic edges are current-based (pA): ``init_w = 0.3*C_m`` (90 pA),
  ``Wmax = 4.5*C_m`` (1350 pA), ``eta = 0.17``, ``tau_Delta = 100 ms`` -- the
  upstream values.

Run:  python examples/nest/urbanczik_synapse_example.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state import (Simulator, SpikeTime, multimeter, pp_cond_exp_mc_urbanczik,
                           spike_generator, spike_recorder, urbanczik_synapse)

#: Resolution (ms) and one pattern's duration (ms).
DT = 0.1
PATTERN = 200.0
#: Dendritic compartment capacitance (pF); sets the weight scale.
C_M = 300.0
#: Presynaptic Poisson rate (spikes/s) for the dendritic pattern.
P_RATE = 10.0
#: urbanczik_synapse parameters (upstream): init/max weight (pA), rate, low-pass (ms).
INIT_W = 0.3 * C_M
WMAX = 4.5 * C_M
ETA = 0.17
TAU_DELTA = 100.0
#: Somatic reversal potentials (mV) and rate-function parameters (shared with the neuron).
E_EX, E_IN = 0.0, -75.0
PHI_MAX, RATE_SLOPE, BETA, THETA = 0.15, 0.5, 1.0 / 3.0, -55.0
#: Somatic conductance teacher amplitudes (nS) and the per-step connection weight.
AMPL_EXC = 0.016 * C_M
OFFSET = 0.018 * C_M
AMPL_INH = 0.06 * C_M
SOMA_W = 10.0 * DT
#: Warm-up / cool-down pattern repetitions framing the driven (teacher-on) window.
N_WARMUP = 2
N_COOLDOWN = 2


def phi(V):
    """Somatic rate function ``phi(U)`` (kHz), matching ``pp_cond_exp_mc_urbanczik``."""
    return PHI_MAX / (1.0 + RATE_SLOPE * np.exp(BETA * (THETA - V)))


def matching_potential(g_E, g_I):
    """Somatic matching potential ``U_M = (g_E E_ex + g_I E_in)/(g_E + g_I)`` (mV)."""
    return (g_E * E_EX + g_I * E_IN) / (g_E + g_I + 1e-12)


def build_pattern(n_pg, n_rep_total, seed):
    """Build the replayed dendritic Poisson pattern as ``SpikeTime`` (indices, times).

    One ``PATTERN``-ms block of ``n_pg`` Poisson trains (rate :data:`P_RATE`) is
    drawn once, then replayed in every repetition (so the *same* pattern recurs,
    which is what the dendrite learns to predict).

    Parameters
    ----------
    n_pg : int
        Number of presynaptic trains (dendritic sources).
    n_rep_total : int
        Total pattern repetitions (driven window plus warm-up/cool-down).
    seed : int
        Seed for the (numpy) Poisson draw.

    Returns
    -------
    indices : jax.Array
        Source index of each spike (``int``), ascending in time.
    times : jax.Array
        Spike time of each spike (ms), ascending.
    spikes_per_pattern : int
        Number of spikes in one pattern block (diagnostic).
    """
    rng = np.random.default_rng(seed)
    pat_idx, pat_t = [], []
    for j in range(n_pg):
        k = rng.poisson(P_RATE * PATTERN / 1000.0)
        for t in np.sort(rng.uniform(0.0, PATTERN, size=k)):
            pat_idx.append(j)
            pat_t.append(t)
    pat_idx = np.asarray(pat_idx, dtype=int)
    pat_t = np.asarray(pat_t, dtype=float)
    idx = np.concatenate([pat_idx for _ in range(n_rep_total)])
    tms = np.concatenate([pat_t + r * PATTERN for r in range(n_rep_total)])
    order = np.argsort(tms)
    return jnp.asarray(idx[order]), jnp.asarray(tms[order]), len(pat_t)


def soma_teacher_weights(steps, t_start, t_end):
    """Per-step somatic exc/inh conductance ``spike_weights`` for the teacher generators.

    Excitatory weight is a sine ``AMPL_EXC sin(2 pi f t) + OFFSET``; inhibitory is a
    constant ``AMPL_INH`` -- both gated to the driven window ``[t_start, t_end)``.
    """
    freq = 2.0 / PATTERN
    drive = (steps >= t_start) & (steps < t_end)
    w_exc = np.where(drive, AMPL_EXC * np.sin(2.0 * np.pi * freq * steps) + OFFSET, 0.0)
    w_inh = np.where(drive, AMPL_INH, 0.0)
    return w_exc, w_inh


def run(n_pg=200, n_pattern_rep=100, seed=1, somatic_seed=0):
    """Train one Urbanczik neuron and return its traces + the learning summary.

    Parameters
    ----------
    n_pg : int, optional
        Number of dendritic presynaptic trains. Default ``200`` (upstream).
    n_pattern_rep : int, optional
        Number of driven (teacher-on) pattern repetitions. Default ``100`` (upstream).
    seed : int, optional
        Seed for the dendritic Poisson pattern. Default ``1``.
    somatic_seed : int, optional
        PRNG seed for the point-process somatic spiking. Default ``0``.

    Returns
    -------
    dict
        Keys: ``t`` (ms), ``V_s``, ``V_d``, ``V_W_star``, ``U_M`` (mV traces),
        ``g_ex``, ``g_in`` (nS), ``weights`` (``(T, n_pg)`` pA), ``soma_spikes``
        (count), ``t_start`` / ``t_end`` (driven window, ms), and the learning
        metrics ``rate_err_first`` / ``rate_err_last`` / ``rate_err_ratio``
        (mean ``|phi(V_s) - phi(V_W*)|`` over the first/last fifth of the driven
        window) and ``rms_first`` / ``rms_last`` (RMS ``|U_M - V_W*|`` mV).
    """
    brainstate.environ.set(dt=DT * u.ms)
    n_rep_total = n_pattern_rep + N_WARMUP + N_COOLDOWN
    T = n_rep_total * PATTERN
    t_start = N_WARMUP * PATTERN
    t_end = (N_WARMUP + n_pattern_rep) * PATTERN

    idx, tms, _ = build_pattern(n_pg, n_rep_total, seed)
    steps = np.arange(DT, T + DT / 2, DT)
    w_exc, w_inh = soma_teacher_weights(steps, t_start, t_end)

    sim = Simulator(dt=DT * u.ms)
    post = sim.create(pp_cond_exp_mc_urbanczik, 1, rng_key=jax.random.PRNGKey(somatic_seed))
    src = sim.create(SpikeTime, n_pg, indices=idx, times=tms * u.ms)
    proj = sim.connect(
        src, post,
        synapse=urbanczik_synapse(weight=INIT_W * u.pA, Wmax=WMAX, eta=ETA,
                                  tau_Delta=TAU_DELTA * u.ms, delay=DT * u.ms),
        receptor_type=3)
    sim.record_weight(proj)
    sg_exc = sim.create(spike_generator, spike_times=steps * u.ms, spike_weights=w_exc)
    sg_inh = sim.create(spike_generator, spike_times=steps * u.ms, spike_weights=w_inh)
    sim.connect(sg_exc, post, weight=SOMA_W * u.nS, receptor_type=1, delay=DT * u.ms)
    sim.connect(sg_inh, post, weight=SOMA_W * u.nS, receptor_type=2, delay=DT * u.ms)
    mm = sim.create(multimeter, record_from=['V_s', 'V_d', 'V_W_star', 'g_ex_s', 'g_in_s'],
                    interval=DT * u.ms)
    sim.connect(mm, post)
    sr = sim.create(spike_recorder)
    sim.connect(post, sr)

    res = sim.simulate(T * u.ms)

    t = np.asarray(u.get_mantissa(res.times / u.ms))
    V_s = np.asarray(u.get_mantissa(res.trace(mm, 'V_s') / u.mV))[:, 0]
    V_d = np.asarray(u.get_mantissa(res.trace(mm, 'V_d') / u.mV))[:, 0]
    V_W_star = np.asarray(u.get_mantissa(res.trace(mm, 'V_W_star') / u.mV))[:, 0]
    g_ex = np.asarray(u.get_mantissa(res.trace(mm, 'g_ex_s') / u.nS))[:, 0]
    g_in = np.asarray(u.get_mantissa(res.trace(mm, 'g_in_s') / u.nS))[:, 0]
    weights = np.asarray(u.get_mantissa(res.weight_trace(proj)))
    U_M = matching_potential(g_ex, g_in)
    soma_spikes = int(np.asarray(res.spikes(sr)).sum())

    # learning summary over the driven window: prediction error first vs last fifth
    di = np.where((t >= t_start) & (t < t_end))[0]
    fifth = max(1, len(di) // 5)
    err = np.abs(phi(V_s) - phi(V_W_star))
    rate_err_first = float(err[di[:fifth]].mean())
    rate_err_last = float(err[di[-fifth:]].mean())
    rms_first = float(np.sqrt(((U_M[di[:fifth]] - V_W_star[di[:fifth]]) ** 2).mean()))
    rms_last = float(np.sqrt(((U_M[di[-fifth:]] - V_W_star[di[-fifth:]]) ** 2).mean()))

    return dict(
        t=t, V_s=V_s, V_d=V_d, V_W_star=V_W_star, U_M=U_M, g_ex=g_ex, g_in=g_in,
        weights=weights, soma_spikes=soma_spikes, t_start=t_start, t_end=t_end,
        rate_err_first=rate_err_first, rate_err_last=rate_err_last,
        rate_err_ratio=rate_err_last / (rate_err_first + 1e-30),
        rms_first=rms_first, rms_last=rms_last)


def main():
    res = run()
    print("Urbanczik-Senn dendritic plasticity (brainpy.state, pp_cond_exp_mc_urbanczik)")
    print(f"  soma spikes (teacher target): {res['soma_spikes']}")
    print(f"  dendritic weights: init {INIT_W:.1f} pA -> "
          f"final mean {res['weights'][-1].mean():.1f} "
          f"(min {res['weights'][-1].min():.1f}, max {res['weights'][-1].max():.1f}) pA")
    print(f"  rate prediction error |phi(U)-phi(V_W*)|: "
          f"first {res['rate_err_first']:.3e} -> last {res['rate_err_last']:.3e} "
          f"(ratio {res['rate_err_ratio']:.3f})")
    print(f"  RMS |U_M - V_W*|: first {res['rms_first']:.2f} -> last {res['rms_last']:.2f} mV")

    try:
        import matplotlib.pyplot as plt
        t, te = res['t'], res['t_end']
        # show the last driven pattern (learned) for a clean Fig-1B-style window
        lo, hi = te - PATTERN, te
        m = (t >= lo) & (t < hi)
        fig, (axA, axB) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axA.plot(t[m], res['V_s'][m], label=r"$U$ (soma)", color="darkblue")
        axA.plot(t[m], res['V_W_star'][m], label=r"$V_W^\ast$ (dendrite pred.)", color="b", ls="--")
        axA.plot(t[m], res['U_M'][m], label=r"$U_M$ (matching)", color="r")
        axA.set_ylabel("membrane pot [mV]")
        axA.set_title("Urbanczik-Senn: dendrite predicts the somatic signal (last pattern)")
        axA.legend()
        axB.plot(t[m], phi(res['V_s'][m]), label=r"$\phi(U)$", color="darkblue")
        axB.plot(t[m], phi(res['V_W_star'][m]), label=r"$\phi(V_W^\ast)$", color="b", ls="--")
        axB.set_ylabel("rate [kHz]")
        axB.set_xlabel("time [ms]")
        axB.legend()
        plt.tight_layout()
        plt.savefig("examples/nest/urbanczik_synapse_example.png", dpi=100)
        print("  wrote examples/nest/urbanczik_synapse_example.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
