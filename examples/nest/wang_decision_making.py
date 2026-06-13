# examples/nest/wang_decision_making.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Wang (2002) spiking decision-making network — DEFERRED placeholder + validated core.

Status
------
**The ``iaf_bw_2001`` neuron is fully validated against live NEST** — single-cell
AMPA+GABA and the two-neuron NMDA presynaptic-offset coupling both match NEST to
machine precision (see
``brainpy_state/_nest/_validation/iaf_bw_2001_nest_parity_test.py``). The full
multi-population decision network, however, is **deferred**: it cannot yet be built on
the brainpy.state ``Simulator`` API because of a missing connectivity seam (detailed
below). This file therefore ships a runnable demonstration of the *validated building
block* — the recurrent NMDA coupling between two ``iaf_bw_2001`` neurons — and documents
exactly what the Simulator needs before the network can be ported faithfully.

The model
---------
Wang's cortical attractor network [1]_ models perceptual decision making. Two selective
excitatory populations (A, B) compete; each strongly excites itself through **slow
NMDA** synapses and inhibits the other via a shared interneuron pool. A small bias in
the input to A vs B is integrated over time until one population wins (ramps to a high
rate) and the other is suppressed — a spiking implementation of evidence accumulation.
Neurons are ``iaf_bw_2001``: leaky integrate-and-fire with AMPA (fast exc), GABA (inh),
and NMDA (slow, voltage-gated exc with a Mg2+ block) receptors. The NMDA recurrence is
*essential* — it is the slow positive feedback that holds the decision attractor.

Why the network is deferred (the Simulator seam gap)
----------------------------------------------------
Routing synaptic input into ``iaf_bw_2001`` through ``sim.connect(..., receptor_type=k)``
requires two things, only the first of which is mechanical:

1. **AMPA (1) / GABA (2) — Ohmic, fixable.** ``iaf_bw_2001`` currently exposes no
   ``n_receptors`` / ``w_by_rec`` multi-receptor bridge, so ``sim.connect(receptor_type=k)``
   raises ``AttributeError: 'iaf_bw_2001' object has no attribute 'n_receptors'``. This
   is the same gap that ``iaf_cond_exp`` had; it is fixed the same way (add
   ``n_receptors``, ``receptor_input_unit``, and a ``w_by_rec`` branch in ``update``).

2. **NMDA (3) — presynaptic-offset coupling, the real blocker.** Unlike every other
   receptor in the framework, an NMDA event does not deposit ``weight``; it deposits
   ``weight * s``, where ``s`` is a **presynaptic** quantity — the *sender's* per-spike
   ``spike_offset`` (computed from its ``s_NMDA_pre`` recurrence; NEST enforces that the
   sender be an ``iaf_bw_2001``). The Simulator's event projection deposits a uniform
   ``weight * spike`` across all receptors; it has no path to read a *per-presynaptic-
   neuron* state and fold it into the deposit. Supporting recurrent NMDA therefore needs
   a new **offset-aware event projection** (a presynaptic-state-gated synapse), not just
   the ``w_by_rec`` blob bridge. Because the Wang attractor lives or dies on recurrent
   NMDA, the network cannot be ported faithfully until that seam exists.

The two-neuron demonstration below shows the coupling working correctly *by hand*
(running the sender, reading its ``spike_offset``, depositing ``weight * offset`` onto
the receiver one delay step later) — exactly what an offset-aware projection would
automate across a population. The feed-forward hand-wiring does not generalise to the
*recurrent* population coupling the full network needs, which is why it stays a demo of
the building block rather than the network.

References
----------
.. [1] Wang X-J. 2002. Probabilistic decision making by slow reverberation in cortical
       circuits. Neuron 36(5):955-968. https://doi.org/10.1016/S0896-6273(02)01092-9
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import saiunit as u
import braintools
import jax.numpy as jnp

from brainpy_state import iaf_bw_2001

#: Wang/Brunel-style single-neuron parameters (NEST ``iaf_bw_2001`` defaults family).
BW = dict(E_L=-70.0, E_ex=0.0, E_in=-70.0, V_th=-55.0, V_reset=-60.0, C_m=500.0,
          g_L=25.0, t_ref=2.0, tau_AMPA=2.0, tau_GABA=5.0, tau_decay_NMDA=100.0,
          tau_rise_NMDA=2.0, alpha=0.5, conc_Mg2=1.0)
DT = 0.1


def _neuron():
    """Construct one ``iaf_bw_2001`` with the shared parameters."""
    return iaf_bw_2001(
        1, E_L=BW['E_L'] * u.mV, E_ex=BW['E_ex'] * u.mV, E_in=BW['E_in'] * u.mV,
        V_th=BW['V_th'] * u.mV, V_reset=BW['V_reset'] * u.mV, C_m=BW['C_m'] * u.pF,
        g_L=BW['g_L'] * u.nS, t_ref=BW['t_ref'] * u.ms, tau_AMPA=BW['tau_AMPA'] * u.ms,
        tau_GABA=BW['tau_GABA'] * u.ms, tau_decay_NMDA=BW['tau_decay_NMDA'] * u.ms,
        tau_rise_NMDA=BW['tau_rise_NMDA'] * u.ms, alpha=BW['alpha'] / u.ms,
        conc_Mg2=BW['conc_Mg2'] * u.mM,
        V_initializer=braintools.init.Constant(BW['E_L'] * u.mV))


def nmda_coupling_demo(drive_times=(10.0, 11.0, 12.0, 30.0, 31.0, 32.0),
                       w_drive=350.0, w_nmda=1.2, T=80.0, dt=DT):
    """Demonstrate the validated recurrent-NMDA building block between two neurons.

    A *sender* ``iaf_bw_2001`` is forced to fire by a strong AMPA drive; each of its
    spikes carries a presynaptic ``spike_offset`` that scales the NMDA deposit onto a
    *receiver* neuron one delay step later. This hand-wired feed-forward coupling is the
    exact mechanism validated to machine precision against live NEST.

    Parameters
    ----------
    drive_times : sequence of float
        Times [ms] of the AMPA pulses that drive the sender to spike.
    w_drive : float
        AMPA weight [nS] of the sender drive (large enough to cross threshold).
    w_nmda : float
        NMDA weight [nS] of the sender -> receiver projection.
    T : float
        Simulation duration [ms].
    dt : float
        Integration step [ms].

    Returns
    -------
    dict
        ``sender_spike_times`` [ms], and the receiver ``s_NMDA`` peak and ``V_m`` peak.
    """
    n = int(round(T / dt))
    dsteps = jnp.array(sorted({int(round(t / dt)) for t in drive_times}))
    with brainstate.environ.context(dt=dt * u.ms):
        snd = _neuron(); snd.init_state()

        def sbody(k):
            aw = jnp.where(jnp.any(k == dsteps), w_drive, 0.0)
            snd.add_delta_input('a', aw * u.nS, label='AMPA')
            with brainstate.environ.context(t=k * dt * u.ms):
                spk = snd.update(x=0.0 * u.pA)
            return spk[0], snd.spike_offset.value[0]

        spk_arr, off_arr = brainstate.transform.for_loop(sbody, jnp.arange(n))
        spk_arr = np.asarray(spk_arr); off_arr = np.asarray(off_arr)

        # NMDA deposit = weight * presynaptic spike_offset, delivered one delay step later.
        nmda_in = np.zeros(n)
        nmda_in[1:] = w_nmda * off_arr[:-1] * (spk_arr[:-1] > 0)
        nmda_in = jnp.asarray(nmda_in)

        rcv = _neuron(); rcv.init_state()

        def rbody(k):
            rcv.add_delta_input('n', nmda_in[k] * u.nS, label='NMDA')
            with brainstate.environ.context(t=k * dt * u.ms):
                rcv.update(x=0.0 * u.pA)
            return rcv.s_NMDA.value[0] / u.nS, rcv.V.value[0] / u.mV

        s_nmda, v_m = brainstate.transform.for_loop(rbody, jnp.arange(n))
    spike_steps = np.nonzero(spk_arr > 0)[0]
    return dict(
        sender_spike_times=[round(float(s) * dt, 1) for s in spike_steps],
        receiver_s_nmda_peak=float(np.asarray(s_nmda).max()),
        receiver_vm_peak=float(np.asarray(v_m).max()),
    )


def main():
    print("Wang (2002) spiking decision-making network — DEFERRED placeholder.")
    print("  Neuron iaf_bw_2001: VALIDATED vs live NEST (AMPA+GABA single-cell and")
    print("  2-neuron NMDA, machine precision — see iaf_bw_2001_nest_parity_test.py).")
    print("  Full network deferred: the Simulator lacks an offset-aware NMDA event")
    print("  projection (NMDA deposits weight * sender spike_offset, not weight * spike).")
    print()
    print("  Demonstrating the validated recurrent-NMDA building block (2 neurons):")
    out = nmda_coupling_demo()
    print(f"    sender fired at {out['sender_spike_times']} ms")
    print(f"    receiver NMDA gate peaked at s_NMDA = {out['receiver_s_nmda_peak']:.4f} nS")
    print(f"    receiver depolarised to V_m peak = {out['receiver_vm_peak']:.4f} mV")
    print("  (slow NMDA accumulation across sender spikes — the substrate of the Wang")
    print("   decision attractor; the full competing-populations network awaits the seam.)")


if __name__ == "__main__":
    main()
