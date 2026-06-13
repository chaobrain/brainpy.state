# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Shared live-NEST drive for the voltage-based ``clopath_synapse`` parity tests.

Unlike the pair-based STDP drive (:mod:`_stdp_drive`), the Clopath rule reads the
**postsynaptic neuron's analog voltages** (``V`` / ``u_bar_plus`` / ``u_bar_minus``)
every step, so the post must be a *real* ``aeif_psc_delta_clopath`` neuron — not a
decoupled ``iaf`` driven to spike. Two drives live here:

* **Neuron voltage parity** (:func:`nest_neuron_trace` / :func:`our_neuron_trace`) —
  a subthreshold constant-current (``I_e``) depolarization, recorded with a
  multimeter. No spikes, so ``V`` and the low-pass filters follow smooth
  trajectories that compare *sample-for-sample* (the post analog states the synapse
  reads are validated in isolation, free of the spike-edge discontinuity).

* **Spike-pairing weight parity** (:func:`nest_pairing_weight` /
  :func:`our_pairing_weight`) — the canonical ``clopath_synapse_spike_pairing.py``
  protocol: a presynaptic ``parrot``/``spike_generator`` relay through a single
  Clopath edge while a strong (80 mV) ``spike_generator`` driver clamps the post to
  ``V_clamp`` at chosen times; the weight is read from NEST's ``weight_recorder`` /
  our ``weight_trace`` after five-or-six pairs.

**NEST divergences encoded here (documented in the spec + CONTEXT.md cluster-07).**

* ``delay_u_bars`` is set to **one resolution step** (0.1 ms), not NEST's 4.0 ms
  default. NEST evaluates LTP/LTD against ring-buffered post voltages delayed by
  ``delay_u_bars``; the substrate reads the post State with its intrinsic one-step
  lag (projections run before neurons), so a one-step ``delay_u_bars`` is the
  apples-to-apples alignment. (The substrate has no analog-state ring buffer, so
  the 4.0 ms default cannot be reproduced online.)
* **Online vs deferred + event-delivery lag.** NEST defers potentiation to the next
  pre ``send`` (summing a decayed-``x_bar`` LTP history), and our spike-generator →
  synapse → neuron delivery lands the post clamp one step later than NEST's. Both
  leave the *stored weight* within a few percent of NEST (LTD near-exact; LTP within
  a documented band that grows with pairing frequency) while the **direction** and
  **frequency-ordering** match exactly — see ``clopath_synapse_parity_test.py``.
"""
import numpy as np

DT = 0.1            # ms, resolution
RELAY_D = 0.1       # spike_generator -> parrot/driver delay (ms) == one step
INIT_W = 0.5        # initial Clopath weight (mV; aeif_psc_delta_clopath is a delta model)
DRIVE_W = 80.0      # post driver weight (mV voltage jump -> forces the V_clamp spike)
DELAY_UBARS = 0.1   # ms; aligned to the substrate's one-step read lag (see module docstring)

# aeif_psc_delta_clopath parameters of the canonical spike-pairing experiment
# (NEST examples/pynest/clopath_synapse_spike_pairing.py), shared by both sides.
NRN_PARAMS = dict(
    V_m=-70.6, E_L=-70.6, C_m=281.0, theta_minus=-70.6, theta_plus=-45.3,
    A_LTD=14.0e-5, A_LTP=8.0e-5, tau_u_bar_minus=10.0, tau_u_bar_plus=7.0,
    delay_u_bars=DELAY_UBARS, a=4.0, b=0.0805, V_reset=-49.6,
    V_clamp=33.0, t_clamp=2.0, t_ref=0.0,
)

# Canonical pre/post spike trains: first five = post-before-pre (depression-leaning),
# last five = pre-before-post (potentiation), at 10/20/30/40/50 Hz pairing rates.
SPIKE_TIMES_PRE = [
    [20.0, 120.0, 220.0, 320.0, 420.0],
    [20.0, 70.0, 120.0, 170.0, 220.0],
    [20.0, 53.3, 86.7, 120.0, 153.3],
    [20.0, 45.0, 70.0, 95.0, 120.0],
    [20.0, 40.0, 60.0, 80.0, 100.0],
    [120.0, 220.0, 320.0, 420.0, 520.0, 620.0],
    [70.0, 120.0, 170.0, 220.0, 270.0, 320.0],
    [53.3, 86.6, 120.0, 153.3, 186.6, 220.0],
    [45.0, 70.0, 95.0, 120.0, 145.0, 170.0],
    [40.0, 60.0, 80.0, 100.0, 120.0, 140.0],
]
SPIKE_TIMES_POST = [
    [10.0, 110.0, 210.0, 310.0, 410.0],
    [10.0, 60.0, 110.0, 160.0, 210.0],
    [10.0, 43.3, 76.7, 110.0, 143.3],
    [10.0, 35.0, 60.0, 85.0, 110.0],
    [10.0, 30.0, 50.0, 70.0, 90.0],
    [130.0, 230.0, 330.0, 430.0, 530.0, 630.0],
    [80.0, 130.0, 180.0, 230.0, 280.0, 330.0],
    [63.3, 96.6, 130.0, 163.3, 196.6, 230.0],
    [55.0, 80.0, 105.0, 130.0, 155.0, 180.0],
    [50.0, 70.0, 90.0, 110.0, 130.0, 150.0],
]
#: Per-train labels (pairing direction + rate); index-aligned with the trains above.
TRAIN_LABELS = [f"post-pre {r}Hz" for r in (10, 20, 30, 40, 50)] + \
               [f"pre-post {r}Hz" for r in (10, 20, 30, 40, 50)]
#: Index range of the pure depression-leaning (post-before-pre) trains.
LTD_TRAINS = range(0, 5)
#: Index range of the potentiation (pre-before-post) trains.
LTP_TRAINS = range(5, 10)


# -- neuron voltage parity (subthreshold dc drive) -------------------------
def nest_neuron_trace(I_e, T, record=("V_m", "u_bar_plus", "u_bar_minus")):
    """Subthreshold-``I_e`` ``aeif_psc_delta_clopath`` multimeter trace in live NEST.

    Returns ``(times, {name: trace})`` for each recorded analog state.
    """
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")
    nrn = nest.Create("aeif_psc_delta_clopath", 1, {**NRN_PARAMS, "I_e": float(I_e)})
    mm = nest.Create("multimeter", params={"record_from": list(record), "interval": DT})
    nest.Connect(mm, nrn)
    nest.Simulate(T)
    ev = mm.get("events")
    return np.asarray(ev["times"]), {name: np.asarray(ev[name]) for name in record}


def our_neuron_trace(I_e, T, record=("V_m", "u_bar_plus", "u_bar_minus")):
    """Subthreshold-``I_e`` ``aeif_psc_delta_clopath`` multimeter trace in brainpy.state."""
    import braintools
    import saiunit as u
    from brainpy_state import Simulator, aeif_psc_delta_clopath, multimeter
    sim = Simulator(dt=DT * u.ms)
    post = _our_clopath_neuron(sim, I_e=I_e)
    mm = sim.create(multimeter, record_from=list(record), interval=DT * u.ms)
    sim.connect(mm, post)
    res = sim.simulate(T * u.ms)
    times = np.asarray(u.get_mantissa(res.times / u.ms))
    traces = {name: np.asarray(u.get_mantissa(res.trace(mm, name)))[:, 0] for name in record}
    return times, traces


# -- spike-pairing weight parity (canonical protocol) ----------------------
def nest_pairing_weight(s_pre, s_post):
    """Final Clopath weight after the canonical pairing protocol, in live NEST."""
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")
    nrn = nest.Create("aeif_psc_delta_clopath", 1, NRN_PARAMS)
    parrot = nest.Create("parrot_neuron", 1)
    sg_pre = nest.Create("spike_generator", 1, {"spike_times": list(s_pre)})
    nest.Connect(sg_pre, parrot, syn_spec={"delay": RELAY_D})
    sg_post = nest.Create("spike_generator", 1, {"spike_times": list(s_post)})
    nest.Connect(sg_post, nrn, syn_spec={"delay": RELAY_D, "weight": DRIVE_W})
    wr = nest.Create("weight_recorder", 1)
    nest.CopyModel("clopath_synapse", "clopath_synapse_rec", {"weight_recorder": wr})
    nest.Connect(parrot, nrn, syn_spec={"synapse_model": "clopath_synapse_rec",
                                        "weight": INIT_W, "delay": RELAY_D})
    nest.Simulate(10.0 + max(s_pre[-1], s_post[-1]))
    return float(wr.get("events")["weights"][-1])


def our_pairing_weight(s_pre, s_post):
    """Final Clopath weight after the canonical pairing protocol, in brainpy.state."""
    import saiunit as u
    from brainpy_state import (Simulator, spike_generator, clopath_synapse,
                               static_synapse)
    sim = Simulator(dt=DT * u.ms)
    post = _our_clopath_neuron(sim)
    sg_pre = sim.create(spike_generator, spike_times=np.asarray(s_pre) * u.ms)
    sg_post = sim.create(spike_generator, spike_times=np.asarray(s_post) * u.ms)
    sim.connect(sg_post, post, synapse=static_synapse(weight=DRIVE_W * u.mV), delay=RELAY_D * u.ms)
    proj = sim.connect(sg_pre, post, synapse=clopath_synapse(weight=INIT_W * u.mV),
                       delay=RELAY_D * u.ms)
    sim.record_weight(proj)
    res = sim.simulate((10.0 + max(s_pre[-1], s_post[-1])) * u.ms)
    return float(u.get_mantissa(res.weight_trace(proj))[-1, 0])


def nest_pairing_weights_full(s_pre, s_post):
    """Full ``weight_recorder`` series for the canonical pairing protocol, in live NEST.

    Like :func:`nest_pairing_weight` but returns **every** logged send event (one per
    pre spike), not just the final weight — the input the send-event audit needs to
    check event count and timing as well as value.

    Parameters
    ----------
    s_pre, s_post : sequence of float
        Presynaptic / postsynaptic-driver spike times (ms).

    Returns
    -------
    weights : ndarray
        Weight at each ``weight_recorder`` send, ordered by send time.
    wr_times : ndarray
        The send times (ms), ordered ascending. NEST stamps each at the pre-spike
        emission step (``s_pre`` relayed by ``RELAY_D`` through the parrot).
    """
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")
    nrn = nest.Create("aeif_psc_delta_clopath", 1, NRN_PARAMS)
    parrot = nest.Create("parrot_neuron", 1)
    sg_pre = nest.Create("spike_generator", 1, {"spike_times": list(s_pre)})
    nest.Connect(sg_pre, parrot, syn_spec={"delay": RELAY_D})
    sg_post = nest.Create("spike_generator", 1, {"spike_times": list(s_post)})
    nest.Connect(sg_post, nrn, syn_spec={"delay": RELAY_D, "weight": DRIVE_W})
    wr = nest.Create("weight_recorder", 1)
    nest.CopyModel("clopath_synapse", "clopath_synapse_full_rec", {"weight_recorder": wr})
    nest.Connect(parrot, nrn, syn_spec={"synapse_model": "clopath_synapse_full_rec",
                                        "weight": INIT_W, "delay": RELAY_D})
    nest.Simulate(10.0 + max(s_pre[-1], s_post[-1]))
    ev = wr.get("events")
    order = np.argsort(np.asarray(ev["times"]))
    return np.asarray(ev["weights"])[order], np.asarray(ev["times"])[order]


def our_pairing_weight_trace(s_pre, s_post):
    """Full per-step Clopath ``weight_trace`` for the pairing protocol, in brainpy.state.

    Like :func:`our_pairing_weight` but returns the whole per-step weight trajectory
    (single edge, ``(T,)``) rather than just its last value, so the send-event audit
    can mask it at the send steps via the send-view seam.

    Parameters
    ----------
    s_pre, s_post : sequence of float
        Presynaptic / postsynaptic-driver spike times (ms).

    Returns
    -------
    ndarray
        Post-update stored weight at every step (``(T,)``; bare mV mantissa).
    """
    import saiunit as u
    from brainpy_state import (Simulator, spike_generator, clopath_synapse,
                               static_synapse)
    sim = Simulator(dt=DT * u.ms)
    post = _our_clopath_neuron(sim)
    sg_pre = sim.create(spike_generator, spike_times=np.asarray(s_pre) * u.ms)
    sg_post = sim.create(spike_generator, spike_times=np.asarray(s_post) * u.ms)
    sim.connect(sg_post, post, synapse=static_synapse(weight=DRIVE_W * u.mV), delay=RELAY_D * u.ms)
    proj = sim.connect(sg_pre, post, synapse=clopath_synapse(weight=INIT_W * u.mV),
                       delay=RELAY_D * u.ms)
    sim.record_weight(proj)
    res = sim.simulate((10.0 + max(s_pre[-1], s_post[-1])) * u.ms)
    return np.asarray(u.get_mantissa(res.weight_trace(proj)))[:, 0]


# -- voltage-clamp LTD sanity (sustained subthreshold depolarization) ------
def nest_clamp_weight(I_e, s_pre, T):
    """Pre spikes onto a post held subthreshold by ``I_e`` (no post driver), live NEST.

    A constant ``I_e`` holds ``V`` between ``theta_minus`` and ``theta_plus`` (the
    depression band: ``u_bar_minus > theta_minus`` but ``V < theta_plus``), so each
    presynaptic spike depresses the weight and none potentiate it — the pure
    voltage-gated LTD regime.
    """
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")
    nrn = nest.Create("aeif_psc_delta_clopath", 1, {**NRN_PARAMS, "I_e": float(I_e)})
    parrot = nest.Create("parrot_neuron", 1)
    sg_pre = nest.Create("spike_generator", 1, {"spike_times": list(s_pre)})
    nest.Connect(sg_pre, parrot, syn_spec={"delay": RELAY_D})
    wr = nest.Create("weight_recorder", 1)
    nest.CopyModel("clopath_synapse", "clopath_synapse_clamp_rec", {"weight_recorder": wr})
    nest.Connect(parrot, nrn, syn_spec={"synapse_model": "clopath_synapse_clamp_rec",
                                        "weight": INIT_W, "delay": RELAY_D})
    nest.Simulate(T)
    return float(wr.get("events")["weights"][-1])


def our_clamp_weight(I_e, s_pre, T):
    """Pre spikes onto a post held subthreshold by ``I_e`` (no post driver), brainpy.state."""
    import saiunit as u
    from brainpy_state import Simulator, spike_generator, clopath_synapse
    sim = Simulator(dt=DT * u.ms)
    post = _our_clopath_neuron(sim, I_e=I_e)
    sg_pre = sim.create(spike_generator, spike_times=np.asarray(s_pre) * u.ms)
    proj = sim.connect(sg_pre, post, synapse=clopath_synapse(weight=INIT_W * u.mV),
                       delay=RELAY_D * u.ms)
    sim.record_weight(proj)
    res = sim.simulate(T * u.ms)
    return float(u.get_mantissa(res.weight_trace(proj))[-1, 0])


def _our_clopath_neuron(sim, I_e=0.0, n=1):
    """Build ``n`` shared ``aeif_psc_delta_clopath`` neurons with the canonical parameters.

    Default ``n = 1`` is the single post used by the pairing / voltage drives; the
    small-network example passes ``n > 1`` to build the recurrent population (every
    neuron carries the identical canonical parameters and starts at ``E_L``).
    """
    import braintools
    import saiunit as u
    from brainpy_state import aeif_psc_delta_clopath
    return sim.create(
        aeif_psc_delta_clopath, n,
        E_L=-70.6 * u.mV, V_peak=33.0 * u.mV, C_m=281.0 * u.pF, theta_minus=-70.6 * u.mV,
        theta_plus=-45.3 * u.mV, A_LTD=14.0e-5, A_LTP=8.0e-5, tau_u_bar_minus=10.0 * u.ms,
        tau_u_bar_plus=7.0 * u.ms, delay_u_bars=DELAY_UBARS * u.ms, a=4.0 * u.nS, b=0.0805 * u.pA,
        V_reset=-49.6 * u.mV, V_clamp=33.0 * u.mV, t_clamp=2.0 * u.ms, t_ref=0.0 * u.ms,
        I_e=I_e * u.pA, V_initializer=braintools.init.Constant(-70.6 * u.mV))
