# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Shared live-NEST drive for the dendritic ``urbanczik_synapse`` parity tests.

Like the Clopath rule (:mod:`_clopath_drive`), Urbanczik-Senn plasticity reads a
**postsynaptic analog state** every step — here the dendritic compartment's
*prediction error* ``delta_Pi`` (and the star potential ``V_W_star`` it derives
from), exposed by the two-compartment ``pp_cond_exp_mc_urbanczik`` neuron. So the
post must be a *real* point-process multi-compartment neuron, and two drives live
here:

* **Neuron / dendrite parity** (:func:`nest_neuron_trace` / :func:`our_neuron_trace`)
  — a dendritic excitatory spike train into a neuron whose **soma is held
  subthreshold** by a hyperpolarising ``soma_I_e`` (so it never fires: the rule's
  ``delta_Pi`` is then the deterministic ``-phi(V_W*)*dt*h(V_W*)`` branch). The
  dendritic voltage ``V_d`` (NEST ``V_m.p``) and somatic voltage ``V_s``
  (``V_m.s``) compare sample-for-sample; ``V_W_star`` / ``delta_Pi`` are validated
  for internal consistency against ``V_d`` (the deterministic neuron functions),
  and against NEST transitively through the weight trajectory below.

* **Weight-trajectory parity** (:func:`nest_weight_traj` / :func:`our_weight_traj`)
  — the same subthreshold-soma regime with a *plastic* dendritic edge: a
  presynaptic train depresses the weight (``delta_Pi < 0`` because the dendrite
  predicts a higher rate than the silent soma emits). NEST relays
  ``spike_generator -> parrot -> urbanczik_synapse`` and logs the
  ``weight_recorder``; we drive ``spike_generator -> urbanczik_synapse`` directly
  (the substrate, unlike NEST, allows a generator as a plastic-edge source) and
  read the per-step ``weight_trace``.

**NEST divergences encoded here (documented in the spec + CONTEXT.md cluster-21).**

* ``delay`` is one resolution step (``RELAY_D = 0.1 ms``). NEST integrates
  ``delta_Pi`` over the dendritic-delay window ``(t_last - d, t_spike - d]`` on each
  presynaptic ``send``; the substrate reads the post ``delta_Pi`` State with its
  intrinsic one-step lag (projections run before neurons), so a one-step delay is
  the apples-to-apples alignment.
* **Online (every-step) vs event-driven (every-send) weight.** NEST recomputes the
  stored weight only on a presynaptic ``send`` (piecewise-constant, frozen between
  spikes); the substrate accumulates the same integral every grid step, so the
  weight State drifts *between* spikes and re-synchronises with NEST at every
  presynaptic spike. The two therefore **coincide at the send steps** (where NEST's
  ``weight_recorder`` samples and where the delivered weight is used) — that is the
  comparison :func:`sample_at_send_steps` performs. Comparing the raw final values
  (NEST frozen since the last spike vs ours drifted on) is *not* apples-to-apples.
* **Soma decoupled from the dendrite.** With ``g_ps = 0`` the dendrite receives no
  somatic current, so ``soma_I_e`` moves only ``V_s`` (and the somatic firing
  probability) — ``V_d``, ``V_W_star`` and ``delta_Pi`` are invariant to it. That is
  what lets the soma be clamped silent without perturbing the learning signal.
"""
import numpy as np

DT = 0.1            # ms, resolution
RELAY_D = 0.1       # ms; spike_generator -> parrot / synapse delay == one step
SOMA_IE = -12000.0  # pA; strong hyperpolarising somatic current -> soma held silent.
                    # The soma-dendrite coupling g_sp (600 nS) >> g_L (30 nS) pins V_s
                    # near V_W_star during dendritic drive; the current shifts V_s down by
                    # I_e/(g_L+g_sp) ~ 19 mV without touching V_d/V_W_star/delta_Pi/weight
                    # (g_ps = 0 decouples the dendrite), making the soma robustly silent.

# urbanczik_synapse_example parameters, shared by both sides.
INIT_W = 90.0       # pA  (0.3 * C_m)
WMAX = 1350.0       # pA  (4.5 * C_m)
WMIN = 0.0          # pA
ETA = 0.17
TAU_DELTA = 100.0   # ms

# -- neuron / dendrite parity drive ----------------------------------------
DEND_W = 300.0      # pA, dendritic excitatory drive weight (static, non-plastic)
NEURON_DEND_SPIKES = [10.0, 20.0, 30.0, 40.0, 60.0, 80.0]  # ms
T_NEURON = 120.0    # ms

# -- weight-trajectory parity drive ----------------------------------------
WEIGHT_PRE_SPIKES = list(np.arange(10.0, 186.0, 5.0))  # 10,15,...,185 ms (36 spikes)
T_WEIGHT = 190.0    # ms (captures the last send at 185 + RELAY_D)

# pp_cond_exp_mc_urbanczik defaults shared by both sides (NEST nested params).
PHI_MAX, RATE_SLOPE, BETA, THETA = 0.15, 0.5, 1.0 / 3.0, -55.0
G_SP = 600.0        # nS soma-dendrite coupling
SOMA_C_M = DEND_C_M = 300.0   # pF
SOMA_G_L = DEND_G_L = 30.0    # nS
E_L = -70.0         # mV
TAU_SYN = 3.0       # ms (soma & dendrite, ex & in)


def phi(V):
    """NEST ``pp_cond_exp_mc_urbanczik`` rate function ``phi(u)`` (kHz)."""
    return PHI_MAX / (1.0 + RATE_SLOPE * np.exp(BETA * (THETA - V)))


def h(V):
    """NEST ``pp_cond_exp_mc_urbanczik`` ``h(u) = 15*beta / (1 + (1/slope) e^{-beta(theta-u)})``."""
    return 15.0 * BETA / (1.0 + (1.0 / RATE_SLOPE) * np.exp(-BETA * (THETA - V)))


def v_w_star(V_d):
    """Star potential ``(E_L g_L + V_d g_sp) / (g_L + g_sp)`` (mV) from the dendrite."""
    return (E_L * SOMA_G_L + V_d * G_SP) / (SOMA_G_L + G_SP)


def delta_pi_silent(V_d):
    """Deterministic ``delta_Pi`` for a silent soma: ``(0 - phi(V_W*)*dt) * h(V_W*)``."""
    vws = v_w_star(V_d)
    return (0.0 - phi(vws) * DT) * h(vws)


def _nest_params(soma_I_e=0.0):
    """NEST nested ``pp_cond_exp_mc_urbanczik`` params matching our neuron defaults."""
    return {
        't_ref': 3.0, 'g_sp': G_SP,
        'phi_max': PHI_MAX, 'rate_slope': RATE_SLOPE, 'beta': BETA, 'theta': THETA,
        'soma': {'V_m': E_L, 'C_m': SOMA_C_M, 'E_L': E_L, 'g_L': SOMA_G_L,
                 'E_ex': 0.0, 'E_in': -75.0, 'tau_syn_ex': TAU_SYN, 'tau_syn_in': TAU_SYN,
                 'I_e': float(soma_I_e)},
        'dendritic': {'V_m': E_L, 'C_m': DEND_C_M, 'E_L': E_L, 'g_L': DEND_G_L,
                      'tau_syn_ex': TAU_SYN, 'tau_syn_in': TAU_SYN},
    }


def _our_urbanczik_neuron(sim, soma_I_e=0.0, n=1):
    """Build ``n`` ``pp_cond_exp_mc_urbanczik`` neurons with the shared parameters."""
    import jax
    import saiunit as u
    from brainpy_state import pp_cond_exp_mc_urbanczik
    return sim.create(
        pp_cond_exp_mc_urbanczik, n,
        t_ref=3.0 * u.ms, g_sp=G_SP * u.nS, g_ps=0.0 * u.nS,
        phi_max=PHI_MAX, rate_slope=RATE_SLOPE, beta=BETA, theta=THETA,
        soma_C_m=SOMA_C_M * u.pF, soma_g_L=SOMA_G_L * u.nS, soma_E_L=E_L * u.mV,
        soma_E_ex=0.0 * u.mV, soma_E_in=-75.0 * u.mV,
        soma_tau_syn_ex=TAU_SYN * u.ms, soma_tau_syn_in=TAU_SYN * u.ms,
        soma_I_e=float(soma_I_e) * u.pA,
        dend_C_m=DEND_C_M * u.pF, dend_g_L=DEND_G_L * u.nS, dend_E_L=E_L * u.mV,
        dend_tau_syn_ex=TAU_SYN * u.ms, dend_tau_syn_in=TAU_SYN * u.ms,
        rng_key=jax.random.PRNGKey(0))


# -- neuron / dendrite parity ----------------------------------------------
def nest_neuron_trace():
    """Dendritic-drive ``V_m.p`` / ``V_m.s`` multimeter trace in live NEST.

    Returns ``(times, {'V_d': .., 'V_s': ..})`` and the somatic spike count.
    """
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")
    nrn = nest.Create("pp_cond_exp_mc_urbanczik", params=_nest_params(soma_I_e=SOMA_IE))
    syns = nest.GetDefaults("pp_cond_exp_mc_urbanczik")["receptor_types"]
    sg = nest.Create("spike_generator", params={"spike_times": list(NEURON_DEND_SPIKES)})
    nest.Connect(sg, nrn, syn_spec={"receptor_type": syns["dendritic_exc"],
                                    "weight": DEND_W, "delay": RELAY_D})
    mm = nest.Create("multimeter", params={"record_from": ["V_m.p", "V_m.s"], "interval": DT})
    nest.Connect(mm, nrn)
    sr = nest.Create("spike_recorder")
    nest.Connect(nrn, sr)
    nest.Simulate(T_NEURON)
    ev = mm.get("events")
    n_spikes = len(sr.get("events", "times"))
    return (np.asarray(ev["times"]),
            {"V_d": np.asarray(ev["V_m.p"]), "V_s": np.asarray(ev["V_m.s"])}, n_spikes)


def our_neuron_trace():
    """Dendritic-drive trace in brainpy.state (``V_d``/``V_s``/``V_W_star``/``delta_Pi``)."""
    import saiunit as u
    from brainpy_state import Simulator, spike_generator, spike_recorder, multimeter
    sim = Simulator(dt=DT * u.ms)
    post = _our_urbanczik_neuron(sim, soma_I_e=SOMA_IE)
    sg = sim.create(spike_generator, spike_times=np.asarray(NEURON_DEND_SPIKES) * u.ms)
    sim.connect(sg, post, weight=DEND_W * u.pA, receptor_type=3, delay=RELAY_D * u.ms)
    mm = sim.create(multimeter, record_from=["V_d", "V_s", "V_W_star", "delta_Pi"], interval=DT * u.ms)
    sim.connect(mm, post)
    sr = sim.create(spike_recorder)
    sim.connect(post, sr)
    res = sim.simulate(T_NEURON * u.ms)
    times = np.asarray(u.get_mantissa(res.times / u.ms))
    traces = {
        "V_d": np.asarray(u.get_mantissa(res.trace(mm, "V_d") / u.mV))[:, 0],
        "V_s": np.asarray(u.get_mantissa(res.trace(mm, "V_s") / u.mV))[:, 0],
        "V_W_star": np.asarray(u.get_mantissa(res.trace(mm, "V_W_star") / u.mV))[:, 0],
        "delta_Pi": np.asarray(u.get_mantissa(res.trace(mm, "delta_Pi")))[:, 0],
    }
    n_spikes = int(np.asarray(res.spikes(sr)).sum())
    return times, traces, n_spikes


# -- weight-trajectory parity ----------------------------------------------
def nest_weight_traj():
    """Depression weight trajectory in live NEST: ``(send_times, weights, n_soma_spikes)``."""
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")
    nrn = nest.Create("pp_cond_exp_mc_urbanczik", params=_nest_params(soma_I_e=SOMA_IE))
    syns = nest.GetDefaults("pp_cond_exp_mc_urbanczik")["receptor_types"]
    parrot = nest.Create("parrot_neuron", 1)
    sg = nest.Create("spike_generator", params={"spike_times": list(WEIGHT_PRE_SPIKES)})
    nest.Connect(sg, parrot, syn_spec={"delay": RELAY_D})
    wr = nest.Create("weight_recorder")
    nest.CopyModel("urbanczik_synapse", "urbanczik_synapse_rec", {"weight_recorder": wr[0]})
    nest.Connect(parrot, nrn, syn_spec={"synapse_model": "urbanczik_synapse_rec",
                                        "receptor_type": syns["dendritic_exc"],
                                        "weight": INIT_W, "Wmax": WMAX, "Wmin": WMIN,
                                        "eta": ETA, "tau_Delta": TAU_DELTA, "delay": RELAY_D})
    sr = nest.Create("spike_recorder")
    nest.Connect(nrn, sr)
    nest.Simulate(T_WEIGHT)
    ev = wr.get("events")
    order = np.argsort(np.asarray(ev["times"]))
    n_spikes = len(sr.get("events", "times"))
    return np.asarray(ev["times"])[order], np.asarray(ev["weights"])[order], n_spikes


def our_weight_traj():
    """Depression weight trajectory in brainpy.state: ``(times, per_step_weights, n_soma_spikes)``."""
    import saiunit as u
    from brainpy_state import Simulator, spike_generator, spike_recorder, urbanczik_synapse
    sim = Simulator(dt=DT * u.ms)
    post = _our_urbanczik_neuron(sim, soma_I_e=SOMA_IE)
    sg = sim.create(spike_generator, spike_times=np.asarray(WEIGHT_PRE_SPIKES) * u.ms)
    proj = sim.connect(
        sg, post,
        synapse=urbanczik_synapse(weight=INIT_W * u.pA, Wmax=WMAX, Wmin=WMIN, eta=ETA,
                                  tau_Delta=TAU_DELTA * u.ms, delay=RELAY_D * u.ms),
        receptor_type=3)
    sim.record_weight(proj)
    sr = sim.create(spike_recorder)
    sim.connect(post, sr)
    res = sim.simulate(T_WEIGHT * u.ms)
    times = np.asarray(u.get_mantissa(res.times / u.ms))
    weights = np.asarray(u.get_mantissa(res.weight_trace(proj)))[:, 0]
    n_spikes = int(np.asarray(res.spikes(sr)).sum())
    return times, weights, n_spikes


def sample_at_send_steps(our_times, our_weights, send_times):
    """Sample our per-step ``weight_trace`` at the grid step nearest each NEST send time.

    NEST logs the stored weight only on a presynaptic ``send``; the substrate's
    continuous weight coincides with it at those steps (see the module docstring).
    Send times fall exactly on the ``DT`` grid, so the nearest-step read is exact.
    """
    idx = np.searchsorted(our_times, send_times)
    idx = np.clip(idx, 0, len(our_weights) - 1)
    # snap to the closer of the two straddling grid steps
    lo = np.clip(idx - 1, 0, len(our_weights) - 1)
    pick = np.where(np.abs(our_times[idx] - send_times) <= np.abs(our_times[lo] - send_times), idx, lo)
    return our_weights[pick]
