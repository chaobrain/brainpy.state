# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Shared live-NEST drive for pair-based STDP weight-trajectory parity tests.

Both sides drive a *single* plastic edge with deterministic pre/post spike
trains and read the synaptic weight at every pre ``send``; the per-model parity
tests assert the two trajectories agree. Factored here because every pair-based
STDP model (``stdp_synapse``, ``stdp_synapse_hom``, ``stdp_pl_synapse_hom``,
``jonke_synapse``, ``vogels_sprekeler_synapse``) shares the exact same routing —
only the synapse parameters and tolerances differ. See
``stdp_synapse_parity_test.py`` for the full rationale; the essentials:

* **Decoupled archiving post.** The post must maintain the ``tau_minus`` ``K-``
  history (an ``ArchivingNode``) **and** fire only at chosen times. A
  ``parrot_neuron`` post would relay the STDP-delivered pre spikes into its own
  archive (phantom posts → phantom facilitation), so the post is an
  ``iaf_psc_delta`` forced to spike by a strong suprathreshold driver while the
  plastic EPSP stays subthreshold.
* **Recorded fire times.** Pre/post fire times are recorded with spike recorders
  and replayed on the brainpy side, so the two sides share event times exactly.
* **Dendritic-delay shift.** A NEST STDP synapse of delay ``d`` behaves as if each
  post spike's effect occurs at ``q + d`` (potentiation pairs the pre trace with
  the post-at-synapse arrival; depression reads ``K-`` at ``t_pre - d``). The
  substrate has no dendritic seam, so each post spike is injected ``d`` later on
  the brainpy timeline and the substrate's axonal :class:`InputDelay` is disabled
  (``rule.delay = None``; the kernel never reads it).
* **Online vs deferred.** The substrate potentiates eagerly on post steps and
  depresses on pre steps; NEST defers potentiation to the next ``send``. The
  cumulative op set/order is identical at every send, so the trajectories
  coincide at the pre-send steps — exactly where the weight is sampled.
"""
import numpy as np

DT = 0.1            # ms, resolution
PRE_D = 1.0         # spike_generator -> pre parrot relay delay (ms)
POST_D = 1.0        # spike_generator -> post driver delay (ms)
DEND_D = 1.0        # pre parrot -> synapse dendritic delay (ms); the STDP shift
DRIVE_W = 1e6       # suprathreshold driver weight (forces a post spike)

# Strong-driven archiving post: fires only from the driver; STDP stays subthreshold.
POST_PARAMS = dict(C_m=1.0, tau_m=10.0, t_ref=0.1, E_L=0.0, V_reset=0.0,
                   V_m=0.0, V_th=1e4)


def steps(times_ms):
    """Integer step indices for a sequence of millisecond times."""
    return np.asarray([int(round(t / DT)) for t in times_ms], dtype=int)


def bp_weight_trace(rule, pre_fire, post_fire, n_steps, *, dend_d=DEND_D, delivered=False):
    """Drive a single 0->0 plastic edge; return the per-step trace.

    ``pre_fire`` / ``post_fire`` are recorded NEST fire times (ms). Pre acts at
    ``p``; each post's effect is injected at ``q + dend_d`` (NEST dendritic-delay
    convention). ``rule.delay`` is forced to ``None`` so the substrate's axonal
    :class:`InputDelay` does not shift the pre timing (the weight does not observe
    EPSP delivery). The sweep runs inside ``for_loop`` (the per-step spike is a
    scanned arg) so the loop JIT-compiles once.

    With ``delivered=False`` (default) the per-step value is the stored synaptic
    weight ``proj.weight.value[0]`` — the right observable for the STDP rules,
    which mutate the weight. With ``delivered=True`` it is the per-step *delivered
    amplitude* (the rule's ``w_eff``, gated by the pre spike) — the right
    observable for depression-pool rules like ``ht_synapse`` whose stored weight
    is static and whose ``weight_recorder`` logs ``w * P``. Sample this at the
    pre-send steps (it is ``0`` off pre-spike steps).
    """
    import brainstate
    import jax.numpy as jnp
    import saiunit as u
    from brainstate import transform
    from brainpy_state._network import EventPlasticProj

    rule.delay = None
    box = {'pre': jnp.zeros(1), 'post': jnp.zeros(1)}
    proj = EventPlasticProj(
        pre_spike=lambda: box['pre'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=None, post_local_idx=jnp.arange(1), n_post_pop=1,
        post_spike=lambda: box['post'],
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)

    pre = np.zeros((n_steps, 1))
    pre[steps(pre_fire), 0] = 1.0                        # pre effect at p
    post = np.zeros((n_steps, 1))
    if len(post_fire):
        post[steps([q + dend_d for q in post_fire]), 0] = 1.0   # post effect at q + d
    pre = jnp.asarray(pre)
    post = jnp.asarray(post)
    times = jnp.arange(n_steps) * DT * u.ms
    indices = jnp.arange(n_steps)

    def step(t, i, xp, xq):
        box['pre'] = xp
        box['post'] = xq
        with brainstate.environ.context(t=t, i=i):
            contrib = proj.update()
            if delivered:
                return u.get_mantissa(contrib)[0]
            return proj.weight.value[0]

    return np.asarray(transform.for_loop(step, times, indices, pre, post))


def nest_pair_run(synapse_model, per_conn, post_tau_minus, pre_want, post_want, T,
                  *, common=None, post_params=None):
    """Run the decoupled-iaf NEST drive; return recorded times and weights.

    Parameters
    ----------
    synapse_model : str
        NEST plastic synapse model name (e.g. ``"stdp_synapse"``).
    per_conn : dict
        Per-connection synapse parameters (``weight`` plus any per-synapse
        plasticity params). ``delay`` is added automatically.
    post_tau_minus : float
        ``tau_minus`` (ms) for the postsynaptic archiving node (the ``K-`` trace).
    pre_want, post_want : sequence of float
        Desired pre / post fire times (ms). Realized times are recorded.
    T : float
        Simulation horizon (ms).
    common : dict, optional
        Common (homogeneous) synapse properties set via ``CopyModel`` defaults
        (e.g. the shared params of a ``*_hom`` model). The ``weight_recorder`` is
        always added here.
    post_params : dict, optional
        Extra postsynaptic node params (e.g. ``tau_minus_triplet`` for the triplet
        model). Merged into :data:`POST_PARAMS`.

    Returns
    -------
    pre_fire, post_fire : ndarray
        Sorted recorded pre / post fire times (ms).
    weights, wr_times : ndarray
        Weight-recorder weights and their send times, ordered by time.
    """
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")
    parrot_pre = nest.Create("parrot_neuron")
    pp = {**POST_PARAMS, **(post_params or {}), "tau_minus": post_tau_minus}
    post = nest.Create("iaf_psc_delta", params=pp)
    sg_pre = nest.Create("spike_generator",
                         params={"spike_times": sorted(round(p - PRE_D, 4) for p in pre_want)})
    nest.Connect(sg_pre, parrot_pre, syn_spec={"delay": PRE_D})
    if len(post_want):
        sg_post = nest.Create("spike_generator",
                              params={"spike_times": sorted(round(q - POST_D, 4) for q in post_want)})
        nest.Connect(sg_post, post, syn_spec={"weight": DRIVE_W, "delay": POST_D})
    rec_pre = nest.Create("spike_recorder")
    rec_post = nest.Create("spike_recorder")
    nest.Connect(parrot_pre, rec_pre)
    nest.Connect(post, rec_post)
    wr = nest.Create("weight_recorder")
    model = synapse_model + "_rec"
    nest.CopyModel(synapse_model, model, {"weight_recorder": wr, **(common or {})})
    nest.Connect(parrot_pre, post, syn_spec={"synapse_model": model, "delay": DEND_D, **per_conn})
    nest.Simulate(T)
    pre_fire = np.sort(np.asarray(rec_pre.get("events")["times"]))
    post_fire = np.sort(np.asarray(rec_post.get("events")["times"]))
    ev = wr.get("events")
    order = np.argsort(np.asarray(ev["times"]))
    return pre_fire, post_fire, np.asarray(ev["weights"])[order], np.asarray(ev["times"])[order]
