# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Shared live-NEST drive for the dopamine-modulated STDP parity tests (cluster-08).

The ``stdp_dopamine_synapse`` couples a per-edge eligibility trace ``c`` (built by
spike-timing pairing, exactly the pair-based STDP machinery of :mod:`_stdp_drive`)
to a **broadcast** dopamine concentration ``n(t)`` maintained by a
:class:`~brainpy_state._nest.volume_transmitter`. Parity is therefore layered, and
the **upstream node is validated first**:

1. **VT concentration parity (precondition).** A known dopaminergic spike train is
   relayed into a ``volume_transmitter`` and the resulting ``n(t)`` is read back —
   in NEST off a probe ``stdp_dopamine_synapse`` (``GetConnections(...).get('n')``),
   in brainpy.state off the broadcast :class:`HiddenState`. Both are the pure
   ``update_dopamine_`` recursion ``n <- n e^{-dt/tau_n} + count/tau_n``, so they
   agree to machine precision (and cross-check against a closed-form recursion).
   Validate the broadcast *source* before trusting the weight trajectory.

2. **Weight-trajectory parity.** A single dopamine edge is driven with
   deterministic pre/post/dopa trains; the stored weight is read at every pre
   ``send`` (NEST ``weight_recorder``) and compared to our per-step online integral
   sampled at the same steps. NEST integrates the weight lazily (at ``send`` /
   ``trigger_update_weight`` times, processing the relayed dopa history); our kernel
   integrates every step with the broadcast ``n`` (one-step lag). The cumulative
   ``dw/dt = c(t)(n(t)-b)`` integral coincides at the send steps within a documented
   band, and the **direction** and **ordering** match exactly (the cluster-07 posture).

**Timing alignment (the crux).** A dopaminergic ``spike_generator`` spike at ``s``
relayed ``sg -> parrot -> vt`` (two ``DT`` steps) makes ``n`` jump to ``1/tau_n`` at
``s + DOPA_RELAY``; our broadcast node, fed the same train one step earlier, jumps at
the identical wall-clock step. The drive expresses dopa events by their intended
**VT-arrival time** ``D`` and offsets each side so both ``n(t)`` jump at ``D``.

**Shared STDP conventions (inherited from :mod:`_stdp_drive`).** A decoupled
``iaf_psc_delta`` post (``V_th = 1e4``, forced to spike by a strong driver) keeps the
``tau_minus`` ``K-`` archive without relaying STDP-delivered pre spikes; pre/post fire
times are recorded and replayed on the brainpy side; each post spike's effect is
injected ``DEND_D`` later (the dendritic-delay shift) and the substrate's axonal
:class:`InputDelay` is disabled (``rule.delay = None``).
"""
import numpy as np

DT = 0.1            # ms, resolution
PRE_D = 1.0         # spike_generator -> pre parrot relay delay (ms)
POST_D = 1.0        # spike_generator -> post driver delay (ms)
DEND_D = 1.0        # pre parrot -> synapse dendritic delay (ms); the STDP shift
DOPA_RELAY = 0.2    # dopa spike_generator -> parrot -> volume_transmitter (two DT steps)
DRIVE_W = 1e6       # suprathreshold driver weight (forces a post spike)

#: Strong-driven archiving post: fires only from the driver; STDP stays subthreshold.
POST_PARAMS = dict(C_m=1.0, tau_m=10.0, t_ref=0.1, E_L=0.0, V_reset=0.0,
                   V_m=0.0, V_th=1e4)


def steps(times_ms):
    """Integer step indices for a sequence of millisecond times."""
    return np.asarray([int(round(t / DT)) for t in times_ms], dtype=int)


# ==========================================================================
# 1. volume_transmitter concentration n(t)
# ==========================================================================
def analytic_vt_n_trace(dopa_arrival, T, tau_n, *, sample_dt=1.0):
    """Closed-form ``update_dopamine_`` recursion, sampled every ``sample_dt`` ms.

    ``n`` jumps to ``1/tau_n`` at each VT-arrival step (no decay applied on the
    jump step) and decays by ``exp(-DT/tau_n)`` every step thereafter — exactly
    NEST's recursion and our :meth:`volume_transmitter._advance`.

    Parameters
    ----------
    dopa_arrival : sequence of float
        Intended dopa VT-arrival times (ms): the steps at which ``n`` jumps.
    T : float
        Horizon (ms).
    tau_n : float
        Concentration time constant (ms).
    sample_dt : float, optional
        Sampling period (ms). Default ``1.0``.

    Returns
    -------
    times : ndarray
        Sample times (ms).
    n : ndarray
        Concentration at each sample time.
    """
    n_steps = int(round(T / DT))
    arrival = set(int(s) for s in steps(dopa_arrival))
    decay = np.exp(-DT / tau_n)
    full = np.zeros(n_steps + 1)
    n = 0.0
    for k in range(1, n_steps + 1):
        n = n * decay + (1.0 / tau_n if k in arrival else 0.0)
        full[k] = n
    sample = steps(np.arange(sample_dt, T + 0.5 * sample_dt, sample_dt))
    sample = sample[sample <= n_steps]
    return sample * DT, full[sample]


def nest_vt_n_trace(dopa_arrival, T, tau_n, *, sample_dt=1.0):
    """Live-NEST ``volume_transmitter`` ``n(t)`` read off a probe dopamine synapse.

    A dopa ``spike_generator -> parrot -> volume_transmitter`` pool relays the train;
    a passive ``stdp_dopamine_synapse`` bound to the transmitter exposes ``n`` via
    ``GetConnections(...).get('n')``, sampled at ``sample_dt`` chunk boundaries.

    Parameters
    ----------
    dopa_arrival : sequence of float
        Intended dopa VT-arrival times (ms). The generator is placed
        ``DOPA_RELAY`` earlier so ``n`` jumps at each ``D``.
    T : float
        Horizon (ms).
    tau_n : float
        Concentration time constant (ms); set on the probe synapse.
    sample_dt : float, optional
        Sampling period (ms). Default ``1.0``.

    Returns
    -------
    times : ndarray
        Sample times (ms).
    n : ndarray
        NEST concentration at each sample time.
    """
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")
    vt = nest.Create("volume_transmitter")
    sg = nest.Create("spike_generator",
                     params={"spike_times": sorted(round(d - DOPA_RELAY, 4) for d in dopa_arrival)})
    par = nest.Create("parrot_neuron")
    nest.Connect(sg, par, syn_spec={"delay": DT})
    nest.Connect(par, vt, syn_spec={"delay": DT})
    pre = nest.Create("parrot_neuron")
    post = nest.Create("parrot_neuron")
    nest.CopyModel("stdp_dopamine_synapse", "dopa_probe",
                   {"volume_transmitter": vt, "tau_n": float(tau_n)})
    nest.Connect(pre, post, syn_spec={"synapse_model": "dopa_probe"})
    conn = nest.GetConnections(pre, post, synapse_model="dopa_probe")
    times, ns = [], []
    n_chunks = int(round(T / sample_dt))
    for _ in range(n_chunks):
        nest.Simulate(sample_dt)
        times.append(round(nest.biological_time, 4))
        ns.append(float(conn.get("n")))
    return np.asarray(times), np.asarray(ns)


def our_vt_n_trace(dopa_arrival, T, tau_n, *, sample_dt=1.0):
    """brainpy.state broadcast ``volume_transmitter`` ``n(t)``.

    A dopa ``spike_generator`` is bound to the transmitter through the
    :class:`Simulator` (``connect(dopa, vt)``); the broadcast ``n`` is stepped
    manually and sampled at ``sample_dt`` boundaries. The generator is placed one
    step (``DT``) before each ``D`` so the substrate's one-step holder lag lands the
    jump at ``D`` — the same wall-clock step as NEST's two-step relay.

    Parameters
    ----------
    dopa_arrival : sequence of float
        Intended dopa VT-arrival times (ms).
    T : float
        Horizon (ms).
    tau_n : float
        Concentration time constant (ms).
    sample_dt : float, optional
        Sampling period (ms). Default ``1.0``.

    Returns
    -------
    times : ndarray
        Sample times (ms).
    n : ndarray
        Broadcast concentration at each sample time.
    """
    import brainstate
    import numpy as _np
    import brainunit as u
    from brainpy_state import Simulator, spike_generator, volume_transmitter

    sim = Simulator(dt=DT * u.ms)
    dopa = sim.create(spike_generator,
                      spike_times=_np.asarray([d - DT for d in dopa_arrival]) * u.ms)
    vt = sim.create(volume_transmitter, tau_n=float(tau_n) * u.ms)
    sim.connect(dopa, vt)
    vt_mod = vt.segments[0].population
    brainstate.nn.init_all_states(sim)

    n_steps = int(round(T / DT))
    every = int(round(sample_dt / DT))
    times, ns = [], []
    for k in range(1, n_steps + 1):
        t = k * DT
        with brainstate.environ.context(t=t * u.ms, i=k):
            sim.update(t * u.ms)
        if k % every == 0:
            times.append(round(t, 4))
            ns.append(float(_np.asarray(vt_mod.n.value)[0]))
    return np.asarray(times), np.asarray(ns)


# ==========================================================================
# 2. weight trajectory under dopamine modulation
# ==========================================================================
def nest_dopamine_run(per_conn, common, post_tau_minus, pre_want, post_want, dopa_arrival, T):
    """Run the decoupled dopamine drive in live NEST; return fire times + weights.

    Wires a single ``stdp_dopamine_synapse`` edge ``pre_parrot -> iaf_post`` bound to
    a ``volume_transmitter`` fed by a dopa ``spike_generator -> parrot -> vt`` pool.
    The weight is logged at every pre ``send`` by a ``weight_recorder``.

    Parameters
    ----------
    per_conn : dict
        Per-connection synapse params (``weight``; ``delay`` is added automatically).
    common : dict
        Common (``CopyModel``) dopamine properties: ``A_plus``, ``A_minus``,
        ``tau_plus``, ``tau_c``, ``tau_n``, ``b``, ``Wmin``, ``Wmax`` (the
        ``volume_transmitter`` and ``weight_recorder`` are added automatically).
    post_tau_minus : float
        ``tau_minus`` (ms) for the post archiving node (the ``K-`` trace).
    pre_want, post_want : sequence of float
        Desired pre / post fire times (ms). Realized times are recorded.
    dopa_arrival : sequence of float
        Intended dopa VT-arrival times (ms); ``n`` jumps at each.
    T : float
        Horizon (ms).

    Returns
    -------
    pre_fire, post_fire : ndarray
        Sorted recorded pre / post fire times (ms).
    weights, wr_times : ndarray
        Weight-recorder weights and their send times, ordered by time.
    final : dict
        ``{'c': ..., 'n': ..., 'weight': ...}`` read off the connection at ``T``.
    """
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")

    vt = nest.Create("volume_transmitter")
    sg_dopa = nest.Create("spike_generator",
                          params={"spike_times": sorted(round(d - DOPA_RELAY, 4) for d in dopa_arrival)})
    par_dopa = nest.Create("parrot_neuron")
    nest.Connect(sg_dopa, par_dopa, syn_spec={"delay": DT})
    nest.Connect(par_dopa, vt, syn_spec={"delay": DT})

    parrot_pre = nest.Create("parrot_neuron")
    post = nest.Create("iaf_psc_delta", params={**POST_PARAMS, "tau_minus": post_tau_minus})
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
    nest.CopyModel("stdp_dopamine_synapse", "dopa_rec",
                   {"weight_recorder": wr, "volume_transmitter": vt, **common})
    nest.Connect(parrot_pre, post,
                 syn_spec={"synapse_model": "dopa_rec", "delay": DEND_D, **per_conn})
    conn = nest.GetConnections(parrot_pre, post, synapse_model="dopa_rec")

    nest.Simulate(T)
    pre_fire = np.sort(np.asarray(rec_pre.get("events")["times"]))
    post_fire = np.sort(np.asarray(rec_post.get("events")["times"]))
    ev = wr.get("events")
    order = np.argsort(np.asarray(ev["times"]))
    final = {"c": float(conn.get("c")), "n": float(conn.get("n")),
             "weight": float(conn.get("weight"))}
    return (pre_fire, post_fire, np.asarray(ev["weights"])[order],
            np.asarray(ev["times"])[order], final)


def dopamine_trajectory(common, post_tau_minus, pre_want, post_want, dopa_arrival, T,
                        *, init_w=50.0, c0=0.0):
    """Run both sides of a dopamine weight-trajectory scenario; align at send steps.

    Convenience over :func:`nest_dopamine_run` + :func:`bp_dopamine_weight_trace`: it
    runs the live-NEST drive, builds the matching :class:`stdp_dopamine_synapse` spec
    from the same ``common`` properties, replays NEST's recorded fire times through our
    substrate, and samples our per-step weight at the ``weight_recorder`` send times.

    Parameters
    ----------
    common : dict
        Common dopamine properties (``A_plus``, ``A_minus``, ``tau_plus``, ``tau_c``,
        ``tau_n``, ``b``, ``Wmin``, ``Wmax``).
    post_tau_minus : float
        ``tau_minus`` (ms) for the post ``K-`` archive **and** the spec.
    pre_want, post_want, dopa_arrival : sequence of float
        Desired pre / post fire and dopa VT-arrival times (ms).
    T : float
        Horizon (ms).
    init_w : float, optional
        Initial weight (pA). Default ``50``.
    c0 : float, optional
        Initial eligibility trace. Default ``0``.

    Returns
    -------
    wr_times : ndarray
        Weight-recorder send times (ms).
    nest_w : ndarray
        NEST weight at each send time.
    our_w : ndarray
        Our online weight sampled at the same send steps.
    final : dict
        NEST connection ``{'c', 'n', 'weight'}`` at ``T`` (NEST's own deferred
        integral through ``T``, including dynamics after the last pre send).
    our_final : float
        Our online weight at the last step (``T``) — the apples-to-apples partner of
        ``final['weight']`` (use this, not ``our_w[-1]``, when the decisive dynamics
        happen *after* the last pre send, e.g. a delayed dopa read-out pulse).
    """
    import brainunit as u
    from brainpy_state import stdp_dopamine_synapse

    pre_fire, post_fire, weights, wr_times, final = nest_dopamine_run(
        per_conn=dict(weight=init_w), common=common, post_tau_minus=post_tau_minus,
        pre_want=pre_want, post_want=post_want, dopa_arrival=dopa_arrival, T=T)
    rule = stdp_dopamine_synapse(
        weight=init_w * u.pA, A_plus=common['A_plus'], A_minus=common['A_minus'],
        tau_plus=common['tau_plus'] * u.ms, tau_minus=post_tau_minus * u.ms,
        tau_c=common['tau_c'] * u.ms, tau_n=common['tau_n'] * u.ms,
        b=common['b'], Wmin=common['Wmin'], Wmax=common['Wmax'], c=c0)
    w_trace = bp_dopamine_weight_trace(
        rule, pre_fire, post_fire, dopa_arrival, int(round(T / DT)), tau_n=common['tau_n'])
    our_w = w_trace[steps(wr_times)]
    return wr_times, weights, our_w, final, float(w_trace[-1])


def bp_dopamine_weight_trace(rule, pre_fire, post_fire, dopa_arrival, n_steps, *,
                             dend_d=DEND_D, tau_n=200.0):
    """Drive a single dopamine edge with a bound transmitter; return the weight trace.

    ``pre_fire`` / ``post_fire`` are recorded NEST fire times (ms): pre acts at ``p``,
    each post effect is injected at ``q + dend_d`` (the dendritic-delay shift), and
    ``rule.delay`` is forced to ``None``. A real :class:`volume_transmitter` is fed the
    dopa train (``n`` jumps at each ``dopa_arrival`` step) and bound as the ``n`` signal
    source; the transmitter is advanced **before** the projection each step (the
    Simulator's phase order). The per-step value is the stored weight
    ``proj.weight.value[0]``; sample it at the pre-send steps. Runs inside
    ``for_loop`` (per-step spikes are scanned args) so the loop JIT-compiles once.

    Parameters
    ----------
    rule : stdp_dopamine_synapse
        The dopamine synapse spec (``signal_reads=('n',)``).
    pre_fire, post_fire : sequence of float
        Recorded pre / post fire times (ms).
    dopa_arrival : sequence of float
        Intended dopa VT-arrival times (ms).
    n_steps : int
        Number of steps to run.
    dend_d : float, optional
        Dendritic-delay shift for the post effect (ms). Default :data:`DEND_D`.
    tau_n : float, optional
        Transmitter concentration time constant (ms); must equal the rule's. Default ``200``.

    Returns
    -------
    ndarray
        Per-step stored weight (``(n_steps,)``).
    """
    import brainstate
    import jax.numpy as jnp
    import brainunit as u
    from brainstate import transform
    from brainpy_state._network._event_plastic import VoltageCoupledPlasticProj
    from brainpy_state._nest.volume_transmitter import volume_transmitter

    class _Sink:                                   # delivery target (ignored); not None
        def add_delta_input(self, key, val):
            pass

    rule.delay = None
    box = {'pre': jnp.zeros(1), 'post': jnp.zeros(1), 'dopa': jnp.zeros(1)}
    vt = volume_transmitter(1, tau_n=tau_n * u.ms)
    vt.bind_dopa(lambda: box['dopa'], jnp.array([0]))
    proj = VoltageCoupledPlasticProj(
        pre_spike=lambda: box['pre'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=_Sink(), post_local_idx=jnp.arange(1), n_post_pop=1,
        post_spike=lambda: box['post'],
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule,
        signal_sources={'n': (vt, 'n')})
    brainstate.nn.init_all_states(vt)
    brainstate.nn.init_all_states(proj)

    pre = np.zeros((n_steps, 1))
    pre[steps(pre_fire), 0] = 1.0
    post = np.zeros((n_steps, 1))
    if len(post_fire):
        post[steps([q + dend_d for q in post_fire]), 0] = 1.0
    dopa = np.zeros((n_steps, 1))
    dopa[steps(dopa_arrival), 0] = 1.0
    pre = jnp.asarray(pre)
    post = jnp.asarray(post)
    dopa = jnp.asarray(dopa)
    times = jnp.arange(n_steps) * DT * u.ms
    indices = jnp.arange(n_steps)

    def step(t, i, xp, xq, xd):
        box['pre'] = xp
        box['post'] = xq
        box['dopa'] = xd
        with brainstate.environ.context(t=t, i=i):
            vt.update()                            # phase 0: advance broadcast n
            proj.update()                          # reads vt.n via the signal seam
            return proj.weight.value[0]

    return np.asarray(transform.for_loop(step, times, indices, pre, post, dopa))
