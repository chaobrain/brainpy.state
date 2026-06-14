# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""EventPlasticProj — JAX-native, event-driven plastic projection substrate.

This is the first of the three typed plasticity primitives described in
``CONTEXT.md`` Part 2.5. It owns the *compute*: a CSR edge layout (reusing the
:class:`~brainpy_state._network._projections._SparseEventMatMul` convention),
an :class:`~brainpy_state._brainpy._delay.InputDelay` axonal delay seam,
rule-declared per-edge / per-neuron :class:`brainstate.State` allocation, and
the ``brainevent.CSR`` event matmul that delivers weighted spikes into
``post.add_delta_input``.

The *fidelity* lives in the synapse spec (``_nest/<model>_synapse.py``): a
frozen :class:`PlasticSynapse` exposes the NEST parameter spec plus a pure,
vectorized rule kernel ``update(state, ctx) -> (state, w_eff)``. The substrate
hands the kernel a :class:`KernelContext` each step and gates delivery by the
actual presynaptic spikes.

The whole hot path is ``jit`` / ``vmap`` / ``for_loop`` safe: every State has a
static shape fixed at trace time, there is no Python-list growth, no
data-dependent host control flow, and all elementwise math is ``jnp``.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Protocol

import brainevent
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u

from brainpy_state._brainpy._delay import InputDelay
from brainpy_state._network._event_proj import EventProjection

__all__ = ['EventPlasticProj', 'VoltageCoupledPlasticProj', 'KernelContext', 'PlasticSynapse']


def _trace_spec(attr, mode='all_to_all'):
    """Normalize a rule's per-side trace-tau declaration.

    Returns ``None`` (no trace), ``('single', tau_ms, mode)`` for a scalar
    ``Quantity`` (cluster-01 contract: 1-D per-neuron State), or
    ``('multi', (tau0_ms, ...), mode)`` for a tuple/list of taus (``stdp_triplet``,
    later clopath: one per-neuron column per tau). ``mode`` is the per-side
    pairing mode ``'all_to_all'`` (decay-then-add, the default) or ``'nearest'``
    (reset-to-1 on spike; cluster-05 nearest-neighbour STDP).
    """
    if attr is None:
        return None
    if isinstance(attr, (tuple, list)):
        return ('multi', tuple(float(u.Quantity(t).to_decimal(u.ms)) for t in attr), mode)
    return ('single', float(u.Quantity(attr).to_decimal(u.ms)), mode)


class KernelContext(NamedTuple):
    """Per-step inputs the substrate hands a rule kernel.

    All per-edge arrays are length ``E = n_edges`` in CSR (sorted-by-pre) edge
    order; scalars are 0-d. Every value is a unit-free mantissa — the substrate
    re-attaches the pA unit on delivery.

    Attributes
    ----------
    pre_spike : jax.Array
        ``(E,)`` — did this edge's (delayed) presynaptic neuron fire this step.
    post_spike : jax.Array
        ``(E,)`` — did this edge's postsynaptic neuron fire this step (zeros if
        the projection has no post-spike reader).
    pre_trace : jax.Array
        ``(E,)`` presynaptic trace gathered per edge (zeros if the rule declares
        no ``pre_trace_tau``). When the rule declares a *tuple* of taus this is
        the first column of ``pre_traces``.
    post_trace : jax.Array
        ``(E,)`` postsynaptic trace gathered per edge (zeros if none); first
        column of ``post_traces`` for multi-trace rules.
    t_now : jax.Array
        Scalar current simulation time, ms mantissa.
    dt : jax.Array
        Scalar timestep, ms mantissa.
    key : jax.Array
        Per-step PRNG key (used by stochastic rules; deterministic rules ignore).
    pre_traces : jax.Array
        ``(E, k)`` all presynaptic traces gathered per edge, one column per tau
        when ``pre_trace_tau`` is a tuple (``stdp_triplet`` / clopath seam).
        ``(E, 1)`` for a scalar tau, ``(E, 0)`` for none. ``pre_trace`` aliases
        column 0.
    post_traces : jax.Array
        ``(E, k)`` all postsynaptic traces gathered per edge (see ``pre_traces``).
    post_states : dict
        ``{name: (E,)}`` per-edge gathers of the post-neuron analog State variables
        named by the rule's ``post_state_reads`` (the primitive-#2
        :class:`VoltageCoupledPlasticProj` reader; e.g. ``'V'`` / ``'u_bar_plus'`` /
        ``'u_bar_minus'`` for ``clopath_synapse``). Each value is a unit-stripped
        mantissa in the State's stored unit, in CSR (sorted-by-pre) edge order.
        ``None`` for the base :class:`EventPlasticProj` (primitive #1 reads no post
        State).
    signals : dict
        ``{name: scalar}`` broadcast modulatory signals read from a bound third-party
        node (the :class:`~brainpy_state._nest.volume_transmitter`) and shared by
        **every edge** of the projection (cluster-08; e.g. ``'n'``, the dopamine
        concentration, for ``stdp_dopamine_synapse``). Each value is a unit-stripped
        scalar mantissa broadcast against the ``(E,)`` per-edge arrays — a superset
        of :attr:`post_states` (1->E broadcast vs N->E gather). ``None`` for any
        projection that reads no broadcast signal (primitives #1 and post-only #2).
    """
    pre_spike: jax.Array
    post_spike: jax.Array
    pre_trace: jax.Array
    post_trace: jax.Array
    t_now: jax.Array
    dt: jax.Array
    key: jax.Array
    pre_traces: jax.Array = None
    post_traces: jax.Array = None
    post_states: dict = None
    signals: dict = None


class PlasticSynapse(Protocol):
    """Structural protocol every rebuilt ``_nest`` synapse spec satisfies.

    The concrete specs live in ``_nest/<model>_synapse.py``; this substrate only
    relies on the attributes and the two methods below.
    """
    weight: object              # per-edge init (pA); scalar if homogeneous
    delay: object               # homogeneous axonal delay (Quantity) or None
    is_homogeneous_weight: bool  # 'weight' State is a shared 0-d scalar
    stochastic: bool            # needs ctx.key
    pre_trace_tau: object       # None | Quantity | tuple[Quantity,...] (multi-trace)
    post_trace_tau: object      # None | Quantity | tuple[Quantity,...] (multi-trace)
    weight_unit: object         # pA
    # optional (default 'all_to_all'); 'nearest' resets the trace to 1 on each
    # spike instead of accumulating (cluster-05 nearest-neighbour STDP):
    # pre_trace_mode, post_trace_mode
    # optional (default False); when True the substrate honours a sub-dt delay by
    # delivering at the integer floor delay and splitting the post amplitude
    # across the two bracketing grid steps (cluster-06 cont_delay_synapse):
    # fractional_delay

    def edge_state_init(self) -> dict: ...

    def update(self, state: dict, ctx: KernelContext) -> tuple[dict, jax.Array]: ...


class _StaticTestRule:
    """Test-only constant-weight rule (constant ``w_eff``; no aux, no traces).

    Lives here so the substrate is testable before the real ``_nest`` specs are
    rebuilt. Production code uses the specs in ``_nest/``.
    """
    is_homogeneous_weight = False
    stochastic = False
    pre_trace_tau = None
    post_trace_tau = None
    weight_unit = u.pA

    def __init__(self, weight=1.0, delay=None):
        self.weight = weight
        self.delay = delay

    def edge_state_init(self) -> dict:
        return {}

    def update(self, state, ctx):
        return state, state['weight']


class EventPlasticProj(brainstate.nn.Module):
    """Event-driven plastic projection from one population segment.

    Each step it reads the pre population's captured spike vector via
    ``pre_spike()``, applies the axonal :class:`InputDelay`, restricts to this
    projection's pre/post segments, maintains any rule-declared per-neuron
    traces, calls the synapse rule kernel to obtain the per-edge effective
    weight, and delivers it through a ``brainevent.CSR`` event matmul into
    ``post.add_delta_input`` (summing multapses, scattering into sub-segments).

    Parameters
    ----------
    pre_spike : Callable[[], jax.Array]
        Returns the full pre-population spike vector, shape ``(n_pre_pop,)``.
    n_pre_pop : int
        Size of the full pre population.
    pre_local_idx : array_like
        Local indices into the pre population selected by this projection.
    post : Dynamics or None
        Post-synaptic population (receives ``add_delta_input``). ``None`` is
        permitted for substrate-only tests that do not deliver.
    post_local_idx : array_like
        Local indices into the post population targeted by this projection.
    rule : PlasticSynapse
        The synapse spec carrying the pure ``update`` kernel and the parameter
        declarations (weight, delay, traces, stochastic, homogeneity).
    conn : ConnRule, optional
        Connectivity sampler; used when explicit ``pre_idx``/``post_idx`` edges
        are not supplied.
    pre_idx, post_idx : array_like, optional
        Explicit segment-local edges (skip sampling). Both or neither.
    n_post_pop : int, optional
        Size of the full post population (defaults to ``len(post_local_idx)``).
    post_spike : Callable[[], jax.Array], optional
        Returns the full post-population spike vector (STDP seam).
    pre_is_post, allow_autapses, allow_multapses : bool
        Forwarded to the connectivity sampler.
    seed : int, optional
        Connectivity sampling seed.
    delta_key : str, optional
        Unique ``add_delta_input`` key (defaults to one derived from ``id``).
    receptor_type : int, optional
        Named-channel routing for a multi-compartment / named-channel post (a
        post exposing ``delta_label_for_receptor`` and no ``n_receptors``, e.g.
        ``pp_cond_exp_mc_urbanczik``). Resolved once to a delta-input channel
        label (mirroring the static :class:`EventProjection` seam); each per-step
        deposit is then tagged with that label so the deposit reaches the correct
        compartment/sign. ``None`` (default) delivers to the unlabeled key.

    Examples
    --------
    .. code-block:: python

       >>> import jax.numpy as jnp, brainstate, saiunit as u
       >>> from brainpy_state._network._event_plastic import EventPlasticProj, _StaticTestRule
       >>> class _Sink:
       ...     def add_delta_input(self, key, val): self.last = val
       >>> sink = _Sink()
       >>> brainstate.environ.set(dt=0.1 * u.ms)
       >>> proj = EventPlasticProj(
       ...     pre_spike=lambda: jnp.array([1., 0.]), n_pre_pop=2,
       ...     pre_local_idx=jnp.arange(2), post=sink,
       ...     post_local_idx=jnp.arange(2), n_post_pop=2,
       ...     pre_idx=jnp.array([0, 1]), post_idx=jnp.array([0, 1]),
       ...     rule=_StaticTestRule(weight=jnp.array([3., 4.]) * u.pA))
       >>> _ = brainstate.nn.init_all_states(proj)
       >>> with brainstate.environ.context(t=0.1 * u.ms, i=1):
       ...     _ = proj.update()
       >>> u.get_mantissa(sink.last).tolist()
       [3.0, 0.0]
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        *,
        pre_spike: Callable[[], jax.Array],
        n_pre_pop: int,
        pre_local_idx,
        post,
        post_local_idx,
        rule: PlasticSynapse,
        conn=None,
        pre_idx=None,
        post_idx=None,
        n_post_pop: Optional[int] = None,
        post_spike: Optional[Callable[[], jax.Array]] = None,
        pre_is_post: bool = False,
        allow_autapses: bool = True,
        allow_multapses: bool = True,
        seed: Optional[int] = None,
        delta_key: Optional[str] = None,
        receptor_type=None,
    ):
        super().__init__()
        self.pre_spike = pre_spike
        self.post_spike = post_spike
        self.post = post
        self.rule = rule
        self.pre_local_idx = jnp.asarray(pre_local_idx)
        self.post_local_idx = jnp.asarray(post_local_idx)
        self._n_pre_pop = int(n_pre_pop)
        self._n_post_pop = int(n_post_pop if n_post_pop is not None
                               else self.post_local_idx.shape[0])
        self._delta_key = delta_key or f'event_plastic_{id(self)}'
        # Named-channel single-label routing (mc post, e.g. the Urbanczik
        # neuron): resolve receptor_type -> delta channel label once, then tag
        # every deposit so the plastic weight reaches the right compartment.
        # ``None`` for an unrouted or stacked-n_receptors post (delivers
        # unlabeled, the historical behavior).
        self._channel_label = EventProjection._resolve_channel_label(post, receptor_type)
        # Runtime-rng seed for stochastic rules. ``init_state`` keys the per-step
        # ``rng`` from this so the seed survives ``init_all_states`` (which the
        # Simulator runs inside ``simulate``); ``None`` keeps the historical key(0)
        # default. Shares the integer with connectivity sampling (mirrors NEST's
        # single ``rng_seed``); the two consume independent key instances.
        self._rng_seed = seed

        n_pre = int(self.pre_local_idx.shape[0])
        n_post = int(self.post_local_idx.shape[0])

        # -- edges (segment-local), sorted once by pre into CSR order ---------
        if pre_idx is None or post_idx is None:
            if conn is None:
                raise ValueError('EventPlasticProj needs either explicit edges '
                                 '(pre_idx, post_idx) or a connectivity rule (conn).')
            key = jax.random.key(0 if seed is None else int(seed))
            spec = conn.sample(n_pre, n_post, key=key, pre_is_post=pre_is_post,
                               allow_autapses=allow_autapses, allow_multapses=allow_multapses)
            pre_idx, post_idx = spec.pre_idx, spec.post_idx
        pre_np = np.asarray(pre_idx)
        post_np = np.asarray(post_idx)
        order = np.argsort(pre_np, kind='stable')           # group edges by pre row
        self._pre_idx = jnp.asarray(pre_np[order])
        self._post_idx = jnp.asarray(post_np[order])
        self._indices = jnp.asarray(post_np[order])
        self._indptr = jnp.asarray(
            np.concatenate([[0], np.cumsum(np.bincount(pre_np, minlength=n_pre))]))
        self._shape = (n_pre, n_post)
        self._E = int(pre_np.shape[0])

        # -- weight init mantissa + unit (kept off the hot path) --------------
        w = rule.weight
        if isinstance(w, u.Quantity):
            self._w_unit = w.unit
            w_m = jnp.asarray(w.mantissa)
        else:
            self._w_unit = getattr(rule, 'weight_unit', u.UNITLESS)
            w_m = jnp.asarray(w)
        self._w_init = (jnp.reshape(w_m, ()) if rule.is_homogeneous_weight
                        else jnp.broadcast_to(w_m, (self._E,)))

        # -- axonal delay seam (identity when delay is None) ------------------
        # A rule may opt into a *sub-dt* delay via ``fractional_delay = True``
        # (cont_delay_synapse): the substrate then delivers the binary event at
        # the integer floor delay (so ``BinaryArray`` is not fed a fractional
        # vector) and splits the post amplitude across the two bracketing grid
        # steps with a 1-step output carry (built in ``init_state``, where ``dt``
        # is known). Default-off: every other rule keeps the unchanged path.
        self._fractional_delay = bool(getattr(rule, 'fractional_delay', False))
        self.delay_seam = (InputDelay((self._n_pre_pop,), rule.delay)
                           if (rule.delay is not None and not self._fractional_delay)
                           else None)

        # -- pre-computed (Python bool) full-post fast path -------------------
        self._post_is_full = (
            n_post == self._n_post_pop
            and bool(jnp.all(self.post_local_idx == jnp.arange(self._n_post_pop)))
        )

        # -- per-side trace specs (None | single | multi) + mode, fixed at trace time
        self._pre_trace_spec = _trace_spec(
            rule.pre_trace_tau, getattr(rule, 'pre_trace_mode', 'all_to_all'))
        self._post_trace_spec = _trace_spec(
            rule.post_trace_tau, getattr(rule, 'post_trace_mode', 'all_to_all'))

    def init_state(self, *args, **kwargs):
        dftype = brainstate.environ.dftype()
        self.weight = brainstate.ParamState(jnp.asarray(self._w_init, dtype=dftype))
        self.aux = {
            name: brainstate.HiddenState(jnp.full((self._E,), float(v), dtype=dftype))
            for name, v in self.rule.edge_state_init().items()
        }
        self.pre_trace = self._alloc_trace(self._pre_trace_spec, self._n_pre_pop, dftype)
        self.post_trace = self._alloc_trace(self._post_trace_spec, self._n_post_pop, dftype)
        self.rng = (brainstate.State(jax.random.key(
            0 if self._rng_seed is None else int(self._rng_seed)))
            if self.rule.stochastic else None)

        # -- sub-dt (fractional) delay seam (default-off) ---------------------
        # Decompose the homogeneous delay into an integer floor ``k_lo`` and a
        # fraction ``frac = d/dt - k_lo`` (NEST cont_delay's ``delay_steps`` /
        # ``delay_offset_``). Deliver the binary event at ``k_lo`` steps (clean
        # floor frame for ``BinaryArray``) and FIR-split the post amplitude
        # ``[1-frac, frac]`` across the two bracketing grid steps. ``frac == 0``
        # (integer delay) -> no carry -> byte-identical to a plain grid delay.
        self.delay_carry = None
        self._delay_frac = 0.0
        if self._fractional_delay and self.rule.delay is not None:
            dt = brainstate.environ.get_dt()
            steps = (float(u.Quantity(self.rule.delay).to_decimal(u.ms))
                     / float(u.Quantity(dt).to_decimal(u.ms)))
            k_lo = int(np.floor(steps + 1e-9))
            frac = float(steps - k_lo)
            self._delay_frac = 0.0 if frac < 1e-9 else frac
            if k_lo >= 1:
                self.delay_seam = InputDelay((self._n_pre_pop,), k_lo * dt)
                brainstate.nn.init_all_states(self.delay_seam)
            if self._delay_frac > 0.0:
                self.delay_carry = brainstate.HiddenState(
                    jnp.zeros((self._shape[1],), dtype=dftype))

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _alloc_trace(spec, n, dftype):
        """Per-neuron trace State: 1-D ``(n,)`` for single tau, ``(n, k)`` for multi."""
        if spec is None:
            return None
        kind, taus, _mode = spec
        shape = (n,) if kind == 'single' else (n, len(taus))
        return brainstate.HiddenState(jnp.zeros(shape, dtype=dftype))

    @staticmethod
    def _advance_trace(state_obj, spec, x_full, dt, gather, E):
        """Decay a per-neuron trace, gather per edge; store per the trace mode.

        Returns ``(trace_edge (E,), traces_edge (E, k))``. ``spec`` ``None`` ->
        zeros. Single tau keeps the 1-D State + ``(E,)`` gather (``(E, 1)`` matrix
        view); a tuple of taus decays each column by its own tau.

        Both modes GATHER the same ``decayed + spike`` (so the cluster-04 kernel
        exclusion ``k = ctx.trace - ctx.spike`` reduces to the strictly-prior
        value, which for ``nearest`` is NEST's "second-latest preceding partner"
        on a coinciding step); they differ only in what is STORED for the next
        step -- ``all_to_all`` accumulates ``decayed + spike``, ``nearest`` resets
        to 1 on the spike (``where(spike, 1, decayed)``).
        """
        if spec is None:
            return jnp.zeros((E,)), jnp.zeros((E, 0))
        kind, taus, mode = spec
        if kind == 'single':
            decayed = state_obj.value * jnp.exp(-dt / taus)
            accumulated = decayed + x_full                          # kernel-facing
            state_obj.value = (jnp.where(x_full > 0, 1.0, decayed)
                               if mode == 'nearest' else accumulated)
            e = accumulated[gather]                                 # (E,)
            return e, e[:, None]                                    # (E, 1)
        taus_arr = jnp.asarray(taus)                                # (k,)
        decayed = state_obj.value * jnp.exp(-dt / taus_arr)
        accumulated = decayed + x_full[:, None]
        state_obj.value = (jnp.where(x_full[:, None] > 0, 1.0, decayed)
                           if mode == 'nearest' else accumulated)
        g = accumulated[gather]                                     # (E, k)
        return g[:, 0], g
    @staticmethod
    def _t_dt_ms():
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        t_ms = u.Quantity(t).to_decimal(u.ms) if isinstance(t, u.Quantity) else jnp.asarray(t)
        dt_ms = u.Quantity(dt).to_decimal(u.ms)
        return jnp.asarray(t_ms), jnp.asarray(dt_ms)

    def _scatter(self, y):
        """Place per-segment contributions into a full ``(n_post_pop,)`` vector."""
        base = jnp.zeros(self._n_post_pop, dtype=y.dtype)
        return base.at[self.post_local_idx].add(y)

    def _gather_post_states(self):
        """Post-neuron analog-State reads for the kernel (``None`` for primitive #1).

        The base :class:`EventPlasticProj` reads no post State; the override in
        :class:`VoltageCoupledPlasticProj` returns the ``{name: (E,)}`` per-edge
        gather declared by ``rule.post_state_reads``.
        """
        return None

    def _gather_signals(self):
        """Broadcast modulatory-signal reads for the kernel (``None`` by default).

        The base :class:`EventPlasticProj` reads no broadcast signal; the override
        in :class:`VoltageCoupledPlasticProj` returns the ``{name: scalar}`` dict
        declared by ``rule.signal_reads`` and bound via ``signal_sources`` (the
        cluster-08 ``volume_transmitter`` seam). Read-only.
        """
        return None

    # -- step --------------------------------------------------------------
    def update(self):
        x_full = jnp.asarray(self.pre_spike())
        if self.delay_seam is not None:
            x_full = jnp.asarray(self.delay_seam.update(x_full))
        pre_seg = x_full[self.pre_local_idx]
        pre_fired = pre_seg[self._pre_idx]

        if self.post_spike is not None:
            post_full = jnp.asarray(self.post_spike())
            post_seg = post_full[self.post_local_idx]
            post_fired = post_seg[self._post_idx]
        else:
            post_full = None
            post_fired = jnp.zeros((self._E,))

        t_now, dt = self._t_dt_ms()

        # per-neuron traces (decay-then-add, gather post-update). Single tau ->
        # 1-D State + (E,) edge trace; a tuple of taus -> (N, k) State + (E, k).
        pre_trace_edge, pre_traces_edge = self._advance_trace(
            self.pre_trace, self._pre_trace_spec, x_full, dt,
            self.pre_local_idx[self._pre_idx], self._E)
        if self.post_trace is not None and post_full is not None:
            post_trace_edge, post_traces_edge = self._advance_trace(
                self.post_trace, self._post_trace_spec, post_full, dt,
                self.post_local_idx[self._post_idx], self._E)
        else:
            post_trace_edge = jnp.zeros((self._E,))
            post_traces_edge = jnp.zeros((self._E, 0))

        key = jax.random.key(0)
        if self.rng is not None:
            key, sub = jax.random.split(self.rng.value)
            self.rng.value = key
            key = sub

        state = {'weight': self.weight.value, **{k: v.value for k, v in self.aux.items()}}
        ctx = KernelContext(pre_fired, post_fired, pre_trace_edge, post_trace_edge,
                            t_now, dt, key, pre_traces_edge, post_traces_edge,
                            self._gather_post_states(), self._gather_signals())
        new_state, w_eff = self.rule.update(state, ctx)
        self.weight.value = new_state['weight']
        for k, v in self.aux.items():
            v.value = new_state[k]

        # deliver via CSR event-matmul (gated by the actual pre spikes)
        w_eff = jnp.broadcast_to(jnp.asarray(w_eff), (self._E,))
        csr = brainevent.CSR((w_eff, self._indices, self._indptr), shape=self._shape)
        y = jnp.asarray(brainevent.BinaryArray(pre_seg) @ csr)   # (n_post_seg,)
        if self.delay_carry is not None:
            # sub-dt delay: deliver (1-frac) of this floor-delayed amplitude now,
            # carry frac to the next grid step (1-step FIR -> exact first moment).
            frac = self._delay_frac
            y, self.delay_carry.value = (1.0 - frac) * y + self.delay_carry.value, frac * y
        contrib = y if self._post_is_full else self._scatter(y)
        if self._w_unit is not u.UNITLESS:
            contrib = u.Quantity(contrib, unit=self._w_unit)
        if self.post is not None:
            if self._channel_label is not None:
                # Named-channel post: tag the deposit so it lands on the resolved
                # compartment/sign channel (read back via sum_delta_inputs(label=)).
                self.post.add_delta_input(self._delta_key, contrib, label=self._channel_label)
            else:
                self.post.add_delta_input(self._delta_key, contrib)
        return contrib


class VoltageCoupledPlasticProj(EventPlasticProj):
    """Voltage-coupled plastic projection — primitive #2 of the typed family.

    A superset of :class:`EventPlasticProj` that adds a **post-neuron analog-state
    reader**. The rule declares a tuple of post-neuron ``State`` attribute names in
    ``post_state_reads`` (e.g. ``('u_bar_minus', 'u_bar_plus', 'V')`` for
    ``clopath_synapse``); each step the projection gathers those per-post-neuron
    State columns **per edge** — in CSR (sorted-by-pre) edge order, exactly the
    post-trace gather (``post_local_idx[post_idx]``) — and hands them to the kernel
    as :attr:`KernelContext.post_states`, a ``{name: (E,)}`` dict of unit-stripped
    mantissas (in each State's stored unit). This samples a continuous post-neuron
    quantity (membrane / filtered voltage) that a spike-driven trace cannot
    reconstruct.

    Everything else — CSR delivery, axonal delay, rule-declared per-neuron traces
    (``x_bar`` via ``pre_trace_tau``), the weight-recording / ``_stdp_drive`` seams —
    is inherited unchanged. The post population module supplied as ``post`` is the
    read source (``getattr(post, name).value``); it must be present and the rule
    must declare a non-empty ``post_state_reads``.

    Examples
    --------
    .. code-block:: python

       >>> import jax.numpy as jnp, brainstate, saiunit as u
       >>> from brainpy_state._network._event_plastic import (
       ...     VoltageCoupledPlasticProj, _StaticTestRule)
       >>> class _Post:
       ...     def __init__(self): self.V = type('S', (), {'value': jnp.array([3.]) * u.mV})()
       ...     def add_delta_input(self, key, val): self.last = val
       >>> class _Read(_StaticTestRule):
       ...     post_state_reads = ('V',)
       ...     def update(self, state, ctx):
       ...         return state, state['weight'] + ctx.post_states['V']
       >>> brainstate.environ.set(dt=0.1 * u.ms)
       >>> post = _Post()
       >>> proj = VoltageCoupledPlasticProj(
       ...     pre_spike=lambda: jnp.array([1.]), n_pre_pop=1, pre_local_idx=jnp.arange(1),
       ...     post=post, post_local_idx=jnp.arange(1), n_post_pop=1,
       ...     pre_idx=jnp.array([0]), post_idx=jnp.array([0]),
       ...     rule=_Read(weight=jnp.array([1.]) * u.pA))
       >>> _ = brainstate.nn.init_all_states(proj)
       >>> with brainstate.environ.context(t=0.1 * u.ms, i=1):
       ...     _ = proj.update()
       >>> u.get_mantissa(post.last).tolist()
       [4.0]
    """
    __module__ = 'brainpy.state'

    def __init__(self, *args, signal_sources=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._post_state_reads = tuple(getattr(self.rule, 'post_state_reads', ()) or ())
        self._signal_reads = tuple(getattr(self.rule, 'signal_reads', ()) or ())
        self._signal_sources = dict(signal_sources or {})
        if not self._post_state_reads and not self._signal_reads:
            raise ValueError(
                'VoltageCoupledPlasticProj requires the rule to declare a non-empty '
                "'post_state_reads' (post-neuron State sampled per edge) or "
                "'signal_reads' (a broadcast scalar read from a bound node); use "
                'EventPlasticProj for a projection that reads neither.'
            )
        if self.post is None:
            raise ValueError(
                'VoltageCoupledPlasticProj needs a post population to read State from '
                '(post=None is only valid for the base EventPlasticProj).'
            )
        missing = [name for name in self._signal_reads if name not in self._signal_sources]
        if missing:
            raise ValueError(
                f'VoltageCoupledPlasticProj: rule declares signal_reads={self._signal_reads} '
                f'but no source was bound for {missing}; pass '
                'signal_sources={name: (node, attr)} (the Simulator wires this from '
                'connect(..., vt=...)).'
            )
        # per-edge population-local post index (only the post-state gather needs it)
        self._post_gather = (self.post_local_idx[self._post_idx]
                             if self._post_state_reads else None)

    def _gather_post_states(self):
        if not self._post_state_reads:
            return None
        return {name: u.get_mantissa(getattr(self.post, name).value)[self._post_gather]
                for name in self._post_state_reads}

    def _gather_signals(self):
        if not self._signal_sources:
            return None
        return {name: u.get_mantissa(getattr(node, attr).value)
                for name, (node, attr) in self._signal_sources.items()}
