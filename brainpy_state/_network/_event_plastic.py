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

__all__ = ['EventPlasticProj', 'KernelContext', 'PlasticSynapse']


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
        no ``pre_trace_tau``).
    post_trace : jax.Array
        ``(E,)`` postsynaptic trace gathered per edge (zeros if none).
    t_now : jax.Array
        Scalar current simulation time, ms mantissa.
    dt : jax.Array
        Scalar timestep, ms mantissa.
    key : jax.Array
        Per-step PRNG key (used by stochastic rules; deterministic rules ignore).
    """
    pre_spike: jax.Array
    post_spike: jax.Array
    pre_trace: jax.Array
    post_trace: jax.Array
    t_now: jax.Array
    dt: jax.Array
    key: jax.Array


class PlasticSynapse(Protocol):
    """Structural protocol every rebuilt ``_nest`` synapse spec satisfies.

    The concrete specs live in ``_nest/<model>_synapse.py``; this substrate only
    relies on the attributes and the two methods below.
    """
    weight: object              # per-edge init (pA); scalar if homogeneous
    delay: object               # homogeneous axonal delay (Quantity) or None
    is_homogeneous_weight: bool  # 'weight' State is a shared 0-d scalar
    stochastic: bool            # needs ctx.key
    pre_trace_tau: object       # Quantity or None — enables the pre trace State
    post_trace_tau: object      # Quantity or None — enables the post trace State
    weight_unit: object         # pA

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
        self.delay_seam = (InputDelay((self._n_pre_pop,), rule.delay)
                           if rule.delay is not None else None)

        # -- pre-computed (Python bool) full-post fast path -------------------
        self._post_is_full = (
            n_post == self._n_post_pop
            and bool(jnp.all(self.post_local_idx == jnp.arange(self._n_post_pop)))
        )

    def init_state(self, *args, **kwargs):
        dftype = brainstate.environ.dftype()
        self.weight = brainstate.ParamState(jnp.asarray(self._w_init, dtype=dftype))
        self.aux = {
            name: brainstate.HiddenState(jnp.full((self._E,), float(v), dtype=dftype))
            for name, v in self.rule.edge_state_init().items()
        }
        self.pre_trace = (brainstate.HiddenState(jnp.zeros(self._n_pre_pop, dtype=dftype))
                          if self.rule.pre_trace_tau is not None else None)
        self.post_trace = (brainstate.HiddenState(jnp.zeros(self._n_post_pop, dtype=dftype))
                           if self.rule.post_trace_tau is not None else None)
        self.rng = brainstate.State(jax.random.key(0)) if self.rule.stochastic else None

    # -- helpers -----------------------------------------------------------
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

        # per-neuron traces (decay-then-add, gather post-update)
        if self.pre_trace is not None:
            tau = u.Quantity(self.rule.pre_trace_tau).to_decimal(u.ms)
            self.pre_trace.value = self.pre_trace.value * jnp.exp(-dt / tau) + x_full
            pre_trace_edge = self.pre_trace.value[self.pre_local_idx[self._pre_idx]]
        else:
            pre_trace_edge = jnp.zeros((self._E,))
        if self.post_trace is not None and post_full is not None:
            tau = u.Quantity(self.rule.post_trace_tau).to_decimal(u.ms)
            self.post_trace.value = self.post_trace.value * jnp.exp(-dt / tau) + post_full
            post_trace_edge = self.post_trace.value[self.post_local_idx[self._post_idx]]
        else:
            post_trace_edge = jnp.zeros((self._E,))

        key = jax.random.key(0)
        if self.rng is not None:
            key, sub = jax.random.split(self.rng.value)
            self.rng.value = key
            key = sub

        state = {'weight': self.weight.value, **{k: v.value for k, v in self.aux.items()}}
        ctx = KernelContext(pre_fired, post_fired, pre_trace_edge, post_trace_edge,
                            t_now, dt, key)
        new_state, w_eff = self.rule.update(state, ctx)
        self.weight.value = new_state['weight']
        for k, v in self.aux.items():
            v.value = new_state[k]

        # deliver via CSR event-matmul (gated by the actual pre spikes)
        w_eff = jnp.broadcast_to(jnp.asarray(w_eff), (self._E,))
        csr = brainevent.CSR((w_eff, self._indices, self._indptr), shape=self._shape)
        y = jnp.asarray(brainevent.BinaryArray(pre_seg) @ csr)   # (n_post_seg,)
        contrib = y if self._post_is_full else self._scatter(y)
        if self._w_unit is not u.UNITLESS:
            contrib = u.Quantity(contrib, unit=self._w_unit)
        if self.post is not None:
            self.post.add_delta_input(self._delta_key, contrib)
        return contrib
