# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""EventProjection — single-population delta + delay event projection.

Routes delayed, weighted (pA) pre-synaptic spike events into
``post.add_delta_input``, matching how NEST current-based neurons (e.g.
``iaf_psc_alpha``) ingest spikes: the weight is a current amplitude in pA,
sign-split into excitatory/inhibitory channels inside the neuron.
"""
from __future__ import annotations

import inspect
import itertools
from typing import Callable, Optional

import brainstate
import jax
import jax.numpy as jnp
import saiunit as u

from brainpy_state._brainpy._delay import InputDelay
from brainpy_state._network._connectivity import resolve_param
from brainpy_state._network._nodeview import _flat_size
from brainpy_state._network._projections import _DenseMatMul, _ReceptorScatter, _SparseEventMatMul
from brainpy_state._network._rules import ConnRule, _OneToOne

__all__ = ['EventProjection']

# Unique delta-input keys per projection (brainstate does not auto-assign a
# usable ``self.name`` here, and multiple projections target the same post).
_DELTA_KEY_COUNTER = itertools.count()


class EventProjection(brainstate.nn.Module):
    """Delayed, weighted delta-event projection from one population segment.

    Each step it reads the pre population's captured spike via ``pre_spike()``
    (a callable returning the full pre-population spike/counts vector), applies
    an :class:`~brainpy_state._brainpy._delay.InputDelay` (full-delay
    convention, as in ``AlignPostProj``), restricts to this projection's pre
    segment, maps it to the post segment (dense weighted matmul, or element-wise
    for ``one_to_one``), and registers the result as a delta input on ``post``.

    Parameters
    ----------
    pre_spike : Callable[[], jax.Array]
        Returns the full pre-population spike (or generator counts) vector,
        shape ``(n_pre_pop,)``.
    n_pre_pop : int
        Size of the full pre population (the dimension ``pre_spike`` returns).
    pre_local_idx : jax.Array
        Local indices into the pre population selected by this projection.
    post : Dynamics
        Post-synaptic population; receives ``add_delta_input``.
    post_local_idx : jax.Array
        Local indices into the post population targeted by this projection.
    rule : ConnRule
        Connection rule. ``one_to_one`` triggers the element-wise path.
    weight : ArrayLike or Quantity
        Synaptic weight in pA (signed: positive excitatory, negative inhibitory).
    delay : ArrayLike or Quantity or None
        Axonal delay; ``None`` for instantaneous delivery.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        *,
        pre_spike: Callable[[], jnp.ndarray],
        n_pre_pop: int,
        pre_local_idx: jnp.ndarray,
        post,
        post_local_idx: jnp.ndarray,
        rule: ConnRule,
        weight,
        delay=None,
        comm: str = 'dense',
        receptor_type=None,
        pre_is_post: bool = False,
        allow_autapses: bool = True,
        allow_multapses: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._delta_key = f'event_proj_{next(_DELTA_KEY_COUNTER)}'
        self.pre_spike = pre_spike
        self.post = post
        self.pre_local_idx = jnp.asarray(pre_local_idx)
        self.post_local_idx = jnp.asarray(post_local_idx)
        self._n_pre_pop = int(n_pre_pop)
        self._n_post_pop = _flat_size(post)
        self._one_to_one = isinstance(rule, _OneToOne)
        # Per-receptor routing: each edge targets one of the post neuron's
        # receptor ports (``iaf_psc_exp_multisynapse``), drawn uniformly.
        self._receptor = receptor_type is not None
        self._n_receptors = int(post.n_receptors) if self._receptor else 0
        # Deposit mode. Models exposing ``w_by_rec`` in their update signature
        # (``iaf``/``aeif``/``gif_cond_exp_multisynapse``) are Simulator-bridged
        # from one ``(N, n_receptors)`` blob; all others (the GLIF models)
        # self-pull each port via ``sum_delta_inputs(label='receptor_k')``.
        self._receptor_keyed = self._receptor and (
            'w_by_rec' not in inspect.signature(type(post).update).parameters
        )

        n_pre = int(self.pre_local_idx.shape[0])
        n_post = int(self.post_local_idx.shape[0])
        key = jax.random.key(0 if seed is None else int(seed))
        k_conn, k_w, k_rec = jax.random.split(key, 3)

        if self._receptor:
            if self._one_to_one:
                pre_idx, post_idx, n_edges = jnp.arange(n_pre), jnp.arange(n_post), n_pre
            else:
                spec = rule.sample(n_pre, n_post, key=k_conn, pre_is_post=pre_is_post,
                                   allow_autapses=allow_autapses, allow_multapses=allow_multapses)
                pre_idx, post_idx, n_edges = spec.pre_idx, spec.post_idx, spec.n_edges
            # ``'uniform'`` draws a port per edge; an int ``k`` (1-based, NEST
            # ``receptor_type`` convention) routes every edge to internal port ``k-1``.
            if isinstance(receptor_type, str):
                if receptor_type != 'uniform':
                    raise ValueError(f"receptor_type string must be 'uniform', got {receptor_type!r}")
                rec_idx = jax.random.randint(k_rec, (n_edges,), 0, self._n_receptors)
            else:
                k = int(receptor_type)
                if not (1 <= k <= self._n_receptors):
                    raise ValueError(f"receptor_type {k} out of range [1, {self._n_receptors}]")
                rec_idx = jnp.full((n_edges,), k - 1, dtype=jnp.int32)
            w_mant, w_unit = self._edge_weight(weight, n_edges, k_w)
            self.comm = _ReceptorScatter(pre_idx, post_idx, rec_idx, w_mant, w_unit,
                                         n_post=n_post, n_receptors=self._n_receptors)
        elif self._one_to_one:
            # Element-wise: a scalar pA weight applied per matched element.
            self._weight = weight
            self.comm = None
        else:
            if comm not in ('dense', 'sparse'):
                raise ValueError(f"comm must be 'dense' or 'sparse', got {comm!r}")
            spec = rule.sample(n_pre, n_post, key=k_conn, pre_is_post=pre_is_post,
                               allow_autapses=allow_autapses, allow_multapses=allow_multapses)
            if comm == 'dense':
                if spec.n_edges == 0:
                    W_with_unit = jnp.zeros((n_pre, n_post))
                else:
                    w_mant, w_unit = self._edge_weight(weight, spec.n_edges, k_w)
                    W = jnp.zeros((n_pre, n_post), dtype=w_mant.dtype).at[spec.pre_idx, spec.post_idx].add(w_mant)
                    W_with_unit = u.Quantity(W, unit=w_unit) if w_unit is not u.UNITLESS else W
                self._W = brainstate.ParamState(W_with_unit)
                self.comm = _DenseMatMul(self._W)
            else:  # sparse CSR event matmul — memory-light for large fan-out
                w_mant, w_unit = self._edge_weight(weight, spec.n_edges, k_w)
                self.comm = _SparseEventMatMul(spec.pre_idx, spec.post_idx, w_mant, w_unit,
                                               n_pre=n_pre, n_post=n_post)

        # Delay buffers the FULL pre-population vector (axonal granularity).
        self.delay_seam = InputDelay((self._n_pre_pop,), delay) if delay is not None else None

        # Precompute (Python bool) whether this projection targets the whole
        # post population, so update() can skip the scatter on the fast path.
        self._post_is_full = (
            n_post == self._n_post_pop
            and bool(jnp.all(self.post_local_idx == jnp.arange(self._n_post_pop)))
        )

    @staticmethod
    def _edge_weight(weight, n_edges, key):
        """Resolve ``weight`` to a per-edge ``(mantissa, unit)`` pair."""
        w_edge = resolve_param(weight, (n_edges,), key)
        if isinstance(w_edge, u.Quantity):
            return u.split_mantissa_unit(w_edge)
        return jnp.asarray(w_edge), u.UNITLESS

    def update(self):
        x_full = self.pre_spike()                       # (n_pre_pop,)
        if self.delay_seam is not None:
            x_full = self.delay_seam.update(x_full)
        x_seg = jnp.asarray(x_full)[self.pre_local_idx]  # (n_pre,)
        if self._receptor:
            y = self.comm(x_seg)                        # (n_post, n_receptors)
            contrib = y if self._post_is_full else self._scatter_receptor(y)
            if self._receptor_keyed:
                # GLIF-style self-pull: one labelled deposit per port. The label
                # composes to key ``'receptor_k // <delta_key>'``, which the post's
                # ``sum_delta_inputs(label='receptor_k')`` selects.
                for k in range(self._n_receptors):
                    self.post.add_delta_input(self._delta_key, contrib[..., k], label=f'receptor_{k}')
            else:
                # Blob: one (n_post, n_receptors) deposit, assembled by the bridge.
                self.post.add_delta_input(self._delta_key, contrib)
        else:
            if self._one_to_one:
                y = x_seg * self._weight                # (n_post,) pA
            else:
                y = self.comm(x_seg)                    # (n_post,) pA
            contrib = y if self._post_is_full else self._scatter(y)
            self.post.add_delta_input(self._delta_key, contrib)

    def _scatter(self, y):
        """Place per-segment contributions into a full (n_post_pop,) vector."""
        if isinstance(y, u.Quantity):
            base = jnp.zeros(self._n_post_pop, dtype=y.mantissa.dtype)
            return u.Quantity(base.at[self.post_local_idx].add(y.mantissa), unit=y.unit)
        base = jnp.zeros(self._n_post_pop, dtype=y.dtype)
        return base.at[self.post_local_idx].add(y)

    def _scatter_receptor(self, y):
        """Place per-segment (n_post, n_receptors) contributions into the full population."""
        shape = (self._n_post_pop, self._n_receptors)
        if isinstance(y, u.Quantity):
            base = jnp.zeros(shape, dtype=y.mantissa.dtype)
            return u.Quantity(base.at[self.post_local_idx].add(y.mantissa), unit=y.unit)
        base = jnp.zeros(shape, dtype=y.dtype)
        return base.at[self.post_local_idx].add(y)
