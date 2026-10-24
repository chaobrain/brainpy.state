# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Connectivity rules: ``rule.build(pre_gids, post_gids, key) -> EdgeSet`` (K0 spec §4.4).

Semantics follow NEST 3.10 (``nestkernel/conn_builder.cpp``) with three guarantees
that fix the goal-01 bugs:

* **Global ids.** Autapses are decided on global ids, so disjoint slices of one
  population (``pop[0:5] -> pop[5:10]``) lose no edges, and partially overlapping or
  concatenated views exclude exactly the self pairs.
* **One population per view.** A concatenated view is passed as one gid array and
  sampled as one population (``fixed_indegree(K)`` from ``E + I`` gives ``K`` edges
  per target, not ``K`` per part).
* **Storage/batch independence.** Random rules draw from per-post-neuron keys
  (:func:`~brainspike._kernel.rng.post_keys`) or a projection-wide stream, with
  fixed 32-bit dtypes, so the realised :class:`EdgeSet` depends only on the
  projection key, the views and the rule — never on storage, batch size or
  ``jax_enable_x64``.

All rules are vectorised (no Python loop over neurons). ``pairwise_bernoulli`` and
the without-replacement branch of ``fixed_indegree`` work through bounded blocks of
post neurons and never materialise an ``(N_pre, N_post)`` matrix.

Edges are returned in canonical order: post-major, then pre ascending (local
indices).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.connectivity.edgeset import EdgeSet, as_gids
from brainspike._kernel.rng import CONNECT_STEP, GLOBAL_BLOCK, block_key, post_keys, step_key
from brainspike._kernel.xla import kernel_jax_jit

__all__ = [
    'ConnRule',
    'AllToAll',
    'OneToOne',
    'FixedIndegree',
    'FixedTotalNumber',
    'PairwiseBernoulli',
    'all_to_all',
    'one_to_one',
    'fixed_indegree',
    'fixed_total_number',
    'pairwise_bernoulli',
]

#: Upper bound on random numbers drawn per block (bounds peak memory of the
#: blockwise rules to a few tens of MB).
_BLOCK_ELEMS = 1 << 22


def _check_bool(x: Any, name: str) -> bool:
    if not isinstance(x, (bool, np.bool_)):
        raise TypeError(f'{name} must be a bool, got {x!r}')
    return bool(x)


def _check_count(x: Any, name: str) -> int:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise TypeError(f'{name} must be an int, got {x!r}')
    if x < 0:
        raise ValueError(f'{name} must be non-negative, got {x}')
    return int(x)


def _self_positions(pre_gids: np.ndarray, post_gids: np.ndarray) -> np.ndarray:
    """For each post neuron, its local index in the pre view, or -1 if absent."""
    if pre_gids.size == 0 or post_gids.size == 0:
        return np.full(post_gids.shape, -1, np.int64)
    order = np.argsort(pre_gids, kind='stable')
    s = pre_gids[order]
    pos = np.clip(np.searchsorted(s, post_gids), 0, s.size - 1)
    return np.where(s[pos] == post_gids, order[pos], -1).astype(np.int64)


def _edges(pre_idx: Any, post_idx: Any, pre_gids: np.ndarray, post_gids: np.ndarray) -> EdgeSet:
    """Canonicalise (post-major, pre ascending) and build the EdgeSet."""
    pi = np.asarray(pre_idx, np.int64).reshape(-1)
    qi = np.asarray(post_idx, np.int64).reshape(-1)
    order = np.lexsort((pi, qi))
    return EdgeSet.create(pi[order], qi[order], pre_gids, post_gids)


@dataclass(frozen=True)
class ConnRule:
    """Base class of connectivity rules.

    Parameters
    ----------
    allow_autapses : bool, default True
        Allow an edge from a neuron to itself (decided on global ids).
    allow_multapses : bool, default True
        Allow several edges between the same ``(pre, post)`` pair.
    """

    allow_autapses: bool = True
    allow_multapses: bool = True

    #: Whether :meth:`build` consumes random numbers (and therefore needs a key).
    random: ClassVar[bool] = False

    def __post_init__(self) -> None:
        _check_bool(self.allow_autapses, 'allow_autapses')
        _check_bool(self.allow_multapses, 'allow_multapses')

    def build(self, pre_gids: Any, post_gids: Any, key: jax.Array | None = None, *,
              post_keys: jax.Array | None = None) -> EdgeSet:
        """Realise the edges between two views.

        Parameters
        ----------
        pre_gids, post_gids : array_like of int
            Global ids of the source and target views (concatenations allowed; no
            duplicates within a view).
        key : jax.Array, optional
            Projection key ``rng.object_key(seed, name)``; required by random rules.
        post_keys : jax.Array, optional
            Per-post-neuron keys, shape ``(n_post, 2)``, used by the per-target
            rules instead of ``rng.post_keys(key, post_gids)``. The compiler passes
            ``rng.view_post_keys`` (population name + local id, D-g) so the edges do
            not depend on global-id offsets.

        Returns
        -------
        EdgeSet
            Edges in canonical order, without weights or delays.
        """
        pre = as_gids(pre_gids, 'pre_gids')
        post = as_gids(post_gids, 'post_gids')
        if self.random and key is None:
            raise ValueError(f'{type(self).__name__} is random and needs a key')
        if post_keys is not None:
            post_keys = jnp.asarray(post_keys)
            if post_keys.shape[:1] != (post.size,):
                raise ValueError(f'post_keys must have {post.size} rows (one per post neuron), '
                                 f'got shape {post_keys.shape}')
        return self._build(pre, post, key, post_keys)

    @staticmethod
    def _keys_of(key: Any, post: np.ndarray, pkeys: Any) -> jax.Array:
        """Per-post-neuron keys: the ones passed to :meth:`build`, else from the gids."""
        return post_keys(key, post) if pkeys is None else pkeys

    def _build(self, pre: np.ndarray, post: np.ndarray, key: Any, pkeys: Any = None) -> EdgeSet:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class AllToAll(ConnRule):
    """Every pre neuron to every post neuron (NEST ``all_to_all``).

    ``allow_multapses`` has no effect (each pair is connected once).
    """

    def _build(self, pre, post, key, pkeys=None):
        n_pre, n_post = pre.size, post.size
        qi = np.repeat(np.arange(n_post, dtype=np.int64), n_pre)
        pi = np.tile(np.arange(n_pre, dtype=np.int64), n_post)
        if not self.allow_autapses:
            keep = pre[pi] != post[qi]
            pi, qi = pi[keep], qi[keep]
        return EdgeSet.create(pi, qi, pre, post)


@dataclass(frozen=True)
class OneToOne(ConnRule):
    """The ``i``-th pre neuron to the ``i``-th post neuron (NEST ``one_to_one``).

    Raises
    ------
    ValueError
        At build time if the views differ in size.
    """

    def _build(self, pre, post, key, pkeys=None):
        if pre.size != post.size:
            raise ValueError(f'one_to_one requires equal sizes, got n_pre={pre.size}, n_post={post.size}')
        i = np.arange(pre.size, dtype=np.int64)
        if not self.allow_autapses:
            i = i[pre != post]
        return EdgeSet.create(i, i, pre, post)


@dataclass(frozen=True)
class FixedIndegree(ConnRule):
    """Exactly ``indegree`` incoming edges per post neuron (NEST ``fixed_indegree``).

    Sources are drawn uniformly from the pre view, excluding the post neuron itself
    when ``allow_autapses=False``; without multapses they are distinct.

    Parameters
    ----------
    indegree : int
        Edges per post neuron.
    allow_autapses, allow_multapses : bool
        See :class:`ConnRule`.

    Raises
    ------
    ValueError
        At build time if a post neuron has fewer available sources than required
        (none at all, or fewer than ``indegree`` without multapses).

    Notes
    -----
    With multapses each source is ``randint`` over the available sources. Without
    multapses, small ``indegree`` uses Floyd's algorithm (``O(K**2)`` per target);
    otherwise the ``indegree`` largest of ``n_pre`` uniforms (``top_k``), computed in
    bounded blocks of post neurons. Both are exactly uniform over ``K``-subsets.
    """

    indegree: int = 0
    random: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_count(self.indegree, 'indegree')

    def _build(self, pre, post, key, pkeys=None):
        k = self.indegree
        n_pre, n_post = pre.size, post.size
        if k == 0 or n_post == 0:
            return EdgeSet.create(np.zeros(0, np.int64), np.zeros(0, np.int64), pre, post)
        selfpos = (np.full(n_post, -1, np.int64) if self.allow_autapses
                   else _self_positions(pre, post))
        has_self = selfpos >= 0
        n_avail = n_pre - has_self.astype(np.int64)
        if n_avail.min() == 0:
            raise ValueError('fixed_indegree: a target has no available source '
                             '(empty pre view, or its only source is itself with allow_autapses=False)')
        if not self.allow_multapses and k > n_avail.min():
            raise ValueError(f'fixed_indegree: indegree {k} exceeds the {int(n_avail.min())} available '
                             f'sources of a target with allow_multapses=False')
        keys = self._keys_of(key, post, pkeys)
        sp = jnp.asarray(selfpos, jnp.int32)
        av = jnp.asarray(n_avail, jnp.int32)
        if self.allow_multapses:
            idx = _sample_with_replacement(keys, av, sp, k)
        elif k * k <= 4 * n_pre:
            idx = _sample_floyd(keys, av, sp, k)
        else:
            idx = _sample_topk(keys, sp, k, n_pre)
        qi = np.repeat(np.arange(n_post, dtype=np.int64), k)
        return _edges(np.asarray(idx).reshape(-1), qi, pre, post)


def _skip_self(idx: jax.Array, selfpos: jax.Array) -> jax.Array:
    """Map ``idx`` in ``[0, n_avail)`` to pre indices skipping ``selfpos`` (if >= 0)."""
    return idx + ((selfpos >= 0) & (idx >= selfpos)).astype(idx.dtype)


@kernel_jax_jit(static_argnums=3)
def _sample_with_replacement(keys, n_avail, selfpos, k):
    def one(key, m, s):
        idx = jax.random.randint(key, (k,), 0, m, dtype=jnp.int32)
        return _skip_self(idx, s)
    return jax.vmap(one)(keys, n_avail, selfpos)


@kernel_jax_jit(static_argnums=3)
def _sample_floyd(keys, n_avail, selfpos, k):
    def one(key, m, s):
        def body(i, chosen):
            j = m - k + i
            t = jax.random.randint(jax.random.fold_in(key, i), (), 0, j + 1, dtype=jnp.int32)
            return chosen.at[i].set(jnp.where(jnp.any(chosen == t), j, t))
        chosen = jax.lax.fori_loop(0, k, body, jnp.full((k,), -1, jnp.int32))
        return _skip_self(chosen, s)
    return jax.vmap(one)(keys, n_avail, selfpos)


@kernel_jax_jit(static_argnums=(2, 3))
def _sample_topk(keys, selfpos, k, n_pre):
    ar = jnp.arange(n_pre, dtype=jnp.int32)

    def one(args):
        key, s = args
        score = jax.random.uniform(key, (n_pre,), jnp.float32)
        score = jnp.where(ar == s, -1.0, score)
        return jax.lax.top_k(score, k)[1].astype(jnp.int32)

    return jax.lax.map(one, (keys, selfpos), batch_size=max(1, _BLOCK_ELEMS // max(n_pre, 1)))


@dataclass(frozen=True)
class PairwiseBernoulli(ConnRule):
    """Each ``(pre, post)`` pair independently with probability ``p`` (NEST ``pairwise_bernoulli``).

    ``allow_multapses`` has no effect (each pair is drawn once), as in NEST.

    Parameters
    ----------
    p : float
        Connection probability in ``[0, 1]``.
    allow_autapses, allow_multapses : bool
        See :class:`ConnRule`.

    Notes
    -----
    Post neurons are processed in blocks of ``~2**22 / n_pre`` targets; each block
    is one jitted draw followed by a host compaction, so peak memory is bounded
    independently of ``n_pre * n_post``.
    """

    p: float = 0.0
    random: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        p = float(self.p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'p must be in [0, 1], got {self.p}')
        object.__setattr__(self, 'p', p)

    def _build(self, pre, post, key, pkeys=None):
        n_pre, n_post = pre.size, post.size
        if self.p == 0.0 or n_pre == 0 or n_post == 0:
            return EdgeSet.create(np.zeros(0, np.int64), np.zeros(0, np.int64), pre, post)
        selfpos = (np.full(n_post, -1, np.int64) if self.allow_autapses
                   else _self_positions(pre, post))
        keys = self._keys_of(key, post, pkeys)
        sp = jnp.asarray(selfpos, jnp.int32)
        chunk = max(1, min(n_post, _BLOCK_ELEMS // n_pre))
        pis, qis = [], []
        for start in range(0, n_post, chunk):  # loop over blocks of targets, not neurons
            stop = min(start + chunk, n_post)
            kb, sb = keys[start:stop], sp[start:stop]
            if stop - start < chunk:  # pad so every block reuses one compilation
                pad = chunk - (stop - start)
                kb = jnp.concatenate([kb, jnp.zeros((pad,) + kb.shape[1:], kb.dtype)])
                sb = jnp.concatenate([sb, jnp.full((pad,), -1, jnp.int32)])
            mask = np.asarray(_bernoulli_block(kb, sb, self.p, n_pre))[: stop - start]
            q, p_ = np.nonzero(mask)
            pis.append(p_)
            qis.append(q + start)
        return EdgeSet.create(np.concatenate(pis), np.concatenate(qis), pre, post)


@kernel_jax_jit(static_argnums=(2, 3))
def _bernoulli_block(keys, selfpos, p, n_pre):
    ar = jnp.arange(n_pre, dtype=jnp.int32)

    def one(key, s):
        return (jax.random.uniform(key, (n_pre,), jnp.float32) < p) & (ar != s)

    return jax.vmap(one)(keys, selfpos)


@dataclass(frozen=True)
class FixedTotalNumber(ConnRule):
    """Exactly ``N`` edges, each uniform over the valid pairs (NEST ``fixed_total_number``).

    Mirrors NEST: pairs ``(pre, post)`` are drawn uniformly and autapses are rejected
    and redrawn when ``allow_autapses=False``; multapses are always allowed.

    Parameters
    ----------
    N : int
        Total number of edges.
    allow_autapses : bool
        See :class:`ConnRule`.
    allow_multapses : bool
        Must be True: NEST raises ``NotImplemented`` for ``False`` and so does this rule.

    Raises
    ------
    NotImplementedError
        On ``allow_multapses=False`` (at construction).
    ValueError
        At build time if ``N > 0`` and there is no valid pair.

    Notes
    -----
    The pairs come from one projection-wide stream (``gid_block = GLOBAL_BLOCK``),
    drawn in rounds of bounded size; accepted pairs are kept in draw order, which is
    exactly NEST's sequential rejection sampling.
    """

    N: int = 0
    random: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_count(self.N, 'N')
        if not self.allow_multapses:
            raise NotImplementedError(
                "fixed_total_number does not support allow_multapses=False (as in NEST: "
                "\"Connect doesn't support the suppression of multapses in the FixedTotalNumber connector\")")

    def _build(self, pre, post, key, pkeys=None):
        n_pre, n_post = pre.size, post.size
        n_total = self.N
        if n_total == 0:
            return EdgeSet.create(np.zeros(0, np.int64), np.zeros(0, np.int64), pre, post)
        n_self = 0 if self.allow_autapses else int((_self_positions(pre, post) >= 0).sum())
        n_valid = n_pre * n_post - n_self
        if n_valid <= 0:
            raise ValueError('fixed_total_number: there is no valid (pre, post) pair')
        accept = n_valid / (n_pre * n_post)
        gkey = step_key(block_key(key, GLOBAL_BLOCK), CONNECT_STEP)
        pis, qis, remaining, rnd = [], [], n_total, 0
        while remaining > 0:  # rounds of bounded size, not a loop over edges
            m = min(_BLOCK_ELEMS, int(math.ceil(remaining / accept * 1.05)) + 32)
            s, t = _pair_draw(jax.random.fold_in(gkey, rnd), m, n_pre, n_post)
            s, t = np.asarray(s, np.int64), np.asarray(t, np.int64)
            if not self.allow_autapses:
                ok = pre[s] != post[t]
                s, t = s[ok], t[ok]
            s, t = s[:remaining], t[:remaining]
            pis.append(s)
            qis.append(t)
            remaining -= s.size
            rnd += 1
        return _edges(np.concatenate(pis), np.concatenate(qis), pre, post)


@kernel_jax_jit(static_argnums=(1, 2, 3))
def _pair_draw(key, m, n_pre, n_post):
    ks, kt = jax.random.split(key)
    return (jax.random.randint(ks, (m,), 0, n_pre, dtype=jnp.int32),
            jax.random.randint(kt, (m,), 0, n_post, dtype=jnp.int32))


# NEST-vocabulary aliases: ``rule=fixed_indegree(80)``.
all_to_all = AllToAll
one_to_one = OneToOne


def fixed_indegree(indegree: int, allow_autapses: bool = True,
                   allow_multapses: bool = True) -> FixedIndegree:
    """:class:`FixedIndegree` with ``indegree`` as the first positional argument."""
    return FixedIndegree(allow_autapses=allow_autapses, allow_multapses=allow_multapses,
                         indegree=indegree)


def fixed_total_number(N: int, allow_autapses: bool = True,
                       allow_multapses: bool = True) -> FixedTotalNumber:
    """:class:`FixedTotalNumber` with ``N`` as the first positional argument."""
    return FixedTotalNumber(allow_autapses=allow_autapses, allow_multapses=allow_multapses, N=N)


def pairwise_bernoulli(p: float, allow_autapses: bool = True,
                       allow_multapses: bool = True) -> PairwiseBernoulli:
    """:class:`PairwiseBernoulli` with ``p`` as the first positional argument."""
    return PairwiseBernoulli(allow_autapses=allow_autapses, allow_multapses=allow_multapses, p=p)
