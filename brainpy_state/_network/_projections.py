# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Rule-based projection classes for the Network API."""
from __future__ import annotations

from typing import Optional

import brainstate
import jax
import jax.numpy as jnp
import saiunit as u

from brainpy_state._base import Dynamics
from brainpy_state._brainpy.projection import AlignPostProj
from brainpy_state._network._connectivity import (
    ConnSpec,
    sample_one_to_one,
    sample_all_to_all,
    sample_pairwise_bernoulli,
    sample_fixed_indegree,
    sample_fixed_outdegree,
    sample_fixed_total_number,
    sample_pairwise_poisson,
    resolve_param,
)

__all__ = [
    'OneToOneProj', 'AllToAllProj',
    'PairwiseBernoulliProj', 'SymmetricPairwiseBernoulliProj',
    'FixedIndegreeProj', 'FixedOutdegreeProj',
    'FixedTotalNumberProj', 'PairwisePoissonProj',
]


def _pre_output(pre: Dynamics):
    """Pull the per-step output from a pre-synaptic module.

    Convention: spiking populations expose a ``spike`` State; rate
    populations expose ``r`` or a callable. We default to ``.spike``
    when present, else call the module.
    """
    if hasattr(pre, 'spike'):
        return pre.spike.value
    return pre()


def _size(module: Dynamics) -> int:
    sz = module.in_size if hasattr(module, 'in_size') else module.varshape
    if isinstance(sz, tuple):
        n = 1
        for s in sz:
            n *= int(s)
        return n
    return int(sz)


class _DenseMatMul(brainstate.nn.Module):
    """Tiny comm module: input @ W.

    Delay support is deferred to a follow-up — pass ``delay=None`` only.
    """
    __module__ = 'brainpy.state'

    def __init__(self, weight_state: brainstate.ParamState):
        super().__init__()
        self._W = weight_state

    def update(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __call__(self, x):
        W = self._W.value
        if isinstance(W, u.Quantity):
            return u.Quantity(jnp.asarray(x) @ W.mantissa, unit=W.unit)
        return jnp.asarray(x) @ W


class _RuleProj(brainstate.nn.Module):
    """Shared base for rule-based projections.

    Subclasses override ``_build_conn_spec(self, key) -> ConnSpec``.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        pre: Dynamics,
        post: Dynamics,
        *,
        weight,
        delay=None,
        syn,
        out,
        allow_autapses: bool = True,
        allow_multapses: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if not isinstance(pre, Dynamics):
            raise TypeError(f'pre must be a Dynamics instance, got {type(pre).__name__}')
        if not isinstance(post, Dynamics):
            raise TypeError(f'post must be a Dynamics instance, got {type(post).__name__}')
        if delay is not None:
            raise NotImplementedError(
                'delay support is deferred to a follow-up; v1 supports delay=None only'
            )

        self.pre = pre
        self.post = post
        self.allow_autapses = allow_autapses
        self.allow_multapses = allow_multapses
        self._pre_is_post = pre is post

        # 1. Sample connectivity
        key = jax.random.key(0 if seed is None else int(seed))
        k_conn, k_w = jax.random.split(key, 2)
        spec = self._build_conn_spec(k_conn)

        # 2. Per-edge weight, then scatter into a dense (n_pre, n_post) matrix
        n_pre = _size(pre)
        n_post = _size(post)
        if spec.n_edges == 0:
            W_with_unit = jnp.zeros((n_pre, n_post))
        else:
            w_per_edge = resolve_param(weight, (spec.n_edges,), k_w)
            if isinstance(w_per_edge, u.Quantity):
                w_mantissa, w_unit = u.split_mantissa_unit(w_per_edge)
            else:
                w_mantissa, w_unit = jnp.asarray(w_per_edge), u.UNITLESS
            W = jnp.zeros((n_pre, n_post), dtype=w_mantissa.dtype)
            W = W.at[spec.pre_idx, spec.post_idx].add(w_mantissa)
            W_with_unit = (u.Quantity(W, unit=w_unit) if w_unit is not u.UNITLESS else W)
        self._weight_matrix = brainstate.ParamState(W_with_unit)

        self.delay = delay

        # 3. Build comm and inner AlignPostProj
        comm = _DenseMatMul(self._weight_matrix)
        self._inner = AlignPostProj(comm=comm, syn=syn, out=out, post=post)

    def _build_conn_spec(self, key) -> ConnSpec:  # pragma: no cover - abstract
        raise NotImplementedError

    def update(self, *args, **kwargs):
        x = _pre_output(self.pre)
        self._inner(x)


class OneToOneProj(_RuleProj):
    r"""One-to-one connection: edge ``(i, i)`` for ``i = 0..N-1``.

    Requires ``len(pre) == len(post)``.
    """
    __module__ = 'brainpy.state'

    def _build_conn_spec(self, key):
        del key
        return sample_one_to_one(_size(self.pre), _size(self.post))


class AllToAllProj(_RuleProj):
    """All-to-all connectivity. Honors ``allow_autapses`` when pre is post."""
    __module__ = 'brainpy.state'

    def _build_conn_spec(self, key):
        del key
        return sample_all_to_all(
            _size(self.pre), _size(self.post),
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
        )


class PairwiseBernoulliProj(_RuleProj):
    """Each (pre, post) pair has independent Bernoulli probability ``p``."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, p: float, **kwargs):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f'p must be in [0, 1], got {p}')
        self._p = p
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_pairwise_bernoulli(
            _size(self.pre), _size(self.post),
            p=self._p, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
            allow_multapses=self.allow_multapses,
        )


class SymmetricPairwiseBernoulliProj(_RuleProj):
    """Symmetric Bernoulli: if edge (i,j) exists then (j,i) exists too.

    Requires pre is post.
    """
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, p: float, **kwargs):
        if pre is not post:
            raise ValueError(
                'symmetric_pairwise_bernoulli requires pre is post'
            )
        if not (0.0 <= p <= 1.0):
            raise ValueError(f'p must be in [0, 1], got {p}')
        self._p = p
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        # Draw upper triangle, mirror to lower.
        n = _size(self.pre)
        upper = jax.random.uniform(key, (n, n)) < self._p
        upper = jnp.triu(upper, k=0 if self.allow_autapses else 1)
        mask = upper | upper.T
        if not self.allow_autapses:
            mask = mask & (~jnp.eye(n, dtype=bool))
        pre, post = jnp.where(mask)
        return ConnSpec(pre, post, int(pre.shape[0]))


class FixedIndegreeProj(_RuleProj):
    """Each post-synaptic neuron receives exactly ``K`` incoming edges."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, K: int, **kwargs):
        if K < 0:
            raise ValueError(f'K must be >= 0, got {K}')
        self._K = int(K)
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_fixed_indegree(
            _size(self.pre), _size(self.post),
            K=self._K, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
            allow_multapses=self.allow_multapses,
        )


class FixedOutdegreeProj(_RuleProj):
    """Each pre-synaptic neuron has exactly ``K`` outgoing edges."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, K: int, **kwargs):
        if K < 0:
            raise ValueError(f'K must be >= 0, got {K}')
        self._K = int(K)
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_fixed_outdegree(
            _size(self.pre), _size(self.post),
            K=self._K, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
            allow_multapses=self.allow_multapses,
        )


class FixedTotalNumberProj(_RuleProj):
    """Exactly ``N`` edges drawn uniformly over the (pre, post) grid."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, N: int, **kwargs):
        if N < 0:
            raise ValueError(f'N must be >= 0, got {N}')
        self._N = int(N)
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_fixed_total_number(
            _size(self.pre), _size(self.post),
            N=self._N, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
            allow_multapses=self.allow_multapses,
        )


class PairwisePoissonProj(_RuleProj):
    """Each (pre, post) pair has a Poisson-distributed number of edges with mean ``mean``."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, mean: float, **kwargs):
        if mean < 0:
            raise ValueError(f'mean must be >= 0, got {mean}')
        self._mean = float(mean)
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_pairwise_poisson(
            _size(self.pre), _size(self.post),
            mean=self._mean, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
        )
