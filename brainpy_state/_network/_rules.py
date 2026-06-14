# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Connection rules as values, wrapping the internal connectivity samplers."""
from __future__ import annotations

from brainpy_state._network._connectivity import (
    ConnSpec,
    sample_all_to_all,
    sample_one_to_one,
    sample_fixed_indegree,
    sample_pairwise_bernoulli,
)

__all__ = ['ConnRule', 'all_to_all', 'one_to_one', 'fixed_indegree',
           'pairwise_bernoulli']


class ConnRule:
    """Base class for connection rules.

    A rule maps ``(n_pre, n_post)`` plus sampling flags to a
    :class:`~brainpy_state._network._connectivity.ConnSpec` of edge indices.
    """
    __module__ = 'brainpy.state'

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses) -> ConnSpec:
        raise NotImplementedError


class _AllToAll(ConnRule):
    __module__ = 'brainpy.state'

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        return sample_all_to_all(n_pre, n_post, pre_is_post=pre_is_post,
                                 allow_autapses=allow_autapses)


class _OneToOne(ConnRule):
    __module__ = 'brainpy.state'

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        return sample_one_to_one(n_pre, n_post)


class _FixedIndegree(ConnRule):
    """Each post-synaptic neuron receives exactly ``K`` incoming edges."""
    __module__ = 'brainpy.state'

    def __init__(self, K: int):
        self.K = int(K)

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        return sample_fixed_indegree(n_pre, n_post, K=self.K, key=key,
                                     pre_is_post=pre_is_post,
                                     allow_autapses=allow_autapses,
                                     allow_multapses=allow_multapses)


class _PairwiseBernoulli(ConnRule):
    """Each ordered ``(pre, post)`` pair is connected independently with prob. ``p``."""
    __module__ = 'brainpy.state'

    def __init__(self, p: float):
        self.p = float(p)

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        return sample_pairwise_bernoulli(n_pre, n_post, p=self.p, key=key,
                                         pre_is_post=pre_is_post,
                                         allow_autapses=allow_autapses,
                                         allow_multapses=allow_multapses)


all_to_all = _AllToAll()
one_to_one = _OneToOne()


def fixed_indegree(K: int) -> _FixedIndegree:
    """Return a fixed-indegree rule: each post neuron gets exactly ``K`` edges."""
    if int(K) < 0:
        raise ValueError(f'K must be >= 0, got {K}')
    return _FixedIndegree(int(K))


def pairwise_bernoulli(p: float) -> _PairwiseBernoulli:
    """Return a pairwise-Bernoulli rule: connect each ``(pre, post)`` pair with prob. ``p``."""
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f'p must be in [0, 1], got {p}')
    return _PairwiseBernoulli(p)
