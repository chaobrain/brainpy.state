# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Connection rules as values, wrapping the internal connectivity samplers."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from brainpy_state._nest_network.connectivity import (
    ConnSpec,
    sample_all_to_all,
    sample_one_to_one,
    sample_fixed_indegree,
    sample_pairwise_bernoulli,
    sample_fixed_total_number,
    build_pool_map,
    sample_third_factor_pairing,
)

__all__ = ['ConnRule', 'all_to_all', 'one_to_one', 'fixed_indegree',
           'pairwise_bernoulli', 'fixed_total_number',
           'third_factor_bernoulli_with_pool', 'explicit_edges']


class ConnRule:
    """Base class for connection rules.

    A rule maps ``(n_pre, n_post)`` plus sampling flags to a
    :class:`~brainpy_state._nest_network.connectivity.ConnSpec` of edge indices.
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


class _FixedTotalNumber(ConnRule):
    """Exactly ``N`` edges drawn uniformly over the ``(pre, post)`` grid."""
    __module__ = 'brainpy.state'

    def __init__(self, N: int):
        self.N = int(N)

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        return sample_fixed_total_number(n_pre, n_post, N=self.N, key=key,
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


def fixed_total_number(N: int) -> _FixedTotalNumber:
    """Return a fixed-total-number rule: exactly ``N`` edges over the ``(pre, post)`` grid."""
    if int(N) < 0:
        raise ValueError(f'N must be >= 0, got {N}')
    return _FixedTotalNumber(int(N))


class _ExplicitEdges(ConnRule):
    """A rule that returns a *precomputed* :class:`ConnSpec`, ignoring sampling args.

    Lets :meth:`~brainpy_state._nest_network.simulator.Simulator.tripartite_connect`
    feed a single shared edge sample into the existing
    :class:`~brainpy_state._nest_network.event_proj.EventProjection` /
    ``_connect_pair`` / ``_connect_sic`` paths -- which would otherwise re-sample
    their own connectivity from ``(rule, seed)``. ``sample`` returns the wrapped
    spec verbatim, so the key / autapse / multapse flags (already applied when the
    spec was built) are deliberately ignored.
    """
    __module__ = 'brainpy.state'

    def __init__(self, spec: ConnSpec):
        self._spec = spec

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        return self._spec


def explicit_edges(pre_idx, post_idx) -> _ExplicitEdges:
    """Return a rule wiring exactly the given ``(pre_idx[i], post_idx[i])`` edges.

    A connection rule built from a *precomputed* edge list, for topologies that are
    easier to enumerate directly than to express through a sampling rule (e.g. the
    structured inhibitory graph of a constraint-satisfaction network). The edges are
    handed to the same projection paths used by the sampling rules, so a single call
    realizes the whole topology as **one** projection -- avoiding the per-edge /
    per-group projection explosion (and the per-``cont()`` retrace) of many small
    :meth:`~brainpy.state.Simulator.connect` calls.

    Parameters
    ----------
    pre_idx, post_idx : array_like of int
        Equal-length 1-D arrays of segment-local source/target neuron indices, in the
        coordinate frame of the populations passed to
        :meth:`~brainpy.state.Simulator.connect`. Edge ``i`` connects
        ``pre_idx[i] -> post_idx[i]``. Order is preserved and duplicates are kept --
        the caller controls multapses.

    Returns
    -------
    _ExplicitEdges
        A connection rule returning the precomputed edge set verbatim (the sampling
        ``key`` / autapse / multapse flags are ignored).

    Raises
    ------
    ValueError
        If ``pre_idx`` / ``post_idx`` are not 1-D, differ in length, or are not
        integer-typed.

    See Also
    --------
    brainpy.state.all_to_all
    brainpy.state.Simulator.connect

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from brainpy import state as bp
        >>> sim = bp.Simulator(dt=0.1 * u.ms)
        >>> pop = sim.create(bp.iaf_psc_exp, 5)
        >>> # wire 0->1, 0->4, 2->3 as one sparse projection
        >>> pre = np.array([0, 0, 2])
        >>> post = np.array([1, 4, 3])
        >>> _ = sim.connect(pop, pop, rule=bp.explicit_edges(pre, post),
        ...                 weight=-0.2 * u.pA, comm='sparse')
        >>> len(sim.get_connections(source=pop, target=pop))
        3
    """
    pre = jnp.asarray(pre_idx)
    post = jnp.asarray(post_idx)
    if pre.ndim != 1 or post.ndim != 1:
        raise ValueError(
            f'pre_idx/post_idx must be 1-D, got {pre.ndim}-D and {post.ndim}-D')
    if pre.shape[0] != post.shape[0]:
        raise ValueError(
            f'pre_idx/post_idx must have equal length, got {pre.shape[0]} and {post.shape[0]}')
    if not (jnp.issubdtype(pre.dtype, jnp.integer) and jnp.issubdtype(post.dtype, jnp.integer)):
        raise ValueError(
            f'pre_idx/post_idx must be integer arrays, got dtypes {pre.dtype} and {post.dtype}')
    return _ExplicitEdges(ConnSpec(pre, post, int(pre.shape[0])))


class _ThirdFactorBernoulliWithPool:
    """The ``third_factor_bernoulli_with_pool`` connection spec (NEST tripartite rule).

    Holds the conditional pairing probability and pool parameters; given a realized
    primary edge sample it derives the ``third_in`` (pre->astro) and ``third_out``
    (astro->post) edge sets via :func:`build_pool_map` +
    :func:`sample_third_factor_pairing`. Built by
    :func:`third_factor_bernoulli_with_pool` and consumed by
    :meth:`~brainpy_state._nest_network.simulator.Simulator.tripartite_connect`.
    """
    __module__ = 'brainpy.state'

    def __init__(self, p: float, pool_size: int, pool_type: str):
        self.p = float(p)
        self.pool_size = int(pool_size)
        self.pool_type = pool_type

    def sample_third(self, primary_spec: ConnSpec, n_post: int, n_third: int, *, key):
        """Derive the ``(third_in, third_out)`` :class:`ConnSpec` pair from the primary sample.

        Parameters
        ----------
        primary_spec : ConnSpec
            The realized primary ``pre->post`` edges (segment-local indices).
        n_post : int
            Target-population size (for the pool map / block index).
        n_third : int
            Third-factor (astrocyte) population size.
        key : jax.Array
            PRNG key; split into a pool-assignment key and a pairing key.

        Returns
        -------
        third_in : ConnSpec
            ``pre_i -> astro`` edges (segment-local), one per paired primary edge.
        third_out : ConnSpec
            ``astro -> post_j`` edges (segment-local), one per paired primary edge.
        """
        k_pool, k_pair = jax.random.split(key)
        pool_map = build_pool_map(n_post, n_third, pool_size=self.pool_size,
                                  pool_type=self.pool_type, key=k_pool)
        tin_pre, astro, tout_post = sample_third_factor_pairing(
            primary_spec.pre_idx, primary_spec.post_idx, pool_map, p=self.p, key=k_pair)
        n = int(tin_pre.shape[0])
        return ConnSpec(tin_pre, astro, n), ConnSpec(astro, tout_post, n)


def third_factor_bernoulli_with_pool(*, p: float, pool_size: int, pool_type: str):
    """Return a ``third_factor_bernoulli_with_pool`` spec (NEST tripartite astro-pool rule).

    Used as the ``third_factor_conn_spec`` of
    :meth:`~brainpy_state._nest_network.simulator.Simulator.tripartite_connect`: for
    each realized primary ``pre->post`` edge, an independent Bernoulli(``p``) trial
    decides whether it is paired with one astrocyte drawn from the target neuron's
    pool of ``pool_size`` astrocytes.

    Parameters
    ----------
    p : float
        Conditional pairing probability ``p_third_if_primary`` in ``[0, 1]``.
    pool_size : int
        Astrocytes per target pool (``>= 1``; the upper bound ``<= n_third`` is
        checked when the populations are known).
    pool_type : {'random', 'block'}
        Pool-assignment scheme (see :func:`build_pool_map`).

    Returns
    -------
    _ThirdFactorBernoulliWithPool
        The spec object.

    Raises
    ------
    ValueError
        If ``p`` is outside ``[0, 1]``, ``pool_size < 1``, or ``pool_type`` is unknown.

    See Also
    --------
    brainpy.state.Simulator.tripartite_connect

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> rule = bp.third_factor_bernoulli_with_pool(
        ...     p=0.5, pool_size=10, pool_type='random')
        >>> rule.p, rule.pool_size, rule.pool_type
        (0.5, 10, 'random')

    The returned spec is passed as the ``third_factor_conn_spec`` of
    :meth:`~brainpy.state.Simulator.tripartite_connect`.
    """
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f'p must be in [0, 1], got {p}')
    if int(pool_size) < 1:
        raise ValueError(f'pool_size must be >= 1, got {pool_size}')
    if pool_type not in ('random', 'block'):
        raise ValueError(f"pool_type must be 'random' or 'block', got {pool_type!r}")
    return _ThirdFactorBernoulliWithPool(p, int(pool_size), pool_type)
