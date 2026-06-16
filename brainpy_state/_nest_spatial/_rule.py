# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Distance-dependent pairwise-Bernoulli connection rule.

``spatial_pairwise_bernoulli`` is an ordinary :class:`~brainpy_state._network._rules.ConnRule`
that additionally needs the pre/post node coordinates. It is constructed *unbound*; the
:class:`~brainpy_state.Simulator` binds sliced positions via :meth:`SpatialConnRule.with_coords`
at ``connect`` time, then the existing static / plastic projection paths sample it unchanged.
The distance + Bernoulli draw are fully vectorized over the ``(n_pre, n_post)`` grid -- no
Python pairwise loop (CLAUDE.md rule 10).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from brainpy_state._network._rules import ConnRule
from brainpy_state._network._connectivity import ConnSpec
from brainpy_state._nest_spatial._distance import pairwise_distance

__all__ = ['SpatialConnRule', 'spatial_pairwise_bernoulli']


class SpatialConnRule(ConnRule):
    """Spatially-resolved pairwise-Bernoulli rule: ``p(d)`` within an optional ``mask``.

    Each ordered ``(pre, post)`` pair is connected independently with probability
    ``mask(d) * p(d)``, where ``d`` is their Euclidean distance. Constructed unbound;
    :meth:`Simulator.connect` binds sliced ``(n_pre, d)`` / ``(n_post, d)`` positions.

    Parameters
    ----------
    p : callable or float
        A distance kernel ``p(d)`` (e.g. :func:`~brainpy_state._nest_spatial._kernels.gaussian`)
        or a constant probability.
    mask : object, optional
        A mask with ``contains(pre_pos, post_pos) -> bool (n_pre, n_post)`` (hard cutoff).
    allow_autapses : bool, optional
        Rule-level self-connection switch (ANDed with the connect-level flag).
    """
    __module__ = 'brainpy.state'
    _is_spatial = True

    def __init__(self, p, mask=None, allow_autapses=True, _pre_pos=None, _post_pos=None):
        self.p = p
        self.mask = mask
        self.allow_autapses = bool(allow_autapses)
        self._pre_pos = _pre_pos
        self._post_pos = _post_pos

    def with_coords(self, pre_pos, post_pos) -> 'SpatialConnRule':
        """Return a bound clone carrying sliced ``(n_pre, d)`` / ``(n_post, d)`` positions."""
        return SpatialConnRule(self.p, self.mask, self.allow_autapses, pre_pos, post_pos)

    def _prob_matrix(self):
        if hasattr(self.p, '_eval_pair'):                          # spatial kernel (scalar / per-axis)
            prob = jnp.asarray(self.p._eval_pair(self._pre_pos, self._post_pos))
        elif callable(self.p):                                     # bare p(distance) callable
            d = pairwise_distance(self._pre_pos, self._post_pos)   # Quantity (n_pre, n_post)
            prob = jnp.asarray(self.p(d))
        else:
            prob = jnp.broadcast_to(jnp.asarray(float(self.p)),
                                    (self._pre_pos.shape[0], self._post_pos.shape[0]))
        prob = jnp.clip(prob, 0.0, 1.0)
        if self.mask is not None:
            cand = self.mask.contains(self._pre_pos, self._post_pos)
            prob = jnp.where(cand, prob, 0.0)
        return prob

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        # Multapses are not meaningful for a single Bernoulli trial per pair (flag exists
        # for API symmetry with the base rules).
        del allow_multapses
        if self._pre_pos is None or self._post_pos is None:
            raise ValueError(
                'spatial rule used without bound positions; connect through a Simulator '
                'whose pre/post populations were created with create(positions=...).'
            )
        prob = self._prob_matrix()
        if pre_is_post and not (allow_autapses and self.allow_autapses):
            prob = prob * (1.0 - jnp.eye(n_pre, n_post, dtype=prob.dtype))
        draw = jax.random.uniform(key, (n_pre, n_post)) < prob
        pre_idx, post_idx = jnp.where(draw)
        return ConnSpec(pre_idx, post_idx, int(pre_idx.shape[0]))


def spatial_pairwise_bernoulli(p, mask=None, allow_autapses=True) -> SpatialConnRule:
    """Distance-dependent pairwise-Bernoulli connection rule (NEST spatial ``pairwise_bernoulli``).

    Parameters
    ----------
    p : callable or float
        Distance kernel ``p(d)`` (e.g. ``gaussian(distance, std=...)``) or a constant.
    mask : object, optional
        A spatial mask (``circular`` / ``spherical`` / ``box``) hard cutoff.
    allow_autapses : bool, optional
        Whether self-connections are permitted on a layer-to-itself connect.

    Returns
    -------
    SpatialConnRule
        A rule for ``Simulator.connect(pre, post, rule=...)`` where both populations
        were created with ``create(positions=...)``.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> rule = bp.spatial.spatial_pairwise_bernoulli(
        ...     p=bp.spatial.gaussian(bp.spatial.distance, std=0.5),
        ...     mask=bp.spatial.circular(3.0))
    """
    return SpatialConnRule(p, mask=mask, allow_autapses=allow_autapses)
