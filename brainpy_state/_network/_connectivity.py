# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Internal connectivity samplers and parameter resolver.

Each sampler returns a ``ConnSpec`` of (pre_idx, post_idx) int arrays
plus n_edges. Projection classes turn these into a dense
``(n_pre, n_post)`` weight matrix.

These helpers are private to ``brainpy_state._network``. Public APIs
are the ``*Proj`` classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u

from brainpy_state._dist import Distribution


@dataclass
class ConnSpec:
    pre_idx: jnp.ndarray
    post_idx: jnp.ndarray
    n_edges: int


def sample_one_to_one(n_pre: int, n_post: int) -> ConnSpec:
    if n_pre != n_post:
        raise ValueError(
            f'one_to_one requires equal sizes, got n_pre={n_pre}, n_post={n_post}'
        )
    idx = jnp.arange(n_pre)
    return ConnSpec(idx, idx, int(n_pre))


def sample_all_to_all(
    n_pre: int,
    n_post: int,
    *,
    pre_is_post: bool,
    allow_autapses: bool,
) -> ConnSpec:
    pre = jnp.repeat(jnp.arange(n_pre), n_post)
    post = jnp.tile(jnp.arange(n_post), n_pre)
    if pre_is_post and not allow_autapses:
        mask = pre != post
        pre = pre[mask]
        post = post[mask]
    return ConnSpec(pre, post, int(pre.shape[0]))


def sample_pairwise_bernoulli(
    n_pre: int,
    n_post: int,
    *,
    p: float,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
    allow_multapses: bool,
) -> ConnSpec:
    # Multapses not meaningful for Bernoulli (single trial per pair) — flag
    # exists for API symmetry. allow_multapses is ignored here.
    del allow_multapses
    mask = jax.random.uniform(key, (n_pre, n_post)) < p
    if pre_is_post and not allow_autapses:
        mask = mask & (1 - jnp.eye(n_pre, n_post, dtype=jnp.int32)).astype(bool)
    pre, post = jnp.where(mask)
    return ConnSpec(pre, post, int(pre.shape[0]))


def sample_fixed_indegree(
    n_pre: int,
    n_post: int,
    *,
    K: int,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
    allow_multapses: bool,
) -> ConnSpec:
    if K < 0:
        raise ValueError(f'K must be >= 0, got {K}')
    pre_lists = []
    post_lists = []
    for j in range(n_post):
        sub = jax.random.fold_in(key, j)
        if pre_is_post and not allow_autapses:
            candidates = jnp.concatenate([
                jnp.arange(0, j),
                jnp.arange(j + 1, n_pre),
            ])
        else:
            candidates = jnp.arange(n_pre)
        if allow_multapses:
            chosen = jax.random.choice(sub, candidates, (K,), replace=True)
        else:
            if K > candidates.shape[0]:
                raise ValueError(
                    f'cannot pick {K} unique pre for post {j} from '
                    f'{candidates.shape[0]} candidates with allow_multapses=False'
                )
            chosen = jax.random.choice(sub, candidates, (K,), replace=False)
        pre_lists.append(chosen)
        post_lists.append(jnp.full((K,), j))
    return ConnSpec(jnp.concatenate(pre_lists), jnp.concatenate(post_lists), int(K * n_post))


def sample_fixed_outdegree(
    n_pre: int,
    n_post: int,
    *,
    K: int,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
    allow_multapses: bool,
) -> ConnSpec:
    if K < 0:
        raise ValueError(f'K must be >= 0, got {K}')
    pre_lists = []
    post_lists = []
    for i in range(n_pre):
        sub = jax.random.fold_in(key, i)
        if pre_is_post and not allow_autapses:
            candidates = jnp.concatenate([
                jnp.arange(0, i),
                jnp.arange(i + 1, n_post),
            ])
        else:
            candidates = jnp.arange(n_post)
        if allow_multapses:
            chosen = jax.random.choice(sub, candidates, (K,), replace=True)
        else:
            if K > candidates.shape[0]:
                raise ValueError(
                    f'cannot pick {K} unique post for pre {i} from '
                    f'{candidates.shape[0]} candidates with allow_multapses=False'
                )
            chosen = jax.random.choice(sub, candidates, (K,), replace=False)
        pre_lists.append(jnp.full((K,), i))
        post_lists.append(chosen)
    return ConnSpec(jnp.concatenate(pre_lists), jnp.concatenate(post_lists), int(K * n_pre))


def sample_fixed_total_number(
    n_pre: int,
    n_post: int,
    *,
    N: int,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
    allow_multapses: bool,
) -> ConnSpec:
    if N < 0:
        raise ValueError(f'N must be >= 0, got {N}')
    k1, k2 = jax.random.split(key)
    pre = jax.random.randint(k1, (N,), 0, n_pre)
    post = jax.random.randint(k2, (N,), 0, n_post)
    if pre_is_post and not allow_autapses:
        # Resample autapses one extra round; if any remain, raise.
        mask = pre == post
        if bool(jnp.any(mask)):
            k3 = jax.random.fold_in(key, 1)
            replacement = jax.random.randint(k3, (int(jnp.sum(mask)),), 0, n_post)
            post = post.at[mask].set(replacement)
            if bool(jnp.any(pre == post)):
                raise ValueError(
                    'failed to remove autapses in fixed_total_number with one resample pass'
                )
    if not allow_multapses:
        # Deduplicate pairs; document that this can produce fewer than N edges
        # — NEST documents the same behaviour.
        pairs_np = np.stack([np.asarray(pre), np.asarray(post)], axis=1)
        _, unique_idx = np.unique(pairs_np, axis=0, return_index=True)
        unique_idx = jnp.asarray(np.sort(unique_idx))
        pre = pre[unique_idx]
        post = post[unique_idx]
    return ConnSpec(pre, post, int(pre.shape[0]))


def sample_pairwise_poisson(
    n_pre: int,
    n_post: int,
    *,
    mean: float,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
) -> ConnSpec:
    counts = jax.random.poisson(key, mean, (n_pre, n_post))
    if pre_is_post and not allow_autapses:
        diag_n = min(n_pre, n_post)
        counts = counts.at[jnp.arange(diag_n), jnp.arange(diag_n)].set(0)
    pre = jnp.repeat(jnp.arange(n_pre)[:, None], n_post, axis=1).reshape(-1)
    post = jnp.tile(jnp.arange(n_post), n_pre)
    counts_flat = counts.reshape(-1)
    repeats = counts_flat.astype(jnp.int32)
    total = int(jnp.sum(repeats))
    pre = jnp.repeat(pre, repeats, total_repeat_length=total)
    post = jnp.repeat(post, repeats, total_repeat_length=total)
    return ConnSpec(pre, post, total)


def resolve_param(value, shape, key):
    """Turn scalar | array | Distribution into a concrete array of ``shape``."""
    if isinstance(value, Distribution):
        return value.sample(shape, key)
    if isinstance(value, u.Quantity):
        mantissa = jnp.asarray(value.mantissa)
        if mantissa.ndim == 0 or mantissa.shape == (1,):
            return u.Quantity(jnp.broadcast_to(mantissa, shape), unit=value.unit)
        if tuple(mantissa.shape) != tuple(shape):
            raise ValueError(
                f'parameter array shape {mantissa.shape} does not match target {shape}'
            )
        return u.Quantity(mantissa, unit=value.unit)
    arr = jnp.asarray(value)
    if arr.ndim == 0 or arr.shape == (1,):
        return jnp.broadcast_to(arr, shape)
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(
            f'parameter array shape {arr.shape} does not match target {shape}'
        )
    return arr
