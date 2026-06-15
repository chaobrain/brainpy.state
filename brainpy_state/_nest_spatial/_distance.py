# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Pairwise displacement and Euclidean distance.

Flat (non-periodic) boundary; ``edge_wrap`` / torus geometry is a documented
non-goal for v1 (no ported demo requires it). Both functions are fully vectorized
over the ``(n_pre, n_post)`` grid -- no Python pairwise loop (CLAUDE.md rule 10).
"""
from __future__ import annotations

import brainunit as u

__all__ = ['displacement', 'pairwise_distance']


def displacement(pre_pos, post_pos):
    """Pairwise displacement ``post - pre``.

    Parameters
    ----------
    pre_pos : Quantity
        ``(n_pre, d)`` source positions.
    post_pos : Quantity
        ``(n_post, d)`` target positions.

    Returns
    -------
    Quantity
        ``(n_pre, n_post, d)`` displacement vectors (target minus source).
    """
    return post_pos[None, :, :] - pre_pos[:, None, :]


def pairwise_distance(pre_pos, post_pos):
    """Pairwise Euclidean distance.

    Parameters
    ----------
    pre_pos : Quantity
        ``(n_pre, d)`` source positions.
    post_pos : Quantity
        ``(n_post, d)`` target positions.

    Returns
    -------
    Quantity
        ``(n_pre, n_post)`` Euclidean distances.
    """
    disp = displacement(pre_pos, post_pos)
    return u.math.sqrt(u.math.sum(disp ** 2, axis=-1))
