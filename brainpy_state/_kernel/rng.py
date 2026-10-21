# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Counter-based RNG key derivation (K0 spec §4.5, architecture doc §7.2).

Every random stream of a simulation is a pure function of four counters::

    key(seed, uid, gid_block, step)
        = fold_in(fold_in(fold_in(PRNGKey(seed), uid), gid_block), step)

``uid`` is the object's deterministic **insertion index** in the ``Network`` (one
counter shared by populations and projections, see :class:`UidAllocator`), never
``id()``. Keys therefore do not depend on evaluation order, storage, batch size or
Python object identity.

Streams
-------
* **Population, per step** (:func:`population_key`): the key a population receives in
  ``StepContext.key`` for step ``n``.
* **Connectivity** (:func:`connectivity_key`, :func:`post_keys`): a projection's
  edges are generated per post-gid block with the reserved step value
  :data:`CONNECT_STEP`, so they never collide with a simulation step; each post
  neuron gets its own lane ``gid % GID_BLOCK`` inside its block.
* **Trials** (:func:`trial_keys`): under ``vmap`` batching the trial index is folded
  in last. ``'independent'`` gives trial ``b`` the key ``fold_in(key, b)``;
  ``'common'`` gives every trial the key of trial 0. An unbatched run *is* trial 0,
  so trial 0 is identical for any batch size and either mode.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'CONNECT_STEP',
    'GLOBAL_BLOCK',
    'GID_BLOCK',
    'RNG_MODES',
    'UidAllocator',
    'root_key',
    'object_key',
    'block_key',
    'step_key',
    'derive_key',
    'population_key',
    'connectivity_key',
    'post_keys',
    'trial_keys',
]

#: Reserved ``step`` counter of connectivity streams (never reached by a simulation:
#: 2**32 - 1 steps of 0.1 ms is about five days of model time).
CONNECT_STEP = 0xFFFFFFFF

#: Reserved ``gid_block`` counter of projection-wide streams that are not tied to a
#: post-gid block (e.g. the pair draws of ``fixed_total_number``).
GLOBAL_BLOCK = 0xFFFFFFFF

#: Number of consecutive global ids sharing one ``gid_block`` counter.
GID_BLOCK = 1024

#: Batched-trial RNG semantics.
RNG_MODES = ('independent', 'common')

_U32_MAX = 0xFFFFFFFF


def _counter(x: Any, name: str) -> Any:
    """Validate a fold-in counter: Python ints in ``[0, 2**32)``; arrays cast to uint32."""
    if isinstance(x, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer, got {x!r}')
    if isinstance(x, (int, np.integer)):
        x = int(x)
        if not 0 <= x <= _U32_MAX:
            raise ValueError(f'{name} must be in [0, 2**32), got {x}')
        return x
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.integer):
        raise TypeError(f'{name} must be an integer, got dtype {x.dtype}')
    return x.astype(jnp.uint32)


def _fold(key: jax.Array, data: Any) -> jax.Array:
    return jax.random.fold_in(key, data)


class UidAllocator:
    """Deterministic insertion-index counter for network objects.

    The ``Network`` owns one allocator and calls :meth:`next` for every population
    *and* projection it adds, so uids are unique across object kinds and reproducible
    from the order of ``add`` / ``connect`` calls alone.

    Parameters
    ----------
    start : int, default 0
        First uid handed out.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._kernel.rng import UidAllocator
        >>> uids = UidAllocator()
        >>> uids.next(), uids.next(), uids.count
        (0, 1, 2)
    """

    def __init__(self, start: int = 0) -> None:
        self._next = _counter(start, 'start')
        self._start = self._next

    def next(self) -> int:
        """Return the next uid and advance the counter."""
        uid = self._next
        if uid > _U32_MAX:
            raise OverflowError('uid counter exhausted (2**32 objects)')
        self._next += 1
        return uid

    @property
    def count(self) -> int:
        """Number of uids handed out so far."""
        return self._next - self._start


def root_key(seed: int) -> jax.Array:
    """Root key of a simulation.

    Parameters
    ----------
    seed : int
        Simulation seed in ``[0, 2**32)``.

    Returns
    -------
    jax.Array
        Raw ``uint32[2]`` threefry key (``jax.random.PRNGKey(seed)``).
    """
    return jax.random.PRNGKey(_counter(seed, 'seed'))


def object_key(seed: int, uid: int) -> jax.Array:
    """Key of one network object: ``fold_in(root_key(seed), uid)``.

    This is the ``key`` a connectivity rule's ``build`` receives for the projection
    with insertion index ``uid``.
    """
    return _fold(root_key(seed), _counter(uid, 'uid'))


def block_key(obj_key: jax.Array, gid_block: Any) -> jax.Array:
    """``fold_in(obj_key, gid_block)`` (``gid_block`` may be an array under ``vmap``)."""
    return _fold(obj_key, _counter(gid_block, 'gid_block'))


def step_key(blk_key: jax.Array, step: Any) -> jax.Array:
    """``fold_in(blk_key, step)`` (``step`` may be a traced integer)."""
    return _fold(blk_key, _counter(step, 'step'))


def derive_key(seed: int, uid: int, gid_block: Any = 0, step: Any = 0) -> jax.Array:
    """The §4.5 key ``key(seed, uid, gid_block, step)``.

    Parameters
    ----------
    seed : int
        Simulation seed.
    uid : int
        Insertion index of the object.
    gid_block : int or jax.Array, default 0
        Global-id block (``gid // GID_BLOCK``).
    step : int or jax.Array, default 0
        Step counter (traced values allowed).

    Returns
    -------
    jax.Array
        Raw ``uint32[2]`` key.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> from brainpy_state._kernel.rng import derive_key, root_key
        >>> k = derive_key(0, 3, 0, 7)
        >>> f = jax.random.fold_in
        >>> bool((k == f(f(f(root_key(0), 3), 0), 7)).all())
        True
    """
    return step_key(block_key(object_key(seed, uid), gid_block), step)


def population_key(seed: int, uid: int, step: Any, gid_block: Any = 0) -> jax.Array:
    """Per-population key for simulation step ``step`` (``StepContext.key``).

    K0 populations draw all their neurons from one key (``gid_block=0``); a sharded
    runtime would pass the block of each shard instead.
    """
    return derive_key(seed, uid, gid_block, step)


def connectivity_key(seed: int, uid: int, gid_block: Any) -> jax.Array:
    """Connectivity key of projection ``uid`` for one post-gid block.

    Equal to ``derive_key(seed, uid, gid_block, CONNECT_STEP)``.
    """
    return derive_key(seed, uid, gid_block, CONNECT_STEP)


def post_keys(obj_key: jax.Array, post_gids: Any, block_size: int = GID_BLOCK) -> jax.Array:
    """One connectivity key per post neuron.

    ``key_j = fold_in(step_key(block_key(obj_key, gid_j // block_size), CONNECT_STEP),
    gid_j % block_size)``: the key of a post neuron depends only on the projection key
    and its *global* id, not on which other neurons are in the view.

    Parameters
    ----------
    obj_key : jax.Array
        Projection key (:func:`object_key`).
    post_gids : array_like of int, shape (n,)
        Global ids of the post neurons.
    block_size : int, default GID_BLOCK
        Gids per block.

    Returns
    -------
    jax.Array
        Keys of shape ``(n, 2)``.
    """
    gids = np.asarray(post_gids, dtype=np.int64).reshape(-1)
    if gids.size and (gids.min() < 0 or gids.max() // block_size > _U32_MAX):
        raise ValueError('post gids must be non-negative and their blocks < 2**32')
    blocks = jnp.asarray(gids // block_size, jnp.uint32)
    lanes = jnp.asarray(gids % block_size, jnp.uint32)

    def one(b, lane):
        return _fold(step_key(block_key(obj_key, b), CONNECT_STEP), lane)

    if gids.size == 0:
        return jnp.zeros((0,) + obj_key.shape, obj_key.dtype)
    return jax.vmap(one)(blocks, lanes)


def trial_keys(key: jax.Array, batch_size: int | None = None,
               mode: str = 'independent') -> jax.Array:
    """Per-trial keys for batched (``vmap``) runs.

    Parameters
    ----------
    key : jax.Array
        Unbatched key of the stream (e.g. a :func:`population_key`).
    batch_size : int or None
        Number of trials; ``None`` for an unbatched run (returns trial 0's key).
    mode : {'independent', 'common'}
        ``'independent'``: trial ``b`` uses ``fold_in(key, b)``; ``'common'``: every
        trial uses trial 0's key (common random numbers).

    Returns
    -------
    jax.Array
        Shape ``key.shape`` when ``batch_size`` is ``None``, else
        ``(batch_size,) + key.shape``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainpy_state._kernel.rng import root_key, trial_keys
        >>> k = root_key(1)
        >>> ks = trial_keys(k, 4, 'independent')
        >>> bool((ks[0] == trial_keys(k)).all()), bool((ks[1] == ks[0]).all())
        (True, False)
        >>> kc = trial_keys(k, 4, 'common')
        >>> bool((kc == kc[0]).all())
        True
    """
    if mode not in RNG_MODES:
        raise ValueError(f'mode must be one of {RNG_MODES}, got {mode!r}')
    k0 = _fold(key, 0)
    if batch_size is None:
        return k0
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError(f'batch_size must be a positive int or None, got {batch_size!r}')
    batch_size = int(batch_size)
    if mode == 'common':
        return jnp.broadcast_to(k0, (batch_size,) + k0.shape)
    return jax.vmap(lambda b: _fold(key, b))(jnp.arange(batch_size, dtype=jnp.uint32))
