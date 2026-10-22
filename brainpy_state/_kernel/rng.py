# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Counter-based RNG key derivation (K0 spec §4.5, architecture doc §7.2, D-g).

Every random stream of a simulation is a pure function of four counters::

    key(seed, obj, gid_block, step)
        = fold_in(fold_in(object_key(seed, obj), gid_block), step)

``obj`` identifies the network object. Inside a ``Network`` it is the object's
**name** (K1a-6, lead decision D-g): :func:`object_key` folds a stable 64-bit hash
of the name (:func:`name_id`) into the root key, so adding, removing or reordering
an unrelated population or projection leaves every other object's streams
unchanged. Unnamed objects are auto-named ``<model>_<k>`` per model class by the
``Network``, so insertion order matters only among unnamed objects of the same class.
A plain integer ``obj`` is folded in as a raw counter (low-level use and tests).

Streams
-------
* **Dynamics noise, per step** (:func:`population_block_keys`, :func:`step_block_keys`,
  :func:`draw_blocked`): a population's per-step draws are generated block by block
  over fixed blocks of :data:`GID_BLOCK` *population-local* ids; block ``b`` of step
  ``n`` uses ``step_key(block_key(object_key, b), n)`` (then the trial index), and a
  neuron's numbers depend only on ``(seed, name, local block, lane, step, trial)``.
  A population extended beyond a block, or split into block-aligned shards, draws
  the same numbers for the neurons it had. ``StepContext.key`` holds these
  ``(n_blocks, 2)`` keys; models draw through :func:`draw_blocked`.
* **Connectivity** (:func:`connectivity_key`, :func:`post_keys`,
  :func:`view_post_keys`): a projection's edges are generated per post neuron with
  the reserved step value :data:`CONNECT_STEP`, so they never collide with a
  simulation step; each post neuron gets its own lane ``i % GID_BLOCK`` inside block
  ``i // GID_BLOCK`` of its population's local ids (the population's name is folded
  into the projection key first).
* **Trials** (:func:`trial_keys`): under ``vmap`` batching the trial index is folded
  in last. ``'independent'`` gives trial ``b`` the key ``fold_in(key, b)``;
  ``'common'`` gives every trial the key of trial 0. An unbatched run *is* trial 0,
  so trial 0 is identical for any batch size and either mode.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'CONNECT_STEP',
    'GLOBAL_BLOCK',
    'GID_BLOCK',
    'RNG_MODES',
    'name_id',
    'n_blocks',
    'root_key',
    'object_key',
    'block_key',
    'step_key',
    'derive_key',
    'population_key',
    'connectivity_key',
    'post_keys',
    'view_post_keys',
    'population_block_keys',
    'step_block_keys',
    'draw_blocked',
    'trial_keys',
]

#: Reserved ``step`` counter of connectivity streams (never reached by a simulation:
#: 2**32 - 1 steps of 0.1 ms is about five days of model time).
CONNECT_STEP = 0xFFFFFFFF

#: Reserved ``gid_block`` counter of projection-wide streams that are not tied to a
#: post-gid block (e.g. the pair draws of ``fixed_total_number``).
GLOBAL_BLOCK = 0xFFFFFFFF

#: Number of consecutive (population-local) ids sharing one ``gid_block`` counter; one
#: constant for connectivity and dynamics noise (lead decision 2026-10-22, K1a-6).
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


def name_id(name: str) -> tuple[int, int]:
    """Stable 64-bit id of an object name, as two ``uint32`` words ``(hi, lo)``.

    The first eight bytes of ``blake2b(name.encode('utf-8'))`` (big-endian), so the id
    is the same in every process, Python version and platform (unlike ``hash``).

    Parameters
    ----------
    name : str
        Non-empty object name.

    Returns
    -------
    tuple of int
        ``(hi, lo)``, each in ``[0, 2**32)``.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._kernel.rng import name_id
        >>> name_id('E') == name_id('E'), name_id('E') == name_id('I')
        (True, False)
    """
    if not isinstance(name, str) or not name:
        raise TypeError(f'object name must be a non-empty str, got {name!r}')
    d = hashlib.blake2b(name.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(d[:4], 'big'), int.from_bytes(d[4:], 'big')


def object_key(seed: int, obj: str | int) -> jax.Array:
    """Key of one network object.

    Parameters
    ----------
    seed : int
        Simulation seed.
    obj : str or int
        Object **name** (``fold_in(fold_in(root_key(seed), hi), lo)`` with
        ``(hi, lo) = name_id(obj)``; what a ``Network`` uses), or a raw integer
        counter (``fold_in(root_key(seed), obj)``; low-level use).

    Returns
    -------
    jax.Array
        Raw ``uint32[2]`` key. For a projection this is the ``key`` its connectivity
        rule receives; for a population the root of its init and noise streams.
    """
    root = root_key(seed)
    if isinstance(obj, str):
        hi, lo = name_id(obj)
        return _fold(_fold(root, hi), lo)
    return _fold(root, _counter(obj, 'obj'))


def block_key(obj_key: jax.Array, gid_block: Any) -> jax.Array:
    """``fold_in(obj_key, gid_block)`` (``gid_block`` may be an array under ``vmap``)."""
    return _fold(obj_key, _counter(gid_block, 'gid_block'))


def step_key(blk_key: jax.Array, step: Any) -> jax.Array:
    """``fold_in(blk_key, step)`` (``step`` may be a traced integer)."""
    return _fold(blk_key, _counter(step, 'step'))


def derive_key(seed: int, obj: str | int, gid_block: Any = 0, step: Any = 0) -> jax.Array:
    """The §4.5 key ``key(seed, obj, gid_block, step)``.

    Parameters
    ----------
    seed : int
        Simulation seed.
    obj : str or int
        Object name (or raw integer counter), see :func:`object_key`.
    gid_block : int or jax.Array, default 0
        Id block (``local_id // GID_BLOCK``).
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
    return step_key(block_key(object_key(seed, obj), gid_block), step)


def population_key(seed: int, obj: str | int, step: Any, gid_block: Any = 0) -> jax.Array:
    """Key of block ``gid_block`` of population ``obj`` at step ``step`` (before the trial).

    Equal to ``derive_key(seed, obj, gid_block, step)``; the runner folds the trial index
    into it (:func:`step_block_keys`).
    """
    return derive_key(seed, obj, gid_block, step)


def connectivity_key(seed: int, obj: str | int, gid_block: Any) -> jax.Array:
    """Connectivity key of projection ``obj`` for one post-id block.

    Equal to ``derive_key(seed, obj, gid_block, CONNECT_STEP)``.
    """
    return derive_key(seed, obj, gid_block, CONNECT_STEP)


def post_keys(obj_key: jax.Array, post_gids: Any, block_size: int = GID_BLOCK) -> jax.Array:
    """One connectivity key per post neuron.

    ``key_j = fold_in(step_key(block_key(obj_key, gid_j // block_size), CONNECT_STEP),
    gid_j % block_size)``: the key of a post neuron depends only on the key and its id,
    not on which other neurons are in the view. A ``Network`` passes population-local
    ids with a per-population key (:func:`view_post_keys`); a rule built without them
    uses the views' ids directly.

    Parameters
    ----------
    obj_key : jax.Array
        Projection key (:func:`object_key`).
    post_gids : array_like of int, shape (n,)
        Ids of the post neurons.
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


def view_post_keys(obj_key: jax.Array, parts: Any) -> jax.Array:
    """Connectivity keys of a post view, from population names and local ids.

    For each part ``(population name, local ids)`` (in view order) the population's
    name is folded into the projection key and :func:`post_keys` is applied to the
    *local* ids: ``post_keys(fold_in(fold_in(obj_key, hi), lo), local)`` with ``(hi,
    lo) = name_id(name)``. A post neuron's key then depends only on the projection,
    its population's name and its local id -- not on the global id offset, so adding
    or removing an unrelated (earlier) population leaves the realised edges unchanged.

    Parameters
    ----------
    obj_key : jax.Array
        Projection key (:func:`object_key`).
    parts : sequence of (str, array_like of int)
        Contiguous runs of the post view: population name and local ids.

    Returns
    -------
    jax.Array
        Keys of shape ``(n_post, 2)``.
    """
    out = []
    for name, local in parts:
        hi, lo = name_id(name)
        out.append(post_keys(_fold(_fold(obj_key, hi), lo), local))
    if not out:
        return jnp.zeros((0,) + obj_key.shape, obj_key.dtype)
    return out[0] if len(out) == 1 else jnp.concatenate(out, axis=0)


def n_blocks(n: int, block_size: int = GID_BLOCK) -> int:
    """Number of id blocks covering ``n`` neurons (``ceil(n / block_size)``)."""
    return -(-int(n) // int(block_size))


def population_block_keys(seed: int, obj: str | int, n: int) -> jax.Array:
    """Per-block keys of a population, before the step: ``block_key(object_key, b)``.

    Parameters
    ----------
    seed : int
        Simulation seed.
    obj : str or int
        Population name (or raw counter).
    n : int
        Population size.

    Returns
    -------
    jax.Array
        Shape ``(n_blocks(n), 2)``; row ``b`` covers local ids
        ``[b * GID_BLOCK, (b + 1) * GID_BLOCK)``.
    """
    okey = object_key(seed, obj)
    nb = n_blocks(n)
    if nb == 0:
        return jnp.zeros((0,) + okey.shape, okey.dtype)
    return jax.vmap(lambda b: block_key(okey, b))(jnp.arange(nb, dtype=jnp.uint32))


def step_block_keys(blk_keys: jax.Array, step: Any, trial: Any = 0) -> jax.Array:
    """``fold_in(step_key(blk_key, step), trial)`` for every block key (traced ok).

    This is ``StepContext.key`` of a population at step ``step`` of trial ``trial``
    (``trial = 0`` for an unbatched run and every trial under ``rng='common'``).

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> from brainpy_state._kernel import rng
        >>> bk = rng.population_block_keys(0, 'E', 2500)
        >>> ks = rng.step_block_keys(bk, 7)
        >>> ks.shape
        (3, 2)
        >>> bool((ks[2] == jax.random.fold_in(rng.population_key(0, 'E', 7, 2), 0)).all())
        True
    """
    step = _counter(step, 'step')
    trial = _counter(trial, 'trial')
    return jax.vmap(lambda k: _fold(_fold(k, step), trial))(blk_keys)


def _is_single_key(key: jax.Array) -> bool:
    if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        return key.ndim == 0
    return key.ndim == 1


def draw_blocked(key: jax.Array, n: int, sampler: Callable[..., jax.Array], *lane_args: Any,
                 tail: tuple = ()) -> jax.Array:
    """Draw ``n`` per-neuron random values block by block (dynamics-noise layout, D-g).

    Block ``b`` (local ids ``[b B, (b + 1) B)``, ``B = GID_BLOCK``) calls
    ``sampler(key_b, (B,) + tail, *args_b)`` with its own key and *always* a full
    block, so neuron ``i``'s value depends only on ``key_{i // B}``, its lane
    ``i % B`` and its own lane arguments: identical whether the population has
    ``i + 1`` or ``10**6`` neurons, and whether it is drawn as one array or as
    block-aligned pieces (as a sharded runtime would).

    Parameters
    ----------
    key : jax.Array
        Either the stacked block keys ``(n_blocks(n), 2)`` (raw) / ``(n_blocks(n),)``
        (typed) -- what the runner puts in ``StepContext.key`` -- or one key, from
        which block ``b`` uses ``fold_in(key, b)`` (standalone use of a model).
    n : int
        Number of neurons (static).
    sampler : callable
        ``sampler(key, shape, *args) -> jax.Array`` of shape ``shape`` (e.g.
        ``lambda k, s: jax.random.uniform(k, s)``). Must be element-wise in the lane
        arguments for the invariance above to hold.
    *lane_args : array_like
        Per-neuron arguments, broadcast to ``(n,)`` and padded with zeros to whole
        blocks; the sampler receives them with shape ``(B,)``.
    tail : tuple of int, default ()
        Trailing shape of every neuron's draw.

    Returns
    -------
    jax.Array
        Shape ``(n,) + tail``.

    Raises
    ------
    ValueError
        If stacked keys do not have ``n_blocks(n)`` rows.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> from brainpy_state._kernel import rng
        >>> keys = rng.step_block_keys(rng.population_block_keys(0, 'noise', 3000), 5)
        >>> uni = lambda k, s: jax.random.uniform(k, s)
        >>> a = rng.draw_blocked(keys, 3000, uni)
        >>> b = rng.draw_blocked(keys[:2], 1500, uni)   # a smaller population
        >>> a.shape, bool((a[:1500] == b).all())
        ((3000,), True)
    """
    n = int(n)
    tail = tuple(int(t) for t in tail)
    nb = n_blocks(n)
    key = jnp.asarray(key) if not isinstance(key, jax.Array) else key
    if _is_single_key(key):
        key = jax.vmap(lambda b: _fold(key, b))(jnp.arange(nb, dtype=jnp.uint32))
    elif key.shape[0] != nb:
        raise ValueError(f'expected {nb} block keys for {n} neurons (blocks of {GID_BLOCK}), '
                         f'got key array of shape {key.shape}')
    B = GID_BLOCK
    args = []
    for a in lane_args:
        a = jnp.broadcast_to(jnp.asarray(a), (n,))
        args.append(jnp.pad(a, (0, nb * B - n)).reshape(nb, B))
    out = jax.vmap(lambda k, *xs: sampler(k, (B,) + tail, *xs))(key, *args)
    return out.reshape((nb * B,) + tail)[:n]


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
