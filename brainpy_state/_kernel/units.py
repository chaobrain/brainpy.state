# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Dtype and unit policy helpers for the K0 kernel.

Two numeric tiers exist: ``'float32'`` (default, GPU friendly) and ``'float64'``
(requires ``jax_enable_x64``). Values carry physical units as
:class:`brainunit.Quantity`; kernel code converts them to plain *mantissas* in a
declared unit at the boundary (:func:`to_mantissa`) and wraps results back with
:func:`from_mantissa`. Every unit check raises :class:`UnitMismatchError`, so callers
can rely on one exception type for dimension errors.
"""
from __future__ import annotations

from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'PRECISIONS',
    'UnitMismatchError',
    'resolve_dtype',
    'index_dtype',
    'as_unit',
    'unit_of',
    'same_dim',
    'check_unit',
    'to_mantissa',
    'from_mantissa',
    'grid_steps',
]

#: Distance (in steps) within which a time snaps to the nearest grid point
#: before rounding up (absorbs float error such as ``2.0 / 0.1 = 19.999...``).
GRID_SNAP = 1e-3

#: Supported floating-point tiers.
PRECISIONS = ('float32', 'float64')


class UnitMismatchError(ValueError):
    """A value does not have the physical dimension the kernel expects."""


def resolve_dtype(precision: Any = 'float32') -> np.dtype:
    """Resolve a precision tier to a floating dtype.

    Parameters
    ----------
    precision : {'float32', 'float64'} or numpy/jax float dtype, default 'float32'
        Requested tier. ``None`` means ``'float32'``.

    Returns
    -------
    numpy.dtype
        ``float32`` or ``float64``.

    Raises
    ------
    ValueError
        If ``precision`` is not one of the two tiers.
    RuntimeError
        If ``float64`` is requested while ``jax_enable_x64`` is off (JAX would
        silently downcast to float32 otherwise).
    """
    if precision is None:
        precision = 'float32'
    try:
        dtype = np.dtype(precision)
    except TypeError:
        raise ValueError(f'unknown precision {precision!r}; expected one of {PRECISIONS}') from None
    if dtype.name not in PRECISIONS:
        raise ValueError(f'unknown precision {precision!r}; expected one of {PRECISIONS}')
    if dtype == np.float64 and not jax.config.jax_enable_x64:
        raise RuntimeError(
            "precision 'float64' requires jax_enable_x64 "
            "(call jax.config.update('jax_enable_x64', True) before building the model)"
        )
    return dtype


def index_dtype(n: int) -> np.dtype:
    """Integer dtype able to index ``n`` elements (int32 below 2**31, else int64).

    Parameters
    ----------
    n : int
        Number of elements to index (e.g. edges of a projection).

    Returns
    -------
    numpy.dtype
        ``int32`` or ``int64``.

    Raises
    ------
    ValueError
        If ``n`` is negative.
    RuntimeError
        If int64 is needed while ``jax_enable_x64`` is off.
    """
    n = int(n)
    if n < 0:
        raise ValueError(f'n must be non-negative, got {n}')
    if n < 2 ** 31:
        return np.dtype(np.int32)
    if not jax.config.jax_enable_x64:
        raise RuntimeError(f'{n} elements need int64 indices; enable jax_enable_x64')
    return np.dtype(np.int64)


def as_unit(unit: Any) -> u.Unit:
    """Normalise a unit specification; ``None`` means dimensionless.

    Parameters
    ----------
    unit : brainunit.Unit or None
        Unit to normalise.

    Returns
    -------
    brainunit.Unit
        ``unit`` itself, or ``brainunit.UNITLESS`` for ``None``.

    Raises
    ------
    TypeError
        If ``unit`` is neither a ``Unit`` nor ``None`` (e.g. a ``Quantity`` such as
        ``2 * u.mV`` passed by mistake).
    """
    if unit is None:
        return u.UNITLESS
    if isinstance(unit, u.Unit):
        return unit
    raise TypeError(f'expected a brainunit Unit or None, got {type(unit).__name__}: {unit!r}')


def unit_of(x: Any) -> u.Unit:
    """Unit of ``x`` (``UNITLESS`` for plain numbers and arrays)."""
    return u.get_unit(x)


def same_dim(a: Any, b: Any) -> bool:
    """Whether two values/units have the same physical dimension.

    ``None`` is treated as dimensionless.
    """
    a = u.UNITLESS if a is None else a
    b = u.UNITLESS if b is None else b
    return u.get_dim(a) == u.get_dim(b)


def check_unit(x: Any, unit: Any, name: str = 'value') -> None:
    """Raise :class:`UnitMismatchError` unless ``x`` has the dimension of ``unit``.

    Parameters
    ----------
    x : Any
        Value, Quantity or Unit to check.
    unit : brainunit.Unit or None
        Expected unit (only its dimension matters); ``None`` means dimensionless.
    name : str
        Name used in the error message.
    """
    unit = as_unit(unit)
    if not same_dim(x, unit):
        raise UnitMismatchError(
            f'{name}: expected a value with the dimension of {unit} '
            f'({u.get_dim(unit)}), got unit {unit_of(x)} ({u.get_dim(x)})'
        )


def to_mantissa(x: Any, unit: Any = None, dtype: Any = None) -> jax.Array:
    """Convert ``x`` to a plain array expressed in ``unit``.

    Parameters
    ----------
    x : array_like or brainunit.Quantity
        Value to convert. Its dimension must match ``unit``.
    unit : brainunit.Unit or None
        Target unit; ``None`` means dimensionless.
    dtype : dtype, optional
        Cast the mantissa to this dtype.

    Returns
    -------
    jax.Array
        Mantissa of ``x`` in ``unit``.

    Raises
    ------
    UnitMismatchError
        If the dimensions differ.
    """
    unit = as_unit(unit)
    check_unit(x, unit)
    if isinstance(x, u.Quantity):
        m = x.to_decimal(unit)
    else:
        # dimensionless plain value; a scaled dimensionless unit (e.g. percent) scales it
        m = x if unit.is_unitless else u.Quantity(x).to_decimal(unit)
    m = jnp.asarray(m)
    if dtype is not None:
        m = m.astype(dtype)
    return m


def from_mantissa(m: Any, unit: Any = None) -> Any:
    """Wrap a mantissa back into a Quantity (plain array when dimensionless).

    Parameters
    ----------
    m : array_like
        Mantissa.
    unit : brainunit.Unit or None
        Unit of the mantissa.

    Returns
    -------
    brainunit.Quantity or jax.Array
    """
    unit = as_unit(unit)
    if unit.is_unitless:
        return jnp.asarray(m)
    return u.Quantity(m, unit=unit)


def grid_steps(t: Any, dt: Any) -> jax.Array:
    """Number of steps of a time span, rounded **up** to the grid (NEST semantics).

    NEST converts ``t_ref``, device ``start``/``stop``/``origin`` and spike times
    with ``Time(ms(t)).get_steps()``, which rounds up to the next step. Values
    within :data:`GRID_SNAP` steps (or a few ulp of ``t / dt`` for large times)
    of a grid point snap to it, so float error (``2.0 / 0.1``) does not add a step.

    Parameters
    ----------
    t, dt : array_like
        Time and step size as mantissas in the same unit (e.g. ms); may be
        traced. ``t = inf`` gives ``inf``.

    Returns
    -------
    jax.Array
        Float array of whole step counts (``ceil``-like), same dtype as ``t / dt``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainpy_state._kernel.units import grid_steps
        >>> grid_steps(jnp.array([2.0, 0.25, 0.04, 0.0]), 0.1).tolist()
        [20.0, 3.0, 1.0, 0.0]
    """
    x = jnp.asarray(t) / jnp.asarray(dt)  # true division: always floating
    r = jnp.round(x)
    # snap window: GRID_SNAP steps, widened to a few ulp for large times in float32
    snap = jnp.maximum(GRID_SNAP, 8 * jnp.finfo(x.dtype).eps * jnp.abs(r))
    return jnp.where(jnp.abs(x - r) < snap, r, jnp.ceil(x))
