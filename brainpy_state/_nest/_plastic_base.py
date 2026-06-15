# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Shared, pure helpers for the rebuilt event-plastic synapse specs.

The rebuilt ``_nest/<model>_synapse.py`` models are NEST-faithful parameter
specs plus a pure ``update(state, ctx)`` rule kernel that runs on the
:class:`~brainpy_state._network._event_plastic.EventPlasticProj` substrate.
This module factors out the construction-time scalar coercion / validation
(NEST error strings preserved) and the per-edge freeze used by every kernel, so
each model file stays focused on its parameters and equations.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import brainunit as u

__all__ = [
    'to_scalar_float', 'to_ms', 'to_unit_interval', 'to_scalar_int',
    'validate_delay', 'validate_receptor_type', 'weight_to_pa', 'unit_of', 'frozen',
]


def to_scalar_float(value, *, name: str) -> float:
    """Coerce ``value`` to a Python ``float``, raising if it is not scalar."""
    arr = np.asarray(u.math.asarray(value, dtype=float), dtype=float)
    if arr.size != 1:
        raise ValueError(f'{name} must be scalar.')
    return float(arr.reshape(()))


def to_ms(value, *, name: str) -> float:
    """Return ``value`` in milliseconds (bare numbers are interpreted as ms)."""
    if isinstance(value, u.Quantity):
        return float(value.to_decimal(u.ms))
    return to_scalar_float(value, name=name)


def to_unit_interval(value, *, name: str) -> float:
    """Coerce to a scalar float and require it lie in ``[0, 1]`` (NEST message)."""
    v = to_scalar_float(value, name=name)
    if v < 0.0 or v > 1.0:
        raise ValueError(f"'{name}' must be in [0,1].")
    return v


def to_scalar_int(value, *, name: str) -> int:
    """Coerce to an integer, raising if ``value`` is not integral (NEST message)."""
    v = to_scalar_float(value, name=name)
    if not float(v).is_integer():
        raise ValueError(f"'{name}' must be an integer.")
    return int(v)


def validate_delay(delay) -> None:
    """Require a finite, strictly-positive axonal delay.

    Grid quantization (NEST rounds to integer steps, minimum one step) is
    delegated to :class:`~brainpy_state._brainpy._delay.InputDelay` on the
    substrate, so the spec only enforces finiteness and positivity here.
    """
    d = float(u.Quantity(delay).to_decimal(u.ms)) if isinstance(delay, u.Quantity) else float(delay)
    if not np.isfinite(d):
        raise ValueError('delay must be finite.')
    if d <= 0.0:
        raise ValueError('delay must be strictly positive.')


def validate_receptor_type(receptor_type) -> int:
    """Require a non-negative integer receptor port."""
    r = int(receptor_type)
    if r < 0:
        raise ValueError('receptor_type must be >= 0.')
    return r


def weight_to_pa(weight):
    """Attach the pA unit to a bare weight; pass a unitful weight through."""
    return weight if isinstance(weight, u.Quantity) else weight * u.pA


def unit_of(q):
    """Return the unit of a Quantity (helper so specs that bind a parameter
    named ``u`` need not reference the shadowed ``brainunit`` module by name)."""
    return u.get_unit(q)


def frozen(fired, new, old):
    """Keep ``new`` where the edge fired this step, else hold ``old``."""
    return jnp.where(fired, new, old)
