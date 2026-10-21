# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""The per-step context passed explicitly to every model ``step``.

Models never read ``brainstate.environ``: time, step size, step index and the
population's RNG key for the step all arrive in a :class:`StepContext`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp

from brainpy_state._kernel.units import check_unit

__all__ = ['StepContext']


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class StepContext:
    """Immutable per-step context (a JAX pytree).

    Parameters
    ----------
    t : brainunit.Quantity
        Simulation time at the **start** of the step (time dimension).
    dt : brainunit.Quantity
        Step size (time dimension).
    step : jax.Array
        Integer step index ``n`` (``t == n * dt`` for runs starting at 0).
    key : jax.Array or None
        RNG key for this population and step (spec §4.5), ``None`` for
        deterministic populations.

    Notes
    -----
    The constructor performs no validation so that JAX can rebuild the pytree
    under transformations; build contexts with :meth:`create`, which validates
    units and normalises dtypes.
    """

    t: Any
    dt: Any
    step: Any
    key: Any = None

    @classmethod
    def create(cls, t: Any, dt: Any, step: Any = 0, key: Any = None) -> 'StepContext':
        """Validated constructor.

        Parameters
        ----------
        t, dt : brainunit.Quantity
            Time at step start and step size; both must have the time dimension.
        step : int or jax.Array, default 0
            Step index; converted to an int32 array.
        key : jax.Array or None
            Per-population RNG key.

        Returns
        -------
        StepContext

        Raises
        ------
        UnitMismatchError
            If ``t`` or ``dt`` is not a time.
        """
        check_unit(t, u.ms, 't')
        check_unit(dt, u.ms, 'dt')
        step = jnp.asarray(step, dtype=jnp.int32)
        if step.ndim != 0:
            raise ValueError(f'step must be a scalar, got shape {step.shape}')
        return cls(t=t, dt=dt, step=step, key=key)

    def replace(self, **changes: Any) -> 'StepContext':
        """Return a copy with ``changes`` applied (no validation)."""
        return dataclasses.replace(self, **changes)

    def advance(self) -> 'StepContext':
        """Context of the next step: ``t + dt``, ``step + 1``, same ``key``.

        The caller derives the next key (spec §4.5); this only moves the clock.
        """
        return dataclasses.replace(self, t=self.t + self.dt, step=self.step + 1)

