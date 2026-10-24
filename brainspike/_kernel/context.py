# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""The per-step context passed explicitly to every model ``step``.

Models never read ``brainstate.environ``: time, step size, step index and the
population's RNG key for the step all arrive in a :class:`StepContext`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp

from brainspike._kernel.units import check_unit

__all__ = ['StepContext', 'SPIKE_MODES']

#: Forward spike modes of a ``Program`` (architecture doc §8.2-2).
SPIKE_MODES = ('hard', 'relaxed')


def check_spike_options(spikes: Any, detach_reset: Any) -> None:
    """Validate the ``spikes`` / ``detach_reset`` options of a program or context."""
    if spikes not in SPIKE_MODES:
        raise ValueError(f'spikes must be one of {SPIKE_MODES}, got {spikes!r}')
    if detach_reset is not None and not isinstance(detach_reset, bool):
        raise TypeError(f'detach_reset must be a bool or None, got {detach_reset!r}')


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
        RNG keys of this population and step (spec §4.5, D-g). Inside a plan: one
        key per block of :data:`~brainspike._kernel.rng.GID_BLOCK` local ids,
        shape ``(n_blocks, 2)``; models draw per-neuron noise with
        :func:`~brainspike._kernel.rng.draw_blocked`, which also accepts a
        single key (standalone use). ``None`` for deterministic populations.
    spikes : {'hard', 'relaxed'}, default 'hard'
        Forward spike mode of the running ``Program`` (static pytree field):
        ``'hard'`` emits exact counts with a surrogate backward pass,
        ``'relaxed'`` smooth spikes with soft jumps (``SpikeRule.check``).
    detach_reset : bool or None, default None
        The running ``Program``'s default ``detach_reset`` for every spike rule
        (static field); ``None`` defers to the model's own setting. A rule's
        explicit ``detach_reset`` always wins.

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
    spikes: str = field(default='hard', metadata=dict(static=True))
    detach_reset: bool | None = field(default=None, metadata=dict(static=True))

    @property
    def relaxed(self) -> bool:
        """Whether the relaxed (smooth) spike forward pass is selected."""
        return self.spikes == 'relaxed'

    def detach_default(self, model_default: bool | None) -> bool | None:
        """Default ``detach_reset`` for a model: this context's, else ``model_default``."""
        return self.detach_reset if self.detach_reset is not None else model_default

    @classmethod
    def create(cls, t: Any, dt: Any, step: Any = 0, key: Any = None, spikes: str = 'hard',
               detach_reset: bool | None = None) -> 'StepContext':
        """Validated constructor.

        Parameters
        ----------
        t, dt : brainunit.Quantity
            Time at step start and step size; both must have the time dimension.
        step : int or jax.Array, default 0
            Step index; converted to an int32 array.
        key : jax.Array or None
            Per-population RNG key(s) (see the class parameters).
        spikes : {'hard', 'relaxed'}, default 'hard'
            Forward spike mode.
        detach_reset : bool or None, default None
            Program-level default ``detach_reset``.

        Returns
        -------
        StepContext

        Raises
        ------
        UnitMismatchError
            If ``t`` or ``dt`` is not a time.
        ValueError
            If ``spikes`` is not a spike mode.
        """
        check_spike_options(spikes, detach_reset)
        check_unit(t, u.ms, 't')
        check_unit(dt, u.ms, 'dt')
        step = jnp.asarray(step, dtype=jnp.int32)
        if step.ndim != 0:
            raise ValueError(f'step must be a scalar, got shape {step.shape}')
        return cls(t=t, dt=dt, step=step, key=key, spikes=spikes, detach_reset=detach_reset)

    def replace(self, **changes: Any) -> 'StepContext':
        """Return a copy with ``changes`` applied (no validation)."""
        return dataclasses.replace(self, **changes)

    def advance(self) -> 'StepContext':
        """Context of the next step: ``t + dt``, ``step + 1``, same ``key``.

        The caller derives the next key (spec §4.5); this only moves the clock.
        """
        return dataclasses.replace(self, t=self.t + self.dt, step=self.step + 1)

