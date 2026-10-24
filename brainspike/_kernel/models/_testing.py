# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Test helpers for the K0 models: single-population rollouts with ``for_loop``.

Used by the model tests (and the NEST parity tests) until the T5 runner exists.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from brainspike._kernel.context import StepContext
from brainspike._kernel.model import NeuronModel

__all__ = ['rollout', 'step_inputs']


def step_inputs(n_steps: int, n: int, events: Mapping[str, tuple[Sequence[int], Sequence[float], Any]],
                dtype: Any = None) -> dict[str, Any]:
    """Dense per-step input arrays from sparse ``(steps, values, unit)`` lists.

    Parameters
    ----------
    n_steps, n : int
        Number of steps and of neurons.
    events : mapping
        ``port -> (steps, values, unit)``; ``values`` are added to every neuron at
        the given steps (repeated steps accumulate).
    dtype : dtype, optional
        Mantissa dtype (default: JAX's default float).

    Returns
    -------
    dict of str to brainunit.Quantity
        ``port -> (n_steps, n)`` arrays.
    """
    out = {}
    for port, (steps, values, unit) in events.items():
        arr = jnp.zeros((n_steps, n), dtype=dtype if dtype is not None else float)
        arr = arr.at[jnp.asarray(steps, jnp.int32)].add(jnp.asarray(values, arr.dtype)[:, None])
        out[port] = u.Quantity(arr, unit=unit)
    return out


def rollout(model: NeuronModel, n_steps: int, dt: Any, inputs: Mapping[str, Any] | None = None,
            record: Sequence[str] = (), key: jax.Array | None = None,
            input_fn: Callable[[Any], Mapping[str, Any]] | None = None) -> tuple[Any, dict]:
    """Run ``model`` for ``n_steps`` with ``brainstate.transform.for_loop``.

    Parameters
    ----------
    model : NeuronModel
        Model (states are advanced in place).
    n_steps : int
        Number of steps.
    dt : brainunit.Quantity
        Step size.
    inputs : mapping, optional
        ``port -> (n_steps, n)`` per-step payloads (see :func:`step_inputs`).
    record : sequence of str
        State names to record after every step.
    key : jax.Array, optional
        Root key; step ``i`` receives ``fold_in(key, i)``.
    input_fn : callable, optional
        ``input_fn(i) -> {port: payload}`` used instead of ``inputs``.

    Returns
    -------
    outputs : dict
        Stacked output ports, shape ``(n_steps, n)``.
    trace : dict
        Stacked recorded states.
    """
    inputs = dict(inputs or {})

    def body(i):
        k = None if key is None else jax.random.fold_in(key, i)
        ctx = StepContext(t=i * dt, dt=dt, step=i, key=k)
        inp = input_fn(i) if input_fn is not None else {p: v[i] for p, v in inputs.items()}
        out = model.update(ctx, inp)
        return out, {r: model.dyn_states[r].value for r in record}

    return brainstate.transform.for_loop(body, jnp.arange(n_steps))
