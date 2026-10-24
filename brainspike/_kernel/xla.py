# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Per-jit XLA compiler options of the kernel's compiled programs (K1a-9).

Every program the kernel compiles (the runner's rollout and single step, the
learners, the connectivity samplers) is jitted through :func:`kernel_jit`, which
passes :data:`KERNEL_COMPILER_OPTIONS` as **per-jit** ``compiler_options``: the
user's own ``jax.jit`` functions and the process-wide ``XLA_FLAGS`` are untouched.

``xla_gpu_redzone_padding_bytes``
    While autotuning a fusion, XLA's GPU compiler allocates every operand and result
    of each candidate with a redzone of this many bytes on either side (default
    8 MiB) and checks the redzones after the trial run. The scratch is part of the
    device's ``peak_bytes_in_use`` and dominates the peak of small networks (Brunel
    order 200: ≈ 60 MiB of redzones for a ≈ 5 MiB network). The kernel uses 64 KiB.

    **Trade-off.** Redzones detect out-of-bounds writes of *candidate* kernels
    during autotuning (a bug in an XLA emitter, Triton or a library kernel), so a
    candidate writing more than 64 KiB past a buffer is no longer caught by the
    redzone check. It does not change which kernel runs, numerics or step time
    (measured on Brunel orders 200 and 2500: no systematic change of ms/step between
    8 MiB, 1 MiB, 64 KiB and 0), and XLA's own kernels are not expected to write out
    of bounds; on CPU the option has no effect. To restore XLA's default, clear
    ``KERNEL_COMPILER_OPTIONS`` before building a runner or learner (the
    connectivity samplers read it when :mod:`~brainspike._kernel.connectivity.rules`
    is imported).
"""
from __future__ import annotations

from typing import Any, Callable

import brainstate
import jax

__all__ = ['KERNEL_COMPILER_OPTIONS', 'compiler_options', 'kernel_jit', 'kernel_jax_jit']

#: XLA compiler options passed to every kernel jit (read when the jit is created).
KERNEL_COMPILER_OPTIONS: dict[str, Any] = {'xla_gpu_redzone_padding_bytes': 64 * 1024}


def compiler_options() -> dict[str, Any]:
    """A copy of :data:`KERNEL_COMPILER_OPTIONS` (what a new kernel jit receives)."""
    return dict(KERNEL_COMPILER_OPTIONS)


def kernel_jit(fun: Callable, **kwargs: Any) -> Any:
    """``brainstate.transform.jit`` with the kernel's per-jit compiler options.

    Parameters
    ----------
    fun : callable
        Function to compile (it may read and write ``brainstate.State``).
    **kwargs
        Forwarded to :func:`brainstate.transform.jit` (``static_argnums`` ...).

    Returns
    -------
    JittedFunction

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainspike._kernel.xla import kernel_jit
        >>> f = kernel_jit(lambda x: x * 2)
        >>> float(f(jnp.float32(3.0)))
        6.0
    """
    return brainstate.transform.jit(fun, compiler_options=compiler_options(), **kwargs)


def kernel_jax_jit(**kwargs: Any) -> Callable[[Callable], Any]:
    """Decorator: ``jax.jit`` (pure functions) with the kernel's per-jit compiler options."""
    def wrap(fun: Callable) -> Any:
        return jax.jit(fun, compiler_options=compiler_options(), **kwargs)
    return wrap
