# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
Corrected exponential-Euler integrator + a guarded runtime patch (temporary).

Why this module exists
----------------------
``brainstate.nn.exp_euler_step`` rebuilds the unit of the linearised Jacobian as
``unit(drift) / unit(jacobian)``. ``vector_grad`` returns the Jacobian with its
unit stripped (``unit(jacobian) == 1``), so the reconstructed unit becomes
``unit(drift)`` (e.g. ``mV/ms``) instead of the correct ``unit(drift) /
unit(state)`` (e.g. ``1/ms``). The *mantissa* is correct, so the bug stayed
latent while ``brainunit.math.exprel`` silently accepted dimensional inputs.

``brainunit``/``brainunit`` ``0.4.0`` made ``exprel`` strict — it now raises when
given a dimensional argument without ``unit_to_scale`` — surfacing the latent bug
as ``TypeError: exprel requires a dimensionless "x" ...`` across every model that
integrates with units.

:func:`exp_euler_step` below is a faithful copy of
``brainstate.nn.exp_euler_step`` with the single unit-reconstruction line
corrected. :func:`install_exp_euler_patch` overwrites the buggy upstream symbol
at ``brainpy_state`` import time, but **only when the installed brainstate is
actually affected** (probed at runtime). This means:

- model code, user docstring examples, and tests that call the public
  ``brainstate.nn.exp_euler_step`` all work without edits;
- once a fixed ``brainstate`` is installed, the probe passes and the patch
  becomes a no-op automatically.

TODO(remove): delete this module and its call from ``brainpy_state/__init__.py``
once a fixed ``brainstate`` release is published and pinned in
``pyproject.toml``. See ``dev/superpowers/exprel-exp-euler-fix.md``.
"""

from __future__ import annotations

from typing import Callable, Union

import jax
import jax.numpy as jnp
import brainunit as u
from brainstate import environ, random
from brainstate.transform import vector_grad

__all__ = [
    'exp_euler_step',
    'install_exp_euler_patch',
]


def exp_euler_step(
    fn: Callable, *args, **kwargs
) -> Union[jax.Array, u.Quantity]:
    r"""
    One-step exponential Euler method for solving ODEs and SDEs.

    This is a drop-in replacement for :func:`brainstate.nn.exp_euler_step` that
    corrects the Jacobian unit reconstruction (see module docstring). It updates
    the state via

    .. math::
        x_{n+1} = x_n + dt \cdot \varphi(dt \cdot J) \cdot f(x_n, t_n),
        \qquad \varphi(z) = (e^z - 1) / z,

    where :math:`J = \partial f / \partial x` and ``dt`` is read from
    ``brainstate.environ``.

    Parameters
    ----------
    fn : Callable
        The drift function :math:`f(x, t)`. Takes the state variable as the first
        argument and returns the derivative :math:`dx/dt`.
    *args
        Variable arguments. If the first argument is callable it is treated as the
        diffusion function for SDE integration; otherwise the first argument is
        the state variable :math:`x`.
    **kwargs
        Additional keyword arguments forwarded to the drift (and diffusion)
        function.

    Returns
    -------
    x_next : jax.Array or brainunit.Quantity
        The state after one integration step of size ``dt``.

    Raises
    ------
    ValueError
        If the state dtype is not float16, bfloat16, float32, or float64, or if
        the drift and diffusion terms have incompatible units.

    Notes
    -----
    If the state :math:`x` has unit :math:`[X]`, the drift :math:`f` must have
    unit :math:`[X]/[T]` and the diffusion :math:`g` unit :math:`[X]/\sqrt{[T]}`.
    """
    # Validate inputs
    assert callable(fn), 'The drift function should be callable.'
    assert len(args) > 0, 'The input arguments should not be empty.'

    # Parse arguments: check if first arg is diffusion function
    diffusion = None
    if callable(args[0]):
        diffusion = args[0]
        args = args[1:]
        assert len(args) > 0, 'State variable is required after diffusion function.'

    # Validate state variable dtype
    state = u.math.asarray(args[0])
    dtype = u.math.get_dtype(state)
    if dtype not in [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]:
        raise ValueError(
            f'State variable dtype must be float16, bfloat16, float32, or float64 '
            f'for Exponential Euler method, but got {dtype}.'
        )

    # Get time step from environment
    dt = environ.get_dt()

    # Compute drift term with Jacobian
    # vector_grad returns (Jacobian, function_value)
    jacobian, drift_value = vector_grad(fn, argnums=0, return_value=True)(*args, **kwargs)

    # Convert Jacobian to proper units: [derivative_unit / state_unit] = [1/T].
    # NOTE: divide by the *state* unit, not the Jacobian unit. ``vector_grad``
    # strips the Jacobian unit to dimensionless, so dividing by ``unit(jacobian)``
    # (as upstream brainstate does) mislabels the result as ``unit(drift)`` and
    # breaks the strict ``exprel`` in brainunit>=0.4.0.
    jacobian_with_unit = u.Quantity(
        u.get_mantissa(jacobian),
        u.get_unit(drift_value) / u.get_unit(state)
    )

    # Compute phi function: phi(z) = (exp(z) - 1) / z
    # This is the exponential-related function for stability
    phi = u.math.exprel(dt * jacobian_with_unit)

    # Update state using exponential Euler scheme
    x_next = state + dt * phi * drift_value

    # Add diffusion term for SDE if provided
    if diffusion is not None:
        # Compute diffusion coefficient
        diffusion_coef = diffusion(*args, **kwargs)

        # Generate random noise and scale by sqrt(dt)
        noise = random.randn_like(state)
        diffusion_term = diffusion_coef * u.math.sqrt(dt) * noise

        # Validate unit compatibility between drift and diffusion
        if u.get_dim(x_next) != u.get_dim(diffusion_term):
            drift_unit = u.get_unit(x_next)
            time_unit = u.get_unit(dt)
            expected_diffusion_unit = drift_unit / time_unit ** 0.5
            actual_diffusion_unit = u.get_unit(diffusion_term)
            raise ValueError(
                f"Unit mismatch between drift and diffusion terms. "
                f"State has unit {u.get_unit(state)}, "
                f"drift produces unit {drift_unit}, "
                f"expected diffusion unit {expected_diffusion_unit}, "
                f"but got {actual_diffusion_unit}."
            )

        x_next = x_next + diffusion_term

    return x_next


# Sentinel set once the upstream symbol has been patched, so the install is
# idempotent across repeated imports.
_PATCH_APPLIED = False


def _upstream_exp_euler_is_broken() -> bool:
    """Probe whether the installed ``brainstate.nn.exp_euler_step`` rejects a
    dimensional ODE (the regression). Returns ``True`` only for that specific
    failure, so a fixed brainstate — or an older lenient ``brainunit`` where the
    bug never triggers — is left untouched."""
    import brainstate

    try:
        with environ.context(dt=0.1 * u.ms):
            brainstate.nn.exp_euler_step(
                lambda v: -v / (10.0 * u.ms),
                jnp.ones(1) * u.mV,
            )
        return False
    except TypeError:
        # The strict-``exprel`` rejection of a dimensional argument.
        return True
    except Exception:
        # Any other failure is unrelated to this regression; do not patch.
        return False


def install_exp_euler_patch() -> bool:
    """Install the corrected :func:`exp_euler_step` over the buggy
    ``brainstate.nn.exp_euler_step`` when (and only when) the installed
    brainstate is affected.

    Returns
    -------
    bool
        ``True`` if the patch was applied, ``False`` if it was unnecessary
        (already fixed upstream, lenient ``brainunit``, or already patched).
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    import brainstate

    if not _upstream_exp_euler_is_broken():
        return False

    # Overwrite both the public re-export and the defining module attribute so
    # every reference path resolves to the corrected implementation.
    brainstate.nn.exp_euler_step = exp_euler_step
    _exp_euler_mod = getattr(brainstate.nn, '_exp_euler', None)
    if _exp_euler_mod is not None:
        _exp_euler_mod.exp_euler_step = exp_euler_step

    _PATCH_APPLIED = True
    return True
