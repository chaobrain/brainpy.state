# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Fixed-substep integrators with an in-substep event hook (decision I1).

Integrators advance a pytree ``y`` of plain float arrays (mantissas) over one
simulation step ``dt`` split into ``substeps`` equal substeps of length
``h = dt / substeps``. The substeps run as **one** ``lax.scan`` of static length
``substeps - 1`` followed by the peeled last substep (architecture doc §7.3: no
Python loop whose trip count depends on the model inside the traced step; no
``while_loop``), so the body is traced once whatever the substep count, the step
is reverse-mode differentiable, has a static cost, and keeps braintrace's graph
analysis intact. Inputs (synaptic/port values) are whatever the derivative
function closes over: the caller computes them **once** per step, outside the
loop, so they are constant across substeps.

After every substep an optional ``hook(k, t, y, aux, final) -> (y, aux)`` runs;
the generated ``NeuronModel.step`` uses it to check the spike condition, apply the
jump and the refractory clamp, and count spikes inside the step. ``final`` is a
static Python bool, ``True`` only for the last substep, so per-step work (a spike
check once per step) is expressed without branching on the traced index ``k``.
Everything the hook changes is carried explicitly in ``(y, aux)``; it must not
mutate Python objects (the body is traced once and reused for every substep).

The vector field has the signature ``f(t, y, aux) -> dy/dt`` (same pytree
structure as ``y``; ``t`` and ``h`` in one consistent time unit, ms in the
kernel). ``aux`` is carried through the hook and is constant within a substep.

Available integrators
---------------------
=================  =====  ===============================================
integrator         order  notes
=================  =====  ===============================================
:class:`Exact`     --     marker: the model overrides ``step`` with exact
                          propagators (linear models only)
:class:`ExpEuler`  1      exponential Euler on the diagonal linearisation;
                          exact for elementwise-linear ODEs
:class:`RK`        1,2,4  explicit Runge-Kutta: Euler, Heun, classic RK4
=================  =====  ===============================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from brainspike._kernel.propagators import phi1

__all__ = [
    'Integrator',
    'Exact',
    'ExpEuler',
    'RK',
    'integrate',
]

VectorField = Callable[[Any, Any, Any], Any]
Hook = Callable[[Any, Any, Any, Any, bool], tuple[Any, Any]]


def _check_substeps(substeps: Any) -> int:
    if isinstance(substeps, bool) or int(substeps) != substeps or substeps < 1:
        raise ValueError(f'substeps must be an int >= 1, got {substeps!r}')
    return int(substeps)


def _axpy(y: Any, h: Any, k: Any) -> Any:
    """``y + h * k`` over matching pytrees."""
    return jax.tree.map(lambda a, b: a + h * b, y, k)


class Integrator:
    """Base class of fixed-substep integrators.

    Attributes
    ----------
    order : int or None
        Global order of accuracy (``None`` for :class:`Exact`).
    substeps : int
        Number of equal substeps per simulation step (static).
    """

    order: int | None
    substeps: int

    #: Whether models using this integrator must hand-write ``step``.
    exact: bool = False

    def substep(self, f: VectorField, t: Any, y: Any, h: Any, aux: Any = None) -> Any:
        """Advance ``y`` from ``t`` to ``t + h``.

        Parameters
        ----------
        f : callable
            Vector field ``f(t, y, aux) -> dy/dt``.
        t : array_like
            Time at the start of the substep.
        y : pytree of jax.Array
            State (float mantissas).
        h : array_like
            Substep length (may be traced).
        aux : Any
            Constant auxiliary data passed to ``f``.

        Returns
        -------
        pytree
            State at ``t + h``.
        """
        raise NotImplementedError  # pragma: no cover


@dataclass(frozen=True)
class Exact(Integrator):
    """Marker for models integrated with hand-written exact propagators.

    A :class:`~brainspike._kernel.model.NeuronModel` declaring ``Exact()`` must
    override ``step``; it is a definition error otherwise. Only linear
    subthreshold dynamics admit exact propagators.
    """

    order = None
    substeps: int = 1
    exact = True

    def substep(self, f, t, y, h, aux=None):
        raise TypeError('Exact() has no generic substep; models declaring Exact() '
                        'must override NeuronModel.step with their propagators')


@dataclass(frozen=True, init=False)
class ExpEuler(Integrator):
    r"""Exponential Euler on the diagonal linearisation (order 1).

    For :math:`\dot y_i = f_i(y)` with :math:`A_i = \partial f_i/\partial y_i`
    (elementwise, evaluated at the start of the substep):

    .. math::

       y_i \leftarrow y_i + h\,\varphi_1(A_i h)\, f_i(y),\qquad
       \varphi_1(z) = (e^z - 1)/z .

    Order 1 in general (order 2 for autonomous scalar ODEs, where it is the
    exponential Rosenbrock-Euler method). Exact when each :math:`f_i` is affine
    in :math:`y_i` with the other
    components held fixed over the substep (e.g. a leaky membrane with constant
    input). :math:`A_i` is computed with one forward-mode JVP per state leaf,
    assuming neurons are independent (the Jacobian of each leaf w.r.t. itself is
    diagonal), which holds for neuron models.

    Parameters
    ----------
    substeps : int, default 1
        Number of substeps per simulation step.
    """

    order = 1
    substeps: int

    def __init__(self, substeps: int = 1) -> None:
        object.__setattr__(self, 'substeps', _check_substeps(substeps))

    def substep(self, f, t, y, h, aux=None):
        fy, lin = jax.linearize(lambda yy: f(t, yy, aux), y)
        leaves, treedef = jax.tree.flatten(y)
        fy_leaves = jax.tree.leaves(fy)
        new = []
        for i, (yi, fi) in enumerate(zip(leaves, fy_leaves)):
            tangent = [jnp.ones_like(l) if j == i else jnp.zeros_like(l) for j, l in enumerate(leaves)]
            a_i = jax.tree.leaves(lin(jax.tree.unflatten(treedef, tangent)))[i]
            new.append(yi + h * phi1(a_i * h).astype(yi.dtype) * fi)
        return jax.tree.unflatten(treedef, new)


_RK_ORDERS = (1, 2, 4)


@dataclass(frozen=True, init=False)
class RK(Integrator):
    """Explicit Runge-Kutta with a fixed number of substeps.

    Parameters
    ----------
    order : {1, 2, 4}, default 4
        1 = forward Euler, 2 = Heun (explicit trapezoid), 4 = classic RK4.
    substeps : int, default 1
        Number of substeps per simulation step (static; run as a ``lax.scan``).

    Raises
    ------
    ValueError
        If ``order`` or ``substeps`` is invalid.
    """

    order: int
    substeps: int

    def __init__(self, order: int = 4, substeps: int = 1) -> None:
        if order not in _RK_ORDERS:
            raise ValueError(f'RK order must be one of {_RK_ORDERS}, got {order!r}')
        object.__setattr__(self, 'order', int(order))
        object.__setattr__(self, 'substeps', _check_substeps(substeps))

    def substep(self, f, t, y, h, aux=None):
        if self.order == 1:
            return _axpy(y, h, f(t, y, aux))
        if self.order == 2:
            k1 = f(t, y, aux)
            k2 = f(t + h, _axpy(y, h, k1), aux)
            return jax.tree.map(lambda a, b, c: a + 0.5 * h * (b + c), y, k1, k2)
        k1 = f(t, y, aux)
        k2 = f(t + 0.5 * h, _axpy(y, 0.5 * h, k1), aux)
        k3 = f(t + 0.5 * h, _axpy(y, 0.5 * h, k2), aux)
        k4 = f(t + h, _axpy(y, h, k3), aux)
        return jax.tree.map(lambda a, b, c, d, e: a + h / 6.0 * (b + 2.0 * c + 2.0 * d + e),
                            y, k1, k2, k3, k4)


def _like(new: Any, old: Any) -> Any:
    """Cast every leaf of ``new`` to the dtype of the matching leaf of ``old``.

    Keeps the ``scan`` carry type-stable when a substep or the hook promotes a
    leaf (e.g. a Python-scalar ``aux`` or a jump to a parameter of another dtype).
    """
    return jax.tree.map(lambda a, b: jnp.asarray(a).astype(jnp.asarray(b).dtype), new, old)


def integrate(integrator: Integrator, f: VectorField, y: Any, t0: Any, dt: Any, *,
              hook: Hook | None = None, aux: Any = None) -> tuple[Any, Any]:
    """Advance ``y`` over one step ``[t0, t0 + dt]`` with fixed substeps.

    Substeps ``0 .. substeps - 2`` run as one ``lax.scan`` (traced once); the last
    substep is peeled so that ``hook`` sees a static ``final=True`` there. With
    ``substeps == 1`` there is no scan.

    Parameters
    ----------
    integrator : Integrator
        :class:`ExpEuler` or :class:`RK` (``Exact`` raises ``TypeError``).
    f : callable
        Vector field ``f(t, y, aux) -> dy/dt`` with the pytree structure of ``y``.
    y : pytree of jax.Array
        Float state mantissas at ``t0``.
    t0, dt : array_like
        Step start and length (same time unit as ``f`` expects); may be traced.
    hook : callable, optional
        ``hook(k, t, y, aux, final) -> (y, aux)`` called after substep ``k``
        (``0 <= k < substeps``; a traced int32 inside the scan, a Python int for the
        last substep) with ``t`` the time at the end of the substep and ``final`` a
        Python bool, ``True`` only after the last substep. Used for in-substep spike
        detection, jumps and refractory handling. It must return ``y`` and ``aux``
        with the input structure; leaves are cast back to the input dtypes.
    aux : Any, optional
        Carried auxiliary data (e.g. spike counts, refractory counters); passed to
        ``f`` and threaded through ``hook``. Python scalars become arrays.

    Returns
    -------
    y : pytree
        State at ``t0 + dt``.
    aux : Any
        Final auxiliary data.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainspike._kernel.integrators import RK, integrate
        >>> f = lambda t, y, aux: {'v': -y['v']}
        >>> y, _ = integrate(RK(4, 8), f, {'v': jnp.array(1.0)}, 0.0, 1.0)
        >>> round(float(y['v']), 6)  # exp(-1)
        0.367879
    """
    n = integrator.substeps
    h = dt / n
    aux = jax.tree.map(jnp.asarray, aux)

    def substep(k, y, aux, final):
        t = t0 + k * h
        y_new = integrator.substep(f, t, y, h, aux)
        aux_new = aux
        if hook is not None:
            y_new, aux_new = hook(k, t + h, y_new, aux, final)
        return y_new, aux_new

    if n > 1:
        def body(carry, k):
            y_c, aux_c = carry
            return _like(substep(k, y_c, aux_c, False), carry), None

        (y, aux), _ = jax.lax.scan(body, (y, aux), jnp.arange(n - 1, dtype=jnp.int32))
    return substep(n - 1, y, aux, True)
