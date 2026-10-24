# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Saturating NMDA receptor kinetics (K0 T8), the owner of the NMDA open fraction.

Kinetics of ``_brainpy/synapse.py`` ``BioNMDA`` with the goal-01 saturation bug
fixed by construction: the states are dimensionless fractions, so ``1 - x`` and
``1 - s`` never depend on a conductance unit; the conductance scale ``g_max`` is the
weight of the projection that delivers ``s`` to a (Mg-gated) ``conductance`` port.

Until the K0 T6 ``SynapseModel`` lands, the kinetics live in the :class:`NMDA`
population (one receptor pool per presynaptic neuron, i.e. pre-aligned). In T6 the
pure :func:`nmda_kinetics` step becomes a pre-aligned, nonlinear ``SynapseModel``
(excluded from the post-merge pass) and :class:`NMDA` a compatibility wrapper. See
``docs/specs/2026-10-22-k0-t8-nmda-gated-conductance.md``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import brainunit as u
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import Exact, Integrator
from brainspike._kernel.model import Capabilities, NeuronModel
from brainspike._kernel.models._common import as_record
from brainspike._kernel.params import Init, Params, Record, States
from brainspike._kernel.ports import Event, Graded, Ports
from brainspike._kernel.propagators import phi1
from brainspike._kernel.units import from_mantissa, grid_steps, to_mantissa

__all__ = ['NMDA', 'nmda_kinetics']

_PER_MS = u.get_unit(1.0 / u.ms)
_PER_MS_MM = u.get_unit(1.0 / (u.ms * u.mM))


def nmda_kinetics(x: Any, s: Any, pulse: Any, arrived: Any, p: Record, dt: Any
                  ) -> tuple[Any, Any, Any]:
    r"""One step of the saturating NMDA kinetics (pure, mantissa level).

    .. math::

       \frac{dx}{dt} = \alpha_2 T(t)(1 - x) - \beta_2 x, \qquad
       \frac{ds}{dt} = \alpha_1 x (1 - s) - \beta_1 s

    An arrival restarts the transmitter pulse (``T`` for ``T_dur``, rounded up to
    whole steps); ``x`` is advanced exactly with ``T`` constant over the step, then
    ``s`` exactly with the new ``x`` constant over the step -- the scheme of
    ``BioNMDA`` (two exponential-Euler steps of linear ODEs).

    Parameters
    ----------
    x, s : jax.Array
        Dimensionless states (fractions in ``[0, 1]``).
    pulse : jax.Array of int32
        Remaining steps of the transmitter pulse.
    arrived : jax.Array of bool
        Presynaptic arrival this step.
    p : Record
        Parameters ``alpha1``, ``beta1``, ``alpha2``, ``beta2``, ``T``, ``T_dur``
        (Quantities).
    dt : brainunit.Quantity
        Step size.

    Returns
    -------
    x, s, pulse : jax.Array
        New values (``pulse`` already decremented for the next step).
    """
    dtype = jnp.result_type(x)
    dt_ms = jnp.asarray(to_mantissa(dt, u.ms), dtype)
    a1 = to_mantissa(p.alpha1, _PER_MS, dtype)
    b1 = to_mantissa(p.beta1, _PER_MS, dtype)
    a2 = to_mantissa(p.alpha2, _PER_MS_MM, dtype)
    b2 = to_mantissa(p.beta2, _PER_MS, dtype)
    T = to_mantissa(p.T, u.mM, dtype)
    n_pulse = grid_steps(to_mantissa(p.T_dur, u.ms, dtype), dt_ms).astype(jnp.int32)
    pulse = jnp.where(arrived, n_pulse, pulse)
    TT = jnp.where(pulse > 0, T, jnp.zeros_like(T))
    fx = a2 * TT * (1 - x) - b2 * x
    x = x + dt_ms * phi1(-(a2 * TT + b2) * dt_ms) * fx
    fs = a1 * x * (1 - s) - b1 * s
    s = s + dt_ms * phi1(-(a1 * x + b1) * dt_ms) * fs
    return x.astype(dtype), s.astype(dtype), jnp.maximum(pulse - 1, 0).astype(jnp.int32)


class NMDA(NeuronModel):
    r"""Pre-aligned NMDA receptor pools with saturating second-order kinetics.

    One pool per presynaptic neuron. Spikes arriving on ``spike`` release a
    transmitter pulse of concentration ``T`` lasting ``T_dur``; the output ``s``
    (open fraction, dimensionless) is delivered to a ``conductance`` port of the
    targets with the weight ``g_max`` (a conductance), typically Mg-gated:

    .. code-block:: text

        pre --(event, one_to_one)--> NMDA --(graded s, weight = g_max [nS])--> post.nmda
        g_nmda,i = sum_j w_ij s_j,   I_i = g_nmda,i B(V_i) (E_nmda - V_i)

    Parameters
    ----------
    n : int
        Number of pools.
    params : mapping, optional
        Overrides of ``alpha1`` (2 /ms), ``beta1`` (0.01 /ms), ``alpha2``
        (1 /(ms mM)), ``beta2`` (0.5 /ms), ``T`` (1 mM), ``T_dur`` (0.5 ms).
    init : mapping, optional
        Initial ``x``, ``s`` (default 0) and ``pulse`` (default 0 steps).
    **kwargs
        ``precision``, ``key``, ``trainable``, ``name`` (see ``NeuronModel``).

    Notes
    -----
    - Input ``spike``: ``event`` with a dimensionless weight; any payload > 0
      is an arrival (the pulse saturates, multiplicity and weight magnitude are
      ignored), restarting the pulse counter.
    - Timing: an arrival in step ``n`` drives ``x`` from the start of step ``n``
      (``BioNMDA`` semantics). With the default one-step ``pre -> NMDA`` delay
      the pulse starts at the presynaptic spike's NEST stamp ``(n + 1) dt``; a
      zero delay reproduces ``BioNMDA``'s same-step onset.
    - Forward-only in K0 (``differentiable=False``): the pulse onset is a hard
      threshold on the payload.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainspike._kernel import StepContext
        >>> from brainspike._kernel.models.nmda import NMDA
        >>> pool = NMDA(1)
        >>> ctx = StepContext.create(0.0 * u.ms, 0.1 * u.ms)
        >>> s = pool.update(ctx, {'spike': 1.0})['s']
        >>> bool(s[0] > 0)
        True
        >>> int(pool.dyn_states['pulse'].value[0])  # 5 pulse steps, 1 used
        4
    """

    params = Params(alpha1=2.0 / u.ms, beta1=0.01 / u.ms, alpha2=1.0 / (u.ms * u.mM),
                    beta2=0.5 / u.ms, T=1.0 * u.mM, T_dur=0.5 * u.ms)
    states = States(x=Init(0.0), s=Init(0.0), pulse=Init(0, dtype=np.int32))
    inputs = Ports(spike=Event())
    outputs = Ports(s=Graded(None))
    integrator: Integrator = Exact()
    capabilities: ClassVar[Capabilities] = Capabilities(differentiable=False, etrace=False)

    def output_states(self) -> dict[str, str]:
        """``{'s': 's'}``: the output is read from the open-fraction state."""
        return {'s': 's'}

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Advance the kinetics one step (:func:`nmda_kinetics`)."""
        inp = as_record(inp)
        x0 = to_mantissa(s.x)
        arrived = to_mantissa(inp.spike) > 0
        x, o, pulse = nmda_kinetics(x0, to_mantissa(s.s), s.pulse, arrived, p, ctx.dt)
        s = s.replace(x=from_mantissa(x), s=from_mantissa(o), pulse=pulse)
        return s, {'s': s.s}
