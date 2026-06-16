# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``tsodyks2_synapse`` — multiplicative STP spec + pure rule.

Rebuilt as a frozen spec + pure ``update(state, ctx)`` rule kernel on
:class:`~brainpy_state._nest_network.event_plastic.EventPlasticProj`. Per-edge
``weight`` and per-edge state ``u`` (utilization), ``x`` (available resources),
``t_lastspike``; shared ``U``, ``tau_rec``, ``tau_fac``. The delivered amplitude
is ``w_eff = x*u * weight`` with ``x`` updated **before** ``u`` (using the old
``u``), exactly as in NEST ``models/tsodyks2_synapse.h``.
"""
from __future__ import annotations
from brainpy_state._nest_base.base import NESTSynapse

import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from brainpy_state._nest_base.plastic_base import (
    frozen, to_ms, to_scalar_float, to_unit_interval, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['tsodyks2_synapse']

_UNSET = object()


class tsodyks2_synapse(NESTSynapse):
    r"""Tsodyks (2-variable) short-term-plasticity synapse spec.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge weight (pA; bare numbers interpreted as pA). Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    U : float, optional
        Baseline utilization, in ``[0, 1]``. Default ``0.5``.
    u : float, optional
        Initial utilization, in ``[0, 1]``. Defaults to ``U``. Setting ``U`` does
        not implicitly change ``u``.
    x : float, optional
        Initial available-resource fraction (not range-checked). Default ``1.0``.
    tau_rec : Quantity, optional
        Recovery constant (> 0). Default ``800.0 ms``.
    tau_fac : Quantity, optional
        Facilitation constant (>= 0; exactly ``0`` resets ``u`` to ``U`` on every
        non-first spike). Default ``0.0 ms``.

    Notes
    -----
    ``t_lastspike`` initialises to ``-1.0``; the first spike skips the decay
    (delivering ``x_init*u_init*weight``). ``tau_fac == 0`` is matched by exact
    floating-point equality (matching NEST), unlike ``quantal_stp_synapse``'s
    ``< 1e-10`` threshold.

    References
    ----------
    .. [1] NEST ``models/tsodyks2_synapse.h``; Fuhrmann et al. (2002).

    Examples
    --------
    .. code-block:: python

       >>> from brainpy.state import tsodyks2_synapse
       >>> s = tsodyks2_synapse(U=0.5)
       >>> s.edge_state_init()['t_lastspike']
       -1.0
       >>> s.edge_state_init()['u']
       0.5
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False
    pre_trace_tau = None
    post_trace_tau = None

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        U: ArrayLike = 0.5,
        u: ArrayLike = _UNSET,
        x: ArrayLike = 1.0,
        tau_rec: ArrayLike = 800.0 * u.ms,
        tau_fac: ArrayLike = 0.0 * u.ms,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.U = to_unit_interval(U, name='U')
        u0 = self.U if u is _UNSET else to_unit_interval(u, name='u')
        x0 = to_scalar_float(x, name='x')
        self.tau_rec = to_ms(tau_rec, name='tau_rec')
        self.tau_fac = to_ms(tau_fac, name='tau_fac')

        if self.tau_rec <= 0.0:
            raise ValueError("'tau_rec' must be > 0.")
        if self.tau_fac < 0.0:
            raise ValueError("'tau_fac' must be >= 0.")

        self._x0, self._u0 = float(x0), float(u0)

    def edge_state_init(self) -> dict:
        return {'u': self._u0, 'x': self._x0, 't_lastspike': -1.0}

    def update(self, state, ctx):
        tau_rec, tau_fac, U = self.tau_rec, self.tau_fac, self.U
        x, u_, t_last = state['x'], state['u'], state['t_lastspike']
        h = ctx.t_now - t_last
        decay = t_last >= 0.0
        x_decay = jnp.exp(-h / tau_rec)
        u_decay = 0.0 if tau_fac == 0.0 else jnp.exp(-h / tau_fac)

        x_new = jnp.where(decay, 1.0 + (x - x * u_ - 1.0) * x_decay, x)   # uses OLD u
        u_new = jnp.where(decay, U + u_ * (1.0 - U) * u_decay, u_)
        w_eff = x_new * u_new * state['weight']

        fired = ctx.pre_spike > 0
        new_state = {
            'weight': state['weight'],
            'x': frozen(fired, x_new, state['x']),
            'u': frozen(fired, u_new, state['u']),
            't_lastspike': frozen(fired, ctx.t_now, state['t_lastspike']),
        }
        return new_state, w_eff
