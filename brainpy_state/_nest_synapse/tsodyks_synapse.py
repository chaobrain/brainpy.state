# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``tsodyks_synapse`` — Tsodyks (2000) STP spec + pure rule.

Rebuilt as a frozen parameter spec plus a pure ``update(state, ctx)`` rule
kernel on :class:`~brainpy_state._nest_network._event_plastic.EventPlasticProj`.
Per-edge ``weight`` and per-edge state ``x`` (recovered), ``y`` (active),
``u`` (utilization), ``t_lastspike``; shared ``U`` and time constants. Uses
NEST's ``expm1`` propagator form (``Pzz = expm1(-h/tau_rec)``, ``x -= Pzz*z``);
the homogeneous :class:`~brainpy_state._nest_synapse.tsodyks_synapse_hom.tsodyks_synapse_hom`
uses the algebraically-equal plain-exp form. Each is kept exactly as NEST has it
(they are floating-point distinct).
"""
from __future__ import annotations
from brainpy_state._nest_base._base import NESTSynapse

import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from brainpy_state._nest_base._plastic_base import (
    frozen, to_ms, to_scalar_float, to_unit_interval, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['tsodyks_synapse']


class tsodyks_synapse(NESTSynapse):
    r"""Tsodyks, Uziel & Markram (2000) short-term-plasticity synapse spec.

    On each presynaptic spike the per-edge state is propagated over the
    inter-spike interval ``h = t_now - t_lastspike`` and the released fraction
    ``delta = u*x`` is delivered as ``w_eff = delta * weight``. The update order
    is load-bearing (NEST "don't change the order"): propagate ``u, x, y``,
    then facilitate ``u``, then release.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge synaptic weight (pA; bare numbers interpreted as pA, sign
        preserved). Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    tau_psc : Quantity, optional
        Synaptic-current decay constant (> 0). Default ``3.0 ms``.
    tau_fac : Quantity, optional
        Facilitation constant (>= 0; ``0`` disables facilitation). Default
        ``0.0 ms``.
    tau_rec : Quantity, optional
        Recovery constant (> 0). Default ``800.0 ms``.
    U : float, optional
        Baseline utilization increment, in ``[0, 1]``. Default ``0.5``.
    x, y, u : float, optional
        Initial recovered / active / utilization fractions, each in ``[0, 1]``
        with ``x + y <= 1``. Defaults ``1.0``, ``0.0``, ``0.0``.

    Notes
    -----
    ``t_lastspike`` initialises to ``0.0`` with **no** first-spike guard; with
    the default ``x=1, y=0`` (so ``z=0``) the first spike is interval-invariant,
    matching NEST. The propagator ``Pxy`` divides by ``tau_psc - tau_rec`` with
    no singular guard (as in NEST) — avoid ``tau_psc ≈ tau_rec``.

    References
    ----------
    .. [1] Tsodyks M, Uziel A, Markram H (2000). Synchrony generation in
           recurrent networks with frequency-dependent synapses. J. Neurosci.
           20(RC50):1-5.
    .. [2] NEST ``models/tsodyks_synapse.h``.

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy_state import tsodyks_synapse
       >>> s = tsodyks_synapse(U=0.5)
       >>> s.is_homogeneous_weight
       False
       >>> s.edge_state_init()['t_lastspike']
       0.0
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
        tau_psc: ArrayLike = 3.0 * u.ms,
        tau_fac: ArrayLike = 0.0 * u.ms,
        tau_rec: ArrayLike = 800.0 * u.ms,
        U: ArrayLike = 0.5,
        x: ArrayLike = 1.0,
        y: ArrayLike = 0.0,
        u: ArrayLike = 0.0,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.tau_psc = to_ms(tau_psc, name='tau_psc')
        self.tau_fac = to_ms(tau_fac, name='tau_fac')
        self.tau_rec = to_ms(tau_rec, name='tau_rec')
        self.U = to_unit_interval(U, name='U')

        x0 = to_scalar_float(x, name='x')
        y0 = to_scalar_float(y, name='y')
        u0 = to_unit_interval(u, name='u')

        if self.tau_psc <= 0.0:
            raise ValueError("'tau_psc' must be > 0.")
        if self.tau_fac < 0.0:
            raise ValueError("'tau_fac' must be >= 0.")
        if self.tau_rec <= 0.0:
            raise ValueError("'tau_rec' must be > 0.")
        if x0 + y0 > 1.0:
            raise ValueError('x + y must be <= 1.0.')

        self._x0, self._y0, self._u0 = float(x0), float(y0), float(u0)

    def edge_state_init(self) -> dict:
        return {'u': self._u0, 'x': self._x0, 'y': self._y0, 't_lastspike': 0.0}

    def update(self, state, ctx):
        tau_psc, tau_fac, tau_rec, U = self.tau_psc, self.tau_fac, self.tau_rec, self.U
        h = ctx.t_now - state['t_lastspike']
        Puu = 0.0 if tau_fac == 0.0 else jnp.exp(-h / tau_fac)
        Pyy = jnp.exp(-h / tau_psc)
        Pzz = jnp.expm1(-h / tau_rec)                      # e^(-h/tau_rec) - 1
        Pxy = (Pzz * tau_rec - (Pyy - 1.0) * tau_psc) / (tau_psc - tau_rec)

        x, y, u_ = state['x'], state['y'], state['u']
        z = 1.0 - x - y
        u_ = u_ * Puu
        x = x + Pxy * y - Pzz * z                          # MINUS (expm1 carries -1)
        y = y * Pyy
        u_ = u_ + U * (1.0 - u_)
        delta = u_ * x
        x = x - delta
        y = y + delta
        w_eff = delta * state['weight']

        fired = ctx.pre_spike > 0
        new_state = {
            'weight': state['weight'],
            'u': frozen(fired, u_, state['u']),
            'x': frozen(fired, x, state['x']),
            'y': frozen(fired, y, state['y']),
            't_lastspike': frozen(fired, ctx.t_now, state['t_lastspike']),
        }
        return new_state, w_eff
