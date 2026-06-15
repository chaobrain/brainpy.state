# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``tsodyks_synapse_hom`` — homogeneous Tsodyks STP spec + rule.

Rebuilt as a frozen parameter spec plus a pure ``update(state, ctx)`` rule
kernel that runs on :class:`~brainpy_state._network._event_plastic.EventPlasticProj`.
The ``_hom`` variant shares ``weight``, ``U`` and the time constants across all
connections (NEST common properties) and keeps ``x, y, u, t_lastspike``
per-edge. It uses NEST's *plain-exp* propagator form (``Pzz = exp(-h/tau_rec)``,
``x += Pxy*y + Pxz*z``) — algebraically identical to the non-homogeneous
``tsodyks_synapse`` (which uses ``expm1`` / ``- Pzz*z``) but floating-point
distinct, so each is kept exactly as NEST has it.
"""
from __future__ import annotations

import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from ._plastic_base import (
    frozen, to_ms, to_scalar_float, unit_of, validate_delay,
    validate_receptor_type, weight_to_pa,
)

__all__ = ['tsodyks_synapse_hom']


class tsodyks_synapse_hom:
    r"""Homogeneous Tsodyks (2000) short-term-plasticity synapse spec.

    Shares ``weight``, ``U``, ``tau_psc``, ``tau_fac`` and ``tau_rec`` across all
    edges; ``x`` (recovered), ``y`` (active) and ``u`` (utilization) are per-edge
    state propagated over the inter-spike interval ``h = t_now - t_lastspike`` and
    updated on each presynaptic spike. The effective amplitude delivered for a
    spike is ``w_eff = u*x * weight``.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Shared synaptic weight (pA; bare numbers are interpreted as pA). Default
        ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    tau_psc : Quantity, optional
        Synaptic-current decay time constant (> 0). Default ``3.0 ms``.
    tau_fac : Quantity, optional
        Facilitation time constant (>= 0; ``0`` disables facilitation). Default
        ``0.0 ms``.
    tau_rec : Quantity, optional
        Recovery time constant (> 0). Default ``800.0 ms``.
    U : float, optional
        Baseline utilization increment, in ``[0, 1]``. Default ``0.5``.
    x, y, u : float, optional
        Initial recovered / active / utilization fractions. ``x + y <= 1`` is
        enforced; ``u`` is **not** range-checked (preserving the NEST ``_hom``
        asymmetry). Defaults ``1.0``, ``0.0``, ``0.0``.

    Notes
    -----
    NEST ``_hom`` validation messages drop the quotes around parameter names
    (``"tau_psc must be > 0."``, ``"U must be in [0,1]."``). The propagator
    ``Pxy`` has a ``tau_psc - tau_rec`` denominator; NEST adds no singular
    guard, so neither does this spec — avoid ``tau_psc ≈ tau_rec``.

    References
    ----------
    .. [1] Tsodyks M, Uziel A, Markram H (2000). Synchrony generation in
           recurrent networks with frequency-dependent synapses. J. Neurosci.
           20(RC50):1-5.
    .. [2] NEST ``models/tsodyks_synapse_hom.h``.

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy_state import tsodyks_synapse_hom
       >>> s = tsodyks_synapse_hom(U=0.5)
       >>> s.is_homogeneous_weight
       True
       >>> sorted(s.edge_state_init())
       ['t_lastspike', 'u', 'x', 'y']
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = True
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
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.tau_psc = to_ms(tau_psc, name='tau_psc')
        self.tau_fac = to_ms(tau_fac, name='tau_fac')
        self.tau_rec = to_ms(tau_rec, name='tau_rec')
        self.U = _unit_interval_no_quotes(U, name='U')

        x0 = to_scalar_float(x, name='x')
        y0 = to_scalar_float(y, name='y')
        u0 = to_scalar_float(u, name='u')   # _hom does NOT range-check u

        if self.tau_psc <= 0.0:
            raise ValueError('tau_psc must be > 0.')
        if self.tau_fac < 0.0:
            raise ValueError('tau_fac must be >= 0.')
        if self.tau_rec <= 0.0:
            raise ValueError('tau_rec must be > 0.')
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
        Pzz = jnp.exp(-h / tau_rec)                       # plain exp (NEST _hom)
        Pxy = ((Pzz - 1.0) * tau_rec - (Pyy - 1.0) * tau_psc) / (tau_psc - tau_rec)
        Pxz = 1.0 - Pzz

        x, y, u_ = state['x'], state['y'], state['u']
        z = 1.0 - x - y
        u_ = u_ * Puu
        x = x + Pxy * y + Pxz * z                          # PLUS Pxz*z
        y = y * Pyy
        u_ = u_ + U * (1.0 - u_)
        delta = u_ * x
        x = x - delta
        y = y + delta
        w_eff = delta * state['weight']                   # shared scalar weight

        fired = ctx.pre_spike > 0
        new_state = {
            'weight': state['weight'],
            'u': frozen(fired, u_, state['u']),
            'x': frozen(fired, x, state['x']),
            'y': frozen(fired, y, state['y']),
            't_lastspike': frozen(fired, ctx.t_now, state['t_lastspike']),
        }
        return new_state, w_eff


def _unit_interval_no_quotes(value, *, name: str) -> float:
    v = to_scalar_float(value, name=name)
    if v < 0.0 or v > 1.0:
        raise ValueError(f'{name} must be in [0,1].')
    return v
