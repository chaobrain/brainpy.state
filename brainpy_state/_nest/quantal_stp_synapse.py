# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``quantal_stp_synapse`` — probabilistic-release STP spec + rule.

Rebuilt as a frozen spec + a pure, *stochastic* ``update(state, ctx)`` rule
kernel on :class:`~brainpy_state._network._event_plastic.EventPlasticProj`.
Each connection has ``n`` (static) release sites; ``a`` are currently available.
On a spike: facilitate ``u``, stochastically recover depleted sites, then
release ``n_rel ~ Binomial(a, u)`` sites, delivering ``w_eff = n_rel * weight``
and depleting ``a``. The PRNG differs from NEST, so parity is **distributional**
(mean release converges to the ``tsodyks2`` limit).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import saiunit as u
from brainstate.typing import ArrayLike

from ._plastic_base import (
    frozen, to_ms, to_scalar_int, to_unit_interval, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['quantal_stp_synapse']

_UNSET = object()


class quantal_stp_synapse:
    r"""Quantal (binomial) short-term-plasticity synapse spec.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge, per-site weight (pA; the maximum delivered amplitude is
        ``n * weight``). Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    U : float, optional
        Baseline release probability, in ``[0, 1]``. Default ``0.5``.
    u : float, optional
        Initial release probability, in ``[0, 1]``. Defaults to ``U``.
    n : int, optional
        Number of release sites (static). Default ``1``.
    a : int, optional
        Initial number of available sites. Defaults to ``n``.
    tau_rec : Quantity, optional
        Recovery constant (> 0). Default ``800.0 ms``.
    tau_fac : Quantity, optional
        Facilitation constant (>= 0). Default ``0.0 ms``.

    Notes
    -----
    ``t_lastspike`` initialises to ``-1.0`` (first spike skips decay/recovery).
    Facilitation uses a ``tau_fac < 1e-10`` threshold (NOT exact equality, unlike
    ``tsodyks2_synapse``). The delivered amplitude is ``n_rel * weight`` with no
    ``/n`` normalisation, and ``a`` is not hard-clamped (the dynamics keep it in
    ``[0, n]``). Recovery and release use :func:`jax.random.binomial`, so the
    stream differs from NEST — compare distributionally.

    References
    ----------
    .. [1] NEST ``models/quantal_stp_synapse.h``; Loebel et al. (2009).

    Examples
    --------
    .. code-block:: python

       >>> from brainpy_state import quantal_stp_synapse
       >>> s = quantal_stp_synapse()
       >>> s.stochastic
       True
       >>> s.n
       1
       >>> s.edge_state_init()['a']
       1.0
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = True
    pre_trace_tau = None
    post_trace_tau = None

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        U: ArrayLike = 0.5,
        u: ArrayLike = _UNSET,
        n: ArrayLike = 1,
        a: ArrayLike = _UNSET,
        tau_rec: ArrayLike = 800.0 * u.ms,
        tau_fac: ArrayLike = 0.0 * u.ms,
    ):
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.U = to_unit_interval(U, name='U')
        self._u0 = self.U if u is _UNSET else to_unit_interval(u, name='u')
        self.n = to_scalar_int(n, name='n')
        self._a0 = self.n if a is _UNSET else to_scalar_int(a, name='a')
        self.tau_rec = to_ms(tau_rec, name='tau_rec')
        self.tau_fac = to_ms(tau_fac, name='tau_fac')

        if self.tau_rec <= 0.0:
            raise ValueError("'tau_rec' must be > 0.")
        if self.tau_fac < 0.0:
            raise ValueError("'tau_fac' must be >= 0.")

    def edge_state_init(self) -> dict:
        return {'u': float(self._u0), 'a': float(self._a0), 't_lastspike': -1.0}

    def update(self, state, ctx):
        tau_rec, tau_fac, U, n = self.tau_rec, self.tau_fac, self.U, self.n
        k_rec, k_rel = jax.random.split(ctx.key)
        u_, a, t_last = state['u'], state['a'], state['t_lastspike']
        h = ctx.t_now - t_last
        decay = t_last >= 0.0
        p_decay = jnp.exp(-h / tau_rec)
        u_decay = 0.0 if tau_fac < 1e-10 else jnp.exp(-h / tau_fac)

        # facilitate (only on non-first spikes)
        u_new = jnp.where(decay, U + u_ * (1.0 - U) * u_decay, u_)
        # stochastic recovery of depleted sites
        n_rec = jax.random.binomial(k_rec, n - a, 1.0 - p_decay)
        a_rec = jnp.where(decay, a + n_rec, a)
        # stochastic release (every spike)
        n_rel = jax.random.binomial(k_rel, a_rec, u_new)
        w_eff = n_rel * state['weight']
        a_new = a_rec - n_rel

        fired = ctx.pre_spike > 0
        new_state = {
            'weight': state['weight'],
            'u': frozen(fired, u_new, state['u']),
            'a': frozen(fired, a_new, state['a']),
            't_lastspike': frozen(fired, ctx.t_now, state['t_lastspike']),
        }
        return new_state, w_eff
