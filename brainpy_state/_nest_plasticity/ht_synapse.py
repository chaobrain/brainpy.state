# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-faithful ``ht_synapse`` — Hill-Tononi vesicle-pool depression spec + rule.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._nest_network.event_plastic.EventPlasticProj`. The Hill-Tononi
(2005) model is **depression-only and presynaptic**: a normalized vesicle pool
:math:`P \in [0, 1]` recovers exponentially toward ``1`` between spikes and
depletes multiplicatively on each presynaptic spike; the delivered amplitude is
the baseline weight scaled by the *recovered* pool. It is trace-free (no STDP
window, no ``Wmax``), structurally a depression-only sibling of
:class:`~brainpy_state._nest_synapse.tsodyks2_synapse.tsodyks2_synapse`. The previous
imperative implementation lived in this same module (legacy ``NESTSynapse`` base).
"""
from __future__ import annotations
from brainpy_state._nest_base.base import NESTPlasticity

import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from brainpy_state._nest_base.plastic_base import (
    frozen, to_ms, to_scalar_float, to_unit_interval, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['ht_synapse']


class ht_synapse(NESTPlasticity):
    r"""Hill-Tononi vesicle-pool depression synapse spec (NEST ``ht_synapse``).

    Trace-free and presynaptic: the kernel keeps a per-edge vesicle pool ``P`` and
    last-spike time ``t_lastspike`` (no ``pre_trace``/``post_trace`` seam). On each
    presynaptic spike, with the previous spike at :math:`t_\text{last}`, it applies
    the NEST ``send()`` order **recover → emit → deplete → update**:

    .. math::

       P_\text{send} &= 1 - (1 - P)\,\exp\!\big(-(t - t_\text{last}) / \tau_P\big) \\
       w_\text{eff}  &= w \cdot P_\text{send} \\
       P_\text{new}  &= (1 - \delta_P)\,P_\text{send} \\
       t_\text{last} &\leftarrow t

    The stored ``weight`` is **static** — depression lives in ``P``, so the
    *delivered* amplitude ``w_eff = w * P_send`` is the observable (this is what
    NEST's ``weight_recorder`` logs). Each pre-spike's pool/time writeback is
    ``frozen(...)``-gated on ``ctx.pre_spike`` so non-firing edges hold.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge baseline weight (pA; bare numbers are pA). May be positive
        (excitatory) or negative (inhibitory). Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    tau_P : Quantity, optional
        Vesicle-pool recovery constant (> 0); larger is slower recovery / stronger
        depression. Default ``500.0 ms``.
    delta_P : float, optional
        Fractional depletion per spike, in ``[0, 1]`` (``0`` disables depression,
        ``1`` fully depletes). Default ``0.125``.
    P : float, optional
        Initial pool availability, in ``[0, 1]``. Default ``1.0`` (fully available).

    Notes
    -----
    ``t_lastspike`` initialises to ``0.0`` ms (the NEST default), **not** the
    ``-1.0`` first-spike-skip sentinel used by ``tsodyks2_synapse``. With the
    default ``P = 1`` the first spike recovers as a no-op (``1 - P = 0``), but for
    a *partial* initial ``P`` the first spike correctly recovers from ``t = 0`` —
    matching NEST exactly. Unlike the STDP-window models, ``ht_synapse`` maintains
    no ``K-`` trace, so there is no ``tau_minus`` post-neuron divergence to note.

    References
    ----------
    .. [1] NEST ``models/ht_connection.h``; Hill & Tononi (2005),
       J. Neurophysiol. 93(3):1671-1698.

    Examples
    --------
    .. code-block:: python

       >>> from brainpy_state import ht_synapse
       >>> s = ht_synapse()
       >>> s.is_homogeneous_weight, s.stochastic
       (False, False)
       >>> s.edge_state_init()
       {'P': 1.0, 't_lastspike': 0.0}
       >>> s.tau_P, s.delta_P
       (500.0, 0.125)
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
        tau_P: ArrayLike = 500.0 * u.ms,
        delta_P: ArrayLike = 0.125,
        P: ArrayLike = 1.0,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.tau_P = to_ms(tau_P, name='tau_P')
        self.delta_P = to_unit_interval(delta_P, name='delta_P')
        self._P0 = to_unit_interval(P, name='P')

        if self.tau_P <= 0.0:
            raise ValueError("'tau_P' must be > 0.")

    def edge_state_init(self) -> dict:
        # NEST inits t_lastspike_ = 0.0 (recover from t=0 on the first spike).
        return {'P': self._P0, 't_lastspike': 0.0}

    def update(self, state, ctx):
        P, t_last = state['P'], state['t_lastspike']
        h = ctx.t_now - t_last
        P_send = 1.0 - (1.0 - P) * jnp.exp(-h / self.tau_P)   # recover
        w_eff = state['weight'] * P_send                      # emit w * P
        P_new = (1.0 - self.delta_P) * P_send                 # deplete

        fired = ctx.pre_spike > 0
        new_state = {
            'weight': state['weight'],
            'P': frozen(fired, P_new, P),
            't_lastspike': frozen(fired, ctx.t_now, t_last),
        }
        return new_state, w_eff
