# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``stdp_synapse`` — pair-based STDP spec + pure rule kernel.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._nest_network.event_plastic.EventPlasticProj`. The substrate
maintains the per-pre-neuron ``K+`` trace (``pre_trace_tau=tau_plus``) and the
per-post-neuron ``K-`` trace (``post_trace_tau=tau_minus``); the kernel applies
**potentiation on the post spike** (using ``K+``) and **depression on the pre
spike** (using ``K-``), the online all-to-all scheme (Morrison et al. 2008) that
is equal to NEST's deferred ``stdp_synapse::send()`` at every send (pre-spike)
time — where NEST's ``weight_recorder`` samples.
"""
from __future__ import annotations
from brainpy_state._nest_base.base import NESTPlasticity

import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from brainpy_state._nest_base.plastic_base import (
    frozen, to_ms, to_scalar_float, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['stdp_synapse']


def _nest_sign(v: float) -> int:
    # NEST set_status sign check: (x >= 0) - (x < 0); zero counts as positive.
    return int(v >= 0.0) - int(v < 0.0)


class stdp_synapse(NESTPlasticity):
    r"""Pair-based spike-timing-dependent plasticity synapse spec (NEST ``stdp_synapse``).

    On the substrate the per-pre ``K+`` and per-post ``K-`` traces decay-then-add
    each step; this kernel gates its own weight writeback:

    .. math::

       \hat w \leftarrow \hat w + \lambda (1-\hat w)^{\mu_+} K^+ \quad\text{(post spike)}

       \hat w \leftarrow \hat w - \alpha \lambda \hat w^{\mu_-} K^- \quad\text{(pre spike)}

    with :math:`\hat w = w/W_{\max}`, the weight clamped to :math:`[0, W_{\max}]`
    inside each update (matching NEST ``facilitate_``/``depress_``). Each side
    excludes the current step's own spike from the opposite trace
    (``K+ = pre_trace - pre_spike``, ``K- = post_trace - post_spike``), the
    simultaneous-spike convention that mirrors NEST's half-open window +
    strictly-prior ``get_K_value``.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge weight (pA; bare numbers are pA). Same sign as ``Wmax``.
        Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    tau_plus : Quantity, optional
        Presynaptic ``K+`` trace constant (> 0). Default ``20.0 ms``.
    tau_minus : Quantity, optional
        Postsynaptic ``K-`` trace constant (> 0). Default ``20.0 ms``.
    lambda_ : float, optional
        Learning rate :math:`\lambda`. Default ``0.01``.
    alpha : float, optional
        Depression asymmetry :math:`\alpha`. Default ``1.0``.
    mu_plus, mu_minus : float, optional
        Potentiation/depression exponents (``0`` additive, ``1`` multiplicative
        soft-bound). Default ``1.0``.
    Wmax : float, optional
        Weight bound (same sign as ``weight``). Default ``100.0``.
    Kplus : float, optional
        Initial ``K+`` (>= 0). Default ``0.0`` (the substrate seeds traces at 0,
        the NEST default).

    Notes
    -----
    **NEST divergence — ``tau_minus`` location.** In NEST ``tau_minus`` is a
    parameter of the *postsynaptic neuron* (``ArchivingNode``), not the synapse;
    here it is a synapse-spec attribute that drives the substrate's per-post
    ``K-`` trace, so STDP runs standalone. See ``develop/NEST_PARITY_LEDGER.md`` Lessons (cluster 04).

    Online vs deferred: the substrate applies potentiation eagerly at post-spike
    steps, whereas NEST defers it to the next pre spike; the two coincide at
    pre-spike (send) times, so parity is asserted there.

    **Parity note.** The consolidated NEST vs. brainpy.state divergence reference
    — trace-storage move, the family parameter-location map, and the parity-test
    links — is in :doc:`/nest-style/divergences/stdp` (:ref:`stdp-tau-minus`).

    References
    ----------
    .. [1] NEST ``models/stdp_synapse.h``; Morrison et al. (2008); Guetig et al. (2003).

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy.state import stdp_synapse
       >>> s = stdp_synapse(weight=1.0, tau_plus=20.0 * u.ms)
       >>> s.is_homogeneous_weight, s.edge_state_init()
       (False, {})
       >>> float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms))
       20.0
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        lambda_: ArrayLike = 0.01,
        alpha: ArrayLike = 1.0,
        mu_plus: ArrayLike = 1.0,
        mu_minus: ArrayLike = 1.0,
        Wmax: ArrayLike = 100.0,
        Kplus: ArrayLike = 0.0,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.tau_plus = to_ms(tau_plus, name='tau_plus') * u.ms
        self.tau_minus = to_ms(tau_minus, name='tau_minus') * u.ms
        self.lambda_ = to_scalar_float(lambda_, name='lambda')
        self.alpha = to_scalar_float(alpha, name='alpha')
        self.mu_plus = to_scalar_float(mu_plus, name='mu_plus')
        self.mu_minus = to_scalar_float(mu_minus, name='mu_minus')
        self.Wmax = to_scalar_float(Wmax, name='Wmax')
        self.Kplus = to_scalar_float(Kplus, name='Kplus')

        if to_ms(tau_plus, name='tau_plus') <= 0.0:
            raise ValueError("'tau_plus' must be > 0.")
        if to_ms(tau_minus, name='tau_minus') <= 0.0:
            raise ValueError("'tau_minus' must be > 0.")
        w0 = float(u.get_mantissa(self.weight))
        if _nest_sign(w0) != _nest_sign(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')
        if self.Kplus < 0.0:
            raise ValueError('Kplus must be non-negative.')

        # per-side trace seams (substrate allocates + decays these)
        self.pre_trace_tau = self.tau_plus
        self.post_trace_tau = self.tau_minus

    def edge_state_init(self) -> dict:
        return {}

    # -- rule kernel -------------------------------------------------------
    def _facilitate(self, w, kplus):
        norm_w = w / self.Wmax + self.lambda_ * (1.0 - w / self.Wmax) ** self.mu_plus * kplus
        return jnp.where(norm_w < 1.0, norm_w * self.Wmax, self.Wmax)

    def _depress(self, w, kminus):
        norm_w = w / self.Wmax - self.alpha * self.lambda_ * (w / self.Wmax) ** self.mu_minus * kminus
        return jnp.where(norm_w > 0.0, norm_w * self.Wmax, 0.0)

    def update(self, state, ctx):
        w = state['weight']
        # exclude this step's own spike from the opposite-side trace
        kplus = ctx.pre_trace - ctx.pre_spike
        kminus = ctx.post_trace - ctx.post_spike
        # potentiation on post spike, then depression on pre spike (NEST order)
        w = frozen(ctx.post_spike > 0, self._facilitate(w, kplus), w)
        w = frozen(ctx.pre_spike > 0, self._depress(w, kminus), w)
        return {'weight': w}, w
