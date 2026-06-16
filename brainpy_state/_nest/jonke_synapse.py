# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-faithful ``jonke_synapse`` — exponential-weight-dependence STDP spec + rule.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._network._event_plastic.EventPlasticProj`. The Jonke et al.
(2015) rule generalises pair STDP with an **exponential weight factor**
:math:`\Phi_\pm(w) = \exp(\mu_\pm w)` on each side plus a constant **offset**
:math:`\beta` (a heterosynaptic / activity-independent bias).
"""
from __future__ import annotations

import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from ._plastic_base import (
    frozen, to_ms, to_scalar_float, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['jonke_synapse']


class jonke_synapse:
    r"""Exponential-weight-dependence STDP synapse spec (NEST ``jonke_synapse``).

    The substrate maintains the per-pre ``K+`` trace (``pre_trace_tau=tau_plus``)
    and the per-post ``K-`` trace (``post_trace_tau=tau_minus``); this kernel gates
    its own writeback — **potentiation on the post spike**, **depression on the pre
    spike** — the online all-to-all scheme equal to NEST's deferred ``send()`` at
    every send (pre-spike) time:

    .. math::

       w \leftarrow \min\!\big(w + \lambda(e^{\mu_+ w} K^+ - \beta),\; W_{\max}\big)
       \quad\text{(post spike)}

       w \leftarrow \max\!\big(w + \lambda(-\alpha e^{\mu_- w} K^- - \beta),\; 0\big)
       \quad\text{(pre spike)}

    The clips are **one-sided per side** (NEST ``facilitate_``/``depress_``):
    facilitation hard-bounds only from *above* at ``Wmax``, depression only from
    *below* at ``0``. With the default :math:`\beta = 0` and :math:`\mu_\pm = 0`
    the rule reduces to additive STDP; :math:`\mu_+ > 0` makes potentiation grow
    with the weight (note: *not* a soft upper bound — the ``Wmax`` clip is). Each
    side excludes the current step's own spike from the opposite trace
    (``K+ = pre_trace - pre_spike``, ``K- = post_trace - post_spike``). When
    :math:`\lambda = 0` the weight is returned unchanged (learning disabled),
    skipping the clip — matching NEST.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge weight (pA; bare numbers are pA). Default ``1.0`` pA.
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
        Depression amplitude scaling :math:`\alpha`. Default ``1.0``.
    mu_plus, mu_minus : float, optional
        Exponential weight-dependence exponents (inverse weight units).
        Default ``0.0`` (additive).
    beta : float, optional
        Constant offset :math:`\beta` applied each update (positive biases toward
        depression). Default ``0.0``.
    Wmax : float, optional
        Upper weight clip in facilitation. Default ``100.0``.
    Kplus : float, optional
        Initial ``K+`` (the substrate seeds traces at 0, the NEST default).
        Default ``0.0``.

    Notes
    -----
    **NEST divergence — ``tau_minus`` location.** In NEST ``tau_minus`` is a
    parameter of the postsynaptic neuron (``ArchivingNode``), not the synapse;
    here it is a synapse-spec attribute driving the substrate's per-post ``K-``
    trace, so STDP runs standalone. See ``CONTEXT.md`` Lessons (cluster 04).

    Online vs deferred: the substrate potentiates eagerly at post-spike steps,
    whereas NEST defers it to the next pre spike; the two coincide at pre-spike
    (send) times, so parity is asserted there.

    References
    ----------
    .. [1] NEST ``models/jonke_synapse.h``; Jonke, Legenstein, Habenschuss & Maass (2015).

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy_state import jonke_synapse
       >>> s = jonke_synapse(weight=10.0, mu_plus=0.1)
       >>> s.is_homogeneous_weight, s.edge_state_init()
       (False, {})
       >>> s.beta, s.mu_plus
       (0.0, 0.1)
       >>> float(u.Quantity(s.post_trace_tau).to_decimal(u.ms))
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
        mu_plus: ArrayLike = 0.0,
        mu_minus: ArrayLike = 0.0,
        beta: ArrayLike = 0.0,
        Wmax: ArrayLike = 100.0,
        Kplus: ArrayLike = 0.0,
    ):
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
        self.beta = to_scalar_float(beta, name='beta')
        self.Wmax = to_scalar_float(Wmax, name='Wmax')
        self.Kplus = to_scalar_float(Kplus, name='Kplus')

        if to_ms(tau_plus, name='tau_plus') <= 0.0:
            raise ValueError("'tau_plus' must be > 0.")
        if to_ms(tau_minus, name='tau_minus') <= 0.0:
            raise ValueError("'tau_minus' must be > 0.")

        # per-side trace seams (substrate allocates + decays these)
        self.pre_trace_tau = self.tau_plus
        self.post_trace_tau = self.tau_minus

    def edge_state_init(self) -> dict:
        return {}

    # -- rule kernel -------------------------------------------------------
    def _facilitate(self, w, kplus):
        if self.lambda_ == 0.0:
            return w                                        # learning disabled (no clip)
        new_w = w + self.lambda_ * (jnp.exp(self.mu_plus * w) * kplus - self.beta)
        return jnp.minimum(new_w, self.Wmax)               # one-sided upper clip

    def _depress(self, w, kminus):
        if self.lambda_ == 0.0:
            return w
        new_w = w + self.lambda_ * (-self.alpha * jnp.exp(self.mu_minus * w) * kminus - self.beta)
        return jnp.maximum(new_w, 0.0)                     # one-sided lower clip

    def update(self, state, ctx):
        w = state['weight']
        # exclude this step's own spike from the opposite-side trace
        kplus = ctx.pre_trace - ctx.pre_spike
        kminus = ctx.post_trace - ctx.post_spike
        # potentiation on post spike, then depression on pre spike (NEST order)
        w = frozen(ctx.post_spike > 0, self._facilitate(w, kplus), w)
        w = frozen(ctx.pre_spike > 0, self._depress(w, kminus), w)
        return {'weight': w}, w
