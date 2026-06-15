# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-faithful ``stdp_pl_synapse_hom`` — power-law STDP spec + pure rule kernel.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._network._event_plastic.EventPlasticProj`. Power-law STDP
(Morrison et al. 2007): potentiation is **multiplicative and sub-linear** in the
weight (:math:`w^\mu`, :math:`\mu < 1`), depression is **linear** in the weight,
and there is **no upper bound** ``Wmax`` — the sub-linear potentiation provides
the soft bound, while a hard lower clip keeps the weight non-negative. As for the
other ``*_hom`` models the parameters are NEST *common* properties; here they are
rule-level. The previous imperative implementation lives in
:mod:`brainpy_state._nest._legacy_imperative` (shared base).
"""
from __future__ import annotations

import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from ._plastic_base import (
    frozen, to_ms, to_scalar_float, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['stdp_pl_synapse_hom']


class stdp_pl_synapse_hom:
    r"""Power-law spike-timing-dependent plasticity synapse spec (NEST ``stdp_pl_synapse_hom``).

    The substrate maintains the per-pre ``K+`` trace (``pre_trace_tau=tau_plus``)
    and the per-post ``K-`` trace (``post_trace_tau=tau_minus``); this kernel gates
    its own writeback — **potentiation on the post spike**, **depression on the pre
    spike** — using the online all-to-all scheme equal to NEST's deferred
    ``send()`` at every send (pre-spike) time:

    .. math::

       w \leftarrow w + \lambda \, w^{\mu} \, K^+ \quad\text{(post spike)}

       w \leftarrow \max\!\big(w - \alpha \lambda \, w \, K^-,\; 0\big)
       \quad\text{(pre spike)}

    There is no ``Wmax``: potentiation's :math:`w^{\mu}` weight dependence
    (:math:`\mu \approx 0.4`) is the only upper soft-bound, and the only hard clip
    is the lower clip to ``0`` in depression (matching NEST ``depress_``). Each
    side excludes the current step's own spike from the opposite trace
    (``K+ = pre_trace - pre_spike``, ``K- = post_trace - post_spike``).

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
        Learning rate :math:`\lambda` (>= 0). Default ``0.1``.
    alpha : float, optional
        Depression scaling :math:`\alpha` (>= 0). Default ``1.0``.
    mu : float, optional
        Power-law potentiation exponent :math:`\mu`. Default ``0.4``.
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

    **Parity note.** The consolidated NEST vs. brainpy.state divergence reference
    — trace-storage move, the family parameter-location map, and the parity-test
    links — is in :doc:`/nest-guide/stdp-divergences` (:ref:`stdp-tau-minus`).

    References
    ----------
    .. [1] NEST ``models/stdp_pl_synapse_hom.h``; Morrison, Aertsen & Diesmann (2007).

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy_state import stdp_pl_synapse_hom
       >>> s = stdp_pl_synapse_hom(weight=5.0, mu=0.4, lambda_=0.1)
       >>> s.is_homogeneous_weight, s.edge_state_init()
       (False, {})
       >>> s.mu, s.alpha
       (0.4, 1.0)
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
        lambda_: ArrayLike = 0.1,
        alpha: ArrayLike = 1.0,
        mu: ArrayLike = 0.4,
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
        self.mu = to_scalar_float(mu, name='mu')
        self.Kplus = to_scalar_float(Kplus, name='Kplus')

        if to_ms(tau_plus, name='tau_plus') <= 0.0:
            raise ValueError("'tau_plus' must be > 0.")
        if to_ms(tau_minus, name='tau_minus') <= 0.0:
            raise ValueError("'tau_minus' must be > 0.")
        if self.lambda_ < 0.0:
            raise ValueError("'lambda' must be non-negative.")
        if self.alpha < 0.0:
            raise ValueError("'alpha' must be non-negative.")

        # per-side trace seams (substrate allocates + decays these)
        self.pre_trace_tau = self.tau_plus
        self.post_trace_tau = self.tau_minus

    def edge_state_init(self) -> dict:
        return {}

    # -- rule kernel -------------------------------------------------------
    def _facilitate(self, w, kplus):
        # multiplicative, sub-linear (w^mu); no Wmax upper clip.
        return w + self.lambda_ * jnp.power(w, self.mu) * kplus

    def _depress(self, w, kminus):
        # linear in w; hard lower clip to 0 (NEST depress_).
        new_w = w - self.alpha * self.lambda_ * w * kminus
        return jnp.where(new_w > 0.0, new_w, 0.0)

    def update(self, state, ctx):
        w = state['weight']
        # exclude this step's own spike from the opposite-side trace
        kplus = ctx.pre_trace - ctx.pre_spike
        kminus = ctx.post_trace - ctx.post_spike
        # potentiation on post spike, then depression on pre spike (NEST order)
        w = frozen(ctx.post_spike > 0, self._facilitate(w, kplus), w)
        w = frozen(ctx.pre_spike > 0, self._depress(w, kminus), w)
        return {'weight': w}, w
