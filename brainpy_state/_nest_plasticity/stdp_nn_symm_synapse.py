# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-faithful ``stdp_nn_symm_synapse`` — symmetric nearest-neighbour STDP spec + rule.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._nest_network.event_plastic.EventPlasticProj`. The *symmetric*
nearest-neighbour pairing scheme (Morrison, Diesmann & Gerstner 2008, fig. 7A) pairs
each spike only with its nearest partner on the other side: a post spike facilitates
with the nearest preceding pre spike, a pre spike depresses with the nearest preceding
post spike. NEST realises this by having both traces **reset to 1 on their own spike**
rather than accumulate (``stdp_nn_symm_synapse.h:66-71``); here both the per-pre ``K+``
and per-post ``K-`` traces declare the substrate's ``'nearest'`` trace mode, so the
rule kernel itself is **byte-identical to** :class:`stdp_synapse` — the nearest-ness
lives entirely in what the substrate *stores*.
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

__all__ = ['stdp_nn_symm_synapse']


def _nest_sign(v: float) -> int:
    # NEST set_status sign check: (x >= 0) - (x < 0); zero counts as positive.
    return int(v >= 0.0) - int(v < 0.0)


class stdp_nn_symm_synapse(NESTPlasticity):
    r"""Symmetric nearest-neighbour STDP synapse spec (NEST ``stdp_nn_symm_synapse``).

    Both the per-pre ``K+`` and per-post ``K-`` traces run in the substrate's
    ``'nearest'`` mode — each **resets to 1 on its own spike** and decays otherwise
    (``tau_plus`` / ``tau_minus``) — so the value gathered at a partner spike is the
    single nearest preceding pairing, not the all-to-all sum. The kernel is the same
    as :class:`stdp_synapse`:

    .. math::

       \hat w \leftarrow \hat w + \lambda (1-\hat w)^{\mu_+} K^+ \quad\text{(post spike)}

       \hat w \leftarrow \hat w - \alpha \lambda \hat w^{\mu_-} K^- \quad\text{(pre spike)}

    with :math:`\hat w = w/W_{\max}`, the weight clamped to :math:`[0, W_{\max}]` inside
    each update (NEST ``facilitate_``/``depress_``, ``stdp_nn_symm_synapse.h:210-227``).
    Each side excludes the current step's own spike (``K+ = pre_trace - pre_spike``,
    ``K- = post_trace - post_spike``): in the nearest mode this recovers the
    **second-latest** preceding partner when a pair coincides exactly, matching NEST's
    "pairs exactly coinciding ... are discarded; paired with the second latest"
    convention (``stdp_nn_symm_synapse.h:60-64``).

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge weight (pA; bare numbers are pA). Same sign as ``Wmax``. Default ``1.0`` pA.
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

    Notes
    -----
    **NEST divergence — ``tau_minus`` location.** In NEST ``tau_minus`` is a parameter
    of the *postsynaptic neuron* (``ArchivingNode``), not the synapse; here it is a
    synapse-spec attribute driving the substrate's per-post ``K-`` trace, so STDP runs
    standalone. See ``develop/NEST_PARITY_LEDGER.md`` Lessons (cluster 04). The live-NEST parity drive sets
    the post node's ``tau_minus`` to match.

    **No ``Kplus`` parameter.** Unlike :class:`stdp_synapse`, the symmetric scheme has no
    accumulating presynaptic trace to seed — the substrate resets ``K+`` to 1 on each pre
    spike — so ``Kplus`` is intentionally absent from the constructor (it was dropped from
    the public API in the legacy model too).

    The kernel is identical to :class:`stdp_synapse`; the nearest-neighbour behaviour
    lives entirely in the substrate trace mode. The substrate potentiates eagerly at
    post-spike steps, whereas NEST defers it to the next pre spike; the two coincide at
    pre-spike (send) times, so parity is asserted there.

    **Parity note.** The exact nearest-neighbour pairing convention, the NEST
    source citation, and the single-pair regression test are documented in
    :doc:`/nest-guide/stdp-divergences` (:ref:`stdp-nn-symm`).

    References
    ----------
    .. [1] NEST ``models/stdp_nn_symm_synapse.h`` (``send()`` lines 246-297; ``facilitate_``
       at 280, nearest ``Kminus`` depress at 286-287). Morrison, Aertsen & Diesmann (2007)
       Neural Comput. 19:1437-1467; Morrison, Diesmann & Gerstner (2008) Biol. Cybern.
       98:459-478, fig. 7A.

    See Also
    --------
    stdp_synapse, stdp_nn_restr_synapse, stdp_nn_pre_centered_synapse

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy_state import stdp_nn_symm_synapse
       >>> s = stdp_nn_symm_synapse(weight=1.0, tau_plus=20.0 * u.ms)
       >>> s.is_homogeneous_weight, s.edge_state_init()
       (False, {})
       >>> s.pre_trace_mode, s.post_trace_mode
       ('nearest', 'nearest')
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False
    # both traces pair nearest-neighbour: reset-to-1 on spike (substrate seam)
    pre_trace_mode = 'nearest'
    post_trace_mode = 'nearest'

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

        if to_ms(tau_plus, name='tau_plus') <= 0.0:
            raise ValueError("'tau_plus' must be > 0.")
        if to_ms(tau_minus, name='tau_minus') <= 0.0:
            raise ValueError("'tau_minus' must be > 0.")
        w0 = float(u.get_mantissa(self.weight))
        if _nest_sign(w0) != _nest_sign(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')

        # per-side trace seams (substrate allocates + decays these, nearest mode)
        self.pre_trace_tau = self.tau_plus
        self.post_trace_tau = self.tau_minus

    def edge_state_init(self) -> dict:
        return {}

    # -- rule kernel (identical to stdp_synapse; nearest-ness is in the trace) --
    def _facilitate(self, w, kplus):
        norm_w = w / self.Wmax + self.lambda_ * (1.0 - w / self.Wmax) ** self.mu_plus * kplus
        return jnp.where(norm_w < 1.0, norm_w * self.Wmax, self.Wmax)

    def _depress(self, w, kminus):
        norm_w = w / self.Wmax - self.alpha * self.lambda_ * (w / self.Wmax) ** self.mu_minus * kminus
        return jnp.where(norm_w > 0.0, norm_w * self.Wmax, 0.0)

    def update(self, state, ctx):
        w = state['weight']
        # exclude this step's own spike from the opposite-side (nearest) trace
        kplus = ctx.pre_trace - ctx.pre_spike
        kminus = ctx.post_trace - ctx.post_spike
        # potentiation on post spike, then depression on pre spike (NEST order)
        w = frozen(ctx.post_spike > 0, self._facilitate(w, kplus), w)
        w = frozen(ctx.pre_spike > 0, self._depress(w, kminus), w)
        return {'weight': w}, w
