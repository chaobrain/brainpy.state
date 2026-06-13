# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-faithful ``stdp_nn_pre_centered_synapse`` — presynaptic-centered nearest-neighbour STDP.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._network._event_plastic.EventPlasticProj`. In the
presynaptic-centered scheme (Morrison, Diesmann & Gerstner 2008, fig. 7B) the
presynaptic trace ``Kplus`` **accumulates** (``+1`` per pre, decays at ``tau_plus``)
but is **reset to 0 on every post spike** (``stdp_nn_pre_centered_synapse.h:69-74``):
a post spike facilitates with the sum of all pres since the previous post, and a pre
spike depresses with the nearest preceding post. Because the reset is triggered by the
*postsynaptic* spike, ``Kplus`` is genuinely **per-edge** (two synapses from the same
pre to different posts reset at different times), so it is carried in
:meth:`edge_state_init` and decayed in-kernel — *not* the substrate's per-pre-neuron
trace (``pre_trace_tau = None``). Only the nearest ``K-`` comes from the substrate
(``post_trace_mode = 'nearest'``). The previous imperative implementation lived in
:mod:`brainpy_state._nest._legacy_stdp_synapse`.
"""
from __future__ import annotations

import jax.numpy as jnp
import saiunit as u
from brainstate.typing import ArrayLike

from ._plastic_base import (
    frozen, to_ms, to_scalar_float, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['stdp_nn_pre_centered_synapse']


def _nest_sign(v: float) -> int:
    # NEST set_status sign check: (x >= 0) - (x < 0); zero counts as positive.
    return int(v >= 0.0) - int(v < 0.0)


class stdp_nn_pre_centered_synapse:
    r"""Presynaptic-centered nearest-neighbour STDP synapse spec (NEST ``stdp_nn_pre_centered_synapse``).

    The presynaptic trace ``Kplus`` is **per-edge**: it decays at ``tau_plus``,
    increments by 1 on each pre spike, and **resets to 0 on each post spike**, so it
    holds the sum of all pres since the previous post. A post spike facilitates with
    that accumulated ``Kplus`` (then erases it); a pre spike depresses with the nearest
    preceding post (substrate ``'nearest'`` ``K-``). The weight maps are those of
    :class:`stdp_synapse`:

    .. math::

       \hat w \leftarrow \hat w + \lambda (1-\hat w)^{\mu_+} K^+ \quad\text{(post spike)}

       \hat w \leftarrow \hat w - \alpha \lambda \hat w^{\mu_-} K^- \quad\text{(pre spike)}

    with :math:`\hat w = w/W_{\max}`, clamped to :math:`[0, W_{\max}]`. The kernel
    decays ``Kplus`` *before* adding this step's ``+1``, so a pre coinciding with a post
    is excluded from that step's facilitation — the second-latest convention of
    ``stdp_nn_pre_centered_synapse.h:63-67``. This also makes the model immune to the
    phantom-pre-at-0 (facilitation is ``Kplus·exp``, and ``Kplus`` starts at 0).

    Divergences: vs :class:`stdp_nn_symm_synapse`, a post pairs with the *sum* of pres
    since the last post (not just the nearest); vs all-to-all :class:`stdp_synapse`, the
    post-triggered reset means earlier pres are forgotten after each post.

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
    Kplus : float, optional
        Initial per-edge ``K+`` (>= 0). Default ``0.0`` (NEST default).

    Notes
    -----
    **NEST divergence — ``tau_minus`` location.** In NEST ``tau_minus`` is a parameter
    of the *postsynaptic neuron* (``ArchivingNode``), not the synapse; here it drives the
    substrate's per-post ``K-`` trace. See ``CONTEXT.md`` Lessons (cluster 04).

    The eager substrate applies potentiation at post-spike steps and depression at
    pre-spike steps; NEST defers both to the next ``send`` (facilitating only the *first*
    post since the last pre, which equals the eager scheme because the reset zeroes
    ``Kplus`` after that first post). The op sets coincide at the pre-spike (send) steps
    the ``weight_recorder`` samples.

    **Parity note.** The exact nearest-neighbour pairing convention, the NEST
    source citation, and the single-pair regression test are documented in
    :doc:`/nest-guide/stdp-divergences` (:ref:`stdp-nn-pre-centered`).

    References
    ----------
    .. [1] NEST ``models/stdp_nn_pre_centered_synapse.h`` (``send()`` 249-317: accumulated
       ``Kplus`` facilitation + reset at 287-294, nearest ``Kminus`` depress at 299-302,
       ``Kplus`` decay/+1 at 304). Morrison, Diesmann & Gerstner (2008) Biol. Cybern.
       98:459-478, fig. 7B; Izhikevich & Desai (2003) Neural Comput. 15:1511-1523.

    See Also
    --------
    stdp_nn_symm_synapse, stdp_nn_restr_synapse, stdp_synapse

    Examples
    --------
    .. code-block:: python

       >>> import saiunit as u
       >>> from brainpy_state import stdp_nn_pre_centered_synapse
       >>> s = stdp_nn_pre_centered_synapse(weight=1.0, tau_plus=20.0 * u.ms)
       >>> s.post_trace_mode, s.pre_trace_tau
       ('nearest', None)
       >>> s.edge_state_init()
       {'Kplus': 0.0}
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False
    # only the postsynaptic K- comes from the substrate (per-post-neuron, nearest);
    # the presynaptic Kplus is per-edge (reset by post) and lives in edge_state_init.
    post_trace_mode = 'nearest'
    pre_trace_tau = None

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
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self._tau_plus_ms = to_ms(tau_plus, name='tau_plus')
        self.tau_plus = self._tau_plus_ms * u.ms
        self.tau_minus = to_ms(tau_minus, name='tau_minus') * u.ms
        self.lambda_ = to_scalar_float(lambda_, name='lambda')
        self.alpha = to_scalar_float(alpha, name='alpha')
        self.mu_plus = to_scalar_float(mu_plus, name='mu_plus')
        self.mu_minus = to_scalar_float(mu_minus, name='mu_minus')
        self.Wmax = to_scalar_float(Wmax, name='Wmax')
        self.Kplus = to_scalar_float(Kplus, name='Kplus')

        if self._tau_plus_ms <= 0.0:
            raise ValueError("'tau_plus' must be > 0.")
        if to_ms(tau_minus, name='tau_minus') <= 0.0:
            raise ValueError("'tau_minus' must be > 0.")
        w0 = float(u.get_mantissa(self.weight))
        if _nest_sign(w0) != _nest_sign(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')
        if self.Kplus < 0.0:
            raise ValueError('Kplus must be non-negative.')

        # only the postsynaptic trace is allocated by the substrate
        self.post_trace_tau = self.tau_minus

    def edge_state_init(self) -> dict:
        # per-edge accumulating presynaptic trace (reset by post spikes)
        return {'Kplus': self.Kplus}

    # -- rule kernel -------------------------------------------------------
    def _facilitate(self, w, kplus):
        norm_w = w / self.Wmax + self.lambda_ * (1.0 - w / self.Wmax) ** self.mu_plus * kplus
        return jnp.where(norm_w < 1.0, norm_w * self.Wmax, self.Wmax)

    def _depress(self, w, kminus):
        norm_w = w / self.Wmax - self.alpha * self.lambda_ * (w / self.Wmax) ** self.mu_minus * kminus
        return jnp.where(norm_w > 0.0, norm_w * self.Wmax, 0.0)

    def update(self, state, ctx):
        w = state['weight']
        kplus = state['Kplus']
        pre_fired = ctx.pre_spike > 0
        post_fired = ctx.post_spike > 0
        kminus = ctx.post_trace - ctx.post_spike            # nearest post (2nd-latest on coincide)
        # decay the per-edge presynaptic trace to this step (before this pre's +1)
        kplus = kplus * jnp.exp(-ctx.dt / self._tau_plus_ms)
        # POST spike: facilitate with the accumulated Kplus, then erase it
        w = frozen(post_fired, self._facilitate(w, kplus), w)
        kplus = jnp.where(post_fired, 0.0, kplus)
        # PRE spike: depress with the nearest post, then accumulate this pre
        w = frozen(pre_fired, self._depress(w, kminus), w)
        kplus = kplus + jnp.where(pre_fired, 1.0, 0.0)
        return {'weight': w, 'Kplus': kplus}, w
