# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-faithful ``stdp_nn_restr_synapse`` — restricted symmetric nearest-neighbour STDP.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._network._event_plastic.EventPlasticProj`. The *restricted*
symmetric nearest-neighbour scheme (Morrison, Diesmann & Gerstner 2008, fig. 7C) is
the symmetric scheme plus a one-pairing-per-spike restriction: a post spike
facilitates with the nearest preceding pre **only if a pre has occurred since the
previous post**, and a pre spike depresses with the nearest preceding post **only if
a post has occurred since the previous pre** (``stdp_nn_restr_synapse.h:54-60``). NEST
realises this in ``send()`` by gating both updates on ``start != finish`` — whether the
postsynaptic history window ``(t_lastspike-d, t_spike-d]`` is non-empty. Here the
nearest traces come from the substrate (``pre_trace_mode = post_trace_mode = 'nearest'``)
and the gates are two per-edge eligibility flags carried in :meth:`edge_state_init`: a
spike makes its own side *available* and *consumes* the opposite side, so each spike
pairs at most once. The previous imperative implementation lived in
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

__all__ = ['stdp_nn_restr_synapse']


def _nest_sign(v: float) -> int:
    # NEST set_status sign check: (x >= 0) - (x < 0); zero counts as positive.
    return int(v >= 0.0) - int(v < 0.0)


class stdp_nn_restr_synapse:
    r"""Restricted symmetric nearest-neighbour STDP synapse spec (NEST ``stdp_nn_restr_synapse``).

    Like :class:`stdp_nn_symm_synapse`, both ``K+`` and ``K-`` run in the substrate's
    ``'nearest'`` mode (reset-to-1 on spike); the difference is a **one-pair-per-spike
    restriction** carried by two per-edge eligibility flags:

    * ``pre_avail`` — a pre has occurred since the previous post (a post may facilitate),
    * ``post_avail`` — a post has occurred since the previous pre (a pre may depress).

    On a post spike, potentiation fires **only if** ``pre_avail`` (then the pre is
    consumed); on a pre spike, depression fires **only if** ``post_avail`` (then the
    post is consumed). A spike sets its own side available and clears the opposite. The
    weight maps are the same as :class:`stdp_synapse`:

    .. math::

       \hat w \leftarrow \hat w + \lambda (1-\hat w)^{\mu_+} K^+
       \quad\text{(post spike, if } \mathtt{pre\_avail})

       \hat w \leftarrow \hat w - \alpha \lambda \hat w^{\mu_-} K^-
       \quad\text{(pre spike, if } \mathtt{post\_avail})

    with :math:`\hat w = w/W_{\max}`, clamped to :math:`[0, W_{\max}]`. The gathered
    trace excludes the current step's own spike (``K+ = pre_trace - pre_spike``,
    ``K- = post_trace - post_spike``), which recovers the second-latest partner on an
    exactly-coinciding step (``stdp_nn_restr_synapse.h:62-66``). The restriction makes
    restr diverge from symm whenever several spikes of one side fall between two of the
    other: symm pairs every one, restr only the first.

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
    synapse-spec attribute driving the substrate's per-post ``K-`` trace. See
    ``CONTEXT.md`` Lessons (cluster 04).

    **Phantom-pre-at-0 (shared with symm).** NEST's first send (``t_lastspike_=0``)
    facilitates a post preceding the first pre against a virtual pre at ``t=0``; the
    substrate's ``pre_avail`` flag starts at 0, so that post is simply not eligible — the
    physically correct nearest behaviour. Parity is asserted where this is absent/below
    tolerance. See ``CONTEXT.md`` Lessons (05).

    The eager substrate applies potentiation at post-spike steps and depression at
    pre-spike steps; NEST defers both to the next ``send``. The cumulative op set is
    identical at every send, so the trajectories coincide where the ``weight_recorder``
    samples (pre-spike steps).

    **Parity note.** The exact nearest-neighbour pairing convention, the NEST
    source citation, and the single-pair regression test are documented in
    :doc:`/nest-guide/stdp-divergences` (:ref:`stdp-nn-restr`).

    References
    ----------
    .. [1] NEST ``models/stdp_nn_restr_synapse.h`` (``send()`` 244-307: facilitation gated
       on ``start != finish`` at 270-283, nearest ``Kminus`` depress at 287-297). Morrison,
       Diesmann & Gerstner (2008) Biol. Cybern. 98:459-478, fig. 7C.

    See Also
    --------
    stdp_nn_symm_synapse, stdp_synapse, stdp_nn_pre_centered_synapse

    Examples
    --------
    .. code-block:: python

       >>> import saiunit as u
       >>> from brainpy_state import stdp_nn_restr_synapse
       >>> s = stdp_nn_restr_synapse(weight=1.0, tau_plus=20.0 * u.ms)
       >>> s.pre_trace_mode, s.post_trace_mode
       ('nearest', 'nearest')
       >>> s.edge_state_init()
       {'pre_avail': 0.0, 'post_avail': 0.0}
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False
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

        self.pre_trace_tau = self.tau_plus
        self.post_trace_tau = self.tau_minus

    def edge_state_init(self) -> dict:
        # two eligibility flags; both start 0 (NEST t_lastspike_=0: nothing pairable yet)
        return {'pre_avail': 0.0, 'post_avail': 0.0}

    # -- rule kernel -------------------------------------------------------
    def _facilitate(self, w, kplus):
        norm_w = w / self.Wmax + self.lambda_ * (1.0 - w / self.Wmax) ** self.mu_plus * kplus
        return jnp.where(norm_w < 1.0, norm_w * self.Wmax, self.Wmax)

    def _depress(self, w, kminus):
        norm_w = w / self.Wmax - self.alpha * self.lambda_ * (w / self.Wmax) ** self.mu_minus * kminus
        return jnp.where(norm_w > 0.0, norm_w * self.Wmax, 0.0)

    def update(self, state, ctx):
        w = state['weight']
        pre_avail = state['pre_avail']
        post_avail = state['post_avail']
        pre_fired = ctx.pre_spike > 0
        post_fired = ctx.post_spike > 0
        # nearest partner, excluding this step's own spike (second-latest on coincidence)
        kplus = ctx.pre_trace - ctx.pre_spike
        kminus = ctx.post_trace - ctx.post_spike
        # potentiation on post spike, restricted to an available (unused) pre
        w = frozen(post_fired & (pre_avail > 0), self._facilitate(w, kplus), w)
        # depression on pre spike, restricted to an available (unused) post
        w = frozen(pre_fired & (post_avail > 0), self._depress(w, kminus), w)
        # each spike makes its own side available and consumes the opposite side
        new_pre_avail = jnp.where(pre_fired, 1.0, jnp.where(post_fired, 0.0, pre_avail))
        new_post_avail = jnp.where(post_fired, 1.0, jnp.where(pre_fired, 0.0, post_avail))
        return {'weight': w, 'pre_avail': new_pre_avail, 'post_avail': new_post_avail}, w
