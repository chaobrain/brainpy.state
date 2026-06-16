# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-faithful ``vogels_sprekeler_synapse`` — symmetric inhibitory plasticity.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._network._event_plastic.EventPlasticProj`. The
Vogels-Sprekeler (2011) rule is a **symmetric** STDP with a **constant**
presynaptic depression, designed to homeostatically balance excitation and
inhibition: every pre↔post pairing potentiates by :math:`\eta K`, and every pre
spike additionally depresses by a constant :math:`\alpha\eta`, driving the
postsynaptic firing rate toward a target set by :math:`\alpha`.
"""
from __future__ import annotations
from ._base import NESTPlasticity

import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from ._plastic_base import (
    frozen, to_ms, to_scalar_float, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['vogels_sprekeler_synapse']


def _nest_sign(v: float) -> int:
    # NEST set_status sign check: (x >= 0) - (x < 0); zero counts as positive.
    return int(v >= 0.0) - int(v < 0.0)


class vogels_sprekeler_synapse(NESTPlasticity):
    r"""Symmetric inhibitory-plasticity synapse spec (NEST ``vogels_sprekeler_synapse``).

    The substrate maintains a **single, symmetric** trace constant ``tau`` on both
    sides (``pre_trace_tau = post_trace_tau = tau``). This kernel gates its own
    writeback — the online all-to-all scheme equal to NEST's deferred ``send()`` at
    every send (pre-spike) time:

    * **post spike** — symmetric facilitation using the pre trace ``K+``;
    * **pre spike** — symmetric facilitation using the post trace ``K-``, then a
      single **constant** depression (independent of any trace).

    with the sign-aware, magnitude-saturating operations (NEST
    ``facilitate_``/``depress_``)

    .. math::

       \operatorname{facilitate}(w, k) =
       \operatorname{copysign}\!\big(\min(|w| + \eta k,\ |W_{\max}|),\ W_{\max}\big)

       \operatorname{depress}(w) =
       \operatorname{copysign}\!\big(\max(|w| - \alpha\eta,\ 0),\ W_{\max}\big)

    so the weight saturates at :math:`\pm |W_{\max}|` while keeping ``Wmax``'s sign
    (weights are typically inhibitory). Each side excludes the current step's own
    spike from the opposite trace (``K+ = pre_trace - pre_spike``,
    ``K- = post_trace - post_spike``).

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge weight (pA; bare numbers are pA). Same sign as ``Wmax`` (if
        non-zero). Default ``0.5`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    tau : Quantity, optional
        Symmetric pre/post trace constant (> 0). Default ``20.0 ms``.
    eta : float, optional
        Learning rate :math:`\eta`. Default ``0.001``.
    alpha : float, optional
        Constant depression factor :math:`\alpha` (sets the target rate).
        Default ``0.12``.
    Wmax : float, optional
        Weight bound (magnitude; sign defines the weight sign). Default ``1.0``.
    Kplus : float, optional
        Initial ``K+`` (the substrate seeds traces at 0, the NEST default).
        Default ``0.0``.

    Notes
    -----
    **NEST divergence — ``tau`` location.** In NEST the postsynaptic trace ``K-``
    is maintained by the post neuron (``ArchivingNode``) under its ``tau_minus``;
    the symmetric rule requires ``tau_minus = tau``. Here ``tau`` is a single
    synapse-spec attribute driving both per-neuron traces, so the rule runs
    standalone. See ``CONTEXT.md`` Lessons (cluster 04).

    Online vs deferred: the substrate facilitates eagerly at post-spike steps,
    whereas NEST defers it to the next pre spike; the two coincide at pre-spike
    (send) times, so parity is asserted there.

    References
    ----------
    .. [1] NEST ``models/vogels_sprekeler_connection.h``; Vogels et al. (2011), Science.

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy_state import vogels_sprekeler_synapse
       >>> s = vogels_sprekeler_synapse(weight=0.5, alpha=0.12, eta=0.001)
       >>> s.is_homogeneous_weight, s.edge_state_init()
       (False, {})
       >>> s.eta, s.alpha, s.Wmax
       (0.001, 0.12, 1.0)
       >>> bool(u.Quantity(s.pre_trace_tau).to_decimal(u.ms) == u.Quantity(s.post_trace_tau).to_decimal(u.ms))
       True
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False

    def __init__(
        self,
        weight: ArrayLike = 0.5,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau: ArrayLike = 20.0 * u.ms,
        eta: ArrayLike = 0.001,
        alpha: ArrayLike = 0.12,
        Wmax: ArrayLike = 1.0,
        Kplus: ArrayLike = 0.0,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.tau = to_ms(tau, name='tau') * u.ms
        self.eta = to_scalar_float(eta, name='eta')
        self.alpha = to_scalar_float(alpha, name='alpha')
        self.Wmax = to_scalar_float(Wmax, name='Wmax')
        self.Kplus = to_scalar_float(Kplus, name='Kplus')

        if to_ms(tau, name='tau') <= 0.0:
            raise ValueError("'tau' must be > 0.")
        w0 = float(u.get_mantissa(self.weight))
        if w0 != 0.0 and _nest_sign(w0) != _nest_sign(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')

        # symmetric single-tau trace seam on both sides (substrate allocates them)
        self.pre_trace_tau = self.tau
        self.post_trace_tau = self.tau

    def edge_state_init(self) -> dict:
        return {}

    # -- rule kernel (sign-aware, magnitude-saturating) --------------------
    def _facilitate(self, w, k):
        mag = jnp.minimum(jnp.abs(w) + self.eta * k, abs(self.Wmax))
        return jnp.copysign(mag, self.Wmax)

    def _depress(self, w):
        mag = jnp.maximum(jnp.abs(w) - self.alpha * self.eta, 0.0)
        return jnp.copysign(mag, self.Wmax)

    def update(self, state, ctx):
        w = state['weight']
        # exclude this step's own spike from the opposite-side trace
        kplus = ctx.pre_trace - ctx.pre_spike
        kminus = ctx.post_trace - ctx.post_spike
        # post spike: symmetric facilitation using the pre trace K+
        w = frozen(ctx.post_spike > 0, self._facilitate(w, kplus), w)
        # pre spike: symmetric facilitation using the post trace K-, then constant depression
        w = frozen(ctx.pre_spike > 0, self._depress(self._facilitate(w, kminus)), w)
        return {'weight': w}, w
