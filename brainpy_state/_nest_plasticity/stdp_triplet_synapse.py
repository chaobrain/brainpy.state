# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-faithful ``stdp_triplet_synapse`` — Pfister-Gerstner triplet STDP spec + rule.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._nest_network.event_plastic.EventPlasticProj`. The triplet rule
(Pfister & Gerstner, 2006) augments pair STDP with a **second, slower trace on each
side**: potentiation at a post spike is scaled by the slow *post* trace and
depression at a pre spike by the slow *pre* trace, capturing frequency-dependent
plasticity that pair models miss. This is the first model to use the substrate's
**multi-trace seam** — ``pre_trace_tau`` / ``post_trace_tau`` are *tuples*
``(fast, slow)`` so the substrate allocates two per-neuron trace columns per side.
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

__all__ = ['stdp_triplet_synapse']


class stdp_triplet_synapse(NESTPlasticity):
    r"""Triplet spike-timing-dependent plasticity synapse spec (NEST ``stdp_triplet_synapse``).

    Four traces drive the rule — a fast/slow pair on each side, all decay-then-add
    on the substrate (current spike included):

    * ``r1`` (pre, ``tau_plus``), ``r2`` (pre, ``tau_plus_triplet``);
    * ``o1`` (post, ``tau_minus``), ``o2`` (post, ``tau_minus_triplet``).

    The kernel gates its own weight writeback — the online all-to-all scheme equal
    to NEST's deferred ``send()`` at every send (pre-spike) time:

    .. math::

       w &\leftarrow \min\!\big(w + r_1 (A_2^+ + A_3^+\, o_2),\ W_{\max}\big)
       \quad\text{(post spike)} \\
       w &\leftarrow \max\!\big(w - o_1 (A_2^- + A_3^-\, r_2),\ 0\big)
       \quad\text{(pre spike)}

    matching NEST ``facilitate_``/``depress_``: potentiation reads the fast *pre*
    trace ``r1`` weighted by the slow *post* trace ``o2``; depression reads the
    fast *post* trace ``o1`` weighted by the slow *pre* trace ``r2``. The slow
    trace on the triggering side is taken **just before** the current spike's own
    increment (the ``t-epsilon`` of the triplet rule), realised by excluding the
    current step's spike from *every* trace (``r = pre_traces - pre_spike``,
    ``o = post_traces - post_spike``).

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge weight (pA; bare numbers are pA). Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    tau_plus : Quantity, optional
        Fast presynaptic trace ``r1`` constant (> 0). Default ``16.8 ms``.
    tau_plus_triplet : Quantity, optional
        Slow presynaptic trace ``r2`` constant (> 0). Default ``101.0 ms``.
    tau_minus : Quantity, optional
        Fast postsynaptic trace ``o1`` constant (> 0). Default ``20.0 ms``.
    tau_minus_triplet : Quantity, optional
        Slow postsynaptic trace ``o2`` constant (> 0). Default ``110.0 ms``.
    Aplus : float, optional
        Pair potentiation amplitude :math:`A_2^+`. Default ``5e-10``.
    Aplus_triplet : float, optional
        Triplet potentiation amplitude :math:`A_3^+`. Default ``0.0062``.
    Aminus : float, optional
        Pair depression amplitude :math:`A_2^-`. Default ``0.007``.
    Aminus_triplet : float, optional
        Triplet depression amplitude :math:`A_3^-`. Default ``0.00023``.
    Wmax : float, optional
        Upper weight bound (depression floors at ``0``). Default ``100.0``.
    Kplus : float, optional
        Initial fast pre trace ``r1`` (the substrate seeds traces at 0, the NEST
        default). Default ``0.0``.
    Kplus_triplet : float, optional
        Initial slow pre trace ``r2`` (seeded at 0). Default ``0.0``.

    Notes
    -----
    **NEST divergence — ``tau_minus`` / ``tau_minus_triplet`` location.** In NEST
    both postsynaptic trace constants live on the *postsynaptic neuron*
    (``ArchivingNode``), not the synapse; here they are synapse-spec attributes
    driving the substrate's per-post ``o1`` / ``o2`` trace columns, so the triplet
    rule runs standalone.

    Online vs deferred: the substrate potentiates eagerly at post-spike steps,
    whereas NEST defers it to the next pre spike; the two coincide at pre-spike
    (send) times, so parity is asserted there.

    **Parity note.** The consolidated trace-storage and parameter-location
    reference (both post-trace constants move onto the synapse) and the parity-test
    links are in :doc:`/nest-style/divergences/stdp` (:ref:`stdp-tau-minus`).

    References
    ----------
    .. [1] NEST ``models/stdp_triplet_connection.h``; Pfister & Gerstner (2006),
       J. Neurosci. 26(38):9673-9682.

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy.state import stdp_triplet_synapse
       >>> s = stdp_triplet_synapse(weight=5.0)
       >>> s.is_homogeneous_weight, s.edge_state_init()
       (False, {})
       >>> len(s.pre_trace_tau), len(s.post_trace_tau)        # fast + slow per side
       (2, 2)
       >>> float(u.Quantity(s.pre_trace_tau[1]).to_decimal(u.ms))
       101.0
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 16.8 * u.ms,
        tau_plus_triplet: ArrayLike = 101.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        tau_minus_triplet: ArrayLike = 110.0 * u.ms,
        Aplus: ArrayLike = 5e-10,
        Aplus_triplet: ArrayLike = 0.0062,
        Aminus: ArrayLike = 0.007,
        Aminus_triplet: ArrayLike = 0.00023,
        Wmax: ArrayLike = 100.0,
        Kplus: ArrayLike = 0.0,
        Kplus_triplet: ArrayLike = 0.0,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.tau_plus = to_ms(tau_plus, name='tau_plus') * u.ms
        self.tau_plus_triplet = to_ms(tau_plus_triplet, name='tau_plus_triplet') * u.ms
        self.tau_minus = to_ms(tau_minus, name='tau_minus') * u.ms
        self.tau_minus_triplet = to_ms(tau_minus_triplet, name='tau_minus_triplet') * u.ms
        self.Aplus = to_scalar_float(Aplus, name='Aplus')
        self.Aplus_triplet = to_scalar_float(Aplus_triplet, name='Aplus_triplet')
        self.Aminus = to_scalar_float(Aminus, name='Aminus')
        self.Aminus_triplet = to_scalar_float(Aminus_triplet, name='Aminus_triplet')
        self.Wmax = to_scalar_float(Wmax, name='Wmax')
        self.Kplus = to_scalar_float(Kplus, name='Kplus')
        self.Kplus_triplet = to_scalar_float(Kplus_triplet, name='Kplus_triplet')

        for nm, q in (('tau_plus', tau_plus), ('tau_plus_triplet', tau_plus_triplet),
                      ('tau_minus', tau_minus), ('tau_minus_triplet', tau_minus_triplet)):
            if to_ms(q, name=nm) <= 0.0:
                raise ValueError(f"'{nm}' must be > 0.")

        # multi-trace seams: (fast, slow) per side; the substrate allocates two
        # per-neuron trace columns each and gathers (E, 2) into ctx.{pre,post}_traces.
        self.pre_trace_tau = (self.tau_plus, self.tau_plus_triplet)
        self.post_trace_tau = (self.tau_minus, self.tau_minus_triplet)

    def edge_state_init(self) -> dict:
        return {}

    # -- rule kernel (NEST facilitate_/depress_) ---------------------------
    def _facilitate(self, w, r1, o2):
        new_w = w + r1 * (self.Aplus + self.Aplus_triplet * o2)
        return jnp.minimum(new_w, self.Wmax)

    def _depress(self, w, o1, r2):
        new_w = w - o1 * (self.Aminus + self.Aminus_triplet * r2)
        return jnp.maximum(new_w, 0.0)

    def update(self, state, ctx):
        w = state['weight']
        # exclude this step's own spike from every trace (the triplet t-eps rule)
        r1 = ctx.pre_traces[:, 0] - ctx.pre_spike      # fast pre
        r2 = ctx.pre_traces[:, 1] - ctx.pre_spike      # slow pre
        o1 = ctx.post_traces[:, 0] - ctx.post_spike    # fast post
        o2 = ctx.post_traces[:, 1] - ctx.post_spike    # slow post
        # potentiation on post spike (fast pre x slow post), then depression on
        # pre spike (fast post x slow pre) — NEST send() order.
        w = frozen(ctx.post_spike > 0, self._facilitate(w, r1, o2), w)
        w = frozen(ctx.pre_spike > 0, self._depress(w, o1, r2), w)
        return {'weight': w}, w
