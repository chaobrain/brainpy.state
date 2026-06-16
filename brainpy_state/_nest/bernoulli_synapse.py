# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``bernoulli_synapse`` — static synapse with stochastic transmission.

Rebuilt as a frozen parameter spec plus a pure, *stochastic*
``update(state, ctx)`` rule kernel on
:class:`~brainpy_state._network._event_plastic.EventPlasticProj`. The synapse is
non-plastic: on each presynaptic spike it delivers the full (per-edge) ``weight``
with probability ``p_transmit`` and drops it otherwise — a per-edge Bernoulli gate
on the delivered amplitude, with **no weight state evolving**.
"""
from __future__ import annotations
from ._base import NESTSynapse

import jax
import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from ._plastic_base import (
    to_unit_interval, unit_of, validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['bernoulli_synapse']


class bernoulli_synapse(NESTSynapse):
    r"""Static synapse with stochastic (Bernoulli) transmission (NEST ``bernoulli_synapse``).

    Each presynaptic spike on edge ``e`` is transmitted *independently per
    connection* with probability ``p_transmit``: with that probability the full
    amplitude ``weight[e]`` (pA) is delivered, otherwise nothing. The weight is
    otherwise static — no per-edge state evolves (the synapse is memoryless).

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge synaptic weight (pA; bare numbers interpreted as pA, sign
        preserved). Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay; must be finite and strictly positive. Grid
        quantization is handled by the substrate's
        :class:`~brainpy_state._brainpy._delay.InputDelay`. Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (non-negative integer). Default ``0``.
    p_transmit : float, optional
        Spike transmission probability, in ``[0, 1]``. ``p_transmit = 1``
        reproduces :class:`static_synapse` exactly; ``p_transmit = 0`` delivers
        nothing. Default ``1.0``.

    Raises
    ------
    ValueError
        If ``p_transmit`` is outside ``[0, 1]`` (NEST ``BadProperty``), the delay
        is non-positive/non-finite, or ``receptor_type < 0``.

    See Also
    --------
    static_synapse : Deterministic fixed-weight delivery (the ``p_transmit=1`` limit).
    quantal_stp_synapse : Probabilistic (binomial) short-term plasticity.

    Notes
    -----
    NEST ``models/bernoulli_synapse.h`` ``send()`` (lines 164-175) draws one
    uniform variate ``u`` from the per-connection RNG and delivers the full weight
    iff ``u < p_transmit_``; ``set_status`` (lines 214-217) validates
    ``p_transmit \in [0, 1]``.

    **PRNG carve-out (distributional parity only).** NEST draws one Bernoulli per
    spike from a per-thread RNG; this kernel draws
    ``jax.random.uniform(ctx.key, (E,)) < p_transmit`` — a length-``E`` vector from
    the substrate's single per-step key. JAX's counter-based (threefry) PRNG makes
    the ``E`` draws **independent across edges** (output position ``j`` derives from
    counter ``j``), so two edges sharing a presynaptic neuron, and multapses, gate
    independently; across steps the substrate re-splits the key. The streams are
    *not* bit-identical to NEST, so parity is **distributional** (the transmitted
    fraction converges to ``p_transmit``; the per-step delivered count is
    ``Binomial(n_fired, p_transmit)``). No per-edge ``jax.random.split`` seam is
    needed — the shape-``(E,)`` draw already supplies edge-axis independence.

    References
    ----------
    .. [1] NEST ``models/bernoulli_synapse.h`` ``send()`` (lines 164-175).
    .. [2] Lefort S, Tomm C, Sarria J-C F, Petersen CCH (2009). The excitatory
           neuronal network of the C2 barrel column in mouse primary somatosensory
           cortex. Neuron, 61(2):301-316.

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy_state import bernoulli_synapse
       >>> s = bernoulli_synapse(weight=20.0 * u.pA, p_transmit=0.5)
       >>> s.stochastic
       True
       >>> s.edge_state_init()
       {}
       >>> float(s.p_transmit)
       0.5
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = True                    # makes the substrate pass a non-None ctx.key
    pre_trace_tau = None
    post_trace_tau = None

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        p_transmit: ArrayLike = 1.0,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)
        self.p_transmit = to_unit_interval(p_transmit, name='p_transmit')

    def edge_state_init(self) -> dict:
        return {}

    def update(self, state, ctx):
        # Per-edge Bernoulli gate on the delivered amplitude (no weight mutated):
        # draw one uniform per edge from the step key (independent across edges),
        # deliver the full weight where the edge fired AND the trial succeeds.
        mask = jax.random.uniform(ctx.key, ctx.pre_spike.shape) < self.p_transmit
        w_eff = jnp.where((ctx.pre_spike > 0) & mask, state['weight'], 0.0)
        return state, w_eff
