# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``static_synapse`` — fixed-weight synapse spec + trivial rule.

Rebuilt as a frozen parameter spec plus the simplest possible pure rule kernel
on :class:`~brainpy_state._nest_network.event_plastic.EventPlasticProj`: the
effective weight is just the (per-edge) constant ``weight``, and no state
evolves.
"""
from __future__ import annotations
from brainpy_state._nest_base.base import NESTSynapse

import brainunit as u
from brainstate.typing import ArrayLike

from brainpy_state._nest_base.plastic_base import unit_of, validate_delay, validate_receptor_type, weight_to_pa

__all__ = ['static_synapse']


class static_synapse(NESTSynapse):
    r"""Fixed-weight, fixed-delay synapse spec (NEST ``static_synapse``).

    A presynaptic spike on edge ``e`` delivers the constant amplitude
    ``weight[e]`` (pA) to the postsynaptic neuron after ``delay``; no per-edge
    state evolves.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge synaptic weight (pA; bare numbers interpreted as pA, sign
        preserved: positive excitatory, negative inhibitory). Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay; must be finite and strictly positive. Grid
        quantization (NEST rounds to integer steps, minimum one step) is handled
        by the substrate's :class:`~brainpy_state._brainpy.delay.InputDelay`.
        Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (non-negative integer). Default ``0``.

    See Also
    --------
    static_synapse_hom_w : Variant with a single weight shared across edges.

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy_state import static_synapse
       >>> s = static_synapse(weight=20.0 * u.pA)
       >>> s.is_homogeneous_weight
       False
       >>> s.edge_state_init()
       {}
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
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

    def edge_state_init(self) -> dict:
        return {}

    def update(self, state, ctx):
        return state, state['weight']
