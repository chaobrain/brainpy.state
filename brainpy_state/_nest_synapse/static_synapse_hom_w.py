# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``static_synapse_hom_w`` — homogeneous-weight static synapse.

Rebuilt as a frozen spec + trivial pure rule on
:class:`~brainpy_state._nest_network.event_plastic.EventPlasticProj`. Identical to
:class:`~brainpy_state._nest_synapse.static_synapse.static_synapse` except the weight is
a single value shared by every connection (NEST common property): the substrate
stores ``'weight'`` as a 0-d :class:`brainstate.ParamState`. Per-connection
``delay`` and ``receptor_type`` remain settable.
"""
from __future__ import annotations
from brainpy_state._nest_base.base import NESTSynapse

from collections.abc import Mapping

import brainunit as u
from brainstate.typing import ArrayLike

from brainpy_state._nest_base.plastic_base import unit_of, validate_delay, validate_receptor_type, weight_to_pa

__all__ = ['static_synapse_hom_w']


class static_synapse_hom_w(NESTSynapse):
    r"""Static synapse with a single weight shared across all connections.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        The common synaptic weight (pA; bare numbers are interpreted as pA).
        Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.

    See Also
    --------
    static_synapse : Per-connection-weight static synapse.

    Notes
    -----
    Attempting to give individual connections their own weight is rejected, as
    in NEST: :meth:`check_synapse_params` raises when a connection ``syn_spec``
    carries a ``'weight'`` key, and :meth:`set_weight` raises with NEST's
    *"Setting of individual weights is not possible!"* message.

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy.state import static_synapse_hom_w
       >>> s = static_synapse_hom_w(weight=2.0 * u.pA)
       >>> s.is_homogeneous_weight
       True
       >>> s.edge_state_init()
       {}
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = True
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

    @staticmethod
    def check_synapse_params(syn_spec: Mapping[str, object] | None) -> None:
        """Reject per-connection ``weight`` in a connection spec (NEST parity)."""
        if syn_spec is None:
            return
        if 'weight' in syn_spec:
            raise ValueError(
                'Weight cannot be specified since it needs to be equal for all '
                'connections when static_synapse_hom_w is used.'
            )

    @staticmethod
    def set_weight(weight) -> None:
        """Reject setting an individual weight (NEST parity)."""
        del weight
        raise ValueError(
            'Setting of individual weights is not possible! The common weights '
            'can be changed via CopyModel().'
        )
