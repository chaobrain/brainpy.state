# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
from typing import Dict

import brainstate

from brainpy_state._base import Neuron
from brainpy_state._brainpy.projection import Projection
from brainpy_state._nest._base import NESTDevice

__all__ = ['Network']


class Network(brainstate.nn.Module):
    """brainpy.state network base class.

    Subclass and define populations, projections, and devices as
    attributes. ``update()`` walks the immediate module-tree children
    in projection-first then dynamics order.
    """
    __module__ = 'brainpy.state'

    def update(self, t=None) -> None:
        # Depth-1 traversal — matches the existing brainpy.state convention
        # documented in brainpy_state/_brainpy/projection.py:46-51. Networks
        # that need nested projection chains (Projection containing
        # Projection) currently must override update() explicitly; this is
        # tracked as an open question in the design spec §12.
        children = self.nodes(allowed_hierarchy=(1, 1))
        projections = [m for m in children.values() if isinstance(m, Projection)]
        others = [m for m in children.values() if not isinstance(m, Projection)]
        for m in projections:
            m()
        for m in others:
            m()

    @property
    def populations(self) -> Dict[str, Neuron]:
        return {k[-1]: m for k, m in self.nodes(allowed_hierarchy=(1, 1)).items()
                if isinstance(m, Neuron)}

    @property
    def projections(self) -> Dict[str, Projection]:
        return {k[-1]: m for k, m in self.nodes(allowed_hierarchy=(1, 1)).items()
                if isinstance(m, Projection)}

    @property
    def devices(self) -> Dict[str, NESTDevice]:
        return {k[-1]: m for k, m in self.nodes(allowed_hierarchy=(1, 1)).items()
                if isinstance(m, NESTDevice)}
