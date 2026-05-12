# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import brainstate

__all__ = ['Network']


class Network(brainstate.nn.Module):
    """brainpy.state network base class.

    Subclass and define populations, projections, and devices as
    attributes. See ``brainpy.state.Builder`` for an imperative
    variant of the same underlying object.
    """
    __module__ = 'brainpy.state'
