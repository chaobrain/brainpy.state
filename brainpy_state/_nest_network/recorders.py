# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Recorder — wires a passive NESTDevice to a source population attribute."""
from __future__ import annotations

from collections.abc import Callable

import brainstate

from brainpy_state._base import Dynamics

__all__ = ['Recorder']


class Recorder(brainstate.nn.Module):
    """Forward a source-population signal to a passive NESTDevice each step.

    Parameters
    ----------
    source : Dynamics
        Module to read from.
    attr : str or callable
        - If ``str``: attribute name on ``source`` whose ``.value`` is read
          (e.g. ``'V'``).
        - If callable: invoked as ``attr(source)`` each step. Useful for
          quantities that aren't persisted as a ``State`` — e.g. spikes:
          ``attr=lambda s: s.get_spike(s.V.value)``.
    device : NESTDevice
        Recording device with a compatible ``update`` signature
        (e.g. ``spike_recorder``, ``multimeter``).
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        source: Dynamics,
        attr: str | Callable,
        device,
    ):
        super().__init__()
        self.source = source
        self.attr = attr
        self.device = device

    def update(self, *args, **kwargs):
        if callable(self.attr):
            val = self.attr(self.source)
        else:
            if not hasattr(self.source, self.attr):
                raise AttributeError(
                    f'source {type(self.source).__name__} has no attribute '
                    f'{self.attr!r} — did you call '
                    f'brainstate.nn.init_all_states(net)?'
                )
            val = getattr(self.source, self.attr)
            if hasattr(val, 'value'):
                val = val.value
        self.device.update(val)
