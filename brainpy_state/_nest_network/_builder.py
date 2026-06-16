# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Builder — imperative subclass of Network for scripts and notebooks."""
from __future__ import annotations

import itertools
from typing import Union

import brainstate

from brainpy_state._base import Dynamics
from brainpy_state._brainpy.projection import Projection
from brainpy_state._nest_network._base import Network

__all__ = ['Builder']


class Builder(Network):
    """Imperative variant of :class:`Network`.

    Use ``add(name, module)`` to register a population/device and
    ``connect(pre, post, rule=..., **params)`` to register a projection.
    Otherwise identical to :class:`Network`.
    """
    __module__ = 'brainpy.state'

    def __init__(self):
        super().__init__()
        self._proj_counter = itertools.count()

    def add(self, name: str, module: brainstate.nn.Module) -> brainstate.nn.Module:
        """Register ``module`` as ``self.<name>``; return the module."""
        if hasattr(self, name):
            raise ValueError(f'attribute {name!r} already exists on this Builder')
        setattr(self, name, module)
        return module

    def connect(
        self,
        pre: Union[str, Dynamics],
        post: Union[str, Dynamics],
        *,
        rule: type,
        **kwargs,
    ) -> Projection:
        """Instantiate ``rule(pre, post, **kwargs)`` and register it."""
        pre_mod = getattr(self, pre) if isinstance(pre, str) else pre
        post_mod = getattr(self, post) if isinstance(post, str) else post
        proj = rule(pre_mod, post_mod, **kwargs)
        attr = f'_proj_{next(self._proj_counter)}'
        setattr(self, attr, proj)
        return proj
