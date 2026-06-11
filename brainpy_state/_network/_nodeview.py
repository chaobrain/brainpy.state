# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NodeView — NEST NodeCollection-style view over (population, local indices)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import jax.numpy as jnp

__all__ = ['NodeView']


def _flat_size(module) -> int:
    """Total number of elements along a population/device neuron dimension.

    Mirrors ``brainpy_state._network._projections._size`` (``in_size`` first,
    falling back to ``varshape``) so the network layer agrees on one notion of
    population size.
    """
    sz = module.in_size if hasattr(module, 'in_size') else module.varshape
    if isinstance(sz, tuple):
        n = 1
        for s in sz:
            n *= int(s)
        return n
    return int(sz)


@dataclass(frozen=True)
class _Segment:
    """A contiguous reference into one population: ``(population, local idx)``."""
    population: object
    indices: jnp.ndarray  # 1-D int array of local indices into the population


class NodeView:
    """A view over one or more slices of populations/devices (NEST-style).

    Mimics NEST ``NodeCollection`` algebra: ``a + b`` concatenates two views
    (preserving segment boundaries) and ``a[sl]`` slices a single-segment view.
    Each segment references a population and a 1-D array of local indices into
    that population's flattened neuron dimension.
    """
    __module__ = 'brainpy.state'

    def __init__(self, segments: List[_Segment]):
        self._segments = list(segments)

    @classmethod
    def of(cls, population) -> 'NodeView':
        """Build a full-population view over ``population``."""
        return cls([_Segment(population, jnp.arange(_flat_size(population)))])

    @property
    def segments(self) -> List[_Segment]:
        return self._segments

    @property
    def size(self) -> int:
        return int(sum(int(s.indices.shape[0]) for s in self._segments))

    def __add__(self, other: 'NodeView') -> 'NodeView':
        if not isinstance(other, NodeView):
            return NotImplemented
        return NodeView(self._segments + other._segments)

    def __getitem__(self, item) -> 'NodeView':
        if len(self._segments) != 1:
            raise NotImplementedError('slicing is supported on single-segment views only')
        seg = self._segments[0]
        idx = seg.indices[item]
        idx = idx[None] if idx.ndim == 0 else idx
        return NodeView([_Segment(seg.population, idx)])

    def __len__(self) -> int:
        return self.size
