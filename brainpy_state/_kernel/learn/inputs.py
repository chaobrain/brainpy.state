# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Data entry point of a learning network: :class:`ExternalInput`.

Learners feed per-step data through ``Program.step(..., ext=...)`` into input ports.
``ExternalInput`` is a stateless relay population whose ``x`` port receives that
data and whose ``spike`` output re-emits it unchanged, so the data reaches the rest
of the network through ordinary (trainable) projections, delays and storages.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.integrators import Exact, Integrator
from brainpy_state._kernel.model import NeuronModel
from brainpy_state._kernel.params import Record
from brainpy_state._kernel.ports import Event, Ports
from brainpy_state._kernel.units import to_mantissa

__all__ = ['ExternalInput']


class ExternalInput(NeuronModel):
    """Relay of external per-step data (spike counts or any float payload).

    The value given to port ``x`` at step ``n`` is the ``spike`` output of step
    ``n`` (no latch, no state). The output declares ``max_multiplicity=None``, so
    projections use the exact float payload path and any float value (including
    relaxed spikes in gradient checks) is transmitted as is.

    Parameters
    ----------
    n : int
        Number of input channels.
    **kwargs
        ``NeuronModel`` arguments (``precision``, ``name``, ...).

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel import StepContext
        >>> from brainpy_state._kernel.learn.inputs import ExternalInput
        >>> relay = ExternalInput(3)
        >>> ctx = StepContext.create(0.0 * u.ms, 0.1 * u.ms)
        >>> relay.update(ctx, {'x': [0.0, 1.0, 2.0]})['spike'].tolist()
        [0.0, 1.0, 2.0]
    """

    inputs = Ports(x=Event(max_multiplicity=None))
    outputs = Ports(spike=Event(max_multiplicity=None))
    integrator: Integrator = Exact()

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Emit this step's ``x`` input on ``spike``."""
        return s, {'spike': to_mantissa(inp['x'])}
