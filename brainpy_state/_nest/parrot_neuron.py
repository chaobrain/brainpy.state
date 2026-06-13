# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

from typing import Optional

import brainstate
import jax.numpy as jnp
import saiunit as u
from brainstate.typing import Size

from ._base import NESTNeuron

__all__ = [
    'parrot_neuron',
]


class parrot_neuron(NESTNeuron):
    r"""NEST-compatible ``parrot_neuron`` — repeats every incoming spike.

    Description
    -----------

    The parrot neuron emits exactly **one spike for every incoming spike**. Its
    canonical use is to fan a single stochastic train out to many targets: a
    ``poisson_generator`` sends an *independent* train to each of its targets, so
    connecting one generator to a single ``parrot_neuron`` and then the parrot to
    a group of neurons delivers the *same* Poisson train to every member of the
    group. In ``brainpy.state`` it also bridges a stimulation device (e.g. a
    ``poisson_generator`` window) onto a specific receptor port of a
    multi-receptor neuron via ``connect(parrot, neuron, receptor_type=k)``.

    Weights on connections *to* the parrot are **ignored** — a spike is a spike,
    so any non-zero arriving event triggers exactly one relayed spike. Weights on
    connections *from* the parrot are honored normally by the downstream
    projection. Transmission delays are honored on both sides.

    Parameters
    ----------
    in_size : int or sequence of int
        Number of parrot channels (population shape). Each channel relays its own
        incoming events independently.
    name : str, optional
        Name identifier for the population. If ``None``, an automatic name is
        generated.

    See Also
    --------
    spike_generator : Deterministic spike-train source.
    poisson_generator : Stochastic (Poisson) spike-train source.

    Notes
    -----
    - The model is **memoryless**: its output at step :math:`t` depends only on
      the events delivered at step :math:`t`. It therefore holds no
      :class:`~brainstate.HiddenState`; ``init_state`` / ``reset_state`` are
      no-ops.
    - Incoming events are read through the standard delta-input seam
      (:meth:`~brainpy_state._base.Dynamics.sum_delta_inputs`), the same channel
      an :class:`~brainpy_state._network._event_proj.EventProjection` deposits
      into. A relayed spike is emitted whenever the summed arriving input is
      non-zero. Because connection weights to the parrot are ignored, drive it
      with a plain (unitless) gate weight such as ``weight=1.0``.
    - Inside the :class:`~brainpy_state.network.Simulator`, the per-population
      one-step spike holder plus each connection's delay reproduce NEST's relay
      latency (the parrot's spike is delivered to its own targets one step after
      it is emitted, just as for any spiking population).
    - NEST's ``parrot_neuron`` accepts events on a second port (port 1) that are
      *not* repeated, used to set exact pre/post spike times for STDP protocols.
      That secondary port is out of scope here: this model exposes a single relay
      port.

    Examples
    --------
    Fan one Poisson train out to a population so every target sees the *same*
    train:

    .. code-block:: python

        >>> import saiunit as u
        >>> from brainpy_state import (Simulator, parrot_neuron,
        ...                            poisson_generator, iaf_psc_alpha)
        >>> sim = Simulator(dt=0.1 * u.ms)
        >>> pg = sim.create(poisson_generator, rate=1000. * u.Hz)
        >>> relay = sim.create(parrot_neuron, 1)
        >>> pop = sim.create(iaf_psc_alpha, 50)
        >>> sim.connect(pg, relay, weight=1.0, delay=1. * u.ms)   # weight ignored
        >>> sim.connect(relay, pop, weight=100. * u.pA, delay=1.5 * u.ms)
        >>> res = sim.simulate(100. * u.ms)
    """
    __module__ = 'brainpy.state'

    def __init__(self, in_size: Size, name: Optional[str] = None):
        super().__init__(in_size, name=name)

    def init_state(self, batch_size=None, **kwargs):
        r"""Initialize runtime state.

        The parrot is memoryless and holds no state, so this is a no-op kept for
        API compatibility with :func:`brainstate.nn.init_all_states`.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused; accepted for base-state API compatibility.
        **kwargs
            Unused compatibility parameters.
        """

    def reset_state(self, batch_size=None, **kwargs):
        r"""Reset runtime state.

        No-op: the parrot holds no state to reset.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused; accepted for base-state API compatibility.
        **kwargs
            Unused compatibility parameters.
        """

    def get_spike(self, inp=None):
        r"""Map arriving input to a relayed spike.

        Parameters
        ----------
        inp : ArrayLike or None, optional
            Summed delta input delivered this step, broadcast-compatible with
            ``self.varshape``. If ``None``, no input is treated as arriving and
            the output is all zeros.

        Returns
        -------
        jax.Array
            Binary spike output (``1.0`` where a spike arrived, else ``0.0``),
            shape ``self.varshape``.
        """
        dftype = brainstate.environ.dftype()
        if inp is None:
            return jnp.zeros(self.varshape, dtype=dftype)
        # Connection weights to the parrot are ignored: any non-zero arriving
        # event (positive or negative) relays as exactly one spike.
        arrived = u.get_mantissa(inp) != 0
        return arrived.astype(dftype)

    def update(self, x=None):
        r"""Advance one step: re-emit a spike for every arriving event.

        Parameters
        ----------
        x : ArrayLike, optional
            Unused. Incoming events are read from the delta-input seam, not from
            a positional argument; ``x`` is accepted only for update-signature
            uniformity with other neuron models.

        Returns
        -------
        jax.Array
            Binary relayed spike output for this step, shape ``self.varshape``.
        """
        inp = self.sum_delta_inputs(jnp.zeros(self.varshape, dtype=brainstate.environ.dftype()))
        return self.get_spike(inp)
