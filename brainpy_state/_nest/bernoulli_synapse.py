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

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike

from .static_synapse import static_synapse

__all__ = [
    'bernoulli_synapse',
]


_UNSET = object()


class bernoulli_synapse(static_synapse):
    r"""NEST-compatible ``bernoulli_synapse`` connection model.

    Short description
    -----------------

    Static synapse with stochastic transmission.

    Description
    -----------

    ``bernoulli_synapse`` mirrors NEST ``models/bernoulli_synapse.h``.
    The model is non-plastic and stores fixed parameters:

    - synaptic weight ``weight``,
    - synaptic delay ``delay``,
    - receiver port ``receptor_type``,
    - transmission probability ``p_transmit``.

    On each outgoing event, one Bernoulli trial is drawn and the event is
    transmitted only if the trial succeeds:

    .. math::

       \mathrm{send} \iff U < p_{\mathrm{transmit}}, \quad U \sim \mathrm{Uniform}(0, 1).

    A failed trial drops the event entirely.

    Event send ordering (NEST source equivalent)
    --------------------------------------------

    NEST ``models/bernoulli_synapse.h`` performs:

    1. Draw Bernoulli decision from uniform random number and ``p_transmit``.
    2. If successful:
       ``e.set_weight(weight_)``
    3. ``e.set_delay_steps(get_delay_steps())``
    4. ``e.set_receiver(*get_target(tid))``
    5. ``e.set_rport(get_rport())``
    6. ``e()`` (deliver event)

    This implementation preserves the same semantics: stochastic transmission
    is decided before scheduling; when accepted, inherited
    :class:`static_synapse` scheduling applies weight, delay steps, receiver
    and receptor port in the same effective order.

    Parameters
    ----------
    weight : ArrayLike, optional
        Fixed synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    p_transmit : ArrayLike, optional
        Spike transmission probability in ``[0, 1]``.
        Default: ``1.0``.
    post : object, optional
        Default receiver object.
    event_type : str, optional
        Event transmission type (same options as :class:`static_synapse`).
        Default: ``'spike'``.
    name : str, optional
        Object name.

    Notes
    -----
    - This model does not implement plasticity.
    - Random draws use NumPy's global RNG state. Use ``np.random.seed(...)``
      for reproducible test traces when needed.

    References
    ----------
    .. [1] NEST source: ``models/bernoulli_synapse.h`` and
           ``models/bernoulli_synapse.cpp``.
    .. [2] Lefort S, Tomm C, Sarria J-C F, Petersen CCH (2009).
           The excitatory neuronal network of the C2 barrel column in mouse
           primary somatosensory cortex. Neuron, 61(2):301-316.
           DOI: https://doi.org/10.1016/j.neuron.2008.12.020
    .. [3] Teramae J, Tsubo Y, Fukai T (2012). Optimal spike-based
           communication in excitable networks with strong-sparse and
           weak-dense links. Scientific Reports 2, 485.
           DOI: https://doi.org/10.1038/srep00485
    .. [4] Omura Y, Carvalho MM, Inokuchi K, Fukai T (2015).
           A lognormal recurrent network model for burst generation during
           hippocampal sharp waves. Journal of Neuroscience, 35(43):14585-14601.
           DOI: https://doi.org/10.1523/JNEUROSCI.4944-14.2015
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        p_transmit: ArrayLike = 1.0,
        post=None,
        event_type: str = 'spike',
        name: str | None = None,
    ):
        super().__init__(
            weight=weight,
            delay=delay,
            receptor_type=receptor_type,
            post=post,
            event_type=event_type,
            name=name,
        )

        self.p_transmit = self._to_scalar_probability(p_transmit)
        self._validate_probability(self.p_transmit)

    @staticmethod
    def _to_scalar_probability(value: ArrayLike) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError('p_transmit must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _validate_probability(p_transmit: float):
        if p_transmit < 0.0 or p_transmit > 1.0:
            raise ValueError('Spike transmission probability must be in [0, 1].')

    @staticmethod
    def _sample_uniform() -> float:
        return float(np.random.random())

    def get(self) -> dict:
        """Return current public parameters."""
        params = super().get()
        params['p_transmit'] = float(self.p_transmit)
        params['synapse_model'] = 'bernoulli_synapse'
        return params

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        p_transmit: ArrayLike | object = _UNSET,
        post: object = _UNSET,
        event_type: str | object = _UNSET,
    ):
        """Set NEST-style public parameters."""
        new_p = (
            self.p_transmit
            if p_transmit is _UNSET
            else self._to_scalar_probability(p_transmit)
        )
        self._validate_probability(new_p)

        super_kwargs = {}
        if weight is not _UNSET:
            super_kwargs['weight'] = weight
        if delay is not _UNSET:
            super_kwargs['delay'] = delay
        if receptor_type is not _UNSET:
            super_kwargs['receptor_type'] = receptor_type
        if post is not _UNSET:
            super_kwargs['post'] = post
        if event_type is not _UNSET:
            super_kwargs['event_type'] = event_type
        if super_kwargs:
            super().set(**super_kwargs)

        self.p_transmit = float(new_p)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
        event_type: str | None = None,
    ) -> bool:
        """Stochastically schedule one outgoing event."""
        if not self._is_nonzero(multiplicity):
            return False

        send_event = self._sample_uniform() < self.p_transmit
        if not send_event:
            return False

        return super().send(
            multiplicity,
            post=post,
            receptor_type=receptor_type,
            event_type=event_type,
        )
