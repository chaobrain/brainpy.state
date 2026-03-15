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


import saiunit as u
import jax.numpy as jnp
import numpy as np
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

    **1. Mathematical formulation**

    Stochastic transmission probability: Each spike arriving at the synapse
    triggers a Bernoulli trial with success probability :math:`p_{\mathrm{transmit}}`.
    The trial outcome determines whether the event is forwarded to the
    postsynaptic target. Formally:

    .. math::

       P(\text{transmit spike}) = p_{\mathrm{transmit}}

    The transmission decision for each spike is independent of previous trials.
    No state variables are maintained across spikes — the synapse is memoryless.

    **2. Implementation details**

    Random number generation uses NumPy's global RNG state (``np.random.random()``).
    The trial outcome is computed by comparing a uniform random variate
    :math:`U \sim \mathrm{Uniform}(0, 1)` to the threshold :math:`p_{\mathrm{transmit}}`:

    .. math::

       \text{transmit} = \begin{cases}
         \text{True}, & U < p_{\mathrm{transmit}} \\
         \text{False}, & \text{otherwise}
       \end{cases}

    If the trial fails, the spike is discarded before scheduling. If successful,
    the spike is forwarded using the parent :class:`static_synapse` delivery
    mechanism with the configured ``weight`` and ``delay``.

    **3. Event send ordering (NEST source equivalent)**

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

    **4. Biological motivation**

    Synaptic transmission in real neural circuits is stochastic due to finite
    vesicle release probabilities and quantal failures. This model approximates
    such unreliable transmission using a simple Bernoulli process, which is
    computationally efficient and captures the essential statistical properties
    of transmission failures.

    Applications include:

    - Sparse networks with probabilistic connectivity (e.g., cortical column
      models, [2]_).
    - Networks with strong-sparse and weak-dense connectivity motifs [3]_.
    - Recurrent networks where transmission failures contribute to decorrelation
      and robustness [4]_.

    **5. Computational considerations**

    - Random number generation introduces non-determinism. For reproducible
      simulations, seed the NumPy RNG before initializing the network:
      ``np.random.seed(42)``.
    - Each spike invokes one call to ``np.random.random()``, adding negligible
      computational overhead compared to deterministic synapses.
    - The model is stateless and suitable for large-scale network simulations
      where memory footprint per synapse is a constraint.

    Parameters
    ----------
    weight : ArrayLike, optional
        Fixed synaptic weight (dimensionless or receiver-dependent units such as
        pA, nS, or mV). Must be scalar. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Must be scalar. Converted to integer simulation
        steps using round-to-nearest with midpoint-up (NEST convention).
        Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Used to target specific synaptic receptor
        channels on the postsynaptic neuron. Default: ``0``.
    p_transmit : ArrayLike, optional
        Spike transmission probability in ``[0, 1]``. Must be scalar. Values
        outside this range raise ``ValueError``. Default: ``1.0``.
    post : object, optional
        Default receiver object (postsynaptic neuron or target module). If not
        provided, a receiver must be passed explicitly to :meth:`send` when
        scheduling events.
    event_type : str, optional
        Event transmission type. Inherited from :class:`static_synapse`. One of
        ``'spike'``, ``'rate'``, ``'current'``, ``'conductance'``,
        ``'double_data'``, ``'data_logging'``. Default: ``'spike'``.
    name : str, optional
        Object name for debugging and introspection.

    Raises
    ------
    ValueError
        If ``p_transmit`` is not scalar or is outside ``[0, 1]``.

    See Also
    --------
    static_synapse : Parent class implementing deterministic event scheduling.
    tsodyks_synapse : Short-term plasticity model with dynamic transmission.

    Notes
    -----
    - This model does not implement plasticity (no weight updates, no state
      evolution).
    - Random draws use NumPy's global RNG state. Use ``np.random.seed(...)``
      for reproducible test traces when needed.
    - The model is compatible with all event types supported by
      :class:`static_synapse`, though biological applications typically use
      ``event_type='spike'``.
    - Transmission failures are independent across spikes and across different
      synapse instances.

    Examples
    --------
    Create a stochastic synapse with 50% transmission probability:

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import saiunit as u
       >>> post_neuron = bst.LIF(1)
       >>> syn = bst.bernoulli_synapse(
       ...     weight=5.0,
       ...     delay=2.0 * u.ms,
       ...     p_transmit=0.5,
       ...     post=post_neuron,
       ... )

    Simulate stochastic transmission and observe dropped spikes:

    .. code-block:: python

       >>> import numpy as np
       >>> np.random.seed(42)  # for reproducibility
       >>> transmitted = [syn.send(multiplicity=1.0) for _ in range(10)]
       >>> transmitted
       [False, True, False, True, True, False, True, False, False, True]

    Query current parameters:

    .. code-block:: python

       >>> params = syn.get()
       >>> params['p_transmit']
       0.5

    Update transmission probability dynamically:

    .. code-block:: python

       >>> syn.set(p_transmit=0.8)
       >>> syn.p_transmit
       0.8

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
        r"""Convert input to scalar float for probability validation.

        Parameters
        ----------
        value : ArrayLike
            Input value to convert. Must be scalar-compatible (size 1).

        Returns
        -------
        float
            Scalar float value extracted from input.

        Raises
        ------
        ValueError
            If input is not scalar (size != 1).
        """
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError('p_transmit must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _validate_probability(p_transmit: float):
        r"""Validate that transmission probability is in valid range.

        Parameters
        ----------
        p_transmit : float
            Transmission probability to validate.

        Raises
        ------
        ValueError
            If ``p_transmit`` is outside the interval ``[0, 1]``.
        """
        if p_transmit < 0.0 or p_transmit > 1.0:
            raise ValueError('Spike transmission probability must be in [0, 1].')

    @staticmethod
    def _sample_uniform() -> float:
        r"""Draw one uniform random variate in [0, 1).

        Uses NumPy's global RNG state (``np.random.random()``).

        Returns
        -------
        float
            Uniform random value in ``[0, 1)``.

        Notes
        -----
        This method is stateless and relies on NumPy's global random number
        generator. For reproducible results, seed the RNG before simulation:
        ``np.random.seed(42)``.
        """
        return float(np.random.random())

    def get(self) -> dict:
        r"""Return current public parameters.

        Returns
        -------
        dict
            Dictionary containing all connection parameters:

            - ``'weight'`` : float — synaptic weight.
            - ``'delay'`` : float — synaptic delay in ms.
            - ``'receptor_type'`` : int — receiver port id.
            - ``'p_transmit'`` : float — transmission probability in ``[0, 1]``.
            - ``'synapse_model'`` : str — model identifier (``'bernoulli_synapse'``).

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.bernoulli_synapse(weight=2.0, p_transmit=0.7)
           >>> params = syn.get()
           >>> params['p_transmit']
           0.7
        """
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
        r"""Set NEST-style public parameters.

        Updates one or more connection parameters. All parameters are optional;
        unspecified parameters retain their current values.

        Parameters
        ----------
        weight : ArrayLike, optional
            New synaptic weight. Must be scalar.
        delay : ArrayLike, optional
            New synaptic delay in ms. Must be scalar.
        receptor_type : int, optional
            New receiver port/receptor id.
        p_transmit : ArrayLike, optional
            New transmission probability in ``[0, 1]``. Must be scalar.
        post : object, optional
            New default receiver object.
        event_type : str, optional
            New event transmission type (``'spike'``, ``'rate'``, ``'current'``,
            ``'conductance'``, ``'double_data'``, ``'data_logging'``).

        Raises
        ------
        ValueError
            If ``p_transmit`` is provided and is not scalar or is outside ``[0, 1]``.

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.bernoulli_synapse(weight=1.0, p_transmit=0.5)
           >>> syn.set(p_transmit=0.8, weight=2.0)
           >>> syn.p_transmit
           0.8
        """
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
        r"""Stochastically schedule one outgoing event.

        Performs a Bernoulli trial to decide whether to transmit the event. If
        the trial succeeds, the event is scheduled for delivery using the
        inherited :class:`static_synapse` mechanism. If the trial fails, the
        event is discarded.

        Parameters
        ----------
        multiplicity : ArrayLike, optional
            Event payload value (spike count, rate, or current). If zero or
            negligible (< 1e-12), the event is dropped without drawing a random
            number. Default: ``1.0``.
        post : object, optional
            Receiver object (postsynaptic neuron). Overrides the default
            receiver set in :meth:`__init__` or :meth:`set`. If both are
            ``None``, scheduling will fail.
        receptor_type : int, optional
            Receiver port/receptor id. Overrides the default receptor type.
        event_type : str, optional
            Event transmission type. Overrides the default event type.

        Returns
        -------
        bool
            ``True`` if the event was successfully transmitted and scheduled.
            ``False`` if the event was zero, or if the Bernoulli trial failed.

        Notes
        -----
        - Zero-valued events (``multiplicity < 1e-12``) are dropped before the
          Bernoulli trial to avoid unnecessary random number generation.
        - Each call draws one uniform random variate from NumPy's global RNG.
        - The transmission decision is independent of previous calls.

        Examples
        --------
        Simulate stochastic transmission with 50% probability:

        .. code-block:: python

           >>> import numpy as np
           >>> np.random.seed(42)
           >>> syn = bst.bernoulli_synapse(p_transmit=0.5, post=post_neuron)
           >>> results = [syn.send(multiplicity=1.0) for _ in range(10)]
           >>> results
           [False, True, False, True, True, False, True, False, False, True]
           >>> sum(results)  # approximately 5 out of 10
           5

        Zero-valued events are dropped without drawing random numbers:

        .. code-block:: python

           >>> syn.send(multiplicity=0.0)
           False
        """
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
