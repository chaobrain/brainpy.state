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


from collections.abc import Mapping

from brainstate.typing import ArrayLike

from .static_synapse import static_synapse

__all__ = [
    'static_synapse_hom_w',
]


class static_synapse_hom_w(static_synapse):
    r"""NEST-compatible ``static_synapse_hom_w`` with shared homogeneous weight.

    ``static_synapse_hom_w`` implements a static (non-plastic) synaptic connection
    with a single shared weight parameter across all connections instantiated from
    this model. This design reduces memory overhead for networks with many synapses
    that share identical weights, mirroring NEST's ``CommonPropertiesHomW`` template.

    The model inherits all event transmission, delay scheduling, and receptor routing
    semantics from :class:`static_synapse`, but enforces a critical constraint: the
    weight parameter is a **model-level property** rather than a per-connection
    attribute. Attempting to set individual connection weights raises a ``ValueError``.

    **1. Mathematical Model**

    The event transmission equation is identical to :class:`static_synapse`:

    .. math::

       \text{output}(t + d) = w_{\text{common}} \cdot \text{input}(t)

    where :math:`w_{\text{common}}` is the single shared weight value. All connections
    using this model apply the same weight scaling factor.

    **2. Homogeneous Weight Semantics**

    **Model-level weight management:**

    - **Initialization**: The ``weight`` parameter passed to ``__init__`` becomes the
      shared weight for all connections.
    - **Runtime modification**: Calling ``set(weight=...)`` updates the common weight,
      affecting **all** connections immediately.
    - **Per-connection restriction**: ``set_weight(weight)`` raises ``ValueError`` to
      prevent accidental per-connection weight assignment.
    - **Connection specification**: Providing ``weight`` in synapse specification dicts
      (e.g., NEST ``syn_spec``) is forbidden; checked by ``check_synapse_params``.

    This constraint ensures that:

    1. Memory usage scales with number of unique weight values (one float), not
       number of connections.
    2. Weight modifications propagate instantly to all connections.
    3. The API prevents accidental violations of homogeneity.

    **3. Event Processing Pipeline**

    NEST ``static_synapse_hom_w`` event transmission follows this sequence
    (from ``models/static_synapse_hom_w.h:send()``):

    1. **Weight retrieval**: ``e.set_weight(cp.get_weight())`` — fetch common weight
       from ``CommonPropertiesHomW``
    2. **Delay assignment**: ``e.set_delay_steps(get_delay_steps())``
    3. **Receiver resolution**: ``e.set_receiver(*get_target(tid))``
    4. **Receptor port**: ``e.set_rport(get_rport())``
    5. **Event delivery**: ``e()`` — trigger receiver's event handler

    This implementation preserves the same ordering by inheriting the
    :meth:`static_synapse.send` and :meth:`static_synapse.update` methods unchanged.
    The weight value is accessed via ``self.weight`` during event scheduling.

    **4. Use Cases and Design Rationale**

    **When to use ``static_synapse_hom_w``:**

    - Large-scale networks with uniform connection strengths within populations
    - Memory-constrained environments (embedded systems, large-scale simulations)
    - Networks where connection weights are controlled algorithmically (e.g., all
      inhibitory→excitatory synapses have weight -0.5)
    - Rapid exploration of weight parameter space (single-value updates)

    **When to use ``static_synapse`` instead:**

    - Heterogeneous connection strengths (e.g., distance-dependent weights)
    - Per-connection weight modifications (e.g., normalization, homeostatic scaling)
    - Connection pruning or weight initialization from learned patterns

    **Memory trade-offs:**

    - ``static_synapse``: :math:`O(N)` weight storage for :math:`N` connections
    - ``static_synapse_hom_w``: :math:`O(1)` weight storage (one shared value)

    For a network with :math:`10^6` connections, this reduces weight memory from
    ~4 MB (float32) to 4 bytes.

    **5. NEST Compatibility Notes**

    This implementation replicates NEST behavior with minor API differences:

    **Matching NEST semantics:**

    - ``GetStatus()`` returns ``synapse_model: 'static_synapse_hom_w'``
    - ``set_weight()`` raises error: "individual weights cannot be set"
    - Weight modification only via model-level ``CopyModel()`` equivalent
    - Event ordering matches NEST ``Connection::send()`` pipeline

    **brainpy.state API adaptations:**

    - NEST uses ``CopyModel()`` to create weight variants; brainpy.state uses
      ``set(weight=...)`` on the model instance
    - NEST checks weight specification at connection creation; brainpy.state checks
      in ``check_synapse_params()`` (typically called by projection classes)
    - ``delay`` remains per-connection (NEST allows heterogeneous delays even with
      homogeneous weights)

    Parameters
    ----------
    weight : float, array-like, or Quantity, optional
        Common synaptic weight shared across all connections using this model.
        Scalar value, dimensionless or with units (e.g., ``0.5*u.nS`` for
        conductance-based, ``100*u.pA`` for current-based).
        Default: ``1.0`` (dimensionless).
    delay : float, array-like, or Quantity, optional
        Synaptic transmission delay. Can be connection-specific despite homogeneous
        weight. Must be positive scalar with time units (recommended: ``saiunit.ms``).
        Will be discretized to integer time steps according to simulation resolution.
        Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receptor port identifier on the postsynaptic neuron. Non-negative integer
        specifying which input channel receives events. Can be connection-specific.
        Default: ``0`` (primary receptor port).
    post : Dynamics, optional
        Default postsynaptic receiver object. If provided, :meth:`send` and
        :meth:`update` will target this receiver unless overridden.
        Default: ``None`` (must provide receiver explicitly in method calls).
    event_type : str, optional
        Type of event to transmit. Determines delivery method and receiver handling.
        Must be one of: ``'spike'``, ``'rate'``, ``'current'``, ``'conductance'``,
        ``'double_data'``, ``'data_logging'``.
        Default: ``'spike'`` (binary spike events).
    name : str, optional
        Unique identifier for this synapse model instance.
        Default: auto-generated.

    Parameter Mapping

    NEST ``static_synapse_hom_w`` parameters map to this implementation as follows:

    =======================  ====================  ========================================
    NEST Parameter           brainpy.state Param   Notes
    =======================  ====================  ========================================
    ``weight`` (common)      ``weight``            Scalar, model-level (not per-connection)
    ``delay``                ``delay``             Per-connection, converted to ms
    ``receptor_type``        ``receptor_type``     Per-connection, integer ≥ 0
    (connection target)      ``post``              Explicit receiver object
    (event class)            ``event_type``        String identifier for event routing
    =======================  ====================  ========================================

    Attributes
    ----------
    weight : float or Quantity
        Current common synaptic weight (read/write via :meth:`set`, NOT :meth:`set_weight`).
    delay : float
        Effective transmission delay in milliseconds (quantized to time steps).
        Inherited from :class:`static_synapse`.
    receptor_type : int
        Current receptor port identifier. Inherited from :class:`static_synapse`.
    post : Dynamics or None
        Default postsynaptic receiver. Inherited from :class:`static_synapse`.
    event_type : str
        Current event transmission type. Inherited from :class:`static_synapse`.

    Raises
    ------
    ValueError
        If ``set_weight(weight)`` is called (per-connection weight assignment forbidden).
    ValueError
        If ``check_synapse_params(syn_spec)`` receives ``weight`` in ``syn_spec`` dict.

    See Also
    --------
    static_synapse : Base static synapse with per-connection weights
    tsodyks_synapse : Short-term plasticity extension (also has homogeneous variant)

    Notes
    -----
    **Design pattern for large-scale networks:**

    When building networks with multiple weight classes (e.g., strong/weak synapses),
    create separate ``static_synapse_hom_w`` instances rather than one instance with
    heterogeneous weights:

    .. code-block:: python

        # Good: Two model instances, two weight values
        strong_syn = static_synapse_hom_w(weight=2.0, delay=1.0*u.ms)
        weak_syn = static_synapse_hom_w(weight=0.5, delay=1.0*u.ms)

        # Bad: Would require per-connection weights → use static_synapse instead
        # mixed_syn = static_synapse(weight=weights_array, delay=1.0*u.ms)

    **Thread safety and concurrency:**

    Not thread-safe. Concurrent calls to ``set(weight=...)`` from multiple threads
    will cause race conditions. Use thread-local synapse instances or external locking.

    **Performance characteristics:**

    - Memory: :math:`O(1)` for weight storage regardless of connection count
    - Computation: Identical to ``static_synapse`` (weight is a scalar lookup)
    - Weight update latency: :math:`O(1)` to modify, affects all connections instantly

    References
    ----------
    .. [1] NEST source code: ``models/static_synapse_hom_w.h``
           https://github.com/nest/nest-simulator
    .. [2] NEST source code: ``nestkernel/common_properties_hom_w.h``
           Template for homogeneous connection properties
           https://github.com/nest/nest-simulator
    .. [3] Morrison, A., Straube, S., Plesser, H. E., & Diesmann, M. (2007).
           Exact subthreshold integration with continuous spike times in discrete-time
           neural network simulations. *Neural Computation*, 19(1), 47-79.
           https://doi.org/10.1162/neco.2007.19.1.47

    Examples
    --------
    **Basic usage with homogeneous weight:**

    .. code-block:: python

        >>> import brainpy.state as bs
        >>> import saiunit as u
        >>> import brainstate
        >>> with brainstate.environ.context(dt=0.1 * u.ms):
        ...     # Create postsynaptic population
        ...     post_neurons = bs.LIF(100, V_rest=-65*u.mV, V_th=-50*u.mV, tau=20*u.ms)
        ...
        ...     # Create synapse with common weight
        ...     syn = bs.static_synapse_hom_w(
        ...         weight=0.5*u.nS,  # Shared across all connections
        ...         delay=1.5*u.ms,
        ...         receptor_type=0,
        ...         post=post_neurons[0],
        ...     )
        ...
        ...     # All connections use weight=0.5
        ...     syn.send(multiplicity=1.0)  # Delivers 0.5*1.0 = 0.5 to receiver

    **Modifying common weight at runtime:**

    .. code-block:: python

        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     syn = bs.static_synapse_hom_w(weight=1.0, delay=1.0*u.ms)
        ...
        ...     # Update common weight (affects all connections)
        ...     syn.set(weight=2.0)
        ...     assert syn.weight == 2.0
        ...
        ...     # Verify via GetStatus
        ...     params = syn.get()
        ...     assert params['synapse_model'] == 'static_synapse_hom_w'
        ...     assert params['weight'] == 2.0

    **Forbidden per-connection weight assignment:**

    .. code-block:: python

        >>> syn = bs.static_synapse_hom_w(weight=1.0)
        >>> try:
        ...     syn.set_weight(2.5)  # Attempt per-connection weight change
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Setting of individual weights is not possible! The common weights can be changed via CopyModel().

    **Checking synapse specifications (projection API):**

    .. code-block:: python

        >>> syn = bs.static_synapse_hom_w(weight=1.0)
        >>>
        >>> # Valid: delay and receptor_type can be per-connection
        >>> syn.check_synapse_params({'delay': 2.0*u.ms, 'receptor_type': 1})
        >>>
        >>> # Invalid: weight cannot be specified per-connection
        >>> try:
        ...     syn.check_synapse_params({'weight': 2.0})
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Weight cannot be specified since it needs to be equal for all connections when static_synapse_hom_w is used.

    **Memory-efficient large-scale network:**

    .. code-block:: python

        >>> # 10^6 excitatory connections, all weight=0.8
        >>> exc_syn = bs.static_synapse_hom_w(weight=0.8*u.nS, delay=1.0*u.ms)
        >>> # Memory: ~4 bytes for weight vs ~4 MB for per-connection weights
        >>>
        >>> # 10^6 inhibitory connections, all weight=-0.4
        >>> inh_syn = bs.static_synapse_hom_w(weight=-0.4*u.nS, delay=1.2*u.ms)
        >>> # Total memory: ~8 bytes for two weight values

    **Global weight scaling (homeostatic regulation):**

    .. code-block:: python

        >>> # Simulate 1000 steps with homeostatic scaling
        >>> syn = bs.static_synapse_hom_w(weight=1.0, post=neuron)
        >>> for step in range(1000):
        ...     delivered = syn.update(pre_spike=spike_train[step])
        ...
        ...     # Every 100 steps, reduce weight by 1%
        ...     if step % 100 == 0:
        ...         current_weight = syn.weight
        ...         syn.set(weight=current_weight * 0.99)

    **Comparing with per-connection weights:**

    .. code-block:: python

        >>> # Homogeneous weights: efficient for uniform connections
        >>> hom_syn = bs.static_synapse_hom_w(weight=1.0)
        >>> print(f"Weight storage: {hom_syn.weight}")  # Single scalar
        >>>
        >>> # Heterogeneous weights: use static_synapse instead
        >>> het_syn = bs.static_synapse(weight=np.random.uniform(0.5, 1.5, 1000))
        >>> # (Note: static_synapse API may differ; this is conceptual)

    **Multi-receptor configuration with shared weights:**

    .. code-block:: python

        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     target = bs.LIF(1, V_rest=-65*u.mV, V_th=-50*u.mV, tau=20*u.ms)
        ...
        ...     # Excitatory inputs on receptor 0 (all weight=0.8)
        ...     exc_syn = bs.static_synapse_hom_w(
        ...         weight=0.8*u.nS,
        ...         delay=1.0*u.ms,
        ...         receptor_type=0,
        ...         post=target,
        ...     )
        ...
        ...     # Inhibitory inputs on receptor 1 (all weight=-0.4)
        ...     inh_syn = bs.static_synapse_hom_w(
        ...         weight=-0.4*u.nS,
        ...         delay=1.2*u.ms,
        ...         receptor_type=1,
        ...         post=target,
        ...     )
        ...
        ...     # Multiple connections can share each synapse model
        ...     for pre_idx in range(100):
        ...         exc_syn.send(multiplicity=1.0)  # All use weight=0.8
        ...     for pre_idx in range(50):
        ...         inh_syn.send(multiplicity=1.0)  # All use weight=-0.4
    """

    __module__ = 'brainpy.state'

    def get(self) -> dict:
        r"""Retrieve current synapse parameters (NEST ``GetStatus`` equivalent).

        Returns a dictionary of all public synapse parameters, identical to
        :meth:`static_synapse.get` but with ``synapse_model`` set to
        ``'static_synapse_hom_w'`` for NEST compatibility.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``'weight'`` : float — Current common synaptic weight (shared across connections)
            - ``'delay'`` : float — Effective delay in milliseconds (quantized)
            - ``'delay_steps'`` : int — Delay in simulation time steps
            - ``'receptor_type'`` : int — Receptor port identifier
            - ``'event_type'`` : str — Event transmission type
            - ``'synapse_model'`` : str — Always ``'static_synapse_hom_w'`` (distinguishes from base class)

        Notes
        -----
        The returned ``weight`` value is the **common model-level weight**, not a
        per-connection value. All connections using this synapse model share this
        single weight parameter.

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bs
            >>> import saiunit as u
            >>> import brainstate
            >>> with brainstate.environ.context(dt=0.1*u.ms):
            ...     syn = bs.static_synapse_hom_w(weight=1.5, delay=2.0*u.ms, receptor_type=1)
            ...     params = syn.get()
            ...     print(params['synapse_model'])
            static_synapse_hom_w
            >>> assert params['weight'] == 1.5
            >>> assert params['synapse_model'] == 'static_synapse_hom_w'

        See Also
        --------
        static_synapse.get : Base class parameter retrieval
        set : Update synapse parameters
        """
        params = super().get()
        params['synapse_model'] = 'static_synapse_hom_w'
        return params

    def set_weight(self, weight: ArrayLike):
        r"""Reject per-connection weight assignment (NEST compatibility constraint).

        This method is intentionally disabled for ``static_synapse_hom_w`` to enforce
        the homogeneous weight constraint. NEST prevents per-connection weight
        modifications for ``static_synapse_hom_w`` connections because the weight is
        a model-level property shared across all connections, not a per-connection
        attribute.

        Parameters
        ----------
        weight : float, array-like, or Quantity
            Attempted per-connection weight value (always rejected).

        Raises
        ------
        ValueError
            Always raised with message indicating weights must be changed at the
            model level via :meth:`set` (equivalent to NEST ``CopyModel()``).

        Notes
        -----
        **NEST API correspondence:**

        In NEST, attempting to set individual weights on ``static_synapse_hom_w``
        connections fails with error:

        .. code-block:: text

            BadProperty: Setting of individual weights is not possible!
            The common weights can be changed via CopyModel().

        This implementation replicates that behavior by raising ``ValueError`` with
        an identical error message.

        **Correct weight modification approach:**

        To change the common weight, use :meth:`set` instead:

        .. code-block:: python

            syn.set(weight=new_value)  # Correct: updates model-level weight

        This updates the shared weight for **all** connections using this synapse model.

        **Design rationale:**

        The restriction exists to prevent accidental violations of weight homogeneity.
        If per-connection weights are needed, use :class:`static_synapse` instead.

        See Also
        --------
        set : Update common synapse weight (correct method to use)
        check_synapse_params : Connection-level weight specification validator

        Examples
        --------
        **Attempting per-connection weight modification:**

        .. code-block:: python

            >>> import brainpy.state as bs
            >>> syn = bs.static_synapse_hom_w(weight=1.0)
            >>>
            >>> # This raises ValueError
            >>> try:
            ...     syn.set_weight(2.5)
            ... except ValueError as e:
            ...     print(str(e))
            Setting of individual weights is not possible! The common weights can be changed via CopyModel().

        **Correct weight modification:**

        .. code-block:: python

            >>> # Use set() to update common weight
            >>> syn.set(weight=2.5)
            >>> assert syn.weight == 2.5  # All connections now use weight=2.5

        **NEST script translation:**

        NEST code:

        .. code-block:: python

            # NEST: Create model with homogeneous weight
            nest.CopyModel("static_synapse_hom_w", "exc_syn", {"weight": 1.5})
            nest.Connect(pre, post, syn_spec="exc_syn")

            # NEST: Modify common weight
            nest.SetDefaults("exc_syn", {"weight": 2.0})

        brainpy.state equivalent:

        .. code-block:: python

            # brainpy.state: Create synapse with homogeneous weight
            exc_syn = bs.static_synapse_hom_w(weight=1.5)

            # brainpy.state: Modify common weight
            exc_syn.set(weight=2.0)
        """
        del weight
        raise ValueError(
            'Setting of individual weights is not possible! The common weights '
            'can be changed via CopyModel().'
        )

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        r"""Validate synapse specification parameters for NEST compatibility.

        Checks that connection-level synapse specifications do not attempt to set
        per-connection weights, which violates the homogeneous weight constraint.
        This method is typically called by projection/connection classes during
        network construction to enforce model semantics.

        Parameters
        ----------
        syn_spec : Mapping[str, object] or None
            Synapse specification dictionary containing connection-level parameters.
            Common keys include ``'delay'``, ``'receptor_type'``, etc.
            If ``None``, validation is skipped (no parameters to check).

        Raises
        ------
        ValueError
            If ``syn_spec`` contains a ``'weight'`` key, indicating an attempt to
            specify per-connection weights.

        Notes
        -----
        **Allowed per-connection parameters:**

        The following parameters CAN be specified per-connection even with homogeneous
        weights:

        - ``delay`` : Each connection can have a different transmission delay
        - ``receptor_type`` : Each connection can target a different receptor port
        - Other NEST connection parameters (e.g., ``label``, ``synapse_model``)

        **Forbidden per-connection parameters:**

        - ``weight`` : Must be uniform across all connections using this model

        **NEST API correspondence:**

        In NEST, attempting to specify weights in the synapse specification for
        ``static_synapse_hom_w`` connections raises:

        .. code-block:: text

            BadProperty: Weight cannot be specified since it needs to be
            equal for all connections when static_synapse_hom_w is used.

        This method replicates that validation logic.

        **When this method is called:**

        Projection classes (e.g., ``AlignPostProj``, ``DeltaProj``) should call this
        during connection setup:

        .. code-block:: python

            syn_model = static_synapse_hom_w(weight=1.0)
            syn_model.check_synapse_params(user_provided_syn_spec)
            # Proceed with connection creation if no exception raised

        **Design rationale:**

        Early validation prevents runtime errors when users accidentally mix
        homogeneous-weight models with heterogeneous connection specifications,
        providing clear error messages at network construction time.

        See Also
        --------
        set_weight : Per-connection weight setter (also raises error)
        set : Correct method for updating common weight

        Examples
        --------
        **Valid synapse specification (no weight):**

        .. code-block:: python

            >>> import brainpy.state as bs
            >>> syn = bs.static_synapse_hom_w(weight=1.0)
            >>>
            >>> # OK: delay and receptor_type can vary per-connection
            >>> syn.check_synapse_params({
            ...     'delay': 2.0,
            ...     'receptor_type': 1,
            ... })
            >>> # No exception raised

        **Invalid synapse specification (contains weight):**

        .. code-block:: python

            >>> # Error: weight cannot be specified per-connection
            >>> try:
            ...     syn.check_synapse_params({'weight': 2.0})
            ... except ValueError as e:
            ...     print(str(e))
            Weight cannot be specified since it needs to be equal for all connections when static_synapse_hom_w is used.

        **Null specification (skipped validation):**

        .. code-block:: python

            >>> # OK: None means no parameters to validate
            >>> syn.check_synapse_params(None)

        **Usage in projection class:**

        .. code-block:: python

            >>> class MyProjection:
            ...     def __init__(self, syn_model, syn_spec):
            ...         # Validate synapse specification early
            ...         syn_model.check_synapse_params(syn_spec)
            ...         # ... proceed with connection creation ...
            >>>
            >>> proj = MyProjection(
            ...     syn_model=bs.static_synapse_hom_w(weight=1.0),
            ...     syn_spec={'delay': 1.5},  # Valid
            ... )

        **NEST script translation:**

        NEST code:

        .. code-block:: python

            # NEST: This would raise error during Connect()
            nest.Connect(pre, post,
                syn_spec={"synapse_model": "static_synapse_hom_w",
                          "weight": 2.0})  # ERROR

        brainpy.state equivalent:

        .. code-block:: python

            # brainpy.state: Raises error during validation
            syn = bs.static_synapse_hom_w(weight=1.0)
            syn.check_synapse_params({"weight": 2.0})  # Raises ValueError
        """
        if syn_spec is None:
            return
        if 'weight' in syn_spec:
            raise ValueError(
                'Weight cannot be specified since it needs to be equal for all '
                'connections when static_synapse_hom_w is used.'
            )
