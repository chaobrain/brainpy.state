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

from .stdp_synapse import stdp_synapse

__all__ = [
    'stdp_synapse_hom',
]


class stdp_synapse_hom(stdp_synapse):
    r"""NEST-compatible ``stdp_synapse_hom`` connection model with homogeneous plasticity parameters.

    ``stdp_synapse_hom`` implements pair-based spike-timing dependent plasticity (STDP)
    following Guetig et al. (2003) and the NEST reference implementation from
    ``models/stdp_synapse_hom.h``. The model is identical to :class:`stdp_synapse` in
    its plasticity dynamics but enforces that ``tau_plus``, ``lambda``, ``alpha``,
    ``mu_plus``, ``mu_minus``, and ``Wmax`` are **common model properties** shared by
    all connections of this type, rather than per-connection parameters.

    This design mirrors NEST's homogeneous synapse convention, where plasticity
    hyperparameters are set once at the model level (via ``CopyModel``/``SetDefaults``)
    and cannot be overridden on individual connections. Per-connection state remains
    limited to ``weight`` and ``Kplus`` (presynaptic eligibility trace).

    **1. Mathematical Model**

    The STDP dynamics are identical to :class:`stdp_synapse`. See that class for full
    mathematical derivation. In brief:

    **State Variables (per connection):**

    - ``w``: Synaptic weight (plastic, bounded to :math:`[0, W_{\max}]` or :math:`[W_{\max}, 0]`)
    - ``K^+``: Presynaptic eligibility trace (decays with :math:`\tau_+`)

    **Shared Plasticity Parameters (model-level):**

    - :math:`\tau_+` -- Presynaptic trace time constant
    - :math:`\lambda` -- Potentiation learning rate
    - :math:`\alpha` -- Depression/potentiation ratio
    - :math:`\mu_+` -- Potentiation weight-dependence exponent
    - :math:`\mu_-` -- Depression weight-dependence exponent
    - :math:`W_{\max}` -- Maximum allowed weight magnitude

    **Weight Updates:**

    Upon presynaptic spike at time :math:`t_{\text{pre}}` with dendritic delay :math:`d`:

    1. **Facilitation** from past postsynaptic spikes in :math:`(t_{\text{last}} - d,\, t_{\text{pre}} - d]`:

       .. math::

          \hat{w} \leftarrow \hat{w} + \lambda (1 - \hat{w})^{\mu_+} K^+_{\text{eff}}

    2. **Depression** from current postsynaptic trace :math:`K^-(t_{\text{pre}} - d)`:

       .. math::

          \hat{w} \leftarrow \hat{w} - \alpha \lambda \hat{w}^{\mu_-} K^-_{\text{eff}}

    3. **Send** weighted spike event to postsynaptic neuron.

    4. **Update** presynaptic trace:

       .. math::

          K^+ \leftarrow K^+ e^{(t_{\text{last}} - t_{\text{pre}}) / \tau_+} + 1

    where :math:`\hat{w} = w / W_{\max}` is the normalized weight.

    **2. Homogeneous Property Semantics**

    In NEST, ``stdp_synapse_hom`` models enforce that plasticity hyperparameters
    (``tau_plus``, ``lambda``, ``alpha``, ``mu_plus``, ``mu_minus``, ``Wmax``) are
    **common properties** set once at the model level, not per-connection. This
    implementation replicates that constraint by:

    - Accepting these parameters only during model construction (``__init__``) or
      global model updates (``set`` called on the model instance).
    - Rejecting these parameters in connection-time synapse specifications passed
      to :meth:`check_synapse_params` (called by NEST-style ``Connect`` APIs).
    - Storing a single copy of each parameter shared by all connections.

    **Workflow example:**

    .. code-block:: python

       >>> # Set common properties at model construction
       >>> stdp_model = stdp_synapse_hom(
       ...     weight=1.0,
       ...     delay=1.0 * u.ms,
       ...     tau_plus=20.0 * u.ms,
       ...     lambda_=0.01,
       ...     alpha=1.05,
       ...     mu_plus=1.0,
       ...     mu_minus=1.0,
       ...     Wmax=100.0,
       ... )
       >>> # OK: per-connection weight at connect time
       >>> stdp_model.check_synapse_params({'weight': 2.5})
       >>> # ERROR: cannot override common property at connect time
       >>> stdp_model.check_synapse_params({'lambda': 0.02})  # raises ValueError

    **3. Validation Semantics**

    Unlike :class:`stdp_synapse`, NEST ``stdp_synapse_hom`` does **not** enforce:

    - The ``weight``/``Wmax`` sign consistency check (allowing mixed signs).
    - The ``Kplus >= 0`` non-negativity constraint (allowing negative traces).

    This implementation replicates NEST behavior by overriding the validation methods
    to no-ops:

    - :meth:`_validate_non_negative` (disables ``Kplus >= 0`` check)
    - :meth:`_validate_weight_wmax_sign` (disables sign consistency check)

    **4. Event Timing and Ordering**

    Event processing follows the same sequence as :class:`stdp_synapse`:

    1. Query postsynaptic spike history in window :math:`(t_{\text{last}} - d,\, t_{\text{pre}} - d]`
    2. Apply facilitation for each retrieved postsynaptic spike
    3. Compute postsynaptic trace :math:`K^-` at :math:`t_{\text{pre}} - d`
    4. Apply depression based on :math:`K^-`
    5. Schedule weighted spike event for delivery after delay :math:`d`
    6. Update presynaptic trace :math:`K^+` and timestamp ``t_lastspike``

    **Note:** Event timing uses on-grid spike stamps and ignores sub-step offsets.

    **5. Assumptions, Constraints, and Failure Modes**

    **Constraints enforced at construction:**

    - ``tau_plus > 0`` (presynaptic trace time constant must be positive)
    - ``lambda >= 0`` (learning rate must be non-negative)
    - ``alpha >= 0`` (depression/potentiation ratio must be non-negative)
    - ``Wmax != 0`` (maximum weight must be nonzero)
    - ``tau_minus > 0`` (postsynaptic trace time constant must be positive, inherited)

    **Failure modes:**

    - Attempting to set ``tau_plus``, ``lambda``, ``alpha``, ``mu_plus``, ``mu_minus``,
      or ``Wmax`` in connection-time ``syn_spec`` → raises ``ValueError`` from
      :meth:`check_synapse_params`.
    - Setting ``Wmax = 0`` → division by zero in weight normalization.
    - Very small ``tau_plus`` or ``tau_minus`` → numerical instability in exponential decay.
    - Large ``mu_plus`` or ``mu_minus`` → gradient explosion in weight-dependent terms.

    **Computational complexity:**

    - Per presynaptic spike: :math:`O(N_{\text{post}})` where :math:`N_{\text{post}}` is
      the number of postsynaptic spikes in the potentiation window.
    - Per postsynaptic spike: :math:`O(1)` trace update.

    Parameters
    ----------
    weight : float, array-like, or Quantity, optional
        **Per-connection parameter.** Initial synaptic weight. Scalar value,
        dimensionless or with units (pA for current-based, nS for conductance-based).
        Can be positive (excitatory) or negative (inhibitory). Default: ``1.0``.
    delay : float, array-like, or Quantity, optional
        **Per-connection parameter.** Synaptic transmission delay in ms. Must be
        positive, will be discretized to integer time steps. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        **Per-connection parameter.** Receptor port identifier on postsynaptic neuron.
        Non-negative integer. Default: ``0``.
    tau_plus : float or Quantity, optional
        **Common model property.** Presynaptic eligibility trace time constant
        :math:`\tau_+` in ms. Must be strictly positive. Shared by all connections
        of this model type. Default: ``20.0 * u.ms``.
    lambda_ : float, optional
        **Common model property.** Potentiation learning rate :math:`\lambda`
        (dimensionless). Must be non-negative. Shared by all connections.
        Default: ``0.01``.
    alpha : float, optional
        **Common model property.** Depression/potentiation ratio :math:`\alpha`
        (dimensionless). Must be non-negative. Shared by all connections.
        Default: ``1.0``.
    mu_plus : float, optional
        **Common model property.** Potentiation weight-dependence exponent
        :math:`\mu_+` (dimensionless). Controls how potentiation saturates near
        :math:`W_{\max}`. Shared by all connections. Default: ``1.0``.
    mu_minus : float, optional
        **Common model property.** Depression weight-dependence exponent
        :math:`\mu_-` (dimensionless). Controls how depression saturates near zero.
        Shared by all connections. Default: ``1.0``.
    Wmax : float, optional
        **Common model property.** Maximum allowed weight magnitude
        :math:`W_{\max}` (dimensionless or with same units as ``weight``). Weights
        are clipped to :math:`[0, W_{\max}]` or :math:`[W_{\max}, 0]` depending on
        sign. Must be nonzero. Shared by all connections. Default: ``100.0``.
    tau_minus : float or Quantity, optional
        Postsynaptic eligibility trace time constant :math:`\tau_-` in ms. Must be
        strictly positive. In NEST, this belongs to the postsynaptic neuron's
        archiving system; here it is stored on the synapse for standalone use.
        Default: ``20.0 * u.ms``.
    post : Dynamics, optional
        Default postsynaptic receiver object. Must implement spike history archiving
        and ``add_delta_input`` or ``add_current_input`` methods. Default: ``None``.
    event_type : str, optional
        Type of event to transmit. Typically ``'spike'`` for STDP. Default: ``'spike'``.
    name : str, optional
        Unique identifier for this synapse instance. Default: auto-generated.

    Parameter Mapping
    -----------------

    NEST ``stdp_synapse_hom`` parameters map to this implementation as follows:

    ========================  ======================  ===========================================
    NEST Parameter            brainpy.state Param     Scope
    ========================  ======================  ===========================================
    ``weight``                ``weight``              Per-connection (can vary)
    ``delay``                 ``delay``               Per-connection (can vary)
    ``receptor_type``         ``receptor_type``       Per-connection (can vary)
    ``tau_plus``              ``tau_plus``            **Common property** (model-level)
    ``lambda``                ``lambda_``             **Common property** (model-level)
    ``alpha``                 ``alpha``               **Common property** (model-level)
    ``mu_plus``               ``mu_plus``             **Common property** (model-level)
    ``mu_minus``              ``mu_minus``            **Common property** (model-level)
    ``Wmax``                  ``Wmax``                **Common property** (model-level)
    (postsynaptic archiving)  ``tau_minus``           Synapse-level (NEST: neuron property)
    ========================  ======================  ===========================================

    Attributes
    ----------
    weight : float or Quantity
        Current synaptic weight (mutable, per-connection state).
    Kplus : float
        Presynaptic eligibility trace :math:`K^+` (mutable, per-connection state).
    t_lastspike : float
        Timestamp of last presynaptic spike in ms (mutable, per-connection state).
    tau_plus : Quantity
        Presynaptic trace time constant (immutable common property).
    lambda_ : float
        Potentiation learning rate (immutable common property).
    alpha : float
        Depression/potentiation ratio (immutable common property).
    mu_plus : float
        Potentiation exponent (immutable common property).
    mu_minus : float
        Depression exponent (immutable common property).
    Wmax : float
        Maximum weight magnitude (immutable common property).
    tau_minus : Quantity
        Postsynaptic trace time constant (synapse-level parameter).

    Notes
    -----
    **Design differences from NEST:**

    1. **Parameter scope enforcement**: NEST enforces homogeneous-property semantics
       at connection-time via its C++ connection API. This implementation replicates
       that by validating ``syn_spec`` in :meth:`check_synapse_params`.

    2. **Validation relaxation**: NEST ``stdp_synapse_hom`` intentionally omits the
       ``weight``/``Wmax`` sign check and ``Kplus >= 0`` check present in
       ``stdp_synapse``. This class follows that behavior by overriding validation
       methods to no-ops.

    3. **Postsynaptic archiving**: In NEST, ``tau_minus`` and spike history belong
       to the postsynaptic neuron. This implementation stores ``tau_minus`` on the
       synapse for standalone compatibility, avoiding tight coupling to neuron
       archiving APIs.

    4. **Event timing**: NEST uses precise spike timing with sub-step offsets. This
       implementation uses on-grid timestamps (ignoring offsets) for simplicity.

    **Typical usage patterns:**

    - **Large-scale STDP networks**: Define shared plasticity rules with heterogeneous
      initial weights and delays per connection.
    - **Parameter space exploration**: Systematically vary common properties across
      model instances to study learning dynamics.
    - **Memory-efficient plasticity**: Share hyperparameters across millions of
      connections to reduce memory footprint.

    See Also
    --------
    stdp_synapse : Heterogeneous STDP variant (per-connection plasticity parameters)
    stdp_triplet_synapse : Triplet-based STDP (better fit to experimental data)
    stdp_dopamine_synapse : Reward-modulated STDP
    vogels_sprekeler_synapse : Inhibitory STDP for E/I balance

    References
    ----------
    .. [1] NEST source: ``models/stdp_synapse_hom.h`` and
           ``models/stdp_synapse_hom.cpp``.
    .. [2] Guetig R, Aharonov R, Rotter S, Sompolinsky H (2003).
           Learning input correlations through nonlinear temporally asymmetric
           Hebbian plasticity. *Journal of Neuroscience*, 23(9):3697-3714.
           https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003

    Examples
    --------
    **Basic STDP learning with homogeneous plasticity parameters:**

    .. code-block:: python

       >>> import brainstate as bst
       >>> import saiunit as u
       >>> from brainpy_state._nest import stdp_synapse_hom
       >>> # Create STDP model with common properties
       >>> stdp = stdp_synapse_hom(
       ...     weight=5.0,
       ...     delay=1.0 * u.ms,
       ...     tau_plus=20.0 * u.ms,
       ...     tau_minus=20.0 * u.ms,
       ...     lambda_=0.01,
       ...     alpha=1.05,
       ...     mu_plus=1.0,
       ...     mu_minus=1.0,
       ...     Wmax=100.0,
       ... )
       >>> # Inspect model properties
       >>> params = stdp.get()
       >>> params['synapse_model']
       'stdp_synapse_hom'
       >>> params['tau_plus']
       20.0 * ms

    **Verify common property enforcement:**

    .. code-block:: python

       >>> # OK: per-connection parameters
       >>> stdp.check_synapse_params({'weight': 10.0, 'delay': 2.0 * u.ms})
       >>> # ERROR: common properties cannot be set per-connection
       >>> try:
       ...     stdp.check_synapse_params({'lambda': 0.02})
       ... except ValueError as e:
       ...     print(e)
       lambda cannot be specified in connect-time synapse parameters for stdp_synapse_hom; set common properties on the model itself (for example via CopyModel()/SetDefaults()).

    **Use in network simulation:**

    .. code-block:: python

       >>> # Define pre/post neuron populations
       >>> from brainpy_state._nest import iaf_psc_exp
       >>> pre_neurons = iaf_psc_exp(in_size=100)
       >>> post_neurons = iaf_psc_exp(in_size=50)
       >>> # Create STDP connection with shared plasticity rules
       >>> stdp_conn = stdp_synapse_hom(
       ...     tau_plus=16.8 * u.ms,
       ...     tau_minus=33.7 * u.ms,
       ...     lambda_=0.005,
       ...     alpha=1.05,
       ...     Wmax=50.0,
       ...     post=post_neurons,
       ... )
       >>> # Connect with heterogeneous weights/delays
       >>> for i in range(100):
       ...     for j in range(50):
       ...         stdp_conn.check_synapse_params({
       ...             'weight': np.random.uniform(0, 10),
       ...             'delay': np.random.uniform(1.0, 5.0) * u.ms,
       ...         })  # Per-connection parameters OK
    """

    __module__ = 'brainpy.state'

    @staticmethod
    def _validate_non_negative(value: float, *, name: str):
        r"""Override parent validation to disable ``Kplus >= 0`` constraint.

        NEST ``stdp_synapse_hom::set_status`` does not enforce non-negativity
        constraints on the presynaptic eligibility trace ``Kplus``, allowing
        negative trace values. This method replicates that behavior by disabling
        the validation check from :class:`stdp_synapse`.

        Parameters
        ----------
        value : float
            Value to validate (ignored, no validation performed).
        name : str
            Parameter name (ignored, no validation performed).

        Notes
        -----
        This is a deliberate no-op override. The parent class :class:`stdp_synapse`
        enforces ``Kplus >= 0``, but NEST ``stdp_synapse_hom`` does not. See NEST
        source ``models/stdp_synapse_hom.cpp::set_status`` for reference.
        """
        del value, name

    @classmethod
    def _validate_weight_wmax_sign(cls, weight: float, Wmax: float):
        r"""Override parent validation to disable ``weight``/``Wmax`` sign check.

        NEST ``stdp_synapse_hom::set_status`` does not enforce sign consistency
        between ``weight`` and ``Wmax``, allowing mixed positive/negative values.
        This method replicates that behavior by disabling the validation check
        from :class:`stdp_synapse`.

        Parameters
        ----------
        weight : float
            Synaptic weight value (ignored, no validation performed).
        Wmax : float
            Maximum weight magnitude (ignored, no validation performed).

        Notes
        -----
        This is a deliberate no-op override. The parent class :class:`stdp_synapse`
        enforces ``sign(weight) == sign(Wmax)``, but NEST ``stdp_synapse_hom`` does
        not. See NEST source ``models/stdp_synapse_hom.cpp::set_status`` for reference.

        **Rationale**: Relaxing the sign constraint allows more flexible weight
        initialization and mixed excitatory/inhibitory dynamics, at the cost of
        potentially counter-intuitive clipping behavior if signs mismatch.
        """
        del cls, weight, Wmax

    def get(self) -> dict:
        r"""Return current public parameters and mutable connection state.

        Retrieves all model parameters, common properties, and per-connection state
        variables as a dictionary. The returned dictionary includes both model-level
        shared parameters (``tau_plus``, ``lambda``, ``alpha``, ``mu_plus``,
        ``mu_minus``, ``Wmax``) and per-connection mutable state (``weight``,
        ``Kplus``, ``t_lastspike``). The ``synapse_model`` field identifies this
        as a ``'stdp_synapse_hom'`` connection.

        Returns
        -------
        dict
            Dictionary with the following structure:

            - ``'synapse_model'`` (str): Always ``'stdp_synapse_hom'``.
            - ``'weight'`` (float or Quantity): Current synaptic weight.
            - ``'delay'`` (float): Transmission delay in ms (quantized to steps).
            - ``'receptor_type'`` (int): Receptor port identifier.
            - ``'tau_plus'`` (Quantity): Presynaptic trace time constant (common property).
            - ``'lambda'`` (float): Potentiation learning rate (common property).
            - ``'alpha'`` (float): Depression/potentiation ratio (common property).
            - ``'mu_plus'`` (float): Potentiation exponent (common property).
            - ``'mu_minus'`` (float): Depression exponent (common property).
            - ``'Wmax'`` (float): Maximum weight magnitude (common property).
            - ``'tau_minus'`` (Quantity): Postsynaptic trace time constant.
            - ``'Kplus'`` (float): Presynaptic eligibility trace (mutable state).
            - ``'t_lastspike'`` (float): Last presynaptic spike timestamp in ms (mutable state).
            - Additional fields inherited from :class:`stdp_synapse` and :class:`static_synapse`.

        Notes
        -----
        The returned dictionary is a snapshot of the current state. Modifying the
        returned dictionary does not affect the synapse internal state. To update
        parameters, use :meth:`set` instead.

        **Common property values** (``tau_plus``, ``lambda``, ``alpha``, ``mu_plus``,
        ``mu_minus``, ``Wmax``) are shared by all connections of this model type.
        They cannot be modified per-connection; attempting to pass them in
        connection-time ``syn_spec`` will raise ``ValueError`` from
        :meth:`check_synapse_params`.

        Examples
        --------
        .. code-block:: python

           >>> import saiunit as u
           >>> from brainpy_state._nest import stdp_synapse_hom
           >>> stdp = stdp_synapse_hom(
           ...     weight=5.0,
           ...     tau_plus=20.0 * u.ms,
           ...     lambda_=0.01,
           ...     Wmax=100.0,
           ... )
           >>> params = stdp.get()
           >>> params['synapse_model']
           'stdp_synapse_hom'
           >>> params['tau_plus']
           20.0 * ms
           >>> params['lambda']
           0.01
        """
        params = super().get()
        params['synapse_model'] = 'stdp_synapse_hom'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        r"""Validate connection-time synapse parameters and reject common properties.

        Enforces that common model properties (``tau_plus``, ``lambda``, ``alpha``,
        ``mu_plus``, ``mu_minus``, ``Wmax``) cannot be specified in per-connection
        synapse specifications. This replicates NEST ``stdp_synapse_hom`` semantics,
        where plasticity hyperparameters are set once at the model level and shared
        by all connections.

        Per-connection parameters (``weight``, ``delay``, ``receptor_type``) are
        allowed and should be specified in ``syn_spec`` when creating individual
        connections.

        Parameters
        ----------
        syn_spec : Mapping[str, object] or None
            Connection-time synapse specification dictionary. If ``None``, validation
            is skipped (no parameters to check). Keys are parameter names, values are
            the requested per-connection values.

            **Allowed keys**: ``'weight'``, ``'delay'``, ``'receptor_type'``, and
            any other per-connection state variables.

            **Forbidden keys**: ``'tau_plus'``, ``'lambda'``, ``'alpha'``, ``'mu_plus'``,
            ``'mu_minus'``, ``'Wmax'`` (common properties, must be set on model).

        Raises
        ------
        ValueError
            If ``syn_spec`` contains any of the forbidden common-property keys
            (``tau_plus``, ``lambda``, ``alpha``, ``mu_plus``, ``mu_minus``, ``Wmax``).
            The error message identifies the disallowed key and suggests setting it
            via model construction or global model update instead.

        Notes
        -----
        **Design rationale**: In NEST, ``stdp_synapse_hom`` models enforce that
        plasticity hyperparameters are **homogeneous** (shared across all connections)
        by preventing their specification in ``Connect()`` synapse dictionaries. This
        constraint is enforced at the C++ API level in NEST. This method replicates
        that behavior in Python by validating the ``syn_spec`` dictionary.

        **Workflow**:

        1. Create model with common properties: ``model = stdp_synapse_hom(tau_plus=20*u.ms, ...)``
        2. Connect with per-connection parameters: ``model.check_synapse_params({'weight': 5.0})``
        3. ERROR if common properties appear: ``model.check_synapse_params({'lambda': 0.02})``

        **Implementation note**: The check uses key name ``'lambda'`` (not ``'lambda_'``)
        to match NEST naming conventions. Users should use ``lambda_`` in Python
        code but ``'lambda'`` in synapse specification dictionaries.

        Examples
        --------
        .. code-block:: python

           >>> import saiunit as u
           >>> from brainpy_state._nest import stdp_synapse_hom
           >>> stdp = stdp_synapse_hom(
           ...     weight=1.0,
           ...     tau_plus=20.0 * u.ms,
           ...     lambda_=0.01,
           ...     Wmax=100.0,
           ... )
           >>> # OK: per-connection parameters
           >>> stdp.check_synapse_params({'weight': 5.0, 'delay': 2.0 * u.ms})
           >>> # OK: None syn_spec (no parameters to validate)
           >>> stdp.check_synapse_params(None)
           >>> # ERROR: common property
           >>> try:
           ...     stdp.check_synapse_params({'lambda': 0.02})
           ... except ValueError as e:
           ...     print(e)
           lambda cannot be specified in connect-time synapse parameters for stdp_synapse_hom; set common properties on the model itself (for example via CopyModel()/SetDefaults()).

        See Also
        --------
        get : Retrieve current model parameters and state
        set : Update model parameters (for common properties, call on model instance)
        """
        if syn_spec is None:
            return
        disallowed = ('tau_plus', 'lambda', 'alpha', 'mu_plus', 'mu_minus', 'Wmax')
        for key in disallowed:
            if key in syn_spec:
                raise ValueError(
                    f'{key} cannot be specified in connect-time synapse parameters '
                    'for stdp_synapse_hom; set common properties on the model '
                    'itself (for example via CopyModel()/SetDefaults()).'
                )
