from typing import Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

from ._base import NESTSynapse

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

__all__ = [
    'diffusion_connection',
]


class diffusion_connection(NESTSynapse):
    r"""NEST-compatible ``diffusion_connection`` connection model.

    ``diffusion_connection`` implements instantaneous diffusion-based coupling
    between rate-based neuron populations, specifically designed for use with
    ``siegert_neuron`` mean-field models. Unlike standard synaptic connections
    that transmit weighted spikes or rates, this connection simultaneously affects
    both the drift (mean) and diffusion (variance) components of the target
    neuron's input statistics.

    This model replaces the single synaptic weight with two independent scaling
    factors: ``drift_factor`` controls the contribution to the mean input current
    (first moment), while ``diffusion_factor`` controls the contribution to input
    variance (second moment). This dual-factor design enables accurate representation
    of population-level fluctuations in mean-field models.

    **1. Mathematical Model**

    The connection transforms presynaptic rate activity :math:`r_j(t)` into dual
    postsynaptic inputs:

    .. math::

       \Delta\mu_i &= g_{\mu}\,r_j(t), \\
       \Delta\sigma^2_i &= g_{\sigma}\,r_j(t),

    where:

    - :math:`g_{\mu}` is ``drift_factor`` (dimensionless scaling coefficient)
    - :math:`g_{\sigma}` is ``diffusion_factor`` (dimensionless scaling coefficient)
    - :math:`r_j(t)` is the presynaptic firing rate (Hz)
    - :math:`\Delta\mu_i` is the drift (mean current) contribution to postsynaptic neuron :math:`i`
    - :math:`\Delta\sigma^2_i` is the diffusion (variance) contribution

    The target ``siegert_neuron`` accumulates these contributions from all incoming
    connections to compute its total effective input statistics:

    .. math::

       \mu_{\mathrm{total}} &= \sum_j g_{\mu,j}\,r_j, \\
       \sigma^2_{\mathrm{total}} &= \sum_j g_{\sigma,j}\,r_j.

    **2. Waveform Relaxation (WFR) Event Semantics**

    For networks using waveform relaxation iterative solvers, the connection transmits
    multi-lag coefficient arrays representing interpolated rate values across a time
    window:

    **Event structure:**

    .. math::

       \text{coeffarray} = [r(t_0), r(t_1), \ldots, r(t_{n-1})]

    **Target accumulation (per lag :math:`i`):**

    .. math::

       \mu_i &\leftarrow \mu_i + g_{\mu}\cdot r(t_i), \\
       \sigma^2_i &\leftarrow \sigma^2_i + g_{\sigma}\cdot r(t_i).

    This allows the target neuron to iteratively refine its solution using updated
    presynaptic activity estimates without re-sending events at every substep.

    **3. Design Constraints**

    To maintain consistency with NEST's implementation, this model enforces several
    architectural constraints:

    **No transmission delay:**

    Unlike spike-based synapses, diffusion connections are instantaneous. The absence
    of delay reflects the mean-field assumption that population dynamics operate on
    slower timescales than individual spike transmission.

    **No standard weight parameter:**

    The traditional ``weight`` parameter is intentionally unsupported. Instead, users
    must explicitly specify ``drift_factor`` and ``diffusion_factor`` to clarify the
    distinction between mean and variance contributions. Attempting to set ``weight``
    raises an error with NEST's original message (preserving the typo ``"specifiy"``
    for exact compatibility).

    **Symmetric status API:**

    The ``get_status()`` method returns ``weight: 1.0`` and ``delay: None`` for API
    consistency, but ``set_status()`` rejects attempts to modify these fields.

    **4. Usage Context**

    ``diffusion_connection`` is specialized for mean-field population models:

    **Typical use cases:**

    - Connecting multiple ``siegert_neuron`` populations in a network
    - Representing effective connectivity in population density approaches
    - Modeling input fluctuations from background population activity
    - Implementing approximate mesoscale network dynamics

    **Not suitable for:**

    - Spiking neuron models (use ``static_synapse`` or variants)
    - Conductance-based synapses (no reversal potential mechanism)
    - Plastic connections (no learning rule support)

    Parameters
    ----------
    drift_factor : float, array-like, or Quantity, optional
        Scaling coefficient for presynaptic rate contribution to postsynaptic
        drift (mean current) input. Must be scalar, dimensionless. Positive
        values increase excitatory drive; negative values contribute inhibition.
        Default: ``1.0`` (unscaled transmission).
    diffusion_factor : float, array-like, or Quantity, optional
        Scaling coefficient for presynaptic rate contribution to postsynaptic
        diffusion (variance) input. Must be scalar, dimensionless. Typically
        non-negative to preserve physical interpretation of variance, though
        negative values are permitted for specialized modeling scenarios.
        Default: ``1.0`` (unscaled transmission).
    name : str, optional
        Unique identifier for this connection instance. Used for debugging and
        logging. If ``None``, no name is assigned.
        Default: ``None``.

    Attributes
    ----------
    drift_factor : float
        Current drift scaling factor (read/write via :meth:`set_drift_factor`
        or :meth:`set_status`).
    diffusion_factor : float
        Current diffusion scaling factor (read/write via :meth:`set_diffusion_factor`
        or :meth:`set_status`).
    weight : float
        Read-only pseudo-parameter, always ``1.0``. Present for NEST API parity.
        Attempting to modify via :meth:`set_status` or :meth:`set_weight` raises
        ``ValueError``.
    name : str or None
        Connection instance name.
    SUPPORTS_WFR : bool (class attribute)
        Always ``True``. Indicates waveform relaxation compatibility.
    HAS_DELAY : bool (class attribute)
        Always ``False``. Indicates absence of transmission delay.

    Notes
    -----
    **Design differences from NEST:**

    1. **Type system**: This implementation validates input types via ``_to_float_scalar``,
       rejecting non-scalar values. NEST relies on C++ type system and templating.

    2. **Event handling**: Methods like :meth:`prepare_secondary_event` and
       :meth:`project_coeffarray` provide explicit event construction APIs not
       directly exposed in NEST's C++ interface.

    3. **Unit handling**: Supports ``brainunit.Quantity`` inputs (mantissa extracted
       automatically). NEST is unit-agnostic at the connection level.

    **Error message compatibility:**

    The weight-setting error preserves NEST's original typo: ``"specifiy"`` instead
    of ``"specify"``. This intentional deviation maintains exact string matching for
    compatibility testing and migration scripts.

    **Negative diffusion factors:**

    While physically questionable (variance should be non-negative), negative
    ``diffusion_factor`` values are permitted. This flexibility supports:

    - Variance reduction mechanisms (anti-correlated noise cancellation)
    - Numerical experiments requiring signed contributions
    - Surrogate models with unconventional interpretations

    Users should validate biological plausibility of their parameter choices.

    **Performance considerations:**

    - Lightweight: No state variables, minimal computation overhead
    - Memory: Stores only two scalar factors (drift, diffusion)
    - Thread safety: Not thread-safe; use separate instances per thread

    See Also
    --------
    gap_junction : Electrical coupling connection (voltage-difference driven)
    rate_connection_instantaneous : Generic instantaneous rate connection
    siegert_neuron : Target neuron model for diffusion connections

    References
    ----------
    .. [1] Schwalger, T., Deger, M., & Gerstner, W. (2017). Towards a theory of
           cortical columns: From spiking neurons to interacting neural populations
           of finite size. *PLoS Computational Biology*, 13(4), e1005507.
           https://doi.org/10.1371/journal.pcbi.1005507
    .. [2] Fourcaud, N., & Brunel, N. (2002). Dynamics of the firing probability
           of noisy integrate-and-fire neurons. *Neural Computation*, 14(9), 2057-2110.
           https://doi.org/10.1162/089976602320264015
    .. [3] NEST Initiative (2025). NEST Simulator Documentation: diffusion_connection.
           https://nest-simulator.readthedocs.io/en/stable/models/diffusion_connection.html
    .. [4] NEST source code: ``models/diffusion_connection.h``, ``models/diffusion_connection.cpp``,
           ``models/siegert_neuron.cpp`` (``handle(DiffusionConnectionEvent&)`` method).
           https://github.com/nest/nest-simulator

    Examples
    --------
    **Basic connection between two Siegert neurons:**

    .. code-block:: python

        >>> import brainpy.state as bs
        >>> # Create two mean-field neurons (pseudo-code; siegert_neuron not yet implemented)
        >>> source = bs.siegert_neuron(1, mu=10.0, sigma=5.0)
        >>> target = bs.siegert_neuron(1, mu=5.0, sigma=3.0)
        >>>
        >>> # Create diffusion connection
        >>> conn = bs.diffusion_connection(
        ...     drift_factor=0.8,      # Strong mean current contribution
        ...     diffusion_factor=0.3,  # Moderate variance contribution
        ...     name='excitatory_diffusion',
        ... )
        >>>
        >>> # Query connection parameters
        >>> status = conn.get_status()
        >>> print(f"Drift: {status['drift_factor']}, Diffusion: {status['diffusion_factor']}")
        Drift: 0.8, Diffusion: 0.3

    **Creating a WFR secondary event:**

    .. code-block:: python

        >>> import numpy as np
        >>> # Presynaptic rate trajectory over 5 lags
        >>> rate_coeffs = np.array([10.0, 12.0, 11.5, 10.2, 9.8])
        >>>
        >>> # Prepare event payload
        >>> event = conn.prepare_secondary_event(rate_coeffs)
        >>> print(event)
        {'coeffarray': array([10. , 12. , 11.5, 10.2,  9.8]),
         'drift_factor': 0.8, 'diffusion_factor': 0.3}

    **Projecting coefficients to drift/diffusion inputs:**

    .. code-block:: python

        >>> # Apply connection factors to coefficient array
        >>> drift_input, diffusion_input = conn.project_coeffarray(rate_coeffs)
        >>> print(f"Drift contribution: {drift_input}")
        >>> print(f"Diffusion contribution: {diffusion_input}")
        Drift contribution: [ 8.   9.6  9.2  8.16 7.84]
        Diffusion contribution: [3.  3.6 3.45 3.06 2.94]

    **Creating step-wise delayed events:**

    .. code-block:: python

        >>> # Convert multi-lag coefficients to step-indexed events
        >>> events = conn.coeffarray_to_step_events(
        ...     rate_coeffs,
        ...     first_delay_steps=5,  # Start at delay step 5
        ...     multiplicity=1.0,
        ... )
        >>> for i, evt in enumerate(events):
        ...     print(f"Lag {i}: coeff={evt['coeff']}, delay_steps={evt['delay_steps']}")
        Lag 0: coeff=10.0, delay_steps=5
        Lag 1: coeff=12.0, delay_steps=6
        Lag 2: coeff=11.5, delay_steps=7
        Lag 3: coeff=10.2, delay_steps=8
        Lag 4: coeff=9.8, delay_steps=9

    **Modifying parameters during simulation:**

    .. code-block:: python

        >>> # Update drift factor only
        >>> conn.set_drift_factor(1.2)
        >>> assert conn.drift_factor == 1.2
        >>>
        >>> # Update both factors via set_status
        >>> conn.set_status(drift_factor=0.5, diffusion_factor=0.1)
        >>> status = conn.get_status()
        >>> assert status['drift_factor'] == 0.5
        >>> assert status['diffusion_factor'] == 0.1

    **Attempting to set weight (error demonstration):**

    .. code-block:: python

        >>> try:
        ...     conn.set_weight(2.0)
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Please use the parameters drift_factor and diffusion_factor to specifiy the weights.

    **Attempting to set delay (error demonstration):**

    .. code-block:: python

        >>> try:
        ...     conn.set_status(delay=1.0)
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: diffusion_connection has no delay.

    **Heterogeneous population connectivity:**

    .. code-block:: python

        >>> # Multiple connections with different factor profiles
        >>> excitatory_conn = bs.diffusion_connection(
        ...     drift_factor=1.5,    # Strong excitatory drive
        ...     diffusion_factor=0.5, # Moderate noise
        ... )
        >>> inhibitory_conn = bs.diffusion_connection(
        ...     drift_factor=-1.0,   # Inhibitory drive (negative)
        ...     diffusion_factor=0.2, # Low noise
        ... )
        >>> background_conn = bs.diffusion_connection(
        ...     drift_factor=0.1,    # Weak drive
        ...     diffusion_factor=2.0, # Strong noise (background fluctuations)
        ... )

    **Single-step event creation for custom integration:**

    .. code-block:: python

        >>> # Create a single-step event for custom handling
        >>> single_event = conn.to_siegert_event(
        ...     coeff=15.0,          # Rate value
        ...     delay_steps=10,      # Delivery at step 10
        ...     multiplicity=1.0,    # Single event
        ... )
        >>> print(single_event)
        {'coeff': 15.0, 'drift_factor': 0.5, 'diffusion_factor': 0.1,
         'delay_steps': 10, 'multiplicity': 1.0}

    **Inspecting connection properties:**

    .. code-block:: python

        >>> props = conn.properties
        >>> print(f"Supports WFR: {props['supports_wfr']}")
        >>> print(f"Has delay: {props['has_delay']}")
        Supports WFR: True
        Has delay: False

    **Using get() method for flexible parameter access:**

    .. code-block:: python

        >>> # Get full status dictionary
        >>> full_status = conn.get('status')
        >>>
        >>> # Get specific parameter
        >>> drift = conn.get('drift_factor')
        >>> print(f"Drift factor: {drift}")
        Drift factor: 0.5
        >>>
        >>> # Attempting unsupported key raises error
        >>> try:
        ...     conn.get('unsupported_key')
        ... except KeyError as e:
        ...     print(f"Error: {e}")
        Error: 'Unsupported key "unsupported_key" for diffusion_connection.get().'
    """

    __module__ = 'brainpy.state'

    SUPPORTS_WFR = True
    HAS_DELAY = False

    _WEIGHT_ERROR = (
        'Please use the parameters drift_factor and diffusion_factor to specifiy the weights.'
    )
    _DELAY_ERROR = 'diffusion_connection has no delay.'

    def __init__(
        self,
        drift_factor: ArrayLike = 1.0,
        diffusion_factor: ArrayLike = 1.0,
        name: str | None = None,
    ):
        self.name = name
        # Keep a status ``weight`` field for parity with NEST model status.
        self.weight = 1.0
        self.drift_factor = self._to_float_scalar(drift_factor, name='drift_factor')
        self.diffusion_factor = self._to_float_scalar(diffusion_factor, name='diffusion_factor')

    @property
    def properties(self) -> dict[str, Any]:
        return {
            'supports_wfr': self.SUPPORTS_WFR,
            'has_delay': self.HAS_DELAY,
        }

    def get_status(self) -> dict[str, Any]:
        r"""Retrieve current connection parameters (NEST ``GetStatus`` equivalent).

        Returns a dictionary of all connection parameters, including pseudo-parameters
        ``weight`` and ``delay`` for NEST API compatibility.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``'weight'`` : float — Always ``1.0`` (read-only pseudo-parameter)
            - ``'delay'`` : None — Always ``None`` (diffusion connections have no delay)
            - ``'drift_factor'`` : float — Current drift scaling factor
            - ``'diffusion_factor'`` : float — Current diffusion scaling factor
            - ``'supports_wfr'`` : bool — Always ``True`` (WFR compatibility)
            - ``'has_delay'`` : bool — Always ``False`` (instantaneous connection)

        Notes
        -----
        **NEST API compatibility:**

        The returned dictionary structure matches NEST's ``GetStatus`` output format.
        The ``weight`` and ``delay`` keys are present for API parity but represent
        immutable pseudo-parameters.

        **Implementation detail:**

        All returned values are cast to native Python types (``float``, ``bool``,
        ``None``) rather than JAX/NumPy arrays, ensuring JSON serializability and
        consistent behavior across different numerical backends.

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection(drift_factor=0.8, diffusion_factor=0.3)
            >>> status = conn.get_status()
            >>> print(status)
            {'weight': 1.0, 'delay': None, 'drift_factor': 0.8,
             'diffusion_factor': 0.3, 'supports_wfr': True, 'has_delay': False}
        """
        return {
            'weight': float(self.weight),
            'delay': None,
            'drift_factor': float(self.drift_factor),
            'diffusion_factor': float(self.diffusion_factor),
            'supports_wfr': self.SUPPORTS_WFR,
            'has_delay': self.HAS_DELAY,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update connection parameters (NEST ``SetStatus`` equivalent).

        Modifies ``drift_factor`` and/or ``diffusion_factor`` while rejecting
        attempts to set unsupported parameters (``weight``, ``delay``). Accepts
        parameters via dictionary argument or keyword arguments.

        Parameters
        ----------
        status : dict[str, Any], optional
            Dictionary of parameter updates. Valid keys: ``'drift_factor'``,
            ``'diffusion_factor'``. Invalid keys ``'weight'`` and ``'delay'``
            raise ``ValueError``. If ``None``, only keyword arguments are applied.
            Default: ``None``.
        **kwargs
            Additional parameter updates via keyword arguments. Overrides values
            in ``status`` dictionary for duplicate keys. Same validation rules apply.

        Raises
        ------
        ValueError
            If ``'delay'`` key is present (message: ``"diffusion_connection has no delay"``).
        ValueError
            dftype = brainstate.environ.dftype()
            If ``'weight'`` key is present (message: ``"Please use the parameters
            drift_factor and diffusion_factor to specifiy the weights."``).
            Note: NEST's original typo ``"specifiy"`` is preserved.
        ValueError
            If ``drift_factor`` or ``diffusion_factor`` values are not scalar.

        Notes
        -----
        **Parameter merging:**

        When both ``status`` dict and ``**kwargs`` are provided:

        .. code-block:: python

            updates = {}
            updates.update(status)      # Apply dictionary parameters
            updates.update(kwargs)      # Keyword arguments override dictionary

        **Validation order:**

        1. Check for ``'delay'`` (immediate error)
        2. Check for ``'weight'`` (immediate error)
        3. Apply ``'drift_factor'`` via :meth:`set_drift_factor`
        4. Apply ``'diffusion_factor'`` via :meth:`set_diffusion_factor`

        Validation errors in steps 3-4 prevent partial updates (atomic failure).

        **NEST compatibility:**

        The error messages exactly match NEST's ``BadProperty`` exception strings,
        including the intentional typo in the weight error message.

        Examples
        --------
        **Update via dictionary:**

        .. code-block:: python

            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection(drift_factor=1.0, diffusion_factor=1.0)
            >>> conn.set_status({'drift_factor': 0.8, 'diffusion_factor': 0.3})
            >>> assert conn.drift_factor == 0.8
            >>> assert conn.diffusion_factor == 0.3

        **Update via keyword arguments:**

        .. code-block:: python

            >>> conn.set_status(drift_factor=1.2, diffusion_factor=0.5)
            >>> assert conn.drift_factor == 1.2

        **Keyword arguments override dictionary:**

        .. code-block:: python

            >>> conn.set_status(
            ...     {'drift_factor': 0.5},  # Dictionary value
            ...     drift_factor=1.0,       # Keyword overrides to 1.0
            ... )
            >>> assert conn.drift_factor == 1.0

        **Update single parameter:**

        .. code-block:: python

            >>> conn.set_status(drift_factor=2.0)  # Only drift changes
            >>> # diffusion_factor remains unchanged

        **Attempting to set delay (error):**

        .. code-block:: python

            >>> try:
            ...     conn.set_status(delay=1.0)
            ... except ValueError as e:
            ...     print(e)
            diffusion_connection has no delay

        **Attempting to set weight (error):**

        .. code-block:: python

            >>> try:
            ...     conn.set_status(weight=2.0)
            ... except ValueError as e:
            ...     print(e)
            Please use the parameters drift_factor and diffusion_factor to specifiy the weights.
        """
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        if 'delay' in updates:
            raise ValueError(self._DELAY_ERROR)
        if 'weight' in updates:
            raise ValueError(self._WEIGHT_ERROR)
        if 'drift_factor' in updates:
            self.set_drift_factor(updates['drift_factor'])
        if 'diffusion_factor' in updates:
            self.set_diffusion_factor(updates['diffusion_factor'])

    def get(self, key: str = 'status'):
        r"""Retrieve connection parameter(s) with flexible key-based access.

        Provides unified access to connection status via string keys. Can return
        the full status dictionary or individual parameter values.

        Parameters
        ----------
        key : str, optional
            Parameter key to retrieve. Valid keys: ``'status'`` (returns full
            dictionary), ``'weight'``, ``'delay'``, ``'drift_factor'``,
            ``'diffusion_factor'``, ``'supports_wfr'``, ``'has_delay'``.
            Default: ``'status'`` (returns full status dictionary).

        Returns
        -------
        dict or scalar
            If ``key='status'``: full status dictionary from :meth:`get_status`.
            If key matches a status field: the corresponding scalar value.

        Raises
        ------
        KeyError
            If ``key`` is not ``'status'`` and does not match any status dictionary key.

        Notes
        -----
        **Supported keys:**

        ======================  ===========  =========================================
        Key                     Return Type  Description
        ======================  ===========  =========================================
        ``'status'``            dict         Full status dictionary (default)
        ``'weight'``            float        Always ``1.0`` (pseudo-parameter)
        ``'delay'``             None         Always ``None`` (no delay)
        ``'drift_factor'``      float        Current drift scaling factor
        ``'diffusion_factor'``  float        Current diffusion scaling factor
        ``'supports_wfr'``      bool         Always ``True``
        ``'has_delay'``         bool         Always ``False``
        ======================  ===========  =========================================

        **Implementation strategy:**

        The method calls :meth:`get_status` once, then performs dictionary lookup
        for specific keys. This ensures consistency (all values come from a single
        status snapshot) but duplicates computation for single-key queries.

        Examples
        --------
        **Get full status dictionary:**

        .. code-block:: python

            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection(drift_factor=0.8, diffusion_factor=0.3)
            >>> status = conn.get('status')
            >>> print(status)
            {'weight': 1.0, 'delay': None, 'drift_factor': 0.8, ...}

        **Get specific parameter:**

        .. code-block:: python

            >>> drift = conn.get('drift_factor')
            >>> print(drift)
            0.8

        **Get multiple parameters:**

        .. code-block:: python

            >>> drift = conn.get('drift_factor')
            >>> diffusion = conn.get('diffusion_factor')
            >>> print(f"Factors: drift={drift}, diffusion={diffusion}")
            Factors: drift=0.8, diffusion=0.3

        **Default behavior (omit key argument):**

        .. code-block:: python

            >>> # Equivalent to conn.get('status')
            >>> status = conn.get()
            >>> print(type(status))
            <class 'dict'>

        **Invalid key error:**

        .. code-block:: python

            >>> try:
            ...     conn.get('invalid_key')
            ... except KeyError as e:
            ...     print(e)
            'Unsupported key "invalid_key" for diffusion_connection.get().'
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for diffusion_connection.get().')

    def set_drift_factor(self, drift_factor: ArrayLike):
        r"""Update drift scaling factor.

        Convenience method to modify ``drift_factor`` independently of other
        parameters. Validates and converts input to scalar float.

        Parameters
        ----------
        drift_factor : float, array-like, or Quantity
            New drift scaling factor. Must be scalar. If ``brainunit.Quantity``,
            mantissa is extracted (assumed dimensionless).

        Raises
        ------
        ValueError
            If ``drift_factor`` is not scalar (size != 1).

        See Also
        --------
        set_status : General parameter update method
        set_diffusion_factor : Update diffusion factor

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection(drift_factor=1.0)
            >>> conn.set_drift_factor(0.5)
            >>> assert conn.drift_factor == 0.5
        """
        self.drift_factor = self._to_float_scalar(drift_factor, name='drift_factor')

    def set_diffusion_factor(self, diffusion_factor: ArrayLike):
        r"""Update diffusion scaling factor.

        Convenience method to modify ``diffusion_factor`` independently of other
        parameters. Validates and converts input to scalar float.

        Parameters
        ----------
        diffusion_factor : float, array-like, or Quantity
            New diffusion scaling factor. Must be scalar. If ``brainunit.Quantity``,
            mantissa is extracted (assumed dimensionless).

        Raises
        ------
        ValueError
            If ``diffusion_factor`` is not scalar (size != 1).

        See Also
        --------
        set_status : General parameter update method
        set_drift_factor : Update drift factor

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection(diffusion_factor=1.0)
            >>> conn.set_diffusion_factor(0.3)
            >>> assert conn.diffusion_factor == 0.3
        """
        self.diffusion_factor = self._to_float_scalar(diffusion_factor, name='diffusion_factor')

    def set_weight(self, _):
        r"""Reject attempts to set weight (NEST compatibility stub).

        ``diffusion_connection`` does not support the standard ``weight`` parameter.
        This method exists solely for NEST API compatibility and always raises an error.

        Parameters
        ----------
        _ : Any
            Ignored. Any value triggers error.

        Raises
        ------
        ValueError
            Always raised with message: ``"Please use the parameters drift_factor and
            diffusion_factor to specifiy the weights."``
            Note: NEST's original typo ``"specifiy"`` is preserved for exact compatibility.

        Notes
        -----
        **Why this exists:**

        NEST's connection API allows querying ``weight`` via ``GetStatus`` but forbids
        setting it via ``SetStatus`` for certain connection types. This asymmetric
        design prevents accidental misuse while maintaining uniform status dictionaries.

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection()
            >>> try:
            ...     conn.set_weight(2.0)
            ... except ValueError as e:
            ...     print(e)
            Please use the parameters drift_factor and diffusion_factor to specifiy the weights.
        """
        raise ValueError(self._WEIGHT_ERROR)

    def set_delay(self, _):
        r"""Reject attempts to set delay (NEST compatibility stub).

        ``diffusion_connection`` is instantaneous and does not support transmission
        delay. This method exists solely for NEST API compatibility and always raises
        an error.

        Parameters
        ----------
        _ : Any
            Ignored. Any value triggers error.

        Raises
        ------
        ValueError
            Always raised with message: ``"diffusion_connection has no delay"``.

        Notes
        -----
        **Design rationale:**

        Diffusion connections represent mean-field population coupling, which operates
        on timescales slower than individual spike transmission. The absence of delay
        reflects this theoretical abstraction.

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection()
            >>> try:
            ...     conn.set_delay(1.0)
            ... except ValueError as e:
            ...     print(e)
            diffusion_connection has no delay
        """
        raise ValueError(self._DELAY_ERROR)

    def prepare_secondary_event(self, coeffarray: ArrayLike) -> dict[str, Any]:
        r"""Construct a WFR secondary event payload for transmission.

        Packages presynaptic rate coefficients with connection scaling factors into
        a dictionary suitable for delivery to target ``siegert_neuron`` instances
        during waveform relaxation iterations.

        Parameters
        ----------
        coeffarray : array-like or Quantity
            Presynaptic rate coefficient array, typically representing interpolated
            firing rates across multiple time lags. Shape: ``(n_lags,)`` where
            ``n_lags`` is the number of WFR substeps per iteration window. If
            ``brainunit.Quantity``, mantissa is extracted (assumed Hz or dimensionless).
            Must be non-empty 1D array after conversion.

        Returns
        -------
        dict[str, Any]
            Event payload dictionary with keys:

            - ``'coeffarray'`` : np.ndarray — Validated coefficient array (float64)
            - ``'drift_factor'`` : float — Connection's drift scaling factor
            - ``'diffusion_factor'`` : float — Connection's diffusion scaling factor

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty after conversion and reshaping.

        Notes
        -----
        **Event structure rationale:**

        The returned dictionary contains all information needed by the target neuron
        to accumulate multi-lag inputs without re-querying the connection object:

        .. math::

           \mu_i \leftarrow \mu_i + \text{drift\_factor} \cdot \text{coeffarray}[i], \\
           \sigma^2_i \leftarrow \sigma^2_i + \text{diffusion\_factor} \cdot \text{coeffarray}[i].

        **Conversion and validation:**

        1. Extract mantissa if ``brainunit.Quantity``
        2. Convert to NumPy ``float64`` array via ``u.math.asarray``
        3. Flatten to 1D (``reshape(-1)``)
        4. Validate non-empty (``size > 0``)

        **Usage in simulation loop:**

        Typically called by presynaptic neuron during WFR event emission:

        .. code-block:: python

            # Presynaptic neuron computes interpolation coefficients
            rate_coeffs = presynaptic.get_wfr_coefficients()

            # Connection packages them into event
            event = connection.prepare_secondary_event(rate_coeffs)

            # Deliver to all postsynaptic targets
            for target in postsynaptic_neurons:
                target.handle_diffusion_event(event)

        Examples
        --------
        **Basic event preparation:**

        .. code-block:: python

            >>> import numpy as np
            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection(drift_factor=0.8, diffusion_factor=0.3)
            >>> coeffs = np.array([10.0, 12.0, 11.5, 10.2, 9.8])
            >>> event = conn.prepare_secondary_event(coeffs)
            >>> print(event['coeffarray'])
            [10.  12.  11.5 10.2  9.8]
            >>> print(event['drift_factor'])
            0.8

        **Event with brainunit Quantity:**

        .. code-block:: python

            >>> import brainunit as u
            >>> rate_Hz = np.array([20.0, 25.0, 22.5]) * u.Hz
            >>> event = conn.prepare_secondary_event(rate_Hz)
            >>> # Mantissa extracted, units stripped
            >>> print(event['coeffarray'])
            [20.  25.  22.5]

        **Empty coeffarray error:**

        .. code-block:: python

            >>> try:
            ...     conn.prepare_secondary_event(np.array([]))
            ... except ValueError as e:
            ...     print(e)
            Coefficient array must not be empty.
        """
        coeff_np = self._to_coeff_array(coeffarray)
        return {
            'coeffarray': coeff_np,
            'drift_factor': float(self.drift_factor),
            'diffusion_factor': float(self.diffusion_factor),
        }

    def project_coeffarray(self, coeffarray: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        r"""Apply connection scaling factors to coefficient array.

        Transforms presynaptic rate coefficients into separate drift and diffusion
        input contributions by element-wise multiplication with ``drift_factor`` and
        ``diffusion_factor``.

        Parameters
        ----------
        coeffarray : array-like or Quantity
            Presynaptic rate coefficient array. Shape: ``(n_lags,)`` where ``n_lags``
            is the number of time substeps in the WFR window. If ``brainunit.Quantity``,
            mantissa is extracted. Must be non-empty 1D array after conversion.

        Returns
        -------
        drift_contribution : np.ndarray
            Drift (mean current) input per lag, computed as
            ``drift_factor * coeffarray``. Shape: ``(n_lags,)``, dtype: ``float64``.
        diffusion_contribution : np.ndarray
            Diffusion (variance) input per lag, computed as
            ``diffusion_factor * coeffarray``. Shape: ``(n_lags,)``, dtype: ``float64``.

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty after conversion.

        Notes
        -----
        **Mathematical operation:**

        For each lag index :math:`i`:

        .. math::

           \text{drift}_i &= g_{\mu} \cdot r_i, \\
           \text{diffusion}_i &= g_{\sigma} \cdot r_i,

        where :math:`r_i` is ``coeffarray[i]``, :math:`g_{\mu}` is ``drift_factor``,
        and :math:`g_{\sigma}` is ``diffusion_factor``.

        **Return value interpretation:**

        The returned arrays represent **additive contributions** to the target neuron's
        input statistics. The target accumulates these across all incoming connections:

        .. math::

           \mu_{\mathrm{total},i} &= \sum_j \text{drift}_{j,i}, \\
           \sigma^2_{\mathrm{total},i} &= \sum_j \text{diffusion}_{j,i}.

        **Relationship to** :meth:`prepare_secondary_event`:

        This method performs the projection computation explicitly, while
        :meth:`prepare_secondary_event` packages the raw factors for target-side
        computation. Use this method when pre-computing projected inputs on the
        sender side; use :meth:`prepare_secondary_event` for standard event transmission.

        Examples
        --------
        **Basic projection:**

        .. code-block:: python

            >>> import numpy as np
            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection(drift_factor=0.8, diffusion_factor=0.3)
            >>> coeffs = np.array([10.0, 12.0, 11.5, 10.2, 9.8])
            >>> drift, diffusion = conn.project_coeffarray(coeffs)
            >>> print("Drift contributions:", drift)
            Drift contributions: [ 8.   9.6  9.2  8.16 7.84]
            >>> print("Diffusion contributions:", diffusion)
            Diffusion contributions: [3.   3.6  3.45 3.06 2.94]

        **Accumulation across multiple connections:**

        .. code-block:: python

            >>> # Target neuron receives from multiple sources
            >>> conn1 = bs.diffusion_connection(drift_factor=0.8, diffusion_factor=0.2)
            >>> conn2 = bs.diffusion_connection(drift_factor=0.5, diffusion_factor=0.4)
            >>>
            >>> coeffs1 = np.array([10.0, 11.0, 12.0])
            >>> coeffs2 = np.array([5.0, 6.0, 7.0])
            >>>
            >>> drift1, diff1 = conn1.project_coeffarray(coeffs1)
            >>> drift2, diff2 = conn2.project_coeffarray(coeffs2)
            >>>
            >>> total_drift = drift1 + drift2
            >>> total_diffusion = diff1 + diff2
            >>> print("Total drift per lag:", total_drift)
            Total drift per lag: [10.5 11.8 13.1]

        **Identity transformation (unit factors):**

        .. code-block:: python

            >>> conn_identity = bs.diffusion_connection(drift_factor=1.0, diffusion_factor=1.0)
            >>> coeffs = np.array([5.0, 10.0, 15.0])
            >>> drift, diffusion = conn_identity.project_coeffarray(coeffs)
            >>> # Both outputs equal input
            >>> assert np.allclose(drift, coeffs)
            >>> assert np.allclose(diffusion, coeffs)

        **Negative drift (inhibitory effect):**

        .. code-block:: python

            >>> conn_inhibitory = bs.diffusion_connection(drift_factor=-0.5, diffusion_factor=0.1)
            >>> coeffs = np.array([10.0, 10.0, 10.0])
            >>> drift, diffusion = conn_inhibitory.project_coeffarray(coeffs)
            >>> print("Drift (inhibitory):", drift)
            Drift (inhibitory): [-5. -5. -5.]
            >>> print("Diffusion (positive):", diffusion)
            Diffusion (positive): [1. 1. 1.]
        """
        coeff_np = self._to_coeff_array(coeffarray)
        return self.drift_factor * coeff_np, self.diffusion_factor * coeff_np

    def to_siegert_event(
        self,
        coeff: ArrayLike,
        delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
    ) -> dict[str, Any]:
        r"""Create a single-step event payload for custom ``siegert_neuron`` handling.

        Constructs a minimal event dictionary representing a single rate value
        scheduled for delivery at a specific future time step with optional
        multiplicity scaling. Useful for custom integration loops or non-standard
        event scheduling.

        Parameters
        ----------
        coeff : float, array-like, or Quantity
            Single rate coefficient value (typically Hz or dimensionless). Must be
            scalar. If ``brainunit.Quantity``, mantissa is extracted.
        delay_steps : int, array-like, or Quantity, optional
            Number of simulation time steps until event delivery. Must be scalar
            non-negative integer. Note: Unlike standard ``diffusion_connection``
            semantics, this parameter exists for flexible custom scheduling.
            Default: ``1`` (deliver next step).
        multiplicity : float, array-like, or Quantity, optional
            Event weight multiplier. Must be scalar. Scaled coefficient becomes
            ``coeff * multiplicity``. Used for representing multiple simultaneous
            events or fractional event contributions.
            Default: ``1.0`` (no scaling).

        Returns
        -------
        dict[str, Any]
            Event payload dictionary with keys:

            - ``'coeff'`` : float — Single rate coefficient value
            - ``'drift_factor'`` : float — Connection's drift scaling factor
            - ``'diffusion_factor'`` : float — Connection's diffusion scaling factor
            - ``'delay_steps'`` : int — Delivery time step offset
            - ``'multiplicity'`` : float — Event multiplicity weight

        Raises
        ------
        ValueError
            If ``coeff``, ``delay_steps``, or ``multiplicity`` is not scalar.

        Notes
        -----
        **Relationship to** :meth:`prepare_secondary_event`:

        - :meth:`prepare_secondary_event` : Multi-lag coefficient arrays for WFR
        - :meth:`to_siegert_event` : Single-step events for custom scheduling

        **Delay step semantics:**

        While ``diffusion_connection`` is nominally instantaneous (``HAS_DELAY = False``),
        this method accepts ``delay_steps`` for compatibility with event-driven
        simulation frameworks that require explicit delivery time specification.

        **Multiplicity interpretation:**

        The target neuron typically applies multiplicity as:

        .. math::

           \Delta\mu &= m \cdot g_{\mu} \cdot c, \\
           \Delta\sigma^2 &= m \cdot g_{\sigma} \cdot c,

        where :math:`m` is multiplicity, :math:`g` are connection factors, and
        :math:`c` is the coefficient.

        **Use cases:**

        - Custom event queues separate from WFR infrastructure
        - Hybrid simulations mixing time-driven and event-driven updates
        - Debugging individual event contributions
        - Implementing non-standard delivery schedules

        Examples
        --------
        **Basic single-step event:**

        .. code-block:: python

            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection(drift_factor=0.8, diffusion_factor=0.3)
            >>> event = conn.to_siegert_event(coeff=15.0, delay_steps=5, multiplicity=1.0)
            >>> print(event)
            {'coeff': 15.0, 'drift_factor': 0.8, 'diffusion_factor': 0.3,
             'delay_steps': 5, 'multiplicity': 1.0}

        **Immediate delivery (zero delay):**

        .. code-block:: python

            >>> immediate_event = conn.to_siegert_event(coeff=20.0, delay_steps=0)
            >>> print(immediate_event['delay_steps'])
            0

        **Fractional multiplicity (population averaging):**

        .. code-block:: python

            >>> # 100-neuron population, 37 neurons fire this step
            >>> # Average rate contribution per neuron
            >>> avg_event = conn.to_siegert_event(coeff=42.0, multiplicity=37.0/100.0)
            >>> print(avg_event['multiplicity'])
            0.37

        **Custom event queue integration:**

        .. code-block:: python

            >>> # Build event queue for future delivery
            >>> event_queue = []
            >>> for t in range(10):
            ...     rate_value = 10.0 + 2.0 * t  # Ramping rate
            ...     evt = conn.to_siegert_event(coeff=rate_value, delay_steps=t)
            ...     event_queue.append(evt)
            >>>
            >>> # Process queue in simulation loop
            >>> for step in range(10):
            ...     due_events = [e for e in event_queue if e['delay_steps'] == step]
            ...     for evt in due_events:
            ...         # Deliver to target neuron
            ...         pass

        **Comparison with multi-lag event:**

        .. code-block:: python

            >>> import numpy as np
            >>> # Multi-lag WFR event
            >>> multi_lag = conn.prepare_secondary_event(np.array([10.0, 12.0, 11.5]))
            >>> print(multi_lag['coeffarray'])
            [10.  12.  11.5]
            >>>
            >>> # Equivalent single-step events
            >>> single_events = [
            ...     conn.to_siegert_event(coeff=10.0, delay_steps=0),
            ...     conn.to_siegert_event(coeff=12.0, delay_steps=1),
            ...     conn.to_siegert_event(coeff=11.5, delay_steps=2),
            ... ]
        """
        return {
            'coeff': self._to_float_scalar(coeff, name='coeff'),
            'drift_factor': float(self.drift_factor),
            'diffusion_factor': float(self.diffusion_factor),
            'delay_steps': self._to_int_scalar(delay_steps, name='delay_steps'),
            'multiplicity': self._to_float_scalar(multiplicity, name='multiplicity'),
        }

    def coeffarray_to_step_events(
        self,
        coeffarray: ArrayLike,
        first_delay_steps: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
    ) -> list[dict[str, Any]]:
        r"""Convert multi-lag coefficient array into sequential single-step events.

        Transforms a NEST-style lag-indexed coefficient array (used in WFR secondary
        events) into a list of single-step event dictionaries with linearly increasing
        delay steps. Each coefficient becomes a separate event scheduled for sequential
        future time steps.

        Parameters
        ----------
        coeffarray : array-like or Quantity
            Multi-lag coefficient array representing rate values at sequential time
            substeps. Shape: ``(n_lags,)`` where ``n_lags`` is the number of events
            to generate. If ``brainunit.Quantity``, mantissa is extracted. Must be
            non-empty 1D array.
        first_delay_steps : int, array-like, or Quantity, optional
            Time step offset for the first event (lag 0). Subsequent events are
            scheduled at ``first_delay_steps + lag_index``. Must be scalar non-negative
            integer.
            Default: ``0`` (immediate delivery sequence starting current step).
        multiplicity : float, array-like, or Quantity, optional
            Shared event weight multiplier applied to all events. Must be scalar.
            Typically ``1.0`` (no scaling) or represents population fraction.
            Default: ``1.0`` (no scaling).

        Returns
        -------
        list[dict[str, Any]]
            List of event dictionaries, length ``n_lags``. Each dictionary contains:

            - ``'coeff'`` : float — Coefficient value from ``coeffarray[i]``
            - ``'drift_factor'`` : float — Connection's drift scaling factor
            - ``'diffusion_factor'`` : float — Connection's diffusion scaling factor
            - ``'delay_steps'`` : int — Delivery step = ``first_delay_steps + i``
            - ``'multiplicity'`` : float — Shared multiplicity value

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty after conversion.
        ValueError
            If ``first_delay_steps`` is negative.
        ValueError
            If ``first_delay_steps`` or ``multiplicity`` is not scalar.

        Notes
        -----
        **Delay step calculation:**

        For lag index :math:`i \in [0, n_{\text{lags}}-1]`:

        .. math::

           \text{delay\_steps}_i = d_0 + i,

        where :math:`d_0` is ``first_delay_steps``.

        **Event ordering:**

        Events are returned in ascending delay order (lag 0 first). This matches the
        typical chronological event processing order in simulation loops.

        **Multiplicity semantics:**

        All events share the same multiplicity. For per-event scaling, call
        :meth:`to_siegert_event` individually in a loop with different multiplicity
        values.

        **Use cases:**

        - Converting WFR multi-lag events to single-step event queue format
        - Implementing custom delay profiles in event-driven simulators
        - Debugging event delivery sequences
        - Bridging between time-driven (coeffarray) and event-driven representations

        **Performance note:**

        This method creates ``n_lags`` separate dictionary objects. For large coefficient
        arrays (n_lags > 1000), consider using :meth:`prepare_secondary_event` with
        native WFR handling instead.

        Examples
        --------
        **Basic conversion with zero initial delay:**

        .. code-block:: python

            >>> import numpy as np
            >>> import brainpy.state as bs
            >>> conn = bs.diffusion_connection(drift_factor=0.8, diffusion_factor=0.3)
            >>> coeffs = np.array([10.0, 12.0, 11.5, 10.2, 9.8])
            >>> events = conn.coeffarray_to_step_events(coeffs, first_delay_steps=0)
            >>> for i, evt in enumerate(events):
            ...     print(f"Lag {i}: coeff={evt['coeff']:.1f}, delay={evt['delay_steps']}")
            Lag 0: coeff=10.0, delay=0
            Lag 1: coeff=12.0, delay=1
            Lag 2: coeff=11.5, delay=2
            Lag 3: coeff=10.2, delay=3
            Lag 4: coeff=9.8, delay=4

        **Non-zero initial delay:**

        .. code-block:: python

            >>> coeffs = np.array([5.0, 6.0, 7.0])
            >>> events = conn.coeffarray_to_step_events(coeffs, first_delay_steps=10)
            >>> for evt in events:
            ...     print(f"Delay: {evt['delay_steps']}, Coeff: {evt['coeff']}")
            Delay: 10, Coeff: 5.0
            Delay: 11, Coeff: 6.0
            Delay: 12, Coeff: 7.0

        **With multiplicity scaling:**

        .. code-block:: python

            >>> # Population of 100 neurons, 75 active
            >>> coeffs = np.array([20.0, 25.0, 22.0])
            >>> events = conn.coeffarray_to_step_events(
            ...     coeffs,
            ...     first_delay_steps=0,
            ...     multiplicity=0.75,  # 75% population activity
            ... )
            >>> print(events[0]['multiplicity'])
            0.75

        **Integration with event queue:**

        .. code-block:: python

            >>> # Convert WFR coefficients to event queue
            >>> from collections import defaultdict
            >>> coeffs = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
            >>> events = conn.coeffarray_to_step_events(coeffs, first_delay_steps=5)
            >>>
            >>> # Organize by delivery step
            >>> event_queue = defaultdict(list)
            >>> for evt in events:
            ...     event_queue[evt['delay_steps']].append(evt)
            >>>
            >>> # Process events at step 7
            >>> step_7_events = event_queue[7]
            >>> print(f"Step 7 receives {len(step_7_events)} event(s)")
            >>> print(f"Coefficient: {step_7_events[0]['coeff']}")
            Step 7 receives 1 event(s)
            Coefficient: 12.0

        **Negative first_delay_steps error:**

        .. code-block:: python

            >>> try:
            ...     conn.coeffarray_to_step_events(coeffs, first_delay_steps=-5)
            ... except ValueError as e:
            ...     print(e)
            first_delay_steps must be >= 0.

        **Verifying event structure:**

        .. code-block:: python

            >>> coeffs = np.array([15.0, 16.0])
            >>> events = conn.coeffarray_to_step_events(coeffs)
            >>> evt = events[0]
            >>> print(f"Keys: {sorted(evt.keys())}")
            >>> print(f"Drift factor: {evt['drift_factor']}")
            >>> print(f"Diffusion factor: {evt['diffusion_factor']}")
            Keys: ['coeff', 'delay_steps', 'diffusion_factor', 'drift_factor', 'multiplicity']
            Drift factor: 0.8
            Diffusion factor: 0.3
        """
        coeff_np = self._to_coeff_array(coeffarray)
        d0 = self._to_int_scalar(first_delay_steps, name='first_delay_steps')
        mult = self._to_float_scalar(multiplicity, name='multiplicity')
        if d0 < 0:
            raise ValueError('first_delay_steps must be >= 0.')

        events = []
        for i, c in enumerate(coeff_np):
            events.append(
                {
                    'coeff': float(c),
                    'drift_factor': float(self.drift_factor),
                    'diffusion_factor': float(self.diffusion_factor),
                    'delay_steps': int(d0 + i),
                    'multiplicity': float(mult),
                }
            )
        return events

    @staticmethod
    def _to_coeff_array(value: ArrayLike) -> np.ndarray:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size == 0:
            raise ValueError('Coefficient array must not be empty.')
        return arr

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr[0])

    @staticmethod
    def _to_int_scalar(value: ArrayLike, name: str) -> int:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return int(arr[0])
