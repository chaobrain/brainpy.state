from typing import Any

import brainstate
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

    **2. Network Routing (Dual-Channel Seam Deposit)**

    The connection is a thin NEST-parity *status spec*: it carries no state and no
    host event queue. When wired through the :class:`~brainpy_state.Simulator`
    (``net.connect(pre, post, synapse=diffusion_connection(...))``) the simulator
    dispatches on the :attr:`_IS_DIFFUSION` marker and builds two seam-(H)
    continuous-rate deposits from the single presynaptic rate :math:`r_j`:

    .. math::

       \text{drift channel ('diffusion\_mu')} &:\quad g_{\mu}\,r_j \;\to\; \mu_i, \\
       \text{variance channel ('diffusion\_sigma2')} &:\quad g_{\sigma}\,r_j \;\to\; \sigma^2_i.

    The two deposits use distinct delta-channel labels so a target ``siegert_neuron``
    reads them back independently (``sum_delta_inputs(0.0, label='diffusion_mu')`` and
    ``sum_delta_inputs(0.0, label='diffusion_sigma2')``), exactly mirroring NEST's
    single ``DiffusionConnectionEvent`` that deposits both a drift and a variance
    coefficient. This reproduces NEST's ``min_delay = 1`` one-step coupling lag.

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

    2. **Event handling**: NEST delivers a C++ ``DiffusionConnectionEvent`` per step;
       here the :class:`~brainpy_state.Simulator` routes the connection as two
       labeled seam deposits (drift, variance) dispatched on :attr:`_IS_DIFFUSION`,
       so the model object itself stays stateless.

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
    **Basic connection parameters:**

    .. code-block:: python

        >>> from brainpy import state as bs
        >>> # A diffusion connection is a stateless status spec.
        >>> conn = bs.diffusion_connection(
        ...     drift_factor=0.8,      # mean (drift) scaling g_mu
        ...     diffusion_factor=0.3,  # variance (diffusion) scaling g_sigma
        ...     name='excitatory_diffusion',
        ... )
        >>> status = conn.get_status()
        >>> print(f"Drift: {status['drift_factor']}, Diffusion: {status['diffusion_factor']}")
        Drift: 0.8, Diffusion: 0.3

    **Wiring two Siegert populations through the Simulator:**

    .. code-block:: python

        >>> import brainunit as u
        >>> net = bs.Simulator(dt=0.1 * u.ms)
        >>> src = net.create(bs.siegert_neuron, 1, params=dict(tau_m=10.0 * u.ms, theta=15.0, mean=20.0))
        >>> tgt = net.create(bs.siegert_neuron, 1, params=dict(tau_m=10.0 * u.ms, theta=15.0))
        >>> _ = net.connect(
        ...     src, tgt,
        ...     synapse=bs.diffusion_connection(drift_factor=0.8, diffusion_factor=0.3),
        ... )
        >>> # net.simulate(50.0 * u.ms) relaxes tgt toward Phi(0.8*r_src, 0.3*r_src).

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

    #: Marker the Simulator dispatches on to route this connection as a dual-channel
    #: diffusion deposit (drift -> mu, diffusion -> sigma^2) instead of a plastic
    #: projection. See ``_network/_simulator.py::_build_siegert_diffusion``.
    _IS_DIFFUSION = True

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
        super().__init__(in_size=1, name=name)
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

            >>> from brainpy import state as bs
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

            >>> from brainpy import state as bs
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

            >>> from brainpy import state as bs
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

            >>> from brainpy import state as bs
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

            >>> from brainpy import state as bs
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

            >>> from brainpy import state as bs
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

            >>> from brainpy import state as bs
            >>> conn = bs.diffusion_connection()
            >>> try:
            ...     conn.set_delay(1.0)
            ... except ValueError as e:
            ...     print(e)
            diffusion_connection has no delay
        """
        raise ValueError(self._DELAY_ERROR)

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        dftype = brainstate.environ.dftype()
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr[0])

