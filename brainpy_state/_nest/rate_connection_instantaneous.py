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

from typing import Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

__all__ = [
    'rate_connection_instantaneous',
]


class rate_connection_instantaneous:
    r"""NEST-compatible ``rate_connection_instantaneous`` connection model.

    Implements connection-level semantics for instantaneous (zero-delay) rate connections
    following NEST's ``rate_connection_instantaneous`` model. This class represents a
    synapse that transmits rate signals without delay, supporting secondary event propagation
    and waveform relaxation for rate-based network simulations.

    Unlike ``rate_connection_delayed``, this model enforces zero delay and rejects any
    attempt to configure a delay parameter. Instantaneous connections are essential for
    modeling fast synaptic interactions and implementing implicit integration schemes.

    Parameters
    ----------
    weight : float or array-like, optional
        Connection gain/strength applied to transmitted rate signals. Must be scalar.
        Default: ``1.0``.
    name : str or None, optional
        Optional name identifier for this connection instance. Default: ``None``.

    Attributes
    ----------
    weight : float
        Connection gain (validated scalar).
    delay : int
        Compatibility field, always ``1``. Exposed in ``get_status`` for NEST API parity
        but ignored by transmission logic. Cannot be modified.
    name : str or None
        Instance name.
    HAS_DELAY : bool
        Class attribute, always ``False`` (this model enforces zero delay).
    SUPPORTS_WFR : bool
        Class attribute, always ``True`` (supports waveform relaxation).

    Parameter Mapping
    -----------------
    The following table maps NEST parameters to this implementation:

    ================================  ====================  ========================================
    NEST Parameter                    brainpy.state         Notes
    ================================  ====================  ========================================
    ``weight``                        ``weight``            Connection gain (scalar float)
    ``delay``                         ``delay``             Read-only, always ``1`` (compatibility)
    ``has_delay``                     ``HAS_DELAY``         Always ``False`` (class attribute)
    ``supports_wfr``                  ``SUPPORTS_WFR``      Always ``True`` (class attribute)
    ================================  ====================  ========================================

    Mathematical Description
    ------------------------
    **1. Instantaneous Connection Model**

    A ``rate_connection_instantaneous`` synapse transmits a rate signal :math:`r_\text{pre}(t)`
    from a presynaptic neuron to a postsynaptic neuron without delay, applying only the
    connection weight :math:`w`:

    .. math::

       r_\text{post}(t) = w \cdot r_\text{pre}(t)

    In contrast to delayed connections, the postsynaptic rate depends on the current
    (not past) presynaptic rate. This enables instantaneous feedback loops and is required
    for certain implicit integration methods.

    **2. Secondary Event Handling with Waveform Relaxation**

    NEST rate neurons support waveform relaxation (WFR), an iterative method for solving
    coupled differential equations. During WFR iterations, neurons exchange secondary
    events containing coefficient arrays :math:`\{c_0, c_1, \ldots, c_{n-1}\}` representing
    contributions at multiple time lags within the current WFR interval.

    For instantaneous connections, the receiver processes each coefficient :math:`c_i` by
    adding it to input buffer slot :math:`i`:

    .. math::

       I_{\text{instant}}[i] \mathrel{{+}{=}} w \cdot c_i, \quad i = 0, 1, \ldots, n-1

    This direct indexing reflects the zero-delay nature of the connection—coefficients
    apply to the current and immediate future steps without temporal offset.

    **3. Event Payload Structure**

    An instantaneous rate event contains:

    - ``rate``: Scalar rate value or coefficient (dimensionless)
    - ``weight``: Connection weight
    - ``delay_steps``: Always ``0`` for instantaneous events
    - ``multiplicity``: Optional scaling factor for probabilistic connections

    The method :meth:`coeffarray_to_step_events` expands a coefficient array into
    individual per-step events with sequential delay offsets starting from
    ``first_delay_steps``.

    Implementation Notes
    --------------------
    **Delay Restriction**

    This model enforces zero delay for all transmissions. Any attempt to set a delay
    via ``set_delay()``, ``set_delay_steps()``, or ``set_status(delay=...)`` raises
    a ``ValueError`` with the message:

        ``"rate_connection_instantaneous has no delay. Please use rate_connection_delayed."``

    The ``delay`` attribute is exposed in ``get_status()`` for NEST API compatibility
    (always returns ``1``), but this value is ignored by the transmission logic and
    cannot be modified.

    **Unit Handling**

    All parameters accept ``brainunit.Quantity`` objects or plain numeric values. If a
    ``Quantity`` is provided, its mantissa is extracted. Internally, values are stored
    as dimensionless floats or integers.

    **Compatibility with NEST**

    - NEST stores connection properties in synapse objects and enforces delay restrictions
      at runtime. This implementation replicates that behavior by raising errors when
      delay modification is attempted.
    - Secondary event expansion follows NEST's direct buffer indexing for instantaneous
      events (no delay offset applied).
    - The ``supports_wfr`` flag is set to ``True``, indicating compatibility with NEST's
      waveform relaxation solver.

    **Coefficient Array Indexing**

    The method :meth:`coeffarray_to_step_events` maps each coefficient ``coeffarray[i]``
    to an event with ``delay_steps = first_delay_steps + i``. This allows instantaneous
    connections to participate in multi-step implicit integration by distributing
    coefficients across future time steps, starting from a specified base delay.

    Raises
    ------
    ValueError
        If ``weight`` is not scalar.
    ValueError
        If any method attempts to set ``delay``, ``delay_steps``, or if ``to_rate_event``
        is called with non-zero ``delay_steps``.
    ValueError
        If coefficient array is empty or ``first_delay_steps < 0`` in
        ``coeffarray_to_step_events``.

    See Also
    --------
    rate_connection_delayed : Delayed rate connection model (NEST equivalent)
    rate_neuron_ipn : Input rate neuron (instantaneous event receiver)
    rate_neuron_opn : Output rate neuron (instantaneous event receiver)

    References
    ----------
    .. [1] Hahne, J., et al. (2015). "A unified framework for spiking and rate-based
           neural networks." Frontiers in Neuroinformatics, 9, 22.
    .. [2] NEST Simulator documentation: Rate neuron models.
           https://nest-simulator.readthedocs.io/en/stable/models/rate_connection_instantaneous.html
    .. [3] NEST source: ``models/rate_connection_instantaneous.{h,cpp}``.
    .. [4] NEST instantaneous-rate receiver handling: ``models/rate_neuron_ipn_impl.h``
           and ``models/rate_neuron_opn_impl.h``.
    .. [5] Lelarasmee, E., et al. (1982). "The waveform relaxation method for time-domain
           analysis of large scale integrated circuits." IEEE Trans. CAD, 1(3), 131-145.

    Examples
    --------
    **Basic Usage**

    Create an instantaneous connection with weight 2.0:

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> conn = bst.rate_connection_instantaneous(weight=2.0)
       >>> conn.get_status()
       {'weight': 2.0, 'delay': 1, 'has_delay': False, 'supports_wfr': True}

    **Creating Rate Events**

    Transmit an instantaneous rate signal:

    .. code-block:: python

       >>> event = conn.to_rate_event(rate=5.0, multiplicity=1.0)
       >>> event
       {'rate': 5.0, 'weight': 2.0, 'delay_steps': 0, 'multiplicity': 1.0}

    **Secondary Event Expansion**

    Map a coefficient array to per-step events for multi-step integration:

    .. code-block:: python

       >>> coeffs = [0.5, 1.0, 0.3]  # Three lag coefficients
       >>> events = conn.coeffarray_to_step_events(coeffs, first_delay_steps=0)
       >>> len(events)
       3
       >>> events[0]
       {'rate': 0.5, 'weight': 2.0, 'delay_steps': 0, 'multiplicity': 1.0}
       >>> events[1]
       {'rate': 1.0, 'weight': 2.0, 'delay_steps': 1, 'multiplicity': 1.0}
       >>> events[2]
       {'rate': 0.3, 'weight': 2.0, 'delay_steps': 2, 'multiplicity': 1.0}

    **Delay Restriction Enforcement**

    Attempting to set a delay raises an error:

    .. code-block:: python

       >>> conn.set_delay(3)
       Traceback (most recent call last):
           ...
       ValueError: rate_connection_instantaneous has no delay. Please use rate_connection_delayed.

    **Dynamic Weight Updates**

    Update connection weight at runtime:

    .. code-block:: python

       >>> conn.set_weight(1.5)
       >>> conn.get('weight')
       1.5
       >>> conn.set_status(weight=3.0)
       >>> conn.get('weight')
       3.0

    **Using with Rate Neurons**

    Typical usage in a rate-based network (conceptual example):

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import brainunit as u
       >>> # Create rate neurons (assuming rate_neuron_ipn exists)
       >>> pre = bst.rate_neuron_ipn(size=10, tau=10.0*u.ms)
       >>> post = bst.rate_neuron_ipn(size=5, tau=10.0*u.ms)
       >>> # Define instantaneous connections for fast feedback
       >>> conns = [bst.rate_connection_instantaneous(weight=w)
       ...          for w in [0.8, 1.2, 0.5]]
       >>> # Transmit events during WFR iterations
       >>> for conn in conns:
       ...     event = conn.to_rate_event(pre.rate, multiplicity=1.0)
       ...     # post.handle_instantaneous_event(event)  # Receiver-side logic
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = False
    SUPPORTS_WFR = True

    _DELAY_ERROR = (
        'rate_connection_instantaneous has no delay. Please use '
        'rate_connection_delayed.'
    )

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        name: str | None = None,
    ):
        self.name = name
        self.weight = self._to_float_scalar(weight, name='weight')
        # Kept for status parity with NEST; not used in transmission logic.
        self.delay = 1

    @property
    def properties(self) -> dict[str, Any]:
        r"""Return connection model properties.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``'has_delay'`` (bool): Always ``False`` for this model.
            - ``'supports_wfr'`` (bool): Always ``True`` (waveform relaxation supported).
        """
        return {
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        r"""Retrieve all connection parameters as a dictionary.

        Follows NEST's ``GetStatus`` API convention.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``'weight'`` (float): Connection gain.
            - ``'delay'`` (int): Always ``1`` (compatibility field, read-only).
            - ``'has_delay'`` (bool): Always ``False``.
            - ``'supports_wfr'`` (bool): Always ``True``.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_instantaneous(weight=1.5)
           >>> status = conn.get_status()
           >>> status['weight']
           1.5
           >>> status['delay']
           1
           >>> status['has_delay']
           False
        """
        return {
            'weight': float(self.weight),
            'delay': int(self.delay),
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update connection parameters from a dictionary or keyword arguments.

        Follows NEST's ``SetStatus`` API convention. Only allows updating ``weight``.
        Any attempt to set ``delay`` or ``delay_steps`` raises a ``ValueError``.

        Parameters
        ----------
        status : dict or None, optional
            Dictionary of parameters to update. Only ``'weight'`` is allowed.
        **kwargs
            Alternative parameter specification as keyword arguments. Merged with
            ``status`` (keyword args take precedence).

        Raises
        ------
        ValueError
            If ``delay`` or ``delay_steps`` is present in updates (with NEST-matching
            error message).
        ValueError
            If ``weight`` fails validation (non-scalar).

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_instantaneous(weight=1.0)
           >>> conn.set_status({'weight': 2.5})
           >>> conn.get('weight')
           2.5
           >>> conn.set_status(weight=3.0)  # Keyword argument style
           >>> conn.get('weight')
           3.0

        Delay updates are rejected:

        .. code-block:: python

           >>> conn.set_status(delay=3)
           Traceback (most recent call last):
               ...
           ValueError: rate_connection_instantaneous has no delay. Please use rate_connection_delayed.
        """
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        # Match NEST behavior: reject delay updates before applying any weight.
        if 'delay' in updates or 'delay_steps' in updates:
            raise ValueError(self._DELAY_ERROR)

        if 'weight' in updates:
            self.set_weight(updates['weight'])

    def get(self, key: str = 'status'):
        r"""Retrieve a specific parameter or full status dictionary.

        Parameters
        ----------
        key : str, optional
            Parameter name to retrieve. Special value ``'status'`` returns full
            status dictionary. Supported keys: ``'status'``, ``'weight'``,
            ``'delay'``, ``'has_delay'``, ``'supports_wfr'``.
            Default: ``'status'``.

        Returns
        -------
        dict or scalar
            If ``key == 'status'``, returns full status dictionary. Otherwise,
            returns the requested parameter value.

        Raises
        ------
        KeyError
            If ``key`` is not a recognized parameter name.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_instantaneous(weight=2.0)
           >>> conn.get('weight')
           2.0
           >>> conn.get('has_delay')
           False
           >>> conn.get('status')
           {'weight': 2.0, 'delay': 1, 'has_delay': False, 'supports_wfr': True}
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for rate_connection_instantaneous.get().')

    def set_weight(self, weight: ArrayLike):
        r"""Update the connection weight.

        Parameters
        ----------
        weight : float or array-like
            New connection gain. Must be scalar. Accepts ``brainunit.Quantity``
            (mantissa will be extracted).

        Raises
        ------
        ValueError
            If ``weight`` is not scalar.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_instantaneous()
           >>> conn.set_weight(2.5)
           >>> conn.get('weight')
           2.5
        """
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, _):
        r"""Reject delay modification (instantaneous model has no delay).

        This method always raises a ``ValueError`` to enforce NEST semantics.

        Parameters
        ----------
        _ : any
            Ignored. Delay cannot be set for instantaneous connections.

        Raises
        ------
        ValueError
            Always raised with NEST-matching error message.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_instantaneous()
           >>> conn.set_delay(5)
           Traceback (most recent call last):
               ...
           ValueError: rate_connection_instantaneous has no delay. Please use rate_connection_delayed.
        """
        raise ValueError(self._DELAY_ERROR)

    def set_delay_steps(self, _):
        r"""Reject delay_steps modification (instantaneous model has no delay).

        This method always raises a ``ValueError`` to enforce NEST semantics.

        Parameters
        ----------
        _ : any
            Ignored. Delay cannot be set for instantaneous connections.

        Raises
        ------
        ValueError
            Always raised with NEST-matching error message.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_instantaneous()
           >>> conn.set_delay_steps(3)
           Traceback (most recent call last):
               ...
           ValueError: rate_connection_instantaneous has no delay. Please use rate_connection_delayed.
        """
        raise ValueError(self._DELAY_ERROR)

    def prepare_secondary_event(self, coeffarray: ArrayLike) -> dict[str, Any]:
        r"""Create an instantaneous secondary-event payload.

        Secondary events are used in waveform relaxation (WFR) and implicit integration
        schemes for rate neurons. The coefficient array represents contributions at
        multiple time lags within the current WFR interval.

        Parameters
        ----------
        coeffarray : array-like
            1D array of coefficients, shape ``(n,)``, representing rate contributions
            at ``n`` consecutive time lags. Must be non-empty. Accepts
            ``brainunit.Quantity`` (mantissa will be extracted).

        Returns
        -------
        dict
            Event payload with keys:

            - ``'coeffarray'`` (ndarray): Validated coefficient array, shape ``(n,)``,
              dtype ``float64``.
            - ``'weight'`` (float): Connection weight.

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty or cannot be converted to a 1D array.

        See Also
        --------
        coeffarray_to_step_events : Expand coefficient array to per-step events.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_instantaneous(weight=2.0)
           >>> event = conn.prepare_secondary_event([0.5, 1.0, 0.3])
           >>> event['coeffarray']
           array([0.5, 1. , 0.3])
           >>> event['weight']
           2.0
        """
        return {
            'coeffarray': self._to_coeff_array(coeffarray),
            'weight': float(self.weight),
        }

    def to_rate_event(
        self,
        rate: ArrayLike,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike = 0,
    ) -> dict[str, Any]:
        r"""Create an instantaneous rate-event payload for step-based simulation APIs.

        Constructs an event dictionary that can be passed to rate neuron receivers
        to transmit a rate signal with the connection's weight and zero delay.

        Parameters
        ----------
        rate : float or array-like
            Rate value to transmit. Can be scalar (single rate) or array (multiple
            rates). Accepts ``brainunit.Quantity`` (mantissa will be extracted).
        multiplicity : float or array-like, optional
            Scaling factor for stochastic or probabilistic connections. Must be
            scalar. Default: ``1.0``.
        delay_steps : int or array-like, optional
            Must be ``0`` for instantaneous connections. Included for API consistency
            with ``rate_connection_delayed``. Default: ``0``.

        Returns
        -------
        dict
            Event payload with keys:

            - ``'rate'`` (float or ndarray): Rate value (scalar or array).
            - ``'weight'`` (float): Connection weight.
            - ``'delay_steps'`` (int): Always ``0`` for instantaneous connections.
            - ``'multiplicity'`` (float): Scaling factor.

        Raises
        ------
        ValueError
            If ``delay_steps`` is not ``0`` (instantaneous connections enforce zero delay).
        ValueError
            If ``multiplicity`` is not scalar.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_instantaneous(weight=2.0)
           >>> event = conn.to_rate_event(rate=5.0)
           >>> event
           {'rate': 5.0, 'weight': 2.0, 'delay_steps': 0, 'multiplicity': 1.0}

        Non-zero delay raises an error:

        .. code-block:: python

           >>> conn.to_rate_event(rate=5.0, delay_steps=3)
           Traceback (most recent call last):
               ...
           ValueError: delay_steps for rate_connection_instantaneous must be 0.
        """
        d = self._to_int_scalar(delay_steps, name='delay_steps')
        if d != 0:
            raise ValueError('delay_steps for rate_connection_instantaneous must be 0.')
        return {
            'rate': self._to_rate_value(rate),
            'weight': float(self.weight),
            'delay_steps': 0,
            'multiplicity': self._to_float_scalar(multiplicity, name='multiplicity'),
        }

    def coeffarray_to_step_events(
        self,
        coeffarray: ArrayLike,
        first_delay_steps: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
    ) -> list[dict[str, Any]]:
        r"""Map lag-indexed coefficient array to per-step instantaneous-rate events.

        Expands a coefficient array representing multiple time lags into individual
        per-step rate events, following NEST's instantaneous-rate receiver buffer indexing.
        Each coefficient :math:`c_i` is mapped to an event with delay
        :math:`d_{\text{first}} + i`, where :math:`d_{\text{first}}` is the base delay offset.

        Unlike delayed connections, instantaneous connections apply coefficients directly
        to buffer slots without subtracting a minimum delay. This method is used to
        distribute waveform relaxation (WFR) coefficients across future time steps.

        Parameters
        ----------
        coeffarray : array-like
            1D array of coefficients, shape ``(n,)``. Each element represents a rate
            contribution at a successive time lag. Must be non-empty. Accepts
            ``brainunit.Quantity`` (mantissa will be extracted).
        first_delay_steps : int or array-like, optional
            Base delay offset for the first coefficient. Subsequent coefficients are
            mapped to ``first_delay_steps + i``. Must be integer-valued scalar ``>= 0``.
            Default: ``0``.
        multiplicity : float or array-like, optional
            Scaling factor for all events. Must be scalar. Default: ``1.0``.

        Returns
        -------
        list of dict
            List of event dictionaries, one per coefficient. Each event has keys:

            - ``'rate'`` (float): Coefficient value.
            - ``'weight'`` (float): Connection weight.
            - ``'delay_steps'`` (int): Computed delay = ``first_delay_steps + i``.
            - ``'multiplicity'`` (float): Scaling factor.

            Length of list equals ``len(coeffarray)``.

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty.
        ValueError
            If ``first_delay_steps < 0`` (instantaneous events start from step 0 or later).
        ValueError
            If ``first_delay_steps`` or ``multiplicity`` is not scalar.

        Notes
        -----
        **NEST Buffer Indexing for Instantaneous Connections**

        NEST rate neurons handle instantaneous events by directly indexing into the
        input buffer. For each coefficient :math:`c_i`, the receiver adds the weighted
        contribution to buffer slot :math:`i`:

        .. math::

           I_{\text{instant}}[d_{\text{first}} + i] \mathrel{{+}{=}} w \cdot c_i

        This direct indexing reflects the zero-delay nature—no network-wide minimum
        delay offset is subtracted.

        **Comparison with Delayed Connections**

        Delayed connections compute buffer slots as :math:`(d - d_{\text{min}}) + i`
        (see ``rate_connection_delayed.coeffarray_to_step_events``). Instantaneous
        connections use :math:`d_{\text{first}} + i` without delay adjustment.

        **Example Calculation**

        Suppose ``first_delay_steps = 0`` and ``coeffarray = [0.5, 1.0, 0.3]``:

        - Coefficient 0 (``0.5``): ``delay_steps = 0 + 0 = 0``
        - Coefficient 1 (``1.0``): ``delay_steps = 0 + 1 = 1``
        - Coefficient 2 (``0.3``): ``delay_steps = 0 + 2 = 2``

        If ``first_delay_steps = 2``:

        - Coefficient 0: ``delay_steps = 2 + 0 = 2``
        - Coefficient 1: ``delay_steps = 2 + 1 = 3``
        - Coefficient 2: ``delay_steps = 2 + 2 = 4``

        See Also
        --------
        prepare_secondary_event : Create secondary event with coefficient array.
        to_rate_event : Create single-rate event.

        References
        ----------
        .. [3] NEST instantaneous-rate receiver implementation:
               ``models/rate_neuron_ipn_impl.h`` and ``models/rate_neuron_opn_impl.h``.
        .. [4] Hahne, J., et al. (2015). "A unified framework for spiking and rate-based
               neural networks." Frontiers in Neuroinformatics, 9, 22.

        Examples
        --------
        Basic usage with 3 coefficients starting from step 0:

        .. code-block:: python

           >>> conn = rate_connection_instantaneous(weight=2.0)
           >>> coeffs = [0.5, 1.0, 0.3]
           >>> events = conn.coeffarray_to_step_events(coeffs, first_delay_steps=0)
           >>> len(events)
           3
           >>> events[0]
           {'rate': 0.5, 'weight': 2.0, 'delay_steps': 0, 'multiplicity': 1.0}
           >>> events[1]
           {'rate': 1.0, 'weight': 2.0, 'delay_steps': 1, 'multiplicity': 1.0}
           >>> events[2]
           {'rate': 0.3, 'weight': 2.0, 'delay_steps': 2, 'multiplicity': 1.0}

        Starting from a later step (e.g., for WFR multi-step integration):

        .. code-block:: python

           >>> events = conn.coeffarray_to_step_events(coeffs, first_delay_steps=5)
           >>> events[0]['delay_steps']
           5
           >>> events[1]['delay_steps']
           6
           >>> events[2]['delay_steps']
           7

        Error when first_delay_steps is negative:

        .. code-block:: python

           >>> conn.coeffarray_to_step_events([0.5, 1.0], first_delay_steps=-1)
           Traceback (most recent call last):
               ...
           ValueError: first_delay_steps must be >= 0.

        Using with multiplicity scaling:

        .. code-block:: python

           >>> conn = rate_connection_instantaneous(weight=1.0)
           >>> events = conn.coeffarray_to_step_events([0.5, 1.0], first_delay_steps=0, multiplicity=0.8)
           >>> events[0]['multiplicity']
           0.8
        """
        coeff = self._to_coeff_array(coeffarray)
        d0 = self._to_int_scalar(first_delay_steps, name='first_delay_steps')
        if d0 < 0:
            raise ValueError('first_delay_steps must be >= 0.')
        mult = self._to_float_scalar(multiplicity, name='multiplicity')

        events = []
        for i, c in enumerate(coeff):
            events.append(
                {
                    'rate': float(c),
                    'weight': float(self.weight),
                    'delay_steps': int(d0 + i),
                    'multiplicity': float(mult),
                }
            )
        return events

    @staticmethod
    def _to_coeff_array(value: ArrayLike) -> np.ndarray:
        r"""Convert input to a validated 1D float64 coefficient array.

        Parameters
        ----------
        value : array-like
            Input value. Accepts ``brainunit.Quantity`` (mantissa extracted).

        Returns
        -------
        ndarray
            1D array with dtype ``float64``.

        Raises
        ------
        ValueError
            If array is empty after conversion.
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError('Coefficient array must not be empty.')
        return arr

    @staticmethod
    def _to_rate_value(value: ArrayLike):
        r"""Convert input to a rate value (scalar float or array).

        Parameters
        ----------
        value : array-like
            Input value. Accepts ``brainunit.Quantity`` (mantissa extracted).

        Returns
        -------
        float or ndarray
            If input has exactly one element, returns scalar float. Otherwise,
            returns array with dtype ``float64``.
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return arr

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        r"""Convert input to a validated scalar float.

        Parameters
        ----------
        value : array-like
            Input value. Accepts ``brainunit.Quantity`` (mantissa extracted).
        name : str
            Parameter name for error messages.

        Returns
        -------
        float
            Scalar float value.

        Raises
        ------
        ValueError
            If input is not scalar (size != 1).
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr[0])

    @staticmethod
    def _to_int_scalar(value: ArrayLike, name: str) -> int:
        r"""Convert input to a validated integer scalar.

        Parameters
        ----------
        value : array-like
            Input value. Accepts ``brainunit.Quantity`` (mantissa extracted).
        name : str
            Parameter name for error messages.

        Returns
        -------
        int
            Integer value (rounded from float if within tolerance ``1e-12``).

        Raises
        ------
        ValueError
            If input is not scalar, not finite, or not integer-valued
            (``|value - round(value)| > 1e-12``).
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr[0])
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        vr = int(round(v))
        if abs(v - vr) > 1e-12:
            raise ValueError(f'{name} must be integer-valued.')
        return vr
