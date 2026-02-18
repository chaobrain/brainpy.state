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



from typing import Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

__all__ = [
    'rate_connection_delayed',
]


class rate_connection_delayed(NESTSynapse):
    r"""NEST-compatible ``rate_connection_delayed`` connection model.

    Implements connection-level semantics for delayed rate connections following NEST's
    ``rate_connection_delayed`` model. This class represents a synapse that transmits
    rate signals with a configurable delay, supporting secondary event propagation
    for rate-based network simulations.

    Unlike ``rate_connection_instantaneous``, this model enforces a minimum delay of
    one simulation step and supports delayed secondary events through coefficient arrays.

    Parameters
    ----------
    weight : float or array-like, optional
        Connection gain/strength applied to transmitted rate signals. Must be scalar.
        Default: ``1.0``.
    delay_steps : int or array-like, optional
        Transmission delay in discrete simulation steps. Must be integer-valued and
        ``>= 1``. Default: ``1``.
    name : str or None, optional
        Optional name identifier for this connection instance. Default: ``None``.

    Attributes
    ----------
    weight : float
        Connection gain (validated scalar).
    delay_steps : int
        Delay in simulation steps (validated ``>= 1``).
    name : str or None
        Instance name.
    HAS_DELAY : bool
        Class attribute, always ``True`` (this model enforces delay).
    SUPPORTS_WFR : bool
        Class attribute, always ``False`` (waveform relaxation not supported).

    Parameter Mapping
    -----------------
    The following table maps NEST parameters to this implementation:

    ================================  ====================  ========================================
    NEST Parameter                    brainpy.state         Notes
    ================================  ====================  ========================================
    ``weight``                        ``weight``            Connection gain (scalar float)
    ``delay``                         ``delay_steps``       Delay in steps (integer ``>= 1``)
    ``has_delay``                     ``HAS_DELAY``         Always ``True`` (class attribute)
    ``supports_wfr``                  ``SUPPORTS_WFR``      Always ``False`` (class attribute)
    ================================  ====================  ========================================

    Mathematical Description
    ------------------------
    **1. Connection Model**

    A ``rate_connection_delayed`` synapse transmits a rate signal :math:`r_\text{pre}(t)`
    from a presynaptic neuron to a postsynaptic neuron with delay :math:`d` and weight
    :math:`w`:

    .. math::

       r_\text{post}(t) = w \cdot r_\text{pre}(t - d)

    In discrete-time simulation with step size :math:`\Delta t`, the delay is quantized
    to an integer number of steps:

    .. math::

       d_\text{steps} = \left\lceil \frac{d}{\Delta t} \right\rceil, \quad d_\text{steps} \geq 1

    **2. Secondary Event Handling**

    NEST rate neurons support secondary events for implicit integration schemes. A
    secondary event carries a coefficient array :math:`\{c_0, c_1, \ldots, c_{n-1}\}`
    representing contributions at multiple time lags.

    The receiver maps each coefficient :math:`c_i` to a target buffer slot:

    .. math::

       \text{target\_slot} = (d_\text{steps} - d_\text{min}) + i

    where :math:`d_\text{min}` is the minimum delay in the network. This ensures that
    coefficients are applied to the correct future time steps.

    **3. Event Payload Structure**

    A delayed rate event contains:

    - ``rate``: Scalar rate value or coefficient array
    - ``weight``: Connection weight
    - ``delay_steps``: Integer delay
    - ``multiplicity``: Optional scaling factor for probabilistic connections

    The method :meth:`coeffarray_to_step_events` expands a coefficient array into
    individual per-step events, each with the appropriate delay offset.

    Implementation Notes
    --------------------
    **Delay Validation**

    This model enforces ``delay_steps >= 1`` to comply with NEST semantics. Attempting
    to set ``delay_steps = 0`` raises a ``ValueError``. For instantaneous (zero-delay)
    connections, use ``rate_connection_instantaneous`` instead.

    **Unit Handling**

    All parameters accept ``brainunit.Quantity`` objects or plain numeric values. If a
    ``Quantity`` is provided, its mantissa is extracted. Internally, values are stored
    as dimensionless floats or integers.

    **Compatibility with NEST**

    - NEST stores delays in time units (ms), while this implementation uses discrete
      steps to match BrainPy's event system.
    - The ``set_status`` method accepts both ``delay`` and ``delay_steps`` as aliases
      for backward compatibility.
    - Secondary event expansion follows NEST's buffer indexing logic exactly.

    Raises
    ------
    ValueError
        If ``weight`` or ``delay_steps`` is not scalar, or if ``delay_steps < 1``.
    ValueError
        If ``delay`` and ``delay_steps`` are both provided in ``set_status`` with
        conflicting values.
    ValueError
        If coefficient array is empty or delay is less than ``min_delay_steps`` in
        ``coeffarray_to_step_events``.

    See Also
    --------
    rate_connection_instantaneous : Zero-delay rate connection model (NEST equivalent)
    rate_neuron_ipn : Input rate neuron (delayed event receiver)
    rate_neuron_opn : Output rate neuron (delayed event receiver)

    References
    ----------
    .. [1] Hahne, J., et al. (2015). "A unified framework for spiking and rate-based
           neural networks." Frontiers in Neuroinformatics, 9, 22.
    .. [2] NEST Simulator documentation: Rate neuron models.
           https://nest-simulator.readthedocs.io/en/stable/models/rate_connection_delayed.html
    .. [3] NEST source: ``models/rate_connection_delayed.{h,cpp}``.
    .. [4] NEST delayed-rate receiver handling: ``models/rate_neuron_ipn_impl.h``
           and ``models/rate_neuron_opn_impl.h``.

    Examples
    --------
    **Basic Usage**

    Create a delayed connection with weight 2.0 and 3-step delay:

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> conn = bst.rate_connection_delayed(weight=2.0, delay_steps=3)
       >>> conn.get_status()
       {'weight': 2.0, 'delay_steps': 3, 'delay': 3, 'has_delay': True, 'supports_wfr': False}

    **Creating Rate Events**

    Transmit a rate signal with the connection's parameters:

    .. code-block:: python

       >>> event = conn.to_rate_event(rate=5.0, multiplicity=1.0)
       >>> event
       {'rate': 5.0, 'weight': 2.0, 'delay_steps': 3, 'multiplicity': 1.0}

    **Secondary Event Expansion**

    Map a coefficient array to per-step events (e.g., for implicit integration):

    .. code-block:: python

       >>> coeffs = [0.5, 1.0, 0.3]  # Three lag coefficients
       >>> events = conn.coeffarray_to_step_events(coeffs, min_delay_steps=1)
       >>> len(events)
       3
       >>> events[0]
       {'rate': 0.5, 'weight': 2.0, 'delay_steps': 2, 'multiplicity': 1.0}
       >>> events[1]
       {'rate': 1.0, 'weight': 2.0, 'delay_steps': 3, 'multiplicity': 1.0}
       >>> events[2]
       {'rate': 0.3, 'weight': 2.0, 'delay_steps': 4, 'multiplicity': 1.0}

    **Dynamic Parameter Updates**

    Update connection parameters at runtime:

    .. code-block:: python

       >>> conn.set_status(weight=1.5, delay_steps=5)
       >>> conn.get('weight')
       1.5
       >>> conn.get('delay_steps')
       5

    **Using with Rate Neurons**

    Typical usage in a rate-based network (conceptual example):

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import brainunit as u
       >>> # Create rate neurons (assuming rate_neuron_ipn exists)
       >>> pre = bst.rate_neuron_ipn(size=10, tau=10.0*u.ms)
       >>> post = bst.rate_neuron_ipn(size=5, tau=10.0*u.ms)
       >>> # Define delayed connections
       >>> conns = [bst.rate_connection_delayed(weight=w, delay_steps=d)
       ...          for w, d in zip([0.8, 1.2, 0.5], [2, 3, 1])]
       >>> # Transmit events during simulation
       >>> for conn in conns:
       ...     event = conn.to_rate_event(pre.rate, multiplicity=1.0)
       ...     # post.handle_delayed_event(event)  # Receiver-side logic
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    SUPPORTS_WFR = False

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        name: str | None = None,
    ):
        self.name = name
        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay_steps = self._validate_delay_steps(delay_steps)

    @property
    def properties(self) -> dict[str, Any]:
        r"""Return connection model properties.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``'has_delay'`` (bool): Always ``True`` for this model.
            - ``'supports_wfr'`` (bool): Always ``False`` (waveform relaxation not supported).
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
            - ``'delay_steps'`` (int): Delay in simulation steps.
            - ``'delay'`` (int): Alias for ``delay_steps`` (NEST compatibility).
            - ``'has_delay'`` (bool): Always ``True``.
            - ``'supports_wfr'`` (bool): Always ``False``.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_delayed(weight=1.5, delay_steps=2)
           >>> status = conn.get_status()
           >>> status['weight']
           1.5
           >>> status['delay']
           2
        """
        return {
            'weight': float(self.weight),
            'delay_steps': int(self.delay_steps),
            'delay': int(self.delay_steps),
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update connection parameters from a dictionary or keyword arguments.

        Follows NEST's ``SetStatus`` API convention. Accepts both ``delay`` and
        ``delay_steps`` as aliases (if both provided, they must match).

        Parameters
        ----------
        status : dict or None, optional
            Dictionary of parameters to update. Keys: ``'weight'``, ``'delay'``,
            ``'delay_steps'``.
        **kwargs
            Alternative parameter specification as keyword arguments. Merged with
            ``status`` (keyword args take precedence).

        Raises
        ------
        ValueError
            If ``delay`` and ``delay_steps`` are both provided with conflicting values.
        ValueError
            If any parameter fails validation (e.g., non-scalar, delay < 1).

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_delayed(weight=1.0, delay_steps=1)
           >>> conn.set_status({'weight': 2.5, 'delay_steps': 4})
           >>> conn.get('weight')
           2.5
           >>> conn.set_status(weight=3.0)  # Keyword argument style
           >>> conn.get('weight')
           3.0
        """
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        if 'weight' in updates:
            self.set_weight(updates['weight'])

        has_delay = 'delay' in updates
        has_delay_steps = 'delay_steps' in updates
        if has_delay and has_delay_steps:
            d = self._to_int_scalar(updates['delay'], name='delay')
            ds = self._to_int_scalar(updates['delay_steps'], name='delay_steps')
            if d != ds:
                raise ValueError('delay and delay_steps must be identical when both are provided.')
            self.set_delay_steps(ds)
        elif has_delay_steps:
            self.set_delay_steps(updates['delay_steps'])
        elif has_delay:
            self.set_delay(updates['delay'])

    def get(self, key: str = 'status'):
        r"""Retrieve a specific parameter or full status dictionary.

        Parameters
        ----------
        key : str, optional
            Parameter name to retrieve. Special value ``'status'`` returns full
            status dictionary. Supported keys: ``'status'``, ``'weight'``,
            ``'delay'``, ``'delay_steps'``, ``'has_delay'``, ``'supports_wfr'``.
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

           >>> conn = rate_connection_delayed(weight=2.0, delay_steps=3)
           >>> conn.get('weight')
           2.0
           >>> conn.get('delay_steps')
           3
           >>> conn.get('status')
           {'weight': 2.0, 'delay_steps': 3, ...}
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for rate_connection_delayed.get().')

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

           >>> conn = rate_connection_delayed()
           >>> conn.set_weight(2.5)
           >>> conn.get('weight')
           2.5
        """
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, delay: ArrayLike):
        r"""Update the connection delay (alias for ``set_delay_steps``).

        Parameters
        ----------
        delay : int or array-like
            New delay in simulation steps. Must be integer-valued scalar ``>= 1``.
            Accepts ``brainunit.Quantity`` (mantissa will be extracted).

        Raises
        ------
        ValueError
            If ``delay`` is not scalar, not integer-valued, or ``< 1``.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_delayed()
           >>> conn.set_delay(5)
           >>> conn.get('delay_steps')
           5
        """
        self.delay_steps = self._validate_delay_steps(delay, name='delay')

    def set_delay_steps(self, delay_steps: ArrayLike):
        r"""Update the connection delay in simulation steps.

        Parameters
        ----------
        delay_steps : int or array-like
            New delay in simulation steps. Must be integer-valued scalar ``>= 1``.
            Accepts ``brainunit.Quantity`` (mantissa will be extracted).

        Raises
        ------
        ValueError
            If ``delay_steps`` is not scalar, not integer-valued, or ``< 1``.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_delayed()
           >>> conn.set_delay_steps(3)
           >>> conn.get('delay_steps')
           3
        """
        self.delay_steps = self._validate_delay_steps(delay_steps, name='delay_steps')

    def prepare_secondary_event(self, coeffarray: ArrayLike) -> dict[str, Any]:
        r"""Create a delayed secondary-event payload.

        Secondary events are used in implicit integration schemes for rate neurons.
        The coefficient array represents contributions at multiple time lags.

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
            - ``'delay_steps'`` (int): Connection delay in steps.

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

           >>> conn = rate_connection_delayed(weight=2.0, delay_steps=3)
           >>> event = conn.prepare_secondary_event([0.5, 1.0, 0.3])
           >>> event['coeffarray']
           array([0.5, 1. , 0.3])
           >>> event['weight']
           2.0
        """
        return {
            'coeffarray': self._to_coeff_array(coeffarray),
            'weight': float(self.weight),
            'delay_steps': int(self.delay_steps),
        }

    def to_rate_event(
        self,
        rate: ArrayLike,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        r"""Create a delayed-rate event payload for step-based simulation APIs.

        Constructs an event dictionary that can be passed to rate neuron receivers
        to transmit a rate signal with the connection's weight and delay.

        Parameters
        ----------
        rate : float or array-like
            Rate value to transmit. Can be scalar (single rate) or array (multiple
            rates). Accepts ``brainunit.Quantity`` (mantissa will be extracted).
        multiplicity : float or array-like, optional
            Scaling factor for stochastic or probabilistic connections. Must be
            scalar. Default: ``1.0``.
        delay_steps : int or array-like or None, optional
            Override delay for this event. Must be integer-valued scalar ``>= 1``.
            If ``None``, uses the connection's ``self.delay_steps``. Default: ``None``.

        Returns
        -------
        dict
            Event payload with keys:

            - ``'rate'`` (float or ndarray): Rate value (scalar or array).
            - ``'weight'`` (float): Connection weight.
            - ``'delay_steps'`` (int): Delay in steps.
            - ``'multiplicity'`` (float): Scaling factor.

        Raises
        ------
        ValueError
            If ``multiplicity`` or ``delay_steps`` is not scalar, or if
            ``delay_steps < 1``.

        Examples
        --------
        .. code-block:: python

           >>> conn = rate_connection_delayed(weight=2.0, delay_steps=3)
           >>> event = conn.to_rate_event(rate=5.0)
           >>> event
           {'rate': 5.0, 'weight': 2.0, 'delay_steps': 3, 'multiplicity': 1.0}

        Override delay for a specific event:

        .. code-block:: python

           >>> event = conn.to_rate_event(rate=5.0, delay_steps=5)
           >>> event['delay_steps']
           5
        """
        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        return {
            'rate': self._to_rate_value(rate),
            'weight': float(self.weight),
            'delay_steps': int(d),
            'multiplicity': self._to_float_scalar(multiplicity, name='multiplicity'),
        }

    def coeffarray_to_step_events(
        self,
        coeffarray: ArrayLike,
        min_delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
    ) -> list[dict[str, Any]]:
        r"""Map lag-indexed coefficient array to per-step delayed-rate events.

        Expands a coefficient array representing multiple time lags into individual
        per-step rate events, following NEST's delayed-rate receiver buffer indexing.
        Each coefficient :math:`c_i` is mapped to an event with delay offset
        :math:`(d - d_{\text{min}}) + i`, where :math:`d` is the connection delay
        and :math:`d_{\text{min}}` is the minimum network delay.

        This method is used to distribute secondary event coefficients across future
        time steps for implicit integration schemes.

        Parameters
        ----------
        coeffarray : array-like
            1D array of coefficients, shape ``(n,)``. Each element represents a rate
            contribution at a successive time lag. Must be non-empty. Accepts
            ``brainunit.Quantity`` (mantissa will be extracted).
        min_delay_steps : int or array-like, optional
            Minimum delay in the network (in simulation steps). Used to compute the
            base delay offset. Must be integer-valued scalar ``>= 1``. Default: ``1``.
        multiplicity : float or array-like, optional
            Scaling factor for all events. Must be scalar. Default: ``1.0``.

        Returns
        -------
        list of dict
            List of event dictionaries, one per coefficient. Each event has keys:

            - ``'rate'`` (float): Coefficient value.
            - ``'weight'`` (float): Connection weight.
            - ``'delay_steps'`` (int): Computed delay = ``(self.delay_steps - min_delay_steps) + i``.
            - ``'multiplicity'`` (float): Scaling factor.

            Length of list equals ``len(coeffarray)``.

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty.
        ValueError
            If ``self.delay_steps < min_delay_steps`` (would produce negative delay).
        ValueError
            If ``min_delay_steps`` or ``multiplicity`` is not scalar, or if
            ``min_delay_steps < 1``.

        Notes
        -----
        **NEST Buffer Indexing**

        NEST rate neurons with delayed connections use a ring buffer indexed by:

        .. math::

           \text{buffer\_slot} = (d - d_{\text{min}}) + i

        where :math:`d` is the connection delay, :math:`d_{\text{min}}` is the
        minimum delay, and :math:`i` is the coefficient index. This ensures that
        coefficients are applied to the correct future time steps during implicit
        integration.

        **Example Calculation**

        Suppose ``self.delay_steps = 5``, ``min_delay_steps = 2``, and
        ``coeffarray = [0.5, 1.0, 0.3]``:

        - Coefficient 0 (``0.5``): ``delay_steps = (5 - 2) + 0 = 3``
        - Coefficient 1 (``1.0``): ``delay_steps = (5 - 2) + 1 = 4``
        - Coefficient 2 (``0.3``): ``delay_steps = (5 - 2) + 2 = 5``

        See Also
        --------
        prepare_secondary_event : Create secondary event with coefficient array.
        to_rate_event : Create single-rate event.

        Examples
        --------
        Basic usage with 3 coefficients:

        .. code-block:: python

           >>> conn = rate_connection_delayed(weight=2.0, delay_steps=5)
           >>> coeffs = [0.5, 1.0, 0.3]
           >>> events = conn.coeffarray_to_step_events(coeffs, min_delay_steps=2)
           >>> len(events)
           3
           >>> events[0]
           {'rate': 0.5, 'weight': 2.0, 'delay_steps': 3, 'multiplicity': 1.0}
           >>> events[1]
           {'rate': 1.0, 'weight': 2.0, 'delay_steps': 4, 'multiplicity': 1.0}
           >>> events[2]
           {'rate': 0.3, 'weight': 2.0, 'delay_steps': 5, 'multiplicity': 1.0}

        Error when connection delay is less than minimum delay:

        .. code-block:: python

           >>> conn = rate_connection_delayed(weight=1.0, delay_steps=1)
           >>> conn.coeffarray_to_step_events([0.5, 1.0], min_delay_steps=3)
           Traceback (most recent call last):
               ...
           ValueError: delay_steps must be >= min_delay_steps.

        Using with multiplicity scaling:

        .. code-block:: python

           >>> conn = rate_connection_delayed(weight=1.0, delay_steps=3)
           >>> events = conn.coeffarray_to_step_events([0.5, 1.0], min_delay_steps=1, multiplicity=0.8)
           >>> events[0]['multiplicity']
           0.8
        """
        coeff = self._to_coeff_array(coeffarray)
        min_delay = self._validate_delay_steps(min_delay_steps, name='min_delay_steps')
        base_delay = int(self.delay_steps - min_delay)
        if base_delay < 0:
            raise ValueError('delay_steps must be >= min_delay_steps.')
        mult = self._to_float_scalar(multiplicity, name='multiplicity')

        events = []
        for i, c in enumerate(coeff):
            events.append(
                {
                    'rate': float(c),
                    'weight': float(self.weight),
                    'delay_steps': int(base_delay + i),
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

    def _validate_delay_steps(self, delay_steps: ArrayLike, name: str = 'delay_steps') -> int:
        r"""Validate and convert delay parameter to integer >= 1.

        Parameters
        ----------
        delay_steps : array-like
            Delay value in simulation steps. Accepts ``brainunit.Quantity``
            (mantissa extracted).
        name : str, optional
            Parameter name for error messages. Default: ``'delay_steps'``.

        Returns
        -------
        int
            Validated integer delay value.

        Raises
        ------
        ValueError
            If delay is not scalar, not integer-valued, or ``< 1``.
        """
        d = self._to_int_scalar(delay_steps, name=name)
        if d < 1:
            raise ValueError(f'{name} must be >= 1.')
        return d
