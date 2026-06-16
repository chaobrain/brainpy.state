from typing import Any

import brainstate
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

from brainpy_state._nest_base._base import NESTSynapse

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
    'rate_connection_delayed',
]


class rate_connection_delayed(NESTSynapse):
    r"""NEST-compatible ``rate_connection_delayed`` connection spec.

    Carries the NEST-facing parameters and status of a delayed rate connection: a scalar
    ``weight`` and an integer ``delay_steps`` (``>= 1``). It mirrors NEST's ``GetStatus`` /
    ``SetStatus`` surface (``weight``, ``delay`` / ``delay_steps`` aliases, ``has_delay``,
    ``supports_wfr``) so connection parameters round-trip identically to the C++ model.

    The delayed rate coupling itself is realized by the Simulator's continuous-rate
    (seam-(H)) emission path: a presynaptic rate neuron emits its graded ``rate`` each step,
    and the connection routes ``weight · rate`` through an input delay line of
    ``delay_steps`` steps (``delay_steps · dt`` of latency) into the postsynaptic neuron's
    delta input channel, which the post reads via ``sum_delta_inputs``. This spec object
    carries the parameters; the Simulator builds the delay line from them.

    Unlike ``rate_connection_instantaneous``, this model enforces a minimum delay of
    one simulation step.

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
    A delayed connection transmits a rate signal :math:`r_\text{pre}` to the postsynaptic
    neuron with delay :math:`d` and weight :math:`w`:

    .. math::

       r_\text{post}(t) = w \cdot r_\text{pre}(t - d)

    In discrete-time simulation with step size :math:`\Delta t`, the delay is quantized
    to an integer number of steps :math:`d_\text{steps} \geq 1`, and on the JAX substrate
    is realized by an input delay line of length :math:`d_\text{steps}`:

    .. math::

       d_\text{steps} = \left\lceil \frac{d}{\Delta t} \right\rceil, \quad d_\text{steps} \geq 1

    The presynaptic rate captured at step :math:`k` is therefore deposited into the
    postsynaptic input at step :math:`k + d_\text{steps}`, matching NEST's delayed rate
    delivery.

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
      (if both are provided, they must be identical).

    Raises
    ------
    ValueError
        If ``weight`` or ``delay_steps`` is not scalar, or if ``delay_steps < 1``.
    ValueError
        If ``delay`` and ``delay_steps`` are both provided in ``set_status`` with
        conflicting values.

    See Also
    --------
    rate_connection_instantaneous : Zero-delay rate connection model (NEST equivalent)
    rate_neuron_ipn : Input rate neuron (delayed rate receiver)
    rate_neuron_opn : Output rate neuron (delayed rate receiver)

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

       >>> from brainpy import state as bst
       >>> conn = bst.rate_connection_delayed(weight=2.0, delay_steps=3)
       >>> conn.get_status()
       {'weight': 2.0, 'delay_steps': 3, 'delay': 3, 'has_delay': True, 'supports_wfr': False}

    **Dynamic Parameter Updates**

    Update connection parameters at runtime:

    .. code-block:: python

       >>> conn.set_status(weight=1.5, delay_steps=5)
       >>> conn.get('weight')
       1.5
       >>> conn.get('delay_steps')
       5

    **Delay Validation**

    Zero or sub-step delays are rejected:

    .. code-block:: python

       >>> conn.set_delay_steps(0)
       Traceback (most recent call last):
           ...
       ValueError: delay_steps must be >= 1.

    **Using with Rate Neurons**

    Typical usage in a rate-based network — the Simulator builds the delay line from the
    connection's ``delay_steps``:

    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy import state as bst
       >>> sim = bst.Simulator(dt=0.1 * u.ms)
       >>> pre = sim.create(bst.lin_rate_ipn, 10, params=dict(tau=10.0 * u.ms))
       >>> post = sim.create(bst.lin_rate_ipn, 5, params=dict(tau=10.0 * u.ms))
       >>> proj = sim.connect(pre, post, weight=0.5, delay=0.3 * u.ms, comm='dense')  # doctest: +SKIP
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
        super().__init__(in_size=1, name=name)
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
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
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
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
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
