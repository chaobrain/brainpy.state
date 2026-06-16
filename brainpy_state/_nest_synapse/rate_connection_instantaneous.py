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
    'rate_connection_instantaneous',
]


class rate_connection_instantaneous(NESTSynapse):
    r"""NEST-compatible ``rate_connection_instantaneous`` connection spec.

    Carries the NEST-facing parameters and status of an instantaneous (zero-delay)
    rate connection: a scalar ``weight`` plus the delay-rejection semantics that
    distinguish this model from :class:`rate_connection_delayed`. It mirrors NEST's
    ``GetStatus`` / ``SetStatus`` surface (``weight``, a read-only ``delay`` compatibility
    field, ``has_delay``, ``supports_wfr``) so that connection parameters round-trip
    identically to the C++ model.

    The instantaneous rate coupling itself is realized by the Simulator's continuous-rate
    (seam-(H)) emission path: a presynaptic rate neuron emits its graded ``rate`` each step,
    the connection deposits ``weight · rate`` into the postsynaptic neuron's delta input
    channel, and the post reads it via ``sum_delta_inputs``. The one-step pipeline lag of
    that path is exactly NEST's ``use_wfr=False`` instantaneous fixed-point seed — so this
    spec object only needs to carry the parameters, not route the signal.

    Unlike ``rate_connection_delayed``, this model enforces zero delay and rejects any
    attempt to configure a delay parameter.

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
        Class attribute, always ``True`` (NEST advertises waveform-relaxation support for
        instantaneous connections).

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
    An instantaneous connection transmits a rate signal :math:`r_\text{pre}` to the
    postsynaptic neuron without delay, applying only the connection weight :math:`w`:

    .. math::

       r_\text{post}(t) = w \cdot r_\text{pre}(t)

    On the JAX substrate this is delivered with a single-step pipeline lag — the presynaptic
    rate captured at step :math:`k` is deposited into the postsynaptic input at step
    :math:`k+1`. That lag is the discrete-time counterpart of NEST's instantaneous
    (``use_wfr=False``) coupling and seeds the same network fixed point
    :math:`r^* = (I - gW)^{-1}\mu` for a linear rate network.

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
    as dimensionless floats.

    **Compatibility with NEST**

    - NEST stores connection properties in synapse objects and enforces delay restrictions
      at runtime. This implementation replicates that behavior by raising errors when
      delay modification is attempted.
    - The ``supports_wfr`` flag is set to ``True``, matching NEST's advertisement that
      instantaneous connections participate in its waveform-relaxation solver.

    Raises
    ------
    ValueError
        If ``weight`` is not scalar.
    ValueError
        If any method attempts to set ``delay`` or ``delay_steps``.

    See Also
    --------
    rate_connection_delayed : Delayed rate connection model (NEST equivalent)
    rate_neuron_ipn : Input rate neuron (instantaneous rate receiver)
    rate_neuron_opn : Output rate neuron (instantaneous rate receiver)

    References
    ----------
    .. [1] Hahne, J., et al. (2015). "A unified framework for spiking and rate-based
           neural networks." Frontiers in Neuroinformatics, 9, 22.
    .. [2] NEST Simulator documentation: Rate neuron models.
           https://nest-simulator.readthedocs.io/en/stable/models/rate_connection_instantaneous.html
    .. [3] NEST source: ``models/rate_connection_instantaneous.{h,cpp}``.
    .. [4] NEST instantaneous-rate receiver handling: ``models/rate_neuron_ipn_impl.h``
           and ``models/rate_neuron_opn_impl.h``.

    Examples
    --------
    **Basic Usage**

    Create an instantaneous connection with weight 2.0 and inspect its status:

    .. code-block:: python

       >>> from brainpy import state as bst
       >>> conn = bst.rate_connection_instantaneous(weight=2.0)
       >>> conn.get_status()
       {'weight': 2.0, 'delay': 1, 'has_delay': False, 'supports_wfr': True}

    **Dynamic Weight Updates**

    Update the connection weight at runtime:

    .. code-block:: python

       >>> conn.set_weight(1.5)
       >>> conn.get('weight')
       1.5
       >>> conn.set_status(weight=3.0)
       >>> conn.get('weight')
       3.0

    **Delay Restriction Enforcement**

    Attempting to set a delay raises an error:

    .. code-block:: python

       >>> conn.set_delay(3)
       Traceback (most recent call last):
           ...
       ValueError: rate_connection_instantaneous has no delay. Please use rate_connection_delayed.

    **Using with Rate Neurons**

    Typical usage in a rate-based network — the Simulator wires the seam-(H) rate path
    from the connection's parameters:

    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy import state as bst
       >>> sim = bst.Simulator(dt=0.1 * u.ms)
       >>> pre = sim.create(bst.lin_rate_ipn, 10, params=dict(tau=10.0 * u.ms))
       >>> post = sim.create(bst.lin_rate_ipn, 5, params=dict(tau=10.0 * u.ms))
       >>> proj = sim.connect(pre, post, weight=0.8, comm='dense')  # doctest: +SKIP
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
        super().__init__(in_size=1, name=name)
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
