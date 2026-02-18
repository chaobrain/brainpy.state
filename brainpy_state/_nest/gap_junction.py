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
    'gap_junction',
]


class gap_junction(NESTSynapse):
    r"""NEST-compatible ``gap_junction`` connection model for electrical synaptic coupling.

    This class implements electrical gap junction connections that transmit
    ``GapJunctionEvent`` data and contribute currents proportional to membrane
    voltage differences between coupled neurons. It follows the connection-level
    semantics of NEST's ``models/gap_junction.{h,cpp}`` implementation, including
    waveform-relaxation (WFR) support for multi-timestep integration.

    Parameters
    ----------
    weight : float or array-like, optional
        Gap junction conductance weight (unitless coupling strength).
        Must be a scalar value. Default: ``1.0``.
    name : str or None, optional
        Optional model instance name for identification. Default: ``None``.

    Attributes
    ----------
    name : str or None
        Model instance name.
    weight : float
        Gap junction conductance weight.
    sumj_g_ij : float
        Sum of incoming gap junction weights (runtime state).
    interpolation_coefficients : ndarray
        Buffer of summed interpolation coefficients from incoming events,
        shape ``(min_delay_steps * (interpolation_order + 1),)``.
    REQUIRES_SYMMETRIC : bool
        Class constant, always ``True`` for gap junctions.
    SUPPORTS_WFR : bool
        Class constant, always ``True`` for WFR compatibility.
    SUPPORTED_WFR_INTERPOLATION_ORDERS : tuple
        Supported interpolation orders: ``(0, 1, 3)``.

    Parameter Mapping
    -----------------

    .. list-table::
       :header-rows: 1

       * - NEST Name
         - brainpy.state Name
         - Description
       * - weight
         - weight
         - Gap junction conductance (nS)
       * - delay
         - *unsupported*
         - Delay not allowed for gap junctions

    Mathematical Formulation
    ------------------------

    *1. Gap Junction Current*

    The gap junction current for a post-synaptic neuron :math:`i` coupled to
    pre-synaptic neurons :math:`j` is given by:

    .. math::

        I_{\text{gap},i} = -\sum_j g_{ij} V_i + P_{\text{interp}}(t)

    where:

    - :math:`g_{ij}` is the gap junction conductance (weight) from neuron
      :math:`j` to :math:`i`
    - :math:`V_i` is the membrane potential of the post-synaptic neuron
    - :math:`P_{\text{interp}}(t)` is the interpolation polynomial constructed
      from pre-synaptic voltage contributions

    *2. Waveform Relaxation (WFR) Event Accumulation*

    During WFR cycles, incoming ``GapJunctionEvent`` objects update two quantities:

    .. math::

        \Sigma g_{ij} &\leftarrow \Sigma g_{ij} + w_{ij} \\
        c_k &\leftarrow c_k + w_{ij} \cdot a_k

    where:

    - :math:`w_{ij}` is the connection weight
    - :math:`a_k` are source neuron interpolation coefficients
    - :math:`c_k` are target-side summed coefficients for polynomial
      reconstruction

    *3. Interpolation Polynomials*

    The interpolation term :math:`P_{\text{interp}}(t)` depends on the WFR order:

    - **Order 0** (constant interpolation):

      .. math::

          P_{\text{interp}}(t) = c_0

    - **Order 1** (linear interpolation):

      .. math::

          P_{\text{interp}}(t) = c_0 + c_1 \cdot t

    - **Order 3** (cubic Hermite interpolation):

      .. math::

          P_{\text{interp}}(t) = c_0 + c_1 \cdot t + c_2 \cdot t^2 + c_3 \cdot t^3

    where :math:`t \in [0, 1]` is the normalized time within the current WFR
    substep.

    **Implementation Details**

    *1. Update Ordering (matching NEST)*

    The WFR cycle follows this sequence:

    1. **Begin WFR window**: Clear ``sumj_g_ij`` and interpolation coefficient
       buffer.
    2. **Event accumulation**: For each arriving secondary event, add weight and
       weighted coefficients.
    3. **ODE substeps**: Evaluate :math:`I_{\text{gap}}` from current voltage,
       lag, interpolation order, and normalized substep time.
    4. **End WFR window**: Reset runtime buffers for the next window.

    *2. Symmetric Connectivity Requirement*

    Gap junctions require symmetric connections: if neuron :math:`i` is coupled to
    neuron :math:`j` with weight :math:`w_{ij}`, then neuron :math:`j` must also be
    coupled to neuron :math:`i` with weight :math:`w_{ji}`. Asymmetric configurations
    will produce incorrect dynamics.

    *3. Delay Constraint*

    Unlike chemical synapses, gap junctions are instantaneous (or near-instantaneous)
    electrical connections. Setting a delay raises a ``ValueError``.

    *4. Coefficient Buffer Sizing*

    For WFR with ``min_delay_steps`` lags and interpolation order :math:`n`, the
    coefficient buffer size is:

    .. math::

        \text{buffer\_size} = \text{min\_delay\_steps} \times (n + 1)

    Examples
    --------
    **Basic gap junction setup:**

    .. code-block:: python

        >>> import brainpy_state as bp
        >>> import brainunit as u
        >>>
        >>> # Create gap junction with 10 nS conductance
        >>> gap = bp.nest.gap_junction(weight=10.0)
        >>>
        >>> # Query connection properties
        >>> gap.get_status()
        {'weight': 10.0, 'delay': None, 'requires_symmetric': True,
         'supports_wfr': True, 'supported_wfr_interpolation_orders': (0, 1, 3)}

    **WFR cycle simulation:**

    .. code-block:: python

        >>> import numpy as np
        >>>
        >>> # Initialize WFR with 5 delay steps and linear interpolation
        >>> gap.begin_wfr_cycle(min_delay_steps=5, interpolation_order=1)
        >>>
        >>> # Simulate incoming event with pre-synaptic coefficients
        >>> pre_coeffs = np.array([1.0, 0.5, 1.2, 0.8, 1.5, 0.3, 1.1, 0.9, 1.3, 0.7])
        >>> gap.handle_gap_event(coeffarray=pre_coeffs)
        >>>
        >>> # Evaluate gap current at lag 2, normalized time t=0.5
        >>> V_post = -65.0  # mV
        >>> I_gap = gap.evaluate_gap_current(V_m=V_post, lag=2, t=0.5)
        >>> print(f"Gap junction current: {I_gap:.2f}")
        Gap junction current: ...

    **Error: attempting to set delay:**

    .. code-block:: python

        >>> gap.set_delay(1.5)
        ValueError: gap_junction connection has no delay

    See Also
    --------
    hh_psc_alpha_gap : Hodgkin-Huxley neuron with gap junction support
    hh_cond_beta_gap_traub : Traub HH neuron with gap junction support

    Notes
    -----
    - **Delay is intentionally unsupported**: Matches NEST's constraint that gap
      junctions have no synaptic delay.
    - **Connection-level semantics only**: This class represents gap junction
      connection properties. Neuron-specific ODE dynamics (voltage interpolation,
      coefficient broadcasting) remain in the neuron model.
    - **Interpolation coefficient ownership**: The neuron model owns and updates
      interpolation coefficients based on its voltage history. This class only
      accumulates weighted coefficients from incoming events.
    - **Thread safety**: Not thread-safe. Runtime state (``sumj_g_ij``,
      ``interpolation_coefficients``) assumes single-threaded access within each
      WFR cycle.

    References
    ----------
    .. [1] NEST Simulator gap junction implementation:
           ``models/gap_junction.h`` and ``models/gap_junction.cpp``.
    .. [2] NEST gap current evaluation in neuron models:
           ``models/hh_psc_alpha_gap.cpp`` and ``models/hh_cond_beta_gap_traub.cpp``.
    .. [3] Hahne, J., et al. (2015). "A unified framework for spiking and gap-junction
           interactions in distributed neuronal network simulations."
           *Frontiers in Neuroinformatics*, 9:22. doi:10.3389/fninf.2015.00022
    """

    __module__ = 'brainpy.state'

    REQUIRES_SYMMETRIC = True
    SUPPORTS_WFR = True
    SUPPORTED_WFR_INTERPOLATION_ORDERS = (0, 1, 3)

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        name: str | None = None,
    ):
        self.name = name
        self.weight = self._to_float_scalar(weight, name='weight')

        # Runtime state accumulated from incoming GapJunctionEvent payloads.
        self.sumj_g_ij: float = 0.0
        dftype = brainstate.environ.dftype()
        self.interpolation_coefficients = np.zeros((0,), dtype=dftype)
        self._interpolation_order = 0

    @property
    def properties(self) -> dict[str, Any]:
        r"""Get static connection model properties.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
              - ``'requires_symmetric'`` (bool): Always ``True``.
              - ``'supports_wfr'`` (bool): Always ``True``.

        Examples
        --------
        .. code-block:: python

            >>> gap = bp.nest.gap_junction(weight=5.0)
            >>> gap.properties
            {'requires_symmetric': True, 'supports_wfr': True}
        """
        return {
            'requires_symmetric': self.REQUIRES_SYMMETRIC,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        r"""Retrieve the current connection status and parameters.

        Returns
        -------
        dict[str, Any]
            Status dictionary containing:
              - ``'weight'`` (float): Current gap junction weight.
              - ``'delay'`` (None): Always ``None`` (delays not supported).
              - ``'requires_symmetric'`` (bool): Symmetric connectivity requirement.
              - ``'supports_wfr'`` (bool): WFR support flag.
              - ``'supported_wfr_interpolation_orders'`` (tuple): Tuple of supported
                interpolation orders ``(0, 1, 3)``.

        Examples
        --------
        .. code-block:: python

            >>> gap = bp.nest.gap_junction(weight=2.5)
            >>> status = gap.get_status()
            >>> status['weight']
            2.5
            >>> status['delay'] is None
            True

        See Also
        --------
        set_status : Update connection parameters
        get : Retrieve specific status keys
        """
        # Keep "delay" present (as None) for API parity with model status.
        return {
            'weight': float(self.weight),
            'delay': None,
            'requires_symmetric': self.REQUIRES_SYMMETRIC,
            'supports_wfr': self.SUPPORTS_WFR,
            'supported_wfr_interpolation_orders': self.SUPPORTED_WFR_INTERPOLATION_ORDERS,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update connection parameters from a status dictionary or keyword arguments.

        Parameters
        ----------
        status : dict[str, Any] or None, optional
            Dictionary of parameter updates. Keys must be valid parameter names.
        **kwargs
            Additional parameter updates passed as keyword arguments. These override
            values in ``status`` if both are provided.

        Raises
        ------
        ValueError
            If attempting to set ``'delay'`` (gap junctions have no delay).

        Examples
        --------
        .. code-block:: python

            >>> gap = bp.nest.gap_junction(weight=1.0)
            >>> gap.set_status({'weight': 5.0})
            >>> gap.get_status()['weight']
            5.0
            >>> gap.set_status(weight=10.0)
            >>> gap.weight
            10.0

        **Error case:**

        .. code-block:: python

            >>> gap.set_status({'delay': 1.0})
            ValueError: gap_junction connection has no delay

        See Also
        --------
        get_status : Retrieve current status
        set_weight : Update weight parameter directly
        """
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)
        if 'delay' in updates:
            raise ValueError('gap_junction connection has no delay')
        if 'weight' in updates:
            self.set_weight(updates['weight'])

    def get(self, key: str = 'status'):
        r"""Retrieve a specific status parameter or the full status dictionary.

        Parameters
        ----------
        key : str, optional
            Parameter name to retrieve. Use ``'status'`` for the full dictionary.
            Default: ``'status'``.

        Returns
        -------
        Any
            If ``key == 'status'``, returns the full status dictionary.
            Otherwise, returns the value associated with ``key``.

        Raises
        ------
        KeyError
            If ``key`` is not a valid status parameter name.

        Examples
        --------
        .. code-block:: python

            >>> gap = bp.nest.gap_junction(weight=3.5)
            >>> gap.get('weight')
            3.5
            >>> gap.get('requires_symmetric')
            True
            >>> gap.get()  # defaults to 'status'
            {'weight': 3.5, 'delay': None, ...}

        See Also
        --------
        get_status : Full status dictionary retrieval
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for gap_junction.get().')

    def set_weight(self, weight: ArrayLike):
        r"""Update the gap junction conductance weight.

        Parameters
        ----------
        weight : float or array-like
            New gap junction weight (must be scalar).

        Raises
        ------
        ValueError
            If ``weight`` is not a scalar value.

        Examples
        --------
        .. code-block:: python

            >>> gap = bp.nest.gap_junction(weight=1.0)
            >>> gap.set_weight(5.5)
            >>> gap.weight
            5.5

        See Also
        --------
        set_status : Update multiple parameters at once
        """
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, _):
        r"""Attempt to set delay (always raises an error).

        Gap junctions are instantaneous electrical connections and do not support
        synaptic delays. This method exists for API compatibility with NEST but
        always raises a ``ValueError``.

        Parameters
        ----------
        _ : Any
            Ignored (delay value).

        Raises
        ------
        ValueError
            Always raised with message ``'gap_junction connection has no delay'``.

        Examples
        --------
        .. code-block:: python

            >>> gap = bp.nest.gap_junction()
            >>> gap.set_delay(1.5)
            ValueError: gap_junction connection has no delay

        See Also
        --------
        set_status : General parameter update method (also rejects delay)
        """
        raise ValueError('gap_junction connection has no delay')

    def begin_wfr_cycle(self, min_delay_steps: ArrayLike, interpolation_order: int = 0):
        r"""Initialize a new waveform-relaxation (WFR) cycle.

        Resets runtime state (``sumj_g_ij`` and ``interpolation_coefficients``)
        and allocates the coefficient buffer based on the number of delay steps
        and the interpolation order.

        Parameters
        ----------
        min_delay_steps : int or array-like
            Minimum delay in discrete timesteps (must be > 0). Determines the
            number of lag slots in the interpolation coefficient buffer.
        interpolation_order : int, optional
            Interpolation polynomial order. Must be in ``{0, 1, 3}``:

            - ``0`` -- Constant interpolation (piecewise constant).
            - ``1`` -- Linear interpolation.
            - ``3`` -- Cubic Hermite interpolation.

            Default: ``0``.

        Raises
        ------
        ValueError
            - If ``min_delay_steps <= 0``.
            - If ``interpolation_order`` is not in ``{0, 1, 3}``.

        Notes
        -----
        The coefficient buffer size is ``min_delay_steps * (interpolation_order + 1)``.
        For example, with ``min_delay_steps=5`` and ``interpolation_order=1``, the
        buffer will have 10 elements (2 coefficients per lag).

        Examples
        --------
        .. code-block:: python

            >>> gap = bp.nest.gap_junction(weight=2.0)
            >>> gap.begin_wfr_cycle(min_delay_steps=3, interpolation_order=1)
            >>> gap.interpolation_coefficients.shape
            (6,)  # 3 lags * 2 coefficients per lag
            >>> gap.sumj_g_ij
            0.0

        See Also
        --------
        reset_runtime_state : Clear runtime state without reallocating buffer
        handle_gap_event : Accumulate incoming gap junction events
        """
        min_delay_steps = self._to_int_scalar(min_delay_steps, name='min_delay_steps')
        if min_delay_steps <= 0:
            raise ValueError('min_delay_steps must be > 0.')

        interpolation_order = self._validate_interpolation_order(interpolation_order)
        coeff_len = min_delay_steps * (interpolation_order + 1)

        self._interpolation_order = interpolation_order
        self.sumj_g_ij = 0.0
        dftype = brainstate.environ.dftype()
        self.interpolation_coefficients = np.zeros((coeff_len,), dtype=dftype)

    def reset_runtime_state(self):
        r"""Clear runtime state accumulated from gap junction events.

        Resets ``sumj_g_ij`` to ``0.0`` and zeros out the interpolation coefficient
        buffer without reallocating it. This method is typically called at the end
        of a WFR cycle to prepare for the next cycle.

        Examples
        --------
        .. code-block:: python

            >>> gap = bp.nest.gap_junction(weight=2.0)
            >>> gap.begin_wfr_cycle(min_delay_steps=3, interpolation_order=0)
            >>> gap.handle_gap_event(coeffarray=np.array([1.0, 2.0, 3.0]))
            >>> gap.sumj_g_ij
            2.0
            >>> gap.reset_runtime_state()
            >>> gap.sumj_g_ij
            0.0
            >>> np.all(gap.interpolation_coefficients == 0.0)
            True

        See Also
        --------
        begin_wfr_cycle : Initialize WFR cycle (resets and reallocates buffer)
        handle_gap_event : Accumulate gap junction events
        """
        self.sumj_g_ij = 0.0
        if self.interpolation_coefficients.size > 0:
            self.interpolation_coefficients.fill(0.0)

    def prepare_secondary_event(self, coeffarray: ArrayLike) -> dict[str, Any]:
        r"""Prepare a secondary gap junction event payload for transmission.

        Packages the connection weight and pre-synaptic interpolation coefficients
        into a dictionary suitable for transmission as a ``GapJunctionEvent`` in
        a WFR simulation.

        Parameters
        ----------
        coeffarray : array-like
            Pre-synaptic neuron's interpolation coefficients. Must be a non-empty
            1D array. For interpolation order :math:`n` and ``min_delay_steps`` lags,
            the expected size is ``min_delay_steps * (n + 1)``.

        Returns
        -------
        dict[str, Any]
            Event payload containing:
              - ``'weight'`` (float): Connection weight.
              - ``'coeffarray'`` (ndarray): 1D float64 array of interpolation
                coefficients.

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty or cannot be converted to a 1D array.

        Examples
        --------
        .. code-block:: python

            >>> import numpy as np
            >>> gap = bp.nest.gap_junction(weight=3.0)
            >>> pre_coeffs = np.array([0.5, 1.0, 1.5])
            >>> event = gap.prepare_secondary_event(pre_coeffs)
            >>> event['weight']
            3.0
            >>> event['coeffarray']
            array([0.5, 1. , 1.5])

        See Also
        --------
        handle_gap_event : Process incoming gap junction events
        """
        coeff_np = self._to_coeff_array(coeffarray)
        return {
            'weight': float(self.weight),
            'coeffarray': coeff_np,
        }

    def handle_gap_event(self, coeffarray: ArrayLike, weight: ArrayLike | None = None):
        r"""Process an incoming gap junction event and accumulate contributions.

        Updates runtime state by adding the event weight to ``sumj_g_ij`` and
        accumulating weighted interpolation coefficients. This method matches
        NEST's ``handle(GapJunctionEvent&)`` semantics.

        Parameters
        ----------
        coeffarray : array-like
            Pre-synaptic neuron's interpolation coefficients. Must be a 1D array
            with size matching the current coefficient buffer size.
        weight : float, array-like, or None, optional
            Event-specific weight override. If ``None`` (default), uses the
            connection's stored ``self.weight``.

        Raises
        ------
        ValueError
            - If ``coeffarray`` is empty.
            - If ``coeffarray`` size does not match the current buffer size.
            - If ``weight`` is provided but not a scalar.

        Notes
        -----
        **Accumulation rule**:

        .. math::

            \Sigma g_{ij} &\leftarrow \Sigma g_{ij} + w \\
            c_k &\leftarrow c_k + w \cdot a_k

        where :math:`w` is the event weight and :math:`a_k` are the incoming
        coefficients.

        **Buffer initialization**: If the coefficient buffer is empty when the
        first event arrives, it is initialized to match the incoming array size.

        Examples
        --------
        .. code-block:: python

            >>> import numpy as np
            >>> gap = bp.nest.gap_junction(weight=2.0)
            >>> gap.begin_wfr_cycle(min_delay_steps=3, interpolation_order=0)
            >>> pre_coeffs = np.array([1.0, 1.5, 2.0])
            >>> gap.handle_gap_event(coeffarray=pre_coeffs)
            >>> gap.sumj_g_ij
            2.0
            >>> gap.interpolation_coefficients
            array([2. , 3. , 4. ])  # weight * coeffarray

        **Multiple events accumulate:**

        .. code-block:: python

            >>> gap.handle_gap_event(coeffarray=np.array([0.5, 0.5, 0.5]))
            >>> gap.sumj_g_ij
            4.0  # 2.0 + 2.0
            >>> gap.interpolation_coefficients
            array([3. , 4. , 5. ])  # previous + 2.0 * [0.5, 0.5, 0.5]

        See Also
        --------
        prepare_secondary_event : Create event payload
        evaluate_gap_current : Compute gap junction current from accumulated events
        """
        coeff_np = self._to_coeff_array(coeffarray)

        if self.interpolation_coefficients.size == 0:
            dftype = brainstate.environ.dftype()
            self.interpolation_coefficients = np.zeros_like(coeff_np, dtype=dftype)
        if coeff_np.size != self.interpolation_coefficients.size:
            raise ValueError(
                f'Coefficient size mismatch: got {coeff_np.size}, expected {self.interpolation_coefficients.size}.'
            )

        w = self.weight if weight is None else self._to_float_scalar(weight, name='weight')
        self.sumj_g_ij += w

        # Matches NEST handle(GapJunctionEvent&): interpolation_coefficients += weight * coeffarray.
        self.interpolation_coefficients += w * coeff_np

    def evaluate_gap_current(
        self,
        V_m: ArrayLike,
        lag: int,
        t: ArrayLike = 0.0,
        interpolation_order: int | None = None,
    ):
        r"""Evaluate the gap junction current at a specific WFR lag and substep time.

        Computes the gap junction current contribution using the accumulated weight
        sum and interpolation coefficients from incoming events. The current is
        computed as:

        .. math::

            I_{\text{gap}} = -\Sigma g_{ij} \cdot V_m + P_{\text{interp}}(t)

        where :math:`P_{\text{interp}}(t)` is the interpolation polynomial evaluated
        at normalized time :math:`t \in [0, 1]` within the current substep.

        Parameters
        ----------
        V_m : float or array-like
            Post-synaptic membrane potential (scalar or array). Units should match
            the voltage scale used in the neuron model (typically mV).
        lag : int
            WFR lag index (delay step offset). Must be in range ``[0, n_lags)``,
            where ``n_lags = buffer_size / (interpolation_order + 1)``.
        t : float or array-like, optional
            Normalized substep time in :math:`[0, 1]`. Represents the position
            within the current integration step. Default: ``0.0``.
        interpolation_order : int or None, optional
            Interpolation order override. If ``None``, uses the order set in
            ``begin_wfr_cycle()``. Must be in ``{0, 1, 3}``.

        Returns
        -------
        float or ndarray
            Gap junction current :math:`I_{\text{gap}}`. Shape matches ``V_m``.
            Units depend on weight scaling (typically nA for conductance-based models).

        Raises
        ------
        ValueError
            - If ``interpolation_coefficients`` buffer is empty (call
              ``begin_wfr_cycle()`` first).
            - If buffer size is incompatible with ``interpolation_order``.
            - If ``lag`` is out of bounds.
            - If ``interpolation_order`` is not in ``{0, 1, 3}``.

        Notes
        -----
        **Interpolation formulas**:

        - **Order 0** (constant):

          .. math::

              P_{\text{interp}}(t) = c_0

        - **Order 1** (linear):

          .. math::

              P_{\text{interp}}(t) = c_0 + c_1 \cdot t

        - **Order 3** (cubic Hermite):

          .. math::

              P_{\text{interp}}(t) = c_0 + c_1 \cdot t + c_2 \cdot t^2 + c_3 \cdot t^3

        **Coefficient indexing**: Coefficients are stored in a flat array with
        layout ``[lag0_c0, lag0_c1, ..., lag1_c0, lag1_c1, ...]``. For lag :math:`L`
        and order :math:`n`, coefficients start at index :math:`L \times (n + 1)`.

        Examples
        --------
        **Constant interpolation (order 0):**

        .. code-block:: python

            >>> import numpy as np
            >>> gap = bp.nest.gap_junction(weight=2.0)
            >>> gap.begin_wfr_cycle(min_delay_steps=3, interpolation_order=0)
            >>> gap.handle_gap_event(coeffarray=np.array([10.0, 20.0, 30.0]))
            >>> V_post = -65.0  # mV
            >>> I_gap = gap.evaluate_gap_current(V_m=V_post, lag=1)
            >>> I_gap
            170.0  # -2.0 * (-65.0) + 40.0

        **Linear interpolation (order 1):**

        .. code-block:: python

            >>> gap2 = bp.nest.gap_junction(weight=1.5)
            >>> gap2.begin_wfr_cycle(min_delay_steps=2, interpolation_order=1)
            >>> gap2.handle_gap_event(coeffarray=np.array([5.0, 10.0, 15.0, 20.0]))
            >>> I_gap = gap2.evaluate_gap_current(V_m=-70.0, lag=0, t=0.5)
            >>> # I_gap = -1.5 * (-70.0) + (7.5 + 15.0 * 0.5)
            >>> I_gap
            120.0

        **Cubic interpolation (order 3):**

        .. code-block:: python

            >>> gap3 = bp.nest.gap_junction(weight=1.0)
            >>> gap3.begin_wfr_cycle(min_delay_steps=1, interpolation_order=3)
            >>> gap3.handle_gap_event(coeffarray=np.array([1.0, 2.0, 3.0, 4.0]))
            >>> I_gap = gap3.evaluate_gap_current(V_m=-60.0, lag=0, t=0.3)
            >>> # I_gap = -1.0 * (-60.0) + (1.0 + 2.0*0.3 + 3.0*0.09 + 4.0*0.027)
            >>> I_gap
            62.438

        See Also
        --------
        begin_wfr_cycle : Initialize WFR cycle with interpolation order
        handle_gap_event : Accumulate gap junction events
        reset_runtime_state : Clear accumulated state
        """
        if self.interpolation_coefficients.size == 0:
            raise ValueError('No interpolation coefficients available. Call begin_wfr_cycle() first.')

        order = self._interpolation_order if interpolation_order is None else interpolation_order
        order = self._validate_interpolation_order(order)

        n_per_lag = order + 1
        if self.interpolation_coefficients.size % n_per_lag != 0:
            raise ValueError('Interpolation coefficient buffer size is incompatible with interpolation order.')

        lag = self._to_int_scalar(lag, name='lag')
        n_lags = self.interpolation_coefficients.size // n_per_lag
        if lag < 0 or lag >= n_lags:
            raise ValueError(f'lag {lag} is out of bounds for {n_lags} lag slots.')

        dftype = brainstate.environ.dftype()
        v = np.asarray(u.math.asarray(V_m), dtype=dftype)
        t = np.asarray(u.math.asarray(t), dtype=dftype)
        base = -self.sumj_g_ij * v

        if order == 0:
            return base + self.interpolation_coefficients[lag]
        if order == 1:
            i0 = lag * 2
            return base + self.interpolation_coefficients[i0] + self.interpolation_coefficients[i0 + 1] * t

        i0 = lag * 4
        t2 = t * t
        return (
            base
            + self.interpolation_coefficients[i0]
            + self.interpolation_coefficients[i0 + 1] * t
            + self.interpolation_coefficients[i0 + 2] * t2
            + self.interpolation_coefficients[i0 + 3] * t2 * t
        )

    @classmethod
    def _validate_interpolation_order(cls, order: int) -> int:
        r"""Validate that the interpolation order is supported.

        Parameters
        ----------
        order : int
            Interpolation order to validate.

        Returns
        -------
        int
            Validated interpolation order.

        Raises
        ------
        ValueError
            If ``order`` is not in ``{0, 1, 3}``.
        """
        order = cls._to_int_scalar(order, name='interpolation_order')
        if order not in cls.SUPPORTED_WFR_INTERPOLATION_ORDERS:
            raise ValueError('Interpolation order must be 0, 1, or 3.')
        return order

    @staticmethod
    def _to_coeff_array(value: ArrayLike) -> np.ndarray:
        r"""Convert input to a 1D float64 coefficient array.

        Parameters
        ----------
        value : array-like
            Input value (may be a brainunit Quantity, ndarray, or scalar).

        Returns
        -------
        ndarray
            1D float64 array with at least one element.

        Raises
        ------
        ValueError
            If the resulting array is empty.
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size == 0:
            raise ValueError('Coefficient array must not be empty.')
        return arr

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        r"""Convert input to a scalar float value.

        Parameters
        ----------
        value : array-like
            Input value (may be a brainunit Quantity, ndarray, or scalar).
        name : str
            Parameter name for error messages.

        Returns
        -------
        float
            Scalar float value.

        Raises
        ------
        ValueError
            If the input is not a scalar (size != 1).
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
        r"""Convert input to a scalar integer value.

        Parameters
        ----------
        value : array-like
            Input value (may be a brainunit Quantity, ndarray, or scalar).
        name : str
            Parameter name for error messages.

        Returns
        -------
        int
            Scalar integer value (rounded if necessary).

        Raises
        ------
        ValueError
            If the input is not a scalar (size != 1).
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return int(arr[0])
